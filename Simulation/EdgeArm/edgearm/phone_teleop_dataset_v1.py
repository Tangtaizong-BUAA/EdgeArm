"""Atomic synthetic wrist RGB-D logging for phone-driven MuJoCo episodes.

This schema is deliberately separate from the physical RGB-D recorder.  Every
file is permanently labelled as simulated, contains zero physical samples, and
cannot be mistaken for calibrated real-camera evidence.  It is suitable for
synthetic ACT/VLA demonstrations and for validating the complete causal order:
pre-action wrist observation -> phone intent -> submitted action -> next state.
"""

from __future__ import annotations

import json
import hashlib
import time
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import h5py
import numpy as np

from .phone_teleop_runtime_v1 import (
    PHONE_CARTESIAN_MAPPING_VERSION,
    PHONE_TELEOP_RUNTIME_VERSION,
    PhoneControlCycle,
)
from .phone_task_language_v1 import (
    PhoneTaskMetadata,
    read_phone_task_metadata,
    write_phone_task_metadata,
)
from .generalization_task_language_v1 import GeneralizationTaskMetadataV1


PHONE_SIM_DATASET_SCHEMA_VERSION = "edgearm-phone-sim-wrist-rgbd-v1"
SIM_TELEOP_CONTROL_SOURCES = ("phone", "keyboard")


def _control_source(value: str) -> str:
    source = str(value).strip().lower()
    if source not in SIM_TELEOP_CONTROL_SOURCES:
        raise ValueError(
            f"control_source must be one of {SIM_TELEOP_CONTROL_SOURCES}"
        )
    return source


def _positive_int(value: Any, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _json_ready(value: Any) -> Any:
    if value is None or isinstance(value, (bool, str, int)):
        return value
    if isinstance(value, float):
        return value if np.isfinite(value) else repr(value)
    if isinstance(value, (np.integer, np.bool_)):
        return value.item()
    if isinstance(value, np.floating):
        item = float(value)
        return item if np.isfinite(item) else repr(item)
    if isinstance(value, np.ndarray):
        return _json_ready(value.tolist())
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return repr(value)


def _json_text(value: Any) -> str:
    return json.dumps(
        _json_ready(value),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


@dataclass(frozen=True)
class SimWristObservation:
    wrist_rgb: np.ndarray
    depth_wrist_mm: np.ndarray
    segmentation_wrist: np.ndarray
    capture_monotonic_ns: int

    def __post_init__(self) -> None:
        rgb = np.asarray(self.wrist_rgb)
        depth = np.asarray(self.depth_wrist_mm)
        segmentation = np.asarray(self.segmentation_wrist)
        if rgb.ndim != 3 or rgb.shape[-1] != 3 or rgb.dtype != np.uint8:
            raise ValueError("wrist_rgb must be uint8 [H,W,3]")
        if depth.shape != rgb.shape[:2] or depth.dtype != np.uint16:
            raise ValueError("depth_wrist_mm must be uint16 on the RGB pixel grid")
        if segmentation.shape != rgb.shape[:2] or segmentation.dtype != np.uint8:
            raise ValueError("segmentation_wrist must be uint8 on the RGB pixel grid")
        for name, value in (
            ("wrist_rgb", rgb),
            ("depth_wrist_mm", depth),
            ("segmentation_wrist", segmentation),
        ):
            frozen = value.copy()
            frozen.setflags(write=False)
            object.__setattr__(self, name, frozen)
        object.__setattr__(
            self,
            "capture_monotonic_ns",
            _positive_int(self.capture_monotonic_ns, "capture_monotonic_ns"),
        )


class HDF5PhoneSimWriter:
    """Append-only synthetic teleop logger with phone-compatible columns.

    The historical class/schema names are retained for existing datasets.  The
    immutable ``operator_input_source`` attribute distinguishes phone and
    keyboard demonstrations so keyboard data can never claim phone transport.
    """

    def __init__(
        self,
        path: Path,
        *,
        image_shape: tuple[int, int, int],
        episode_seed: int,
        environment_profile: str,
        camera_metadata: Mapping[str, Any],
        task_metadata: PhoneTaskMetadata | GeneralizationTaskMetadataV1 | None = None,
        scene_contract: Mapping[str, Any] | None = None,
        control_source: str = "phone",
        control_runtime_version: str | None = None,
        overwrite: bool = False,
    ) -> None:
        if (
            len(image_shape) != 3
            or image_shape[-1] != 3
            or any(type(value) is not int or value <= 0 for value in image_shape)
        ):
            raise ValueError("image_shape must be positive [H,W,3]")
        if type(episode_seed) is not int:
            raise ValueError("episode_seed must be an integer")
        if not environment_profile:
            raise ValueError("environment_profile must be non-empty")
        self.path = Path(path)
        self.control_source = _control_source(control_source)
        self.control_runtime_version = str(
            control_runtime_version
            or (
                PHONE_TELEOP_RUNTIME_VERSION
                if self.control_source == "phone"
                else "unspecified-keyboard-runtime"
            )
        ).strip()
        if not self.control_runtime_version:
            raise ValueError("control_runtime_version must be non-empty")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        mode = "w" if overwrite else "a"
        self._file = h5py.File(self.path, mode)
        self._shape = tuple(image_shape)
        self._initialize_or_validate(
            episode_seed=episode_seed,
            environment_profile=environment_profile,
            camera_metadata=camera_metadata,
            task_metadata=task_metadata,
            scene_contract=scene_contract,
            control_source=self.control_source,
            control_runtime_version=self.control_runtime_version,
        )
        self._recover_tail()

    @property
    def records_written(self) -> int:
        return int(self._file.attrs["committed_rows"])

    def _initialize_or_validate(
        self,
        *,
        episode_seed: int,
        environment_profile: str,
        camera_metadata: Mapping[str, Any],
        task_metadata: PhoneTaskMetadata | GeneralizationTaskMetadataV1 | None,
        scene_contract: Mapping[str, Any] | None,
        control_source: str,
        control_runtime_version: str,
    ) -> None:
        scene_contract_json = (
            _json_text(dict(scene_contract)) if scene_contract is not None else ""
        )
        if isinstance(task_metadata, GeneralizationTaskMetadataV1):
            if not scene_contract_json:
                raise ValueError("generalization task requires a scene contract")
            if task_metadata.scene_contract_sha256 != hashlib.sha256(
                scene_contract_json.encode("utf-8")
            ).hexdigest():
                raise ValueError("generalization task scene hash differs from the dataset scene")
        if "steps" in self._file:
            if self._file.attrs.get("schema_version", "") != PHONE_SIM_DATASET_SCHEMA_VERSION:
                raise ValueError("existing file has a different phone dataset schema")
            if bool(self._file.attrs.get("finalized", False)):
                raise ValueError("cannot append to a finalized phone episode")
            stored_shape = tuple(json.loads(str(self._file.attrs["image_shape_json"])))
            if stored_shape != self._shape:
                raise ValueError("existing phone dataset image shape differs")
            if int(self._file.attrs.get("episode_seed", -1)) != episode_seed:
                raise ValueError("existing phone dataset episode seed differs")
            if self._file.attrs.get("environment_profile", "") != environment_profile:
                raise ValueError("existing phone dataset environment profile differs")
            if self._file.attrs.get("camera_metadata_json", "") != _json_text(camera_metadata):
                raise ValueError("existing phone dataset camera metadata differs")
            stored_source = str(
                self._file.attrs.get("operator_input_source", "phone")
            )
            if stored_source != control_source:
                raise ValueError("existing dataset operator input source differs")
            stored_runtime = str(
                self._file.attrs.get(
                    "operator_input_runtime_version",
                    PHONE_TELEOP_RUNTIME_VERSION,
                )
            )
            if stored_runtime != control_runtime_version:
                raise ValueError("existing dataset operator runtime version differs")
            stored_task = read_phone_task_metadata(
                self._file,
                required=task_metadata is not None,
            )
            if task_metadata is not None and stored_task != task_metadata:
                raise ValueError("existing phone dataset task metadata differs")
            expected_scene_json = scene_contract_json
            stored_scene_json = str(self._file.attrs.get("scene_contract_json", ""))
            if stored_scene_json != expected_scene_json:
                raise ValueError("existing phone dataset scene contract differs")
            if stored_scene_json:
                stored_scene_hash = str(
                    self._file.attrs.get("scene_contract_sha256", "")
                )
                if stored_scene_hash != hashlib.sha256(
                    stored_scene_json.encode("utf-8")
                ).hexdigest():
                    raise ValueError("existing phone dataset scene contract hash mismatch")
            return
        now_ns = time.time_ns()
        self._file.attrs.update(
            {
                "schema_version": PHONE_SIM_DATASET_SCHEMA_VERSION,
                "phone_runtime_version": PHONE_TELEOP_RUNTIME_VERSION,
                "phone_cartesian_mapping_version": (
                    PHONE_CARTESIAN_MAPPING_VERSION
                    if control_source == "phone"
                    else "not_applicable"
                ),
                "operator_input_source": control_source,
                "operator_input_runtime_version": control_runtime_version,
                "created_unix_ns": now_ns,
                "episode_seed": episode_seed,
                "environment_profile": environment_profile,
                "backend": "mujoco",
                "claim_level": f"synthetic_{control_source}_demonstration",
                "official_lerobot_phone_transport": control_source == "phone",
                "simulated_wrist_rgbd": True,
                "physical_capture_claimed": False,
                "physical_hardware_connected": False,
                "physical_samples": 0,
                "physical_trials": 0,
                "wrist_camera_physically_calibrated": False,
                "schema_supports_synthetic_act_training": True,
                "schema_supports_synthetic_vla_training": True,
                "eligible_for_synthetic_act_training": False,
                "eligible_for_synthetic_vla_training": False,
                "eligible_for_physical_act_training": False,
                "causal_order": (
                    "pre_action_rgbd+q+operator_intent -> action -> q_next"
                ),
                "image_shape_json": _json_text(list(self._shape)),
                "camera_metadata_json": _json_text(camera_metadata),
                "committed_rows": 0,
                "finalized": False,
            }
        )
        if task_metadata is not None:
            write_phone_task_metadata(self._file, task_metadata)
        else:
            self._file.attrs["language_condition_present"] = False
        if scene_contract is not None:
            self._file.attrs["scene_contract_present"] = True
            self._file.attrs["scene_contract_json"] = scene_contract_json
            self._file.attrs["scene_contract_sha256"] = hashlib.sha256(
                scene_contract_json.encode("utf-8")
            ).hexdigest()
        else:
            self._file.attrs["scene_contract_present"] = False
        height, width, _channels = self._shape
        steps = self._file.create_group("steps")
        string_dtype = h5py.string_dtype(encoding="utf-8")
        specs: dict[str, tuple[tuple[int, ...], Any]] = {
            "wrist_rgb": ((height, width, 3), np.uint8),
            "depth_wrist_mm": ((height, width), np.uint16),
            "segmentation_wrist": ((height, width), np.uint8),
            "capture_monotonic_ns": ((), np.int64),
            "phone_present": ((), np.bool_),
            "phone_sequence": ((), np.int64),
            "phone_receive_monotonic_ns": ((), np.int64),
            "phone_position_m": ((3,), np.float64),
            "phone_orientation_xyzw": ((4,), np.float64),
            "phone_enabled": ((), np.bool_),
            "safety_active": ((), np.bool_),
            "safety_sample_age_ns": ((), np.int64),
            "safety_new_sample": ((), np.bool_),
            "cartesian_target_present": ((), np.bool_),
            "target_world_from_ee": ((4, 4), np.float64),
            "target_gripper_rad": ((), np.float64),
            "ik_present": ((), np.bool_),
            "ik_target_q_rad": ((6,), np.float64),
            "ik_position_error_m": ((), np.float64),
            "ik_orientation_error_rad": ((), np.float64),
            "ik_converged": ((), np.bool_),
            "ik_application_scale": ((), np.float64),
            "q_before_rad": ((6,), np.float64),
            "dq_before_rad_s": ((6,), np.float64),
            "target_q_rad": ((6,), np.float64),
            "queued_target_q_rad": ((6,), np.float64),
            "submitted_normalized_action": ((6,), np.float64),
            "q_after_rad": ((6,), np.float64),
            "dq_after_rad_s": ((6,), np.float64),
            "reward": ((), np.float64),
            "terminated": ((), np.bool_),
            "truncated": ((), np.bool_),
            "command_queued_monotonic_ns": ((), np.int64),
            "completed_monotonic_ns": ((), np.int64),
            "row_committed": ((), np.bool_),
        }
        for name, (tail_shape, dtype) in specs.items():
            # One image per chunk prevents a single append from repeatedly
            # reading and recompressing h5py's former auto-chunk spanning 64
            # time rows. LZF is lossless and optimized for live acquisition.
            compression = "lzf" if len(tail_shape) >= 2 else None
            chunks = (1, *tail_shape) if compression else None
            steps.create_dataset(
                name,
                shape=(0, *tail_shape),
                maxshape=(None, *tail_shape),
                dtype=dtype,
                compression=compression,
                chunks=chunks,
                shuffle=bool(compression),
            )
        for name in ("phone_audit_json", "safety_stop_reason", "ik_safety_reason"):
            steps.create_dataset(name, shape=(0,), maxshape=(None,), dtype=string_dtype)
        steps.create_dataset(
            "step_info_json_zlib",
            shape=(0,),
            maxshape=(None,),
            dtype=h5py.vlen_dtype(np.dtype("uint8")),
        )
        self._file.attrs["step_info_codec"] = "zlib-json-v1"
        self._file.flush()

    def _recover_tail(self) -> None:
        steps = self._file["steps"]
        lengths = {dataset.shape[0] for dataset in steps.values()}
        if len(lengths) != 1:
            raise ValueError("phone dataset columns have inconsistent lengths")
        length = lengths.pop()
        committed = np.asarray(steps["row_committed"][:], dtype=np.bool_)
        false_rows = np.flatnonzero(~committed)
        target = int(false_rows[0]) if false_rows.size else length
        if false_rows.size and committed[target:].any():
            raise ValueError("phone dataset contains a non-tail uncommitted row")
        for dataset in steps.values():
            dataset.resize(target, axis=0)
        self._file.attrs["committed_rows"] = target
        self._file.flush()

    def append(self, observation: SimWristObservation, cycle: PhoneControlCycle) -> None:
        if bool(self._file.attrs["finalized"]):
            raise RuntimeError("cannot append after finalization")
        if observation.wrist_rgb.shape != self._shape:
            raise ValueError("observation image shape differs from the file contract")
        step = cycle.step_result
        if observation.capture_monotonic_ns > step.command_queued_monotonic_ns:
            raise ValueError("wrist observation must be captured before action submission")
        sample = cycle.sample
        target = cycle.cartesian_target
        ik = cycle.ik_result
        row: dict[str, Any] = {
            "wrist_rgb": observation.wrist_rgb,
            "depth_wrist_mm": observation.depth_wrist_mm,
            "segmentation_wrist": observation.segmentation_wrist,
            "capture_monotonic_ns": observation.capture_monotonic_ns,
            "phone_present": sample is not None,
            "phone_sequence": sample.sequence if sample is not None else 0,
            "phone_receive_monotonic_ns": (
                sample.receive_monotonic_ns if sample is not None else 0
            ),
            "phone_position_m": sample.position_m if sample is not None else np.zeros(3),
            "phone_orientation_xyzw": (
                sample.orientation_xyzw if sample is not None else np.zeros(4)
            ),
            "phone_enabled": sample.enabled if sample is not None else False,
            "phone_audit_json": sample.audit_json() if sample is not None else "",
            "safety_active": cycle.decision.active,
            "safety_stop_reason": cycle.decision.stop_reason.value,
            "safety_sample_age_ns": cycle.decision.sample_age_ns,
            "safety_new_sample": cycle.decision.new_sample,
            "cartesian_target_present": target is not None,
            "target_world_from_ee": (
                target.transform_world_from_ee if target is not None else np.zeros((4, 4))
            ),
            "target_gripper_rad": target.gripper_position_rad if target is not None else 0.0,
            "ik_present": ik is not None,
            "ik_target_q_rad": ik.target_joint_position_rad if ik is not None else np.zeros(6),
            "ik_position_error_m": ik.position_error_m if ik is not None else np.nan,
            "ik_orientation_error_rad": ik.orientation_error_rad if ik is not None else np.nan,
            "ik_converged": ik.converged if ik is not None else False,
            "ik_application_scale": ik.application_scale if ik is not None else 0.0,
            "ik_safety_reason": ik.safety_reason if ik is not None else "",
            "q_before_rad": step.q_before_rad,
            "dq_before_rad_s": step.dq_before_rad_s,
            "target_q_rad": step.target_q_rad,
            "queued_target_q_rad": step.queued_target_q_rad,
            "submitted_normalized_action": step.submitted_normalized_action,
            "q_after_rad": step.q_after_rad,
            "dq_after_rad_s": step.dq_after_rad_s,
            "reward": step.reward,
            "terminated": step.terminated,
            "truncated": step.truncated,
            "command_queued_monotonic_ns": step.command_queued_monotonic_ns,
            "completed_monotonic_ns": step.completed_monotonic_ns,
            (
                "step_info_json_zlib"
                if "step_info_json_zlib" in self._file["steps"]
                else "step_info_json"
            ): (
                np.frombuffer(
                    zlib.compress(_json_text(step.info).encode("utf-8"), level=6),
                    dtype=np.uint8,
                )
                if "step_info_json_zlib" in self._file["steps"]
                else _json_text(step.info)
            ),
            "row_committed": False,
        }
        steps = self._file["steps"]
        index = self.records_written
        for dataset in steps.values():
            dataset.resize(index + 1, axis=0)
        try:
            for name, value in row.items():
                steps[name][index] = value
            self._file.flush()
            steps["row_committed"][index] = True
            self._file.attrs["committed_rows"] = index + 1
            self._file.flush()
        except Exception:
            self._file.flush()
            raise

    def finalize(
        self,
        *,
        success: bool,
        termination_reason: str,
        replay_audit: Mapping[str, Any] | None = None,
    ) -> None:
        if not isinstance(success, bool):
            raise ValueError("success must be bool")
        if not termination_reason:
            raise ValueError("termination_reason must be non-empty")
        if replay_audit is not None:
            audit = dict(replay_audit)
            mode = audit.get("sensor_materialization_mode")
            if mode not in {
                "exact_deterministic_state_replay",
                "exact_pre_action_visual_state_restore",
                "human_command_sequence_reexecution",
            }:
                raise ValueError("replay audit sensor materialization mode is invalid")
            if type(audit.get("state_exact")) is not bool:
                raise TypeError("replay audit state_exact must be bool")
            self._file.attrs["replay_audit_json"] = _json_text(audit)
            self._file.attrs["sensor_materialization_mode"] = str(mode)
            self._file.attrs["state_exact_replay"] = bool(audit["state_exact"])
        self._file.attrs["finalized"] = True
        self._file.attrs["episode_success"] = success
        self._file.attrs["eligible_for_synthetic_act_training"] = bool(
            success and self.records_written > 0
        )
        self._file.attrs["eligible_for_synthetic_vla_training"] = bool(
            success
            and self.records_written > 0
            and read_phone_task_metadata(self._file, required=False) is not None
        )
        self._file.attrs["termination_reason"] = termination_reason
        self._file.attrs["finalized_unix_ns"] = time.time_ns()
        self._file.flush()

    def close(self) -> None:
        if self._file:
            self._file.close()

    def __enter__(self) -> HDF5PhoneSimWriter:
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()


__all__ = [
    "PHONE_SIM_DATASET_SCHEMA_VERSION",
    "SIM_TELEOP_CONTROL_SOURCES",
    "HDF5PhoneSimWriter",
    "SimWristObservation",
]
