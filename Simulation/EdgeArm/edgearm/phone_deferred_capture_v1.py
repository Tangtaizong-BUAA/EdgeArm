"""Two-pass low-latency phone motion capture and deterministic RGB-D replay."""

from __future__ import annotations

import json
import hashlib
import os
import time
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import h5py
import mujoco
import numpy as np

from .multimodal import CameraCaptureConfig, TrueMultimodalRenderer
from .phone_teleop_dataset_v1 import HDF5PhoneSimWriter, SimWristObservation
from .phone_task_language_v1 import (
    PhoneTaskMetadata,
    read_phone_task_metadata,
    write_phone_task_metadata,
)
from .generalization_task_language_v1 import GeneralizationTaskMetadataV1
from .keyboard_cartesian_runtime_v1 import KEYBOARD_CARTESIAN_RUNTIME_VERSION
from .phone_teleop_runtime_v1 import (
    PHONE_CARTESIAN_MAPPING_VERSION,
    PHONE_TELEOP_RUNTIME_VERSION,
    CartesianTarget,
    IKResult,
    PhoneControlCycle,
    PhonePoseSample,
    PhoneSafetyDecision,
    SimStepResult,
    StopReason,
    TrackingState,
)
from .sim2real_env_v10 import RealisticEdgeArmEnvV10, RealisticEnvV10Config
from .sim2real_env_v12 import (
    MOUNTED_WRIST_CAMERA_DYNAMICS_PROFILE_V12,
    RealisticEdgeArmEnvV12,
    RealisticEnvV12Config,
)
from .sim2real_env_v13 import (
    GENERALIZATION_DYNAMICS_PROFILE_V13,
    VISUAL_REPLAY_SNAPSHOT_FORMAT_V28,
    RealisticEdgeArmEnvV13,
    RealisticEnvV13Config,
    keyboard_teleop_env_config_v13,
)


PHONE_MOTION_TRACE_SCHEMA_VERSION = "edgearm-phone-motion-trace-v1"
EXACT_VISUAL_STATE_RESTORE_MODE_V28 = "exact_pre_action_visual_state_restore"


def _decode(value: Any) -> str:
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    return repr(value)


def _json_text(value: Any) -> str:
    return json.dumps(
        value,
        default=_json_default,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def phone_step_strict_success(info: dict[str, Any]) -> bool:
    """Training admission gate shared by live capture and deterministic replay."""

    realism = info.get("realism_v6", {})
    base_success = bool(
        info.get("success", False)
        and realism.get("raw_strict_success", False)
        and realism.get("strict_contained", False)
        and realism.get("strict_settled", False)
        and float(realism.get("strict_target_coverage", 0.0)) >= 0.95
        and int(realism.get("strict_success_streak", 0)) >= 6
    )
    realism_v13 = info.get("realism_v13")
    if realism_v13 is None:
        return base_success
    return bool(
        base_success
        and realism_v13.get("strict_stable_3s", False)
        and float(realism_v13.get("continuous_stable_seconds", 0.0)) >= 3.0
        and realism_v13.get("strict_no_obstacle_contact", False)
    )


@dataclass(frozen=True)
class PhoneMotionTrace:
    path: Path
    episode_seed: int
    environment_profile: str
    episode_success: bool
    termination_reason: str
    task_metadata: PhoneTaskMetadata | GeneralizationTaskMetadataV1 | None
    scene_contract: dict[str, Any] | None
    control_source: str
    control_runtime_version: str
    initial_arm_pose_overridden: bool
    initial_arm_qpos_rad: np.ndarray | None
    initial_arm_qvel_rad_s: np.ndarray | None
    initial_arm_ctrl_rad: np.ndarray | None
    cycles: tuple[PhoneControlCycle, ...]


def _snapshot_vector(
    snapshot: Mapping[str, Any],
    key: str,
    length: int,
) -> np.ndarray:
    value = np.asarray(snapshot.get(key), dtype=np.float64)
    if value.shape != (length,) or not np.isfinite(value).all():
        raise ValueError(f"V28 visual replay snapshot {key} must be finite [{length}]")
    return value


def _snapshot_matrix(
    snapshot: Mapping[str, Any],
    key: str,
    shape: tuple[int, int],
) -> np.ndarray:
    value = np.asarray(snapshot.get(key), dtype=np.float64)
    if value.shape != shape or not np.isfinite(value).all():
        raise ValueError(f"V28 visual replay snapshot {key} has an invalid shape")
    return value


def _visual_replay_snapshot_v28(cycle: PhoneControlCycle) -> Mapping[str, Any] | None:
    snapshot = cycle.step_result.info.get("visual_replay_snapshot_v28")
    if snapshot is None:
        return None
    if not isinstance(snapshot, Mapping):
        raise ValueError("V28 visual replay snapshot must be an object")
    if snapshot.get("format") != VISUAL_REPLAY_SNAPSHOT_FORMAT_V28:
        raise ValueError("V28 visual replay snapshot format mismatch")
    if snapshot.get("supports_exact_visual_state_restore") is not True:
        raise ValueError("V28 snapshot does not authorize exact visual state restore")
    if snapshot.get("supports_exact_dynamic_resume") is not False:
        raise ValueError("V28 snapshot must not claim exact dynamic resume")
    if int(snapshot.get("physical_samples", -1)) != 0:
        raise ValueError("V28 synthetic snapshot carries an invalid physical claim")
    if snapshot.get("production_admission") is not False:
        raise ValueError("V28 synthetic snapshot must not claim production admission")
    return snapshot


def _trace_has_complete_visual_snapshots_v28(trace: PhoneMotionTrace) -> bool:
    snapshots = [_visual_replay_snapshot_v28(cycle) for cycle in trace.cycles]
    present = [snapshot is not None for snapshot in snapshots]
    if any(present) and not all(present):
        raise ValueError("V28 motion trace contains a partial visual snapshot sequence")
    if not all(present):
        return False
    previous_post: Mapping[str, Any] | None = None
    for row_index, snapshot_or_none in enumerate(snapshots):
        assert snapshot_or_none is not None
        snapshot = snapshot_or_none
        pre_time = float(snapshot.get("pre_action_time_seconds", np.nan))
        post_time = float(snapshot.get("post_action_time_seconds", np.nan))
        if not np.isfinite(pre_time) or not np.isfinite(post_time) or post_time < pre_time:
            raise ValueError(f"V28 snapshot time bounds are invalid at row {row_index}")
        if previous_post is not None:
            for pre_key, post_key in (
                ("pre_action_qpos", "post_action_qpos"),
                ("pre_action_qvel", "post_action_qvel"),
                ("pre_action_ctrl", "post_action_ctrl"),
            ):
                current = np.asarray(snapshot.get(pre_key), dtype=np.float64)
                previous = np.asarray(previous_post.get(post_key), dtype=np.float64)
                if current.shape != previous.shape or not np.array_equal(current, previous):
                    raise ValueError(
                        f"V28 visual snapshot continuity changed at row {row_index}: {pre_key}"
                    )
            if pre_time != float(previous_post.get("post_action_time_seconds", np.nan)):
                raise ValueError(f"V28 visual snapshot time is discontinuous at row {row_index}")
        previous_post = snapshot
    return True


def _restore_visual_replay_snapshot_v28(
    env: RealisticEdgeArmEnvV13,
    snapshot: Mapping[str, Any],
    *,
    camera_pose_tolerance: float = 2.0e-8,
) -> dict[str, float]:
    """Restore a recorded pre-action MuJoCo state for image-only materialization.

    This deliberately does not resume dynamics.  The original transition and
    labels remain authoritative; only the causal pre-action camera tensor is
    rendered from the fully recorded simulator state.
    """

    qpos = _snapshot_vector(snapshot, "pre_action_qpos", env.model.nq)
    qvel = _snapshot_vector(snapshot, "pre_action_qvel", env.model.nv)
    ctrl = _snapshot_vector(snapshot, "pre_action_ctrl", env.model.nu)
    post_qpos = _snapshot_vector(snapshot, "post_action_qpos", env.model.nq)
    _snapshot_vector(snapshot, "post_action_qvel", env.model.nv)
    _snapshot_vector(snapshot, "post_action_ctrl", env.model.nu)
    local_position = _snapshot_vector(
        snapshot,
        "pre_action_wrist_camera_local_position",
        3,
    )
    local_quaternion = _snapshot_vector(
        snapshot,
        "pre_action_wrist_camera_local_quaternion_wxyz",
        4,
    )
    world_position = _snapshot_vector(
        snapshot,
        "pre_action_wrist_camera_world_position",
        3,
    )
    world_rotation = _snapshot_matrix(
        snapshot,
        "pre_action_wrist_camera_world_rotation",
        (3, 3),
    )
    if abs(float(np.linalg.norm(local_quaternion)) - 1.0) > 1.0e-8:
        raise ValueError("V28 visual replay camera quaternion is not normalized")
    pre_time = float(snapshot.get("pre_action_time_seconds", np.nan))
    post_time = float(snapshot.get("post_action_time_seconds", np.nan))
    if not np.isfinite(pre_time) or not np.isfinite(post_time) or post_time < pre_time:
        raise ValueError("V28 visual replay snapshot time bounds are invalid")
    camera_id = int(env._ids["cameras"]["wrist"])
    env.data.qpos[:] = qpos
    env.data.qvel[:] = qvel
    env.data.ctrl[:] = ctrl
    env.data.time = pre_time
    env.model.cam_pos[camera_id] = local_position
    env.model.cam_quat[camera_id] = local_quaternion
    mujoco.mj_forward(env.model, env.data)
    position_error = float(np.max(np.abs(env.data.cam_xpos[camera_id] - world_position)))
    rotation_error = float(
        np.max(
            np.abs(
                env.data.cam_xmat[camera_id].reshape(3, 3) - world_rotation
            )
        )
    )
    if max(position_error, rotation_error) > camera_pose_tolerance:
        raise RuntimeError(
            "V28 restored wrist-camera pose differs from the recorded snapshot: "
            f"{max(position_error, rotation_error):.9g}"
        )
    return {
        "camera_position_error": position_error,
        "camera_rotation_error": rotation_error,
        "recorded_transition_qpos_max_delta": float(np.max(np.abs(post_qpos - qpos))),
    }


def write_phone_motion_trace(
    path: Path,
    cycles: list[PhoneControlCycle],
    *,
    episode_seed: int,
    environment_profile: str,
    episode_success: bool,
    termination_reason: str,
    task_metadata: PhoneTaskMetadata | GeneralizationTaskMetadataV1 | None = None,
    scene_contract: Mapping[str, Any] | None = None,
    control_source: str = "phone",
    control_runtime_version: str = PHONE_TELEOP_RUNTIME_VERSION,
    initial_arm_pose_overridden: bool = False,
    initial_arm_qpos_rad: np.ndarray | None = None,
    initial_arm_qvel_rad_s: np.ndarray | None = None,
    initial_arm_ctrl_rad: np.ndarray | None = None,
    overwrite: bool = False,
) -> Path:
    if not cycles:
        raise ValueError("cannot write an empty phone motion trace")
    control_source = str(control_source).strip().lower()
    if control_source not in {"phone", "keyboard"}:
        raise ValueError("control_source must be phone or keyboard")
    control_runtime_version = str(control_runtime_version).strip()
    if not control_runtime_version:
        raise ValueError("control_runtime_version must be non-empty")
    if not isinstance(initial_arm_pose_overridden, bool):
        raise ValueError("initial_arm_pose_overridden must be bool")
    if isinstance(task_metadata, GeneralizationTaskMetadataV1):
        if scene_contract is None:
            raise ValueError("generalization task requires a scene contract")
        if task_metadata.scene_contract_sha256 != hashlib.sha256(
            _json_text(dict(scene_contract)).encode("utf-8")
        ).hexdigest():
            raise ValueError("generalization task scene hash differs from motion trace scene")
    initial_values = (
        initial_arm_qpos_rad,
        initial_arm_qvel_rad_s,
        initial_arm_ctrl_rad,
    )
    if initial_arm_pose_overridden and any(value is None for value in initial_values):
        raise ValueError(
            "an overridden initial arm pose requires qpos, qvel, and ctrl"
        )
    if not initial_arm_pose_overridden and any(value is not None for value in initial_values):
        raise ValueError("initial plant state requires initial_arm_pose_overridden")
    frozen_initial: tuple[np.ndarray, ...] | None = None
    if initial_arm_pose_overridden:
        arrays = tuple(np.asarray(value, dtype=np.float64) for value in initial_values)
        if any(array.shape != (6,) or not np.isfinite(array).all() for array in arrays):
            raise ValueError("initial qpos, qvel, and ctrl must be finite [6]")
        frozen_initial = arrays
    path = Path(path)
    if path.exists() and not overwrite:
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    if temporary.exists():
        temporary.unlink()
    count = len(cycles)
    string_dtype = h5py.string_dtype("utf-8")
    try:
        with h5py.File(temporary, "w") as stream:
            stream.attrs.update(
                {
                    "schema_version": PHONE_MOTION_TRACE_SCHEMA_VERSION,
                    "phone_runtime_version": PHONE_TELEOP_RUNTIME_VERSION,
                    "phone_cartesian_mapping_version": PHONE_CARTESIAN_MAPPING_VERSION,
                    "operator_input_source": control_source,
                    "operator_input_runtime_version": control_runtime_version,
                    "initial_arm_pose_overridden": initial_arm_pose_overridden,
                    "episode_seed": episode_seed,
                    "environment_profile": environment_profile,
                    "episode_success": episode_success,
                    "termination_reason": termination_reason,
                    "rows": count,
                    "backend": "mujoco",
                    "contains_images": False,
                    "physical_samples": 0,
                    "physical_capture_claimed": False,
                    "language_condition_present": task_metadata is not None,
                    "created_unix_ns": time.time_ns(),
                }
            )
            if task_metadata is not None:
                write_phone_task_metadata(stream, task_metadata)
            if scene_contract is not None:
                scene_contract_json = _json_text(dict(scene_contract))
                stream.attrs["scene_contract_json"] = scene_contract_json
                stream.attrs["scene_contract_sha256"] = hashlib.sha256(
                    scene_contract_json.encode("utf-8")
                ).hexdigest()
            if frozen_initial is not None:
                initial = stream.create_group("initial_plant_state")
                initial.create_dataset("arm_qpos_rad", data=frozen_initial[0])
                initial.create_dataset("arm_qvel_rad_s", data=frozen_initial[1])
                initial.create_dataset("arm_ctrl_rad", data=frozen_initial[2])
            steps = stream.create_group("steps")

            def stack(name: str) -> np.ndarray:
                return np.stack([getattr(cycle.step_result, name) for cycle in cycles])

            for name in (
                "q_before_rad",
                "dq_before_rad_s",
                "target_q_rad",
                "queued_target_q_rad",
                "submitted_normalized_action",
                "q_after_rad",
                "dq_after_rad_s",
            ):
                steps.create_dataset(name, data=stack(name), chunks=True)
            for name, dtype in (
                ("reward", np.float64),
                ("terminated", np.bool_),
                ("truncated", np.bool_),
                ("command_queued_monotonic_ns", np.int64),
                ("completed_monotonic_ns", np.int64),
            ):
                steps.create_dataset(
                    name,
                    data=np.asarray(
                        [getattr(cycle.step_result, name) for cycle in cycles],
                        dtype=dtype,
                    ),
                )
            compressed_info = [
                zlib.compress(_json_text(cycle.step_result.info).encode("utf-8"), level=6)
                for cycle in cycles
            ]
            info_offsets = np.zeros(count + 1, dtype=np.int64)
            info_offsets[1:] = np.cumsum(
                np.asarray([len(payload) for payload in compressed_info], dtype=np.int64)
            )
            info_bytes = np.frombuffer(b"".join(compressed_info), dtype=np.uint8)
            steps.create_dataset("step_info_zlib_bytes", data=info_bytes, chunks=True)
            steps.create_dataset("step_info_zlib_offsets", data=info_offsets)
            stream.attrs["step_info_codec"] = "zlib-json-concatenated-v1"

            sample_present = np.asarray([cycle.sample is not None for cycle in cycles], dtype=np.bool_)
            steps.create_dataset("phone_present", data=sample_present)
            steps.create_dataset(
                "phone_audit_json",
                data=np.asarray(
                    [cycle.sample.audit_json() if cycle.sample is not None else "" for cycle in cycles],
                    dtype=object,
                ),
                dtype=string_dtype,
            )
            for name, dtype in (
                ("safety_active", np.bool_),
                ("safety_sample_age_ns", np.int64),
                ("safety_new_sample", np.bool_),
            ):
                attribute = {
                    "safety_active": "active",
                    "safety_sample_age_ns": "sample_age_ns",
                    "safety_new_sample": "new_sample",
                }[name]
                steps.create_dataset(
                    name,
                    data=np.asarray(
                        [getattr(cycle.decision, attribute) for cycle in cycles],
                        dtype=dtype,
                    ),
                )
            steps.create_dataset(
                "safety_stop_reason",
                data=np.asarray([cycle.decision.stop_reason.value for cycle in cycles], dtype=object),
                dtype=string_dtype,
            )

            cart_present = np.asarray(
                [cycle.cartesian_target is not None for cycle in cycles], dtype=np.bool_
            )
            steps.create_dataset("cartesian_target_present", data=cart_present)
            steps.create_dataset(
                "target_world_from_ee",
                data=np.stack(
                    [
                        cycle.cartesian_target.transform_world_from_ee
                        if cycle.cartesian_target is not None
                        else np.zeros((4, 4))
                        for cycle in cycles
                    ]
                ),
            )
            steps.create_dataset(
                "target_gripper_rad",
                data=np.asarray(
                    [
                        cycle.cartesian_target.gripper_position_rad
                        if cycle.cartesian_target is not None
                        else 0.0
                        for cycle in cycles
                    ]
                ),
            )
            for flag in (
                "phone_translation_clipped",
                "phone_rotation_clipped",
                "workspace_clipped",
                "rate_limited",
            ):
                steps.create_dataset(
                    f"target_{flag}",
                    data=np.asarray(
                        [
                            getattr(cycle.cartesian_target, flag)
                            if cycle.cartesian_target is not None
                            else False
                            for cycle in cycles
                        ],
                        dtype=np.bool_,
                    ),
                )

            ik_present = np.asarray([cycle.ik_result is not None for cycle in cycles], dtype=np.bool_)
            steps.create_dataset("ik_present", data=ik_present)
            steps.create_dataset(
                "ik_target_q_rad",
                data=np.stack(
                    [
                        cycle.ik_result.target_joint_position_rad
                        if cycle.ik_result is not None
                        else np.zeros(6)
                        for cycle in cycles
                    ]
                ),
            )
            for name, dtype, fallback in (
                ("position_error_m", np.float64, np.nan),
                ("orientation_error_rad", np.float64, np.nan),
                ("converged", np.bool_, False),
                ("joint_or_workspace_clipped", np.bool_, False),
            ):
                steps.create_dataset(
                    f"ik_{name}",
                    data=np.asarray(
                        [
                            getattr(cycle.ik_result, name) if cycle.ik_result is not None else fallback
                            for cycle in cycles
                        ],
                        dtype=dtype,
                    ),
                )
            steps.create_dataset(
                "ik_safety_reason",
                data=np.asarray(
                    [
                        cycle.ik_result.safety_reason if cycle.ik_result is not None else ""
                        for cycle in cycles
                    ],
                    dtype=object,
                ),
                dtype=string_dtype,
            )
            steps.create_dataset(
                "ik_application_scale",
                data=np.asarray(
                    [
                        cycle.ik_result.application_scale
                        if cycle.ik_result is not None
                        else 0.0
                        for cycle in cycles
                    ],
                    dtype=np.float64,
                ),
            )
            stream.flush()
        os.replace(temporary, path)
    except Exception:
        if temporary.exists():
            temporary.unlink()
        raise
    return path.resolve()


def read_phone_motion_trace(path: Path) -> PhoneMotionTrace:
    path = Path(path).resolve()
    with h5py.File(path, "r") as stream:
        if stream.attrs.get("schema_version", "") != PHONE_MOTION_TRACE_SCHEMA_VERSION:
            raise ValueError("motion trace schema mismatch")
        if bool(stream.attrs.get("contains_images", True)):
            raise ValueError("motion trace unexpectedly claims image payloads")
        if int(stream.attrs.get("physical_samples", -1)) != 0 or bool(
            stream.attrs.get("physical_capture_claimed", True)
        ):
            raise ValueError("motion trace carries invalid physical claims")
        task_metadata = read_phone_task_metadata(stream, required=False)
        scene_contract_text = _decode(stream.attrs.get("scene_contract_json", ""))
        scene_contract: dict[str, Any] | None = None
        if scene_contract_text:
            expected_scene_hash = _decode(stream.attrs.get("scene_contract_sha256", ""))
            observed_scene_hash = hashlib.sha256(
                scene_contract_text.encode("utf-8")
            ).hexdigest()
            if expected_scene_hash != observed_scene_hash:
                raise ValueError("motion trace scene contract hash mismatch")
            loaded_scene_contract = json.loads(scene_contract_text)
            if not isinstance(loaded_scene_contract, dict):
                raise ValueError("motion trace scene contract must be an object")
            scene_contract = loaded_scene_contract
        if isinstance(task_metadata, GeneralizationTaskMetadataV1):
            if scene_contract is None:
                raise ValueError("generalization task has no scene contract")
            if task_metadata.scene_contract_sha256 != hashlib.sha256(
                _json_text(scene_contract).encode("utf-8")
            ).hexdigest():
                raise ValueError("generalization task is bound to a different scene contract")
        steps = stream["steps"]
        count = int(stream.attrs["rows"])
        row_datasets = [
            dataset
            for name, dataset in steps.items()
            if name not in {"step_info_zlib_bytes", "step_info_zlib_offsets"}
        ]
        if {dataset.shape[0] for dataset in row_datasets} != {count}:
            raise ValueError("motion trace columns have inconsistent lengths")
        if "step_info_zlib_bytes" in steps:
            offsets = np.asarray(steps["step_info_zlib_offsets"][:], dtype=np.int64)
            if offsets.shape != (count + 1,) or offsets[0] != 0:
                raise ValueError("motion trace compressed step-info offsets are invalid")
            info_bytes = np.asarray(steps["step_info_zlib_bytes"][:], dtype=np.uint8)
            if offsets[-1] != info_bytes.size or np.any(np.diff(offsets) < 0):
                raise ValueError("motion trace compressed step-info bounds are invalid")
        else:
            offsets = None
            info_bytes = None
        cycles: list[PhoneControlCycle] = []
        for index in range(count):
            sample = None
            if bool(steps["phone_present"][index]):
                payload = json.loads(_decode(steps["phone_audit_json"][index]))
                sample = PhonePoseSample(
                    sequence=int(payload["sequence"]),
                    receive_monotonic_ns=int(payload["receive_monotonic_ns"]),
                    position_m=np.asarray(payload["position_m"]),
                    orientation_xyzw=np.asarray(payload["orientation_xyzw"]),
                    enabled=bool(payload["enabled"]),
                    raw_inputs=payload["raw_inputs"],
                    tracking_state=TrackingState(payload["tracking_state"]),
                    device_timestamp_ns=payload["device_timestamp_ns"],
                    source=str(payload["source"]),
                )
            decision = PhoneSafetyDecision(
                active=bool(steps["safety_active"][index]),
                stop_reason=StopReason(_decode(steps["safety_stop_reason"][index])),
                sample_age_ns=int(steps["safety_sample_age_ns"][index]),
                new_sample=bool(steps["safety_new_sample"][index]),
            )
            cartesian = None
            if bool(steps["cartesian_target_present"][index]):
                cartesian = CartesianTarget(
                    transform_world_from_ee=steps["target_world_from_ee"][index],
                    gripper_position_rad=float(steps["target_gripper_rad"][index]),
                    phone_translation_clipped=bool(steps["target_phone_translation_clipped"][index]),
                    phone_rotation_clipped=bool(steps["target_phone_rotation_clipped"][index]),
                    workspace_clipped=bool(steps["target_workspace_clipped"][index]),
                    rate_limited=bool(steps["target_rate_limited"][index]),
                )
            ik = None
            if bool(steps["ik_present"][index]):
                ik = IKResult(
                    target_joint_position_rad=steps["ik_target_q_rad"][index],
                    position_error_m=float(steps["ik_position_error_m"][index]),
                    orientation_error_rad=float(steps["ik_orientation_error_rad"][index]),
                    converged=bool(steps["ik_converged"][index]),
                    joint_or_workspace_clipped=bool(steps["ik_joint_or_workspace_clipped"][index]),
                    safety_reason=_decode(steps["ik_safety_reason"][index]),
                    application_scale=(
                        float(steps["ik_application_scale"][index])
                        if "ik_application_scale" in steps
                        else 1.0
                    ),
                )
            if offsets is not None and info_bytes is not None:
                start = int(offsets[index])
                stop = int(offsets[index + 1])
                info = json.loads(zlib.decompress(info_bytes[start:stop].tobytes()))
            else:
                info = json.loads(_decode(steps["step_info_json"][index]))
            step = SimStepResult(
                q_before_rad=steps["q_before_rad"][index],
                dq_before_rad_s=steps["dq_before_rad_s"][index],
                target_q_rad=steps["target_q_rad"][index],
                queued_target_q_rad=steps["queued_target_q_rad"][index],
                submitted_normalized_action=steps["submitted_normalized_action"][index],
                q_after_rad=steps["q_after_rad"][index],
                dq_after_rad_s=steps["dq_after_rad_s"][index],
                reward=float(steps["reward"][index]),
                terminated=bool(steps["terminated"][index]),
                truncated=bool(steps["truncated"][index]),
                info=info,
                command_queued_monotonic_ns=int(steps["command_queued_monotonic_ns"][index]),
                completed_monotonic_ns=int(steps["completed_monotonic_ns"][index]),
            )
            cycles.append(PhoneControlCycle(sample, decision, cartesian, ik, step))
        initial = stream.get("initial_plant_state")
        return PhoneMotionTrace(
            path=path,
            episode_seed=int(stream.attrs["episode_seed"]),
            environment_profile=str(stream.attrs["environment_profile"]),
            episode_success=bool(stream.attrs["episode_success"]),
            termination_reason=str(stream.attrs["termination_reason"]),
            task_metadata=task_metadata,
            scene_contract=scene_contract,
            control_source=str(
                stream.attrs.get("operator_input_source", "phone")
            ),
            control_runtime_version=str(
                stream.attrs.get(
                    "operator_input_runtime_version",
                    PHONE_TELEOP_RUNTIME_VERSION,
                )
            ),
            initial_arm_pose_overridden=bool(
                stream.attrs.get("initial_arm_pose_overridden", False)
            ),
            initial_arm_qpos_rad=(
                np.asarray(initial["arm_qpos_rad"][:], dtype=np.float64)
                if initial is not None
                else None
            ),
            initial_arm_qvel_rad_s=(
                np.asarray(initial["arm_qvel_rad_s"][:], dtype=np.float64)
                if initial is not None
                else None
            ),
            initial_arm_ctrl_rad=(
                np.asarray(initial["arm_ctrl_rad"][:], dtype=np.float64)
                if initial is not None
                else None
            ),
            cycles=tuple(cycles),
        )


def replay_phone_motion_trace(
    trace: PhoneMotionTrace,
    output: Path,
    *,
    width: int = 320,
    height: int = 240,
    overwrite: bool = False,
    joint_tolerance_rad: float = 2.0e-5,
    state_match_mode: str = "strict_exact",
) -> dict[str, Any]:
    if not trace.cycles:
        raise ValueError("motion trace is empty")
    if state_match_mode not in {"strict_exact", "command_reexecution"}:
        raise ValueError("state_match_mode must be strict_exact or command_reexecution")
    if trace.environment_profile == GENERALIZATION_DYNAMICS_PROFILE_V13:
        replay_config = (
            keyboard_teleop_env_config_v13(max_steps=len(trace.cycles))
            if trace.control_source == "keyboard"
            and trace.control_runtime_version == KEYBOARD_CARTESIAN_RUNTIME_VERSION
            else RealisticEnvV13Config(max_steps=len(trace.cycles))
        )
        env = RealisticEdgeArmEnvV13(
            replay_config,
            seed=trace.episode_seed,
        )
    elif trace.environment_profile == MOUNTED_WRIST_CAMERA_DYNAMICS_PROFILE_V12:
        env = RealisticEdgeArmEnvV12(
            RealisticEnvV12Config(max_steps=len(trace.cycles)),
            seed=trace.episode_seed,
        )
    else:
        # Preserve deterministic replay for already-written V10 traces.  Any
        # unknown profile still fails closed in the equality check below.
        env = RealisticEdgeArmEnvV10(
            RealisticEnvV10Config(max_steps=len(trace.cycles)),
            seed=trace.episode_seed,
        )
    if trace.environment_profile == GENERALIZATION_DYNAMICS_PROFILE_V13:
        if trace.scene_contract is None:
            raise ValueError("V13 motion trace has no realized scene contract")
        env.reset(seed=trace.episode_seed, scenario=trace.scene_contract)
    else:
        obstacle_enabled = (
            trace.task_metadata.obstacle_enabled if trace.task_metadata is not None else False
        )
        env.reset(seed=trace.episode_seed, obstacle=obstacle_enabled)
    if env.profile_version != trace.environment_profile:
        raise ValueError("motion trace environment profile differs from replay")
    if trace.task_metadata is not None:
        replay_task = (
            GeneralizationTaskMetadataV1.from_scenario(env.current_scenario)
            if isinstance(trace.task_metadata, GeneralizationTaskMetadataV1)
            else PhoneTaskMetadata.from_environment(env)
        )
        if replay_task != trace.task_metadata:
            raise ValueError("motion trace language task differs from reconstructed scene")
    if trace.initial_arm_pose_overridden:
        if (
            trace.initial_arm_qpos_rad is None
            or trace.initial_arm_qvel_rad_s is None
            or trace.initial_arm_ctrl_rad is None
        ):
            raise ValueError("overridden initial pose has no exact plant state")
        env.data.qpos[:6] = trace.initial_arm_qpos_rad
        env.data.qvel[:6] = trace.initial_arm_qvel_rad_s
        env.data.ctrl[:6] = trace.initial_arm_ctrl_rad
        mujoco.mj_forward(env.model, env.data)
    snapshot_restore = _trace_has_complete_visual_snapshots_v28(trace)
    if snapshot_restore and not isinstance(env, RealisticEdgeArmEnvV13):
        raise ValueError("V28 visual state snapshots are only valid for the V13 plant")
    maximum_camera_position_error = 0.0
    maximum_camera_rotation_error = 0.0
    if snapshot_restore:
        first_snapshot = _visual_replay_snapshot_v28(trace.cycles[0])
        assert first_snapshot is not None
        first_restore = _restore_visual_replay_snapshot_v28(env, first_snapshot)
        maximum_camera_position_error = first_restore["camera_position_error"]
        maximum_camera_rotation_error = first_restore["camera_rotation_error"]
    renderer = TrueMultimodalRenderer(
        env,
        CameraCaptureConfig(width=width, height=height, cameras=("wrist",)),
    )
    renderer.begin_episode(trace.episode_seed)
    maximum_before_error = 0.0
    maximum_after_error = 0.0
    first_state_divergence_row: int | None = None
    try:
        with HDF5PhoneSimWriter(
            output,
            image_shape=(height, width, 3),
            episode_seed=trace.episode_seed,
            environment_profile=env.profile_version,
            camera_metadata=renderer.calibration_metadata(),
            task_metadata=trace.task_metadata,
            scene_contract=trace.scene_contract,
            control_source=trace.control_source,
            control_runtime_version=trace.control_runtime_version,
            overwrite=overwrite,
        ) as writer:
            replay_success = False
            replay_strict_success = False
            for row_index, original in enumerate(trace.cycles):
                if snapshot_restore:
                    snapshot = _visual_replay_snapshot_v28(original)
                    assert snapshot is not None
                    restore_audit = _restore_visual_replay_snapshot_v28(env, snapshot)
                    maximum_camera_position_error = max(
                        maximum_camera_position_error,
                        restore_audit["camera_position_error"],
                    )
                    maximum_camera_rotation_error = max(
                        maximum_camera_rotation_error,
                        restore_audit["camera_rotation_error"],
                    )
                capture_ns = time.monotonic_ns()
                capture = renderer.capture()
                if snapshot_restore:
                    # Images are rendered from the exact recorded pre-action
                    # plant state.  The original transition stays authoritative;
                    # advancing contact dynamics here would create a new episode.
                    queued_ns = time.monotonic_ns()
                    completed_ns = time.monotonic_ns()
                    current = np.concatenate(
                        (
                            np.asarray(original.step_result.q_before_rad, dtype=np.float64),
                            np.asarray(original.step_result.dq_before_rad_s, dtype=np.float64),
                        )
                    )
                    state_after = np.concatenate(
                        (
                            np.asarray(original.step_result.q_after_rad, dtype=np.float64),
                            np.asarray(original.step_result.dq_after_rad_s, dtype=np.float64),
                        )
                    )
                    reward = original.step_result.reward
                    terminated = original.step_result.terminated
                    truncated = original.step_result.truncated
                    info = original.step_result.info
                else:
                    current = np.asarray(env.observation()["joint_state"], dtype=np.float64)
                    before_error = float(
                        np.max(np.abs(current[:6] - original.step_result.q_before_rad))
                    )
                    maximum_before_error = max(maximum_before_error, before_error)
                    if before_error > joint_tolerance_rad and first_state_divergence_row is None:
                        first_state_divergence_row = row_index
                    if before_error > joint_tolerance_rad and state_match_mode == "strict_exact":
                        raise RuntimeError(
                            f"deterministic replay q_before diverged by {before_error:.6g} rad"
                        )
                    action = original.step_result.submitted_normalized_action
                    queued_ns = time.monotonic_ns()
                    observation, reward, terminated, truncated, info = env.step(action)
                    completed_ns = time.monotonic_ns()
                    state_after = np.asarray(observation["joint_state"], dtype=np.float64)
                    after_error = float(
                        np.max(np.abs(state_after[:6] - original.step_result.q_after_rad))
                    )
                    maximum_after_error = max(maximum_after_error, after_error)
                    if after_error > joint_tolerance_rad and first_state_divergence_row is None:
                        first_state_divergence_row = row_index
                    if after_error > joint_tolerance_rad and state_match_mode == "strict_exact":
                        raise RuntimeError(
                            f"deterministic replay q_after diverged by {after_error:.6g} rad"
                        )
                replay_step = SimStepResult(
                    q_before_rad=current[:6],
                    dq_before_rad_s=current[6:],
                    target_q_rad=original.step_result.target_q_rad,
                    queued_target_q_rad=original.step_result.queued_target_q_rad,
                    submitted_normalized_action=(
                        original.step_result.submitted_normalized_action
                    ),
                    q_after_rad=state_after[:6],
                    dq_after_rad_s=state_after[6:],
                    reward=float(reward),
                    terminated=bool(terminated),
                    truncated=bool(truncated),
                    info=info,
                    command_queued_monotonic_ns=queued_ns,
                    completed_monotonic_ns=completed_ns,
                )
                replay_cycle = PhoneControlCycle(
                    original.sample,
                    original.decision,
                    original.cartesian_target,
                    original.ik_result,
                    replay_step,
                )
                writer.append(
                    SimWristObservation(
                        wrist_rgb=capture["rgb_wrist"],
                        depth_wrist_mm=capture["depth_wrist_mm"],
                        segmentation_wrist=capture["segmentation_wrist"],
                        capture_monotonic_ns=capture_ns,
                    ),
                    replay_cycle,
                )
                replay_success = replay_success or bool(info.get("success", False))
                replay_strict_success = replay_strict_success or phone_step_strict_success(info)
            admitted_success = bool(trace.episode_success and replay_strict_success)
            state_exact = bool(snapshot_restore or first_state_divergence_row is None)
            if snapshot_restore:
                sensor_materialization_mode = EXACT_VISUAL_STATE_RESTORE_MODE_V28
            elif state_exact:
                sensor_materialization_mode = "exact_deterministic_state_replay"
            else:
                sensor_materialization_mode = "human_command_sequence_reexecution"
            writer.finalize(
                success=admitted_success,
                termination_reason=(
                    f"deferred_snapshot_restore:{trace.termination_reason}"
                    if snapshot_restore
                    else f"deferred_replay:{trace.termination_reason}"
                    if state_exact
                    else f"command_reexecution:{trace.termination_reason}"
                ),
                replay_audit={
                    "requested_state_match_mode": state_match_mode,
                    "sensor_materialization_mode": sensor_materialization_mode,
                    "state_exact": state_exact,
                    "joint_tolerance_rad": joint_tolerance_rad,
                    "first_state_divergence_row": first_state_divergence_row,
                    "maximum_q_before_error_rad": maximum_before_error,
                    "maximum_q_after_error_rad": maximum_after_error,
                    "maximum_camera_position_restore_error": (
                        maximum_camera_position_error
                    ),
                    "maximum_camera_rotation_restore_error": (
                        maximum_camera_rotation_error
                    ),
                    "exact_pre_action_visual_state_restored": snapshot_restore,
                    "original_transition_labels_reused": snapshot_restore,
                    "supports_exact_dynamic_resume": False,
                    "human_command_sequence_unchanged": True,
                    "replay_outcome_recomputed": not snapshot_restore,
                    "production_admission": False,
                },
            )
            records = writer.records_written
    finally:
        renderer.close()
    return {
        "output": str(Path(output).resolve()),
        "records": records,
        "trace_success": trace.episode_success,
        "replay_success": replay_success,
        "replay_strict_success": replay_strict_success,
        "act_eligible": admitted_success,
        "vla_eligible": bool(admitted_success and trace.task_metadata is not None),
        "task_id": trace.task_metadata.task_id if trace.task_metadata is not None else None,
        "operator_input_source": trace.control_source,
        "max_q_before_error_rad": maximum_before_error,
        "max_q_after_error_rad": maximum_after_error,
        "state_exact": state_exact,
        "first_state_divergence_row": first_state_divergence_row,
        "sensor_materialization_mode": sensor_materialization_mode,
        "maximum_camera_position_restore_error": maximum_camera_position_error,
        "maximum_camera_rotation_restore_error": maximum_camera_rotation_error,
        "exact_pre_action_visual_state_restored": snapshot_restore,
        "original_transition_labels_reused": snapshot_restore,
        "human_command_sequence_unchanged": True,
        "replay_outcome_recomputed": not snapshot_restore,
        "physical_samples": 0,
    }


__all__ = [
    "PHONE_MOTION_TRACE_SCHEMA_VERSION",
    "PhoneMotionTrace",
    "phone_step_strict_success",
    "read_phone_motion_trace",
    "replay_phone_motion_trace",
    "write_phone_motion_trace",
]
