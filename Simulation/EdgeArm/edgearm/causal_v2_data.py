"""Fail-closed ACT view over execution-grounded Causal V2 trajectories.

The policy input boundary in this module is intentionally small.  A sample may
read the current wrist RGB-D packet, reported joint state, decision-aligned
reported execution history, and camera geometry/clock state.  Simulator
physical state, object state, contact, segmentation, teacher diagnostics, and
same-row ``effect_*`` execution are never returned as policy inputs.

The metadata-only trajectory-contract audit is a prerequisite, not a payload
validator.  This loader therefore also checks every consumed dataset's shape,
dtype, finite values, source-specific action admission, masks, and
rolling-shutter/world-model eligibility before it creates an ACT index.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from .action_admission_v1 import (
    ACTION_ADMISSION_GATE_FORMAT,
    EPISODE_INTEGRITY_GATE_FORMAT,
    GENERIC_ACTION_LABEL_DATASET,
    SUPPORTED_SOURCE_TYPES,
    V11_SOURCE_TYPE,
    V11_TEACHER_DIAGNOSTIC_ALIAS,
    action_source_registry_json,
    validate_action_admission_record,
)
from .trajectory_contract_v1 import (
    CAUSAL_V2_FORMAT,
    CONTRACT_VERSION,
    Purpose,
    audit_h5,
    valid_metric_depth_contract,
)
from .v11_promotion_gate_v1 import V11_PROMOTION_GATE_FORMAT


REPORTED_ACTION_HISTORY_DATASET = "decision_previous_reported_executed_joint_delta_normalized"
REPORTED_ACTION_HISTORY_RAD_DATASET = "decision_previous_reported_executed_joint_delta_rad"
REPORTED_ACTION_HISTORY_VALID_DATASET = "decision_previous_reported_executed_action_valid"
REPORTED_ACTION_HISTORY_ALIGNMENT = (
    "decision_previous_reported_executed_joint_delta_* at row t equals "
    "effect_reported_executed_joint_delta_* at row t-1; row 0 is zero "
    "with valid=0; same-row effect_* is forbidden as policy input"
)
ROLLING_SHUTTER_GEOMETRY_MODEL_ID = "previous_current_row_blend_v1"
ROLLING_SHUTTER_ALPHA_FORMULA = "alpha(y)=1-f+f*y/(H-1)"

# This is the complete dataset allow-list for policy inputs.  Labels and masks
# are intentionally kept in separate constants below.  In particular, no
# simulator physical/object/contact/segmentation field and no same-row effect
# execution field appears here.
POLICY_INPUT_DATASETS = frozenset(
    {
        "rgb_wrist",
        "depth_wrist_mm",
        "decision_reported_joint_position",
        "decision_reported_joint_velocity",
        REPORTED_ACTION_HISTORY_DATASET,
        REPORTED_ACTION_HISTORY_RAD_DATASET,
        REPORTED_ACTION_HISTORY_VALID_DATASET,
        "camera_pose_wrist",
        "camera_pose_wrist_previous_endpoint",
        "camera_pose_wrist_current_endpoint",
        "camera_source_frame_index",
        "camera_delivered_frame_index",
        "camera_delivered_frame_age_steps",
        "camera_device_timestamp_ns",
        "camera_host_timestamp_ns",
        "camera_device_period_seconds",
        "camera_host_period_seconds",
        "camera_state_age_seconds",
        "camera_rolling_shutter_used_previous_frame",
        "camera_rolling_shutter_readout_fraction",
        "camera_rolling_shutter_dual_endpoint_complete",
        "camera_rolling_shutter_previous_endpoint_source_frame_index",
        "camera_rolling_shutter_current_endpoint_source_frame_index",
        "camera_rolling_shutter_row_start_control_time_seconds",
        "camera_rolling_shutter_readout_duration_control_seconds",
        "camera_4d_reconstructable_mask",
        "camera_geometry_alignment_exact",
        "camera_world_model_training_mask",
    }
)

# Source datasets used to construct the causal wrist RGB-D geometry window.
# Keeping this list explicit makes the 4D policy boundary reviewable: every
# field is already in the policy allow-list above, and no same-row execution,
# simulator state, object/contact state, segmentation, or teacher diagnostic is
# admitted into the visual history.
TEMPORAL_WRIST_INPUT_DATASETS = frozenset(
    {
        "rgb_wrist",
        "depth_wrist_mm",
        "camera_pose_wrist",
        "camera_source_frame_index",
        "camera_delivered_frame_index",
        "camera_device_timestamp_ns",
        "camera_host_timestamp_ns",
        "camera_rolling_shutter_used_previous_frame",
        "camera_4d_reconstructable_mask",
        "camera_geometry_alignment_exact",
        "camera_world_model_training_mask",
    }
)
if not TEMPORAL_WRIST_INPUT_DATASETS <= POLICY_INPUT_DATASETS:  # pragma: no cover
    raise RuntimeError("temporal wrist inputs escaped the policy dataset allow-list")

_LABEL_DATASET = GENERIC_ACTION_LABEL_DATASET
_TRAINING_MASK_DATASET = "effect_policy_intent_action_training_mask"
_TRAINING_MASK_ALIAS = "effect_action_training_mask"
_EXECUTION_UNMODIFIED_DATASET = "policy_intent_execution_unmodified_mask"
_UNMODIFIED_TRAINING_MASK_DATASET = "effect_unmodified_policy_execution_training_mask"
_WORLD_MASK_ALIAS = "effect_world_model_training_mask"
_LABEL_VALID_DATASET = "policy_intent_action_label_valid"
_EFFECT_REPORTED_POSITION_DATASET = "effect_reported_joint_position"
_EFFECT_REPORTED_DELTA_RAD_DATASET = "effect_reported_executed_joint_delta_rad"
_EFFECT_REPORTED_DELTA_NORMALIZED_DATASET = "effect_reported_executed_joint_delta_normalized"
_EFFECT_REPORTED_VALID_DATASET = "effect_reported_executed_action_measurement_valid"

_ROOT_SEMANTICS: Mapping[str, Any] = {
    "format": CAUSAL_V2_FORMAT,
    "trajectory_contract_version": CONTRACT_VERSION,
    "act_primary_label_dataset": _LABEL_DATASET,
    "act_action_history_dataset": REPORTED_ACTION_HISTORY_DATASET,
    "act_action_history_valid_dataset": REPORTED_ACTION_HISTORY_VALID_DATASET,
    "act_action_history_alignment": REPORTED_ACTION_HISTORY_ALIGNMENT,
    "act_runtime_shield_required": True,
    "pre_guard_target_is_physics_applied_truth": False,
    "episode_integrity_gate_format": EPISODE_INTEGRITY_GATE_FORMAT,
    "episode_integrity_gate_required": True,
    "action_admission_gate_format": ACTION_ADMISSION_GATE_FORMAT,
    "action_admission_gate_required": True,
    "action_admission_source_registry": action_source_registry_json(),
    "rolling_shutter_geometry_model_id": ROLLING_SHUTTER_GEOMETRY_MODEL_ID,
    "rolling_shutter_readout_direction": "top_to_bottom",
    "rolling_shutter_alpha_formula": ROLLING_SHUTTER_ALPHA_FORMULA,
    # This loader does not implement row-wise SE(3) reconstruction.  Mixed
    # frames must consequently remain masked out of world-model supervision.
    "world_model_loader_supports_rolling_shutter_geometry_model": False,
}

_EPISODE_SEMANTICS: Mapping[str, Any] = {
    "dataset_format": CAUSAL_V2_FORMAT,
    "trajectory_contract_version": CONTRACT_VERSION,
    "act_primary_label_dataset": _LABEL_DATASET,
    "act_action_history_dataset": REPORTED_ACTION_HISTORY_DATASET,
    "act_action_history_valid_dataset": REPORTED_ACTION_HISTORY_VALID_DATASET,
    "act_action_history_alignment": REPORTED_ACTION_HISTORY_ALIGNMENT,
    "act_runtime_shield_required": True,
    "episode_integrity_gate_format": EPISODE_INTEGRITY_GATE_FORMAT,
    "action_admission_gate_format": ACTION_ADMISSION_GATE_FORMAT,
    "camera_mount": "wrist",
}

_VECTOR6_DATASETS = (
    "decision_reported_joint_position",
    "decision_reported_joint_velocity",
    REPORTED_ACTION_HISTORY_DATASET,
    REPORTED_ACTION_HISTORY_RAD_DATASET,
    _LABEL_DATASET,
)
_POSE_DATASETS = (
    "camera_pose_wrist",
    "camera_pose_wrist_previous_endpoint",
    "camera_pose_wrist_current_endpoint",
)
_SCALAR_DATASETS = (
    "camera_source_frame_index",
    "camera_delivered_frame_index",
    "camera_delivered_frame_age_steps",
    "camera_device_timestamp_ns",
    "camera_host_timestamp_ns",
    "camera_device_period_seconds",
    "camera_host_period_seconds",
    "camera_state_age_seconds",
    "camera_rolling_shutter_used_previous_frame",
    "camera_rolling_shutter_readout_fraction",
    "camera_rolling_shutter_dual_endpoint_complete",
    "camera_rolling_shutter_previous_endpoint_source_frame_index",
    "camera_rolling_shutter_current_endpoint_source_frame_index",
    "camera_rolling_shutter_row_start_control_time_seconds",
    "camera_rolling_shutter_readout_duration_control_seconds",
    "camera_4d_reconstructable_mask",
    "camera_geometry_alignment_exact",
    "camera_world_model_training_mask",
    REPORTED_ACTION_HISTORY_VALID_DATASET,
    _TRAINING_MASK_DATASET,
    _TRAINING_MASK_ALIAS,
    _EXECUTION_UNMODIFIED_DATASET,
    _UNMODIFIED_TRAINING_MASK_DATASET,
    _WORLD_MASK_ALIAS,
    _LABEL_VALID_DATASET,
)
_BINARY_DATASETS = (
    "camera_rolling_shutter_used_previous_frame",
    "camera_rolling_shutter_dual_endpoint_complete",
    "camera_4d_reconstructable_mask",
    "camera_geometry_alignment_exact",
    "camera_world_model_training_mask",
    REPORTED_ACTION_HISTORY_VALID_DATASET,
    _TRAINING_MASK_DATASET,
    _TRAINING_MASK_ALIAS,
    _EXECUTION_UNMODIFIED_DATASET,
    _UNMODIFIED_TRAINING_MASK_DATASET,
    _WORLD_MASK_ALIAS,
    _LABEL_VALID_DATASET,
)


@dataclass(frozen=True)
class _EpisodeRef:
    episode_index: int
    shard_path: Path
    group_name: str
    frames: int
    camera_intrinsics: np.ndarray


def _scalar(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="strict")
    if hasattr(value, "item"):
        try:
            return value.item()
        except (AttributeError, ValueError):
            pass
    return value


def _bool(value: Any, *, name: str) -> bool:
    value = _scalar(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1"}:
            return True
        if normalized in {"false", "0"}:
            return False
        raise ValueError(f"{name} must be an explicit boolean")
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)) and int(value) in {0, 1}:
        return bool(value)
    raise ValueError(f"{name} must be an explicit boolean")


def _matches(value: Any, expected: Any, *, name: str) -> bool:
    if isinstance(expected, bool):
        return _bool(value, name=name) is expected
    return str(_scalar(value)) == str(expected)


def _json_value(value: Any, *, name: str) -> Any:
    value = _scalar(value)
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError as error:
            raise ValueError(f"{name} is not valid JSON") from error
    return value


def _require_semantics(
    attrs: h5py.AttributeManager,
    expected: Mapping[str, Any],
    *,
    scope: str,
) -> None:
    for name, expected_value in expected.items():
        if name not in attrs:
            raise ValueError(f"{scope} is missing required attribute: {name}")
        if not _matches(attrs[name], expected_value, name=f"{scope}.{name}"):
            raise ValueError(f"{scope} attribute mismatch: {name}")


def _read_manifest(path: Path) -> list[dict[str, Any]]:
    if not path.is_file() or path.stat().st_size == 0:
        raise ValueError(f"empty or missing Causal V2 manifest: {path}")
    entries: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid manifest JSON at line {line_number}") from error
        if not isinstance(entry, dict):
            raise ValueError(f"manifest line {line_number} is not an object")
        entries.append(entry)
    if not entries:
        raise ValueError(f"Causal V2 manifest contains no episodes: {path}")
    return entries


def _binary_array(group: h5py.Group, name: str, frames: int) -> np.ndarray:
    values = np.asarray(group[name][...])
    if values.shape != (frames,):
        raise ValueError(f"{group.name}/{name} must have shape [T]")
    if not np.all(np.isin(values, (0, 1))):
        raise ValueError(f"{group.name}/{name} must be binary")
    return values.astype(bool, copy=False)


def _finite_array(group: h5py.Group, name: str) -> np.ndarray:
    values = np.asarray(group[name][...])
    if not np.issubdtype(values.dtype, np.number) or not np.isfinite(values).all():
        raise ValueError(f"{group.name}/{name} must contain only finite numbers")
    return values


class CausalV2ACTDataset(Dataset):
    """Strict, lazy ACT dataset for one Causal V2 manifest split.

    ``index`` contains only admitted, current-row supervised samples.  Row zero
    is admissible with an empty history.  Action chunks never cross an episode
    boundary, and each future element retains its own authoritative
    policy-intent training mask.
    """

    def __init__(
        self,
        root: Path,
        split: str,
        *,
        action_chunk_size: int = 16,
        action_history_steps: int = 8,
        visual_history_steps: int = 1,
        max_depth_m: float = 5.0,
    ) -> None:
        if not isinstance(split, str) or not split or Path(split).name != split:
            raise ValueError("split must be one non-empty path component")
        for name, value in (
            ("action_chunk_size", action_chunk_size),
            ("action_history_steps", action_history_steps),
            ("visual_history_steps", visual_history_steps),
        ):
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if not np.isfinite(max_depth_m) or float(max_depth_m) <= 0.0:
            raise ValueError("max_depth_m must be finite and positive")

        self.root = Path(root).expanduser().resolve()
        self.split = split
        self.split_dir = (self.root / split).resolve()
        if self.split_dir.parent != self.root:
            raise ValueError("split escapes the Causal V2 root")
        self.action_chunk_size = int(action_chunk_size)
        self.action_history_steps = int(action_history_steps)
        self.visual_history_steps = int(visual_history_steps)
        self.max_depth_m = float(max_depth_m)
        self.episodes: list[_EpisodeRef] = []
        self.index: list[tuple[int, int]] = []
        self._files: dict[Path, h5py.File] = {}

        manifest_entries = _read_manifest(self.split_dir / "manifest.jsonl")
        audited_shards: set[Path] = set()
        seen_groups: set[tuple[Path, str]] = set()
        for manifest_position, entry in enumerate(manifest_entries):
            self._validate_manifest_entry(entry, manifest_position)
            shard_path = (self.split_dir / str(entry["shard"])).resolve()
            if shard_path.parent != self.split_dir or not shard_path.is_file():
                raise ValueError(f"manifest references an invalid shard: {shard_path}")
            group_name = str(entry["group"]).lstrip("/")
            identity = (shard_path, group_name)
            if identity in seen_groups:
                raise ValueError(f"duplicate Causal V2 episode reference: {identity}")
            seen_groups.add(identity)
            if shard_path not in audited_shards:
                self._require_schema_a(shard_path)
                audited_shards.add(shard_path)

            with h5py.File(shard_path, "r") as stream:
                _require_semantics(stream.attrs, _ROOT_SEMANTICS, scope=str(shard_path))
                if group_name not in stream or not isinstance(stream[group_name], h5py.Group):
                    raise ValueError(f"manifest group is missing: {shard_path}/{group_name}")
                group = stream[group_name]
                intrinsics, training_mask = self._validate_group(
                    stream,
                    group,
                    entry,
                )

            episode_ref = _EpisodeRef(
                episode_index=int(entry["episode_index"]),
                shard_path=shard_path,
                group_name=group_name,
                frames=int(entry["frames"]),
                camera_intrinsics=intrinsics,
            )
            episode_slot = len(self.episodes)
            self.episodes.append(episode_ref)
            # Row zero is a valid current observation/label with an explicitly
            # empty history.  ``previous_valid`` controls history attention; it
            # is not a current-label admission mask.
            eligible = training_mask
            self.index.extend((episode_slot, int(frame)) for frame in np.flatnonzero(eligible))
        if not self.index:
            raise ValueError("Causal V2 source contains no admitted ACT samples")

    @staticmethod
    def _validate_manifest_entry(entry: Mapping[str, Any], position: int) -> None:
        if int(entry.get("episode_index", -1)) != position:
            raise ValueError("Causal V2 manifest episode indices must be contiguous")
        expected = {
            "dataset_format": CAUSAL_V2_FORMAT,
            "camera_mount": "wrist",
            "camera_streams": ["wrist"],
        }
        for name, value in expected.items():
            if entry.get(name) != value:
                raise ValueError(f"Causal V2 manifest {name} mismatch at episode {position}")
        shard = str(entry.get("shard", ""))
        if (
            not shard
            or Path(shard).name != shard
            or Path(shard).suffix.lower()
            not in {
                ".h5",
                ".hdf5",
            }
        ):
            raise ValueError(f"invalid Causal V2 shard name at episode {position}")
        expected_group = f"/episode_{position:07d}"
        if str(entry.get("group", "")) != expected_group:
            raise ValueError(f"non-canonical Causal V2 group at episode {position}")
        if int(entry.get("frames", 0)) < 1:
            raise ValueError(f"invalid frame count at episode {position}")
        if entry.get("source_type") not in SUPPORTED_SOURCE_TYPES:
            raise ValueError(f"unsupported Causal V2 source at episode {position}")
        for name in ("episode_integrity_pass", "action_admission_pass"):
            _bool(entry.get(name), name=f"manifest[{position}].{name}")

    @staticmethod
    def _require_schema_a(path: Path) -> None:
        report = audit_h5(path)
        purpose = report["purposes"][Purpose.ACTION_SUPERVISION.value]
        if report.get("format") != CAUSAL_V2_FORMAT:
            raise ValueError(f"not a Causal V2 shard: {path}")
        if purpose.get("grade") != "A" or not purpose.get("schema_presence_eligible"):
            raise ValueError(f"Causal V2 action-supervision schema is not grade A: {path}")
        if not report.get("metadata_only", False):
            raise ValueError("trajectory contract audit unexpectedly changed semantics")

    def _validate_group(
        self,
        stream: h5py.File,
        group: h5py.Group,
        entry: Mapping[str, Any],
    ) -> tuple[np.ndarray, np.ndarray]:
        _require_semantics(group.attrs, _EPISODE_SEMANTICS, scope=group.name)
        frames = int(entry["frames"])
        if int(group.attrs.get("frames", frames)) != frames:
            raise ValueError(f"{group.name} frame metadata mismatch")
        if _json_value(group.attrs.get("camera_streams", "null"), name="camera_streams") != ["wrist"]:
            raise ValueError(f"{group.name} must contain only the wrist camera")
        root_source = str(_scalar(stream.attrs.get("source_type", "")))
        episode_source = str(_scalar(group.attrs.get("source_type", "")))
        manifest_source = str(entry.get("source_type", ""))
        if (
            root_source not in SUPPORTED_SOURCE_TYPES
            or episode_source != root_source
            or manifest_source != root_source
        ):
            raise ValueError(f"{group.name} source_type registry mismatch")

        root_scale = float(_scalar(stream.attrs.get("max_joint_delta_rad", np.nan)))
        episode_scale = float(_scalar(group.attrs.get("max_joint_delta_rad", np.nan)))
        if (
            not np.isfinite(root_scale)
            or root_scale <= 0.0
            or not np.isfinite(episode_scale)
            or not np.isclose(root_scale, episode_scale, rtol=0.0, atol=0.0)
        ):
            raise ValueError(f"{group.name} max_joint_delta_rad contract mismatch")

        required = set(POLICY_INPUT_DATASETS) | {
            _LABEL_DATASET,
            _TRAINING_MASK_DATASET,
            _TRAINING_MASK_ALIAS,
            _EXECUTION_UNMODIFIED_DATASET,
            _UNMODIFIED_TRAINING_MASK_DATASET,
            _WORLD_MASK_ALIAS,
            _LABEL_VALID_DATASET,
            _EFFECT_REPORTED_POSITION_DATASET,
            _EFFECT_REPORTED_DELTA_RAD_DATASET,
            _EFFECT_REPORTED_DELTA_NORMALIZED_DATASET,
            _EFFECT_REPORTED_VALID_DATASET,
        }
        if root_source == V11_SOURCE_TYPE:
            required.add(V11_TEACHER_DIAGNOSTIC_ALIAS)
        elif (
            V11_TEACHER_DIAGNOSTIC_ALIAS in group
            or "promotion_gate_format" in stream.attrs
            or "promotion_gate_format" in group.attrs
            or "promotion_gate_evidence" in group.attrs
            or "promotion_valid_success" in group.attrs
        ):
            raise ValueError(f"{group.name} scratch source contains V11-only semantics")
        missing = sorted(required - set(group))
        if missing:
            raise ValueError(f"{group.name} is missing ACT payload fields: {missing}")
        if any(len(group[name]) != frames for name in required):
            raise ValueError(f"{group.name} ACT payload fields have inconsistent lengths")

        rgb = group["rgb_wrist"]
        depth = group["depth_wrist_mm"]
        if rgb.dtype != np.dtype(np.uint8) or rgb.ndim != 4 or rgb.shape[-1] != 3:
            raise ValueError(f"{group.name}/rgb_wrist must be uint8 [T,H,W,3]")
        if depth.dtype != np.dtype(np.uint16) or depth.shape != rgb.shape[:-1]:
            raise ValueError(f"{group.name}/depth_wrist_mm must be uint16 [T,H,W]")
        height, width = int(rgb.shape[1]), int(rgb.shape[2])
        if height < 1 or width < 1:
            raise ValueError(f"{group.name} wrist image resolution must be positive")

        for name in _VECTOR6_DATASETS:
            dataset = group[name]
            if dataset.dtype != np.dtype(np.float32) or dataset.shape != (frames, 6):
                raise ValueError(f"{group.name}/{name} must be float32 [T,6]")
            _finite_array(group, name)
        for name in (
            _EFFECT_REPORTED_POSITION_DATASET,
            _EFFECT_REPORTED_DELTA_RAD_DATASET,
            _EFFECT_REPORTED_DELTA_NORMALIZED_DATASET,
        ):
            dataset = group[name]
            if dataset.dtype != np.dtype(np.float32) or dataset.shape != (frames, 6):
                raise ValueError(f"{group.name}/{name} must be float32 [T,6]")
            _finite_array(group, name)
        for name in _POSE_DATASETS:
            dataset = group[name]
            if dataset.dtype != np.dtype(np.float32) or dataset.shape != (frames, 12):
                raise ValueError(f"{group.name}/{name} must be float32 [T,12]")
            _finite_array(group, name)
        for name in _SCALAR_DATASETS:
            if group[name].shape != (frames,):
                raise ValueError(f"{group.name}/{name} must have shape [T]")
        for name in _BINARY_DATASETS:
            _binary_array(group, name, frames)
        effect_reported_valid = _binary_array(group, _EFFECT_REPORTED_VALID_DATASET, frames)

        for name in (
            "camera_device_period_seconds",
            "camera_host_period_seconds",
            "camera_state_age_seconds",
            "camera_rolling_shutter_readout_fraction",
            "camera_rolling_shutter_row_start_control_time_seconds",
            "camera_rolling_shutter_readout_duration_control_seconds",
        ):
            _finite_array(group, name)
        device_period = np.asarray(group["camera_device_period_seconds"][...], dtype=np.float64)
        host_period = np.asarray(group["camera_host_period_seconds"][...], dtype=np.float64)
        readout_fraction = np.asarray(group["camera_rolling_shutter_readout_fraction"][...], dtype=np.float64)
        readout_duration = np.asarray(
            group["camera_rolling_shutter_readout_duration_control_seconds"][...],
            dtype=np.float64,
        )
        if np.any(device_period <= 0.0) or np.any(host_period <= 0.0):
            raise ValueError(f"{group.name} camera periods must be positive")
        if np.any((readout_fraction < 0.0) | (readout_fraction > 1.0)):
            raise ValueError(f"{group.name} rolling-shutter fraction must be in [0,1]")
        if np.any(readout_duration < 0.0):
            raise ValueError(f"{group.name} rolling-shutter duration must be non-negative")

        # A temporal geometry model cannot safely reorder camera observations.
        # Repeated IDs/timestamps are valid (for example, a delivered frame may
        # be held across transport latency), but reversal is always rejected.
        integer_timeline_datasets = (
            "camera_source_frame_index",
            "camera_delivered_frame_index",
            "camera_device_timestamp_ns",
            "camera_host_timestamp_ns",
        )
        timeline: dict[str, np.ndarray] = {}
        for name in integer_timeline_datasets:
            dataset = group[name]
            if not np.issubdtype(dataset.dtype, np.integer):
                raise ValueError(f"{group.name}/{name} must have an integer dtype")
            values = np.asarray(dataset[...], dtype=np.int64)
            if np.any(values < 0):
                raise ValueError(f"{group.name}/{name} must be non-negative")
            if values.size > 1 and np.any(values[1:] < values[:-1]):
                raise ValueError(f"{group.name}/{name} must be monotonically nondecreasing")
            timeline[name] = values
        if np.any(timeline["camera_delivered_frame_index"] > timeline["camera_source_frame_index"]):
            raise ValueError(f"{group.name} delivered camera frame is not causal")

        previous_valid = _binary_array(group, REPORTED_ACTION_HISTORY_VALID_DATASET, frames)
        previous_normalized = np.asarray(group[REPORTED_ACTION_HISTORY_DATASET][...])
        previous_rad = np.asarray(group[REPORTED_ACTION_HISTORY_RAD_DATASET][...])
        if previous_valid[0] or np.any(previous_normalized[0]) or np.any(previous_rad[0]):
            raise ValueError(f"{group.name} row 0 reported action history must be zero/invalid")
        if not np.allclose(
            previous_rad / root_scale,
            previous_normalized,
            rtol=1.0e-5,
            atol=1.0e-6,
        ):
            raise ValueError(f"{group.name} reported action history normalization mismatch")

        decision_reported_position = np.asarray(
            group["decision_reported_joint_position"][...], dtype=np.float32
        )
        effect_reported_position = np.asarray(group[_EFFECT_REPORTED_POSITION_DATASET][...], dtype=np.float32)
        effect_reported_rad = np.asarray(group[_EFFECT_REPORTED_DELTA_RAD_DATASET][...], dtype=np.float32)
        effect_reported_normalized = np.asarray(
            group[_EFFECT_REPORTED_DELTA_NORMALIZED_DATASET][...], dtype=np.float32
        )
        recomputed_effect_rad = effect_reported_position - decision_reported_position
        if not np.allclose(
            recomputed_effect_rad,
            effect_reported_rad,
            rtol=1.0e-5,
            atol=1.0e-6,
        ):
            raise ValueError(f"{group.name} reported effect delta does not match qpos transition")
        if not np.allclose(
            effect_reported_rad / root_scale,
            effect_reported_normalized,
            rtol=1.0e-5,
            atol=1.0e-6,
        ):
            raise ValueError(f"{group.name} reported effect normalization mismatch")
        expected_previous_rad = np.zeros_like(previous_rad)
        expected_previous_normalized = np.zeros_like(previous_normalized)
        expected_previous_valid = np.zeros_like(previous_valid)
        if frames > 1:
            expected_previous_rad[1:] = effect_reported_rad[:-1]
            expected_previous_normalized[1:] = effect_reported_normalized[:-1]
            expected_previous_valid[1:] = effect_reported_valid[:-1]
        if (
            not np.allclose(
                previous_rad,
                expected_previous_rad,
                rtol=1.0e-5,
                atol=1.0e-6,
            )
            or not np.allclose(
                previous_normalized,
                expected_previous_normalized,
                rtol=1.0e-5,
                atol=1.0e-6,
            )
            or not np.array_equal(previous_valid, expected_previous_valid)
        ):
            raise ValueError(f"{group.name} reported action history is not a one-row shift")

        policy_safe_action = np.asarray(group[_LABEL_DATASET][...], dtype=np.float32)
        if not np.isfinite(policy_safe_action).all():
            raise ValueError(f"{group.name} decision_safe_action must be finite")
        if np.any(np.abs(policy_safe_action) > 1.00001):
            raise ValueError(f"{group.name} decision_safe_action exceeds normalized bounds")

        integrity_pass = _bool(
            group.attrs.get("episode_integrity_pass"),
            name=f"{group.name}.episode_integrity_pass",
        )
        manifest_integrity = _bool(
            entry.get("episode_integrity_pass"),
            name=f"manifest[{entry['episode_index']}].episode_integrity_pass",
        )
        if not integrity_pass or manifest_integrity is not integrity_pass:
            raise ValueError(f"{group.name} episode integrity gate mismatch")
        strict_success = _bool(group.attrs.get("strict_success"), name=f"{group.name}.strict_success")
        manifest_success = _bool(
            entry.get("strict_success"),
            name=f"manifest[{entry['episode_index']}].strict_success",
        )
        if strict_success != manifest_success:
            raise ValueError(f"{group.name} strict-success mismatch")

        v11_promotion: dict[str, Any] | None = None
        if root_source == V11_SOURCE_TYPE:
            teacher_dataset = group[V11_TEACHER_DIAGNOSTIC_ALIAS]
            if teacher_dataset.dtype != np.dtype(np.float32) or teacher_dataset.shape != (frames, 6):
                raise ValueError(f"{group.name}/{V11_TEACHER_DIAGNOSTIC_ALIAS} must be float32 [T,6]")
            v11_promotion = _json_value(
                group.attrs.get("promotion_gate_evidence", "null"),
                name=f"{group.name}.promotion_gate_evidence",
            )
            if not isinstance(v11_promotion, dict):
                raise ValueError(f"{group.name} promotion_gate_evidence must be an object")
            root_profile = str(_scalar(stream.attrs.get("promotion_gate_profile_hash", "")))
            if (
                v11_promotion.get("format") != V11_PROMOTION_GATE_FORMAT
                or v11_promotion.get("profile_hash") != root_profile
                or not root_profile
            ):
                raise ValueError(f"{group.name} promotion profile mismatch")
            teacher_alias = np.asarray(group[V11_TEACHER_DIAGNOSTIC_ALIAS][...], dtype=np.float32)
            if not np.array_equal(teacher_alias, policy_safe_action):
                raise ValueError(f"{group.name} V11 teacher diagnostic alias mismatch")

        admission_evidence = _json_value(
            group.attrs.get("action_admission_gate_evidence", "null"),
            name=f"{group.name}.action_admission_gate_evidence",
        )
        admission_reasons = _json_value(
            group.attrs.get("action_admission_reason_codes", "null"),
            name=f"{group.name}.action_admission_reason_codes",
        )
        admission_pass = _bool(
            group.attrs.get("action_admission_pass"),
            name=f"{group.name}.action_admission_pass",
        )
        admission_record = {
            "format": str(_scalar(group.attrs.get("action_admission_gate_format", ""))),
            "passed": admission_pass,
            "evidence": admission_evidence,
            "reason_codes": admission_reasons,
        }
        admission_failures = validate_action_admission_record(
            admission_record,
            source_type=root_source,
            episode_integrity_pass=integrity_pass,
            strict_success=strict_success,
            v11_promotion=v11_promotion,
        )
        if admission_failures:
            raise ValueError(f"{group.name} action admission invalid: {list(admission_failures)}")
        manifest_admission = _bool(
            entry.get("action_admission_pass"),
            name=f"manifest[{entry['episode_index']}].action_admission_pass",
        )
        action_eligible = _bool(
            group.attrs.get("action_label_eligible"),
            name=f"{group.name}.action_label_eligible",
        )
        manifest_eligible = _bool(
            entry.get("action_label_eligible"),
            name=f"manifest[{entry['episode_index']}].action_label_eligible",
        )
        if len({admission_pass, manifest_admission, action_eligible, manifest_eligible}) != 1:
            raise ValueError(f"{group.name} action admission/eligibility mismatch")
        if root_source == V11_SOURCE_TYPE:
            promotion_pass = _bool(
                group.attrs.get("promotion_valid_success"),
                name=f"{group.name}.promotion_valid_success",
            )
            manifest_promotion = _bool(
                entry.get("promotion_valid_success"),
                name=f"manifest[{entry['episode_index']}].promotion_valid_success",
            )
            if promotion_pass is not admission_pass or manifest_promotion is not admission_pass:
                raise ValueError(f"{group.name} V11 promotion/action admission mismatch")

        training_mask = _binary_array(group, _TRAINING_MASK_DATASET, frames)
        training_alias = _binary_array(group, _TRAINING_MASK_ALIAS, frames)
        execution_unmodified = _binary_array(group, _EXECUTION_UNMODIFIED_DATASET, frames)
        unmodified_training = _binary_array(group, _UNMODIFIED_TRAINING_MASK_DATASET, frames)
        label_valid = _binary_array(group, _LABEL_VALID_DATASET, frames)
        if not np.array_equal(training_mask, training_alias):
            raise ValueError(f"{group.name} authoritative action masks disagree")
        expected_training = label_valid if admission_pass else np.zeros(frames, dtype=bool)
        if not np.array_equal(training_mask, expected_training):
            raise ValueError(f"{group.name} action mask violates admission semantics")
        expected_unmodified = execution_unmodified if admission_pass else np.zeros(frames, dtype=bool)
        if not np.array_equal(unmodified_training, expected_unmodified):
            raise ValueError(f"{group.name} unmodified-action mask violates admission semantics")
        if "action_training_eligible_rows" in group.attrs and int(
            group.attrs["action_training_eligible_rows"]
        ) != int(np.count_nonzero(training_mask)):
            raise ValueError(f"{group.name} action-training row count mismatch")

        rolling = _binary_array(group, "camera_rolling_shutter_used_previous_frame", frames)
        dual = _binary_array(group, "camera_rolling_shutter_dual_endpoint_complete", frames)
        reconstructable = _binary_array(group, "camera_4d_reconstructable_mask", frames)
        geometry_exact = _binary_array(group, "camera_geometry_alignment_exact", frames)
        camera_world = _binary_array(group, "camera_world_model_training_mask", frames)
        effect_world = _binary_array(group, _WORLD_MASK_ALIAS, frames)
        expected_reconstructable = (~rolling) | dual
        expected_geometry = ~rolling
        if not np.array_equal(reconstructable, expected_reconstructable):
            raise ValueError(f"{group.name} 4D reconstructable mask mismatch")
        if not np.array_equal(geometry_exact, expected_geometry):
            raise ValueError(f"{group.name} camera geometry mask mismatch")
        if not np.array_equal(camera_world, effect_world):
            raise ValueError(f"{group.name} world-model masks disagree")
        if np.any(camera_world & rolling) or not np.array_equal(camera_world, geometry_exact):
            raise ValueError(f"{group.name} rolling-shutter frame enabled without loader geometry support")

        intrinsics = self._camera_intrinsics(group, height=height, width=width)
        return intrinsics, training_mask

    @staticmethod
    def _camera_intrinsics(group: h5py.Group, *, height: int, width: int) -> np.ndarray:
        calibration = _json_value(
            group.attrs.get("camera_calibration", "null"),
            name=f"{group.name}.camera_calibration",
        )
        try:
            wrist = calibration["wrist"]
            intrinsics = np.asarray(wrist["intrinsics"], dtype=np.float32)
            calibration_width = int(wrist["width"])
            calibration_height = int(wrist["height"])
            metric_depth_contract = wrist["metric_depth_contract"]
        except (KeyError, TypeError, ValueError, IndexError) as error:
            raise ValueError(f"{group.name} wrist calibration is incomplete") from error
        if (
            intrinsics.shape != (3, 3)
            or not np.isfinite(intrinsics).all()
            or intrinsics[0, 0] <= 0.0
            or intrinsics[1, 1] <= 0.0
            or not np.allclose(intrinsics[2], (0.0, 0.0, 1.0), atol=1.0e-6)
            or calibration_width != width
            or calibration_height != height
            or not valid_metric_depth_contract(metric_depth_contract)
        ):
            raise ValueError(f"{group.name} wrist calibration does not match payload")
        return intrinsics.copy()

    def __len__(self) -> int:
        return len(self.index)

    def _file(self, path: Path) -> h5py.File:
        stream = self._files.get(path)
        if stream is None:
            stream = h5py.File(path, "r", swmr=True)
            self._files[path] = stream
        return stream

    def _group(self, episode_slot: int) -> h5py.Group:
        episode = self.episodes[episode_slot]
        return self._file(episode.shard_path)[episode.group_name]

    def history_indices(self, frame: int) -> np.ndarray:
        start = max(0, int(frame) - self.action_history_steps + 1)
        return np.arange(start, int(frame) + 1, dtype=np.int64)

    def visual_history_indices(self, frame: int) -> np.ndarray:
        """Return only current/past rows from the current episode."""

        start = max(0, int(frame) - self.visual_history_steps + 1)
        return np.arange(start, int(frame) + 1, dtype=np.int64)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        episode_slot, frame = self.index[index]
        episode = self.episodes[episode_slot]
        group = self._group(episode_slot)

        rgb = np.asarray(group["rgb_wrist"][frame], dtype=np.uint8).copy()
        depth_m_raw = np.asarray(group["depth_wrist_mm"][frame], dtype=np.float32) / 1000.0
        depth_valid = np.isfinite(depth_m_raw) & (depth_m_raw > 0.0) & (depth_m_raw <= self.max_depth_m)
        depth_m = np.where(depth_valid, depth_m_raw, 0.0).astype(np.float32)

        visual_rows = self.visual_history_indices(frame)
        visual_count = len(visual_rows)
        visual_offset = self.visual_history_steps - visual_count
        image_shape = tuple(int(value) for value in group["rgb_wrist"].shape[1:])
        depth_shape = tuple(int(value) for value in group["depth_wrist_mm"].shape[1:])
        rgb_window = np.zeros((self.visual_history_steps, *image_shape), dtype=np.uint8)
        depth_window = np.zeros((self.visual_history_steps, *depth_shape), dtype=np.float32)
        depth_valid_window = np.zeros((self.visual_history_steps, *depth_shape), dtype=bool)
        camera_pose_window = np.zeros((self.visual_history_steps, 12), dtype=np.float32)
        visual_history_mask = np.zeros(self.visual_history_steps, dtype=bool)
        source_frame_window = np.zeros(self.visual_history_steps, dtype=np.int64)
        delivered_frame_window = np.zeros(self.visual_history_steps, dtype=np.int64)
        device_timestamp_window = np.zeros(self.visual_history_steps, dtype=np.int64)
        host_timestamp_window = np.zeros(self.visual_history_steps, dtype=np.int64)
        device_time_delta_window = np.zeros(self.visual_history_steps, dtype=np.float64)
        host_time_delta_window = np.zeros(self.visual_history_steps, dtype=np.float64)
        reconstructable_window = np.zeros(self.visual_history_steps, dtype=bool)
        geometry_window = np.zeros(self.visual_history_steps, dtype=bool)
        world_window = np.zeros(self.visual_history_steps, dtype=bool)

        rgb_window[visual_offset:] = np.asarray(group["rgb_wrist"][visual_rows], dtype=np.uint8)
        visual_depth_raw = np.asarray(group["depth_wrist_mm"][visual_rows], dtype=np.float32) / 1000.0
        visual_depth_valid = (
            np.isfinite(visual_depth_raw) & (visual_depth_raw > 0.0) & (visual_depth_raw <= self.max_depth_m)
        )
        depth_valid_window[visual_offset:] = visual_depth_valid
        depth_window[visual_offset:] = np.where(
            visual_depth_valid,
            visual_depth_raw,
            0.0,
        ).astype(np.float32)
        camera_pose_window[visual_offset:] = np.asarray(
            group["camera_pose_wrist"][visual_rows], dtype=np.float32
        )
        visual_history_mask[visual_offset:] = True

        source_values = np.asarray(group["camera_source_frame_index"][visual_rows], dtype=np.int64)
        delivered_values = np.asarray(group["camera_delivered_frame_index"][visual_rows], dtype=np.int64)
        device_values = np.asarray(group["camera_device_timestamp_ns"][visual_rows], dtype=np.int64)
        host_values = np.asarray(group["camera_host_timestamp_ns"][visual_rows], dtype=np.int64)
        source_frame_window[visual_offset:] = source_values
        delivered_frame_window[visual_offset:] = delivered_values
        device_timestamp_window[visual_offset:] = device_values
        host_timestamp_window[visual_offset:] = host_values
        device_time_delta_window[visual_offset:] = (
            device_values.astype(np.float64) - float(device_values[-1])
        ) * 1.0e-9
        host_time_delta_window[visual_offset:] = (
            host_values.astype(np.float64) - float(host_values[-1])
        ) * 1.0e-9

        rolling_values = np.asarray(
            group["camera_rolling_shutter_used_previous_frame"][visual_rows],
            dtype=bool,
        )
        geometry_supported = ~rolling_values
        # The root contract explicitly states that this loader has no row-wise
        # rolling-shutter SE(3) model.  Such RGB frames remain usable by ACT,
        # while all geometry/4D/world supervision for them is masked out.
        reconstructable_window[visual_offset:] = (
            np.asarray(group["camera_4d_reconstructable_mask"][visual_rows], dtype=bool) & geometry_supported
        )
        geometry_window[visual_offset:] = (
            np.asarray(group["camera_geometry_alignment_exact"][visual_rows], dtype=bool) & geometry_supported
        )
        world_window[visual_offset:] = (
            np.asarray(group["camera_world_model_training_mask"][visual_rows], dtype=bool)
            & geometry_supported
        )

        position = np.asarray(group["decision_reported_joint_position"][frame], dtype=np.float32).copy()
        velocity = np.asarray(group["decision_reported_joint_velocity"][frame], dtype=np.float32).copy()

        history_rows = self.history_indices(frame)
        history_count = len(history_rows)
        history_offset = self.action_history_steps - history_count
        history = np.zeros((self.action_history_steps, 6), dtype=np.float32)
        history_rad = np.zeros_like(history)
        history_mask = np.zeros(self.action_history_steps, dtype=bool)
        history_frame_indices = np.full(self.action_history_steps, -1, dtype=np.int64)
        history[history_offset:] = np.asarray(
            group[REPORTED_ACTION_HISTORY_DATASET][history_rows], dtype=np.float32
        )
        history_rad[history_offset:] = np.asarray(
            group[REPORTED_ACTION_HISTORY_RAD_DATASET][history_rows], dtype=np.float32
        )
        history_mask[history_offset:] = np.asarray(
            group[REPORTED_ACTION_HISTORY_VALID_DATASET][history_rows], dtype=bool
        )
        history_frame_indices[history_offset:] = history_rows

        action_chunk = np.zeros((self.action_chunk_size, 6), dtype=np.float32)
        action_mask = np.zeros(self.action_chunk_size, dtype=bool)
        chunk_stop = min(episode.frames, frame + self.action_chunk_size)
        chunk_count = chunk_stop - frame
        action_chunk[:chunk_count] = np.asarray(group[_LABEL_DATASET][frame:chunk_stop], dtype=np.float32)
        # Each future position is masked by its own row, never by the current
        # sample's eligibility and never by an episode-level aggregate.
        action_mask[:chunk_count] = np.asarray(group[_TRAINING_MASK_DATASET][frame:chunk_stop], dtype=bool)

        def scalar_tensor(name: str, dtype: torch.dtype) -> torch.Tensor:
            return torch.tensor(_scalar(group[name][frame]), dtype=dtype)

        camera_clock = np.asarray(
            [
                float(group["camera_device_period_seconds"][frame]),
                float(group["camera_host_period_seconds"][frame]),
                float(group["camera_state_age_seconds"][frame]),
                float(group["camera_rolling_shutter_row_start_control_time_seconds"][frame]),
                float(group["camera_rolling_shutter_readout_duration_control_seconds"][frame]),
            ],
            dtype=np.float64,
        )
        return {
            "rgb_wrist": torch.from_numpy(rgb),
            "depth_wrist_m": torch.from_numpy(depth_m.copy()),
            "depth_valid_mask": torch.from_numpy(depth_valid.copy()),
            "rgb_wrist_window": torch.from_numpy(rgb_window),
            "depth_wrist_m_window": torch.from_numpy(depth_window),
            "depth_valid_mask_window": torch.from_numpy(depth_valid_window),
            "camera_pose_wrist_window": torch.from_numpy(camera_pose_window),
            "visual_history_mask": torch.from_numpy(visual_history_mask),
            "camera_source_frame_index_window": torch.from_numpy(source_frame_window),
            "camera_delivered_frame_index_window": torch.from_numpy(delivered_frame_window),
            "camera_device_timestamp_ns_window": torch.from_numpy(device_timestamp_window),
            "camera_host_timestamp_ns_window": torch.from_numpy(host_timestamp_window),
            "camera_device_time_delta_s_window": torch.from_numpy(device_time_delta_window),
            "camera_host_time_delta_s_window": torch.from_numpy(host_time_delta_window),
            "camera_4d_reconstructable_mask_window": torch.from_numpy(reconstructable_window),
            "camera_geometry_alignment_exact_window": torch.from_numpy(geometry_window),
            "world_model_training_mask_window": torch.from_numpy(world_window),
            "decision_reported_joint_position": torch.from_numpy(position),
            "decision_reported_joint_velocity": torch.from_numpy(velocity),
            "robot_state": torch.from_numpy(np.concatenate((position, velocity))),
            "action_history": torch.from_numpy(history),
            "action_history_rad": torch.from_numpy(history_rad),
            "action_history_mask": torch.from_numpy(history_mask),
            "action_history_frame_indices": torch.from_numpy(history_frame_indices),
            "camera_pose_wrist": torch.from_numpy(
                np.asarray(group["camera_pose_wrist"][frame], dtype=np.float32).copy()
            ),
            "camera_pose_wrist_previous_endpoint": torch.from_numpy(
                np.asarray(group["camera_pose_wrist_previous_endpoint"][frame], dtype=np.float32).copy()
            ),
            "camera_pose_wrist_current_endpoint": torch.from_numpy(
                np.asarray(group["camera_pose_wrist_current_endpoint"][frame], dtype=np.float32).copy()
            ),
            "camera_intrinsics": torch.from_numpy(episode.camera_intrinsics.copy()),
            "camera_clock": torch.from_numpy(camera_clock),
            "camera_source_frame_index": scalar_tensor("camera_source_frame_index", torch.int64),
            "camera_delivered_frame_index": scalar_tensor("camera_delivered_frame_index", torch.int64),
            "camera_delivered_frame_age_steps": scalar_tensor(
                "camera_delivered_frame_age_steps", torch.int64
            ),
            "camera_device_timestamp_ns": scalar_tensor("camera_device_timestamp_ns", torch.int64),
            "camera_host_timestamp_ns": scalar_tensor("camera_host_timestamp_ns", torch.int64),
            "camera_rolling_shutter_used_previous_frame": scalar_tensor(
                "camera_rolling_shutter_used_previous_frame", torch.bool
            ),
            "camera_rolling_shutter_readout_fraction": scalar_tensor(
                "camera_rolling_shutter_readout_fraction", torch.float32
            ),
            "camera_rolling_shutter_dual_endpoint_complete": scalar_tensor(
                "camera_rolling_shutter_dual_endpoint_complete", torch.bool
            ),
            "camera_4d_reconstructable_mask": scalar_tensor("camera_4d_reconstructable_mask", torch.bool),
            "camera_geometry_alignment_exact": scalar_tensor("camera_geometry_alignment_exact", torch.bool),
            "world_model_training_mask": scalar_tensor("camera_world_model_training_mask", torch.bool),
            "policy_safe_action": torch.from_numpy(action_chunk[0].copy()),
            "action_chunk": torch.from_numpy(action_chunk),
            "action_chunk_mask": torch.from_numpy(action_mask),
            "episode_index": torch.tensor(episode.episode_index, dtype=torch.int64),
            "frame_index": torch.tensor(frame, dtype=torch.int64),
        }

    def close(self) -> None:
        for stream in self._files.values():
            try:
                stream.close()
            except (TypeError, ValueError):
                pass
        self._files.clear()

    def __getstate__(self) -> dict[str, Any]:
        self.close()
        state = dict(self.__dict__)
        state["_files"] = {}
        return state

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
