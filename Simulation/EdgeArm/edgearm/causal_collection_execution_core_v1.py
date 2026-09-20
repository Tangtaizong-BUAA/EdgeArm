"""Source-neutral interfaces for one Causal V2 episode execution loop.

The existing V11 collector currently mixes three responsibilities: executing
the MuJoCo plant, asking a command source for an action, and writing
source-specific provenance.  A scratch-PPO collector must not copy that loop or
import the V11 expert.  This module defines the narrow boundary needed to move
both command sources behind one execution core without changing the current
V11 behavior while the refactor is staged.

The execution core owns reset/render/capture, command submission, ``env.step``,
command feedback, runtime-guard lineage, source-neutral effect geometry,
execution phase, and action history.  An adapter owns only its policy decision,
source-specific row evidence, immutable producer artifacts, and episode-final
admission evidence.  The interfaces perform no collection, MuJoCo stepping,
HDF5 writes, checkpoint loads, or artifact publication themselves.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import re
from types import MappingProxyType
from typing import Any, Protocol, runtime_checkable

import numpy as np


CAUSAL_COLLECTION_EXECUTION_CORE_INTERFACE_FORMAT = "edgearm-causal-collection-execution-core-interface-v1"
SOURCE_SCHEMA_PROFILE_FORMAT = "edgearm-causal-v2-source-schema-profile-v1"
SOURCE_ADAPTER_INTERFACE_FORMAT = "edgearm-causal-v2-source-adapter-interface-v1"
H5_ROW_DATASET_CONTRACT_FORMAT = "edgearm-causal-v2-h5-row-dataset-contract-v1"
SOURCE_PREPARATION_PROFILE_COMPONENT_FIELDS = (
    "source_schema_profile_format",
    "source_schema_profile_sha256",
    "source_type",
    "controller_identity",
    "controller_version",
)
SOURCE_SCHEMA_PROVENANCE_FIELDS = (
    "source_schema_profile_format",
    "source_schema_profile_sha256",
    "source_preparation_binding_sha256",
)

EXECUTION_CORE_OWNED_STAGES = (
    "reset_and_episode_randomization",
    "wrist_rgbd_capture_and_temporal_alignment",
    "prequeue_safe_action_dispatch",
    "environment_step",
    "command_and_runtime_guard_lineage",
    "source_neutral_effect_geometry",
    "execution_phase_tracking",
    "decision_aligned_action_and_feedback_history",
    "common_payload_and_metadata",
)
SOURCE_ADAPTER_OWNED_STAGES = (
    "immutable_producer_artifact_preparation",
    "policy_decision",
    "source_specific_decision_evidence",
    "source_specific_post_effect_evidence",
    "episode_action_admission",
    "source_specific_sidecar_publication",
    "immutable_producer_artifact_revalidation",
)

# Exact source-neutral evidence dictionaries crossing the shared execution
# core -> source-adapter boundary.  Both V11 and scratch-PPO receive the same
# complete dictionaries even when one source consumes only a subset.  Keeping
# these finite tuples in the neutral interface prevents a source adapter from
# silently accepting a weaker collector contract or an unreviewed field.
COMMON_EFFECT_EVIDENCE_FIELDS_V1 = (
    "submitted_command_id",
    "applied_command_id",
    "effect_step",
    "effect_execution_phase_name",
    "effect_execution_phase_valid",
    "effect_valid_push_side_contact_any",
    "effect_push_directional_block_displacement_m",
    "effect_progress_toward_target_m",
    "effect_strict_success",
    "runtime_guard_safety_stop",
    "raw_tool_block_contact_any",
)
COMMON_FINALIZATION_METADATA_FIELDS_V1 = (
    "strict_success",
    "final_terminated",
    "final_truncated",
    "collision_failure_any",
    "tool_block_contact_steps",
    "valid_push_side_contact_steps",
    "episode_net_block_displacement_m",
    "minimum_pusher_desk_signed_distance_m",
    "pusher_desk_hard_floor_m",
    "runtime_guard_static_infeasible_events",
    "runtime_guard_dynamic_infeasible_events",
    "runtime_guard_safety_stop",
    "invalid_geometry_contact_steps",
    "forbidden_pusher_desk_penetration_substeps",
    "forbidden_non_tool_robot_desk_penetration_any",
    "episode_integrity_pass",
)

_JOINTS = 6
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_H5_DTYPES = frozenset({"float32", "float64", "int16", "int32", "int64", "uint8", "uint32", "uint64"})

# These are source-neutral storage fields emitted by the shared execution
# pipeline but not consumed as policy inputs and not required for every future
# collector.  The tuple is intentionally finite and explicit: there is no
# prefix, regex, or substring wildcard.  A new stored dataset must therefore be
# reviewed and added here (or to a source row contract) before it can cross the
# HDF5 writer boundary.
EXECUTION_CORE_OPTIONAL_COMMON_DATASETS_V1 = (
    "action_label_valid",
    "camera_configured_latency_steps",
    "camera_delivered_frame_repeated",
    "camera_delivered_occluded",
    "camera_delivered_target_visible",
    "camera_frame_dropped",
    "camera_transport_candidate_frame_index",
    "command_applied_delayed_action",
    "command_applied_parent_decision_step",
    "command_applied_parent_observation_frame_index",
    "command_control_mode_id",
    "command_hold_duration_seconds",
    "command_hold_duration_steps",
    "command_lost",
    "command_parent_decision_step",
    "command_parent_observation_frame_index",
    "command_requested_joint_target",
    "command_safety_stage_mask",
    "command_target_changed_mask",
    "decision_block_pose_xyz_wxyz",
    "decision_block_twist_linear_angular",
    "decision_control_step",
    "decision_control_time_seconds",
    "decision_obstacle_xy",
    "decision_physical_joint_acceleration",
    "decision_physical_joint_velocity",
    "decision_target_xy",
    "decision_tool_pose",
    "decision_tool_twist_linear_angular",
    "effect_block_angular_speed_rad_s",
    "effect_block_linear_speed_m_s",
    "effect_block_pose_xyz_wxyz",
    "effect_block_twist_linear_angular",
    "effect_causal_transition_training_mask",
    "effect_collision_failure",
    "effect_contact_count",
    "effect_control_step",
    "effect_control_time_seconds",
    "effect_execution_push_directional_block_displacement_m",
    "effect_forbidden_non_tool_robot_desk_penetration",
    "effect_forbidden_tool_desk_penetration",
    "effect_intervention_type_id",
    "effect_obstacle_xy",
    "effect_physical_joint_acceleration",
    "effect_physical_joint_velocity",
    "effect_policy_source_switch",
    "effect_progress_toward_target_m",
    "effect_reported_joint_position",
    "effect_reported_joint_velocity",
    "effect_reward",
    "effect_runtime_guard_dynamic_projection_count",
    "effect_runtime_guard_static_projection_count",
    "effect_safety_cost",
    "effect_strict_contained",
    "effect_strict_settled",
    "effect_strict_success_streak",
    "effect_strict_target_coverage",
    "effect_substep_geometric_push_side_contact_count",
    "effect_substep_non_tool_robot_desk_contact_count",
    "effect_substep_obstacle_contact_counts",
    "effect_substep_representative_block_local_contact_xyz_m",
    "effect_substep_representative_edge_side_contact_valid",
    "effect_substep_representative_normal_tool_to_block_world",
    "effect_substep_representative_point_xyz_m",
    "effect_substep_sample_time_seconds",
    "effect_substep_tool_block_contact_count",
    "effect_substep_tool_block_total_normal_force_n",
    "effect_substep_tool_desk_contact_count",
    "effect_substep_tool_desk_signed_distance_m",
    "effect_substep_valid_push_side_contact_count",
    "effect_substep_valid_push_side_total_normal_force_n",
    "effect_success",
    "effect_target_xy",
    "effect_terminated",
    "effect_tool_block_contact_point_xyz",
    "effect_tool_block_contact_wrench_contact_frame",
    "effect_tool_pose",
    "effect_tool_twist_linear_angular",
    "effect_tracking_error_joint_target_minus_position_rad",
    "effect_tracking_error_physics_applied_target_minus_position_rad",
    "effect_truncated",
    "effect_valid_push_side_contact_substep_count",
    "effect_world_model_training_mask",
    "exposure_block_pose_xyz_wxyz",
    "exposure_block_twist_linear_angular",
    "exposure_control_step",
    "exposure_control_time_seconds",
    "exposure_current_endpoint_block_pose_xyz_wxyz",
    "exposure_current_endpoint_block_twist_linear_angular",
    "exposure_current_endpoint_control_step",
    "exposure_current_endpoint_control_time_seconds",
    "exposure_current_endpoint_obstacle_xy",
    "exposure_current_endpoint_physical_joint_acceleration",
    "exposure_current_endpoint_physical_joint_position",
    "exposure_current_endpoint_physical_joint_velocity",
    "exposure_current_endpoint_reported_joint_position",
    "exposure_current_endpoint_reported_joint_velocity",
    "exposure_current_endpoint_target_xy",
    "exposure_current_endpoint_tool_pose",
    "exposure_current_endpoint_tool_twist_linear_angular",
    "exposure_obstacle_xy",
    "exposure_physical_joint_acceleration",
    "exposure_physical_joint_position",
    "exposure_physical_joint_velocity",
    "exposure_previous_endpoint_block_pose_xyz_wxyz",
    "exposure_previous_endpoint_block_twist_linear_angular",
    "exposure_previous_endpoint_control_step",
    "exposure_previous_endpoint_control_time_seconds",
    "exposure_previous_endpoint_obstacle_xy",
    "exposure_previous_endpoint_physical_joint_acceleration",
    "exposure_previous_endpoint_physical_joint_position",
    "exposure_previous_endpoint_physical_joint_velocity",
    "exposure_previous_endpoint_reported_joint_position",
    "exposure_previous_endpoint_reported_joint_velocity",
    "exposure_previous_endpoint_target_xy",
    "exposure_previous_endpoint_tool_pose",
    "exposure_previous_endpoint_tool_twist_linear_angular",
    "exposure_reported_joint_position",
    "exposure_reported_joint_velocity",
    "exposure_target_xy",
    "exposure_tool_pose",
    "exposure_tool_twist_linear_angular",
    "segmentation_wrist",
)

# Attribute namespaces need optional fields because collection/run metadata is
# not row data.  They remain exact finite allow-lists so an unreviewed source,
# private, privileged, or simulator-state attribute cannot hide beside valid
# metadata.  Privileged fields that are intentionally retained for audit (for
# example ``observable_privileged_boundary``) are named individually.
EXECUTION_CORE_OPTIONAL_ROOT_ATTRIBUTES_V1 = (
    "action_unit",
    "added_contact_tool",
    "bus_protocol",
    "camera_exposure_state_contract",
    "camera_mount",
    "camera_streams",
    "capture_config",
    "causal_action_contract",
    "clock_domains",
    "collection_profile_hash",
    "collection_provenance",
    "command_ack_semantics",
    "contact_measurement_semantics",
    "controller_identity",
    "controller_version",
    "deployment_equivalent",
    "deployment_reset_equivalent",
    "domain",
    "environment_config",
    "environment_profile",
    "environment_version",
    "episode_commit_protocol_format",
    "episode_commit_receipt_semantics",
    "expert_config",
    "format",
    "hdf5_compression",
    "hdf5_compression_level",
    "hdf5_frame_chunked",
    "hdf5_shuffle",
    "joint_acceleration_unit",
    "joint_order",
    "joint_position_unit",
    "joint_velocity_unit",
    "label_contract",
    "measured_executed_action_unit",
    "observable_privileged_boundary",
    "physical_camera_data",
    "physical_hardware_connected",
    "physical_samples",
    "physical_trials",
    "physical_validation",
    "physically_calibrated",
    "policy_intent_action_unit",
    "policy_intent_execution_unmodified_mask_semantics",
    "robot_model",
    "rolling_shutter_alpha_formula",
    "rolling_shutter_geometry_model_id",
    "rolling_shutter_readout_direction",
    "rolling_shutter_state_contract",
    "split",
    "stock_follower_unmodified",
    "synthetic",
    "synthetic_camera",
    "task_aligned_privileged_reset",
    "temporal_alignment_contract",
    "trajectory_contract_version",
    "tool_profile",
    "world_model_loader_supports_rolling_shutter_geometry_model",
)

EXECUTION_CORE_OPTIONAL_EPISODE_ATTRIBUTES_V1 = (
    "action_label_eligible",
    "added_contact_tool",
    "action_training_eligible_ratio",
    "action_training_eligible_rows",
    "action_unit",
    "bus_protocol",
    "camera_calibration",
    "camera_calibration_sha256",
    "camera_extrinsic_sha256",
    "camera_mount",
    "camera_streams",
    "capture_config",
    "causal_action_contract",
    "causal_push_applied_parent_after_precontact_completion",
    "causal_valid_push_directional_displacement_m",
    "causal_valid_push_target_progress_m",
    "causal_valid_push_transition_steps",
    "clock_domains",
    "collection_profile_hash",
    "collection_provenance",
    "command_ack_semantics",
    "compiled_model_bytes",
    "compiled_model_hash_semantics",
    "compiled_model_sha256",
    "contact_measurement_semantics",
    "controller_identity",
    "controller_source_sha256",
    "controller_version",
    "dataset_format",
    "deployment_equivalent",
    "deployment_reset_equivalent",
    "diagnostic_teacher_planned_phase_names",
    "domain",
    "dynamics",
    "environment_config",
    "environment_profile",
    "environment_version",
    "episode_commit_protocol_format",
    "episode_commit_receipt_semantics",
    "episode_commit_receipt_sha256",
    "episode_commit_state",
    "episode_content_sha256",
    "episode_domain",
    "episode_net_block_displacement_m",
    "execution_divergent_submission_command_ids",
    "execution_phase_threshold_profile_hash",
    "expert_config",
    "failure_code",
    "final_compiled_model_bytes",
    "final_compiled_model_sha256",
    "final_teacher_metadata",
    "firmware_version",
    "first_causal_valid_push_effect_step",
    "frames",
    "initial_robot_object_state_hash",
    "intervention_summary",
    "invalid_geometry_contact_steps",
    "joint_acceleration_unit",
    "joint_order",
    "joint_position_unit",
    "joint_velocity_unit",
    "label_contract",
    "lighting",
    "measured_executed_action_unit",
    "minimum_pusher_desk_signed_distance_m",
    "model_profile_sha256",
    "mujoco_version",
    "nominal_control_period_seconds",
    "observable_privileged_boundary",
    "obstacle",
    "occlusion",
    "physical_camera_data",
    "physical_hardware_connected",
    "physical_samples",
    "physical_trials",
    "physical_validation",
    "physical_wrist_camera_calibrated",
    "physically_calibrated",
    "planned_phase_semantics",
    "policy_checkpoint_status",
    "policy_intent_action_training_eligible_ratio",
    "policy_intent_action_training_eligible_rows",
    "policy_intent_action_unit",
    "precontact_block_contact_steps",
    "precontact_completed_once",
    "precontact_plan_infeasible_steps_before_first_causal_push",
    "precontact_plan_infeasible_steps_total",
    "randomization_profile_sha256",
    "randomization_realization_sha256",
    "robot_description_artifact_path",
    "robot_description_format",
    "robot_description_sha256",
    "robot_description_source",
    "robot_model",
    "scenario_id",
    "seed",
    "stress",
    "strict_success",
    "stock_follower_unmodified",
    "strict_success_effect_step",
    "success",
    "synthetic",
    "synthetic_camera",
    "task",
    "task_aligned_privileged_reset",
    "task_zh",
    "temporal_alignment_contract",
    "tool_block_contact_steps",
    "tool_profile",
    "trajectory_contract_version",
    "unique_live_episode_identity",
    "unique_live_episode_key",
    "unmodified_policy_execution_training_eligible_ratio",
    "unmodified_policy_execution_training_eligible_rows",
    "valid_push_side_contact_steps",
    "verified_recovery",
)

EXECUTION_CORE_OPTIONAL_MANIFEST_FIELDS_V1 = (
    "camera_mount",
    "camera_streams",
    "episode_commit_protocol_format",
    "episode_commit_receipt_semantics",
    "episode_commit_receipt_sha256",
    "episode_content_sha256",
    "physical_camera_data",
    "physical_hardware_connected",
    "physical_samples",
    "physical_trials",
    "promotion_valid_success",
    "recovery_event_ledger_format",
    "recovery_injector_format",
    "recovery_injector_profile_sha256",
    "recovery_injector_source_sha256",
    "success",
    "synthetic",
    "synthetic_camera",
)


def _nonempty_name(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value or any(ord(character) < 0x20 for character in value):
        raise ValueError(f"{label} must be a non-empty printable string")
    return value


def _name_tuple(values: Sequence[str], *, label: str) -> tuple[str, ...]:
    normalized = tuple(_nonempty_name(value, label=label) for value in values)
    if len(normalized) != len(set(normalized)):
        raise ValueError(f"{label} contains duplicates")
    return normalized


def _freeze_preparation_value(value: Any, *, label: str) -> Any:
    """Return a recursively immutable, JSON-compatible provenance value.

    ``dataclass(frozen=True)`` only prevents field reassignment; it does not
    protect dictionaries or lists stored inside a field.  Preparation evidence
    participates in profile hashes and is later copied into three persistence
    layers, so accepting a mutable nested value would create a time-of-check /
    time-of-write provenance split.
    """

    if isinstance(value, Mapping):
        frozen: dict[str, Any] = {}
        for key, item in value.items():
            try:
                canonical_key = _nonempty_name(key, label=f"{label} key")
            except ValueError as error:
                raise TypeError(f"{label} keys must be non-empty printable strings") from error
            frozen[canonical_key] = _freeze_preparation_value(
                item,
                label=f"{label}.{canonical_key}",
            )
        return MappingProxyType(frozen)
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_preparation_value(item, label=f"{label}[]") for item in value)
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not np.isfinite(value):
            raise ValueError(f"{label} must not contain non-finite floats")
        return value
    raise TypeError(f"{label} must contain only JSON-compatible provenance values")


def _preparation_json_value(value: Any) -> Any:
    """Convert an immutable preparation value back to canonical JSON types."""

    if isinstance(value, Mapping):
        return {key: _preparation_json_value(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_preparation_json_value(item) for item in value]
    return value


def _finite_float32_vector(value: object, *, name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.shape != (_JOINTS,) or not np.issubdtype(array.dtype, np.number):
        raise ValueError(f"{name} must be a numeric ({_JOINTS},) vector")
    canonical = np.asarray(array, dtype=np.float32)
    if not np.all(np.isfinite(canonical)):
        raise ValueError(f"{name} must be finite")
    if name.endswith("action") and np.any(np.abs(canonical) > 1.00001):
        raise ValueError(f"{name} must remain in normalized action bounds")
    return canonical


@dataclass(frozen=True, slots=True)
class H5RowDatasetContractV1:
    """Shape/dtype contract for one source-specific HDF5 row dataset."""

    name: str
    dtype: str
    trailing_shape: tuple[int, ...]
    policy_input_eligible: bool = False
    constant_scalar: int | float | None = None

    def __post_init__(self) -> None:
        _nonempty_name(self.name, label="dataset name")
        if self.dtype not in _H5_DTYPES:
            raise ValueError(f"unsupported HDF5 row dtype: {self.dtype}")
        if any(
            isinstance(size, bool) or not isinstance(size, int) or size < 1 for size in self.trailing_shape
        ):
            raise ValueError("dataset trailing dimensions must be positive integers")
        if type(self.policy_input_eligible) is not bool:
            raise TypeError("policy_input_eligible must be boolean")
        if self.constant_scalar is not None and self.trailing_shape:
            raise ValueError("constant_scalar is valid only for scalar row datasets")
        if self.constant_scalar is not None:
            try:
                canonical = np.asarray(self.constant_scalar, dtype=np.dtype(self.dtype)).item()
            except (TypeError, ValueError, OverflowError) as error:
                raise ValueError("constant_scalar cannot be represented by the declared dtype") from error
            if canonical != self.constant_scalar:
                raise ValueError("constant_scalar changes under the declared dtype")

    def validate_row_v1(self, value: object) -> np.ndarray:
        array = np.asarray(value)
        expected_dtype = np.dtype(self.dtype)
        if array.dtype != expected_dtype or array.shape != self.trailing_shape:
            raise ValueError(
                f"{self.name} row must have dtype={expected_dtype} shape={self.trailing_shape}; "
                f"got dtype={array.dtype} shape={array.shape}"
            )
        if np.issubdtype(array.dtype, np.floating) and not np.all(np.isfinite(array)):
            raise ValueError(f"{self.name} row must be finite")
        if self.constant_scalar is not None and array.item() != self.constant_scalar:
            raise ValueError(f"{self.name} row differs from its source ID")
        return array

    def validate_payload_v1(self, value: object, *, row_count: int) -> np.ndarray:
        array = np.asarray(value)
        expected_dtype = np.dtype(self.dtype)
        expected_shape = (row_count, *self.trailing_shape)
        if array.dtype != expected_dtype or array.shape != expected_shape:
            raise ValueError(
                f"{self.name} payload must have dtype={expected_dtype} shape={expected_shape}; "
                f"got dtype={array.dtype} shape={array.shape}"
            )
        if np.issubdtype(array.dtype, np.floating) and not np.all(np.isfinite(array)):
            raise ValueError(f"{self.name} payload must be finite")
        if self.constant_scalar is not None and not np.all(array == self.constant_scalar):
            raise ValueError(f"{self.name} payload differs from its source ID")
        return array


@dataclass(frozen=True, slots=True)
class SourceSchemaProfileV1:
    """Complete namespace boundary consumed by the future shared writer.

    ``required_common_datasets`` is the exact shared execution contract.  The
    source-specific contracts describe only rows emitted by an adapter.  Root,
    episode, and manifest binding fields are separate because silently binding
    an artifact at just one output level is insufficient for resumable
    collection and formal co-audit.
    """

    source_type: str
    source_numeric_id: int
    controller_identity: str
    controller_version: str
    primary_action_label_dataset: str
    raw_policy_action_dataset: str
    policy_source_id_dataset: str
    policy_input_keys: tuple[str, ...]
    policy_input_datasets: tuple[str, ...]
    required_common_datasets: tuple[str, ...]
    decision_dataset_contracts: tuple[H5RowDatasetContractV1, ...]
    effect_dataset_contracts: tuple[H5RowDatasetContractV1, ...]
    required_root_attributes: tuple[str, ...]
    required_episode_attributes: tuple[str, ...]
    required_manifest_fields: tuple[str, ...]
    root_binding_fields: tuple[str, ...]
    episode_binding_fields: tuple[str, ...]
    manifest_binding_fields: tuple[str, ...]
    optional_common_datasets: tuple[str, ...] = EXECUTION_CORE_OPTIONAL_COMMON_DATASETS_V1
    optional_root_attributes: tuple[str, ...] = EXECUTION_CORE_OPTIONAL_ROOT_ATTRIBUTES_V1
    optional_episode_attributes: tuple[str, ...] = EXECUTION_CORE_OPTIONAL_EPISODE_ATTRIBUTES_V1
    optional_manifest_fields: tuple[str, ...] = EXECUTION_CORE_OPTIONAL_MANIFEST_FIELDS_V1
    forbidden_datasets: tuple[str, ...] = ()
    forbidden_root_attributes: tuple[str, ...] = ()
    forbidden_episode_attributes: tuple[str, ...] = ()
    forbidden_manifest_fields: tuple[str, ...] = ()
    forbidden_dataset_tokens: tuple[str, ...] = ()
    forbidden_attribute_tokens: tuple[str, ...] = ()
    exact_root_attribute_values: tuple[tuple[str, str | int | float | bool], ...] = ()
    exact_episode_attribute_values: tuple[tuple[str, str | int | float | bool], ...] = ()
    exact_manifest_field_values: tuple[tuple[str, str | int | float | bool], ...] = ()
    replay_sidecar_required: bool = False
    replay_commitment_dataset: str | None = None
    safe_action_diagnostic_alias_dataset: str | None = None
    forbidden_post_submission_rewrite_reason_mask: int = 0
    format: str = SOURCE_SCHEMA_PROFILE_FORMAT

    def __post_init__(self) -> None:
        if self.format != SOURCE_SCHEMA_PROFILE_FORMAT:
            raise ValueError("source schema profile format mismatch")
        _nonempty_name(self.source_type, label="source_type")
        if isinstance(self.source_numeric_id, bool) or not 0 <= self.source_numeric_id <= 255:
            raise ValueError("source_numeric_id must fit uint8")
        for value, label in (
            (self.controller_identity, "controller_identity"),
            (self.controller_version, "controller_version"),
            (self.primary_action_label_dataset, "primary action label"),
            (self.raw_policy_action_dataset, "raw policy action dataset"),
            (self.policy_source_id_dataset, "policy source ID dataset"),
        ):
            _nonempty_name(value, label=label)
        tuple_fields = (
            "policy_input_keys",
            "policy_input_datasets",
            "required_common_datasets",
            "required_root_attributes",
            "required_episode_attributes",
            "required_manifest_fields",
            "root_binding_fields",
            "episode_binding_fields",
            "manifest_binding_fields",
            "optional_common_datasets",
            "optional_root_attributes",
            "optional_episode_attributes",
            "optional_manifest_fields",
            "forbidden_datasets",
            "forbidden_root_attributes",
            "forbidden_episode_attributes",
            "forbidden_manifest_fields",
            "forbidden_dataset_tokens",
            "forbidden_attribute_tokens",
        )
        for field_name in tuple_fields:
            object.__setattr__(
                self,
                field_name,
                _name_tuple(getattr(self, field_name), label=field_name),
            )
        if type(self.replay_sidecar_required) is not bool:
            raise TypeError("replay_sidecar_required must be boolean")
        if (
            isinstance(self.forbidden_post_submission_rewrite_reason_mask, bool)
            or not isinstance(self.forbidden_post_submission_rewrite_reason_mask, int)
            or not 0 <= self.forbidden_post_submission_rewrite_reason_mask <= np.iinfo(np.uint32).max
        ):
            raise ValueError("forbidden rewrite-reason mask must fit uint32")

        decision_names = tuple(contract.name for contract in self.decision_dataset_contracts)
        effect_names = tuple(contract.name for contract in self.effect_dataset_contracts)
        if len(decision_names) != len(set(decision_names)) or len(effect_names) != len(set(effect_names)):
            raise ValueError("source row dataset contracts contain duplicates")
        if set(decision_names) & set(effect_names):
            raise ValueError("one source dataset cannot be emitted at both decision and effect stages")
        source_names = set(decision_names) | set(effect_names)
        common_names = set(self.required_common_datasets)
        policy_dataset_names = set(self.policy_input_datasets)
        optional_common_names = set(self.optional_common_datasets)
        forbidden_names = set(self.forbidden_datasets)
        if source_names & common_names:
            raise ValueError("source-specific dataset contracts overlap common execution datasets")
        if (source_names | common_names) & forbidden_names:
            raise ValueError("required and forbidden datasets overlap")
        if self.primary_action_label_dataset not in common_names:
            raise ValueError("primary action label must be owned by the execution core")
        if self.primary_action_label_dataset in policy_dataset_names:
            raise ValueError("the action label cannot enter the policy input dataset allow-list")
        if source_names & policy_dataset_names:
            raise ValueError("source-specific datasets cannot extend the policy input allow-list")
        declared_required_names = source_names | common_names | policy_dataset_names
        if optional_common_names & declared_required_names:
            raise ValueError("optional common datasets overlap required or source datasets")
        if self.raw_policy_action_dataset not in source_names:
            raise ValueError("raw policy action must be declared as a source row dataset")
        if self.policy_source_id_dataset not in source_names:
            raise ValueError("policy source ID must be declared as a source row dataset")
        source_id_contract = self.row_contracts_v1()[self.policy_source_id_dataset]
        if source_id_contract.dtype != "uint8" or source_id_contract.trailing_shape:
            raise ValueError("policy source ID dataset must be scalar uint8")
        if source_id_contract.constant_scalar != self.source_numeric_id:
            raise ValueError("policy source ID contract must bind source_numeric_id")

        for required, binding, label in (
            (self.required_root_attributes, self.root_binding_fields, "root"),
            (self.required_episode_attributes, self.episode_binding_fields, "episode"),
            (self.required_manifest_fields, self.manifest_binding_fields, "manifest"),
        ):
            if not set(binding) <= set(required):
                raise ValueError(f"{label} binding fields must also be required")
        if set(self.required_root_attributes) & set(self.forbidden_root_attributes):
            raise ValueError("required and forbidden root attributes overlap")
        if set(self.required_episode_attributes) & set(self.forbidden_episode_attributes):
            raise ValueError("required and forbidden episode attributes overlap")
        if set(self.required_manifest_fields) & set(self.forbidden_manifest_fields):
            raise ValueError("required and forbidden manifest fields overlap")
        for pairs, required, label in (
            (self.exact_root_attribute_values, self.required_root_attributes, "root"),
            (self.exact_episode_attribute_values, self.required_episode_attributes, "episode"),
            (self.exact_manifest_field_values, self.required_manifest_fields, "manifest"),
        ):
            names: list[str] = []
            for pair in pairs:
                if not isinstance(pair, tuple) or len(pair) != 2:
                    raise ValueError(f"exact {label} values must contain key/value pairs")
                name, value = pair
                names.append(_nonempty_name(name, label=f"exact {label} value name"))
                if not isinstance(value, (str, int, float, bool)):
                    raise TypeError(f"exact {label} value must be a JSON scalar")
                if isinstance(value, float) and not np.isfinite(value):
                    raise ValueError(f"exact {label} float must be finite")
            if len(names) != len(set(names)):
                raise ValueError(f"exact {label} value names contain duplicates")
            if not set(names) <= set(required):
                raise ValueError(f"exact {label} value names must also be required")

        if self.replay_sidecar_required:
            if not self.replay_commitment_dataset:
                raise ValueError("a replay sidecar requires an HDF5 commitment dataset")
            contract = self.row_contracts_v1().get(self.replay_commitment_dataset)
            if contract is None or contract.dtype != "uint8" or contract.trailing_shape != (32,):
                raise ValueError("replay commitment must be a source uint8[32] row dataset")
            if contract.policy_input_eligible:
                raise ValueError("replay commitments can never be policy inputs")
            if self.replay_commitment_dataset in self.policy_input_keys:
                raise ValueError("replay commitment escaped into the policy whitelist")
        elif self.replay_commitment_dataset is not None:
            raise ValueError("replay_commitment_dataset requires replay_sidecar_required")
        if self.safe_action_diagnostic_alias_dataset is not None:
            _nonempty_name(
                self.safe_action_diagnostic_alias_dataset,
                label="safe action diagnostic alias dataset",
            )
            alias_contract = self.row_contracts_v1().get(self.safe_action_diagnostic_alias_dataset)
            if (
                alias_contract is None
                or alias_contract.dtype != "float32"
                or alias_contract.trailing_shape != (_JOINTS,)
            ):
                raise ValueError("safe action diagnostic alias must be a source float32[6] row")
            if self.safe_action_diagnostic_alias_dataset in policy_dataset_names:
                raise ValueError("safe action diagnostic alias cannot be a policy input")
        if any(contract.policy_input_eligible for contract in self.row_contracts_v1().values()):
            raise ValueError("source-specific audit datasets cannot extend the policy whitelist")

    def row_contracts_v1(self) -> dict[str, H5RowDatasetContractV1]:
        return {
            contract.name: contract
            for contract in (*self.decision_dataset_contracts, *self.effect_dataset_contracts)
        }

    def collection_episode_binding_fields_v1(self) -> tuple[str, ...]:
        """Return episode bindings frozen before any episode is executed.

        A binding present at the root is collection-static by construction.
        Episode bindings that have no root counterpart (for example one
        scratch replay receipt per episode) are deliberately excluded and
        must be supplied by ``SourceEpisodeFinalizationV1``.
        """

        root = set(self.root_binding_fields)
        return tuple(name for name in self.episode_binding_fields if name in root)

    def dynamic_episode_binding_fields_v1(self) -> tuple[str, ...]:
        """Return per-episode bindings that cannot be frozen at collection prepare."""

        root = set(self.root_binding_fields)
        return tuple(name for name in self.episode_binding_fields if name not in root)

    def collection_manifest_binding_fields_v1(self) -> tuple[str, ...]:
        """Return manifest bindings copied from collection-static root evidence."""

        root = set(self.root_binding_fields)
        return tuple(name for name in self.manifest_binding_fields if name in root)

    def dynamic_manifest_binding_fields_v1(self) -> tuple[str, ...]:
        """Return manifest bindings produced only after an episode executes."""

        root = set(self.root_binding_fields)
        return tuple(name for name in self.manifest_binding_fields if name not in root)

    def allowed_dataset_names_v1(self) -> frozenset[str]:
        """Return the exact finite HDF5 dataset namespace for this profile."""

        return frozenset(
            set(self.required_common_datasets)
            | set(self.policy_input_datasets)
            | set(self.row_contracts_v1())
            | set(self.optional_common_datasets)
        )

    def payload_v1(self) -> dict[str, Any]:
        """Return the complete canonical schema declaration for hashing."""

        return asdict(self)

    def sha256_v1(self) -> str:
        encoded = json.dumps(
            self.payload_v1(),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def validate_namespaces_v1(
        self,
        *,
        root_attributes: Mapping[str, Any],
        episode_attributes: Mapping[str, Any],
        manifest_fields: Mapping[str, Any] | Sequence[str],
        dataset_names: Sequence[str],
    ) -> None:
        """Reject missing source bindings and source-namespace contamination."""

        root_names = set(root_attributes)
        episode_names = set(episode_attributes)
        manifest_names = set(manifest_fields)
        datasets = set(dataset_names)
        required_datasets = (
            set(self.required_common_datasets)
            | set(self.policy_input_datasets)
            | set(self.row_contracts_v1())
        )
        checks = (
            (set(self.required_root_attributes), root_names, "root attribute"),
            (set(self.required_episode_attributes), episode_names, "episode attribute"),
            (set(self.required_manifest_fields), manifest_names, "manifest field"),
            (required_datasets, datasets, "dataset"),
        )
        for required, actual, label in checks:
            missing = sorted(required - actual)
            if missing:
                raise ValueError(f"missing required {label}s: {missing}")
        forbidden_checks = (
            (set(self.forbidden_root_attributes), root_names, "root attribute"),
            (set(self.forbidden_episode_attributes), episode_names, "episode attribute"),
            (set(self.forbidden_manifest_fields), manifest_names, "manifest field"),
            (set(self.forbidden_datasets), datasets, "dataset"),
        )
        for forbidden, actual, label in forbidden_checks:
            present = sorted(forbidden & actual)
            if present:
                raise ValueError(f"forbidden {label}s are present: {present}")
        for names, tokens, label in (
            (datasets, self.forbidden_dataset_tokens, "dataset"),
            (root_names, self.forbidden_attribute_tokens, "root attribute"),
            (episode_names, self.forbidden_attribute_tokens, "episode attribute"),
            (manifest_names, self.forbidden_attribute_tokens, "manifest field"),
        ):
            contaminated = sorted(
                name for name in names if any(token.casefold() in name.casefold() for token in tokens)
            )
            if contaminated:
                raise ValueError(f"source-forbidden {label} namespace: {contaminated}")
        unknown_checks = (
            (
                datasets - self.allowed_dataset_names_v1(),
                "dataset",
            ),
            (
                root_names - set(self.required_root_attributes) - set(self.optional_root_attributes),
                "root attribute",
            ),
            (
                episode_names - set(self.required_episode_attributes) - set(self.optional_episode_attributes),
                "episode attribute",
            ),
            (
                manifest_names - set(self.required_manifest_fields) - set(self.optional_manifest_fields),
                "manifest field",
            ),
        )
        for undeclared, label in unknown_checks:
            if undeclared:
                raise ValueError(f"undeclared {label}s: {sorted(undeclared)}")
        if root_attributes.get("source_type") != self.source_type:
            raise ValueError("root source_type differs from source schema profile")
        if episode_attributes.get("source_type") != self.source_type:
            raise ValueError("episode source_type differs from source schema profile")
        for attributes, label in ((root_attributes, "root"), (episode_attributes, "episode")):
            if attributes.get("act_primary_label_dataset") != self.primary_action_label_dataset:
                raise ValueError(f"{label} primary action label differs from source profile")
        for values, expected_pairs, label in (
            (root_attributes, self.exact_root_attribute_values, "root attribute"),
            (episode_attributes, self.exact_episode_attribute_values, "episode attribute"),
            (manifest_fields, self.exact_manifest_field_values, "manifest field"),
        ):
            if expected_pairs and not isinstance(values, Mapping):
                raise TypeError(f"exact {label} validation requires a mapping")
            for name, expected in expected_pairs:
                assert isinstance(values, Mapping)
                if values[name] != expected:
                    raise ValueError(f"{label} {name} differs from its exact source contract")

    def validate_payload_v1(self, payload: Mapping[str, object], *, row_count: int) -> None:
        if isinstance(row_count, bool) or not isinstance(row_count, int) or row_count < 1:
            raise ValueError("row_count must be positive")
        names = set(payload)
        required = (
            set(self.required_common_datasets)
            | set(self.policy_input_datasets)
            | set(self.row_contracts_v1())
        )
        missing = sorted(required - names)
        if missing:
            raise ValueError(f"source payload is missing datasets: {missing}")
        present_forbidden = sorted(set(self.forbidden_datasets) & names)
        if present_forbidden:
            raise ValueError(f"source payload contains forbidden datasets: {present_forbidden}")
        contaminated = sorted(
            name
            for name in names
            if any(token.casefold() in name.casefold() for token in self.forbidden_dataset_tokens)
        )
        if contaminated:
            raise ValueError(f"source payload contains forbidden dataset namespaces: {contaminated}")
        undeclared = sorted(names - self.allowed_dataset_names_v1())
        if undeclared:
            raise ValueError(f"source payload contains undeclared datasets: {undeclared}")
        for name, value in payload.items():
            try:
                length = len(np.asarray(value))
            except TypeError as error:
                raise ValueError(f"dataset {name} has no row axis") from error
            if length != row_count:
                raise ValueError(f"dataset {name} row count differs from the episode")
        for name, contract in self.row_contracts_v1().items():
            contract.validate_payload_v1(payload[name], row_count=row_count)
        safe = np.asarray(payload[self.primary_action_label_dataset])
        raw = np.asarray(payload[self.raw_policy_action_dataset])
        if (
            safe.shape != (row_count, _JOINTS)
            or not np.issubdtype(safe.dtype, np.floating)
            or not np.all(np.isfinite(safe))
            or np.any(np.abs(safe) > 1.00001)
        ):
            raise ValueError("primary safe action payload must be finite float[T,6]")
        if (
            raw.shape != (row_count, _JOINTS)
            or not np.issubdtype(raw.dtype, np.floating)
            or not np.all(np.isfinite(raw))
            or np.any(np.abs(raw) > 1.00001)
        ):
            raise ValueError("raw policy action payload must be finite float[T,6]")
        if self.safe_action_diagnostic_alias_dataset is not None:
            alias = np.asarray(payload[self.safe_action_diagnostic_alias_dataset])
            if not np.array_equal(alias, safe):
                raise ValueError("safe action diagnostic alias differs from primary action label")
        if self.forbidden_post_submission_rewrite_reason_mask:
            masks = np.asarray(payload["command_applied_post_submission_rewrite_reason_mask"])
            if not np.issubdtype(masks.dtype, np.integer) or masks.shape != (row_count,):
                raise ValueError("post-submission rewrite-reason mask must be integer[T]")
            if np.any(
                np.bitwise_and(
                    masks.astype(np.uint64),
                    np.uint64(self.forbidden_post_submission_rewrite_reason_mask),
                )
            ):
                raise ValueError("source-forbidden post-submission rewrite reason was emitted")


@dataclass(frozen=True, slots=True)
class SourceCollectionPreparationV1:
    """Source artifacts and immutable bindings prepared before writer resume."""

    source_type: str
    collection_profile_components: Mapping[str, Any]
    root_attributes: Mapping[str, Any]
    manifest_fields: Mapping[str, Any]

    def __post_init__(self) -> None:
        _nonempty_name(self.source_type, label="collection preparation source_type")
        for field_name in (
            "collection_profile_components",
            "root_attributes",
            "manifest_fields",
        ):
            value = getattr(self, field_name)
            if not isinstance(value, Mapping):
                raise TypeError(f"{field_name} must be a mapping")
            object.__setattr__(
                self,
                field_name,
                _freeze_preparation_value(value, label=field_name),
            )

    def validate_for_profile_v1(self, profile: SourceSchemaProfileV1) -> None:
        if self.source_type != profile.source_type:
            raise ValueError("collection preparation source_type mismatch")
        if not isinstance(self.collection_profile_components, Mapping):
            raise TypeError("collection profile components must be a mapping")
        if set(self.collection_profile_components) != set(SOURCE_PREPARATION_PROFILE_COMPONENT_FIELDS):
            raise ValueError("collection preparation profile components are incomplete or unknown")
        expected_components = {
            "source_schema_profile_format": profile.format,
            "source_schema_profile_sha256": profile.sha256_v1(),
            "source_type": profile.source_type,
            "controller_identity": profile.controller_identity,
            "controller_version": profile.controller_version,
        }
        if dict(self.collection_profile_components) != expected_components:
            raise ValueError("collection preparation profile hash or identity mismatch")
        collection_episode_fields = profile.collection_episode_binding_fields_v1()
        collection_manifest_fields = profile.collection_manifest_binding_fields_v1()
        dynamic_episode_fields = set(profile.dynamic_episode_binding_fields_v1())
        dynamic_manifest_fields = set(profile.dynamic_manifest_binding_fields_v1())
        for values, required, label in (
            (self.root_attributes, profile.root_binding_fields, "root"),
            (self.manifest_fields, collection_manifest_fields, "manifest"),
        ):
            if not isinstance(values, Mapping):
                raise TypeError(f"source {label} bindings must be a mapping")
            missing = sorted(set(required) - set(values))
            if missing:
                raise ValueError(f"source {label} bindings are incomplete: {missing}")
        missing_episode = sorted(
            set(collection_episode_fields) - (set(self.root_attributes) | set(self.manifest_fields))
        )
        if missing_episode:
            raise ValueError(f"source episode bindings have no prepared value: {missing_episode}")
        prepared_dynamic = sorted(
            (dynamic_episode_fields | dynamic_manifest_fields)
            & (set(self.root_attributes) | set(self.manifest_fields))
        )
        if prepared_dynamic:
            raise ValueError(
                "episode-dynamic source bindings cannot be frozen during collection preparation: "
                f"{prepared_dynamic}"
            )
        collection_binding_fields = (
            set(profile.root_binding_fields)
            | set(collection_episode_fields)
            | set(collection_manifest_fields)
        )
        for name in collection_binding_fields & set(self.root_attributes) & set(self.manifest_fields):
            if self.root_attributes[name] != self.manifest_fields[name]:
                raise ValueError(f"source root/manifest binding differs for {name}")

    def episode_binding_values_v1(
        self,
        profile: SourceSchemaProfileV1,
    ) -> Mapping[str, Any]:
        """Resolve only collection-static episode bindings.

        Episode-dynamic bindings are intentionally absent here.  Freezing a
        replay receipt in this object would incorrectly force every episode to
        claim the same replay bundle.
        """

        self.validate_for_profile_v1(profile)
        return MappingProxyType(
            {
                name: (
                    self.root_attributes[name] if name in self.root_attributes else self.manifest_fields[name]
                )
                for name in profile.collection_episode_binding_fields_v1()
            }
        )

    def binding_sha256_v1(self, profile: SourceSchemaProfileV1) -> str:
        self.validate_for_profile_v1(profile)
        payload = {
            "collection_profile_components": _preparation_json_value(self.collection_profile_components),
            "root_binding": {
                name: _preparation_json_value(self.root_attributes[name])
                for name in profile.root_binding_fields
            },
            "manifest_binding": {
                name: _preparation_json_value(self.manifest_fields[name])
                for name in profile.collection_manifest_binding_fields_v1()
            },
        }
        encoded = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True, slots=True)
class SourceEpisodeContextV1:
    episode_index: int
    seed: int
    split: str
    obstacle: bool
    stress: bool
    collection_uid: str
    collection_profile_sha256: str
    episode_identity: Mapping[str, Any]

    def __post_init__(self) -> None:
        for name in ("episode_index", "seed"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        _nonempty_name(self.split, label="split")
        _nonempty_name(self.collection_uid, label="collection_uid")
        if _SHA256_RE.fullmatch(self.collection_profile_sha256) is None:
            raise ValueError("collection_profile_sha256 must be a lowercase SHA-256")
        if type(self.obstacle) is not bool or type(self.stress) is not bool:
            raise TypeError("obstacle and stress must be boolean")
        if not isinstance(self.episode_identity, Mapping) or not self.episode_identity:
            raise ValueError("episode_identity must be a non-empty mapping")


@dataclass(frozen=True, slots=True)
class SourceEpisodeStartRequestV1:
    """Adapter-visible episode start state after the shared core resets the plant."""

    episode: SourceEpisodeContextV1
    env: Any


@dataclass(frozen=True, slots=True)
class SourceDecisionRequestV1:
    episode: SourceEpisodeContextV1
    row_index: int
    control_step: int
    env: Any
    observation: Mapping[str, object]

    def __post_init__(self) -> None:
        for name in ("row_index", "control_step"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if not isinstance(self.observation, Mapping):
            raise TypeError("observation must be a mapping")


@dataclass(frozen=True, slots=True)
class SourceStepDecisionV1:
    """One adapter decision before the shared core calls ``env.step``."""

    source_type: str
    raw_policy_action: np.ndarray
    safe_action: np.ndarray
    requested_joint_target: np.ndarray
    queued_safe_joint_target: np.ndarray
    safety_reason: str
    row_datasets: Mapping[str, object]
    replay_sidecar_canonical_json: str | None = None
    replay_commitment_sha256: str | None = None

    def validate_for_profile_v1(self, profile: SourceSchemaProfileV1) -> None:
        if self.source_type != profile.source_type:
            raise ValueError("source decision type differs from adapter profile")
        vectors = {
            "raw_policy_action": self.raw_policy_action,
            "safe_action": self.safe_action,
            "requested_joint_target": self.requested_joint_target,
            "queued_safe_joint_target": self.queued_safe_joint_target,
        }
        for name, value in vectors.items():
            _finite_float32_vector(value, name=name)
        if not isinstance(self.safety_reason, str):
            raise TypeError("safety_reason must be a string")
        expected_names = {contract.name for contract in profile.decision_dataset_contracts}
        if set(self.row_datasets) != expected_names:
            raise ValueError("decision source datasets differ from the schema profile")
        for contract in profile.decision_dataset_contracts:
            contract.validate_row_v1(self.row_datasets[contract.name])
        stored_raw = np.asarray(self.row_datasets[profile.raw_policy_action_dataset])
        if not np.array_equal(stored_raw, np.asarray(self.raw_policy_action, dtype=stored_raw.dtype)):
            raise ValueError("raw policy action row differs from dispatch decision")
        if profile.safe_action_diagnostic_alias_dataset is not None:
            alias = np.asarray(self.row_datasets[profile.safe_action_diagnostic_alias_dataset])
            if not np.array_equal(alias, np.asarray(self.safe_action, dtype=alias.dtype)):
                raise ValueError("safe action diagnostic alias differs from dispatch decision")

        if profile.replay_sidecar_required:
            if not isinstance(self.replay_sidecar_canonical_json, str):
                raise ValueError("scratch decision lacks its replay-sidecar record")
            if (
                not isinstance(self.replay_commitment_sha256, str)
                or _SHA256_RE.fullmatch(self.replay_commitment_sha256) is None
            ):
                raise ValueError("replay commitment must be a lowercase SHA-256")
            try:
                record = json.loads(self.replay_sidecar_canonical_json)
            except json.JSONDecodeError as error:
                raise ValueError("replay-sidecar row is not JSON") from error
            if not isinstance(record, Mapping) or record.get("commitment_sha256") != (
                self.replay_commitment_sha256
            ):
                raise ValueError("replay-sidecar row and commitment differ")
            commitment = np.asarray(self.row_datasets[profile.replay_commitment_dataset])
            if bytes(commitment).hex() != self.replay_commitment_sha256:
                raise ValueError("HDF5 replay commitment row differs from sidecar")
        elif self.replay_sidecar_canonical_json is not None or self.replay_commitment_sha256 is not None:
            raise ValueError("source profile does not permit replay-sidecar evidence")


@dataclass(frozen=True, slots=True)
class SourceEffectRequestV1:
    episode: SourceEpisodeContextV1
    row_index: int
    control_step: int
    decision: SourceStepDecisionV1
    effect_info: Mapping[str, Any]
    common_row_evidence: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class SourceEffectAnnotationV1:
    source_type: str
    row_datasets: Mapping[str, object]
    action_label_valid: bool

    def validate_for_profile_v1(self, profile: SourceSchemaProfileV1) -> None:
        if self.source_type != profile.source_type:
            raise ValueError("source effect type differs from adapter profile")
        if type(self.action_label_valid) is not bool:
            raise TypeError("source effect action_label_valid must be boolean")
        expected_names = {contract.name for contract in profile.effect_dataset_contracts}
        if set(self.row_datasets) != expected_names:
            raise ValueError("effect source datasets differ from the schema profile")
        for contract in profile.effect_dataset_contracts:
            contract.validate_row_v1(self.row_datasets[contract.name])


@dataclass(frozen=True, slots=True)
class SourceEpisodeFinalizationV1:
    """Source-only additions returned after the common payload is complete."""

    source_type: str
    payload_updates: Mapping[str, np.ndarray]
    episode_attributes: Mapping[str, Any]
    manifest_fields: Mapping[str, Any]
    action_admission_pass: bool

    def validate_for_profile_v1(self, profile: SourceSchemaProfileV1) -> None:
        if self.source_type != profile.source_type:
            raise ValueError("source finalization type differs from adapter profile")
        if type(self.action_admission_pass) is not bool:
            raise TypeError("action_admission_pass must be boolean")
        if not isinstance(self.payload_updates, Mapping):
            raise TypeError("payload_updates must be a mapping")
        expected_payload_updates = (
            {profile.replay_commitment_dataset} if profile.replay_sidecar_required else set()
        )
        if set(self.payload_updates) != expected_payload_updates:
            raise ValueError("source finalization payload updates differ from the profile")
        if profile.replay_sidecar_required:
            assert profile.replay_commitment_dataset is not None
            commitment = np.asarray(self.payload_updates[profile.replay_commitment_dataset])
            if commitment.ndim != 2 or commitment.shape[0] < 1:
                raise ValueError("replay commitment payload update must have a positive row axis")
            profile.row_contracts_v1()[profile.replay_commitment_dataset].validate_payload_v1(
                commitment,
                row_count=int(commitment.shape[0]),
            )
        for values, required, label in (
            (
                self.episode_attributes,
                profile.dynamic_episode_binding_fields_v1(),
                "episode",
            ),
            (
                self.manifest_fields,
                profile.dynamic_manifest_binding_fields_v1(),
                "manifest",
            ),
        ):
            if not isinstance(values, Mapping):
                raise TypeError(f"source {label} finalization must be a mapping")
            missing = sorted(set(required) - set(values))
            if missing:
                raise ValueError(f"source dynamic {label} bindings are incomplete: {missing}")
        static_episode_overrides = sorted(
            set(self.episode_attributes) & set(profile.collection_episode_binding_fields_v1())
        )
        static_manifest_overrides = sorted(
            set(self.manifest_fields) & set(profile.collection_manifest_binding_fields_v1())
        )
        if static_episode_overrides or static_manifest_overrides:
            raise ValueError("source finalization cannot override collection-static bindings")


@dataclass(frozen=True, slots=True)
class SourcePersistedEpisodeRequestV1:
    """Exact persisted rows supplied after HDF5 commit and before manifest append."""

    dataset_root: Path
    split: str
    episode_index: int
    row_count: int
    root_attributes: Mapping[str, Any]
    episode_attributes: Mapping[str, Any]
    planned_manifest_fields: Mapping[str, Any]
    persisted_source_datasets: Mapping[str, np.ndarray]

    def __post_init__(self) -> None:
        _nonempty_name(self.split, label="persisted episode split")
        for name in ("episode_index", "row_count"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
        if self.episode_index < 0 or self.row_count < 1:
            raise ValueError("persisted episode index/count are invalid")
        for name in (
            "root_attributes",
            "episode_attributes",
            "planned_manifest_fields",
            "persisted_source_datasets",
        ):
            if not isinstance(getattr(self, name), Mapping):
                raise TypeError(f"{name} must be a mapping")


@dataclass(frozen=True, slots=True)
class SourcePersistedEpisodeVerificationV1:
    """Fail-closed source receipt reconstructed from already persisted bytes."""

    source_type: str
    row_count: int
    verified: bool
    episode_binding_values: Mapping[str, Any]
    manifest_binding_values: Mapping[str, Any]

    def validate_for_profile_v1(self, profile: SourceSchemaProfileV1) -> None:
        if self.source_type != profile.source_type:
            raise ValueError("persisted verification source type differs from profile")
        if isinstance(self.row_count, bool) or not isinstance(self.row_count, int) or self.row_count < 1:
            raise ValueError("persisted verification row_count must be positive")
        if self.verified is not True:
            raise ValueError("persisted source verification did not pass")
        for values, expected, label in (
            (
                self.episode_binding_values,
                profile.dynamic_episode_binding_fields_v1(),
                "episode",
            ),
            (
                self.manifest_binding_values,
                profile.dynamic_manifest_binding_fields_v1(),
                "manifest",
            ),
        ):
            if not isinstance(values, Mapping) or set(values) != set(expected):
                raise ValueError(f"persisted dynamic {label} bindings differ from profile")


@runtime_checkable
class CausalCollectionSourceAdapterV1(Protocol):
    """The only source interface called by the shared episode core.

    A V11 adapter and a scratch-PPO adapter implement this same lifecycle.  The
    core never performs an ``isinstance`` or source-name branch.
    """

    @property
    def schema_profile_v1(self) -> SourceSchemaProfileV1: ...

    def prepare_collection_v1(
        self,
        dataset_root: Path,
        *,
        collection_uid: str,
    ) -> SourceCollectionPreparationV1: ...

    def revalidate_collection_v1(self, dataset_root: Path) -> None: ...

    def begin_episode_v1(self, request: SourceEpisodeStartRequestV1) -> None: ...

    def decide_v1(self, request: SourceDecisionRequestV1) -> SourceStepDecisionV1: ...

    def observe_effect_v1(self, request: SourceEffectRequestV1) -> SourceEffectAnnotationV1: ...

    def finalize_episode_v1(
        self,
        context: SourceEpisodeContextV1,
        *,
        common_payload: Mapping[str, np.ndarray],
        common_metadata: Mapping[str, Any],
    ) -> SourceEpisodeFinalizationV1: ...

    def verify_persisted_episode_v1(
        self,
        request: SourcePersistedEpisodeRequestV1,
    ) -> SourcePersistedEpisodeVerificationV1: ...


@dataclass(frozen=True, slots=True)
class CausalEpisodeExecutionRequestV1:
    episode: SourceEpisodeContextV1
    common_collection_context: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class CausalEpisodeExecutionResultV1:
    payload: Mapping[str, np.ndarray]
    metadata: Mapping[str, Any]
    manifest_fields: Mapping[str, Any]


@runtime_checkable
class CausalEpisodeExecutionCoreV1(Protocol):
    """One loop implementation shared by every Causal V2 command source."""

    format: str

    def execute_episode_v1(
        self,
        request: CausalEpisodeExecutionRequestV1,
        source: CausalCollectionSourceAdapterV1,
    ) -> CausalEpisodeExecutionResultV1: ...


__all__ = [
    "CAUSAL_COLLECTION_EXECUTION_CORE_INTERFACE_FORMAT",
    "COMMON_EFFECT_EVIDENCE_FIELDS_V1",
    "COMMON_FINALIZATION_METADATA_FIELDS_V1",
    "EXECUTION_CORE_OWNED_STAGES",
    "EXECUTION_CORE_OPTIONAL_COMMON_DATASETS_V1",
    "EXECUTION_CORE_OPTIONAL_EPISODE_ATTRIBUTES_V1",
    "EXECUTION_CORE_OPTIONAL_MANIFEST_FIELDS_V1",
    "EXECUTION_CORE_OPTIONAL_ROOT_ATTRIBUTES_V1",
    "H5_ROW_DATASET_CONTRACT_FORMAT",
    "SOURCE_ADAPTER_INTERFACE_FORMAT",
    "SOURCE_ADAPTER_OWNED_STAGES",
    "SOURCE_PREPARATION_PROFILE_COMPONENT_FIELDS",
    "SOURCE_SCHEMA_PROVENANCE_FIELDS",
    "SOURCE_SCHEMA_PROFILE_FORMAT",
    "CausalCollectionSourceAdapterV1",
    "CausalEpisodeExecutionCoreV1",
    "CausalEpisodeExecutionRequestV1",
    "CausalEpisodeExecutionResultV1",
    "H5RowDatasetContractV1",
    "SourceCollectionPreparationV1",
    "SourceDecisionRequestV1",
    "SourceEffectAnnotationV1",
    "SourceEffectRequestV1",
    "SourceEpisodeContextV1",
    "SourceEpisodeFinalizationV1",
    "SourceEpisodeStartRequestV1",
    "SourcePersistedEpisodeRequestV1",
    "SourcePersistedEpisodeVerificationV1",
    "SourceSchemaProfileV1",
    "SourceStepDecisionV1",
]
