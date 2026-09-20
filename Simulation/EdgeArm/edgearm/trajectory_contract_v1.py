"""Unified, read-only trajectory schema contract for EdgeArm HDF5 artifacts.

The contract intentionally separates four uses of a trajectory.  A file that is
useful for a world model is not automatically valid action supervision, and a
synthetic file can never pass the physical-deployment contract.  The auditor in
this module normally reads HDF5 attributes, group names, dataset names, dtypes
and shapes.  For causal V2 it additionally indexes only tiny one-dimensional
recovery lineage arrays so a disabled injector cannot be forged by changing row
values.  It never decodes RGB, depth, segmentation, or other large observations.

Grades have the following stable meaning:

``A``
    Complete V1 causal chain and provenance for the requested use.
``B``
    Usable only by an explicit masked/restricted loader; it does not pass the
    formal training gate because V1 audit/diagnostic fields are missing.
``C``
    Partial evidence only.  Keep for diagnostics or a narrower derived use.
``D``
    Ineligible for the requested use, or contradicted by its own provenance.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import importlib
import json
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping

import h5py
import numpy as np

from .action_admission_v1 import (
    ACTION_ADMISSION_GATE_FORMAT,
    EPISODE_INTEGRITY_GATE_FORMAT,
    GENERIC_ACTION_LABEL_DATASET,
    SCRATCH_SOURCE_TYPE,
    SUPPORTED_SOURCE_TYPES,
    V11_SOURCE_TYPE,
    V11_TEACHER_DIAGNOSTIC_ALIAS,
    action_source_registry_json,
)
from .causal_collection_execution_core_v1 import (
    SOURCE_SCHEMA_PROFILE_FORMAT,
    SOURCE_SCHEMA_PROVENANCE_FIELDS,
)
from .config import JOINT_NAMES
from .execution_phase_v1 import (
    EXECUTION_GEOMETRY_FORMAT,
    RECOVERY_EVIDENCE_FORMAT,
    RECOVERY_INTERVENTION_TYPE_IDS,
    RECOVERY_ORIGIN_INTERVENTION_CODE_IDS,
    TERMINAL_FAILURE_CODES,
)
from .reported_wrist_pose_v1 import (
    FIXED_NOMINAL_CALIBRATION_SOURCE,
    REPORTED_WRIST_POSE_FORMAT,
    REPORTED_WRIST_POSE_SEMANTICS,
    SUPPORTED_CALIBRATION_PARAMETER_SOURCES,
)
from .recovery_injection_v1 import (
    DISABLED_RECOVERY_INJECTOR_PROFILE_SHA256,
    RECOVERY_EVENT_LEDGER_FORMAT,
    RECOVERY_INJECTOR_FORMAT,
    TRAJECTORY_ORIGIN_LIVE_PRIMARY_EXECUTION,
    is_sha256,
    validate_event_ledger,
)


CONTRACT_VERSION = "edgearm-trajectory-contract-v1"
CAUSAL_V2_FORMAT = "edgearm-wrist-causal-trajectory-v2"
METRIC_DEPTH_CONTRACT_FORMAT = "edgearm-metric-depth-validity-v1"
FIXED_BLOCK_PUSH_TASK_SCOPE = "fixed_block_push_task_v1"
CAUSAL_V2_CONTROLLER_PREFLIGHT_PROFILE = "edgearm-controller-preflight-v1"
CAUSAL_V2_CONTROLLER_PREFLIGHT_ALIGNMENT = (
    "row t command_applied_id selects the delayed command; "
    "command_applied_queued_safe_joint_target is controller-preflight input; "
    "command_delayed_joint_target is controller-preflight output; "
    "effect_controller_preflight_previewed_servo_endpoint_after_joint_target_rad "
    "equals effect_servo_proposed_joint_target_rad"
)
CAUSAL_V2_COMMAND_APPLIED_SAFETY_STAGE_MAP = json.dumps(
    {
        "0": "submit",
        "1": "apply_including_controller_preflight",
        "2": "runtime_servo",
    },
    sort_keys=True,
    separators=(",", ":"),
)


def valid_metric_depth_contract(value: object) -> bool:
    """Return whether calibration declares fail-closed metric-depth validity."""

    fields = {
        "format",
        "clip_near_m",
        "clip_far_m",
        "valid_min_exclusive_m",
        "valid_max_exclusive_m",
        "invalid_depth_mm",
        "far_plane_no_hit_is_zeroed_before_sensor_noise",
    }
    if not isinstance(value, Mapping) or set(value) != fields:
        return False
    if (
        value.get("format") != METRIC_DEPTH_CONTRACT_FORMAT
        or type(value.get("invalid_depth_mm")) is not int
        or value.get("invalid_depth_mm") != 0
        or value.get("far_plane_no_hit_is_zeroed_before_sensor_noise") is not True
    ):
        return False
    try:
        near = float(value["clip_near_m"])
        far = float(value["clip_far_m"])
        valid_min = float(value["valid_min_exclusive_m"])
        valid_max = float(value["valid_max_exclusive_m"])
    except (TypeError, ValueError):
        return False
    return bool(
        all(np.isfinite(number) for number in (near, far, valid_min, valid_max))
        and 0.0 < near == valid_min < valid_max < far
    )


def valid_reported_wrist_pose_profile(value: object) -> bool:
    """Return whether a policy-visible pose profile is deployable and fail-closed.

    This metadata-only check intentionally does not prove that the declared
    calibration matches a particular episode.  The coverage auditor performs
    that deeper binding and recomputes every stored pose from reported joints.
    """

    if not isinstance(value, Mapping):
        return False
    if (
        value.get("format") != REPORTED_WRIST_POSE_FORMAT
        or value.get("semantics") != REPORTED_WRIST_POSE_SEMANTICS
        or type(value.get("joint_count")) is not int
        or value.get("joint_count") != len(JOINT_NAMES)
        or type(value.get("camera_body_id")) is not int
        or int(value["camera_body_id"]) < 0
        or value.get("calibration_parameter_source") not in SUPPORTED_CALIBRATION_PARAMETER_SOURCES
        or value.get("synthetic_calibration") is not True
        or value.get("physically_calibrated") is not False
        or value.get("simulator_ground_truth_used") is not False
        or value.get("dynamic_simulator_state_used") is not False
        or not is_sha256(str(value.get("profile_sha256", "")))
    ):
        return False
    try:
        mount_position = np.asarray(value["mount_position_m"], dtype=np.float64)
        mount_quaternion = np.asarray(
            value["mount_quaternion_wxyz"],
            dtype=np.float64,
        )
        joint_position_offset = np.asarray(
            value["joint_position_offset_rad"],
            dtype=np.float64,
        )
    except (KeyError, TypeError, ValueError):
        return False
    if (
        mount_position.shape != (3,)
        or mount_quaternion.shape != (4,)
        or joint_position_offset.shape != (len(JOINT_NAMES),)
        or not np.all(np.isfinite(mount_position))
        or not np.all(np.isfinite(mount_quaternion))
        or not np.all(np.isfinite(joint_position_offset))
        or not np.isclose(
            np.linalg.norm(mount_quaternion),
            1.0,
            rtol=0.0,
            atol=1.0e-9,
        )
    ):
        return False
    if value.get("calibration_parameter_source") == FIXED_NOMINAL_CALIBRATION_SOURCE and np.any(
        joint_position_offset != 0.0
    ):
        return False
    return True


CAUSAL_V2_POST_SUBMISSION_REWRITE_REASON_MAP = json.dumps(
    {
        "1": "runtime_dynamic_clearance_projection",
        "2": "v11_invalid_contact_escape",
        "4": "controller_preflight_dynamic_clearance_projection",
        str(1 << 31): "unknown_fail_closed",
    },
    sort_keys=True,
    separators=(",", ":"),
)
CAUSAL_V2_CONTACT_TELEMETRY_STAGE_MAP = {
    "effect_kinematic_stage": ("post_integration_qpos_plus_independent_scratch_mj_forward"),
    "forecast_clearance_stage": ("post_each_forecast_integration_qpos_plus_scratch_mj_fwdPosition"),
    "kinematic_scratch_contact_force_used": False,
    "solver_contact_force_source": ("live_mj_step_solver_contacts_without_position_cache_rewrite"),
    "solver_contact_interval": "substep_start_to_substep_end",
    "solver_contact_stage": ("mj_step_solver_contacts_before_final_position_cache_refresh"),
    "tool_desk_signed_distance_stage": "effect_kinematic_stage",
}
CAUSAL_V2_CONTACT_TELEMETRY_STAGE_MAP_JSON = json.dumps(
    CAUSAL_V2_CONTACT_TELEMETRY_STAGE_MAP,
    sort_keys=True,
    separators=(",", ":"),
)
CAUSAL_V2_EXECUTION_GEOMETRY_SEMANTICS = (
    "post_env_step_source_neutral_geometry_from_tool_block_target_and_geom_metadata; "
    "teacher_or_expert_height_not_used; simulator_privileged_and_forbidden_as_policy_input"
)
CAUSAL_V2_EXECUTION_GEOMETRY_HEIGHT_FORMULA = (
    "desk_top_m + orientation_invariant_tool_half_diagonal_m + declared_runtime_clearance_m"
)
CAUSAL_V2_RECOVERY_EVIDENCE_SEMANTICS = (
    "only explicitly scripted nonterminal interventions may open a recovery epoch; "
    "random transport loss, runtime safety holds, and terminal failures are diagnostics "
    "and never verified recovery origins"
)
CAUSAL_V2_RECOVERY_INTERVENTION_TYPE_ID_MAP = json.dumps(
    dict(RECOVERY_INTERVENTION_TYPE_IDS),
    sort_keys=True,
    separators=(",", ":"),
)
CAUSAL_V2_RECOVERY_ORIGIN_INTERVENTION_CODE_ID_MAP = json.dumps(
    dict(RECOVERY_ORIGIN_INTERVENTION_CODE_IDS),
    sort_keys=True,
    separators=(",", ":"),
)
CAUSAL_V2_V11_PRODUCER_BINDING_ATTRS = (
    "producer_artifact_kind",
    "producer_artifact_manifest_path",
    "producer_artifact_manifest_sha256",
    "runtime_asset_sha256",
    "producer_runtime_closure_sha256",
    "policy_artifact_sha256",
)
CAUSAL_V2_V11_EXECUTION_EVIDENCE_ATTRS = ("collection_execution_evidence_level",)
CAUSAL_V2_V11_EPISODE_REQUEST_EVIDENCE_ATTRS = ("frozen_worker_request_sha256",)
CAUSAL_V2_TERMINAL_FAILURE_CODE_TAXONOMY = json.dumps(
    TERMINAL_FAILURE_CODES,
    separators=(",", ":"),
)
CAUSAL_V2_EXECUTION_DATASETS = (
    "decision_safe_action",
    "command_policy_dispatched_action",
    "command_original_action",
    "command_submitted_action",
    "command_submitted_id",
    "command_applied_id",
    "command_applied_submitted_action",
    "command_queued_safe_joint_target",
    "command_applied_queued_safe_joint_target",
    "command_delayed_joint_target",
    "command_safety_target_changed_mask",
    "command_applied_safety_stage_mask",
    "command_applied_safety_target_changed_mask",
    "command_runtime_target_changed_mask",
    "command_submitted_send_step",
    "command_submitted_send_time_seconds",
    "command_applied_send_step",
    "command_applied_send_time_seconds",
    "command_apply_step",
    "command_apply_time_seconds",
    "command_actual_delay_steps",
    "command_actual_delay_seconds",
    "command_submitted_ingress_lost",
    "command_applied_ingress_lost",
    "command_ack_available",
    "command_synthetic_application_truth",
    "decision_physical_joint_position",
    "effect_physical_joint_position",
    "effect_executed_joint_delta_rad",
    "effect_executed_joint_delta_normalized",
    "effect_reported_executed_joint_delta_rad",
    "effect_reported_executed_joint_delta_normalized",
    "effect_reported_executed_action_measurement_valid",
    "effect_execution_phase_id",
    "effect_execution_phase_valid",
    "effect_execution_phase_transition",
    "effect_execution_phase_evidence_mask",
    "effect_execution_tool_geom_center_xyz_m",
    "effect_execution_desk_geom_center_xyz_m",
    "effect_execution_desk_geom_rotation_matrix_flat",
    "effect_execution_desk_geom_half_extents_m",
    "effect_execution_tool_geom_half_extents_m",
    "effect_execution_tool_block_direction_xy",
    "effect_execution_tool_block_along_m",
    "effect_execution_tool_block_lateral_m",
    "effect_execution_tool_height_error_m",
    "effect_execution_operational_tool_height_reference_m",
    "effect_execution_desk_top_m",
    "effect_execution_orientation_invariant_tool_half_diagonal_m",
    "effect_execution_declared_runtime_clearance_m",
    "effect_execution_recovery_epoch_id",
    "effect_execution_recovery_epoch_start_transition_index",
    "effect_execution_recovery_epoch_step_index",
    "effect_recovery_intervention_event_index",
    "effect_recovery_intervention_type_id",
    "effect_recovery_injection_event_index",
    "effect_recovery_injection_requested",
    "effect_recovery_injection_hold_substeps_mask",
    "effect_execution_recovery_origin_intervention_code_id",
    "effect_valid_push_side_contact_any",
    "effect_block_xy_displacement_m",
    "effect_minimum_pusher_desk_signed_distance_m",
    "effect_runtime_guard_static_infeasible_count",
    "effect_runtime_guard_dynamic_infeasible_count",
    "effect_runtime_guard_safety_stop",
    "effect_servo_proposed_joint_target_rad",
    "effect_controller_preflight_previewed_servo_endpoint_before_joint_target_rad",
    "effect_controller_preflight_previewed_servo_endpoint_after_joint_target_rad",
    "effect_controller_preflight_dynamic_forecast_feasible",
    "effect_delayed_controller_joint_target_rad",
    "effect_physics_applied_joint_target_rad",
    "effect_substep_guard_input_joint_target_rad",
    "effect_substep_static_projected_joint_target_rad",
    "effect_substep_physics_applied_joint_target_rad",
    "effect_substep_static_projection_changed_mask",
    "effect_substep_dynamic_projection_changed_mask",
    "effect_runtime_guard_action_modified",
    "effect_guard_queue_rewrite_affected_command_id",
    "effect_guard_queue_rewrite_valid_mask",
    "effect_guard_queue_rewrite_target_before_rad",
    "effect_guard_queue_rewrite_target_after_rad",
    "effect_guard_queue_rewrite_target_changed_mask",
    "effect_guard_queue_rewrite_event_count",
    "effect_guard_queue_mutated",
    "command_applied_post_submission_rewrite_count",
    "command_applied_post_submission_rewrite_reason_mask",
    "command_applied_post_submission_target_changed_mask",
    "effect_command_feedback_available",
    "effect_command_feedback_epoch",
    "effect_command_feedback_transition_step",
    "effect_command_feedback_submitted_id",
    "effect_command_feedback_applied_id",
    "effect_command_feedback_next_command_id",
    "effect_command_feedback_applied_ingress_lost",
    "effect_command_feedback_applied_is_virtual_hold",
    "effect_command_feedback_pending_virtual_hold_count",
    "effect_command_feedback_applied_action_was_safety_modified",
    "decision_previous_command_feedback_available",
    "decision_previous_command_feedback_epoch",
    "decision_previous_command_feedback_transition_step",
    "decision_previous_command_feedback_submitted_id",
    "decision_previous_command_feedback_applied_id",
    "decision_previous_command_feedback_next_command_id",
    "decision_previous_command_feedback_applied_ingress_lost",
    "decision_previous_command_feedback_applied_is_virtual_hold",
    "decision_previous_command_feedback_pending_virtual_hold_count",
    "decision_previous_command_feedback_applied_action_was_safety_modified",
    "decision_previous_executed_joint_delta_rad",
    "decision_previous_executed_joint_delta_normalized",
    "decision_previous_executed_action_valid",
    "decision_previous_reported_executed_joint_delta_rad",
    "decision_previous_reported_executed_joint_delta_normalized",
    "decision_previous_reported_executed_action_valid",
    "policy_intent_action_label_valid",
    "policy_intent_execution_unmodified_mask",
    "effect_executed_action_measurement_valid",
    "effect_policy_intent_action_training_mask",
    "effect_unmodified_policy_execution_training_mask",
    "effect_action_training_mask",
)
CAUSAL_V2_V11_DIAGNOSTIC_DATASETS = (
    V11_TEACHER_DIAGNOSTIC_ALIAS,
    "diagnostic_teacher_precontact_plan_feasible",
    "diagnostic_teacher_precontact_completion_event",
    "diagnostic_teacher_precontact_completed_once",
    "effect_applied_parent_precontact_completed_once",
    "effect_causal_valid_push_transition",
)
CAUSAL_V2_ROOT_ATTRS = (
    "task_scope",
    "trajectory_origin",
    "recovery_injector_format",
    "recovery_injector_profile_sha256",
    "recovery_injector_source_sha256",
    "recovery_event_ledger_format",
    "camera_pose_policy_format",
    "camera_pose_policy_semantics",
    "camera_pose_policy_simulator_ground_truth_used",
    "execution_phase_format",
    "execution_phase_names",
    "execution_phase_semantics",
    "execution_phase_threshold_profile_hash",
    "execution_geometry_format",
    "execution_geometry_semantics",
    "execution_geometry_height_reference_formula",
    "execution_geometry_declared_runtime_clearance_m",
    "recovery_evidence_format",
    "recovery_evidence_semantics",
    "recovery_intervention_type_id_map",
    "recovery_origin_intervention_code_id_map",
    "terminal_failure_codes",
    "formal_recovery_intervention_injection_enabled",
    "episode_integrity_gate_format",
    "episode_integrity_gate_required",
    "action_admission_gate_format",
    "action_admission_gate_required",
    "action_admission_source_registry",
    "act_primary_label_dataset",
    "act_action_history_dataset",
    "act_action_history_valid_dataset",
    "act_action_history_alignment",
    "act_runtime_shield_required",
    "pre_guard_target_is_physics_applied_truth",
    "command_feedback_profile",
    "command_feedback_semantics",
    "command_feedback_alignment",
    "controller_preflight_profile",
    "controller_preflight_alignment",
    "command_applied_safety_stage_map",
    "command_applied_post_submission_rewrite_reason_map",
    "solver_contact_stage",
    "solver_contact_interval",
    "effect_kinematic_stage",
    "tool_desk_signed_distance_stage",
    "kinematic_scratch_contact_force_used",
    "forecast_clearance_stage",
    "solver_contact_force_source",
    "contact_telemetry_stage_map",
    "max_joint_delta_rad",
    *SOURCE_SCHEMA_PROVENANCE_FIELDS,
)
CAUSAL_V2_EPISODE_ATTRS = (
    "task_scope",
    "trajectory_origin",
    "recovery_injector_format",
    "recovery_injector_profile_sha256",
    "recovery_injector_source_sha256",
    "recovery_event_ledger_format",
    "camera_pose_policy_format",
    "camera_pose_policy_semantics",
    "camera_pose_policy_simulator_ground_truth_used",
    "episode_integrity_gate_format",
    "episode_integrity_pass",
    "action_admission_gate_format",
    "action_admission_pass",
    "action_admission_gate_evidence",
    "action_admission_reason_codes",
    "execution_phase_format",
    "execution_phase_names",
    "execution_phase_semantics",
    "execution_geometry_format",
    "execution_geometry_semantics",
    "execution_geometry_height_reference_formula",
    "execution_geometry_declared_runtime_clearance_m",
    "recovery_evidence_format",
    "recovery_evidence_semantics",
    "recovery_intervention_type_id_map",
    "recovery_origin_intervention_code_id_map",
    "terminal_failure_codes",
    "recovery_origin_intervention_codes",
    "recovery_intervention_types",
    "recovery_epoch_count",
    "recovery_intervention_event_count",
    "formal_recovery_evidence_available",
    "formal_recovery_intervention_injection_enabled",
    "act_primary_label_dataset",
    "act_action_history_dataset",
    "act_action_history_valid_dataset",
    "act_action_history_alignment",
    "act_runtime_shield_required",
    "command_feedback_profile",
    "command_feedback_semantics",
    "command_feedback_alignment",
    "controller_preflight_profile",
    "controller_preflight_alignment",
    "command_applied_safety_stage_map",
    "command_applied_post_submission_rewrite_reason_map",
    "solver_contact_stage",
    "solver_contact_interval",
    "effect_kinematic_stage",
    "tool_desk_signed_distance_stage",
    "kinematic_scratch_contact_force_used",
    "forecast_clearance_stage",
    "solver_contact_force_source",
    "contact_telemetry_stage_map",
    "max_joint_delta_rad",
    *SOURCE_SCHEMA_PROVENANCE_FIELDS,
)
CAUSAL_V2_V11_ROOT_ATTRS = (
    "promotion_gate_format",
    "promotion_gate_profile_hash",
    "v11_teacher_diagnostic_alias_dataset",
    "diagnostic_teacher_planned_phase_names",
    "planned_phase_semantics",
    *CAUSAL_V2_V11_PRODUCER_BINDING_ATTRS[:-1],
    *CAUSAL_V2_V11_EXECUTION_EVIDENCE_ATTRS,
)
CAUSAL_V2_V11_EPISODE_ATTRS = (
    "promotion_valid_success",
    "promotion_gate_format",
    "promotion_gate_evidence",
    "v11_teacher_diagnostic_alias_dataset",
    *CAUSAL_V2_V11_PRODUCER_BINDING_ATTRS[:-1],
    *CAUSAL_V2_V11_EXECUTION_EVIDENCE_ATTRS,
    *CAUSAL_V2_V11_EPISODE_REQUEST_EVIDENCE_ATTRS,
)


class Purpose(str, Enum):
    CAUSAL_MOTION = "causal_motion"
    ACTION_SUPERVISION = "action_supervision"
    WORLD_MODEL = "world_model"
    PHYSICAL_DEPLOYMENT = "physical_deployment"


class Grade(str, Enum):
    A = "A"
    B = "B"
    C = "C"
    D = "D"


@dataclass(frozen=True)
class ContractSpec:
    hard_fields: tuple[str, ...]
    complete_fields: tuple[str, ...]
    complete_root_attrs: tuple[str, ...]
    complete_episode_attrs: tuple[str, ...]


# Canonical fields are semantic roles, not promises that similarly named legacy
# arrays have the same meaning.  In particular, legacy ``action_joint_delta`` is
# promoted to an executed action only when an executed-safe contract is present.
FIELD_ALIASES: Mapping[str, tuple[str, ...]] = {
    "rgb_wrist": ("rgb_wrist",),
    "depth_wrist": ("depth_wrist_mm", "depth_wrist"),
    "segmentation_wrist": ("segmentation_wrist",),
    "camera_pose_wrist": ("camera_pose_wrist",),
    "camera_source_frame": ("camera_source_frame_index",),
    "camera_delivered_frame": ("camera_delivered_frame_index",),
    "camera_device_timestamp": ("camera_device_timestamp_ns", "camera_device_timestamp"),
    "camera_host_timestamp": (
        "camera_host_timestamp_ns",
        "camera_host_timestamp",
        "camera_delivered_timestamp",
    ),
    "camera_device_period": ("camera_device_period_seconds",),
    "camera_host_period": ("camera_host_period_seconds",),
    "camera_latency_state": (
        "camera_configured_latency_steps",
        "camera_delivered_frame_age_steps",
    ),
    "camera_rolling_shutter_state": (
        "camera_rolling_shutter_used_previous_frame",
        "camera_rolling_shutter_readout_fraction",
    ),
    "camera_rolling_shutter_dual_endpoint_complete": ("camera_rolling_shutter_dual_endpoint_complete",),
    "camera_4d_reconstructable_mask": ("camera_4d_reconstructable_mask",),
    "camera_pose_wrist_previous_endpoint": ("camera_pose_wrist_previous_endpoint",),
    "camera_pose_wrist_current_endpoint": ("camera_pose_wrist_current_endpoint",),
    "camera_exposure_previous_endpoint_joint_position": (
        "exposure_previous_endpoint_reported_joint_position",
        "exposure_previous_endpoint_physical_joint_position",
    ),
    "camera_exposure_current_endpoint_joint_position": (
        "exposure_current_endpoint_reported_joint_position",
        "exposure_current_endpoint_physical_joint_position",
    ),
    "camera_exposure_previous_endpoint_joint_velocity": (
        "exposure_previous_endpoint_reported_joint_velocity",
        "exposure_previous_endpoint_physical_joint_velocity",
    ),
    "camera_exposure_current_endpoint_joint_velocity": (
        "exposure_current_endpoint_reported_joint_velocity",
        "exposure_current_endpoint_physical_joint_velocity",
    ),
    "camera_exposure_previous_endpoint_tool_pose": ("exposure_previous_endpoint_tool_pose",),
    "camera_exposure_current_endpoint_tool_pose": ("exposure_current_endpoint_tool_pose",),
    "camera_exposure_previous_endpoint_tool_twist": ("exposure_previous_endpoint_tool_twist_linear_angular",),
    "camera_exposure_current_endpoint_tool_twist": ("exposure_current_endpoint_tool_twist_linear_angular",),
    "camera_exposure_previous_endpoint_object_state": ("exposure_previous_endpoint_block_pose_xyz_wxyz",),
    "camera_exposure_current_endpoint_object_state": ("exposure_current_endpoint_block_pose_xyz_wxyz",),
    "camera_exposure_previous_endpoint_object_twist": (
        "exposure_previous_endpoint_block_twist_linear_angular",
    ),
    "camera_exposure_current_endpoint_object_twist": (
        "exposure_current_endpoint_block_twist_linear_angular",
    ),
    "camera_geometry_alignment_mask": (
        "camera_geometry_alignment_exact",
        "camera_world_model_training_mask",
    ),
    "camera_exposure_joint_position": (
        "exposure_reported_joint_position",
        "exposure_physical_joint_position",
        "camera_exposure_reported_joint_position",
        "camera_exposure_physical_joint_position",
    ),
    "camera_exposure_joint_velocity": (
        "exposure_reported_joint_velocity",
        "exposure_physical_joint_velocity",
        "camera_exposure_reported_joint_velocity",
        "camera_exposure_physical_joint_velocity",
    ),
    "camera_exposure_tool_pose": (
        "exposure_tool_pose",
        "camera_exposure_tool_pose",
    ),
    "joint_position": (
        "decision_physical_joint_position",
        "decision_reported_joint_position",
        "physical_joint_position",
        "reported_joint_position",
        "joint_position",
        "observed_joint_position",
    ),
    "joint_velocity": (
        "decision_physical_joint_velocity",
        "decision_reported_joint_velocity",
        "physical_joint_velocity",
        "reported_joint_velocity",
        "joint_velocity",
        "actual_joint_velocity_rad_s",
        "observed_joint_velocity",
    ),
    "joint_acceleration": (
        "decision_physical_joint_acceleration",
        "joint_acceleration",
    ),
    "next_joint_position": (
        "effect_physical_joint_position",
        "effect_reported_joint_position",
        "next_physical_joint_position",
        "next_reported_joint_position",
        "next_joint_position",
    ),
    "next_joint_velocity": (
        "effect_physical_joint_velocity",
        "effect_reported_joint_velocity",
        "next_physical_joint_velocity",
        "next_reported_joint_velocity",
        "next_joint_velocity",
    ),
    "next_joint_acceleration": (
        "effect_physical_joint_acceleration",
        "next_joint_acceleration",
    ),
    "tool_pose": ("decision_tool_pose", "tool_pose", "ee_pose"),
    "next_tool_pose": ("effect_tool_pose", "next_tool_pose", "next_ee_pose"),
    "tool_twist": ("decision_tool_twist_linear_angular", "tool_twist"),
    "next_tool_twist": ("effect_tool_twist_linear_angular", "next_tool_twist"),
    "planned_action": (
        "teacher_reference_action",
        "planned_reference_joint_delta",
        "plan_joint_delta",
        "baseline_action_joint_delta",
        "policy_action_joint_delta",
    ),
    "requested_action": (
        "teacher_reference_action",
        "command_original_action",
        "policy_requested_joint_delta",
        "teacher_requested_command",
        "student_requested_command",
        "requested_command_joint_delta",
        "requested_action_joint_delta",
    ),
    "safety_projected_action": (
        "decision_safe_action",
        "teacher_safe_action",
        "command_policy_dispatched_action",
        "safety_projected_joint_delta",
        "teacher_safe_command",
        "student_safe_command",
        "student_final_queued_safe_command",
        "student_queued_safe_command",
        "teacher_queued_safe_command",
    ),
    "command_sent": (
        "command_submitted_action",
        "command_sent_joint_delta",
        "student_applied_command",
        "applied_command",
    ),
    "command_application_truth": (
        "command_applied_id",
        "applied_command_id",
        "command_applied_submitted_action",
        "command_synthetic_application_truth",
    ),
    "command_acknowledged": (
        "command_acknowledged",
        "command_accepted",
        "command_receipt",
    ),
    "command_ack_available": ("command_ack_available",),
    "command_parent_observation": (
        "command_parent_observation_frame_index",
        "command_parent_decision_step",
    ),
    "command_hold_duration": (
        "command_hold_duration_steps",
        "command_hold_duration_seconds",
    ),
    "command_control_mode": ("command_control_mode_id", "command_control_mode"),
    "submitted_command_id": ("command_submitted_id", "submitted_command_id"),
    "applied_command_id": ("command_applied_id", "applied_command_id"),
    "applied_command_target": (
        "effect_physics_applied_joint_target_rad",
        "effect_substep_physics_applied_joint_target_rad",
        "effect_servo_proposed_joint_target_rad",
        "command_delayed_joint_target",
        "command_applied_queued_safe_joint_target",
        "delayed_joint_target",
        "applied_joint_target",
    ),
    "executed_action": (
        "effect_executed_joint_delta_normalized",
        "effect_executed_joint_delta_rad",
        "measured_joint_delta",
        "executed_joint_delta",
        "executed_joint_delta_rad",
        "reported_executed_normalized_delta",
        "student_executed_transition",
    ),
    "tracking_error": (
        "effect_tracking_error_physics_applied_target_minus_position_rad",
        "effect_tracking_error_joint_target_minus_position_rad",
        "tracking_error",
    ),
    "supervision_action": (
        "decision_safe_action",
        "teacher_safe_action",
        "action_label",
        "correction_target",
        "teacher_safe_command",
        "teacher_queued_safe_command",
        "measured_joint_delta",
        "executed_joint_delta",
        "reported_executed_normalized_delta",
        "student_executed_transition",
    ),
    "action_label_valid": (
        "policy_intent_action_label_valid",
        "action_label_valid",
        "action_mask",
        "teacher_qualified",
    ),
    "reward": ("effect_reward", "reward"),
    "safety_cost": ("effect_safety_cost", "safety_cost"),
    "done": ("effect_terminated", "effect_truncated", "done", "episode_end", "terminal"),
    "terminated": ("effect_terminated", "terminated"),
    "truncated": ("effect_truncated", "truncated"),
    "timestamp": (
        "decision_control_time_seconds",
        "timestamp",
        "control_timestamp",
        "monotonic_timestamp_ns",
    ),
    "effect_timestamp": ("effect_control_time_seconds", "next_timestamp"),
    "strict_outcome_state": (
        "effect_success",
        "effect_strict_success_streak",
    ),
    "target_containment_state": (
        "effect_strict_target_coverage",
        "effect_strict_contained",
    ),
    "settling_state": (
        "effect_strict_settled",
        "effect_block_linear_speed_m_s",
    ),
    "contact_state": (
        "effect_contact_count",
        "contact_count",
        "contact_state",
        "obstacle_contact_block",
        "collision_failure",
    ),
    "contact_geometry": (
        "effect_tool_block_contact_point_xyz",
        "contact_point_xyz",
    ),
    "contact_wrench": (
        "effect_tool_block_contact_wrench_world_frame",
        "effect_tool_block_contact_wrench_contact_frame",
        "contact_wrench",
    ),
    "safety_state": (
        "command_safety_stage_mask",
        "command_safety_target_changed_mask",
        "safety_clipped",
        "teacher_safety_clipped",
        "student_execution_safety_clipped",
        "student_safety_clipped",
        "command_lost",
    ),
    "failure_state": (
        "effect_failure_code",
        "effect_collision_failure",
        "failure",
        "collision_failure",
        "command_lost",
    ),
    "object_state": (
        "decision_block_pose_xyz_wxyz",
        "exposure_block_pose_xyz_wxyz",
        "effect_block_pose_xyz_wxyz",
        "decision_block_pose",
        "exposure_block_pose",
        "effect_block_pose",
        "block_pose",
        "block_xy",
        "privileged_state",
        "entity_state",
    ),
    "object_twist": (
        "decision_block_twist_linear_angular",
        "exposure_block_twist_linear_angular",
        "effect_block_twist_linear_angular",
        "decision_block_twist",
        "exposure_block_twist",
        "effect_block_twist",
        "block_twist",
    ),
    "policy_source_step": ("teacher_source_id", "policy_source_id", "controller_source_id"),
    "policy_intervention": (
        "effect_policy_source_switch",
        "effect_intervention_type_id",
    ),
    "action_training_mask": ("effect_action_training_mask", "action_training_mask"),
    "world_model_training_mask": (
        "effect_world_model_training_mask",
        "camera_world_model_training_mask",
        "world_model_training_mask",
    ),
    "actuator_effort": ("actuator_force_nm", "motor_current_a", "joint_effort"),
    "motor_temperature": ("motor_temperature_c",),
    "supply_voltage": ("supply_voltage_v", "loaded_voltage_v"),
}


COMMON_COMPLETE_ROOT_ATTRS = (
    "format",
    "trajectory_contract_version",
    "source_type",
    "physical_samples",
    "physical_hardware_connected",
    "synthetic_camera",
    "collection_provenance",
    "robot_model",
    "joint_order",
    "joint_position_unit",
    "joint_velocity_unit",
    "joint_acceleration_unit",
    "action_unit",
    "controller_version",
    "bus_protocol",
    "command_ack_semantics",
    "clock_domains",
    "camera_exposure_state_contract",
    "rolling_shutter_state_contract",
    "rolling_shutter_geometry_model_id",
    "rolling_shutter_readout_direction",
    "rolling_shutter_alpha_formula",
    "world_model_loader_supports_rolling_shutter_geometry_model",
    "contact_measurement_semantics",
    "observable_privileged_boundary",
)
COMMON_COMPLETE_EPISODE_ATTRS = (
    "seed",
    "success",
    "task",
    "source_type",
    "controller_identity",
    "action_label_eligible",
    "unique_live_episode_key",
    "unique_live_episode_identity",
    "domain",
    "scenario_id",
    "strict_success",
    "failure_code",
    "robot_model",
    "robot_description_format",
    "robot_description_sha256",
    "compiled_model_sha256",
    "model_profile_sha256",
    "joint_order",
    "joint_position_unit",
    "joint_velocity_unit",
    "joint_acceleration_unit",
    "action_unit",
    "firmware_version",
    "bus_protocol",
    "command_ack_semantics",
    "controller_version",
    "controller_source_sha256",
    "policy_checkpoint_status",
    "policy_artifact_sha256",
    "camera_calibration_sha256",
    "camera_extrinsic_sha256",
    "randomization_profile_sha256",
    "randomization_realization_sha256",
    "nominal_control_period_seconds",
    "clock_domains",
    "intervention_summary",
    "observable_privileged_boundary",
    "contact_measurement_semantics",
)

CAUSAL_COMPLETE_ROOT_ATTRS = COMMON_COMPLETE_ROOT_ATTRS + (
    "causal_action_contract",
    "temporal_alignment_contract",
)


CONTRACTS: Mapping[Purpose, ContractSpec] = {
    Purpose.CAUSAL_MOTION: ContractSpec(
        hard_fields=(
            "rgb_wrist",
            "camera_exposure_joint_position",
            "joint_position",
            "requested_action",
            "safety_projected_action",
            "command_sent",
            "submitted_command_id",
            "applied_command_id",
            "applied_command_target",
            "executed_action",
            "next_joint_position",
            "terminated",
            "truncated",
        ),
        complete_fields=(
            "planned_action",
            "command_application_truth",
            "command_ack_available",
            "command_parent_observation",
            "command_hold_duration",
            "command_control_mode",
            "policy_source_step",
            "camera_exposure_joint_velocity",
            "camera_exposure_tool_pose",
            "joint_velocity",
            "joint_acceleration",
            "next_joint_velocity",
            "next_joint_acceleration",
            "tool_pose",
            "next_tool_pose",
            "tool_twist",
            "next_tool_twist",
            "tracking_error",
            "reward",
            "safety_cost",
            "timestamp",
            "effect_timestamp",
            "strict_outcome_state",
            "target_containment_state",
            "settling_state",
            "camera_pose_wrist",
            "camera_source_frame",
            "camera_delivered_frame",
            "camera_device_timestamp",
            "camera_host_timestamp",
            "camera_device_period",
            "camera_host_period",
            "camera_latency_state",
            "camera_rolling_shutter_state",
            "camera_rolling_shutter_dual_endpoint_complete",
            "camera_4d_reconstructable_mask",
            "camera_pose_wrist_previous_endpoint",
            "camera_pose_wrist_current_endpoint",
            "camera_exposure_previous_endpoint_joint_position",
            "camera_exposure_current_endpoint_joint_position",
            "camera_exposure_previous_endpoint_joint_velocity",
            "camera_exposure_current_endpoint_joint_velocity",
            "camera_exposure_previous_endpoint_tool_pose",
            "camera_exposure_current_endpoint_tool_pose",
            "camera_exposure_previous_endpoint_tool_twist",
            "camera_exposure_current_endpoint_tool_twist",
            "camera_exposure_previous_endpoint_object_state",
            "camera_exposure_current_endpoint_object_state",
            "camera_exposure_previous_endpoint_object_twist",
            "camera_exposure_current_endpoint_object_twist",
            "camera_geometry_alignment_mask",
            "contact_state",
            "contact_geometry",
            "contact_wrench",
            "safety_state",
            "failure_state",
            "policy_intervention",
            "action_training_mask",
            "world_model_training_mask",
        ),
        complete_root_attrs=CAUSAL_COMPLETE_ROOT_ATTRS,
        complete_episode_attrs=COMMON_COMPLETE_EPISODE_ATTRS,
    ),
    Purpose.ACTION_SUPERVISION: ContractSpec(
        hard_fields=(
            "rgb_wrist",
            "joint_position",
            "supervision_action",
            "next_joint_position",
            "done",
        ),
        complete_fields=(
            "joint_velocity",
            "executed_action",
            "requested_action",
            "submitted_command_id",
            "applied_command_id",
            "applied_command_target",
            "camera_exposure_joint_position",
            "action_label_valid",
            "reward",
            "timestamp",
            "camera_source_frame",
            "camera_delivered_frame",
            "action_training_mask",
        ),
        complete_root_attrs=CAUSAL_COMPLETE_ROOT_ATTRS + ("label_contract",),
        complete_episode_attrs=COMMON_COMPLETE_EPISODE_ATTRS,
    ),
    Purpose.WORLD_MODEL: ContractSpec(
        hard_fields=(
            "rgb_wrist",
            "camera_exposure_joint_position",
            "joint_position",
            "applied_command_target",
            "next_joint_position",
            "terminated",
            "truncated",
        ),
        complete_fields=(
            "depth_wrist",
            "segmentation_wrist",
            "camera_pose_wrist",
            "camera_source_frame",
            "camera_delivered_frame",
            "camera_device_timestamp",
            "camera_host_timestamp",
            "camera_device_period",
            "camera_host_period",
            "camera_latency_state",
            "camera_rolling_shutter_state",
            "camera_rolling_shutter_dual_endpoint_complete",
            "camera_4d_reconstructable_mask",
            "camera_pose_wrist_previous_endpoint",
            "camera_pose_wrist_current_endpoint",
            "camera_exposure_previous_endpoint_joint_position",
            "camera_exposure_current_endpoint_joint_position",
            "camera_exposure_previous_endpoint_joint_velocity",
            "camera_exposure_current_endpoint_joint_velocity",
            "camera_exposure_previous_endpoint_tool_pose",
            "camera_exposure_current_endpoint_tool_pose",
            "camera_exposure_previous_endpoint_tool_twist",
            "camera_exposure_current_endpoint_tool_twist",
            "camera_exposure_previous_endpoint_object_state",
            "camera_exposure_current_endpoint_object_state",
            "camera_exposure_previous_endpoint_object_twist",
            "camera_exposure_current_endpoint_object_twist",
            "camera_geometry_alignment_mask",
            "camera_exposure_joint_velocity",
            "camera_exposure_tool_pose",
            "joint_velocity",
            "next_joint_velocity",
            "tool_pose",
            "next_tool_pose",
            "tool_twist",
            "next_tool_twist",
            "object_state",
            "object_twist",
            "reward",
            "safety_cost",
            "timestamp",
            "tracking_error",
            "world_model_training_mask",
            "effect_timestamp",
            "strict_outcome_state",
            "target_containment_state",
            "settling_state",
        ),
        complete_root_attrs=CAUSAL_COMPLETE_ROOT_ATTRS,
        complete_episode_attrs=COMMON_COMPLETE_EPISODE_ATTRS + ("camera_calibration",),
    ),
    Purpose.PHYSICAL_DEPLOYMENT: ContractSpec(
        hard_fields=(
            "rgb_wrist",
            "camera_device_timestamp",
            "camera_host_timestamp",
            "camera_exposure_joint_position",
            "joint_position",
            "joint_velocity",
            "requested_action",
            "safety_projected_action",
            "command_sent",
            "command_acknowledged",
            "command_application_truth",
            "command_ack_available",
            "submitted_command_id",
            "applied_command_id",
            "applied_command_target",
            "executed_action",
            "next_joint_position",
            "terminated",
            "truncated",
        ),
        complete_fields=(
            "depth_wrist",
            "camera_pose_wrist",
            "camera_source_frame",
            "camera_delivered_frame",
            "camera_exposure_joint_velocity",
            "camera_exposure_tool_pose",
            "next_joint_velocity",
            "tool_pose",
            "next_tool_pose",
            "contact_state",
            "safety_state",
            "failure_state",
            "actuator_effort",
            "motor_temperature",
            "supply_voltage",
        ),
        complete_root_attrs=CAUSAL_COMPLETE_ROOT_ATTRS
        + (
            "physical_camera_data",
            "physically_calibrated",
            "calibration_id",
            "robot_serial",
            "controller_identity",
        ),
        complete_episode_attrs=COMMON_COMPLETE_EPISODE_ATTRS,
    ),
}


def _scalar(value: Any) -> Any:
    """Convert an HDF5 scalar attribute to a small JSON-compatible value."""

    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if hasattr(value, "item"):
        try:
            return value.item()
        except (ValueError, AttributeError):
            pass
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    # Large array-valued attributes are not part of this schema summary.
    return str(value)


def _bool_attr(attrs: Mapping[str, Any], name: str) -> bool | None:
    if name not in attrs:
        return None
    value = _scalar(attrs[name])
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes"}:
            return True
        if lowered in {"false", "0", "no", ""}:
            return False
    try:
        return bool(int(value))
    except (TypeError, ValueError):
        return bool(value)


def _int_attr(attrs: Mapping[str, Any], name: str) -> int | None:
    if name not in attrs:
        return None
    try:
        return int(_scalar(attrs[name]))
    except (TypeError, ValueError):
        return None


def _dataset_names(group: h5py.Group) -> set[str]:
    return {name for name, item in group.items() if isinstance(item, h5py.Dataset)}


def _primary_group(episode: h5py.Group) -> h5py.Group:
    if _dataset_names(episode):
        return episode
    live = episode.get("live")
    if isinstance(live, h5py.Group):
        return live
    return episode


def _episode_groups(stream: h5py.File) -> list[h5py.Group]:
    episodes = [
        item for name, item in stream.items() if isinstance(item, h5py.Group) and name.startswith("episode_")
    ]
    if episodes:
        return episodes
    if _dataset_names(stream):
        return [stream]
    return []


def _frame_count(group: h5py.Group) -> int:
    for attr_name in ("frames", "steps"):
        value = _int_attr(group.attrs, attr_name)
        if value is not None and value >= 0:
            return value
    for name in (
        "rgb_wrist",
        "joint_position",
        "reported_joint_position",
        "robot_state",
        "done",
        "step_index",
    ):
        item = group.get(name)
        if isinstance(item, h5py.Dataset) and item.shape:
            return int(item.shape[0])
    return 0


def _has_executed_safe_contract(root_attrs: Mapping[str, Any], episode: h5py.Group) -> bool:
    values = [
        _scalar(root_attrs.get("action_contract", "")),
        _scalar(root_attrs.get("label_contract", "")),
        _scalar(episode.attrs.get("action_contract", "")),
        _scalar(episode.attrs.get("action_target_contract", "")),
        _scalar(episode.attrs.get("action_semantics", "")),
        _scalar(episode.attrs.get("label_contract", "")),
    ]
    text = " ".join(str(value).lower() for value in values)
    return any(
        marker in text
        for marker in (
            "executed_safe",
            "executed safe",
            "executed_joint_delta",
            "queued_safe",
            "applied_command",
        )
    )


def _is_legacy_requested_not_executed(root_attrs: Mapping[str, Any], episodes: Iterable[h5py.Group]) -> bool:
    if str(_scalar(root_attrs.get("format", ""))) != "edgearm-wrist-production-multimodal-v2":
        return False
    episode_list = list(episodes)
    if not episode_list:
        return False
    # Formal M4 shards use the same historic format string, but carry the
    # explicit executed-safe action target contract on every episode.
    return any(not _has_executed_safe_contract(root_attrs, episode) for episode in episode_list)


def _artifact_role(format_name: str) -> str:
    lowered = format_name.lower()
    if "qfilter" in lowered:
        return "derived_counterfactual_qfilter"
    if "dynamic-recovery" in lowered:
        return "derived_counterfactual_recovery"
    if "dagger" in lowered or "temporal" in lowered:
        return "derived_dagger"
    return "primary_trajectory"


def _infer_sources(root_attrs: Mapping[str, Any], episodes: Iterable[h5py.Group]) -> list[str]:
    candidates: set[str] = set()
    format_name = str(_scalar(root_attrs.get("format", "unknown"))).lower()
    role = _artifact_role(format_name)
    candidates.add(role)
    for name in (
        "source_type",
        "teacher_source",
        "executed_policy",
        "live_policy",
        "selected_policy_identity",
        "teacher_identity",
    ):
        value = str(_scalar(root_attrs.get(name, ""))).strip()
        if value:
            candidates.add(value)
    for episode in episodes:
        for name in ("source_type", "teacher_source", "teacher_type", "executed_policy"):
            value = str(_scalar(episode.attrs.get(name, ""))).strip()
            if value:
                candidates.add(value)
    return sorted(candidates)


def _claim_restrictions(
    root_attrs: Mapping[str, Any], episodes: Iterable[h5py.Group], legacy_requested: bool
) -> dict[str, str]:
    restrictions: dict[str, str] = {}
    episode_list = list(episodes)
    physical_samples = _int_attr(root_attrs, "physical_samples")
    hardware = _bool_attr(root_attrs, "physical_hardware_connected")
    synthetic = _bool_attr(root_attrs, "synthetic_camera")
    physical_camera = _bool_attr(root_attrs, "physical_camera_data")
    calibrated = _bool_attr(root_attrs, "physically_calibrated")
    deployment = _bool_attr(root_attrs, "production_deployment_approved")
    episode_physical_samples = [_int_attr(episode.attrs, "physical_samples") for episode in episode_list]
    episode_hardware = [_bool_attr(episode.attrs, "physical_hardware_connected") for episode in episode_list]
    episode_synthetic = [_bool_attr(episode.attrs, "synthetic_camera") for episode in episode_list]
    episode_physical_camera = [_bool_attr(episode.attrs, "physical_camera_data") for episode in episode_list]
    episode_calibrated = [_bool_attr(episode.attrs, "physically_calibrated") for episode in episode_list]
    simulated_camera_source = any(
        "simulat" in str(_scalar(episode.attrs.get("camera_source", ""))).lower() for episode in episode_list
    )
    if physical_samples == 0 or 0 in episode_physical_samples:
        restrictions["physical_samples"] = "declared physical_samples=0"
    if synthetic is True or True in episode_synthetic or simulated_camera_source:
        restrictions["real_wrist_camera_data"] = "declared synthetic_camera=true"
    if hardware is False or False in episode_hardware:
        restrictions["physical_robot_execution"] = "declared physical_hardware_connected=false"
    if physical_camera is False or False in episode_physical_camera:
        restrictions["physical_camera_data"] = "declared physical_camera_data=false"
    if calibrated is False or False in episode_calibrated:
        restrictions["physical_calibration"] = "declared physically_calibrated=false"
    if deployment is False:
        restrictions["production_deployment"] = "declared production_deployment_approved=false"
    if legacy_requested:
        restrictions["executed_action_ground_truth"] = (
            "legacy M3 action_joint_delta/action_joint_target are requested pre-safety labels"
        )
    return restrictions


def _field_presence(
    root_attrs: Mapping[str, Any], episodes: list[h5py.Group]
) -> tuple[dict[str, int], list[dict[str, Any]]]:
    counts = {field: 0 for field in FIELD_ALIASES}
    episode_rows: list[dict[str, Any]] = []
    for episode in episodes:
        primary = _primary_group(episode)
        names = _dataset_names(primary)
        safe_contract = _has_executed_safe_contract(root_attrs, episode)
        present: set[str] = set()
        for field, aliases in FIELD_ALIASES.items():
            if any(alias in names for alias in aliases):
                present.add(field)
        if safe_contract and "action_joint_delta" in names:
            present.update(
                {"safety_projected_action", "command_sent", "executed_action", "supervision_action"}
            )
        if safe_contract and "action_joint_target" in names:
            present.add("command_sent")
        for field in present:
            counts[field] += 1
        success = _bool_attr(episode.attrs, "success")
        if success is None:
            success = _bool_attr(episode.attrs, "live_success")
        seed = _int_attr(episode.attrs, "seed")
        phase_present = any(
            name in names
            for name in (
                "effect_execution_phase_name",
                "effect_execution_phase_id",
            )
        )
        source_present = any(
            name in episode.attrs
            for name in ("source_type", "teacher_source", "teacher_type", "executed_policy")
        ) or any(
            name in root_attrs
            for name in (
                "source_type",
                "teacher_source",
                "executed_policy",
                "live_policy",
                "teacher_identity",
            )
        )
        episode_rows.append(
            {
                "group": episode.name,
                "frames": _frame_count(primary),
                "seed": seed,
                "success": success,
                "source_field_present": source_present,
                "phase_field_present": phase_present,
                "fields": sorted(present),
                "episode_attrs": sorted(str(name) for name in episode.attrs),
            }
        )
    return counts, episode_rows


def _missing_fields(
    required: Iterable[str], presence: Mapping[str, int], episode_count: int
) -> list[dict[str, Any]]:
    return [
        {
            "field": field,
            "present_episodes": int(presence.get(field, 0)),
            "required_episodes": episode_count,
            "aliases": list(FIELD_ALIASES[field]),
        }
        for field in required
        if episode_count == 0 or presence.get(field, 0) != episode_count
    ]


def _missing_episode_attrs(
    required: Iterable[str], episode_rows: list[dict[str, Any]], episodes: list[h5py.Group]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for name in required:
        count = sum(name in episode.attrs for episode in episodes)
        # Older camera calibration attrs use a JSON suffix but retain the same
        # semantic role.
        if name == "camera_calibration":
            count = sum(
                "camera_calibration" in episode.attrs or "camera_calibration_json" in episode.attrs
                for episode in episodes
            )
        if count != len(episode_rows):
            rows.append(
                {"attribute": name, "present_episodes": count, "required_episodes": len(episode_rows)}
            )
    return rows


def _missing_exact_datasets(required: Iterable[str], episodes: list[h5py.Group]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for name in required:
        count = sum(name in _dataset_names(_primary_group(episode)) for episode in episodes)
        if count != len(episodes):
            rows.append(
                {
                    "field": name,
                    "present_episodes": count,
                    "required_episodes": len(episodes),
                    "aliases": [name],
                }
            )
    return rows


def _source_schema_profile_for_audit(source_type: str) -> Any:
    """Resolve the authoritative profile lazily to avoid an import cycle."""

    from .causal_collection_v11_source_profile_v1 import (  # noqa: PLC0415
        V11_SOURCE_SCHEMA_PROFILE_V1,
    )

    if source_type == V11_SOURCE_TYPE:
        return V11_SOURCE_SCHEMA_PROFILE_V1
    if source_type == SCRATCH_SOURCE_TYPE:
        scratch_profiles = importlib.import_module(
            ".causal_collection_scratch_source_profile_v1",
            __package__,
        )
        return scratch_profiles.SCRATCH_PPO_SOURCE_SCHEMA_PROFILE_V1
    return None


def _causal_v2_semantic_attr_failures(
    root_attrs: Mapping[str, Any],
    episodes: list[h5py.Group],
) -> tuple[list[str], list[dict[str, Any]]]:
    """Reject present-but-contradictory V2 loader semantics without reading payloads."""

    expected: dict[str, Any] = {
        "task_scope": FIXED_BLOCK_PUSH_TASK_SCOPE,
        "trajectory_origin": TRAJECTORY_ORIGIN_LIVE_PRIMARY_EXECUTION,
        "recovery_injector_format": RECOVERY_INJECTOR_FORMAT,
        "recovery_injector_profile_sha256": DISABLED_RECOVERY_INJECTOR_PROFILE_SHA256,
        "recovery_event_ledger_format": RECOVERY_EVENT_LEDGER_FORMAT,
        "camera_pose_policy_format": REPORTED_WRIST_POSE_FORMAT,
        "camera_pose_policy_semantics": REPORTED_WRIST_POSE_SEMANTICS,
        "camera_pose_policy_simulator_ground_truth_used": False,
        "act_primary_label_dataset": GENERIC_ACTION_LABEL_DATASET,
        "act_action_history_dataset": ("decision_previous_reported_executed_joint_delta_normalized"),
        "act_action_history_valid_dataset": ("decision_previous_reported_executed_action_valid"),
        "act_action_history_alignment": (
            "decision_previous_reported_executed_joint_delta_* at row t equals "
            "effect_reported_executed_joint_delta_* at row t-1; row 0 is zero "
            "with valid=0; same-row effect_* is forbidden as policy input"
        ),
        "act_runtime_shield_required": True,
        "pre_guard_target_is_physics_applied_truth": False,
        "episode_integrity_gate_format": EPISODE_INTEGRITY_GATE_FORMAT,
        "action_admission_gate_format": ACTION_ADMISSION_GATE_FORMAT,
        "command_feedback_profile": "edgearm-command-feedback-v1",
        "command_feedback_alignment": "effect_t_to_decision_t_plus_1",
        "controller_preflight_profile": CAUSAL_V2_CONTROLLER_PREFLIGHT_PROFILE,
        "controller_preflight_alignment": CAUSAL_V2_CONTROLLER_PREFLIGHT_ALIGNMENT,
        "command_applied_safety_stage_map": (CAUSAL_V2_COMMAND_APPLIED_SAFETY_STAGE_MAP),
        "command_applied_post_submission_rewrite_reason_map": (CAUSAL_V2_POST_SUBMISSION_REWRITE_REASON_MAP),
        "solver_contact_stage": CAUSAL_V2_CONTACT_TELEMETRY_STAGE_MAP["solver_contact_stage"],
        "solver_contact_interval": CAUSAL_V2_CONTACT_TELEMETRY_STAGE_MAP["solver_contact_interval"],
        "effect_kinematic_stage": CAUSAL_V2_CONTACT_TELEMETRY_STAGE_MAP["effect_kinematic_stage"],
        "tool_desk_signed_distance_stage": (
            CAUSAL_V2_CONTACT_TELEMETRY_STAGE_MAP["tool_desk_signed_distance_stage"]
        ),
        "kinematic_scratch_contact_force_used": False,
        "forecast_clearance_stage": CAUSAL_V2_CONTACT_TELEMETRY_STAGE_MAP["forecast_clearance_stage"],
        "solver_contact_force_source": CAUSAL_V2_CONTACT_TELEMETRY_STAGE_MAP["solver_contact_force_source"],
        "contact_telemetry_stage_map": CAUSAL_V2_CONTACT_TELEMETRY_STAGE_MAP_JSON,
        "execution_geometry_format": EXECUTION_GEOMETRY_FORMAT,
        "execution_geometry_semantics": CAUSAL_V2_EXECUTION_GEOMETRY_SEMANTICS,
        "execution_geometry_height_reference_formula": (CAUSAL_V2_EXECUTION_GEOMETRY_HEIGHT_FORMULA),
        "recovery_evidence_format": RECOVERY_EVIDENCE_FORMAT,
        "recovery_evidence_semantics": CAUSAL_V2_RECOVERY_EVIDENCE_SEMANTICS,
        "recovery_intervention_type_id_map": (CAUSAL_V2_RECOVERY_INTERVENTION_TYPE_ID_MAP),
        "recovery_origin_intervention_code_id_map": (CAUSAL_V2_RECOVERY_ORIGIN_INTERVENTION_CODE_ID_MAP),
        "terminal_failure_codes": CAUSAL_V2_TERMINAL_FAILURE_CODE_TAXONOMY,
        "formal_recovery_intervention_injection_enabled": False,
    }
    root_only_expected: dict[str, Any] = {
        "episode_integrity_gate_required": True,
        "action_admission_gate_required": True,
        "action_admission_source_registry": action_source_registry_json(),
    }

    def matches(value: Any, expected_value: Any) -> bool:
        value = _scalar(value)
        if isinstance(expected_value, bool):
            if isinstance(value, str):
                normalized = value.strip().lower()
                return normalized in ({"true", "1"} if expected_value else {"false", "0"})
            return bool(value) is expected_value
        return str(value) == str(expected_value)

    root_failures = [
        f"semantic:{name}"
        for name, expected_value in expected.items()
        if name in root_attrs and not matches(root_attrs[name], expected_value)
    ]
    root_failures.extend(
        f"semantic:{name}"
        for name, expected_value in root_only_expected.items()
        if name in root_attrs and not matches(root_attrs[name], expected_value)
    )
    root_source = str(_scalar(root_attrs.get("source_type", "")))
    if root_source not in SUPPORTED_SOURCE_TYPES:
        root_failures.append("semantic:source_type")
    source_profile = _source_schema_profile_for_audit(root_source)
    if source_profile is not None:
        if str(_scalar(root_attrs.get("source_schema_profile_format", ""))) != (SOURCE_SCHEMA_PROFILE_FORMAT):
            root_failures.append("semantic:source_schema_profile_format")
        if str(_scalar(root_attrs.get("source_schema_profile_sha256", ""))) != (source_profile.sha256_v1()):
            root_failures.append("semantic:source_schema_profile_sha256")
        if not is_sha256(str(_scalar(root_attrs.get("source_preparation_binding_sha256", "")))):
            root_failures.append("semantic:source_preparation_binding_sha256")
    if root_source == SCRATCH_SOURCE_TYPE:
        forbidden = (set(CAUSAL_V2_V11_ROOT_ATTRS) - {"policy_artifact_sha256"}) & set(root_attrs)
        root_failures.extend(f"semantic:scratch_forbids:{name}" for name in forbidden)
    elif root_source == V11_SOURCE_TYPE:
        root_failures.extend(
            f"semantic:v11:{name}" for name in CAUSAL_V2_V11_ROOT_ATTRS if name not in root_attrs
        )
        if "policy_artifact_sha256" not in root_attrs:
            root_failures.append("semantic:v11:policy_artifact_sha256")
        v11_source_binding_fields = (
            *CAUSAL_V2_V11_PRODUCER_BINDING_ATTRS,
            *CAUSAL_V2_V11_EXECUTION_EVIDENCE_ATTRS,
        )
        if all(name in root_attrs for name in v11_source_binding_fields):
            binding = {name: str(_scalar(root_attrs[name])) for name in v11_source_binding_fields}
            if binding["producer_artifact_kind"] != "v11_controller_bundle_manifest":
                root_failures.append("semantic:producer_artifact_kind")
            for name in (
                "producer_artifact_manifest_sha256",
                "runtime_asset_sha256",
                "producer_runtime_closure_sha256",
                "policy_artifact_sha256",
            ):
                if not is_sha256(binding[name]):
                    root_failures.append(f"semantic:{name}")
            expected_manifest_path = (
                "producer_artifacts/v11_controller_"
                f"{binding['producer_runtime_closure_sha256']}/manifest.json"
            )
            declared_path = PurePosixPath(binding["producer_artifact_manifest_path"])
            if (
                declared_path.is_absolute()
                or declared_path.as_posix() != binding["producer_artifact_manifest_path"]
                or any(part in {"", ".", ".."} for part in declared_path.parts)
                or binding["producer_artifact_manifest_path"] != expected_manifest_path
            ):
                root_failures.append("semantic:producer_artifact_manifest_path")
            evidence_level = binding["collection_execution_evidence_level"]
            if evidence_level not in {
                "frozen_worker_synthetic",
                "in_process_smoke_synthetic",
            }:
                root_failures.append("semantic:collection_execution_evidence_level")
            try:
                from .causal_collection_v11_source_profile_v1 import (  # noqa: PLC0415
                    build_v11_source_collection_preparation_v1,
                )

                preparation = build_v11_source_collection_preparation_v1(binding)
                expected_preparation_sha256 = preparation.binding_sha256_v1(source_profile)
            except (TypeError, ValueError):
                root_failures.append("semantic:v11_producer_binding")
            else:
                if (
                    str(_scalar(root_attrs.get("source_preparation_binding_sha256", "")))
                    != expected_preparation_sha256
                ):
                    root_failures.append("semantic:source_preparation_binding_sha256")
    if "max_joint_delta_rad" in root_attrs:
        try:
            scale = float(_scalar(root_attrs["max_joint_delta_rad"]))
        except (TypeError, ValueError):
            scale = float("nan")
        if not np.isfinite(scale) or scale <= 0.0:
            root_failures.append("semantic:max_joint_delta_rad")
    if "execution_geometry_declared_runtime_clearance_m" in root_attrs:
        try:
            clearance = float(_scalar(root_attrs["execution_geometry_declared_runtime_clearance_m"]))
        except (TypeError, ValueError):
            clearance = float("nan")
        if not np.isfinite(clearance) or clearance < 0.0:
            root_failures.append("semantic:execution_geometry_declared_runtime_clearance_m")
    if not is_sha256(str(_scalar(root_attrs.get("recovery_injector_source_sha256", "")))):
        root_failures.append("semantic:recovery_injector_source_sha256")

    episode_failures: list[dict[str, Any]] = []
    for name, expected_value in expected.items():
        invalid = sum(
            name in episode.attrs
            and (
                not matches(episode.attrs[name], expected_value)
                or (name in root_attrs and not matches(episode.attrs[name], root_attrs[name]))
            )
            for episode in episodes
        )
        if invalid:
            episode_failures.append(
                {
                    "attribute": f"semantic:{name}",
                    "invalid_episodes": int(invalid),
                    "required_episodes": len(episodes),
                }
            )

    for episode in episodes:
        source = str(_scalar(episode.attrs.get("source_type", root_source)))
        if source != root_source or source not in SUPPORTED_SOURCE_TYPES:
            episode_failures.append(
                {
                    "attribute": "semantic:source_type",
                    "invalid_episodes": 1,
                    "required_episodes": len(episodes),
                }
            )
        if source == SCRATCH_SOURCE_TYPE:
            for name in CAUSAL_V2_V11_EPISODE_ATTRS:
                if name != "policy_artifact_sha256" and name in episode.attrs:
                    episode_failures.append(
                        {
                            "attribute": f"semantic:scratch_forbids:{name}",
                            "invalid_episodes": 1,
                            "required_episodes": len(episodes),
                        }
                    )
        elif source == V11_SOURCE_TYPE:
            for name in CAUSAL_V2_V11_EPISODE_ATTRS:
                if name not in episode.attrs:
                    episode_failures.append(
                        {
                            "attribute": f"semantic:v11:{name}",
                            "invalid_episodes": 1,
                            "required_episodes": len(episodes),
                        }
                    )
            request_sha256 = str(_scalar(episode.attrs.get("frozen_worker_request_sha256", "")))
            evidence_level = str(
                _scalar(
                    episode.attrs.get(
                        "collection_execution_evidence_level",
                        root_attrs.get("collection_execution_evidence_level", ""),
                    )
                )
            )
            if evidence_level == "frozen_worker_synthetic":
                if not is_sha256(request_sha256):
                    episode_failures.append(
                        {
                            "attribute": "semantic:frozen_worker_request_sha256",
                            "invalid_episodes": 1,
                            "required_episodes": len(episodes),
                        }
                    )
            elif evidence_level == "in_process_smoke_synthetic":
                if request_sha256 != "":
                    episode_failures.append(
                        {
                            "attribute": "semantic:frozen_worker_request_sha256",
                            "invalid_episodes": 1,
                            "required_episodes": len(episodes),
                        }
                    )
        binding_fields = SOURCE_SCHEMA_PROVENANCE_FIELDS
        if source == V11_SOURCE_TYPE:
            binding_fields = (
                *binding_fields,
                *CAUSAL_V2_V11_PRODUCER_BINDING_ATTRS,
                *CAUSAL_V2_V11_EXECUTION_EVIDENCE_ATTRS,
            )
        for name in binding_fields:
            if (
                name in root_attrs
                and name in episode.attrs
                and str(_scalar(episode.attrs[name])) != str(_scalar(root_attrs[name]))
            ):
                episode_failures.append(
                    {
                        "attribute": f"semantic:{name}",
                        "invalid_episodes": 1,
                        "required_episodes": len(episodes),
                    }
                )
        calibration_value = _scalar(episode.attrs.get("camera_calibration", ""))
        try:
            calibration = (
                calibration_value
                if isinstance(calibration_value, Mapping)
                else json.loads(str(calibration_value))
            )
        except (json.JSONDecodeError, TypeError):
            calibration = None
        policy_pose = (
            calibration.get("wrist", {}).get("policy_pose_estimator")
            if isinstance(calibration, Mapping) and isinstance(calibration.get("wrist"), Mapping)
            else None
        )
        if not valid_reported_wrist_pose_profile(policy_pose):
            episode_failures.append(
                {
                    "attribute": "semantic:camera_calibration.policy_pose_estimator",
                    "invalid_episodes": 1,
                    "required_episodes": len(episodes),
                }
            )
        depth_contract = (
            calibration.get("wrist", {}).get("metric_depth_contract")
            if isinstance(calibration, Mapping) and isinstance(calibration.get("wrist"), Mapping)
            else None
        )
        if not valid_metric_depth_contract(depth_contract):
            episode_failures.append(
                {
                    "attribute": "semantic:camera_calibration.metric_depth_contract",
                    "invalid_episodes": 1,
                    "required_episodes": len(episodes),
                }
            )
        if not is_sha256(str(_scalar(episode.attrs.get("recovery_injector_source_sha256", "")))):
            episode_failures.append(
                {
                    "attribute": "semantic:recovery_injector_source_sha256",
                    "invalid_episodes": 1,
                    "required_episodes": len(episodes),
                }
            )
        row_failures = _causal_v2_recovery_injection_row_failures(episode)
        episode_failures.extend(
            {
                "attribute": f"semantic:{failure}",
                "invalid_episodes": 1,
                "required_episodes": len(episodes),
            }
            for failure in row_failures
        )
    if "max_joint_delta_rad" in root_attrs:
        invalid_scale = sum(
            "max_joint_delta_rad" in episode.attrs
            and not matches(
                episode.attrs["max_joint_delta_rad"],
                root_attrs["max_joint_delta_rad"],
            )
            for episode in episodes
        )
        if invalid_scale:
            episode_failures.append(
                {
                    "attribute": "semantic:max_joint_delta_rad",
                    "invalid_episodes": int(invalid_scale),
                    "required_episodes": len(episodes),
                }
            )
    if "execution_geometry_declared_runtime_clearance_m" in root_attrs:
        invalid_clearance = sum(
            "execution_geometry_declared_runtime_clearance_m" in episode.attrs
            and not matches(
                episode.attrs["execution_geometry_declared_runtime_clearance_m"],
                root_attrs["execution_geometry_declared_runtime_clearance_m"],
            )
            for episode in episodes
        )
        if invalid_clearance:
            episode_failures.append(
                {
                    "attribute": ("semantic:execution_geometry_declared_runtime_clearance_m"),
                    "invalid_episodes": int(invalid_clearance),
                    "required_episodes": len(episodes),
                }
            )
    return root_failures, episode_failures


def _causal_v2_recovery_injection_row_failures(episode: h5py.Group) -> list[str]:
    """Read only tiny recovery lineage arrays and fail closed on forged claims."""

    failures: list[str] = []
    frames = _int_attr(episode.attrs, "frames")
    if frames is None:
        dataset = episode.get("effect_recovery_injection_event_index")
        frames = int(dataset.shape[0]) if isinstance(dataset, h5py.Dataset) and dataset.ndim == 1 else 0
    fields = {
        "effect_recovery_injection_event_index": -1,
        "effect_recovery_injection_requested": 0,
        "effect_recovery_injection_hold_substeps_mask": 0,
    }
    arrays: dict[str, np.ndarray] = {}
    for name, disabled_value in fields.items():
        dataset = episode.get(name)
        if not isinstance(dataset, h5py.Dataset) or dataset.shape != (frames,):
            failures.append(f"{name}_shape")
            continue
        if dataset.dtype.kind not in {"b", "i", "u"}:
            failures.append(f"{name}_dtype")
            continue
        array = np.asarray(dataset[:])
        arrays[name] = array
        if np.any(array != disabled_value):
            failures.append(f"disabled_injector_nonzero:{name}")

    formal = (
        _bool_attr(episode.attrs, "formal_recovery_evidence_available") is True
        or _bool_attr(episode.attrs, "verified_recovery") is True
    )
    recovery_names = (
        "effect_execution_recovery_epoch_id",
        "effect_recovery_intervention_type_id",
        "effect_execution_recovery_origin_intervention_code_id",
    )
    nonzero_recovery = False
    for name in recovery_names:
        dataset = episode.get(name)
        if isinstance(dataset, h5py.Dataset) and dataset.ndim == 1:
            nonzero_recovery = nonzero_recovery or bool(np.any(np.asarray(dataset[:]) != 0))
    claimed = formal or nonzero_recovery
    if not claimed:
        if "recovery_event_ledger_json" in episode.attrs:
            try:
                ledger = validate_event_ledger(
                    str(_scalar(episode.attrs["recovery_event_ledger_json"])),
                    frame_count=max(frames, 1),
                    require_replay_verified=True,
                )
            except ValueError:
                failures.append("recovery_event_ledger_invalid")
            else:
                if ledger:
                    failures.append("unclaimed_recovery_event_ledger_not_empty")
        return failures

    if "recovery_event_ledger_json" not in episode.attrs:
        failures.append("formal_recovery_missing_event_ledger")
        return failures
    try:
        ledger = validate_event_ledger(
            str(_scalar(episode.attrs["recovery_event_ledger_json"])),
            frame_count=max(frames, 1),
            require_replay_verified=True,
        )
    except ValueError:
        failures.append("formal_recovery_event_ledger_invalid_or_unverified")
        return failures
    if not ledger or any(record["outcome"] != "injected_effect_observed" for record in ledger):
        failures.append("formal_recovery_event_ledger_incomplete")
    return failures


def _physical_evidence_ok(root_attrs: Mapping[str, Any]) -> tuple[bool, list[str]]:
    failures: list[str] = []
    samples = _int_attr(root_attrs, "physical_samples")
    if samples is None or samples <= 0:
        failures.append("physical_samples must be > 0")
    if _bool_attr(root_attrs, "physical_hardware_connected") is not True:
        failures.append("physical_hardware_connected must be true")
    if _bool_attr(root_attrs, "synthetic_camera") is not False:
        failures.append("synthetic_camera must be explicitly false")
    if _bool_attr(root_attrs, "physical_camera_data") is not True:
        failures.append("physical_camera_data must be true")
    if _bool_attr(root_attrs, "physically_calibrated") is not True:
        failures.append("physically_calibrated must be true")
    return not failures, failures


def _grade(
    purpose: Purpose,
    *,
    hard_missing: list[dict[str, Any]],
    complete_missing: list[dict[str, Any]],
    missing_root_attrs: list[str],
    missing_episode_attrs: list[dict[str, Any]],
    legacy_requested: bool,
    physical_evidence_ok: bool,
    has_rgb: bool,
    has_any_action: bool,
) -> Grade:
    if legacy_requested:
        # The old M3 images and state streams remain useful as auxiliary
        # world-model evidence, but their pre-safety action fields break causal
        # transition semantics.  Therefore world-model use is diagnostic C,
        # while all action/control uses are ineligible D.
        return Grade.C if purpose is Purpose.WORLD_MODEL else Grade.D
    if purpose is Purpose.PHYSICAL_DEPLOYMENT and not physical_evidence_ok:
        return Grade.D
    if hard_missing:
        return Grade.C if has_rgb and has_any_action else Grade.D
    if complete_missing or missing_root_attrs or missing_episode_attrs:
        return Grade.B
    return Grade.A


def audit_h5(path: Path | str, *, include_episode_index: bool = False) -> dict[str, Any]:
    """Audit one HDF5 file without reading any dataset payload."""

    resolved = Path(path).expanduser().resolve()
    with h5py.File(resolved, "r") as stream:
        root_attrs = {str(name): _scalar(value) for name, value in stream.attrs.items()}
        episodes = _episode_groups(stream)
        presence, episode_rows = _field_presence(root_attrs, episodes)
        format_name = str(root_attrs.get("format", "unknown"))
        role = _artifact_role(format_name)
        legacy_requested = _is_legacy_requested_not_executed(root_attrs, episodes)
        restrictions = _claim_restrictions(root_attrs, episodes, legacy_requested)
        physical_ok, physical_failures = _physical_evidence_ok(root_attrs)
        episode_count = len(episodes)
        successes = sum(row["success"] is True for row in episode_rows)
        failures = sum(row["success"] is False for row in episode_rows)
        unknown_outcomes = episode_count - successes - failures
        frames = sum(int(row["frames"]) for row in episode_rows)
        source_field_count = sum(bool(row["source_field_present"]) for row in episode_rows)
        phase_field_count = sum(bool(row["phase_field_present"]) for row in episode_rows)

        purpose_results: dict[str, Any] = {}
        for purpose, spec in CONTRACTS.items():
            hard_missing = _missing_fields(spec.hard_fields, presence, episode_count)
            complete_missing = _missing_fields(spec.complete_fields, presence, episode_count)
            missing_root = [name for name in spec.complete_root_attrs if name not in root_attrs]
            missing_episode = _missing_episode_attrs(spec.complete_episode_attrs, episode_rows, episodes)
            if format_name == CAUSAL_V2_FORMAT and purpose is not Purpose.PHYSICAL_DEPLOYMENT:
                complete_missing.extend(_missing_exact_datasets(CAUSAL_V2_EXECUTION_DATASETS, episodes))
                root_source = str(root_attrs.get("source_type", ""))
                if root_source == V11_SOURCE_TYPE:
                    complete_missing.extend(
                        _missing_exact_datasets(
                            CAUSAL_V2_V11_DIAGNOSTIC_DATASETS,
                            episodes,
                        )
                    )
                missing_root.extend(name for name in CAUSAL_V2_ROOT_ATTRS if name not in root_attrs)
                missing_episode.extend(
                    _missing_episode_attrs(CAUSAL_V2_EPISODE_ATTRS, episode_rows, episodes)
                )
                semantic_root, semantic_episode = _causal_v2_semantic_attr_failures(root_attrs, episodes)
                missing_root.extend(semantic_root)
                missing_episode.extend(semantic_episode)
            grade = _grade(
                purpose,
                hard_missing=hard_missing,
                complete_missing=complete_missing,
                missing_root_attrs=missing_root,
                missing_episode_attrs=missing_episode,
                legacy_requested=legacy_requested,
                physical_evidence_ok=physical_ok,
                has_rgb=presence.get("rgb_wrist", 0) == episode_count and episode_count > 0,
                has_any_action=any(
                    presence.get(name, 0) > 0
                    for name in (
                        "requested_action",
                        "safety_projected_action",
                        "executed_action",
                        "supervision_action",
                    )
                ),
            )
            # This auditor deliberately never indexes trajectory payloads.  A
            # therefore means schema-presence A, not formal training admission.
            # Payload integrity, episode semantics, and the explicit coverage
            # plan are evaluated by audit_trajectory_coverage_v1.
            schema_presence_eligible = grade is Grade.A
            restricted_schema_candidate = grade in {Grade.A, Grade.B}
            purpose_results[purpose.value] = {
                "grade": grade.value,
                "schema_presence_eligible": schema_presence_eligible,
                "restricted_schema_candidate": restricted_schema_candidate,
                "training_eligible": False,
                "restricted_training_eligible": False,
                "training_eligibility_evaluated": False,
                "requires_payload_integrity_gate": True,
                "requires_episode_semantics_gate": True,
                "requires_coverage_plan_gate": True,
                "eligibility_scope": (
                    "schema_complete_payload_unverified"
                    if grade is Grade.A
                    else "schema_restricted_payload_unverified"
                    if grade is Grade.B
                    else "diagnostic_only"
                    if grade is Grade.C
                    else "ineligible"
                ),
                "missing_hard_fields": hard_missing,
                "missing_complete_fields": complete_missing,
                "missing_root_attributes": missing_root,
                "missing_episode_attributes": missing_episode,
                "physical_evidence_failures": (
                    physical_failures if purpose is Purpose.PHYSICAL_DEPLOYMENT else []
                ),
            }

        report = {
            "contract_version": CONTRACT_VERSION,
            "path": str(resolved),
            "format": format_name,
            "artifact_role": role,
            "source_types": _infer_sources(root_attrs, episodes),
            "legacy_m3_requested_not_executed": legacy_requested,
            "claim_restrictions": restrictions,
            "root_attributes_present": sorted(root_attrs),
            "field_episode_counts": presence,
            "coverage": {
                "episode_records": episode_count,
                "frames": frames,
                "successes": successes,
                "failures": failures,
                "unknown_outcomes": unknown_outcomes,
                "source_field_episodes": source_field_count,
                "phase_field_episodes": phase_field_count,
            },
            "purposes": purpose_results,
            "training_eligible": {
                purpose: result["training_eligible"] for purpose, result in purpose_results.items()
            },
            "restricted_training_eligible": {
                purpose: result["restricted_training_eligible"] for purpose, result in purpose_results.items()
            },
            "schema_presence_eligible": {
                purpose: result["schema_presence_eligible"] for purpose, result in purpose_results.items()
            },
            "restricted_schema_candidate": {
                purpose: result["restricted_schema_candidate"] for purpose, result in purpose_results.items()
            },
            "formal_admission_decision": "not_evaluated_metadata_only",
            "training_eligibility_evaluated": False,
            "metadata_only": True,
        }
        if role in {"derived_counterfactual_qfilter", "derived_counterfactual_recovery"}:
            recognized = []
            for field in (
                "tool_pose",
                "next_tool_pose",
                "camera_device_timestamp",
                "camera_host_timestamp",
            ):
                if presence.get(field, 0) != episode_count:
                    recognized.append(field)
            report["recognized_schema_gaps"] = recognized
        elif legacy_requested:
            report["recognized_schema_gaps"] = ["legacy_m3_requested_not_executed"]
        else:
            report["recognized_schema_gaps"] = []
        if include_episode_index:
            report["episode_index"] = episode_rows
        return report


def discover_h5(path: Path | str) -> list[Path]:
    resolved = Path(path).expanduser().resolve()
    if resolved.is_file():
        if resolved.suffix.lower() not in {".h5", ".hdf5"}:
            raise ValueError(f"not an HDF5 file: {resolved}")
        return [resolved]
    if not resolved.is_dir():
        raise FileNotFoundError(resolved)
    return sorted(
        candidate
        for candidate in resolved.rglob("*")
        if candidate.is_file() and candidate.suffix.lower() in {".h5", ".hdf5"}
    )


def _episode_key(file_report: Mapping[str, Any], episode: Mapping[str, Any]) -> tuple[Any, ...]:
    seed = episode.get("seed")
    if seed is None:
        return (file_report["path"], episode["group"])
    # Format/split distinguish independent seed namespaces while still
    # deduplicating copied shards and repeated derived views of one trajectory.
    path = Path(str(file_report["path"]))
    split = next(
        (part for part in ("train", "val", "validation", "stress", "smoke") if part in path.parts),
        "",
    )
    return (file_report["format"], split, int(seed))


def audit_path(path: Path | str) -> dict[str, Any]:
    """Audit one file or a directory tree and return a JSON-compatible report.

    Primary trajectories and derived/counterfactual artifacts are summarized in
    separate buckets.  Derived files are deliberately *not* added to the unique
    primary episode/frame totals.
    """

    files = discover_h5(path)
    reports: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    for file_path in files:
        try:
            reports.append(audit_h5(file_path, include_episode_index=True))
        except (OSError, ValueError, KeyError) as exc:
            errors.append({"path": str(file_path), "error": f"{type(exc).__name__}: {exc}"})

    primary_seen: dict[tuple[Any, ...], int] = {}
    primary_success: dict[tuple[Any, ...], bool | None] = {}
    derived_records = 0
    derived_frames = 0
    for report in reports:
        if report["artifact_role"] != "primary_trajectory":
            derived_records += int(report["coverage"]["episode_records"])
            derived_frames += int(report["coverage"]["frames"])
            continue
        for episode in report["episode_index"]:
            key = _episode_key(report, episode)
            primary_seen.setdefault(key, int(episode["frames"]))
            primary_success.setdefault(key, episode["success"])

    purpose_summary: dict[str, Any] = {}
    rank = {"A": 0, "B": 1, "C": 2, "D": 3}
    for purpose in Purpose:
        counts = {grade.value: 0 for grade in Grade}
        for report in reports:
            counts[report["purposes"][purpose.value]["grade"]] += 1
        worst = max(
            (report["purposes"][purpose.value]["grade"] for report in reports),
            key=lambda value: rank[value],
            default="D",
        )
        purpose_summary[purpose.value] = {
            "file_grades": counts,
            "worst_grade": worst,
            "all_files_schema_presence_eligible": bool(reports)
            and all(report["purposes"][purpose.value]["schema_presence_eligible"] for report in reports),
            "all_files_restricted_schema_candidate": bool(reports)
            and all(report["purposes"][purpose.value]["restricted_schema_candidate"] for report in reports),
            "all_files_training_eligible": bool(reports)
            and all(report["purposes"][purpose.value]["training_eligible"] for report in reports),
            "all_files_restricted_training_eligible": bool(reports)
            and all(report["purposes"][purpose.value]["restricted_training_eligible"] for report in reports),
        }

    unique_successes = sum(value is True for value in primary_success.values())
    unique_failures = sum(value is False for value in primary_success.values())
    public_reports: list[dict[str, Any]] = []
    for report in reports:
        public_report = dict(report)
        public_report.pop("episode_index", None)
        public_reports.append(public_report)

    return {
        "contract_version": CONTRACT_VERSION,
        "path": str(Path(path).expanduser().resolve()),
        "files_discovered": len(files),
        "files_audited": len(reports),
        "file_errors": errors,
        "coverage": {
            "unique_primary_episodes": len(primary_seen),
            "unique_primary_frames": sum(primary_seen.values()),
            "unique_primary_successes": unique_successes,
            "unique_primary_failures": unique_failures,
            "unique_primary_unknown_outcomes": len(primary_seen) - unique_successes - unique_failures,
            "derived_episode_records_not_added": derived_records,
            "derived_frame_records_not_added": derived_frames,
            "deduplication_key": "format + split + seed; fallback path + group",
        },
        "purpose_summary": purpose_summary,
        "training_eligible": {
            purpose: summary["all_files_training_eligible"] for purpose, summary in purpose_summary.items()
        },
        "restricted_training_eligible": {
            purpose: summary["all_files_restricted_training_eligible"]
            for purpose, summary in purpose_summary.items()
        },
        "schema_presence_eligible": {
            purpose: summary["all_files_schema_presence_eligible"]
            for purpose, summary in purpose_summary.items()
        },
        "restricted_schema_candidate": {
            purpose: summary["all_files_restricted_schema_candidate"]
            for purpose, summary in purpose_summary.items()
        },
        "formal_admission_decision": "not_evaluated_metadata_only",
        "training_eligibility_evaluated": False,
        "files": public_reports,
        "metadata_only": True,
    }


def report_json(report: Mapping[str, Any], *, pretty: bool = True) -> str:
    return json.dumps(report, ensure_ascii=False, indent=2 if pretty else None, sort_keys=True)


__all__ = [
    "CONTRACT_VERSION",
    "CONTRACTS",
    "FIELD_ALIASES",
    "FIXED_BLOCK_PUSH_TASK_SCOPE",
    "METRIC_DEPTH_CONTRACT_FORMAT",
    "ContractSpec",
    "Grade",
    "Purpose",
    "audit_h5",
    "audit_path",
    "discover_h5",
    "report_json",
    "valid_metric_depth_contract",
    "valid_reported_wrist_pose_profile",
]
