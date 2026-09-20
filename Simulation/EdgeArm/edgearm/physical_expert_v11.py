"""Joint-space pre-contact successor to the synthetic privileged V9 expert.

V9 uses a local operational-space update immediately after reset.  Around a
folded SO-101 configuration that update can saturate while the tilted pusher
contacts the top of the block, crosses its centerline, and then keeps moving in
the wrong direction.  V11 first plans one collision-audited, task-aligned joint
staged joint-space route behind the block.  The simulated force-limited arm
must execute every waypoint; joint state is never written by the expert.  V9
terminal control begins only after measured joint and Cartesian errors verify
arrival.  Runtime collision validity is measured from V7 physics-substep
telemetry rather than inferred from the planned waypoints.

The planner consumes exact simulator block and target state.  It is therefore
a privileged synthetic demonstration source, not a physical controller or a
claim of real-camera perception.  Physical samples and trials remain zero.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping

import mujoco
import numpy as np

from .contact_telemetry_v1 import PushSideContactThresholds
from .joint_path_planner_v1 import JointPathPlannerV1, _CollisionChecker
from .physical_expert_v7 import EXPERT_CLAIM_LEVEL
from .physical_expert_v9 import (
    PHYSICAL_EXPERT_V9_VERSION,
    PhysicalClosedLoopExpertV9,
    PhysicalExpertV9Config,
)
from .sim2real_env_v6 import RealisticEdgeArmEnvV6
from .sim2real_env_v7 import RealisticEdgeArmEnvV7
from .side_contact_ik_v1 import SideContactIKConfig, SideContactIKPlannerV1


PHYSICAL_EXPERT_V11_VERSION = "edgearm-privileged-physical-expert-v11"
PHYSICAL_EXPERT_V11_PARAMETER_SOURCE = "synthetic_joint_space_precontact_planner"
PHYSICAL_EXPERT_V11_CONFIG_FORMAT = "edgearm-deterministic-physical-expert-v11-config"
PHYSICAL_EXPERT_V11_CONFIG_SCHEMA_VERSION = 1
_JOINTS = 6
_CENTRAL_FULL_PUSH_NORMAL_ALIGNMENT_SAFETY_FLOOR = 0.80
_CENTRAL_FULL_PUSH_REAR_SUPPORT_SAFETY_FLOOR = 0.50
_EDGE_FULL_PUSH_NORMAL_ALIGNMENT_SAFETY_FLOOR = 0.94
_EDGE_FULL_PUSH_REAR_SUPPORT_SAFETY_FLOOR = 0.735
_VERTICAL_RECENTER_REALIZED_PROGRESS_TOLERANCE_M = 1.0e-6
_CONTACT_CLEARANCE_BRIDGE_TARGET_TOLERANCE_RAD = 1.0e-8


@dataclass(frozen=True)
class PhysicalExpertV11Config:
    """Synthetic pre-contact and frozen V9 geometry parameters."""

    precontact_standoff_m: float = 0.095
    precontact_joint_error_tolerance_rad: float = 0.040
    precontact_vertical_error_tolerance_m: float = 0.050
    precontact_lateral_error_tolerance_m: float = 0.025
    precontact_minimum_behind_m: float = 0.050
    precontact_replan_block_motion_m: float = 0.002
    precontact_high_standoff_m: float = 0.105
    precontact_high_clearance_height_m: float = 0.125
    precontact_entry_height_m: float = 0.095
    contact_standoff_m: float = 0.040
    tool_height_m: float = 0.082
    approach_step_limit_m: float = 0.019
    push_step_m: float = 0.021
    push_gate_along_m: float = 0.063
    push_gate_lateral_m: float = 0.045
    push_gate_minimum_behind_m: float = 0.005
    push_gate_vertical_error_m: float = 0.025
    minimum_pusher_desk_clearance_m: float = 0.002
    max_global_reacquisitions: int = 8
    precontact_cartesian_error_tolerance_m: float = 0.008
    precontact_face_horizontal_minimum: float = 0.94
    precontact_face_alignment_minimum: float = 0.94
    precontact_goal_gap_minimum_m: float = 0.0001
    precontact_goal_gap_maximum_m: float = 0.012
    side_contact_acquisition_step_m: float = 0.0025
    side_push_step_m: float = 0.004
    precision_push_coverage_start: float = 0.70
    fine_push_coverage_start: float = 0.88
    precision_side_push_step_m: float = 0.002
    fine_side_push_step_m: float = 0.0012
    central_full_push_minimum_normal_alignment: float = _CENTRAL_FULL_PUSH_NORMAL_ALIGNMENT_SAFETY_FLOOR
    central_full_push_minimum_rear_support_ratio: float = _CENTRAL_FULL_PUSH_REAR_SUPPORT_SAFETY_FLOOR
    edge_full_push_minimum_normal_alignment: float = _EDGE_FULL_PUSH_NORMAL_ALIGNMENT_SAFETY_FLOOR
    edge_full_push_minimum_rear_support_ratio: float = _EDGE_FULL_PUSH_REAR_SUPPORT_SAFETY_FLOOR
    invalid_contact_escape_xy_m: float = 0.004
    invalid_contact_escape_z_m: float = 0.0015
    invalid_contact_escape_steps: int = 4
    precontact_joint_command_step_rad: float = 0.055
    terminal_unload_step_m: float = 0.004
    terminal_lift_step_m: float = 0.0015
    operational_tracking_failure_replan_score: int = 3
    joint_limit_recovery_trigger_tolerance_rad: float = 0.001
    joint_limit_recovery_inset_rad: float = 0.003
    joint_limit_recovery_step_rad: float = 0.010
    runtime_clearance_recovery_joint_step_rad: float = 0.020
    runtime_clearance_recovery_stable_steps: int = 3

    def __post_init__(self) -> None:
        positive = (
            "precontact_standoff_m",
            "precontact_joint_error_tolerance_rad",
            "precontact_vertical_error_tolerance_m",
            "precontact_lateral_error_tolerance_m",
            "precontact_minimum_behind_m",
            "precontact_replan_block_motion_m",
            "precontact_high_standoff_m",
            "precontact_high_clearance_height_m",
            "precontact_entry_height_m",
            "push_gate_minimum_behind_m",
            "push_gate_vertical_error_m",
            "minimum_pusher_desk_clearance_m",
            "precontact_cartesian_error_tolerance_m",
            "precontact_goal_gap_minimum_m",
            "precontact_goal_gap_maximum_m",
            "side_contact_acquisition_step_m",
            "side_push_step_m",
            "precision_side_push_step_m",
            "fine_side_push_step_m",
            "invalid_contact_escape_xy_m",
            "invalid_contact_escape_z_m",
            "precontact_joint_command_step_rad",
            "terminal_unload_step_m",
            "terminal_lift_step_m",
            "joint_limit_recovery_inset_rad",
            "joint_limit_recovery_step_rad",
            "joint_limit_recovery_trigger_tolerance_rad",
            "runtime_clearance_recovery_joint_step_rad",
        )
        for name in positive:
            value = float(getattr(self, name))
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if not 0.085 <= self.precontact_standoff_m <= 0.105:
            raise ValueError("precontact_standoff_m must be in [0.085, 0.105]")
        if not 0.010 <= self.precontact_joint_error_tolerance_rad <= 0.050:
            raise ValueError("precontact_joint_error_tolerance_rad must be in [0.010, 0.050]")
        if self.precontact_minimum_behind_m >= self.precontact_standoff_m:
            raise ValueError("precontact_minimum_behind_m must be below the waypoint standoff")
        if not self.precontact_standoff_m < self.precontact_high_standoff_m <= 0.120:
            raise ValueError("precontact_high_standoff_m must be above final standoff and at most 0.12")
        if not 0.115 <= self.precontact_high_clearance_height_m <= 0.160:
            raise ValueError("precontact_high_clearance_height_m must be in [0.115, 0.160]")
        if not 0.090 <= self.precontact_entry_height_m <= 0.120:
            raise ValueError("precontact_entry_height_m must be in [0.090, 0.120]")
        if self.push_gate_minimum_behind_m >= self.push_gate_along_m:
            raise ValueError("push_gate_minimum_behind_m must be below push_gate_along_m")
        if not 0.010 <= self.push_gate_vertical_error_m <= 0.040:
            raise ValueError("push_gate_vertical_error_m must be in [0.010, 0.040]")
        if not 0.001 <= self.minimum_pusher_desk_clearance_m <= 0.010:
            raise ValueError("minimum_pusher_desk_clearance_m must be in [0.001, 0.010]")
        if not isinstance(self.max_global_reacquisitions, int) or not (
            1 <= self.max_global_reacquisitions <= 8
        ):
            raise ValueError("max_global_reacquisitions must be an integer in [1, 8]")
        for name in (
            "precontact_face_horizontal_minimum",
            "precontact_face_alignment_minimum",
            "central_full_push_minimum_normal_alignment",
            "central_full_push_minimum_rear_support_ratio",
            "edge_full_push_minimum_normal_alignment",
            "edge_full_push_minimum_rear_support_ratio",
        ):
            value = float(getattr(self, name))
            if not 0.5 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0.5, 1.0]")
        if self.precontact_goal_gap_minimum_m >= self.precontact_goal_gap_maximum_m:
            raise ValueError("precontact goal gap interval is empty")
        if not (0.0 < self.precision_push_coverage_start < self.fine_push_coverage_start < 1.0):
            raise ValueError("precision/fine push coverage thresholds must be ordered in (0, 1)")
        if not (self.fine_side_push_step_m < self.precision_side_push_step_m < self.side_push_step_m):
            raise ValueError("fine, precision, and ordinary push steps must be strictly ordered")
        semantic_thresholds = PushSideContactThresholds()
        if (
            self.central_full_push_minimum_normal_alignment
            < max(
                semantic_thresholds.minimum_contact_normal_push_alignment,
                _CENTRAL_FULL_PUSH_NORMAL_ALIGNMENT_SAFETY_FLOOR,
            )
            or self.central_full_push_minimum_rear_support_ratio
            < max(
                semantic_thresholds.minimum_rear_support_ratio,
                _CENTRAL_FULL_PUSH_REAR_SUPPORT_SAFETY_FLOOR,
            )
            or self.central_full_push_minimum_normal_alignment > self.edge_full_push_minimum_normal_alignment
            or self.central_full_push_minimum_rear_support_ratio
            > self.edge_full_push_minimum_rear_support_ratio
        ):
            raise ValueError(
                "central full-push margins must stay above the ordinary "
                "contact gate and no stricter than the edge margins"
            )
        if self.edge_full_push_minimum_normal_alignment < max(
            semantic_thresholds.minimum_edge_normal_push_alignment,
            _EDGE_FULL_PUSH_NORMAL_ALIGNMENT_SAFETY_FLOOR,
        ) or self.edge_full_push_minimum_rear_support_ratio < max(
            semantic_thresholds.minimum_edge_rear_support_ratio,
            _EDGE_FULL_PUSH_REAR_SUPPORT_SAFETY_FLOOR,
        ):
            raise ValueError("full edge-push margins cannot be weaker than the frozen execution safety floor")
        if not isinstance(self.invalid_contact_escape_steps, int) or not (
            1 <= self.invalid_contact_escape_steps <= 12
        ):
            raise ValueError("invalid_contact_escape_steps must be an integer in [1, 12]")
        if not isinstance(self.operational_tracking_failure_replan_score, int) or not (
            1 <= self.operational_tracking_failure_replan_score <= 12
        ):
            raise ValueError("operational_tracking_failure_replan_score must be an integer in [1, 12]")
        if self.precontact_joint_command_step_rad > 0.055:
            raise ValueError("precontact_joint_command_step_rad must be at most 0.055")
        if not (
            self.joint_limit_recovery_trigger_tolerance_rad
            < self.joint_limit_recovery_inset_rad
            < self.joint_limit_recovery_step_rad
        ):
            raise ValueError("joint-limit recovery tolerance, inset, and command step must be ordered")
        if not isinstance(self.runtime_clearance_recovery_stable_steps, int) or not (
            1 <= self.runtime_clearance_recovery_stable_steps <= 12
        ):
            raise ValueError("runtime_clearance_recovery_stable_steps must be an integer in [1, 12]")
        if self.runtime_clearance_recovery_joint_step_rad > 0.055:
            raise ValueError("runtime clearance recovery joint step must be at most 0.055")
        self.v9_config()

    def v9_config(self) -> PhysicalExpertV9Config:
        return PhysicalExpertV9Config(
            contact_standoff_m=self.contact_standoff_m,
            tool_height_m=self.tool_height_m,
            approach_step_limit_m=self.approach_step_limit_m,
            push_step_m=self.push_step_m,
            push_gate_along_m=self.push_gate_along_m,
            push_gate_lateral_m=self.push_gate_lateral_m,
        )


class PhysicalClosedLoopExpertV11(PhysicalClosedLoopExpertV9):
    """Privileged V9 teacher preceded by an executed joint-space waypoint."""

    teacher_type = "physical_expert_v11_synthetic_privileged"
    selected_update = 11
    checkpoint = ""

    def __init__(
        self,
        env: RealisticEdgeArmEnvV7 | None = None,
        config: PhysicalExpertV11Config | None = None,
    ) -> None:
        if env is not None and not isinstance(env, RealisticEdgeArmEnvV7):
            raise TypeError("PhysicalClosedLoopExpertV11 requires RealisticEdgeArmEnvV7")
        self.v11_config = config or PhysicalExpertV11Config()
        self._precontact_joint_target = np.zeros(_JOINTS, dtype=np.float64)
        self._precontact_target_xyz = np.zeros(3, dtype=np.float64)
        self._precontact_joint_waypoints: list[np.ndarray] = []
        self._precontact_xyz_waypoints: list[np.ndarray] = []
        self._precontact_stage_names: tuple[str, ...] = ("unplanned",)
        self._precontact_stage_index = 0
        self._precontact_block_reference = np.zeros(2, dtype=np.float64)
        self._precontact_direction = np.zeros(2, dtype=np.float64)
        self._precontact_complete = False
        self._precontact_steps = 0
        self._precontact_replans = 0
        self._precontact_unexpected_contact_steps = 0
        self._precontact_forbidden_desk_steps = 0
        self._precontact_last_trace_step = -1
        self._global_reacquisition_count = 0
        self._valid_push_side_contact_steps = 0
        self._invalid_raw_contact_steps = 0
        self._invalid_contact_advance_attempts = 0
        self._contact_semantics_last_step = -1
        self._precontact_mode = "initial_alignment"
        self._precontact_path_audit: dict[str, Any] = {}
        self._precontact_side_ik_report: dict[str, Any] = {}
        self._precontact_joint_path_report: dict[str, Any] = {}
        self._precontact_plan_feasible = False
        self._invalid_contact_escape_remaining = 0
        self._side_contact_ik_planner: SideContactIKPlannerV1 | None = None
        self._invalid_contact_escape_total_steps = 0
        self._operational_tracking_failure_score = 0
        self._joint_limit_recovery_steps = 0
        self._runtime_clearance_recovery_active = False
        self._runtime_clearance_recovery_resume_operational = False
        self._runtime_clearance_recovery_steps = 0
        self._runtime_clearance_recovery_stable_count = 0
        self._runtime_clearance_recovery_joint_target = np.zeros(_JOINTS, dtype=np.float64)
        self._operational_height_reference_m = float("nan")
        self._adaptive_edge_push_inflight_hold_total_steps = 0
        self._contact_acquisition_inflight_hold_total_steps = 0
        self._pending_effect_epoch = -1
        self._pending_effect_command_id = -1
        self._pending_effect_kind = ""
        self._pending_effect_send_step = -1
        self._pending_effect_wait_steps = 0
        self._pending_effect_total_wait_steps = 0
        self._pending_effect_vertical_recentering = False
        self._pending_effect_vertical_recentering_signed_delta_z_m = 0.0
        self._last_vertical_recenter_requested_delta_z_m = 0.0
        self._last_vertical_recenter_realized_delta_z_m = 0.0
        self._last_vertical_recenter_completion_success = False
        self._contact_clearance_bridge_active = False
        self._contact_clearance_bridge_failed = False
        self._contact_clearance_bridge_failure_reason = ""
        self._contact_clearance_bridge_target = np.zeros(_JOINTS, dtype=np.float64)
        self._contact_clearance_bridge_epoch = -1
        self._contact_clearance_bridge_first_command_id = -1
        self._contact_clearance_bridge_next_audit_command_id = -1
        self._contact_clearance_bridge_last_submitted_command_id = -1
        self._contact_clearance_bridge_audited_effects = 0
        self._contact_clearance_bridge_last_start_audit: dict[str, Any] = {}
        self._contact_clearance_bridge_tracking: dict[str, Any] = {}
        self._last_consumed_applied_command_id = -1
        self._last_consumed_effect_kind = ""
        self._command_feedback_gap_count = 0
        self._adaptive_edge_reacquisition_count = 0
        super().__init__(env=env, config=self.v11_config.v9_config())
        if env is not None:
            self.reset(env)

    def reset(self, env: RealisticEdgeArmEnvV6 | None = None) -> None:
        if env is not None and not isinstance(env, RealisticEdgeArmEnvV7):
            raise TypeError("PhysicalClosedLoopExpertV11 requires RealisticEdgeArmEnvV7")
        super().reset(env)
        if not isinstance(self.env, RealisticEdgeArmEnvV7):
            raise TypeError("PhysicalClosedLoopExpertV11 requires RealisticEdgeArmEnvV7")
        self._precontact_complete = False
        self._precontact_steps = 0
        self._precontact_replans = 0
        self._precontact_unexpected_contact_steps = 0
        self._precontact_forbidden_desk_steps = 0
        self._precontact_last_trace_step = int(self.env.step_count)
        self._global_reacquisition_count = 0
        self._valid_push_side_contact_steps = 0
        self._invalid_raw_contact_steps = 0
        self._invalid_contact_advance_attempts = 0
        self._contact_semantics_last_step = -1
        self._precontact_mode = "initial_alignment"
        self._invalid_contact_escape_remaining = 0
        self._invalid_contact_escape_total_steps = 0
        self._operational_tracking_failure_score = 0
        self._joint_limit_recovery_steps = 0
        self._runtime_clearance_recovery_active = False
        self._runtime_clearance_recovery_resume_operational = False
        self._runtime_clearance_recovery_steps = 0
        self._runtime_clearance_recovery_stable_count = 0
        self._runtime_clearance_recovery_joint_target = self.env.data.qpos[:_JOINTS].copy()
        self._operational_height_reference_m = float("nan")
        self._adaptive_edge_push_inflight_hold_total_steps = 0
        self._contact_acquisition_inflight_hold_total_steps = 0
        self._pending_effect_epoch = -1
        self._pending_effect_command_id = -1
        self._pending_effect_kind = ""
        self._pending_effect_send_step = -1
        self._pending_effect_wait_steps = 0
        self._pending_effect_total_wait_steps = 0
        self._pending_effect_vertical_recentering = False
        self._pending_effect_vertical_recentering_signed_delta_z_m = 0.0
        self._last_vertical_recenter_requested_delta_z_m = 0.0
        self._last_vertical_recenter_realized_delta_z_m = 0.0
        self._last_vertical_recenter_completion_success = False
        self._contact_clearance_bridge_active = False
        self._contact_clearance_bridge_failed = False
        self._contact_clearance_bridge_failure_reason = ""
        self._contact_clearance_bridge_target = np.zeros(_JOINTS, dtype=np.float64)
        self._contact_clearance_bridge_epoch = -1
        self._contact_clearance_bridge_first_command_id = -1
        self._contact_clearance_bridge_next_audit_command_id = -1
        self._contact_clearance_bridge_last_submitted_command_id = -1
        self._contact_clearance_bridge_audited_effects = 0
        self._contact_clearance_bridge_last_start_audit = {}
        self._contact_clearance_bridge_tracking = {}
        self._last_consumed_applied_command_id = -1
        self._last_consumed_effect_kind = ""
        self._command_feedback_gap_count = 0
        self._adaptive_edge_reacquisition_count = 0
        self._side_contact_ik_planner = self._make_side_contact_ik_planner()
        self._plan_precontact_waypoints()

    def _make_side_contact_ik_planner(self) -> SideContactIKPlannerV1:
        if not isinstance(self.env, RealisticEdgeArmEnvV7):  # pragma: no cover
            raise RuntimeError("side-contact IK planner requires a V7 environment")
        return SideContactIKPlannerV1(
            self.env,
            SideContactIKConfig(
                minimum_pusher_desk_clearance_m=(self.v11_config.minimum_pusher_desk_clearance_m)
            ),
        )

    def _plan_precontact_waypoints(self) -> None:
        if not isinstance(self.env, RealisticEdgeArmEnvV7):
            raise RuntimeError("pre-contact planning requires a V7 environment")
        block = self.env.block_xy()
        direction = self.env._unit(self.env.target_xy - block)
        if self._side_contact_ik_planner is None:
            self._side_contact_ik_planner = self._make_side_contact_ik_planner()
        ik_report = self._side_contact_ik_planner.solve(
            direction,
            initial=self.env.data.qpos[:_JOINTS].copy(),
        )
        self._precontact_side_ik_report = ik_report
        path_report: dict[str, Any] = {
            "feasible": False,
            "failure_code": "side_contact_ik_infeasible",
            "joint_waypoints_rad": [],
            "live_state_written": False,
        }
        if bool(ik_report["feasible"]):
            goal = np.asarray(ik_report["best"]["joint_position_rad"], dtype=np.float64)
            path_report = JointPathPlannerV1(self.env).solve(
                self.env.data.qpos[:_JOINTS].copy(),
                goal,
            )
        self._precontact_joint_path_report = path_report
        self._precontact_plan_feasible = bool(ik_report["feasible"] and path_report["feasible"])
        self._precontact_path_audit = {
            "method": "five_constraint_side_ik_plus_dense_audited_joint_rrt_connect",
            "live_state_written": False,
            "mode": self._precontact_mode,
            "valid": self._precontact_plan_feasible,
            "side_contact_ik": ik_report,
            "joint_path": path_report,
        }
        self._precontact_joint_waypoints = []
        self._precontact_xyz_waypoints = []
        self._precontact_stage_names = ()
        if self._precontact_plan_feasible:
            path = np.asarray(path_report["joint_waypoints_rad"], dtype=np.float64)
            joint_waypoints = [row.copy() for row in path[1:]]
            goal = np.asarray(ik_report["best"]["joint_position_rad"], dtype=np.float64)
            if not joint_waypoints or not np.allclose(joint_waypoints[-1], goal, rtol=0.0, atol=1.0e-10):
                joint_waypoints.append(goal.copy())
            scratch = mujoco.MjData(self.env.model)
            xyz_waypoints: list[np.ndarray] = []
            for joint_position in joint_waypoints:
                scratch.qpos[:] = self.env.data.qpos
                scratch.qvel[:] = 0.0
                scratch.qpos[:_JOINTS] = joint_position
                mujoco.mj_forward(self.env.model, scratch)
                xyz_waypoints.append(scratch.site_xpos[self.env._ids["tool_site"]].copy())
            self._precontact_joint_waypoints = joint_waypoints
            self._precontact_xyz_waypoints = xyz_waypoints
            self._precontact_stage_names = tuple(
                "verified_side_precontact"
                if index == len(joint_waypoints) - 1
                else f"collision_free_transit_{index + 1:02d}"
                for index in range(len(joint_waypoints))
            )
            self._precontact_stage_index = 0
            self._activate_precontact_stage()
        self._precontact_block_reference = block.copy()
        self._precontact_direction = direction.copy()
        self._operational_tracking_failure_score = 0

    def _activate_precontact_stage(self) -> None:
        self._precontact_joint_target = self._precontact_joint_waypoints[self._precontact_stage_index].copy()
        self._precontact_target_xyz = self._precontact_xyz_waypoints[self._precontact_stage_index].copy()

    def action(
        self,
        env_or_observation: RealisticEdgeArmEnvV6 | Mapping[str, np.ndarray] | None = None,
        observation: Mapping[str, np.ndarray] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        if isinstance(env_or_observation, RealisticEdgeArmEnvV6):
            if not isinstance(env_or_observation, RealisticEdgeArmEnvV7):
                raise TypeError("PhysicalClosedLoopExpertV11 requires RealisticEdgeArmEnvV7")
            if env_or_observation is not self.env:
                self.reset(env_or_observation)
            current_observation = observation
        else:
            if observation is not None:
                raise TypeError("two-argument action requires the environment first")
            current_observation = env_or_observation
        if not isinstance(self.env, RealisticEdgeArmEnvV7):
            raise RuntimeError("expert.action requires a RealisticEdgeArmEnvV7")
        self._validate_observation(current_observation)
        if self.env.step_count == 0 and self._precontact_steps > 0:
            self.reset()

        pending_resolution = self._resolve_pending_command_effect()
        if pending_resolution is not None:
            return pending_resolution
        if self._contact_clearance_bridge_active:
            return self._contact_clearance_bridge_action()

        recovery_required = self._runtime_clearance_recovery_required()
        if self._runtime_clearance_recovery_active or recovery_required:
            evidence = self._runtime_clearance_recovery_evidence()
            trace = self.env._physics_substep_contact_v1
            valid_contact = bool(trace is not None and bool(trace.get("valid_push_side_contact_any", False)))
            if not self._runtime_clearance_recovery_active:
                self._runtime_clearance_recovery_resume_operational = bool(
                    self._precontact_complete
                    and evidence["preflight_changed"]
                    and evidence["preflight_feasible"]
                    and not evidence["live_warning"]
                    and valid_contact
                )
            elif self._runtime_clearance_recovery_resume_operational and (
                evidence["live_warning"] or not evidence["preflight_feasible"] or not valid_contact
            ):
                # A forward-looking, feasible rewrite may preserve the
                # operational state.  Any later live warning, infeasibility,
                # invalid contact or contact loss revokes that privilege and
                # restores the original fail-closed reacquisition path.
                self._runtime_clearance_recovery_resume_operational = False
            self._runtime_clearance_recovery_active = True
            self._precontact_mode = "runtime_clearance_recovery"
            if not self._runtime_clearance_recovery_resume_operational:
                self._precontact_complete = False
            if not self._runtime_clearance_recovery_complete():
                return self._runtime_clearance_recovery_action()
            self._runtime_clearance_recovery_active = False
            self._runtime_clearance_recovery_stable_count = 0
            if self._runtime_clearance_recovery_resume_operational:
                self._runtime_clearance_recovery_resume_operational = False
                self._precontact_complete = True
                self._precontact_mode = "runtime_clearance_operational_resume"
            else:
                self._precontact_mode = "runtime_clearance_reacquisition"
                self._plan_precontact_waypoints()

        if self._precontact_mode == "joint_limit_recovery":
            if not self._joint_limit_recovery_complete():
                return self._joint_limit_recovery_action()
            self._precontact_mode = "joint_limit_reacquisition"
            self._precontact_complete = False
            self._plan_precontact_waypoints()
        elif (
            self._authored_joint_limit_violation_linf_rad()
            > self.v11_config.joint_limit_recovery_trigger_tolerance_rad
        ):
            self._precontact_mode = "joint_limit_recovery"
            self._precontact_complete = False
            return self._joint_limit_recovery_action()

        if self._precontact_mode == "invalid_contact_escape":
            invalid_contact_still_present = self._invalid_push_contact_present()
            if self._invalid_contact_escape_remaining > 0 or invalid_contact_still_present:
                return self._invalid_contact_escape_action()
            self._precontact_mode = "invalid_contact_reacquisition"
            self._plan_precontact_waypoints()

        if not self._precontact_plan_feasible:
            return self._precontact_plan_infeasible_action()

        if not self._precontact_complete:
            self._ingest_precontact_execution_telemetry()
            trace = self.env._physics_substep_contact_v1
            trace_contact = bool(trace is not None and bool(trace["contact_any"]))
            trace_valid_side = bool(trace is not None and bool(trace["valid_push_side_contact_any"]))
            trace_geometric_side = bool(
                trace is not None and np.any(np.asarray(trace["geometric_push_side_contact_count"]) > 0)
            )
            unexpected_contact = bool(trace_contact and not trace_valid_side and not trace_geometric_side)
            if unexpected_contact:
                self._invalid_raw_contact_steps += 1
                self._precontact_mode = "invalid_contact_escape"
                self._invalid_contact_escape_remaining = self.v11_config.invalid_contact_escape_steps
                return self._invalid_contact_escape_action()
            if trace_contact and (trace_valid_side or trace_geometric_side):
                # The executed waypoint reached the intended push face a few
                # substeps before its kinematic arrival tolerance.  This is an
                # early contact-boundary arrival, not an invalid collision.
                # Hand control to semantic acquisition immediately so the
                # remaining waypoint cannot drive through the block.
                self._precontact_complete = True
                self._operational_height_reference_m = float(self.env.tool_xyz()[2])
            if (
                np.linalg.norm(self.env.block_xy() - self._precontact_block_reference)
                > self.v11_config.precontact_replan_block_motion_m
            ):
                self._plan_precontact_waypoints()
                self._precontact_replans += 1
                if not self._precontact_plan_feasible:
                    return self._precontact_plan_infeasible_action()
            diagnostics = self._precontact_diagnostics()
            if bool(diagnostics["arrival_verified"]):
                if self._precontact_stage_index + 1 < len(self._precontact_joint_waypoints):
                    self._precontact_stage_index += 1
                    self._activate_precontact_stage()
                    diagnostics = self._precontact_diagnostics()
                else:
                    self._precontact_complete = True
                    # The force-limited plant can settle several millimetres
                    # away from the kinematic waypoint.  Operational IK must
                    # preserve the measured, collision-audited arrival height
                    # instead of snapping back to an unreachable nominal z.
                    self._operational_height_reference_m = float(self.env.tool_xyz()[2])
            if not self._precontact_complete:
                return self._precontact_action(diagnostics)

        previous_trace = self.env._physics_substep_contact_v1
        invalid_previous_contact = bool(
            previous_trace is not None
            and bool(previous_trace["contact_any"])
            and not bool(previous_trace["valid_push_side_contact_any"])
            and not np.any(np.asarray(previous_trace["geometric_push_side_contact_count"]) > 0)
        )
        if invalid_previous_contact:
            self._invalid_raw_contact_steps += 1
            self._contact_semantics_last_step = int(self.env.step_count)
            self._had_contact = False
            self._contact_loss_steps = 0
            self._clear_pending_command_effect()
            self._precontact_mode = "invalid_contact_escape"
            self._precontact_complete = False
            self._invalid_contact_escape_remaining = self.v11_config.invalid_contact_escape_steps
            return self._invalid_contact_escape_action()

        tracking_replan_required = bool(
            self._operational_tracking_failure_score
            >= self.v11_config.operational_tracking_failure_replan_score
        )
        if (
            self._requires_global_reacquisition() or tracking_replan_required
        ) and self._global_reacquisition_count < self.v11_config.max_global_reacquisitions:
            self._global_reacquisition_count += 1
            self._precontact_mode = "global_reacquisition"
            self._precontact_complete = False
            self._plan_precontact_waypoints()
            if not self._precontact_plan_feasible:
                return self._precontact_plan_infeasible_action()
            diagnostics = self._precontact_diagnostics()
            return self._precontact_action(diagnostics)

        action, metadata = self._v11_operational_action()
        if metadata["phase"] == "side_pose_tracking_infeasible":
            self._operational_tracking_failure_score += 1
        else:
            self._operational_tracking_failure_score = max(
                self._operational_tracking_failure_score - 1,
                0,
            )
        if bool(metadata.get("contact_geometry_adaptive_edge_step_limited", False)):
            self._commit_pending_command_effect(
                "adaptive_edge_push",
                vertical_recentering_signed_delta_z_m=float(
                    metadata.get("contact_geometry_vertical_recentering_correction_m", 0.0)
                    if metadata.get("contact_geometry_vertical_recentering_commanded", False)
                    else 0.0
                ),
            )
        metadata.update(self._common_metadata())
        metadata.update(
            {
                "version": PHYSICAL_EXPERT_V11_VERSION,
                "delegate_version": PHYSICAL_EXPERT_V9_VERSION,
                "precontact_complete": True,
                "precontact_steps": self._precontact_steps,
                "precontact_replan_count": self._precontact_replans,
                "planned_phase": metadata["phase"],
                "phase_semantics": "teacher_plan_before_env_step",
            }
        )
        return action, metadata

    def _authored_joint_limit_violation_linf_rad(self) -> float:
        if not isinstance(self.env, RealisticEdgeArmEnvV7):  # pragma: no cover
            raise RuntimeError("joint-limit audit requires a V7 environment")
        q = self.env.data.qpos[:5]
        lower = self.env.joint_ranges[:5, 0]
        upper = self.env.joint_ranges[:5, 1]
        violation = np.maximum(np.maximum(lower - q, q - upper), 0.0)
        return float(np.max(violation))

    def _runtime_clearance_recovery_required(self) -> bool:
        """React to the previous executed transition's desk-barrier warning."""

        evidence = self._runtime_clearance_recovery_evidence()
        return bool(evidence["enabled"] and evidence["warning"])

    def _runtime_clearance_recovery_evidence(self) -> dict[str, bool]:
        """Classify causal warning evidence from the immediately prior effect."""

        if not isinstance(self.env, RealisticEdgeArmEnvV7):  # pragma: no cover
            raise RuntimeError("runtime clearance recovery requires a V7 environment")
        guard = self.env._runtime_pusher_desk_guard_v1
        enabled = bool(guard and guard.get("enabled", False))
        preflight = guard.get("controller_preflight")
        preflight_changed = bool(isinstance(preflight, Mapping) and preflight.get("changed", False))
        preflight_feasible = bool(
            not isinstance(preflight, Mapping) or preflight.get("dynamic_forecast_feasible") is not False
        )
        live_warning = bool(guard.get("dynamic_forecast_triggered", False) or guard.get("triggered", False))
        warning = bool(live_warning or preflight_changed or not preflight_feasible)
        return {
            "enabled": enabled,
            "warning": warning,
            "live_warning": live_warning,
            "preflight_changed": preflight_changed,
            "preflight_feasible": preflight_feasible,
        }

    def _current_tool_desk_clearance_m(self) -> float:
        """Return the V11 single-tool desk distance.

        This indirection deliberately preserves the frozen V11 geometry while
        allowing stock-gripper successors to audit their complete physical
        safety union without reimplementing the recovery state machine.
        """

        if not isinstance(self.env, RealisticEdgeArmEnvV7):  # pragma: no cover
            raise RuntimeError("tool/desk clearance requires a V7 environment")
        return float(
            self.env._geom_box_signed_distance(
                self.env._ids["tool_geom"],
                self.env._desk_geom,
            )
        )

    def _current_tool_block_gap_interval_m(self) -> tuple[float, float]:
        """Return the V11 single-tool block distance as a degenerate interval."""

        if not isinstance(self.env, RealisticEdgeArmEnvV7):  # pragma: no cover
            raise RuntimeError("tool/block gap requires a V7 environment")
        gap = float(
            self.env._geom_box_signed_distance(
                self.env._ids["tool_geom"],
                self.env._ids["block_geom"],
            )
        )
        return gap, gap

    def _runtime_clearance_recovery_complete(self) -> bool:
        """Require geometric reserve and a warning-free dynamic forecast."""

        if not isinstance(self.env, RealisticEdgeArmEnvV7):  # pragma: no cover
            raise RuntimeError("runtime clearance recovery requires a V7 environment")
        config = self.env.contact_feasible_config
        margin = float(config.runtime_pusher_desk_clearance_m)
        reserve = float(config.runtime_pusher_desk_guard_reserve_m)
        if margin <= 0.0:
            return True
        current_clearance = self._current_tool_desk_clearance_m()
        target = self._runtime_clearance_recovery_joint_target
        forecast = self.env._pusher_desk_dynamic_clearance_forecast(target)
        guard = self.env._runtime_pusher_desk_guard_v1
        warning_free = not bool(
            guard.get("dynamic_forecast_triggered", False) or guard.get("triggered", False)
        )
        stable = bool(
            current_clearance >= margin + reserve - 1.0e-5
            and float(np.min(forecast)) >= margin + reserve - 1.0e-5
            and warning_free
        )
        self._runtime_clearance_recovery_stable_count = (
            self._runtime_clearance_recovery_stable_count + 1 if stable else 0
        )
        return bool(
            self._runtime_clearance_recovery_stable_count
            >= self.v11_config.runtime_clearance_recovery_stable_steps
        )

    def _runtime_clearance_gradient_target(self) -> np.ndarray:
        """Build a scratch-audited joint target that increases exact clearance."""

        if not isinstance(self.env, RealisticEdgeArmEnvV7):  # pragma: no cover
            raise RuntimeError("runtime clearance recovery requires a V7 environment")
        current = self.env.data.qpos[:_JOINTS].copy()
        epsilon = float(self.env.contact_feasible_config.runtime_pusher_desk_guard_finite_difference_rad)
        gradient = np.zeros(_JOINTS, dtype=np.float64)
        for joint_index in range(_JOINTS - 1):
            below = current.copy()
            above = current.copy()
            below[joint_index] = max(
                self.env.joint_ranges[joint_index, 0],
                below[joint_index] - epsilon,
            )
            above[joint_index] = min(
                self.env.joint_ranges[joint_index, 1],
                above[joint_index] + epsilon,
            )
            width = float(above[joint_index] - below[joint_index])
            if width <= 1.0e-12:
                continue
            gradient[joint_index] = (
                self.env._pusher_desk_distance_at_joint_position(above)
                - self.env._pusher_desk_distance_at_joint_position(below)
            ) / width
        denominator = float(np.dot(gradient, gradient))
        if denominator <= 1.0e-12:
            return current
        config = self.env.contact_feasible_config
        current_clearance = self._current_tool_desk_clearance_m()
        desired = float(
            config.runtime_pusher_desk_clearance_m
            + config.runtime_pusher_desk_guard_reserve_m
            + config.runtime_pusher_desk_guard_recovery_step_m
        )
        delta = max(desired - current_clearance, 0.0) * gradient / denominator
        linf = float(np.max(np.abs(delta)))
        if linf > self.v11_config.runtime_clearance_recovery_joint_step_rad:
            delta *= self.v11_config.runtime_clearance_recovery_joint_step_rad / linf
        target = current + delta
        target[:5] = np.clip(
            target[:5],
            self.env.joint_ranges[:5, 0],
            self.env.joint_ranges[:5, 1],
        )
        target[5] = current[5]
        return target

    def _runtime_clearance_recovery_action(self) -> tuple[np.ndarray, dict[str, Any]]:
        """Latch the barrier's safe target instead of returning a zero hold."""

        if not isinstance(self.env, RealisticEdgeArmEnvV7):  # pragma: no cover
            raise RuntimeError("runtime clearance recovery requires a V7 environment")
        current = self.env.data.qpos[:_JOINTS].copy()
        guard = self.env._runtime_pusher_desk_guard_v1
        candidates = [self._runtime_clearance_gradient_target()]
        applied = np.asarray(guard.get("physics_applied_joint_target", []), dtype=np.float64)
        if applied.shape == (_JOINTS,) and np.all(np.isfinite(applied)):
            candidates.append(applied.copy())
        candidates.append(self._runtime_clearance_recovery_joint_target.copy())
        target = max(
            candidates,
            key=lambda value: self.env._pusher_desk_distance_at_joint_position(value),
        ).copy()
        delta = target - current
        linf = float(np.max(np.abs(delta)))
        if linf > self.v11_config.runtime_clearance_recovery_joint_step_rad:
            delta *= self.v11_config.runtime_clearance_recovery_joint_step_rad / linf
            target = current + delta
        action = delta / self.env.config.max_joint_delta
        action, prefilter_reason = self._prefilter_action(action)
        action = np.asarray(action, dtype=np.float32)
        self._runtime_clearance_recovery_joint_target = target.copy()
        self._runtime_clearance_recovery_steps += 1
        self._precontact_steps += 1
        self._last_step_count = int(self.env.step_count)
        current_clearance = self._current_tool_desk_clearance_m()
        forecast = self.env._pusher_desk_dynamic_clearance_forecast(target)
        metadata = self._common_metadata()
        metadata.update(
            {
                "version": PHYSICAL_EXPERT_V11_VERSION,
                "phase": "runtime_clearance_recovery",
                "planned_phase": "runtime_clearance_recovery",
                "phase_semantics": "teacher_plan_before_env_step",
                "teacher_confidence": 0.0,
                "precontact_complete": False,
                "runtime_clearance_recovery_target_rad": target.astype(np.float32),
                "runtime_clearance_recovery_current_m": current_clearance,
                "runtime_clearance_recovery_forecast_minimum_m": float(np.min(forecast)),
                "runtime_clearance_recovery_steps": self._runtime_clearance_recovery_steps,
                "runtime_clearance_recovery_stable_count": (self._runtime_clearance_recovery_stable_count),
                "workspace_prefilter_applied": bool(prefilter_reason),
                "workspace_prefilter_reason": prefilter_reason,
                "normalized_action_linf": float(np.max(np.abs(action))),
            }
        )
        return action, metadata

    def _joint_limit_recovery_complete(self) -> bool:
        if not isinstance(self.env, RealisticEdgeArmEnvV7):  # pragma: no cover
            raise RuntimeError("joint-limit recovery requires a V7 environment")
        inset = self.v11_config.joint_limit_recovery_inset_rad
        q = self.env.data.qpos[:5]
        lower = self.env.joint_ranges[:5, 0] + inset
        upper = self.env.joint_ranges[:5, 1] - inset
        return bool(np.all(q >= lower) and np.all(q <= upper))

    def _joint_limit_recovery_action(self) -> tuple[np.ndarray, dict[str, Any]]:
        """Command a small inward move before handing state back to RRT."""

        if not isinstance(self.env, RealisticEdgeArmEnvV7):  # pragma: no cover
            raise RuntimeError("joint-limit recovery requires a V7 environment")
        current = self.env.data.qpos[:_JOINTS].copy()
        inset = self.v11_config.joint_limit_recovery_inset_rad
        target = current.copy()
        target[:5] = np.clip(
            current[:5],
            self.env.joint_ranges[:5, 0] + inset,
            self.env.joint_ranges[:5, 1] - inset,
        )
        delta = target - current
        linf = float(np.max(np.abs(delta)))
        if linf > self.v11_config.joint_limit_recovery_step_rad:
            delta *= self.v11_config.joint_limit_recovery_step_rad / linf
            target = current + delta
        action = delta / self.env.config.max_joint_delta
        action, prefilter_reason = self._prefilter_action(action)
        action = np.asarray(action, dtype=np.float32)
        self._joint_limit_recovery_steps += 1
        self._precontact_steps += 1
        self._last_step_count = int(self.env.step_count)
        metadata = self._common_metadata()
        metadata.update(
            {
                "version": PHYSICAL_EXPERT_V11_VERSION,
                "phase": "joint_limit_recovery",
                "planned_phase": "joint_limit_recovery",
                "phase_semantics": "teacher_plan_before_env_step",
                "teacher_confidence": 0.0,
                "precontact_complete": False,
                "joint_limit_recovery_target_rad": target.astype(np.float32),
                "joint_limit_violation_linf_rad": (self._authored_joint_limit_violation_linf_rad()),
                "joint_limit_recovery_steps": self._joint_limit_recovery_steps,
                "workspace_prefilter_applied": bool(prefilter_reason),
                "workspace_prefilter_reason": prefilter_reason,
                "normalized_action_linf": float(np.max(np.abs(action))),
            }
        )
        return action, metadata

    def _v11_operational_action(self) -> tuple[np.ndarray, dict[str, Any]]:
        """Run V11 side-normal tracking without the legacy V7/V8 rewrites.

        V7 independently rescales the base and the four movable arm joints and
        treats any raw contact followed by two contact-free decisions as a
        reposition request.  Both operations invalidate a five-constraint IK
        command: the former changes its Cartesian direction and face normal,
        while the latter retracts after normal intermittent sliding contact.
        V11 therefore owns the operational and terminal state machine.  The
        command is still executed through the unchanged delayed, rate-limited,
        force-limited MuJoCo plant.
        """

        if not isinstance(self.env, RealisticEdgeArmEnvV7):  # pragma: no cover
            raise RuntimeError("V11 operational action requires a V7 environment")
        block = self.env.block_xy()
        target = self.env.target_xy.copy()
        direct_direction = self.env._unit(target - block)
        route_direction, obstacle_phase = self._route_direction(
            block,
            target,
            direct_direction,
        )
        coverage = float(self.env.block_target_coverage())
        threshold = float(self.env.realism_config.strict_coverage_threshold)
        linear_speed, angular_speed = self.env._block_speeds()
        moving = bool(
            linear_speed > self.env.realism_config.strict_linear_speed_m_s
            or angular_speed > self.env.realism_config.strict_angular_speed_rad_s
        )
        current_contact = bool(self.env._tool_block_contacts() > 0)
        trace = self.env._physics_substep_contact_v1
        trace_contact = bool(trace is not None and bool(trace["contact_any"]))

        if coverage >= threshold:
            if not self._settle_active:
                self._settle_active = True
                self._settle_age = 0
            self._settle_age += 1
            raw_metadata: dict[str, Any] = {
                "phase": "released_hold",
                "teacher_confidence": 1.0,
                "joint_target": self.env.data.qpos[:_JOINTS].copy().astype(np.float32),
                "contact_geometry_actual_contact": bool(current_contact or trace_contact),
                "contact_geometry_current_contact": current_contact,
                "contact_geometry_trace_raw_contact": trace_contact,
                "contact_geometry_verified_contact_gate": False,
                "contact_geometry_advance_gate": False,
                "contact_geometry_terminal_unload": False,
            }
            raw_action = np.zeros(_JOINTS, dtype=np.float64)
            if current_contact or trace_contact or moving:
                tool = self.env.tool_xyz()
                unload_target = tool + np.array(
                    [
                        -direct_direction[0] * self.v11_config.terminal_unload_step_m,
                        -direct_direction[1] * self.v11_config.terminal_unload_step_m,
                        self.v11_config.terminal_lift_step_m,
                    ],
                    dtype=np.float64,
                )
                if self._side_contact_ik_planner is None:
                    self._side_contact_ik_planner = self._make_side_contact_ik_planner()
                tracking = self._side_contact_ik_planner.track_target(
                    unload_target,
                    direct_direction,
                    initial=self.env.data.qpos[:_JOINTS].copy(),
                )
                if bool(tracking["feasible"]):
                    joint_target = np.asarray(
                        tracking["best"]["joint_position_rad"],
                        dtype=np.float64,
                    )
                    raw_action = np.clip(
                        (joint_target - self.env.data.qpos[:_JOINTS]) / self.env.config.max_joint_delta,
                        -1.0,
                        1.0,
                    )
                    raw_metadata.update(
                        {
                            "phase": (
                                "contact_unload" if current_contact or trace_contact else "velocity_brake"
                            ),
                            "joint_target": joint_target.astype(np.float32),
                            "contact_geometry_terminal_unload": True,
                            "contact_geometry_tracking_ik": tracking,
                            "contact_geometry_commanded_target_xyz_m": unload_target.tolist(),
                        }
                    )
                else:
                    raw_metadata.update(
                        {
                            "phase": "terminal_unload_tracking_infeasible_hold",
                            "teacher_confidence": 0.0,
                            "contact_geometry_tracking_ik": tracking,
                            "contact_geometry_commanded_target_xyz_m": unload_target.tolist(),
                        }
                    )
        else:
            self._settle_active = False
            self._settle_age = 0
            raw_action, raw_metadata = self._geometry_operational_action(
                route_direction,
                contact_opens_gate=True,
                environment_obstacle_detour=False,
            )

        action = np.clip(np.asarray(raw_action, dtype=np.float64), -1.0, 1.0)
        action, prefilter_reason = self._prefilter_action(action)
        action = np.asarray(action, dtype=np.float32)
        self._last_step_count = int(self.env.step_count)
        requested_target = (
            self.env.data.qpos[:_JOINTS]
            + self.env._encoder_position_noise
            + action * self.env.config.max_joint_delta
        )
        metadata = {
            **raw_metadata,
            "strict_target_coverage": coverage,
            "strict_coverage_threshold": threshold,
            "block_linear_speed_m_s": linear_speed,
            "block_angular_speed_rad_s": angular_speed,
            "settle_active": self._settle_active,
            "settle_age_steps": self._settle_age,
            "route_phase": obstacle_phase,
            "obstacle_route_active": bool(obstacle_phase == "waypoint"),
            "workspace_prefilter_applied": bool(prefilter_reason),
            "workspace_prefilter_reason": prefilter_reason,
            "raw_joint_target": np.asarray(raw_metadata["joint_target"], dtype=np.float32),
            "joint_target_semantics": "global_operational_ik_target_not_applied_transition_target",
            "teacher_requested_one_step_joint_target": requested_target.astype(np.float32),
            "normalized_action_linf": float(np.max(np.abs(action))),
            "legacy_v7_joint_rescaling_applied": False,
            "legacy_v7_raw_contact_recovery_applied": False,
        }
        return action, metadata

    def _precontact_plan_infeasible_action(self) -> tuple[np.ndarray, dict[str, Any]]:
        action = np.zeros(_JOINTS, dtype=np.float32)
        metadata = self._common_metadata()
        metadata.update(
            {
                "version": PHYSICAL_EXPERT_V11_VERSION,
                "phase": "precontact_plan_infeasible",
                "planned_phase": "precontact_plan_infeasible",
                "phase_semantics": "teacher_plan_before_env_step",
                "teacher_confidence": 0.0,
                "precontact_complete": False,
                "precontact_plan_feasible": False,
                "precontact_plan_failure_code": self._precontact_joint_path_report.get(
                    "failure_code", "unknown_precontact_plan_failure"
                ),
                "normalized_action_linf": 0.0,
            }
        )
        return action, metadata

    def _commit_pending_command_effect(
        self,
        kind: str,
        *,
        vertical_recentering_signed_delta_z_m: float = 0.0,
    ) -> None:
        """Bind a single-flight effect wait to the next real command ID."""

        if not isinstance(self.env, RealisticEdgeArmEnvV7):  # pragma: no cover
            raise RuntimeError("command effect commit requires a V7 environment")
        if self._pending_effect_command_id >= 0:
            raise RuntimeError("cannot commit a second command effect while one is pending")
        if kind not in {"adaptive_edge_push", "contact_acquisition"}:
            raise ValueError(f"unsupported pending command effect kind: {kind}")
        requested_delta_z = float(vertical_recentering_signed_delta_z_m)
        if not np.isfinite(requested_delta_z):
            raise ValueError("vertical recenter delta z must be finite")
        if kind != "adaptive_edge_push" and abs(requested_delta_z) > 0.0:
            raise ValueError("only adaptive edge pushes may bind a vertical recenter")
        self._pending_effect_epoch = int(self.env.command_epoch)
        self._pending_effect_command_id = int(self.env.next_command_id)
        self._pending_effect_kind = kind
        self._pending_effect_send_step = int(self.env.step_count)
        self._pending_effect_wait_steps = 0
        self._pending_effect_vertical_recentering = bool(abs(requested_delta_z) > 0.0)
        self._pending_effect_vertical_recentering_signed_delta_z_m = requested_delta_z

    def _clear_pending_command_effect(self) -> None:
        self._pending_effect_epoch = -1
        self._pending_effect_command_id = -1
        self._pending_effect_kind = ""
        self._pending_effect_send_step = -1
        self._pending_effect_wait_steps = 0
        self._pending_effect_vertical_recentering = False
        self._pending_effect_vertical_recentering_signed_delta_z_m = 0.0

    def _resolve_pending_command_effect(
        self,
    ) -> tuple[np.ndarray, dict[str, Any]] | None:
        """Wait for the exact applied ID, then consume its aligned effect."""

        if self._pending_effect_command_id < 0:
            return None
        if not isinstance(self.env, RealisticEdgeArmEnvV7):  # pragma: no cover
            raise RuntimeError("command effect resolution requires a V7 environment")
        feedback = self.env.last_command_feedback_v1
        epoch_matches = bool(feedback.get("epoch", -1) == self._pending_effect_epoch)
        applied_id = int(feedback.get("applied_id", -1))
        if (
            not bool(feedback.get("available", False))
            or not epoch_matches
            or applied_id < self._pending_effect_command_id
        ):
            self._pending_effect_wait_steps += 1
            self._pending_effect_total_wait_steps += 1
            return self._pending_command_effect_hold_action()
        if applied_id > self._pending_effect_command_id:
            self._command_feedback_gap_count += 1
            self._pending_effect_wait_steps += 1
            self._pending_effect_total_wait_steps += 1
            return self._pending_command_effect_hold_action(feedback_gap=True)

        kind = self._pending_effect_kind
        vertical_recentering = self._pending_effect_vertical_recentering
        vertical_recentering_requested_delta_z_m = self._pending_effect_vertical_recentering_signed_delta_z_m
        self._last_consumed_applied_command_id = applied_id
        self._last_consumed_effect_kind = kind
        self._clear_pending_command_effect()
        effect_valid = bool(feedback.get("effect_valid_push_side_contact_any", False))
        effect_raw = bool(feedback.get("effect_contact_any", False))
        vertical_recenter_realized_delta_z_m = 0.0
        vertical_recenter_completion_success = False
        if vertical_recentering:
            trace = self.env._physics_substep_contact_v1
            try:
                displacement = np.asarray(
                    trace["tool_xyz_displacement_m"] if trace is not None else (),
                    dtype=np.float64,
                )
                if displacement.shape != (3,) or not np.all(np.isfinite(displacement)):
                    raise ValueError("invalid exact-effect tool displacement")
                vertical_recenter_realized_delta_z_m = float(displacement[2])
                directional_progress_m = float(
                    np.sign(vertical_recentering_requested_delta_z_m) * vertical_recenter_realized_delta_z_m
                )
                vertical_recenter_completion_success = bool(
                    directional_progress_m > _VERTICAL_RECENTER_REALIZED_PROGRESS_TOLERANCE_M
                )
            except (KeyError, TypeError, ValueError):
                vertical_recenter_completion_success = False
            self._last_vertical_recenter_requested_delta_z_m = float(vertical_recentering_requested_delta_z_m)
            self._last_vertical_recenter_realized_delta_z_m = float(vertical_recenter_realized_delta_z_m)
            self._last_vertical_recenter_completion_success = bool(vertical_recenter_completion_success)
        vertical_recenter_invalid_contact = bool(vertical_recentering and effect_raw and not effect_valid)
        vertical_recenter_progress_failed = bool(
            vertical_recentering and not vertical_recenter_completion_success
        )
        adaptive_contact_lost = bool(
            kind == "adaptive_edge_push" and not vertical_recentering and not effect_valid
        )
        if vertical_recenter_invalid_contact:
            self._adaptive_edge_reacquisition_count += 1
            self._precontact_mode = "contact_clearance_bridge"
            self._precontact_complete = False
            return self._start_contact_clearance_bridge()
        if vertical_recenter_progress_failed or adaptive_contact_lost:
            self._adaptive_edge_reacquisition_count += 1
            self._precontact_mode = "adaptive_edge_contact_loss_reacquisition"
            self._precontact_complete = False
            self._plan_precontact_waypoints()
            if not self._precontact_plan_feasible:
                return self._precontact_plan_infeasible_action()
            return self._precontact_action(self._precontact_diagnostics())
        if vertical_recentering:
            # Contact-free completion proves only that the exact recenter was
            # safe; it does not prove that the delayed, force-limited plant
            # moved in the requested direction.  Anchor the measured height
            # only after the same applied-ID trace reports signed z progress.
            # Reverse/stalled motion and any raw-but-invalid contact take the
            # existing strong reacquisition branch above.
            self._operational_height_reference_m = float(self.env.tool_xyz()[2])
        return None

    def _pending_command_effect_hold_action(
        self,
        *,
        feedback_gap: bool = False,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Submit an observable hold while a forward command effect is pending."""

        if not isinstance(self.env, RealisticEdgeArmEnvV7):  # pragma: no cover
            raise RuntimeError("command effect hold requires a V7 environment")
        kind = self._pending_effect_kind
        phase = (
            "adaptive_edge_push_inflight_hold"
            if kind == "adaptive_edge_push"
            else "contact_acquisition_inflight_hold"
        )
        action, prefilter_reason = self._prefilter_action(np.zeros(_JOINTS, dtype=np.float64))
        action = np.asarray(action, dtype=np.float32)
        if kind == "adaptive_edge_push":
            self._adaptive_edge_push_inflight_hold_total_steps += 1
        elif kind == "contact_acquisition":
            self._contact_acquisition_inflight_hold_total_steps += 1
        self._last_step_count = int(self.env.step_count)
        requested_target = (
            self.env.data.qpos[:_JOINTS]
            + self.env._encoder_position_noise
            + action * self.env.config.max_joint_delta
        )
        metadata = self._common_metadata()
        metadata.update(
            {
                "version": PHYSICAL_EXPERT_V11_VERSION,
                "phase": phase,
                "planned_phase": phase,
                "phase_semantics": "teacher_plan_before_env_step",
                "teacher_confidence": 0.0 if feedback_gap else 1.0,
                "precontact_complete": not feedback_gap,
                "joint_target": requested_target.astype(np.float32),
                "raw_joint_target": requested_target.astype(np.float32),
                "joint_target_semantics": "safety_filtered_command_effect_wait_hold",
                "teacher_requested_one_step_joint_target": (requested_target.astype(np.float32)),
                "workspace_prefilter_applied": bool(prefilter_reason),
                "workspace_prefilter_reason": prefilter_reason,
                "normalized_action_linf": float(np.max(np.abs(action))),
                "command_feedback_gap": feedback_gap,
            }
        )
        return action, metadata

    def _joint_path_start_audit(
        self,
        joint_position: np.ndarray | None = None,
    ) -> dict[str, Any]:
        """Run the original path checker's endpoint contract without planning."""

        if not isinstance(self.env, RealisticEdgeArmEnvV7):  # pragma: no cover
            raise RuntimeError("contact-clearance bridge requires a V7 environment")
        position = (
            self.env.data.qpos[:_JOINTS].copy()
            if joint_position is None
            else np.asarray(joint_position, dtype=np.float64)
        )
        if position.shape != (_JOINTS,) or not np.all(np.isfinite(position)):
            raise ValueError("endpoint audit requires six finite joints")
        planner = JointPathPlannerV1(self.env)
        checker = _CollisionChecker(self.env, planner.config)
        audit = asdict(checker.state(position[:5]))
        minimum_non_tool_desk = float(audit["minimum_non_tool_robot_desk_contact_distance_m"])
        if not np.isfinite(minimum_non_tool_desk):
            audit["minimum_non_tool_robot_desk_contact_distance_m"] = None
        return audit

    def _contact_clearance_bridge_fail_closed(
        self,
        reason: str,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Latch a non-planning hold after a causal bridge failure."""

        self._contact_clearance_bridge_active = True
        self._contact_clearance_bridge_failed = True
        self._contact_clearance_bridge_failure_reason = str(reason)
        self._precontact_mode = "contact_clearance_bridge_fail_closed"
        self._precontact_complete = False
        action, prefilter_reason = self._prefilter_action(np.zeros(_JOINTS, dtype=np.float64))
        action = np.asarray(action, dtype=np.float32)
        self._last_step_count = int(self.env.step_count)
        metadata = self._common_metadata()
        metadata.update(
            {
                "version": PHYSICAL_EXPERT_V11_VERSION,
                "phase": "contact_clearance_bridge_fail_closed_hold",
                "planned_phase": "contact_clearance_bridge_fail_closed_hold",
                "phase_semantics": "teacher_plan_before_env_step",
                "teacher_confidence": 0.0,
                "precontact_complete": False,
                "workspace_prefilter_applied": bool(prefilter_reason),
                "workspace_prefilter_reason": prefilter_reason,
                "normalized_action_linf": float(np.max(np.abs(action))),
            }
        )
        return action, metadata

    def _contact_clearance_bridge_target_action(
        self,
        *,
        phase: str,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Submit a new command that points to the one latched safe target."""

        command_reference = self.env._command_reference_reported_position() - self.env._zero_offset
        raw_action = (
            self._contact_clearance_bridge_target - command_reference
        ) / self.env.config.max_joint_delta
        if not np.all(np.isfinite(raw_action)) or np.max(np.abs(raw_action)) > 1.0 + 1.0e-9:
            return self._contact_clearance_bridge_fail_closed("latched_target_outside_one_command_reach")
        # Do not use the inherited V7 expert prefilter here: it reconstructs
        # targets from the unquantized encoder-noise proxy, whereas the V6/V7
        # transport uses the encoder-quantized reported position.  The bridge
        # must keep every appended command aimed at one absolute target, so it
        # forms the transport action from that exact observable reference and
        # checks the endpoint with the environment filter directly.
        action = np.clip(raw_action, -1.0, 1.0).astype(np.float32)
        # V6/V7 quantize the reported encoder state used by the command
        # transport.  Reconstruct the physical target from that exact command
        # reference rather than from the unquantized synthetic noise vector.
        submitted_target = command_reference + action.astype(np.float64) * self.env.config.max_joint_delta
        submitted_target, exact_filter_reason = self.env._safety_filter(submitted_target)
        if not np.allclose(
            submitted_target,
            self._contact_clearance_bridge_target,
            rtol=0.0,
            # The public expert/collector action contract is float32.  One
            # float32 action ULP maps to less than 1e-8 rad at the frozen
            # 0.055-rad command scale, so this accepts representation noise
            # while still rejecting any physically distinct endpoint.
            atol=_CONTACT_CLEARANCE_BRIDGE_TARGET_TOLERANCE_RAD,
        ):
            return self._contact_clearance_bridge_fail_closed("latched_target_changed_by_submission_filter")
        prefilter_reason = exact_filter_reason
        submitted_id = int(self.env.next_command_id)
        self._contact_clearance_bridge_last_submitted_command_id = submitted_id
        self._last_step_count = int(self.env.step_count)
        metadata = self._common_metadata()
        metadata.update(
            {
                "version": PHYSICAL_EXPERT_V11_VERSION,
                "phase": phase,
                "planned_phase": phase,
                "phase_semantics": "teacher_plan_before_env_step",
                "teacher_confidence": 0.0,
                "precontact_complete": False,
                "joint_target": self._contact_clearance_bridge_target.astype(np.float32),
                "raw_joint_target": self._contact_clearance_bridge_target.astype(np.float32),
                "joint_target_semantics": ("causal_contact_clearance_bridge_latched_absolute_target"),
                "teacher_requested_one_step_joint_target": (
                    self._contact_clearance_bridge_target.astype(np.float32)
                ),
                "workspace_prefilter_applied": bool(prefilter_reason),
                "workspace_prefilter_reason": prefilter_reason,
                "normalized_action_linf": float(np.max(np.abs(action))),
                "contact_clearance_bridge_submitted_command_id": submitted_id,
                "contact_clearance_bridge_queue_tail_same_target_by_construction": True,
                "contact_clearance_bridge_queue_inspected_or_mutated": False,
            }
        )
        return action, metadata

    def _start_contact_clearance_bridge(self) -> tuple[np.ndarray, dict[str, Any]]:
        """Append a causal retract without touching any older queued command."""

        if not isinstance(self.env, RealisticEdgeArmEnvV7):  # pragma: no cover
            raise RuntimeError("contact-clearance bridge requires a V7 environment")
        self._contact_clearance_bridge_active = True
        self._contact_clearance_bridge_failed = False
        self._contact_clearance_bridge_failure_reason = ""
        self._contact_clearance_bridge_audited_effects = 0
        self._contact_clearance_bridge_last_start_audit = {}
        direction = self.env._unit(self.env.target_xy - self.env.block_xy())
        tool = self.env.tool_xyz()
        target_xyz = np.array(
            [
                tool[0] - direction[0] * self.v11_config.invalid_contact_escape_xy_m,
                tool[1] - direction[1] * self.v11_config.invalid_contact_escape_xy_m,
                min(
                    tool[2] + self.v11_config.invalid_contact_escape_z_m,
                    self.env.config.workspace_z[1] - 0.005,
                ),
            ],
            dtype=np.float64,
        )
        if self._side_contact_ik_planner is None:
            self._side_contact_ik_planner = self._make_side_contact_ik_planner()
        tracking = self._side_contact_ik_planner.track_target(
            target_xyz,
            direction,
            initial=self.env.data.qpos[:_JOINTS].copy(),
        )
        self._contact_clearance_bridge_tracking = tracking
        if not bool(tracking["feasible"]):
            return self._contact_clearance_bridge_fail_closed("retract_tracking_ik_infeasible")
        command_reference = self.env._command_reference_reported_position() - self.env._zero_offset
        desired = np.asarray(tracking["best"]["joint_position_rad"], dtype=np.float64)
        initial_action = np.clip(
            (desired - command_reference) / self.env.config.max_joint_delta,
            -1.0,
            1.0,
        )
        target = command_reference + initial_action * self.env.config.max_joint_delta
        target, target_filter_reason = self.env._safety_filter(target)
        target_audit = self._joint_path_start_audit(target)
        if target_filter_reason or not bool(target_audit["valid"]):
            self._contact_clearance_bridge_last_start_audit = target_audit
            return self._contact_clearance_bridge_fail_closed("retract_target_not_original_checker_safe")
        self._contact_clearance_bridge_target = np.asarray(target, dtype=np.float64).copy()
        first_id = int(self.env.next_command_id)
        self._contact_clearance_bridge_epoch = int(self.env.command_epoch)
        self._contact_clearance_bridge_first_command_id = first_id
        self._contact_clearance_bridge_next_audit_command_id = first_id
        self._contact_clearance_bridge_last_submitted_command_id = first_id - 1
        return self._contact_clearance_bridge_target_action(phase="contact_clearance_bridge_retract")

    def _contact_clearance_bridge_action(self) -> tuple[np.ndarray, dict[str, Any]]:
        """Wait on exact same-target effects, then invoke the unchanged planner."""

        if self._contact_clearance_bridge_failed:
            return self._contact_clearance_bridge_fail_closed(self._contact_clearance_bridge_failure_reason)
        feedback = self.env.last_command_feedback_v1
        if not bool(feedback.get("available", False)):
            return self._contact_clearance_bridge_target_action(phase="contact_clearance_bridge_target_flood")
        if int(feedback.get("epoch", -1)) != self._contact_clearance_bridge_epoch:
            return self._contact_clearance_bridge_fail_closed("command_feedback_epoch_gap")
        applied_id = int(feedback.get("applied_id", -1))
        expected_id = self._contact_clearance_bridge_next_audit_command_id
        if applied_id < expected_id:
            return self._contact_clearance_bridge_target_action(phase="contact_clearance_bridge_target_flood")
        if applied_id > expected_id:
            return self._contact_clearance_bridge_fail_closed("command_feedback_applied_id_gap")

        self._contact_clearance_bridge_audited_effects += 1
        start_audit = self._joint_path_start_audit()
        self._contact_clearance_bridge_last_start_audit = start_audit
        if bool(start_audit["valid"]):
            self._contact_clearance_bridge_active = False
            self._precontact_mode = "contact_clearance_bridge_reacquisition"
            self._precontact_complete = False
            self._plan_precontact_waypoints()
            if not self._precontact_plan_feasible:
                return self._precontact_plan_infeasible_action()
            action, metadata = self._precontact_action(self._precontact_diagnostics())
            metadata["contact_clearance_bridge_completed"] = True
            metadata["contact_clearance_bridge_completed_applied_command_id"] = applied_id
            return action, metadata
        if self._contact_clearance_bridge_audited_effects >= self.v11_config.invalid_contact_escape_steps:
            return self._contact_clearance_bridge_fail_closed("collision_free_start_audit_budget_exhausted")
        self._contact_clearance_bridge_next_audit_command_id = expected_id + 1
        if self._contact_clearance_bridge_next_audit_command_id > max(
            self._contact_clearance_bridge_last_submitted_command_id,
            int(self.env.next_command_id),
        ):
            return self._contact_clearance_bridge_fail_closed("same_target_flood_missing_expected_command")
        return self._contact_clearance_bridge_target_action(phase="contact_clearance_bridge_target_flood")

    def _requires_global_reacquisition(self) -> bool:
        if not isinstance(self.env, RealisticEdgeArmEnvV7):  # pragma: no cover
            raise RuntimeError("global reacquisition requires a V7 environment")
        coverage = float(self.env.block_target_coverage())
        if coverage >= self.env.realism_config.strict_coverage_threshold:
            return False
        block = self.env.block_xy()
        direction = self.env._unit(self.env.target_xy - block)
        along = float(np.dot(block - self.env.tool_xyz()[:2], direction))
        return along < self.v11_config.push_gate_minimum_behind_m

    def _invalid_push_contact_present(self) -> bool:
        if not isinstance(self.env, RealisticEdgeArmEnvV7):  # pragma: no cover
            raise RuntimeError("invalid-contact check requires a V7 environment")
        direction = self.env._unit(self.env.target_xy - self.env.block_xy())
        current = self.env.current_push_side_contact_metrics(direction)
        current_raw = bool(self.env._tool_block_contacts() > 0)
        current_invalid = bool(current_raw and not current["valid_side_contact_any"])
        trace = self.env._physics_substep_contact_v1
        trace_invalid = bool(
            trace is not None
            and bool(trace["contact_any"])
            and not bool(trace["valid_push_side_contact_any"])
            and not np.any(np.asarray(trace["geometric_push_side_contact_count"]) > 0)
        )
        # A completed transition trace is the authoritative effect record.  A
        # current MuJoCo contact can remain listed at the decision boundary
        # after its force has fallen below the side-contact threshold; treating
        # that stale raw pair as a new invalid event makes escape self-latching.
        return trace_invalid if trace is not None else current_invalid

    def _invalid_contact_escape_action(self) -> tuple[np.ndarray, dict[str, Any]]:
        if not isinstance(self.env, RealisticEdgeArmEnvV7):  # pragma: no cover
            raise RuntimeError("invalid-contact escape requires a V7 environment")
        direction = self.env._unit(self.env.target_xy - self.env.block_xy())
        tool = self.env.tool_xyz()
        target_xyz = np.array(
            [
                tool[0] - direction[0] * self.v11_config.invalid_contact_escape_xy_m,
                tool[1] - direction[1] * self.v11_config.invalid_contact_escape_xy_m,
                min(
                    tool[2] + self.v11_config.invalid_contact_escape_z_m,
                    self.env.config.workspace_z[1] - 0.005,
                ),
            ],
            dtype=np.float64,
        )
        if self._side_contact_ik_planner is None:
            self._side_contact_ik_planner = self._make_side_contact_ik_planner()
        tracking = self._side_contact_ik_planner.track_target(
            target_xyz,
            direction,
            initial=self.env.data.qpos[:_JOINTS].copy(),
        )
        tracking_feasible = bool(tracking["feasible"])
        joint_target = (
            np.asarray(tracking["best"]["joint_position_rad"], dtype=np.float64)
            if tracking_feasible
            else self.env.data.qpos[:_JOINTS].copy()
        )
        action = np.clip(
            (joint_target - self.env.data.qpos[:_JOINTS]) / self.env.config.max_joint_delta,
            -1.0,
            1.0,
        )
        action, prefilter_reason = self._prefilter_action(action)
        action = np.asarray(action, dtype=np.float32)
        # An escape command must supersede delayed forward commands that were
        # submitted before the invalid contact was observed.
        queued_target = (
            self.env.data.qpos[:_JOINTS]
            + self.env._encoder_position_noise
            + action * self.env.config.max_joint_delta
        )
        for queued in self.env._command_queue:
            queued.rewrite_target(
                queued_target,
                reason="v11_invalid_contact_escape",
            )
        self._invalid_contact_escape_remaining = max(
            self._invalid_contact_escape_remaining - 1,
            0,
        )
        self._invalid_contact_escape_total_steps += 1
        self._precontact_steps += 1
        self._last_step_count = int(self.env.step_count)
        side_metrics = self.env.current_push_side_contact_metrics(direction)
        metadata = self._common_metadata()
        metadata.update(
            {
                "version": PHYSICAL_EXPERT_V11_VERSION,
                "phase": (
                    "invalid_contact_escape"
                    if tracking_feasible
                    else "invalid_contact_escape_tracking_infeasible_hold"
                ),
                "planned_phase": (
                    "invalid_contact_escape"
                    if tracking_feasible
                    else "invalid_contact_escape_tracking_infeasible_hold"
                ),
                "phase_semantics": "teacher_plan_before_env_step",
                "teacher_confidence": 0.0,
                "precontact_complete": False,
                "precontact_plan_feasible": self._precontact_plan_feasible,
                "invalid_contact_escape_target_xyz_m": target_xyz.tolist(),
                "invalid_contact_escape_remaining_steps": (self._invalid_contact_escape_remaining),
                "invalid_contact_escape_total_steps": self._invalid_contact_escape_total_steps,
                "invalid_contact_escape_tracking_ik": tracking,
                "invalid_contact_escape_tracking_feasible": tracking_feasible,
                "invalid_contact_escape_queue_overridden": True,
                "contact_geometry_actual_contact": bool(self.env._tool_block_contacts() > 0),
                "contact_geometry_trace_raw_contact": bool(
                    self.env._physics_substep_contact_v1 is not None
                    and self.env._physics_substep_contact_v1["contact_any"]
                ),
                "contact_geometry_valid_push_side_contact": False,
                "contact_geometry_invalid_raw_contact": True,
                "contact_geometry_side_contact_metrics": side_metrics,
                "contact_geometry_verified_contact_gate": False,
                "contact_geometry_advance_gate": False,
                "workspace_prefilter_applied": bool(prefilter_reason),
                "workspace_prefilter_reason": prefilter_reason,
                "normalized_action_linf": float(np.max(np.abs(action))),
            }
        )
        return action, metadata

    def _ingest_precontact_execution_telemetry(self) -> None:
        if not isinstance(self.env, RealisticEdgeArmEnvV7):  # pragma: no cover
            raise RuntimeError("pre-contact telemetry requires a V7 environment")
        if self.env.step_count <= self._precontact_last_trace_step:
            return
        trace = self.env._physics_substep_contact_v1
        if trace is not None:
            invalid_contact = bool(
                trace["contact_any"]
                and not trace["valid_push_side_contact_any"]
                and not np.any(np.asarray(trace["geometric_push_side_contact_count"]) > 0)
            )
            self._precontact_unexpected_contact_steps += int(invalid_contact)
            self._precontact_forbidden_desk_steps += int(
                bool(trace["forbidden_tool_desk_penetration_any"])
                or bool(trace["forbidden_non_tool_robot_desk_penetration_any"])
            )
        self._precontact_last_trace_step = int(self.env.step_count)

    def _precontact_diagnostics(self) -> dict[str, float | bool]:
        if not isinstance(self.env, RealisticEdgeArmEnvV7):  # pragma: no cover
            raise RuntimeError("pre-contact diagnostics require a V7 environment")
        block = self.env.block_xy()
        tool = self.env.tool_xyz()
        direction = self.env._unit(self.env.target_xy - block)
        offset = block - tool[:2]
        along = float(np.dot(offset, direction))
        lateral = float(abs(direction[0] * offset[1] - direction[1] * offset[0]))
        vertical_error = float(abs(tool[2] - self._precontact_target_xyz[2]))
        target_xyz_error = float(np.linalg.norm(tool - self._precontact_target_xyz))
        joint_error = float(np.max(np.abs(self._precontact_joint_target[:5] - self.env.data.qpos[:5])))
        contact_count = int(self.env._tool_block_contacts())
        pusher_desk_clearance = self._current_tool_desk_clearance_m()
        vertical_tolerance = self.v11_config.precontact_vertical_error_tolerance_m
        lateral_tolerance = self.v11_config.precontact_lateral_error_tolerance_m
        final_stage = bool(self._precontact_stage_index == len(self._precontact_joint_waypoints) - 1)
        side_metrics = self.env.current_push_side_contact_metrics(direction)
        face_orientation_verified = bool(
            float(side_metrics["tool_face_horizontal_norm"])
            >= self.v11_config.precontact_face_horizontal_minimum
            and float(side_metrics["tool_face_push_alignment"])
            >= self.v11_config.precontact_face_alignment_minimum
        )
        pusher_block_gap, pusher_block_maximum_gap = (
            self._current_tool_block_gap_interval_m()
        )
        goal_gap_verified = bool(
            self.v11_config.precontact_goal_gap_minimum_m
            <= pusher_block_gap
            and pusher_block_maximum_gap
            <= self.v11_config.precontact_goal_gap_maximum_m
        )
        alignment_verified = bool(not final_stage or (face_orientation_verified and goal_gap_verified))
        arrival = bool(
            joint_error <= self.v11_config.precontact_joint_error_tolerance_rad
            and target_xyz_error <= self.v11_config.precontact_cartesian_error_tolerance_m
            and vertical_error <= vertical_tolerance
            and alignment_verified
            and contact_count == 0
            and pusher_desk_clearance >= -self.env.contact_feasible_config.reset_penetration_tolerance_m
        )
        return {
            "arrival_verified": arrival,
            "joint_error_linf_rad": joint_error,
            "vertical_error_m": vertical_error,
            "target_xyz_error_m": target_xyz_error,
            "along_m": along,
            "lateral_m": lateral,
            "contact_count": contact_count,
            "pusher_desk_clearance_m": pusher_desk_clearance,
            "vertical_tolerance_m": vertical_tolerance,
            "lateral_tolerance_m": lateral_tolerance,
            "alignment_verified": alignment_verified,
            "final_stage": final_stage,
            "face_orientation_verified": face_orientation_verified,
            "goal_gap_verified": goal_gap_verified,
            "pusher_block_gap_m": pusher_block_gap,
            "pusher_block_maximum_gap_m": pusher_block_maximum_gap,
            "tool_face_horizontal_norm": float(side_metrics["tool_face_horizontal_norm"]),
            "tool_face_push_alignment": float(side_metrics["tool_face_push_alignment"]),
        }

    def _precontact_action(
        self,
        diagnostics: Mapping[str, float | bool],
    ) -> tuple[np.ndarray, dict[str, Any]]:
        if not isinstance(self.env, RealisticEdgeArmEnvV7):  # pragma: no cover
            raise RuntimeError("pre-contact action requires a V7 environment")
        joint_error = self._precontact_joint_target - self.env.data.qpos[:_JOINTS]
        command_delta = joint_error.copy()
        command_linf = float(np.max(np.abs(command_delta)))
        if command_linf > self.v11_config.precontact_joint_command_step_rad:
            command_delta *= self.v11_config.precontact_joint_command_step_rad / command_linf
        action = command_delta / self.env.config.max_joint_delta
        action, prefilter_reason = self._prefilter_action(action)
        action = np.asarray(action, dtype=np.float32)
        teacher_requested_target = (
            self.env.data.qpos[:_JOINTS]
            + self.env._encoder_position_noise
            + action * self.env.config.max_joint_delta
        )
        self._precontact_steps += 1
        self._last_step_count = int(self.env.step_count)
        metadata = self._common_metadata()
        metadata.update(
            {
                "version": PHYSICAL_EXPERT_V11_VERSION,
                "delegate_version": PHYSICAL_EXPERT_V9_VERSION,
                "phase": "pre_contact_alignment",
                "planned_phase": "pre_contact_alignment",
                "phase_semantics": "teacher_plan_before_env_step",
                "teacher_confidence": float(np.exp(-float(diagnostics["joint_error_linf_rad"]))),
                "joint_target": self._precontact_joint_target.astype(np.float32),
                "raw_joint_target": self._precontact_joint_target.astype(np.float32),
                "joint_target_semantics": "global_precontact_waypoint_not_applied_transition_target",
                "precontact_global_joint_waypoint": self._precontact_joint_target.astype(np.float32),
                "teacher_requested_one_step_joint_target": teacher_requested_target.astype(np.float32),
                "precontact_target_xyz_m": self._precontact_target_xyz.tolist(),
                "precontact_stage_index": self._precontact_stage_index,
                "precontact_stage_count": len(self._precontact_joint_waypoints),
                "precontact_stage_name": self._precontact_stage_names[self._precontact_stage_index],
                "precontact_complete": False,
                "precontact_steps": self._precontact_steps,
                "precontact_replan_count": self._precontact_replans,
                "precontact_arrival_verified": bool(diagnostics["arrival_verified"]),
                "precontact_joint_error_linf_rad": float(diagnostics["joint_error_linf_rad"]),
                "precontact_vertical_error_m": float(diagnostics["vertical_error_m"]),
                "precontact_target_xyz_error_m": float(diagnostics["target_xyz_error_m"]),
                "precontact_cartesian_error_tolerance_m": (
                    self.v11_config.precontact_cartesian_error_tolerance_m
                ),
                "precontact_vertical_error_tolerance_m": float(diagnostics["vertical_tolerance_m"]),
                "precontact_lateral_error_tolerance_m": float(diagnostics["lateral_tolerance_m"]),
                "precontact_alignment_verified": bool(diagnostics["alignment_verified"]),
                "precontact_final_stage": bool(diagnostics["final_stage"]),
                "precontact_face_orientation_verified": bool(diagnostics["face_orientation_verified"]),
                "precontact_goal_gap_verified": bool(diagnostics["goal_gap_verified"]),
                "precontact_pusher_block_gap_m": float(diagnostics["pusher_block_gap_m"]),
                "precontact_pusher_block_maximum_gap_m": float(
                    diagnostics["pusher_block_maximum_gap_m"]
                ),
                "precontact_tool_face_horizontal_norm": float(diagnostics["tool_face_horizontal_norm"]),
                "precontact_tool_face_push_alignment": float(diagnostics["tool_face_push_alignment"]),
                "tool_block_along_m": float(diagnostics["along_m"]),
                "tool_block_lateral_m": float(diagnostics["lateral_m"]),
                "contact_count": int(diagnostics["contact_count"]),
                "pusher_desk_signed_distance_m": float(diagnostics["pusher_desk_clearance_m"]),
                "strict_target_coverage": float(self.env.block_target_coverage()),
                "workspace_prefilter_applied": bool(prefilter_reason),
                "workspace_prefilter_reason": prefilter_reason,
                "normalized_action_linf": float(np.max(np.abs(action))),
                "precontact_joint_command_step_rad": (self.v11_config.precontact_joint_command_step_rad),
            }
        )
        return action, metadata

    def _full_push_contact_quality(
        self,
        trace: Mapping[str, Any] | None,
        current: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Decide whether a valid side contact has margin for a full push.

        The completed transition trace is authoritative because wrist/control
        decisions often occur after the instantaneous MuJoCo contact has
        released.  A current contact is used only when no valid trace contact
        exists.  Every valid side manifold must retain a normal/rear margin;
        edge-only manifolds use the stricter frozen edge floors.
        """

        config = self.v11_config
        source = "none"
        edge_only = False
        alignment = 0.0
        rear_support = 0.0
        central_vertical_sideband_reserve_m = 0.0
        vertical_contact_local_z_m = 0.0
        evidence_valid = False
        side_thresholds = PushSideContactThresholds()
        block_half_height_m = float(self.env.model.geom_size[self.env._ids["block_geom"]][2])
        central_sideband_limit_m = block_half_height_m * (
            1.0 - side_thresholds.block_top_bottom_exclusion_fraction
        )
        observed_inflight_vertical_displacement_m = 0.0
        minimum_central_vertical_sideband_reserve_m = side_thresholds.maximum_edge_vertical_excess_m
        if trace is not None and bool(trace.get("valid_push_side_contact_any", False)):
            source = "previous_effect_trace"
            try:
                valid = np.asarray(trace["valid_push_side_contact_count"], dtype=np.int64).reshape(-1) > 0
                sideband = np.asarray(trace["representative_side_band_valid"], dtype=np.int64).reshape(-1) > 0
                edge = (
                    np.asarray(trace["representative_edge_side_contact_valid"], dtype=np.int64).reshape(-1)
                    > 0
                )
                alignments = np.asarray(
                    trace["representative_contact_normal_push_alignment"],
                    dtype=np.float64,
                ).reshape(-1)
                rear_supports = np.asarray(
                    trace["representative_rear_support_ratio"], dtype=np.float64
                ).reshape(-1)
                tool_displacement = np.asarray(
                    trace.get("tool_xyz_displacement_m", np.zeros(3)),
                    dtype=np.float64,
                )
                if tool_displacement.shape != (3,):
                    raise ValueError("trace tool displacement must have shape (3,)")
                observed_inflight_vertical_displacement_m = abs(float(tool_displacement[2]))
                minimum_central_vertical_sideband_reserve_m = (
                    observed_inflight_vertical_displacement_m + side_thresholds.maximum_edge_vertical_excess_m
                )
                block_local_points = np.asarray(
                    trace["representative_block_local_contact_xyz_m"],
                    dtype=np.float64,
                ).reshape(-1, 3)
                lengths = {
                    len(valid),
                    len(sideband),
                    len(edge),
                    len(alignments),
                    len(rear_supports),
                    len(block_local_points),
                }
                if len(lengths) != 1 or not np.any(valid):
                    raise ValueError("trace side-contact arrays are inconsistent")
                edge_mask = valid & edge & ~sideband
                evidence_valid = True
                edge_only = bool(np.any(edge_mask))
                # A central side-band manifold is not automatically a robust
                # full-push manifold.  Its normal can rotate toward the gate
                # threshold and its contact point can run out of rear support
                # while it is still semantically valid.  Use every valid
                # substep, not only edge-exception substeps, so the delayed
                # controller cannot queue another full push on vanishing
                # contact margin.
                alignment = float(np.min(alignments[valid]))
                rear_support = float(np.min(rear_supports[valid]))
                valid_local_z = block_local_points[valid, 2]
                worst_vertical_index = int(np.argmax(np.abs(valid_local_z)))
                vertical_contact_local_z_m = float(valid_local_z[worst_vertical_index])
                central_vertical_sideband_reserve_m = float(
                    central_sideband_limit_m - abs(vertical_contact_local_z_m)
                )
            except (KeyError, TypeError, ValueError):
                evidence_valid = False
        elif bool(current.get("valid_side_contact_any", False)):
            source = "current_contact"
            evidence_valid = True
            sideband = bool(current.get("representative_side_band_valid", False))
            edge = bool(current.get("representative_edge_side_contact_valid", False))
            edge_only = bool(edge and not sideband)
            if edge_only:
                alignment = float(current.get("representative_contact_normal_push_alignment", 0.0))
                rear_support = float(current.get("representative_rear_support_ratio", 0.0))
            else:
                alignment = float(current.get("representative_contact_normal_push_alignment", 0.0))
                rear_support = float(current.get("representative_rear_support_ratio", 0.0))
            block_local_point = np.asarray(
                current.get(
                    "representative_block_local_contact_xyz_m",
                    np.full(3, np.nan),
                ),
                dtype=np.float64,
            )
            if block_local_point.shape == (3,):
                vertical_contact_local_z_m = float(block_local_point[2])
                central_vertical_sideband_reserve_m = float(
                    central_sideband_limit_m - abs(vertical_contact_local_z_m)
                )

        finite = bool(
            np.isfinite(alignment)
            and np.isfinite(rear_support)
            and np.isfinite(central_vertical_sideband_reserve_m)
        )
        minimum_alignment = float(
            config.edge_full_push_minimum_normal_alignment
            if edge_only
            else config.central_full_push_minimum_normal_alignment
        )
        minimum_rear_support = float(
            config.edge_full_push_minimum_rear_support_ratio
            if edge_only
            else config.central_full_push_minimum_rear_support_ratio
        )
        full_push_allowed = bool(
            evidence_valid
            and finite
            and alignment >= minimum_alignment
            and rear_support >= minimum_rear_support
            and (
                edge_only
                or central_vertical_sideband_reserve_m
                >= minimum_central_vertical_sideband_reserve_m - 1.0e-12
            )
        )
        vertical_centering_deficit_m = (
            max(
                minimum_central_vertical_sideband_reserve_m - central_vertical_sideband_reserve_m,
                0.0,
            )
            if evidence_valid and finite
            else 0.0
        )
        vertical_centering_speed_limit_m = float(config.fine_side_push_step_m)
        vertical_centering_magnitude_m = min(
            vertical_centering_deficit_m,
            vertical_centering_speed_limit_m,
            abs(vertical_contact_local_z_m),
        )
        vertical_centering_correction_m = (
            -float(np.copysign(vertical_centering_magnitude_m, vertical_contact_local_z_m))
            if vertical_centering_magnitude_m > 1.0e-12
            else 0.0
        )
        vertical_centering_required = bool(vertical_centering_magnitude_m > 1.0e-12)
        return {
            "source": source,
            "evidence_valid": evidence_valid,
            "edge_only": edge_only,
            "normal_push_alignment": alignment,
            "rear_support_ratio": rear_support,
            "minimum_normal_push_alignment": minimum_alignment,
            "minimum_rear_support_ratio": minimum_rear_support,
            "central_side_contact_margin_required": True,
            "central_vertical_sideband_reserve_m": (central_vertical_sideband_reserve_m),
            "minimum_central_vertical_sideband_reserve_m": (minimum_central_vertical_sideband_reserve_m),
            "observed_inflight_vertical_displacement_m": (observed_inflight_vertical_displacement_m),
            "vertical_contact_local_z_m": vertical_contact_local_z_m,
            "vertical_centering_deficit_m": vertical_centering_deficit_m,
            "vertical_centering_speed_limit_m": vertical_centering_speed_limit_m,
            "vertical_centering_correction_m": vertical_centering_correction_m,
            "vertical_centering_required": vertical_centering_required,
            "central_vertical_sideband_reserve_source": (
                "completed_effect_abs_tool_z_displacement_plus_edge_excess"
            ),
            "full_push_allowed": full_push_allowed,
        }

    def _geometry_operational_action(
        self,
        direction: np.ndarray,
        *,
        contact_opens_gate: bool,
        environment_obstacle_detour: bool,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """V11 contact command with signed front-side and height gates."""

        if not isinstance(self.env, RealisticEdgeArmEnvV7):
            raise RuntimeError("V11 operational action requires a V7 environment")
        config = self.v11_config
        direction = np.asarray(direction, dtype=np.float64)
        if direction.shape != (2,) or not np.all(np.isfinite(direction)):
            raise ValueError("direction must be a finite two-vector")
        block = self.env.block_xy()
        tool = self.env.tool_xyz()
        operational_height = self._operational_tool_height_m()
        contact_xy = block - direction * config.contact_standoff_m
        offset = block - tool[:2]
        along = float(np.dot(offset, direction))
        lateral = float(abs(direction[0] * offset[1] - direction[1] * offset[0]))
        vertical_error = float(abs(tool[2] - operational_height))
        current_contact = self.env._tool_block_contacts() > 0
        side_metrics = self.env.current_push_side_contact_metrics(direction)
        trace = self.env._physics_substep_contact_v1
        trace_raw_contact = bool(trace is not None and bool(trace["contact_any"]))
        contact = bool(current_contact or trace_raw_contact)
        trace_side_valid = bool(trace is not None and bool(trace["valid_push_side_contact_any"]))
        trace_geometric_side = bool(
            trace is not None and np.any(np.asarray(trace["geometric_push_side_contact_count"]) > 0)
        )
        current_side_valid = bool(side_metrics["valid_side_contact_any"])
        current_geometric_side = bool(side_metrics["geometric_side_contact_count"] > 0)
        # The trace describes the entire action-to-effect transition and is
        # allowed to contain a valid transient contact.  Requiring its final
        # substep and the next decision-boundary contact to both remain valid
        # incorrectly turns ordinary stick-slip into contact loss.  Current
        # state is used only before a trace exists or as additional positive
        # evidence; a completed trace is authoritative for invalid contact.
        semantic_side_contact = bool(trace_side_valid or current_side_valid)
        invalid_raw_contact = bool(
            (trace_raw_contact and not trace_side_valid)
            and not trace_geometric_side
            or (trace is None and current_contact and not current_side_valid and not current_geometric_side)
        )
        if self._contact_semantics_last_step != self.env.step_count:
            self._valid_push_side_contact_steps += int(semantic_side_contact)
            self._invalid_raw_contact_steps += int(invalid_raw_contact)
            self._contact_semantics_last_step = int(self.env.step_count)
        signed_alignment_gate = bool(
            config.push_gate_minimum_behind_m <= along < config.push_gate_along_m
            and lateral < config.push_gate_lateral_m
            and vertical_error < config.push_gate_vertical_error_m
        )
        verified_contact_gate = bool(semantic_side_contact and signed_alignment_gate)
        benign_side_acquisition = bool(
            (trace_geometric_side or current_geometric_side) and not semantic_side_contact
        )
        contact_acquisition_gate = bool(signed_alignment_gate and (not contact or benign_side_acquisition))
        advance_gate = bool(verified_contact_gate or contact_acquisition_gate)
        phase = "approach"
        if invalid_raw_contact:
            phase = "invalid_contact_reacquisition"
        elif verified_contact_gate:
            phase = "push"
        elif contact_acquisition_gate:
            phase = "contact_acquisition"
        pusher_block_gap, pusher_block_maximum_gap = (
            self._current_tool_block_gap_interval_m()
        )
        full_push_quality = self._full_push_contact_quality(trace, side_metrics)
        lateral_full_push_reserve = max(
            config.push_gate_lateral_m - lateral,
            0.0,
        )
        # The ordinary semantic gate is open until ``push_gate_lateral_m``.
        # Preserve one already-configured precision step of reserve before
        # allowing another delayed full push.  The extra in-flight command can
        # then consume that reserve while the newly issued fine step enters the
        # queue, without slowing well-centered side contacts.
        minimum_lateral_full_push_reserve = config.precision_side_push_step_m
        lateral_full_push_allowed = bool(
            lateral_full_push_reserve >= minimum_lateral_full_push_reserve - 1.0e-12
        )
        vertical_recentering_required = bool(
            verified_contact_gate and full_push_quality["vertical_centering_required"]
        )
        robust_full_push_allowed = bool(
            full_push_quality["full_push_allowed"]
            and lateral_full_push_allowed
            and not vertical_recentering_required
        )
        advance_step = 0.0
        adaptive_edge_step_limited = False
        if verified_contact_gate:
            coverage = float(self.env.block_target_coverage())
            if vertical_recentering_required:
                # Recenter only in z.  The causal contact point came from the
                # previous completed effect, so advancing x/y while correcting
                # it would mix two interventions and could drive farther over
                # the same top/bottom edge before the delayed effect arrives.
                advance_step = 0.0
                adaptive_edge_step_limited = True
                phase = "contact_manifold_vertical_recenter"
            elif coverage >= config.fine_push_coverage_start:
                advance_step = config.fine_side_push_step_m
            elif coverage >= config.precision_push_coverage_start:
                advance_step = config.precision_side_push_step_m
            elif not robust_full_push_allowed:
                advance_step = config.fine_side_push_step_m
                adaptive_edge_step_limited = True
            else:
                advance_step = config.side_push_step_m
        elif contact_acquisition_gate:
            advance_step = min(
                config.side_contact_acquisition_step_m,
                max(pusher_block_gap + 0.0005, 0.0015),
            )
        desired_xy = tool[:2] + direction * advance_step if advance_gate else contact_xy
        if (
            not vertical_recentering_required
            and environment_obstacle_detour
            and self.env.obstacle_enabled
            and self.env._path_intersects_obstacle(block, self.env.target_xy)
        ):
            normal = np.array([-direction[1], direction[0]])
            side = np.sign(np.dot(block - self.env.obstacle_xy, normal)) or 1.0
            waypoint = self.env.obstacle_xy + normal * side * 0.095
            local = self.env._unit(waypoint - block)
            desired_xy = block - local * config.contact_standoff_m
            direction = local
            phase = "avoid"
            if np.linalg.norm(tool[:2] - desired_xy) < 0.025:
                desired_xy = tool[:2] + local * 0.018
        vertical_centering_correction_m = float(
            full_push_quality["vertical_centering_correction_m"] if vertical_recentering_required else 0.0
        )
        target_height = (
            tool[2] + vertical_centering_correction_m if vertical_recentering_required else operational_height
        )
        target_xyz = np.array([desired_xy[0], desired_xy[1], target_height], dtype=np.float64)
        position_error = target_xyz - tool
        position_norm = float(np.linalg.norm(position_error))
        if position_norm > config.approach_step_limit_m:
            position_error *= config.approach_step_limit_m / position_norm
        commanded_target_xyz = tool + position_error
        if self._side_contact_ik_planner is None:
            self._side_contact_ik_planner = self._make_side_contact_ik_planner()
        tracking = self._side_contact_ik_planner.track_target(
            commanded_target_xyz,
            direction,
            initial=self.env.data.qpos[:_JOINTS].copy(),
        )
        joint_target = np.asarray(tracking["best"]["joint_position_rad"], dtype=np.float64)
        tracking_feasible = bool(tracking["feasible"])
        if not tracking_feasible:
            joint_target = self.env.data.qpos[:_JOINTS].copy()
            phase = "side_pose_tracking_infeasible"
            advance_gate = False
        vertical_recentering_commanded = bool(vertical_recentering_required and tracking_feasible)
        adaptive_edge_step_limited = bool(adaptive_edge_step_limited and tracking_feasible)
        normalized = np.clip(
            (joint_target - self.env.data.qpos[:_JOINTS]) / self.env.config.max_joint_delta,
            -1.0,
            1.0,
        ).astype(np.float32)
        return normalized, {
            "phase": phase,
            "teacher_confidence": float(np.exp(-5.0 * min(position_norm, 0.5))),
            "joint_target": joint_target.astype(np.float32),
            "contact_geometry_push_gate": verified_contact_gate,
            "contact_geometry_advance_gate": advance_gate,
            "contact_geometry_contact_acquisition_gate": contact_acquisition_gate,
            "contact_geometry_verified_contact_gate": verified_contact_gate,
            "contact_geometry_signed_alignment_gate": signed_alignment_gate,
            "contact_geometry_actual_contact": contact,
            "contact_geometry_current_contact": current_contact,
            "contact_geometry_trace_raw_contact": trace_raw_contact,
            "contact_geometry_valid_push_side_contact": semantic_side_contact,
            "contact_geometry_current_side_contact_valid": current_side_valid,
            "contact_geometry_trace_side_contact_valid": trace_side_valid,
            "contact_geometry_trace_geometric_side_contact": trace_geometric_side,
            "contact_geometry_current_geometric_side_contact": current_geometric_side,
            "contact_geometry_benign_side_acquisition": benign_side_acquisition,
            "contact_geometry_invalid_raw_contact": invalid_raw_contact,
            "contact_geometry_side_contact_metrics": side_metrics,
            "contact_geometry_contact_opens_gate": contact_opens_gate,
            "contact_geometry_environment_obstacle_detour": (environment_obstacle_detour),
            "contact_geometry_along_m": along,
            "contact_geometry_lateral_m": lateral,
            "contact_geometry_vertical_error_m": vertical_error,
            "contact_geometry_pusher_block_gap_m": pusher_block_gap,
            "contact_geometry_pusher_block_maximum_gap_m": (
                pusher_block_maximum_gap
            ),
            "contact_geometry_advance_step_m": advance_step,
            "contact_geometry_full_push_quality": full_push_quality,
            "contact_geometry_lateral_full_push_reserve_m": (lateral_full_push_reserve),
            "contact_geometry_minimum_lateral_full_push_reserve_m": (minimum_lateral_full_push_reserve),
            "contact_geometry_lateral_full_push_allowed": (lateral_full_push_allowed),
            "contact_geometry_robust_full_push_allowed": (robust_full_push_allowed),
            "contact_geometry_vertical_recentering_required": (vertical_recentering_required),
            "contact_geometry_vertical_recentering_correction_m": (vertical_centering_correction_m),
            "contact_geometry_vertical_recentering_target_z_m": target_height,
            "contact_geometry_vertical_recentering_xy_hold": bool(vertical_recentering_required),
            "contact_geometry_vertical_recentering_commanded": (vertical_recentering_commanded),
            "contact_geometry_adaptive_full_push_step_limited": (adaptive_edge_step_limited),
            "contact_geometry_adaptive_edge_step_limited": (adaptive_edge_step_limited),
            "contact_geometry_operational_height_m": operational_height,
            "contact_geometry_solver": "five_constraint_side_normal_tracking_ik",
            "contact_geometry_tracking_ik": tracking,
            "contact_geometry_commanded_target_xyz_m": commanded_target_xyz.tolist(),
        }

    def _common_metadata(self) -> dict[str, Any]:
        return {
            "claim_level": EXPERT_CLAIM_LEVEL,
            "parameter_source": PHYSICAL_EXPERT_V11_PARAMETER_SOURCE,
            "physical_samples": 0,
            "physical_trials": 0,
            "physically_calibrated": False,
            "physical_hardware_connected": False,
            "physical_validation": False,
            "deployment_equivalent": False,
            "privileged_state_used": [
                "block_pose",
                "target_pose",
                "tool_pose",
                "simulator_joint_state",
                "tool_block_contact",
                "tool_block_contact_frame",
                "tool_block_contact_point",
                "tool_block_contact_force",
                "target_footprint_coverage",
            ],
            "precontact_planner": "five_constraint_side_ik_plus_dense_audited_joint_rrt_connect",
            "precontact_planner_uses_privileged_state": True,
            "precontact_planner_deployment_equivalent": False,
            "precontact_waypoint_collision_audited": bool(self._precontact_path_audit.get("valid", False)),
            "precontact_waypoint_path_audit": self._precontact_path_audit,
            "precontact_side_contact_ik": self._precontact_side_ik_report,
            "precontact_joint_path": self._precontact_joint_path_report,
            "precontact_plan_feasible": self._precontact_plan_feasible,
            "precontact_execution_substep_audited": True,
            "precontact_unexpected_contact_steps": (self._precontact_unexpected_contact_steps),
            "precontact_forbidden_desk_steps": self._precontact_forbidden_desk_steps,
            "precontact_mode": self._precontact_mode,
            "global_reacquisition_count": self._global_reacquisition_count,
            "valid_push_side_contact_steps": self._valid_push_side_contact_steps,
            "invalid_raw_contact_steps": self._invalid_raw_contact_steps,
            "invalid_contact_advance_attempts": self._invalid_contact_advance_attempts,
            "operational_tracking_failure_score": self._operational_tracking_failure_score,
            "joint_limit_recovery_steps": self._joint_limit_recovery_steps,
            "runtime_clearance_recovery_active": self._runtime_clearance_recovery_active,
            "runtime_clearance_recovery_steps": self._runtime_clearance_recovery_steps,
            "adaptive_edge_push_inflight_hold_steps_remaining": (
                max(
                    self.env.command_delay_steps - self._pending_effect_wait_steps,
                    0,
                )
                if self._pending_effect_kind == "adaptive_edge_push"
                else 0
            ),
            "adaptive_edge_push_inflight_hold_total_steps": (
                self._adaptive_edge_push_inflight_hold_total_steps
            ),
            "contact_acquisition_inflight_hold_total_steps": (
                self._contact_acquisition_inflight_hold_total_steps
            ),
            "adaptive_edge_push_feedback_pending": bool(self._pending_effect_kind == "adaptive_edge_push"),
            "pending_effect_epoch": self._pending_effect_epoch,
            "pending_effect_command_id": self._pending_effect_command_id,
            "pending_effect_kind": self._pending_effect_kind,
            "pending_effect_send_step": self._pending_effect_send_step,
            "pending_effect_wait_steps": self._pending_effect_wait_steps,
            "pending_effect_total_wait_steps": self._pending_effect_total_wait_steps,
            "pending_effect_vertical_recentering": (self._pending_effect_vertical_recentering),
            "pending_effect_vertical_recentering_signed_delta_z_m": (
                self._pending_effect_vertical_recentering_signed_delta_z_m
            ),
            "last_vertical_recenter_requested_delta_z_m": (self._last_vertical_recenter_requested_delta_z_m),
            "last_vertical_recenter_realized_delta_z_m": (self._last_vertical_recenter_realized_delta_z_m),
            "last_vertical_recenter_completion_success": (self._last_vertical_recenter_completion_success),
            "vertical_recenter_realized_progress_tolerance_m": (
                _VERTICAL_RECENTER_REALIZED_PROGRESS_TOLERANCE_M
            ),
            "contact_clearance_bridge_active": (self._contact_clearance_bridge_active),
            "contact_clearance_bridge_failed": (self._contact_clearance_bridge_failed),
            "contact_clearance_bridge_failure_reason": (self._contact_clearance_bridge_failure_reason),
            "contact_clearance_bridge_target_rad": (self._contact_clearance_bridge_target.astype(np.float32)),
            "contact_clearance_bridge_epoch": self._contact_clearance_bridge_epoch,
            "contact_clearance_bridge_first_command_id": (self._contact_clearance_bridge_first_command_id),
            "contact_clearance_bridge_next_audit_command_id": (
                self._contact_clearance_bridge_next_audit_command_id
            ),
            "contact_clearance_bridge_last_submitted_command_id": (
                self._contact_clearance_bridge_last_submitted_command_id
            ),
            "contact_clearance_bridge_audited_effects": (self._contact_clearance_bridge_audited_effects),
            "contact_clearance_bridge_last_start_audit": (self._contact_clearance_bridge_last_start_audit),
            "contact_clearance_bridge_queue_inspected_or_mutated": False,
            "last_consumed_applied_command_id": (self._last_consumed_applied_command_id),
            "last_consumed_effect_kind": self._last_consumed_effect_kind,
            "command_feedback_gap_count": self._command_feedback_gap_count,
            "adaptive_edge_reacquisition_count": (self._adaptive_edge_reacquisition_count),
            "push_side_contact_semantics": PushSideContactThresholds().profile_hash,
            "global_reacquisition_exhausted": bool(
                self._global_reacquisition_count >= self.v11_config.max_global_reacquisitions
                and self._requires_global_reacquisition()
            ),
            "demonstration_candidate_valid": bool(
                self._precontact_path_audit.get("valid", False)
                and self._precontact_unexpected_contact_steps == 0
                and self._precontact_forbidden_desk_steps == 0
                and self._valid_push_side_contact_steps > 0
                and self._invalid_contact_advance_attempts == 0
            ),
            "v11_config": asdict(self.v11_config),
            "operational_tool_height_m": self._operational_tool_height_m(),
            "operational_height_reference_m": (
                None
                if not np.isfinite(self._operational_height_reference_m)
                else self._operational_height_reference_m
            ),
            "operational_height_source": (
                "executed_precontact_pose"
                if np.isfinite(self._operational_height_reference_m)
                else "nominal_orientation_invariant_clearance"
            ),
            "tool_height_clearance_method": ("nominal_orientation_invariant_box_half_diagonal_target"),
            "tool_height_runtime_clearance_guarantee": False,
        }

    def _operational_tool_height_m(self) -> float:
        """Return a conservative nominal center-height target.

        ``||half_extent||`` bounds the vertical projection of a box under any
        rotation.  Adding it to the workbench top provides a useful nominal
        target, but command delay and force-limited dynamics can still diverge
        from that target.  Only V7 physics-substep telemetry and its opt-in
        runtime guard can establish execution-time clearance.
        """

        if not isinstance(self.env, RealisticEdgeArmEnvV7):
            return float(self.v11_config.tool_height_m)
        if np.isfinite(self._operational_height_reference_m):
            return float(self._operational_height_reference_m)
        tool = self.env._ids["tool_geom"]
        desk = self.env._desk_geom
        tool_radius_bound = float(np.linalg.norm(self.env.model.geom_size[tool]))
        desk_rotation = self.env.data.geom_xmat[desk].reshape(3, 3)
        desk_vertical_radius = float(np.dot(np.abs(desk_rotation[2]), self.env.model.geom_size[desk]))
        desk_top = float(self.env.data.geom_xpos[desk, 2] + desk_vertical_radius)
        clearance_height = desk_top + tool_radius_bound + self.v11_config.minimum_pusher_desk_clearance_m
        return float(max(self.v11_config.tool_height_m, clearance_height))


def load_physical_expert_v11(
    checkpoint: object | None = None,
    *,
    config: PhysicalExpertV11Config | None = None,
) -> PhysicalClosedLoopExpertV11:
    if checkpoint is not None:
        raise ValueError("physical expert V11 has no learned checkpoint")
    return PhysicalClosedLoopExpertV11(config=config)


__all__ = [
    "PHYSICAL_EXPERT_V11_CONFIG_FORMAT",
    "PHYSICAL_EXPERT_V11_CONFIG_SCHEMA_VERSION",
    "PHYSICAL_EXPERT_V11_PARAMETER_SOURCE",
    "PHYSICAL_EXPERT_V11_VERSION",
    "PhysicalClosedLoopExpertV11",
    "PhysicalExpertV11Config",
    "load_physical_expert_v11",
]
