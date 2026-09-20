"""Formal stock-gripper successor to the privileged synthetic V11 expert.

V12 deliberately reuses the executed, delayed-control state machine from V11
but binds it to exactly one plant: :class:`RealisticEdgeArmEnvV9` with the
exact :class:`RealisticEnvV9Config`.  Geometry queries are upgraded from the
legacy solid planning envelope to the two independent distal-tip references
and the complete CAD-derived CoACD safety union.

This remains a simulator-privileged demonstration source.  It contains no
physical robot samples, no real-camera calibration, and no physical success
claim.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping

import numpy as np

from .joint_path_planner_v1 import JointPathPlannerConfig
from .physical_expert_v11 import (
    PHYSICAL_EXPERT_V11_VERSION,
    PhysicalClosedLoopExpertV11,
    PhysicalExpertV11Config,
)
from .sim2real_env_v6 import RealisticEdgeArmEnvV6
from .sim2real_env_v9 import (
    STOCK_GRIPPER_DYNAMICS_PROFILE_V9,
    STOCK_GRIPPER_GEOMETRY_VERSION_V9,
    RealisticEdgeArmEnvV9,
    RealisticEnvV9Config,
)
from .side_contact_ik_v1 import SideContactIKConfig, SideContactIKPlannerV1


PHYSICAL_EXPERT_V12_VERSION = (
    "edgearm-privileged-physical-expert-v12-stock-gripper-v9-contact-envelope"
)
PHYSICAL_EXPERT_V12_TEACHER_TYPE = "physical_expert_v12_stock_gripper_v9_synthetic_privileged"
PHYSICAL_EXPERT_V12_PARAMETER_SOURCE = (
    "synthetic_v9_stock_gripper_per_jaw_tip_ik_and_full_coacd_safety_path"
)
PHYSICAL_EXPERT_V12_CONFIG_FORMAT = "edgearm-deterministic-physical-expert-v12-config"
PHYSICAL_EXPERT_V12_CONFIG_SCHEMA_VERSION = 2
PHYSICAL_EXPERT_V12_SIDE_CONTACT_HEIGHT_CANDIDATES_M = (
    0.045,
    0.047,
    0.043,
    0.050,
    0.040,
    0.052,
)
PHYSICAL_EXPERT_V12_STOCK_NOMINAL_OPERATIONAL_HEIGHT_M = 0.045
PHYSICAL_EXPERT_V12_CONTACT_PART_CENTRAL_SIDE_MARGIN_M = 0.004


@dataclass(frozen=True)
class PhysicalExpertV12Config(PhysicalExpertV11Config):
    """V11 control limits plus frozen stock-gripper V9 geometry priors."""

    # ``PhysicalExpertV9Config`` still classifies its legacy scalar height in
    # [0.070, 0.082].  V12 does not use that scalar for operational control;
    # it retains the lowest compatible value solely for inherited constructor
    # compatibility and records that distinction explicitly in provenance.
    tool_height_m: float = 0.070
    stock_nominal_operational_height_m: float = (
        PHYSICAL_EXPERT_V12_STOCK_NOMINAL_OPERATIONAL_HEIGHT_M
    )
    side_contact_height_candidates_m: tuple[float, ...] = (
        PHYSICAL_EXPERT_V12_SIDE_CONTACT_HEIGHT_CANDIDATES_M
    )

    def __post_init__(self) -> None:
        super().__post_init__()
        if not np.isclose(
            self.stock_nominal_operational_height_m,
            PHYSICAL_EXPERT_V12_STOCK_NOMINAL_OPERATIONAL_HEIGHT_M,
            rtol=0.0,
            atol=1.0e-12,
        ):
            raise ValueError(
                "stock_nominal_operational_height_m is frozen to 0.045 for V12"
            )
        candidates = tuple(float(value) for value in self.side_contact_height_candidates_m)
        if candidates != PHYSICAL_EXPERT_V12_SIDE_CONTACT_HEIGHT_CANDIDATES_M:
            raise ValueError(
                "side_contact_height_candidates_m is frozen to the ordered "
                "V12 stock-gripper search profile"
            )


class PhysicalClosedLoopExpertV12(PhysicalClosedLoopExpertV11):
    """V11 execution logic bound fail-closed to the exact V9 stock plant."""

    teacher_type = PHYSICAL_EXPERT_V12_TEACHER_TYPE
    selected_update = 12
    checkpoint = ""

    def __init__(
        self,
        env: RealisticEdgeArmEnvV9 | None = None,
        config: PhysicalExpertV12Config | None = None,
    ) -> None:
        if config is not None and type(config) is not PhysicalExpertV12Config:
            raise TypeError("PhysicalClosedLoopExpertV12 requires PhysicalExpertV12Config")
        if env is not None:
            self._require_exact_v9_environment(env)
        self.v12_config = config or PhysicalExpertV12Config()
        self._v12_invalid_tool_block_contact_steps = 0
        self._v12_invalid_tool_block_contact_last_step = -1
        super().__init__(env=env, config=self.v12_config)
        # V11 owns this attribute throughout its implementation.  Retain that
        # compatibility alias while exposing the formal V12 name as authority.
        self.v12_config = self.v11_config

    @staticmethod
    def _require_exact_v9_environment(env: object) -> RealisticEdgeArmEnvV9:
        if type(env) is not RealisticEdgeArmEnvV9:
            raise TypeError(
                "PhysicalClosedLoopExpertV12 requires exact RealisticEdgeArmEnvV9"
            )
        if type(env.stock_distal_tip_config) is not RealisticEnvV9Config:
            raise TypeError(
                "PhysicalClosedLoopExpertV12 requires exact RealisticEnvV9Config"
            )
        if env.profile_version != STOCK_GRIPPER_DYNAMICS_PROFILE_V9:
            raise TypeError("PhysicalClosedLoopExpertV12 requires the V9 stock profile")
        return env

    def reset(self, env: RealisticEdgeArmEnvV6 | None = None) -> None:
        if env is not None:
            self._require_exact_v9_environment(env)
        elif getattr(self, "env", None) is not None:
            self._require_exact_v9_environment(self.env)
        self._v12_invalid_tool_block_contact_steps = 0
        self._v12_invalid_tool_block_contact_last_step = -1
        super().reset(env)
        self._require_exact_v9_environment(self.env)

    def _make_side_contact_ik_planner(self) -> SideContactIKPlannerV1:
        env = self._require_exact_v9_environment(self.env)
        return SideContactIKPlannerV1(
            env,
            SideContactIKConfig(
                minimum_pusher_desk_clearance_m=(
                    self.v12_config.minimum_pusher_desk_clearance_m
                ),
                minimum_tool_block_safety_clearance_m=(
                    JointPathPlannerConfig().minimum_tool_block_clearance_m
                ),
                minimum_contact_part_central_side_margin_m=(
                    PHYSICAL_EXPERT_V12_CONTACT_PART_CENTRAL_SIDE_MARGIN_M
                ),
                height_candidates_m=(
                    PHYSICAL_EXPERT_V12_SIDE_CONTACT_HEIGHT_CANDIDATES_M
                ),
            ),
        )

    def _current_tool_desk_clearance_m(self) -> float:
        env = self._require_exact_v9_environment(self.env)
        return float(
            env._minimum_tool_safety_signed_distance_for_data(
                env._desk_geom,
                env.data,
            )
        )

    def _current_tool_block_gap_interval_m(self) -> tuple[float, float]:
        env = self._require_exact_v9_environment(self.env)
        distances = env._tool_planning_signed_distances_for_data(
            env._ids["block_geom"],
            env.data,
        )
        if distances.shape != (2,):
            raise RuntimeError("V12 requires exactly two distal-tip planning distances")
        return float(np.min(distances)), float(np.max(distances))

    def _current_tool_block_safety_clearance_m(self) -> float:
        """Return the minimum block clearance over the complete CoACD union."""

        env = self._require_exact_v9_environment(self.env)
        return float(
            env._minimum_tool_safety_signed_distance_for_data(
                env._ids["block_geom"],
                env.data,
            )
        )

    @staticmethod
    def _minimum_path_endpoint_block_clearance_m() -> float:
        return float(JointPathPlannerConfig().minimum_tool_block_clearance_m)

    @staticmethod
    def _trace_invalid_tool_block_contact_count(
        trace: Mapping[str, Any] | None,
    ) -> int:
        if trace is None:
            return 0
        raw = np.asarray(
            trace.get("invalid_tool_block_contact_count", ()),
            dtype=np.int64,
        ).reshape(-1)
        count = int(np.sum(np.maximum(raw, 0))) if raw.size else 0
        if bool(trace.get("invalid_tool_block_contact_any", False)):
            count = max(count, 1)
        return count

    def _current_invalid_tool_block_contact_count(self) -> int:
        env = self._require_exact_v9_environment(self.env)
        direction = env._unit(env.target_xy - env.block_xy())
        metrics = env.current_push_side_contact_metrics(direction)
        return int(metrics.get("invalid_tool_block_contact_count", 0))

    def _invalid_tool_block_contact_evidence(self) -> tuple[int, int]:
        env = self._require_exact_v9_environment(self.env)
        return (
            self._trace_invalid_tool_block_contact_count(
                env._physics_substep_contact_v1
            ),
            self._current_invalid_tool_block_contact_count(),
        )

    def _record_v12_invalid_contact_step(self) -> None:
        env = self._require_exact_v9_environment(self.env)
        if self._v12_invalid_tool_block_contact_last_step == env.step_count:
            return
        self._v12_invalid_tool_block_contact_steps += 1
        self._v12_invalid_tool_block_contact_last_step = int(env.step_count)

    def _invalid_push_contact_present(self) -> bool:
        trace_count, current_count = self._invalid_tool_block_contact_evidence()
        # Escape must continue until the *whole* stock gripper is a valid path
        # start, not merely until MuJoCo drops the last active contact pair.
        # Otherwise a proximal CoACD part can remain inside the planner's
        # positive clearance margin while both non-colliding tip references
        # already report a comfortable gap, and reacquisition deadlocks at an
        # ``invalid_endpoint`` start state.
        safety_clearance = self._current_tool_block_safety_clearance_m()
        return bool(
            trace_count > 0
            or current_count > 0
            or safety_clearance
            < self._minimum_path_endpoint_block_clearance_m()
        )

    def _ingest_precontact_execution_telemetry(self) -> None:
        env = self._require_exact_v9_environment(self.env)
        if env.step_count <= self._precontact_last_trace_step:
            return
        trace = env._physics_substep_contact_v1
        if trace is not None:
            invalid_count = self._trace_invalid_tool_block_contact_count(trace)
            legacy_unexpected = bool(
                trace["contact_any"]
                and not trace["valid_push_side_contact_any"]
                and not np.any(
                    np.asarray(trace["geometric_push_side_contact_count"]) > 0
                )
            )
            invalid_contact = bool(invalid_count > 0 or legacy_unexpected)
            self._precontact_unexpected_contact_steps += int(invalid_contact)
            if invalid_contact:
                self._record_v12_invalid_contact_step()
            self._precontact_forbidden_desk_steps += int(
                bool(trace["forbidden_tool_desk_penetration_any"])
                or bool(trace["forbidden_non_tool_robot_desk_penetration_any"])
            )
        self._precontact_last_trace_step = int(env.step_count)

    def action(
        self,
        env_or_observation: RealisticEdgeArmEnvV6 | Mapping[str, np.ndarray] | None = None,
        observation: Mapping[str, np.ndarray] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        if isinstance(env_or_observation, RealisticEdgeArmEnvV6):
            env = self._require_exact_v9_environment(env_or_observation)
            if env is not self.env:
                self.reset(env)
            current_observation = observation
        else:
            if observation is not None:
                raise TypeError("two-argument action requires the environment first")
            current_observation = env_or_observation
        env = self._require_exact_v9_environment(self.env)
        self._validate_observation(current_observation)
        if env.step_count == 0 and self._precontact_steps > 0:
            self.reset()
            env = self._require_exact_v9_environment(self.env)

        trace_invalid_count, current_invalid_count = (
            self._invalid_tool_block_contact_evidence()
        )
        actual_invalid_contact = bool(
            trace_invalid_count > 0 or current_invalid_count > 0
        )
        precontact_clearance_escape = bool(
            not self._precontact_complete
            and self._current_tool_block_safety_clearance_m()
            < self._minimum_path_endpoint_block_clearance_m()
        )
        if actual_invalid_contact or precontact_clearance_escape:
            if actual_invalid_contact:
                self._record_v12_invalid_contact_step()
            if self._precontact_mode != "invalid_contact_escape":
                self._invalid_raw_contact_steps += int(actual_invalid_contact)
                self._contact_semantics_last_step = int(env.step_count)
                self._had_contact = False
                self._contact_loss_steps = 0
                self._clear_pending_command_effect()
                self._precontact_mode = "invalid_contact_escape"
                self._precontact_complete = False
                self._invalid_contact_escape_remaining = (
                    self.v12_config.invalid_contact_escape_steps
                )
            action, metadata = self._invalid_contact_escape_action()
        else:
            action, metadata = super().action(current_observation)
        return action, self._stamp_v12_metadata(metadata)

    def _invalid_contact_escape_action(self) -> tuple[np.ndarray, dict[str, Any]]:
        """Retreat from contact or a path-invalid near-clearance state."""

        action, metadata = super()._invalid_contact_escape_action()
        trace_count, current_count = self._invalid_tool_block_contact_evidence()
        actual_invalid_contact = bool(trace_count > 0 or current_count > 0)
        safety_clearance = self._current_tool_block_safety_clearance_m()
        clearance_only = bool(
            not actual_invalid_contact
            and safety_clearance < self._minimum_path_endpoint_block_clearance_m()
        )
        if clearance_only:
            tracking_feasible = bool(
                metadata.get("invalid_contact_escape_tracking_feasible", False)
            )
            phase = (
                "precontact_safety_clearance_escape"
                if tracking_feasible
                else "precontact_safety_clearance_escape_tracking_infeasible_hold"
            )
            metadata.update(
                {
                    "phase": phase,
                    "planned_phase": phase,
                    "contact_geometry_invalid_raw_contact": False,
                    "precontact_safety_clearance_escape": True,
                }
            )
        metadata.update(
            {
                "precontact_safety_clearance_escape": clearance_only,
                "current_tool_block_full_safety_clearance_m": safety_clearance,
                "minimum_path_endpoint_block_clearance_m": (
                    self._minimum_path_endpoint_block_clearance_m()
                ),
            }
        )
        return action, metadata

    def _common_metadata(self) -> dict[str, Any]:
        metadata = super()._common_metadata()
        return self._stamp_v12_metadata(metadata)

    def _stamp_v12_metadata(self, metadata: Mapping[str, Any]) -> dict[str, Any]:
        result = dict(metadata)
        env = self._require_exact_v9_environment(self.env)
        trace_invalid_count, current_invalid_count = (
            self._invalid_tool_block_contact_evidence()
        )
        invalid_any = bool(trace_invalid_count > 0 or current_invalid_count > 0)
        result.pop("v11_config", None)
        result.update(
            {
                "version": PHYSICAL_EXPERT_V12_VERSION,
                "teacher_type": PHYSICAL_EXPERT_V12_TEACHER_TYPE,
                "delegate_controller_version": PHYSICAL_EXPERT_V11_VERSION,
                "delegate_version": PHYSICAL_EXPERT_V11_VERSION,
                "parameter_source": PHYSICAL_EXPERT_V12_PARAMETER_SOURCE,
                "config_format": PHYSICAL_EXPERT_V12_CONFIG_FORMAT,
                "config_schema_version": PHYSICAL_EXPERT_V12_CONFIG_SCHEMA_VERSION,
                "environment_type": "RealisticEdgeArmEnvV9",
                "environment_config_type": "RealisticEnvV9Config",
                "environment_profile_version": STOCK_GRIPPER_DYNAMICS_PROFILE_V9,
                "environment_geometry_version": STOCK_GRIPPER_GEOMETRY_VERSION_V9,
                "environment_requires_exact_v9_type": True,
                "expert_provenance": {
                    "controller_version": PHYSICAL_EXPERT_V12_VERSION,
                    "teacher_type": PHYSICAL_EXPERT_V12_TEACHER_TYPE,
                    "inherited_execution_state_machine": PHYSICAL_EXPERT_V11_VERSION,
                    "environment_profile_version": STOCK_GRIPPER_DYNAMICS_PROFILE_V9,
                    "environment_geometry_version": STOCK_GRIPPER_GEOMETRY_VERSION_V9,
                    "environment_type": "RealisticEdgeArmEnvV9",
                    "environment_config_type": "RealisticEnvV9Config",
                    "physical_samples": 0,
                    "physical_trials": 0,
                    "wrist_camera_calibration_status": "uncalibrated",
                },
                "stock_follower_unmodified": True,
                "added_contact_tool": False,
                "tool_planning_geometry_mode": env._ids[
                    "tool_planning_geometry_mode"
                ],
                "tool_safety_geometry_mode": env._ids[
                    "tool_safety_geometry_mode"
                ],
                "tool_planning_geom_count": len(env._ids["tool_planning_geoms"]),
                "tool_safety_geom_count": len(env._ids["tool_safety_geoms"]),
                "side_contact_ik_profile_hash": (
                    self._side_contact_ik_planner.config.profile_hash
                    if self._side_contact_ik_planner is not None
                    else None
                ),
                "side_contact_height_candidates_m": list(
                    PHYSICAL_EXPERT_V12_SIDE_CONTACT_HEIGHT_CANDIDATES_M
                ),
                "side_contact_minimum_full_safety_block_clearance_m": (
                    self._side_contact_ik_planner.config.minimum_tool_block_safety_clearance_m
                    if self._side_contact_ik_planner is not None
                    else JointPathPlannerConfig().minimum_tool_block_clearance_m
                ),
                "side_contact_minimum_contact_part_central_side_margin_m": (
                    PHYSICAL_EXPERT_V12_CONTACT_PART_CENTRAL_SIDE_MARGIN_M
                ),
                "physical_samples": 0,
                "physical_trials": 0,
                "physically_calibrated": False,
                "physical_hardware_connected": False,
                "physical_validation": False,
                "deployment_equivalent": False,
                "wrist_camera_physical_samples": 0,
                "wrist_camera_physically_calibrated": False,
                "wrist_camera_calibration_status": "uncalibrated",
                "camera_physical_samples": 0,
                "camera_physically_calibrated": False,
                "camera_calibration_status": "uncalibrated",
                "invalid_tool_block_contact_any": invalid_any,
                "invalid_tool_block_contact_count": (
                    trace_invalid_count + current_invalid_count
                ),
                "invalid_tool_block_contact_trace_count": trace_invalid_count,
                "invalid_tool_block_contact_current_count": current_invalid_count,
                "invalid_tool_block_contact_steps": (
                    self._v12_invalid_tool_block_contact_steps
                ),
                "current_tool_block_full_safety_clearance_m": (
                    self._current_tool_block_safety_clearance_m()
                ),
                "minimum_path_endpoint_block_clearance_m": (
                    self._minimum_path_endpoint_block_clearance_m()
                ),
                "invalid_contact_semantics": (
                    "v2_explicit_invalid_count_fail_closed_no_geometric_any_waiver"
                ),
                "v12_config": asdict(self.v12_config),
                "legacy_delegate_tool_height_m_not_used_for_v12_operational_height": (
                    self.v12_config.tool_height_m
                ),
                "operational_tool_height_m": self._operational_tool_height_m(),
                "operational_height_source": (
                    "executed_precontact_pose"
                    if np.isfinite(self._operational_height_reference_m)
                    else "v12_stock_geometry_derived_nominal_0.045m"
                ),
                "tool_height_clearance_method": (
                    "executed_precontact_height_else_frozen_stock_nominal"
                ),
                "tool_height_runtime_clearance_guarantee": False,
            }
        )
        result["demonstration_candidate_valid"] = bool(
            result.get("demonstration_candidate_valid", False)
            and not invalid_any
            and self._v12_invalid_tool_block_contact_steps == 0
        )
        return result

    def _operational_tool_height_m(self) -> float:
        if np.isfinite(self._operational_height_reference_m):
            return float(self._operational_height_reference_m)
        return float(self.v12_config.stock_nominal_operational_height_m)


def load_physical_expert_v12(
    checkpoint: object | None = None,
    *,
    config: PhysicalExpertV12Config | None = None,
) -> PhysicalClosedLoopExpertV12:
    if checkpoint is not None:
        raise ValueError("physical expert V12 has no learned checkpoint")
    return PhysicalClosedLoopExpertV12(config=config)


__all__ = [
    "PHYSICAL_EXPERT_V12_CONFIG_FORMAT",
    "PHYSICAL_EXPERT_V12_CONFIG_SCHEMA_VERSION",
    "PHYSICAL_EXPERT_V12_CONTACT_PART_CENTRAL_SIDE_MARGIN_M",
    "PHYSICAL_EXPERT_V12_PARAMETER_SOURCE",
    "PHYSICAL_EXPERT_V12_SIDE_CONTACT_HEIGHT_CANDIDATES_M",
    "PHYSICAL_EXPERT_V12_STOCK_NOMINAL_OPERATIONAL_HEIGHT_M",
    "PHYSICAL_EXPERT_V12_TEACHER_TYPE",
    "PHYSICAL_EXPERT_V12_VERSION",
    "PhysicalClosedLoopExpertV12",
    "PhysicalExpertV12Config",
    "load_physical_expert_v12",
]
