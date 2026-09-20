"""V22 stock-gripper task-frame wiring for semantic broad-face pushing.

V12/V13 reset under the historical 96/2/94 contact identity because those
artifacts are immutable.  Once the reset and its original hold audit pass,
this module opts the same exact V10 plant into the reversible V22 96/8/88
identity and repeats the latched-hold proof with the V4 guard.  No collision
mesh, actuator, controller, action space, or robot hardware is changed.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, replace
from typing import Any

import numpy as np

from .asymmetric_multiview_ppo_v1 import (
    SOURCE_TYPE,
    MultiViewRendererProtocolV1,
    StockGripperTaskFrameAdapterV13,
    StockGripperTaskSpaceActionConfigV12,
    StockTaskFrameNoSafeRecoveryV13,
    TaskSpaceActionResultV5,
    StockTaskFrameResetInfeasibleV12,
    reset_stock_taskframe_episode_v12,
)
from .sim2real_env_v10 import RealisticEdgeArmEnvV10
from .stock_gripper_action_guard_v3 import LatchedAbsoluteTargetInfeasibleV3
from .stock_gripper_action_guard_v4 import (
    STOCK_GRIPPER_ACTION_GUARD_FORMAT_V4,
    StockGripperActionGuardV4,
)
from .stock_gripper_push_face_contact_v22 import (
    STOCK_GRIPPER_PUSH_FACE_CONTACT_FORMAT_V22,
    configure_stock_gripper_push_face_contact_v22,
    restore_stock_gripper_distal_contact_v22,
)


STOCK_GRIPPER_TASKFRAME_RUNTIME_FORMAT_V22 = "edgearm-stock-gripper-semantic-push-face-taskframe-runtime-v22"
STOCK_GRIPPER_TASKFRAME_RESET_RETRY_FORMAT_V22 = "edgearm-stock-gripper-semantic-push-face-reset-retry-v22"
STOCK_GRIPPER_CONTACT_TELEMETRY_FORMAT_V22 = "edgearm-stock-gripper-semantic-push-face-transition-contact-v22"
STOCK_GRIPPER_TRANSIENT_DIRECTION_RECOVERY_REASON_V26 = (
    "stock_taskframe_v22_guarded_transient_direction_recovery"
)
STOCK_GRIPPER_POLICY_PROJECTION_RECOVERY_REASON_V26 = (
    "stock_taskframe_v22_guarded_policy_projection_recovery"
)
_PARENT_PREDICTED_DIRECTION_REVERSAL_REASON = (
    "stock_taskframe_v12_predicted_direction_reversal"
)
_PARENT_GUARDED_RECOVERY_REASONS = {
    "stock_taskframe_v12_no_directionally_valid_candidate",
    "stock_taskframe_v12_ik_not_feasible",
    "stock_taskframe_v12_target_not_representable",
    "stock_taskframe_v12_guard_no_safe_candidate",
    _PARENT_PREDICTED_DIRECTION_REVERSAL_REASON,
}


class StockTaskFrameResetInfeasibleV22(StockTaskFrameResetInfeasibleV12):
    """Expected rejection when the post-reset V22 hold proof fails."""


class StockGripperTaskFrameAdapterV22(StockGripperTaskFrameAdapterV13):
    """V13 task-frame transport with the opt-in V22 contact identity."""

    def __init__(
        self,
        env: RealisticEdgeArmEnvV10,
        config: StockGripperTaskSpaceActionConfigV12 | None = None,
    ) -> None:
        super().__init__(env, config)
        self.push_face_profile_v22: dict[str, Any] = {}

    def _recovery_tail_anchor_rad(
        self,
        requested_recovery_target_rad: np.ndarray,
        workspace_recovery_anchor_rad: np.ndarray,
    ) -> np.ndarray:
        """Hold the verified intermediate target before replanning.

        Directly chasing the global joint-space anchor can sweep a broad push
        face through the block on the first tail step even when the selected
        intermediate recovery command itself is safe.  V22 therefore proves
        a hold at that intermediate target and replans on the next real
        control decision; the exact V4 guard and zero-invalid-contact rule are
        unchanged.
        """

        del workspace_recovery_anchor_rad
        return np.asarray(
            requested_recovery_target_rad, dtype=np.float64
        ).copy()

    def _recovery_continuation_policy_name(self) -> str:
        return "receding_horizon_verified_intermediate_target_hold"

    def begin_episode(self, seed: int) -> None:
        # A failed V22 attempt must never leak its expanded contact identity
        # into the immutable V12 reset planner on the next retry.
        restore_stock_gripper_distal_contact_v22(self.env)
        super().begin_episode(seed)

        try:
            profile = configure_stock_gripper_push_face_contact_v22(self.env)
            guard = StockGripperActionGuardV4(self.env, self.config.guard)
            if self._latched_joint_target is None:
                raise RuntimeError("V22 reset lost its latched joint target")
            hold_action, hold_identity = guard.action_for_absolute_target(self._latched_joint_target)
            selected_hold, hold_report = guard.select(
                hold_action,
                baseline_action=hold_action,
                require_hold_tail=True,
            )
            if selected_hold is None or not np.array_equal(selected_hold, hold_action):
                raise RuntimeError("V22 dynamic latched-hold audit failed")
            hold_tail = dict(hold_report["selected_hold_tail_forecast"])
            if not (
                bool(hold_tail["hard_valid"])
                and bool(hold_tail["planning_margin_valid"])
                and bool(hold_tail["remaining_submissions_target_latched_holds"])
            ):
                raise RuntimeError("V22 hold-tail proof is incomplete")
        except (LatchedAbsoluteTargetInfeasibleV3, RuntimeError) as error:
            self._episode_active = False
            restore_stock_gripper_distal_contact_v22(self.env)
            raise StockTaskFrameResetInfeasibleV22(
                "stock task-frame V22 post-reset hold audit failed"
            ) from error

        self.guard = guard
        self.last_guard_report = hold_report
        self.push_face_profile_v22 = deepcopy(profile)
        self.env.episode_domain["stock_gripper_taskframe_runtime_v22"] = {
            "format": STOCK_GRIPPER_TASKFRAME_RUNTIME_FORMAT_V22,
            "seed": int(seed),
            "source_type": SOURCE_TYPE,
            "base_runtime_contract": ("edgearm-stock-gripper-taskframe-runtime-v13"),
            "contact_identity_format": (STOCK_GRIPPER_PUSH_FACE_CONTACT_FORMAT_V22),
            "contact_candidate_geom_count": int(profile["contact_candidate_geom_count"]),
            "safety_only_geom_count": int(profile["safety_only_geom_count"]),
            "collision_geometry_changed": False,
            "policy_action_space_changed_from_v13": False,
            "policy_guard_deadlock_terminates_before_env_step": (
                "only_when_no_guard_verified_submission_exists"
            ),
            "guarded_transient_direction_recovery_is_environment_transition": (
                True
            ),
            "guarded_policy_projection_recovery_is_environment_transition": (
                True
            ),
            "guarded_transient_direction_recovery_is_ik_success": False,
            "guarded_transient_direction_recovery_requires_outcome_admission": (
                True
            ),
            "guarded_recovery_continuation_policy": (
                "receding_horizon_verified_intermediate_target_hold"
            ),
            "guarded_recovery_continuation_exact_plant_forecast": True,
            "safety_guard_format": STOCK_GRIPPER_ACTION_GUARD_FORMAT_V4,
            "safety_guard_config": asdict(self.config.guard),
            "initial_hold_target_identity": hold_identity,
            "initial_hold_tail_minimum_clearance_m": float(
                hold_tail["minimum_forecast_safety_only_block_distance_m"]
            ),
            "actual_contact_requires_geometric_semantic_gate": True,
            "stock_follower_unmodified": True,
            "added_contact_tool": False,
            "simulator_privileged": True,
            "physical_samples": 0,
            "production_admission": False,
        }

    def translate(
        self,
        policy_action: np.ndarray,
        *,
        preserve_latched_target: bool = False,
    ) -> TaskSpaceActionResultV5:
        """Separate guard-verified projection from a true safety deadlock.

        A rejected policy request may still produce a recovery command proven
        by the same one-step and braking guard. That command is a legitimate
        projected environment transition, with ``ik_converged=False`` so the
        admission gates can reject repeated holds. Only the V13 exception or a
        malformed recovery proof becomes a pre-step safety terminal.
        """

        translated = (
            super().translate(
                policy_action,
                preserve_latched_target=True,
            )
            if preserve_latched_target
            else super().translate(policy_action)
        )
        if not translated.ik_converged and not translated.guard_safe_candidate:
            recovery = dict(getattr(self, "last_recovery_report", {}))
            recovery["triggering_parent_failure_reason"] = str(
                translated.failure_reason
            )
            self.last_recovery_report = recovery
            selected_scale = recovery.get("selected_guard_scale")
            one_step_clearance = recovery.get("minimum_one_step_clearance_m")
            braking_clearance = recovery.get("minimum_braking_clearance_m")
            numeric_proof = all(
                not isinstance(value, bool)
                and isinstance(value, (int, float))
                and np.isfinite(float(value))
                for value in (
                    selected_scale,
                    one_step_clearance,
                    braking_clearance,
                )
            )
            recovery_is_guard_verified = bool(
                translated.failure_reason in _PARENT_GUARDED_RECOVERY_REASONS
                and recovery.get("same_guard_as_policy_action") is True
                and numeric_proof
                and float(one_step_clearance)
                >= self.config.guard.hard_executed_safety_only_clearance_m
                - 1.0e-12
                and float(braking_clearance)
                >= self.config.guard.hard_executed_safety_only_clearance_m
                - 1.0e-12
            )
            if not recovery_is_guard_verified:
                raise StockTaskFrameNoSafeRecoveryV13(
                    "V22 policy request has no guard-safe candidate; terminate and reset"
                )
            direction_recovery = bool(
                translated.failure_reason
                == _PARENT_PREDICTED_DIRECTION_REVERSAL_REASON
            )
            return replace(
                translated,
                face_label=(
                    "guarded_transient_direction_recovery_v22"
                    if direction_recovery
                    else "guarded_policy_projection_recovery_v22"
                ),
                failure_reason=(
                    STOCK_GRIPPER_TRANSIENT_DIRECTION_RECOVERY_REASON_V26
                    if direction_recovery
                    else STOCK_GRIPPER_POLICY_PROJECTION_RECOVERY_REASON_V26
                ),
                guard_safe_candidate=True,
                guard_selected_scale=float(selected_scale),
                guard_minimum_one_step_clearance_m=float(one_step_clearance),
                guard_minimum_braking_clearance_m=float(braking_clearance),
            )
        return translated


def reset_stock_taskframe_episode_v22(
    env: RealisticEdgeArmEnvV10,
    renderer: MultiViewRendererProtocolV1,
    action_adapter: StockGripperTaskFrameAdapterV22,
    *,
    requested_seed: int,
    obstacle: bool,
    stress: bool,
) -> dict[str, Any]:
    """Run the historical V12 reset, then retain the proven V22 runtime."""

    if type(action_adapter) is not StockGripperTaskFrameAdapterV22:
        raise TypeError("stock V22 reset retry requires exact adapter V22")
    restore_stock_gripper_distal_contact_v22(env)
    base = reset_stock_taskframe_episode_v12(
        env,
        renderer,
        action_adapter,
        requested_seed=requested_seed,
        obstacle=obstacle,
        stress=stress,
    )
    profile = dict(action_adapter.push_face_profile_v22)
    if not profile:
        raise RuntimeError("stock V22 reset completed without a contact profile")
    audit = dict(base)
    audit.update(
        {
            "format": STOCK_GRIPPER_TASKFRAME_RESET_RETRY_FORMAT_V22,
            "base_reset_format": str(base["format"]),
            "contact_identity_format": (STOCK_GRIPPER_PUSH_FACE_CONTACT_FORMAT_V22),
            "contact_candidate_geom_count": int(profile["contact_candidate_geom_count"]),
            "safety_only_geom_count": int(profile["safety_only_geom_count"]),
            "collision_geometry_changed": False,
            "safety_guard_format": STOCK_GRIPPER_ACTION_GUARD_FORMAT_V4,
            "post_reset_v22_latched_hold_audited": True,
        }
    )
    env.episode_domain["stock_gripper_taskframe_reset_retry_v22"] = deepcopy(audit)
    return audit


def transition_contact_telemetry_v22(
    info: dict[str, Any],
    *,
    block_before_xy_m: np.ndarray,
    block_after_xy_m: np.ndarray,
) -> dict[str, Any]:
    """Aggregate one V22 transition over every physics substep."""

    trace = info.get("physics_substep_contact_v1")
    if not isinstance(trace, dict):
        raise RuntimeError("V22 rollout lost physics-substep contact telemetry")
    physics_substeps = int(trace.get("physics_substeps", -1))
    safety_ids = tuple(int(value) for value in trace.get("tool_safety_geom_ids", ()))
    contact_ids = tuple(int(value) for value in trace.get("tool_contact_geom_ids", ()))
    contact_roles = tuple(str(value) for value in trace.get("tool_contact_role_names", ()))
    if (
        physics_substeps < 1
        or len(safety_ids) != 96
        or len(set(safety_ids)) != 96
        or len(contact_ids) != 8
        or len(set(contact_ids)) != 8
        or len(contact_roles) != 8
        or len(set(contact_roles)) != 8
        or not set(contact_ids).issubset(safety_ids)
    ):
        raise RuntimeError("V22 rollout contact geometry identity changed")
    distances = np.asarray(
        trace.get("tool_safety_block_signed_distance_m", ()),
        dtype=np.float64,
    )
    raw_counts = np.asarray(trace.get("tool_block_contact_count", ()), dtype=np.int64)
    valid_counts = np.asarray(trace.get("valid_push_side_contact_count", ()), dtype=np.int64)
    invalid_counts = np.asarray(trace.get("invalid_tool_block_contact_count", ()), dtype=np.int64)
    raw_counts_by_role = np.asarray(
        trace.get("tool_block_contact_count_by_role", ()), dtype=np.int64
    )
    valid_counts_by_role = np.asarray(
        trace.get("valid_push_side_contact_count_by_role", ()), dtype=np.int64
    )
    invalid_counts_by_role = np.asarray(
        trace.get("invalid_tool_block_contact_count_by_role", ()), dtype=np.int64
    )
    expected_vector = (physics_substeps,)
    expected_role_matrix = (physics_substeps, len(contact_roles))
    if (
        distances.shape != (physics_substeps, 96)
        or raw_counts.shape != expected_vector
        or valid_counts.shape != expected_vector
        or invalid_counts.shape != expected_vector
        or raw_counts_by_role.shape != expected_role_matrix
        or valid_counts_by_role.shape != expected_role_matrix
        or invalid_counts_by_role.shape != expected_role_matrix
        or not np.all(np.isfinite(distances))
        or np.any(raw_counts < 0)
        or np.any(valid_counts < 0)
        or np.any(invalid_counts < 0)
        or np.any(raw_counts_by_role < 0)
        or np.any(valid_counts_by_role < 0)
        or np.any(invalid_counts_by_role < 0)
        or not np.array_equal(np.sum(raw_counts_by_role, axis=1), raw_counts)
        or not np.array_equal(np.sum(valid_counts_by_role, axis=1), valid_counts)
        or not np.array_equal(np.sum(invalid_counts_by_role, axis=1), invalid_counts)
    ):
        raise RuntimeError("V22 rollout contact substep arrays are invalid")
    contact_columns = {safety_ids.index(value) for value in contact_ids}
    safety_only_columns = [index for index in range(len(safety_ids)) if index not in contact_columns]
    if len(safety_only_columns) != 88:
        raise RuntimeError("V22 rollout requires 88 safety-only stock parts")
    before = np.asarray(block_before_xy_m, dtype=np.float64)
    after = np.asarray(block_after_xy_m, dtype=np.float64)
    if before.shape != (2,) or after.shape != (2,) or not np.all(np.isfinite(np.r_[before, after])):
        raise ValueError("V22 rollout block displacement endpoints are invalid")
    peak_force = float(trace.get("valid_push_side_peak_normal_force_n", 0.0))
    impulse = float(trace.get("valid_push_side_normal_impulse_discrete_ns", 0.0))
    if not np.isfinite(peak_force) or peak_force < 0.0:
        raise RuntimeError("V22 rollout valid-contact peak force is invalid")
    if not np.isfinite(impulse) or impulse < 0.0:
        raise RuntimeError("V22 rollout valid-contact impulse is invalid")
    geometric_valid_any = bool(np.any(valid_counts > 0))
    observed_valid_any = bool(trace.get("valid_push_side_contact_observed_any", False))
    valid_any = bool(trace.get("valid_push_side_contact_any", False))
    invalid_any = bool(np.any(invalid_counts > 0))
    if observed_valid_any and not geometric_valid_any:
        raise RuntimeError("V22 valid-contact impulse lacks a geometric substep")
    if valid_any != bool(observed_valid_any and not invalid_any):
        raise RuntimeError("V22 admissible-contact summary is inconsistent")
    if invalid_any != bool(trace.get("invalid_tool_block_contact_any", False)):
        raise RuntimeError("V22 invalid-contact summary disagrees with substeps")
    return {
        "format": STOCK_GRIPPER_CONTACT_TELEMETRY_FORMAT_V22,
        "physics_substeps": physics_substeps,
        "contact_candidate_geom_count": len(contact_ids),
        "safety_only_geom_count": len(safety_only_columns),
        "tool_block_contact_any": bool(np.any(raw_counts > 0)),
        "tool_block_contact_substep_count": int(np.count_nonzero(raw_counts > 0)),
        "valid_push_side_contact_any": valid_any,
        "valid_push_side_contact_substep_count": int(np.count_nonzero(valid_counts > 0) if valid_any else 0),
        "valid_push_side_contact_transient_only": bool(valid_any and int(valid_counts[-1]) == 0),
        "invalid_tool_block_contact_any": invalid_any,
        "contact_role_names": contact_roles,
        "raw_tool_block_contact_count_by_role": tuple(
            int(value) for value in np.sum(raw_counts_by_role, axis=0)
        ),
        "valid_push_side_contact_count_by_role": tuple(
            int(value) for value in np.sum(valid_counts_by_role, axis=0)
        ),
        "invalid_tool_block_contact_count_by_role": tuple(
            int(value) for value in np.sum(invalid_counts_by_role, axis=0)
        ),
        "invalid_tool_block_contact_substep_count_by_role": tuple(
            int(value) for value in np.count_nonzero(invalid_counts_by_role > 0, axis=0)
        ),
        "valid_push_side_peak_normal_force_n": peak_force,
        "valid_push_side_normal_impulse_discrete_ns": impulse,
        "minimum_executed_safety_only_block_clearance_m": float(np.min(distances[:, safety_only_columns])),
        "step_block_displacement_m": float(np.linalg.norm(after - before)),
    }


__all__ = [
    "STOCK_GRIPPER_POLICY_PROJECTION_RECOVERY_REASON_V26",
    "STOCK_GRIPPER_TRANSIENT_DIRECTION_RECOVERY_REASON_V26",
    "STOCK_GRIPPER_CONTACT_TELEMETRY_FORMAT_V22",
    "STOCK_GRIPPER_TASKFRAME_RESET_RETRY_FORMAT_V22",
    "STOCK_GRIPPER_TASKFRAME_RUNTIME_FORMAT_V22",
    "StockGripperTaskFrameAdapterV22",
    "StockTaskFrameResetInfeasibleV22",
    "reset_stock_taskframe_episode_v22",
    "transition_contact_telemetry_v22",
]
