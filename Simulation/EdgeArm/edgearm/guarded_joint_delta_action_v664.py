"""Guarded five-joint delta action layer for the stock SO-ARM follower.

The previous RL teacher exposed only task-frame XYZ and delegated every local
pose to one constrained IK branch.  Fixed-seed V661 evidence showed that this
interface can stall even under privileged Cartesian goal feedback.  V664
keeps the stock robot, fixed push-gripper setpoint, V10 plant, V22 contact
semantics, and V4 online guard, but lets an RL policy choose bounded deltas for
the five arm joints.  This permits learned non-linear acquisition paths while
the unchanged guard remains the final execution authority.

V664 is an action interface, not an expert, planner, waypoint generator, or
demonstration source.  Safety intervention outcomes remain ordinary RL
environment transitions and must be represented honestly in replay.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from .asymmetric_multiview_ppo_v1 import (
    SOURCE_TYPE,
    StockTaskFrameNoSafeRecoveryV13,
)
from .stock_gripper_action_guard_v3 import LatchedAbsoluteTargetInfeasibleV3
from .stock_gripper_taskframe_v22 import (
    StockGripperTaskFrameAdapterV22,
)


GUARDED_JOINT_DELTA_ACTION_FORMAT_V664 = (
    "edgearm-v664-guarded-five-joint-delta-action-v3"
)
CURRENT_PHYSICAL_JOINT_REBASE_FORMAT_V792 = (
    "edgearm-v792-guard-verified-current-physical-joint-rebase-v1"
)
ARM_JOINT_ACTION_DIM_V664 = 5


class GuardedJointNoSafeActionV664(RuntimeError):
    """Raised before env.step when no V4-verified command exists."""


@dataclass(frozen=True)
class GuardedJointDeltaActionConfigV664:
    """Normalized actor support mapped to a bounded joint target increment."""

    maximum_joint_target_step_rad: float = 0.025

    def validate(self) -> None:
        value = float(self.maximum_joint_target_step_rad)
        if not np.isfinite(value) or not 0.0 < value <= 0.05:
            raise ValueError("V664 maximum joint target step is invalid")


@dataclass(frozen=True)
class GuardedJointActionResultV664:
    requested_joint_action: np.ndarray
    applied_joint_action: np.ndarray
    submitted_joint_action: np.ndarray
    latched_joint_target_before_rad: np.ndarray
    requested_joint_target_rad: np.ndarray
    selected_physics_endpoint_rad: np.ndarray
    guard_safe_candidate: bool
    guard_selected_scale: float
    guard_intervened: bool
    selected_is_hold: bool
    minimum_one_step_clearance_m: float
    minimum_braking_clearance_m: float
    guard_report: dict[str, Any]
    format: str = GUARDED_JOINT_DELTA_ACTION_FORMAT_V664

    def validate(self) -> None:
        arrays = {
            "requested_joint_action": (
                self.requested_joint_action,
                (ARM_JOINT_ACTION_DIM_V664,),
            ),
            "applied_joint_action": (
                self.applied_joint_action,
                (ARM_JOINT_ACTION_DIM_V664,),
            ),
            "submitted_joint_action": (self.submitted_joint_action, (6,)),
            "latched_joint_target_before_rad": (
                self.latched_joint_target_before_rad,
                (6,),
            ),
            "requested_joint_target_rad": (
                self.requested_joint_target_rad,
                (6,),
            ),
            "selected_physics_endpoint_rad": (
                self.selected_physics_endpoint_rad,
                (6,),
            ),
        }
        for name, (value, shape) in arrays.items():
            array = np.asarray(value)
            if array.shape != shape or not np.all(np.isfinite(array)):
                raise ValueError(f"V664 {name} is invalid")
        if (
            np.any(np.abs(self.requested_joint_action) > 1.0 + 1.0e-6)
            or np.any(np.abs(self.applied_joint_action) > 1.0 + 1.0e-6)
            or np.any(np.abs(self.submitted_joint_action) > 1.0 + 1.0e-6)
        ):
            raise ValueError("V664 normalized action escaped bounds")
        if self.guard_safe_candidate is not True:
            raise ValueError("V664 result lacks a safe guard candidate")
        if (
            not np.isfinite(self.guard_selected_scale)
            or not 0.0 <= self.guard_selected_scale <= 1.0
        ):
            raise ValueError("V664 selected guard scale is invalid")
        for name in (
            "minimum_one_step_clearance_m",
            "minimum_braking_clearance_m",
        ):
            if not np.isfinite(float(getattr(self, name))):
                raise ValueError(f"V664 {name} is invalid")
        if type(self.guard_report) is not dict:
            raise TypeError("V664 guard report must be a dictionary")


@dataclass(frozen=True)
class GuardedJointActionPreflightV672:
    """Non-mutating exact V4 forecast for one policy action candidate."""

    requested_joint_action: np.ndarray
    latched_joint_target_rad: np.ndarray
    requested_joint_target_rad: np.ndarray
    selected_submitted_action: np.ndarray | None
    target_encoding: dict[str, Any]
    baseline_identity: dict[str, Any]
    guard_report: dict[str, Any]
    format: str = "edgearm-v672-guarded-joint-candidate-preflight-v1"

    @property
    def safe_candidate_found(self) -> bool:
        return self.selected_submitted_action is not None


def requested_joint_target_v664(
    latched_joint_target_rad: np.ndarray,
    normalized_arm_action: np.ndarray,
    *,
    maximum_joint_target_step_rad: float,
    joint_ranges_rad: np.ndarray,
    fixed_gripper_joint_position_rad: float,
) -> np.ndarray:
    """Map a five-vector actor action to one clipped six-joint target."""

    latched = np.asarray(latched_joint_target_rad, dtype=np.float64)
    action = np.asarray(normalized_arm_action, dtype=np.float64)
    ranges = np.asarray(joint_ranges_rad, dtype=np.float64)
    step = float(maximum_joint_target_step_rad)
    gripper = float(fixed_gripper_joint_position_rad)
    if (
        latched.shape != (6,)
        or action.shape != (ARM_JOINT_ACTION_DIM_V664,)
        or ranges.shape != (6, 2)
        or not np.all(np.isfinite(np.r_[latched, action, ranges.reshape(-1), step, gripper]))
        or step <= 0.0
        or step > 0.05
        or np.any(ranges[:, 0] >= ranges[:, 1])
        or not ranges[5, 0] <= gripper <= ranges[5, 1]
    ):
        raise ValueError("V664 joint target inputs are invalid")
    bounded_action = np.clip(action, -1.0, 1.0)
    result = latched.copy()
    result[:ARM_JOINT_ACTION_DIM_V664] += bounded_action * step
    result[:ARM_JOINT_ACTION_DIM_V664] = np.clip(
        result[:ARM_JOINT_ACTION_DIM_V664],
        ranges[:ARM_JOINT_ACTION_DIM_V664, 0],
        ranges[:ARM_JOINT_ACTION_DIM_V664, 1],
    )
    result[5] = gripper
    return result


def applied_joint_action_v664(
    latched_joint_target_before_rad: np.ndarray,
    selected_physics_endpoint_rad: np.ndarray,
    *,
    maximum_joint_target_step_rad: float,
) -> np.ndarray:
    """Recover the normalized physical endpoint effect seen by the critic."""

    before = np.asarray(latched_joint_target_before_rad, dtype=np.float64)
    after = np.asarray(selected_physics_endpoint_rad, dtype=np.float64)
    step = float(maximum_joint_target_step_rad)
    if (
        before.shape != (6,)
        or after.shape != (6,)
        or not np.all(np.isfinite(np.r_[before, after, step]))
        or step <= 0.0
        or step > 0.05
    ):
        raise ValueError("V664 applied-action inputs are invalid")
    return np.clip(
        (after[:ARM_JOINT_ACTION_DIM_V664] - before[:ARM_JOINT_ACTION_DIM_V664])
        / step,
        -1.0,
        1.0,
    ).astype(np.float32)


def guard_intervened_v664(
    *,
    selected_scale: float,
    selected_is_baseline_hold: bool,
    raw_command_clipped: bool,
    execution_filter_reason: str,
) -> bool:
    """Separate a safety-command rewrite from ordinary plant lag.

    The guard approves a command before the force-limited plant evolves.  A
    physical endpoint that has not yet reached that command is useful applied
    action evidence, but it is not a guard intervention.  Only guard scaling,
    fallback to the proven baseline hold, command-support clipping, or V10
    safety-filter rewriting counts here.
    """

    scale = float(selected_scale)
    if (
        not np.isfinite(scale)
        or not 0.0 <= scale <= 1.0
        or type(selected_is_baseline_hold) is not bool
        or type(raw_command_clipped) is not bool
        or type(execution_filter_reason) is not str
    ):
        raise ValueError("V664 guard intervention inputs are invalid")
    return bool(
        scale < 1.0 - 1.0e-12
        or selected_is_baseline_hold
        or raw_command_clipped
        or execution_filter_reason not in {"", "none"}
    )


class GuardedJointDeltaActionV664:
    """Translate five-joint RL actions through the exact V4 execution guard."""

    def __init__(
        self,
        adapter: StockGripperTaskFrameAdapterV22,
        config: GuardedJointDeltaActionConfigV664 | None = None,
    ) -> None:
        if not isinstance(adapter, StockGripperTaskFrameAdapterV22):
            raise TypeError("V664 requires a V22-compatible guarded adapter")
        selected = config or GuardedJointDeltaActionConfigV664()
        if type(selected) is not GuardedJointDeltaActionConfigV664:
            raise TypeError("V664 requires the exact V664 config")
        selected.validate()
        self.adapter = adapter
        self.env = adapter.env
        self.config = selected
        self.last_guard_report: dict[str, Any] = {}
        # V784 closes a liveness gap between the V22 recovery proof and live
        # execution.  A V22 proof certifies repeated receding commands toward
        # one fixed intermediate anchor.  Recomputing a different anchor on
        # the next control decision executes a policy that was never covered
        # by that proof.  Keep the certified anchor until an ordinary guarded
        # action succeeds, while re-running the unchanged V4 proof every step.
        self._recovery_guard_identity_v784: int | None = (
            None if adapter.guard is None else id(adapter.guard)
        )
        self._active_recovery_tail_anchor_rad_v784: np.ndarray | None = None
        self._active_recovery_source_index_v784: int | None = None
        self._recovery_sequence_count_v784 = 0
        self._persisted_recovery_continuation_count_v784 = 0
        if selected.maximum_joint_target_step_rad > min(
            float(adapter.config.maximum_joint_target_delta_rad),
            float(self.env.config.max_joint_delta),
        ):
            raise ValueError("V664 joint step exceeds deployed command support")

    def reset_recovery_tail_state_v784(self) -> None:
        """Clear episode-local recovery state without changing the guard."""

        self._recovery_guard_identity_v784 = (
            None if self.adapter.guard is None else id(self.adapter.guard)
        )
        self._active_recovery_tail_anchor_rad_v784 = None
        self._active_recovery_source_index_v784 = None
        self._recovery_sequence_count_v784 = 0
        self._persisted_recovery_continuation_count_v784 = 0

    def _synchronize_recovery_episode_v784(self) -> None:
        """Prevent a recovery anchor from leaking across V22 episodes."""

        current_identity = (
            None if self.adapter.guard is None else id(self.adapter.guard)
        )
        if current_identity != self._recovery_guard_identity_v784:
            self.reset_recovery_tail_state_v784()

    def _clear_active_recovery_tail_v784(self) -> None:
        self._active_recovery_tail_anchor_rad_v784 = None
        self._active_recovery_source_index_v784 = None

    def runtime_audit(self) -> dict[str, Any]:
        self._synchronize_recovery_episode_v784()
        return {
            "format": GUARDED_JOINT_DELTA_ACTION_FORMAT_V664,
            "configuration": asdict(self.config),
            "policy_action_dimension": ARM_JOINT_ACTION_DIM_V664,
            "controlled_joints": [0, 1, 2, 3, 4],
            "fixed_gripper_joint_index": 5,
            "fixed_gripper_joint_position_rad": float(
                self.env.tool_gripper_joint_position_rad
            ),
            "reset_adapter_type": type(self.adapter).__name__,
            "action_semantics": (
                "bounded_delta_from_guard_latched_physical_joint_endpoint"
            ),
            "same_v10_plant": True,
            "same_v22_contact_semantics": True,
            "same_v4_online_guard": True,
            "automatic_cartesian_route_or_waypoint": False,
            "automatic_ik_target": False,
            "policy_must_learn_non_linear_acquisition_path": True,
            "orientation_is_learned_not_scripted": True,
            "persistent_verified_recovery_tail_v784": True,
            "recovery_tail_state_v784": {
                "active": self._active_recovery_tail_anchor_rad_v784 is not None,
                "source_recovery_index": self._active_recovery_source_index_v784,
                "recovery_sequence_count": self._recovery_sequence_count_v784,
                "persisted_continuation_count": (
                    self._persisted_recovery_continuation_count_v784
                ),
                "same_anchor_reverified_each_live_step": True,
                "unsafe_persisted_anchor_fails_closed": True,
            },
            "expert_action_used": False,
            "behavior_cloning_steps": 0,
            "simulator_privileged_guard": True,
            "physical_deployment_requires_guard_replacement_or_validation": True,
            "source_type": SOURCE_TYPE,
            "production_admission": False,
        }

    def preflight(
        self, policy_action: np.ndarray
    ) -> GuardedJointActionPreflightV672:
        """Forecast one candidate with V4 while preserving all live state."""

        self._synchronize_recovery_episode_v784()
        requested = np.asarray(policy_action, dtype=np.float64)
        if (
            requested.shape != (ARM_JOINT_ACTION_DIM_V664,)
            or not np.all(np.isfinite(requested))
        ):
            raise ValueError("V664 policy action must be a finite five-vector")
        requested = np.clip(requested, -1.0, 1.0)
        adapter = self.adapter
        guard = adapter.guard
        latched_value = adapter._latched_joint_target
        if guard is None or latched_value is None:
            raise RuntimeError("V664 requires an active V22 guarded episode")
        latched = np.asarray(latched_value, dtype=np.float64).copy()
        try:
            baseline_action, baseline_identity = (
                guard.action_for_absolute_target(latched)
            )
        except LatchedAbsoluteTargetInfeasibleV3 as error:
            raise GuardedJointNoSafeActionV664(
                "V664 latched baseline is not representable"
            ) from error
        target = requested_joint_target_v664(
            latched,
            requested,
            maximum_joint_target_step_rad=(
                self.config.maximum_joint_target_step_rad
            ),
            joint_ranges_rad=self.env.joint_ranges,
            fixed_gripper_joint_position_rad=(
                self.env.tool_gripper_joint_position_rad
            ),
        )
        reference = np.asarray(
            self.env._command_reference_reported_position(), dtype=np.float64
        ) - np.asarray(self.env._zero_offset, dtype=np.float64)
        raw_action = (target - reference) / float(self.env.config.max_joint_delta)
        proposed_action = np.clip(raw_action, -1.0, 1.0).astype(np.float32)
        reconstructed = guard.absolute_target_for_action(proposed_action)
        filtered_target, filter_reason = self.env._safety_filter(reconstructed)
        target_encoding = {
            "mode": "bounded_float32_joint_delta_projection",
            "latched_target_rad": latched.tolist(),
            "requested_target_rad": target.tolist(),
            "observable_reference_rad": reference.tolist(),
            "raw_command_action": raw_action.tolist(),
            "command_action_clipped": bool(np.any(np.abs(raw_action) > 1.0)),
            "float32_command_action_hex": proposed_action.tobytes().hex(),
            "reconstructed_target_rad": reconstructed.tolist(),
            "filtered_target_rad": np.asarray(
                filtered_target, dtype=np.float64
            ).tolist(),
            "execution_filter_reason": str(filter_reason),
        }
        selected, guard_report = guard.select(
            proposed_action,
            baseline_action=baseline_action,
            require_hold_tail=True,
        )
        return GuardedJointActionPreflightV672(
            requested_joint_action=requested.astype(np.float32),
            latched_joint_target_rad=latched,
            requested_joint_target_rad=target,
            selected_submitted_action=(
                None
                if selected is None
                else np.asarray(selected, dtype=np.float32).copy()
            ),
            target_encoding=deepcopy(target_encoding),
            baseline_identity=deepcopy(baseline_identity),
            guard_report=deepcopy(guard_report),
        )

    def rebase_latched_target_to_current_v792(
        self,
    ) -> GuardedJointActionResultV664:
        """Safely replace a stale latch with the current physical joint pose.

        A zero task-space request is not a physical hold: the DLS layer may
        still correct orientation or joint margin, and a zero joint delta
        retains the previous absolute latch.  V792 is an explicit diagnostic
        control-state transaction.  It first asks the unchanged V4 guard to
        prove the current measured arm pose as both the candidate and the
        preserved braking-tail target.  Only a complete, non-mutating proof
        permits the live latch to change.
        """

        self._synchronize_recovery_episode_v784()
        adapter = self.adapter
        guard = adapter.guard
        old_latched_value = adapter._latched_joint_target
        if guard is None or old_latched_value is None:
            raise RuntimeError("V792 rebase requires an active V22 guarded episode")
        old_latched = np.asarray(old_latched_value, dtype=np.float64).copy()
        current_measured = np.asarray(
            self.env.data.qpos[:6], dtype=np.float64
        ).copy()
        if (
            old_latched.shape != (6,)
            or current_measured.shape != (6,)
            or not np.all(np.isfinite(np.r_[old_latched, current_measured]))
        ):
            raise RuntimeError("V792 rebase joint state is malformed")
        current_target = current_measured.copy()
        current_target[5] = float(self.env.tool_gripper_joint_position_rad)
        ranges = np.asarray(self.env.joint_ranges, dtype=np.float64)
        tolerance = float(guard.config.absolute_target_tolerance_rad)
        if (
            ranges.shape != (6, 2)
            or not np.all(np.isfinite(ranges))
            or np.any(current_target < ranges[:, 0] - tolerance)
            or np.any(current_target > ranges[:, 1] + tolerance)
        ):
            raise GuardedJointNoSafeActionV664(
                "V792 current physical target lies outside joint support"
            )
        try:
            rebase_action, target_identity = guard.action_for_absolute_target(
                current_target
            )
        except LatchedAbsoluteTargetInfeasibleV3 as error:
            raise GuardedJointNoSafeActionV664(
                "V792 current physical target is not exactly representable"
            ) from error
        selected, guard_report = guard.select(
            rebase_action,
            baseline_action=rebase_action,
            require_hold_tail=True,
            preserve_baseline_target=True,
        )
        if selected is None:
            failed = deepcopy(guard_report)
            failed.update(
                {
                    "format": CURRENT_PHYSICAL_JOINT_REBASE_FORMAT_V792,
                    "v792_current_physical_joint_rebase": True,
                    "v792_old_latched_target_rad": old_latched.tolist(),
                    "v792_current_measured_joint_position_rad": (
                        current_measured.tolist()
                    ),
                    "v792_requested_rebase_target_rad": current_target.tolist(),
                    "v792_target_identity": deepcopy(target_identity),
                    "v792_latch_mutated": False,
                    "failed_closed": True,
                    "replay_admitted": False,
                    "production_admission": False,
                }
            )
            self.last_guard_report = deepcopy(failed)
            adapter.last_guard_report = deepcopy(failed)
            raise GuardedJointNoSafeActionV664(
                "V792 current physical target lacks a V4-safe hold tail"
            )
        selected_action = np.asarray(selected, dtype=np.float32)
        if not np.array_equal(selected_action, rebase_action):
            raise RuntimeError("V792 rebase guard changed its baseline command")
        selected_forecast = dict(guard_report.get("selected_forecast", {}))
        candidate_rows = [
            row
            for row in selected_forecast.get("applications", ())
            if bool(row.get("candidate_command", False))
        ]
        hold_tail = dict(
            guard_report.get("selected_hold_tail_forecast", {})
        )
        if len(candidate_rows) != 1:
            raise RuntimeError("V792 rebase lost forecast candidate identity")
        physics_endpoint = np.asarray(
            candidate_rows[0].get("physics_endpoint_rad"), dtype=np.float64
        )
        proof_complete = bool(
            physics_endpoint.shape == (6,)
            and np.all(np.isfinite(physics_endpoint))
            and guard_report.get("safe_candidate_found") is True
            and guard_report.get("selected_is_baseline_hold") is True
            and guard_report.get("preserve_baseline_target_requested") is True
            and bool(selected_forecast.get("valid", False))
            and not bool(
                selected_forecast.get(
                    "unauthorized_contact_part_penetration_any", True
                )
            )
            and hold_tail.get("hard_valid") is True
            and hold_tail.get("planning_margin_valid") is True
            and hold_tail.get(
                "remaining_submissions_target_latched_holds"
            )
            is True
        )
        if not proof_complete:
            raise RuntimeError("V792 rebase V4 proof is incomplete")
        one_step = float(
            selected_forecast[
                "minimum_forecast_safety_only_block_distance_m"
            ]
        )
        braking = float(
            hold_tail["minimum_forecast_safety_only_block_distance_m"]
        )
        if not np.all(np.isfinite((one_step, braking))):
            raise RuntimeError("V792 rebase clearance proof is invalid")

        # This is the only live mutation, and it occurs after the complete V4
        # proof above.  Preserve the requested current pose rather than the
        # first forecast endpoint: the tail was explicitly proved against this
        # absolute baseline target while delayed commands drain.
        adapter._latched_joint_target = current_target.copy()
        self._clear_active_recovery_tail_v784()
        applied = applied_joint_action_v664(
            old_latched,
            physics_endpoint,
            maximum_joint_target_step_rad=(
                self.config.maximum_joint_target_step_rad
            ),
        )
        audited = deepcopy(guard_report)
        audited.update(
            {
                "format": CURRENT_PHYSICAL_JOINT_REBASE_FORMAT_V792,
                "v792_current_physical_joint_rebase": True,
                "v792_old_latched_target_rad": old_latched.tolist(),
                "v792_current_measured_joint_position_rad": (
                    current_measured.tolist()
                ),
                "v792_requested_rebase_target_rad": current_target.tolist(),
                "v792_selected_first_physics_endpoint_rad": (
                    physics_endpoint.tolist()
                ),
                "v792_target_identity": deepcopy(target_identity),
                "v792_latch_mutated": True,
                "v792_latched_target_after_rad": current_target.tolist(),
                "v792_same_v4_guard": True,
                "v792_guard_verified_before_latch_mutation": True,
                "v792_policy_action_used": False,
                "v792_simulator_privileged_diagnostic_control": True,
                "replay_admitted": False,
                "production_admission": False,
            }
        )
        self.last_guard_report = deepcopy(audited)
        adapter.last_guard_report = deepcopy(audited)
        result = GuardedJointActionResultV664(
            requested_joint_action=np.zeros(5, dtype=np.float32),
            applied_joint_action=applied,
            submitted_joint_action=selected_action.copy(),
            latched_joint_target_before_rad=old_latched,
            requested_joint_target_rad=current_target.copy(),
            selected_physics_endpoint_rad=physics_endpoint,
            guard_safe_candidate=True,
            guard_selected_scale=float(guard_report["selected_scale"]),
            guard_intervened=True,
            selected_is_hold=True,
            minimum_one_step_clearance_m=one_step,
            minimum_braking_clearance_m=braking,
            guard_report=audited,
        )
        result.validate()
        return result

    def _continue_persisted_recovery_tail_v784(
        self,
        current_joint: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, dict[str, Any], dict[str, Any]]:
        """Re-verify and execute the exact recovery policy certified earlier.

        The active anchor is immutable for the recovery sequence.  The
        float32 command is reconstructed from the live encoder reference, so
        this is still a receding-horizon controller, but its policy identity
        matches the tail that V4 proved on the preceding decision.
        """

        adapter = self.adapter
        guard = adapter.guard
        anchor_value = self._active_recovery_tail_anchor_rad_v784
        if guard is None or anchor_value is None:
            raise RuntimeError("V784 persisted recovery requires an active guard and anchor")
        anchor = np.asarray(anchor_value, dtype=np.float64).copy()
        live = np.asarray(current_joint, dtype=np.float64).copy()
        if (
            anchor.shape != (6,)
            or live.shape != (6,)
            or not np.all(np.isfinite(np.r_[anchor, live]))
        ):
            raise RuntimeError("V784 persisted recovery state is malformed")

        filtered, filter_reason = self.env._safety_filter(live)
        target_encoding_mode = "exact_absolute_target"
        exact_encoding_error = "none"
        try:
            candidate_action, identity = guard.action_for_absolute_target(anchor)
        except LatchedAbsoluteTargetInfeasibleV3 as error:
            exact_encoding_error = str(error)
            reference = np.asarray(
                self.env._command_reference_reported_position(),
                dtype=np.float64,
            ) - np.asarray(self.env._zero_offset, dtype=np.float64)
            raw_action = (anchor - reference) / float(
                self.env.config.max_joint_delta
            )
            candidate_action = np.clip(raw_action, -1.0, 1.0).astype(
                np.float32
            )
            reconstructed = guard.absolute_target_for_action(candidate_action)
            executable_target, executable_filter_reason = (
                self.env._safety_filter(reconstructed)
            )
            target_encoding_mode = "filtered_float32_command_projection"
            identity = {
                "format": (
                    "edgearm-v784-persisted-recovery-float32-command-v1"
                ),
                "requested_absolute_target_rad": anchor.tolist(),
                "observable_reference_rad": reference.tolist(),
                "raw_action_before_clip": raw_action.tolist(),
                "action_clipped": bool(np.any(np.abs(raw_action) > 1.0)),
                "float32_action_hex": candidate_action.tobytes().hex(),
                "reconstructed_absolute_target_rad": reconstructed.tolist(),
                "executable_filtered_target_rad": np.asarray(
                    executable_target, dtype=np.float64
                ).tolist(),
                "execution_filter_reason": str(executable_filter_reason),
                "exact_absolute_target_encoding_error": exact_encoding_error,
            }

        baseline_action: np.ndarray | None = None
        baseline_error = "none"
        if adapter._latched_joint_target is not None:
            try:
                baseline_action, _baseline_identity = (
                    guard.action_for_absolute_target(
                        adapter._latched_joint_target
                    )
                )
            except LatchedAbsoluteTargetInfeasibleV3 as error:
                baseline_error = str(error)
        recovery_baseline = (
            candidate_action if baseline_action is None else baseline_action
        )
        selected, guard_report = guard.select(
            candidate_action,
            baseline_action=recovery_baseline,
            require_hold_tail=True,
            recovery_tail_anchor_rad=anchor,
        )
        if selected is None:
            failed_report = {
                "format": "edgearm-v784-persisted-recovery-tail-failure-v1",
                "live_joint_position_rad": live.tolist(),
                "filtered_boundary_target_rad": np.asarray(
                    filtered, dtype=np.float64
                ).tolist(),
                "filter_reason": str(filter_reason),
                "verified_recovery_tail_anchor_rad": anchor.tolist(),
                "source_recovery_index_v784": (
                    self._active_recovery_source_index_v784
                ),
                "target_identity": identity,
                "baseline_encoding_error": baseline_error,
                "guard_report": deepcopy(guard_report),
                "persisted_recovery_tail_anchor_v784": True,
                "same_anchor_reverified_each_live_step": True,
                "failed_closed": True,
                "production_admission": False,
            }
            adapter.last_recovery_report = deepcopy(failed_report)
            adapter.last_guard_report = deepcopy(guard_report)
            raise GuardedJointNoSafeActionV664(
                "V784 persisted recovery tail no longer has a V4-safe candidate"
            )

        selected_forecast = dict(guard_report["selected_forecast"])
        candidate_rows = [
            row
            for row in selected_forecast["applications"]
            if bool(row["candidate_command"])
        ]
        if len(candidate_rows) != 1:
            raise RuntimeError("V784 persisted recovery lost forecast identity")
        selected_target = np.asarray(
            candidate_rows[0]["physics_endpoint_rad"], dtype=np.float64
        )
        hold_tail = dict(guard_report["selected_hold_tail_forecast"])
        if not (
            selected_target.shape == (6,)
            and np.all(np.isfinite(selected_target))
            and bool(guard_report["safe_candidate_found"])
            and bool(hold_tail["hard_valid"])
            and bool(hold_tail["planning_margin_valid"])
            and bool(
                hold_tail[
                    "remaining_submissions_follow_verified_backup_policy"
                ]
            )
        ):
            raise RuntimeError("V784 persisted recovery proof is incomplete")

        adapter.recovery_count += 1
        self._recovery_sequence_count_v784 += 1
        self._persisted_recovery_continuation_count_v784 += 1
        recovery_report = {
            "format": "edgearm-v784-persisted-verified-recovery-tail-v1",
            "recovery_index": adapter.recovery_count,
            "live_joint_position_rad": live.tolist(),
            "filtered_boundary_target_rad": np.asarray(
                filtered, dtype=np.float64
            ).tolist(),
            "filter_reason": str(filter_reason),
            "requested_recovery_target_rad": anchor.tolist(),
            "verified_recovery_tail_anchor_rad": anchor.tolist(),
            "selected_guard_scale": float(guard_report["selected_scale"]),
            "selected_target_encoding_mode": target_encoding_mode,
            "selected_exact_absolute_target_encoding_error": (
                exact_encoding_error
            ),
            "selected_target_rad": selected_target.tolist(),
            "target_identity": identity,
            "minimum_one_step_clearance_m": float(
                selected_forecast[
                    "minimum_forecast_safety_only_block_distance_m"
                ]
            ),
            "minimum_braking_clearance_m": float(
                hold_tail["minimum_forecast_safety_only_block_distance_m"]
            ),
            "baseline_encoding_error": baseline_error,
            "persisted_recovery_tail_anchor_v784": True,
            "source_recovery_index_v784": (
                self._active_recovery_source_index_v784
            ),
            "recovery_sequence_index_v784": (
                self._recovery_sequence_count_v784
            ),
            "verified_continuation_policy": (
                "persistent_fixed_verified_intermediate_target_v784"
            ),
            "same_anchor_reverified_each_live_step": True,
            "same_guard_as_policy_action": True,
            "simulator_privileged": True,
            "production_admission": False,
        }
        adapter.last_guard_report = deepcopy(guard_report)
        adapter.last_recovery_report = deepcopy(recovery_report)
        return (
            selected_target,
            np.asarray(selected, dtype=np.float32),
            recovery_report,
            deepcopy(guard_report),
        )

    def recover_to_guard_verified_interior(
        self,
        policy_action: np.ndarray,
        *,
        primary_guard_failure: dict[str, Any] | None = None,
        requested_joint_target_rad: np.ndarray | None = None,
        target_encoding: dict[str, Any] | None = None,
        baseline_identity: dict[str, Any] | None = None,
    ) -> GuardedJointActionResultV664:
        """Execute V22's verified inward recovery without requiring a hold.

        ``preflight`` intentionally proves the latched baseline before it
        evaluates a policy proposal.  At a command-representation boundary the
        baseline itself can be unrepresentable, even though V22 can still
        produce an inward command with a complete V4 one-step and tail proof.
        This explicit entry point prevents callers from accidentally blocking
        that recovery behind an impossible hold precondition.
        """

        self._synchronize_recovery_episode_v784()
        requested = np.asarray(policy_action, dtype=np.float64)
        if (
            requested.shape != (ARM_JOINT_ACTION_DIM_V664,)
            or not np.all(np.isfinite(requested))
        ):
            raise ValueError("V664 policy action must be a finite five-vector")
        requested = np.clip(requested, -1.0, 1.0)
        adapter = self.adapter
        latched_value = adapter._latched_joint_target
        if latched_value is None:
            raise RuntimeError("V664 requires an active V22 guarded episode")
        latched = np.asarray(latched_value, dtype=np.float64).copy()
        if requested_joint_target_rad is None:
            target = requested_joint_target_v664(
                latched,
                requested,
                maximum_joint_target_step_rad=(
                    self.config.maximum_joint_target_step_rad
                ),
                joint_ranges_rad=self.env.joint_ranges,
                fixed_gripper_joint_position_rad=(
                    self.env.tool_gripper_joint_position_rad
                ),
            )
        else:
            target = np.asarray(
                requested_joint_target_rad, dtype=np.float64
            ).copy()
            if target.shape != (6,) or not np.all(np.isfinite(target)):
                raise ValueError("V664 recovery requested target is invalid")

        current_joint = np.asarray(
            self.env.data.qpos[:6], dtype=np.float64
        ).copy()
        recovery_anchor_reused_v784 = bool(
            self._active_recovery_tail_anchor_rad_v784 is not None
        )
        if recovery_anchor_reused_v784:
            (
                recovery_target,
                recovery_action,
                recovery_report,
                recovery_guard_report,
            ) = self._continue_persisted_recovery_tail_v784(current_joint)
        else:
            try:
                recovery_target, recovery_action = (
                    adapter._representable_interior_recovery_hold(current_joint)
                )
            except StockTaskFrameNoSafeRecoveryV13 as error:
                raise GuardedJointNoSafeActionV664(
                    "V664 policy, baseline hold, and V22 recovery have no "
                    "V4-safe candidate"
                ) from error
            recovery_target = np.asarray(recovery_target, dtype=np.float64)
            recovery_action = np.asarray(recovery_action, dtype=np.float32)
            recovery_report = deepcopy(adapter.last_recovery_report)
            recovery_guard_report = deepcopy(adapter.last_guard_report)
            recovery_anchor = np.asarray(
                recovery_report.get("verified_recovery_tail_anchor_rad"),
                dtype=np.float64,
            )
            if (
                recovery_anchor.shape != (6,)
                or not np.all(np.isfinite(recovery_anchor))
            ):
                raise RuntimeError("V664 V22 recovery anchor is malformed")
            self._active_recovery_tail_anchor_rad_v784 = (
                recovery_anchor.copy()
            )
            self._active_recovery_source_index_v784 = int(
                recovery_report["recovery_index"]
            )
            self._recovery_sequence_count_v784 += 1
        if (
            recovery_target.shape != (6,)
            or recovery_action.shape != (6,)
            or not np.all(np.isfinite(np.r_[recovery_target, recovery_action]))
            or not bool(recovery_guard_report.get("safe_candidate_found", False))
        ):
            raise RuntimeError("V664 V22 recovery proof is malformed")
        adapter._latched_joint_target = recovery_target.copy()
        recovery_applied = applied_joint_action_v664(
            latched,
            recovery_target,
            maximum_joint_target_step_rad=(
                self.config.maximum_joint_target_step_rad
            ),
        )
        recovery_one_step = float(
            recovery_report["minimum_one_step_clearance_m"]
        )
        recovery_braking = float(
            recovery_report["minimum_braking_clearance_m"]
        )
        recovery_scale = float(recovery_guard_report["selected_scale"])
        audited_recovery = deepcopy(recovery_guard_report)
        audited_recovery.update(
            {
                "v664_target_encoding": deepcopy(target_encoding or {}),
                "v664_baseline_identity": deepcopy(baseline_identity or {}),
                "v664_requested_actor_action": requested.tolist(),
                "v664_applied_actor_action": recovery_applied.tolist(),
                "v664_primary_guard_failure": deepcopy(
                    primary_guard_failure or {}
                ),
                "v664_v22_guarded_recovery": recovery_report,
                "v664_intervened": True,
                "v664_recovery_entered_without_hold_precondition": True,
                "v664_recovery_tail_anchor_rad": np.asarray(
                    self._active_recovery_tail_anchor_rad_v784,
                    dtype=np.float64,
                ).tolist(),
                "v664_recovery_tail_anchor_persisted_v784": True,
                "v664_recovery_tail_anchor_reused_v784": (
                    recovery_anchor_reused_v784
                ),
                "v664_recovery_sequence_index_v784": (
                    self._recovery_sequence_count_v784
                ),
                "production_admission": False,
            }
        )
        self.last_guard_report = deepcopy(audited_recovery)
        adapter.last_guard_report = deepcopy(audited_recovery)
        recovery_result = GuardedJointActionResultV664(
            requested_joint_action=requested.astype(np.float32),
            applied_joint_action=recovery_applied,
            submitted_joint_action=recovery_action,
            latched_joint_target_before_rad=latched,
            requested_joint_target_rad=target,
            selected_physics_endpoint_rad=recovery_target,
            guard_safe_candidate=True,
            guard_selected_scale=recovery_scale,
            guard_intervened=True,
            selected_is_hold=bool(
                np.max(np.abs(recovery_target - latched)) <= 1.0e-8
            ),
            minimum_one_step_clearance_m=recovery_one_step,
            minimum_braking_clearance_m=recovery_braking,
            guard_report=audited_recovery,
        )
        recovery_result.validate()
        return recovery_result

    def translate(
        self,
        policy_action: np.ndarray,
        *,
        verified_preflight: GuardedJointActionPreflightV672 | None = None,
    ) -> GuardedJointActionResultV664:
        self._synchronize_recovery_episode_v784()
        requested = np.asarray(policy_action, dtype=np.float64)
        if (
            requested.shape != (ARM_JOINT_ACTION_DIM_V664,)
            or not np.all(np.isfinite(requested))
        ):
            raise ValueError("V664 policy action must be a finite five-vector")
        requested = np.clip(requested, -1.0, 1.0)
        adapter = self.adapter
        current_latched = adapter._latched_joint_target
        if current_latched is None:
            raise RuntimeError("V664 requires an active V22 guarded episode")
        if verified_preflight is None:
            preflight = self.preflight(requested)
        else:
            preflight = verified_preflight
            if type(preflight) is not GuardedJointActionPreflightV672:
                raise TypeError("V664 verified preflight has the wrong type")
            if (
                not np.array_equal(
                    preflight.requested_joint_action.astype(np.float64),
                    requested.astype(np.float32).astype(np.float64),
                )
                or not np.array_equal(
                    preflight.latched_joint_target_rad,
                    np.asarray(current_latched, dtype=np.float64),
                )
            ):
                raise ValueError("V664 verified preflight is stale or mismatched")
        latched = preflight.latched_joint_target_rad.copy()
        target = preflight.requested_joint_target_rad.copy()
        selected = (
            None
            if preflight.selected_submitted_action is None
            else preflight.selected_submitted_action.copy()
        )
        guard_report = deepcopy(preflight.guard_report)
        target_encoding = deepcopy(preflight.target_encoding)
        baseline_identity = deepcopy(preflight.baseline_identity)
        raw_action = np.asarray(
            target_encoding["raw_command_action"], dtype=np.float64
        )
        filter_reason = str(target_encoding["execution_filter_reason"])
        self.last_guard_report = deepcopy(guard_report)
        adapter.last_guard_report = deepcopy(guard_report)
        if selected is None:
            return self.recover_to_guard_verified_interior(
                requested,
                primary_guard_failure=guard_report,
                requested_joint_target_rad=target,
                target_encoding=target_encoding,
                baseline_identity=baseline_identity,
            )
        selected_forecast = dict(guard_report["selected_forecast"])
        candidate_rows = [
            row
            for row in selected_forecast["applications"]
            if bool(row["candidate_command"])
        ]
        if len(candidate_rows) != 1:
            raise RuntimeError("V664 guard lost candidate forecast identity")
        physics_endpoint = np.asarray(
            candidate_rows[0]["physics_endpoint_rad"], dtype=np.float64
        )
        if physics_endpoint.shape != (6,) or not np.all(np.isfinite(physics_endpoint)):
            raise RuntimeError("V664 guard produced an invalid physics endpoint")
        if not np.isclose(
            float(physics_endpoint[5]),
            float(self.env.tool_gripper_joint_position_rad),
            rtol=0.0,
            atol=2.0e-3,
        ):
            raise RuntimeError("V664 guard changed the fixed gripper setpoint")
        adapter._latched_joint_target = physics_endpoint.copy()
        applied = applied_joint_action_v664(
            latched,
            physics_endpoint,
            maximum_joint_target_step_rad=(
                self.config.maximum_joint_target_step_rad
            ),
        )
        selected_scale = float(guard_report["selected_scale"])
        one_step = float(
            selected_forecast["minimum_forecast_safety_only_block_distance_m"]
        )
        braking = float(
            guard_report["selected_hold_tail_forecast"]
            ["minimum_forecast_safety_only_block_distance_m"]
        )
        selected_is_hold = bool(
            float(np.max(np.abs(physics_endpoint - latched))) <= 1.0e-8
        )
        intervention = guard_intervened_v664(
            selected_scale=selected_scale,
            selected_is_baseline_hold=bool(
                guard_report["selected_is_baseline_hold"]
            ),
            raw_command_clipped=bool(np.any(np.abs(raw_action) > 1.0)),
            execution_filter_reason=str(filter_reason),
        )
        active_recovery_anchor_before = (
            None
            if self._active_recovery_tail_anchor_rad_v784 is None
            else self._active_recovery_tail_anchor_rad_v784.tolist()
        )
        audited_report = deepcopy(guard_report)
        audited_report.update(
            {
                "v664_target_encoding": target_encoding,
                "v664_baseline_identity": baseline_identity,
                "v664_requested_actor_action": requested.tolist(),
                "v664_applied_actor_action": applied.tolist(),
                "v664_intervened": intervention,
                "v664_recovery_tail_anchor_active_before_normal_v784": (
                    active_recovery_anchor_before
                ),
                "v664_recovery_tail_anchor_cleared_after_normal_v784": bool(
                    active_recovery_anchor_before is not None
                ),
                "production_admission": False,
            }
        )
        result = GuardedJointActionResultV664(
            requested_joint_action=requested.astype(np.float32),
            applied_joint_action=applied,
            submitted_joint_action=np.asarray(selected, dtype=np.float32),
            latched_joint_target_before_rad=latched,
            requested_joint_target_rad=target,
            selected_physics_endpoint_rad=physics_endpoint,
            guard_safe_candidate=True,
            guard_selected_scale=selected_scale,
            guard_intervened=intervention,
            selected_is_hold=selected_is_hold,
            minimum_one_step_clearance_m=one_step,
            minimum_braking_clearance_m=braking,
            guard_report=audited_report,
        )
        result.validate()
        self._clear_active_recovery_tail_v784()
        return result


__all__ = [
    "ARM_JOINT_ACTION_DIM_V664",
    "CURRENT_PHYSICAL_JOINT_REBASE_FORMAT_V792",
    "GUARDED_JOINT_DELTA_ACTION_FORMAT_V664",
    "GuardedJointActionResultV664",
    "GuardedJointActionPreflightV672",
    "GuardedJointDeltaActionConfigV664",
    "GuardedJointDeltaActionV664",
    "GuardedJointNoSafeActionV664",
    "applied_joint_action_v664",
    "guard_intervened_v664",
    "requested_joint_target_v664",
]
