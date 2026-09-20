"""Latched-absolute-target safety guard for stock-gripper scratch RL.

V2 scaled every candidate toward the literal zero action and described that
endpoint as a hold.  In the V10 transport, however, an action is relative to
the current encoder reading.  Encoder noise therefore makes literal zero
resubmit a different physical joint target at every decision.  V3 scales
toward an explicit float32-reconstructable absolute target and proves a tail
that keeps resubmitting latched absolute targets.

The collision authority is unchanged: all 94 non-contact stock-gripper parts
must retain the planning margin, while the two distal contact parts use the
same-role contact authorization contract.  The guard remains simulator
privileged and is not a deployable perception component.
"""

from __future__ import annotations

from collections import deque
from copy import deepcopy
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from .sim2real_env_v10 import RealisticEdgeArmEnvV10, V10SafetyFilterInfeasible
from .stock_gripper_action_guard_v2 import (
    StockGripperActionGuardConfigV2,
    StockGripperActionGuardV2,
)


STOCK_GRIPPER_ACTION_GUARD_FORMAT_V3 = (
    "edgearm-v10-stock-gripper-float32-latched-absolute-target-braking-guard-v3"
)
_JOINTS = 6
_TARGET_TOLERANCE_RAD = 1.0e-8


class LatchedAbsoluteTargetInfeasibleV3(RuntimeError):
    """Raised when one absolute hold target cannot be encoded by one action."""


@dataclass(frozen=True)
class StockGripperActionGuardConfigV3(StockGripperActionGuardConfigV2):
    """V2 collision margins plus exact absolute-target identity tolerance."""

    absolute_target_tolerance_rad: float = _TARGET_TOLERANCE_RAD

    def __post_init__(self) -> None:
        super().__post_init__()
        tolerance = float(self.absolute_target_tolerance_rad)
        if not np.isfinite(tolerance) or not 0.0 < tolerance <= 1.0e-6:
            raise ValueError("absolute_target_tolerance_rad must be finite in (0, 1e-6]")


class StockGripperActionGuardV3(StockGripperActionGuardV2):
    """Select between a proposal and a latched absolute-target baseline."""

    guard_format = STOCK_GRIPPER_ACTION_GUARD_FORMAT_V3

    def __init__(
        self,
        env: RealisticEdgeArmEnvV10,
        config: StockGripperActionGuardConfigV3 | None = None,
    ) -> None:
        selected = config or StockGripperActionGuardConfigV3()
        if type(selected) is not StockGripperActionGuardConfigV3:
            raise TypeError("stock-gripper action guard V3 requires exact V3 config")
        base = StockGripperActionGuardConfigV2(
            hard_executed_safety_only_clearance_m=(selected.hard_executed_safety_only_clearance_m),
            planning_safety_only_margin_m=selected.planning_safety_only_margin_m,
            candidate_scales=selected.candidate_scales,
            braking_hold_steps=selected.braking_hold_steps,
        )
        super().__init__(env, base)
        self.config = selected

    def action_for_absolute_target(
        self,
        target_joint_position_rad: np.ndarray,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Encode one fixed physical target using the live observable reference."""

        target = np.asarray(target_joint_position_rad, dtype=np.float64)
        if target.shape != (_JOINTS,) or not np.all(np.isfinite(target)):
            raise ValueError("absolute joint target must be a finite six-vector")
        env = self.env
        reference = np.asarray(env._command_reference_reported_position(), dtype=np.float64) - np.asarray(
            env._zero_offset, dtype=np.float64
        )
        raw = (target - reference) / float(env.config.max_joint_delta)
        tolerance = float(self.config.absolute_target_tolerance_rad)
        if float(np.max(np.abs(raw))) > 1.0 + tolerance:
            raise LatchedAbsoluteTargetInfeasibleV3("absolute target lies outside one normalized command")
        action = self._execution_action(raw).astype(np.float32)
        reconstructed = reference + action.astype(np.float64) * float(env.config.max_joint_delta)
        filtered, filter_reason = env._safety_filter(reconstructed)
        error = float(np.max(np.abs(np.asarray(filtered) - target)))
        if error > tolerance:
            raise LatchedAbsoluteTargetInfeasibleV3(
                "absolute target changed during float32 reconstruction or filtering"
            )
        return action, {
            "format": self.guard_format,
            "observable_reference_rad": reference.tolist(),
            "requested_absolute_target_rad": target.tolist(),
            "reconstructed_absolute_target_rad": reconstructed.tolist(),
            "filtered_absolute_target_rad": np.asarray(filtered).tolist(),
            "maximum_absolute_target_error_rad": error,
            "absolute_target_tolerance_rad": tolerance,
            "filter_reason": str(filter_reason),
            "float32_action_hex": action.tobytes().hex(),
        }

    def absolute_target_for_action(self, action: np.ndarray) -> np.ndarray:
        """Reconstruct the physical target that V10 will form before filtering."""

        execution = self._execution_action(action)
        env = self.env
        reference = np.asarray(env._command_reference_reported_position(), dtype=np.float64) - np.asarray(
            env._zero_offset, dtype=np.float64
        )
        return reference + execution * float(env.config.max_joint_delta)

    def forecast_modified_action_with_latched_tail(
        self,
        first_action: np.ndarray,
        *,
        baseline_absolute_target_rad: np.ndarray,
        preserve_baseline_target: bool = False,
        recovery_anchor_rad: np.ndarray | None = None,
    ) -> dict[str, Any]:
        """Forecast a candidate plus fixed-target or recovery-policy tail.

        Until the candidate reaches the head of a delayed queue, subsequent
        commands keep targeting the supplied baseline.  Once the candidate is
        physically applied, its realized physics endpoint is latched and every
        later tail submission reconstructs that same absolute target from the
        then-current noisy encoder reference.  Recovery commands may instead
        use a receding-horizon controller toward a fixed workspace-interior
        anchor; every command produced by that backup policy is still run
        through the exact plant and contact audit before acceptance.
        """

        env = self.env
        first = self._execution_action(first_action)
        baseline = np.asarray(baseline_absolute_target_rad, dtype=np.float64)
        if baseline.shape != (_JOINTS,) or not np.all(np.isfinite(baseline)):
            raise ValueError("braking baseline must be a finite six-vector")
        recovery_anchor: np.ndarray | None = None
        if recovery_anchor_rad is not None:
            recovery_anchor = np.asarray(
                recovery_anchor_rad, dtype=np.float64
            )
            if recovery_anchor.shape != (_JOINTS,) or not np.all(
                np.isfinite(recovery_anchor)
            ):
                raise ValueError(
                    "recovery anchor must be a finite six-vector"
                )
            if preserve_baseline_target:
                raise ValueError(
                    "recovery-policy tail cannot preserve a fixed baseline"
                )
        queue_depth = len(env._command_queue)
        application_count = 2 * queue_depth + 1 + int(self.config.braking_hold_steps)
        snapshot = env._snapshot_terminal_viability_state()
        counters = self.forecast._counter_snapshot(env)
        v6_rng_state = deepcopy(env._v6_rng.bit_generator.state)
        original_step_count = int(env.step_count)
        candidate_command_id = original_step_count
        latched_endpoint: np.ndarray | None = None
        tail_hold_target: np.ndarray | None = (
            baseline.copy() if preserve_baseline_target else None
        )
        reports: list[dict[str, Any]] = []
        try:
            env._command_queue = deque(deepcopy(list(env._command_queue)))
            for offset in range(application_count):
                env.step_count = original_step_count + offset
                if offset == 0:
                    submitted_action = first
                    submitted_target = self.absolute_target_for_action(first)
                    target_semantics = "candidate"
                elif recovery_anchor is not None:
                    reference = np.asarray(
                        env._command_reference_reported_position(),
                        dtype=np.float64,
                    ) - np.asarray(env._zero_offset, dtype=np.float64)
                    raw_recovery = (recovery_anchor - reference) / float(
                        env.config.max_joint_delta
                    )
                    submitted_action = self._execution_action(
                        np.clip(raw_recovery, -1.0, 1.0)
                    ).astype(np.float32)
                    submitted_target = self.absolute_target_for_action(
                        submitted_action
                    )
                    target_semantics = (
                        "receding_horizon_workspace_anchor_recovery"
                    )
                else:
                    submitted_target = (
                        baseline
                        if tail_hold_target is None
                        else tail_hold_target
                    )
                    submitted_action, _identity = self.action_for_absolute_target(submitted_target)
                    target_semantics = (
                        "baseline_latched_hold"
                        if tail_hold_target is None or preserve_baseline_target
                        else "candidate_endpoint_latched_hold"
                    )
                submitted = self._simulated_command(
                    submitted_action,
                    command_id=original_step_count + offset,
                    send_step=original_step_count + offset,
                )
                env._command_queue.append(submitted)
                applied = env._command_queue.popleft()
                env._runtime_pusher_desk_safety_stop_requested = False
                env._controller_preflight_v1 = {}
                preflight_reason = env._preflight_applied_delayed_command(applied)
                delayed_target, execution_reason = env._safety_filter(np.asarray(applied, dtype=np.float64))
                start = env.data.qpos[:_JOINTS].copy()
                endpoint, physical_velocity, runtime_reason, _, _ = env._servo_step(delayed_target)
                env._advance_physics(
                    start,
                    endpoint,
                    physical_velocity,
                    delayed_target,
                )
                env._last_actual_velocity = (env.data.qpos[:_JOINTS] - start) / max(
                    float(env.control_dt), 1.0e-9
                )
                candidate_applied = bool(int(applied.command_id) == candidate_command_id)
                if candidate_applied:
                    latched_endpoint = np.asarray(env.data.qpos[:_JOINTS], dtype=np.float64).copy()
                    if (
                        recovery_anchor is None
                        and not preserve_baseline_target
                    ):
                        tail_hold_target = latched_endpoint.copy()
                env._refresh_encoder_noise()
                trace = env._physics_substep_contact_v1
                if trace is None:  # pragma: no cover
                    raise RuntimeError("guard V3 braking forecast produced no trace")
                audit = self.forecast.audit_trace(trace)
                block_joint_id = int(env._ids["block_joint"])
                block_qpos_address = int(
                    env.model.jnt_qposadr[block_joint_id]
                )
                block_dof_address = int(
                    env.model.jnt_dofadr[block_joint_id]
                )
                application = {
                    "application_index": offset,
                    "submitted_command_id": int(submitted.command_id),
                    "submitted_action_linf": float(np.max(np.abs(submitted_action))),
                    "submitted_absolute_target_rad": np.asarray(submitted_target, dtype=np.float64).tolist(),
                    "submitted_target_semantics": target_semantics,
                    "applied_command_id": int(applied.command_id),
                    "applied_is_modified_command": candidate_applied,
                    "physics_endpoint_rad": np.asarray(env.data.qpos[:_JOINTS], dtype=np.float64).tolist(),
                    "block_xy_m": np.asarray(env.block_xy(), dtype=np.float64).tolist(),
                    "block_pose_xyz_quaternion_wxyz": np.asarray(
                        env.data.qpos[
                            block_qpos_address : block_qpos_address + 7
                        ],
                        dtype=np.float64,
                    ).tolist(),
                    "block_velocity_linear_angular": np.asarray(
                        env.data.qvel[
                            block_dof_address : block_dof_address + 6
                        ],
                        dtype=np.float64,
                    ).tolist(),
                    "block_linear_angular_speed": [
                        float(value) for value in env._block_speeds()
                    ],
                    "simulation_time_seconds": float(env.data.time),
                    "safety_only_geom_count": int(audit["safety_only_geom_count"]),
                    "minimum_safety_only_block_distance_m": float(
                        audit["minimum_safety_only_block_distance_m"]
                    ),
                    "unauthorized_contact_part_penetration_any": bool(
                        audit["unauthorized_contact_part_penetration_any"]
                    ),
                    "hard_valid": bool(audit["valid"]),
                    "planning_margin_valid": bool(
                        audit["valid"]
                        and float(audit["minimum_safety_only_block_distance_m"])
                        >= self.config.planning_safety_only_margin_m - 1.0e-12
                    ),
                    "preflight_reason": str(preflight_reason),
                    "execution_filter_reason": str(execution_reason),
                    "runtime_filter_reason": str(runtime_reason),
                }
                # V4 extends the base audit with the exact semantic-contact
                # reason that can make an otherwise well-separated command
                # invalid.  Preserve those fields in every recursive tail
                # application so a shield terminal can be attributed to the
                # concrete role and physics substep instead of only exposing
                # a generic ``hard_valid=False`` result.
                for audit_field in (
                    "strict_zero_invalid_contact_required",
                    "invalid_tool_block_contact_count",
                    "invalid_tool_block_contact_substeps",
                    "first_invalid_tool_block_contact",
                ):
                    if audit_field in audit:
                        application[audit_field] = deepcopy(audit[audit_field])
                if int(application["safety_only_geom_count"]) == 94:
                    application["minimum_94_safety_only_block_distance_m"] = application[
                        "minimum_safety_only_block_distance_m"
                    ]
                reports.append(application)
        finally:
            env.step_count = original_step_count
            env._restore_terminal_viability_state(snapshot)
            env._v6_rng.bit_generator.state = deepcopy(v6_rng_state)
            self.forecast._restore_counters(env, counters)

        modified = [row for row in reports if row["applied_is_modified_command"]]
        if len(modified) != 1:
            raise RuntimeError("guard V3 braking forecast lost candidate identity")
        if latched_endpoint is None:  # pragma: no cover
            raise RuntimeError("guard V3 braking forecast never latched the candidate")
        candidate_index = int(modified[0]["application_index"])
        tail = reports[candidate_index + 1 :]
        required_semantics = (
            "receding_horizon_workspace_anchor_recovery"
            if recovery_anchor is not None
            else (
                "baseline_latched_hold"
                if preserve_baseline_target
                else "candidate_endpoint_latched_hold"
            )
        )
        endpoint_tail = [
            row
            for row in tail
            if row["submitted_target_semantics"] == required_semantics
        ]
        required = int(self.config.braking_hold_steps)
        identity_target = (
            recovery_anchor
            if recovery_anchor is not None
            else (baseline if preserve_baseline_target else latched_endpoint)
        )
        recovery_policy_identity = bool(
            recovery_anchor is not None
            and len(endpoint_tail) >= required
            and all(
                row["submitted_target_semantics"] == required_semantics
                for row in endpoint_tail
            )
        )
        endpoint_target_identity = bool(
            recovery_anchor is None
            and len(endpoint_tail) >= required
            and all(
                np.max(
                    np.abs(
                        np.asarray(
                            row["submitted_absolute_target_rad"],
                            dtype=np.float64,
                        )
                        - identity_target
                    )
                )
                <= self.config.absolute_target_tolerance_rad
                for row in endpoint_tail
            )
        )
        limiting = min(
            range(len(reports)),
            key=lambda index: float(reports[index]["minimum_safety_only_block_distance_m"]),
        )
        minimum = float(reports[limiting]["minimum_safety_only_block_distance_m"])
        report = {
            "format": self.guard_format,
            "online_before_submission": True,
            "simulator_privileged": True,
            "live_state_restored": True,
            "queue_depth": queue_depth,
            "application_count": application_count,
            "braking_hold_steps": required,
            "candidate_effect_application_index": candidate_index,
            "baseline_absolute_target_rad": baseline.tolist(),
            "latched_candidate_endpoint_rad": latched_endpoint.tolist(),
            "tail_hold_absolute_target_rad": identity_target.tolist(),
            "tail_target_strategy": (
                "receding_horizon_workspace_anchor_recovery"
                if recovery_anchor is not None
                else (
                    "preserve_baseline_absolute_target"
                    if preserve_baseline_target
                    else "latch_candidate_physics_endpoint"
                )
            ),
            "baseline_hold_target_preserved": bool(
                preserve_baseline_target
            ),
            "safety_only_geom_count": len(self.forecast._safety_only_geoms),
            "minimum_forecast_safety_only_block_distance_m": minimum,
            "hard_valid": bool(all(row["hard_valid"] for row in reports)),
            "planning_margin_valid": bool(all(row["planning_margin_valid"] for row in reports)),
            "remaining_submissions_are_holds": endpoint_target_identity,
            "remaining_submissions_target_latched_holds": endpoint_target_identity,
            "remaining_submissions_follow_verified_backup_policy": (
                recovery_policy_identity
            ),
            "literal_zero_action_hold_semantics": False,
            "applications": reports,
        }
        if len(self.forecast._safety_only_geoms) == 94:
            report["minimum_forecast_94_safety_only_block_distance_m"] = minimum
        return report

    def select(
        self,
        proposed_action: np.ndarray,
        *,
        baseline_action: np.ndarray,
        require_hold_tail: bool = True,
        preserve_baseline_target: bool = False,
        recovery_tail_anchor_rad: np.ndarray | None = None,
    ) -> tuple[np.ndarray | None, dict[str, Any]]:
        """Return the safest largest interpolation from baseline to proposal.

        Ordinary actions must admit a fixed-target braking tail.  A privileged
        recovery action may instead prove a receding-horizon backup controller
        toward ``recovery_tail_anchor_rad``.  Both paths use the same exact
        plant/contact forecast and planning margin.
        """

        proposed = self._execution_action(proposed_action)
        baseline = self._execution_action(baseline_action)
        baseline_target = self.absolute_target_for_action(baseline)
        before = self.queue_identity()
        attempts: list[dict[str, Any]] = []
        selected_action: np.ndarray | None = None
        selected_forecast: dict[str, Any] = {}
        selected_hold_tail: dict[str, Any] = {}
        selected_scale: float | None = None
        selected_is_baseline_hold = False
        attempted_actions: set[bytes] = set()
        for scale in self.config.candidate_scales:
            candidate = self._execution_action(baseline + float(scale) * (proposed - baseline))
            key = candidate.astype(np.float32).tobytes()
            if key in attempted_actions:
                continue
            attempted_actions.add(key)
            candidate_is_baseline_hold = bool(
                np.array_equal(candidate, baseline)
            )
            try:
                forecast = self.forecast.forecast(candidate)
            except V10SafetyFilterInfeasible as error:
                if self.queue_identity() != before:
                    raise RuntimeError("guard V3 infeasible forecast changed the live queue") from error
                attempts.append(
                    {
                        "scale": float(scale),
                        "candidate_action_float32_hex": key.hex(),
                        "planning_margin_valid": False,
                        "minimum_one_step_clearance_m": None,
                        "minimum_braking_clearance_m": None,
                        "forecast_infeasible": True,
                        "forecast_error_type": type(error).__name__,
                        "forecast_error": str(error),
                    }
                )
                continue
            if self.queue_identity() != before:
                raise RuntimeError("guard V3 one-step forecast changed the live queue")
            planning_valid = bool(
                forecast["valid"]
                and not forecast["unauthorized_contact_part_penetration_any"]
                and all(
                    float(row["minimum_safety_only_block_distance_m"])
                    >= self.config.planning_safety_only_margin_m - 1.0e-12
                    for row in forecast["applications"]
                )
            )
            hold_tail: dict[str, Any] = {}
            if planning_valid and require_hold_tail:
                try:
                    hold_tail = self.forecast_modified_action_with_latched_tail(
                        candidate,
                        baseline_absolute_target_rad=baseline_target,
                        preserve_baseline_target=bool(
                            preserve_baseline_target
                            and candidate_is_baseline_hold
                        ),
                        recovery_anchor_rad=recovery_tail_anchor_rad,
                    )
                except (V10SafetyFilterInfeasible, LatchedAbsoluteTargetInfeasibleV3) as error:
                    if self.queue_identity() != before:
                        raise RuntimeError("guard V3 infeasible tail changed the live queue") from error
                    planning_valid = False
                    hold_tail = {
                        "forecast_infeasible": True,
                        "forecast_error_type": type(error).__name__,
                        "forecast_error": str(error),
                    }
                if self.queue_identity() != before:
                    raise RuntimeError("guard V3 braking forecast changed the live queue")
                if not bool(hold_tail.get("forecast_infeasible", False)):
                    continuation_identity = bool(
                        hold_tail[
                            "remaining_submissions_target_latched_holds"
                        ]
                        or hold_tail.get(
                            "remaining_submissions_follow_verified_backup_policy",
                            False,
                        )
                    )
                    planning_valid = bool(
                        hold_tail["planning_margin_valid"]
                        and continuation_identity
                    )
            limiting_application = min(
                forecast["applications"],
                key=lambda row: float(
                    row["minimum_safety_only_block_distance_m"]
                ),
            )
            limiting_trace_audit = dict(
                limiting_application.get("trace_audit") or {}
            )
            hold_tail_applications = (
                []
                if not hold_tail
                or bool(hold_tail.get("forecast_infeasible", False))
                else list(hold_tail.get("applications", []))
            )
            hold_tail_invalid_applications = [
                row
                for row in hold_tail_applications
                if not bool(row.get("hard_valid", False))
            ]
            hold_tail_failure_summary = {
                "application_count": len(hold_tail_applications),
                "hard_invalid_application_count": sum(
                    not bool(row.get("hard_valid", False))
                    for row in hold_tail_applications
                ),
                "planning_margin_invalid_application_count": sum(
                    not bool(row.get("planning_margin_valid", False))
                    for row in hold_tail_applications
                ),
                "unauthorized_contact_part_penetration_application_count": sum(
                    bool(
                        row.get(
                            "unauthorized_contact_part_penetration_any", False
                        )
                    )
                    for row in hold_tail_applications
                ),
                "invalid_tool_block_contact_application_count": sum(
                    int(row.get("invalid_tool_block_contact_count", 0)) > 0
                    for row in hold_tail_applications
                ),
                "continuation_identity_valid": bool(
                    hold_tail
                    and (
                        hold_tail.get(
                            "remaining_submissions_target_latched_holds", False
                        )
                        or hold_tail.get(
                            "remaining_submissions_follow_verified_backup_policy",
                            False,
                        )
                    )
                ),
                "first_hard_invalid_application": (
                    {}
                    if not hold_tail_invalid_applications
                    else {
                        key: deepcopy(
                            hold_tail_invalid_applications[0].get(key)
                        )
                        for key in (
                            "application_index",
                            "submitted_command_id",
                            "submitted_target_semantics",
                            "applied_command_id",
                            "applied_is_modified_command",
                            "preflight_reason",
                            "execution_filter_reason",
                            "runtime_filter_reason",
                            "minimum_safety_only_block_distance_m",
                            "unauthorized_contact_part_penetration_any",
                            "invalid_tool_block_contact_count",
                            "first_invalid_tool_block_contact",
                        )
                    }
                ),
            }
            attempts.append(
                {
                    "scale": float(scale),
                    "candidate_action_float32_hex": key.hex(),
                    "candidate_delta_from_baseline_linf": float(np.max(np.abs(candidate - baseline))),
                    "candidate_is_baseline_hold": candidate_is_baseline_hold,
                    "planning_margin_valid": planning_valid,
                    "minimum_one_step_clearance_m": float(
                        forecast["minimum_forecast_safety_only_block_distance_m"]
                    ),
                    "minimum_one_step_clearance_location": deepcopy(
                        forecast.get(
                            "minimum_forecast_safety_only_block_distance_location"
                        )
                    ),
                    "one_step_failure_reasons": deepcopy(
                        forecast.get("failure_reasons", [])
                    ),
                    "one_step_limiting_application": {
                        key: deepcopy(limiting_application.get(key))
                        for key in (
                            "applied_command_id",
                            "forecast_apply_step",
                            "physics_endpoint_rad",
                            "block_xy_m",
                            "block_pose_xyz_quaternion_wxyz",
                            "block_velocity_linear_angular",
                            "block_linear_angular_speed",
                            "simulation_time_seconds",
                            "minimum_safety_only_block_distance_m",
                            "minimum_safety_only_block_distance_location",
                            "unauthorized_contact_part_penetration_any",
                            "first_unauthorized_contact_part_penetration",
                        )
                    }
                    | {
                        "strict_zero_invalid_contact_required": (
                            limiting_trace_audit.get(
                                "strict_zero_invalid_contact_required"
                            )
                        ),
                        "invalid_tool_block_contact_count": (
                            limiting_trace_audit.get(
                                "invalid_tool_block_contact_count"
                            )
                        ),
                        "first_invalid_tool_block_contact": deepcopy(
                            limiting_trace_audit.get(
                                "first_invalid_tool_block_contact"
                            )
                        ),
                    },
                    "minimum_braking_clearance_m": (
                        None
                        if not hold_tail or bool(hold_tail.get("forecast_infeasible", False))
                        else float(hold_tail["minimum_forecast_safety_only_block_distance_m"])
                    ),
                    "forecast_infeasible": bool(hold_tail.get("forecast_infeasible", False)),
                    "forecast_error_type": str(hold_tail.get("forecast_error_type", "none")),
                    "forecast_error": str(hold_tail.get("forecast_error", "none")),
                    "hold_tail_failure_summary": hold_tail_failure_summary,
                }
            )
            if planning_valid:
                selected_action = candidate
                selected_forecast = forecast
                selected_hold_tail = hold_tail
                selected_scale = float(scale)
                selected_is_baseline_hold = candidate_is_baseline_hold
                break

        if self.queue_identity() != before:
            raise RuntimeError("guard V3 changed an already-submitted queue")
        return_action = None if selected_action is None else selected_action.astype(np.float32)
        if return_action is not None and not np.array_equal(
            return_action.astype(np.float64), selected_action
        ):
            raise RuntimeError("guard V3 forecast/execution float32 identity failed")
        report = {
            "format": self.guard_format,
            "configuration": asdict(self.config),
            "online_before_submission": True,
            "simulator_privileged": True,
            "expert_actions": 0,
            "expert_paths": 0,
            "float32_forecast_execution_identity": True,
            "candidate_scaling_origin": "latched_absolute_target_baseline",
            "literal_zero_action_hold_semantics": False,
            "braking_hold_steps": self.config.braking_hold_steps,
            "safe_candidate_found": return_action is not None,
            "selected_scale": selected_scale,
            "selected_is_baseline_hold": selected_is_baseline_hold,
            "preserve_baseline_target_requested": bool(
                preserve_baseline_target
            ),
            "recovery_tail_anchor_requested": bool(
                recovery_tail_anchor_rad is not None
            ),
            "baseline_action_float32_hex": baseline.astype(np.float32).tobytes().hex(),
            "baseline_absolute_target_rad": baseline_target.tolist(),
            "selected_forecast": selected_forecast,
            "selected_hold_tail_forecast": selected_hold_tail,
            "attempts": attempts,
            "queue_identity_before": before,
            "queue_identity_after": self.queue_identity(),
        }
        return return_action, report


__all__ = [
    "LatchedAbsoluteTargetInfeasibleV3",
    "STOCK_GRIPPER_ACTION_GUARD_FORMAT_V3",
    "StockGripperActionGuardConfigV3",
    "StockGripperActionGuardV3",
]
