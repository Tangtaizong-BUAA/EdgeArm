"""Exact-execution stock-gripper safety guard for V10 scratch RL.

This module leaves the source-bound V15/V9 guard unchanged.  It ports the
same 96-part collision contract to the joint-bounded V10 plant, forecasts the
exact float32 command that will be submitted, and proves an eight-decision
braking tail even when the transport delay queue is empty.

The guard is simulator privileged.  It is a data-generation safety shield,
not a deployable perception module and not an expert trajectory source.
"""

from __future__ import annotations

from collections import deque
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import numpy as np

from .all_stage_action_guard_v1 import AllStageActionGuardV1
from .delayed_servo_acquisition_forecast_v1 import (
    DelayedServoAcquisitionForecastConfigV1,
    DelayedServoAcquisitionForecastV1,
)
from .joint_path_planner_v1 import JointPathPlannerConfig
from .sim2real_env_v10 import (
    RealisticEdgeArmEnvV10,
    RealisticEnvV10Config,
    V10SafetyFilterInfeasible,
)


STOCK_GRIPPER_ACTION_GUARD_FORMAT_V2 = (
    "edgearm-v10-stock-gripper-float32-eight-hold-braking-guard-v2"
)
STOCK_GRIPPER_FORECAST_FORMAT_V2 = (
    "edgearm-v10-stock-gripper-exact-float32-servo-forecast-v2"
)
_JOINTS = 6
_HOLD_TOLERANCE = 1.0e-8


@dataclass(frozen=True)
class StockGripperActionGuardConfigV2:
    """Frozen initial scratch-RL safety curriculum."""

    hard_executed_safety_only_clearance_m: float = (
        JointPathPlannerConfig().minimum_tool_block_clearance_m
    )
    planning_safety_only_margin_m: float = 0.00035
    candidate_scales: tuple[float, ...] = (1.0, 0.75, 0.5, 0.25, 0.125, 0.0)
    braking_hold_steps: int = 8

    def __post_init__(self) -> None:
        hard = float(self.hard_executed_safety_only_clearance_m)
        margin = float(self.planning_safety_only_margin_m)
        if not np.isfinite(hard) or hard <= 0.0:
            raise ValueError("hard executed clearance must be finite and positive")
        if not np.isfinite(margin) or margin < hard:
            raise ValueError("planning margin must be finite and no smaller than hard clearance")
        if type(self.braking_hold_steps) is not int or self.braking_hold_steps < 1:
            raise ValueError("braking_hold_steps must be a positive integer")
        scales = tuple(float(value) for value in self.candidate_scales)
        if (
            len(scales) < 2
            or not np.all(np.isfinite(scales))
            or not np.isclose(scales[0], 1.0, rtol=0.0, atol=1.0e-12)
            or not np.isclose(scales[-1], 0.0, rtol=0.0, atol=1.0e-12)
            or any(not 0.0 <= value <= 1.0 for value in scales)
            or any(left <= right for left, right in zip(scales, scales[1:]))
        ):
            raise ValueError("candidate_scales must strictly descend from one to zero")


class StockGripperDelayedServoForecastV2(DelayedServoAcquisitionForecastV1):
    """V10-bound forecast reusing the audited V1 trace semantics."""

    def __init__(
        self,
        env: RealisticEdgeArmEnvV10,
        config: DelayedServoAcquisitionForecastConfigV1 | None = None,
    ) -> None:
        if type(env) is not RealisticEdgeArmEnvV10:
            raise TypeError("stock-gripper forecast V2 requires exact RealisticEdgeArmEnvV10")
        if config is not None and type(config) is not DelayedServoAcquisitionForecastConfigV1:
            raise TypeError("stock-gripper forecast V2 requires exact forecast config")
        self.env = env
        self.config = config or DelayedServoAcquisitionForecastConfigV1()
        safety = tuple(int(value) for value in env._ids["tool_safety_geoms"])
        contacts = tuple(int(value) for value in env._ids["tool_contact_geoms"])
        roles = tuple(str(value) for value in env._ids["tool_contact_geom_roles"])
        if (
            len(safety) != 96
            or len(contacts) != 2
            or len(set(contacts)) != 2
            or not set(contacts).issubset(safety)
            or roles != ("fixed_tip", "moving_tip")
        ):
            raise RuntimeError("stock-gripper forecast V2 requires the exact 96/2 contract")
        self._safety_geoms = safety
        self._contact_geoms = contacts
        self._contact_roles = roles
        self._contact_columns = np.asarray(
            [safety.index(geom_id) for geom_id in contacts],
            dtype=np.int64,
        )
        self._safety_only_geoms = tuple(
            geom_id for geom_id in safety if geom_id not in contacts
        )
        if len(self._safety_only_geoms) != 94:
            raise RuntimeError("stock-gripper forecast V2 requires 94 safety-only parts")
        self._safety_only_columns = np.asarray(
            [safety.index(geom_id) for geom_id in self._safety_only_geoms],
            dtype=np.int64,
        )
        self._penetration_tolerance_m = float(
            env.contact_feasible_config.reset_penetration_tolerance_m
        )

    def forecast(self, candidate_action: np.ndarray) -> dict[str, Any]:
        report = super().forecast(candidate_action)
        report["format"] = STOCK_GRIPPER_FORECAST_FORMAT_V2
        report["environment"] = "exact RealisticEdgeArmEnvV10"
        report["float32_execution_action_required"] = True
        return report


class StockGripperActionGuardV2(AllStageActionGuardV1):
    """Select the largest exact-float32 action with a braking proof."""

    def __init__(
        self,
        env: RealisticEdgeArmEnvV10,
        config: StockGripperActionGuardConfigV2 | None = None,
    ) -> None:
        if type(env) is not RealisticEdgeArmEnvV10:
            raise TypeError("stock-gripper action guard V2 requires exact RealisticEdgeArmEnvV10")
        if config is not None and type(config) is not StockGripperActionGuardConfigV2:
            raise TypeError("stock-gripper action guard V2 requires exact guard config")
        if type(env.config) is not RealisticEnvV10Config:
            raise TypeError("stock-gripper action guard V2 requires exact RealisticEnvV10Config")
        self.env = env
        self.config = config or StockGripperActionGuardConfigV2()
        if (
            float(env.config.command_loss_probability) > 0.0
            or float(env.config.command_burst_start_probability) > 0.0
        ):
            raise RuntimeError("guard V2 initial curriculum requires deterministic transport")
        self.forecast = StockGripperDelayedServoForecastV2(
            env,
            DelayedServoAcquisitionForecastConfigV1(
                minimum_safety_only_block_clearance_m=(
                    self.config.hard_executed_safety_only_clearance_m
                )
            ),
        )

    @staticmethod
    def _execution_action(value: np.ndarray) -> np.ndarray:
        """Return the exact float64 representation of submitted float32 bytes."""

        action = np.asarray(value, dtype=np.float64)
        if action.shape != (_JOINTS,) or not np.all(np.isfinite(action)):
            raise ValueError("proposed action must be a finite six-vector")
        clipped = np.clip(action, -1.0, 1.0)
        return clipped.astype(np.float32).astype(np.float64)

    def forecast_modified_action_with_hold_tail(
        self,
        first_action: np.ndarray,
    ) -> dict[str, Any]:
        """Forecast the candidate, delayed queue, and fixed braking horizon."""

        env = self.env
        first = self._execution_action(first_action)
        queue_depth = len(env._command_queue)
        application_count = 2 * queue_depth + 1 + self.config.braking_hold_steps
        snapshot = env._snapshot_terminal_viability_state()
        counters = self.forecast._counter_snapshot(env)
        v6_rng_state = deepcopy(env._v6_rng.bit_generator.state)
        original_step_count = int(env.step_count)
        reports: list[dict[str, Any]] = []
        try:
            env._command_queue = deque(deepcopy(list(env._command_queue)))
            for offset in range(application_count):
                env.step_count = original_step_count + offset
                submitted_action = (
                    first
                    if offset == 0
                    else np.zeros(_JOINTS, dtype=np.float64)
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
                delayed_target, execution_reason = env._safety_filter(
                    np.asarray(applied, dtype=np.float64)
                )
                start = env.data.qpos[:_JOINTS].copy()
                endpoint, physical_velocity, runtime_reason, _, _ = env._servo_step(
                    delayed_target
                )
                env._advance_physics(
                    start,
                    endpoint,
                    physical_velocity,
                    delayed_target,
                )
                env._last_actual_velocity = (
                    env.data.qpos[:_JOINTS] - start
                ) / max(float(env.control_dt), 1.0e-9)
                env._refresh_encoder_noise()
                trace = env._physics_substep_contact_v1
                if trace is None:  # pragma: no cover
                    raise RuntimeError("guard V2 braking forecast produced no trace")
                audit = self.forecast.audit_trace(trace)
                reports.append(
                    {
                        "application_index": offset,
                        "submitted_command_id": int(submitted.command_id),
                        "submitted_action_linf": float(
                            np.max(np.abs(submitted_action))
                        ),
                        "applied_command_id": int(applied.command_id),
                        "applied_is_modified_command": bool(
                            int(applied.command_id) == original_step_count
                        ),
                        "minimum_94_safety_only_block_distance_m": float(
                            audit["minimum_94_safety_only_block_distance_m"]
                        ),
                        "unauthorized_contact_part_penetration_any": bool(
                            audit["unauthorized_contact_part_penetration_any"]
                        ),
                        "hard_valid": bool(audit["valid"]),
                        "planning_margin_valid": bool(
                            audit["valid"]
                            and float(
                                audit[
                                    "minimum_94_safety_only_block_distance_m"
                                ]
                            )
                            >= self.config.planning_safety_only_margin_m - 1.0e-12
                        ),
                        "preflight_reason": str(preflight_reason),
                        "execution_filter_reason": str(execution_reason),
                        "runtime_filter_reason": str(runtime_reason),
                    }
                )
        finally:
            env.step_count = original_step_count
            env._restore_terminal_viability_state(snapshot)
            env._v6_rng.bit_generator.state = deepcopy(v6_rng_state)
            self.forecast._restore_counters(env, counters)

        modified = [row for row in reports if row["applied_is_modified_command"]]
        if len(modified) != 1:
            raise RuntimeError("guard V2 braking forecast lost modified command identity")
        limiting = min(
            range(len(reports)),
            key=lambda index: float(
                reports[index]["minimum_94_safety_only_block_distance_m"]
            ),
        )
        minimum = float(
            reports[limiting]["minimum_94_safety_only_block_distance_m"]
        )
        return {
            "format": STOCK_GRIPPER_ACTION_GUARD_FORMAT_V2,
            "online_before_submission": True,
            "simulator_privileged": True,
            "live_state_restored": True,
            "queue_depth": queue_depth,
            "application_count": application_count,
            "braking_hold_steps": self.config.braking_hold_steps,
            "minimum_forecast_94_safety_only_block_distance_m": minimum,
            "hard_valid": bool(all(row["hard_valid"] for row in reports)),
            "planning_margin_valid": bool(
                all(row["planning_margin_valid"] for row in reports)
            ),
            "remaining_submissions_are_holds": bool(
                all(
                    float(row["submitted_action_linf"]) <= _HOLD_TOLERANCE
                    for row in reports[1:]
                )
            ),
            "applications": reports,
        }

    def select(
        self,
        proposed_action: np.ndarray,
        *,
        require_hold_tail: bool = True,
    ) -> tuple[np.ndarray | None, dict[str, Any]]:
        """Return an action whose forecast bytes equal the execution bytes."""

        proposed = self._execution_action(proposed_action)
        before = self.queue_identity()
        attempts: list[dict[str, Any]] = []
        selected_action: np.ndarray | None = None
        selected_forecast: dict[str, Any] = {}
        selected_hold_tail: dict[str, Any] = {}
        selected_scale: float | None = None
        attempted_actions: set[bytes] = set()
        for scale in self.config.candidate_scales:
            candidate = self._execution_action(proposed * float(scale))
            key = candidate.astype(np.float32).tobytes()
            if key in attempted_actions:
                continue
            attempted_actions.add(key)
            try:
                forecast = self.forecast.forecast(candidate)
            except V10SafetyFilterInfeasible as error:
                if self.queue_identity() != before:
                    raise RuntimeError(
                        "guard V2 infeasible one-step forecast changed the live queue"
                    ) from error
                attempts.append(
                    {
                        "scale": float(scale),
                        "candidate_action_float32_hex": (
                            candidate.astype(np.float32).tobytes().hex()
                        ),
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
                raise RuntimeError("guard V2 one-step forecast changed the live queue")
            planning_valid = bool(
                forecast["valid"]
                and not forecast["unauthorized_contact_part_penetration_any"]
                and all(
                    float(row["minimum_94_safety_only_block_distance_m"])
                    >= self.config.planning_safety_only_margin_m - 1.0e-12
                    for row in forecast["applications"]
                )
            )
            hold_tail: dict[str, Any] = {}
            if planning_valid and require_hold_tail:
                try:
                    hold_tail = self.forecast_modified_action_with_hold_tail(candidate)
                except V10SafetyFilterInfeasible as error:
                    if self.queue_identity() != before:
                        raise RuntimeError(
                            "guard V2 infeasible braking forecast changed the live queue"
                        ) from error
                    planning_valid = False
                    hold_tail = {
                        "forecast_infeasible": True,
                        "forecast_error_type": type(error).__name__,
                        "forecast_error": str(error),
                    }
                if self.queue_identity() != before:
                    raise RuntimeError("guard V2 braking forecast changed the live queue")
                if not bool(hold_tail.get("forecast_infeasible", False)):
                    planning_valid = bool(
                        hold_tail["planning_margin_valid"]
                        and hold_tail["remaining_submissions_are_holds"]
                    )
            attempts.append(
                {
                    "scale": float(scale),
                    "candidate_action_float32_hex": (
                        candidate.astype(np.float32).tobytes().hex()
                    ),
                    "planning_margin_valid": planning_valid,
                    "minimum_one_step_clearance_m": float(
                        forecast[
                            "minimum_forecast_94_safety_only_block_distance_m"
                        ]
                    ),
                    "minimum_braking_clearance_m": (
                        None
                        if not hold_tail
                        or bool(hold_tail.get("forecast_infeasible", False))
                        else float(
                            hold_tail[
                                "minimum_forecast_94_safety_only_block_distance_m"
                            ]
                        )
                    ),
                    "forecast_infeasible": bool(
                        hold_tail.get("forecast_infeasible", False)
                    ),
                    "forecast_error_type": str(
                        hold_tail.get("forecast_error_type", "none")
                    ),
                    "forecast_error": str(hold_tail.get("forecast_error", "none")),
                }
            )
            if planning_valid:
                selected_action = candidate
                selected_forecast = forecast
                selected_hold_tail = hold_tail
                selected_scale = float(scale)
                break

        if self.queue_identity() != before:
            raise RuntimeError("guard V2 changed an already-submitted queue")
        return_action = (
            None
            if selected_action is None
            else selected_action.astype(np.float32)
        )
        if return_action is not None and not np.array_equal(
            return_action.astype(np.float64), selected_action
        ):
            raise RuntimeError("guard V2 forecast/execution float32 identity failed")
        report = {
            "format": STOCK_GRIPPER_ACTION_GUARD_FORMAT_V2,
            "online_before_submission": True,
            "simulator_privileged": True,
            "expert_actions": 0,
            "expert_paths": 0,
            "float32_forecast_execution_identity": True,
            "braking_hold_steps": self.config.braking_hold_steps,
            "safe_candidate_found": return_action is not None,
            "selected_scale": selected_scale,
            "selected_forecast": selected_forecast,
            "selected_hold_tail_forecast": selected_hold_tail,
            "attempts": attempts,
            "queue_identity_before": before,
            "queue_identity_after": self.queue_identity(),
        }
        return return_action, report


__all__ = [
    "STOCK_GRIPPER_ACTION_GUARD_FORMAT_V2",
    "STOCK_GRIPPER_FORECAST_FORMAT_V2",
    "StockGripperActionGuardConfigV2",
    "StockGripperActionGuardV2",
    "StockGripperDelayedServoForecastV2",
]
