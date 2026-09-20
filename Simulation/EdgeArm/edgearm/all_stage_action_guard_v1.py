"""Online all-stage delayed-queue safety selection for Physical Expert V15.

Every candidate is appended behind an immutable snapshot of the real V9 delay
queue and evaluated with the real stateful servo plus V7 physics substeps.  A
candidate is admissible only when all 94 safety-only CoACD parts keep the V15
planning margin and both authored contact parts satisfy the exact Causal-V3
same-role contact authorization rule.  The executed hard limit remains the
unchanged 0.25 mm V14 contract.
"""

from __future__ import annotations

from collections import deque
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
from typing import Any

import numpy as np

from .delayed_servo_acquisition_forecast_v1 import (
    DelayedServoAcquisitionForecastConfigV1,
    DelayedServoAcquisitionForecastV1,
)
from .joint_path_planner_v1 import JointPathPlannerConfig
from .sim2real_env import _QueuedCommand
from .sim2real_env_v9 import RealisticEdgeArmEnvV9


ALL_STAGE_ACTION_GUARD_FORMAT_V1 = (
    "edgearm-v15-online-all-stage-delayed-queue-servo-96-part-guard-v1"
)
_JOINTS = 6
_HOLD_TOLERANCE = 1.0e-8


@dataclass(frozen=True)
class AllStageActionGuardConfigV1:
    hard_executed_safety_only_clearance_m: float = (
        JointPathPlannerConfig().minimum_tool_block_clearance_m
    )
    planning_safety_only_margin_m: float = 0.001
    candidate_scales: tuple[float, ...] = tuple(
        float(value) / 16.0 for value in range(16, -1, -1)
    )

    def __post_init__(self) -> None:
        hard = float(self.hard_executed_safety_only_clearance_m)
        margin = float(self.planning_safety_only_margin_m)
        if not np.isfinite(hard) or hard <= 0.0:
            raise ValueError("hard executed clearance must be finite and positive")
        if not np.isfinite(margin) or margin < hard:
            raise ValueError("planning margin must be finite and no smaller than hard clearance")
        scales = tuple(float(value) for value in self.candidate_scales)
        if (
            len(scales) < 2
            or not np.all(np.isfinite(scales))
            or not np.isclose(scales[0], 1.0, rtol=0.0, atol=1.0e-12)
            or not np.isclose(scales[-1], 0.0, rtol=0.0, atol=1.0e-12)
            or any(not 0.0 <= value <= 1.0 for value in scales)
            or any(left <= right for left, right in zip(scales, scales[1:]))
        ):
            raise ValueError(
                "candidate_scales must strictly descend from 1.0 to 0.0"
            )


class AllStageActionGuardV1:
    """Select the largest deterministic action scale with an online proof."""

    def __init__(
        self,
        env: RealisticEdgeArmEnvV9,
        config: AllStageActionGuardConfigV1 | None = None,
    ) -> None:
        if type(env) is not RealisticEdgeArmEnvV9:
            raise TypeError("AllStageActionGuardV1 requires exact RealisticEdgeArmEnvV9")
        if config is not None and type(config) is not AllStageActionGuardConfigV1:
            raise TypeError("guard requires exact AllStageActionGuardConfigV1")
        self.env = env
        self.config = config or AllStageActionGuardConfigV1()
        if (
            float(env.config.command_loss_probability) > 0.0
            or float(env.config.command_burst_start_probability) > 0.0
        ):
            raise RuntimeError(
                "V15 guard is fail-closed outside deterministic transport: "
                "loss and burst probabilities must both be zero"
            )
        self.forecast = DelayedServoAcquisitionForecastV1(
            env,
            DelayedServoAcquisitionForecastConfigV1(
                minimum_safety_only_block_clearance_m=(
                    self.config.hard_executed_safety_only_clearance_m
                )
            ),
        )

    @staticmethod
    def _action(value: np.ndarray) -> np.ndarray:
        action = np.asarray(value, dtype=np.float64)
        if action.shape != (_JOINTS,) or not np.all(np.isfinite(action)):
            raise ValueError("proposed action must be a finite six-vector")
        return np.clip(action, -1.0, 1.0)

    @staticmethod
    def _queue_row(value: np.ndarray) -> dict[str, Any]:
        return {
            "command_id": int(value.command_id),
            "target_hex": np.asarray(value, dtype=np.float64).tobytes().hex(),
            "original_action_hex": np.asarray(
                value.original_action, dtype=np.float64
            ).tobytes().hex(),
            "submitted_action_hex": np.asarray(
                value.submitted_action, dtype=np.float64
            ).tobytes().hex(),
            "send_step": int(value.send_step),
            "send_time_seconds": float(value.send_time_seconds),
            "submission_safety_reason": str(value.submission_safety_reason),
            "submission_target_changed_mask_hex": np.asarray(
                value.submission_target_changed_mask, dtype=np.uint8
            ).tobytes().hex(),
            "ingress_lost": bool(value.ingress_lost),
            "post_submission_rewrite_count": int(
                value.post_submission_rewrite_count
            ),
            "post_submission_target_changed_mask_hex": np.asarray(
                value.post_submission_target_changed_mask, dtype=np.uint8
            ).tobytes().hex(),
            "post_submission_rewrite_reasons": list(
                value.post_submission_rewrite_reasons
            ),
        }

    def queue_identity(self) -> dict[str, Any]:
        rows = [self._queue_row(value) for value in self.env._command_queue]
        encoded = json.dumps(
            rows,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        return {
            "depth": len(rows),
            "command_ids": [int(row["command_id"]) for row in rows],
            "sha256": hashlib.sha256(encoded).hexdigest(),
        }

    @staticmethod
    def _planning_valid(
        forecast: dict[str, Any],
        planning_margin_m: float,
    ) -> bool:
        return bool(
            forecast["valid"]
            and not forecast["unauthorized_contact_part_penetration_any"]
            and all(
                float(row["minimum_94_safety_only_block_distance_m"])
                >= planning_margin_m - 1.0e-12
                for row in forecast["applications"]
            )
        )

    def _simulated_command(
        self,
        action: np.ndarray,
        *,
        command_id: int,
        send_step: int,
    ) -> _QueuedCommand:
        env = self.env
        reported = np.asarray(
            env._command_reference_reported_position(), dtype=np.float64
        )
        requested_physical = (
            reported
            + action * float(env.config.max_joint_delta)
            - np.asarray(env._zero_offset, dtype=np.float64)
        )
        target, reason = env._safety_filter(requested_physical)
        return _QueuedCommand(
            target,
            command_id=int(command_id),
            original_action=action,
            submitted_action=action,
            send_step=int(send_step),
            send_time_seconds=float(env.data.time),
            submission_safety_reason=str(reason),
            submission_target_changed_mask=env._target_changed_mask(
                requested_physical, target
            ),
            ingress_lost=False,
        )

    def forecast_modified_action_with_hold_tail(
        self,
        first_action: np.ndarray,
    ) -> dict[str, Any]:
        """Forecast a modified command and the exact-ID wait-hold pipeline.

        One zero-action hold is submitted at every following simulated decision.
        The rollout continues until the modified command and ``queue_depth``
        later holds have all taken effect, proving both braking viability and
        that the remaining delayed queue can consist exclusively of holds.
        """

        env = self.env
        first = self._action(first_action)
        queue_depth = len(env._command_queue)
        application_count = 2 * queue_depth + 1
        snapshot = env._snapshot_terminal_viability_state()
        counters = self.forecast._counter_snapshot(env)
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
                (
                    endpoint,
                    physical_velocity,
                    runtime_reason,
                    _tracking_noise,
                    _runtime_changed_mask,
                ) = env._servo_step(delayed_target)
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
                    raise RuntimeError("V15 hold-tail rollout produced no trace")
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
                            audit[
                                "minimum_94_safety_only_block_distance_m"
                            ]
                        ),
                        "minimum_94_safety_only_block_distance_location": dict(
                            audit[
                                "minimum_94_safety_only_block_distance_location"
                            ]
                        ),
                        "unauthorized_contact_part_penetration_any": bool(
                            audit[
                                "unauthorized_contact_part_penetration_any"
                            ]
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
            self.forecast._restore_counters(env, counters)

        modified_rows = [
            row for row in reports if bool(row["applied_is_modified_command"])
        ]
        if len(modified_rows) != 1:
            raise RuntimeError("V15 hold-tail rollout lost modified command identity")
        limiting_index = min(
            range(len(reports)),
            key=lambda index: float(
                reports[index]["minimum_94_safety_only_block_distance_m"]
            ),
        )
        minimum = float(
            reports[limiting_index]["minimum_94_safety_only_block_distance_m"]
        )
        return {
            "online_before_submission": True,
            "diagnostic_replay": False,
            "transport_contract": "deterministic_no_loss_no_burst_only",
            "live_state_restored": True,
            "queue_depth": queue_depth,
            "application_count": application_count,
            "modified_command_id": original_step_count,
            "modified_command_effect_application_index": int(
                modified_rows[0]["application_index"]
            ),
            "tail_hold_effect_count": queue_depth,
            "minimum_forecast_94_safety_only_block_distance_m": minimum,
            "minimum_forecast_location": {
                "application_index": limiting_index,
                "applied_command_id": int(
                    reports[limiting_index]["applied_command_id"]
                ),
                **dict(
                    reports[limiting_index][
                        "minimum_94_safety_only_block_distance_location"
                    ]
                ),
            },
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
        """Return a safe scaled action, or ``None`` with a fail-closed proof."""

        proposed = self._action(proposed_action)
        before = self.queue_identity()
        attempts: list[dict[str, Any]] = []
        selected_action: np.ndarray | None = None
        selected_forecast: dict[str, Any] = {}
        selected_hold_tail_forecast: dict[str, Any] = {}
        selected_scale: float | None = None
        attempted_actions: set[bytes] = set()
        for scale in self.config.candidate_scales:
            candidate = np.asarray(proposed * float(scale), dtype=np.float64)
            key = candidate.tobytes()
            if key in attempted_actions:
                continue
            attempted_actions.add(key)
            forecast = self.forecast.forecast(candidate)
            after_attempt = self.queue_identity()
            if after_attempt != before:
                raise RuntimeError(
                    "V15 online forecast changed an already-submitted queue identity"
                )
            planning_valid = self._planning_valid(
                forecast,
                self.config.planning_safety_only_margin_m,
            )
            hold_tail: dict[str, Any] = {}
            if planning_valid and require_hold_tail:
                hold_tail = self.forecast_modified_action_with_hold_tail(candidate)
                after_tail = self.queue_identity()
                if after_tail != before:
                    raise RuntimeError(
                        "V15 hold-tail forecast changed an already-submitted queue identity"
                    )
                planning_valid = bool(
                    hold_tail["planning_margin_valid"]
                    and hold_tail["remaining_submissions_are_holds"]
                )
            attempts.append(
                {
                    "scale": float(scale),
                    "candidate_action_linf": float(np.max(np.abs(candidate))),
                    "hard_executed_clearance_valid": bool(forecast["valid"]),
                    "planning_margin_valid": planning_valid,
                    "minimum_forecast_94_safety_only_block_distance_m": float(
                        forecast[
                            "minimum_forecast_94_safety_only_block_distance_m"
                        ]
                    ),
                    "minimum_forecast_94_safety_only_block_distance_location": dict(
                        forecast[
                            "minimum_forecast_94_safety_only_block_distance_location"
                        ]
                    ),
                    "unauthorized_contact_part_penetration_any": bool(
                        forecast[
                            "unauthorized_contact_part_penetration_any"
                        ]
                    ),
                    "failure_reasons": list(forecast["failure_reasons"]),
                    "hold_tail_forecast_run": bool(hold_tail),
                    "hold_tail_planning_margin_valid": (
                        None
                        if not hold_tail
                        else bool(hold_tail["planning_margin_valid"])
                    ),
                    "hold_tail_minimum_forecast_94_safety_only_block_distance_m": (
                        None
                        if not hold_tail
                        else float(
                            hold_tail[
                                "minimum_forecast_94_safety_only_block_distance_m"
                            ]
                        )
                    ),
                }
            )
            if planning_valid:
                selected_action = candidate
                selected_forecast = forecast
                selected_hold_tail_forecast = hold_tail
                selected_scale = float(scale)
                break

        after = self.queue_identity()
        if after != before:
            raise RuntimeError("V15 guard changed an already-submitted queue identity")
        full_scale = attempts[0]
        safe_found = selected_action is not None
        report = {
            "format": ALL_STAGE_ACTION_GUARD_FORMAT_V1,
            "online_before_submission": True,
            "diagnostic_replay": False,
            "simulator_privileged": True,
            "physically_calibrated": False,
            "physical_samples": 0,
            "decision_step": int(self.env.step_count),
            "candidate_command_id": int(self.env.next_command_id),
            "hard_executed_safety_only_clearance_m": (
                self.config.hard_executed_safety_only_clearance_m
            ),
            "planning_safety_only_margin_m": (
                self.config.planning_safety_only_margin_m
            ),
            "hold_tail_required": bool(require_hold_tail),
            "proposed_action_linf": float(np.max(np.abs(proposed))),
            "full_scale_hard_valid": bool(
                full_scale["hard_executed_clearance_valid"]
            ),
            "full_scale_planning_margin_valid": bool(
                full_scale["planning_margin_valid"]
            ),
            "full_scale_minimum_forecast_94_safety_only_block_distance_m": float(
                full_scale[
                    "minimum_forecast_94_safety_only_block_distance_m"
                ]
            ),
            "full_scale_minimum_location": dict(
                full_scale[
                    "minimum_forecast_94_safety_only_block_distance_location"
                ]
            ),
            "safe_candidate_found": safe_found,
            "selected_scale": selected_scale,
            "selected_action_linf": (
                None
                if selected_action is None
                else float(np.max(np.abs(selected_action)))
            ),
            "selected_forecast": selected_forecast,
            "selected_hold_tail_forecast": selected_hold_tail_forecast,
            "attempts": attempts,
            "queue_identity_before": before,
            "queue_identity_after": after,
            "queue_identity_preserved": bool(before == after),
            "already_submitted_queue_mutated": False,
        }
        return (
            None
            if selected_action is None
            else np.asarray(selected_action, dtype=np.float32),
            report,
        )


__all__ = [
    "ALL_STAGE_ACTION_GUARD_FORMAT_V1",
    "AllStageActionGuardConfigV1",
    "AllStageActionGuardV1",
]
