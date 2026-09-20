"""V25 held-out kernel with a wider deterministic feasible-reset search."""

from __future__ import annotations

import re
from typing import Any

from .asymmetric_multiview_ppo_v1 import (
    MAX_CURRICULUM_RESET_ATTEMPTS_V16,
    RESET_RETRY_STRIDE_V12,
    MultiViewRendererProtocolV1,
    StockGripperTaskFrameAdapterV13,
    StockTaskFrameResetInfeasibleV12,
)
from .sim2real_env_v10 import RealisticEdgeArmEnvV10
from .stock_gripper_rollout_kernel_v22 import (
    STOCK_GRIPPER_ROLLOUT_FORMAT_V22,
    StockGripperRolloutKernelV22,
)
from .stock_gripper_taskframe_v22 import (
    StockGripperTaskFrameAdapterV22,
    reset_stock_taskframe_episode_v22,
)


STOCK_GRIPPER_EVALUATION_KERNEL_FORMAT_V25 = (
    "edgearm-stock-gripper-heldout-feasible-reset-kernel-v25"
)
STOCK_GRIPPER_EVALUATION_FORMAT_V25 = (
    "edgearm-semantic-push-face-heldout-evaluation-v25"
)
STOCK_GRIPPER_EVALUATION_RESET_FORMAT_V25 = (
    "edgearm-stock-gripper-heldout-reset-retry-v25"
)
MAX_EVALUATION_RESET_ATTEMPTS_V25 = 64
_REJECTED_SEED_PATTERN_V25 = re.compile(
    r"seed=(?P<seed>\d+):(?P<reason>.*?)(?=; seed=|$)"
)


def _rejected_window_v25(
    error: StockTaskFrameResetInfeasibleV12,
    *,
    global_attempt_offset: int,
    requested_seed: int,
) -> list[dict[str, Any]]:
    message = str(error)
    parsed = [
        {
            "attempt_index": global_attempt_offset + local_index,
            "seed": int(match.group("seed")),
            "error_type": type(error).__name__,
            "reason": match.group("reason"),
        }
        for local_index, match in enumerate(
            _REJECTED_SEED_PATTERN_V25.finditer(message)
        )
    ]
    if len(parsed) != MAX_CURRICULUM_RESET_ATTEMPTS_V16:
        raise RuntimeError("V25 could not preserve rejected reset evidence") from error
    for local_index, row in enumerate(parsed):
        expected_seed = requested_seed + (
            global_attempt_offset + local_index
        ) * RESET_RETRY_STRIDE_V12
        if row["seed"] != expected_seed:
            raise RuntimeError("V25 rejected reset seed lattice changed") from error
    return parsed


def reset_stock_gripper_evaluation_episode_v25(
    env: RealisticEdgeArmEnvV10,
    renderer: MultiViewRendererProtocolV1,
    action_adapter: StockGripperTaskFrameAdapterV22,
    *,
    requested_seed: int,
    obstacle: bool,
    stress: bool,
) -> dict[str, Any]:
    """Search four exact V22 retry windows and preserve every rejection."""

    if type(action_adapter) is not StockGripperTaskFrameAdapterV22:
        raise TypeError("V25 evaluation reset requires exact V22 adapter")
    if MAX_EVALUATION_RESET_ATTEMPTS_V25 % MAX_CURRICULUM_RESET_ATTEMPTS_V16:
        raise RuntimeError("V25 reset budget is not an exact V22 window multiple")
    rejected: list[dict[str, Any]] = []
    window_count = (
        MAX_EVALUATION_RESET_ATTEMPTS_V25
        // MAX_CURRICULUM_RESET_ATTEMPTS_V16
    )
    for window_index in range(window_count):
        global_offset = window_index * MAX_CURRICULUM_RESET_ATTEMPTS_V16
        window_seed = requested_seed + global_offset * RESET_RETRY_STRIDE_V12
        try:
            base = reset_stock_taskframe_episode_v22(
                env,
                renderer,
                action_adapter,
                requested_seed=window_seed,
                obstacle=obstacle,
                stress=stress,
            )
        except StockTaskFrameResetInfeasibleV12 as error:
            rejected.extend(
                _rejected_window_v25(
                    error,
                    global_attempt_offset=global_offset,
                    requested_seed=requested_seed,
                )
            )
            continue
        local_attempt = int(base["selected_attempt_index"])
        selected_attempt = global_offset + local_attempt
        current_window_rejected: list[dict[str, Any]] = []
        for row in base["rejected_attempts"]:
            copied = dict(row)
            copied["attempt_index"] = global_offset + int(row["attempt_index"])
            expected_seed = requested_seed + (
                int(copied["attempt_index"]) * RESET_RETRY_STRIDE_V12
            )
            if int(copied["seed"]) != expected_seed:
                raise RuntimeError("V25 successful reset window seed lattice changed")
            current_window_rejected.append(copied)
        combined = [*rejected, *current_window_rejected]
        if len(combined) != selected_attempt:
            raise RuntimeError("V25 reset rejection count lost attempt identity")
        expected_selected_seed = (
            requested_seed + selected_attempt * RESET_RETRY_STRIDE_V12
        )
        if int(base["selected_seed"]) != expected_selected_seed:
            raise RuntimeError("V25 selected reset seed lattice changed")
        audit = dict(base)
        audit.update(
            {
                "format": STOCK_GRIPPER_EVALUATION_RESET_FORMAT_V25,
                "requested_seed": requested_seed,
                "selected_attempt_index": selected_attempt,
                "maximum_attempts": MAX_EVALUATION_RESET_ATTEMPTS_V25,
                "retry_stride": RESET_RETRY_STRIDE_V12,
                "rejected_attempts": combined,
                "base_reset_format": str(base["format"]),
                "base_retry_window_size": MAX_CURRICULUM_RESET_ATTEMPTS_V16,
                "selected_retry_window_index": window_index,
                "evaluation_only_extended_reset_budget": True,
                "production_admission": False,
            }
        )
        env.episode_domain["stock_gripper_evaluation_reset_retry_v25"] = audit
        return audit
    reasons = "; ".join(
        f"seed={row['seed']}:{row['reason']}" for row in rejected
    )
    raise StockTaskFrameResetInfeasibleV12(
        "V25 evaluation reset exhausted "
        f"{MAX_EVALUATION_RESET_ATTEMPTS_V25} attempts: {reasons}"
    )


class StockGripperEvaluationKernelV25(StockGripperRolloutKernelV22):
    """Keep V22 dynamics/contact/safety and widen held-out reset search only."""

    format = STOCK_GRIPPER_EVALUATION_KERNEL_FORMAT_V25
    rollout_format = STOCK_GRIPPER_ROLLOUT_FORMAT_V22
    evaluation_format = STOCK_GRIPPER_EVALUATION_FORMAT_V25

    def reset_episode(
        self,
        env: RealisticEdgeArmEnvV10,
        renderer: MultiViewRendererProtocolV1,
        action_adapter: StockGripperTaskFrameAdapterV13,
        *,
        requested_seed: int,
        obstacle: bool,
        stress: bool,
    ) -> dict[str, Any]:
        if type(action_adapter) is not StockGripperTaskFrameAdapterV22:
            raise TypeError("V25 evaluation kernel requires exact V22 adapter")
        return reset_stock_gripper_evaluation_episode_v25(
            env,
            renderer,
            action_adapter,
            requested_seed=requested_seed,
            obstacle=obstacle,
            stress=stress,
        )


__all__ = [
    "MAX_EVALUATION_RESET_ATTEMPTS_V25",
    "STOCK_GRIPPER_EVALUATION_FORMAT_V25",
    "STOCK_GRIPPER_EVALUATION_KERNEL_FORMAT_V25",
    "STOCK_GRIPPER_EVALUATION_RESET_FORMAT_V25",
    "StockGripperEvaluationKernelV25",
    "reset_stock_gripper_evaluation_episode_v25",
]
