"""V26 closed-loop effect-size gate layered on V24/V25 safety backtracking.

Floating-point or micrometre-scale changes can satisfy a strict greater-than
comparison while being irrelevant to the physical stock follower.  This gate
keeps every existing paired safety/executability guard and additionally
requires a preregistered minimum task effect before an optimizer direction may
be committed.
"""

from __future__ import annotations

from typing import Any

from .on_policy_recurrent_ppo_v20 import frontier_balanced_closed_loop_gates_v24


MEANINGFUL_EFFECT_BACKTRACKING_FORMAT_V26 = (
    "edgearm-v26-meaningful-effect-backtracked-ppo-v1"
)
MINIMUM_MEAN_FINAL_DISTANCE_IMPROVEMENT_M_V26 = 0.00025
MINIMUM_MEAN_BLOCK_PROGRESS_IMPROVEMENT_V26 = 0.00125
MINIMUM_VALID_CONTACT_TRANSITION_GAIN_V26 = 2


def meaningful_effect_audit_v26(
    comparison: dict[str, Any],
) -> dict[str, Any]:
    """Measure task effects without confusing alternative signals with gates."""

    delta = comparison.get("candidate_minus_parent")
    if not isinstance(delta, dict):
        raise TypeError("V26 meaningful-effect comparison is incomplete")
    required = (
        "strict_success_rate",
        "valid_contact_transition_count",
        "mean_final_block_target_distance_m",
        "mean_final_block_progress",
    )
    if any(name not in delta for name in required):
        raise TypeError("V26 meaningful-effect deltas are incomplete")
    strict_success_gain = float(delta["strict_success_rate"]) > 0.0
    contact_gain = (
        int(delta["valid_contact_transition_count"])
        >= MINIMUM_VALID_CONTACT_TRANSITION_GAIN_V26
    )
    distance_gain = (
        float(delta["mean_final_block_target_distance_m"])
        <= -MINIMUM_MEAN_FINAL_DISTANCE_IMPROVEMENT_M_V26
    )
    normalized_progress_gain = (
        float(delta["mean_final_block_progress"])
        >= MINIMUM_MEAN_BLOCK_PROGRESS_IMPROVEMENT_V26
    )
    return {
        "format": MEANINGFUL_EFFECT_BACKTRACKING_FORMAT_V26,
        "strict_success_gain": strict_success_gain,
        "valid_contact_gain": contact_gain,
        "mean_final_distance_gain": distance_gain,
        "mean_block_progress_gain": normalized_progress_gain,
        "strict_success_rate_delta": float(delta["strict_success_rate"]),
        "valid_contact_transition_delta": int(delta["valid_contact_transition_count"]),
        "mean_final_distance_delta_m": float(
            delta["mean_final_block_target_distance_m"]
        ),
        "mean_block_progress_delta": float(delta["mean_final_block_progress"]),
        "minimum_meaningful_primary_task_effect": any(
            (
                strict_success_gain,
                contact_gain,
                distance_gain,
                normalized_progress_gain,
            )
        ),
        "contract": meaningful_effect_contract_v26(),
        "production_admission": False,
    }


def meaningful_effect_closed_loop_gates_v26(
    comparison: dict[str, Any],
) -> dict[str, bool]:
    """Reject safe but physically negligible paired policy changes."""

    checks = frontier_balanced_closed_loop_gates_v24(comparison)
    audit = meaningful_effect_audit_v26(comparison)
    checks["minimum_meaningful_primary_task_effect"] = bool(
        audit["minimum_meaningful_primary_task_effect"]
    )
    return checks


def meaningful_effect_contract_v26() -> dict[str, Any]:
    return {
        "format": MEANINGFUL_EFFECT_BACKTRACKING_FORMAT_V26,
        "minimum_mean_final_distance_improvement_m": (
            MINIMUM_MEAN_FINAL_DISTANCE_IMPROVEMENT_M_V26
        ),
        "minimum_mean_block_progress_improvement": (
            MINIMUM_MEAN_BLOCK_PROGRESS_IMPROVEMENT_V26
        ),
        "minimum_valid_contact_transition_gain": (
            MINIMUM_VALID_CONTACT_TRANSITION_GAIN_V26
        ),
        "strict_success_gain_always_meaningful": True,
        "existing_v24_safety_and_executability_gates_retained": True,
        "production_admission": False,
    }


__all__ = [
    "MEANINGFUL_EFFECT_BACKTRACKING_FORMAT_V26",
    "MINIMUM_MEAN_BLOCK_PROGRESS_IMPROVEMENT_V26",
    "MINIMUM_MEAN_FINAL_DISTANCE_IMPROVEMENT_M_V26",
    "MINIMUM_VALID_CONTACT_TRANSITION_GAIN_V26",
    "meaningful_effect_closed_loop_gates_v26",
    "meaningful_effect_audit_v26",
    "meaningful_effect_contract_v26",
]
