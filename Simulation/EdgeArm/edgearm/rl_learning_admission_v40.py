"""Admission contract for simulated RL learning data, separate from promotion.

Failed simulated episodes are legitimate PPO evidence when their transitions
are finite, reconstructable, task-active, and explicitly labelled. They are
not successful demonstrations and they never relax the independent closed-loop
policy-promotion gates.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping


RL_LEARNING_SOURCE_ADMISSION_FORMAT_V40 = (
    "edgearm-v40-simulated-rl-learning-source-admission-v1"
)


@dataclass(frozen=True)
class RLLearningSourceAdmissionConfigV40:
    """Bounds that preserve usable negative outcomes without accepting collapse."""

    minimum_complete_episodes_per_condition: int = 2
    minimum_active_episode_fraction: float = 0.50
    minimum_contact_transitions_per_condition: int = 2
    minimum_net_target_progress_m_per_condition: float = 0.001
    maximum_invalid_contact_fraction_per_condition: float = 0.02
    maximum_shield_rejection_episode_fraction: float = 0.50
    maximum_safety_stop_episode_fraction: float = 0.50
    maximum_terminal_failure_episode_fraction: float = 0.50

    def validate(self) -> None:
        if self.minimum_complete_episodes_per_condition < 1:
            raise ValueError("V40 RL-learning episode minimum must be positive")
        if self.minimum_contact_transitions_per_condition < 1:
            raise ValueError("V40 RL-learning contact minimum must be positive")
        if not 0.0 < self.minimum_active_episode_fraction <= 1.0:
            raise ValueError("V40 RL-learning active fraction must lie in (0,1]")
        if self.minimum_net_target_progress_m_per_condition <= 0.0:
            raise ValueError("V40 RL-learning progress minimum must be positive")
        for name in (
            "maximum_invalid_contact_fraction_per_condition",
            "maximum_shield_rejection_episode_fraction",
            "maximum_safety_stop_episode_fraction",
            "maximum_terminal_failure_episode_fraction",
        ):
            value = getattr(self, name)
            if not 0.0 <= value < 1.0:
                raise ValueError(f"V40 RL-learning {name} must lie in [0,1)")


def _failure_episode_fraction(rows: list[dict[str, Any]], field: str) -> float:
    return sum(int(int(row[field]) > 0) for row in rows) / len(rows)


def rl_learning_source_admission_v40(
    rollout_audit: Mapping[str, Any],
    projection_health_checks: Mapping[str, bool],
    config: RLLearningSourceAdmissionConfigV40 | None = None,
) -> dict[str, Any]:
    """Admit a simulated rollout for learning, not for policy promotion.

    Safety-stop, shield-rejection, and terminal-failure episodes are retained as
    negative learning signals within bounded fractions. Successful-demo and
    production admission remain false regardless of this result.
    """

    selected = config or RLLearningSourceAdmissionConfigV40()
    selected.validate()
    episodes = rollout_audit.get("episodes")
    aggregate = rollout_audit.get("aggregate")
    if not isinstance(episodes, list) or not episodes or not isinstance(aggregate, dict):
        raise TypeError("V40 RL-learning admission requires a complete rollout audit")
    if not projection_health_checks or any(
        type(value) is not bool for value in projection_health_checks.values()
    ):
        raise TypeError("V40 RL-learning projection checks must be non-empty booleans")

    grouped: dict[tuple[bool, bool], list[dict[str, Any]]] = {}
    for episode in episodes:
        if not isinstance(episode, dict):
            raise TypeError("V40 RL-learning episode is not a record")
        grouped.setdefault(
            (bool(episode["obstacle"]), bool(episode["stress"])), []
        ).append(episode)

    condition_results: dict[str, Any] = {}
    condition_checks_flat: list[bool] = []
    for (obstacle, stress), rows in sorted(grouped.items()):
        episode_count = len(rows)
        active_episode_count = sum(
            int(
                int(row["valid_contact_transitions"]) > 0
                and float(row["net_target_progress_m"]) > 0.0
            )
            for row in rows
        )
        valid_contact_count = sum(
            int(row["valid_contact_transitions"]) for row in rows
        )
        invalid_contact_count = sum(
            int(row["invalid_contact_transitions"]) for row in rows
        )
        transition_count = sum(int(row["rows"]) for row in rows)
        net_target_progress_m = sum(
            float(row["net_target_progress_m"]) for row in rows
        )
        active_fraction = active_episode_count / episode_count
        invalid_fraction = invalid_contact_count / transition_count
        shield_fraction = _failure_episode_fraction(
            rows, "action_shield_rejection_transitions"
        )
        safety_stop_fraction = _failure_episode_fraction(
            rows, "safety_stop_transitions"
        )
        terminal_failure_fraction = _failure_episode_fraction(
            rows, "terminal_failure_transitions"
        )
        checks = {
            "enough_complete_episodes": (
                episode_count >= selected.minimum_complete_episodes_per_condition
            ),
            "enough_task_active_episodes": (
                active_fraction >= selected.minimum_active_episode_fraction
            ),
            "enough_valid_contact": (
                valid_contact_count
                >= selected.minimum_contact_transitions_per_condition
            ),
            "enough_net_target_progress": (
                net_target_progress_m
                >= selected.minimum_net_target_progress_m_per_condition
            ),
            "invalid_contact_fraction_within_bound": (
                invalid_fraction
                <= selected.maximum_invalid_contact_fraction_per_condition
            ),
            "shield_rejection_episode_fraction_within_learning_bound": (
                shield_fraction
                <= selected.maximum_shield_rejection_episode_fraction
            ),
            "safety_stop_episode_fraction_within_learning_bound": (
                safety_stop_fraction
                <= selected.maximum_safety_stop_episode_fraction
            ),
            "terminal_failure_episode_fraction_within_learning_bound": (
                terminal_failure_fraction
                <= selected.maximum_terminal_failure_episode_fraction
            ),
        }
        condition_checks_flat.extend(checks.values())
        condition_results[
            f"obstacle={int(obstacle)},stress={int(stress)}"
        ] = {
            "episode_count": episode_count,
            "active_episode_count": active_episode_count,
            "active_episode_fraction": active_fraction,
            "valid_contact_transitions": valid_contact_count,
            "invalid_contact_transitions": invalid_contact_count,
            "invalid_contact_fraction": invalid_fraction,
            "net_target_progress_m": net_target_progress_m,
            "shield_rejection_episode_fraction": shield_fraction,
            "safety_stop_episode_fraction": safety_stop_fraction,
            "terminal_failure_episode_fraction": terminal_failure_fraction,
            "checks": checks,
        }

    projection_health_pass = all(projection_health_checks.values())
    admitted = projection_health_pass and all(condition_checks_flat)
    return {
        "format": RL_LEARNING_SOURCE_ADMISSION_FORMAT_V40,
        "config": asdict(selected),
        "projection_health_checks": dict(projection_health_checks),
        "projection_health_pass": projection_health_pass,
        "condition_count": len(condition_results),
        "conditions": condition_results,
        "learning_update_allowed": admitted,
        "simulated_failures_retained_as_negative_learning_signal": True,
        "failed_episode_action_supervision_eligible": False,
        "successful_demonstration_admission": False,
        "strict_policy_promotion_gate_relaxed": False,
        "physical_samples": 0,
        "production_admission": False,
    }


__all__ = [
    "RL_LEARNING_SOURCE_ADMISSION_FORMAT_V40",
    "RLLearningSourceAdmissionConfigV40",
    "rl_learning_source_admission_v40",
]
