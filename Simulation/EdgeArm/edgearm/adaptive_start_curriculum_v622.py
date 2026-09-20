"""Sequential reverse start-state curriculum for complete pushing.

The pre-contact reset remains useful for retaining contact transport, but it
must not dominate training or be confused with a full robot episode.  V627
refines the original coarse approach jumps into smaller steps and unlocks only
the earliest unmastered frontier.  Home probes and easy-start retention remain
guaranteed by schedule.

Only exact Home evaluation may open the downstream data-generation gate.
Every other tier is training-only privileged curriculum state.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from typing import Any, Sequence

import numpy as np


ADAPTIVE_START_CURRICULUM_FORMAT_V622 = (
    "edgearm-v637-recent-mastery-sequential-reverse-start-curriculum-v1"
)
SEQUENTIAL_REVERSE_CURRICULUM_FORMAT_V627 = (
    ADAPTIVE_START_CURRICULUM_FORMAT_V622
)


@dataclass(frozen=True)
class StartTierV622:
    index: int
    code: str
    reset_kind: str
    home_to_precontact_fraction: float
    privileged_training_reset: bool
    eligible_for_final_data: bool

    def validate(self) -> None:
        if type(self.index) is not int or self.index < 0:
            raise ValueError("V622 start-tier index is invalid")
        if not self.code or not self.code.replace("_", "").isalnum():
            raise ValueError("V622 start-tier code is invalid")
        if self.reset_kind not in {
            "task_aligned_precontact",
            "interpolated_approach",
            "task_independent_home",
        }:
            raise ValueError("V622 reset kind is invalid")
        if not 0.0 <= self.home_to_precontact_fraction <= 1.0:
            raise ValueError("V622 interpolation fraction is invalid")
        if self.eligible_for_final_data and (
            self.privileged_training_reset
            or self.reset_kind != "task_independent_home"
            or self.home_to_precontact_fraction != 0.0
        ):
            raise ValueError("V622 only exact Home may be final-data eligible")


START_TIERS_V622 = (
    StartTierV622(
        0,
        "precontact",
        "task_aligned_precontact",
        1.0,
        True,
        False,
    ),
    StartTierV622(
        1,
        "approach_95",
        "interpolated_approach",
        0.95,
        True,
        False,
    ),
    StartTierV622(
        2,
        "approach_90",
        "interpolated_approach",
        0.90,
        True,
        False,
    ),
    StartTierV622(
        3,
        "approach_80",
        "interpolated_approach",
        0.80,
        True,
        False,
    ),
    StartTierV622(
        4,
        "approach_65",
        "interpolated_approach",
        0.65,
        True,
        False,
    ),
    StartTierV622(
        5,
        "approach_45",
        "interpolated_approach",
        0.45,
        True,
        False,
    ),
    StartTierV622(
        6,
        "approach_25",
        "interpolated_approach",
        0.25,
        True,
        False,
    ),
    StartTierV622(
        7,
        "home",
        "task_independent_home",
        0.0,
        False,
        True,
    ),
)

for _tier in START_TIERS_V622:
    _tier.validate()


def start_tier_v622(code: str) -> StartTierV622:
    for tier in START_TIERS_V622:
        if tier.code == code:
            return tier
    raise ValueError(f"unknown V622 start tier: {code}")


def _tier_evidence_v622(
    records: Sequence[dict[str, Any]],
) -> dict[str, dict[str, float | int]]:
    result: dict[str, dict[str, float | int]] = {}
    for tier in START_TIERS_V622:
        rows = [
            row for row in records if row.get("start_tier_v622") == tier.code
        ]
        count = len(rows)
        successes = sum(int(bool(row.get("strict_success"))) for row in rows)
        contacts = sum(int(int(row.get("valid_contact_steps", 0)) > 0) for row in rows)
        recent = rows[-4:]
        recent_count = len(recent)
        recent_successes = sum(
            int(bool(row.get("strict_success"))) for row in recent
        )
        recent_contacts = sum(
            int(int(row.get("valid_contact_steps", 0)) > 0)
            for row in recent
        )
        result[tier.code] = {
            "episode_count": count,
            "strict_success_count": successes,
            "strict_success_rate": successes / count if count else 0.0,
            "contact_episode_count": contacts,
            "contact_episode_rate": contacts / count if count else 0.0,
            "recent_window_size": 4,
            "recent_episode_count": recent_count,
            "recent_strict_success_count": recent_successes,
            "recent_strict_success_rate": (
                recent_successes / recent_count if recent_count else 0.0
            ),
            "recent_contact_episode_count": recent_contacts,
            "recent_contact_episode_rate": (
                recent_contacts / recent_count if recent_count else 0.0
            ),
        }
    return result


def _tier_mastered_v637(row: dict[str, float | int]) -> bool:
    count = int(row["recent_episode_count"])
    contact_count = int(row["recent_contact_episode_count"])
    strict_count = int(row["recent_strict_success_count"])
    return bool(
        count >= 4
        and (
            (
                contact_count >= 3
                and float(row["recent_contact_episode_rate"]) >= 0.75
            )
            or (
                strict_count >= 2
                and float(row["recent_strict_success_rate"]) >= 0.50
            )
        )
    )


def _sequential_frontier_v627(
    evidence: dict[str, dict[str, float | int]],
) -> tuple[StartTierV622, list[str]]:
    mastered: list[str] = []
    for tier in START_TIERS_V622[1:-1]:
        if _tier_mastered_v637(evidence[tier.code]):
            mastered.append(tier.code)
            continue
        return tier, mastered
    return START_TIERS_V622[-2], mastered


def select_start_tier_v622(
    *,
    episode_index: int,
    records: Sequence[dict[str, Any]],
    seed: int,
) -> tuple[StartTierV622, dict[str, Any]]:
    """Select a reproducible sequential frontier with Home/easy probes."""

    if type(episode_index) is not int or episode_index < 1:
        raise ValueError("V622 episode index must be positive")
    if type(seed) is not int or seed < 0:
        raise ValueError("V622 selection seed must be non-negative")
    evidence = _tier_evidence_v622(records)
    frontier, mastered_tiers = _sequential_frontier_v627(evidence)
    selection_reason: str
    probabilities: dict[str, float]
    if episode_index == 1:
        selected = START_TIERS_V622[0]
        selection_reason = "guaranteed_initial_precontact"
        probabilities = {tier.code: float(tier is selected) for tier in START_TIERS_V622}
    elif episode_index % 5 == 0:
        selected = START_TIERS_V622[-1]
        selection_reason = "guaranteed_exact_home_probe"
        probabilities = {tier.code: float(tier is selected) for tier in START_TIERS_V622}
    elif episode_index % 4 == 0:
        selected = START_TIERS_V622[0]
        selection_reason = "guaranteed_precontact_retention"
        probabilities = {tier.code: float(tier is selected) for tier in START_TIERS_V622}
    else:
        predecessor = START_TIERS_V622[max(frontier.index - 1, 1)]
        candidates = (
            (frontier,)
            if predecessor is frontier
            else (frontier, predecessor)
        )
        normalized = (
            np.asarray((1.0,), dtype=np.float64)
            if len(candidates) == 1
            else np.asarray((0.80, 0.20), dtype=np.float64)
        )
        rng = np.random.default_rng(seed)
        selected_index = int(rng.choice(len(candidates), p=normalized))
        selected = candidates[selected_index]
        selection_reason = "sequential_reverse_approach_frontier"
        probabilities = {tier.code: 0.0 for tier in START_TIERS_V622}
        for tier, probability in zip(candidates, normalized, strict=True):
            probabilities[tier.code] = float(probability)

    audit = {
        "format": ADAPTIVE_START_CURRICULUM_FORMAT_V622,
        "episode_index": episode_index,
        "selection_seed": seed,
        "selected_tier": asdict(selected),
        "selection_reason": selection_reason,
        "selection_probabilities": probabilities,
        "evidence_before_selection": evidence,
        "sequential_frontier_tier": frontier.code,
        "mastered_approach_tiers": mastered_tiers,
        "frontier_mastery_rule": {
            "most_recent_episode_window": 4,
            "minimum_episode_count": 4,
            "minimum_contact_episode_count": 3,
            "minimum_contact_episode_rate": 0.75,
            "strict_success_override_minimum_count": 2,
            "strict_success_override_minimum_rate": 0.50,
        },
        "precontact_retention_interval": 4,
        "exact_home_probe_interval": 5,
        "only_exact_home_eligible_for_final_data": True,
        "production_admission": False,
    }
    return selected, audit


def curriculum_cohort_audit_v622(
    records: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    evidence = _tier_evidence_v622(records)
    counts = Counter(str(row.get("start_tier_v622", "missing")) for row in records)
    unknown = sorted(set(counts) - {tier.code for tier in START_TIERS_V622})
    if unknown:
        raise ValueError(f"V622 records contain unknown tiers: {unknown}")
    home = evidence["home"]
    return {
        "format": "edgearm-v637-start-curriculum-cohort-audit-v1",
        "episode_count": len(records),
        "tier_evidence": evidence,
        "all_tiers_observed": all(int(evidence[tier.code]["episode_count"]) > 0 for tier in START_TIERS_V622),
        "exact_home_episode_count": int(home["episode_count"]),
        "exact_home_strict_success_count": int(home["strict_success_count"]),
        "curriculum_episode_count": len(records) - int(home["episode_count"]),
        "curriculum_trajectories_final_data_eligible": False,
        "production_admission": False,
    }


__all__ = [
    "ADAPTIVE_START_CURRICULUM_FORMAT_V622",
    "SEQUENTIAL_REVERSE_CURRICULUM_FORMAT_V627",
    "START_TIERS_V622",
    "StartTierV622",
    "curriculum_cohort_audit_v622",
    "select_start_tier_v622",
    "start_tier_v622",
]
