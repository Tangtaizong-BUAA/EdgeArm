"""Fail-closed start-geometry contract for full-task RL/VLA trajectories."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np


FULL_TASK_START_CONTRACT_FORMAT_V669 = (
    "edgearm-v669-full-task-start-geometry-contract-v1"
)
FULL_TASK_MINIMUM_INITIAL_DISTANCE_M_V669 = 0.180
FULL_TASK_MAXIMUM_INITIAL_DISTANCE_M_V669 = 0.260
FULL_TASK_MAXIMUM_INITIAL_COVERAGE_V669 = 1.0e-6


def audit_full_task_start_v669(record: Mapping[str, Any]) -> dict[str, Any]:
    """Prove that a trajectory starts from the real, separated Home task."""

    if not isinstance(record, Mapping):
        raise TypeError("V669 full-task start record must be a mapping")
    try:
        initial_distance = float(record["initial_object_target_distance_m"])
        initial_coverage = float(record["initial_target_coverage"])
    except (KeyError, TypeError, ValueError):
        initial_distance = float("nan")
        initial_coverage = float("nan")
    checks = {
        "exact_home_start": record.get("exact_home_start") is True,
        "non_privileged_reset": (
            record.get("task_aligned_privileged_reset") is False
        ),
        "finite_initial_distance": bool(np.isfinite(initial_distance)),
        "initial_distance_not_near_target": bool(
            np.isfinite(initial_distance)
            and initial_distance >= FULL_TASK_MINIMUM_INITIAL_DISTANCE_M_V669
        ),
        "initial_distance_inside_current_task_workspace": bool(
            np.isfinite(initial_distance)
            and initial_distance <= FULL_TASK_MAXIMUM_INITIAL_DISTANCE_M_V669
        ),
        "finite_initial_coverage": bool(np.isfinite(initial_coverage)),
        "initial_target_coverage_zero": bool(
            np.isfinite(initial_coverage)
            and initial_coverage <= FULL_TASK_MAXIMUM_INITIAL_COVERAGE_V669
        ),
    }
    return {
        "format": FULL_TASK_START_CONTRACT_FORMAT_V669,
        "initial_object_target_distance_m": initial_distance,
        "initial_target_coverage": initial_coverage,
        "allowed_initial_distance_band_m": [
            FULL_TASK_MINIMUM_INITIAL_DISTANCE_M_V669,
            FULL_TASK_MAXIMUM_INITIAL_DISTANCE_M_V669,
        ],
        "maximum_initial_target_coverage": (
            FULL_TASK_MAXIMUM_INITIAL_COVERAGE_V669
        ),
        "checks": checks,
        "eligible": bool(all(checks.values())),
        "production_admission": False,
    }


def full_task_start_geometry_eligible_v669(record: Mapping[str, Any]) -> bool:
    return bool(audit_full_task_start_v669(record)["eligible"])


__all__ = [
    "FULL_TASK_MAXIMUM_INITIAL_COVERAGE_V669",
    "FULL_TASK_MAXIMUM_INITIAL_DISTANCE_M_V669",
    "FULL_TASK_MINIMUM_INITIAL_DISTANCE_M_V669",
    "FULL_TASK_START_CONTRACT_FORMAT_V669",
    "audit_full_task_start_v669",
    "full_task_start_geometry_eligible_v669",
]
