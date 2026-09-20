"""Fail-closed reverse curriculum for scratch stock-gripper pushing.

The curriculum starts with a short but non-trivial push into the target and
expands the block-to-target distance and domain difficulty only after a policy
demonstrates repeated three-second settled success.  It supplies reset and
promotion metadata only; no expert action, path, or privileged state is ever an
actor input.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any

import numpy as np


REVERSE_CURRICULUM_FORMAT_V26 = "edgearm-v26-scratch-reverse-curriculum-v1"
STRICT_SUCCESS_HOLD_SECONDS_V26 = 3.0
NOMINAL_CONTROL_FPS_V26 = 30
STRICT_SUCCESS_HOLD_STEPS_V26 = 90
PRETERMINAL_BOOTSTRAP_STAGE_INDEX_V29 = 29
CONTACT_SETTLE_BOOTSTRAP_STAGE_INDEX_V34 = 30
GEOMETRY_BRIDGE_STAGE_INDEX_V509 = 31
GEOMETRY_FULLPUSH_STAGE_INDEX_V509 = 32
LONG_RANGE_STAGE_INDEX_V648 = 33
EXTENDED_RANGE_STAGE_INDEX_V648 = 34


def strict_success_hold_steps_v26(
    fps: int,
    *,
    hold_seconds: float = STRICT_SUCCESS_HOLD_SECONDS_V26,
) -> int:
    """Return the smallest step count whose physical duration is at least 3 s."""

    if type(fps) is not int or fps < 1:
        raise ValueError("V26 strict-success fps must be a positive integer")
    if not np.isfinite(hold_seconds) or hold_seconds <= 0.0:
        raise ValueError("V26 strict-success hold seconds must be positive")
    steps = int(math.ceil(float(fps) * float(hold_seconds)))
    if steps / fps + 1.0e-12 < hold_seconds:
        raise RuntimeError("V26 strict-success step conversion undershot duration")
    return steps


@dataclass(frozen=True)
class ReverseCurriculumStageV26:
    index: int
    name: str
    block_target_distance_range_m: tuple[float, float]
    tip_gap_range_m: tuple[float, float]
    obstacle_probability: float
    stress_probability: float
    minimum_evaluation_episodes: int = 32
    minimum_strict_success_rate: float = 0.70
    minimum_wilson_lower_bound: float = 0.50
    minimum_contact_episode_fraction: float = 0.75
    maximum_invalid_contact_fraction: float = 0.01
    maximum_shield_terminal_fraction: float = 0.02

    def validate(self) -> None:
        if type(self.index) is not int or self.index < 0:
            raise ValueError("V26 curriculum stage index must be non-negative")
        if not self.name or not self.name.replace("_", "").isalnum():
            raise ValueError("V26 curriculum stage name is invalid")
        distance = np.asarray(self.block_target_distance_range_m, dtype=np.float64)
        gap = np.asarray(self.tip_gap_range_m, dtype=np.float64)
        extended_index = self.index in {
            LONG_RANGE_STAGE_INDEX_V648,
            EXTENDED_RANGE_STAGE_INDEX_V648,
        }
        maximum_distance_m = 0.260 if extended_index else 0.190
        if (
            distance.shape != (2,)
            or not np.all(np.isfinite(distance))
            or distance[0] < 0.03
            or distance[0] >= distance[1]
            or distance[1] > maximum_distance_m
        ):
            raise ValueError("V26 block-target distance band is invalid")
        if (
            gap.shape != (2,)
            or not np.all(np.isfinite(gap))
            or gap[0] < 0.00025
            or gap[0] >= gap[1]
            or gap[1] > 0.020
        ):
            raise ValueError("V26 tip-gap band is invalid")
        probabilities = (
            self.obstacle_probability,
            self.stress_probability,
            self.minimum_strict_success_rate,
            self.minimum_wilson_lower_bound,
            self.minimum_contact_episode_fraction,
            self.maximum_invalid_contact_fraction,
            self.maximum_shield_terminal_fraction,
        )
        if any(not np.isfinite(value) or not 0.0 <= value <= 1.0 for value in probabilities):
            raise ValueError("V26 curriculum probability or fraction is outside [0,1]")
        if type(self.minimum_evaluation_episodes) is not int or self.minimum_evaluation_episodes < 8:
            raise ValueError("V26 promotion requires at least eight evaluation episodes")
        if extended_index:
            expected = {
                LONG_RANGE_STAGE_INDEX_V648: "v648_long_range_obstacle_free",
                EXTENDED_RANGE_STAGE_INDEX_V648: (
                    "v648_extended_range_obstacle_free"
                ),
            }[self.index]
            if (
                self.name != expected
                or self.obstacle_probability != 0.0
                or self.stress_probability != 0.0
            ):
                raise ValueError(
                    "V648 extended stages must be exact obstacle-free profiles"
                )


@dataclass(frozen=True)
class ReverseCurriculumTaskV26:
    stage_index: int
    stress: bool
    block_xy_m: tuple[float, float]
    target_xy_m: tuple[float, float]
    block_target_distance_m: float
    direction_angle_rad: float
    sampling_attempt_index: int
    format: str = REVERSE_CURRICULUM_FORMAT_V26


def sample_reverse_curriculum_task_v26(
    rng: np.random.Generator,
    stage: ReverseCurriculumStageV26,
    *,
    stress: bool,
) -> ReverseCurriculumTaskV26:
    """Sample a deterministic reachable task from one curriculum distance band."""

    if type(rng) is not np.random.Generator:
        raise TypeError("V26 curriculum task sampling requires numpy Generator")
    if type(stress) is not bool:
        raise TypeError("V26 curriculum stress selector must be boolean")
    stage.validate()
    extended_range = stage.index in {
        LONG_RANGE_STAGE_INDEX_V648,
        EXTENDED_RANGE_STAGE_INDEX_V648,
    }
    if extended_range:
        if stress:
            raise ValueError("V648 long-range sampling is obstacle-free nominal only")
        # Preserve a uniform distance distribution rather than repeatedly
        # redrawing easier, shorter distances after geometry rejection.
        distance = float(rng.uniform(*stage.block_target_distance_range_m))
        angle = float(rng.uniform(-0.22, 0.22))
        direction = np.asarray(
            [math.cos(angle), math.sin(angle)], dtype=np.float64
        )
        delta = distance * direction
        # The 50 mm block must sit at least 5 mm inside the x=0.10 m desk
        # front edge, hence center x >= 0.130 m.  The lateral center bound
        # leaves additional stock-gripper workspace margin.
        block_x_min = 0.130
        block_y_abs_max = 0.145
        target_x_low = max(0.300, block_x_min + float(delta[0]))
        target_x_high = 0.390
        target_y_low = max(-0.130, -block_y_abs_max + float(delta[1]))
        target_y_high = min(0.130, block_y_abs_max + float(delta[1]))
        if target_x_low >= target_x_high or target_y_low >= target_y_high:
            raise RuntimeError("V648 long-range geometry has no supported task")
        target = rng.uniform(
            [target_x_low, target_y_low],
            [target_x_high, target_y_high],
        )
        block = target - delta
        realized = float(np.linalg.norm(target - block))
        if not (
            np.isclose(realized, distance, rtol=0.0, atol=1.0e-12)
            and block_x_min <= block[0] <= 0.30
            and abs(float(block[1])) <= block_y_abs_max
        ):
            raise RuntimeError("V648 supported sampler violated its geometry")
        return ReverseCurriculumTaskV26(
            stage_index=stage.index,
            stress=False,
            block_xy_m=(float(block[0]), float(block[1])),
            target_xy_m=(float(target[0]), float(target[1])),
            block_target_distance_m=realized,
            direction_angle_rad=angle,
            sampling_attempt_index=0,
        )
    angle_limit = 0.48 if stress else 0.22
    for attempt_index in range(300):
        target = rng.uniform(
            [0.295, -0.15] if stress else [0.300, -0.13],
            [0.395, 0.15] if stress else [0.390, 0.13],
        )
        distance = float(rng.uniform(*stage.block_target_distance_range_m))
        angle = float(rng.uniform(-angle_limit, angle_limit))
        direction = np.asarray([math.cos(angle), math.sin(angle)], dtype=np.float64)
        block = target - distance * direction
        if 0.12 <= block[0] <= 0.30 and -0.18 <= block[1] <= 0.18:
            realized = float(np.linalg.norm(target - block))
            if not np.isclose(realized, distance, rtol=0.0, atol=1.0e-12):
                raise RuntimeError("V26 curriculum task distance changed during sampling")
            return ReverseCurriculumTaskV26(
                stage_index=stage.index,
                stress=stress,
                block_xy_m=(float(block[0]), float(block[1])),
                target_xy_m=(float(target[0]), float(target[1])),
                block_target_distance_m=realized,
                direction_angle_rad=angle,
                sampling_attempt_index=attempt_index,
            )
    raise RuntimeError("V26 could not sample a reachable reverse-curriculum task")


REVERSE_CURRICULUM_STAGES_V26 = (
    ReverseCurriculumStageV26(
        index=0,
        name="short_entry_push",
        block_target_distance_range_m=(0.045, 0.065),
        tip_gap_range_m=(0.0025, 0.0048),
        obstacle_probability=0.0,
        stress_probability=0.0,
    ),
    ReverseCurriculumStageV26(
        index=1,
        name="short_nominal_push",
        block_target_distance_range_m=(0.060, 0.085),
        tip_gap_range_m=(0.0025, 0.0060),
        obstacle_probability=0.0,
        stress_probability=0.10,
    ),
    ReverseCurriculumStageV26(
        index=2,
        name="medium_mixed_push",
        block_target_distance_range_m=(0.080, 0.115),
        tip_gap_range_m=(0.0015, 0.0080),
        obstacle_probability=0.15,
        stress_probability=0.20,
    ),
    ReverseCurriculumStageV26(
        index=3,
        name="production_nominal_push",
        block_target_distance_range_m=(0.105, 0.150),
        tip_gap_range_m=(0.0005, 0.0120),
        obstacle_probability=0.30,
        stress_probability=0.30,
        minimum_strict_success_rate=0.65,
        minimum_wilson_lower_bound=0.45,
    ),
    ReverseCurriculumStageV26(
        index=4,
        name="production_stress_push",
        block_target_distance_range_m=(0.115, 0.190),
        tip_gap_range_m=(0.00025, 0.0200),
        obstacle_probability=0.50,
        stress_probability=0.50,
        minimum_strict_success_rate=0.60,
        minimum_wilson_lower_bound=0.40,
    ),
)

PRETERMINAL_BOOTSTRAP_STAGE_V29 = ReverseCurriculumStageV26(
    index=PRETERMINAL_BOOTSTRAP_STAGE_INDEX_V29,
    name="v29_preterminal_bootstrap",
    block_target_distance_range_m=(0.034, 0.042),
    tip_gap_range_m=(0.0025, 0.0075),
    obstacle_probability=0.0,
    stress_probability=0.0,
)


# V34 is a common-seed, single-configuration-variable predecessor to V29.  It
# preserves the stock-gripper tip gap and every domain condition while
# shortening only the block-to-target distance.  Reachability rejection can
# change the realized target for an individual seed, so downstream audits must
# not claim exact per-task pairing.  The 31 mm lower bound is deliberately
# above the approximately 28 mm footprint distance that already satisfies 95%
# coverage; even the audited 1 mm reset-settle tolerance therefore cannot
# pre-solve the task.  A policy must still make contact, move the block, and
# hold it settled for the exact three-second contract.
CONTACT_SETTLE_BOOTSTRAP_STAGE_V34 = ReverseCurriculumStageV26(
    index=CONTACT_SETTLE_BOOTSTRAP_STAGE_INDEX_V34,
    name="v34_contact_settle_bootstrap",
    block_target_distance_range_m=(0.031, 0.034),
    tip_gap_range_m=(0.0025, 0.0075),
    obstacle_probability=0.0,
    stress_probability=0.0,
)


# V509 adds two obstacle-free geometry stages without changing the meaning or
# ordering of the historical V26 stages.  Special indices mirror the existing
# V29/V34 bootstrap convention and keep old checkpoints reproducible.
GEOMETRY_BRIDGE_STAGE_V509 = ReverseCurriculumStageV26(
    index=GEOMETRY_BRIDGE_STAGE_INDEX_V509,
    name="geometry_bridge_obstacle_free",
    block_target_distance_range_m=(0.150, 0.170),
    tip_gap_range_m=(0.0005, 0.0120),
    obstacle_probability=0.0,
    stress_probability=0.0,
    minimum_strict_success_rate=0.60,
    minimum_wilson_lower_bound=0.40,
)


GEOMETRY_FULLPUSH_STAGE_V509 = ReverseCurriculumStageV26(
    index=GEOMETRY_FULLPUSH_STAGE_INDEX_V509,
    name="geometry_fullpush_obstacle_free",
    block_target_distance_range_m=(0.170, 0.190),
    tip_gap_range_m=(0.0005, 0.0120),
    obstacle_probability=0.0,
    stress_probability=0.0,
    minimum_strict_success_rate=0.60,
    minimum_wilson_lower_bound=0.40,
)


# V648 adds genuinely longer obstacle-free tasks without changing any
# historical stage or widening the meaning of stage 32.  These bands require
# substantially more center travel than the 17--19 cm curriculum and remain
# isolated until exact stock-gripper reset feasibility is audited.
LONG_RANGE_STAGE_V648 = ReverseCurriculumStageV26(
    index=LONG_RANGE_STAGE_INDEX_V648,
    name="v648_long_range_obstacle_free",
    block_target_distance_range_m=(0.205, 0.230),
    tip_gap_range_m=(0.0005, 0.0120),
    obstacle_probability=0.0,
    stress_probability=0.0,
    minimum_strict_success_rate=0.65,
    minimum_wilson_lower_bound=0.45,
)


EXTENDED_RANGE_STAGE_V648 = ReverseCurriculumStageV26(
    index=EXTENDED_RANGE_STAGE_INDEX_V648,
    name="v648_extended_range_obstacle_free",
    block_target_distance_range_m=(0.230, 0.255),
    tip_gap_range_m=(0.0005, 0.0120),
    obstacle_probability=0.0,
    stress_probability=0.0,
    minimum_strict_success_rate=0.60,
    minimum_wilson_lower_bound=0.40,
)


def reverse_curriculum_stage_v26(index: int) -> ReverseCurriculumStageV26:
    """Return one exact stage and reject bool/negative/out-of-range aliases."""

    if type(index) is not int:
        raise ValueError("V26 reverse-curriculum stage index is invalid")
    if index == PRETERMINAL_BOOTSTRAP_STAGE_INDEX_V29:
        return PRETERMINAL_BOOTSTRAP_STAGE_V29
    if index == CONTACT_SETTLE_BOOTSTRAP_STAGE_INDEX_V34:
        return CONTACT_SETTLE_BOOTSTRAP_STAGE_V34
    if index == GEOMETRY_BRIDGE_STAGE_INDEX_V509:
        return GEOMETRY_BRIDGE_STAGE_V509
    if index == GEOMETRY_FULLPUSH_STAGE_INDEX_V509:
        return GEOMETRY_FULLPUSH_STAGE_V509
    if index == LONG_RANGE_STAGE_INDEX_V648:
        return LONG_RANGE_STAGE_V648
    if index == EXTENDED_RANGE_STAGE_INDEX_V648:
        return EXTENDED_RANGE_STAGE_V648
    if not 0 <= index < len(REVERSE_CURRICULUM_STAGES_V26):
        raise ValueError("V26 reverse-curriculum stage index is invalid")
    return REVERSE_CURRICULUM_STAGES_V26[index]


def evaluation_condition_schedule_v26(
    stage: ReverseCurriculumStageV26,
    episodes: int,
) -> tuple[tuple[bool, bool], ...]:
    """Build a deterministic held-out schedule matching one stage's mixture.

    Obstacle and stress marginals are rounded to the nearest realizable count.
    Two independent, versioned permutations distribute the conditions across
    the episode sequence so that a same-seed parent/candidate comparison sees
    exactly the same task-condition identity.
    """

    stage.validate()
    if type(episodes) is not int or episodes < 1:
        raise ValueError("V26 evaluation episode count must be positive")
    obstacle_count = int(round(stage.obstacle_probability * episodes))
    stress_count = int(round(stage.stress_probability * episodes))
    obstacle = np.zeros(episodes, dtype=bool)
    stress = np.zeros(episodes, dtype=bool)
    obstacle_rng = np.random.default_rng(0x26A000 + stage.index * 101 + episodes)
    stress_rng = np.random.default_rng(0x26B000 + stage.index * 103 + episodes)
    if obstacle_count:
        obstacle[obstacle_rng.permutation(episodes)[:obstacle_count]] = True
    if stress_count:
        stress[stress_rng.permutation(episodes)[:stress_count]] = True
    schedule = tuple(
        (bool(obstacle[index]), bool(stress[index]))
        for index in range(episodes)
    )
    if sum(int(item[0]) for item in schedule) != obstacle_count:
        raise RuntimeError("V26 obstacle schedule count changed")
    if sum(int(item[1]) for item in schedule) != stress_count:
        raise RuntimeError("V26 stress schedule count changed")
    return schedule


def wilson_lower_bound_v26(successes: int, episodes: int, *, z: float = 1.96) -> float:
    """Compute a two-sided Wilson interval lower endpoint for Bernoulli success."""

    if type(successes) is not int or type(episodes) is not int:
        raise TypeError("V26 Wilson counts must be integers")
    if episodes < 1 or successes < 0 or successes > episodes:
        raise ValueError("V26 Wilson counts are invalid")
    if not np.isfinite(z) or z <= 0.0:
        raise ValueError("V26 Wilson z must be positive")
    probability = successes / episodes
    denominator = 1.0 + z * z / episodes
    center = probability + z * z / (2.0 * episodes)
    radius = z * math.sqrt(
        probability * (1.0 - probability) / episodes
        + z * z / (4.0 * episodes * episodes)
    )
    return float((center - radius) / denominator)


def curriculum_promotion_gate_v26(
    stage: ReverseCurriculumStageV26,
    *,
    evaluation_episodes: int,
    strict_success_episodes: int,
    contact_episodes: int,
    invalid_contact_transitions: int,
    contact_transitions: int,
    shield_terminal_episodes: int,
    safety_violation_episodes: int,
    observed_fps: int,
    observed_hold_steps: int,
    expert_calls: int,
    behavior_cloning_steps: int,
) -> dict[str, Any]:
    """Gate curriculum expansion without trading task progress for safety."""

    stage.validate()
    integer_counts = (
        evaluation_episodes,
        strict_success_episodes,
        contact_episodes,
        invalid_contact_transitions,
        contact_transitions,
        shield_terminal_episodes,
        safety_violation_episodes,
        observed_hold_steps,
        expert_calls,
        behavior_cloning_steps,
    )
    if any(type(value) is not int or value < 0 for value in integer_counts):
        raise ValueError("V26 promotion evidence counts must be non-negative integers")
    if type(observed_fps) is not int or observed_fps < 1:
        raise ValueError("V26 promotion evidence fps is invalid")
    for count in (
        strict_success_episodes,
        contact_episodes,
        shield_terminal_episodes,
        safety_violation_episodes,
    ):
        if count > evaluation_episodes:
            raise ValueError("V26 episode evidence exceeds evaluation episode count")
    if invalid_contact_transitions > contact_transitions:
        raise ValueError("V26 invalid contacts exceed all contact transitions")
    success_rate = strict_success_episodes / max(evaluation_episodes, 1)
    contact_fraction = contact_episodes / max(evaluation_episodes, 1)
    invalid_fraction = invalid_contact_transitions / max(contact_transitions, 1)
    shield_fraction = shield_terminal_episodes / max(evaluation_episodes, 1)
    wilson = (
        wilson_lower_bound_v26(strict_success_episodes, evaluation_episodes)
        if evaluation_episodes > 0
        else 0.0
    )
    required_hold_steps = strict_success_hold_steps_v26(observed_fps)
    checks = {
        "minimum_episode_count": evaluation_episodes >= stage.minimum_evaluation_episodes,
        "strict_success_rate": success_rate >= stage.minimum_strict_success_rate,
        "strict_success_wilson_lower_bound": wilson >= stage.minimum_wilson_lower_bound,
        "contact_episode_fraction": contact_fraction >= stage.minimum_contact_episode_fraction,
        "invalid_contact_fraction": invalid_fraction <= stage.maximum_invalid_contact_fraction,
        "shield_terminal_fraction": shield_fraction <= stage.maximum_shield_terminal_fraction,
        "zero_safety_violation_episodes": safety_violation_episodes == 0,
        "three_second_hold_contract": observed_hold_steps == required_hold_steps,
        "scratch_rl_only": expert_calls == 0 and behavior_cloning_steps == 0,
    }
    return {
        "format": REVERSE_CURRICULUM_FORMAT_V26,
        "stage": asdict(stage),
        "strict_success_hold_seconds": STRICT_SUCCESS_HOLD_SECONDS_V26,
        "required_hold_steps": required_hold_steps,
        "observed_hold_steps": observed_hold_steps,
        "metrics": {
            "strict_success_rate": success_rate,
            "strict_success_wilson_lower_bound": wilson,
            "contact_episode_fraction": contact_fraction,
            "invalid_contact_fraction": invalid_fraction,
            "shield_terminal_fraction": shield_fraction,
        },
        "checks": checks,
        "promotion_allowed": all(checks.values()),
        "production_admission": False,
    }


for _stage_index, _stage in enumerate(REVERSE_CURRICULUM_STAGES_V26):
    _stage.validate()
    if _stage.index != _stage_index:
        raise RuntimeError("V26 curriculum stage indices are not contiguous")
PRETERMINAL_BOOTSTRAP_STAGE_V29.validate()
CONTACT_SETTLE_BOOTSTRAP_STAGE_V34.validate()
GEOMETRY_BRIDGE_STAGE_V509.validate()
GEOMETRY_FULLPUSH_STAGE_V509.validate()
LONG_RANGE_STAGE_V648.validate()
EXTENDED_RANGE_STAGE_V648.validate()


__all__ = [
    "CONTACT_SETTLE_BOOTSTRAP_STAGE_INDEX_V34",
    "CONTACT_SETTLE_BOOTSTRAP_STAGE_V34",
    "GEOMETRY_BRIDGE_STAGE_INDEX_V509",
    "GEOMETRY_BRIDGE_STAGE_V509",
    "GEOMETRY_FULLPUSH_STAGE_INDEX_V509",
    "GEOMETRY_FULLPUSH_STAGE_V509",
    "LONG_RANGE_STAGE_INDEX_V648",
    "LONG_RANGE_STAGE_V648",
    "EXTENDED_RANGE_STAGE_INDEX_V648",
    "EXTENDED_RANGE_STAGE_V648",
    "NOMINAL_CONTROL_FPS_V26",
    "PRETERMINAL_BOOTSTRAP_STAGE_INDEX_V29",
    "PRETERMINAL_BOOTSTRAP_STAGE_V29",
    "REVERSE_CURRICULUM_FORMAT_V26",
    "REVERSE_CURRICULUM_STAGES_V26",
    "STRICT_SUCCESS_HOLD_SECONDS_V26",
    "STRICT_SUCCESS_HOLD_STEPS_V26",
    "ReverseCurriculumStageV26",
    "ReverseCurriculumTaskV26",
    "curriculum_promotion_gate_v26",
    "evaluation_condition_schedule_v26",
    "reverse_curriculum_stage_v26",
    "sample_reverse_curriculum_task_v26",
    "strict_success_hold_steps_v26",
    "wilson_lower_bound_v26",
]
