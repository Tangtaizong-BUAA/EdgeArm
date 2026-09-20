"""Isolated Reward V3 candidate for exact stock-gripper V9 scratch PPO.

This module intentionally leaves the published Reward V1/V2 implementation and
runner byte-for-byte untouched.  V3 keeps the same random full-six-joint actor,
critic, and PPO optimizer, but replaces the weak worst-tip planar potential with
an exact-V9, role-selective 3-D contact-approach potential.  Solver contact
telemetry is recorded only as rollout diagnostics and is never a reward input.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
import torch

from .ppo_utils_v1 import sample_squashed_gaussian_v1, state_dict_sha256_v1
from .privileged_effect_state_v1 import (
    PRIVILEGED_EFFECT_STATE_DIM,
    PRIVILEGED_EFFECT_STATE_LAYOUT_V1,
    PRIVILEGED_EFFECT_STATE_SCHEMA_SHA256,
    build_privileged_effect_state_v1,
)
from .production_env import (
    STOCK_GRIPPER_FIXED_SAFETY_CONVEX_PREFIX,
    STOCK_GRIPPER_MOVING_SAFETY_CONVEX_PREFIX,
)
from .scratch_ppo_v1 import (
    ACTION_DIM,
    ACTOR_ARCHITECTURE,
    CHECKPOINT_SCHEMA_VERSION,
    CRITIC_ARCHITECTURE,
    POLICY_PARAMETERIZATION,
    SOURCE_TYPE,
    FullActionScratchActorV1,
    PPOUpdateMetricsV1,
    PrivilegedEffectCriticV1,
    ScratchPPOConfigV1,
    ScratchPotentialTransitionV1,
    ScratchRolloutBatchV1,
    ppo_update_v1,
)
from .sim2real_env_v9 import RealisticEdgeArmEnvV9, RealisticEnvV9Config


POTENTIAL_REWARD_V3_CANDIDATE_VERSION = (
    "edgearm-v9-role-selective-contact-progress-potential-reward-v3-candidate"
)
POTENTIAL_REWARD_V3_CANDIDATE_FORMULA = (
    "12*coverage+6*(1-clip(block_target_distance/0.20))"
    "+10*(1-softmin(0.70*clip(positive_tip_block_gap/0.08)"
    "+0.30*clip(per_tip_precontact_xy_error/0.08),temperature=0.08))"
    "-8*clip(max(runtime_desk_clearance-min_full_safety_desk_distance,0)/0.02)"
)
CHECKPOINT_FORMAT_V3_CANDIDATE = (
    "edgearm-realism-v9-full-action-ppo-from-scratch-v3-candidate"
)
TRAINING_GENESIS_V3_CANDIDATE_VERSION = (
    "edgearm-full-action-scratch-ppo-genesis-v3-candidate"
)
ROLLOUT_DIAGNOSTICS_V3_CANDIDATE_FORMAT = (
    "edgearm-scratch-ppo-rollout-contact-progress-diagnostics-v3-candidate"
)
_TIP_ROLES = ("fixed_tip", "moving_tip")
_EXPECTED_SAFETY_GEOM_COUNT = 96
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


def _canonical_sha256(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class ScratchPotentialRewardV3CandidateConfig:
    """Frozen coefficients for the isolated V3 admission candidate."""

    coverage_coefficient: float = 12.0
    normalized_block_progress_coefficient: float = 6.0
    contact_approach_progress_coefficient: float = 10.0
    safety_clearance_deficit_coefficient: float = 8.0
    block_target_distance_scale_m: float = 0.20
    desired_tip_block_gap_m: float = 0.0
    tip_block_gap_scale_m: float = 0.08
    precontact_gap_m: float = 0.010
    tip_precontact_xy_scale_m: float = 0.08
    tip_block_gap_role_weight: float = 0.70
    tip_precontact_xy_role_weight: float = 0.30
    role_softmin_temperature: float = 0.08
    safety_clearance_deficit_scale_m: float = 0.02

    def validate(self) -> None:
        canonical = type(self)()
        for field in fields(self):
            value = getattr(self, field.name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not np.isfinite(value)
            ):
                raise ValueError(f"V3 candidate reward {field.name} must be finite numeric")
            if float(value) != float(getattr(canonical, field.name)):
                raise ValueError(
                    f"V3 candidate reward {field.name} is frozen at "
                    f"{getattr(canonical, field.name)}"
                )
        if not math.isclose(
            self.tip_block_gap_role_weight + self.tip_precontact_xy_role_weight,
            1.0,
            rel_tol=0.0,
            abs_tol=1.0e-12,
        ):
            raise ValueError("V3 candidate role weights must sum to one")

    def sha256(self) -> str:
        self.validate()
        return _canonical_sha256(
            {
                "version": POTENTIAL_REWARD_V3_CANDIDATE_VERSION,
                "formula": POTENTIAL_REWARD_V3_CANDIDATE_FORMULA,
                "environment_binding": {
                    "environment_class": "RealisticEdgeArmEnvV9",
                    "config_class": "RealisticEnvV9Config",
                    "planning_geometry_mode": "per_jaw_distal_tip_references",
                    "safety_geometry_mode": "coacd_convex_union",
                    "ordered_tip_roles": list(_TIP_ROLES),
                    "complete_safety_geom_count": _EXPECTED_SAFETY_GEOM_COUNT,
                },
                "reward_inputs": (
                    "current_exact_v9_geometry_only_no_contact_telemetry"
                ),
                "aggregation": {
                    "contact_role": "stable_mean_normalized_softmin",
                    "safety": "minimum_signed_distance_over_complete_union",
                    "transition": "env_reward+gamma*phi_next-phi_current",
                },
                "config": asdict(self),
            }
        )


def potential_reward_v3_candidate_config_from_dict(
    payload: object,
) -> ScratchPotentialRewardV3CandidateConfig:
    expected = {field.name for field in fields(ScratchPotentialRewardV3CandidateConfig)}
    if not isinstance(payload, dict) or set(payload) != expected:
        raise ValueError("V3 candidate reward config fields are incomplete or unexpected")
    try:
        config = ScratchPotentialRewardV3CandidateConfig(**payload)
        config.validate()
    except (TypeError, ValueError) as error:
        raise ValueError("V3 candidate reward config is invalid") from error
    if asdict(config) != payload:
        raise ValueError("V3 candidate reward config values are non-canonical")
    return config


@dataclass(frozen=True)
class ScratchPotentialEvaluationV3Candidate:
    potential: float
    coverage: float
    normalized_block_progress: float
    block_target_distance_m: float
    tip_roles: tuple[str, str]
    tip_block_signed_distance_m: tuple[float, float]
    positive_tip_block_gap_m: tuple[float, float]
    normalized_tip_block_gap: tuple[float, float]
    precontact_xy_by_tip_m: tuple[tuple[float, float], tuple[float, float]]
    tip_precontact_xy_error_m: tuple[float, float]
    normalized_tip_precontact_xy_error: tuple[float, float]
    role_cost: tuple[float, float]
    selected_role_index: int
    selected_role: str
    softmin_role_cost: float
    contact_approach_progress: float
    safety_geom_roles: tuple[str, ...]
    safety_desk_signed_distance_m: tuple[float, ...]
    minimum_safety_desk_signed_distance_m: float
    safety_clearance_deficit_m: float
    normalized_safety_clearance_deficit: float
    runtime_desk_clearance_m: float


def _named_geom_ids_with_prefix(
    env: RealisticEdgeArmEnvV9,
    prefix: str,
) -> tuple[int, ...]:
    matching = [
        geom_id
        for geom_id in range(env.model.ngeom)
        if (mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or "").startswith(
            prefix
        )
    ]
    return tuple(
        sorted(
            matching,
            key=lambda geom_id: mujoco.mj_id2name(
                env.model, mujoco.mjtObj.mjOBJ_GEOM, geom_id
            )
            or "",
        )
    )


def _exact_v9_geometry_contract(
    env: object,
) -> tuple[
    RealisticEdgeArmEnvV9,
    tuple[int, int],
    tuple[int, int],
    tuple[int, ...],
    tuple[str, ...],
]:
    if type(env) is not RealisticEdgeArmEnvV9:
        raise TypeError(
            "V3 candidate reward requires exact RealisticEdgeArmEnvV9, not a base or subclass"
        )
    exact_env = env
    if (
        type(exact_env.contact_feasible_config) is not RealisticEnvV9Config
        or type(exact_env.stock_distal_tip_config) is not RealisticEnvV9Config
    ):
        raise TypeError("V3 candidate reward requires exact RealisticEnvV9Config")
    planning = tuple(int(value) for value in exact_env._ids.get("tool_planning_geoms", ()))
    contacts = tuple(int(value) for value in exact_env._ids.get("tool_contact_geoms", ()))
    safety = tuple(int(value) for value in exact_env._ids.get("tool_safety_geoms", ()))
    tip_roles = tuple(str(value) for value in exact_env._ids.get("tool_contact_geom_roles", ()))
    safety_roles = tuple(
        str(value) for value in exact_env._ids.get("tool_safety_geom_roles", ())
    )
    fixed_safety = _named_geom_ids_with_prefix(
        exact_env, STOCK_GRIPPER_FIXED_SAFETY_CONVEX_PREFIX
    )
    moving_safety = _named_geom_ids_with_prefix(
        exact_env, STOCK_GRIPPER_MOVING_SAFETY_CONVEX_PREFIX
    )
    expected_safety = (
        contacts[0],
        *fixed_safety,
        contacts[1],
        *moving_safety,
    ) if len(contacts) == 2 else ()
    expected_safety_roles = (
        "fixed_jaw_distal_contact",
        *("fixed_jaw_safety" for _ in fixed_safety),
        "moving_jaw_distal_contact",
        *("moving_jaw_safety" for _ in moving_safety),
    )
    if (
        exact_env._ids.get("tool_planning_geometry_mode")
        != "per_jaw_distal_tip_references"
        or exact_env._ids.get("tool_safety_geometry_mode") != "coacd_convex_union"
        or len(planning) != 2
        or len(contacts) != 2
        or tip_roles != _TIP_ROLES
        or safety != expected_safety
        or safety_roles != expected_safety_roles
        or len(safety) != _EXPECTED_SAFETY_GEOM_COUNT
    ):
        raise RuntimeError(
            "V3 candidate reward requires both ordered distal roles and the complete "
            "96-geometry V9 CoACD safety union"
        )
    return exact_env, planning, contacts, safety, safety_roles


def _stable_mean_normalized_softmin(values: np.ndarray, temperature: float) -> float:
    if values.shape != (2,) or not np.all(np.isfinite(values)):
        raise ValueError("V3 candidate role costs must be two finite values")
    minimum = float(np.min(values))
    shifted = np.exp(-(values - minimum) / temperature)
    return float(minimum - temperature * np.log(float(np.mean(shifted))))


class ScratchPotentialRewardV3Candidate:
    """Potential-only V9 geometry reward with role-selective contact approach."""

    version = POTENTIAL_REWARD_V3_CANDIDATE_VERSION
    formula = POTENTIAL_REWARD_V3_CANDIDATE_FORMULA

    def __init__(
        self,
        config: ScratchPotentialRewardV3CandidateConfig | None = None,
    ) -> None:
        self.config = config or ScratchPotentialRewardV3CandidateConfig()
        self.config.validate()
        self.config_sha256 = self.config.sha256()

    def evaluate(
        self,
        env: RealisticEdgeArmEnvV9,
    ) -> ScratchPotentialEvaluationV3Candidate:
        exact_env, planning_geoms, contact_geoms, safety_geoms, safety_roles = (
            _exact_v9_geometry_contract(env)
        )
        block_geom = int(exact_env._ids["block_geom"])
        desk_geom = int(exact_env._desk_geom)
        block_xy = np.asarray(exact_env.block_xy(), dtype=np.float64)
        target_xy = np.asarray(exact_env.target_xy, dtype=np.float64)
        target_delta = target_xy - block_xy
        target_distance = float(np.linalg.norm(target_delta))
        if target_distance > 1.0e-12:
            push_direction = target_delta / target_distance
        else:
            tip_mean = np.mean(
                np.asarray(
                    [exact_env.data.geom_xpos[geom_id, :2] for geom_id in planning_geoms],
                    dtype=np.float64,
                ),
                axis=0,
            )
            fallback = block_xy - tip_mean
            fallback_norm = float(np.linalg.norm(fallback))
            push_direction = (
                fallback / fallback_norm
                if fallback_norm > 1.0e-12
                else np.array([1.0, 0.0], dtype=np.float64)
            )
        direction_world = np.array(
            [push_direction[0], push_direction[1], 0.0], dtype=np.float64
        )
        block_rotation = np.asarray(
            exact_env.data.geom_xmat[block_geom], dtype=np.float64
        ).reshape(3, 3)
        block_support = float(
            np.dot(
                np.abs(block_rotation.T @ direction_world),
                np.asarray(exact_env.model.geom_size[block_geom], dtype=np.float64),
            )
        )

        precontact_xy: list[tuple[float, float]] = []
        xy_errors: list[float] = []
        for geom_id in planning_geoms:
            rotation = np.asarray(
                exact_env.data.geom_xmat[geom_id], dtype=np.float64
            ).reshape(3, 3)
            support = float(
                np.dot(
                    np.abs(rotation.T @ direction_world),
                    np.asarray(exact_env.model.geom_size[geom_id], dtype=np.float64),
                )
            )
            desired_xy = block_xy - push_direction * (
                block_support + support + self.config.precontact_gap_m
            )
            tip_xy = np.asarray(exact_env.data.geom_xpos[geom_id, :2], dtype=np.float64)
            precontact_xy.append((float(desired_xy[0]), float(desired_xy[1])))
            xy_errors.append(float(np.linalg.norm(tip_xy - desired_xy)))

        tip_block_distances = np.asarray(
            [
                exact_env._geom_signed_distance_for_data(
                    geom_id,
                    block_geom,
                    exact_env.data,
                    cutoff_m=1.0,
                )
                for geom_id in contact_geoms
            ],
            dtype=np.float64,
        )
        positive_gaps = np.maximum(
            tip_block_distances - self.config.desired_tip_block_gap_m,
            0.0,
        )
        normalized_gaps = np.clip(
            positive_gaps / self.config.tip_block_gap_scale_m, 0.0, 1.0
        )
        normalized_xy_errors = np.clip(
            np.asarray(xy_errors, dtype=np.float64)
            / self.config.tip_precontact_xy_scale_m,
            0.0,
            1.0,
        )
        role_costs = (
            self.config.tip_block_gap_role_weight * normalized_gaps
            + self.config.tip_precontact_xy_role_weight * normalized_xy_errors
        )
        softmin_cost = _stable_mean_normalized_softmin(
            role_costs, self.config.role_softmin_temperature
        )
        selected_role_index = int(np.argmin(role_costs))
        contact_approach_progress = float(1.0 - np.clip(softmin_cost, 0.0, 1.0))

        safety_distances = np.asarray(
            [
                exact_env._geom_signed_distance_for_data(
                    geom_id,
                    desk_geom,
                    exact_env.data,
                    cutoff_m=1.0,
                )
                for geom_id in safety_geoms
            ],
            dtype=np.float64,
        )
        minimum_safety_distance = float(np.min(safety_distances))
        runtime_clearance = float(
            exact_env.contact_feasible_config.runtime_pusher_desk_clearance_m
        )
        safety_deficit = float(max(runtime_clearance - minimum_safety_distance, 0.0))
        normalized_safety_deficit = float(
            np.clip(
                safety_deficit / self.config.safety_clearance_deficit_scale_m,
                0.0,
                1.0,
            )
        )
        coverage = float(exact_env.block_target_coverage())
        normalized_block_progress = float(
            1.0
            - np.clip(
                target_distance / self.config.block_target_distance_scale_m,
                0.0,
                1.0,
            )
        )
        potential = float(
            self.config.coverage_coefficient * coverage
            + self.config.normalized_block_progress_coefficient
            * normalized_block_progress
            + self.config.contact_approach_progress_coefficient
            * contact_approach_progress
            - self.config.safety_clearance_deficit_coefficient
            * normalized_safety_deficit
        )
        finite_values = np.asarray(
            [
                potential,
                coverage,
                normalized_block_progress,
                target_distance,
                *tip_block_distances,
                *positive_gaps,
                *normalized_gaps,
                *np.asarray(precontact_xy, dtype=np.float64).reshape(-1),
                *xy_errors,
                *normalized_xy_errors,
                *role_costs,
                softmin_cost,
                contact_approach_progress,
                *safety_distances,
                minimum_safety_distance,
                safety_deficit,
                normalized_safety_deficit,
                runtime_clearance,
            ],
            dtype=np.float64,
        )
        if not np.all(np.isfinite(finite_values)):
            raise RuntimeError("V3 candidate reward produced non-finite geometry")
        return ScratchPotentialEvaluationV3Candidate(
            potential=potential,
            coverage=coverage,
            normalized_block_progress=normalized_block_progress,
            block_target_distance_m=target_distance,
            tip_roles=_TIP_ROLES,
            tip_block_signed_distance_m=tuple(
                float(value) for value in tip_block_distances
            ),
            positive_tip_block_gap_m=tuple(float(value) for value in positive_gaps),
            normalized_tip_block_gap=tuple(float(value) for value in normalized_gaps),
            precontact_xy_by_tip_m=tuple(precontact_xy),  # type: ignore[arg-type]
            tip_precontact_xy_error_m=tuple(float(value) for value in xy_errors),
            normalized_tip_precontact_xy_error=tuple(
                float(value) for value in normalized_xy_errors
            ),
            role_cost=tuple(float(value) for value in role_costs),
            selected_role_index=selected_role_index,
            selected_role=_TIP_ROLES[selected_role_index],
            softmin_role_cost=softmin_cost,
            contact_approach_progress=contact_approach_progress,
            safety_geom_roles=safety_roles,
            safety_desk_signed_distance_m=tuple(
                float(value) for value in safety_distances
            ),
            minimum_safety_desk_signed_distance_m=minimum_safety_distance,
            safety_clearance_deficit_m=safety_deficit,
            normalized_safety_clearance_deficit=normalized_safety_deficit,
            runtime_desk_clearance_m=runtime_clearance,
        )

    def shape_transition(
        self,
        *,
        env_reward: float,
        potential_before: float,
        potential_after: float,
        gamma: float,
        terminated: bool,
        truncated: bool,
    ) -> ScratchPotentialTransitionV1:
        values = np.asarray(
            [env_reward, potential_before, potential_after, gamma], dtype=np.float64
        )
        if not np.all(np.isfinite(values)) or not 0.0 < gamma <= 1.0:
            raise ValueError(
                "V3 candidate shaping inputs must be finite and gamma in (0, 1]"
            )
        potential_next = 0.0 if terminated else float(potential_after)
        return ScratchPotentialTransitionV1(
            env_reward=float(env_reward),
            potential_before=float(potential_before),
            potential_after=float(potential_after),
            potential_next_for_shaping=potential_next,
            shaped_reward=float(
                env_reward + gamma * potential_next - potential_before
            ),
            terminated=bool(terminated),
            truncated=bool(truncated),
        )


@dataclass(frozen=True)
class ScratchRolloutProgressDiagnosticsV3Candidate:
    """Transition-aligned V3 geometry/contact telemetry for diagnosis only."""

    coverage_after: np.ndarray
    block_target_distance_after_m: np.ndarray
    tip_block_signed_distance_after_m: np.ndarray
    tip_precontact_xy_error_after_m: np.ndarray
    role_cost_after: np.ndarray
    selected_role_index_after: np.ndarray
    softmin_role_cost_after: np.ndarray
    contact_approach_progress_after: np.ndarray
    minimum_safety_desk_signed_distance_after_m: np.ndarray
    valid_push_side_contact: np.ndarray
    valid_push_side_contact_by_role: np.ndarray
    invalid_tool_block_contact: np.ndarray
    format: str = ROLLOUT_DIAGNOSTICS_V3_CANDIDATE_FORMAT
    tip_roles: tuple[str, str] = _TIP_ROLES
    reward_input: bool = False

    def validate(self, transition_count: int) -> None:
        if type(transition_count) is not int or transition_count < 1:
            raise ValueError("V3 candidate diagnostics transition count is invalid")
        expected_shapes = {
            "coverage_after": (transition_count,),
            "block_target_distance_after_m": (transition_count,),
            "tip_block_signed_distance_after_m": (transition_count, 2),
            "tip_precontact_xy_error_after_m": (transition_count, 2),
            "role_cost_after": (transition_count, 2),
            "selected_role_index_after": (transition_count,),
            "softmin_role_cost_after": (transition_count,),
            "contact_approach_progress_after": (transition_count,),
            "minimum_safety_desk_signed_distance_after_m": (transition_count,),
            "valid_push_side_contact": (transition_count,),
            "valid_push_side_contact_by_role": (transition_count, 2),
            "invalid_tool_block_contact": (transition_count,),
        }
        for name, shape in expected_shapes.items():
            value = np.asarray(getattr(self, name))
            if value.shape != shape:
                raise ValueError(
                    f"V3 candidate diagnostics {name} shape mismatch: {value.shape} != {shape}"
                )
        finite_names = (
            "coverage_after",
            "block_target_distance_after_m",
            "tip_block_signed_distance_after_m",
            "tip_precontact_xy_error_after_m",
            "role_cost_after",
            "softmin_role_cost_after",
            "contact_approach_progress_after",
            "minimum_safety_desk_signed_distance_after_m",
        )
        for name in finite_names:
            if not np.all(np.isfinite(np.asarray(getattr(self, name)))):
                raise ValueError(f"V3 candidate diagnostics {name} is non-finite")
        if self.format != ROLLOUT_DIAGNOSTICS_V3_CANDIDATE_FORMAT:
            raise ValueError("V3 candidate diagnostics format mismatch")
        if self.tip_roles != _TIP_ROLES:
            raise ValueError("V3 candidate diagnostics tip role order changed")
        if self.reward_input is not False:
            raise ValueError("V3 candidate contact diagnostics cannot be a reward input")
        for name in (
            "valid_push_side_contact",
            "valid_push_side_contact_by_role",
            "invalid_tool_block_contact",
        ):
            if np.asarray(getattr(self, name)).dtype != np.dtype(bool):
                raise ValueError(f"V3 candidate diagnostics {name} must be boolean")
        selected = np.asarray(self.selected_role_index_after)
        if not np.issubdtype(selected.dtype, np.integer) or np.any(
            (selected < 0) | (selected >= 2)
        ):
            raise ValueError("V3 candidate selected role indices are invalid")
        if not np.array_equal(selected, np.argmin(self.role_cost_after, axis=1)):
            raise ValueError("V3 candidate selected role indices do not match role costs")
        expected_progress = 1.0 - np.clip(self.softmin_role_cost_after, 0.0, 1.0)
        if not np.allclose(
            self.contact_approach_progress_after,
            expected_progress,
            rtol=0.0,
            atol=1.0e-12,
        ):
            raise ValueError("V3 candidate contact progress does not match softmin cost")
        if np.any((self.coverage_after < 0.0) | (self.coverage_after > 1.0)):
            raise ValueError("V3 candidate coverage is outside [0, 1]")

    def metric_fields(self) -> dict[str, object]:
        transition_count = int(np.asarray(self.coverage_after).size)
        self.validate(transition_count)
        selected_counts = np.bincount(
            np.asarray(self.selected_role_index_after, dtype=np.int64), minlength=2
        )
        valid_by_role = np.count_nonzero(
            self.valid_push_side_contact_by_role, axis=0
        )
        return {
            "rollout_max_coverage": float(np.max(self.coverage_after)),
            "rollout_min_block_target_distance_m": float(
                np.min(self.block_target_distance_after_m)
            ),
            "rollout_min_tip_block_signed_distance_m": float(
                np.min(self.tip_block_signed_distance_after_m)
            ),
            "rollout_min_tip_precontact_xy_error_m": float(
                np.min(self.tip_precontact_xy_error_after_m)
            ),
            "rollout_min_softmin_role_cost": float(
                np.min(self.softmin_role_cost_after)
            ),
            "rollout_max_contact_approach_progress": float(
                np.max(self.contact_approach_progress_after)
            ),
            "rollout_minimum_full_safety_desk_signed_distance_m": float(
                np.min(self.minimum_safety_desk_signed_distance_after_m)
            ),
            "rollout_valid_contact_transition_count": int(
                np.count_nonzero(self.valid_push_side_contact)
            ),
            "rollout_any_valid_contact": bool(np.any(self.valid_push_side_contact)),
            "rollout_valid_contact_transition_count_by_role": {
                role: int(valid_by_role[index])
                for index, role in enumerate(self.tip_roles)
            },
            "rollout_selected_role_transition_count_by_role": {
                role: int(selected_counts[index])
                for index, role in enumerate(self.tip_roles)
            },
            "rollout_invalid_tool_block_contact_transition_count": int(
                np.count_nonzero(self.invalid_tool_block_contact)
            ),
        }


@dataclass(frozen=True)
class ScratchRolloutBatchV3Candidate:
    """A V1-compatible PPO batch plus V3-only diagnostic telemetry."""

    ppo_batch: ScratchRolloutBatchV1
    progress_diagnostics: ScratchRolloutProgressDiagnosticsV3Candidate
    potential_reward_version: str = POTENTIAL_REWARD_V3_CANDIDATE_VERSION

    def __getattr__(self, name: str) -> Any:
        return getattr(self.ppo_batch, name)

    def validate(self) -> None:
        if self.potential_reward_version != POTENTIAL_REWARD_V3_CANDIDATE_VERSION:
            raise ValueError("V3 candidate rollout reward version mismatch")
        self.ppo_batch.validate()
        self.progress_diagnostics.validate(int(self.ppo_batch.shaped_rewards.size))


def _module_device(module: torch.nn.Module) -> torch.device:
    try:
        return next(module.parameters()).device
    except StopIteration as error:  # pragma: no cover
        raise RuntimeError("V3 candidate PPO module has no parameters") from error


def _torch_generator(device: torch.device, seed: int) -> torch.Generator | None:
    if device.type == "mps":
        torch.manual_seed(seed)
        if hasattr(torch, "mps"):
            torch.mps.manual_seed(seed)
        return None
    generator = torch.Generator(device=device.type)
    generator.manual_seed(seed)
    return generator


def _contact_diagnostic_after_step(
    info: dict[str, Any],
) -> tuple[bool, tuple[bool, bool], bool]:
    trace = info.get("physics_substep_contact_v1")
    if not isinstance(trace, dict):
        raise RuntimeError("V3 candidate rollout is missing physics contact telemetry")
    roles = tuple(str(value) for value in trace.get("tool_contact_role_names", ()))
    if roles != _TIP_ROLES:
        raise RuntimeError("V3 candidate contact telemetry role order changed")
    admissible = np.asarray(trace.get("tool_block_contact_admissible_by_role"))
    if admissible.ndim != 2 or admissible.shape[1:] != (2,):
        raise RuntimeError("V3 candidate per-role contact telemetry is malformed")
    valid_by_role = tuple(bool(value) for value in np.any(admissible.astype(bool), axis=0))
    return (
        bool(trace.get("valid_push_side_contact_any", False)),
        valid_by_role,  # type: ignore[return-value]
        bool(trace.get("invalid_tool_block_contact_any", False)),
    )


def collect_scratch_rollout_v3_candidate(
    env: RealisticEdgeArmEnvV9,
    actor: FullActionScratchActorV1,
    critic: PrivilegedEffectCriticV1,
    *,
    steps: int,
    seed: int,
    obstacle_probability: float = 0.50,
    stress_probability: float = 0.30,
    gamma: float = 0.99,
    potential_reward: ScratchPotentialRewardV3Candidate | None = None,
) -> ScratchRolloutBatchV3Candidate:
    """Collect complete episodes while retaining V3 progress diagnostics."""

    _exact_v9_geometry_contract(env)
    if type(steps) is not int or steps < 1 or type(seed) is not int or seed < 0:
        raise ValueError("V3 candidate rollout steps/seed are invalid")
    for name, probability in (
        ("obstacle_probability", obstacle_probability),
        ("stress_probability", stress_probability),
    ):
        if not np.isfinite(probability) or not 0.0 <= probability <= 1.0:
            raise ValueError(f"{name} must be finite and in [0, 1]")
    if not np.isfinite(gamma) or not 0.0 < gamma <= 1.0:
        raise ValueError("V3 candidate rollout gamma must be in (0, 1]")
    reward = potential_reward or ScratchPotentialRewardV3Candidate()
    if type(reward) is not ScratchPotentialRewardV3Candidate:
        raise TypeError("V3 candidate rollout requires the exact V3 reward strategy")
    actor_device = _module_device(actor)
    if _module_device(critic) != actor_device:
        raise ValueError("V3 candidate actor and critic must share a device")
    generator = _torch_generator(actor_device, seed)
    episode_rng = np.random.default_rng(seed ^ 0x5C12A7)

    states: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    pre_tanh: list[np.ndarray] = []
    old_log_probs: list[float] = []
    env_rewards: list[float] = []
    potential_before_values: list[float] = []
    potential_after_values: list[float] = []
    potential_next_values: list[float] = []
    shaped_rewards: list[float] = []
    values: list[float] = []
    next_values: list[float] = []
    terminated_flags: list[bool] = []
    truncated_flags: list[bool] = []
    strict_success_flags: list[bool] = []
    terminal_failure_flags: list[bool] = []
    safety_stop_flags: list[bool] = []
    terminal_reasons: list[str] = []
    episode_ids: list[int] = []
    obstacle_flags: list[bool] = []
    evaluations_after: list[ScratchPotentialEvaluationV3Candidate] = []
    valid_contacts: list[bool] = []
    valid_contacts_by_role: list[tuple[bool, bool]] = []
    invalid_contacts: list[bool] = []

    episode_id = 0
    obstacle = bool(episode_rng.random() < obstacle_probability)
    stress = bool(episode_rng.random() < stress_probability)
    env.reset(seed=seed, obstacle=obstacle, stress=stress)
    actor.eval()
    critic.eval()
    while len(states) < steps or not (terminated_flags[-1] or truncated_flags[-1]):
        evaluation_before = reward.evaluate(env)
        state = build_privileged_effect_state_v1(env)
        state_tensor = torch.from_numpy(state).to(actor_device).unsqueeze(0)
        with torch.no_grad():
            distribution = actor.distribution(state_tensor)
            sample = sample_squashed_gaussian_v1(distribution, generator=generator)
            value = critic(state_tensor).item()
        action = sample.action.squeeze(0).cpu().numpy().astype(np.float32)
        _, env_reward, terminated, truncated, info = env.step(action)
        evaluation_after = reward.evaluate(env)
        shaped = reward.shape_transition(
            env_reward=env_reward,
            potential_before=evaluation_before.potential,
            potential_after=evaluation_after.potential,
            gamma=gamma,
            terminated=bool(terminated),
            truncated=bool(truncated),
        )
        next_state = build_privileged_effect_state_v1(env)
        with torch.no_grad():
            next_value = critic(
                torch.from_numpy(next_state).to(actor_device).unsqueeze(0)
            ).item()

        states.append(state)
        actions.append(action)
        pre_tanh.append(sample.pre_tanh.squeeze(0).cpu().numpy().astype(np.float32))
        old_log_probs.append(float(sample.log_prob.item()))
        env_rewards.append(shaped.env_reward)
        potential_before_values.append(shaped.potential_before)
        potential_after_values.append(shaped.potential_after)
        potential_next_values.append(shaped.potential_next_for_shaping)
        shaped_rewards.append(shaped.shaped_reward)
        values.append(float(value))
        next_values.append(float(next_value))
        terminated_flags.append(bool(terminated))
        truncated_flags.append(bool(truncated))
        strict_success = bool(info.get("success", False))
        safety_stop = bool(info.get("safety_stop"))
        terminal_failure = bool(
            info.get("terminal_failure", bool(terminated and not strict_success))
        )
        terminal_reason = str(info.get("terminal_reason", ""))
        if not terminal_reason:
            if strict_success:
                terminal_reason = "strict_success"
            elif safety_stop:
                terminal_reason = f"safety_stop:{info['safety_stop']}"
            elif terminal_failure:
                terminal_reason = "terminal_failure"
            elif truncated:
                terminal_reason = "time_limit"
            else:
                terminal_reason = "nonterminal"
        strict_success_flags.append(strict_success)
        terminal_failure_flags.append(terminal_failure)
        safety_stop_flags.append(safety_stop)
        terminal_reasons.append(terminal_reason)
        episode_ids.append(episode_id)
        obstacle_flags.append(bool(env.obstacle_enabled))
        evaluations_after.append(evaluation_after)
        valid_contact, valid_by_role, invalid_contact = _contact_diagnostic_after_step(
            info
        )
        valid_contacts.append(valid_contact)
        valid_contacts_by_role.append(valid_by_role)
        invalid_contacts.append(invalid_contact)

        if (terminated or truncated) and len(states) < steps:
            episode_id += 1
            obstacle = bool(episode_rng.random() < obstacle_probability)
            stress = bool(episode_rng.random() < stress_probability)
            env.reset(seed=seed + episode_id, obstacle=obstacle, stress=stress)

    ppo_batch = ScratchRolloutBatchV1(
        states=np.asarray(states, dtype=np.float32),
        actions=np.asarray(actions, dtype=np.float32),
        pre_tanh=np.asarray(pre_tanh, dtype=np.float32),
        old_log_probs=np.asarray(old_log_probs, dtype=np.float32),
        env_rewards=np.asarray(env_rewards, dtype=np.float32),
        potential_before=np.asarray(potential_before_values, dtype=np.float32),
        potential_after=np.asarray(potential_after_values, dtype=np.float32),
        potential_next_for_shaping=np.asarray(potential_next_values, dtype=np.float32),
        shaped_rewards=np.asarray(shaped_rewards, dtype=np.float32),
        values=np.asarray(values, dtype=np.float32),
        next_values=np.asarray(next_values, dtype=np.float32),
        terminated=np.asarray(terminated_flags, dtype=bool),
        truncated=np.asarray(truncated_flags, dtype=bool),
        strict_success=np.asarray(strict_success_flags, dtype=bool),
        terminal_failure=np.asarray(terminal_failure_flags, dtype=bool),
        safety_stop=np.asarray(safety_stop_flags, dtype=bool),
        terminal_reason=np.asarray(terminal_reasons, dtype=str),
        episode_ids=np.asarray(episode_ids, dtype=np.int64),
        obstacle_enabled=np.asarray(obstacle_flags, dtype=bool),
        shaping_gamma=float(gamma),
        potential_reward_config_sha256=reward.config_sha256,
    )
    diagnostics = ScratchRolloutProgressDiagnosticsV3Candidate(
        coverage_after=np.asarray(
            [evaluation.coverage for evaluation in evaluations_after],
            dtype=np.float64,
        ),
        block_target_distance_after_m=np.asarray(
            [evaluation.block_target_distance_m for evaluation in evaluations_after],
            dtype=np.float64,
        ),
        tip_block_signed_distance_after_m=np.asarray(
            [evaluation.tip_block_signed_distance_m for evaluation in evaluations_after],
            dtype=np.float64,
        ),
        tip_precontact_xy_error_after_m=np.asarray(
            [evaluation.tip_precontact_xy_error_m for evaluation in evaluations_after],
            dtype=np.float64,
        ),
        role_cost_after=np.asarray(
            [evaluation.role_cost for evaluation in evaluations_after],
            dtype=np.float64,
        ),
        selected_role_index_after=np.asarray(
            [evaluation.selected_role_index for evaluation in evaluations_after],
            dtype=np.int64,
        ),
        softmin_role_cost_after=np.asarray(
            [evaluation.softmin_role_cost for evaluation in evaluations_after],
            dtype=np.float64,
        ),
        contact_approach_progress_after=np.asarray(
            [evaluation.contact_approach_progress for evaluation in evaluations_after],
            dtype=np.float64,
        ),
        minimum_safety_desk_signed_distance_after_m=np.asarray(
            [
                evaluation.minimum_safety_desk_signed_distance_m
                for evaluation in evaluations_after
            ],
            dtype=np.float64,
        ),
        valid_push_side_contact=np.asarray(valid_contacts, dtype=bool),
        valid_push_side_contact_by_role=np.asarray(valid_contacts_by_role, dtype=bool),
        invalid_tool_block_contact=np.asarray(invalid_contacts, dtype=bool),
    )
    batch = ScratchRolloutBatchV3Candidate(
        ppo_batch=ppo_batch,
        progress_diagnostics=diagnostics,
    )
    batch.validate()
    return batch


def scratch_v3_candidate_source_hashes() -> dict[str, str]:
    directory = Path(__file__).resolve().parent
    paths = {
        "ppo_utils_v1.py": directory / "ppo_utils_v1.py",
        "privileged_effect_state_v1.py": directory / "privileged_effect_state_v1.py",
        "scratch_ppo_v1.py": directory / "scratch_ppo_v1.py",
        "scratch_ppo_v3_candidate.py": Path(__file__).resolve(),
    }
    return {name: _sha256_file(path) for name, path in paths.items()}


@dataclass(frozen=True)
class ScratchPPOV3CandidateProvenance:
    source_type: str
    checkpoint_format: str
    initialization_seed: int
    random_initialization: bool
    full_six_joint_action: bool
    expert_action_inputs: int
    phase_inputs: int
    behavior_cloning_steps: int
    actor_initial_state_sha256: str
    critic_initial_state_sha256: str
    privileged_state_schema_sha256: str
    potential_reward_version: str
    potential_reward_config_sha256: str
    contact_telemetry_is_reward_input: bool
    trainer_source_hashes: dict[str, str]
    genesis_sha256: str

    def _genesis_payload(self) -> dict[str, Any]:
        payload = asdict(self)
        payload.pop("genesis_sha256")
        return {
            "genesis_version": TRAINING_GENESIS_V3_CANDIDATE_VERSION,
            "actor_architecture": ACTOR_ARCHITECTURE,
            "critic_architecture": CRITIC_ARCHITECTURE,
            "policy_parameterization": POLICY_PARAMETERIZATION,
            **payload,
        }

    def validate(self, *, verify_current_sources: bool = True) -> None:
        exact = {
            "source_type": SOURCE_TYPE,
            "checkpoint_format": CHECKPOINT_FORMAT_V3_CANDIDATE,
            "random_initialization": True,
            "full_six_joint_action": True,
            "expert_action_inputs": 0,
            "phase_inputs": 0,
            "behavior_cloning_steps": 0,
            "privileged_state_schema_sha256": PRIVILEGED_EFFECT_STATE_SCHEMA_SHA256,
            "potential_reward_version": POTENTIAL_REWARD_V3_CANDIDATE_VERSION,
            "contact_telemetry_is_reward_input": False,
        }
        for name, expected in exact.items():
            if getattr(self, name) != expected:
                raise ValueError(f"V3 candidate provenance {name} mismatch")
        if type(self.initialization_seed) is not int or self.initialization_seed < 0:
            raise ValueError("V3 candidate initialization seed is invalid")
        for name in (
            "actor_initial_state_sha256",
            "critic_initial_state_sha256",
            "privileged_state_schema_sha256",
            "potential_reward_config_sha256",
            "genesis_sha256",
        ):
            if _SHA256_PATTERN.fullmatch(str(getattr(self, name))) is None:
                raise ValueError(f"V3 candidate provenance hash is malformed: {name}")
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(self.initialization_seed)
            actor = FullActionScratchActorV1()
            critic = PrivilegedEffectCriticV1()
        if self.actor_initial_state_sha256 != state_dict_sha256_v1(actor.state_dict()):
            raise ValueError("V3 candidate actor genesis is not canonical")
        if self.critic_initial_state_sha256 != state_dict_sha256_v1(critic.state_dict()):
            raise ValueError("V3 candidate critic genesis is not canonical")
        if self.potential_reward_config_sha256 != (
            ScratchPotentialRewardV3CandidateConfig().sha256()
        ):
            raise ValueError("V3 candidate reward config provenance mismatch")
        expected_source_names = {
            "ppo_utils_v1.py",
            "privileged_effect_state_v1.py",
            "scratch_ppo_v1.py",
            "scratch_ppo_v3_candidate.py",
        }
        if set(self.trainer_source_hashes) != expected_source_names or any(
            _SHA256_PATTERN.fullmatch(value) is None
            for value in self.trainer_source_hashes.values()
        ):
            raise ValueError("V3 candidate trainer source hashes are malformed")
        if verify_current_sources and self.trainer_source_hashes != (
            scratch_v3_candidate_source_hashes()
        ):
            raise ValueError("V3 candidate trainer sources changed since genesis")
        if self.genesis_sha256 != _canonical_sha256(self._genesis_payload()):
            raise ValueError("V3 candidate genesis hash mismatch")

    @classmethod
    def from_dict(cls, payload: object) -> ScratchPPOV3CandidateProvenance:
        expected = {field.name for field in fields(cls)}
        if not isinstance(payload, dict) or set(payload) != expected:
            raise ValueError("V3 candidate provenance fields are incomplete or unexpected")
        provenance = cls(**payload)
        provenance.validate()
        return provenance


@dataclass
class ScratchPPOV3CandidateBundle:
    actor: FullActionScratchActorV1
    critic: PrivilegedEffectCriticV1
    provenance: ScratchPPOV3CandidateProvenance
    potential_reward_config: ScratchPotentialRewardV3CandidateConfig


def initialize_scratch_ppo_v3_candidate(
    seed: int,
    *,
    device: str | torch.device = "cpu",
) -> ScratchPPOV3CandidateBundle:
    if type(seed) is not int or seed < 0:
        raise ValueError("V3 candidate seed must be a non-negative integer")
    config = ScratchPotentialRewardV3CandidateConfig()
    config.validate()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        actor = FullActionScratchActorV1()
        critic = PrivilegedEffectCriticV1()
    actor_hash = state_dict_sha256_v1(actor.state_dict())
    critic_hash = state_dict_sha256_v1(critic.state_dict())
    base: dict[str, Any] = {
        "source_type": SOURCE_TYPE,
        "checkpoint_format": CHECKPOINT_FORMAT_V3_CANDIDATE,
        "initialization_seed": seed,
        "random_initialization": True,
        "full_six_joint_action": True,
        "expert_action_inputs": 0,
        "phase_inputs": 0,
        "behavior_cloning_steps": 0,
        "actor_initial_state_sha256": actor_hash,
        "critic_initial_state_sha256": critic_hash,
        "privileged_state_schema_sha256": PRIVILEGED_EFFECT_STATE_SCHEMA_SHA256,
        "potential_reward_version": POTENTIAL_REWARD_V3_CANDIDATE_VERSION,
        "potential_reward_config_sha256": config.sha256(),
        "contact_telemetry_is_reward_input": False,
        "trainer_source_hashes": scratch_v3_candidate_source_hashes(),
    }
    provenance = ScratchPPOV3CandidateProvenance(
        **base,
        genesis_sha256=_canonical_sha256(
            {
                "genesis_version": TRAINING_GENESIS_V3_CANDIDATE_VERSION,
                "actor_architecture": ACTOR_ARCHITECTURE,
                "critic_architecture": CRITIC_ARCHITECTURE,
                "policy_parameterization": POLICY_PARAMETERIZATION,
                **base,
            }
        ),
    )
    provenance.validate()
    return ScratchPPOV3CandidateBundle(
        actor=actor.to(device),
        critic=critic.to(device),
        provenance=provenance,
        potential_reward_config=config,
    )


def train_one_scratch_update_v3_candidate(
    env: RealisticEdgeArmEnvV9,
    bundle: ScratchPPOV3CandidateBundle,
    config: ScratchPPOConfigV1,
    *,
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[
    ScratchRolloutBatchV3Candidate,
    PPOUpdateMetricsV1,
    torch.optim.Optimizer,
]:
    config.validate()
    bundle.provenance.validate()
    bundle.potential_reward_config.validate()
    reward = ScratchPotentialRewardV3Candidate(bundle.potential_reward_config)
    rollout = collect_scratch_rollout_v3_candidate(
        env,
        bundle.actor,
        bundle.critic,
        steps=config.rollout_steps,
        seed=config.seed,
        obstacle_probability=config.obstacle_probability,
        stress_probability=config.stress_probability,
        gamma=config.gamma,
        potential_reward=reward,
    )
    metrics, optimizer = ppo_update_v1(
        bundle.actor,
        bundle.critic,
        rollout,  # type: ignore[arg-type]
        config,
        optimizer=optimizer,
    )
    return rollout, metrics, optimizer


def build_scratch_checkpoint_payload_v3_candidate(
    bundle: ScratchPPOV3CandidateBundle,
    config: ScratchPPOConfigV1,
    *,
    updates_completed: int,
) -> dict[str, Any]:
    bundle.provenance.validate()
    config.validate()
    bundle.potential_reward_config.validate()
    if type(updates_completed) is not int or updates_completed < 0:
        raise ValueError("V3 candidate updates_completed must be non-negative")
    actor_state = bundle.actor.state_dict()
    critic_state = bundle.critic.state_dict()
    return {
        "format": CHECKPOINT_FORMAT_V3_CANDIDATE,
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "source_type": SOURCE_TYPE,
        "actor_architecture": ACTOR_ARCHITECTURE,
        "critic_architecture": CRITIC_ARCHITECTURE,
        "policy_parameterization": POLICY_PARAMETERIZATION,
        "state_dim": PRIVILEGED_EFFECT_STATE_DIM,
        "action_dim": ACTION_DIM,
        "state_layout": list(PRIVILEGED_EFFECT_STATE_LAYOUT_V1),
        "state_schema_sha256": PRIVILEGED_EFFECT_STATE_SCHEMA_SHA256,
        "potential_reward_version": POTENTIAL_REWARD_V3_CANDIDATE_VERSION,
        "potential_reward_formula": POTENTIAL_REWARD_V3_CANDIDATE_FORMULA,
        "potential_reward_config": asdict(bundle.potential_reward_config),
        "potential_reward_config_sha256": bundle.potential_reward_config.sha256(),
        "actor_state": actor_state,
        "critic_state": critic_state,
        "actor_state_sha256": state_dict_sha256_v1(actor_state),
        "critic_state_sha256": state_dict_sha256_v1(critic_state),
        "config": asdict(config),
        "updates_completed": updates_completed,
        "provenance": asdict(bundle.provenance),
        "admission_status": "candidate_not_admitted_for_causal_collection",
    }


def validate_scratch_checkpoint_payload_v3_candidate(
    payload: object,
) -> ScratchPPOV3CandidateProvenance:
    expected_fields = {
        "format",
        "schema_version",
        "source_type",
        "actor_architecture",
        "critic_architecture",
        "policy_parameterization",
        "state_dim",
        "action_dim",
        "state_layout",
        "state_schema_sha256",
        "potential_reward_version",
        "potential_reward_formula",
        "potential_reward_config",
        "potential_reward_config_sha256",
        "actor_state",
        "critic_state",
        "actor_state_sha256",
        "critic_state_sha256",
        "config",
        "updates_completed",
        "provenance",
        "admission_status",
    }
    if not isinstance(payload, dict) or set(payload) != expected_fields:
        raise ValueError("V3 candidate checkpoint fields are incomplete or unexpected")
    exact = {
        "format": CHECKPOINT_FORMAT_V3_CANDIDATE,
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "source_type": SOURCE_TYPE,
        "actor_architecture": ACTOR_ARCHITECTURE,
        "critic_architecture": CRITIC_ARCHITECTURE,
        "policy_parameterization": POLICY_PARAMETERIZATION,
        "state_dim": PRIVILEGED_EFFECT_STATE_DIM,
        "action_dim": ACTION_DIM,
        "state_layout": list(PRIVILEGED_EFFECT_STATE_LAYOUT_V1),
        "state_schema_sha256": PRIVILEGED_EFFECT_STATE_SCHEMA_SHA256,
        "potential_reward_version": POTENTIAL_REWARD_V3_CANDIDATE_VERSION,
        "potential_reward_formula": POTENTIAL_REWARD_V3_CANDIDATE_FORMULA,
        "admission_status": "candidate_not_admitted_for_causal_collection",
    }
    for name, expected in exact.items():
        if payload.get(name) != expected:
            raise ValueError(f"V3 candidate checkpoint {name} mismatch")
    reward_config = potential_reward_v3_candidate_config_from_dict(
        payload.get("potential_reward_config")
    )
    if payload.get("potential_reward_config_sha256") != reward_config.sha256():
        raise ValueError("V3 candidate checkpoint reward config hash mismatch")
    if not isinstance(payload.get("actor_state"), dict) or not isinstance(
        payload.get("critic_state"), dict
    ):
        raise ValueError("V3 candidate checkpoint model states are missing")
    if state_dict_sha256_v1(payload["actor_state"]) != payload.get(
        "actor_state_sha256"
    ):
        raise ValueError("V3 candidate checkpoint actor hash mismatch")
    if state_dict_sha256_v1(payload["critic_state"]) != payload.get(
        "critic_state_sha256"
    ):
        raise ValueError("V3 candidate checkpoint critic hash mismatch")
    config_payload = payload.get("config")
    if not isinstance(config_payload, dict) or set(config_payload) != {
        field.name for field in fields(ScratchPPOConfigV1)
    }:
        raise ValueError("V3 candidate checkpoint PPO config is malformed")
    try:
        config = ScratchPPOConfigV1(**config_payload)
        config.validate()
    except (TypeError, ValueError) as error:
        raise ValueError("V3 candidate checkpoint PPO config is invalid") from error
    if asdict(config) != config_payload:
        raise ValueError("V3 candidate checkpoint PPO config is non-canonical")
    if type(payload.get("updates_completed")) is not int or payload["updates_completed"] < 0:
        raise ValueError("V3 candidate checkpoint update count is invalid")
    provenance = ScratchPPOV3CandidateProvenance.from_dict(payload.get("provenance"))
    if provenance.potential_reward_config_sha256 != reward_config.sha256():
        raise ValueError("V3 candidate checkpoint provenance/config mismatch")
    return provenance


def save_scratch_checkpoint_v3_candidate(
    path: str | Path,
    bundle: ScratchPPOV3CandidateBundle,
    config: ScratchPPOConfigV1,
    *,
    updates_completed: int,
) -> Path:
    destination = Path(path)
    if destination.exists():
        raise FileExistsError(
            f"V3 candidate refuses to overwrite an existing artifact: {destination}"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = build_scratch_checkpoint_payload_v3_candidate(
        bundle, config, updates_completed=updates_completed
    )
    torch.save(payload, destination)
    return destination


def load_scratch_checkpoint_v3_candidate(
    path: str | Path,
    *,
    device: str | torch.device = "cpu",
) -> ScratchPPOV3CandidateBundle:
    payload = torch.load(Path(path), map_location=device, weights_only=True)
    provenance = validate_scratch_checkpoint_payload_v3_candidate(payload)
    reward_config = potential_reward_v3_candidate_config_from_dict(
        payload["potential_reward_config"]
    )
    actor = FullActionScratchActorV1().to(device)
    critic = PrivilegedEffectCriticV1().to(device)
    actor.load_state_dict(payload["actor_state"], strict=True)
    critic.load_state_dict(payload["critic_state"], strict=True)
    return ScratchPPOV3CandidateBundle(
        actor=actor,
        critic=critic,
        provenance=provenance,
        potential_reward_config=reward_config,
    )


__all__ = [
    "CHECKPOINT_FORMAT_V3_CANDIDATE",
    "POTENTIAL_REWARD_V3_CANDIDATE_FORMULA",
    "POTENTIAL_REWARD_V3_CANDIDATE_VERSION",
    "ROLLOUT_DIAGNOSTICS_V3_CANDIDATE_FORMAT",
    "ScratchPPOV3CandidateBundle",
    "ScratchPPOV3CandidateProvenance",
    "ScratchPotentialEvaluationV3Candidate",
    "ScratchPotentialRewardV3Candidate",
    "ScratchPotentialRewardV3CandidateConfig",
    "ScratchRolloutBatchV3Candidate",
    "ScratchRolloutProgressDiagnosticsV3Candidate",
    "build_scratch_checkpoint_payload_v3_candidate",
    "collect_scratch_rollout_v3_candidate",
    "initialize_scratch_ppo_v3_candidate",
    "load_scratch_checkpoint_v3_candidate",
    "potential_reward_v3_candidate_config_from_dict",
    "save_scratch_checkpoint_v3_candidate",
    "scratch_v3_candidate_source_hashes",
    "train_one_scratch_update_v3_candidate",
    "validate_scratch_checkpoint_payload_v3_candidate",
]
