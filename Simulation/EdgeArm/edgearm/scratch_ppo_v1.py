"""Expert-free full-action PPO core with versioned EdgeArm geometry rewards.

The actor emits all six normalized joint commands.  There is no baseline
action, residual composition, warm start, behavior cloning, or expert call in
this module.  Historical V7/V8 runs retain reward V1 exactly; the stock-gripper
V9 plant is explicitly bound to the per-jaw/safety-union reward V2.  Simulator-
only privileged effect state is a teacher input; later collection code can
distill successful teacher executions into wrist-observable ACT/VLA data without
exposing this state to the deployed policy.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from .ppo_utils_v1 import (
    compute_gae_termination_truncation_v1,
    finite_module_parameters_v1,
    sample_squashed_gaussian_v1,
    squashed_gaussian_log_prob_v1,
    state_dict_sha256_v1,
)
from .privileged_effect_state_v1 import (
    PRIVILEGED_EFFECT_STATE_DIM,
    PRIVILEGED_EFFECT_STATE_LAYOUT_V1,
    PRIVILEGED_EFFECT_STATE_SCHEMA_SHA256,
    build_privileged_effect_state_v1,
)
from .sim2real_env_v7 import RealisticEdgeArmEnvV7


SOURCE_TYPE = "synthetic_ppo_from_scratch"
CHECKPOINT_FORMAT = "edgearm-realism-v7-full-action-ppo-from-scratch-v1"
CHECKPOINT_SCHEMA_VERSION = 1
ACTION_DIM = 6
ACTOR_ARCHITECTURE = "mlp_163_256_256_tanh_squashed_gaussian_full_action_v1"
CRITIC_ARCHITECTURE = "mlp_163_256_256_tanh_scalar_value_v1"
POLICY_PARAMETERIZATION = (
    "six_dimensional_normal_latent_then_single_tanh_no_baseline_v1"
)
TRAINING_GENESIS_VERSION = "edgearm-full-action-scratch-ppo-genesis-v1"
POTENTIAL_REWARD_V1_VERSION = "edgearm-v7-source-neutral-potential-reward-v1"
POTENTIAL_REWARD_V1_FORMULA = (
    "8*coverage-4*block_target_distance-1.5*clip(tool_precontact_xy_distance)"
    "-0.5*clip(abs(tool_z-operational_z_ref))"
)
# Public compatibility aliases.  Existing V7/V8 checkpoints and callers must
# continue to name the exact single-geometry reward they were created with.
POTENTIAL_REWARD_VERSION = POTENTIAL_REWARD_V1_VERSION
POTENTIAL_REWARD_FORMULA = POTENTIAL_REWARD_V1_FORMULA
POTENTIAL_REWARD_V2_VERSION = "edgearm-v9-per-jaw-safety-union-potential-reward-v2"
POTENTIAL_REWARD_V2_FORMULA = (
    "8*coverage-4*block_target_distance-1.5*clip(max_per_tip_precontact_xy_error)"
    "-0.5*clip(abs(min_full_safety_desk_distance-runtime_desk_clearance))"
)
CHECKPOINT_FORMAT_V1 = CHECKPOINT_FORMAT
CHECKPOINT_FORMAT_V2 = "edgearm-realism-v9-full-action-ppo-from-scratch-v2"
SUPPORTED_POTENTIAL_REWARD_VERSIONS = (
    POTENTIAL_REWARD_V1_VERSION,
    POTENTIAL_REWARD_V2_VERSION,
)
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


def _canonical_sha256(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class ScratchPotentialRewardV1Config:
    """Fixed coefficients and geometric clip limits for source-neutral shaping."""

    coverage_coefficient: float = 8.0
    block_target_distance_coefficient: float = 4.0
    tool_precontact_xy_coefficient: float = 1.5
    operational_z_error_coefficient: float = 0.5
    precontact_gap_m: float = 0.010
    tool_precontact_xy_clip_m: float = 0.40
    operational_z_error_clip_m: float = 0.25

    def validate(self) -> None:
        required = {
            "coverage_coefficient": 8.0,
            "block_target_distance_coefficient": 4.0,
            "tool_precontact_xy_coefficient": 1.5,
            "operational_z_error_coefficient": 0.5,
        }
        for field in fields(self):
            value = getattr(self, field.name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not np.isfinite(value)
            ):
                raise ValueError(
                    f"potential reward {field.name} must be finite numeric"
                )
        for name, expected in required.items():
            if float(getattr(self, name)) != expected:
                raise ValueError(f"potential reward {name} is fixed at {expected}")
        if not 0.0 <= self.precontact_gap_m <= 0.05:
            raise ValueError("potential reward precontact_gap_m must be in [0, 0.05]")
        if (
            self.tool_precontact_xy_clip_m <= 0.0
            or self.operational_z_error_clip_m <= 0.0
        ):
            raise ValueError("potential reward distance clips must be positive")

    def sha256(self) -> str:
        self.validate()
        return _canonical_sha256(
            {
                "version": POTENTIAL_REWARD_VERSION,
                "formula": POTENTIAL_REWARD_FORMULA,
                "operational_z_reference": (
                    "desk_top_plus_orientation_invariant_tool_half_diagonal_plus_"
                    "runtime_pusher_desk_clearance_m"
                ),
                "precontact_geometry": (
                    "block_support_plus_orientation_invariant_tool_half_diagonal_plus_gap"
                ),
                "config": asdict(self),
            }
        )


@dataclass(frozen=True)
class ScratchPotentialRewardV2Config:
    """Fixed V9 coefficients for per-jaw and full-safety-union shaping."""

    coverage_coefficient: float = 8.0
    block_target_distance_coefficient: float = 4.0
    worst_tip_precontact_xy_coefficient: float = 1.5
    safety_desk_clearance_error_coefficient: float = 0.5
    precontact_gap_m: float = 0.010
    worst_tip_precontact_xy_clip_m: float = 0.40
    safety_desk_clearance_error_clip_m: float = 0.25

    def validate(self) -> None:
        required = {
            "coverage_coefficient": 8.0,
            "block_target_distance_coefficient": 4.0,
            "worst_tip_precontact_xy_coefficient": 1.5,
            "safety_desk_clearance_error_coefficient": 0.5,
        }
        for field in fields(self):
            value = getattr(self, field.name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not np.isfinite(value)
            ):
                raise ValueError(
                    f"V9 potential reward {field.name} must be finite numeric"
                )
        for name, expected in required.items():
            if float(getattr(self, name)) != expected:
                raise ValueError(f"V9 potential reward {name} is fixed at {expected}")
        if not 0.0 <= self.precontact_gap_m <= 0.05:
            raise ValueError(
                "V9 potential reward precontact_gap_m must be in [0, 0.05]"
            )
        if (
            self.worst_tip_precontact_xy_clip_m <= 0.0
            or self.safety_desk_clearance_error_clip_m <= 0.0
        ):
            raise ValueError("V9 potential reward distance clips must be positive")

    def sha256(self) -> str:
        self.validate()
        return _canonical_sha256(
            {
                "version": POTENTIAL_REWARD_V2_VERSION,
                "formula": POTENTIAL_REWARD_V2_FORMULA,
                "environment_binding": {
                    "planning_geometry_mode": "per_jaw_distal_tip_references",
                    "safety_geometry_mode": "coacd_convex_union",
                },
                "operational_height_geometry": (
                    "minimum_signed_distance_from_complete_ordered_coacd_safety_union_to_desk"
                ),
                "precontact_geometry": (
                    "ordered_per_jaw_tip_obb_directional_support_plus_block_support_plus_gap"
                ),
                "aggregation": {
                    "precontact_xy": "maximum_error_across_ordered_tip_references",
                    "safety_desk": "minimum_signed_distance_across_complete_safety_union",
                },
                "config": asdict(self),
            }
        )


def _potential_reward_config_from_dict_v1(
    payload: object,
) -> ScratchPotentialRewardV1Config:
    expected = {field.name for field in fields(ScratchPotentialRewardV1Config)}
    if not isinstance(payload, dict) or set(payload) != expected:
        raise ValueError("potential reward config fields are incomplete or unexpected")
    try:
        config = ScratchPotentialRewardV1Config(**payload)
        config.validate()
    except (TypeError, ValueError) as error:
        raise ValueError("potential reward config is invalid") from error
    if asdict(config) != payload:
        raise ValueError("potential reward config values are non-canonical")
    return config


def _potential_reward_config_from_dict_v2(
    payload: object,
) -> ScratchPotentialRewardV2Config:
    expected = {field.name for field in fields(ScratchPotentialRewardV2Config)}
    if not isinstance(payload, dict) or set(payload) != expected:
        raise ValueError(
            "V9 potential reward config fields are incomplete or unexpected"
        )
    try:
        config = ScratchPotentialRewardV2Config(**payload)
        config.validate()
    except (TypeError, ValueError) as error:
        raise ValueError("V9 potential reward config is invalid") from error
    if asdict(config) != payload:
        raise ValueError("V9 potential reward config values are non-canonical")
    return config


ScratchPotentialRewardConfig = (
    ScratchPotentialRewardV1Config | ScratchPotentialRewardV2Config
)


def potential_reward_config_from_dict_v1(
    version: str,
    payload: object,
) -> ScratchPotentialRewardConfig:
    """Decode a reward config without reinterpreting its geometry version."""

    if version == POTENTIAL_REWARD_V1_VERSION:
        return _potential_reward_config_from_dict_v1(payload)
    if version == POTENTIAL_REWARD_V2_VERSION:
        return _potential_reward_config_from_dict_v2(payload)
    raise ValueError(f"unsupported scratch potential reward version: {version}")


def _potential_reward_formula_v1(version: str) -> str:
    if version == POTENTIAL_REWARD_V1_VERSION:
        return POTENTIAL_REWARD_V1_FORMULA
    if version == POTENTIAL_REWARD_V2_VERSION:
        return POTENTIAL_REWARD_V2_FORMULA
    raise ValueError(f"unsupported scratch potential reward version: {version}")


def _checkpoint_format_for_potential_reward_v1(version: str) -> str:
    if version == POTENTIAL_REWARD_V1_VERSION:
        return CHECKPOINT_FORMAT_V1
    if version == POTENTIAL_REWARD_V2_VERSION:
        return CHECKPOINT_FORMAT_V2
    raise ValueError(f"unsupported scratch potential reward version: {version}")


@dataclass(frozen=True)
class ScratchPotentialEvaluationV1:
    potential: float
    coverage: float
    block_target_distance_m: float
    precontact_xy_m: tuple[float, float]
    tool_precontact_xy_distance_m: float
    clipped_tool_precontact_xy_distance_m: float
    tool_z_error_m: float
    clipped_tool_z_error_m: float
    operational_z_ref_m: float
    desk_top_m: float
    orientation_invariant_tool_half_diagonal_m: float
    runtime_desk_clearance_m: float


@dataclass(frozen=True)
class ScratchPotentialEvaluationV2:
    potential: float
    coverage: float
    block_target_distance_m: float
    tip_roles: tuple[str, ...]
    precontact_xy_by_tip_m: tuple[tuple[float, float], ...]
    tip_directional_support_radius_m: tuple[float, ...]
    tip_precontact_xy_error_m: tuple[float, ...]
    worst_tip_precontact_xy_error_m: float
    clipped_worst_tip_precontact_xy_error_m: float
    safety_geom_roles: tuple[str, ...]
    safety_desk_signed_distance_m: tuple[float, ...]
    minimum_safety_desk_signed_distance_m: float
    safety_desk_clearance_error_m: float
    clipped_safety_desk_clearance_error_m: float
    runtime_desk_clearance_m: float
    desk_top_m: float


@dataclass(frozen=True)
class ScratchPotentialTransitionV1:
    env_reward: float
    potential_before: float
    potential_after: float
    potential_next_for_shaping: float
    shaped_reward: float
    terminated: bool
    truncated: bool


class ScratchPotentialRewardV1:
    """Potential-based dense reward computed only from V7 simulator geometry."""

    version = POTENTIAL_REWARD_V1_VERSION
    formula = POTENTIAL_REWARD_V1_FORMULA

    def __init__(self, config: ScratchPotentialRewardV1Config | None = None) -> None:
        self.config = config or ScratchPotentialRewardV1Config()
        self.config.validate()
        self.config_sha256 = self.config.sha256()

    def evaluate(self, env: RealisticEdgeArmEnvV7) -> ScratchPotentialEvaluationV1:
        if not isinstance(env, RealisticEdgeArmEnvV7):
            raise TypeError("scratch potential reward requires RealisticEdgeArmEnvV7")
        planning_geoms = tuple(
            int(value)
            for value in env._ids.get("tool_planning_geoms", (env._ids["tool_geom"],))
        )
        safety_geoms = tuple(
            int(value) for value in env._ids.get("tool_safety_geoms", planning_geoms)
        )
        if len(planning_geoms) != 1 or len(safety_geoms) != 1:
            raise RuntimeError(
                "scratch potential reward V1 is bound to historical single-geometry V7/V8; "
                "V9 requires ScratchPotentialRewardV2"
            )
        tool_geom = int(env._ids["tool_geom"])
        block_geom = int(env._ids["block_geom"])
        desk_geom = int(env._desk_geom)
        tool_half_diagonal = float(np.linalg.norm(env.model.geom_size[tool_geom]))
        if not np.isfinite(tool_half_diagonal) or tool_half_diagonal <= 0.0:
            raise RuntimeError("V7 tool geometry has no finite positive half-diagonal")

        desk_rotation = env.data.geom_xmat[desk_geom].reshape(3, 3)
        desk_vertical_radius = float(
            np.dot(np.abs(desk_rotation[2]), env.model.geom_size[desk_geom])
        )
        desk_top = float(env.data.geom_xpos[desk_geom, 2] + desk_vertical_radius)
        runtime_clearance = float(
            env.contact_feasible_config.runtime_pusher_desk_clearance_m
        )
        if not np.isfinite(runtime_clearance) or runtime_clearance < 0.0:
            raise RuntimeError(
                "V7 declared runtime desk clearance must be finite and non-negative"
            )
        operational_z_ref = desk_top + tool_half_diagonal + runtime_clearance

        block_xy = np.asarray(env.block_xy(), dtype=np.float64)
        target_xy = np.asarray(env.target_xy, dtype=np.float64)
        tool_position = np.asarray(env.data.geom_xpos[tool_geom], dtype=np.float64)
        direction_delta = target_xy - block_xy
        target_distance = float(np.linalg.norm(direction_delta))
        if not np.isfinite(target_distance):
            raise RuntimeError("V7 block-target distance must be finite")
        if target_distance > 1.0e-12:
            direction = direction_delta / target_distance
        else:
            # At the exact target center the task direction is mathematically
            # undefined.  Keep the terminal potential finite by using the
            # source-neutral tool-to-block approach direction (and a fixed
            # axis only in the fully coincident degenerate case).
            approach = block_xy - tool_position[:2]
            approach_norm = float(np.linalg.norm(approach))
            direction = (
                approach / approach_norm
                if approach_norm > 1.0e-12
                else np.array([1.0, 0.0], dtype=np.float64)
            )
        direction_world = np.array([direction[0], direction[1], 0.0], dtype=np.float64)
        block_rotation = env.data.geom_xmat[block_geom].reshape(3, 3)
        block_support = float(
            np.dot(
                np.abs(block_rotation.T @ direction_world),
                env.model.geom_size[block_geom],
            )
        )
        precontact_standoff = (
            block_support + tool_half_diagonal + self.config.precontact_gap_m
        )
        precontact_xy = block_xy - direction * precontact_standoff
        tool_precontact_distance = float(
            np.linalg.norm(tool_position[:2] - precontact_xy)
        )
        tool_z_error = float(abs(tool_position[2] - operational_z_ref))
        clipped_xy = float(
            np.clip(
                tool_precontact_distance, 0.0, self.config.tool_precontact_xy_clip_m
            )
        )
        clipped_z = float(
            np.clip(tool_z_error, 0.0, self.config.operational_z_error_clip_m)
        )
        coverage = float(env.block_target_coverage())
        potential = (
            self.config.coverage_coefficient * coverage
            - self.config.block_target_distance_coefficient * target_distance
            - self.config.tool_precontact_xy_coefficient * clipped_xy
            - self.config.operational_z_error_coefficient * clipped_z
        )
        values = np.asarray(
            [
                potential,
                coverage,
                target_distance,
                *precontact_xy,
                tool_precontact_distance,
                clipped_xy,
                tool_z_error,
                clipped_z,
                operational_z_ref,
                desk_top,
                tool_half_diagonal,
                runtime_clearance,
            ],
            dtype=np.float64,
        )
        if not np.all(np.isfinite(values)):
            raise RuntimeError("scratch potential reward produced non-finite geometry")
        return ScratchPotentialEvaluationV1(
            potential=float(potential),
            coverage=coverage,
            block_target_distance_m=target_distance,
            precontact_xy_m=(float(precontact_xy[0]), float(precontact_xy[1])),
            tool_precontact_xy_distance_m=tool_precontact_distance,
            clipped_tool_precontact_xy_distance_m=clipped_xy,
            tool_z_error_m=tool_z_error,
            clipped_tool_z_error_m=clipped_z,
            operational_z_ref_m=float(operational_z_ref),
            desk_top_m=desk_top,
            orientation_invariant_tool_half_diagonal_m=tool_half_diagonal,
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
        values = np.asarray([env_reward, potential_before, potential_after, gamma])
        if not np.all(np.isfinite(values)) or not 0.0 < gamma <= 1.0:
            raise ValueError(
                "potential shaping inputs and gamma must be finite, with gamma in (0, 1]"
            )
        potential_next = 0.0 if terminated else float(potential_after)
        shaped_reward = float(env_reward + gamma * potential_next - potential_before)
        return ScratchPotentialTransitionV1(
            env_reward=float(env_reward),
            potential_before=float(potential_before),
            potential_after=float(potential_after),
            potential_next_for_shaping=potential_next,
            shaped_reward=shaped_reward,
            terminated=bool(terminated),
            truncated=bool(truncated),
        )


class ScratchPotentialRewardV2:
    """V9 reward using both distal tips and the complete CoACD safety union."""

    version = POTENTIAL_REWARD_V2_VERSION
    formula = POTENTIAL_REWARD_V2_FORMULA

    def __init__(self, config: ScratchPotentialRewardV2Config | None = None) -> None:
        self.config = config or ScratchPotentialRewardV2Config()
        self.config.validate()
        self.config_sha256 = self.config.sha256()

    def evaluate(self, env: RealisticEdgeArmEnvV7) -> ScratchPotentialEvaluationV2:
        if not isinstance(env, RealisticEdgeArmEnvV7):
            raise TypeError(
                "scratch potential reward V2 requires RealisticEdgeArmEnvV7"
            )
        planning_mode = str(env._ids.get("tool_planning_geometry_mode", ""))
        safety_mode = str(env._ids.get("tool_safety_geometry_mode", ""))
        planning_geoms = tuple(
            int(value) for value in env._ids.get("tool_planning_geoms", ())
        )
        safety_geoms = tuple(
            int(value) for value in env._ids.get("tool_safety_geoms", ())
        )
        if (
            planning_mode != "per_jaw_distal_tip_references"
            or safety_mode != "coacd_convex_union"
            or len(planning_geoms) != 2
            or len(safety_geoms) <= len(planning_geoms)
        ):
            raise RuntimeError(
                "scratch potential reward V2 is bound to V9 per-jaw planning references "
                "and the complete CoACD safety union"
            )
        block_geom = int(env._ids["block_geom"])
        desk_geom = int(env._desk_geom)
        runtime_clearance = float(
            env.contact_feasible_config.runtime_pusher_desk_clearance_m
        )
        if not np.isfinite(runtime_clearance) or runtime_clearance < 0.0:
            raise RuntimeError(
                "V9 declared runtime desk clearance must be finite and non-negative"
            )

        block_xy = np.asarray(env.block_xy(), dtype=np.float64)
        target_xy = np.asarray(env.target_xy, dtype=np.float64)
        tip_positions = np.asarray(
            [env.data.geom_xpos[geom_id] for geom_id in planning_geoms],
            dtype=np.float64,
        )
        direction_delta = target_xy - block_xy
        target_distance = float(np.linalg.norm(direction_delta))
        if not np.isfinite(target_distance):
            raise RuntimeError("V9 block-target distance must be finite")
        if target_distance > 1.0e-12:
            direction = direction_delta / target_distance
        else:
            approach = block_xy - np.mean(tip_positions[:, :2], axis=0)
            approach_norm = float(np.linalg.norm(approach))
            direction = (
                approach / approach_norm
                if approach_norm > 1.0e-12
                else np.array([1.0, 0.0], dtype=np.float64)
            )
        direction_world = np.array([direction[0], direction[1], 0.0], dtype=np.float64)
        block_rotation = np.asarray(
            env.data.geom_xmat[block_geom], dtype=np.float64
        ).reshape(3, 3)
        block_support = float(
            np.dot(
                np.abs(block_rotation.T @ direction_world),
                np.asarray(env.model.geom_size[block_geom], dtype=np.float64),
            )
        )

        tip_supports: list[float] = []
        precontact_positions: list[tuple[float, float]] = []
        tip_errors: list[float] = []
        for geom_id, tip_position in zip(planning_geoms, tip_positions, strict=True):
            tip_rotation = np.asarray(
                env.data.geom_xmat[geom_id], dtype=np.float64
            ).reshape(3, 3)
            tip_support = float(
                np.dot(
                    np.abs(tip_rotation.T @ direction_world),
                    np.asarray(env.model.geom_size[geom_id], dtype=np.float64),
                )
            )
            desired_xy = block_xy - direction * (
                block_support + tip_support + self.config.precontact_gap_m
            )
            tip_supports.append(tip_support)
            precontact_positions.append((float(desired_xy[0]), float(desired_xy[1])))
            tip_errors.append(float(np.linalg.norm(tip_position[:2] - desired_xy)))
        worst_tip_error = float(np.max(tip_errors))
        clipped_tip_error = float(
            np.clip(
                worst_tip_error,
                0.0,
                self.config.worst_tip_precontact_xy_clip_m,
            )
        )

        safety_desk_distances = np.asarray(
            [
                env._geom_signed_distance_for_data(
                    geom_id,
                    desk_geom,
                    env.data,
                    cutoff_m=1.0,
                )
                for geom_id in safety_geoms
            ],
            dtype=np.float64,
        )
        if safety_desk_distances.shape != (len(safety_geoms),):
            raise RuntimeError(
                "V9 safety-desk distance vector does not match the safety union"
            )
        minimum_safety_desk_distance = float(np.min(safety_desk_distances))
        safety_desk_error = float(abs(minimum_safety_desk_distance - runtime_clearance))
        clipped_safety_desk_error = float(
            np.clip(
                safety_desk_error,
                0.0,
                self.config.safety_desk_clearance_error_clip_m,
            )
        )
        desk_rotation = np.asarray(
            env.data.geom_xmat[desk_geom], dtype=np.float64
        ).reshape(3, 3)
        desk_vertical_radius = float(
            np.dot(
                np.abs(desk_rotation[2]),
                np.asarray(env.model.geom_size[desk_geom], dtype=np.float64),
            )
        )
        desk_top = float(env.data.geom_xpos[desk_geom, 2] + desk_vertical_radius)
        coverage = float(env.block_target_coverage())
        potential = (
            self.config.coverage_coefficient * coverage
            - self.config.block_target_distance_coefficient * target_distance
            - self.config.worst_tip_precontact_xy_coefficient * clipped_tip_error
            - self.config.safety_desk_clearance_error_coefficient
            * clipped_safety_desk_error
        )
        values = np.asarray(
            [
                potential,
                coverage,
                target_distance,
                block_support,
                *tip_supports,
                *np.asarray(precontact_positions, dtype=np.float64).reshape(-1),
                *tip_errors,
                worst_tip_error,
                clipped_tip_error,
                *safety_desk_distances,
                minimum_safety_desk_distance,
                safety_desk_error,
                clipped_safety_desk_error,
                runtime_clearance,
                desk_top,
            ],
            dtype=np.float64,
        )
        if not np.all(np.isfinite(values)):
            raise RuntimeError(
                "scratch potential reward V2 produced non-finite geometry"
            )
        tip_roles = tuple(
            str(value) for value in env._ids.get("tool_contact_geom_roles", ())
        )
        safety_roles = tuple(
            str(value) for value in env._ids.get("tool_safety_geom_roles", ())
        )
        if len(tip_roles) != len(planning_geoms) or len(safety_roles) != len(
            safety_geoms
        ):
            raise RuntimeError("V9 ordered tool geometry roles are incomplete")
        return ScratchPotentialEvaluationV2(
            potential=float(potential),
            coverage=coverage,
            block_target_distance_m=target_distance,
            tip_roles=tip_roles,
            precontact_xy_by_tip_m=tuple(precontact_positions),
            tip_directional_support_radius_m=tuple(
                float(value) for value in tip_supports
            ),
            tip_precontact_xy_error_m=tuple(float(value) for value in tip_errors),
            worst_tip_precontact_xy_error_m=worst_tip_error,
            clipped_worst_tip_precontact_xy_error_m=clipped_tip_error,
            safety_geom_roles=safety_roles,
            safety_desk_signed_distance_m=tuple(
                float(value) for value in safety_desk_distances
            ),
            minimum_safety_desk_signed_distance_m=minimum_safety_desk_distance,
            safety_desk_clearance_error_m=safety_desk_error,
            clipped_safety_desk_clearance_error_m=clipped_safety_desk_error,
            runtime_desk_clearance_m=runtime_clearance,
            desk_top_m=desk_top,
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
        values = np.asarray([env_reward, potential_before, potential_after, gamma])
        if not np.all(np.isfinite(values)) or not 0.0 < gamma <= 1.0:
            raise ValueError(
                "potential shaping inputs and gamma must be finite, with gamma in (0, 1]"
            )
        potential_next = 0.0 if terminated else float(potential_after)
        shaped_reward = float(env_reward + gamma * potential_next - potential_before)
        return ScratchPotentialTransitionV1(
            env_reward=float(env_reward),
            potential_before=float(potential_before),
            potential_after=float(potential_after),
            potential_next_for_shaping=potential_next,
            shaped_reward=shaped_reward,
            terminated=bool(terminated),
            truncated=bool(truncated),
        )


ScratchPotentialReward = ScratchPotentialRewardV1 | ScratchPotentialRewardV2


def potential_reward_from_config_v1(
    config: ScratchPotentialRewardConfig,
) -> ScratchPotentialReward:
    """Construct the geometry strategy explicitly committed by ``config``."""

    if isinstance(config, ScratchPotentialRewardV1Config):
        return ScratchPotentialRewardV1(config)
    if isinstance(config, ScratchPotentialRewardV2Config):
        return ScratchPotentialRewardV2(config)
    raise TypeError("unsupported scratch potential reward config type")


class FullActionScratchActorV1(nn.Module):
    """A 163 -> 256 -> 256 actor whose output is the full six-joint action."""

    def __init__(self) -> None:
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(PRIVILEGED_EFFECT_STATE_DIM, 256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh(),
        )
        self.mean_head = nn.Linear(256, ACTION_DIM)
        self.log_std = nn.Parameter(torch.full((ACTION_DIM,), -0.5))

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        if state.shape[-1] != PRIVILEGED_EFFECT_STATE_DIM:
            raise ValueError(f"actor requires {PRIVILEGED_EFFECT_STATE_DIM}-D state")
        return self.mean_head(self.trunk(state))

    def distribution(self, state: torch.Tensor) -> torch.distributions.Normal:
        mean = self(state)
        standard_deviation = self.log_std.clamp(-5.0, 1.0).exp()
        return torch.distributions.Normal(mean, standard_deviation)

    def deterministic(self, state: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self(state))


class PrivilegedEffectCriticV1(nn.Module):
    """A 163 -> 256 -> 256 scalar state-value network."""

    def __init__(self) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(PRIVILEGED_EFFECT_STATE_DIM, 256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh(),
            nn.Linear(256, 1),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        if state.shape[-1] != PRIVILEGED_EFFECT_STATE_DIM:
            raise ValueError(f"critic requires {PRIVILEGED_EFFECT_STATE_DIM}-D state")
        return self.network(state).squeeze(-1)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def trainer_source_hashes_v1() -> dict[str, str]:
    """Hash the complete three-file algorithm/state implementation boundary."""

    directory = Path(__file__).resolve().parent
    paths = {
        "ppo_utils_v1.py": directory / "ppo_utils_v1.py",
        "privileged_effect_state_v1.py": directory / "privileged_effect_state_v1.py",
        "scratch_ppo_v1.py": directory / "scratch_ppo_v1.py",
    }
    return {name: _sha256_file(path) for name, path in paths.items()}


def _canonical_initial_state_hashes_v1(seed: int) -> tuple[str, str]:
    """Recreate the only accepted random genesis for a recorded seed."""

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        actor = FullActionScratchActorV1()
        critic = PrivilegedEffectCriticV1()
    return (
        state_dict_sha256_v1(actor.state_dict()),
        state_dict_sha256_v1(critic.state_dict()),
    )


@dataclass(frozen=True)
class ScratchPPOProvenanceV1:
    """Fail-closed evidence that a checkpoint genesis was expert-free."""

    source_type: str
    checkpoint_format: str
    random_initialization: bool
    initialization_seed: int
    expert_calls: int
    warm_start: bool
    behavior_cloning_steps: int
    actor_initial_state_sha256: str
    critic_initial_state_sha256: str
    privileged_state_schema_sha256: str
    potential_reward_version: str
    potential_reward_config_sha256: str
    trainer_source_hashes: dict[str, str]
    genesis_sha256: str

    def _genesis_payload(self) -> dict[str, Any]:
        payload = asdict(self)
        payload.pop("genesis_sha256")
        return {
            "genesis_version": TRAINING_GENESIS_VERSION,
            "actor_architecture": ACTOR_ARCHITECTURE,
            "critic_architecture": CRITIC_ARCHITECTURE,
            "policy_parameterization": POLICY_PARAMETERIZATION,
            **payload,
        }

    def validate(self, *, verify_current_trainer_sources: bool = True) -> None:
        if self.source_type != SOURCE_TYPE:
            raise ValueError("scratch PPO provenance source_type mismatch")
        if self.potential_reward_version not in SUPPORTED_POTENTIAL_REWARD_VERSIONS:
            raise ValueError("scratch PPO potential reward version mismatch")
        if self.checkpoint_format != _checkpoint_format_for_potential_reward_v1(
            self.potential_reward_version
        ):
            raise ValueError("scratch PPO provenance checkpoint format mismatch")
        if self.random_initialization is not True:
            raise ValueError("scratch PPO genesis must use random initialization")
        if (
            not isinstance(self.initialization_seed, int)
            or self.initialization_seed < 0
        ):
            raise ValueError(
                "scratch PPO initialization seed must be a non-negative integer"
            )
        if self.expert_calls != 0:
            raise ValueError("scratch PPO provenance forbids expert calls")
        if self.warm_start is not False:
            raise ValueError("scratch PPO provenance forbids warm starts")
        if self.behavior_cloning_steps != 0:
            raise ValueError("scratch PPO provenance forbids behavior-cloning steps")
        for name, value in (
            ("actor_initial_state_sha256", self.actor_initial_state_sha256),
            ("critic_initial_state_sha256", self.critic_initial_state_sha256),
            ("privileged_state_schema_sha256", self.privileged_state_schema_sha256),
            ("potential_reward_config_sha256", self.potential_reward_config_sha256),
            ("genesis_sha256", self.genesis_sha256),
        ):
            if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
                raise ValueError(f"invalid provenance hash: {name}")
        canonical_actor_hash, canonical_critic_hash = (
            _canonical_initial_state_hashes_v1(self.initialization_seed)
        )
        if self.actor_initial_state_sha256 != canonical_actor_hash:
            raise ValueError(
                "scratch PPO actor genesis is not the canonical seeded initialization"
            )
        if self.critic_initial_state_sha256 != canonical_critic_hash:
            raise ValueError(
                "scratch PPO critic genesis is not the canonical seeded initialization"
            )
        if self.privileged_state_schema_sha256 != PRIVILEGED_EFFECT_STATE_SCHEMA_SHA256:
            raise ValueError("scratch PPO privileged state schema hash mismatch")
        expected_names = {
            "ppo_utils_v1.py",
            "privileged_effect_state_v1.py",
            "scratch_ppo_v1.py",
        }
        if set(self.trainer_source_hashes) != expected_names:
            raise ValueError("scratch PPO trainer source hash set is incomplete")
        if any(
            _SHA256_PATTERN.fullmatch(value) is None
            for value in self.trainer_source_hashes.values()
        ):
            raise ValueError("scratch PPO trainer source hash is malformed")
        if (
            verify_current_trainer_sources
            and self.trainer_source_hashes != trainer_source_hashes_v1()
        ):
            raise ValueError(
                "scratch PPO trainer sources differ from checkpoint genesis"
            )
        if self.genesis_sha256 != _canonical_sha256(self._genesis_payload()):
            raise ValueError("scratch PPO genesis hash mismatch")

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ScratchPPOProvenanceV1:
        if not isinstance(payload, dict):
            raise TypeError("scratch PPO provenance must be a dictionary")
        expected = {field.name for field in fields(cls)}
        if set(payload) != expected:
            raise ValueError(
                "scratch PPO provenance fields are incomplete or unexpected"
            )
        provenance = cls(**payload)
        provenance.validate()
        return provenance


def _make_provenance_v1(
    *,
    initialization_seed: int,
    actor_initial_state_sha256: str,
    critic_initial_state_sha256: str,
    potential_reward_version: str,
    potential_reward_config_sha256: str,
) -> ScratchPPOProvenanceV1:
    checkpoint_format = _checkpoint_format_for_potential_reward_v1(
        potential_reward_version
    )
    base: dict[str, Any] = {
        "source_type": SOURCE_TYPE,
        "checkpoint_format": checkpoint_format,
        "random_initialization": True,
        "initialization_seed": initialization_seed,
        "expert_calls": 0,
        "warm_start": False,
        "behavior_cloning_steps": 0,
        "actor_initial_state_sha256": actor_initial_state_sha256,
        "critic_initial_state_sha256": critic_initial_state_sha256,
        "privileged_state_schema_sha256": PRIVILEGED_EFFECT_STATE_SCHEMA_SHA256,
        "potential_reward_version": potential_reward_version,
        "potential_reward_config_sha256": potential_reward_config_sha256,
        "trainer_source_hashes": trainer_source_hashes_v1(),
    }
    genesis_payload = {
        "genesis_version": TRAINING_GENESIS_VERSION,
        "actor_architecture": ACTOR_ARCHITECTURE,
        "critic_architecture": CRITIC_ARCHITECTURE,
        "policy_parameterization": POLICY_PARAMETERIZATION,
        **base,
    }
    provenance = ScratchPPOProvenanceV1(
        **base,
        genesis_sha256=_canonical_sha256(genesis_payload),
    )
    provenance.validate()
    return provenance


@dataclass
class ScratchPPOBundleV1:
    actor: FullActionScratchActorV1
    critic: PrivilegedEffectCriticV1
    provenance: ScratchPPOProvenanceV1
    potential_reward_config: ScratchPotentialRewardConfig


def initialize_scratch_ppo_v1(
    seed: int,
    *,
    device: str | torch.device = "cpu",
    potential_reward_config: ScratchPotentialRewardConfig | None = None,
) -> ScratchPPOBundleV1:
    """Create random actor/critic weights without loading any prior model."""

    if not isinstance(seed, int) or seed < 0:
        raise ValueError("scratch PPO seed must be a non-negative integer")
    reward_config = potential_reward_config or ScratchPotentialRewardV1Config()
    if not isinstance(
        reward_config,
        (ScratchPotentialRewardV1Config, ScratchPotentialRewardV2Config),
    ):
        raise TypeError("potential_reward_config must be a supported versioned config")
    reward_config.validate()
    reward_model = potential_reward_from_config_v1(reward_config)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        actor = FullActionScratchActorV1()
        critic = PrivilegedEffectCriticV1()
    actor_hash = state_dict_sha256_v1(actor.state_dict())
    critic_hash = state_dict_sha256_v1(critic.state_dict())
    provenance = _make_provenance_v1(
        initialization_seed=seed,
        actor_initial_state_sha256=actor_hash,
        critic_initial_state_sha256=critic_hash,
        potential_reward_version=reward_model.version,
        potential_reward_config_sha256=reward_model.config_sha256,
    )
    return ScratchPPOBundleV1(
        actor=actor.to(device),
        critic=critic.to(device),
        provenance=provenance,
        potential_reward_config=reward_config,
    )


@dataclass(frozen=True)
class ScratchPPOConfigV1:
    rollout_steps: int = 512
    update_epochs: int = 6
    batch_size: int = 128
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.20
    value_clip_ratio: float = 0.20
    learning_rate: float = 3.0e-4
    entropy_coef: float = 0.002
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: float = 0.025
    obstacle_probability: float = 0.50
    stress_probability: float = 0.30
    seed: int = 7_400_000

    def validate(self) -> None:
        integer_fields = ("rollout_steps", "update_epochs", "batch_size", "seed")
        for name in integer_fields:
            if type(getattr(self, name)) is not int:
                raise ValueError(f"{name} must be an integer")
        for field in fields(self):
            value = getattr(self, field.name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{field.name} must be a finite numeric value")
            if not np.isfinite(value):
                raise ValueError(f"{field.name} must be finite")
        if self.rollout_steps < 1 or self.update_epochs < 1 or self.batch_size < 1:
            raise ValueError(
                "rollout_steps, update_epochs, and batch_size must be positive"
            )
        if not 0.0 < self.gamma <= 1.0 or not 0.0 <= self.gae_lambda <= 1.0:
            raise ValueError("gamma and gae_lambda are outside their valid ranges")
        if not 0.0 < self.clip_ratio < 1.0 or not 0.0 < self.value_clip_ratio < 1.0:
            raise ValueError("policy and value clip ratios must be in (0, 1)")
        if (
            self.learning_rate <= 0.0
            or self.max_grad_norm <= 0.0
            or self.target_kl <= 0.0
        ):
            raise ValueError(
                "learning_rate, max_grad_norm, and target_kl must be positive"
            )
        if self.entropy_coef < 0.0 or self.value_coef < 0.0:
            raise ValueError("entropy_coef and value_coef must be non-negative")
        for name in ("obstacle_probability", "stress_probability"):
            if not 0.0 <= getattr(self, name) <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        if self.seed < 0:
            raise ValueError("seed must be a non-negative integer")


@dataclass(frozen=True)
class ScratchRolloutBatchV1:
    states: np.ndarray
    actions: np.ndarray
    pre_tanh: np.ndarray
    old_log_probs: np.ndarray
    env_rewards: np.ndarray
    potential_before: np.ndarray
    potential_after: np.ndarray
    potential_next_for_shaping: np.ndarray
    shaped_rewards: np.ndarray
    values: np.ndarray
    next_values: np.ndarray
    terminated: np.ndarray
    truncated: np.ndarray
    strict_success: np.ndarray
    terminal_failure: np.ndarray
    safety_stop: np.ndarray
    terminal_reason: np.ndarray
    episode_ids: np.ndarray
    obstacle_enabled: np.ndarray
    shaping_gamma: float
    potential_reward_config_sha256: str
    source_type: str = SOURCE_TYPE

    @property
    def rewards(self) -> np.ndarray:
        """PPO-compatible alias; the stored rewards are explicitly shaped."""

        return self.shaped_rewards

    @property
    def completed_episode_count(self) -> int:
        """Number of complete environment episodes retained in this batch."""

        return int(np.count_nonzero(self.terminated | self.truncated))

    def validate(self) -> None:
        count = int(np.asarray(self.shaped_rewards).size)
        expected_shapes = {
            "states": (count, PRIVILEGED_EFFECT_STATE_DIM),
            "actions": (count, ACTION_DIM),
            "pre_tanh": (count, ACTION_DIM),
            "old_log_probs": (count,),
            "env_rewards": (count,),
            "potential_before": (count,),
            "potential_after": (count,),
            "potential_next_for_shaping": (count,),
            "shaped_rewards": (count,),
            "values": (count,),
            "next_values": (count,),
            "terminated": (count,),
            "truncated": (count,),
            "strict_success": (count,),
            "terminal_failure": (count,),
            "safety_stop": (count,),
            "terminal_reason": (count,),
            "episode_ids": (count,),
            "obstacle_enabled": (count,),
        }
        if count < 1:
            raise ValueError("scratch PPO rollout must contain at least one transition")
        for name, shape in expected_shapes.items():
            value = np.asarray(getattr(self, name))
            if value.shape != shape:
                raise ValueError(
                    f"rollout {name} shape mismatch: {value.shape} != {shape}"
                )
        for name in (
            "states",
            "actions",
            "pre_tanh",
            "old_log_probs",
            "env_rewards",
            "potential_before",
            "potential_after",
            "potential_next_for_shaping",
            "shaped_rewards",
            "values",
            "next_values",
        ):
            if not np.all(np.isfinite(np.asarray(getattr(self, name)))):
                raise ValueError(f"rollout {name} contains non-finite values")
        if np.any(np.abs(self.actions) > 1.0):
            raise ValueError("scratch PPO rollout action exceeds tanh bounds")
        if not np.allclose(
            self.actions,
            np.tanh(self.pre_tanh),
            rtol=1.0e-6,
            atol=1.0e-6,
        ):
            raise ValueError("scratch PPO rollout action is not tanh(pre_tanh)")
        if self.source_type != SOURCE_TYPE:
            raise ValueError("scratch PPO rollout source_type mismatch")
        for name in (
            "terminated",
            "truncated",
            "strict_success",
            "terminal_failure",
            "safety_stop",
            "obstacle_enabled",
        ):
            if np.asarray(getattr(self, name)).dtype != np.dtype(bool):
                raise ValueError(f"rollout {name} must be an explicit boolean vector")
        if np.asarray(self.terminal_reason).dtype.kind not in {"U", "S"}:
            raise ValueError(
                "rollout terminal_reason must be an explicit string vector"
            )
        if np.any(self.terminated & self.truncated):
            raise ValueError(
                "rollout transition cannot be both terminated and truncated"
            )
        boundary = self.terminated | self.truncated
        if not bool(boundary[-1]):
            raise ValueError(
                "scratch PPO rollout must end at a complete episode boundary"
            )
        if np.any(self.strict_success & ~self.terminated):
            raise ValueError(
                "rollout strict success must be a true terminal transition"
            )
        if np.any(self.terminal_failure & ~self.terminated):
            raise ValueError(
                "rollout terminal failure must be a true terminal transition"
            )
        if np.any(self.strict_success & self.terminal_failure):
            raise ValueError(
                "rollout transition cannot be both strict success and terminal failure"
            )
        if np.any(self.safety_stop & ~self.terminal_failure):
            raise ValueError("rollout safety stop must be a terminal failure")
        reasons = np.asarray(self.terminal_reason).astype(str)
        if np.any((~boundary) & (reasons != "nonterminal")):
            raise ValueError(
                "nonterminal rollout transitions must use the nonterminal reason"
            )
        if np.any(boundary & (reasons == "nonterminal")):
            raise ValueError("episode boundaries require an explicit terminal reason")
        if not np.issubdtype(np.asarray(self.episode_ids).dtype, np.integer):
            raise ValueError("rollout episode_ids must be an integer vector")
        episode_ids = np.asarray(self.episode_ids, dtype=np.int64)
        if episode_ids[0] != 0 or np.any(episode_ids < 0):
            raise ValueError(
                "rollout episode_ids must start at zero and stay non-negative"
            )
        if count > 1:
            increments = np.diff(episode_ids)
            expected_increments = (self.terminated[:-1] | self.truncated[:-1]).astype(
                np.int64
            )
            if not np.array_equal(increments, expected_increments):
                raise ValueError("rollout episode_ids do not match terminal boundaries")
        if not np.isfinite(self.shaping_gamma) or not 0.0 < self.shaping_gamma <= 1.0:
            raise ValueError("rollout shaping_gamma must be finite and in (0, 1]")
        if (
            not isinstance(self.potential_reward_config_sha256, str)
            or _SHA256_PATTERN.fullmatch(self.potential_reward_config_sha256) is None
        ):
            raise ValueError("rollout potential reward config hash is malformed")
        expected_next = np.where(self.terminated, 0.0, self.potential_after)
        if not np.allclose(
            self.potential_next_for_shaping,
            expected_next,
            rtol=0.0,
            atol=1.0e-6,
        ):
            raise ValueError(
                "rollout potential termination/truncation bootstrap mismatch"
            )
        expected_reward = (
            self.env_rewards
            + self.shaping_gamma * self.potential_next_for_shaping
            - self.potential_before
        )
        if not np.allclose(
            self.shaped_rewards, expected_reward, rtol=1.0e-6, atol=1.0e-6
        ):
            raise ValueError(
                "rollout shaped reward does not match potential difference"
            )


def _module_device(module: nn.Module) -> torch.device:
    try:
        return next(module.parameters()).device
    except (
        StopIteration
    ) as error:  # pragma: no cover - these networks always have parameters
        raise RuntimeError("scratch PPO module has no parameters") from error


def _torch_generator(device: torch.device, seed: int) -> torch.Generator | None:
    if device.type == "mps":
        # Current PyTorch does not expose a device-local MPS Generator.  Seed
        # both global streams explicitly; CPU remains the deterministic audit
        # path used by tests and formal reproducibility checks.
        torch.manual_seed(seed)
        if hasattr(torch, "mps"):
            torch.mps.manual_seed(seed)
        return None
    generator = torch.Generator(device=device.type)
    generator.manual_seed(seed)
    return generator


def collect_scratch_rollout_v1(
    env: RealisticEdgeArmEnvV7,
    actor: FullActionScratchActorV1,
    critic: PrivilegedEffectCriticV1,
    *,
    steps: int,
    seed: int,
    obstacle_probability: float = 0.50,
    stress_probability: float = 0.30,
    gamma: float = 0.99,
    potential_reward: ScratchPotentialReward | None = None,
) -> ScratchRolloutBatchV1:
    """Collect at least ``steps`` transitions and finish the current episode.

    Every returned batch ends on a true termination or time-limit truncation.
    No partially observed long-horizon episode is discarded when the trainer
    creates a fresh environment for the next PPO update.
    """

    if not isinstance(env, RealisticEdgeArmEnvV7):
        raise TypeError("scratch PPO rollout requires RealisticEdgeArmEnvV7")
    if steps < 1 or seed < 0:
        raise ValueError("rollout steps must be positive and seed non-negative")
    for name, probability in (
        ("obstacle_probability", obstacle_probability),
        ("stress_probability", stress_probability),
    ):
        if not np.isfinite(probability) or not 0.0 <= probability <= 1.0:
            raise ValueError(f"{name} must be finite and in [0, 1]")
    if not np.isfinite(gamma) or not 0.0 < gamma <= 1.0:
        raise ValueError("rollout gamma must be finite and in (0, 1]")
    reward_model = potential_reward or ScratchPotentialRewardV1()
    if not isinstance(
        reward_model, (ScratchPotentialRewardV1, ScratchPotentialRewardV2)
    ):
        raise TypeError(
            "potential_reward must be a supported versioned reward strategy"
        )
    actor_device = _module_device(actor)
    if _module_device(critic) != actor_device:
        raise ValueError("scratch PPO actor and critic must be on the same device")
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
    rewards: list[float] = []
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

    episode_id = 0
    obstacle = bool(episode_rng.random() < obstacle_probability)
    stress = bool(episode_rng.random() < stress_probability)
    env.reset(seed=seed, obstacle=obstacle, stress=stress)
    actor.eval()
    critic.eval()
    while len(states) < steps or not (terminated_flags[-1] or truncated_flags[-1]):
        potential_before = reward_model.evaluate(env).potential
        state = build_privileged_effect_state_v1(env)
        state_tensor = torch.from_numpy(state).to(actor_device).unsqueeze(0)
        with torch.no_grad():
            distribution = actor.distribution(state_tensor)
            sample = sample_squashed_gaussian_v1(distribution, generator=generator)
            value = critic(state_tensor).item()
        action = sample.action.squeeze(0).cpu().numpy().astype(np.float32)
        _, env_reward, terminated, truncated, info = env.step(action)
        potential_after = reward_model.evaluate(env).potential
        shaped = reward_model.shape_transition(
            env_reward=env_reward,
            potential_before=potential_before,
            potential_after=potential_after,
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
        rewards.append(shaped.shaped_reward)
        values.append(float(value))
        next_values.append(float(next_value))
        terminated_flags.append(bool(terminated))
        truncated_flags.append(bool(truncated))
        strict_success = bool(info.get("success", False))
        safety_stop = bool(info.get("safety_stop"))
        terminal_failure = bool(
            info.get(
                "terminal_failure",
                bool(terminated and not strict_success),
            )
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

        if (terminated or truncated) and len(states) < steps:
            episode_id += 1
            obstacle = bool(episode_rng.random() < obstacle_probability)
            stress = bool(episode_rng.random() < stress_probability)
            env.reset(seed=seed + episode_id, obstacle=obstacle, stress=stress)

    batch = ScratchRolloutBatchV1(
        states=np.asarray(states, dtype=np.float32),
        actions=np.asarray(actions, dtype=np.float32),
        pre_tanh=np.asarray(pre_tanh, dtype=np.float32),
        old_log_probs=np.asarray(old_log_probs, dtype=np.float32),
        env_rewards=np.asarray(env_rewards, dtype=np.float32),
        potential_before=np.asarray(potential_before_values, dtype=np.float32),
        potential_after=np.asarray(potential_after_values, dtype=np.float32),
        potential_next_for_shaping=np.asarray(potential_next_values, dtype=np.float32),
        shaped_rewards=np.asarray(rewards, dtype=np.float32),
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
        potential_reward_config_sha256=reward_model.config_sha256,
    )
    batch.validate()
    return batch


@dataclass(frozen=True)
class PPOUpdateMetricsV1:
    optimizer_steps: int
    epochs_completed: int
    target_kl_triggered: bool
    policy_loss: float
    value_loss: float
    entropy: float
    approximate_kl: float
    policy_clip_fraction: float
    value_clip_fraction: float
    maximum_preclip_gradient_norm: float


def ppo_update_v1(
    actor: FullActionScratchActorV1,
    critic: PrivilegedEffectCriticV1,
    batch: ScratchRolloutBatchV1,
    config: ScratchPPOConfigV1,
    *,
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[PPOUpdateMetricsV1, torch.optim.Optimizer]:
    """Run one clipped PPO update, including value clipping and target-KL stop."""

    config.validate()
    batch.validate()
    if batch.shaping_gamma != config.gamma:
        raise ValueError("rollout shaping gamma differs from PPO return gamma")
    device = _module_device(actor)
    if _module_device(critic) != device:
        raise ValueError("scratch PPO actor and critic must be on the same device")
    if optimizer is None:
        optimizer = torch.optim.Adam(
            [*actor.parameters(), *critic.parameters()], lr=config.learning_rate
        )

    advantages, returns = compute_gae_termination_truncation_v1(
        batch.rewards,
        batch.values,
        batch.next_values,
        batch.terminated,
        batch.truncated,
        gamma=config.gamma,
        gae_lambda=config.gae_lambda,
    )
    normalized_advantages = (advantages - advantages.mean()) / (
        advantages.std() + 1.0e-8
    )
    states = torch.from_numpy(batch.states).to(device)
    pre_tanh = torch.from_numpy(batch.pre_tanh).to(device)
    old_log_probs = torch.from_numpy(batch.old_log_probs).to(device)
    old_values = torch.from_numpy(batch.values).to(device)
    return_tensor = torch.from_numpy(returns).to(device)
    advantage_tensor = torch.from_numpy(normalized_advantages).to(device)
    rng = np.random.default_rng(config.seed)
    entropy_generator = _torch_generator(device, config.seed ^ 0x71B3)
    parameters = [*actor.parameters(), *critic.parameters()]

    with torch.no_grad():
        expected_old_log_probs = squashed_gaussian_log_prob_v1(
            actor.distribution(states), pre_tanh
        )
    if not torch.allclose(
        expected_old_log_probs,
        old_log_probs,
        rtol=1.0e-5,
        atol=1.0e-5,
    ):
        raise ValueError("rollout old_log_probs do not belong to the current actor")

    optimizer_steps = 0
    target_kl_triggered = False
    epochs_completed = 0
    metric_rows: list[tuple[float, float, float, float, float, float, float]] = []
    indices = np.arange(len(batch.rewards))
    actor.train()
    critic.train()
    for epoch in range(config.update_epochs):
        rng.shuffle(indices)
        for start in range(0, len(indices), config.batch_size):
            selected = torch.from_numpy(indices[start : start + config.batch_size]).to(
                device=device, dtype=torch.long
            )
            distribution = actor.distribution(states[selected])
            new_log_prob = squashed_gaussian_log_prob_v1(
                distribution, pre_tanh[selected]
            )
            log_ratio = new_log_prob - old_log_probs[selected]
            ratio = log_ratio.exp()
            unclipped_objective = ratio * advantage_tensor[selected]
            clipped_objective = (
                ratio.clamp(1.0 - config.clip_ratio, 1.0 + config.clip_ratio)
                * advantage_tensor[selected]
            )
            policy_loss = -torch.minimum(unclipped_objective, clipped_objective).mean()

            value = critic(states[selected])
            clipped_value = old_values[selected] + (value - old_values[selected]).clamp(
                -config.value_clip_ratio, config.value_clip_ratio
            )
            value_loss_unclipped = (value - return_tensor[selected]).square()
            value_loss_clipped = (clipped_value - return_tensor[selected]).square()
            value_loss = (
                0.5 * torch.maximum(value_loss_unclipped, value_loss_clipped).mean()
            )

            entropy_sample = sample_squashed_gaussian_v1(
                distribution, generator=entropy_generator
            )
            entropy = -entropy_sample.log_prob.mean()
            total_loss = (
                policy_loss
                + config.value_coef * value_loss
                - config.entropy_coef * entropy
            )
            optimizer.zero_grad(set_to_none=True)
            total_loss.backward()
            gradient_norm = nn.utils.clip_grad_norm_(parameters, config.max_grad_norm)
            optimizer.step()
            optimizer_steps += 1

            with torch.no_grad():
                post_distribution = actor.distribution(states[selected])
                post_log_prob = squashed_gaussian_log_prob_v1(
                    post_distribution, pre_tanh[selected]
                )
                post_log_ratio = post_log_prob - old_log_probs[selected]
                post_ratio = post_log_ratio.exp()
                approximate_kl = ((post_ratio - 1.0) - post_log_ratio).mean()
                policy_clip_fraction = (
                    ((post_ratio - 1.0).abs() > config.clip_ratio).float().mean()
                )
                post_value = critic(states[selected])
                value_clip_fraction = (
                    (
                        (post_value - old_values[selected]).abs()
                        > config.value_clip_ratio
                    )
                    .float()
                    .mean()
                )
            row = (
                float(policy_loss.detach().item()),
                float(value_loss.detach().item()),
                float(entropy.detach().item()),
                float(approximate_kl.item()),
                float(policy_clip_fraction.item()),
                float(value_clip_fraction.item()),
                float(gradient_norm.item()),
            )
            if not np.all(np.isfinite(row)):
                raise RuntimeError("scratch PPO update produced non-finite metrics")
            metric_rows.append(row)
            if row[3] > config.target_kl:
                target_kl_triggered = True
                break
        epochs_completed = epoch + 1
        if target_kl_triggered:
            break

    if (
        optimizer_steps < 1
        or not finite_module_parameters_v1(actor)
        or not finite_module_parameters_v1(critic)
    ):
        raise RuntimeError("scratch PPO update did not produce a finite optimizer step")
    means = np.asarray(metric_rows, dtype=np.float64).mean(axis=0)
    return (
        PPOUpdateMetricsV1(
            optimizer_steps=optimizer_steps,
            epochs_completed=epochs_completed,
            target_kl_triggered=target_kl_triggered,
            policy_loss=float(means[0]),
            value_loss=float(means[1]),
            entropy=float(means[2]),
            approximate_kl=float(means[3]),
            policy_clip_fraction=float(means[4]),
            value_clip_fraction=float(means[5]),
            maximum_preclip_gradient_norm=float(np.asarray(metric_rows)[:, 6].max()),
        ),
        optimizer,
    )


def train_one_scratch_update_v1(
    env: RealisticEdgeArmEnvV7,
    bundle: ScratchPPOBundleV1,
    config: ScratchPPOConfigV1,
    *,
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[ScratchRolloutBatchV1, PPOUpdateMetricsV1, torch.optim.Optimizer]:
    """Collect one curriculum rollout and immediately optimize it."""

    config.validate()
    bundle.provenance.validate()
    reward_model = potential_reward_from_config_v1(bundle.potential_reward_config)
    if reward_model.version != bundle.provenance.potential_reward_version:
        raise ValueError("scratch PPO bundle potential reward version mismatch")
    if reward_model.config_sha256 != bundle.provenance.potential_reward_config_sha256:
        raise ValueError("scratch PPO bundle potential reward provenance mismatch")
    rollout = collect_scratch_rollout_v1(
        env,
        bundle.actor,
        bundle.critic,
        steps=config.rollout_steps,
        seed=config.seed,
        obstacle_probability=config.obstacle_probability,
        stress_probability=config.stress_probability,
        gamma=config.gamma,
        potential_reward=reward_model,
    )
    metrics, optimizer = ppo_update_v1(
        bundle.actor,
        bundle.critic,
        rollout,
        config,
        optimizer=optimizer,
    )
    return rollout, metrics, optimizer


def build_scratch_checkpoint_payload_v1(
    bundle: ScratchPPOBundleV1,
    config: ScratchPPOConfigV1,
    *,
    updates_completed: int,
) -> dict[str, Any]:
    """Build a self-describing checkpoint dictionary with strict provenance."""

    config.validate()
    bundle.provenance.validate()
    bundle.potential_reward_config.validate()
    reward_model = potential_reward_from_config_v1(bundle.potential_reward_config)
    potential_reward_config_sha256 = reward_model.config_sha256
    if (
        potential_reward_config_sha256
        != bundle.provenance.potential_reward_config_sha256
    ):
        raise ValueError("scratch PPO checkpoint potential reward provenance mismatch")
    if reward_model.version != bundle.provenance.potential_reward_version:
        raise ValueError(
            "scratch PPO checkpoint potential reward version provenance mismatch"
        )
    if type(updates_completed) is not int or updates_completed < 0:
        raise ValueError("updates_completed must be a non-negative integer")
    if not finite_module_parameters_v1(bundle.actor) or not finite_module_parameters_v1(
        bundle.critic
    ):
        raise ValueError("scratch PPO checkpoint model state must be finite")
    actor_state = bundle.actor.state_dict()
    critic_state = bundle.critic.state_dict()
    return {
        "format": bundle.provenance.checkpoint_format,
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "source_type": SOURCE_TYPE,
        "actor_architecture": ACTOR_ARCHITECTURE,
        "critic_architecture": CRITIC_ARCHITECTURE,
        "policy_parameterization": POLICY_PARAMETERIZATION,
        "state_dim": PRIVILEGED_EFFECT_STATE_DIM,
        "action_dim": ACTION_DIM,
        "state_layout": list(PRIVILEGED_EFFECT_STATE_LAYOUT_V1),
        "state_schema_sha256": PRIVILEGED_EFFECT_STATE_SCHEMA_SHA256,
        "potential_reward_version": reward_model.version,
        "potential_reward_formula": reward_model.formula,
        "potential_reward_config": asdict(bundle.potential_reward_config),
        "potential_reward_config_sha256": potential_reward_config_sha256,
        "actor_state": actor_state,
        "critic_state": critic_state,
        "actor_state_sha256": state_dict_sha256_v1(actor_state),
        "critic_state_sha256": state_dict_sha256_v1(critic_state),
        "config": asdict(config),
        "updates_completed": updates_completed,
        "provenance": asdict(bundle.provenance),
    }


def validate_scratch_checkpoint_payload_v1(
    payload: dict[str, Any],
) -> ScratchPPOProvenanceV1:
    """Reject checkpoints that omit or alter any full-action scratch boundary."""

    if not isinstance(payload, dict):
        raise TypeError("scratch PPO checkpoint must be a dictionary")
    potential_reward_version = payload.get("potential_reward_version")
    if potential_reward_version not in SUPPORTED_POTENTIAL_REWARD_VERSIONS:
        raise ValueError("scratch PPO checkpoint potential_reward_version mismatch")
    exact = {
        "format": _checkpoint_format_for_potential_reward_v1(potential_reward_version),
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "source_type": SOURCE_TYPE,
        "actor_architecture": ACTOR_ARCHITECTURE,
        "critic_architecture": CRITIC_ARCHITECTURE,
        "policy_parameterization": POLICY_PARAMETERIZATION,
        "state_dim": PRIVILEGED_EFFECT_STATE_DIM,
        "action_dim": ACTION_DIM,
        "state_layout": list(PRIVILEGED_EFFECT_STATE_LAYOUT_V1),
        "state_schema_sha256": PRIVILEGED_EFFECT_STATE_SCHEMA_SHA256,
        "potential_reward_version": potential_reward_version,
        "potential_reward_formula": _potential_reward_formula_v1(
            potential_reward_version
        ),
    }
    for name, expected in exact.items():
        if payload.get(name) != expected:
            raise ValueError(f"scratch PPO checkpoint {name} mismatch")
    potential_reward_config = potential_reward_config_from_dict_v1(
        potential_reward_version,
        payload.get("potential_reward_config"),
    )
    potential_reward_config_sha256 = payload.get("potential_reward_config_sha256")
    if (
        not isinstance(potential_reward_config_sha256, str)
        or _SHA256_PATTERN.fullmatch(potential_reward_config_sha256) is None
        or potential_reward_config.sha256() != potential_reward_config_sha256
    ):
        raise ValueError("scratch PPO checkpoint potential reward config hash mismatch")
    if not isinstance(payload.get("actor_state"), dict) or not isinstance(
        payload.get("critic_state"), dict
    ):
        raise ValueError("scratch PPO checkpoint is missing model state dictionaries")
    for name in ("actor_state_sha256", "critic_state_sha256"):
        value = payload.get(name)
        if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
            raise ValueError(f"scratch PPO checkpoint {name} is malformed")
    if state_dict_sha256_v1(payload["actor_state"]) != payload["actor_state_sha256"]:
        raise ValueError("scratch PPO checkpoint current actor state hash mismatch")
    if state_dict_sha256_v1(payload["critic_state"]) != payload["critic_state_sha256"]:
        raise ValueError("scratch PPO checkpoint current critic state hash mismatch")
    config_payload = payload.get("config")
    expected_config_fields = {field.name for field in fields(ScratchPPOConfigV1)}
    if (
        not isinstance(config_payload, dict)
        or set(config_payload) != expected_config_fields
    ):
        raise ValueError(
            "scratch PPO checkpoint config fields are incomplete or unexpected"
        )
    try:
        checkpoint_config = ScratchPPOConfigV1(**config_payload)
        checkpoint_config.validate()
    except (TypeError, ValueError) as error:
        raise ValueError("scratch PPO checkpoint config is invalid") from error
    if asdict(checkpoint_config) != config_payload:
        raise ValueError("scratch PPO checkpoint config values are non-canonical")
    if (
        not isinstance(payload.get("updates_completed"), int)
        or payload["updates_completed"] < 0
    ):
        raise ValueError("scratch PPO checkpoint updates_completed is invalid")
    provenance = ScratchPPOProvenanceV1.from_dict(payload.get("provenance"))
    if provenance.potential_reward_version != potential_reward_version:
        raise ValueError(
            "scratch PPO checkpoint potential reward version provenance mismatch"
        )
    if provenance.potential_reward_config_sha256 != potential_reward_config_sha256:
        raise ValueError("scratch PPO checkpoint potential reward provenance mismatch")
    return provenance


def save_scratch_checkpoint_v1(
    path: str | Path,
    bundle: ScratchPPOBundleV1,
    config: ScratchPPOConfigV1,
    *,
    updates_completed: int,
) -> None:
    payload = build_scratch_checkpoint_payload_v1(
        bundle, config, updates_completed=updates_completed
    )
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, destination)


def load_scratch_checkpoint_v1(
    path: str | Path,
    *,
    device: str | torch.device = "cpu",
) -> ScratchPPOBundleV1:
    payload = torch.load(Path(path), map_location=device, weights_only=True)
    provenance = validate_scratch_checkpoint_payload_v1(payload)
    potential_reward_config = potential_reward_config_from_dict_v1(
        payload["potential_reward_version"],
        payload["potential_reward_config"],
    )
    actor = FullActionScratchActorV1().to(device)
    critic = PrivilegedEffectCriticV1().to(device)
    actor.load_state_dict(payload["actor_state"], strict=True)
    critic.load_state_dict(payload["critic_state"], strict=True)
    if not finite_module_parameters_v1(actor) or not finite_module_parameters_v1(
        critic
    ):
        raise ValueError("scratch PPO checkpoint contains non-finite model state")
    return ScratchPPOBundleV1(
        actor=actor,
        critic=critic,
        provenance=provenance,
        potential_reward_config=potential_reward_config,
    )
