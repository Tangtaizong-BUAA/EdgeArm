"""Isolated Reward V4 candidate for exact stock-gripper V9 scratch PPO.

V4 fixes the terminal-credit cliff in Reward V3 without changing V1/V2/V3 or
the V9 plant.  Its phase-free task-progress potential lies in ``[0, 28]``.
Unlike a negative remaining-cost potential, a stationary transition receives
``(gamma - 1) * potential <= 0`` and therefore has no positive living bonus.
An explicit, bounded terminal outcome fixes V3's success cliff.

The 96-part substep geometry/contact trace is an explicit simulator-privileged
scratch-training reward input.  It is never appended to the actor observation
and is unavailable to a deployed policy.  Ninety-four safety-only parts are
checked against the block.  The two distal contact parts may penetrate the
block only when the same role has raw, geometrically valid solver contact.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
import torch

from .contact_telemetry_v1 import (
    PHYSICS_SUBSTEP_CONTACT_FORMAT,
    TOOL_CONTACT_IDENTITY_FORMAT,
    TOOL_SAFETY_GEOM_ORDER_FORMAT,
    stable_tool_safety_geom_order_sha256,
)
from .privileged_effect_state_v1 import (
    PRIVILEGED_EFFECT_STATE_DIM,
    PRIVILEGED_EFFECT_STATE_LAYOUT_V1,
    PRIVILEGED_EFFECT_STATE_SCHEMA_SHA256,
    build_privileged_effect_state_v1,
)
from .ppo_utils_v1 import sample_squashed_gaussian_v1, state_dict_sha256_v1
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
    ScratchRolloutBatchV1,
    ppo_update_v1,
)
from .scratch_ppo_v3_candidate import (
    ScratchPotentialEvaluationV3Candidate,
    ScratchPotentialRewardV3Candidate,
)
from .sim2real_env_v9 import RealisticEdgeArmEnvV9, RealisticEnvV9Config


POTENTIAL_REWARD_V4_CANDIDATE_VERSION = "edgearm-v9-privileged-integrity-task-progress-reward-v4-candidate"
CHECKPOINT_FORMAT_V4_CANDIDATE = "edgearm-realism-v9-full-action-ppo-from-scratch-v4-candidate"
TRAINING_GENESIS_V4_CANDIDATE_VERSION = "edgearm-full-action-scratch-ppo-genesis-v4-candidate"
POTENTIAL_REWARD_V4_CANDIDATE_FORMULA = (
    "clip(env_reward,-16,16)+gamma*Phi_next-Phi_before"
    "-24*max(safety_costs)+48*integrity_success-48*safety_terminal;"
    "Phi=12*coverage+6*target_progress+10*contact_progress"
)
ROLLOUT_DIAGNOSTICS_V4_CANDIDATE_FORMAT = "edgearm-scratch-ppo-privileged-integrity-diagnostics-v4-candidate"
REWARD_INPUT_DISCLOSURE_V4_CANDIDATE = {
    "policy_observation_changed": False,
    "expert_action_inputs": 0,
    "controller_phase_inputs": 0,
    "contact_telemetry_is_reward_input": True,
    "contact_telemetry_privilege": ("simulator_privileged_scratch_training_only_not_policy_observation"),
    "tool_safety_geometry_is_reward_input": True,
    "deployment_requires_contact_telemetry": False,
}

_TIP_ROLES = ("fixed_tip", "moving_tip")
_EXPECTED_SAFETY_GEOM_COUNT = 96
_EXPECTED_SAFETY_ONLY_GEOM_COUNT = 94
_SHA256_LENGTH = 64


def _canonical_sha256(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == _SHA256_LENGTH
        and all(character in "0123456789abcdef" for character in value)
    )


def _is_real_numeric_dtype(dtype: np.dtype[Any]) -> bool:
    return bool(np.issubdtype(dtype, np.integer) or np.issubdtype(dtype, np.floating))


def _finite_numeric_matrix(
    value: object,
    *,
    name: str,
    columns: int,
) -> np.ndarray:
    array = np.asarray(value)
    if (
        array.ndim != 2
        or array.shape[0] < 1
        or array.shape[1] != columns
        or np.issubdtype(array.dtype, np.bool_)
        or not _is_real_numeric_dtype(array.dtype)
    ):
        raise RuntimeError(f"V4 {name} must be a numeric matrix with {columns} columns")
    converted = array.astype(np.float64, copy=False)
    if not np.all(np.isfinite(converted)):
        raise RuntimeError(f"V4 {name} contains non-finite values")
    return converted


def _nonnegative_count_matrix(
    value: object,
    *,
    name: str,
    rows: int,
) -> np.ndarray:
    array = np.asarray(value)
    if (
        array.shape != (rows, 2)
        or np.issubdtype(array.dtype, np.bool_)
        or not _is_real_numeric_dtype(array.dtype)
    ):
        raise RuntimeError(f"V4 {name} must be a numeric ({rows}, 2) matrix")
    numeric = array.astype(np.float64, copy=False)
    if (
        not np.all(np.isfinite(numeric))
        or np.any(numeric < 0.0)
        or not np.array_equal(numeric, np.floor(numeric))
        or np.any(numeric > np.iinfo(np.int64).max)
    ):
        raise RuntimeError(f"V4 {name} must contain finite non-negative integers")
    return numeric.astype(np.int64)


def _binary_indicator_matrix(
    value: object,
    *,
    name: str,
    rows: int,
) -> np.ndarray:
    array = np.asarray(value)
    if (
        array.shape != (rows, 2)
        or np.issubdtype(array.dtype, np.bool_)
        or not _is_real_numeric_dtype(array.dtype)
    ):
        raise RuntimeError(f"V4 {name} must be a numeric ({rows}, 2) matrix")
    numeric = array.astype(np.float64, copy=False)
    if not np.all(np.isfinite(numeric)) or np.any((numeric != 0.0) & (numeric != 1.0)):
        raise RuntimeError(f"V4 {name} must contain only numeric 0/1 values")
    return numeric.astype(bool)


def _exact_integer_vector(
    value: object,
    *,
    name: str,
    length: int,
) -> tuple[int, ...]:
    array = np.asarray(value)
    if (
        array.shape != (length,)
        or np.issubdtype(array.dtype, np.bool_)
        or not np.issubdtype(array.dtype, np.integer)
        or np.any(array < 0)
    ):
        raise RuntimeError(f"V4 {name} must be an exact non-negative integer vector")
    return tuple(int(item) for item in array)


@dataclass(frozen=True)
class ScratchPotentialRewardV4CandidateConfig:
    """Frozen coefficients and auditable terminal-dominance bounds."""

    coverage_progress_coefficient: float = 12.0
    block_progress_coefficient: float = 6.0
    contact_progress_coefficient: float = 10.0
    environment_reward_clip_abs: float = 16.0
    privileged_safety_penalty_coefficient: float = 24.0
    terminal_outcome_magnitude: float = 48.0
    safety_only_block_clearance_m: float = 0.00025
    penetration_tolerance_m: float = 0.00010
    safety_depth_scale_m: float = 0.005

    @property
    def maximum_task_potential(self) -> float:
        return float(
            self.coverage_progress_coefficient
            + self.block_progress_coefficient
            + self.contact_progress_coefficient
        )

    @property
    def terminal_failure_upper_bound(self) -> float:
        """Largest possible safety-terminal reward before safety penalty."""

        return float(self.environment_reward_clip_abs - self.terminal_outcome_magnitude)

    @property
    def integrity_success_lower_bound(self) -> float:
        """Smallest possible safe-success terminal reward."""

        return float(
            -self.environment_reward_clip_abs - self.maximum_task_potential + self.terminal_outcome_magnitude
        )

    def validate(self) -> None:
        canonical = type(self)()
        for field in fields(self):
            value = getattr(self, field.name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value):
                raise ValueError(f"V4 candidate reward {field.name} must be finite numeric")
            if float(value) != float(getattr(canonical, field.name)):
                raise ValueError(
                    f"V4 candidate reward {field.name} is frozen at {getattr(canonical, field.name)}"
                )
        if not 0.0 <= self.penetration_tolerance_m < self.safety_only_block_clearance_m:
            raise ValueError("V4 penetration tolerance must be below clearance")
        if self.safety_depth_scale_m <= 0.0:
            raise ValueError("V4 safety depth scale must be positive")
        if not self.terminal_outcome_magnitude > (
            self.environment_reward_clip_abs + self.maximum_task_potential
        ):
            raise ValueError("V4 terminal outcome must strictly dominate clipped env reward and shaping")
        if self.terminal_failure_upper_bound >= 0.0:
            raise ValueError("V4 safety-terminal upper bound must be negative")
        if self.integrity_success_lower_bound <= 0.0:
            raise ValueError("V4 integrity-success lower bound must be positive")

    def sha256(self) -> str:
        self.validate()
        return _canonical_sha256(
            {
                "version": POTENTIAL_REWARD_V4_CANDIDATE_VERSION,
                "formula": POTENTIAL_REWARD_V4_CANDIDATE_FORMULA,
                "environment_class": "exact RealisticEdgeArmEnvV9",
                "ordered_tip_roles": list(_TIP_ROLES),
                "complete_safety_geom_count": _EXPECTED_SAFETY_GEOM_COUNT,
                "safety_only_geom_count": _EXPECTED_SAFETY_ONLY_GEOM_COUNT,
                "reward_input_disclosure": REWARD_INPUT_DISCLOSURE_V4_CANDIDATE,
                "config": asdict(self),
            }
        )


@dataclass(frozen=True)
class ScratchPotentialEvaluationV4Candidate:
    potential: float
    coverage_progress_value: float
    block_progress_value: float
    contact_progress_value: float
    coverage: float
    normalized_block_progress: float
    contact_approach_progress: float
    block_target_distance_m: float
    tip_block_signed_distance_m: tuple[float, float]
    role_cost: tuple[float, float]


@dataclass(frozen=True)
class ScratchSafetyEvidenceV4Candidate:
    """One transition of privileged, role-preserving V9 safety evidence."""

    physics_substeps: int
    safety_geom_count: int
    safety_only_geom_count: int
    contact_safety_geom_indices: tuple[int, int]
    minimum_safety_only_block_signed_distance_m: float
    limiting_safety_only_geom_index: int
    minimum_contact_part_block_signed_distance_by_role_m: tuple[float, float]
    minimum_full_safety_desk_signed_distance_m: float
    safety_only_clearance_violation: bool
    safety_only_penetration: bool
    unauthorized_contact_part_penetration: bool
    full_safety_desk_penetration: bool
    unauthorized_contact_part_penetration_count: int
    authorized_contact_part_penetration_count: int
    normalized_safety_only_block_cost: float
    normalized_unauthorized_contact_cost: float
    normalized_full_safety_desk_cost: float
    normalized_safety_cost: float
    safety_penalty: float
    hard_safety_violation: bool
    hard_safety_reason: str
    privileged_reward_input: bool = True


@dataclass(frozen=True)
class ScratchPotentialTransitionV4Candidate:
    v9_env_reward_raw: float
    clipped_environment_reward: float
    reward_before_potential: float
    potential_before: float
    potential_after: float
    potential_next_for_shaping: float
    potential_shaping: float
    safety_penalty: float
    terminal_outcome_reward: float
    shaped_reward: float
    env_terminated: bool
    env_truncated: bool
    credit_terminated: bool
    credit_truncated: bool
    v9_surface_success: bool
    v4_integrity_success: bool
    safety_credit_terminal: bool
    episode_safety_violation: bool
    terminal_reason: str


class ScratchPotentialRewardV4Candidate:
    """Phase-free task progress plus privileged V9 integrity adjudication."""

    version = POTENTIAL_REWARD_V4_CANDIDATE_VERSION
    formula = POTENTIAL_REWARD_V4_CANDIDATE_FORMULA
    reward_input_disclosure = REWARD_INPUT_DISCLOSURE_V4_CANDIDATE

    def __init__(
        self,
        config: ScratchPotentialRewardV4CandidateConfig | None = None,
    ) -> None:
        self.config = config or ScratchPotentialRewardV4CandidateConfig()
        self.config.validate()
        self.config_sha256 = self.config.sha256()
        self._v3_geometry = ScratchPotentialRewardV3Candidate()

    @staticmethod
    def _require_exact_v9(env: object) -> RealisticEdgeArmEnvV9:
        if type(env) is not RealisticEdgeArmEnvV9:
            raise TypeError("V4 candidate reward requires exact RealisticEdgeArmEnvV9")
        if (
            type(env.contact_feasible_config) is not RealisticEnvV9Config
            or type(env.stock_distal_tip_config) is not RealisticEnvV9Config
        ):
            raise TypeError("V4 candidate reward requires exact RealisticEnvV9Config")
        return env

    def evaluate(self, env: RealisticEdgeArmEnvV9) -> ScratchPotentialEvaluationV4Candidate:
        """Evaluate a bounded task-progress potential from current geometry."""

        exact_env = self._require_exact_v9(env)
        geometry: ScratchPotentialEvaluationV3Candidate = self._v3_geometry.evaluate(exact_env)
        coverage_value = self.config.coverage_progress_coefficient * geometry.coverage
        block_value = self.config.block_progress_coefficient * geometry.normalized_block_progress
        contact_value = self.config.contact_progress_coefficient * geometry.contact_approach_progress
        potential = float(coverage_value + block_value + contact_value)
        if not -1.0e-12 <= potential <= self.config.maximum_task_potential + 1.0e-12:
            raise RuntimeError("V4 task-progress potential escaped its proven bounds")
        return ScratchPotentialEvaluationV4Candidate(
            potential=potential,
            coverage_progress_value=float(coverage_value),
            block_progress_value=float(block_value),
            contact_progress_value=float(contact_value),
            coverage=geometry.coverage,
            normalized_block_progress=geometry.normalized_block_progress,
            contact_approach_progress=geometry.contact_approach_progress,
            block_target_distance_m=geometry.block_target_distance_m,
            tip_block_signed_distance_m=geometry.tip_block_signed_distance_m,
            role_cost=geometry.role_cost,
        )

    def evaluate_transition_safety(
        self,
        env: RealisticEdgeArmEnvV9,
        info: dict[str, Any],
    ) -> ScratchSafetyEvidenceV4Candidate:
        """Consume all 96 V9 substep columns as privileged reward evidence."""

        exact_env = self._require_exact_v9(env)
        trace = info.get("physics_substep_contact_v1")
        if not isinstance(trace, dict):
            raise RuntimeError("V4 candidate requires physics-substep contact telemetry")
        if trace.get("format") != PHYSICS_SUBSTEP_CONTACT_FORMAT:
            raise RuntimeError("V4 candidate contact telemetry format changed")
        if trace.get("contact_identity_format") != TOOL_CONTACT_IDENTITY_FORMAT:
            raise RuntimeError("V4 candidate contact identity format changed")
        if trace.get("simulator_privileged_truth") is not True:
            raise RuntimeError("V4 reward input must declare simulator privilege")
        if trace.get("tool_safety_geom_order_format") != TOOL_SAFETY_GEOM_ORDER_FORMAT:
            raise RuntimeError("V4 safety geometry order format changed")
        roles = tuple(str(value) for value in trace.get("tool_contact_role_names", ()))
        if roles != _TIP_ROLES:
            raise RuntimeError("V4 contact role order changed")

        expected_safety_ids = tuple(int(value) for value in exact_env._ids.get("tool_safety_geoms", ()))
        expected_contact_ids = tuple(int(value) for value in exact_env._ids.get("tool_contact_geoms", ()))
        expected_role_identity = tuple(
            str(value) for value in exact_env._ids.get("tool_contact_geom_roles", ())
        )
        if expected_role_identity != _TIP_ROLES:
            raise RuntimeError("V4 exact V9 contact-role identity changed")
        trace_safety_ids = _exact_integer_vector(
            trace.get("tool_safety_geom_ids", ()),
            name="tool_safety_geom_ids",
            length=_EXPECTED_SAFETY_GEOM_COUNT,
        )
        trace_contact_ids = _exact_integer_vector(
            trace.get("tool_contact_geom_ids", ()),
            name="tool_contact_geom_ids",
            length=2,
        )
        expected_safety_names = tuple(
            mujoco.mj_id2name(exact_env.model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or ""
            for geom_id in expected_safety_ids
        )
        trace_safety_names = tuple(trace.get("tool_safety_geom_names", ()))
        if (
            len(expected_safety_ids) != _EXPECTED_SAFETY_GEOM_COUNT
            or trace_safety_ids != expected_safety_ids
            or trace_contact_ids != expected_contact_ids
            or len(expected_contact_ids) != 2
            or len(trace_safety_names) != _EXPECTED_SAFETY_GEOM_COUNT
            or any(not isinstance(name, str) or not name.strip() for name in trace_safety_names)
            or len(set(trace_safety_names)) != _EXPECTED_SAFETY_GEOM_COUNT
            or trace_safety_names != expected_safety_names
            or trace.get("tool_safety_geom_order_sha256")
            != stable_tool_safety_geom_order_sha256(trace_safety_names)
            or trace.get("tool_safety_geom_count") != _EXPECTED_SAFETY_GEOM_COUNT
            or trace.get("tool_safety_distance_sampled_each_substep") is not True
        ):
            raise RuntimeError("V4 reward requires the exact ordered V9 96-part union")
        contact_indices = tuple(trace_safety_ids.index(geom_id) for geom_id in expected_contact_ids)
        if len(set(contact_indices)) != 2:
            raise RuntimeError("V4 contact-part safety indices are not unique")
        safety_only_indices = tuple(
            index for index in range(_EXPECTED_SAFETY_GEOM_COUNT) if index not in contact_indices
        )
        if len(safety_only_indices) != _EXPECTED_SAFETY_ONLY_GEOM_COUNT:
            raise RuntimeError("V4 reward did not resolve exactly 94 safety-only parts")

        block = _finite_numeric_matrix(
            trace.get("tool_safety_block_signed_distance_m"),
            name="tool_safety_block_signed_distance_m",
            columns=_EXPECTED_SAFETY_GEOM_COUNT,
        )
        desk = _finite_numeric_matrix(
            trace.get("tool_safety_desk_signed_distance_m"),
            name="tool_safety_desk_signed_distance_m",
            columns=_EXPECTED_SAFETY_GEOM_COUNT,
        )
        if desk.shape != block.shape:
            raise RuntimeError("V4 safety desk/block matrices have different shapes")
        physics_substeps = trace.get("physics_substeps")
        if (
            isinstance(physics_substeps, bool)
            or not isinstance(physics_substeps, (int, np.integer))
            or int(physics_substeps) != block.shape[0]
        ):
            raise RuntimeError("V4 physics_substeps disagrees with safety trace rows")
        raw = _nonnegative_count_matrix(
            trace.get("tool_block_contact_count_by_role"),
            name="tool_block_contact_count_by_role",
            rows=block.shape[0],
        )
        invalid = _nonnegative_count_matrix(
            trace.get("invalid_tool_block_contact_count_by_role"),
            name="invalid_tool_block_contact_count_by_role",
            rows=block.shape[0],
        )
        all_valid_bool = _binary_indicator_matrix(
            trace.get("all_tool_block_contacts_geometrically_valid_by_role"),
            name="all_tool_block_contacts_geometrically_valid_by_role",
            rows=block.shape[0],
        )
        if np.any(invalid > raw):
            raise RuntimeError("V4 invalid contact count exceeds raw count")
        if not np.array_equal(all_valid_bool, invalid == 0):
            raise RuntimeError("V4 geometric-valid flags disagree with invalid counts")

        safety_only = block[:, np.asarray(safety_only_indices, dtype=np.int64)]
        flat_limiting = int(np.argmin(safety_only))
        _substep_index, limiting_local = np.unravel_index(flat_limiting, safety_only.shape)
        limiting_index = int(safety_only_indices[int(limiting_local)])
        minimum_safety_only = float(np.min(safety_only))
        minimum_desk = float(np.min(desk))

        contact_minimums: list[float] = []
        unauthorized_count = 0
        authorized_count = 0
        maximum_unauthorized_depth = 0.0
        for role_index, contact_index in enumerate(contact_indices):
            distances = block[:, contact_index]
            contact_minimums.append(float(np.min(distances)))
            penetrating = distances < -self.config.penetration_tolerance_m
            authorized = (
                (raw[:, role_index] > 0) & all_valid_bool[:, role_index] & (invalid[:, role_index] == 0)
            )
            unauthorized = penetrating & ~authorized
            unauthorized_count += int(np.count_nonzero(unauthorized))
            authorized_count += int(np.count_nonzero(penetrating & authorized))
            if np.any(unauthorized):
                maximum_unauthorized_depth = max(
                    maximum_unauthorized_depth,
                    float(np.max(-distances[unauthorized])),
                )

        safety_only_clearance_violation = bool(
            minimum_safety_only < self.config.safety_only_block_clearance_m
        )
        safety_only_penetration = bool(minimum_safety_only < -self.config.penetration_tolerance_m)
        unauthorized_penetration = unauthorized_count > 0
        desk_penetration = bool(minimum_desk < -self.config.penetration_tolerance_m)
        safety_only_depth = max(
            self.config.safety_only_block_clearance_m - minimum_safety_only,
            0.0,
        )
        desk_depth = max(-minimum_desk - self.config.penetration_tolerance_m, 0.0)
        unauthorized_depth = max(
            maximum_unauthorized_depth - self.config.penetration_tolerance_m,
            0.0,
        )
        normalized_safety_only = float(
            np.clip(safety_only_depth / self.config.safety_depth_scale_m, 0.0, 1.0)
        )
        normalized_unauthorized = float(
            np.clip(
                unauthorized_depth / self.config.safety_depth_scale_m,
                0.0,
                1.0,
            )
        )
        normalized_desk = float(np.clip(desk_depth / self.config.safety_depth_scale_m, 0.0, 1.0))
        normalized_safety = max(
            normalized_safety_only,
            normalized_unauthorized,
            normalized_desk,
        )
        safety_penalty = -self.config.privileged_safety_penalty_coefficient * (normalized_safety)
        hard_reasons: list[str] = []
        if safety_only_clearance_violation:
            hard_reasons.append("safety_only_block_clearance_violation")
        if unauthorized_penetration:
            hard_reasons.append("unauthorized_contact_part_penetration")
        if desk_penetration:
            hard_reasons.append("full_safety_union_desk_penetration")
        return ScratchSafetyEvidenceV4Candidate(
            physics_substeps=int(block.shape[0]),
            safety_geom_count=_EXPECTED_SAFETY_GEOM_COUNT,
            safety_only_geom_count=_EXPECTED_SAFETY_ONLY_GEOM_COUNT,
            contact_safety_geom_indices=contact_indices,  # type: ignore[arg-type]
            minimum_safety_only_block_signed_distance_m=minimum_safety_only,
            limiting_safety_only_geom_index=limiting_index,
            minimum_contact_part_block_signed_distance_by_role_m=(
                float(contact_minimums[0]),
                float(contact_minimums[1]),
            ),
            minimum_full_safety_desk_signed_distance_m=minimum_desk,
            safety_only_clearance_violation=safety_only_clearance_violation,
            safety_only_penetration=safety_only_penetration,
            unauthorized_contact_part_penetration=unauthorized_penetration,
            full_safety_desk_penetration=desk_penetration,
            unauthorized_contact_part_penetration_count=unauthorized_count,
            authorized_contact_part_penetration_count=authorized_count,
            normalized_safety_only_block_cost=normalized_safety_only,
            normalized_unauthorized_contact_cost=normalized_unauthorized,
            normalized_full_safety_desk_cost=normalized_desk,
            normalized_safety_cost=normalized_safety,
            safety_penalty=float(safety_penalty),
            hard_safety_violation=bool(hard_reasons),
            hard_safety_reason="+".join(hard_reasons),
        )

    def shape_transition(
        self,
        *,
        env_reward: float,
        potential_before: float,
        potential_after: float,
        gamma: float,
        env_terminated: bool,
        env_truncated: bool,
        v9_surface_success: bool,
        env_terminal_failure: bool,
        safety: ScratchSafetyEvidenceV4Candidate,
        episode_safety_violation_before: bool,
    ) -> ScratchPotentialTransitionV4Candidate:
        """Adjudicate one transition and cut credit at the first hard violation."""

        numeric = np.asarray([env_reward, potential_before, potential_after, gamma], dtype=np.float64)
        if not np.all(np.isfinite(numeric)) or not 0.0 < gamma <= 1.0:
            raise ValueError("V4 shaping inputs must be finite and gamma in (0, 1]")
        if not -1.0e-9 <= potential_before <= self.config.maximum_task_potential + 1.0e-9:
            raise ValueError("V4 potential_before is outside the proven range")
        if not -1.0e-9 <= potential_after <= self.config.maximum_task_potential + 1.0e-9:
            raise ValueError("V4 potential_after is outside the proven range")
        if env_terminated and env_truncated:
            raise ValueError("V4 transition cannot be terminated and truncated")
        if v9_surface_success and (not env_terminated or env_terminal_failure):
            raise ValueError("V9 surface success must be a non-failure termination")
        if episode_safety_violation_before and v9_surface_success:
            # Full-trace audits may reach this state, but a real V4 rollout may not.
            integrity_success = False
        else:
            integrity_success = bool(
                v9_surface_success
                and not episode_safety_violation_before
                and not safety.hard_safety_violation
            )
        episode_safety_violation = bool(episode_safety_violation_before or safety.hard_safety_violation)
        safety_credit_terminal = bool(
            episode_safety_violation or env_terminal_failure or (env_terminated and not integrity_success)
        )
        credit_terminated = bool(integrity_success or safety_credit_terminal)
        credit_truncated = bool(env_truncated and not credit_terminated)
        potential_next = 0.0 if credit_terminated else float(potential_after)
        potential_shaping = float(gamma * potential_next - float(potential_before))
        clipped_env = float(
            np.clip(
                env_reward,
                -self.config.environment_reward_clip_abs,
                self.config.environment_reward_clip_abs,
            )
        )
        if integrity_success:
            terminal_outcome = self.config.terminal_outcome_magnitude
            terminal_reason = "v4_integrity_success"
        elif safety_credit_terminal:
            terminal_outcome = -self.config.terminal_outcome_magnitude
            if safety.hard_safety_violation:
                terminal_reason = f"v4_safety_terminal:{safety.hard_safety_reason}"
            else:
                terminal_reason = "v4_safety_terminal:environment_terminal_failure"
        elif credit_truncated:
            terminal_outcome = 0.0
            terminal_reason = "time_limit"
        else:
            terminal_outcome = 0.0
            terminal_reason = "nonterminal"
        reward_before_potential = float(clipped_env + safety.safety_penalty + terminal_outcome)
        shaped_reward = float(reward_before_potential + potential_shaping)
        return ScratchPotentialTransitionV4Candidate(
            v9_env_reward_raw=float(env_reward),
            clipped_environment_reward=clipped_env,
            reward_before_potential=reward_before_potential,
            potential_before=float(potential_before),
            potential_after=float(potential_after),
            potential_next_for_shaping=potential_next,
            potential_shaping=potential_shaping,
            safety_penalty=float(safety.safety_penalty),
            terminal_outcome_reward=float(terminal_outcome),
            shaped_reward=shaped_reward,
            env_terminated=bool(env_terminated),
            env_truncated=bool(env_truncated),
            credit_terminated=credit_terminated,
            credit_truncated=credit_truncated,
            v9_surface_success=bool(v9_surface_success),
            v4_integrity_success=integrity_success,
            safety_credit_terminal=safety_credit_terminal,
            episode_safety_violation=episode_safety_violation,
            terminal_reason=terminal_reason,
        )


@dataclass(frozen=True)
class ScratchRolloutDiagnosticsV4Candidate:
    v9_env_reward_raw: np.ndarray
    v9_surface_success: np.ndarray
    v4_integrity_success: np.ndarray
    hard_safety_violation: np.ndarray
    minimum_94_safety_only_block_distance_m: np.ndarray
    minimum_contact_part_block_distance_by_role_m: np.ndarray
    unauthorized_contact_part_penetration_count: np.ndarray
    privileged_safety_penalty: np.ndarray
    format: str = ROLLOUT_DIAGNOSTICS_V4_CANDIDATE_FORMAT
    privileged_reward_input: bool = True

    def validate(self, transition_count: int) -> None:
        expected = {
            "v9_env_reward_raw": (transition_count,),
            "v9_surface_success": (transition_count,),
            "v4_integrity_success": (transition_count,),
            "hard_safety_violation": (transition_count,),
            "minimum_94_safety_only_block_distance_m": (transition_count,),
            "minimum_contact_part_block_distance_by_role_m": (transition_count, 2),
            "unauthorized_contact_part_penetration_count": (transition_count,),
            "privileged_safety_penalty": (transition_count,),
        }
        for name, shape in expected.items():
            value = np.asarray(getattr(self, name))
            if value.shape != shape:
                raise ValueError(f"V4 diagnostics {name} shape mismatch")
        for name in (
            "v9_env_reward_raw",
            "minimum_94_safety_only_block_distance_m",
            "minimum_contact_part_block_distance_by_role_m",
            "privileged_safety_penalty",
        ):
            if not np.all(np.isfinite(np.asarray(getattr(self, name)))):
                raise ValueError(f"V4 diagnostics {name} contains non-finite values")
        for name in (
            "v9_surface_success",
            "v4_integrity_success",
            "hard_safety_violation",
        ):
            if np.asarray(getattr(self, name)).dtype != np.dtype(bool):
                raise ValueError(f"V4 diagnostics {name} must be boolean")
        counts = np.asarray(self.unauthorized_contact_part_penetration_count)
        if not np.issubdtype(counts.dtype, np.integer) or np.any(counts < 0):
            raise ValueError("V4 unauthorized contact counts are invalid")
        if np.any(self.v4_integrity_success & ~self.v9_surface_success):
            raise ValueError("V4 integrity success must imply V9 surface success")
        if np.any(self.v4_integrity_success & self.hard_safety_violation):
            raise ValueError("V4 integrity success cannot include a safety violation")
        if self.format != ROLLOUT_DIAGNOSTICS_V4_CANDIDATE_FORMAT:
            raise ValueError("V4 diagnostics format mismatch")
        if self.privileged_reward_input is not True:
            raise ValueError("V4 safety diagnostics must declare reward privilege")

    def metric_fields(self) -> dict[str, object]:
        count = int(np.asarray(self.v9_env_reward_raw).size)
        self.validate(count)
        return {
            "rollout_v9_surface_success_count": int(np.count_nonzero(self.v9_surface_success)),
            "rollout_v4_integrity_success_count": int(np.count_nonzero(self.v4_integrity_success)),
            "rollout_v4_hard_safety_violation_count": int(np.count_nonzero(self.hard_safety_violation)),
            "rollout_v4_minimum_94_safety_only_block_distance_m": float(
                np.min(self.minimum_94_safety_only_block_distance_m)
            ),
            "rollout_v4_unauthorized_contact_part_penetration_count": int(
                np.sum(self.unauthorized_contact_part_penetration_count)
            ),
        }


@dataclass(frozen=True)
class ScratchRolloutBatchV4Candidate:
    ppo_batch: ScratchRolloutBatchV1
    integrity_diagnostics: ScratchRolloutDiagnosticsV4Candidate
    potential_reward_version: str = POTENTIAL_REWARD_V4_CANDIDATE_VERSION

    def __getattr__(self, name: str) -> Any:
        return getattr(self.ppo_batch, name)

    def validate(self) -> None:
        if self.potential_reward_version != POTENTIAL_REWARD_V4_CANDIDATE_VERSION:
            raise ValueError("V4 rollout reward version mismatch")
        self.ppo_batch.validate()
        count = int(self.ppo_batch.shaped_rewards.size)
        self.integrity_diagnostics.validate(count)
        if not np.array_equal(
            self.ppo_batch.strict_success,
            self.integrity_diagnostics.v4_integrity_success,
        ):
            raise ValueError("V4 PPO success flags differ from integrity success")
        hard = np.asarray(self.integrity_diagnostics.hard_safety_violation, dtype=bool)
        unauthorized = np.asarray(
            self.integrity_diagnostics.unauthorized_contact_part_penetration_count,
            dtype=np.int64,
        )
        integrity = np.asarray(self.integrity_diagnostics.v4_integrity_success, dtype=bool)
        surface = np.asarray(self.integrity_diagnostics.v9_surface_success, dtype=bool)
        if np.any(
            hard & ~(self.ppo_batch.terminated & self.ppo_batch.terminal_failure & self.ppo_batch.safety_stop)
        ):
            raise ValueError("V4 hard safety violation must terminate as failure and safety stop")
        if np.any((unauthorized > 0) & ~hard):
            raise ValueError("V4 unauthorized contact penetration must be a hard violation")
        if np.any(
            integrity
            & ~(self.ppo_batch.terminated & ~self.ppo_batch.terminal_failure & ~self.ppo_batch.safety_stop)
        ):
            raise ValueError("V4 integrity success must be a non-failure terminal transition")
        if np.any(surface & ~self.ppo_batch.terminated):
            raise ValueError("V4 V9 surface success must end the credit episode")
        episode_ids = np.asarray(self.ppo_batch.episode_ids, dtype=np.int64)
        hard_indices = np.flatnonzero(hard)
        for index in hard_indices:
            if index + 1 < episode_ids.size and episode_ids[index + 1] == episode_ids[index]:
                raise ValueError("V4 rollout continued an episode after a hard violation")


def _module_device(module: torch.nn.Module) -> torch.device:
    try:
        return next(module.parameters()).device
    except StopIteration as error:  # pragma: no cover
        raise RuntimeError("V4 candidate PPO module has no parameters") from error


def _torch_generator(device: torch.device, seed: int) -> torch.Generator | None:
    if device.type == "mps":
        torch.manual_seed(seed)
        if hasattr(torch, "mps"):
            torch.mps.manual_seed(seed)
        return None
    generator = torch.Generator(device=device.type)
    generator.manual_seed(seed)
    return generator


def collect_scratch_rollout_v4_candidate(
    env: RealisticEdgeArmEnvV9,
    actor: FullActionScratchActorV1,
    critic: PrivilegedEffectCriticV1,
    *,
    steps: int,
    seed: int,
    obstacle_probability: float = 0.50,
    stress_probability: float = 0.30,
    gamma: float = 0.99,
    potential_reward: ScratchPotentialRewardV4Candidate | None = None,
) -> ScratchRolloutBatchV4Candidate:
    """Collect complete V4 credit episodes, resetting on the first violation."""

    reward = potential_reward or ScratchPotentialRewardV4Candidate()
    reward._require_exact_v9(env)
    if type(reward) is not ScratchPotentialRewardV4Candidate:
        raise TypeError("V4 rollout requires the exact V4 reward strategy")
    if type(steps) is not int or steps < 1 or type(seed) is not int or seed < 0:
        raise ValueError("V4 rollout steps/seed are invalid")
    for name, probability in (
        ("obstacle_probability", obstacle_probability),
        ("stress_probability", stress_probability),
    ):
        if not np.isfinite(probability) or not 0.0 <= probability <= 1.0:
            raise ValueError(f"{name} must be finite and in [0, 1]")
    if not np.isfinite(gamma) or not 0.0 < gamma <= 1.0:
        raise ValueError("V4 rollout gamma must be in (0, 1]")
    device = _module_device(actor)
    if _module_device(critic) != device:
        raise ValueError("V4 actor and critic must share a device")
    generator = _torch_generator(device, seed)
    episode_rng = np.random.default_rng(seed ^ 0x7A41C9)

    rows: dict[str, list[Any]] = {
        name: []
        for name in (
            "states",
            "actions",
            "pre_tanh",
            "old_log_probs",
            "reward_before_potential",
            "potential_before",
            "potential_after",
            "potential_next",
            "shaped_rewards",
            "values",
            "next_values",
            "terminated",
            "truncated",
            "integrity_success",
            "terminal_failure",
            "safety_stop",
            "terminal_reason",
            "episode_ids",
            "obstacle",
            "raw_env_reward",
            "surface_success",
            "hard_violation",
            "minimum_94",
            "contact_minimums",
            "unauthorized_count",
            "safety_penalty",
        )
    }
    episode_id = 0

    def reset_episode(current_id: int) -> None:
        obstacle = bool(episode_rng.random() < obstacle_probability)
        stress = bool(episode_rng.random() < stress_probability)
        env.reset(seed=seed + current_id, obstacle=obstacle, stress=stress)

    reset_episode(episode_id)
    actor.eval()
    critic.eval()
    while len(rows["states"]) < steps or not (rows["terminated"][-1] or rows["truncated"][-1]):
        before = reward.evaluate(env)
        state = build_privileged_effect_state_v1(env)
        state_tensor = torch.from_numpy(state).to(device).unsqueeze(0)
        with torch.no_grad():
            distribution = actor.distribution(state_tensor)
            sample = sample_squashed_gaussian_v1(distribution, generator=generator)
            value = float(critic(state_tensor).item())
        action = sample.action.squeeze(0).cpu().numpy().astype(np.float32)
        _, env_reward, env_terminated, env_truncated, info = env.step(action)
        after = reward.evaluate(env)
        safety = reward.evaluate_transition_safety(env, info)
        surface_success = bool(info.get("success", False))
        env_failure = bool(info.get("terminal_failure", env_terminated and not surface_success))
        transition = reward.shape_transition(
            env_reward=env_reward,
            potential_before=before.potential,
            potential_after=after.potential,
            gamma=gamma,
            env_terminated=bool(env_terminated),
            env_truncated=bool(env_truncated),
            v9_surface_success=surface_success,
            env_terminal_failure=env_failure,
            safety=safety,
            episode_safety_violation_before=False,
        )
        next_state = build_privileged_effect_state_v1(env)
        with torch.no_grad():
            next_value = float(critic(torch.from_numpy(next_state).to(device).unsqueeze(0)).item())

        append_values = {
            "states": state,
            "actions": action,
            "pre_tanh": sample.pre_tanh.squeeze(0).cpu().numpy().astype(np.float32),
            "old_log_probs": float(sample.log_prob.item()),
            "reward_before_potential": transition.reward_before_potential,
            "potential_before": transition.potential_before,
            "potential_after": transition.potential_after,
            "potential_next": transition.potential_next_for_shaping,
            "shaped_rewards": transition.shaped_reward,
            "values": value,
            "next_values": next_value,
            "terminated": transition.credit_terminated,
            "truncated": transition.credit_truncated,
            "integrity_success": transition.v4_integrity_success,
            "terminal_failure": transition.safety_credit_terminal,
            "safety_stop": bool(safety.hard_safety_violation or info.get("safety_stop")),
            "terminal_reason": transition.terminal_reason,
            "episode_ids": episode_id,
            "obstacle": bool(env.obstacle_enabled),
            "raw_env_reward": transition.v9_env_reward_raw,
            "surface_success": transition.v9_surface_success,
            "hard_violation": safety.hard_safety_violation,
            "minimum_94": safety.minimum_safety_only_block_signed_distance_m,
            "contact_minimums": safety.minimum_contact_part_block_signed_distance_by_role_m,
            "unauthorized_count": safety.unauthorized_contact_part_penetration_count,
            "safety_penalty": safety.safety_penalty,
        }
        for name, value_to_append in append_values.items():
            rows[name].append(value_to_append)

        if (transition.credit_terminated or transition.credit_truncated) and len(rows["states"]) < steps:
            episode_id += 1
            reset_episode(episode_id)

    ppo_batch = ScratchRolloutBatchV1(
        states=np.asarray(rows["states"], dtype=np.float32),
        actions=np.asarray(rows["actions"], dtype=np.float32),
        pre_tanh=np.asarray(rows["pre_tanh"], dtype=np.float32),
        old_log_probs=np.asarray(rows["old_log_probs"], dtype=np.float32),
        env_rewards=np.asarray(rows["reward_before_potential"], dtype=np.float32),
        potential_before=np.asarray(rows["potential_before"], dtype=np.float32),
        potential_after=np.asarray(rows["potential_after"], dtype=np.float32),
        potential_next_for_shaping=np.asarray(rows["potential_next"], dtype=np.float32),
        shaped_rewards=np.asarray(rows["shaped_rewards"], dtype=np.float32),
        values=np.asarray(rows["values"], dtype=np.float32),
        next_values=np.asarray(rows["next_values"], dtype=np.float32),
        terminated=np.asarray(rows["terminated"], dtype=bool),
        truncated=np.asarray(rows["truncated"], dtype=bool),
        strict_success=np.asarray(rows["integrity_success"], dtype=bool),
        terminal_failure=np.asarray(rows["terminal_failure"], dtype=bool),
        safety_stop=np.asarray(rows["safety_stop"], dtype=bool),
        terminal_reason=np.asarray(rows["terminal_reason"], dtype=str),
        episode_ids=np.asarray(rows["episode_ids"], dtype=np.int64),
        obstacle_enabled=np.asarray(rows["obstacle"], dtype=bool),
        shaping_gamma=float(gamma),
        potential_reward_config_sha256=reward.config_sha256,
    )
    diagnostics = ScratchRolloutDiagnosticsV4Candidate(
        v9_env_reward_raw=np.asarray(rows["raw_env_reward"], dtype=np.float64),
        v9_surface_success=np.asarray(rows["surface_success"], dtype=bool),
        v4_integrity_success=np.asarray(rows["integrity_success"], dtype=bool),
        hard_safety_violation=np.asarray(rows["hard_violation"], dtype=bool),
        minimum_94_safety_only_block_distance_m=np.asarray(rows["minimum_94"], dtype=np.float64),
        minimum_contact_part_block_distance_by_role_m=np.asarray(rows["contact_minimums"], dtype=np.float64),
        unauthorized_contact_part_penetration_count=np.asarray(rows["unauthorized_count"], dtype=np.int64),
        privileged_safety_penalty=np.asarray(rows["safety_penalty"], dtype=np.float64),
    )
    result = ScratchRolloutBatchV4Candidate(
        ppo_batch=ppo_batch,
        integrity_diagnostics=diagnostics,
    )
    result.validate()
    return result


def scratch_v4_candidate_source_hashes() -> dict[str, str]:
    """Hash the complete V4 policy/reward implementation closure."""

    directory = Path(__file__).resolve().parent
    paths = {
        "ppo_utils_v1.py": directory / "ppo_utils_v1.py",
        "privileged_effect_state_v1.py": directory / "privileged_effect_state_v1.py",
        "scratch_ppo_v1.py": directory / "scratch_ppo_v1.py",
        "scratch_ppo_v3_candidate.py": directory / "scratch_ppo_v3_candidate.py",
        "scratch_ppo_v4_candidate.py": Path(__file__).resolve(),
    }
    return {name: _sha256_file(path) for name, path in paths.items()}


@dataclass(frozen=True)
class ScratchPPOV4CandidateProvenance:
    source_type: str
    checkpoint_format: str
    initialization_seed: int
    random_initialization: bool
    full_six_joint_action: bool
    expert_action_inputs: int
    controller_phase_inputs: int
    behavior_cloning_steps: int
    policy_observation_changed: bool
    contact_telemetry_is_reward_input: bool
    contact_telemetry_privilege: str
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
            "genesis_version": TRAINING_GENESIS_V4_CANDIDATE_VERSION,
            "actor_architecture": ACTOR_ARCHITECTURE,
            "critic_architecture": CRITIC_ARCHITECTURE,
            "policy_parameterization": POLICY_PARAMETERIZATION,
            **payload,
        }

    def validate(self, *, verify_current_sources: bool = True) -> None:
        exact = {
            "source_type": SOURCE_TYPE,
            "checkpoint_format": CHECKPOINT_FORMAT_V4_CANDIDATE,
            "random_initialization": True,
            "full_six_joint_action": True,
            "expert_action_inputs": 0,
            "controller_phase_inputs": 0,
            "behavior_cloning_steps": 0,
            "policy_observation_changed": False,
            "contact_telemetry_is_reward_input": True,
            "contact_telemetry_privilege": REWARD_INPUT_DISCLOSURE_V4_CANDIDATE[
                "contact_telemetry_privilege"
            ],
            "privileged_state_schema_sha256": PRIVILEGED_EFFECT_STATE_SCHEMA_SHA256,
            "potential_reward_version": POTENTIAL_REWARD_V4_CANDIDATE_VERSION,
        }
        for name, expected in exact.items():
            if getattr(self, name) != expected:
                raise ValueError(f"V4 candidate provenance {name} mismatch")
        if type(self.initialization_seed) is not int or self.initialization_seed < 0:
            raise ValueError("V4 candidate initialization seed is invalid")
        for name in (
            "actor_initial_state_sha256",
            "critic_initial_state_sha256",
            "privileged_state_schema_sha256",
            "potential_reward_config_sha256",
            "genesis_sha256",
        ):
            if not _is_sha256(getattr(self, name)):
                raise ValueError(f"V4 candidate provenance hash is malformed: {name}")
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(self.initialization_seed)
            actor = FullActionScratchActorV1()
            critic = PrivilegedEffectCriticV1()
        if self.actor_initial_state_sha256 != state_dict_sha256_v1(actor.state_dict()):
            raise ValueError("V4 candidate actor genesis is not canonical")
        if self.critic_initial_state_sha256 != state_dict_sha256_v1(critic.state_dict()):
            raise ValueError("V4 candidate critic genesis is not canonical")
        if self.potential_reward_config_sha256 != (ScratchPotentialRewardV4CandidateConfig().sha256()):
            raise ValueError("V4 candidate reward config provenance mismatch")
        expected_source_names = {
            "ppo_utils_v1.py",
            "privileged_effect_state_v1.py",
            "scratch_ppo_v1.py",
            "scratch_ppo_v3_candidate.py",
            "scratch_ppo_v4_candidate.py",
        }
        if set(self.trainer_source_hashes) != expected_source_names or any(
            not _is_sha256(value) for value in self.trainer_source_hashes.values()
        ):
            raise ValueError("V4 candidate trainer source hashes are malformed")
        if verify_current_sources and self.trainer_source_hashes != (scratch_v4_candidate_source_hashes()):
            raise ValueError("V4 candidate trainer sources changed since genesis")
        if self.genesis_sha256 != _canonical_sha256(self._genesis_payload()):
            raise ValueError("V4 candidate genesis hash mismatch")


@dataclass
class ScratchPPOV4CandidateBundle:
    actor: FullActionScratchActorV1
    critic: PrivilegedEffectCriticV1
    provenance: ScratchPPOV4CandidateProvenance
    potential_reward_config: ScratchPotentialRewardV4CandidateConfig


def initialize_scratch_ppo_v4_candidate(
    seed: int,
    *,
    device: str | torch.device = "cpu",
) -> ScratchPPOV4CandidateBundle:
    if type(seed) is not int or seed < 0:
        raise ValueError("V4 candidate seed must be a non-negative integer")
    reward_config = ScratchPotentialRewardV4CandidateConfig()
    reward_config.validate()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        actor = FullActionScratchActorV1()
        critic = PrivilegedEffectCriticV1()
    base: dict[str, Any] = {
        "source_type": SOURCE_TYPE,
        "checkpoint_format": CHECKPOINT_FORMAT_V4_CANDIDATE,
        "initialization_seed": seed,
        "random_initialization": True,
        "full_six_joint_action": True,
        "expert_action_inputs": 0,
        "controller_phase_inputs": 0,
        "behavior_cloning_steps": 0,
        "policy_observation_changed": False,
        "contact_telemetry_is_reward_input": True,
        "contact_telemetry_privilege": REWARD_INPUT_DISCLOSURE_V4_CANDIDATE["contact_telemetry_privilege"],
        "actor_initial_state_sha256": state_dict_sha256_v1(actor.state_dict()),
        "critic_initial_state_sha256": state_dict_sha256_v1(critic.state_dict()),
        "privileged_state_schema_sha256": PRIVILEGED_EFFECT_STATE_SCHEMA_SHA256,
        "potential_reward_version": POTENTIAL_REWARD_V4_CANDIDATE_VERSION,
        "potential_reward_config_sha256": reward_config.sha256(),
        "trainer_source_hashes": scratch_v4_candidate_source_hashes(),
    }
    provenance = ScratchPPOV4CandidateProvenance(
        **base,
        genesis_sha256=_canonical_sha256(
            {
                "genesis_version": TRAINING_GENESIS_V4_CANDIDATE_VERSION,
                "actor_architecture": ACTOR_ARCHITECTURE,
                "critic_architecture": CRITIC_ARCHITECTURE,
                "policy_parameterization": POLICY_PARAMETERIZATION,
                **base,
            }
        ),
    )
    provenance.validate()
    return ScratchPPOV4CandidateBundle(
        actor=actor.to(device),
        critic=critic.to(device),
        provenance=provenance,
        potential_reward_config=reward_config,
    )


def build_scratch_checkpoint_payload_v4_candidate(
    bundle: ScratchPPOV4CandidateBundle,
    config: ScratchPPOConfigV1,
    *,
    updates_completed: int,
) -> dict[str, Any]:
    bundle.provenance.validate()
    bundle.potential_reward_config.validate()
    config.validate()
    if type(updates_completed) is not int or updates_completed < 0:
        raise ValueError("V4 candidate updates_completed must be non-negative")
    actor_state = bundle.actor.state_dict()
    critic_state = bundle.critic.state_dict()
    return {
        "format": CHECKPOINT_FORMAT_V4_CANDIDATE,
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "source_type": SOURCE_TYPE,
        "actor_architecture": ACTOR_ARCHITECTURE,
        "critic_architecture": CRITIC_ARCHITECTURE,
        "policy_parameterization": POLICY_PARAMETERIZATION,
        "state_dim": PRIVILEGED_EFFECT_STATE_DIM,
        "action_dim": ACTION_DIM,
        "state_layout": list(PRIVILEGED_EFFECT_STATE_LAYOUT_V1),
        "state_schema_sha256": PRIVILEGED_EFFECT_STATE_SCHEMA_SHA256,
        "potential_reward_version": POTENTIAL_REWARD_V4_CANDIDATE_VERSION,
        "potential_reward_formula": POTENTIAL_REWARD_V4_CANDIDATE_FORMULA,
        "potential_reward_config": asdict(bundle.potential_reward_config),
        "potential_reward_config_sha256": (bundle.potential_reward_config.sha256()),
        "reward_input_disclosure": dict(REWARD_INPUT_DISCLOSURE_V4_CANDIDATE),
        "actor_state": actor_state,
        "critic_state": critic_state,
        "actor_state_sha256": state_dict_sha256_v1(actor_state),
        "critic_state_sha256": state_dict_sha256_v1(critic_state),
        "config": asdict(config),
        "updates_completed": updates_completed,
        "provenance": asdict(bundle.provenance),
        "production_admission": False,
        "physical_samples": 0,
        "physical_trials": 0,
        "admission_status": "candidate_not_admitted_for_causal_collection",
    }


def validate_scratch_checkpoint_payload_v4_candidate(
    payload: object,
) -> ScratchPPOV4CandidateProvenance:
    """Validate the exact V4 core schema and all self-declared state hashes."""

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
        "reward_input_disclosure",
        "actor_state",
        "critic_state",
        "actor_state_sha256",
        "critic_state_sha256",
        "config",
        "updates_completed",
        "provenance",
        "production_admission",
        "physical_samples",
        "physical_trials",
        "admission_status",
    }
    if not isinstance(payload, dict) or set(payload) != expected_fields:
        raise ValueError("V4 candidate checkpoint fields are incomplete or unexpected")
    exact = {
        "format": CHECKPOINT_FORMAT_V4_CANDIDATE,
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "source_type": SOURCE_TYPE,
        "actor_architecture": ACTOR_ARCHITECTURE,
        "critic_architecture": CRITIC_ARCHITECTURE,
        "policy_parameterization": POLICY_PARAMETERIZATION,
        "state_dim": PRIVILEGED_EFFECT_STATE_DIM,
        "action_dim": ACTION_DIM,
        "state_layout": list(PRIVILEGED_EFFECT_STATE_LAYOUT_V1),
        "state_schema_sha256": PRIVILEGED_EFFECT_STATE_SCHEMA_SHA256,
        "potential_reward_version": POTENTIAL_REWARD_V4_CANDIDATE_VERSION,
        "potential_reward_formula": POTENTIAL_REWARD_V4_CANDIDATE_FORMULA,
        "reward_input_disclosure": REWARD_INPUT_DISCLOSURE_V4_CANDIDATE,
        "production_admission": False,
        "physical_samples": 0,
        "physical_trials": 0,
        "admission_status": "candidate_not_admitted_for_causal_collection",
    }
    for name, expected in exact.items():
        if payload.get(name) != expected:
            raise ValueError(f"V4 candidate checkpoint {name} mismatch")
    reward_payload = payload.get("potential_reward_config")
    if not isinstance(reward_payload, dict) or set(reward_payload) != {
        field.name for field in fields(ScratchPotentialRewardV4CandidateConfig)
    }:
        raise ValueError("V4 candidate checkpoint reward config is malformed")
    try:
        reward_config = ScratchPotentialRewardV4CandidateConfig(**reward_payload)
        reward_config.validate()
    except (TypeError, ValueError) as error:
        raise ValueError("V4 candidate checkpoint reward config is invalid") from error
    if (
        asdict(reward_config) != reward_payload
        or payload.get("potential_reward_config_sha256") != reward_config.sha256()
    ):
        raise ValueError("V4 candidate checkpoint reward config hash mismatch")
    actor_state = payload.get("actor_state")
    critic_state = payload.get("critic_state")
    if not isinstance(actor_state, dict) or not isinstance(critic_state, dict):
        raise ValueError("V4 candidate checkpoint model states are missing")
    if state_dict_sha256_v1(actor_state) != payload.get("actor_state_sha256"):
        raise ValueError("V4 candidate checkpoint actor hash mismatch")
    if state_dict_sha256_v1(critic_state) != payload.get("critic_state_sha256"):
        raise ValueError("V4 candidate checkpoint critic hash mismatch")
    if any(
        not isinstance(value, torch.Tensor) or not bool(torch.all(torch.isfinite(value)).item())
        for state in (actor_state, critic_state)
        for value in state.values()
    ):
        raise ValueError("V4 candidate checkpoint model state is non-tensor or non-finite")
    canonical_actor = FullActionScratchActorV1()
    canonical_critic = PrivilegedEffectCriticV1()
    for name, state, canonical_state in (
        ("actor", actor_state, canonical_actor.state_dict()),
        ("critic", critic_state, canonical_critic.state_dict()),
    ):
        if set(state) != set(canonical_state) or any(
            state[key].shape != canonical_state[key].shape or state[key].dtype != canonical_state[key].dtype
            for key in canonical_state
        ):
            raise ValueError(f"V4 candidate checkpoint {name} tensor schema mismatch")
    try:
        canonical_actor.load_state_dict(actor_state, strict=True)
        canonical_critic.load_state_dict(critic_state, strict=True)
    except RuntimeError as error:
        raise ValueError("V4 candidate checkpoint model state schema mismatch") from error
    config_payload = payload.get("config")
    if not isinstance(config_payload, dict) or set(config_payload) != {
        field.name for field in fields(ScratchPPOConfigV1)
    }:
        raise ValueError("V4 candidate checkpoint PPO config is malformed")
    try:
        config = ScratchPPOConfigV1(**config_payload)
        config.validate()
    except (TypeError, ValueError) as error:
        raise ValueError("V4 candidate checkpoint PPO config is invalid") from error
    if asdict(config) != config_payload:
        raise ValueError("V4 candidate checkpoint PPO config is non-canonical")
    if type(payload.get("updates_completed")) is not int or payload["updates_completed"] < 0:
        raise ValueError("V4 candidate checkpoint update count is invalid")
    provenance_payload = payload.get("provenance")
    if not isinstance(provenance_payload, dict) or set(provenance_payload) != {
        field.name for field in fields(ScratchPPOV4CandidateProvenance)
    }:
        raise ValueError("V4 candidate checkpoint provenance is malformed")
    try:
        provenance = ScratchPPOV4CandidateProvenance(**provenance_payload)
        provenance.validate()
    except (TypeError, ValueError) as error:
        raise ValueError("V4 candidate checkpoint provenance is invalid") from error
    if provenance.potential_reward_config_sha256 != reward_config.sha256():
        raise ValueError("V4 candidate checkpoint provenance/config mismatch")
    return provenance


def train_one_scratch_update_v4_candidate(
    env: RealisticEdgeArmEnvV9,
    bundle: ScratchPPOV4CandidateBundle,
    config: ScratchPPOConfigV1,
    *,
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[
    ScratchRolloutBatchV4Candidate,
    PPOUpdateMetricsV1,
    torch.optim.Optimizer,
]:
    """Run exactly one PPO update whose collected rewards are exclusively V4."""

    config.validate()
    bundle.provenance.validate()
    bundle.potential_reward_config.validate()
    reward = ScratchPotentialRewardV4Candidate(bundle.potential_reward_config)
    reward._require_exact_v9(env)
    rollout = collect_scratch_rollout_v4_candidate(
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


__all__ = [
    "CHECKPOINT_FORMAT_V4_CANDIDATE",
    "POTENTIAL_REWARD_V4_CANDIDATE_FORMULA",
    "POTENTIAL_REWARD_V4_CANDIDATE_VERSION",
    "REWARD_INPUT_DISCLOSURE_V4_CANDIDATE",
    "ROLLOUT_DIAGNOSTICS_V4_CANDIDATE_FORMAT",
    "ScratchPotentialEvaluationV4Candidate",
    "ScratchPotentialRewardV4Candidate",
    "ScratchPotentialRewardV4CandidateConfig",
    "ScratchPotentialTransitionV4Candidate",
    "ScratchPPOV4CandidateBundle",
    "ScratchPPOV4CandidateProvenance",
    "ScratchRolloutBatchV4Candidate",
    "ScratchRolloutDiagnosticsV4Candidate",
    "ScratchSafetyEvidenceV4Candidate",
    "build_scratch_checkpoint_payload_v4_candidate",
    "collect_scratch_rollout_v4_candidate",
    "initialize_scratch_ppo_v4_candidate",
    "scratch_v4_candidate_source_hashes",
    "train_one_scratch_update_v4_candidate",
    "validate_scratch_checkpoint_payload_v4_candidate",
]
