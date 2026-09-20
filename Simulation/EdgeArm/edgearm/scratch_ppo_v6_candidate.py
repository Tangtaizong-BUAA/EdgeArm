"""Isolated Reward V6 candidate with one bit-exact float32 reward contract.

V6 keeps the exact V10 plant, V5 task/safety semantics, V1 actor/critic, and
unchanged PPO update.  Its only mathematical change is deliberate: every
operand persisted in a rollout is first canonicalized to float32, and both
collection and validation use the same explicitly ordered float32 operations.
This removes the V5 ambiguity where a float64 result and independently cast
float32 operands could describe two different rewards.

V6 has new reward, rollout, provenance, environment-identity, genesis, and
checkpoint identities.  It never relabels a V5 checkpoint or artifact.
"""

from __future__ import annotations

import ast
from collections import deque
import hashlib
import json
from dataclasses import asdict, dataclass, fields, is_dataclass, replace
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
import torch

from .causal_runtime_snapshot_v1 import (
    CAUSAL_RUNTIME_MAIN_LOGICAL_PATH,
    capture_causal_runtime_snapshot_v1,
    runtime_bundle_sha256_v1,
)
from .config import SCENE_PATH
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
from .scratch_ppo_v5_candidate import (
    ScratchPotentialEvaluationV5Candidate,
    ScratchPotentialRewardV5Candidate,
    ScratchPotentialRewardV5CandidateConfig,
    ScratchSafetyEvidenceV5Candidate,
)
from . import scratch_ppo_v5_candidate as _v5
from .production_env import ProductionMjcfBundleV1
from .sim2real_env_v10 import RealisticEdgeArmEnvV10, RealisticEnvV10Config


POTENTIAL_REWARD_V6_CANDIDATE_VERSION = (
    "edgearm-v10-privileged-integrity-task-progress-reward-v6-candidate-f32"
)
CHECKPOINT_FORMAT_V6_CANDIDATE = "edgearm-realism-v10-full-action-ppo-from-scratch-v6-candidate-f32"
TRAINING_GENESIS_V6_CANDIDATE_VERSION = "edgearm-v10-full-action-scratch-ppo-genesis-v6-candidate-f32"
REWARD_ARITHMETIC_V6_CANDIDATE_FORMAT = "edgearm-reward-arithmetic-canonical-float32-left-associated-v1"
POTENTIAL_REWARD_V6_CANDIDATE_FORMULA = (
    "f32_sub(f32_add(reward_before_potential_f32,"
    "f32_mul(gamma_f32,Phi_next_f32)),Phi_before_f32);"
    "reward_before_potential_f32=f32_add(f32_add("
    "clip(env_reward_f32,-16f,16f),safety_penalty_f32),terminal_outcome_f32);"
    "Phi_next_f32=0f iff credit_terminated else Phi_after_f32"
)
ROLLOUT_DIAGNOSTICS_V6_CANDIDATE_FORMAT = (
    "edgearm-scratch-ppo-v10-privileged-integrity-diagnostics-v6-candidate"
)
EPISODE_RANDOMIZATION_IDENTITY_V6_CANDIDATE_FORMAT = (
    "edgearm-v10-episode-randomized-model-and-domain-identity-v1"
)
EPISODE_STATIC_EXECUTION_CONTRACT_V6_CANDIDATE_FORMAT = (
    "edgearm-v10-stock-gripper-static-safety-geometry-contract-v1"
)
V10_ENVIRONMENT_IDENTITY_V6_CANDIDATE_FORMAT = (
    "edgearm-v10-pristine-static-execution-contract-and-f32-reward-identity-v3"
)
V10_PRISTINE_BINDING_V6_CANDIDATE_FORMAT = "edgearm-v10-fresh-environment-before-first-reset-binding-v1"
V10_EXECUTION_CONTRACT_V6_CANDIDATE_FORMAT = "edgearm-v10-complete-fresh-instance-derived-state-contract-v1"
REWARD_INPUT_DISCLOSURE_V6_CANDIDATE = {
    "policy_observation_changed": False,
    "expert_action_inputs": 0,
    "controller_phase_inputs": 0,
    "behavior_cloning_steps": 0,
    "contact_telemetry_is_reward_input": True,
    "contact_telemetry_privilege": ("simulator_privileged_scratch_training_only_not_policy_observation"),
    "tool_safety_geometry_is_reward_input": True,
    "deployment_requires_contact_telemetry": False,
    "environment_class": "exact RealisticEdgeArmEnvV10",
    "environment_config_class": "exact RealisticEnvV10Config",
    "persisted_reward_dtype": "float32",
    "persisted_reward_validation": "bit_exact",
    "reward_arithmetic_format": REWARD_ARITHMETIC_V6_CANDIDATE_FORMAT,
}

_TIP_ROLES = ("fixed_tip", "moving_tip")
_EXPECTED_SAFETY_GEOM_COUNT = 96
_EXPECTED_SAFETY_ONLY_GEOM_COUNT = 94
_SHA256_LENGTH = 64
_F32 = np.dtype(np.float32)
_V10_RAW_ENV_REWARD_ABSOLUTE_BOUND_V6 = 64.0
_SIMULATOR_DISTANCE_ABSOLUTE_BOUND_M_V6 = 2.0
_RESET_SEED_STRIDE_V6 = 1_000_003


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


def _canonical_f32_array(name: str, value: object) -> np.ndarray:
    source = np.asarray(value)
    if source.dtype == np.dtype(bool) or not np.issubdtype(source.dtype, np.number):
        raise TypeError(f"V6 {name} must be real numeric")
    if np.issubdtype(source.dtype, np.complexfloating):
        raise TypeError(f"V6 {name} must be real numeric")
    with np.errstate(over="ignore", invalid="ignore"):
        result = np.array(value, dtype=np.float32, copy=True)
    if not np.all(np.isfinite(result)):
        raise ValueError(f"V6 {name} is non-finite after float32 canonicalization")
    return result


def _same_f32_bits(left: object, right: object) -> bool:
    left_array = np.asarray(left)
    right_array = np.asarray(right)
    return (
        left_array.dtype == _F32
        and right_array.dtype == _F32
        and left_array.shape == right_array.shape
        and left_array.tobytes(order="C") == right_array.tobytes(order="C")
    )


def canonicalize_scratch_ppo_config_v6_candidate(
    config: ScratchPPOConfigV1,
) -> ScratchPPOConfigV1:
    """Return the actual V6 PPO config, including canonical float32 gamma.

    The returned dataclass still contains an ordinary Python ``float`` so it
    satisfies the unchanged V1 config schema, but that float is exactly the
    value represented by the persisted float32 gamma operand.
    """

    if type(config) is not ScratchPPOConfigV1:
        raise TypeError("V6 requires exact ScratchPPOConfigV1")
    canonical_types = ScratchPPOConfigV1()
    for item in fields(config):
        value = getattr(config, item.name)
        expected_type = type(getattr(canonical_types, item.name))
        if type(value) is not expected_type:
            raise TypeError(f"V6 PPO config {item.name} requires exact {expected_type.__name__}")
    config.validate()
    gamma = _canonical_f32_array("PPO config gamma", config.gamma)
    if gamma.shape != () or not np.float32(0.0) < gamma <= np.float32(1.0):
        raise ValueError("V6 PPO config gamma is invalid")
    canonical = replace(config, gamma=float(gamma))
    canonical.validate()
    return canonical


@dataclass(frozen=True)
class CanonicalPotentialRewardArithmeticV6Candidate:
    """Canonical operands and intermediates produced by one float32 path."""

    reward_before_potential: np.ndarray
    potential_before: np.ndarray
    potential_next_for_shaping: np.ndarray
    shaping_gamma: np.float32
    discounted_potential_next: np.ndarray
    reward_plus_discounted_potential: np.ndarray
    shaped_reward: np.ndarray


def canonical_potential_reward_arithmetic_v6_candidate(
    *,
    reward_before_potential: object,
    potential_before: object,
    potential_next_for_shaping: object,
    shaping_gamma: object,
) -> CanonicalPotentialRewardArithmeticV6Candidate:
    """Apply the sole V6 reward path, rounding after every named operation.

    Operation order intentionally matches ``ScratchRolloutBatchV1.validate``:
    ``(reward + gamma * next) - before``.  Inputs are copied/quantized and are
    never mutated.  The returned arrays always have dtype float32.
    """

    reward32 = _canonical_f32_array("reward_before_potential", reward_before_potential)
    before32 = _canonical_f32_array("potential_before", potential_before)
    next32 = _canonical_f32_array("potential_next_for_shaping", potential_next_for_shaping)
    if reward32.shape != before32.shape or reward32.shape != next32.shape:
        raise ValueError("V6 canonical reward operands must have identical shapes")
    gamma_array = _canonical_f32_array("shaping_gamma", shaping_gamma)
    if gamma_array.shape != ():
        raise ValueError("V6 shaping_gamma must be scalar")
    gamma32 = np.float32(gamma_array.item())
    if not np.float32(0.0) < gamma32 <= np.float32(1.0):
        raise ValueError("V6 shaping_gamma must be in (0, 1]")
    discounted = np.multiply(gamma32, next32, dtype=np.float32)
    reward_plus_discounted = np.add(reward32, discounted, dtype=np.float32)
    shaped = np.subtract(reward_plus_discounted, before32, dtype=np.float32)
    return CanonicalPotentialRewardArithmeticV6Candidate(
        reward_before_potential=reward32,
        potential_before=before32,
        potential_next_for_shaping=next32,
        shaping_gamma=gamma32,
        discounted_potential_next=discounted,
        reward_plus_discounted_potential=reward_plus_discounted,
        shaped_reward=shaped,
    )


@dataclass(frozen=True)
class ScratchPotentialRewardV6CandidateConfig(ScratchPotentialRewardV5CandidateConfig):
    """Frozen V5 coefficients under the V6 float32 arithmetic identity."""

    def validate(self) -> None:
        canonical = type(self)()
        for item in fields(self):
            value = getattr(self, item.name)
            expected = getattr(canonical, item.name)
            if type(value) is not type(expected) or not np.isfinite(value):
                raise ValueError(f"V6 candidate reward {item.name} must be finite numeric")
            if float(value) != float(expected):
                raise ValueError(f"V6 candidate reward {item.name} is frozen at {expected}")
        if not 0.0 <= self.penetration_tolerance_m < self.safety_only_block_clearance_m:
            raise ValueError("V6 penetration tolerance must be below clearance")
        if self.safety_depth_scale_m <= 0.0:
            raise ValueError("V6 safety depth scale must be positive")
        if not self.terminal_outcome_magnitude > (
            self.environment_reward_clip_abs + self.maximum_task_potential
        ):
            raise ValueError("V6 terminal outcome does not dominate shaping")

    def sha256(self) -> str:
        self.validate()
        return _canonical_sha256(
            {
                "version": POTENTIAL_REWARD_V6_CANDIDATE_VERSION,
                "formula": POTENTIAL_REWARD_V6_CANDIDATE_FORMULA,
                "arithmetic_format": REWARD_ARITHMETIC_V6_CANDIDATE_FORMAT,
                "environment_class": "exact RealisticEdgeArmEnvV10",
                "environment_config_class": "exact RealisticEnvV10Config",
                "ordered_tip_roles": list(_TIP_ROLES),
                "complete_safety_geom_count": _EXPECTED_SAFETY_GEOM_COUNT,
                "safety_only_geom_count": _EXPECTED_SAFETY_ONLY_GEOM_COUNT,
                "reward_input_disclosure": REWARD_INPUT_DISCLOSURE_V6_CANDIDATE,
                "config": asdict(self),
            }
        )


@dataclass(frozen=True)
class ScratchPotentialEvaluationV6Candidate:
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
class ScratchSafetyEvidenceV6Candidate(ScratchSafetyEvidenceV5Candidate):
    """V6-labelled exact-V10 safety evidence."""

    maximum_unauthorized_contact_penetration_depth_m: float = 0.0


@dataclass(frozen=True)
class ScratchPotentialTransitionV6Candidate:
    v10_env_reward_raw: float
    clipped_environment_reward: float
    reward_before_potential: float
    potential_before: float
    potential_after: float
    potential_next_for_shaping: float
    potential_shaping: float
    safety_penalty: float
    terminal_outcome_reward: float
    shaped_reward: float
    shaping_gamma: float
    reward_arithmetic_format: str
    env_terminated: bool
    env_truncated: bool
    credit_terminated: bool
    credit_truncated: bool
    v10_surface_success: bool
    v6_integrity_success: bool
    safety_credit_terminal: bool
    episode_safety_violation: bool
    terminal_reason: str


class ScratchPotentialRewardV6Candidate:
    """V5 task/safety meaning with V6-only canonical reward arithmetic."""

    version = POTENTIAL_REWARD_V6_CANDIDATE_VERSION
    formula = POTENTIAL_REWARD_V6_CANDIDATE_FORMULA
    reward_input_disclosure = REWARD_INPUT_DISCLOSURE_V6_CANDIDATE
    arithmetic_format = REWARD_ARITHMETIC_V6_CANDIDATE_FORMAT

    def __init__(
        self,
        config: ScratchPotentialRewardV6CandidateConfig | None = None,
    ) -> None:
        self.config = config or ScratchPotentialRewardV6CandidateConfig()
        if type(self.config) is not ScratchPotentialRewardV6CandidateConfig:
            raise TypeError("V6 reward requires exact ScratchPotentialRewardV6CandidateConfig")
        self.config.validate()
        self.config_sha256 = self.config.sha256()
        self._v5_kernel = ScratchPotentialRewardV5Candidate(
            ScratchPotentialRewardV5CandidateConfig(**asdict(self.config))
        )

    @staticmethod
    def _require_exact_v10(env: object) -> RealisticEdgeArmEnvV10:
        if type(env) is not RealisticEdgeArmEnvV10:
            raise TypeError("V6 candidate requires exact RealisticEdgeArmEnvV10")
        for name in (
            "sim2real_config",
            "realism_config",
            "contact_feasible_config",
            "stock_gripper_config",
            "stock_distal_tip_config",
            "joint_bounded_config",
            "config",
        ):
            candidate_config = getattr(env, name, None)
            if type(candidate_config) is not RealisticEnvV10Config:
                raise TypeError(f"V6 candidate requires exact RealisticEnvV10Config at {name}")
            if candidate_config is not env.config:
                raise RuntimeError(f"V6 exact V10 config alias diverged from env.config at {name}")
        return env

    def evaluate(self, env: RealisticEdgeArmEnvV10) -> ScratchPotentialEvaluationV6Candidate:
        exact = self._require_exact_v10(env)
        evaluation: ScratchPotentialEvaluationV5Candidate = self._v5_kernel.evaluate(exact)
        return ScratchPotentialEvaluationV6Candidate(**asdict(evaluation))

    def evaluate_transition_safety(
        self,
        env: RealisticEdgeArmEnvV10,
        info: dict[str, Any],
    ) -> ScratchSafetyEvidenceV6Candidate:
        exact = self._require_exact_v10(env)
        evidence = self._v5_kernel.evaluate_transition_safety(exact, info)
        trace = info.get("physics_substep_contact_v1")
        if type(trace) is not dict:
            raise RuntimeError("V6 safety evidence lost its validated physics trace")
        safety_ids = tuple(int(value) for value in exact._ids["tool_safety_geoms"])
        contact_ids = tuple(int(value) for value in exact._ids["tool_contact_geoms"])
        contact_indices = tuple(safety_ids.index(value) for value in contact_ids)
        block = np.asarray(trace["tool_safety_block_signed_distance_m"], dtype=np.float64)
        raw = np.asarray(trace["tool_block_contact_count_by_role"], dtype=np.int64)
        invalid = np.asarray(trace["invalid_tool_block_contact_count_by_role"], dtype=np.int64)
        all_valid = (
            np.asarray(
                trace["all_tool_block_contacts_geometrically_valid_by_role"],
                dtype=np.float64,
            )
            == 1.0
        )
        maximum_unauthorized_depth = 0.0
        derived_count = 0
        for role_index, contact_index in enumerate(contact_indices):
            distances = block[:, contact_index]
            penetrating = distances < -self.config.penetration_tolerance_m
            authorized = (raw[:, role_index] > 0) & all_valid[:, role_index] & (invalid[:, role_index] == 0)
            unauthorized = penetrating & ~authorized
            derived_count += int(np.count_nonzero(unauthorized))
            if np.any(unauthorized):
                maximum_unauthorized_depth = max(
                    maximum_unauthorized_depth,
                    float(np.max(-distances[unauthorized])),
                )
        normalized_unauthorized = float(
            np.clip(
                max(
                    maximum_unauthorized_depth - self.config.penetration_tolerance_m,
                    0.0,
                )
                / self.config.safety_depth_scale_m,
                0.0,
                1.0,
            )
        )
        if derived_count != evidence.unauthorized_contact_part_penetration_count or np.float32(
            normalized_unauthorized
        ) != np.float32(evidence.normalized_unauthorized_contact_cost):
            raise RuntimeError("V6 unauthorized safety operands disagree with V5 kernel")
        return ScratchSafetyEvidenceV6Candidate(
            **asdict(evidence),
            maximum_unauthorized_contact_penetration_depth_m=(maximum_unauthorized_depth),
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
        v10_surface_success: bool,
        env_terminal_failure: bool,
        safety: ScratchSafetyEvidenceV6Candidate,
        episode_safety_violation_before: bool,
    ) -> ScratchPotentialTransitionV6Candidate:
        if type(safety) is not ScratchSafetyEvidenceV6Candidate:
            raise TypeError("V6 shaping requires exact V6 safety evidence")
        numeric = np.asarray([env_reward, potential_before, potential_after, gamma], dtype=np.float64)
        if not np.all(np.isfinite(numeric)) or not 0.0 < gamma <= 1.0:
            raise ValueError("V6 shaping inputs are invalid")
        maximum = self.config.maximum_task_potential
        if not -1.0e-9 <= potential_before <= maximum + 1.0e-9:
            raise ValueError("V6 potential_before is outside the proven range")
        if not -1.0e-9 <= potential_after <= maximum + 1.0e-9:
            raise ValueError("V6 potential_after is outside the proven range")
        if env_terminated and env_truncated:
            raise ValueError("V6 transition cannot terminate and truncate")
        if v10_surface_success and (not env_terminated or env_terminal_failure):
            raise ValueError("V10 surface success must be a non-failure termination")

        integrity_success = bool(
            v10_surface_success and not episode_safety_violation_before and not safety.hard_safety_violation
        )
        episode_safety_violation = bool(episode_safety_violation_before or safety.hard_safety_violation)
        safety_terminal = bool(
            episode_safety_violation or env_terminal_failure or (env_terminated and not integrity_success)
        )
        credit_terminated = bool(integrity_success or safety_terminal)
        credit_truncated = bool(env_truncated and not credit_terminated)

        env_reward32 = np.float32(env_reward)
        clip32 = np.float32(self.config.environment_reward_clip_abs)
        clipped32 = np.clip(env_reward32, -clip32, clip32).astype(np.float32)
        normalized_safety32 = np.float32(safety.normalized_safety_cost)
        safety32 = np.multiply(
            np.float32(-self.config.privileged_safety_penalty_coefficient),
            normalized_safety32,
            dtype=np.float32,
        )
        if integrity_success:
            terminal32 = np.float32(self.config.terminal_outcome_magnitude)
            terminal_reason = "v6_integrity_success"
        elif safety_terminal:
            terminal32 = np.float32(-self.config.terminal_outcome_magnitude)
            if safety.hard_safety_violation:
                reason = safety.hard_safety_reason
            elif episode_safety_violation_before:
                reason = "prior_episode_safety_violation"
            elif env_terminal_failure:
                reason = "environment_terminal_failure"
            else:
                reason = "environment_termination_without_integrity_success"
            terminal_reason = f"v6_safety_terminal:{reason}"
        elif credit_truncated:
            terminal32 = np.float32(0.0)
            terminal_reason = "time_limit"
        else:
            terminal32 = np.float32(0.0)
            terminal_reason = "nonterminal"

        clipped_plus_safety = np.add(clipped32, safety32, dtype=np.float32)
        reward_before32 = np.add(clipped_plus_safety, terminal32, dtype=np.float32)
        before32 = np.float32(potential_before)
        after32 = np.float32(potential_after)
        next32 = np.float32(0.0) if credit_terminated else after32
        arithmetic = canonical_potential_reward_arithmetic_v6_candidate(
            reward_before_potential=reward_before32,
            potential_before=before32,
            potential_next_for_shaping=next32,
            shaping_gamma=gamma,
        )
        potential_shaping32 = np.subtract(
            arithmetic.discounted_potential_next,
            arithmetic.potential_before,
            dtype=np.float32,
        )
        return ScratchPotentialTransitionV6Candidate(
            v10_env_reward_raw=float(env_reward),
            clipped_environment_reward=float(clipped32),
            reward_before_potential=float(arithmetic.reward_before_potential),
            potential_before=float(arithmetic.potential_before),
            potential_after=float(after32),
            potential_next_for_shaping=float(arithmetic.potential_next_for_shaping),
            potential_shaping=float(potential_shaping32),
            safety_penalty=float(safety32),
            terminal_outcome_reward=float(terminal32),
            shaped_reward=float(arithmetic.shaped_reward),
            shaping_gamma=float(arithmetic.shaping_gamma),
            reward_arithmetic_format=REWARD_ARITHMETIC_V6_CANDIDATE_FORMAT,
            env_terminated=bool(env_terminated),
            env_truncated=bool(env_truncated),
            credit_terminated=credit_terminated,
            credit_truncated=credit_truncated,
            v10_surface_success=bool(v10_surface_success),
            v6_integrity_success=integrity_success,
            safety_credit_terminal=safety_terminal,
            episode_safety_violation=episode_safety_violation,
            terminal_reason=terminal_reason,
        )


@dataclass(frozen=True)
class ScratchRolloutDiagnosticsV6Candidate:
    v10_env_reward_raw: np.ndarray
    clipped_environment_reward: np.ndarray
    terminal_outcome_reward: np.ndarray
    v10_surface_success: np.ndarray
    v6_integrity_success: np.ndarray
    hard_safety_violation: np.ndarray
    safety_only_clearance_violation: np.ndarray
    unauthorized_contact_part_penetration: np.ndarray
    full_safety_desk_penetration: np.ndarray
    physics_substeps: np.ndarray
    stress_enabled: np.ndarray
    minimum_94_safety_only_block_distance_m: np.ndarray
    minimum_contact_part_block_distance_by_role_m: np.ndarray
    minimum_full_safety_desk_signed_distance_m: np.ndarray
    unauthorized_contact_part_penetration_count: np.ndarray
    maximum_unauthorized_contact_penetration_depth_m: np.ndarray
    normalized_safety_only_block_cost: np.ndarray
    normalized_unauthorized_contact_cost: np.ndarray
    normalized_full_safety_desk_cost: np.ndarray
    normalized_safety_cost: np.ndarray
    privileged_safety_penalty: np.ndarray
    format: str = ROLLOUT_DIAGNOSTICS_V6_CANDIDATE_FORMAT
    privileged_reward_input: bool = True

    def validate(self, transition_count: int) -> None:
        if type(transition_count) is not int or transition_count < 1:
            raise ValueError("V6 diagnostics transition count is invalid")
        expected = {
            "v10_env_reward_raw": (transition_count,),
            "clipped_environment_reward": (transition_count,),
            "terminal_outcome_reward": (transition_count,),
            "v10_surface_success": (transition_count,),
            "v6_integrity_success": (transition_count,),
            "hard_safety_violation": (transition_count,),
            "safety_only_clearance_violation": (transition_count,),
            "unauthorized_contact_part_penetration": (transition_count,),
            "full_safety_desk_penetration": (transition_count,),
            "physics_substeps": (transition_count,),
            "stress_enabled": (transition_count,),
            "minimum_94_safety_only_block_distance_m": (transition_count,),
            "minimum_contact_part_block_distance_by_role_m": (transition_count, 2),
            "minimum_full_safety_desk_signed_distance_m": (transition_count,),
            "unauthorized_contact_part_penetration_count": (transition_count,),
            "maximum_unauthorized_contact_penetration_depth_m": (transition_count,),
            "normalized_safety_only_block_cost": (transition_count,),
            "normalized_unauthorized_contact_cost": (transition_count,),
            "normalized_full_safety_desk_cost": (transition_count,),
            "normalized_safety_cost": (transition_count,),
            "privileged_safety_penalty": (transition_count,),
        }
        for name, shape in expected.items():
            if np.asarray(getattr(self, name)).shape != shape:
                raise ValueError(f"V6 diagnostics {name} shape mismatch")
        for name in (
            "v10_env_reward_raw",
            "clipped_environment_reward",
            "terminal_outcome_reward",
            "minimum_94_safety_only_block_distance_m",
            "minimum_contact_part_block_distance_by_role_m",
            "minimum_full_safety_desk_signed_distance_m",
            "maximum_unauthorized_contact_penetration_depth_m",
            "normalized_safety_only_block_cost",
            "normalized_unauthorized_contact_cost",
            "normalized_full_safety_desk_cost",
            "normalized_safety_cost",
            "privileged_safety_penalty",
        ):
            if not np.all(np.isfinite(np.asarray(getattr(self, name)))):
                raise ValueError(f"V6 diagnostics {name} is non-finite")
        for name in (
            "v10_env_reward_raw",
            "minimum_94_safety_only_block_distance_m",
            "minimum_contact_part_block_distance_by_role_m",
            "minimum_full_safety_desk_signed_distance_m",
            "maximum_unauthorized_contact_penetration_depth_m",
        ):
            if np.asarray(getattr(self, name)).dtype != np.dtype(np.float64):
                raise ValueError(f"V6 diagnostics {name} must be canonical float64")
        for name in (
            "v10_surface_success",
            "v6_integrity_success",
            "hard_safety_violation",
            "safety_only_clearance_violation",
            "unauthorized_contact_part_penetration",
            "full_safety_desk_penetration",
            "stress_enabled",
        ):
            if np.asarray(getattr(self, name)).dtype != np.dtype(bool):
                raise ValueError(f"V6 diagnostics {name} must be boolean")
        counts = np.asarray(self.unauthorized_contact_part_penetration_count)
        physics_substeps = np.asarray(self.physics_substeps)
        if (
            counts.dtype != np.dtype(np.int64)
            or np.any(counts < 0)
            or physics_substeps.dtype != np.dtype(np.int64)
            or np.any(physics_substeps < 1)
            or np.any(counts > 2 * physics_substeps)
        ):
            raise ValueError("V6 unauthorized contact counts are invalid")
        raw = np.asarray(self.v10_env_reward_raw, dtype=np.float64)
        if np.any(np.abs(raw) > _V10_RAW_ENV_REWARD_ABSOLUTE_BOUND_V6):
            raise ValueError("V6 raw environment reward is outside its audit bound")
        for name in (
            "minimum_94_safety_only_block_distance_m",
            "minimum_contact_part_block_distance_by_role_m",
            "minimum_full_safety_desk_signed_distance_m",
            "maximum_unauthorized_contact_penetration_depth_m",
        ):
            if np.any(
                np.abs(np.asarray(getattr(self, name), dtype=np.float64))
                > _SIMULATOR_DISTANCE_ABSOLUTE_BOUND_M_V6
            ):
                raise ValueError(f"V6 diagnostics {name} is outside its audit bound")
        for name in (
            "clipped_environment_reward",
            "terminal_outcome_reward",
            "normalized_safety_only_block_cost",
            "normalized_unauthorized_contact_cost",
            "normalized_full_safety_desk_cost",
            "normalized_safety_cost",
            "privileged_safety_penalty",
        ):
            if np.asarray(getattr(self, name)).dtype != _F32:
                raise ValueError(f"V6 diagnostics {name} must be canonical float32")
        for name in (
            "normalized_safety_only_block_cost",
            "normalized_unauthorized_contact_cost",
            "normalized_full_safety_desk_cost",
            "normalized_safety_cost",
        ):
            values = np.asarray(getattr(self, name), dtype=np.float32)
            if np.any((values < np.float32(0.0)) | (values > np.float32(1.0))):
                raise ValueError(f"V6 diagnostics {name} escaped [0, 1]")
        clearance = np.asarray(self.safety_only_clearance_violation, dtype=bool)
        unauthorized_flag = np.asarray(self.unauthorized_contact_part_penetration, dtype=bool)
        desk = np.asarray(self.full_safety_desk_penetration, dtype=bool)
        hard = np.asarray(self.hard_safety_violation, dtype=bool)
        if not np.array_equal(hard, clearance | unauthorized_flag | desk):
            raise ValueError("V6 hard safety flag differs from canonical components")
        if not np.array_equal(unauthorized_flag, counts > 0):
            raise ValueError("V6 unauthorized contact flag/count mismatch")
        reward_config = ScratchPotentialRewardV6CandidateConfig()
        minimum_94 = np.asarray(self.minimum_94_safety_only_block_distance_m, dtype=np.float64)
        minimum_desk = np.asarray(self.minimum_full_safety_desk_signed_distance_m, dtype=np.float64)
        clearance_expected = minimum_94 < reward_config.safety_only_block_clearance_m
        desk_expected = minimum_desk < -reward_config.penetration_tolerance_m
        if not np.array_equal(clearance, clearance_expected):
            raise ValueError("V6 safety-only clearance flag/distance mismatch")
        if not np.array_equal(desk, desk_expected):
            raise ValueError("V6 desk penetration flag/distance mismatch")
        normalized_safety_only_expected = np.clip(
            np.maximum(
                reward_config.safety_only_block_clearance_m - minimum_94,
                0.0,
            )
            / reward_config.safety_depth_scale_m,
            0.0,
            1.0,
        ).astype(np.float32)
        normalized_desk_expected = np.clip(
            np.maximum(
                -minimum_desk - reward_config.penetration_tolerance_m,
                0.0,
            )
            / reward_config.safety_depth_scale_m,
            0.0,
            1.0,
        ).astype(np.float32)
        if not _same_f32_bits(
            self.normalized_safety_only_block_cost,
            normalized_safety_only_expected,
        ):
            raise ValueError("V6 safety-only normalized cost is not bit exact")
        if not _same_f32_bits(
            self.normalized_full_safety_desk_cost,
            normalized_desk_expected,
        ):
            raise ValueError("V6 desk normalized cost is not bit exact")
        normalized_unauthorized = np.asarray(self.normalized_unauthorized_contact_cost, dtype=np.float32)
        maximum_unauthorized_depth = np.asarray(
            self.maximum_unauthorized_contact_penetration_depth_m,
            dtype=np.float64,
        )
        if np.any(maximum_unauthorized_depth < 0.0):
            raise ValueError("V6 maximum unauthorized penetration depth is negative")
        unauthorized_expected = maximum_unauthorized_depth > (reward_config.penetration_tolerance_m)
        if not np.array_equal(unauthorized_flag, unauthorized_expected):
            raise ValueError("V6 unauthorized contact flag/depth mismatch")
        contact_minimums = np.asarray(
            self.minimum_contact_part_block_distance_by_role_m,
            dtype=np.float64,
        )
        maximum_possible_depth = np.maximum(-np.min(contact_minimums, axis=1), 0.0)
        if np.any(maximum_unauthorized_depth > maximum_possible_depth):
            raise ValueError("V6 unauthorized depth exceeds contact distance evidence")
        normalized_unauthorized_expected = np.clip(
            np.maximum(
                maximum_unauthorized_depth - reward_config.penetration_tolerance_m,
                0.0,
            )
            / reward_config.safety_depth_scale_m,
            0.0,
            1.0,
        ).astype(np.float32)
        if not _same_f32_bits(
            normalized_unauthorized,
            normalized_unauthorized_expected,
        ):
            raise ValueError("V6 unauthorized normalized cost is not bit exact")
        if np.any(self.v6_integrity_success & ~self.v10_surface_success):
            raise ValueError("V6 integrity success must imply V10 surface success")
        if np.any(self.v6_integrity_success & self.hard_safety_violation):
            raise ValueError("V6 integrity success cannot include a hard violation")
        if self.format != ROLLOUT_DIAGNOSTICS_V6_CANDIDATE_FORMAT:
            raise ValueError("V6 diagnostics format mismatch")
        if self.privileged_reward_input is not True:
            raise ValueError("V6 diagnostics must declare reward privilege")

    def metric_fields(self) -> dict[str, object]:
        count = int(np.asarray(self.v10_env_reward_raw).size)
        self.validate(count)
        return {
            "rollout_v10_surface_success_count": int(np.count_nonzero(self.v10_surface_success)),
            "rollout_v6_integrity_success_count": int(np.count_nonzero(self.v6_integrity_success)),
            "rollout_v6_hard_safety_violation_count": int(np.count_nonzero(self.hard_safety_violation)),
            "rollout_v6_minimum_94_safety_only_block_distance_m": float(
                np.min(self.minimum_94_safety_only_block_distance_m)
            ),
            "rollout_v6_unauthorized_contact_part_penetration_count": int(
                np.sum(self.unauthorized_contact_part_penetration_count)
            ),
        }


@dataclass(frozen=True)
class ScratchEpisodeRandomizationIdentityV6Candidate:
    format: str
    episode_id: int
    requested_reset_seed: int
    accepted_reset_seed: int
    reset_resample_attempt_index: int
    reset_attempt_count: int
    command_epoch: int
    obstacle_enabled: bool
    stress_enabled: bool
    compiled_model_sha256: str
    compiled_model_bytes: int
    static_execution_contract_sha256: str
    post_reset_execution_contract_sha256: str
    canonical_reset_replay_verified: bool
    episode_domain_sha256: str
    episode_sim2real_sha256: str
    episode_v6_sha256: str
    identity_sha256: str

    def _identity_payload(self) -> dict[str, object]:
        payload = asdict(self)
        payload.pop("identity_sha256")
        return payload

    def validate(self) -> None:
        if self.format != EPISODE_RANDOMIZATION_IDENTITY_V6_CANDIDATE_FORMAT:
            raise ValueError("V6 episode randomization format mismatch")
        for name in (
            "episode_id",
            "requested_reset_seed",
            "accepted_reset_seed",
            "reset_resample_attempt_index",
            "reset_attempt_count",
        ):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"V6 episode randomization {name} is invalid")
        if self.reset_attempt_count != self.reset_resample_attempt_index + 1:
            raise ValueError("V6 episode reset attempt evidence is inconsistent")
        expected_accepted_seed = int(
            (self.requested_reset_seed + self.reset_resample_attempt_index * _RESET_SEED_STRIDE_V6)
            % np.iinfo(np.int64).max
        )
        if self.accepted_reset_seed != expected_accepted_seed:
            raise ValueError("V6 episode accepted reset seed is non-canonical")
        if type(self.command_epoch) is not int or self.command_epoch < 1:
            raise ValueError("V6 episode randomization command epoch is invalid")
        for name in ("obstacle_enabled", "stress_enabled"):
            if type(getattr(self, name)) is not bool:
                raise TypeError(f"V6 episode randomization {name} must be bool")
        if self.canonical_reset_replay_verified is not True:
            raise ValueError("V6 episode canonical reset replay was not verified")
        for name in (
            "compiled_model_sha256",
            "static_execution_contract_sha256",
            "post_reset_execution_contract_sha256",
            "episode_domain_sha256",
            "episode_sim2real_sha256",
            "episode_v6_sha256",
            "identity_sha256",
        ):
            if not _is_sha256(getattr(self, name)):
                raise ValueError(f"V6 episode randomization {name} is malformed")
        if type(self.compiled_model_bytes) is not int or self.compiled_model_bytes < 1:
            raise ValueError("V6 episode randomized model byte count is invalid")
        if self.identity_sha256 != _canonical_sha256(self._identity_payload()):
            raise ValueError("V6 episode randomization identity hash mismatch")


def _build_episode_randomization_identity_v6(
    env: RealisticEdgeArmEnvV10,
    *,
    episode_id: int,
    requested_reset_seed: int,
    obstacle_enabled: bool,
    stress_enabled: bool,
    expected_static_execution_contract_sha256: str,
) -> ScratchEpisodeRandomizationIdentityV6Candidate:
    exact = ScratchPotentialRewardV6Candidate._require_exact_v10(env)
    if (
        type(episode_id) is not int
        or episode_id < 0
        or type(requested_reset_seed) is not int
        or requested_reset_seed < 0
        or type(obstacle_enabled) is not bool
        or type(stress_enabled) is not bool
        or not _is_sha256(expected_static_execution_contract_sha256)
    ):
        raise ValueError("V6 episode reset evidence is invalid")
    if (
        type(exact.seed) is not int
        or exact.step_count != 0
        or exact.estop is not False
        or type(exact.command_epoch) is not int
        or exact.command_epoch < 1
        or bool(exact.obstacle_enabled) is not obstacle_enabled
        or not isinstance(exact.episode_domain, dict)
        or not exact.episode_domain
        or not isinstance(exact.episode_sim2real, dict)
        or not isinstance(exact._episode_v6, dict)
        or not exact._episode_v6
    ):
        raise RuntimeError("V6 reset did not establish canonical episode evidence")
    realism_v7 = exact.episode_domain.get("realism_v7")
    reset_audit = exact._reset_collision_audit
    if type(realism_v7) is not dict or type(reset_audit) is not dict:
        raise RuntimeError("V6 reset resampling audit is missing")
    accepted_reset_seed = realism_v7.get("accepted_candidate_seed")
    reset_attempt_count = realism_v7.get("reset_attempt_count")
    reset_resample_attempt_index = reset_audit.get("resample_attempt_index")
    rejected_attempts = realism_v7.get("rejected_reset_attempts")
    if (
        realism_v7.get("requested_seed") != requested_reset_seed
        or reset_audit.get("requested_seed") != requested_reset_seed
        or type(accepted_reset_seed) is not int
        or exact.seed != accepted_reset_seed
        or exact.episode_sim2real.get("seed") != accepted_reset_seed
        or reset_audit.get("accepted_candidate_seed") != accepted_reset_seed
        or type(reset_attempt_count) is not int
        or reset_attempt_count < 1
        or type(reset_resample_attempt_index) is not int
        or reset_resample_attempt_index != reset_attempt_count - 1
        or type(rejected_attempts) is not list
        or len(rejected_attempts) != reset_resample_attempt_index
    ):
        raise RuntimeError("V6 reset resampling evidence is inconsistent")
    expected_candidate_seeds = [
        int((requested_reset_seed + attempt * _RESET_SEED_STRIDE_V6) % np.iinfo(np.int64).max)
        for attempt in range(reset_attempt_count)
    ]
    if [entry.get("candidate_seed") for entry in rejected_attempts] != expected_candidate_seeds[
        :-1
    ] or accepted_reset_seed != expected_candidate_seeds[-1]:
        raise RuntimeError("V6 reset candidate seed sequence is non-canonical")
    command_delay = exact.episode_sim2real.get("command_delay_steps")
    if (
        type(command_delay) is not int
        or command_delay < 0
        or len(exact._command_queue) != command_delay
        or exact.episode_sim2real.get("command_epoch") != exact.command_epoch
        or type(exact.current_stress) is not bool
        or exact.current_stress is not stress_enabled
        or exact.episode_domain.get("stress") is not stress_enabled
    ):
        raise RuntimeError("V6 reset command-delay evidence is inconsistent")
    if type(exact.model) is not mujoco.MjModel or type(exact.data) is not mujoco.MjData:
        raise TypeError("V6 randomized episode requires exact MuJoCo runtime objects")
    if exact.data.model is not exact.model:
        raise RuntimeError("V6 randomized episode data is bound to a foreign model")
    static_execution_contract = _episode_static_execution_contract_v6(exact)
    static_execution_contract_sha256 = _canonical_sha256(static_execution_contract)
    if static_execution_contract_sha256 != expected_static_execution_contract_sha256:
        raise RuntimeError("V6 reset changed static safety geometry or execution identity")
    _source_mode, main_logical_path, files = _v5._runtime_files_for_exact_v10(exact)
    reference = RealisticEdgeArmEnvV10(
        RealisticEnvV10Config(**asdict(exact.config)),
        seed=requested_reset_seed,
        model_scene_bundle=ProductionMjcfBundleV1(
            main_logical_path=main_logical_path,
            files=files,
        ),
    )
    ScratchPotentialRewardV6Candidate._require_exact_v10(reference)
    previous_command_epoch = exact.command_epoch - reset_attempt_count
    if previous_command_epoch < 0:
        raise RuntimeError("V6 cumulative reset command epoch is invalid")
    reference._command_epoch = previous_command_epoch
    reference.reset(
        seed=requested_reset_seed,
        obstacle=obstacle_enabled,
        stress=stress_enabled,
    )
    reference_v7 = reference.episode_domain.get("realism_v7")
    if (
        type(reference_v7) is not dict
        or reference.seed != accepted_reset_seed
        or reference_v7.get("reset_attempt_count") != reset_attempt_count
        or reference._reset_collision_audit.get("resample_attempt_index") != reset_resample_attempt_index
    ):
        raise RuntimeError("V6 canonical reset replay chose different evidence")
    compiled_sha256, compiled_bytes = _v5._compiled_model_identity_v5(exact)
    reference_compiled_sha256, reference_compiled_bytes = _v5._compiled_model_identity_v5(reference)
    if compiled_sha256 != reference_compiled_sha256 or compiled_bytes != reference_compiled_bytes:
        raise RuntimeError("V6 randomized model differs from canonical reset replay")
    reference_static_execution_contract = _episode_static_execution_contract_v6(reference)
    if static_execution_contract != reference_static_execution_contract:
        raise RuntimeError("V6 static execution state differs from canonical reset replay")
    post_reset_execution_contract = _post_reset_execution_contract_v6(exact)
    reference_post_reset_execution_contract = _post_reset_execution_contract_v6(reference)
    if post_reset_execution_contract != reference_post_reset_execution_contract:
        raise RuntimeError("V6 post-reset state differs from canonical reset replay")
    post_reset_execution_contract_sha256 = _canonical_sha256(post_reset_execution_contract)
    base: dict[str, object] = {
        "format": EPISODE_RANDOMIZATION_IDENTITY_V6_CANDIDATE_FORMAT,
        "episode_id": episode_id,
        "requested_reset_seed": requested_reset_seed,
        "accepted_reset_seed": accepted_reset_seed,
        "reset_resample_attempt_index": reset_resample_attempt_index,
        "reset_attempt_count": reset_attempt_count,
        "command_epoch": exact.command_epoch,
        "obstacle_enabled": obstacle_enabled,
        "stress_enabled": stress_enabled,
        "compiled_model_sha256": compiled_sha256,
        "compiled_model_bytes": compiled_bytes,
        "static_execution_contract_sha256": static_execution_contract_sha256,
        "post_reset_execution_contract_sha256": (post_reset_execution_contract_sha256),
        "canonical_reset_replay_verified": True,
        "episode_domain_sha256": _canonical_sha256(_execution_contract_value_v6(exact.episode_domain)),
        "episode_sim2real_sha256": _canonical_sha256(_execution_contract_value_v6(exact.episode_sim2real)),
        "episode_v6_sha256": _canonical_sha256(_execution_contract_value_v6(exact._episode_v6)),
    }
    identity = ScratchEpisodeRandomizationIdentityV6Candidate(
        **base,
        identity_sha256=_canonical_sha256(base),
    )
    identity.validate()
    return identity


@dataclass(frozen=True)
class ScratchRolloutBatchV6Candidate:
    ppo_batch: ScratchRolloutBatchV1
    integrity_diagnostics: ScratchRolloutDiagnosticsV6Candidate
    episode_randomization_records: tuple[ScratchEpisodeRandomizationIdentityV6Candidate, ...]
    rollout_seed: int
    potential_reward_version: str = POTENTIAL_REWARD_V6_CANDIDATE_VERSION
    reward_arithmetic_format: str = REWARD_ARITHMETIC_V6_CANDIDATE_FORMAT

    def __getattr__(self, name: str) -> Any:
        return getattr(self.ppo_batch, name)

    def validate(self) -> None:
        if type(self.ppo_batch) is not ScratchRolloutBatchV1:
            raise TypeError("V6 rollout requires exact ScratchRolloutBatchV1 storage")
        if self.potential_reward_version != POTENTIAL_REWARD_V6_CANDIDATE_VERSION:
            raise ValueError("V6 rollout reward version mismatch")
        if self.reward_arithmetic_format != REWARD_ARITHMETIC_V6_CANDIDATE_FORMAT:
            raise ValueError("V6 rollout reward arithmetic format mismatch")
        if type(self.rollout_seed) is not int or self.rollout_seed < 0:
            raise ValueError("V6 rollout seed is invalid")
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
            if np.asarray(getattr(self.ppo_batch, name)).dtype != _F32:
                raise ValueError(f"V6 rollout {name} must be canonical float32")
        if np.asarray(self.ppo_batch.episode_ids).dtype != np.dtype(np.int64):
            raise ValueError("V6 rollout episode_ids must be canonical int64")
        gamma32 = _canonical_f32_array("rollout shaping_gamma", self.shaping_gamma)
        if gamma32.shape != () or float(gamma32) != self.shaping_gamma:
            raise ValueError("V6 rollout shaping_gamma is not canonical float32")
        reward_config = ScratchPotentialRewardV6CandidateConfig()
        reward_config.validate()
        if self.potential_reward_config_sha256 != reward_config.sha256():
            raise ValueError("V6 rollout potential reward config hash mismatch")
        maximum32 = np.float32(reward_config.maximum_task_potential)
        for name in ("potential_before", "potential_after"):
            values = np.asarray(getattr(self.ppo_batch, name), dtype=np.float32)
            if np.any((values < np.float32(0.0)) | (values > maximum32)):
                raise ValueError(f"V6 rollout {name} escaped the frozen potential range")
        expected_next = np.where(
            self.terminated,
            np.float32(0.0),
            self.potential_after,
        ).astype(np.float32, copy=False)
        if not _same_f32_bits(self.potential_next_for_shaping, expected_next):
            raise ValueError("V6 rollout potential-next representation is not bit exact")
        self.ppo_batch.validate()
        count = int(self.ppo_batch.shaped_rewards.size)
        self.integrity_diagnostics.validate(count)
        diagnostics = self.integrity_diagnostics
        clipped_expected = np.clip(
            np.asarray(diagnostics.v10_env_reward_raw, dtype=np.float32),
            np.float32(-reward_config.environment_reward_clip_abs),
            np.float32(reward_config.environment_reward_clip_abs),
        ).astype(np.float32, copy=False)
        if not _same_f32_bits(diagnostics.clipped_environment_reward, clipped_expected):
            raise ValueError("V6 clipped environment reward is not bit exact")
        safety_components = np.stack(
            (
                diagnostics.normalized_safety_only_block_cost,
                diagnostics.normalized_unauthorized_contact_cost,
                diagnostics.normalized_full_safety_desk_cost,
            ),
            axis=1,
        ).astype(np.float32, copy=False)
        normalized_expected = np.max(safety_components, axis=1).astype(np.float32, copy=False)
        if not _same_f32_bits(diagnostics.normalized_safety_cost, normalized_expected):
            raise ValueError("V6 normalized safety cost is not bit exact")
        safety_expected = np.multiply(
            np.float32(-reward_config.privileged_safety_penalty_coefficient),
            normalized_expected,
            dtype=np.float32,
        )
        if not _same_f32_bits(diagnostics.privileged_safety_penalty, safety_expected):
            raise ValueError("V6 privileged safety penalty is not bit exact")
        terminal_expected = np.where(
            self.strict_success,
            np.float32(reward_config.terminal_outcome_magnitude),
            np.where(
                self.terminal_failure,
                np.float32(-reward_config.terminal_outcome_magnitude),
                np.float32(0.0),
            ),
        ).astype(np.float32, copy=False)
        if not _same_f32_bits(diagnostics.terminal_outcome_reward, terminal_expected):
            raise ValueError("V6 terminal outcome reward is not bit exact")
        clipped_plus_safety = np.add(clipped_expected, safety_expected, dtype=np.float32)
        reward_before_expected = np.add(clipped_plus_safety, terminal_expected, dtype=np.float32)
        if not _same_f32_bits(self.env_rewards, reward_before_expected):
            raise ValueError("V6 environment reward decomposition is not bit exact")
        arithmetic = canonical_potential_reward_arithmetic_v6_candidate(
            reward_before_potential=reward_before_expected,
            potential_before=self.potential_before,
            potential_next_for_shaping=self.potential_next_for_shaping,
            shaping_gamma=self.shaping_gamma,
        )
        if not _same_f32_bits(self.shaped_rewards, arithmetic.shaped_reward):
            raise ValueError("V6 rollout shaped reward is not bit exact")
        integrity = np.asarray(self.integrity_diagnostics.v6_integrity_success, dtype=bool)
        hard = np.asarray(self.integrity_diagnostics.hard_safety_violation, dtype=bool)
        surface = np.asarray(self.integrity_diagnostics.v10_surface_success, dtype=bool)
        unauthorized = np.asarray(
            self.integrity_diagnostics.unauthorized_contact_part_penetration_count,
            dtype=np.int64,
        )
        if not np.array_equal(self.ppo_batch.strict_success, integrity):
            raise ValueError("V6 PPO success flags differ from integrity success")
        if np.any(
            hard & ~(self.ppo_batch.terminated & self.ppo_batch.terminal_failure & self.ppo_batch.safety_stop)
        ):
            raise ValueError("V6 hard violation must terminate as safety failure")
        if np.any((unauthorized > 0) & ~hard):
            raise ValueError("V6 unauthorized penetration must be a hard violation")
        if np.any(
            integrity
            & ~(self.ppo_batch.terminated & ~self.ppo_batch.terminal_failure & ~self.ppo_batch.safety_stop)
        ):
            raise ValueError("V6 integrity success must be a safe terminal")
        if np.any(surface & ~self.ppo_batch.terminated):
            raise ValueError("V6 V10 surface success must end the credit episode")
        episode_ids = np.asarray(self.ppo_batch.episode_ids, dtype=np.int64)
        expected_episode_count = int(episode_ids[-1]) + 1
        if (
            type(self.episode_randomization_records) is not tuple
            or len(self.episode_randomization_records) != expected_episode_count
        ):
            raise ValueError("V6 episode randomization record count mismatch")
        expected_command_epoch = 0
        for expected_id, record in enumerate(self.episode_randomization_records):
            if type(record) is not ScratchEpisodeRandomizationIdentityV6Candidate:
                raise TypeError("V6 episode randomization record type mismatch")
            record.validate()
            if record.episode_id != expected_id:
                raise ValueError("V6 episode randomization record order mismatch")
            if record.requested_reset_seed != self.rollout_seed + expected_id:
                raise ValueError("V6 episode randomization reset seed mismatch")
            expected_command_epoch += record.reset_attempt_count
            if record.command_epoch != expected_command_epoch:
                raise ValueError("V6 episode randomization command epoch mismatch")
            mask = episode_ids == expected_id
            if not np.any(mask) or np.any(self.obstacle_enabled[mask] != record.obstacle_enabled):
                raise ValueError("V6 episode randomization obstacle evidence mismatch")
            if np.any(self.integrity_diagnostics.stress_enabled[mask] != record.stress_enabled):
                raise ValueError("V6 episode randomization stress evidence mismatch")
        model_byte_counts = {record.compiled_model_bytes for record in self.episode_randomization_records}
        if len(model_byte_counts) != 1:
            raise ValueError("V6 episode randomized model byte count changed")
        for index in np.flatnonzero(hard):
            if index + 1 < episode_ids.size and episode_ids[index + 1] == episode_ids[index]:
                raise ValueError("V6 rollout continued after a hard violation")


def _module_device(module: torch.nn.Module) -> torch.device:
    try:
        return next(module.parameters()).device
    except StopIteration as error:  # pragma: no cover
        raise RuntimeError("V6 PPO module has no parameters") from error


def _torch_generator(device: torch.device, seed: int) -> torch.Generator | None:
    if device.type == "mps":
        torch.manual_seed(seed)
        if hasattr(torch, "mps"):
            torch.mps.manual_seed(seed)
        return None
    generator = torch.Generator(device=device.type)
    generator.manual_seed(seed)
    return generator


def collect_scratch_rollout_v6_candidate(
    env: RealisticEdgeArmEnvV10,
    actor: FullActionScratchActorV1,
    critic: PrivilegedEffectCriticV1,
    *,
    steps: int,
    seed: int,
    obstacle_probability: float = 0.50,
    stress_probability: float = 0.30,
    gamma: float = 0.99,
    potential_reward: ScratchPotentialRewardV6Candidate | None = None,
) -> ScratchRolloutBatchV6Candidate:
    """Collect complete V6 credit episodes from the exact V10 plant."""

    reward = potential_reward or ScratchPotentialRewardV6Candidate()
    ScratchPotentialRewardV6Candidate._require_exact_v10(env)
    if type(reward) is not ScratchPotentialRewardV6Candidate:
        raise TypeError("V6 rollout requires the exact V6 reward strategy")
    if type(steps) is not int or steps < 1 or type(seed) is not int or seed < 0:
        raise ValueError("V6 rollout steps/seed are invalid")
    if env.step_count != 0 or env.command_epoch != 0 or env.estop is not False:
        raise ValueError("V6 rollout requires a fresh environment before first reset")
    for name, probability in (
        ("obstacle_probability", obstacle_probability),
        ("stress_probability", stress_probability),
    ):
        if not np.isfinite(probability) or not 0.0 <= probability <= 1.0:
            raise ValueError(f"{name} must be finite and in [0, 1]")
    gamma32 = _canonical_f32_array("rollout gamma", gamma)
    if gamma32.shape != () or not np.float32(0.0) < gamma32 <= np.float32(1.0):
        raise ValueError("V6 rollout gamma is invalid")
    canonical_gamma = float(gamma32)
    device = _module_device(actor)
    if _module_device(critic) != device:
        raise ValueError("V6 actor and critic must share a device")
    generator = _torch_generator(device, seed)
    episode_rng = np.random.default_rng(seed ^ 0x7A41C9)
    static_execution_contract_sha256 = _canonical_sha256(_episode_static_execution_contract_v6(env))
    names = (
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
        "clipped_env_reward",
        "terminal_outcome_reward",
        "surface_success",
        "hard_violation",
        "safety_only_clearance_violation",
        "unauthorized_contact_part_penetration",
        "full_safety_desk_penetration",
        "physics_substeps",
        "stress_enabled",
        "minimum_94",
        "contact_minimums",
        "minimum_desk",
        "unauthorized_count",
        "maximum_unauthorized_depth",
        "normalized_safety_only",
        "normalized_unauthorized",
        "normalized_desk",
        "normalized_safety",
        "safety_penalty",
    )
    rows: dict[str, list[Any]] = {name: [] for name in names}
    episode_randomization_records: list[ScratchEpisodeRandomizationIdentityV6Candidate] = []
    episode_id = 0

    def reset_episode(current_id: int) -> None:
        obstacle = bool(episode_rng.random() < obstacle_probability)
        stress = bool(episode_rng.random() < stress_probability)
        reset_seed = seed + current_id
        env.reset(seed=reset_seed, obstacle=obstacle, stress=stress)
        if current_id != len(episode_randomization_records):
            raise RuntimeError("V6 episode randomization record sequence diverged")
        episode_randomization_records.append(
            _build_episode_randomization_identity_v6(
                env,
                episode_id=current_id,
                requested_reset_seed=reset_seed,
                obstacle_enabled=obstacle,
                stress_enabled=stress,
                expected_static_execution_contract_sha256=(static_execution_contract_sha256),
            )
        )

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
            gamma=canonical_gamma,
            env_terminated=bool(env_terminated),
            env_truncated=bool(env_truncated),
            v10_surface_success=surface_success,
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
            "integrity_success": transition.v6_integrity_success,
            "terminal_failure": transition.safety_credit_terminal,
            "safety_stop": bool(safety.hard_safety_violation or info.get("safety_stop")),
            "terminal_reason": transition.terminal_reason,
            "episode_ids": episode_id,
            "obstacle": bool(env.obstacle_enabled),
            "raw_env_reward": transition.v10_env_reward_raw,
            "clipped_env_reward": transition.clipped_environment_reward,
            "terminal_outcome_reward": transition.terminal_outcome_reward,
            "surface_success": transition.v10_surface_success,
            "hard_violation": safety.hard_safety_violation,
            "safety_only_clearance_violation": safety.safety_only_clearance_violation,
            "unauthorized_contact_part_penetration": (safety.unauthorized_contact_part_penetration),
            "full_safety_desk_penetration": safety.full_safety_desk_penetration,
            "physics_substeps": safety.physics_substeps,
            "stress_enabled": bool(env.current_stress),
            "minimum_94": safety.minimum_safety_only_block_signed_distance_m,
            "contact_minimums": safety.minimum_contact_part_block_signed_distance_by_role_m,
            "minimum_desk": safety.minimum_full_safety_desk_signed_distance_m,
            "unauthorized_count": safety.unauthorized_contact_part_penetration_count,
            "maximum_unauthorized_depth": (safety.maximum_unauthorized_contact_penetration_depth_m),
            "normalized_safety_only": safety.normalized_safety_only_block_cost,
            "normalized_unauthorized": safety.normalized_unauthorized_contact_cost,
            "normalized_desk": safety.normalized_full_safety_desk_cost,
            "normalized_safety": safety.normalized_safety_cost,
            "safety_penalty": transition.safety_penalty,
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
        shaping_gamma=canonical_gamma,
        potential_reward_config_sha256=reward.config_sha256,
    )
    diagnostics = ScratchRolloutDiagnosticsV6Candidate(
        v10_env_reward_raw=np.asarray(rows["raw_env_reward"], dtype=np.float64),
        clipped_environment_reward=np.asarray(rows["clipped_env_reward"], dtype=np.float32),
        terminal_outcome_reward=np.asarray(rows["terminal_outcome_reward"], dtype=np.float32),
        v10_surface_success=np.asarray(rows["surface_success"], dtype=bool),
        v6_integrity_success=np.asarray(rows["integrity_success"], dtype=bool),
        hard_safety_violation=np.asarray(rows["hard_violation"], dtype=bool),
        safety_only_clearance_violation=np.asarray(rows["safety_only_clearance_violation"], dtype=bool),
        unauthorized_contact_part_penetration=np.asarray(
            rows["unauthorized_contact_part_penetration"], dtype=bool
        ),
        full_safety_desk_penetration=np.asarray(rows["full_safety_desk_penetration"], dtype=bool),
        physics_substeps=np.asarray(rows["physics_substeps"], dtype=np.int64),
        stress_enabled=np.asarray(rows["stress_enabled"], dtype=bool),
        minimum_94_safety_only_block_distance_m=np.asarray(rows["minimum_94"], dtype=np.float64),
        minimum_contact_part_block_distance_by_role_m=np.asarray(rows["contact_minimums"], dtype=np.float64),
        minimum_full_safety_desk_signed_distance_m=np.asarray(rows["minimum_desk"], dtype=np.float64),
        unauthorized_contact_part_penetration_count=np.asarray(rows["unauthorized_count"], dtype=np.int64),
        maximum_unauthorized_contact_penetration_depth_m=np.asarray(
            rows["maximum_unauthorized_depth"], dtype=np.float64
        ),
        normalized_safety_only_block_cost=np.asarray(rows["normalized_safety_only"], dtype=np.float32),
        normalized_unauthorized_contact_cost=np.asarray(rows["normalized_unauthorized"], dtype=np.float32),
        normalized_full_safety_desk_cost=np.asarray(rows["normalized_desk"], dtype=np.float32),
        normalized_safety_cost=np.asarray(rows["normalized_safety"], dtype=np.float32),
        privileged_safety_penalty=np.asarray(rows["safety_penalty"], dtype=np.float32),
    )
    result = ScratchRolloutBatchV6Candidate(
        ppo_batch=ppo_batch,
        integrity_diagnostics=diagnostics,
        episode_randomization_records=tuple(episode_randomization_records),
        rollout_seed=seed,
    )
    result.validate()
    return result


def _capture_local_import_closure_v6() -> dict[str, str]:
    """Capture the complete recursive local Python import closure once."""

    directory = Path(__file__).resolve().parent
    pending = ["__init__.py", Path(__file__).name]
    payloads: dict[str, bytes] = {}

    def local_filename(module: str) -> str | None:
        if not module or any(part in {"", ".", ".."} for part in module.split(".")):
            return None
        relative = Path(*module.split(".")).with_suffix(".py")
        candidate = directory / relative
        if not candidate.is_file():
            return None
        return relative.as_posix()

    while pending:
        filename = pending.pop()
        if filename in payloads:
            continue
        path = directory / filename
        try:
            payload = path.read_bytes()
            tree = ast.parse(payload, filename=str(path))
        except (OSError, SyntaxError) as error:
            raise RuntimeError(f"V6 cannot capture local source dependency {filename}") from error
        payloads[filename] = payload
        discovered: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                if node.level == 1:
                    if node.module:
                        candidate = local_filename(node.module)
                        if candidate is not None:
                            discovered.add(candidate)
                    else:
                        for alias in node.names:
                            candidate = local_filename(alias.name)
                            if candidate is not None:
                                discovered.add(candidate)
                elif node.level == 0 and node.module:
                    prefix = "edgearm."
                    if node.module.startswith(prefix):
                        candidate = local_filename(node.module[len(prefix) :])
                        if candidate is not None:
                            discovered.add(candidate)
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    prefix = "edgearm."
                    if alias.name.startswith(prefix):
                        candidate = local_filename(alias.name[len(prefix) :])
                        if candidate is not None:
                            discovered.add(candidate)
        pending.extend(sorted(discovered, reverse=True))
    return {name: hashlib.sha256(payloads[name]).hexdigest() for name in sorted(payloads)}


def scratch_v6_candidate_source_hashes() -> dict[str, str]:
    """Double-read the recursive local implementation closure fail-closed."""

    first = _capture_local_import_closure_v6()
    second = _capture_local_import_closure_v6()
    if first != second:
        raise RuntimeError("V6 local source closure changed across capture")
    return first


_SOURCE_FILENAMES_V6 = tuple(scratch_v6_candidate_source_hashes())


_FRESH_EXECUTION_EXCLUDED_ATTRIBUTES_V6 = frozenset(
    {
        "model",
        "data",
        "rng",
        "_episode_rng",
        "_v6_rng",
        "seed",
        "_model_scene_path",
        "_model_scene_bundle",
        "_clearance_guard_scratch",
        "_clearance_guard_dynamics_scratch",
    }
)


def _execution_contract_value_v6(value: object) -> object:
    """Encode one fresh-instance value with exact type/order/byte semantics."""

    if value is None:
        return {"type": "none"}
    if type(value) is bool:
        return {"type": "bool", "value": value}
    if type(value) is int:
        return {"type": "int", "value": str(value)}
    if type(value) is float:
        return {
            "type": "float64",
            "bytes_le": np.asarray([value], dtype="<f8").tobytes().hex(),
        }
    if type(value) is str:
        return {"type": "str", "value": value}
    if type(value) is bytes:
        return {
            "type": "bytes",
            "byte_count": len(value),
            "sha256": hashlib.sha256(value).hexdigest(),
        }
    if isinstance(value, np.generic):
        scalar = np.asarray(value)
        return {
            "type": "numpy_scalar",
            "dtype": scalar.dtype.str,
            "bytes": scalar.tobytes().hex(),
        }
    if type(value) is np.ndarray:
        array = np.ascontiguousarray(value)
        return {
            "type": "numpy_ndarray",
            "dtype": array.dtype.str,
            "shape": list(array.shape),
            "byte_count": int(array.nbytes),
            "sha256": hashlib.sha256(array.tobytes()).hexdigest(),
        }
    if is_dataclass(value) and not isinstance(value, type):
        return {
            "type": "dataclass",
            "class": f"{type(value).__module__}.{type(value).__qualname__}",
            "fields": [
                [item.name, _execution_contract_value_v6(getattr(value, item.name))] for item in fields(value)
            ],
        }
    if type(value) is dict:
        entries = [
            [
                _execution_contract_value_v6(key),
                _execution_contract_value_v6(item),
            ]
            for key, item in value.items()
        ]
        entries.sort(key=lambda row: json.dumps(row[0], sort_keys=True, separators=(",", ":")))
        return {"type": "dict", "entries": entries}
    if type(value) in {tuple, list, deque}:
        return {
            "type": type(value).__name__,
            "items": [_execution_contract_value_v6(item) for item in value],
        }
    if type(value) in {set, frozenset}:
        items = [_execution_contract_value_v6(item) for item in value]
        items.sort(key=lambda item: json.dumps(item, sort_keys=True, separators=(",", ":")))
        return {"type": type(value).__name__, "items": items}
    raise TypeError(f"V6 fresh execution contract encountered unsupported value type: {type(value)!r}")


def _episode_static_execution_contract_v6(
    env: RealisticEdgeArmEnvV10,
) -> dict[str, object]:
    """Capture reset-invariant safety geometry and derived execution caches."""

    exact = ScratchPotentialRewardV6Candidate._require_exact_v10(env)
    if type(exact.model) is not mujoco.MjModel:
        raise TypeError("V6 static episode contract requires exact MjModel")
    ids = exact._ids
    if type(ids) is not dict:
        raise TypeError("V6 static episode contract requires exact ID dictionary")
    safety_ids = tuple(int(value) for value in ids.get("tool_safety_geoms", ()))
    contact_ids = tuple(int(value) for value in ids.get("tool_contact_geoms", ()))
    if (
        len(safety_ids) != _EXPECTED_SAFETY_GEOM_COUNT
        or len(set(safety_ids)) != _EXPECTED_SAFETY_GEOM_COUNT
        or len(contact_ids) != 2
        or not set(contact_ids).issubset(safety_ids)
        or any(value < 0 or value >= exact.model.ngeom for value in safety_ids)
    ):
        raise RuntimeError("V6 static episode safety geometry IDs are invalid")
    safety_index = np.asarray(safety_ids, dtype=np.int64)
    model_arrays: list[list[object]] = []
    for name in sorted(dir(exact.model)):
        if not (name.startswith("geom_") or name.startswith("mesh_")):
            continue
        value = getattr(exact.model, name)
        if type(value) is not np.ndarray:
            continue
        selected = value
        if name.startswith("geom_") and value.ndim >= 1 and value.shape[0] == exact.model.ngeom:
            selected = value[safety_index]
        model_arrays.append([name, _execution_contract_value_v6(selected)])
    cache_names = {
        name for name in vars(exact) if name.startswith("_default_") or name.startswith("default_")
    }
    cache_names.update(
        {
            "_ids",
            "_pair_ids",
            "_pusher_block_pair_ids",
            "_block_dof_address",
            "_robot_geoms",
            "joint_ranges",
            "tool_gripper_joint_position_rad",
            "_coverage_points",
            "_desk_corrected_x_bounds",
        }
    )
    missing = cache_names - set(vars(exact))
    if missing:
        raise RuntimeError(
            "V6 static episode contract is missing execution caches: " + ",".join(sorted(missing))
        )
    return {
        "format": EPISODE_STATIC_EXECUTION_CONTRACT_V6_CANDIDATE_FORMAT,
        "safety_geom_ids": list(safety_ids),
        "contact_geom_ids": list(contact_ids),
        "model_arrays": model_arrays,
        "execution_caches": [
            [name, _execution_contract_value_v6(getattr(exact, name))] for name in sorted(cache_names)
        ],
    }


def _queued_command_contract_v6(value: object) -> dict[str, object]:
    if (
        type(value).__module__ != "edgearm.sim2real_env"
        or type(value).__qualname__ != "_QueuedCommand"
        or type(vars(value)) is not dict
    ):
        raise TypeError("V6 post-reset queue contains a foreign command object")
    target = np.asarray(value)
    if (
        target.shape != (ACTION_DIM,)
        or target.dtype != np.dtype(np.float64)
        or not np.all(np.isfinite(target))
    ):
        raise ValueError("V6 post-reset queued target is non-canonical")
    return {
        "class": f"{type(value).__module__}.{type(value).__qualname__}",
        "target": _execution_contract_value_v6(target),
        "attributes": [
            [name, _execution_contract_value_v6(item)] for name, item in sorted(vars(value).items())
        ],
    }


def _post_reset_execution_contract_v6(
    env: RealisticEdgeArmEnvV10,
) -> dict[str, object]:
    """Capture deterministic reset state independently of cumulative epoch IDs."""

    exact = ScratchPotentialRewardV6Candidate._require_exact_v10(env)
    if type(exact.data) is not mujoco.MjData or exact.data.model is not exact.model:
        raise RuntimeError("V6 post-reset data is bound to a foreign model")
    episode_domain = dict(exact.episode_domain)
    episode_sim2real = dict(exact.episode_sim2real)
    data_contract: list[list[object]] = []
    for name in sorted(dir(exact.data)):
        if name.startswith("_"):
            continue
        if name.endswith(("_colind", "_rowadr", "_rownnz", "_rowsuper")):
            # These sparse-layout work buffers are unused/uninitialized under
            # the dense solver used by this frozen runtime.  Their numerical
            # counterparts remain covered below.
            continue
        value = getattr(exact.data, name)
        if type(value) is np.ndarray:
            data_contract.append([name, _execution_contract_value_v6(value)])
    if not data_contract:
        raise RuntimeError("V6 post-reset MjData array contract is empty")
    contact_contract: list[object] = []
    for index in range(exact.data.ncon):
        contact = exact.data.contact[index]
        contact_contract.append(
            {
                "scalars": {
                    name: _execution_contract_value_v6(getattr(contact, name))
                    for name in (
                        "dim",
                        "dist",
                        "efc_address",
                        "exclude",
                        "includemargin",
                        "mu",
                    )
                },
                "arrays": {
                    name: _execution_contract_value_v6(getattr(contact, name))
                    for name in (
                        "H",
                        "elem",
                        "flex",
                        "frame",
                        "friction",
                        "geom",
                        "pos",
                        "solimp",
                        "solref",
                        "solreffriction",
                        "vert",
                    )
                },
            }
        )
    for name in (
        "_clearance_guard_scratch",
        "_clearance_guard_dynamics_scratch",
    ):
        scratch = getattr(exact, name, None)
        if type(scratch) is not mujoco.MjData or scratch.model is not exact.model:
            raise RuntimeError(f"V6 post-reset {name} is bound to a foreign model")
    excluded_attributes = {
        "model",
        "data",
        "_clearance_guard_scratch",
        "_clearance_guard_dynamics_scratch",
        "_model_scene_path",
        "_model_scene_bundle",
        "rng",
        "_episode_rng",
        "_v6_rng",
        "_command_queue",
        "episode_domain",
        "episode_sim2real",
        "_episode_v6",
    }
    missing_excluded = excluded_attributes - set(vars(exact))
    if missing_excluded:
        raise RuntimeError(
            "V6 post-reset contract is missing excluded runtime objects: "
            + ",".join(sorted(missing_excluded))
        )
    rng_contract: list[list[object]] = []
    for name in ("rng", "_episode_rng", "_v6_rng"):
        generator = getattr(exact, name, None)
        if type(generator) is not np.random.Generator or type(generator.bit_generator) is not np.random.PCG64:
            raise TypeError(f"V6 post-reset RNG type mismatch: {name}")
        rng_contract.append([name, _execution_contract_value_v6(generator.bit_generator.state)])
    if type(exact._command_queue) is not deque:
        raise TypeError("V6 post-reset command queue type mismatch")
    return {
        "format": "edgearm-v10-canonical-post-reset-execution-contract-v1",
        "episode_domain": _execution_contract_value_v6(episode_domain),
        "episode_sim2real": _execution_contract_value_v6(episode_sim2real),
        "episode_v6": _execution_contract_value_v6(exact._episode_v6),
        "data_scalars": {
            "time": _execution_contract_value_v6(float(exact.data.time)),
            "ncon": _execution_contract_value_v6(int(exact.data.ncon)),
            "nefc": _execution_contract_value_v6(int(exact.data.nefc)),
        },
        "data_arrays": data_contract,
        "contacts": contact_contract,
        "all_execution_attributes": [
            [name, _execution_contract_value_v6(value)]
            for name, value in sorted(vars(exact).items())
            if name not in excluded_attributes
        ],
        "rng_states": rng_contract,
        "command_queue": [_queued_command_contract_v6(command) for command in exact._command_queue],
    }


def _fresh_execution_contract_v6(env: RealisticEdgeArmEnvV10) -> dict[str, object]:
    attributes = vars(env)
    missing = _FRESH_EXECUTION_EXCLUDED_ATTRIBUTES_V6 - set(attributes)
    if missing:
        raise RuntimeError(
            "V6 fresh environment is missing excluded runtime attributes: " + ",".join(sorted(missing))
        )
    return {
        "format": V10_EXECUTION_CONTRACT_V6_CANDIDATE_FORMAT,
        "attributes": [
            [name, _execution_contract_value_v6(attributes[name])]
            for name in sorted(set(attributes) - _FRESH_EXECUTION_EXCLUDED_ATTRIBUTES_V6)
        ],
    }


def _assert_pristine_runtime_objects_v6(
    env: RealisticEdgeArmEnvV10,
    reference: RealisticEdgeArmEnvV10,
) -> None:
    if type(env.seed) is not int or env.seed < 0:
        raise ValueError("V6 pristine environment seed is invalid")
    if (
        env.step_count != 0
        or env.command_epoch != 0
        or env.estop is not False
        or env.success_streak != 0
        or env._strict_success_streak != 0
        or len(env._command_queue) != 0
        or env._sim2real_ready is not False
        or env._realism_ready is not False
        or env._workspace_recovery_anchor_v10 is not None
        or env.episode_domain != {}
        or env.episode_sim2real != {}
        or env._episode_v6 != {}
    ):
        raise ValueError(
            "V6 environment identity binding requires a fresh environment before its first reset"
        )
    for selected in (env, reference):
        if type(selected.model) is not mujoco.MjModel:
            raise TypeError("V6 pristine environment requires exact MjModel")
        for name in (
            "data",
            "_clearance_guard_scratch",
            "_clearance_guard_dynamics_scratch",
        ):
            selected_data = getattr(selected, name, None)
            if type(selected_data) is not mujoco.MjData or selected_data.model is not selected.model:
                raise RuntimeError(f"V6 pristine environment {name} is not bound to its exact model")
    for name in ("rng", "_episode_rng", "_v6_rng"):
        actual_rng = getattr(env, name, None)
        reference_rng = getattr(reference, name, None)
        if (
            type(actual_rng) is not np.random.Generator
            or type(reference_rng) is not np.random.Generator
            or type(actual_rng.bit_generator) is not np.random.PCG64
            or type(reference_rng.bit_generator) is not np.random.PCG64
            or _execution_contract_value_v6(actual_rng.bit_generator.state)
            != _execution_contract_value_v6(reference_rng.bit_generator.state)
        ):
            raise RuntimeError(f"V6 pristine environment RNG state mismatch: {name}")


@dataclass(frozen=True)
class ScratchV10EnvironmentIdentityV6Candidate:
    format: str
    environment_class: str
    environment_config_class: str
    environment_config: dict[str, Any]
    environment_config_sha256: str
    runtime_source_mode: str
    runtime_main_logical_path: str
    runtime_file_manifest: tuple[tuple[str, int, str], ...]
    runtime_file_manifest_sha256: str
    runtime_bundle_sha256: str
    compiled_model_sha256: str
    compiled_model_bytes: int
    pristine_binding_format: str
    pristine_binding_verified: bool
    execution_contract_format: str
    execution_contract_sha256: str
    reward_arithmetic_format: str
    identity_sha256: str

    def _identity_payload(self) -> dict[str, Any]:
        payload = asdict(self)
        payload.pop("identity_sha256")
        return payload

    def validate(self, *, verify_current_default_runtime: bool = True) -> None:
        exact = {
            "format": V10_ENVIRONMENT_IDENTITY_V6_CANDIDATE_FORMAT,
            "environment_class": "RealisticEdgeArmEnvV10",
            "environment_config_class": "RealisticEnvV10Config",
            "pristine_binding_format": V10_PRISTINE_BINDING_V6_CANDIDATE_FORMAT,
            "pristine_binding_verified": True,
            "execution_contract_format": V10_EXECUTION_CONTRACT_V6_CANDIDATE_FORMAT,
            "reward_arithmetic_format": REWARD_ARITHMETIC_V6_CANDIDATE_FORMAT,
        }
        for name, expected in exact.items():
            actual = getattr(self, name)
            if type(actual) is not type(expected) or actual != expected:
                raise ValueError(f"V6 environment identity {name} mismatch")
        canonical_config = _v5._validated_environment_config_payload_v10(self.environment_config)
        if self.environment_config_sha256 != _canonical_sha256(canonical_config) or not _is_sha256(
            self.environment_config_sha256
        ):
            raise ValueError("V6 environment config hash mismatch")
        if self.runtime_source_mode not in {
            "immutable_production_mjcf_bundle_v1",
            "double_read_default_runtime_snapshot_v1",
        }:
            raise ValueError("V6 environment runtime source mode mismatch")
        if not isinstance(self.runtime_main_logical_path, str) or not (self.runtime_main_logical_path):
            raise ValueError("V6 environment runtime main path is invalid")
        manifest = self.runtime_file_manifest
        if not isinstance(manifest, tuple) or not manifest:
            raise ValueError("V6 environment runtime manifest is empty")
        paths: list[str] = []
        for entry in manifest:
            if (
                type(entry) is not tuple
                or len(entry) != 3
                or not isinstance(entry[0], str)
                or not entry[0]
                or type(entry[1]) is not int
                or entry[1] < 1
                or not _is_sha256(entry[2])
            ):
                raise ValueError("V6 environment runtime manifest entry is invalid")
            paths.append(entry[0])
        if len(set(paths)) != len(paths) or self.runtime_main_logical_path not in paths:
            raise ValueError("V6 environment runtime manifest paths are invalid")
        if self.runtime_file_manifest_sha256 != _canonical_sha256(manifest) or not _is_sha256(
            self.runtime_bundle_sha256
        ):
            raise ValueError("V6 environment runtime manifest hash mismatch")
        for name in ("compiled_model_sha256", "execution_contract_sha256"):
            if not _is_sha256(getattr(self, name)):
                raise ValueError(f"V6 environment hash is malformed: {name}")
        if type(self.compiled_model_bytes) is not int or self.compiled_model_bytes < 1:
            raise ValueError("V6 environment compiled model byte count is invalid")
        if self.identity_sha256 != _canonical_sha256(self._identity_payload()):
            raise ValueError("V6 environment identity hash mismatch")
        if (
            verify_current_default_runtime
            and self.runtime_source_mode == "double_read_default_runtime_snapshot_v1"
        ):
            snapshot = capture_causal_runtime_snapshot_v1(SCENE_PATH.parents[1])
            current_manifest = _v5._runtime_file_manifest_v5(snapshot.files)
            if (
                self.runtime_main_logical_path != CAUSAL_RUNTIME_MAIN_LOGICAL_PATH
                or self.runtime_bundle_sha256 != snapshot.runtime_bundle_sha256
                or self.runtime_file_manifest != current_manifest
            ):
                raise ValueError("V6 default runtime changed since environment binding")


def _build_environment_identity_v6(
    env: RealisticEdgeArmEnvV10,
) -> ScratchV10EnvironmentIdentityV6Candidate:
    exact_env = ScratchPotentialRewardV6Candidate._require_exact_v10(env)
    config_payload = _v5._validated_environment_config_payload_v10(asdict(exact_env.config))
    source_mode, main_logical_path, files = _v5._runtime_files_for_exact_v10(exact_env)
    manifest = _v5._runtime_file_manifest_v5(files)
    reference_bundle = ProductionMjcfBundleV1(
        main_logical_path=main_logical_path,
        files=files,
    )
    reference = RealisticEdgeArmEnvV10(
        RealisticEnvV10Config(**config_payload),
        seed=exact_env.seed,
        model_scene_bundle=reference_bundle,
    )
    ScratchPotentialRewardV6Candidate._require_exact_v10(reference)
    _assert_pristine_runtime_objects_v6(exact_env, reference)
    compiled_sha256, compiled_bytes = _v5._compiled_model_identity_v5(exact_env)
    reference_compiled_sha256, reference_compiled_bytes = _v5._compiled_model_identity_v5(reference)
    if compiled_sha256 != reference_compiled_sha256 or compiled_bytes != reference_compiled_bytes:
        raise ValueError("V6 fresh environment compiled model differs from its captured runtime")
    execution_contract = _fresh_execution_contract_v6(exact_env)
    reference_execution_contract = _fresh_execution_contract_v6(reference)
    if execution_contract != reference_execution_contract:
        raise ValueError("V6 fresh environment derived execution state differs from its captured runtime")
    execution_contract_sha256 = _canonical_sha256(execution_contract)
    base: dict[str, Any] = {
        "format": V10_ENVIRONMENT_IDENTITY_V6_CANDIDATE_FORMAT,
        "environment_class": "RealisticEdgeArmEnvV10",
        "environment_config_class": "RealisticEnvV10Config",
        "environment_config": config_payload,
        "environment_config_sha256": _canonical_sha256(config_payload),
        "runtime_source_mode": source_mode,
        "runtime_main_logical_path": main_logical_path,
        "runtime_file_manifest": manifest,
        "runtime_file_manifest_sha256": _canonical_sha256(manifest),
        "runtime_bundle_sha256": runtime_bundle_sha256_v1(files),
        "compiled_model_sha256": compiled_sha256,
        "compiled_model_bytes": compiled_bytes,
        "pristine_binding_format": V10_PRISTINE_BINDING_V6_CANDIDATE_FORMAT,
        "pristine_binding_verified": True,
        "execution_contract_format": V10_EXECUTION_CONTRACT_V6_CANDIDATE_FORMAT,
        "execution_contract_sha256": execution_contract_sha256,
        "reward_arithmetic_format": REWARD_ARITHMETIC_V6_CANDIDATE_FORMAT,
    }
    identity = ScratchV10EnvironmentIdentityV6Candidate(
        **base,
        identity_sha256=_canonical_sha256(base),
    )
    identity.validate()
    return identity


@dataclass(frozen=True)
class ScratchPPOV6CandidateProvenance:
    source_type: str
    checkpoint_format: str
    environment_class: str
    environment_config_class: str
    environment_source_sha256: str
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
    reward_arithmetic_format: str
    trainer_source_hashes: dict[str, str]
    physical_samples: int
    physical_trials: int
    production_admission: bool
    genesis_sha256: str

    def _genesis_payload(self) -> dict[str, Any]:
        payload = asdict(self)
        payload.pop("genesis_sha256")
        return {
            "genesis_version": TRAINING_GENESIS_V6_CANDIDATE_VERSION,
            "actor_architecture": ACTOR_ARCHITECTURE,
            "critic_architecture": CRITIC_ARCHITECTURE,
            "policy_parameterization": POLICY_PARAMETERIZATION,
            **payload,
        }

    def validate(self, *, verify_current_sources: bool = True) -> None:
        exact = {
            "source_type": SOURCE_TYPE,
            "checkpoint_format": CHECKPOINT_FORMAT_V6_CANDIDATE,
            "environment_class": "RealisticEdgeArmEnvV10",
            "environment_config_class": "RealisticEnvV10Config",
            "random_initialization": True,
            "full_six_joint_action": True,
            "expert_action_inputs": 0,
            "controller_phase_inputs": 0,
            "behavior_cloning_steps": 0,
            "policy_observation_changed": False,
            "contact_telemetry_is_reward_input": True,
            "contact_telemetry_privilege": REWARD_INPUT_DISCLOSURE_V6_CANDIDATE[
                "contact_telemetry_privilege"
            ],
            "privileged_state_schema_sha256": PRIVILEGED_EFFECT_STATE_SCHEMA_SHA256,
            "potential_reward_version": POTENTIAL_REWARD_V6_CANDIDATE_VERSION,
            "reward_arithmetic_format": REWARD_ARITHMETIC_V6_CANDIDATE_FORMAT,
            "physical_samples": 0,
            "physical_trials": 0,
            "production_admission": False,
        }
        for name, expected in exact.items():
            actual = getattr(self, name)
            if type(actual) is not type(expected) or actual != expected:
                raise ValueError(f"V6 provenance {name} mismatch")
        if type(self.initialization_seed) is not int or self.initialization_seed < 0:
            raise ValueError("V6 provenance initialization seed is invalid")
        for name in (
            "environment_source_sha256",
            "actor_initial_state_sha256",
            "critic_initial_state_sha256",
            "privileged_state_schema_sha256",
            "potential_reward_config_sha256",
            "genesis_sha256",
        ):
            if not _is_sha256(getattr(self, name)):
                raise ValueError(f"V6 provenance hash is malformed: {name}")
        if (
            type(self.trainer_source_hashes) is not dict
            or (set(self.trainer_source_hashes) != set(_SOURCE_FILENAMES_V6))
            or any(not _is_sha256(value) for value in self.trainer_source_hashes.values())
        ):
            raise ValueError("V6 trainer source hashes are malformed")
        if self.environment_source_sha256 != self.trainer_source_hashes["sim2real_env_v10.py"]:
            raise ValueError("V6 provenance is not bound to its V10 source hash")
        if verify_current_sources and self.trainer_source_hashes != (scratch_v6_candidate_source_hashes()):
            raise ValueError("V6 trainer sources changed since genesis")
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(self.initialization_seed)
            actor = FullActionScratchActorV1()
            critic = PrivilegedEffectCriticV1()
        if self.actor_initial_state_sha256 != state_dict_sha256_v1(actor.state_dict()):
            raise ValueError("V6 actor genesis is not canonical")
        if self.critic_initial_state_sha256 != state_dict_sha256_v1(critic.state_dict()):
            raise ValueError("V6 critic genesis is not canonical")
        expected_reward_sha = ScratchPotentialRewardV6CandidateConfig().sha256()
        if self.potential_reward_config_sha256 != expected_reward_sha:
            raise ValueError("V6 reward config provenance mismatch")
        if self.genesis_sha256 != _canonical_sha256(self._genesis_payload()):
            raise ValueError("V6 genesis hash mismatch")


@dataclass
class ScratchPPOV6CandidateBundle:
    actor: FullActionScratchActorV1
    critic: PrivilegedEffectCriticV1
    provenance: ScratchPPOV6CandidateProvenance
    potential_reward_config: ScratchPotentialRewardV6CandidateConfig
    environment_identity: ScratchV10EnvironmentIdentityV6Candidate | None = None


def bind_scratch_ppo_v6_environment_identity(
    bundle: ScratchPPOV6CandidateBundle,
    env: RealisticEdgeArmEnvV10,
) -> ScratchV10EnvironmentIdentityV6Candidate:
    """Bind one V6 bundle to one exact V10 config/runtime/model identity."""

    if type(bundle) is not ScratchPPOV6CandidateBundle:
        raise TypeError("V6 environment binding requires exact V6 bundle")
    existing = bundle.environment_identity
    if existing is not None:
        existing.validate()
    # Always serialize the compiled model again.  Reusing the first digest for
    # the same Python object would miss later static model mutation.
    candidate = _build_environment_identity_v6(env)
    if existing is not None and candidate != existing:
        raise ValueError("V6 bundle environment identity changed across updates")
    bundle.environment_identity = candidate
    return candidate


def initialize_scratch_ppo_v6_candidate(
    seed: int,
    *,
    device: str | torch.device = "cpu",
) -> ScratchPPOV6CandidateBundle:
    if type(seed) is not int or seed < 0:
        raise ValueError("V6 seed must be a non-negative integer")
    reward_config = ScratchPotentialRewardV6CandidateConfig()
    reward_config.validate()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        actor = FullActionScratchActorV1()
        critic = PrivilegedEffectCriticV1()
    source_hashes = scratch_v6_candidate_source_hashes()
    base: dict[str, Any] = {
        "source_type": SOURCE_TYPE,
        "checkpoint_format": CHECKPOINT_FORMAT_V6_CANDIDATE,
        "environment_class": "RealisticEdgeArmEnvV10",
        "environment_config_class": "RealisticEnvV10Config",
        "environment_source_sha256": source_hashes["sim2real_env_v10.py"],
        "initialization_seed": seed,
        "random_initialization": True,
        "full_six_joint_action": True,
        "expert_action_inputs": 0,
        "controller_phase_inputs": 0,
        "behavior_cloning_steps": 0,
        "policy_observation_changed": False,
        "contact_telemetry_is_reward_input": True,
        "contact_telemetry_privilege": REWARD_INPUT_DISCLOSURE_V6_CANDIDATE["contact_telemetry_privilege"],
        "actor_initial_state_sha256": state_dict_sha256_v1(actor.state_dict()),
        "critic_initial_state_sha256": state_dict_sha256_v1(critic.state_dict()),
        "privileged_state_schema_sha256": PRIVILEGED_EFFECT_STATE_SCHEMA_SHA256,
        "potential_reward_version": POTENTIAL_REWARD_V6_CANDIDATE_VERSION,
        "potential_reward_config_sha256": reward_config.sha256(),
        "reward_arithmetic_format": REWARD_ARITHMETIC_V6_CANDIDATE_FORMAT,
        "trainer_source_hashes": source_hashes,
        "physical_samples": 0,
        "physical_trials": 0,
        "production_admission": False,
    }
    provenance = ScratchPPOV6CandidateProvenance(
        **base,
        genesis_sha256=_canonical_sha256(
            {
                "genesis_version": TRAINING_GENESIS_V6_CANDIDATE_VERSION,
                "actor_architecture": ACTOR_ARCHITECTURE,
                "critic_architecture": CRITIC_ARCHITECTURE,
                "policy_parameterization": POLICY_PARAMETERIZATION,
                **base,
            }
        ),
    )
    provenance.validate()
    return ScratchPPOV6CandidateBundle(
        actor=actor.to(device),
        critic=critic.to(device),
        provenance=provenance,
        potential_reward_config=reward_config,
    )


def build_scratch_checkpoint_payload_v6_candidate(
    bundle: ScratchPPOV6CandidateBundle,
    config: ScratchPPOConfigV1,
    *,
    updates_completed: int,
    env: RealisticEdgeArmEnvV10 | None = None,
) -> dict[str, Any]:
    if type(bundle) is not ScratchPPOV6CandidateBundle:
        raise TypeError("V6 checkpoint requires exact V6 bundle")
    canonical_config = canonicalize_scratch_ppo_config_v6_candidate(config)
    bundle.provenance.validate()
    bundle.potential_reward_config.validate()
    if type(updates_completed) is not int or updates_completed < 0:
        raise ValueError("V6 updates_completed must be non-negative")
    if env is not None:
        bind_scratch_ppo_v6_environment_identity(bundle, env)
    environment_identity = bundle.environment_identity
    if environment_identity is None:
        raise ValueError("V6 checkpoint environment identity is unbound")
    environment_identity.validate()
    actor_state = dict(bundle.actor.state_dict())
    critic_state = dict(bundle.critic.state_dict())
    actor_state_sha256 = state_dict_sha256_v1(actor_state)
    critic_state_sha256 = state_dict_sha256_v1(critic_state)
    if updates_completed == 0 and (
        actor_state_sha256 != bundle.provenance.actor_initial_state_sha256
        or critic_state_sha256 != bundle.provenance.critic_initial_state_sha256
    ):
        raise ValueError("V6 zero-update checkpoint differs from canonical genesis")
    return {
        "format": CHECKPOINT_FORMAT_V6_CANDIDATE,
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "source_type": SOURCE_TYPE,
        "environment_class": "RealisticEdgeArmEnvV10",
        "environment_config_class": "RealisticEnvV10Config",
        "environment_source_sha256": bundle.provenance.environment_source_sha256,
        "environment_identity": asdict(environment_identity),
        "environment_identity_sha256": environment_identity.identity_sha256,
        "actor_architecture": ACTOR_ARCHITECTURE,
        "critic_architecture": CRITIC_ARCHITECTURE,
        "policy_parameterization": POLICY_PARAMETERIZATION,
        "state_dim": PRIVILEGED_EFFECT_STATE_DIM,
        "action_dim": ACTION_DIM,
        "state_layout": list(PRIVILEGED_EFFECT_STATE_LAYOUT_V1),
        "state_schema_sha256": PRIVILEGED_EFFECT_STATE_SCHEMA_SHA256,
        "potential_reward_version": POTENTIAL_REWARD_V6_CANDIDATE_VERSION,
        "potential_reward_formula": POTENTIAL_REWARD_V6_CANDIDATE_FORMULA,
        "potential_reward_config": asdict(bundle.potential_reward_config),
        "potential_reward_config_sha256": bundle.potential_reward_config.sha256(),
        "reward_arithmetic_format": REWARD_ARITHMETIC_V6_CANDIDATE_FORMAT,
        "reward_input_disclosure": dict(REWARD_INPUT_DISCLOSURE_V6_CANDIDATE),
        "actor_state": actor_state,
        "critic_state": critic_state,
        "actor_state_sha256": actor_state_sha256,
        "critic_state_sha256": critic_state_sha256,
        "config": asdict(canonical_config),
        "updates_completed": updates_completed,
        "provenance": asdict(bundle.provenance),
        "stock_follower_unmodified": True,
        "added_contact_tool": False,
        "production_admission": False,
        "physical_samples": 0,
        "physical_trials": 0,
        "admission_status": "v6_v10_f32_candidate_not_admitted_for_causal_collection",
    }


def _validate_model_state_v6(
    state: object,
    declared_hash: object,
    canonical_module: torch.nn.Module,
    *,
    name: str,
) -> None:
    if type(state) is not dict:
        raise ValueError(f"V6 checkpoint {name} state is missing")
    if state_dict_sha256_v1(state) != declared_hash:
        raise ValueError(f"V6 checkpoint {name} hash mismatch")
    if any(
        not isinstance(value, torch.Tensor) or not bool(torch.all(torch.isfinite(value)).item())
        for value in state.values()
    ):
        raise ValueError(f"V6 checkpoint {name} state is non-finite")
    canonical = canonical_module.state_dict()
    if set(state) != set(canonical) or any(
        state[key].shape != canonical[key].shape or state[key].dtype != canonical[key].dtype
        for key in canonical
    ):
        raise ValueError(f"V6 checkpoint {name} tensor schema mismatch")
    try:
        canonical_module.load_state_dict(state, strict=True)
    except RuntimeError as error:
        raise ValueError(f"V6 checkpoint {name} state schema mismatch") from error


def _environment_identity_from_payload_v6(
    payload: object,
    *,
    verify_current_default_runtime: bool,
) -> ScratchV10EnvironmentIdentityV6Candidate:
    expected = {item.name for item in fields(ScratchV10EnvironmentIdentityV6Candidate)}
    if type(payload) is not dict or set(payload) != expected:
        raise ValueError("V6 checkpoint environment identity schema mismatch")
    try:
        identity = ScratchV10EnvironmentIdentityV6Candidate(**payload)
        identity.validate(
            verify_current_default_runtime=verify_current_default_runtime,
        )
    except (TypeError, ValueError) as error:
        raise ValueError("V6 checkpoint environment identity is invalid") from error
    if asdict(identity) != payload:
        raise ValueError("V6 checkpoint environment identity is non-canonical")
    return identity


def validate_scratch_checkpoint_payload_v6_candidate(
    payload: object,
    *,
    verify_current_sources: bool = True,
) -> ScratchPPOV6CandidateProvenance:
    """Validate the exact V6/V10/float32 checkpoint schema and hashes."""

    expected_fields = {
        "format",
        "schema_version",
        "source_type",
        "environment_class",
        "environment_config_class",
        "environment_source_sha256",
        "environment_identity",
        "environment_identity_sha256",
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
        "reward_arithmetic_format",
        "reward_input_disclosure",
        "actor_state",
        "critic_state",
        "actor_state_sha256",
        "critic_state_sha256",
        "config",
        "updates_completed",
        "provenance",
        "stock_follower_unmodified",
        "added_contact_tool",
        "production_admission",
        "physical_samples",
        "physical_trials",
        "admission_status",
    }
    if type(payload) is not dict or set(payload) != expected_fields:
        raise ValueError("V6 checkpoint fields are incomplete or unexpected")
    exact = {
        "format": CHECKPOINT_FORMAT_V6_CANDIDATE,
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "source_type": SOURCE_TYPE,
        "environment_class": "RealisticEdgeArmEnvV10",
        "environment_config_class": "RealisticEnvV10Config",
        "actor_architecture": ACTOR_ARCHITECTURE,
        "critic_architecture": CRITIC_ARCHITECTURE,
        "policy_parameterization": POLICY_PARAMETERIZATION,
        "state_dim": PRIVILEGED_EFFECT_STATE_DIM,
        "action_dim": ACTION_DIM,
        "state_layout": list(PRIVILEGED_EFFECT_STATE_LAYOUT_V1),
        "state_schema_sha256": PRIVILEGED_EFFECT_STATE_SCHEMA_SHA256,
        "potential_reward_version": POTENTIAL_REWARD_V6_CANDIDATE_VERSION,
        "potential_reward_formula": POTENTIAL_REWARD_V6_CANDIDATE_FORMULA,
        "reward_arithmetic_format": REWARD_ARITHMETIC_V6_CANDIDATE_FORMAT,
        "reward_input_disclosure": REWARD_INPUT_DISCLOSURE_V6_CANDIDATE,
        "stock_follower_unmodified": True,
        "added_contact_tool": False,
        "production_admission": False,
        "physical_samples": 0,
        "physical_trials": 0,
        "admission_status": "v6_v10_f32_candidate_not_admitted_for_causal_collection",
    }
    for name, expected in exact.items():
        actual = payload.get(name)
        if type(actual) is not type(expected) or actual != expected:
            raise ValueError(f"V6 checkpoint {name} mismatch")
    if not _is_sha256(payload.get("environment_source_sha256")):
        raise ValueError("V6 checkpoint V10 source hash is malformed")
    identity = _environment_identity_from_payload_v6(
        payload.get("environment_identity"),
        verify_current_default_runtime=verify_current_sources,
    )
    if payload.get("environment_identity_sha256") != identity.identity_sha256:
        raise ValueError("V6 checkpoint environment identity hash mismatch")
    reward_payload = payload.get("potential_reward_config")
    if type(reward_payload) is not dict or set(reward_payload) != {
        item.name for item in fields(ScratchPotentialRewardV6CandidateConfig)
    }:
        raise ValueError("V6 checkpoint reward config is malformed")
    try:
        reward_config = ScratchPotentialRewardV6CandidateConfig(**reward_payload)
        reward_config.validate()
    except (TypeError, ValueError) as error:
        raise ValueError("V6 checkpoint reward config is invalid") from error
    if (
        asdict(reward_config) != reward_payload
        or payload.get("potential_reward_config_sha256") != reward_config.sha256()
    ):
        raise ValueError("V6 checkpoint reward config hash mismatch")
    with torch.random.fork_rng(devices=[]):
        canonical_actor = FullActionScratchActorV1()
        canonical_critic = PrivilegedEffectCriticV1()
    _validate_model_state_v6(
        payload.get("actor_state"),
        payload.get("actor_state_sha256"),
        canonical_actor,
        name="actor",
    )
    _validate_model_state_v6(
        payload.get("critic_state"),
        payload.get("critic_state_sha256"),
        canonical_critic,
        name="critic",
    )
    config_payload = payload.get("config")
    if type(config_payload) is not dict or set(config_payload) != {
        item.name for item in fields(ScratchPPOConfigV1)
    }:
        raise ValueError("V6 checkpoint PPO config is malformed")
    try:
        config = ScratchPPOConfigV1(**config_payload)
        canonical_config = canonicalize_scratch_ppo_config_v6_candidate(config)
    except (TypeError, ValueError) as error:
        raise ValueError("V6 checkpoint PPO config is invalid") from error
    if asdict(config) != config_payload or any(
        type(getattr(canonical_config, item.name)) is not type(getattr(config, item.name))
        or getattr(canonical_config, item.name) != getattr(config, item.name)
        for item in fields(config)
    ):
        raise ValueError("V6 checkpoint PPO config is non-canonical")
    if type(payload.get("updates_completed")) is not int or payload["updates_completed"] < 0:
        raise ValueError("V6 checkpoint update count is invalid")
    provenance_payload = payload.get("provenance")
    if type(provenance_payload) is not dict or set(provenance_payload) != {
        item.name for item in fields(ScratchPPOV6CandidateProvenance)
    }:
        raise ValueError("V6 checkpoint provenance is malformed")
    try:
        provenance = ScratchPPOV6CandidateProvenance(**provenance_payload)
        provenance.validate(verify_current_sources=verify_current_sources)
    except (TypeError, ValueError) as error:
        raise ValueError("V6 checkpoint provenance is invalid") from error
    if provenance.potential_reward_config_sha256 != reward_config.sha256():
        raise ValueError("V6 checkpoint provenance/reward mismatch")
    if provenance.reward_arithmetic_format != payload["reward_arithmetic_format"]:
        raise ValueError("V6 checkpoint provenance/arithmetic mismatch")
    if (
        provenance.environment_source_sha256 != payload["environment_source_sha256"]
        or provenance.environment_source_sha256 != provenance.trainer_source_hashes["sim2real_env_v10.py"]
    ):
        raise ValueError("V6 checkpoint V10 source binding mismatch")
    if payload["updates_completed"] == 0 and (
        payload["actor_state_sha256"] != provenance.actor_initial_state_sha256
        or payload["critic_state_sha256"] != provenance.critic_initial_state_sha256
    ):
        raise ValueError("V6 zero-update checkpoint does not match canonical genesis")
    return provenance


def _validate_adam_optimizer_v6_candidate(
    optimizer: torch.optim.Optimizer,
    bundle: ScratchPPOV6CandidateBundle,
    config: ScratchPPOConfigV1,
    *,
    expected_optimizer_steps: int,
) -> None:
    if type(expected_optimizer_steps) is not int or expected_optimizer_steps < 0:
        raise ValueError("V6 expected Adam step count is invalid")
    if type(optimizer) is not torch.optim.Adam:
        raise TypeError("V6 update requires exact torch.optim.Adam")
    if type(bundle.actor) is not FullActionScratchActorV1:
        raise TypeError("V6 update requires exact FullActionScratchActorV1")
    if type(bundle.critic) is not PrivilegedEffectCriticV1:
        raise TypeError("V6 update requires exact PrivilegedEffectCriticV1")
    expected_parameters = [*bundle.actor.parameters(), *bundle.critic.parameters()]
    if len({id(parameter) for parameter in expected_parameters}) != len(expected_parameters):
        raise RuntimeError("V6 actor/critic parameter identity is not unique")
    if len(optimizer.param_groups) != 1:
        raise ValueError("V6 Adam requires exactly one parameter group")
    group = optimizer.param_groups[0]
    actual_parameters = group.get("params")
    if (
        not isinstance(actual_parameters, list)
        or len(actual_parameters) != len(expected_parameters)
        or any(
            actual is not expected
            for actual, expected in zip(
                actual_parameters,
                expected_parameters,
                strict=True,
            )
        )
    ):
        raise ValueError("V6 Adam parameters must be exact ordered actor+critic parameters")
    canonical = torch.optim.Adam(expected_parameters, lr=config.learning_rate)
    canonical_group = canonical.param_groups[0]
    if set(group) != set(canonical_group):
        raise ValueError("V6 Adam parameter-group schema mismatch")
    for name, expected in canonical_group.items():
        if name != "params" and group[name] != expected:
            raise ValueError(f"V6 Adam hyperparameter mismatch: {name}")
    state_parameters = set(optimizer.state)
    expected_parameter_set = set(expected_parameters)
    if not state_parameters:
        if expected_optimizer_steps != 0:
            raise ValueError("V6 Adam state is empty after optimizer steps")
        if (
            state_dict_sha256_v1(bundle.actor.state_dict()) != bundle.provenance.actor_initial_state_sha256
            or state_dict_sha256_v1(bundle.critic.state_dict())
            != bundle.provenance.critic_initial_state_sha256
        ):
            raise ValueError("V6 empty Adam state is not bound to canonical genesis")
    if state_parameters and state_parameters != expected_parameter_set:
        raise ValueError("V6 Adam state does not cover exact model parameters")
    for parameter, parameter_state in optimizer.state.items():
        if parameter not in expected_parameter_set or not isinstance(parameter_state, dict):
            raise ValueError("V6 Adam state contains a foreign parameter")
        expected_state_fields = {"step", "exp_avg", "exp_avg_sq"}
        if bool(group["amsgrad"]):
            expected_state_fields.add("max_exp_avg_sq")
        if set(parameter_state) != expected_state_fields:
            raise ValueError("V6 Adam per-parameter state schema mismatch")
        step = parameter_state["step"]
        if (
            type(step) is not torch.Tensor
            or step.numel() != 1
            or step.ndim != 0
            or step.dtype is not torch.float32
            or not bool(torch.all(torch.isfinite(step)).item())
            or float(step.item()) != float(expected_optimizer_steps)
        ):
            raise ValueError("V6 Adam step state is invalid")
        for name in expected_state_fields - {"step"}:
            value = parameter_state[name]
            if (
                not isinstance(value, torch.Tensor)
                or value.shape != parameter.shape
                or value.dtype != parameter.dtype
                or value.device != parameter.device
                or not bool(torch.all(torch.isfinite(value)).item())
                or (name in {"exp_avg_sq", "max_exp_avg_sq"} and torch.any(value < 0))
            ):
                raise ValueError(f"V6 Adam tensor state is invalid: {name}")


def train_one_scratch_update_v6_candidate(
    env: RealisticEdgeArmEnvV10,
    bundle: ScratchPPOV6CandidateBundle,
    config: ScratchPPOConfigV1,
    *,
    optimizer: torch.optim.Optimizer | None = None,
    expected_optimizer_steps_before: int = 0,
) -> tuple[
    ScratchRolloutBatchV6Candidate,
    PPOUpdateMetricsV1,
    torch.optim.Optimizer,
]:
    """Run unchanged PPO V1 using only exact-V10, canonical-f32 V6 data."""

    ScratchPotentialRewardV6Candidate._require_exact_v10(env)
    if type(bundle) is not ScratchPPOV6CandidateBundle:
        raise TypeError("V6 update requires exact V6 bundle")
    canonical_config = canonicalize_scratch_ppo_config_v6_candidate(config)
    if type(env.seed) is not int or env.seed != canonical_config.seed:
        raise ValueError("V6 update environment seed differs from PPO rollout seed")
    bundle.provenance.validate()
    bundle.potential_reward_config.validate()
    if optimizer is None:
        optimizer = torch.optim.Adam(
            [*bundle.actor.parameters(), *bundle.critic.parameters()],
            lr=canonical_config.learning_rate,
        )
    _validate_adam_optimizer_v6_candidate(
        optimizer,
        bundle,
        canonical_config,
        expected_optimizer_steps=expected_optimizer_steps_before,
    )
    bind_scratch_ppo_v6_environment_identity(bundle, env)
    reward = ScratchPotentialRewardV6Candidate(bundle.potential_reward_config)
    rollout = collect_scratch_rollout_v6_candidate(
        env,
        bundle.actor,
        bundle.critic,
        steps=canonical_config.rollout_steps,
        seed=canonical_config.seed,
        obstacle_probability=canonical_config.obstacle_probability,
        stress_probability=canonical_config.stress_probability,
        gamma=canonical_config.gamma,
        potential_reward=reward,
    )
    metrics, returned_optimizer = ppo_update_v1(
        bundle.actor,
        bundle.critic,
        rollout,  # type: ignore[arg-type]
        canonical_config,
        optimizer=optimizer,
    )
    if returned_optimizer is not optimizer:
        raise RuntimeError("V6 PPO update replaced the bound Adam optimizer")
    if (
        type(metrics) is not PPOUpdateMetricsV1
        or type(metrics.optimizer_steps) is not int
        or metrics.optimizer_steps < 0
    ):
        raise TypeError("V6 PPO update returned invalid optimizer-step metrics")
    _validate_adam_optimizer_v6_candidate(
        returned_optimizer,
        bundle,
        canonical_config,
        expected_optimizer_steps=(expected_optimizer_steps_before + metrics.optimizer_steps),
    )
    return rollout, metrics, returned_optimizer


__all__ = [
    "CHECKPOINT_FORMAT_V6_CANDIDATE",
    "POTENTIAL_REWARD_V6_CANDIDATE_FORMULA",
    "POTENTIAL_REWARD_V6_CANDIDATE_VERSION",
    "REWARD_ARITHMETIC_V6_CANDIDATE_FORMAT",
    "REWARD_INPUT_DISCLOSURE_V6_CANDIDATE",
    "ROLLOUT_DIAGNOSTICS_V6_CANDIDATE_FORMAT",
    "V10_ENVIRONMENT_IDENTITY_V6_CANDIDATE_FORMAT",
    "CanonicalPotentialRewardArithmeticV6Candidate",
    "ScratchPotentialEvaluationV6Candidate",
    "ScratchPotentialRewardV6Candidate",
    "ScratchPotentialRewardV6CandidateConfig",
    "ScratchPotentialTransitionV6Candidate",
    "ScratchPPOV6CandidateBundle",
    "ScratchPPOV6CandidateProvenance",
    "ScratchRolloutBatchV6Candidate",
    "ScratchRolloutDiagnosticsV6Candidate",
    "ScratchSafetyEvidenceV6Candidate",
    "ScratchV10EnvironmentIdentityV6Candidate",
    "bind_scratch_ppo_v6_environment_identity",
    "build_scratch_checkpoint_payload_v6_candidate",
    "canonical_potential_reward_arithmetic_v6_candidate",
    "canonicalize_scratch_ppo_config_v6_candidate",
    "collect_scratch_rollout_v6_candidate",
    "initialize_scratch_ppo_v6_candidate",
    "scratch_v6_candidate_source_hashes",
    "train_one_scratch_update_v6_candidate",
    "validate_scratch_checkpoint_payload_v6_candidate",
]
