"""Isolated Reward V5 candidate bound to the exact joint-bounded V10 plant.

V5 preserves the V4 actor, critic, six-dimensional action, PPO algorithm, and
reward coefficients.  It does not reuse V4/V9 artifact identity: the reward
version, config digest, source closure, genesis, diagnostics, and checkpoint
format are all V5-specific and bind the exact V10 environment and config.

The 96-part contact trace remains simulator-privileged scratch-training truth.
It is not appended to the policy observation and cannot grant production
admission.  This module has no expert, controller-phase, or behavior-cloning
input.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field as dataclass_field, fields
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
from .production_env import ProductionMjcfBundleV1
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
from .scratch_ppo_v3_candidate import ScratchPotentialRewardV3CandidateConfig
from .scratch_ppo_v4_candidate import (
    ScratchPotentialRewardV4Candidate,
    ScratchPotentialRewardV4CandidateConfig,
    ScratchSafetyEvidenceV4Candidate,
)
from .sim2real_env_v10 import RealisticEdgeArmEnvV10, RealisticEnvV10Config


POTENTIAL_REWARD_V5_CANDIDATE_VERSION = (
    "edgearm-v10-privileged-integrity-task-progress-reward-v5-candidate"
)
CHECKPOINT_FORMAT_V5_CANDIDATE = (
    "edgearm-realism-v10-full-action-ppo-from-scratch-v5-candidate"
)
TRAINING_GENESIS_V5_CANDIDATE_VERSION = (
    "edgearm-v10-full-action-scratch-ppo-genesis-v5-candidate"
)
POTENTIAL_REWARD_V5_CANDIDATE_FORMULA = (
    "clip(env_reward,-16,16)+gamma*Phi_next-Phi_before"
    "-24*max(safety_costs)+48*integrity_success-48*safety_terminal;"
    "Phi=12*coverage+6*target_progress+10*contact_progress"
)
ROLLOUT_DIAGNOSTICS_V5_CANDIDATE_FORMAT = (
    "edgearm-scratch-ppo-v10-privileged-integrity-diagnostics-v5-candidate"
)
V10_ENVIRONMENT_IDENTITY_V5_CANDIDATE_FORMAT = (
    "edgearm-v10-environment-config-runtime-and-compiled-model-identity-v1"
)
REWARD_INPUT_DISCLOSURE_V5_CANDIDATE = {
    "policy_observation_changed": False,
    "expert_action_inputs": 0,
    "controller_phase_inputs": 0,
    "behavior_cloning_steps": 0,
    "contact_telemetry_is_reward_input": True,
    "contact_telemetry_privilege": (
        "simulator_privileged_scratch_training_only_not_policy_observation"
    ),
    "tool_safety_geometry_is_reward_input": True,
    "deployment_requires_contact_telemetry": False,
    "environment_class": "exact RealisticEdgeArmEnvV10",
    "environment_config_class": "exact RealisticEnvV10Config",
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


@dataclass(frozen=True)
class ScratchPotentialRewardV5CandidateConfig(ScratchPotentialRewardV4CandidateConfig):
    """The exact V4 coefficients under a new V10-bound identity."""

    def validate(self) -> None:
        canonical = type(self)()
        for field in fields(self):
            value = getattr(self, field.name)
            expected = getattr(canonical, field.name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not np.isfinite(value)
            ):
                raise ValueError(f"V5 candidate reward {field.name} must be finite numeric")
            if float(value) != float(expected):
                raise ValueError(f"V5 candidate reward {field.name} is frozen at {expected}")
        if not 0.0 <= self.penetration_tolerance_m < self.safety_only_block_clearance_m:
            raise ValueError("V5 penetration tolerance must be below clearance")
        if self.safety_depth_scale_m <= 0.0:
            raise ValueError("V5 safety depth scale must be positive")
        if not self.terminal_outcome_magnitude > (
            self.environment_reward_clip_abs + self.maximum_task_potential
        ):
            raise ValueError("V5 terminal outcome does not dominate shaping")
        if self.terminal_failure_upper_bound >= 0.0:
            raise ValueError("V5 safety-terminal upper bound must be negative")
        if self.integrity_success_lower_bound <= 0.0:
            raise ValueError("V5 integrity-success lower bound must be positive")

    def sha256(self) -> str:
        self.validate()
        return _canonical_sha256(
            {
                "version": POTENTIAL_REWARD_V5_CANDIDATE_VERSION,
                "formula": POTENTIAL_REWARD_V5_CANDIDATE_FORMULA,
                "environment_class": "exact RealisticEdgeArmEnvV10",
                "environment_config_class": "exact RealisticEnvV10Config",
                "ordered_tip_roles": list(_TIP_ROLES),
                "complete_safety_geom_count": _EXPECTED_SAFETY_GEOM_COUNT,
                "safety_only_geom_count": _EXPECTED_SAFETY_ONLY_GEOM_COUNT,
                "reward_input_disclosure": REWARD_INPUT_DISCLOSURE_V5_CANDIDATE,
                "config": asdict(self),
            }
        )


@dataclass(frozen=True)
class ScratchPotentialEvaluationV5Candidate:
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
class ScratchSafetyEvidenceV5Candidate(ScratchSafetyEvidenceV4Candidate):
    """V5-labelled safety evidence produced only after the exact V10 gate."""


@dataclass(frozen=True)
class ScratchPotentialTransitionV5Candidate:
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
    env_terminated: bool
    env_truncated: bool
    credit_terminated: bool
    credit_truncated: bool
    v10_surface_success: bool
    v5_integrity_success: bool
    safety_credit_terminal: bool
    episode_safety_violation: bool
    terminal_reason: str


class _V10SafetyParsingKernel(ScratchPotentialRewardV4Candidate):
    """Reuse only V4's 96-part mathematical parser behind the V10 gate."""

    @staticmethod
    def _require_exact_v9(env: object) -> RealisticEdgeArmEnvV10:
        return ScratchPotentialRewardV5Candidate._require_exact_v10(env)


class ScratchPotentialRewardV5Candidate:
    """V4-equivalent reward mathematics with exact V10 identity and gating."""

    version = POTENTIAL_REWARD_V5_CANDIDATE_VERSION
    formula = POTENTIAL_REWARD_V5_CANDIDATE_FORMULA
    reward_input_disclosure = REWARD_INPUT_DISCLOSURE_V5_CANDIDATE

    def __init__(
        self,
        config: ScratchPotentialRewardV5CandidateConfig | None = None,
    ) -> None:
        self.config = config or ScratchPotentialRewardV5CandidateConfig()
        if type(self.config) is not ScratchPotentialRewardV5CandidateConfig:
            raise TypeError("V5 reward requires exact ScratchPotentialRewardV5CandidateConfig")
        self.config.validate()
        self.config_sha256 = self.config.sha256()
        self._geometry_config = ScratchPotentialRewardV3CandidateConfig()
        self._geometry_config.validate()
        self._safety_kernel = _V10SafetyParsingKernel(
            ScratchPotentialRewardV4CandidateConfig(**asdict(self.config))
        )

    @staticmethod
    def _require_exact_v10(env: object) -> RealisticEdgeArmEnvV10:
        if type(env) is not RealisticEdgeArmEnvV10:
            raise TypeError("V5 candidate requires exact RealisticEdgeArmEnvV10")
        for name in (
            "joint_bounded_config",
            "stock_distal_tip_config",
            "stock_gripper_config",
            "contact_feasible_config",
            "config",
        ):
            candidate_config = getattr(env, name, None)
            if type(candidate_config) is not RealisticEnvV10Config:
                raise TypeError(
                    f"V5 candidate requires exact RealisticEnvV10Config at {name}"
                )
            if candidate_config is not env.config:
                raise RuntimeError(
                    f"V5 exact V10 config alias diverged from env.config at {name}"
                )
        return env

    @staticmethod
    def _stable_softmin(values: np.ndarray, temperature: float) -> float:
        if values.shape != (2,) or not np.all(np.isfinite(values)):
            raise RuntimeError("V5 role costs must contain two finite values")
        minimum = float(np.min(values))
        shifted = np.exp(-(values - minimum) / temperature)
        return float(minimum - temperature * np.log(float(np.mean(shifted))))

    def evaluate(
        self,
        env: RealisticEdgeArmEnvV10,
    ) -> ScratchPotentialEvaluationV5Candidate:
        exact_env = self._require_exact_v10(env)
        planning = tuple(int(value) for value in exact_env._ids.get("tool_planning_geoms", ()))
        contacts = tuple(int(value) for value in exact_env._ids.get("tool_contact_geoms", ()))
        safety = tuple(int(value) for value in exact_env._ids.get("tool_safety_geoms", ()))
        roles = tuple(str(value) for value in exact_env._ids.get("tool_contact_geom_roles", ()))
        if (
            exact_env._ids.get("tool_planning_geometry_mode")
            != "per_jaw_distal_tip_references"
            or exact_env._ids.get("tool_safety_geometry_mode") != "coacd_convex_union"
            or len(planning) != 2
            or len(contacts) != 2
            or len(safety) != _EXPECTED_SAFETY_GEOM_COUNT
            or not set(contacts).issubset(safety)
            or roles != _TIP_ROLES
        ):
            raise RuntimeError("V5 exact V10 stock-gripper geometry contract changed")

        geometry = self._geometry_config
        block_geom = int(exact_env._ids["block_geom"])
        block_xy = np.asarray(exact_env.block_xy(), dtype=np.float64)
        target_xy = np.asarray(exact_env.target_xy, dtype=np.float64)
        target_delta = target_xy - block_xy
        target_distance = float(np.linalg.norm(target_delta))
        if target_distance > 1.0e-12:
            push_direction = target_delta / target_distance
        else:
            tip_mean = np.mean(
                np.asarray(
                    [exact_env.data.geom_xpos[geom_id, :2] for geom_id in planning],
                    dtype=np.float64,
                ),
                axis=0,
            )
            fallback = block_xy - tip_mean
            norm = float(np.linalg.norm(fallback))
            push_direction = fallback / norm if norm > 1.0e-12 else np.array([1.0, 0.0])
        direction = np.array([push_direction[0], push_direction[1], 0.0], dtype=np.float64)
        block_rotation = np.asarray(exact_env.data.geom_xmat[block_geom]).reshape(3, 3)
        block_support = float(
            np.dot(
                np.abs(block_rotation.T @ direction),
                np.asarray(exact_env.model.geom_size[block_geom], dtype=np.float64),
            )
        )
        xy_errors: list[float] = []
        for geom_id in planning:
            rotation = np.asarray(exact_env.data.geom_xmat[geom_id]).reshape(3, 3)
            support = float(
                np.dot(
                    np.abs(rotation.T @ direction),
                    np.asarray(exact_env.model.geom_size[geom_id], dtype=np.float64),
                )
            )
            desired = block_xy - push_direction * (
                block_support + support + geometry.precontact_gap_m
            )
            tip_xy = np.asarray(exact_env.data.geom_xpos[geom_id, :2], dtype=np.float64)
            xy_errors.append(float(np.linalg.norm(tip_xy - desired)))
        tip_distances = np.asarray(
            [
                exact_env._geom_signed_distance_for_data(
                    geom_id, block_geom, exact_env.data, cutoff_m=1.0
                )
                for geom_id in contacts
            ],
            dtype=np.float64,
        )
        normalized_gaps = np.clip(
            np.maximum(tip_distances - geometry.desired_tip_block_gap_m, 0.0)
            / geometry.tip_block_gap_scale_m,
            0.0,
            1.0,
        )
        normalized_xy = np.clip(
            np.asarray(xy_errors, dtype=np.float64) / geometry.tip_precontact_xy_scale_m,
            0.0,
            1.0,
        )
        role_cost = (
            geometry.tip_block_gap_role_weight * normalized_gaps
            + geometry.tip_precontact_xy_role_weight * normalized_xy
        )
        softmin = self._stable_softmin(role_cost, geometry.role_softmin_temperature)
        contact_progress = float(1.0 - np.clip(softmin, 0.0, 1.0))
        coverage = float(exact_env.block_target_coverage())
        block_progress = float(
            1.0
            - np.clip(target_distance / geometry.block_target_distance_scale_m, 0.0, 1.0)
        )
        coverage_value = self.config.coverage_progress_coefficient * coverage
        block_value = self.config.block_progress_coefficient * block_progress
        contact_value = self.config.contact_progress_coefficient * contact_progress
        potential = float(coverage_value + block_value + contact_value)
        finite = np.asarray(
            [potential, coverage, block_progress, contact_progress, target_distance, *tip_distances, *role_cost],
            dtype=np.float64,
        )
        if not np.all(np.isfinite(finite)):
            raise RuntimeError("V5 task potential produced non-finite geometry")
        if not -1.0e-12 <= potential <= self.config.maximum_task_potential + 1.0e-12:
            raise RuntimeError("V5 task potential escaped its proven bounds")
        return ScratchPotentialEvaluationV5Candidate(
            potential=potential,
            coverage_progress_value=float(coverage_value),
            block_progress_value=float(block_value),
            contact_progress_value=float(contact_value),
            coverage=coverage,
            normalized_block_progress=block_progress,
            contact_approach_progress=contact_progress,
            block_target_distance_m=target_distance,
            tip_block_signed_distance_m=(float(tip_distances[0]), float(tip_distances[1])),
            role_cost=(float(role_cost[0]), float(role_cost[1])),
        )

    def evaluate_transition_safety(
        self,
        env: RealisticEdgeArmEnvV10,
        info: dict[str, Any],
    ) -> ScratchSafetyEvidenceV5Candidate:
        self._require_exact_v10(env)
        try:
            evidence = self._safety_kernel.evaluate_transition_safety(env, info)
        except RuntimeError as error:
            message = str(error).replace("V4", "V5").replace("V9", "V10")
            raise RuntimeError(message) from error
        return ScratchSafetyEvidenceV5Candidate(**asdict(evidence))

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
        safety: ScratchSafetyEvidenceV5Candidate,
        episode_safety_violation_before: bool,
    ) -> ScratchPotentialTransitionV5Candidate:
        if type(safety) is not ScratchSafetyEvidenceV5Candidate:
            raise TypeError("V5 shaping requires exact V5 safety evidence")
        numeric = np.asarray([env_reward, potential_before, potential_after, gamma])
        if not np.all(np.isfinite(numeric)) or not 0.0 < gamma <= 1.0:
            raise ValueError("V5 shaping inputs are invalid")
        maximum = self.config.maximum_task_potential
        if not -1.0e-9 <= potential_before <= maximum + 1.0e-9:
            raise ValueError("V5 potential_before is outside the proven range")
        if not -1.0e-9 <= potential_after <= maximum + 1.0e-9:
            raise ValueError("V5 potential_after is outside the proven range")
        if env_terminated and env_truncated:
            raise ValueError("V5 transition cannot terminate and truncate")
        if v10_surface_success and (not env_terminated or env_terminal_failure):
            raise ValueError("V10 surface success must be a non-failure termination")
        integrity_success = bool(
            v10_surface_success
            and not episode_safety_violation_before
            and not safety.hard_safety_violation
        )
        episode_safety_violation = bool(
            episode_safety_violation_before or safety.hard_safety_violation
        )
        safety_terminal = bool(
            episode_safety_violation
            or env_terminal_failure
            or (env_terminated and not integrity_success)
        )
        credit_terminated = bool(integrity_success or safety_terminal)
        credit_truncated = bool(env_truncated and not credit_terminated)
        potential_next = 0.0 if credit_terminated else float(potential_after)
        potential_shaping = float(gamma * potential_next - potential_before)
        clipped = float(
            np.clip(
                env_reward,
                -self.config.environment_reward_clip_abs,
                self.config.environment_reward_clip_abs,
            )
        )
        if integrity_success:
            terminal_outcome = self.config.terminal_outcome_magnitude
            terminal_reason = "v5_integrity_success"
        elif safety_terminal:
            terminal_outcome = -self.config.terminal_outcome_magnitude
            if safety.hard_safety_violation:
                reason = safety.hard_safety_reason
            elif episode_safety_violation_before:
                reason = "prior_episode_safety_violation"
            elif env_terminal_failure:
                reason = "environment_terminal_failure"
            else:
                reason = "environment_termination_without_integrity_success"
            terminal_reason = f"v5_safety_terminal:{reason}"
        elif credit_truncated:
            terminal_outcome = 0.0
            terminal_reason = "time_limit"
        else:
            terminal_outcome = 0.0
            terminal_reason = "nonterminal"
        reward_before = float(clipped + safety.safety_penalty + terminal_outcome)
        shaped = float(reward_before + potential_shaping)
        return ScratchPotentialTransitionV5Candidate(
            v10_env_reward_raw=float(env_reward),
            clipped_environment_reward=clipped,
            reward_before_potential=reward_before,
            potential_before=float(potential_before),
            potential_after=float(potential_after),
            potential_next_for_shaping=potential_next,
            potential_shaping=potential_shaping,
            safety_penalty=float(safety.safety_penalty),
            terminal_outcome_reward=float(terminal_outcome),
            shaped_reward=shaped,
            env_terminated=bool(env_terminated),
            env_truncated=bool(env_truncated),
            credit_terminated=credit_terminated,
            credit_truncated=credit_truncated,
            v10_surface_success=bool(v10_surface_success),
            v5_integrity_success=integrity_success,
            safety_credit_terminal=safety_terminal,
            episode_safety_violation=episode_safety_violation,
            terminal_reason=terminal_reason,
        )


@dataclass(frozen=True)
class ScratchRolloutDiagnosticsV5Candidate:
    v10_env_reward_raw: np.ndarray
    v10_surface_success: np.ndarray
    v5_integrity_success: np.ndarray
    hard_safety_violation: np.ndarray
    minimum_94_safety_only_block_distance_m: np.ndarray
    minimum_contact_part_block_distance_by_role_m: np.ndarray
    unauthorized_contact_part_penetration_count: np.ndarray
    privileged_safety_penalty: np.ndarray
    format: str = ROLLOUT_DIAGNOSTICS_V5_CANDIDATE_FORMAT
    privileged_reward_input: bool = True

    def validate(self, transition_count: int) -> None:
        if type(transition_count) is not int or transition_count < 1:
            raise ValueError("V5 diagnostics transition count is invalid")
        expected = {
            "v10_env_reward_raw": (transition_count,),
            "v10_surface_success": (transition_count,),
            "v5_integrity_success": (transition_count,),
            "hard_safety_violation": (transition_count,),
            "minimum_94_safety_only_block_distance_m": (transition_count,),
            "minimum_contact_part_block_distance_by_role_m": (transition_count, 2),
            "unauthorized_contact_part_penetration_count": (transition_count,),
            "privileged_safety_penalty": (transition_count,),
        }
        for name, shape in expected.items():
            if np.asarray(getattr(self, name)).shape != shape:
                raise ValueError(f"V5 diagnostics {name} shape mismatch")
        for name in (
            "v10_env_reward_raw",
            "minimum_94_safety_only_block_distance_m",
            "minimum_contact_part_block_distance_by_role_m",
            "privileged_safety_penalty",
        ):
            if not np.all(np.isfinite(np.asarray(getattr(self, name)))):
                raise ValueError(f"V5 diagnostics {name} is non-finite")
        for name in (
            "v10_surface_success",
            "v5_integrity_success",
            "hard_safety_violation",
        ):
            if np.asarray(getattr(self, name)).dtype != np.dtype(bool):
                raise ValueError(f"V5 diagnostics {name} must be boolean")
        counts = np.asarray(self.unauthorized_contact_part_penetration_count)
        if not np.issubdtype(counts.dtype, np.integer) or np.any(counts < 0):
            raise ValueError("V5 unauthorized contact counts are invalid")
        if np.any(self.v5_integrity_success & ~self.v10_surface_success):
            raise ValueError("V5 integrity success must imply V10 surface success")
        if np.any(self.v5_integrity_success & self.hard_safety_violation):
            raise ValueError("V5 integrity success cannot include a hard violation")
        if self.format != ROLLOUT_DIAGNOSTICS_V5_CANDIDATE_FORMAT:
            raise ValueError("V5 diagnostics format mismatch")
        if self.privileged_reward_input is not True:
            raise ValueError("V5 diagnostics must declare reward privilege")

    def metric_fields(self) -> dict[str, object]:
        count = int(np.asarray(self.v10_env_reward_raw).size)
        self.validate(count)
        return {
            "rollout_v10_surface_success_count": int(
                np.count_nonzero(self.v10_surface_success)
            ),
            "rollout_v5_integrity_success_count": int(
                np.count_nonzero(self.v5_integrity_success)
            ),
            "rollout_v5_hard_safety_violation_count": int(
                np.count_nonzero(self.hard_safety_violation)
            ),
            "rollout_v5_minimum_94_safety_only_block_distance_m": float(
                np.min(self.minimum_94_safety_only_block_distance_m)
            ),
            "rollout_v5_unauthorized_contact_part_penetration_count": int(
                np.sum(self.unauthorized_contact_part_penetration_count)
            ),
        }


@dataclass(frozen=True)
class ScratchRolloutBatchV5Candidate:
    ppo_batch: ScratchRolloutBatchV1
    integrity_diagnostics: ScratchRolloutDiagnosticsV5Candidate
    potential_reward_version: str = POTENTIAL_REWARD_V5_CANDIDATE_VERSION

    def __getattr__(self, name: str) -> Any:
        return getattr(self.ppo_batch, name)

    def validate(self) -> None:
        if self.potential_reward_version != POTENTIAL_REWARD_V5_CANDIDATE_VERSION:
            raise ValueError("V5 rollout reward version mismatch")
        self.ppo_batch.validate()
        count = int(self.ppo_batch.shaped_rewards.size)
        self.integrity_diagnostics.validate(count)
        integrity = np.asarray(self.integrity_diagnostics.v5_integrity_success, dtype=bool)
        hard = np.asarray(self.integrity_diagnostics.hard_safety_violation, dtype=bool)
        surface = np.asarray(self.integrity_diagnostics.v10_surface_success, dtype=bool)
        unauthorized = np.asarray(
            self.integrity_diagnostics.unauthorized_contact_part_penetration_count,
            dtype=np.int64,
        )
        if not np.array_equal(self.ppo_batch.strict_success, integrity):
            raise ValueError("V5 PPO success flags differ from integrity success")
        if np.any(
            hard
            & ~(
                self.ppo_batch.terminated
                & self.ppo_batch.terminal_failure
                & self.ppo_batch.safety_stop
            )
        ):
            raise ValueError("V5 hard violation must terminate as safety failure")
        if np.any((unauthorized > 0) & ~hard):
            raise ValueError("V5 unauthorized penetration must be a hard violation")
        if np.any(
            integrity
            & ~(
                self.ppo_batch.terminated
                & ~self.ppo_batch.terminal_failure
                & ~self.ppo_batch.safety_stop
            )
        ):
            raise ValueError("V5 integrity success must be a safe terminal")
        if np.any(surface & ~self.ppo_batch.terminated):
            raise ValueError("V5 V10 surface success must end the credit episode")
        episode_ids = np.asarray(self.ppo_batch.episode_ids, dtype=np.int64)
        for index in np.flatnonzero(hard):
            if index + 1 < episode_ids.size and episode_ids[index + 1] == episode_ids[index]:
                raise ValueError("V5 rollout continued after a hard violation")


def _module_device(module: torch.nn.Module) -> torch.device:
    try:
        return next(module.parameters()).device
    except StopIteration as error:  # pragma: no cover
        raise RuntimeError("V5 PPO module has no parameters") from error


def _torch_generator(device: torch.device, seed: int) -> torch.Generator | None:
    if device.type == "mps":
        torch.manual_seed(seed)
        if hasattr(torch, "mps"):
            torch.mps.manual_seed(seed)
        return None
    generator = torch.Generator(device=device.type)
    generator.manual_seed(seed)
    return generator


def collect_scratch_rollout_v5_candidate(
    env: RealisticEdgeArmEnvV10,
    actor: FullActionScratchActorV1,
    critic: PrivilegedEffectCriticV1,
    *,
    steps: int,
    seed: int,
    obstacle_probability: float = 0.50,
    stress_probability: float = 0.30,
    gamma: float = 0.99,
    potential_reward: ScratchPotentialRewardV5Candidate | None = None,
) -> ScratchRolloutBatchV5Candidate:
    """Collect complete V5 credit episodes from the exact V10 plant."""

    reward = potential_reward or ScratchPotentialRewardV5Candidate()
    ScratchPotentialRewardV5Candidate._require_exact_v10(env)
    if type(reward) is not ScratchPotentialRewardV5Candidate:
        raise TypeError("V5 rollout requires the exact V5 reward strategy")
    if type(steps) is not int or steps < 1 or type(seed) is not int or seed < 0:
        raise ValueError("V5 rollout steps/seed are invalid")
    for name, probability in (
        ("obstacle_probability", obstacle_probability),
        ("stress_probability", stress_probability),
    ):
        if not np.isfinite(probability) or not 0.0 <= probability <= 1.0:
            raise ValueError(f"{name} must be finite and in [0, 1]")
    if not np.isfinite(gamma) or not 0.0 < gamma <= 1.0:
        raise ValueError("V5 rollout gamma is invalid")
    device = _module_device(actor)
    if _module_device(critic) != device:
        raise ValueError("V5 actor and critic must share a device")
    generator = _torch_generator(device, seed)
    episode_rng = np.random.default_rng(seed ^ 0x7A41C9)
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
        "surface_success",
        "hard_violation",
        "minimum_94",
        "contact_minimums",
        "unauthorized_count",
        "safety_penalty",
    )
    rows: dict[str, list[Any]] = {name: [] for name in names}
    episode_id = 0

    def reset_episode(current_id: int) -> None:
        obstacle = bool(episode_rng.random() < obstacle_probability)
        stress = bool(episode_rng.random() < stress_probability)
        env.reset(seed=seed + current_id, obstacle=obstacle, stress=stress)

    reset_episode(episode_id)
    actor.eval()
    critic.eval()
    while len(rows["states"]) < steps or not (
        rows["terminated"][-1] or rows["truncated"][-1]
    ):
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
        env_failure = bool(
            info.get("terminal_failure", env_terminated and not surface_success)
        )
        transition = reward.shape_transition(
            env_reward=env_reward,
            potential_before=before.potential,
            potential_after=after.potential,
            gamma=gamma,
            env_terminated=bool(env_terminated),
            env_truncated=bool(env_truncated),
            v10_surface_success=surface_success,
            env_terminal_failure=env_failure,
            safety=safety,
            episode_safety_violation_before=False,
        )
        next_state = build_privileged_effect_state_v1(env)
        with torch.no_grad():
            next_value = float(
                critic(torch.from_numpy(next_state).to(device).unsqueeze(0)).item()
            )
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
            "integrity_success": transition.v5_integrity_success,
            "terminal_failure": transition.safety_credit_terminal,
            "safety_stop": bool(safety.hard_safety_violation or info.get("safety_stop")),
            "terminal_reason": transition.terminal_reason,
            "episode_ids": episode_id,
            "obstacle": bool(env.obstacle_enabled),
            "raw_env_reward": transition.v10_env_reward_raw,
            "surface_success": transition.v10_surface_success,
            "hard_violation": safety.hard_safety_violation,
            "minimum_94": safety.minimum_safety_only_block_signed_distance_m,
            "contact_minimums": safety.minimum_contact_part_block_signed_distance_by_role_m,
            "unauthorized_count": safety.unauthorized_contact_part_penetration_count,
            "safety_penalty": safety.safety_penalty,
        }
        for name, value_to_append in append_values.items():
            rows[name].append(value_to_append)
        if (
            transition.credit_terminated or transition.credit_truncated
        ) and len(rows["states"]) < steps:
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
    diagnostics = ScratchRolloutDiagnosticsV5Candidate(
        v10_env_reward_raw=np.asarray(rows["raw_env_reward"], dtype=np.float64),
        v10_surface_success=np.asarray(rows["surface_success"], dtype=bool),
        v5_integrity_success=np.asarray(rows["integrity_success"], dtype=bool),
        hard_safety_violation=np.asarray(rows["hard_violation"], dtype=bool),
        minimum_94_safety_only_block_distance_m=np.asarray(
            rows["minimum_94"], dtype=np.float64
        ),
        minimum_contact_part_block_distance_by_role_m=np.asarray(
            rows["contact_minimums"], dtype=np.float64
        ),
        unauthorized_contact_part_penetration_count=np.asarray(
            rows["unauthorized_count"], dtype=np.int64
        ),
        privileged_safety_penalty=np.asarray(rows["safety_penalty"], dtype=np.float64),
    )
    result = ScratchRolloutBatchV5Candidate(
        ppo_batch=ppo_batch,
        integrity_diagnostics=diagnostics,
    )
    result.validate()
    return result


_SOURCE_FILENAMES_V5 = (
    "causal_runtime_snapshot_v1.py",
    "config.py",
    "contact_telemetry_v1.py",
    "multimodal.py",
    "ppo_utils_v1.py",
    "privileged_effect_state_v1.py",
    "production_env.py",
    "scratch_ppo_v1.py",
    "scratch_ppo_v3_candidate.py",
    "scratch_ppo_v4_candidate.py",
    "scratch_ppo_v5_candidate.py",
    "sim2real_env.py",
    "sim2real_env_v6.py",
    "sim2real_env_v7.py",
    "sim2real_env_v8.py",
    "sim2real_env_v9.py",
    "sim2real_env_v10.py",
    "stock_gripper_convex_decomposition_v1.py",
    "trajectory_contract_v1.py",
)


def scratch_v5_candidate_source_hashes() -> dict[str, str]:
    """Hash the V5 policy/reward/V10 environment implementation closure."""

    directory = Path(__file__).resolve().parent
    return {
        name: _sha256_file(directory / name)
        for name in _SOURCE_FILENAMES_V5
    }


def _validated_environment_config_payload_v10(
    payload: object,
) -> dict[str, Any]:
    if not isinstance(payload, dict) or set(payload) != {
        item.name for item in fields(RealisticEnvV10Config)
    }:
        raise ValueError("V5 exact V10 environment config schema mismatch")
    canonical_types = RealisticEnvV10Config()
    for item in fields(canonical_types):
        value = payload[item.name]
        expected = getattr(canonical_types, item.name)
        if isinstance(expected, tuple):
            if type(value) is not tuple or len(value) != len(expected):
                raise ValueError(
                    f"V5 exact V10 environment config type mismatch: {item.name}"
                )
            if any(
                type(actual) is not type(template)
                for actual, template in zip(value, expected, strict=True)
            ):
                raise ValueError(
                    f"V5 exact V10 environment config element type mismatch: {item.name}"
                )
        elif type(value) is not type(expected):
            raise ValueError(
                f"V5 exact V10 environment config type mismatch: {item.name}"
            )
    try:
        config = RealisticEnvV10Config(**payload)
    except (TypeError, ValueError) as error:
        raise ValueError("V5 exact V10 environment config is invalid") from error
    canonical = asdict(config)
    if canonical != payload:
        raise ValueError("V5 exact V10 environment config is non-canonical")
    return canonical


def _runtime_files_for_exact_v10(
    env: RealisticEdgeArmEnvV10,
) -> tuple[str, str, tuple[tuple[str, bytes], ...]]:
    bundle = getattr(env, "_model_scene_bundle", None)
    scene_path = getattr(env, "_model_scene_path", None)
    if bundle is not None:
        if type(bundle) is not ProductionMjcfBundleV1:
            raise TypeError(
                "V5 exact V10 runtime requires exact ProductionMjcfBundleV1"
            )
        return (
            "immutable_production_mjcf_bundle_v1",
            bundle.main_logical_path,
            bundle.files,
        )
    selected = SCENE_PATH if scene_path is None else Path(scene_path)
    try:
        selected_resolved = selected.resolve(strict=True)
        default_resolved = SCENE_PATH.resolve(strict=True)
    except OSError as error:
        raise ValueError("V5 exact V10 runtime scene is missing") from error
    if selected_resolved != default_resolved:
        raise ValueError(
            "V5 exact V10 custom scene paths require an immutable MJCF bundle"
        )
    snapshot = capture_causal_runtime_snapshot_v1(SCENE_PATH.parents[1])
    return (
        "double_read_default_runtime_snapshot_v1",
        CAUSAL_RUNTIME_MAIN_LOGICAL_PATH,
        snapshot.files,
    )


def _runtime_file_manifest_v5(
    files: tuple[tuple[str, bytes], ...],
) -> tuple[tuple[str, int, str], ...]:
    return tuple(
        (logical_path, len(payload), hashlib.sha256(payload).hexdigest())
        for logical_path, payload in files
    )


def _compiled_model_identity_v5(
    env: RealisticEdgeArmEnvV10,
) -> tuple[str, int]:
    byte_count = int(mujoco.mj_sizeModel(env.model))
    if byte_count < 1:
        raise RuntimeError("V5 exact V10 compiled model has no bytes")
    buffer = np.empty(byte_count, dtype=np.uint8)
    mujoco.mj_saveModel(env.model, buffer=buffer)
    return hashlib.sha256(buffer.tobytes()).hexdigest(), byte_count


@dataclass(frozen=True)
class ScratchV10EnvironmentIdentityV5Candidate:
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
    identity_sha256: str

    def _identity_payload(self) -> dict[str, Any]:
        payload = asdict(self)
        payload.pop("identity_sha256")
        return payload

    def validate(self, *, verify_current_default_runtime: bool = True) -> None:
        exact = {
            "format": V10_ENVIRONMENT_IDENTITY_V5_CANDIDATE_FORMAT,
            "environment_class": "RealisticEdgeArmEnvV10",
            "environment_config_class": "RealisticEnvV10Config",
        }
        for name, expected in exact.items():
            if getattr(self, name) != expected:
                raise ValueError(f"V5 environment identity {name} mismatch")
        canonical_config = _validated_environment_config_payload_v10(
            self.environment_config
        )
        if (
            self.environment_config_sha256 != _canonical_sha256(canonical_config)
            or not _is_sha256(self.environment_config_sha256)
        ):
            raise ValueError("V5 environment config hash mismatch")
        if self.runtime_source_mode not in {
            "immutable_production_mjcf_bundle_v1",
            "double_read_default_runtime_snapshot_v1",
        }:
            raise ValueError("V5 environment runtime source mode mismatch")
        if not isinstance(self.runtime_main_logical_path, str) or not (
            self.runtime_main_logical_path
        ):
            raise ValueError("V5 environment runtime main path is invalid")
        manifest = self.runtime_file_manifest
        if not isinstance(manifest, tuple) or not manifest:
            raise ValueError("V5 environment runtime manifest is empty")
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
                raise ValueError("V5 environment runtime manifest entry is invalid")
            paths.append(entry[0])
        if len(set(paths)) != len(paths) or self.runtime_main_logical_path not in paths:
            raise ValueError("V5 environment runtime manifest paths are invalid")
        if (
            self.runtime_file_manifest_sha256 != _canonical_sha256(manifest)
            or not _is_sha256(self.runtime_bundle_sha256)
        ):
            raise ValueError("V5 environment runtime manifest hash mismatch")
        if not _is_sha256(self.compiled_model_sha256):
            raise ValueError("V5 environment compiled model hash is malformed")
        if type(self.compiled_model_bytes) is not int or self.compiled_model_bytes < 1:
            raise ValueError("V5 environment compiled model byte count is invalid")
        if self.identity_sha256 != _canonical_sha256(self._identity_payload()):
            raise ValueError("V5 environment identity hash mismatch")
        if (
            verify_current_default_runtime
            and self.runtime_source_mode == "double_read_default_runtime_snapshot_v1"
        ):
            snapshot = capture_causal_runtime_snapshot_v1(SCENE_PATH.parents[1])
            current_manifest = _runtime_file_manifest_v5(snapshot.files)
            if (
                self.runtime_main_logical_path != CAUSAL_RUNTIME_MAIN_LOGICAL_PATH
                or self.runtime_bundle_sha256 != snapshot.runtime_bundle_sha256
                or self.runtime_file_manifest != current_manifest
            ):
                raise ValueError("V5 default runtime changed since environment binding")


def _build_environment_identity_v5(
    env: RealisticEdgeArmEnvV10,
    *,
    compiled_model_override: tuple[str, int] | None = None,
) -> ScratchV10EnvironmentIdentityV5Candidate:
    exact_env = ScratchPotentialRewardV5Candidate._require_exact_v10(env)
    config_payload = _validated_environment_config_payload_v10(asdict(exact_env.config))
    source_mode, main_logical_path, files = _runtime_files_for_exact_v10(exact_env)
    manifest = _runtime_file_manifest_v5(files)
    compiled_sha256, compiled_bytes = (
        _compiled_model_identity_v5(exact_env)
        if compiled_model_override is None
        else compiled_model_override
    )
    base: dict[str, Any] = {
        "format": V10_ENVIRONMENT_IDENTITY_V5_CANDIDATE_FORMAT,
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
    }
    identity = ScratchV10EnvironmentIdentityV5Candidate(
        **base,
        identity_sha256=_canonical_sha256(base),
    )
    identity.validate()
    return identity


@dataclass(frozen=True)
class ScratchPPOV5CandidateProvenance:
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
    trainer_source_hashes: dict[str, str]
    physical_samples: int
    physical_trials: int
    production_admission: bool
    genesis_sha256: str

    def _genesis_payload(self) -> dict[str, Any]:
        payload = asdict(self)
        payload.pop("genesis_sha256")
        return {
            "genesis_version": TRAINING_GENESIS_V5_CANDIDATE_VERSION,
            "actor_architecture": ACTOR_ARCHITECTURE,
            "critic_architecture": CRITIC_ARCHITECTURE,
            "policy_parameterization": POLICY_PARAMETERIZATION,
            **payload,
        }

    def validate(self, *, verify_current_sources: bool = True) -> None:
        exact = {
            "source_type": SOURCE_TYPE,
            "checkpoint_format": CHECKPOINT_FORMAT_V5_CANDIDATE,
            "environment_class": "RealisticEdgeArmEnvV10",
            "environment_config_class": "RealisticEnvV10Config",
            "random_initialization": True,
            "full_six_joint_action": True,
            "expert_action_inputs": 0,
            "controller_phase_inputs": 0,
            "behavior_cloning_steps": 0,
            "policy_observation_changed": False,
            "contact_telemetry_is_reward_input": True,
            "contact_telemetry_privilege": REWARD_INPUT_DISCLOSURE_V5_CANDIDATE[
                "contact_telemetry_privilege"
            ],
            "privileged_state_schema_sha256": PRIVILEGED_EFFECT_STATE_SCHEMA_SHA256,
            "potential_reward_version": POTENTIAL_REWARD_V5_CANDIDATE_VERSION,
            "physical_samples": 0,
            "physical_trials": 0,
            "production_admission": False,
        }
        for name, expected in exact.items():
            if getattr(self, name) != expected:
                raise ValueError(f"V5 provenance {name} mismatch")
        if type(self.initialization_seed) is not int or self.initialization_seed < 0:
            raise ValueError("V5 provenance initialization seed is invalid")
        for name in (
            "environment_source_sha256",
            "actor_initial_state_sha256",
            "critic_initial_state_sha256",
            "privileged_state_schema_sha256",
            "potential_reward_config_sha256",
            "genesis_sha256",
        ):
            if not _is_sha256(getattr(self, name)):
                raise ValueError(f"V5 provenance hash is malformed: {name}")
        if set(self.trainer_source_hashes) != set(_SOURCE_FILENAMES_V5) or any(
            not _is_sha256(value) for value in self.trainer_source_hashes.values()
        ):
            raise ValueError("V5 trainer source hashes are malformed")
        if self.environment_source_sha256 != self.trainer_source_hashes[
            "sim2real_env_v10.py"
        ]:
            raise ValueError("V5 provenance is not bound to its V10 source hash")
        if verify_current_sources and self.trainer_source_hashes != (
            scratch_v5_candidate_source_hashes()
        ):
            raise ValueError("V5 trainer sources changed since genesis")
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(self.initialization_seed)
            actor = FullActionScratchActorV1()
            critic = PrivilegedEffectCriticV1()
        if self.actor_initial_state_sha256 != state_dict_sha256_v1(actor.state_dict()):
            raise ValueError("V5 actor genesis is not canonical")
        if self.critic_initial_state_sha256 != state_dict_sha256_v1(critic.state_dict()):
            raise ValueError("V5 critic genesis is not canonical")
        expected_reward_sha = ScratchPotentialRewardV5CandidateConfig().sha256()
        if self.potential_reward_config_sha256 != expected_reward_sha:
            raise ValueError("V5 reward config provenance mismatch")
        if self.genesis_sha256 != _canonical_sha256(self._genesis_payload()):
            raise ValueError("V5 genesis hash mismatch")


@dataclass
class ScratchPPOV5CandidateBundle:
    actor: FullActionScratchActorV1
    critic: PrivilegedEffectCriticV1
    provenance: ScratchPPOV5CandidateProvenance
    potential_reward_config: ScratchPotentialRewardV5CandidateConfig
    environment_identity: ScratchV10EnvironmentIdentityV5Candidate | None = None
    _bound_environment: RealisticEdgeArmEnvV10 | None = dataclass_field(
        default=None,
        repr=False,
        compare=False,
    )


def bind_scratch_ppo_v5_environment_identity(
    bundle: ScratchPPOV5CandidateBundle,
    env: RealisticEdgeArmEnvV10,
) -> ScratchV10EnvironmentIdentityV5Candidate:
    """Bind one bundle to one exact V10 config/runtime/model identity."""

    if type(bundle) is not ScratchPPOV5CandidateBundle:
        raise TypeError("V5 environment binding requires exact V5 bundle")
    existing = bundle.environment_identity
    if existing is not None:
        existing.validate()
    override = None
    if existing is not None and bundle._bound_environment is env:
        override = (existing.compiled_model_sha256, existing.compiled_model_bytes)
    candidate = _build_environment_identity_v5(
        env,
        compiled_model_override=override,
    )
    if existing is not None and candidate != existing:
        raise ValueError("V5 bundle environment identity changed across updates")
    bundle.environment_identity = candidate
    bundle._bound_environment = env
    return candidate


def initialize_scratch_ppo_v5_candidate(
    seed: int,
    *,
    device: str | torch.device = "cpu",
) -> ScratchPPOV5CandidateBundle:
    if type(seed) is not int or seed < 0:
        raise ValueError("V5 seed must be a non-negative integer")
    reward_config = ScratchPotentialRewardV5CandidateConfig()
    reward_config.validate()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        actor = FullActionScratchActorV1()
        critic = PrivilegedEffectCriticV1()
    source_hashes = scratch_v5_candidate_source_hashes()
    base: dict[str, Any] = {
        "source_type": SOURCE_TYPE,
        "checkpoint_format": CHECKPOINT_FORMAT_V5_CANDIDATE,
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
        "contact_telemetry_privilege": REWARD_INPUT_DISCLOSURE_V5_CANDIDATE[
            "contact_telemetry_privilege"
        ],
        "actor_initial_state_sha256": state_dict_sha256_v1(actor.state_dict()),
        "critic_initial_state_sha256": state_dict_sha256_v1(critic.state_dict()),
        "privileged_state_schema_sha256": PRIVILEGED_EFFECT_STATE_SCHEMA_SHA256,
        "potential_reward_version": POTENTIAL_REWARD_V5_CANDIDATE_VERSION,
        "potential_reward_config_sha256": reward_config.sha256(),
        "trainer_source_hashes": source_hashes,
        "physical_samples": 0,
        "physical_trials": 0,
        "production_admission": False,
    }
    provenance = ScratchPPOV5CandidateProvenance(
        **base,
        genesis_sha256=_canonical_sha256(
            {
                "genesis_version": TRAINING_GENESIS_V5_CANDIDATE_VERSION,
                "actor_architecture": ACTOR_ARCHITECTURE,
                "critic_architecture": CRITIC_ARCHITECTURE,
                "policy_parameterization": POLICY_PARAMETERIZATION,
                **base,
            }
        ),
    )
    provenance.validate()
    return ScratchPPOV5CandidateBundle(
        actor=actor.to(device),
        critic=critic.to(device),
        provenance=provenance,
        potential_reward_config=reward_config,
    )


def build_scratch_checkpoint_payload_v5_candidate(
    bundle: ScratchPPOV5CandidateBundle,
    config: ScratchPPOConfigV1,
    *,
    updates_completed: int,
    env: RealisticEdgeArmEnvV10 | None = None,
) -> dict[str, Any]:
    if type(bundle) is not ScratchPPOV5CandidateBundle:
        raise TypeError("V5 checkpoint requires exact V5 bundle")
    if type(config) is not ScratchPPOConfigV1:
        raise TypeError("V5 checkpoint requires exact ScratchPPOConfigV1")
    bundle.provenance.validate()
    bundle.potential_reward_config.validate()
    config.validate()
    if type(updates_completed) is not int or updates_completed < 0:
        raise ValueError("V5 updates_completed must be non-negative")
    if env is not None:
        bind_scratch_ppo_v5_environment_identity(bundle, env)
    environment_identity = bundle.environment_identity
    if environment_identity is None:
        raise ValueError("V5 checkpoint environment identity is unbound")
    environment_identity.validate()
    actor_state = bundle.actor.state_dict()
    critic_state = bundle.critic.state_dict()
    actor_state_sha256 = state_dict_sha256_v1(actor_state)
    critic_state_sha256 = state_dict_sha256_v1(critic_state)
    if updates_completed == 0 and (
        actor_state_sha256 != bundle.provenance.actor_initial_state_sha256
        or critic_state_sha256 != bundle.provenance.critic_initial_state_sha256
    ):
        raise ValueError("V5 zero-update checkpoint differs from canonical genesis")
    return {
        "format": CHECKPOINT_FORMAT_V5_CANDIDATE,
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
        "potential_reward_version": POTENTIAL_REWARD_V5_CANDIDATE_VERSION,
        "potential_reward_formula": POTENTIAL_REWARD_V5_CANDIDATE_FORMULA,
        "potential_reward_config": asdict(bundle.potential_reward_config),
        "potential_reward_config_sha256": bundle.potential_reward_config.sha256(),
        "reward_input_disclosure": dict(REWARD_INPUT_DISCLOSURE_V5_CANDIDATE),
        "actor_state": actor_state,
        "critic_state": critic_state,
        "actor_state_sha256": actor_state_sha256,
        "critic_state_sha256": critic_state_sha256,
        "config": asdict(config),
        "updates_completed": updates_completed,
        "provenance": asdict(bundle.provenance),
        "stock_follower_unmodified": True,
        "added_contact_tool": False,
        "production_admission": False,
        "physical_samples": 0,
        "physical_trials": 0,
        "admission_status": "v5_v10_candidate_not_admitted_for_causal_collection",
    }


def _validate_model_state(
    state: object,
    declared_hash: object,
    canonical_module: torch.nn.Module,
    *,
    name: str,
) -> None:
    if not isinstance(state, dict):
        raise ValueError(f"V5 checkpoint {name} state is missing")
    if state_dict_sha256_v1(state) != declared_hash:
        raise ValueError(f"V5 checkpoint {name} hash mismatch")
    if any(
        not isinstance(value, torch.Tensor)
        or not bool(torch.all(torch.isfinite(value)).item())
        for value in state.values()
    ):
        raise ValueError(f"V5 checkpoint {name} state is non-finite")
    canonical = canonical_module.state_dict()
    if set(state) != set(canonical) or any(
        state[key].shape != canonical[key].shape or state[key].dtype != canonical[key].dtype
        for key in canonical
    ):
        raise ValueError(f"V5 checkpoint {name} tensor schema mismatch")
    try:
        canonical_module.load_state_dict(state, strict=True)
    except RuntimeError as error:
        raise ValueError(f"V5 checkpoint {name} state schema mismatch") from error


def _environment_identity_from_payload_v5(
    payload: object,
    *,
    verify_current_default_runtime: bool,
) -> ScratchV10EnvironmentIdentityV5Candidate:
    expected = {
        item.name for item in fields(ScratchV10EnvironmentIdentityV5Candidate)
    }
    if not isinstance(payload, dict) or set(payload) != expected:
        raise ValueError("V5 checkpoint environment identity schema mismatch")
    try:
        identity = ScratchV10EnvironmentIdentityV5Candidate(**payload)
        identity.validate(
            verify_current_default_runtime=verify_current_default_runtime,
        )
    except (TypeError, ValueError) as error:
        raise ValueError("V5 checkpoint environment identity is invalid") from error
    if asdict(identity) != payload:
        raise ValueError("V5 checkpoint environment identity is non-canonical")
    return identity


def validate_scratch_checkpoint_payload_v5_candidate(
    payload: object,
    *,
    verify_current_sources: bool = True,
) -> ScratchPPOV5CandidateProvenance:
    """Validate the exact V5/V10 checkpoint schema and internal hashes."""

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
    if not isinstance(payload, dict) or set(payload) != expected_fields:
        raise ValueError("V5 checkpoint fields are incomplete or unexpected")
    exact = {
        "format": CHECKPOINT_FORMAT_V5_CANDIDATE,
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
        "potential_reward_version": POTENTIAL_REWARD_V5_CANDIDATE_VERSION,
        "potential_reward_formula": POTENTIAL_REWARD_V5_CANDIDATE_FORMULA,
        "reward_input_disclosure": REWARD_INPUT_DISCLOSURE_V5_CANDIDATE,
        "stock_follower_unmodified": True,
        "added_contact_tool": False,
        "production_admission": False,
        "physical_samples": 0,
        "physical_trials": 0,
        "admission_status": "v5_v10_candidate_not_admitted_for_causal_collection",
    }
    for name, expected in exact.items():
        if payload.get(name) != expected:
            raise ValueError(f"V5 checkpoint {name} mismatch")
    if not _is_sha256(payload.get("environment_source_sha256")):
        raise ValueError("V5 checkpoint V10 source hash is malformed")
    identity = _environment_identity_from_payload_v5(
        payload.get("environment_identity"),
        verify_current_default_runtime=verify_current_sources,
    )
    if payload.get("environment_identity_sha256") != identity.identity_sha256:
        raise ValueError("V5 checkpoint environment identity hash mismatch")
    reward_payload = payload.get("potential_reward_config")
    if not isinstance(reward_payload, dict) or set(reward_payload) != {
        field.name for field in fields(ScratchPotentialRewardV5CandidateConfig)
    }:
        raise ValueError("V5 checkpoint reward config is malformed")
    try:
        reward_config = ScratchPotentialRewardV5CandidateConfig(**reward_payload)
        reward_config.validate()
    except (TypeError, ValueError) as error:
        raise ValueError("V5 checkpoint reward config is invalid") from error
    if (
        asdict(reward_config) != reward_payload
        or payload.get("potential_reward_config_sha256") != reward_config.sha256()
    ):
        raise ValueError("V5 checkpoint reward config hash mismatch")
    _validate_model_state(
        payload.get("actor_state"),
        payload.get("actor_state_sha256"),
        FullActionScratchActorV1(),
        name="actor",
    )
    _validate_model_state(
        payload.get("critic_state"),
        payload.get("critic_state_sha256"),
        PrivilegedEffectCriticV1(),
        name="critic",
    )
    config_payload = payload.get("config")
    if not isinstance(config_payload, dict) or set(config_payload) != {
        field.name for field in fields(ScratchPPOConfigV1)
    }:
        raise ValueError("V5 checkpoint PPO config is malformed")
    try:
        config = ScratchPPOConfigV1(**config_payload)
        config.validate()
    except (TypeError, ValueError) as error:
        raise ValueError("V5 checkpoint PPO config is invalid") from error
    if asdict(config) != config_payload:
        raise ValueError("V5 checkpoint PPO config is non-canonical")
    if type(payload.get("updates_completed")) is not int or payload["updates_completed"] < 0:
        raise ValueError("V5 checkpoint update count is invalid")
    provenance_payload = payload.get("provenance")
    if not isinstance(provenance_payload, dict) or set(provenance_payload) != {
        field.name for field in fields(ScratchPPOV5CandidateProvenance)
    }:
        raise ValueError("V5 checkpoint provenance is malformed")
    try:
        provenance = ScratchPPOV5CandidateProvenance(**provenance_payload)
        provenance.validate(verify_current_sources=verify_current_sources)
    except (TypeError, ValueError) as error:
        raise ValueError("V5 checkpoint provenance is invalid") from error
    if provenance.potential_reward_config_sha256 != reward_config.sha256():
        raise ValueError("V5 checkpoint provenance/reward mismatch")
    if (
        provenance.environment_source_sha256 != payload["environment_source_sha256"]
        or provenance.environment_source_sha256
        != provenance.trainer_source_hashes["sim2real_env_v10.py"]
    ):
        raise ValueError("V5 checkpoint V10 source binding mismatch")
    if payload["updates_completed"] == 0 and (
        payload["actor_state_sha256"] != provenance.actor_initial_state_sha256
        or payload["critic_state_sha256"] != provenance.critic_initial_state_sha256
    ):
        raise ValueError("V5 zero-update checkpoint does not match canonical genesis")
    return provenance


def _validate_adam_optimizer_v5_candidate(
    optimizer: torch.optim.Optimizer,
    bundle: ScratchPPOV5CandidateBundle,
    config: ScratchPPOConfigV1,
) -> None:
    if type(optimizer) is not torch.optim.Adam:
        raise TypeError("V5 update requires exact torch.optim.Adam")
    if type(bundle.actor) is not FullActionScratchActorV1:
        raise TypeError("V5 update requires exact FullActionScratchActorV1")
    if type(bundle.critic) is not PrivilegedEffectCriticV1:
        raise TypeError("V5 update requires exact PrivilegedEffectCriticV1")
    expected_parameters = [*bundle.actor.parameters(), *bundle.critic.parameters()]
    if len({id(parameter) for parameter in expected_parameters}) != len(
        expected_parameters
    ):
        raise RuntimeError("V5 actor/critic parameter identity is not unique")
    if len(optimizer.param_groups) != 1:
        raise ValueError("V5 Adam requires exactly one parameter group")
    group = optimizer.param_groups[0]
    actual_parameters = group.get("params")
    if (
        not isinstance(actual_parameters, list)
        or len(actual_parameters) != len(expected_parameters)
        or len({id(parameter) for parameter in actual_parameters})
        != len(actual_parameters)
        or any(
            actual is not expected
            for actual, expected in zip(
                actual_parameters,
                expected_parameters,
                strict=True,
            )
        )
    ):
        raise ValueError(
            "V5 Adam parameters must be the exact ordered actor+critic parameters"
        )
    canonical = torch.optim.Adam(
        expected_parameters,
        lr=config.learning_rate,
    )
    canonical_group = canonical.param_groups[0]
    if set(group) != set(canonical_group):
        raise ValueError("V5 Adam parameter-group schema mismatch")
    for name, expected in canonical_group.items():
        if name != "params" and group[name] != expected:
            raise ValueError(f"V5 Adam hyperparameter mismatch: {name}")
    state_parameters = set(optimizer.state)
    expected_parameter_set = set(expected_parameters)
    if state_parameters and state_parameters != expected_parameter_set:
        raise ValueError("V5 Adam state does not cover the exact model parameters")
    for parameter, parameter_state in optimizer.state.items():
        if parameter not in expected_parameter_set or not isinstance(
            parameter_state, dict
        ):
            raise ValueError("V5 Adam state contains a foreign parameter")
        expected_state_fields = {"step", "exp_avg", "exp_avg_sq"}
        if bool(group["amsgrad"]):
            expected_state_fields.add("max_exp_avg_sq")
        if set(parameter_state) != expected_state_fields:
            raise ValueError("V5 Adam per-parameter state schema mismatch")
        step = parameter_state["step"]
        if (
            not isinstance(step, torch.Tensor)
            or step.numel() != 1
            or not bool(torch.all(torch.isfinite(step)).item())
            or float(step.item()) < 0.0
        ):
            raise ValueError("V5 Adam step state is invalid")
        for name in expected_state_fields - {"step"}:
            value = parameter_state[name]
            if (
                not isinstance(value, torch.Tensor)
                or value.shape != parameter.shape
                or value.dtype != parameter.dtype
                or value.device != parameter.device
                or not bool(torch.all(torch.isfinite(value)).item())
            ):
                raise ValueError(f"V5 Adam tensor state is invalid: {name}")


def train_one_scratch_update_v5_candidate(
    env: RealisticEdgeArmEnvV10,
    bundle: ScratchPPOV5CandidateBundle,
    config: ScratchPPOConfigV1,
    *,
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[
    ScratchRolloutBatchV5Candidate,
    PPOUpdateMetricsV1,
    torch.optim.Optimizer,
]:
    """Run one unchanged PPO update using only exact-V10 Reward V5 data."""

    ScratchPotentialRewardV5Candidate._require_exact_v10(env)
    if type(bundle) is not ScratchPPOV5CandidateBundle:
        raise TypeError("V5 update requires exact V5 bundle")
    if type(config) is not ScratchPPOConfigV1:
        raise TypeError("V5 update requires exact ScratchPPOConfigV1")
    config.validate()
    bundle.provenance.validate()
    bundle.potential_reward_config.validate()
    if optimizer is None:
        optimizer = torch.optim.Adam(
            [*bundle.actor.parameters(), *bundle.critic.parameters()],
            lr=config.learning_rate,
        )
    _validate_adam_optimizer_v5_candidate(optimizer, bundle, config)
    bind_scratch_ppo_v5_environment_identity(bundle, env)
    reward = ScratchPotentialRewardV5Candidate(bundle.potential_reward_config)
    rollout = collect_scratch_rollout_v5_candidate(
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
    metrics, returned_optimizer = ppo_update_v1(
        bundle.actor,
        bundle.critic,
        rollout,  # type: ignore[arg-type]
        config,
        optimizer=optimizer,
    )
    if returned_optimizer is not optimizer:
        raise RuntimeError("V5 PPO update replaced the bound Adam optimizer")
    _validate_adam_optimizer_v5_candidate(returned_optimizer, bundle, config)
    return rollout, metrics, returned_optimizer


__all__ = [
    "CHECKPOINT_FORMAT_V5_CANDIDATE",
    "POTENTIAL_REWARD_V5_CANDIDATE_FORMULA",
    "POTENTIAL_REWARD_V5_CANDIDATE_VERSION",
    "REWARD_INPUT_DISCLOSURE_V5_CANDIDATE",
    "ROLLOUT_DIAGNOSTICS_V5_CANDIDATE_FORMAT",
    "V10_ENVIRONMENT_IDENTITY_V5_CANDIDATE_FORMAT",
    "ScratchPotentialEvaluationV5Candidate",
    "ScratchPotentialRewardV5Candidate",
    "ScratchPotentialRewardV5CandidateConfig",
    "ScratchPotentialTransitionV5Candidate",
    "ScratchPPOV5CandidateBundle",
    "ScratchPPOV5CandidateProvenance",
    "ScratchRolloutBatchV5Candidate",
    "ScratchRolloutDiagnosticsV5Candidate",
    "ScratchSafetyEvidenceV5Candidate",
    "ScratchV10EnvironmentIdentityV5Candidate",
    "bind_scratch_ppo_v5_environment_identity",
    "build_scratch_checkpoint_payload_v5_candidate",
    "collect_scratch_rollout_v5_candidate",
    "initialize_scratch_ppo_v5_candidate",
    "scratch_v5_candidate_source_hashes",
    "train_one_scratch_update_v5_candidate",
    "validate_scratch_checkpoint_payload_v5_candidate",
]
