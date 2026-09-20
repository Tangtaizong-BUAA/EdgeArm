"""Direction-preserving execution-aware reward for V22 on-policy PPO.

The V21 Euclidean projection cost penalized safe magnitude backtracking even
when the adapter preserved the requested task direction.  In the first V22
trial that cost dominated sparse contact/progress reward and taught the actor
to prefer easily executable retreat actions.  V22 therefore penalizes only
direction mismatch (plus an optional zero-application term); same-direction
attenuation has zero cost.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Any

import numpy as np
import torch

from .asymmetric_multiview_ppo_v1 import (
    POLICY_ACTION_DIM,
    AsymmetricMultiViewPPOConfigV1,
    AsymmetricMultiViewRolloutBatchV1,
    AsymmetricPrivilegedCriticV1,
    PPOUpdateMetricsV1,
    SelectedViewRecurrentActorV1,
    ppo_update_asymmetric_multiview_v1,
)
from .feasible_on_policy_ppo_v21 import (
    RolloutEligibilityConfigV21,
    build_batch_projection_audit_v21,
    rollout_update_gates_v21,
)
from .on_policy_recurrent_ppo_v20 import (
    OnPolicyIntrinsicRewardConfigV20,
    build_on_policy_learning_batch_v20,
)
from .ppo_utils_v1 import state_dict_sha256_v1


DIRECTIONAL_FEASIBLE_ON_POLICY_PPO_FORMAT_V22 = (
    "edgearm-v22-direction-preserving-execution-aware-on-policy-ppo-v1"
)


@dataclass(frozen=True)
class DirectionalFeasibleRewardConfigV22:
    outcome_reward: OnPolicyIntrinsicRewardConfigV20 = OnPolicyIntrinsicRewardConfigV20()
    maximum_direction_mismatch_penalty: float = 0.025
    direction_mismatch_scale: float = 0.25
    zero_application_penalty: float = 0.0
    minimum_action_norm: float = 1.0e-3

    def validate(self) -> None:
        self.outcome_reward.validate()
        values = np.asarray(
            [
                self.maximum_direction_mismatch_penalty,
                self.direction_mismatch_scale,
                self.zero_application_penalty,
                self.minimum_action_norm,
            ],
            dtype=np.float64,
        )
        if not np.all(np.isfinite(values)) or np.any(values < 0.0):
            raise ValueError("V22 directional reward parameters must be finite and non-negative")
        if not 0.0 < self.direction_mismatch_scale <= 2.0:
            raise ValueError("V22 direction mismatch scale must be in (0,2]")
        if not 0.0 < self.minimum_action_norm <= 0.05:
            raise ValueError("V22 minimum action norm must be in (0,0.05]")
        if self.maximum_direction_mismatch_penalty > 0.05:
            raise ValueError("V22 direction mismatch penalty exceeds its safety bound")
        if self.zero_application_penalty > 0.05:
            raise ValueError("V22 zero-application penalty exceeds its safety bound")
        if self.maximum_direction_mismatch_penalty + self.zero_application_penalty > 0.075:
            raise ValueError("V22 combined execution penalty exceeds its safety bound")


@dataclass(frozen=True)
class DirectionalFeasibleRewardAuditV22:
    transition_count: int
    source_reward_mean: float
    outcome_intrinsic_reward_mean: float
    direction_mismatch_mean: float
    direction_mismatch_p95: float
    direction_penalty_mean: float
    direction_penalty_p95: float
    direction_penalty_maximum: float
    valid_contact_direction_penalty_mean: float
    block_motion_direction_penalty_mean: float
    zero_application_transition_count: int
    learning_reward_mean: float
    outcome_reward_audit: dict[str, Any]
    format: str = DIRECTIONAL_FEASIBLE_ON_POLICY_PPO_FORMAT_V22


def compute_directional_projection_penalty_v22(
    *,
    policy_action: np.ndarray,
    applied_task_action: np.ndarray,
    config: DirectionalFeasibleRewardConfigV22,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return bounded direction-only cost, mismatch, and zero-apply mask."""

    config.validate()
    requested = np.asarray(policy_action)
    applied = np.asarray(applied_task_action)
    if (
        requested.ndim != 2
        or requested.shape[1:] != (POLICY_ACTION_DIM,)
        or applied.shape != requested.shape
        or requested.dtype != np.float32
        or applied.dtype != np.float32
        or not np.all(np.isfinite(requested))
        or not np.all(np.isfinite(applied))
        or np.any(np.abs(requested) > 1.0 + 1.0e-6)
        or np.any(np.abs(applied) > 1.0 + 1.0e-6)
    ):
        raise ValueError("V22 directional projection inputs are invalid")
    requested_norm = np.linalg.norm(requested, axis=1).astype(np.float32)
    applied_norm = np.linalg.norm(applied, axis=1).astype(np.float32)
    requested_nonzero = requested_norm > np.float32(config.minimum_action_norm)
    applied_nonzero = applied_norm > np.float32(config.minimum_action_norm)
    paired_nonzero = requested_nonzero & applied_nonzero
    cosine = np.ones(len(requested), dtype=np.float32)
    cosine[paired_nonzero] = np.divide(
        np.sum(
            requested[paired_nonzero] * applied[paired_nonzero],
            axis=1,
            dtype=np.float32,
        ),
        requested_norm[paired_nonzero] * applied_norm[paired_nonzero],
    )
    direction_mismatch = np.subtract(
        np.float32(1.0),
        np.clip(cosine, -1.0, 1.0),
        dtype=np.float32,
    )
    zero_application = requested_nonzero & ~applied_nonzero
    direction_penalty = np.multiply(
        np.float32(config.maximum_direction_mismatch_penalty),
        np.clip(
            direction_mismatch / np.float32(config.direction_mismatch_scale),
            0.0,
            1.0,
        ),
        dtype=np.float32,
    )
    penalty = np.add(
        direction_penalty,
        zero_application.astype(np.float32) * np.float32(config.zero_application_penalty),
        dtype=np.float32,
    )
    if not all(np.all(np.isfinite(value)) for value in (penalty, direction_mismatch)):
        raise RuntimeError("V22 directional projection cost became non-finite")
    return penalty, direction_mismatch, zero_application


def _masked_mean(value: np.ndarray, mask: np.ndarray) -> float:
    return float(value[mask].mean()) if np.any(mask) else 0.0


def build_directional_feasible_learning_batch_v22(
    batch: AsymmetricMultiViewRolloutBatchV1,
    config: DirectionalFeasibleRewardConfigV22,
) -> tuple[AsymmetricMultiViewRolloutBatchV1, DirectionalFeasibleRewardAuditV22]:
    config.validate()
    outcome_batch, outcome_audit = build_on_policy_learning_batch_v20(
        batch,
        config.outcome_reward,
    )
    penalty, mismatch, zero_application = compute_directional_projection_penalty_v22(
        policy_action=batch.policy_action,
        applied_task_action=batch.applied_task_action,
        config=config,
    )
    learning_rewards = np.subtract(
        outcome_batch.shaped_rewards,
        penalty,
        dtype=np.float32,
    )
    if not np.all(np.isfinite(learning_rewards)):
        raise RuntimeError("V22 directional learning reward became non-finite")
    learning_batch = replace(batch, shaped_rewards=learning_rewards)
    learning_batch.validate()
    return learning_batch, DirectionalFeasibleRewardAuditV22(
        transition_count=len(learning_rewards),
        source_reward_mean=float(batch.shaped_rewards.mean()),
        outcome_intrinsic_reward_mean=float(outcome_audit.intrinsic_reward_mean),
        direction_mismatch_mean=float(mismatch.mean()),
        direction_mismatch_p95=float(np.quantile(mismatch, 0.95)),
        direction_penalty_mean=float(penalty.mean()),
        direction_penalty_p95=float(np.quantile(penalty, 0.95)),
        direction_penalty_maximum=float(penalty.max()),
        valid_contact_direction_penalty_mean=_masked_mean(
            penalty,
            batch.valid_push_side_contact_any,
        ),
        block_motion_direction_penalty_mean=_masked_mean(
            penalty,
            batch.step_block_displacement_m > 0.00005,
        ),
        zero_application_transition_count=int(np.count_nonzero(zero_application)),
        learning_reward_mean=float(learning_rewards.mean()),
        outcome_reward_audit=asdict(outcome_audit),
    )


def directional_feasible_on_policy_ppo_update_v22(
    actor: SelectedViewRecurrentActorV1,
    critic: AsymmetricPrivilegedCriticV1,
    rollout: AsymmetricMultiViewRolloutBatchV1,
    ppo_config: AsymmetricMultiViewPPOConfigV1,
    reward_config: DirectionalFeasibleRewardConfigV22,
    eligibility_config: RolloutEligibilityConfigV21,
    *,
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[
    PPOUpdateMetricsV1,
    DirectionalFeasibleRewardAuditV22,
    torch.optim.Optimizer,
]:
    projection_audit = build_batch_projection_audit_v21(
        rollout,
        eligibility_config,
    )
    gate_checks = rollout_update_gates_v21(projection_audit, eligibility_config)
    failed = [name for name, passed in gate_checks.items() if not passed]
    if failed:
        raise ValueError("V22 rollout is ineligible for directional actor update: " + ",".join(failed))
    actor_before = state_dict_sha256_v1(actor.state_dict())
    learning_batch, reward_audit = build_directional_feasible_learning_batch_v22(
        rollout,
        reward_config,
    )
    metrics, updated_optimizer = ppo_update_asymmetric_multiview_v1(
        actor,
        critic,
        learning_batch,
        ppo_config,
        optimizer=optimizer,
    )
    actor_after = state_dict_sha256_v1(actor.state_dict())
    if actor_after == actor_before:
        raise RuntimeError("V22 directional PPO update did not change the actor")
    if metrics.approximate_kl > ppo_config.target_kl + 1.0e-6:
        raise RuntimeError("V22 directional PPO update escaped its KL stop")
    return metrics, reward_audit, updated_optimizer


__all__ = [
    "DIRECTIONAL_FEASIBLE_ON_POLICY_PPO_FORMAT_V22",
    "DirectionalFeasibleRewardAuditV22",
    "DirectionalFeasibleRewardConfigV22",
    "build_directional_feasible_learning_batch_v22",
    "compute_directional_projection_penalty_v22",
    "directional_feasible_on_policy_ppo_update_v22",
]
