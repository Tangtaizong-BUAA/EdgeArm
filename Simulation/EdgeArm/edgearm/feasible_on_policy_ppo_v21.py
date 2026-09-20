"""Execution-aware feasibility gates and rewards for EdgeArm V21 PPO.

V20 can construct a mathematically valid PPO gradient from a rollout that has
no task-positive transitions.  Such a gradient is not useful for the pushing
objective, and post-hoc rollback still spends an optimizer step on an aliased
action distribution.  V21 therefore separates three decisions:

1. every rollout is retained as causal simulator evidence;
2. only a rollout with contact, block motion, and adequate executability may
   update the actor/critic;
3. an eligible proposal must still pass the paired closed-loop V20 gate.

The policy action and the task-space action admitted by IK/guard are both
persisted.  Their normalized L2 distance is a training-only projection cost;
it never alters the environment reward or the recorded source trajectory.
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
from .on_policy_recurrent_ppo_v20 import (
    OnPolicyIntrinsicRewardConfigV20,
    build_on_policy_learning_batch_v20,
)
from .ppo_utils_v1 import state_dict_sha256_v1


FEASIBLE_ON_POLICY_PPO_FORMAT_V21 = (
    "edgearm-v21-execution-aware-feasible-on-policy-ppo-v1"
)
FEASIBLE_ON_POLICY_CHECKPOINT_FORMAT_V21 = (
    "edgearm-v21-execution-aware-feasible-on-policy-ppo-checkpoint-v1"
)
FEASIBLE_ON_POLICY_H5_FORMAT_V21 = (
    "edgearm-v21-execution-aware-feasible-on-policy-trajectory-h5-v1"
)


@dataclass(frozen=True)
class RolloutEligibilityConfigV21:
    """Fail-closed evidence requirements before any PPO optimizer step."""

    minimum_valid_contact_transitions: int = 1
    minimum_block_motion_transitions: int = 1
    block_motion_threshold_m: float = 0.00005
    maximum_ik_failure_fraction: float = 0.50
    minimum_nonzero_applied_action_fraction: float = 0.50
    minimum_mean_application_scale: float = 0.10
    maximum_projection_l2_p95: float = 0.75
    nonzero_applied_action_l2_threshold: float = 1.0e-6
    projection_alias_l2_threshold: float = 1.0e-4

    def validate(self) -> None:
        for name in (
            "minimum_valid_contact_transitions",
            "minimum_block_motion_transitions",
        ):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"V21 {name} must be a positive integer")
        values = np.asarray(
            [
                self.block_motion_threshold_m,
                self.maximum_ik_failure_fraction,
                self.minimum_nonzero_applied_action_fraction,
                self.minimum_mean_application_scale,
                self.maximum_projection_l2_p95,
                self.nonzero_applied_action_l2_threshold,
                self.projection_alias_l2_threshold,
            ],
            dtype=np.float64,
        )
        if not np.all(np.isfinite(values)):
            raise ValueError("V21 rollout eligibility configuration is non-finite")
        if self.block_motion_threshold_m <= 0.0:
            raise ValueError("V21 block motion threshold must be positive")
        if not 0.0 <= self.maximum_ik_failure_fraction < 1.0:
            raise ValueError("V21 maximum IK failure fraction must be in [0,1)")
        if not 0.0 < self.minimum_nonzero_applied_action_fraction <= 1.0:
            raise ValueError("V21 nonzero applied-action fraction must be in (0,1]")
        if not 0.0 <= self.minimum_mean_application_scale <= 1.0:
            raise ValueError("V21 mean application scale must be in [0,1]")
        if not 0.0 < self.maximum_projection_l2_p95 <= 2.0 * np.sqrt(
            POLICY_ACTION_DIM
        ):
            raise ValueError("V21 projection P95 bound is invalid")
        if self.nonzero_applied_action_l2_threshold <= 0.0:
            raise ValueError("V21 nonzero applied-action threshold must be positive")
        if self.projection_alias_l2_threshold <= 0.0:
            raise ValueError("V21 projection alias threshold must be positive")


@dataclass(frozen=True)
class RolloutProjectionAuditV21:
    transition_count: int
    valid_contact_transition_count: int
    block_motion_transition_count: int
    ik_failure_transition_count: int
    ik_failure_fraction: float
    nonzero_applied_action_transition_count: int
    nonzero_applied_action_fraction: float
    zero_applied_action_transition_count: int
    mean_application_scale: float
    policy_action_l2_mean: float
    applied_action_l2_mean: float
    projection_alias_transition_count: int
    projection_alias_fraction: float
    projection_l2_mean: float
    projection_l2_p50: float
    projection_l2_p95: float
    projection_l2_maximum: float
    format: str = FEASIBLE_ON_POLICY_PPO_FORMAT_V21


@dataclass(frozen=True)
class FeasibleIntrinsicRewardConfigV21:
    """Outcome reward plus a bounded requested-to-applied projection cost."""

    outcome_reward: OnPolicyIntrinsicRewardConfigV20 = (
        OnPolicyIntrinsicRewardConfigV20()
    )
    maximum_projection_penalty: float = 0.25
    projection_distance_scale: float = 0.25

    def validate(self) -> None:
        self.outcome_reward.validate()
        values = np.asarray(
            [self.maximum_projection_penalty, self.projection_distance_scale],
            dtype=np.float64,
        )
        if not np.all(np.isfinite(values)) or np.any(values <= 0.0):
            raise ValueError("V21 projection reward parameters must be finite and positive")
        if self.maximum_projection_penalty > 0.50:
            raise ValueError("V21 maximum projection penalty exceeds its safety bound")


@dataclass(frozen=True)
class FeasibleRewardAuditV21:
    transition_count: int
    source_reward_mean: float
    outcome_intrinsic_reward_mean: float
    projection_penalty_mean: float
    projection_penalty_p95: float
    projection_penalty_maximum: float
    learning_reward_mean: float
    outcome_reward_audit: dict[str, Any]
    format: str = FEASIBLE_ON_POLICY_PPO_FORMAT_V21


def _validate_action_audit_inputs_v21(
    *,
    policy_action: np.ndarray,
    applied_task_action: np.ndarray,
    ik_application_scale: np.ndarray,
    ik_converged: np.ndarray,
    valid_contact: np.ndarray,
    block_displacement_m: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    requested = np.asarray(policy_action)
    applied = np.asarray(applied_task_action)
    scale = np.asarray(ik_application_scale)
    feasible = np.asarray(ik_converged)
    contact = np.asarray(valid_contact)
    displacement = np.asarray(block_displacement_m)
    if requested.ndim != 2 or requested.shape[1:] != (POLICY_ACTION_DIM,):
        raise ValueError("V21 policy action audit input must have shape [N,3]")
    count = requested.shape[0]
    if count < 1 or applied.shape != requested.shape:
        raise ValueError("V21 applied action audit input shape changed")
    if any(value.shape != (count,) for value in (scale, feasible, contact, displacement)):
        raise ValueError("V21 scalar action audit input shape changed")
    if requested.dtype != np.float32 or applied.dtype != np.float32:
        raise ValueError("V21 action audit inputs must be float32")
    if scale.dtype != np.float32 or displacement.dtype != np.float32:
        raise ValueError("V21 scalar action audit inputs must be float32")
    if feasible.dtype != np.dtype(bool) or contact.dtype != np.dtype(bool):
        raise ValueError("V21 action audit masks must be boolean")
    if not all(np.all(np.isfinite(value)) for value in (requested, applied, scale, displacement)):
        raise ValueError("V21 action audit inputs must be finite")
    if np.any(np.abs(requested) > 1.0 + 1.0e-6) or np.any(
        np.abs(applied) > 1.0 + 1.0e-6
    ):
        raise ValueError("V21 action audit input escaped normalized bounds")
    if np.any((scale < 0.0) | (scale > 1.0)):
        raise ValueError("V21 IK application scale escaped [0,1]")
    if np.any(feasible != (scale > 0.0)):
        raise ValueError("V21 IK convergence and application scale disagree")
    if np.any(displacement < 0.0):
        raise ValueError("V21 block displacement cannot be negative")
    return requested, applied, scale, feasible, contact, displacement


def build_rollout_projection_audit_v21(
    *,
    policy_action: np.ndarray,
    applied_task_action: np.ndarray,
    ik_application_scale: np.ndarray,
    ik_converged: np.ndarray,
    valid_contact: np.ndarray,
    block_displacement_m: np.ndarray,
    config: RolloutEligibilityConfigV21,
) -> RolloutProjectionAuditV21:
    """Measure action aliasing and task-positive evidence in one rollout."""

    config.validate()
    requested, applied, scale, feasible, contact, displacement = (
        _validate_action_audit_inputs_v21(
            policy_action=policy_action,
            applied_task_action=applied_task_action,
            ik_application_scale=ik_application_scale,
            ik_converged=ik_converged,
            valid_contact=valid_contact,
            block_displacement_m=block_displacement_m,
        )
    )
    projection_l2 = np.linalg.norm(requested - applied, axis=1)
    requested_l2 = np.linalg.norm(requested, axis=1)
    applied_l2 = np.linalg.norm(applied, axis=1)
    nonzero_applied = applied_l2 > config.nonzero_applied_action_l2_threshold
    aliased = projection_l2 > config.projection_alias_l2_threshold
    count = len(projection_l2)
    return RolloutProjectionAuditV21(
        transition_count=count,
        valid_contact_transition_count=int(np.count_nonzero(contact)),
        block_motion_transition_count=int(
            np.count_nonzero(displacement > config.block_motion_threshold_m)
        ),
        ik_failure_transition_count=int(np.count_nonzero(~feasible)),
        ik_failure_fraction=float(np.count_nonzero(~feasible) / count),
        nonzero_applied_action_transition_count=int(np.count_nonzero(nonzero_applied)),
        nonzero_applied_action_fraction=float(np.count_nonzero(nonzero_applied) / count),
        zero_applied_action_transition_count=int(np.count_nonzero(~nonzero_applied)),
        mean_application_scale=float(scale.mean()),
        policy_action_l2_mean=float(requested_l2.mean()),
        applied_action_l2_mean=float(applied_l2.mean()),
        projection_alias_transition_count=int(np.count_nonzero(aliased)),
        projection_alias_fraction=float(np.count_nonzero(aliased) / count),
        projection_l2_mean=float(projection_l2.mean()),
        projection_l2_p50=float(np.quantile(projection_l2, 0.50)),
        projection_l2_p95=float(np.quantile(projection_l2, 0.95)),
        projection_l2_maximum=float(projection_l2.max()),
    )


def build_batch_projection_audit_v21(
    batch: AsymmetricMultiViewRolloutBatchV1,
    config: RolloutEligibilityConfigV21,
) -> RolloutProjectionAuditV21:
    batch.validate()
    return build_rollout_projection_audit_v21(
        policy_action=batch.policy_action,
        applied_task_action=batch.applied_task_action,
        ik_application_scale=batch.ik_application_scale,
        ik_converged=batch.ik_converged,
        valid_contact=batch.valid_push_side_contact_any,
        block_displacement_m=batch.step_block_displacement_m,
        config=config,
    )


def rollout_update_gates_v21(
    audit: RolloutProjectionAuditV21,
    config: RolloutEligibilityConfigV21,
) -> dict[str, bool]:
    """Return independent fail-closed gates; all must pass before PPO."""

    config.validate()
    if type(audit) is not RolloutProjectionAuditV21:
        raise TypeError("V21 rollout update gate requires its exact projection audit")
    return {
        "has_valid_contact": (
            audit.valid_contact_transition_count
            >= config.minimum_valid_contact_transitions
        ),
        "has_block_motion": (
            audit.block_motion_transition_count
            >= config.minimum_block_motion_transitions
        ),
        "ik_failure_fraction_within_bound": (
            audit.ik_failure_fraction <= config.maximum_ik_failure_fraction
        ),
        "nonzero_applied_action_fraction_within_bound": (
            audit.nonzero_applied_action_fraction
            >= config.minimum_nonzero_applied_action_fraction
        ),
        "mean_application_scale_within_bound": (
            audit.mean_application_scale >= config.minimum_mean_application_scale
        ),
        "projection_p95_within_bound": (
            audit.projection_l2_p95 <= config.maximum_projection_l2_p95
        ),
    }


def compute_projection_penalty_v21(
    *,
    policy_action: np.ndarray,
    applied_task_action: np.ndarray,
    config: FeasibleIntrinsicRewardConfigV21,
) -> np.ndarray:
    """Bound the task-action projection cost without modifying source reward."""

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
        raise ValueError("V21 projection reward inputs are invalid")
    distance = np.linalg.norm(requested - applied, axis=1).astype(np.float32)
    penalty = np.multiply(
        np.float32(config.maximum_projection_penalty),
        np.clip(
            distance / np.float32(config.projection_distance_scale),
            0.0,
            1.0,
        ),
        dtype=np.float32,
    )
    if not np.all(np.isfinite(penalty)):
        raise RuntimeError("V21 projection penalty became non-finite")
    return penalty


def build_feasible_learning_batch_v21(
    batch: AsymmetricMultiViewRolloutBatchV1,
    config: FeasibleIntrinsicRewardConfigV21,
) -> tuple[AsymmetricMultiViewRolloutBatchV1, FeasibleRewardAuditV21]:
    """Relabel one eligible rollout with outcome reward minus projection cost."""

    config.validate()
    outcome_batch, outcome_audit = build_on_policy_learning_batch_v20(
        batch,
        config.outcome_reward,
    )
    penalty = compute_projection_penalty_v21(
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
        raise RuntimeError("V21 learning reward became non-finite")
    learning_batch = replace(batch, shaped_rewards=learning_rewards)
    learning_batch.validate()
    return learning_batch, FeasibleRewardAuditV21(
        transition_count=len(learning_rewards),
        source_reward_mean=float(batch.shaped_rewards.mean()),
        outcome_intrinsic_reward_mean=float(outcome_audit.intrinsic_reward_mean),
        projection_penalty_mean=float(penalty.mean()),
        projection_penalty_p95=float(np.quantile(penalty, 0.95)),
        projection_penalty_maximum=float(penalty.max()),
        learning_reward_mean=float(learning_rewards.mean()),
        outcome_reward_audit=asdict(outcome_audit),
    )


def feasible_on_policy_ppo_update_v21(
    actor: SelectedViewRecurrentActorV1,
    critic: AsymmetricPrivilegedCriticV1,
    rollout: AsymmetricMultiViewRolloutBatchV1,
    ppo_config: AsymmetricMultiViewPPOConfigV1,
    reward_config: FeasibleIntrinsicRewardConfigV21,
    eligibility_config: RolloutEligibilityConfigV21,
    *,
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[PPOUpdateMetricsV1, FeasibleRewardAuditV21, torch.optim.Optimizer]:
    """Perform one V21 update only after positive-evidence eligibility passes."""

    projection_audit = build_batch_projection_audit_v21(rollout, eligibility_config)
    gate_checks = rollout_update_gates_v21(projection_audit, eligibility_config)
    failed = [name for name, passed in gate_checks.items() if not passed]
    if failed:
        raise ValueError(
            "V21 rollout is ineligible for actor update: " + ",".join(failed)
        )
    actor_before = state_dict_sha256_v1(actor.state_dict())
    learning_batch, reward_audit = build_feasible_learning_batch_v21(
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
        raise RuntimeError("V21 PPO update did not change the actor")
    if metrics.approximate_kl > ppo_config.target_kl + 1.0e-6:
        raise RuntimeError("V21 PPO update escaped its on-policy KL stop")
    return metrics, reward_audit, updated_optimizer


__all__ = [
    "FEASIBLE_ON_POLICY_CHECKPOINT_FORMAT_V21",
    "FEASIBLE_ON_POLICY_H5_FORMAT_V21",
    "FEASIBLE_ON_POLICY_PPO_FORMAT_V21",
    "FeasibleIntrinsicRewardConfigV21",
    "FeasibleRewardAuditV21",
    "RolloutEligibilityConfigV21",
    "RolloutProjectionAuditV21",
    "build_batch_projection_audit_v21",
    "build_feasible_learning_batch_v21",
    "build_rollout_projection_audit_v21",
    "compute_projection_penalty_v21",
    "feasible_on_policy_ppo_update_v21",
    "rollout_update_gates_v21",
]
