"""Goal-directed, execution-aware on-policy PPO reward for V23.

Contact and absolute block motion are useful curriculum signals, but neither
proves that the block moved toward its target.  V23 derives signed target
distance progress from the persisted privileged training state, gives most of
the intrinsic reward to positive goal progress, keeps contact as a small
bootstrap bonus, penalizes regression, and retains V22's bounded direction-
only execution penalty.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace

import numpy as np
import torch

from .asymmetric_multiview_ppo_v1 import (
    AsymmetricMultiViewPPOConfigV1,
    AsymmetricMultiViewRolloutBatchV1,
    AsymmetricPrivilegedCriticV1,
    PPOUpdateMetricsV1,
    SelectedViewRecurrentActorV1,
    _actor_tensors,
    autoregressive_action_distribution_v1,
    ppo_update_asymmetric_multiview_v1,
    visual_geometry_target_from_privileged_v1,
)
from .directional_feasible_on_policy_ppo_v22 import (
    DirectionalFeasibleRewardConfigV22,
    compute_directional_projection_penalty_v22,
)
from .feasible_on_policy_ppo_v21 import (
    RolloutEligibilityConfigV21,
    build_batch_projection_audit_v21,
    rollout_update_gates_v21,
)
from .ppo_utils_v1 import squashed_gaussian_log_prob_v1, state_dict_sha256_v1


GOAL_DIRECTED_FEASIBLE_ON_POLICY_PPO_FORMAT_V23 = "edgearm-v23-goal-directed-execution-aware-on-policy-ppo-v1"
FRONTIER_BALANCED_GOAL_DIRECTED_PPO_FORMAT_V24 = (
    "edgearm-v24-frontier-balanced-goal-directed-on-policy-ppo-v1"
)
TARGET_OFFSET_NORMALIZATION_M_V23 = np.asarray([0.25, 0.25], dtype=np.float32)
MAXIMUM_BATCHED_REPLAY_LOG_PROBABILITY_DELTA_V25 = 1.0e-4
LOG_PROBABILITY_ALIGNMENT_MODE_V25 = (
    "causal-cache-collection-to-batched-training-numerical-alignment-v25"
)


@dataclass(frozen=True)
class GoalDirectedFeasibleRewardConfigV23:
    valid_contact_bonus: float = 0.25
    maximum_target_progress_bonus: float = 2.0
    target_progress_scale_m: float = 0.0005
    maximum_target_regression_penalty: float = 0.25
    target_regression_scale_m: float = 0.0005
    maximum_positive_potential_bonus: float = 0.10
    positive_potential_scale: float = 0.10
    ik_failure_penalty: float = 0.05
    maximum_direction_mismatch_penalty: float = 0.025
    direction_mismatch_scale: float = 0.25
    zero_application_penalty: float = 0.0
    minimum_action_norm: float = 1.0e-3

    def validate(self) -> None:
        values = np.asarray(list(asdict(self).values()), dtype=np.float64)
        if not np.all(np.isfinite(values)) or np.any(values < 0.0):
            raise ValueError("V23 goal reward parameters must be finite and non-negative")
        for name in (
            "target_progress_scale_m",
            "target_regression_scale_m",
            "positive_potential_scale",
            "direction_mismatch_scale",
            "minimum_action_norm",
        ):
            if float(getattr(self, name)) <= 0.0:
                raise ValueError(f"V23 {name} must be positive")
        if self.direction_mismatch_scale > 2.0:
            raise ValueError("V23 direction mismatch scale exceeds two")
        if self.minimum_action_norm > 0.05:
            raise ValueError("V23 minimum action norm exceeds 0.05")
        maximum_positive = (
            self.valid_contact_bonus
            + self.maximum_target_progress_bonus
            + self.maximum_positive_potential_bonus
        )
        maximum_negative = (
            self.maximum_target_regression_penalty
            + self.ik_failure_penalty
            + self.maximum_direction_mismatch_penalty
            + self.zero_application_penalty
        )
        if maximum_positive >= 3.0:
            raise ValueError("V23 maximum positive intrinsic reward is too large")
        if maximum_negative > 0.50:
            raise ValueError("V23 maximum negative intrinsic reward is too large")
        DirectionalFeasibleRewardConfigV22(
            maximum_direction_mismatch_penalty=(self.maximum_direction_mismatch_penalty),
            direction_mismatch_scale=self.direction_mismatch_scale,
            zero_application_penalty=self.zero_application_penalty,
            minimum_action_norm=self.minimum_action_norm,
        ).validate()


@dataclass(frozen=True)
class GoalDirectedRewardAuditV23:
    transition_count: int
    source_reward_mean: float
    goal_intrinsic_reward_mean: float
    learning_reward_mean: float
    valid_contact_transition_count: int
    positive_target_progress_transition_count: int
    target_regression_transition_count: int
    total_target_progress_m: float
    valid_contact_target_progress_mean_m: float
    maximum_target_progress_bonus: float
    maximum_target_regression_penalty: float
    positive_potential_transition_count: int
    ik_failure_transition_count: int
    direction_penalty_mean: float
    direction_penalty_p95: float
    direction_penalty_maximum: float
    valid_contact_direction_penalty_mean: float
    zero_application_transition_count: int
    ppo_log_probability_alignment_mode: str = "not_requested"
    ppo_log_probability_alignment_maximum_absolute_delta: float = 0.0
    ppo_log_probability_alignment_mean_absolute_delta: float = 0.0
    ppo_log_probability_alignment_tolerance: float = 0.0
    format: str = GOAL_DIRECTED_FEASIBLE_ON_POLICY_PPO_FORMAT_V23


def _bounded_log_probability_alignment_v25(
    stored_log_probabilities: np.ndarray,
    batched_replay_log_probabilities: np.ndarray,
    *,
    maximum_absolute_delta: float = MAXIMUM_BATCHED_REPLAY_LOG_PROBABILITY_DELTA_V25,
) -> tuple[np.ndarray, dict[str, float | str]]:
    """Fail closed before aligning two numerically equivalent PPO forwards.

    V24 collection encodes each new camera row independently and caches the
    resulting feature.  PPO minibatches re-encode the persisted history window
    in larger convolution batches.  CPU convolution accumulation order can
    therefore differ by a few ulps even though the actor weights, frames, and
    causal history are identical.  The exact causal-cache replay remains the
    provenance gate; this helper only aligns the denominator used by the
    batched PPO surrogate, under a deliberately small absolute bound.
    """

    stored = np.asarray(stored_log_probabilities)
    replayed = np.asarray(batched_replay_log_probabilities)
    if stored.shape != replayed.shape or stored.dtype != np.float32 or replayed.dtype != np.float32:
        raise ValueError("V25 PPO log-probability alignment requires matching float32 vectors")
    if stored.ndim != 1 or not np.all(np.isfinite(stored)) or not np.all(np.isfinite(replayed)):
        raise ValueError("V25 PPO log-probability alignment inputs must be finite vectors")
    if not np.isfinite(maximum_absolute_delta) or maximum_absolute_delta <= 0.0:
        raise ValueError("V25 PPO log-probability alignment tolerance must be positive")
    absolute_delta = np.abs(replayed - stored)
    observed_maximum = float(absolute_delta.max(initial=np.float32(0.0)))
    if observed_maximum > maximum_absolute_delta:
        raise ValueError(
            "V25 batched PPO replay diverged from exact causal collection: "
            f"maximum_absolute_delta={observed_maximum:.9g}, "
            f"tolerance={maximum_absolute_delta:.9g}"
        )
    return replayed.copy(), {
        "mode": LOG_PROBABILITY_ALIGNMENT_MODE_V25,
        "maximum_absolute_delta": observed_maximum,
        "mean_absolute_delta": float(absolute_delta.mean(dtype=np.float64)),
        "tolerance": float(maximum_absolute_delta),
    }


def align_cached_collection_log_probabilities_for_batched_ppo_v25(
    actor: SelectedViewRecurrentActorV1,
    batch: AsymmetricMultiViewRolloutBatchV1,
    config: AsymmetricMultiViewPPOConfigV1,
) -> tuple[AsymmetricMultiViewRolloutBatchV1, dict[str, float | str]]:
    """Recompute the PPO denominator with the exact batched training forward."""

    batch.validate()
    config.validate()
    try:
        device = next(actor.parameters()).device
    except StopIteration as error:  # pragma: no cover
        raise ValueError("V25 actor has no parameters") from error
    actor.eval()
    with torch.no_grad():
        previous_pre_tanh = torch.from_numpy(batch.previous_policy_pre_tanh).to(device)
        distribution = autoregressive_action_distribution_v1(
            actor.distribution(*_actor_tensors(batch, device)),
            previous_pre_tanh,
            config.action_autoregressive_rho,
        )
        replayed = (
            squashed_gaussian_log_prob_v1(
                distribution,
                torch.from_numpy(batch.pre_tanh).to(device),
            )
            .cpu()
            .numpy()
            .astype(np.float32, copy=False)
        )
    aligned, audit = _bounded_log_probability_alignment_v25(
        batch.old_log_probs,
        replayed,
    )
    aligned_batch = replace(batch, old_log_probs=aligned)
    aligned_batch.validate()
    return aligned_batch, audit


def target_distance_progress_from_batch_v23(
    batch: AsymmetricMultiViewRolloutBatchV1,
) -> np.ndarray:
    batch.validate()
    before_offset_m = np.multiply(
        batch.visual_geometry_target[:, -2:],
        TARGET_OFFSET_NORMALIZATION_M_V23,
        dtype=np.float32,
    )
    next_geometry = visual_geometry_target_from_privileged_v1(batch.next_privileged_state)
    after_offset_m = np.multiply(
        next_geometry[:, -2:],
        TARGET_OFFSET_NORMALIZATION_M_V23,
        dtype=np.float32,
    )
    before_distance = np.linalg.norm(before_offset_m, axis=1).astype(np.float32)
    after_distance = np.linalg.norm(after_offset_m, axis=1).astype(np.float32)
    progress = np.subtract(before_distance, after_distance, dtype=np.float32)
    if not np.all(np.isfinite(progress)):
        raise RuntimeError("V23 target-distance progress became non-finite")
    return progress


def compute_goal_directed_intrinsic_reward_v23(
    *,
    valid_contact: np.ndarray,
    target_distance_progress_m: np.ndarray,
    potential_delta: np.ndarray,
    ik_converged: np.ndarray,
    config: GoalDirectedFeasibleRewardConfigV23,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    config.validate()
    contact = np.asarray(valid_contact)
    progress = np.asarray(target_distance_progress_m)
    potential = np.asarray(potential_delta)
    feasible = np.asarray(ik_converged)
    count = len(contact)
    if (
        contact.shape != (count,)
        or progress.shape != (count,)
        or potential.shape != (count,)
        or feasible.shape != (count,)
        or contact.dtype != np.bool_
        or feasible.dtype != np.bool_
        or progress.dtype != np.float32
        or potential.dtype != np.float32
        or not np.all(np.isfinite(progress))
        or not np.all(np.isfinite(potential))
    ):
        raise ValueError("V23 goal-directed reward inputs are invalid")
    contact_bonus = contact.astype(np.float32) * np.float32(config.valid_contact_bonus)
    progress_bonus = np.float32(config.maximum_target_progress_bonus) * np.clip(
        np.maximum(progress, np.float32(0.0)) / np.float32(config.target_progress_scale_m),
        0.0,
        1.0,
    )
    regression_penalty = np.float32(config.maximum_target_regression_penalty) * np.clip(
        np.maximum(-progress, np.float32(0.0)) / np.float32(config.target_regression_scale_m),
        0.0,
        1.0,
    )
    potential_bonus = np.float32(config.maximum_positive_potential_bonus) * np.clip(
        np.maximum(potential, np.float32(0.0)) / np.float32(config.positive_potential_scale),
        0.0,
        1.0,
    )
    ik_penalty = (~feasible).astype(np.float32) * np.float32(config.ik_failure_penalty)
    intrinsic = np.subtract(
        np.add(
            np.add(contact_bonus, progress_bonus, dtype=np.float32),
            potential_bonus,
            dtype=np.float32,
        ),
        np.add(regression_penalty, ik_penalty, dtype=np.float32),
        dtype=np.float32,
    )
    if not np.all(np.isfinite(intrinsic)):
        raise RuntimeError("V23 goal-directed intrinsic reward became non-finite")
    return intrinsic, {
        "contact_bonus": contact_bonus,
        "target_progress_bonus": progress_bonus,
        "target_regression_penalty": regression_penalty,
        "positive_potential_bonus": potential_bonus,
        "ik_failure_penalty": ik_penalty,
    }


def _masked_mean(value: np.ndarray, mask: np.ndarray) -> float:
    return float(value[mask].mean()) if np.any(mask) else 0.0


def build_goal_directed_learning_batch_v23(
    batch: AsymmetricMultiViewRolloutBatchV1,
    config: GoalDirectedFeasibleRewardConfigV23,
) -> tuple[AsymmetricMultiViewRolloutBatchV1, GoalDirectedRewardAuditV23]:
    config.validate()
    batch.validate()
    target_progress = target_distance_progress_from_batch_v23(batch)
    potential_delta = np.subtract(
        batch.potential_after,
        batch.potential_before,
        dtype=np.float32,
    )
    intrinsic, components = compute_goal_directed_intrinsic_reward_v23(
        valid_contact=batch.valid_push_side_contact_any,
        target_distance_progress_m=target_progress,
        potential_delta=potential_delta,
        ik_converged=batch.ik_converged,
        config=config,
    )
    directional_config = DirectionalFeasibleRewardConfigV22(
        maximum_direction_mismatch_penalty=(config.maximum_direction_mismatch_penalty),
        direction_mismatch_scale=config.direction_mismatch_scale,
        zero_application_penalty=config.zero_application_penalty,
        minimum_action_norm=config.minimum_action_norm,
    )
    direction_penalty, _mismatch, zero_application = compute_directional_projection_penalty_v22(
        policy_action=batch.policy_action,
        applied_task_action=batch.applied_task_action,
        config=directional_config,
    )
    learning_rewards = np.subtract(
        np.add(batch.shaped_rewards, intrinsic, dtype=np.float32),
        direction_penalty,
        dtype=np.float32,
    )
    if not np.all(np.isfinite(learning_rewards)):
        raise RuntimeError("V23 learning reward became non-finite")
    learning_batch = replace(batch, shaped_rewards=learning_rewards)
    learning_batch.validate()
    contact = batch.valid_push_side_contact_any
    return learning_batch, GoalDirectedRewardAuditV23(
        transition_count=len(learning_rewards),
        source_reward_mean=float(batch.shaped_rewards.mean()),
        goal_intrinsic_reward_mean=float(intrinsic.mean()),
        learning_reward_mean=float(learning_rewards.mean()),
        valid_contact_transition_count=int(np.count_nonzero(contact)),
        positive_target_progress_transition_count=int(np.count_nonzero(target_progress > 0.0)),
        target_regression_transition_count=int(np.count_nonzero(target_progress < 0.0)),
        total_target_progress_m=float(target_progress.sum()),
        valid_contact_target_progress_mean_m=_masked_mean(
            target_progress,
            contact,
        ),
        maximum_target_progress_bonus=float(components["target_progress_bonus"].max()),
        maximum_target_regression_penalty=float(components["target_regression_penalty"].max()),
        positive_potential_transition_count=int(np.count_nonzero(potential_delta > 0.0)),
        ik_failure_transition_count=int(np.count_nonzero(~batch.ik_converged)),
        direction_penalty_mean=float(direction_penalty.mean()),
        direction_penalty_p95=float(np.quantile(direction_penalty, 0.95)),
        direction_penalty_maximum=float(direction_penalty.max()),
        valid_contact_direction_penalty_mean=_masked_mean(
            direction_penalty,
            contact,
        ),
        zero_application_transition_count=int(np.count_nonzero(zero_application)),
    )


def goal_directed_feasible_on_policy_ppo_update_v23(
    actor: SelectedViewRecurrentActorV1,
    critic: AsymmetricPrivilegedCriticV1,
    rollout: AsymmetricMultiViewRolloutBatchV1,
    ppo_config: AsymmetricMultiViewPPOConfigV1,
    reward_config: GoalDirectedFeasibleRewardConfigV23,
    eligibility_config: RolloutEligibilityConfigV21,
    *,
    optimizer: torch.optim.Optimizer | None = None,
    stratify_advantages_by_episode: bool = False,
    align_cached_collection_numerics_v25: bool = False,
) -> tuple[PPOUpdateMetricsV1, GoalDirectedRewardAuditV23, torch.optim.Optimizer]:
    projection_audit = build_batch_projection_audit_v21(
        rollout,
        eligibility_config,
    )
    gate_checks = rollout_update_gates_v21(projection_audit, eligibility_config)
    failed = [name for name, passed in gate_checks.items() if not passed]
    if failed:
        raise ValueError("V23 rollout is ineligible for goal-directed update: " + ",".join(failed))
    actor_before = state_dict_sha256_v1(actor.state_dict())
    learning_batch, reward_audit = build_goal_directed_learning_batch_v23(
        rollout,
        reward_config,
    )
    if type(stratify_advantages_by_episode) is not bool:
        raise TypeError("V24 episode advantage stratification selector must be boolean")
    if type(align_cached_collection_numerics_v25) is not bool:
        raise TypeError("V25 cached collection numeric alignment selector must be boolean")
    if align_cached_collection_numerics_v25:
        learning_batch, alignment_audit = (
            align_cached_collection_log_probabilities_for_batched_ppo_v25(
                actor,
                learning_batch,
                ppo_config,
            )
        )
        reward_audit = replace(
            reward_audit,
            ppo_log_probability_alignment_mode=str(alignment_audit["mode"]),
            ppo_log_probability_alignment_maximum_absolute_delta=float(
                alignment_audit["maximum_absolute_delta"]
            ),
            ppo_log_probability_alignment_mean_absolute_delta=float(
                alignment_audit["mean_absolute_delta"]
            ),
            ppo_log_probability_alignment_tolerance=float(alignment_audit["tolerance"]),
        )
    advantage_group_ids: np.ndarray | None = None
    if stratify_advantages_by_episode:
        starts = learning_batch.episode_step_ids == 0
        advantage_group_ids = np.cumsum(starts, dtype=np.int64) - 1
        if (
            len(advantage_group_ids) == 0
            or advantage_group_ids[0] != 0
            or int(advantage_group_ids[-1]) + 1 != learning_batch.completed_episode_count
        ):
            raise RuntimeError("V24 could not reconstruct complete episode advantage groups")
    metrics, updated_optimizer = ppo_update_asymmetric_multiview_v1(
        actor,
        critic,
        learning_batch,
        ppo_config,
        optimizer=optimizer,
        advantage_group_ids=advantage_group_ids,
    )
    if state_dict_sha256_v1(actor.state_dict()) == actor_before:
        raise RuntimeError("V23 goal-directed PPO update did not change the actor")
    if metrics.approximate_kl > ppo_config.target_kl + 1.0e-6:
        raise RuntimeError("V23 goal-directed PPO update escaped its KL stop")
    return metrics, reward_audit, updated_optimizer


__all__ = [
    "FRONTIER_BALANCED_GOAL_DIRECTED_PPO_FORMAT_V24",
    "GOAL_DIRECTED_FEASIBLE_ON_POLICY_PPO_FORMAT_V23",
    "GoalDirectedFeasibleRewardConfigV23",
    "GoalDirectedRewardAuditV23",
    "LOG_PROBABILITY_ALIGNMENT_MODE_V25",
    "MAXIMUM_BATCHED_REPLAY_LOG_PROBABILITY_DELTA_V25",
    "_bounded_log_probability_alignment_v25",
    "align_cached_collection_log_probabilities_for_batched_ppo_v25",
    "build_goal_directed_learning_batch_v23",
    "compute_goal_directed_intrinsic_reward_v23",
    "goal_directed_feasible_on_policy_ppo_update_v23",
    "target_distance_progress_from_batch_v23",
]
