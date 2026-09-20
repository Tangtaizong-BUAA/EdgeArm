"""V15 shield-aware asymmetric DrQ-SAC.

V14 learned task value from sparse contact replay but could exploit actions that
the online IK/guard converted into latched holds.  V15 adds a privileged
training-only action-executability classifier.  The deployable actor is
unchanged and still consumes only causal RGB, joint, action, mask, task, and
autoregressive history.  No expert action or behavior-cloning objective is
introduced.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as functional

from .asymmetric_drq_sac_v1 import (
    AsymmetricDrQSACConfigV1,
    TwinPrivilegedActionCriticV1,
    _actor_inputs,
    _module_device,
    _sample_policy_action,
    _soft_update_target_v1,
    _torch_generator,
)
from .asymmetric_multiview_ppo_v1 import (
    POLICY_ACTION_DIM,
    SelectedViewRecurrentActorV1,
    autoregressive_action_distribution_v1,
)
from .ppo_utils_v1 import finite_module_parameters_v1, state_dict_sha256_v1
from .privileged_effect_state_v1 import PRIVILEGED_EFFECT_STATE_DIM
from .shield_aware_replay_v1 import ShieldAwareReplayBatchV1


ASYMMETRIC_SHIELD_AWARE_DRQ_SAC_FORMAT_V1 = (
    "edgearm-v15-asymmetric-drq-sac-shield-feasibility-v1"
)
ASYMMETRIC_SHIELD_AWARE_DRQ_SAC_CHECKPOINT_FORMAT_V1 = (
    "edgearm-v15-asymmetric-drq-sac-shield-feasibility-checkpoint-v1"
)
ACTION_FEASIBILITY_ARCHITECTURE_V1 = (
    "privileged-effect-action-executability-166-plus-3-128-128-v1"
)


@dataclass(frozen=True)
class AsymmetricShieldAwareDrQSACConfigV1(AsymmetricDrQSACConfigV1):
    feasibility_learning_rate: float = 3.0e-4
    actor_infeasibility_coefficient: float = 0.75

    def validate(self) -> None:
        super().validate()
        for name in (
            "feasibility_learning_rate",
            "actor_infeasibility_coefficient",
        ):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not np.isfinite(value)
                or value <= 0.0
            ):
                raise ValueError(f"shield-aware SAC {name} must be positive")


class PrivilegedActionFeasibilityNetworkV1(nn.Module):
    """Predict whether the exact online adapter executes a proposed action."""

    def __init__(self) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(PRIVILEGED_EFFECT_STATE_DIM + POLICY_ACTION_DIM, 128),
            nn.LayerNorm(128),
            nn.SiLU(),
            nn.Linear(128, 128),
            nn.LayerNorm(128),
            nn.SiLU(),
            nn.Linear(128, 1),
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        if state.ndim != 2 or state.shape[1] != PRIVILEGED_EFFECT_STATE_DIM:
            raise ValueError(
                "feasibility classifier privileged-state shape changed"
            )
        if action.shape != (state.shape[0], POLICY_ACTION_DIM):
            raise ValueError("feasibility classifier action shape changed")
        if not torch.isfinite(state).all() or not torch.isfinite(action).all():
            raise ValueError("feasibility classifier inputs must be finite")
        return self.network(torch.cat((state, action), dim=-1)).squeeze(-1)


@dataclass
class AsymmetricShieldAwareDrQSACBundleV1:
    actor: SelectedViewRecurrentActorV1
    critic: TwinPrivilegedActionCriticV1
    target_critic: TwinPrivilegedActionCriticV1
    feasibility: PrivilegedActionFeasibilityNetworkV1
    actor_optimizer: torch.optim.Optimizer
    critic_optimizer: torch.optim.Optimizer
    feasibility_optimizer: torch.optim.Optimizer
    initialization_seed: int
    actor_initial_state_sha256: str
    critic_initial_state_sha256: str
    feasibility_initial_state_sha256: str
    parent_actor_loaded: bool
    parent_critic_loaded: bool
    format: str = ASYMMETRIC_SHIELD_AWARE_DRQ_SAC_FORMAT_V1


@dataclass(frozen=True)
class AsymmetricShieldAwareDrQSACUpdateMetricsV1:
    update_index: int
    critic_loss: float
    bellman_q1_loss: float
    bellman_q2_loss: float
    conservative_q1_loss: float
    conservative_q2_loss: float
    feasibility_classifier_loss: float
    feasibility_classifier_accuracy: float
    feasibility_positive_accuracy: float
    feasibility_negative_accuracy: float
    source_action_feasible_fraction: float
    actor_loss: float
    entropy_objective: float
    actor_q_objective: float
    actor_infeasibility_penalty: float
    actor_predicted_feasible_probability: float
    visual_geometry_auxiliary_loss: float
    mean_target_q: float
    mean_data_q1: float
    mean_data_q2: float
    mean_policy_action_abs: float
    mean_source_n_step_reward: float
    mean_intrinsic_n_step_bonus: float
    maximum_actor_preclip_gradient_norm: float
    maximum_critic_preclip_gradient_norm: float
    maximum_feasibility_preclip_gradient_norm: float
    actor_state_sha256: str
    critic_state_sha256: str
    feasibility_state_sha256: str
    format: str = ASYMMETRIC_SHIELD_AWARE_DRQ_SAC_FORMAT_V1


def initialize_asymmetric_shield_aware_drq_sac_v1(
    seed: int,
    *,
    device: str | torch.device = "cpu",
    config: AsymmetricShieldAwareDrQSACConfigV1 | None = None,
    parent_actor_state_dict: dict[str, Any] | None = None,
    parent_critic_state_dict: dict[str, Any] | None = None,
    parent_target_critic_state_dict: dict[str, Any] | None = None,
) -> AsymmetricShieldAwareDrQSACBundleV1:
    if type(seed) is not int or seed < 0:
        raise ValueError("shield-aware SAC seed must be non-negative")
    selected = config or AsymmetricShieldAwareDrQSACConfigV1()
    selected.validate()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        actor = SelectedViewRecurrentActorV1(task_count=1)
        critic = TwinPrivilegedActionCriticV1()
        feasibility = PrivilegedActionFeasibilityNetworkV1()
    if parent_actor_state_dict is not None:
        actor.load_state_dict(parent_actor_state_dict, strict=True)
    if parent_critic_state_dict is not None:
        critic.load_state_dict(parent_critic_state_dict, strict=True)
    target = deepcopy(critic)
    if parent_target_critic_state_dict is not None:
        if parent_critic_state_dict is None:
            raise ValueError("target critic cannot load without its parent critic")
        target.load_state_dict(parent_target_critic_state_dict, strict=True)
    actor_hash = state_dict_sha256_v1(actor.state_dict())
    critic_hash = state_dict_sha256_v1(critic.state_dict())
    feasibility_hash = state_dict_sha256_v1(feasibility.state_dict())
    actor = actor.to(device)
    critic = critic.to(device)
    target = target.to(device)
    feasibility = feasibility.to(device)
    target.eval()
    for parameter in target.parameters():
        parameter.requires_grad_(False)
    return AsymmetricShieldAwareDrQSACBundleV1(
        actor=actor,
        critic=critic,
        target_critic=target,
        feasibility=feasibility,
        actor_optimizer=torch.optim.Adam(
            actor.parameters(), lr=selected.actor_learning_rate
        ),
        critic_optimizer=torch.optim.Adam(
            critic.parameters(), lr=selected.critic_learning_rate
        ),
        feasibility_optimizer=torch.optim.Adam(
            feasibility.parameters(), lr=selected.feasibility_learning_rate
        ),
        initialization_seed=seed,
        actor_initial_state_sha256=actor_hash,
        critic_initial_state_sha256=critic_hash,
        feasibility_initial_state_sha256=feasibility_hash,
        parent_actor_loaded=parent_actor_state_dict is not None,
        parent_critic_loaded=parent_critic_state_dict is not None,
    )


def _balanced_feasibility_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    positive_fraction = labels.mean().clamp(1.0e-4, 1.0 - 1.0e-4)
    positive_weight = (1.0 - positive_fraction) / positive_fraction
    return functional.binary_cross_entropy_with_logits(
        logits,
        labels,
        pos_weight=positive_weight.detach(),
    )


def _binary_accuracy(
    predictions: torch.Tensor,
    labels: torch.Tensor,
    selected: torch.Tensor | None = None,
) -> torch.Tensor:
    matches = predictions == labels.bool()
    if selected is None:
        return matches.float().mean()
    if not bool(selected.any()):
        return torch.ones((), device=matches.device)
    return matches[selected].float().mean()


def asymmetric_shield_aware_drq_sac_update_v1(
    bundle: AsymmetricShieldAwareDrQSACBundleV1,
    batch: ShieldAwareReplayBatchV1,
    config: AsymmetricShieldAwareDrQSACConfigV1,
    *,
    update_index: int,
    seed: int,
) -> AsymmetricShieldAwareDrQSACUpdateMetricsV1:
    config.validate()
    batch.validate()
    if type(update_index) is not int or update_index < 1:
        raise ValueError("shield-aware SAC update index must be positive")
    if type(seed) is not int or seed < 0:
        raise ValueError("shield-aware SAC update seed must be non-negative")
    if bundle.format != ASYMMETRIC_SHIELD_AWARE_DRQ_SAC_FORMAT_V1:
        raise ValueError("shield-aware SAC bundle identity changed")
    device = _module_device(bundle.actor)
    if any(
        _module_device(module) != device
        for module in (
            bundle.critic,
            bundle.target_critic,
            bundle.feasibility,
        )
    ):
        raise ValueError("shield-aware SAC modules must share one device")

    base = batch.base
    current_inputs = _actor_inputs(
        base,
        device,
        bootstrap=False,
        shift_padding=config.random_shift_padding_pixels,
        shift_seed=seed ^ 0x1234,
    )
    bootstrap_inputs = _actor_inputs(
        base,
        device,
        bootstrap=True,
        shift_padding=config.random_shift_padding_pixels,
        shift_seed=seed ^ 0x5678,
    )
    state = torch.from_numpy(base.privileged_state).to(device)
    bootstrap_state = torch.from_numpy(base.bootstrap_privileged_state).to(
        device
    )
    replay_action = torch.from_numpy(base.replay_action).to(device)
    reward = torch.from_numpy(base.n_step_reward).to(device)
    discount = torch.from_numpy(base.bootstrap_discount).to(device)
    weights = torch.from_numpy(base.importance_weight).to(device)
    previous_pre_tanh = torch.from_numpy(
        base.previous_policy_pre_tanh
    ).to(device)
    bootstrap_previous_pre_tanh = torch.from_numpy(
        base.bootstrap_previous_policy_pre_tanh
    ).to(device)
    geometry_target = torch.from_numpy(base.visual_geometry_target).to(device)
    feasibility_label = torch.from_numpy(
        batch.source_action_feasible.astype(np.float32)
    ).to(device)

    bundle.actor.train()
    bundle.critic.train()
    bundle.feasibility.train()
    with torch.no_grad():
        next_base = bundle.actor.distribution(*bootstrap_inputs)
        next_distribution = autoregressive_action_distribution_v1(
            next_base,
            bootstrap_previous_pre_tanh,
            config.action_autoregressive_rho,
        )
        next_action, _, next_log_probability = _sample_policy_action(
            next_distribution,
            generator=_torch_generator(device, seed ^ 0xA511),
        )
        target_q1, target_q2 = bundle.target_critic(
            bootstrap_state, next_action
        )
        target_q = reward + discount * (
            torch.minimum(target_q1, target_q2)
            - config.entropy_temperature * next_log_probability
        )

    data_q1, data_q2 = bundle.critic(state, replay_action)
    bellman_q1 = functional.smooth_l1_loss(
        data_q1, target_q, reduction="none"
    )
    bellman_q2 = functional.smooth_l1_loss(
        data_q2, target_q, reduction="none"
    )
    weighted_q1 = (weights * bellman_q1).mean()
    weighted_q2 = (weights * bellman_q2).mean()

    random_count = config.conservative_random_action_count
    random_actions = (
        2.0
        * torch.rand(
            len(reward),
            random_count,
            POLICY_ACTION_DIM,
            device=device,
            generator=_torch_generator(device, seed ^ 0xC011),
        )
        - 1.0
    )
    repeated_state = state.unsqueeze(1).expand(-1, random_count, -1).reshape(
        -1, PRIVILEGED_EFFECT_STATE_DIM
    )
    random_q1, random_q2 = bundle.critic(
        repeated_state,
        random_actions.reshape(-1, POLICY_ACTION_DIM),
    )
    random_q1 = random_q1.reshape(len(reward), random_count)
    random_q2 = random_q2.reshape(len(reward), random_count)
    with torch.no_grad():
        conservative_base = bundle.actor.distribution(*current_inputs)
        conservative_distribution = autoregressive_action_distribution_v1(
            conservative_base,
            previous_pre_tanh,
            config.action_autoregressive_rho,
        )
        conservative_action, _, _ = _sample_policy_action(
            conservative_distribution,
            generator=_torch_generator(device, seed ^ 0xC012),
        )
    policy_q1, policy_q2 = bundle.critic(state, conservative_action)
    conservative_q1 = (
        torch.logsumexp(
            torch.cat((random_q1, policy_q1.unsqueeze(1)), dim=1),
            dim=1,
        )
        - data_q1
    ).mean()
    conservative_q2 = (
        torch.logsumexp(
            torch.cat((random_q2, policy_q2.unsqueeze(1)), dim=1),
            dim=1,
        )
        - data_q2
    ).mean()
    critic_loss = (
        weighted_q1
        + weighted_q2
        + config.conservative_q_coefficient
        * (conservative_q1 + conservative_q2)
    )
    bundle.critic_optimizer.zero_grad(set_to_none=True)
    critic_loss.backward()
    critic_gradient_norm = nn.utils.clip_grad_norm_(
        bundle.critic.parameters(), config.maximum_critic_gradient_norm
    )
    bundle.critic_optimizer.step()

    feasibility_logit = bundle.feasibility(state, replay_action)
    feasibility_loss = _balanced_feasibility_loss(
        feasibility_logit,
        feasibility_label,
    )
    bundle.feasibility_optimizer.zero_grad(set_to_none=True)
    feasibility_loss.backward()
    feasibility_gradient_norm = nn.utils.clip_grad_norm_(
        bundle.feasibility.parameters(),
        config.maximum_critic_gradient_norm,
    )
    bundle.feasibility_optimizer.step()

    for module in (bundle.critic, bundle.feasibility):
        for parameter in module.parameters():
            parameter.requires_grad_(False)
    actor_base, geometry_prediction = (
        bundle.actor.distribution_and_visual_geometry(*current_inputs)
    )
    actor_distribution = autoregressive_action_distribution_v1(
        actor_base,
        previous_pre_tanh,
        config.action_autoregressive_rho,
    )
    actor_action, _, actor_log_probability = _sample_policy_action(
        actor_distribution,
        generator=_torch_generator(device, seed ^ 0xAC70),
    )
    actor_q1, actor_q2 = bundle.critic(state, actor_action)
    actor_q = torch.minimum(actor_q1, actor_q2)
    actor_feasibility_logit = bundle.feasibility(state, actor_action)
    entropy_objective = config.entropy_temperature * actor_log_probability
    actor_q_objective = -actor_q
    actor_infeasibility = functional.softplus(-actor_feasibility_logit)
    policy_loss = (
        weights
        * (
            entropy_objective
            + actor_q_objective
            + config.actor_infeasibility_coefficient * actor_infeasibility
        )
    ).mean()
    geometry_loss = functional.smooth_l1_loss(
        geometry_prediction,
        geometry_target,
    )
    actor_loss = policy_loss + (
        config.visual_geometry_auxiliary_coefficient * geometry_loss
    )
    bundle.actor_optimizer.zero_grad(set_to_none=True)
    actor_loss.backward()
    actor_gradient_norm = nn.utils.clip_grad_norm_(
        bundle.actor.parameters(), config.maximum_actor_gradient_norm
    )
    bundle.actor_optimizer.step()
    for module in (bundle.critic, bundle.feasibility):
        for parameter in module.parameters():
            parameter.requires_grad_(True)

    _soft_update_target_v1(
        bundle.critic,
        bundle.target_critic,
        config.target_critic_tau,
    )
    if not all(
        finite_module_parameters_v1(module)
        for module in (
            bundle.actor,
            bundle.critic,
            bundle.target_critic,
            bundle.feasibility,
        )
    ):
        raise RuntimeError("shield-aware SAC update produced non-finite state")

    with torch.no_grad():
        predicted = feasibility_logit >= 0.0
        positive = feasibility_label > 0.5
        negative = ~positive
        accuracy = _binary_accuracy(predicted, positive)
        positive_accuracy = _binary_accuracy(predicted, positive, positive)
        negative_accuracy = _binary_accuracy(predicted, positive, negative)
        actor_feasible_probability = torch.sigmoid(
            actor_feasibility_logit
        ).mean()
    numeric = np.asarray(
        [
            critic_loss.item(),
            weighted_q1.item(),
            weighted_q2.item(),
            conservative_q1.item(),
            conservative_q2.item(),
            feasibility_loss.item(),
            accuracy.item(),
            positive_accuracy.item(),
            negative_accuracy.item(),
            feasibility_label.mean().item(),
            actor_loss.item(),
            entropy_objective.mean().item(),
            actor_q_objective.mean().item(),
            actor_infeasibility.mean().item(),
            actor_feasible_probability.item(),
            geometry_loss.item(),
            target_q.mean().item(),
            data_q1.mean().item(),
            data_q2.mean().item(),
            actor_action.abs().mean().item(),
            actor_gradient_norm.item(),
            critic_gradient_norm.item(),
            feasibility_gradient_norm.item(),
        ],
        dtype=np.float64,
    )
    if not np.all(np.isfinite(numeric)):
        raise RuntimeError("shield-aware SAC metrics are non-finite")
    return AsymmetricShieldAwareDrQSACUpdateMetricsV1(
        update_index=update_index,
        critic_loss=float(numeric[0]),
        bellman_q1_loss=float(numeric[1]),
        bellman_q2_loss=float(numeric[2]),
        conservative_q1_loss=float(numeric[3]),
        conservative_q2_loss=float(numeric[4]),
        feasibility_classifier_loss=float(numeric[5]),
        feasibility_classifier_accuracy=float(numeric[6]),
        feasibility_positive_accuracy=float(numeric[7]),
        feasibility_negative_accuracy=float(numeric[8]),
        source_action_feasible_fraction=float(numeric[9]),
        actor_loss=float(numeric[10]),
        entropy_objective=float(numeric[11]),
        actor_q_objective=float(numeric[12]),
        actor_infeasibility_penalty=float(numeric[13]),
        actor_predicted_feasible_probability=float(numeric[14]),
        visual_geometry_auxiliary_loss=float(numeric[15]),
        mean_target_q=float(numeric[16]),
        mean_data_q1=float(numeric[17]),
        mean_data_q2=float(numeric[18]),
        mean_policy_action_abs=float(numeric[19]),
        mean_source_n_step_reward=float(base.source_n_step_reward.mean()),
        mean_intrinsic_n_step_bonus=float(base.intrinsic_n_step_bonus.mean()),
        maximum_actor_preclip_gradient_norm=float(numeric[20]),
        maximum_critic_preclip_gradient_norm=float(numeric[21]),
        maximum_feasibility_preclip_gradient_norm=float(numeric[22]),
        actor_state_sha256=state_dict_sha256_v1(bundle.actor.state_dict()),
        critic_state_sha256=state_dict_sha256_v1(bundle.critic.state_dict()),
        feasibility_state_sha256=state_dict_sha256_v1(
            bundle.feasibility.state_dict()
        ),
    )


__all__ = [
    "ACTION_FEASIBILITY_ARCHITECTURE_V1",
    "ASYMMETRIC_SHIELD_AWARE_DRQ_SAC_CHECKPOINT_FORMAT_V1",
    "ASYMMETRIC_SHIELD_AWARE_DRQ_SAC_FORMAT_V1",
    "AsymmetricShieldAwareDrQSACBundleV1",
    "AsymmetricShieldAwareDrQSACConfigV1",
    "AsymmetricShieldAwareDrQSACUpdateMetricsV1",
    "PrivilegedActionFeasibilityNetworkV1",
    "asymmetric_shield_aware_drq_sac_update_v1",
    "initialize_asymmetric_shield_aware_drq_sac_v1",
]
