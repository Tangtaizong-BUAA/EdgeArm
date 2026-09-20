"""Asymmetric DrQ-style SAC updates for V13 contact-prioritized replay.

The deployable actor remains the causal four-view RGB/joint/action policy from
V13.  Twin Q critics receive simulator privileged state plus the proposed
task-frame action during training only.  Random translations are shared over
time for each camera view, preserving motion while applying DrQ-style visual
augmentation.  A small conservative-Q term limits offline extrapolation; no
expert action or behavior-cloning loss is used.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as functional

from .asymmetric_multiview_ppo_v1 import (
    POLICY_ACTION_DIM,
    SelectedViewRecurrentActorV1,
    autoregressive_action_distribution_v1,
)
from .contact_prioritized_replay_v1 import (
    ContactPrioritizedReplayBatchV1,
)
from .ppo_utils_v1 import (
    finite_module_parameters_v1,
    squashed_gaussian_log_prob_v1,
    state_dict_sha256_v1,
)
from .privileged_effect_state_v1 import PRIVILEGED_EFFECT_STATE_DIM


ASYMMETRIC_DRQ_SAC_FORMAT_V1 = (
    "edgearm-v14-asymmetric-four-view-drq-sac-contact-replay-v1"
)
ASYMMETRIC_DRQ_SAC_CHECKPOINT_FORMAT_V1 = (
    "edgearm-v14-asymmetric-four-view-drq-sac-checkpoint-v1"
)
ASYMMETRIC_DRQ_SAC_CRITIC_ARCHITECTURE_V1 = (
    "twin-privileged-effect-action-q-166-256-256-v1"
)


@dataclass(frozen=True)
class AsymmetricDrQSACConfigV1:
    actor_learning_rate: float = 1.0e-4
    critic_learning_rate: float = 3.0e-4
    entropy_temperature: float = 0.05
    target_critic_tau: float = 0.01
    action_autoregressive_rho: float = 0.8
    random_shift_padding_pixels: int = 4
    visual_geometry_auxiliary_coefficient: float = 0.20
    conservative_q_coefficient: float = 0.10
    conservative_random_action_count: int = 4
    maximum_actor_gradient_norm: float = 10.0
    maximum_critic_gradient_norm: float = 10.0

    def validate(self) -> None:
        for name in (
            "actor_learning_rate",
            "critic_learning_rate",
            "entropy_temperature",
            "target_critic_tau",
            "action_autoregressive_rho",
            "visual_geometry_auxiliary_coefficient",
            "conservative_q_coefficient",
            "maximum_actor_gradient_norm",
            "maximum_critic_gradient_norm",
        ):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not np.isfinite(value)
            ):
                raise ValueError(f"DrQ-SAC {name} must be finite numeric")
        if self.actor_learning_rate <= 0.0 or self.critic_learning_rate <= 0.0:
            raise ValueError("DrQ-SAC learning rates must be positive")
        if self.entropy_temperature < 0.0:
            raise ValueError("DrQ-SAC entropy temperature must be non-negative")
        if not 0.0 < self.target_critic_tau <= 1.0:
            raise ValueError("DrQ-SAC target tau must be in (0,1]")
        if not 0.0 <= self.action_autoregressive_rho < 0.99:
            raise ValueError("DrQ-SAC AR rho must be in [0,0.99)")
        if (
            type(self.random_shift_padding_pixels) is not int
            or self.random_shift_padding_pixels < 0
        ):
            raise ValueError("DrQ-SAC random-shift padding must be non-negative")
        if (
            type(self.conservative_random_action_count) is not int
            or self.conservative_random_action_count < 1
        ):
            raise ValueError("DrQ-SAC conservative action count must be positive")
        if (
            self.visual_geometry_auxiliary_coefficient < 0.0
            or self.conservative_q_coefficient < 0.0
            or self.maximum_actor_gradient_norm <= 0.0
            or self.maximum_critic_gradient_norm <= 0.0
        ):
            raise ValueError("DrQ-SAC loss/gradient coefficients are invalid")


class PrivilegedActionQNetworkV1(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(PRIVILEGED_EFFECT_STATE_DIM + POLICY_ACTION_DIM, 256),
            nn.LayerNorm(256),
            nn.SiLU(),
            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.SiLU(),
            nn.Linear(256, 1),
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        if state.ndim != 2 or state.shape[1] != PRIVILEGED_EFFECT_STATE_DIM:
            raise ValueError(
                f"Q critic requires [B,{PRIVILEGED_EFFECT_STATE_DIM}] state"
            )
        if action.shape != (state.shape[0], POLICY_ACTION_DIM):
            raise ValueError("Q critic action shape changed")
        if not torch.isfinite(state).all() or not torch.isfinite(action).all():
            raise ValueError("Q critic inputs must be finite")
        return self.network(torch.cat((state, action), dim=-1)).squeeze(-1)


class TwinPrivilegedActionCriticV1(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.q1 = PrivilegedActionQNetworkV1()
        self.q2 = PrivilegedActionQNetworkV1()

    def forward(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.q1(state, action), self.q2(state, action)


@dataclass
class AsymmetricDrQSACBundleV1:
    actor: SelectedViewRecurrentActorV1
    critic: TwinPrivilegedActionCriticV1
    target_critic: TwinPrivilegedActionCriticV1
    actor_optimizer: torch.optim.Optimizer
    critic_optimizer: torch.optim.Optimizer
    initialization_seed: int
    actor_initial_state_sha256: str
    critic_initial_state_sha256: str
    parent_actor_loaded: bool
    format: str = ASYMMETRIC_DRQ_SAC_FORMAT_V1


@dataclass(frozen=True)
class AsymmetricDrQSACUpdateMetricsV1:
    update_index: int
    critic_loss: float
    bellman_q1_loss: float
    bellman_q2_loss: float
    conservative_q1_loss: float
    conservative_q2_loss: float
    actor_loss: float
    entropy_objective: float
    actor_q_objective: float
    visual_geometry_auxiliary_loss: float
    mean_target_q: float
    mean_data_q1: float
    mean_data_q2: float
    mean_policy_action_abs: float
    mean_source_n_step_reward: float
    mean_intrinsic_n_step_bonus: float
    maximum_actor_preclip_gradient_norm: float
    maximum_critic_preclip_gradient_norm: float
    actor_state_sha256: str
    critic_state_sha256: str
    format: str = ASYMMETRIC_DRQ_SAC_FORMAT_V1


def initialize_asymmetric_drq_sac_v1(
    seed: int,
    *,
    device: str | torch.device = "cpu",
    config: AsymmetricDrQSACConfigV1 | None = None,
    parent_actor_state_dict: dict[str, Any] | None = None,
) -> AsymmetricDrQSACBundleV1:
    if type(seed) is not int or seed < 0:
        raise ValueError("DrQ-SAC initialization seed must be non-negative")
    selected_config = config or AsymmetricDrQSACConfigV1()
    selected_config.validate()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        actor = SelectedViewRecurrentActorV1(task_count=1)
        critic = TwinPrivilegedActionCriticV1()
    parent_loaded = parent_actor_state_dict is not None
    if parent_actor_state_dict is not None:
        actor.load_state_dict(parent_actor_state_dict, strict=True)
    actor_hash = state_dict_sha256_v1(actor.state_dict())
    critic_hash = state_dict_sha256_v1(critic.state_dict())
    actor = actor.to(device)
    critic = critic.to(device)
    target = deepcopy(critic).to(device)
    target.eval()
    for parameter in target.parameters():
        parameter.requires_grad_(False)
    actor_optimizer = torch.optim.Adam(
        actor.parameters(), lr=selected_config.actor_learning_rate
    )
    critic_optimizer = torch.optim.Adam(
        critic.parameters(), lr=selected_config.critic_learning_rate
    )
    return AsymmetricDrQSACBundleV1(
        actor=actor,
        critic=critic,
        target_critic=target,
        actor_optimizer=actor_optimizer,
        critic_optimizer=critic_optimizer,
        initialization_seed=seed,
        actor_initial_state_sha256=actor_hash,
        critic_initial_state_sha256=critic_hash,
        parent_actor_loaded=parent_loaded,
    )


def causal_multiview_random_shift_v1(
    rgb_history: torch.Tensor,
    *,
    padding_pixels: int,
    seed: int,
) -> torch.Tensor:
    """Apply one integer crop per sample/view, shared across causal time."""

    if rgb_history.ndim != 6 or rgb_history.shape[-1] != 3:
        raise ValueError("DrQ random shift requires [B,T,V,H,W,3]")
    if type(padding_pixels) is not int or padding_pixels < 0:
        raise ValueError("DrQ random-shift padding must be non-negative")
    if type(seed) is not int or seed < 0:
        raise ValueError("DrQ random-shift seed must be non-negative")
    if padding_pixels == 0:
        return rgb_history.clone()
    batch, steps, views, height, width, channels = rgb_history.shape
    rng = np.random.default_rng(seed)
    offsets = rng.integers(
        0,
        2 * padding_pixels + 1,
        size=(batch, views, 2),
        dtype=np.int64,
    )
    result = torch.empty_like(rgb_history)
    for batch_index in range(batch):
        for view_index in range(views):
            source = rgb_history[batch_index, :, view_index].permute(
                0, 3, 1, 2
            )
            padded = functional.pad(
                source,
                (
                    padding_pixels,
                    padding_pixels,
                    padding_pixels,
                    padding_pixels,
                ),
                mode="replicate",
            )
            y_offset = int(offsets[batch_index, view_index, 0])
            x_offset = int(offsets[batch_index, view_index, 1])
            crop = padded[
                :,
                :,
                y_offset : y_offset + height,
                x_offset : x_offset + width,
            ]
            result[batch_index, :, view_index] = crop.permute(0, 2, 3, 1)
    if result.shape != (batch, steps, views, height, width, channels):
        raise RuntimeError("DrQ random shift changed the causal image shape")
    return result


def _module_device(module: nn.Module) -> torch.device:
    return next(module.parameters()).device


def _torch_generator(device: torch.device, seed: int) -> torch.Generator | None:
    if device.type == "mps":
        torch.manual_seed(seed)
        torch.mps.manual_seed(seed)
        return None
    generator = torch.Generator(device=device.type)
    generator.manual_seed(seed)
    return generator


def _sample_policy_action(
    distribution: torch.distributions.Normal,
    *,
    generator: torch.Generator | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    epsilon = torch.randn(
        distribution.loc.shape,
        dtype=distribution.loc.dtype,
        device=distribution.loc.device,
        generator=generator,
    )
    pre_tanh = distribution.loc + distribution.scale * epsilon
    action = torch.tanh(pre_tanh)
    log_probability = squashed_gaussian_log_prob_v1(distribution, pre_tanh)
    return action, pre_tanh, log_probability


def _actor_inputs(
    batch: ContactPrioritizedReplayBatchV1,
    device: torch.device,
    *,
    bootstrap: bool,
    shift_padding: int,
    shift_seed: int,
) -> tuple[torch.Tensor, ...]:
    if bootstrap:
        rgb = batch.bootstrap_rgb_history
        joints = batch.bootstrap_joint_history
        actions = batch.bootstrap_action_history
        history_valid = batch.bootstrap_history_valid
        view_valid = batch.bootstrap_view_history_valid
        tasks = batch.bootstrap_task_ids
    else:
        rgb = batch.rgb_history
        joints = batch.joint_history
        actions = batch.action_history
        history_valid = batch.history_valid
        view_valid = batch.view_history_valid
        tasks = batch.task_ids
    shifted = causal_multiview_random_shift_v1(
        torch.from_numpy(rgb).to(device=device, dtype=torch.float32),
        padding_pixels=shift_padding,
        seed=shift_seed,
    )
    return (
        shifted,
        torch.from_numpy(joints).to(device),
        torch.from_numpy(actions).to(device),
        torch.from_numpy(history_valid).to(device),
        torch.from_numpy(view_valid).to(device),
        torch.from_numpy(tasks).to(device=device, dtype=torch.long),
    )


def _soft_update_target_v1(
    source: nn.Module,
    target: nn.Module,
    tau: float,
) -> None:
    with torch.no_grad():
        for source_parameter, target_parameter in zip(
            source.parameters(), target.parameters()
        ):
            target_parameter.mul_(1.0 - tau).add_(
                source_parameter, alpha=tau
            )


def asymmetric_drq_sac_update_v1(
    bundle: AsymmetricDrQSACBundleV1,
    batch: ContactPrioritizedReplayBatchV1,
    config: AsymmetricDrQSACConfigV1,
    *,
    update_index: int,
    seed: int,
) -> AsymmetricDrQSACUpdateMetricsV1:
    config.validate()
    batch.validate()
    if type(update_index) is not int or update_index < 1:
        raise ValueError("DrQ-SAC update_index must be positive")
    if type(seed) is not int or seed < 0:
        raise ValueError("DrQ-SAC update seed must be non-negative")
    if bundle.format != ASYMMETRIC_DRQ_SAC_FORMAT_V1:
        raise ValueError("DrQ-SAC bundle identity changed")
    device = _module_device(bundle.actor)
    if (
        _module_device(bundle.critic) != device
        or _module_device(bundle.target_critic) != device
    ):
        raise ValueError("DrQ-SAC modules must share one device")

    current_inputs = _actor_inputs(
        batch,
        device,
        bootstrap=False,
        shift_padding=config.random_shift_padding_pixels,
        shift_seed=seed ^ 0x1234,
    )
    bootstrap_inputs = _actor_inputs(
        batch,
        device,
        bootstrap=True,
        shift_padding=config.random_shift_padding_pixels,
        shift_seed=seed ^ 0x5678,
    )
    state = torch.from_numpy(batch.privileged_state).to(device)
    bootstrap_state = torch.from_numpy(batch.bootstrap_privileged_state).to(
        device
    )
    replay_action = torch.from_numpy(batch.replay_action).to(device)
    reward = torch.from_numpy(batch.n_step_reward).to(device)
    discount = torch.from_numpy(batch.bootstrap_discount).to(device)
    weights = torch.from_numpy(batch.importance_weight).to(device)
    previous_pre_tanh = torch.from_numpy(
        batch.previous_policy_pre_tanh
    ).to(device)
    bootstrap_previous_pre_tanh = torch.from_numpy(
        batch.bootstrap_previous_policy_pre_tanh
    ).to(device)
    geometry_target = torch.from_numpy(batch.visual_geometry_target).to(device)

    bundle.actor.train()
    bundle.critic.train()
    with torch.no_grad():
        next_base = bundle.actor.distribution(*bootstrap_inputs)
        next_distribution = autoregressive_action_distribution_v1(
            next_base,
            bootstrap_previous_pre_tanh,
            config.action_autoregressive_rho,
        )
        next_action, _next_pre_tanh, next_log_probability = _sample_policy_action(
            next_distribution,
            generator=_torch_generator(device, seed ^ 0xA511),
        )
        target_q1, target_q2 = bundle.target_critic(
            bootstrap_state, next_action
        )
        entropy_adjusted_target = torch.minimum(target_q1, target_q2) - (
            config.entropy_temperature * next_log_probability
        )
        target_q = reward + discount * entropy_adjusted_target

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
    random_generator = _torch_generator(device, seed ^ 0xC011)
    random_actions = (
        2.0
        * torch.rand(
            len(reward),
            random_count,
            POLICY_ACTION_DIM,
            device=device,
            generator=random_generator,
        )
        - 1.0
    )
    repeated_state = state.unsqueeze(1).expand(-1, random_count, -1).reshape(
        -1, PRIVILEGED_EFFECT_STATE_DIM
    )
    flat_random = random_actions.reshape(-1, POLICY_ACTION_DIM)
    random_q1, random_q2 = bundle.critic(repeated_state, flat_random)
    random_q1 = random_q1.reshape(len(reward), random_count)
    random_q2 = random_q2.reshape(len(reward), random_count)
    with torch.no_grad():
        current_base = bundle.actor.distribution(*current_inputs)
        current_distribution = autoregressive_action_distribution_v1(
            current_base,
            previous_pre_tanh,
            config.action_autoregressive_rho,
        )
        conservative_policy_action, _, _ = _sample_policy_action(
            current_distribution,
            generator=_torch_generator(device, seed ^ 0xC012),
        )
    policy_q1, policy_q2 = bundle.critic(state, conservative_policy_action)
    conservative_q1 = (
        torch.logsumexp(
            torch.cat((random_q1, policy_q1.unsqueeze(1)), dim=1), dim=1
        )
        - data_q1
    ).mean()
    conservative_q2 = (
        torch.logsumexp(
            torch.cat((random_q2, policy_q2.unsqueeze(1)), dim=1), dim=1
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

    for parameter in bundle.critic.parameters():
        parameter.requires_grad_(False)
    actor_base, geometry_prediction = (
        bundle.actor.distribution_and_visual_geometry(*current_inputs)
    )
    actor_distribution = autoregressive_action_distribution_v1(
        actor_base,
        previous_pre_tanh,
        config.action_autoregressive_rho,
    )
    actor_action, _actor_pre_tanh, actor_log_probability = _sample_policy_action(
        actor_distribution,
        generator=_torch_generator(device, seed ^ 0xAC70),
    )
    actor_q1, actor_q2 = bundle.critic(state, actor_action)
    actor_q = torch.minimum(actor_q1, actor_q2)
    entropy_objective = config.entropy_temperature * actor_log_probability
    actor_q_objective = -actor_q
    policy_loss = (weights * (entropy_objective + actor_q_objective)).mean()
    geometry_loss = functional.smooth_l1_loss(
        geometry_prediction, geometry_target
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
    for parameter in bundle.critic.parameters():
        parameter.requires_grad_(True)

    _soft_update_target_v1(
        bundle.critic, bundle.target_critic, config.target_critic_tau
    )
    if not (
        finite_module_parameters_v1(bundle.actor)
        and finite_module_parameters_v1(bundle.critic)
        and finite_module_parameters_v1(bundle.target_critic)
    ):
        raise RuntimeError("DrQ-SAC update produced non-finite parameters")
    numeric = np.asarray(
        [
            critic_loss.item(),
            weighted_q1.item(),
            weighted_q2.item(),
            conservative_q1.item(),
            conservative_q2.item(),
            actor_loss.item(),
            entropy_objective.mean().item(),
            actor_q_objective.mean().item(),
            geometry_loss.item(),
            target_q.mean().item(),
            data_q1.mean().item(),
            data_q2.mean().item(),
            actor_action.abs().mean().item(),
            actor_gradient_norm.item(),
            critic_gradient_norm.item(),
        ],
        dtype=np.float64,
    )
    if not np.all(np.isfinite(numeric)):
        raise RuntimeError("DrQ-SAC metrics are non-finite")
    return AsymmetricDrQSACUpdateMetricsV1(
        update_index=update_index,
        critic_loss=float(numeric[0]),
        bellman_q1_loss=float(numeric[1]),
        bellman_q2_loss=float(numeric[2]),
        conservative_q1_loss=float(numeric[3]),
        conservative_q2_loss=float(numeric[4]),
        actor_loss=float(numeric[5]),
        entropy_objective=float(numeric[6]),
        actor_q_objective=float(numeric[7]),
        visual_geometry_auxiliary_loss=float(numeric[8]),
        mean_target_q=float(numeric[9]),
        mean_data_q1=float(numeric[10]),
        mean_data_q2=float(numeric[11]),
        mean_policy_action_abs=float(numeric[12]),
        mean_source_n_step_reward=float(batch.source_n_step_reward.mean()),
        mean_intrinsic_n_step_bonus=float(batch.intrinsic_n_step_bonus.mean()),
        maximum_actor_preclip_gradient_norm=float(numeric[13]),
        maximum_critic_preclip_gradient_norm=float(numeric[14]),
        actor_state_sha256=state_dict_sha256_v1(bundle.actor.state_dict()),
        critic_state_sha256=state_dict_sha256_v1(bundle.critic.state_dict()),
    )


__all__ = [
    "ASYMMETRIC_DRQ_SAC_CHECKPOINT_FORMAT_V1",
    "ASYMMETRIC_DRQ_SAC_CRITIC_ARCHITECTURE_V1",
    "ASYMMETRIC_DRQ_SAC_FORMAT_V1",
    "AsymmetricDrQSACBundleV1",
    "AsymmetricDrQSACConfigV1",
    "AsymmetricDrQSACUpdateMetricsV1",
    "PrivilegedActionQNetworkV1",
    "TwinPrivilegedActionCriticV1",
    "asymmetric_drq_sac_update_v1",
    "causal_multiview_random_shift_v1",
    "initialize_asymmetric_drq_sac_v1",
]
