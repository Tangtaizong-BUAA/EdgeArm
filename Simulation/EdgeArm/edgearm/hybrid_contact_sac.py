"""Conservative discrete-contact-mode actor-critic for staged pushing.

This module is deliberately independent of MuJoCo.  It provides the frozen
learning core for the replacement EdgeArm data generator:

* one learned categorical contact mode with a continuous task-frame residual;
* exact expectation over every discrete mode in the critic target;
* immutable successful prior replay plus online replay sampled exactly 50/50;
* critics trained on the policy proposal, never on a shield-substituted action;
* explicit terminal surrogate transitions when a safety authority intervenes.
* delayed, low-frequency advantage-weighted actor regression, rather than an
  unconstrained SAC actor step that can exploit a young contact critic.

The module does not grant production, export, or ACT admission.  Those gates
belong to the evaluation/data-product layer and remain false until a learned
policy passes exact-Home multi-seed evaluation.
"""

from __future__ import annotations

import copy
import hashlib
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


HYBRID_CONTACT_SAC_FORMAT = "edgearm-staged-hybrid-contact-awac-1.0.0"
HYBRID_REPLAY_FORMAT = "edgearm-symmetric-prior-online-replay-1.1.0"
LEGACY_HYBRID_REPLAY_FORMAT = "edgearm-symmetric-prior-online-replay-1.0.0"
TASK_ACTION_DIM = 3
CONTACT_MODE_NAMES = (
    "approach",
    "stick_push",
    "slide_left",
    "slide_right",
    "separate_recontact",
    "settle_hold",
)
CONTACT_MODE_COUNT = len(CONTACT_MODE_NAMES)

# These are exploration centres, not a route or an expert.  Every centre is a
# normalized local [forward, lateral, vertical] task-frame action.  The actor
# learns both the mode probability and a residual around each centre.
DEFAULT_MODE_ACTION_CENTRES = np.asarray(
    [
        [0.55, 0.00, -0.05],
        [0.65, 0.00, 0.00],
        [0.35, 0.45, 0.00],
        [0.35, -0.45, 0.00],
        [-0.35, 0.00, 0.20],
        [0.00, 0.00, 0.00],
    ],
    dtype=np.float32,
)


@dataclass(frozen=True)
class HybridContactSACConfig:
    observation_dim: int
    mode_mask_start: int | None = None
    hidden_dim: int = 256
    batch_size: int = 256
    prior_capacity: int = 200_000
    online_capacity: int = 500_000
    gamma: float = 0.995
    target_tau: float = 0.005
    actor_learning_rate: float = 3.0e-4
    online_actor_learning_rate: float = 3.0e-5
    critic_learning_rate: float = 3.0e-4
    continuous_entropy_temperature: float = 0.05
    mode_entropy_temperature: float = 0.02
    online_mode_exploration_fraction: float = 0.0
    online_sample_actor_categorical: bool = True
    online_sample_continuous_action: bool = True
    residual_scale: float = 0.45
    exploration_log_std: float = -2.75
    maximum_gradient_norm: float = 10.0
    prior_fraction: float = 0.50
    actor_update_delay: int = 256
    actor_update_interval: int = 4
    advantage_temperature: float = 1.0
    minimum_online_advantage: float = 0.05
    maximum_advantage_weight: float = 5.0
    minimum_online_actor_samples: int = 8
    minimum_online_actor_evidence_per_mode: int = 1
    minimum_online_actor_recovery_modes: int = 0
    minimum_online_actor_effective_sample_size: float = 4.0
    prior_behavior_cloning_coefficient: float = 1.0
    online_advantage_regression_coefficient: float = 0.25
    maximum_actor_batch_action_delta: float = 0.02
    intervention_terminal_penalty: float = 12.0

    def validate(self) -> None:
        positive_ints = (
            self.observation_dim,
            self.hidden_dim,
            self.batch_size,
            self.prior_capacity,
            self.online_capacity,
            self.actor_update_interval,
            self.minimum_online_actor_samples,
            self.minimum_online_actor_evidence_per_mode,
        )
        if any(type(value) is not int or value < 1 for value in positive_ints):
            raise ValueError("hybrid SAC integer configuration is invalid")
        if self.batch_size % 2 != 0:
            raise ValueError("symmetric replay requires an even batch size")
        if self.mode_mask_start is not None and (
            type(self.mode_mask_start) is not int
            or self.mode_mask_start < 0
            or self.mode_mask_start + CONTACT_MODE_COUNT > self.observation_dim
        ):
            raise ValueError("hybrid SAC mode-mask slice is invalid")
        if type(self.actor_update_delay) is not int or self.actor_update_delay < 0:
            raise ValueError("hybrid actor update delay is invalid")
        if (
            type(self.minimum_online_actor_recovery_modes) is not int
            or not 0 <= self.minimum_online_actor_recovery_modes <= 3
        ):
            raise ValueError("minimum online actor recovery-mode count is invalid")
        if (
            type(self.online_sample_actor_categorical) is not bool
            or type(self.online_sample_continuous_action) is not bool
        ):
            raise TypeError("online behavior sampling flags must be boolean")
        if self.prior_capacity < self.batch_size // 2 or self.online_capacity < self.batch_size // 2:
            raise ValueError("hybrid replay capacity is smaller than half a batch")
        probabilities = (self.gamma, self.target_tau, self.prior_fraction)
        if any(not np.isfinite(value) or not 0.0 < value <= 1.0 for value in probabilities):
            raise ValueError("hybrid SAC probability configuration is invalid")
        if not np.isclose(self.prior_fraction, 0.5, rtol=0.0, atol=1.0e-12):
            raise ValueError("RLPD symmetric replay is frozen at exactly 50 percent prior data")
        if (
            not np.isfinite(self.online_mode_exploration_fraction)
            or not 0.0 <= self.online_mode_exploration_fraction <= 1.0
        ):
            raise ValueError("online mode exploration fraction must be in [0, 1]")
        positives = (
            self.actor_learning_rate,
            self.online_actor_learning_rate,
            self.critic_learning_rate,
            self.continuous_entropy_temperature,
            self.mode_entropy_temperature,
            self.residual_scale,
            self.maximum_gradient_norm,
            self.advantage_temperature,
            self.maximum_advantage_weight,
            self.minimum_online_actor_effective_sample_size,
            self.maximum_actor_batch_action_delta,
            self.intervention_terminal_penalty,
        )
        if any(not np.isfinite(value) or value <= 0.0 for value in positives):
            raise ValueError("hybrid SAC positive configuration is invalid")
        if self.maximum_advantage_weight < 1.0:
            raise ValueError("maximum advantage weight must be at least one")
        nonnegative = (
            self.minimum_online_advantage,
            self.prior_behavior_cloning_coefficient,
            self.online_advantage_regression_coefficient,
        )
        if any(not np.isfinite(value) or value < 0.0 for value in nonnegative):
            raise ValueError("hybrid actor regression coefficient is invalid")
        if (
            not np.isfinite(self.exploration_log_std)
            or not -5.0 <= self.exploration_log_std <= 0.5
        ):
            raise ValueError("hybrid actor exploration log standard deviation is invalid")


class HybridContactActor(nn.Module):
    """Categorical contact mode with a tanh-Gaussian residual per mode."""

    def __init__(
        self,
        observation_dim: int,
        hidden_dim: int,
        residual_scale: float,
        exploration_log_std: float,
        mode_mask_start: int | None = None,
        mode_action_centres: np.ndarray | None = None,
    ) -> None:
        super().__init__()
        if type(observation_dim) is not int or observation_dim < 1:
            raise ValueError("actor observation dimension is invalid")
        if type(hidden_dim) is not int or hidden_dim < 1:
            raise ValueError("actor hidden dimension is invalid")
        if not np.isfinite(residual_scale) or residual_scale <= 0.0:
            raise ValueError("actor residual scale is invalid")
        centres = np.asarray(
            DEFAULT_MODE_ACTION_CENTRES if mode_action_centres is None else mode_action_centres,
            dtype=np.float32,
        )
        if (
            centres.shape != (CONTACT_MODE_COUNT, TASK_ACTION_DIM)
            or not np.all(np.isfinite(centres))
            or np.any(np.abs(centres) >= 1.0)
        ):
            raise ValueError("actor mode centres must be finite and strictly inside [-1, 1]")
        centre_logits = np.arctanh(centres).astype(np.float32)
        self.observation_dim = observation_dim
        self.residual_scale = float(residual_scale)
        self.mode_mask_start = mode_mask_start
        self.register_buffer("mode_centre_logits", torch.from_numpy(centre_logits))
        self.trunk = nn.Sequential(
            nn.LayerNorm(observation_dim),
            nn.Linear(observation_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.mode_logits = nn.Linear(hidden_dim, CONTACT_MODE_COUNT)
        self.residual_mean = nn.Linear(hidden_dim, CONTACT_MODE_COUNT * TASK_ACTION_DIM)
        self.residual_log_std = nn.Linear(hidden_dim, CONTACT_MODE_COUNT * TASK_ACTION_DIM)
        nn.init.zeros_(self.residual_mean.weight)
        nn.init.zeros_(self.residual_mean.bias)
        nn.init.zeros_(self.residual_log_std.weight)
        nn.init.constant_(self.residual_log_std.bias, exploration_log_std)

    def _distribution_parameters(
        self,
        observation: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if observation.ndim != 2 or observation.shape[-1] != self.observation_dim:
            raise ValueError("actor observation shape is invalid")
        hidden = self.trunk(observation)
        logits = self.mode_logits(hidden)
        if self.mode_mask_start is not None:
            mask = observation[
                :,
                self.mode_mask_start : self.mode_mask_start + CONTACT_MODE_COUNT,
            ] > 0.5
            if torch.any(~torch.any(mask, dim=-1)):
                raise ValueError("actor observation contains an empty contact-mode mask")
            logits = logits.masked_fill(~mask, torch.finfo(logits.dtype).min)
        log_mode_probability = F.log_softmax(logits, dim=-1)
        mode_probability = log_mode_probability.exp()
        residual_mean = self.residual_mean(hidden).view(
            -1,
            CONTACT_MODE_COUNT,
            TASK_ACTION_DIM,
        )
        log_std = self.residual_log_std(hidden).view(
            -1,
            CONTACT_MODE_COUNT,
            TASK_ACTION_DIM,
        )
        log_std = torch.clamp(log_std, -5.0, 0.5)
        latent_mean = self.mode_centre_logits.unsqueeze(0) + self.residual_scale * residual_mean
        return logits, mode_probability, log_mode_probability, latent_mean, log_std

    def sample_all_modes(
        self,
        observation: torch.Tensor,
        *,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return probabilities and one continuous action for every mode."""

        _, probability, log_probability, mean, log_std = self._distribution_parameters(observation)
        distribution = torch.distributions.Normal(mean, log_std.exp())
        latent = mean if deterministic else distribution.rsample()
        action = torch.tanh(latent)
        continuous_log_probability = (
            distribution.log_prob(latent) - torch.log(1.0 - action.square() + 1.0e-6)
        ).sum(dim=-1)
        return probability, log_probability, action, continuous_log_probability

    def action_for_mode(
        self,
        observation: torch.Tensor,
        mode: torch.Tensor,
        *,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if mode.ndim != 1 or mode.shape[0] != observation.shape[0]:
            raise ValueError("actor mode shape is invalid")
        if torch.any(mode < 0) or torch.any(mode >= CONTACT_MODE_COUNT):
            raise ValueError("actor mode index is invalid")
        _, _, all_actions, all_log_probability = self.sample_all_modes(
            observation,
            deterministic=deterministic,
        )
        batch = torch.arange(observation.shape[0], device=observation.device)
        return all_actions[batch, mode], all_log_probability[batch, mode]

    def data_log_probability(
        self,
        observation: torch.Tensor,
        mode: torch.Tensor,
        action: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Evaluate recorded mode/action pairs for the decaying prior anchor."""

        if action.shape != (observation.shape[0], TASK_ACTION_DIM):
            raise ValueError("actor replay action shape is invalid")
        _, _, mode_log_probability, mean, log_std = self._distribution_parameters(observation)
        batch = torch.arange(observation.shape[0], device=observation.device)
        selected_mean = mean[batch, mode]
        selected_log_std = log_std[batch, mode]
        distribution = torch.distributions.Normal(selected_mean, selected_log_std.exp())
        bounded = torch.clamp(action, -0.999999, 0.999999)
        latent = torch.atanh(bounded)
        action_log_probability = (
            distribution.log_prob(latent) - torch.log(1.0 - bounded.square() + 1.0e-6)
        ).sum(dim=-1)
        return mode_log_probability[batch, mode], action_log_probability

    def mode_probabilities(self, observation: torch.Tensor) -> torch.Tensor:
        return self._distribution_parameters(observation)[1]

    def contact_mode_logits(self, observation: torch.Tensor) -> torch.Tensor:
        """Expose learned mode logits for explicitly balanced supervision."""

        return self._distribution_parameters(observation)[0]


class _ModeActionValue(nn.Module):
    def __init__(self, observation_dim: int, hidden_dim: int) -> None:
        super().__init__()
        input_dim = observation_dim + CONTACT_MODE_COUNT + TASK_ACTION_DIM
        self.network = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        observation: torch.Tensor,
        mode: torch.Tensor,
        action: torch.Tensor,
    ) -> torch.Tensor:
        one_hot = F.one_hot(mode, CONTACT_MODE_COUNT).to(observation.dtype)
        return self.network(torch.cat((observation, one_hot, action), dim=-1)).squeeze(-1)

    def all_modes(self, observation: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        if action.shape != (observation.shape[0], CONTACT_MODE_COUNT, TASK_ACTION_DIM):
            raise ValueError("critic all-mode action shape is invalid")
        batch = observation.shape[0]
        expanded_observation = observation[:, None, :].expand(-1, CONTACT_MODE_COUNT, -1)
        mode = torch.arange(CONTACT_MODE_COUNT, device=observation.device)[None, :].expand(batch, -1)
        value = self.forward(
            expanded_observation.reshape(batch * CONTACT_MODE_COUNT, -1),
            mode.reshape(-1),
            action.reshape(batch * CONTACT_MODE_COUNT, TASK_ACTION_DIM),
        )
        return value.view(batch, CONTACT_MODE_COUNT)


class TwinModeActionCritic(nn.Module):
    def __init__(self, observation_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.q1 = _ModeActionValue(observation_dim, hidden_dim)
        self.q2 = _ModeActionValue(observation_dim, hidden_dim)

    def forward(
        self,
        observation: torch.Tensor,
        mode: torch.Tensor,
        action: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.q1(observation, mode, action), self.q2(observation, mode, action)

    def all_modes(
        self,
        observation: torch.Tensor,
        action: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.q1.all_modes(observation, action), self.q2.all_modes(observation, action)


@dataclass
class HybridContactSACBundle:
    actor: HybridContactActor
    reference_actor: HybridContactActor
    critic: TwinModeActionCritic
    target_critic: TwinModeActionCritic
    actor_optimizer: torch.optim.Optimizer
    critic_optimizer: torch.optim.Optimizer
    update_index: int = 0
    actor_update_count: int = 0
    format: str = HYBRID_CONTACT_SAC_FORMAT


def initialize_hybrid_contact_sac(
    seed: int,
    *,
    device: str | torch.device,
    config: HybridContactSACConfig,
) -> HybridContactSACBundle:
    config.validate()
    if type(seed) is not int or seed < 0:
        raise ValueError("hybrid SAC seed is invalid")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        actor = HybridContactActor(
            config.observation_dim,
            config.hidden_dim,
            config.residual_scale,
            config.exploration_log_std,
            mode_mask_start=config.mode_mask_start,
        )
        critic = TwinModeActionCritic(config.observation_dim, config.hidden_dim)
        target = TwinModeActionCritic(config.observation_dim, config.hidden_dim)
    target.load_state_dict(critic.state_dict(), strict=True)
    for parameter in target.parameters():
        parameter.requires_grad_(False)
    actor = actor.to(device)
    reference_actor = copy.deepcopy(actor).to(device)
    reference_actor.eval()
    for parameter in reference_actor.parameters():
        parameter.requires_grad_(False)
    critic = critic.to(device)
    target = target.to(device)
    return HybridContactSACBundle(
        actor=actor,
        reference_actor=reference_actor,
        critic=critic,
        target_critic=target,
        actor_optimizer=torch.optim.Adam(actor.parameters(), lr=config.actor_learning_rate),
        critic_optimizer=torch.optim.Adam(critic.parameters(), lr=config.critic_learning_rate),
    )


def synchronize_reference_actor(bundle: HybridContactSACBundle) -> None:
    """Freeze the current accepted actor as the next AWAC baseline."""

    bundle.reference_actor.load_state_dict(bundle.actor.state_dict(), strict=True)
    bundle.reference_actor.eval()
    for parameter in bundle.reference_actor.parameters():
        parameter.requires_grad_(False)


def _module_state_sha256(module: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


class HybridReplayBuffer:
    """Fixed-schema replay.  A frozen instance is immutable prior evidence."""

    def __init__(self, capacity: int, observation_dim: int) -> None:
        if type(capacity) is not int or capacity < 1 or type(observation_dim) is not int or observation_dim < 1:
            raise ValueError("hybrid replay dimensions are invalid")
        self.capacity = capacity
        self.observation_dim = observation_dim
        self.size = 0
        self.cursor = 0
        self.frozen = False
        self.observation = np.empty((capacity, observation_dim), dtype=np.float32)
        self.next_observation = np.empty_like(self.observation)
        self.proposed_action = np.empty((capacity, TASK_ACTION_DIM), dtype=np.float32)
        self.projected_action = np.empty_like(self.proposed_action)
        self.executed_action = np.empty_like(self.proposed_action)
        self.mode = np.empty(capacity, dtype=np.int64)
        self.reward = np.empty(capacity, dtype=np.float32)
        self.terminal = np.empty(capacity, dtype=bool)
        self.intervention = np.empty(capacity, dtype=bool)
        self.actor_learning_eligible = np.empty(capacity, dtype=bool)
        self.proposal_execution_l2 = np.empty(capacity, dtype=np.float32)
        self.proposal_projection_l2 = np.empty(capacity, dtype=np.float32)
        self.projection_tracking_l2 = np.empty(capacity, dtype=np.float32)
        self.projection_provenance_observed = np.empty(capacity, dtype=bool)
        self.analytic_projection_applied = np.empty(capacity, dtype=bool)
        self.command_provenance_observed = np.empty(capacity, dtype=bool)
        self.command_safety_rewrite = np.empty(capacity, dtype=bool)
        self.strict_success = np.empty(capacity, dtype=bool)
        self.stage = np.empty(capacity, dtype=np.int16)

    def add(
        self,
        *,
        observation: np.ndarray,
        next_observation: np.ndarray,
        proposed_action: np.ndarray,
        projected_action: np.ndarray | None = None,
        executed_action: np.ndarray,
        mode: int,
        reward: float,
        terminal: bool,
        intervention: bool,
        strict_success: bool,
        stage: int,
        actor_learning_eligible: bool = True,
        proposal_execution_l2: float | None = None,
        proposal_projection_l2: float | None = None,
        projection_tracking_l2: float | None = None,
        projection_provenance_observed: bool = False,
        analytic_projection_applied: bool = False,
        command_provenance_observed: bool = False,
        command_safety_rewrite: bool = False,
    ) -> None:
        if self.frozen:
            raise RuntimeError("frozen prior replay cannot be mutated")
        vectors = {
            "observation": (observation, (self.observation_dim,)),
            "next_observation": (next_observation, (self.observation_dim,)),
            "proposed_action": (proposed_action, (TASK_ACTION_DIM,)),
            "projected_action": (
                proposed_action if projected_action is None else projected_action,
                (TASK_ACTION_DIM,),
            ),
            "executed_action": (executed_action, (TASK_ACTION_DIM,)),
        }
        normalized: dict[str, np.ndarray] = {}
        for name, (value, shape) in vectors.items():
            array = np.asarray(value, dtype=np.float32)
            if array.shape != shape or not np.all(np.isfinite(array)):
                raise ValueError(f"hybrid replay {name} is invalid")
            normalized[name] = array
        execution_l2 = (
            float(np.linalg.norm(normalized["proposed_action"] - normalized["executed_action"]))
            if proposal_execution_l2 is None
            else float(proposal_execution_l2)
        )
        proposal_projection = (
            float(
                np.linalg.norm(
                    normalized["proposed_action"] - normalized["projected_action"]
                )
            )
            if proposal_projection_l2 is None
            else float(proposal_projection_l2)
        )
        projection_tracking = (
            float(
                np.linalg.norm(
                    normalized["projected_action"] - normalized["executed_action"]
                )
            )
            if projection_tracking_l2 is None
            else float(projection_tracking_l2)
        )
        if (
            type(mode) is not int
            or not 0 <= mode < CONTACT_MODE_COUNT
            or not np.isfinite(reward)
            or type(terminal) is not bool
            or type(intervention) is not bool
            or type(actor_learning_eligible) is not bool
            or type(projection_provenance_observed) is not bool
            or type(analytic_projection_applied) is not bool
            or type(command_provenance_observed) is not bool
            or type(command_safety_rewrite) is not bool
            or type(strict_success) is not bool
            or type(stage) is not int
            or stage < 0
            or intervention
            and not terminal
            or strict_success
            and not terminal
            or not np.isfinite(execution_l2)
            or execution_l2 < 0.0
            or not np.isfinite(proposal_projection)
            or proposal_projection < 0.0
            or not np.isfinite(projection_tracking)
            or projection_tracking < 0.0
            or analytic_projection_applied
            and not projection_provenance_observed
            or command_safety_rewrite
            and not command_provenance_observed
        ):
            raise ValueError("hybrid replay scalar metadata is invalid")
        index = self.cursor
        for name, array in normalized.items():
            getattr(self, name)[index] = array
        self.mode[index] = mode
        self.reward[index] = reward
        self.terminal[index] = terminal
        self.intervention[index] = intervention
        self.actor_learning_eligible[index] = bool(
            actor_learning_eligible
            and not intervention
            and projection_provenance_observed
            and command_provenance_observed
            and not command_safety_rewrite
        )
        self.proposal_execution_l2[index] = execution_l2
        self.proposal_projection_l2[index] = proposal_projection
        self.projection_tracking_l2[index] = projection_tracking
        self.projection_provenance_observed[index] = projection_provenance_observed
        self.analytic_projection_applied[index] = analytic_projection_applied
        self.command_provenance_observed[index] = command_provenance_observed
        self.command_safety_rewrite[index] = command_safety_rewrite
        self.strict_success[index] = strict_success
        self.stage[index] = stage
        self.cursor = (self.cursor + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def freeze(self) -> None:
        if self.size < 1:
            raise RuntimeError("empty prior replay cannot be frozen")
        self.frozen = True

    def _sample_indices(self, indices: np.ndarray) -> dict[str, np.ndarray]:
        selected = np.asarray(indices, dtype=np.int64)
        if (
            selected.ndim != 1
            or selected.size < 1
            or np.any(selected < 0)
            or np.any(selected >= self.size)
        ):
            raise ValueError("hybrid replay sample indices are invalid")
        names = (
            "observation",
            "next_observation",
            "proposed_action",
            "projected_action",
            "executed_action",
            "mode",
            "reward",
            "terminal",
            "intervention",
            "actor_learning_eligible",
            "proposal_execution_l2",
            "proposal_projection_l2",
            "projection_tracking_l2",
            "projection_provenance_observed",
            "analytic_projection_applied",
            "command_provenance_observed",
            "command_safety_rewrite",
            "strict_success",
            "stage",
        )
        return {name: getattr(self, name)[selected].copy() for name in names}

    def sample(self, count: int, rng: np.random.Generator) -> dict[str, np.ndarray]:
        if type(count) is not int or count < 1 or self.size < count:
            raise ValueError("hybrid replay does not contain the requested sample")
        indices = rng.integers(0, self.size, size=count)
        return self._sample_indices(indices)

    def sample_balanced_modes(
        self,
        count: int,
        rng: np.random.Generator,
    ) -> dict[str, np.ndarray]:
        """Sample every mode present in replay equally, with replacement."""

        if type(count) is not int or count < 1 or self.size < 1:
            raise ValueError("balanced mode sample count is invalid")
        present = np.unique(self.mode[: self.size])
        if present.size < 1:
            raise RuntimeError("balanced mode sampler found no recorded mode")
        base = count // int(present.size)
        remainder = count % int(present.size)
        pieces: list[np.ndarray] = []
        for order, mode in enumerate(present):
            mode_indices = np.flatnonzero(self.mode[: self.size] == mode)
            take = base + int(order < remainder)
            if take > 0:
                pieces.append(rng.choice(mode_indices, size=take, replace=True))
        indices = np.concatenate(pieces)
        return self._sample_indices(indices[rng.permutation(indices.size)])

    def actor_eligible_mode_counts(self) -> np.ndarray:
        """Return raw unique online actor evidence counts for every mode."""

        eligible = self.actor_learning_eligible[: self.size] & ~self.intervention[: self.size]
        return np.bincount(
            self.mode[: self.size][eligible],
            minlength=CONTACT_MODE_COUNT,
        ).astype(np.int64)

    def sample_balanced_actor_evidence(
        self,
        count: int,
        rng: np.random.Generator,
        *,
        minimum_per_mode: int,
    ) -> dict[str, np.ndarray]:
        """Sample eligible modes round-robin without replacement.

        Modes with fewer than ``minimum_per_mode`` raw unique rows are held
        out.  This prevents one rare transition from being duplicated many
        times and falsely inflating actor effective sample size.
        """

        if (
            type(count) is not int
            or count < 1
            or type(minimum_per_mode) is not int
            or minimum_per_mode < 1
            or self.size < 1
        ):
            raise ValueError("balanced actor-evidence sample arguments are invalid")
        eligible = self.actor_learning_eligible[: self.size] & ~self.intervention[: self.size]
        counts = self.actor_eligible_mode_counts()
        qualified = np.flatnonzero(counts >= minimum_per_mode)
        if qualified.size < 1:
            raise RuntimeError("no contact mode has sufficient actor evidence")
        pools = {
            int(mode): rng.permutation(
                np.flatnonzero(eligible & (self.mode[: self.size] == mode))
            ).tolist()
            for mode in qualified
        }
        target = min(count, sum(len(indices) for indices in pools.values()))
        selected: list[int] = []
        positions = {mode: 0 for mode in pools}
        while len(selected) < target:
            progressed = False
            for mode in rng.permutation(qualified):
                mode_value = int(mode)
                position = positions[mode_value]
                if position < len(pools[mode_value]):
                    selected.append(int(pools[mode_value][position]))
                    positions[mode_value] = position + 1
                    progressed = True
                    if len(selected) == target:
                        break
            if not progressed:
                break
        indices = np.asarray(selected, dtype=np.int64)
        if indices.size != target or np.unique(indices).size != indices.size:
            raise RuntimeError("balanced actor-evidence sampler lost uniqueness")
        result = self._sample_indices(indices)
        result["replay_index"] = indices.copy()
        return result

    def state_dict(self) -> dict[str, Any]:
        names = (
            "observation",
            "next_observation",
            "proposed_action",
            "projected_action",
            "executed_action",
            "mode",
            "reward",
            "terminal",
            "intervention",
            "actor_learning_eligible",
            "proposal_execution_l2",
            "proposal_projection_l2",
            "projection_tracking_l2",
            "projection_provenance_observed",
            "analytic_projection_applied",
            "command_provenance_observed",
            "command_safety_rewrite",
            "strict_success",
            "stage",
        )
        return {
            "format": HYBRID_REPLAY_FORMAT,
            "capacity": self.capacity,
            "observation_dim": self.observation_dim,
            "size": self.size,
            "cursor": self.cursor,
            "frozen": self.frozen,
            **{name: getattr(self, name)[: self.size].copy() for name in names},
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore a canonical replay snapshot with strict schema checks."""

        replay_format = state.get("format")
        if replay_format not in (HYBRID_REPLAY_FORMAT, LEGACY_HYBRID_REPLAY_FORMAT):
            raise ValueError("hybrid replay snapshot format is invalid")
        if state.get("observation_dim") != self.observation_dim:
            raise ValueError("hybrid replay snapshot observation dimension differs")
        size = state.get("size")
        frozen = state.get("frozen")
        if type(size) is not int or not 0 <= size <= self.capacity or type(frozen) is not bool:
            raise ValueError("hybrid replay snapshot scalar metadata is invalid")
        schema = {
            "observation": (np.float32, (size, self.observation_dim)),
            "next_observation": (np.float32, (size, self.observation_dim)),
            "proposed_action": (np.float32, (size, TASK_ACTION_DIM)),
            "executed_action": (np.float32, (size, TASK_ACTION_DIM)),
            "mode": (np.int64, (size,)),
            "reward": (np.float32, (size,)),
            "terminal": (bool, (size,)),
            "intervention": (bool, (size,)),
            "actor_learning_eligible": (bool, (size,)),
            "proposal_execution_l2": (np.float32, (size,)),
            "strict_success": (bool, (size,)),
            "stage": (np.int16, (size,)),
        }
        if replay_format == HYBRID_REPLAY_FORMAT:
            schema.update(
                {
                    "projected_action": (np.float32, (size, TASK_ACTION_DIM)),
                    "proposal_projection_l2": (np.float32, (size,)),
                    "projection_tracking_l2": (np.float32, (size,)),
                    "projection_provenance_observed": (bool, (size,)),
                    "analytic_projection_applied": (bool, (size,)),
                    "command_provenance_observed": (bool, (size,)),
                    "command_safety_rewrite": (bool, (size,)),
                }
            )
        restored: dict[str, np.ndarray] = {}
        for name, (dtype, shape) in schema.items():
            value = np.asarray(state.get(name), dtype=dtype)
            if value.shape != shape:
                raise ValueError(f"hybrid replay snapshot {name} shape is invalid")
            restored[name] = value
        if replay_format == LEGACY_HYBRID_REPLAY_FORMAT:
            restored.update(
                {
                    "projected_action": restored["proposed_action"].copy(),
                    "proposal_projection_l2": np.zeros(size, dtype=np.float32),
                    "projection_tracking_l2": restored[
                        "proposal_execution_l2"
                    ].copy(),
                    "projection_provenance_observed": np.zeros(size, dtype=bool),
                    "analytic_projection_applied": np.zeros(size, dtype=bool),
                    "command_provenance_observed": np.zeros(size, dtype=bool),
                    "command_safety_rewrite": np.zeros(size, dtype=bool),
                }
            )
            # Legacy snapshots cannot prove the projection/command lineage
            # now required for online actor regression.  They remain usable
            # as immutable priors and critic evidence.
            restored["actor_learning_eligible"] = np.zeros(size, dtype=bool)
        floating = np.concatenate(
            [
                restored["observation"].reshape(-1),
                restored["next_observation"].reshape(-1),
                restored["proposed_action"].reshape(-1),
                restored["projected_action"].reshape(-1),
                restored["executed_action"].reshape(-1),
                restored["reward"].reshape(-1),
                restored["proposal_execution_l2"].reshape(-1),
                restored["proposal_projection_l2"].reshape(-1),
                restored["projection_tracking_l2"].reshape(-1),
            ]
        )
        if (
            not np.all(np.isfinite(floating))
            or np.any(restored["mode"] < 0)
            or np.any(restored["mode"] >= CONTACT_MODE_COUNT)
            or np.any(restored["stage"] < 0)
            or np.any(restored["proposal_execution_l2"] < 0.0)
            or np.any(restored["proposal_projection_l2"] < 0.0)
            or np.any(restored["projection_tracking_l2"] < 0.0)
            or np.any(restored["intervention"] & ~restored["terminal"])
            or np.any(restored["strict_success"] & ~restored["terminal"])
            or np.any(restored["intervention"] & restored["actor_learning_eligible"])
            or np.any(
                restored["analytic_projection_applied"]
                & ~restored["projection_provenance_observed"]
            )
            or np.any(
                restored["command_safety_rewrite"]
                & ~restored["command_provenance_observed"]
            )
            or np.any(
                restored["actor_learning_eligible"]
                & (
                    ~restored["projection_provenance_observed"]
                    | ~restored["command_provenance_observed"]
                    | restored["command_safety_rewrite"]
                )
            )
        ):
            raise ValueError("hybrid replay snapshot contents are invalid")
        for name, value in restored.items():
            getattr(self, name)[:size] = value
        self.size = size
        self.cursor = size % self.capacity
        self.frozen = frozen


class SymmetricPriorOnlineReplay:
    """RLPD sampler with an auditable exact 50/50 source split."""

    def __init__(self, config: HybridContactSACConfig) -> None:
        config.validate()
        self.config = config
        self.prior = HybridReplayBuffer(config.prior_capacity, config.observation_dim)
        self.online = HybridReplayBuffer(config.online_capacity, config.observation_dim)

    def sample(self, rng: np.random.Generator) -> dict[str, np.ndarray]:
        half = self.config.batch_size // 2
        prior = self.prior.sample(half, rng)
        online = self.online.sample(half, rng)
        permutation = rng.permutation(self.config.batch_size)
        result: dict[str, np.ndarray] = {}
        for name in prior:
            result[name] = np.concatenate((prior[name], online[name]), axis=0)[permutation]
        result["source_is_prior"] = np.concatenate(
            (np.ones(half, dtype=bool), np.zeros(half, dtype=bool)),
            axis=0,
        )[permutation]
        return result

    @property
    def ready(self) -> bool:
        half = self.config.batch_size // 2
        return self.prior.frozen and self.prior.size >= half and self.online.size >= half


def add_intervention_surrogate(
    replay: HybridReplayBuffer,
    *,
    observation: np.ndarray,
    next_observation: np.ndarray,
    proposed_action: np.ndarray,
    executed_action: np.ndarray,
    mode: int,
    stage: int,
    penalty: float,
) -> None:
    """Record a shield substitution as a terminal surrogate-MDP event."""

    if not np.isfinite(penalty) or penalty <= 0.0:
        raise ValueError("intervention penalty must be finite and positive")
    replay.add(
        observation=observation,
        next_observation=next_observation,
        proposed_action=proposed_action,
        executed_action=executed_action,
        mode=mode,
        reward=-float(penalty),
        terminal=True,
        intervention=True,
        actor_learning_eligible=False,
        strict_success=False,
        stage=stage,
    )


def _torch_batch(batch: dict[str, np.ndarray], device: str | torch.device) -> dict[str, torch.Tensor]:
    result: dict[str, torch.Tensor] = {}
    for name, value in batch.items():
        array = np.asarray(value)
        if array.dtype == bool:
            result[name] = torch.as_tensor(array, dtype=torch.bool, device=device)
        elif np.issubdtype(array.dtype, np.integer):
            result[name] = torch.as_tensor(array, dtype=torch.long, device=device)
        else:
            result[name] = torch.as_tensor(array, dtype=torch.float32, device=device)
    return result


def pretrain_hybrid_actor_from_prior(
    bundle: HybridContactSACBundle,
    prior: HybridReplayBuffer,
    *,
    config: HybridContactSACConfig,
    rng: np.random.Generator,
    device: str | torch.device,
) -> dict[str, float]:
    """One explicit supervised warm-start update on successful prior data."""

    count = min(config.batch_size, prior.size)
    if count < 1:
        raise ValueError("cannot pretrain from an empty prior replay")
    batch = _torch_batch(prior.sample_balanced_modes(count, rng), device)
    logits = bundle.actor.contact_mode_logits(batch["observation"])
    mode_loss = F.cross_entropy(logits, batch["mode"])
    predicted_action, _ = bundle.actor.action_for_mode(
        batch["observation"],
        batch["mode"],
        deterministic=True,
    )
    action_loss = F.mse_loss(predicted_action, batch["proposed_action"])
    # The two terms are deliberately explicit.  A joint log-likelihood can
    # become negative by shrinking Gaussian variance and swamp the much
    # smaller mode-classification gradient, which caused the first smoke
    # policy to select stick_push from the initial approach state.
    loss = mode_loss + 4.0 * action_loss
    bundle.actor_optimizer.zero_grad(set_to_none=True)
    loss.backward()
    gradient = torch.nn.utils.clip_grad_norm_(bundle.actor.parameters(), config.maximum_gradient_norm)
    bundle.actor_optimizer.step()
    return {
        "behavior_cloning_loss": float(loss.detach().cpu()),
        "mode_classification_loss": float(mode_loss.detach().cpu()),
        "action_regression_loss": float(action_loss.detach().cpu()),
        "actor_gradient_norm": float(torch.as_tensor(gradient).detach().cpu()),
        "sample_count": float(count),
        "balanced_mode_sampling": 1.0,
    }


def update_hybrid_contact_sac(
    bundle: HybridContactSACBundle,
    replay: SymmetricPriorOnlineReplay,
    *,
    config: HybridContactSACConfig,
    rng: np.random.Generator,
    device: str | torch.device,
) -> dict[str, float | int | bool | str]:
    """Update the critic once and the conservative actor only when scheduled.

    The actor never differentiates through Q.  The twin critics instead score
    replay proposals against the current deterministic policy baseline.  Every
    prior action remains an immutable anchor, while an online action is eligible
    for regression only if its clipped double-Q advantage is positive and its
    executor provenance was independently marked usable.
    """

    if not replay.ready:
        raise RuntimeError("symmetric prior/online replay is not ready")
    batch = _torch_batch(replay.sample(rng), device)
    observation = batch["observation"]
    next_observation = batch["next_observation"]
    proposed_action = batch["proposed_action"]
    mode = batch["mode"]
    reward = batch["reward"]
    terminal = batch["terminal"].to(torch.float32)

    with torch.no_grad():
        probability, _log_mode_probability, next_action, _continuous_log_probability = (
            bundle.actor.sample_all_modes(next_observation, deterministic=True)
        )
        next_q1, next_q2 = bundle.target_critic.all_modes(next_observation, next_action)
        next_q = torch.minimum(next_q1, next_q2)
        next_value = torch.sum(probability * next_q, dim=-1)
        target = reward + config.gamma * (1.0 - terminal) * next_value

    q1, q2 = bundle.critic(observation, mode, proposed_action)
    critic_loss = F.mse_loss(q1, target) + F.mse_loss(q2, target)
    bundle.critic_optimizer.zero_grad(set_to_none=True)
    critic_loss.backward()
    critic_gradient = torch.nn.utils.clip_grad_norm_(
        bundle.critic.parameters(),
        config.maximum_gradient_norm,
    )
    bundle.critic_optimizer.step()

    source_is_prior = batch["source_is_prior"]
    next_update_index = bundle.update_index + 1
    actor_update_scheduled = bool(
        next_update_index > config.actor_update_delay
        and (next_update_index - config.actor_update_delay) % config.actor_update_interval == 0
    )
    actor_update_applied = False
    actor_update_attempted = False
    actor_update_skip_reason = "not_scheduled"
    actor_loss = torch.zeros((), dtype=critic_loss.dtype, device=critic_loss.device)
    behavior_cloning_loss = torch.zeros_like(actor_loss)
    online_advantage_loss = torch.zeros_like(actor_loss)
    online_mode_classification_loss = torch.zeros_like(actor_loss)
    online_action_regression_loss = torch.zeros_like(actor_loss)
    mode_classification_loss = torch.zeros_like(actor_loss)
    action_regression_loss = torch.zeros_like(actor_loss)
    actor_gradient = torch.zeros_like(actor_loss)
    online_selected = torch.zeros(0, dtype=torch.bool, device=critic_loss.device)
    online_weights = torch.zeros(0, dtype=reward.dtype, device=critic_loss.device)
    replay_advantage = torch.zeros(0, dtype=reward.dtype, device=critic_loss.device)
    selected_online_fraction = 0.0
    mean_selected_online_weight = 0.0
    maximum_selected_online_weight = 0.0
    effective_online_sample_size = 0.0
    eligible_online_sample_count = 0
    selected_online_sample_count = 0
    selected_proposal_execution_l2_mean = 0.0
    selected_proposal_execution_l2_maximum = 0.0
    eligible_online_advantage_minimum = 0.0
    eligible_online_advantage_mean = 0.0
    eligible_online_advantage_maximum = 0.0
    mean_actor_batch_action_delta = 0.0
    maximum_actor_batch_action_delta = 0.0
    actor_parameter_sha256_before = (
        _module_state_sha256(bundle.actor) if actor_update_scheduled else ""
    )
    actor_prior_batch: dict[str, torch.Tensor] | None = None
    actor_online_batch: dict[str, torch.Tensor] | None = None
    actor_evidence_raw_counts = replay.online.actor_eligible_mode_counts()
    actor_evidence_qualified_modes = np.flatnonzero(
        actor_evidence_raw_counts >= config.minimum_online_actor_evidence_per_mode
    )
    recovery_mode_indices = np.asarray([2, 3, 4], dtype=np.int64)
    actor_evidence_recovery_mode_count = int(
        np.count_nonzero(
            actor_evidence_raw_counts[recovery_mode_indices]
            >= config.minimum_online_actor_evidence_per_mode
        )
    )
    actor_evidence_online_count = 0
    actor_evidence_prior_count = 0
    actor_evidence_selected_mode_counts = np.zeros(CONTACT_MODE_COUNT, dtype=np.int64)
    if actor_update_scheduled:
        if (
            actor_evidence_recovery_mode_count
            < config.minimum_online_actor_recovery_modes
        ):
            actor_update_skip_reason = "insufficient_recovery_mode_evidence"
        elif actor_evidence_qualified_modes.size < 1:
            actor_update_skip_reason = "insufficient_mode_stratified_actor_evidence"
        else:
            online_numpy = replay.online.sample_balanced_actor_evidence(
                config.batch_size // 2,
                rng,
                minimum_per_mode=config.minimum_online_actor_evidence_per_mode,
            )
            actor_evidence_online_count = int(online_numpy["mode"].size)
            prior_numpy = replay.prior.sample_balanced_modes(
                actor_evidence_online_count,
                rng,
            )
            actor_evidence_prior_count = int(prior_numpy["mode"].size)
            actor_prior_batch = _torch_batch(prior_numpy, device)
            actor_online_batch = _torch_batch(online_numpy, device)
            online_observation = actor_online_batch["observation"]
            online_mode = actor_online_batch["mode"]
            online_proposed_action = actor_online_batch["proposed_action"]
            with torch.no_grad():
                (
                    policy_probability,
                    _policy_log_probability,
                    policy_action,
                    _policy_action_logp,
                ) = bundle.reference_actor.sample_all_modes(
                    online_observation,
                    deterministic=True,
                )
                policy_q1, policy_q2 = bundle.target_critic.all_modes(
                    online_observation,
                    policy_action,
                )
                policy_value = torch.sum(
                    policy_probability * torch.minimum(policy_q1, policy_q2),
                    dim=-1,
                )
                data_q1, data_q2 = bundle.target_critic(
                    online_observation,
                    online_mode,
                    online_proposed_action,
                )
                replay_advantage = torch.minimum(data_q1, data_q2) - policy_value
                online_selected = replay_advantage > config.minimum_online_advantage
                online_weights = torch.exp(
                    torch.clamp(
                        replay_advantage / config.advantage_temperature,
                        max=float(np.log(config.maximum_advantage_weight)),
                    )
                )
                online_weights = torch.where(
                    online_selected,
                    online_weights,
                    torch.zeros_like(online_weights),
                )
                eligible_online_sample_count = actor_evidence_online_count
                selected_online_sample_count = int(online_selected.sum().item())
                selected_online_fraction = selected_online_sample_count / max(
                    actor_evidence_online_count,
                    1,
                )
                if eligible_online_sample_count > 0:
                    eligible_online_advantage_minimum = float(
                        replay_advantage.min().cpu()
                    )
                    eligible_online_advantage_mean = float(
                        replay_advantage.mean().cpu()
                    )
                    eligible_online_advantage_maximum = float(
                        replay_advantage.max().cpu()
                    )
                if selected_online_sample_count > 0:
                    selected_weights = online_weights[online_selected]
                    mean_selected_online_weight = float(selected_weights.mean().cpu())
                    maximum_selected_online_weight = float(selected_weights.max().cpu())
                    effective_online_sample_size = float(
                        (
                            selected_weights.sum().square()
                            / selected_weights.square().sum()
                        ).cpu()
                    )
                    selected_execution_l2 = actor_online_batch[
                        "proposal_execution_l2"
                    ][online_selected]
                    selected_proposal_execution_l2_mean = float(
                        selected_execution_l2.mean().cpu()
                    )
                    selected_proposal_execution_l2_maximum = float(
                        selected_execution_l2.max().cpu()
                    )
                    actor_evidence_selected_mode_counts = np.bincount(
                        actor_online_batch["mode"][online_selected]
                        .detach()
                        .cpu()
                        .numpy(),
                        minlength=CONTACT_MODE_COUNT,
                    ).astype(np.int64)
            actor_update_applied = bool(
                selected_online_sample_count >= config.minimum_online_actor_samples
                and effective_online_sample_size
                >= config.minimum_online_actor_effective_sample_size
            )
            actor_update_skip_reason = (
                "applied"
                if actor_update_applied
                else "insufficient_positive_online_evidence"
            )

    if actor_update_applied:
        if actor_prior_batch is None or actor_online_batch is None:
            raise RuntimeError("actor update lacks mode-stratified evidence batches")
        actor_update_attempted = True
        prior_observation = actor_prior_batch["observation"]
        prior_mode = actor_prior_batch["mode"]
        prior_proposed_action = actor_prior_batch["proposed_action"]
        prior_logits = bundle.actor.contact_mode_logits(prior_observation)
        prior_mode_loss = F.cross_entropy(prior_logits, prior_mode, reduction="none")
        prior_predicted_action, _ = bundle.actor.action_for_mode(
            prior_observation,
            prior_mode,
            deterministic=True,
        )
        prior_action_loss = F.mse_loss(
            prior_predicted_action,
            prior_proposed_action,
            reduction="none",
        ).mean(dim=-1)
        prior_loss = prior_mode_loss + 4.0 * prior_action_loss
        mode_classification_loss = prior_mode_loss.mean()
        action_regression_loss = prior_action_loss.mean()
        behavior_cloning_loss = prior_loss.mean()

        selected_online_observation = actor_online_batch["observation"][online_selected]
        selected_online_mode = actor_online_batch["mode"][online_selected]
        selected_online_action = actor_online_batch["proposed_action"][online_selected]
        online_logits = bundle.actor.contact_mode_logits(selected_online_observation)
        per_online_mode_loss = F.cross_entropy(
            online_logits,
            selected_online_mode,
            reduction="none",
        )
        online_predicted_action, _ = bundle.actor.action_for_mode(
            selected_online_observation,
            selected_online_mode,
            deterministic=True,
        )
        per_online_action_loss = F.mse_loss(
            online_predicted_action,
            selected_online_action,
            reduction="none",
        ).mean(dim=-1)
        per_online_loss = per_online_mode_loss + 4.0 * per_online_action_loss
        selected_weights = online_weights[online_selected]
        online_mode_classification_loss = (
            selected_weights * per_online_mode_loss
        ).sum() / selected_weights.sum().clamp_min(1.0e-8)
        online_action_regression_loss = (
            selected_weights * per_online_action_loss
        ).sum() / selected_weights.sum().clamp_min(1.0e-8)
        online_advantage_loss = (
            selected_weights * per_online_loss
        ).sum() / selected_weights.sum().clamp_min(1.0e-8)
        actor_loss = (
            config.prior_behavior_cloning_coefficient * behavior_cloning_loss
            + config.online_advantage_regression_coefficient * online_advantage_loss
        )
        trust_observation = torch.cat(
            (prior_observation, selected_online_observation),
            dim=0,
        )
        trust_mode = torch.cat((prior_mode, selected_online_mode), dim=0)
        predicted_action_before, _ = bundle.actor.action_for_mode(
            trust_observation,
            trust_mode,
            deterministic=True,
        )
        predicted_action_before = predicted_action_before.detach().clone()
        actor_state_before = copy.deepcopy(bundle.actor.state_dict())
        actor_optimizer_state_before = copy.deepcopy(bundle.actor_optimizer.state_dict())
        bundle.actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        actor_gradient = torch.nn.utils.clip_grad_norm_(
            bundle.actor.parameters(),
            config.maximum_gradient_norm,
        )
        bundle.actor_optimizer.step()
        with torch.no_grad():
            predicted_action_after, _ = bundle.actor.action_for_mode(
                trust_observation,
                trust_mode,
                deterministic=True,
            )
            action_delta = torch.linalg.vector_norm(
                predicted_action_after - predicted_action_before,
                dim=-1,
            )
            mean_actor_batch_action_delta = float(action_delta.mean().cpu())
            maximum_actor_batch_action_delta = float(action_delta.max().cpu())
        if maximum_actor_batch_action_delta > config.maximum_actor_batch_action_delta:
            bundle.actor.load_state_dict(actor_state_before, strict=True)
            bundle.actor_optimizer.load_state_dict(actor_optimizer_state_before)
            actor_update_applied = False
            actor_update_skip_reason = "local_action_trust_region_exceeded"
        else:
            bundle.actor_update_count += 1
    actor_parameter_sha256_after = (
        _module_state_sha256(bundle.actor) if actor_update_scheduled else ""
    )

    with torch.no_grad():
        for target_parameter, parameter in zip(
            bundle.target_critic.parameters(),
            bundle.critic.parameters(),
            strict=True,
        ):
            target_parameter.mul_(1.0 - config.target_tau).add_(parameter, alpha=config.target_tau)
    bundle.update_index += 1
    prior_count = int(source_is_prior.sum().item())
    with torch.no_grad():
        current_probability = bundle.actor.mode_probabilities(observation)
        current_log_probability = torch.log(current_probability.clamp_min(1.0e-12))
    return {
        "format": HYBRID_CONTACT_SAC_FORMAT,
        "update_index": bundle.update_index,
        "critic_loss": float(critic_loss.detach().cpu()),
        "actor_loss": float(actor_loss.detach().cpu()),
        "behavior_cloning_loss": float(behavior_cloning_loss.detach().cpu()),
        "online_advantage_weighted_loss": float(online_advantage_loss.detach().cpu()),
        "online_mode_classification_loss": float(
            online_mode_classification_loss.detach().cpu()
        ),
        "online_action_regression_loss": float(
            online_action_regression_loss.detach().cpu()
        ),
        "mode_classification_loss": float(mode_classification_loss.detach().cpu()),
        "action_regression_loss": float(action_regression_loss.detach().cpu()),
        "behavior_cloning_coefficient": float(config.prior_behavior_cloning_coefficient),
        "online_advantage_regression_coefficient": float(
            config.online_advantage_regression_coefficient
        ),
        "actor_update_scheduled": actor_update_scheduled,
        "actor_update_attempted": actor_update_attempted,
        "actor_updated": actor_update_applied,
        "actor_update_skip_reason": actor_update_skip_reason,
        "actor_parameter_sha256_before": actor_parameter_sha256_before,
        "actor_parameter_sha256_after": actor_parameter_sha256_after,
        "actor_parameters_changed": (
            actor_parameter_sha256_before != actor_parameter_sha256_after
        ),
        "actor_update_count": bundle.actor_update_count,
        "actor_update_delay": config.actor_update_delay,
        "actor_update_interval": config.actor_update_interval,
        "online_actor_learning_rate": config.online_actor_learning_rate,
        "mean_actor_batch_action_delta": mean_actor_batch_action_delta,
        "maximum_actor_batch_action_delta": maximum_actor_batch_action_delta,
        "maximum_actor_batch_action_delta_limit": config.maximum_actor_batch_action_delta,
        "online_positive_advantage_selected_fraction": selected_online_fraction,
        "eligible_online_sample_count": eligible_online_sample_count,
        "selected_online_sample_count": selected_online_sample_count,
        "actor_evidence_sampling": "eligible_mode_stratified_without_replacement",
        "actor_evidence_prior_count": actor_evidence_prior_count,
        "actor_evidence_online_count": actor_evidence_online_count,
        "actor_evidence_raw_count_by_mode": {
            name: int(actor_evidence_raw_counts[index])
            for index, name in enumerate(CONTACT_MODE_NAMES)
        },
        "actor_evidence_qualified_modes": [
            CONTACT_MODE_NAMES[int(index)] for index in actor_evidence_qualified_modes
        ],
        "actor_evidence_recovery_mode_count": actor_evidence_recovery_mode_count,
        "minimum_online_actor_evidence_per_mode": (
            config.minimum_online_actor_evidence_per_mode
        ),
        "minimum_online_actor_recovery_modes": (
            config.minimum_online_actor_recovery_modes
        ),
        "actor_evidence_selected_count_by_mode": {
            name: int(actor_evidence_selected_mode_counts[index])
            for index, name in enumerate(CONTACT_MODE_NAMES)
        },
        "minimum_online_advantage": config.minimum_online_advantage,
        "eligible_online_advantage_minimum": eligible_online_advantage_minimum,
        "eligible_online_advantage_mean": eligible_online_advantage_mean,
        "eligible_online_advantage_maximum": eligible_online_advantage_maximum,
        "mean_selected_online_advantage_weight": mean_selected_online_weight,
        "maximum_selected_online_advantage_weight": maximum_selected_online_weight,
        "effective_online_sample_size": effective_online_sample_size,
        "selected_proposal_execution_l2_mean": selected_proposal_execution_l2_mean,
        "selected_proposal_execution_l2_maximum": selected_proposal_execution_l2_maximum,
        "mean_replay_advantage": (
            float(replay_advantage.mean().cpu())
            if replay_advantage.numel() > 0
            else 0.0
        ),
        "replay_advantage_observed": bool(replay_advantage.numel() > 0),
        "critic_gradient_norm": float(torch.as_tensor(critic_gradient).detach().cpu()),
        "actor_gradient_norm": float(torch.as_tensor(actor_gradient).detach().cpu()),
        "mean_target_q": float(target.mean().detach().cpu()),
        "mean_mode_entropy": float(
            (-(current_probability * current_log_probability).sum(-1)).mean().detach().cpu()
        ),
        "prior_sample_count": prior_count,
        "online_sample_count": int(config.batch_size - prior_count),
        "exact_symmetric_sampling": prior_count * 2 == config.batch_size,
        "intervention_fraction": float(batch["intervention"].to(torch.float32).mean().cpu()),
        "critic_conditions_on_policy_proposal": True,
        "shield_substitution_used_as_action_label": False,
        "raw_q_gradient_actor_objective_used": False,
        "actor_update_rule": "positive_advantage_weighted_regression_with_prior_anchor",
        "production_admission": False,
    }


@dataclass
class PersistentModeState:
    mode: int = -1
    remaining_steps: int = 0


class PersistentModeSampler:
    """Hold a sampled mode long enough to express a contact regime."""

    def __init__(self, minimum_steps: int = 4, maximum_steps: int = 12) -> None:
        if (
            type(minimum_steps) is not int
            or type(maximum_steps) is not int
            or minimum_steps < 1
            or maximum_steps < minimum_steps
        ):
            raise ValueError("persistent mode duration is invalid")
        self.minimum_steps = minimum_steps
        self.maximum_steps = maximum_steps

    def choose(
        self,
        probabilities: np.ndarray,
        state: PersistentModeState,
        rng: np.random.Generator,
        *,
        deterministic: bool,
    ) -> tuple[int, PersistentModeState, bool]:
        value = np.asarray(probabilities, dtype=np.float64)
        if (
            value.shape != (CONTACT_MODE_COUNT,)
            or not np.all(np.isfinite(value))
            or np.any(value < 0.0)
            or not np.isclose(value.sum(), 1.0, atol=1.0e-6, rtol=0.0)
        ):
            raise ValueError("mode probabilities are invalid")
        if state.remaining_steps > 0:
            if not 0 <= state.mode < CONTACT_MODE_COUNT:
                raise ValueError("persistent mode state is invalid")
            return state.mode, PersistentModeState(state.mode, state.remaining_steps - 1), False
        mode = int(np.argmax(value)) if deterministic else int(rng.choice(CONTACT_MODE_COUNT, p=value))
        duration = self.maximum_steps if deterministic else int(
            rng.integers(self.minimum_steps, self.maximum_steps + 1)
        )
        return mode, PersistentModeState(mode, duration - 1), True


def sample_epsilon_mixed_contact_mode(
    actor_probabilities: np.ndarray,
    allowed_mode_mask: np.ndarray,
    *,
    exploration_fraction: float,
    rng: np.random.Generator,
    deterministic: bool,
    sample_actor_categorical: bool = True,
) -> tuple[int, np.ndarray, str]:
    """Sample a structurally valid mode from actor/uniform mixture.

    The uniform component is restricted to modes declared available by the
    observation.  It therefore repairs categorical collapse without allowing
    transport/recovery modes during pre-contact or hold modes before target
    entry.  Deterministic evaluation is intentionally actor-only.
    """

    probabilities = np.asarray(actor_probabilities, dtype=np.float64)
    mask = np.asarray(allowed_mode_mask, dtype=np.bool_)
    if (
        probabilities.shape != (CONTACT_MODE_COUNT,)
        or mask.shape != (CONTACT_MODE_COUNT,)
        or not np.all(np.isfinite(probabilities))
        or np.any(probabilities < 0.0)
        or not np.any(mask)
        or not np.isfinite(exploration_fraction)
        or not 0.0 <= exploration_fraction <= 1.0
        or type(sample_actor_categorical) is not bool
    ):
        raise ValueError("epsilon-mixed contact-mode arguments are invalid")
    masked_probability = np.where(mask, probabilities, 0.0)
    if float(np.sum(probabilities[~mask])) > 1.0e-6:
        raise ValueError("actor assigns probability to a structurally masked mode")
    probability_sum = float(masked_probability.sum())
    if probability_sum <= 0.0:
        raise ValueError("actor has no probability on an available contact mode")
    masked_probability /= probability_sum

    if deterministic:
        return (
            int(np.argmax(masked_probability)),
            masked_probability,
            "deterministic_actor",
        )

    actor_behavior_probability = masked_probability.copy()
    if not sample_actor_categorical:
        actor_behavior_probability.fill(0.0)
        actor_behavior_probability[int(np.argmax(masked_probability))] = 1.0
    uniform_probability = mask.astype(np.float64) / float(mask.sum())
    behavior_probability = (
        (1.0 - exploration_fraction) * actor_behavior_probability
        + exploration_fraction * uniform_probability
    )
    behavior_probability /= float(behavior_probability.sum())
    if exploration_fraction > 0.0 and float(rng.random()) < exploration_fraction:
        mode = int(rng.choice(np.flatnonzero(mask)))
        source = "uniform_allowed_exploration"
    elif not sample_actor_categorical:
        mode = int(np.argmax(masked_probability))
        source = "actor_argmax"
    else:
        mode = int(rng.choice(CONTACT_MODE_COUNT, p=masked_probability))
        source = "actor_categorical"
    return mode, behavior_probability, source


def hybrid_bundle_state_dict(
    bundle: HybridContactSACBundle,
    config: HybridContactSACConfig,
) -> dict[str, Any]:
    return {
        "format": HYBRID_CONTACT_SAC_FORMAT,
        "configuration": asdict(config),
        "actor": bundle.actor.state_dict(),
        "reference_actor": bundle.reference_actor.state_dict(),
        "critic": bundle.critic.state_dict(),
        "target_critic": bundle.target_critic.state_dict(),
        "actor_optimizer": bundle.actor_optimizer.state_dict(),
        "critic_optimizer": bundle.critic_optimizer.state_dict(),
        "update_index": bundle.update_index,
        "actor_update_count": bundle.actor_update_count,
        "production_admission": False,
        "wrist_multimodal_export_started": False,
        "act_training_started": False,
    }


def load_hybrid_bundle_state_dict(
    bundle: HybridContactSACBundle,
    state: dict[str, Any],
    *,
    config: HybridContactSACConfig,
    load_actor_optimizer: bool = False,
    load_critic_optimizer: bool = True,
) -> None:
    """Restore one compatible checkpoint without silently changing topology."""

    if state.get("format") != HYBRID_CONTACT_SAC_FORMAT:
        raise ValueError("hybrid actor-critic checkpoint format is invalid")
    saved_config = state.get("configuration")
    if not isinstance(saved_config, dict):
        raise ValueError("hybrid actor-critic checkpoint configuration is missing")
    structural_fields = (
        "observation_dim",
        "mode_mask_start",
        "hidden_dim",
        "residual_scale",
        "exploration_log_std",
    )
    current_config = asdict(config)
    if any(saved_config.get(name) != current_config[name] for name in structural_fields):
        raise ValueError("hybrid actor-critic checkpoint topology differs")
    bundle.actor.load_state_dict(state["actor"], strict=True)
    bundle.critic.load_state_dict(state["critic"], strict=True)
    bundle.target_critic.load_state_dict(state["target_critic"], strict=True)
    if "reference_actor" in state:
        bundle.reference_actor.load_state_dict(state["reference_actor"], strict=True)
    else:
        synchronize_reference_actor(bundle)
    if load_actor_optimizer:
        bundle.actor_optimizer.load_state_dict(copy.deepcopy(state["actor_optimizer"]))
    if load_critic_optimizer:
        bundle.critic_optimizer.load_state_dict(copy.deepcopy(state["critic_optimizer"]))
    update_index = state.get("update_index")
    actor_update_count = state.get("actor_update_count", 0)
    if (
        type(update_index) is not int
        or update_index < 0
        or type(actor_update_count) is not int
        or actor_update_count < 0
    ):
        raise ValueError("hybrid actor-critic checkpoint counters are invalid")
    bundle.update_index = update_index
    bundle.actor_update_count = actor_update_count
    bundle.reference_actor.eval()
    for parameter in bundle.reference_actor.parameters():
        parameter.requires_grad_(False)


__all__ = [
    "CONTACT_MODE_COUNT",
    "CONTACT_MODE_NAMES",
    "DEFAULT_MODE_ACTION_CENTRES",
    "HYBRID_CONTACT_SAC_FORMAT",
    "HYBRID_REPLAY_FORMAT",
    "LEGACY_HYBRID_REPLAY_FORMAT",
    "TASK_ACTION_DIM",
    "HybridContactActor",
    "HybridContactSACBundle",
    "HybridContactSACConfig",
    "HybridReplayBuffer",
    "PersistentModeSampler",
    "PersistentModeState",
    "SymmetricPriorOnlineReplay",
    "add_intervention_surrogate",
    "hybrid_bundle_state_dict",
    "initialize_hybrid_contact_sac",
    "load_hybrid_bundle_state_dict",
    "pretrain_hybrid_actor_from_prior",
    "sample_epsilon_mixed_contact_mode",
    "synchronize_reference_actor",
    "update_hybrid_contact_sac",
]
