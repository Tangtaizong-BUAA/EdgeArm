"""Prepared Run37 reward/credit/short-option core; no training auto-start.

The frozen ACT and anchor remain unchanged. Privileged task state is accepted
only by reward calculation, never by actor.forward. Stage classifiers are
optional supervised probes, not a hard-coded controller.
"""

from dataclasses import dataclass
import math

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

from .act_residual_ppo_run33 import ResidualHead


@dataclass(frozen=True)
class RewardConfig:
    gamma: float = 0.999
    distance_weight: float = 6.0
    coverage_weight: float = 1.0
    hold_weight: float = 4.0
    step_cost: float = 0.002
    success_bonus: float = 8.0
    hard_failure_cost: float = 2.0
    hold_seconds: float = 3.0

    def __post_init__(self):
        if not 0 < self.gamma <= 1 or self.hold_seconds <= 0:
            raise ValueError("invalid discount or hold duration")
        if not all(math.isfinite(v) for v in vars(self).values()):
            raise ValueError("nonfinite reward configuration")
        if any(v < 0 for k, v in vars(self).items() if k not in ("gamma", "hold_seconds")):
            raise ValueError("negative reward coefficient")


def potential(task_state, cfg=RewardConfig()):
    """Reward-only state [..., distance_m, coverage, consecutive_hold_s]."""
    s = np.asarray(task_state, np.float64)
    if s.shape[-1] != 3 or not np.isfinite(s).all():
        raise ValueError("finite reward-state triple required")
    if np.any(s[..., 0] < 0) or np.any((s[..., 1] < 0) | (s[..., 1] > 1)) or np.any(s[..., 2] < 0):
        raise ValueError("reward-state range")
    return (
        -cfg.distance_weight * s[..., 0]
        + cfg.coverage_weight * s[..., 1]
        + cfg.hold_weight * np.minimum(s[..., 2] / cfg.hold_seconds, 1)
    )


def shaped_rewards(states, *, end_kind, cfg=RewardConfig()):
    """Return per-control-step rewards plus separately auditable components.

    states has n+1 rows, including the state immediately before the first
    action. A real finite-horizon timeout is terminal. A sampler cut is not:
    retain its endpoint potential and bootstrap the shaped value separately.
    There is no reward for selecting the hold action as such.
    """
    if end_kind not in ("success", "hard_failure", "task_failure", "finite_timeout", "sampler_cut"):
        raise ValueError("explicit episode ending required")
    s = np.asarray(states, np.float64)
    if s.ndim != 2 or s.shape[1] != 3 or len(s) < 2:
        raise ValueError("need initial state plus at least one transition")
    phi = potential(s, cfg)
    after = phi[1:].copy()
    if end_kind != "sampler_cut":
        after[-1] = 0.0  # Absorbing terminal potential: preserves telescoping.
    shaping = cfg.gamma * after - phi[:-1]
    time = np.full(len(shaping), -cfg.step_cost)
    terminal = np.zeros(len(shaping))
    if end_kind == "success":
        terminal[-1] = cfg.success_bonus
    elif end_kind == "hard_failure":
        terminal[-1] = -cfg.hard_failure_cost
    parts = dict(potential=shaping, time=time, terminal=terminal)
    return sum(parts.values()).astype(np.float32), parts


def credit_batch(
    decisions, rewards, *, episode_terminal, bootstrap_value=None, gamma=0.999, lambda_per_eight_steps=0.95
):
    """Duration-aware GAE, with explicit cut/terminal distinction and audit data.

    Lambda is defined per eight physical control steps here. Run34 instead
    used one lambda per option: that was a different time scale, not a missing
    gamma**duration implementation.
    """
    if not 0 < gamma <= 1 or not 0 <= lambda_per_eight_steps <= 1:
        raise ValueError("invalid GAE factors")
    if episode_terminal:
        if bootstrap_value not in (None, 0, 0.0):
            raise ValueError("terminal episodes must not bootstrap")
        bootstrap_value = 0.0
    elif bootstrap_value is None or not math.isfinite(float(bootstrap_value)):
        raise ValueError("sampler cut requires an explicit shaped-value bootstrap")
    rewards = np.asarray(rewards, np.float64)
    if rewards.ndim != 1 or not np.isfinite(rewards).all():
        raise ValueError("finite per-control reward vector required")
    rows = [d for d in decisions if d["step"] < len(rewards)]
    if not rows:
        return None
    steps = [d["step"] for d in rows]
    if steps[0] < 0 or any(b <= a for a, b in zip(steps, steps[1:])):
        raise ValueError("strictly increasing decision steps required")
    x = torch.cat([d["features"] for d in rows]).clone()
    values = torch.cat([d["value"] for d in rows]).clone()
    advantages = torch.zeros_like(values)
    option_rewards, durations, td_errors = [], [], []
    carry = torch.zeros((), device=values.device)
    for i in reversed(range(len(rows))):
        begin = steps[i]
        end = steps[i + 1] if i + 1 < len(rows) else len(rewards)
        duration = end - begin
        discounted = float(np.dot(gamma ** np.arange(duration), rewards[begin:end]))
        next_value = values[i + 1] if i + 1 < len(rows) else bootstrap_value
        td = discounted + gamma**duration * next_value - values[i]
        carry = td + gamma**duration * lambda_per_eight_steps ** (duration / 8) * carry
        advantages[i] = carry
        durations.append(duration)
        option_rewards.append(discounted)
        td_errors.append(float(td))
    data = dict(
        x=x,
        options=torch.cat([d["option"] for d in rows]).clone(),
        old_logits=torch.cat([d["logits"] for d in rows]).clone(),
        advantages=advantages,
        returns=advantages + values,
    )
    audit = dict(
        steps=steps,
        durations=list(reversed(durations)),
        discounted_option_rewards=list(reversed(option_rewards)),
        values=values.detach().cpu().tolist(),
        td_errors=list(reversed(td_errors)),
        advantages=advantages.detach().cpu().tolist(),
        episode_terminal=episode_terminal,
        bootstrap_value=float(bootstrap_value),
    )
    return data, audit


class FeatureStandardizer(nn.Module):
    """Train-split-only statistics; constant features are not extrapolated."""

    def __init__(self, dim):
        super().__init__()
        self.register_buffer("mean", torch.zeros(dim))
        self.register_buffer("inverse_scale", torch.ones(dim))

    def fit(self, features):
        with torch.no_grad():
            self.mean.copy_(features.mean(0))
            std = features.std(0, unbiased=False)
            self.inverse_scale.copy_(torch.where(std > 1e-6, std.clamp_min(0.01).reciprocal(), 0))

    def forward(self, features):
        return ((features - self.mean) * self.inverse_scale).clamp(-10, 10)


class ReplanOptionACT(nn.Module):
    """Equal-duration continue/hold options with a trainable state readout.

    Default 8/8 decisions replace the 8/96 lock. Shorter hold alone can make
    sparse exploration harder, so a data/feature probe is required before
    scaling. It is NOT presented as a proven fix.
    """

    def __init__(self, base, anchor_state, limit, decision_steps=8, initial_hold_probability=0.5):
        super().__init__()
        if type(decision_steps) is not int or not 1 <= decision_steps <= 32:
            raise ValueError("decision interval must be an integer in [1,32]")
        if not 0 < initial_hold_probability < 1 or not 0 < limit <= 1:
            raise ValueError("invalid probability or anchor limit")
        self.base = base.eval().requires_grad_(False)
        self.config = base.config
        self.anchor = ResidualHead(self.config.model_dim + 18)
        self.anchor.load_state_dict(anchor_state)
        self.anchor.eval().requires_grad_(False)
        self.limit, self.decision_steps = limit, decision_steps
        dim = self.config.model_dim * 2 + 31

        def network(outputs):
            return nn.Sequential(FeatureStandardizer(dim), nn.LayerNorm(dim),
                                 nn.Linear(dim, 64), nn.Tanh(), nn.Linear(64, outputs))

        self.actor, self.critic = network(2), network(1)
        self.phase_head = nn.Linear(64, 3)
        nn.init.zeros_(self.actor[-1].weight)
        with torch.no_grad():
            self.actor[-1].bias.copy_(
                torch.tensor([0.0, math.log(initial_hold_probability / (1 - initial_hold_probability))])
            )
        self.explore = True
        self.reset_episode()

    def reset_episode(self, prefix=None):
        self.prefix = np.empty((0, 6), np.float32) if prefix is None else np.asarray(prefix, np.float32)
        if (
            self.prefix.ndim != 2
            or self.prefix.shape[1] != 6
            or not np.isfinite(self.prefix).all()
            or np.any(abs(self.prefix) > 1)
        ):
            raise ValueError("invalid normalized command prefix")
        self.step, self.next_decision, self.hold = 0, len(self.prefix), False
        self.decisions = []

    def phase_logits(self, causal_features):
        # Training-only labels supervise this readout. Labels never enter forward.
        return self.phase_head(self.actor[:-1](causal_features))

    def causal_readout(self, inputs, previous_hold=False):
        """Stateless frozen readout, also used by same-state branch experiments.

        Does not consume a policy decision, future command, or task geometry.
        Keeping this identical in sampling and deployment avoids a probe-only
        feature path. ``previous_hold`` is known controller history.
        """
        memory, padding, _ = self.base.policy.encode_observations(inputs)
        actions, decoded = self.base.policy.decode_observations(memory, padding)
        anchor_features = torch.cat([decoded[:, 0], inputs["robot_state"], actions[:, 0]], -1)
        distribution, _ = self.anchor.distribution(anchor_features)
        actions = actions.clone()
        actions[:, 0] = (actions[:, 0] + self.limit * distribution.mean.tanh()).clamp(-1, 1)
        valid = ~padding
        summary = (memory * valid[..., None]).sum(1) / valid.sum(1, keepdim=True).clamp_min(1)
        feedback = torch.where(
            inputs["command_feedback_mask"][:, -2, None], inputs["tracking_error_history"][:, -2], 0.0
        )
        features = torch.cat(
            [anchor_features, summary, actions[:, 0], feedback,
             torch.full_like(inputs["robot_state"][:, :1], float(previous_hold))], -1
        ).detach()
        return actions, features

    def forward(self, inputs, return_aux=False):
        device = inputs["robot_state"].device
        if self.step < len(self.prefix):
            actions = torch.zeros((1, self.config.action_chunk_size, 6), device=device)
            actions[:, 0] = torch.as_tensor(self.prefix[self.step], device=device)
        elif self.hold and self.step < self.next_decision:
            actions = torch.zeros((1, self.config.action_chunk_size, 6), device=device)
        else:
            actions, features = self.causal_readout(inputs, self.hold)
            if self.step >= self.next_decision:
                logits = self.actor(features)
                policy = Categorical(logits=logits)
                option = policy.sample() if self.explore else logits.argmax(-1)
                self.hold = bool(option.item())
                self.next_decision = self.step + self.decision_steps
                self.decisions.append(
                    dict(
                        step=self.step,
                        features=features.detach(),
                        option=option.detach(),
                        logits=logits.detach(),
                        value=self.critic(features).squeeze(-1).detach(),
                    )
                )
            if self.hold:
                actions[:, 0] = 0
        self.step += 1
        return {"action": actions} if return_aux else actions
