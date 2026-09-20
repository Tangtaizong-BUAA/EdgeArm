"""Continuous Command-V2 residual SAC core. No simulator access in actor."""

from copy import deepcopy
import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions import Normal

from .act_residual_ppo_run33 import ResidualHead


class FixedNormalizer(nn.Module):
    """Offline-fit scale without erasing channels constant in the seed data."""

    def __init__(self, dim):
        super().__init__()
        self.register_buffer("mean", torch.zeros(dim))
        self.register_buffer("inverse_scale", torch.ones(dim))

    def fit(self, x):
        with torch.no_grad():
            self.mean.copy_(x.mean(0))
            self.inverse_scale.copy_(x.std(0, unbiased=False).clamp_min(0.05).reciprocal())

    def forward(self, x):
        return ((x - self.mean) * self.inverse_scale).clamp(-10, 10)


class FrozenReadout(nn.Module):
    """Same frozen ACT+Run33 anchor as the Run37 continue baseline."""

    def __init__(self, base, anchor):
        super().__init__()
        self.base = base.eval().requires_grad_(False)
        self.config = base.config
        self.anchor = ResidualHead(self.config.model_dim + 18)
        self.anchor.load_state_dict(anchor["head"])
        self.anchor.eval().requires_grad_(False)
        self.limit = float(anchor["limit"])
        self.feature_dim = 2 * self.config.model_dim + 37

    def forward(self, inputs):
        memory, padding, _ = self.base.policy.encode_observations(inputs)
        actions, decoded = self.base.policy.decode_observations(memory, padding)
        h = torch.cat((decoded[:, 0], inputs["robot_state"], actions[:, 0]), -1)
        dist, _ = self.anchor.distribution(h)
        b = (actions[:, 0] + self.limit * dist.mean.tanh()).clamp(-1, 1)
        valid = ~padding
        pooled = (memory * valid[..., None]).sum(1) / valid.sum(1, keepdim=True).clamp_min(1)
        # History is already shifted by make_sample: last completed step is -1.
        mask = inputs["command_feedback_mask"][:, -1, None]
        feedback = torch.where(mask, inputs["tracking_error_history"][:, -1], 0.0)
        command = torch.where(mask, inputs["command_history"][:, -1], 0.0)
        x = torch.cat((h, pooled, b, feedback, command, mask.to(b.dtype)), -1)
        return x.detach(), b.detach()


class ContinuousActor(nn.Module):
    def __init__(self, feature_dim, width=256):
        super().__init__()
        self.normalizer = FixedNormalizer(feature_dim)
        self.net = nn.Sequential(nn.Linear(feature_dim, width), nn.LayerNorm(width), nn.SiLU(),
                                 nn.Linear(width, width), nn.LayerNorm(width), nn.SiLU())
        self.mean = nn.Linear(width, 6)
        self.logstd = nn.Linear(width, 6)
        nn.init.zeros_(self.mean.weight)
        nn.init.zeros_(self.mean.bias)
        nn.init.zeros_(self.logstd.weight)
        nn.init.constant_(self.logstd.bias, math.log(0.05))

    def forward(self, x, base, *, deterministic=False):
        h = self.net(self.normalizer(x))
        location = torch.atanh(base.clamp(-0.9999, 0.9999)) + self.mean(h)
        dist = Normal(location, self.logstd(h).clamp(-5, -1).exp())
        raw = location if deterministic else dist.rsample()
        action = raw.tanh()
        # Stable tanh Jacobian, including saturated outputs.
        log_jacobian = 2 * (math.log(2) - raw - F.softplus(-2 * raw))
        logp = (dist.log_prob(raw) - log_jacobian).sum(-1)
        return action, logp


class QEnsemble(nn.Module):
    def __init__(self, feature_dim, privileged_dim, width=256, count=4):
        super().__init__()
        self.xnorm = FixedNormalizer(feature_dim)
        self.pnorm = FixedNormalizer(privileged_dim)
        self.networks = nn.ModuleList([
            nn.Sequential(nn.Linear(feature_dim + privileged_dim + 6, width), nn.LayerNorm(width), nn.SiLU(),
                          nn.Linear(width, width), nn.LayerNorm(width), nn.SiLU(), nn.Linear(width, 1))
            for _ in range(count)
        ])

    def forward(self, x, privileged, action):
        h = torch.cat((self.xnorm(x), self.pnorm(privileged), action), -1)
        return torch.stack([net(h).squeeze(-1) for net in self.networks])


FIELDS = ("x", "base", "privileged", "action", "reward", "next_x", "next_base", "next_privileged", "terminal")


class EpisodeReplay:
    """Immutable completed fragments; sample episode uniformly, then a row.

    Fragment cuts never become terminals. CPU storage; transfer batches only.
    """

    def __init__(self):
        self.episodes = {}
        self.rows = 0
        self._cache = None

    def add(self, episode_id, batch):
        if set(batch) != set(FIELDS):
            raise ValueError("replay schema mismatch")
        arrays = {k: np.asarray(batch[k], np.float32) for k in FIELDS}
        n = len(arrays["reward"])
        if not n or any(len(a) != n or not np.isfinite(a).all() for a in arrays.values()):
            raise ValueError("empty, nonfinite or misaligned replay")
        if np.any(np.abs(arrays["action"]) > 1.00001):
            raise ValueError("replay action is not submitted normalized command")
        if not np.isin(arrays["terminal"], [0, 1]).all():
            raise ValueError("terminal must be binary")
        self.episodes.setdefault(episode_id, []).append({k: v.copy() for k, v in arrays.items()})
        self.rows += n
        self._cache = None

    def sample_tensor(self, n, device):
        # Rebuild only when new data arrive, not at every gradient update.
        # The pilot's <=60k transitions fit easily beside the frozen encoders.
        if self._cache is None:
            episodes = [{k: np.concatenate([c[k] for c in chunks]) for k in FIELDS}
                        for chunks in self.episodes.values()]
            if not episodes:
                raise ValueError("empty replay")
            lengths = np.array([len(e["reward"]) for e in episodes])
            self._cache = (
                {k: torch.as_tensor(np.concatenate([e[k] for e in episodes]), device=device) for k in FIELDS},
                torch.as_tensor(lengths, device=device),
                torch.as_tensor(np.r_[0, lengths.cumsum()[:-1]], device=device),
            )
        data, sizes, starts = self._cache
        ids = torch.randint(len(sizes), (n,), device=device)
        rows = starts[ids] + (torch.rand(n, device=device) * sizes[ids]).long()
        return {k: v[rows] for k, v in data.items()}

    def sample(self, n, rng):
        keys = list(self.episodes)
        if not keys:
            raise ValueError("empty replay")
        selected = {k: [] for k in FIELDS}
        for _ in range(n):
            chunks = self.episodes[keys[int(rng.integers(len(keys)))]]
            sizes = [len(v["reward"]) for v in chunks]
            row = int(rng.integers(sum(sizes)))
            for chunk, size in zip(chunks, sizes):
                if row < size:
                    for k in FIELDS:
                        selected[k].append(chunk[k][row])
                    break
                row -= size
        return {k: np.asarray(v, np.float32) for k, v in selected.items()}


def mixed_batch(offline, online, size, rng, device):
    if size % 2 or size < 2:
        raise ValueError("even batch required for 50/50 sampling")
    a, b = offline.sample_tensor(size // 2, device), online.sample_tensor(size // 2, device)
    return {k: torch.cat((a[k], b[k])) for k in FIELDS}


def bellman_target(reward, terminal, next_q, next_logp, gamma=0.999, alpha=0.01):
    return reward + gamma * (1 - terminal) * (next_q - alpha * next_logp)


class SACLearner:
    def __init__(self, feature_dim, privileged_dim, device, *, alpha=0.01):
        self.actor = ContinuousActor(feature_dim).to(device)
        self.q = QEnsemble(feature_dim, privileged_dim).to(device)
        self.target = deepcopy(self.q).requires_grad_(False)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=3e-4)
        self.q_opt = torch.optim.Adam(self.q.parameters(), lr=3e-4)
        self.alpha, self.steps, self.actor_steps = alpha, 0, 0

    def initialize_normalizers(self, x, privileged):
        self.actor.normalizer.fit(x)
        self.q.xnorm.fit(x)
        self.q.pnorm.fit(privileged)
        self.target.load_state_dict(self.q.state_dict())

    def update(self, b, *, train_actor=True):
        with torch.no_grad():
            na, nlog = self.actor(b["next_x"], b["next_base"])
            subset = torch.randperm(len(self.target.networks), device=na.device)[:2]
            nq = self.target(b["next_x"], b["next_privileged"], na)[subset].min(0).values
            target = bellman_target(b["reward"], b["terminal"], nq, nlog, alpha=self.alpha)
        values = self.q(b["x"], b["privileged"], b["action"])
        qloss = F.mse_loss(values, target[None].expand_as(values))
        if not torch.isfinite(qloss):
            raise ValueError("nonfinite critic loss")
        self.q_opt.zero_grad(set_to_none=True)
        qloss.backward()
        nn.utils.clip_grad_norm_(self.q.parameters(), 10, error_if_nonfinite=True)
        self.q_opt.step()
        aloss = None
        if train_actor and self.steps % 4 == 0:
            self.q.requires_grad_(False)
            a, logp = self.actor(b["x"], b["base"])
            aloss = (self.alpha * logp - self.q(b["x"], b["privileged"], a).mean(0)).mean()
            if not torch.isfinite(aloss):
                raise ValueError("nonfinite actor loss")
            self.actor_opt.zero_grad(set_to_none=True)
            aloss.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), 1, error_if_nonfinite=True)
            self.actor_opt.step()
            self.q.requires_grad_(True)
            self.actor_steps += 1
        with torch.no_grad():
            for a, bparam in zip(self.target.parameters(), self.q.parameters()):
                a.lerp_(bparam, 0.005)
        self.steps += 1
        return dict(q_loss=float(qloss.detach()), actor_loss=None if aloss is None else float(aloss.detach()),
                    q_mean=float(values.detach().mean()), q_disagreement=float(values.detach().std(0).mean()),
                    target_mean=float(target.mean()), critic_updates=self.steps, actor_updates=self.actor_steps)
