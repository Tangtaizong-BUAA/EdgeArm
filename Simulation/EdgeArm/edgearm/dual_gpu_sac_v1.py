"""Run39 learner: packed independent critics, bounded replay, block telemetry.

Same six-dimensional Command-V2 SAC objective as Run38. This module never
imports a simulator. Compilation is optional; FP32 is the default.
"""

from collections import OrderedDict
from copy import deepcopy
import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .continuous_recovery_run38 import ContinuousActor, FixedNormalizer, QEnsemble, FIELDS, bellman_target


class PackedQ(nn.Module):
    """Four independent MLPs evaluated with batched GEMMs, not a Python loop.

    The ensemble axis is NOT a shared hidden layer. LayerNorm is over each
    critic's feature axis, with independent affine parameters.
    """

    def __init__(self, feature_dim, privileged_dim, width=256, count=4):
        super().__init__()
        self.xnorm = FixedNormalizer(feature_dim)
        self.pnorm = FixedNormalizer(privileged_dim)
        self.count = count
        reference = QEnsemble(feature_dim, privileged_dim, width, count)
        self.weights = nn.ParameterList([
            nn.Parameter(torch.stack([net[i].weight.T.detach().clone() for net in reference.networks]))
            for i in (0, 3, 6)
        ])
        self.biases = nn.ParameterList([
            nn.Parameter(torch.stack([net[i].bias.detach().clone() for net in reference.networks])[:, None])
            for i in (0, 3, 6)
        ])
        self.scales = nn.ParameterList([nn.Parameter(torch.ones(count, 1, width)) for _ in range(2)])
        self.shifts = nn.ParameterList([nn.Parameter(torch.zeros(count, 1, width)) for _ in range(2)])

    @torch.no_grad()
    def copy_from_run38(self, reference):
        self.xnorm.load_state_dict(reference.xnorm.state_dict())
        self.pnorm.load_state_dict(reference.pnorm.state_dict())
        for j, i in enumerate((0, 3, 6)):
            self.weights[j].copy_(torch.stack([n[i].weight.T for n in reference.networks]))
            self.biases[j].copy_(torch.stack([n[i].bias for n in reference.networks])[:, None])
        for j, i in enumerate((1, 4)):
            self.scales[j].copy_(torch.stack([n[i].weight for n in reference.networks])[:, None])
            self.shifts[j].copy_(torch.stack([n[i].bias for n in reference.networks])[:, None])

    def forward(self, x, privileged, action):
        h = torch.cat((self.xnorm(x), self.pnorm(privileged), action), -1)
        h = h.unsqueeze(0).expand(self.count, -1, -1)
        for i in range(2):
            h = torch.bmm(h, self.weights[i]) + self.biases[i]
            h = F.silu(F.layer_norm(h, (h.shape[-1],)) * self.scales[i] + self.shifts[i])
        return (torch.bmm(h, self.weights[2]) + self.biases[2]).squeeze(-1)


def sample_action(actor, x, base):
    """Normal.rsample + tanh without distribution argument-validation syncs."""
    h = actor.net(actor.normalizer(x))
    location = torch.atanh(base.clamp(-0.9999, 0.9999)) + actor.mean(h)
    logstd = actor.logstd(h).clamp(-5, -1)
    noise = torch.randn_like(location)
    raw = location + noise * logstd.exp()
    logp = -.5 * (noise.square() + math.log(2 * math.pi)) - logstd
    jacobian = 2 * (math.log(2) - raw - F.softplus(-2 * raw))
    return raw.tanh(), (logp - jacobian).sum(-1)


class DeviceReplay:
    """Fixed-capacity transition ring, still uniform-episode then uniform-row.

    Only new transitions cross PCIe. Eviction updates an index table; historical
    feature tensors are never concatenated/reuploaded after every fragment.
    Sampling tables are rebuilt once per ingestion wave, not per SGD update.
    """

    def __init__(self, capacity, device):
        if capacity <= 0:
            raise ValueError("positive replay capacity required")
        self.capacity, self.device = int(capacity), torch.device(device)
        self.data = None
        self.owners = [None] * self.capacity
        self.episodes = OrderedDict()
        self.cursor = self.rows = self.inserted = self.evicted = 0
        self._index = None

    def add(self, episode_id, batch):
        if set(batch) != set(FIELDS):
            raise ValueError("replay schema mismatch")
        arrays = {k: np.asarray(batch[k], dtype=np.float32) for k in FIELDS}
        n = len(arrays['reward'])
        if not n or n > self.capacity or any(len(v) != n or not np.isfinite(v).all() for v in arrays.values()):
            raise ValueError("invalid replay length/nonfinite transition")
        if arrays['action'].shape != (n, 6) or np.any(abs(arrays['action']) > 1.00001):
            raise ValueError("invalid Command-V2 action")
        if not np.isin(arrays['terminal'], (0, 1)).all():
            raise ValueError("terminal must be binary")
        if self.data is not None and any(v.shape[1:] != self.data[k].shape[1:] for k, v in arrays.items()):
            raise ValueError("replay shape changed")
        if self.data is None:
            self.data = {k: torch.empty((self.capacity, *v.shape[1:]), dtype=torch.float32, device=self.device)
                         for k, v in arrays.items()}
        indices = (np.arange(n) + self.cursor) % self.capacity
        for index in indices.tolist():
            previous = self.owners[index]
            if previous is not None:
                del self.episodes[previous][index]
                if not self.episodes[previous]:
                    del self.episodes[previous]
                self.evicted += 1
            self.owners[index] = episode_id
            self.episodes.setdefault(episode_id, OrderedDict())[index] = None
        ids = torch.as_tensor(indices, device=self.device)
        for key, array in arrays.items():
            self.data[key].index_copy_(0, ids, torch.as_tensor(array, device=self.device))
        self.cursor = int((self.cursor + n) % self.capacity)
        self.inserted += n
        self.rows = min(self.capacity, self.rows + n)
        self._index = None

    def sample_tensor(self, n, device=None):
        if device is not None and torch.device(device) != self.device:
            raise ValueError("device-resident replay cannot silently migrate")
        if n <= 0 or not self.rows:
            raise ValueError("empty replay or invalid batch")
        if self._index is None:
            sizes = np.asarray([len(rows) for rows in self.episodes.values()], np.int64)
            flat = np.fromiter((i for rows in self.episodes.values() for i in rows), dtype=np.int64)
            self._index = tuple(torch.as_tensor(v, device=self.device) for v in
                                (flat, sizes, np.r_[0, sizes.cumsum()[:-1]]))
        flat, sizes, starts = self._index
        episode = torch.randint(len(sizes), (n,), device=self.device)
        index = flat[starts[episode] + (torch.rand(n, device=self.device) * sizes[episode]).long()]
        return {k: value[index] for k, value in self.data.items()}

    @property
    def allocated_bytes(self):
        return 0 if self.data is None else sum(v.numel() * v.element_size() for v in self.data.values())


def replay_batch(offline, online, size):
    if size < 2 or size % 2:
        raise ValueError("even batch required for 50/50 replay")
    a, b = offline.sample_tensor(size // 2), online.sample_tensor(size // 2)
    return {k: torch.cat((a[k], b[k])) for k in FIELDS}


class FastSAC:
    def __init__(self, feature_dim, privileged_dim, device, *, alpha=.01, compile_mode='none', width=256):
        if compile_mode not in ('none', 'default', 'reduce-overhead'):
            raise ValueError("unsupported compile mode")
        self.device = torch.device(device)
        self.actor = ContinuousActor(feature_dim, width).to(self.device)
        self.q = PackedQ(feature_dim, privileged_dim, width).to(self.device)
        self.target = deepcopy(self.q).requires_grad_(False)
        fused = self.device.type == 'cuda'
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=3e-4, fused=fused)
        self.q_opt = torch.optim.Adam(self.q.parameters(), lr=3e-4, fused=fused)
        self.alpha = float(alpha)
        self.steps = self.actor_steps = 0
        self.metrics = torch.zeros(8, device=self.device)
        self.valid = torch.ones((), dtype=torch.bool, device=self.device)
        self.compile_mode = compile_mode
        self.critic_fn, self.actor_fn = self._critic_loss, self._actor_loss
        if compile_mode != 'none':
            # Separate graphs have stable requires_grad and actor-update cadence.
            # No fallback hides a compile failure: the run records and stops it.
            self.critic_fn = torch.compile(self._critic_loss, mode=compile_mode, fullgraph=True)
            self.actor_fn = torch.compile(self._actor_loss, mode=compile_mode, fullgraph=True)

    def initialize_normalizers(self, x, privileged):
        self.actor.normalizer.fit(x)
        self.q.xnorm.fit(x)
        self.q.pnorm.fit(privileged)
        self.target.load_state_dict(self.q.state_dict())

    def _critic_loss(self, b):
        with torch.no_grad():
            action, logp = sample_action(self.actor, b['next_x'], b['next_base'])
            subset = torch.randperm(self.target.count, device=self.device)[:2]
            value = self.target(b['next_x'], b['next_privileged'], action)[subset].amin(0)
            target = bellman_target(b['reward'], b['terminal'], value, logp, alpha=self.alpha)
        values = self.q(b['x'], b['privileged'], b['action'])
        loss = F.mse_loss(values, target[None].expand_as(values))
        return loss, torch.stack((values.detach().mean(), target.mean(), values.detach().std(0).mean()))

    def _actor_loss(self, b):
        action, logp = sample_action(self.actor, b['x'], b['base'])
        return (self.alpha * logp - self.q(b['x'], b['privileged'], action).mean(0)).mean()

    def update(self, b, *, train_actor=True):
        if self.compile_mode == 'reduce-overhead':
            torch.compiler.cudagraph_mark_step_begin()
        self.q_opt.zero_grad(set_to_none=True)
        loss, diagnostics = self.critic_fn(b)
        loss.backward()
        norm = nn.utils.clip_grad_norm_(self.q.parameters(), 10, foreach=True)
        self.valid.logical_and_(torch.isfinite(loss.detach()) & torch.isfinite(norm))
        self.q_opt.step()
        self.metrics[0].add_(loss.detach())
        self.metrics[1:4].add_(diagnostics)
        self.metrics[6].add_(1)
        if train_actor and self.steps % 4 == 0:
            self.q.requires_grad_(False)
            self.actor_opt.zero_grad(set_to_none=True)
            aloss = self.actor_fn(b)
            aloss.backward()
            anorm = nn.utils.clip_grad_norm_(self.actor.parameters(), 1, foreach=True)
            self.valid.logical_and_(torch.isfinite(aloss.detach()) & torch.isfinite(anorm))
            self.actor_opt.step()
            self.q.requires_grad_(True)
            self.metrics[4].add_(aloss.detach())
            self.metrics[5].add_(1)
            self.actor_steps += 1
        with torch.no_grad():
            torch._foreach_lerp_(list(self.target.parameters()), list(self.q.parameters()), .005)
        self.steps += 1

    def flush_metrics(self):
        # One transfer per reporting block, not float(cuda_tensor) every update.
        row = torch.cat((self.metrics, self.valid[None].to(self.metrics.dtype))).detach().cpu().numpy()
        self.metrics.zero_()
        if row[-1] != 1 or not np.isfinite(row).all():
            raise FloatingPointError("nonfinite learner block; do not publish this policy")
        count, acount = max(float(row[6]), 1), float(row[5])
        return dict(q_loss=float(row[0] / count), q_mean=float(row[1] / count),
                    target_mean=float(row[2] / count), q_disagreement=float(row[3] / count),
                    actor_loss=float(row[4] / acount) if acount else None,
                    block_critic_updates=int(row[6]), block_actor_updates=int(acount),
                    critic_updates=self.steps, actor_updates=self.actor_steps)
