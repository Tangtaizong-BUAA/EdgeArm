"""Behavior-rebased, demonstration-curriculum RL on frozen Run48 ACT features.

The deployable actor has no task index, simulator coordinates or teacher action
input. A supervised bridge can change the ACT behavior substantially; only the
subsequent RL residual is limited to +/-0.12 around that learned behavior.
"""
from copy import deepcopy

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .constrained_recovery_run40 import BoundedActor, CalibratedLearner, CompleteReplay
from .continuous_recovery_run38 import FixedNormalizer


class BehaviorBridge(nn.Module):
    def __init__(self, feature_dim, width=256):
        super().__init__()
        self.normalizer = FixedNormalizer(feature_dim)
        self.net = nn.Sequential(nn.Linear(feature_dim, width), nn.LayerNorm(width), nn.SiLU(),
                                 nn.Linear(width, width), nn.LayerNorm(width), nn.SiLU())
        self.mean = nn.Linear(width, 6)
        nn.init.zeros_(self.mean.weight)
        nn.init.zeros_(self.mean.bias)

    def forward(self, x, base):
        # Any legal command can be represented. This is a learned behavior
        # migration, NOT an unbounded RL exploration amplitude.
        return (base + 2 * self.mean(self.net(self.normalizer(x))).tanh()).clamp(-1, 1)


class RebasedActor(nn.Module):
    def __init__(self, feature_dim, residual_limit=.12):
        super().__init__()
        self.bridge = BehaviorBridge(feature_dim)
        self.residual = BoundedActor(feature_dim, residual_limit=residual_limit)
        self.register_buffer('bridge_strength', torch.tensor(1.))

    @property
    def residual_limit(self):
        return self.residual.residual_limit

    def anchor(self, x, base):
        bridge = self.bridge(x, base)
        return (base + self.bridge_strength * (bridge - base)).clamp(-1, 1)

    def forward(self, x, base, *, noise_std=0., raw_noise=None):
        anchor = self.anchor(x, base)
        raw = self.residual.mean(self.residual.net(self.residual.normalizer(x)))
        if raw_noise is not None:
            raw = raw + raw_noise
        elif noise_std:
            raw = raw + float(noise_std) * torch.randn_like(raw)
        return (anchor + self.residual_limit * raw.tanh()).clamp(-1, 1)

    def freeze_bridge(self):
        self.bridge.requires_grad_(False)


class RouteReplay(CompleteReplay):
    """Equal route, then equal present phase; sample actual recorded rows.

    Geometry labels are for data balancing only, never concatenated to x.
    Short collision episodes retain sampling weight instead of disappearing
    among 900-step timeout and 90-step hold tails.
    """
    def __init__(self, device):
        super().__init__(device)
        self.routes = []
        self.phase_codes = []
        self.buckets = None

    def add(self, episode_id, batch, *, teacher=False, route):
        if route not in range(9):
            raise ValueError('invalid route metadata')
        super().add(episode_id, batch, teacher=teacher)
        n = len(batch['reward'])
        phase = np.where(np.asarray(batch['privileged'])[:, -4] > 0, 2,
                         np.where(np.arange(n) < max(1, n // 3), 0, 1))
        self.routes.extend([route] * n)
        self.phase_codes.extend(phase.tolist())
        self.buckets = None

    def sample(self, size, *, terminal_fraction=.125, routes=None):
        data = self.materialize()
        if self.buckets is None:
            r, p = np.asarray(self.routes), np.asarray(self.phase_codes)
            self.buckets = {int(route): [torch.as_tensor(np.flatnonzero((r == route) & (p == phase)),
                                                       device=self.device)
                                         for phase in np.unique(p[r == route])]
                            for route in np.unique(r)}
        keys = sorted(self.buckets) if routes is None else [r for r in routes if r in self.buckets]
        if not keys:
            raise ValueError('no training data for requested routes')
        pieces = []
        route_tensor = torch.as_tensor(self.routes, device=self.device)
        for i, route in enumerate(keys):
            count = size // len(keys) + (i < size % len(keys))
            buckets = self.buckets[route]
            route_pieces = []
            for j, rows in enumerate(buckets):
                n = count // len(buckets) + (j < count % len(buckets))
                route_pieces.append(rows[torch.randint(len(rows), (n,), device=self.device)])
            selected_route = torch.cat(route_pieces)
            nterm = int(count * terminal_fraction)
            if nterm:
                eligible = self.terminals[route_tensor[self.terminals] == route]
                selected_route[-nterm:] = eligible[torch.randint(len(eligible), (nterm,), device=self.device)]
            pieces.append(selected_route)
        selected = torch.cat(pieces)
        return {k: v[selected] for k, v in data.items()}


class RebasedLearner(CalibratedLearner):
    def __init__(self, feature_dim, privileged_dim, device):
        super().__init__(feature_dim, privileged_dim, device)
        self.actor = RebasedActor(feature_dim).to(device)
        self.target_actor = deepcopy(self.actor).requires_grad_(False)
        self.bridge_opt = torch.optim.Adam(self.actor.bridge.parameters(), lr=1e-4,
                                          fused=self.device.type == 'cuda')
        self.actor_opt = torch.optim.Adam(self.actor.residual.parameters(), lr=1e-5,
                                         fused=self.device.type == 'cuda')
        self.bridge_steps = 0

    def fit_normalizers(self, data):
        self.actor.bridge.normalizer.fit(data['x'])
        self.actor.residual.normalizer.fit(data['x'])
        self.q.xnorm.fit(data['x'])
        self.q.pnorm.fit(data['privileged'])
        self.sync_targets()

    def bridge_update(self, teacher):
        if not torch.all(teacher['bc_mask'] == 1):
            raise ValueError('failed action is not a BC demonstration')
        action = self.actor.bridge(teacher['x'], teacher['base'])
        loss = F.smooth_l1_loss(action / .05, teacher['action'] / .05)
        self.bridge_opt.zero_grad(set_to_none=True)
        loss.backward()
        norm = nn.utils.clip_grad_norm_(self.actor.bridge.parameters(), 1)
        if not torch.isfinite(loss) or not torch.isfinite(norm):
            raise FloatingPointError('nonfinite behavior bridge')
        self.bridge_opt.step()
        self.bridge_steps += 1
        return float(loss.detach())

    def actor_update(self, b, teacher, *, bc_only=False):
        self.actor.freeze_bridge()
        self.q.requires_grad_(False)
        action = self.actor(b['x'], b['base'])
        anchor = self.actor.anchor(b['x'], b['base']).detach()
        ta = self.actor(teacher['x'], teacher['base'])
        tanchor = self.actor.anchor(teacher['x'], teacher['base']).detach()
        scale = self.actor.residual_limit
        mask = ((teacher['action'] - tanchor).abs().amax(-1) <= scale + 1e-6).float()
        bc = ((((ta - teacher['action']) / scale).square().mean(-1)) * mask).sum() / mask.sum().clamp_min(1)
        trust = ((action - anchor) / scale).square().mean()
        with torch.no_grad():
            qanchor = self.q(b['x'], b['privileged'], anchor).amin(0)
        q = self.q(b['x'], b['privileged'], action).amin(0)
        advantage = (q - qanchor) / qanchor.abs().mean().clamp_min(1)
        loss = (0 if bc_only else -advantage.mean()) + .5 * bc + .1 * trust
        self.actor_opt.zero_grad(set_to_none=True)
        loss.backward()
        norm = nn.utils.clip_grad_norm_(self.actor.residual.parameters(), 1)
        if not torch.isfinite(loss) or not torch.isfinite(norm):
            raise FloatingPointError('nonfinite bounded RL residual')
        self.actor_opt.step()
        self.q.requires_grad_(True)
        self.actor_steps += 1
        self.metric_sum[3].add_(loss.detach())
        self.metric_sum[4].add_(bc.detach())
        self.metric_sum[5].add_(1)
        with torch.no_grad():
            for a, btarget in zip(self.actor.residual.parameters(), self.target_actor.residual.parameters()):
                btarget.lerp_(a, .005)
        self.last_bc_reachable_fraction = float(mask.mean())

    @torch.inference_mode()
    def audit(self, replay):
        result = super().audit(replay)
        result['maximum_action_delta_from_ACT'] = result.pop('maximum_action_delta')
        data = replay.materialize()
        residual = torch.zeros((), device=self.device)
        for begin in range(0, replay.rows, 512):
            x, base = data['x'][begin:begin+512], data['base'][begin:begin+512]
            residual = torch.maximum(residual, (self.actor(x, base)-self.actor.anchor(x, base)).abs().max())
        result['maximum_RL_delta_from_learned_anchor'] = float(residual)
        if residual > self.actor.residual_limit + 1e-6:
            raise ValueError('RL residual exceeded learned-anchor bound')
        return result


def acceptable(candidate, baseline, *, routes=None):
    """No loss of any already-successful route, no additional hard failures."""
    keys = list(baseline['pairs']) if routes is None else [str(r) for r in routes]
    return (candidate['hard_failures'] <= baseline['hard_failures']
            and candidate['successes'] >= baseline['successes']
            and candidate['mean_coverage'] >= .9 * baseline['mean_coverage']
            and all(candidate['pairs'][k]['successes'] >= baseline['pairs'][k]['successes'] for k in keys))
