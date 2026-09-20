"""One shared policy: fused gradient AllReduce BEFORE clipping/Adam.

Each rank owns a shard of genuine experience. Both optimizers have identical
states and step on the same global gradient; this is data parallel learning,
not independent candidates. Frozen visual encoders are not optimized.
"""

from collections import deque
from copy import deepcopy

import numpy as np
import torch
import torch.distributed as dist
from torch import nn
from torch.nn import functional as F

from ..constrained_recovery_run40 import CalibratedLearner, complete_episode_targets
from .domain import CONTEXT_DIM, add_context


def world_size():
    return dist.get_world_size() if dist.is_initialized() else 1


def global_mean(value):
    result = value.detach().clone()
    if dist.is_initialized():
        dist.all_reduce(result)
        result /= world_size()
    return result


def average_gradients(module):
    parameters = [p for p in module.parameters() if p.requires_grad]
    if any(p.grad is None for p in parameters):
        raise ValueError('missing gradient: ranks must optimize the same complete parameter set')
    flat = torch.cat([p.grad.reshape(-1) for p in parameters])
    if dist.is_initialized():
        dist.all_reduce(flat, op=dist.ReduceOp.SUM)
        flat /= world_size()
    offset = 0
    for p in parameters:
        p.grad.copy_(flat[offset:offset + p.numel()].view_as(p))
        offset += p.numel()


def expand_critic_state(state, feature_dim, old_privileged_dim):
    result = deepcopy(state)
    for key, fill in [('pnorm.mean', 0), ('pnorm.inverse_scale', 1)]:
        value = state[key]
        result[key] = torch.cat((value[:-6], value.new_full((CONTEXT_DIM,), fill), value[-6:]))
    w = state['weights.0']
    split = feature_dim + old_privileged_dim - 6
    result['weights.0'] = torch.cat((w[:, :split], w.new_zeros((w.shape[0], CONTEXT_DIM, w.shape[2])), w[:, split:]), dim=1)
    return result


def expand_reference(batch):
    b = dict(batch)
    for key in ('privileged', 'next_privileged'):
        b[key] = add_context(b[key], np.zeros(CONTEXT_DIM, np.float32))
    return b


class DistributedLearner(CalibratedLearner):
    def __init__(self, saved, device, *, compile_mode='none'):
        f, p = saved['feature_dim'], saved['privileged_dim']
        super().__init__(f, p + CONTEXT_DIM, device)
        self.actor.load_state_dict(saved['actor'])
        self.q.load_state_dict(expand_critic_state(saved['critic'], f, p))
        self.sync_targets()
        self.q_forward, self.actor_forward = self.q, self.actor
        if compile_mode != 'none':
            self.q_forward = torch.compile(self.q, mode=compile_mode)
            self.actor_forward = torch.compile(self.actor, mode=compile_mode)

    def critic_update(self, b, *, calibration=False):
        with torch.no_grad():
            if calibration:
                target = b['mc_return']
            else:
                action = self.target_actor(b['boot_x'], b['boot_base'], noise_std=.1)
                target = b['n_reward'] + b['n_discount'] * self.target_q(b['boot_x'], b['boot_privileged'], action).amin(0)
        values = self.q_forward(b['x'], b['privileged'], b['action'])
        td = F.smooth_l1_loss(values, target[None].expand_as(values))
        anchor = F.smooth_l1_loss(values, b['mc_return'][None].expand_as(values))
        loss = td if calibration else td + .25 * anchor
        self.q_opt.zero_grad(set_to_none=True)
        loss.backward()
        average_gradients(self.q)
        norm = nn.utils.clip_grad_norm_(self.q.parameters(), 5, foreach=True)
        self.valid.logical_and_(torch.isfinite(loss.detach()) & torch.isfinite(norm))
        self.q_opt.step()
        self.critic_steps += 1
        self.metric_sum[0].add_(loss.detach())
        self.metric_sum[1].add_(values.detach().mean())
        self.metric_sum[2].add_(1)
        with torch.no_grad():
            torch._foreach_lerp_(list(self.target_q.parameters()), list(self.q.parameters()), .005)

    def actor_update(self, b, teacher, *, bc_only=False):
        self.q.requires_grad_(False)
        action = self.actor_forward(b['x'], b['base'])
        a_teacher = self.actor_forward(teacher['x'], teacher['base'])
        bc = ((a_teacher - teacher['action']) / self.actor.residual_limit).square().mean()
        anchor = ((action - b['base']) / self.actor.residual_limit).square().mean()
        if bc_only:
            loss = bc + .05 * anchor
        else:
            with torch.no_grad():
                base = self.q(b['x'], b['privileged'], b['base']).amin(0)
                normalizer = global_mean(base.abs().mean()).clamp_min(1.)
            value = self.q(b['x'], b['privileged'], action).amin(0)
            loss = -((value - base) / normalizer).mean() + .5 * bc + .1 * anchor
        self.actor_opt.zero_grad(set_to_none=True)
        loss.backward()
        average_gradients(self.actor)
        norm = nn.utils.clip_grad_norm_(self.actor.parameters(), 1, foreach=True)
        self.valid.logical_and_(torch.isfinite(loss.detach()) & torch.isfinite(norm))
        self.actor_opt.step()
        self.q.requires_grad_(True)
        self.actor_steps += 1
        self.metric_sum[3].add_(loss.detach())
        self.metric_sum[4].add_(bc.detach())
        self.metric_sum[5].add_(1)
        with torch.no_grad():
            torch._foreach_lerp_(list(self.target_actor.parameters()), list(self.actor.parameters()), .005)


class EpisodeRing:
    """Preallocated GPU replay. A sampler cut can never create a MC target."""
    def __init__(self, capacity, device):
        if capacity < 900:
            raise ValueError('replay must hold at least one full episode')
        self.capacity, self.device = capacity, torch.device(device)
        self.data, self.total, self.rows = None, 0, 0
        self.terminals = deque()
        self.terminal_tensor = None

    def add(self, raw):
        batch = complete_episode_targets(raw)
        n = len(batch['reward'])
        if n > self.capacity:
            raise ValueError('episode exceeds replay capacity')
        if self.data is None:
            self.data = {k: torch.empty((self.capacity, *v.shape[1:]), dtype=torch.float32, device=self.device)
                         for k, v in batch.items()}
        slots = (torch.arange(n, device=self.device) + self.total) % self.capacity
        for key, value in batch.items():
            self.data[key][slots] = torch.as_tensor(value, device=self.device)
        self.total += n
        self.rows = min(self.total, self.capacity)
        self.terminals.append(self.total - 1)
        while self.terminals and self.terminals[0] < self.total - self.capacity:
            self.terminals.popleft()
        self.terminal_tensor = torch.tensor([i % self.capacity for i in self.terminals], device=self.device)

    def sample(self, size, terminal_fraction=.125):
        if not self.rows or self.terminal_tensor is None:
            raise ValueError('cannot sample empty replay')
        nterm = int(size * terminal_fraction)
        ids = torch.cat((torch.randint(self.rows, (size - nterm,), device=self.device),
                         self.terminal_tensor[torch.randint(len(self.terminals), (nterm,), device=self.device)]))
        return {k: v[ids] for k, v in self.data.items()}

    @property
    def allocated_bytes(self):
        return sum(v.numel() * v.element_size() for v in self.data.values()) if self.data else 0
