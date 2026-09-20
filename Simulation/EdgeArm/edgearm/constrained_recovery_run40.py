"""Single-GPU bounded residual TD3+BC-inspired corrective experiment.

No entropy bonus, no restoration of Run38's diverged actor/critic. Offline
Monte Carlo targets are behavior-return calibration anchors, NOT claims of
unbiased evaluation of the current policy. Online RL uses 16-step TD targets.
"""

from copy import deepcopy

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .continuous_recovery_run38 import FixedNormalizer, FIELDS
from .dual_gpu_sac_v1 import PackedQ


class BoundedActor(nn.Module):
    def __init__(self, feature_dim, width=256, residual_limit=.12):
        super().__init__()
        if not 0 < residual_limit <= .25:
            raise ValueError('small normalized residual required')
        self.register_buffer('residual_limit', torch.tensor(float(residual_limit)))
        self.normalizer = FixedNormalizer(feature_dim)
        self.net = nn.Sequential(nn.Linear(feature_dim, width), nn.LayerNorm(width), nn.SiLU(),
                                 nn.Linear(width, width), nn.LayerNorm(width), nn.SiLU())
        self.mean = nn.Linear(width, 6)
        nn.init.zeros_(self.mean.weight)
        nn.init.zeros_(self.mean.bias)

    def forward(self, x, base, *, noise_std=0.):
        raw = self.mean(self.net(self.normalizer(x)))
        if noise_std:
            raw = raw + float(noise_std) * torch.randn_like(raw)
        # Both sampled and deterministic actions obey the SAME delta bound.
        return (base + self.residual_limit * raw.tanh()).clamp(-1, 1)


def complete_episode_targets(batch, *, gamma=.999, nstep=16, teacher=False):
    if set(batch) != set(FIELDS) or not 0 < gamma <= 1 or nstep < 1:
        raise ValueError('invalid replay contract')
    b = {k: np.asarray(v, np.float32).copy() for k, v in batch.items()}
    n = len(b['reward'])
    if not n or any(len(v) != n or not np.isfinite(v).all() for v in b.values()):
        raise ValueError('empty/nonfinite/misaligned episode')
    if b['action'].shape != (n, 6) or abs(b['action']).max() > 1.00001:
        raise ValueError('invalid Command-V2 action')
    if not np.array_equal(b['terminal'], np.r_[np.zeros(n - 1), 1]):
        raise ValueError('complete single episode required; never make sampler cuts terminal')
    for after, current in [('next_x', 'x'), ('next_base', 'base'), ('next_privileged', 'privileged')]:
        if not np.array_equal(b[after][:-1], b[current][1:]):
            raise ValueError('non-contiguous episode ' + current)
    returns = np.empty(n, np.float32)
    carry = 0.
    for i in range(n - 1, -1, -1):
        carry = float(b['reward'][i]) + gamma * carry
        returns[i] = carry
    endpoint = np.minimum(np.arange(n) + nstep, n)
    reward = np.array([np.dot(gamma ** np.arange(end - i), b['reward'][i:end])
                       for i, end in enumerate(endpoint)], np.float32)
    discount = np.where(endpoint == n, 0., gamma ** (endpoint - np.arange(n))).astype(np.float32)
    b.update(mc_return=returns, n_reward=reward, n_discount=discount,
             boot_x=b['next_x'][endpoint - 1], boot_base=b['next_base'][endpoint - 1],
             boot_privileged=b['next_privileged'][endpoint - 1],
             bc_mask=np.full(n, float(teacher), np.float32))
    return b


class CompleteReplay:
    def __init__(self, device):
        self.device = torch.device(device)
        self.chunks, self.ids = [], set()
        self.data = None
        self.rows = 0

    def add(self, episode_id, batch, *, teacher=False):
        if episode_id in self.ids:
            raise ValueError('duplicate episode')
        b = complete_episode_targets(batch, teacher=teacher)
        self.ids.add(episode_id)
        self.chunks.append(b)
        self.rows += len(b['reward'])
        self.data = None

    def materialize(self):
        if self.data is None:
            if not self.rows:
                raise ValueError('empty replay')
            self.data = {k: torch.as_tensor(np.concatenate([c[k] for c in self.chunks]), device=self.device)
                         for k in self.chunks[0]}
            self.terminals = torch.where(self.data['terminal'] == 1)[0]
        return self.data

    def sample(self, size, *, terminal_fraction=.125):
        data = self.materialize()
        nterm = int(size * terminal_fraction)
        rows = torch.cat((torch.randint(self.rows, (size - nterm,), device=self.device),
                          self.terminals[torch.randint(len(self.terminals), (nterm,), device=self.device)]))
        return {k: v[rows] for k, v in data.items()}


def combine_batches(*batches):
    return {k: torch.cat([b[k] for b in batches]) for k in batches[0]}


def task_potential(p):
    distance, coverage, hold = p[:, -6], p[:, -5], p[:, -4]
    return -6 * distance + coverage + 4 * (hold / 3).clamp(max=1)


def raw_return_upper_bound(p):
    # No entropy in Run40: potential telescopes; all costs are nonpositive.
    return 8 - task_potential(p)


class CalibratedLearner:
    def __init__(self, feature_dim, privileged_dim, device, *, width=256, residual_limit=.12):
        self.device = torch.device(device)
        self.actor = BoundedActor(feature_dim, width, residual_limit).to(self.device)
        self.q = PackedQ(feature_dim, privileged_dim, width=width, count=2).to(self.device)
        self.target_actor = deepcopy(self.actor).requires_grad_(False)
        self.target_q = deepcopy(self.q).requires_grad_(False)
        fused = self.device.type == 'cuda'
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=3e-5, fused=fused)
        self.q_opt = torch.optim.Adam(self.q.parameters(), lr=1e-4, fused=fused)
        self.actor_steps = self.critic_steps = 0
        self.valid = torch.ones((), dtype=torch.bool, device=self.device)
        self.metric_sum = torch.zeros(6, device=self.device)

    def fit_normalizers(self, data):
        self.actor.normalizer.fit(data['x'])
        self.q.xnorm.fit(data['x'])
        self.q.pnorm.fit(data['privileged'])
        self.sync_targets()

    def sync_targets(self):
        self.target_q.load_state_dict(self.q.state_dict())
        self.target_actor.load_state_dict(self.actor.state_dict())

    def critic_update(self, b, *, calibration=False):
        with torch.no_grad():
            if calibration:
                target = b['mc_return']
            else:
                action = self.target_actor(b['boot_x'], b['boot_base'], noise_std=.10)
                qnext = self.target_q(b['boot_x'], b['boot_privileged'], action).amin(0)
                target = b['n_reward'] + b['n_discount'] * qnext
        values = self.q(b['x'], b['privileged'], b['action'])
        td_loss = F.smooth_l1_loss(values, target[None].expand_as(values))
        # Behavior-return regularization stabilizes scale. It is intentionally
        # a biased anchor, not relabeled as an on-policy Monte Carlo estimate.
        anchor = F.smooth_l1_loss(values, b['mc_return'][None].expand_as(values))
        loss = td_loss if calibration else td_loss + .25 * anchor
        self.q_opt.zero_grad(set_to_none=True)
        loss.backward()
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
        action = self.actor(b['x'], b['base'])
        teacher_action = self.actor(teacher['x'], teacher['base'])
        scale = self.actor.residual_limit
        bc = ((teacher_action - teacher['action']) / scale).square().mean()
        anchor = ((action - b['base']) / scale).square().mean()
        if bc_only:
            loss = bc + .05 * anchor
        else:
            # Learn an advantage relative to frozen ACT at the SAME state.
            # Detach the baseline; never train from privileged actor inputs.
            with torch.no_grad():
                base_value = self.q(b['x'], b['privileged'], b['base']).amin(0)
                normalizer = base_value.abs().mean().clamp_min(1.)
            value = self.q(b['x'], b['privileged'], action).amin(0)
            rl = -((value - base_value) / normalizer).mean()
            loss = rl + .5 * bc + .1 * anchor
        self.actor_opt.zero_grad(set_to_none=True)
        loss.backward()
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

    def flush_metrics(self):
        v = torch.cat((self.metric_sum, self.valid[None].float())).detach().cpu().numpy()
        self.metric_sum.zero_()
        if not v[-1] or not np.isfinite(v).all():
            raise FloatingPointError('nonfinite learner; do not sample this policy')
        return dict(q_loss=float(v[0] / max(v[2], 1)), q_mean=float(v[1] / max(v[2], 1)),
                    actor_loss=float(v[3] / max(v[5], 1)), bc_loss=float(v[4] / max(v[5], 1)),
                    block_critic_updates=int(v[2]), block_actor_updates=int(v[5]),
                    critic_updates=self.critic_steps, actor_updates=self.actor_steps)

    @torch.inference_mode()
    def audit(self, replay):
        b = replay.materialize()
        outputs, policy_outputs, deltas = [], [], []
        for begin in range(0, replay.rows, 512):
            s = {k: v[begin:begin + 512] for k, v in b.items()}
            a = self.actor(s['x'], s['base'])
            outputs.append(self.q(s['x'], s['privileged'], s['action']).mean(0))
            policy_outputs.append(self.q(s['x'], s['privileged'], a).mean(0))
            deltas.append(abs(a - s['base']).amax(1))
        values, policy_values, delta = torch.cat(outputs), torch.cat(policy_outputs), torch.cat(deltas)
        error = abs(values - b['mc_return'])
        terminal = b['terminal'] == 1
        bound = raw_return_upper_bound(b['privileged'])
        over = policy_values - bound
        row = torch.stack((error.mean(), error[terminal].mean(), error[terminal].max(),
                           (over > .25).float().mean(), over.max(), values.mean(), delta.max()))
        v = row.cpu().tolist()
        return dict(rows=replay.rows, behavior_return_mae=v[0], terminal_mae=v[1], terminal_max_error=v[2],
                    policy_Q_over_bound_fraction=v[3], maximum_Q_excess=v[4], q_mean=v[5], maximum_action_delta=v[6])


def calibration_ok(audit):
    return (audit['behavior_return_mae'] <= .75 and audit['terminal_mae'] <= .5
            and audit['terminal_max_error'] <= 1. and audit['policy_Q_over_bound_fraction'] <= .01)


def summarize(results):
    n = max(len(results), 1)
    return dict(episodes=len(results), successes=sum(r['safe_success'] for r in results),
                mean_coverage=sum(r['maximum_coverage'] for r in results) / n,
                mean_hold_s=sum(r['maximum_hold_s'] for r in results) / n,
                contact_episodes=sum(r['valid_contact_steps'] > 0 for r in results),
                hard_failures=sum(r['end_kind'] == 'hard_failure' for r in results),
                mean_rewrite_fraction=sum(r['safety_rewrite_steps'] / max(r['steps'], 1) for r in results) / n)


def nonregression(candidate, baseline):
    return (candidate['successes'] >= baseline['successes']
            and candidate['mean_coverage'] >= .85 * baseline['mean_coverage']
            and candidate['contact_episodes'] >= baseline['contact_episodes'] - 1
            and candidate['hard_failures'] <= baseline['hard_failures']
            and candidate['mean_rewrite_fraction'] <= baseline['mean_rewrite_fraction'] + .15)


def selection_score(summary):
    return (summary['successes'], -summary['hard_failures'], summary['mean_coverage'], summary['mean_hold_s'])
