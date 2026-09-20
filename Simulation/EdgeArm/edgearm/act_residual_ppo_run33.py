"""Bounded simulation residual PPO on a frozen ACT encoder. Not production.

Actor/critic consume only causal ACT observation features, q/dq and base action.
Simulator geometry appears only in reward computation, never policy inputs.
Finite-horizon episodes use zero terminal bootstrap by definition.
"""

import argparse
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.distributions import Normal
from .evaluate_multimodal_act_v5 import load_model, rollout
from .train_multimodal_act_v5 import sha256
from .train_staged_hybrid_contact_sac import _atomic_json


class ResidualHead(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim), nn.Linear(dim, 128), nn.Tanh(), nn.Linear(128, 128), nn.Tanh()
        )
        self.actor = nn.Linear(128, 6)
        self.critic = nn.Linear(128, 1)
        self.logstd = nn.Parameter(torch.full((6,), -1.5))
        nn.init.zeros_(self.actor.weight)
        nn.init.zeros_(self.actor.bias)

    def distribution(self, x):
        h = self.net(x)
        return Normal(self.actor(h), self.logstd.clamp(-4, -0.5).exp()), self.critic(h).squeeze(-1)


class ResidualACT(nn.Module):
    def __init__(self, base, limit=0.08):
        super().__init__()
        if not 0 < limit <= 1.0:
            raise ValueError("bounded residual limit required")
        self.base = base.eval().requires_grad_(False)
        self.config = base.config
        self.head = ResidualHead(self.config.model_dim + 18)
        self.limit = limit
        self.explore = True
        self.buffer = []

    def forward(self, inputs, return_aux=False):
        memory, padding, _ = self.base.policy.encode_observations(inputs)
        actions, decoded = self.base.policy.decode_observations(memory, padding)
        features = torch.cat([decoded[:, 0], inputs["robot_state"], actions[:, 0]], -1).detach()
        dist, value = self.head.distribution(features)
        raw = dist.sample() if self.explore else dist.mean
        delta = raw.tanh() * self.limit
        actions = actions.clone()
        actions[:, 0] = (actions[:, 0] + delta).clamp(-1, 1)
        self.buffer.append(
            (features.detach(), raw.detach(), dist.log_prob(raw).sum(-1).detach(), value.detach())
        )
        return {"action": actions} if return_aux else actions


def episode_rewards(telemetry, result):
    a = np.asarray(telemetry)
    if not len(a):
        return np.zeros(0, np.float32)
    distances = np.r_[result["initial_distance_m"], a[:, 1]]
    coverage = np.r_[result["initial_coverage"], a[:, 2]]
    hold = np.r_[0.0, a[:, 3]]
    rewards = 10 * (-np.diff(distances)) + 2 * np.diff(coverage) + 0.2 * np.diff(hold) - 0.002
    if result["safe_success"]:
        rewards[-1] += 5
    elif result["reason"] in ("unsafe_contact", "safety_filter_infeasible"):
        rewards[-1] -= 2
    # The old direction-only violation is diagnostic, not a terminal penalty.
    return rewards.astype(np.float32)


def ppo_update(agent, optimizer, rewards):
    n = len(rewards)
    if not n:
        return {"updates": 0}
    x, u, old_log, values = [torch.cat([row[i] for row in agent.buffer[:n]]) for i in range(4)]
    r = torch.as_tensor(rewards, device=x.device)
    r = r - 0.01 * (u.tanh() * agent.limit).square().sum(-1)
    adv = torch.zeros_like(r)
    carry = torch.zeros((), device=x.device)
    for i in reversed(range(n)):
        next_v = values[i + 1] if i + 1 < n else 0.0
        carry = r[i] + 0.99 * next_v - values[i] + 0.99 * 0.95 * carry
        adv[i] = carry
    returns = adv + values
    normalized = (adv - adv.mean()) / adv.std(unbiased=False).clamp_min(1e-6)
    records = []
    for _ in range(4):
        for ids in torch.randperm(n, device=x.device).split(128):
            distribution, value = agent.head.distribution(x[ids])
            log = distribution.log_prob(u[ids]).sum(-1)
            ratio = (log - old_log[ids]).exp()
            policy = -torch.minimum(ratio * normalized[ids], ratio.clamp(0.8, 1.2) * normalized[ids]).mean()
            value_loss = (value - returns[ids]).square().mean()
            loss = policy + 0.5 * value_loss
            if not torch.isfinite(loss):
                raise ValueError("nonfinite residual PPO loss")
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(agent.head.parameters(), 0.5, error_if_nonfinite=True)
            optimizer.step()
            records.append(float(loss.detach()))
    return {"updates": len(records), "mean_loss": float(np.mean(records)), "reward_sum": float(r.sum())}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--train-seeds", type=int, nargs="+", required=True)
    p.add_argument("--eval-seeds", type=int, nargs="+", required=True)
    p.add_argument("--episodes", type=int, default=12)
    p.add_argument("--max-steps", type=int, default=600)
    p.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    p.add_argument("--seed", type=int, default=3301)
    p.add_argument("--residual-limit", type=float, default=0.5)
    p.add_argument("--init-residual")
    p.add_argument("--eval-every", type=int, default=12)
    a = p.parse_args()
    if set(a.train_seeds) & set(a.eval_seeds) or a.episodes < 1 or a.max_steps < 1:
        p.error("positive limits and disjoint seeds required")
    torch.set_num_threads(4)
    torch.manual_seed(a.seed)
    out = Path(a.output_dir)
    out.mkdir(parents=True, exist_ok=False)
    base, checkpoint = load_model(a.checkpoint)
    agent = ResidualACT(base, limit=a.residual_limit).to(a.device).eval()
    optimizer = torch.optim.Adam(agent.head.parameters(), lr=1e-4)
    base_hash = sha256(a.checkpoint)
    if a.init_residual:
        initial = torch.load(a.init_residual, map_location=a.device, weights_only=True)
        if initial["base_checkpoint_sha256"] != base_hash or initial["limit"] != a.residual_limit:
            raise ValueError("residual checkpoint base/limit mismatch")
        agent.head.load_state_dict(initial["head"])
    state = dict(
        status="running",
        production_admission=False,
        export_admission=False,
        final_vla_acceptance=False,
        base_checkpoint_sha256=base_hash,
        arguments=vars(a),
        contact_profile="task_goal_v1",
        residual_limit=a.residual_limit,
    )
    _atomic_json(out / "run_state.json", state)
    try:
        for episode in range(a.episodes):
            folder = out / f"train_{episode:03d}"
            folder.mkdir()
            agent.buffer.clear()
            seed = a.train_seeds[episode % len(a.train_seeds)]
            result = rollout(agent, checkpoint, seed, folder, a.max_steps, "task_goal_v1")
            telemetry = np.load(folder / f"{seed}_trace.npz")["evaluation_telemetry"]
            metrics = ppo_update(agent, optimizer, episode_rewards(telemetry, result))
            _atomic_json(folder / "result.json", dict(result=result, ppo=metrics))
            torch.save(
                dict(
                    head=agent.head.state_dict(),
                    optimizer=optimizer.state_dict(),
                    episodes_completed=episode + 1,
                    limit=agent.limit,
                    base_checkpoint_sha256=state["base_checkpoint_sha256"],
                ),
                out / "residual_last.pt",
            )
            state.update(episodes_completed=episode + 1, last_result=result, last_update=metrics)
            _atomic_json(out / "run_state.json", state)
            if a.eval_every > 0 and (episode + 1) % a.eval_every == 0:
                agent.explore = False
                evaluation = []
                for eval_seed in a.eval_seeds:
                    eval_folder = out / f"checkpoint_{episode + 1:03d}_eval_{eval_seed}"
                    eval_folder.mkdir()
                    agent.buffer.clear()
                    evaluation.append(rollout(agent, checkpoint, eval_seed, eval_folder, 900, "task_goal_v1"))
                torch.save(
                    dict(head=agent.head.state_dict(), limit=agent.limit, base_checkpoint_sha256=base_hash),
                    out / f"residual_{episode + 1:03d}.pt",
                )
                state.update(latest_evaluation=evaluation, evaluation_at_episode=episode + 1)
                _atomic_json(out / "run_state.json", state)
                agent.explore = True
        agent.explore = False
        results = []
        for seed in a.eval_seeds:
            folder = out / f"eval_{seed}"
            folder.mkdir()
            agent.buffer.clear()
            results.append(rollout(agent, checkpoint, seed, folder, 900, "task_goal_v1"))
        state.update(status="complete_pending_review", evaluation=results)
    except BaseException as exc:
        state.update(status="failed", exception=repr(exc))
        raise
    finally:
        _atomic_json(out / "run_state.json", state)


if __name__ == "__main__":
    main()
