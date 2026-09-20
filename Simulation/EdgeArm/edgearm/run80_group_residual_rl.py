"""Bounded online residual RL around frozen Run77 + disclosed Run78 finishing.

Leave-one-out episode returns, PPO-style per-decision clipping, no critic and
no teacher labels. This is a robotics adaptation, not an exact RLOO replication.
Only real wrist images, reported joints and past commands enter the actor.
Simulator geometry is confined to reward and post-action audit.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import math
import multiprocessing as mp
from pathlib import Path
import shutil
import time

import numpy as np
import torch
from torch import nn

from .candidate_command_contract_v2 import ACTION_CONTRACT
from .constrained_recovery_run40 import summarize
from .run34_repeat_eval import deterministic_runtime
from .run42.domain import sample_domain
from .run42.session import DomainSession
from .run63_control_probe import TinyTarget
from .run67_visual_state import VisualState, actor_input, language_indices
from .run74_observation_probe import observation_prefix, observation_history_indices
from .run78_completion_probe import CompletionLatch, TerminalServo
from .sparse_4d_vla_act_v26 import Sparse4DVLAConfigV26
from .train_staged_hybrid_contact_sac import _atomic_json


POLICY_KIND = 'group_residual_rl_v80'
DECISION_INTERVAL = 4
STD = .7
RESIDUAL_LIMIT = .06


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class ResidualActor(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('xmean', torch.zeros(118))
        self.register_buffer('xscale', torch.ones(118))
        self.net = nn.Sequential(nn.Linear(118, 128), nn.Tanh(), nn.Linear(128, 128),
                                 nn.Tanh(), nn.Linear(128, 6))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x):
        if x.shape[-1] != 118:
            raise ValueError('deployable 118-D state required')
        return self.net(((x-self.xmean)/self.xscale).clamp(-20, 20))


def latent_log_prob(z, mean, std=STD):
    # Likelihood is of the latent sampled BEFORE tanh/clipping/safety. Those
    # deterministic transformations are part of the environment transition.
    return (-.5*((z-mean)/std).square()-math.log(std)-.5*math.log(2*math.pi)).sum(-1)


def gaussian_kl(new_mean, old_mean, std=STD):
    return .5*((new_mean-old_mean)/std).square().sum(-1)


def residual_command(base, latent, limit=RESIDUAL_LIMIT):
    if not 0 < limit <= .1:
        raise ValueError('bounded residual required')
    return np.clip(np.asarray(base)+limit*np.tanh(np.asarray(latent)), -1., 1.).astype(np.float32)


def potential(metrics):
    distance, coverage, hold = np.asarray(metrics, np.float64)
    if not np.isfinite([distance, coverage, hold]).all():
        raise ValueError('finite reward measurements required')
    return -10.*np.clip(distance, 0., .5)+2.*np.clip(coverage, 0., 1.)+.5*np.clip(hold/3., 0., 1.)


def transition_reward(before, after, kind, reason):
    # Signed potential differences telescope; repeated contact/oscillation
    # cannot accumulate a free positive progress bonus. Success is unchanged.
    failure = kind == 'hard_failure' or reason == 'block_out_of_bounds'
    return float(potential(after)-potential(before)+10.*(kind == 'success')-5.*failure-.001)


def leave_one_out(returns, context_ids):
    returns, context_ids = np.asarray(returns, np.float64), np.asarray(context_ids)
    if returns.ndim != 1 or returns.shape != context_ids.shape or not np.isfinite(returns).all():
        raise ValueError('finite per-episode returns and matching context IDs required')
    advantages = np.empty_like(returns)
    for context in np.unique(context_ids):
        ids = np.flatnonzero(context_ids == context)
        if len(ids) < 2:
            raise ValueError('at least two independent samples of each context required')
        values = returns[ids]
        advantages[ids] = values-(values.sum()-values)/(len(ids)-1)
    return advantages.astype(np.float32)


def validate_group(seed, stochastic):
    group = int(seed)//9
    if stochastic:
        if not 100300000 <= group < 100300100:
            raise ValueError('collection must use dedicated new training groups')
    elif not 97100000 <= group < 97100004:
        raise ValueError('this entry only allows fixed development, not independent acceptance')


def load_residual(path, device, vision_path, base_path):
    saved = torch.load(path, map_location='cpu', weights_only=True)
    if saved.get('policy_kind') != POLICY_KIND or saved.get('actor_uses_simulator_state') is not False:
        raise ValueError('wrong policy provenance')
    if saved['vision_sha256'] != digest(vision_path) or saved['base_sha256'] != digest(base_path):
        raise ValueError('frozen anchor pairing mismatch')
    if (saved['decision_interval'], saved['std'], saved['residual_limit']) != (DECISION_INTERVAL, STD, RESIDUAL_LIMIT):
        raise ValueError('sampling/execution contract mismatch')
    actor = ResidualActor().to(device).eval()
    actor.load_state_dict(saved['model'])
    return actor


def episode(job):
    seed, replica, stochastic, vision_path, base_path, residual_path, output, noise_seed = job
    validate_group(seed, stochastic)
    deterministic_runtime(); torch.set_num_threads(1)
    rng = np.random.default_rng(noise_seed)
    saved = torch.load(vision_path, map_location='cpu', weights_only=True)
    visual = VisualState(recent_block_seconds=saved.get('recent_block_seconds')).cuda().eval()
    visual.load_state_dict(saved['model'])
    base = TinyTarget(118, 512).cuda().eval()
    base.load_state_dict(torch.load(base_path, map_location='cpu', weights_only=True)['model'])
    actor = load_residual(residual_path, 'cuda', vision_path, base_path)
    folder = Path(output)/f'episode_{seed}_replica_{replica}'
    folder.mkdir(parents=True, exist_ok=False)
    session = DomainSession(Sparse4DVLAConfigV26(language_max_tokens=128,
        visual_memory_mode='episode_anchors_v54'), ACTION_CONTRACT, seed, sample_domain(seed+6001, 0))
    selection = language_indices(session.buffer.instruction)
    latch = CompletionLatch()
    servo = TerminalServo(session.env.model, session.env._ids['tool_site'], 'lift')
    q_history, commands, frames, audit = [], [], [], []
    xs, zs, mus, logps, decision_steps = [], [], [], [], []
    maximum_coverage = maximum_hold = total_reward = 0.
    contacts = effectful = rewrites = 0
    latent = np.zeros(6, np.float32)
    next_decision = 220
    started = time.time()
    try:
        session.observe(); initial = session.reported[:6].copy()
        if session.reward_state()[1] != 0.: raise ValueError('nonzero initial coverage')
        survey_audit = observation_prefix(session, initial, q_history, commands, frames)
        for step in range(len(commands), 900):
            if session.end_kind != 'sampler_cut': break
            session.observe(); rows = session.buffer.rows
            ids = observation_history_indices(step, True)
            rgb = np.stack([rows[int(i)]['rgb'] for i in ids])
            pose = np.stack([rows[int(i)]['camera_pose'] for i in ids])
            age = np.asarray([rows[int(i)]['time']-rows[-1]['time'] for i in ids], np.float32)
            previous = rows[-2]['applied_target']
            with torch.inference_mode():
                world = visual(torch.from_numpy(rgb)[None].cuda(), torch.from_numpy(pose)[None].cuda(),
                    torch.from_numpy(session.buffer.K)[None].cuda(), torch.from_numpy(age)[None].cuda())[0].cpu().numpy()
                x = actor_input(session.reported, rows[-1]['tool'], q_history, commands, previous, initial, world, selection)
                xt = torch.from_numpy(x)[None].cuda()
                base_command = base(xt)[0].cpu().numpy().clip(-1, 1)
                near = latch.update(world[list(selection)], step)
                if near:
                    action = servo.command(session.reported, previous)
                else:
                    if step >= next_decision:
                        mean = actor(xt)[0]
                        latent = mean.cpu().numpy().copy()
                        if stochastic: latent += STD*rng.standard_normal(6).astype(np.float32)
                        xs.append(x.copy()); zs.append(latent.copy()); mus.append(mean.cpu().numpy().copy())
                        logps.append(float(latent_log_prob(torch.from_numpy(latent).cuda(), mean)))
                        decision_steps.append(step); next_decision = step+DECISION_INTERVAL
                    action = residual_command(base_command, latent)
            # Policy decision is now fixed. True geometry is only reward/audit.
            truth = np.stack((session.env.block_xy(), session.env.target_xy))
            before = session.reward_state()
            audit.append(np.r_[step, world[list(selection)].ravel(), truth.ravel(), before,
                np.linalg.norm(world[list(selection)]-truth, axis=1), near])
            q_history.append(session.reported.copy()); commands.append(action.copy())
            if step % 8 == 0: frames.append(rows[-1]['rgb'].copy())
            result = session.advance(action); after = session.reward_state()
            total_reward += transition_reward(before, after, result['kind'], session.reason)
            maximum_coverage = max(maximum_coverage, after[1]); maximum_hold = max(maximum_hold, after[2])
            contacts += int(result.get('valid_contact', False)); effectful += int(result.get('effectful_contact', False))
            rewrites += int(result.get('safety_rewrite', False))
            if result['kind'] != 'sampler_cut': break
        kind = session.end_kind if session.end_kind != 'sampler_cut' else 'finite_timeout'
        result = dict(seed=seed, replica=replica, stochastic=stochastic, noise_seed=noise_seed,
            safe_success=kind=='success', end_kind=kind, terminal_reason=session.reason,
            episode_return=total_reward, steps=len(commands), decision_count=len(xs),
            seconds=time.time()-started, maximum_coverage=float(maximum_coverage),
            maximum_hold_s=float(maximum_hold), valid_contact_steps=contacts,
            effectful_contact_steps=effectful, safety_rewrite_steps=rewrites,
            final_distance_m=float(session.reward_state()[0]), final_coverage=float(session.reward_state()[1]),
            latch_step=latch.trigger_step, survey_audit=survey_audit,
            actor_uses_simulator_state=False, teacher_assisted=False, scripted_completion=True,
            fixed_wrist_survey=True, fixed_survey_steps=220, max_steps=900, initial_coverage=0.,
            original_ACT_checkpoint=False, independent_acceptance=False,
            production_admission=False, export_admission=False, final_vla_acceptance=False)
        np.savez_compressed(folder/'policy_rollout.npz', x=np.asarray(xs, np.float32).reshape(-1, 118),
            latent=np.asarray(zs, np.float32).reshape(-1, 6), old_mean=np.asarray(mus, np.float32).reshape(-1, 6),
            old_logp=np.asarray(logps, np.float32), decision_steps=np.asarray(decision_steps, np.int32))
        np.savez_compressed(folder/'trace.npz', audit=np.asarray(audit), reported=np.asarray(q_history),
            command=np.asarray(commands), wrist_rgb=np.asarray(frames))
        result['policy_rollout'] = str(folder/'policy_rollout.npz')
        _atomic_json(folder/'result.json', result)
        return result
    finally:
        session.close()


def optimize(actor, optimizer, results, epochs=8, clip=.15, target_kl=.02):
    if any(not row['stochastic'] for row in results):
        raise ValueError('only fresh on-policy stochastic training episodes may update actor')
    for row in results: validate_group(row['seed'], True)
    returns = [row['episode_return'] for row in results]
    advantages = leave_one_out(returns, [row['seed'] for row in results])/5.
    parts = {key: [] for key in ('x', 'latent', 'old_mean', 'old_logp', 'advantage')}
    for row, advantage in zip(results, advantages):
        with np.load(row['policy_rollout']) as data:
            for key in ('x', 'latent', 'old_mean', 'old_logp'): parts[key].append(data[key].copy())
            parts['advantage'].append(np.full(len(data['x']), advantage, np.float32))
    device = next(actor.parameters()).device
    data = {k: torch.from_numpy(np.concatenate(v)).to(device) for k, v in parts.items()}
    if not len(data['x']): raise ValueError('no executed stochastic decisions')
    with torch.inference_mode():
        reproduced = latent_log_prob(data['latent'], actor(data['x']))
        reproduction_error = float((reproduced-data['old_logp']).abs().max())
        if reproduction_error > 1e-4: raise ValueError('old action likelihood cannot be reproduced')
    losses, updates = [], 0
    initial = {k: v.detach().clone() for k, v in actor.state_dict().items()}
    stop = False
    for _ in range(epochs):
        for ids in torch.randperm(len(data['x']), device=device).split(512):
            mean = actor(data['x'][ids])
            logp = latent_log_prob(data['latent'][ids], mean)
            ratio = (logp-data['old_logp'][ids]).clamp(-20, 20).exp()
            advantage = data['advantage'][ids]
            surrogate = torch.minimum(ratio*advantage, ratio.clamp(1-clip, 1+clip)*advantage)
            anchor_kl = gaussian_kl(mean, torch.zeros_like(mean)).mean()
            loss = -surrogate.mean()+.01*anchor_kl
            optimizer.zero_grad(set_to_none=True); loss.backward()
            nn.utils.clip_grad_norm_(actor.parameters(), 1., error_if_nonfinite=True)
            optimizer.step(); losses.append(float(loss.detach())); updates += 1
            with torch.inference_mode():
                new_mean = actor(data['x'])
                kl = float(gaussian_kl(new_mean, data['old_mean']).mean())
            if kl > target_kl:
                stop = True; break
        if stop: break
    with torch.inference_mode():
        mean = actor(data['x'])
        ratio = (latent_log_prob(data['latent'], mean)-data['old_logp']).clamp(-20, 20).exp()
        parameter_delta = sum(float((value-initial[key]).square().sum()) for key, value in actor.state_dict().items())**.5
    return dict(loss=float(np.mean(losses)), optimizer_updates=updates,
        kl_old=kl, early_stop_kl=stop, clip_fraction=float(((ratio-1).abs()>clip).float().mean()),
        old_logp_max_error=reproduction_error, parameter_l2_change=parameter_delta,
        mean_absolute_deterministic_residual=float((RESIDUAL_LIMIT*mean.tanh()).abs().mean()),
        training_return_mean=float(np.mean(returns)), training_return_std=float(np.std(returns)),
        episode_advantage_std=float(np.std(advantages)), decisions=len(data['x']))


def save_policy(actor, path, vision, base, iteration):
    torch.save(dict(model={k:v.detach().cpu().clone() for k,v in actor.state_dict().items()},
        policy_kind=POLICY_KIND, vision_sha256=digest(vision), base_sha256=digest(base),
        decision_interval=DECISION_INTERVAL, std=STD, residual_limit=RESIDUAL_LIMIT,
        iteration=iteration, actor_uses_simulator_state=False, teacher_assisted=False,
        scripted_completion=True, fixed_wrist_survey=True, original_ACT_checkpoint=False,
        production_admission=False, export_admission=False, final_vla_acceptance=False), path)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('vision', 'base', 'output'): p.add_argument('--'+key, type=Path, required=True)
    p.add_argument('--iterations', type=int, default=12)
    p.add_argument('--replicas', type=int, choices=(2, 4), default=4)
    p.add_argument('--workers', type=int, default=9)
    p.add_argument('--eval-every', type=int, default=3)
    p.add_argument('--wall-seconds', type=int, default=3300)
    p.add_argument('--group-start', type=int, default=100300000)
    p.add_argument('--smoke-only', action='store_true')
    a = p.parse_args()
    if not 1 <= a.iterations <= 12 or not 1 <= a.workers <= 9 or not 1 <= a.eval_every <= 3:
        raise ValueError('bounded online RL trial required')
    if not 300 <= a.wall_seconds <= 6900 or not 100300000 <= a.group_start < a.group_start+a.iterations <= 100300100:
        raise ValueError('explicit time bound and unused training namespace required')
    a.output.mkdir(parents=True, exist_ok=False)
    deterministic_runtime(); torch.set_num_threads(2)
    actor = ResidualActor().cuda()
    base = torch.load(a.base, map_location='cpu', weights_only=True)['model']
    actor.xmean.copy_(base['xmean']); actor.xscale.copy_(base['xscale'].clamp_min(.02))
    optimizer = torch.optim.Adam(actor.parameters(), lr=3e-4)
    state = dict(run='Run80', status='running', started=time.time(), phase='initializing', step=0,
        round=0, rounds=a.iterations, evaluations=[], collections=[], metrics={},
        training_kind='online_group_leave_one_out_clipped_residual_policy_gradient', critic_used=False,
        teacher_assisted=False, actor_uses_simulator_state=False, frozen_perception=True, frozen_base=True,
        renewed_compute_authorization=True, old_six_hour_budget_reused=False,
        max_steps=900, fixed_survey_steps=220, scripted_completion='visual_latch_lift_hold',
        original_ACT_checkpoint=False, independent_acceptance=False,
        production_admission=False, export_admission=False, final_vla_acceptance=False,
        group_start=a.group_start, replicas=a.replicas, residual_limit=RESIDUAL_LIMIT, std=STD,
        decision_interval=DECISION_INTERVAL, learning_rate=3e-4, epochs_per_batch=8,
        reward='potential_delta +10 success -5 hard_or_oob -.001 step',
        potential='-10 min(distance,.5) +2 coverage +.5 min(hold/3,1)',
        vision=str(a.vision), base=str(a.base), vision_sha256=digest(a.vision), base_sha256=digest(a.base))
    source_names=('run80_group_residual_rl.py', 'run78_completion_probe.py', 'run74_observation_probe.py',
        'run67_visual_state.py', 'run63_control_probe.py', 'run61_active_view.py',
        'run42/session.py', 'run42/domain.py', 'branch_session_run37.py', 'candidate_command_contract_v2.py')
    state['source_hashes']={name:digest(Path(__file__).parent/name) for name in source_names}
    for name in source_names:
        dest=a.output/'source_snapshot'/name; dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(Path(__file__).parent/name, dest)
    deadline = state['started']+a.wall_seconds
    def publish(phase=None):
        if phase: state['phase'] = phase
        state.update(updated=time.time(), elapsed_seconds=time.time()-state['started'])
        _atomic_json(a.output/'run_state.json', state)
    def run_jobs(jobs, folder, phase):
        results=[]; state.update(phase_episodes_completed=0, phase_episodes_total=len(jobs)); publish(phase)
        with ProcessPoolExecutor(a.workers, mp_context=mp.get_context('spawn')) as pool:
            for future in as_completed([pool.submit(episode, job) for job in jobs]):
                results.append(future.result()); state.update(phase_episodes_completed=len(results), partial=summarize(results)); publish()
        summary = summarize(results)|dict(round=state['round'],
            block_out_of_bounds=sum(r['terminal_reason']=='block_out_of_bounds' for r in results),
            mean_return=float(np.mean([r['episode_return'] for r in results])),
            pairs={str(r):summarize([x for x in results if x['seed']%9==r]) for r in range(9)})
        _atomic_json(folder/'summary.json', dict(summary=summary, results=results))
        return summary, results
    def evaluate(path, label):
        folder = a.output/label
        jobs=[(g*9+r, 0, False, str(a.vision), str(a.base), str(path), str(folder), 0)
              for g in range(97100000, 97100004) for r in range(9)]
        summary, _ = run_jobs(jobs, folder, 'autonomous_development')
        state['evaluations'].append(summary)
        rank=(summary['successes'], -summary['hard_failures']-summary['block_out_of_bounds'], summary['mean_coverage'])
        if 'best_development_rank' not in state or rank > tuple(state['best_development_rank']):
            state.update(best_development_rank=list(rank), best_residual=str(path))
        publish(); return summary
    try:
        path=a.output/'residual_0.pt'; save_policy(actor, path, a.vision, a.base, 0)
        publish()
        if not a.smoke_only: evaluate(path, 'development_0')
        for iteration in range(1, a.iterations+1):
            if time.time() > deadline-240:
                state.update(stopping_reason='reserve_result_and_evaluation_time'); break
            state['round']=iteration; group=a.group_start+iteration-1
            folder=a.output/f'collection_{iteration}'
            jobs=[(group*9+r, k, True, str(a.vision), str(a.base), str(path), str(folder),
                   group*100+r*10+k) for r in range(9) for k in range(a.replicas)]
            summary, results=run_jobs(jobs, folder, 'on_policy_physical_collection')
            state['collections'].append(summary); publish('group_relative_policy_update')
            state['metrics']=optimize(actor, optimizer, results)
            state['step']+=state['metrics']['optimizer_updates']
            path=a.output/f'residual_{iteration}.pt'; save_policy(actor, path, a.vision, a.base, iteration)
            state['latest_residual']=str(path); publish()
            if a.smoke_only:
                state['status']='smoke_complete'; publish('finished'); return
            if iteration%a.eval_every == 0 or iteration == a.iterations:
                evaluate(path, f'development_{iteration}')
                if state['best_development_rank'][0] >= 30: break
        state['status']='complete_pending_review'; publish('finished')
    except BaseException as exc:
        state.update(status='failed', error=repr(exc)); publish(); raise


if __name__ == '__main__': main()
