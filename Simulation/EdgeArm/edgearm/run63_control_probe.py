"""Minimal supervised control diagnostic, NOT a visual-language policy.

Same nine successful training scenes: replay submitted commands, feedback-track
their requested absolute targets, then learn those targets with a small MLP.
The learned controller only uses joints, past commands and an elapsed clock.
It is a diagnostic of learnability, never the user's 80% autonomous VLA result.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import multiprocessing as mp
import os
from pathlib import Path
import time

for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(key, '1')
os.environ.setdefault('MUJOCO_GL', 'egl')
os.environ.setdefault('PYOPENGL_PLATFORM', 'egl')

import numpy as np
import torch
from torch import nn

from .candidate_command_contract_v2 import ACTION_CONTRACT
from .constrained_recovery_run40 import summarize
from .run34_repeat_eval import deterministic_runtime
from .run42.domain import sample_domain
from .run42.session import DomainSession
from .run54_train import diagnostic_nine
from .sparse_4d_vla_act_v26 import Sparse4DVLAConfigV26
from .train_staged_hybrid_contact_sac import _atomic_json


def features(reported, initial, commands, step):
    """No route/seed/object state. Current report and completed commands only."""
    previous = np.zeros((4, 6), np.float32)
    n = min(4, len(commands))
    if n:
        previous[-n:] = commands[-n:]
    phase = float(step) / 900.
    angles = 2*np.pi*phase*np.asarray([1, 2, 4, 8, 16, 32], np.float32)
    return np.r_[reported[:6], np.clip(reported[6:], -10, 10), initial[:6],
                 previous.ravel(), phase, np.sin(angles), np.cos(angles)].astype(np.float32)


class TinyTarget(nn.Module):
    def __init__(self, inputs=55, width=256, phase_only=False):
        super().__init__()
        self.register_buffer('xmean', torch.zeros(inputs))
        self.register_buffer('xscale', torch.ones(inputs))
        self.register_buffer('ymean', torch.zeros(6))
        self.register_buffer('yscale', torch.ones(6))
        mask = torch.ones(inputs)
        if phase_only:
            mask[:12] = 0
            mask[18:42] = 0
        self.register_buffer('input_mask', mask)
        self.net = nn.Sequential(nn.Linear(inputs, width), nn.SiLU(),
            nn.Linear(width, width), nn.SiLU(), nn.Linear(width, width), nn.SiLU(),
            nn.Linear(width, 6))

    def forward(self, x):
        return self.ymean + self.yscale * self.net(self.input_mask*(x-self.xmean)/self.xscale)


def load_rows(records):
    xs, ys, route_ids = [], [], []
    for route, row in enumerate(records):
        q = np.load(Path(row['store'])/'joint.npy', mmap_mode='r')
        commands = np.load(Path(row['store'])/'command.npy', mmap_mode='r')
        for t in row['valid_times']:
            xs.append(features(q[t], q[0], commands[max(0, t-4):t], t))
            ys.append(q[t, :6] + .055*commands[t])
            route_ids.append(route)
    return np.asarray(xs), np.asarray(ys), np.asarray(route_ids)


def train(records, output, updates, publish, state, deadline, feedback_noise=0., phase_only=False):
    x, y, route = load_rows(records)
    x, y = torch.from_numpy(x).cuda(), torch.from_numpy(y).cuda()
    model = TinyTarget(x.shape[1], phase_only=phase_only).cuda()
    with torch.no_grad():
        model.xmean.copy_(x.mean(0)); model.xscale.copy_(x.std(0).clamp_min(.02))
        model.ymean.copy_(y.mean(0)); model.yscale.copy_(y.std(0).clamp_min(.05))
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=1e-6, fused=True)
    pools = [torch.from_numpy(np.flatnonzero(route == r)).cuda() for r in range(9)]
    best_mae, best_state = float('inf'), None
    for step in range(updates):
        if time.time() > deadline-180:
            break
        ids = torch.cat([pool[torch.randint(len(pool), (114,), device='cuda')] for pool in pools])
        batch_x = x[ids].clone()
        if feedback_noise:
            # Desired reference is unchanged while observed feedback deviates.
            # This tests local reference attraction, not a simulated rollout.
            batch_x[:, :6] += feedback_noise*torch.randn_like(batch_x[:, :6])
            batch_x[:, 6:12] += feedback_noise*4*torch.randn_like(batch_x[:, 6:12])
            batch_x[:, 18:42] += feedback_noise*5*torch.randn_like(batch_x[:, 18:42])
        prediction = model(batch_x)
        loss = ((prediction-y[ids]) / model.yscale).square().mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
        optimizer.step()
        for group in optimizer.param_groups:
            group['lr'] = .00005 + .00095*.5*(1+np.cos(np.pi*(step+1)/updates))
        if (step+1) % 100 == 0 or step+1 == updates:
            with torch.inference_mode():
                error = (model(x)-y).abs()
                mae = float(error.mean())
                by_route = [float(error[torch.from_numpy(route == r).cuda()].mean()) for r in range(9)]
            state.update(step=step+1, metrics=dict(loss=float(loss.detach()), target_mae_rad=mae,
                         worst_joint_error_rad=float(error.max()), route_target_mae_rad=by_route))
            if mae < best_mae:
                best_mae = mae
                best_state = {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
            publish('tiny_supervised_fit')
    if best_state is None:
        raise RuntimeError('no completed tiny fitting checkpoint')
    path = output/'tiny_target.pt'
    torch.save(dict(model=best_state, input_dim=x.shape[1], diagnostic_only=True,
                    actor_uses_simulator_state=False, visual_grounding=False), path)
    del model, x, y, optimizer
    torch.cuda.empty_cache()
    return path


def episode(job):
    record, mode, checkpoint, output = job
    deterministic_runtime()
    torch.set_num_threads(1)
    seed = record['seed']
    session = DomainSession(Sparse4DVLAConfigV26(language_max_tokens=128), ACTION_CONTRACT,
                            seed, sample_domain(seed+6001, 0))
    folder = Path(output)/mode/f'episode_{seed}'
    folder.mkdir(parents=True, exist_ok=False)
    q = np.load(Path(record['store'])/'joint.npy', mmap_mode='r')
    commands = np.load(Path(record['store'])/'command.npy', mmap_mode='r')
    last = int(record['valid_times'][-1])
    model = None
    if mode == 'tiny_target':
        saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
        model = TinyTarget(saved['input_dim']).eval()
        model.load_state_dict(saved['model'])
    trace, history, frames = [], [], []
    maximum_coverage = maximum_hold = 0.
    contacts = effectful = rewrites = 0
    started = time.time()
    try:
        initial_metrics = session.reward_state()
        if initial_metrics[1] != 0:
            raise ValueError('initial target overlap')
        session.observe()
        initial = session.reported.copy()
        reset_error = float(np.max(np.abs(initial[:6]-q[0, :6])))
        for step in range(900):
            session.observe()
            index = min(step, last)
            if mode == 'command_replay':
                action = commands[index].copy()
            elif mode == 'target_replay':
                target = q[index, :6] + .055*commands[index]
                action = np.clip((target-session.reported[:6])/.055, -1, 1)
            else:
                x = features(session.reported, initial, history, step)
                with torch.inference_mode():
                    target = model(torch.from_numpy(x)[None]).numpy()[0]
                action = np.clip((target-session.reported[:6])/.055, -1, 1)
            if step % 8 == 0:
                frames.append(session.buffer.rows[-1]['rgb'].copy())
            trace.append(np.r_[session.reported, action, session.reward_state()])
            history.append(action.copy())
            result = session.advance(action)
            metrics = session.reward_state()
            maximum_coverage = max(maximum_coverage, metrics[1])
            maximum_hold = max(maximum_hold, metrics[2])
            contacts += int(result.get('valid_contact', False))
            effectful += int(result.get('effectful_contact', False))
            rewrites += int(result.get('safety_rewrite', False))
            if result['kind'] != 'sampler_cut':
                break
        kind = session.end_kind if session.end_kind != 'sampler_cut' else 'finite_timeout'
        summary = dict(seed=seed, mode=mode, safe_success=kind == 'success', end_kind=kind,
            terminal_reason=session.reason, steps=step+1, seconds=time.time()-started,
            maximum_coverage=float(maximum_coverage), maximum_hold_s=float(maximum_hold),
            valid_contact_steps=contacts, effectful_contact_steps=effectful,
            safety_rewrite_steps=rewrites, safety_rewrite_fraction=rewrites/(step+1),
            initial_coverage=0., reset_joint_error_rad=reset_error,
            diagnostic_only=True, training_seed_replay=True, visual_grounding=False,
            trajectory_reference_used=mode != 'tiny_target', actor_uses_simulator_state=False,
            elapsed_clock_input=mode == 'tiny_target', final_vla_acceptance=False,
            production_admission=False, export_admission=False)
        np.savez_compressed(folder/'trace.npz', trace=np.asarray(trace, np.float32),
                            wrist_rgb=np.asarray(frames, np.uint8))
        _atomic_json(folder/'result.json', summary)
        return summary
    finally:
        session.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--workers', type=int, default=9)
    p.add_argument('--updates', type=int, default=4000)
    p.add_argument('--wall-seconds', type=int, default=1800)
    p.add_argument('--feedback-noise', type=float, default=0.)
    p.add_argument('--phase-only', action='store_true')
    p.add_argument('--only-tiny', action='store_true')
    p.add_argument('--run-label', choices=('Run63', 'Run63b', 'Run63c'), default='Run63')
    a = p.parse_args()
    if not 1 <= a.workers <= 9 or not 100 <= a.updates <= 10000 or not 300 <= a.wall_seconds <= 2400:
        raise ValueError('bounded diagnostic required')
    if not 0 <= a.feedback_noise <= .03 or (a.phase_only and a.feedback_noise):
        raise ValueError('choose one feedback intervention')
    a.output.mkdir(parents=True, exist_ok=False)
    deterministic_runtime()
    records = diagnostic_nine(json.loads(a.manifest.read_text())['records'])
    state = dict(run=a.run_label, status='running', started=time.time(), phase='initializing',
                 step=0, total_updates=a.updates, evaluations=[], training_episodes=9,
                 diagnostic_only=True, visual_grounding=False, target_rate=.80,
                 training_seed_replay_is_not_acceptance=True, actor_uses_simulator_state=False,
                 production_admission=False, export_admission=False, final_vla_acceptance=False,
                 start_stage='CONTACT_TRANSPORT_HOLD', exact_home_evaluated=False,
                 feedback_noise=a.feedback_noise, phase_only=a.phase_only)

    def publish(phase=None):
        if phase:
            state['phase'] = phase
        state.update(updated=time.time(), elapsed_seconds=time.time()-state['started'])
        _atomic_json(a.output/'run_state.json', state)

    publish()
    try:
        checkpoint = train(records, a.output, a.updates, publish, state,
                           state['started']+a.wall_seconds, a.feedback_noise, a.phase_only)
        modes = ('tiny_target',) if a.only_tiny else ('command_replay', 'target_replay', 'tiny_target')
        for mode in modes:
            if time.time() > state['started']+a.wall_seconds-90:
                raise TimeoutError('control probe budget exhausted')
            results = []
            state.update(phase_episodes_completed=0, phase_episodes_total=9)
            publish(mode)
            with ProcessPoolExecutor(a.workers, mp_context=mp.get_context('spawn')) as pool:
                futures = [pool.submit(episode, (r, mode, str(checkpoint), str(a.output))) for r in records]
                for future in as_completed(futures):
                    results.append(future.result())
                    state.update(phase_episodes_completed=len(results), partial=summarize(results))
                    publish()
            summary = summarize(results) | dict(label=mode, diagnostic_only=True,
                visual_grounding=False, pairs={str(r):summarize([x for x in results if x['seed']%9 == r])
                                               for r in range(9)})
            _atomic_json(a.output/mode/'summary.json', dict(summary=summary, results=results))
            state['evaluations'].append(summary)
            publish()
        state['status'] = 'complete_pending_review'
        publish('finished')
    except BaseException as exc:
        state.update(status='failed', error=repr(exc))
        publish()
        raise


if __name__ == '__main__':
    main()
