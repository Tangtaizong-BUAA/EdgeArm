"""Isolated state-policy control diagnostic. NEVER a deployable VLA candidate.

Compare full simulator state with reported proprioception plus true selected
object/goal XY. Both oracle conditions are diagnostic-only and cannot consume
the independent acceptance set or set any deployment admission flag.
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
from .run42.domain import add_context, domain_context, sample_domain
from .run42.session import DomainSession
from .run54_train import diagnostic_nine, verify_nominal_scene
from .run63_control_probe import TinyTarget
from .sampler_run38 import privileged_state
from .sparse_4d_vla_act_v26 import Sparse4DVLAConfigV26
from .train_staged_hybrid_contact_sac import _atomic_json


def history4(values, width):
    out = np.zeros((4, width), np.float32)
    n = min(4, len(values))
    if n:
        out[-n:] = np.asarray(values[-n:])[:, :width]
    return out.ravel()


def state_features(mode, privileged, reported, tool, past_q, past_command, previous_target):
    past = history4(past_command, 6)
    if mode == 'full_state':
        return np.r_[privileged, reported, past].astype(np.float32)
    if mode != 'observable_plus_task':
        raise ValueError('unknown diagnostic input mode')
    task = np.r_[privileged[6:8]-tool[:2], privileged[-24:-22]-tool[:2]]
    return np.r_[reported, tool, history4(past_q, 12), past, previous_target,
                 task].astype(np.float32)


def load_data(records, mode):
    xs, ys, routes = [], [], []
    for row in records:
        path = Path(row['store'])
        q = np.load(path/'joint.npy', mmap_mode='r')
        tool = np.load(path/'tool.npy', mmap_mode='r')
        commands = np.load(path/'command.npy', mmap_mode='r')
        targets = np.load(path/'applied_target.npy', mmap_mode='r')
        with np.load(Path(row['parent_source_path'])/'transitions.npz', allow_pickle=False) as data:
            truth = data['privileged'].copy()
        start = row['teacher_start_step']
        if truth.ndim != 2 or truth.shape[1] != 123:
            raise ValueError('unknown full-state training schema')
        for t in row['valid_times']:
            if not 0 <= t-start < len(truth):
                raise ValueError('truth/command alignment')
            xs.append(state_features(mode, truth[t-start], q[t], tool[t],
                q[max(0, t-4):t], commands[max(0, t-4):t], targets[t-1] if t else q[t, :6]))
            ys.append(commands[t].copy())
            routes.append(row['seed'] % 9)
    return np.asarray(xs), np.asarray(ys), np.asarray(routes)


def fit(records, mode, output, updates, publish, state, deadline):
    x, y, routes = load_data(records, mode)
    x, y = torch.from_numpy(x).cuda(), torch.from_numpy(y).cuda()
    model = TinyTarget(x.shape[1], width=512).cuda()
    with torch.no_grad():
        model.xmean.copy_(x.mean(0)); model.xscale.copy_(x.std(0).clamp_min(.02))
        model.ymean.copy_(y.mean(0)); model.yscale.copy_(y.std(0).clamp_min(.025))
    optimizer = torch.optim.AdamW(model.parameters(), lr=.0005, weight_decay=1e-6, fused=True)
    pools = [torch.from_numpy(np.flatnonzero(routes == r)).cuda() for r in range(9)]
    best, saved = float('inf'), None
    for step in range(updates):
        if time.time() > deadline-180:
            break
        ids = torch.cat([pool[torch.randint(len(pool), (228,), device='cuda')] for pool in pools])
        loss = ((model(x[ids])-y[ids])/model.yscale).square().mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
        optimizer.step()
        for g in optimizer.param_groups:
            g['lr'] = .00002+.00048*.5*(1+np.cos(np.pi*(step+1)/updates))
        if (step+1) % 100 == 0:
            with torch.inference_mode():
                error = torch.cat([(model(x[i:i+4096])-y[i:i+4096]).abs()
                                   for i in range(0, len(x), 4096)])
                mae = float(error.mean())
                metrics = dict(loss=float(loss.detach()), command_mae=mae,
                    target_mae_rad=.055*mae, worst_command_error=float(error.max()),
                    route_command_mae=[float(error[torch.from_numpy(routes == r).cuda()].mean())
                                       for r in range(9)])
            if mae < best:
                best = mae
                saved = {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
            state.update(step=step+1, mode=mode, metrics=metrics)
            publish('fit_'+mode)
    if saved is None:
        raise RuntimeError('no completed state model')
    checkpoint = output/(mode+'.pt')
    torch.save(dict(model=saved, width=512, input_dim=x.shape[1], mode=mode,
                    diagnostic_only=True, actor_uses_simulator_state=True,
                    production_admission=False, export_admission=False,
                    final_vla_acceptance=False), checkpoint)
    del model, optimizer, x, y
    torch.cuda.empty_cache()
    return checkpoint


def evaluate_one(job):
    seed, checkpoint, output, training_record = job
    deterministic_runtime()
    torch.set_num_threads(1)
    saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
    if saved.get('diagnostic_only') is not True or not saved.get('actor_uses_simulator_state'):
        raise ValueError('oracle checkpoints must never masquerade as deployable models')
    model = TinyTarget(saved['input_dim'], saved['width']).eval()
    model.load_state_dict(saved['model'])
    mode = saved['mode']
    parameters = sample_domain(seed+6001, 0)
    session = DomainSession(Sparse4DVLAConfigV26(language_max_tokens=128), ACTION_CONTRACT, seed, parameters)
    folder = Path(output)/f'episode_{seed}'
    folder.mkdir(parents=True, exist_ok=False)
    started = time.time()
    maximum_coverage = maximum_hold = 0.
    contacts = rewrites = effectful = 0
    trace, frames, features_trace = [], [], []
    q_history, commands = [], []
    initial_input_difference = None
    try:
        if session.reward_state()[1] != 0:
            raise ValueError('nonzero initial coverage')
        for step in range(900):
            session.observe()
            row = session.buffer.rows[-1]
            previous_target = session.buffer.rows[-2]['applied_target'] if step else session.reported[:6]
            p = add_context(privileged_state(session), domain_context(parameters))
            x = state_features(mode, p, session.reported, row['tool'], q_history, commands, previous_target)
            if step == 0 and training_record is not None:
                offline, _, _ = load_data([training_record], mode)
                initial_input_difference = float(np.max(np.abs(x-offline[0])))
            with torch.inference_mode():
                action = model(torch.from_numpy(x)[None]).numpy()[0].clip(-1, 1)
            if step % 8 == 0:
                frames.append(row['rgb'].copy())
            features_trace.append(x)
            trace.append(np.r_[session.reported, action, session.reward_state()])
            q_history.append(session.reported.copy()); commands.append(action.copy())
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
        result = dict(seed=seed, mode=mode, safe_success=kind == 'success', end_kind=kind,
            steps=step+1, seconds=time.time()-started, terminal_reason=session.reason,
            maximum_coverage=float(maximum_coverage), maximum_hold_s=float(maximum_hold),
            valid_contact_steps=contacts, effectful_contact_steps=effectful,
            safety_rewrite_steps=rewrites, initial_coverage=0.,
            initial_input_difference=initial_input_difference,
            diagnostic_only=True, visual_grounding=False, actor_uses_simulator_state=True,
            teacher_assisted=False, oracle_diagnostic_not_autonomous_acceptance=True,
            production_admission=False, export_admission=False, final_vla_acceptance=False)
        np.savez_compressed(folder/'trace.npz', trace=np.asarray(trace, np.float32),
            diagnostic_inputs=np.asarray(features_trace, np.float32), wrist_rgb=np.asarray(frames, np.uint8))
        _atomic_json(folder/'result.json', result)
        return result
    finally:
        session.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--updates', type=int, default=6000)
    parser.add_argument('--workers', type=int, default=9)
    parser.add_argument('--wall-seconds', type=int, default=1700)
    parser.add_argument('--all-recoveries', action='store_true')
    parser.add_argument('--modes', nargs='+', choices=('full_state','observable_plus_task'),
                        default=['full_state', 'observable_plus_task'])
    a = parser.parse_args()
    if not 100 <= a.updates <= 15000 or not 1 <= a.workers <= 9 or not 300 <= a.wall_seconds <= 3500:
        raise ValueError('bounded state diagnosis required')
    a.output.mkdir(parents=True, exist_ok=False)
    deterministic_runtime()
    rows = json.loads(a.manifest.read_text())['records']
    nine = diagnostic_nine(rows)
    records = [r for r in rows if r['split'] == 'train' and r['run54_pool'] == 'recovery'] if a.all_recoveries else nine
    for row in records:
        verify_nominal_scene(json.loads((Path(row['parent_source_path'])/'scenario.json').read_text()))
    state = dict(run='Run64', status='running', started=time.time(), step=0,
        total_updates=a.updates, training_episodes=len(records), evaluations=[], target_rate=.80,
        diagnostic_only=True, actor_uses_simulator_state=True, visual_grounding=False,
        independent_acceptance=False, production_admission=False, export_admission=False,
        final_vla_acceptance=False, start_stage='CONTACT_TRANSPORT_HOLD')

    def publish(phase=None):
        if phase:
            state['phase'] = phase
        state.update(updated=time.time(), elapsed_seconds=time.time()-state['started'])
        _atomic_json(a.output/'run_state.json', state)

    def evaluate(mode, checkpoint, seeds, records_by_seed, label):
        results = []
        state.update(mode=mode, phase_episodes_completed=0, phase_episodes_total=len(seeds))
        folder = a.output/label
        publish(label)
        with ProcessPoolExecutor(a.workers, mp_context=mp.get_context('spawn')) as pool:
            futures = [pool.submit(evaluate_one, (s, str(checkpoint), str(folder), records_by_seed.get(s))) for s in seeds]
            for future in as_completed(futures):
                results.append(future.result())
                state.update(phase_episodes_completed=len(results), partial=summarize(results))
                publish()
        summary = summarize(results) | dict(label=label, mode=mode, actor_uses_simulator_state=True,
            independent_acceptance=False, diagnostic_only=True,
            pairs={str(r):summarize([x for x in results if x['seed']%9 == r]) for r in range(9)})
        _atomic_json(folder/'summary.json', dict(summary=summary, results=results))
        state['evaluations'].append(summary)
        publish()
        return summary

    publish('initializing')
    try:
        for mode in a.modes:
            if time.time() > state['started']+a.wall_seconds-240:
                raise TimeoutError('state diagnosis budget exhausted')
            checkpoint = fit(records, mode, a.output, a.updates, publish, state,
                             state['started']+a.wall_seconds)
            result = evaluate(mode, checkpoint, [r['seed'] for r in nine],
                              {r['seed']:r for r in nine}, mode+'_training')
            if result['successes'] >= 8 and result['hard_failures'] == 0:
                evaluate(mode, checkpoint, [g*9+r for g in range(97000000,97000002) for r in range(9)],
                         {}, mode+'_development')
        state['status'] = 'complete_pending_review'
        publish('finished')
    except BaseException as exc:
        state.update(status='failed', error=repr(exc))
        publish()
        raise


if __name__ == '__main__':
    main()
