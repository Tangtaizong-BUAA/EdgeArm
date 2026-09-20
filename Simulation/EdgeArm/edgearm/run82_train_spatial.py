"""Continuous-sequence joint spatial-memory/action learning, bounded pilot."""
import argparse
from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import shutil
import time

import numpy as np
import torch
from torch import nn

from .constrained_recovery_run40 import summarize
from .run34_repeat_eval import deterministic_runtime
from .run74_observation_probe import observation_history_indices
from .run82_prepare_sequences import training_split
from .run82_spatial_model import SparseSpatialPolicy, detach_state, spatial_losses
from .train_staged_hybrid_contact_sac import _atomic_json


class Sequences:
    def __init__(self, manifest):
        self.records = json.loads(Path(manifest).read_text())['records']
        self.cache = OrderedDict()
        for r in self.records:
            if r['split'] != training_split(r['seed']):
                raise ValueError('sequence split mismatch')

    def load(self, row):
        folder = row['folder']
        if folder not in self.cache:
            with np.load(Path(folder)/'inputs.npz', allow_pickle=False) as z:
                inputs = {k: z[k].copy() for k in z.files}
            if set(inputs) != {'rgb', 'pose', 'K', 'proprio', 'time_step', 'selected'}:
                raise ValueError('actor packet contains undeclared fields')
            with np.load(Path(folder)/'labels.npz', allow_pickle=False) as z:
                labels = {k: z[k].copy() for k in z.files}
            if np.any(np.diff(inputs['time_step']) <= 0) or inputs['proprio'].shape[-1] != 114:
                raise ValueError('causal action history required')
            self.cache[folder] = inputs, labels
            while len(self.cache) > 36:
                self.cache.popitem(last=False)
        else:
            self.cache.move_to_end(folder)
        return self.cache[folder]


def sequence_chunk(items, start, length, *, blind=None, device='cuda'):
    """All history indices precede current input; hidden state is never labelled."""
    inputs = {k: [] for k in ('rgb', 'pose', 'K', 'age', 'proprio', 'selected', 'dt')}
    labels = {k: [] for k in ('xyz', 'points', 'present', 'visible', 'command', 'action_valid', 'prior_valid', 'frame_valid')}
    valid_rows = []
    for n, (obs, supervision) in enumerate(items):
        times = obs['time_step']; count = len(times)
        indices = np.minimum(np.arange(start, start+length), count-1)
        valid = np.arange(start, start+length) < count
        blind_frames = np.zeros(count, bool) if blind is None else blind[n]
        visibility = supervision['visible'] & ~blind_frames[:, None]
        ever = np.maximum.accumulate(visibility, axis=0)
        prior_valid = np.r_[np.zeros((1, 7), bool), ever[:-1]]
        history = []
        for k in indices:
            requested = observation_history_indices(int(times[k]), times[k] >= 220)
            ids = np.maximum(0, np.searchsorted(times, requested, side='right')-1)
            if np.any(times[ids] > times[k]):
                raise ValueError('future visual observation')
            history.append(ids)
        history = np.asarray(history)
        rgb = obs['rgb'][history].copy()
        rgb[blind_frames[history]] = 0
        inputs['rgb'].append(rgb); inputs['pose'].append(obs['pose'][history])
        inputs['K'].append(np.broadcast_to(obs['K'], (length, 3, 3)))
        inputs['age'].append((times[history]-times[indices, None]).astype(np.float32)/30)
        inputs['proprio'].append(obs['proprio'][indices])
        inputs['selected'].append(np.broadcast_to(obs['selected'], (length, 2)))
        dt = (times[indices]-times[np.maximum(indices-1, 0)])/30
        inputs['dt'].append(dt.astype(np.float32))
        for key in ('xyz', 'points', 'command'):
            labels[key].append(supervision[key][indices])
        labels['present'].append(supervision['present'][indices] & ever[indices] & valid[:, None])
        labels['prior_valid'].append(supervision['present'][indices] & prior_valid[indices] & valid[:, None])
        labels['visible'].append(visibility[indices])
        labels['action_valid'].append(supervision['action_valid'][indices] & valid)
        labels['frame_valid'].append(valid)
        valid_rows.append(valid)
    as_tensor = lambda rows: torch.from_numpy(np.asarray(rows).copy()).to(device)
    return ({k: as_tensor(v) for k, v in inputs.items()},
            {k: as_tensor(v) for k, v in labels.items()}, as_tensor(valid_rows))


def encode_chunk(model, inputs):
    b, t = inputs['rgb'].shape[:2]
    xy, feature = model.observation(inputs['rgb'].flatten(0, 1), inputs['pose'].flatten(0, 1),
                                   inputs['K'].flatten(0, 1), inputs['age'].flatten(0, 1))
    return xy.reshape(b, t, 7, 2), feature.reshape(b, t, 7, 128)


def validate(model, dataset, *, erase_memory=False, erase_action=False, blind_test=False):
    # Two held-out-from-this-fit physical groups, original colors only. These
    # scenes were used by an older model, so not an independent acceptance set.
    rows = [r for r in dataset.records if r['split'] == 'validation' and r['variant'] == -1]
    if not rows:
        return dict(samples=0, reason='smoke_manifest_has_no_validation')
    errors, unseen, maes = [], [], []
    model.eval()
    with torch.inference_mode():
        for row in rows:
            item = dataset.load(row); n = len(item[0]['time_step'])
            times = item[0]['time_step']
            blind = [(times >= 360) & (times < 420)] if blind_test else None
            state = None
            for start in range(0, n, 16):
                ins, labs, valid = sequence_chunk([item], start, min(16, n-start), blind=blind)
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    xy, features = encode_chunk(model, ins)
                    for t in range(ins['rgb'].shape[1]):
                        current = {k: v[:, t] for k, v in ins.items()}
                        out, state = model.step(**current, state=state, encoded=(xy[:, t], features[:, t]),
                            erase_memory=erase_memory, erase_action=erase_action)
                        selection = current['selected']
                        distance = (out['xyz']-labs['xyz'][:, t]).norm(dim=-1).gather(1, selection)
                        known = labs['present'][:, t].gather(1, selection)
                        visible = labs['visible'][:, t].gather(1, selection)
                        errors.extend(distance[known].float().cpu().tolist())
                        unseen.extend(distance[known & ~visible].float().cpu().tolist())
                        if bool(labs['action_valid'][0, t]):
                            maes.append(float((out['action']-labs['command'][:, t]).abs().mean()))
    return dict(samples=len(errors), position_mean_mm=float(np.mean(errors)*1000),
                position_p90_mm=float(np.quantile(errors, .9)*1000),
                occluded_samples=len(unseen), occluded_mean_mm=float(np.mean(unseen)*1000) if unseen else None,
                command_mae=float(np.mean(maes)), erase_memory=erase_memory,
                erase_action_transition=erase_action, blind_test=blind_test,
                independent_acceptance=False)


def save_model(model, path, state):
    torch.save(dict(model={k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
        policy_kind=model.kind, step=state['step'], memory_update_stride=8,
        actor_uses_simulator_state=False, original_ACT_checkpoint=False,
        production_admission=False, export_admission=False, final_vla_acceptance=False), path)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('manifest', 'vision', 'control', 'output'):
        p.add_argument('--'+key, type=Path, required=True)
    p.add_argument('--updates', type=int, default=1500)
    p.add_argument('--eval-every', type=int, default=500)
    p.add_argument('--chunk', type=int, default=12)
    p.add_argument('--workers', type=int, default=9)
    p.add_argument('--wall-seconds', type=int, default=3300)
    p.add_argument('--smoke-only', action='store_true')
    a = p.parse_args()
    if not 2 <= a.updates <= 3000 or not 2 <= a.chunk <= 16 or not 1 <= a.workers <= 9:
        raise ValueError('bounded pilot training')
    if not 120 <= a.wall_seconds <= 6600:
        raise ValueError('bounded wall time')
    deterministic_runtime(); torch.set_num_threads(2)
    a.output.mkdir(parents=True, exist_ok=False)
    state = dict(run='Run82', status='running', phase='initializing', started=time.time(),
        step=0, total_updates=a.updates, evaluations=[], training_kind='joint_sparse_world_memory_supervision_not_RL',
        source_manifest=str(a.manifest), sparse_landmarks=7, sparse_surface_points=56,
        sparse_neighbors=3, memory_update_stride=8, current_teacher_action_input=False,
        prior_truth_teacher_forcing=False, fixed_survey_steps=220, max_steps=900,
        original_ACT_checkpoint=False, independent_acceptance=False, actor_uses_simulator_state=False,
        production_admission=False, export_admission=False, final_vla_acceptance=False)
    def publish(phase=None):
        if phase: state['phase'] = phase
        state.update(updated=time.time(), elapsed_seconds=time.time()-state['started'])
        _atomic_json(a.output/'run_state.json', state)
    snapshot = a.output/'source_snapshot'; snapshot.mkdir()
    for filename in ('run82_spatial_model.py', 'run82_train_spatial.py', 'run82_evaluate.py',
                     'run67_visual_state.py', 'run78_completion_probe.py'):
        source = Path(__file__).with_name(filename)
        if source.exists(): shutil.copy2(source, snapshot/filename)
    state['source_hashes'] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in snapshot.iterdir()}
    state['manifest_sha256'] = hashlib.sha256(a.manifest.read_bytes()).hexdigest()
    publish(); deadline = state['started']+a.wall_seconds
    try:
        dataset = Sequences(a.manifest)
        rows = [r for r in dataset.records if r['split'] == 'train']
        pools = [[r for r in rows if r['route'] == route] for route in range(9)]
        if not a.smoke_only and any(not pool for pool in pools):
            raise ValueError('equal nine-route physical coverage required')
        model = SparseSpatialPolicy().cuda()
        model.initialize_from(torch.load(a.vision, map_location='cpu', weights_only=True),
                              torch.load(a.control, map_location='cpu', weights_only=True))
        groups = [dict(params=model.visual.parameters(), lr=1e-5),
                  dict(params=model.control.parameters(), lr=3e-5),
                  dict(params=[p for name, p in model.named_parameters()
                               if not name.startswith(('visual.', 'control.'))], lr=2e-4)]
        optimizer = torch.optim.AdamW(groups, weight_decay=1e-5, fused=True)
        rng = np.random.default_rng(8201)
        best_rank = None
        state.update(parameters=sum(p.numel() for p in model.parameters()),
            trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
            training_episodes=len({r['seed'] for r in rows}), training_sequences=len(rows),
            validation_episodes=len({r['seed'] for r in dataset.records if r['split'] == 'validation'}))
        if not a.smoke_only:
            from .run78_completion_probe import episode as baseline_episode
            baseline = []; baseline_folder = a.output/'frozen_baseline'
            state.update(phase_episodes_completed=0, phase_episodes_total=36)
            publish('frozen_baseline_development')
            with ProcessPoolExecutor(a.workers, mp_context=mp.get_context('spawn')) as pool:
                jobs = [(g*9+r, str(a.vision), str(a.control), str(baseline_folder), 'lift')
                        for g in range(97100000, 97100004) for r in range(9)]
                for future in as_completed([pool.submit(baseline_episode, j) for j in jobs]):
                    baseline.append(future.result())
                    state.update(phase_episodes_completed=len(baseline), partial=summarize(baseline)); publish()
            state['frozen_baseline'] = summarize(baseline) | dict(
                pairs={str(r): summarize([x for x in baseline if x['seed'] % 9 == r]) for r in range(9)})
            _atomic_json(baseline_folder/'summary.json', dict(summary=state['frozen_baseline'], results=baseline))
            publish()
        while state['step'] < a.updates:
            if time.time() > deadline-240:
                raise TimeoutError('reserve closed-loop evaluation time')
            chosen = [pool[int(rng.integers(len(pool)))] for pool in pools if pool]
            items = [dataset.load(r) for r in chosen]
            blind = []
            for obs, _ in items:
                times = obs['time_step']; mask = np.zeros(len(times), bool)
                if rng.random() < .35 and times[-1] > 400:
                    begin = int(rng.integers(260, max(261, int(times[-1])-60)))
                    mask = (times >= begin) & (times < begin+48)
                blind.append(mask)
            memory = None; model.train()
            length = max(len(item[0]['time_step']) for item in items)
            for start in range(0, length, a.chunk):
                if state['step'] >= a.updates: break
                if time.time() > deadline-240: raise TimeoutError('reserve evaluation time')
                ins, labs, valid = sequence_chunk(items, start, min(a.chunk, length-start), blind=blind)
                optimizer.zero_grad(set_to_none=True)
                sums = {}; loss = 0.
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    xy, features = encode_chunk(model, ins)
                    for t in range(ins['rgb'].shape[1]):
                        current = {k: v[:, t] for k, v in ins.items()}
                        out, memory = model.step(**current, state=memory, encoded=(xy[:, t], features[:, t]))
                        labels = {k: v[:, t] for k, v in labs.items()}
                        piece, metrics = spatial_losses(out, labels, model.control.yscale.clamp_min(.05),
                            action_weight=.25 if state['step'] < min(200, a.updates//4) else 1.)
                        loss = loss+piece
                        for key, value in metrics.items(): sums[key] = sums.get(key, 0.)+float(value.detach())
                    loss = loss/ins['rgb'].shape[1]
                loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
                optimizer.step(); memory = detach_state(memory)
                state['step'] += 1
                state['metrics'] = {k: v/ins['rgb'].shape[1] for k, v in sums.items()} | dict(loss=float(loss.detach()))
                if state['step'] % 20 == 0 or state['step'] == 1:
                    with (a.output/'training_metrics.jsonl').open('a') as log:
                        log.write(json.dumps(dict(step=state['step'], metrics=state['metrics']))+'\n')
                    publish('joint_spatial_memory_training')
                if state['step'] % a.eval_every == 0 or state['step'] == a.updates:
                    checkpoint = a.output/f'spatial_step_{state["step"]}.pt'
                    save_model(model, checkpoint, state); state['latest_checkpoint'] = str(checkpoint)
                    if a.smoke_only:
                        publish(); continue
                    publish('sequence_memory_validation')
                    state['spatial_validation'] = validate(model, dataset)
                    publish('autonomous_development')
                    from .run82_evaluate import episode
                    results = []
                    folder = a.output/f'development_{state["step"]}'
                    state.update(phase_episodes_total=36, phase_episodes_completed=0); publish()
                    with ProcessPoolExecutor(a.workers, mp_context=mp.get_context('spawn')) as pool:
                        jobs = [(g*9+r, str(checkpoint), str(folder), 'memory')
                                for g in range(97100000, 97100004) for r in range(9)]
                        for future in as_completed([pool.submit(episode, j) for j in jobs]):
                            results.append(future.result())
                            state.update(phase_episodes_completed=len(results), partial=summarize(results)); publish()
                    summary = summarize(results) | dict(step=state['step'],
                        block_out_of_bounds=sum(r['terminal_reason'] == 'block_out_of_bounds' for r in results),
                        pairs={str(r): summarize([x for x in results if x['seed'] % 9 == r]) for r in range(9)})
                    _atomic_json(folder/'summary.json', dict(summary=summary, results=results))
                    state['evaluations'].append(summary)
                    rank = (summary['successes'], -summary['hard_failures'], summary['mean_coverage'])
                    if best_rank is None or rank > best_rank:
                        best_rank = rank; state.update(best_checkpoint=str(checkpoint), best_development_rank=list(rank))
                    publish()
                    model.train()
        if not a.smoke_only and time.time() < deadline-120:
            publish('memory_action_ablations')
            best = torch.load(state['best_checkpoint'], map_location='cpu', weights_only=True)
            model.load_state_dict(best['model'])
            state['ablations'] = dict(full=validate(model, dataset),
                no_memory=validate(model, dataset, erase_memory=True),
                no_action_transition=validate(model, dataset, erase_action=True),
                blind_full=validate(model, dataset, blind_test=True))
        state.update(status='smoke_complete' if a.smoke_only else 'complete_pending_review')
        publish('finished')
    except BaseException as exc:
        state.update(status='failed', error=repr(exc)); publish(); raise


if __name__ == '__main__': main()
