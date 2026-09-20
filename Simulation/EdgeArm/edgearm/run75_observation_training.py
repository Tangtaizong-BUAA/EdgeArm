"""Train matching observation histories, rather than bolt them onto a policy.

The fixed 220-command survey is disclosed and consumes the 900-command budget.
Only dedicated collection scenes supply labels. All-color world positions are
training labels, never actor inputs. This is supervised DAgger, not RL.
"""
import argparse
from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from .candidate_command_contract_v2 import ACTION_CONTRACT
from .constrained_recovery_run40 import summarize
from .run34_repeat_eval import deterministic_runtime
from .run42.domain import sample_domain
from .run42.session import DomainSession
from .run53_recovery import RecoveryTeacher
from .run61_active_view import SurveyAndReturn
from .run63_control_probe import TinyTarget
from .run67_visual_state import COLORS, VisualState, actor_input, evaluate_episode, language_indices, visual_batch
from .run68_visual_control import estimated_inputs, fit
from .run74_observation_probe import observation_history_indices
from .sparse_4d_vla_act_v26 import Sparse4DVLAConfigV26
from .train_staged_hybrid_contact_sac import _atomic_json


def collection_seed(seed):
    if not 100100000 <= seed // 9 < 100100012:
        raise ValueError('only the new, predeclared training collection is allowed')
    return seed


def label_positions(session):
    """Called only after the submitted command has been selected."""
    scene = session.episode.multichoice.contract
    env = session.env
    other = [i for i in range(3) if i != scene['selected_block']]
    geoms = [(env._ids['block_geom'], scene['selected_block'])]
    geoms += list(zip(session.episode.multichoice.geom_ids, other))
    xy = np.zeros((7, 2), np.float32); mask = np.zeros(7, bool)
    for geom, slot in geoms:
        index = COLORS.index(scene['block_colors'][slot])
        xy[index] = env.data.geom_xpos[geom, :2]; mask[index] = True
    for slot, color in enumerate(scene['target_colors']):
        index = COLORS.index(color)
        xy[index] = scene['target_positions'][slot]; mask[index] = True
    return xy, mask


def collect(job):
    seed, vision_path, control_path, output = job[:4]
    options = job[4] if len(job)>4 else {}
    if options:
        if set(options)-{'beta','recolor_variants','route6_speed','run76'}: raise ValueError('unknown collection option')
        if options.get('run76') is not True or not 100200000<=seed//9<100200016:
            raise ValueError('dedicated Run76 training groups required')
    else: collection_seed(seed)
    deterministic_runtime(); torch.set_num_threads(1)
    session = DomainSession(Sparse4DVLAConfigV26(language_max_tokens=128,
        visual_memory_mode='episode_anchors_v54'), ACTION_CONTRACT, seed, sample_domain(seed+6001, 0))
    survey = SurveyAndReturn(session.env.model, session.env._ids['tool_site'], session.env._ids['cameras']['wrist'])
    teacher = RecoveryTeacher(session, seed % 9, goal_retreat_coverage=.995)
    # Every fourth group adds genuinely student-visited post-survey states.
    beta = options.get('beta',.5 if (seed // 9 - 100100000) % 4 == 3 else 1.)
    if not 0<=beta<=1: raise ValueError('bounded teacher execution')
    speed=options.get('route6_speed',.3)
    if not .3<=speed<=.6: raise ValueError('bounded teacher speed, unchanged plant')
    if seed%9==6: teacher.p['speed']=speed
    variants=options.get('recolor_variants',0)
    if variants not in (0,3): raise ValueError('three fixed color-counterfactual renders only')
    vision = control = None
    if beta < 1:
        vision = VisualState().cuda().eval()
        vision.load_state_dict(torch.load(vision_path, map_location='cpu', weights_only=True)['model'])
        control = TinyTarget(118, 512).cuda().eval()
        control.load_state_dict(torch.load(control_path, map_location='cpu', weights_only=True)['model'])
    folder = Path(output)/f'episode_{seed}'; folder.mkdir(parents=True, exist_ok=False)
    rng = np.random.default_rng(seed+75)
    q_history, commands, images, poses, labels, masks, states, actions, times = ([] for _ in range(9))
    render_states=[]
    contacts = rewrites = 0; coverage = hold = block_displacement = 0.
    return_error = None; started = time.time()
    try:
        session.observe(); initial = session.reported[:6].copy()
        initial_block = session.env.block_xy().copy()  # audit only
        if session.reward_state()[1] != 0: raise ValueError('initial overlap')
        selection = language_indices(session.buffer.instruction)
        for step in range(900):
            session.observe(); rows = session.buffer.rows
            previous = rows[-2]['applied_target'] if step else initial
            # No object position is placed in the stored proprioceptive fields.
            dummy = actor_input(session.reported, rows[-1]['tool'], q_history, commands,
                                previous, initial, np.zeros((7, 2), np.float32), selection)
            proprio = np.r_[dummy[:108], dummy[112:118]].astype(np.float32)
            if step < 220:
                label = action = survey.command(session.reported)
            else:
                if step == 220: return_error = float(np.max(np.abs(session.reported[:6]-initial)))
                prediction = None
                if vision is not None:
                    ids = observation_history_indices(step, True)
                    rgb = np.stack([rows[int(i)]['rgb'] for i in ids])
                    pose = np.stack([rows[int(i)]['camera_pose'] for i in ids])
                    age = np.asarray([rows[int(i)]['time']-rows[-1]['time'] for i in ids], np.float32)
                    with torch.inference_mode():
                        world = vision(torch.from_numpy(rgb)[None].cuda(), torch.from_numpy(pose)[None].cuda(),
                            torch.from_numpy(session.buffer.K)[None].cuda(), torch.from_numpy(age)[None].cuda())
                        x = estimated_inputs(torch.from_numpy(proprio)[None].cuda(), world,
                                             torch.tensor([selection], device='cuda'))
                        prediction = control(x)[0].cpu().numpy().clip(-1, 1)
                # Privileged label is queried after the student prediction.
                label = teacher.command()
                if step % 15 == 10 or step == 220: use_teacher = rng.random() < beta
                action = label if prediction is None or use_teacher else prediction
            if step % 4 == 0 or step in (90, 110):
                xy, mask = label_positions(session)
                images.append(rows[-1]['rgb'].copy()); poses.append(rows[-1]['camera_pose'].copy())
                labels.append(xy); masks.append(mask); states.append(proprio)
                actions.append(label.copy()); times.append(step)
                if variants: render_states.append(session.env.data.qpos.copy())
            q_history.append(session.reported.copy()); commands.append(action.copy())
            result = session.advance(action); metrics = session.reward_state()
            if step < 220:
                block_displacement = max(block_displacement, float(np.linalg.norm(session.env.block_xy()-initial_block)))
            contacts += int(result.get('valid_contact', False)); rewrites += int(result.get('safety_rewrite', False))
            coverage = max(coverage, metrics[1]); hold = max(hold, metrics[2])
            if result['kind'] != 'sampler_cut': break
        kind = session.end_kind if session.end_kind != 'sampler_cut' else 'finite_timeout'
        np.savez_compressed(folder/'frames.npz', rgb=np.asarray(images, np.uint8), pose=np.asarray(poses, np.float32),
            xy=np.asarray(labels), mask=np.asarray(masks), proprio=np.asarray(states), command=np.asarray(actions),
            time_step=np.asarray(times), K=session.buffer.K, selected=np.asarray(selection, np.int64))
        result = dict(seed=seed, safe_success=kind=='success', end_kind=kind, steps=len(commands),
            seconds=time.time()-started, collection=True, teacher_assisted=True, teacher_beta=beta,
            maximum_coverage=float(coverage), maximum_hold_s=float(hold), valid_contact_steps=contacts,
            safety_rewrite_steps=rewrites, frames=len(images), survey_return_error_rad=return_error,
            survey_maximum_block_displacement_m=block_displacement, instruction=session.buffer.instruction,
            actor_uses_simulator_state=False, teacher_uses_simulator_state=True,
            training_labels_include_failed_episodes=True, production_admission=False,
            export_admission=False, final_vla_acceptance=False)
        result.update(recolor_variants=variants,teacher_route6_speed=speed)
        _atomic_json(folder/'result.json', result)
        if variants:
            from .run76_color_dagger import recolor_episode
            recolor_episode(session,folder,np.asarray(render_states),variants)
        return result
    finally: session.close()


class SurveyFrames(Dataset):
    # Only RGB needs mapping. Mapping every small metadata array multiplied
    # descriptor use by ten and exhausted the worker's 1024-file limit.
    mapped_cache_limit = 32

    def __init__(self, root, validation=False, action_only=False):
        self.rows = []; self.index = []; self.cache = OrderedDict()
        self.action_only = action_only
        for path in sorted(Path(root).glob('episode_*/frames.npz')):
            meta = json.loads((path.parent/'result.json').read_text()); seed = collection_seed(meta['seed'])
            if not meta.get('collection'): raise ValueError('collection metadata required')
            if (seed // 9 >= 100100010) != validation: continue
            row = len(self.rows); self.rows.append((path, seed))
            with np.load(path, allow_pickle=False) as z: times = z['time_step']
            self.index.extend((row, k) for k, t in enumerate(times) if not action_only or t >= 220)
        if not self.index: raise ValueError('empty survey training split')

    def __len__(self): return len(self.index)

    def __getitem__(self, index):
        row, k = self.index[index]; path, seed = self.rows[row]
        if row not in self.cache:
            if path.is_dir():
                while len(self.cache) >= self.mapped_cache_limit:
                    _, evicted = self.cache.popitem(last=False)
                    for value in evicted.values():
                        if isinstance(value, np.memmap): value._mmap.close()
                self.cache[row]={p.stem:np.load(p,mmap_mode='r' if p.stem=='rgb' else None,
                                              allow_pickle=False) for p in path.glob('*.npy')}
            else:
                with np.load(path, allow_pickle=False) as z: self.cache[row] = {key: z[key].copy() for key in z.files}
            # The authorized host has 90 GiB RAM. Retain this bounded 90-episode
            # split per worker to avoid decompressing an episode for each draw.
            while len(self.cache) > 128: self.cache.popitem(last=False)
        else: self.cache.move_to_end(row)
        a = self.cache[row]; t = int(a['time_step'][k])
        requested = observation_history_indices(t, t >= 220)
        ids = np.maximum(0, np.searchsorted(a['time_step'], requested, side='right')-1)
        if np.any(a['time_step'][ids] > t): raise ValueError('future image')
        return dict(rgb=a['rgb'][ids], pose=a['pose'][ids], K=a['K'],
            age=np.asarray((a['time_step'][ids]-t)/30, np.float32), xy=a['xy'][k], mask=a['mask'][k],
            selected=a['selected'], time_step=np.int64(t), proprio=a['proprio'][k], command=a['command'][k],
            route=np.int64(seed % 9))


def route_sampler(dataset, samples):
    routes = np.asarray([dataset.rows[row][1] % 9 for row, _ in dataset.index])
    counts = np.bincount(routes, minlength=9)
    if np.any(counts == 0): raise ValueError('all nine routes required')
    return WeightedRandomSampler(torch.from_numpy(1/counts[routes]), samples, replacement=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('vision', 'control', 'output'): p.add_argument('--'+key, type=Path, required=True)
    p.add_argument('--collection', type=Path)
    p.add_argument('--collect-only', action='store_true'); p.add_argument('--smoke-only', action='store_true')
    p.add_argument('--groups', type=int, default=12); p.add_argument('--workers', type=int, default=9)
    p.add_argument('--visual-updates', type=int, default=3000); p.add_argument('--control-updates', type=int, default=4000)
    p.add_argument('--batch', type=int, default=64); p.add_argument('--wall-seconds', type=int, default=2100)
    a = p.parse_args()
    if not 1 <= a.workers <= 9 or a.groups not in (1, 12) or not 50 <= a.visual_updates <= 6000:
        raise ValueError('bounded observation training required')
    if not 100 <= a.control_updates <= 6000 or not 300 <= a.wall_seconds <= 3500: raise ValueError('finite bounds')
    a.output.mkdir(parents=True, exist_ok=False); deterministic_runtime(); torch.set_num_threads(2)
    state = dict(run='Run75', started=time.time(), status='running', step=0,
        total_updates=0 if a.collect_only else a.visual_updates+a.control_updates,
        evaluations=[], training_kind='matching_active_observation_supervision_not_RL',
        original_ACT_checkpoint=False, fixed_wrist_survey=True, fixed_survey_steps=220,
        retain_overview=True, actor_uses_simulator_state=False, teacher_uses_simulator_state=True,
        max_steps=900, validation_groups=[100100010, 100100011],
        independent_acceptance=False, production_admission=False, export_admission=False, final_vla_acceptance=False)
    def publish(phase=None):
        if phase: state['phase'] = phase
        state.update(updated=time.time(), elapsed_seconds=time.time()-state['started'])
        _atomic_json(a.output/'run_state.json', state)
    publish('initializing'); deadline = state['started']+a.wall_seconds
    try:
        if a.collect_only:
            results = []; state.update(phase_episodes_completed=0, phase_episodes_total=a.groups*9)
            publish('observation_training_collection')
            with ProcessPoolExecutor(a.workers, mp_context=mp.get_context('spawn')) as pool:
                jobs = [(g*9+r, str(a.vision), str(a.control), str(a.output/'collection'))
                        for g in range(100100000, 100100000+a.groups) for r in range(9)]
                for future in as_completed([pool.submit(collect, j) for j in jobs]):
                    results.append(future.result()); state.update(phase_episodes_completed=len(results), partial=summarize(results)); publish()
            state['collection_summary'] = summarize(results)
            _atomic_json(a.output/'collection_summary.json', dict(summary=state['collection_summary'], results=results))
        else:
            if a.collection is None: raise ValueError('dedicated survey collection required')
            dataset = SurveyFrames(a.collection); validation = SurveyFrames(a.collection, validation=True)
            loader = DataLoader(dataset, batch_size=a.batch, sampler=route_sampler(dataset, a.visual_updates*a.batch),
                num_workers=6, pin_memory=True, persistent_workers=True)
            val_loader = DataLoader(validation, batch_size=a.batch, shuffle=False, num_workers=2, pin_memory=True)
            state.update(training_frames=len(dataset), validation_frames=len(validation),
                training_episodes=len(dataset.rows), validation_episodes=len(validation.rows))
            visual = VisualState().cuda(); visual.load_state_dict(torch.load(a.vision, map_location='cpu', weights_only=True)['model'])
            optimizer = torch.optim.AdamW(visual.parameters(), lr=.0001, weight_decay=.0001, fused=True)
            from .run67_visual_state import validate
            state['validation_before'] = validate(visual, val_loader, limit=10000); publish('observation_visual_fit')
            for step, batch in enumerate(loader):
                if time.time() > deadline-600: raise TimeoutError('reserve time for action adaptation and evaluation')
                b = visual_batch(batch)
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    world = visual(b['rgb'], b['pose'], b['K'], b['age'])
                    error = F.smooth_l1_loss(world/.02, b['xy']/.02, beta=.05, reduction='none').mean(-1)
                    loss = ((error*b['mask']).sum(-1)/b['mask'].sum(-1).clamp_min(1)).mean()
                optimizer.zero_grad(set_to_none=True); loss.backward()
                nn.utils.clip_grad_norm_(visual.parameters(), 5, error_if_nonfinite=True); optimizer.step()
                state['step'] = step+1
                if (step+1) % 50 == 0:
                    state['metrics'] = dict(loss=float(loss.detach()), position_training_mean_mm=float(
                        torch.linalg.vector_norm(world.detach()-b['xy'], dim=-1)[b['mask']].mean()*1000)); publish()
            vision_path = a.output/'vision.pt'
            torch.save(dict(model={k:v.detach().cpu().clone() for k,v in visual.state_dict().items()},
                actor_uses_simulator_state=False, original_ACT_checkpoint=False,
                fixed_wrist_survey=True, retain_overview=True), vision_path)
            state['visual_validation'] = validate(visual, val_loader, limit=10000)
            del optimizer, loader, dataset, validation, val_loader
            visual.eval(); action_data = SurveyFrames(a.collection, action_only=True)
            action_loader = DataLoader(action_data, batch_size=96, shuffle=False, num_workers=6, pin_memory=True)
            xs, ys, routes = [], [], []
            with torch.inference_mode():
                for batch in action_loader:
                    b = visual_batch(batch); world = visual(b['rgb'], b['pose'], b['K'], b['age'])
                    xs.append(estimated_inputs(b['proprio'], world, b['selected']).cpu().numpy())
                    ys.append(batch['command'].numpy()); routes.append(batch['route'].numpy())
                    state['recode_completed'] = sum(len(v) for v in xs); publish('observation_control_recode')
            train = tuple(np.concatenate(v) for v in (xs, ys, routes))
            np.savez_compressed(a.output/'recoded_commands.npz', x=train[0], action=train[1], route=train[2])
            del visual, action_loader, action_data; torch.cuda.empty_cache()
            control = TinyTarget(118, 512).cuda()
            control.load_state_dict(torch.load(a.control, map_location='cpu', weights_only=True)['model'])
            a.vision = vision_path; a.updates = a.control_updates; a.learning_rate = .0001
            state.update(round=0, vision_sha256=hashlib.sha256(vision_path.read_bytes()).hexdigest(),
                control_training_states=len(train[0]))
            checkpoint = fit(control, train, None, a, state, publish, deadline)
            del control; torch.cuda.empty_cache()
            state.update(vision_checkpoint=str(vision_path), control_checkpoint=str(checkpoint))
            if not a.smoke_only:
                results = []; folder = a.output/'development'
                state.update(phase_episodes_completed=0, phase_episodes_total=36); publish('observation_learned_development')
                options = dict(fixed_wrist_survey=True, retain_overview=True)
                with ProcessPoolExecutor(a.workers, mp_context=mp.get_context('spawn')) as pool:
                    jobs = [(g*9+r, str(vision_path), str(checkpoint), str(folder), None, options)
                            for g in range(97100000, 97100004) for r in range(9)]
                    for future in as_completed([pool.submit(evaluate_episode, j) for j in jobs]):
                        results.append(future.result()); state.update(phase_episodes_completed=len(results), partial=summarize(results)); publish()
                summary = summarize(results) | dict(pairs={str(r):summarize([x for x in results if x['seed'] % 9 == r]) for r in range(9)})
                _atomic_json(folder/'summary.json', dict(summary=summary, results=results)); state['evaluations'].append(summary)
        state['status'] = 'complete_pending_review'; publish('finished')
    except BaseException as exc:
        state.update(status='failed', error=repr(exc)); publish(); raise


if __name__ == '__main__': main()
