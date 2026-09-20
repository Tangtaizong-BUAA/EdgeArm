"""Paired, deployable-input completion control probe after renewed authorization.

Frozen Run77 perception and motion policy are unchanged. The optional finishing
servo is explicitly scripted, not newly learned ACT/RL. It uses estimated XY,
reported joints, and static arm kinematics only. Truth is audit-only.
"""
import argparse
from collections import deque
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import multiprocessing as mp
from pathlib import Path
import time

import mujoco
import numpy as np
import torch

from .candidate_command_contract_v2 import ACTION_CONTRACT
from .constrained_recovery_run40 import summarize
from .run34_repeat_eval import deterministic_runtime
from .run42.domain import sample_domain
from .run42.session import DomainSession
from .run63_control_probe import TinyTarget
from .run67_visual_state import VisualState, actor_input, language_indices
from .run74_observation_probe import observation_prefix, observation_history_indices
from .sparse_4d_vla_act_v26 import Sparse4DVLAConfigV26
from .train_staged_hybrid_contact_sac import _atomic_json


class CompletionLatch:
    def __init__(self, radius=.017, consecutive=5):
        if not 0 < radius <= .02 or not 2 <= consecutive <= 15:
            raise ValueError('bounded visual completion margin')
        self.radius = radius
        self.history = deque(maxlen=consecutive)
        self.latched = False
        self.trigger_step = None

    def update(self, selected_xy, step):
        xy = np.asarray(selected_xy, np.float64)
        if xy.shape != (2, 2) or not np.isfinite(xy).all():
            raise ValueError('two finite image-estimated XY coordinates required')
        self.history.append(xy.copy())
        if not self.latched and len(self.history) == self.history.maxlen:
            history = np.stack(self.history)
            near = np.linalg.norm(history[:, 0]-history[:, 1], axis=1)
            spread = np.max(np.linalg.norm(history[:, 0]-history[-1, 0], axis=1))
            if np.all(near <= self.radius) and spread <= .010:
                self.latched, self.trigger_step = True, int(step)
        return self.latched


class TerminalServo:
    def __init__(self, model, tool_site, mode, joint_delta=.055):
        if mode not in ('hold', 'lift'):
            raise ValueError('explicit finishing mode required')
        self.model, self.data = model, mujoco.MjData(model)
        self.tool_site, self.mode, self.joint_delta = tool_site, mode, joint_delta
        self.anchor = None

    def command(self, reported, previous_target):
        q = np.asarray(reported, np.float64)[:6]
        previous = np.asarray(previous_target, np.float64)
        if q.shape != (6,) or previous.shape != (6,) or not np.isfinite(np.r_[q, previous]).all():
            raise ValueError('finite reported state and completed target required')
        if self.anchor is None:
            # Preserve the last actual queued target; do not integrate fresh
            # reported-position noise as zero-relative commands would do.
            self.anchor = previous.copy()
        if self.mode == 'lift':
            self.data.qpos[:6] = q
            self.data.qvel[:] = 0
            mujoco.mj_forward(self.model, self.data)
            if self.data.site_xpos[self.tool_site, 2] < .12:
                jp = np.zeros((3, self.model.nv)); jr = np.zeros_like(jp)
                mujoco.mj_jacSite(self.model, self.data, jp, jr, self.tool_site)
                j = jp[:, :5]
                dq = j.T @ np.linalg.solve(j @ j.T+1e-4*np.eye(3), [0., 0., .0015])
                self.anchor = q.copy()
                self.anchor[:5] += np.clip(dq, -.018, .018)
            else:
                self.mode = 'hold'
                self.anchor = previous.copy()
        return np.clip((self.anchor-q)/self.joint_delta, -1., 1.).astype(np.float32)


def episode(job):
    seed, vision_path, control_path, output, mode = job
    if mode not in ('baseline', 'hold', 'lift') or not 97100000 <= seed//9 < 97100004:
        raise ValueError('this paired mechanism probe only uses existing development groups')
    deterministic_runtime(); torch.set_num_threads(1)
    saved = torch.load(vision_path, map_location='cpu', weights_only=True)
    visual = VisualState(recent_block_seconds=saved.get('recent_block_seconds')).cuda().eval()
    visual.load_state_dict(saved['model'])
    control_saved = torch.load(control_path, map_location='cpu', weights_only=True)
    increment_policy = control_saved.get('policy_kind') == 'target_increment_v79'
    if increment_policy:
        from .run79_target_increment import TargetIncrementPolicy
        control = TargetIncrementPolicy().cuda().eval()
    else:
        control = TinyTarget(118, 512).cuda().eval()
    control.load_state_dict(control_saved['model'])
    session = DomainSession(Sparse4DVLAConfigV26(language_max_tokens=128,
        visual_memory_mode='episode_anchors_v54'), ACTION_CONTRACT, seed, sample_domain(seed+6001, 0))
    folder = Path(output)/mode/f'episode_{seed}'; folder.mkdir(parents=True, exist_ok=False)
    selection = language_indices(session.buffer.instruction)
    latch = CompletionLatch()
    servo = TerminalServo(session.env.model, session.env._ids['tool_site'], mode) if mode != 'baseline' else None
    q_history, commands, frames, trace = [], [], [], []
    maximum_coverage = maximum_hold = 0.
    contacts = rewrites = effectful = 0
    trigger_audit = None; started = time.time()
    try:
        session.observe(); initial = session.reported[:6].copy()
        if session.reward_state()[1] != 0: raise ValueError('initial overlap')
        survey_audit = observation_prefix(session, initial, q_history, commands, frames)
        for step in range(len(commands), 900):
            if session.end_kind != 'sampler_cut': break
            session.observe(); rows = session.buffer.rows
            ids = observation_history_indices(step, True)
            rgb = np.stack([rows[int(i)]['rgb'] for i in ids])
            pose = np.stack([rows[int(i)]['camera_pose'] for i in ids])
            age = np.asarray([rows[int(i)]['time']-rows[-1]['time'] for i in ids], np.float32)
            previous = rows[-2]['applied_target'] if step else initial
            with torch.inference_mode():
                world = visual(torch.from_numpy(rgb)[None].cuda(), torch.from_numpy(pose)[None].cuda(),
                    torch.from_numpy(session.buffer.K)[None].cuda(), torch.from_numpy(age)[None].cuda())[0].cpu().numpy()
                x = actor_input(session.reported, rows[-1]['tool'], q_history, commands, previous, initial, world, selection)
                action = control(torch.from_numpy(x)[None].cuda())[0].cpu().numpy().clip(-1, 1)
            near = latch.update(world[list(selection)], step)
            if servo is not None and near:
                action = servo.command(session.reported, previous)
            # All following true geometry is read AFTER selecting the action.
            truth = np.stack((session.env.block_xy(), session.env.target_xy))
            before = session.reward_state()
            if near and trigger_audit is None:
                trigger_audit = dict(step=step, estimated_distance_m=float(np.linalg.norm(world[selection[0]]-world[selection[1]])),
                    true_distance_m=float(before[0]), true_coverage=float(before[1]), audit_only=True)
            trace.append(np.r_[step, world[list(selection)].ravel(), truth.ravel(), before,
                np.linalg.norm(world[list(selection)]-truth, axis=1), int(near)])
            q_history.append(session.reported.copy()); commands.append(action.copy())
            if step % 8 == 0: frames.append(rows[-1]['rgb'].copy())
            result = session.advance(action); metrics = session.reward_state()
            maximum_coverage = max(maximum_coverage, metrics[1]); maximum_hold = max(maximum_hold, metrics[2])
            contacts += int(result.get('valid_contact', False)); rewrites += int(result.get('safety_rewrite', False))
            effectful += int(result.get('effectful_contact', False))
            if result['kind'] != 'sampler_cut': break
        kind = session.end_kind if session.end_kind != 'sampler_cut' else 'finite_timeout'
        result = dict(seed=seed, mode=mode, safe_success=kind=='success', end_kind=kind,
            terminal_reason=session.reason, steps=len(commands), seconds=time.time()-started,
            maximum_coverage=float(maximum_coverage), maximum_hold_s=float(maximum_hold),
            valid_contact_steps=contacts, effectful_contact_steps=effectful, safety_rewrite_steps=rewrites,
            trigger_audit=trigger_audit, survey_audit=survey_audit, actor_uses_simulator_state=False,
            fixed_wrist_survey=True, fixed_survey_steps=220, max_steps=900, initial_coverage=0.,
            teacher_assisted=False, scripted_completion=servo is not None, model_updated=increment_policy,
            policy_kind='target_increment_v79' if increment_policy else 'relative_joint_command_v77',
            independent_acceptance=False, original_ACT_checkpoint=False,
            production_admission=False, export_admission=False, final_vla_acceptance=False)
        np.savez_compressed(folder/'trace.npz', audit=np.asarray(trace), reported=np.asarray(q_history),
            command=np.asarray(commands), wrist_rgb=np.asarray(frames))
        _atomic_json(folder/'result.json', result)
        return result
    finally:
        session.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('vision', 'control', 'output'): p.add_argument('--'+key, type=Path, required=True)
    p.add_argument('--workers', type=int, default=9)
    p.add_argument('--groups', type=int, choices=(1, 4), default=4)
    p.add_argument('--modes', nargs='+', choices=('baseline', 'hold', 'lift'), default=['baseline', 'hold', 'lift'])
    a = p.parse_args()
    if not 1 <= a.workers <= 9 or len(set(a.modes)) != len(a.modes): raise ValueError('bounded paired probe')
    a.output.mkdir(parents=True, exist_ok=False)
    state = dict(run='Run78', status='running', started=time.time(), step=0, total_updates=0,
        phase='paired_completion_probe', evaluations=[], modes=a.modes,
        training_kind='frozen_model_scripted_completion_ablation_not_training', target_rate=.8,
        renewed_compute_authorization=True, old_six_hour_budget_reused=False,
        model_updated=False, actor_uses_simulator_state=False, independent_acceptance=False,
        vision_sha256=hashlib.sha256(a.vision.read_bytes()).hexdigest(),
        control_sha256=hashlib.sha256(a.control.read_bytes()).hexdigest(),
        production_admission=False, export_admission=False, final_vla_acceptance=False)
    def publish():
        state.update(updated=time.time(), elapsed_seconds=time.time()-state['started'])
        _atomic_json(a.output/'run_state.json', state)
    try:
        for mode in a.modes:
            results = []; state.update(mode=mode, phase_episodes_completed=0, phase_episodes_total=a.groups*9); publish()
            jobs = [(g*9+r, str(a.vision), str(a.control), str(a.output), mode)
                for g in range(97100000, 97100000+a.groups) for r in range(9)]
            with ProcessPoolExecutor(a.workers, mp_context=mp.get_context('spawn')) as pool:
                for future in as_completed([pool.submit(episode, j) for j in jobs]):
                    results.append(future.result())
                    state.update(phase_episodes_completed=len(results), partial=summarize(results)); publish()
            summary = summarize(results)|dict(mode=mode,
                block_out_of_bounds=sum(r['terminal_reason']=='block_out_of_bounds' for r in results),
                trigger_count=sum(r['trigger_audit'] is not None for r in results),
                pairs={str(r):summarize([x for x in results if x['seed']%9==r]) for r in range(9)})
            _atomic_json(a.output/mode/'summary.json', dict(summary=summary, results=results))
            state['evaluations'].append(summary); publish()
        state.update(status='complete_pending_review', phase='finished'); publish()
    except BaseException as exc:
        state.update(status='failed', error=repr(exc)); publish(); raise


if __name__ == '__main__': main()
