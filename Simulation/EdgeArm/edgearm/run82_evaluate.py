"""Autonomous learned spatial-memory rollout; no privileged actor arguments."""
from pathlib import Path
import time

import numpy as np
import torch

from .candidate_command_contract_v2 import ACTION_CONTRACT
from .run34_repeat_eval import deterministic_runtime
from .run42.domain import sample_domain
from .run42.session import DomainSession
from .run61_active_view import SurveyAndReturn
from .run67_visual_state import actor_input, language_indices
from .run74_observation_probe import observation_history_indices
from .run78_completion_probe import CompletionLatch, TerminalServo
from .run82_spatial_model import SparseSpatialPolicy
from .sparse_4d_vla_act_v26 import Sparse4DVLAConfigV26
from .train_staged_hybrid_contact_sac import _atomic_json


def episode(job):
    seed, checkpoint, output, mode = job
    if not 97100000 <= seed//9 < 97100004 or mode not in ('memory', 'no_memory', 'no_action_transition'):
        raise ValueError('explicit development-only mechanism probe')
    deterministic_runtime(); torch.set_num_threads(1)
    saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
    if saved.get('policy_kind') != SparseSpatialPolicy.kind or saved.get('memory_update_stride') != 8:
        raise ValueError('wrong policy or observation cadence')
    policy = SparseSpatialPolicy().cuda().eval(); policy.load_state_dict(saved['model'])
    session = DomainSession(Sparse4DVLAConfigV26(language_max_tokens=128,
        visual_memory_mode='episode_anchors_v54'), ACTION_CONTRACT, seed, sample_domain(seed+6001, 0))
    survey = SurveyAndReturn(session.env.model, session.env._ids['tool_site'], session.env._ids['cameras']['wrist'])
    servo = TerminalServo(session.env.model, session.env._ids['tool_site'], 'lift')
    latch = CompletionLatch()
    folder = Path(output)/f'episode_{seed}'; folder.mkdir(parents=True, exist_ok=False)
    selection = language_indices(session.buffer.instruction)
    selected = torch.tensor([selection], device='cuda')
    q_history, commands, frames, trace, point_history = [], [], [], [], []
    contacts = effectful = rewrites = 0
    maximum_coverage = maximum_hold = 0.
    memory = None; last_memory_step = 0; started = time.time(); available_steps = []
    try:
        session.observe(); initial = session.reported[:6].copy()
        if session.reward_state()[1] != 0.: raise ValueError('nonzero initial coverage')
        initial_block = session.env.block_xy().copy()  # audit only
        survey_displacement = 0.; return_error = None
        for step in range(900):
            session.observe(); rows = session.buffer.rows
            previous = rows[-2]['applied_target'] if step else initial
            dummy = actor_input(session.reported, rows[-1]['tool'], q_history, commands, previous,
                                initial, np.zeros((7, 2), np.float32), selection)
            proprio = torch.from_numpy(np.r_[dummy[:108], dummy[112:118]])[None].cuda()
            with torch.inference_mode():
                if step % 8 == 0 or step in (90, 110, 220):
                    available_steps.append(step)
                    requested = observation_history_indices(step, step >= 220)
                    positions = np.maximum(0, np.searchsorted(available_steps, requested, side='right')-1)
                    ids = np.asarray(available_steps)[positions]
                    rgb = torch.from_numpy(np.stack([rows[int(i)]['rgb'] for i in ids]))[None].cuda()
                    pose = torch.from_numpy(np.stack([rows[int(i)]['camera_pose'] for i in ids]))[None].cuda()
                    age = torch.tensor([[rows[int(i)]['time']-rows[-1]['time'] for i in ids]], device='cuda')
                    K = torch.from_numpy(session.buffer.K)[None].cuda()
                    dt = torch.tensor([(step-last_memory_step)/30], device='cuda')
                    out, memory = policy.step(rgb, pose, K, age, proprio, selected, dt, memory,
                        erase_memory=mode == 'no_memory', erase_action=mode == 'no_action_transition')
                    last_memory_step = step
                    point_history.append(np.r_[step, out['points'][0].cpu().numpy().ravel()])
                if step < 220:
                    action = survey.command(session.reported)
                else:
                    world = memory['xyz'][0].cpu().numpy()
                    action = policy.command(proprio, selected, memory)[0].cpu().numpy().clip(-1, 1)
                    if latch.update(world[list(selection), :2], step):
                        action = servo.command(session.reported, previous)
            # All true object geometry below is AFTER the selected action.
            truth = np.stack((session.env.block_xy(), session.env.target_xy))
            if step == 220: return_error = float(np.max(np.abs(session.reported[:6]-initial)))
            if step < 220:
                survey_displacement = max(survey_displacement, float(np.linalg.norm(truth[0]-initial_block)))
            estimate = memory['xyz'][0, list(selection), :2].cpu().numpy()
            trace.append(np.r_[step, estimate.ravel(), truth.ravel(), session.reward_state(),
                np.linalg.norm(estimate-truth, axis=1), latch.latched])
            q_history.append(session.reported.copy()); commands.append(action.copy())
            if step % 8 == 0: frames.append(rows[-1]['rgb'].copy())
            result = session.advance(action); metrics = session.reward_state()
            contacts += int(result.get('valid_contact', False))
            effectful += int(result.get('effectful_contact', False))
            rewrites += int(result.get('safety_rewrite', False))
            maximum_coverage = max(maximum_coverage, metrics[1]); maximum_hold = max(maximum_hold, metrics[2])
            if result['kind'] != 'sampler_cut': break
        kind = session.end_kind if session.end_kind != 'sampler_cut' else 'finite_timeout'
        result = dict(seed=seed, mode=mode, safe_success=kind == 'success', end_kind=kind,
            terminal_reason=session.reason, steps=len(commands), seconds=time.time()-started,
            maximum_coverage=float(maximum_coverage), maximum_hold_s=float(maximum_hold),
            valid_contact_steps=contacts, effectful_contact_steps=effectful, safety_rewrite_steps=rewrites,
            actor_uses_simulator_state=False, teacher_assisted=False, learned_sparse_spatial_memory=True,
            sparse_surface_points=56, memory_update_stride=8, actuator_feedback_hz=30,
            scripted_completion=True, fixed_wrist_survey=True, fixed_survey_steps=220,
            survey_return_error_rad=return_error, survey_maximum_block_displacement_m=survey_displacement,
            original_ACT_checkpoint=False, independent_acceptance=False,
            production_admission=False, export_admission=False, final_vla_acceptance=False)
        np.savez_compressed(folder/'trace.npz', audit=np.asarray(trace), reported=np.asarray(q_history),
            command=np.asarray(commands), wrist_rgb=np.asarray(frames), sparse_points=np.asarray(point_history))
        _atomic_json(folder/'result.json', result)
        return result
    finally:
        session.close()


def main():
    import argparse
    from concurrent.futures import ProcessPoolExecutor, as_completed
    import multiprocessing as mp
    from .constrained_recovery_run40 import summarize
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--groups', type=int, choices=(1, 4), default=1)
    p.add_argument('--workers', type=int, choices=range(1, 10), default=3)
    p.add_argument('--mode', choices=('memory', 'no_memory', 'no_action_transition'), default='memory')
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    results = []
    state = dict(run='Run82', status='running', phase='rollout_interface_smoke',
                 phase_episodes_completed=0, phase_episodes_total=a.groups*9,
                 independent_acceptance=False, actor_uses_simulator_state=False,
                 production_admission=False, export_admission=False, final_vla_acceptance=False)
    _atomic_json(a.output/'run_state.json', state)
    try:
        with ProcessPoolExecutor(a.workers, mp_context=mp.get_context('spawn')) as pool:
            jobs = [(g*9+r, a.checkpoint, str(a.output), a.mode)
                    for g in range(97100000, 97100000+a.groups) for r in range(9)]
            for future in as_completed([pool.submit(episode, j) for j in jobs]):
                results.append(future.result())
                state.update(phase_episodes_completed=len(results), partial=summarize(results))
                _atomic_json(a.output/'run_state.json', state)
        summary = summarize(results) | dict(
            block_out_of_bounds=sum(r['terminal_reason'] == 'block_out_of_bounds' for r in results))
        _atomic_json(a.output/'summary.json', dict(summary=summary, results=results))
        state.update(status='complete_pending_review', phase='finished', summary=summary)
        _atomic_json(a.output/'run_state.json', state)
    except BaseException as exc:
        state.update(status='failed', error=repr(exc)); _atomic_json(a.output/'run_state.json', state); raise


if __name__ == '__main__': main()
