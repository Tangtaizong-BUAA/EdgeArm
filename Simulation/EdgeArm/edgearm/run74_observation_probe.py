"""Paired observation-and-memory diagnosis on a broader development cohort.

The 220-step observation prefix is scripted, not learned; it consumes the same
900-step task budget. No teacher, object truth, or future frame enters policy.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import multiprocessing as mp
from pathlib import Path
import time

import numpy as np

from .constrained_recovery_run40 import summarize
from .run61_active_view import SurveyAndReturn
from .train_staged_hybrid_contact_sac import _atomic_json


def observation_history_indices(step, retain_overview=False):
    from .run67_visual_state import history_indices
    ids = history_indices(step)
    if retain_overview:
        if step < 220: raise ValueError('overview anchors require the completed real observation prefix')
        ids[:2] = [90, 110]
    if np.any(ids > step): raise ValueError('future observation forbidden')
    return ids


def observation_prefix(session, initial, q_history, commands, frames):
    """Only reported q and static calibration enter the scripted controller."""
    control = SurveyAndReturn(session.env.model, session.env._ids['tool_site'],
                              session.env._ids['cameras']['wrist'])
    initial_block = session.env.block_xy().copy()  # audit-only, never passed to control
    audit = dict(steps=0, maximum_block_displacement_m=0., learned_observation=False,
                 controller_uses_object_truth=False)
    for step in range(control.prefix_steps):
        session.observe()
        action = control.command(session.reported)
        q_history.append(session.reported.copy()); commands.append(action.copy())
        if step % 8 == 0: frames.append(session.buffer.rows[-1]['rgb'].copy())
        result = session.advance(action)
        audit['steps'] = step+1
        audit['maximum_block_displacement_m'] = max(audit['maximum_block_displacement_m'],
            float(np.linalg.norm(session.env.block_xy()-initial_block)))
        if result['kind'] != 'sampler_cut': break
    session.observe()
    audit['reported_return_error_rad'] = float(np.max(np.abs(session.reported[:6]-initial)))
    audit['end_kind'] = session.end_kind
    return audit


def main():
    from .run67_visual_state import evaluate_episode
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('vision', 'control', 'tracker', 'output'): p.add_argument('--'+key, type=Path, required=True)
    p.add_argument('--workers', type=int, default=9)
    p.add_argument('--groups', type=int, default=4)
    p.add_argument('--group-start', type=int, default=97100000)
    a = p.parse_args()
    if not 1 <= a.workers <= 9 or not 2 <= a.groups <= 4 or a.group_start != 97100000:
        raise ValueError('fixed broader development cohort, not independent acceptance')
    a.output.mkdir(parents=True, exist_ok=False)
    state = dict(run='Run74', status='running', phase='initializing', started=time.time(), evaluations=[],
        training_kind='frozen_observation_memory_diagnosis_no_training', step=0, total_updates=0,
        max_steps=900, fixed_survey_steps=220, actor_uses_simulator_state=False,
        original_ACT_checkpoint=False, independent_acceptance=False,
        development_groups=list(range(a.group_start, a.group_start+a.groups)),
        next_reserved_independent_groups=list(range(98000100, 98000108)),
        hashes={k: hashlib.sha256(getattr(a, k).read_bytes()).hexdigest() for k in ('vision', 'control', 'tracker')},
        production_admission=False, export_admission=False, final_vla_acceptance=False)
    def publish(phase=None):
        if phase: state['phase'] = phase
        state.update(updated=time.time(), elapsed_seconds=time.time()-state['started'])
        _atomic_json(a.output/'run_state.json', state)
    publish()
    try:
        conditions = [('without_survey', False, False), ('survey_recent_memory', True, False),
                      ('survey_retained_overview', True, True)]
        for label, survey, anchors in conditions:
            results = []; folder = a.output/label
            state.update(phase_episodes_completed=0, phase_episodes_total=9*a.groups); publish(label)
            options = dict(fixed_wrist_survey=survey, retain_overview=anchors)
            with ProcessPoolExecutor(a.workers, mp_context=mp.get_context('spawn')) as pool:
                jobs = [(g*9+r, str(a.vision), str(a.control), str(folder), str(a.tracker), options)
                        for g in state['development_groups'] for r in range(9)]
                for future in as_completed([pool.submit(evaluate_episode, job) for job in jobs]):
                    results.append(future.result())
                    state.update(phase_episodes_completed=len(results), partial=summarize(results)); publish()
            summary = summarize(results) | dict(label=label, fixed_wrist_survey=survey, retain_overview=anchors,
                teacher_assisted=False, independent_acceptance=False,
                pairs={str(r): summarize([s for s in results if s['seed'] % 9 == r]) for r in range(9)})
            _atomic_json(folder/'summary.json', dict(summary=summary, results=results))
            state['evaluations'].append(summary); publish()
        state['status'] = 'complete_pending_review'; publish('finished')
    except BaseException as exc:
        state.update(status='failed', error=repr(exc)); publish(); raise


if __name__ == '__main__': main()
