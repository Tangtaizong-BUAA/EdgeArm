"""State-RL teacher supplement, with online causal wrist/action recording.

The teacher uses simulator state; the exported student inputs do not. One
unchanged Hybrid SAC checkpoint is replicated across CPU workers; two GPUs
render the physical wrist cameras. No optimizer or new expert policy exists.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from copy import deepcopy
import json
import multiprocessing as mp
import os
from pathlib import Path
import shutil
import time
from types import SimpleNamespace
from unittest.mock import patch

os.environ.setdefault('MUJOCO_GL', 'egl')
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
import mujoco
import numpy as np
from PIL import Image
import torch

from . import train_staged_hybrid_contact_sac as teacher_module
from .candidate_command_contract_v2 import ACTION_CONTRACT
from .contact_audit_run33 import profile_failure
from .hybrid_contact_sac import HybridContactSACConfig, initialize_hybrid_contact_sac, load_hybrid_bundle_state_dict
from .materialize_temporal_packets_v3 import kinematics
from .run42.domain import DomainScene, sample_domain
from .run43_collection_plan import admission, counts, next_wave, strata, teacher_wave
from .run43_data_collection import CollectionEpisode
from .staged_push_rl import StagedPushEpisode
from .temporal_online_v3 import TemporalOnlineBuffer
from .train_multimodal_act_v5 import sha256
from .train_staged_hybrid_contact_sac import _atomic_json

TEACHER = None


class TaskGoalChoice(DomainScene):
    @property
    def failed(self):
        # Compatibility with the OLD option executor's extra termination rule.
        # Incidental block contacts remain in the ledger, but the user-approved
        # task_goal_v1 does not terminate on them. Hard contacts are checked
        # independently by the recorder after every physical env.step.
        return False

    def audit(self):
        return super().audit() | dict(contact_profile='task_goal_v1',
            unselected_objects_untouched=(self.invalid_contact_events == 0
                and self.max_unselected_displacement_m <= .002))


class PhysicalStop(RuntimeError):
    pass


class TeacherRecorder(CollectionEpisode):
    def __init__(self, episode, job):
        self.path = Path(job['output'])
        self.path.mkdir(parents=True, exist_ok=False)
        e = episode.env
        cam = e._ids['cameras']['wrist']
        f = 60 / np.tan(np.deg2rad(e.model.cam_fovy[cam])/2)
        k = np.asarray([[f, 0, 79.5], [0, f, 59.5], [0, 0, 1]], np.float32)
        cfg = SimpleNamespace(visual_history_steps=5, proprio_history_steps=8)
        self.session = SimpleNamespace(env=e, buffer=TemporalOnlineBuffer(cfg,
            episode.multichoice.contract['instruction'], k), end_kind='sampler_cut')
        self.renderer = mujoco.Renderer(e.model, width=640, height=480)
        self.rows, self.telemetry, self.parts = [], [], []
        self.persisted, self.started, self.job = 0, time.monotonic(), job
        self.initial_coverage = float(e.block_target_coverage())
        self.maximum_cov, self.maximum_hold, self.reason = self.initial_coverage, 0., 'nonterminal'
        self.rewrites = self.contacts = self.effectful = 0
        _atomic_json(self.path/'scenario.json', dict(scene=episode.multichoice.contract,
            parameters=job['parameters'], source='privileged_hybrid_sac_with_option_executor',
            teacher_sha256=job['teacher_sha256'], action_contract=ACTION_CONTRACT,
            contact_profile='task_goal_v1', start_stage='CONTACT_TRANSPORT_HOLD',
            teacher_uses_simulator_state=True, simulator_state_is_student_input=False,
            K=k.tolist(), focal_fovy_deg=float(e.model.cam_fovy[cam])))
        self.original_step = e.step
        e.step = self.step

    def observe(self):
        e, b = self.session.env, self.session.buffer
        if b.pending:
            return
        self.renderer.update_scene(e.data, camera='edgearm_wrist')
        rgb = np.asarray(Image.fromarray(self.renderer.render()).resize((160, 120)))
        joint = np.asarray(e.observation()['joint_state'], np.float32)
        tool, pose = kinematics(e, joint[:6], joint[6:])
        b.observe(rgb=rgb, joint=joint, tool=tool, camera_pose=pose,
                  time_s=float(e.data.time), geometry_valid=True)

    def step(self, command):
        self.observe()
        before = self.session.env.block_xy().copy()
        result = self.original_step(command)
        _, reward, _, _, info = result
        e, b = self.session.env, self.session.buffer
        after_q = np.asarray(e.observation()['joint_state'], np.float32)
        transfer = info['sim2real_v2']
        b.complete_action(submitted_command=command,
            applied_target=transfer['applied_queued_safe_joint_target'], reported_next_q=after_q[:6],
            feedback_valid=not transfer['submitted_command_ingress_lost'])
        self.rows.append(deepcopy(b.rows[-1]))
        coverage = float(e.block_target_coverage())
        hold = float(e._strict_success_streak) * e.control_dt
        self.maximum_cov = max(self.maximum_cov, coverage)
        self.maximum_hold = max(self.maximum_hold, hold)
        self.rewrites += bool(e.last_command_feedback_v1.get('applied_action_was_safety_modified', False))
        trace = info['physics_substep_contact_v1']
        hard = profile_failure('task_goal_v1', trace, False, False)
        self.telemetry.append([float(e.data.time), float(e.distance_to_target()), coverage, hold,
                               1., float(np.linalg.norm(e.block_xy()-before)), float(hard), float(reward)])
        self.observe()  # Explicit next-state endpoint, never a future actor input.
        if hard:
            self.session.end_kind, self.reason = 'hard_failure', 'unsafe_contact'
            raise PhysicalStop('forbidden physical contact')
        if len(self.rows) % 128 == 0:
            self.persist(final=True)
        elif len(self.rows) % 32 == 0:
            self.persist()
        return result

    def summary(self):
        e = self.session.env
        return dict(seed=self.job['seed'], source='privileged_hybrid_sac_with_option_executor',
            steps=len(self.rows), end_kind=self.session.end_kind, reason=self.reason,
            safe_success=self.session.end_kind=='success', initial_coverage=self.initial_coverage,
            maximum_coverage=self.maximum_cov, maximum_hold_s=self.maximum_hold,
            training_rollin_steps=0, rejected_commands=int(self.reason.startswith('v10_safety_filter')),
            final_reward_state=[float(e.distance_to_target()),float(e.block_target_coverage()),
                                float(e._strict_success_streak)*e.control_dt],
            safety_rewrite_steps=self.rewrites, elapsed_seconds=time.monotonic()-self.started,
            instruction=self.session.buffer.instruction, start_stage='CONTACT_TRANSPORT_HOLD',
            exact_home_evaluated=False, production_admission=False, export_admission=False,
            final_vla_acceptance=False)

    def close(self):
        self.session.env.step = self.original_step
        self.renderer.close()


def initialize_teacher(checkpoint, device_queue):
    global TEACHER
    os.environ['MUJOCO_EGL_DEVICE_ID'] = str(device_queue.get())
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    state = torch.load(checkpoint, map_location='cpu', weights_only=False)
    config = HybridContactSACConfig(**state['configuration'])
    bundle = initialize_hybrid_contact_sac(43043, device='cpu', config=config)
    load_hybrid_bundle_state_dict(bundle, state, config=config, load_actor_optimizer=False,
                                  load_critic_optimizer=False)
    TEACHER = bundle


def collect_teacher(job):
    recorder = None
    episode = None

    def factory(*a, **kw):
        nonlocal episode
        episode = StagedPushEpisode(*a, **kw)
        episode.multichoice = TaskGoalChoice(episode.env, job['parameters'])
        original_reset = episode.reset

        def reset(*args, **kwargs):
            nonlocal recorder
            obs = original_reset(*args, **kwargs)
            recorder = TeacherRecorder(episode, job)
            return obs
        episode.reset = reset
        return episode

    try:
        with patch.object(teacher_module, 'StagedPushEpisode', factory):
            try:
                summary, _ = teacher_module.run_learned_stage_one_episode(TEACHER,
                    seed=job['seed'], rng=np.random.default_rng(job['seed']), device='cpu',
                    deterministic=True, scene_mode='multichoice_v1')
                recorder.reason = summary['terminal_reason']
                recorder.session.end_kind = 'success' if summary['strict_success'] else 'task_failure'
                _atomic_json(recorder.path/'teacher_option_audit.json', summary)
            except PhysicalStop:
                pass
        recorder.observe()
        recorder.persist(final=True)
        result = recorder.summary() | dict(causal_packet_audit=recorder.audit())
    except Exception as error:
        # A broken recording/control transaction cannot become a demonstration.
        path = Path(job['output'])
        path.mkdir(parents=True, exist_ok=True)
        result = dict(seed=job['seed'], safe_success=False, end_kind='collection_error',
                      reason=repr(error), accepted=False)
    finally:
        if recorder is not None:
            recorder.close()
    result.update(stratum=job['stratum'], split=job['split'], geometry_group=job['seed']//9,
        pair=[job['seed']%9//3, job['seed']%3], episode_path=job['output'],
        teacher_sha256=job['teacher_sha256'])
    result['accepted'] = admission(result)
    _atomic_json(Path(job['output'])/'result.json', result)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--max-hours', type=float, default=4.)
    p.add_argument('--seed-base', type=int, default=47000000)
    p.add_argument('--pilot-only', action='store_true')
    p.add_argument('--resume',action='store_true')
    args = p.parse_args()
    if not 1 <= args.workers <= 36 or not 0 < args.max_hours <= 8:
        raise ValueError('invalid bounded execution')
    args.output.mkdir(parents=True, exist_ok=args.resume)
    groups = [g for g in strata() if not g['jitter_m']]
    records, eligible = [], set()
    started = time.time()
    state = dict(status='running', phase='pilot', started=started, deadline=started+3600*args.max_hours,
        teacher_sha256=sha256(args.checkpoint), source='privileged_hybrid_sac_with_option_executor',
        model_updates=0, quotas=groups, records=records, production_admission=False,
        export_admission=False, final_vla_acceptance=False, act_training_started=False,
        exact_home_evaluated=False, start_stage='CONTACT_TRANSPORT_HOLD', action_contract=ACTION_CONTRACT)
    if args.resume:
        saved=json.loads((args.output/'run_state.json').read_text())
        if saved['status']!='paused_for_resize' or saved['teacher_sha256']!=state['teacher_sha256']:
            raise ValueError('resume requires drained owned run and unchanged teacher')
        state=saved
        records=state['records']
        eligible=set(state['eligible_strata'])
        state['status']='running'
    state.update(workers=args.workers,scheduler='continuous_refill_low_yield_one_slot_v3',seed_base=args.seed_base)

    def publish():
        state.update(updated=time.time(), counts=counts(groups, records), completed=len(records),
                     accepted=sum(r['accepted'] for r in records), eligible_strata=sorted(eligible))
        _atomic_json(args.output/'run_state.json', state)
        _atomic_json(args.output/'success_manifest.json', dict(records=[r for r in records if r['accepted']],
            complete=False, production_admission=False, export_admission=False))

    context = mp.get_context('spawn')
    devices = context.Queue()
    for w in range(args.workers):
        devices.put(w % 2)
    publish()
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=context,
            initializer=initialize_teacher, initargs=(str(args.checkpoint), devices)) as pool:
        futures={}
        stop_reason=None

        def job_for(g):
            gi=next(i for i,x in enumerate(groups) if x['name']==g['name'])
            geometry=args.seed_base+gi*100000+g['attempt']
            return dict(seed=geometry*9+g['pair'],parameters=sample_domain(geometry+43,0),
                output=str(args.output/g['name']/str(geometry*9+g['pair'])),
                teacher_sha256=state['teacher_sha256'],stratum=g['name'],
                split='validation' if geometry%10==0 else 'train')

        try:
            while True:
                if state['phase']=='bulk':
                    if time.time()>state['deadline']-300:
                        stop_reason=stop_reason or 'time_budget_stopped'
                    if shutil.disk_usage(args.output).free<20*1024**3:
                        stop_reason=stop_reason or 'disk_budget_stopped'
                    jobs=[] if stop_reason else [job_for(g) for g in teacher_wave(groups,records,
                        max(0,args.workers*2-len(futures)),eligible,in_flight=futures.values())]
                    # Publish assignments BEFORE submitting: a resize either
                    # drains every recorded job or safely resumes this parent.
                    state['current_wave']=[{k:v for k,v in j.items() if k!='parameters'}
                                           for j in list(futures.values())+jobs]
                    publish()
                    for job in jobs:
                        futures[pool.submit(collect_teacher,job)]=job
                    if not futures:
                        state['status']=stop_reason or 'partial_quota_complete'
                        break
                    done,_=wait(futures,timeout=10,return_when=FIRST_COMPLETED)
                    for future in done:
                        result=future.result()
                        futures.pop(future)
                        records.append(result)
                        if result['end_kind']=='collection_error':
                            stop_reason='recording_error_needs_review'
                    state['current_wave']=[{k:v for k,v in j.items() if k!='parameters'} for j in futures.values()]
                    publish()
                    continue
                if time.time() > state['deadline']-300:
                    state['status'] = 'time_budget_stopped'
                    break
                if shutil.disk_usage(args.output).free < 20*1024**3:
                    state['status'] = 'disk_budget_stopped'
                    break
                wave = (teacher_wave(groups, records, args.workers, eligible) if state['phase']=='bulk'
                        else next_wave(groups, records, args.workers, phase='pilot'))
                if not wave:
                    if state['phase']=='pilot' and not args.pilot_only:
                        eligible = {g['name'] for g in groups if g['pair'] != 4 and counts(groups, records)[g['name']]['accepted']}
                        if not eligible:
                            state['status']='no_edge_teacher_yield'
                            break
                        state['phase']='bulk'
                        publish()
                        continue
                    state['status']='pilot_complete' if args.pilot_only else 'partial_quota_complete'
                    break
                jobs = []
                for g in wave:
                    gi = next(i for i, x in enumerate(groups) if x['name']==g['name'])
                    geometry = args.seed_base + gi*100000 + g['attempt']
                    jobs.append(dict(seed=geometry*9+g['pair'], parameters=sample_domain(geometry+43, 0),
                        output=str(args.output/g['name']/str(geometry*9+g['pair'])),
                        teacher_sha256=state['teacher_sha256'], stratum=g['name'],
                        split='validation' if geometry % 10 == 0 else 'train'))
                state['current_wave'] = [{k:v for k,v in j.items() if k!='parameters'} for j in jobs]
                publish()
                records.extend(pool.map(collect_teacher, jobs))
                publish()
                errors = [r for r in records[-len(jobs):] if r['end_kind']=='collection_error']
                if errors:
                    state.update(status='recording_error_needs_review', errors=errors)
                    break
        except BaseException as error:
            state.update(status='failed', error=repr(error))
            raise
        finally:
            state['current_wave'] = []
            publish()


if __name__=='__main__':
    main()
