"""One batched causal encoder per GPU, CPU physics and bounded IPC queues."""

from dataclasses import asdict
import multiprocessing as mp
from multiprocessing.shared_memory import SharedMemory
import os
from pathlib import Path
from queue import Empty, Full
import time
import traceback

import numpy as np
import torch

from ..constrained_recovery_run40 import BoundedActor
from ..continuous_recovery_run38 import FrozenReadout
from ..dual_gpu_sampler_v1 import CPUEpisode, ReadoutClient, SamplerPool, shared_views
from ..evaluate_multimodal_act_v5 import load_model
from ..run34_repeat_eval import deterministic_runtime
from ..sampler_run38 import tensor_batch
from ..sparse_4d_vla_act_v26 import Sparse4DVLAConfigV26
from ..train_multimodal_act_v5 import sha256
from ..train_staged_hybrid_contact_sac import _atomic_json
from .domain import add_context, domain_context
from .sensors import EstimatedDepthPerturbation
from .session import DomainSession
from .reset_failure import ik_reset_audit, record_invalid_reset


SENSOR_KEYS = ('seed', 'stage', 'depth_scale_error', 'depth_bias_m', 'depth_noise_relative',
               'depth_dropout', 'uncertainty_scale')


class SensorClient(ReadoutClient):
    context = None

    def __call__(self, inputs):
        if self.context is None:
            raise ValueError('missing causal sensor context')
        descriptor = self.shared.write(inputs)
        self.sequence += 1
        self.requests.put(dict(worker=self.worker, sequence=self.sequence, sent=time.monotonic(),
                               context=self.context, **descriptor), timeout=self.timeout)
        response = self.replies.get(timeout=self.timeout)
        if response.get('error'):
            raise RuntimeError(response['error'])
        if response['sequence'] != self.sequence:
            raise RuntimeError('observation response identity mismatch')
        return torch.from_numpy(response['x']), torch.from_numpy(response['base'])


class DomainEpisode(CPUEpisode):
    def __init__(self, readout, contract, job):
        from ..run61_active_view import validate_survey_job
        self.active_wrist_survey = validate_survey_job(job)
        self.readout = readout
        self.session = DomainSession(readout.config, contract, job['seed'], job['parameters'])
        if job.get('workspace_adapter'):
            from ..run52_workspace import install
            install(self.session, job['workspace_adapter'])
        self.path = Path(job['output'])
        self.path.mkdir(parents=True, exist_ok=False)
        self.rng = np.random.default_rng(job['sampling_seed'])
        self.source, self.seed = job['source'], int(job['seed'])
        self.rows, self.telemetry = [], []
        self.initial = self.session.reward_state()
        self.max_cov, self.max_hold = self.initial[1:]
        self.contacts = self.effectful = self.rewrites = self.rejections = 0
        self.controlled = self.prefix_steps = self.fragment_id = self.persisted = 0
        self.started = time.monotonic()
        self.cached = None
        self.closed = False
        self.parts = []
        self.compress_wrist = bool(job.get('compress_wrist', False))
        self.recovery_teacher = job.get('recovery_teacher')
        self.survey_audit = None
        _atomic_json(self.path / 'scenario.json', dict(parameters=job['parameters'],
                     scene=self.session.episode.multichoice.contract, camera=self.session.camera_audit,
                     policy_version=job['policy_version'], source=self.source,
                     workspace_adapter=job.get('workspace_adapter', 'legacy_v10'),
                     recovery_teacher=self.recovery_teacher,
                     active_wrist_survey=self.active_wrist_survey,
                     actual_K_is_audit_only=True, privileged_context_is_critic_only=True))

    def features(self):
        if self.cached is None:
            p = self.session.parameters
            self.readout.context = dict(domain={k: p[k] for k in SENSOR_KEYS}, time_s=float(self.session.env.data.time))
            x, b, truth = super().features()
            self.cached = x, b, add_context(truth, domain_context(p))
        return self.cached

    def summary(self):
        return super().summary() | dict(domain_stage=self.session.parameters['stage'],
              teacher_assisted=bool(self.recovery_teacher),
              autonomous_policy_success=bool(not self.recovery_teacher and self.session.end_kind=='success'),
              teacher_start_step=None if not self.recovery_teacher else self.recovery_teacher['start'],
              scripted_observation_prefix=self.active_wrist_survey,
              survey_audit=self.survey_audit,
              workspace_adapter=getattr(self.session, 'run52_workspace_stats', None),
              domain_seed=self.session.parameters['seed'], focal_equivalent_mm=self.session.parameters['focal_equivalent_mm'],
              block_count=self.session.parameters['block_count'], target_count=self.session.parameters['target_count'],
              camera_geometric_visibility=self.session.visibility_summary())


def encoder_service(checkpoint, anchor, device, requests, replies, ready_queue, stats_path, max_batch, wait_ms,
                    action_query_index=0):
    handles, views, staging, pending = {}, {}, {}, []
    try:
        deterministic_runtime()
        torch.set_num_threads(2)
        torch.set_num_interop_threads(1)
        device = torch.device(device)
        if device.type == 'cuda':
            torch.cuda.set_device(device)
        base, saved = load_model(checkpoint)
        anchor_state = None if anchor is None else torch.load(anchor, map_location='cpu', weights_only=True)
        if anchor_state is not None and anchor_state['base_checkpoint_sha256'] != sha256(checkpoint):
            raise ValueError('anchor lineage mismatch')
        perturbation = EstimatedDepthPerturbation(base.policy.core.depth_student)
        base.policy.core.depth_student = perturbation
        if anchor_state is None:
            from ..run44_posttrain.recovery import ACTPriorReadout
            readout = ACTPriorReadout(base, action_query_index=action_query_index).to(device).eval()
        else:
            if action_query_index:
                raise ValueError('lookahead cannot reuse an old residual anchor')
            readout = FrozenReadout(base, anchor_state).to(device).eval()
        ready_queue.put(dict(config=asdict(base.config), action_contract=saved['action_contract'],
                             feature_dim=readout.feature_dim, action_query_index=action_query_index))
        stats = dict(status='running', device=str(device), batches=0, observations=0, maximum_batch=0,
                     action_query_index=action_query_index,
                     batch_equivalence_checked=False, batch_histogram={})
        started = published = time.monotonic()
        latency = []
        while True:
            first = requests.get()
            if first is None:
                break
            pending = [first]
            deadline = time.monotonic() + wait_ms / 1000
            while len(pending) < max_batch:
                try:
                    request = requests.get(timeout=max(0, deadline - time.monotonic()))
                    if request is None:
                        raise RuntimeError('encoder shutdown with outstanding requests')
                    pending.append(request)
                except Empty:
                    break
            workers = [r['worker'] for r in pending]
            if len(set(workers)) != len(workers):
                raise ValueError('multiple pending observations from same worker')
            for request in pending:
                w = request['worker']
                if set(request['context']['domain']) != set(SENSOR_KEYS):
                    raise ValueError('only sensor uncertainty metadata allowed, no task truth')
                if w not in handles:
                    if set(request['fields']) != set(saved['input_keys']):
                        raise ValueError('causal actor input schema changed')
                    handles[w] = SharedMemory(name=request['name'])
                    views[w] = shared_views(handles[w], request['fields'])
                if handles[w].name != request['name']:
                    raise ValueError('shared observation slot changed')
            n, inputs = len(pending), {}
            for key in saved['input_keys']:
                arrays = [torch.from_numpy(views[w][key]) for w in workers]
                if key not in staging:
                    staging[key] = torch.empty((max_batch, *arrays[0].shape[1:]), dtype=arrays[0].dtype,
                                               pin_memory=device.type == 'cuda')
                torch.cat(arrays, dim=0, out=staging[key][:n])
                inputs[key] = staging[key][:n].to(device, non_blocking=device.type == 'cuda')
            contexts = [r['context'] for r in pending]
            with torch.inference_mode():
                x, b = readout(perturbation.configure(contexts, inputs))
                if n > 1 and not stats['batch_equivalence_checked']:
                    for i in range(min(n, 2)):
                        sub = {k: v[i:i + 1] for k, v in inputs.items()}
                        sx, sb = readout(perturbation.configure(contexts[i:i + 1], sub))
                        if not torch.allclose(sx, x[i:i + 1], atol=1e-4, rtol=1e-4) or not torch.allclose(sb, b[i:i + 1], atol=1e-5, rtol=1e-4):
                            raise ValueError('sensor-aware batch/single encoder mismatch')
                    stats['batch_equivalence_checked'] = True
                x, b = x.cpu().numpy(), b.cpu().numpy()
            now = time.monotonic()
            for i, request in enumerate(pending):
                replies[request['worker']].put(dict(sequence=request['sequence'], x=x[i:i + 1].copy(),
                                                    base=b[i:i + 1].copy()), timeout=10)
                latency.append(now - request['sent'])
            pending = []
            stats['batches'] += 1
            stats['observations'] += n
            stats['maximum_batch'] = max(stats['maximum_batch'], n)
            stats['batch_histogram'][str(n)] = stats['batch_histogram'].get(str(n), 0) + 1
            if now - published > 5:
                stats.update(elapsed_seconds=now - started, mean_batch=stats['observations'] / stats['batches'],
                             observations_per_second=stats['observations'] / (now - started),
                             recent_rpc_p95_seconds=float(np.quantile(latency[-1024:], .95)))
                _atomic_json(Path(stats_path), stats)
                published, latency = now, latency[-1024:]
        stats['status'] = 'stopped'
        _atomic_json(Path(stats_path), stats)
    except BaseException:
        error = traceback.format_exc()
        _atomic_json(Path(stats_path), dict(status='failed', error=error))
        try:
            ready_queue.put_nowait(dict(error=error))
        except Full:
            pass
        for q in replies:
            try:
                q.put_nowait(dict(error=error))
            except Full:
                pass
        raise
    finally:
        views.clear()
        for memory in handles.values():
            memory.close()


def worker_main(slot, config, feature_dim, contract, requests, replies, commands, results, render_device, seed):
    if render_device is not None:
        os.environ['MUJOCO_EGL_DEVICE_ID'] = str(render_device)
    deterministic_runtime()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    client = SensorClient(Sparse4DVLAConfigV26(**config), feature_dim, slot, requests, replies)
    actor = BoundedActor(feature_dim).eval()
    task = runner = None
    try:
        while True:
            task = commands.get()
            if task is None:
                break
            job = task['job']
            began = time.monotonic()
            torch.manual_seed(job['sampling_seed'])
            if job.get('actor_kind') == 'run50_rebased':
                from ..run50_recovery import RebasedActor
                if not isinstance(actor, RebasedActor):
                    actor = RebasedActor(feature_dim).eval()
            elif not isinstance(actor, BoundedActor):
                actor = BoundedActor(feature_dim).eval()
            actor.load_state_dict({k: torch.from_numpy(v) for k, v in job['actor'].items()})
            try:
                runner = DomainEpisode(client, contract, job)
            except RuntimeError as error:
                audit = ik_reset_audit(error)
                if audit is None:
                    raise
                value = record_invalid_reset(job, audit)
                results.put(dict(task=task['id'], worker=slot, value=value,
                                 seconds=time.monotonic() - began), timeout=60)
                continue
            transitions = []
            try:
                demonstration = None
                if job.get('reference_store'):
                    store = Path(job['reference_store'])
                    reference_commands = np.load(store/'command.npy', mmap_mode='r')
                    valid = np.load(store/'command_valid.npy', mmap_mode='r')
                    n = int(valid.sum())
                    if not valid[:n].all() or valid[n:].any():
                        raise ValueError('reference commands must be a complete contiguous prefix')
                    demonstration = reference_commands[:n]
                prefix = job.get('reference_prefix_steps')
                if prefix is not None and (demonstration is None or not 0 <= prefix < len(demonstration)):
                    raise ValueError('curriculum prefix needs a real, shorter reference command sequence')
                pause = int(job.get('prefix_pause_steps', 0)) if prefix is not None else 0
                if not 0 <= pause <= 30:
                    raise ValueError('bounded physical pause required')
                noise = None
                recovery = None
                survey = None
                if runner.active_wrist_survey:
                    from ..run61_active_view import SurveyAndReturn
                    env = runner.session.env
                    survey = SurveyAndReturn(env.model, env._ids['tool_site'], env._ids['cameras']['wrist'])
                    survey_initial_block = env.block_xy().copy()  # audit only, never a controller input
                    survey_max_displacement = 0.
                if job.get('recovery_teacher'):
                    if demonstration is not None or job.get('workspace_adapter'):
                        raise ValueError('recovery collection uses the unchanged controller and no demonstration prefix')
                    from ..run53_recovery import RecoveryTeacher, teacher_phase
                    spec = job['recovery_teacher']
                    teacher_phase(0, spec['start'])
                    recovery = RecoveryTeacher(runner.session, job['seed'] % 9, spec['variant'])
                for step in range(job.get('max_steps', 900)):
                    if runner.session.end_kind != 'sampler_cut':
                        break
                    x, base, _ = runner.features()
                    learn = True
                    if survey is not None and step < survey.prefix_steps:
                        action = survey.command(runner.session.reported)
                        learn = False
                    elif demonstration is not None and prefix is None:
                        if step >= len(demonstration):
                            break  # An unfinished replay stays a sampler cut, never a fake terminal.
                        action = demonstration[step].copy()
                    elif prefix is not None and step < prefix + pause:
                        action = demonstration[step].copy() if step < prefix else np.zeros(6, np.float32)
                        learn = False
                    else:
                        with torch.inference_mode():
                            kwargs = dict(noise_std=job.get('noise_std', 0.))
                            if job.get('actor_kind') == 'run50_rebased' and job.get('noise_hold_steps', 1) > 1:
                                if noise is None or runner.controlled % job['noise_hold_steps'] == 0:
                                    noise = torch.randn(1, 6) * job.get('noise_std', 0.)
                                kwargs['raw_noise'] = noise
                            action = actor(torch.from_numpy(x[None]), torch.from_numpy(base[None]), **kwargs)[0].numpy()
                    if recovery is not None:
                        learn = teacher_phase(step, spec['start'])
                        if learn:
                            action = recovery.command()
                    transition = runner.advance(action, learn=learn)
                    if survey is not None and step < survey.prefix_steps:
                        survey_max_displacement = max(survey_max_displacement,
                            float(np.linalg.norm(runner.session.env.block_xy()-survey_initial_block)))
                        runner.survey_audit = dict(steps=step+1,
                            max_block_displacement_m=survey_max_displacement,
                            reported_return_error_rad=float(np.max(np.abs(
                                runner.session.reported[:6]-survey.initial_q))),
                            geometric_visibility=runner.session.visibility_summary(),
                            controller_uses_object_truth=False)
                    if learn:
                        transitions.append(transition)
                    if (step + 1) % 128 == 0:
                        runner.persist(final=True)
                    elif (step + 1) % 32 == 0:
                        runner.persist(final=False)  # Lightweight live metrics; no extra RGB shard.
                runner.persist(final=True)
                batch = tensor_batch(transitions) if transitions else None
                if batch is not None:
                    np.savez_compressed(runner.path / 'transitions.npz', **batch)
                value = dict(result=runner.summary(), batch=batch, episode_id=job['output'],
                             policy_version=job['policy_version'])
            finally:
                runner.close()
                runner = None
            results.put(dict(task=task['id'], worker=slot, value=value, seconds=time.monotonic() - began), timeout=60)
    except BaseException:
        results.put(dict(task=None if task is None else task['id'], worker=slot, error=traceback.format_exc()), timeout=10)
        raise
    finally:
        if runner is not None:
            runner.close()
        client.close()


class DomainPool(SamplerPool):
    def __init__(self, checkpoint, anchor, output, workers, device, render_device, *, max_batch=16, wait_ms=6., seed=4201,
                 action_query_index=0):
        self.context = mp.get_context('spawn')
        self.requests = self.context.Queue(maxsize=workers)
        self.replies = [self.context.Queue(maxsize=1) for _ in range(workers)]
        self.commands = [self.context.Queue(maxsize=1) for _ in range(workers)]
        self.results = self.context.Queue(maxsize=workers)
        ready = self.context.Queue(maxsize=2)
        self.inference = self.context.Process(target=encoder_service, args=(checkpoint, anchor, device, self.requests,
            self.replies, ready, str(Path(output) / 'inference_state.json'), max_batch, wait_ms, action_query_index))
        self.workers, self.pending, self.sequence = [], {}, 0
        self.inference.start()
        try:
            self.metadata = ready.get(timeout=180)
            if self.metadata.get('error'):
                raise RuntimeError(self.metadata['error'])
            for slot in range(workers):
                process = self.context.Process(target=worker_main, args=(slot, self.metadata['config'],
                    self.metadata['feature_dim'], self.metadata['action_contract'], self.requests, self.replies[slot],
                    self.commands[slot], self.results, render_device, seed + slot))
                process.start()
                self.workers.append(process)
        except BaseException:
            self.close()
            raise
