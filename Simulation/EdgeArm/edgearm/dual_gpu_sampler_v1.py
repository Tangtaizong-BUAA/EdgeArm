"""GPU1 batched frozen encoder + dedicated CPU physics workers.

One shared-memory observation slot per worker; queues carry descriptors, not
multi-megabyte RGB histories. Workers never initialize a CUDA torch model.
Each worker owns its plant across fragments and evaluates in isolated plants.
"""

from copy import deepcopy
from dataclasses import asdict
import multiprocessing as mp
from multiprocessing.shared_memory import SharedMemory
import os
from pathlib import Path
from queue import Empty, Full
import random
import time
import traceback

import numpy as np
import torch

from .continuous_recovery_run38 import ContinuousActor, FrozenReadout
from .dual_gpu_sac_v1 import sample_action
from .evaluate_multimodal_act_v5 import load_model
from .run34_repeat_eval import deterministic_runtime
from .sampler_run38 import EpisodeRun, tensor_batch
from .sparse_4d_vla_act_v26 import Sparse4DVLAConfigV26
from .train_multimodal_act_v5 import sha256
from .train_staged_hybrid_contact_sac import _atomic_json


class SharedInputs:
    def __init__(self):
        self.memory = None
        self.fields = {}
        self.views = {}

    def write(self, inputs):
        if any(v.device.type != 'cpu' for v in inputs.values()):
            raise ValueError("physics worker must send CPU observations")
        arrays = {k: v.detach().contiguous().numpy() for k, v in inputs.items()}
        if self.memory is None:
            offset = 0
            for k, value in arrays.items():
                offset = (offset + 63) // 64 * 64
                self.fields[k] = dict(offset=offset, shape=list(value.shape), dtype=value.dtype.str)
                offset += value.nbytes
            self.memory = SharedMemory(create=True, size=max(offset, 1))
            self.views = shared_views(self.memory, self.fields)
        if arrays.keys() != self.fields.keys():
            raise ValueError("causal input schema changed")
        for key, value in arrays.items():
            if value.shape != self.views[key].shape or value.dtype != self.views[key].dtype:
                raise ValueError("causal input shape/dtype changed")
            np.copyto(self.views[key], value)
        return dict(name=self.memory.name, fields=self.fields)

    def close(self):
        self.views.clear()
        if self.memory is not None:
            self.memory.close()
            self.memory.unlink()
            self.memory = None


def shared_views(memory, fields):
    return {k: np.ndarray(v['shape'], dtype=np.dtype(v['dtype']), buffer=memory.buf, offset=v['offset'])
            for k, v in fields.items()}


class ReadoutClient:
    def __init__(self, config, feature_dim, worker, requests, replies, timeout=180):
        self.config, self.feature_dim = config, feature_dim
        self.worker, self.requests, self.replies = worker, requests, replies
        self.timeout, self.sequence = timeout, 0
        self.shared = SharedInputs()

    def __call__(self, inputs):
        descriptor = self.shared.write(inputs)
        self.sequence += 1
        request = dict(worker=self.worker, sequence=self.sequence, sent=time.monotonic(), **descriptor)
        self.requests.put(request, timeout=self.timeout)
        response = self.replies.get(timeout=self.timeout)
        if response.get('error'):
            raise RuntimeError(response['error'])
        if response['sequence'] != self.sequence:
            raise RuntimeError("out-of-order observation response")
        return torch.from_numpy(response['x']), torch.from_numpy(response['base'])

    def close(self):
        self.shared.close()


def inference_service(checkpoint, anchor, device, requests, replies, ready_queue, stats_path,
                      max_batch=16, wait_ms=3., validate_batch=True):
    handles, views, staging = {}, {}, {}
    pending = []
    try:
        deterministic_runtime()
        torch.set_num_threads(2)
        torch.set_num_interop_threads(1)
        device = torch.device(device)
        if device.type == 'cuda':
            torch.cuda.set_device(device)
        base, saved = load_model(checkpoint)
        saved_anchor = torch.load(anchor, map_location='cpu', weights_only=True)
        if saved_anchor['base_checkpoint_sha256'] != sha256(checkpoint):
            raise ValueError("anchor lineage mismatch")
        readout = FrozenReadout(base, saved_anchor).to(device).eval()
        keys = set(saved['input_keys'])
        ready_queue.put(dict(config=asdict(base.config), action_contract=saved['action_contract'],
                             feature_dim=readout.feature_dim))
        stats = dict(status='running', batches=0, observations=0, maximum_batch=0,
                     batch_histogram={}, batch_equivalence_checked=False, device=str(device))
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
                        raise RuntimeError("inference shutdown while requests remain")
                    pending.append(request)
                except Empty:
                    break
            workers = [r['worker'] for r in pending]
            if len(set(workers)) != len(workers):
                raise ValueError("more than one pending observation per worker")
            for request in pending:
                worker = request['worker']
                if worker not in handles:
                    if set(request['fields']) != keys:
                        raise ValueError("only registered causal actor inputs may enter encoder")
                    handles[worker] = SharedMemory(name=request['name'])
                    views[worker] = shared_views(handles[worker], request['fields'])
                if handles[worker].name != request['name']:
                    raise ValueError("worker changed shared memory slot")
            n = len(pending)
            inputs = {}
            for key in keys:
                arrays = [torch.from_numpy(views[w][key]) for w in workers]
                shape = (max_batch, *arrays[0].shape[1:])
                if key not in staging:
                    staging[key] = torch.empty(shape, dtype=arrays[0].dtype, pin_memory=device.type == 'cuda')
                torch.cat(arrays, dim=0, out=staging[key][:n])
                inputs[key] = staging[key][:n].to(device, non_blocking=device.type == 'cuda')
            with torch.inference_mode():
                x, b = readout(inputs)
                if validate_batch and n > 1 and not stats['batch_equivalence_checked']:
                    for index in range(min(n, 2)):
                        sx, sb = readout({k: v[index:index + 1] for k, v in inputs.items()})
                        if not torch.allclose(sx, x[index:index + 1], atol=1e-4, rtol=1e-4) or not torch.allclose(
                                sb, b[index:index + 1], atol=1e-5, rtol=1e-4):
                            raise ValueError("batched/single causal readout mismatch")
                    stats['batch_equivalence_checked'] = True
                # Copy the small outputs per batch, rather than per worker.
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
                stats.update(mean_batch=stats['observations'] / stats['batches'], elapsed_seconds=now - started,
                             recent_rpc_p95_seconds=float(np.quantile(latency[-1024:], .95)))
                _atomic_json(Path(stats_path), stats)
                published, latency = now, latency[-1024:]
        stats.update(status='stopped', mean_batch=stats['observations'] / max(stats['batches'], 1))
        _atomic_json(Path(stats_path), stats)
    except BaseException:
        error = traceback.format_exc()
        _atomic_json(Path(stats_path), dict(status='failed', error=error))
        try:
            ready_queue.put_nowait(dict(error=error))
        except Full:
            pass
        for queue in replies:
            try:
                queue.put_nowait(dict(error=error))
            except Full:
                pass
        raise
    finally:
        views.clear()
        for memory in handles.values():
            memory.close()


class CPUEpisode(EpisodeRun):
    def __init__(self, readout, checkpoint, seed, path, source, sampling_seed):
        # Run38's defaults are deliberately left untouched for the running job.
        from .branch_session_run37 import BranchSession
        self.readout = readout
        self.session = BranchSession(readout.config, checkpoint['action_contract'], int(seed), 'cpu')
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=False)
        self.rng = np.random.default_rng(sampling_seed)
        self.source, self.seed = source, int(seed)
        self.rows, self.telemetry = [], []
        self.initial = self.session.reward_state()
        self.max_cov, self.max_hold = self.initial[1:]
        self.contacts = self.effectful = self.rewrites = self.rejections = 0
        self.controlled = self.prefix_steps = self.fragment_id = 0
        self.started = time.monotonic()
        self.cached = None
        self.closed = False
        self.persisted = 0
        self.parts = []

    def persist(self, final=False):
        _atomic_json(self.path / 'episode_progress.json', self.summary())
        if not final or len(self.rows) == self.persisted:
            return
        begin, end = self.persisted, len(self.rows)
        rows = self.rows[begin:end] + [deepcopy(self.session.buffer.rows[-1])]
        data = {k: np.asarray([r[k] for r in rows]) for k in rows[0]}
        data['K'] = self.session.buffer.K
        data['evaluation_telemetry'] = np.asarray(self.telemetry[begin:end])
        name = f'wrist_part_{len(self.parts):05d}.npz'
        # Uncompressed incremental RGB: no repeated compression of the entire
        # history at every fragment. Exactly one pending endpoint per part.
        temporary = self.path / (name + '.tmp')
        with temporary.open('wb') as stream:
            writer = np.savez_compressed if getattr(self, 'compress_wrist', False) else np.savez
            writer(stream, **data)
        temporary.replace(self.path / name)
        self.parts.append(dict(path=name, action_begin=begin, action_end=end, endpoint_rows=1))
        self.persisted = end
        _atomic_json(self.path / 'episode_manifest.json', dict(
            schema='run39-wrist-parts-v1', parts=self.parts, instruction=self.session.buffer.instruction,
            complete=self.session.end_kind != 'sampler_cut', export_admission=False,
            note='Each part has n completed action rows plus one next-observation endpoint.'))
        _atomic_json(self.path / 'result.json', self.summary())


def worker_main(slot, config, feature_dim, contract, requests, replies, commands, results, render_device, seed):
    if render_device is not None:
        os.environ['MUJOCO_EGL_DEVICE_ID'] = str(render_device)
    deterministic_runtime()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.manual_seed(seed)
    client = ReadoutClient(Sparse4DVLAConfigV26(**config), feature_dim, slot, requests, replies)
    actor = ContinuousActor(feature_dim).eval()
    checkpoint = dict(action_contract=contract)
    live, number = None, 0
    task = None

    def action(runner, deterministic=False):
        x, base, _ = runner.features()
        with torch.inference_mode():
            inputs = torch.from_numpy(x[None]), torch.from_numpy(base[None])
            if deterministic:
                return actor(*inputs, deterministic=True)[0][0].numpy()
            return sample_action(actor, *inputs)[0][0].numpy()

    try:
        while True:
            task = commands.get()
            if task is None:
                break
            began = time.monotonic()
            job = task['job']
            if task['kind'] == 'collect':
                actor.load_state_dict({k: torch.from_numpy(v) for k, v in job['actor'].items()})
                count = prefix_count = 0
                completed, chunks = [], []
                while count < job['steps']:
                    if live is None:
                        record = job['records'][(slot + number * job['workers']) % len(job['records'])]
                        mode = ('normal', 'normal', 'near', 'recovery')[(slot + number) % 4]
                        path = Path(job['output']) / f'worker_{slot:02d}' / f'episode_{number:06d}_{mode}'
                        live = CPUEpisode(client, checkpoint, record['seed'], path, mode, seed + number)
                        number += 1
                        live.rollin(mode, record['packet'])
                        prefix_count += live.prefix_steps
                        if live.session.end_kind != 'sampler_cut':
                            live.persist(final=True)
                            completed.append(live.summary())
                            live.close()
                            live = None
                            continue
                    rows = []
                    while count < job['steps'] and live.session.end_kind == 'sampler_cut':
                        rows.append(live.advance(action(live), learn=True))
                        count += 1
                    batch = tensor_batch(rows)
                    name = live.path / f'replay_{live.fragment_id:05d}.npz'
                    np.savez(name, **batch, behavior_version=job['policy_version'])
                    chunks.append(dict(episode_id=str(live.path), batch=batch))
                    live.fragment_id += 1
                    live.persist(final=True)
                    if live.session.end_kind != 'sampler_cut':
                        completed.append(live.summary())
                        live.close()
                        live = None
                value = dict(chunks=chunks, completed=completed, controls=count, prefix_controls=prefix_count,
                             policy_version=job['policy_version'], active=None if live is None else live.summary())
            else:
                # Save both stochastic state and actor weights; evaluation is
                # not permitted to alter the paused training actor or plant.
                rng = torch.get_rng_state()
                numpy_rng, python_rng = np.random.get_state(), random.getstate()
                old_actor = {k: v.clone() for k, v in actor.state_dict().items()}
                torch.manual_seed(job['sampling_seed'])
                runner = CPUEpisode(client, checkpoint, job['seed'], job['output'], job['source'], job['sampling_seed'])
                transitions = []
                try:
                    teacher = None
                    if task['kind'] == 'offline':
                        with np.load(job['packet']) as archive:
                            teacher = archive['command'].copy()
                    elif task['kind'] == 'evaluate':
                        actor.load_state_dict({k: torch.from_numpy(v) for k, v in job['actor'].items()})
                    else:
                        raise ValueError("unknown sampler job")
                    limit = min(job.get('max_steps', 900), len(teacher) if teacher is not None else 900)
                    for t in range(limit):
                        command = teacher[t] if teacher is not None else action(runner, deterministic=True)
                        transitions.append(runner.advance(command, learn=True))
                        if (t + 1) % 128 == 0:
                            runner.persist(final=True)
                        if runner.session.end_kind != 'sampler_cut':
                            break
                    runner.persist(final=True)
                    batch = tensor_batch(transitions) if teacher is not None else None
                    if batch is not None:
                        np.savez(runner.path / 'transitions.npz', **batch)
                    value = dict(result=runner.summary(), episode_id=str(runner.path), batch=batch)
                finally:
                    runner.close()
                    actor.load_state_dict(old_actor)
                    torch.set_rng_state(rng)
                    np.random.set_state(numpy_rng)
                    random.setstate(python_rng)
            results.put(dict(task=task['id'], worker=slot, value=value, seconds=time.monotonic() - began), timeout=60)
    except BaseException:
        results.put(dict(task=None if task is None else task['id'], worker=slot, error=traceback.format_exc()), timeout=10)
        raise
    finally:
        if live is not None:
            live.persist(final=True)
            live.close()
        client.close()


class SamplerPool:
    """Pinned worker identities and at most ONE bounded in-flight wave."""

    def __init__(self, checkpoint, anchor, output, workers, inference_device, *, max_batch=16,
                 batch_wait_ms=3., render_device=1, seed=3901, worker_target=worker_main):
        self.context = mp.get_context('spawn')
        self.requests = self.context.Queue(maxsize=workers)
        self.replies = [self.context.Queue(maxsize=1) for _ in range(workers)]
        self.commands = [self.context.Queue(maxsize=1) for _ in range(workers)]
        self.results = self.context.Queue(maxsize=workers)
        ready = self.context.Queue(maxsize=2)
        self.inference = self.context.Process(target=inference_service, args=(
            checkpoint, anchor, inference_device, self.requests, self.replies, ready,
            str(Path(output) / 'inference_state.json'), max_batch, batch_wait_ms))
        self.workers, self.pending, self.sequence = [], {}, 0
        self.inference.start()
        try:
            self.metadata = ready.get(timeout=180)
            if self.metadata.get('error'):
                raise RuntimeError(self.metadata['error'])
            for slot in range(workers):
                process = self.context.Process(target=worker_target, args=(
                    slot, self.metadata['config'], self.metadata['feature_dim'], self.metadata['action_contract'],
                    self.requests, self.replies[slot], self.commands[slot], self.results, render_device, seed + slot * 1000))
                process.start()
                self.workers.append(process)
        except BaseException:
            self.close()
            raise

    def submit(self, kind, jobs):
        if self.pending:
            raise RuntimeError("bounded pipeline already has an in-flight wave")
        if not 0 < len(jobs) <= len(self.workers):
            raise ValueError("one job per pinned worker")
        self.started = time.monotonic()
        for slot, job in enumerate(jobs):
            self.sequence += 1
            self.pending[self.sequence] = slot
            self.commands[slot].put(dict(id=self.sequence, kind=kind, job=job), timeout=10)

    def collect(self, timeout=900):
        if not self.pending:
            raise RuntimeError("no pending sampler wave")
        rows = []
        deadline = self.started + timeout
        while self.pending:
            try:
                row = self.results.get(timeout=min(5, max(.01, deadline - time.monotonic())))
            except Empty:
                if not self.inference.is_alive() or any(not p.is_alive() for p in self.workers):
                    raise RuntimeError("sampler/inference child exited; inspect child traceback")
                if time.monotonic() >= deadline:
                    raise TimeoutError("sampler wave timed out")
                continue
            if row.get('error'):
                raise RuntimeError(row['error'])
            if self.pending.pop(row['task'], None) != row['worker']:
                raise RuntimeError("sampler response identity mismatch")
            rows.append(row)
        rows.sort(key=lambda r: r['worker'])
        return [r['value'] for r in rows], dict(wall_seconds=time.monotonic() - self.started,
                                               slowest_worker_seconds=max(r['seconds'] for r in rows))

    def close(self):
        for queue in self.commands:
            try:
                queue.put_nowait(None)
            except Full:
                pass
        for process in self.workers:
            process.join(timeout=5)
            if process.is_alive():
                process.terminate()  # Only owned simulation child, never an external job.
                process.join(timeout=5)
        try:
            self.requests.put_nowait(None)
        except Full:
            pass
        self.inference.join(timeout=10)
        if self.inference.is_alive():
            self.inference.terminate()
            self.inference.join(timeout=5)
        for queue in [self.requests, self.results, *self.replies, *self.commands]:
            queue.cancel_join_thread()
            queue.close()


def numpy_state(module):
    return {k: v.detach().cpu().numpy().copy() for k, v in module.state_dict().items()}
