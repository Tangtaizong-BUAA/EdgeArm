"""Two GPU shards of one frozen policy, recording quota-balanced ACT candidates.

No optimizer, hardware control, production export, or automatic ACT training.
The existing stage-1 origin is explicit; these are not Home demonstrations.
"""
import argparse
from copy import deepcopy
from datetime import timedelta
import json
import multiprocessing as mp
import os
from pathlib import Path
import shutil
import time
import traceback

os.environ.setdefault('MUJOCO_GL', 'egl')
import numpy as np
import torch
import torch.distributed as dist

from .constrained_recovery_run40 import BoundedActor
from .dual_gpu_sampler_v1 import SamplerPool, numpy_state
from .run34_repeat_eval import deterministic_runtime
from .sparse_4d_vla_act_v26 import Sparse4DVLAConfigV26
from .train_multimodal_act_v5 import sha256
from .train_staged_hybrid_contact_sac import _atomic_json
from .run42.domain import sample_domain
from .run42.reset_failure import ik_reset_audit, record_invalid_reset
from .run42.sampler import DomainEpisode, SensorClient, encoder_service
from .run43_collection_plan import admission, counts, next_wave, replay_proposal, strata


class CollectionEpisode(DomainEpisode):
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
        temporary = self.path / (name + '.tmp')
        with temporary.open('wb') as stream:
            np.savez_compressed(stream, **data)
        temporary.replace(self.path / name)
        self.parts.append(dict(path=name, action_begin=begin, action_end=end,
            endpoint_rows=1, sha256=sha256(self.path / name), bytes=(self.path / name).stat().st_size))
        self.persisted = end
        _atomic_json(self.path / 'episode_manifest.json', dict(
            schema='run39-wrist-parts-v1', collection_schema='run43-balanced-causal-v1',
            parts=self.parts, instruction=self.session.buffer.instruction,
            complete=self.session.end_kind != 'sampler_cut', export_admission=False,
            ground_truth_geometry_actor_input=False, native_render_wh=[640, 480],
            stored_rgb_wh=[160, 120], depth_source='regenerate RGBDepthStudent from causal RGB',
            note='n completed action rows plus one next-observation endpoint per part'))

    def audit(self):
        problems = []
        previous_endpoint = None
        total = 0
        for part in self.parts:
            with np.load(self.path / part['path'], allow_pickle=False) as z:
                n = part['action_end'] - part['action_begin']
                if len(z['rgb']) != n + 1 or z['rgb'].shape[1:] != (120, 160, 3):
                    problems.append('rgb_count_or_shape')
                if not np.all(np.diff(z['time']) > 0):
                    problems.append('nonmonotonic_time')
                if not z['command_valid'][:-1].all() or z['command_valid'][-1]:
                    problems.append('command_alignment')
                if not z['feedback_valid'][:-1].all():
                    problems.append('missing_execution_feedback')
                for k in ('joint', 'tool', 'camera_pose', 'K', 'command', 'applied_target', 'tracking_error'):
                    if not np.isfinite(z[k]).all():
                        problems.append('nonfinite_' + k)
                if previous_endpoint is not None:
                    for k, value in previous_endpoint.items():
                        if not np.array_equal(value, z[k][0]):
                            problems.append('shard_boundary_' + k)
                previous_endpoint = {k: z[k][-1].copy() for k in ('rgb', 'joint', 'time')}
                total += n
        if total != len(self.rows) or total == 0:
            problems.append('action_count')
        return dict(passed=not problems, problems=sorted(set(problems)), actions=total,
                    stored_rgb_wh=[160, 120], bytes=sum(p['bytes'] for p in self.parts))


def collection_worker(slot, config, feature_dim, contract, requests, replies,
                      commands, results, render_device):
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
            actor.load_state_dict({k: torch.from_numpy(v) for k, v in job['actor'].items()})
            try:
                runner = CollectionEpisode(client, contract, job)
            except RuntimeError as error:
                audit = ik_reset_audit(error)
                if audit is None:
                    raise
                value = record_invalid_reset(job, audit)
                results.put(dict(task=task['id'], worker=slot, value=value, seconds=time.monotonic()-began))
                continue
            try:
                for step in range(900):
                    if runner.session.end_kind != 'sampler_cut':
                        break
                    x, base, _ = runner.features()
                    with torch.inference_mode():
                        action = actor(torch.from_numpy(x[None]), torch.from_numpy(base[None]), noise_std=0.)[0].numpy()
                    # Store raw causal inputs/actions, not duplicate cached latents.
                    runner.advance(action, learn=True)
                    if (step + 1) % 128 == 0:
                        runner.persist(final=True)
                    elif (step + 1) % 32 == 0:
                        runner.persist()
                runner.persist(final=True)
                result = runner.summary() | dict(causal_packet_audit=runner.audit())
                _atomic_json(runner.path / 'result.json', result)
                value = dict(result=result, episode_id=job['output'], policy_version=job['policy_version'])
            finally:
                runner.close()
                runner = None
            results.put(dict(task=task['id'], worker=slot, value=value, seconds=time.monotonic()-began), timeout=60)
    except BaseException:
        results.put(dict(task=None if task is None else task['id'], worker=slot, error=traceback.format_exc()), timeout=10)
        raise
    finally:
        if runner is not None:
            runner.close()
        client.close()


class CollectionPool(SamplerPool):
    def __init__(self, checkpoint, anchor, output, workers, rank):
        self.context = mp.get_context('spawn')
        self.requests = self.context.Queue(maxsize=workers)
        self.replies = [self.context.Queue(maxsize=1) for _ in range(workers)]
        self.commands = [self.context.Queue(maxsize=1) for _ in range(workers)]
        self.results = self.context.Queue(maxsize=workers)
        ready = self.context.Queue(maxsize=2)
        self.inference = self.context.Process(target=encoder_service, args=(
            str(checkpoint), str(anchor), f'cuda:{rank}', self.requests, self.replies, ready,
            str(output / 'inference_state.json'), workers, 10.))
        self.workers, self.pending, self.sequence = [], {}, 0
        self.inference.start()
        try:
            self.metadata = ready.get(timeout=180)
            if self.metadata.get('error'):
                raise RuntimeError(self.metadata['error'])
            for slot in range(workers):
                p = self.context.Process(target=collection_worker, args=(slot, self.metadata['config'],
                    self.metadata['feature_dim'], self.metadata['action_contract'], self.requests,
                    self.replies[slot], self.commands[slot], self.results, rank))
                p.start()
                self.workers.append(p)
        except BaseException:
            self.close()
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint', 'anchor', 'candidate', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--phase', choices=('pilot', 'bulk'), default='pilot')
    parser.add_argument('--pilot-state', type=Path)
    parser.add_argument('--workers-per-gpu', type=int, default=16)
    parser.add_argument('--target', type=int, default=1000)
    parser.add_argument('--seed-base', type=int, default=43000000)
    parser.add_argument('--max-hours', type=float, default=1.)
    parser.add_argument('--disk-floor-gib', type=float, default=20.)
    args = parser.parse_args()
    if not 0 < args.max_hours <= 8 or not 1 <= args.workers_per_gpu <= 24:
        raise ValueError('invalid bounded runtime')
    rank, size = int(os.environ['RANK']), int(os.environ['WORLD_SIZE'])
    if size != 2:
        raise ValueError('requires two GPU shards of the same frozen model')
    torch.cuda.set_device(rank)
    torch.set_num_threads(2)
    deterministic_runtime()
    dist.init_process_group('nccl', timeout=timedelta(minutes=20))
    groups = strata(args.target)
    records, eligible = [], set()
    if args.phase == 'bulk':
        if args.pilot_state is None:
            raise ValueError('bulk requires completed pilot evidence')
        pilot = json.loads(args.pilot_state.read_text())
        if pilot['status'] != 'pilot_complete' or pilot['candidate_sha256'] != sha256(args.candidate):
            raise ValueError('pilot not complete or policy changed')
        eligible = {name for name, c in pilot['counts'].items() if c['accepted'] > 0}
        if 'familiar_center' not in eligible or len(eligible) < 2:
            raise ValueError('no evidence of both retention and expanded coverage')
    start = time.time()
    state = dict(schema='run43-balanced-causal-v1', status='initializing', phase=args.phase,
        started=start, deadline=start+3600*args.max_hours, arguments=vars(args) | {
            k: str(v) for k, v in vars(args).items() if isinstance(v, Path)},
        candidate_sha256=sha256(args.candidate), base_sha256=sha256(args.checkpoint),
        anchor_sha256=sha256(args.anchor), quotas=groups, records=records,
        eligible_strata=sorted(eligible), replay_proposal=replay_proposal(),
        start_stage='CONTACT_TRANSPORT_HOLD', exact_home_evaluated=False,
        production_admission=False, export_admission=False, final_vla_acceptance=False,
        act_training_started=False, model_updates=0, worker_processes_per_gpu=args.workers_per_gpu)
    pool = None

    def publish():
        state.update(updated=time.time(), counts=counts(groups, records), completed=len(records),
                     accepted=sum(r['accepted'] for r in records))
        if rank == 0:
            _atomic_json(args.output / 'run_state.json', state)
            _atomic_json(args.output / 'success_manifest.json', dict(schema=state['schema'],
                records=[r for r in records if r['accepted']], complete=state['status']=='quota_complete',
                production_admission=False, export_admission=False))

    try:
        if rank == 0:
            args.output.mkdir(parents=True, exist_ok=False)
        dist.barrier()
        local = args.output / f'rank_{rank}'
        local.mkdir()
        saved = torch.load(args.candidate, map_location='cpu', weights_only=True)
        if saved['base_sha256'] != state['base_sha256'] or saved['anchor_sha256'] != state['anchor_sha256'] or saved['smoke_only']:
            raise ValueError('candidate lineage mismatch')
        actor = BoundedActor(saved['feature_dim']).eval()
        actor.load_state_dict(saved['actor'])
        actor_state = numpy_state(actor)
        state['policy_version'] = saved['policy_version']
        publish()
        pool = CollectionPool(args.checkpoint, args.anchor, local, args.workers_per_gpu, rank)
        state['action_contract'] = pool.metadata['action_contract']
        while True:
            # One rank decides the stop boundary; ranks must not disagree and
            # strand one another in the next collective at a deadline.
            stop = [None]
            if rank == 0:
                if time.time() >= state['deadline'] - 300:
                    stop[0] = 'time_budget_stopped'
                elif shutil.disk_usage(args.output).free < args.disk_floor_gib * 1024**3:
                    stop[0] = 'disk_budget_stopped'
            dist.broadcast_object_list(stop, src=0)
            if stop[0]:
                state['status'] = stop[0]
                break
            wave = next_wave(groups, records, size * args.workers_per_gpu,
                             phase=args.phase, eligible=eligible)
            if not wave:
                all_filled = all(c['accepted'] >= c['target'] for c in counts(groups, records).values())
                state['status'] = 'pilot_complete' if args.phase == 'pilot' else ('quota_complete' if all_filled else 'partial_quota_complete')
                break
            state.update(status='running', current_wave=[{k: v for k, v in g.items()} for g in wave])
            publish()
            jobs = []
            for index, g in enumerate(wave):
                if index % size != rank:
                    continue
                gi = next(i for i, x in enumerate(groups) if x['name'] == g['name'])
                geometry = args.seed_base + gi*100000 + g['attempt']
                seed = geometry*9 + g['pair']
                parameters = sample_domain(geometry+43001, 0)
                if g['jitter_m']:
                    parameters.update(stage=1, stage_name='position_only_collection', position_jitter_m=g['jitter_m'])
                jobs.append(dict(seed=seed, parameters=parameters, output=str(local/g['name']/str(seed)),
                    source='frozen_visual_run42_for_act_finetune', actor=actor_state,
                    sampling_seed=geometry, policy_version=saved['policy_version'],
                    collection_stratum=g['name'], split='validation' if geometry % 10 == 0 else 'train'))
            values = []
            if jobs:
                pool.submit('evaluate', jobs)
                rows, timing = pool.collect(timeout=900)
                for job, row in zip(jobs, rows):
                    r = row['result']
                    values.append(r | dict(accepted=admission(r), stratum=job['collection_stratum'],
                        split=job['split'], geometry_group=job['seed']//9,
                        pair=[job['seed']%9//3, job['seed']%3], episode_path=row['episode_id'],
                        teacher_sha256=state['candidate_sha256']))
            shards = [None] * size
            dist.all_gather_object(shards, values)
            records.extend(r for shard in shards for r in shard)
            publish()
        state['current_wave'] = []
        publish()
    except BaseException as error:
        state.update(status='failed', error=repr(error))
        publish()
        raise
    finally:
        if pool is not None:
            pool.close()
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
