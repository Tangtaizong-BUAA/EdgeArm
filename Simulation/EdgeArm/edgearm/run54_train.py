"""Route-balanced ACT representation repair with whole-episode visual anchors.

This is supervised ACT fine-tuning, not residual RL. Simulator task coordinates
are optional TRAINING TARGETS only; inference uses the unchanged causal inputs.
After each bounded training block run fully autonomous, equal-route development
episodes. A fresh 72-episode holdout is used once, only after development >60%.
"""
import argparse
from collections import defaultdict
from dataclasses import asdict, replace
import json
import math
import os
from pathlib import Path
import time

os.environ.setdefault('MUJOCO_GL', 'egl')
os.environ.setdefault('PYOPENGL_PLATFORM', 'egl')
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_key, '1')

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Sampler

from .candidate_command_contract_v2 import ACTION_CONTRACT
from .constrained_recovery_run40 import BoundedActor, summarize
from .dual_gpu_sampler_v1 import numpy_state
from .evaluate_multimodal_act_v5 import load_model
from .multimodal_act_v5 import MultimodalACTV5
from .run42.domain import sample_domain
from .run42.sampler import DomainPool
from .run44_posttrain.data import MMapEpisodes, check_disjoint, digest
from .run44_posttrain.objective import configure_trainable
from .run44_posttrain.train import cpu_worker, move
from .run59_sampling import OnPolicyRecoveryBatches, NominalRecoveryBatches, SurveyRecoveryBatches, survey_training_rows
from .temporal_input_contract_v3 import INPUT_KEYS
from .train_multimodal_act_v5 import FORMAT
from .train_staged_hybrid_contact_sac import _atomic_json


class RouteFirstBatches(Sampler):
    """10% human; 90% equal routes, THEN 60% recovery / 40% old per route."""
    def __init__(self, records, batch_size, batches, seed):
        self.records, self.batch_size, self.batches, self.seed = records, batch_size, batches, seed
        self.pools = defaultdict(list)
        self.times = []
        for i, r in enumerate(records):
            if r['split'] != 'train' or not r['valid_times']:
                raise ValueError('train trajectories with valid labels required')
            if r['run54_pool'] == 'human':
                key = ('human', 'human')
            else:
                pair = r.get('pair')
                if isinstance(pair, list):
                    pair = ','.join(map(str, pair))
                if pair not in {f'{b},{g}' for b in range(3) for g in range(3)}:
                    raise ValueError('unknown geometric route: '+str(pair))
                key = (pair, r['run54_pool'])
            self.pools[key].append(i)
            self.times.append([a for a in np.array_split(np.asarray(r['valid_times'], np.int64), 3) if len(a)])
        for b in range(3):
            for g in range(3):
                for source in ('old', 'recovery'):
                    if not self.pools[(f'{b},{g}', source)]:
                        raise ValueError('missing route/source rehearsal')
        if not self.pools[('human','human')]:
            raise ValueError('preserve human rehearsal')

    def __len__(self):
        return self.batches

    def __iter__(self):
        rng = np.random.default_rng(self.seed)
        cycle, cursor, decks = [], 0, {}
        for _ in range(self.batches):
            batch = []
            for _ in range(self.batch_size):
                if cursor == len(cycle):
                    # Exactly 10 examples of each route and 10 human per 100.
                    cycle, cursor = rng.permutation(np.repeat(np.arange(10), 10)).tolist(), 0
                route, cursor = cycle[cursor], cursor+1
                if route == 9:
                    key = ('human', 'human')
                else:
                    key = (f'{route//3},{route%3}', 'recovery' if rng.random() < .6 else 'old')
                if not decks.get(key):
                    decks[key] = rng.permutation(self.pools[key]).tolist()
                ri = decks[key].pop()
                phase = self.times[ri][int(rng.integers(len(self.times[ri])))]
                batch.append((ri, int(rng.choice(phase))))
            yield batch


class GroundedEpisodes(MMapEpisodes):
    def __getitem__(self, index):
        sample = super().__getitem__(index)
        ri, t = index
        row = self.records[ri]
        if getattr(self.cfg, 'action_head_space', 'relative_command') in (
                'applied_target_delta_v56', 'absolute_joint_target_v57'):
            z = self.cache[ri]
            count = min(self.cfg.action_chunk_size, len(z['joint'])-t)
            sample['absolute_target'] = np.zeros_like(sample['target'])
            sample['absolute_target'][:count] = z['joint'][t:t+count, :6] + .055*z['command'][t:t+count]
        sample['task_target'] = np.zeros(4, np.float32)
        sample['task_mask'] = np.bool_(False)
        sample['route'] = np.int64(-1 if row['run54_pool'] == 'human' else
            3*int(row['pair'][0])+int(row['pair'][-1]))
        sample['recovery'] = np.bool_(row['run54_pool'] == 'recovery')
        sample['run59_fresh'] = np.bool_(row.get('run59_fresh', False))
        if row['run54_pool'] == 'recovery':
            cache = self.cache[ri]
            if 'task_target' not in cache:
                cache['task_target'] = np.load(Path(row['store'])/'task_state_supervision.npy', mmap_mode='r')
                cache['task_mask'] = np.load(Path(row['store'])/'task_state_supervision_mask.npy', mmap_mode='r')
            sample['task_target'] = cache['task_target'][t].copy()
            sample['task_mask'] = cache['task_mask'][t].copy()
        return sample


def diagnostic_nine(records):
    """One complete training demonstration per route; NEVER an acceptance set."""
    selected = []
    for route in range(9):
        pair = f'{route//3},{route%3}'
        matches = [r for r in records if r['split'] == 'train' and r['run54_pool'] == 'recovery'
                   and r['pair'] == pair and r.get('teacher_start_step') == 0
                   and r['valid_times'][0] == 0]
        if not matches:
            raise ValueError('missing full successful diagnostic demonstration: '+pair)
        selected.append(matches[0])
    return selected


def verify_nominal_scene(scene):
    """Check the actual domain parameters, not just a nominal stage label."""
    parameters = scene['parameters']
    if (parameters != sample_domain(parameters['seed'], 0)
            or scene.get('workspace_adapter') not in (None, 'legacy_v10')):
        raise ValueError('Run60 ablation requires the unchanged nominal plant')


class DiagnosticBatches(Sampler):
    def __init__(self, records, batch_size, batches, seed):
        self.records, self.batch_size, self.batches, self.seed = records, batch_size, batches, seed

    def __len__(self):
        return self.batches

    def __iter__(self):
        rng = np.random.default_rng(self.seed)
        for _ in range(self.batches):
            batch = []
            for _ in range(self.batch_size):
                ri = int(rng.integers(len(self.records)))
                valid = self.records[ri]['valid_times']
                # Include reset/phase boundaries; no label outside the demonstration.
                t = 0 if rng.random() < .1 else int(rng.choice(valid))
                batch.append((ri, t))
            yield batch


class GroundedObjective(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.task_head = nn.Sequential(nn.LayerNorm(model.config.model_dim),
            nn.Linear(model.config.model_dim, 128), nn.SiLU(), nn.Linear(128, 4))

    def forward(self, batch, step):
        m = self.model
        memory, padding, _ = m.policy.encode_observations(batch['inputs'])
        prior, decoded = m.policy.decode_observations(memory, padding)
        posterior_target = (batch['absolute_target']/torch.pi
                            if m.config.action_head_space == 'absolute_joint_target_v57'
                            else batch['target'])
        mu, logvar, kl = m.posterior(batch['inputs']['robot_state'], posterior_target, batch['mask'])
        latent = mu + (logvar*.5).exp()*torch.randn_like(mu)
        posterior, _ = m.policy.decode_observations(memory, padding, m.latent_projection(latent))
        mask = batch['mask']
        target = batch['target']
        def action_loss(predicted):
            if m.config.action_head_space in ('applied_target_delta_v56', 'absolute_joint_target_v57'):
                reference = batch['inputs']['robot_state'][:, None, :6]
                prediction = reference + .055*predicted.float()
                truth, scale = batch['absolute_target'], .005
            else:
                prediction, truth, scale = predicted.float(), target.float(), .05
            error = F.smooth_l1_loss(prediction/scale, truth/scale,
                                     reduction='none').mean(-1)
            chunk = (error*mask).sum(-1)/mask.sum(-1).clamp_min(1)
            return (.3*chunk+error[:, 0]).mean()
        prior_loss, posterior_loss = action_loss(prior), action_loss(posterior)
        predicted_state = self.task_head(decoded[:, 0])
        task_error = F.smooth_l1_loss(predicted_state.float(), batch['task_target']/.2,
                                     reduction='none').mean(-1)
        task_mask = batch['task_mask']
        task_loss = (task_error*task_mask).sum()/task_mask.sum().clamp_min(1)
        current = batch['inputs']['kinematic_history'][:, -1]
        future = m.future_head(decoded, current[:, :3], current[:, 12:18])
        fm = batch['future_tool_mask']
        error = F.smooth_l1_loss(future.float()/.1, batch['future_tool_xyz']/.1,
                               reduction='none').mean(-1)
        future_loss = (error*fm).sum()/fm.sum().clamp_min(1)
        loss = prior_loss + .25*posterior_loss + .2*task_loss + .05*future_loss
        loss = loss + .0005*min((step+1)/100, 1)*kl.float()
        first_mae = (prior[:, 0].float()-target[:, 0]).abs().mean()
        return loss, torch.stack((loss.detach(), prior_loss.detach(), posterior_loss.detach(),
            task_loss.detach(), future_loss.detach(), first_mae.detach()))


def ranking(summary):
    # Research checkpoint selection; never deployment admission by itself.
    return (summary['successes'], -summary['hard_failures'], summary['mean_coverage'])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--initial-checkpoint', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--rounds', type=int, default=3)
    p.add_argument('--updates-per-round', type=int, default=400)
    p.add_argument('--batch', type=int, default=64)
    p.add_argument('--workers', type=int, default=10)
    p.add_argument('--eval-workers', type=int, default=18)
    p.add_argument('--wall-seconds', type=int, default=7200)
    p.add_argument('--learning-rate', type=float, default=5e-5)
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--diagnostic-nine', action='store_true',
                   help='Overfit nine TRAINING episodes; same-seed tests are not generalization')
    p.add_argument('--target-reference-head', action='store_true',
                   help='Run56: predict actuator target increments with analytic joint feedback')
    p.add_argument('--absolute-target-head', action='store_true',
                   help='Run57: absolute joint targets, consistent posterior, FP32 control head')
    p.add_argument('--on-policy-repair', action='store_true',
                   help='Run59: fresh current-ACT recovery suffixes plus route-balanced rehearsal')
    p.add_argument('--nominal-recovery-only', action='store_true',
                   help='Run60 data-only ablation: verified nominal recoveries, preserve all legacy data on disk')
    p.add_argument('--active-view-repair', action='store_true',
                   help='Run62: train matching survey history with 50% nominal recovery rehearsal')
    p.add_argument('--unseen-after-probe', action='store_true',
                   help='If nine-demo diagnostic passes, run unseen development then a fresh holdout')
    p.add_argument('--wait-for-run-state', type=Path,
                   help='Wait at most 20 minutes for an existing experiment to release the GPU')
    a = p.parse_args()
    if a.active_view_repair:
        a.nominal_recovery_only = True
    if a.nominal_recovery_only:
        a.on_policy_repair = True
    if a.target_reference_head and a.absolute_target_head:
        raise ValueError('choose exactly one action-head experiment')
    if a.on_policy_repair and (a.diagnostic_nine or a.absolute_target_head):
        raise ValueError('Run59 uses all training pools and preserves the Run56 target head')
    run_name = 'Run62' if a.active_view_repair else 'Run60' if a.nominal_recovery_only else 'Run59' if a.on_policy_repair else 'Run57' if a.absolute_target_head else ('Run56' if a.target_reference_head else (
        'Run55' if a.diagnostic_nine else 'Run54'))
    if not (1 <= a.rounds <= 6 and 1 <= a.updates_per_round <= 1000 and 8 <= a.batch <= 128
            and 0 <= a.workers <= 12 and 1 <= a.eval_workers <= 20 and 600 <= a.wall_seconds <= 10800):
        raise ValueError('bounded single GPU experiment required')
    a.output.mkdir(parents=True, exist_ok=False)
    if a.wait_for_run_state:
        deadline = time.monotonic()+1200
        while True:
            previous = json.loads(a.wait_for_run_state.read_text())
            if previous['status'] == 'failed':
                _atomic_json(a.output/'run_state.json', dict(run=run_name, status='failed',
                    phase='previous_experiment_failed', updated=time.time(),
                    production_admission=False, export_admission=False, final_vla_acceptance=False))
                raise RuntimeError('do not train after a failed prerequisite')
            if previous['status'] in ('complete_pending_review', 'failed', 'smoke_complete'):
                break
            _atomic_json(a.output/'run_state.json', dict(run=run_name,
                status='queued', phase='waiting_for_previous_experiment', updated=time.time(),
                production_admission=False, export_admission=False, final_vla_acceptance=False))
            if time.monotonic() > deadline:
                raise TimeoutError('previous experiment did not release its GPU within 20 minutes')
            time.sleep(5)
    torch.set_num_threads(2)
    torch.manual_seed(5401)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.benchmark = True
    manifest = json.loads(a.manifest.read_text())
    if manifest['schema'] != 'run54-recovery-act-v1' or manifest['action_contract'] != ACTION_CONTRACT:
        raise ValueError('Run54 data/action contract mismatch')
    check_disjoint(manifest['records'])
    rows = manifest['records']
    occupied = {r['seed']//9 for r in rows if isinstance(r.get('seed'), int)}
    if any(97000000 <= g < 99000000 for g in occupied):
        raise ValueError('Run54 fresh evaluation overlaps existing training data')
    train = [r for r in rows if r['split'] == 'train']
    preserved_train_count = len(train)
    if a.nominal_recovery_only:
        train = [r for r in train if r['run54_pool'] == 'recovery']
        for row in train:
            scene = json.loads((Path(row['parent_source_path'])/'scenario.json').read_text())
            verify_nominal_scene(scene)
            if a.active_view_repair and row.get('run62_fresh') and scene.get('active_wrist_survey') is not True:
                raise ValueError('new recovery must have real active wrist survey provenance')
    if a.active_view_repair:
        train = survey_training_rows(train)
    if a.diagnostic_nine:
        train = diagnostic_nine(train)
    initial, _ = load_model(a.initial_checkpoint)
    if a.on_policy_repair:
        fresh = [r for r in train if r.get('run59_fresh')]
        if len(fresh) < 18 or initial.config.action_head_space != 'applied_target_delta_v56':
            raise ValueError('fresh corrections of the Run56 ACT are required')
        expected_digest = digest(a.initial_checkpoint)
        if any(r.get('rollin_checkpoint_sha256') != expected_digest for r in fresh):
            raise ValueError('fresh roll-ins must come from this exact initial ACT')
    head_space = ('absolute_joint_target_v57' if a.absolute_target_head else
                  'applied_target_delta_v56' if a.target_reference_head or a.on_policy_repair else 'relative_command')
    cfg = replace(initial.config, visual_memory_mode='episode_anchors_v54', action_head_space=head_space)
    model = MultimodalACTV5(cfg)
    model.load_state_dict(initial.state_dict())
    if a.absolute_target_head and initial.config.action_head_space != head_space:
        # The old head predicts increments; do not reinterpret those weights as
        # radians. Initialize ONLY this changed output map from training joints.
        sums, count = np.zeros(6, np.float64), 0
        for row in train:
            q = np.load(Path(row['store'])/'joint.npy', mmap_mode='r')
            cmd = np.load(Path(row['store'])/'command.npy', mmap_mode='r')
            ids = row['valid_times']
            targets = q[ids, :6]+.055*cmd[ids]
            if np.any(np.abs(targets) >= np.pi):
                raise ValueError('absolute target exceeds versioned pi normalization')
            sums += targets.sum(axis=0, dtype=np.float64)
            count += len(ids)
        nn.init.normal_(model.policy.core.action_head.weight, std=.0001)
        with torch.no_grad():
            model.policy.core.action_head.bias.copy_(torch.from_numpy(np.arctanh(sums/count/np.pi)).float())
    del initial
    configure_trainable(model, 'all')
    model.cuda()
    objective = GroundedObjective(model).cuda()
    groups = []
    for name, param in objective.named_parameters():
        if param.requires_grad:
            scale = .3 if '.rgbd_encoder.' in name else 1.
            groups.append(dict(params=[param], lr=a.learning_rate*scale,
                initial_lr=a.learning_rate*scale, weight_decay=.0001 if param.ndim >= 2 else 0.))
    optimizer = torch.optim.AdamW(groups, fused=True)
    state = dict(run=run_name, status='running', phase='initializing', started=time.time(),
        total_updates=a.rounds*a.updates_per_round, step=0, round=0, rounds=a.rounds,
        batch=a.batch, learning_rate=a.learning_rate, loader_workers=a.workers,
        training_episodes=len(train), recovery_episodes=manifest['recovery_episodes'],
        visual_memory_mode=cfg.visual_memory_mode, visual_and_policy_trainable=True,
        action_head_space=cfg.action_head_space,
        posterior_target_space='absolute_radians_div_pi' if a.absolute_target_head else 'relative_command',
        control_head_fp32=a.absolute_target_head,
        depth_frozen=True, teacher_coordinates_are_labels_only=True, actor_uses_simulator_state=False,
        cohort_mix=dict(human=.1, old=.36, recovery=.54),
        route_equal_sampling=True, start_stage='CONTACT_TRANSPORT_HOLD',
        exact_home_evaluated=False, production_admission=False, export_admission=False,
        final_vla_acceptance=False, target_rate=.60, evaluations=[],
        initial_sha256=digest(a.initial_checkpoint), manifest_sha256=digest(a.manifest),
        parameters=sum(x.numel() for x in model.parameters()),
        trainable_parameters=sum(x.numel() for x in model.parameters() if x.requires_grad))
    if a.diagnostic_nine:
        state.update(diagnostic_only=True, cohort_mix=dict(recovery=1.),
                     recovery_episodes=9, training_seed_replay_is_not_acceptance=True,
                     diagnostic_seeds=[r['seed'] for r in train])
    if a.on_policy_repair:
        preview = (SurveyRecoveryBatches if a.active_view_repair else
                   NominalRecoveryBatches if a.nominal_recovery_only else OnPolicyRecoveryBatches)(train, a.batch, 1, 5901)
        state.update(training_kind='supervised_current_policy_recovery_not_RL',
            fresh_recovery_episodes=len(fresh),
            cohort_mix=dict(human=.1, old=.18, previous_recovery=.225, current_policy_recovery=.495),
            cohort_mix_is_requested=True, fresh_missing_routes=preview.missing_fresh_routes,
            missing_fresh_fallback='previous_successful_recovery_same_route',
            takeover_boundary_oversampling=True, initial_checkpoint_matches_rollin=True)
        if a.nominal_recovery_only:
            state.update(training_kind='nominal_data_mixture_ablation_not_RL',
                cohort_mix=dict(previous_recovery=.3125, current_policy_recovery=.6875),
                legacy_trajectories_preserved_but_not_sampled=preserved_train_count-len(train),
                action_labels_unchanged=True, nominal_plant_verified=True)
        if a.active_view_repair:
            state.update(training_kind='supervised_active_view_history_repair_not_RL',
                cohort_mix=dict(previous_recovery=.5, current_survey_recovery=.5),
                scripted_observation_prefix=True, survey_prefix_steps=220,
                fixed_acquisition_is_not_learned=True,
                original_no_survey_capability_not_revalidated=True)
    names = ('loss','prior_loss','posterior_loss','task_state_loss','future_tool_loss','first_command_mae')
    def publish(phase=None):
        if phase:
            state['phase'] = phase
        state.update(updated=time.time(), elapsed_seconds=time.time()-state['started'])
        _atomic_json(a.output/'run_state.json', state)
    def save(name):
        target = a.output/name
        value = dict(format=FORMAT, config=asdict(cfg), model=model.state_dict(),
            action_contract=ACTION_CONTRACT, input_keys=sorted(INPUT_KEYS), step=state['step'],
            initial_checkpoint_sha256=state['initial_sha256'],
            posttrain_schema='run54-representation-repair-v1', production_admission=False,
            diagnostic_task_head=objective.task_head.state_dict(),
            diagnostic_only=a.diagnostic_nine, export_admission=False, final_vla_acceptance=False)
        if a.active_view_repair:
            value['required_observation_profile'] = 'run61_survey_return_220'
        torch.save(value, target.with_suffix('.tmp'))
        target.with_suffix('.tmp').replace(target)
        return target
    def evaluate(checkpoint, label, heldout=False, unseen=False):
        publish(label)
        seed_groups = range(98000000, 98000008) if heldout else range(97000000, 97000002)
        seeds = [g*9+r for g in seed_groups for r in range(9)]
        training_replay = a.diagnostic_nine and not heldout and not unseen
        if training_replay:
            seeds = [r['seed'] for r in train]
        actor = BoundedActor(2*cfg.model_dim+37).eval()  # Exactly zero residual, fresh ACT prior.
        folder = a.output/label
        folder.mkdir()
        pool = DomainPool(str(checkpoint), None, folder, a.eval_workers, 'cuda:0', 0,
                          max_batch=a.eval_workers, wait_ms=3, seed=5401)
        results = []
        state.update(phase_episodes_completed=0, phase_episodes_total=len(seeds))
        publish()
        try:
            for i in range(0, len(seeds), a.eval_workers):
                jobs = [dict(seed=s, parameters=sample_domain(s+6001, 0), source=label,
                    actor=numpy_state(actor), sampling_seed=s+5401, policy_version=state['step'],
                    max_steps=900, noise_std=0., compress_wrist=True,
                    active_wrist_survey=a.active_view_repair,
                    output=str(folder/f'episode_{s}')) for s in seeds[i:i+a.eval_workers]]
                pool.submit('episode', jobs)
                values, _ = pool.collect(timeout=900)
                batch = [v['result'] for v in values]
                for r in batch:
                    if r['teacher_assisted'] or r['initial_coverage'] > 0:
                        raise ValueError('autonomous evaluation contamination')
                    if a.active_view_repair:
                        if not r.get('scripted_observation_prefix') or r['training_rollin_steps'] > 220:
                            raise ValueError('unrecognized observation prefix')
                        if r['safe_success'] and r['training_rollin_steps'] != 220:
                            raise ValueError('success without completed acquisition protocol')
                    elif r['training_rollin_steps']:
                        raise ValueError('unexpected autonomous prefix')
                results.extend(batch)
                state.update(phase_episodes_completed=len(results), partial=summarize(results))
                publish()
        finally:
            pool.close()
        summary = summarize(results) | dict(label=label, step=state['step'],
            pairs={str(r):summarize([x for x in results if x['seed']%9 == r]) for r in range(9)})
        if a.active_view_repair:
            summary.update(scripted_observation_prefix=True, learned_acquisition=False,
                           total_step_budget=900, survey_step_budget=220)
        if training_replay:
            summary.update(training_seed_replay=True, generalization_acceptance=False)
        _atomic_json(folder/'summary.json', dict(summary=summary, results=results))
        state['evaluations'].append(summary)
        publish()
        return summary

    publish()
    best = None
    try:
        for round_ in range(a.rounds):
            if time.time()-state['started'] > a.wall_seconds-240:
                state['budget_stopped'] = True
                break
            state['round'] = round_+1
            sampler_type = (SurveyRecoveryBatches if a.active_view_repair else
                            NominalRecoveryBatches if a.nominal_recovery_only else
                            OnPolicyRecoveryBatches if a.on_policy_repair else
                            DiagnosticBatches if a.diagnostic_nine else RouteFirstBatches)
            sampler = sampler_type(train, a.batch, a.updates_per_round, 5401+round_*101)
            kwargs = dict(batch_sampler=sampler, num_workers=a.workers, pin_memory=True, worker_init_fn=cpu_worker)
            if a.workers:
                kwargs.update(persistent_workers=False, prefetch_factor=1, multiprocessing_context='spawn')
            loader = DataLoader(GroundedEpisodes(train, cfg, cache_size=16), **kwargs)
            objective.train()
            model.policy.depth_student.eval()
            values = torch.zeros(len(names), device='cuda')
            tick, logged = time.time(), state['step']
            counts = np.zeros(10, np.int64)
            source_counts = np.zeros(4, np.int64)
            publish('act_representation_repair')
            for host in loader:
                counts += np.bincount(np.where(host['route'].numpy() < 0, 9, host['route'].numpy()), minlength=10)
                source = np.where(host['route'].numpy() < 0, 0,
                    np.where(host['run59_fresh'].numpy(), 3, np.where(host['recovery'].numpy(), 2, 1)))
                source_counts += np.bincount(source, minlength=4)
                batch = move(host, 'cuda:0')
                ratio = state['step']/max(state['total_updates'],1)
                scale = min((state['step']+1)/50,1) * (.25+.75*.5*(1+math.cos(math.pi*ratio)))
                for group in optimizer.param_groups:
                    group['lr'] = group['initial_lr']*scale
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    loss, metrics = objective(batch, state['step'])
                if not torch.isfinite(loss):
                    raise FloatingPointError('nonfinite ACT objective')
                loss.backward()
                torch.nn.utils.clip_grad_norm_(objective.parameters(), 1., error_if_nonfinite=True)
                optimizer.step()
                values += metrics
                state['step'] += 1
                if state['step']%10 == 0 or a.smoke:
                    n = state['step']-logged
                    state.update(metrics=dict(zip(names, (values/n).cpu().tolist())),
                        samples_per_second=n*a.batch/max(time.time()-tick,1e-6),
                        sampled_routes_plus_human=counts.tolist(),
                        sampled_sources_human_old_previous_current=source_counts.tolist(),
                        peak_gpu_GiB=torch.cuda.max_memory_allocated()/2**30)
                    publish()
                    with (a.output/'training_metrics.jsonl').open('a') as stream:
                        stream.write(json.dumps(state)+'\n')
                    values.zero_(); tick, logged = time.time(), state['step']
                if time.time()-state['started'] > a.wall_seconds-240:
                    state['budget_stopped'] = True
                    break
            del loader
            optimizer.zero_grad(set_to_none=True)
            del batch, loss, metrics, host
            torch.cuda.empty_cache()
            model.eval()
            checkpoint = save(f'checkpoint_round_{round_+1}.pt')
            if a.smoke:
                state['status'] = 'smoke_complete'
                publish('finished')
                return
            summary = evaluate(checkpoint, f'development_r{round_+1}')
            if best is None or ranking(summary) > ranking(best):
                best = summary
                state['selected'] = summary
                # Saved candidate, not promoted to deployment.
                import shutil
                shutil.copy2(checkpoint, a.output/'checkpoint_selected.pt')
            if summary['successes']/summary['episodes'] > .60 and summary['hard_failures'] == 0:
                break
        qualifies = bool(best and best['successes']/best['episodes'] > .60 and best['hard_failures'] == 0)
        if a.diagnostic_nine and qualifies and a.unseen_after_probe:
            development = evaluate(a.output/'checkpoint_selected.pt', 'unseen_development', unseen=True)
            qualifies = development['successes']/development['episodes'] > .60 and development['hard_failures'] == 0
            state['unseen_development'] = development
        elif a.diagnostic_nine:
            qualifies = False
        if qualifies:
            heldout = evaluate(a.output/'checkpoint_selected.pt', 'fresh_heldout', heldout=True)
            state['target_60_observed'] = heldout['successes']/heldout['episodes'] > .60
            state['safe_60_gate_passed'] = state['target_60_observed'] and heldout['hard_failures'] == 0
        else:
            state['fresh_heldout_not_spent'] = True
            state['target_60_observed'] = False
        state['status'] = 'complete_pending_review'
        publish('finished')
    except BaseException as exc:
        state.update(status='failed', error=repr(exc))
        publish()
        raise


if __name__ == '__main__':
    main()
