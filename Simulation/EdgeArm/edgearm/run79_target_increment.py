"""Learn increments of the completed servo target, with phase-balanced BC.

Reported-position feedback is explicit: command = target_increment +
(previous_applied_target - reported_q)/joint_delta. A zero increment holds a
fixed absolute target instead of following sensor noise. No truth in actor.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from .constrained_recovery_run40 import summarize
from .run34_repeat_eval import deterministic_runtime
from .run63_control_probe import TinyTarget
from .run67_visual_state import VisualState, visual_batch
from .run68_visual_control import estimated_inputs
from .run75_observation_training import SurveyFrames
from .run76_color_dagger import ColorFrames
from .train_staged_hybrid_contact_sac import _atomic_json


JOINT_DELTA = .055


def increment_labels(x, commands):
    if x.shape[-1] != 118 or commands.shape != (*x.shape[:-1], 6):
        raise ValueError('matching deployable observation / command schema required')
    return commands+(x[..., :6]-x[..., 102:108])/JOINT_DELTA


def training_phase(batch):
    """Truth used to balance training, never routed to inference."""
    selected = batch['xy'].gather(1, batch['selected'][:, :, None].expand(-1, -1, 2))
    distance = torch.linalg.vector_norm(selected[:, 0]-selected[:, 1], dim=1)
    phase = torch.where(distance > .08, 0, 1)
    near = distance <= .028
    phase = torch.where(near, torch.where(batch['proprio'][:, 14] < .115, 2, 3), phase)
    return phase.long()


class TargetIncrementPolicy(TinyTarget):
    def __init__(self):
        super().__init__(118, 512)

    def predict_increment(self, x):
        return super().forward(x)

    def forward(self, x):
        return self.predict_increment(x)+(x[..., 102:108]-x[..., :6])/JOINT_DELTA


def recode(visual, dataset, output, publish):
    loader = DataLoader(dataset, batch_size=96, shuffle=False, num_workers=6, pin_memory=True)
    pieces = [[] for _ in range(4)]
    with torch.inference_mode():
        for i, batch in enumerate(loader):
            b = visual_batch(batch)
            world = visual(b['rgb'], b['pose'], b['K'], b['age'])
            x = estimated_inputs(b['proprio'], world, b['selected'])
            for dest, value in zip(pieces, (x, b['command'], b['route'], training_phase(b))):
                dest.append(value.cpu().numpy())
            if i % 30 == 0: publish('frozen_visual_recode')
    data = tuple(np.concatenate(p) for p in pieces)
    np.savez_compressed(output, x=data[0], command=data[1], route=data[2], phase=data[3])
    return data


def phase_pools(routes, phases):
    pools=[]
    for route in range(9):
        available=[torch.where((routes == route) & (phases == phase))[0] for phase in range(4)]
        available=[pool for pool in available if len(pool)]
        if not available: raise ValueError('all nine training routes required')
        pools.append(available)
    return pools


def balanced_ids(routes, phases, count_per_route=112, pools=None):
    """Equal routes, then equal available phases within each route."""
    ids = []
    for available in phase_pools(routes, phases) if pools is None else pools:
        for index, pool in enumerate(available):
            n = count_per_route//len(available)+int(index < count_per_route % len(available))
            ids.append(pool[torch.randint(len(pool), (n,), device=pool.device)])
    return torch.cat(ids)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('vision', 'control', 'source-state', 'old-collection', 'output'):
        p.add_argument('--'+key, type=Path, required=True)
    p.add_argument('--updates-per-round', type=int, default=2000)
    p.add_argument('--rounds', type=int, default=3)
    p.add_argument('--workers', type=int, default=9)
    p.add_argument('--wall-seconds', type=int, default=1800)
    p.add_argument('--smoke-only', action='store_true')
    a = p.parse_args()
    if not 1 <= a.workers <= 9 or not 1 <= a.rounds <= 3 or not 10 <= a.updates_per_round <= 3000:
        raise ValueError('bounded control adaptation')
    if not 300 <= a.wall_seconds <= 3600: raise ValueError('finite experiment deadline')
    source = json.loads(a.source_state.read_text())
    roots = [Path(source['reused_first_collection'])]
    roots += [x for x in sorted(a.source_state.parent.glob('collection_*')) if (x/'summary.json').is_file()]
    a.output.mkdir(parents=True, exist_ok=False)
    deterministic_runtime(); torch.set_num_threads(2)
    state = dict(run='Run79', status='running', started=time.time(), step=0, round=0, evaluations=[],
        total_updates=a.updates_per_round*a.rounds, rounds=a.rounds,
        training_kind='phase_balanced_absolute_target_increment_supervision_not_RL',
        action_space='increment_of_previous_applied_servo_target', joint_delta=JOINT_DELTA,
        actor_uses_simulator_state=False, phase_uses_truth_only_for_training_sampling=True,
        frozen_perception=True, original_ACT_checkpoint=False, scripted_completion='visual_latch_then_lift_and_hold',
        retained_old_source_fraction=.5, route_equal_sampling=True, renewed_compute_authorization=True,
        old_six_hour_budget_reused=False, max_steps=900, fixed_survey_steps=220,
        independent_acceptance=False, production_admission=False, export_admission=False, final_vla_acceptance=False,
        vision=str(a.vision), vision_sha256=hashlib.sha256(a.vision.read_bytes()).hexdigest())
    deadline = state['started']+a.wall_seconds
    def publish(phase=None):
        if phase: state['phase'] = phase
        state.update(updated=time.time(), elapsed_seconds=time.time()-state['started'])
        _atomic_json(a.output/'run_state.json', state)
    try:
        publish('initializing')
        saved = torch.load(a.vision, map_location='cpu', weights_only=True)
        visual = VisualState(recent_block_seconds=saved.get('recent_block_seconds')).cuda().eval()
        visual.load_state_dict(saved['model'])
        old = recode(visual, SurveyFrames(a.old_collection, action_only=True), a.output/'old_recoded.npz', publish)
        new = recode(visual, ColorFrames(roots, action_only=True), a.output/'new_recoded.npz', publish)
        del visual; torch.cuda.empty_cache()
        state['source_counts'] = [dict(rows=len(data[0]), route_phase_counts=[
            [int(np.sum((data[2]==r)&(data[3]==phase))) for phase in range(4)] for r in range(9)]) for data in (old, new)]
        sources = [tuple(torch.from_numpy(x).cuda() for x in data) for data in (old, new)]
        cached_pools = [phase_pools(route, phase) for _, _, route, phase in sources]
        model = TargetIncrementPolicy().cuda()
        model.load_state_dict(torch.load(a.control, map_location='cpu', weights_only=True)['model'])
        # Preserve learned feature weights and x normalization. Only the output
        # normalization changes to match the new physical action representation.
        with torch.no_grad():
            labels = torch.cat([increment_labels(x, y) for x, y, _, _ in sources])
            model.ymean.copy_(labels.mean(0)); model.yscale.copy_(labels.std(0).clamp_min(.025))
            del labels
        optimizer = torch.optim.AdamW(model.parameters(), lr=.0001, weight_decay=1e-6, fused=True)
        best_rank = None
        for round_index in range(a.rounds):
            state['round'] = round_index+1
            model.train(); publish('phase_balanced_target_increment_fit')
            for _ in range(a.updates_per_round):
                if time.time() > deadline-180: raise TimeoutError('reserve closed-loop evaluation time')
                loss = 0.
                for (x, y, route, phase), pools in zip(sources, cached_pools):
                    ids = balanced_ids(route, phase, pools=pools)
                    batch = x[ids].clone()
                    target = increment_labels(batch, y[ids])
                    # Finite reported-position jitter, unchanged absolute target
                    # increment. This teaches feedback invariance, not new states.
                    batch[:, :6] += torch.randn_like(batch[:, :6])*.0001
                    loss += .5*((model.predict_increment(batch)-target)/model.yscale).square().mean()
                optimizer.zero_grad(set_to_none=True); loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 5, error_if_nonfinite=True); optimizer.step()
                state['step'] += 1
                if state['step'] % 100 == 0:
                    with torch.inference_mode():
                        audit = []
                        for x, y, route, phase in sources:
                            error = (model(x)-y).abs().mean(-1)
                            audit.append([float(error[phase==k].mean()) if torch.any(phase==k) else None for k in range(4)])
                    state['metrics'] = dict(loss=float(loss.detach()), command_mae_by_source_phase=audit)
                    publish()
            path = a.output/f'round_{round_index+1}.pt'
            torch.save(dict(model={k:v.detach().cpu().clone() for k,v in model.state_dict().items()},
                policy_kind='target_increment_v79', vision_sha256=state['vision_sha256'],
                joint_delta=JOINT_DELTA, actor_uses_simulator_state=False, original_ACT_checkpoint=False,
                production_admission=False, export_admission=False, final_vla_acceptance=False), path)
            if a.smoke_only:
                state.update(status='smoke_complete', checkpoint=str(path)); publish('finished'); return
            model.eval(); results=[]
            state.update(phase_episodes_completed=0, phase_episodes_total=36); publish('autonomous_development')
            from .run78_completion_probe import episode
            with ProcessPoolExecutor(a.workers, mp_context=mp.get_context('spawn')) as pool:
                jobs = [(g*9+r, str(a.vision), str(path), str(a.output/f'development_{round_index+1}'), 'lift')
                    for g in range(97100000,97100004) for r in range(9)]
                for future in as_completed([pool.submit(episode, j) for j in jobs]):
                    results.append(future.result()); state.update(phase_episodes_completed=len(results), partial=summarize(results)); publish()
            summary = summarize(results)|dict(round=round_index+1,
                block_out_of_bounds=sum(r['terminal_reason']=='block_out_of_bounds' for r in results),
                pairs={str(r):summarize([x for x in results if x['seed']%9==r]) for r in range(9)})
            _atomic_json(a.output/f'development_{round_index+1}'/'summary.json', dict(summary=summary, results=results))
            state['evaluations'].append(summary)
            rank=(summary['successes'],-summary['hard_failures'],-summary['block_out_of_bounds'],summary['mean_coverage'])
            if best_rank is None or rank > best_rank:
                best_rank=rank; state.update(best_control=str(path), best_development_rank=list(rank))
            publish()
            if summary['successes'] >= 30: break
        state['status']='complete_pending_review'; publish('finished')
    except BaseException as exc:
        state.update(status='failed', error=repr(exc)); publish(); raise


if __name__ == '__main__': main()
