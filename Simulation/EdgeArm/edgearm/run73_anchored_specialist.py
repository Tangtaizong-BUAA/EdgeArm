"""Preserve a frozen visual anchor while testing a learned geometry router.

The router sees only the six initially reported joint angles, which the existing
policy already receives. Route IDs supervise the router on collection data but
are never deployment inputs. This is a narrow task-conditioned mixture, not
general VLA, original ACT fine-tuning, or reinforcement learning.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import time

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

import numpy as np
import torch
from torch import nn

from .constrained_recovery_run40 import summarize
from .run34_repeat_eval import deterministic_runtime
from .run63_control_probe import TinyTarget
from .train_staged_hybrid_contact_sac import _atomic_json


class InitialGeometryRouter(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('mean', torch.zeros(6))
        self.register_buffer('scale', torch.ones(6))
        self.net = nn.Sequential(nn.Linear(6, 64), nn.SiLU(), nn.Linear(64, 64),
                                 nn.SiLU(), nn.Linear(64, 9))

    def forward(self, initial_report):
        if initial_report.shape[-1] != 6:
            raise ValueError('only six initially reported joint angles enter the router')
        return self.net((initial_report-self.mean)/self.scale)


class AnchoredSpecialist(nn.Module):
    geometry_gated = True

    def __init__(self, specialist_route=2):
        super().__init__()
        self.anchor = TinyTarget(118, 512)
        self.specialist = TinyTarget(118, 512)
        self.router = InitialGeometryRouter()
        self.specialist_route = int(specialist_route)

    def routing(self, raw_input):
        probabilities = self.router(raw_input[..., 112:118]).softmax(-1)
        chosen = probabilities.argmax(-1)
        return chosen == self.specialist_route, probabilities[..., self.specialist_route]

    def forward(self, corrected_input, raw_input):
        if raw_input.shape[-1] != 118 or raw_input.shape != corrected_input.shape:
            raise ValueError('matching raw and tracked deployable input schemas required')
        use_specialist, _ = self.routing(raw_input)
        return torch.where(use_specialist[..., None], self.specialist(corrected_input), self.anchor(raw_input))


def load_visual_control(path, device='cuda'):
    saved = torch.load(path, map_location='cpu', weights_only=True)
    if saved.get('policy_kind') == 'anchored_geometry_specialist':
        if saved.get('actor_uses_simulator_state') is not False:
            raise ValueError('deployable non-oracle mixture required')
        model = AnchoredSpecialist(saved['specialist_route'])
    elif saved.get('policy_kind') == 'target_increment_v79':
        from .run79_target_increment import TargetIncrementPolicy, JOINT_DELTA
        if saved.get('actor_uses_simulator_state') is not False or saved.get('joint_delta') != JOINT_DELTA:
            raise ValueError('deployable target-increment contract required')
        model = TargetIncrementPolicy()
    else:
        model = TinyTarget(118, 512)
    model.load_state_dict(saved['model'])
    return model.to(device).eval()


def router_examples(sequences):
    groups = {int(s['seed'])//9 for s in sequences}
    if any(97000000 <= group < 99000000 for group in groups):
        raise ValueError('development or independent groups cannot train the router')
    examples = []
    for sequence in sequences:
        x = np.asarray(sequence['x'])
        if x.ndim != 2 or x.shape[1] != 246 or not len(x):
            raise ValueError('complete deployment-feature sequences required')
        if not np.allclose(x[:, 112:118], x[0, 112:118], atol=1e-6, rtol=0):
            raise ValueError('initial reported pose must stay fixed across the episode')
        examples.append((x[0, 112:118].copy(), int(sequence['route']), bool(sequence['validation'])))
    return examples


def router_metrics(model, x, y, specialist_route):
    with torch.inference_mode():
        predictions = model(x).argmax(-1)
    return dict(episodes=len(y), correct=int((predictions == y).sum()),
        specialist_true_positives=int(((predictions == specialist_route) & (y == specialist_route)).sum()),
        specialist_false_positives=int(((predictions == specialist_route) & (y != specialist_route)).sum()),
        specialist_false_negatives=int(((predictions != specialist_route) & (y == specialist_route)).sum()),
        per_route={str(route): dict(episodes=int((y == route).sum()),
            correct=int(((predictions == y) & (y == route)).sum())) for route in range(9)})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('vision', 'tracker', 'anchor', 'specialist', 'encoded', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--specialist-route', type=int, default=2)
    p.add_argument('--updates', type=int, default=1500)
    p.add_argument('--workers', type=int, default=6)
    p.add_argument('--wall-seconds', type=int, default=900)
    p.add_argument('--smoke-only', action='store_true')
    a = p.parse_args()
    if not 0 <= a.specialist_route < 9 or not 100 <= a.updates <= 3000 or not 1 <= a.workers <= 9:
        raise ValueError('bounded router diagnosis required')
    if not 300 <= a.wall_seconds <= 1800:
        raise ValueError('bounded wall time required')
    a.output.mkdir(parents=True, exist_ok=False)
    deterministic_runtime(); torch.set_num_threads(2)
    hashes = {name: hashlib.sha256(getattr(a, name).read_bytes()).hexdigest()
              for name in ('vision', 'tracker', 'anchor', 'specialist')}
    parent = json.loads((a.encoded.parent/'run_state.json').read_text())
    if parent['vision_sha256'] != hashes['vision']:
        raise ValueError('router data must use the frozen visual representation')
    specialist = torch.load(a.specialist, map_location='cpu', weights_only=True)
    tracker = torch.load(a.tracker, map_location='cpu', weights_only=True)
    anchor = torch.load(a.anchor, map_location='cpu', weights_only=True)
    if specialist.get('tracker_sha256') != hashes['tracker'] or tracker['vision_sha256'] != hashes['vision']:
        raise ValueError('specialist, tracker, and vision provenance mismatch')
    if any(s.get('actor_uses_simulator_state') is not False for s in (specialist, anchor, tracker)):
        raise ValueError('oracle weights cannot enter this policy')
    state = dict(run='Run73', status='running', phase='router_training', started=time.time(), step=0,
        total_updates=a.updates, target_rate=.8, evaluations=[], hashes=hashes,
        training_kind='supervised_geometry_router_with_frozen_action_experts_not_RL',
        policy_kind='anchored_geometry_specialist', specialist_route=a.specialist_route,
        router_inputs='six_initial_reported_joint_angles_only', deployment_route_id_input=False,
        start_stage='CONTACT_TRANSPORT_HOLD', actor_uses_simulator_state=False,
        original_ACT_checkpoint=False, independent_acceptance=False, visual_grounding=True,
        production_admission=False, export_admission=False, final_vla_acceptance=False)
    def publish(phase=None):
        if phase: state['phase'] = phase
        state.update(updated=time.time(), elapsed_seconds=time.time()-state['started'])
        _atomic_json(a.output/'run_state.json', state)
    publish()
    try:
        examples = router_examples(torch.load(a.encoded, map_location='cpu', weights_only=False))
        datasets = []
        for validation in (False, True):
            rows = [e for e in examples if e[2] == validation]
            datasets.append((torch.from_numpy(np.stack([e[0] for e in rows])).cuda(),
                             torch.tensor([e[1] for e in rows], device='cuda')))
        (x, y), (vx, vy) = datasets
        model = AnchoredSpecialist(a.specialist_route).cuda()
        model.anchor.load_state_dict(anchor['model']); model.specialist.load_state_dict(specialist['model'])
        model.anchor.requires_grad_(False); model.specialist.requires_grad_(False)
        model.router.mean.copy_(x.mean(0)); model.router.scale.copy_(x.std(0).clamp_min(.02))
        pools = [torch.nonzero(y == route).flatten() for route in range(9)]
        if any(len(pool) == 0 for pool in pools): raise ValueError('router must cover all nine training routes')
        optimizer = torch.optim.AdamW(model.router.parameters(), lr=.001, weight_decay=.0001, fused=True)
        for step in range(a.updates):
            if time.time() > state['started']+a.wall_seconds-180: raise TimeoutError('router budget reached')
            ids = torch.cat([pool[torch.randint(len(pool), (16,), device='cuda')] for pool in pools])
            # Tiny reported-angle augmentation is training-only; deployment and physics do not change.
            loss = nn.functional.cross_entropy(model.router(x[ids]+.002*torch.randn_like(x[ids])), y[ids])
            optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
            state['step'] = step+1
            if (step+1) % 100 == 0:
                state['metrics'] = dict(router_cross_entropy=float(loss.detach()))
                publish()
        state['router_training'] = router_metrics(model.router, x, y, a.specialist_route)
        state['router_validation'] = router_metrics(model.router, vx, vy, a.specialist_route)
        checkpoint = a.output/'policy.pt'
        torch.save(dict(model={k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
            policy_kind='anchored_geometry_specialist', specialist_route=a.specialist_route,
            router_inputs='initial_reported_joints_only', deployment_route_id_input=False,
            component_hashes=hashes, tracker_sha256=hashes['tracker'], vision_sha256=hashes['vision'],
            actor_uses_simulator_state=False, original_ACT_checkpoint=False,
            production_admission=False, export_admission=False, final_vla_acceptance=False), checkpoint)
        state['checkpoint'] = str(checkpoint); publish('router_complete')
        if not a.smoke_only:
            from .run67_visual_state import evaluate_episode
            folder = a.output/'development'; results = []
            state.update(phase_episodes_completed=0, phase_episodes_total=18); publish('development')
            with ProcessPoolExecutor(a.workers, mp_context=mp.get_context('spawn')) as pool:
                jobs = [(g*9+r, str(a.vision), str(checkpoint), str(folder), str(a.tracker))
                        for g in range(97000000, 97000002) for r in range(9)]
                for future in as_completed([pool.submit(evaluate_episode, job) for job in jobs]):
                    results.append(future.result())
                    state.update(phase_episodes_completed=len(results), partial=summarize(results)); publish()
            summary = summarize(results) | dict(label='development', teacher_assisted=False,
                actor_uses_simulator_state=False, independent_acceptance=False,
                pairs={str(r): summarize([s for s in results if s['seed'] % 9 == r]) for r in range(9)})
            _atomic_json(folder/'summary.json', dict(summary=summary, results=results))
            state['evaluations'].append(summary)
        state['status'] = 'smoke_complete' if a.smoke_only else 'complete_pending_review'; publish('finished')
    except BaseException as exc:
        state.update(status='failed', error=repr(exc)); publish(); raise


if __name__ == '__main__': main()
