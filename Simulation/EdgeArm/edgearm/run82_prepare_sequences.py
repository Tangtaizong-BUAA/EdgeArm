"""Reconstruct sparse 3D/visibility LABELS from authorized training replays.

Actor inputs and privileged labels are written to separate files. No training
or evaluation simulator truth is added to deployment inputs. Color variants
remain image augmentation of one physical episode, not extra interactions.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import multiprocessing as mp
import os
from pathlib import Path
import time

for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(key, '1')
os.environ.setdefault('MUJOCO_GL', 'egl')
os.environ.setdefault('PYOPENGL_PLATFORM', 'egl')

import mujoco
import numpy as np
from PIL import Image

from .candidate_command_contract_v2 import ACTION_CONTRACT
from .run42.domain import sample_domain
from .run42.session import DomainSession
from .run67_visual_state import COLORS
from .sparse_4d_vla_act_v26 import Sparse4DVLAConfigV26
from .train_staged_hybrid_contact_sac import _atomic_json


def training_split(seed):
    group = seed//9
    if not 100200000 <= group < 100200012:
        raise ValueError('only explicit Run76 collection groups; no development/independent data')
    return 'train' if group < 100200010 else 'validation'


def geometry_ids(session):
    scene = session.episode.multichoice.contract
    env = session.env
    other = [i for i in range(3) if i != scene['selected_block']]
    geoms = [(env._ids['block_geom'], COLORS.index(scene['block_colors'][scene['selected_block']]))]
    geoms += [(g, COLORS.index(scene['block_colors'][slot]))
              for g, slot in zip(session.episode.multichoice.geom_ids, other)]
    geoms += [(env._ids['target_geom'], COLORS.index(scene['target_colors'][scene['selected_target']]))]
    for i, slot in enumerate(j for j in range(3) if j != scene['selected_target']):
        geom = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_GEOM, f'choice_target_{i}')
        geoms.append((geom, COLORS.index(scene['target_colors'][slot])))
    return geoms


def sparse_geometry(model, data, geometries):
    centers = np.zeros((7, 3), np.float32)
    points = np.zeros((7, 8, 3), np.float32)
    present = np.zeros(7, bool)
    corners = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)])
    angles = np.arange(8)*np.pi/4
    for geom, color in geometries:
        centers[color] = data.geom_xpos[geom]
        present[color] = True
        if color < 4:
            local = corners*model.geom_size[geom]
            points[color] = local@data.geom_xmat[geom].reshape(3, 3).T+centers[color]
        else:
            radius = model.geom_size[geom, 0]
            points[color] = centers[color]+np.c_[radius*np.cos(angles), radius*np.sin(angles), np.zeros(8)]
    return centers, points, present


def prepare(job):
    source, output, stride, variants = job
    folder, output = Path(source), Path(output)
    metadata = json.loads((folder/'result.json').read_text())
    seed = metadata['seed']; split = training_split(seed)
    if not metadata.get('collection'):
        raise ValueError('only physical collection sources')
    with np.load(folder/'frames.npz', allow_pickle=False) as z:
        original = {k: z[k].copy() for k in z.files}
    with np.load(folder/'render_audit.npz', allow_pickle=False) as z:
        states, times = z['qpos'].copy(), z['frame_steps'].copy()
    if not np.array_equal(times, original['time_step']):
        raise ValueError('render/observation temporal mismatch')
    indices = np.flatnonzero((times % stride == 0) | np.isin(times, [90, 110, 220]))
    session = DomainSession(Sparse4DVLAConfigV26(language_max_tokens=128,
        visual_memory_mode='episode_anchors_v54'), ACTION_CONTRACT, seed, sample_domain(seed+6001, 0))
    geometries = geometry_ids(session)
    xyz, points, present, visible = [], [], [], []
    started = time.time()
    try:
        session.env.data.qpos[:] = states[0]
        mujoco.mj_forward(session.env.model, session.env.data)
        session.renderer.update_scene(session.env.data, camera='edgearm_wrist')
        probe = np.asarray(Image.fromarray(session.renderer.render()).resize((160, 120)))
        delta = np.abs(probe.astype(np.int16)-original['rgb'][0].astype(np.int16))
        if delta.max() > 2 or delta.mean() > .01:
            raise ValueError('stored qpos does not reproduce source RGB')
        session.renderer.enable_segmentation_rendering()
        for index in indices:
            session.env.data.qpos[:] = states[index]
            mujoco.mj_forward(session.env.model, session.env.data)
            session.renderer.update_scene(session.env.data, camera='edgearm_wrist')
            segmentation = session.renderer.render()
            geom_pixels = np.asarray(Image.fromarray(segmentation[..., 0].astype(np.int32)).resize((160, 120), Image.Resampling.NEAREST))
            object_types = np.asarray(Image.fromarray(segmentation[..., 1].astype(np.int32)).resize((160, 120), Image.Resampling.NEAREST))
            centers, cloud, exists = sparse_geometry(session.env.model, session.env.data, geometries)
            counts = np.zeros(7, np.int64)
            for geom, color in geometries:
                counts[color] = np.count_nonzero((geom_pixels == geom) & (object_types == mujoco.mjtObj.mjOBJ_GEOM))
            xyz.append(centers); points.append(cloud); present.append(exists); visible.append(counts >= 8)
        label_arrays = dict(xyz=np.asarray(xyz), points=np.asarray(points),
                            present=np.asarray(present), visible=np.asarray(visible))
        records = []
        for variant in [-1]+list(range(variants)):
            if variant < 0:
                data = original; perm = np.arange(7)
            else:
                path = folder/f'variant_{variant}'
                meta = json.loads((path/'result.json').read_text())
                perm = np.asarray(meta['color_old_to_new'])
                if sorted(perm[:4].tolist()) != list(range(4)) or sorted(perm[4:].tolist()) != [4, 5, 6]:
                    raise ValueError('within-role color bijection required')
                with np.load(path/'frames.npz', allow_pickle=False) as z:
                    data = {k: z[k].copy() for k in z.files}
                if not np.array_equal(data['time_step'], times):
                    raise ValueError('color variant changed physical timeline')
            labels = {}
            for key, value in label_arrays.items():
                labels[key] = np.empty_like(value)
                labels[key][:, perm] = value
            labels['command'] = data['command'][indices]
            labels['action_valid'] = times[indices] >= 220
            dest = output/f'episode_{seed}_color_{variant+1}'
            dest.mkdir(parents=True, exist_ok=False)
            # Raw geometry/qpos/segmentation never appears in inputs.npz.
            np.savez_compressed(dest/'inputs.npz', rgb=data['rgb'][indices], pose=data['pose'][indices],
                K=data['K'], proprio=data['proprio'][indices], time_step=times[indices], selected=data['selected'])
            np.savez_compressed(dest/'labels.npz', **labels)
            records.append(dict(folder=str(dest), seed=seed, route=seed % 9, split=split,
                physical_parent=str(folder), variant=variant, frames=len(indices),
                source_rgb_replay_mean_error=float(delta.mean()), source_rgb_replay_max_error=int(delta.max())))
        return dict(seed=seed, records=records, seconds=time.time()-started)
    finally:
        session.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--roots', type=Path, nargs='+', required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--workers', type=int, default=6)
    p.add_argument('--stride', type=int, choices=(4, 8), default=8)
    p.add_argument('--variants', type=int, choices=(0, 3), default=3)
    p.add_argument('--limit', type=int, default=108)
    a = p.parse_args()
    if not 1 <= a.workers <= 9 or not 1 <= a.limit <= 108:
        raise ValueError('bounded label preparation')
    paths = sorted({path for root in a.roots for path in root.glob('episode_*/render_audit.npz')})[:a.limit]
    if not paths:
        raise ValueError('no replayable physical collection')
    a.output.mkdir(parents=True, exist_ok=False)
    state = dict(run='Run82', status='running', phase='sparse_geometry_label_preparation',
        started=time.time(), completed=0, total=len(paths), records=[],
        actor_uses_simulator_state=False, label_geometry_from_simulator=True,
        original_ACT_checkpoint=False, independent_acceptance=False,
        production_admission=False, export_admission=False, final_vla_acceptance=False)
    def publish():
        state.update(updated=time.time(), elapsed_seconds=time.time()-state['started'])
        _atomic_json(a.output/'run_state.json', state)
    publish()
    try:
        jobs = [(str(p.parent), str(a.output/'sequences'), a.stride, a.variants) for p in paths]
        with ProcessPoolExecutor(a.workers, mp_context=mp.get_context('spawn')) as pool:
            for future in as_completed([pool.submit(prepare, j) for j in jobs]):
                result = future.result()
                state['completed'] += 1; state['records'].extend(result['records']); publish()
        manifest = dict(records=sorted(state['records'], key=lambda x: (x['seed'], x['variant'])),
            physical_episodes=len(paths), augmented_sequences=len(state['records']),
            validation_is_not_new_independent=True, labels_never_actor_inputs=True)
        _atomic_json(a.output/'manifest.json', manifest)
        state.update(status='complete', phase='labels_ready'); publish()
    except BaseException as exc:
        state.update(status='failed', error=repr(exc)); publish(); raise


if __name__ == '__main__':
    main()
