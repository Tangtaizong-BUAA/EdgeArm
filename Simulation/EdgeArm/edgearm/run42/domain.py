"""Bounded, seed-recorded task-preserving randomization; no robot I/O."""

from dataclasses import replace
import hashlib
import json
import math
import types

import mujoco
import numpy as np

from ..multichoice_scene_v1 import MultiChoiceScene, scene_contract
from ..production_env import BLOCK_COLORS
from ..staged_push_rl import nominal_v10_training_config


STAGES = ('nominal', 'physical', 'layout_material', 'camera_16_24', 'mixed_16_24_35')
CONTEXT_DIM = 16


def equivalent_intrinsics(focal_mm, width=160, height=120):
    """Diagonal-equivalent focal length relative to a 36x24 mm full frame.

    Preserve the requested output aspect ratio, with virtual sensor diagonal
    sqrt(36^2+24^2). Thus '24 mm equivalent' is not a 24 mm tiny-sensor lens.
    """
    if not math.isfinite(focal_mm) or not 8 <= focal_mm <= 85 or min(width, height) <= 0:
        raise ValueError('invalid equivalent focal length/resolution')
    diagonal = math.hypot(36., 24.)
    sensor_h = diagonal / math.sqrt(1 + (width / height) ** 2)
    f_pixels = focal_mm * height / sensor_h
    k = np.array([[f_pixels, 0, (width - 1) / 2], [0, f_pixels, (height - 1) / 2], [0, 0, 1]], np.float32)
    return k, math.degrees(2 * math.atan(sensor_h / (2 * focal_mm)))


def sample_domain(seed, stage):
    if stage not in range(len(STAGES)) or seed < 0:
        raise ValueError('unknown curriculum stage/seed')
    rng = np.random.default_rng(seed)
    p = dict(seed=int(seed), stage=int(stage), stage_name=STAGES[stage], mass_scale=1.,
             friction=.7, pusher_friction=1., position_jitter_m=0., block_count=3, target_count=3,
             light_gain=1., desk_reflectance=0., block_specular=0., texture_strength=0.,
             focal_equivalent_mm=None, focal_definition='36x24mm diagonal equivalent',
             intrinsics_scale_error=0., exposure_ev=0., exposure_drift_ev=0., white_balance=[1., 1., 1.],
             noise_std_255=0., shot_noise=0., blur_radius_px=0., depth_scale_error=0.,
             depth_bias_m=0., depth_noise_relative=0., depth_dropout=0., uncertainty_scale=1.)
    if stage == 0:
        return p
    strength = (.25, .50, .75, 1.)[stage - 1]
    p.update(mass_scale=float(rng.uniform(1 - .45 * strength, 1 + .50 * strength)),
             friction=float(rng.uniform(.7 - .30 * strength, .7 + .40 * strength)),
             pusher_friction=float(rng.uniform(1 - .20 * strength, 1 + .20 * strength)))
    if stage >= 2:
        p.update(position_jitter_m=.025 * strength, block_count=int(rng.integers(1, 4)),
                 target_count=int(rng.integers(1, 4)), light_gain=float(rng.uniform(.7, 1.3)),
                 desk_reflectance=float(rng.uniform(0, .3 * strength)),
                 block_specular=float(rng.uniform(0, .7 * strength)), texture_strength=.35 * strength,
                 exposure_ev=float(rng.uniform(-.7, .7) * strength),
                 exposure_drift_ev=float(rng.uniform(0, .2) * strength),
                 white_balance=rng.uniform(.92, 1.08, 3).tolist(),
                 noise_std_255=float(rng.uniform(0, 6.) * strength), shot_noise=.006 * strength,
                 blur_radius_px=float(rng.uniform(0, .9) * strength),
                 depth_scale_error=float(rng.uniform(-.15, .15) * strength),
                 depth_bias_m=float(rng.uniform(-.015, .015) * strength),
                 depth_noise_relative=.06 * strength, depth_dropout=.10 * strength,
                 uncertainty_scale=float(rng.uniform(1, 2.5)))
    if stage >= 3:
        lenses = [None, 16., 24.] if stage == 3 else [None, 16., 24., 35.]
        p['focal_equivalent_mm'] = lenses[int(rng.integers(len(lenses)))]
        p['intrinsics_scale_error'] = float(rng.uniform(-.025, .025) * strength)
    return p


def domain_context(p):
    # Training critic only; zero is the original plant. Never an actor input.
    if p['stage'] == 0:
        return np.zeros(CONTEXT_DIM, np.float32)
    return np.asarray([p['mass_scale'] - 1, p['friction'] - .7, p['pusher_friction'] - 1,
        p['position_jitter_m'] / .025, (p['block_count'] - 3) / 2, (p['target_count'] - 3) / 2,
        p['light_gain'] - 1, p['desk_reflectance'], p['block_specular'], p['texture_strength'],
        (p['focal_equivalent_mm'] or 0) / 35, p['exposure_ev'], p['noise_std_255'] / 6,
        p['depth_scale_error'], p['depth_noise_relative'], p['depth_dropout']], np.float32)


def add_context(privileged, context):
    p, c = np.asarray(privileged), np.asarray(context)
    c = np.broadcast_to(c, (*p.shape[:-1], CONTEXT_DIM))
    return np.concatenate((p[..., :-6], c, p[..., -6:]), axis=-1).astype(np.float32)


def environment_config(p):
    cfg = nominal_v10_training_config()
    return replace(cfg, block_mass_scale_range=(p['mass_scale'],) * 2,
                   desk_block_slide_friction_range=(p['friction'],) * 2,
                   pusher_block_slide_friction_range=(p['pusher_friction'],) * 2)


def randomized_layout(seed, p):
    scene = scene_contract(seed)
    if p['stage'] == 0:
        return scene
    rng = np.random.default_rng(p['seed'] + 10)
    b, g = scene['selected_block'], scene['selected_target']
    active_b = [b] + rng.permutation([i for i in range(3) if i != b]).tolist()[:p['block_count'] - 1]
    active_g = [g] + rng.permutation([i for i in range(3) if i != g]).tolist()[:p['target_count'] - 1]
    nominal_b, nominal_g = np.asarray(scene['block_positions']), np.asarray(scene['target_positions'])
    for _ in range(64):
        blocks = nominal_b + rng.uniform(-p['position_jitter_m'], p['position_jitter_m'], (3, 2))
        goals = nominal_g + rng.uniform(-p['position_jitter_m'], p['position_jitter_m'], (3, 2))
        direction = goals[g] - blocks[b]
        length = np.linalg.norm(direction)
        gaps = [np.linalg.norm(goals[i] - goals[j]) for i in active_g for j in active_g if i < j]
        clear = [np.linalg.norm(blocks[i] - blocks[b] - np.clip(np.dot(blocks[i] - blocks[b], direction)
                 / np.dot(direction, direction), 0, 1) * direction) for i in active_b if i != b]
        if .12 <= length <= .27 and min(gaps, default=1) > .12 and min(clear, default=1) > .08:
            break
    else:
        raise ValueError('no bounded non-overlapping layout; do not silently substitute an easy seed')
    scene.update(block_positions=blocks.tolist(), target_positions=goals.tolist(), active_blocks=active_b,
                 active_targets=active_g, block_count=len(active_b), target_count=len(active_g),
                 minimum_route_clearance_m=min(clear, default=1) - 2 * math.sqrt(2) * .025,
                 minimum_target_edge_gap_m=min(gaps, default=1) - .11,
                 format='run42-bounded-multichoice-v1', domain_seed=p['seed'])
    scene['scene_id'] = hashlib.sha256(json.dumps(scene, sort_keys=True).encode()).hexdigest()
    return scene


class DomainScene(MultiChoiceScene):
    def __init__(self, env, parameters):
        super().__init__(env)
        self.parameters = parameters

    def prepare(self, seed):
        self.contract = randomized_layout(seed, self.parameters)
        scene = self.contract
        def sample(env, stress):
            if stress:
                raise ValueError('unregistered stress mode')
            return np.asarray(scene['block_positions'][scene['selected_block']]), np.asarray(scene['target_positions'][scene['selected_target']])
        self.env._sample_task = types.MethodType(sample, self.env)
        self.env._multichoice_substep_observer = None

    def install(self):
        super().install()
        if self.parameters['stage'] == 0:
            return
        e, p, s = self.env, self.parameters, self.contract
        for slot, index in enumerate(i for i in range(3) if i != s['selected_block']):
            geom, joint = self.geom_ids[slot], self.joint_ids[slot]
            body = e.model.geom_bodyid[geom]
            if index not in s['active_blocks']:
                e.model.geom_contype[geom] = e.model.geom_conaffinity[geom] = 0
                e.model.geom_rgba[geom, 3] = 0
                e.model.body_gravcomp[body] = 1
                address = e.model.jnt_qposadr[joint]
                e.data.qpos[address:address + 3] = [2 + slot, 2, .2]
                self.initial_unselected[slot] = np.asarray([2 + slot, 2])
            else:
                e.model.body_mass[body] *= p['mass_scale']
                e.model.body_inertia[body] *= p['mass_scale']
                e.model.geom_friction[geom] = [p['friction'], .018, .002]
        for slot, index in enumerate(i for i in range(3) if i != s['selected_target']):
            geom = mujoco.mj_name2id(e.model, mujoco.mjtObj.mjOBJ_GEOM, f'choice_target_{slot}')
            if index not in s['active_targets']:
                e.model.geom_rgba[geom, 3] = 0
        if p['stage'] >= 2:
            self._materials()
            e.model.light_diffuse[:] *= p['light_gain']
        if p['focal_equivalent_mm'] is not None:
            _, fovy = equivalent_intrinsics(p['focal_equivalent_mm'])
            e.model.cam_fovy[e._ids['cameras']['wrist']] = fovy
        # Mass/inertia edits must update model constants without changing state.
        q, v, ctrl = e.data.qpos.copy(), e.data.qvel.copy(), e.data.ctrl.copy()
        mujoco.mj_setConst(e.model, e.data)
        e.data.qpos[:], e.data.qvel[:], e.data.ctrl[:] = q, v, ctrl
        mujoco.mj_forward(e.model, e.data)
        if e.block_target_coverage() > 0:
            raise ValueError('randomized task starts inside target')

    def _materials(self):
        e, p, s = self.env, self.parameters, self.contract
        materials = list(e._surface_materials)
        if len(materials) < 5:
            raise ValueError('need distinct desk/block/background materials')
        rng = np.random.default_rng(p['seed'] + 20)
        e.model.geom_matid[e._desk_geom] = materials[0]
        e.model.geom_matid[e._floor_geom] = materials[-1]
        e.model.mat_rgba[materials[0]] = [*.8 * np.ones(3), 1]
        e.model.mat_reflectance[materials[0]] = p['desk_reflectance']
        e.model.mat_specular[materials[0]] = .1 + p['desk_reflectance']
        others = [i for i in range(3) if i != s['selected_block']]
        geoms = [(e._ids['block_geom'], s['selected_block']), *zip(self.geom_ids, others)]
        for slot, (geom, index) in enumerate(geoms, 1):
            mat = materials[slot]
            e.model.geom_matid[geom] = mat
            rgba = list(BLOCK_COLORS[s['block_colors'][index]][0])
            rgba[3] = float(index in s['active_blocks'])
            e.model.mat_rgba[mat] = rgba
            e.model.mat_specular[mat] = p['block_specular']
            e.model.mat_reflectance[mat] = 0.02 * p['block_specular']
        for mat in materials[:4]:
            # Neutral multiplicative patterns retain instruction color identity.
            for tex in set(int(i) for i in e.model.mat_texid[mat].ravel() if i >= 0):
                h, w, channels = int(e.model.tex_height[tex]), int(e.model.tex_width[tex]), int(e.model.tex_nchannel[tex])
                yy, xx = np.mgrid[:h, :w]
                pattern = .5 + .3 * np.sin(xx * rng.uniform(.1, .5) + yy * rng.uniform(.05, .2))
                pattern += rng.normal(0, .08, (h, w))
                intensity = np.clip(1 - p['texture_strength'] * pattern, .55, 1) * 255
                image = np.repeat(intensity[..., None], channels, axis=-1).astype(np.uint8)
                if channels == 4:
                    image[..., 3] = 255
                begin = int(e.model.tex_adr[tex])
                e.model.tex_data[begin:begin + image.size] = image.ravel()
