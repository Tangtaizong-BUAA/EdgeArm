"""Three physical blocks, three independent goals and nine geometric tasks."""
from __future__ import annotations

import hashlib
import json
import types

import mujoco
import numpy as np

from .production_env import BLOCK_COLORS, TARGET_COLORS


def scene_contract(seed: int) -> dict:
    geometry_seed, pair = divmod(seed, 9)
    rng = np.random.default_rng(geometry_seed)
    block_colors = rng.choice(list(BLOCK_COLORS), 3, replace=False).tolist()
    target_colors = rng.permutation(list(TARGET_COLORS)).tolist()
    starts = np.asarray([[0.20, y] for y in (-0.14, 0.0, 0.14)])
    targets = np.asarray([[0.385, y] for y in (-0.14, 0.0, 0.14)])
    targets += rng.uniform(-0.002, 0.002, size=(3, 2))
    block_index, target_index = divmod(pair, 3)
    clearances = []
    for i, start in enumerate(starts):
        for goal in targets:
            delta = goal - start
            for j, other in enumerate(starts):
                if i == j:
                    continue
                fraction = np.clip(np.dot(other - start, delta) / np.dot(delta, delta), 0, 1)
                clearances.append(float(np.linalg.norm(other - start - fraction * delta) - 2 * np.sqrt(2) * 0.025))
    target_gap = min(float(np.linalg.norm(targets[i] - targets[j]) - 0.11) for i in range(3) for j in range(i))
    if min(clearances) < 0.005 or target_gap <= 0:
        raise ValueError("multi-choice geometry does not have clear, nonoverlapping routes")
    layout = dict(block_positions=starts.tolist(), target_positions=targets.tolist(), block_colors=block_colors, target_colors=target_colors)
    layout_hash = hashlib.sha256(json.dumps(layout, sort_keys=True).encode()).hexdigest()
    return dict(
        **layout, format="edgearm-physical-multichoice-v1", seed=seed,
        scene_id=layout_hash, selected_block=block_index, selected_target=target_index,
        instruction=f"把{BLOCK_COLORS[block_colors[block_index]][1]}方块推到{TARGET_COLORS[target_colors[target_index]][1]}目标区域。",
        minimum_route_clearance_m=min(clearances), minimum_target_edge_gap_m=target_gap,
        color_is_policy_input=False, neutral_home_reset=False,
    )


class MultiChoiceScene:
    def __init__(self, env):
        self.env = env
        self.contract = None
        self.invalid_contact_events = 0
        self.max_unselected_displacement_m = 0.0
        self.geom_ids = [mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_GEOM, f"choice_block_geom_{i}") for i in range(2)]
        self.joint_ids = [mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_JOINT, f"choice_block_joint_{i}") for i in range(2)]
        if min(self.geom_ids + self.joint_ids) < 0:
            raise ValueError("physical multi-choice bodies are missing")

    def prepare(self, seed):
        self.contract = scene_contract(seed)
        scene = self.contract
        def sample(_env, stress):
            if stress:
                raise ValueError("multi-choice bootstrap does not enable stress")
            return np.asarray(scene['block_positions'][scene['selected_block']]), np.asarray(scene['target_positions'][scene['selected_target']])
        self.env._sample_task = types.MethodType(sample, self.env)
        self.env._multichoice_substep_observer = None

    def install(self):
        env, scene = self.env, self.contract
        self.invalid_contact_events = 0
        self.max_unselected_displacement_m = 0.0
        self.initial_unselected = []
        others = [i for i in range(3) if i != scene['selected_block']]
        for slot, index in enumerate(others):
            joint = self.joint_ids[slot]
            address = env.model.jnt_qposadr[joint]
            velocity = env.model.jnt_dofadr[joint]
            xy = scene['block_positions'][index]
            env.data.qpos[address:address+7] = [*xy, 0.051, 1, 0, 0, 0]
            env.data.qvel[velocity:velocity+6] = 0
            env.model.geom_rgba[self.geom_ids[slot]] = BLOCK_COLORS[scene['block_colors'][index]][0]
            self.initial_unselected.append(np.asarray(xy))
        other_targets = [i for i in range(3) if i != scene['selected_target']]
        for slot, index in enumerate(other_targets):
            geom = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_GEOM, f"choice_target_{slot}")
            env.model.geom_pos[geom, :2] = scene['target_positions'][index]
            env.model.geom_rgba[geom] = TARGET_COLORS[scene['target_colors'][index]][0]
        env.color_name = scene['block_colors'][scene['selected_block']]
        env.target_color_name = scene['target_colors'][scene['selected_target']]
        env.model.geom_rgba[env._ids['block_geom']] = BLOCK_COLORS[env.color_name][0]
        env.model.geom_rgba[env._ids['target_geom']] = TARGET_COLORS[env.target_color_name][0]
        env._set_task_language()
        # Decorative objects are not task candidates.
        for geom in env._clutter_geoms:
            env.model.geom_rgba[geom, 3] = 0
        mujoco.mj_forward(env.model, env.data)
        self.observe()
        if self.invalid_contact_events:
            raise ValueError("multi-choice reset intersects another object")
        env._multichoice_substep_observer = self.observe

    def observe(self):
        env = self.env
        allowed = {env._desk_geom, env._floor_geom}
        for contact in env.data.contact:
            a, b = int(contact.geom1), int(contact.geom2)
            if (a in self.geom_ids and b not in allowed) or (b in self.geom_ids and a not in allowed):
                self.invalid_contact_events += 1
        for slot, joint in enumerate(self.joint_ids):
            address = env.model.jnt_qposadr[joint]
            distance = np.linalg.norm(env.data.qpos[address:address+2] - self.initial_unselected[slot])
            self.max_unselected_displacement_m = max(self.max_unselected_displacement_m, float(distance))

    @property
    def failed(self):
        return self.invalid_contact_events > 0 or self.max_unselected_displacement_m > 0.002

    def audit(self):
        return dict(**self.contract, unselected_contact_events=self.invalid_contact_events,
                    max_unselected_displacement_m=self.max_unselected_displacement_m,
                    unselected_objects_untouched=not self.failed)
