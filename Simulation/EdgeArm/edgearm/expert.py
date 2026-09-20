"""Privileged geometric teacher used by M3 and as the M4 RL prior."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .env import EdgeArmEnv


@dataclass
class ExpertConfig:
    contact_offset: float = 0.042
    push_overshoot: float = 0.018
    z_height: float = 0.062
    damping: float = 0.002
    gain: float = 0.90
    obstacle_clearance: float = 0.07


class GeometricExpert:
    def __init__(self, env: EdgeArmEnv, config: ExpertConfig | None = None):
        self.env = env
        self.config = config or ExpertConfig()

    def action(self) -> np.ndarray:
        env = self.env
        block = env.block_xy()
        target = env.target_xy
        direction = env._unit(target - block)
        ee = env.tool_xyz()

        contact_xy = block - direction * self.config.contact_offset
        along = float(np.dot(block - ee[:2], direction))
        offset = block - ee[:2]
        lateral = float(abs(direction[0] * offset[1] - direction[1] * offset[0]))
        if along < self.config.contact_offset + 0.022 and lateral < 0.045:
            # Small operational-space increments are deliberately used instead
            # of a single far-away IK target; this avoids joint-limit jumps and
            # yields the low-speed corrective behavior required by the spec.
            desired_xy = ee[:2] + direction * 0.026
        else:
            desired_xy = contact_xy

        if env.obstacle_enabled and self._segment_hits_obstacle(block, target):
            normal = np.array([-direction[1], direction[0]])
            side = np.sign(np.dot(block - env.obstacle_xy, normal)) or 1.0
            waypoint = env.obstacle_xy + normal * side * (
                env.obstacle_half.max() + self.config.obstacle_clearance
            )
            if np.linalg.norm(block - waypoint) > 0.055:
                local_direction = env._unit(waypoint - block)
                desired_xy = block - local_direction * self.config.contact_offset
                if np.linalg.norm(ee[:2] - desired_xy) < 0.022:
                    desired_xy = ee[:2] + local_direction * 0.022

        delta = desired_xy - ee[:2]
        return np.clip(
            self.config.gain * delta / env.config.max_cartesian_delta, -1.0, 1.0
        ).astype(np.float32)

    def _segment_hits_obstacle(self, start: np.ndarray, end: np.ndarray) -> bool:
        center = self.env.obstacle_xy
        direction = end - start
        denom = float(np.dot(direction, direction))
        if denom < 1e-8:
            return False
        t = np.clip(np.dot(center - start, direction) / denom, 0.0, 1.0)
        closest = start + t * direction
        inflated = self.env.obstacle_half + 0.035
        return bool(np.all(np.abs(closest - center) <= inflated))
