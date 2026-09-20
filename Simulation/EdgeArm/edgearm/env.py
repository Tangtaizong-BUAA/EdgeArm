"""M2 MuJoCo environment with deploy-shaped observations and safe joint actions."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import mujoco
import numpy as np

from .config import JOINT_NAMES, SCENE_PATH


@dataclass(frozen=True)
class EdgeArmEnvConfig:
    image_size: int = 64
    max_steps: int = 72
    frame_skip: int = 5
    max_cartesian_delta: float = 0.018
    success_radius: float = 0.055
    success_hold_steps: int = 3
    workspace_x: tuple[float, float] = (0.07, 0.43)
    workspace_y: tuple[float, float] = (-0.27, 0.27)
    workspace_z: tuple[float, float] = (0.0, 0.30)
    block_xy_low: tuple[float, float] = (0.17, -0.13)
    block_xy_high: tuple[float, float] = (0.24, 0.13)
    target_xy_low: tuple[float, float] = (0.29, -0.15)
    target_xy_high: tuple[float, float] = (0.37, 0.15)
    obstacle_probability: float = 0.35
    domain_randomization: float = 1.0


class EdgeArmEnv:
    """Small continuous-control environment without a Gymnasium dependency.

    The contact plate is a kinematic proxy that is updated from the real SO-101
    gripper site. It cannot be moved directly: the only external action is a
    bounded six-joint delta. This keeps headless training deterministic while
    preserving MuJoCo contact dynamics for the block, desk, and obstacles.
    """

    action_dim = 2

    def __init__(self, config: EdgeArmEnvConfig | None = None, seed: int = 0):
        self.config = config or EdgeArmEnvConfig()
        self.model = mujoco.MjModel.from_xml_path(str(SCENE_PATH))
        self.data = mujoco.MjData(self.model)
        self.rng = np.random.default_rng(seed)
        self.seed = seed
        self.estop = False
        self.step_count = 0
        self.success_streak = 0
        self.last_distance = 0.0
        self.obstacle_enabled = False
        self.target_xy = np.zeros(2, dtype=np.float64)
        self.obstacle_xy = np.zeros(2, dtype=np.float64)
        self.obstacle_half = np.array([0.035, 0.055], dtype=np.float64)
        self._tool_xy = np.zeros(2, dtype=np.float64)
        self._ids = self._resolve_ids()
        self.joint_ranges = np.asarray(
            [self.model.jnt_range[self._ids["joints"][name]] for name in JOINT_NAMES], dtype=np.float64
        )
        self._default_friction = self.model.geom_friction[self._ids["block_geom"]].copy()
        self._default_mass = float(self.model.body_mass[self._ids["block_body"]])

    def _resolve_ids(self) -> dict[str, Any]:
        def obj(kind: mujoco.mjtObj, name: str) -> int:
            value = mujoco.mj_name2id(self.model, kind, name)
            if value < 0:
                raise ValueError(f"Scene is missing {name}")
            return value

        return {
            "joints": {name: obj(mujoco.mjtObj.mjOBJ_JOINT, name) for name in JOINT_NAMES},
            "actuators": {name: obj(mujoco.mjtObj.mjOBJ_ACTUATOR, name) for name in JOINT_NAMES},
            "block_joint": obj(mujoco.mjtObj.mjOBJ_JOINT, "push_block_freejoint"),
            "block_body": obj(mujoco.mjtObj.mjOBJ_BODY, "push_block"),
            "block_geom": obj(mujoco.mjtObj.mjOBJ_GEOM, "push_block_geom"),
            "target_geom": obj(mujoco.mjtObj.mjOBJ_GEOM, "target_zone"),
            "obstacle_geom": obj(mujoco.mjtObj.mjOBJ_GEOM, "obstacle"),
            "plate_body": obj(mujoco.mjtObj.mjOBJ_BODY, "arm_push_plate"),
            "plate_geom": obj(mujoco.mjtObj.mjOBJ_GEOM, "arm_push_plate_geom"),
            "gripper_site": obj(mujoco.mjtObj.mjOBJ_SITE, "gripperframe"),
            "link_bodies": [
                obj(mujoco.mjtObj.mjOBJ_BODY, name)
                for name in ("base", "shoulder", "upper_arm", "lower_arm", "wrist", "gripper")
            ],
        }

    @property
    def observation_dim(self) -> int:
        return self.state_vector().shape[0]

    def config_dict(self) -> dict[str, Any]:
        return asdict(self.config)

    def reset(
        self,
        seed: int | None = None,
        *,
        obstacle: bool | None = None,
        randomize: bool = True,
    ) -> dict[str, np.ndarray]:
        if seed is not None:
            self.seed = seed
            self.rng = np.random.default_rng(seed)
        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        self.step_count = 0
        self.success_streak = 0
        self.estop = False

        if randomize:
            block_xy, target_xy = self._sample_task()
        else:
            block_xy = np.array([0.20, -0.08])
            target_xy = np.array([0.33, 0.08])
        self.target_xy = target_xy
        self._set_block_pose(block_xy)
        self._set_target_pose(target_xy)

        self.obstacle_enabled = (
            bool(obstacle)
            if obstacle is not None
            else bool(self.rng.random() < self.config.obstacle_probability)
        )
        self._configure_obstacle(block_xy, target_xy)
        self._randomize_dynamics() if randomize else self._restore_dynamics()

        direction = self._unit(target_xy - block_xy)
        start_xy = block_xy - direction * 0.095
        q = self.solve_ik(np.array([start_xy[0], start_xy[1], 0.064]))
        self._tool_xy = start_xy.copy()
        self.data.qpos[:6] = q
        self.data.qvel[:6] = 0.0
        self.data.ctrl[:] = q
        mujoco.mj_forward(self.model, self.data)
        self._sync_plate()
        mujoco.mj_forward(self.model, self.data)
        self.last_distance = self.distance_to_target()
        return self.observation()

    def _sample_task(self) -> tuple[np.ndarray, np.ndarray]:
        for _ in range(200):
            block = self.rng.uniform(self.config.block_xy_low, self.config.block_xy_high)
            distance = float(self.rng.uniform(0.12, 0.18))
            angle = float(self.rng.uniform(-0.30, 0.30))
            target = block + distance * np.array([np.cos(angle), np.sin(angle)])
            if (
                self.config.target_xy_low[0] <= target[0] <= self.config.target_xy_high[0]
                and self.config.target_xy_low[1] <= target[1] <= self.config.target_xy_high[1]
            ):
                return block, target
        raise RuntimeError("Could not sample a valid EdgeArm task")

    def _set_block_pose(self, xy: np.ndarray) -> None:
        address = self.model.jnt_qposadr[self._ids["block_joint"]]
        self.data.qpos[address : address + 7] = [xy[0], xy[1], 0.051, 1.0, 0.0, 0.0, 0.0]

    def _set_target_pose(self, xy: np.ndarray) -> None:
        self.model.geom_pos[self._ids["target_geom"], :2] = xy

    def _configure_obstacle(self, block: np.ndarray, target: np.ndarray) -> None:
        geom = self._ids["obstacle_geom"]
        if self.obstacle_enabled:
            midpoint = 0.48 * block + 0.52 * target
            normal = np.array([-(target - block)[1], (target - block)[0]])
            normal = self._unit(normal)
            self.obstacle_xy = midpoint + normal * float(self.rng.uniform(-0.035, 0.035))
            self.model.geom_pos[geom, :2] = self.obstacle_xy
            self.model.geom_contype[geom] = 1
            self.model.geom_conaffinity[geom] = 1
            self.model.geom_rgba[geom, 3] = 1.0
        else:
            self.obstacle_xy = np.array([0.0, 0.0])
            self.model.geom_pos[geom, :2] = [0.55, 0.28]
            self.model.geom_contype[geom] = 0
            self.model.geom_conaffinity[geom] = 0
            self.model.geom_rgba[geom, 3] = 0.0

    def _randomize_dynamics(self) -> None:
        scale = self.config.domain_randomization
        geom = self._ids["block_geom"]
        self.model.geom_friction[geom, 0] = float(self.rng.uniform(0.55, 1.15) ** scale)
        self.model.body_mass[self._ids["block_body"]] = self._default_mass * float(
            self.rng.uniform(0.75, 1.35) ** scale
        )

    def _restore_dynamics(self) -> None:
        self.model.geom_friction[self._ids["block_geom"]] = self._default_friction
        self.model.body_mass[self._ids["block_body"]] = self._default_mass

    def solve_ik(self, target_xyz: np.ndarray, initial: np.ndarray | None = None) -> np.ndarray:
        scratch = mujoco.MjData(self.model)
        mujoco.mj_resetDataKeyframe(self.model, scratch, 0)
        if initial is not None:
            scratch.qpos[:6] = np.asarray(initial)
        site = self._ids["gripper_site"]
        for _ in range(140):
            mujoco.mj_forward(self.model, scratch)
            error = np.asarray(target_xyz) - scratch.site_xpos[site]
            if np.linalg.norm(error) < 2e-4:
                break
            jac = np.zeros((3, self.model.nv))
            jac_rot = np.zeros((3, self.model.nv))
            mujoco.mj_jacSite(self.model, scratch, jac, jac_rot, site)
            reduced = jac[:, :5]
            delta = reduced.T @ np.linalg.solve(reduced @ reduced.T + 1e-3 * np.eye(3), error)
            scratch.qpos[:5] += np.clip(delta, -0.08, 0.08)
            scratch.qpos[:5] = np.clip(scratch.qpos[:5], self.joint_ranges[:5, 0], self.joint_ranges[:5, 1])
        scratch.qpos[5] = 0.35
        return scratch.qpos[:6].copy()

    def step(self, action: np.ndarray) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        action = np.asarray(action, dtype=np.float64)
        if action.shape != (self.action_dim,) or not np.all(np.isfinite(action)):
            raise ValueError(f"Expected a finite ({self.action_dim},) action")
        if self.estop:
            return self.observation(), -1.0, False, True, {"safety_stop": "estop"}

        action = np.clip(action, -1.0, 1.0)
        requested_xy = self._tool_xy + action * self.config.max_cartesian_delta
        safe_xy, safe_q, safety_reason = self._safety_filter(requested_xy)
        self._tool_xy = safe_xy
        self.data.ctrl[:] = safe_q
        for _ in range(self.config.frame_skip):
            # M2-M4 output a bounded table-plane push action. The safety layer
            # maps it to SO-101 joint targets; servo dynamics are an M5/M7 concern.
            self.data.qpos[:6] = safe_q
            self.data.qvel[:6] = 0.0
            mujoco.mj_forward(self.model, self.data)
            self._sync_plate()
            mujoco.mj_step(self.model, self.data)
        self.data.qpos[:6] = safe_q
        self.data.qvel[:6] = 0.0
        mujoco.mj_forward(self.model, self.data)
        self._sync_plate()
        self.step_count += 1

        distance = self.distance_to_target()
        progress = self.last_distance - distance
        self.last_distance = distance
        if distance < self.config.success_radius:
            self.success_streak += 1
        else:
            self.success_streak = 0
        success = self.success_streak >= self.config.success_hold_steps
        out_of_bounds = not self._block_in_workspace()
        terminated = success or out_of_bounds
        truncated = self.step_count >= self.config.max_steps
        reward = 24.0 * progress - 0.015 * float(np.square(action).sum()) - 0.02
        if success:
            reward += 8.0
        if out_of_bounds:
            reward -= 5.0
        if safety_reason:
            reward -= 0.1
        info = {
            "success": success,
            "distance": distance,
            "progress": progress,
            "safety_clipped": bool(safety_reason),
            "safety_reason": safety_reason,
            "obstacle": self.obstacle_enabled,
        }
        return self.observation(), float(reward), terminated, truncated, info

    def _safety_filter(self, requested_xy: np.ndarray) -> tuple[np.ndarray, np.ndarray, str]:
        clipped_xy = np.array(
            [
                np.clip(requested_xy[0], *self.config.workspace_x),
                np.clip(requested_xy[1], *self.config.workspace_y),
            ],
            dtype=np.float64,
        )
        reasons: list[str] = []
        if not np.allclose(clipped_xy, requested_xy):
            reasons.append("workspace")
        target_q = self.solve_ik(
            np.array([clipped_xy[0], clipped_xy[1], 0.064]), initial=self.data.qpos[:6]
        )
        clipped_q = np.clip(target_q, self.joint_ranges[:, 0], self.joint_ranges[:, 1])
        if not np.allclose(target_q, clipped_q):
            reasons.append("joint_limit")
        return clipped_xy, clipped_q, "+".join(reasons)

    def _sync_plate(self) -> None:
        body = self._ids["plate_body"]
        mocap = self.model.body_mocapid[body]
        position = np.array([self._tool_xy[0], self._tool_xy[1], 0.057], dtype=np.float64)
        # The M2 task tool is a passive, task-aligned wide pad. Wrist-level
        # orientation dynamics are intentionally deferred to the M5 servo-data
        # route; the contact point itself still follows only the arm joints.
        direction = self._unit(self.target_xy - self.block_xy())
        normal_yaw = float(np.arctan2(direction[1], direction[0]))
        yaw = normal_yaw - np.pi / 2.0
        self.data.mocap_pos[mocap] = position
        self.data.mocap_quat[mocap] = [np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)]

    def observation(self) -> dict[str, np.ndarray]:
        return {
            "image": self.render_topdown(),
            "joint_state": np.concatenate([self.data.qpos[:6], self.data.qvel[:6]]).astype(np.float32),
            "task": np.concatenate(
                [self.target_xy, self.obstacle_xy, [float(self.obstacle_enabled)]], dtype=np.float64
            ).astype(np.float32),
            "state": self.state_vector(),
        }

    def state_vector(self) -> np.ndarray:
        block = self.block_xy()
        delta = self.target_xy - block
        vector = np.concatenate(
            [
                self.data.qpos[:6],
                np.clip(self.data.qvel[:6], -4.0, 4.0) / 4.0,
                self.tool_xyz(),
                block,
                self.target_xy,
                delta,
                self.obstacle_xy,
                self.obstacle_half,
                [float(self.obstacle_enabled), self.step_count / self.config.max_steps],
            ]
        )
        return vector.astype(np.float32)

    def block_xy(self) -> np.ndarray:
        return self.data.xpos[self._ids["block_body"], :2].copy()

    def ee_xyz(self) -> np.ndarray:
        return self.data.site_xpos[self._ids["gripper_site"]].copy()

    def tool_xyz(self) -> np.ndarray:
        return np.array([self._tool_xy[0], self._tool_xy[1], 0.057], dtype=np.float64)

    def distance_to_target(self) -> float:
        return float(np.linalg.norm(self.block_xy() - self.target_xy))

    def _block_in_workspace(self) -> bool:
        x, y = self.block_xy()
        return 0.08 <= x <= 0.43 and -0.28 <= y <= 0.28

    @staticmethod
    def _unit(vector: np.ndarray) -> np.ndarray:
        norm = float(np.linalg.norm(vector))
        return vector / max(norm, 1e-8)

    def render_topdown(self) -> np.ndarray:
        """Deterministic RGB raster for headless training and deployment-interface tests."""
        size = self.config.image_size
        image = np.full((size, size, 3), [154, 158, 146], dtype=np.uint8)

        def pixel(xy: np.ndarray) -> tuple[int, int]:
            x = int(np.clip((xy[0] - 0.02) / 0.46 * (size - 1), 0, size - 1))
            y = int(np.clip((0.30 - xy[1]) / 0.60 * (size - 1), 0, size - 1))
            return x, y

        def disk(xy: np.ndarray, radius_m: float, color: tuple[int, int, int]) -> None:
            cx, cy = pixel(xy)
            radius = max(1, int(radius_m / 0.46 * size))
            yy, xx = np.ogrid[:size, :size]
            mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= radius**2
            image[mask] = color

        def rect(xy: np.ndarray, half: np.ndarray, color: tuple[int, int, int]) -> None:
            x0, y0 = pixel(xy - np.array([half[0], -half[1]]))
            x1, y1 = pixel(xy + np.array([half[0], -half[1]]))
            image[min(y0, y1) : max(y0, y1) + 1, min(x0, x1) : max(x0, x1) + 1] = color

        def line(a: np.ndarray, b: np.ndarray, color: tuple[int, int, int], width: int = 1) -> None:
            x0, y0 = pixel(a)
            x1, y1 = pixel(b)
            count = max(abs(x1 - x0), abs(y1 - y0), 1) + 1
            xs = np.linspace(x0, x1, count).round().astype(int)
            ys = np.linspace(y0, y1, count).round().astype(int)
            for x, y in zip(xs, ys, strict=True):
                image[max(0, y - width) : min(size, y + width + 1), max(0, x - width) : min(size, x + width + 1)] = color

        disk(self.target_xy, 0.055, (55, 186, 75))
        if self.obstacle_enabled:
            rect(self.obstacle_xy, self.obstacle_half, (186, 55, 43))
        points = [self.data.xpos[body, :2].copy() for body in self._ids["link_bodies"]]
        for start, end in zip(points, points[1:]):
            line(start, end, (236, 188, 35), width=1)
        disk(self.block_xy(), 0.025, (32, 76, 226))
        disk(self._tool_xy, 0.012, (240, 240, 225))
        return image
