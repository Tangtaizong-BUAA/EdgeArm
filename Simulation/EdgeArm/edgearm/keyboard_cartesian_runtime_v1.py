"""Keyboard Cartesian control with automatic block-face gripper alignment."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from enum import Enum
from typing import Any

import mujoco
import numpy as np
from scipy.optimize import least_squares

from .phone_teleop_runtime_v1 import IKResult


KEYBOARD_CARTESIAN_RUNTIME_VERSION = (
    "edgearm-keyboard-cartesian-runtime-v4-planner-seed-hold-sync"
)


class KeyboardEvent(str, Enum):
    MOVE = "move"
    RESET = "reset"
    QUIT = "quit"
    NONE = "none"


@dataclass(frozen=True)
class KeyboardCartesianConfig:
    translation_step_m: float = 0.006
    minimum_z_m: float = 0.055
    maximum_z_m: float = 0.095
    position_tolerance_m: float = 0.0025
    face_normal_tolerance_rad: float = 0.085
    direction_weight: float = 0.20
    damping: float = 1.0e-3
    iterations: int = 120
    joint_step_limit_rad: float = 0.07
    face_switch_hysteresis_m: float = 0.008
    table_angle_step_rad: float = np.deg2rad(1.0)
    minimum_table_angle_rad: float = np.deg2rad(65.0)
    maximum_table_angle_rad: float = np.deg2rad(90.0)
    face_normal_slew_rate_rad_s: float = np.deg2rad(75.0)
    maximum_ik_target_step_rad: float = 0.20

    def __post_init__(self) -> None:
        for name in (
            "translation_step_m",
            "position_tolerance_m",
            "face_normal_tolerance_rad",
            "direction_weight",
            "damping",
            "joint_step_limit_rad",
            "face_switch_hysteresis_m",
            "table_angle_step_rad",
            "face_normal_slew_rate_rad_s",
            "maximum_ik_target_step_rad",
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not np.isfinite(self.minimum_z_m) or not np.isfinite(self.maximum_z_m):
            raise ValueError("keyboard z bounds must be finite")
        if self.minimum_z_m >= self.maximum_z_m:
            raise ValueError("keyboard z bounds must be ordered")
        if (
            not np.isfinite(self.minimum_table_angle_rad)
            or not np.isfinite(self.maximum_table_angle_rad)
            or self.minimum_table_angle_rad >= self.maximum_table_angle_rad
        ):
            raise ValueError("keyboard table-angle bounds must be finite and ordered")
        if not (
            0.0 < self.minimum_table_angle_rad < np.pi
            and 0.0 < self.maximum_table_angle_rad < np.pi
        ):
            raise ValueError("keyboard table-angle bounds must remain in (0,pi)")
        if not isinstance(self.iterations, int) or isinstance(self.iterations, bool):
            raise ValueError("iterations must be an integer")
        if self.iterations <= 0:
            raise ValueError("iterations must be positive")


def slew_parallel_plane_normal(
    current_normal_world: np.ndarray,
    requested_normal_world: np.ndarray,
    maximum_step_rad: float,
) -> np.ndarray:
    """Slew a sign-invariant plane normal without an instantaneous face flip.

    A gripper plane is unchanged when its normal is negated.  The requested
    sign nearest to the currently accepted normal is selected first, then a
    spherical interpolation limits the angular change.  This prevents a
    nearest-face change at a box corner from becoming a one-frame wrist jump.
    """

    current = np.asarray(current_normal_world, dtype=np.float64).copy()
    requested = np.asarray(requested_normal_world, dtype=np.float64).copy()
    if current.shape != (3,) or requested.shape != (3,):
        raise ValueError("plane normals must be [3]")
    if not np.isfinite(current).all() or not np.isfinite(requested).all():
        raise ValueError("plane normals must be finite")
    if not np.isfinite(maximum_step_rad) or maximum_step_rad <= 0.0:
        raise ValueError("maximum_step_rad must be finite and positive")
    current_norm = float(np.linalg.norm(current))
    requested_norm = float(np.linalg.norm(requested))
    if current_norm < 1.0e-8 or requested_norm < 1.0e-8:
        raise ValueError("plane normals must be nonzero")
    current /= current_norm
    requested /= requested_norm
    if float(np.dot(current, requested)) < 0.0:
        requested = -requested
    cosine = float(np.clip(np.dot(current, requested), -1.0, 1.0))
    angle = float(np.arccos(cosine))
    if angle <= maximum_step_rad:
        return requested
    sine = float(np.sin(angle))
    if sine < 1.0e-8:
        blended = current + (maximum_step_rad / angle) * (requested - current)
        return blended / np.linalg.norm(blended)
    fraction = maximum_step_rad / angle
    blended = (
        np.sin((1.0 - fraction) * angle) / sine * current
        + np.sin(fraction * angle) / sine * requested
    )
    return blended / np.linalg.norm(blended)


@dataclass(frozen=True)
class KeyboardCommand:
    event: KeyboardEvent
    delta_world_m: np.ndarray
    delta_table_angle_rad: float = 0.0

    def __post_init__(self) -> None:
        delta = np.asarray(self.delta_world_m, dtype=np.float64)
        if delta.shape != (3,) or not np.isfinite(delta).all():
            raise ValueError("delta_world_m must be finite [3]")
        delta = delta.copy()
        delta.setflags(write=False)
        object.__setattr__(self, "delta_world_m", delta)
        if not np.isfinite(self.delta_table_angle_rad):
            raise ValueError("delta_table_angle_rad must be finite")


class KeyboardCommandBuffer:
    """Thread-safe key callback target for the MuJoCo UI thread."""

    def __init__(self, config: KeyboardCartesianConfig | None = None) -> None:
        self.config = config or KeyboardCartesianConfig()
        self._lock = threading.Lock()
        self._delta = np.zeros(3, dtype=np.float64)
        self._table_angle_delta = 0.0
        self._reset = False
        self._quit = False

    def on_key(self, keycode: int) -> None:
        from mujoco.glfw import glfw

        step = self.config.translation_step_m
        mapping = {
            glfw.KEY_W: np.asarray([step, 0.0, 0.0]),
            glfw.KEY_S: np.asarray([-step, 0.0, 0.0]),
            glfw.KEY_A: np.asarray([0.0, step, 0.0]),
            glfw.KEY_D: np.asarray([0.0, -step, 0.0]),
            glfw.KEY_UP: np.asarray([0.0, 0.0, step]),
            glfw.KEY_DOWN: np.asarray([0.0, 0.0, -step]),
        }
        with self._lock:
            if keycode in mapping:
                self._delta += mapping[keycode]
            elif keycode == glfw.KEY_E:
                self._table_angle_delta += self.config.table_angle_step_rad
            elif keycode == glfw.KEY_R:
                self._table_angle_delta -= self.config.table_angle_step_rad
            elif keycode == glfw.KEY_X:
                self._reset = True
            elif keycode in (glfw.KEY_Q, glfw.KEY_ESCAPE):
                self._quit = True

    def drain(self) -> KeyboardCommand:
        with self._lock:
            delta = self._delta.copy()
            table_angle_delta = self._table_angle_delta
            self._delta[:] = 0.0
            self._table_angle_delta = 0.0
            reset, quit_requested = self._reset, self._quit
            self._reset = False
        if quit_requested:
            event = KeyboardEvent.QUIT
        elif reset:
            event = KeyboardEvent.RESET
        elif np.any(delta) or table_angle_delta != 0.0:
            event = KeyboardEvent.MOVE
        else:
            event = KeyboardEvent.NONE
        return KeyboardCommand(event, delta, table_angle_delta)


@dataclass(frozen=True)
class BlockFaceTarget:
    axis_index: int
    outward_sign: int
    tool_face_normal_world: np.ndarray
    label: str

    def __post_init__(self) -> None:
        normal = np.asarray(self.tool_face_normal_world, dtype=np.float64)
        if normal.shape != (3,) or not np.isfinite(normal).all():
            raise ValueError("tool_face_normal_world must be finite [3]")
        norm = float(np.linalg.norm(normal))
        if norm < 1.0e-8:
            raise ValueError("tool face normal must be nonzero")
        normal = normal / norm
        normal.setflags(write=False)
        object.__setattr__(self, "tool_face_normal_world", normal)


class NearestBlockFaceTracker:
    """Select the nearest vertical block face with corner hysteresis."""

    def __init__(self, hysteresis_m: float = 0.008) -> None:
        self.hysteresis_m = float(hysteresis_m)
        self._selection: tuple[int, int] | None = None

    def reset(self) -> None:
        self._selection = None

    def selection_state(self) -> tuple[int, int] | None:
        """Return the last accepted face selection for transactional IK trials."""

        return self._selection

    def restore_selection_state(self, selection: tuple[int, int] | None) -> None:
        """Restore a face selection after an unaccepted candidate solve."""

        if selection is not None:
            axis, sign = selection
            if axis not in (0, 1, 2) or sign not in (-1, 1):
                raise ValueError("invalid block-face selection state")
        self._selection = selection

    def select(self, env: Any, tool_position_world: np.ndarray) -> BlockFaceTarget:
        tool = np.asarray(tool_position_world, dtype=np.float64)
        block_body = int(env._ids["block_body"])
        block_position = np.asarray(env.data.xpos[block_body], dtype=np.float64)
        block_rotation = np.asarray(env.data.xmat[block_body], dtype=np.float64).reshape(3, 3)
        local_offset = block_rotation.T @ (tool - block_position)
        object_shape = str(getattr(env, "manipulated_object_shape", "box"))
        scenario = getattr(env, "current_scenario", None)
        if object_shape in {"cylinder", "ellipsoid"} and scenario is not None:
            # Smooth objects do not have discrete X/Y faces.  Align the wide
            # gripper plane with the local surface normal (ellipse gradient),
            # then project it horizontally to preserve the table-angle control
            # contract.  A tipped object that exposes no useful side normal
            # falls through to the conservative discrete-axis selector below.
            x_radius, y_radius, _z_radius = scenario.object_half_extents_m
            outward_local = np.asarray(
                [
                    local_offset[0] / max(x_radius * x_radius, 1.0e-12),
                    local_offset[1] / max(y_radius * y_radius, 1.0e-12),
                    0.0,
                ],
                dtype=np.float64,
            )
            outward_norm = float(np.linalg.norm(outward_local))
            if outward_norm > 1.0e-8:
                desired = -(block_rotation @ (outward_local / outward_norm))
                desired[2] = 0.0
                desired_norm = float(np.linalg.norm(desired))
                if desired_norm >= 1.0e-8:
                    self._selection = None
                    return BlockFaceTarget(
                        0,
                        1,
                        desired / desired_norm,
                        f"{object_shape}_radial_surface",
                    )
        # A manipulated cube can tip onto any face.  Select among local axes
        # whose world-space face normal still has a useful horizontal
        # projection instead of assuming local X/Y remain vertical forever.
        horizontal_strength = np.linalg.norm(block_rotation[:2, :], axis=0)
        eligible_axes = np.flatnonzero(horizontal_strength >= 0.25)
        if eligible_axes.size == 0:
            raise RuntimeError("block has no side face with a horizontal component")
        candidate_axis = int(
            eligible_axes[
                np.argmax(np.abs(local_offset[eligible_axes]) * horizontal_strength[eligible_axes])
            ]
        )
        candidate_sign = 1 if local_offset[candidate_axis] >= 0.0 else -1
        if self._selection is not None:
            selected_axis, selected_sign = self._selection
            selected_eligible = bool(horizontal_strength[selected_axis] >= 0.25)
            selected_score = (
                selected_sign
                * float(local_offset[selected_axis])
                * float(horizontal_strength[selected_axis])
            )
            candidate_score = (
                abs(float(local_offset[candidate_axis]))
                * float(horizontal_strength[candidate_axis])
            )
            if selected_eligible and candidate_score <= selected_score + self.hysteresis_m:
                candidate_axis, candidate_sign = selected_axis, selected_sign
        self._selection = (candidate_axis, candidate_sign)
        outward_local = np.zeros(3, dtype=np.float64)
        outward_local[candidate_axis] = candidate_sign
        # The gripper wide-face normal points from the tool into the block,
        # opposite the block's outward normal on the nearest face.
        desired = -(block_rotation @ outward_local)
        desired[2] = 0.0
        norm = float(np.linalg.norm(desired))
        if norm < 1.0e-8:
            raise RuntimeError("nearest block side face has no horizontal normal")
        desired /= norm
        axis_name = ("X", "Y", "Z")[candidate_axis]
        sign_name = "+" if candidate_sign > 0 else "-"
        return BlockFaceTarget(
            candidate_axis,
            candidate_sign,
            desired,
            f"block_{sign_name}{axis_name}_face",
        )


def table_angle_face_normal(
    horizontal_face_normal_world: np.ndarray,
    table_angle_rad: float,
) -> np.ndarray:
    """Tilt a block-facing plane to an oriented angle relative to the desk.

    ``pi/2`` keeps the gripper wide plane vertical.  Larger/smaller angles
    tilt it to opposite sides while preserving the automatically selected
    horizontal azimuth toward the block face.
    """

    horizontal = np.asarray(horizontal_face_normal_world, dtype=np.float64).copy()
    if horizontal.shape != (3,) or not np.isfinite(horizontal).all():
        raise ValueError("horizontal_face_normal_world must be finite [3]")
    horizontal[2] = 0.0
    norm = float(np.linalg.norm(horizontal))
    if norm < 1.0e-8:
        raise ValueError("horizontal face normal must have a nonzero XY component")
    if not np.isfinite(table_angle_rad) or not 0.0 < table_angle_rad < np.pi:
        raise ValueError("table_angle_rad must be finite in (0,pi)")
    horizontal /= norm
    normal_tilt = (np.pi / 2.0) - float(table_angle_rad)
    normal = np.cos(normal_tilt) * horizontal
    normal[2] = np.sin(normal_tilt)
    normal /= np.linalg.norm(normal)
    return normal


def _direction_jacobian(normal: np.ndarray, rotation_jacobian: np.ndarray) -> np.ndarray:
    nx, ny, nz = normal
    skew = np.asarray(
        [[0.0, -nz, ny], [nz, 0.0, -nx], [-ny, nx, 0.0]],
        dtype=np.float64,
    )
    return -skew @ rotation_jacobian


class PositionFaceAlignedIK:
    """Five-DOF IK for XYZ plus a wide-face normal, leaving face roll free."""

    def __init__(self, env: Any, config: KeyboardCartesianConfig | None = None) -> None:
        self.env = env
        self.config = config or KeyboardCartesianConfig()
        self._scratch = mujoco.MjData(env.model)

    def current_pose(self, q: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        data = self.env.data
        if q is not None:
            self._scratch.qpos[:] = self.env.data.qpos
            self._scratch.qvel[:] = self.env.data.qvel
            self._scratch.qpos[:6] = np.asarray(q, dtype=np.float64)
            mujoco.mj_forward(self.env.model, self._scratch)
            data = self._scratch
        site = int(self.env._ids["tool_site"])
        position = np.asarray(data.site_xpos[site], dtype=np.float64).copy()
        normal = np.asarray(data.site_xmat[site], dtype=np.float64).reshape(3, 3)[:, 1].copy()
        return position, normal

    def _errors(
        self,
        q: np.ndarray,
        target_position: np.ndarray,
        target_normal: np.ndarray,
    ) -> tuple[float, float]:
        position, normal = self.current_pose(q)
        position_error = float(np.linalg.norm(position - target_position))
        normal_error = float(
            np.arccos(np.clip(float(np.dot(normal, target_normal)), -1.0, 1.0))
        )
        return position_error, normal_error

    def solve(
        self,
        current_q: np.ndarray,
        target_position_world: np.ndarray,
        target_face_normal_world: np.ndarray,
    ) -> IKResult:
        current = np.asarray(current_q, dtype=np.float64)
        target_position = np.asarray(target_position_world, dtype=np.float64)
        target_normal = np.asarray(target_face_normal_world, dtype=np.float64).copy()
        target_normal = target_normal / np.linalg.norm(target_normal)
        self._scratch.qpos[:] = self.env.data.qpos
        self._scratch.qvel[:] = self.env.data.qvel
        self._scratch.qpos[:6] = current
        site = int(self.env._ids["tool_site"])
        ranges = np.asarray(self.env.joint_ranges, dtype=np.float64)
        for _iteration in range(self.config.iterations):
            mujoco.mj_forward(self.env.model, self._scratch)
            position = np.asarray(self._scratch.site_xpos[site], dtype=np.float64)
            rotation = np.asarray(self._scratch.site_xmat[site], dtype=np.float64).reshape(3, 3)
            normal = rotation[:, 1]
            position_error = target_position - position
            direction_error = target_normal - normal
            if (
                np.linalg.norm(position_error) <= self.config.position_tolerance_m
                and np.arccos(np.clip(normal @ target_normal, -1.0, 1.0))
                <= self.config.face_normal_tolerance_rad
            ):
                break
            jac_position = np.zeros((3, self.env.model.nv), dtype=np.float64)
            jac_rotation = np.zeros((3, self.env.model.nv), dtype=np.float64)
            mujoco.mj_jacSite(
                self.env.model,
                self._scratch,
                jac_position,
                jac_rotation,
                site,
            )
            direction_jacobian = _direction_jacobian(
                normal,
                jac_rotation[:, :5],
            )
            weight = self.config.direction_weight
            jacobian = np.vstack(
                [jac_position[:, :5], weight * direction_jacobian]
            )
            error = np.concatenate([position_error, weight * direction_error])
            normal_matrix = jacobian @ jacobian.T
            delta = jacobian.T @ np.linalg.solve(
                normal_matrix + self.config.damping * np.eye(6),
                error,
            )
            delta = np.clip(
                delta,
                -self.config.joint_step_limit_rad,
                self.config.joint_step_limit_rad,
            )
            self._scratch.qpos[:5] = np.clip(
                self._scratch.qpos[:5] + delta,
                ranges[:5, 0],
                ranges[:5, 1],
            )
        candidate = np.asarray(self._scratch.qpos[:6], dtype=np.float64).copy()
        candidate[5] = float(self.env.tool_gripper_joint_position_rad)
        filtered, safety_reason = self.env._safety_filter(candidate)
        filtered = np.asarray(filtered, dtype=np.float64)
        position_error, normal_error = self._errors(
            filtered,
            target_position,
            target_normal,
        )
        return IKResult(
            target_joint_position_rad=filtered,
            position_error_m=position_error,
            orientation_error_rad=normal_error,
            converged=(
                position_error <= self.config.position_tolerance_m
                and normal_error <= self.config.face_normal_tolerance_rad
            ),
            joint_or_workspace_clipped=not np.allclose(filtered, candidate),
            safety_reason=str(safety_reason),
        )

    @staticmethod
    def _solution_score(result: IKResult, current_q: np.ndarray) -> tuple[float, float, float]:
        joint_distance = float(
            np.linalg.norm(result.target_joint_position_rad[:5] - current_q[:5])
        )
        task_error = result.position_error_m + 0.02 * result.orientation_error_rad
        if result.converged:
            return (0.0, joint_distance, task_error)
        return (
            1.0,
            task_error,
            joint_distance,
        )

    def solve_parallel_face(
        self,
        current_q: np.ndarray,
        target_position_world: np.ndarray,
        parallel_face_normal_world: np.ndarray,
        *,
        preferred_normal_world: np.ndarray | None = None,
    ) -> tuple[IKResult, np.ndarray]:
        """Choose either normal sign because parallel planes are sign-invariant."""

        axis = np.asarray(parallel_face_normal_world, dtype=np.float64).copy()
        axis /= np.linalg.norm(axis)
        candidates = [axis, -axis]
        if preferred_normal_world is not None:
            preferred = np.asarray(preferred_normal_world, dtype=np.float64)
            if np.dot(preferred, candidates[1]) > np.dot(preferred, candidates[0]):
                candidates.reverse()
        first = (
            self.solve(current_q, target_position_world, candidates[0]),
            candidates[0].copy(),
        )
        # Preserve the preferred sign whenever it remains feasible.  The
        # second solve is a recovery path, not a mandatory per-frame cost.
        if first[0].converged:
            return first
        second = (
            self.solve(current_q, target_position_world, candidates[1]),
            candidates[1].copy(),
        )
        return min(
            (first, second),
            key=lambda item: self._solution_score(item[0], np.asarray(current_q)),
        )

    def find_aligned_start(
        self,
        target_position_world: np.ndarray,
        target_face_normal_world: np.ndarray,
    ) -> IKResult:
        """Deterministic multi-start solve used only for simulation reset."""

        target_position = np.asarray(target_position_world, dtype=np.float64)
        target_normal = np.asarray(target_face_normal_world, dtype=np.float64).copy()
        target_normal /= np.linalg.norm(target_normal)
        base = np.asarray(self.env.data.qpos, dtype=np.float64).copy()
        lower = np.asarray(self.env.joint_ranges[:5, 0], dtype=np.float64) + 1.0e-7
        upper = np.asarray(self.env.joint_ranges[:5, 1], dtype=np.float64) - 1.0e-7
        rng = np.random.default_rng(20260824)
        starts = [np.clip(base[:5], lower, upper)]
        for wrist_roll in np.linspace(lower[4], upper[4], 7):
            start = starts[0].copy()
            start[4] = wrist_roll
            starts.append(start)
        starts.extend(rng.uniform(lower, upper, size=(12, 5)))

        def residual(q: np.ndarray) -> np.ndarray:
            probe = base.copy()
            probe[:5] = q
            position, normal = self.current_pose(probe[:6])
            return np.concatenate(
                [
                    (position - target_position) / self.config.position_tolerance_m,
                    (normal - target_normal) / self.config.face_normal_tolerance_rad,
                ]
            )

        solutions = [
            least_squares(
                residual,
                start,
                bounds=(lower, upper),
                max_nfev=500,
                ftol=1.0e-10,
                xtol=1.0e-10,
                gtol=1.0e-10,
            )
            for start in starts
        ]
        best = min(solutions, key=lambda result: float(np.linalg.norm(result.fun)))
        candidate = base[:6].copy()
        candidate[:5] = best.x
        candidate[5] = float(self.env.tool_gripper_joint_position_rad)
        filtered, safety_reason = self.env._safety_filter(candidate)
        filtered = np.asarray(filtered, dtype=np.float64)
        position_error, normal_error = self._errors(
            filtered,
            target_position,
            target_normal,
        )
        return IKResult(
            target_joint_position_rad=filtered,
            position_error_m=position_error,
            orientation_error_rad=normal_error,
            converged=(
                position_error <= self.config.position_tolerance_m
                and normal_error <= self.config.face_normal_tolerance_rad
            ),
            joint_or_workspace_clipped=not np.allclose(filtered, candidate),
            safety_reason=str(safety_reason),
        )

    def find_parallel_aligned_start(
        self,
        target_position_world: np.ndarray,
        parallel_face_normal_world: np.ndarray,
    ) -> tuple[IKResult, np.ndarray]:
        axis = np.asarray(parallel_face_normal_world, dtype=np.float64).copy()
        axis /= np.linalg.norm(axis)
        current_q = np.asarray(self.env.observation()["joint_state"][:6], dtype=np.float64)
        solved = [
            (self.find_aligned_start(target_position_world, normal), normal.copy())
            for normal in (axis, -axis)
        ]
        return min(
            solved,
            key=lambda item: self._solution_score(item[0], current_q),
        )


__all__ = [
    "KEYBOARD_CARTESIAN_RUNTIME_VERSION",
    "BlockFaceTarget",
    "KeyboardCartesianConfig",
    "KeyboardCommand",
    "KeyboardCommandBuffer",
    "KeyboardEvent",
    "NearestBlockFaceTracker",
    "PositionFaceAlignedIK",
    "slew_parallel_plane_normal",
    "table_angle_face_normal",
]
