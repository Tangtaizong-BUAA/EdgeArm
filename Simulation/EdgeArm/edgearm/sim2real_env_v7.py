"""Versioned contact-feasible successor to the synthetic V6 environment.

V6 is intentionally left unchanged because existing datasets and checkpoints
bind to its exact source and geometry.  V7 fixes a model-construction error in
which the rectangular work surface extended through the robot mounting area.
That overlap placed a shoulder collision mesh 8.8 mm inside the desk for every
reset and made the force-limited shoulder-pan actuator effectively immobile.

The corrected surface starts in front of the mounting volume, matching a robot
installed at the edge of a workbench.  Reset also raises the wrist pusher to a
collision-feasible approach pose.  All values remain synthetic engineering
priors: this module contains no physical calibration or hardware evidence.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
import time
from typing import Any

import mujoco
import numpy as np

from .contact_telemetry_v1 import (
    ContactTelemetryGeometry,
    PHYSICS_SUBSTEP_CONTACT_FORMAT,
    PhysicsSubstepContactRecorder,
    PushSideContactThresholds,
    push_side_contact_metrics,
)
from .sim2real_env import _JOINTS, _QueuedCommand
from .sim2real_env_v6 import (
    PARAMETER_SOURCE,
    REALISM_CLAIM_LEVEL,
    REALISTIC_DYNAMICS_PROFILE_VERSION,
    RealisticEdgeArmEnvV6,
    RealisticEnvV6Config,
)
from .production_env import ProductionMjcfBundleV1


CONTACT_FEASIBLE_DYNAMICS_PROFILE_VERSION = "edgearm-sim2real-dynamics-v7"
CONTACT_FEASIBLE_GEOMETRY_VERSION = "edgearm-workbench-edge-clearance-v1"
CONTACT_FEASIBLE_STATE_UPDATE = "mujoco_force_limited_actuator_with_workbench_mount_clearance"
CONTACT_FEASIBLE_RESET_SOURCE = "synthetic_task_aligned_privileged_pose_ik"
CONTACT_FEASIBLE_BOX_DISTANCE_VERSION = "edgearm-exact-obb-distance-v1"
CONTROLLER_PREFLIGHT_PROFILE_VERSION = "edgearm-controller-preflight-v1"
CONTROLLER_PREFLIGHT_REWRITE_REASON = "controller_preflight_dynamic_clearance_projection"
CONTROLLER_PREFLIGHT_BISECTION_ITERATIONS = 8
CONTROLLER_TERMINAL_VIABILITY_PROFILE_VERSION = "edgearm-controller-terminal-viability-v1"
_RESET_SEED_STRIDE = 1_000_003
_BOX_SIGNS = np.array(
    [
        [-1.0, -1.0, -1.0],
        [-1.0, -1.0, 1.0],
        [-1.0, 1.0, -1.0],
        [-1.0, 1.0, 1.0],
        [1.0, -1.0, -1.0],
        [1.0, -1.0, 1.0],
        [1.0, 1.0, -1.0],
        [1.0, 1.0, 1.0],
    ],
    dtype=np.float64,
)
_BOX_EDGE_INDEX_PAIRS = (
    (0, 1),
    (0, 2),
    (0, 4),
    (1, 3),
    (1, 5),
    (2, 3),
    (2, 6),
    (3, 7),
    (4, 5),
    (4, 6),
    (5, 7),
    (6, 7),
)


def _point_to_obb_distance(
    point: np.ndarray,
    center: np.ndarray,
    rotation: np.ndarray,
    half_extent: np.ndarray,
) -> float:
    local = rotation.T @ (point - center)
    return float(np.linalg.norm(local - np.clip(local, -half_extent, half_extent)))


def _segment_segment_distance(
    first_start: np.ndarray,
    first_end: np.ndarray,
    second_start: np.ndarray,
    second_end: np.ndarray,
) -> float:
    direction_first = first_end - first_start
    direction_second = second_end - second_start
    offset = first_start - second_start
    aa = float(np.dot(direction_first, direction_first))
    ab = float(np.dot(direction_first, direction_second))
    bb = float(np.dot(direction_second, direction_second))
    ao = float(np.dot(direction_first, offset))
    bo = float(np.dot(direction_second, offset))
    candidates: list[tuple[float, float]] = [
        (0.0, float(np.clip(bo / bb, 0.0, 1.0))),
        (1.0, float(np.clip((bo + ab) / bb, 0.0, 1.0))),
        (float(np.clip(-ao / aa, 0.0, 1.0)), 0.0),
        (float(np.clip((ab - ao) / aa, 0.0, 1.0)), 1.0),
    ]
    determinant = aa * bb - ab * ab
    if determinant > 1.0e-15:
        solution = np.linalg.solve(
            np.array([[aa, -ab], [-ab, bb]], dtype=np.float64),
            np.array([-ao, bo], dtype=np.float64),
        )
        if np.all((0.0 <= solution) & (solution <= 1.0)):
            candidates.append((float(solution[0]), float(solution[1])))
    return min(
        float(
            np.linalg.norm(offset + first_parameter * direction_first - second_parameter * direction_second)
        )
        for first_parameter, second_parameter in candidates
    )


def _box_box_signed_distance(
    first_center: np.ndarray,
    first_rotation: np.ndarray,
    first_half_extent: np.ndarray,
    second_center: np.ndarray,
    second_rotation: np.ndarray,
    second_half_extent: np.ndarray,
) -> float:
    """Return exact positive OBB clearance or negative SAT penetration."""

    first_center = np.asarray(first_center, dtype=np.float64)
    first_rotation = np.asarray(first_rotation, dtype=np.float64).reshape(3, 3)
    first_half_extent = np.asarray(first_half_extent, dtype=np.float64)
    second_center = np.asarray(second_center, dtype=np.float64)
    second_rotation = np.asarray(second_rotation, dtype=np.float64).reshape(3, 3)
    second_half_extent = np.asarray(second_half_extent, dtype=np.float64)
    center_offset = second_center - first_center
    axes = [first_rotation[:, index] for index in range(3)]
    axes.extend(second_rotation[:, index] for index in range(3))
    axes.extend(
        np.cross(first_rotation[:, first], second_rotation[:, second])
        for first in range(3)
        for second in range(3)
    )
    gaps: list[float] = []
    for raw_axis in axes:
        norm = float(np.linalg.norm(raw_axis))
        if norm <= 1.0e-12:
            continue
        axis = raw_axis / norm
        first_radius = float(np.dot(first_half_extent, np.abs(first_rotation.T @ axis)))
        second_radius = float(np.dot(second_half_extent, np.abs(second_rotation.T @ axis)))
        gaps.append(float(abs(np.dot(center_offset, axis)) - first_radius - second_radius))
    if gaps and max(gaps) <= 1.0e-12:
        return -float(min(-gap for gap in gaps))

    first_vertices = first_center + (_BOX_SIGNS * first_half_extent) @ first_rotation.T
    second_vertices = second_center + (_BOX_SIGNS * second_half_extent) @ second_rotation.T
    distances = [
        _point_to_obb_distance(
            vertex,
            second_center,
            second_rotation,
            second_half_extent,
        )
        for vertex in first_vertices
    ]
    distances.extend(
        _point_to_obb_distance(
            vertex,
            first_center,
            first_rotation,
            first_half_extent,
        )
        for vertex in second_vertices
    )
    for first_start, first_end in _BOX_EDGE_INDEX_PAIRS:
        for second_start, second_end in _BOX_EDGE_INDEX_PAIRS:
            distances.append(
                _segment_segment_distance(
                    first_vertices[first_start],
                    first_vertices[first_end],
                    second_vertices[second_start],
                    second_vertices[second_end],
                )
            )
    return float(min(distances))


def _box_box_separation_lower_bound(
    first_center: np.ndarray,
    first_rotation: np.ndarray,
    first_half_extent: np.ndarray,
    second_center: np.ndarray,
    second_rotation: np.ndarray,
    second_half_extent: np.ndarray,
) -> float:
    """Return the largest OBB SAT gap without the exact edge-distance pass.

    A positive result proves separation and lower-bounds Euclidean distance.
    A non-positive result is deliberately only an overlap/unknown signal.
    This is used as a cheap, conservative repair for generic mesh-distance
    false zeros; exact box/box callers continue to use
    :func:`_box_box_signed_distance`.
    """

    first_center = np.asarray(first_center, dtype=np.float64)
    first_rotation = np.asarray(first_rotation, dtype=np.float64).reshape(3, 3)
    first_half_extent = np.asarray(first_half_extent, dtype=np.float64)
    second_center = np.asarray(second_center, dtype=np.float64)
    second_rotation = np.asarray(second_rotation, dtype=np.float64).reshape(3, 3)
    second_half_extent = np.asarray(second_half_extent, dtype=np.float64)
    center_offset = second_center - first_center
    axes = [first_rotation[:, index] for index in range(3)]
    axes.extend(second_rotation[:, index] for index in range(3))
    axes.extend(
        np.cross(first_rotation[:, first], second_rotation[:, second])
        for first in range(3)
        for second in range(3)
    )
    maximum_gap = -np.inf
    for raw_axis in axes:
        norm = float(np.linalg.norm(raw_axis))
        if norm <= 1.0e-12:
            continue
        axis = raw_axis / norm
        first_radius = float(np.dot(first_half_extent, np.abs(first_rotation.T @ axis)))
        second_radius = float(np.dot(second_half_extent, np.abs(second_rotation.T @ axis)))
        maximum_gap = max(
            maximum_gap,
            float(abs(np.dot(center_offset, axis)) - first_radius - second_radius),
        )
    if not np.isfinite(maximum_gap):  # pragma: no cover - three box axes always survive
        raise RuntimeError("OBB separation lower-bound evaluation produced no valid axis")
    return float(maximum_gap)


@dataclass(frozen=True)
class RealisticEnvV7Config(RealisticEnvV6Config):
    """Synthetic V7 priors for a robot mounted at the workbench edge."""

    desk_front_edge_x_m: float = 0.100
    reset_tool_height_m: float = 0.122
    reset_tool_standoff_m: float = 0.070
    reset_pose_error_tolerance_m: float = 0.050
    reset_standoff_error_tolerance_m: float = 0.030
    reset_lateral_error_tolerance_m: float = 0.015
    reset_vertical_error_tolerance_m: float = 0.045
    reset_penetration_tolerance_m: float = 1.0e-5
    reset_tool_block_clearance_m: float = 0.002
    reset_obstacle_block_clearance_m: float = 0.005
    reset_support_margin_m: float = 0.005
    reset_block_settle_xy_tolerance_m: float = 0.001
    reset_settle_substeps: int = 10
    reset_max_resample_attempts: int = 32
    runtime_pusher_desk_clearance_m: float = 0.0
    runtime_pusher_desk_guard_reserve_m: float = 0.0002
    runtime_pusher_desk_guard_recovery_step_m: float = 0.0005
    runtime_pusher_desk_guard_finite_difference_rad: float = 0.0002
    runtime_pusher_desk_guard_path_samples: int = 5
    runtime_pusher_desk_guard_max_iterations: int = 4
    runtime_pusher_desk_hard_floor_m: float = 0.002
    runtime_pusher_desk_forecast_substeps: int = 5

    def __post_init__(self) -> None:
        super().__post_init__()
        for name in (
            "desk_front_edge_x_m",
            "reset_tool_height_m",
            "reset_tool_standoff_m",
            "reset_pose_error_tolerance_m",
            "reset_standoff_error_tolerance_m",
            "reset_lateral_error_tolerance_m",
            "reset_vertical_error_tolerance_m",
            "reset_penetration_tolerance_m",
            "reset_tool_block_clearance_m",
            "reset_obstacle_block_clearance_m",
            "reset_support_margin_m",
            "reset_block_settle_xy_tolerance_m",
            "runtime_pusher_desk_clearance_m",
            "runtime_pusher_desk_guard_reserve_m",
            "runtime_pusher_desk_guard_recovery_step_m",
            "runtime_pusher_desk_guard_finite_difference_rad",
            "runtime_pusher_desk_hard_floor_m",
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
        if not 0.075 <= self.desk_front_edge_x_m <= 0.140:
            raise ValueError("desk_front_edge_x_m must preserve the task workspace")
        if not 0.115 <= self.reset_tool_height_m <= 0.135:
            raise ValueError("reset_tool_height_m must be a collision-free approach height")
        if not 0.070 <= self.reset_tool_standoff_m <= 0.120:
            raise ValueError("reset_tool_standoff_m must preserve a visible approach segment")
        if not 0.010 <= self.reset_pose_error_tolerance_m <= 0.050:
            raise ValueError("reset_pose_error_tolerance_m must be in [0.01, 0.05]")
        if not 0.005 <= self.reset_standoff_error_tolerance_m <= 0.040:
            raise ValueError("reset_standoff_error_tolerance_m must be in [0.005, 0.04]")
        if not 0.002 <= self.reset_lateral_error_tolerance_m <= 0.025:
            raise ValueError("reset_lateral_error_tolerance_m must be in [0.002, 0.025]")
        if not 0.005 <= self.reset_vertical_error_tolerance_m <= 0.045:
            raise ValueError("reset_vertical_error_tolerance_m must be in [0.005, 0.045]")
        if self.reset_penetration_tolerance_m > 1.0e-3:
            raise ValueError("reset_penetration_tolerance_m cannot exceed 1 mm")
        if self.reset_tool_block_clearance_m > 0.050:
            raise ValueError("reset_tool_block_clearance_m cannot exceed 50 mm")
        if self.reset_obstacle_block_clearance_m > 0.050:
            raise ValueError("reset_obstacle_block_clearance_m cannot exceed 50 mm")
        if self.reset_support_margin_m > 0.050:
            raise ValueError("reset_support_margin_m cannot exceed 50 mm")
        if self.reset_block_settle_xy_tolerance_m > 0.010:
            raise ValueError("reset_block_settle_xy_tolerance_m cannot exceed 10 mm")
        if self.runtime_pusher_desk_clearance_m > 0.020:
            raise ValueError("runtime_pusher_desk_clearance_m cannot exceed 20 mm")
        if self.runtime_pusher_desk_hard_floor_m > 0.010:
            raise ValueError("runtime_pusher_desk_hard_floor_m cannot exceed 10 mm")
        if (
            self.runtime_pusher_desk_clearance_m > 0.0
            and self.runtime_pusher_desk_clearance_m < self.runtime_pusher_desk_hard_floor_m
        ):
            raise ValueError("runtime desk clearance trigger cannot be below its hard floor")
        if not 0.00005 <= self.runtime_pusher_desk_guard_reserve_m <= 0.002:
            raise ValueError("runtime_pusher_desk_guard_reserve_m must be in [0.05, 2] mm")
        if not 0.0001 <= self.runtime_pusher_desk_guard_recovery_step_m <= 0.005:
            raise ValueError("runtime_pusher_desk_guard_recovery_step_m must be in [0.1, 5] mm")
        if not 1.0e-5 <= self.runtime_pusher_desk_guard_finite_difference_rad <= 1.0e-3:
            raise ValueError("runtime_pusher_desk_guard_finite_difference_rad must be in [1e-5, 1e-3]")
        if not isinstance(self.reset_settle_substeps, int) or not 1 <= self.reset_settle_substeps <= 100:
            raise ValueError("reset_settle_substeps must be an integer in [1, 100]")
        if (
            not isinstance(self.reset_max_resample_attempts, int)
            or not 1 <= self.reset_max_resample_attempts <= 128
        ):
            raise ValueError("reset_max_resample_attempts must be an integer in [1, 128]")
        if (
            not isinstance(self.runtime_pusher_desk_guard_path_samples, int)
            or not 3 <= self.runtime_pusher_desk_guard_path_samples <= 17
        ):
            raise ValueError("runtime_pusher_desk_guard_path_samples must be in [3, 17]")
        if (
            not isinstance(self.runtime_pusher_desk_guard_max_iterations, int)
            or not 1 <= self.runtime_pusher_desk_guard_max_iterations <= 20
        ):
            raise ValueError("runtime_pusher_desk_guard_max_iterations must be in [1, 20]")
        if (
            not isinstance(self.runtime_pusher_desk_forecast_substeps, int)
            or not 1 <= self.runtime_pusher_desk_forecast_substeps <= 20
        ):
            raise ValueError("runtime_pusher_desk_forecast_substeps must be in [1, 20]")


class RealisticEdgeArmEnvV7(RealisticEdgeArmEnvV6):
    """V6 dynamics with corrected workbench mounting and reset geometry."""

    profile_version = CONTACT_FEASIBLE_DYNAMICS_PROFILE_VERSION

    def __init__(
        self,
        config: RealisticEnvV7Config | None = None,
        seed: int = 0,
        *,
        model_scene_path: Path | None = None,
        model_scene_bundle: ProductionMjcfBundleV1 | None = None,
    ):
        if config is not None and not isinstance(config, RealisticEnvV7Config):
            raise TypeError("RealisticEdgeArmEnvV7 requires RealisticEnvV7Config")
        self.contact_feasible_config = config or RealisticEnvV7Config()
        self._desk_original_x_bounds = np.zeros(2, dtype=np.float64)
        self._desk_corrected_x_bounds = np.zeros(2, dtype=np.float64)
        self._reset_collision_audit: dict[str, Any] = {}
        self._obstacle_placement_audit: dict[str, Any] = {}
        self._physics_substep_contact_v1: dict[str, object] | None = None
        self._runtime_pusher_desk_guard_v1: dict[str, object] = {}
        self._controller_preflight_v1: dict[str, object] = {}
        self._terminal_viability_gate_evaluated_decisions = 0
        self._terminal_viability_gate_skipped_decisions = 0
        self._terminal_viability_gate_reason_counts = {
            "selected_search_index_positive": 0,
            "bisection_lambda_below_one": 0,
            "forecast_minimum_within_reserve": 0,
        }
        self._terminal_viability_exact_evaluation_count = 0
        self._terminal_viability_exact_evaluation_wall_seconds = 0.0
        self._clearance_guard_scratch: mujoco.MjData | None = None
        self._clearance_guard_dynamics_scratch: mujoco.MjData | None = None
        self._runtime_pusher_desk_safety_stop_requested = False
        super().__init__(
            self.contact_feasible_config,
            seed=seed,
            model_scene_path=model_scene_path,
            model_scene_bundle=model_scene_bundle,
        )
        self._clearance_guard_scratch = mujoco.MjData(self.model)
        self._clearance_guard_dynamics_scratch = mujoco.MjData(self.model)

    def _build_model(self) -> mujoco.MjModel:
        model = super()._build_model()
        desk = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "edgearm_desk")
        if desk < 0:  # pragma: no cover - guarded by the production model builder
            raise ValueError("V7 model is missing edgearm_desk")
        original_left = float(model.geom_pos[desk, 0] - model.geom_size[desk, 0])
        original_right = float(model.geom_pos[desk, 0] + model.geom_size[desk, 0])
        front = float(self.contact_feasible_config.desk_front_edge_x_m)
        if front >= original_right:
            raise ValueError("desk front edge would remove the complete work surface")
        model.geom_pos[desk, 0] = 0.5 * (front + original_right)
        model.geom_size[desk, 0] = 0.5 * (original_right - front)
        self._desk_original_x_bounds = np.array([original_left, original_right], dtype=np.float64)
        self._desk_corrected_x_bounds = np.array([front, original_right], dtype=np.float64)
        return model

    def _set_obstacle(self, block: np.ndarray, target: np.ndarray) -> None:
        """Keep an enabled obstacle on-path but outside the initial block."""

        super()._set_obstacle(block, target)
        if not self.obstacle_enabled:
            self._obstacle_placement_audit = {
                "enabled": False,
                "support_mode": "disabled_collision_and_hidden",
            }
            return
        direct = np.asarray(target, dtype=np.float64) - np.asarray(block, dtype=np.float64)
        distance = float(np.linalg.norm(direct))
        direction = direct / max(distance, 1e-12)
        normal = np.array([-direction[1], direction[0]], dtype=np.float64)
        original_center = np.asarray(block, dtype=np.float64) + 0.48 * direct
        lateral_offset = float(np.dot(self.obstacle_xy - original_center, normal))
        obstacle_half = self.model.geom_size[self._ids["obstacle_geom"], :2]
        block_half = self.model.geom_size[self._ids["block_geom"], :2]
        separating_radius = float(
            np.dot(np.abs(direction), obstacle_half + block_half)
            + self.contact_feasible_config.reset_obstacle_block_clearance_m
        )
        minimum_fraction = separating_radius / max(distance, 1e-12)
        selected_fraction = float(max(0.48, minimum_fraction))
        if selected_fraction >= 0.90:
            raise RuntimeError("V7 obstacle placement cannot separate the initial block")
        self.obstacle_xy = (
            np.asarray(block, dtype=np.float64) + selected_fraction * direct + normal * lateral_offset
        )
        self.model.geom_pos[self._ids["obstacle_geom"], :2] = self.obstacle_xy
        self._obstacle_placement_audit = {
            "enabled": True,
            "support_mode": "world_fixed_static_geom",
            "original_path_fraction": 0.48,
            "selected_path_fraction": selected_fraction,
            "minimum_separating_path_fraction": minimum_fraction,
            "lateral_offset_m": lateral_offset,
            "required_block_clearance_m": (self.contact_feasible_config.reset_obstacle_block_clearance_m),
        }

    def reset(
        self,
        seed: int | None = None,
        *,
        obstacle: bool | None = None,
        stress: bool = False,
    ) -> dict[str, np.ndarray]:
        requested_seed = int(seed) if seed is not None else int(self.rng.integers(0, np.iinfo(np.int64).max))
        rejected_attempts: list[dict[str, Any]] = []
        target_xyz = np.zeros(3, dtype=np.float64)
        accepted_seed = requested_seed
        for attempt in range(self.contact_feasible_config.reset_max_resample_attempts):
            accepted_seed = int((requested_seed + attempt * _RESET_SEED_STRIDE) % np.iinfo(np.int64).max)
            super().reset(seed=accepted_seed, obstacle=obstacle, stress=stress)
            block = self.block_xy()
            direction = self._unit(self.target_xy - block)
            start = block - direction * self.contact_feasible_config.reset_tool_standoff_m
            target_xyz = np.array(
                [start[0], start[1], self.contact_feasible_config.reset_tool_height_m],
                dtype=np.float64,
            )
            joint_position = self.solve_pose_ik(
                target_xyz,
                direction,
                initial=self.data.qpos[:_JOINTS].copy(),
            )
            self._install_reset_joint_state(joint_position)
            block_before_settle = self.block_xy()
            for _ in range(self.contact_feasible_config.reset_settle_substeps):
                mujoco.mj_step(self.model, self.data)
            block_settle_displacement = float(np.linalg.norm(self.block_xy() - block_before_settle))
            self._install_reset_joint_state(self.data.qpos[:_JOINTS].copy())
            self.data.time = 0.0
            mujoco.mj_forward(self.model, self.data)
            self.last_distance = self.distance_to_target()
            self._reset_collision_audit = self._audit_reset_contacts(
                target_xyz,
                direction,
                block_settle_displacement_m=block_settle_displacement,
            )
            self._reset_collision_audit.update(
                {
                    "requested_seed": requested_seed,
                    "accepted_candidate_seed": accepted_seed,
                    "resample_attempt_index": attempt,
                }
            )
            if self._reset_collision_audit["reset_valid"]:
                break
            rejected_attempts.append(
                {
                    "candidate_seed": accepted_seed,
                    "failure_reasons": list(self._reset_collision_audit["reset_failure_reasons"]),
                }
            )
        else:
            raise RuntimeError(
                f"V7 reset exhausted deterministic clean-state resampling: {self._reset_collision_audit}"
            )

        profile = {
            "profile_version": self.profile_version,
            "base_dynamics_profile_version": REALISTIC_DYNAMICS_PROFILE_VERSION,
            "effective_profile_version": self.profile_version,
            "geometry_version": CONTACT_FEASIBLE_GEOMETRY_VERSION,
            "claim_level": REALISM_CLAIM_LEVEL,
            "parameter_source": PARAMETER_SOURCE,
            "physical_samples": 0,
            "physical_trials": 0,
            "physically_calibrated": False,
            "physical_hardware_connected": False,
            "state_update": CONTACT_FEASIBLE_STATE_UPDATE,
            "reset_state_source": CONTACT_FEASIBLE_RESET_SOURCE,
            "task_aligned_privileged_reset": True,
            "deployment_reset_equivalent": False,
            "reset_privileged_state_used": ["block_pose", "target_pose"],
            "requested_seed": requested_seed,
            "accepted_candidate_seed": accepted_seed,
            "reset_attempt_count": len(rejected_attempts) + 1,
            "rejected_reset_attempts": rejected_attempts,
            "desk_original_x_bounds_m": self._desk_original_x_bounds.tolist(),
            "desk_corrected_x_bounds_m": self._desk_corrected_x_bounds.tolist(),
            "reset_tool_target_xyz_m": target_xyz.tolist(),
            "obstacle_placement_audit": self._obstacle_placement_audit,
            "reset_collision_audit": self._reset_collision_audit,
        }
        self.episode_domain["realism_v7"] = profile
        self._physics_substep_contact_v1 = None
        self._runtime_pusher_desk_guard_v1 = {}
        self._controller_preflight_v1 = {}
        self._terminal_viability_gate_evaluated_decisions = 0
        self._terminal_viability_gate_skipped_decisions = 0
        self._terminal_viability_gate_reason_counts = {
            "selected_search_index_positive": 0,
            "bisection_lambda_below_one": 0,
            "forecast_minimum_within_reserve": 0,
        }
        self._terminal_viability_exact_evaluation_count = 0
        self._terminal_viability_exact_evaluation_wall_seconds = 0.0
        return self.observation()

    def step(self, action: np.ndarray) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        self._runtime_pusher_desk_safety_stop_requested = False
        self._controller_preflight_v1 = {}
        observation, reward, terminated, truncated, info = super().step(action)
        realism_v6 = dict(info["realism_v6"])
        realism_v6["profile_version"] = REALISTIC_DYNAMICS_PROFILE_VERSION
        realism_v6["effective_profile_version"] = self.profile_version
        info["realism_v6"] = realism_v6
        realism_v7 = dict(self.episode_domain["realism_v7"])
        realism_v7.update(realism_v6)
        realism_v7.update(
            {
                "profile_version": self.profile_version,
                "base_dynamics_profile_version": REALISTIC_DYNAMICS_PROFILE_VERSION,
                "effective_profile_version": self.profile_version,
                "geometry_version": CONTACT_FEASIBLE_GEOMETRY_VERSION,
                "reset_collision_audit": self._reset_collision_audit,
                "physics_substep_contact_format": PHYSICS_SUBSTEP_CONTACT_FORMAT,
                "runtime_pusher_desk_guard_enabled": bool(
                    self.contact_feasible_config.runtime_pusher_desk_clearance_m > 0.0
                ),
            }
        )
        info["realism_v7"] = realism_v7
        if info.get("safety_stop") == "estop":
            # No physics transition is executed while the latched e-stop is
            # active, so there is deliberately no fabricated substep trace.
            info["physics_transition_executed"] = False
            return observation, reward, True, False, info
        if self._physics_substep_contact_v1 is None:  # pragma: no cover
            raise RuntimeError("V7 step completed without physics-substep contact telemetry")
        info["physics_substep_contact_v1"] = self._physics_substep_contact_v1
        info["runtime_pusher_desk_guard_v1"] = self._runtime_pusher_desk_guard_v1
        feedback = dict(self._last_command_feedback_v1)
        guard = self._runtime_pusher_desk_guard_v1
        trace = self._physics_substep_contact_v1
        applied_safety_changed = bool(
            np.any(feedback["applied_command_safety_target_changed_mask"])
            or feedback["applied_command_post_submission_rewrite_count"] > 0
            or guard["physics_applied_target_changed_any"]
        )
        feedback.update(
            {
                "runtime_guard_profile": str(guard["format"]),
                "runtime_guard_action_modified": bool(guard["physics_applied_target_changed_any"]),
                "runtime_guard_static_projection_count": int(guard["projection_count"]),
                "runtime_guard_dynamic_projection_count": int(guard["dynamic_projection_count"]),
                "runtime_guard_queue_rewrite_event_count": int(guard["queue_rewrite_event_count"]),
                "runtime_guard_controller_preflight_modified": bool(guard["controller_preflight_changed"]),
                "applied_action_was_safety_modified": applied_safety_changed,
                "controller_preflight_profile": str(guard["controller_preflight_profile"]),
                "controller_preflight_applied_command_id": int(
                    guard["controller_preflight_applied_command_id"]
                ),
                "controller_preflight_input_joint_target": np.asarray(
                    guard["controller_preflight_input_joint_target"],
                    dtype=np.float32,
                ).copy(),
                "controller_preflight_output_joint_target": np.asarray(
                    guard["controller_preflight_output_joint_target"],
                    dtype=np.float32,
                ).copy(),
                "controller_preflight_target_changed_mask": np.asarray(
                    guard["controller_preflight_target_changed_mask"],
                    dtype=np.uint8,
                ).copy(),
                "controller_preflight_previewed_servo_endpoint_before_joint_target": np.asarray(
                    guard["controller_preflight_previewed_servo_endpoint_before_joint_target"],
                    dtype=np.float32,
                ).copy(),
                "controller_preflight_previewed_servo_endpoint_after_joint_target": np.asarray(
                    guard["controller_preflight_previewed_servo_endpoint_after_joint_target"],
                    dtype=np.float32,
                ).copy(),
                "controller_preflight_changed": bool(guard["controller_preflight_changed"]),
                "controller_preflight_controller_search_attempt_count": int(
                    guard["controller_preflight_controller_search_attempt_count"]
                ),
                "controller_preflight_controller_search_fallback": str(
                    guard["controller_preflight_controller_search_fallback"]
                ),
                "controller_preflight_controller_bisection_iteration_count": int(
                    guard["controller_preflight_controller_bisection_iteration_count"]
                ),
                "controller_preflight_controller_bisection_selected_lambda": float(
                    guard["controller_preflight_controller_bisection_selected_lambda"]
                ),
                "controller_preflight_dynamic_forecast_feasible": bool(
                    guard["controller_preflight_dynamic_forecast_feasible"]
                ),
                "controller_preflight_dynamic_forecast_fallback": str(
                    guard["controller_preflight_dynamic_forecast_fallback"]
                ),
                "controller_preflight_infeasible_stop_suppressed": bool(
                    guard["controller_preflight_infeasible_stop_suppressed"]
                ),
                "controller_terminal_viability": deepcopy(guard["controller_terminal_viability"]),
                "controller_terminal_viability_gate_triggered": bool(
                    guard["controller_terminal_viability_gate_triggered"]
                ),
                "controller_terminal_viability_evaluated": bool(
                    guard["controller_terminal_viability_evaluated"]
                ),
                "controller_terminal_viability_rewrite_performed": bool(
                    guard["controller_terminal_viability_rewrite_performed"]
                ),
                "controller_terminal_viability_episode_exact_evaluation_per_decision_density": float(
                    guard["controller_terminal_viability_episode_exact_evaluation_per_decision_density"]
                ),
                "servo_endpoint_joint_target": np.asarray(
                    guard["servo_endpoint_joint_target"], dtype=np.float32
                ).copy(),
                "delayed_controller_joint_target": np.asarray(
                    guard["delayed_controller_joint_target"], dtype=np.float32
                ).copy(),
                "physics_applied_joint_target": np.asarray(
                    guard["physics_applied_joint_target"], dtype=np.float32
                ).copy(),
                "physics_applied_joint_target_by_substep": np.asarray(
                    guard["physics_applied_joint_target_by_substep"],
                    dtype=np.float32,
                ).copy(),
                "physics_applied_target_changed_mask_by_substep": np.asarray(
                    guard["physics_applied_target_changed_from_servo_endpoint_mask_by_substep"],
                    dtype=np.uint8,
                ).copy(),
                "effect_contact_any": bool(trace["contact_any"]),
                "effect_valid_push_side_contact_any": bool(trace["valid_push_side_contact_any"]),
                "effect_geometric_push_side_contact_any": bool(
                    np.any(np.asarray(trace["geometric_push_side_contact_count"]) > 0)
                ),
            }
        )
        self._last_command_feedback_v1 = feedback
        if self._runtime_pusher_desk_safety_stop_requested:
            # This is a physical safety failure, not an administrative horizon.
            # Mark it terminal so PPO never bootstraps through an unsafe state.
            terminated = True
            truncated = False
            info["success"] = False
            info["terminated"] = True
            info["truncated"] = False
            info["terminal_failure"] = True
            info["terminal_reason"] = "safety_stop:runtime_pusher_desk_dynamic_forecast_infeasible"
            info["safety_clipped"] = True
            info["safety_reason"] = self._combine_reasons(
                str(info.get("safety_reason", "")),
                "runtime_pusher_desk_dynamic_forecast_infeasible",
            )
            info["safety_stop"] = "runtime_pusher_desk_dynamic_forecast_infeasible"
        return observation, reward, terminated, truncated, info

    def _preflight_applied_delayed_command(
        self,
        applied_command: _QueuedCommand,
    ) -> str:
        """Project the exact popped command before stateful servo shaping.

        Each candidate is passed through the same execution-time safety filter
        as :meth:`Sim2RealEdgeArmEnv.step`, then through an exact, state/RNG
        restoring preview of the stateful servo shaper.  The previewed endpoint
        is the actuator target used by the conservative dynamics forecast.
        An infeasible *preflight* forecast may rewrite the command but cannot
        request an episode stop.  The unchanged per-physics-substep barrier
        remains the fail-closed authority during live integration.
        """

        input_target = np.asarray(applied_command, dtype=np.float64).copy()
        forecast_horizon = (
            self.config.physics_substeps + self.contact_feasible_config.runtime_pusher_desk_forecast_substeps
        )
        candidate, execution_filter_reason = self._safety_filter(input_target)
        attempts: list[dict[str, object]] = []
        seen_candidates: list[np.ndarray] = []
        maximum_attempts = self.contact_feasible_config.runtime_pusher_desk_guard_max_iterations + 1
        for search_index in range(maximum_attempts):
            candidate = np.asarray(candidate, dtype=np.float64).copy()
            servo_endpoint = self._preview_servo_endpoint(candidate)
            projected_endpoint, event = self._project_runtime_pusher_desk_dynamic_target(
                self.data.qpos[:_JOINTS],
                servo_endpoint,
                substep_index=-1,
                request_safety_stop=False,
                forecast_substeps=forecast_horizon,
            )
            raw_feasible = bool(event["dynamic_forecast_feasible"] and not event["dynamic_target_changed"])
            attempts.append(
                {
                    "search_index": int(search_index),
                    "controller_candidate_joint_target": candidate.copy(),
                    "previewed_servo_endpoint_joint_target": servo_endpoint.copy(),
                    "raw_forecast_feasible": raw_feasible,
                    "raw_forecast_minimum_m": float(event["dynamic_forecast_minimum_before_m"]),
                    "raw_forecast_endpoint_m": float(event["dynamic_forecast_endpoint_before_m"]),
                    "actuator_projection_feasible": bool(event["dynamic_forecast_feasible"]),
                    "actuator_projection_target_changed": bool(event["dynamic_target_changed"]),
                    "actuator_projection_fallback": str(event["dynamic_forecast_fallback"]),
                    "actuator_projection_iterations": int(event["dynamic_forecast_iterations"]),
                    "event": event,
                }
            )
            if raw_feasible:
                break
            next_candidate, _next_filter_reason = self._safety_filter(projected_endpoint)
            if any(
                np.allclose(
                    next_candidate,
                    previous,
                    rtol=0.0,
                    atol=1.0e-10,
                )
                for previous in (*seen_candidates, candidate)
            ):
                break
            seen_candidates.append(candidate.copy())
            candidate = next_candidate

        safe_attempts = [attempt for attempt in attempts if bool(attempt["raw_forecast_feasible"])]
        if safe_attempts:
            selected = safe_attempts[0]
            controller_search_fallback = (
                "raw_preview_already_safe" if int(selected["search_index"]) == 0 else "projected_preview_safe"
            )
        else:
            selected = max(
                attempts,
                key=lambda attempt: (
                    float(attempt["raw_forecast_minimum_m"]),
                    float(attempt["raw_forecast_endpoint_m"]),
                ),
            )
            controller_search_fallback = "least_dangerous_preview_no_safe_candidate"
        bisection_attempts: list[dict[str, object]] = []
        bisection_selected_lambda = 1.0
        if safe_attempts and int(selected["search_index"]) > 0:
            unsafe_controller = np.asarray(
                attempts[0]["controller_candidate_joint_target"],
                dtype=np.float64,
            )
            safe_controller = np.asarray(
                selected["controller_candidate_joint_target"],
                dtype=np.float64,
            )
            lower_lambda = 0.0
            upper_lambda = 1.0
            for bisection_index in range(CONTROLLER_PREFLIGHT_BISECTION_ITERATIONS):
                candidate_lambda = 0.5 * (lower_lambda + upper_lambda)
                midpoint = unsafe_controller + candidate_lambda * (safe_controller - unsafe_controller)
                midpoint, _midpoint_filter_reason = self._safety_filter(midpoint)
                midpoint_servo = self._preview_servo_endpoint(midpoint)
                midpoint_forecast = self._pusher_desk_dynamic_clearance_forecast(
                    midpoint_servo,
                    substeps=forecast_horizon,
                )
                midpoint_feasible = self._runtime_pusher_desk_dynamic_forecast_acceptable(
                    float(attempts[0]["event"]["dynamic_current_clearance_m"]),
                    midpoint_forecast,
                )
                bisection_attempts.append(
                    {
                        "bisection_index": int(bisection_index),
                        "lambda": float(candidate_lambda),
                        "controller_candidate_joint_target": midpoint.copy(),
                        "previewed_servo_endpoint_joint_target": (midpoint_servo.copy()),
                        "raw_forecast_feasible": bool(midpoint_feasible),
                        "raw_forecast_minimum_m": float(np.min(midpoint_forecast)),
                        "raw_forecast_endpoint_m": float(midpoint_forecast[-1]),
                    }
                )
                if midpoint_feasible:
                    upper_lambda = candidate_lambda
                else:
                    lower_lambda = candidate_lambda

            bisection_selected_lambda = upper_lambda
            bisected_controller = unsafe_controller + upper_lambda * (safe_controller - unsafe_controller)
            bisected_controller, _bisected_filter_reason = self._safety_filter(bisected_controller)
            bisected_servo = self._preview_servo_endpoint(bisected_controller)
            bisected_projected, bisected_event = self._project_runtime_pusher_desk_dynamic_target(
                self.data.qpos[:_JOINTS],
                bisected_servo,
                substep_index=-1,
                request_safety_stop=False,
                forecast_substeps=forecast_horizon,
            )
            bisected_raw_feasible = bool(
                bisected_event["dynamic_forecast_feasible"] and not bisected_event["dynamic_target_changed"]
            )
            if bisected_raw_feasible:
                selected = {
                    "search_index": int(selected["search_index"]),
                    "controller_candidate_joint_target": (bisected_controller.copy()),
                    "previewed_servo_endpoint_joint_target": (bisected_servo.copy()),
                    "raw_forecast_feasible": True,
                    "raw_forecast_minimum_m": float(bisected_event["dynamic_forecast_minimum_before_m"]),
                    "raw_forecast_endpoint_m": float(bisected_event["dynamic_forecast_endpoint_before_m"]),
                    "actuator_projection_feasible": True,
                    "actuator_projection_target_changed": False,
                    "actuator_projection_fallback": str(bisected_event["dynamic_forecast_fallback"]),
                    "actuator_projection_iterations": int(bisected_event["dynamic_forecast_iterations"]),
                    "event": bisected_event,
                }
                controller_search_fallback = "bisected_minimum_safe_preview"
            else:  # pragma: no cover - guards non-monotone numerical edges
                del bisected_projected
                controller_search_fallback = "projected_preview_safe_bisection_validation_fallback"
        output_target = np.asarray(selected["controller_candidate_joint_target"], dtype=np.float64).copy()
        servo_endpoint_before = np.asarray(
            attempts[0]["previewed_servo_endpoint_joint_target"],
            dtype=np.float64,
        ).copy()
        servo_endpoint_after = np.asarray(
            selected["previewed_servo_endpoint_joint_target"],
            dtype=np.float64,
        ).copy()
        event = dict(selected["event"])
        feasible = bool(selected["raw_forecast_feasible"])
        filtered_output, output_filter_reason = self._safety_filter(output_target)
        if not np.array_equal(filtered_output, output_target):  # pragma: no cover
            raise RuntimeError("controller preflight selected a non-idempotent execution target")
        changed_mask = (
            ~np.isclose(
                input_target,
                output_target,
                rtol=0.0,
                atol=1.0e-12,
            )
        ).astype(np.uint8)
        changed = bool(np.any(changed_mask))
        self._controller_preflight_v1 = {
            "format": CONTROLLER_PREFLIGHT_PROFILE_VERSION,
            "claim_level": "L1_SYNTHETIC_HARDWARE_INSPIRED",
            "physical_samples": 0,
            "applied_command_id": int(applied_command.command_id),
            "input_joint_target": input_target.astype(np.float32),
            "output_joint_target": np.asarray(output_target, dtype=np.float32).copy(),
            "target_changed_mask": changed_mask.astype(np.uint8),
            "changed": changed,
            "execution_filter_reason_before_search": str(execution_filter_reason),
            "execution_filter_reason_selected": str(output_filter_reason),
            "execution_filter_selected_target_idempotent": True,
            "previewed_servo_endpoint_before_joint_target": (servo_endpoint_before.astype(np.float32)),
            "previewed_servo_endpoint_after_joint_target": (servo_endpoint_after.astype(np.float32)),
            "servo_preview_state_and_rng_restored": True,
            "controller_search_attempt_count": len(attempts),
            "controller_search_selected_index": int(selected["search_index"]),
            "controller_search_fallback": controller_search_fallback,
            "controller_bisection_iteration_count": len(bisection_attempts),
            "controller_bisection_selected_lambda": float(bisection_selected_lambda),
            "controller_bisection_attempts": [
                {
                    key: (
                        np.asarray(value, dtype=np.float32).copy() if isinstance(value, np.ndarray) else value
                    )
                    for key, value in attempt.items()
                }
                for attempt in bisection_attempts
            ],
            "controller_search_attempts": [
                {
                    key: (
                        np.asarray(value, dtype=np.float32).copy() if isinstance(value, np.ndarray) else value
                    )
                    for key, value in attempt.items()
                    if key != "event"
                }
                for attempt in attempts
            ],
            "dynamic_current_clearance_m": float(event["dynamic_current_clearance_m"]),
            "dynamic_forecast_triggered": bool(
                any(bool(dict(attempt["event"])["dynamic_forecast_triggered"]) for attempt in attempts)
            ),
            "dynamic_forecast_feasible": feasible,
            "dynamic_forecast_minimum_before_m": float(
                dict(attempts[0]["event"])["dynamic_forecast_minimum_before_m"]
            ),
            "dynamic_forecast_minimum_after_m": float(selected["raw_forecast_minimum_m"]),
            "dynamic_forecast_endpoint_before_m": float(
                dict(attempts[0]["event"])["dynamic_forecast_endpoint_before_m"]
            ),
            "dynamic_forecast_endpoint_after_m": float(selected["raw_forecast_endpoint_m"]),
            "dynamic_forecast_iterations": int(
                sum(int(attempt["actuator_projection_iterations"]) for attempt in attempts)
            ),
            "dynamic_forecast_substeps": int(event["dynamic_forecast_substeps"]),
            "dynamic_forecast_fallback": controller_search_fallback,
            "selected_actuator_projection_fallback": str(selected["actuator_projection_fallback"]),
            "infeasible_stop_suppressed": bool(not feasible),
            "servo_shaping_pending": True,
            "safety_stop_authority": "servo_shaped_physics_substep_guard",
        }
        output_target = self._refine_controller_preflight_terminal_viability(output_target)
        changed_mask = applied_command.rewrite_target(
            output_target,
            reason=CONTROLLER_PREFLIGHT_REWRITE_REASON,
        )
        changed = bool(np.any(changed_mask))
        self._controller_preflight_v1["output_joint_target"] = output_target.astype(np.float32)
        self._controller_preflight_v1["target_changed_mask"] = changed_mask.astype(np.uint8)
        self._controller_preflight_v1["changed"] = changed
        return CONTROLLER_PREFLIGHT_REWRITE_REASON if changed else ""

    def _refine_controller_preflight_terminal_viability(
        self,
        production_target: np.ndarray,
    ) -> np.ndarray:
        """Refine a near-boundary preflight target only when no brake remains.

        The cheap gate uses telemetry already computed by the ordinary
        controller preflight.  Only a projected search result (selected index
        above zero or bisection lambda below one) triggers the expensive exact
        preview.  Near-reserve forecasts remain diagnostic because the live
        substep guard is the safety authority; making reserve proximity a gate
        by itself destroys controller liveness without changing the command.
        A triggered gate runs an exact, state-restoring eight-substep preview
        and then checks source-neutral brake targets from the predicted
        next-decision state.  A production target that already retains any
        brake is never changed.  If refinement is required, its upper endpoint
        must be an existing *raw-forecast-feasible* preflight search attempt
        with ``search_index > 0``.  Consequently an unsafe attempt rescued only
        by the live substep guard can never masquerade as the safe bisection
        endpoint.
        """

        production_target = np.asarray(production_target, dtype=np.float64)
        preflight = self._controller_preflight_v1
        margin = float(self.contact_feasible_config.runtime_pusher_desk_clearance_m)
        reserve = float(self.contact_feasible_config.runtime_pusher_desk_guard_reserve_m)
        gate_reasons = {
            "selected_search_index_positive": bool(int(preflight["controller_search_selected_index"]) > 0),
            "bisection_lambda_below_one": bool(
                float(preflight["controller_bisection_selected_lambda"]) < 1.0 - 1.0e-12
            ),
            "forecast_minimum_within_reserve": bool(
                float(preflight["dynamic_forecast_minimum_after_m"]) <= margin + reserve
            ),
        }
        gate_triggered = bool(
            margin > 0.0
            and (gate_reasons["selected_search_index_positive"] or gate_reasons["bisection_lambda_below_one"])
        )
        if margin > 0.0:
            for name, triggered in gate_reasons.items():
                self._terminal_viability_gate_reason_counts[name] += int(triggered)
        evaluation_count_before = self._terminal_viability_exact_evaluation_count
        evaluation_wall_before = self._terminal_viability_exact_evaluation_wall_seconds
        telemetry: dict[str, object] = {
            "format": CONTROLLER_TERMINAL_VIABILITY_PROFILE_VERSION,
            "claim_level": "L1_SYNTHETIC_HARDWARE_INSPIRED",
            "physical_samples": 0,
            "enabled": bool(margin > 0.0),
            "source_neutral": True,
            "gate_threshold_m": margin + reserve,
            "gate_reasons": gate_reasons,
            "gate_triggered": gate_triggered,
            "evaluated": False,
            "current_interval_criterion": (
                "exact_8_substep_canonical_and_zero_live_dynamic_infeasible_and_no_stop"
            ),
            "terminal_criterion": (
                "at_least_one_source_neutral_brake_canonical_feasible_for_both_5_and_13_substeps"
            ),
            "rewrite_required": False,
            "rewrite_performed": False,
            "rewrite_fallback": "gate_not_triggered",
            "production_target": production_target.astype(np.float32),
            "selected_target": production_target.astype(np.float32),
            "selected_lambda": float(preflight["controller_bisection_selected_lambda"]),
            "production_selected_lambda": float(preflight["controller_bisection_selected_lambda"]),
            "safe_upper_search_index": -1,
            "safe_upper_raw_forecast_feasible": False,
            "unsafe_attempt0_rejected_as_safe_upper": True,
            "bisection_attempts": [],
        }
        if not gate_triggered:
            self._terminal_viability_gate_skipped_decisions += 1
            self._finalize_terminal_viability_density(
                telemetry,
                evaluation_count_before,
                evaluation_wall_before,
            )
            preflight["terminal_viability"] = telemetry
            return production_target.copy()

        self._terminal_viability_gate_evaluated_decisions += 1
        telemetry["evaluated"] = True
        telemetry["rewrite_fallback"] = "terminal_brake_retained"
        production_evaluation = self._preview_interval_terminal_viability(production_target)
        telemetry["production_evaluation"] = self._compact_terminal_viability_evaluation(
            production_evaluation
        )
        rewrite_required = bool(
            production_evaluation["current_interval_safe"]
            and not dict(production_evaluation["terminal"])["strict_viable"]
        )
        telemetry["rewrite_required"] = rewrite_required
        selected_target = production_target.copy()
        selected_evaluation = production_evaluation
        selected_lambda = float(preflight["controller_bisection_selected_lambda"])

        if rewrite_required:
            attempts = list(preflight["controller_search_attempts"])
            safe_upper_attempt = next(
                (
                    attempt
                    for attempt in attempts
                    if int(attempt["search_index"]) > 0 and bool(attempt["raw_forecast_feasible"])
                ),
                None,
            )
            if safe_upper_attempt is None:
                telemetry["rewrite_fallback"] = "no_raw_feasible_safe_upper_beyond_attempt0"
            else:
                safe_upper_index = int(safe_upper_attempt["search_index"])
                telemetry["safe_upper_search_index"] = safe_upper_index
                telemetry["safe_upper_raw_forecast_feasible"] = True
                unsafe = np.asarray(
                    attempts[0]["controller_candidate_joint_target"],
                    dtype=np.float64,
                )
                safe_upper = np.asarray(
                    safe_upper_attempt["controller_candidate_joint_target"],
                    dtype=np.float64,
                )
                safe_upper_evaluation = self._preview_interval_terminal_viability(safe_upper)
                telemetry["safe_upper_evaluation"] = self._compact_terminal_viability_evaluation(
                    safe_upper_evaluation
                )
                upper_admissible = self._terminal_viability_candidate_admissible(safe_upper_evaluation)
                telemetry["safe_upper_two_stage_admissible"] = upper_admissible
                segment_target = unsafe + selected_lambda * (safe_upper - unsafe)
                production_on_segment = bool(
                    np.allclose(
                        segment_target,
                        production_target,
                        rtol=0.0,
                        atol=1.0e-6,
                    )
                )
                telemetry["production_target_on_safe_segment"] = production_on_segment
                if not upper_admissible:
                    telemetry["rewrite_fallback"] = "raw_feasible_safe_upper_terminal_inadmissible"
                elif not production_on_segment:
                    telemetry["rewrite_fallback"] = "production_target_not_on_raw_safe_segment"
                else:
                    lower_lambda = selected_lambda
                    upper_lambda = 1.0
                    upper_evaluation = safe_upper_evaluation
                    bisection_attempts: list[dict[str, object]] = []
                    for bisection_index in range(CONTROLLER_PREFLIGHT_BISECTION_ITERATIONS):
                        candidate_lambda = 0.5 * (lower_lambda + upper_lambda)
                        candidate = unsafe + candidate_lambda * (safe_upper - unsafe)
                        candidate, _filter_reason = self._safety_filter(candidate)
                        evaluation = self._preview_interval_terminal_viability(candidate)
                        admissible = self._terminal_viability_candidate_admissible(evaluation)
                        bisection_attempts.append(
                            {
                                "bisection_index": bisection_index,
                                "lambda": candidate_lambda,
                                "admissible": admissible,
                                "current_interval_minimum_m": float(evaluation["current_interval_minimum_m"]),
                                "terminal_current_clearance_m": float(
                                    dict(evaluation["terminal"])["current_clearance_m"]
                                ),
                                "terminal_viable_candidate_names": list(
                                    dict(evaluation["terminal"])["strict_viable_candidate_names"]
                                ),
                            }
                        )
                        if admissible:
                            upper_lambda = candidate_lambda
                            upper_evaluation = evaluation
                        else:
                            lower_lambda = candidate_lambda
                    selected_lambda = upper_lambda
                    selected_target = unsafe + upper_lambda * (safe_upper - unsafe)
                    selected_target, _filter_reason = self._safety_filter(selected_target)
                    if not np.allclose(
                        np.asarray(
                            upper_evaluation["controller_target"],
                            dtype=np.float64,
                        ),
                        selected_target,
                        rtol=0.0,
                        atol=1.0e-12,
                    ):
                        upper_evaluation = self._preview_interval_terminal_viability(selected_target)
                    telemetry["bisection_attempts"] = bisection_attempts
                    if self._terminal_viability_candidate_admissible(upper_evaluation):
                        selected_evaluation = upper_evaluation
                        telemetry["rewrite_performed"] = bool(
                            not np.allclose(
                                selected_target,
                                production_target,
                                rtol=0.0,
                                atol=1.0e-12,
                            )
                        )
                        telemetry["rewrite_fallback"] = "bisected_minimum_terminal_brake_viable_preview"
                    else:  # pragma: no cover - validation edge
                        selected_target = production_target.copy()
                        selected_evaluation = production_evaluation
                        selected_lambda = float(preflight["controller_bisection_selected_lambda"])
                        telemetry["rewrite_fallback"] = "terminal_bisection_validation_fallback"

        telemetry["selected_target"] = selected_target.astype(np.float32)
        telemetry["selected_lambda"] = selected_lambda
        telemetry["selected_evaluation"] = self._compact_terminal_viability_evaluation(selected_evaluation)
        telemetry["target_delta_from_production_linf_rad"] = float(
            np.max(np.abs(selected_target - production_target))
        )
        # Every exact preview restores the controller-preflight dictionary by
        # value.  Rebind after the final preview so telemetry and any refined
        # target update the restored live dictionary rather than the detached
        # pre-preview object referenced at method entry.
        preflight = self._controller_preflight_v1
        if bool(telemetry["rewrite_performed"]):
            selected_preview = np.asarray(selected_evaluation["servo_endpoint"], dtype=np.float64)
            original_input = np.asarray(preflight["input_joint_target"], dtype=np.float64)
            preflight["output_joint_target"] = selected_target.astype(np.float32)
            preflight["target_changed_mask"] = (
                ~np.isclose(
                    original_input,
                    selected_target,
                    rtol=0.0,
                    atol=1.0e-12,
                )
            ).astype(np.uint8)
            preflight["changed"] = bool(np.any(preflight["target_changed_mask"]))
            preflight["previewed_servo_endpoint_after_joint_target"] = selected_preview.astype(np.float32)
            preflight["controller_search_fallback"] = "bisected_minimum_terminal_brake_viable_preview"
            preflight["controller_bisection_selected_lambda"] = selected_lambda
            preflight["dynamic_forecast_feasible"] = True
            preflight["dynamic_forecast_minimum_after_m"] = float(
                selected_evaluation["current_forecast_minimum_m"]
            )
            preflight["dynamic_forecast_endpoint_after_m"] = float(
                selected_evaluation["current_forecast_endpoint_m"]
            )
            preflight["dynamic_forecast_fallback"] = "bisected_minimum_terminal_brake_viable_preview"
            preflight["infeasible_stop_suppressed"] = False

        self._finalize_terminal_viability_density(
            telemetry,
            evaluation_count_before,
            evaluation_wall_before,
        )
        preflight["terminal_viability"] = telemetry
        return selected_target.copy()

    def _finalize_terminal_viability_density(
        self,
        telemetry: dict[str, object],
        evaluation_count_before: int,
        evaluation_wall_before: float,
    ) -> None:
        decisions = max(int(self.step_count) + 1, 1)
        decision_evaluations = self._terminal_viability_exact_evaluation_count - evaluation_count_before
        decision_wall = self._terminal_viability_exact_evaluation_wall_seconds - evaluation_wall_before
        telemetry.update(
            {
                "decision_exact_evaluation_count": decision_evaluations,
                "decision_exact_evaluation_wall_seconds": decision_wall,
                "episode_gate_evaluated_decisions": (self._terminal_viability_gate_evaluated_decisions),
                "episode_gate_skipped_decisions": (self._terminal_viability_gate_skipped_decisions),
                "episode_gate_reason_counts": dict(self._terminal_viability_gate_reason_counts),
                "episode_exact_evaluation_count": (self._terminal_viability_exact_evaluation_count),
                "episode_exact_evaluation_wall_seconds": (
                    self._terminal_viability_exact_evaluation_wall_seconds
                ),
                "episode_gate_evaluated_decision_density": (
                    self._terminal_viability_gate_evaluated_decisions / decisions
                ),
                "episode_exact_evaluation_per_decision_density": (
                    self._terminal_viability_exact_evaluation_count / decisions
                ),
                "episode_exact_evaluation_mean_wall_seconds": (
                    self._terminal_viability_exact_evaluation_wall_seconds
                    / max(
                        self._terminal_viability_exact_evaluation_count,
                        1,
                    )
                ),
            }
        )

    def _terminal_viability_candidate_admissible(
        self,
        evaluation: dict[str, object],
    ) -> bool:
        return bool(
            evaluation["current_interval_safe"]
            and evaluation["current_forecast_feasible"]
            and dict(evaluation["terminal"])["strict_viable"]
        )

    def _preview_interval_terminal_viability(
        self,
        controller_target: np.ndarray,
    ) -> dict[str, object]:
        """Preview one complete decision and its terminal brake set exactly."""

        snapshot = self._snapshot_terminal_viability_state()
        started = time.perf_counter()
        try:
            target, filter_reason = self._safety_filter(controller_target)
            current_clearance = self._pusher_desk_distance_at_joint_position(self.data.qpos[:_JOINTS])
            servo_endpoint = self._preview_servo_endpoint(target)
            forecast_horizon = (
                self.config.physics_substeps
                + self.contact_feasible_config.runtime_pusher_desk_forecast_substeps
            )
            current_forecast = self._pusher_desk_dynamic_clearance_forecast(
                servo_endpoint,
                substeps=forecast_horizon,
            )
            current_forecast_feasible = self._runtime_pusher_desk_dynamic_forecast_acceptable(
                current_clearance,
                current_forecast,
            )
            start = self.data.qpos[:_JOINTS].copy()
            self._runtime_pusher_desk_safety_stop_requested = False
            (
                actual_servo_endpoint,
                physical_velocity,
                _runtime_safety,
                _tracking_noise,
                _runtime_changed_mask,
            ) = self._servo_step(target)
            self._advance_physics(
                start,
                actual_servo_endpoint,
                physical_velocity,
                target,
            )
            self._last_actual_velocity = (self.data.qpos[:_JOINTS] - start) / max(self.control_dt, 1.0e-9)
            # Base Sim2RealEdgeArmEnv.step consumes the same encoder draws after
            # physics.  Advancing them here makes the terminal servo preview
            # exact for the *next* decision; the complete RNG state is restored
            # in ``finally``.
            self._refresh_encoder_noise()
            trace = deepcopy(self._physics_substep_contact_v1)
            guard = deepcopy(self._runtime_pusher_desk_guard_v1)
            if trace is None:  # pragma: no cover - _advance_physics contract
                raise RuntimeError("terminal preview produced no physics trace")
            clearance = np.asarray(trace["tool_desk_signed_distance_m"], dtype=np.float64)
            exact_canonical = self._runtime_pusher_desk_dynamic_forecast_acceptable(
                current_clearance,
                clearance,
            )
            current_safe = bool(
                exact_canonical
                and int(guard["dynamic_infeasible_count"]) == 0
                and not bool(guard["safety_stop_requested"])
            )
            terminal = self._terminal_source_neutral_brake_set(target)
            return {
                "controller_target": target.copy(),
                "static_filter_reason": str(filter_reason),
                "servo_endpoint": servo_endpoint.copy(),
                "servo_endpoint_matches_exact_step": bool(
                    np.array_equal(servo_endpoint, actual_servo_endpoint)
                ),
                "current_clearance_m": float(current_clearance),
                "current_forecast_minimum_m": float(np.min(current_forecast)),
                "current_forecast_endpoint_m": float(current_forecast[-1]),
                "current_forecast_feasible": bool(current_forecast_feasible),
                "current_interval_clearance_by_substep_m": clearance.copy(),
                "current_interval_minimum_m": float(np.min(clearance)),
                "current_interval_endpoint_m": float(clearance[-1]),
                "current_interval_canonical_feasible": bool(exact_canonical),
                "current_interval_dynamic_infeasible_count": int(guard["dynamic_infeasible_count"]),
                "current_interval_safety_stop": bool(guard["safety_stop_requested"]),
                "current_interval_safe": current_safe,
                "predicted_post_qvel_rad_s": self.data.qvel[:_JOINTS].copy(),
                "predicted_post_servo_velocity_rad_s": (self._servo_velocity.copy()),
                "terminal": terminal,
            }
        finally:
            self._restore_terminal_viability_state(snapshot)
            self._terminal_viability_exact_evaluation_count += 1
            self._terminal_viability_exact_evaluation_wall_seconds += time.perf_counter() - started

    def _terminal_source_neutral_brake_set(
        self,
        controller_target: np.ndarray,
    ) -> dict[str, object]:
        post_q = self.data.qpos[:_JOINTS].copy()
        post_qvel = self.data.qvel[:_JOINTS].copy()
        max_step = (
            np.asarray(
                self.sim2real_config.max_joint_velocity,
                dtype=np.float64,
            )
            * self.control_dt
        )
        source_targets = {
            "post_state_hold": post_q,
            "controller_target_continuation": np.asarray(controller_target, dtype=np.float64),
            "post_actuator_ctrl_continuation": self.data.ctrl[:_JOINTS].copy(),
            "qvel_opposing_velocity_bounded": post_q
            - np.clip(post_qvel * self.control_dt, -max_step, max_step),
        }
        current_clearance = self._pusher_desk_distance_at_joint_position(post_q)
        short_horizon = int(self.contact_feasible_config.runtime_pusher_desk_forecast_substeps)
        long_horizon = int(self.config.physics_substeps + short_horizon)
        candidates: list[dict[str, object]] = []
        endpoints: list[tuple[str, np.ndarray, np.ndarray]] = []
        for name, raw_target in source_targets.items():
            target, filter_reason = self._safety_filter(raw_target)
            endpoint = self._preview_servo_endpoint(target)
            forecast13 = self._pusher_desk_dynamic_clearance_forecast(
                endpoint,
                substeps=long_horizon,
            )
            forecast5 = forecast13[:short_horizon]
            feasible5 = self._runtime_pusher_desk_dynamic_forecast_acceptable(
                current_clearance,
                forecast5,
            )
            feasible13 = self._runtime_pusher_desk_dynamic_forecast_acceptable(
                current_clearance,
                forecast13,
            )
            candidates.append(
                {
                    "name": name,
                    "candidate_stage": ("next_controller_target_through_exact_servo_preview"),
                    "target": target.copy(),
                    "static_filter_reason": str(filter_reason),
                    "servo_endpoint": endpoint.copy(),
                    "minimum5_m": float(np.min(forecast5)),
                    "endpoint5_m": float(forecast5[-1]),
                    "feasible5": bool(feasible5),
                    "minimum13_m": float(np.min(forecast13)),
                    "endpoint13_m": float(forecast13[-1]),
                    "feasible13": bool(feasible13),
                    "strict_feasible_both": bool(feasible5 and feasible13),
                }
            )
            endpoints.append((name, endpoint, forecast13))
        viable = [candidate for candidate in candidates if bool(candidate["strict_feasible_both"])]
        if not viable:
            source_name, requested, _forecast = max(
                endpoints,
                key=lambda item: (
                    float(np.min(item[2])),
                    float(item[2][-1]),
                ),
            )
            projected, event = self._project_runtime_pusher_desk_dynamic_target(
                post_q,
                requested,
                substep_index=-1,
                request_safety_stop=False,
                forecast_substeps=long_horizon,
            )
            forecast13 = self._pusher_desk_dynamic_clearance_forecast(
                projected,
                substeps=long_horizon,
            )
            forecast5 = forecast13[:short_horizon]
            feasible5 = self._runtime_pusher_desk_dynamic_forecast_acceptable(
                current_clearance,
                forecast5,
            )
            feasible13 = self._runtime_pusher_desk_dynamic_forecast_acceptable(
                current_clearance,
                forecast13,
            )
            gradient_candidate = {
                "name": "existing_dynamic_gradient_best_brake",
                "derived_from": source_name,
                "candidate_stage": "next_live_actuator_target",
                "target": projected.copy(),
                "minimum5_m": float(np.min(forecast5)),
                "endpoint5_m": float(forecast5[-1]),
                "feasible5": bool(feasible5),
                "minimum13_m": float(np.min(forecast13)),
                "endpoint13_m": float(forecast13[-1]),
                "feasible13": bool(feasible13),
                "strict_feasible_both": bool(feasible5 and feasible13),
                "projection_fallback": str(event["dynamic_forecast_fallback"]),
                "projection_reported_feasible": bool(event["dynamic_forecast_feasible"]),
            }
            candidates.append(gradient_candidate)
            if bool(gradient_candidate["strict_feasible_both"]):
                viable.append(gradient_candidate)
        return {
            "current_clearance_m": float(current_clearance),
            "short_horizon": short_horizon,
            "long_horizon": long_horizon,
            "candidates": candidates,
            "strict_viable": bool(viable),
            "strict_viable_candidate_names": [str(candidate["name"]) for candidate in viable],
        }

    @staticmethod
    def _compact_terminal_viability_evaluation(
        evaluation: dict[str, object],
    ) -> dict[str, object]:
        terminal = dict(evaluation["terminal"])
        return {
            "controller_target": np.asarray(evaluation["controller_target"], dtype=np.float32).copy(),
            "servo_endpoint": np.asarray(evaluation["servo_endpoint"], dtype=np.float32).copy(),
            "servo_endpoint_matches_exact_step": bool(evaluation["servo_endpoint_matches_exact_step"]),
            "current_clearance_m": float(evaluation["current_clearance_m"]),
            "current_forecast_minimum_m": float(evaluation["current_forecast_minimum_m"]),
            "current_forecast_endpoint_m": float(evaluation["current_forecast_endpoint_m"]),
            "current_forecast_feasible": bool(evaluation["current_forecast_feasible"]),
            "current_interval_clearance_by_substep_m": np.asarray(
                evaluation["current_interval_clearance_by_substep_m"],
                dtype=np.float32,
            ).copy(),
            "current_interval_minimum_m": float(evaluation["current_interval_minimum_m"]),
            "current_interval_endpoint_m": float(evaluation["current_interval_endpoint_m"]),
            "current_interval_canonical_feasible": bool(evaluation["current_interval_canonical_feasible"]),
            "current_interval_dynamic_infeasible_count": int(
                evaluation["current_interval_dynamic_infeasible_count"]
            ),
            "current_interval_safety_stop": bool(evaluation["current_interval_safety_stop"]),
            "current_interval_safe": bool(evaluation["current_interval_safe"]),
            "predicted_post_qvel_rad_s": np.asarray(
                evaluation["predicted_post_qvel_rad_s"],
                dtype=np.float32,
            ).copy(),
            "predicted_post_servo_velocity_rad_s": np.asarray(
                evaluation["predicted_post_servo_velocity_rad_s"],
                dtype=np.float32,
            ).copy(),
            "terminal_current_clearance_m": float(terminal["current_clearance_m"]),
            "terminal_viable": bool(terminal["strict_viable"]),
            "terminal_viable_candidate_names": list(terminal["strict_viable_candidate_names"]),
            "terminal_candidates": [
                {
                    key: value
                    for key, value in dict(candidate).items()
                    if key
                    in {
                        "name",
                        "derived_from",
                        "candidate_stage",
                        "minimum5_m",
                        "endpoint5_m",
                        "feasible5",
                        "minimum13_m",
                        "endpoint13_m",
                        "feasible13",
                        "strict_feasible_both",
                        "projection_fallback",
                    }
                }
                for candidate in list(terminal["candidates"])
            ],
        }

    def _snapshot_terminal_viability_state(self) -> dict[str, object]:
        data = mujoco.MjData(self.model)
        mujoco.mj_copyData(data, self.model, self.data)
        camera = self._ids["cameras"]["wrist"]
        return {
            "data": data,
            "episode_rng": deepcopy(self._episode_rng.bit_generator.state),
            "servo_velocity": self._servo_velocity.copy(),
            "backlash_remaining": self._backlash_remaining.copy(),
            "last_motor_direction": self._last_motor_direction.copy(),
            "encoder_position_noise": self._encoder_position_noise.copy(),
            "encoder_velocity_noise": self._encoder_velocity_noise.copy(),
            "last_actual_velocity": self._last_actual_velocity.copy(),
            "last_actuator_force": self._last_actuator_force.copy(),
            "loaded_voltage_v": float(self._loaded_voltage_v),
            "motor_temperature_c": self._motor_temperature_c.copy(),
            "actuator_forcerange": self.model.actuator_forcerange.copy(),
            "camera_pos": self.model.cam_pos[camera].copy(),
            "camera_quat": self.model.cam_quat[camera].copy(),
            "command_queue": deepcopy(self._command_queue),
            "physics_trace": deepcopy(self._physics_substep_contact_v1),
            "runtime_guard": deepcopy(self._runtime_pusher_desk_guard_v1),
            "controller_preflight": deepcopy(self._controller_preflight_v1),
            "safety_stop": bool(self._runtime_pusher_desk_safety_stop_requested),
        }

    def _restore_terminal_viability_state(
        self,
        snapshot: dict[str, object],
    ) -> None:
        camera = self._ids["cameras"]["wrist"]
        self.model.actuator_forcerange[:] = np.asarray(snapshot["actuator_forcerange"], dtype=np.float64)
        self.model.cam_pos[camera] = np.asarray(snapshot["camera_pos"], dtype=np.float64)
        self.model.cam_quat[camera] = np.asarray(snapshot["camera_quat"], dtype=np.float64)
        mujoco.mj_copyData(
            self.data,
            self.model,
            snapshot["data"],
        )
        self._episode_rng.bit_generator.state = deepcopy(snapshot["episode_rng"])
        self._servo_velocity[:] = np.asarray(snapshot["servo_velocity"], dtype=np.float64)
        self._backlash_remaining[:] = np.asarray(snapshot["backlash_remaining"], dtype=np.float64)
        self._last_motor_direction[:] = np.asarray(snapshot["last_motor_direction"], dtype=np.int8)
        self._encoder_position_noise[:] = np.asarray(snapshot["encoder_position_noise"], dtype=np.float64)
        self._encoder_velocity_noise[:] = np.asarray(snapshot["encoder_velocity_noise"], dtype=np.float64)
        self._last_actual_velocity[:] = np.asarray(snapshot["last_actual_velocity"], dtype=np.float64)
        self._last_actuator_force[:] = np.asarray(snapshot["last_actuator_force"], dtype=np.float64)
        self._loaded_voltage_v = float(snapshot["loaded_voltage_v"])
        self._motor_temperature_c[:] = np.asarray(snapshot["motor_temperature_c"], dtype=np.float64)
        self._command_queue = deepcopy(snapshot["command_queue"])
        self._physics_substep_contact_v1 = deepcopy(snapshot["physics_trace"])
        self._runtime_pusher_desk_guard_v1 = deepcopy(snapshot["runtime_guard"])
        self._controller_preflight_v1 = deepcopy(snapshot["controller_preflight"])
        self._runtime_pusher_desk_safety_stop_requested = bool(snapshot["safety_stop"])

    def _preview_servo_endpoint(self, target: np.ndarray) -> np.ndarray:
        """Return the exact next servo endpoint without consuming live state/RNG."""

        target = np.asarray(target, dtype=np.float64)
        if target.shape != (_JOINTS,) or not np.all(np.isfinite(target)):
            raise ValueError("servo preview requires a finite six-joint target")
        servo_velocity = self._servo_velocity.copy()
        backlash_remaining = self._backlash_remaining.copy()
        last_motor_direction = self._last_motor_direction.copy()
        rng_state = deepcopy(self._episode_rng.bit_generator.state)
        try:
            endpoint = self._servo_step(target)[0]
        finally:
            self._servo_velocity[:] = servo_velocity
            self._backlash_remaining[:] = backlash_remaining
            self._last_motor_direction[:] = last_motor_direction
            self._episode_rng.bit_generator.state = rng_state
        return np.asarray(endpoint, dtype=np.float64).copy()

    def contact_telemetry_geometry(self) -> ContactTelemetryGeometry:
        """Return the resolved geom contract shared by telemetry and teachers."""

        return ContactTelemetryGeometry(
            tool_geom=self._ids["tool_geom"],
            tool_contact_geoms=frozenset(self._ids["tool_contact_geoms"]),
            block_geom=self._ids["block_geom"],
            desk_geom=self._desk_geom,
            obstacle_geom=self._ids["obstacle_geom"],
            camera_housing_geom=self._camera_housing_geom,
            robot_geoms=frozenset(self._robot_geoms),
            tool_contact_geom_order=tuple(self._ids["tool_contact_geoms"]),
            tool_contact_geom_roles=tuple(self._ids["tool_contact_geom_roles"]),
        )

    def current_push_side_contact_metrics(
        self,
        intended_push_direction_xy: np.ndarray | None = None,
        thresholds: PushSideContactThresholds | None = None,
    ) -> dict[str, object]:
        """Classify the current simulator contact state with the trace's semantics."""

        direction = (
            self._unit(self.target_xy - self.block_xy())
            if intended_push_direction_xy is None
            else np.asarray(intended_push_direction_xy, dtype=np.float64)
        )
        return push_side_contact_metrics(
            self.model,
            self.data,
            self.contact_telemetry_geometry(),
            direction,
            thresholds,
        )

    def _advance_physics(
        self,
        start: np.ndarray,
        end: np.ndarray,
        velocity: np.ndarray,
        controller_target: np.ndarray,
    ) -> None:
        """Advance the unchanged V6 plant while sampling every physics substep."""

        del start, velocity
        # V9's historical contract exposes two distal contact parts.  Newer
        # stock-gripper contact profiles may expose a larger, ordered subset
        # of the same immutable 96-part CoACD union and let the existing
        # geometric side-contact gate decide which *observed* manifolds are
        # admissible.  Full safety telemetry depends on the 96-part identity,
        # not on the number of conditionally contact-capable parts.
        safety_geom_ids = tuple(self._tool_safety_geom_ids())
        contact_geom_ids = tuple(int(value) for value in self._ids.get("tool_contact_geoms", ()))
        full_stock_safety = bool(
            len(safety_geom_ids) == 96
            and len(contact_geom_ids) >= 2
            and len(set(contact_geom_ids)) == len(contact_geom_ids)
            and set(contact_geom_ids).issubset(safety_geom_ids)
        )
        recorder = PhysicsSubstepContactRecorder(
            self.model,
            self.contact_telemetry_geometry(),
            physics_substeps=self.config.physics_substeps,
            decision_time_seconds=float(self.data.time),
            obstacle_enabled=self.obstacle_enabled,
            penetration_tolerance_m=(self.contact_feasible_config.reset_penetration_tolerance_m),
            tool_desk_signed_distance=lambda data: (
                self._minimum_tool_safety_signed_distance_for_data(
                    self._desk_geom,
                    data,
                )
            ),
            tool_safety_desk_signed_distances=(
                (
                    lambda data: self._tool_safety_signed_distances_for_data(
                        self._desk_geom,
                        data,
                    )
                )
                if full_stock_safety
                else None
            ),
            tool_safety_block_signed_distances=(
                (
                    lambda data: self._tool_safety_signed_distances_for_data(
                        self._ids["block_geom"],
                        data,
                    )
                )
                if full_stock_safety
                else None
            ),
            tool_safety_geom_ids=(self._tool_safety_geom_ids() if full_stock_safety else ()),
            initial_block_xy=self.block_xy(),
            initial_tool_xyz=self.tool_xyz(),
            intended_push_direction_xy=self._unit(self.target_xy - self.block_xy()),
        )
        guard_events: list[dict[str, object]] = []
        queue_rewrite_events: list[dict[str, object]] = []
        guard_input_targets: list[np.ndarray] = []
        static_projected_targets: list[np.ndarray] = []
        physics_applied_targets: list[np.ndarray] = []
        servo_endpoint = np.asarray(end, dtype=np.float64).copy()
        delayed_controller_target = np.asarray(controller_target, dtype=np.float64).copy()
        applied_target = np.asarray(end, dtype=np.float64).copy()
        for substep_index in range(self.config.physics_substeps):
            guard_input_target = applied_target.copy()
            applied_target, guard_event = self._project_runtime_pusher_desk_target(
                self.data.qpos[:_JOINTS],
                applied_target,
                substep_index=substep_index,
            )
            static_projected_target = applied_target.copy()
            applied_target, dynamic_event = self._project_runtime_pusher_desk_dynamic_target(
                self.data.qpos[:_JOINTS],
                applied_target,
                substep_index=substep_index,
            )
            guard_event = {**guard_event, **dynamic_event}
            if bool(dynamic_event["dynamic_target_changed"]):
                queued_ids = np.asarray(
                    [int(queued.command_id) for queued in self._command_queue],
                    dtype=np.int64,
                )
                queued_targets_before = np.asarray(
                    [np.asarray(queued, dtype=np.float64).copy() for queued in self._command_queue],
                    dtype=np.float64,
                ).reshape(-1, _JOINTS)
                for queued in self._command_queue:
                    queued.rewrite_target(
                        applied_target,
                        reason="runtime_dynamic_clearance_projection",
                    )
                if queued_ids.size:
                    queue_rewrite_events.append(
                        {
                            "substep_index": int(substep_index),
                            "queued_command_ids": queued_ids,
                            "queued_targets_before": queued_targets_before,
                            "queued_targets_after": np.repeat(
                                applied_target[None, :], queued_ids.size, axis=0
                            ),
                            "queued_target_changed_mask": (
                                ~np.isclose(
                                    queued_targets_before,
                                    applied_target[None, :],
                                    rtol=0.0,
                                    atol=1.0e-12,
                                )
                            ).astype(np.uint8),
                            "reason": "runtime_dynamic_clearance_projection",
                        }
                    )
            applied_target_changed_mask = ~np.isclose(
                applied_target,
                servo_endpoint,
                rtol=0.0,
                atol=1.0e-12,
            )
            guard_input_targets.append(guard_input_target)
            static_projected_targets.append(static_projected_target)
            physics_applied_targets.append(applied_target.copy())
            guard_event.update(
                {
                    "guard_input_joint_target": guard_input_target.copy(),
                    "static_projected_joint_target": (static_projected_target.copy()),
                    "dynamic_projected_joint_target": applied_target.copy(),
                    "servo_endpoint_joint_target": servo_endpoint.copy(),
                    "delayed_controller_joint_target": (delayed_controller_target.copy()),
                    "physics_applied_joint_target": applied_target.copy(),
                    "physics_applied_target_changed_from_servo_endpoint_mask": (
                        applied_target_changed_mask.astype(np.uint8)
                    ),
                }
            )
            guard_events.append(guard_event)
            self.data.ctrl[:_JOINTS] = applied_target
            mujoco.mj_step(self.model, self.data)
            recorder.record(self.data, substep_index)
            observer = getattr(self, "_multichoice_substep_observer", None)
            if observer is not None:
                observer()
        self._last_actuator_force = self.data.actuator_force[:_JOINTS].copy()
        self._update_electrical_and_thermal_state()
        self._update_camera_flex()
        self._physics_substep_contact_v1 = recorder.finalize(
            self.data,
            final_block_xy=self.block_xy(),
            final_tool_xyz=self.tool_xyz(),
        )
        contact_identity_override = self._ids.get("tool_contact_identity_format")
        if contact_identity_override is not None:
            if not isinstance(contact_identity_override, str) or not contact_identity_override:
                raise RuntimeError("tool contact identity override must be a non-empty string")
            self._physics_substep_contact_v1["base_contact_identity_format"] = (
                self._physics_substep_contact_v1["contact_identity_format"]
            )
            self._physics_substep_contact_v1["contact_identity_format"] = contact_identity_override
        self._physics_substep_contact_v1.update(
            {
                "solver_contact_stage": ("mj_step_solver_contacts_before_final_position_cache_refresh"),
                "solver_contact_interval": "substep_start_to_substep_end",
                "effect_kinematic_stage": ("post_integration_qpos_plus_independent_scratch_mj_forward"),
                "tool_desk_signed_distance_stage": "effect_kinematic_stage",
                "kinematic_scratch_contact_force_used": False,
            }
        )
        if not self._controller_preflight_v1:  # pragma: no cover
            raise RuntimeError("V7 physics advanced without controller preflight")
        controller_preflight = deepcopy(self._controller_preflight_v1)
        controller_terminal_viability = dict(
            controller_preflight.get(
                "terminal_viability",
                {
                    "format": CONTROLLER_TERMINAL_VIABILITY_PROFILE_VERSION,
                    "gate_triggered": False,
                    "evaluated": False,
                    "rewrite_performed": False,
                    "episode_exact_evaluation_per_decision_density": (
                        self._terminal_viability_exact_evaluation_count / max(int(self.step_count) + 1, 1)
                    ),
                    "preview_in_progress": True,
                },
            )
        )
        self._runtime_pusher_desk_guard_v1 = {
            "format": "edgearm-runtime-pusher-desk-guard-v2",
            "enabled": bool(self.contact_feasible_config.runtime_pusher_desk_clearance_m > 0.0),
            "claim_level": "L1_SYNTHETIC_HARDWARE_INSPIRED",
            "physical_samples": 0,
            "physical_safety_validation": False,
            "clearance_margin_m": (self.contact_feasible_config.runtime_pusher_desk_clearance_m),
            "triggered": any(bool(event["triggered"]) for event in guard_events),
            "projection_count": sum(bool(event["target_changed"]) for event in guard_events),
            "infeasible_count": sum(not bool(event["feasible"]) for event in guard_events),
            "dynamic_forecast_triggered": any(
                bool(event["dynamic_forecast_triggered"]) for event in guard_events
            ),
            "dynamic_projection_count": sum(bool(event["dynamic_target_changed"]) for event in guard_events),
            "dynamic_infeasible_count": sum(
                not bool(event["dynamic_forecast_feasible"]) for event in guard_events
            ),
            "dynamic_hard_floor_m": (self.contact_feasible_config.runtime_pusher_desk_hard_floor_m),
            "dynamic_forecast_substeps": (self.contact_feasible_config.runtime_pusher_desk_forecast_substeps),
            "forecast_clearance_stage": ("post_each_forecast_integration_qpos_plus_scratch_mj_fwdPosition"),
            "solver_contact_force_source": ("live_mj_step_solver_contacts_without_position_cache_rewrite"),
            "controller_preflight": controller_preflight,
            "controller_terminal_viability": deepcopy(controller_terminal_viability),
            "controller_terminal_viability_gate_triggered": bool(
                controller_terminal_viability["gate_triggered"]
            ),
            "controller_terminal_viability_evaluated": bool(controller_terminal_viability["evaluated"]),
            "controller_terminal_viability_rewrite_performed": bool(
                controller_terminal_viability["rewrite_performed"]
            ),
            "controller_terminal_viability_episode_exact_evaluation_per_decision_density": float(
                controller_terminal_viability["episode_exact_evaluation_per_decision_density"]
            ),
            "controller_preflight_profile": str(controller_preflight["format"]),
            "controller_preflight_applied_command_id": int(controller_preflight["applied_command_id"]),
            "controller_preflight_input_joint_target": np.asarray(
                controller_preflight["input_joint_target"], dtype=np.float32
            ).copy(),
            "controller_preflight_output_joint_target": np.asarray(
                controller_preflight["output_joint_target"], dtype=np.float32
            ).copy(),
            "controller_preflight_target_changed_mask": np.asarray(
                controller_preflight["target_changed_mask"], dtype=np.uint8
            ).copy(),
            "controller_preflight_previewed_servo_endpoint_before_joint_target": np.asarray(
                controller_preflight["previewed_servo_endpoint_before_joint_target"],
                dtype=np.float32,
            ).copy(),
            "controller_preflight_previewed_servo_endpoint_after_joint_target": np.asarray(
                controller_preflight["previewed_servo_endpoint_after_joint_target"],
                dtype=np.float32,
            ).copy(),
            "controller_preflight_changed": bool(controller_preflight["changed"]),
            "controller_preflight_controller_search_attempt_count": int(
                controller_preflight["controller_search_attempt_count"]
            ),
            "controller_preflight_controller_search_fallback": str(
                controller_preflight["controller_search_fallback"]
            ),
            "controller_preflight_controller_bisection_iteration_count": int(
                controller_preflight["controller_bisection_iteration_count"]
            ),
            "controller_preflight_controller_bisection_selected_lambda": float(
                controller_preflight["controller_bisection_selected_lambda"]
            ),
            "controller_preflight_dynamic_forecast_feasible": bool(
                controller_preflight["dynamic_forecast_feasible"]
            ),
            "controller_preflight_dynamic_forecast_fallback": str(
                controller_preflight["dynamic_forecast_fallback"]
            ),
            "controller_preflight_infeasible_stop_suppressed": bool(
                controller_preflight["infeasible_stop_suppressed"]
            ),
            "safety_stop_requested": self._runtime_pusher_desk_safety_stop_requested,
            "servo_endpoint_joint_target": servo_endpoint.astype(np.float32),
            "delayed_controller_joint_target": (delayed_controller_target.astype(np.float32)),
            "physics_applied_joint_target": applied_target.astype(np.float32),
            "guard_input_joint_target_by_substep": np.asarray(guard_input_targets, dtype=np.float32),
            "static_projected_joint_target_by_substep": np.asarray(
                static_projected_targets, dtype=np.float32
            ),
            "physics_applied_joint_target_by_substep": np.asarray(physics_applied_targets, dtype=np.float32),
            "physics_applied_target_changed_from_servo_endpoint_mask_by_substep": (
                ~np.isclose(
                    np.asarray(physics_applied_targets, dtype=np.float64),
                    servo_endpoint[None, :],
                    rtol=0.0,
                    atol=1.0e-12,
                )
            ).astype(np.uint8),
            "physics_applied_target_changed_any": bool(
                np.any(
                    ~np.isclose(
                        np.asarray(physics_applied_targets, dtype=np.float64),
                        servo_endpoint[None, :],
                        rtol=0.0,
                        atol=1.0e-12,
                    )
                )
            ),
            "queue_mutated_by_runtime_guard": bool(queue_rewrite_events),
            "queue_rewrite_event_count": len(queue_rewrite_events),
            "queue_rewrite_events": queue_rewrite_events,
            "substep_events": guard_events,
        }

    def _runtime_pusher_desk_dynamic_forecast_acceptable(
        self,
        current_clearance: float,
        values: np.ndarray,
    ) -> bool:
        """Apply the one canonical margin/recovery/hard-floor forecast gate."""

        config = self.contact_feasible_config
        margin = float(config.runtime_pusher_desk_clearance_m)
        floor = float(config.runtime_pusher_desk_hard_floor_m)
        current_clearance = float(current_clearance)
        values = np.asarray(values, dtype=np.float64)
        if (
            not np.isfinite(current_clearance)
            or values.ndim != 1
            or values.size == 0
            or not np.all(np.isfinite(values))
        ):
            raise ValueError("dynamic forecast acceptability requires finite clearance samples")
        minimum = float(np.min(values))
        endpoint = float(values[-1])
        if current_clearance < margin:
            recovery_goal = min(
                margin,
                current_clearance + config.runtime_pusher_desk_guard_recovery_step_m,
            )
            return bool(
                minimum >= max(floor, current_clearance - 1.0e-5) and endpoint >= recovery_goal - 1.0e-5
            )
        return bool(minimum >= margin - 1.0e-5)

    def _project_runtime_pusher_desk_dynamic_target(
        self,
        current: np.ndarray,
        requested: np.ndarray,
        *,
        substep_index: int,
        request_safety_stop: bool = True,
        forecast_substeps: int | None = None,
    ) -> tuple[np.ndarray, dict[str, object]]:
        """Apply a short MuJoCo dynamics forecast after the static OBB guard.

        The static guard validates joint interpolation but cannot see inertial
        sag from the live ``qvel`` and actuator state.  This second barrier
        copies the complete ``MjData`` into scratch space, forecasts repeated
        force-limited ``mj_step`` transitions, and projects the actuator target
        before the live step.  Failure requests a fail-closed episode stop; the
        least-dangerous forecast target is still applied for the current
        substep because an emergency stop cannot erase physical momentum.
        """

        config = self.contact_feasible_config
        margin = float(config.runtime_pusher_desk_clearance_m)
        floor = float(config.runtime_pusher_desk_hard_floor_m)
        current = np.asarray(current, dtype=np.float64)
        requested = np.asarray(requested, dtype=np.float64)
        if current.shape != (_JOINTS,) or not np.all(np.isfinite(current)):
            raise ValueError("dynamic clearance guard requires a finite six-joint state")
        forecast_horizon = (
            self.contact_feasible_config.runtime_pusher_desk_forecast_substeps
            if forecast_substeps is None
            else int(forecast_substeps)
        )
        if forecast_horizon <= 0:
            raise ValueError("dynamic clearance forecast horizon must be positive")
        current_clearance = self._pusher_desk_distance_at_joint_position(current)
        disabled = {
            "dynamic_forecast_substep_index": int(substep_index),
            "dynamic_forecast_substeps": forecast_horizon,
            "dynamic_current_clearance_m": current_clearance,
            "dynamic_forecast_triggered": False,
            "dynamic_target_changed": False,
            "dynamic_forecast_feasible": True,
            "dynamic_forecast_minimum_before_m": current_clearance,
            "dynamic_forecast_minimum_after_m": current_clearance,
            "dynamic_forecast_endpoint_before_m": current_clearance,
            "dynamic_forecast_endpoint_after_m": current_clearance,
            "dynamic_forecast_iterations": 0,
            "dynamic_forecast_fallback": "disabled",
            "dynamic_forecast_safety_stop_enabled": bool(request_safety_stop),
            "dynamic_forecast_infeasible_stop_suppressed": False,
        }
        if margin <= 0.0:
            return requested.copy(), disabled

        # Far from the work surface, the static guard has ample reserve and a
        # dynamics rollout would add cost without changing the decision.
        if current_clearance >= margin + 0.010:
            disabled["dynamic_forecast_fallback"] = "far_clearance_fast_path"
            return requested.copy(), disabled

        forecast_before = self._pusher_desk_dynamic_clearance_forecast(
            requested,
            substeps=forecast_horizon,
        )
        recovery_mode = current_clearance < margin
        recovery_goal = min(
            margin,
            current_clearance + config.runtime_pusher_desk_guard_recovery_step_m,
        )

        def acceptable(values: np.ndarray) -> bool:
            return self._runtime_pusher_desk_dynamic_forecast_acceptable(
                current_clearance,
                values,
            )

        if acceptable(forecast_before):
            return requested.copy(), {
                **disabled,
                "dynamic_forecast_minimum_before_m": float(np.min(forecast_before)),
                "dynamic_forecast_minimum_after_m": float(np.min(forecast_before)),
                "dynamic_forecast_endpoint_before_m": float(forecast_before[-1]),
                "dynamic_forecast_endpoint_after_m": float(forecast_before[-1]),
                "dynamic_forecast_fallback": "forecast_already_safe",
            }

        max_joint_step = (
            np.asarray(self.sim2real_config.max_joint_velocity, dtype=np.float64) * self.control_dt
        )
        lower = np.maximum(self.joint_ranges[:, 0], current - max_joint_step)
        upper = np.minimum(self.joint_ranges[:, 1], current + max_joint_step)
        target = np.clip(requested, lower, upper)
        target[5] = requested[5]
        iterations = 0
        epsilon = config.runtime_pusher_desk_guard_finite_difference_rad
        for iterations in range(1, config.runtime_pusher_desk_guard_max_iterations + 1):
            forecast = self._pusher_desk_dynamic_clearance_forecast(
                target,
                substeps=forecast_horizon,
            )
            if acceptable(forecast):
                break
            use_endpoint = bool(
                recovery_mode and float(np.min(forecast)) >= max(floor, current_clearance - 1.0e-5)
            )
            metric = float(forecast[-1] if use_endpoint else np.min(forecast))
            desired = recovery_goal if use_endpoint else (floor if recovery_mode else margin)
            gradient = np.zeros(_JOINTS, dtype=np.float64)
            for joint_index in range(_JOINTS - 1):
                below = target.copy()
                above = target.copy()
                below[joint_index] = max(lower[joint_index], below[joint_index] - epsilon)
                above[joint_index] = min(upper[joint_index], above[joint_index] + epsilon)
                width = float(above[joint_index] - below[joint_index])
                if width <= 1.0e-12:
                    continue
                below_forecast = self._pusher_desk_dynamic_clearance_forecast(
                    below,
                    substeps=forecast_horizon,
                )
                above_forecast = self._pusher_desk_dynamic_clearance_forecast(
                    above,
                    substeps=forecast_horizon,
                )
                below_metric = float(below_forecast[-1] if use_endpoint else np.min(below_forecast))
                above_metric = float(above_forecast[-1] if use_endpoint else np.min(above_forecast))
                gradient[joint_index] = (above_metric - below_metric) / width
            denominator = float(np.dot(gradient, gradient))
            if denominator <= 1.0e-12:
                break
            deficit = desired - metric + config.runtime_pusher_desk_guard_reserve_m
            updated = target + deficit * gradient / denominator
            updated = np.clip(updated, lower, upper)
            updated[5] = requested[5]
            updated, _reason = self._safety_filter(updated)
            updated = np.clip(updated, lower, upper)
            updated[5] = requested[5]
            if np.allclose(updated, target, rtol=0.0, atol=1.0e-10):
                break
            target = updated

        forecast_after = self._pusher_desk_dynamic_clearance_forecast(
            target,
            substeps=forecast_horizon,
        )
        feasible = acceptable(forecast_after)
        fallback = "dynamic_gradient_projection"
        if not feasible:
            hold = current.copy()
            hold[5] = requested[5]
            hold_forecast = self._pusher_desk_dynamic_clearance_forecast(
                hold,
                substeps=forecast_horizon,
            )
            candidates = ((target, forecast_after), (hold, hold_forecast))
            target, forecast_after = max(
                candidates,
                key=lambda item: (float(np.min(item[1])), float(item[1][-1])),
            )
            feasible = acceptable(forecast_after)
            fallback = "dynamic_safe_hold" if feasible else "dynamic_fail_closed_best_brake"
        if not feasible and request_safety_stop:
            self._runtime_pusher_desk_safety_stop_requested = True
        return target.copy(), {
            **disabled,
            "dynamic_forecast_triggered": True,
            "dynamic_target_changed": bool(not np.allclose(target, requested, rtol=0.0, atol=1.0e-10)),
            "dynamic_forecast_feasible": feasible,
            "dynamic_forecast_minimum_before_m": float(np.min(forecast_before)),
            "dynamic_forecast_minimum_after_m": float(np.min(forecast_after)),
            "dynamic_forecast_endpoint_before_m": float(forecast_before[-1]),
            "dynamic_forecast_endpoint_after_m": float(forecast_after[-1]),
            "dynamic_forecast_iterations": iterations,
            "dynamic_forecast_fallback": fallback,
            "dynamic_forecast_infeasible_stop_suppressed": bool(not feasible and not request_safety_stop),
        }

    def _pusher_desk_dynamic_clearance_forecast(
        self,
        target: np.ndarray,
        *,
        substeps: int | None = None,
    ) -> np.ndarray:
        if self._clearance_guard_dynamics_scratch is None:  # pragma: no cover
            raise RuntimeError("dynamic clearance guard scratch data is unavailable")
        target = np.asarray(target, dtype=np.float64)
        if target.shape != (_JOINTS,) or not np.all(np.isfinite(target)):
            raise ValueError("dynamic clearance forecast requires a finite six-joint target")
        scratch = self._clearance_guard_dynamics_scratch
        mujoco.mj_copyData(scratch, self.model, self.data)
        forecast_horizon = (
            self.contact_feasible_config.runtime_pusher_desk_forecast_substeps
            if substeps is None
            else int(substeps)
        )
        if forecast_horizon <= 0:
            raise ValueError("dynamic clearance forecast horizon must be positive")
        values = np.zeros(forecast_horizon, dtype=np.float64)
        for index in range(values.size):
            scratch.ctrl[:_JOINTS] = target
            mujoco.mj_step(self.model, scratch)
            # ``mj_step`` may leave position-dependent geom caches at the
            # input position stage even though qpos has been integrated.  The
            # forecast scratch is geometry-only: refreshing its position stage
            # makes each clearance sample refer to the new qpos.  No contact
            # force is ever read from this scratch, so solver contact indices
            # cannot be mixed with refreshed geometry.
            mujoco.mj_fwdPosition(self.model, scratch)
            values[index] = self._minimum_tool_safety_signed_distance_for_data(
                self._desk_geom,
                scratch,
            )
        return values

    def _project_runtime_pusher_desk_target(
        self,
        current: np.ndarray,
        requested: np.ndarray,
        *,
        substep_index: int,
    ) -> tuple[np.ndarray, dict[str, object]]:
        """Project an actuator target through an opt-in exact-OBB barrier.

        Candidate geometry is evaluated only in an independent ``MjData``.
        The returned target is still executed by MuJoCo actuators; this method
        never writes the live joint state.
        """

        config = self.contact_feasible_config
        current = np.asarray(current, dtype=np.float64)
        requested = np.asarray(requested, dtype=np.float64)
        if current.shape != (_JOINTS,) or requested.shape != (_JOINTS,):
            raise ValueError("runtime clearance guard requires six-joint targets")
        margin = float(config.runtime_pusher_desk_clearance_m)
        # A MuJoCo integration step can leave live ``geom_xpos`` at the input
        # position stage after qpos integration (including IMPLICITFAST).  All
        # guard geometry must refer
        # to one physical instant, so derive current clearance from the passed
        # qpos in scratch data just like the path samples below.  Never call
        # ``mj_forward`` on live data here because that would perturb the plant
        # pipeline we are trying to guard.
        current_clearance = self._pusher_desk_distance_at_joint_position(current)
        disabled_event: dict[str, object] = {
            "substep_index": int(substep_index),
            "triggered": False,
            "recovery_mode": False,
            "target_changed": False,
            "feasible": True,
            "iterations": 0,
            "current_clearance_m": current_clearance,
            "minimum_path_clearance_before_m": current_clearance,
            "minimum_path_clearance_after_m": current_clearance,
            "target_clearance_before_m": current_clearance,
            "target_clearance_after_m": current_clearance,
            "fallback": "disabled",
        }
        if margin <= 0.0:
            return requested.copy(), disabled_event

        original = requested.copy()
        target = requested.copy()
        if substep_index > 0 and current_clearance >= margin:
            quick_samples = np.array([0.0, 0.5, 1.0], dtype=np.float64)
            quick_clearances = self._pusher_desk_clearance_path(
                current,
                target,
                quick_samples,
            )
            quick_minimum = float(np.min(quick_clearances))
            if quick_minimum >= (margin + config.runtime_pusher_desk_guard_reserve_m):
                return target, {
                    "substep_index": int(substep_index),
                    "triggered": False,
                    "recovery_mode": False,
                    "target_changed": False,
                    "feasible": True,
                    "iterations": 0,
                    "current_clearance_m": current_clearance,
                    "minimum_path_clearance_before_m": quick_minimum,
                    "minimum_path_clearance_after_m": quick_minimum,
                    "target_clearance_before_m": float(quick_clearances[-1]),
                    "target_clearance_after_m": float(quick_clearances[-1]),
                    "fallback": "three_point_substep_fast_path",
                }
        samples = np.linspace(
            0.0,
            1.0,
            config.runtime_pusher_desk_guard_path_samples,
        )
        before_clearances = self._pusher_desk_clearance_path(current, target, samples)
        recovery_mode = current_clearance < margin
        initial_minimum = float(np.min(before_clearances))
        initial_target_clearance = float(before_clearances[-1])
        triggered = bool(
            initial_minimum < margin - 1.0e-5
            if not recovery_mode
            else initial_target_clearance
            < min(
                margin,
                current_clearance + config.runtime_pusher_desk_guard_recovery_step_m,
            )
            - 1.0e-5
        )
        if not triggered:
            return target, {
                "substep_index": int(substep_index),
                "triggered": False,
                "recovery_mode": recovery_mode,
                "target_changed": False,
                "feasible": True,
                "iterations": 0,
                "current_clearance_m": current_clearance,
                "minimum_path_clearance_before_m": initial_minimum,
                "minimum_path_clearance_after_m": initial_minimum,
                "target_clearance_before_m": initial_target_clearance,
                "target_clearance_after_m": initial_target_clearance,
                "fallback": "full_path_already_safe",
            }
        iterations = 0
        fallback = "none"
        feasible = not triggered
        max_joint_step = (
            np.asarray(self.sim2real_config.max_joint_velocity, dtype=np.float64) * self.control_dt
        )
        lower = np.maximum(self.joint_ranges[:, 0], current - max_joint_step)
        upper = np.minimum(self.joint_ranges[:, 1], current + max_joint_step)
        target = np.clip(target, lower, upper)
        target[5] = original[5]
        for iterations in range(1, config.runtime_pusher_desk_guard_max_iterations + 1):
            clearances = self._pusher_desk_clearance_path(current, target, samples)
            if recovery_mode:
                recovery_goal = min(
                    margin,
                    current_clearance + config.runtime_pusher_desk_guard_recovery_step_m,
                )
                minimum_after_start = float(np.min(clearances[1:]))
                endpoint_clearance = float(clearances[-1])
                no_regression_floor = current_clearance - 1.0e-5
                if (
                    endpoint_clearance >= recovery_goal - 1.0e-5
                    and minimum_after_start >= no_regression_floor
                ):
                    feasible = True
                    break
                if minimum_after_start < no_regression_floor:
                    local_index = int(np.argmin(clearances[1:])) + 1
                    desired_clearance = no_regression_floor
                else:
                    local_index = len(samples) - 1
                    desired_clearance = recovery_goal
            else:
                local_index = int(np.argmin(clearances))
                if float(clearances[local_index]) >= margin - 1.0e-5:
                    feasible = True
                    break
                desired_clearance = margin
            fraction = float(samples[local_index])
            probe = current + fraction * (target - current)
            gradient = self._pusher_desk_clearance_gradient(probe)
            effective_gradient = fraction * gradient
            denominator = float(np.dot(effective_gradient, effective_gradient))
            if denominator <= 1.0e-12:
                fallback = "zero_clearance_gradient"
                break
            deficit = (
                desired_clearance
                - float(clearances[local_index])
                + config.runtime_pusher_desk_guard_reserve_m
            )
            updated = target + deficit * effective_gradient / denominator
            updated = np.clip(updated, lower, upper)
            updated[5] = original[5]
            updated, _workspace_reason = self._safety_filter(updated)
            updated = np.clip(updated, lower, upper)
            updated[5] = original[5]
            if np.allclose(updated, target, rtol=0.0, atol=1.0e-10):
                fallback = "projection_stalled"
                break
            target = updated

        after_clearances = self._pusher_desk_clearance_path(current, target, samples)
        if recovery_mode:
            recovery_goal = min(
                margin,
                current_clearance + config.runtime_pusher_desk_guard_recovery_step_m,
            )
            feasible = bool(
                float(after_clearances[-1]) >= recovery_goal - 1.0e-5
                and float(np.min(after_clearances[1:])) >= current_clearance - 1.0e-5
            )
        else:
            feasible = bool(float(np.min(after_clearances)) >= margin - 1.0e-5)
        if triggered and not feasible:
            hold_clearances = self._pusher_desk_clearance_path(
                current,
                current,
                samples,
            )
            if not recovery_mode and float(np.min(hold_clearances)) >= margin - 1.0e-5:
                target = current.copy()
                target[5] = original[5]
                after_clearances = hold_clearances
                feasible = True
                fallback = "safe_hold"
            else:
                candidates = [target, current.copy()]
                candidate_clearances = [
                    self._pusher_desk_distance_at_joint_position(candidate) for candidate in candidates
                ]
                best_index = int(np.argmax(candidate_clearances))
                target = candidates[best_index]
                after_clearances = self._pusher_desk_clearance_path(
                    current,
                    target,
                    samples,
                )
                fallback = "best_effort_recovery"
        return target, {
            "substep_index": int(substep_index),
            "triggered": triggered,
            "recovery_mode": recovery_mode,
            "target_changed": bool(not np.allclose(target, original, rtol=0.0, atol=1.0e-10)),
            "feasible": feasible,
            "iterations": iterations if triggered else 0,
            "current_clearance_m": current_clearance,
            "minimum_path_clearance_before_m": initial_minimum,
            "minimum_path_clearance_after_m": float(np.min(after_clearances)),
            "target_clearance_before_m": initial_target_clearance,
            "target_clearance_after_m": float(after_clearances[-1]),
            "fallback": fallback,
        }

    def _pusher_desk_clearance_path(
        self,
        current: np.ndarray,
        target: np.ndarray,
        samples: np.ndarray,
    ) -> np.ndarray:
        return np.asarray(
            [
                self._pusher_desk_distance_at_joint_position(current + float(fraction) * (target - current))
                for fraction in samples
            ],
            dtype=np.float64,
        )

    def _pusher_desk_clearance_gradient(self, joint_position: np.ndarray) -> np.ndarray:
        epsilon = self.contact_feasible_config.runtime_pusher_desk_guard_finite_difference_rad
        gradient = np.zeros(_JOINTS, dtype=np.float64)
        for joint_index in range(_JOINTS - 1):
            lower = joint_position.copy()
            upper = joint_position.copy()
            lower[joint_index] -= epsilon
            upper[joint_index] += epsilon
            gradient[joint_index] = (
                self._pusher_desk_distance_at_joint_position(upper)
                - self._pusher_desk_distance_at_joint_position(lower)
            ) / (2.0 * epsilon)
        return gradient

    def _pusher_desk_distance_at_joint_position(
        self,
        joint_position: np.ndarray,
    ) -> float:
        joint_position = np.asarray(joint_position, dtype=np.float64)
        if joint_position.shape != (_JOINTS,) or not np.all(np.isfinite(joint_position)):
            raise ValueError("clearance prediction requires a finite six-joint vector")
        if self._clearance_guard_scratch is None:  # pragma: no cover
            raise RuntimeError("clearance guard scratch data is unavailable")
        scratch = self._clearance_guard_scratch
        scratch.qpos[:] = self.data.qpos
        scratch.qvel[:] = self.data.qvel
        scratch.qpos[:_JOINTS] = joint_position
        mujoco.mj_forward(self.model, scratch)
        return self._minimum_tool_safety_signed_distance_for_data(
            self._desk_geom,
            scratch,
        )

    def _install_reset_joint_state(self, joint_position: np.ndarray) -> None:
        joint_position = np.asarray(joint_position, dtype=np.float64)
        if joint_position.shape != (_JOINTS,) or not np.all(np.isfinite(joint_position)):
            raise ValueError("V7 reset joint position must be a finite six-vector")
        self.data.qpos[:_JOINTS] = joint_position
        self.data.qvel[:_JOINTS] = 0.0
        self.data.ctrl[:_JOINTS] = joint_position
        self.data.qacc_warmstart[:_JOINTS] = 0.0
        self._servo_velocity.fill(0.0)
        self._last_motor_direction.fill(0)
        self._backlash_remaining.fill(0.0)
        self._last_actual_velocity.fill(0.0)
        self._last_actuator_force.fill(0.0)
        for queued in self._command_queue:
            queued[:] = joint_position
        mujoco.mj_forward(self.model, self.data)

    def _geom_world_aabb(self, geom_id: int) -> tuple[np.ndarray, np.ndarray]:
        center = np.asarray(self.data.geom_xpos[geom_id], dtype=np.float64)
        rotation = np.asarray(self.data.geom_xmat[geom_id], dtype=np.float64).reshape(3, 3)
        half_extent = np.abs(rotation) @ np.asarray(self.model.geom_size[geom_id], dtype=np.float64)
        return center - half_extent, center + half_extent

    def _geom_box_signed_distance(self, first: int, second: int) -> float:
        return self._geom_box_signed_distance_for_data(first, second, self.data)

    def _geom_box_signed_distance_for_data(
        self,
        first: int,
        second: int,
        data: mujoco.MjData,
    ) -> float:
        box_type = int(mujoco.mjtGeom.mjGEOM_BOX)
        if int(self.model.geom_type[first]) != box_type or int(self.model.geom_type[second]) != box_type:
            raise TypeError("V7 exact reset clearance supports box geoms only")
        return _box_box_signed_distance(
            data.geom_xpos[first],
            data.geom_xmat[first],
            self.model.geom_size[first],
            data.geom_xpos[second],
            data.geom_xmat[second],
            self.model.geom_size[second],
        )

    def _tool_planning_geom_ids(self) -> tuple[int, ...]:
        """Return the ordered boxes used for tool clearance and contact planning."""

        values = self._ids.get("tool_planning_geoms", (self._ids["tool_geom"],))
        result = tuple(int(value) for value in values)
        if not result:  # pragma: no cover - guarded by production model resolution
            raise RuntimeError("tool planning geometry set is empty")
        return result

    def _tool_safety_geom_ids(self) -> tuple[int, ...]:
        """Return the ordered convex union used for robot/tool safety."""

        values = self._ids.get("tool_safety_geoms", self._tool_planning_geom_ids())
        result = tuple(int(value) for value in values)
        if not result:  # pragma: no cover - guarded by production model resolution
            raise RuntimeError("tool safety geometry set is empty")
        return result

    def _geom_signed_distance_for_data(
        self,
        first: int,
        second: int,
        data: mujoco.MjData,
        *,
        cutoff_m: float = 0.030,
    ) -> float:
        """Distance for either exact boxes or MuJoCo convex primitives/meshes.

        MuJoCo's generic mesh distance query can return ``0`` for a separated
        mesh/box pair once ``distmax`` extends beyond the narrow-phase search
        range.  Every compiled geom also carries a local bounding box.  The
        distance between the corresponding world OBBs is a conservative lower
        bound on the distance between the enclosed shapes, so a positive OBB
        separation is safe evidence with which to repair that false zero.
        """

        box_type = int(mujoco.mjtGeom.mjGEOM_BOX)
        if int(self.model.geom_type[first]) == box_type and int(self.model.geom_type[second]) == box_type:
            return self._geom_box_signed_distance_for_data(first, second, data)
        if not np.isfinite(cutoff_m) or cutoff_m <= 0.0:
            raise ValueError("generic geom distance cutoff must be finite and positive")
        closest = np.zeros(6, dtype=np.float64)
        distance = float(
            mujoco.mj_geomDistance(
                self.model,
                data,
                int(first),
                int(second),
                float(cutoff_m),
                closest,
            )
        )
        if not np.isfinite(distance):
            raise RuntimeError("generic geom distance evaluation produced a non-finite value")
        if distance != 0.0:
            return distance
        mesh_type = int(mujoco.mjtGeom.mjGEOM_MESH)
        if int(self.model.geom_type[first]) == mesh_type and int(self.model.geom_type[second]) == box_type:
            convex_lower_bound = self._convex_mesh_box_separation_lower_bound_for_data(
                first,
                second,
                data,
            )
            if convex_lower_bound > 0.0:
                return convex_lower_bound
        elif int(self.model.geom_type[second]) == mesh_type and int(self.model.geom_type[first]) == box_type:
            convex_lower_bound = self._convex_mesh_box_separation_lower_bound_for_data(
                second,
                first,
                data,
            )
            if convex_lower_bound > 0.0:
                return convex_lower_bound
        first_rotation = np.asarray(data.geom_xmat[first], dtype=np.float64).reshape(3, 3)
        second_rotation = np.asarray(data.geom_xmat[second], dtype=np.float64).reshape(3, 3)
        first_aabb = np.asarray(self.model.geom_aabb[first], dtype=np.float64)
        second_aabb = np.asarray(self.model.geom_aabb[second], dtype=np.float64)
        conservative_lower_bound = _box_box_separation_lower_bound(
            np.asarray(data.geom_xpos[first], dtype=np.float64) + first_rotation @ first_aabb[:3],
            first_rotation,
            first_aabb[3:],
            np.asarray(data.geom_xpos[second], dtype=np.float64) + second_rotation @ second_aabb[:3],
            second_rotation,
            second_aabb[3:],
        )
        return float(max(distance, conservative_lower_bound))

    def _convex_mesh_box_separation_lower_bound_for_data(
        self,
        mesh_geom: int,
        box_geom: int,
        data: mujoco.MjData,
    ) -> float:
        """Return the complete convex-polytope/OBB SAT separation lower bound."""

        cache = getattr(self, "_convex_mesh_sat_cache_v1", None)
        if cache is None:
            cache = {}
            self._convex_mesh_sat_cache_v1 = cache
        mesh_id = int(self.model.geom_dataid[mesh_geom])
        cached = cache.get(mesh_id)
        if cached is None:
            vertex_start = int(self.model.mesh_vertadr[mesh_id])
            vertex_count = int(self.model.mesh_vertnum[mesh_id])
            face_start = int(self.model.mesh_faceadr[mesh_id])
            face_count = int(self.model.mesh_facenum[mesh_id])
            vertices = np.asarray(
                self.model.mesh_vert[vertex_start : vertex_start + vertex_count],
                dtype=np.float64,
            ).copy()
            faces = np.asarray(
                self.model.mesh_face[face_start : face_start + face_count],
                dtype=np.int64,
            ).copy()
            triangles = vertices[faces]
            face_normals = np.cross(
                triangles[:, 1] - triangles[:, 0],
                triangles[:, 2] - triangles[:, 0],
            )
            normal_norms = np.linalg.norm(face_normals, axis=1)
            face_normals = face_normals[normal_norms > 1.0e-12]
            face_normals /= np.linalg.norm(face_normals, axis=1, keepdims=True)
            edge_pairs = np.vstack(
                [
                    faces[:, [0, 1]],
                    faces[:, [1, 2]],
                    faces[:, [2, 0]],
                ]
            )
            edge_pairs.sort(axis=1)
            edge_pairs = np.unique(edge_pairs, axis=0)
            edge_directions = vertices[edge_pairs[:, 1]] - vertices[edge_pairs[:, 0]]
            edge_norms = np.linalg.norm(edge_directions, axis=1)
            edge_directions = edge_directions[edge_norms > 1.0e-12]
            edge_directions /= np.linalg.norm(edge_directions, axis=1, keepdims=True)
            cached = (vertices, face_normals, edge_directions)
            cache[mesh_id] = cached
        vertices, face_normals, edge_directions = cached
        mesh_rotation = np.asarray(
            data.geom_xmat[mesh_geom],
            dtype=np.float64,
        ).reshape(3, 3)
        mesh_vertices = np.asarray(data.geom_xpos[mesh_geom], dtype=np.float64) + vertices @ mesh_rotation.T
        world_face_normals = face_normals @ mesh_rotation.T
        world_edge_directions = edge_directions @ mesh_rotation.T
        box_rotation = np.asarray(data.geom_xmat[box_geom], dtype=np.float64).reshape(3, 3)
        box_axes = box_rotation.T
        edge_cross_axes = np.cross(
            world_edge_directions[:, None, :],
            box_axes[None, :, :],
        ).reshape(-1, 3)
        raw_axes = np.vstack([world_face_normals, box_axes, edge_cross_axes])
        axis_norms = np.linalg.norm(raw_axes, axis=1)
        axes = raw_axes[axis_norms > 1.0e-12]
        axes /= np.linalg.norm(axes, axis=1, keepdims=True)
        if not np.all(np.isfinite(mesh_vertices)) or not np.all(np.isfinite(axes)):
            raise RuntimeError("convex mesh SAT received non-finite vertices or axes")
        # Some MuJoCo narrow-phase calls leave floating-point status flags set
        # even though the returned arrays are finite.  Suppress those stale
        # flags around BLAS and validate the actual result immediately after.
        with np.errstate(all="ignore"):
            mesh_projection = mesh_vertices @ axes.T
        if not np.all(np.isfinite(mesh_projection)):
            raise RuntimeError("convex mesh SAT projection produced a non-finite value")
        box_center_projection = axes @ np.asarray(
            data.geom_xpos[box_geom],
            dtype=np.float64,
        )
        box_radius = np.abs(axes @ box_rotation) @ np.asarray(
            self.model.geom_size[box_geom],
            dtype=np.float64,
        )
        gaps = np.maximum(
            box_center_projection - box_radius - np.max(mesh_projection, axis=0),
            np.min(mesh_projection, axis=0) - box_center_projection - box_radius,
        )
        return float(np.max(gaps))

    def _tool_planning_signed_distances_for_data(
        self,
        other: int,
        data: mujoco.MjData,
    ) -> np.ndarray:
        """Return one exact OBB distance per ordered planning reference."""

        distances = np.asarray(
            [
                self._geom_box_signed_distance_for_data(geom_id, other, data)
                for geom_id in self._tool_planning_geom_ids()
            ],
            dtype=np.float64,
        )
        if distances.ndim != 1 or not np.all(np.isfinite(distances)):
            raise RuntimeError("tool planning distance evaluation produced a non-finite value")
        return distances

    def _tool_safety_signed_distances_for_data(
        self,
        other: int,
        data: mujoco.MjData,
    ) -> np.ndarray:
        distances = np.asarray(
            [
                self._geom_signed_distance_for_data(geom_id, other, data)
                for geom_id in self._tool_safety_geom_ids()
            ],
            dtype=np.float64,
        )
        if distances.ndim != 1 or not np.all(np.isfinite(distances)):
            raise RuntimeError("tool safety distance evaluation produced a non-finite value")
        return distances

    def _tool_planning_signed_distance_rows_for_data(
        self,
        other: int,
        data: mujoco.MjData,
    ) -> tuple[dict[str, object], ...]:
        distances = self._tool_planning_signed_distances_for_data(other, data)
        return tuple(
            {
                "geom_id": geom_id,
                "geom_name": mujoco.mj_id2name(
                    self.model,
                    mujoco.mjtObj.mjOBJ_GEOM,
                    geom_id,
                ),
                "signed_distance_m": float(distance),
            }
            for geom_id, distance in zip(
                self._tool_planning_geom_ids(),
                distances,
                strict=True,
            )
        )

    def _minimum_tool_planning_signed_distance_for_data(
        self,
        other: int,
        data: mujoco.MjData,
    ) -> float:
        return float(np.min(self._tool_planning_signed_distances_for_data(other, data)))

    def _minimum_tool_safety_signed_distance_for_data(
        self,
        other: int,
        data: mujoco.MjData,
    ) -> float:
        return float(np.min(self._tool_safety_signed_distances_for_data(other, data)))

    def _maximum_tool_planning_signed_distance_for_data(
        self,
        other: int,
        data: mujoco.MjData,
    ) -> float:
        return float(np.max(self._tool_planning_signed_distances_for_data(other, data)))

    def _minimum_tool_planning_signed_distance(self, other: int) -> float:
        return self._minimum_tool_planning_signed_distance_for_data(other, self.data)

    def _minimum_tool_safety_signed_distance(self, other: int) -> float:
        return self._minimum_tool_safety_signed_distance_for_data(other, self.data)

    def _maximum_tool_planning_signed_distance(self, other: int) -> float:
        return self._maximum_tool_planning_signed_distance_for_data(other, self.data)

    def _tool_face_support_radius_for_data(self, data: mujoco.MjData) -> float:
        """Bound every planning reference along the orientation face normal."""

        orientation = int(self._ids["tool_geom"])
        face_axis = np.asarray(data.geom_xmat[orientation], dtype=np.float64).reshape(3, 3)[:, 1]
        site_position = np.asarray(data.site_xpos[self._ids["tool_site"]], dtype=np.float64)
        support = 0.0
        for geom_id in self._tool_planning_geom_ids():
            rotation = np.asarray(data.geom_xmat[geom_id], dtype=np.float64).reshape(3, 3)
            center_offset = float(
                np.dot(np.asarray(data.geom_xpos[geom_id], dtype=np.float64) - site_position, face_axis)
            )
            radius = float(
                np.dot(
                    np.abs(rotation.T @ face_axis),
                    np.asarray(self.model.geom_size[geom_id], dtype=np.float64),
                )
            )
            support = max(support, abs(center_offset) + radius)
        if not np.isfinite(support) or support <= 0.0:
            raise RuntimeError("tool face support radius is invalid")
        return support

    def _audit_reset_contacts(
        self,
        target_xyz: np.ndarray,
        push_direction: np.ndarray,
        *,
        block_settle_displacement_m: float,
    ) -> dict[str, Any]:
        desk = self._desk_geom
        block = self._ids["block_geom"]
        tool = self._ids["tool_geom"]
        tool_contact_geoms = frozenset(self._ids["tool_contact_geoms"])
        obstacle = self._ids["obstacle_geom"]
        tolerance = self.contact_feasible_config.reset_penetration_tolerance_m
        forbidden: list[dict[str, Any]] = []
        tool_block_contacts: list[dict[str, Any]] = []
        obstacle_block_contact_count = 0
        block_desk_contact_count = 0
        robot_obstacle_contacts = {
            "pusher": 0,
            "camera_housing": 0,
            "robot_other": 0,
        }
        tool_desk_min_distance = np.inf
        non_tool_robot_desk_min_distance = np.inf
        tool_block_min_distance = np.inf
        for index in range(self.data.ncon):
            contact = self.data.contact[index]
            first = int(contact.geom1)
            second = int(contact.geom2)
            pair = {first, second}
            distance = float(contact.dist)
            if block in pair and bool(pair & tool_contact_geoms):
                tool_block_min_distance = min(tool_block_min_distance, distance)
                tool_block_contacts.append(
                    {
                        "distance_m": distance,
                        "position_m": np.asarray(contact.pos, dtype=np.float64).tolist(),
                    }
                )
            if pair == {obstacle, block}:
                obstacle_block_contact_count += 1
            if pair == {desk, block}:
                block_desk_contact_count += 1
            if obstacle in pair:
                other = second if first == obstacle else first
                if other in tool_contact_geoms:
                    robot_obstacle_contacts["pusher"] += 1
                elif other == self._camera_housing_geom:
                    robot_obstacle_contacts["camera_housing"] += 1
                elif other in self._robot_geoms:
                    robot_obstacle_contacts["robot_other"] += 1
            if desk not in pair:
                continue
            other = second if first == desk else first
            if other == block:
                continue
            if other in tool_contact_geoms:
                tool_desk_min_distance = min(tool_desk_min_distance, distance)
            elif other in self._robot_geoms and other != tool:
                non_tool_robot_desk_min_distance = min(non_tool_robot_desk_min_distance, distance)
            else:
                continue
            if distance < -tolerance:
                forbidden.append(
                    {
                        "other_geom_id": other,
                        "other_geom_name": mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, other),
                        "other_body_name": mujoco.mj_id2name(
                            self.model,
                            mujoco.mjtObj.mjOBJ_BODY,
                            int(self.model.geom_bodyid[other]),
                        ),
                        "distance_m": distance,
                    }
                )

        def finite_or_none(value: float) -> float | None:
            return None if not np.isfinite(value) else float(value)

        target_xyz = np.asarray(target_xyz, dtype=np.float64)
        push_direction = self._unit(np.asarray(push_direction, dtype=np.float64))
        actual_xyz = np.asarray(self.tool_xyz(), dtype=np.float64)
        pose_error = float(np.linalg.norm(actual_xyz - target_xyz))
        pose_tolerance = float(self.contact_feasible_config.reset_pose_error_tolerance_m)
        block_xy = self.block_xy()
        tool_to_block = block_xy - actual_xyz[:2]
        normal = np.array([-push_direction[1], push_direction[0]], dtype=np.float64)
        achieved_standoff = float(np.dot(tool_to_block, push_direction))
        requested_standoff = float(self.contact_feasible_config.reset_tool_standoff_m)
        standoff_error = float(abs(achieved_standoff - requested_standoff))
        lateral_error = float(abs(np.dot(tool_to_block, normal)))
        vertical_error = float(abs(actual_xyz[2] - target_xyz[2]))
        joint_position = np.asarray(self.data.qpos[:_JOINTS], dtype=np.float64)
        joint_limit_margin = np.minimum(
            joint_position - self.joint_ranges[:, 0],
            self.joint_ranges[:, 1] - joint_position,
        )
        joint_limit_active = joint_limit_margin <= 1.0e-4
        ik_converged = bool(
            pose_error <= pose_tolerance
            and standoff_error <= self.contact_feasible_config.reset_standoff_error_tolerance_m
            and lateral_error <= self.contact_feasible_config.reset_lateral_error_tolerance_m
            and vertical_error <= self.contact_feasible_config.reset_vertical_error_tolerance_m
        )

        tool_block_distance_rows = self._tool_planning_signed_distance_rows_for_data(
            block,
            self.data,
        )
        tool_block_distances = np.asarray(
            [float(row["signed_distance_m"]) for row in tool_block_distance_rows],
            dtype=np.float64,
        )
        tool_block_distance = float(np.min(tool_block_distances))
        block_desk_distance = self._geom_box_signed_distance(block, desk)
        obstacle_block_distance = (
            self._geom_box_signed_distance(obstacle, block) if self.obstacle_enabled else None
        )
        if self.obstacle_enabled:
            goal_block_center = np.array(
                [
                    self.target_xy[0],
                    self.target_xy[1],
                    self.data.geom_xpos[block, 2],
                ],
                dtype=np.float64,
            )
            obstacle_goal_block_distance = _box_box_signed_distance(
                self.data.geom_xpos[obstacle],
                self.data.geom_xmat[obstacle],
                self.model.geom_size[obstacle],
                goal_block_center,
                self.data.geom_xmat[block],
                self.model.geom_size[block],
            )
            obstacle_path_relevant = bool(self._path_intersects_obstacle(block_xy, self.target_xy))
        else:
            obstacle_goal_block_distance = None
            obstacle_path_relevant = False
        tool_aabb_min, tool_aabb_max = self._geom_world_aabb(tool)
        block_aabb_min, block_aabb_max = self._geom_world_aabb(block)
        desk_aabb_min, desk_aabb_max = self._geom_world_aabb(desk)
        obstacle_aabb_min, obstacle_aabb_max = self._geom_world_aabb(obstacle)
        block_support_margin = float(
            min(
                np.min(block_aabb_min[:2] - desk_aabb_min[:2]),
                np.min(desk_aabb_max[:2] - block_aabb_max[:2]),
            )
        )
        obstacle_support_margin = (
            float(
                min(
                    np.min(obstacle_aabb_min[:2] - desk_aabb_min[:2]),
                    np.min(desk_aabb_max[:2] - obstacle_aabb_max[:2]),
                )
            )
            if self.obstacle_enabled
            else None
        )

        failure_reasons: list[str] = []
        if forbidden:
            failure_reasons.append("forbidden_workbench_robot_penetration")
        if tool_block_contacts:
            failure_reasons.append("initial_tool_block_contact")
        if tool_block_distance < self.contact_feasible_config.reset_tool_block_clearance_m:
            failure_reasons.append("insufficient_tool_block_signed_clearance")
        if self.obstacle_enabled and (
            obstacle_block_contact_count > 0
            or obstacle_block_distance is None
            or obstacle_block_distance
            < self.contact_feasible_config.reset_obstacle_block_clearance_m - 1.0e-9
        ):
            failure_reasons.append("initial_obstacle_block_overlap_or_clearance")
        if self.obstacle_enabled and (
            obstacle_goal_block_distance is None
            or obstacle_goal_block_distance
            < self.contact_feasible_config.reset_obstacle_block_clearance_m - 1.0e-9
        ):
            failure_reasons.append("obstacle_overlaps_goal_block_footprint")
        if self.obstacle_enabled and not obstacle_path_relevant:
            failure_reasons.append("obstacle_not_relevant_to_direct_push_path")
        if any(robot_obstacle_contacts.values()):
            failure_reasons.append("initial_robot_obstacle_contact")
        if not ik_converged:
            failure_reasons.append("task_aligned_reset_ik_not_converged")
        if block_support_margin < self.contact_feasible_config.reset_support_margin_m:
            failure_reasons.append("block_outside_workbench_support_bounds")
        if (
            self.obstacle_enabled
            and obstacle_support_margin is not None
            and obstacle_support_margin < self.contact_feasible_config.reset_support_margin_m
        ):
            failure_reasons.append("obstacle_outside_workbench_support_bounds")
        if block_desk_contact_count <= 0:
            failure_reasons.append("block_not_supported_after_reset_settle")
        if block_settle_displacement_m > self.contact_feasible_config.reset_block_settle_xy_tolerance_m:
            failure_reasons.append("block_moved_during_reset_settle")

        return {
            "reset_valid": not failure_reasons,
            "reset_failure_reasons": failure_reasons,
            "forbidden_penetration_count": len(forbidden),
            "forbidden_contacts": forbidden,
            "tool_block_contact_count": len(tool_block_contacts),
            "tool_block_contacts": tool_block_contacts,
            "tool_block_min_contact_distance_m": finite_or_none(tool_block_min_distance),
            "box_distance_method": CONTACT_FEASIBLE_BOX_DISTANCE_VERSION,
            "tool_block_signed_distance_m": tool_block_distance,
            "tool_block_maximum_signed_distance_m": float(np.max(tool_block_distances)),
            "tool_block_signed_distance_by_planning_geom": tool_block_distance_rows,
            "tool_block_required_clearance_m": (self.contact_feasible_config.reset_tool_block_clearance_m),
            "obstacle_block_contact_count": obstacle_block_contact_count,
            "obstacle_block_signed_distance_m": obstacle_block_distance,
            "obstacle_goal_block_signed_distance_m": obstacle_goal_block_distance,
            "obstacle_path_relevant": obstacle_path_relevant,
            "obstacle_block_required_clearance_m": (
                self.contact_feasible_config.reset_obstacle_block_clearance_m
            ),
            "robot_obstacle_contact_count_by_class": robot_obstacle_contacts,
            "tool_desk_min_contact_distance_m": finite_or_none(tool_desk_min_distance),
            "non_tool_robot_desk_min_contact_distance_m": finite_or_none(non_tool_robot_desk_min_distance),
            "block_desk_contact_count": block_desk_contact_count,
            "block_desk_signed_distance_m": block_desk_distance,
            "block_horizontal_support_margin_m": block_support_margin,
            "obstacle_horizontal_support_margin_m": obstacle_support_margin,
            "required_horizontal_support_margin_m": (self.contact_feasible_config.reset_support_margin_m),
            "block_support_mode": "dynamic_contact_on_edge_workbench",
            "obstacle_support_mode": ("world_fixed_static_geom" if self.obstacle_enabled else "disabled"),
            "reset_settle_substeps": self.contact_feasible_config.reset_settle_substeps,
            "block_settle_xy_displacement_m": block_settle_displacement_m,
            "block_settle_xy_tolerance_m": (self.contact_feasible_config.reset_block_settle_xy_tolerance_m),
            "requested_tool_xyz_m": target_xyz.tolist(),
            "actual_tool_xyz_m": actual_xyz.tolist(),
            "requested_tool_standoff_m": requested_standoff,
            "achieved_tool_standoff_m": achieved_standoff,
            "tool_standoff_error_m": standoff_error,
            "tool_standoff_error_tolerance_m": (
                self.contact_feasible_config.reset_standoff_error_tolerance_m
            ),
            "tool_lateral_error_m": lateral_error,
            "tool_lateral_error_tolerance_m": (self.contact_feasible_config.reset_lateral_error_tolerance_m),
            "tool_vertical_error_m": vertical_error,
            "tool_vertical_error_tolerance_m": (
                self.contact_feasible_config.reset_vertical_error_tolerance_m
            ),
            "tool_pose_position_error_m": pose_error,
            "pose_error_tolerance_m": pose_tolerance,
            "pose_error_within_tolerance": pose_error <= pose_tolerance,
            "ik_converged": ik_converged,
            "ik_joint_limit_active_mask": joint_limit_active.astype(np.uint8).tolist(),
            "ik_joint_limit_min_margin_rad": float(np.min(joint_limit_margin)),
            "tool_geom_center_xyz_m": self.data.geom_xpos[tool].astype(float).tolist(),
            "tool_geom_world_aabb_min_m": tool_aabb_min.tolist(),
            "tool_geom_world_aabb_max_m": tool_aabb_max.tolist(),
            "penetration_tolerance_m": tolerance,
        }


__all__ = [
    "CONTACT_FEASIBLE_DYNAMICS_PROFILE_VERSION",
    "CONTACT_FEASIBLE_GEOMETRY_VERSION",
    "CONTACT_FEASIBLE_RESET_SOURCE",
    "CONTACT_FEASIBLE_STATE_UPDATE",
    "RealisticEdgeArmEnvV7",
    "RealisticEnvV7Config",
]
