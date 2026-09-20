"""Deterministic collision-audited joint path planning for EdgeArm stock transit.

The planner operates only on scratch ``MjData`` instances.  It first checks the
direct edge, then runs a bounded bidirectional RRT-Connect in the five movable
arm joints, shortcuts the result, and finally performs an independent dense
audit.  V9 checks every convex stock-gripper safety part, while earlier model
profiles retain their authored safety geometry.  A static pass is not a dynamic
safety claim; physics-substep telemetry and its runtime desk guard remain
mandatory during execution.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from typing import Any

import mujoco
import numpy as np

from .sim2real_env_v7 import RealisticEdgeArmEnvV7


JOINT_PATH_FORMAT = "edgearm-deterministic-joint-rrt-connect-v1"
_MOVABLE_JOINTS = 5


def conservative_obb_separation_lower_bound(
    first_center: np.ndarray,
    first_rotation: np.ndarray,
    first_half_extent: np.ndarray,
    second_center: np.ndarray,
    second_rotation: np.ndarray,
    second_half_extent: np.ndarray,
) -> float:
    """Return the largest SAT projection gap, a safe clearance lower bound."""

    first_rotation = np.asarray(first_rotation, dtype=np.float64).reshape(3, 3)
    second_rotation = np.asarray(second_rotation, dtype=np.float64).reshape(3, 3)
    first_axes = first_rotation.T
    second_axes = second_rotation.T
    cross_axes = np.cross(first_axes[:, None, :], second_axes[None, :, :]).reshape(-1, 3)
    raw_axes = np.vstack([first_axes, second_axes, cross_axes])
    norms = np.linalg.norm(raw_axes, axis=1)
    axes = raw_axes[norms > 1.0e-12] / norms[norms > 1.0e-12, None]
    center_offset = np.asarray(second_center, dtype=np.float64) - np.asarray(
        first_center, dtype=np.float64
    )
    first_radius = np.abs(axes @ first_rotation) @ np.asarray(
        first_half_extent, dtype=np.float64
    )
    second_radius = np.abs(axes @ second_rotation) @ np.asarray(
        second_half_extent, dtype=np.float64
    )
    gaps = np.abs(axes @ center_offset) - first_radius - second_radius
    return float(np.max(gaps))


@dataclass(frozen=True)
class JointPathPlannerConfig:
    """Static planning and dense verification thresholds."""

    minimum_tool_block_clearance_m: float = 0.00025
    minimum_tool_desk_clearance_m: float = 0.002
    non_tool_desk_penetration_tolerance_m: float = 1.0e-9
    joint_limit_state_tolerance_rad: float = 0.001
    normalized_extend_step: float = 0.085
    planning_edge_resolution_rad: float = 0.012
    shortcut_edge_resolution_rad: float = 0.006
    dense_audit_resolution_rad: float = 0.0015
    maximum_iterations: int = 6_000
    shortcut_attempts: int = 300
    root_bias_probability: float = 0.18
    corridor_probability: float = 0.52

    def __post_init__(self) -> None:
        positive = (
            "minimum_tool_block_clearance_m",
            "minimum_tool_desk_clearance_m",
            "non_tool_desk_penetration_tolerance_m",
            "joint_limit_state_tolerance_rad",
            "normalized_extend_step",
            "planning_edge_resolution_rad",
            "shortcut_edge_resolution_rad",
            "dense_audit_resolution_rad",
            "maximum_iterations",
            "shortcut_attempts",
        )
        for name in positive:
            value = getattr(self, name)
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if self.planning_edge_resolution_rad < self.shortcut_edge_resolution_rad:
            raise ValueError("shortcut edges must be checked at least as densely as planning edges")
        if self.shortcut_edge_resolution_rad < self.dense_audit_resolution_rad:
            raise ValueError("dense audit must be at least as dense as shortcut checks")
        if not 0.0 <= self.root_bias_probability < 1.0:
            raise ValueError("root_bias_probability must be in [0, 1)")
        if not 0.0 <= self.corridor_probability < 1.0:
            raise ValueError("corridor_probability must be in [0, 1)")
        if self.root_bias_probability + self.corridor_probability >= 1.0:
            raise ValueError("root and corridor sampling probabilities must leave uniform mass")

    @property
    def profile_hash(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class _GeomClearance:
    geom_id: int
    geom_name: str
    role: str
    signed_distance_m: float


@dataclass(frozen=True)
class _StateAudit:
    valid: bool
    tool_block_clearance_m: float
    tool_desk_clearance_m: float
    tool_block_clearance_by_safety_geom: tuple[_GeomClearance, ...]
    tool_desk_clearance_by_safety_geom: tuple[_GeomClearance, ...]
    limiting_tool_block_safety_geom: _GeomClearance
    limiting_tool_desk_safety_geom: _GeomClearance
    minimum_non_tool_robot_desk_contact_distance_m: float
    robot_block_contact: bool
    robot_obstacle_contact: bool


@dataclass
class _Tree:
    root_kind: str
    joint_positions: list[np.ndarray]
    parents: list[int]

    @classmethod
    def root(cls, kind: str, joint_position: np.ndarray) -> _Tree:
        return cls(kind, [joint_position.copy()], [-1])

    def nearest(self, target: np.ndarray, scale: np.ndarray) -> int:
        values = np.asarray(self.joint_positions)
        return int(np.argmin(np.linalg.norm((values - target) / scale, axis=1)))

    def append(self, joint_position: np.ndarray, parent: int) -> int:
        self.joint_positions.append(joint_position.copy())
        self.parents.append(parent)
        return len(self.joint_positions) - 1

    def branch(self, index: int) -> list[np.ndarray]:
        result: list[np.ndarray] = []
        while index >= 0:
            result.append(self.joint_positions[index])
            index = self.parents[index]
        result.reverse()
        return result


class _CollisionChecker:
    def __init__(self, env: RealisticEdgeArmEnvV7, config: JointPathPlannerConfig) -> None:
        self.env = env
        self.config = config
        self.data = mujoco.MjData(env.model)
        self._fixed_qpos = env.data.qpos.copy()
        self._tool = env._ids["tool_geom"]
        self._tool_contact_geoms = frozenset(env._ids["tool_contact_geoms"])
        self._tool_safety_geoms = env._tool_safety_geom_ids()
        safety_roles = tuple(
            str(role) for role in env._ids.get("tool_safety_geom_roles", ())
        )
        if len(safety_roles) != len(self._tool_safety_geoms):
            safety_roles = tuple("tool_safety" for _ in self._tool_safety_geoms)
        self._tool_safety_roles = safety_roles
        self._tool_component_geoms = frozenset(
            (self._tool, *self._tool_contact_geoms, *self._tool_safety_geoms)
        )
        self._block = env._ids["block_geom"]
        self._desk = env._desk_geom
        self._obstacle = env._ids["obstacle_geom"]
        self._robot = set(env._robot_geoms)
        self._cache: dict[tuple[float, ...], _StateAudit] = {}
        self.unique_state_evaluations = 0
        self.exact_distance_fallbacks = 0

    def _clearance(self, first: int, second: int, required: float) -> float:
        box_type = int(mujoco.mjtGeom.mjGEOM_BOX)
        if (
            int(self.env.model.geom_type[first]) == box_type
            and int(self.env.model.geom_type[second]) == box_type
        ):
            lower_bound = conservative_obb_separation_lower_bound(
                self.data.geom_xpos[first],
                self.data.geom_xmat[first],
                self.env.model.geom_size[first],
                self.data.geom_xpos[second],
                self.data.geom_xmat[second],
                self.env.model.geom_size[second],
            )
            if lower_bound >= required:
                return lower_bound
        self.exact_distance_fallbacks += 1
        return float(self.env._geom_signed_distance_for_data(first, second, self.data))

    def _tool_safety_clearances(
        self,
        other: int,
        required: float,
    ) -> tuple[_GeomClearance, ...]:
        rows: list[_GeomClearance] = []
        for geom_id, role in zip(
            self._tool_safety_geoms,
            self._tool_safety_roles,
            strict=True,
        ):
            rows.append(
                _GeomClearance(
                    geom_id=geom_id,
                    geom_name=(
                        mujoco.mj_id2name(
                            self.env.model,
                            mujoco.mjtObj.mjOBJ_GEOM,
                            geom_id,
                        )
                        or f"geom_{geom_id}"
                    ),
                    role=role,
                    signed_distance_m=self._clearance(geom_id, other, required),
                )
            )
        return tuple(rows)

    def state(self, joint_position: np.ndarray) -> _StateAudit:
        key = tuple(np.round(joint_position, 8))
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        self.unique_state_evaluations += 1
        self.data.qpos[:] = self._fixed_qpos
        self.data.qvel[:] = 0.0
        self.data.qpos[:_MOVABLE_JOINTS] = joint_position
        self.data.qpos[5] = self.env.tool_gripper_joint_position_rad
        mujoco.mj_forward(self.env.model, self.data)
        tool_block_rows = self._tool_safety_clearances(
            self._block,
            self.config.minimum_tool_block_clearance_m,
        )
        tool_desk_rows = self._tool_safety_clearances(
            self._desk,
            self.config.minimum_tool_desk_clearance_m,
        )
        limiting_tool_block = min(
            tool_block_rows,
            key=lambda row: row.signed_distance_m,
        )
        limiting_tool_desk = min(
            tool_desk_rows,
            key=lambda row: row.signed_distance_m,
        )
        tool_block = limiting_tool_block.signed_distance_m
        tool_desk = limiting_tool_desk.signed_distance_m
        minimum_non_tool_desk = math.inf
        robot_block_contact = False
        robot_obstacle_contact = False
        for contact_index in range(self.data.ncon):
            contact = self.data.contact[contact_index]
            first = int(contact.geom1)
            second = int(contact.geom2)
            distance = float(contact.dist)
            pair = {first, second}
            if self._desk in pair:
                other = second if first == self._desk else first
                if other in self._robot - self._tool_component_geoms:
                    minimum_non_tool_desk = min(minimum_non_tool_desk, distance)
            if self._block in pair:
                other = second if first == self._block else first
                if other in self._robot and distance <= 0.0:
                    robot_block_contact = True
            if self.env.obstacle_enabled and self._obstacle in pair:
                other = second if first == self._obstacle else first
                if other in self._robot and distance <= 0.0:
                    robot_obstacle_contact = True
        valid = bool(
            tool_block >= self.config.minimum_tool_block_clearance_m
            and tool_desk >= self.config.minimum_tool_desk_clearance_m
            and minimum_non_tool_desk
            >= -self.config.non_tool_desk_penetration_tolerance_m
            and not robot_block_contact
            and not robot_obstacle_contact
        )
        audit = _StateAudit(
            valid=valid,
            tool_block_clearance_m=tool_block,
            tool_desk_clearance_m=tool_desk,
            tool_block_clearance_by_safety_geom=tool_block_rows,
            tool_desk_clearance_by_safety_geom=tool_desk_rows,
            limiting_tool_block_safety_geom=limiting_tool_block,
            limiting_tool_desk_safety_geom=limiting_tool_desk,
            minimum_non_tool_robot_desk_contact_distance_m=minimum_non_tool_desk,
            robot_block_contact=robot_block_contact,
            robot_obstacle_contact=robot_obstacle_contact,
        )
        self._cache[key] = audit
        return audit

    def edge(self, first: np.ndarray, second: np.ndarray, resolution: float) -> bool:
        sample_count = max(
            1,
            int(math.ceil(float(np.max(np.abs(second - first))) / resolution)),
        )
        for index in range(1, sample_count + 1):
            joint_position = first + (index / sample_count) * (second - first)
            if not self.state(joint_position).valid:
                return False
        return True


class JointPathPlannerV1:
    """Plan and independently audit a static five-joint transit path."""

    def __init__(
        self,
        env: RealisticEdgeArmEnvV7,
        config: JointPathPlannerConfig | None = None,
    ) -> None:
        if not isinstance(env, RealisticEdgeArmEnvV7):
            raise TypeError("JointPathPlannerV1 requires RealisticEdgeArmEnvV7")
        self.env = env
        self.config = config or JointPathPlannerConfig()
        self._lower = np.asarray(env.joint_ranges[:_MOVABLE_JOINTS, 0], dtype=np.float64)
        self._upper = np.asarray(env.joint_ranges[:_MOVABLE_JOINTS, 1], dtype=np.float64)
        self._scale = self._upper - self._lower

    def solve(self, start: np.ndarray, goal: np.ndarray) -> dict[str, Any]:
        start_q = self._validate_joint_position(start, "start")
        goal_q = self._validate_joint_position(goal, "goal")
        checker = _CollisionChecker(self.env, self.config)
        start_audit = checker.state(start_q)
        goal_audit = checker.state(goal_q)
        if not start_audit.valid or not goal_audit.valid:
            return self._failure_report(
                "invalid_endpoint",
                checker,
                start_audit=start_audit,
                goal_audit=goal_audit,
            )
        planner_seed = self._planner_seed(start_q, goal_q)
        raw_path: list[np.ndarray] | None
        planner_stats: dict[str, int | str]
        if checker.edge(start_q, goal_q, self.config.planning_edge_resolution_rad):
            raw_path = [start_q, goal_q]
            planner_stats = {
                "method": "direct_edge",
                "iterations": 0,
                "nodes_start_tree": 1,
                "nodes_goal_tree": 1,
            }
        else:
            raw_path, planner_stats = self._rrt_connect(
                start_q,
                goal_q,
                checker,
                planner_seed,
            )
        if raw_path is None:
            return self._failure_report(
                "rrt_connect_exhausted",
                checker,
                start_audit=start_audit,
                goal_audit=goal_audit,
                planner_stats=planner_stats,
            )
        path = self._shortcut(raw_path, checker, planner_seed)
        audit = self._dense_audit(path, checker)
        if not audit["valid"]:
            return self._failure_report(
                "dense_path_audit_failed",
                checker,
                start_audit=start_audit,
                goal_audit=goal_audit,
                planner_stats=planner_stats,
                dense_audit=audit,
            )
        return {
            **self._provenance(),
            "feasible": True,
            "failure_code": "",
            "planner_seed": planner_seed,
            "raw_path_nodes": len(raw_path),
            "shortcut_path_nodes": len(path),
            "joint_waypoints_rad": [
                np.r_[joint_position, self.env.tool_gripper_joint_position_rad].tolist()
                for joint_position in path
            ],
            "planner": planner_stats,
            "dense_audit": audit,
            "checker_unique_state_evaluations": checker.unique_state_evaluations,
            "checker_exact_distance_fallbacks": checker.exact_distance_fallbacks,
        }

    def _validate_joint_position(self, value: np.ndarray, name: str) -> np.ndarray:
        array = np.asarray(value, dtype=np.float64)
        if array.shape == (6,):
            array = array[:5]
        if array.shape != (_MOVABLE_JOINTS,) or not np.all(np.isfinite(array)):
            raise ValueError(f"{name} must contain five or six finite joints")
        tolerance = self.config.joint_limit_state_tolerance_rad
        if np.any(array < self._lower - tolerance) or np.any(array > self._upper + tolerance):
            raise ValueError(f"{name} lies outside the authored joint limits")
        return array.copy()

    def _planner_seed(self, start: np.ndarray, goal: np.ndarray) -> int:
        payload = np.r_[np.round(start, 8), np.round(goal, 8)].astype("<f8", copy=False)
        return int.from_bytes(hashlib.sha256(payload.tobytes()).digest()[:8], "little")

    def _steer(self, source: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, bool]:
        normalized_delta = (target - source) / self._scale
        distance = float(np.linalg.norm(normalized_delta))
        if distance <= self.config.normalized_extend_step:
            return target.copy(), True
        step = normalized_delta * (self.config.normalized_extend_step / distance)
        return source + step * self._scale, False

    def _extend(
        self,
        tree: _Tree,
        target: np.ndarray,
        checker: _CollisionChecker,
    ) -> tuple[str, int | None]:
        nearest = tree.nearest(target, self._scale)
        candidate, reached = self._steer(tree.joint_positions[nearest], target)
        if not checker.edge(
            tree.joint_positions[nearest],
            candidate,
            self.config.planning_edge_resolution_rad,
        ):
            return "trapped", None
        index = tree.append(candidate, nearest)
        return ("reached" if reached else "advanced"), index

    def _connect(
        self,
        tree: _Tree,
        target: np.ndarray,
        checker: _CollisionChecker,
    ) -> tuple[str, int | None]:
        last_index: int | None = None
        while True:
            status, index = self._extend(tree, target, checker)
            if index is not None:
                last_index = index
            if status != "advanced":
                return status, last_index

    @staticmethod
    def _assemble(
        first: _Tree,
        first_index: int,
        second: _Tree,
        second_index: int,
    ) -> list[np.ndarray]:
        first_branch = first.branch(first_index)
        second_branch = second.branch(second_index)
        if first.root_kind == "start":
            start_branch, goal_branch = first_branch, second_branch
        else:
            start_branch, goal_branch = second_branch, first_branch
        return start_branch + list(reversed(goal_branch))[1:]

    def _rrt_connect(
        self,
        start: np.ndarray,
        goal: np.ndarray,
        checker: _CollisionChecker,
        seed: int,
    ) -> tuple[list[np.ndarray] | None, dict[str, int | str]]:
        rng = np.random.default_rng(seed ^ 0x52A771)
        first = _Tree.root("start", start)
        second = _Tree.root("goal", goal)
        root_boundary = self.config.root_bias_probability
        corridor_boundary = root_boundary + self.config.corridor_probability
        for iteration in range(1, self.config.maximum_iterations + 1):
            draw = float(rng.random())
            if draw < root_boundary:
                sample = second.joint_positions[0]
            elif draw < corridor_boundary:
                fraction = float(rng.random())
                center = start + fraction * (goal - start)
                sigma = self._scale * (0.06 + 0.18 * math.sin(math.pi * fraction))
                sample = np.clip(center + rng.normal(0.0, sigma), self._lower, self._upper)
            else:
                sample = rng.uniform(self._lower, self._upper)
            status, first_index = self._extend(first, sample, checker)
            if status != "trapped" and first_index is not None:
                second_status, second_index = self._connect(
                    second,
                    first.joint_positions[first_index],
                    checker,
                )
                if second_status == "reached" and second_index is not None:
                    return self._assemble(first, first_index, second, second_index), {
                        "method": "bidirectional_rrt_connect",
                        "iterations": iteration,
                        "nodes_start_tree": (
                            len(first.joint_positions)
                            if first.root_kind == "start"
                            else len(second.joint_positions)
                        ),
                        "nodes_goal_tree": (
                            len(second.joint_positions)
                            if second.root_kind == "goal"
                            else len(first.joint_positions)
                        ),
                    }
            first, second = second, first
        return None, {
            "method": "bidirectional_rrt_connect",
            "iterations": self.config.maximum_iterations,
            "nodes_start_tree": len(first.joint_positions),
            "nodes_goal_tree": len(second.joint_positions),
        }

    def _shortcut(
        self,
        path: list[np.ndarray],
        checker: _CollisionChecker,
        seed: int,
    ) -> list[np.ndarray]:
        rng = np.random.default_rng(seed ^ 0xC07C47)
        result = [joint_position.copy() for joint_position in path]
        for _ in range(self.config.shortcut_attempts):
            if len(result) <= 2:
                break
            first, second = sorted(rng.integers(0, len(result), size=2).tolist())
            if second <= first + 1:
                continue
            if checker.edge(
                result[first],
                result[second],
                self.config.shortcut_edge_resolution_rad,
            ):
                result = result[: first + 1] + result[second:]
        return result

    def _dense_audit(
        self,
        path: list[np.ndarray],
        checker: _CollisionChecker,
    ) -> dict[str, Any]:
        audits = [checker.state(path[0])]
        sample_count = 1
        for first, second in zip(path, path[1:]):
            count = max(
                1,
                int(
                    math.ceil(
                        float(np.max(np.abs(second - first)))
                        / self.config.dense_audit_resolution_rad
                    )
                ),
            )
            sample_count += count
            for index in range(1, count + 1):
                audits.append(checker.state(first + (index / count) * (second - first)))
        finite_non_tool_desk = [
            audit.minimum_non_tool_robot_desk_contact_distance_m
            for audit in audits
            if np.isfinite(audit.minimum_non_tool_robot_desk_contact_distance_m)
        ]
        block_rows = self._minimum_safety_clearance_rows(
            audits,
            "tool_block_clearance_by_safety_geom",
        )
        desk_rows = self._minimum_safety_clearance_rows(
            audits,
            "tool_desk_clearance_by_safety_geom",
        )
        return {
            "valid": all(audit.valid for audit in audits),
            "sample_count": sample_count,
            "maximum_joint_increment_audited_rad": self.config.dense_audit_resolution_rad,
            "minimum_tool_block_clearance_m": min(
                audit.tool_block_clearance_m for audit in audits
            ),
            "minimum_tool_desk_clearance_m": min(
                audit.tool_desk_clearance_m for audit in audits
            ),
            "minimum_non_tool_robot_desk_contact_distance_m": (
                min(finite_non_tool_desk) if finite_non_tool_desk else None
            ),
            "tool_block_minimum_clearance_by_safety_geom": block_rows,
            "tool_desk_minimum_clearance_by_safety_geom": desk_rows,
            "limiting_tool_block_safety_geom": min(
                block_rows,
                key=lambda row: row["minimum_signed_distance_m"],
            ),
            "limiting_tool_desk_safety_geom": min(
                desk_rows,
                key=lambda row: row["minimum_signed_distance_m"],
            ),
            "robot_block_contact_any": any(audit.robot_block_contact for audit in audits),
            "robot_obstacle_contact_any": any(audit.robot_obstacle_contact for audit in audits),
        }

    @staticmethod
    def _minimum_safety_clearance_rows(
        audits: list[_StateAudit],
        attribute: str,
    ) -> list[dict[str, Any]]:
        first_rows = getattr(audits[0], attribute)
        minima = {
            row.geom_id: row.signed_distance_m
            for row in first_rows
        }
        for audit in audits[1:]:
            rows = getattr(audit, attribute)
            if tuple(row.geom_id for row in rows) != tuple(minima):
                raise RuntimeError("tool safety geometry ordering changed during path audit")
            for row in rows:
                minima[row.geom_id] = min(minima[row.geom_id], row.signed_distance_m)
        return [
            {
                "geom_id": row.geom_id,
                "geom_name": row.geom_name,
                "role": row.role,
                "minimum_signed_distance_m": float(minima[row.geom_id]),
            }
            for row in first_rows
        ]

    def _failure_report(
        self,
        failure_code: str,
        checker: _CollisionChecker,
        **evidence: Any,
    ) -> dict[str, Any]:
        def serialize(value: Any) -> Any:
            if isinstance(value, _StateAudit):
                row = asdict(value)
                if not np.isfinite(row["minimum_non_tool_robot_desk_contact_distance_m"]):
                    row["minimum_non_tool_robot_desk_contact_distance_m"] = None
                return row
            return value

        return {
            **self._provenance(),
            "feasible": False,
            "failure_code": failure_code,
            "joint_waypoints_rad": [],
            "checker_unique_state_evaluations": checker.unique_state_evaluations,
            "checker_exact_distance_fallbacks": checker.exact_distance_fallbacks,
            **{name: serialize(value) for name, value in evidence.items()},
        }

    def _provenance(self) -> dict[str, Any]:
        return {
            "format": JOINT_PATH_FORMAT,
            "claim_level": "L1_SYNTHETIC_HARDWARE_INSPIRED",
            "parameter_source": "synthetic_scratch_joint_space_rrt_connect",
            "simulator_privileged_truth": True,
            "physical_samples": 0,
            "physical_trials": 0,
            "physically_calibrated": False,
            "physical_hardware_connected": False,
            "live_state_written": False,
            "tool_gripper_joint_position_rad": float(
                self.env.tool_gripper_joint_position_rad
            ),
            "static_path_only": True,
            "dynamic_execution_safety_claim": False,
            "configuration_hash": self.config.profile_hash,
            "configuration": asdict(self.config),
            "clearance_method": (
                "all_tool_safety_geoms_with_box_sat_lower_bound_and_environment_"
                "generic_signed_distance_fallback_for_convex_meshes"
            ),
            "tool_safety_geometry_mode": self.env._ids.get(
                "tool_safety_geometry_mode",
                "legacy_single_tool_geom",
            ),
        }


__all__ = [
    "JOINT_PATH_FORMAT",
    "JointPathPlannerConfig",
    "JointPathPlannerV1",
    "conservative_obb_separation_lower_bound",
]
