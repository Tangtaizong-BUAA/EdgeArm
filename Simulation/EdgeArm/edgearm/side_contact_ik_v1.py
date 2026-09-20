"""Deterministic five-constraint IK for an SO-101-style pushing tool.

The first five joints can control tool position (three constraints) and the
direction of the tool's thin local-Y axis (two independent constraints).  A
full quaternion would overconstrain this arm.  This solver therefore aligns
the physical pushing-face normal, searches both equivalent faces, preserves a
hard joint-limit margin, and reports feasibility instead of silently returning
a saturated pose.  Stock-gripper profiles are accepted only when every
distal-jaw planning reference satisfies the pre-contact gap and central
side-height gates, while desk clearance is evaluated over the complete tool
safety geometry.

Every candidate is evaluated in a scratch ``MjData``.  No live simulator state
is written and all thresholds remain uncalibrated synthetic engineering priors.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any

import mujoco
import numpy as np

from .sim2real_env_v7 import RealisticEdgeArmEnvV7


SIDE_CONTACT_IK_FORMAT = "edgearm-five-constraint-side-contact-ik-v2-contact-envelope"
_MOVABLE_JOINTS = 5
_ALL_JOINTS = 6


@dataclass(frozen=True)
class SideContactIKConfig:
    """Bounded deterministic search and acceptance thresholds."""

    position_scale_m: float = 0.010
    normal_scale_candidates: tuple[float, ...] = (0.08, 0.20)
    maximum_iterations: int = 90
    initial_damping: float = 1.0e-3
    maximum_joint_step_rad: float = 0.12
    line_search_steps: int = 5
    joint_margin_fraction: float = 0.015
    maximum_position_error_m: float = 0.003
    # A 50 um numerical hysteresis prevents a valid force-limited pose from
    # toggling between feasible/infeasible at the solver boundary.  Normal,
    # desk-clearance and actual-motion constraints remain independent gates.
    tracking_maximum_position_error_m: float = 0.00085
    minimum_normal_axis_alignment: float = 0.98
    minimum_normal_heading_alignment: float = 0.99
    minimum_normal_horizontal_norm: float = 0.98
    minimum_pusher_desk_clearance_m: float = 0.002
    # Optional full-tool safety-union clearance from the block.  The legacy
    # V7/V8 controllers leave this at zero and retain their previous
    # acceptance semantics.  Stock-gripper V9 binds it to the joint path
    # planner's positive endpoint clearance so IK cannot return a goal which
    # the very next planning stage must reject.
    minimum_tool_block_safety_clearance_m: float = 0.0
    minimum_pusher_block_gap_m: float = 0.00025
    maximum_pusher_block_gap_m: float = 0.008
    preferred_pusher_block_gap_m: float = 0.0015
    block_top_bottom_exclusion_fraction: float = 0.20
    minimum_tip_central_side_overlap_m: float = 0.001
    # Minimum distance from every authoritative contact-part vertex envelope
    # to either boundary of the block's central side band.  Zero preserves
    # historical V7/V8 acceptance; V12 binds a positive dynamic reserve.
    minimum_contact_part_central_side_margin_m: float = 0.0
    standoff_offsets_m: tuple[float, ...] = (0.0, -0.002, 0.002, -0.004, 0.004)
    height_candidates_m: tuple[float, ...] = (0.08525, 0.082, 0.086, 0.080, 0.088)
    lateral_candidates_m: tuple[float, ...] = (0.0, 0.030, -0.030, 0.040, -0.040, 0.020, -0.020)
    deterministic_random_starts: int = 2
    maximum_target_candidates: int = 175

    def __post_init__(self) -> None:
        positive_names = (
            "position_scale_m",
            "maximum_iterations",
            "initial_damping",
            "maximum_joint_step_rad",
            "line_search_steps",
            "maximum_position_error_m",
            "tracking_maximum_position_error_m",
            "minimum_pusher_desk_clearance_m",
            "minimum_pusher_block_gap_m",
            "maximum_pusher_block_gap_m",
            "preferred_pusher_block_gap_m",
            "minimum_tip_central_side_overlap_m",
            "maximum_target_candidates",
        )
        for name in positive_names:
            value = getattr(self, name)
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if (
            not np.isfinite(self.minimum_tool_block_safety_clearance_m)
            or self.minimum_tool_block_safety_clearance_m < 0.0
        ):
            raise ValueError(
                "minimum_tool_block_safety_clearance_m must be finite and non-negative"
            )
        if not 0.0 < self.joint_margin_fraction < 0.10:
            raise ValueError("joint_margin_fraction must be in (0, 0.10)")
        for name in (
            "minimum_normal_axis_alignment",
            "minimum_normal_heading_alignment",
            "minimum_normal_horizontal_norm",
        ):
            value = float(getattr(self, name))
            if not 0.0 < value <= 1.0:
                raise ValueError(f"{name} must be in (0, 1]")
        if self.minimum_pusher_block_gap_m >= self.maximum_pusher_block_gap_m:
            raise ValueError("pusher/block gap interval is empty")
        if not 0.0 <= self.block_top_bottom_exclusion_fraction < 1.0:
            raise ValueError("block_top_bottom_exclusion_fraction must be in [0, 1)")
        if (
            not np.isfinite(self.minimum_contact_part_central_side_margin_m)
            or self.minimum_contact_part_central_side_margin_m < 0.0
        ):
            raise ValueError(
                "minimum_contact_part_central_side_margin_m must be finite and non-negative"
            )
        if self.tracking_maximum_position_error_m >= self.maximum_position_error_m:
            raise ValueError(
                "tracking_maximum_position_error_m must be below maximum_position_error_m"
            )
        if not (
            self.minimum_pusher_block_gap_m
            <= self.preferred_pusher_block_gap_m
            <= self.maximum_pusher_block_gap_m
        ):
            raise ValueError("preferred pusher/block gap must lie inside the accepted interval")
        if self.deterministic_random_starts < 0:
            raise ValueError("deterministic_random_starts must be non-negative")
        tuple_names = (
            "normal_scale_candidates",
            "standoff_offsets_m",
            "height_candidates_m",
            "lateral_candidates_m",
        )
        for name in tuple_names:
            values = tuple(float(value) for value in getattr(self, name))
            if not values or not np.all(np.isfinite(values)):
                raise ValueError(f"{name} must be a non-empty finite tuple")
        if any(value <= 0.0 for value in self.normal_scale_candidates):
            raise ValueError("normal_scale_candidates must be positive")

    @property
    def profile_hash(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class SideContactIKPlannerV1:
    """Search a reachable collision-audited pre-side-contact joint pose."""

    def __init__(
        self,
        env: RealisticEdgeArmEnvV7,
        config: SideContactIKConfig | None = None,
    ) -> None:
        if not isinstance(env, RealisticEdgeArmEnvV7):
            raise TypeError("SideContactIKPlannerV1 requires RealisticEdgeArmEnvV7")
        self.env = env
        self.config = config or SideContactIKConfig()
        self._scratch = mujoco.MjData(env.model)
        self._evaluation_scratch = mujoco.MjData(env.model)
        ranges = np.asarray(env.joint_ranges[:_MOVABLE_JOINTS], dtype=np.float64)
        span = np.ptp(ranges, axis=1)
        margin = self.config.joint_margin_fraction * span
        self._lower = ranges[:, 0] + margin
        self._upper = ranges[:, 1] - margin
        self._joint_span = span

    def solve(
        self,
        intended_push_direction_xy: np.ndarray,
        *,
        initial: np.ndarray | None = None,
    ) -> dict[str, Any]:
        """Return the best structured candidate without mutating live state."""

        direction = np.asarray(intended_push_direction_xy, dtype=np.float64)
        if direction.shape != (2,) or not np.all(np.isfinite(direction)):
            raise ValueError("intended_push_direction_xy must be a finite two-vector")
        norm = float(np.linalg.norm(direction))
        if norm <= 1.0e-12:
            raise ValueError("intended_push_direction_xy must be non-zero")
        direction = direction / norm
        side = np.array([-direction[1], direction[0]], dtype=np.float64)
        block_xy = self.env.block_xy()
        block_rotation = self.env.data.geom_xmat[self.env._ids["block_geom"]].reshape(3, 3)
        block_half = self.env.model.geom_size[self.env._ids["block_geom"]]
        direction_world = np.array([direction[0], direction[1], 0.0], dtype=np.float64)
        block_support = float(np.dot(np.abs(block_rotation.T @ direction_world), block_half))
        tool_face_support = float(
            self.env._tool_face_support_radius_for_data(self.env.data)
        )
        nominal_standoff = (
            block_support
            + tool_face_support
            + self.config.preferred_pusher_block_gap_m
        )
        starts = self._initial_starts(direction, initial)
        candidates: list[dict[str, Any]] = []
        target_count = 0

        target_grid = [
            (float(height), float(lateral), float(standoff_offset))
            for height in self.config.height_candidates_m
            for lateral in self.config.lateral_candidates_m
            for standoff_offset in self.config.standoff_offsets_m
        ]
        nominal_height = float(self.config.height_candidates_m[0])

        def target_priority(values: tuple[float, float, float]) -> tuple[float, ...]:
            height, lateral, standoff_offset = values
            changed = int(not np.isclose(height, nominal_height))
            changed += int(not np.isclose(lateral, 0.0))
            changed += int(not np.isclose(standoff_offset, 0.0))
            normalized_change = abs(height - nominal_height) / 0.004
            normalized_change += abs(lateral) / 0.030
            normalized_change += abs(standoff_offset) / 0.002
            return changed, normalized_change, abs(lateral), abs(height - nominal_height)

        ordered_targets = sorted(target_grid, key=target_priority)
        search_passes = (
            ((1.0,), starts[:1], "fast_current_branch"),
            ((1.0, -1.0), starts, "robust_multistart_both_faces"),
        )
        for normal_signs, pass_starts, search_pass in search_passes:
            pass_target_count = 0
            for height, lateral, standoff_offset in ordered_targets:
                if pass_target_count >= self.config.maximum_target_candidates:
                    break
                pass_target_count += 1
                target_count += 1
                standoff = nominal_standoff + standoff_offset
                target_xyz = np.array(
                    [
                        block_xy[0] - direction[0] * standoff + side[0] * lateral,
                        block_xy[1] - direction[1] * standoff + side[1] * lateral,
                        height,
                    ],
                    dtype=np.float64,
                )
                for normal_sign in normal_signs:
                    desired_normal = normal_sign * direction_world
                    for normal_scale in self.config.normal_scale_candidates:
                        for start_index, start in enumerate(pass_starts):
                            joint_position, solver = self._solve_local(
                                target_xyz,
                                desired_normal,
                                start,
                                normal_scale=normal_scale,
                            )
                            candidate = self._evaluate(
                                joint_position,
                                target_xyz,
                                direction,
                                standoff_m=standoff,
                                height_m=height,
                                lateral_m=lateral,
                                normal_sign=normal_sign,
                                normal_scale=normal_scale,
                                start_index=start_index,
                                solver={**solver, "search_pass": search_pass},
                            )
                            candidates.append(candidate)
                            if candidate["feasible"]:
                                return self._report(
                                    candidate,
                                    candidates,
                                    direction,
                                    block_support,
                                    nominal_standoff,
                                    target_count,
                                )
        best = min(candidates, key=self._rank)
        return self._report(
            best,
            candidates,
            direction,
            block_support,
            nominal_standoff,
            target_count,
        )

    def track_target(
        self,
        target_xyz: np.ndarray,
        intended_push_direction_xy: np.ndarray,
        *,
        initial: np.ndarray,
    ) -> dict[str, Any]:
        """Track one fixed tool target while preserving the reachable face normal.

        This kinematic helper intentionally does not apply block-clearance gates:
        during a verified push the plate and block are already in contact.  The
        caller remains responsible for contact semantics and runtime safety.
        """

        target = np.asarray(target_xyz, dtype=np.float64)
        direction = np.asarray(intended_push_direction_xy, dtype=np.float64)
        initial_q = np.asarray(initial, dtype=np.float64)
        if target.shape != (3,) or not np.all(np.isfinite(target)):
            raise ValueError("target_xyz must be a finite three-vector")
        if direction.shape != (2,) or not np.all(np.isfinite(direction)):
            raise ValueError("intended_push_direction_xy must be a finite two-vector")
        direction_norm = float(np.linalg.norm(direction))
        if direction_norm <= 1.0e-12:
            raise ValueError("intended_push_direction_xy must be non-zero")
        if initial_q.shape not in {(_MOVABLE_JOINTS,), (_ALL_JOINTS,)} or not np.all(
            np.isfinite(initial_q)
        ):
            raise ValueError("initial must contain five or six finite joints")
        direction = direction / direction_norm
        current_q = np.clip(initial_q[:_MOVABLE_JOINTS], self._lower, self._upper)
        self._set_scratch(self._scratch, current_q)
        current_normal = self._scratch.site_xmat[self.env._ids["tool_site"]].reshape(3, 3)[:, 1]
        direction_world = np.r_[direction, 0.0]
        preferred_sign = 1.0 if float(np.dot(current_normal, direction_world)) >= 0.0 else -1.0
        candidates: list[dict[str, Any]] = []
        for normal_sign in (preferred_sign, -preferred_sign):
            desired_normal = normal_sign * direction_world
            for normal_scale in self.config.normal_scale_candidates:
                joint_position, solver = self._solve_local(
                    target,
                    desired_normal,
                    current_q,
                    normal_scale=normal_scale,
                    position_tolerance_m=(
                        0.75 * self.config.tracking_maximum_position_error_m
                    ),
                )
                self._set_scratch(self._evaluation_scratch, joint_position)
                actual_xyz = self._evaluation_scratch.site_xpos[self.env._ids["tool_site"]].copy()
                normal = self._evaluation_scratch.site_xmat[
                    self.env._ids["tool_site"]
                ].reshape(3, 3)[:, 1]
                pusher_desk_clearance = float(
                    self.env._minimum_tool_safety_signed_distance_for_data(
                        self.env._desk_geom,
                        self._evaluation_scratch,
                    )
                )
                horizontal_norm = float(np.linalg.norm(normal[:2]))
                signed_alignment = float(np.dot(normal[:2], direction))
                axis_alignment = abs(signed_alignment)
                candidate = {
                    "joint_position_rad": np.r_[
                        joint_position,
                        self.env.tool_gripper_joint_position_rad,
                    ].tolist(),
                    "target_xyz_m": target.tolist(),
                    "actual_xyz_m": actual_xyz.tolist(),
                    "position_error_m": float(np.linalg.norm(target - actual_xyz)),
                    "tool_face_normal_world": normal.tolist(),
                    "normal_sign": normal_sign,
                    "normal_scale": normal_scale,
                    "normal_axis_alignment": axis_alignment,
                    "normal_horizontal_norm": horizontal_norm,
                    "normal_heading_alignment": axis_alignment / max(horizontal_norm, 1.0e-12),
                    "pusher_desk_signed_distance_m": pusher_desk_clearance,
                    "solver": solver,
                }
                candidates.append(candidate)
                if (
                    candidate["position_error_m"]
                    <= self.config.tracking_maximum_position_error_m
                    and candidate["normal_axis_alignment"]
                    >= self.config.minimum_normal_axis_alignment
                    and candidate["normal_horizontal_norm"]
                    >= self.config.minimum_normal_horizontal_norm
                    and candidate["pusher_desk_signed_distance_m"]
                    >= self.config.minimum_pusher_desk_clearance_m
                ):
                    return {
                        "format": SIDE_CONTACT_IK_FORMAT,
                        "live_state_written": False,
                        "tool_gripper_joint_position_rad": float(
                            self.env.tool_gripper_joint_position_rad
                        ),
                        "feasible": True,
                        "best": candidate,
                        "candidate_count": len(candidates),
                    }
        best = min(
            candidates,
            key=lambda candidate: (
                candidate["position_error_m"],
                -candidate["normal_axis_alignment"],
                -candidate["normal_horizontal_norm"],
            ),
        )
        return {
            "format": SIDE_CONTACT_IK_FORMAT,
            "live_state_written": False,
            "tool_gripper_joint_position_rad": float(
                self.env.tool_gripper_joint_position_rad
            ),
            "feasible": False,
            "best": best,
            "candidate_count": len(candidates),
        }

    def _initial_starts(
        self,
        direction: np.ndarray,
        initial: np.ndarray | None,
    ) -> list[np.ndarray]:
        source = self.env.data.qpos[:_MOVABLE_JOINTS] if initial is None else np.asarray(initial)[:5]
        if source.shape != (_MOVABLE_JOINTS,) or not np.all(np.isfinite(source)):
            raise ValueError("initial joint position must contain five finite movable joints")
        midpoint = 0.5 * (self._lower + self._upper)
        canonical = np.array([0.49, -0.52, 0.47, 1.608, -1.13], dtype=np.float64)
        mirrored = np.array([-0.52, -0.40, 0.35, 1.608, -1.93], dtype=np.float64)
        heading = float(np.arctan2(direction[1], direction[0]))
        canonical[0] = np.clip(canonical[0] + heading, self._lower[0], self._upper[0])
        mirrored[0] = np.clip(mirrored[0] + heading, self._lower[0], self._upper[0])
        starts = [
            np.clip(source, self._lower, self._upper),
            midpoint,
            np.clip(canonical, self._lower, self._upper),
            np.clip(mirrored, self._lower, self._upper),
        ]
        seed_payload = np.r_[np.round(self.env.block_xy(), 6), np.round(direction, 6)]
        seed_bytes = seed_payload.astype("<f8", copy=False).tobytes()
        local_seed = int.from_bytes(hashlib.sha256(seed_bytes).digest()[:8], "little")
        rng = np.random.default_rng(local_seed)
        for _ in range(self.config.deterministic_random_starts):
            starts.append(
                self._lower
                + rng.uniform(0.08, 0.92, _MOVABLE_JOINTS) * (self._upper - self._lower)
            )
        return starts

    def _set_scratch(self, data: mujoco.MjData, joint_position: np.ndarray) -> None:
        data.qpos[:] = self.env.data.qpos
        data.qvel[:] = 0.0
        data.qpos[:_MOVABLE_JOINTS] = joint_position
        data.qpos[5] = self.env.tool_gripper_joint_position_rad
        mujoco.mj_forward(self.env.model, data)

    def _residual_and_jacobian(
        self,
        joint_position: np.ndarray,
        target_xyz: np.ndarray,
        desired_normal: np.ndarray,
        normal_scale: float,
        *,
        with_jacobian: bool,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        self._set_scratch(self._scratch, joint_position)
        site = self.env._ids["tool_site"]
        current_xyz = np.asarray(self._scratch.site_xpos[site], dtype=np.float64)
        current_normal = self._scratch.site_xmat[site].reshape(3, 3)[:, 1]
        residual = np.r_[
            (target_xyz - current_xyz) / self.config.position_scale_m,
            (desired_normal - current_normal) / normal_scale,
        ]
        if not with_jacobian:
            return residual, None
        jacobian_position = np.zeros((3, self.env.model.nv), dtype=np.float64)
        jacobian_rotation = np.zeros((3, self.env.model.nv), dtype=np.float64)
        mujoco.mj_jacSite(
            self.env.model,
            self._scratch,
            jacobian_position,
            jacobian_rotation,
            site,
        )
        normal_jacobian = np.cross(
            jacobian_rotation[:, :_MOVABLE_JOINTS].T,
            current_normal,
        ).T
        jacobian = np.vstack(
            [
                jacobian_position[:, :_MOVABLE_JOINTS] / self.config.position_scale_m,
                normal_jacobian / normal_scale,
            ]
        )
        return residual, jacobian

    def _solve_local(
        self,
        target_xyz: np.ndarray,
        desired_normal: np.ndarray,
        start: np.ndarray,
        *,
        normal_scale: float,
        position_tolerance_m: float = 0.0025,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        q = np.clip(np.asarray(start, dtype=np.float64), self._lower, self._upper)
        damping = self.config.initial_damping
        accepted_steps = 0
        iterations = 0
        for iterations in range(1, self.config.maximum_iterations + 1):
            residual, jacobian = self._residual_and_jacobian(
                q,
                target_xyz,
                desired_normal,
                normal_scale,
                with_jacobian=True,
            )
            if jacobian is None:  # pragma: no cover
                raise RuntimeError("side-contact IK Jacobian was not computed")
            current_normal = desired_normal - residual[3:] * normal_scale
            current_position_error = float(
                np.linalg.norm(residual[:3]) * self.config.position_scale_m
            )
            current_normal_alignment = float(np.dot(current_normal, desired_normal))
            if (
                current_position_error <= position_tolerance_m
                and current_normal_alignment >= 0.995
                and np.linalg.norm(current_normal[:2]) >= self.config.minimum_normal_horizontal_norm
            ):
                break
            objective = float(np.dot(residual, residual))
            normal_matrix = jacobian.T @ jacobian + damping * np.eye(_MOVABLE_JOINTS)
            try:
                delta = np.linalg.solve(normal_matrix, jacobian.T @ residual)
            except np.linalg.LinAlgError:
                delta = np.linalg.pinv(normal_matrix) @ (jacobian.T @ residual)
            largest = float(np.max(np.abs(delta)))
            if largest > self.config.maximum_joint_step_rad:
                delta *= self.config.maximum_joint_step_rad / largest
            best_q = q
            best_objective = objective
            for line_index in range(self.config.line_search_steps):
                alpha = 0.5**line_index
                candidate_q = np.clip(q + alpha * delta, self._lower, self._upper)
                candidate_residual, _ = self._residual_and_jacobian(
                    candidate_q,
                    target_xyz,
                    desired_normal,
                    normal_scale,
                    with_jacobian=False,
                )
                candidate_objective = float(np.dot(candidate_residual, candidate_residual))
                if candidate_objective + 1.0e-12 < best_objective:
                    best_q = candidate_q
                    best_objective = candidate_objective
            if np.array_equal(best_q, q):
                damping = min(damping * 10.0, 1.0e6)
                if float(np.max(np.abs(delta))) < 1.0e-8 or damping >= 1.0e6:
                    break
                continue
            q = best_q
            accepted_steps += 1
            damping = max(damping * 0.5, 1.0e-9)
            if abs(objective - best_objective) < 1.0e-12:
                break
        residual, _ = self._residual_and_jacobian(
            q,
            target_xyz,
            desired_normal,
            normal_scale,
            with_jacobian=False,
        )
        return q.copy(), {
            "method": "bounded_multistart_damped_gauss_newton_with_line_search",
            "iterations": iterations,
            "accepted_steps": accepted_steps,
            "final_normalized_residual_l2": float(np.linalg.norm(residual)),
            "final_damping": damping,
        }

    def _evaluate(
        self,
        joint_position: np.ndarray,
        target_xyz: np.ndarray,
        direction: np.ndarray,
        *,
        standoff_m: float,
        height_m: float,
        lateral_m: float,
        normal_sign: float,
        normal_scale: float,
        start_index: int,
        solver: dict[str, Any],
    ) -> dict[str, Any]:
        self._set_scratch(self._evaluation_scratch, joint_position)
        data = self._evaluation_scratch
        block = self.env._ids["block_geom"]
        site = self.env._ids["tool_site"]
        actual_xyz = np.asarray(data.site_xpos[site], dtype=np.float64)
        normal = data.site_xmat[site].reshape(3, 3)[:, 1].copy()
        horizontal_norm = float(np.linalg.norm(normal[:2]))
        signed_axis_alignment = float(np.dot(normal[:2], direction))
        axis_alignment = abs(signed_axis_alignment)
        heading_alignment = axis_alignment / max(horizontal_norm, 1.0e-12)
        pusher_desk_clearance = float(
            self.env._minimum_tool_safety_signed_distance_for_data(
                self.env._desk_geom,
                data,
            )
        )
        tool_block_safety_clearance = float(
            self.env._minimum_tool_safety_signed_distance_for_data(
                block,
                data,
            )
        )
        pusher_block_rows = tuple(
            self.env._tool_planning_signed_distance_rows_for_data(block, data)
        )
        pusher_block_gaps = np.asarray(
            [float(row["signed_distance_m"]) for row in pusher_block_rows],
            dtype=np.float64,
        )
        pusher_block_gap = float(np.min(pusher_block_gaps))
        pusher_block_maximum_gap = float(np.max(pusher_block_gaps))
        central_overlap_rows = self._tip_central_side_overlap_rows_for_data(
            block,
            data,
        )
        central_overlaps = np.asarray(
            [float(row["overlap_m"]) for row in central_overlap_rows],
            dtype=np.float64,
        )
        minimum_central_overlap = float(np.min(central_overlaps))
        maximum_central_overlap = float(np.max(central_overlaps))
        contact_part_margin_required = bool(
            self.config.minimum_contact_part_central_side_margin_m > 0.0
        )
        contact_part_envelope_rows = (
            self._contact_part_central_side_envelope_rows_for_data(block, data)
            if contact_part_margin_required
            else ()
        )
        minimum_contact_part_central_side_margin = (
            min(
                float(row["central_side_margin_m"])
                for row in contact_part_envelope_rows
            )
            if contact_part_margin_required
            else None
        )
        ranges = np.asarray(self.env.joint_ranges[:_MOVABLE_JOINTS], dtype=np.float64)
        joint_margin = np.minimum(joint_position - ranges[:, 0], ranges[:, 1] - joint_position)
        normalized_joint_margin = joint_margin / self._joint_span
        tolerance = float(self.env.contact_feasible_config.reset_penetration_tolerance_m)
        robot_desk_penetrations = 0
        robot_block_contacts = 0
        for contact_index in range(data.ncon):
            contact = data.contact[contact_index]
            pair = {int(contact.geom1), int(contact.geom2)}
            if self.env._desk_geom in pair:
                other = next(iter(pair - {self.env._desk_geom}), -1)
                if (
                    other in self.env._robot_geoms
                    and float(contact.dist) < -tolerance
                ):
                    robot_desk_penetrations += 1
            if block in pair:
                other = next(iter(pair - {block}), -1)
                if other in self.env._robot_geoms:
                    # This is a pre-contact target: even a distal jaw/block
                    # contact is premature and therefore invalid here.
                    robot_block_contacts += 1
        position_error = float(np.linalg.norm(target_xyz - actual_xyz))
        feasible = bool(
            position_error <= self.config.maximum_position_error_m
            and axis_alignment >= self.config.minimum_normal_axis_alignment
            and heading_alignment >= self.config.minimum_normal_heading_alignment
            and horizontal_norm >= self.config.minimum_normal_horizontal_norm
            and pusher_desk_clearance >= self.config.minimum_pusher_desk_clearance_m
            and tool_block_safety_clearance
            >= self.config.minimum_tool_block_safety_clearance_m
            and self.config.minimum_pusher_block_gap_m
            <= pusher_block_gap
            and pusher_block_maximum_gap <= self.config.maximum_pusher_block_gap_m
            and minimum_central_overlap
            >= self.config.minimum_tip_central_side_overlap_m
            and (
                not contact_part_margin_required
                or (
                    minimum_contact_part_central_side_margin is not None
                    and minimum_contact_part_central_side_margin
                    >= self.config.minimum_contact_part_central_side_margin_m
                )
            )
            and float(np.min(normalized_joint_margin)) >= self.config.joint_margin_fraction - 1.0e-9
            and robot_desk_penetrations == 0
            and robot_block_contacts == 0
        )
        return {
            "feasible": feasible,
            "joint_position_rad": np.r_[
                joint_position,
                self.env.tool_gripper_joint_position_rad,
            ].tolist(),
            "target_xyz_m": target_xyz.tolist(),
            "actual_xyz_m": actual_xyz.tolist(),
            "position_error_m": position_error,
            "tool_face_normal_world": normal.tolist(),
            "normal_sign": normal_sign,
            "normal_axis_alignment": axis_alignment,
            "normal_signed_axis_alignment": signed_axis_alignment,
            "normal_heading_alignment": heading_alignment,
            "normal_horizontal_norm": horizontal_norm,
            "normal_abs_vertical": abs(float(normal[2])),
            "pusher_desk_signed_distance_m": pusher_desk_clearance,
            "tool_block_safety_signed_distance_m": tool_block_safety_clearance,
            "pusher_block_signed_distance_m": pusher_block_gap,
            "pusher_block_maximum_signed_distance_m": pusher_block_maximum_gap,
            "pusher_block_signed_distance_by_planning_geom": [
                dict(row) for row in pusher_block_rows
            ],
            "tip_central_side_overlap_m_by_planning_geom": [
                dict(row) for row in central_overlap_rows
            ],
            "minimum_tip_central_side_overlap_m": minimum_central_overlap,
            "maximum_tip_central_side_overlap_m": maximum_central_overlap,
            "contact_part_central_side_envelope_by_role": [
                dict(row) for row in contact_part_envelope_rows
            ],
            "minimum_contact_part_central_side_margin_m": (
                minimum_contact_part_central_side_margin
            ),
            "minimum_joint_margin_rad": float(np.min(joint_margin)),
            "minimum_normalized_joint_margin": float(np.min(normalized_joint_margin)),
            "robot_desk_penetrations": robot_desk_penetrations,
            "robot_block_contacts": robot_block_contacts,
            # Compatibility aliases for existing report consumers.  They now
            # deliberately include every robot geometry because a pre-contact
            # pose may not hide a premature distal-tool collision.
            "non_tool_robot_desk_penetrations": robot_desk_penetrations,
            "non_tool_robot_block_contacts": robot_block_contacts,
            "standoff_m": standoff_m,
            "height_m": height_m,
            "lateral_m": lateral_m,
            "normal_scale": normal_scale,
            "start_index": start_index,
            "solver": solver,
        }

    def _tip_central_side_overlap_rows_for_data(
        self,
        block: int,
        data: mujoco.MjData,
    ) -> tuple[dict[str, object], ...]:
        """Project each planning OBB onto the block's central local-Z band."""

        block_rotation = np.asarray(data.geom_xmat[block], dtype=np.float64).reshape(3, 3)
        block_vertical_axis = block_rotation[:, 2]
        block_center = np.asarray(data.geom_xpos[block], dtype=np.float64)
        block_half_height = float(self.env.model.geom_size[block, 2])
        central_half_height = block_half_height * (
            1.0 - self.config.block_top_bottom_exclusion_fraction
        )
        rows: list[dict[str, object]] = []
        for geom_id in self.env._tool_planning_geom_ids():
            geom_rotation = np.asarray(
                data.geom_xmat[geom_id],
                dtype=np.float64,
            ).reshape(3, 3)
            center_coordinate = float(
                np.dot(
                    np.asarray(data.geom_xpos[geom_id], dtype=np.float64) - block_center,
                    block_vertical_axis,
                )
            )
            projection_radius = float(
                np.dot(
                    np.abs(geom_rotation.T @ block_vertical_axis),
                    np.asarray(self.env.model.geom_size[geom_id], dtype=np.float64),
                )
            )
            tip_low = center_coordinate - projection_radius
            tip_high = center_coordinate + projection_radius
            central_low = -central_half_height
            central_high = central_half_height
            overlap = max(
                0.0,
                min(tip_high, central_high) - max(tip_low, central_low),
            )
            rows.append(
                {
                    "geom_id": int(geom_id),
                    "geom_name": mujoco.mj_id2name(
                        self.env.model,
                        mujoco.mjtObj.mjOBJ_GEOM,
                        int(geom_id),
                    ),
                    "tip_interval_block_local_z_m": [tip_low, tip_high],
                    "block_central_side_band_local_z_m": [
                        central_low,
                        central_high,
                    ],
                    "overlap_m": float(overlap),
                }
            )
        if not rows:  # pragma: no cover - guarded by environment model resolution
            raise RuntimeError("tool planning geometry set is empty")
        return tuple(rows)

    def _contact_part_central_side_envelope_rows_for_data(
        self,
        block: int,
        data: mujoco.MjData,
    ) -> tuple[dict[str, object], ...]:
        """Project authoritative contact geometry onto block-local vertical.

        The small planning boxes identify the intended distal faces but do
        not collide.  This gate audits the actual collision-part vertices, so
        a larger convex part cannot hit the top/bottom edge before its planning
        reference reaches the intended side face.
        """

        block_rotation = np.asarray(data.geom_xmat[block], dtype=np.float64).reshape(3, 3)
        block_vertical_axis = block_rotation[:, 2]
        block_center = np.asarray(data.geom_xpos[block], dtype=np.float64)
        central_half_height = float(self.env.model.geom_size[block, 2]) * (
            1.0 - self.config.block_top_bottom_exclusion_fraction
        )
        contact_geoms = tuple(int(value) for value in self.env._ids["tool_contact_geoms"])
        roles = tuple(
            str(value)
            for value in self.env._ids.get(
                "tool_contact_geom_roles",
                tuple("tool_contact" for _ in contact_geoms),
            )
        )
        if len(roles) != len(contact_geoms):
            raise RuntimeError("tool contact role identity is inconsistent")
        mesh_type = int(mujoco.mjtGeom.mjGEOM_MESH)
        rows: list[dict[str, object]] = []
        for role, geom_id in zip(roles, contact_geoms, strict=True):
            geom_rotation = np.asarray(
                data.geom_xmat[geom_id],
                dtype=np.float64,
            ).reshape(3, 3)
            geom_position = np.asarray(data.geom_xpos[geom_id], dtype=np.float64)
            if int(self.env.model.geom_type[geom_id]) == mesh_type:
                mesh_id = int(self.env.model.geom_dataid[geom_id])
                vertex_start = int(self.env.model.mesh_vertadr[mesh_id])
                vertex_count = int(self.env.model.mesh_vertnum[mesh_id])
                local_vertices = np.asarray(
                    self.env.model.mesh_vert[
                        vertex_start : vertex_start + vertex_count
                    ],
                    dtype=np.float64,
                )
            else:
                local_aabb = np.asarray(
                    self.env.model.geom_aabb[geom_id],
                    dtype=np.float64,
                )
                signs = np.asarray(
                    [
                        (x, y, z)
                        for x in (-1.0, 1.0)
                        for y in (-1.0, 1.0)
                        for z in (-1.0, 1.0)
                    ],
                    dtype=np.float64,
                )
                local_vertices = local_aabb[:3] + signs * local_aabb[3:]
            world_vertices = geom_position + local_vertices @ geom_rotation.T
            vertical_coordinates = (world_vertices - block_center) @ block_vertical_axis
            if not np.all(np.isfinite(vertical_coordinates)):
                raise RuntimeError("contact-part vertical envelope is non-finite")
            low = float(np.min(vertical_coordinates))
            high = float(np.max(vertical_coordinates))
            lower_margin = low + central_half_height
            upper_margin = central_half_height - high
            rows.append(
                {
                    "geom_id": geom_id,
                    "geom_name": mujoco.mj_id2name(
                        self.env.model,
                        mujoco.mjtObj.mjOBJ_GEOM,
                        geom_id,
                    ),
                    "role": role,
                    "contact_part_interval_block_local_z_m": [low, high],
                    "block_central_side_band_local_z_m": [
                        -central_half_height,
                        central_half_height,
                    ],
                    "lower_central_side_margin_m": float(lower_margin),
                    "upper_central_side_margin_m": float(upper_margin),
                    "central_side_margin_m": float(min(lower_margin, upper_margin)),
                }
            )
        if not rows:  # pragma: no cover - model resolution rejects this first
            raise RuntimeError("tool contact geometry set is empty")
        return tuple(rows)

    def _rank(self, candidate: dict[str, Any]) -> tuple[object, ...]:
        return (
            not bool(candidate["feasible"]),
            int(candidate["robot_desk_penetrations"]),
            int(candidate["robot_block_contacts"]),
            max(
                0.0,
                self.config.minimum_pusher_desk_clearance_m
                - float(candidate["pusher_desk_signed_distance_m"]),
            ),
            max(
                0.0,
                self.config.minimum_tool_block_safety_clearance_m
                - float(candidate["tool_block_safety_signed_distance_m"]),
            ),
            max(0.0, float(candidate["position_error_m"]) - self.config.maximum_position_error_m),
            max(
                0.0,
                self.config.minimum_normal_axis_alignment
                - float(candidate["normal_axis_alignment"]),
            ),
            max(
                0.0,
                self.config.minimum_pusher_block_gap_m
                - float(candidate["pusher_block_signed_distance_m"]),
                float(candidate["pusher_block_maximum_signed_distance_m"])
                - self.config.maximum_pusher_block_gap_m,
            ),
            max(
                0.0,
                self.config.minimum_tip_central_side_overlap_m
                - float(candidate["minimum_tip_central_side_overlap_m"]),
            ),
            max(
                0.0,
                self.config.minimum_contact_part_central_side_margin_m
                - float(candidate["minimum_contact_part_central_side_margin_m"])
                if self.config.minimum_contact_part_central_side_margin_m > 0.0
                else 0.0,
            ),
            max(
                abs(
                    float(row["signed_distance_m"])
                    - self.config.preferred_pusher_block_gap_m
                )
                for row in candidate["pusher_block_signed_distance_by_planning_geom"]
            ),
            float(candidate["position_error_m"]),
            -float(candidate["pusher_desk_signed_distance_m"]),
            -float(candidate["minimum_normalized_joint_margin"]),
        )

    def _report(
        self,
        best: dict[str, Any],
        candidates: list[dict[str, Any]],
        direction: np.ndarray,
        block_support: float,
        nominal_standoff: float,
        target_count: int,
    ) -> dict[str, Any]:
        return {
            "format": SIDE_CONTACT_IK_FORMAT,
            "claim_level": "L1_SYNTHETIC_HARDWARE_INSPIRED",
            "parameter_source": "synthetic_five_constraint_kinematic_search",
            "simulator_privileged_truth": True,
            "physical_samples": 0,
            "physical_trials": 0,
            "physically_calibrated": False,
            "physical_hardware_connected": False,
            "live_state_written": False,
            "tool_gripper_joint_position_rad": float(
                self.env.tool_gripper_joint_position_rad
            ),
            "configuration_hash": self.config.profile_hash,
            "configuration": asdict(self.config),
            "intended_push_direction_xy": direction.tolist(),
            "block_support_along_push_m": block_support,
            "nominal_center_standoff_m": nominal_standoff,
            "target_candidates_evaluated": target_count,
            "ik_candidates_evaluated": len(candidates),
            "feasible_candidate_count": sum(bool(candidate["feasible"]) for candidate in candidates),
            "feasible": bool(best["feasible"]),
            "best": best,
        }


__all__ = [
    "SIDE_CONTACT_IK_FORMAT",
    "SideContactIKConfig",
    "SideContactIKPlannerV1",
]
