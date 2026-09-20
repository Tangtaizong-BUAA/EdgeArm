"""Hierarchical damped task-frame action map for Home acquisition.

The policy-facing action has three components: forward, lateral, and vertical
tool translation in the current block-to-target task frame.  This module maps
that local Cartesian request into the five movable arm joints.  Translation
is the primary task; the two stock-gripper push-face constraints are corrected
in its local null space.  The resulting joint delta still has to pass the
unchanged V4 execution guard before it can reach the plant.

This is an action coordinate transform, not an expert, route, waypoint
generator, or success controller.  In particular, it never reads the
precontact goal and cannot decide which task-frame action the policy should
take.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import mujoco
import numpy as np

from .guarded_joint_delta_action_v664 import ARM_JOINT_ACTION_DIM_V664
from .orientation_projected_joint_action_v667 import (
    orientation_constraint_state_v667,
)
from .sim2real_env_v10 import RealisticEdgeArmEnvV10


TASKFRAME_DLS_JOINT_ACTION_FORMAT_V688 = "edgearm-v688-hierarchical-damped-taskframe-joint-action-v1"
JOINT_MARGIN_AWARE_DLS_FORMAT_V746 = "edgearm-v746-position-nullspace-joint-margin-avoidance-v1"
BOUND_CONSTRAINED_HIERARCHICAL_DLS_FORMAT_V755 = "edgearm-v755-bound-constrained-hierarchical-dls-v1"
TASKFRAME_ACTION_DIM_V688 = 3


@dataclass(frozen=True)
class TaskFrameDLSJointActionConfigV688:
    """Numerically bounded Cartesian-to-joint action contract."""

    forward_translation_step_m: float = 0.0015
    lateral_translation_step_m: float = 0.0010
    vertical_translation_step_m: float = 0.0040
    maximum_joint_target_step_rad: float = 0.025
    position_damping_lambda: float = 0.010
    orientation_damping_lambda: float = 0.015
    singular_value_soft_floor: float = 0.030
    adaptive_damping_gain: float = 0.20
    orientation_correction_gain: float = 0.30
    intervention_l2_threshold: float = 1.0e-6
    joint_margin_avoidance_v746: bool = False
    bound_constrained_hierarchical_dls_v755: bool = False
    directional_face_yaw_v778: bool = False
    joint_margin_activation_fraction_v746: float = 0.20
    joint_margin_avoidance_maximum_step_rad_v746: float = 0.009
    joint_margin_orientation_degradation_tolerance_v746: float = 0.0025
    joint_margin_position_deviation_tolerance_m_v746: float = 5.0e-5
    joint_margin_minimum_weighted_improvement_v746: float = 1.0e-8
    joint_margin_backtracking_scales_v746: tuple[float, ...] = (
        1.0,
        0.5,
        0.25,
        0.125,
    )

    def validate(self) -> None:
        positive = np.asarray(
            [
                self.forward_translation_step_m,
                self.lateral_translation_step_m,
                self.vertical_translation_step_m,
                self.maximum_joint_target_step_rad,
                self.position_damping_lambda,
                self.orientation_damping_lambda,
                self.singular_value_soft_floor,
                self.adaptive_damping_gain,
                self.orientation_correction_gain,
                self.intervention_l2_threshold,
                self.joint_margin_activation_fraction_v746,
                self.joint_margin_avoidance_maximum_step_rad_v746,
                self.joint_margin_orientation_degradation_tolerance_v746,
                self.joint_margin_position_deviation_tolerance_m_v746,
                self.joint_margin_minimum_weighted_improvement_v746,
            ],
            dtype=np.float64,
        )
        if not np.all(np.isfinite(positive)) or np.any(positive <= 0.0):
            raise ValueError("V688 DLS configuration must be finite and positive")
        if self.maximum_joint_target_step_rad > 0.05:
            raise ValueError("V688 joint step exceeds deployed command support")
        if self.orientation_correction_gain > 1.0:
            raise ValueError("V688 orientation correction gain exceeds one")
        scales = tuple(float(value) for value in self.joint_margin_backtracking_scales_v746)
        if (
            type(self.joint_margin_avoidance_v746) is not bool
            or type(self.bound_constrained_hierarchical_dls_v755) is not bool
            or type(self.directional_face_yaw_v778) is not bool
            or self.joint_margin_activation_fraction_v746 > 0.5
            or self.joint_margin_avoidance_maximum_step_rad_v746 > self.maximum_joint_target_step_rad
            or not scales
            or scales[0] != 1.0
            or any(not np.isfinite(value) or not 0.0 < value <= 1.0 for value in scales)
            or any(left <= right for left, right in zip(scales, scales[1:]))
        ):
            raise ValueError("V746 joint-margin avoidance configuration is invalid")


@dataclass(frozen=True)
class TaskFrameDLSJointActionV688:
    requested_task_action: np.ndarray
    requested_local_delta_m: np.ndarray
    requested_world_delta_m: np.ndarray
    projected_joint_action: np.ndarray
    projected_joint_delta_rad: np.ndarray
    predicted_world_delta_m: np.ndarray
    predicted_local_delta_m: np.ndarray
    orientation_residual_before: np.ndarray
    predicted_orientation_residual_after: np.ndarray
    position_singular_values: np.ndarray
    orientation_nullspace_singular_values: np.ndarray
    position_damping_used: float
    orientation_damping_used: float
    joint_limit_scale: float
    requested_to_predicted_task_l2: float
    intervened: bool
    audit: dict[str, Any]
    format: str = TASKFRAME_DLS_JOINT_ACTION_FORMAT_V688

    def validate(self) -> None:
        arrays = {
            "requested_task_action": (self.requested_task_action, (3,)),
            "requested_local_delta_m": (self.requested_local_delta_m, (3,)),
            "requested_world_delta_m": (self.requested_world_delta_m, (3,)),
            "projected_joint_action": (self.projected_joint_action, (5,)),
            "projected_joint_delta_rad": (self.projected_joint_delta_rad, (5,)),
            "predicted_world_delta_m": (self.predicted_world_delta_m, (3,)),
            "predicted_local_delta_m": (self.predicted_local_delta_m, (3,)),
            "orientation_residual_before": (
                self.orientation_residual_before,
                (2,),
            ),
            "predicted_orientation_residual_after": (
                self.predicted_orientation_residual_after,
                (2,),
            ),
            "position_singular_values": (self.position_singular_values, (3,)),
            "orientation_nullspace_singular_values": (
                self.orientation_nullspace_singular_values,
                (2,),
            ),
        }
        for name, (value, shape) in arrays.items():
            array = np.asarray(value)
            if array.shape != shape or not np.all(np.isfinite(array)):
                raise ValueError(f"V688 {name} is invalid")
        scalars = np.asarray(
            [
                self.position_damping_used,
                self.orientation_damping_used,
                self.joint_limit_scale,
                self.requested_to_predicted_task_l2,
            ],
            dtype=np.float64,
        )
        if (
            not np.all(np.isfinite(scalars))
            or np.any(scalars < 0.0)
            or self.joint_limit_scale > 1.0 + 1.0e-12
            or np.any(np.abs(self.requested_task_action) > 1.0 + 1.0e-6)
            or np.any(np.abs(self.projected_joint_action) > 1.0 + 1.0e-6)
            or type(self.intervened) is not bool
            or type(self.audit) is not dict
        ):
            raise ValueError("V688 action result metadata is invalid")


def applied_task_action_from_tool_delta_v688(
    tool_delta_world_m: np.ndarray,
    *,
    forward_xy: np.ndarray,
    lateral_xy: np.ndarray,
    translation_scale_xyz_m: np.ndarray,
) -> np.ndarray:
    """Recover the normalized task action from actual tool displacement."""

    delta = np.asarray(tool_delta_world_m, dtype=np.float64)
    forward = np.asarray(forward_xy, dtype=np.float64)
    lateral = np.asarray(lateral_xy, dtype=np.float64)
    scale = np.asarray(translation_scale_xyz_m, dtype=np.float64)
    if (
        delta.shape != (3,)
        or forward.shape != (2,)
        or lateral.shape != (2,)
        or scale.shape != (3,)
        or not np.all(np.isfinite(np.r_[delta, forward, lateral, scale]))
        or np.any(scale <= 0.0)
        or not np.isclose(np.linalg.norm(forward), 1.0, atol=1.0e-6, rtol=0.0)
        or not np.isclose(np.linalg.norm(lateral), 1.0, atol=1.0e-6, rtol=0.0)
        or not np.isclose(np.dot(forward, lateral), 0.0, atol=1.0e-6, rtol=0.0)
    ):
        raise ValueError("V688 applied task-action inputs are invalid")
    local = np.asarray(
        [
            float(np.dot(delta[:2], forward)),
            float(np.dot(delta[:2], lateral)),
            float(delta[2]),
        ],
        dtype=np.float64,
    )
    return np.clip(local / scale, -1.0, 1.0).astype(np.float32)


def _adaptive_damping_v688(
    singular_values: np.ndarray,
    *,
    base: float,
    soft_floor: float,
    gain: float,
) -> float:
    values = np.asarray(singular_values, dtype=np.float64)
    if values.ndim != 1 or values.size < 1 or not np.all(np.isfinite(values)):
        raise ValueError("V688 singular values are invalid")
    minimum = float(np.min(values))
    deficit = max(soft_floor - minimum, 0.0) / soft_floor
    return float(base + gain * deficit * deficit)


def _damped_right_inverse_v688(
    jacobian: np.ndarray,
    damping: float,
) -> np.ndarray:
    matrix = np.asarray(jacobian, dtype=np.float64)
    value = float(damping)
    if (
        matrix.ndim != 2
        or matrix.shape[0] < 1
        or matrix.shape[1] < matrix.shape[0]
        or not np.all(np.isfinite(matrix))
        or not np.isfinite(value)
        or value <= 0.0
    ):
        raise ValueError("V688 damped inverse inputs are invalid")
    normal = matrix @ matrix.T + value * value * np.eye(matrix.shape[0])
    return matrix.T @ np.linalg.solve(normal, np.eye(matrix.shape[0]))


def _damped_least_squares_v755(
    matrix: np.ndarray,
    target: np.ndarray,
    damping: float,
) -> np.ndarray:
    """Solve a small damped least-squares system for any matrix shape."""

    design = np.asarray(matrix, dtype=np.float64)
    desired = np.asarray(target, dtype=np.float64)
    value = float(damping)
    if (
        design.ndim != 2
        or desired.shape != (design.shape[0],)
        or design.shape[1] < 1
        or not np.all(np.isfinite(np.r_[design.ravel(), desired, value]))
        or value <= 0.0
    ):
        raise ValueError("V755 damped least-squares inputs are invalid")
    normal = design.T @ design + value * value * np.eye(design.shape[1])
    return np.linalg.solve(normal, design.T @ desired)


def bound_constrained_hierarchical_delta_v755(
    *,
    position_jacobian: np.ndarray,
    orientation_jacobian: np.ndarray,
    orientation_residual: np.ndarray,
    requested_world_delta_m: np.ndarray,
    lower_joint_delta_rad: np.ndarray,
    upper_joint_delta_rad: np.ndarray,
    position_damping: float,
    orientation_damping: float,
    orientation_correction_gain: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Solve translation and orientation inside the live joint-delta box.

    V688 historically solved an unconstrained hierarchy and clipped the final
    joints afterward. Near a joint limit that clipping can reverse the task
    direction. V755 instead clamps one violating joint at a time and resolves
    the remaining translation/orientation hierarchy over the free joints.
    """

    position = np.asarray(position_jacobian, dtype=np.float64)
    orientation = np.asarray(orientation_jacobian, dtype=np.float64)
    residual = np.asarray(orientation_residual, dtype=np.float64)
    requested = np.asarray(requested_world_delta_m, dtype=np.float64)
    lower = np.asarray(lower_joint_delta_rad, dtype=np.float64)
    upper = np.asarray(upper_joint_delta_rad, dtype=np.float64)
    if (
        position.shape != (3, ARM_JOINT_ACTION_DIM_V664)
        or orientation.shape != (2, ARM_JOINT_ACTION_DIM_V664)
        or residual.shape != (2,)
        or requested.shape != (3,)
        or lower.shape != (ARM_JOINT_ACTION_DIM_V664,)
        or upper.shape != lower.shape
        or not np.all(
            np.isfinite(
                np.r_[
                    position.ravel(),
                    orientation.ravel(),
                    residual,
                    requested,
                    lower,
                    upper,
                    position_damping,
                    orientation_damping,
                    orientation_correction_gain,
                ]
            )
        )
        or np.any(lower > upper)
        or position_damping <= 0.0
        or orientation_damping <= 0.0
        or not 0.0 < orientation_correction_gain <= 1.0
    ):
        raise ValueError("V755 bound-constrained hierarchy inputs are invalid")

    joint_count = ARM_JOINT_ACTION_DIM_V664
    fixed = np.zeros(joint_count, dtype=bool)
    delta = np.zeros(joint_count, dtype=np.float64)
    initial_unconstrained: np.ndarray | None = None
    active_order: list[int] = []
    iterations = 0
    for iterations in range(1, joint_count + 2):
        free_indices = np.flatnonzero(~fixed)
        fixed_indices = np.flatnonzero(fixed)
        if len(free_indices) == 0:
            break
        position_free = position[:, free_indices]
        fixed_position = (
            position[:, fixed_indices] @ delta[fixed_indices]
            if len(fixed_indices)
            else np.zeros(3, dtype=np.float64)
        )
        primary = _damped_least_squares_v755(
            position_free,
            requested - fixed_position,
            position_damping,
        )
        position_inverse = np.linalg.solve(
            position_free.T @ position_free + position_damping * position_damping * np.eye(len(free_indices)),
            position_free.T,
        )
        position_nullspace = np.eye(len(free_indices)) - position_inverse @ position_free
        orientation_free = orientation[:, free_indices]
        fixed_orientation = (
            orientation[:, fixed_indices] @ delta[fixed_indices]
            if len(fixed_indices)
            else np.zeros(2, dtype=np.float64)
        )
        residual_after_primary = residual + fixed_orientation + orientation_free @ primary
        orientation_operator = orientation_free @ position_nullspace
        secondary_coordinates = _damped_least_squares_v755(
            orientation_operator,
            -orientation_correction_gain * residual_after_primary,
            orientation_damping,
        )
        candidate = delta.copy()
        candidate[free_indices] = primary + position_nullspace @ secondary_coordinates
        if initial_unconstrained is None:
            initial_unconstrained = candidate.copy()
        below = lower - candidate
        above = candidate - upper
        violation = np.maximum(below, above)
        if float(np.max(violation)) <= 1.0e-12:
            delta = candidate
            break
        normalized = violation / np.maximum(upper - lower, 1.0e-12)
        violating_index = int(np.argmax(normalized))
        if fixed[violating_index]:
            raise RuntimeError("V755 active-set solver selected a fixed joint")
        delta[violating_index] = float(
            np.clip(candidate[violating_index], lower[violating_index], upper[violating_index])
        )
        fixed[violating_index] = True
        active_order.append(violating_index)
    else:  # pragma: no cover - finite active set must terminate first
        raise RuntimeError("V755 active-set solver exceeded its finite bound")

    delta = np.clip(delta, lower, upper)
    predicted_position = position @ delta
    predicted_orientation = residual + orientation @ delta
    if initial_unconstrained is None:
        initial_unconstrained = delta.copy()
    audit = {
        "format": BOUND_CONSTRAINED_HIERARCHICAL_DLS_FORMAT_V755,
        "enabled": True,
        "active_joint_indices_in_order": active_order,
        "active_joint_count": len(active_order),
        "iterations": iterations,
        "lower_joint_delta_rad": lower.tolist(),
        "upper_joint_delta_rad": upper.tolist(),
        "initial_unconstrained_joint_delta_rad": (initial_unconstrained.tolist()),
        "selected_joint_delta_rad": delta.tolist(),
        "requested_world_delta_m": requested.tolist(),
        "predicted_world_delta_m": predicted_position.tolist(),
        "position_error_l2_m": float(np.linalg.norm(requested - predicted_position)),
        "predicted_orientation_residual_l2": float(np.linalg.norm(predicted_orientation)),
        "automatic_route_or_waypoint": False,
        "precontact_goal_read": False,
        "expert_action_used": False,
        "same_v4_guard_still_required": True,
        "production_admission": False,
    }
    return delta, audit


def _joint_limit_margin_fraction_v746(
    joint_position: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> np.ndarray:
    """Return each joint's distance to its nearest limit as a range fraction."""

    position = np.asarray(joint_position, dtype=np.float64)
    low = np.asarray(lower, dtype=np.float64)
    high = np.asarray(upper, dtype=np.float64)
    if (
        position.shape != (ARM_JOINT_ACTION_DIM_V664,)
        or low.shape != position.shape
        or high.shape != position.shape
        or not np.all(np.isfinite(np.r_[position, low, high]))
        or np.any(high <= low)
    ):
        raise ValueError("V746 joint-margin inputs are invalid")
    return np.minimum(position - low, high - position) / (high - low)


def _apply_joint_margin_avoidance_v746(
    *,
    current_joint: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    base_joint_delta: np.ndarray,
    position_jacobian: np.ndarray,
    orientation_residual: np.ndarray,
    orientation_jacobian: np.ndarray,
    config: TaskFrameDLSJointActionConfigV688,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Use only the Cartesian-position null space to retreat from joint limits.

    V746 does not select a task action. It modifies the V688 inverse map only
    when a joint is inside a configured limit band, and accepts the largest
    deterministic backtracking candidate that improves weighted joint margin
    without materially changing the policy-requested Cartesian displacement or
    degrading the two stock-gripper orientation constraints beyond tolerance.
    The resulting proposal still passes the unchanged V4 online guard.
    """

    current = np.asarray(current_joint, dtype=np.float64)
    low = np.asarray(lower, dtype=np.float64)
    high = np.asarray(upper, dtype=np.float64)
    base_delta = np.asarray(base_joint_delta, dtype=np.float64)
    position = np.asarray(position_jacobian, dtype=np.float64)
    residual = np.asarray(orientation_residual, dtype=np.float64)
    orientation = np.asarray(orientation_jacobian, dtype=np.float64)
    if (
        current.shape != (ARM_JOINT_ACTION_DIM_V664,)
        or low.shape != current.shape
        or high.shape != current.shape
        or base_delta.shape != current.shape
        or position.shape != (3, ARM_JOINT_ACTION_DIM_V664)
        or residual.shape != (2,)
        or orientation.shape != (2, ARM_JOINT_ACTION_DIM_V664)
        or not np.all(
            np.isfinite(
                np.r_[
                    current,
                    low,
                    high,
                    base_delta,
                    position.ravel(),
                    residual,
                    orientation.ravel(),
                ]
            )
        )
    ):
        raise ValueError("V746 projection inputs are invalid")

    margin_before = _joint_limit_margin_fraction_v746(current, low, high)
    base_target = current + base_delta
    margin_after_base = _joint_limit_margin_fraction_v746(base_target, low, high)
    danger = (
        np.clip(
            (config.joint_margin_activation_fraction_v746 - margin_before)
            / config.joint_margin_activation_fraction_v746,
            0.0,
            1.0,
        )
        ** 2
    )
    base_orientation = residual + orientation @ base_delta
    default_audit: dict[str, Any] = {
        "format": JOINT_MARGIN_AWARE_DLS_FORMAT_V746,
        "enabled": bool(config.joint_margin_avoidance_v746),
        "activated": bool(np.any(danger > 0.0)),
        "applied": False,
        "selected_scale": 0.0,
        "position_nullspace_rank": 0,
        "position_nullspace_dimension": 0,
        "minimum_margin_fraction_before": float(np.min(margin_before)),
        "minimum_margin_fraction_after_base": float(np.min(margin_after_base)),
        "minimum_margin_fraction_after_selected": float(np.min(margin_after_base)),
        "margin_fraction_before_by_joint": margin_before.tolist(),
        "margin_fraction_after_base_by_joint": margin_after_base.tolist(),
        "margin_fraction_after_selected_by_joint": margin_after_base.tolist(),
        "minimum_margin_joint_index_before": int(np.argmin(margin_before)),
        "minimum_margin_joint_index_after_base": int(np.argmin(margin_after_base)),
        "minimum_margin_joint_index_after_selected": int(np.argmin(margin_after_base)),
        "weighted_margin_improvement": 0.0,
        "position_deviation_from_base_m": 0.0,
        "orientation_residual_l2_after_base": float(np.linalg.norm(base_orientation)),
        "orientation_residual_l2_after_selected": float(np.linalg.norm(base_orientation)),
        "unscaled_nullspace_bias_rad": np.zeros(
            ARM_JOINT_ACTION_DIM_V664,
            dtype=np.float64,
        ).tolist(),
        "selected_nullspace_bias_rad": np.zeros(
            ARM_JOINT_ACTION_DIM_V664,
            dtype=np.float64,
        ).tolist(),
        "automatic_route_or_waypoint": False,
        "precontact_goal_read_by_joint_margin_avoidance": False,
        "expert_action_used": False,
        "same_v4_guard_still_required": True,
        "production_admission": False,
    }
    if not config.joint_margin_avoidance_v746 or not np.any(danger > 0.0):
        return base_delta.copy(), default_audit

    _u, singular_values, vh = np.linalg.svd(position, full_matrices=True)
    rank_tolerance = max(position.shape) * np.finfo(np.float64).eps * max(float(singular_values[0]), 1.0)
    rank = int(np.sum(singular_values > rank_tolerance))
    null_basis = vh[rank:].T
    default_audit["position_nullspace_rank"] = rank
    default_audit["position_nullspace_dimension"] = int(null_basis.shape[1])
    if null_basis.shape[1] < 1:
        return base_delta.copy(), default_audit

    span = high - low
    midpoint = 0.5 * (low + high)
    center_direction = np.clip(
        (midpoint - current) / (0.5 * span),
        -1.0,
        1.0,
    )
    raw_bias = null_basis @ (null_basis.T @ (danger * center_direction))
    maximum_raw = float(np.max(np.abs(raw_bias)))
    if maximum_raw <= 1.0e-12:
        return base_delta.copy(), default_audit
    unscaled_bias = raw_bias * config.joint_margin_avoidance_maximum_step_rad_v746 / maximum_raw
    default_audit["unscaled_nullspace_bias_rad"] = unscaled_bias.tolist()
    weighted_margin_after_base = float(np.dot(danger, margin_after_base))

    for scale in config.joint_margin_backtracking_scales_v746:
        selected_bias = float(scale) * unscaled_bias
        candidate = base_delta + selected_bias
        candidate_target = current + candidate
        if (
            float(np.max(np.abs(candidate))) > config.maximum_joint_target_step_rad + 1.0e-12
            or np.any(candidate_target < low - 1.0e-12)
            or np.any(candidate_target > high + 1.0e-12)
        ):
            continue
        position_deviation = float(np.linalg.norm(position @ selected_bias))
        candidate_orientation = residual + orientation @ candidate
        candidate_orientation_l2 = float(np.linalg.norm(candidate_orientation))
        if (
            position_deviation > config.joint_margin_position_deviation_tolerance_m_v746
            or candidate_orientation_l2
            > float(np.linalg.norm(base_orientation))
            + config.joint_margin_orientation_degradation_tolerance_v746
        ):
            continue
        margin_after_candidate = _joint_limit_margin_fraction_v746(
            candidate_target,
            low,
            high,
        )
        improvement = float(np.dot(danger, margin_after_candidate) - weighted_margin_after_base)
        if improvement < config.joint_margin_minimum_weighted_improvement_v746:
            continue
        return candidate, {
            **default_audit,
            "applied": True,
            "selected_scale": float(scale),
            "minimum_margin_fraction_after_selected": float(np.min(margin_after_candidate)),
            "margin_fraction_after_selected_by_joint": (margin_after_candidate.tolist()),
            "minimum_margin_joint_index_after_selected": int(np.argmin(margin_after_candidate)),
            "weighted_margin_improvement": improvement,
            "position_deviation_from_base_m": position_deviation,
            "orientation_residual_l2_after_selected": candidate_orientation_l2,
            "selected_nullspace_bias_rad": selected_bias.tolist(),
        }
    return base_delta.copy(), default_audit


def project_taskframe_action_v688(
    env: RealisticEdgeArmEnvV10,
    task_action: np.ndarray,
    config: TaskFrameDLSJointActionConfigV688 | None = None,
) -> TaskFrameDLSJointActionV688:
    """Map one policy-chosen task-frame translation to a bounded joint delta."""

    if type(env) is not RealisticEdgeArmEnvV10:
        raise TypeError("V688 task-frame projection requires exact V10 environment")
    selected = config or TaskFrameDLSJointActionConfigV688()
    if type(selected) is not TaskFrameDLSJointActionConfigV688:
        raise TypeError("V688 projection requires its exact configuration")
    selected.validate()
    requested = np.asarray(task_action, dtype=np.float64)
    if (
        requested.shape != (TASKFRAME_ACTION_DIM_V688,)
        or not np.all(np.isfinite(requested))
        or np.any(np.abs(requested) > 1.0 + 1.0e-6)
    ):
        raise ValueError("V688 task action must be a finite normalized three-vector")
    requested = np.clip(requested, -1.0, 1.0)

    orientation = orientation_constraint_state_v667(
        env,
        directional_face_yaw_v778=selected.directional_face_yaw_v778,
    )
    forward = orientation.intended_push_direction_xy
    lateral = np.asarray([-forward[1], forward[0]], dtype=np.float64)
    local_scale = np.asarray(
        [
            selected.forward_translation_step_m,
            selected.lateral_translation_step_m,
            selected.vertical_translation_step_m,
        ],
        dtype=np.float64,
    )
    requested_local = requested * local_scale
    requested_world = np.asarray(
        [
            forward[0] * requested_local[0] + lateral[0] * requested_local[1],
            forward[1] * requested_local[0] + lateral[1] * requested_local[1],
            requested_local[2],
        ],
        dtype=np.float64,
    )

    jacobian_position = np.zeros((3, env.model.nv), dtype=np.float64)
    jacobian_rotation = np.zeros((3, env.model.nv), dtype=np.float64)
    mujoco.mj_jacSite(
        env.model,
        env.data,
        jacobian_position,
        jacobian_rotation,
        int(env._ids["tool_site"]),
    )
    del jacobian_rotation
    position_jacobian = jacobian_position[:, :ARM_JOINT_ACTION_DIM_V664]
    position_singular = np.linalg.svd(position_jacobian, compute_uv=False)
    position_damping = _adaptive_damping_v688(
        position_singular,
        base=selected.position_damping_lambda,
        soft_floor=selected.singular_value_soft_floor,
        gain=selected.adaptive_damping_gain,
    )
    position_inverse = _damped_right_inverse_v688(position_jacobian, position_damping)
    primary_delta = position_inverse @ requested_world
    position_nullspace = np.eye(ARM_JOINT_ACTION_DIM_V664) - position_inverse @ position_jacobian

    orientation_jacobian = orientation.jacobian
    nullspace_orientation = orientation_jacobian @ position_nullspace
    orientation_singular = np.linalg.svd(nullspace_orientation, compute_uv=False)
    orientation_damping = _adaptive_damping_v688(
        orientation_singular,
        base=selected.orientation_damping_lambda,
        soft_floor=selected.singular_value_soft_floor,
        gain=selected.adaptive_damping_gain,
    )
    current_joint = np.asarray(env.data.qpos[:ARM_JOINT_ACTION_DIM_V664], dtype=np.float64)
    lower = np.asarray(env.model.jnt_range[:ARM_JOINT_ACTION_DIM_V664, 0], dtype=np.float64)
    upper = np.asarray(env.model.jnt_range[:ARM_JOINT_ACTION_DIM_V664, 1], dtype=np.float64)
    lower_delta = np.maximum(
        lower - current_joint,
        -selected.maximum_joint_target_step_rad,
    )
    upper_delta = np.minimum(
        upper - current_joint,
        selected.maximum_joint_target_step_rad,
    )
    if selected.bound_constrained_hierarchical_dls_v755:
        joint_delta, bound_constrained_audit_v755 = bound_constrained_hierarchical_delta_v755(
            position_jacobian=position_jacobian,
            orientation_jacobian=orientation_jacobian,
            orientation_residual=orientation.residual,
            requested_world_delta_m=requested_world,
            lower_joint_delta_rad=lower_delta,
            upper_joint_delta_rad=upper_delta,
            position_damping=position_damping,
            orientation_damping=orientation_damping,
            orientation_correction_gain=(selected.orientation_correction_gain),
        )
        unconstrained = np.asarray(
            bound_constrained_audit_v755["initial_unconstrained_joint_delta_rad"],
            dtype=np.float64,
        )
        maximum_delta = float(np.max(np.abs(unconstrained)))
        joint_limit_scale = min(
            1.0,
            selected.maximum_joint_target_step_rad
            / max(maximum_delta, selected.maximum_joint_target_step_rad),
        )
    else:
        orientation_inverse = _damped_right_inverse_v688(
            nullspace_orientation,
            orientation_damping,
        )
        residual_after_primary = orientation.residual + orientation_jacobian @ primary_delta
        secondary_delta = (
            position_nullspace
            @ orientation_inverse
            @ (-selected.orientation_correction_gain * residual_after_primary)
        )
        joint_delta = primary_delta + secondary_delta
        maximum_delta = float(np.max(np.abs(joint_delta)))
        joint_limit_scale = 1.0
        if maximum_delta > selected.maximum_joint_target_step_rad:
            joint_limit_scale = selected.maximum_joint_target_step_rad / maximum_delta
            joint_delta *= joint_limit_scale
        target_joint = np.clip(current_joint + joint_delta, lower, upper)
        joint_delta = target_joint - current_joint
        bound_constrained_audit_v755 = {
            "format": BOUND_CONSTRAINED_HIERARCHICAL_DLS_FORMAT_V755,
            "enabled": False,
            "selection_unchanged": True,
            "production_admission": False,
        }
    joint_delta, joint_margin_audit_v746 = _apply_joint_margin_avoidance_v746(
        current_joint=current_joint,
        lower=lower,
        upper=upper,
        base_joint_delta=joint_delta,
        position_jacobian=position_jacobian,
        orientation_residual=orientation.residual,
        orientation_jacobian=orientation_jacobian,
        config=selected,
    )
    joint_action = np.clip(
        joint_delta / selected.maximum_joint_target_step_rad,
        -1.0,
        1.0,
    )
    predicted_world = position_jacobian @ joint_delta
    predicted_local = np.asarray(
        [
            float(np.dot(predicted_world[:2], forward)),
            float(np.dot(predicted_world[:2], lateral)),
            float(predicted_world[2]),
        ],
        dtype=np.float64,
    )
    predicted_orientation = orientation.residual + orientation_jacobian @ joint_delta
    requested_to_predicted = float(
        np.linalg.norm(requested_local / local_scale - predicted_local / local_scale)
    )
    intervention = bool(
        requested_to_predicted > selected.intervention_l2_threshold
        or joint_limit_scale < 1.0 - 1.0e-12
        or joint_margin_audit_v746["applied"]
    )
    audit = {
        "format": TASKFRAME_DLS_JOINT_ACTION_FORMAT_V688,
        "configuration": asdict(selected),
        "policy_action_dimension": TASKFRAME_ACTION_DIM_V688,
        "execution_joint_dimension": ARM_JOINT_ACTION_DIM_V664,
        "primary_task": "task_frame_xyz_translation",
        "secondary_task": ("vertical_and_block_to_target_aligned_stock_gripper_broad_face"),
        "solver": "hierarchical_adaptive_damped_right_inverse",
        "automatic_route_or_waypoint": False,
        "precontact_goal_read_by_transform": False,
        "expert_action_used": False,
        "behavior_cloning_steps": 0,
        "same_v4_guard_still_required": True,
        "joint_margin_avoidance_v746": joint_margin_audit_v746,
        "bound_constrained_hierarchical_dls_v755": (bound_constrained_audit_v755),
        "simulator_privileged_task_axis": True,
        "production_admission": False,
    }
    result = TaskFrameDLSJointActionV688(
        requested_task_action=requested.astype(np.float32),
        requested_local_delta_m=requested_local.astype(np.float32),
        requested_world_delta_m=requested_world.astype(np.float32),
        projected_joint_action=joint_action.astype(np.float32),
        projected_joint_delta_rad=joint_delta.astype(np.float32),
        predicted_world_delta_m=predicted_world.astype(np.float32),
        predicted_local_delta_m=predicted_local.astype(np.float32),
        orientation_residual_before=orientation.residual.astype(np.float32),
        predicted_orientation_residual_after=predicted_orientation.astype(np.float32),
        position_singular_values=position_singular.astype(np.float32),
        orientation_nullspace_singular_values=orientation_singular.astype(np.float32),
        position_damping_used=position_damping,
        orientation_damping_used=orientation_damping,
        joint_limit_scale=joint_limit_scale,
        requested_to_predicted_task_l2=requested_to_predicted,
        intervened=intervention,
        audit=audit,
    )
    result.validate()
    return result


__all__ = [
    "BOUND_CONSTRAINED_HIERARCHICAL_DLS_FORMAT_V755",
    "JOINT_MARGIN_AWARE_DLS_FORMAT_V746",
    "TASKFRAME_ACTION_DIM_V688",
    "TASKFRAME_DLS_JOINT_ACTION_FORMAT_V688",
    "TaskFrameDLSJointActionConfigV688",
    "TaskFrameDLSJointActionV688",
    "applied_task_action_from_tool_delta_v688",
    "bound_constrained_hierarchical_delta_v755",
    "project_taskframe_action_v688",
]
