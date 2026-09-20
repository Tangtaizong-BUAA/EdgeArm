"""Task-priority joint action projection for the stock SO-101 gripper.

The five arm joints are not redundant for a full six-dimensional tool pose,
but the push task only needs two orientation constraints: the broad pushing
face must remain vertical and its horizontal normal must stay parallel to the
block-to-target direction.  These two constraints leave a three-dimensional
local null space in which RL can discover XYZ motion.

The policy still emits five normalized joint deltas.  A damped task-priority
projection executes the closest local action composed of an orientation
correction plus the policy motion that lies in the constraint null space.  It
contains no route, expert action, IK waypoint, or phase-labelled controller.
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from .guarded_joint_delta_action_v664 import ARM_JOINT_ACTION_DIM_V664
from .sim2real_env_v10 import RealisticEdgeArmEnvV10


ORIENTATION_PROJECTED_JOINT_ACTION_FORMAT_V667 = (
    "edgearm-v667-task-priority-orientation-projected-joint-action-v1"
)


@dataclass(frozen=True)
class OrientationProjectedJointActionConfigV667:
    policy_joint_target_step_rad: float = 0.025
    orientation_correction_gain: float = 0.35
    damped_pseudoinverse_lambda: float = 0.010
    intervention_l2_threshold: float = 1.0e-6
    direction_norm_epsilon_m: float = 1.0e-7

    def validate(self) -> None:
        positive = np.asarray(
            [
                self.policy_joint_target_step_rad,
                self.orientation_correction_gain,
                self.damped_pseudoinverse_lambda,
                self.intervention_l2_threshold,
                self.direction_norm_epsilon_m,
            ],
            dtype=np.float64,
        )
        if not np.all(np.isfinite(positive)) or np.any(positive <= 0.0):
            raise ValueError("V667 projection configuration must be positive")
        if self.orientation_correction_gain > 1.0:
            raise ValueError("V667 orientation correction gain exceeds one")


@dataclass(frozen=True)
class OrientationConstraintStateV667:
    residual: np.ndarray
    jacobian: np.ndarray
    tool_face_axis_world: np.ndarray
    intended_push_direction_xy: np.ndarray
    tool_face_horizontal_norm: float
    tool_face_push_alignment: float
    format: str = ORIENTATION_PROJECTED_JOINT_ACTION_FORMAT_V667


@dataclass(frozen=True)
class OrientationProjectedJointActionV667:
    requested_action: np.ndarray
    projected_action: np.ndarray
    requested_joint_delta_rad: np.ndarray
    projected_joint_delta_rad: np.ndarray
    orientation_residual_before: np.ndarray
    predicted_linearized_residual_after: np.ndarray
    projection_l2: float
    intervened: bool
    tool_face_horizontal_norm_before: float
    tool_face_push_alignment_before: float
    format: str = ORIENTATION_PROJECTED_JOINT_ACTION_FORMAT_V667


def orientation_constraint_state_v667(
    env: RealisticEdgeArmEnvV10,
    *,
    directional_face_yaw_v778: bool = False,
) -> OrientationConstraintStateV667:
    """Return the two task constraints and their exact MuJoCo Jacobian."""

    if type(env) is not RealisticEdgeArmEnvV10:
        raise TypeError("V667 orientation state requires exact V10 environment")
    direction = np.asarray(env.target_xy - env.block_xy(), dtype=np.float64)
    direction_norm = float(np.linalg.norm(direction))
    if not np.isfinite(direction_norm) or direction_norm <= 1.0e-7:
        raise RuntimeError("V667 block-to-target direction is degenerate")
    direction /= direction_norm
    perpendicular = np.asarray(
        [-direction[1], direction[0], 0.0], dtype=np.float64
    )
    site_id = int(env._ids["tool_site"])
    rotation = np.asarray(env.data.site_xmat[site_id], dtype=np.float64).reshape(
        3, 3
    )
    face_axis = rotation[:, 1].copy()
    jacobian_position = np.zeros((3, env.model.nv), dtype=np.float64)
    jacobian_rotation = np.zeros((3, env.model.nv), dtype=np.float64)
    mujoco.mj_jacSite(
        env.model,
        env.data,
        jacobian_position,
        jacobian_rotation,
        site_id,
    )
    skew_face = np.asarray(
        [
            [0.0, -face_axis[2], face_axis[1]],
            [face_axis[2], 0.0, -face_axis[0]],
            [-face_axis[1], face_axis[0], 0.0],
        ],
        dtype=np.float64,
    )
    face_jacobian = -skew_face @ jacobian_rotation[
        :, :ARM_JOINT_ACTION_DIM_V664
    ]
    lateral_component = float(np.dot(perpendicular, face_axis))
    forward_3d = np.asarray([direction[0], direction[1], 0.0], dtype=np.float64)
    forward_component = float(np.dot(forward_3d, face_axis))
    if directional_face_yaw_v778:
        yaw_residual = float(np.arctan2(lateral_component, forward_component))
        lateral_jacobian = perpendicular @ face_jacobian
        forward_jacobian = forward_3d @ face_jacobian
        yaw_jacobian = (
            forward_component * lateral_jacobian
            - lateral_component * forward_jacobian
        ) / max(
            forward_component * forward_component
            + lateral_component * lateral_component,
            1.0e-12,
        )
        residual = np.asarray([face_axis[2], yaw_residual], dtype=np.float64)
        constraint_jacobian = np.vstack((face_jacobian[2], yaw_jacobian))
    else:
        residual = np.asarray(
            [face_axis[2], lateral_component],
            dtype=np.float64,
        )
        constraint_jacobian = np.vstack(
            (face_jacobian[2], perpendicular @ face_jacobian)
        )
    horizontal_norm = float(np.linalg.norm(face_axis[:2]))
    push_alignment = abs(float(np.dot(face_axis[:2], direction))) / max(
        horizontal_norm, 1.0e-12
    )
    if (
        residual.shape != (2,)
        or constraint_jacobian.shape
        != (2, ARM_JOINT_ACTION_DIM_V664)
        or not np.all(
            np.isfinite(
                np.r_[
                    residual,
                    constraint_jacobian.ravel(),
                    face_axis,
                    direction,
                    horizontal_norm,
                    push_alignment,
                ]
            )
        )
    ):
        raise RuntimeError("V667 orientation constraint state is invalid")
    return OrientationConstraintStateV667(
        residual=residual,
        jacobian=constraint_jacobian,
        tool_face_axis_world=face_axis,
        intended_push_direction_xy=direction,
        tool_face_horizontal_norm=horizontal_norm,
        tool_face_push_alignment=float(np.clip(push_alignment, 0.0, 1.0)),
    )


def project_joint_action_v667(
    env: RealisticEdgeArmEnvV10,
    requested_action: np.ndarray,
    config: OrientationProjectedJointActionConfigV667 | None = None,
) -> OrientationProjectedJointActionV667:
    """Project one policy delta into the local orientation-safe null space."""

    selected = config or OrientationProjectedJointActionConfigV667()
    selected.validate()
    requested = np.asarray(requested_action, dtype=np.float64)
    if (
        requested.shape != (ARM_JOINT_ACTION_DIM_V664,)
        or not np.all(np.isfinite(requested))
        or np.any(np.abs(requested) > 1.0 + 1.0e-6)
    ):
        raise ValueError("V667 requested action must be finite normalized [5]")
    state = orientation_constraint_state_v667(env)
    jacobian = state.jacobian
    damping_square = selected.damped_pseudoinverse_lambda**2
    damped_inverse = np.linalg.solve(
        jacobian @ jacobian.T + damping_square * np.eye(2),
        np.eye(2),
    )
    pseudoinverse = jacobian.T @ damped_inverse
    null_projection = (
        np.eye(ARM_JOINT_ACTION_DIM_V664) - pseudoinverse @ jacobian
    )
    requested_delta = requested * selected.policy_joint_target_step_rad
    correction = (
        -selected.orientation_correction_gain
        * pseudoinverse
        @ state.residual
    )
    projected_delta = correction + null_projection @ requested_delta
    maximum_absolute_delta = float(np.max(np.abs(projected_delta)))
    if maximum_absolute_delta > selected.policy_joint_target_step_rad:
        projected_delta *= (
            selected.policy_joint_target_step_rad / maximum_absolute_delta
        )
    current_joint = np.asarray(
        env.data.qpos[:ARM_JOINT_ACTION_DIM_V664], dtype=np.float64
    )
    lower = np.asarray(
        env.model.jnt_range[:ARM_JOINT_ACTION_DIM_V664, 0], dtype=np.float64
    )
    upper = np.asarray(
        env.model.jnt_range[:ARM_JOINT_ACTION_DIM_V664, 1], dtype=np.float64
    )
    target_joint = np.clip(current_joint + projected_delta, lower, upper)
    projected_delta = target_joint - current_joint
    projected = np.clip(
        projected_delta / selected.policy_joint_target_step_rad,
        -1.0,
        1.0,
    )
    predicted_residual = state.residual + jacobian @ projected_delta
    projection_l2 = float(np.linalg.norm(requested - projected))
    if not np.all(
        np.isfinite(
            np.r_[
                projected,
                requested_delta,
                projected_delta,
                predicted_residual,
                projection_l2,
            ]
        )
    ):
        raise RuntimeError("V667 projected action became non-finite")
    return OrientationProjectedJointActionV667(
        requested_action=requested.astype(np.float32),
        projected_action=projected.astype(np.float32),
        requested_joint_delta_rad=requested_delta.astype(np.float32),
        projected_joint_delta_rad=projected_delta.astype(np.float32),
        orientation_residual_before=state.residual.astype(np.float32),
        predicted_linearized_residual_after=predicted_residual.astype(
            np.float32
        ),
        projection_l2=projection_l2,
        intervened=bool(
            projection_l2 > selected.intervention_l2_threshold
        ),
        tool_face_horizontal_norm_before=state.tool_face_horizontal_norm,
        tool_face_push_alignment_before=state.tool_face_push_alignment,
    )


__all__ = [
    "ORIENTATION_PROJECTED_JOINT_ACTION_FORMAT_V667",
    "OrientationConstraintStateV667",
    "OrientationProjectedJointActionConfigV667",
    "OrientationProjectedJointActionV667",
    "orientation_constraint_state_v667",
    "project_joint_action_v667",
]
