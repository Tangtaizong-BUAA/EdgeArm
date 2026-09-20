"""Planar task-priority joint action projection for stock-gripper pushing.

V667 keeps the broad pushing face vertical and target-aligned but deliberately
leaves XYZ motion to the policy.  Failure replay showed that random vertical
motion can lift the tool from the audited 47--52 mm reset band to roughly
65--70 mm and turn otherwise aligned side contact into top-edge contact.

For the current obstacle-free table task, vertical exploration has no useful
task authority.  V668 therefore adds the tool-site height as a third local
constraint.  Five raw policy joint deltas are projected onto a two-dimensional
planar tangent space while a task-priority correction maintains:

1. a vertical broad pushing face;
2. a face normal parallel to the block-to-target direction; and
3. the 50 mm centre height of the audited stock-gripper reset candidates.

The policy still emits five normalized joint deltas and the plant still
records all five executed servo trajectories.  This module contains no route,
expert action, IK waypoint, phase label, or success shortcut.
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from .guarded_joint_delta_action_v664 import ARM_JOINT_ACTION_DIM_V664
from .orientation_projected_joint_action_v667 import (
    orientation_constraint_state_v667,
)
from .sim2real_env_v10 import RealisticEdgeArmEnvV10


PLANAR_PUSH_PROJECTED_JOINT_ACTION_FORMAT_V668 = (
    "edgearm-v668-task-priority-planar-push-projected-joint-action-v1"
)


@dataclass(frozen=True)
class PlanarPushProjectedJointActionConfigV668:
    policy_joint_target_step_rad: float = 0.025
    constraint_correction_gain: float = 0.35
    damped_pseudoinverse_lambda: float = 0.010
    target_tool_height_m: float = 0.050
    height_residual_scale_m: float = 0.020
    intervention_l2_threshold: float = 1.0e-6

    def validate(self) -> None:
        positive = np.asarray(
            [
                self.policy_joint_target_step_rad,
                self.constraint_correction_gain,
                self.damped_pseudoinverse_lambda,
                self.target_tool_height_m,
                self.height_residual_scale_m,
                self.intervention_l2_threshold,
            ],
            dtype=np.float64,
        )
        if not np.all(np.isfinite(positive)) or np.any(positive <= 0.0):
            raise ValueError("V668 projection configuration must be positive")
        if self.constraint_correction_gain > 1.0:
            raise ValueError("V668 constraint correction gain exceeds one")
        if not 0.0455 <= self.target_tool_height_m <= 0.0580:
            raise ValueError(
                "V668 target height must remain inside the audited V12 band"
            )


@dataclass(frozen=True)
class PlanarPushConstraintStateV668:
    residual: np.ndarray
    jacobian: np.ndarray
    tool_face_axis_world: np.ndarray
    intended_push_direction_xy: np.ndarray
    tool_face_horizontal_norm: float
    tool_face_push_alignment: float
    tool_height_m: float
    target_tool_height_m: float
    tool_height_error_m: float
    format: str = PLANAR_PUSH_PROJECTED_JOINT_ACTION_FORMAT_V668


@dataclass(frozen=True)
class PlanarPushProjectedJointActionV668:
    requested_action: np.ndarray
    projected_action: np.ndarray
    requested_joint_delta_rad: np.ndarray
    projected_joint_delta_rad: np.ndarray
    constraint_residual_before: np.ndarray
    predicted_linearized_residual_after: np.ndarray
    projection_l2: float
    intervened: bool
    tool_face_horizontal_norm_before: float
    tool_face_push_alignment_before: float
    tool_height_m_before: float
    target_tool_height_m: float
    tool_height_error_m_before: float
    format: str = PLANAR_PUSH_PROJECTED_JOINT_ACTION_FORMAT_V668


def planar_push_constraint_state_v668(
    env: RealisticEdgeArmEnvV10,
    config: PlanarPushProjectedJointActionConfigV668 | None = None,
) -> PlanarPushConstraintStateV668:
    """Return three normalized planar-push constraints and exact Jacobian."""

    if type(env) is not RealisticEdgeArmEnvV10:
        raise TypeError("V668 planar state requires exact V10 environment")
    selected = config or PlanarPushProjectedJointActionConfigV668()
    selected.validate()
    orientation = orientation_constraint_state_v667(env)
    site_id = int(env._ids["tool_site"])
    jacobian_position = np.zeros((3, env.model.nv), dtype=np.float64)
    jacobian_rotation = np.zeros((3, env.model.nv), dtype=np.float64)
    mujoco.mj_jacSite(
        env.model,
        env.data,
        jacobian_position,
        jacobian_rotation,
        site_id,
    )
    del jacobian_rotation
    tool_height = float(env.data.site_xpos[site_id, 2])
    height_error = tool_height - selected.target_tool_height_m
    residual = np.r_[
        orientation.residual,
        height_error / selected.height_residual_scale_m,
    ].astype(np.float64)
    jacobian = np.vstack(
        (
            orientation.jacobian,
            jacobian_position[
                2, :ARM_JOINT_ACTION_DIM_V664
            ]
            / selected.height_residual_scale_m,
        )
    )
    if (
        residual.shape != (3,)
        or jacobian.shape != (3, ARM_JOINT_ACTION_DIM_V664)
        or not np.all(
            np.isfinite(
                np.r_[
                    residual,
                    jacobian.ravel(),
                    tool_height,
                    height_error,
                ]
            )
        )
    ):
        raise RuntimeError("V668 planar constraint state is invalid")
    return PlanarPushConstraintStateV668(
        residual=residual,
        jacobian=jacobian,
        tool_face_axis_world=orientation.tool_face_axis_world,
        intended_push_direction_xy=orientation.intended_push_direction_xy,
        tool_face_horizontal_norm=orientation.tool_face_horizontal_norm,
        tool_face_push_alignment=orientation.tool_face_push_alignment,
        tool_height_m=tool_height,
        target_tool_height_m=selected.target_tool_height_m,
        tool_height_error_m=height_error,
    )


def project_planar_push_joint_action_v668(
    env: RealisticEdgeArmEnvV10,
    requested_action: np.ndarray,
    config: PlanarPushProjectedJointActionConfigV668 | None = None,
) -> PlanarPushProjectedJointActionV668:
    """Project one five-joint policy delta into the planar push tangent."""

    selected = config or PlanarPushProjectedJointActionConfigV668()
    selected.validate()
    requested = np.asarray(requested_action, dtype=np.float64)
    if (
        requested.shape != (ARM_JOINT_ACTION_DIM_V664,)
        or not np.all(np.isfinite(requested))
        or np.any(np.abs(requested) > 1.0 + 1.0e-6)
    ):
        raise ValueError("V668 requested action must be finite normalized [5]")
    state = planar_push_constraint_state_v668(env, selected)
    jacobian = state.jacobian
    damping_square = selected.damped_pseudoinverse_lambda**2
    damped_inverse = np.linalg.solve(
        jacobian @ jacobian.T + damping_square * np.eye(3),
        np.eye(3),
    )
    pseudoinverse = jacobian.T @ damped_inverse
    null_projection = (
        np.eye(ARM_JOINT_ACTION_DIM_V664) - pseudoinverse @ jacobian
    )
    requested_delta = requested * selected.policy_joint_target_step_rad
    correction = (
        -selected.constraint_correction_gain
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
        raise RuntimeError("V668 projected action became non-finite")
    return PlanarPushProjectedJointActionV668(
        requested_action=requested.astype(np.float32),
        projected_action=projected.astype(np.float32),
        requested_joint_delta_rad=requested_delta.astype(np.float32),
        projected_joint_delta_rad=projected_delta.astype(np.float32),
        constraint_residual_before=state.residual.astype(np.float32),
        predicted_linearized_residual_after=predicted_residual.astype(
            np.float32
        ),
        projection_l2=projection_l2,
        intervened=bool(
            projection_l2 > selected.intervention_l2_threshold
        ),
        tool_face_horizontal_norm_before=(
            state.tool_face_horizontal_norm
        ),
        tool_face_push_alignment_before=state.tool_face_push_alignment,
        tool_height_m_before=state.tool_height_m,
        target_tool_height_m=state.target_tool_height_m,
        tool_height_error_m_before=state.tool_height_error_m,
    )


__all__ = [
    "PLANAR_PUSH_PROJECTED_JOINT_ACTION_FORMAT_V668",
    "PlanarPushConstraintStateV668",
    "PlanarPushProjectedJointActionConfigV668",
    "PlanarPushProjectedJointActionV668",
    "planar_push_constraint_state_v668",
    "project_planar_push_joint_action_v668",
]
