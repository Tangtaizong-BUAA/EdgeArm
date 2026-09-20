"""Fine-grained evidence audit for Home-to-contact acquisition rollouts."""

from __future__ import annotations

from typing import Any

import numpy as np

from .privileged_effect_state_v1 import (
    PRIVILEGED_EFFECT_STATE_DIM,
    privileged_effect_state_slices_v1,
)


ACQUISITION_ROLLOUT_AUDIT_FORMAT_V642 = (
    "edgearm-v642-home-acquisition-motion-audit-v1"
)
_SLICES_V642 = privileged_effect_state_slices_v1()


def _geometry_v642(
    state: np.ndarray,
    desired_goal: np.ndarray,
    *,
    precontact_standoff_m: float,
    precontact_tool_height_m: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    block_pose = state[:, _SLICES_V642["block_pose_xyz_quaternion_wxyz"]]
    tool_pose = state[:, _SLICES_V642["tool_pose_position_rotation"]]
    block_xy = block_pose[:, :2]
    direction = desired_goal - block_xy
    norm = np.linalg.norm(direction, axis=-1, keepdims=True)
    fallback = np.zeros_like(direction)
    fallback[:, 0] = 1.0
    forward = np.where(norm > 1.0e-7, direction / np.maximum(norm, 1.0e-7), fallback)
    precontact = np.concatenate(
        (
            block_xy - precontact_standoff_m * forward,
            np.full((len(state), 1), precontact_tool_height_m, dtype=np.float64),
        ),
        axis=-1,
    )
    tool_xyz = tool_pose[:, :3]
    distance = np.linalg.norm(tool_xyz - precontact, axis=-1)
    rotation = tool_pose[:, 3:12].reshape(-1, 3, 3)
    broad_face_xy = rotation[:, :, 1][:, :2]
    horizontal_norm = np.linalg.norm(broad_face_xy, axis=-1)
    alignment = np.abs(np.sum(broad_face_xy * forward, axis=-1))
    alignment /= np.maximum(horizontal_norm, 1.0e-7)
    return (
        distance,
        np.clip(alignment, 0.0, 1.0),
        tool_xyz,
        precontact,
    )


def acquisition_rollout_audit_v642(
    neutral_state: np.ndarray,
    next_neutral_state: np.ndarray,
    desired_goal: np.ndarray,
    valid_contact: np.ndarray,
    action_feasible: np.ndarray,
    action: np.ndarray,
    applied_action: np.ndarray,
    *,
    precontact_standoff_m: float = 0.055,
    precontact_tool_height_m: float = 0.055,
    contact_ready_distance_m: float = 0.020,
    aligned_threshold: float = 0.90,
    meaningful_best_progress_m: float = 0.005,
    regression_tolerance_m: float = 0.010,
) -> dict[str, Any]:
    """Classify why a rollout did or did not acquire valid contact.

    The audit consumes only logged transition state.  It does not supply an
    action, waypoint sequence, expert label, or controller override.
    """

    state = np.asarray(neutral_state, dtype=np.float64)
    next_state = np.asarray(next_neutral_state, dtype=np.float64)
    goal = np.asarray(desired_goal, dtype=np.float64)
    contact = np.asarray(valid_contact, dtype=bool)
    feasible = np.asarray(action_feasible, dtype=bool)
    proposed = np.asarray(action, dtype=np.float64)
    applied = np.asarray(applied_action, dtype=np.float64)
    count = len(state)
    if (
        count < 1
        or state.shape != (count, PRIVILEGED_EFFECT_STATE_DIM)
        or next_state.shape != state.shape
        or goal.shape != (count, 2)
        or contact.shape != (count,)
        or feasible.shape != (count,)
        or proposed.shape != (count, 3)
        or applied.shape != (count, 3)
        or not all(
            np.all(np.isfinite(value))
            for value in (state, next_state, goal, proposed, applied)
        )
        or not np.allclose(goal, goal[0], atol=1.0e-7, rtol=0.0)
    ):
        raise ValueError("V642 rollout arrays are invalid")
    thresholds = np.asarray(
        [
            precontact_standoff_m,
            precontact_tool_height_m,
            contact_ready_distance_m,
            aligned_threshold,
            meaningful_best_progress_m,
            regression_tolerance_m,
        ],
        dtype=np.float64,
    )
    if (
        not np.all(np.isfinite(thresholds))
        or np.any(thresholds <= 0.0)
        or contact_ready_distance_m >= 0.120
        or aligned_threshold > 1.0
    ):
        raise ValueError("V642 audit thresholds are invalid")

    before_distance, before_alignment, before_tool, before_precontact = (
        _geometry_v642(
            state,
            goal,
            precontact_standoff_m=precontact_standoff_m,
            precontact_tool_height_m=precontact_tool_height_m,
        )
    )
    after_distance, after_alignment, after_tool, _after_precontact = _geometry_v642(
        next_state,
        goal,
        precontact_standoff_m=precontact_standoff_m,
        precontact_tool_height_m=precontact_tool_height_m,
    )
    trajectory_distance = np.concatenate((before_distance[:1], after_distance))
    trajectory_alignment = np.concatenate((before_alignment[:1], after_alignment))
    trajectory_tool = np.concatenate((before_tool[:1], after_tool), axis=0)
    minimum_index = int(np.argmin(trajectory_distance))
    initial_distance = float(trajectory_distance[0])
    final_distance = float(trajectory_distance[-1])
    minimum_distance = float(trajectory_distance[minimum_index])
    best_progress = initial_distance - minimum_distance
    net_progress = initial_distance - final_distance
    step_progress = before_distance - after_distance
    tool_step = np.linalg.norm(np.diff(trajectory_tool, axis=0), axis=-1)
    tool_path_length = float(np.sum(tool_step))
    contact_rows = np.flatnonzero(contact)
    first_contact_step = None if not len(contact_rows) else int(contact_rows[0])
    alignment_at_minimum = float(trajectory_alignment[minimum_index])

    if first_contact_step is not None:
        diagnosis = "valid_contact_acquired"
    elif minimum_distance > 0.120:
        diagnosis = "never_entered_acquisition_region"
    elif best_progress < meaningful_best_progress_m:
        diagnosis = "no_meaningful_precontact_progress"
    elif minimum_distance > contact_ready_distance_m:
        if final_distance > minimum_distance + regression_tolerance_m:
            diagnosis = "approach_then_regressed_before_contact"
        else:
            diagnosis = "approach_stalled_outside_contact_ready_region"
    elif alignment_at_minimum < aligned_threshold:
        diagnosis = "contact_ready_position_but_face_misaligned"
    else:
        diagnosis = "contact_ready_geometry_without_valid_contact"

    return {
        "format": ACQUISITION_ROLLOUT_AUDIT_FORMAT_V642,
        "diagnosis": diagnosis,
        "row_count": count,
        "initial_tool_precontact_distance_m": initial_distance,
        "final_tool_precontact_distance_m": final_distance,
        "minimum_tool_precontact_distance_m": minimum_distance,
        "minimum_distance_step": minimum_index,
        "best_tool_precontact_progress_m": best_progress,
        "net_tool_precontact_progress_m": net_progress,
        "tool_path_length_m": tool_path_length,
        "best_progress_over_path_length": (
            0.0 if tool_path_length <= 1.0e-12 else best_progress / tool_path_length
        ),
        "positive_precontact_progress_step_fraction": float(
            np.mean(step_progress > 1.0e-5)
        ),
        "negative_precontact_progress_step_fraction": float(
            np.mean(step_progress < -1.0e-5)
        ),
        "alignment_at_minimum_distance": alignment_at_minimum,
        "maximum_precontact_face_alignment": float(np.max(trajectory_alignment)),
        "first_valid_contact_step": first_contact_step,
        "valid_contact_steps": int(np.count_nonzero(contact)),
        "action_feasible_fraction": float(np.mean(feasible)),
        "mean_proposed_action": np.mean(proposed, axis=0).tolist(),
        "mean_applied_action": np.mean(applied, axis=0).tolist(),
        "mean_action_projection_l2": float(
            np.mean(np.linalg.norm(proposed - applied, axis=-1))
        ),
        "initial_tool_xyz_m": trajectory_tool[0].tolist(),
        "closest_tool_xyz_m": trajectory_tool[minimum_index].tolist(),
        "initial_precontact_goal_xyz_m": before_precontact[0].tolist(),
        "contact_ready_distance_m": float(contact_ready_distance_m),
        "aligned_threshold": float(aligned_threshold),
        "contains_no_expert_action_or_waypoint_path": True,
        "production_admission": False,
    }


__all__ = [
    "ACQUISITION_ROLLOUT_AUDIT_FORMAT_V642",
    "acquisition_rollout_audit_v642",
]
