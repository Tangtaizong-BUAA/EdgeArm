"""Goal-conditioned off-policy RL with persistent replay and future HER.

V43 is a simulator-privileged RL teacher used to make the no-obstacle push
task learnable before wrist-image distillation.  It deliberately separates
three identities that older one-batch PPO experiments conflated:

* simulator transitions (successful and failed) are reusable RL experience;
* future-HER goals are learning-only counterfactuals;
* only the environment's exact three-second strict success is eligible to
  become a successful generated trajectory.

The actor never receives an expert action or path.  The critic and actor use a
goal-neutral form of the V7 privileged effect state plus an explicit desired
goal.  This is an RL data-generator teacher, not a deployable camera policy.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import h5py
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .privileged_effect_state_v1 import (
    PRIVILEGED_EFFECT_STATE_DIM,
    privileged_effect_state_slices_v1,
)
from .se_rl_safeguard_projection_v720 import (
    squared_safeguard_intervention_penalty_v720,
)


GOAL_CONDITIONED_HER_SAC_FORMAT_V43 = (
    "edgearm-v43-goal-conditioned-persistent-her-sac-v1"
)
GOAL_CONDITIONED_HER_REPLAY_FORMAT_V43 = (
    "edgearm-v43-goal-conditioned-persistent-replay-v1"
)
GOAL_CONDITIONED_HER_CHECKPOINT_FORMAT_V43 = (
    "edgearm-v43-goal-conditioned-her-sac-checkpoint-v1"
)

ACTION_DIM_V43 = 3
GOAL_DIM_V43 = 2
OBSERVATION_DIM_V43 = PRIVILEGED_EFFECT_STATE_DIM + GOAL_DIM_V43
FORWARD_ACTION_STEP_M_V43 = 0.0015
LATERAL_ACTION_STEP_M_V43 = 0.0010

_SLICES = privileged_effect_state_slices_v1()
_TARGET_DEPENDENT_FIELDS = (
    "target_xy_m",
    "strict_target_coverage",
    "strict_success_streak_over_hold_steps",
    "block_target_distance_m",
)


def goal_neutral_privileged_state_v43(state: np.ndarray) -> np.ndarray:
    """Remove goal-derived values before an explicit goal is appended.

    This makes future-goal relabelling internally consistent: an HER sample
    cannot retain the original target, distance, coverage, or success streak
    in a hidden privileged slot.
    """

    value = np.asarray(state, dtype=np.float32)
    if value.shape[-1] != PRIVILEGED_EFFECT_STATE_DIM:
        raise ValueError("V43 privileged state has the wrong final dimension")
    if not np.all(np.isfinite(value)):
        raise ValueError("V43 privileged state contains non-finite values")
    result = value.copy()
    for name in _TARGET_DEPENDENT_FIELDS:
        result[..., _SLICES[name]] = np.float32(0.0)
    return result


def achieved_goal_from_privileged_v43(state: np.ndarray) -> np.ndarray:
    value = np.asarray(state, dtype=np.float32)
    if value.shape[-1] != PRIVILEGED_EFFECT_STATE_DIM:
        raise ValueError("V43 goal extraction received the wrong state shape")
    block = value[..., _SLICES["block_pose_xyz_quaternion_wxyz"]]
    return block[..., :2].astype(np.float32, copy=True)


def desired_goal_from_privileged_v43(state: np.ndarray) -> np.ndarray:
    value = np.asarray(state, dtype=np.float32)
    if value.shape[-1] != PRIVILEGED_EFFECT_STATE_DIM:
        raise ValueError("V43 target extraction received the wrong state shape")
    return value[..., _SLICES["target_xy_m"]].astype(np.float32, copy=True)


def observation_with_goal_v43(
    neutral_state: np.ndarray,
    desired_goal: np.ndarray,
) -> np.ndarray:
    state = np.asarray(neutral_state, dtype=np.float32)
    goal = np.asarray(desired_goal, dtype=np.float32)
    if state.shape[:-1] != goal.shape[:-1] or state.shape[-1] != PRIVILEGED_EFFECT_STATE_DIM:
        raise ValueError("V43 state and desired-goal batch shapes disagree")
    if goal.shape[-1] != GOAL_DIM_V43:
        raise ValueError("V43 desired goal must be planar")
    result = np.concatenate((state, goal), axis=-1, dtype=np.float32)
    if result.shape[-1] != OBSERVATION_DIM_V43 or not np.all(np.isfinite(result)):
        raise RuntimeError("V43 goal-conditioned observation is invalid")
    return result


def _unit_direction(delta: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(delta, axis=-1, keepdims=True)
    fallback = np.zeros_like(delta)
    fallback[..., 0] = 1.0
    return np.where(norm > 1.0e-7, delta / np.maximum(norm, 1.0e-7), fallback)


def reframe_task_action_for_goal_v43(
    action: np.ndarray,
    achieved_goal: np.ndarray,
    original_goal: np.ndarray,
    relabelled_goal: np.ndarray,
) -> np.ndarray:
    """Rotate task-frame XY actions when HER changes the goal direction.

    The V22 adapter defines forward/lateral from block to target.  Reusing an
    action under a new HER target without this rotation would silently change
    its world-space meaning.
    """

    source_action = np.asarray(action, dtype=np.float32)
    achieved = np.asarray(achieved_goal, dtype=np.float32)
    original = np.asarray(original_goal, dtype=np.float32)
    relabelled = np.asarray(relabelled_goal, dtype=np.float32)
    if (
        source_action.shape[-1] != ACTION_DIM_V43
        or achieved.shape[-1] != GOAL_DIM_V43
        or original.shape != achieved.shape
        or relabelled.shape != achieved.shape
        or source_action.shape[:-1] != achieved.shape[:-1]
    ):
        raise ValueError("V43 action reframing shapes disagree")
    old_forward = _unit_direction(original - achieved)
    old_lateral = np.stack((-old_forward[..., 1], old_forward[..., 0]), axis=-1)
    new_forward = _unit_direction(relabelled - achieved)
    new_lateral = np.stack((-new_forward[..., 1], new_forward[..., 0]), axis=-1)
    world_xy = (
        old_forward * source_action[..., :1] * FORWARD_ACTION_STEP_M_V43
        + old_lateral * source_action[..., 1:2] * LATERAL_ACTION_STEP_M_V43
    )
    result = source_action.copy()
    result[..., 0] = np.sum(world_xy * new_forward, axis=-1) / FORWARD_ACTION_STEP_M_V43
    result[..., 1] = np.sum(world_xy * new_lateral, axis=-1) / LATERAL_ACTION_STEP_M_V43
    return np.clip(result, -1.0, 1.0).astype(np.float32)


@dataclass(frozen=True)
class GoalConditionedHerSACConfigV43:
    gamma: float = 0.99
    target_tau: float = 0.005
    entropy_temperature: float = 0.08
    actor_learning_rate: float = 3.0e-4
    critic_learning_rate: float = 3.0e-4
    feasibility_learning_rate: float = 3.0e-4
    batch_size: int = 256
    hidden_dim: int = 256
    future_relabel_probability: float = 0.80
    minimum_future_displacement_m: float = 0.0010
    her_success_distance_m: float = 0.0060
    contained_distance_m: float = 0.0250
    progress_scale_m: float = 0.0010
    maximum_progress_reward: float = 2.0
    contact_bonus: float = 0.20
    # V509 changes the contact term from an unconditional occupancy bonus into
    # an outcome-shaped phase signal.  It is opt-in so historical checkpoints
    # preserve their exact learning semantics when loaded.
    phase_progress_reward_active: bool = False
    contact_progress_threshold_m: float = 2.0e-5
    contact_acquisition_bonus: float = 0.05
    stalled_contact_penalty: float = 0.10
    target_coverage_gain_bonus: float = 1.0
    strict_hold_progress_bonus: float = 2.0
    # V511 teaches the critic that an unfinished push may not silently drop
    # contact at the target-entry boundary.  Coverage/hold-complete release is
    # explicitly exempt so the learned reward does not fight the strict
    # terminal state machine.
    phase_retention_reward_active: bool = False
    contact_loss_penalty: float = 0.25
    target_coverage_loss_penalty: float = 1.50
    contact_loss_priority: float = 8.0
    target_entry_priority: float = 6.0
    target_entry_coverage_threshold: float = 0.01
    terminal_release_coverage_threshold: float = 0.95
    # V606 makes the already-required three-second terminal hold learnable.
    # Once the original task reaches strict coverage, large actions are
    # penalized and those rows are replay-prioritized.  No action is supplied;
    # the actor must discover a low-motion command from its own transitions.
    target_settle_action_penalty: float = 0.0
    target_settle_priority: float = 0.0
    target_settle_coverage_threshold: float = 0.95
    # Task-metric geometry is replay metadata, not part of the actor
    # observation.  These values allow legacy neutralized NPZ files to be
    # upgraded without leaking target-derived fields back into the policy.
    coverage_block_half_extent_m: float = 0.025
    coverage_target_radius_m: float = 0.055
    coverage_grid_resolution: int = 17
    strict_coverage_threshold: float = 0.95
    strict_linear_speed_m_s: float = 0.025
    strict_angular_speed_rad_s: float = 0.55
    strict_success_hold_steps: int = 90
    contained_bonus: float = 0.12
    her_success_bonus: float = 2.0
    strict_success_bonus: float = 10.0
    action_penalty: float = 0.008
    projection_penalty: float = 0.10
    # V720 separates the safety projection from delayed plant response.  The
    # legacy scalar ||proposal-live_effect|| remains available for diagnostics,
    # but it is not a valid action-aliasing penalty.
    safeguard_projection_penalty_v720: bool = False
    infeasible_action_penalty: float = 0.60
    safety_penalty: float = 3.0
    # V598 restores a task-independent Home start. Before first contact the
    # object remains stationary, so object-goal HER cannot credit reaching.
    # This opt-in auxiliary intention rewards geometric progress toward a
    # pre-contact state without supplying an action sequence or route. It is
    # applied only to original goals; future-HER remains object-goal-only.
    home_acquisition_reward_active: bool = False
    home_acquisition_standoff_m: float = 0.055
    home_acquisition_tool_height_m: float = 0.055
    home_acquisition_progress_scale_m: float = 0.0015
    home_acquisition_maximum_progress_reward: float = 0.30
    home_acquisition_alignment_activation_distance_m: float = 0.120
    home_acquisition_alignment_progress_scale: float = 0.020
    home_acquisition_maximum_alignment_reward: float = 0.08
    home_acquisition_credit_priority: float = 4.0
    home_acquisition_suppress_future_her_above_distance_m: float = 0.0
    home_acquisition_suppress_future_her_above_tool_height_m: float = 0.0
    home_acquisition_suppress_future_her_below_alignment: float = 0.0
    actor_infeasibility_coefficient: float = 0.30
    critic_conservative_coefficient: float = 0.01
    maximum_gradient_norm: float = 10.0
    policy_update_period: int = 2
    # Contact-Prioritized Experience Replay (CPER): retain a short causal
    # window before every contact/object-motion transition.  Those rows teach
    # the approach that made HER informative, while long no-contact and
    # terminal-settle tails remain available without dominating updates.
    contact_credit_preceding_steps: int = 8
    contact_credit_priority: float = 8.0
    successful_contact_credit_priority: float = 4.0
    effectful_progress_minimum_m: float = 1.0e-5

    def validate(self) -> None:
        probabilities = (
            self.gamma,
            self.target_tau,
            self.future_relabel_probability,
        )
        if any(not np.isfinite(value) or not 0.0 < value <= 1.0 for value in probabilities):
            raise ValueError("V43 probability/discount configuration is invalid")
        positives = (
            self.entropy_temperature,
            self.actor_learning_rate,
            self.critic_learning_rate,
            self.feasibility_learning_rate,
            self.minimum_future_displacement_m,
            self.her_success_distance_m,
            self.contained_distance_m,
            self.progress_scale_m,
            self.maximum_gradient_norm,
            self.contact_credit_priority,
            self.successful_contact_credit_priority,
            self.effectful_progress_minimum_m,
            self.home_acquisition_standoff_m,
            self.home_acquisition_tool_height_m,
            self.home_acquisition_progress_scale_m,
            self.home_acquisition_maximum_progress_reward,
            self.home_acquisition_alignment_activation_distance_m,
            self.home_acquisition_alignment_progress_scale,
            self.home_acquisition_maximum_alignment_reward,
            self.home_acquisition_credit_priority,
        )
        if any(not np.isfinite(value) or value <= 0.0 for value in positives):
            raise ValueError("V43 positive configuration field is invalid")
        if self.her_success_distance_m >= self.contained_distance_m:
            raise ValueError("V43 HER success must be tighter than target containment")
        if type(self.phase_progress_reward_active) is not bool:
            raise ValueError("V509 phase-progress reward flag is invalid")
        if type(self.phase_retention_reward_active) is not bool:
            raise ValueError("V511 phase-retention reward flag is invalid")
        if type(self.home_acquisition_reward_active) is not bool:
            raise ValueError("V598 Home-acquisition reward flag is invalid")
        if type(self.safeguard_projection_penalty_v720) is not bool:
            raise ValueError("V720 safeguard projection reward flag is invalid")
        if not 0.030 <= self.home_acquisition_standoff_m <= 0.100:
            raise ValueError("V598 Home-acquisition standoff is invalid")
        if not 0.045 <= self.home_acquisition_tool_height_m <= 0.100:
            raise ValueError("V598 Home-acquisition tool height is invalid")
        if (
            self.home_acquisition_alignment_activation_distance_m
            <= self.home_acquisition_standoff_m
        ):
            raise ValueError(
                "V598 alignment activation must exceed the standoff"
            )
        her_suppression_distance = (
            self.home_acquisition_suppress_future_her_above_distance_m
        )
        her_suppression_height = (
            self.home_acquisition_suppress_future_her_above_tool_height_m
        )
        her_suppression_alignment = (
            self.home_acquisition_suppress_future_her_below_alignment
        )
        if (
            not np.isfinite(her_suppression_distance)
            or not 0.0 <= her_suppression_distance <= 0.400
            or not np.isfinite(her_suppression_height)
            or not 0.0 <= her_suppression_height <= 0.400
            or not np.isfinite(her_suppression_alignment)
            or not 0.0 <= her_suppression_alignment <= 1.0
        ):
            raise ValueError("V626 acquisition HER suppression is invalid")
        phase_values = np.asarray(
            [
                self.contact_progress_threshold_m,
                self.contact_acquisition_bonus,
                self.stalled_contact_penalty,
                self.target_coverage_gain_bonus,
                self.strict_hold_progress_bonus,
                self.contact_loss_penalty,
                self.target_coverage_loss_penalty,
                self.contact_loss_priority,
                self.target_entry_priority,
                self.target_settle_action_penalty,
                self.target_settle_priority,
                self.coverage_block_half_extent_m,
                self.coverage_target_radius_m,
                self.strict_linear_speed_m_s,
                self.strict_angular_speed_rad_s,
            ],
            dtype=np.float64,
        )
        if not np.all(np.isfinite(phase_values)) or np.any(phase_values < 0.0):
            raise ValueError("V509 phase-progress reward values are invalid")
        penalty_values = np.asarray(
            [
                self.action_penalty,
                self.projection_penalty,
                self.infeasible_action_penalty,
                self.safety_penalty,
                self.actor_infeasibility_coefficient,
                self.critic_conservative_coefficient,
            ],
            dtype=np.float64,
        )
        if (
            not np.all(np.isfinite(penalty_values))
            or np.any(penalty_values < 0.0)
        ):
            raise ValueError("V606 penalty configuration is invalid")
        if self.contact_progress_threshold_m > self.progress_scale_m:
            raise ValueError("V509 contact-progress threshold is too large")
        for name in (
            "target_entry_coverage_threshold",
            "terminal_release_coverage_threshold",
            "strict_coverage_threshold",
            "target_settle_coverage_threshold",
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"V511 {name} is invalid")
        if (
            self.target_entry_coverage_threshold
            >= self.terminal_release_coverage_threshold
        ):
            raise ValueError("V511 coverage thresholds are not ordered")
        for name in (
            "batch_size",
            "hidden_dim",
            "policy_update_period",
            "contact_credit_preceding_steps",
            "coverage_grid_resolution",
            "strict_success_hold_steps",
        ):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"V43 {name} must be a positive integer")


def phase_progress_reward_adjustment_v509(
    neutral_state: np.ndarray,
    next_neutral_state: np.ndarray,
    achieved_goal: np.ndarray,
    next_achieved_goal: np.ndarray,
    desired_goal: np.ndarray,
    valid_contact: np.ndarray,
    relabelled: np.ndarray,
    *,
    config: GoalConditionedHerSACConfigV43,
    strict_target_coverage: np.ndarray | None = None,
    next_strict_target_coverage: np.ndarray | None = None,
    strict_hold_fraction: np.ndarray | None = None,
    next_strict_hold_fraction: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Return contact/coverage/hold reward without prescribing an action."""

    config.validate()
    state = np.asarray(neutral_state, dtype=np.float32)
    next_state = np.asarray(next_neutral_state, dtype=np.float32)
    achieved = np.asarray(achieved_goal, dtype=np.float32)
    next_achieved = np.asarray(next_achieved_goal, dtype=np.float32)
    desired = np.asarray(desired_goal, dtype=np.float32)
    contact = np.asarray(valid_contact, dtype=bool)
    her = np.asarray(relabelled, dtype=bool)
    count = len(state)
    if (
        state.shape != (count, PRIVILEGED_EFFECT_STATE_DIM)
        or next_state.shape != state.shape
        or achieved.shape != (count, GOAL_DIM_V43)
        or next_achieved.shape != achieved.shape
        or desired.shape != achieved.shape
        or contact.shape != (count,)
        or her.shape != (count,)
        or not np.all(np.isfinite(state))
        or not np.all(np.isfinite(next_state))
        or not np.all(np.isfinite(achieved))
        or not np.all(np.isfinite(next_achieved))
        or not np.all(np.isfinite(desired))
    ):
        raise ValueError("V509 phase-progress reward inputs are invalid")
    if (
        not config.phase_progress_reward_active
        and not config.phase_retention_reward_active
    ):
        adjustment = config.contact_bonus * contact.astype(np.float32)
        zeros = np.zeros(count, dtype=bool)
        return adjustment.astype(np.float32), {
            "productive_contact": zeros,
            "stalled_contact": zeros,
            "acquired_contact": zeros,
            "lost_contact": zeros,
            "coverage_gain": np.zeros(count, dtype=np.float32),
            "coverage_loss": np.zeros(count, dtype=np.float32),
            "hold_gain": np.zeros(count, dtype=np.float32),
        }

    before_distance = np.linalg.norm(achieved - desired, axis=-1)
    after_distance = np.linalg.norm(next_achieved - desired, axis=-1)
    progress_m = before_distance - after_distance
    productive_contact = contact & (
        progress_m >= config.contact_progress_threshold_m
    )
    stalled_contact = contact & ~productive_contact
    contact_before = (
        state[:, _SLICES["tool_block_contact_count"]][:, 0] > 0.0
    )
    acquired_contact = contact & ~contact_before
    coverage_before = np.asarray(
        state[:, _SLICES["strict_target_coverage"]][:, 0]
        if strict_target_coverage is None
        else strict_target_coverage,
        dtype=np.float32,
    )
    coverage_after = np.asarray(
        next_state[:, _SLICES["strict_target_coverage"]][:, 0]
        if next_strict_target_coverage is None
        else next_strict_target_coverage,
        dtype=np.float32,
    )
    hold_before = np.asarray(
        state[:, _SLICES["strict_success_streak_over_hold_steps"]][:, 0]
        if strict_hold_fraction is None
        else strict_hold_fraction,
        dtype=np.float32,
    )
    hold_after = np.asarray(
        next_state[:, _SLICES["strict_success_streak_over_hold_steps"]][:, 0]
        if next_strict_hold_fraction is None
        else next_strict_hold_fraction,
        dtype=np.float32,
    )
    for name, values in (
        ("strict_target_coverage", coverage_before),
        ("next_strict_target_coverage", coverage_after),
        ("strict_hold_fraction", hold_before),
        ("next_strict_hold_fraction", hold_after),
    ):
        if (
            values.shape != (count,)
            or not np.all(np.isfinite(values))
            or np.any(values < 0.0)
            or np.any(values > 1.0)
        ):
            raise ValueError(f"V512 {name} is invalid")
    original_goal = ~her
    coverage_gain = (
        np.maximum(coverage_after - coverage_before, 0.0) * original_goal
    ).astype(np.float32)
    hold_gain = (
        np.maximum(hold_after - hold_before, 0.0) * original_goal
    ).astype(np.float32)
    adjustment = np.zeros(count, dtype=np.float32)
    if config.phase_progress_reward_active:
        adjustment += config.contact_bonus * productive_contact
        adjustment += config.contact_acquisition_bonus * acquired_contact
        adjustment -= config.stalled_contact_penalty * stalled_contact
        adjustment += config.target_coverage_gain_bonus * coverage_gain
        adjustment += config.strict_hold_progress_bonus * hold_gain
    else:
        adjustment += config.contact_bonus * contact.astype(np.float32)

    release_allowed = (
        np.maximum(coverage_before, coverage_after)
        >= config.terminal_release_coverage_threshold
    ) | (np.maximum(hold_before, hold_after) > 0.0)
    lost_contact = (
        contact_before
        & ~contact
        & original_goal
        & ~release_allowed
    )
    coverage_loss = (
        np.maximum(coverage_before - coverage_after, 0.0) * original_goal
    ).astype(np.float32)
    if config.phase_retention_reward_active:
        adjustment -= config.contact_loss_penalty * lost_contact
        adjustment -= config.target_coverage_loss_penalty * coverage_loss
    return adjustment, {
        "productive_contact": productive_contact,
        "stalled_contact": stalled_contact,
        "acquired_contact": acquired_contact,
        "lost_contact": lost_contact,
        "coverage_gain": coverage_gain,
        "coverage_loss": coverage_loss,
        "hold_gain": hold_gain,
    }


def home_acquisition_geometry_v598(
    neutral_state: np.ndarray,
    desired_goal: np.ndarray,
    *,
    config: GoalConditionedHerSACConfigV43,
) -> tuple[np.ndarray, np.ndarray]:
    """Return tool-to-precontact distance and broad-face alignment.

    The pre-contact point is a task-space auxiliary goal, not a commanded
    waypoint: the actor remains free to choose every XYZ action.  The point is
    recomputed from the current block and desired object goal, so it remains
    valid across randomized tasks and supplies no stored expert trajectory.
    """

    config.validate()
    state = np.asarray(neutral_state, dtype=np.float32)
    goal = np.asarray(desired_goal, dtype=np.float32)
    count = len(state)
    if (
        state.shape != (count, PRIVILEGED_EFFECT_STATE_DIM)
        or goal.shape != (count, GOAL_DIM_V43)
        or not np.all(np.isfinite(state))
        or not np.all(np.isfinite(goal))
    ):
        raise ValueError("V598 Home-acquisition geometry inputs are invalid")
    block_pose = state[:, _SLICES["block_pose_xyz_quaternion_wxyz"]]
    tool_pose = state[:, _SLICES["tool_pose_position_rotation"]]
    block_xy = block_pose[:, :2]
    direction = goal - block_xy
    direction_norm = np.linalg.norm(direction, axis=-1, keepdims=True)
    fallback = np.zeros_like(direction)
    fallback[:, 0] = 1.0
    forward = np.where(
        direction_norm > 1.0e-7,
        direction / np.maximum(direction_norm, 1.0e-7),
        fallback,
    )
    precontact = np.concatenate(
        (
            block_xy - config.home_acquisition_standoff_m * forward,
            np.full(
                (count, 1),
                config.home_acquisition_tool_height_m,
                dtype=np.float32,
            ),
        ),
        axis=-1,
    )
    distance = np.linalg.norm(tool_pose[:, :3] - precontact, axis=-1)
    rotation = tool_pose[:, 3:12].reshape(count, 3, 3)
    broad_face_normal_xy = rotation[:, :, 1][:, :2]
    horizontal_norm = np.linalg.norm(
        broad_face_normal_xy,
        axis=-1,
    )
    alignment = np.abs(np.sum(broad_face_normal_xy * forward, axis=-1))
    alignment /= np.maximum(horizontal_norm, 1.0e-7)
    return (
        distance.astype(np.float32),
        np.clip(alignment, 0.0, 1.0).astype(np.float32),
    )


def home_acquisition_reward_adjustment_v598(
    neutral_state: np.ndarray,
    next_neutral_state: np.ndarray,
    desired_goal: np.ndarray,
    valid_contact: np.ndarray,
    relabelled: np.ndarray,
    *,
    config: GoalConditionedHerSACConfigV43,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Reward original-goal approach progress before first valid contact."""

    state = np.asarray(neutral_state, dtype=np.float32)
    next_state = np.asarray(next_neutral_state, dtype=np.float32)
    goal = np.asarray(desired_goal, dtype=np.float32)
    contact = np.asarray(valid_contact, dtype=bool)
    her = np.asarray(relabelled, dtype=bool)
    count = len(state)
    if (
        next_state.shape != state.shape
        or contact.shape != (count,)
        or her.shape != (count,)
    ):
        raise ValueError("V598 Home-acquisition reward inputs are invalid")
    zeros = np.zeros(count, dtype=np.float32)
    false = np.zeros(count, dtype=bool)
    if not config.home_acquisition_reward_active:
        return zeros, {
            "approach_transition": false,
            "distance_before_m": zeros,
            "distance_after_m": zeros,
            "distance_progress_m": zeros,
            "alignment_before": zeros,
            "alignment_after": zeros,
            "alignment_progress": zeros,
            "alignment_gate": zeros,
            "her_suppressed": her.copy(),
        }

    distance_before, alignment_before = home_acquisition_geometry_v598(
        state,
        goal,
        config=config,
    )
    distance_after, alignment_after = home_acquisition_geometry_v598(
        next_state,
        goal,
        config=config,
    )
    contact_before = (
        state[:, _SLICES["tool_block_contact_count"]][:, 0] > 0.0
    )
    approach_transition = ~her & ~contact_before & ~contact
    distance_progress = distance_before - distance_after
    distance_reward = np.clip(
        distance_progress / config.home_acquisition_progress_scale_m,
        -1.0,
        1.0,
    ) * config.home_acquisition_maximum_progress_reward
    alignment_progress = alignment_after - alignment_before
    alignment_gate = np.clip(
        1.0
        - np.minimum(distance_before, distance_after)
        / config.home_acquisition_alignment_activation_distance_m,
        0.0,
        1.0,
    )
    alignment_reward = np.clip(
        alignment_progress
        / config.home_acquisition_alignment_progress_scale,
        -1.0,
        1.0,
    ) * config.home_acquisition_maximum_alignment_reward
    adjustment = approach_transition.astype(np.float32) * (
        distance_reward + alignment_gate * alignment_reward
    )
    return adjustment.astype(np.float32), {
        "approach_transition": approach_transition,
        "distance_before_m": distance_before,
        "distance_after_m": distance_after,
        "distance_progress_m": distance_progress.astype(np.float32),
        "alignment_before": alignment_before,
        "alignment_after": alignment_after,
        "alignment_progress": alignment_progress.astype(np.float32),
        "alignment_gate": alignment_gate.astype(np.float32),
        "her_suppressed": her.copy(),
    }


_REPLAY_FIELDS = (
    "neutral_state",
    "next_neutral_state",
    "achieved_goal",
    "next_achieved_goal",
    "desired_goal",
    "action",
    "applied_action",
    "terminal",
    "failure_terminal",
    "strict_success",
    "action_feasible",
    "safety_violation",
    "valid_contact",
    "invalid_contact",
    "step_block_displacement_m",
    "projection_l2",
    "strict_target_coverage",
    "next_strict_target_coverage",
    "strict_hold_fraction",
    "next_strict_hold_fraction",
    "episode_index",
    "episode_step",
)

_TASK_METRIC_REPLAY_FIELDS_V512 = (
    "strict_target_coverage",
    "next_strict_target_coverage",
    "strict_hold_fraction",
    "next_strict_hold_fraction",
)


def _empty_replay_arrays() -> dict[str, np.ndarray]:
    return {
        "neutral_state": np.empty((0, PRIVILEGED_EFFECT_STATE_DIM), np.float32),
        "next_neutral_state": np.empty((0, PRIVILEGED_EFFECT_STATE_DIM), np.float32),
        "achieved_goal": np.empty((0, GOAL_DIM_V43), np.float32),
        "next_achieved_goal": np.empty((0, GOAL_DIM_V43), np.float32),
        "desired_goal": np.empty((0, GOAL_DIM_V43), np.float32),
        "action": np.empty((0, ACTION_DIM_V43), np.float32),
        "applied_action": np.empty((0, ACTION_DIM_V43), np.float32),
        "terminal": np.empty((0,), bool),
        "failure_terminal": np.empty((0,), bool),
        "strict_success": np.empty((0,), bool),
        "action_feasible": np.empty((0,), bool),
        "safety_violation": np.empty((0,), bool),
        "valid_contact": np.empty((0,), bool),
        "invalid_contact": np.empty((0,), bool),
        "step_block_displacement_m": np.empty((0,), np.float32),
        "projection_l2": np.empty((0,), np.float32),
        "strict_target_coverage": np.empty((0,), np.float32),
        "next_strict_target_coverage": np.empty((0,), np.float32),
        "strict_hold_fraction": np.empty((0,), np.float32),
        "next_strict_hold_fraction": np.empty((0,), np.float32),
        "episode_index": np.empty((0,), np.int64),
        "episode_step": np.empty((0,), np.int64),
}


def _coverage_from_neutral_state_goal_v512(
    neutral_state: np.ndarray,
    desired_goal: np.ndarray,
    config: GoalConditionedHerSACConfigV43,
) -> np.ndarray:
    """Reconstruct the simulator's 17x17 footprint coverage exactly."""

    state = np.asarray(neutral_state, dtype=np.float32)
    goal = np.asarray(desired_goal, dtype=np.float32)
    count = len(state)
    if (
        state.shape != (count, PRIVILEGED_EFFECT_STATE_DIM)
        or goal.shape != (count, GOAL_DIM_V43)
    ):
        raise ValueError("V512 coverage reconstruction inputs are invalid")
    pose = state[:, _SLICES["block_pose_xyz_quaternion_wxyz"]]
    quaternion = pose[:, 3:7].astype(np.float64)
    quaternion /= np.maximum(
        np.linalg.norm(quaternion, axis=-1, keepdims=True), 1.0e-12
    )
    w, x, y, z = quaternion.T
    yaw = np.arctan2(
        2.0 * (w * z + x * y),
        1.0 - 2.0 * (y * y + z * z),
    )
    grid = np.linspace(
        -config.coverage_block_half_extent_m,
        config.coverage_block_half_extent_m,
        config.coverage_grid_resolution,
        dtype=np.float64,
    )
    xx, yy = np.meshgrid(grid, grid, indexing="xy")
    local = np.stack((xx.ravel(), yy.ravel()), axis=-1)
    result = np.empty(count, dtype=np.float32)
    for start in range(0, count, 4_096):
        stop = min(start + 4_096, count)
        cosine = np.cos(yaw[start:stop])[:, None]
        sine = np.sin(yaw[start:stop])[:, None]
        world_x = (
            pose[start:stop, 0:1]
            + cosine * local[None, :, 0]
            - sine * local[None, :, 1]
        )
        world_y = (
            pose[start:stop, 1:2]
            + sine * local[None, :, 0]
            + cosine * local[None, :, 1]
        )
        squared_distance = np.square(world_x - goal[start:stop, 0:1])
        squared_distance += np.square(world_y - goal[start:stop, 1:2])
        result[start:stop] = np.mean(
            squared_distance <= np.square(config.coverage_target_radius_m),
            axis=1,
        ).astype(np.float32)
    return result


def _reconstruct_task_metrics_v512(
    arrays: dict[str, np.ndarray],
    config: GoalConditionedHerSACConfigV43,
) -> dict[str, np.ndarray]:
    """Upgrade legacy neutralized replay with reward-only task metrics."""

    coverage = _coverage_from_neutral_state_goal_v512(
        arrays["neutral_state"], arrays["desired_goal"], config
    )
    next_coverage = _coverage_from_neutral_state_goal_v512(
        arrays["next_neutral_state"], arrays["desired_goal"], config
    )
    speed_slice = _SLICES["block_linear_angular_speed"]
    next_speed = arrays["next_neutral_state"][:, speed_slice]
    hold = np.zeros(len(coverage), dtype=np.float32)
    next_hold = np.zeros(len(coverage), dtype=np.float32)
    for episode_index in np.unique(arrays["episode_index"]):
        rows = np.flatnonzero(arrays["episode_index"] == episode_index)
        streak = 0
        for row in rows:
            hold[row] = np.float32(
                min(streak / config.strict_success_hold_steps, 1.0)
            )
            contained = (
                next_coverage[row] >= config.strict_coverage_threshold
            )
            settled = bool(
                next_speed[row, 0] <= config.strict_linear_speed_m_s
                and next_speed[row, 1] <= config.strict_angular_speed_rad_s
            )
            streak = streak + 1 if contained and settled else 0
            next_hold[row] = np.float32(
                min(streak / config.strict_success_hold_steps, 1.0)
            )
    return {
        "strict_target_coverage": coverage,
        "next_strict_target_coverage": next_coverage,
        "strict_hold_fraction": hold,
        "next_strict_hold_fraction": next_hold,
    }


class GoalConditionedHerReplayV43:
    """Persistent replay over complete episodes with contact-prioritized HER."""

    def __init__(self, config: GoalConditionedHerSACConfigV43 | None = None) -> None:
        self.config = config or GoalConditionedHerSACConfigV43()
        self.config.validate()
        self.arrays = _empty_replay_arrays()
        self.source_rows: list[dict[str, Any]] = []
        self._next_episode_index = 0

    @property
    def transition_count(self) -> int:
        return int(len(self.arrays["terminal"]))

    @property
    def episode_count(self) -> int:
        return int(self._next_episode_index)

    def add_episode(self, episode: dict[str, np.ndarray], *, source: str) -> None:
        missing = (
            set(_REPLAY_FIELDS)
            - {"episode_index", *_TASK_METRIC_REPLAY_FIELDS_V512}
            - set(episode)
        )
        if missing:
            raise ValueError(f"V43 replay episode is missing fields: {sorted(missing)}")
        supplied_task_metrics = {
            name for name in _TASK_METRIC_REPLAY_FIELDS_V512 if name in episode
        }
        if supplied_task_metrics and supplied_task_metrics != set(
            _TASK_METRIC_REPLAY_FIELDS_V512
        ):
            raise ValueError("V512 replay task metrics must be supplied together")
        count = int(len(np.asarray(episode["terminal"])))
        if count < 1:
            raise ValueError("V43 replay cannot add an empty episode")
        normalized: dict[str, np.ndarray] = {}
        expected = _empty_replay_arrays()
        for name in _REPLAY_FIELDS:
            if name == "episode_index":
                value = np.full(count, self._next_episode_index, dtype=np.int64)
            elif (
                name in _TASK_METRIC_REPLAY_FIELDS_V512
                and name not in episode
            ):
                continue
            else:
                value = np.asarray(episode[name], dtype=expected[name].dtype)
            if value.shape != (count, *expected[name].shape[1:]):
                raise ValueError(f"V43 replay field {name} has shape {value.shape}")
            if np.issubdtype(value.dtype, np.floating) and not np.all(np.isfinite(value)):
                raise ValueError(f"V43 replay field {name} contains non-finite values")
            normalized[name] = value
        if not supplied_task_metrics:
            normalized.update(
                _reconstruct_task_metrics_v512(normalized, self.config)
            )
        if not np.array_equal(normalized["episode_step"], np.arange(count, dtype=np.int64)):
            raise ValueError("V43 replay episode steps must be contiguous from zero")
        if bool(np.any(normalized["strict_success"] & ~normalized["terminal"])):
            raise ValueError("V43 strict success must be a terminal transition")
        start = self.transition_count
        for name in _REPLAY_FIELDS:
            self.arrays[name] = np.concatenate((self.arrays[name], normalized[name]), axis=0)
        self.source_rows.append(
            {
                "source": str(source),
                "episode_index": self._next_episode_index,
                "start_row": start,
                "row_count": count,
                "strict_success": bool(np.any(normalized["strict_success"])),
                "failure_terminal": bool(np.any(normalized["failure_terminal"])),
            }
        )
        self._next_episode_index += 1

    @classmethod
    def from_h5(
        cls,
        paths: Iterable[Path],
        config: GoalConditionedHerSACConfigV43 | None = None,
    ) -> "GoalConditionedHerReplayV43":
        result = cls(config)
        for source_path in paths:
            path = Path(source_path).expanduser().resolve()
            if not path.is_file():
                raise FileNotFoundError(f"V43 seed replay is missing: {path}")
            with h5py.File(path, "r") as stream:
                privileged = np.asarray(
                    stream["training_only/privileged_state"][:], dtype=np.float32
                )
                next_privileged = np.asarray(
                    stream["training_only/next_privileged_state"][:], dtype=np.float32
                )
                count = int(len(privileged))
                episode_ids = np.asarray(stream["execution/episode_ids"][:], dtype=np.int64)
                episode_steps = np.asarray(
                    stream["execution/episode_step_ids"][:], dtype=np.int64
                )
                action = np.asarray(stream["execution/policy_action"][:], dtype=np.float32)
                applied = np.asarray(
                    stream["execution/applied_task_action"][:], dtype=np.float32
                )
                attempted = np.asarray(
                    stream["execution/execution_attempted"][:], dtype=bool
                )
                guard_safe = np.asarray(
                    stream["execution/guard_safe_candidate"][:], dtype=bool
                )
                ik = np.asarray(stream["execution/ik_converged"][:], dtype=bool)
                scale = np.asarray(
                    stream["execution/ik_application_scale"][:], dtype=np.float32
                )
                terminal = np.asarray(
                    stream["reward_and_outcome/terminated"][:], dtype=bool
                ) | np.asarray(stream["reward_and_outcome/truncated"][:], dtype=bool)
                failure = np.asarray(
                    stream["reward_and_outcome/terminal_failure"][:], dtype=bool
                )
                safety = np.asarray(
                    stream["reward_and_outcome/safety_stop"][:], dtype=bool
                )
                strict = np.asarray(
                    stream["reward_and_outcome/strict_success"][:], dtype=bool
                )
                contact = np.asarray(
                    stream["execution/valid_push_side_contact_any"][:], dtype=bool
                )
                invalid = np.asarray(
                    stream["execution/invalid_tool_block_contact_any"][:], dtype=bool
                )
                displacement = np.asarray(
                    stream["execution/step_block_displacement_m"][:], dtype=np.float32
                )
            shapes = (
                privileged.shape == (count, PRIVILEGED_EFFECT_STATE_DIM),
                next_privileged.shape == (count, PRIVILEGED_EFFECT_STATE_DIM),
                episode_ids.shape == (count,),
                episode_steps.shape == (count,),
                action.shape == (count, ACTION_DIM_V43),
                applied.shape == (count, ACTION_DIM_V43),
            )
            if not all(shapes):
                raise ValueError("V43 seed replay H5 shapes changed")
            neutral = goal_neutral_privileged_state_v43(privileged)
            next_neutral = goal_neutral_privileged_state_v43(next_privileged)
            achieved = achieved_goal_from_privileged_v43(privileged)
            next_achieved = achieved_goal_from_privileged_v43(next_privileged)
            desired = desired_goal_from_privileged_v43(privileged)
            projection = np.linalg.norm(action - applied, axis=-1).astype(np.float32)
            feasible = attempted & guard_safe & ik & (scale > np.float32(0.0))
            for episode_id in np.unique(episode_ids):
                selected = np.flatnonzero(episode_ids == episode_id)
                if not np.array_equal(
                    episode_steps[selected], np.arange(len(selected), dtype=np.int64)
                ):
                    raise ValueError("V43 seed replay episode is not contiguous")
                result.add_episode(
                    {
                        "neutral_state": neutral[selected],
                        "next_neutral_state": next_neutral[selected],
                        "achieved_goal": achieved[selected],
                        "next_achieved_goal": next_achieved[selected],
                        "desired_goal": desired[selected],
                        "action": action[selected],
                        "applied_action": applied[selected],
                        "terminal": terminal[selected],
                        "failure_terminal": failure[selected],
                        "strict_success": strict[selected],
                        "action_feasible": feasible[selected],
                        "safety_violation": safety[selected] | invalid[selected],
                        "valid_contact": contact[selected],
                        "invalid_contact": invalid[selected],
                        "step_block_displacement_m": displacement[selected],
                        "projection_l2": projection[selected],
                        "strict_target_coverage": privileged[
                            selected, _SLICES["strict_target_coverage"]
                        ][:, 0],
                        "next_strict_target_coverage": next_privileged[
                            selected, _SLICES["strict_target_coverage"]
                        ][:, 0],
                        "strict_hold_fraction": privileged[
                            selected,
                            _SLICES[
                                "strict_success_streak_over_hold_steps"
                            ],
                        ][:, 0],
                        "next_strict_hold_fraction": next_privileged[
                            selected,
                            _SLICES[
                                "strict_success_streak_over_hold_steps"
                            ],
                        ][:, 0],
                        "episode_step": episode_steps[selected],
                    },
                    source=str(path),
                )
        if result.transition_count < 1:
            raise ValueError("V43 replay received no transitions")
        return result

    def _episode_end_rows(self) -> np.ndarray:
        episodes = self.arrays["episode_index"]
        result = np.empty(self.transition_count, dtype=np.int64)
        for episode_index in np.unique(episodes):
            rows = np.flatnonzero(episodes == episode_index)
            result[rows] = rows[-1]
        return result

    def _sampling_weights(self) -> np.ndarray:
        arrays = self.arrays
        profiles = self._contact_learning_profiles()
        weights = np.ones(self.transition_count, dtype=np.float64)
        weights += 4.0 * arrays["valid_contact"].astype(np.float64)
        weights += 4.0 * (
            arrays["step_block_displacement_m"] >= np.float32(5.0e-5)
        ).astype(np.float64)
        weights += 8.0 * arrays["strict_success"].astype(np.float64)
        weights += 2.0 * (~arrays["action_feasible"]).astype(np.float64)
        weights += 2.0 * arrays["failure_terminal"].astype(np.float64)
        weights += self.config.contact_credit_priority * profiles[
            "contact_credit_source"
        ].astype(np.float64)
        weights += self.config.successful_contact_credit_priority * (
            profiles["contact_credit_source"]
            & profiles["strict_success_episode_source"]
        ).astype(np.float64)
        if self.config.phase_retention_reward_active:
            slices = privileged_effect_state_slices_v1()
            contact_before = (
                arrays["neutral_state"][
                    :, slices["tool_block_contact_count"]
                ][:, 0]
                > 0.0
            )
            coverage_before = arrays["strict_target_coverage"]
            coverage_after = arrays["next_strict_target_coverage"]
            release_allowed = (
                np.maximum(coverage_before, coverage_after)
                >= self.config.terminal_release_coverage_threshold
            )
            contact_loss = (
                contact_before
                & ~arrays["valid_contact"]
                & ~release_allowed
            )
            target_entry = (
                np.maximum(coverage_before, coverage_after)
                >= self.config.target_entry_coverage_threshold
            ) & ~arrays["strict_success"]
            weights += self.config.contact_loss_priority * contact_loss.astype(
                np.float64
            )
            weights += self.config.target_entry_priority * target_entry.astype(
                np.float64
            )
            target_settle = (
                np.maximum(coverage_before, coverage_after)
                >= self.config.target_settle_coverage_threshold
            ) & ~arrays["strict_success"]
            weights += self.config.target_settle_priority * (
                target_settle.astype(np.float64)
            )
        if self.config.home_acquisition_reward_active:
            home_profiles = self._home_acquisition_learning_profiles()
            weights += self.config.home_acquisition_credit_priority * (
                home_profiles["effectful_approach_source"].astype(
                    np.float64
                )
            )
        return weights / float(np.sum(weights))

    def _home_acquisition_future_her_suppression_mask_v626(
        self,
    ) -> np.ndarray:
        """Keep far acquisition transitions on their original task goal."""

        count = self.transition_count
        distance_threshold = float(
            self.config
            .home_acquisition_suppress_future_her_above_distance_m
        )
        height_threshold = float(
            self.config
            .home_acquisition_suppress_future_her_above_tool_height_m
        )
        alignment_threshold = float(
            self.config
            .home_acquisition_suppress_future_her_below_alignment
        )
        if not self.config.home_acquisition_reward_active or not any(
            threshold > 0.0
            for threshold in (
                distance_threshold,
                height_threshold,
                alignment_threshold,
            )
        ):
            return np.zeros(count, dtype=bool)
        arrays = self.arrays
        distance, alignment = home_acquisition_geometry_v598(
            arrays["neutral_state"],
            arrays["desired_goal"],
            config=self.config,
        )
        tool_height = arrays["neutral_state"][
            :, _SLICES["tool_pose_position_rotation"]
        ][:, 2]
        contact_before = (
            arrays["neutral_state"][
                :, _SLICES["tool_block_contact_count"]
            ][:, 0]
            > 0.0
        )
        acquisition_phase = np.zeros(count, dtype=bool)
        if distance_threshold > 0.0:
            acquisition_phase |= distance > distance_threshold
        if height_threshold > 0.0:
            acquisition_phase |= tool_height > height_threshold
        if alignment_threshold > 0.0:
            acquisition_phase |= alignment < alignment_threshold
        return (~contact_before & acquisition_phase).astype(bool)

    def _home_acquisition_learning_profiles(
        self,
    ) -> dict[str, np.ndarray]:
        """Identify safe original-goal transitions that improved approach."""

        count = self.transition_count
        if not self.config.home_acquisition_reward_active:
            return {
                "effectful_approach_source": np.zeros(count, dtype=bool),
                "precontact_distance_progress_m": np.zeros(
                    count, dtype=np.float32
                ),
            }
        arrays = self.arrays
        before, _alignment_before = home_acquisition_geometry_v598(
            arrays["neutral_state"],
            arrays["desired_goal"],
            config=self.config,
        )
        after, _alignment_after = home_acquisition_geometry_v598(
            arrays["next_neutral_state"],
            arrays["desired_goal"],
            config=self.config,
        )
        progress = (before - after).astype(np.float32)
        contact_before = (
            arrays["neutral_state"][
                :, _SLICES["tool_block_contact_count"]
            ][:, 0]
            > 0.0
        )
        effectful = (
            ~contact_before
            & ~arrays["valid_contact"]
            & arrays["action_feasible"]
            & ~arrays["safety_violation"]
            & (
                progress
                >= np.float32(
                    0.05 * self.config.home_acquisition_progress_scale_m
                )
            )
        )
        return {
            "effectful_approach_source": effectful,
            "precontact_distance_progress_m": progress,
        }

    def _contact_learning_profiles(self) -> dict[str, np.ndarray]:
        """Build causal contact windows and effectful self-imitation labels.

        Immediate contact-only prioritization misses the approach actions that
        made contact possible.  Conversely, marking every feasible transition
        as desirable anchors the actor to long no-effect trajectories.  This
        profile keeps both identities explicit and episode bounded.
        """

        arrays = self.arrays
        count = self.transition_count
        contact_credit = np.zeros(count, dtype=bool)
        contact_episode = np.zeros(count, dtype=bool)
        strict_success_episode = np.zeros(count, dtype=bool)
        motion = arrays["step_block_displacement_m"] >= np.float32(5.0e-5)
        contact_or_motion = arrays["valid_contact"] | motion
        before_distance = np.linalg.norm(
            arrays["achieved_goal"] - arrays["desired_goal"], axis=-1
        )
        after_distance = np.linalg.norm(
            arrays["next_achieved_goal"] - arrays["desired_goal"], axis=-1
        )
        original_progress_m = before_distance - after_distance
        effectful_behavior = (
            contact_or_motion
            & arrays["action_feasible"]
            & ~arrays["safety_violation"]
            & (
                original_progress_m
                >= self.config.effectful_progress_minimum_m
            )
        )
        preceding = int(self.config.contact_credit_preceding_steps)
        episodes = arrays["episode_index"]
        for episode_index in np.unique(episodes):
            rows = np.flatnonzero(episodes == episode_index)
            if not len(rows):  # pragma: no cover - replay invariant
                continue
            strict_success_episode[rows] = bool(
                np.any(arrays["strict_success"][rows])
            )
            local_signals = np.flatnonzero(contact_or_motion[rows])
            if not len(local_signals):
                continue
            contact_episode[rows] = True
            for local_index in local_signals:
                first = max(0, int(local_index) - preceding)
                contact_credit[rows[first] : rows[int(local_index)] + 1] = True
        return {
            "contact_credit_source": contact_credit,
            "contact_episode_source": contact_episode,
            "strict_success_episode_source": strict_success_episode,
            "effectful_behavior_source": effectful_behavior,
            "original_task_progress_m": original_progress_m.astype(
                np.float32
            ),
        }

    def _episode_initial_distance_by_transition(self) -> np.ndarray:
        """Return the original task distance attached to each episode row.

        HER changes the sampled goal later, so curriculum membership must be
        computed from the first transition's original achieved/desired goal
        and then copied across that complete episode.
        """

        arrays = self.arrays
        result = np.empty(self.transition_count, dtype=np.float32)
        episodes = arrays["episode_index"]
        for episode_index in np.unique(episodes):
            rows = np.flatnonzero(episodes == episode_index)
            if not len(rows):  # pragma: no cover - replay invariant
                continue
            initial_distance = np.linalg.norm(
                arrays["achieved_goal"][rows[0]]
                - arrays["desired_goal"][rows[0]]
            )
            result[rows] = np.float32(initial_distance)
        return result

    def _sample_root_rows(
        self,
        *,
        batch_size: int,
        rng: np.random.Generator,
        frontier_initial_distance_range_m: tuple[float, float] | None,
        frontier_sampling_fraction: float | None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        probability = self._sampling_weights()
        positive_probability = probability > 0.0
        positive_probability_count = int(
            np.count_nonzero(positive_probability)
        )
        if positive_probability_count < 1:
            raise RuntimeError("V43 replay has no positive-probability rows")
        initial_distance = self._episode_initial_distance_by_transition()
        if frontier_initial_distance_range_m is None:
            roots = rng.choice(
                self.transition_count,
                size=batch_size,
                replace=positive_probability_count < batch_size,
                p=probability,
            ).astype(np.int64)
            return roots, probability[roots], np.zeros(batch_size, dtype=bool)

        if frontier_sampling_fraction is None:
            raise ValueError(
                "V43 frontier replay requires a sampling fraction"
            )
        lower, upper = (
            float(frontier_initial_distance_range_m[0]),
            float(frontier_initial_distance_range_m[1]),
        )
        if (
            not np.isfinite(lower)
            or not np.isfinite(upper)
            or lower < 0.0
            or upper <= lower
        ):
            raise ValueError("V43 frontier distance range is invalid")
        if (
            not np.isfinite(frontier_sampling_fraction)
            or not 0.0 < frontier_sampling_fraction <= 1.0
        ):
            raise ValueError("V43 frontier sampling fraction is invalid")

        tolerance = np.float32(1.0e-6)
        frontier_mask = (
            initial_distance >= np.float32(lower) - tolerance
        ) & (
            initial_distance <= np.float32(upper) + tolerance
        ) & positive_probability
        frontier_rows = np.flatnonzero(frontier_mask)
        history_rows = np.flatnonzero(~frontier_mask & positive_probability)
        if not len(frontier_rows):
            raise RuntimeError(
                "V43 curriculum-stratified replay has no current-frontier "
                "episode"
            )

        def episode_balanced_probability(rows: np.ndarray) -> np.ndarray:
            """Give every episode equal mass, then prioritize within it."""

            result = np.zeros(self.transition_count, dtype=np.float64)
            row_episodes = self.arrays["episode_index"][rows]
            unique_episodes = np.unique(row_episodes)
            episode_mass = 1.0 / len(unique_episodes)
            for episode_index in unique_episodes:
                episode_rows = rows[row_episodes == episode_index]
                local_probability = probability[episode_rows]
                result[episode_rows] = (
                    episode_mass
                    * local_probability
                    / float(np.sum(local_probability))
                )
            return result

        frontier_count = int(
            np.clip(
                round(batch_size * frontier_sampling_fraction),
                1,
                batch_size,
            )
        )
        if not len(history_rows):
            frontier_count = batch_size
        history_count = batch_size - frontier_count
        frontier_probability = episode_balanced_probability(frontier_rows)
        frontier_roots = rng.choice(
            frontier_rows,
            size=frontier_count,
            replace=len(frontier_rows) < frontier_count,
            p=frontier_probability[frontier_rows],
        ).astype(np.int64)

        roots = frontier_roots
        sampled_probability = (
            (frontier_count / batch_size)
            * frontier_probability[frontier_roots]
        )
        frontier_source = np.ones(frontier_count, dtype=bool)
        if history_count:
            history_probability = episode_balanced_probability(history_rows)
            history_roots = rng.choice(
                history_rows,
                size=history_count,
                replace=len(history_rows) < history_count,
                p=history_probability[history_rows],
            ).astype(np.int64)
            roots = np.concatenate((roots, history_roots))
            sampled_probability = np.concatenate(
                (
                    sampled_probability,
                    (history_count / batch_size)
                    * history_probability[history_roots],
                )
            )
            frontier_source = np.concatenate(
                (frontier_source, np.zeros(history_count, dtype=bool))
            )
        permutation = rng.permutation(batch_size)
        return (
            roots[permutation],
            sampled_probability[permutation],
            frontier_source[permutation],
        )

    def sample(
        self,
        *,
        batch_size: int,
        seed: int,
        frontier_initial_distance_range_m: tuple[float, float] | None = None,
        frontier_sampling_fraction: float | None = None,
    ) -> dict[str, np.ndarray]:
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError("V43 replay batch size must be positive")
        if type(seed) is not int or seed < 0:
            raise ValueError("V43 replay seed must be non-negative")
        if self.transition_count < 1:
            raise RuntimeError("V43 replay is empty")
        rng = np.random.default_rng(seed)
        learning_profiles = self._contact_learning_profiles()
        roots, sampled_probability, frontier_source = self._sample_root_rows(
            batch_size=batch_size,
            rng=rng,
            frontier_initial_distance_range_m=(
                frontier_initial_distance_range_m
            ),
            frontier_sampling_fraction=frontier_sampling_fraction,
        )
        arrays = self.arrays
        episode_initial_distance = self._episode_initial_distance_by_transition()
        desired = arrays["desired_goal"][roots].copy()
        action = arrays["action"][roots].copy()
        applied_action = arrays["applied_action"][roots].copy()
        safeguard_projected_action_v720 = applied_action.copy()
        safeguard_projection_valid_v720 = np.zeros(batch_size, dtype=bool)
        if self.config.safeguard_projection_penalty_v720:
            projected_source = getattr(
                self,
                "safeguard_projected_action_v720",
                None,
            )
            projection_valid_source = getattr(
                self,
                "safeguard_projection_valid_v720",
                None,
            )
            if (
                not isinstance(projected_source, np.ndarray)
                or projected_source.shape != (self.transition_count, 3)
                or not np.all(np.isfinite(projected_source))
                or not isinstance(projection_valid_source, np.ndarray)
                or projection_valid_source.shape != (self.transition_count,)
                or projection_valid_source.dtype != np.dtype(bool)
            ):
                raise RuntimeError(
                    "V720 replay lacks valid pre-plant safeguard projections"
                )
            safeguard_projected_action_v720 = projected_source[roots].copy()
            safeguard_projection_valid_v720 = (
                projection_valid_source[roots].copy()
            )
        relabelled = np.zeros(batch_size, dtype=bool)
        acquisition_her_suppressed = (
            self._home_acquisition_future_her_suppression_mask_v626()
        )
        episode_end = self._episode_end_rows()
        for batch_index, row in enumerate(roots):
            if acquisition_her_suppressed[row]:
                continue
            if rng.random() >= self.config.future_relabel_probability:
                continue
            final = int(episode_end[row])
            future_row = int(rng.integers(row, final + 1))
            future_goal = arrays["next_achieved_goal"][future_row]
            if (
                np.linalg.norm(future_goal - arrays["achieved_goal"][row])
                < self.config.minimum_future_displacement_m
            ):
                continue
            desired[batch_index] = future_goal
            action[batch_index] = reframe_task_action_for_goal_v43(
                arrays["action"][row : row + 1],
                arrays["achieved_goal"][row : row + 1],
                arrays["desired_goal"][row : row + 1],
                future_goal[None],
            )[0]
            applied_action[batch_index] = reframe_task_action_for_goal_v43(
                arrays["applied_action"][row : row + 1],
                arrays["achieved_goal"][row : row + 1],
                arrays["desired_goal"][row : row + 1],
                future_goal[None],
            )[0]
            if self.config.safeguard_projection_penalty_v720:
                safeguard_projected_action_v720[batch_index] = (
                    reframe_task_action_for_goal_v43(
                        projected_source[row : row + 1],
                        arrays["achieved_goal"][row : row + 1],
                        arrays["desired_goal"][row : row + 1],
                        future_goal[None],
                    )[0]
                )
            relabelled[batch_index] = True

        # Keep the delayed live effect for causal diagnostics.  When V720 is
        # enabled, only the pre-plant safe action receives the SE-RL
        # action-aliasing penalty.  Both action vectors are rotated under HER.
        actual_effect_projection_l2_v720 = np.linalg.norm(
            action - applied_action, axis=-1
        ).astype(np.float32)
        if self.config.safeguard_projection_penalty_v720:
            (
                projection_penalty_component_v720,
                safeguard_projection_l2_v720,
            ) = squared_safeguard_intervention_penalty_v720(
                action,
                safeguard_projected_action_v720,
                safeguard_projection_valid_v720,
                coefficient=self.config.projection_penalty,
            )
            projection_l2 = np.where(
                safeguard_projection_valid_v720,
                safeguard_projection_l2_v720,
                np.float32(0.0),
            ).astype(np.float32)
        else:
            projection_l2 = actual_effect_projection_l2_v720
            projection_penalty_component_v720 = (
                np.float32(self.config.projection_penalty)
                * np.square(projection_l2, dtype=np.float32)
            )

        achieved = arrays["achieved_goal"][roots]
        next_achieved = arrays["next_achieved_goal"][roots]
        before_distance = np.linalg.norm(achieved - desired, axis=-1)
        after_distance = np.linalg.norm(next_achieved - desired, axis=-1)
        progress = np.clip(
            (before_distance - after_distance) / self.config.progress_scale_m,
            -self.config.maximum_progress_reward,
            self.config.maximum_progress_reward,
        )
        contained = after_distance <= self.config.contained_distance_m
        her_success = relabelled & (
            after_distance <= self.config.her_success_distance_m
        )
        reward = progress.astype(np.float32)
        phase_adjustment, phase_components = (
            phase_progress_reward_adjustment_v509(
                arrays["neutral_state"][roots],
                arrays["next_neutral_state"][roots],
                achieved,
                next_achieved,
                desired,
                arrays["valid_contact"][roots],
                relabelled,
                config=self.config,
                strict_target_coverage=arrays[
                    "strict_target_coverage"
                ][roots],
                next_strict_target_coverage=arrays[
                    "next_strict_target_coverage"
                ][roots],
                strict_hold_fraction=arrays["strict_hold_fraction"][roots],
                next_strict_hold_fraction=arrays[
                    "next_strict_hold_fraction"
                ][roots],
            )
        )
        reward += phase_adjustment
        home_adjustment, home_components = (
            home_acquisition_reward_adjustment_v598(
                arrays["neutral_state"][roots],
                arrays["next_neutral_state"][roots],
                desired,
                arrays["valid_contact"][roots],
                relabelled,
                config=self.config,
            )
        )
        reward += home_adjustment
        reward += self.config.contained_bonus * contained
        reward += self.config.her_success_bonus * her_success
        reward += self.config.strict_success_bonus * (
            arrays["strict_success"][roots] & ~relabelled
        )
        reward -= self.config.action_penalty * np.sum(np.square(action), axis=-1)
        target_settle_source = (
            ~relabelled
            & (
                np.maximum(
                    arrays["strict_target_coverage"][roots],
                    arrays["next_strict_target_coverage"][roots],
                )
                >= self.config.target_settle_coverage_threshold
            )
        )
        reward -= (
            self.config.target_settle_action_penalty
            * target_settle_source.astype(np.float32)
            * np.sum(np.square(action), axis=-1)
        )
        reward -= projection_penalty_component_v720
        reward -= self.config.infeasible_action_penalty * (
            ~arrays["action_feasible"][roots]
        )
        reward -= self.config.safety_penalty * arrays["safety_violation"][roots]
        done = arrays["terminal"][roots] | arrays["failure_terminal"][roots] | her_success
        importance = np.power(
            self.transition_count * sampled_probability, -0.4
        )
        importance /= max(float(np.max(importance)), 1.0e-12)
        batch = {
            "observation": observation_with_goal_v43(
                arrays["neutral_state"][roots], desired
            ),
            "next_observation": observation_with_goal_v43(
                arrays["next_neutral_state"][roots], desired
            ),
            "action": action.astype(np.float32),
            "reward": reward.astype(np.float32),
            "done": done.astype(np.float32),
            "action_feasible": arrays["action_feasible"][roots].astype(np.float32),
            "projection_l2": projection_l2,
            "actual_effect_projection_l2_v720": (
                actual_effect_projection_l2_v720
            ),
            "safeguard_projected_action_v720": (
                safeguard_projected_action_v720.astype(np.float32)
            ),
            "safeguard_projection_valid_v720": (
                safeguard_projection_valid_v720
            ),
            "projection_penalty_component_v720": (
                projection_penalty_component_v720
            ),
            "safeguard_projection_penalty_active_v720": np.full(
                batch_size,
                self.config.safeguard_projection_penalty_v720,
                dtype=bool,
            ),
            "importance_weight": importance.astype(np.float32),
            "source_row_index": roots,
            "source_episode_index": arrays["episode_index"][roots],
            "curriculum_frontier_source": frontier_source,
            "episode_initial_distance_m": episode_initial_distance[roots],
            "her_relabelled": relabelled,
            "home_acquisition_future_her_suppressed_v626": (
                acquisition_her_suppressed[roots]
            ),
            "strict_success_source": arrays["strict_success"][roots],
            "home_acquisition_reward_v598": home_adjustment.astype(
                np.float32
            ),
            "home_acquisition_source_v598": home_components[
                "approach_transition"
            ],
            "home_acquisition_distance_progress_m_v598": home_components[
                "distance_progress_m"
            ],
            "home_acquisition_alignment_progress_v598": home_components[
                "alignment_progress"
            ],
            "contact_credit_source": learning_profiles[
                "contact_credit_source"
            ][roots],
            "contact_episode_source": learning_profiles[
                "contact_episode_source"
            ][roots],
            "strict_success_episode_source": learning_profiles[
                "strict_success_episode_source"
            ][roots],
            "effectful_behavior_source": learning_profiles[
                "effectful_behavior_source"
            ][roots],
            "original_task_progress_m": learning_profiles[
                "original_task_progress_m"
            ][roots],
            "productive_contact_reward_source_v509": phase_components[
                "productive_contact"
            ],
            "stalled_contact_penalty_source_v509": phase_components[
                "stalled_contact"
            ],
            "contact_acquisition_reward_source_v509": phase_components[
                "acquired_contact"
            ],
            "contact_loss_penalty_source_v511": phase_components[
                "lost_contact"
            ],
            "target_coverage_gain_v509": phase_components["coverage_gain"],
            "target_coverage_loss_v511": phase_components["coverage_loss"],
            "strict_hold_progress_gain_v509": phase_components["hold_gain"],
            "target_settle_action_penalty_source_v606": (
                target_settle_source
            ),
        }
        if not all(len(value) == batch_size for value in batch.values()):
            raise RuntimeError("V43 replay batch row counts disagree")
        return batch

    def manifest(self) -> dict[str, Any]:
        profiles = self._contact_learning_profiles()
        home_profiles = self._home_acquisition_learning_profiles()
        acquisition_her_suppression = (
            self._home_acquisition_future_her_suppression_mask_v626()
        )
        return {
            "format": GOAL_CONDITIONED_HER_REPLAY_FORMAT_V43,
            "transition_count": self.transition_count,
            "episode_count": self.episode_count,
            "strict_success_episode_count": int(
                sum(bool(row["strict_success"]) for row in self.source_rows)
            ),
            "failure_episode_count": int(
                sum(bool(row["failure_terminal"]) for row in self.source_rows)
            ),
            "valid_contact_transition_count": int(
                np.count_nonzero(self.arrays["valid_contact"])
            ),
            "block_motion_transition_count": int(
                np.count_nonzero(
                    self.arrays["step_block_displacement_m"] >= np.float32(5.0e-5)
                )
            ),
            "contact_credit_transition_count": int(
                np.count_nonzero(profiles["contact_credit_source"])
            ),
            "effectful_behavior_transition_count": int(
                np.count_nonzero(profiles["effectful_behavior_source"])
            ),
            "contact_episode_transition_count": int(
                np.count_nonzero(profiles["contact_episode_source"])
            ),
            "strict_success_episode_transition_count": int(
                np.count_nonzero(
                    profiles["strict_success_episode_source"]
                )
            ),
            "contact_credit_preceding_steps": (
                self.config.contact_credit_preceding_steps
            ),
            "contact_credit_priority": self.config.contact_credit_priority,
            "successful_contact_credit_priority": (
                self.config.successful_contact_credit_priority
            ),
            "effectful_progress_minimum_m": (
                self.config.effectful_progress_minimum_m
            ),
            "home_acquisition_reward_active": (
                self.config.home_acquisition_reward_active
            ),
            "target_settle_action_penalty": (
                self.config.target_settle_action_penalty
            ),
            "target_settle_priority": self.config.target_settle_priority,
            "target_settle_coverage_threshold": (
                self.config.target_settle_coverage_threshold
            ),
            "effectful_home_acquisition_transition_count": int(
                np.count_nonzero(
                    home_profiles["effectful_approach_source"]
                )
            ),
            "home_acquisition_credit_priority": (
                self.config.home_acquisition_credit_priority
            ),
            "home_acquisition_reward_suppressed_on_her_rows": True,
            "home_acquisition_future_her_suppression_distance_m_v626": (
                self.config
                .home_acquisition_suppress_future_her_above_distance_m
            ),
            "home_acquisition_future_her_suppression_tool_height_m_v630": (
                self.config
                .home_acquisition_suppress_future_her_above_tool_height_m
            ),
            "home_acquisition_future_her_suppression_alignment_v630": (
                self.config
                .home_acquisition_suppress_future_her_below_alignment
            ),
            "home_acquisition_future_her_suppressed_transition_count_v626": int(
                np.count_nonzero(acquisition_her_suppression)
            ),
            "home_acquisition_auxiliary_goal_is_not_action_path": True,
            "behavior_anchor_semantics": (
                "effectful_transition_or_original_goal_contact_credit_window_"
                "in_strict_success_episode"
            ),
            "successful_causal_her_rows_excluded": True,
            "long_no_effect_success_tails_excluded": True,
            "failed_contact_windows_excluded_unless_immediately_effectful": True,
            "future_relabel_probability": self.config.future_relabel_probability,
            "her_is_learning_only": True,
            "her_success_is_strict_success": False,
            "failed_episodes_retained": True,
            "source_rows": list(self.source_rows),
            "expert_calls": 0,
            "behavior_cloning_steps": 0,
            "production_admission": False,
        }

    def save_npz(self, path: Path) -> None:
        destination = Path(path).expanduser().resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        partial = destination.with_suffix(destination.suffix + ".partial")
        with partial.open("wb") as stream:
            np.savez_compressed(
                stream,
                **self.arrays,
                next_episode_index=np.asarray([self._next_episode_index], np.int64),
            )
        partial.replace(destination)

    @classmethod
    def load_npz(
        cls,
        path: Path,
        config: GoalConditionedHerSACConfigV43 | None = None,
    ) -> "GoalConditionedHerReplayV43":
        source = Path(path).expanduser().resolve()
        result = cls(config)
        with np.load(source, allow_pickle=False) as archive:
            expected = _empty_replay_arrays()
            for name in _REPLAY_FIELDS:
                if (
                    name in _TASK_METRIC_REPLAY_FIELDS_V512
                    and name not in archive
                ):
                    continue
                value = np.asarray(archive[name], dtype=expected[name].dtype)
                if value.shape[1:] != expected[name].shape[1:]:
                    raise ValueError(f"V43 persisted replay field {name} changed")
                result.arrays[name] = value
            result._next_episode_index = int(archive["next_episode_index"][0])
        if any(
            len(result.arrays[name]) == 0
            for name in _TASK_METRIC_REPLAY_FIELDS_V512
        ):
            result.arrays.update(
                _reconstruct_task_metrics_v512(result.arrays, result.config)
            )
        count = result.transition_count
        if any(len(value) != count for value in result.arrays.values()):
            raise ValueError("V43 persisted replay row counts disagree")
        for episode_index in range(result._next_episode_index):
            rows = np.flatnonzero(result.arrays["episode_index"] == episode_index)
            if not len(rows):
                raise ValueError("V43 persisted replay lost an episode")
            result.source_rows.append(
                {
                    "source": str(source),
                    "episode_index": episode_index,
                    "start_row": int(rows[0]),
                    "row_count": int(len(rows)),
                    "strict_success": bool(
                        np.any(result.arrays["strict_success"][rows])
                    ),
                    "failure_terminal": bool(
                        np.any(result.arrays["failure_terminal"][rows])
                    ),
                }
            )
        return result


class GoalConditionedActorV43(nn.Module):
    def __init__(self, hidden_dim: int = 256) -> None:
        super().__init__()
        self.trunk = nn.Sequential(
            nn.LayerNorm(OBSERVATION_DIM_V43),
            nn.Linear(OBSERVATION_DIM_V43, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.mean = nn.Linear(hidden_dim, ACTION_DIM_V43)
        self.log_std = nn.Linear(hidden_dim, ACTION_DIM_V43)

    def distribution(self, observation: torch.Tensor) -> torch.distributions.Normal:
        hidden = self.trunk(observation)
        mean = self.mean(hidden)
        log_std = torch.clamp(self.log_std(hidden), -5.0, 1.0)
        return torch.distributions.Normal(mean, log_std.exp())

    def sample(
        self,
        observation: torch.Tensor,
        *,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        distribution = self.distribution(observation)
        pre_tanh = distribution.mean if deterministic else distribution.rsample()
        action = torch.tanh(pre_tanh)
        if deterministic:
            log_probability = torch.zeros(
                action.shape[0], device=action.device, dtype=action.dtype
            )
        else:
            correction = torch.log(1.0 - action.square() + 1.0e-6)
            log_probability = (
                distribution.log_prob(pre_tanh) - correction
            ).sum(dim=-1)
        return action, log_probability


class _ActionValueV43(nn.Module):
    def __init__(self, hidden_dim: int = 256) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.LayerNorm(OBSERVATION_DIM_V43 + ACTION_DIM_V43),
            nn.Linear(OBSERVATION_DIM_V43 + ACTION_DIM_V43, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, observation: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.network(torch.cat((observation, action), dim=-1)).squeeze(-1)


class TwinGoalConditionedCriticV43(nn.Module):
    def __init__(self, hidden_dim: int = 256) -> None:
        super().__init__()
        self.q1 = _ActionValueV43(hidden_dim)
        self.q2 = _ActionValueV43(hidden_dim)

    def forward(
        self, observation: torch.Tensor, action: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.q1(observation, action), self.q2(observation, action)


class ActionFeasibilityV43(_ActionValueV43):
    pass


@dataclass
class GoalConditionedHerSACBundleV43:
    actor: GoalConditionedActorV43
    critic: TwinGoalConditionedCriticV43
    target_critic: TwinGoalConditionedCriticV43
    feasibility: ActionFeasibilityV43
    actor_optimizer: torch.optim.Optimizer
    critic_optimizer: torch.optim.Optimizer
    feasibility_optimizer: torch.optim.Optimizer
    update_index: int = 0
    format: str = GOAL_CONDITIONED_HER_SAC_FORMAT_V43


def initialize_goal_conditioned_her_sac_v43(
    seed: int,
    *,
    device: str | torch.device,
    config: GoalConditionedHerSACConfigV43 | None = None,
) -> GoalConditionedHerSACBundleV43:
    if type(seed) is not int or seed < 0:
        raise ValueError("V43 initialization seed must be non-negative")
    selected = config or GoalConditionedHerSACConfigV43()
    selected.validate()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        actor = GoalConditionedActorV43(selected.hidden_dim)
        critic = TwinGoalConditionedCriticV43(selected.hidden_dim)
        target = TwinGoalConditionedCriticV43(selected.hidden_dim)
        feasibility = ActionFeasibilityV43(selected.hidden_dim)
    target.load_state_dict(critic.state_dict(), strict=True)
    for parameter in target.parameters():
        parameter.requires_grad_(False)
    actor = actor.to(device)
    critic = critic.to(device)
    target = target.to(device)
    feasibility = feasibility.to(device)
    return GoalConditionedHerSACBundleV43(
        actor=actor,
        critic=critic,
        target_critic=target,
        feasibility=feasibility,
        actor_optimizer=torch.optim.Adam(
            actor.parameters(), lr=selected.actor_learning_rate
        ),
        critic_optimizer=torch.optim.Adam(
            critic.parameters(), lr=selected.critic_learning_rate
        ),
        feasibility_optimizer=torch.optim.Adam(
            feasibility.parameters(), lr=selected.feasibility_learning_rate
        ),
    )


@dataclass(frozen=True)
class GoalConditionedHerSACUpdateMetricsV43:
    update_index: int
    critic_loss: float
    actor_loss: float
    feasibility_loss: float
    feasibility_accuracy: float
    mean_q_target: float
    mean_q_data: float
    mean_policy_action_abs: float
    mean_policy_forward_action: float
    predicted_policy_feasibility: float
    her_relabel_fraction: float
    strict_success_source_fraction: float
    mean_learning_reward: float
    actor_updated: bool
    format: str = GOAL_CONDITIONED_HER_SAC_FORMAT_V43


def _module_device(module: nn.Module) -> torch.device:
    return next(module.parameters()).device


def _balanced_binary_loss(logit: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    positive = torch.clamp(target.sum(), min=1.0)
    negative = torch.clamp((1.0 - target).sum(), min=1.0)
    weight = torch.where(target > 0.5, 0.5 / positive, 0.5 / negative)
    weight = weight * target.numel()
    return (F.binary_cross_entropy_with_logits(logit, target, reduction="none") * weight).mean()


def goal_conditioned_her_sac_update_v43(
    bundle: GoalConditionedHerSACBundleV43,
    batch: dict[str, np.ndarray],
    config: GoalConditionedHerSACConfigV43,
) -> GoalConditionedHerSACUpdateMetricsV43:
    config.validate()
    if bundle.format != GOAL_CONDITIONED_HER_SAC_FORMAT_V43:
        raise ValueError("V43 bundle identity changed")
    device = _module_device(bundle.actor)
    observation = torch.from_numpy(batch["observation"]).to(device)
    next_observation = torch.from_numpy(batch["next_observation"]).to(device)
    action = torch.from_numpy(batch["action"]).to(device)
    reward = torch.from_numpy(batch["reward"]).to(device)
    done = torch.from_numpy(batch["done"]).to(device)
    feasible = torch.from_numpy(batch["action_feasible"]).to(device)
    importance = torch.from_numpy(batch["importance_weight"]).to(device)
    if observation.ndim != 2 or observation.shape[1] != OBSERVATION_DIM_V43:
        raise ValueError("V43 update observation shape changed")

    bundle.update_index += 1
    update_index = bundle.update_index
    bundle.feasibility.train()
    feasibility_logit = bundle.feasibility(observation, action)
    feasibility_loss = _balanced_binary_loss(feasibility_logit, feasible)
    bundle.feasibility_optimizer.zero_grad(set_to_none=True)
    feasibility_loss.backward()
    nn.utils.clip_grad_norm_(
        bundle.feasibility.parameters(), config.maximum_gradient_norm
    )
    bundle.feasibility_optimizer.step()

    with torch.no_grad():
        next_action, next_log_probability = bundle.actor.sample(next_observation)
        target_q1, target_q2 = bundle.target_critic(next_observation, next_action)
        target_value = torch.minimum(target_q1, target_q2) - (
            config.entropy_temperature * next_log_probability
        )
        q_target = reward + config.gamma * (1.0 - done) * target_value

    bundle.critic.train()
    q1, q2 = bundle.critic(observation, action)
    critic_td = F.smooth_l1_loss(q1, q_target, reduction="none") + F.smooth_l1_loss(
        q2, q_target, reduction="none"
    )
    random_action = torch.empty_like(action).uniform_(-1.0, 1.0)
    random_q1, random_q2 = bundle.critic(observation, random_action)
    conservative = (
        torch.logsumexp(torch.stack((random_q1, q1), dim=0), dim=0) - q1
        + torch.logsumexp(torch.stack((random_q2, q2), dim=0), dim=0) - q2
    ).mean()
    critic_loss = (importance * critic_td).mean() + (
        config.critic_conservative_coefficient * conservative
    )
    bundle.critic_optimizer.zero_grad(set_to_none=True)
    critic_loss.backward()
    nn.utils.clip_grad_norm_(bundle.critic.parameters(), config.maximum_gradient_norm)
    bundle.critic_optimizer.step()

    actor_updated = update_index % config.policy_update_period == 0
    actor_loss_value = 0.0
    mean_policy_action_abs = 0.0
    mean_policy_forward = 0.0
    predicted_policy_feasibility = 0.0
    if actor_updated:
        for module in (bundle.critic, bundle.feasibility):
            for parameter in module.parameters():
                parameter.requires_grad_(False)
        policy_action, log_probability = bundle.actor.sample(observation)
        policy_q1, policy_q2 = bundle.critic(observation, policy_action)
        policy_feasibility = torch.sigmoid(
            bundle.feasibility(observation, policy_action)
        )
        actor_loss = (
            config.entropy_temperature * log_probability
            - torch.minimum(policy_q1, policy_q2)
            + config.actor_infeasibility_coefficient * (1.0 - policy_feasibility)
        ).mean()
        bundle.actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        nn.utils.clip_grad_norm_(bundle.actor.parameters(), config.maximum_gradient_norm)
        bundle.actor_optimizer.step()
        for module in (bundle.critic, bundle.feasibility):
            for parameter in module.parameters():
                parameter.requires_grad_(True)
        actor_loss_value = float(actor_loss.item())
        mean_policy_action_abs = float(policy_action.abs().mean().item())
        mean_policy_forward = float(policy_action[:, 0].mean().item())
        predicted_policy_feasibility = float(policy_feasibility.mean().item())

    with torch.no_grad():
        for target_parameter, parameter in zip(
            bundle.target_critic.parameters(), bundle.critic.parameters(), strict=True
        ):
            target_parameter.mul_(1.0 - config.target_tau).add_(
                parameter, alpha=config.target_tau
            )
        accuracy = ((feasibility_logit >= 0.0) == (feasible > 0.5)).float().mean()

    metrics = GoalConditionedHerSACUpdateMetricsV43(
        update_index=update_index,
        critic_loss=float(critic_loss.item()),
        actor_loss=actor_loss_value,
        feasibility_loss=float(feasibility_loss.item()),
        feasibility_accuracy=float(accuracy.item()),
        mean_q_target=float(q_target.mean().item()),
        mean_q_data=float(torch.minimum(q1, q2).mean().item()),
        mean_policy_action_abs=mean_policy_action_abs,
        mean_policy_forward_action=mean_policy_forward,
        predicted_policy_feasibility=predicted_policy_feasibility,
        her_relabel_fraction=float(np.mean(batch["her_relabelled"])),
        strict_success_source_fraction=float(
            np.mean(batch["strict_success_source"])
        ),
        mean_learning_reward=float(reward.mean().item()),
        actor_updated=actor_updated,
    )
    numeric = asdict(metrics)
    for name, value in numeric.items():
        if isinstance(value, float) and not np.isfinite(value):
            raise RuntimeError(f"V43 update metric {name} is non-finite")
    return metrics


__all__ = [
    "ACTION_DIM_V43",
    "GOAL_CONDITIONED_HER_CHECKPOINT_FORMAT_V43",
    "GOAL_CONDITIONED_HER_REPLAY_FORMAT_V43",
    "GOAL_CONDITIONED_HER_SAC_FORMAT_V43",
    "GOAL_DIM_V43",
    "OBSERVATION_DIM_V43",
    "ActionFeasibilityV43",
    "GoalConditionedActorV43",
    "GoalConditionedHerReplayV43",
    "GoalConditionedHerSACBundleV43",
    "GoalConditionedHerSACConfigV43",
    "GoalConditionedHerSACUpdateMetricsV43",
    "TwinGoalConditionedCriticV43",
    "achieved_goal_from_privileged_v43",
    "desired_goal_from_privileged_v43",
    "goal_conditioned_her_sac_update_v43",
    "goal_neutral_privileged_state_v43",
    "home_acquisition_geometry_v598",
    "home_acquisition_reward_adjustment_v598",
    "initialize_goal_conditioned_her_sac_v43",
    "observation_with_goal_v43",
    "reframe_task_action_for_goal_v43",
]
