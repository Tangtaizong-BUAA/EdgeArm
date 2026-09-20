"""Goal-conditioned SAC for full-task five-joint robot control.

V665 is the learning core for the replacement RL data generator.  It does not
terminate at precontact: every transition remains in the same credit episode
until strict three-second success, a true failure, or the time limit.  The
teacher policy receives simulator state during RL training; wrist/multiview
observations are recorded later for distillation and are never inferred to be
present here.

The observation explicitly includes the controller target, previous applied
action, and an episode-latched admissible-contact bit so safety projection and
transport-phase attribution do not create hidden recurrent state.
The requested policy action, rather than the projected action, conditions the
critic because it is the causal choice that produced the environment outcome.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .goal_conditioned_her_sac_v43 import (
    GOAL_DIM_V43,
    goal_neutral_privileged_state_v43,
)
from .guarded_joint_delta_action_v664 import ARM_JOINT_ACTION_DIM_V664
from .privileged_effect_state_v1 import (
    PRIVILEGED_EFFECT_STATE_DIM,
    privileged_effect_state_slices_v1,
)


JOINT_GOAL_SAC_FORMAT_V665 = "edgearm-v665-full-task-joint-goal-sac-v12"
JOINT_GOAL_REPLAY_FORMAT_V665 = "edgearm-v665-full-task-joint-replay-v3"
CONTROLLER_TARGET_DIM_V665 = 6
PREVIOUS_ACTION_DIM_V665 = ARM_JOINT_ACTION_DIM_V664
CONTACT_LATCH_DIM_V665 = 1
JOINT_GOAL_OBSERVATION_DIM_V665 = (
    PRIVILEGED_EFFECT_STATE_DIM
    + GOAL_DIM_V43
    + CONTROLLER_TARGET_DIM_V665
    + PREVIOUS_ACTION_DIM_V665
    + CONTACT_LATCH_DIM_V665
)
JOINT_TASKFRAME_FEATURE_FORMAT_V682 = "edgearm-v682-current-taskframe-precontact-error-feature-v1"
JOINT_TASKFRAME_FEATURE_DIM_V682 = 3
JOINT_TASKFRAME_FEATURE_SCALE_V682 = np.asarray((0.25, 0.20, 0.15), dtype=np.float32)
JOINT_TASKFRAME_FEATURE_CLIP_V682 = 2.0
JOINT_PRECONTACT_STANDOFF_M_V682 = 0.055
JOINT_PRECONTACT_HEIGHT_M_V682 = 0.055
_PRIVILEGED_SLICES_V682 = privileged_effect_state_slices_v1()
JOINT_ACQUISITION_CONTEXT_FORMAT_V683 = "edgearm-v683-contact-conditioned-acquisition-policy-context-v1"
JOINT_ACQUISITION_CONTEXT_DIM_V683 = (
    JOINT_TASKFRAME_FEATURE_DIM_V682
    + ARM_JOINT_ACTION_DIM_V664
    + ARM_JOINT_ACTION_DIM_V664
    + PREVIOUS_ACTION_DIM_V665
)
JOINT_ACQUISITION_PROGRESS_FORMAT_V686 = "edgearm-v687-executed-action-axis-progress-ensemble-v2"
JOINT_ACQUISITION_PROGRESS_DIM_V686 = JOINT_TASKFRAME_FEATURE_DIM_V682


@dataclass(frozen=True)
class JointGoalSACConfigV665:
    gamma: float = 0.995
    target_tau: float = 0.005
    entropy_temperature: float = 0.08
    actor_learning_rate: float = 3.0e-4
    critic_learning_rate: float = 3.0e-4
    feasibility_learning_rate: float = 3.0e-4
    acquisition_progress_model_learning_rate: float = 3.0e-4
    acquisition_progress_model_huber_delta_m: float = 2.0e-3
    hidden_dim: int = 256
    batch_size: int = 256
    replay_capacity: int = 100_000
    maximum_gradient_norm: float = 10.0
    actor_feasibility_coefficient: float = 0.20
    actor_projection_distillation_coefficient: float = 1.00
    actor_projection_distillation_minimum_gap: float = 0.05
    actor_acquisition_self_imitation_coefficient: float = 1.00
    actor_acquisition_self_imitation_progress_scale_m: float = 0.002
    actor_acquisition_self_imitation_minimum_progress_m: float = 5.0e-4
    actor_acquisition_self_imitation_maximum_weight: float = 5.0
    actor_acquisition_self_imitation_minimum_forward_improvement_m: float = 1.0e-4
    actor_acquisition_self_imitation_maximum_lateral_regression_m: float = 2.5e-4
    actor_acquisition_self_imitation_maximum_height_regression_m: float = 5.0e-4
    importance_beta: float = 0.40
    time_penalty: float = 0.004
    action_penalty: float = 0.002
    projection_penalty: float = 0.08
    guard_intervention_penalty: float = 0.03
    acquisition_progress_scale_m: float = 0.002
    acquisition_progress_reward: float = 0.40
    acquisition_alignment_activation_m: float = 0.18
    alignment_progress_scale: float = 0.02
    alignment_progress_reward: float = 0.10
    object_progress_scale_m: float = 0.001
    object_progress_reward: float = 1.25
    contact_acquisition_bonus: float = 0.50
    productive_contact_bonus: float = 0.12
    stalled_contact_penalty: float = 0.04
    contact_loss_penalty: float = 0.50
    contact_absence_penalty: float = 0.03
    contact_reacquisition_bonus: float = 0.35
    contact_recovery_progress_reward: float = 0.40
    contact_recovery_alignment_reward: float = 0.10
    contact_loss_priority_bonus: float = 10.0
    contact_reacquisition_priority_bonus: float = 12.0
    contact_loss_coverage_exemption: float = 0.95
    coverage_gain_reward: float = 4.0
    strict_hold_gain_reward: float = 6.0
    settle_action_penalty: float = 0.03
    strict_success_bonus: float = 30.0
    failure_terminal_penalty: float = 8.0
    invalid_contact_penalty: float = 6.0
    safety_violation_penalty: float = 10.0
    effectful_block_displacement_m: float = 2.0e-5
    acquisition_progress_priority_threshold_m: float = 5.0e-4
    acquisition_progress_priority_bonus: float = 6.0
    contact_acquisition_priority_bonus: float = 12.0
    contact_episode_priority_bonus: float = 6.0

    def validate(self) -> None:
        for name in ("gamma", "target_tau", "importance_beta"):
            value = float(getattr(self, name))
            if not np.isfinite(value) or not 0.0 < value <= 1.0:
                raise ValueError(f"V665 {name} is invalid")
        positives = (
            self.entropy_temperature,
            self.actor_learning_rate,
            self.critic_learning_rate,
            self.feasibility_learning_rate,
            self.acquisition_progress_model_learning_rate,
            self.acquisition_progress_model_huber_delta_m,
            self.maximum_gradient_norm,
            self.acquisition_progress_scale_m,
            self.acquisition_alignment_activation_m,
            self.alignment_progress_scale,
            self.object_progress_scale_m,
            self.effectful_block_displacement_m,
            self.acquisition_progress_priority_threshold_m,
            self.actor_projection_distillation_minimum_gap,
            self.actor_acquisition_self_imitation_progress_scale_m,
            self.actor_acquisition_self_imitation_minimum_progress_m,
            self.actor_acquisition_self_imitation_maximum_weight,
            self.actor_acquisition_self_imitation_minimum_forward_improvement_m,
            self.actor_acquisition_self_imitation_maximum_lateral_regression_m,
            self.actor_acquisition_self_imitation_maximum_height_regression_m,
        )
        if any(not np.isfinite(value) or value <= 0.0 for value in positives):
            raise ValueError("V665 positive configuration field is invalid")
        nonnegative = np.asarray(
            [
                self.actor_feasibility_coefficient,
                self.actor_projection_distillation_coefficient,
                self.actor_acquisition_self_imitation_coefficient,
                self.time_penalty,
                self.action_penalty,
                self.projection_penalty,
                self.guard_intervention_penalty,
                self.acquisition_progress_reward,
                self.alignment_progress_reward,
                self.object_progress_reward,
                self.contact_acquisition_bonus,
                self.productive_contact_bonus,
                self.stalled_contact_penalty,
                self.contact_loss_penalty,
                self.contact_absence_penalty,
                self.contact_reacquisition_bonus,
                self.contact_recovery_progress_reward,
                self.contact_recovery_alignment_reward,
                self.contact_loss_priority_bonus,
                self.contact_reacquisition_priority_bonus,
                self.coverage_gain_reward,
                self.strict_hold_gain_reward,
                self.settle_action_penalty,
                self.strict_success_bonus,
                self.failure_terminal_penalty,
                self.invalid_contact_penalty,
                self.safety_violation_penalty,
                self.acquisition_progress_priority_bonus,
                self.contact_acquisition_priority_bonus,
                self.contact_episode_priority_bonus,
            ],
            dtype=np.float64,
        )
        if not np.all(np.isfinite(nonnegative)) or np.any(nonnegative < 0.0):
            raise ValueError("V665 reward/penalty configuration is invalid")
        if (
            not np.isfinite(self.contact_loss_coverage_exemption)
            or not 0.0 <= self.contact_loss_coverage_exemption <= 1.0
        ):
            raise ValueError("V665 contact-loss coverage exemption is invalid")
        for name in ("hidden_dim", "batch_size", "replay_capacity"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"V665 {name} must be a positive integer")
        if self.replay_capacity < self.batch_size:
            raise ValueError("V665 replay capacity is smaller than one batch")


def joint_goal_observation_v665(
    privileged_state: np.ndarray,
    desired_goal_xy_m: np.ndarray,
    controller_target_rad: np.ndarray,
    previous_applied_action: np.ndarray,
    admissible_contact_latched: bool,
) -> np.ndarray:
    """Build a Markov goal-conditioned teacher observation."""

    privileged = np.asarray(privileged_state, dtype=np.float32)
    goal = np.asarray(desired_goal_xy_m, dtype=np.float32)
    target = np.asarray(controller_target_rad, dtype=np.float32)
    previous = np.asarray(previous_applied_action, dtype=np.float32)
    if (
        privileged.shape[-1] != PRIVILEGED_EFFECT_STATE_DIM
        or goal.shape[-1] != GOAL_DIM_V43
        or target.shape[-1] != CONTROLLER_TARGET_DIM_V665
        or previous.shape[-1] != PREVIOUS_ACTION_DIM_V665
        or type(admissible_contact_latched) is not bool
        or privileged.shape[:-1] != goal.shape[:-1]
        or privileged.shape[:-1] != target.shape[:-1]
        or privileged.shape[:-1] != previous.shape[:-1]
        or not np.all(np.isfinite(privileged))
        or not np.all(np.isfinite(goal))
        or not np.all(np.isfinite(target))
        or not np.all(np.isfinite(previous))
    ):
        raise ValueError("V665 observation components are invalid")
    neutral = goal_neutral_privileged_state_v43(privileged)
    contact_latch = np.full(
        neutral.shape[:-1] + (CONTACT_LATCH_DIM_V665,),
        float(admissible_contact_latched),
        dtype=np.float32,
    )
    result = np.concatenate(
        (neutral, goal, target, previous, contact_latch),
        axis=-1,
        dtype=np.float32,
    )
    if result.shape[-1] != JOINT_GOAL_OBSERVATION_DIM_V665 or not np.all(np.isfinite(result)):
        raise RuntimeError("V665 constructed observation is invalid")
    return result


def joint_taskframe_features_v682(observation: np.ndarray) -> np.ndarray:
    """Encode the current precontact error in goal-aligned task coordinates.

    This is an algebraic view of fields already present in the Markov state.
    It supplies no route, action, waypoint, phase label, or future information.
    """

    value = np.asarray(observation, dtype=np.float32)
    if value.shape[-1] != JOINT_GOAL_OBSERVATION_DIM_V665 or not np.all(np.isfinite(value)):
        raise ValueError("V682 task-frame feature observation is invalid")
    block_slice = _PRIVILEGED_SLICES_V682["block_pose_xyz_quaternion_wxyz"]
    tool_slice = _PRIVILEGED_SLICES_V682["tool_pose_position_rotation"]
    block = value[..., block_slice.start : block_slice.start + 2]
    tool = value[..., tool_slice.start : tool_slice.start + 3]
    goal_start = PRIVILEGED_EFFECT_STATE_DIM
    goal = value[..., goal_start : goal_start + GOAL_DIM_V43]
    direction = goal - block
    norm = np.linalg.norm(direction, axis=-1, keepdims=True)
    fallback = np.zeros_like(direction)
    fallback[..., 0] = 1.0
    forward = np.where(
        norm > 1.0e-7,
        direction / np.maximum(norm, 1.0e-7),
        fallback,
    )
    lateral = np.stack((-forward[..., 1], forward[..., 0]), axis=-1)
    precontact_xy = block - JOINT_PRECONTACT_STANDOFF_M_V682 * forward
    error_xy = precontact_xy - tool[..., :2]
    raw = np.stack(
        (
            np.sum(error_xy * forward, axis=-1),
            np.sum(error_xy * lateral, axis=-1),
            JOINT_PRECONTACT_HEIGHT_M_V682 - tool[..., 2],
        ),
        axis=-1,
    )
    result = np.clip(
        raw / JOINT_TASKFRAME_FEATURE_SCALE_V682,
        -JOINT_TASKFRAME_FEATURE_CLIP_V682,
        JOINT_TASKFRAME_FEATURE_CLIP_V682,
    ).astype(np.float32)
    if result.shape != value.shape[:-1] + (JOINT_TASKFRAME_FEATURE_DIM_V682,) or not np.all(
        np.isfinite(result)
    ):
        raise RuntimeError("V682 task-frame feature construction failed")
    return result


def _joint_taskframe_features_torch_v682(
    observation: torch.Tensor,
) -> torch.Tensor:
    if observation.shape[-1] != JOINT_GOAL_OBSERVATION_DIM_V665:
        raise ValueError("V682 torch task-frame observation is invalid")
    block_slice = _PRIVILEGED_SLICES_V682["block_pose_xyz_quaternion_wxyz"]
    tool_slice = _PRIVILEGED_SLICES_V682["tool_pose_position_rotation"]
    block = observation[..., block_slice.start : block_slice.start + 2]
    tool = observation[..., tool_slice.start : tool_slice.start + 3]
    goal_start = PRIVILEGED_EFFECT_STATE_DIM
    goal = observation[..., goal_start : goal_start + GOAL_DIM_V43]
    direction = goal - block
    norm = torch.linalg.vector_norm(direction, dim=-1, keepdim=True)
    fallback = torch.zeros_like(direction)
    fallback[..., 0] = 1.0
    forward = torch.where(
        norm > 1.0e-7,
        direction / norm.clamp_min(1.0e-7),
        fallback,
    )
    lateral = torch.stack((-forward[..., 1], forward[..., 0]), dim=-1)
    precontact_xy = block - JOINT_PRECONTACT_STANDOFF_M_V682 * forward
    error_xy = precontact_xy - tool[..., :2]
    raw = torch.stack(
        (
            torch.sum(error_xy * forward, dim=-1),
            torch.sum(error_xy * lateral, dim=-1),
            JOINT_PRECONTACT_HEIGHT_M_V682 - tool[..., 2],
        ),
        dim=-1,
    )
    scale = torch.as_tensor(
        JOINT_TASKFRAME_FEATURE_SCALE_V682,
        dtype=observation.dtype,
        device=observation.device,
    )
    return torch.clamp(
        raw / scale,
        -JOINT_TASKFRAME_FEATURE_CLIP_V682,
        JOINT_TASKFRAME_FEATURE_CLIP_V682,
    )


def joint_acquisition_context_v683(observation: np.ndarray) -> np.ndarray:
    """Return task error plus current joint/control history for acquisition."""

    value = np.asarray(observation, dtype=np.float32)
    if value.shape[-1] != JOINT_GOAL_OBSERVATION_DIM_V665 or not np.all(np.isfinite(value)):
        raise ValueError("V683 acquisition context observation is invalid")
    controller_start = PRIVILEGED_EFFECT_STATE_DIM + GOAL_DIM_V43
    previous_start = controller_start + CONTROLLER_TARGET_DIM_V665
    result = np.concatenate(
        (
            joint_taskframe_features_v682(value),
            value[..., :ARM_JOINT_ACTION_DIM_V664],
            value[
                ...,
                controller_start : controller_start + ARM_JOINT_ACTION_DIM_V664,
            ],
            value[
                ...,
                previous_start : previous_start + PREVIOUS_ACTION_DIM_V665,
            ],
        ),
        axis=-1,
        dtype=np.float32,
    )
    if result.shape != value.shape[:-1] + (JOINT_ACQUISITION_CONTEXT_DIM_V683,) or not np.all(
        np.isfinite(result)
    ):
        raise RuntimeError("V683 acquisition context construction failed")
    return result


def _joint_acquisition_context_torch_v683(
    observation: torch.Tensor,
) -> torch.Tensor:
    if observation.shape[-1] != JOINT_GOAL_OBSERVATION_DIM_V665:
        raise ValueError("V683 torch acquisition context is invalid")
    controller_start = PRIVILEGED_EFFECT_STATE_DIM + GOAL_DIM_V43
    previous_start = controller_start + CONTROLLER_TARGET_DIM_V665
    return torch.cat(
        (
            _joint_taskframe_features_torch_v682(observation),
            observation[..., :ARM_JOINT_ACTION_DIM_V664],
            observation[
                ...,
                controller_start : controller_start + ARM_JOINT_ACTION_DIM_V664,
            ],
            observation[
                ...,
                previous_start : previous_start + PREVIOUS_ACTION_DIM_V665,
            ],
        ),
        dim=-1,
    )


@dataclass(frozen=True)
class JointTeacherRewardV665:
    reward: float
    acquisition_progress_m: float
    alignment_progress: float
    object_progress_m: float
    coverage_gain: float
    strict_hold_gain: float
    contact_acquired: bool
    productive_contact: bool
    stalled_contact: bool
    contact_lost: bool
    contact_absent: bool
    contact_reacquired: bool
    recovery_shaping_active: bool
    transport_evidence: bool
    projection_l2: float
    format: str = JOINT_GOAL_SAC_FORMAT_V665


def joint_teacher_reward_v665(
    *,
    object_distance_before_m: float,
    object_distance_after_m: float,
    precontact_distance_before_m: float,
    precontact_distance_after_m: float,
    alignment_before: float,
    alignment_after: float,
    target_coverage_before: float,
    target_coverage_after: float,
    strict_hold_fraction_before: float,
    strict_hold_fraction_after: float,
    contact_before: bool,
    valid_contact_before: bool,
    valid_contact: bool,
    block_step_displacement_m: float,
    requested_action: np.ndarray,
    applied_action: np.ndarray,
    guard_intervened: bool,
    invalid_contact: bool,
    safety_violation: bool,
    strict_success: bool,
    failure_terminal: bool,
    config: JointGoalSACConfigV665 | None = None,
) -> JointTeacherRewardV665:
    """Full-task dense reward with no action label or scripted phase exit."""

    selected = config or JointGoalSACConfigV665()
    selected.validate()
    requested = np.asarray(requested_action, dtype=np.float64)
    applied = np.asarray(applied_action, dtype=np.float64)
    values = np.asarray(
        [
            object_distance_before_m,
            object_distance_after_m,
            precontact_distance_before_m,
            precontact_distance_after_m,
            alignment_before,
            alignment_after,
            target_coverage_before,
            target_coverage_after,
            strict_hold_fraction_before,
            strict_hold_fraction_after,
            block_step_displacement_m,
        ],
        dtype=np.float64,
    )
    if (
        requested.shape != (ARM_JOINT_ACTION_DIM_V664,)
        or applied.shape != requested.shape
        or not np.all(np.isfinite(np.r_[values, requested, applied]))
        or np.any(np.abs(requested) > 1.0 + 1.0e-6)
        or np.any(np.abs(applied) > 1.0 + 1.0e-6)
        or block_step_displacement_m < 0.0
        or any(
            type(value) is not bool
            for value in (
                contact_before,
                valid_contact_before,
                valid_contact,
                guard_intervened,
                invalid_contact,
                safety_violation,
                strict_success,
                failure_terminal,
            )
        )
    ):
        raise ValueError("V665 reward inputs are invalid")
    if np.any(values[4:10] < -1.0e-6) or np.any(values[4:10] > 1.0 + 1.0e-6):
        raise ValueError("V665 alignment/coverage/hold values escaped [0,1]")

    acquisition_progress = float(precontact_distance_before_m - precontact_distance_after_m)
    alignment_progress = float(alignment_after - alignment_before)
    object_progress = float(object_distance_before_m - object_distance_after_m)
    coverage_gain = float(max(target_coverage_after - target_coverage_before, 0.0))
    hold_gain = float(max(strict_hold_fraction_after - strict_hold_fraction_before, 0.0))
    contact_acquired = bool(not contact_before and valid_contact)
    below_target_coverage = bool(
        max(target_coverage_before, target_coverage_after) < selected.contact_loss_coverage_exemption
    )
    contact_absent = bool(contact_before and not valid_contact and below_target_coverage)
    contact_lost = bool(contact_absent and valid_contact_before)
    contact_reacquired = bool(
        contact_before and not valid_contact_before and valid_contact and below_target_coverage
    )
    # The first objective reaches the moving pre-contact geometry.  It remains
    # active on the transition that establishes first contact and is restored
    # whenever a latched transport episode has lost contact.  This supplies a
    # learnable recovery direction without prescribing an action or path.
    acquisition_phase = bool(not contact_before)
    recovery_phase = contact_absent
    acquisition_term = 0.0
    alignment_term = 0.0
    if acquisition_phase or recovery_phase:
        progress_weight = (
            selected.contact_recovery_progress_reward
            if recovery_phase
            else selected.acquisition_progress_reward
        )
        alignment_weight = (
            selected.contact_recovery_alignment_reward
            if recovery_phase
            else selected.alignment_progress_reward
        )
        acquisition_term = progress_weight * float(
            np.clip(
                acquisition_progress / selected.acquisition_progress_scale_m,
                -1.0,
                1.0,
            )
        )
        alignment_gate = float(
            np.clip(
                1.0
                - min(precontact_distance_before_m, precontact_distance_after_m)
                / selected.acquisition_alignment_activation_m,
                0.0,
                1.0,
            )
        )
        alignment_term = (
            alignment_weight
            * alignment_gate
            * float(
                np.clip(
                    alignment_progress / selected.alignment_progress_scale,
                    -1.0,
                    1.0,
                )
            )
        )
    # Object/coverage/hold motion is only learnable evidence after this episode
    # has established an admissible push-side contact.  This prevents passive
    # drift or a safety-only robot part (for example the camera housing) from
    # being rewarded as if the gripper had pushed the block.
    transport_evidence = bool(contact_before or valid_contact)
    object_term = 0.0
    if transport_evidence:
        object_term = selected.object_progress_reward * float(
            np.clip(
                object_progress / selected.object_progress_scale_m,
                -1.0,
                1.0,
            )
        )
    productive_contact = bool(
        valid_contact
        and block_step_displacement_m >= selected.effectful_block_displacement_m
        and object_progress > 0.0
    )
    stalled_contact = bool(contact_before and valid_contact and not productive_contact)
    projection_l2 = float(np.linalg.norm(requested - applied))
    action_square = float(np.square(requested).sum())
    reward = -selected.time_penalty
    reward += acquisition_term + alignment_term + object_term
    reward += selected.contact_acquisition_bonus * float(contact_acquired)
    reward += selected.productive_contact_bonus * float(productive_contact)
    reward -= selected.stalled_contact_penalty * float(stalled_contact)
    reward -= selected.contact_loss_penalty * float(contact_lost)
    reward -= selected.contact_absence_penalty * float(contact_absent)
    reward += selected.contact_reacquisition_bonus * float(contact_reacquired)
    reward += selected.coverage_gain_reward * coverage_gain * float(transport_evidence)
    reward += selected.strict_hold_gain_reward * hold_gain * float(transport_evidence)
    reward -= selected.action_penalty * action_square
    reward -= selected.projection_penalty * projection_l2
    reward -= selected.guard_intervention_penalty * float(guard_intervened)
    if max(target_coverage_before, target_coverage_after) >= 0.95:
        reward -= selected.settle_action_penalty * action_square
    reward += selected.strict_success_bonus * float(strict_success)
    reward -= selected.failure_terminal_penalty * float(failure_terminal)
    reward -= selected.invalid_contact_penalty * float(invalid_contact)
    reward -= selected.safety_violation_penalty * float(safety_violation)
    if not np.isfinite(reward):
        raise RuntimeError("V665 reward became non-finite")
    return JointTeacherRewardV665(
        reward=float(reward),
        acquisition_progress_m=acquisition_progress,
        alignment_progress=alignment_progress,
        object_progress_m=object_progress,
        coverage_gain=coverage_gain,
        strict_hold_gain=hold_gain,
        contact_acquired=contact_acquired,
        productive_contact=productive_contact,
        stalled_contact=stalled_contact,
        contact_lost=contact_lost,
        contact_absent=contact_absent,
        contact_reacquired=contact_reacquired,
        recovery_shaping_active=recovery_phase,
        transport_evidence=transport_evidence,
        projection_l2=projection_l2,
    )


class JointGoalActorV665(nn.Module):
    def __init__(self, hidden_dim: int = 256) -> None:
        super().__init__()
        # The transport branch is intentionally kept byte-for-byte compatible
        # with V682 so a trained contact policy is not discarded.  V683 adds a
        # separate acquisition branch whose inputs are limited to current
        # Markov state: task-frame error, physical joints, controller target,
        # and the previously applied action.
        self.trunk = nn.Sequential(
            nn.LayerNorm(JOINT_GOAL_OBSERVATION_DIM_V665),
            nn.Linear(JOINT_GOAL_OBSERVATION_DIM_V665, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.mean = nn.Linear(hidden_dim, ARM_JOINT_ACTION_DIM_V664)
        self.log_std = nn.Linear(hidden_dim, ARM_JOINT_ACTION_DIM_V664)
        self.taskframe_feature_projection = nn.Linear(
            JOINT_TASKFRAME_FEATURE_DIM_V682,
            hidden_dim,
            bias=False,
        )
        nn.init.zeros_(self.taskframe_feature_projection.weight)
        self.acquisition_trunk = nn.Sequential(
            nn.LayerNorm(JOINT_ACQUISITION_CONTEXT_DIM_V683),
            nn.Linear(JOINT_ACQUISITION_CONTEXT_DIM_V683, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.acquisition_mean = nn.Linear(hidden_dim, ARM_JOINT_ACTION_DIM_V664)
        self.acquisition_log_std = nn.Linear(hidden_dim, ARM_JOINT_ACTION_DIM_V664)
        nn.init.zeros_(self.acquisition_mean.weight)
        nn.init.zeros_(self.acquisition_mean.bias)
        nn.init.zeros_(self.acquisition_log_std.weight)
        nn.init.zeros_(self.acquisition_log_std.bias)

    def _distribution(
        self,
        observation: torch.Tensor,
    ) -> torch.distributions.Normal:
        taskframe = _joint_taskframe_features_torch_v682(observation)
        hidden = self.trunk[1](self.trunk[0](observation))
        hidden = hidden + self.taskframe_feature_projection(taskframe)
        for layer in self.trunk[2:]:
            hidden = layer(hidden)
        transport_mean = self.mean(hidden)
        transport_log_std = self.log_std(hidden)

        acquisition_context = _joint_acquisition_context_torch_v683(observation)
        acquisition_hidden = self.acquisition_trunk(acquisition_context)
        acquisition_mean = self.acquisition_mean(acquisition_hidden)
        acquisition_log_std = self.acquisition_log_std(acquisition_hidden)

        contact_latch = observation[..., -1:].clamp(0.0, 1.0)
        mean = contact_latch * transport_mean + (1.0 - contact_latch) * acquisition_mean
        log_std = torch.clamp(
            contact_latch * transport_log_std + (1.0 - contact_latch) * acquisition_log_std,
            -5.0,
            1.0,
        )
        return torch.distributions.Normal(mean, log_std.exp())

    def sample(
        self,
        observation: torch.Tensor,
        *,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        distribution = self._distribution(observation)
        mean = distribution.mean
        pre_tanh = mean if deterministic else distribution.rsample()
        action = torch.tanh(pre_tanh)
        if deterministic:
            log_probability = torch.zeros(action.shape[0], dtype=action.dtype, device=action.device)
        else:
            log_probability = (
                distribution.log_prob(pre_tanh) - torch.log(1.0 - action.square() + 1.0e-6)
            ).sum(dim=-1)
        return action, log_probability

    def data_log_probability(
        self,
        observation: torch.Tensor,
        action: torch.Tensor,
    ) -> torch.Tensor:
        """Evaluate replay actions under the current tanh-Gaussian policy."""

        if action.shape != observation.shape[:-1] + (ARM_JOINT_ACTION_DIM_V664,):
            raise ValueError("V684 replay action shape is invalid")
        bounded_action = torch.clamp(action, -0.999999, 0.999999)
        pre_tanh = torch.atanh(bounded_action)
        distribution = self._distribution(observation)
        return (distribution.log_prob(pre_tanh) - torch.log(1.0 - bounded_action.square() + 1.0e-6)).sum(
            dim=-1
        )


class _JointActionValueV665(nn.Module):
    def __init__(self, hidden_dim: int = 256) -> None:
        super().__init__()
        input_dim = JOINT_GOAL_OBSERVATION_DIM_V665 + ARM_JOINT_ACTION_DIM_V664
        self.network = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.taskframe_feature_projection = nn.Linear(
            JOINT_TASKFRAME_FEATURE_DIM_V682,
            hidden_dim,
            bias=False,
        )
        nn.init.zeros_(self.taskframe_feature_projection.weight)

    def forward(self, observation: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        base = torch.cat((observation, action), dim=-1)
        taskframe = _joint_taskframe_features_torch_v682(observation)
        hidden = self.network[1](self.network[0](base))
        hidden = hidden + self.taskframe_feature_projection(taskframe)
        for layer in self.network[2:]:
            hidden = layer(hidden)
        return hidden.squeeze(-1)


class TwinJointGoalCriticV665(nn.Module):
    def __init__(self, hidden_dim: int = 256) -> None:
        super().__init__()
        self.q1 = _JointActionValueV665(hidden_dim)
        self.q2 = _JointActionValueV665(hidden_dim)

    def forward(self, observation: torch.Tensor, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.q1(observation, action), self.q2(observation, action)


class JointActionFeasibilityV665(_JointActionValueV665):
    pass


class _JointAcquisitionAxisProgressV686(nn.Module):
    """Predict one-step absolute task-frame error reduction in metres."""

    def __init__(self, hidden_dim: int = 256) -> None:
        super().__init__()
        input_dim = JOINT_ACQUISITION_CONTEXT_DIM_V683 + ARM_JOINT_ACTION_DIM_V664
        self.network = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, JOINT_ACQUISITION_PROGRESS_DIM_V686),
        )
        # An untrained migration is neutral instead of inventing a preferred
        # direction.  The two independently initialized trunks can diverge as
        # they learn, while their conservative minimum rejects optimism.
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def forward(self, observation: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        if action.shape != observation.shape[:-1] + (ARM_JOINT_ACTION_DIM_V664,):
            raise ValueError("V686 acquisition action shape is invalid")
        acquisition_context = _joint_acquisition_context_torch_v683(observation)
        return self.network(torch.cat((acquisition_context, action), dim=-1))


class TwinJointAcquisitionAxisProgressV686(nn.Module):
    """Twin one-step models used pessimistically during Home acquisition."""

    def __init__(self, hidden_dim: int = 256) -> None:
        super().__init__()
        self.p1 = _JointAcquisitionAxisProgressV686(hidden_dim)
        self.p2 = _JointAcquisitionAxisProgressV686(hidden_dim)

    def forward(
        self,
        observation: torch.Tensor,
        action: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.p1(observation, action), self.p2(observation, action)


@dataclass
class JointGoalSACBundleV665:
    actor: JointGoalActorV665
    critic: TwinJointGoalCriticV665
    target_critic: TwinJointGoalCriticV665
    feasibility: JointActionFeasibilityV665
    acquisition_progress_model: TwinJointAcquisitionAxisProgressV686
    actor_optimizer: torch.optim.Optimizer
    critic_optimizer: torch.optim.Optimizer
    feasibility_optimizer: torch.optim.Optimizer
    acquisition_progress_optimizer: torch.optim.Optimizer
    update_index: int = 0
    format: str = JOINT_GOAL_SAC_FORMAT_V665


def initialize_joint_goal_sac_v665(
    seed: int,
    *,
    device: str | torch.device,
    config: JointGoalSACConfigV665 | None = None,
) -> JointGoalSACBundleV665:
    selected = config or JointGoalSACConfigV665()
    selected.validate()
    if type(seed) is not int or seed < 0:
        raise ValueError("V665 initialization seed is invalid")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        actor = JointGoalActorV665(selected.hidden_dim)
        critic = TwinJointGoalCriticV665(selected.hidden_dim)
        target = TwinJointGoalCriticV665(selected.hidden_dim)
        feasibility = JointActionFeasibilityV665(selected.hidden_dim)
        acquisition_progress_model = TwinJointAcquisitionAxisProgressV686(selected.hidden_dim)
    target.load_state_dict(critic.state_dict(), strict=True)
    for parameter in target.parameters():
        parameter.requires_grad_(False)
    actor = actor.to(device)
    critic = critic.to(device)
    target = target.to(device)
    feasibility = feasibility.to(device)
    acquisition_progress_model = acquisition_progress_model.to(device)
    return JointGoalSACBundleV665(
        actor=actor,
        critic=critic,
        target_critic=target,
        feasibility=feasibility,
        acquisition_progress_model=acquisition_progress_model,
        actor_optimizer=torch.optim.Adam(actor.parameters(), lr=selected.actor_learning_rate),
        critic_optimizer=torch.optim.Adam(critic.parameters(), lr=selected.critic_learning_rate),
        feasibility_optimizer=torch.optim.Adam(
            feasibility.parameters(), lr=selected.feasibility_learning_rate
        ),
        acquisition_progress_optimizer=torch.optim.Adam(
            acquisition_progress_model.parameters(),
            lr=selected.acquisition_progress_model_learning_rate,
        ),
    )


class JointGoalReplayV665:
    """Prioritized replay that preserves sparse acquisition evidence.

    Once an episode reaches admissible contact, all of its already-observed
    pre-contact transitions receive a persistent base-priority floor.  This is
    a causal backward credit mechanism: it does not invent an expert action or
    relabel failure as success, but prevents rare approach evidence from being
    overwritten by later TD-error updates.
    """

    def __init__(
        self,
        capacity: int,
        *,
        acquisition_progress_priority_bonus: float = 6.0,
        contact_acquisition_priority_bonus: float = 12.0,
        contact_episode_priority_bonus: float = 6.0,
        contact_loss_priority_bonus: float = 10.0,
        contact_reacquisition_priority_bonus: float = 12.0,
    ) -> None:
        if type(capacity) is not int or capacity < 1:
            raise ValueError("V665 replay capacity is invalid")
        bonuses = np.asarray(
            [
                acquisition_progress_priority_bonus,
                contact_acquisition_priority_bonus,
                contact_episode_priority_bonus,
                contact_loss_priority_bonus,
                contact_reacquisition_priority_bonus,
            ],
            dtype=np.float64,
        )
        if not np.all(np.isfinite(bonuses)) or np.any(bonuses < 0.0):
            raise ValueError("V665 replay event priority bonus is invalid")
        self.capacity = capacity
        self.acquisition_progress_priority_bonus = float(bonuses[0])
        self.contact_acquisition_priority_bonus = float(bonuses[1])
        self.contact_episode_priority_bonus = float(bonuses[2])
        self.contact_loss_priority_bonus = float(bonuses[3])
        self.contact_reacquisition_priority_bonus = float(bonuses[4])
        self.size = 0
        self.cursor = 0
        self.observation = np.empty((capacity, JOINT_GOAL_OBSERVATION_DIM_V665), dtype=np.float32)
        self.next_observation = np.empty_like(self.observation)
        self.action = np.empty((capacity, ARM_JOINT_ACTION_DIM_V664), dtype=np.float32)
        self.applied_action = np.empty_like(self.action)
        self.reward = np.empty(capacity, dtype=np.float32)
        self.terminal = np.empty(capacity, dtype=bool)
        self.action_feasible = np.empty(capacity, dtype=bool)
        self.strict_success = np.empty(capacity, dtype=bool)
        self.valid_contact = np.empty(capacity, dtype=bool)
        self.effectful_block_motion = np.empty(capacity, dtype=bool)
        self.exact_home_start = np.empty(capacity, dtype=bool)
        self.acquisition_progress = np.empty(capacity, dtype=bool)
        self.contact_acquired = np.empty(capacity, dtype=bool)
        self.contact_lost = np.empty(capacity, dtype=bool)
        self.contact_absent = np.empty(capacity, dtype=bool)
        self.contact_reacquired = np.empty(capacity, dtype=bool)
        self.contact_episode_evidence = np.empty(capacity, dtype=bool)
        self.base_priority = np.empty(capacity, dtype=np.float32)
        self.priority = np.empty(capacity, dtype=np.float32)
        self.episode_index = np.empty(capacity, dtype=np.int64)
        self.episode_step = np.empty(capacity, dtype=np.int32)

    def add(
        self,
        *,
        observation: np.ndarray,
        next_observation: np.ndarray,
        action: np.ndarray,
        applied_action: np.ndarray,
        reward: float,
        terminal: bool,
        action_feasible: bool,
        strict_success: bool,
        valid_contact: bool,
        effectful_block_motion: bool,
        exact_home_start: bool,
        acquisition_progress: bool,
        contact_acquired: bool,
        contact_lost: bool,
        contact_absent: bool,
        contact_reacquired: bool,
        episode_index: int,
        episode_step: int,
    ) -> None:
        vectors = {
            "observation": (
                observation,
                (JOINT_GOAL_OBSERVATION_DIM_V665,),
            ),
            "next_observation": (
                next_observation,
                (JOINT_GOAL_OBSERVATION_DIM_V665,),
            ),
            "action": (action, (ARM_JOINT_ACTION_DIM_V664,)),
            "applied_action": (
                applied_action,
                (ARM_JOINT_ACTION_DIM_V664,),
            ),
        }
        normalized: dict[str, np.ndarray] = {}
        for name, (value, shape) in vectors.items():
            array = np.asarray(value, dtype=np.float32)
            if array.shape != shape or not np.all(np.isfinite(array)):
                raise ValueError(f"V665 replay {name} is invalid")
            normalized[name] = array
        if (
            not np.isfinite(float(reward))
            or type(episode_index) is not int
            or episode_index < 0
            or type(episode_step) is not int
            or episode_step < 0
            or any(
                type(value) is not bool
                for value in (
                    terminal,
                    action_feasible,
                    strict_success,
                    valid_contact,
                    effectful_block_motion,
                    exact_home_start,
                    acquisition_progress,
                    contact_acquired,
                    contact_lost,
                    contact_absent,
                    contact_reacquired,
                )
            )
            or strict_success
            and not terminal
            or contact_acquired
            and not valid_contact
            or contact_lost
            and not contact_absent
            or contact_reacquired
            and not valid_contact
            or contact_reacquired
            and contact_absent
        ):
            raise ValueError("V665 replay scalar metadata is invalid")
        index = self.cursor
        self.observation[index] = normalized["observation"]
        self.next_observation[index] = normalized["next_observation"]
        self.action[index] = normalized["action"]
        self.applied_action[index] = normalized["applied_action"]
        self.reward[index] = np.float32(reward)
        self.terminal[index] = terminal
        self.action_feasible[index] = action_feasible
        self.strict_success[index] = strict_success
        self.valid_contact[index] = valid_contact
        self.effectful_block_motion[index] = effectful_block_motion
        self.exact_home_start[index] = exact_home_start
        self.acquisition_progress[index] = acquisition_progress
        self.contact_acquired[index] = contact_acquired
        self.contact_lost[index] = contact_lost
        self.contact_absent[index] = contact_absent
        self.contact_reacquired[index] = contact_reacquired
        self.contact_episode_evidence[index] = False
        self.episode_index[index] = episode_index
        self.episode_step[index] = episode_step
        base_priority = 1.0
        base_priority += 4.0 * float(not action_feasible)
        base_priority += 8.0 * float(valid_contact)
        base_priority += 8.0 * float(effectful_block_motion)
        base_priority += 32.0 * float(strict_success)
        base_priority += 2.0 * float(exact_home_start)
        base_priority += self.acquisition_progress_priority_bonus * float(acquisition_progress)
        base_priority += self.contact_acquisition_priority_bonus * float(contact_acquired)
        base_priority += self.contact_loss_priority_bonus * float(contact_lost)
        base_priority += self.contact_reacquisition_priority_bonus * float(contact_reacquired)
        self.base_priority[index] = np.float32(base_priority)
        self.priority[index] = np.float32(base_priority)
        self.cursor = (self.cursor + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def mark_contact_episode(self, episode_index: int) -> int:
        """Boost stored approach steps after the episode first reaches contact."""

        if type(episode_index) is not int or episode_index < 0:
            raise ValueError("V665 contact episode index is invalid")
        indices = np.flatnonzero(
            (self.episode_index[: self.size] == episode_index) & ~self.contact_episode_evidence[: self.size]
        )
        if indices.size == 0:
            return 0
        self.contact_episode_evidence[indices] = True
        self.base_priority[indices] += np.float32(self.contact_episode_priority_bonus)
        self.priority[indices] = np.maximum(self.priority[indices], self.base_priority[indices])
        return int(indices.size)

    def sample(
        self,
        batch_size: int,
        *,
        rng: np.random.Generator,
        importance_beta: float,
    ) -> dict[str, np.ndarray]:
        if self.size < 1 or type(batch_size) is not int or batch_size < 1:
            raise ValueError("V665 replay sample request is invalid")
        priorities = np.maximum(self.priority[: self.size].astype(np.float64), 1.0e-6)
        probability = priorities / float(np.sum(priorities))
        indices = rng.choice(
            self.size,
            size=batch_size,
            replace=self.size < batch_size,
            p=probability,
        ).astype(np.int64)
        weights = np.power(self.size * probability[indices], -float(importance_beta))
        weights /= max(float(np.max(weights)), 1.0e-12)
        return {
            "indices": indices,
            "importance": weights.astype(np.float32),
            "observation": self.observation[indices].copy(),
            "next_observation": self.next_observation[indices].copy(),
            "action": self.action[indices].copy(),
            "applied_action": self.applied_action[indices].copy(),
            "reward": self.reward[indices].copy(),
            "terminal": self.terminal[indices].copy(),
            "action_feasible": self.action_feasible[indices].copy(),
            "exact_home_start": self.exact_home_start[indices].copy(),
            "acquisition_progress": self.acquisition_progress[indices].copy(),
        }

    def update_priorities(self, indices: np.ndarray, values: np.ndarray) -> None:
        selected = np.asarray(indices, dtype=np.int64)
        priorities = np.asarray(values, dtype=np.float32)
        if (
            selected.shape != priorities.shape
            or selected.ndim != 1
            or np.any(selected < 0)
            or np.any(selected >= self.size)
            or not np.all(np.isfinite(priorities))
            or np.any(priorities <= 0.0)
        ):
            raise ValueError("V665 replay priority update is invalid")
        for index in np.unique(selected):
            submitted = priorities[selected == index]
            self.priority[index] = np.float32(
                max(
                    float(np.max(submitted)),
                    float(self.base_priority[index]),
                    1.0e-3,
                )
            )

    def save(self, path: Path) -> None:
        destination = Path(path).expanduser().resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(destination.name + ".tmp.npz")
        np.savez(
            temporary,
            format=np.asarray(JOINT_GOAL_REPLAY_FORMAT_V665),
            capacity=np.asarray(self.capacity, dtype=np.int64),
            size=np.asarray(self.size, dtype=np.int64),
            cursor=np.asarray(self.cursor, dtype=np.int64),
            observation=self.observation[: self.size],
            next_observation=self.next_observation[: self.size],
            action=self.action[: self.size],
            applied_action=self.applied_action[: self.size],
            reward=self.reward[: self.size],
            terminal=self.terminal[: self.size],
            action_feasible=self.action_feasible[: self.size],
            strict_success=self.strict_success[: self.size],
            valid_contact=self.valid_contact[: self.size],
            effectful_block_motion=self.effectful_block_motion[: self.size],
            exact_home_start=self.exact_home_start[: self.size],
            acquisition_progress=self.acquisition_progress[: self.size],
            contact_acquired=self.contact_acquired[: self.size],
            contact_lost=self.contact_lost[: self.size],
            contact_absent=self.contact_absent[: self.size],
            contact_reacquired=self.contact_reacquired[: self.size],
            contact_episode_evidence=(self.contact_episode_evidence[: self.size]),
            base_priority=self.base_priority[: self.size],
            priority=self.priority[: self.size],
            episode_index=self.episode_index[: self.size],
            episode_step=self.episode_step[: self.size],
        )
        temporary.replace(destination)

    def restore(self, path: Path) -> dict[str, Any]:
        """Restore one exact-format replay into an empty continuation buffer.

        Checkpoints preserve the learned networks, but a true off-policy
        continuation must also preserve the rare approach/contact/recovery
        transitions that trained those networks.  This loader is deliberately
        fail-closed: it accepts only the current replay schema, validates every
        stored array and causal event relationship, and never silently truncates
        a source replay to fit a smaller destination.
        """

        source = Path(path).expanduser().resolve()
        if self.size != 0 or self.cursor != 0:
            raise ValueError("V665 replay restore destination is not empty")
        if not source.is_file():
            raise FileNotFoundError(f"V665 replay restore source missing: {source}")
        required = {
            "format",
            "capacity",
            "size",
            "cursor",
            "observation",
            "next_observation",
            "action",
            "applied_action",
            "reward",
            "terminal",
            "action_feasible",
            "strict_success",
            "valid_contact",
            "effectful_block_motion",
            "exact_home_start",
            "acquisition_progress",
            "contact_acquired",
            "contact_lost",
            "contact_absent",
            "contact_reacquired",
            "contact_episode_evidence",
            "base_priority",
            "priority",
            "episode_index",
            "episode_step",
        }
        with np.load(source, allow_pickle=False) as payload:
            if set(payload.files) != required:
                missing = sorted(required - set(payload.files))
                extra = sorted(set(payload.files) - required)
                raise ValueError(f"V665 replay restore schema mismatch: missing={missing}, extra={extra}")
            source_format = str(payload["format"].item())
            source_capacity = int(payload["capacity"].item())
            source_size = int(payload["size"].item())
            source_cursor = int(payload["cursor"].item())
            if (
                source_format != JOINT_GOAL_REPLAY_FORMAT_V665
                or source_capacity < 1
                or source_size < 0
                or source_size > source_capacity
                or source_size > self.capacity
                or source_cursor < 0
                or source_cursor >= source_capacity
                or source_size < source_capacity
                and source_cursor != source_size
                or source_size == source_capacity
                and self.capacity != source_capacity
            ):
                raise ValueError("V665 replay restore metadata is incompatible")

            vector_shapes = {
                "observation": (source_size, JOINT_GOAL_OBSERVATION_DIM_V665),
                "next_observation": (
                    source_size,
                    JOINT_GOAL_OBSERVATION_DIM_V665,
                ),
                "action": (source_size, ARM_JOINT_ACTION_DIM_V664),
                "applied_action": (
                    source_size,
                    ARM_JOINT_ACTION_DIM_V664,
                ),
            }
            scalar_names = required - {
                "format",
                "capacity",
                "size",
                "cursor",
                *vector_shapes,
            }
            arrays: dict[str, np.ndarray] = {}
            for name, shape in vector_shapes.items():
                value = np.asarray(payload[name])
                if value.shape != shape or not np.all(np.isfinite(value)):
                    raise ValueError(f"V665 restored replay {name} is invalid")
                arrays[name] = value
            for name in scalar_names:
                value = np.asarray(payload[name])
                if value.shape != (source_size,):
                    raise ValueError(f"V665 restored replay {name} is invalid")
                arrays[name] = value

            floating = (
                "reward",
                "base_priority",
                "priority",
            )
            if any(not np.all(np.isfinite(arrays[name])) for name in floating) or any(
                np.any(arrays[name] <= 0.0) for name in ("base_priority", "priority")
            ):
                raise ValueError("V665 restored replay priorities are invalid")
            boolean_names = (
                "terminal",
                "action_feasible",
                "strict_success",
                "valid_contact",
                "effectful_block_motion",
                "exact_home_start",
                "acquisition_progress",
                "contact_acquired",
                "contact_lost",
                "contact_absent",
                "contact_reacquired",
                "contact_episode_evidence",
            )
            if any(arrays[name].dtype != np.bool_ for name in boolean_names):
                raise ValueError("V665 restored replay boolean dtype is invalid")
            if (
                np.any(arrays["strict_success"] & ~arrays["terminal"])
                or np.any(arrays["contact_acquired"] & ~arrays["valid_contact"])
                or np.any(arrays["contact_lost"] & ~arrays["contact_absent"])
                or np.any(arrays["contact_reacquired"] & ~arrays["valid_contact"])
                or np.any(arrays["contact_reacquired"] & arrays["contact_absent"])
                or np.any(arrays["episode_index"] < 0)
                or np.any(arrays["episode_step"] < 0)
                or np.any(np.abs(arrays["action"]) > 1.0 + 1.0e-6)
                or np.any(np.abs(arrays["applied_action"]) > 1.0 + 1.0e-6)
            ):
                raise ValueError("V665 restored replay causal metadata is invalid")

            for name in (
                "observation",
                "next_observation",
                "action",
                "applied_action",
                "reward",
                "terminal",
                "action_feasible",
                "strict_success",
                "valid_contact",
                "effectful_block_motion",
                "exact_home_start",
                "acquisition_progress",
                "contact_acquired",
                "contact_lost",
                "contact_absent",
                "contact_reacquired",
                "contact_episode_evidence",
                "base_priority",
                "priority",
                "episode_index",
                "episode_step",
            ):
                getattr(self, name)[:source_size] = arrays[name]
        self.size = source_size
        self.cursor = source_cursor if source_size == self.capacity else source_size
        return {
            "format": "edgearm-v678-replay-continuation-audit-v1",
            "source_replay": str(source),
            "source_replay_format": source_format,
            "source_capacity": source_capacity,
            "restored_transition_count": source_size,
            "destination_capacity": self.capacity,
            "schema_exact": True,
            "causal_metadata_validated": True,
            "production_admission": False,
        }


def _acquisition_self_imitation_loss_v684(
    actor: JointGoalActorV665,
    *,
    observation: torch.Tensor,
    next_observation: torch.Tensor,
    applied_action: torch.Tensor,
    action_feasible: torch.Tensor,
    exact_home_start: torch.Tensor,
    recorded_acquisition_progress: torch.Tensor,
    importance: torch.Tensor,
    config: JointGoalSACConfigV665,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Imitate only self-generated safe Home actions with measured progress."""

    scale = torch.as_tensor(
        JOINT_TASKFRAME_FEATURE_SCALE_V682,
        dtype=observation.dtype,
        device=observation.device,
    )
    distance_before = torch.linalg.vector_norm(
        _joint_taskframe_features_torch_v682(observation) * scale,
        dim=-1,
    )
    distance_after = torch.linalg.vector_norm(
        _joint_taskframe_features_torch_v682(next_observation) * scale,
        dim=-1,
    )
    measured_progress = distance_before - distance_after
    absolute_axis_improvement = torch.abs(
        _joint_taskframe_features_torch_v682(observation) * scale
    ) - torch.abs(_joint_taskframe_features_torch_v682(next_observation) * scale)
    contact_latched = observation[..., -1] >= 0.5
    candidate_mask = (
        (exact_home_start >= 0.5)
        & ~contact_latched
        & (action_feasible >= 0.5)
        & (recorded_acquisition_progress >= 0.5)
        & (measured_progress >= config.actor_acquisition_self_imitation_minimum_progress_m)
    )
    axis_consistent = (
        (
            absolute_axis_improvement[..., 0]
            >= config.actor_acquisition_self_imitation_minimum_forward_improvement_m
        )
        & (
            absolute_axis_improvement[..., 1]
            >= -config.actor_acquisition_self_imitation_maximum_lateral_regression_m
        )
        & (
            absolute_axis_improvement[..., 2]
            >= -config.actor_acquisition_self_imitation_maximum_height_regression_m
        )
    )
    mask = (candidate_mask & axis_consistent).float()
    progress_weight = torch.clamp(
        measured_progress / config.actor_acquisition_self_imitation_progress_scale_m,
        min=0.0,
        max=config.actor_acquisition_self_imitation_maximum_weight,
    )
    weight = mask * importance * progress_weight
    data_log_probability = actor.data_log_probability(
        observation,
        applied_action.detach(),
    )
    loss = -(weight * data_log_probability).sum() / weight.sum().clamp_min(1.0)
    return (
        loss,
        mask,
        measured_progress,
        weight,
        absolute_axis_improvement,
        candidate_mask.float(),
    )


@dataclass(frozen=True)
class JointAcquisitionSelfImitationMetricsV684:
    loss: float
    sample_fraction: float
    qualified_sample_count: int
    mean_qualified_progress_m: float
    axis_regression_rejected_sample_count: int
    actor_acquisition_parameter_l2: float
    format: str = JOINT_GOAL_SAC_FORMAT_V665


def pretrain_joint_goal_acquisition_v684(
    bundle: JointGoalSACBundleV665,
    replay: JointGoalReplayV665,
    *,
    config: JointGoalSACConfigV665,
    rng: np.random.Generator,
) -> JointAcquisitionSelfImitationMetricsV684:
    """Run one actor-only update from the agent's own effective Home steps."""

    config.validate()
    if replay.size < config.batch_size:
        raise ValueError("V684 replay is too small for acquisition pretraining")
    batch = replay.sample(
        config.batch_size,
        rng=rng,
        importance_beta=config.importance_beta,
    )
    device = next(bundle.actor.parameters()).device
    observation = torch.from_numpy(batch["observation"]).to(device)
    next_observation = torch.from_numpy(batch["next_observation"]).to(device)
    applied_action = torch.from_numpy(batch["applied_action"]).to(device)
    action_feasible = torch.from_numpy(batch["action_feasible"].astype(np.float32)).to(device)
    exact_home_start = torch.from_numpy(batch["exact_home_start"].astype(np.float32)).to(device)
    acquisition_progress = torch.from_numpy(batch["acquisition_progress"].astype(np.float32)).to(device)
    importance = torch.from_numpy(batch["importance"]).to(device)
    (
        loss,
        mask,
        measured_progress,
        _weight,
        _absolute_axis_improvement,
        candidate_mask,
    ) = _acquisition_self_imitation_loss_v684(
        bundle.actor,
        observation=observation,
        next_observation=next_observation,
        applied_action=applied_action,
        action_feasible=action_feasible,
        exact_home_start=exact_home_start,
        recorded_acquisition_progress=acquisition_progress,
        importance=importance,
        config=config,
    )
    qualified_count = int(mask.sum().item())
    if qualified_count > 0:
        bundle.actor_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(
            bundle.actor.parameters(),
            config.maximum_gradient_norm,
        )
        bundle.actor_optimizer.step()
    qualified_progress = (measured_progress * mask).sum() / mask.sum().clamp_min(1.0)
    with torch.no_grad():
        acquisition_l2 = torch.sqrt(
            sum(
                torch.square(parameter).sum()
                for name, parameter in bundle.actor.named_parameters()
                if name.startswith("acquisition_")
            )
        )
    return JointAcquisitionSelfImitationMetricsV684(
        loss=float(loss.item()),
        sample_fraction=float(mask.mean().item()),
        qualified_sample_count=qualified_count,
        mean_qualified_progress_m=float(qualified_progress.item()),
        axis_regression_rejected_sample_count=int((candidate_mask.sum() - mask.sum()).item()),
        actor_acquisition_parameter_l2=float(acquisition_l2.item()),
    )


def _acquisition_axis_progress_loss_v686(
    model: TwinJointAcquisitionAxisProgressV686,
    *,
    observation: torch.Tensor,
    next_observation: torch.Tensor,
    executed_action: torch.Tensor,
    exact_home_start: torch.Tensor,
    importance: torch.Tensor,
    config: JointGoalSACConfigV665,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fit executed actions to their measured three-axis plant outcome."""

    scale = torch.as_tensor(
        JOINT_TASKFRAME_FEATURE_SCALE_V682,
        dtype=observation.dtype,
        device=observation.device,
    )
    error_before = torch.abs(_joint_taskframe_features_torch_v682(observation) * scale)
    error_after = torch.abs(_joint_taskframe_features_torch_v682(next_observation) * scale)
    target = (error_before - error_after).detach()
    contact_latched = observation[..., -1] >= 0.5
    mask = ((exact_home_start >= 0.5) & ~contact_latched).float()
    prediction1, prediction2 = model(observation, executed_action)
    loss1 = F.smooth_l1_loss(
        prediction1,
        target,
        reduction="none",
        beta=config.acquisition_progress_model_huber_delta_m,
    ).mean(dim=-1)
    loss2 = F.smooth_l1_loss(
        prediction2,
        target,
        reduction="none",
        beta=config.acquisition_progress_model_huber_delta_m,
    ).mean(dim=-1)
    weight = mask * importance
    loss = (weight * (loss1 + loss2)).sum() / weight.sum().clamp_min(1.0)
    conservative_prediction = torch.minimum(prediction1, prediction2)
    return loss, mask, target, conservative_prediction


@dataclass(frozen=True)
class JointAcquisitionAxisProgressMetricsV686:
    loss: float
    sample_fraction: float
    qualified_sample_count: int
    mean_absolute_error_m: float
    per_axis_mean_absolute_error_m: tuple[float, float, float]
    format: str = JOINT_ACQUISITION_PROGRESS_FORMAT_V686


def pretrain_joint_acquisition_progress_v686(
    bundle: JointGoalSACBundleV665,
    replay: JointGoalReplayV665,
    *,
    config: JointGoalSACConfigV665,
    rng: np.random.Generator,
) -> JointAcquisitionAxisProgressMetricsV686:
    """Train the Home-only one-step acquisition model from replay."""

    config.validate()
    if replay.size < config.batch_size:
        raise ValueError("V686 replay is too small for acquisition-model pretraining")
    batch = replay.sample(
        config.batch_size,
        rng=rng,
        importance_beta=config.importance_beta,
    )
    device = next(bundle.acquisition_progress_model.parameters()).device
    observation = torch.from_numpy(batch["observation"]).to(device)
    next_observation = torch.from_numpy(batch["next_observation"]).to(device)
    executed_action = torch.from_numpy(batch["applied_action"]).to(device)
    exact_home_start = torch.from_numpy(batch["exact_home_start"].astype(np.float32)).to(device)
    importance = torch.from_numpy(batch["importance"]).to(device)
    loss, mask, target, prediction = _acquisition_axis_progress_loss_v686(
        bundle.acquisition_progress_model,
        observation=observation,
        next_observation=next_observation,
        executed_action=executed_action,
        exact_home_start=exact_home_start,
        importance=importance,
        config=config,
    )
    qualified_count = int(mask.sum().item())
    if qualified_count > 0:
        bundle.acquisition_progress_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(
            bundle.acquisition_progress_model.parameters(),
            config.maximum_gradient_norm,
        )
        bundle.acquisition_progress_optimizer.step()
    with torch.no_grad():
        absolute_error = torch.abs(prediction - target)
        per_axis = (absolute_error * mask[:, None]).sum(dim=0) / mask.sum().clamp_min(1.0)
    return JointAcquisitionAxisProgressMetricsV686(
        loss=float(loss.item()),
        sample_fraction=float(mask.mean().item()),
        qualified_sample_count=qualified_count,
        mean_absolute_error_m=float(per_axis.mean().item()),
        per_axis_mean_absolute_error_m=tuple(float(value) for value in per_axis.tolist()),
    )


@dataclass(frozen=True)
class JointGoalSACUpdateMetricsV665:
    update_index: int
    critic_loss: float
    actor_loss: float
    actor_task_loss: float
    actor_projection_distillation_loss: float
    projection_distillation_sample_fraction: float
    mean_requested_applied_action_gap: float
    actor_taskframe_projection_l2: float
    critic_taskframe_projection_l2: float
    feasibility_taskframe_projection_l2: float
    acquisition_policy_sample_fraction: float
    actor_acquisition_parameter_l2: float
    acquisition_self_imitation_loss: float
    acquisition_self_imitation_sample_fraction: float
    acquisition_self_imitation_mean_progress_m: float
    acquisition_self_imitation_axis_rejection_fraction: float
    acquisition_progress_model_loss: float
    acquisition_progress_model_sample_fraction: float
    acquisition_progress_model_mean_absolute_error_m: float
    feasibility_loss: float
    feasibility_accuracy: float
    mean_q_target: float
    mean_q_data: float
    mean_policy_action_abs: float
    predicted_policy_feasibility: float
    mean_reward: float
    maximum_td_error: float
    format: str = JOINT_GOAL_SAC_FORMAT_V665


def update_joint_goal_sac_v665(
    bundle: JointGoalSACBundleV665,
    replay: JointGoalReplayV665,
    *,
    config: JointGoalSACConfigV665,
    rng: np.random.Generator,
) -> JointGoalSACUpdateMetricsV665:
    if bundle.format != JOINT_GOAL_SAC_FORMAT_V665:
        raise ValueError("V665 bundle format changed")
    config.validate()
    if replay.size < config.batch_size:
        raise ValueError("V665 replay is too small for one update")
    batch = replay.sample(
        config.batch_size,
        rng=rng,
        importance_beta=config.importance_beta,
    )
    device = next(bundle.actor.parameters()).device
    observation = torch.from_numpy(batch["observation"]).to(device)
    next_observation = torch.from_numpy(batch["next_observation"]).to(device)
    action = torch.from_numpy(batch["action"]).to(device)
    applied_action = torch.from_numpy(batch["applied_action"]).to(device)
    reward = torch.from_numpy(batch["reward"]).to(device)
    terminal = torch.from_numpy(batch["terminal"].astype(np.float32)).to(device)
    feasible_target = torch.from_numpy(batch["action_feasible"].astype(np.float32)).to(device)
    exact_home_start = torch.from_numpy(batch["exact_home_start"].astype(np.float32)).to(device)
    acquisition_progress = torch.from_numpy(batch["acquisition_progress"].astype(np.float32)).to(device)
    importance = torch.from_numpy(batch["importance"]).to(device)

    with torch.no_grad():
        next_action, next_log_probability = bundle.actor.sample(next_observation)
        target_q1, target_q2 = bundle.target_critic(next_observation, next_action)
        target_q = torch.minimum(target_q1, target_q2)
        target_q -= config.entropy_temperature * next_log_probability
        q_target = reward + config.gamma * (1.0 - terminal) * target_q

    bundle.critic.train()
    q1, q2 = bundle.critic(observation, action)
    td1 = q1 - q_target
    td2 = q2 - q_target
    critic_loss = (
        importance
        * (
            F.smooth_l1_loss(q1, q_target, reduction="none")
            + F.smooth_l1_loss(q2, q_target, reduction="none")
        )
    ).mean()
    bundle.critic_optimizer.zero_grad(set_to_none=True)
    critic_loss.backward()
    nn.utils.clip_grad_norm_(bundle.critic.parameters(), config.maximum_gradient_norm)
    bundle.critic_optimizer.step()

    feasibility_logit = bundle.feasibility(observation, action)
    feasibility_loss = F.binary_cross_entropy_with_logits(feasibility_logit, feasible_target)
    bundle.feasibility_optimizer.zero_grad(set_to_none=True)
    feasibility_loss.backward()
    nn.utils.clip_grad_norm_(bundle.feasibility.parameters(), config.maximum_gradient_norm)
    bundle.feasibility_optimizer.step()

    (
        acquisition_progress_model_loss,
        acquisition_progress_model_mask,
        acquisition_progress_model_target,
        acquisition_progress_model_prediction,
    ) = _acquisition_axis_progress_loss_v686(
        bundle.acquisition_progress_model,
        observation=observation,
        next_observation=next_observation,
        executed_action=applied_action,
        exact_home_start=exact_home_start,
        importance=importance,
        config=config,
    )
    if int(acquisition_progress_model_mask.sum().item()) > 0:
        bundle.acquisition_progress_optimizer.zero_grad(set_to_none=True)
        acquisition_progress_model_loss.backward()
        nn.utils.clip_grad_norm_(
            bundle.acquisition_progress_model.parameters(),
            config.maximum_gradient_norm,
        )
        bundle.acquisition_progress_optimizer.step()

    for module in (bundle.critic, bundle.feasibility):
        for parameter in module.parameters():
            parameter.requires_grad_(False)
    policy_action, log_probability = bundle.actor.sample(observation)
    policy_q1, policy_q2 = bundle.critic(observation, policy_action)
    policy_feasibility = bundle.feasibility(observation, policy_action)
    actor_task_loss = (
        config.entropy_temperature * log_probability
        - torch.minimum(policy_q1, policy_q2)
        + config.actor_feasibility_coefficient * F.softplus(-policy_feasibility)
    ).mean()
    requested_applied_gap = torch.linalg.vector_norm(action - applied_action, dim=-1)
    contact_latched = observation[..., -1] >= 0.5
    projection_distillation_mask = (
        (feasible_target >= 0.5)
        & (contact_latched | (acquisition_progress >= 0.5))
        & (requested_applied_gap >= config.actor_projection_distillation_minimum_gap)
    ).float()
    projection_distillation_weight = projection_distillation_mask * importance
    projection_distillation_loss = (
        projection_distillation_weight * torch.square(policy_action - applied_action.detach()).mean(dim=-1)
    ).sum() / projection_distillation_weight.sum().clamp_min(1.0)
    (
        acquisition_self_imitation_loss,
        acquisition_self_imitation_mask,
        acquisition_measured_progress,
        _acquisition_self_imitation_weight,
        _acquisition_absolute_axis_improvement,
        acquisition_self_imitation_candidate_mask,
    ) = _acquisition_self_imitation_loss_v684(
        bundle.actor,
        observation=observation,
        next_observation=next_observation,
        applied_action=applied_action,
        action_feasible=feasible_target,
        exact_home_start=exact_home_start,
        recorded_acquisition_progress=acquisition_progress,
        importance=importance,
        config=config,
    )
    actor_loss = (
        actor_task_loss
        + config.actor_projection_distillation_coefficient * projection_distillation_loss
        + config.actor_acquisition_self_imitation_coefficient * acquisition_self_imitation_loss
    )
    bundle.actor_optimizer.zero_grad(set_to_none=True)
    actor_loss.backward()
    nn.utils.clip_grad_norm_(bundle.actor.parameters(), config.maximum_gradient_norm)
    bundle.actor_optimizer.step()
    for module in (bundle.critic, bundle.feasibility):
        for parameter in module.parameters():
            parameter.requires_grad_(True)

    with torch.no_grad():
        for target_parameter, parameter in zip(
            bundle.target_critic.parameters(),
            bundle.critic.parameters(),
            strict=True,
        ):
            target_parameter.mul_(1.0 - config.target_tau)
            target_parameter.add_(config.target_tau * parameter)
    td_error = torch.maximum(td1.abs(), td2.abs()).detach().cpu().numpy()
    replay.update_priorities(
        batch["indices"],
        (1.0 + td_error).astype(np.float32),
    )
    bundle.update_index += 1
    with torch.no_grad():
        accuracy = ((torch.sigmoid(feasibility_logit) >= 0.5) == (feasible_target >= 0.5)).float().mean()
        acquisition_progress_model_absolute_error = torch.abs(
            acquisition_progress_model_prediction - acquisition_progress_model_target
        )
        acquisition_progress_model_mean_absolute_error = (
            acquisition_progress_model_absolute_error * acquisition_progress_model_mask[:, None]
        ).sum() / (acquisition_progress_model_mask.sum().clamp_min(1.0) * JOINT_ACQUISITION_PROGRESS_DIM_V686)
    return JointGoalSACUpdateMetricsV665(
        update_index=bundle.update_index,
        critic_loss=float(critic_loss.item()),
        actor_loss=float(actor_loss.item()),
        actor_task_loss=float(actor_task_loss.item()),
        actor_projection_distillation_loss=float(projection_distillation_loss.item()),
        projection_distillation_sample_fraction=float(projection_distillation_mask.mean().item()),
        mean_requested_applied_action_gap=float(requested_applied_gap.mean().item()),
        actor_taskframe_projection_l2=float(
            torch.linalg.vector_norm(bundle.actor.taskframe_feature_projection.weight).item()
        ),
        critic_taskframe_projection_l2=float(
            torch.sqrt(
                torch.square(bundle.critic.q1.taskframe_feature_projection.weight).sum()
                + torch.square(bundle.critic.q2.taskframe_feature_projection.weight).sum()
            ).item()
        ),
        feasibility_taskframe_projection_l2=float(
            torch.linalg.vector_norm(bundle.feasibility.taskframe_feature_projection.weight).item()
        ),
        acquisition_policy_sample_fraction=float((observation[..., -1] < 0.5).float().mean().item()),
        actor_acquisition_parameter_l2=float(
            torch.sqrt(
                sum(
                    torch.square(parameter).sum()
                    for name, parameter in bundle.actor.named_parameters()
                    if name.startswith("acquisition_")
                )
            ).item()
        ),
        acquisition_self_imitation_loss=float(acquisition_self_imitation_loss.item()),
        acquisition_self_imitation_sample_fraction=float(acquisition_self_imitation_mask.mean().item()),
        acquisition_self_imitation_mean_progress_m=float(
            (acquisition_measured_progress * acquisition_self_imitation_mask).sum().item()
            / max(float(acquisition_self_imitation_mask.sum().item()), 1.0)
        ),
        acquisition_self_imitation_axis_rejection_fraction=float(
            (acquisition_self_imitation_candidate_mask - acquisition_self_imitation_mask).mean().item()
        ),
        acquisition_progress_model_loss=float(acquisition_progress_model_loss.item()),
        acquisition_progress_model_sample_fraction=float(acquisition_progress_model_mask.mean().item()),
        acquisition_progress_model_mean_absolute_error_m=float(
            acquisition_progress_model_mean_absolute_error.item()
        ),
        feasibility_loss=float(feasibility_loss.item()),
        feasibility_accuracy=float(accuracy.item()),
        mean_q_target=float(q_target.mean().item()),
        mean_q_data=float(torch.minimum(q1, q2).mean().item()),
        mean_policy_action_abs=float(policy_action.abs().mean().item()),
        predicted_policy_feasibility=float(torch.sigmoid(policy_feasibility).mean().item()),
        mean_reward=float(reward.mean().item()),
        maximum_td_error=float(np.max(td_error)),
    )


def joint_goal_checkpoint_payload_v665(
    bundle: JointGoalSACBundleV665,
    config: JointGoalSACConfigV665,
) -> dict[str, Any]:
    config.validate()
    return {
        "format": JOINT_GOAL_SAC_FORMAT_V665,
        "taskframe_feature_format": JOINT_TASKFRAME_FEATURE_FORMAT_V682,
        "acquisition_context_format": JOINT_ACQUISITION_CONTEXT_FORMAT_V683,
        "acquisition_progress_format": JOINT_ACQUISITION_PROGRESS_FORMAT_V686,
        "configuration": asdict(config),
        "observation_dimension": JOINT_GOAL_OBSERVATION_DIM_V665,
        "action_dimension": ARM_JOINT_ACTION_DIM_V664,
        "actor_state_dict": bundle.actor.state_dict(),
        "critic_state_dict": bundle.critic.state_dict(),
        "target_critic_state_dict": bundle.target_critic.state_dict(),
        "feasibility_state_dict": bundle.feasibility.state_dict(),
        "acquisition_progress_state_dict": (bundle.acquisition_progress_model.state_dict()),
        "actor_optimizer_state_dict": bundle.actor_optimizer.state_dict(),
        "critic_optimizer_state_dict": bundle.critic_optimizer.state_dict(),
        "feasibility_optimizer_state_dict": (bundle.feasibility_optimizer.state_dict()),
        "acquisition_progress_optimizer_state_dict": (bundle.acquisition_progress_optimizer.state_dict()),
        "update_index": bundle.update_index,
        "production_admission": False,
    }


__all__ = [
    "JOINT_ACQUISITION_CONTEXT_DIM_V683",
    "JOINT_ACQUISITION_CONTEXT_FORMAT_V683",
    "JOINT_ACQUISITION_PROGRESS_DIM_V686",
    "JOINT_ACQUISITION_PROGRESS_FORMAT_V686",
    "JOINT_GOAL_OBSERVATION_DIM_V665",
    "JOINT_GOAL_REPLAY_FORMAT_V665",
    "JOINT_GOAL_SAC_FORMAT_V665",
    "JOINT_TASKFRAME_FEATURE_DIM_V682",
    "JOINT_TASKFRAME_FEATURE_FORMAT_V682",
    "JointGoalActorV665",
    "JointAcquisitionSelfImitationMetricsV684",
    "JointAcquisitionAxisProgressMetricsV686",
    "JointGoalReplayV665",
    "JointGoalSACBundleV665",
    "JointGoalSACConfigV665",
    "JointGoalSACUpdateMetricsV665",
    "JointTeacherRewardV665",
    "TwinJointGoalCriticV665",
    "TwinJointAcquisitionAxisProgressV686",
    "initialize_joint_goal_sac_v665",
    "joint_goal_checkpoint_payload_v665",
    "joint_goal_observation_v665",
    "joint_acquisition_context_v683",
    "joint_taskframe_features_v682",
    "joint_teacher_reward_v665",
    "pretrain_joint_goal_acquisition_v684",
    "pretrain_joint_acquisition_progress_v686",
    "update_joint_goal_sac_v665",
]
