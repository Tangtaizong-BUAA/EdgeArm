"""Frozen nominal stage environment for the replacement EdgeArm RL line.

The first implemented stage is deliberately narrow: a task-aligned reset,
valid stock-gripper contact, goal-directed transport, and the unchanged
90-control-step (three second) strict hold.  Home acquisition is a separate
stage and is not silently approximated here.

One policy decision is a four-control-step option.  The categorical action is
therefore a genuinely persistent contact mode instead of per-step Gaussian
noise.  The continuous action remains the normalized V688 task-frame
[forward, lateral, vertical] request.  Training uses a cheap deterministic
nominal V10 plant path; exact V4 guarding remains an evaluation/export gate.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, replace
from enum import IntEnum
from typing import Any

import numpy as np

from .hybrid_contact_sac import (
    CONTACT_MODE_NAMES,
    HybridReplayBuffer,
    add_intervention_surrogate,
)
from .sim2real_env_v10 import (
    RealisticEdgeArmEnvV10,
    RealisticEnvV10Config,
    V10SafetyFilterInfeasible,
)
from .stock_gripper_push_face_contact_v22 import (
    configure_stock_gripper_push_face_contact_v22,
    restore_stock_gripper_distal_contact_v22,
)
from .stock_gripper_taskframe_v22 import transition_contact_telemetry_v22
from .taskframe_dls_joint_action_v688 import (
    TaskFrameDLSJointActionConfigV688,
    applied_task_action_from_tool_delta_v688,
    project_taskframe_action_v688,
)
from .train_full_task_joint_sac_v666 import fast_joint_command_v666


STAGED_PUSH_RL_FORMAT = "edgearm-staged-push-rl-1.0.0"
STAGED_PUSH_OBSERVATION_FORMAT = "edgearm-compact-nominal-teacher-state-1.0.0"
STAGED_PUSH_MODE_MASK_START = 40
STAGED_PUSH_OBSERVATION_DIM = 46
DEFAULT_OPTION_CONTROL_STEPS = 4
DIRECT_PRECONTACT_STANDOFF_M = 0.006
DIRECT_PRECONTACT_TOOL_HEIGHT_M = 0.046


class StagedPushStage(IntEnum):
    CONTACT_TRANSPORT_HOLD = 0
    HOME_ACQUISITION = 1
    FULL_EXACT_HOME = 2


@dataclass(frozen=True)
class StagedPushRewardConfig:
    time_penalty: float = 0.004
    action_penalty: float = 0.002
    tracking_residual_penalty: float = 0.03
    precontact_progress_scale_m: float = 0.002
    precontact_progress_reward: float = 0.40
    object_progress_scale_m: float = 0.001
    object_progress_reward: float = 1.25
    contact_acquisition_bonus: float = 0.50
    productive_contact_bonus: float = 0.12
    contact_loss_penalty: float = 0.50
    invalid_contact_penalty: float = 6.0
    coverage_gain_reward: float = 4.0
    hold_gain_reward: float = 6.0
    strict_success_bonus: float = 30.0
    failure_terminal_penalty: float = 8.0
    safety_intervention_penalty: float = 12.0

    def validate(self) -> None:
        values = np.asarray(list(asdict(self).values()), dtype=np.float64)
        if not np.all(np.isfinite(values)) or np.any(values < 0.0):
            raise ValueError("staged push reward configuration is invalid")
        if self.precontact_progress_scale_m <= 0.0 or self.object_progress_scale_m <= 0.0:
            raise ValueError("staged push reward scales must be positive")


@dataclass(frozen=True)
class StagedPushExecutorConfig:
    option_control_steps: int = DEFAULT_OPTION_CONTROL_STEPS
    policy_joint_target_step_rad: float = 0.025
    tracking_residual_alert_l2_threshold: float = 0.75
    workspace_tangent_projection: bool = False
    workspace_inset_m: float = 0.0005

    def validate(self) -> None:
        if type(self.option_control_steps) is not int or self.option_control_steps < 1:
            raise ValueError("option control-step count is invalid")
        values = (
            self.policy_joint_target_step_rad,
            self.tracking_residual_alert_l2_threshold,
        )
        if any(not np.isfinite(value) or value <= 0.0 for value in values):
            raise ValueError("staged push executor configuration is invalid")
        if self.policy_joint_target_step_rad > 0.05:
            raise ValueError("staged push joint target step exceeds deployed support")
        if type(self.workspace_tangent_projection) is not bool:
            raise TypeError("workspace projection switch must be boolean")
        if not np.isfinite(self.workspace_inset_m) or not 0.0 <= self.workspace_inset_m <= 0.002:
            raise ValueError("workspace inset must be between zero and two millimetres")


@dataclass(frozen=True)
class StagedPushStep:
    next_observation: np.ndarray
    reward: float
    terminal: bool
    strict_success: bool
    intervention: bool
    proposed_action: np.ndarray
    projected_action: np.ndarray
    executed_action: np.ndarray
    proposal_projection_l2: float
    projection_tracking_l2: float
    tracking_residual_alert: bool
    analytic_projection_control_steps: int
    command_provenance_observed_steps: int
    command_safety_rewrite_steps: int
    mode: int
    valid_contact_before: bool
    valid_contact_after: bool
    control_steps: int
    valid_contact_steps: int
    invalid_contact_steps: int
    contact_role_names: tuple[str, ...]
    valid_contact_event_count_by_role: tuple[int, ...]
    invalid_contact_event_count_by_role: tuple[int, ...]
    invalid_contact_substep_count_by_role: tuple[int, ...]
    effectful_push_steps: int
    target_coverage: float
    strict_hold_fraction: float
    object_target_distance_m: float
    failure_reason: str


@dataclass(frozen=True)
class _SubmittedActionProjection:
    """One micro-step command plus task-frame projection provenance."""

    joint_command: np.ndarray
    projected_task_action: np.ndarray
    proposal_projection_l2: float
    analytic_projection_applied: bool


def nominal_v10_training_config(*, max_steps: int = 900) -> RealisticEnvV10Config:
    """Return one fixed nominal plant; no DR, loss, latency, noise, or obstacles."""

    if type(max_steps) is not int or max_steps < 128:
        raise ValueError("nominal staged horizon must be at least 128 control steps")
    return RealisticEnvV10Config(
        max_steps=max_steps,
        physics_substeps=8,
        strict_success_hold_steps=90,
        obstacle_probability=0.0,
        failure_recovery_probability=0.0,
        command_delay_steps_range=(0, 0),
        servo_response_range=(1.0, 1.0),
        backlash_range_rad=(0.0, 0.0),
        command_deadband_range_rad=(0.0, 0.0),
        joint_zero_offset_std_rad=0.0,
        joint_zero_offset_limit_rad=0.0,
        encoder_position_noise_std_rad=0.0,
        encoder_velocity_noise_std_rad_s=0.0,
        tracking_noise_std_rad=0.0,
        actuator_kp_scale_range=(1.0, 1.0),
        actuator_kv_scale_range=(1.0, 1.0),
        actuator_force_scale_range=(1.0, 1.0),
        joint_damping_scale_range=(1.0, 1.0),
        joint_armature_scale_range=(1.0, 1.0),
        supply_voltage_range_v=(12.0, 12.0),
        voltage_sag_per_unit_effort_v=0.0,
        ambient_temperature_range_c=(25.0, 25.0),
        initial_motor_temperature_rise_range_c=(0.0, 0.0),
        thermal_derating_start_range_c=(60.0, 60.0),
        thermal_heating_rate_c_s=0.0,
        thermal_cooling_rate_s=0.0,
        command_loss_probability=0.0,
        command_burst_start_probability=0.0,
        block_mass_scale_range=(1.0, 1.0),
        block_com_jitter_range_m=(0.0, 0.0),
        desk_block_slide_friction_range=(0.70, 0.70),
        desk_block_torsional_friction_range=(0.018, 0.018),
        desk_block_rolling_friction_range=(0.002, 0.002),
        pusher_block_slide_friction_range=(1.0, 1.0),
        contact_time_constant_range_s=(0.008, 0.008),
        wrist_payload_mass_range_kg=(0.040, 0.040),
        wrist_payload_com_range_m=(0.050, 0.050),
        camera_mount_rotation_std_rad=0.0,
        camera_mount_rotation_limit_rad=0.0,
        camera_flex_translation_per_rad_s_m=0.0,
    )


def nominal_taskframe_dls_config() -> TaskFrameDLSJointActionConfigV688:
    return TaskFrameDLSJointActionConfigV688(
        orientation_correction_gain=0.03,
        joint_margin_avoidance_v746=False,
        bound_constrained_hierarchical_dls_v755=True,
        directional_face_yaw_v778=True,
    )


def _task_frame_geometry(
    env: RealisticEdgeArmEnvV10,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    block = np.asarray(env.block_xy(), dtype=np.float64)
    target = np.asarray(env.target_xy, dtype=np.float64)
    tool = np.asarray(env.observation()["tool_pose"][:3], dtype=np.float64)
    direction = target - block
    norm = float(np.linalg.norm(direction))
    if not np.isfinite(norm) or norm <= 1.0e-8:
        forward = np.asarray([1.0, 0.0], dtype=np.float64)
    else:
        forward = direction / norm
    lateral = np.asarray([-forward[1], forward[0]], dtype=np.float64)
    desired = np.r_[block - DIRECT_PRECONTACT_STANDOFF_M * forward, DIRECT_PRECONTACT_TOOL_HEIGHT_M]
    error = desired - tool
    local_error = np.asarray(
        [np.dot(error[:2], forward), np.dot(error[:2], lateral), error[2]],
        dtype=np.float64,
    )
    return forward, lateral, tool, local_error


def staged_push_observation(
    env: RealisticEdgeArmEnvV10,
    *,
    stage: StagedPushStage,
    previous_proposed_action: np.ndarray,
    previous_executed_action: np.ndarray,
    valid_contact_latched: bool,
    valid_contact_now: bool,
) -> np.ndarray:
    if type(env) is not RealisticEdgeArmEnvV10:
        raise TypeError("staged push observation requires exact V10 environment")
    if type(stage) is not StagedPushStage:
        raise TypeError("staged push observation stage is invalid")
    proposed = np.asarray(previous_proposed_action, dtype=np.float64)
    executed = np.asarray(previous_executed_action, dtype=np.float64)
    if (
        proposed.shape != (3,)
        or executed.shape != (3,)
        or not np.all(np.isfinite(np.r_[proposed, executed]))
        or type(valid_contact_latched) is not bool
        or type(valid_contact_now) is not bool
    ):
        raise ValueError("staged push observation history is invalid")
    forward, _lateral, tool, local_error = _task_frame_geometry(env)
    block = np.asarray(env.block_xy(), dtype=np.float64)
    target = np.asarray(env.target_xy, dtype=np.float64)
    qpos = np.asarray(env.data.qpos[:6], dtype=np.float64) / np.pi
    maximum_velocity = np.asarray(env.config.max_joint_velocity, dtype=np.float64)
    qvel = np.clip(np.asarray(env.data.qvel[:6], dtype=np.float64) / maximum_velocity, -2.0, 2.0)
    tool_normalized = np.asarray(
        [(tool[0] - 0.25) / 0.20, tool[1] / 0.30, (tool[2] - 0.15) / 0.15],
        dtype=np.float64,
    )
    block_normalized = np.asarray([(block[0] - 0.25) / 0.20, block[1] / 0.30])
    target_normalized = np.asarray([(target[0] - 0.25) / 0.20, target[1] / 0.30])
    error_normalized = np.clip(
        local_error / np.asarray([0.25, 0.20, 0.15], dtype=np.float64),
        -2.0,
        2.0,
    )
    linear_speed, angular_speed = env._block_speeds()
    coverage = float(env.block_target_coverage())
    hold_fraction = min(
        float(env._strict_success_streak) / float(env.config.strict_success_hold_steps),
        1.0,
    )
    stage_one_hot = np.zeros(len(StagedPushStage), dtype=np.float64)
    stage_one_hot[int(stage)] = 1.0
    remaining = max(env.config.max_steps - env.step_count, 0) / env.config.max_steps
    mode_availability = np.zeros(len(CONTACT_MODE_NAMES), dtype=np.float64)
    if coverage >= 0.95:
        mode_availability[CONTACT_MODE_NAMES.index("settle_hold")] = 1.0
    elif valid_contact_latched:
        for name in ("stick_push", "slide_left", "slide_right", "separate_recontact"):
            mode_availability[CONTACT_MODE_NAMES.index(name)] = 1.0
    else:
        # The latch is cleared only by an explicit separate/recontact option,
        # not by one noisy contact-telemetry frame.  Approach then reacquires
        # contact and re-arms the transport regime.
        mode_availability[CONTACT_MODE_NAMES.index("approach")] = 1.0
    result = np.concatenate(
        (
            qpos,
            qvel,
            tool_normalized,
            block_normalized,
            target_normalized,
            error_normalized,
            forward,
            np.asarray([linear_speed / 0.10, angular_speed / 2.0]),
            np.asarray([coverage, hold_fraction]),
            np.asarray([float(valid_contact_now), float(valid_contact_latched)]),
            proposed,
            executed,
            stage_one_hot,
            np.asarray([remaining]),
            mode_availability,
        ),
        dtype=np.float64,
    ).astype(np.float32)
    if result.shape != (STAGED_PUSH_OBSERVATION_DIM,) or not np.all(np.isfinite(result)):
        raise RuntimeError("staged push observation construction failed")
    return result


def _micro_reward(
    *,
    precontact_distance_before: float,
    precontact_distance_after: float,
    object_distance_before: float,
    object_distance_after: float,
    coverage_before: float,
    coverage_after: float,
    hold_before: float,
    hold_after: float,
    contact_before: bool,
    valid_contact: bool,
    invalid_contact: bool,
    block_displacement_m: float,
    proposed_action: np.ndarray,
    projected_action: np.ndarray,
    executed_action: np.ndarray,
    strict_success: bool,
    failure_terminal: bool,
    config: StagedPushRewardConfig,
) -> float:
    acquisition_progress = precontact_distance_before - precontact_distance_after
    object_progress = object_distance_before - object_distance_after
    contact_acquired = valid_contact and not contact_before
    productive = valid_contact and block_displacement_m >= 2.0e-5 and object_progress > 0.0
    contact_lost = contact_before and not valid_contact and coverage_after < 0.95
    reward = -config.time_penalty
    if not contact_before:
        reward += config.precontact_progress_reward * float(
            np.clip(acquisition_progress / config.precontact_progress_scale_m, -1.0, 1.0)
        )
    if contact_before or valid_contact:
        reward += config.object_progress_reward * float(
            np.clip(object_progress / config.object_progress_scale_m, -1.0, 1.0)
        )
        reward += config.coverage_gain_reward * max(coverage_after - coverage_before, 0.0)
        reward += config.hold_gain_reward * max(hold_after - hold_before, 0.0)
    reward += config.contact_acquisition_bonus * float(contact_acquired)
    reward += config.productive_contact_bonus * float(productive)
    reward -= config.contact_loss_penalty * float(contact_lost)
    reward -= config.invalid_contact_penalty * float(invalid_contact)
    reward -= config.action_penalty * float(np.square(proposed_action).sum())
    # DLS projection is part of the declared nominal action map.  Penalize
    # plant tracking error after that projection, not the legitimate
    # proposal-to-projection difference.
    reward -= config.tracking_residual_penalty * float(
        np.linalg.norm(np.asarray(projected_action) - np.asarray(executed_action))
    )
    reward += config.strict_success_bonus * float(strict_success)
    reward -= config.failure_terminal_penalty * float(failure_terminal)
    if not np.isfinite(reward):
        raise RuntimeError("staged push reward became non-finite")
    return float(reward)


class StagedPushEpisode:
    """One nominal staged episode with four-step hybrid actions."""

    def __init__(
        self,
        *,
        seed: int,
        environment_config: RealisticEnvV10Config | None = None,
        dls_config: TaskFrameDLSJointActionConfigV688 | None = None,
        executor_config: StagedPushExecutorConfig | None = None,
        reward_config: StagedPushRewardConfig | None = None,
        scene_mode: str = "single",
    ) -> None:
        if type(seed) is not int or seed < 0:
            raise ValueError("staged push seed is invalid")
        self.environment_config = environment_config or nominal_v10_training_config()
        if scene_mode not in ("single", "multichoice_v1"):
            raise ValueError("unknown staged scene mode")
        self.scene_mode = scene_mode
        if scene_mode == "multichoice_v1":
            self.environment_config = replace(self.environment_config, multichoice_blocks=True)
        self.dls_config = dls_config or nominal_taskframe_dls_config()
        self.executor_config = executor_config or StagedPushExecutorConfig()
        self.reward_config = reward_config or StagedPushRewardConfig()
        self.executor_config.validate()
        self.reward_config.validate()
        self.env = RealisticEdgeArmEnvV10(self.environment_config, seed=seed)
        self.multichoice = None
        if scene_mode == "multichoice_v1":
            from .multichoice_scene_v1 import MultiChoiceScene
            self.multichoice = MultiChoiceScene(self.env)
        self.stage = StagedPushStage.CONTACT_TRANSPORT_HOLD
        self.previous_proposed_action = np.zeros(3, dtype=np.float32)
        self.previous_executed_action = np.zeros(3, dtype=np.float32)
        self.valid_contact_latched = False
        self.valid_contact_now = False
        self.terminal = False

    def reset(self, *, seed: int, stage: StagedPushStage) -> np.ndarray:
        if type(seed) is not int or seed < 0 or type(stage) is not StagedPushStage:
            raise ValueError("staged push reset arguments are invalid")
        if stage is not StagedPushStage.CONTACT_TRANSPORT_HOLD:
            raise NotImplementedError(
                "Home acquisition is a separate acceptance stage and is not implemented by nominal stage one"
            )
        restore_stock_gripper_distal_contact_v22(self.env)
        if self.multichoice is not None:
            self.multichoice.prepare(seed)
        self.env.reset(seed=seed, obstacle=False, stress=False)
        configure_stock_gripper_push_face_contact_v22(self.env)
        if self.multichoice is not None:
            self.multichoice.install()
        self.stage = stage
        self.previous_proposed_action.fill(0.0)
        self.previous_executed_action.fill(0.0)
        self.valid_contact_latched = False
        self.valid_contact_now = False
        self.terminal = False
        return self.observation()

    def observation(self) -> np.ndarray:
        return staged_push_observation(
            self.env,
            stage=self.stage,
            previous_proposed_action=self.previous_proposed_action,
            previous_executed_action=self.previous_executed_action,
            valid_contact_latched=self.valid_contact_latched,
            valid_contact_now=self.valid_contact_now,
        )

    def _submitted_action(
        self,
        proposed_action: np.ndarray,
        mode: int,
    ) -> _SubmittedActionProjection:
        # settle_hold is a discrete low-level option: it holds the live physical
        # target and intentionally ignores the residual.  This makes a 90-step
        # settle achievable without V688's orientation null-space correction.
        if CONTACT_MODE_NAMES[mode] == "settle_hold" and self.env.block_target_coverage() >= 0.90:
            reported = np.asarray(self.env._command_reference_reported_position(), dtype=np.float64)
            submitted = np.zeros(6, dtype=np.float32)
            submitted[5] = np.clip(
                (
                    self.env.tool_gripper_joint_position_rad
                    - (reported[5] - self.env._zero_offset[5])
                )
                / self.env.config.max_joint_delta,
                -1.0,
                1.0,
            )
            projected_task_action = np.zeros(3, dtype=np.float32)
            return _SubmittedActionProjection(
                joint_command=submitted,
                projected_task_action=projected_task_action,
                proposal_projection_l2=float(
                    np.linalg.norm(
                        np.asarray(proposed_action, dtype=np.float64)
                        - projected_task_action.astype(np.float64)
                    )
                ),
                analytic_projection_applied=False,
            )
        feasible_action = np.asarray(proposed_action, dtype=np.float64).copy()
        if self.executor_config.workspace_tangent_projection and CONTACT_MODE_NAMES[mode] == "approach":
            # Project Cartesian components separately before IK.  The plant's
            # unchanged joint/workspace filter remains the final authority.
            # This constrains the existing task frame, not a task waypoint.
            forward, lateral, tool, _ = _task_frame_geometry(self.env)
            basis = np.asarray([
                [forward[0], lateral[0], 0.0],
                [forward[1], lateral[1], 0.0],
                [0.0, 0.0, 1.0],
            ])
            action_scale = np.asarray([
                self.dls_config.forward_translation_step_m,
                self.dls_config.lateral_translation_step_m,
                self.dls_config.vertical_translation_step_m,
            ])
            bounds = np.asarray([
                self.env.config.workspace_x,
                self.env.config.workspace_y,
                self.env.config.workspace_z,
            ])
            inset = self.executor_config.workspace_inset_m
            desired = np.clip(
                tool + basis @ (feasible_action * action_scale),
                bounds[:, 0] + inset,
                bounds[:, 1] - inset,
            )
            feasible_action = np.clip(basis.T @ (desired - tool) / action_scale, -1.0, 1.0)
        projected = project_taskframe_action_v688(self.env, feasible_action, self.dls_config)
        reported = np.asarray(self.env._command_reference_reported_position(), dtype=np.float64)
        scale = np.asarray(
            [
                self.dls_config.forward_translation_step_m,
                self.dls_config.lateral_translation_step_m,
                self.dls_config.vertical_translation_step_m,
            ],
            dtype=np.float64,
        )
        return _SubmittedActionProjection(
            joint_command=fast_joint_command_v666(
                policy_action=projected.projected_joint_action,
                reported_joint_position_rad=reported,
                zero_offset_rad=np.asarray(self.env._zero_offset),
                fixed_gripper_joint_position_rad=self.env.tool_gripper_joint_position_rad,
                plant_max_joint_delta_rad=self.env.config.max_joint_delta,
                policy_joint_target_step_rad=self.executor_config.policy_joint_target_step_rad,
            ),
            projected_task_action=(
                np.asarray(projected.predicted_local_delta_m, dtype=np.float64) / scale
            ).astype(np.float32),
            proposal_projection_l2=float(np.linalg.norm(
                np.asarray(proposed_action, dtype=np.float64)
                - np.asarray(projected.predicted_local_delta_m, dtype=np.float64) / scale
            )),
            analytic_projection_applied=bool(
                projected.intervened or not np.allclose(feasible_action, proposed_action, atol=1e-7, rtol=0.0)
            ),
        )

    def step_option(self, proposed_action: np.ndarray, mode: int) -> StagedPushStep:
        action = np.asarray(proposed_action, dtype=np.float32)
        if self.terminal:
            raise RuntimeError("cannot step a terminal staged episode")
        if (
            action.shape != (3,)
            or not np.all(np.isfinite(action))
            or np.any(np.abs(action) > 1.0 + 1.0e-6)
            or type(mode) is not int
            or not 0 <= mode < len(CONTACT_MODE_NAMES)
        ):
            raise ValueError("staged option action is invalid")
        action = np.clip(action, -1.0, 1.0).astype(np.float32)
        accumulated_reward = 0.0
        projected_actions: list[np.ndarray] = []
        applied_actions: list[np.ndarray] = []
        proposal_projection_l2_values: list[float] = []
        projection_tracking_l2_values: list[float] = []
        analytic_projection_steps = 0
        command_provenance_observed_steps = 0
        command_safety_rewrite_steps = 0
        valid_steps = 0
        invalid_steps = 0
        effectful_steps = 0
        contact_role_names: tuple[str, ...] = ()
        valid_contact_events_by_role = np.zeros(0, dtype=np.int64)
        invalid_contact_events_by_role = np.zeros(0, dtype=np.int64)
        invalid_contact_substeps_by_role = np.zeros(0, dtype=np.int64)
        intervention = False
        strict_success = False
        failure_reason = "nonterminal"
        executed_control_steps = 0
        valid_contact_before = bool(self.valid_contact_latched)
        for _ in range(self.executor_config.option_control_steps):
            block_before = np.asarray(self.env.block_xy(), dtype=np.float64).copy()
            tool_before = np.asarray(self.env.observation()["tool_pose"][:3], dtype=np.float64)
            forward, lateral, _tool, local_error_before = _task_frame_geometry(self.env)
            precontact_before = float(np.linalg.norm(local_error_before))
            object_before = float(self.env.distance_to_target())
            coverage_before = float(self.env.block_target_coverage())
            hold_before = min(
                float(self.env._strict_success_streak) / self.env.config.strict_success_hold_steps,
                1.0,
            )
            contact_before = self.valid_contact_latched
            try:
                submission = self._submitted_action(action, mode)
                _observation, _environment_reward, terminated, truncated, info = self.env.step(
                    submission.joint_command
                )
            except V10SafetyFilterInfeasible as error:
                intervention = True
                failure_reason = f"v10_safety_filter:{type(error).__name__}"
                break
            executed_control_steps += 1
            tool_after = np.asarray(self.env.observation()["tool_pose"][:3], dtype=np.float64)
            applied = applied_task_action_from_tool_delta_v688(
                tool_after - tool_before,
                forward_xy=forward,
                lateral_xy=lateral,
                translation_scale_xyz_m=np.asarray(
                    [
                        self.dls_config.forward_translation_step_m,
                        self.dls_config.lateral_translation_step_m,
                        self.dls_config.vertical_translation_step_m,
                    ]
                ),
            )
            projected_actions.append(submission.projected_task_action)
            applied_actions.append(applied)
            proposal_projection_l2_values.append(submission.proposal_projection_l2)
            projection_tracking_l2_values.append(
                float(
                    np.linalg.norm(
                        np.asarray(submission.projected_task_action, dtype=np.float64)
                        - np.asarray(applied, dtype=np.float64)
                    )
                )
            )
            analytic_projection_steps += int(submission.analytic_projection_applied)
            command_feedback = self.env.last_command_feedback_v1
            command_provenance_observed = bool(
                command_feedback.get("available", False)
                and not command_feedback.get("applied_is_virtual_hold", True)
                and int(command_feedback.get("submitted_id", -1))
                == int(command_feedback.get("applied_id", -2))
                and "applied_action_was_safety_modified" in command_feedback
            )
            command_provenance_observed_steps += int(command_provenance_observed)
            if command_provenance_observed:
                command_safety_rewrite_steps += int(
                    bool(command_feedback["applied_action_was_safety_modified"])
                )
            telemetry = transition_contact_telemetry_v22(
                info,
                block_before_xy_m=block_before,
                block_after_xy_m=self.env.block_xy(),
            )
            valid_contact = bool(telemetry["valid_push_side_contact_any"])
            invalid_contact = bool(telemetry["invalid_tool_block_contact_any"])
            current_roles = tuple(str(value) for value in telemetry["contact_role_names"])
            current_valid_by_role = np.asarray(
                telemetry["valid_push_side_contact_count_by_role"], dtype=np.int64
            )
            current_invalid_by_role = np.asarray(
                telemetry["invalid_tool_block_contact_count_by_role"], dtype=np.int64
            )
            current_invalid_substeps_by_role = np.asarray(
                telemetry["invalid_tool_block_contact_substep_count_by_role"],
                dtype=np.int64,
            )
            if not contact_role_names:
                contact_role_names = current_roles
                valid_contact_events_by_role = np.zeros(len(current_roles), dtype=np.int64)
                invalid_contact_events_by_role = np.zeros(len(current_roles), dtype=np.int64)
                invalid_contact_substeps_by_role = np.zeros(len(current_roles), dtype=np.int64)
            if (
                current_roles != contact_role_names
                or current_valid_by_role.shape != valid_contact_events_by_role.shape
                or current_invalid_by_role.shape != invalid_contact_events_by_role.shape
                or current_invalid_substeps_by_role.shape
                != invalid_contact_substeps_by_role.shape
            ):
                raise RuntimeError("staged push contact-role identity changed")
            valid_contact_events_by_role += current_valid_by_role
            invalid_contact_events_by_role += current_invalid_by_role
            invalid_contact_substeps_by_role += current_invalid_substeps_by_role
            self.valid_contact_now = valid_contact
            self.valid_contact_latched = self.valid_contact_latched or valid_contact
            valid_steps += int(valid_contact)
            invalid_steps += int(invalid_contact)
            effectful_steps += int(
                valid_contact
                and float(telemetry["step_block_displacement_m"]) >= 2.0e-5
            )
            _forward_after, _lateral_after, _tool_after, local_error_after = _task_frame_geometry(
                self.env
            )
            coverage_after = float(self.env.block_target_coverage())
            hold_after = min(
                float(self.env._strict_success_streak) / self.env.config.strict_success_hold_steps,
                1.0,
            )
            strict_success = bool(info.get("success", False))
            failure_terminal = bool(info.get("terminal_failure", False))
            accumulated_reward += _micro_reward(
                precontact_distance_before=precontact_before,
                precontact_distance_after=float(np.linalg.norm(local_error_after)),
                object_distance_before=object_before,
                object_distance_after=float(self.env.distance_to_target()),
                coverage_before=coverage_before,
                coverage_after=coverage_after,
                hold_before=hold_before,
                hold_after=hold_after,
                contact_before=contact_before,
                valid_contact=valid_contact,
                invalid_contact=invalid_contact,
                block_displacement_m=float(telemetry["step_block_displacement_m"]),
                proposed_action=action,
                projected_action=submission.projected_task_action,
                executed_action=applied,
                strict_success=strict_success,
                failure_terminal=failure_terminal,
                config=self.reward_config,
            )
            # ``workspace_scaled`` is the declared cheap analytic constraint
            # of the nominal plant, not a backup-policy takeover.  Treating it
            # as a terminal shield intervention recreates the old 98%-filtered
            # replay pathology.  Only a failed V10 transaction (caught above)
            # or an environment terminal failure is a surrogate-MDP event.
            if self.multichoice is not None and self.multichoice.failed:
                intervention = True
                strict_success = False
                failure_reason = "unselected_object_contact_or_motion"
                break
            if terminated or truncated:
                failure_reason = str(info.get("terminal_reason", "terminal"))
                break
        if projected_actions and applied_actions:
            projected_action = np.mean(np.stack(projected_actions), axis=0).astype(np.float32)
            executed = np.mean(np.stack(applied_actions), axis=0).astype(np.float32)
            proposal_projection_l2 = float(np.mean(proposal_projection_l2_values))
            projection_tracking_l2 = float(np.mean(projection_tracking_l2_values))
        else:
            projected_action = np.zeros(3, dtype=np.float32)
            executed = np.zeros(3, dtype=np.float32)
            proposal_projection_l2 = 0.0
            projection_tracking_l2 = 0.0
        if intervention:
            accumulated_reward = -self.reward_config.safety_intervention_penalty
        self.previous_proposed_action = action.copy()
        self.previous_executed_action = executed.copy()
        valid_contact_after = bool(self.valid_contact_now)
        if CONTACT_MODE_NAMES[mode] == "separate_recontact" and not intervention:
            # This categorical option is phase one of an explicit option
            # chain.  Clearing the regime latch makes the next observation
            # approach-only until contact is physically reacquired.
            self.valid_contact_latched = False
        self.terminal = bool(
            intervention
            or strict_success
            or self.env.step_count >= self.env.config.max_steps
            or failure_reason not in ("nonterminal", "")
            and failure_reason != "nonterminal"
        )
        # A normal option that used all micro-steps is nonterminal even though
        # failure_reason retained its default string.
        if not intervention and not strict_success and self.env.step_count < self.env.config.max_steps:
            self.terminal = failure_reason not in ("nonterminal", "")
        next_observation = self.observation()
        return StagedPushStep(
            next_observation=next_observation,
            reward=float(accumulated_reward),
            terminal=self.terminal,
            strict_success=strict_success,
            intervention=intervention,
            proposed_action=action.copy(),
            projected_action=projected_action,
            executed_action=executed,
            proposal_projection_l2=proposal_projection_l2,
            projection_tracking_l2=projection_tracking_l2,
            tracking_residual_alert=bool(
                projection_tracking_l2
                > self.executor_config.tracking_residual_alert_l2_threshold
            ),
            analytic_projection_control_steps=analytic_projection_steps,
            command_provenance_observed_steps=command_provenance_observed_steps,
            command_safety_rewrite_steps=command_safety_rewrite_steps,
            mode=mode,
            valid_contact_before=valid_contact_before,
            valid_contact_after=valid_contact_after,
            control_steps=executed_control_steps,
            valid_contact_steps=valid_steps,
            invalid_contact_steps=invalid_steps,
            contact_role_names=contact_role_names,
            valid_contact_event_count_by_role=tuple(
                int(value) for value in valid_contact_events_by_role
            ),
            invalid_contact_event_count_by_role=tuple(
                int(value) for value in invalid_contact_events_by_role
            ),
            invalid_contact_substep_count_by_role=tuple(
                int(value) for value in invalid_contact_substeps_by_role
            ),
            effectful_push_steps=effectful_steps,
            target_coverage=float(self.env.block_target_coverage()),
            strict_hold_fraction=min(
                float(self.env._strict_success_streak) / self.env.config.strict_success_hold_steps,
                1.0,
            ),
            object_target_distance_m=float(self.env.distance_to_target()),
            failure_reason=failure_reason,
        )


class ScriptedSuccessfulPriorPolicy:
    """Geometry-only prior collector for stage one, never a production policy."""

    def __call__(self, episode: StagedPushEpisode) -> tuple[int, np.ndarray]:
        env = episode.env
        forward, lateral, tool, local_error = _task_frame_geometry(env)
        coverage = float(env.block_target_coverage())
        if coverage >= 0.95:
            return CONTACT_MODE_NAMES.index("settle_hold"), np.zeros(3, dtype=np.float32)
        if not episode.valid_contact_latched:
            action = np.asarray(
                [
                    local_error[0] / 0.0015,
                    local_error[1] / 0.0010,
                    local_error[2] / 0.0040,
                ],
                dtype=np.float32,
            )
            return CONTACT_MODE_NAMES.index("approach"), np.clip(action, -1.0, 1.0)
        lateral_offset = float(np.dot(tool[:2] - env.block_xy(), lateral))
        action = np.asarray(
            [
                0.80,
                np.clip(-lateral_offset / 0.006, -0.50, 0.50),
                np.clip((DIRECT_PRECONTACT_TOOL_HEIGHT_M - tool[2]) / 0.008, -0.30, 0.30),
            ],
            dtype=np.float32,
        )
        return CONTACT_MODE_NAMES.index("stick_push"), action


def collect_successful_prior_episode(
    replay: HybridReplayBuffer,
    *,
    seed: int,
    stage: StagedPushStage = StagedPushStage.CONTACT_TRANSPORT_HOLD,
    maximum_options: int = 300,
    scene_mode: str = "single",
) -> dict[str, Any]:
    """Collect one episode atomically; only strict success enters prior replay."""

    if replay.frozen:
        raise RuntimeError("cannot collect into a frozen prior replay")
    if type(maximum_options) is not int or maximum_options < 1:
        raise ValueError("maximum prior options is invalid")
    episode = StagedPushEpisode(seed=seed, scene_mode=scene_mode)
    observation = episode.reset(seed=seed, stage=stage)
    initial_distance = float(episode.env.distance_to_target())
    initial_coverage = float(episode.env.block_target_coverage())
    policy = ScriptedSuccessfulPriorPolicy()
    pending: list[tuple[np.ndarray, StagedPushStep]] = []
    total_valid = 0
    total_invalid = 0
    total_effectful = 0
    valid_contact_events_by_role: Counter[str] = Counter()
    invalid_contact_events_by_role: Counter[str] = Counter()
    invalid_contact_substeps_by_role: Counter[str] = Counter()
    maximum_coverage = initial_coverage
    for _ in range(maximum_options):
        mode, proposed_action = policy(episode)
        step = episode.step_option(proposed_action, mode)
        pending.append((observation.copy(), step))
        observation = step.next_observation
        total_valid += step.valid_contact_steps
        total_invalid += step.invalid_contact_steps
        total_effectful += step.effectful_push_steps
        for role, count in zip(
            step.contact_role_names,
            step.valid_contact_event_count_by_role,
            strict=True,
        ):
            valid_contact_events_by_role[role] += count
        for role, event_count, substep_count in zip(
            step.contact_role_names,
            step.invalid_contact_event_count_by_role,
            step.invalid_contact_substep_count_by_role,
            strict=True,
        ):
            invalid_contact_events_by_role[role] += event_count
            invalid_contact_substeps_by_role[role] += substep_count
        maximum_coverage = max(maximum_coverage, step.target_coverage)
        if step.terminal:
            break
    strict_success = bool(pending and pending[-1][1].strict_success)
    if strict_success and total_invalid == 0 and initial_coverage < 0.95:
        for before, step in pending:
            if step.intervention:
                add_intervention_surrogate(
                    replay,
                    observation=before,
                    next_observation=step.next_observation,
                    proposed_action=step.proposed_action,
                    executed_action=step.executed_action,
                    mode=step.mode,
                    stage=int(stage),
                    penalty=episode.reward_config.safety_intervention_penalty,
                )
            else:
                replay.add(
                    observation=before,
                    next_observation=step.next_observation,
                    proposed_action=step.proposed_action,
                    projected_action=step.projected_action,
                    executed_action=step.executed_action,
                    mode=step.mode,
                    reward=step.reward,
                    terminal=step.terminal,
                    intervention=False,
                    projection_provenance_observed=True,
                    proposal_projection_l2=step.proposal_projection_l2,
                    projection_tracking_l2=step.projection_tracking_l2,
                    analytic_projection_applied=(
                        step.analytic_projection_control_steps > 0
                    ),
                    command_provenance_observed=(
                        step.command_provenance_observed_steps == step.control_steps
                    ),
                    command_safety_rewrite=(step.command_safety_rewrite_steps > 0),
                    strict_success=step.strict_success,
                    stage=int(stage),
                )
    return {
        "format": STAGED_PUSH_RL_FORMAT,
        "seed": seed,
        "scene_mode": scene_mode,
        "multichoice_scene": episode.multichoice.audit() if episode.multichoice is not None else None,
        "stage": stage.name,
        "strict_success": strict_success,
        "episode_admitted_to_successful_prior": bool(
            strict_success and total_invalid == 0 and initial_coverage < 0.95
        ),
        "option_count": len(pending),
        "control_step_count": int(sum(step.control_steps for _, step in pending)),
        "initial_object_target_distance_m": initial_distance,
        "initial_target_coverage": initial_coverage,
        "maximum_target_coverage": maximum_coverage,
        "final_target_coverage": float(episode.env.block_target_coverage()),
        "final_hold_fraction": min(
            float(episode.env._strict_success_streak) / episode.env.config.strict_success_hold_steps,
            1.0,
        ),
        "valid_contact_control_steps": total_valid,
        "invalid_contact_control_steps": total_invalid,
        "effectful_push_control_steps": total_effectful,
        "valid_contact_event_count_by_role": dict(valid_contact_events_by_role),
        "invalid_contact_event_count_by_role": dict(invalid_contact_events_by_role),
        "invalid_contact_substep_count_by_role": dict(invalid_contact_substeps_by_role),
        "terminal_reason": pending[-1][1].failure_reason if pending else "no_transition",
        "production_admission": False,
        "wrist_multimodal_export_started": False,
        "act_training_started": False,
    }


__all__ = [
    "DEFAULT_OPTION_CONTROL_STEPS",
    "DIRECT_PRECONTACT_STANDOFF_M",
    "DIRECT_PRECONTACT_TOOL_HEIGHT_M",
    "STAGED_PUSH_OBSERVATION_DIM",
    "STAGED_PUSH_OBSERVATION_FORMAT",
    "STAGED_PUSH_MODE_MASK_START",
    "STAGED_PUSH_RL_FORMAT",
    "ScriptedSuccessfulPriorPolicy",
    "StagedPushEpisode",
    "StagedPushExecutorConfig",
    "StagedPushRewardConfig",
    "StagedPushStage",
    "StagedPushStep",
    "collect_successful_prior_episode",
    "nominal_taskframe_dls_config",
    "nominal_v10_training_config",
    "staged_push_observation",
]
