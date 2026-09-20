"""Versioned sim-to-real dynamics and wrist-camera transport for EdgeArm.

The production v1 environment intentionally keeps its historical ideal joint
teleportation contract because existing datasets and checkpoints depend on it.
This module is an explicit v2 opt-in.  It adds a deterministic, episode-seeded
surrogate for affordable position servos and a wrist-only camera transport
model without changing :mod:`edgearm.production_env` or
:mod:`edgearm.multimodal` defaults.
"""

from __future__ import annotations

from collections import deque
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import mujoco
import numpy as np

from .multimodal import CameraCaptureConfig, TrueMultimodalRenderer
from .production_env import ProductionEdgeArmEnv, ProductionEnvConfig


SIM2REAL_PROFILE_VERSION = "edgearm-sim2real-v2"
SIM2REAL_CAMERA_PROFILE_VERSION = "edgearm-wrist-camera-sim2real-v2"
COMMAND_FEEDBACK_PROFILE_VERSION = "edgearm-command-feedback-v1"
_JOINTS = 6


class _QueuedCommand(np.ndarray):
    """Array-compatible delayed command with causal submission metadata.

    Several existing policy-state encoders intentionally inspect
    ``env._command_queue`` and treat every entry as a joint-target ndarray.
    Keeping the target as an ndarray subclass preserves that contract while
    attaching the command identity and timing needed to audit which submitted
    action actually reached the delayed controller.
    """

    command_id: int
    original_action: np.ndarray
    submitted_action: np.ndarray
    send_step: int
    send_time_seconds: float
    submission_safety_reason: str
    submission_target_changed_mask: np.ndarray
    ingress_lost: bool
    post_submission_rewrite_count: int
    post_submission_target_changed_mask: np.ndarray
    post_submission_rewrite_reasons: tuple[str, ...]

    def __new__(
        cls,
        target: np.ndarray,
        *,
        command_id: int,
        original_action: np.ndarray,
        submitted_action: np.ndarray,
        send_step: int,
        send_time_seconds: float,
        submission_safety_reason: str = "",
        submission_target_changed_mask: np.ndarray | None = None,
        ingress_lost: bool = False,
    ) -> _QueuedCommand:
        value = np.asarray(target, dtype=np.float64).copy().view(cls)
        value.command_id = int(command_id)
        value.original_action = np.asarray(original_action, dtype=np.float64).copy()
        value.submitted_action = np.asarray(submitted_action, dtype=np.float64).copy()
        value.send_step = int(send_step)
        value.send_time_seconds = float(send_time_seconds)
        value.submission_safety_reason = str(submission_safety_reason)
        value.ingress_lost = bool(ingress_lost)
        if submission_target_changed_mask is None:
            submission_target_changed_mask = np.zeros(_JOINTS, dtype=np.bool_)
        value.submission_target_changed_mask = np.asarray(
            submission_target_changed_mask, dtype=np.bool_
        ).copy()
        value.post_submission_rewrite_count = 0
        value.post_submission_target_changed_mask = np.zeros(
            _JOINTS, dtype=np.bool_
        )
        value.post_submission_rewrite_reasons = ()
        return value

    def __array_finalize__(self, source: np.ndarray | None) -> None:
        if source is None:
            return
        self.command_id = int(getattr(source, "command_id", -1))
        self.original_action = np.asarray(
            getattr(source, "original_action", np.zeros(_JOINTS)), dtype=np.float64
        ).copy()
        self.submitted_action = np.asarray(
            getattr(source, "submitted_action", np.zeros(_JOINTS)), dtype=np.float64
        ).copy()
        self.send_step = int(getattr(source, "send_step", -1))
        self.send_time_seconds = float(getattr(source, "send_time_seconds", 0.0))
        self.submission_safety_reason = str(getattr(source, "submission_safety_reason", ""))
        self.ingress_lost = bool(getattr(source, "ingress_lost", False))
        self.submission_target_changed_mask = np.asarray(
            getattr(
                source,
                "submission_target_changed_mask",
                np.zeros(_JOINTS, dtype=np.bool_),
            ),
            dtype=np.bool_,
        ).copy()
        self.post_submission_rewrite_count = int(
            getattr(source, "post_submission_rewrite_count", 0)
        )
        self.post_submission_target_changed_mask = np.asarray(
            getattr(
                source,
                "post_submission_target_changed_mask",
                np.zeros(_JOINTS, dtype=np.bool_),
            ),
            dtype=np.bool_,
        ).copy()
        self.post_submission_rewrite_reasons = tuple(
            str(value)
            for value in getattr(source, "post_submission_rewrite_reasons", ())
        )

    def rewrite_target(self, target: np.ndarray, *, reason: str) -> np.ndarray:
        """Apply an explicit post-submission safety transform with provenance."""

        target = np.asarray(target, dtype=np.float64)
        if target.shape != (_JOINTS,) or not np.all(np.isfinite(target)):
            raise ValueError("rewritten queued target must be a finite six-vector")
        changed = ~np.isclose(self, target, rtol=0.0, atol=1.0e-12)
        self[:] = target
        if np.any(changed):
            self.post_submission_rewrite_count += 1
            self.post_submission_target_changed_mask = np.logical_or(
                self.post_submission_target_changed_mask,
                changed,
            )
            self.post_submission_rewrite_reasons = (
                *self.post_submission_rewrite_reasons,
                str(reason),
            )
        return changed.astype(np.uint8)


def _validate_range(name: str, value: tuple[float, float], *, nonnegative: bool = False) -> None:
    if len(value) != 2 or not np.all(np.isfinite(value)) or value[0] > value[1]:
        raise ValueError(f"{name} must be a finite ordered (low, high) pair")
    if nonnegative and value[0] < 0:
        raise ValueError(f"{name} must be non-negative")


def _validate_joint_vector(name: str, value: tuple[float, ...]) -> None:
    if len(value) != _JOINTS or not np.all(np.isfinite(value)) or np.any(np.asarray(value) <= 0):
        raise ValueError(f"{name} must contain six finite positive values")


@dataclass(frozen=True)
class Sim2RealEnvConfig(ProductionEnvConfig):
    """Episode-randomized actuator and encoder profile for the v2 environment.

    Ranges are sampled from a generator derived only from the reset seed, so a
    task can be replayed bit-for-bit with the same config, seed, and actions.
    Velocity and acceleration limits are expressed in rad/s and rad/s^2.
    """

    max_steps: int = 64
    command_delay_steps_range: tuple[int, int] = (0, 3)
    servo_response_range: tuple[float, float] = (0.72, 0.95)
    max_joint_velocity: tuple[float, ...] = (1.80, 1.70, 1.70, 2.00, 2.20, 1.80)
    max_joint_acceleration: tuple[float, ...] = (10.0, 10.0, 10.0, 11.0, 12.0, 10.0)
    backlash_range_rad: tuple[float, float] = (0.0004, 0.0030)
    command_deadband_range_rad: tuple[float, float] = (0.0002, 0.0010)
    joint_zero_offset_std_rad: float = 0.0040
    joint_zero_offset_limit_rad: float = 0.0160
    encoder_position_noise_std_rad: float = 0.0008
    encoder_velocity_noise_std_rad_s: float = 0.0060
    tracking_noise_std_rad: float = 0.00020

    def __post_init__(self) -> None:
        if self.fps <= 0 or self.physics_substeps <= 0 or self.max_steps <= 0:
            raise ValueError("fps, physics_substeps, and max_steps must be positive")
        delay = self.command_delay_steps_range
        if (
            len(delay) != 2
            or not all(isinstance(value, int) for value in delay)
            or delay[0] < 0
            or delay[0] > delay[1]
        ):
            raise ValueError("command_delay_steps_range must be an ordered non-negative pair")
        _validate_range("servo_response_range", self.servo_response_range, nonnegative=True)
        if self.servo_response_range[1] > 1.0:
            raise ValueError("servo_response_range cannot exceed 1.0")
        _validate_range("backlash_range_rad", self.backlash_range_rad, nonnegative=True)
        _validate_range("command_deadband_range_rad", self.command_deadband_range_rad, nonnegative=True)
        _validate_joint_vector("max_joint_velocity", self.max_joint_velocity)
        _validate_joint_vector("max_joint_acceleration", self.max_joint_acceleration)
        for name in (
            "joint_zero_offset_std_rad",
            "joint_zero_offset_limit_rad",
            "encoder_position_noise_std_rad",
            "encoder_velocity_noise_std_rad_s",
            "tracking_noise_std_rad",
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")


class Sim2RealEdgeArmEnv(ProductionEdgeArmEnv):
    """Opt-in v2 environment with delayed, rate-limited servo tracking."""

    profile_version = SIM2REAL_PROFILE_VERSION

    def __init__(self, config: Sim2RealEnvConfig | None = None, seed: int = 0):
        if config is not None and not isinstance(config, Sim2RealEnvConfig):
            raise TypeError("Sim2RealEdgeArmEnv requires Sim2RealEnvConfig")
        self.sim2real_config = config or Sim2RealEnvConfig()
        self._sim2real_ready = False
        self._episode_rng = np.random.default_rng(seed ^ 0x51A2EA1)
        self._command_queue: deque[_QueuedCommand] = deque()
        self._command_delay_steps = 0
        self._servo_response = 1.0
        self._backlash = np.zeros(_JOINTS, dtype=np.float64)
        self._deadband = np.zeros(_JOINTS, dtype=np.float64)
        self._zero_offset = np.zeros(_JOINTS, dtype=np.float64)
        self._encoder_position_noise = np.zeros(_JOINTS, dtype=np.float64)
        self._encoder_velocity_noise = np.zeros(_JOINTS, dtype=np.float64)
        self._servo_velocity = np.zeros(_JOINTS, dtype=np.float64)
        self._last_motor_direction = np.zeros(_JOINTS, dtype=np.int8)
        self._backlash_remaining = np.zeros(_JOINTS, dtype=np.float64)
        self._submission_ingress_lost = False
        self._command_epoch = 0
        self._last_command_feedback_v1: dict[str, Any] = {}
        self.episode_sim2real: dict[str, Any] = {}
        super().__init__(self.sim2real_config, seed=seed)
        # Formal datasets timestamp policy steps at 1/fps.  Keep the MuJoCo
        # clock on that same contract by dividing the control period evenly
        # across physics substeps.  Each environment owns its model, so this
        # explicit v2 change cannot affect a v1 ProductionEdgeArmEnv instance.
        self.model.opt.timestep = 1.0 / (self.config.fps * self.config.physics_substeps)
        self._last_command_feedback_v1 = self._unavailable_command_feedback_v1()

    @property
    def control_dt(self) -> float:
        """Physical duration advanced by one policy step."""

        return float(self.model.opt.timestep * self.config.physics_substeps)

    @property
    def command_delay_steps(self) -> int:
        return self._command_delay_steps

    @property
    def command_epoch(self) -> int:
        """Monotonic identity of the currently active command stream."""

        return int(self._command_epoch)

    @property
    def next_command_id(self) -> int:
        """Identity that the next successful command submission will receive."""

        return int(self.step_count)

    @property
    def last_command_feedback_v1(self) -> dict[str, Any]:
        """Detached snapshot of the latest completed command transition.

        The environment retains its own deep copy so neither mutation of an
        ``info`` payload nor mutation of this exported value can rewrite the
        command history observed by a controller.
        """

        return deepcopy(self._last_command_feedback_v1)

    def reset(
        self,
        seed: int | None = None,
        *,
        obstacle: bool | None = None,
        stress: bool = False,
    ) -> dict[str, np.ndarray]:
        # ProductionEdgeArmEnv.reset calls self.observation(); keep that call on
        # physical state until the new episode's sensor profile is initialized.
        self._sim2real_ready = False
        super().reset(seed=seed, obstacle=obstacle, stress=stress)
        episode_seed = (
            int(self.seed) if seed is not None else int(self.rng.integers(0, np.iinfo(np.int64).max))
        )
        self._episode_rng = np.random.default_rng(episode_seed ^ 0x51A2EA1)
        low_delay, high_delay = self.sim2real_config.command_delay_steps_range
        self._command_delay_steps = int(self._episode_rng.integers(low_delay, high_delay + 1))
        self._servo_response = float(self._episode_rng.uniform(*self.sim2real_config.servo_response_range))
        self._backlash = self._episode_rng.uniform(*self.sim2real_config.backlash_range_rad, size=_JOINTS)
        self._deadband = self._episode_rng.uniform(
            *self.sim2real_config.command_deadband_range_rad, size=_JOINTS
        )
        zero_offset = self._episode_rng.normal(
            0.0, self.sim2real_config.joint_zero_offset_std_rad, size=_JOINTS
        )
        limit = self.sim2real_config.joint_zero_offset_limit_rad
        self._zero_offset = np.clip(zero_offset, -limit, limit)
        self._servo_velocity.fill(0.0)
        self._last_motor_direction.fill(0)
        self._backlash_remaining.fill(0.0)
        self._submission_ingress_lost = False
        current = self.data.qpos[:_JOINTS].copy()
        zero_action = np.zeros(_JOINTS, dtype=np.float64)
        self._command_queue = deque()
        for slot in range(self._command_delay_steps):
            # These are virtual pre-episode hold submissions.  Their negative
            # send steps make the measured delay exact from the first policy
            # transition, while id=-1 distinguishes them from real commands.
            send_step = slot - self._command_delay_steps
            self._command_queue.append(
                _QueuedCommand(
                    current,
                    command_id=-1,
                    original_action=zero_action,
                    submitted_action=zero_action,
                    send_step=send_step,
                    send_time_seconds=send_step * self.control_dt,
                )
            )
        self._refresh_encoder_noise()
        self._sim2real_ready = True
        self._command_epoch += 1
        self._last_command_feedback_v1 = self._unavailable_command_feedback_v1()
        self.episode_sim2real = {
            "profile_version": self.profile_version,
            "command_epoch": self._command_epoch,
            "seed": episode_seed,
            "command_delay_steps": self._command_delay_steps,
            "control_dt_seconds": self.control_dt,
            "nominal_camera_period_seconds": 1.0 / self.config.fps,
            "physics_substep_seconds": float(self.model.opt.timestep),
            "servo_response": self._servo_response,
            "max_joint_velocity_rad_s": list(self.sim2real_config.max_joint_velocity),
            "max_joint_acceleration_rad_s2": list(self.sim2real_config.max_joint_acceleration),
            "backlash_rad": self._backlash.tolist(),
            "command_deadband_rad": self._deadband.tolist(),
            "joint_zero_offset_rad": self._zero_offset.tolist(),
            "encoder_position_noise_std_rad": (self.sim2real_config.encoder_position_noise_std_rad),
            "encoder_velocity_noise_std_rad_s": (self.sim2real_config.encoder_velocity_noise_std_rad_s),
            "tracking_noise_std_rad": self.sim2real_config.tracking_noise_std_rad,
        }
        self.episode_domain["sim2real_v2"] = self.episode_sim2real
        return self.observation()

    def observation(self) -> dict[str, np.ndarray]:
        observation = super().observation()
        if self._sim2real_ready:
            reported_position = self.data.qpos[:_JOINTS] + self._zero_offset + self._encoder_position_noise
            reported_velocity = self.data.qvel[:_JOINTS] + self._encoder_velocity_noise
            observation["joint_state"] = np.concatenate([reported_position, reported_velocity]).astype(
                np.float32
            )
        return observation

    def _command_reference_reported_position(self) -> np.ndarray:
        """Return the reported position used to form a relative servo target."""

        return (
            self.data.qpos[:_JOINTS]
            + self._zero_offset
            + self._encoder_position_noise
        ).astype(np.float64)

    def step(self, action: np.ndarray) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        action = np.asarray(action, dtype=np.float64)
        if action.shape != (_JOINTS,) or not np.all(np.isfinite(action)):
            raise ValueError("Sim2real action must be a finite six-joint vector")
        if self.estop:
            info = {
                "safety_stop": "estop",
                "success": False,
                "terminated": True,
                "truncated": False,
                "time_limit_reached": False,
                "terminal_failure": True,
                "terminal_reason": "safety_stop:estop",
                "sim2real_v2": {"profile_version": self.profile_version},
            }
            return self.observation(), -2.0, True, False, info

        current = self.data.qpos[:_JOINTS].copy()
        submitted_command_id = int(self.step_count)
        submitted_send_step = int(self.step_count)
        submitted_send_time = float(self.data.time)
        original_action = action.copy()
        submitted_action = np.clip(action, -1.0, 1.0)
        submitted_action_changed_mask = self._target_changed_mask(original_action, submitted_action)
        submitted_ingress_lost = bool(self._submission_ingress_lost)
        # A position servo receives an absolute target formed from the same
        # reported joint position that was available to the policy.  Subclasses
        # may add encoder quantization in ``observation()``; bypassing that
        # observable value here would make a pre-queue label impossible to
        # reconstruct causally from the decision row.
        reported_position = self._command_reference_reported_position()
        requested_measured = (
            reported_position + submitted_action * self.config.max_joint_delta
        )
        requested_physical = requested_measured - self._zero_offset
        queued_target, command_safety = self._safety_filter(requested_physical)
        command_changed_mask = self._target_changed_mask(requested_physical, queued_target)
        self._command_queue.append(
            _QueuedCommand(
                queued_target,
                command_id=submitted_command_id,
                original_action=original_action,
                submitted_action=submitted_action,
                send_step=submitted_send_step,
                send_time_seconds=submitted_send_time,
                submission_safety_reason=command_safety,
                submission_target_changed_mask=command_changed_mask,
                ingress_lost=submitted_ingress_lost,
            )
        )
        applied_command = self._command_queue.popleft()
        delayed_queued_target = np.asarray(applied_command, dtype=np.float64).copy()
        # This is deliberately an identity operation for the base profile.  A
        # force-limited subclass may rewrite the *exact command that was
        # popped* before the stateful servo shaper observes it.  Capturing the
        # queued target first preserves the submitted/application boundary;
        # any subclass rewrite is additionally recorded on ``applied_command``
        # as post-submission provenance instead of masquerading as policy
        # intent.
        preflight_safety = self._preflight_applied_delayed_command(applied_command)
        preflighted_delayed_target = np.asarray(
            applied_command, dtype=np.float64
        ).copy()
        delayed_target, execution_safety = self._safety_filter(
            preflighted_delayed_target
        )
        execution_safety = self._combine_reasons(
            preflight_safety,
            execution_safety,
        )
        execution_changed_mask = self._target_changed_mask(
            delayed_queued_target,
            delayed_target,
        )
        if preflight_safety:
            # The historical mask uses NumPy's loose default tolerance.  Keep
            # that exact base behavior, but never let a deliberately minimal
            # subclass preflight rewrite disappear from application lineage.
            execution_changed_mask = np.logical_or(
                execution_changed_mask,
                ~np.isclose(
                    delayed_queued_target,
                    delayed_target,
                    rtol=0.0,
                    atol=1.0e-12,
                ),
            )
        applied_command_id = int(applied_command.command_id)
        apply_step = int(self.step_count)
        apply_time = float(self.data.time)
        actual_delay_steps = apply_step - int(applied_command.send_step)
        actual_delay_seconds = apply_time - float(applied_command.send_time_seconds)
        actually_applied_delayed_action = (delayed_target - current) / float(self.config.max_joint_delta)

        (
            next_position,
            physical_velocity,
            runtime_safety,
            tracking_noise,
            runtime_changed_mask,
        ) = self._servo_step(delayed_target)
        safety = self._combine_reasons(command_safety, execution_safety, runtime_safety)
        self._advance_physics(current, next_position, physical_velocity, delayed_target)
        self.step_count += 1
        distance = self.distance_to_target()
        progress = self.last_distance - distance
        self.last_distance = distance
        self.success_streak = self.success_streak + 1 if distance < self.config.success_radius else 0
        success = self.success_streak >= self.config.success_hold_steps
        block = self.block_xy()
        out = not (0.06 <= block[0] <= 0.45 and -0.29 <= block[1] <= 0.29)
        time_limit_reached = self.step_count >= self.config.max_steps
        terminated = bool(success or out)
        # Gymnasium semantics: a true terminal state always takes precedence
        # over the administrative time limit when both happen on the same
        # simulator step.  This keeps value bootstrapping unambiguous.
        truncated = bool(time_limit_reached and not terminated)
        terminal_failure = bool(out)
        if success:
            terminal_reason = "legacy_success"
        elif out:
            terminal_reason = "block_out_of_bounds"
        elif truncated:
            terminal_reason = "time_limit"
        else:
            terminal_reason = "nonterminal"
        reward = 28.0 * progress - 0.012 * float(np.square(action).sum()) - 0.015
        if success:
            reward += 10.0
        if out:
            reward -= 6.0
        if safety:
            reward -= 0.08

        self._refresh_encoder_noise()
        info = {
            "success": success,
            "terminated": terminated,
            "truncated": truncated,
            "time_limit_reached": bool(time_limit_reached),
            "terminal_failure": terminal_failure,
            "terminal_reason": terminal_reason,
            "distance": distance,
            "progress": progress,
            "safety_clipped": bool(safety),
            "safety_reason": safety,
            "contact_count": self._tool_block_contacts(),
            "obstacle": self.obstacle_enabled,
            "sim2real_v2": {
                "profile_version": self.profile_version,
                "command_delay_steps": self._command_delay_steps,
                "submitted_command_id": submitted_command_id,
                "applied_command_id": applied_command_id,
                "original_action": original_action.astype(np.float32),
                "submitted_action": submitted_action.astype(np.float32),
                "submitted_action_changed_mask": submitted_action_changed_mask.astype(np.uint8),
                "submitted_command_ingress_lost": submitted_ingress_lost,
                "applied_command_original_action": applied_command.original_action.astype(np.float32),
                "applied_command_submitted_action": applied_command.submitted_action.astype(np.float32),
                "applied_command_ingress_lost": bool(applied_command.ingress_lost),
                "applied_command_post_submission_rewrite_count": int(
                    applied_command.post_submission_rewrite_count
                ),
                "applied_command_post_submission_target_changed_mask": (
                    applied_command.post_submission_target_changed_mask.astype(
                        np.uint8
                    )
                ),
                "applied_command_post_submission_rewrite_reasons": (
                    applied_command.post_submission_rewrite_reasons
                ),
                "delayed_submitted_action": applied_command.submitted_action.astype(np.float32),
                "actually_applied_delayed_action": actually_applied_delayed_action.astype(np.float32),
                # Backward-friendly concise alias for consumers that already
                # use the word "applied" for the delayed controller command.
                "applied_delayed_action": actually_applied_delayed_action.astype(np.float32),
                "submitted_command_send_step": submitted_send_step,
                "submitted_command_send_time_seconds": submitted_send_time,
                "applied_command_send_step": int(applied_command.send_step),
                "applied_command_send_time_seconds": float(applied_command.send_time_seconds),
                "applied_command_apply_step": apply_step,
                "applied_command_apply_time_seconds": apply_time,
                "send_step": int(applied_command.send_step),
                "send_time_seconds": float(applied_command.send_time_seconds),
                "apply_step": apply_step,
                "apply_time_seconds": apply_time,
                "actual_delay_steps": actual_delay_steps,
                "actual_delay_seconds": actual_delay_seconds,
                "actual_command_delay_steps": actual_delay_steps,
                "actual_command_delay_seconds": actual_delay_seconds,
                "requested_joint_target": requested_physical.astype(np.float32),
                "queued_safe_joint_target": queued_target.astype(np.float32),
                "applied_queued_safe_joint_target": delayed_queued_target.astype(np.float32),
                "delayed_joint_target": delayed_target.astype(np.float32),
                "safety_stage_names": (
                    "submit_current",
                    "apply_delayed",
                    "runtime_delayed",
                ),
                "safety_stage_mask": np.asarray(
                    [bool(command_safety), bool(execution_safety), bool(runtime_safety)],
                    dtype=np.uint8,
                ),
                "safety_stage_mask_by_name": {
                    "submit_current": bool(command_safety),
                    "apply_delayed": bool(execution_safety),
                    "runtime_delayed": bool(runtime_safety),
                },
                "safety_target_changed_mask": np.stack(
                    [
                        command_changed_mask,
                        execution_changed_mask,
                        runtime_changed_mask,
                    ]
                ).astype(np.uint8),
                "target_changed_mask": np.logical_or.reduce(
                    [
                        command_changed_mask,
                        execution_changed_mask,
                        runtime_changed_mask,
                    ]
                ).astype(np.uint8),
                "submitted_command_target_changed_mask": command_changed_mask.astype(np.uint8),
                "applied_command_submission_target_changed_mask": (
                    applied_command.submission_target_changed_mask.astype(np.uint8)
                ),
                "applied_command_safety_stage_names": (
                    "submit",
                    "apply",
                    "runtime",
                ),
                "applied_command_safety_stage_mask": np.asarray(
                    [
                        bool(applied_command.submission_safety_reason),
                        bool(execution_safety),
                        bool(runtime_safety),
                    ],
                    dtype=np.uint8,
                ),
                "applied_command_safety_target_changed_mask": np.stack(
                    [
                        applied_command.submission_target_changed_mask,
                        execution_changed_mask,
                        runtime_changed_mask,
                    ]
                ).astype(np.uint8),
                "application_target_changed_mask": execution_changed_mask.astype(np.uint8),
                "runtime_target_changed_mask": runtime_changed_mask.astype(np.uint8),
                "submitted_command_safety_reason": command_safety,
                "applied_command_submission_safety_reason": (applied_command.submission_safety_reason),
                "application_safety_reason": execution_safety,
                "runtime_safety_reason": runtime_safety,
                "physical_joint_position": self.data.qpos[:_JOINTS].copy().astype(np.float32),
                "physical_joint_velocity": physical_velocity.astype(np.float32),
                "reported_joint_position": self.observation()["joint_state"][:_JOINTS],
                "servo_velocity": self._servo_velocity.copy().astype(np.float32),
                "backlash_remaining_rad": self._backlash_remaining.copy().astype(np.float32),
                "tracking_noise_rad": tracking_noise.astype(np.float32),
            },
        }
        self._last_command_feedback_v1 = self._command_feedback_from_transport_v1(
            info["sim2real_v2"]
        )
        return self.observation(), float(reward), terminated, truncated, info

    def _unavailable_command_feedback_v1(self) -> dict[str, Any]:
        """Return the fixed-shape no-transition state used at init/reset."""

        zero_action = np.zeros(_JOINTS, dtype=np.float32)
        zero_mask = np.zeros(_JOINTS, dtype=np.uint8)
        zero_stage_mask = np.zeros(3, dtype=np.uint8)
        zero_stage_target_mask = np.zeros((3, _JOINTS), dtype=np.uint8)
        return {
            "available": False,
            "profile": COMMAND_FEEDBACK_PROFILE_VERSION,
            "source_profile": self.profile_version,
            "epoch": int(self._command_epoch),
            "transition_step": -1,
            "submitted_id": -1,
            "applied_id": -1,
            "next_command_id": int(self.step_count),
            "applied_ingress_lost": False,
            "applied_is_virtual_hold": False,
            "pending_virtual_hold_count": sum(
                int(command.command_id < 0) for command in self._command_queue
            ),
            "applied_command_submitted_action": zero_action.copy(),
            "actually_applied_delayed_action": zero_action.copy(),
            "applied_queued_safe_target": zero_action.copy(),
            "delayed_target": zero_action.copy(),
            "safety_stage_mask": zero_stage_mask.copy(),
            "applied_command_safety_stage_mask": zero_stage_mask.copy(),
            "safety_target_changed_mask": zero_stage_target_mask.copy(),
            "applied_command_safety_target_changed_mask": zero_stage_target_mask.copy(),
            "target_changed_mask": zero_mask.copy(),
            "applied_command_post_submission_rewrite_count": 0,
            "applied_command_post_submission_target_changed_mask": zero_mask.copy(),
            "applied_command_post_submission_rewrite_reasons": (),
        }

    def _command_feedback_from_transport_v1(self, transport: dict[str, Any]) -> dict[str, Any]:
        """Project one completed ``sim2real_v2`` transition into feedback v1."""

        applied_id = int(transport["applied_command_id"])
        feedback = {
            "available": True,
            "profile": COMMAND_FEEDBACK_PROFILE_VERSION,
            "source_profile": str(transport["profile_version"]),
            "epoch": int(self._command_epoch),
            "transition_step": int(transport["apply_step"]),
            "submitted_id": int(transport["submitted_command_id"]),
            "applied_id": applied_id,
            "next_command_id": int(self.step_count),
            "applied_ingress_lost": bool(transport["applied_command_ingress_lost"]),
            "applied_command_post_submission_rewrite_count": int(
                transport["applied_command_post_submission_rewrite_count"]
            ),
            "applied_command_post_submission_target_changed_mask": transport[
                "applied_command_post_submission_target_changed_mask"
            ],
            "applied_command_post_submission_rewrite_reasons": tuple(
                str(value)
                for value in transport[
                    "applied_command_post_submission_rewrite_reasons"
                ]
            ),
            "applied_is_virtual_hold": applied_id < 0,
            "pending_virtual_hold_count": sum(
                int(command.command_id < 0) for command in self._command_queue
            ),
            "applied_command_submitted_action": transport[
                "applied_command_submitted_action"
            ],
            "actually_applied_delayed_action": transport[
                "actually_applied_delayed_action"
            ],
            "applied_queued_safe_target": transport[
                "applied_queued_safe_joint_target"
            ],
            "delayed_target": transport["delayed_joint_target"],
            "safety_stage_mask": transport["safety_stage_mask"],
            "applied_command_safety_stage_mask": transport[
                "applied_command_safety_stage_mask"
            ],
            "safety_target_changed_mask": transport["safety_target_changed_mask"],
            "applied_command_safety_target_changed_mask": transport[
                "applied_command_safety_target_changed_mask"
            ],
            "target_changed_mask": transport["target_changed_mask"],
        }
        return deepcopy(feedback)

    def _preflight_applied_delayed_command(
        self,
        applied_command: _QueuedCommand,
    ) -> str:
        """Identity hook before stateful servo shaping of a delayed command.

        The base V2 surrogate intentionally preserves its historical behavior.
        Subclasses may use :meth:`_QueuedCommand.rewrite_target` here so an
        application-time safety transform remains attributable to the exact
        applied command ID and is exported as post-submission provenance.
        """

        del applied_command
        return ""

    def _servo_step(self, target: np.ndarray) -> tuple[np.ndarray, np.ndarray, str, np.ndarray, np.ndarray]:
        current = self.data.qpos[:_JOINTS].copy()
        dt = self.control_dt
        error = target - current
        active_error = np.where(np.abs(error) > self._deadband, error, 0.0)
        desired_velocity = self._servo_response * active_error / max(dt, 1e-9)
        max_velocity = np.asarray(self.sim2real_config.max_joint_velocity, dtype=np.float64)
        desired_velocity = np.clip(desired_velocity, -max_velocity, max_velocity)
        max_acceleration = np.asarray(self.sim2real_config.max_joint_acceleration, dtype=np.float64)
        velocity_delta = np.clip(
            desired_velocity - self._servo_velocity,
            -max_acceleration * dt,
            max_acceleration * dt,
        )
        self._servo_velocity += velocity_delta
        nominal_step = self._servo_velocity * dt
        same_direction = np.sign(nominal_step) == np.sign(active_error)
        nominal_step = np.where(
            same_direction & (np.abs(nominal_step) > np.abs(active_error)),
            active_error,
            nominal_step,
        )

        direction = np.sign(nominal_step).astype(np.int8)
        reversal = (
            (direction != 0) & (self._last_motor_direction != 0) & (direction != self._last_motor_direction)
        )
        self._backlash_remaining[reversal] = self._backlash[reversal]
        moving = direction != 0
        take_up = np.minimum(self._backlash_remaining, np.abs(nominal_step))
        transmitted_step = direction * (np.abs(nominal_step) - take_up)
        self._backlash_remaining[moving] -= take_up[moving]
        self._last_motor_direction[moving] = direction[moving]

        tracking_noise = np.zeros(_JOINTS, dtype=np.float64)
        if self.sim2real_config.tracking_noise_std_rad > 0:
            tracking_noise[moving] = self._episode_rng.normal(
                0.0,
                self.sim2real_config.tracking_noise_std_rad,
                int(np.count_nonzero(moving)),
            )
        proposed = current + transmitted_step + tracking_noise
        max_step = max_velocity * dt
        proposed = current + np.clip(proposed - current, -max_step, max_step)
        next_position, runtime_safety = self._safety_filter(proposed)
        runtime_changed_mask = self._target_changed_mask(proposed, next_position)
        physical_velocity = (next_position - current) / max(dt, 1e-9)
        return (
            next_position,
            physical_velocity,
            runtime_safety,
            tracking_noise,
            runtime_changed_mask,
        )

    def _advance_physics(
        self,
        start: np.ndarray,
        end: np.ndarray,
        velocity: np.ndarray,
        controller_target: np.ndarray,
    ) -> None:
        self.data.ctrl[:] = controller_target
        substeps = self.config.physics_substeps
        for substep in range(1, substeps + 1):
            fraction = substep / substeps
            self.data.qpos[:_JOINTS] = start + fraction * (end - start)
            self.data.qvel[:_JOINTS] = velocity
            mujoco.mj_forward(self.model, self.data)
            mujoco.mj_step(self.model, self.data)
        # MuJoCo actuators are not the servo model in this surrogate. Restore
        # the explicitly rate-limited state after advancing contact dynamics.
        self.data.qpos[:_JOINTS] = end
        self.data.qvel[:_JOINTS] = velocity
        mujoco.mj_forward(self.model, self.data)

    def _refresh_encoder_noise(self) -> None:
        position_std = self.sim2real_config.encoder_position_noise_std_rad
        velocity_std = self.sim2real_config.encoder_velocity_noise_std_rad_s
        self._encoder_position_noise = self._episode_rng.normal(0.0, position_std, _JOINTS)
        self._encoder_velocity_noise = self._episode_rng.normal(0.0, velocity_std, _JOINTS)

    @staticmethod
    def _target_changed_mask(before: np.ndarray, after: np.ndarray) -> np.ndarray:
        """Return the per-joint counterpart of ``np.allclose`` safety checks."""

        return np.logical_not(
            np.isclose(
                np.asarray(before, dtype=np.float64),
                np.asarray(after, dtype=np.float64),
            )
        )

    @staticmethod
    def _combine_reasons(*reasons: str) -> str:
        result: list[str] = []
        for reason in reasons:
            for item in reason.split("+"):
                if item and item not in result:
                    result.append(item)
        return "+".join(result)


@dataclass(frozen=True)
class Sim2RealWristCameraConfig(CameraCaptureConfig):
    """Wrist-only camera degradation and transport profile."""

    rgb_noise_std: float = 4.0
    depth_noise_mm: float = 3.5
    depth_dropout_probability: float = 0.008
    latency_steps_range: tuple[int, int] = (0, 3)
    frame_drop_probability: float = 0.05
    occlusion_probability: float = 0.10
    occlusion_size_range: tuple[float, float] = (0.10, 0.28)

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError("camera width and height must be positive")
        if self.cameras != ("wrist",):
            raise ValueError("Sim2Real v2 formal camera data must be wrist-only")
        delay = self.latency_steps_range
        if (
            len(delay) != 2
            or not all(isinstance(value, int) for value in delay)
            or delay[0] < 0
            or delay[0] > delay[1]
        ):
            raise ValueError("latency_steps_range must be an ordered non-negative pair")
        for name in (
            "rgb_noise_std",
            "depth_noise_mm",
            "depth_dropout_probability",
            "frame_drop_probability",
            "occlusion_probability",
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")
        for name in (
            "depth_dropout_probability",
            "frame_drop_probability",
            "occlusion_probability",
        ):
            if float(getattr(self, name)) > 1.0:
                raise ValueError(f"{name} cannot exceed 1.0")
        _validate_range("occlusion_size_range", self.occlusion_size_range)
        if self.occlusion_size_range[0] <= 0 or self.occlusion_size_range[1] > 1:
            raise ValueError("occlusion_size_range must be within (0, 1]")


@dataclass(frozen=True)
class _WristTransportPacket:
    """One camera source sample and all metadata captured with it."""

    source_index: int
    payload: dict[str, np.ndarray]
    occluded: bool
    occlusion_rectangle: list[int] | None
    source_context: dict[str, Any] | None


class Sim2RealWristTransport:
    """Headless-testable transport for synchronized wrist sensor packets."""

    profile_version = SIM2REAL_CAMERA_PROFILE_VERSION

    def __init__(self, config: Sim2RealWristCameraConfig | None = None, seed: int = 0):
        self.config = config or Sim2RealWristCameraConfig()
        self.rng = np.random.default_rng(seed ^ 0xCA4E2A)
        self.frame_latency_steps = 0
        self.metadata: dict[str, Any] = {}
        self.last_capture_metadata: dict[str, Any] = {}
        self._queues: dict[str, deque[_WristTransportPacket]] = {
            "full": deque(),
            "full_pose": deque(),
            "rgb": deque(),
        }
        self._last_delivered: dict[str, _WristTransportPacket | None] = {
            "full": None,
            "full_pose": None,
            "rgb": None,
        }
        self._frame_indices = {"full": 0, "full_pose": 0, "rgb": 0}

    def begin_episode(self, seed: int) -> None:
        self.rng = np.random.default_rng(seed ^ 0xCA4E2A)
        low, high = self.config.latency_steps_range
        self.frame_latency_steps = int(self.rng.integers(low, high + 1))
        for queue in self._queues.values():
            queue.clear()
        self._last_delivered = {"full": None, "full_pose": None, "rgb": None}
        self._frame_indices = {"full": 0, "full_pose": 0, "rgb": 0}
        self.last_capture_metadata = {}
        self.metadata = {
            "profile_version": self.profile_version,
            "seed": int(seed),
            "camera_mount": "wrist",
            "camera_names": ["wrist"],
            "latency_steps": self.frame_latency_steps,
            "frame_drop_probability": self.config.frame_drop_probability,
            "occlusion_probability": self.config.occlusion_probability,
            "occlusion_size_range": list(self.config.occlusion_size_range),
        }

    def process(
        self,
        raw: dict[str, np.ndarray],
        stream: str,
        *,
        source_context: dict[str, Any] | None = None,
    ) -> dict[str, np.ndarray]:
        if stream == "full":
            expected = {"rgb_wrist", "depth_wrist_mm", "segmentation_wrist"}
        elif stream == "full_pose":
            expected = {
                "rgb_wrist",
                "depth_wrist_mm",
                "segmentation_wrist",
                "camera_pose_wrist",
            }
            camera_pose = np.asarray(raw.get("camera_pose_wrist"))
            if camera_pose.shape != (12,) or not np.all(np.isfinite(camera_pose)):
                raise RuntimeError("camera_pose_wrist must be a finite 12-vector")
        elif stream == "rgb":
            expected = {"rgb_wrist"}
        else:
            raise ValueError(f"Unknown wrist transport stream: {stream}")
        if set(raw) != expected:
            raise RuntimeError(f"Unexpected {stream} camera streams: {sorted(raw)}")
        source_index = self._frame_indices[stream]
        self._frame_indices[stream] += 1
        degraded, occluded, rectangle = self._apply_occlusion(raw)
        packet = _WristTransportPacket(
            source_index=source_index,
            payload=degraded,
            occluded=occluded,
            occlusion_rectangle=rectangle,
            source_context=deepcopy(source_context),
        )
        queue = self._queues[stream]
        queue.append(packet)
        padded = len(queue) <= self.frame_latency_steps
        if padded:
            selected = queue[0]
        else:
            selected = queue.popleft()

        previous = self._last_delivered[stream]
        candidate_index = selected.source_index
        dropped = False
        if previous is not None and self.rng.random() < self.config.frame_drop_probability:
            selected = previous
            dropped = True
        else:
            self._last_delivered[stream] = selected
        delivered_index = selected.source_index
        repeated = previous is not None and delivered_index == previous.source_index
        self.last_capture_metadata = {
            "profile_version": self.profile_version,
            "stream": stream,
            "source_frame_index": source_index,
            "delivered_frame_index": delivered_index,
            "delivered_frame_age_steps": source_index - delivered_index,
            "configured_latency_steps": self.frame_latency_steps,
            "latency_padding": padded,
            "frame_dropped": dropped,
            "delivered_frame_occluded": selected.occluded,
            "occlusion_rectangle_xywh": selected.occlusion_rectangle,
            "transport_candidate_frame_index": candidate_index,
            "delivered_frame_repeated": repeated,
        }
        if selected.source_context is not None:
            self.last_capture_metadata["delivered_source_context"] = deepcopy(selected.source_context)
        return {key: value.copy() for key, value in selected.payload.items()}

    def _apply_occlusion(
        self, raw: dict[str, np.ndarray]
    ) -> tuple[dict[str, np.ndarray], bool, list[int] | None]:
        output = {key: value.copy() for key, value in raw.items()}
        if self.rng.random() >= self.config.occlusion_probability:
            return output, False, None
        height, width = output["rgb_wrist"].shape[:2]
        low, high = self.config.occlusion_size_range
        rectangle_width = max(1, int(round(width * self.rng.uniform(low, high))))
        rectangle_height = max(1, int(round(height * self.rng.uniform(low, high))))
        x = int(self.rng.integers(0, width - rectangle_width + 1))
        y = int(self.rng.integers(0, height - rectangle_height + 1))
        rgb = output["rgb_wrist"]
        rgb[y : y + rectangle_height, x : x + rectangle_width] = self.rng.integers(
            0,
            14,
            size=(rectangle_height, rectangle_width, 3),
            dtype=np.uint8,
        )
        if "depth_wrist_mm" in output:
            output["depth_wrist_mm"][y : y + rectangle_height, x : x + rectangle_width] = 0
        if "segmentation_wrist" in output:
            output["segmentation_wrist"][y : y + rectangle_height, x : x + rectangle_width] = 0
        return output, True, [x, y, rectangle_width, rectangle_height]


class Sim2RealWristRenderer(TrueMultimodalRenderer):
    """True wrist render plus delayed/dropped/occluded sensor delivery."""

    profile_version = SIM2REAL_CAMERA_PROFILE_VERSION

    def __init__(
        self,
        env: ProductionEdgeArmEnv,
        config: Sim2RealWristCameraConfig | None = None,
    ):
        camera_config = config or Sim2RealWristCameraConfig()
        # Validate before allocating an OpenGL renderer.
        if camera_config.cameras != ("wrist",):
            raise ValueError("Sim2Real v2 formal camera data must be wrist-only")
        super().__init__(env, camera_config)
        self.config = camera_config
        self.transport = Sim2RealWristTransport(camera_config, seed=env.seed)

    @property
    def frame_latency_steps(self) -> int:
        return self.transport.frame_latency_steps

    @property
    def transport_metadata(self) -> dict[str, Any]:
        return self.transport.metadata

    @property
    def last_capture_metadata(self) -> dict[str, Any]:
        return self.transport.last_capture_metadata

    def begin_episode(self, seed: int) -> None:
        super().begin_episode(seed)
        self.transport.begin_episode(seed)

    def capture(self) -> dict[str, np.ndarray]:
        return self.transport.process(super().capture(), "full")

    def capture_rgb_only(self) -> dict[str, np.ndarray]:
        return self.transport.process(super().capture_rgb_only(), "rgb")

    def capture_synchronized(self) -> dict[str, np.ndarray]:
        """Capture delayed images and their matching wrist-camera world pose."""

        mujoco.mj_forward(self.env.model, self.env.data)
        packet = super().capture()
        camera_id = self.env._ids["cameras"]["wrist"]
        packet["camera_pose_wrist"] = np.concatenate(
            [self.env.data.cam_xpos[camera_id], self.env.data.cam_xmat[camera_id]]
        ).astype(np.float32)
        return self.transport.process(packet, "full_pose")

    def calibration_metadata(self) -> dict[str, Any]:
        calibration = super().calibration_metadata()
        if set(calibration) != {"wrist"}:
            raise RuntimeError("Sim2Real v2 camera calibration must remain wrist-only")
        calibration["wrist"]["sim2real_v2_transport"] = self.transport.metadata.copy()
        return calibration
