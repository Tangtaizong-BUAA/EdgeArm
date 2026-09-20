"""Searchable privileged V8 expert for the obstacle-free synthetic V6 plant.

V8 keeps the proven V7 approach/push controller, but replaces its discontinuous
``coverage >= threshold -> zero`` transition with a small contact-unload,
brake, hysteretic recovery, and hold state machine.  The controller is a
simulator-only demonstration teacher.  It consumes privileged block pose,
velocity, footprint coverage, and contact state and represents zero physical
robot samples.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import mujoco
import numpy as np

from .physical_expert_v7 import (
    EXPERT_CLAIM_LEVEL,
    PhysicalClosedLoopExpertV7,
    PhysicalExpertV7Config,
)
from .sim2real_env_v6 import RealisticEdgeArmEnvV6


PHYSICAL_EXPERT_V8_VERSION = "edgearm-privileged-physical-expert-v8"
EXPERT_V8_PARAMETER_SOURCE = "synthetic_multistage_control_search"
PHYSICAL_EXPERT_V8_CONFIG_FORMAT = "edgearm-deterministic-physical-expert-v8-config"
PHYSICAL_EXPERT_V8_CONFIG_SCHEMA_VERSION = 1
_JOINTS = 6


@dataclass(frozen=True)
class PhysicalExpertV8Config(PhysicalExpertV7Config):
    """Synthetic search parameters; none are calibrated on physical hardware."""

    contact_recovery_action_scale: float = 1.15
    near_target_coverage_start: float = 0.70
    minimum_push_scale: float = 0.35
    settle_retract_step_m: float = 0.004
    settle_lift_step_m: float = 0.0015
    settle_coverage_hysteresis: float = 0.04
    settle_recovery_scale: float = 0.35
    settle_hold_release_gain: float = 0.035
    settle_hold_decay_steps: float = 4.0
    unload_speed_gain: float = 0.30
    unload_contact_speed_gate_m_s: float = 0.008
    predictive_reverse_gain: float = 0.20

    def __post_init__(self) -> None:
        super().__post_init__()
        positive = (
            "contact_recovery_action_scale",
            "settle_retract_step_m",
            "settle_coverage_hysteresis",
            "settle_recovery_scale",
            "settle_hold_decay_steps",
            "unload_speed_gain",
            "unload_contact_speed_gate_m_s",
            "predictive_reverse_gain",
        )
        for name in positive:
            value = float(getattr(self, name))
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if not 0.0 <= self.near_target_coverage_start < 1.0:
            raise ValueError("near_target_coverage_start must be in [0, 1)")
        if not 0.0 < self.minimum_push_scale <= 1.0:
            raise ValueError("minimum_push_scale must be in (0, 1]")
        if not 0.0 < self.settle_coverage_hysteresis < 1.0:
            raise ValueError("settle_coverage_hysteresis must be in (0, 1)")
        if not np.isfinite(self.settle_lift_step_m) or self.settle_lift_step_m < 0.0:
            raise ValueError("settle_lift_step_m must be finite and non-negative")
        if not 0.0 <= self.settle_hold_release_gain <= 1.0:
            raise ValueError("settle_hold_release_gain must be in [0, 1]")


class PhysicalClosedLoopExpertV8(PhysicalClosedLoopExpertV7):
    """V7-compatible privileged expert with continuous terminal control."""

    teacher_type = "physical_expert_v8_synthetic_privileged"
    selected_update = 8
    checkpoint = ""

    def __init__(
        self,
        env: RealisticEdgeArmEnvV6 | None = None,
        config: PhysicalExpertV8Config | None = None,
    ) -> None:
        self.v8_config = config or PhysicalExpertV8Config()
        super().__init__(env=env, config=self.v8_config)
        self._settle_active = False
        self._settle_age = 0
        self._settle_reference_action = np.zeros(_JOINTS, dtype=np.float64)
        self._last_drive_action = np.zeros(_JOINTS, dtype=np.float64)
        self._threshold_exit_count = 0
        self._unload_steps = 0

    def reset(self, env: RealisticEdgeArmEnvV6 | None = None) -> None:
        super().reset(env)
        self._settle_active = False
        self._settle_age = 0
        self._settle_reference_action.fill(0.0)
        self._last_drive_action.fill(0.0)
        self._threshold_exit_count = 0
        self._unload_steps = 0

    def action(
        self,
        env_or_observation: RealisticEdgeArmEnvV6 | Mapping[str, np.ndarray] | None = None,
        observation: Mapping[str, np.ndarray] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Return one command while exposing terminal-control diagnostics."""

        v7_action, metadata = super().action(env_or_observation, observation)
        if self.env is None:  # pragma: no cover - enforced by the V7 parent
            raise RuntimeError("expert.action requires a RealisticEdgeArmEnvV6")

        config = self.v8_config
        coverage = float(self.env.block_target_coverage())
        threshold = float(self.env.realism_config.strict_coverage_threshold)
        linear_speed, angular_speed = self.env._block_speeds()
        block = self.env.block_xy()
        direction = self.env._unit(self.env.target_xy - block)
        block_velocity = self.env.data.qvel[
            self.env._block_dof_address : self.env._block_dof_address + 3
        ]
        speed_toward_target = float(np.dot(block_velocity[:2], direction))
        contact = self.env._tool_block_contacts() > 0
        # Preserve V7 exactly during ordinary approach/push/reposition.  Only
        # reconstruct the suppressed drive when V7 entered its hard-zero
        # predictive/coverage settle branch.
        if metadata.get("phase") == "settle":
            drive_action = self._drive_action_from_metadata(metadata, v7_action)
        else:
            drive_action = np.asarray(v7_action, dtype=np.float64).copy()

        was_settling = self._settle_active
        hysteresis_floor = threshold - config.settle_coverage_hysteresis
        in_settle_band = coverage >= threshold or (
            self._settle_active and coverage >= hysteresis_floor
        )
        predictive_brake = bool(metadata.get("predictive_brake", False))
        settle_mode = "drive"
        contact_unload = False
        recovery_applied = False
        continuous_brake_fraction = 0.0

        if in_settle_band:
            if not self._settle_active:
                self._settle_active = True
                self._settle_age = 0
                reference = drive_action if np.any(drive_action) else self._last_drive_action
                self._settle_reference_action = np.asarray(reference, dtype=np.float64).copy()
            elif coverage < threshold and was_settling:
                self._threshold_exit_count += 1
            self._settle_age += 1

            moving = bool(
                linear_speed > self.env.realism_config.strict_linear_speed_m_s
                or angular_speed > self.env.realism_config.strict_angular_speed_rad_s
            )
            unload_contact = bool(
                contact and linear_speed > config.unload_contact_speed_gate_m_s
            )
            if unload_contact or moving:
                speed_ratio = np.clip(
                    linear_speed
                    / max(self.env.realism_config.strict_linear_speed_m_s, 1e-9),
                    0.0,
                    2.0,
                )
                retract = config.settle_retract_step_m * (
                    1.0 + config.unload_speed_gain * speed_ratio
                )
                action = self._cartesian_delta_action(
                    np.array(
                        [
                            -direction[0] * retract,
                            -direction[1] * retract,
                            config.settle_lift_step_m,
                        ],
                        dtype=np.float64,
                    )
                )
                settle_mode = "contact_unload" if unload_contact else "velocity_brake"
                contact_unload = unload_contact
                self._unload_steps += 1
                continuous_brake_fraction = float(min(1.0, retract / 0.012))
            elif coverage < threshold:
                deficit = np.clip((threshold - coverage) / config.settle_coverage_hysteresis, 0.0, 1.0)
                action = drive_action * config.settle_recovery_scale * deficit
                settle_mode = "hysteretic_recovery"
                recovery_applied = True
            else:
                decay = np.exp(-self._settle_age / config.settle_hold_decay_steps)
                action = (
                    -config.settle_hold_release_gain
                    * decay
                    * self._settle_reference_action
                )
                settle_mode = "released_hold"
        elif predictive_brake:
            self._settle_active = False
            self._settle_age = 0
            projected_distance = float(metadata["projected_distance_m"])
            boundary = (
                self.env.realism_config.block_half_extent_m
                + config.predictive_brake_margin_m
            )
            position_fraction = np.clip(
                (boundary - projected_distance)
                / max(config.predictive_brake_margin_m + 0.004, 1e-6),
                0.0,
                1.0,
            )
            speed_fraction = np.clip(
                speed_toward_target
                / max(config.high_speed_brake_threshold_m_s, 1e-9),
                0.0,
                2.0,
            )
            continuous_brake_fraction = float(
                np.clip(max(position_fraction, 0.5 * speed_fraction), 0.0, 0.85)
            )
            action = (1.0 - continuous_brake_fraction) * drive_action
            action -= (
                continuous_brake_fraction
                * config.predictive_reverse_gain
                * self._last_drive_action
            )
            settle_mode = "predictive_continuous_brake"
        else:
            if self._settle_active:
                self._threshold_exit_count += 1
            self._settle_active = False
            self._settle_age = 0
            action = drive_action.copy()
            if metadata.get("phase") == "reposition":
                action *= config.contact_recovery_action_scale
                recovery_applied = True
                settle_mode = "contact_recovery"
            else:
                coverage_fraction = np.clip(
                    (coverage - config.near_target_coverage_start)
                    / max(threshold - config.near_target_coverage_start, 1e-9),
                    0.0,
                    1.0,
                )
                push_scale = 1.0 - coverage_fraction * (1.0 - config.minimum_push_scale)
                action *= push_scale
                continuous_brake_fraction = float(1.0 - push_scale)
            self._last_drive_action = np.asarray(drive_action, dtype=np.float64).copy()

        action = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        action, prefilter_reason = self._prefilter_action(action)
        action = action.astype(np.float32)
        metadata.update(
            {
                "version": PHYSICAL_EXPERT_V8_VERSION,
                "parameter_source": EXPERT_V8_PARAMETER_SOURCE,
                "physical_samples": 0,
                "physically_calibrated": False,
                "physical_hardware_connected": False,
                "phase": settle_mode if settle_mode != "drive" else metadata["phase"],
                "settle_mode": settle_mode,
                "settle_active": self._settle_active,
                "settle_age_steps": self._settle_age,
                "settle_hysteresis_floor": hysteresis_floor,
                "contact_unload": contact_unload,
                "contact_unload_steps": self._unload_steps,
                "threshold_exit_count": self._threshold_exit_count,
                "hysteretic_recovery": recovery_applied,
                "continuous_brake_fraction": continuous_brake_fraction,
                "speed_toward_target_m_s": speed_toward_target,
                "workspace_prefilter_applied": bool(prefilter_reason),
                "workspace_prefilter_reason": prefilter_reason,
                "normalized_action_linf": float(np.max(np.abs(action))),
                "claim_level": EXPERT_CLAIM_LEVEL,
            }
        )
        return action, metadata

    def _drive_action_from_metadata(
        self,
        metadata: Mapping[str, Any],
        fallback: np.ndarray,
    ) -> np.ndarray:
        if self.env is None:
            raise RuntimeError("drive reconstruction requires an environment")
        target = np.asarray(metadata.get("raw_joint_target"), dtype=np.float64)
        if target.shape != (_JOINTS,) or not np.all(np.isfinite(target)):
            return np.asarray(fallback, dtype=np.float64).copy()
        action = (target - self.env.data.qpos[:_JOINTS]) / self.env.config.max_joint_delta
        action[0] *= self.v8_config.base_joint_scale
        action[1:5] *= self.v8_config.movable_joint_scale
        action = np.clip(action, -1.0, 1.0)
        action, _ = self._prefilter_action(action)
        return np.asarray(action, dtype=np.float64)

    def _cartesian_delta_action(self, delta_xyz: np.ndarray) -> np.ndarray:
        if self.env is None:
            raise RuntimeError("Cartesian unload requires an environment")
        delta_xyz = np.asarray(delta_xyz, dtype=np.float64)
        if delta_xyz.shape != (3,) or not np.all(np.isfinite(delta_xyz)):
            raise ValueError("delta_xyz must be a finite three-vector")
        jacobian_position = np.zeros((3, self.env.model.nv), dtype=np.float64)
        jacobian_rotation = np.zeros((3, self.env.model.nv), dtype=np.float64)
        mujoco.mj_jacSite(
            self.env.model,
            self.env.data,
            jacobian_position,
            jacobian_rotation,
            self.env._ids["tool_site"],
        )
        jacobian = jacobian_position[:, :5]
        delta = jacobian.T @ np.linalg.solve(
            jacobian @ jacobian.T + 1.5e-3 * np.eye(3),
            delta_xyz,
        )
        action = np.zeros(_JOINTS, dtype=np.float64)
        action[:5] = delta / self.env.config.max_joint_delta
        action[0] *= self.v8_config.base_joint_scale
        action[1:5] *= self.v8_config.movable_joint_scale
        return np.clip(action, -1.0, 1.0)


def load_physical_expert_v8(
    checkpoint: object | None = None,
    *,
    config: PhysicalExpertV8Config | None = None,
) -> PhysicalClosedLoopExpertV8:
    """Factory matching dataset-collector teacher interfaces."""

    if checkpoint is not None:
        raise ValueError("physical expert V8 has no learned checkpoint")
    return PhysicalClosedLoopExpertV8(config=config)


__all__ = [
    "EXPERT_V8_PARAMETER_SOURCE",
    "PHYSICAL_EXPERT_V8_CONFIG_FORMAT",
    "PHYSICAL_EXPERT_V8_CONFIG_SCHEMA_VERSION",
    "PHYSICAL_EXPERT_V8_VERSION",
    "PhysicalClosedLoopExpertV8",
    "PhysicalExpertV8Config",
    "load_physical_expert_v8",
]
