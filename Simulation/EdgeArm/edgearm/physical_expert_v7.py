"""Deterministic privileged closed-loop expert for the synthetic V6 plant.

The controller is deliberately separate from :mod:`sim2real_env_v6`: it is a
demonstration generator, not part of the environment or a claim about a real
SO-101 controller.  It uses simulator-only block pose, contact, footprint
coverage, and velocity to make the approach/contact/push/settle transitions
observable in distilled trajectories.

All gains are synthetic engineering choices.  No physical robot samples,
system identification, or hardware validation are represented here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import mujoco
import numpy as np

from .production_env import ProductionEdgeArmEnv
from .sim2real_env_v6 import RealisticEdgeArmEnvV6


PHYSICAL_EXPERT_V7_VERSION = "edgearm-privileged-physical-expert-v7"
EXPERT_CLAIM_LEVEL = "L1_SYNTHETIC_HARDWARE_INSPIRED"
EXPERT_PARAMETER_SOURCE = "synthetic_expert_prior"
_JOINTS = 6


@dataclass(frozen=True)
class PhysicalExpertV7Config:
    """Synthetic controller constants, intentionally not hardware calibrated."""

    base_joint_scale: float = 0.30
    movable_joint_scale: float = 2.00
    contact_loss_recovery_steps: int = 3
    contact_recovery_along_m: float = 0.090
    contact_recovery_lateral_m: float = 0.060
    predictive_brake_margin_m: float = 0.004
    predictive_brake_horizon_steps: int = 2
    high_speed_brake_threshold_m_s: float = 0.080
    obstacle_clearance_margin_m: float = 0.055
    obstacle_waypoint_reached_m: float = 0.045

    def __post_init__(self) -> None:
        positive = (
            "base_joint_scale",
            "movable_joint_scale",
            "contact_loss_recovery_steps",
            "contact_recovery_along_m",
            "contact_recovery_lateral_m",
            "predictive_brake_horizon_steps",
            "high_speed_brake_threshold_m_s",
            "obstacle_clearance_margin_m",
            "obstacle_waypoint_reached_m",
        )
        for name in positive:
            value = float(getattr(self, name))
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if not np.isfinite(self.predictive_brake_margin_m) or self.predictive_brake_margin_m < 0:
            raise ValueError("predictive_brake_margin_m must be finite and non-negative")


class PhysicalClosedLoopExpertV7:
    """Privileged deterministic controller for ``RealisticEdgeArmEnvV6``.

    ``action`` accepts the current observation so collectors can keep the same
    call shape as learned policies.  Decisions intentionally use the underlying
    simulator state as an expert privilege; that dependency is exported in the
    returned metadata instead of being disguised as an observable VLA input.
    """

    teacher_type = "physical_expert_v7_synthetic_privileged"
    selected_update = 7
    checkpoint = ""

    def __init__(
        self,
        env: RealisticEdgeArmEnvV6 | None = None,
        config: PhysicalExpertV7Config | None = None,
    ) -> None:
        if env is not None and not isinstance(env, RealisticEdgeArmEnvV6):
            raise TypeError("PhysicalClosedLoopExpertV7 requires RealisticEdgeArmEnvV6")
        self.env = env
        self.config = config or PhysicalExpertV7Config()
        self._last_step_count = -1
        self._had_contact = False
        self._contact_loss_steps = 0
        self._recovery_count = 0
        self._avoid_side = 0.0
        self._avoid_stage = "direct"
        self._avoid_waypoint: np.ndarray | None = None

    def reset(self, env: RealisticEdgeArmEnvV6 | None = None) -> None:
        """Reset controller memory without changing the environment."""

        if env is not None:
            if not isinstance(env, RealisticEdgeArmEnvV6):
                raise TypeError("PhysicalClosedLoopExpertV7 requires RealisticEdgeArmEnvV6")
            self.env = env
        if self.env is None:
            raise RuntimeError("expert.reset requires a RealisticEdgeArmEnvV6")
        self._last_step_count = int(self.env.step_count)
        self._had_contact = False
        self._contact_loss_steps = 0
        self._recovery_count = 0
        self._avoid_side = 0.0
        self._avoid_stage = "uninitialized" if self.env.obstacle_enabled else "direct"
        self._avoid_waypoint = None

    def action(
        self,
        env_or_observation: RealisticEdgeArmEnvV6 | Mapping[str, np.ndarray] | None = None,
        observation: Mapping[str, np.ndarray] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Return one normalized six-joint command and provenance-rich metadata."""

        if isinstance(env_or_observation, RealisticEdgeArmEnvV6):
            self.env = env_or_observation
            current_observation = observation
        else:
            if observation is not None:
                raise TypeError("two-argument action requires the environment first")
            current_observation = env_or_observation
        if self.env is None:
            raise RuntimeError("expert.action requires a RealisticEdgeArmEnvV6")
        self._validate_observation(current_observation)
        if self.env.step_count < self._last_step_count or (
            self.env.step_count == 0 and self._last_step_count != 0
        ):
            self.reset()
        self._last_step_count = int(self.env.step_count)

        block = self.env.block_xy()
        target = self.env.target_xy.copy()
        tool = self.env.tool_xyz()
        direct_direction = self.env._unit(target - block)
        direction, obstacle_phase = self._route_direction(block, target, direct_direction)
        normal = np.array([-direction[1], direction[0]], dtype=np.float64)
        tool_to_block = block - tool[:2]
        along = float(np.dot(tool_to_block, direction))
        lateral = float(abs(np.dot(tool_to_block, normal)))
        contact_count = int(self.env._tool_block_contacts())
        contact = contact_count > 0
        coverage = float(self.env.block_target_coverage())
        linear_speed, angular_speed = self.env._block_speeds()
        block_velocity = self.env.data.qvel[
            self.env._block_dof_address : self.env._block_dof_address + 3
        ].copy()
        speed_toward_target = float(np.dot(block_velocity[:2], direction))

        if contact:
            self._had_contact = True
            self._contact_loss_steps = 0
        elif self._had_contact:
            self._contact_loss_steps += 1

        if obstacle_phase == "waypoint":
            raw_action, raw_metadata = self._operational_action(direction)
        else:
            raw_action, raw_metadata = self._direct_operational_action(direct_direction)
        route_phase = str(raw_metadata["phase"])
        recovery = bool(
            self._had_contact
            and not contact
            and (
                self._contact_loss_steps >= self.config.contact_loss_recovery_steps
                or along > self.config.contact_recovery_along_m
                or lateral > self.config.contact_recovery_lateral_m
            )
        )
        if recovery and self._contact_loss_steps == self.config.contact_loss_recovery_steps:
            self._recovery_count += 1

        strict_threshold = self.env.realism_config.strict_coverage_threshold
        delay_horizon = (
            self.env.command_delay_steps + self.config.predictive_brake_horizon_steps
        ) * self.env.control_dt
        projected_distance = self.env.distance_to_target() - max(speed_toward_target, 0.0) * delay_horizon
        predictive_brake = bool(
            coverage < strict_threshold
            and speed_toward_target > self.config.high_speed_brake_threshold_m_s
            and projected_distance
            <= self.env.realism_config.block_half_extent_m + self.config.predictive_brake_margin_m
        )

        if coverage >= strict_threshold:
            phase = "settle"
            action = np.zeros(_JOINTS, dtype=np.float64)
        else:
            # The V6 teacher attenuates its final push before the force-limited
            # plant has covered the remaining distance.  Starting from the raw
            # production operational-space command avoids that horizon-induced
            # stall.  The desk-contact-constrained base is deliberately
            # down-weighted while shoulder/elbow/wrist motion is accelerated.
            action = np.asarray(raw_action, dtype=np.float64).copy()
            action[0] *= self.config.base_joint_scale
            action[1:5] *= self.config.movable_joint_scale
            if predictive_brake:
                phase = "settle"
                action *= 0.0
            elif recovery:
                phase = "reposition"
            elif contact:
                phase = "push"
            elif obstacle_phase == "waypoint" or route_phase == "avoid":
                phase = "avoid"
            else:
                phase = "approach"

        action = np.clip(action, -1.0, 1.0)
        if phase == "settle":
            prefilter_reason = ""
        else:
            action, prefilter_reason = self._prefilter_action(action)
        action = action.astype(np.float32)
        obstacle_route = bool(self.env.obstacle_enabled and self.env._path_intersects_obstacle(block, target))
        metadata: dict[str, Any] = {
            "version": PHYSICAL_EXPERT_V7_VERSION,
            "claim_level": EXPERT_CLAIM_LEVEL,
            "parameter_source": EXPERT_PARAMETER_SOURCE,
            "physical_samples": 0,
            "physically_calibrated": False,
            "physical_hardware_connected": False,
            "tool_gripper_joint_position_rad": float(
                self.env.tool_gripper_joint_position_rad
            ),
            "privileged_state_used": [
                "block_pose",
                "target_pose",
                "tool_block_contact",
                "target_footprint_coverage",
                "block_linear_velocity",
                "block_angular_velocity",
                "simulated_encoder_noise",
            ],
            "phase": phase,
            "route_phase": route_phase,
            "contact_count": contact_count,
            "had_contact": self._had_contact,
            "contact_loss_steps": self._contact_loss_steps,
            "recovery_count": self._recovery_count,
            "strict_target_coverage": coverage,
            "block_linear_speed_m_s": linear_speed,
            "block_angular_speed_rad_s": angular_speed,
            "speed_toward_target_m_s": speed_toward_target,
            "projected_distance_m": projected_distance,
            "predictive_brake": predictive_brake,
            "workspace_prefilter_applied": bool(prefilter_reason),
            "workspace_prefilter_reason": prefilter_reason,
            "tool_block_along_m": along,
            "tool_block_lateral_m": lateral,
            "obstacle_route_active": obstacle_route,
            "obstacle_route_stage": self._avoid_stage,
            "obstacle_route_side": self._avoid_side,
            "obstacle_waypoint_xy": (None if self._avoid_waypoint is None else self._avoid_waypoint.copy()),
            "teacher_confidence": float(raw_metadata["teacher_confidence"]),
            "raw_joint_target": np.asarray(raw_metadata["joint_target"], dtype=np.float32),
            "normalized_action_linf": float(np.max(np.abs(action))),
        }
        return action, metadata

    def _route_direction(
        self,
        block: np.ndarray,
        target: np.ndarray,
        direct_direction: np.ndarray,
    ) -> tuple[np.ndarray, str]:
        """Return a fixed-side waypoint direction for obstacle episodes."""

        if not self.env.obstacle_enabled:
            self._avoid_stage = "direct"
            return direct_direction, "direct"
        if self._avoid_stage == "uninitialized":
            normal = np.array([-direct_direction[1], direct_direction[0]], dtype=np.float64)
            midpoint = 0.5 * (block + target)
            obstacle_offset = float(np.dot(self.env.obstacle_xy - midpoint, normal))
            if abs(obstacle_offset) > 1e-9:
                self._avoid_side = -float(np.sign(obstacle_offset))
            else:
                self._avoid_side = -1.0 if self.env.obstacle_xy[1] >= 0.0 else 1.0
            obstacle_half = self.env.model.geom_size[self.env._ids["obstacle_geom"], :2]
            clearance = (
                float(np.max(obstacle_half))
                + self.env.realism_config.block_half_extent_m
                + self.config.obstacle_clearance_margin_m
            )
            waypoint = self.env.obstacle_xy + normal * self._avoid_side * clearance
            waypoint[0] = np.clip(waypoint[0], 0.11, 0.41)
            waypoint[1] = np.clip(waypoint[1], -0.235, 0.235)
            self._avoid_waypoint = waypoint
            self._avoid_stage = "waypoint"
        if self._avoid_stage == "waypoint" and self._avoid_waypoint is not None:
            waypoint_distance = float(np.linalg.norm(self._avoid_waypoint - block))
            direct_clear = not self._segment_hits_inflated_obstacle(block, target)
            if waypoint_distance <= self.config.obstacle_waypoint_reached_m or direct_clear:
                # The transition is latched, so small contact-induced block
                # oscillations cannot flip the chosen side or re-enter avoid.
                self._avoid_stage = "target"
            else:
                return self.env._unit(self._avoid_waypoint - block), "waypoint"
        return direct_direction, "target"

    def _direct_operational_action(
        self,
        direction: np.ndarray,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Direct-route hook whose default preserves the original V7 teacher.

        ``direction`` is intentionally accepted even though the legacy
        production teacher reconstructs it from privileged state.  Later
        synthetic experts can parameterize the same contact geometry for both
        direct and obstacle-route segments without changing V7/V8 behavior.
        """

        del direction
        if self.env is None:  # pragma: no cover - guarded by ``action``
            raise RuntimeError("operational action requires an environment")
        return ProductionEdgeArmEnv.teacher_action(self.env)

    def _segment_hits_inflated_obstacle(self, start: np.ndarray, end: np.ndarray) -> bool:
        direction = end - start
        denominator = float(np.dot(direction, direction))
        if denominator < 1e-10:
            return False
        fraction = np.clip(np.dot(self.env.obstacle_xy - start, direction) / denominator, 0.0, 1.0)
        closest = start + fraction * direction
        obstacle_half = self.env.model.geom_size[self.env._ids["obstacle_geom"], :2]
        inflated = obstacle_half + self.env.realism_config.block_half_extent_m + 0.012
        return bool(np.all(np.abs(closest - self.env.obstacle_xy) <= inflated))

    def _operational_action(self, direction: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
        """Operational-space approach/push command for a fixed route segment."""

        block = self.env.block_xy()
        tool = self.env.tool_xyz()
        contact_xy = block - direction * 0.039
        offset = block - tool[:2]
        along = float(np.dot(offset, direction))
        lateral = float(abs(direction[0] * offset[1] - direction[1] * offset[0]))
        contact = self.env._tool_block_contacts() > 0
        phase = "push" if contact or (along < 0.063 and lateral < 0.045) else "approach"
        desired_xy = tool[:2] + direction * 0.021 if phase == "push" else contact_xy
        position_error = np.array([desired_xy[0], desired_xy[1], 0.076], dtype=np.float64) - tool
        position_norm = float(np.linalg.norm(position_error))
        if position_norm > 0.018:
            position_error *= 0.018 / position_norm

        rotation = self.env.data.site_xmat[self.env._ids["tool_site"]].reshape(3, 3)
        current_yaw = float(np.arctan2(rotation[1, 1], rotation[0, 1]))
        desired_yaw = float(np.arctan2(direction[1], direction[0]))
        yaw_error = (desired_yaw - current_yaw + np.pi) % (2 * np.pi) - np.pi
        yaw_error = float(np.clip(yaw_error, -0.16, 0.16))
        jacobian_position = np.zeros((3, self.env.model.nv))
        jacobian_rotation = np.zeros((3, self.env.model.nv))
        mujoco.mj_jacSite(
            self.env.model,
            self.env.data,
            jacobian_position,
            jacobian_rotation,
            self.env._ids["tool_site"],
        )
        rotation_weight = 0.18
        jacobian = np.vstack(
            [
                jacobian_position[:, :5],
                rotation_weight * jacobian_rotation[2:3, :5],
            ]
        )
        error = np.concatenate([position_error, [rotation_weight * yaw_error]])
        delta = jacobian.T @ np.linalg.solve(jacobian @ jacobian.T + 1.5e-3 * np.eye(4), error)
        joint_target = self.env.data.qpos[:_JOINTS].copy()
        joint_target[:5] += delta
        joint_target[5] = self.env.tool_gripper_joint_position_rad
        action = np.clip(
            (joint_target - self.env.data.qpos[:_JOINTS]) / self.env.config.max_joint_delta,
            -1.0,
            1.0,
        ).astype(np.float32)
        return action, {
            "phase": phase,
            "teacher_confidence": float(np.exp(-5.0 * min(position_norm, 0.5))),
            "joint_target": joint_target.astype(np.float32),
        }

    def _prefilter_action(self, action: np.ndarray) -> tuple[np.ndarray, str]:
        """Map the command to the environment's safe target before dispatch.

        V6 forms an absolute servo target from the physical joint position plus
        its simulated instantaneous encoder error.  Reusing that deterministic
        synthetic state here avoids emitting an unsafe command merely to have
        the environment map it to the same safe endpoint one line later.
        """

        current = self.env.data.qpos[:_JOINTS].copy()
        encoder_noise = self.env._encoder_position_noise.copy()
        requested = current + encoder_noise + action * self.env.config.max_joint_delta
        safe_target, reason = self.env._safety_filter(requested)
        safe_action = (safe_target - current - encoder_noise) / self.env.config.max_joint_delta
        return np.clip(safe_action, -1.0, 1.0), reason

    @staticmethod
    def _validate_observation(
        observation: Mapping[str, np.ndarray] | None,
    ) -> None:
        if observation is None:
            return
        required = {"joint_state": (12,), "task_vector": (5,), "tool_pose": (12,)}
        for name, shape in required.items():
            if name not in observation:
                raise ValueError(f"observation is missing {name}")
            value = np.asarray(observation[name])
            if value.shape != shape or not np.all(np.isfinite(value)):
                raise ValueError(f"observation[{name!r}] must be finite with shape {shape}")


def load_physical_expert_v7(
    checkpoint: object | None = None,
) -> PhysicalClosedLoopExpertV7:
    """Collector factory for the checkpoint-free deterministic expert."""

    if checkpoint is not None:
        raise ValueError("physical expert V7 has no learned checkpoint")
    return PhysicalClosedLoopExpertV7()


__all__ = [
    "EXPERT_CLAIM_LEVEL",
    "EXPERT_PARAMETER_SOURCE",
    "PHYSICAL_EXPERT_V7_VERSION",
    "PhysicalClosedLoopExpertV7",
    "PhysicalExpertV7Config",
    "load_physical_expert_v7",
]
