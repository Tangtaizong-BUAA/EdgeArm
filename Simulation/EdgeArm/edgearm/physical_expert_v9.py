"""Contact-acquisition geometry expert built on the V8 ``recovery_fast`` arm.

PhysicalExpertV9 changes only six operational-space geometry constants.  The
V8 recovery-fast state machine is frozen so the search can attribute paired
changes to contact acquisition rather than to another simultaneous controller
rewrite.  This remains a privileged synthetic teacher: it consumes simulator
state, contains no learned weights, and represents zero physical trials.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from types import MappingProxyType
from typing import Any, Mapping

import mujoco
import numpy as np

from .physical_expert_v7 import EXPERT_CLAIM_LEVEL
from .physical_expert_v8 import PhysicalClosedLoopExpertV8, PhysicalExpertV8Config
from .sim2real_env_v6 import RealisticEdgeArmEnvV6


PHYSICAL_EXPERT_V9_VERSION = "edgearm-privileged-physical-expert-v9"
EXPERT_V9_PARAMETER_SOURCE = "synthetic_contact_acquisition_geometry_search"
PHYSICAL_EXPERT_V9_CONFIG_FORMAT = "edgearm-deterministic-physical-expert-v9-config"
PHYSICAL_EXPERT_V9_CONFIG_SCHEMA_VERSION = 1
V9_GEOMETRY_PARAMETER_NAMES = (
    "contact_standoff_m",
    "tool_height_m",
    "approach_step_limit_m",
    "push_step_m",
    "push_gate_along_m",
    "push_gate_lateral_m",
)
V9_GEOMETRY_BOUNDS: Mapping[str, tuple[float, float]] = MappingProxyType(
    {
        "contact_standoff_m": (0.032, 0.046),
        "tool_height_m": (0.070, 0.082),
        "approach_step_limit_m": (0.014, 0.022),
        "push_step_m": (0.016, 0.026),
        "push_gate_along_m": (0.052, 0.072),
        "push_gate_lateral_m": (0.035, 0.055),
    }
)
V9_RECOVERY_FAST_FIXED_PARAMETERS: Mapping[str, float | int] = MappingProxyType(
    {
        "base_joint_scale": 0.30,
        "movable_joint_scale": 2.00,
        "contact_loss_recovery_steps": 2,
        "contact_recovery_along_m": 0.075,
        "contact_recovery_lateral_m": 0.045,
        "predictive_brake_margin_m": 0.004,
        "predictive_brake_horizon_steps": 2,
        "high_speed_brake_threshold_m_s": 0.080,
        "obstacle_clearance_margin_m": 0.055,
        "obstacle_waypoint_reached_m": 0.045,
        "contact_recovery_action_scale": 1.30,
        "near_target_coverage_start": 0.94,
        "minimum_push_scale": 1.0,
        "settle_retract_step_m": 0.004,
        "settle_lift_step_m": 0.0015,
        "settle_coverage_hysteresis": 0.04,
        "settle_recovery_scale": 0.25,
        "settle_hold_release_gain": 0.01,
        "settle_hold_decay_steps": 4.0,
        "unload_speed_gain": 0.30,
        "unload_contact_speed_gate_m_s": 0.008,
        "predictive_reverse_gain": 0.30,
    }
)
_JOINTS = 6


def recovery_fast_v8_config() -> PhysicalExpertV8Config:
    """Return the exact preregistered V8 ``recovery_fast`` comparator."""

    return PhysicalExpertV8Config(
        contact_loss_recovery_steps=2,
        contact_recovery_along_m=0.075,
        contact_recovery_lateral_m=0.045,
        contact_recovery_action_scale=1.30,
        near_target_coverage_start=0.94,
        minimum_push_scale=1.0,
        settle_recovery_scale=0.25,
        settle_hold_release_gain=0.01,
        predictive_reverse_gain=0.30,
    )


@dataclass(frozen=True)
class PhysicalExpertV9Config(PhysicalExpertV8Config):
    """V8 recovery-fast plus bounded contact-acquisition geometry."""

    contact_loss_recovery_steps: int = 2
    contact_recovery_along_m: float = 0.075
    contact_recovery_lateral_m: float = 0.045
    contact_recovery_action_scale: float = 1.30
    near_target_coverage_start: float = 0.94
    minimum_push_scale: float = 1.0
    settle_recovery_scale: float = 0.25
    settle_hold_release_gain: float = 0.01
    predictive_reverse_gain: float = 0.30
    contact_standoff_m: float = 0.039
    tool_height_m: float = 0.076
    approach_step_limit_m: float = 0.018
    push_step_m: float = 0.021
    push_gate_along_m: float = 0.063
    push_gate_lateral_m: float = 0.045

    def __post_init__(self) -> None:
        super().__post_init__()
        allowed = set(V9_RECOVERY_FAST_FIXED_PARAMETERS) | set(V9_GEOMETRY_PARAMETER_NAMES)
        actual = {field.name for field in fields(self)}
        if actual != allowed:
            raise RuntimeError(
                "PhysicalExpertV9Config contains an unclassified searchable/frozen field: "
                f"missing={sorted(actual - allowed)} stale={sorted(allowed - actual)}"
            )
        for name, (lower, upper) in V9_GEOMETRY_BOUNDS.items():
            value = float(getattr(self, name))
            if not np.isfinite(value) or not lower <= value <= upper:
                raise ValueError(f"{name} must be finite and in [{lower}, {upper}]")
        for name, expected in V9_RECOVERY_FAST_FIXED_PARAMETERS.items():
            actual = getattr(self, name)
            if isinstance(expected, int):
                matches = actual == expected
            else:
                matches = bool(np.isclose(float(actual), expected, rtol=0.0, atol=1e-12))
            if not matches:
                raise ValueError(f"{name} is frozen to recovery_fast value {expected}; got {actual}")


class PhysicalClosedLoopExpertV9(PhysicalClosedLoopExpertV8):
    """V8 terminal/recovery controller with parameterized contact geometry."""

    teacher_type = "physical_expert_v9_synthetic_privileged"
    selected_update = 9
    checkpoint = ""

    def __init__(
        self,
        env: RealisticEdgeArmEnvV6 | None = None,
        config: PhysicalExpertV9Config | None = None,
    ) -> None:
        self.v9_config = config or PhysicalExpertV9Config()
        super().__init__(env=env, config=self.v9_config)

    def action(
        self,
        env_or_observation: RealisticEdgeArmEnvV6 | Mapping[str, np.ndarray] | None = None,
        observation: Mapping[str, np.ndarray] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        action, metadata = super().action(env_or_observation, observation)
        metadata.update(
            {
                "version": PHYSICAL_EXPERT_V9_VERSION,
                "parameter_source": EXPERT_V9_PARAMETER_SOURCE,
                "geometry_parameters": {
                    name: float(getattr(self.v9_config, name)) for name in V9_GEOMETRY_PARAMETER_NAMES
                },
                "recovery_fast_parameters_frozen": {
                    name: getattr(self.v9_config, name) for name in V9_RECOVERY_FAST_FIXED_PARAMETERS
                },
                "physical_samples": 0,
                "physical_trials": 0,
                "physically_calibrated": False,
                "physical_hardware_connected": False,
                "tool_gripper_joint_position_rad": float(
                    self.env.tool_gripper_joint_position_rad
                ),
                "claim_level": EXPERT_CLAIM_LEVEL,
            }
        )
        return action, metadata

    def _direct_operational_action(
        self,
        direction: np.ndarray,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        # The legacy direct teacher gates push on geometry only.  Keeping this
        # fixed distinction and its environment-level obstacle detour is
        # required for trajectory identity with V8.
        return self._geometry_operational_action(
            direction,
            contact_opens_gate=False,
            environment_obstacle_detour=True,
        )

    def _operational_action(self, direction: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
        # The legacy obstacle-waypoint controller also accepts actual contact.
        return self._geometry_operational_action(
            direction,
            contact_opens_gate=True,
            environment_obstacle_detour=False,
        )

    def _geometry_operational_action(
        self,
        direction: np.ndarray,
        *,
        contact_opens_gate: bool,
        environment_obstacle_detour: bool,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Construct the same V7 command with six searchable geometry values."""

        if self.env is None:  # pragma: no cover - guarded by ``action``
            raise RuntimeError("operational action requires an environment")
        config = self.v9_config
        direction = np.asarray(direction, dtype=np.float64)
        if direction.shape != (2,) or not np.all(np.isfinite(direction)):
            raise ValueError("direction must be a finite two-vector")
        block = self.env.block_xy()
        tool = self.env.tool_xyz()
        contact_xy = block - direction * config.contact_standoff_m
        offset = block - tool[:2]
        along = float(np.dot(offset, direction))
        lateral = float(abs(direction[0] * offset[1] - direction[1] * offset[0]))
        contact = self.env._tool_block_contacts() > 0
        push_gate = bool(
            (contact_opens_gate and contact)
            or (along < config.push_gate_along_m and lateral < config.push_gate_lateral_m)
        )
        phase = "push" if push_gate else "approach"
        desired_xy = tool[:2] + direction * config.push_step_m if push_gate else contact_xy
        if (
            environment_obstacle_detour
            and self.env.obstacle_enabled
            and self.env._path_intersects_obstacle(block, self.env.target_xy)
        ):
            normal = np.array([-direction[1], direction[0]])
            side = np.sign(np.dot(block - self.env.obstacle_xy, normal)) or 1.0
            waypoint = self.env.obstacle_xy + normal * side * 0.095
            local = self.env._unit(waypoint - block)
            desired_xy = block - local * config.contact_standoff_m
            direction = local
            phase = "avoid"
            if np.linalg.norm(tool[:2] - desired_xy) < 0.025:
                # Fixed legacy obstacle advance; it is intentionally not a
                # seventh searchable geometry parameter.
                desired_xy = tool[:2] + local * 0.018
        target_xyz = np.array(
            [desired_xy[0], desired_xy[1], config.tool_height_m],
            dtype=np.float64,
        )
        position_error = target_xyz - tool
        position_norm = float(np.linalg.norm(position_error))
        if position_norm > config.approach_step_limit_m:
            position_error *= config.approach_step_limit_m / position_norm

        rotation = self.env.data.site_xmat[self.env._ids["tool_site"]].reshape(3, 3)
        current_yaw = float(np.arctan2(rotation[1, 1], rotation[0, 1]))
        desired_yaw = float(np.arctan2(direction[1], direction[0]))
        yaw_error = (desired_yaw - current_yaw + np.pi) % (2 * np.pi) - np.pi
        yaw_error = float(np.clip(yaw_error, -0.16, 0.16))
        jacobian_position = np.zeros((3, self.env.model.nv), dtype=np.float64)
        jacobian_rotation = np.zeros((3, self.env.model.nv), dtype=np.float64)
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
        delta = jacobian.T @ np.linalg.solve(
            jacobian @ jacobian.T + 1.5e-3 * np.eye(4),
            error,
        )
        joint_target = self.env.data.qpos[:_JOINTS].copy()
        joint_target[:5] += delta
        joint_target[5] = self.env.tool_gripper_joint_position_rad
        normalized = np.clip(
            (joint_target - self.env.data.qpos[:_JOINTS]) / self.env.config.max_joint_delta,
            -1.0,
            1.0,
        ).astype(np.float32)
        return normalized, {
            "phase": phase,
            "teacher_confidence": float(np.exp(-5.0 * min(position_norm, 0.5))),
            "joint_target": joint_target.astype(np.float32),
            "contact_geometry_push_gate": push_gate,
            "contact_geometry_contact_opens_gate": contact_opens_gate,
            "contact_geometry_environment_obstacle_detour": environment_obstacle_detour,
            "contact_geometry_along_m": along,
            "contact_geometry_lateral_m": lateral,
        }


def load_physical_expert_v9(
    checkpoint: object | None = None,
    *,
    config: PhysicalExpertV9Config | None = None,
) -> PhysicalClosedLoopExpertV9:
    """Factory matching collector teacher interfaces."""

    if checkpoint is not None:
        raise ValueError("physical expert V9 has no learned checkpoint")
    return PhysicalClosedLoopExpertV9(config=config)


__all__ = [
    "EXPERT_V9_PARAMETER_SOURCE",
    "PHYSICAL_EXPERT_V9_CONFIG_FORMAT",
    "PHYSICAL_EXPERT_V9_CONFIG_SCHEMA_VERSION",
    "PHYSICAL_EXPERT_V9_VERSION",
    "PhysicalClosedLoopExpertV9",
    "PhysicalExpertV9Config",
    "V9_GEOMETRY_BOUNDS",
    "V9_GEOMETRY_PARAMETER_NAMES",
    "V9_RECOVERY_FAST_FIXED_PARAMETERS",
    "load_physical_expert_v9",
    "recovery_fast_v8_config",
]
