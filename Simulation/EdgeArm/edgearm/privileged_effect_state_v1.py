"""Source-neutral 163-D privileged effect state for V7 scratch PPO.

This is a *teacher-only* simulator state.  It is intentionally not an ACT/VLA
deployment observation: physical joints, actuator internals, pending commands,
object truth, and obstacle truth are privileged MuJoCo quantities.  The state
contains no expert action, plan, phase, residual, or behavior-cloning signal.

The first 159 dimensions reconstruct the complete V6 execution/effect state in
an independent module.  V7 adds four task-critical values explicitly rather
than padding: obstacle enabled (1), obstacle world position (2), and remaining
episode horizon (1).  Every field, source, and normalization rule participates
in ``PRIVILEGED_EFFECT_STATE_SCHEMA_SHA256``.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass

import numpy as np

from .sim2real_env_v7 import RealisticEdgeArmEnvV7


PRIVILEGED_EFFECT_STATE_SCHEMA_VERSION = "edgearm-v7-privileged-effect-state-v1"
PRIVILEGED_EFFECT_STATE_DIM = 163
ACTION_DIM = 6
MAX_QUEUE_SLOTS = 5


@dataclass(frozen=True)
class PrivilegedEffectStateFieldV1:
    """One contiguous field in the teacher-state vector."""

    name: str
    size: int
    simulator_source: str
    normalization: str


PRIVILEGED_EFFECT_STATE_FIELDS_V1 = (
    PrivilegedEffectStateFieldV1("physical_joint_position_rad", 6, "data.qpos[:6]", "none"),
    PrivilegedEffectStateFieldV1("physical_joint_velocity_rad_s", 6, "data.qvel[:6]", "none"),
    PrivilegedEffectStateFieldV1(
        "reported_joint_position_rad", 6, "observation.joint_state[:6]", "none"
    ),
    PrivilegedEffectStateFieldV1(
        "reported_joint_velocity_rad_s", 6, "observation.joint_state[6:]", "none"
    ),
    PrivilegedEffectStateFieldV1("servo_velocity_rad_s", 6, "_servo_velocity", "none"),
    PrivilegedEffectStateFieldV1(
        "backlash_over_max_delta", 6, "_backlash", "divide_by_max_joint_delta"
    ),
    PrivilegedEffectStateFieldV1(
        "backlash_remaining_over_max_delta",
        6,
        "_backlash_remaining",
        "divide_by_max_joint_delta",
    ),
    PrivilegedEffectStateFieldV1("last_motor_direction", 6, "_last_motor_direction", "none"),
    PrivilegedEffectStateFieldV1(
        "deadband_over_max_delta", 6, "_deadband", "divide_by_max_joint_delta"
    ),
    PrivilegedEffectStateFieldV1(
        "zero_offset_over_max_delta", 6, "_zero_offset", "divide_by_max_joint_delta"
    ),
    PrivilegedEffectStateFieldV1(
        "queued_target_error_over_max_delta",
        30,
        "_command_queue[0:5]-data.qpos[:6]",
        "divide_by_max_joint_delta_then_clip_-4_4",
    ),
    PrivilegedEffectStateFieldV1(
        "queued_target_valid_mask", 5, "len(_command_queue)", "binary_prefix_mask"
    ),
    PrivilegedEffectStateFieldV1(
        "command_delay_over_five", 1, "command_delay_steps", "divide_by_5"
    ),
    PrivilegedEffectStateFieldV1("servo_response", 1, "_servo_response", "none"),
    PrivilegedEffectStateFieldV1(
        "supply_voltage_over_nominal",
        1,
        "_supply_voltage_v",
        "divide_by_nominal_supply_voltage_v",
    ),
    PrivilegedEffectStateFieldV1(
        "loaded_voltage_over_nominal",
        1,
        "_loaded_voltage_v",
        "divide_by_nominal_supply_voltage_v",
    ),
    PrivilegedEffectStateFieldV1(
        "motor_temperature_over_shutdown",
        6,
        "_motor_temperature_c",
        "divide_by_thermal_shutdown_c",
    ),
    PrivilegedEffectStateFieldV1(
        "thermal_derating_start_over_shutdown",
        6,
        "_thermal_derating_start_c",
        "divide_by_thermal_shutdown_c",
    ),
    PrivilegedEffectStateFieldV1("actuator_force_nm", 6, "_last_actuator_force", "none"),
    PrivilegedEffectStateFieldV1(
        "last_actual_velocity_rad_s", 6, "_last_actual_velocity", "none"
    ),
    PrivilegedEffectStateFieldV1(
        "last_command_lost_flag", 1, "_last_command_lost", "bool_to_float"
    ),
    PrivilegedEffectStateFieldV1(
        "command_burst_remaining_over_max",
        1,
        "_command_burst_remaining",
        "divide_by_command_burst_length_range_max",
    ),
    PrivilegedEffectStateFieldV1(
        "block_pose_xyz_quaternion_wxyz", 7, "block_free_joint_qpos", "none"
    ),
    PrivilegedEffectStateFieldV1(
        "block_velocity_linear_angular", 6, "block_free_joint_qvel", "none"
    ),
    PrivilegedEffectStateFieldV1("target_xy_m", 2, "target_xy", "none"),
    PrivilegedEffectStateFieldV1(
        "tool_pose_position_rotation", 12, "observation.tool_pose", "none"
    ),
    PrivilegedEffectStateFieldV1(
        "strict_target_coverage", 1, "block_target_coverage()", "none"
    ),
    PrivilegedEffectStateFieldV1(
        "strict_success_streak_over_hold_steps",
        1,
        "_strict_success_streak",
        "divide_by_strict_success_hold_steps",
    ),
    PrivilegedEffectStateFieldV1(
        "block_target_distance_m", 1, "distance_to_target()", "none"
    ),
    PrivilegedEffectStateFieldV1(
        "block_linear_angular_speed", 2, "_block_speeds()", "none"
    ),
    PrivilegedEffectStateFieldV1(
        "tool_block_contact_count", 1, "_tool_block_contacts()", "none"
    ),
    PrivilegedEffectStateFieldV1("stress_flag", 1, "current_stress", "bool_to_float"),
    PrivilegedEffectStateFieldV1(
        "obstacle_enabled_flag", 1, "obstacle_enabled", "bool_to_float"
    ),
    PrivilegedEffectStateFieldV1("obstacle_xy_m", 2, "obstacle_xy", "none"),
    PrivilegedEffectStateFieldV1(
        "remaining_horizon_fraction",
        1,
        "config.max_steps-step_count",
        "divide_by_config.max_steps",
    ),
)


def _schema_sha256() -> str:
    payload = {
        "schema_version": PRIVILEGED_EFFECT_STATE_SCHEMA_VERSION,
        "dimension": PRIVILEGED_EFFECT_STATE_DIM,
        "dtype": "float32",
        "fields": [asdict(field) for field in PRIVILEGED_EFFECT_STATE_FIELDS_V1],
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


if sum(field.size for field in PRIVILEGED_EFFECT_STATE_FIELDS_V1) != PRIVILEGED_EFFECT_STATE_DIM:
    raise RuntimeError("privileged effect state fields do not sum to 163")

PRIVILEGED_EFFECT_STATE_SCHEMA_SHA256 = _schema_sha256()
PRIVILEGED_EFFECT_STATE_LAYOUT_V1 = tuple(
    f"{field.name}[{field.size}]" for field in PRIVILEGED_EFFECT_STATE_FIELDS_V1
)


def privileged_effect_state_slices_v1() -> dict[str, slice]:
    """Return the exact, non-overlapping slice assigned to every field."""

    result: dict[str, slice] = {}
    start = 0
    for field in PRIVILEGED_EFFECT_STATE_FIELDS_V1:
        result[field.name] = slice(start, start + field.size)
        start += field.size
    if start != PRIVILEGED_EFFECT_STATE_DIM:  # pragma: no cover - import guard above
        raise RuntimeError("privileged state slice construction drifted")
    return result


def _vector(value: object, size: int, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (size,) or not np.all(np.isfinite(result)):
        raise RuntimeError(f"invalid simulator field {name}: expected finite ({size},), got {result.shape}")
    return result


def build_privileged_effect_state_v1(env: RealisticEdgeArmEnvV7) -> np.ndarray:
    """Build the exact 163-D teacher input from one live V7 simulator state."""

    if not isinstance(env, RealisticEdgeArmEnvV7):
        raise TypeError("privileged effect state v1 requires RealisticEdgeArmEnvV7")
    max_delta = float(env.config.max_joint_delta)
    max_steps = int(env.config.max_steps)
    if not np.isfinite(max_delta) or max_delta <= 0.0:
        raise RuntimeError("V7 max_joint_delta must be finite and positive")
    if max_steps < 1 or not 0 <= int(env.step_count) <= max_steps:
        raise RuntimeError("V7 step_count must be inside the configured episode horizon")

    observation = env.observation()
    reported = _vector(observation.get("joint_state"), 12, "observation.joint_state")
    tool_pose = _vector(observation.get("tool_pose"), 12, "observation.tool_pose")
    physical_q = _vector(env.data.qpos[:ACTION_DIM], ACTION_DIM, "data.qpos[:6]")
    physical_dq = _vector(env.data.qvel[:ACTION_DIM], ACTION_DIM, "data.qvel[:6]")

    queue = list(env._command_queue)
    if len(queue) > MAX_QUEUE_SLOTS or int(env.command_delay_steps) > MAX_QUEUE_SLOTS:
        raise RuntimeError("V7 command queue or delay exceeds the five-slot state contract")
    queue_error = np.zeros((MAX_QUEUE_SLOTS, ACTION_DIM), dtype=np.float64)
    queue_mask = np.zeros(MAX_QUEUE_SLOTS, dtype=np.float64)
    for index, target in enumerate(queue):
        queue_target = _vector(target, ACTION_DIM, f"_command_queue[{index}]")
        queue_error[index] = np.clip((queue_target - physical_q) / max_delta, -4.0, 4.0)
        queue_mask[index] = 1.0

    block_qadr = int(env.model.jnt_qposadr[env._ids["block_joint"]])
    block_dadr = int(env.model.jnt_dofadr[env._ids["block_joint"]])
    block_pose = _vector(env.data.qpos[block_qadr : block_qadr + 7], 7, "block_free_joint_qpos")
    block_velocity = _vector(
        env.data.qvel[block_dadr : block_dadr + 6], 6, "block_free_joint_qvel"
    )
    linear_speed, angular_speed = env._block_speeds()
    nominal_voltage = float(env.realism_config.nominal_supply_voltage_v)
    shutdown_temperature = float(env.realism_config.thermal_shutdown_c)
    burst_max = max(int(env.realism_config.command_burst_length_range[1]), 1)
    hold_steps = max(int(env.realism_config.strict_success_hold_steps), 1)
    if nominal_voltage <= 0.0 or shutdown_temperature <= 0.0:
        raise RuntimeError("V7 voltage and thermal normalizers must be positive")

    pieces = (
        physical_q,
        physical_dq,
        reported[:ACTION_DIM],
        reported[ACTION_DIM:],
        _vector(env._servo_velocity, 6, "_servo_velocity"),
        _vector(env._backlash, 6, "_backlash") / max_delta,
        _vector(env._backlash_remaining, 6, "_backlash_remaining") / max_delta,
        _vector(env._last_motor_direction, 6, "_last_motor_direction"),
        _vector(env._deadband, 6, "_deadband") / max_delta,
        _vector(env._zero_offset, 6, "_zero_offset") / max_delta,
        queue_error.reshape(-1),
        queue_mask,
        np.array([env.command_delay_steps / MAX_QUEUE_SLOTS], dtype=np.float64),
        np.array([env._servo_response], dtype=np.float64),
        np.array([env._supply_voltage_v / nominal_voltage], dtype=np.float64),
        np.array([env._loaded_voltage_v / nominal_voltage], dtype=np.float64),
        _vector(env._motor_temperature_c, 6, "_motor_temperature_c") / shutdown_temperature,
        _vector(env._thermal_derating_start_c, 6, "_thermal_derating_start_c")
        / shutdown_temperature,
        _vector(env._last_actuator_force, 6, "_last_actuator_force"),
        _vector(env._last_actual_velocity, 6, "_last_actual_velocity"),
        np.array([float(env._last_command_lost)], dtype=np.float64),
        np.array([env._command_burst_remaining / burst_max], dtype=np.float64),
        block_pose,
        block_velocity,
        _vector(env.target_xy, 2, "target_xy"),
        tool_pose,
        np.array([env.block_target_coverage()], dtype=np.float64),
        np.array([env._strict_success_streak / hold_steps], dtype=np.float64),
        np.array([env.distance_to_target()], dtype=np.float64),
        np.array([linear_speed, angular_speed], dtype=np.float64),
        np.array([env._tool_block_contacts()], dtype=np.float64),
        np.array([float(env.current_stress)], dtype=np.float64),
        np.array([float(env.obstacle_enabled)], dtype=np.float64),
        _vector(env.obstacle_xy, 2, "obstacle_xy"),
        np.array([(max_steps - env.step_count) / max_steps], dtype=np.float64),
    )
    state = np.concatenate(pieces).astype(np.float32)
    if state.shape != (PRIVILEGED_EFFECT_STATE_DIM,) or not np.all(np.isfinite(state)):
        raise RuntimeError(f"invalid privileged effect state: {state.shape}")
    return state
