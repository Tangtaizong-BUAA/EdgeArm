"""Opt-in V6 dynamics and task semantics for harder virtual robustness work.

V1 and V2 remain frozen for reproducibility.  V6 removes the forced joint-state
write used by those historical paths: delayed, rate-limited commands are sent
to MuJoCo's force-limited SO-101 position actuators and contact/load dynamics
decide the state that is reached.  It also fixes mass/inertia and effective
contact randomization, models an uncalibrated wrist payload, and separates the
legacy task score from a stricter containment-and-settling success criterion.

Every range in this module is an uncalibrated engineering prior.  No physical
SO-101, servo, wrist camera, material, power, or timing measurement is implied.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from .production_env import (
    LEGACY_PUSH_PLATE_TOOL_PROFILE,
    STOCK_GRIPPER_CONTACT_PAIR_NAMES,
    STOCK_GRIPPER_DIRECT_PUSH_TOOL_PROFILE,
    ProductionMjcfBundleV1,
    build_production_model,
)
from .sim2real_env import Sim2RealEdgeArmEnv, Sim2RealEnvConfig, _JOINTS, _validate_range


REALISTIC_DYNAMICS_PROFILE_VERSION = "edgearm-sim2real-dynamics-v6"
REALISM_CLAIM_LEVEL = "L1_SYNTHETIC_HARDWARE_INSPIRED"
PARAMETER_SOURCE = "uncalibrated_prior"


def _quat_multiply(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """Multiply scalar-first quaternions and return a normalized quaternion."""

    w1, x1, y1, z1 = np.asarray(first, dtype=np.float64)
    w2, x2, y2, z2 = np.asarray(second, dtype=np.float64)
    result = np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dtype=np.float64,
    )
    return result / max(float(np.linalg.norm(result)), 1e-12)


def _rotation_vector_quat(rotation_vector: np.ndarray) -> np.ndarray:
    angle = float(np.linalg.norm(rotation_vector))
    if angle < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    axis = np.asarray(rotation_vector, dtype=np.float64) / angle
    return np.concatenate([[np.cos(angle / 2.0)], axis * np.sin(angle / 2.0)])


@dataclass(frozen=True)
class RealisticEnvV6Config(Sim2RealEnvConfig):
    """Synthetic V6 priors; all values remain uncalibrated to physical hardware."""

    physics_substeps: int = 8
    max_steps: int = 96
    command_delay_steps_range: tuple[int, int] = (0, 4)
    servo_response_range: tuple[float, float] = (0.65, 0.98)
    actuator_kp_scale_range: tuple[float, float] = (0.65, 1.15)
    actuator_kv_scale_range: tuple[float, float] = (0.70, 1.35)
    actuator_force_scale_range: tuple[float, float] = (0.68, 1.05)
    joint_damping_scale_range: tuple[float, float] = (0.75, 1.40)
    joint_armature_scale_range: tuple[float, float] = (0.75, 1.50)
    supply_voltage_range_v: tuple[float, float] = (10.5, 12.6)
    nominal_supply_voltage_v: float = 12.0
    voltage_sag_per_unit_effort_v: float = 0.55
    ambient_temperature_range_c: tuple[float, float] = (18.0, 32.0)
    initial_motor_temperature_rise_range_c: tuple[float, float] = (1.0, 8.0)
    thermal_derating_start_range_c: tuple[float, float] = (45.0, 60.0)
    thermal_shutdown_c: float = 78.0
    thermal_heating_rate_c_s: float = 1.8
    thermal_cooling_rate_s: float = 0.022
    command_loss_probability: float = 0.018
    command_burst_start_probability: float = 0.008
    command_burst_length_range: tuple[int, int] = (2, 5)
    encoder_ticks_per_revolution: int = 4096
    block_mass_scale_range: tuple[float, float] = (0.55, 1.65)
    block_com_jitter_range_m: tuple[float, float] = (-0.0015, 0.0015)
    desk_block_slide_friction_range: tuple[float, float] = (0.35, 1.20)
    desk_block_torsional_friction_range: tuple[float, float] = (0.006, 0.030)
    desk_block_rolling_friction_range: tuple[float, float] = (0.0005, 0.0040)
    pusher_block_slide_friction_range: tuple[float, float] = (0.65, 1.35)
    contact_time_constant_range_s: tuple[float, float] = (0.004, 0.014)
    wrist_payload_mass_range_kg: tuple[float, float] = (0.025, 0.060)
    wrist_payload_com_range_m: tuple[float, float] = (0.035, 0.070)
    camera_mount_rotation_std_rad: float = 0.018
    camera_mount_rotation_limit_rad: float = 0.055
    camera_flex_translation_per_rad_s_m: float = 0.00018
    target_radius_m: float = 0.055
    block_half_extent_m: float = 0.025
    strict_coverage_threshold: float = 0.95
    strict_linear_speed_m_s: float = 0.025
    strict_angular_speed_rad_s: float = 0.55
    strict_success_hold_steps: int = 6
    obstacle_collision_terminates: bool = True

    def __post_init__(self) -> None:
        super().__post_init__()
        for name in (
            "actuator_kp_scale_range",
            "actuator_kv_scale_range",
            "actuator_force_scale_range",
            "joint_damping_scale_range",
            "joint_armature_scale_range",
            "supply_voltage_range_v",
            "ambient_temperature_range_c",
            "initial_motor_temperature_rise_range_c",
            "thermal_derating_start_range_c",
            "block_mass_scale_range",
            "desk_block_slide_friction_range",
            "desk_block_torsional_friction_range",
            "desk_block_rolling_friction_range",
            "pusher_block_slide_friction_range",
            "contact_time_constant_range_s",
            "wrist_payload_mass_range_kg",
            "wrist_payload_com_range_m",
        ):
            _validate_range(name, getattr(self, name), nonnegative=True)
        _validate_range("block_com_jitter_range_m", self.block_com_jitter_range_m)
        for name in (
            "nominal_supply_voltage_v",
            "thermal_shutdown_c",
            "encoder_ticks_per_revolution",
            "target_radius_m",
            "block_half_extent_m",
            "strict_success_hold_steps",
        ):
            if float(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive")
        for name in (
            "voltage_sag_per_unit_effort_v",
            "thermal_heating_rate_c_s",
            "thermal_cooling_rate_s",
            "camera_mount_rotation_std_rad",
            "camera_mount_rotation_limit_rad",
            "camera_flex_translation_per_rad_s_m",
            "strict_linear_speed_m_s",
            "strict_angular_speed_rad_s",
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")
        for name in (
            "command_loss_probability",
            "command_burst_start_probability",
            "strict_coverage_threshold",
        ):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be within [0, 1]")
        burst = self.command_burst_length_range
        if (
            len(burst) != 2
            or not all(isinstance(value, int) for value in burst)
            or burst[0] <= 0
            or burst[0] > burst[1]
        ):
            raise ValueError("command_burst_length_range must be an ordered positive integer pair")
        if self.thermal_shutdown_c <= self.thermal_derating_start_range_c[1]:
            raise ValueError("thermal_shutdown_c must exceed the derating range")


class RealisticEdgeArmEnvV6(Sim2RealEdgeArmEnv):
    """Force-limited, contact-coupled SO-101 virtual environment."""

    profile_version = REALISTIC_DYNAMICS_PROFILE_VERSION
    model_tool_profile = LEGACY_PUSH_PLATE_TOOL_PROFILE
    include_stock_tip_references = False

    def __init__(
        self,
        config: RealisticEnvV6Config | None = None,
        seed: int = 0,
        *,
        model_scene_path: Path | None = None,
        model_scene_bundle: ProductionMjcfBundleV1 | None = None,
    ):
        self.realism_config = config or RealisticEnvV6Config()
        self._model_scene_path = (
            None if model_scene_path is None else Path(model_scene_path)
        )
        if model_scene_path is not None and model_scene_bundle is not None:
            raise ValueError("model_scene_path and model_scene_bundle are mutually exclusive")
        self._model_scene_bundle = model_scene_bundle
        self._realism_ready = False
        self._v6_rng = np.random.default_rng(seed ^ 0x6D1A6)
        self._command_burst_remaining = 0
        self._last_command_lost = False
        self._strict_success_streak = 0
        self._motor_temperature_c = np.full(_JOINTS, 25.0, dtype=np.float64)
        self._thermal_derating_start_c = np.full(_JOINTS, 55.0, dtype=np.float64)
        self._supply_voltage_v = 12.0
        self._loaded_voltage_v = 12.0
        self._last_actuator_force = np.zeros(_JOINTS, dtype=np.float64)
        self._last_actual_velocity = np.zeros(_JOINTS, dtype=np.float64)
        self._camera_episode_pos = np.zeros(3, dtype=np.float64)
        self._camera_episode_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        self._episode_v6: dict[str, Any] = {}
        super().__init__(self.realism_config, seed=seed)

        self._desk_geom = self._name_id(mujoco.mjtObj.mjOBJ_GEOM, "edgearm_desk")
        self._floor_geom = self._name_id(mujoco.mjtObj.mjOBJ_GEOM, "floor")
        self._camera_housing_geom = self._name_id(mujoco.mjtObj.mjOBJ_GEOM, "production_wrist_camera_housing")
        self._clutter_geoms = [
            self._name_id(mujoco.mjtObj.mjOBJ_GEOM, name)
            for name in (
                "realism_v6_clutter_left",
                "realism_v6_clutter_right",
                "realism_v6_clutter_rear",
            )
        ]
        self._surface_materials = [
            self._name_id(mujoco.mjtObj.mjOBJ_MATERIAL, f"realism_v6_surface_{index}") for index in range(6)
        ]
        pusher_pair_names = (
            STOCK_GRIPPER_CONTACT_PAIR_NAMES
            if self._ids["tool_profile"] == STOCK_GRIPPER_DIRECT_PUSH_TOOL_PROFILE
            else ("realism_v6_pusher_block",)
        )
        pair_names = (
            "realism_v6_desk_block",
            *pusher_pair_names,
            "realism_v6_obstacle_camera",
        )
        self._pair_ids = {
            name: self._name_id(mujoco.mjtObj.mjOBJ_PAIR, name) for name in pair_names
        }
        self._pusher_block_pair_ids = tuple(
            self._pair_ids[name] for name in pusher_pair_names
        )
        self._block_dof_address = int(self.model.jnt_dofadr[self._ids["block_joint"]])
        self._default_block_inertia = self.model.body_inertia[self._ids["block_body"]].copy()
        self._default_block_ipos = self.model.body_ipos[self._ids["block_body"]].copy()
        self._default_gripper_mass = float(self.model.body_mass[self._ids["gripper_body"]])
        self._default_gripper_inertia = self.model.body_inertia[self._ids["gripper_body"]].copy()
        self._default_gripper_ipos = self.model.body_ipos[self._ids["gripper_body"]].copy()
        self._default_actuator_gain = self.model.actuator_gainprm[:, 0].copy()
        self._default_actuator_kv = -self.model.actuator_biasprm[:, 2].copy()
        self._default_actuator_forcerange = self.model.actuator_forcerange.copy()
        self._default_dof_damping = self.model.dof_damping[:_JOINTS].copy()
        self._default_dof_armature = self.model.dof_armature[:_JOINTS].copy()
        self._default_desk_rgba = self.model.geom_rgba[self._desk_geom].copy()
        self._default_surface_rgba = {
            material_id: self.model.mat_rgba[material_id].copy() for material_id in self._surface_materials
        }
        self._default_clutter_pos = {
            geom_id: self.model.geom_pos[geom_id].copy() for geom_id in self._clutter_geoms
        }
        self._default_clutter_rgba = {
            geom_id: self.model.geom_rgba[geom_id].copy() for geom_id in self._clutter_geoms
        }
        self._default_light_pos = self.model.light_pos.copy()
        self._default_light_dir = self.model.light_dir.copy()
        self._default_light_ambient = self.model.light_ambient.copy()
        self._default_light_specular = self.model.light_specular.copy()
        self._robot_geoms = self._resolve_robot_geoms()
        grid = np.linspace(
            -self.realism_config.block_half_extent_m, self.realism_config.block_half_extent_m, 17
        )
        xx, yy = np.meshgrid(grid, grid, indexing="xy")
        self._coverage_points = np.stack([xx.ravel(), yy.ravel()], axis=1)

    def _build_model(self) -> mujoco.MjModel:
        return build_production_model(
            scene_path=self._model_scene_path,
            scene_bundle=self._model_scene_bundle,
            include_camera_housing=True,
            include_visual_clutter=True,
            include_realism_contact_pairs=True,
            include_multichoice_blocks=getattr(self.realism_config, "multichoice_blocks", False),
            include_stock_tip_references=self.include_stock_tip_references,
            tool_profile=self.model_tool_profile,
        )

    def _name_id(self, kind: mujoco.mjtObj, name: str) -> int:
        value = mujoco.mj_name2id(self.model, kind, name)
        if value < 0:
            raise ValueError(f"V6 model is missing {name}")
        return value

    def _resolve_robot_geoms(self) -> set[int]:
        robot_bodies = {
            "base",
            "shoulder",
            "upper_arm",
            "lower_arm",
            "wrist",
            "gripper",
            "moving_jaw_so101_v1",
        }
        result: set[int] = set()
        for geom_id in range(self.model.ngeom):
            body_id = int(self.model.geom_bodyid[geom_id])
            body_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, body_id)
            if body_name in robot_bodies:
                result.add(geom_id)
        result.add(self._ids["tool_geom"])
        result.update(self._ids["tool_contact_geoms"])
        result.add(self._camera_housing_geom)
        return result

    def reset(
        self,
        seed: int | None = None,
        *,
        obstacle: bool | None = None,
        stress: bool = False,
    ) -> dict[str, np.ndarray]:
        self._realism_ready = False
        observation = super().reset(seed=seed, obstacle=obstacle, stress=stress)
        episode_seed = int(self.episode_sim2real["seed"])
        self._v6_rng = np.random.default_rng(episode_seed ^ 0x6D1A6)
        self._command_burst_remaining = 0
        self._last_command_lost = False
        self._strict_success_streak = 0
        ambient = float(self._v6_rng.uniform(*self.realism_config.ambient_temperature_range_c))
        rise = self._v6_rng.uniform(*self.realism_config.initial_motor_temperature_rise_range_c, size=_JOINTS)
        self._motor_temperature_c = ambient + rise
        self._thermal_derating_start_c = self._v6_rng.uniform(
            *self.realism_config.thermal_derating_start_range_c, size=_JOINTS
        )
        self._supply_voltage_v = float(self._v6_rng.uniform(*self.realism_config.supply_voltage_range_v))
        self._loaded_voltage_v = self._supply_voltage_v
        self._last_actuator_force.fill(0.0)
        self._last_actual_velocity.fill(0.0)
        self._episode_v6.update(
            {
                "profile_version": self.profile_version,
                "claim_level": REALISM_CLAIM_LEVEL,
                "parameter_source": PARAMETER_SOURCE,
                "physical_samples": 0,
                "physically_calibrated": False,
                "physical_hardware_connected": False,
                "state_update": "mujoco_force_limited_actuator_no_forced_qpos_after_reset",
                "encoder_origin": "simulator_state",
                "supply_voltage_v": self._supply_voltage_v,
                "ambient_temperature_c": ambient,
                "initial_motor_temperature_c": self._motor_temperature_c.tolist(),
                "thermal_derating_start_c": self._thermal_derating_start_c.tolist(),
                "strict_success": {
                    "coverage_threshold": self.realism_config.strict_coverage_threshold,
                    "linear_speed_m_s": self.realism_config.strict_linear_speed_m_s,
                    "angular_speed_rad_s": self.realism_config.strict_angular_speed_rad_s,
                    "hold_steps": self.realism_config.strict_success_hold_steps,
                },
            }
        )
        self.episode_domain["realism_v6"] = self._episode_v6
        self._realism_ready = True
        return self.observation() if observation is not None else observation

    def _randomize_domain(self, stress: bool) -> None:
        super()._randomize_domain(stress)
        strength = 1.25 if stress else 1.0

        mass_scale = float(self.rng.uniform(*self.realism_config.block_mass_scale_range) ** strength)
        block_body = self._ids["block_body"]
        self.model.body_mass[block_body] = self.default_block_mass * mass_scale
        self.model.body_inertia[block_body] = self._default_block_inertia * mass_scale
        self.model.body_ipos[block_body] = self._default_block_ipos + self.rng.uniform(
            *self.realism_config.block_com_jitter_range_m, size=3
        )

        desk_slide = float(self.rng.uniform(*self.realism_config.desk_block_slide_friction_range))
        desk_torsion = float(self.rng.uniform(*self.realism_config.desk_block_torsional_friction_range))
        desk_roll = float(self.rng.uniform(*self.realism_config.desk_block_rolling_friction_range))
        pusher_slide = float(self.rng.uniform(*self.realism_config.pusher_block_slide_friction_range))
        contact_tau = float(self.rng.uniform(*self.realism_config.contact_time_constant_range_s))
        desk_pair = self._pair_ids["realism_v6_desk_block"]
        self.model.pair_friction[desk_pair] = [
            desk_slide,
            desk_slide,
            desk_torsion,
            desk_roll,
            desk_roll,
        ]
        self.model.pair_solref[desk_pair] = [contact_tau, 1.0]
        for pusher_pair in self._pusher_block_pair_ids:
            self.model.pair_friction[pusher_pair] = [
                pusher_slide,
                pusher_slide,
                desk_torsion,
                desk_roll,
                desk_roll,
            ]
            self.model.pair_solref[pusher_pair] = [
                max(contact_tau * 0.75, 0.002),
                1.0,
            ]

        payload_mass = float(self.rng.uniform(*self.realism_config.wrist_payload_mass_range_kg))
        payload_com_distance = float(self.rng.uniform(*self.realism_config.wrist_payload_com_range_m))
        gripper_body = self._ids["gripper_body"]
        total_mass = self._default_gripper_mass + payload_mass
        payload_com = np.array([-0.010, 0.020, -payload_com_distance], dtype=np.float64)
        combined_com = (
            self._default_gripper_mass * self._default_gripper_ipos + payload_mass * payload_com
        ) / total_mass
        distance2 = np.square(payload_com - combined_com)
        payload_box_inertia = payload_mass * np.array([0.00045, 0.00035, 0.00035])
        parallel_axis = payload_mass * np.array(
            [distance2[1] + distance2[2], distance2[0] + distance2[2], distance2[0] + distance2[1]]
        )
        self.model.body_mass[gripper_body] = total_mass
        self.model.body_ipos[gripper_body] = combined_com
        self.model.body_inertia[gripper_body] = (
            self._default_gripper_inertia + payload_box_inertia + parallel_axis
        )

        kp_scale = self.rng.uniform(*self.realism_config.actuator_kp_scale_range, size=_JOINTS)
        kv_scale = self.rng.uniform(*self.realism_config.actuator_kv_scale_range, size=_JOINTS)
        force_scale = self.rng.uniform(*self.realism_config.actuator_force_scale_range, size=_JOINTS)
        damping_scale = self.rng.uniform(*self.realism_config.joint_damping_scale_range, size=_JOINTS)
        armature_scale = self.rng.uniform(*self.realism_config.joint_armature_scale_range, size=_JOINTS)
        self.model.actuator_gainprm[:_JOINTS, 0] = self._default_actuator_gain[:_JOINTS] * kp_scale
        self.model.actuator_biasprm[:_JOINTS, 1] = -self.model.actuator_gainprm[:_JOINTS, 0]
        self.model.actuator_biasprm[:_JOINTS, 2] = -self._default_actuator_kv[:_JOINTS] * kv_scale
        self.model.actuator_forcerange[:_JOINTS] = (
            self._default_actuator_forcerange[:_JOINTS] * force_scale[:, None]
        )
        self.model.dof_damping[:_JOINTS] = self._default_dof_damping * damping_scale
        self.model.dof_armature[:_JOINTS] = self._default_dof_armature * armature_scale

        camera_id = self._ids["cameras"]["wrist"]
        rotation = self.rng.normal(0.0, self.realism_config.camera_mount_rotation_std_rad, 3)
        limit = self.realism_config.camera_mount_rotation_limit_rad
        rotation = np.clip(rotation, -limit, limit)
        self.model.cam_quat[camera_id] = _quat_multiply(
            self.model.cam_quat[camera_id], _rotation_vector_quat(rotation)
        )
        self._camera_episode_pos = self.model.cam_pos[camera_id].copy()
        self._camera_episode_quat = self.model.cam_quat[camera_id].copy()

        for material_id, rgba in self._default_surface_rgba.items():
            self.model.mat_rgba[material_id] = rgba
        desk_color = np.clip(self.rng.uniform(0.76, 1.0, 3), 0.0, 1.0)
        surface_material = int(self.rng.choice(self._surface_materials))
        background_candidates = [
            material_id for material_id in self._surface_materials if material_id != surface_material
        ]
        background_material = int(self.rng.choice(background_candidates))
        self.model.geom_matid[self._desk_geom] = surface_material
        self.model.geom_matid[self._floor_geom] = background_material
        self.model.geom_rgba[self._desk_geom] = [*desk_color, 1.0]
        self.model.mat_rgba[surface_material] = [*desk_color, 1.0]
        background_tint = self.rng.uniform(0.68, 0.92, 3)
        self.model.mat_rgba[background_material] = [*background_tint, 1.0]
        clutter_palette = np.array(
            [
                [0.09, 0.10, 0.12, 1.0],
                [0.62, 0.58, 0.48, 1.0],
                [0.18, 0.36, 0.55, 1.0],
                [0.54, 0.18, 0.12, 1.0],
                [0.16, 0.42, 0.24, 1.0],
            ]
        )
        for geom_id in self._clutter_geoms:
            self.model.geom_pos[geom_id] = self._default_clutter_pos[geom_id] + self.rng.normal(
                0.0, [0.012, 0.012, 0.004]
            )
            self.model.geom_rgba[geom_id] = clutter_palette[int(self.rng.integers(0, len(clutter_palette)))]

        light_scale = float(self.rng.uniform(0.45, 1.65))
        color_proxy = self.rng.uniform(0.78, 1.22, 3)
        self.model.light_pos[:] = self._default_light_pos + self.rng.normal(
            0.0, [0.08, 0.08, 0.12], self._default_light_pos.shape
        )
        self.model.light_dir[:] = self._default_light_dir + self.rng.normal(
            0.0, 0.08, self._default_light_dir.shape
        )
        norms = np.linalg.norm(self.model.light_dir, axis=1, keepdims=True)
        self.model.light_dir[:] /= np.maximum(norms, 1e-9)
        self.model.light_diffuse[:] = np.clip(
            self.default_light_diffuse * light_scale * color_proxy, 0.05, 1.6
        )
        self.model.light_ambient[:] = np.clip(
            self._default_light_ambient + self.rng.uniform(0.02, 0.20, self._default_light_ambient.shape),
            0.0,
            0.5,
        )
        self.model.light_specular[:] = np.clip(
            self._default_light_specular * self.rng.uniform(0.25, 1.5), 0.0, 1.0
        )

        self._episode_v6 = {
            "parameter_source": PARAMETER_SOURCE,
            "block_mass_scale": mass_scale,
            "block_mass_kg": float(self.model.body_mass[block_body]),
            "block_inertia_kg_m2": self.model.body_inertia[block_body].tolist(),
            "block_com_m": self.model.body_ipos[block_body].tolist(),
            "effective_contact_pairs": {
                "desk_block": self.model.pair_friction[desk_pair].tolist(),
                "pusher_block": self.model.pair_friction[pusher_pair].tolist(),
                "contact_time_constant_s": contact_tau,
            },
            "wrist_payload": {
                "mass_kg": payload_mass,
                "combined_gripper_mass_kg": total_mass,
                "combined_com_m": combined_com.tolist(),
                "combined_inertia_kg_m2": self.model.body_inertia[gripper_body].tolist(),
            },
            "actuator": {
                "kp_scale": kp_scale.tolist(),
                "kv_scale": kv_scale.tolist(),
                "force_scale": force_scale.tolist(),
                "damping_scale": damping_scale.tolist(),
                "armature_scale": armature_scale.tolist(),
            },
            "camera_mount_rotation_error_rad": rotation.tolist(),
            "desk_rgb": desk_color.tolist(),
            "desk_surface_material": mujoco.mj_id2name(
                self.model, mujoco.mjtObj.mjOBJ_MATERIAL, surface_material
            ),
            "background_surface_material": mujoco.mj_id2name(
                self.model, mujoco.mjtObj.mjOBJ_MATERIAL, background_material
            ),
            "light_scale": light_scale,
            "light_color_proxy": color_proxy.tolist(),
        }

    def observation(self) -> dict[str, np.ndarray]:
        observation = super().observation()
        if self._realism_ready:
            tick = 2.0 * np.pi / self.realism_config.encoder_ticks_per_revolution
            joint_state = observation["joint_state"].copy()
            joint_state[:_JOINTS] = np.round(joint_state[:_JOINTS] / tick) * tick
            observation["joint_state"] = joint_state.astype(np.float32)
        return observation

    def _command_reference_reported_position(self) -> np.ndarray:
        """Use the same encoder-quantized state exposed to the V6/V7 policy."""

        return np.asarray(self.observation()["joint_state"][:_JOINTS], dtype=np.float64)

    def teacher_action(self) -> tuple[np.ndarray, dict[str, Any]]:
        """Privileged V6 teacher with an explicit settle phase.

        The historical teacher keeps pushing after entering the broad legacy
        radius.  Under force-limited dynamics that tends to overshoot or leave
        the block moving.  V6 slows the final push and holds once the strict
        footprint criterion is reached so demonstrations contain the missing
        "arrive, then stop" behavior.
        """

        coverage = self.block_target_coverage()
        if coverage >= self.realism_config.strict_coverage_threshold:
            return np.zeros(_JOINTS, dtype=np.float32), {
                "phase": "settle",
                "teacher_confidence": 1.0,
                "joint_target": self.data.qpos[:_JOINTS].copy().astype(np.float32),
                "strict_target_coverage": coverage,
            }
        action, metadata = super().teacher_action()
        distance = self.distance_to_target()
        if metadata["phase"] == "push":
            final_scale = float(np.clip((distance - 0.012) / 0.075, 0.16, 1.0))
            action = action * final_scale
            metadata["final_push_scale"] = final_scale
        metadata["strict_target_coverage"] = coverage
        return action.astype(np.float32), metadata

    def step(self, action: np.ndarray) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        action = np.asarray(action, dtype=np.float64)
        if action.shape != (_JOINTS,) or not np.all(np.isfinite(action)):
            raise ValueError("V6 action must be a finite six-joint vector")
        command_lost = self._sample_command_loss()
        applied_action = np.zeros_like(action) if command_lost else action
        before = self.data.qpos[:_JOINTS].copy()
        self._submission_ingress_lost = command_lost
        try:
            observation, reward, _legacy_terminated, truncated, info = super().step(applied_action)
        finally:
            self._submission_ingress_lost = False
        actual_velocity = (self.data.qpos[:_JOINTS] - before) / max(self.control_dt, 1e-9)
        self._last_actual_velocity = actual_velocity
        if "sim2real_v2" in info:
            info["sim2real_v2"]["physical_joint_velocity"] = actual_velocity.astype(np.float32)
            info["sim2real_v2"]["physical_joint_position_origin"] = "simulator_state"

        coverage = self.block_target_coverage()
        linear_speed, angular_speed = self._block_speeds()
        settled = (
            linear_speed <= self.realism_config.strict_linear_speed_m_s
            and angular_speed <= self.realism_config.strict_angular_speed_rad_s
        )
        contained = coverage >= self.realism_config.strict_coverage_threshold
        self._strict_success_streak = self._strict_success_streak + 1 if contained and settled else 0
        raw_strict_success = (
            self._strict_success_streak >= self.realism_config.strict_success_hold_steps
        )
        collisions = self._obstacle_contacts()
        collision_failure = self.realism_config.obstacle_collision_terminates and any(
            collisions[name] > 0 for name in ("camera_housing", "pusher", "robot_other")
        )
        block = self.block_xy()
        out = not (0.06 <= block[0] <= 0.45 and -0.29 <= block[1] <= 0.29)
        safety_stop = info.get("safety_stop")
        emergency_stop = isinstance(safety_stop, str) and bool(safety_stop)
        terminal_failure = bool(out or collision_failure or emergency_stop)
        strict_success = bool(raw_strict_success and not terminal_failure)
        legacy_success = bool(info.get("success", False))
        if legacy_success:
            reward -= 10.0
        if strict_success:
            reward += 12.0
        if collision_failure:
            reward -= 8.0
        terminated = bool(strict_success or terminal_failure)
        time_limit_reached = self.step_count >= self.config.max_steps
        truncated = bool(time_limit_reached and not terminated)
        if strict_success:
            terminal_reason = "strict_success"
        elif emergency_stop:
            terminal_reason = f"safety_stop:{safety_stop}"
        elif collision_failure:
            terminal_reason = "obstacle_collision"
        elif out:
            terminal_reason = "block_out_of_bounds"
        elif truncated:
            terminal_reason = "time_limit"
        else:
            terminal_reason = "nonterminal"
        safety_reason = str(info.get("safety_reason", ""))
        if collision_failure:
            safety_reason = self._combine_reasons(safety_reason, "obstacle_collision")
        if emergency_stop:
            safety_reason = self._combine_reasons(safety_reason, str(safety_stop))
        info.update(
            {
                "success": strict_success,
                "legacy_success": legacy_success,
                "terminated": terminated,
                "truncated": truncated,
                "time_limit_reached": bool(time_limit_reached),
                "terminal_failure": terminal_failure,
                "terminal_reason": terminal_reason,
                "safety_clipped": bool(info.get("safety_clipped", False) or collision_failure),
                "safety_reason": safety_reason,
                "realism_v6": {
                    "profile_version": self.profile_version,
                    "claim_level": REALISM_CLAIM_LEVEL,
                    "parameter_source": PARAMETER_SOURCE,
                    "physical_samples": 0,
                    "physical_joint_position_origin": "simulator_state",
                    "command_lost": command_lost,
                    "command_burst_remaining": self._command_burst_remaining,
                    "actuator_force_nm": self._last_actuator_force.copy().astype(np.float32),
                    "motor_temperature_c": self._motor_temperature_c.copy().astype(np.float32),
                    "supply_voltage_v": self._supply_voltage_v,
                    "loaded_voltage_v": self._loaded_voltage_v,
                    "actual_joint_velocity_rad_s": actual_velocity.astype(np.float32),
                    "strict_target_coverage": coverage,
                    "strict_contained": contained,
                    "strict_settled": settled,
                    "strict_success_streak": self._strict_success_streak,
                    "raw_strict_success": raw_strict_success,
                    "block_linear_speed_m_s": linear_speed,
                    "block_angular_speed_rad_s": angular_speed,
                    "obstacle_contacts": collisions,
                    "collision_failure": collision_failure,
                },
            }
        )
        return observation, float(reward), terminated, truncated, info

    def _sample_command_loss(self) -> bool:
        if self._command_burst_remaining <= 0 and (
            self._v6_rng.random() < self.realism_config.command_burst_start_probability
        ):
            low, high = self.realism_config.command_burst_length_range
            self._command_burst_remaining = int(self._v6_rng.integers(low, high + 1))
        burst_lost = self._command_burst_remaining > 0
        if burst_lost:
            self._command_burst_remaining -= 1
        self._last_command_lost = bool(
            burst_lost or self._v6_rng.random() < self.realism_config.command_loss_probability
        )
        return self._last_command_lost

    def _advance_physics(
        self,
        start: np.ndarray,
        end: np.ndarray,
        velocity: np.ndarray,
        controller_target: np.ndarray,
    ) -> None:
        del start, velocity, controller_target
        # Crucial V6 difference: no qpos/qvel assignment occurs here.  The
        # rate/backlash-limited target is fed to force-limited MuJoCo actuators.
        self.data.ctrl[:_JOINTS] = end
        for _ in range(self.config.physics_substeps):
            mujoco.mj_step(self.model, self.data)
        self._last_actuator_force = self.data.actuator_force[:_JOINTS].copy()
        self._update_electrical_and_thermal_state()
        self._update_camera_flex()

    def _update_electrical_and_thermal_state(self) -> None:
        limits = np.maximum(np.abs(self._default_actuator_forcerange[:_JOINTS, 1]), 1e-6)
        effort = np.clip(np.abs(self._last_actuator_force) / limits, 0.0, 1.5)
        mean_effort = float(np.mean(effort))
        self._loaded_voltage_v = max(
            7.0,
            self._supply_voltage_v - self.realism_config.voltage_sag_per_unit_effort_v * mean_effort,
        )
        ambient = float(self._episode_v6.get("ambient_temperature_c", 25.0))
        dt = self.control_dt
        self._motor_temperature_c += dt * (
            self.realism_config.thermal_heating_rate_c_s * np.square(effort)
            - self.realism_config.thermal_cooling_rate_s * (self._motor_temperature_c - ambient)
        )
        denominator = np.maximum(
            self.realism_config.thermal_shutdown_c - self._thermal_derating_start_c,
            1e-6,
        )
        thermal_scale = np.clip(
            1.0
            - 0.65
            * np.maximum(self._motor_temperature_c - self._thermal_derating_start_c, 0.0)
            / denominator,
            0.35,
            1.0,
        )
        voltage_scale = np.clip(
            self._loaded_voltage_v / self.realism_config.nominal_supply_voltage_v,
            0.55,
            1.10,
        )
        episode_force_scale = np.asarray(
            self._episode_v6.get("actuator", {}).get("force_scale", np.ones(_JOINTS)),
            dtype=np.float64,
        )
        self.model.actuator_forcerange[:_JOINTS] = (
            self._default_actuator_forcerange[:_JOINTS]
            * (episode_force_scale * thermal_scale * voltage_scale)[:, None]
        )

    def _update_camera_flex(self) -> None:
        camera_id = self._ids["cameras"]["wrist"]
        wrist_speed = float(np.linalg.norm(self.data.qvel[3:5]))
        flex = min(
            wrist_speed * self.realism_config.camera_flex_translation_per_rad_s_m,
            0.0015,
        )
        self.model.cam_pos[camera_id] = self._camera_episode_pos + np.array([0.0, flex, -0.35 * flex])
        self.model.cam_quat[camera_id] = self._camera_episode_quat
        mujoco.mj_forward(self.model, self.data)

    def block_target_coverage(self) -> float:
        address = int(self.model.jnt_qposadr[self._ids["block_joint"]])
        position = self.data.qpos[address : address + 2]
        quat = self.data.qpos[address + 3 : address + 7]
        w, x, y, z = quat / max(float(np.linalg.norm(quat)), 1e-12)
        yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
        cosine, sine = np.cos(yaw), np.sin(yaw)
        rotation = np.array([[cosine, -sine], [sine, cosine]])
        world_points = self._coverage_points @ rotation.T + position
        inside = np.linalg.norm(world_points - self.target_xy, axis=1) <= self.realism_config.target_radius_m
        return float(np.mean(inside))

    def _block_speeds(self) -> tuple[float, float]:
        velocity = self.data.qvel[self._block_dof_address : self._block_dof_address + 6]
        return float(np.linalg.norm(velocity[:3])), float(np.linalg.norm(velocity[3:]))

    def _obstacle_contacts(self) -> dict[str, int]:
        counts = {"camera_housing": 0, "pusher": 0, "robot_other": 0, "block": 0}
        obstacle = self._ids["obstacle_geom"]
        if not self.obstacle_enabled:
            return counts
        for index in range(self.data.ncon):
            pair = {int(self.data.contact[index].geom1), int(self.data.contact[index].geom2)}
            if obstacle not in pair:
                continue
            other = next(iter(pair - {obstacle}), obstacle)
            if other == self._camera_housing_geom:
                counts["camera_housing"] += 1
            elif other in self._ids["tool_contact_geoms"]:
                counts["pusher"] += 1
            elif other == self._ids["block_geom"]:
                counts["block"] += 1
            elif other in self._robot_geoms:
                counts["robot_other"] += 1
        return counts
