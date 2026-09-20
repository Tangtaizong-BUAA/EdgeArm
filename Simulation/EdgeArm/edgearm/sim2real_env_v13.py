"""Replayable scene-generalization successor to the mounted-camera V12 plant.

V13 is opt-in.  It preserves the stock SO-101 follower and mounted wrist camera
while adding three static obstacle slots and per-episode object geometry,
friction, material, and lighting strata.  Every realized value is carried by a
``GeneralizationScenarioV1`` contract so deferred RGB-D replay never guesses.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any, Mapping

import mujoco
import numpy as np

from .generalization_scene_v1 import (
    GeneralizationScenarioV1,
    sample_generalization_scenario,
)
from .generalization_task_language_v1 import GeneralizationTaskMetadataV1
from .production_env import (
    GENERALIZATION_OBSTACLE_GEOMS_V13,
    build_production_model,
)
from .sim2real_env_v12 import RealisticEdgeArmEnvV12, RealisticEnvV12Config


GENERALIZATION_DYNAMICS_PROFILE_V13 = "edgearm-sim2real-v13-mounted-wrist-stratified-generalization-v1"
GENERALIZATION_PARAMETER_SOURCE_V13 = "synthetic_stratified_ranges_unvalidated_against_physical_samples"
GENERALIZATION_CONTACT_AUDIT_V13 = "edgearm-v13-all-obstacle-contact-audit-v1"
GENERALIZATION_REQUIRED_STABLE_SECONDS_V13 = 3.0
GENERALIZATION_KEYBOARD_TELEOP_TRANSPORT_V13 = (
    "edgearm-v13-keyboard-teleop-transport-v1-zero-delay-zero-loss"
)
VISUAL_REPLAY_SNAPSHOT_FORMAT_V28 = "edgearm-v28-pre-action-visual-replay-snapshot-v1"
RealisticEnvV13Config = RealisticEnvV12Config


def keyboard_teleop_env_config_v13(*, max_steps: int) -> RealisticEnvV13Config:
    """Return the V13 plant with a deterministic human-command transport.

    Mechanical, contact, power, payload, camera, object, obstacle, and lighting
    randomization remain active.  Only synthetic packet delay/loss is disabled:
    injecting those faults while a human is producing a demonstration makes
    the operator compensate for invisible transport noise and corrupts the
    intended action label.  Robustness to delay/loss belongs in a separate
    augmentation or evaluation pass.
    """

    return RealisticEnvV13Config(
        max_steps=max_steps,
        command_delay_steps_range=(0, 0),
        command_loss_probability=0.0,
        command_burst_start_probability=0.0,
    )

_GEOM_TYPES = {
    "box": mujoco.mjtGeom.mjGEOM_BOX,
    "cylinder": mujoco.mjtGeom.mjGEOM_CYLINDER,
    "ellipsoid": mujoco.mjtGeom.mjGEOM_ELLIPSOID,
}


def _yaw_quaternion(yaw: float) -> np.ndarray:
    return np.asarray([np.cos(0.5 * yaw), 0.0, 0.0, np.sin(0.5 * yaw)], dtype=np.float64)


def _shape_model_size(shape: str, extents: tuple[float, float, float]) -> np.ndarray:
    x, y, z = extents
    if shape == "cylinder":
        return np.asarray([x, z, 0.0], dtype=np.float64)
    return np.asarray([x, y, z], dtype=np.float64)


def _shape_rbound(shape: str, extents: tuple[float, float, float]) -> float:
    x, y, z = extents
    if shape == "cylinder":
        return float(np.hypot(x, z))
    return float(np.linalg.norm([x, y, z]))


def _principal_inertia(shape: str, mass: float, extents: tuple[float, float, float]) -> np.ndarray:
    x, y, z = extents
    if shape == "box":
        return mass / 3.0 * np.asarray([y * y + z * z, x * x + z * z, x * x + y * y])
    if shape == "cylinder":
        radial = mass * (3.0 * x * x + 4.0 * z * z) / 12.0
        return np.asarray([radial, radial, 0.5 * mass * x * x], dtype=np.float64)
    return mass / 5.0 * np.asarray([y * y + z * z, x * x + z * z, x * x + y * y])


class RealisticEdgeArmEnvV13(RealisticEdgeArmEnvV12):
    """V12 stock arm with exact, replay-bound scene generalization."""

    profile_version = GENERALIZATION_DYNAMICS_PROFILE_V13

    def __init__(
        self,
        config: RealisticEnvV13Config | None = None,
        seed: int = 0,
        **kwargs: Any,
    ) -> None:
        self.current_scenario: GeneralizationScenarioV1 | None = None
        self.manipulated_object_shape = "box"
        self._v13_forbidden_contact_seen = False
        self._v13_stable_start_time_s: float | None = None
        self._v13_stable_duration_s = 0.0
        self._v13_contact_totals = {"block": 0, "pusher": 0, "camera_housing": 0, "robot_other": 0}
        super().__init__(config=config, seed=seed, **kwargs)
        self._v13_obstacle_geoms = tuple(
            self._name_id(mujoco.mjtObj.mjOBJ_GEOM, name) for name in GENERALIZATION_OBSTACLE_GEOMS_V13
        )
        self._v13_active_obstacle_geoms: tuple[int, ...] = ()
        block = self._ids["block_geom"]
        self._v13_base_block_model = {
            "type": int(self.model.geom_type[block]),
            "size": self.model.geom_size[block].copy(),
            "rbound": float(self.model.geom_rbound[block]),
        }
        self._v13_base_obstacle_model = {
            geom_id: {
                "type": int(self.model.geom_type[geom_id]),
                "size": self.model.geom_size[geom_id].copy(),
                "rbound": float(self.model.geom_rbound[geom_id]),
                "pos": self.model.geom_pos[geom_id].copy(),
                "quat": self.model.geom_quat[geom_id].copy(),
                "friction": self.model.geom_friction[geom_id].copy(),
                "rgba": self.model.geom_rgba[geom_id].copy(),
                "contype": int(self.model.geom_contype[geom_id]),
                "conaffinity": int(self.model.geom_conaffinity[geom_id]),
            }
            for geom_id in self._v13_obstacle_geoms
        }
        block_material = int(self.model.geom_matid[block])
        self._v13_block_material_id = block_material
        self._v13_base_block_material = (
            (
                float(self.model.mat_specular[block_material]),
                float(self.model.mat_roughness[block_material]),
            )
            if block_material >= 0
            else None
        )

    def _build_model(self) -> mujoco.MjModel:
        model = build_production_model(
            scene_path=self._model_scene_path,
            scene_bundle=self._model_scene_bundle,
            include_camera_housing=True,
            include_camera_mount=True,
            include_visual_clutter=True,
            include_realism_contact_pairs=True,
            include_generalization_obstacles=True,
            include_stock_tip_references=self.include_stock_tip_references,
            tool_profile=self.model_tool_profile,
        )
        desk = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "edgearm_desk")
        if desk < 0:
            raise ValueError("V13 model is missing edgearm_desk")
        original_left = float(model.geom_pos[desk, 0] - model.geom_size[desk, 0])
        original_right = float(model.geom_pos[desk, 0] + model.geom_size[desk, 0])
        front = float(self.contact_feasible_config.desk_front_edge_x_m)
        if front >= original_right:
            raise ValueError("V13 desk front edge would remove the work surface")
        model.geom_pos[desk, 0] = 0.5 * (front + original_right)
        model.geom_size[desk, 0] = 0.5 * (original_right - front)
        self._desk_original_x_bounds = np.asarray([original_left, original_right], dtype=np.float64)
        self._desk_corrected_x_bounds = np.asarray([front, original_right], dtype=np.float64)
        return model

    def _set_geom_shape(
        self,
        geom_id: int,
        shape: str,
        extents: tuple[float, float, float],
    ) -> None:
        self.model.geom_type[geom_id] = _GEOM_TYPES[shape]
        self.model.geom_size[geom_id] = _shape_model_size(shape, extents)
        self.model.geom_rbound[geom_id] = _shape_rbound(shape, extents)

    def _restore_v12_scene_model(self) -> None:
        """Remove the preceding V13 episode before the inherited reset runs."""

        block = self._ids["block_geom"]
        self.model.geom_type[block] = self._v13_base_block_model["type"]
        self.model.geom_size[block] = self._v13_base_block_model["size"]
        self.model.geom_rbound[block] = self._v13_base_block_model["rbound"]
        for geom_id, values in self._v13_base_obstacle_model.items():
            self.model.geom_type[geom_id] = values["type"]
            self.model.geom_size[geom_id] = values["size"]
            self.model.geom_rbound[geom_id] = values["rbound"]
            self.model.geom_pos[geom_id] = values["pos"]
            self.model.geom_quat[geom_id] = values["quat"]
            self.model.geom_friction[geom_id] = values["friction"]
            self.model.geom_rgba[geom_id] = values["rgba"]
            self.model.geom_contype[geom_id] = values["contype"]
            self.model.geom_conaffinity[geom_id] = values["conaffinity"]
        if self._v13_block_material_id >= 0 and self._v13_base_block_material is not None:
            self.model.mat_specular[self._v13_block_material_id] = self._v13_base_block_material[0]
            self.model.mat_roughness[self._v13_block_material_id] = self._v13_base_block_material[1]
        self._v13_active_obstacle_geoms = ()
        self.manipulated_object_shape = "box"
        mujoco.mj_setConst(self.model, self.data)

    def _coverage_points_for_scenario(self, scenario: GeneralizationScenarioV1) -> np.ndarray:
        x, y, _z = scenario.object_half_extents_m
        grid_x = np.linspace(-x, x, 25)
        grid_y = np.linspace(-y, y, 25)
        xx, yy = np.meshgrid(grid_x, grid_y, indexing="xy")
        points = np.stack([xx.ravel(), yy.ravel()], axis=1)
        if scenario.object_shape in {"cylinder", "ellipsoid"}:
            inside = (points[:, 0] / x) ** 2 + (points[:, 1] / y) ** 2 <= 1.0 + 1.0e-12
            points = points[inside]
        if points.size == 0:
            raise RuntimeError("V13 generated no coverage points")
        return points

    def _apply_scenario(self, scenario: GeneralizationScenarioV1) -> None:
        saved_qpos = np.asarray(self.data.qpos, dtype=np.float64).copy()
        saved_qvel = np.asarray(self.data.qvel, dtype=np.float64).copy()
        saved_ctrl = np.asarray(self.data.ctrl, dtype=np.float64).copy()
        saved_time = float(self.data.time)
        block = self._ids["block_geom"]
        block_body = self._ids["block_body"]
        self.manipulated_object_shape = scenario.object_shape
        self._set_geom_shape(block, scenario.object_shape, scenario.object_half_extents_m)
        self.model.geom_friction[block] = [scenario.block_friction, 0.012, 0.0012]
        self.model.body_mass[block_body] = scenario.block_mass_kg
        self.model.body_inertia[block_body] = _principal_inertia(
            scenario.object_shape,
            scenario.block_mass_kg,
            scenario.object_half_extents_m,
        )
        self.model.body_ipos[block_body] = 0.0
        self._coverage_points = self._coverage_points_for_scenario(scenario)

        block_material = int(self.model.geom_matid[block])
        material_values = {
            "matte": (0.04, 0.92),
            "satin": (0.22, 0.62),
            "glossy": (0.55, 0.30),
        }[scenario.object_material]
        if block_material >= 0:
            self.model.mat_specular[block_material] = material_values[0]
            self.model.mat_roughness[block_material] = material_values[1]

        desk_top = float(self.model.geom_pos[self._desk_geom, 2] + self.model.geom_size[self._desk_geom, 2])
        for geom_id in self._v13_obstacle_geoms:
            self.model.geom_contype[geom_id] = 0
            self.model.geom_conaffinity[geom_id] = 0
            self.model.geom_rgba[geom_id, 3] = 0.0
            self.model.geom_pos[geom_id] = [0.55, 0.28, desk_top + 0.03]
        active: list[int] = []
        for spec in scenario.obstacles:
            geom_id = self._v13_obstacle_geoms[spec.slot]
            extents = (spec.size_xy_m[0], spec.size_xy_m[1], spec.half_height_m)
            self._set_geom_shape(geom_id, spec.shape, extents)
            self.model.geom_pos[geom_id] = [
                spec.center_xy_m[0],
                spec.center_xy_m[1],
                desk_top + spec.half_height_m,
            ]
            self.model.geom_quat[geom_id] = _yaw_quaternion(spec.yaw_rad)
            self.model.geom_friction[geom_id] = [spec.friction, 0.01, 0.001]
            self.model.geom_rgba[geom_id] = spec.rgba
            self.model.geom_contype[geom_id] = 1
            self.model.geom_conaffinity[geom_id] = 1
            active.append(geom_id)
        self._v13_active_obstacle_geoms = tuple(active)
        self.obstacle_enabled = bool(active)
        self.obstacle_xy = (
            np.asarray(scenario.obstacles[0].center_xy_m, dtype=np.float64)
            if scenario.obstacles
            else np.asarray([0.55, 0.28], dtype=np.float64)
        )

        self.model.light_diffuse[:] = np.clip(
            self.default_light_diffuse * scenario.light_scale,
            0.05,
            1.0,
        )
        self.model.light_ambient[:] = np.clip(
            self._default_light_ambient * scenario.light_ambient_scale,
            0.02,
            1.0,
        )
        self.model.light_specular[:] = np.clip(
            self._default_light_specular * scenario.light_scale,
            0.02,
            1.0,
        )
        for pair_id in (self._pair_ids["realism_v6_desk_block"], *self._pusher_block_pair_ids):
            self.model.pair_friction[pair_id, 0] = scenario.block_friction
            self.model.pair_friction[pair_id, 1] = scenario.block_friction

        mujoco.mj_setConst(self.model, self.data)
        # ``mj_setConst`` does not refresh every mutable geom rbound in all
        # MuJoCo builds, so restore the explicitly computed values afterwards.
        self.model.geom_rbound[block] = _shape_rbound(scenario.object_shape, scenario.object_half_extents_m)
        for spec, geom_id in zip(scenario.obstacles, active, strict=True):
            self.model.geom_rbound[geom_id] = _shape_rbound(
                spec.shape,
                (spec.size_xy_m[0], spec.size_xy_m[1], spec.half_height_m),
            )
        # mj_setConst is allowed to overwrite MjData with keyframe state.
        # Restore the exact reset plant state and then install the generalized
        # object's support height; otherwise replay could silently start from
        # the MJCF's historical block coordinates.
        self.data.qpos[:] = saved_qpos
        self.data.qvel[:] = saved_qvel
        self.data.ctrl[:] = saved_ctrl
        self.data.time = saved_time
        block_address = int(self.model.jnt_qposadr[self._ids["block_joint"]])
        self.data.qpos[block_address : block_address + 3] = [
            scenario.block_initial_xy_m[0],
            scenario.block_initial_xy_m[1],
            desk_top + scenario.object_half_extents_m[2] + 5.0e-4,
        ]
        self.data.qpos[block_address + 3 : block_address + 7] = [1.0, 0.0, 0.0, 0.0]
        self.data.qvel[self._block_dof_address : self._block_dof_address + 6] = 0.0
        mujoco.mj_forward(self.model, self.data)
        self.last_distance = self.distance_to_target()
        self._strict_success_streak = 0
        self.success_streak = 0

    def _all_obstacle_contacts(self) -> dict[str, int]:
        counts = {"block": 0, "pusher": 0, "camera_housing": 0, "robot_other": 0}
        active = frozenset(self._v13_active_obstacle_geoms)
        if not active:
            return counts
        for index in range(self.data.ncon):
            first = int(self.data.contact[index].geom1)
            second = int(self.data.contact[index].geom2)
            obstacle = first if first in active else second if second in active else None
            if obstacle is None:
                continue
            other = second if first == obstacle else first
            if other == self._ids["block_geom"]:
                counts["block"] += 1
            elif other in self._ids["tool_contact_geoms"]:
                counts["pusher"] += 1
            elif other == self._camera_housing_geom:
                counts["camera_housing"] += 1
            elif other in self._robot_geoms:
                counts["robot_other"] += 1
        return counts

    def _obstacle_contacts(self) -> dict[str, int]:
        return self._all_obstacle_contacts()

    def _reset_audit_v13(self) -> dict[str, Any]:
        contacts = self._all_obstacle_contacts()
        desk_min = np.asarray(
            [
                self.model.geom_pos[self._desk_geom, 0] - self.model.geom_size[self._desk_geom, 0],
                self.model.geom_pos[self._desk_geom, 1] - self.model.geom_size[self._desk_geom, 1],
            ],
            dtype=np.float64,
        )
        desk_max = np.asarray(
            [
                self.model.geom_pos[self._desk_geom, 0] + self.model.geom_size[self._desk_geom, 0],
                self.model.geom_pos[self._desk_geom, 1] + self.model.geom_size[self._desk_geom, 1],
            ],
            dtype=np.float64,
        )
        support_margins: list[float] = []
        assert self.current_scenario is not None
        for spec in self.current_scenario.obstacles:
            center = np.asarray(spec.center_xy_m)
            radius = float(np.linalg.norm(spec.size_xy_m))
            support_margins.append(
                float(min(np.min(center - radius - desk_min), np.min(desk_max - center - radius)))
            )
        initial_forbidden = int(sum(contacts.values()))
        return {
            "format": GENERALIZATION_CONTACT_AUDIT_V13,
            "active_obstacle_count": len(self._v13_active_obstacle_geoms),
            "initial_contact_count_by_class": contacts,
            "initial_forbidden_contact_count": initial_forbidden,
            "minimum_obstacle_support_margin_m": min(support_margins, default=None),
            "reset_valid": initial_forbidden == 0 and all(value >= 0.0 for value in support_margins),
        }

    def reset(
        self,
        seed: int | None = None,
        *,
        obstacle: bool | None = None,
        stress: bool = False,
        scenario: GeneralizationScenarioV1 | Mapping[str, Any] | None = None,
        schedule_index: int | None = None,
    ) -> dict[str, Any]:
        if obstacle is not None:
            raise ValueError("V13 obstacle layout is controlled by its scene contract")
        requested_seed = int(self.seed if seed is None else seed)
        self._restore_v12_scene_model()
        super().reset(seed=requested_seed, obstacle=False, stress=stress)
        if scenario is None:
            scenario = sample_generalization_scenario(
                requested_seed=requested_seed,
                schedule_index=requested_seed if schedule_index is None else schedule_index,
                block_initial_xy_m=self.block_xy(),
                target_xy_m=self.target_xy,
                block_color=self.color_name,
                target_color=self.target_color_name,
                block_mass_kg=float(self.model.body_mass[self._ids["block_body"]]),
            )
        elif isinstance(scenario, Mapping):
            scenario = GeneralizationScenarioV1.from_contract(scenario)
        if scenario.requested_seed != requested_seed:
            raise ValueError("V13 scenario seed differs from reset seed")
        if not np.allclose(scenario.block_initial_xy_m, self.block_xy(), atol=1.0e-10, rtol=0.0):
            raise ValueError("V13 scenario block start differs from reconstructed reset")
        if not np.allclose(scenario.target_xy_m, self.target_xy, atol=1.0e-10, rtol=0.0):
            raise ValueError("V13 scenario target differs from reconstructed reset")
        if scenario.block_color != self.color_name or scenario.target_color != self.target_color_name:
            raise ValueError("V13 scenario colors differ from reconstructed reset")
        self.current_scenario = scenario
        self._v13_forbidden_contact_seen = False
        self._v13_stable_start_time_s = None
        self._v13_stable_duration_s = 0.0
        self._v13_contact_totals = {"block": 0, "pusher": 0, "camera_housing": 0, "robot_other": 0}
        self._apply_scenario(scenario)
        task = GeneralizationTaskMetadataV1.from_scenario(scenario)
        self.task_text = task.task_text_en
        self.task_text_zh = task.task_text_zh
        reset_audit = self._reset_audit_v13()
        if not reset_audit["reset_valid"]:
            raise RuntimeError(f"V13 generalized reset is invalid: {reset_audit}")
        self.episode_domain["realism_v13"] = {
            "profile_version": GENERALIZATION_DYNAMICS_PROFILE_V13,
            "parameter_source": GENERALIZATION_PARAMETER_SOURCE_V13,
            "physical_samples": 0,
            "physically_calibrated": False,
            "scene_contract": scenario.contract(),
            "scene_contract_sha256": scenario.contract_sha256,
            "reset_audit": reset_audit,
            "realized_light_diffuse": self.model.light_diffuse.tolist(),
            "realized_light_ambient": self.model.light_ambient.tolist(),
            "active_obstacle_geom_ids": list(self._v13_active_obstacle_geoms),
            "strict_stable_hold_required_seconds": GENERALIZATION_REQUIRED_STABLE_SECONDS_V13,
            "configuration": asdict(self.joint_bounded_config),
        }
        return self.observation()

    def step(self, action: Any) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        mujoco.mj_forward(self.model, self.data)
        wrist_camera_id = int(self._ids["cameras"]["wrist"])
        visual_replay_snapshot = {
            "format": VISUAL_REPLAY_SNAPSHOT_FORMAT_V28,
            "pre_action_qpos": self.data.qpos.copy(),
            "pre_action_qvel": self.data.qvel.copy(),
            "pre_action_ctrl": self.data.ctrl.copy(),
            "pre_action_time_seconds": float(self.data.time),
            "pre_action_wrist_camera_local_position": self.model.cam_pos[
                wrist_camera_id
            ].copy(),
            "pre_action_wrist_camera_local_quaternion_wxyz": self.model.cam_quat[
                wrist_camera_id
            ].copy(),
            "pre_action_wrist_camera_world_position": self.data.cam_xpos[
                wrist_camera_id
            ].copy(),
            "pre_action_wrist_camera_world_rotation": self.data.cam_xmat[
                wrist_camera_id
            ].reshape(3, 3).copy(),
            "supports_exact_visual_state_restore": True,
            "supports_exact_dynamic_resume": False,
            "source": "live_v13_pre_action_simulator_state",
            "physical_samples": 0,
            "production_admission": False,
        }
        transition_start_time_s = float(self.data.time)
        observation, reward, terminated, truncated, info = super().step(action)
        visual_replay_snapshot.update(
            {
                "post_action_qpos": self.data.qpos.copy(),
                "post_action_qvel": self.data.qvel.copy(),
                "post_action_ctrl": self.data.ctrl.copy(),
                "post_action_time_seconds": float(self.data.time),
            }
        )
        info["visual_replay_snapshot_v28"] = visual_replay_snapshot
        contacts = self._all_obstacle_contacts()
        for name, count in contacts.items():
            self._v13_contact_totals[name] += int(count)
        if any(contacts.values()):
            self._v13_forbidden_contact_seen = True
        realism_v6 = dict(info.get("realism_v6", {}))
        instant_stable = bool(
            not self._v13_forbidden_contact_seen
            and realism_v6.get("strict_contained", False)
            and realism_v6.get("strict_settled", False)
            and float(realism_v6.get("strict_target_coverage", 0.0)) >= 0.95
        )
        if instant_stable:
            if self._v13_stable_start_time_s is None:
                self._v13_stable_start_time_s = transition_start_time_s
            self._v13_stable_duration_s = max(
                0.0,
                float(self.data.time) - self._v13_stable_start_time_s,
            )
        else:
            self._v13_stable_start_time_s = None
            self._v13_stable_duration_s = 0.0
        stable_three_seconds = bool(
            self._v13_stable_duration_s + 1.0e-12 >= GENERALIZATION_REQUIRED_STABLE_SECONDS_V13
        )

        v13 = dict(self.episode_domain["realism_v13"])
        v13.update(
            {
                "step_contact_count_by_class": contacts,
                "cumulative_contact_count_by_class": dict(self._v13_contact_totals),
                "strict_no_obstacle_contact": not self._v13_forbidden_contact_seen,
                "instant_target_stable": instant_stable,
                "continuous_stable_seconds": self._v13_stable_duration_s,
                "required_stable_seconds": GENERALIZATION_REQUIRED_STABLE_SECONDS_V13,
                "strict_stable_3s": stable_three_seconds,
            }
        )
        info["realism_v13"] = v13
        if not stable_three_seconds:
            info["success"] = False
            realism_v6["raw_strict_success"] = False
            realism_v6["strict_scene_obstacle_contact_free"] = not self._v13_forbidden_contact_seen
            realism_v6["strict_stable_3s"] = False
            info["realism_v6"] = realism_v6
            # V6/V7 may emit their historical short-hold termination after six
            # stable steps.  V13 deliberately keeps advancing zero-action rows
            # until the independent 3.0 s simulation-time gate is satisfied.
            if not truncated:
                terminated = False
        else:
            info["success"] = True
            realism_v6["raw_strict_success"] = True
            realism_v6["strict_scene_obstacle_contact_free"] = True
            realism_v6["strict_stable_3s"] = True
            info["realism_v6"] = realism_v6
            terminated = True
        return observation, reward, terminated, truncated, info


__all__ = [
    "GENERALIZATION_CONTACT_AUDIT_V13",
    "GENERALIZATION_DYNAMICS_PROFILE_V13",
    "GENERALIZATION_KEYBOARD_TELEOP_TRANSPORT_V13",
    "GENERALIZATION_PARAMETER_SOURCE_V13",
    "GENERALIZATION_REQUIRED_STABLE_SECONDS_V13",
    "VISUAL_REPLAY_SNAPSHOT_FORMAT_V28",
    "RealisticEdgeArmEnvV13",
    "RealisticEnvV13Config",
    "keyboard_teleop_env_config_v13",
]
