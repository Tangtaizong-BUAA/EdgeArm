"""Mounted wrist-camera geometry successor for interactive phone collection.

V10 and V11 are frozen evidence environments.  This opt-in V12 profile leaves
their source and compiled-model contracts untouched while correcting the
visible wrist assembly used by new phone demonstrations: all six authored
STS3215 motor meshes remain present, and the synthetic camera housing now has
a rigid printable bridge and a visible lens instead of appearing as a floating
black box.

The mount dimensions are still an uncalibrated prior.  This module does not
claim the user has selected, measured, or installed a physical wrist camera.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

import mujoco
import numpy as np

from .production_env import (
    WRIST_CAMERA_HOUSING_GEOM,
    WRIST_CAMERA_LENS_GEOM,
    WRIST_CAMERA_MOUNT_GEOM,
    WRIST_CAMERA_PHYSICAL_MOUNT_PROFILE_V1,
    build_production_model,
)
from .sim2real_env_v10 import RealisticEdgeArmEnvV10, RealisticEnvV10Config


MOUNTED_WRIST_CAMERA_DYNAMICS_PROFILE_V12 = "edgearm-sim2real-v12-stock-so101-mounted-wrist-camera-v1"
MOUNTED_WRIST_CAMERA_PARAMETER_SOURCE_V12 = (
    "exact_v10_stock_so101_plus_synthetic_rigid_camera_bridge_no_physical_measurement"
)
EXPECTED_MOTOR_VISUAL_BODIES_V12 = (
    "base",
    "shoulder",
    "upper_arm",
    "lower_arm",
    "wrist",
    "gripper",
)

# V12 introduces no new tunable plant parameter.  The exact V10 dataclass is
# retained so the inherited fail-closed constructor remains authoritative.
RealisticEnvV12Config = RealisticEnvV10Config


class RealisticEdgeArmEnvV12(RealisticEdgeArmEnvV10):
    """Exact V10 controller with corrected, explicitly provisional mounting."""

    profile_version = MOUNTED_WRIST_CAMERA_DYNAMICS_PROFILE_V12

    def _build_model(self) -> mujoco.MjModel:
        model = build_production_model(
            scene_path=self._model_scene_path,
            scene_bundle=self._model_scene_bundle,
            include_camera_housing=True,
            include_camera_mount=True,
            include_visual_clutter=True,
            include_realism_contact_pairs=True,
            include_stock_tip_references=self.include_stock_tip_references,
            tool_profile=self.model_tool_profile,
        )
        # V10 reaches V7's workbench-edge correction through super().  V12
        # builds an opt-in MjSpec directly so it repeats that exact geometric
        # correction here without modifying any frozen V7/V10/V11 source.
        desk = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "edgearm_desk")
        if desk < 0:  # pragma: no cover - guarded by the production builder
            raise ValueError("V12 model is missing edgearm_desk")
        original_left = float(model.geom_pos[desk, 0] - model.geom_size[desk, 0])
        original_right = float(model.geom_pos[desk, 0] + model.geom_size[desk, 0])
        front = float(self.contact_feasible_config.desk_front_edge_x_m)
        if front >= original_right:
            raise ValueError("V12 desk front edge would remove the work surface")
        model.geom_pos[desk, 0] = 0.5 * (front + original_right)
        model.geom_size[desk, 0] = 0.5 * (original_right - front)
        self._desk_original_x_bounds = np.asarray([original_left, original_right], dtype=np.float64)
        self._desk_corrected_x_bounds = np.asarray([front, original_right], dtype=np.float64)
        return model

    def wrist_assembly_contract_v12(self) -> dict[str, Any]:
        motor_bodies: list[str] = []
        for geom_id in range(self.model.ngeom):
            if int(self.model.geom_group[geom_id]) != 2:
                continue
            mesh_id = int(self.model.geom_dataid[geom_id])
            if mesh_id < 0 or int(self.model.geom_type[geom_id]) != int(mujoco.mjtGeom.mjGEOM_MESH):
                continue
            mesh_name = mujoco.mj_id2name(
                self.model,
                mujoco.mjtObj.mjOBJ_MESH,
                mesh_id,
            )
            if mesh_name not in {"sts3215_03a_v1", "sts3215_03a_no_horn_v1"}:
                continue
            body_name = mujoco.mj_id2name(
                self.model,
                mujoco.mjtObj.mjOBJ_BODY,
                int(self.model.geom_bodyid[geom_id]),
            )
            if body_name is None:
                raise RuntimeError("motor visual has no owning body")
            motor_bodies.append(body_name)

        mounted_geoms = {}
        for name in (
            WRIST_CAMERA_HOUSING_GEOM,
            WRIST_CAMERA_MOUNT_GEOM,
            WRIST_CAMERA_LENS_GEOM,
        ):
            geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, name)
            if geom_id < 0:
                raise RuntimeError(f"V12 mounted-camera model is missing {name}")
            body_name = mujoco.mj_id2name(
                self.model,
                mujoco.mjtObj.mjOBJ_BODY,
                int(self.model.geom_bodyid[geom_id]),
            )
            mounted_geoms[name] = {
                "geom_id": int(geom_id),
                "body": body_name,
                "colliding": bool(self.model.geom_contype[geom_id]),
            }

        return {
            "profile_version": WRIST_CAMERA_PHYSICAL_MOUNT_PROFILE_V1,
            "parameter_source": MOUNTED_WRIST_CAMERA_PARAMETER_SOURCE_V12,
            "motor_visual_bodies": motor_bodies,
            "expected_motor_visual_bodies": list(EXPECTED_MOTOR_VISUAL_BODIES_V12),
            "six_authored_motor_visuals_present": tuple(motor_bodies) == EXPECTED_MOTOR_VISUAL_BODIES_V12,
            "mounted_geometries": mounted_geoms,
            "all_mount_geometries_owned_by_gripper": all(
                item["body"] == "gripper" for item in mounted_geoms.values()
            ),
            "stock_follower_unmodified": True,
            "added_contact_tool": False,
            "camera_mount_physically_calibrated": False,
            "physical_camera_selected": False,
            "physical_samples": 0,
        }

    def camera_calibration(self, width: int, height: int) -> dict[str, dict[str, Any]]:
        calibration = super().camera_calibration(width, height)
        calibration["wrist"].update(
            {
                "mechanical_mount_profile_version": (WRIST_CAMERA_PHYSICAL_MOUNT_PROFILE_V1),
                "mechanical_mount_parameter_source": (MOUNTED_WRIST_CAMERA_PARAMETER_SOURCE_V12),
                "mechanical_mount_parent_body": "gripper",
                "six_authored_motor_visuals_present": True,
                "physical_camera_selected": False,
                "mechanical_mount_physically_calibrated": False,
            }
        )
        return calibration

    def reset(
        self,
        seed: int | None = None,
        *,
        obstacle: bool | None = None,
        stress: bool = False,
    ) -> dict[str, Any]:
        observation = super().reset(seed=seed, obstacle=obstacle, stress=stress)
        self.episode_domain["realism_v12"] = {
            "profile_version": MOUNTED_WRIST_CAMERA_DYNAMICS_PROFILE_V12,
            "parameter_source": MOUNTED_WRIST_CAMERA_PARAMETER_SOURCE_V12,
            "configuration": asdict(self.joint_bounded_config),
            "wrist_assembly": self.wrist_assembly_contract_v12(),
        }
        return observation
