"""Per-jaw fingertip-planning successor for the stock SO-101 follower.

V8 used each authored non-convex mesh as one collision geom and retained one
solid union box for planning.  MuJoCo collides a mesh through its convex hull,
which more than doubled the occupied volume of both stock parts; the union box
also filled empty space between the tapered fingertips.  V9 replaces those
collision hulls with a deterministic CoACD convex union, labels only the most
distal part of each jaw as a valid pushing contact, and uses two independent
non-colliding distal references for pre-contact planning.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import mujoco

from .production_env import (
    STOCK_GRIPPER_DIRECT_PUSH_TOOL_PROFILE,
    STOCK_GRIPPER_FIXED_AUTHORED_MESH_REFERENCE_GEOM,
    STOCK_GRIPPER_FIXED_TIP_REFERENCE_GEOM,
    STOCK_GRIPPER_FIXED_TIP_REFERENCE_HALF_EXTENTS_M,
    STOCK_GRIPPER_FIXED_TIP_REFERENCE_POSITION_GRIPPER,
    STOCK_GRIPPER_FIXED_TIP_REFERENCE_QUATERNION_WXYZ,
    STOCK_GRIPPER_MOVING_TIP_REFERENCE_GEOM,
    STOCK_GRIPPER_MOVING_TIP_REFERENCE_HALF_EXTENTS_M,
    STOCK_GRIPPER_MOVING_TIP_REFERENCE_POSITION_MOVING_BODY,
    STOCK_GRIPPER_MOVING_TIP_REFERENCE_QUATERNION_WXYZ,
    STOCK_GRIPPER_MOVING_AUTHORED_MESH_REFERENCE_GEOM,
    STOCK_GRIPPER_PLANNING_GEOM,
    STOCK_GRIPPER_TIP_REFERENCE_GEOMS,
    ProductionMjcfBundleV1,
)
from .sim2real_env_v7 import RealisticEdgeArmEnvV7
from .sim2real_env_v8 import RealisticEdgeArmEnvV8, RealisticEnvV8Config
from .stock_gripper_convex_decomposition_v1 import (
    STOCK_GRIPPER_CONVEX_DECOMPOSITION_FORMAT,
    STOCK_GRIPPER_CONVEX_DECOMPOSITION_PAYLOAD_SHA256,
    load_stock_gripper_convex_decomposition_v1,
)


STOCK_GRIPPER_DYNAMICS_PROFILE_V9 = "edgearm-sim2real-dynamics-v9-stock-distal-tips"
STOCK_GRIPPER_GEOMETRY_VERSION_V9 = "edgearm-stock-so101-gripper-contact-v2"
STOCK_GRIPPER_PARAMETER_SOURCE_V9 = (
    "so101_authored_meshes_coacd_convex_union_plus_per_jaw_distal_2mm_references_at_qgripper_-0.16"
)


@dataclass(frozen=True)
class RealisticEnvV9Config(RealisticEnvV8Config):
    """V8 dynamics with per-jaw distal-tip planning and safety references."""


class RealisticEdgeArmEnvV9(RealisticEdgeArmEnvV8):
    """Stock-gripper plant with separate fixed/moving fingertip references."""

    profile_version = STOCK_GRIPPER_DYNAMICS_PROFILE_V9
    model_tool_profile = STOCK_GRIPPER_DIRECT_PUSH_TOOL_PROFILE
    include_stock_tip_references = True

    def __init__(
        self,
        config: RealisticEnvV9Config | None = None,
        seed: int = 0,
        *,
        model_scene_path: Path | None = None,
        model_scene_bundle: ProductionMjcfBundleV1 | None = None,
    ) -> None:
        if config is not None and not isinstance(config, RealisticEnvV9Config):
            raise TypeError("RealisticEdgeArmEnvV9 requires RealisticEnvV9Config")
        self.stock_distal_tip_config = config or RealisticEnvV9Config()
        super().__init__(
            self.stock_distal_tip_config,
            seed=seed,
            model_scene_path=model_scene_path,
            model_scene_bundle=model_scene_bundle,
        )
        planning = tuple(int(value) for value in self._ids["tool_planning_geoms"])
        names = tuple(
            mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
            for geom_id in planning
        )
        if names != STOCK_GRIPPER_TIP_REFERENCE_GEOMS:
            raise RuntimeError("V9 did not compile both ordered distal-tip references")
        orientation = int(self._ids["tool_geom"])
        if (
            mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, orientation)
            != STOCK_GRIPPER_PLANNING_GEOM
        ):
            raise RuntimeError("V9 orientation reference is invalid")
        if orientation in planning:
            raise RuntimeError("V9 orientation and distal-tip references must be distinct")
        if any(
            int(self.model.geom_type[geom_id]) != int(mujoco.mjtGeom.mjGEOM_BOX)
            or int(self.model.geom_contype[geom_id]) != 0
            or int(self.model.geom_conaffinity[geom_id]) != 0
            for geom_id in planning
        ):
            raise RuntimeError("V9 distal-tip references must be non-colliding boxes")
        contacts = tuple(int(value) for value in self._ids["tool_contact_geoms"])
        safety = tuple(int(value) for value in self._ids["tool_safety_geoms"])
        if not set(contacts).issubset(safety) or len(safety) <= len(contacts):
            raise RuntimeError("V9 safety union must contain both distal contact parts")
        if any(
            int(self.model.geom_type[geom_id]) != int(mujoco.mjtGeom.mjGEOM_MESH)
            or int(self.model.geom_contype[geom_id]) == 0
            for geom_id in safety
        ):
            raise RuntimeError("V9 CoACD safety union contains a disabled or non-mesh geom")
        authored = tuple(int(value) for value in self._ids["tool_authored_mesh_reference_geoms"])
        authored_names = tuple(
            mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
            for geom_id in authored
        )
        if authored_names != (
            STOCK_GRIPPER_FIXED_AUTHORED_MESH_REFERENCE_GEOM,
            STOCK_GRIPPER_MOVING_AUTHORED_MESH_REFERENCE_GEOM,
        ) or any(
            int(self.model.geom_contype[geom_id]) != 0
            or int(self.model.geom_conaffinity[geom_id]) != 0
            for geom_id in authored
        ):
            raise RuntimeError("V9 authored non-convex meshes must remain non-colliding references")

    def _stock_gripper_profile(self) -> dict[str, Any]:
        decomposition = load_stock_gripper_convex_decomposition_v1()
        roles = dict(decomposition["roles"])
        contact_names = [
            mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, int(geom_id))
            for geom_id in self._ids["tool_contact_geoms"]
        ]
        return {
            "profile_version": STOCK_GRIPPER_DYNAMICS_PROFILE_V9,
            "geometry_version": STOCK_GRIPPER_GEOMETRY_VERSION_V9,
            "tool_profile": STOCK_GRIPPER_DIRECT_PUSH_TOOL_PROFILE,
            "parameter_source": STOCK_GRIPPER_PARAMETER_SOURCE_V9,
            "stock_follower_unmodified": True,
            "added_contact_tool": False,
            "stock_contact_geom_names": contact_names,
            "contact_authority": "distal_coacd_convex_parts_derived_from_authored_stock_jaw_meshes",
            "authored_nonconvex_mesh_collision_disabled": True,
            "collision_geometry_is_cad_derived_approximation": True,
            "convex_decomposition": {
                "format": STOCK_GRIPPER_CONVEX_DECOMPOSITION_FORMAT,
                "payload_sha256": STOCK_GRIPPER_CONVEX_DECOMPOSITION_PAYLOAD_SHA256,
                "fixed_part_count": int(dict(roles["fixed_jaw"])["part_count"]),
                "moving_part_count": int(dict(roles["moving_jaw"])["part_count"]),
                "safety_geom_count": len(self._ids["tool_safety_geoms"]),
                "safety_geometry_mode": self._ids["tool_safety_geometry_mode"],
            },
            "orientation_reference_geom_name": STOCK_GRIPPER_PLANNING_GEOM,
            "orientation_reference_is_noncolliding": True,
            "planning_geometry_mode": self._ids["tool_planning_geometry_mode"],
            "planning_reference_geom_names": list(STOCK_GRIPPER_TIP_REFERENCE_GEOMS),
            "planning_references_are_noncolliding": True,
            "planning_reference_derivation": {
                "mesh_slice": "distal_2mm_axis_aligned_contact_reference_bounds_in_gripper_frame",
                "gripper_joint_position_rad": self.tool_gripper_joint_position_rad,
                "fixed": {
                    "name": STOCK_GRIPPER_FIXED_TIP_REFERENCE_GEOM,
                    "mount_body": "gripper",
                    "position_m": list(STOCK_GRIPPER_FIXED_TIP_REFERENCE_POSITION_GRIPPER),
                    "quaternion_wxyz": list(
                        STOCK_GRIPPER_FIXED_TIP_REFERENCE_QUATERNION_WXYZ
                    ),
                    "half_extents_m": list(
                        STOCK_GRIPPER_FIXED_TIP_REFERENCE_HALF_EXTENTS_M
                    ),
                },
                "moving": {
                    "name": STOCK_GRIPPER_MOVING_TIP_REFERENCE_GEOM,
                    "mount_body": "moving_jaw_so101_v1",
                    "position_m": list(
                        STOCK_GRIPPER_MOVING_TIP_REFERENCE_POSITION_MOVING_BODY
                    ),
                    "quaternion_wxyz": list(
                        STOCK_GRIPPER_MOVING_TIP_REFERENCE_QUATERNION_WXYZ
                    ),
                    "half_extents_m": list(
                        STOCK_GRIPPER_MOVING_TIP_REFERENCE_HALF_EXTENTS_M
                    ),
                },
            },
            "wrist_camera_physically_calibrated": False,
            "physical_samples": 0,
            "physical_trials": 0,
            "physical_hardware_connected": False,
            "configuration": asdict(self.stock_distal_tip_config),
        }

    def reset(
        self,
        seed: int | None = None,
        *,
        obstacle: bool | None = None,
        stress: bool = False,
    ) -> dict[str, Any]:
        observation = RealisticEdgeArmEnvV7.reset(
            self,
            seed=seed,
            obstacle=obstacle,
            stress=stress,
        )
        self.episode_domain["realism_v9"] = self._stock_gripper_profile()
        return observation

    def step(self, action: Any) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        observation, reward, terminated, truncated, info = RealisticEdgeArmEnvV7.step(
            self,
            action,
        )
        info["realism_v9"] = dict(self.episode_domain["realism_v9"])
        return observation, reward, terminated, truncated, info


__all__ = [
    "RealisticEdgeArmEnvV9",
    "RealisticEnvV9Config",
    "STOCK_GRIPPER_DYNAMICS_PROFILE_V9",
    "STOCK_GRIPPER_GEOMETRY_VERSION_V9",
]
