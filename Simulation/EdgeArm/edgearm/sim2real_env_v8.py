"""Stock-gripper successor to the contact-feasible synthetic V7 plant.

V7 remains reproducible with its historical attached push plate.  V8 models
the user's unmodified SO-101 follower: the authored fixed and moving jaw mesh
collisions push the block directly, while a non-colliding model-derived box is
used only for IK face orientation and predictive clearance.  The wrist camera
remains an explicitly uncalibrated synthetic sensor prior.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import mujoco

from .production_env import (
    STOCK_GRIPPER_DIRECT_PUSH_TOOL_PROFILE,
    STOCK_GRIPPER_PLANNING_GEOM,
    STOCK_GRIPPER_PLANNING_HALF_EXTENTS_M,
    STOCK_GRIPPER_PLANNING_POSITION_GRIPPER,
    STOCK_GRIPPER_PLANNING_QUATERNION_WXYZ,
    ProductionMjcfBundleV1,
)
from .sim2real_env_v7 import RealisticEdgeArmEnvV7, RealisticEnvV7Config


STOCK_GRIPPER_DYNAMICS_PROFILE_VERSION = "edgearm-sim2real-dynamics-v8-stock-gripper"
STOCK_GRIPPER_GEOMETRY_VERSION = "edgearm-stock-so101-gripper-contact-v1"
STOCK_GRIPPER_PARAMETER_SOURCE = (
    "so101_new_calib_authored_collision_meshes_plus_model_derived_closed_tip_envelope"
)


@dataclass(frozen=True)
class RealisticEnvV8Config(RealisticEnvV7Config):
    """V7 dynamics with an unmodified stock SO-101 gripper contact tool."""


class RealisticEdgeArmEnvV8(RealisticEdgeArmEnvV7):
    """Contact-feasible plant whose block contacts come only from stock jaws."""

    profile_version = STOCK_GRIPPER_DYNAMICS_PROFILE_VERSION
    model_tool_profile = STOCK_GRIPPER_DIRECT_PUSH_TOOL_PROFILE

    def __init__(
        self,
        config: RealisticEnvV8Config | None = None,
        seed: int = 0,
        *,
        model_scene_path: Path | None = None,
        model_scene_bundle: ProductionMjcfBundleV1 | None = None,
    ) -> None:
        if config is not None and not isinstance(config, RealisticEnvV8Config):
            raise TypeError("RealisticEdgeArmEnvV8 requires RealisticEnvV8Config")
        self.stock_gripper_config = config or RealisticEnvV8Config()
        super().__init__(
            self.stock_gripper_config,
            seed=seed,
            model_scene_path=model_scene_path,
            model_scene_bundle=model_scene_bundle,
        )
        if self._ids["tool_profile"] != STOCK_GRIPPER_DIRECT_PUSH_TOOL_PROFILE:
            raise RuntimeError("V8 did not compile the stock-gripper tool profile")
        planning = int(self._ids["tool_geom"])
        contacts = tuple(int(value) for value in self._ids["tool_contact_geoms"])
        if len(contacts) != 2 or planning in contacts:
            raise RuntimeError("V8 stock-gripper contact geometry contract is invalid")
        if int(self.model.geom_contype[planning]) != 0 or int(self.model.geom_conaffinity[planning]) != 0:
            raise RuntimeError("V8 planning envelope must remain non-colliding")
        if any(int(self.model.geom_contype[geom]) == 0 for geom in contacts):
            raise RuntimeError("V8 stock jaw collision geometry is disabled")

    def _stock_gripper_profile(self) -> dict[str, Any]:
        contact_names = [
            mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, int(geom_id))
            for geom_id in self._ids["tool_contact_geoms"]
        ]
        return {
            "profile_version": STOCK_GRIPPER_DYNAMICS_PROFILE_VERSION,
            "geometry_version": STOCK_GRIPPER_GEOMETRY_VERSION,
            "tool_profile": STOCK_GRIPPER_DIRECT_PUSH_TOOL_PROFILE,
            "parameter_source": STOCK_GRIPPER_PARAMETER_SOURCE,
            "stock_follower_unmodified": True,
            "added_contact_tool": False,
            "stock_contact_geom_names": contact_names,
            "planning_reference_geom_name": STOCK_GRIPPER_PLANNING_GEOM,
            "planning_reference_is_noncolliding": True,
            "planning_reference_position_gripper_m": list(
                STOCK_GRIPPER_PLANNING_POSITION_GRIPPER
            ),
            "planning_reference_quaternion_wxyz": list(
                STOCK_GRIPPER_PLANNING_QUATERNION_WXYZ
            ),
            "planning_reference_half_extents_m": list(
                STOCK_GRIPPER_PLANNING_HALF_EXTENTS_M
            ),
            "gripper_push_joint_position_rad": self.tool_gripper_joint_position_rad,
            "wrist_camera_physically_calibrated": False,
            "physical_samples": 0,
            "physical_trials": 0,
            "physical_hardware_connected": False,
            "configuration": asdict(self.stock_gripper_config),
        }

    def reset(
        self,
        seed: int | None = None,
        *,
        obstacle: bool | None = None,
        stress: bool = False,
    ) -> dict[str, Any]:
        observation = super().reset(seed=seed, obstacle=obstacle, stress=stress)
        self.episode_domain["realism_v8"] = self._stock_gripper_profile()
        return observation

    def step(self, action: Any) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        observation, reward, terminated, truncated, info = super().step(action)
        info["realism_v8"] = dict(self.episode_domain["realism_v8"])
        return observation, reward, terminated, truncated, info


__all__ = [
    "RealisticEdgeArmEnvV8",
    "RealisticEnvV8Config",
    "STOCK_GRIPPER_DYNAMICS_PROFILE_VERSION",
    "STOCK_GRIPPER_GEOMETRY_VERSION",
]
