"""Causal wrist-camera pose estimate from reported robot joint positions.

The rendered camera pose in MuJoCo is privileged simulator state. A deployed
robot can instead combine encoder-reported joints with a calibrated
camera-to-wrist transform. This module implements that deployable contract in
an independent kinematics workspace and never reads live ``qpos`` or
``cam_xpos`` while estimating a pose.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

import mujoco
import numpy as np

from .config import JOINT_NAMES


REPORTED_WRIST_POSE_FORMAT = "edgearm-reported-wrist-pose-v2"
REPORTED_WRIST_POSE_SEMANTICS = (
    "calibrated_reported_encoder_joint_position_plus_declared_fixed_wrist_camera_extrinsic"
)
FIXED_NOMINAL_CALIBRATION_SOURCE = "fixed_nominal_model"
SYNTHETIC_EPISODE_CALIBRATION_SOURCE = "synthetic_episode_declared_calibration"
SUPPORTED_CALIBRATION_PARAMETER_SOURCES = frozenset(
    {
        FIXED_NOMINAL_CALIBRATION_SOURCE,
        SYNTHETIC_EPISODE_CALIBRATION_SOURCE,
    }
)


class ReportedWristPoseEstimatorV1:
    """Estimate camera-to-world pose without consulting physical simulator state."""

    def __init__(
        self,
        model: mujoco.MjModel,
        *,
        camera_id: int,
        mount_position_m: np.ndarray,
        mount_quaternion_wxyz: np.ndarray,
        joint_position_offset_rad: np.ndarray | None = None,
        calibration_parameter_source: str = FIXED_NOMINAL_CALIBRATION_SOURCE,
    ) -> None:
        if not 0 <= int(camera_id) < model.ncam:
            raise ValueError("camera_id is outside the MuJoCo model")
        position = np.asarray(mount_position_m, dtype=np.float64)
        quaternion = np.asarray(mount_quaternion_wxyz, dtype=np.float64)
        if position.shape != (3,) or not np.all(np.isfinite(position)):
            raise ValueError("camera mount position must be a finite 3-vector")
        if quaternion.shape != (4,) or not np.all(np.isfinite(quaternion)):
            raise ValueError("camera mount quaternion must be a finite wxyz 4-vector")
        norm = float(np.linalg.norm(quaternion))
        if norm <= 1e-12:
            raise ValueError("camera mount quaternion cannot be zero")
        offset = np.zeros(len(JOINT_NAMES), dtype=np.float64)
        if joint_position_offset_rad is not None:
            offset = np.asarray(joint_position_offset_rad, dtype=np.float64)
        if offset.shape != (len(JOINT_NAMES),) or not np.all(np.isfinite(offset)):
            raise ValueError("joint position calibration offset must be a finite joint vector")
        if calibration_parameter_source not in SUPPORTED_CALIBRATION_PARAMETER_SOURCES:
            raise ValueError("unsupported wrist pose calibration parameter source")
        self._model = model
        self._data = mujoco.MjData(model)
        self._reference_qpos = np.asarray(model.qpos0, dtype=np.float64).copy()
        self._camera_body_id = int(model.cam_bodyid[int(camera_id)])
        self._mount_position = position.copy()
        self._mount_quaternion = quaternion / norm
        self._joint_position_offset = offset.copy()
        self._calibration_parameter_source = calibration_parameter_source
        mount_rotation = np.empty(9, dtype=np.float64)
        mujoco.mju_quat2Mat(mount_rotation, self._mount_quaternion)
        self._mount_rotation = mount_rotation.reshape(3, 3)
        self._profile = {
            "format": REPORTED_WRIST_POSE_FORMAT,
            "semantics": REPORTED_WRIST_POSE_SEMANTICS,
            "joint_count": len(JOINT_NAMES),
            "camera_body_id": self._camera_body_id,
            "mount_position_m": self._mount_position.tolist(),
            "mount_quaternion_wxyz": self._mount_quaternion.tolist(),
            "joint_position_offset_rad": self._joint_position_offset.tolist(),
            "calibration_parameter_source": self._calibration_parameter_source,
            "synthetic_calibration": True,
            "physically_calibrated": False,
            "simulator_ground_truth_used": False,
            "dynamic_simulator_state_used": False,
        }
        encoded = json.dumps(
            self._profile,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        self._profile["profile_sha256"] = hashlib.sha256(encoded).hexdigest()

    @classmethod
    def from_environment(cls, env: Any) -> "ReportedWristPoseEstimatorV1":
        camera_id = int(env._ids["cameras"]["wrist"])
        episode_sim2real = getattr(env, "episode_sim2real", None)
        if isinstance(episode_sim2real, dict) and "joint_zero_offset_rad" in episode_sim2real:
            joint_position_offset = np.asarray(
                episode_sim2real["joint_zero_offset_rad"],
                dtype=np.float64,
            )
            calibration_parameter_source = SYNTHETIC_EPISODE_CALIBRATION_SOURCE
        else:
            joint_position_offset = np.zeros(len(JOINT_NAMES), dtype=np.float64)
            calibration_parameter_source = FIXED_NOMINAL_CALIBRATION_SOURCE
        return cls(
            env.model,
            camera_id=camera_id,
            mount_position_m=np.asarray(env.model.cam_pos[camera_id], dtype=np.float64),
            mount_quaternion_wxyz=np.asarray(env.model.cam_quat[camera_id], dtype=np.float64),
            joint_position_offset_rad=joint_position_offset,
            calibration_parameter_source=calibration_parameter_source,
        )

    @property
    def metadata(self) -> dict[str, Any]:
        return dict(self._profile)

    def estimate(self, reported_joint_position: np.ndarray) -> np.ndarray:
        reported = np.asarray(reported_joint_position, dtype=np.float64)
        joint_count = len(JOINT_NAMES)
        if reported.shape != (joint_count,) or not np.all(np.isfinite(reported)):
            raise ValueError(f"reported_joint_position must be a finite {joint_count}-vector")
        self._data.qpos[:] = self._reference_qpos
        self._data.qvel[:] = 0.0
        self._data.qpos[:joint_count] = reported - self._joint_position_offset
        mujoco.mj_fwdPosition(self._model, self._data)
        body_position = np.asarray(
            self._data.xpos[self._camera_body_id],
            dtype=np.float64,
        )
        body_rotation = np.asarray(
            self._data.xmat[self._camera_body_id],
            dtype=np.float64,
        ).reshape(3, 3)
        world_position = body_position + body_rotation @ self._mount_position
        world_rotation = body_rotation @ self._mount_rotation
        pose = np.concatenate((world_position, world_rotation.reshape(9)))
        if pose.shape != (12,) or not np.all(np.isfinite(pose)):  # pragma: no cover
            raise RuntimeError("reported-joint camera pose estimate is invalid")
        return pose.astype(np.float32)


__all__ = [
    "FIXED_NOMINAL_CALIBRATION_SOURCE",
    "REPORTED_WRIST_POSE_FORMAT",
    "REPORTED_WRIST_POSE_SEMANTICS",
    "SUPPORTED_CALIBRATION_PARAMETER_SOURCES",
    "SYNTHETIC_EPISODE_CALIBRATION_SOURCE",
    "ReportedWristPoseEstimatorV1",
]
