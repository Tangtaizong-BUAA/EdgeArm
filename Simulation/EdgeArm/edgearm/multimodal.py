"""True MuJoCo camera capture for RGB, metric depth, and semantic masks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import mujoco
import numpy as np

from .production_env import ProductionEdgeArmEnv
from .trajectory_contract_v1 import METRIC_DEPTH_CONTRACT_FORMAT


HIDDEN_POLICY_RENDER_GEOM_GROUPS_V1 = (3, 5)


def policy_scene_option_v1() -> mujoco.MjvOption:
    """Hide duplicate collision/planning meshes from policy camera pixels."""

    option = mujoco.MjvOption()
    mujoco.mjv_defaultOption(option)
    for group in HIDDEN_POLICY_RENDER_GEOM_GROUPS_V1:
        option.geomgroup[group] = 0
    return option


def configure_policy_viewer_v1(viewer: Any) -> None:
    """Apply the same collision-geometry visibility contract to a live viewer."""

    with viewer.lock():
        for group in HIDDEN_POLICY_RENDER_GEOM_GROUPS_V1:
            viewer.opt.geomgroup[group] = 0


@dataclass(frozen=True)
class CameraCaptureConfig:
    width: int = 320
    height: int = 240
    source_width: int = 640
    source_height: int = 480
    cameras: tuple[str, ...] = ("wrist",)
    rgb_noise_std: float = 2.2
    depth_noise_mm: float = 1.8
    depth_dropout_probability: float = 0.002
    apply_lens_distortion: bool = True


class TrueMultimodalRenderer:
    _FAR_SENTINEL_GUARD_MIN_M = 1.0e-3

    def __init__(self, env: ProductionEdgeArmEnv, config: CameraCaptureConfig | None = None):
        self.env = env
        self.config = config or CameraCaptureConfig()
        extent = float(env.model.stat.extent)
        self._depth_clip_near_m = float(env.model.vis.map.znear) * extent
        self._depth_clip_far_m = float(env.model.vis.map.zfar) * extent
        if (
            not np.isfinite(self._depth_clip_near_m)
            or not np.isfinite(self._depth_clip_far_m)
            or self._depth_clip_near_m <= 0.0
            or self._depth_clip_far_m <= self._depth_clip_near_m
        ):
            raise ValueError("MuJoCo depth clipping planes must be finite and ordered")
        far_guard_m = max(
            self._FAR_SENTINEL_GUARD_MIN_M,
            self._depth_clip_far_m * 1.0e-4,
        )
        self._depth_valid_max_exclusive_m = self._depth_clip_far_m - far_guard_m
        if self._depth_valid_max_exclusive_m <= self._depth_clip_near_m:
            raise ValueError("MuJoCo depth clipping interval is too narrow")
        self.renderer = mujoco.Renderer(
            env.model, height=self.config.height, width=self.config.width, max_geom=10_000
        )
        self.scene_option = policy_scene_option_v1()
        self.rng = np.random.default_rng(env.seed)
        self.episode_noise: dict[str, dict[str, float | list[float]]] = {}
        self._semantic_lut = self._build_semantic_lut()
        yy, xx = np.meshgrid(
            np.linspace(-1.0, 1.0, self.config.height),
            np.linspace(-1.0, 1.0, self.config.width),
            indexing="ij",
        )
        self._grid_x = xx
        self._grid_y = yy

    def begin_episode(self, seed: int) -> None:
        self.rng = np.random.default_rng(seed ^ 0x5A17C0DE)
        self.episode_noise = {}
        for camera in self.config.cameras:
            distortion_scale = 1.0 if self.config.apply_lens_distortion else 0.0
            self.episode_noise[camera] = {
                "exposure": float(self.rng.uniform(0.78, 1.24)),
                "gamma": float(self.rng.uniform(0.88, 1.14)),
                "white_balance": self.rng.uniform(0.92, 1.08, 3).tolist(),
                "vignette": float(self.rng.uniform(0.0, 0.20)),
                "k1": float(self.rng.uniform(-0.035, 0.025) * distortion_scale),
                "k2": float(self.rng.uniform(-0.012, 0.012) * distortion_scale),
                "p1": float(self.rng.uniform(-0.002, 0.002) * distortion_scale),
                "p2": float(self.rng.uniform(-0.002, 0.002) * distortion_scale),
                "motion_blur": float(self.rng.random() < 0.12),
                "lens_distortion_applied": bool(self.config.apply_lens_distortion),
            }

    def capture(self) -> dict[str, np.ndarray]:
        output: dict[str, np.ndarray] = {}
        for camera in self.config.cameras:
            self.renderer.disable_depth_rendering()
            self.renderer.disable_segmentation_rendering()
            self.renderer.update_scene(
                self.env.data,
                camera=f"edgearm_{camera}",
                scene_option=self.scene_option,
            )
            rgb = self.renderer.render().copy()

            self.renderer.enable_depth_rendering()
            self.renderer.update_scene(
                self.env.data,
                camera=f"edgearm_{camera}",
                scene_option=self.scene_option,
            )
            depth = self._sanitize_metric_depth(self.renderer.render().copy())
            self.renderer.disable_depth_rendering()

            self.renderer.enable_segmentation_rendering()
            self.renderer.update_scene(
                self.env.data,
                camera=f"edgearm_{camera}",
                scene_option=self.scene_option,
            )
            raw_segmentation = self.renderer.render().copy()
            self.renderer.disable_segmentation_rendering()

            semantic = self._semantic_mask(raw_segmentation)
            rgb, depth, semantic = self._sensor_model(camera, rgb, depth, semantic)
            output[f"rgb_{camera}"] = rgb
            output[f"depth_{camera}_mm"] = depth
            output[f"segmentation_{camera}"] = semantic
        return output

    def capture_rgb_only(self) -> dict[str, np.ndarray]:
        output: dict[str, np.ndarray] = {}
        for camera in self.config.cameras:
            self.renderer.disable_depth_rendering()
            self.renderer.disable_segmentation_rendering()
            self.renderer.update_scene(
                self.env.data,
                camera=f"edgearm_{camera}",
                scene_option=self.scene_option,
            )
            rgb = self.renderer.render().copy()
            blank_depth = np.ones(rgb.shape[:2], dtype=np.float32)
            blank_segmentation = np.zeros(rgb.shape[:2], dtype=np.uint8)
            rgb, _, _ = self._sensor_model(camera, rgb, blank_depth, blank_segmentation)
            output[f"rgb_{camera}"] = rgb
        return output

    def calibration_metadata(self) -> dict[str, Any]:
        calibration = self.env.camera_calibration(self.config.width, self.config.height)
        for camera, noise in self.episode_noise.items():
            calibration[camera]["distortion_k1_k2_p1_p2_k3"] = [
                noise["k1"],
                noise["k2"],
                noise["p1"],
                noise["p2"],
                0.0,
            ]
            calibration[camera]["sensor_model"] = noise
            # Renderer produces the training tensor directly at width/height.
            # ``source_resolution`` is reference sensor metadata only, not a
            # claim that a native-resolution source frame was captured then
            # downsampled.
            calibration[camera]["source_resolution"] = [
                self.config.source_width,
                self.config.source_height,
            ]
            calibration[camera]["source_resolution_semantics"] = (
                "reference sensor resolution metadata; renderer emits the persisted tensor "
                "directly at capture width/height"
            )
            calibration[camera]["rendered_resolution"] = [
                self.config.width,
                self.config.height,
            ]
            calibration[camera]["stored_resolution"] = [self.config.width, self.config.height]
            calibration[camera]["metric_depth_contract"] = {
                "format": METRIC_DEPTH_CONTRACT_FORMAT,
                "clip_near_m": self._depth_clip_near_m,
                "clip_far_m": self._depth_clip_far_m,
                "valid_min_exclusive_m": self._depth_clip_near_m,
                "valid_max_exclusive_m": self._depth_valid_max_exclusive_m,
                "invalid_depth_mm": 0,
                "far_plane_no_hit_is_zeroed_before_sensor_noise": True,
            }
        return calibration

    def close(self) -> None:
        self.renderer.close()

    def _sanitize_metric_depth(self, depth_m: np.ndarray) -> np.ndarray:
        depth = np.asarray(depth_m, dtype=np.float32)
        valid = (
            np.isfinite(depth)
            & (depth > self._depth_clip_near_m)
            & (depth < self._depth_valid_max_exclusive_m)
        )
        return np.where(valid, depth, 0.0).astype(np.float32, copy=False)

    def _build_semantic_lut(self) -> np.ndarray:
        lut = np.zeros(self.env.model.ngeom, dtype=np.uint8)
        ids = self.env._ids
        lut[ids["block_geom"]] = 2
        lut[ids["target_geom"]] = 3
        lut[ids["obstacle_geom"]] = 4
        for geom_id in range(self.env.model.ngeom):
            name = (
                mujoco.mj_id2name(
                    self.env.model,
                    mujoco.mjtObj.mjOBJ_GEOM,
                    geom_id,
                )
                or ""
            )
            if name.startswith("generalization_obstacle_"):
                lut[geom_id] = 4
        for tool_geom in ids.get("tool_contact_geoms", (ids["tool_geom"],)):
            lut[int(tool_geom)] = 5
        # The physical contact authority lives in hidden group-3 convex parts.
        # Assign the authored group-2 jaw visuals the same tool label so a
        # clean policy render does not lose the gripper semantic class.
        for geom_id in range(self.env.model.ngeom):
            if int(self.env.model.geom_group[geom_id]) != 2 or int(self.env.model.geom_type[geom_id]) != int(
                mujoco.mjtGeom.mjGEOM_MESH
            ):
                continue
            mesh_id = int(self.env.model.geom_dataid[geom_id])
            if mesh_id < 0:
                continue
            mesh_name = mujoco.mj_id2name(
                self.env.model,
                mujoco.mjtObj.mjOBJ_MESH,
                mesh_id,
            )
            if mesh_name in {
                "wrist_roll_follower_so101_v1",
                "moving_jaw_so101_v1",
            }:
                lut[geom_id] = 5
        desk = mujoco.mj_name2id(self.env.model, mujoco.mjtObj.mjOBJ_GEOM, "edgearm_desk")
        if desk >= 0:
            lut[desk] = 6
        for geom_id in range(self.env.model.ngeom):
            body_id = self.env.model.geom_bodyid[geom_id]
            body_name = mujoco.mj_id2name(self.env.model, mujoco.mjtObj.mjOBJ_BODY, body_id) or ""
            if (
                body_name
                in {
                    "base",
                    "shoulder",
                    "upper_arm",
                    "lower_arm",
                    "wrist",
                    "gripper",
                    "moving_jaw_so101_v1",
                }
                and lut[geom_id] == 0
            ):
                lut[geom_id] = 1
        return lut

    def _semantic_mask(self, raw: np.ndarray) -> np.ndarray:
        if raw.ndim != 3 or raw.shape[-1] != 2:
            raise ValueError(f"Unexpected MuJoCo segmentation shape: {raw.shape}")
        # MuJoCo returns segmentation as (object id, object type).
        object_id = raw[..., 0]
        object_type = raw[..., 1]
        mask = np.zeros(raw.shape[:2], dtype=np.uint8)
        geom_pixels = object_type == int(mujoco.mjtObj.mjOBJ_GEOM)
        valid = geom_pixels & (object_id >= 0) & (object_id < len(self._semantic_lut))
        mask[valid] = self._semantic_lut[object_id[valid]]
        return mask

    def _sensor_model(
        self, camera: str, rgb: np.ndarray, depth_m: np.ndarray, semantic: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        noise = self.episode_noise[camera]
        image = rgb.astype(np.float32) / 255.0
        image *= float(noise["exposure"])
        image *= np.asarray(noise["white_balance"], dtype=np.float32)
        image = np.clip(image, 0.0, 1.0) ** float(noise["gamma"])
        radius2 = self._grid_x**2 + self._grid_y**2
        image *= (1.0 - float(noise["vignette"]) * np.clip(radius2, 0.0, 1.0))[..., None]
        image += self.rng.normal(0.0, self.config.rgb_noise_std / 255.0, image.shape)
        if float(noise["motion_blur"]) > 0:
            image = (image + np.roll(image, 1, axis=1) + np.roll(image, 2, axis=1)) / 3.0
        rgb_out = np.clip(image * 255.0, 0, 255).astype(np.uint8)

        depth_valid = np.isfinite(depth_m) & (depth_m > 0.0)
        depth_mm = np.where(depth_valid, depth_m * 1000.0, 0.0)
        depth_noise = self.rng.normal(0.0, self.config.depth_noise_mm, depth_mm.shape)
        depth_mm = np.where(depth_valid, depth_mm + depth_noise, 0.0)
        dropout = (self.rng.random(depth_mm.shape) < self.config.depth_dropout_probability) & depth_valid
        depth_mm[dropout | ~depth_valid] = 0.0
        depth_out = np.clip(depth_mm, 0, 65_535).astype(np.uint16)

        k1, k2 = float(noise["k1"]), float(noise["k2"])
        p1, p2 = float(noise["p1"]), float(noise["p2"])
        if any(abs(value) > 1e-9 for value in (k1, k2, p1, p2)):
            source_x, source_y = self._distortion_map(k1, k2, p1, p2)
            rgb_out = rgb_out[source_y, source_x]
            depth_out = depth_out[source_y, source_x]
            semantic = semantic[source_y, source_x]
        return rgb_out, depth_out, semantic

    def _distortion_map(self, k1: float, k2: float, p1: float, p2: float) -> tuple[np.ndarray, np.ndarray]:
        x, y = self._grid_x, self._grid_y
        r2 = x * x + y * y
        radial = 1.0 + k1 * r2 + k2 * r2 * r2
        distorted_x = x * radial + 2 * p1 * x * y + p2 * (r2 + 2 * x * x)
        distorted_y = y * radial + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y
        source_x = np.clip(
            ((distorted_x + 1.0) * 0.5 * (self.config.width - 1)).round(), 0, self.config.width - 1
        )
        source_y = np.clip(
            ((distorted_y + 1.0) * 0.5 * (self.config.height - 1)).round(), 0, self.config.height - 1
        )
        return source_x.astype(np.int32), source_y.astype(np.int32)
