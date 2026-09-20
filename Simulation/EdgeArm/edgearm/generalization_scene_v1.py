"""Deterministic, replayable scene contracts for V13 keyboard collection.

The sampler is deliberately independent of MuJoCo.  It first assigns every
attempt to explicit strata, then realizes exact metric values and obstacle
poses from the episode's already sampled block/target locations.  The complete
realized contract is stored with the motion trace; replay never resamples it.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from typing import Any, Mapping

import numpy as np


GENERALIZATION_SCENE_SCHEMA_VERSION = "edgearm-generalization-scene-v1"
GENERALIZATION_SCHEDULE_VERSION = "edgearm-stratified-generalization-schedule-v1"
FRICTION_TIERS = ("low", "medium", "high")
LIGHTING_TIERS = ("low", "medium", "high")
OBJECT_SHAPES = ("box", "cylinder", "ellipsoid")
OBJECT_SIZE_TIERS = ("small", "medium", "large")
OBJECT_MATERIALS = ("matte", "satin", "glossy")
OBSTACLE_SHAPES = ("box", "cylinder", "ellipsoid")
OBSTACLE_LAYOUTS = (
    "clear",
    "direct_path",
    "gate",
    "slalom",
    "target_edge",
    "target_clutter",
)


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def contract_sha256(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _finite_tuple(value: Any, length: int, name: str) -> tuple[float, ...]:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (length,) or not np.isfinite(array).all():
        raise ValueError(f"{name} must contain {length} finite values")
    return tuple(float(item) for item in array)


@dataclass(frozen=True)
class ObstacleSpecV1:
    slot: int
    shape: str
    center_xy_m: tuple[float, float]
    size_xy_m: tuple[float, float]
    half_height_m: float
    yaw_rad: float
    friction: float
    rgba: tuple[float, float, float, float]
    along_path_fraction: float

    def __post_init__(self) -> None:
        if self.slot not in (0, 1, 2):
            raise ValueError("obstacle slot must be 0, 1, or 2")
        if self.shape not in OBSTACLE_SHAPES:
            raise ValueError(f"unsupported obstacle shape: {self.shape}")
        object.__setattr__(self, "center_xy_m", _finite_tuple(self.center_xy_m, 2, "center_xy_m"))
        size = _finite_tuple(self.size_xy_m, 2, "size_xy_m")
        if min(size) < 0.008 or max(size) > 0.045:
            raise ValueError("obstacle planar radii/half-extents are outside the V13 contract")
        if self.shape == "cylinder" and not np.isclose(size[0], size[1], atol=1e-12):
            raise ValueError("cylinder obstacle requires equal planar radii")
        object.__setattr__(self, "size_xy_m", size)
        if not np.isfinite(self.half_height_m) or not 0.012 <= self.half_height_m <= 0.080:
            raise ValueError("obstacle half-height is outside the V13 contract")
        if not np.isfinite(self.yaw_rad):
            raise ValueError("obstacle yaw must be finite")
        if not np.isfinite(self.friction) or not 0.2 <= self.friction <= 1.6:
            raise ValueError("obstacle friction is outside the V13 contract")
        rgba = _finite_tuple(self.rgba, 4, "rgba")
        if min(rgba) < 0.0 or max(rgba) > 1.0 or rgba[3] <= 0.0:
            raise ValueError("obstacle RGBA is invalid")
        object.__setattr__(self, "rgba", rgba)
        if not np.isfinite(self.along_path_fraction):
            raise ValueError("along_path_fraction must be finite")

    def contract(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class GeneralizationScenarioV1:
    requested_seed: int
    schedule_index: int
    friction_tier: str
    block_friction: float
    lighting_tier: str
    light_scale: float
    light_ambient_scale: float
    object_shape: str
    object_size_tier: str
    object_half_extents_m: tuple[float, float, float]
    block_mass_kg: float
    object_material: str
    obstacle_layout: str
    obstacles: tuple[ObstacleSpecV1, ...]
    block_color: str
    target_color: str
    block_initial_xy_m: tuple[float, float]
    target_xy_m: tuple[float, float]
    schema_version: str = GENERALIZATION_SCENE_SCHEMA_VERSION
    schedule_version: str = GENERALIZATION_SCHEDULE_VERSION

    def __post_init__(self) -> None:
        if type(self.requested_seed) is not int or self.requested_seed < 0:
            raise ValueError("requested_seed must be a non-negative integer")
        if type(self.schedule_index) is not int or self.schedule_index < 0:
            raise ValueError("schedule_index must be a non-negative integer")
        if self.schema_version != GENERALIZATION_SCENE_SCHEMA_VERSION:
            raise ValueError("generalization scene schema mismatch")
        if self.schedule_version != GENERALIZATION_SCHEDULE_VERSION:
            raise ValueError("generalization schedule version mismatch")
        if self.friction_tier not in FRICTION_TIERS:
            raise ValueError("invalid friction tier")
        if self.lighting_tier not in LIGHTING_TIERS:
            raise ValueError("invalid lighting tier")
        if self.object_shape not in OBJECT_SHAPES:
            raise ValueError("invalid manipulated-object shape")
        if self.object_size_tier not in OBJECT_SIZE_TIERS:
            raise ValueError("invalid manipulated-object size tier")
        if self.object_material not in OBJECT_MATERIALS:
            raise ValueError("invalid manipulated-object material")
        if self.obstacle_layout not in OBSTACLE_LAYOUTS:
            raise ValueError("invalid obstacle layout")
        if not np.isfinite(self.block_friction) or not 0.25 <= self.block_friction <= 1.45:
            raise ValueError("block friction is outside the V13 contract")
        if not np.isfinite(self.light_scale) or not 0.35 <= self.light_scale <= 1.55:
            raise ValueError("light scale is outside the V13 contract")
        if not np.isfinite(self.light_ambient_scale) or not 0.35 <= self.light_ambient_scale <= 1.55:
            raise ValueError("ambient light scale is outside the V13 contract")
        extents = _finite_tuple(self.object_half_extents_m, 3, "object_half_extents_m")
        if min(extents) < 0.014 or max(extents) > 0.034:
            raise ValueError("manipulated-object size is outside the V13 contract")
        if self.object_shape == "cylinder" and not np.isclose(extents[0], extents[1], atol=1e-12):
            raise ValueError("cylinder object requires equal planar radii")
        object.__setattr__(self, "object_half_extents_m", extents)
        if not np.isfinite(self.block_mass_kg) or not 0.01 <= self.block_mass_kg <= 1.0:
            raise ValueError("manipulated-object mass is outside the V13 contract")
        object.__setattr__(
            self, "block_initial_xy_m", _finite_tuple(self.block_initial_xy_m, 2, "block_initial_xy_m")
        )
        object.__setattr__(self, "target_xy_m", _finite_tuple(self.target_xy_m, 2, "target_xy_m"))
        obstacles = tuple(self.obstacles)
        if len(obstacles) > 3 or tuple(item.slot for item in obstacles) != tuple(range(len(obstacles))):
            raise ValueError("obstacles must occupy contiguous slots starting at zero")
        if self.obstacle_layout == "clear" and obstacles:
            raise ValueError("clear layout cannot contain obstacles")
        if self.obstacle_layout != "clear" and not obstacles:
            raise ValueError("non-clear layout requires obstacles")
        object.__setattr__(self, "obstacles", obstacles)
        if not self.block_color or not self.target_color:
            raise ValueError("scene colors must be non-empty")

    @property
    def obstacle_count(self) -> int:
        return len(self.obstacles)

    def contract(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def contract_json(self) -> str:
        return canonical_json(self.contract())

    @property
    def contract_sha256(self) -> str:
        return hashlib.sha256(self.contract_json.encode("utf-8")).hexdigest()

    @classmethod
    def from_contract(cls, value: Mapping[str, Any]) -> "GeneralizationScenarioV1":
        payload = dict(value)
        payload["obstacles"] = tuple(ObstacleSpecV1(**dict(item)) for item in payload.get("obstacles", ()))
        for name in ("object_half_extents_m", "block_initial_xy_m", "target_xy_m"):
            if name in payload:
                payload[name] = tuple(payload[name])
        return cls(**payload)


def _stratum(index: int, values: tuple[str, ...], stride: int, phase: int) -> str:
    return values[(stride * index + index // len(values) + phase) % len(values)]


def _object_extents(shape: str, tier: str, rng: np.random.Generator) -> tuple[float, float, float]:
    bounds = {
        "small": (0.017, 0.021),
        "medium": (0.023, 0.027),
        "large": (0.029, 0.033),
    }[tier]
    nominal = float(rng.uniform(*bounds))
    half_height = float(np.clip(nominal * rng.uniform(0.82, 1.08), 0.014, 0.034))
    if shape == "box":
        aspect = float(rng.uniform(0.82, 1.18))
        return (float(np.clip(nominal * aspect, 0.014, 0.034)), nominal, half_height)
    if shape == "cylinder":
        return (nominal, nominal, half_height)
    return (
        float(np.clip(nominal * rng.uniform(0.90, 1.12), 0.014, 0.034)),
        float(np.clip(nominal * rng.uniform(0.72, 0.92), 0.014, 0.034)),
        half_height,
    )


def _layout_rows(layout: str) -> tuple[tuple[float, float], ...]:
    return {
        "clear": (),
        "direct_path": ((0.54, 0.000),),
        "gate": ((0.55, -0.055), (0.55, 0.055)),
        "slalom": ((0.31, 0.052), (0.55, -0.052), (0.79, 0.052)),
        "target_edge": ((1.00, 0.056),),
        "target_clutter": ((1.00, -0.058), (1.00, 0.058)),
    }[layout]


def sample_generalization_scenario(
    *,
    requested_seed: int,
    schedule_index: int,
    block_initial_xy_m: Any,
    target_xy_m: Any,
    block_color: str,
    target_color: str,
    block_mass_kg: float,
) -> GeneralizationScenarioV1:
    """Return one exact realized scene for a deterministic schedule cell."""

    if type(requested_seed) is not int or requested_seed < 0:
        raise ValueError("requested_seed must be a non-negative integer")
    if type(schedule_index) is not int or schedule_index < 0:
        raise ValueError("schedule_index must be a non-negative integer")
    block = np.asarray(block_initial_xy_m, dtype=np.float64)
    target = np.asarray(target_xy_m, dtype=np.float64)
    if block.shape != (2,) or target.shape != (2,) or not np.isfinite(np.r_[block, target]).all():
        raise ValueError("block and target coordinates must be finite [2]")
    delta = target - block
    distance = float(np.linalg.norm(delta))
    if distance < 0.08:
        raise ValueError("block and target are too close for generalization layouts")
    direction = delta / distance
    normal = np.asarray([-direction[1], direction[0]], dtype=np.float64)
    rng = np.random.default_rng(requested_seed ^ (0x13A7E5 + schedule_index * 0x9E3779B1))

    friction_tier = FRICTION_TIERS[schedule_index % len(FRICTION_TIERS)]
    lighting_tier = _stratum(schedule_index, LIGHTING_TIERS, 1, schedule_index // 3)
    object_shape = _stratum(schedule_index, OBJECT_SHAPES, 2, schedule_index // 5)
    object_size_tier = _stratum(schedule_index, OBJECT_SIZE_TIERS, 1, 2 * (schedule_index // 7))
    object_material = _stratum(schedule_index, OBJECT_MATERIALS, 2, schedule_index // 11)
    obstacle_layout = OBSTACLE_LAYOUTS[schedule_index % len(OBSTACLE_LAYOUTS)]

    friction_bounds = {
        "low": (0.30, 0.52),
        "medium": (0.65, 0.90),
        "high": (1.05, 1.35),
    }[friction_tier]
    light_bounds = {
        "low": (0.42, 0.68),
        "medium": (0.82, 1.05),
        "high": (1.18, 1.48),
    }[lighting_tier]
    block_friction = float(rng.uniform(*friction_bounds))
    light_scale = float(rng.uniform(*light_bounds))
    light_ambient_scale = float(np.clip(light_scale * rng.uniform(0.82, 1.08), 0.35, 1.55))
    object_extents = _object_extents(object_shape, object_size_tier, rng)

    obstacles: list[ObstacleSpecV1] = []
    palette = (
        (0.72, 0.18, 0.12, 1.0),
        (0.16, 0.36, 0.74, 1.0),
        (0.52, 0.22, 0.62, 1.0),
    )
    for slot, (fraction, lateral) in enumerate(_layout_rows(obstacle_layout)):
        shape = OBSTACLE_SHAPES[(schedule_index + slot) % len(OBSTACLE_SHAPES)]
        nominal = float(rng.uniform(0.011, 0.017))
        if shape == "box":
            size_xy = (nominal, float(nominal * rng.uniform(0.85, 1.25)))
        elif shape == "cylinder":
            size_xy = (nominal, nominal)
        else:
            size_xy = (float(nominal * rng.uniform(1.05, 1.30)), float(nominal * rng.uniform(0.68, 0.88)))
        center = block + fraction * delta + lateral * normal
        obstacles.append(
            ObstacleSpecV1(
                slot=slot,
                shape=shape,
                center_xy_m=tuple(center),
                size_xy_m=size_xy,
                half_height_m=float(rng.uniform(0.025, 0.055)),
                yaw_rad=float(rng.uniform(-np.pi, np.pi)),
                friction=float(rng.uniform(0.45, 1.15)),
                rgba=palette[(schedule_index + slot) % len(palette)],
                along_path_fraction=float(fraction),
            )
        )

    return GeneralizationScenarioV1(
        requested_seed=requested_seed,
        schedule_index=schedule_index,
        friction_tier=friction_tier,
        block_friction=block_friction,
        lighting_tier=lighting_tier,
        light_scale=light_scale,
        light_ambient_scale=light_ambient_scale,
        object_shape=object_shape,
        object_size_tier=object_size_tier,
        object_half_extents_m=object_extents,
        block_mass_kg=float(block_mass_kg),
        object_material=object_material,
        obstacle_layout=obstacle_layout,
        obstacles=tuple(obstacles),
        block_color=str(block_color),
        target_color=str(target_color),
        block_initial_xy_m=tuple(block),
        target_xy_m=tuple(target),
    )


__all__ = [
    "FRICTION_TIERS",
    "GENERALIZATION_SCENE_SCHEMA_VERSION",
    "GENERALIZATION_SCHEDULE_VERSION",
    "GeneralizationScenarioV1",
    "LIGHTING_TIERS",
    "OBJECT_SHAPES",
    "OBJECT_SIZE_TIERS",
    "OBSTACLE_LAYOUTS",
    "OBSTACLE_SHAPES",
    "ObstacleSpecV1",
    "canonical_json",
    "contract_sha256",
    "sample_generalization_scenario",
]
