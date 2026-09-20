"""Physics-substep synthetic contact telemetry for EdgeArm V7.

The trace records MuJoCo solver state immediately after every physics substep
inside one control transition.  It is simulator-privileged supervision, not a
physical force/torque sensor stream.  Fixed-size finite arrays make transient
contacts and discrete impulse integrals auditable without falling back to the
last substep only.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from numbers import Integral
from typing import Callable, Sequence

import mujoco
import numpy as np


PHYSICS_SUBSTEP_CONTACT_FORMAT_V1 = "edgearm-physics-substep-contact-v1"
PUSH_SIDE_CONTACT_FORMAT_V1 = "edgearm-push-side-contact-semantics-v1"
PHYSICS_SUBSTEP_CONTACT_FORMAT_V2 = "edgearm-physics-substep-contact-v2"
PUSH_SIDE_CONTACT_FORMAT_V2 = "edgearm-push-side-contact-semantics-v2"
TOOL_CONTACT_IDENTITY_FORMAT = "edgearm-tool-contact-identity-v2"
TOOL_SAFETY_GEOM_ORDER_FORMAT = "edgearm-tool-safety-geom-order-v1"

# Existing HDF5 files retain their V1 format strings and bytes.  New in-memory
# traces explicitly advertise V2 because aggregate admissibility is now
# fail-closed across all jaw contacts.  The current HDF5 writer still selects
# only its reviewed legacy field list, so the new per-role arrays are not
# silently appended to an existing storage schema.
PHYSICS_SUBSTEP_CONTACT_FORMAT = PHYSICS_SUBSTEP_CONTACT_FORMAT_V2
PUSH_SIDE_CONTACT_FORMAT = PUSH_SIDE_CONTACT_FORMAT_V2
OBSTACLE_CONTACT_CLASSES = (
    "camera_housing",
    "pusher",
    "robot_other",
    "block",
)


def stable_tool_safety_geom_order_sha256(names: Sequence[str]) -> str:
    """Hash an ordered geometry-name identity without model-local IDs."""

    resolved = tuple(names)
    if (
        not resolved
        or len(set(resolved)) != len(resolved)
        or any(not isinstance(name, str) or not name.strip() for name in resolved)
    ):
        raise ValueError("tool safety geom names must be unique non-empty strings")
    encoded = json.dumps(
        list(resolved), separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class ContactTelemetryGeometry:
    """Resolved geom identifiers needed by the substep recorder.

    ``tool_geom`` is the non-colliding planning/orientation reference.  Actual
    tool contacts are the geometries in ``tool_contact_geoms``; keeping the two
    concepts separate lets an unmodified gripper use both jaw collision
    geometries without changing the frozen planning frame.  The frozenset is
    retained only for membership checks.  ``tool_contact_geom_order`` and
    ``tool_contact_geom_roles`` are the authoritative identity-preserving V2
    mapping used by metrics and recorders.
    """

    tool_geom: int
    tool_contact_geoms: frozenset[int]
    block_geom: int
    desk_geom: int
    obstacle_geom: int
    camera_housing_geom: int
    robot_geoms: frozenset[int]
    tool_contact_geom_order: tuple[int, ...] = ()
    tool_contact_geom_roles: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        scalar_ids = {
            "tool_geom": self.tool_geom,
            "block_geom": self.block_geom,
            "desk_geom": self.desk_geom,
            "obstacle_geom": self.obstacle_geom,
            "camera_housing_geom": self.camera_housing_geom,
        }
        for name, geom_id in scalar_ids.items():
            if isinstance(geom_id, bool) or not isinstance(geom_id, Integral):
                raise ValueError(f"{name} must be an integer geom identifier")
            if int(geom_id) < 0:
                raise ValueError(f"{name} must be non-negative")
        for name, geom_ids in (
            ("tool_contact_geoms", self.tool_contact_geoms),
            ("robot_geoms", self.robot_geoms),
        ):
            if not isinstance(geom_ids, frozenset):
                raise ValueError(f"{name} must be a frozenset")
            if any(
                isinstance(geom_id, bool)
                or not isinstance(geom_id, Integral)
                or int(geom_id) < 0
                for geom_id in geom_ids
            ):
                raise ValueError(f"{name} must contain only non-negative integer geom IDs")
        if not self.tool_contact_geoms:
            raise ValueError("tool_contact_geoms must be non-empty")
        if not self.tool_contact_geoms.issubset(self.robot_geoms):
            raise ValueError("tool_contact_geoms must be a subset of robot_geoms")
        non_tool_ids = {
            self.block_geom,
            self.desk_geom,
            self.obstacle_geom,
            self.camera_housing_geom,
        }
        if self.tool_contact_geoms.intersection(non_tool_ids):
            raise ValueError("tool_contact_geoms must not contain environment geom IDs")
        order = self.tool_contact_geom_order
        if not order:
            order = tuple(sorted(int(geom_id) for geom_id in self.tool_contact_geoms))
            object.__setattr__(self, "tool_contact_geom_order", order)
        if not isinstance(order, tuple):
            raise ValueError("tool_contact_geom_order must be a tuple")
        if (
            len(order) != len(self.tool_contact_geoms)
            or len(set(order)) != len(order)
            or frozenset(order) != self.tool_contact_geoms
        ):
            raise ValueError(
                "tool_contact_geom_order must contain every tool contact geom exactly once"
            )
        roles = self.tool_contact_geom_roles
        if not roles:
            roles = tuple(f"tool_tip_{index}" for index in range(len(order)))
            object.__setattr__(self, "tool_contact_geom_roles", roles)
        if not isinstance(roles, tuple):
            raise ValueError("tool_contact_geom_roles must be a tuple")
        if (
            len(roles) != len(order)
            or len(set(roles)) != len(roles)
            or any(not isinstance(role, str) or not role.strip() for role in roles)
        ):
            raise ValueError(
                "tool_contact_geom_roles must contain one unique non-empty role per geom"
            )

    @property
    def tool_contact_role_by_geom(self) -> dict[int, str]:
        """Return the explicit ordered geom-to-role identity contract."""

        return dict(zip(self.tool_contact_geom_order, self.tool_contact_geom_roles, strict=True))


def _validate_geometry_model_bounds(
    model: mujoco.MjModel,
    geometry: ContactTelemetryGeometry,
) -> None:
    """Fail clearly before an invalid geom ID reaches a MuJoCo array lookup."""

    geom_ids = {
        geometry.tool_geom,
        geometry.block_geom,
        geometry.desk_geom,
        geometry.obstacle_geom,
        geometry.camera_housing_geom,
        *geometry.tool_contact_geoms,
        *geometry.robot_geoms,
    }
    if any(int(geom_id) >= int(model.ngeom) for geom_id in geom_ids):
        raise ValueError("contact telemetry geom identifier is outside model.ngeom")


def _tool_contact_pair(
    first: int,
    second: int,
    *,
    tool_contact_geoms: frozenset[int],
    other_geom: int,
) -> bool:
    return bool(
        (first in tool_contact_geoms and second == other_geom)
        or (second in tool_contact_geoms and first == other_geom)
    )


@dataclass(frozen=True)
class PushSideContactThresholds:
    """Synthetic semantic gates that distinguish side pushing from top contact."""

    minimum_normal_force_n: float = 0.50
    minimum_normal_impulse_ns: float = 0.002
    minimum_contact_normal_horizontal_norm: float = 0.75
    minimum_contact_normal_push_alignment: float = 0.75
    minimum_tool_face_horizontal_norm: float = 0.75
    minimum_tool_face_push_alignment: float = 0.75
    block_top_bottom_exclusion_fraction: float = 0.20
    maximum_side_plane_distance_m: float = 0.003
    minimum_rear_support_ratio: float = 0.45
    maximum_edge_vertical_excess_m: float = 0.001
    minimum_edge_normal_horizontal_norm: float = 0.95
    minimum_edge_normal_push_alignment: float = 0.90
    minimum_edge_tool_face_horizontal_norm: float = 0.95
    minimum_edge_tool_face_push_alignment: float = 0.90
    minimum_edge_rear_support_ratio: float = 0.70

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
        unit_interval = (
            "minimum_contact_normal_horizontal_norm",
            "minimum_contact_normal_push_alignment",
            "minimum_tool_face_horizontal_norm",
            "minimum_tool_face_push_alignment",
            "block_top_bottom_exclusion_fraction",
            "minimum_rear_support_ratio",
            "minimum_edge_normal_horizontal_norm",
            "minimum_edge_normal_push_alignment",
            "minimum_edge_tool_face_horizontal_norm",
            "minimum_edge_tool_face_push_alignment",
            "minimum_edge_rear_support_ratio",
        )
        for name in unit_interval:
            if float(getattr(self, name)) > 1.0:
                raise ValueError(f"{name} must be at most one")
        if self.maximum_side_plane_distance_m <= 0.0:
            raise ValueError("maximum_side_plane_distance_m must be positive")

    @property
    def profile_hash(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _side_band_or_edge_valid(
    *,
    block_local_z_m: float,
    block_half_height_m: float,
    side_band_valid: bool,
    side_plane_distance_m: float,
    normal_horizontal_norm: float,
    normal_push_alignment: float,
    tool_face_horizontal_norm: float,
    tool_face_push_alignment: float,
    rear_support_ratio: float,
    thresholds: PushSideContactThresholds,
) -> tuple[bool, bool]:
    """Accept a box-edge manifold only with strong independent side evidence.

    MuJoCo may place the representative point of a valid side-face manifold on
    the upper or lower box edge.  The central side band remains the ordinary
    path; an excluded-band point is admitted only when it lies on the physical
    block extent and its normal, tool face, rear support and side plane all
    independently prove the intended push side.  A top contact with a vertical
    normal therefore cannot pass through this exception.
    """

    edge_valid = bool(
        not side_band_valid
        and abs(block_local_z_m)
        <= block_half_height_m + thresholds.maximum_edge_vertical_excess_m
        and side_plane_distance_m <= thresholds.maximum_side_plane_distance_m
        and normal_horizontal_norm
        >= thresholds.minimum_edge_normal_horizontal_norm
        and normal_push_alignment
        >= thresholds.minimum_edge_normal_push_alignment
        and tool_face_horizontal_norm
        >= thresholds.minimum_edge_tool_face_horizontal_norm
        and tool_face_push_alignment
        >= thresholds.minimum_edge_tool_face_push_alignment
        and rear_support_ratio >= thresholds.minimum_edge_rear_support_ratio
    )
    return bool(side_band_valid or edge_valid), edge_valid

def push_side_contact_metrics(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    geometry: ContactTelemetryGeometry,
    intended_push_direction_xy: np.ndarray,
    thresholds: PushSideContactThresholds | None = None,
) -> dict[str, object]:
    """Classify current tool/block contacts using force, face and side evidence.

    MuJoCo stores the contact normal in ``contact.frame[0:3]`` pointing from
    ``geom1`` to ``geom2``.  The normal is therefore canonicalized to point
    from the tool toward the block before applying the directed push gate.
    Aggregate admissibility is fail-closed: at least one force-valid side
    contact must exist and every simultaneous tool/block contact must satisfy
    the geometric side-contact gates.
    """

    _validate_geometry_model_bounds(model, geometry)
    limits = thresholds or PushSideContactThresholds()
    direction_xy = np.asarray(intended_push_direction_xy, dtype=np.float64)
    if direction_xy.shape != (2,) or not np.all(np.isfinite(direction_xy)):
        raise ValueError("intended_push_direction_xy must be a finite two-vector")
    direction_norm = float(np.linalg.norm(direction_xy))
    if direction_norm <= 1.0e-12:
        raise ValueError("intended_push_direction_xy must be non-zero")
    direction_xy = direction_xy / direction_norm
    direction_world = np.array([direction_xy[0], direction_xy[1], 0.0], dtype=np.float64)

    tool_rotation = np.asarray(data.geom_xmat[geometry.tool_geom], dtype=np.float64).reshape(3, 3)
    tool_face_axis = tool_rotation[:, 1]
    tool_face_horizontal_norm = float(np.linalg.norm(tool_face_axis[:2]))
    tool_face_push_alignment = float(
        abs(np.dot(tool_face_axis[:2] / max(tool_face_horizontal_norm, 1.0e-12), direction_xy))
    )
    block_rotation = np.asarray(data.geom_xmat[geometry.block_geom], dtype=np.float64).reshape(3, 3)
    block_center = np.asarray(data.geom_xpos[geometry.block_geom], dtype=np.float64)
    block_half = np.asarray(model.geom_size[geometry.block_geom], dtype=np.float64)
    block_support_along_push = float(
        np.dot(np.abs(block_rotation.T @ direction_world), block_half)
    )

    raw_contact_count = 0
    force_bearing_contact_count = 0
    geometric_side_contact_count = 0
    invalid_tool_block_contact_count = 0
    valid_side_contact_count = 0
    valid_side_total_normal_force_n = 0.0
    role_by_geom = geometry.tool_contact_role_by_geom
    tool_block_contact_by_role: dict[str, dict[str, object]] = {
        role: {
            "geom_id": int(geom_id),
            "raw_contact_count": 0,
            "force_bearing_contact_count": 0,
            "total_normal_force_n": 0.0,
            "geometric_side_contact_count": 0,
            "invalid_tool_block_contact_count": 0,
            "valid_side_contact_count": 0,
            "valid_side_total_normal_force_n": 0.0,
        }
        for geom_id, role in zip(
            geometry.tool_contact_geom_order,
            geometry.tool_contact_geom_roles,
            strict=True,
        )
    }
    strongest_force = -np.inf
    representative: dict[str, object] = {
        "representative_valid": False,
        "representative_normal_tool_to_block_world": np.zeros(3, dtype=np.float64),
        "representative_contact_normal_horizontal_norm": 0.0,
        "representative_contact_normal_push_alignment": 0.0,
        "representative_block_local_contact_xyz_m": np.zeros(3, dtype=np.float64),
        "representative_side_band_valid": False,
        "representative_edge_side_contact_valid": False,
        "representative_side_plane_distance_m": 0.0,
        "representative_rear_support_ratio": 0.0,
        "representative_geometric_side_contact_valid": False,
        "representative_normal_force_n": 0.0,
        "representative_geom1_is_tool": False,
        "representative_tool_contact_geom_id": -1,
        "representative_tool_contact_role": "",
    }
    for contact_index in range(data.ncon):
        contact = data.contact[contact_index]
        first = int(contact.geom1)
        second = int(contact.geom2)
        if not _tool_contact_pair(
            first,
            second,
            tool_contact_geoms=geometry.tool_contact_geoms,
            other_geom=geometry.block_geom,
        ):
            continue
        tool_contact_geom = first if first in geometry.tool_contact_geoms else second
        tool_contact_role = role_by_geom[tool_contact_geom]
        role_metrics = tool_block_contact_by_role[tool_contact_role]
        raw_contact_count += 1
        role_metrics["raw_contact_count"] = int(role_metrics["raw_contact_count"]) + 1
        wrench = np.zeros(6, dtype=np.float64)
        mujoco.mj_contactForce(model, data, contact_index, wrench)
        normal_force = max(float(wrench[0]), 0.0)
        force_bearing_contact_count += int(normal_force > 0.0)
        role_metrics["force_bearing_contact_count"] = int(
            role_metrics["force_bearing_contact_count"]
        ) + int(normal_force > 0.0)
        role_metrics["total_normal_force_n"] = (
            float(role_metrics["total_normal_force_n"]) + normal_force
        )
        normal = np.asarray(contact.frame[:3], dtype=np.float64)
        normal_tool_to_block = normal if first in geometry.tool_contact_geoms else -normal
        normal_horizontal_norm = float(np.linalg.norm(normal_tool_to_block[:2]))
        normal_push_alignment = float(
            np.dot(
                normal_tool_to_block[:2] / max(normal_horizontal_norm, 1.0e-12),
                direction_xy,
            )
        )
        contact_point = np.asarray(contact.pos, dtype=np.float64)
        block_local_point = block_rotation.T @ (contact_point - block_center)
        side_band_limit = float(
            block_half[2] * (1.0 - limits.block_top_bottom_exclusion_fraction)
        )
        side_band_valid = bool(abs(block_local_point[2]) <= side_band_limit)
        side_plane_distance = float(
            min(
                abs(block_half[0] - abs(block_local_point[0])),
                abs(block_half[1] - abs(block_local_point[1])),
            )
        )
        rear_projection = float(np.dot(block_center[:2] - contact_point[:2], direction_xy))
        rear_support_ratio = rear_projection / max(block_support_along_push, 1.0e-12)
        vertical_side_valid, edge_side_contact_valid = _side_band_or_edge_valid(
            block_local_z_m=float(block_local_point[2]),
            block_half_height_m=float(block_half[2]),
            side_band_valid=side_band_valid,
            side_plane_distance_m=side_plane_distance,
            normal_horizontal_norm=normal_horizontal_norm,
            normal_push_alignment=normal_push_alignment,
            tool_face_horizontal_norm=tool_face_horizontal_norm,
            tool_face_push_alignment=tool_face_push_alignment,
            rear_support_ratio=rear_support_ratio,
            thresholds=limits,
        )
        geometric_side_valid = bool(
            normal_horizontal_norm >= limits.minimum_contact_normal_horizontal_norm
            and normal_push_alignment >= limits.minimum_contact_normal_push_alignment
            and tool_face_horizontal_norm >= limits.minimum_tool_face_horizontal_norm
            and tool_face_push_alignment >= limits.minimum_tool_face_push_alignment
            and vertical_side_valid
            and side_plane_distance <= limits.maximum_side_plane_distance_m
            and rear_support_ratio >= limits.minimum_rear_support_ratio
        )
        geometric_side_contact_count += int(geometric_side_valid)
        invalid_tool_block_contact_count += int(not geometric_side_valid)
        role_metrics["geometric_side_contact_count"] = int(
            role_metrics["geometric_side_contact_count"]
        ) + int(geometric_side_valid)
        role_metrics["invalid_tool_block_contact_count"] = int(
            role_metrics["invalid_tool_block_contact_count"]
        ) + int(not geometric_side_valid)
        force_valid = bool(normal_force >= limits.minimum_normal_force_n)
        valid = bool(geometric_side_valid and force_valid)
        valid_side_contact_count += int(valid)
        role_metrics["valid_side_contact_count"] = int(
            role_metrics["valid_side_contact_count"]
        ) + int(valid)
        if valid:
            valid_side_total_normal_force_n += normal_force
            role_metrics["valid_side_total_normal_force_n"] = (
                float(role_metrics["valid_side_total_normal_force_n"]) + normal_force
            )
        if normal_force > strongest_force:
            strongest_force = normal_force
            representative = {
                "representative_valid": True,
                "representative_normal_tool_to_block_world": normal_tool_to_block.copy(),
                "representative_contact_normal_horizontal_norm": normal_horizontal_norm,
                "representative_contact_normal_push_alignment": normal_push_alignment,
                "representative_block_local_contact_xyz_m": block_local_point.copy(),
                "representative_side_band_valid": side_band_valid,
                "representative_edge_side_contact_valid": edge_side_contact_valid,
                "representative_side_plane_distance_m": side_plane_distance,
                "representative_rear_support_ratio": rear_support_ratio,
                "representative_geometric_side_contact_valid": geometric_side_valid,
                "representative_normal_force_n": normal_force,
                "representative_geom1_is_tool": bool(
                    first in geometry.tool_contact_geoms
                ),
                "representative_tool_contact_geom_id": tool_contact_geom,
                "representative_tool_contact_role": tool_contact_role,
            }
    for role_metrics in tool_block_contact_by_role.values():
        invalid_count = int(role_metrics["invalid_tool_block_contact_count"])
        valid_count = int(role_metrics["valid_side_contact_count"])
        role_metrics["all_tool_block_contacts_geometrically_valid"] = bool(
            invalid_count == 0
        )
        role_metrics["valid_side_contact_observed_any"] = bool(valid_count > 0)
        role_metrics["tool_block_contact_admissible"] = bool(
            valid_count > 0 and invalid_count == 0
        )
    valid_side_contact_observed_any = bool(valid_side_contact_count > 0)
    all_tool_block_contacts_geometrically_valid = bool(
        invalid_tool_block_contact_count == 0
    )
    tool_block_contact_admissible = bool(
        valid_side_contact_observed_any
        and all_tool_block_contacts_geometrically_valid
    )
    return {
        "format": PUSH_SIDE_CONTACT_FORMAT,
        "contact_identity_format": TOOL_CONTACT_IDENTITY_FORMAT,
        "simulator_privileged_truth": True,
        "physically_calibrated": False,
        "physical_samples": 0,
        "threshold_profile_hash": limits.profile_hash,
        "thresholds": asdict(limits),
        "intended_push_direction_xy": direction_xy.copy(),
        "tool_face_axis_world": tool_face_axis.copy(),
        "tool_face_horizontal_norm": tool_face_horizontal_norm,
        "tool_face_push_alignment": tool_face_push_alignment,
        "block_support_along_push_m": block_support_along_push,
        "raw_contact_count": raw_contact_count,
        "force_bearing_contact_count": force_bearing_contact_count,
        "geometric_side_contact_count": geometric_side_contact_count,
        "invalid_tool_block_contact_count": invalid_tool_block_contact_count,
        "all_tool_block_contacts_geometrically_valid": (
            all_tool_block_contacts_geometrically_valid
        ),
        "valid_side_contact_count": valid_side_contact_count,
        "valid_side_total_normal_force_n": valid_side_total_normal_force_n,
        "valid_side_contact_observed_any": valid_side_contact_observed_any,
        "valid_side_contact_any": tool_block_contact_admissible,
        "tool_block_contact_admissible": tool_block_contact_admissible,
        "tool_contact_role_names": geometry.tool_contact_geom_roles,
        "tool_contact_geom_ids": geometry.tool_contact_geom_order,
        "tool_block_contact_by_role": tool_block_contact_by_role,
        **representative,
    }


class PhysicsSubstepContactRecorder:
    """Accumulate one fixed-horizon control-transition contact trace."""

    def __init__(
        self,
        model: mujoco.MjModel,
        geometry: ContactTelemetryGeometry,
        *,
        physics_substeps: int,
        decision_time_seconds: float,
        obstacle_enabled: bool,
        penetration_tolerance_m: float,
        tool_desk_signed_distance: Callable[[mujoco.MjData], float],
        initial_block_xy: np.ndarray,
        initial_tool_xyz: np.ndarray,
        intended_push_direction_xy: np.ndarray,
        tool_safety_desk_signed_distances: Callable[[mujoco.MjData], np.ndarray] | None = None,
        tool_safety_block_signed_distances: Callable[[mujoco.MjData], np.ndarray] | None = None,
        tool_safety_geom_ids: Sequence[int] = (),
        side_contact_thresholds: PushSideContactThresholds | None = None,
    ) -> None:
        if physics_substeps <= 0:
            raise ValueError("physics_substeps must be positive")
        _validate_geometry_model_bounds(model, geometry)
        self.model = model
        self.geometry = geometry
        self.physics_substeps = int(physics_substeps)
        self.decision_time_seconds = float(decision_time_seconds)
        self.obstacle_enabled = bool(obstacle_enabled)
        self.penetration_tolerance_m = float(penetration_tolerance_m)
        self._tool_desk_signed_distance = tool_desk_signed_distance
        if (tool_safety_desk_signed_distances is None) != (
            tool_safety_block_signed_distances is None
        ):
            raise ValueError("full safety desk/block distance callbacks must be provided together")
        self._tool_safety_desk_signed_distances = tool_safety_desk_signed_distances
        self._tool_safety_block_signed_distances = tool_safety_block_signed_distances
        resolved_safety_ids = tuple(tool_safety_geom_ids)
        if tool_safety_desk_signed_distances is None:
            if resolved_safety_ids:
                raise ValueError(
                    "tool safety geom IDs require full safety distance callbacks"
                )
            self.tool_safety_geom_ids = np.zeros(0, dtype=np.int32)
            self.tool_safety_geom_names: tuple[str, ...] = ()
            self.tool_safety_geom_order_sha256 = ""
        else:
            if (
                not resolved_safety_ids
                or len(set(resolved_safety_ids)) != len(resolved_safety_ids)
                or any(
                    isinstance(geom_id, bool)
                    or not isinstance(geom_id, Integral)
                    or not 0 <= int(geom_id) < int(model.ngeom)
                    for geom_id in resolved_safety_ids
                )
            ):
                raise ValueError(
                    "full safety distances require unique in-bounds geometry IDs"
                )
            names = tuple(
                mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(geom_id))
                for geom_id in resolved_safety_ids
            )
            if any(name is None for name in names):
                raise ValueError("full safety distance geometry IDs must have names")
            self.tool_safety_geom_ids = np.asarray(
                resolved_safety_ids, dtype=np.int32
            )
            self.tool_safety_geom_names = tuple(str(name) for name in names)
            self.tool_safety_geom_order_sha256 = (
                stable_tool_safety_geom_order_sha256(self.tool_safety_geom_names)
            )
        self._initial_block_xy = np.asarray(initial_block_xy, dtype=np.float64).copy()
        self._initial_tool_xyz = np.asarray(initial_tool_xyz, dtype=np.float64).copy()
        self._intended_push_direction_xy = np.asarray(
            intended_push_direction_xy, dtype=np.float64
        ).copy()
        self.side_contact_thresholds = side_contact_thresholds or PushSideContactThresholds()
        if self._initial_block_xy.shape != (2,) or self._initial_tool_xyz.shape != (3,):
            raise ValueError("initial block/tool positions have invalid shape")
        if (
            self._intended_push_direction_xy.shape != (2,)
            or not np.all(np.isfinite(self._intended_push_direction_xy))
            or np.linalg.norm(self._intended_push_direction_xy) <= 1.0e-12
        ):
            raise ValueError("intended push direction must be a finite non-zero two-vector")
        count = self.physics_substeps
        role_count = len(geometry.tool_contact_geom_roles)
        self.tool_contact_role_names = geometry.tool_contact_geom_roles
        self.tool_contact_geom_ids = np.asarray(
            geometry.tool_contact_geom_order, dtype=np.int32
        )
        self.sample_time_seconds = np.zeros(count, dtype=np.float64)
        self.tool_block_contact_count = np.zeros(count, dtype=np.int16)
        self.tool_block_total_normal_force_n = np.zeros(count, dtype=np.float64)
        self.tool_block_contact_count_by_role = np.zeros(
            (count, role_count), dtype=np.int16
        )
        self.tool_block_total_normal_force_n_by_role = np.zeros(
            (count, role_count), dtype=np.float64
        )
        self.tool_block_min_contact_distance_m = np.zeros(count, dtype=np.float64)
        self.tool_block_min_contact_distance_valid = np.zeros(count, dtype=np.uint8)
        self.representative_valid = np.zeros(count, dtype=np.uint8)
        self.representative_point_xyz_m = np.zeros((count, 3), dtype=np.float64)
        self.representative_frame_world = np.zeros((count, 9), dtype=np.float64)
        self.representative_wrench_contact_frame = np.zeros((count, 6), dtype=np.float64)
        self.representative_geom1_is_tool = np.zeros(count, dtype=np.uint8)
        self.representative_tool_contact_role_index = np.full(
            count, -1, dtype=np.int16
        )
        self.valid_push_side_contact_count = np.zeros(count, dtype=np.int16)
        self.valid_push_side_total_normal_force_n = np.zeros(count, dtype=np.float64)
        self.geometric_push_side_contact_count = np.zeros(count, dtype=np.int16)
        self.valid_push_side_contact_count_by_role = np.zeros(
            (count, role_count), dtype=np.int16
        )
        self.valid_push_side_total_normal_force_n_by_role = np.zeros(
            (count, role_count), dtype=np.float64
        )
        self.geometric_push_side_contact_count_by_role = np.zeros(
            (count, role_count), dtype=np.int16
        )
        self.invalid_tool_block_contact_count = np.zeros(count, dtype=np.int16)
        self.invalid_tool_block_contact_count_by_role = np.zeros(
            (count, role_count), dtype=np.int16
        )
        self.all_tool_block_contacts_geometrically_valid = np.ones(
            count, dtype=np.uint8
        )
        self.all_tool_block_contacts_geometrically_valid_by_role = np.ones(
            (count, role_count), dtype=np.uint8
        )
        self.tool_block_contact_admissible = np.zeros(count, dtype=np.uint8)
        self.tool_block_contact_admissible_by_role = np.zeros(
            (count, role_count), dtype=np.uint8
        )
        self.representative_normal_tool_to_block_world = np.zeros((count, 3), dtype=np.float64)
        self.representative_contact_normal_horizontal_norm = np.zeros(count, dtype=np.float64)
        self.representative_contact_normal_push_alignment = np.zeros(count, dtype=np.float64)
        self.representative_block_local_contact_xyz_m = np.zeros((count, 3), dtype=np.float64)
        self.representative_side_band_valid = np.zeros(count, dtype=np.uint8)
        self.representative_edge_side_contact_valid = np.zeros(count, dtype=np.uint8)
        self.representative_side_plane_distance_m = np.zeros(count, dtype=np.float64)
        self.representative_rear_support_ratio = np.zeros(count, dtype=np.float64)
        self.representative_geometric_side_contact_valid = np.zeros(count, dtype=np.uint8)
        self.tool_face_axis_world = np.zeros((count, 3), dtype=np.float64)
        self.tool_face_horizontal_norm = np.zeros(count, dtype=np.float64)
        self.tool_face_push_alignment = np.zeros(count, dtype=np.float64)
        self.tool_desk_signed_distance_m = np.zeros(count, dtype=np.float64)
        self.tool_desk_contact_count = np.zeros(count, dtype=np.int16)
        self.tool_desk_min_contact_distance_m = np.zeros(count, dtype=np.float64)
        self.tool_desk_min_contact_distance_valid = np.zeros(count, dtype=np.uint8)
        self.non_tool_robot_desk_contact_count = np.zeros(count, dtype=np.int16)
        self.non_tool_robot_desk_min_distance_m = np.zeros(count, dtype=np.float64)
        self.non_tool_robot_desk_min_distance_valid = np.zeros(count, dtype=np.uint8)
        self.obstacle_contact_counts = np.zeros((count, len(OBSTACLE_CONTACT_CLASSES)), dtype=np.int16)
        # Optional raw full-union distances are an in-memory successor field.
        # The reviewed Causal V2 HDF5 writer keeps selecting its legacy field
        # list, so enabling this cannot silently mutate the V2 storage schema.
        self.tool_safety_desk_signed_distance_m: np.ndarray | None = None
        self.tool_safety_block_signed_distance_m: np.ndarray | None = None
        self._recorded = np.zeros(count, dtype=np.uint8)

    def record(self, data: mujoco.MjData, substep_index: int) -> None:
        """Sample contacts immediately after one ``mujoco.mj_step`` call."""

        if not 0 <= substep_index < self.physics_substeps:
            raise IndexError("substep_index is outside the configured trace")
        if self._recorded[substep_index]:
            raise RuntimeError("physics substep was recorded more than once")
        geometry = self.geometry
        self.sample_time_seconds[substep_index] = float(data.time)
        self.tool_desk_signed_distance_m[substep_index] = float(self._tool_desk_signed_distance(data))
        if self._tool_safety_desk_signed_distances is not None:
            desk = np.asarray(
                self._tool_safety_desk_signed_distances(data), dtype=np.float64
            )
            block_callback = self._tool_safety_block_signed_distances
            if block_callback is None:  # pragma: no cover - constructor proves pairing
                raise RuntimeError("full safety block-distance callback disappeared")
            block = np.asarray(block_callback(data), dtype=np.float64)
            if (
                desk.ndim != 1
                or desk.size < 1
                or block.shape != desk.shape
                or not np.all(np.isfinite(desk))
                or not np.all(np.isfinite(block))
            ):
                raise RuntimeError("full safety desk/block distance sample is malformed")
            if self.tool_safety_desk_signed_distance_m is None:
                shape = (self.physics_substeps, int(desk.size))
                self.tool_safety_desk_signed_distance_m = np.empty(shape, dtype=np.float64)
                self.tool_safety_block_signed_distance_m = np.empty(shape, dtype=np.float64)
            if self.tool_safety_desk_signed_distance_m.shape[1:] != desk.shape:
                raise RuntimeError("full safety geometry count changed within a transition")
            if self.tool_safety_block_signed_distance_m is None:  # pragma: no cover
                raise RuntimeError("full safety block-distance storage disappeared")
            self.tool_safety_desk_signed_distance_m[substep_index] = desk
            self.tool_safety_block_signed_distance_m[substep_index] = block
        best_tool_block_normal_force = -np.inf
        tool_block_distances: list[float] = []
        tool_desk_distances: list[float] = []
        non_tool_desk_distances: list[float] = []
        for contact_index in range(data.ncon):
            contact = data.contact[contact_index]
            first = int(contact.geom1)
            second = int(contact.geom2)
            pair = {first, second}
            distance = float(contact.dist)
            if _tool_contact_pair(
                first,
                second,
                tool_contact_geoms=geometry.tool_contact_geoms,
                other_geom=geometry.block_geom,
            ):
                wrench = np.zeros(6, dtype=np.float64)
                mujoco.mj_contactForce(self.model, data, contact_index, wrench)
                normal_force = max(float(wrench[0]), 0.0)
                self.tool_block_contact_count[substep_index] += 1
                self.tool_block_total_normal_force_n[substep_index] += normal_force
                tool_block_distances.append(distance)
                if normal_force > best_tool_block_normal_force:
                    best_tool_block_normal_force = normal_force
                    self.representative_valid[substep_index] = 1
                    self.representative_point_xyz_m[substep_index] = np.asarray(contact.pos, dtype=np.float64)
                    self.representative_frame_world[substep_index] = np.asarray(
                        contact.frame, dtype=np.float64
                    )
                    self.representative_wrench_contact_frame[substep_index] = wrench
                    self.representative_geom1_is_tool[substep_index] = int(
                        first in geometry.tool_contact_geoms
                    )
            if _tool_contact_pair(
                first,
                second,
                tool_contact_geoms=geometry.tool_contact_geoms,
                other_geom=geometry.desk_geom,
            ):
                self.tool_desk_contact_count[substep_index] += 1
                tool_desk_distances.append(distance)
            if geometry.desk_geom in pair:
                other = second if first == geometry.desk_geom else first
                if (
                    other in geometry.robot_geoms
                    and other not in geometry.tool_contact_geoms
                ):
                    self.non_tool_robot_desk_contact_count[substep_index] += 1
                    non_tool_desk_distances.append(distance)
            if self.obstacle_enabled and geometry.obstacle_geom in pair:
                other = second if first == geometry.obstacle_geom else first
                if other == geometry.camera_housing_geom:
                    obstacle_class = "camera_housing"
                elif other in geometry.tool_contact_geoms:
                    obstacle_class = "pusher"
                elif other == geometry.block_geom:
                    obstacle_class = "block"
                elif other in geometry.robot_geoms:
                    obstacle_class = "robot_other"
                else:
                    obstacle_class = ""
                if obstacle_class:
                    class_index = OBSTACLE_CONTACT_CLASSES.index(obstacle_class)
                    self.obstacle_contact_counts[substep_index, class_index] += 1
        side = push_side_contact_metrics(
            self.model,
            data,
            geometry,
            self._intended_push_direction_xy,
            self.side_contact_thresholds,
        )
        self.valid_push_side_contact_count[substep_index] = int(side["valid_side_contact_count"])
        self.valid_push_side_total_normal_force_n[substep_index] = float(
            side["valid_side_total_normal_force_n"]
        )
        self.geometric_push_side_contact_count[substep_index] = int(
            side["geometric_side_contact_count"]
        )
        self.invalid_tool_block_contact_count[substep_index] = int(
            side["invalid_tool_block_contact_count"]
        )
        self.all_tool_block_contacts_geometrically_valid[substep_index] = int(
            bool(side["all_tool_block_contacts_geometrically_valid"])
        )
        self.tool_block_contact_admissible[substep_index] = int(
            bool(side["tool_block_contact_admissible"])
        )
        role_metrics = side["tool_block_contact_by_role"]
        if not isinstance(role_metrics, dict):
            raise RuntimeError("contact role metrics must be a dictionary")
        for role_index, role in enumerate(self.tool_contact_role_names):
            current = role_metrics.get(role)
            if not isinstance(current, dict):
                raise RuntimeError(f"contact role metrics are missing {role}")
            self.tool_block_contact_count_by_role[substep_index, role_index] = int(
                current["raw_contact_count"]
            )
            self.tool_block_total_normal_force_n_by_role[
                substep_index, role_index
            ] = float(current["total_normal_force_n"])
            self.valid_push_side_contact_count_by_role[
                substep_index, role_index
            ] = int(current["valid_side_contact_count"])
            self.valid_push_side_total_normal_force_n_by_role[
                substep_index, role_index
            ] = float(current["valid_side_total_normal_force_n"])
            self.geometric_push_side_contact_count_by_role[
                substep_index, role_index
            ] = int(current["geometric_side_contact_count"])
            self.invalid_tool_block_contact_count_by_role[
                substep_index, role_index
            ] = int(current["invalid_tool_block_contact_count"])
            self.all_tool_block_contacts_geometrically_valid_by_role[
                substep_index, role_index
            ] = int(bool(current["all_tool_block_contacts_geometrically_valid"]))
            self.tool_block_contact_admissible_by_role[
                substep_index, role_index
            ] = int(bool(current["tool_block_contact_admissible"]))
        representative_role = str(side["representative_tool_contact_role"])
        if representative_role:
            self.representative_tool_contact_role_index[substep_index] = (
                self.tool_contact_role_names.index(representative_role)
            )
        self.representative_normal_tool_to_block_world[substep_index] = np.asarray(
            side["representative_normal_tool_to_block_world"], dtype=np.float64
        )
        self.representative_contact_normal_horizontal_norm[substep_index] = float(
            side["representative_contact_normal_horizontal_norm"]
        )
        self.representative_contact_normal_push_alignment[substep_index] = float(
            side["representative_contact_normal_push_alignment"]
        )
        self.representative_block_local_contact_xyz_m[substep_index] = np.asarray(
            side["representative_block_local_contact_xyz_m"], dtype=np.float64
        )
        self.representative_side_band_valid[substep_index] = int(
            bool(side["representative_side_band_valid"])
        )
        self.representative_edge_side_contact_valid[substep_index] = int(
            bool(side["representative_edge_side_contact_valid"])
        )
        self.representative_side_plane_distance_m[substep_index] = float(
            side["representative_side_plane_distance_m"]
        )
        self.representative_rear_support_ratio[substep_index] = float(
            side["representative_rear_support_ratio"]
        )
        self.representative_geometric_side_contact_valid[substep_index] = int(
            bool(side["representative_geometric_side_contact_valid"])
        )
        self.tool_face_axis_world[substep_index] = np.asarray(side["tool_face_axis_world"])
        self.tool_face_horizontal_norm[substep_index] = float(side["tool_face_horizontal_norm"])
        self.tool_face_push_alignment[substep_index] = float(side["tool_face_push_alignment"])
        self._store_optional_minimum(
            tool_block_distances,
            self.tool_block_min_contact_distance_m,
            self.tool_block_min_contact_distance_valid,
            substep_index,
        )
        self._store_optional_minimum(
            tool_desk_distances,
            self.tool_desk_min_contact_distance_m,
            self.tool_desk_min_contact_distance_valid,
            substep_index,
        )
        self._store_optional_minimum(
            non_tool_desk_distances,
            self.non_tool_robot_desk_min_distance_m,
            self.non_tool_robot_desk_min_distance_valid,
            substep_index,
        )
        self._recorded[substep_index] = 1

    def finalize(
        self,
        data: mujoco.MjData,
        *,
        final_block_xy: np.ndarray,
        final_tool_xyz: np.ndarray,
    ) -> dict[str, object]:
        """Return an internally recomputable transition trace and summary."""

        if not np.all(self._recorded):
            raise RuntimeError("contact trace is missing one or more physics substeps")
        if not np.all(np.diff(self.sample_time_seconds) > 0.0):
            raise RuntimeError("physics substep sample times must be strictly increasing")
        substep_dt = float(self.model.opt.timestep)
        contact_mask = self.tool_block_contact_count > 0
        force_mask = self.tool_block_total_normal_force_n > 0.0
        contact_indices = np.flatnonzero(contact_mask)
        side_contact_mask = self.valid_push_side_contact_count > 0
        side_contact_indices = np.flatnonzero(side_contact_mask)
        side_normal_impulse = float(
            np.sum(self.valid_push_side_total_normal_force_n) * substep_dt
        )
        valid_push_side_contact_observed_any = bool(
            np.any(side_contact_mask)
            and side_normal_impulse >= self.side_contact_thresholds.minimum_normal_impulse_ns
        )
        invalid_tool_block_contact_any = bool(
            np.any(self.invalid_tool_block_contact_count > 0)
        )
        valid_push_side_contact_any = bool(
            valid_push_side_contact_observed_any
            and not invalid_tool_block_contact_any
        )
        final_block_xy = np.asarray(final_block_xy, dtype=np.float64)
        final_tool_xyz = np.asarray(final_tool_xyz, dtype=np.float64)
        minimum_tool_desk_distance = float(np.min(self.tool_desk_signed_distance_m))
        non_tool_negative_contact = bool(
            np.any(
                (self.non_tool_robot_desk_min_distance_valid > 0)
                & (self.non_tool_robot_desk_min_distance_m < -self.penetration_tolerance_m)
            )
        )
        full_safety_sampled = self.tool_safety_desk_signed_distance_m is not None
        if full_safety_sampled:
            if self.tool_safety_block_signed_distance_m is None:  # pragma: no cover
                raise RuntimeError("full safety block-distance trace is missing")
            safety_count = int(self.tool_safety_desk_signed_distance_m.shape[1])
            if safety_count != int(self.tool_safety_geom_ids.size):
                raise RuntimeError(
                    "full safety distance columns differ from ordered geometry identity"
                )
            safety_desk = self.tool_safety_desk_signed_distance_m.copy()
            safety_block = self.tool_safety_block_signed_distance_m.copy()
        else:
            safety_count = 0
            safety_desk = np.zeros((self.physics_substeps, 0), dtype=np.float64)
            safety_block = np.zeros((self.physics_substeps, 0), dtype=np.float64)
        return {
            "format": PHYSICS_SUBSTEP_CONTACT_FORMAT,
            "contact_identity_format": TOOL_CONTACT_IDENTITY_FORMAT,
            "sample_origin": "mujoco_solver_state_immediately_after_each_mj_step",
            "claim_level": "L1_SYNTHETIC_HARDWARE_INSPIRED",
            "simulator_privileged_truth": True,
            "physical_force_sensor_data": False,
            "physical_samples": 0,
            "physics_substeps": self.physics_substeps,
            "physics_substep_dt_seconds": substep_dt,
            "decision_time_seconds": self.decision_time_seconds,
            "effect_time_seconds": float(data.time),
            "sample_time_seconds": self.sample_time_seconds.copy(),
            "tool_block_contact_count": self.tool_block_contact_count.copy(),
            "tool_block_total_normal_force_n": self.tool_block_total_normal_force_n.copy(),
            "tool_contact_role_names": self.tool_contact_role_names,
            "tool_contact_geom_ids": self.tool_contact_geom_ids.copy(),
            "tool_safety_distance_sampled_each_substep": full_safety_sampled,
            "tool_safety_geom_count": safety_count,
            "tool_safety_geom_order_format": (
                TOOL_SAFETY_GEOM_ORDER_FORMAT if full_safety_sampled else ""
            ),
            "tool_safety_geom_ids": self.tool_safety_geom_ids.copy(),
            "tool_safety_geom_names": self.tool_safety_geom_names,
            "tool_safety_geom_order_sha256": self.tool_safety_geom_order_sha256,
            "tool_safety_desk_signed_distance_m": safety_desk,
            "tool_safety_block_signed_distance_m": safety_block,
            "tool_block_contact_count_by_role": (
                self.tool_block_contact_count_by_role.copy()
            ),
            "tool_block_total_normal_force_n_by_role": (
                self.tool_block_total_normal_force_n_by_role.copy()
            ),
            "tool_block_min_contact_distance_m": (self.tool_block_min_contact_distance_m.copy()),
            "tool_block_min_contact_distance_valid": (self.tool_block_min_contact_distance_valid.copy()),
            "representative_valid": self.representative_valid.copy(),
            "representative_point_xyz_m": self.representative_point_xyz_m.copy(),
            "representative_frame_world": self.representative_frame_world.copy(),
            "representative_wrench_contact_frame": (self.representative_wrench_contact_frame.copy()),
            "representative_geom1_is_tool": self.representative_geom1_is_tool.copy(),
            "representative_tool_contact_role_index": (
                self.representative_tool_contact_role_index.copy()
            ),
            "push_side_contact_format": PUSH_SIDE_CONTACT_FORMAT,
            "push_side_contact_threshold_profile_hash": self.side_contact_thresholds.profile_hash,
            "push_side_contact_thresholds": asdict(self.side_contact_thresholds),
            "intended_push_direction_xy": self._intended_push_direction_xy.copy(),
            "valid_push_side_contact_count": self.valid_push_side_contact_count.copy(),
            "valid_push_side_total_normal_force_n": (
                self.valid_push_side_total_normal_force_n.copy()
            ),
            "geometric_push_side_contact_count": self.geometric_push_side_contact_count.copy(),
            "valid_push_side_contact_count_by_role": (
                self.valid_push_side_contact_count_by_role.copy()
            ),
            "valid_push_side_total_normal_force_n_by_role": (
                self.valid_push_side_total_normal_force_n_by_role.copy()
            ),
            "geometric_push_side_contact_count_by_role": (
                self.geometric_push_side_contact_count_by_role.copy()
            ),
            "invalid_tool_block_contact_count": (
                self.invalid_tool_block_contact_count.copy()
            ),
            "invalid_tool_block_contact_count_by_role": (
                self.invalid_tool_block_contact_count_by_role.copy()
            ),
            "all_tool_block_contacts_geometrically_valid": (
                self.all_tool_block_contacts_geometrically_valid.copy()
            ),
            "all_tool_block_contacts_geometrically_valid_by_role": (
                self.all_tool_block_contacts_geometrically_valid_by_role.copy()
            ),
            "tool_block_contact_admissible": self.tool_block_contact_admissible.copy(),
            "tool_block_contact_admissible_by_role": (
                self.tool_block_contact_admissible_by_role.copy()
            ),
            "representative_normal_tool_to_block_world": (
                self.representative_normal_tool_to_block_world.copy()
            ),
            "representative_contact_normal_horizontal_norm": (
                self.representative_contact_normal_horizontal_norm.copy()
            ),
            "representative_contact_normal_push_alignment": (
                self.representative_contact_normal_push_alignment.copy()
            ),
            "representative_block_local_contact_xyz_m": (
                self.representative_block_local_contact_xyz_m.copy()
            ),
            "representative_side_band_valid": self.representative_side_band_valid.copy(),
            "representative_edge_side_contact_valid": (
                self.representative_edge_side_contact_valid.copy()
            ),
            "representative_side_plane_distance_m": (
                self.representative_side_plane_distance_m.copy()
            ),
            "representative_rear_support_ratio": self.representative_rear_support_ratio.copy(),
            "representative_geometric_side_contact_valid": (
                self.representative_geometric_side_contact_valid.copy()
            ),
            "tool_face_axis_world": self.tool_face_axis_world.copy(),
            "tool_face_horizontal_norm": self.tool_face_horizontal_norm.copy(),
            "tool_face_push_alignment": self.tool_face_push_alignment.copy(),
            "tool_desk_signed_distance_m": self.tool_desk_signed_distance_m.copy(),
            "tool_desk_contact_count": self.tool_desk_contact_count.copy(),
            "tool_desk_min_contact_distance_m": (self.tool_desk_min_contact_distance_m.copy()),
            "tool_desk_min_contact_distance_valid": (self.tool_desk_min_contact_distance_valid.copy()),
            "non_tool_robot_desk_contact_count": (self.non_tool_robot_desk_contact_count.copy()),
            "non_tool_robot_desk_min_distance_m": (self.non_tool_robot_desk_min_distance_m.copy()),
            "non_tool_robot_desk_min_distance_valid": (self.non_tool_robot_desk_min_distance_valid.copy()),
            "obstacle_contact_class_names": OBSTACLE_CONTACT_CLASSES,
            "obstacle_contact_counts": self.obstacle_contact_counts.copy(),
            "contact_any": bool(np.any(contact_mask)),
            "force_bearing_any": bool(np.any(force_mask)),
            "contact_substep_count": int(np.count_nonzero(contact_mask)),
            "first_contact_substep": (int(contact_indices[0]) if contact_indices.size else -1),
            "last_contact_substep": (int(contact_indices[-1]) if contact_indices.size else -1),
            "peak_normal_force_n": float(np.max(self.tool_block_total_normal_force_n)),
            "normal_impulse_discrete_ns": float(np.sum(self.tool_block_total_normal_force_n) * substep_dt),
            "valid_push_side_contact_any": valid_push_side_contact_any,
            "valid_push_side_contact_observed_any": (
                valid_push_side_contact_observed_any
            ),
            "invalid_tool_block_contact_any": invalid_tool_block_contact_any,
            "invalid_tool_block_contact_substep_count": int(
                np.count_nonzero(self.invalid_tool_block_contact_count > 0)
            ),
            "all_tool_block_contacts_geometrically_valid_transition": bool(
                not invalid_tool_block_contact_any
            ),
            "tool_block_contact_admissible_transition": valid_push_side_contact_any,
            "valid_push_side_contact_substep_count": int(np.count_nonzero(side_contact_mask)),
            "first_valid_push_side_contact_substep": (
                int(side_contact_indices[0]) if side_contact_indices.size else -1
            ),
            "last_valid_push_side_contact_substep": (
                int(side_contact_indices[-1]) if side_contact_indices.size else -1
            ),
            "valid_push_side_peak_normal_force_n": float(
                np.max(self.valid_push_side_total_normal_force_n)
            ),
            "valid_push_side_normal_impulse_discrete_ns": side_normal_impulse,
            "transient_contact_only": bool(np.any(contact_mask) and not contact_mask[-1]),
            "minimum_tool_desk_signed_distance_m": minimum_tool_desk_distance,
            "forbidden_tool_desk_penetration_any": bool(
                minimum_tool_desk_distance < -self.penetration_tolerance_m
            ),
            "forbidden_non_tool_robot_desk_penetration_any": (non_tool_negative_contact),
            "block_xy_displacement_m": (final_block_xy - self._initial_block_xy).copy(),
            "block_xy_displacement_norm_m": float(np.linalg.norm(final_block_xy - self._initial_block_xy)),
            "tool_xyz_displacement_m": (final_tool_xyz - self._initial_tool_xyz).copy(),
        }

    @staticmethod
    def _store_optional_minimum(
        values: list[float],
        destination: np.ndarray,
        valid: np.ndarray,
        index: int,
    ) -> None:
        if values:
            destination[index] = float(min(values))
            valid[index] = 1


__all__ = [
    "ContactTelemetryGeometry",
    "OBSTACLE_CONTACT_CLASSES",
    "PHYSICS_SUBSTEP_CONTACT_FORMAT",
    "PHYSICS_SUBSTEP_CONTACT_FORMAT_V1",
    "PHYSICS_SUBSTEP_CONTACT_FORMAT_V2",
    "PUSH_SIDE_CONTACT_FORMAT",
    "PUSH_SIDE_CONTACT_FORMAT_V1",
    "PUSH_SIDE_CONTACT_FORMAT_V2",
    "TOOL_CONTACT_IDENTITY_FORMAT",
    "TOOL_SAFETY_GEOM_ORDER_FORMAT",
    "PhysicsSubstepContactRecorder",
    "PushSideContactThresholds",
    "push_side_contact_metrics",
    "stable_tool_safety_geom_order_sha256",
]
