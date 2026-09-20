"""Production-aligned SO-101 simulation with versioned end-effector geometry.

Unlike the M2-M4 debug environment, this environment has no task-aligned mocap
tool.  The legacy profile retains its historical attached push plate, while the
stock-gripper profile uses the authored SO-101 fixed and moving jaw collision
meshes directly.  Every policy/teacher action remains a bounded six-joint
delta.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path, PurePosixPath
import stat
from typing import Any

import mujoco
import numpy as np

from .config import JOINT_NAMES, SCENE_PATH


BLOCK_COLORS = {
    "blue": ([0.05, 0.22, 0.92, 1.0], "蓝色"),
    "yellow": ([0.95, 0.68, 0.05, 1.0], "黄色"),
    "purple": ([0.58, 0.18, 0.80, 1.0], "紫色"),
    "cyan": ([0.02, 0.72, 0.82, 1.0], "青色"),
}

TARGET_COLORS = {
    "green": ([0.05, 0.80, 0.25, 0.38], "绿色"),
    "orange": ([0.95, 0.38, 0.04, 0.38], "橙色"),
    "magenta": ([0.85, 0.08, 0.48, 0.38], "品红色"),
}

WRIST_CAMERA_MOUNT_PROFILE_VERSION = "edgearm-synthetic-wrist-mount-framing-v2"
WRIST_CAMERA_MOUNT_PARAMETER_SOURCE = "synthetic_visibility_sweep_20_reset_seeds_no_physical_calibration"
WRIST_CAMERA_OPTICAL_POSITION_GRIPPER = (-0.024, 0.051, -0.032)
WRIST_CAMERA_OPTICAL_QUATERNION_WXYZ = (
    0.9937759588845572,
    0.037109946576033254,
    -0.10202419381945192,
    -0.024965161399317155,
)
WRIST_CAMERA_HOUSING_HALF_EXTENTS_M = (0.018, 0.018, 0.012)
WRIST_CAMERA_PHYSICAL_MOUNT_PROFILE_V1 = "edgearm-synthetic-wrist-camera-rigid-mount-visual-v1"
WRIST_CAMERA_HOUSING_GEOM = "production_wrist_camera_housing"
WRIST_CAMERA_MOUNT_GEOM = "production_wrist_camera_mount"
WRIST_CAMERA_LENS_GEOM = "production_wrist_camera_lens"

# V13-only static obstacle slots.  The historical ``obstacle`` geom remains
# slot zero so all frozen V6-V12 code keeps the same IDs and semantics.  The
# two additional slots are compiled only when an explicitly versioned caller
# opts in; they start invisible and non-colliding.
GENERALIZATION_OBSTACLE_GEOMS_V13 = (
    "obstacle",
    "generalization_obstacle_1_v13",
    "generalization_obstacle_2_v13",
)

# In the mounted-camera profile the MuJoCo optical center lies on the front
# face instead of in the middle of an opaque box.  The housing is shifted by
# one half-depth along camera-local +Z (away from the viewing direction).
WRIST_CAMERA_MOUNTED_HOUSING_POSITION_GRIPPER = (
    -0.026455575460868405,
    0.05017603395696623,
    -0.020282866089485608,
)
WRIST_CAMERA_MOUNT_POSITION_GRIPPER = (-0.024, 0.0272, -0.020)
WRIST_CAMERA_MOUNT_HALF_EXTENTS_M = (0.018, 0.0032, 0.015)
WRIST_CAMERA_LENS_POSITION_GRIPPER = (
    -0.02369205305878407,
    0.05110299575950636,
    -0.03346464174861599,
)

LEGACY_PUSH_PLATE_TOOL_PROFILE = "edgearm-attached-push-plate-v1"
STOCK_GRIPPER_DIRECT_PUSH_TOOL_PROFILE = "edgearm-stock-so101-gripper-direct-push-v1"
STOCK_GRIPPER_PUSH_JOINT_POSITION_RAD = -0.16
LEGACY_PUSH_PLATE_GRIPPER_JOINT_POSITION_RAD = 0.35
STOCK_GRIPPER_FIXED_COLLISION_GEOM = "stock_gripper_fixed_jaw_collision"
STOCK_GRIPPER_MOVING_COLLISION_GEOM = "stock_gripper_moving_jaw_collision"
STOCK_GRIPPER_PLANNING_GEOM = "stock_gripper_tip_planning_envelope"
STOCK_GRIPPER_FIXED_TIP_REFERENCE_GEOM = "stock_gripper_fixed_distal_tip_reference"
STOCK_GRIPPER_MOVING_TIP_REFERENCE_GEOM = "stock_gripper_moving_distal_tip_reference"
STOCK_GRIPPER_TIP_REFERENCE_GEOMS = (
    STOCK_GRIPPER_FIXED_TIP_REFERENCE_GEOM,
    STOCK_GRIPPER_MOVING_TIP_REFERENCE_GEOM,
)
STOCK_GRIPPER_FIXED_AUTHORED_MESH_REFERENCE_GEOM = "stock_gripper_fixed_authored_mesh_noncolliding"
STOCK_GRIPPER_MOVING_AUTHORED_MESH_REFERENCE_GEOM = "stock_gripper_moving_authored_mesh_noncolliding"
STOCK_GRIPPER_FIXED_SAFETY_CONVEX_PREFIX = "stock_gripper_fixed_safety_convex_"
STOCK_GRIPPER_MOVING_SAFETY_CONVEX_PREFIX = "stock_gripper_moving_safety_convex_"
STOCK_GRIPPER_CONTACT_PAIR_NAMES = (
    "realism_v8_fixed_jaw_block",
    "realism_v8_moving_jaw_block",
)

# Model-derived envelope of the closed stock jaw tips at q_gripper=-0.16 rad.
# It is invisible and non-colliding: only the original follower meshes create
# physical contacts.  The box supplies the existing IK and exact-OBB safety
# code with a deterministic reference face and conservative local bounds.
STOCK_GRIPPER_PLANNING_POSITION_GRIPPER = (-0.00548, -0.000218121, -0.0962774)
STOCK_GRIPPER_PLANNING_QUATERNION_WXYZ = (0.707107, 0.0, 0.707107, 0.0)
STOCK_GRIPPER_PLANNING_HALF_EXTENTS_M = (0.0082, 0.0069, 0.0135)

# Distal 2 mm fingertip envelopes derived from the authored collision meshes at
# q_gripper=-0.16 rad.  They are planning references only: contype and
# conaffinity remain zero, so the unmodified fixed and moving jaw meshes stay
# the sole physical contact authority.  Keeping the moving reference on the
# moving-jaw body preserves its true joint transform instead of freezing it in
# the gripper frame.
STOCK_GRIPPER_FIXED_TIP_REFERENCE_POSITION_GRIPPER = (
    -0.010763732021769824,
    -0.00022906946236225648,
    -0.1034259426593117,
)
STOCK_GRIPPER_FIXED_TIP_REFERENCE_QUATERNION_WXYZ = (1.0, 0.0, 0.0, 0.0)
STOCK_GRIPPER_FIXED_TIP_REFERENCE_HALF_EXTENTS_M = (
    0.00286373437067413,
    0.0031252649378993745,
    0.0009993503467570636,
)
STOCK_GRIPPER_MOVING_TIP_REFERENCE_POSITION_MOVING_BODY = (
    -0.009084943942645004,
    -0.0810166169100157,
    0.01890283997865192,
)
STOCK_GRIPPER_MOVING_TIP_REFERENCE_QUATERNION_WXYZ = (
    0.7048452475062174,
    -0.7048452475062171,
    -0.056508203545139746,
    0.05650820354513959,
)
STOCK_GRIPPER_MOVING_TIP_REFERENCE_HALF_EXTENTS_M = (
    0.0031839597687503074,
    0.003143683591655457,
    0.0009947417646483034,
)


@dataclass(frozen=True)
class ProductionEnvConfig:
    fps: int = 30
    physics_substeps: int = 4
    max_steps: int = 48
    max_joint_delta: float = 0.055
    success_radius: float = 0.060
    success_hold_steps: int = 3
    workspace_x: tuple[float, float] = (0.07, 0.43)
    workspace_y: tuple[float, float] = (-0.27, 0.27)
    workspace_z: tuple[float, float] = (0.045, 0.26)
    obstacle_probability: float = 0.30
    failure_recovery_probability: float = 0.18


@dataclass(frozen=True)
class ProductionMjcfBundleV1:
    """Immutable MJCF/include/asset bytes used for one model compilation.

    File names are logical POSIX paths rooted at the same capture root.  The
    representation intentionally contains no filesystem paths: once these
    bytes have been validated and committed, MuJoCo cannot reopen a changed
    source between provenance validation and compilation.
    """

    main_logical_path: str
    files: tuple[tuple[str, bytes], ...]

    def __post_init__(self) -> None:
        if not isinstance(self.main_logical_path, str):
            raise TypeError("main_logical_path must be a string")
        if not isinstance(self.files, tuple) or not self.files:
            raise ValueError("MJCF bundle files must be a non-empty tuple")
        main_path = _validate_bundle_logical_path(
            self.main_logical_path,
            label="main_logical_path",
        )
        observed: set[str] = set()
        for index, entry in enumerate(self.files):
            if not isinstance(entry, tuple) or len(entry) != 2:
                raise TypeError(f"MJCF bundle file {index} must be a (path, bytes) tuple")
            logical_path, payload = entry
            if not isinstance(logical_path, str):
                raise TypeError(f"MJCF bundle file {index} path must be a string")
            path = _validate_bundle_logical_path(
                logical_path,
                label=f"MJCF bundle file {index}",
            )
            if logical_path in observed:
                raise ValueError(f"MJCF bundle contains duplicate path: {logical_path}")
            observed.add(logical_path)
            if not isinstance(payload, bytes) or not payload:
                raise ValueError(f"MJCF bundle file must contain non-empty bytes: {logical_path}")
            try:
                path.relative_to(main_path.parent)
            except ValueError as error:
                raise ValueError("MJCF bundle files must reside below the main MJCF directory") from error
        if self.main_logical_path not in observed:
            raise ValueError("MJCF bundle does not contain its main MJCF file")


def _validate_bundle_logical_path(value: str, *, label: str) -> PurePosixPath:
    if not value or "\\" in value:
        raise ValueError(f"{label} must be a canonical relative POSIX path")
    path = PurePosixPath(value)
    if path.is_absolute() or path.as_posix() != value or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"{label} must be a canonical relative POSIX path")
    return path


def _spec_from_bundle(bundle: ProductionMjcfBundleV1) -> mujoco.MjSpec:
    if not isinstance(bundle, ProductionMjcfBundleV1):
        raise TypeError("scene_bundle must be a ProductionMjcfBundleV1")
    main_path = PurePosixPath(bundle.main_logical_path)
    root = main_path.parent
    main_payload: bytes | None = None
    includes: dict[str, bytes] = {}
    assets: dict[str, bytes] = {}
    for logical_path, payload in bundle.files:
        path = PurePosixPath(logical_path)
        relative_name = path.relative_to(root).as_posix()
        if logical_path == bundle.main_logical_path:
            main_payload = payload
        elif path.suffix.lower() == ".xml":
            includes[relative_name] = payload
        else:
            assets[relative_name] = payload
    if main_payload is None:  # pragma: no cover - guarded by the dataclass
        raise ValueError("MJCF bundle does not contain its main MJCF file")
    try:
        xml = main_payload.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError("Main MJCF must be valid UTF-8") from error
    return mujoco.MjSpec.from_string(xml, include=includes, assets=assets)


def build_production_model(
    *,
    scene_path: Path | None = None,
    scene_bundle: ProductionMjcfBundleV1 | None = None,
    include_camera_housing: bool = False,
    include_camera_mount: bool = False,
    include_visual_clutter: bool = False,
    include_realism_contact_pairs: bool = False,
    include_generalization_obstacles: bool = False,
    include_multichoice_blocks: bool = False,
    include_stock_tip_references: bool = False,
    tool_profile: str = LEGACY_PUSH_PLATE_TOOL_PROFILE,
) -> mujoco.MjModel:
    """Compile the production-aligned model.

    ``scene_bundle`` compiles already captured immutable bytes and is the
    formal provenance path. ``scene_path`` remains available for legacy and
    diagnostic callers; its leaf must be an existing ordinary file. Omitting
    both preserves the historical live-source path and every v1/v2 dataset and
    checkpoint contract.

    The optional model additions are reserved for newer, explicitly
    versioned realism profiles.  Keeping them disabled preserves the original
    production model contract. ``tool_profile`` is explicit so the historical
    plate and the stock follower gripper can never be confused in provenance.
    """

    if tool_profile not in {
        LEGACY_PUSH_PLATE_TOOL_PROFILE,
        STOCK_GRIPPER_DIRECT_PUSH_TOOL_PROFILE,
    }:
        raise ValueError(f"unsupported production tool profile: {tool_profile}")
    if include_stock_tip_references and tool_profile != STOCK_GRIPPER_DIRECT_PUSH_TOOL_PROFILE:
        raise ValueError("stock tip references require the stock-gripper tool profile")
    if include_camera_mount and not include_camera_housing:
        raise ValueError("camera mount requires the camera housing")
    if scene_path is not None and scene_bundle is not None:
        raise ValueError("scene_path and scene_bundle are mutually exclusive")
    if scene_bundle is not None:
        spec = _spec_from_bundle(scene_bundle)
    else:
        selected_scene_path = SCENE_PATH if scene_path is None else Path(scene_path)
        try:
            scene_mode = selected_scene_path.lstat().st_mode
        except FileNotFoundError as error:
            raise FileNotFoundError(f"Production scene does not exist: {selected_scene_path}") from error
        except OSError as error:
            raise ValueError(f"Production scene cannot be inspected: {selected_scene_path}") from error
        if stat.S_ISLNK(scene_mode):
            raise ValueError(f"Production scene must not be a symbolic link: {selected_scene_path}")
        if not stat.S_ISREG(scene_mode):
            raise ValueError(f"Production scene must be a regular file: {selected_scene_path}")
        spec = mujoco.MjSpec.from_file(str(selected_scene_path))
    proxy = spec.geom("arm_push_plate_geom")
    proxy.contype = 0
    proxy.conaffinity = 0
    proxy.rgba = [0.0, 0.0, 0.0, 0.0]
    gripper = spec.body("gripper")
    if tool_profile == LEGACY_PUSH_PLATE_TOOL_PROFILE:
        gripper.add_geom(
            name="production_push_plate",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            pos=[-0.0079, -0.000218121, -0.0981274],
            quat=[0.707107, 0.0, 0.707107, 0.0],
            size=[0.052, 0.008, 0.025],
            rgba=[0.92, 0.92, 0.86, 1.0],
            contype=1,
            conaffinity=1,
            friction=[1.0, 0.01, 0.001],
        )
    else:
        fixed_candidates = [
            geom
            for geom in gripper.geoms
            if int(geom.contype) != 0 and geom.meshname == "wrist_roll_follower_so101_v1"
        ]
        moving_body = spec.body("moving_jaw_so101_v1")
        moving_candidates = [
            geom
            for geom in moving_body.geoms
            if int(geom.contype) != 0 and geom.meshname == "moving_jaw_so101_v1"
        ]
        if len(fixed_candidates) != 1 or len(moving_candidates) != 1:
            raise ValueError("stock SO-101 jaw collision geometry is missing or ambiguous")
        fixed_source = fixed_candidates[0]
        moving_source = moving_candidates[0]
        if include_stock_tip_references:
            from .stock_gripper_convex_decomposition_v1 import (
                load_stock_gripper_convex_decomposition_v1,
            )

            decomposition = load_stock_gripper_convex_decomposition_v1()
            roles = decomposition.get("roles")
            if not isinstance(roles, dict):
                raise ValueError("stock-gripper convex decomposition has invalid roles")

            def add_convex_role(
                *,
                role: str,
                body: mujoco.MjsBody,
                source_geom: mujoco.MjsGeom,
                contact_name: str,
                safety_prefix: str,
            ) -> None:
                role_payload = roles.get(role)
                if not isinstance(role_payload, dict):
                    raise ValueError(f"stock-gripper decomposition is missing {role}")
                parts = role_payload.get("parts")
                contact_index = role_payload.get("contact_part_index")
                if not isinstance(parts, list) or not parts:
                    raise ValueError(f"stock-gripper decomposition has no {role} parts")
                if not isinstance(contact_index, int) or not 0 <= contact_index < len(parts):
                    raise ValueError(f"stock-gripper decomposition has invalid {role} contact part")
                source_position = np.asarray(source_geom.pos, dtype=np.float64).copy().tolist()
                source_quaternion = np.asarray(source_geom.quat, dtype=np.float64).copy().tolist()
                for part_index, part in enumerate(parts):
                    if not isinstance(part, dict):
                        raise ValueError(f"stock-gripper {role} part is invalid")
                    vertices = np.asarray(part.get("vertices"), dtype=np.float64)
                    faces = np.asarray(part.get("faces"), dtype=np.int64)
                    if (
                        vertices.ndim != 2
                        or vertices.shape[1] != 3
                        or faces.ndim != 2
                        or faces.shape[1] != 3
                        or not np.all(np.isfinite(vertices))
                    ):
                        raise ValueError(f"stock-gripper {role} convex part arrays are invalid")
                    mesh_name = f"{safety_prefix}mesh_{part_index:03d}"
                    spec.add_mesh(
                        name=mesh_name,
                        uservert=vertices.reshape(-1).tolist(),
                        userface=faces.reshape(-1).tolist(),
                        maxhullvert=32,
                    )
                    geom_name = (
                        contact_name if part_index == contact_index else f"{safety_prefix}{part_index:03d}"
                    )
                    body.add_geom(
                        name=geom_name,
                        type=mujoco.mjtGeom.mjGEOM_MESH,
                        pos=source_position,
                        quat=source_quaternion,
                        meshname=mesh_name,
                        rgba=[0.0, 0.0, 0.0, 0.0],
                        contype=1,
                        conaffinity=1,
                        group=3,
                        friction=[1.0, 0.01, 0.001],
                        mass=0.0,
                    )

            fixed_source.name = STOCK_GRIPPER_FIXED_AUTHORED_MESH_REFERENCE_GEOM
            fixed_source.contype = 0
            fixed_source.conaffinity = 0
            moving_source.name = STOCK_GRIPPER_MOVING_AUTHORED_MESH_REFERENCE_GEOM
            moving_source.contype = 0
            moving_source.conaffinity = 0
            add_convex_role(
                role="fixed_jaw",
                body=gripper,
                source_geom=fixed_source,
                contact_name=STOCK_GRIPPER_FIXED_COLLISION_GEOM,
                safety_prefix=STOCK_GRIPPER_FIXED_SAFETY_CONVEX_PREFIX,
            )
            add_convex_role(
                role="moving_jaw",
                body=moving_body,
                source_geom=moving_source,
                contact_name=STOCK_GRIPPER_MOVING_COLLISION_GEOM,
                safety_prefix=STOCK_GRIPPER_MOVING_SAFETY_CONVEX_PREFIX,
            )
        else:
            fixed_source.name = STOCK_GRIPPER_FIXED_COLLISION_GEOM
            moving_source.name = STOCK_GRIPPER_MOVING_COLLISION_GEOM
        gripper.add_geom(
            name=STOCK_GRIPPER_PLANNING_GEOM,
            type=mujoco.mjtGeom.mjGEOM_BOX,
            pos=list(STOCK_GRIPPER_PLANNING_POSITION_GRIPPER),
            quat=list(STOCK_GRIPPER_PLANNING_QUATERNION_WXYZ),
            size=list(STOCK_GRIPPER_PLANNING_HALF_EXTENTS_M),
            rgba=[0.0, 0.0, 0.0, 0.0],
            contype=0,
            conaffinity=0,
            group=5,
            mass=0.0,
        )
        if include_stock_tip_references:
            gripper.add_geom(
                name=STOCK_GRIPPER_FIXED_TIP_REFERENCE_GEOM,
                type=mujoco.mjtGeom.mjGEOM_BOX,
                pos=list(STOCK_GRIPPER_FIXED_TIP_REFERENCE_POSITION_GRIPPER),
                quat=list(STOCK_GRIPPER_FIXED_TIP_REFERENCE_QUATERNION_WXYZ),
                size=list(STOCK_GRIPPER_FIXED_TIP_REFERENCE_HALF_EXTENTS_M),
                rgba=[0.0, 0.0, 0.0, 0.0],
                contype=0,
                conaffinity=0,
                group=5,
                mass=0.0,
            )
            moving_body.add_geom(
                name=STOCK_GRIPPER_MOVING_TIP_REFERENCE_GEOM,
                type=mujoco.mjtGeom.mjGEOM_BOX,
                pos=list(STOCK_GRIPPER_MOVING_TIP_REFERENCE_POSITION_MOVING_BODY),
                quat=list(STOCK_GRIPPER_MOVING_TIP_REFERENCE_QUATERNION_WXYZ),
                size=list(STOCK_GRIPPER_MOVING_TIP_REFERENCE_HALF_EXTENTS_M),
                rgba=[0.0, 0.0, 0.0, 0.0],
                contype=0,
                conaffinity=0,
                group=5,
                mass=0.0,
            )
    # Rigid wrist RGB-D camera: local pose is attached to the moving gripper.
    # This v2 synthetic prior pitches the original optical frame -27.5 degrees
    # about camera-local X and widens vertical FOV from 85 to 95 degrees.  A
    # 20-seed MuJoCo visibility sweep kept both block and target fully inside
    # all reset frames.  It is an uncalibrated framing prior, not a claim about
    # the user's eventual physical camera-to-wrist transform.
    gripper.add_camera(
        name="edgearm_wrist",
        pos=list(WRIST_CAMERA_OPTICAL_POSITION_GRIPPER),
        quat=list(WRIST_CAMERA_OPTICAL_QUATERNION_WXYZ),
        fovy=95.0,
        resolution=[640, 480],
    )
    if include_camera_housing:
        # Collision/visual proxy for a compact wrist camera body.  Its mass is
        # applied explicitly by the V6 environment because the SO-101 gripper
        # already has an authored inertial.  Dimensions are an uncalibrated
        # engineering prior, not a claim about the user's physical camera.
        gripper.add_geom(
            name=WRIST_CAMERA_HOUSING_GEOM,
            type=mujoco.mjtGeom.mjGEOM_BOX,
            pos=list(
                WRIST_CAMERA_MOUNTED_HOUSING_POSITION_GRIPPER
                if include_camera_mount
                else WRIST_CAMERA_OPTICAL_POSITION_GRIPPER
            ),
            quat=list(WRIST_CAMERA_OPTICAL_QUATERNION_WXYZ if include_camera_mount else (1.0, 0.0, 0.0, 0.0)),
            size=list(WRIST_CAMERA_HOUSING_HALF_EXTENTS_M),
            rgba=[0.08, 0.09, 0.10, 1.0],
            contype=1,
            conaffinity=1,
            mass=0.0,
        )
        if include_camera_mount:
            # One printable side plate bridges the authored gripper shell and
            # the camera housing.  It is a synthetic mechanical placeholder:
            # dimensions remain explicitly uncalibrated until the user's
            # actual camera and printed bracket are measured.
            gripper.add_geom(
                name=WRIST_CAMERA_MOUNT_GEOM,
                type=mujoco.mjtGeom.mjGEOM_BOX,
                pos=list(WRIST_CAMERA_MOUNT_POSITION_GRIPPER),
                size=list(WRIST_CAMERA_MOUNT_HALF_EXTENTS_M),
                rgba=[1.0, 0.82, 0.12, 1.0],
                contype=1,
                conaffinity=1,
                mass=0.0,
            )
            gripper.add_geom(
                name=WRIST_CAMERA_LENS_GEOM,
                type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                pos=list(WRIST_CAMERA_LENS_POSITION_GRIPPER),
                quat=list(WRIST_CAMERA_OPTICAL_QUATERNION_WXYZ),
                size=[0.006, 0.0015],
                rgba=[0.015, 0.025, 0.035, 1.0],
                contype=0,
                conaffinity=0,
                group=2,
                mass=0.0,
            )
    if include_visual_clutter:
        # Non-colliding edge-of-workspace objects prevent a single flat RGBA
        # tabletop from being the only background.  V6 randomizes these geoms
        # per episode while keeping them independent of the task labels.
        clutter = (
            ("realism_v6_clutter_left", [0.065, -0.285, 0.055], [0.042, 0.018, 0.040]),
            ("realism_v6_clutter_right", [0.465, 0.255, 0.045], [0.030, 0.028, 0.030]),
            ("realism_v6_clutter_rear", [0.105, 0.292, 0.042], [0.075, 0.012, 0.022]),
        )
        for name, pos, size in clutter:
            spec.worldbody.add_geom(
                name=name,
                type=mujoco.mjtGeom.mjGEOM_BOX,
                pos=pos,
                size=size,
                rgba=[0.18, 0.18, 0.17, 1.0],
                contype=0,
                conaffinity=0,
            )
        # Compile several deterministic, non-task-correlated laminate/cloth
        # surfaces.  V6 can switch material IDs per episode without rebuilding
        # the renderer or depending on external image files.
        palettes = (
            ([116, 91, 65], [174, 139, 96]),
            ([86, 91, 94], [154, 158, 160]),
            ([70, 83, 68], [137, 148, 125]),
            ([104, 91, 82], [181, 169, 151]),
            ([61, 74, 91], [126, 143, 160]),
            ([126, 113, 90], [194, 181, 151]),
        )
        yy, xx = np.meshgrid(
            np.linspace(0.0, 1.0, 96),
            np.linspace(0.0, 1.0, 96),
            indexing="ij",
        )
        for index, (low_rgb, high_rgb) in enumerate(palettes):
            texture_rng = np.random.default_rng(0xEA6E00 + index)
            low = np.asarray(low_rgb, dtype=np.float32)
            high = np.asarray(high_rgb, dtype=np.float32)
            grain = 0.5 + 0.28 * np.sin(
                (8.0 + index) * np.pi * xx + 0.7 * np.sin((3.0 + 0.4 * index) * np.pi * yy)
            )
            grain += texture_rng.normal(0.0, 0.075, grain.shape)
            image = low + np.clip(grain, 0.0, 1.0)[..., None] * (high - low)
            texture_name = f"realism_v6_surface_texture_{index}"
            texture = spec.add_texture(
                name=texture_name,
                type=mujoco.mjtTexture.mjTEXTURE_2D,
                builtin=mujoco.mjtBuiltin.mjBUILTIN_NONE,
                width=image.shape[1],
                height=image.shape[0],
            )
            texture.data = np.clip(image, 0, 255).astype(np.uint8).tobytes()
            spec.add_material(
                name=f"realism_v6_surface_{index}",
                textures=[texture_name],
                texrepeat=[3.0 + index % 3, 3.0 + (index + 1) % 3],
                reflectance=0.04 + 0.025 * index,
                roughness=0.62 + 0.05 * (index % 3),
            )
        spec.geom("edgearm_desk").material = "realism_v6_surface_0"
    if include_generalization_obstacles:
        # Runtime V13 code changes type/size/pose for each slot.  Keeping one
        # geom per slot (instead of one geom per possible shape) makes contact
        # identity stable in every trajectory and replay.
        for name in GENERALIZATION_OBSTACLE_GEOMS_V13[1:]:
            spec.worldbody.add_geom(
                name=name,
                type=mujoco.mjtGeom.mjGEOM_BOX,
                pos=[0.55, 0.28, 0.055],
                size=[0.018, 0.018, 0.030],
                rgba=[0.72, 0.18, 0.12, 0.0],
                contype=0,
                conaffinity=0,
            )
    if include_realism_contact_pairs:
        spec.add_pair(
            name="realism_v6_desk_block",
            geomname1="edgearm_desk",
            geomname2="push_block_geom",
            condim=6,
            friction=[0.70, 0.70, 0.015, 0.0015, 0.0015],
            solref=[0.008, 1.0],
            solimp=[0.90, 0.95, 0.001, 0.5, 2.0],
        )
        pusher_geometries = (
            (("realism_v6_pusher_block", "production_push_plate"),)
            if tool_profile == LEGACY_PUSH_PLATE_TOOL_PROFILE
            else tuple(
                zip(
                    STOCK_GRIPPER_CONTACT_PAIR_NAMES,
                    (
                        STOCK_GRIPPER_FIXED_COLLISION_GEOM,
                        STOCK_GRIPPER_MOVING_COLLISION_GEOM,
                    ),
                    strict=True,
                )
            )
        )
        for pair_name, geom_name in pusher_geometries:
            spec.add_pair(
                name=pair_name,
                geomname1=geom_name,
                geomname2="push_block_geom",
                condim=6,
                friction=[0.90, 0.90, 0.020, 0.0020, 0.0020],
                solref=[0.006, 1.0],
                solimp=[0.92, 0.97, 0.001, 0.5, 2.0],
            )
        if include_camera_housing:
            spec.add_pair(
                name="realism_v6_obstacle_camera",
                geomname1="obstacle",
                geomname2="production_wrist_camera_housing",
                condim=3,
                friction=[0.65, 0.65, 0.008, 0.0008, 0.0008],
            )
    if include_multichoice_blocks:
        source_body = spec.body("push_block")
        for index in range(2):
            body = spec.worldbody.add_body(name=f"choice_block_{index}", pos=[0.55, 0.35 + 0.12 * index, 0.051],
                                          mass=source_body.mass, inertia=source_body.inertia,
                                          ipos=[0, 0, 0], iquat=[1, 0, 0, 0], explicitinertial=True)
            body.add_freejoint(name=f"choice_block_joint_{index}")
            body.add_geom(name=f"choice_block_geom_{index}", type=mujoco.mjtGeom.mjGEOM_BOX,
                          size=[0.025, 0.025, 0.025], rgba=[0.2, 0.3, 0.9, 1],
                          contype=1, conaffinity=1, friction=[0.7, 0.018, 0.002])
            spec.worldbody.add_geom(name=f"choice_target_{index}", type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                                    pos=[0.55, 0.35 + 0.12 * index, 0.026], size=[0.055, 0.0005],
                                    rgba=[0.1, 0.8, 0.2, 0.4], contype=0, conaffinity=0)
    return spec.compile()


class ProductionEdgeArmEnv:
    action_dim = 6
    model_tool_profile = LEGACY_PUSH_PLATE_TOOL_PROFILE

    def __init__(self, config: ProductionEnvConfig | None = None, seed: int = 0):
        self.config = config or ProductionEnvConfig()
        self.model = self._build_model()
        self.data = mujoco.MjData(self.model)
        self.rng = np.random.default_rng(seed)
        self.seed = seed
        self.estop = False
        self.step_count = 0
        self.success_streak = 0
        self.target_xy = np.zeros(2, dtype=np.float64)
        self.obstacle_xy = np.zeros(2, dtype=np.float64)
        self.obstacle_enabled = False
        self.last_distance = 0.0
        self.task_text = ""
        self.task_text_zh = ""
        self.current_stress = False
        self.color_name = "blue"
        self.target_color_name = "green"
        self._ids = self._resolve_ids()
        self.tool_gripper_joint_position_rad = (
            STOCK_GRIPPER_PUSH_JOINT_POSITION_RAD
            if self._ids["tool_profile"] == STOCK_GRIPPER_DIRECT_PUSH_TOOL_PROFILE
            else LEGACY_PUSH_PLATE_GRIPPER_JOINT_POSITION_RAD
        )
        self.joint_ranges = np.asarray(
            [self.model.jnt_range[self._ids["joints"][name]] for name in JOINT_NAMES], dtype=np.float64
        )
        self.default_block_mass = float(self.model.body_mass[self._ids["block_body"]])
        self.default_block_friction = self.model.geom_friction[self._ids["block_geom"]].copy()
        self.default_light_diffuse = self.model.light_diffuse.copy()
        self.default_camera_pos = {
            name: self.model.cam_pos[camera_id].copy() for name, camera_id in self._ids["cameras"].items()
        }
        self.default_camera_quat = {
            name: self.model.cam_quat[camera_id].copy() for name, camera_id in self._ids["cameras"].items()
        }
        self.default_camera_fovy = {
            name: float(self.model.cam_fovy[camera_id]) for name, camera_id in self._ids["cameras"].items()
        }
        self.episode_domain: dict[str, Any] = {}

    def _build_model(self) -> mujoco.MjModel:
        """Model-construction hook used by opt-in versioned environments."""

        return build_production_model(tool_profile=self.model_tool_profile)

    def _resolve_ids(self) -> dict[str, Any]:
        def obj(kind: mujoco.mjtObj, name: str) -> int:
            value = mujoco.mj_name2id(self.model, kind, name)
            if value < 0:
                raise ValueError(f"Production scene is missing {name}")
            return value

        legacy_tool = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_GEOM,
            "production_push_plate",
        )
        stock_reference = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_GEOM,
            STOCK_GRIPPER_PLANNING_GEOM,
        )
        stock_tip_references = tuple(
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, name)
            for name in STOCK_GRIPPER_TIP_REFERENCE_GEOMS
        )
        stock_tip_reference_count = sum(value >= 0 for value in stock_tip_references)
        if stock_tip_reference_count not in {0, len(STOCK_GRIPPER_TIP_REFERENCE_GEOMS)}:
            raise ValueError("stock fingertip planning references are incomplete")
        if (legacy_tool >= 0) == (stock_reference >= 0):
            raise ValueError("production model must contain exactly one tool profile")
        if stock_reference >= 0:
            tool_profile = STOCK_GRIPPER_DIRECT_PUSH_TOOL_PROFILE
            tool_geom = stock_reference
            tool_contact_geoms = (
                obj(mujoco.mjtObj.mjOBJ_GEOM, STOCK_GRIPPER_FIXED_COLLISION_GEOM),
                obj(mujoco.mjtObj.mjOBJ_GEOM, STOCK_GRIPPER_MOVING_COLLISION_GEOM),
            )
            tool_planning_geoms = (
                stock_tip_references
                if stock_tip_reference_count == len(STOCK_GRIPPER_TIP_REFERENCE_GEOMS)
                else (stock_reference,)
            )
            tool_contact_geom_roles = (
                ("fixed_tip", "moving_tip") if len(tool_planning_geoms) == 2 else ("fixed_jaw", "moving_jaw")
            )
            tool_planning_geometry_mode = (
                "per_jaw_distal_tip_references" if len(tool_planning_geoms) == 2 else "combined_tip_envelope"
            )
            if len(tool_planning_geoms) == 2:
                fixed_safety = sorted(
                    (
                        geom_id
                        for geom_id in range(self.model.ngeom)
                        if (
                            mujoco.mj_id2name(
                                self.model,
                                mujoco.mjtObj.mjOBJ_GEOM,
                                geom_id,
                            )
                            or ""
                        ).startswith(STOCK_GRIPPER_FIXED_SAFETY_CONVEX_PREFIX)
                    ),
                    key=lambda geom_id: mujoco.mj_id2name(
                        self.model,
                        mujoco.mjtObj.mjOBJ_GEOM,
                        geom_id,
                    ),
                )
                moving_safety = sorted(
                    (
                        geom_id
                        for geom_id in range(self.model.ngeom)
                        if (
                            mujoco.mj_id2name(
                                self.model,
                                mujoco.mjtObj.mjOBJ_GEOM,
                                geom_id,
                            )
                            or ""
                        ).startswith(STOCK_GRIPPER_MOVING_SAFETY_CONVEX_PREFIX)
                    ),
                    key=lambda geom_id: mujoco.mj_id2name(
                        self.model,
                        mujoco.mjtObj.mjOBJ_GEOM,
                        geom_id,
                    ),
                )
                tool_safety_geoms = (
                    tool_contact_geoms[0],
                    *fixed_safety,
                    tool_contact_geoms[1],
                    *moving_safety,
                )
                tool_safety_geom_roles = (
                    "fixed_jaw_distal_contact",
                    *("fixed_jaw_safety" for _ in fixed_safety),
                    "moving_jaw_distal_contact",
                    *("moving_jaw_safety" for _ in moving_safety),
                )
                tool_safety_geometry_mode = "coacd_convex_union"
                authored_reference_names = (
                    STOCK_GRIPPER_FIXED_AUTHORED_MESH_REFERENCE_GEOM,
                    STOCK_GRIPPER_MOVING_AUTHORED_MESH_REFERENCE_GEOM,
                )
                tool_authored_mesh_reference_geoms = tuple(
                    obj(mujoco.mjtObj.mjOBJ_GEOM, name) for name in authored_reference_names
                )
            else:
                tool_safety_geoms = (stock_reference,)
                tool_safety_geom_roles = ("combined_tip_envelope",)
                tool_safety_geometry_mode = "combined_tip_envelope"
                tool_authored_mesh_reference_geoms = ()
        else:
            tool_profile = LEGACY_PUSH_PLATE_TOOL_PROFILE
            tool_geom = legacy_tool
            tool_contact_geoms = (legacy_tool,)
            tool_contact_geom_roles = ("legacy_push_plate",)
            tool_planning_geoms = (legacy_tool,)
            tool_planning_geometry_mode = "physical_legacy_push_plate"
            tool_safety_geoms = (legacy_tool,)
            tool_safety_geom_roles = ("legacy_push_plate",)
            tool_safety_geometry_mode = "physical_legacy_push_plate"
            tool_authored_mesh_reference_geoms = ()
        return {
            "joints": {name: obj(mujoco.mjtObj.mjOBJ_JOINT, name) for name in JOINT_NAMES},
            "actuators": {name: obj(mujoco.mjtObj.mjOBJ_ACTUATOR, name) for name in JOINT_NAMES},
            "block_joint": obj(mujoco.mjtObj.mjOBJ_JOINT, "push_block_freejoint"),
            "block_body": obj(mujoco.mjtObj.mjOBJ_BODY, "push_block"),
            "gripper_body": obj(mujoco.mjtObj.mjOBJ_BODY, "gripper"),
            "block_geom": obj(mujoco.mjtObj.mjOBJ_GEOM, "push_block_geom"),
            "target_geom": obj(mujoco.mjtObj.mjOBJ_GEOM, "target_zone"),
            "obstacle_geom": obj(mujoco.mjtObj.mjOBJ_GEOM, "obstacle"),
            "tool_profile": tool_profile,
            "tool_geom": tool_geom,
            "tool_orientation_geom": tool_geom,
            "tool_contact_geoms": tool_contact_geoms,
            "tool_contact_geom_roles": tool_contact_geom_roles,
            "tool_planning_geoms": tool_planning_geoms,
            "tool_contact_reference_geoms": tool_planning_geoms,
            "tool_planning_geometry_mode": tool_planning_geometry_mode,
            "tool_safety_geoms": tool_safety_geoms,
            "tool_safety_geom_roles": tool_safety_geom_roles,
            "tool_safety_geometry_mode": tool_safety_geometry_mode,
            "tool_authored_mesh_reference_geoms": tool_authored_mesh_reference_geoms,
            "tool_site": obj(mujoco.mjtObj.mjOBJ_SITE, "gripperframe"),
            "cameras": {
                "wrist": obj(mujoco.mjtObj.mjOBJ_CAMERA, "edgearm_wrist"),
            },
        }

    def reset(
        self,
        seed: int | None = None,
        *,
        obstacle: bool | None = None,
        stress: bool = False,
    ) -> dict[str, np.ndarray]:
        if seed is not None:
            self.seed = seed
            self.rng = np.random.default_rng(seed)
        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        self.estop = False
        self.step_count = 0
        self.success_streak = 0
        self.current_stress = stress
        block, target = self._sample_task(stress)
        self.target_xy = target
        self._set_block(block)
        self.model.geom_pos[self._ids["target_geom"], :2] = target
        self.obstacle_enabled = (
            bool(obstacle)
            if obstacle is not None
            else bool(self.rng.random() < (0.55 if stress else self.config.obstacle_probability))
        )
        self._set_obstacle(block, target)
        self._randomize_domain(stress)
        q = self._initial_reset_joint_position(block, target)
        self.data.qpos[:6] = q
        self.data.qvel[:6] = 0.0
        self.data.ctrl[:] = q
        mujoco.mj_forward(self.model, self.data)
        self.last_distance = self.distance_to_target()
        self._set_task_language()
        return self.observation()

    def _initial_reset_joint_position(
        self,
        block: np.ndarray,
        target: np.ndarray,
    ) -> np.ndarray:
        """Return the arm state installed before the reset ``mj_forward``.

        The production baseline keeps its historical task-aligned reset.  A
        versioned successor may override this hook when it needs a genuinely
        task-independent Home reset; keeping the choice ahead of the first
        forward pass avoids a privileged transient arm state touching the
        sampled scene.
        """

        direction = self._unit(target - block)
        start = block - direction * 0.10
        return self.solve_pose_ik(
            np.array([start[0], start[1], 0.076]),
            direction,
        )

    def _sample_task(self, stress: bool) -> tuple[np.ndarray, np.ndarray]:
        angle_limit = 0.48 if stress else 0.22
        for _ in range(300):
            block = self.rng.uniform(
                [0.18, -0.12] if stress else [0.19, -0.085],
                [0.24, 0.12] if stress else [0.23, 0.085],
            )
            distance = float(self.rng.uniform(0.115, 0.19) if stress else self.rng.uniform(0.105, 0.15))
            angle = float(self.rng.uniform(-angle_limit, angle_limit))
            target = block + distance * np.array([np.cos(angle), np.sin(angle)])
            if 0.29 <= target[0] <= 0.40 and -0.17 <= target[1] <= 0.17:
                return block, target
        raise RuntimeError("Could not sample production task")

    def _set_block(self, xy: np.ndarray) -> None:
        address = self.model.jnt_qposadr[self._ids["block_joint"]]
        self.data.qpos[address : address + 7] = [xy[0], xy[1], 0.051, 1.0, 0.0, 0.0, 0.0]

    def _set_obstacle(self, block: np.ndarray, target: np.ndarray) -> None:
        geom = self._ids["obstacle_geom"]
        if not self.obstacle_enabled:
            self.obstacle_xy = np.array([0.55, 0.28])
            self.model.geom_pos[geom, :2] = self.obstacle_xy
            self.model.geom_contype[geom] = 0
            self.model.geom_conaffinity[geom] = 0
            self.model.geom_rgba[geom, 3] = 0.0
            return
        direction = self._unit(target - block)
        normal = np.array([-direction[1], direction[0]])
        offset_limit = 0.055 if self.current_stress else 0.115
        self.obstacle_xy = (
            0.52 * block + 0.48 * target + normal * float(self.rng.uniform(-offset_limit, offset_limit))
        )
        self.model.geom_pos[geom, :2] = self.obstacle_xy
        self.model.geom_contype[geom] = 1
        self.model.geom_conaffinity[geom] = 1
        self.model.geom_rgba[geom] = [0.72, 0.18, 0.12, 1.0]

    def _randomize_domain(self, stress: bool) -> None:
        strength = 1.6 if stress else 1.0
        friction = float(self.rng.uniform(0.48, 1.25) ** strength)
        mass_scale = float(self.rng.uniform(0.65, 1.45) ** strength)
        self.model.geom_friction[self._ids["block_geom"], 0] = friction
        self.model.body_mass[self._ids["block_body"]] = self.default_block_mass * mass_scale
        self.color_name = str(self.rng.choice(list(BLOCK_COLORS)))
        self.target_color_name = str(self.rng.choice(list(TARGET_COLORS)))
        block_rgba = BLOCK_COLORS[self.color_name][0]
        target_rgba = TARGET_COLORS[self.target_color_name][0]
        block_material = self.model.geom_matid[self._ids["block_geom"]]
        target_material = self.model.geom_matid[self._ids["target_geom"]]
        if block_material >= 0:
            self.model.mat_rgba[block_material] = block_rgba
        else:
            self.model.geom_rgba[self._ids["block_geom"]] = block_rgba
        if target_material >= 0:
            self.model.mat_rgba[target_material] = target_rgba
        else:
            self.model.geom_rgba[self._ids["target_geom"]] = target_rgba
        camera_meta: dict[str, Any] = {}
        for name, camera_id in self._ids["cameras"].items():
            position_std = 0.0015 if name == "wrist" else 0.0045
            pos_noise = self.rng.normal(0.0, position_std * strength, 3)
            self.model.cam_pos[camera_id] = self.default_camera_pos[name] + pos_noise
            self.model.cam_quat[camera_id] = self.default_camera_quat[name]
            fovy = self.default_camera_fovy[name] + float(self.rng.normal(0.0, 0.65 * strength))
            fovy_bounds = (75.0, 95.0) if name == "wrist" else (35.0, 55.0)
            self.model.cam_fovy[camera_id] = np.clip(fovy, *fovy_bounds)
            camera_meta[name] = {
                "local_position": self.model.cam_pos[camera_id].tolist(),
                "local_quaternion_wxyz": self.model.cam_quat[camera_id].tolist(),
                "fovy_degrees": float(self.model.cam_fovy[camera_id]),
            }
        light_scale = float(self.rng.uniform(0.65, 1.35))
        self.model.light_diffuse[:] = np.clip(self.default_light_diffuse * light_scale, 0.15, 1.0)
        self.episode_domain = {
            "block_friction": friction,
            "block_mass_kg": float(self.model.body_mass[self._ids["block_body"]]),
            "camera": camera_meta,
            "light_scale": light_scale,
            "stress": stress,
            "obstacle": self.obstacle_enabled,
            "block_color": self.color_name,
            "target_color": self.target_color_name,
            "tool_profile": self._ids["tool_profile"],
            "tool_contact_geometry_count": len(self._ids["tool_contact_geoms"]),
            "tool_gripper_joint_position_rad": self.tool_gripper_joint_position_rad,
        }

    def _set_task_language(self) -> None:
        block_zh = BLOCK_COLORS[self.color_name][1]
        target_zh = TARGET_COLORS[self.target_color_name][1]
        tool_en = (
            "stock gripper"
            if self._ids["tool_profile"] == STOCK_GRIPPER_DIRECT_PUSH_TOOL_PROFILE
            else "wide pusher"
        )
        tool_zh = (
            "原装夹爪" if self._ids["tool_profile"] == STOCK_GRIPPER_DIRECT_PUSH_TOOL_PROFILE else "宽推板"
        )
        variants = [
            f"Push the {self.color_name} block into the {self.target_color_name} target zone.",
            f"Move the {self.color_name} cube to the {self.target_color_name} circle using the robot arm.",
            f"Use the {tool_en} to send the {self.color_name} block to the {self.target_color_name} goal.",
            f"Guide the {self.color_name} object into the {self.target_color_name} destination area.",
        ]
        variants_obstacle = [
            f"Push the {self.color_name} block to the {self.target_color_name} target while avoiding the red obstacle.",
            f"Move the {self.color_name} cube around the red barrier and into the {self.target_color_name} goal.",
            f"Reach the {self.target_color_name} zone with the {self.color_name} block without hitting the red obstacle.",
        ]
        variants_zh = [
            f"把{block_zh}木块推入{target_zh}目标区。",
            f"使用机械臂{tool_zh}将{block_zh}方块送到{target_zh}圆形区域。",
            f"将{block_zh}物体稳定推到{target_zh}终点。",
        ]
        variants_obstacle_zh = [
            f"避开红色障碍物，把{block_zh}木块推入{target_zh}目标区。",
            f"让{block_zh}方块绕过红色挡板并到达{target_zh}区域。",
        ]
        self.task_text = str(self.rng.choice(variants_obstacle if self.obstacle_enabled else variants))
        self.task_text_zh = str(
            self.rng.choice(variants_obstacle_zh if self.obstacle_enabled else variants_zh)
        )

    def solve_pose_ik(
        self,
        target_xyz: np.ndarray,
        push_direction: np.ndarray,
        initial: np.ndarray | None = None,
    ) -> np.ndarray:
        scratch = mujoco.MjData(self.model)
        mujoco.mj_resetDataKeyframe(self.model, scratch, 0)
        if initial is not None:
            scratch.qpos[:6] = initial
        desired_yaw = float(np.arctan2(push_direction[1], push_direction[0]))
        site = self._ids["tool_site"]
        for _ in range(180):
            mujoco.mj_forward(self.model, scratch)
            pos_error = np.asarray(target_xyz) - scratch.site_xpos[site]
            current_rotation = scratch.site_xmat[site].reshape(3, 3)
            normal_yaw = float(np.arctan2(current_rotation[1, 1], current_rotation[0, 1]))
            yaw_error = (desired_yaw - normal_yaw + np.pi) % (2 * np.pi) - np.pi
            if np.linalg.norm(pos_error) < 2.5e-4 and abs(yaw_error) < 0.025:
                break
            jac_pos = np.zeros((3, self.model.nv))
            jac_rot = np.zeros((3, self.model.nv))
            mujoco.mj_jacSite(self.model, scratch, jac_pos, jac_rot, site)
            weight = 0.18
            jacobian = np.vstack([jac_pos[:, :5], weight * jac_rot[2:3, :5]])
            error = np.concatenate([pos_error, [weight * yaw_error]])
            delta = jacobian.T @ np.linalg.solve(jacobian @ jacobian.T + 1e-3 * np.eye(4), error)
            scratch.qpos[:5] += np.clip(delta, -0.07, 0.07)
            scratch.qpos[:5] = np.clip(scratch.qpos[:5], self.joint_ranges[:5, 0], self.joint_ranges[:5, 1])
        scratch.qpos[5] = self.tool_gripper_joint_position_rad
        return scratch.qpos[:6].copy()

    def teacher_action(self) -> tuple[np.ndarray, dict[str, Any]]:
        block = self.block_xy()
        target = self.target_xy
        direction = self._unit(target - block)
        tool = self.tool_xyz()
        contact = block - direction * 0.039
        offset = block - tool[:2]
        along = float(np.dot(offset, direction))
        lateral = float(abs(direction[0] * offset[1] - direction[1] * offset[0]))
        phase = "approach"
        desired_xy = contact
        if along < 0.063 and lateral < 0.045:
            phase = "push"
            desired_xy = tool[:2] + direction * 0.021
        if self.obstacle_enabled and self._path_intersects_obstacle(block, target):
            normal = np.array([-direction[1], direction[0]])
            side = np.sign(np.dot(block - self.obstacle_xy, normal)) or 1.0
            waypoint = self.obstacle_xy + normal * side * 0.095
            local = self._unit(waypoint - block)
            desired_xy = block - local * 0.039
            direction = local
            phase = "avoid"
            if np.linalg.norm(tool[:2] - desired_xy) < 0.025:
                desired_xy = tool[:2] + local * 0.018
        target_xyz = np.array([desired_xy[0], desired_xy[1], 0.076])
        position_error = target_xyz - tool
        position_norm = float(np.linalg.norm(position_error))
        if position_norm > 0.018:
            position_error *= 0.018 / position_norm
        rotation = self.data.site_xmat[self._ids["tool_site"]].reshape(3, 3)
        normal_yaw = float(np.arctan2(rotation[1, 1], rotation[0, 1]))
        desired_yaw = float(np.arctan2(direction[1], direction[0]))
        yaw_error = (desired_yaw - normal_yaw + np.pi) % (2 * np.pi) - np.pi
        yaw_error = float(np.clip(yaw_error, -0.16, 0.16))
        jac_pos = np.zeros((3, self.model.nv))
        jac_rot = np.zeros((3, self.model.nv))
        mujoco.mj_jacSite(self.model, self.data, jac_pos, jac_rot, self._ids["tool_site"])
        weight = 0.18
        jacobian = np.vstack([jac_pos[:, :5], weight * jac_rot[2:3, :5]])
        error = np.concatenate([position_error, [weight * yaw_error]])
        delta = jacobian.T @ np.linalg.solve(jacobian @ jacobian.T + 1.5e-3 * np.eye(4), error)
        desired_q = self.data.qpos[:6].copy()
        desired_q[:5] += delta
        desired_q[5] = self.tool_gripper_joint_position_rad
        action = np.clip((desired_q - self.data.qpos[:6]) / self.config.max_joint_delta, -1.0, 1.0).astype(
            np.float32
        )
        confidence = float(np.exp(-5.0 * min(position_norm, 0.5)))
        return action, {
            "phase": phase,
            "teacher_confidence": confidence,
            "joint_target": desired_q.astype(np.float32),
        }

    def _path_intersects_obstacle(self, start: np.ndarray, end: np.ndarray) -> bool:
        direction = end - start
        denom = float(np.dot(direction, direction))
        if denom < 1e-8:
            return False
        t = np.clip(np.dot(self.obstacle_xy - start, direction) / denom, 0.0, 1.0)
        closest = start + t * direction
        return bool(np.all(np.abs(closest - self.obstacle_xy) <= np.array([0.075, 0.09])))

    def step(self, action: np.ndarray) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        action = np.asarray(action, dtype=np.float64)
        if action.shape != (6,) or not np.all(np.isfinite(action)):
            raise ValueError("Production action must be a finite six-joint vector")
        if self.estop:
            return self.observation(), -2.0, False, True, {"safety_stop": "estop", "success": False}
        requested = self.data.qpos[:6] + np.clip(action, -1.0, 1.0) * self.config.max_joint_delta
        target, safety = self._safety_filter(requested)
        self.data.ctrl[:] = target
        for _ in range(self.config.physics_substeps):
            self.data.qpos[:6] = target
            self.data.qvel[:6] = 0.0
            mujoco.mj_forward(self.model, self.data)
            mujoco.mj_step(self.model, self.data)
        self.data.qpos[:6] = target
        self.data.qvel[:6] = 0.0
        mujoco.mj_forward(self.model, self.data)
        self.step_count += 1
        distance = self.distance_to_target()
        progress = self.last_distance - distance
        self.last_distance = distance
        self.success_streak = self.success_streak + 1 if distance < self.config.success_radius else 0
        success = self.success_streak >= self.config.success_hold_steps
        out = not (0.06 <= self.block_xy()[0] <= 0.45 and -0.29 <= self.block_xy()[1] <= 0.29)
        terminated = success or out
        truncated = self.step_count >= self.config.max_steps
        reward = 28.0 * progress - 0.012 * float(np.square(action).sum()) - 0.015
        if success:
            reward += 10.0
        if out:
            reward -= 6.0
        if safety:
            reward -= 0.08
        info = {
            "success": success,
            "distance": distance,
            "progress": progress,
            "safety_clipped": bool(safety),
            "safety_reason": safety,
            "contact_count": self._tool_block_contacts(),
            "obstacle": self.obstacle_enabled,
        }
        return self.observation(), float(reward), terminated, truncated, info

    def _safety_filter(self, requested: np.ndarray) -> tuple[np.ndarray, str]:
        clipped = np.clip(requested, self.joint_ranges[:, 0], self.joint_ranges[:, 1])
        reasons: list[str] = []
        if not np.allclose(clipped, requested):
            reasons.append("joint_limit")
        trial = mujoco.MjData(self.model)
        trial.qpos[:] = self.data.qpos
        trial.qvel[:] = self.data.qvel

        def valid(q: np.ndarray) -> bool:
            trial.qpos[:6] = q
            mujoco.mj_forward(self.model, trial)
            xyz = trial.site_xpos[self._ids["tool_site"]]
            bounds = (self.config.workspace_x, self.config.workspace_y, self.config.workspace_z)
            return all(low <= value <= high for value, (low, high) in zip(xyz, bounds, strict=True))

        if not valid(clipped):
            current = self.data.qpos[:6].copy()
            for scale in (0.5, 0.25, 0.1, 0.05, 0.0):
                candidate = current + scale * (clipped - current)
                if valid(candidate):
                    clipped = candidate
                    break
            reasons.append("workspace_scaled")
        return clipped, "+".join(reasons)

    def observation(self) -> dict[str, np.ndarray]:
        return {
            "joint_state": np.concatenate([self.data.qpos[:6], self.data.qvel[:6]]).astype(np.float32),
            "task_vector": np.concatenate(
                [self.target_xy, self.obstacle_xy, [float(self.obstacle_enabled)]], dtype=np.float64
            ).astype(np.float32),
            "tool_pose": np.concatenate(
                [self.tool_xyz(), self.data.site_xmat[self._ids["tool_site"]].copy()]
            ).astype(np.float32),
        }

    def tool_xyz(self) -> np.ndarray:
        return self.data.site_xpos[self._ids["tool_site"]].copy()

    def block_xy(self) -> np.ndarray:
        return self.data.xpos[self._ids["block_body"], :2].copy()

    def distance_to_target(self) -> float:
        return float(np.linalg.norm(self.block_xy() - self.target_xy))

    def _tool_block_contacts(self) -> int:
        tool_geoms = frozenset(self._ids["tool_contact_geoms"])
        block = self._ids["block_geom"]
        return sum(
            1
            for index in range(self.data.ncon)
            if block
            in {
                int(self.data.contact[index].geom1),
                int(self.data.contact[index].geom2),
            }
            and bool(
                {
                    int(self.data.contact[index].geom1),
                    int(self.data.contact[index].geom2),
                }
                & tool_geoms
            )
        )

    def camera_calibration(self, width: int, height: int) -> dict[str, dict[str, Any]]:
        mujoco.mj_forward(self.model, self.data)
        result: dict[str, dict[str, Any]] = {}
        for name, camera_id in self._ids["cameras"].items():
            fovy = float(self.model.cam_fovy[camera_id])
            fy = 0.5 * height / np.tan(np.deg2rad(fovy) / 2.0)
            result[name] = {
                "width": width,
                "height": height,
                "intrinsics": [
                    [float(fy), 0.0, (width - 1) / 2.0],
                    [0.0, float(fy), (height - 1) / 2.0],
                    [0.0, 0.0, 1.0],
                ],
                "distortion_k1_k2_p1_p2_k3": [0.0, 0.0, 0.0, 0.0, 0.0],
                "world_position": self.data.cam_xpos[camera_id].copy().tolist(),
                "world_rotation": self.data.cam_xmat[camera_id].reshape(3, 3).copy().tolist(),
                "fovy_degrees": fovy,
                "mount_parent_body": "gripper",
                "mount_type": "rigid_wrist_camera",
                "mount_profile_version": WRIST_CAMERA_MOUNT_PROFILE_VERSION,
                "mount_parameter_source": WRIST_CAMERA_MOUNT_PARAMETER_SOURCE,
                "physically_calibrated": False,
                "tool_profile": self._ids["tool_profile"],
                "stock_gripper_direct_push": bool(
                    self._ids["tool_profile"] == STOCK_GRIPPER_DIRECT_PUSH_TOOL_PROFILE
                ),
                "local_position": self.model.cam_pos[camera_id].copy().tolist(),
                "local_quaternion_wxyz": self.model.cam_quat[camera_id].copy().tolist(),
            }
        return result

    @staticmethod
    def _unit(vector: np.ndarray) -> np.ndarray:
        vector = np.asarray(vector, dtype=np.float64)
        return vector / max(float(np.linalg.norm(vector)), 1e-9)
