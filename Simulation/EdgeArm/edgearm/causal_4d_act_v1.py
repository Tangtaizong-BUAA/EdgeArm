"""Causal wrist RGB-D ACT policy with explicit 4D geometric memory.

This module is deliberately a fixed block-push policy core, not a
language/colour-conditioned VLA.  It accepts current/past wrist observations,
reported robot state, and completed *reported* action history.
Simulator physical state, object/contact state, segmentation, same-row effect
execution, and teacher diagnostics are outside the input schema and are
rejected by :class:`Causal4DACTV1`.

MuJoCo cameras look along local ``-z`` with ``+x`` right and ``+y`` up.  Image
rows increase downwards, so wrist depth backprojection uses
``[x, y, z] = [(u-cx)d/fx, -(v-cy)d/fy, -d]`` before applying the recorded
camera-to-world rotation and translation.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from typing import Any, Mapping

import torch
import torch.nn.functional as F
from torch import nn


CAUSAL_4D_ACT_V1_ARCHITECTURE = "causal_4d_act_v1"
CAUSAL_4D_ACT_V1_CONFIG_VERSION = "1.0"
CAUSAL_4D_ACT_V1_TASK_SCOPE = "fixed_block_push_task_v1"

# This is the complete deployment input boundary.  Training labels and masks
# are intentionally absent.  Callers must select these fields from a loader
# sample before invoking the model; unexpected keys fail closed.
CAUSAL_4D_ACT_V1_INPUT_KEYS = frozenset(
    {
        "rgb_wrist_window",
        "depth_wrist_m_window",
        "depth_valid_mask_window",
        "camera_pose_wrist_window",
        "camera_intrinsics",
        "visual_history_mask",
        "camera_4d_reconstructable_mask_window",
        "camera_geometry_alignment_exact_window",
        "camera_device_time_delta_s_window",
        "camera_host_time_delta_s_window",
        "robot_state",
        "action_history",
        "action_history_mask",
    }
)

_INPUT_SCHEMA = {
    "action_history": "float[B,K,6], normalized reported executed deltas from completed rows only",
    "action_history_mask": "bool[B,K], valid completed reported-action rows",
    "camera_4d_reconstructable_mask_window": "bool[B,T], packet supports 4D reconstruction",
    "camera_device_time_delta_s_window": "float[B,T], causal time relative to current device frame",
    "camera_geometry_alignment_exact_window": "bool[B,T], exact RGB-D-pose alignment",
    "camera_host_time_delta_s_window": "float[B,T], causal time relative to current host frame",
    "camera_intrinsics": "float[B,3,3], wrist pinhole calibration in source-image pixels",
    "camera_pose_wrist_window": "float[B,T,12], xyz then row-major camera-to-world rotation",
    "depth_valid_mask_window": "bool[B,T,H,W], finite positive in-range depth",
    "depth_wrist_m_window": "float[B,T,H,W], wrist depth in metres",
    "rgb_wrist_window": "uint8[B,T,H,W,3] or float[B,T,H,W,3] in [0,1]",
    "robot_state": "float[B,12], current reported joint position then velocity",
    "visual_history_mask": "bool[B,T], current/past delivered wrist frames only",
}
CAUSAL_4D_ACT_V1_INPUT_SCHEMA_HASH = hashlib.sha256(
    json.dumps(_INPUT_SCHEMA, sort_keys=True, separators=(",", ":")).encode("utf-8")
).hexdigest()


@dataclass(frozen=True, slots=True)
class Causal4DACTConfig:
    """Moderate ACT+4D configuration intended to fit Apple-silicon MPS."""

    visual_history_steps: int = 8
    action_history_steps: int = 8
    action_chunk_size: int = 16
    action_dim: int = 6
    robot_state_dim: int = 12
    model_dim: int = 256
    attention_heads: int = 8
    memory_layers: int = 4
    decoder_layers: int = 4
    feedforward_dim: int = 1024
    spatial_grid_height: int = 4
    spatial_grid_width: int = 6
    dropout: float = 0.0
    max_depth_m: float = 5.0

    def __post_init__(self) -> None:
        integer_fields = (
            "visual_history_steps",
            "action_history_steps",
            "action_chunk_size",
            "action_dim",
            "robot_state_dim",
            "model_dim",
            "attention_heads",
            "memory_layers",
            "decoder_layers",
            "feedforward_dim",
            "spatial_grid_height",
            "spatial_grid_width",
        )
        for name in integer_fields:
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.action_dim != 6:
            raise ValueError("Causal4DACTV1 requires the six-joint EdgeArm action space")
        if self.robot_state_dim != 12:
            raise ValueError("Causal4DACTV1 requires reported q/dq as a 12-vector")
        if self.model_dim % self.attention_heads:
            raise ValueError("model_dim must be divisible by attention_heads")
        if not 0.0 <= float(self.dropout) < 1.0:
            raise ValueError("dropout must be in [0,1)")
        if not float(self.max_depth_m) > 0.0:
            raise ValueError("max_depth_m must be positive")

    def payload(self) -> dict[str, Any]:
        """Return a stable checkpoint-ready configuration payload."""

        return {
            "architecture": CAUSAL_4D_ACT_V1_ARCHITECTURE,
            "config_version": CAUSAL_4D_ACT_V1_CONFIG_VERSION,
            "input_schema_hash": CAUSAL_4D_ACT_V1_INPUT_SCHEMA_HASH,
            "task_scope": CAUSAL_4D_ACT_V1_TASK_SCOPE,
            "structured_task_conditioning": False,
            **asdict(self),
        }


def _require_shape(tensor: torch.Tensor, shape: tuple[int, ...], name: str) -> None:
    if tuple(tensor.shape) != shape:
        raise ValueError(f"{name} must have shape {shape}, got {tuple(tensor.shape)}")


def _require_bool(tensor: torch.Tensor, name: str) -> None:
    if tensor.dtype != torch.bool:
        raise ValueError(f"{name} must have boolean dtype")


def _require_finite(tensor: torch.Tensor, name: str) -> None:
    if not bool(torch.isfinite(tensor).all().item()):
        raise ValueError(f"{name} must be finite")


def _pixel_centres(
    *,
    source_height: int,
    source_width: int,
    grid_height: int,
    grid_width: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Return source-image pixel centres for an evenly downsampled grid."""

    rows = (torch.arange(grid_height, device=device, dtype=dtype) + 0.5) * (
        float(source_height) / float(grid_height)
    ) - 0.5
    columns = (torch.arange(grid_width, device=device, dtype=dtype) + 0.5) * (
        float(source_width) / float(grid_width)
    ) - 0.5
    row_grid, column_grid = torch.meshgrid(rows, columns, indexing="ij")
    return torch.stack((column_grid, row_grid), dim=-1)


def backproject_mujoco_wrist_depth(
    depth_m: torch.Tensor,
    camera_intrinsics: torch.Tensor,
    camera_pose_wrist: torch.Tensor,
    *,
    pixel_coordinates_uv: torch.Tensor | None = None,
) -> torch.Tensor:
    """Backproject wrist depth into world XYZ using MuJoCo camera axes.

    Args:
        depth_m: ``[B,T,H,W]`` positive metric depth.
        camera_intrinsics: ``[B,3,3]`` pinhole matrices in the coordinate
            system of ``pixel_coordinates_uv``.
        camera_pose_wrist: ``[B,T,12]``; translation followed by a row-major
            camera-to-world rotation matrix (MuJoCo ``cam_xmat``).
        pixel_coordinates_uv: Optional ``[H,W,2]`` source-image coordinates.
            If omitted, integer pixel centres of the depth tensor are used.

    Returns:
        World points with shape ``[B,T,H,W,3]``.
    """

    if depth_m.ndim != 4:
        raise ValueError("depth_m must have shape [B,T,H,W]")
    batch, steps, height, width = depth_m.shape
    _require_shape(camera_intrinsics, (batch, 3, 3), "camera_intrinsics")
    _require_shape(camera_pose_wrist, (batch, steps, 12), "camera_pose_wrist")
    if not depth_m.is_floating_point():
        raise ValueError("depth_m must have floating dtype")
    dtype = depth_m.dtype
    intrinsics = camera_intrinsics.to(device=depth_m.device, dtype=dtype)
    pose = camera_pose_wrist.to(device=depth_m.device, dtype=dtype)
    _require_finite(depth_m, "depth_m")
    _require_finite(intrinsics, "camera_intrinsics")
    _require_finite(pose, "camera_pose_wrist")

    focal_x = intrinsics[:, 0, 0]
    focal_y = intrinsics[:, 1, 1]
    if bool(((focal_x <= 0.0) | (focal_y <= 0.0)).any().item()):
        raise ValueError("camera focal lengths must be positive")

    if pixel_coordinates_uv is None:
        pixel_coordinates_uv = _pixel_centres(
            source_height=height,
            source_width=width,
            grid_height=height,
            grid_width=width,
            device=depth_m.device,
            dtype=dtype,
        )
    else:
        _require_shape(pixel_coordinates_uv, (height, width, 2), "pixel_coordinates_uv")
        pixel_coordinates_uv = pixel_coordinates_uv.to(device=depth_m.device, dtype=dtype)
        _require_finite(pixel_coordinates_uv, "pixel_coordinates_uv")

    u = pixel_coordinates_uv[..., 0].view(1, 1, height, width)
    v = pixel_coordinates_uv[..., 1].view(1, 1, height, width)
    cx = intrinsics[:, 0, 2].view(batch, 1, 1, 1)
    cy = intrinsics[:, 1, 2].view(batch, 1, 1, 1)
    fx = focal_x.view(batch, 1, 1, 1)
    fy = focal_y.view(batch, 1, 1, 1)

    camera_xyz = torch.stack(
        (
            (u - cx) * depth_m / fx,
            -(v - cy) * depth_m / fy,
            -depth_m,
        ),
        dim=-1,
    )
    translation = pose[..., :3].view(batch, steps, 1, 1, 3)
    world_rotation = pose[..., 3:].reshape(batch, steps, 3, 3)
    return translation + torch.einsum("btij,bthwj->bthwi", world_rotation, camera_xyz)


class Causal4DACTV1(nn.Module):
    """ACT action-chunk decoder over causal wrist-centric 4D memory."""

    def __init__(self, config: Causal4DACTConfig | None = None) -> None:
        super().__init__()
        self.config = config or Causal4DACTConfig()
        cfg = self.config
        hidden = max(16, cfg.model_dim // 4)
        middle = max(32, cfg.model_dim // 2)
        self.rgbd_encoder = nn.Sequential(
            nn.Conv2d(5, hidden, kernel_size=5, stride=2, padding=2),
            nn.GELU(),
            nn.Conv2d(hidden, middle, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(middle, cfg.model_dim, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
        )
        self.visual_norm = nn.LayerNorm(cfg.model_dim)
        # World xyz, device/host relative times, and exact-geometry validity.
        self.xyz_time_projection = nn.Sequential(
            nn.Linear(6, middle),
            nn.GELU(),
            nn.Linear(middle, cfg.model_dim),
        )
        spatial_tokens = cfg.spatial_grid_height * cfg.spatial_grid_width
        self.spatial_position = nn.Parameter(torch.randn(1, 1, spatial_tokens, cfg.model_dim) * 0.01)
        self.temporal_position = nn.Parameter(
            torch.randn(1, cfg.visual_history_steps, 1, cfg.model_dim) * 0.01
        )
        self.visual_modality = nn.Parameter(torch.randn(1, 1, 1, cfg.model_dim) * 0.01)
        self.visual_null_token = nn.Parameter(torch.randn(1, 1, cfg.model_dim) * 0.01)

        memory_layer = nn.TransformerEncoderLayer(
            d_model=cfg.model_dim,
            nhead=cfg.attention_heads,
            dim_feedforward=cfg.feedforward_dim,
            dropout=cfg.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.visual_memory = nn.TransformerEncoder(
            memory_layer,
            num_layers=cfg.memory_layers,
            enable_nested_tensor=False,
        )
        self.visual_memory_norm = nn.LayerNorm(cfg.model_dim)

        self.robot_projection = nn.Sequential(
            nn.LayerNorm(cfg.robot_state_dim),
            nn.Linear(cfg.robot_state_dim, cfg.model_dim),
            nn.GELU(),
            nn.Linear(cfg.model_dim, cfg.model_dim),
        )
        self.action_projection = nn.Sequential(
            nn.LayerNorm(cfg.action_dim),
            nn.Linear(cfg.action_dim, cfg.model_dim),
            nn.GELU(),
            nn.Linear(cfg.model_dim, cfg.model_dim),
        )
        self.action_position = nn.Parameter(torch.randn(1, cfg.action_history_steps, cfg.model_dim) * 0.01)
        self.robot_modality = nn.Parameter(torch.randn(1, 1, cfg.model_dim) * 0.01)
        self.action_modality = nn.Parameter(torch.randn(1, 1, cfg.model_dim) * 0.01)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=cfg.model_dim,
            nhead=cfg.attention_heads,
            dim_feedforward=cfg.feedforward_dim,
            dropout=cfg.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.action_decoder = nn.TransformerDecoder(decoder_layer, num_layers=cfg.decoder_layers)
        self.chunk_queries = nn.Parameter(torch.randn(1, cfg.action_chunk_size, cfg.model_dim) * 0.02)
        self.output_norm = nn.LayerNorm(cfg.model_dim)
        self.action_head = nn.Linear(cfg.model_dim, cfg.action_dim)

        frame_ids = torch.arange(cfg.visual_history_steps).repeat_interleave(spatial_tokens)
        frame_ids = torch.cat((torch.tensor([-1]), frame_ids))
        # A query may attend to the learned null token, its own frame, and
        # earlier frames, but never a later frame.
        visual_causal_mask = frame_ids.unsqueeze(0) > frame_ids.unsqueeze(1)
        self.register_buffer("visual_causal_mask", visual_causal_mask, persistent=False)

    @property
    def parameter_count(self) -> int:
        """Total parameters, including frozen parameters if added later."""

        return sum(parameter.numel() for parameter in self.parameters())

    def metadata(self) -> dict[str, Any]:
        """Return checkpoint metadata required to reproduce this core."""

        return {**self.config.payload(), "parameter_count": self.parameter_count}

    def _validate_inputs(self, inputs: Mapping[str, torch.Tensor]) -> int:
        if not isinstance(inputs, Mapping):
            raise TypeError("inputs must be a mapping of the explicit policy input schema")
        keys = frozenset(inputs)
        missing = sorted(CAUSAL_4D_ACT_V1_INPUT_KEYS - keys)
        unexpected = sorted(keys - CAUSAL_4D_ACT_V1_INPUT_KEYS)
        if missing or unexpected:
            raise ValueError(f"policy input schema mismatch: missing={missing}, unexpected={unexpected}")
        if not all(isinstance(value, torch.Tensor) for value in inputs.values()):
            raise TypeError("every policy input must be a torch.Tensor")

        cfg = self.config
        rgb = inputs["rgb_wrist_window"]
        if rgb.ndim != 5 or rgb.shape[-1] != 3:
            raise ValueError("rgb_wrist_window must have shape [B,T,H,W,3]")
        batch, steps, height, width, _ = rgb.shape
        if batch < 1 or height < 1 or width < 1:
            raise ValueError("rgb_wrist_window dimensions must be non-empty")
        if steps != cfg.visual_history_steps:
            raise ValueError(f"rgb_wrist_window history must be {cfg.visual_history_steps}, got {steps}")
        if rgb.dtype != torch.uint8:
            if not rgb.is_floating_point():
                raise ValueError("rgb_wrist_window must be uint8 or floating point")
            _require_finite(rgb, "rgb_wrist_window")
            if bool(((rgb < 0.0) | (rgb > 1.0)).any().item()):
                raise ValueError("floating rgb_wrist_window must be in [0,1]")

        _require_shape(inputs["depth_wrist_m_window"], (batch, steps, height, width), "depth_wrist_m_window")
        _require_shape(
            inputs["depth_valid_mask_window"], (batch, steps, height, width), "depth_valid_mask_window"
        )
        _require_bool(inputs["depth_valid_mask_window"], "depth_valid_mask_window")
        _require_shape(inputs["camera_pose_wrist_window"], (batch, steps, 12), "camera_pose_wrist_window")
        _require_shape(inputs["camera_intrinsics"], (batch, 3, 3), "camera_intrinsics")
        _require_shape(inputs["robot_state"], (batch, cfg.robot_state_dim), "robot_state")
        _require_shape(
            inputs["action_history"],
            (batch, cfg.action_history_steps, cfg.action_dim),
            "action_history",
        )
        _require_shape(
            inputs["action_history_mask"],
            (batch, cfg.action_history_steps),
            "action_history_mask",
        )
        _require_bool(inputs["action_history_mask"], "action_history_mask")

        frame_mask_names = (
            "visual_history_mask",
            "camera_4d_reconstructable_mask_window",
            "camera_geometry_alignment_exact_window",
        )
        for name in frame_mask_names:
            _require_shape(inputs[name], (batch, steps), name)
            _require_bool(inputs[name], name)
        for name in ("camera_device_time_delta_s_window", "camera_host_time_delta_s_window"):
            _require_shape(inputs[name], (batch, steps), name)

        visual_mask = inputs["visual_history_mask"]
        depth_valid = inputs["depth_valid_mask_window"]
        reconstructable = inputs["camera_4d_reconstructable_mask_window"]
        geometry_exact = inputs["camera_geometry_alignment_exact_window"]
        if bool((depth_valid & ~visual_mask[:, :, None, None]).any().item()):
            raise ValueError("depth_valid_mask_window cannot enable a padded visual frame")
        if bool(((reconstructable | geometry_exact) & ~visual_mask).any().item()):
            raise ValueError("4D/geometry masks cannot enable a padded visual frame")

        geometry_mask = visual_mask & reconstructable & geometry_exact
        depth = inputs["depth_wrist_m_window"]
        if not depth.is_floating_point():
            raise ValueError("depth_wrist_m_window must have floating dtype")
        # Depth from rolling-shutter or otherwise non-reconstructable frames is
        # deliberately outside the 4D input boundary even if the packet-level
        # depth-valid bitmap itself is true.
        consumed_depth_mask = depth_valid & geometry_mask[:, :, None, None]
        valid_depth_values = depth.masked_select(consumed_depth_mask)
        if valid_depth_values.numel():
            _require_finite(valid_depth_values, "valid wrist depth")
            if bool(((valid_depth_values <= 0.0) | (valid_depth_values > cfg.max_depth_m)).any().item()):
                raise ValueError("valid wrist depth must be in (0,max_depth_m]")

        _require_finite(inputs["camera_intrinsics"], "camera_intrinsics")
        focal = inputs["camera_intrinsics"][:, (0, 1), (0, 1)]
        if bool((focal <= 0.0).any().item()):
            raise ValueError("camera focal lengths must be positive")
        _require_finite(inputs["robot_state"], "robot_state")

        if bool(geometry_mask.any().item()):
            geometry_selector = geometry_mask[:, :, None].expand(batch, steps, 12)
            _require_finite(
                inputs["camera_pose_wrist_window"].masked_select(geometry_selector),
                "geometry-enabled camera pose",
            )
            for name in ("camera_device_time_delta_s_window", "camera_host_time_delta_s_window"):
                _require_finite(inputs[name].masked_select(geometry_mask), f"geometry-enabled {name}")

        valid_action_values = inputs["action_history"].masked_select(
            inputs["action_history_mask"][:, :, None].expand(batch, cfg.action_history_steps, cfg.action_dim)
        )
        if valid_action_values.numel():
            _require_finite(valid_action_values, "valid action_history")
            if bool(((valid_action_values < -1.0) | (valid_action_values > 1.0)).any().item()):
                raise ValueError("valid action_history must be normalized to [-1,1]")
        return batch

    def _grid_depth_and_valid(
        self,
        depth: torch.Tensor,
        valid: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cfg = self.config
        batch, steps, height, width = depth.shape
        depth_flat = depth.view(batch * steps, 1, height, width)
        valid_flat = valid.view(batch * steps, 1, height, width).to(dtype=depth.dtype)
        # ``AdaptiveAvgPool2d`` on MPS requires divisibility that the real
        # 240x320 wrist frames and a 4x6 grid do not satisfy.  Bilinear
        # align_corners=False sampling is deterministic, supports arbitrary
        # image sizes on MPS, and uses the same cell-centre convention as the
        # pixel coordinates passed to backprojection below.
        grid_size = (cfg.spatial_grid_height, cfg.spatial_grid_width)
        valid_fraction = F.interpolate(
            valid_flat,
            size=grid_size,
            mode="bilinear",
            align_corners=False,
        )
        depth_sum = F.interpolate(
            depth_flat * valid_flat,
            size=grid_size,
            mode="bilinear",
            align_corners=False,
        )
        grid_depth = depth_sum / valid_fraction.clamp_min(torch.finfo(depth.dtype).eps)
        grid_valid = valid_fraction > 0.0
        return (
            grid_depth.view(batch, steps, cfg.spatial_grid_height, cfg.spatial_grid_width),
            grid_valid.view(batch, steps, cfg.spatial_grid_height, cfg.spatial_grid_width),
        )

    def encode_4d_memory(
        self,
        inputs: Mapping[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode causal RGB-D into spatial-temporal tokens and padding mask.

        The returned sequence starts with one learned, always-valid null token.
        It prevents all-padded windows from producing undefined all-masked
        attention without leaking any padded observation into the model.
        """

        batch = self._validate_inputs(inputs)
        cfg = self.config
        rgb_raw = inputs["rgb_wrist_window"]
        model_dtype = self.chunk_queries.dtype
        rgb = rgb_raw.to(dtype=model_dtype)
        if rgb_raw.dtype == torch.uint8:
            rgb = rgb / 255.0
        visual_mask = inputs["visual_history_mask"]
        geometry_frame_mask = (
            visual_mask
            & inputs["camera_4d_reconstructable_mask_window"]
            & inputs["camera_geometry_alignment_exact_window"]
        )
        rgb = torch.where(visual_mask[:, :, None, None, None], rgb, torch.zeros_like(rgb))

        raw_depth = inputs["depth_wrist_m_window"].to(dtype=model_dtype)
        raw_depth_valid = inputs["depth_valid_mask_window"] & geometry_frame_mask[:, :, None, None]
        depth = torch.where(raw_depth_valid, raw_depth, torch.zeros_like(raw_depth))
        rgbd = torch.cat(
            (
                rgb.permute(0, 1, 4, 2, 3),
                (depth / cfg.max_depth_m).unsqueeze(2),
                raw_depth_valid.to(dtype=model_dtype).unsqueeze(2),
            ),
            dim=2,
        )
        _, _, _, height, width = rgbd.shape
        encoded = self.rgbd_encoder(rgbd.reshape(batch * cfg.visual_history_steps, 5, height, width))
        encoded = F.interpolate(
            encoded,
            size=(cfg.spatial_grid_height, cfg.spatial_grid_width),
            mode="bilinear",
            align_corners=False,
        )
        encoded = encoded.flatten(2).transpose(1, 2).view(batch, cfg.visual_history_steps, -1, cfg.model_dim)

        grid_depth, grid_valid = self._grid_depth_and_valid(depth, raw_depth_valid)
        pixels = _pixel_centres(
            source_height=height,
            source_width=width,
            grid_height=cfg.spatial_grid_height,
            grid_width=cfg.spatial_grid_width,
            device=depth.device,
            dtype=model_dtype,
        )
        sanitized_pose = torch.where(
            geometry_frame_mask[:, :, None],
            inputs["camera_pose_wrist_window"].to(dtype=model_dtype),
            torch.zeros_like(inputs["camera_pose_wrist_window"], dtype=model_dtype),
        )
        # Masked poses are irrelevant to the returned tokens, but an identity
        # matrix makes the deterministic backprojection helper well-defined.
        identity = torch.eye(3, device=depth.device, dtype=model_dtype).flatten()
        sanitized_pose[..., 3:] = torch.where(
            geometry_frame_mask[:, :, None],
            sanitized_pose[..., 3:],
            identity.view(1, 1, 9),
        )
        world_xyz = backproject_mujoco_wrist_depth(
            grid_depth,
            inputs["camera_intrinsics"].to(dtype=model_dtype),
            sanitized_pose,
            pixel_coordinates_uv=pixels,
        )
        point_valid = grid_valid & geometry_frame_mask[:, :, None, None]
        world_xyz = torch.where(point_valid[..., None], world_xyz, torch.zeros_like(world_xyz))
        device_time = torch.where(
            geometry_frame_mask,
            inputs["camera_device_time_delta_s_window"].to(dtype=model_dtype),
            torch.zeros_like(inputs["camera_device_time_delta_s_window"], dtype=model_dtype),
        )
        host_time = torch.where(
            geometry_frame_mask,
            inputs["camera_host_time_delta_s_window"].to(dtype=model_dtype),
            torch.zeros_like(inputs["camera_host_time_delta_s_window"], dtype=model_dtype),
        )
        grid_shape = (batch, cfg.visual_history_steps, cfg.spatial_grid_height, cfg.spatial_grid_width)
        geometry_features = torch.cat(
            (
                world_xyz,
                device_time[:, :, None, None, None].expand(*grid_shape, 1),
                host_time[:, :, None, None, None].expand(*grid_shape, 1),
                point_valid.to(dtype=model_dtype)[..., None],
            ),
            dim=-1,
        ).view(batch, cfg.visual_history_steps, -1, 6)

        tokens = self.visual_norm(encoded) + self.xyz_time_projection(geometry_features)
        tokens = tokens + self.spatial_position + self.temporal_position + self.visual_modality
        tokens = tokens.flatten(1, 2)
        null = self.visual_null_token.expand(batch, -1, -1)
        tokens = torch.cat((null, tokens), dim=1)

        spatial_tokens = cfg.spatial_grid_height * cfg.spatial_grid_width
        padding = ~visual_mask[:, :, None].expand(batch, cfg.visual_history_steps, spatial_tokens)
        padding = torch.cat(
            (torch.zeros(batch, 1, dtype=torch.bool, device=padding.device), padding.flatten(1)),
            dim=1,
        )
        memory = self.visual_memory(
            tokens,
            mask=self.visual_causal_mask,
            src_key_padding_mask=padding,
        )
        return self.visual_memory_norm(memory), padding

    def forward(self, inputs: Mapping[str, torch.Tensor]) -> torch.Tensor:
        """Predict a normalized ACT chunk with shape ``[B,chunk,6]``."""

        visual_memory, visual_padding = self.encode_4d_memory(inputs)
        batch = visual_memory.shape[0]
        cfg = self.config
        model_dtype = self.chunk_queries.dtype

        action_mask = inputs["action_history_mask"]
        action_history = torch.where(
            action_mask[:, :, None],
            inputs["action_history"].to(dtype=model_dtype),
            torch.zeros_like(inputs["action_history"], dtype=model_dtype),
        )
        action_tokens = self.action_projection(action_history) + self.action_position + self.action_modality
        robot_token = self.robot_projection(inputs["robot_state"].to(dtype=model_dtype)).unsqueeze(1)
        robot_token = robot_token + self.robot_modality
        memory = torch.cat((visual_memory, action_tokens, robot_token), dim=1)
        memory_padding = torch.cat(
            (
                visual_padding,
                ~action_mask,
                torch.zeros(batch, 1, dtype=torch.bool, device=visual_padding.device),
            ),
            dim=1,
        )

        queries = self.chunk_queries.expand(batch, cfg.action_chunk_size, cfg.model_dim)
        decoded = self.action_decoder(
            tgt=queries,
            memory=memory,
            memory_key_padding_mask=memory_padding,
        )
        return torch.tanh(self.action_head(self.output_norm(decoded)))


def masked_action_chunk_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Mean L1 ACT loss over valid chunk rows, with fail-closed validation."""

    if prediction.ndim != 3 or prediction.shape[-1] != 6:
        raise ValueError("prediction must have shape [B,C,6]")
    if target.shape != prediction.shape:
        raise ValueError("target must have the same [B,C,6] shape as prediction")
    if mask.shape != prediction.shape[:2]:
        raise ValueError("mask must have shape [B,C]")
    if mask.dtype != torch.bool:
        raise ValueError("mask must have boolean dtype")
    if prediction.shape[0] < 1:
        raise ValueError("action batch must be non-empty")
    if bool((~mask.any(dim=1)).any().item()):
        raise ValueError("every batch item must contain at least one valid action target")
    _require_finite(prediction, "prediction")
    _require_finite(target, "target")
    if bool(((target < -1.0) | (target > 1.0)).any().item()):
        raise ValueError("target actions must be normalized to [-1,1]")

    weights = mask.to(dtype=prediction.dtype).unsqueeze(-1)
    denominator = weights.sum() * prediction.shape[-1]
    return (torch.abs(prediction - target) * weights).sum() / denominator
