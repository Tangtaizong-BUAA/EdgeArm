"""Sparse causal 4D-VLA ACT core for the stock EdgeArm follower.

The deployment boundary contains only delivered wrist RGB-D packets, measured
camera geometry/timing, dynamic language tokens, reported servo state/history,
and completed reported actions.  Simulator segmentation, contacts, object
poses, rewards, expert paths, and future frames are deliberately absent.

The visual memory is a small-hardware implementation of three *structural*
ideas rather than a claim of kernel equivalence to a large language model:

* recent delivered frames keep all spatial tokens in a local attention window;
* older frames are compressed into channel-wise gated delta-memory states;
* a deployable current RGB-D/language/proprio query selects a few old fine
  blocks for exact-token recall.

This is inspired by Native Sparse Attention and Kimi Delta Attention, but uses
ordinary PyTorch operators and no DeepSeek/Kimi custom accelerator kernel.  It
trains ACT chunks for smoother supervision while deployment replans every
control cycle and executes only the first predicted six-joint action.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from typing import Any, Literal, Mapping

import torch
import torch.nn.functional as F
from torch import nn

from .causal_4d_act_v1 import backproject_mujoco_wrist_depth
from .trisource_contract_v26 import (
    TRISOURCE_SOURCE_TYPES_V26,
    trisource_contract_payload_v26,
)


SPARSE_4D_VLA_ACT_ARCHITECTURE_V26 = "sparse_4d_vla_act_v26"
SPARSE_4D_VLA_ACT_CONFIG_VERSION_V26 = "1.0"
Sparse4DVLAScaleV26 = Literal["pilot", "base_40m"]
SPARSE_4D_VLA_ACT_INPUT_KEYS_V26 = frozenset(
    {
        "language_token_ids",
        "language_attention_mask",
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
        "joint_history",
        "action_history",
        "proprio_history_mask",
    }
)

_INPUT_SCHEMA_V26 = {
    "action_history": "float[B,K,6], normalized completed reported joint actions only",
    "camera_4d_reconstructable_mask_window": "bool[B,T], measured packet 4D eligibility",
    "camera_device_time_delta_s_window": "float[B,T], causal relative device time, current=0",
    "camera_geometry_alignment_exact_window": "bool[B,T], calibrated aligned RGB-D-pose",
    "camera_host_time_delta_s_window": "float[B,T], causal relative host time, current=0",
    "camera_intrinsics": "float[B,3,3], measured pinhole intrinsics",
    "camera_pose_wrist_window": "float[B,T,12], measured camera-to-world pose",
    "depth_valid_mask_window": "bool[B,T,H,W], delivered valid metric depth",
    "depth_wrist_m_window": "float[B,T,H,W], delivered metric wrist depth",
    "joint_history": "float[B,K,12], past/current reported q and dq",
    "language_attention_mask": "bool[B,L], every non-padding instruction token retained",
    "language_token_ids": "int64[B,L], dynamic task instruction tokens",
    "proprio_history_mask": "bool[B,K], causal reported joint/action history validity",
    "rgb_wrist_window": "uint8[B,T,H,W,3] or float in [0,1], current and past only",
    "robot_state": "float[B,12], current reported servo q and dq",
    "visual_history_mask": "bool[B,T], delivered wrist frames ending at current frame",
}
SPARSE_4D_VLA_ACT_INPUT_SCHEMA_HASH_V26 = hashlib.sha256(
    json.dumps(_INPUT_SCHEMA_V26, sort_keys=True, separators=(",", ":")).encode("utf-8")
).hexdigest()


@dataclass(frozen=True, slots=True)
class Sparse4DVLAConfigV26:
    visual_memory_mode: str = "recent"
    visual_history_steps: int = 24
    local_visual_steps: int = 4
    old_block_steps: int = 4
    selected_old_blocks: int = 2
    proprio_history_steps: int = 16
    language_max_tokens: int = 96
    language_vocabulary_size: int = 259
    action_chunk_size: int = 16
    action_dim: int = 6
    robot_state_dim: int = 12
    model_dim: int = 256
    attention_heads: int = 8
    local_attention_layers: int = 2
    language_layers: int = 1
    decoder_layers: int = 4
    feedforward_dim: int = 1_024
    spatial_grid_height: int = 4
    spatial_grid_width: int = 6
    encoder_image_height: int = 120
    encoder_image_width: int = 160
    dropout: float = 0.0
    max_depth_m: float = 5.0
    action_head_space: str = "relative_command"

    def __post_init__(self) -> None:
        if self.action_head_space not in ("relative_command", "applied_target_delta_v56", "absolute_joint_target_v57"):
            raise ValueError("unknown learned action reference")
        if self.visual_memory_mode not in ("recent", "episode_anchors_v54"):
            raise ValueError("unknown causal visual memory sampling")
        integers = (
            "visual_history_steps",
            "local_visual_steps",
            "old_block_steps",
            "selected_old_blocks",
            "proprio_history_steps",
            "language_max_tokens",
            "language_vocabulary_size",
            "action_chunk_size",
            "action_dim",
            "robot_state_dim",
            "model_dim",
            "attention_heads",
            "local_attention_layers",
            "language_layers",
            "decoder_layers",
            "feedforward_dim",
            "spatial_grid_height",
            "spatial_grid_width",
            "encoder_image_height",
            "encoder_image_width",
        )
        for name in integers:
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.action_dim != 6 or self.robot_state_dim != 12:
            raise ValueError("V26 requires six actions and 12 reported q/dq values")
        if self.local_visual_steps >= self.visual_history_steps:
            raise ValueError("V26 sparse memory requires at least one older visual block")
        older = self.visual_history_steps - self.local_visual_steps
        if older % self.old_block_steps:
            raise ValueError("V26 older visual steps must divide exactly into blocks")
        if self.selected_old_blocks > older // self.old_block_steps:
            raise ValueError("V26 selected old blocks exceed the available blocks")
        if self.model_dim % self.attention_heads:
            raise ValueError("V26 model_dim must be divisible by attention_heads")
        if not 0.0 <= float(self.dropout) < 1.0:
            raise ValueError("V26 dropout must be in [0,1)")
        if not math.isfinite(float(self.max_depth_m)) or self.max_depth_m <= 0.0:
            raise ValueError("V26 max_depth_m must be finite and positive")

    @property
    def older_visual_steps(self) -> int:
        return self.visual_history_steps - self.local_visual_steps

    @property
    def old_block_count(self) -> int:
        return self.older_visual_steps // self.old_block_steps

    @property
    def spatial_token_count(self) -> int:
        return self.spatial_grid_height * self.spatial_grid_width

    def payload(self) -> dict[str, Any]:
        return {
            "architecture": SPARSE_4D_VLA_ACT_ARCHITECTURE_V26,
            "config_version": SPARSE_4D_VLA_ACT_CONFIG_VERSION_V26,
            "input_schema_hash": SPARSE_4D_VLA_ACT_INPUT_SCHEMA_HASH_V26,
            "source_type_compatibility": list(TRISOURCE_SOURCE_TYPES_V26),
            "trisource_contract": dict(trisource_contract_payload_v26()),
            "attention_contract": {
                "recent_fine_local_attention": True,
                "older_channelwise_delta_memory": True,
                "deployable_query_selected_old_fine_blocks": True,
                "all_nonpadding_language_tokens_retained": True,
                "native_sparse_attention_kernel_equivalence_claimed": False,
                "kimi_delta_attention_kernel_equivalence_claimed": False,
                "custom_accelerator_kernel_present": False,
            },
            "language_tokenizer": "utf8-byte-v1",
            "deployment_contract": {
                "train_action_chunks": True,
                "replan_every_control_cycle": True,
                "execute_only_first_action": True,
                "segmentation_policy_input": False,
                "simulator_privileged_input": False,
            },
            **asdict(self),
        }


def sparse_4d_vla_config_preset_v26(
    scale: Sparse4DVLAScaleV26,
) -> Sparse4DVLAConfigV26:
    """Return an audited capacity tier without implying a training gate passed.

    ``pilot`` is intentionally small enough to validate data semantics and
    learning curves locally. ``base_40m`` enters the preregistered 30M-60M
    capacity band, but callers must still prove sufficient unique admitted
    data and a capacity-limited pilot before spending the larger run.
    """

    if scale == "pilot":
        return Sparse4DVLAConfigV26()
    if scale == "base_40m":
        return Sparse4DVLAConfigV26(
            visual_history_steps=32,
            local_visual_steps=4,
            old_block_steps=4,
            selected_old_blocks=3,
            proprio_history_steps=24,
            model_dim=512,
            attention_heads=8,
            local_attention_layers=3,
            language_layers=2,
            decoder_layers=5,
            feedforward_dim=2_048,
            spatial_grid_height=5,
            spatial_grid_width=8,
        )
    raise ValueError("V26 sparse 4D-VLA scale must be pilot or base_40m")


def _require_shape(value: torch.Tensor, expected: tuple[int, ...], name: str) -> None:
    if tuple(value.shape) != expected:
        raise ValueError(f"{name} must have shape {expected}, got {tuple(value.shape)}")


def _require_bool(value: torch.Tensor, name: str) -> None:
    if value.dtype != torch.bool:
        raise ValueError(f"{name} must have boolean dtype")


def _require_finite(value: torch.Tensor, name: str) -> None:
    if not bool(torch.isfinite(value).all().item()):
        raise ValueError(f"{name} must be finite")


def _grid_pixel_centres_v26(
    height: int,
    width: int,
    grid_height: int,
    grid_width: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    rows = (torch.arange(grid_height, device=device, dtype=dtype) + 0.5) * (
        float(height) / float(grid_height)
    ) - 0.5
    columns = (torch.arange(grid_width, device=device, dtype=dtype) + 0.5) * (
        float(width) / float(grid_width)
    ) - 0.5
    row_grid, column_grid = torch.meshgrid(rows, columns, indexing="ij")
    return torch.stack((column_grid, row_grid), dim=-1)


class ChannelwiseDeltaMemoryV26(nn.Module):
    """Finite-state old-frame compressor with per-channel gates.

    This follows the channel-wise finite-memory motivation of KDA, but it is a
    deliberately small ordinary-PyTorch recurrence, not the KDA DPLR kernel.
    """

    def __init__(self, model_dim: int) -> None:
        super().__init__()
        self.input_norm = nn.LayerNorm(model_dim)
        self.forget_gate = nn.Linear(model_dim, model_dim)
        self.delta_gate = nn.Linear(model_dim, model_dim)
        self.candidate = nn.Linear(model_dim, model_dim)
        self.output_norm = nn.LayerNorm(model_dim)

    def forward(
        self,
        block_summaries: torch.Tensor,
        block_valid: torch.Tensor,
    ) -> torch.Tensor:
        if block_summaries.ndim != 3:
            raise ValueError("V26 delta memory expects [B,N,D] block summaries")
        if block_valid.shape != block_summaries.shape[:2] or block_valid.dtype != torch.bool:
            raise ValueError("V26 delta memory block validity is misaligned")
        batch, blocks, dimension = block_summaries.shape
        state = torch.zeros(
            batch,
            dimension,
            dtype=block_summaries.dtype,
            device=block_summaries.device,
        )
        states: list[torch.Tensor] = []
        for block in range(blocks):
            normalized = self.input_norm(block_summaries[:, block])
            forget = torch.sigmoid(self.forget_gate(normalized))
            delta = torch.sigmoid(self.delta_gate(normalized))
            candidate = torch.tanh(self.candidate(normalized))
            updated = forget * state + (1.0 - forget) * ((1.0 - delta) * state + delta * candidate)
            state = torch.where(block_valid[:, block, None], updated, state)
            states.append(self.output_norm(state))
        return torch.stack(states, dim=1)


class Sparse4DVLAACTV26(nn.Module):
    """Dynamic-language sparse 4D ACT model with next-action deployment."""

    def __init__(self, config: Sparse4DVLAConfigV26 | None = None) -> None:
        super().__init__()
        self.config = config or Sparse4DVLAConfigV26()
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
        self.xyz_time_projection = nn.Sequential(
            nn.Linear(6, middle),
            nn.GELU(),
            nn.Linear(middle, cfg.model_dim),
        )
        self.spatial_position = nn.Parameter(torch.randn(1, 1, cfg.spatial_token_count, cfg.model_dim) * 0.01)
        self.temporal_position = nn.Parameter(
            torch.randn(1, cfg.visual_history_steps, 1, cfg.model_dim) * 0.01
        )
        self.visual_modality = nn.Parameter(torch.randn(1, 1, 1, cfg.model_dim) * 0.01)
        self.local_null_token = nn.Parameter(torch.randn(1, 1, cfg.model_dim) * 0.01)
        self.global_null_token = nn.Parameter(torch.randn(1, 1, cfg.model_dim) * 0.01)

        local_layer = nn.TransformerEncoderLayer(
            d_model=cfg.model_dim,
            nhead=cfg.attention_heads,
            dim_feedforward=cfg.feedforward_dim,
            dropout=cfg.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.local_memory = nn.TransformerEncoder(
            local_layer,
            num_layers=cfg.local_attention_layers,
            enable_nested_tensor=False,
        )
        self.local_memory_norm = nn.LayerNorm(cfg.model_dim)
        self.delta_memory = ChannelwiseDeltaMemoryV26(cfg.model_dim)
        self.compressed_block_position = nn.Parameter(
            torch.randn(1, cfg.old_block_count, cfg.model_dim) * 0.01
        )
        self.selector_query_norm = nn.LayerNorm(cfg.model_dim)
        self.selector_key = nn.Linear(cfg.model_dim, cfg.model_dim, bias=False)

        self.language_embedding = nn.Embedding(
            cfg.language_vocabulary_size,
            cfg.model_dim,
            padding_idx=0,
        )
        self.language_position = nn.Parameter(torch.randn(1, cfg.language_max_tokens, cfg.model_dim) * 0.01)
        self.language_modality = nn.Parameter(torch.randn(1, 1, cfg.model_dim) * 0.01)
        language_layer = nn.TransformerEncoderLayer(
            d_model=cfg.model_dim,
            nhead=cfg.attention_heads,
            dim_feedforward=cfg.feedforward_dim,
            dropout=cfg.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.language_encoder = nn.TransformerEncoder(
            language_layer,
            num_layers=cfg.language_layers,
            enable_nested_tensor=False,
        )
        self.language_norm = nn.LayerNorm(cfg.model_dim)

        self.robot_projection = nn.Sequential(
            nn.LayerNorm(cfg.robot_state_dim),
            nn.Linear(cfg.robot_state_dim, cfg.model_dim),
            nn.GELU(),
            nn.Linear(cfg.model_dim, cfg.model_dim),
        )
        self.proprio_projection = nn.Sequential(
            nn.LayerNorm(cfg.robot_state_dim + cfg.action_dim),
            nn.Linear(cfg.robot_state_dim + cfg.action_dim, cfg.model_dim),
            nn.GELU(),
            nn.Linear(cfg.model_dim, cfg.model_dim),
        )
        self.proprio_position = nn.Parameter(torch.randn(1, cfg.proprio_history_steps, cfg.model_dim) * 0.01)
        self.robot_modality = nn.Parameter(torch.randn(1, 1, cfg.model_dim) * 0.01)
        self.proprio_modality = nn.Parameter(torch.randn(1, 1, cfg.model_dim) * 0.01)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=cfg.model_dim,
            nhead=cfg.attention_heads,
            dim_feedforward=cfg.feedforward_dim,
            dropout=cfg.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.action_decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=cfg.decoder_layers,
        )
        self.action_queries = nn.Parameter(torch.randn(1, cfg.action_chunk_size, cfg.model_dim) * 0.02)
        self.action_norm = nn.LayerNorm(cfg.model_dim)
        self.action_head = nn.Linear(cfg.model_dim, cfg.action_dim)

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def metadata(self) -> dict[str, Any]:
        return {**self.config.payload(), "parameter_count": self.parameter_count}

    def _validate_inputs(self, inputs: Mapping[str, torch.Tensor]) -> tuple[int, int, int]:
        if not isinstance(inputs, Mapping):
            raise TypeError("V26 policy inputs must be a mapping")
        missing = sorted(SPARSE_4D_VLA_ACT_INPUT_KEYS_V26 - frozenset(inputs))
        unexpected = sorted(frozenset(inputs) - SPARSE_4D_VLA_ACT_INPUT_KEYS_V26)
        if missing or unexpected:
            raise ValueError(f"V26 policy input schema mismatch: missing={missing}, unexpected={unexpected}")
        if not all(isinstance(value, torch.Tensor) for value in inputs.values()):
            raise TypeError("V26 every policy input must be a torch.Tensor")

        cfg = self.config
        rgb = inputs["rgb_wrist_window"]
        if rgb.ndim != 5 or rgb.shape[-1] != 3:
            raise ValueError("V26 rgb_wrist_window must have shape [B,T,H,W,3]")
        batch, steps, height, width, _ = rgb.shape
        if batch < 1 or height < 1 or width < 1 or steps != cfg.visual_history_steps:
            raise ValueError("V26 wrist RGB dimensions/history are invalid")
        if rgb.dtype != torch.uint8:
            if not rgb.is_floating_point():
                raise ValueError("V26 wrist RGB must be uint8 or floating point")
            _require_finite(rgb, "rgb_wrist_window")
            if bool(((rgb < 0.0) | (rgb > 1.0)).any().item()):
                raise ValueError("V26 floating wrist RGB must stay in [0,1]")

        visual_mask = inputs["visual_history_mask"]
        _require_shape(visual_mask, (batch, steps), "visual_history_mask")
        _require_bool(visual_mask, "visual_history_mask")
        if not bool(visual_mask[:, -1].all().item()):
            raise ValueError("V26 every sample requires a delivered current wrist frame")
        for name in (
            "camera_4d_reconstructable_mask_window",
            "camera_geometry_alignment_exact_window",
        ):
            _require_shape(inputs[name], (batch, steps), name)
            _require_bool(inputs[name], name)
            if bool((inputs[name] & ~visual_mask).any().item()):
                raise ValueError("V26 camera geometry masks cannot enable padded frames")
        _require_shape(inputs["depth_wrist_m_window"], (batch, steps, height, width), "depth_wrist_m_window")
        _require_shape(
            inputs["depth_valid_mask_window"], (batch, steps, height, width), "depth_valid_mask_window"
        )
        _require_bool(inputs["depth_valid_mask_window"], "depth_valid_mask_window")
        if bool((inputs["depth_valid_mask_window"] & ~visual_mask[:, :, None, None]).any().item()):
            raise ValueError("V26 valid depth cannot enable a padded frame")
        _require_shape(inputs["camera_pose_wrist_window"], (batch, steps, 12), "camera_pose_wrist_window")
        _require_shape(inputs["camera_intrinsics"], (batch, 3, 3), "camera_intrinsics")
        _require_finite(inputs["camera_intrinsics"], "camera_intrinsics")
        if bool((inputs["camera_intrinsics"][:, (0, 1), (0, 1)] <= 0.0).any().item()):
            raise ValueError("V26 camera focal lengths must be positive")

        geometry = (
            visual_mask
            & inputs["camera_4d_reconstructable_mask_window"]
            & inputs["camera_geometry_alignment_exact_window"]
        )
        consumed_depth = inputs["depth_valid_mask_window"] & geometry[:, :, None, None]
        depth_values = inputs["depth_wrist_m_window"].masked_select(consumed_depth)
        if depth_values.numel():
            _require_finite(depth_values, "valid V26 depth")
            if bool(((depth_values <= 0.0) | (depth_values > cfg.max_depth_m)).any().item()):
                raise ValueError("V26 valid depth must be in (0,max_depth_m]")
        if bool(geometry.any().item()):
            pose_selector = geometry[:, :, None].expand(batch, steps, 12)
            _require_finite(
                inputs["camera_pose_wrist_window"].masked_select(pose_selector),
                "geometry-enabled camera pose",
            )
        for name in (
            "camera_device_time_delta_s_window",
            "camera_host_time_delta_s_window",
        ):
            _require_shape(inputs[name], (batch, steps), name)
            times = inputs[name].masked_select(visual_mask)
            _require_finite(times, name)
            if bool((times > 1.0e-5).any().item()):
                raise ValueError("V26 relative camera times cannot be in the future")
            if not bool(
                torch.allclose(inputs[name][:, -1], torch.zeros_like(inputs[name][:, -1]), atol=1e-5)
            ):
                raise ValueError("V26 current camera relative time must equal zero")

        tokens = inputs["language_token_ids"]
        language_mask = inputs["language_attention_mask"]
        _require_shape(tokens, (batch, cfg.language_max_tokens), "language_token_ids")
        _require_shape(language_mask, (batch, cfg.language_max_tokens), "language_attention_mask")
        if tokens.dtype != torch.int64:
            raise ValueError("V26 language token ids must be int64")
        _require_bool(language_mask, "language_attention_mask")
        if bool((~language_mask.any(dim=1)).any().item()):
            raise ValueError("V26 every sample requires at least one instruction token")
        if bool(((tokens < 0) | (tokens >= cfg.language_vocabulary_size)).any().item()):
            raise ValueError("V26 language token id is outside the vocabulary")

        _require_shape(inputs["robot_state"], (batch, cfg.robot_state_dim), "robot_state")
        _require_shape(
            inputs["joint_history"],
            (batch, cfg.proprio_history_steps, cfg.robot_state_dim),
            "joint_history",
        )
        _require_shape(
            inputs["action_history"],
            (batch, cfg.proprio_history_steps, cfg.action_dim),
            "action_history",
        )
        _require_shape(
            inputs["proprio_history_mask"],
            (batch, cfg.proprio_history_steps),
            "proprio_history_mask",
        )
        _require_bool(inputs["proprio_history_mask"], "proprio_history_mask")
        _require_finite(inputs["robot_state"], "robot_state")
        proprio_mask = inputs["proprio_history_mask"]
        joint_values = inputs["joint_history"].masked_select(
            proprio_mask[:, :, None].expand_as(inputs["joint_history"])
        )
        action_values = inputs["action_history"].masked_select(
            proprio_mask[:, :, None].expand_as(inputs["action_history"])
        )
        if joint_values.numel():
            _require_finite(joint_values, "valid joint_history")
        if action_values.numel():
            _require_finite(action_values, "valid action_history")
            if bool(((action_values < -1.0) | (action_values > 1.0)).any().item()):
                raise ValueError("V26 valid action history must stay in [-1,1]")
        return batch, height, width

    def _frame_tokens(
        self,
        inputs: Mapping[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, height, width = self._validate_inputs(inputs)
        cfg = self.config
        dtype = self.action_queries.dtype
        visual_mask = inputs["visual_history_mask"]
        geometry = (
            visual_mask
            & inputs["camera_4d_reconstructable_mask_window"]
            & inputs["camera_geometry_alignment_exact_window"]
        )
        rgb_raw = inputs["rgb_wrist_window"]
        rgb = rgb_raw.to(dtype=dtype)
        if rgb_raw.dtype == torch.uint8:
            rgb = rgb / 255.0
        rgb = torch.where(visual_mask[:, :, None, None, None], rgb, torch.zeros_like(rgb))
        depth_valid = inputs["depth_valid_mask_window"] & geometry[:, :, None, None]
        depth = torch.where(
            depth_valid,
            inputs["depth_wrist_m_window"].to(dtype=dtype),
            torch.zeros_like(inputs["depth_wrist_m_window"], dtype=dtype),
        )
        rgbd = torch.cat(
            (
                rgb.permute(0, 1, 4, 2, 3),
                (depth / cfg.max_depth_m).unsqueeze(2),
                depth_valid.to(dtype=dtype).unsqueeze(2),
            ),
            dim=2,
        )
        encoder_input = F.interpolate(
            rgbd.reshape(batch * cfg.visual_history_steps, 5, height, width),
            size=(cfg.encoder_image_height, cfg.encoder_image_width),
            mode="bilinear",
            align_corners=False,
        )
        encoded = self.rgbd_encoder(encoder_input)
        encoded = F.interpolate(
            encoded,
            size=(cfg.spatial_grid_height, cfg.spatial_grid_width),
            mode="bilinear",
            align_corners=False,
        )
        encoded = (
            encoded.flatten(2)
            .transpose(1, 2)
            .view(
                batch,
                cfg.visual_history_steps,
                cfg.spatial_token_count,
                cfg.model_dim,
            )
        )

        flattened_depth = depth.view(batch * cfg.visual_history_steps, 1, height, width)
        flattened_valid = depth_valid.view(batch * cfg.visual_history_steps, 1, height, width).to(dtype=dtype)
        grid_size = (cfg.spatial_grid_height, cfg.spatial_grid_width)
        valid_fraction = F.interpolate(
            flattened_valid,
            size=grid_size,
            mode="bilinear",
            align_corners=False,
        )
        depth_sum = F.interpolate(
            flattened_depth * flattened_valid,
            size=grid_size,
            mode="bilinear",
            align_corners=False,
        )
        grid_depth = (depth_sum / valid_fraction.clamp_min(torch.finfo(dtype).eps)).view(
            batch,
            cfg.visual_history_steps,
            cfg.spatial_grid_height,
            cfg.spatial_grid_width,
        )
        grid_valid = (valid_fraction > 0.0).view(
            batch,
            cfg.visual_history_steps,
            cfg.spatial_grid_height,
            cfg.spatial_grid_width,
        )
        pixels = _grid_pixel_centres_v26(
            height,
            width,
            cfg.spatial_grid_height,
            cfg.spatial_grid_width,
            device=grid_depth.device,
            dtype=dtype,
        )
        pose = torch.where(
            geometry[:, :, None],
            inputs["camera_pose_wrist_window"].to(dtype=dtype),
            torch.zeros_like(inputs["camera_pose_wrist_window"], dtype=dtype),
        )
        identity = torch.eye(3, dtype=dtype, device=pose.device).flatten()
        pose[..., 3:] = torch.where(
            geometry[:, :, None],
            pose[..., 3:],
            identity.view(1, 1, 9),
        )
        world_xyz = backproject_mujoco_wrist_depth(
            grid_depth,
            inputs["camera_intrinsics"].to(dtype=dtype),
            pose,
            pixel_coordinates_uv=pixels,
        )
        point_valid = grid_valid & geometry[:, :, None, None]
        world_xyz = torch.where(point_valid[..., None], world_xyz, torch.zeros_like(world_xyz))
        device_time = torch.where(
            geometry,
            inputs["camera_device_time_delta_s_window"].to(dtype=dtype),
            torch.zeros_like(inputs["camera_device_time_delta_s_window"], dtype=dtype),
        )
        host_time = torch.where(
            geometry,
            inputs["camera_host_time_delta_s_window"].to(dtype=dtype),
            torch.zeros_like(inputs["camera_host_time_delta_s_window"], dtype=dtype),
        )
        grid_shape = (
            batch,
            cfg.visual_history_steps,
            cfg.spatial_grid_height,
            cfg.spatial_grid_width,
        )
        geometry_features = torch.cat(
            (
                world_xyz,
                device_time[:, :, None, None, None].expand(*grid_shape, 1),
                host_time[:, :, None, None, None].expand(*grid_shape, 1),
                point_valid.to(dtype=dtype)[..., None],
            ),
            dim=-1,
        ).view(batch, cfg.visual_history_steps, cfg.spatial_token_count, 6)
        tokens = self.visual_norm(encoded) + self.xyz_time_projection(geometry_features)
        tokens = tokens + self.spatial_position + self.temporal_position + self.visual_modality
        padding = ~visual_mask[:, :, None].expand(
            batch,
            cfg.visual_history_steps,
            cfg.spatial_token_count,
        )
        return tokens, padding

    def encode_sparse_memory(
        self,
        inputs: Mapping[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor | int]]:
        frame_tokens, frame_padding = self._frame_tokens(inputs)
        cfg = self.config
        batch = frame_tokens.shape[0]
        dimension = cfg.model_dim
        block_fine_count = cfg.old_block_steps * cfg.spatial_token_count

        language_mask = inputs["language_attention_mask"]
        language = (
            self.language_embedding(inputs["language_token_ids"])
            + self.language_position
            + self.language_modality
        )
        language = self.language_encoder(language, src_key_padding_mask=~language_mask)
        language = self.language_norm(language)
        language_summary = (language * language_mask[:, :, None].to(dtype=language.dtype)).sum(
            dim=1
        ) / language_mask.sum(dim=1, keepdim=True).clamp_min(1).to(dtype=language.dtype)

        old_tokens = frame_tokens[:, : cfg.older_visual_steps]
        old_padding = frame_padding[:, : cfg.older_visual_steps]
        block_tokens = old_tokens.reshape(
            batch,
            cfg.old_block_count,
            block_fine_count,
            dimension,
        )
        block_padding = old_padding.reshape(
            batch,
            cfg.old_block_count,
            block_fine_count,
        )
        block_valid = ~block_padding.all(dim=-1)
        block_weights = (~block_padding).to(dtype=frame_tokens.dtype)
        block_summaries = (block_tokens * block_weights[..., None]).sum(dim=2) / block_weights.sum(
            dim=2, keepdim=True
        ).clamp_min(1.0)
        compressed = self.delta_memory(block_summaries, block_valid)
        compressed = compressed + self.compressed_block_position

        current_summary = frame_tokens[:, -1].mean(dim=1)
        robot = self.robot_projection(inputs["robot_state"].to(dtype=frame_tokens.dtype))
        routing_query = self.selector_query_norm(current_summary + language_summary + robot)
        routing_scores = torch.einsum(
            "bd,bnd->bn",
            routing_query,
            self.selector_key(block_summaries),
        ) / math.sqrt(float(dimension))
        routing_scores = routing_scores.masked_fill(~block_valid, torch.finfo(routing_scores.dtype).min)
        selected_scores, selected_indices = torch.topk(
            routing_scores,
            k=cfg.selected_old_blocks,
            dim=1,
            largest=True,
            sorted=True,
        )
        selected_tokens = torch.gather(
            block_tokens,
            1,
            selected_indices[:, :, None, None].expand(
                batch,
                cfg.selected_old_blocks,
                block_fine_count,
                dimension,
            ),
        ).reshape(batch, cfg.selected_old_blocks * block_fine_count, dimension)
        selected_padding = torch.gather(
            block_padding,
            1,
            selected_indices[:, :, None].expand(
                batch,
                cfg.selected_old_blocks,
                block_fine_count,
            ),
        ).reshape(batch, cfg.selected_old_blocks * block_fine_count)

        local_tokens = frame_tokens[:, -cfg.local_visual_steps :].flatten(1, 2)
        local_padding = frame_padding[:, -cfg.local_visual_steps :].flatten(1, 2)
        local_tokens = torch.cat(
            (self.local_null_token.expand(batch, -1, -1), local_tokens),
            dim=1,
        )
        local_padding = torch.cat(
            (
                torch.zeros(batch, 1, dtype=torch.bool, device=local_padding.device),
                local_padding,
            ),
            dim=1,
        )
        local_tokens = self.local_memory(
            local_tokens,
            src_key_padding_mask=local_padding,
        )
        local_tokens = self.local_memory_norm(local_tokens)

        proprio_mask = inputs["proprio_history_mask"]
        joint_history = torch.where(
            proprio_mask[:, :, None],
            inputs["joint_history"].to(dtype=frame_tokens.dtype),
            torch.zeros_like(inputs["joint_history"], dtype=frame_tokens.dtype),
        )
        action_history = torch.where(
            proprio_mask[:, :, None],
            inputs["action_history"].to(dtype=frame_tokens.dtype),
            torch.zeros_like(inputs["action_history"], dtype=frame_tokens.dtype),
        )
        proprio = self.proprio_projection(torch.cat((joint_history, action_history), dim=-1))
        proprio = proprio + self.proprio_position + self.proprio_modality
        robot_token = robot[:, None, :] + self.robot_modality
        global_null = self.global_null_token.expand(batch, -1, -1)

        memory = torch.cat(
            (
                language,
                global_null,
                compressed,
                selected_tokens,
                local_tokens,
                proprio,
                robot_token,
            ),
            dim=1,
        )
        memory_padding = torch.cat(
            (
                ~language_mask,
                torch.zeros(batch, 1, dtype=torch.bool, device=language_mask.device),
                ~block_valid,
                selected_padding,
                local_padding,
                ~proprio_mask,
                torch.zeros(batch, 1, dtype=torch.bool, device=language_mask.device),
            ),
            dim=1,
        )
        audit: dict[str, torch.Tensor | int] = {
            "selected_old_block_indices": selected_indices,
            "selected_old_block_scores": selected_scores,
            "old_block_valid": block_valid,
            "dense_visual_token_count": cfg.visual_history_steps * cfg.spatial_token_count,
            "sparse_visual_memory_token_count": (
                cfg.old_block_count
                + cfg.selected_old_blocks * block_fine_count
                + 1
                + cfg.local_visual_steps * cfg.spatial_token_count
            ),
            "retained_language_token_slots": cfg.language_max_tokens,
        }
        return memory, memory_padding, audit

    def forward(self, inputs: Mapping[str, torch.Tensor]) -> torch.Tensor:
        """Predict normalized action chunks, while deployment uses row zero."""

        memory, memory_padding, _audit = self.encode_sparse_memory(inputs)
        batch = memory.shape[0]
        queries = self.action_queries.expand(
            batch,
            self.config.action_chunk_size,
            self.config.model_dim,
        )
        decoded = self.action_decoder(
            tgt=queries,
            memory=memory,
            memory_key_padding_mask=memory_padding,
        )
        return torch.tanh(self.action_head(self.action_norm(decoded)))

    def predict_next_action(self, inputs: Mapping[str, torch.Tensor]) -> torch.Tensor:
        """Replan now and return only the next six-joint command."""

        return self.forward(inputs)[:, 0]


__all__ = [
    "ChannelwiseDeltaMemoryV26",
    "SPARSE_4D_VLA_ACT_ARCHITECTURE_V26",
    "SPARSE_4D_VLA_ACT_CONFIG_VERSION_V26",
    "SPARSE_4D_VLA_ACT_INPUT_KEYS_V26",
    "SPARSE_4D_VLA_ACT_INPUT_SCHEMA_HASH_V26",
    "Sparse4DVLAConfigV26",
    "Sparse4DVLAScaleV26",
    "Sparse4DVLAACTV26",
    "sparse_4d_vla_config_preset_v26",
]
