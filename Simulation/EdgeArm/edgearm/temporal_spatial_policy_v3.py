"""Reconnect the sparse temporal core with FK, feedback and RGB-estimated 4D tokens.

This is a local integration implementation, not validated object-level SLAM.
Simulator depth is absent from the actor schema. An RGB-only depth student is
trained with separate auxiliary labels; its uncalibrated output is not truth.
"""

import torch
from torch import nn
import torch.nn.functional as F
from .sparse_4d_vla_act_v26 import Sparse4DVLAACTV26, SPARSE_4D_VLA_ACT_INPUT_KEYS_V26, Sparse4DVLAConfigV26
from .causal_4d_act_v1 import backproject_mujoco_wrist_depth
from .temporal_input_contract_v3 import INPUT_KEYS


def last_completed_target(inputs):
    """Causal actuator target, falling back to current reported joints at reset."""
    mask = inputs['command_feedback_mask']
    n = mask.shape[1]
    indices = torch.where(mask, torch.arange(n, device=mask.device), -1).max(1).values
    target = inputs['applied_target_history'][torch.arange(len(mask), device=mask.device),
                                              indices.clamp_min(0)]
    return torch.where((indices >= 0)[:, None], target, inputs['robot_state'][:, :6])


def reference_target_commands(delta, current, anchor, *, bounded):
    """Network predicts actuator-reference increments; feedback converts to Command-V2."""
    command = delta.float() + (anchor.float()-current.float())[:, None, :]/.055
    return command.clamp(-1, 1) if bounded else command


def absolute_target_commands(normalized_target, current, *, bounded):
    """Absolute radians / pi, independent of the last predicted/applied target."""
    command = (normalized_target.float()*torch.pi-current.float()[:, None, :])/.055
    return command.clamp(-1, 1) if bounded else command


def adaptive_pool_bins(x, output_size):
    """Exact adaptive-average bin boundaries using ordinary tensor reductions.

    Metal does not implement non-divisible adaptive pooling in this runtime.
    Keep the original bin geometry instead of resizing/padding the depth map.
    """
    height, width = x.shape[-2:]
    out_h, out_w = output_size
    rows = []
    for row in range(out_h):
        start_h = row * height // out_h
        end_h = ((row + 1) * height + out_h - 1) // out_h
        columns = []
        for column in range(out_w):
            start_w = column * width // out_w
            end_w = ((column + 1) * width + out_w - 1) // out_w
            columns.append(x[..., start_h:end_h, start_w:end_w].mean(dim=(-2, -1)))
        rows.append(torch.stack(columns, dim=-1))
    return torch.stack(rows, dim=-2)


def adaptive_pool_compatible(x, output_size):
    if x.device.type == "mps" and any(a % b for a, b in zip(x.shape[-2:], output_size)):
        return adaptive_pool_bins(x, output_size)
    return F.adaptive_avg_pool2d(x, output_size)


class RGBDepthStudent(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 5, 2, 2),
            nn.GELU(),
            nn.Conv2d(32, 64, 3, 2, 1),
            nn.GELU(),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(64, 2, 1),
        )

    def forward(self, rgb):
        x = self.net(rgb)
        depth = 0.01 + 1.99 * torch.sigmoid(x[:, 0])
        uncertainty = 0.01 + F.softplus(x[:, 1])
        return depth, uncertainty


class EstimatedGeometryCore(Sparse4DVLAACTV26):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.depth_student = RGBDepthStudent()
        self.space_projection = nn.Sequential(
            nn.Linear(8, cfg.model_dim), nn.GELU(), nn.Linear(cfg.model_dim, cfg.model_dim)
        )
        self.geometry_available = None
        self.depth_aux = None

    def _frame_tokens(self, inputs):
        tokens, padding = super()._frame_tokens(inputs)
        b, t, h, w, _ = inputs["rgb_wrist_window"].shape
        rgb = inputs["rgb_wrist_window"].float().permute(0, 1, 4, 2, 3).reshape(b * t, 3, h, w) / 255
        depth, unc = self.depth_student(rgb)
        grid = adaptive_pool_compatible(depth[:, None], (4, 6)).reshape(b, t, 4, 6)
        uncertainty = adaptive_pool_compatible(unc[:, None], (4, 6)).reshape(b, t, 4, 6)
        ys = (torch.arange(4, device=rgb.device) + 0.5) * h / 4 - 0.5
        xs = (torch.arange(6, device=rgb.device) + 0.5) * w / 6 - 0.5
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        pixels = torch.stack((xx, yy), -1)
        pose = inputs["camera_pose_wrist_window"].clone()
        gm = self.geometry_available & inputs["visual_history_mask"]
        pose[~gm] = torch.cat(
            (
                torch.zeros(3, device=pose.device, dtype=pose.dtype),
                torch.eye(3, device=pose.device, dtype=pose.dtype).flatten(),
            )
        )
        # Camera calibration and metric transforms keep FP32 precision under
        # AMP; low-precision convolution/attention does not relax geometry.
        with torch.autocast(device_type=rgb.device.type, enabled=False):
            xyz = backproject_mujoco_wrist_depth(
                grid.float(), inputs["camera_intrinsics"].float(), pose.float(),
                pixel_coordinates_uv=pixels.float()
            )
        # XYZ + event time + depth uncertainty + view validity + image-plane coordinates.
        dt = inputs["camera_device_time_delta_s_window"][:, :, None, None, None].expand(b, t, 4, 6, 1)
        uv = pixels[None, None].expand(b, t, 4, 6, 2) / torch.tensor([w, h], device=rgb.device)
        features = torch.cat(
            (xyz, dt, uncertainty[..., None], gm[:, :, None, None, None].expand(b, t, 4, 6, 1).float(), uv),
            -1,
        )
        space = self.space_projection(features).reshape(b, t, 24, -1)
        self.depth_aux = dict(
            estimated_depth_m=depth.reshape(b, t, *depth.shape[-2:]),
            depth_uncertainty_m=unc.reshape(b, t, *unc.shape[-2:]),
            geometry_tokens_valid=gm.sum() * 24,
        )
        # Fuse into the SAME visual tokens before local/compressed/selected attention.
        # Do not append a dense copy of all old geometric tokens to the decoder.
        return tokens + torch.where(gm[:, :, None, None], space, torch.zeros_like(space)), padding


class TemporalSpatialPolicyV3(nn.Module):
    def __init__(self, cfg=None):
        super().__init__()
        self.config = cfg or Sparse4DVLAConfigV26(proprio_history_steps=32, language_max_tokens=128)
        if (self.config.spatial_grid_height, self.config.spatial_grid_width) != (4, 6):
            raise ValueError("V3 geometry grid requires 4x6")
        self.core = EstimatedGeometryCore(self.config)
        d = self.config.model_dim
        self.kinematic_projection = nn.Sequential(
            nn.LayerNorm(18), nn.Linear(18, d), nn.GELU(), nn.Linear(d, d)
        )
        self.feedback_projection = nn.Sequential(
            nn.LayerNorm(18), nn.Linear(18, d), nn.GELU(), nn.Linear(d, d)
        )
        self.reference_projection = nn.Linear(3, d)
        self.extra_position = nn.Parameter(torch.randn(1, self.config.proprio_history_steps, d) * 0.01)
        self.next_tool_head = nn.Linear(d, 3)

    @property
    def depth_student(self):
        return self.core.depth_student

    def _validate_extra(self, inputs):
        b = inputs["robot_state"].shape[0]
        p = self.config.proprio_history_steps
        t = self.config.visual_history_steps
        if inputs["rgb_wrist_window"].dtype != torch.uint8:
            raise ValueError("V3 RGB must be uint8; normalization occurs once inside the actor")
        shapes = {
            "kinematic_history": (b, p, 18),
            "command_history": (b, p, 6),
            "applied_target_history": (b, p, 6),
            "tracking_error_history": (b, p, 6),
            "reference_path": (b, p, 3),
            "command_feedback_mask": (b, p),
            "reference_path_mask": (b, p),
            "geometry_available_mask": (b, t),
        }
        for key, shape in shapes.items():
            x = inputs[key]
            if tuple(x.shape) != shape:
                raise ValueError(f"{key} shape mismatch")
            if key.endswith("_mask"):
                if x.dtype != torch.bool:
                    raise ValueError(f"{key} must be boolean")
            elif not torch.is_floating_point(x) or not torch.isfinite(x).all():
                raise ValueError(f"{key} must be finite float")
        if (inputs["geometry_available_mask"] & ~inputs["visual_history_mask"]).any():
            raise ValueError("geometry on padded frame")
        poses = inputs["camera_pose_wrist_window"][inputs["geometry_available_mask"]]
        if len(poses):
            if not torch.isfinite(poses).all():
                raise ValueError("nonfinite geometry camera pose")
            rotation = poses[:, 3:].reshape(-1, 3, 3)
            identity = torch.eye(3, device=rotation.device, dtype=rotation.dtype)
            if not torch.allclose(
                rotation @ rotation.transpose(-1, -2), identity.expand_as(rotation), atol=0.002, rtol=0
            ) or not torch.allclose(
                torch.linalg.det(rotation),
                torch.ones(len(rotation), device=rotation.device, dtype=rotation.dtype),
                atol=0.002,
                rtol=0,
            ):
                raise ValueError("invalid geometry camera rotation")
        if (inputs["command_feedback_mask"] & ~inputs["proprio_history_mask"]).any():
            raise ValueError("feedback before episode start")
        if inputs["command_history"].abs().max() > 1.000001:
            raise ValueError("past command outside normalized contract")

    def encode_observations(self, inputs):
        if frozenset(inputs) != INPUT_KEYS:
            raise ValueError("V3 input contract mismatch")
        with torch.autocast(device_type=inputs["robot_state"].device.type, enabled=False):
            self._validate_extra(inputs)
        base = {k: inputs[k] for k in SPARSE_4D_VLA_ACT_INPUT_KEYS_V26}
        if base["depth_valid_mask_window"].any() or torch.count_nonzero(base["depth_wrist_m_window"]):
            raise ValueError("actor received externally supplied depth; use RGB student")
        self.core.geometry_available = inputs["geometry_available_mask"]
        memory, padding, audit = self.core.encode_sparse_memory(base)
        # Mask before projection too: padding values must not influence valid tokens.
        kinematic = (
            self.kinematic_projection(
                torch.where(inputs["proprio_history_mask"][..., None], inputs["kinematic_history"], 0.0)
            )
            + self.extra_position
        )
        f = torch.cat(
            (inputs["command_history"], inputs["applied_target_history"], inputs["tracking_error_history"]),
            -1,
        )
        feedback = (
            self.feedback_projection(torch.where(inputs["command_feedback_mask"][..., None], f, 0.0))
            + self.extra_position
        )
        reference = (
            self.reference_projection(
                torch.where(inputs["reference_path_mask"][..., None], inputs["reference_path"], 0.0)
            )
            + self.extra_position
        )
        memory = torch.cat((memory, kinematic, feedback, reference), 1)
        padding = torch.cat(
            (
                padding,
                ~inputs["proprio_history_mask"],
                ~inputs["command_feedback_mask"],
                ~inputs["reference_path_mask"],
            ),
            1,
        )
        if self.config.action_head_space in ('applied_target_delta_v56', 'absolute_joint_target_v57'):
            # Stateless payload for the analytic output conversion. Fully masked
            # out of attention and pooled representations; both fields are causal.
            side = torch.zeros((len(memory), 1, memory.shape[-1]), device=memory.device, dtype=torch.float32)
            side[:, 0, :6] = inputs['robot_state'][:, :6].float()
            side[:, 0, 6:12] = last_completed_target(inputs).float()
            memory = torch.cat((memory.float(), side), 1)
            padding = torch.cat((padding, torch.ones((len(memory),1), device=padding.device, dtype=torch.bool)), 1)
        return memory, padding, audit

    def decode_observations(self, memory, padding, query_offset=None):
        b = len(memory)
        queries = self.core.action_queries.expand(b, -1, -1)
        if query_offset is not None:
            queries = queries + query_offset[:, None, :]
        decoded = self.core.action_decoder(queries, memory, memory_key_padding_mask=padding)
        if self.config.action_head_space == 'absolute_joint_target_v57':
            # BF16 absolute radians can quantize away milliradian corrections.
            # Keep the final normalization, affine map and reference subtraction
            # in FP32 even when the expensive encoder/decoder uses AMP.
            with torch.autocast(device_type=decoded.device.type, enabled=False):
                raw = self.core.action_head(self.core.action_norm(decoded.float()))
                action = absolute_target_commands(torch.tanh(raw), memory[:, -1, :6],
                                                  bounded=not self.training)
        elif self.config.action_head_space == 'applied_target_delta_v56':
            raw = self.core.action_head(self.core.action_norm(decoded))
            action = reference_target_commands(raw, memory[:, -1, :6], memory[:, -1, 6:12],
                                               bounded=not self.training)
        else:
            raw = self.core.action_head(self.core.action_norm(decoded))
            action = torch.tanh(raw)
        return action, decoded

    def forward(self, inputs, return_aux=False):
        memory, padding, audit = self.encode_observations(inputs)
        action, decoded = self.decode_observations(memory, padding)
        if not return_aux:
            return action
        return dict(
            action=action,
            next_tool_xyz=self.next_tool_head(decoded),
            **self.core.depth_aux,
            sparse_audit=audit,
        )
