"""Causal temporal loader; auxiliary depth and future state stay outside actor."""

from collections import OrderedDict
import numpy as np
import torch
from .vla_data import encode_language
from .sparse_4d_vla_act_v26 import SPARSE_4D_VLA_ACT_INPUT_KEYS_V26

EXTRA_KEYS = frozenset(
    {
        "kinematic_history",
        "command_history",
        "applied_target_history",
        "tracking_error_history",
        "command_feedback_mask",
        "reference_path",
        "reference_path_mask",
        "geometry_available_mask",
    }
)
INPUT_KEYS = SPARSE_4D_VLA_ACT_INPUT_KEYS_V26 | EXTRA_KEYS


def history_indices(t, count, stride=1):
    indices = t - np.arange(count - 1, -1, -1) * stride
    return np.maximum(indices, 0), indices >= 0


def visual_indices(t, cfg):
    """Fixed token budget: recent detail plus causal whole-episode anchors.

    Legacy checkpoints retain their exact original sampling. New checkpoints
    explicitly opt in; offline and online call the same function.
    """
    count = cfg.visual_history_steps
    if getattr(cfg, "visual_memory_mode", "recent") == "recent" or t <= (count - 1) * 4:
        return history_indices(t, count, 4)
    anchors = max(4, (count // 3 // 4) * 4)
    recent = t - np.arange(count - anchors - 1, -1, -1) * 4
    early = np.linspace(0, int(recent[0]) - 4, anchors, dtype=np.int64)
    indices = np.concatenate((early, recent))
    if not (np.diff(indices) > 0).all() or indices[-1] != t:
        raise ValueError("invalid causal episode anchors")
    return indices, np.ones(count, bool)


def make_sample(z, t, instruction, cfg, include_auxiliary_depth=True):
    vi, vm = visual_indices(t, cfg)
    pi, pm = history_indices(t, cfg.proprio_history_steps)
    # Current state includes t; commands and their observed effects must end at t-1.
    ci, cm = history_indices(t - 1, cfg.proprio_history_steps)
    tokens, lm = encode_language(instruction, cfg.language_max_tokens)
    if len(instruction.encode()) + 2 > cfg.language_max_tokens:
        raise ValueError("instruction truncation forbidden")

    def take(key, idx, mask):
        a = z[key][idx].copy()
        a[~mask] = 0
        return a

    rgb = take("rgb", vi, vm)
    height, width = rgb.shape[1:3]
    time = take("time", vi, vm).astype(np.float32)
    time[vm] = (z["time"][vi[vm]] - z["time"][t]).astype(np.float32)
    zero = np.zeros((cfg.visual_history_steps, height, width), np.float32)
    inputs = dict(
        language_token_ids=tokens,
        language_attention_mask=lm,
        rgb_wrist_window=rgb,
        depth_wrist_m_window=zero,
        depth_valid_mask_window=zero.astype(bool),
        camera_pose_wrist_window=take("camera_pose", vi, vm).astype(np.float32),
        camera_intrinsics=z["K"].astype(np.float32),
        visual_history_mask=vm,
        camera_4d_reconstructable_mask_window=np.zeros_like(vm),
        camera_geometry_alignment_exact_window=np.zeros_like(vm),
        camera_device_time_delta_s_window=time,
        camera_host_time_delta_s_window=time.copy(),
        robot_state=z["joint"][t].astype(np.float32),
        joint_history=take("joint", pi, pm).astype(np.float32),
        action_history=np.clip(take("previous_motion", pi, pm), -1, 1),
        proprio_history_mask=pm,
        kinematic_history=take("tool", pi, pm).astype(np.float32),
        command_history=take("command", ci, cm).astype(np.float32),
        applied_target_history=take("applied_target", ci, cm).astype(np.float32),
        tracking_error_history=take("tracking_error", ci, cm).astype(np.float32),
        command_feedback_mask=cm & z["feedback_valid"][ci],
        geometry_available_mask=vm & z["geometry_valid"][vi],
        reference_path=np.zeros((cfg.proprio_history_steps, 3), np.float32),
        reference_path_mask=np.zeros(cfg.proprio_history_steps, bool),
    )
    assert frozenset(inputs) == INPUT_KEYS
    for key, x in inputs.items():
        if np.issubdtype(x.dtype, np.floating) and not np.isfinite(x).all():
            raise ValueError(f"nonfinite {key}")
    if np.any(time[vm] > 0):
        raise ValueError("future frame")
    n = min(cfg.action_chunk_size, len(z["joint"]) - t)
    target = np.zeros((cfg.action_chunk_size, 6), np.float32)
    mask = np.zeros(cfg.action_chunk_size, bool)
    target[:n] = z["command"][t : t + n]
    mask[:n] = z["command_valid"][t : t + n]
    # Future FK positions are supervision, never a planned path or actor input.
    fn = min(cfg.action_chunk_size, len(z["joint"]) - t - 1)
    future_tool = np.zeros((cfg.action_chunk_size, 3), np.float32)
    future_mask = np.zeros(cfg.action_chunk_size, bool)
    future_tool[:fn] = z["tool"][t + 1 : t + 1 + fn, :3]
    future_mask[:fn] = True
    result = dict(
        inputs=inputs,
        target=target,
        mask=mask,
        future_tool_xyz=future_tool,
        future_tool_mask=future_mask,
    )
    if include_auxiliary_depth:
        result.update(
            auxiliary_depth_m=take("auxiliary_depth_mm", vi, vm).astype(np.float32) / 1000,
            auxiliary_depth_mask=vm & z["auxiliary_depth_frame_mask"][vi],
        )
    return result


class TemporalPackets:
    def __init__(self, records, cfg, stride=4, cache_episodes=4, include_auxiliary_depth=True):
        if cache_episodes < 1:
            raise ValueError("cache_episodes must be positive")
        self.records = records
        self.cfg = cfg
        self.cache_episodes = cache_episodes
        self.include_auxiliary_depth = include_auxiliary_depth
        self.cache = OrderedDict()
        self.indices = []
        for i, r in enumerate(records):
            with np.load(r["packet"]) as z:
                valid = z["command_valid"]
            self.indices.extend((i, t) for t in range(0, r["length"], stride) if valid[t])

    def sample(self, index):
        i, t = self.indices[index]
        r = self.records[i]
        p = r["packet"]
        if p not in self.cache:
            with np.load(p) as z:
                self.cache[p] = {
                    k: z[k]
                    for k in z.files
                    if self.include_auxiliary_depth or not k.startswith("auxiliary_depth")
                }
            if len(self.cache) > self.cache_episodes:
                self.cache.popitem(last=False)
        self.cache.move_to_end(p)
        return make_sample(self.cache[p], t, r["instruction"], self.cfg, self.include_auxiliary_depth)

    def batch(self, ids):
        rows = [self.sample(i) for i in ids]
        b = {"inputs": {k: torch.from_numpy(np.stack([r["inputs"][k] for r in rows])) for k in INPUT_KEYS}}
        labels = ["target", "mask", "future_tool_xyz", "future_tool_mask"]
        if self.include_auxiliary_depth:
            labels.extend(("auxiliary_depth_m", "auxiliary_depth_mask"))
        b.update({k: torch.from_numpy(np.stack([r[k] for r in rows])) for k in labels})
        return b
