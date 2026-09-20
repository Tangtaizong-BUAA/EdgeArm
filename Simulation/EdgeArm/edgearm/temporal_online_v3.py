"""Online causal buffer shared with the offline temporal input builder."""

from collections import deque
import numpy as np
import torch
from .temporal_input_contract_v3 import make_sample


class _SelectedRows:
    """Stack only requested frames, not all 901 RGB images at each action."""
    def __init__(self, rows, key):
        self.rows, self.key = rows, key

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        if isinstance(index, tuple):
            value = self[index[0]]
            prefix = () if isinstance(index[0], (int, np.integer)) else (slice(None),)
            return value[prefix + index[1:]]
        if isinstance(index, (int, np.integer)):
            return np.asarray(self.rows[index][self.key])
        ids = np.arange(len(self.rows))[index]
        if not len(ids):
            sample = np.asarray(self.rows[0][self.key])
            return np.empty((0, *sample.shape), sample.dtype)
        return np.asarray([self.rows[int(i)][self.key] for i in ids])


class TemporalOnlineBuffer:
    def __init__(self, cfg, instruction, K):
        self.cfg = cfg
        self.instruction = instruction
        self.K = np.asarray(K, np.float32)
        # Episode anchors must not silently lose their early observations.
        # This remains a bounded 900-action simulator horizon, not global memory.
        size = max((cfg.visual_history_steps - 1) * 4 + 1, cfg.proprio_history_steps + 1)
        if getattr(cfg, "visual_memory_mode", "recent") == "episode_anchors_v54":
            size = max(size, 901)
        self.rows = deque(maxlen=size)
        self.pending = False

    def observe(self, *, rgb, joint, tool, camera_pose, time_s, geometry_valid):
        if self.pending:
            raise ValueError("previous command transaction not completed")
        if self.rows and time_s <= self.rows[-1]["time"]:
            raise ValueError("nonmonotonic observation")
        q = np.asarray(joint, np.float32)
        motion = np.zeros(6, np.float32) if not self.rows else (q[:6] - self.rows[-1]["joint"][:6]) / 0.1
        self.rows.append(
            dict(
                rgb=np.asarray(rgb, np.uint8),
                joint=q,
                previous_motion=motion,
                tool=np.asarray(tool, np.float32),
                camera_pose=np.asarray(camera_pose, np.float32),
                time=float(time_s),
                geometry_valid=bool(geometry_valid),
                command=np.zeros(6, np.float32),
                command_valid=False,
                applied_target=np.zeros(6, np.float32),
                tracking_error=np.zeros(6, np.float32),
                feedback_valid=False,
            )
        )
        self.pending = True

    def inputs(self):
        if not self.pending:
            raise ValueError("observe current state before inference")
        if getattr(self.cfg, "visual_memory_mode", "recent") == "episode_anchors_v54":
            rows = list(self.rows)
            z = {k: _SelectedRows(rows, k) for k in rows[-1]}
            z['K'] = self.K
            return make_sample(z, len(rows)-1, self.instruction, self.cfg,
                               include_auxiliary_depth=False)['inputs']
        z = {k: np.asarray([r[k] for r in self.rows]) for k in self.rows[-1]}
        z["K"] = self.K
        n, h, w, _ = z["rgb"].shape
        # Zero placeholders exist solely to reuse the training builder; no depth sensor/label is read.
        z["auxiliary_depth_mm"] = np.zeros((n, h, w), np.uint16)
        z["auxiliary_depth_frame_mask"] = np.zeros(n, bool)
        return make_sample(z, n - 1, self.instruction, self.cfg)["inputs"]

    def tensors(self):
        return {k: torch.from_numpy(v)[None] for k, v in self.inputs().items()}

    def complete_action(self, *, submitted_command, applied_target, reported_next_q, feedback_valid):
        if not self.pending:
            raise ValueError("no pending observation")
        row = self.rows[-1]
        row["command"] = np.asarray(submitted_command, np.float32)
        row["command_valid"] = True
        row["applied_target"] = np.asarray(applied_target, np.float32)
        row["tracking_error"] = row["applied_target"] - np.asarray(reported_next_q, np.float32)
        row["feedback_valid"] = bool(feedback_valid)
        self.pending = False
