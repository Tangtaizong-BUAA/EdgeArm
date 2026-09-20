"""Training loader for production HDF5 shards and action-chunk supervision."""

from __future__ import annotations

import json
from pathlib import Path
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


PAD_TOKEN = 0
BOS_TOKEN = 1
EOS_TOKEN = 2
BYTE_OFFSET = 3
LANGUAGE_VOCAB_SIZE = 259


def encode_language(text: str, max_length: int = 64) -> tuple[np.ndarray, np.ndarray]:
    values = [BOS_TOKEN, *[int(value) + BYTE_OFFSET for value in text.encode("utf-8")], EOS_TOKEN]
    values = values[:max_length]
    if values[-1] != EOS_TOKEN:
        values[-1] = EOS_TOKEN
    mask = np.zeros(max_length, dtype=bool)
    mask[: len(values)] = True
    tokens = np.full(max_length, PAD_TOKEN, dtype=np.int64)
    tokens[: len(values)] = values
    return tokens, mask


class ProductionVLADataset(Dataset):
    def __init__(
        self,
        root: Path,
        split: str,
        *,
        chunk_size: int = 16,
        successes_only: bool = True,
        include_every_nth_frame: int = 1,
    ):
        self.root = Path(root)
        self.split = split
        self.chunk_size = chunk_size
        manifest = self.root / split / "manifest.jsonl"
        entries = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line]
        if successes_only:
            entries = [entry for entry in entries if entry["success"]]
        self.episodes = entries
        self.index: list[tuple[int, int]] = []
        for episode_index, entry in enumerate(entries):
            for frame in range(0, int(entry["frames"]), include_every_nth_frame):
                self.index.append((episode_index, frame))
        self._files: dict[Path, h5py.File] = {}

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        episode_index, frame = self.index[index]
        entry = self.episodes[episode_index]
        shard_path = self.root / self.split / entry["shard"]
        stream = self._files.get(shard_path)
        if stream is None:
            stream = h5py.File(shard_path, "r", swmr=True)
            self._files[shard_path] = stream
        group = stream[entry["group"]]
        length = int(entry["frames"])
        end = min(frame + self.chunk_size, length)
        frame_indices = np.arange(frame, end)
        mask = np.zeros(self.chunk_size, dtype=bool)
        mask[: len(frame_indices)] = True

        current_q = group["joint_position"][frame_indices]
        target_q = group["action_joint_target"][frame_indices]
        actions = np.clip((target_q - current_q) / float(group.attrs["max_joint_delta_rad"]), -1.0, 1.0)
        action_chunk = np.zeros((self.chunk_size, 6), dtype=np.float32)
        action_chunk[: len(frame_indices)] = actions.astype(np.float32)
        if len(frame_indices):
            action_chunk[len(frame_indices) :] = action_chunk[len(frame_indices) - 1]

        rgb = group["rgb_wrist"][frame][None, ...].astype(np.uint8)
        depth = group["depth_wrist_mm"][frame][None, ...].astype(np.uint16)
        segmentation = group["segmentation_wrist"][frame][None, ...].astype(np.uint8)
        qpos = group["joint_position"][frame].astype(np.float32)
        qvel = group["joint_velocity"][frame].astype(np.float32)
        robot_state = np.concatenate([qpos / np.pi, np.clip(qvel, -4.0, 4.0) / 4.0]).astype(np.float32)
        task_vector = np.concatenate(
            [
                group["target_xy"][frame] / 0.5,
                group["obstacle_xy"][frame] / 0.5,
                [float(bool(group.attrs["obstacle"])), group["teacher_confidence"][frame]],
            ]
        ).astype(np.float32)
        privileged = np.concatenate(
            [robot_state, group["tool_pose"][frame].astype(np.float32), task_vector]
        ).astype(np.float32)
        scene_target = np.concatenate(
            [
                group["block_xy"][frame] / 0.5,
                group["target_xy"][frame] / 0.5,
                group["obstacle_xy"][frame] / 0.5,
                [float(bool(group.attrs["obstacle"]))],
            ]
        ).astype(np.float32)
        task = str(group.attrs["task_zh"] if index % 3 == 0 else group.attrs["task"])
        tokens, language_mask = encode_language(task)
        return {
            "rgb": torch.from_numpy(rgb),
            "depth_mm": torch.from_numpy(depth.astype(np.int32)),
            "segmentation": torch.from_numpy(segmentation.astype(np.int64)),
            "robot_state": torch.from_numpy(robot_state),
            "privileged_state": torch.from_numpy(privileged),
            "scene_target": torch.from_numpy(scene_target),
            "language_tokens": torch.from_numpy(tokens),
            "language_mask": torch.from_numpy(language_mask),
            "action_chunk": torch.from_numpy(action_chunk),
            "action_mask": torch.from_numpy(mask),
            "teacher_confidence": torch.tensor(float(group["teacher_confidence"][frame]), dtype=torch.float32),
            "phase_id": torch.tensor(int(group["phase_id"][frame]), dtype=torch.int64),
        }

    def close(self) -> None:
        for stream in self._files.values():
            try:
                stream.close()
            except (TypeError, ValueError):
                pass
        self._files.clear()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


class TemporalProductionVLADataset(ProductionVLADataset):
    """Wrist-only production samples with causal RGB history from one episode.

    The last history element is always the current frame.  Missing history at
    the beginning of an episode is padded by repeating that episode's first
    frame, never by reading a frame from the previous episode or shard.
    """

    def __init__(
        self,
        root: Path,
        split: str,
        *,
        chunk_size: int = 16,
        successes_only: bool = True,
        include_every_nth_frame: int = 1,
        history_frames: int = 4,
        history_stride: int = 1,
    ):
        if history_frames < 1:
            raise ValueError("history_frames must be positive")
        if history_stride < 1:
            raise ValueError("history_stride must be positive")
        self.history_frames = int(history_frames)
        self.history_stride = int(history_stride)
        super().__init__(
            root,
            split,
            chunk_size=chunk_size,
            successes_only=successes_only,
            include_every_nth_frame=include_every_nth_frame,
        )

    def history_indices(self, frame: int) -> np.ndarray:
        """Return oldest-to-current causal indices with first-frame padding."""
        offsets = np.arange(self.history_frames - 1, -1, -1, dtype=np.int64)
        return np.maximum(0, int(frame) - offsets * self.history_stride)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        item = super().__getitem__(index)
        episode_index, frame = self.index[index]
        entry = self.episodes[episode_index]
        shard_path = self.root / self.split / entry["shard"]
        stream = self._files[shard_path]
        group = stream[entry["group"]]
        frame_indices = self.history_indices(frame)

        # h5py fancy indexing rejects repeated indices, which are intentional
        # here for causal first-frame padding, so load the small K-frame window
        # explicitly.  rgb_wrist is the only visual stream exposed to v2.
        # ProductionVLADataset already decoded the current frame. Reuse it so
        # temporal training does not inflate HDF5 decompression by 25%.
        current_rgb = item["rgb"][0].numpy()
        rgb_history = np.stack(
            [
                current_rgb
                if int(history_frame) == int(frame)
                else group["rgb_wrist"][int(history_frame)]
                for history_frame in frame_indices
            ],
            axis=0,
        ).astype(np.uint8, copy=False)
        item["rgb"] = torch.from_numpy(rgb_history)
        item["history_frame_indices"] = torch.from_numpy(frame_indices.copy())
        # Training v2 deliberately keeps hard/failure trajectories.  Expose
        # episode-level context as tensors so loss weighting can emphasize
        # obstacle, stress, avoid, and recovery states without changing the
        # deployable model input contract.
        item["episode_success"] = torch.tensor(bool(entry["success"]), dtype=torch.bool)
        item["episode_obstacle"] = torch.tensor(bool(group.attrs.get("obstacle", False)), dtype=torch.bool)
        item["episode_stress"] = torch.tensor(bool(group.attrs.get("stress", False)), dtype=torch.bool)
        return item
