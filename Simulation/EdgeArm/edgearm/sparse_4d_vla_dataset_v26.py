"""Fail-closed causal ACT samples from V26 deterministic wrist replay H5.

Only rows admitted by the replay's strict three-second action-supervision mask
become training anchors.  Observation/history windows end at the current
pre-action row; action chunks may look forward only as labels and never cross
an episode boundary.  Segmentation remains in the source for audit but is not
read into the returned policy inputs.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from .replay_rl_multimodal_v26 import RL_MULTIMODAL_REPLAY_FORMAT_V26
from .sparse_4d_vla_act_v26 import (
    SPARSE_4D_VLA_ACT_INPUT_KEYS_V26,
    Sparse4DVLAConfigV26,
)
from .trisource_contract_v26 import SIM_RL_SCRATCH_SOURCE_V26
from .vla_data import LANGUAGE_VOCAB_SIZE, encode_language


SPARSE_4D_VLA_DATASET_FORMAT_V26 = "edgearm-v26-sparse-4d-vla-act-dataset-v1"
LanguageModeV26 = Literal["english", "chinese", "bilingual_alternating"]


def _attribute_text(attributes: h5py.AttributeManager, name: str) -> str:
    if name not in attributes:
        raise ValueError(f"V26 replay is missing required attribute: {name}")
    value = attributes[name]
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def _decode(value: Any) -> str:
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def _require_dataset(stream: h5py.File, name: str) -> h5py.Dataset:
    value = stream.get(name)
    if not isinstance(value, h5py.Dataset):
        raise ValueError(f"V26 replay is missing required dataset: {name}")
    return value


class Sparse4DVLAReplayDatasetV26(Dataset[dict[str, Any]]):
    """Lazy, multiprocessing-safe dataset over one immutable replay artifact."""

    def __init__(
        self,
        replay_path: Path,
        *,
        model_config: Sparse4DVLAConfigV26 | None = None,
        language_mode: LanguageModeV26 = "bilingual_alternating",
        storage_path: Path | None = None,
    ) -> None:
        self.replay_path = Path(replay_path).expanduser().resolve()
        self.storage_path = (
            self.replay_path
            if storage_path is None
            else Path(storage_path).expanduser().resolve()
        )
        self.model_config = model_config or Sparse4DVLAConfigV26()
        self.language_mode = language_mode
        self._stream: h5py.File | None = None
        if not self.replay_path.is_file():
            raise FileNotFoundError(f"V26 replay artifact is missing: {self.replay_path}")
        if not self.storage_path.is_file():
            raise FileNotFoundError(
                f"V26 replay storage artifact is missing: {self.storage_path}"
            )
        if language_mode not in {"english", "chinese", "bilingual_alternating"}:
            raise ValueError("V26 language mode is unsupported")
        if self.model_config.language_vocabulary_size != LANGUAGE_VOCAB_SIZE:
            raise ValueError("V26 model must use the existing utf8-byte-v1 vocabulary")

        with h5py.File(self.storage_path, "r", swmr=True) as stream:
            self._preflight(stream)
            self.episode_ids = np.asarray(
                _require_dataset(stream, "index/episode_id")[...],
                dtype=np.int64,
            )
            self.source_episode_ids = np.asarray(
                _require_dataset(stream, "index/source_episode_id")[...],
                dtype=np.int64,
            )
            self.episode_step_ids = np.asarray(
                _require_dataset(stream, "index/episode_step_id")[...],
                dtype=np.int64,
            )
            action_admission = np.asarray(
                _require_dataset(stream, "admission/act_action_supervision_mask")[...]
            )
            execution_attempted = np.asarray(_require_dataset(stream, "outcome/execution_attempted")[...])
            submission_unchanged = np.asarray(
                _require_dataset(
                    stream,
                    "outcome/current_submission_safety_unchanged",
                )[...]
            )
            self._validate_index_vectors(
                action_admission,
                execution_attempted,
                submission_unchanged,
            )
            self.anchor_rows = np.flatnonzero(action_admission).astype(np.int64)
            if not len(self.anchor_rows):
                raise ValueError("V26 replay contains no admitted ACT/VLA action rows")
            self._action_admission = action_admission.astype(bool, copy=True)
            self._episode_bounds = self._build_episode_bounds()

    def _required_replay_format(self) -> str:
        """Exact immutable artifact format accepted by this dataset class."""

        return RL_MULTIMODAL_REPLAY_FORMAT_V26

    def _preflight_source_contract(self, stream: h5py.File) -> None:
        """Fail-closed provenance checks specific to scratch-RL replay."""

        if int(stream.attrs.get("expert_calls", -1)) != 0:
            raise ValueError("V26 scratch-RL replay contains expert calls")
        if int(stream.attrs.get("behavior_cloning_steps_used_to_generate_actions", -1)) != 0:
            raise ValueError("V26 scratch-RL replay contains behavior-cloned actions")

    def _preflight(self, stream: h5py.File) -> None:
        if _attribute_text(stream.attrs, "format") != self._required_replay_format():
            raise ValueError("V26 sparse dataset requires the exact replay format")
        required_true = (
            "finalized",
            "simulated_wrist_rgbd",
            "camera_geometry_alignment_exact",
            "camera_4d_reconstructable",
            "strict_success_trace_verified",
            "all_state_replays_exact",
        )
        if any(not bool(stream.attrs.get(name, False)) for name in required_true):
            raise ValueError("V26 replay is not finalized with exact 4D/strict-success evidence")
        if bool(stream.attrs.get("segmentation_is_policy_input", True)):
            raise ValueError("V26 replay illegally exposes segmentation to the policy")
        self._preflight_source_contract(stream)
        trace_json = _attribute_text(
            stream["admission"].attrs,
            "strict_success_trace_audit_json",
        )
        trace = json.loads(trace_json)
        if not isinstance(trace, dict) or not bool(trace.get("all_recurrences_exact", False)):
            raise ValueError("V26 strict-success trace audit is invalid")
        if not bool(trace.get("exact_three_second_evidence", False)):
            raise ValueError("V26 replay lacks exact three-second evidence")

        required = {
            "observation/rgb_wrist": 4,
            "observation/depth_wrist_mm": 3,
            "observation/depth_valid_mask_wrist": 3,
            "observation/camera_pose_wrist": 2,
            "observation/camera_intrinsics_wrist": 3,
            "observation/camera_4d_reconstructable_mask": 1,
            "observation/camera_geometry_alignment_exact": 1,
            "observation/joint_state": 2,
            "observation/previous_executed_action": 2,
            "action/submitted_joint_action": 2,
            "timing/camera_device_time_seconds": 1,
            "timing/camera_host_receive_time_seconds": 1,
            "language/row_instruction_en": 1,
            "language/row_instruction_zh": 1,
            "index/episode_id": 1,
            "index/source_episode_id": 1,
            "index/episode_step_id": 1,
            "admission/act_action_supervision_mask": 1,
            "outcome/execution_attempted": 1,
            "outcome/current_submission_safety_unchanged": 1,
        }
        row_count: int | None = None
        for name, rank in required.items():
            dataset = _require_dataset(stream, name)
            if dataset.ndim != rank:
                raise ValueError(f"V26 replay dataset rank changed: {name}")
            if row_count is None:
                row_count = int(dataset.shape[0])
            elif dataset.shape[0] != row_count:
                raise ValueError("V26 replay datasets are not row aligned")
        if row_count is None or row_count < 1:
            raise ValueError("V26 replay contains no rows")
        if stream["observation/rgb_wrist"].dtype != np.uint8:
            raise ValueError("V26 replay wrist RGB must be uint8")
        if stream["observation/depth_wrist_mm"].dtype != np.uint16:
            raise ValueError("V26 replay wrist depth must be uint16 millimetres")
        if stream["observation/joint_state"].shape[1:] != (12,):
            raise ValueError("V26 replay reported servo state must have 12 values")
        if stream["action/submitted_joint_action"].shape[1:] != (6,):
            raise ValueError("V26 replay submitted action must have six joints")
        submitted = stream["action/submitted_joint_action"]
        if (
            _attribute_text(submitted.attrs, "policy_target_semantics")
            != "current post-task-guard, pre-transport normalized command"
        ):
            raise ValueError("V26 replay submitted-action target semantics changed")
        if not bool(
            submitted.attrs.get(
                "action_supervision_requires_current_submission_safety_unchanged",
                False,
            )
        ):
            raise ValueError("V26 replay lost the current-submission safety-label gate")

    def _validate_index_vectors(
        self,
        action_admission: np.ndarray,
        execution_attempted: np.ndarray,
        submission_unchanged: np.ndarray,
    ) -> None:
        row_count = len(self.episode_ids)
        vectors = (
            self.source_episode_ids,
            self.episode_step_ids,
            action_admission,
            execution_attempted,
            submission_unchanged,
        )
        if any(value.shape != (row_count,) for value in vectors):
            raise ValueError("V26 replay index/admission vectors are misaligned")
        if any(
            value.dtype != np.bool_
            for value in (
                action_admission,
                execution_attempted,
                submission_unchanged,
            )
        ):
            raise ValueError("V26 replay admission/execution/safety masks must be boolean")
        if np.any(action_admission & ~execution_attempted):
            raise ValueError("V26 action supervision includes an unexecuted command")
        if np.any(action_admission & ~submission_unchanged):
            raise ValueError("V26 action supervision includes a safety-modified command")
        actions_only = np.flatnonzero(action_admission)
        if len(actions_only) and not np.array_equal(
            self.source_episode_ids[actions_only],
            self.source_episode_ids[actions_only].astype(np.int64),
        ):
            raise ValueError("V26 source episode identity changed")

    def _build_episode_bounds(self) -> dict[int, tuple[int, int]]:
        unique = np.unique(self.episode_ids)
        if not np.array_equal(unique, np.arange(len(unique), dtype=np.int64)):
            raise ValueError("V26 replay episode ids must be contiguous from zero")
        bounds: dict[int, tuple[int, int]] = {}
        for episode_id in unique:
            rows = np.flatnonzero(self.episode_ids == episode_id)
            if not np.array_equal(rows, np.arange(rows[0], rows[-1] + 1)):
                raise ValueError("V26 replay episode rows are not contiguous")
            expected_steps = np.arange(len(rows), dtype=np.int64)
            if not np.array_equal(self.episode_step_ids[rows], expected_steps):
                raise ValueError("V26 replay episode step ids are discontinuous")
            if np.any(self.source_episode_ids[rows] != self.source_episode_ids[rows[0]]):
                raise ValueError("V26 source episode identity changes within an episode")
            bounds[int(episode_id)] = (int(rows[0]), int(rows[-1]) + 1)
        return bounds

    def _open(self) -> h5py.File:
        if self._stream is None:
            self._stream = h5py.File(self.storage_path, "r", swmr=True)
        return self._stream

    def __len__(self) -> int:
        return len(self.anchor_rows)

    def _history_rows(self, row: int, steps: int) -> tuple[np.ndarray, np.ndarray]:
        episode_id = int(self.episode_ids[row])
        episode_start, _episode_end = self._episode_bounds[episode_id]
        first = max(episode_start, row - steps + 1)
        source = np.arange(first, row + 1, dtype=np.int64)
        destination = np.arange(steps - len(source), steps, dtype=np.int64)
        return source, destination

    def _language(self, stream: h5py.File, row: int) -> tuple[np.ndarray, np.ndarray]:
        if self.language_mode == "english":
            key = "language/row_instruction_en"
        elif self.language_mode == "chinese":
            key = "language/row_instruction_zh"
        else:
            key = (
                "language/row_instruction_en"
                if int(self.source_episode_ids[row]) % 2 == 0
                else "language/row_instruction_zh"
            )
        text = _decode(stream[key][row])
        encoded_length = len(text.encode("utf-8")) + 2
        if encoded_length > self.model_config.language_max_tokens:
            raise ValueError(
                "V26 dynamic instruction would be truncated by utf8-byte-v1; increase language_max_tokens"
            )
        return encode_language(text, max_length=self.model_config.language_max_tokens)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = int(self.anchor_rows[index])
        stream = self._open()
        cfg = self.model_config
        history_source, history_destination = self._history_rows(
            row,
            cfg.visual_history_steps,
        )
        rgb_source = stream["observation/rgb_wrist"]
        height, width = int(rgb_source.shape[1]), int(rgb_source.shape[2])
        visual_mask = np.zeros(cfg.visual_history_steps, dtype=bool)
        visual_mask[history_destination] = True
        rgb = np.zeros((cfg.visual_history_steps, height, width, 3), dtype=np.uint8)
        depth_m = np.zeros((cfg.visual_history_steps, height, width), dtype=np.float32)
        depth_valid = np.zeros((cfg.visual_history_steps, height, width), dtype=bool)
        camera_pose = np.zeros((cfg.visual_history_steps, 12), dtype=np.float32)
        reconstructable = np.zeros(cfg.visual_history_steps, dtype=bool)
        geometry_exact = np.zeros(cfg.visual_history_steps, dtype=bool)
        rgb[history_destination] = rgb_source[history_source]
        depth_mm = np.asarray(
            stream["observation/depth_wrist_mm"][history_source],
            dtype=np.uint16,
        )
        depth_m[history_destination] = depth_mm.astype(np.float32) / 1_000.0
        depth_valid[history_destination] = stream["observation/depth_valid_mask_wrist"][history_source]
        camera_pose[history_destination] = stream["observation/camera_pose_wrist"][history_source]
        reconstructable[history_destination] = stream["observation/camera_4d_reconstructable_mask"][
            history_source
        ]
        geometry_exact[history_destination] = stream["observation/camera_geometry_alignment_exact"][
            history_source
        ]

        device_absolute = np.asarray(
            stream["timing/camera_device_time_seconds"][history_source],
            dtype=np.float64,
        )
        host_absolute = np.asarray(
            stream["timing/camera_host_receive_time_seconds"][history_source],
            dtype=np.float64,
        )
        if np.any(np.diff(device_absolute) < -1.0e-12) or np.any(np.diff(host_absolute) < -1.0e-12):
            raise RuntimeError("V26 replay camera clocks run backwards within an episode")
        device_time = np.zeros(cfg.visual_history_steps, dtype=np.float32)
        host_time = np.zeros(cfg.visual_history_steps, dtype=np.float32)
        device_time[history_destination] = (device_absolute - device_absolute[-1]).astype(np.float32)
        host_time[history_destination] = (host_absolute - host_absolute[-1]).astype(np.float32)

        proprio_source, proprio_destination = self._history_rows(
            row,
            cfg.proprio_history_steps,
        )
        proprio_mask = np.zeros(cfg.proprio_history_steps, dtype=bool)
        proprio_mask[proprio_destination] = True
        joint_history = np.zeros((cfg.proprio_history_steps, 12), dtype=np.float32)
        action_history = np.zeros((cfg.proprio_history_steps, 6), dtype=np.float32)
        joint_history[proprio_destination] = stream["observation/joint_state"][proprio_source]
        action_history[proprio_destination] = stream["observation/previous_executed_action"][proprio_source]

        action_chunk = np.zeros((cfg.action_chunk_size, 6), dtype=np.float32)
        action_chunk_mask = np.zeros(cfg.action_chunk_size, dtype=bool)
        _episode_start, episode_end = self._episode_bounds[int(self.episode_ids[row])]
        for offset, target_row in enumerate(range(row, min(row + cfg.action_chunk_size, episode_end))):
            if not self._action_admission[target_row]:
                break
            action = np.asarray(
                stream["action/submitted_joint_action"][target_row],
                dtype=np.float32,
            )
            if action.shape != (6,) or not np.all(np.isfinite(action)):
                raise RuntimeError("V26 admitted target action is invalid")
            if np.any(np.abs(action) > 1.0 + 1.0e-6):
                raise RuntimeError("V26 admitted target action escaped normalized bounds")
            action_chunk[offset] = np.clip(action, -1.0, 1.0)
            action_chunk_mask[offset] = True
        if not action_chunk_mask[0]:
            raise RuntimeError("V26 dataset anchor lost its current action supervision")

        language_tokens, language_mask = self._language(stream, row)
        policy_inputs = {
            "language_token_ids": torch.from_numpy(language_tokens.copy()),
            "language_attention_mask": torch.from_numpy(language_mask.copy()),
            "rgb_wrist_window": torch.from_numpy(rgb),
            "depth_wrist_m_window": torch.from_numpy(depth_m),
            "depth_valid_mask_window": torch.from_numpy(depth_valid),
            "camera_pose_wrist_window": torch.from_numpy(camera_pose),
            "camera_intrinsics": torch.from_numpy(
                np.asarray(
                    stream["observation/camera_intrinsics_wrist"][row],
                    dtype=np.float32,
                )
            ),
            "visual_history_mask": torch.from_numpy(visual_mask),
            "camera_4d_reconstructable_mask_window": torch.from_numpy(reconstructable),
            "camera_geometry_alignment_exact_window": torch.from_numpy(geometry_exact),
            "camera_device_time_delta_s_window": torch.from_numpy(device_time),
            "camera_host_time_delta_s_window": torch.from_numpy(host_time),
            "robot_state": torch.from_numpy(
                np.asarray(stream["observation/joint_state"][row], dtype=np.float32)
            ),
            "joint_history": torch.from_numpy(joint_history),
            "action_history": torch.from_numpy(action_history),
            "proprio_history_mask": torch.from_numpy(proprio_mask),
        }
        if frozenset(policy_inputs) != SPARSE_4D_VLA_ACT_INPUT_KEYS_V26:
            raise RuntimeError("V26 dataset/model policy schemas diverged")
        return {
            "format": SPARSE_4D_VLA_DATASET_FORMAT_V26,
            "source_type": SIM_RL_SCRATCH_SOURCE_V26,
            "policy_inputs": policy_inputs,
            "action_chunk": torch.from_numpy(action_chunk),
            "action_chunk_mask": torch.from_numpy(action_chunk_mask),
            "row_index": torch.tensor(row, dtype=torch.int64),
            "episode_id": torch.tensor(int(self.episode_ids[row]), dtype=torch.int64),
            "source_episode_id": torch.tensor(
                int(self.source_episode_ids[row]),
                dtype=torch.int64,
            ),
        }

    def close(self) -> None:
        if self._stream is not None:
            try:
                self._stream.close()
            finally:
                self._stream = None

    def __getstate__(self) -> dict[str, Any]:
        self.close()
        state = dict(self.__dict__)
        state["_stream"] = None
        return state

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


__all__ = [
    "SPARSE_4D_VLA_DATASET_FORMAT_V26",
    "Sparse4DVLAReplayDatasetV26",
]
