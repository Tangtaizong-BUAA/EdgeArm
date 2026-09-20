"""Causal sparse-4D ACT samples from the legacy keyboard RGB-D batch.

The August keyboard batch is useful operator demonstration data, but its
environment terminated after a six-step hold rather than the current 90-step
contract and its deferred render did not persist an exact per-row camera pose.
This adapter therefore exposes it only as an explicitly admitted, low-weight
``sim_human`` warm-start source.  It never upgrades those episodes to current
strict success and masks the unavailable 4D geometry path while retaining the
aligned wrist RGB-D, language, servo state, action history, and action labels.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal
import zlib

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from .sparse_4d_vla_act_v26 import (
    SPARSE_4D_VLA_ACT_INPUT_KEYS_V26,
    Sparse4DVLAConfigV26,
)
from .trisource_contract_v26 import SIM_HUMAN_SOURCE_V26
from .vla_data import LANGUAGE_VOCAB_SIZE, encode_language


LEGACY_KEYBOARD_SPARSE_4D_DATASET_FORMAT_V40 = (
    "edgearm-v40-legacy-keyboard-sim-human-sparse-4d-warmstart-v1"
)
LEGACY_KEYBOARD_RGBD_FORMAT_V40 = "edgearm-phone-sim-wrist-rgbd-v1"
CURRENT_STRICT_HOLD_STEPS_V40 = 90
LanguageModeV40 = Literal["english", "chinese", "bilingual_by_seed"]


def _text(value: object) -> str:
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def _dataset(stream: h5py.File, name: str) -> h5py.Dataset:
    value = stream.get(name)
    if not isinstance(value, h5py.Dataset):
        raise ValueError(f"V40 keyboard RGB-D is missing dataset: {name}")
    return value


def _step_info(dataset: h5py.Dataset, row: int) -> dict[str, Any]:
    value = json.loads(zlib.decompress(bytes(dataset[row])))
    if not isinstance(value, dict):
        raise TypeError("V40 keyboard step info must decode to an object")
    return value


def _source_hold_steps(info: dict[str, Any]) -> int:
    realism_v10 = info.get("realism_v10")
    if isinstance(realism_v10, dict):
        configuration = realism_v10.get("configuration")
        if isinstance(configuration, dict):
            value = configuration.get("strict_success_hold_steps")
            if type(value) is int and value > 0:
                return value
    realism_v6 = info.get("realism_v6")
    if isinstance(realism_v6, dict):
        contract = realism_v6.get("strict_success")
        if isinstance(contract, dict):
            value = contract.get("hold_steps")
            if type(value) is int and value > 0:
                return value
    raise ValueError("V40 keyboard source omitted its historical hold contract")


def _source_max_joint_delta(info: dict[str, Any]) -> float:
    for key in ("realism_v13", "realism_v10"):
        realism = info.get(key)
        if not isinstance(realism, dict):
            continue
        configuration = realism.get("configuration")
        if not isinstance(configuration, dict):
            continue
        value = configuration.get("max_joint_delta")
        if isinstance(value, (int, float)) and np.isfinite(value) and value > 0.0:
            return float(value)
    raise ValueError("V40 keyboard source omitted max_joint_delta action units")


def _unchanged_submission(transport: dict[str, Any]) -> bool:
    return bool(
        not transport.get("submitted_command_ingress_lost", False)
        and not any(bool(value) for value in transport.get("submitted_action_changed_mask", ()))
        and not any(
            bool(value)
            for value in transport.get("submitted_command_target_changed_mask", ())
        )
    )


class LegacyKeyboardSparse4DVLADatasetV40(Dataset[dict[str, Any]]):
    """Lazy samples from one user-operated synthetic wrist RGB-D episode."""

    def __init__(
        self,
        rgbd_path: Path,
        *,
        model_config: Sparse4DVLAConfigV26 | None = None,
        language_mode: LanguageModeV40 = "bilingual_by_seed",
        operator_demonstration_admitted: bool = False,
        source_loss_weight: float = 0.25,
    ) -> None:
        self.rgbd_path = Path(rgbd_path).expanduser().resolve()
        self.model_config = model_config or Sparse4DVLAConfigV26()
        self.language_mode = language_mode
        self.operator_demonstration_admitted = operator_demonstration_admitted
        self.source_loss_weight = float(source_loss_weight)
        self._stream: h5py.File | None = None
        if not self.rgbd_path.is_file():
            raise FileNotFoundError(self.rgbd_path)
        if self.model_config.language_vocabulary_size != LANGUAGE_VOCAB_SIZE:
            raise ValueError("V40 keyboard adapter requires utf8-byte-v1")
        if language_mode not in {"english", "chinese", "bilingual_by_seed"}:
            raise ValueError("V40 keyboard language mode is unsupported")
        if type(operator_demonstration_admitted) is not bool:
            raise TypeError("V40 operator admission must be an exact boolean")
        if not np.isfinite(self.source_loss_weight) or not 0.0 < self.source_loss_weight <= 1.0:
            raise ValueError("V40 keyboard source weight must be in (0,1]")
        if not operator_demonstration_admitted:
            raise ValueError(
                "V40 legacy keyboard actions require explicit operator-demonstration admission"
            )

        with h5py.File(self.rgbd_path, "r", swmr=True) as stream:
            self._preflight(stream)
            self.row_count = int(stream.attrs["committed_rows"])
            self.episode_seed = int(stream.attrs["episode_seed"])
            self.instruction_en = _text(stream.attrs["task"])
            self.instruction_zh = _text(stream.attrs["task_zh"])
            self.intrinsics, self.constant_camera_pose = self._camera_metadata(stream)
            (
                self.action_admission,
                self.previous_executed_action,
                self.strict_success_streak,
                self.source_hold_steps,
            ) = self._audit_rows(stream)
            self.anchor_rows = np.flatnonzero(self.action_admission).astype(np.int64)
            if not len(self.anchor_rows):
                raise ValueError("V40 keyboard episode has no unchanged action labels")
            self.episode_ids = np.zeros(self.row_count, dtype=np.int64)
            self.source_episode_ids = np.full(
                self.row_count,
                self.episode_seed,
                dtype=np.int64,
            )
            self.episode_step_ids = np.arange(self.row_count, dtype=np.int64)
            self.current_strict_success_qualified = bool(
                self.source_hold_steps == CURRENT_STRICT_HOLD_STEPS_V40
                and np.max(self.strict_success_streak, initial=0)
                >= CURRENT_STRICT_HOLD_STEPS_V40
            )

    def _preflight(self, stream: h5py.File) -> None:
        if _text(stream.attrs.get("schema_version", "")) != LEGACY_KEYBOARD_RGBD_FORMAT_V40:
            raise ValueError("V40 keyboard adapter requires the exact legacy RGB-D format")
        required_true = (
            "finalized",
            "episode_success",
            "simulated_wrist_rgbd",
            "eligible_for_synthetic_act_training",
            "eligible_for_synthetic_vla_training",
        )
        if any(not bool(stream.attrs.get(name, False)) for name in required_true):
            raise ValueError("V40 keyboard source lost its finalized synthetic-demo claims")
        if _text(stream.attrs.get("operator_input_source", "")) != "keyboard":
            raise ValueError("V40 legacy adapter accepts keyboard demonstrations only")
        if bool(stream.attrs.get("physical_capture_claimed", True)):
            raise ValueError("V40 legacy keyboard source cannot claim physical capture")
        if int(stream.attrs.get("physical_samples", -1)) != 0:
            raise ValueError("V40 legacy keyboard source unexpectedly contains physical samples")
        rows = int(stream.attrs.get("committed_rows", -1))
        required = {
            "steps/wrist_rgb": (4, np.dtype(np.uint8)),
            "steps/depth_wrist_mm": (3, np.dtype(np.uint16)),
            "steps/q_before_rad": (2, np.dtype(np.float64)),
            "steps/q_after_rad": (2, np.dtype(np.float64)),
            "steps/dq_before_rad_s": (2, np.dtype(np.float64)),
            "steps/submitted_normalized_action": (2, np.dtype(np.float64)),
            "steps/capture_monotonic_ns": (1, np.dtype(np.int64)),
            "steps/completed_monotonic_ns": (1, np.dtype(np.int64)),
            "steps/row_committed": (1, np.dtype(np.bool_)),
            "steps/step_info_json_zlib": (1, None),
        }
        for name, (rank, dtype) in required.items():
            value = _dataset(stream, name)
            if value.ndim != rank or value.shape[0] != rows:
                raise ValueError(f"V40 keyboard dataset is not row aligned: {name}")
            if dtype is not None and value.dtype != dtype:
                raise ValueError(f"V40 keyboard dataset dtype changed: {name}")
        if (
            stream["steps/q_before_rad"].shape[1:] != (6,)
            or stream["steps/q_after_rad"].shape[1:] != (6,)
        ):
            raise ValueError("V40 keyboard q must contain six joints")
        if stream["steps/submitted_normalized_action"].shape[1:] != (6,):
            raise ValueError("V40 keyboard action must contain six joints")
        if not np.all(np.asarray(stream["steps/row_committed"], dtype=bool)):
            raise ValueError("V40 keyboard source contains an uncommitted row")

    def _camera_metadata(self, stream: h5py.File) -> tuple[np.ndarray, np.ndarray]:
        value = json.loads(_text(stream.attrs.get("camera_metadata_json", "{}")))
        wrist = value.get("wrist") if isinstance(value, dict) else None
        if not isinstance(wrist, dict):
            raise ValueError("V40 keyboard source omitted wrist camera metadata")
        intrinsics = np.asarray(wrist.get("intrinsics"), dtype=np.float32)
        position = np.asarray(wrist.get("world_position"), dtype=np.float32)
        rotation = np.asarray(wrist.get("world_rotation"), dtype=np.float32)
        if intrinsics.shape != (3, 3) or position.shape != (3,) or rotation.shape != (3, 3):
            raise ValueError("V40 keyboard camera metadata is malformed")
        pose = np.concatenate((position, rotation.reshape(-1))).astype(np.float32)
        return intrinsics, pose

    def _audit_rows(
        self,
        stream: h5py.File,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
        steps = stream["steps"]
        raw_actions = np.asarray(steps["submitted_normalized_action"], dtype=np.float64)
        if not np.all(np.isfinite(raw_actions)) or np.any(np.abs(raw_actions) > 1.0 + 1.0e-6):
            raise ValueError("V40 keyboard source has an invalid normalized action")
        admitted = np.zeros(self.row_count, dtype=bool)
        contained = np.zeros(self.row_count, dtype=bool)
        settled = np.zeros(self.row_count, dtype=bool)
        infos = steps["step_info_json_zlib"]
        source_hold_steps: int | None = None
        max_joint_delta: float | None = None
        for row in range(self.row_count):
            info = _step_info(infos, row)
            if row == 0:
                source_hold_steps = _source_hold_steps(info)
                max_joint_delta = _source_max_joint_delta(info)
            transport = info.get("sim2real_v2")
            realism_v6 = info.get("realism_v6")
            if not isinstance(transport, dict) or not isinstance(realism_v6, dict):
                raise ValueError(f"V40 keyboard audit fields are missing at row {row}")
            submitted = np.asarray(transport.get("submitted_action"), dtype=np.float64)
            actually_applied = np.asarray(
                transport.get("actually_applied_delayed_action"),
                dtype=np.float64,
            )
            if submitted.shape != (6,) or actually_applied.shape != (6,):
                raise ValueError(f"V40 keyboard command chain is malformed at row {row}")
            if not bool(transport.get("submitted_command_ingress_lost", False)) and not np.allclose(
                submitted,
                raw_actions[row],
                rtol=0.0,
                atol=2.0e-7,
            ):
                raise RuntimeError(f"V40 keyboard raw/submitted action differs at row {row}")
            if not np.all(np.isfinite(actually_applied)):
                raise ValueError(f"V40 keyboard applied target delta is non-finite at row {row}")
            admitted[row] = bool(
                _unchanged_submission(transport)
                and not bool(info.get("terminal_failure", False))
            )
            contained[row] = bool(realism_v6.get("strict_contained", False))
            settled[row] = bool(realism_v6.get("strict_settled", False))
        if source_hold_steps is None or max_joint_delta is None:
            raise RuntimeError("V40 keyboard source action contract was not resolved")
        q_before = np.asarray(steps["q_before_rad"], dtype=np.float64)
        q_after = np.asarray(steps["q_after_rad"], dtype=np.float64)
        executed = (q_after - q_before) / max_joint_delta
        if not np.all(np.isfinite(executed)):
            raise RuntimeError("V40 keyboard reported executed joint delta is non-finite")
        maximum_executed = float(np.max(np.abs(executed), initial=0.0))
        # The old force-limited plant can coast slightly beyond one commanded
        # max-delta in a frame.  This is measured q motion, not an action label;
        # saturate it as a bounded history feature while retaining an audit.
        if maximum_executed > 1.5:
            raise RuntimeError(
                "V40 keyboard reported joint motion is too large for normalized history"
            )
        self.action_history_preclip_max_abs = maximum_executed
        self.action_history_clipped_value_count = int(
            np.count_nonzero(np.abs(executed) > 1.0)
        )
        executed = np.clip(executed, -1.0, 1.0).astype(np.float32)
        previous = np.zeros_like(executed)
        previous[1:] = executed[:-1]
        streak = np.zeros(self.row_count, dtype=np.int64)
        running = 0
        for row in range(self.row_count):
            running = running + 1 if contained[row] and settled[row] else 0
            streak[row] = running
        return admitted, previous, streak, source_hold_steps

    def _open(self) -> h5py.File:
        if self._stream is None:
            self._stream = h5py.File(self.rgbd_path, "r", swmr=True)
        return self._stream

    def __len__(self) -> int:
        return len(self.anchor_rows)

    def _history_rows(self, row: int, steps: int) -> tuple[np.ndarray, np.ndarray]:
        first = max(0, row - steps + 1)
        source = np.arange(first, row + 1, dtype=np.int64)
        destination = np.arange(steps - len(source), steps, dtype=np.int64)
        return source, destination

    def _language(self) -> tuple[np.ndarray, np.ndarray]:
        text = (
            self.instruction_en
            if self.language_mode == "english"
            or (self.language_mode == "bilingual_by_seed" and self.episode_seed % 2 == 0)
            else self.instruction_zh
        )
        if len(text.encode("utf-8")) + 2 > self.model_config.language_max_tokens:
            raise ValueError("V40 keyboard instruction exceeds configured language length")
        return encode_language(text, max_length=self.model_config.language_max_tokens)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = int(self.anchor_rows[index])
        stream = self._open()
        cfg = self.model_config
        steps = stream["steps"]
        visual_source, visual_destination = self._history_rows(row, cfg.visual_history_steps)
        height, width = steps["wrist_rgb"].shape[1:3]
        visual_mask = np.zeros(cfg.visual_history_steps, dtype=bool)
        visual_mask[visual_destination] = True
        rgb = np.zeros((cfg.visual_history_steps, height, width, 3), dtype=np.uint8)
        depth_m = np.zeros((cfg.visual_history_steps, height, width), dtype=np.float32)
        depth_valid = np.zeros((cfg.visual_history_steps, height, width), dtype=bool)
        pose = np.zeros((cfg.visual_history_steps, 12), dtype=np.float32)
        rgb[visual_destination] = steps["wrist_rgb"][visual_source]
        depth_mm = np.asarray(steps["depth_wrist_mm"][visual_source], dtype=np.uint16)
        depth_m[visual_destination] = depth_mm.astype(np.float32) / 1_000.0
        depth_valid[visual_destination] = depth_mm > 0
        pose[visual_destination] = self.constant_camera_pose
        capture = np.asarray(
            steps["capture_monotonic_ns"][visual_source],
            dtype=np.float64,
        ) / 1.0e9
        completed = np.asarray(
            steps["completed_monotonic_ns"][visual_source],
            dtype=np.float64,
        ) / 1.0e9
        if np.any(np.diff(capture) < -1.0e-12) or np.any(np.diff(completed) < -1.0e-12):
            raise RuntimeError("V40 keyboard clocks run backwards")
        device_time = np.zeros(cfg.visual_history_steps, dtype=np.float32)
        host_time = np.zeros(cfg.visual_history_steps, dtype=np.float32)
        device_time[visual_destination] = (capture - capture[-1]).astype(np.float32)
        host_time[visual_destination] = (completed - completed[-1]).astype(np.float32)

        proprio_source, proprio_destination = self._history_rows(
            row,
            cfg.proprio_history_steps,
        )
        proprio_mask = np.zeros(cfg.proprio_history_steps, dtype=bool)
        proprio_mask[proprio_destination] = True
        q = np.asarray(steps["q_before_rad"][proprio_source], dtype=np.float32)
        dq = np.asarray(steps["dq_before_rad_s"][proprio_source], dtype=np.float32)
        joint_history = np.zeros((cfg.proprio_history_steps, 12), dtype=np.float32)
        joint_history[proprio_destination] = np.concatenate((q, dq), axis=1)
        action_history = np.zeros((cfg.proprio_history_steps, 6), dtype=np.float32)
        action_history[proprio_destination] = self.previous_executed_action[proprio_source]

        action_chunk = np.zeros((cfg.action_chunk_size, 6), dtype=np.float32)
        action_chunk_mask = np.zeros(cfg.action_chunk_size, dtype=bool)
        for offset, target_row in enumerate(
            range(row, min(row + cfg.action_chunk_size, self.row_count))
        ):
            if not self.action_admission[target_row]:
                break
            action_chunk[offset] = np.clip(
                np.asarray(
                    steps["submitted_normalized_action"][target_row],
                    dtype=np.float32,
                ),
                -1.0,
                1.0,
            )
            action_chunk_mask[offset] = True
        if not action_chunk_mask[0]:
            raise RuntimeError("V40 keyboard anchor lost current action supervision")
        language_tokens, language_mask = self._language()
        current_q = np.asarray(steps["q_before_rad"][row], dtype=np.float32)
        current_dq = np.asarray(steps["dq_before_rad_s"][row], dtype=np.float32)
        policy_inputs = {
            "language_token_ids": torch.from_numpy(language_tokens.copy()),
            "language_attention_mask": torch.from_numpy(language_mask.copy()),
            "rgb_wrist_window": torch.from_numpy(rgb),
            "depth_wrist_m_window": torch.from_numpy(depth_m),
            "depth_valid_mask_window": torch.from_numpy(depth_valid),
            "camera_pose_wrist_window": torch.from_numpy(pose),
            "camera_intrinsics": torch.from_numpy(self.intrinsics.copy()),
            "visual_history_mask": torch.from_numpy(visual_mask),
            "camera_4d_reconstructable_mask_window": torch.zeros(
                cfg.visual_history_steps,
                dtype=torch.bool,
            ),
            "camera_geometry_alignment_exact_window": torch.zeros(
                cfg.visual_history_steps,
                dtype=torch.bool,
            ),
            "camera_device_time_delta_s_window": torch.from_numpy(device_time),
            "camera_host_time_delta_s_window": torch.from_numpy(host_time),
            "robot_state": torch.from_numpy(np.concatenate((current_q, current_dq))),
            "joint_history": torch.from_numpy(joint_history),
            "action_history": torch.from_numpy(action_history),
            "proprio_history_mask": torch.from_numpy(proprio_mask),
        }
        if frozenset(policy_inputs) != SPARSE_4D_VLA_ACT_INPUT_KEYS_V26:
            raise RuntimeError("V40 keyboard/model policy schemas diverged")
        return {
            "format": LEGACY_KEYBOARD_SPARSE_4D_DATASET_FORMAT_V40,
            "source_type": SIM_HUMAN_SOURCE_V26,
            "policy_inputs": policy_inputs,
            "action_chunk": torch.from_numpy(action_chunk),
            "action_chunk_mask": torch.from_numpy(action_chunk_mask),
            "source_loss_weight": torch.tensor(
                self.source_loss_weight,
                dtype=torch.float32,
            ),
            "current_strict_success_qualified": torch.tensor(
                self.current_strict_success_qualified,
                dtype=torch.bool,
            ),
            "row_index": torch.tensor(row, dtype=torch.int64),
            "episode_id": torch.tensor(0, dtype=torch.int64),
            "source_episode_id": torch.tensor(self.episode_seed, dtype=torch.int64),
        }

    def audit(self) -> dict[str, Any]:
        return {
            "format": LEGACY_KEYBOARD_SPARSE_4D_DATASET_FORMAT_V40,
            "path": str(self.rgbd_path),
            "source_type": SIM_HUMAN_SOURCE_V26,
            "episode_seed": self.episode_seed,
            "row_count": self.row_count,
            "admitted_action_row_count": int(np.count_nonzero(self.action_admission)),
            "source_loss_weight": self.source_loss_weight,
            "source_hold_steps": self.source_hold_steps,
            "required_current_hold_steps": CURRENT_STRICT_HOLD_STEPS_V40,
            "maximum_reconstructed_strict_streak_steps": int(
                np.max(self.strict_success_streak, initial=0)
            ),
            "current_strict_success_qualified": self.current_strict_success_qualified,
            "operator_demonstration_admitted": self.operator_demonstration_admitted,
            "action_history_semantics": (
                "previous reported q_after-minus-q_before normalized by source max_joint_delta, "
                "then saturated to [-1,1] as a history feature"
            ),
            "action_history_preclip_max_abs": self.action_history_preclip_max_abs,
            "action_history_clipped_value_count": self.action_history_clipped_value_count,
            "camera_4d_reconstructable": False,
            "camera_geometry_alignment_exact": False,
            "physical_samples": 0,
            "production_admission": False,
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
    "CURRENT_STRICT_HOLD_STEPS_V40",
    "LEGACY_KEYBOARD_SPARSE_4D_DATASET_FORMAT_V40",
    "LegacyKeyboardSparse4DVLADatasetV40",
]
