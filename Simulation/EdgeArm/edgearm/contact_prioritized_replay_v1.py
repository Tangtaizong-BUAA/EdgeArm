"""Contact/progress-window replay for safe V13 visual scratch trajectories.

The V13 PPO collector is intentionally on-policy, so a rare useful contact is
consumed once and then discarded.  This module builds a source-closed replay
view over V13 H5 artifacts.  It expands sparse contact, block-progress, strict
success, and pre-step shield events into causal temporal windows and samples
those windows with explicit strata and importance weights.

No expert action, behavior-cloning label, physical sample, or future frame is
introduced.  Time-limit rows are conservatively non-bootstrapping because the
current V13 artifact does not persist the post-action RGB observation at an
episode boundary.  That limitation is part of the replay identity rather than
being hidden by crossing into the next reset episode.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from .asymmetric_multiview_ppo_v1 import (
    ACTION_DIM,
    H5_FORMAT,
    JOINT_STATE_DIM,
    POLICY_ACTION_DIM,
    POLICY_FORMAT,
    SOURCE_TYPE,
    VISUAL_GEOMETRY_TARGET_DIM,
    VIEW_NAMES,
)
from .privileged_effect_state_v1 import PRIVILEGED_EFFECT_STATE_DIM


CONTACT_PRIORITIZED_REPLAY_FORMAT_V1 = (
    "edgearm-v13-contact-progress-shield-window-prioritized-replay-v1"
)
REPLAY_STRATA_V1 = (
    "strict_success_window",
    "valid_contact_window",
    "block_progress_window",
    "shield_rejection_window",
    "background",
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _attribute_text(value: object) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _required_dataset(stream: h5py.File, path: str) -> np.ndarray:
    if path not in stream:
        raise ValueError(f"V13 replay source is missing {path}")
    return np.asarray(stream[path][:])


@dataclass(frozen=True)
class ContactPrioritizedReplayConfigV1:
    history_steps: int = 4
    n_step: int = 3
    gamma: float = 0.99
    importance_beta: float = 0.4
    contact_window_before_steps: int = 32
    contact_window_after_steps: int = 8
    progress_window_before_steps: int = 20
    progress_window_after_steps: int = 6
    shield_window_before_steps: int = 16
    minimum_block_progress_m: float = 0.00005
    minimum_potential_progress: float = 0.02
    valid_contact_intrinsic_bonus: float = 1.5
    maximum_block_displacement_bonus: float = 1.0
    block_displacement_bonus_scale_m: float = 0.001
    maximum_positive_potential_bonus: float = 0.15
    positive_potential_bonus_scale: float = 0.10
    strict_success_fraction: float = 0.10
    valid_contact_fraction: float = 0.35
    block_progress_fraction: float = 0.20
    shield_rejection_fraction: float = 0.10
    background_fraction: float = 0.25

    def validate(self) -> None:
        for name in (
            "history_steps",
            "n_step",
            "contact_window_before_steps",
            "contact_window_after_steps",
            "progress_window_before_steps",
            "progress_window_after_steps",
            "shield_window_before_steps",
        ):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"replay {name} must be a non-negative integer")
        if self.history_steps < 1 or self.n_step < 1:
            raise ValueError("replay history_steps and n_step must be positive")
        numeric = np.asarray(
            [
                self.gamma,
                self.importance_beta,
                self.minimum_block_progress_m,
                self.minimum_potential_progress,
                self.valid_contact_intrinsic_bonus,
                self.maximum_block_displacement_bonus,
                self.block_displacement_bonus_scale_m,
                self.maximum_positive_potential_bonus,
                self.positive_potential_bonus_scale,
                *self.stratum_fractions.values(),
            ],
            dtype=np.float64,
        )
        if not np.all(np.isfinite(numeric)):
            raise ValueError("replay configuration contains non-finite values")
        if not 0.0 < self.gamma <= 1.0:
            raise ValueError("replay gamma must be in (0,1]")
        if not 0.0 <= self.importance_beta <= 1.0:
            raise ValueError("replay importance_beta must be in [0,1]")
        if self.minimum_block_progress_m <= 0.0:
            raise ValueError("replay block-progress threshold must be positive")
        if self.minimum_potential_progress <= 0.0:
            raise ValueError("replay potential-progress threshold must be positive")
        for name in (
            "valid_contact_intrinsic_bonus",
            "maximum_block_displacement_bonus",
            "maximum_positive_potential_bonus",
        ):
            if getattr(self, name) < 0.0:
                raise ValueError(f"replay {name} must be non-negative")
        if (
            self.block_displacement_bonus_scale_m <= 0.0
            or self.positive_potential_bonus_scale <= 0.0
        ):
            raise ValueError("replay intrinsic bonus scales must be positive")
        maximum_n_step_bonus = self.n_step * (
            self.valid_contact_intrinsic_bonus
            + self.maximum_block_displacement_bonus
            + self.maximum_positive_potential_bonus
        )
        if maximum_n_step_bonus >= 16.0:
            raise ValueError(
                "replay intrinsic bonus bound must stay below 16 reward units"
            )
        fractions = np.asarray(tuple(self.stratum_fractions.values()))
        if np.any(fractions < 0.0) or not np.isclose(
            float(fractions.sum()), 1.0, rtol=0.0, atol=1.0e-12
        ):
            raise ValueError("replay stratum fractions must be non-negative and sum to one")

    @property
    def stratum_fractions(self) -> dict[str, float]:
        return {
            "strict_success_window": self.strict_success_fraction,
            "valid_contact_window": self.valid_contact_fraction,
            "block_progress_window": self.block_progress_fraction,
            "shield_rejection_window": self.shield_rejection_fraction,
            "background": self.background_fraction,
        }


@dataclass(frozen=True)
class ContactPrioritizedReplayBatchV1:
    rgb_history: np.ndarray
    joint_history: np.ndarray
    action_history: np.ndarray
    history_valid: np.ndarray
    view_history_valid: np.ndarray
    task_ids: np.ndarray
    previous_policy_pre_tanh: np.ndarray
    privileged_state: np.ndarray
    visual_geometry_target: np.ndarray
    replay_action: np.ndarray
    source_n_step_reward: np.ndarray
    intrinsic_n_step_bonus: np.ndarray
    n_step_reward: np.ndarray
    bootstrap_discount: np.ndarray
    bootstrap_rgb_history: np.ndarray
    bootstrap_joint_history: np.ndarray
    bootstrap_action_history: np.ndarray
    bootstrap_history_valid: np.ndarray
    bootstrap_view_history_valid: np.ndarray
    bootstrap_task_ids: np.ndarray
    bootstrap_previous_policy_pre_tanh: np.ndarray
    bootstrap_privileged_state: np.ndarray
    importance_weight: np.ndarray
    sampling_probability: np.ndarray
    source_row_index: np.ndarray
    bootstrap_row_index: np.ndarray
    n_step_count: np.ndarray
    terminal_within_horizon: np.ndarray
    source_execution_attempted: np.ndarray
    source_shield_rejected_before_step: np.ndarray
    sampled_stratum: np.ndarray
    format: str = CONTACT_PRIORITIZED_REPLAY_FORMAT_V1

    def validate(self) -> None:
        count, history, views, height, width, channels = self.rgb_history.shape
        shapes = {
            "joint_history": (count, history, JOINT_STATE_DIM),
            "action_history": (count, history, ACTION_DIM),
            "history_valid": (count, history),
            "view_history_valid": (count, history, len(VIEW_NAMES)),
            "task_ids": (count,),
            "previous_policy_pre_tanh": (count, POLICY_ACTION_DIM),
            "privileged_state": (count, PRIVILEGED_EFFECT_STATE_DIM),
            "visual_geometry_target": (count, VISUAL_GEOMETRY_TARGET_DIM),
            "replay_action": (count, POLICY_ACTION_DIM),
            "source_n_step_reward": (count,),
            "intrinsic_n_step_bonus": (count,),
            "n_step_reward": (count,),
            "bootstrap_discount": (count,),
            "bootstrap_rgb_history": (
                count,
                history,
                views,
                height,
                width,
                channels,
            ),
            "bootstrap_joint_history": (count, history, JOINT_STATE_DIM),
            "bootstrap_action_history": (count, history, ACTION_DIM),
            "bootstrap_history_valid": (count, history),
            "bootstrap_view_history_valid": (
                count,
                history,
                len(VIEW_NAMES),
            ),
            "bootstrap_task_ids": (count,),
            "bootstrap_previous_policy_pre_tanh": (
                count,
                POLICY_ACTION_DIM,
            ),
            "bootstrap_privileged_state": (
                count,
                PRIVILEGED_EFFECT_STATE_DIM,
            ),
            "importance_weight": (count,),
            "sampling_probability": (count,),
            "source_row_index": (count,),
            "bootstrap_row_index": (count,),
            "n_step_count": (count,),
            "terminal_within_horizon": (count,),
            "source_execution_attempted": (count,),
            "source_shield_rejected_before_step": (count,),
            "sampled_stratum": (count,),
        }
        if (
            count < 1
            or history < 1
            or views != len(VIEW_NAMES)
            or channels != 3
            or self.rgb_history.dtype != np.uint8
            or self.bootstrap_rgb_history.dtype != np.uint8
        ):
            raise ValueError("replay batch image dimensions or dtype are invalid")
        for name, shape in shapes.items():
            if np.asarray(getattr(self, name)).shape != shape:
                raise ValueError(f"replay batch {name} shape mismatch")
        for name in (
            "joint_history",
            "action_history",
            "previous_policy_pre_tanh",
            "privileged_state",
            "visual_geometry_target",
            "replay_action",
            "source_n_step_reward",
            "intrinsic_n_step_bonus",
            "n_step_reward",
            "bootstrap_discount",
            "bootstrap_joint_history",
            "bootstrap_action_history",
            "bootstrap_previous_policy_pre_tanh",
            "bootstrap_privileged_state",
            "importance_weight",
            "sampling_probability",
        ):
            value = np.asarray(getattr(self, name))
            if value.dtype != np.float32 or not np.all(np.isfinite(value)):
                raise ValueError(f"replay batch {name} must be finite float32")
        for name in (
            "history_valid",
            "view_history_valid",
            "bootstrap_history_valid",
            "bootstrap_view_history_valid",
            "terminal_within_horizon",
            "source_execution_attempted",
            "source_shield_rejected_before_step",
        ):
            if np.asarray(getattr(self, name)).dtype != np.dtype(bool):
                raise ValueError(f"replay batch {name} must be boolean")
        for name in (
            "task_ids",
            "bootstrap_task_ids",
            "source_row_index",
            "bootstrap_row_index",
            "n_step_count",
        ):
            if np.asarray(getattr(self, name)).dtype != np.dtype(np.int64):
                raise ValueError(f"replay batch {name} must be int64")
        if self.sampled_stratum.dtype.kind not in {"U", "S"}:
            raise ValueError("replay sampled_stratum must be a string vector")
        if any(value not in REPLAY_STRATA_V1 for value in self.sampled_stratum):
            raise ValueError("replay batch contains an unknown stratum")
        if np.any(self.importance_weight <= 0.0) or np.any(
            self.importance_weight > 1.0 + 1.0e-6
        ):
            raise ValueError("replay importance weights escaped (0,1]")
        if np.any(self.sampling_probability <= 0.0) or np.any(
            self.sampling_probability > 1.0
        ):
            raise ValueError("replay sampling probabilities escaped (0,1]")
        if np.any(self.n_step_count < 1):
            raise ValueError("replay n-step counts must be positive")
        if np.any(self.intrinsic_n_step_bonus < 0.0):
            raise ValueError("replay intrinsic bonus must be non-negative")
        if not np.allclose(
            self.n_step_reward,
            self.source_n_step_reward + self.intrinsic_n_step_bonus,
            rtol=1.0e-6,
            atol=1.0e-6,
        ):
            raise ValueError("replay learning reward differs from source plus intrinsic")
        if np.any(self.bootstrap_discount < 0.0) or np.any(
            self.bootstrap_discount > 1.0
        ):
            raise ValueError("replay bootstrap discount escaped [0,1]")
        if np.any(
            self.terminal_within_horizon
            != (self.bootstrap_row_index < 0)
        ):
            raise ValueError("replay terminal mask and bootstrap row disagree")
        if np.any(
            self.source_execution_attempted
            & self.source_shield_rejected_before_step
        ):
            raise ValueError("replay source cannot execute and reject one action")
        if self.format != CONTACT_PRIORITIZED_REPLAY_FORMAT_V1:
            raise ValueError("replay batch format changed")


class ContactPrioritizedReplayStoreV1:
    """Immutable in-memory index over one or more exact V13 online rollouts."""

    def __init__(
        self,
        *,
        config: ContactPrioritizedReplayConfigV1,
        source_paths: tuple[Path, ...],
        source_sha256: tuple[str, ...],
        arrays: dict[str, np.ndarray],
        history_row_indices: np.ndarray,
        episode_key: np.ndarray,
        next_row_index: np.ndarray,
    ) -> None:
        config.validate()
        self.config = config
        self.source_paths = source_paths
        self.source_sha256 = source_sha256
        self._arrays = arrays
        self.history_row_indices = history_row_indices
        self.episode_key = episode_key
        self.next_row_index = next_row_index
        self.transition_count = int(len(episode_key))
        if self.transition_count < 1:
            raise ValueError("replay store cannot be empty")
        self.event_masks = self._build_event_masks()
        specialized = np.zeros(self.transition_count, dtype=bool)
        for name in REPLAY_STRATA_V1[:-1]:
            specialized |= self.event_masks[name]
        self.event_masks["background"] = ~specialized
        self.pools = {
            name: np.flatnonzero(self.event_masks[name]).astype(np.int64)
            for name in REPLAY_STRATA_V1
        }

    @classmethod
    def _validate_source_contract(cls, stream: h5py.File) -> None:
        """Fail closed on the exact immutable V13 source contract.

        Versioned mixed replay subclasses may override this hook with another
        explicit allowlist.  The V1 public loader remains V13-only.
        """

        if _attribute_text(stream.attrs.get("format", "")) != H5_FORMAT:
            raise ValueError("replay accepts only exact V13 H5 artifacts")
        if _attribute_text(stream.attrs.get("source_type", "")) != SOURCE_TYPE:
            raise ValueError("replay source_type changed")
        if _attribute_text(stream.attrs.get("policy_format", "")) != POLICY_FORMAT:
            raise ValueError("replay policy_format changed")
        if bool(stream.attrs.get("production_admission", True)):
            raise ValueError("replay source must remain non-production")
        if int(stream.attrs.get("expert_calls", -1)) != 0:
            raise ValueError("replay source unexpectedly contains expert calls")
        if int(stream.attrs.get("physical_samples", -1)) != 0:
            raise ValueError("replay source unexpectedly contains physical samples")

    @classmethod
    def from_h5(
        cls,
        paths: list[Path] | tuple[Path, ...],
        config: ContactPrioritizedReplayConfigV1 | None = None,
    ) -> ContactPrioritizedReplayStoreV1:
        selected_config = config or ContactPrioritizedReplayConfigV1()
        selected_config.validate()
        resolved = tuple(Path(path).expanduser().resolve() for path in paths)
        if not resolved or len(set(resolved)) != len(resolved):
            raise ValueError("replay H5 paths must be a non-empty unique sequence")

        collected: dict[str, list[np.ndarray]] = {}
        global_histories: list[np.ndarray] = []
        episode_keys: list[np.ndarray] = []
        next_indices: list[np.ndarray] = []
        hashes: list[str] = []
        offset = 0
        image_shape: tuple[int, ...] | None = None
        history_width: int | None = None
        dataset_paths = {
            "rgb_frames": "policy_observation/rgb_frames",
            "joint_state": "policy_observation/joint_state",
            "previous_executed_action": (
                "policy_observation/previous_executed_action"
            ),
            "view_valid": "policy_observation/view_valid",
            "task_ids": "policy_observation/task_ids",
            "previous_policy_pre_tanh": (
                "policy_observation/previous_policy_pre_tanh"
            ),
            "policy_action": "execution/policy_action",
            "execution_attempted": "execution/execution_attempted",
            "shield_rejected_before_step": (
                "execution/shield_rejected_before_step"
            ),
            "valid_push_side_contact_any": (
                "execution/valid_push_side_contact_any"
            ),
            "step_block_displacement_m": (
                "execution/step_block_displacement_m"
            ),
            "episode_ids": "execution/episode_ids",
            "shaped_rewards": "reward_and_outcome/shaped_rewards",
            "potential_before": "reward_and_outcome/potential_before",
            "potential_after": "reward_and_outcome/potential_after",
            "terminated": "reward_and_outcome/terminated",
            "truncated": "reward_and_outcome/truncated",
            "strict_success": "reward_and_outcome/strict_success",
            "terminal_failure": "reward_and_outcome/terminal_failure",
            "safety_stop": "reward_and_outcome/safety_stop",
            "privileged_state": "training_only/privileged_state",
            "next_privileged_state": "training_only/next_privileged_state",
            "visual_geometry_target": "training_only/visual_geometry_target",
        }
        for file_index, path in enumerate(resolved):
            if not path.is_file():
                raise FileNotFoundError(f"replay H5 source is missing: {path}")
            with h5py.File(path, "r") as stream:
                cls._validate_source_contract(stream)
                local = {
                    name: _required_dataset(stream, dataset_path)
                    for name, dataset_path in dataset_paths.items()
                }
                local_history = _required_dataset(
                    stream, "policy_observation/history_row_indices"
                ).astype(np.int64, copy=False)

            count = int(len(local["rgb_frames"]))
            if count < 1 or any(len(value) != count for value in local.values()):
                raise ValueError("replay H5 datasets have inconsistent row counts")
            if local["rgb_frames"].ndim != 5:
                raise ValueError("replay RGB source must be [N,V,H,W,C]")
            if image_shape is None:
                image_shape = tuple(local["rgb_frames"].shape[1:])
            elif tuple(local["rgb_frames"].shape[1:]) != image_shape:
                raise ValueError("replay RGB resolutions differ across artifacts")
            if local_history.ndim != 2:
                raise ValueError("replay history indices must be a matrix")
            if history_width is None:
                history_width = int(local_history.shape[1])
            elif local_history.shape[1] != history_width:
                raise ValueError("replay history widths differ across artifacts")
            if history_width != selected_config.history_steps:
                raise ValueError(
                    "replay configuration history_steps differs from the source"
                )

            local_episode = np.asarray(local["episode_ids"], dtype=np.int64)
            done = np.asarray(local["terminated"], dtype=bool) | np.asarray(
                local["truncated"], dtype=bool
            )
            if not bool(done[-1]):
                raise ValueError("replay artifact does not end at an episode boundary")
            global_history = np.full(local_history.shape, -1, dtype=np.int64)
            valid = local_history >= 0
            if np.any(local_history[valid] >= count):
                raise ValueError("replay history points outside its source artifact")
            global_history[valid] = local_history[valid] + offset
            for row in range(count):
                valid_local = local_history[row][local_history[row] >= 0]
                if valid_local.size < 1 or int(valid_local[-1]) != row:
                    raise ValueError("replay history does not end at its current row")
                if np.any(local_episode[valid_local] != local_episode[row]):
                    raise ValueError("replay history crosses an episode boundary")

            local_next = np.full(count, -1, dtype=np.int64)
            for row in range(count - 1):
                same_episode = local_episode[row + 1] == local_episode[row]
                if not done[row] and not same_episode:
                    raise ValueError("nonterminal replay row lost its next observation")
                if done[row] and same_episode:
                    raise ValueError("terminal replay row continues inside one episode")
                if not done[row]:
                    local_next[row] = offset + row + 1
            if not done[-1]:  # pragma: no cover - checked above
                raise ValueError("final replay row must be terminal")

            execution = np.asarray(local["execution_attempted"], dtype=bool)
            shield = np.asarray(local["shield_rejected_before_step"], dtype=bool)
            if np.any(execution & shield) or np.any(~execution & ~shield):
                raise ValueError("replay execution and shield masks are inconsistent")
            if np.any(shield & ~(done & np.asarray(local["terminal_failure"], dtype=bool))):
                raise ValueError("replay shield rejection is not a terminal failure")

            for name, value in local.items():
                collected.setdefault(name, []).append(value)
            global_histories.append(global_history)
            episode_keys.append(
                np.asarray(
                    file_index * 1_000_000_000 + local_episode,
                    dtype=np.int64,
                )
            )
            next_indices.append(local_next)
            hashes.append(_sha256_file(path))
            offset += count

        arrays = {
            name: np.concatenate(values, axis=0)
            for name, values in collected.items()
        }
        canonical_float = (
            "joint_state",
            "previous_executed_action",
            "previous_policy_pre_tanh",
            "policy_action",
            "step_block_displacement_m",
            "shaped_rewards",
            "potential_before",
            "potential_after",
            "privileged_state",
            "next_privileged_state",
            "visual_geometry_target",
        )
        for name in canonical_float:
            arrays[name] = np.asarray(arrays[name], dtype=np.float32)
            if not np.all(np.isfinite(arrays[name])):
                raise ValueError(f"replay source {name} contains non-finite values")
        for name in (
            "view_valid",
            "execution_attempted",
            "shield_rejected_before_step",
            "valid_push_side_contact_any",
            "terminated",
            "truncated",
            "strict_success",
            "terminal_failure",
            "safety_stop",
        ):
            arrays[name] = np.asarray(arrays[name], dtype=bool)
        arrays["rgb_frames"] = np.asarray(arrays["rgb_frames"], dtype=np.uint8)
        arrays["task_ids"] = np.asarray(arrays["task_ids"], dtype=np.int64)
        arrays["episode_ids"] = np.asarray(arrays["episode_ids"], dtype=np.int64)
        return cls(
            config=selected_config,
            source_paths=resolved,
            source_sha256=tuple(hashes),
            arrays=arrays,
            history_row_indices=np.concatenate(global_histories, axis=0),
            episode_key=np.concatenate(episode_keys, axis=0),
            next_row_index=np.concatenate(next_indices, axis=0),
        )

    def _window_mask(
        self,
        events: np.ndarray,
        before: int,
        after: int,
    ) -> np.ndarray:
        result = np.zeros(self.transition_count, dtype=bool)
        for event_index in np.flatnonzero(events):
            episode = self.episode_key[event_index]
            lower = max(0, int(event_index) - before)
            upper = min(self.transition_count, int(event_index) + after + 1)
            candidates = np.arange(lower, upper, dtype=np.int64)
            result[candidates[self.episode_key[candidates] == episode]] = True
        return result

    def _build_event_masks(self) -> dict[str, np.ndarray]:
        contact = self._arrays["valid_push_side_contact_any"]
        progress = (
            self._arrays["step_block_displacement_m"]
            >= np.float32(self.config.minimum_block_progress_m)
        ) | (
            self._arrays["potential_after"] - self._arrays["potential_before"]
            >= np.float32(self.config.minimum_potential_progress)
        )
        shield = self._arrays["shield_rejected_before_step"]
        success = self._arrays["strict_success"]
        return {
            "strict_success_window": self._window_mask(
                success,
                self.config.contact_window_before_steps,
                self.config.contact_window_after_steps,
            ),
            "valid_contact_window": self._window_mask(
                contact,
                self.config.contact_window_before_steps,
                self.config.contact_window_after_steps,
            ),
            "block_progress_window": self._window_mask(
                progress,
                self.config.progress_window_before_steps,
                self.config.progress_window_after_steps,
            ),
            "shield_rejection_window": self._window_mask(
                shield,
                self.config.shield_window_before_steps,
                0,
            ),
        }

    def stratum_counts(self) -> dict[str, int]:
        return {name: int(len(self.pools[name])) for name in REPLAY_STRATA_V1}

    def _history_packet(
        self,
        rows: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        batch = int(len(rows))
        history = self.config.history_steps
        _, views, height, width, channels = self._arrays["rgb_frames"].shape
        rgb = np.zeros(
            (batch, history, views, height, width, channels), dtype=np.uint8
        )
        joints = np.zeros((batch, history, JOINT_STATE_DIM), dtype=np.float32)
        actions = np.zeros((batch, history, ACTION_DIM), dtype=np.float32)
        valid = np.zeros((batch, history), dtype=bool)
        view_valid = np.zeros((batch, history, len(VIEW_NAMES)), dtype=bool)
        for batch_index, row in enumerate(rows):
            if row < 0:
                continue
            indices = self.history_row_indices[int(row)]
            selected = indices[indices >= 0]
            offset = history - len(selected)
            rgb[batch_index, offset:] = self._arrays["rgb_frames"][selected]
            joints[batch_index, offset:] = self._arrays["joint_state"][selected]
            actions[batch_index, offset:] = self._arrays[
                "previous_executed_action"
            ][selected]
            valid[batch_index, offset:] = True
            view_valid[batch_index, offset:] = self._arrays["view_valid"][selected]
        return rgb, joints, actions, valid, view_valid

    def _n_step_targets(
        self,
        roots: np.ndarray,
    ) -> tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ]:
        source_rewards = np.zeros(len(roots), dtype=np.float32)
        intrinsic_bonuses = np.zeros(len(roots), dtype=np.float32)
        learning_rewards = np.zeros(len(roots), dtype=np.float32)
        discounts = np.zeros(len(roots), dtype=np.float32)
        bootstrap_rows = np.full(len(roots), -1, dtype=np.int64)
        bootstrap_privileged = np.zeros(
            (len(roots), PRIVILEGED_EFFECT_STATE_DIM), dtype=np.float32
        )
        step_counts = np.zeros(len(roots), dtype=np.int64)
        gamma = np.float32(self.config.gamma)
        for batch_index, root in enumerate(roots):
            current = int(root)
            accumulated_source = np.float32(0.0)
            accumulated_intrinsic = np.float32(0.0)
            multiplier = np.float32(1.0)
            last = current
            terminal = False
            for _ in range(self.config.n_step):
                source_step = self._arrays["shaped_rewards"][current]
                displacement_bonus = np.float32(
                    self.config.maximum_block_displacement_bonus
                    * np.clip(
                        float(self._arrays["step_block_displacement_m"][current])
                        / self.config.block_displacement_bonus_scale_m,
                        0.0,
                        1.0,
                    )
                )
                potential_delta = float(
                    self._arrays["potential_after"][current]
                    - self._arrays["potential_before"][current]
                )
                potential_bonus = np.float32(
                    self.config.maximum_positive_potential_bonus
                    * np.clip(
                        max(potential_delta, 0.0)
                        / self.config.positive_potential_bonus_scale,
                        0.0,
                        1.0,
                    )
                )
                contact_bonus = np.float32(
                    self.config.valid_contact_intrinsic_bonus
                    if self._arrays["valid_push_side_contact_any"][current]
                    else 0.0
                )
                intrinsic_step = np.add(
                    np.add(contact_bonus, displacement_bonus, dtype=np.float32),
                    potential_bonus,
                    dtype=np.float32,
                )
                accumulated_source = np.add(
                    accumulated_source,
                    np.multiply(
                        multiplier,
                        source_step,
                        dtype=np.float32,
                    ),
                    dtype=np.float32,
                )
                accumulated_intrinsic = np.add(
                    accumulated_intrinsic,
                    np.multiply(
                        multiplier,
                        intrinsic_step,
                        dtype=np.float32,
                    ),
                    dtype=np.float32,
                )
                step_counts[batch_index] += 1
                last = current
                if bool(
                    self._arrays["terminated"][current]
                    or self._arrays["truncated"][current]
                ):
                    terminal = True
                    break
                next_row = int(self.next_row_index[current])
                if next_row < 0:
                    raise RuntimeError("replay nonterminal row lost its next row")
                multiplier = np.multiply(multiplier, gamma, dtype=np.float32)
                current = next_row
            source_rewards[batch_index] = accumulated_source
            intrinsic_bonuses[batch_index] = accumulated_intrinsic
            learning_rewards[batch_index] = np.add(
                accumulated_source,
                accumulated_intrinsic,
                dtype=np.float32,
            )
            bootstrap_privileged[batch_index] = self._arrays[
                "next_privileged_state"
            ][last]
            if not terminal:
                next_row = int(self.next_row_index[last])
                if next_row >= 0:
                    bootstrap_rows[batch_index] = next_row
                    discounts[batch_index] = multiplier
        return (
            source_rewards,
            intrinsic_bonuses,
            learning_rewards,
            discounts,
            bootstrap_rows,
            bootstrap_privileged,
            step_counts,
        )

    def sample(
        self,
        *,
        batch_size: int,
        seed: int,
    ) -> ContactPrioritizedReplayBatchV1:
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError("replay batch_size must be positive")
        if type(seed) is not int or seed < 0:
            raise ValueError("replay seed must be non-negative")
        rng = np.random.default_rng(seed)
        fractions = self.config.stratum_fractions
        desired = np.asarray(
            [fractions[name] * batch_size for name in REPLAY_STRATA_V1],
            dtype=np.float64,
        )
        quotas = np.floor(desired).astype(np.int64)
        remaining = batch_size - int(quotas.sum())
        order = np.argsort(-(desired - quotas), kind="stable")
        quotas[order[:remaining]] += 1

        draw_rows: list[np.ndarray] = []
        draw_strata: list[np.ndarray] = []
        draw_probabilities: list[np.ndarray] = []
        all_rows = np.arange(self.transition_count, dtype=np.int64)
        for stratum_index, stratum in enumerate(REPLAY_STRATA_V1):
            quota = int(quotas[stratum_index])
            if quota < 1:
                continue
            pool = self.pools[stratum]
            actual_stratum = stratum
            if len(pool) == 0:
                pool = all_rows
                actual_stratum = "background"
            chosen = rng.choice(pool, size=quota, replace=len(pool) < quota)
            draw_rows.append(np.asarray(chosen, dtype=np.int64))
            draw_strata.append(
                np.full(
                    quota,
                    actual_stratum,
                    dtype=f"<U{max(len(value) for value in REPLAY_STRATA_V1)}",
                )
            )
            draw_probabilities.append(
                np.full(
                    quota,
                    (quota / batch_size) / len(pool),
                    dtype=np.float64,
                )
            )
        roots = np.concatenate(draw_rows)
        strata = np.concatenate(draw_strata)
        probabilities = np.concatenate(draw_probabilities)
        permutation = rng.permutation(batch_size)
        roots = roots[permutation]
        strata = strata[permutation]
        probabilities = probabilities[permutation]
        weights = np.power(
            self.transition_count * probabilities,
            -self.config.importance_beta,
        )
        weights /= float(np.max(weights))

        current_packet = self._history_packet(roots)
        (
            source_n_step_reward,
            intrinsic_n_step_bonus,
            n_step_reward,
            bootstrap_discount,
            bootstrap_rows,
            bootstrap_privileged,
            n_step_count,
        ) = self._n_step_targets(roots)
        bootstrap_packet = self._history_packet(bootstrap_rows)
        valid_bootstrap = bootstrap_rows >= 0
        terminal_bootstrap = ~valid_bootstrap
        for current_value, bootstrap_value in zip(
            current_packet,
            bootstrap_packet,
        ):
            bootstrap_value[terminal_bootstrap] = current_value[
                terminal_bootstrap
            ]
        bootstrap_task = self._arrays["task_ids"][roots].astype(
            np.int64, copy=True
        )
        bootstrap_previous_pre_tanh = np.zeros(
            (batch_size, POLICY_ACTION_DIM), dtype=np.float32
        )
        bootstrap_previous_pre_tanh[terminal_bootstrap] = self._arrays[
            "previous_policy_pre_tanh"
        ][roots[terminal_bootstrap]]
        bootstrap_task[valid_bootstrap] = self._arrays["task_ids"][
            bootstrap_rows[valid_bootstrap]
        ]
        bootstrap_previous_pre_tanh[valid_bootstrap] = self._arrays[
            "previous_policy_pre_tanh"
        ][bootstrap_rows[valid_bootstrap]]
        batch = ContactPrioritizedReplayBatchV1(
            rgb_history=current_packet[0],
            joint_history=current_packet[1],
            action_history=current_packet[2],
            history_valid=current_packet[3],
            view_history_valid=current_packet[4],
            task_ids=self._arrays["task_ids"][roots].astype(np.int64),
            previous_policy_pre_tanh=self._arrays[
                "previous_policy_pre_tanh"
            ][roots].astype(np.float32),
            privileged_state=self._arrays["privileged_state"][roots].astype(
                np.float32
            ),
            visual_geometry_target=self._arrays["visual_geometry_target"][
                roots
            ].astype(np.float32),
            replay_action=self._arrays["policy_action"][roots].astype(np.float32),
            source_n_step_reward=source_n_step_reward,
            intrinsic_n_step_bonus=intrinsic_n_step_bonus,
            n_step_reward=n_step_reward,
            bootstrap_discount=bootstrap_discount,
            bootstrap_rgb_history=bootstrap_packet[0],
            bootstrap_joint_history=bootstrap_packet[1],
            bootstrap_action_history=bootstrap_packet[2],
            bootstrap_history_valid=bootstrap_packet[3],
            bootstrap_view_history_valid=bootstrap_packet[4],
            bootstrap_task_ids=bootstrap_task,
            bootstrap_previous_policy_pre_tanh=bootstrap_previous_pre_tanh,
            bootstrap_privileged_state=bootstrap_privileged,
            importance_weight=weights.astype(np.float32),
            sampling_probability=probabilities.astype(np.float32),
            source_row_index=roots,
            bootstrap_row_index=bootstrap_rows,
            n_step_count=n_step_count,
            terminal_within_horizon=(bootstrap_rows < 0),
            source_execution_attempted=self._arrays["execution_attempted"][roots],
            source_shield_rejected_before_step=self._arrays[
                "shield_rejected_before_step"
            ][roots],
            sampled_stratum=strata,
        )
        batch.validate()
        return batch

    def manifest(self) -> dict[str, Any]:
        return {
            "format": CONTACT_PRIORITIZED_REPLAY_FORMAT_V1,
            "source_type": SOURCE_TYPE,
            "source_h5_format": H5_FORMAT,
            "source_policy_format": POLICY_FORMAT,
            "source_paths": [str(path) for path in self.source_paths],
            "source_sha256": list(self.source_sha256),
            "transition_count": self.transition_count,
            "stratum_counts": self.stratum_counts(),
            "time_limit_bootstrap": False,
            "time_limit_bootstrap_reason": (
                "V13 does not persist post-action RGB at an episode boundary"
            ),
            "intrinsic_reward": {
                "valid_contact_bonus": self.config.valid_contact_intrinsic_bonus,
                "maximum_block_displacement_bonus": (
                    self.config.maximum_block_displacement_bonus
                ),
                "block_displacement_bonus_scale_m": (
                    self.config.block_displacement_bonus_scale_m
                ),
                "maximum_positive_potential_bonus": (
                    self.config.maximum_positive_potential_bonus
                ),
                "positive_potential_bonus_scale": (
                    self.config.positive_potential_bonus_scale
                ),
                "simulator_privileged_training_only": True,
                "actor_input": False,
                "source_reward_preserved_separately": True,
            },
            "expert_calls": 0,
            "physical_samples": 0,
            "production_admission": False,
        }


__all__ = [
    "CONTACT_PRIORITIZED_REPLAY_FORMAT_V1",
    "REPLAY_STRATA_V1",
    "ContactPrioritizedReplayBatchV1",
    "ContactPrioritizedReplayConfigV1",
    "ContactPrioritizedReplayStoreV1",
]
