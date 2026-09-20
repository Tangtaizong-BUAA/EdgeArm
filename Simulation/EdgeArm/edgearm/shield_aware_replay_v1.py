"""Shield-feasibility labels layered over exact V13 causal replay.

The base replay remains byte-verified V13 ``sim_rl_scratch`` data.  This
module adds only outcomes already emitted by the online IK and V13 safety
guard: whether the proposed task-frame action was executable and its applied
scale.  These are training-only RL transition labels, not expert actions or
behavior-cloning targets.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np

from .contact_prioritized_replay_v1 import (
    ContactPrioritizedReplayBatchV1,
    ContactPrioritizedReplayConfigV1,
    ContactPrioritizedReplayStoreV1,
)


SHIELD_AWARE_REPLAY_FORMAT_V1 = (
    "edgearm-v15-v13-contact-replay-with-executability-labels-v1"
)


def _required_vector(
    stream: h5py.File,
    path: str,
    *,
    count: int,
) -> np.ndarray:
    if path not in stream:
        raise ValueError(f"shield-aware replay source is missing {path}")
    value = np.asarray(stream[path][:])
    if value.shape != (count,):
        raise ValueError(f"shield-aware replay {path} shape changed")
    return value


@dataclass(frozen=True)
class ShieldAwareReplayBatchV1:
    base: ContactPrioritizedReplayBatchV1
    source_action_feasible: np.ndarray
    source_ik_converged: np.ndarray
    source_guard_safe_candidate: np.ndarray
    source_application_scale: np.ndarray
    format: str = SHIELD_AWARE_REPLAY_FORMAT_V1

    def validate(self) -> None:
        self.base.validate()
        count = len(self.base.source_row_index)
        for name in (
            "source_action_feasible",
            "source_ik_converged",
            "source_guard_safe_candidate",
        ):
            value = np.asarray(getattr(self, name))
            if value.shape != (count,) or value.dtype != np.dtype(bool):
                raise ValueError(f"shield-aware replay {name} must be boolean")
        scale = np.asarray(self.source_application_scale)
        if (
            scale.shape != (count,)
            or scale.dtype != np.float32
            or not np.all(np.isfinite(scale))
            or np.any(scale < 0.0)
            or np.any(scale > 1.0)
        ):
            raise ValueError(
                "shield-aware replay application scale escaped finite [0,1]"
            )
        expected = (
            self.base.source_execution_attempted
            & self.source_ik_converged
            & self.source_guard_safe_candidate
            & (scale > np.float32(0.0))
        )
        if not np.array_equal(self.source_action_feasible, expected):
            raise ValueError("shield-aware replay feasibility identity changed")
        if np.any(
            self.base.source_shield_rejected_before_step
            & self.source_action_feasible
        ):
            raise ValueError("a pre-step shield rejection cannot be feasible")
        if self.format != SHIELD_AWARE_REPLAY_FORMAT_V1:
            raise ValueError("shield-aware replay batch format changed")


class ShieldAwareReplayStoreV1:
    """V13 contact replay plus immutable online executability outcomes."""

    def __init__(
        self,
        *,
        base: ContactPrioritizedReplayStoreV1,
        action_feasible: np.ndarray,
        ik_converged: np.ndarray,
        guard_safe_candidate: np.ndarray,
        application_scale: np.ndarray,
    ) -> None:
        count = base.transition_count
        boolean = {
            "action_feasible": np.asarray(action_feasible, dtype=bool),
            "ik_converged": np.asarray(ik_converged, dtype=bool),
            "guard_safe_candidate": np.asarray(
                guard_safe_candidate, dtype=bool
            ),
        }
        for name, value in boolean.items():
            if value.shape != (count,):
                raise ValueError(f"shield-aware store {name} shape changed")
        scale = np.asarray(application_scale, dtype=np.float32)
        if (
            scale.shape != (count,)
            or not np.all(np.isfinite(scale))
            or np.any(scale < 0.0)
            or np.any(scale > 1.0)
        ):
            raise ValueError("shield-aware store scale is invalid")
        expected = (
            base._arrays["execution_attempted"]
            & boolean["ik_converged"]
            & boolean["guard_safe_candidate"]
            & (scale > np.float32(0.0))
        )
        if not np.array_equal(boolean["action_feasible"], expected):
            raise ValueError("shield-aware store feasibility label changed")
        self.base = base
        self.action_feasible = boolean["action_feasible"]
        self.ik_converged = boolean["ik_converged"]
        self.guard_safe_candidate = boolean["guard_safe_candidate"]
        self.application_scale = scale
        self.transition_count = count
        self.format = SHIELD_AWARE_REPLAY_FORMAT_V1

    @classmethod
    def from_h5(
        cls,
        paths: list[Path] | tuple[Path, ...],
        config: ContactPrioritizedReplayConfigV1 | None = None,
    ) -> ShieldAwareReplayStoreV1:
        base = ContactPrioritizedReplayStoreV1.from_h5(paths, config)
        ik_rows: list[np.ndarray] = []
        guard_rows: list[np.ndarray] = []
        scale_rows: list[np.ndarray] = []
        for path in base.source_paths:
            with h5py.File(path, "r") as stream:
                count = int(len(stream["policy_observation/rgb_frames"]))
                ik_rows.append(
                    _required_vector(
                        stream,
                        "execution/ik_converged",
                        count=count,
                    ).astype(bool, copy=False)
                )
                guard_rows.append(
                    _required_vector(
                        stream,
                        "execution/guard_safe_candidate",
                        count=count,
                    ).astype(bool, copy=False)
                )
                scale_rows.append(
                    _required_vector(
                        stream,
                        "execution/ik_application_scale",
                        count=count,
                    ).astype(np.float32, copy=False)
                )
        ik = np.concatenate(ik_rows)
        guard = np.concatenate(guard_rows)
        scale = np.concatenate(scale_rows)
        feasible = (
            base._arrays["execution_attempted"]
            & ik
            & guard
            & (scale > np.float32(0.0))
        )
        return cls(
            base=base,
            action_feasible=feasible,
            ik_converged=ik,
            guard_safe_candidate=guard,
            application_scale=scale,
        )

    def sample(self, *, batch_size: int, seed: int) -> ShieldAwareReplayBatchV1:
        base_batch = self.base.sample(batch_size=batch_size, seed=seed)
        rows = base_batch.source_row_index
        result = ShieldAwareReplayBatchV1(
            base=base_batch,
            source_action_feasible=self.action_feasible[rows].copy(),
            source_ik_converged=self.ik_converged[rows].copy(),
            source_guard_safe_candidate=self.guard_safe_candidate[rows].copy(),
            source_application_scale=self.application_scale[rows].copy(),
        )
        result.validate()
        return result

    def manifest(self) -> dict[str, object]:
        payload = dict(self.base.manifest())
        payload.update(
            {
                "format": SHIELD_AWARE_REPLAY_FORMAT_V1,
                "base_replay_format": self.base.manifest()["format"],
                "action_feasible_transition_count": int(
                    np.count_nonzero(self.action_feasible)
                ),
                "action_infeasible_transition_count": int(
                    np.count_nonzero(~self.action_feasible)
                ),
                "action_feasible_fraction": float(
                    np.mean(self.action_feasible)
                ),
                "executability_label_source": (
                    "V13 online IK convergence plus guard-safe positive-scale "
                    "execution outcome"
                ),
                "expert_action_labels": 0,
                "behavior_cloning_targets": 0,
                "actor_input": False,
                "production_admission": False,
            }
        )
        return payload


__all__ = [
    "SHIELD_AWARE_REPLAY_FORMAT_V1",
    "ShieldAwareReplayBatchV1",
    "ShieldAwareReplayStoreV1",
]
