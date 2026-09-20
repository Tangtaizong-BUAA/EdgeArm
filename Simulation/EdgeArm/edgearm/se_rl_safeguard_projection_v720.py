"""Action-aliasing evidence for SE-RL with a projected safety safeguard.

The policy proposes a normalized task-frame action.  The runtime then applies
the contact filter, DLS projection, and the unchanged V4 joint guard before a
command reaches MuJoCo.  The live tool displacement observed after ``env.step``
also contains plant lag and the controller's previously latched target; it is
therefore an *effect*, not the closest safe action used by the safeguard.

V720 keeps both quantities.  ``applied_action`` remains the measured causal
effect used by controller-state and rollout audits.  The action-aliasing
penalty uses the DLS-predicted, guard-scaled safe task action, matching the
SE-RL penalty ``w ||u - u_safe||^2`` without pretending that a delayed plant
must reproduce a new request within one control step.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


SE_RL_SAFEGUARD_PROJECTION_FORMAT_V720 = (
    "edgearm-v720-se-rl-safeguard-projection-penalty-v1"
)


@dataclass(frozen=True)
class SafeguardProjectedTaskActionV720:
    action: np.ndarray
    guard_selected_scale: float
    requested_to_safe_l2: float
    valid: bool
    format: str = SE_RL_SAFEGUARD_PROJECTION_FORMAT_V720

    def validate(self) -> None:
        action = np.asarray(self.action)
        scalars = np.asarray(
            [self.guard_selected_scale, self.requested_to_safe_l2],
            dtype=np.float64,
        )
        if (
            action.shape != (3,)
            or not np.all(np.isfinite(action))
            or np.any(np.abs(action) > 1.0 + 1.0e-6)
            or not np.all(np.isfinite(scalars))
            or not 0.0 <= self.guard_selected_scale <= 1.0
            or self.requested_to_safe_l2 < 0.0
            or type(self.valid) is not bool
        ):
            raise ValueError("V720 safeguard-projected task action is invalid")


def safeguard_projected_task_action_v720(
    translated: Any,
) -> SafeguardProjectedTaskActionV720:
    """Recover the safe normalized task action before plant dynamics.

    ``predicted_local_delta_m`` is the DLS prediction for the selected task
    request.  The V4 guard may further scale the submitted joint delta, so its
    selected scale is applied in the same local linearization.  This quantity
    deliberately excludes the post-step tool displacement.
    """

    projection = getattr(translated, "dls_projection", None)
    if projection is None:
        raise ValueError("V720 translation lacks a DLS projection")
    predicted_local = np.asarray(
        getattr(projection, "predicted_local_delta_m", None),
        dtype=np.float64,
    )
    translation_scale = np.asarray(
        getattr(translated, "translation_scale_xyz_m", None),
        dtype=np.float64,
    )
    requested = np.asarray(
        getattr(translated, "requested_task_action", None),
        dtype=np.float64,
    )
    guard_scale = float(getattr(translated, "guard_selected_scale", float("nan")))
    valid = bool(
        getattr(translated, "ik_converged", False)
        and getattr(translated, "guard_safe_candidate", False)
    )
    if (
        predicted_local.shape != (3,)
        or translation_scale.shape != (3,)
        or requested.shape != (3,)
        or not np.all(
            np.isfinite(
                np.concatenate((predicted_local, translation_scale, requested))
            )
        )
        or np.any(translation_scale <= 0.0)
        or not np.isfinite(guard_scale)
        or not 0.0 <= guard_scale <= 1.0
    ):
        raise ValueError("V720 translation projection metadata is invalid")
    safe_action = np.clip(
        predicted_local / translation_scale * guard_scale,
        -1.0,
        1.0,
    ).astype(np.float32)
    result = SafeguardProjectedTaskActionV720(
        action=safe_action,
        guard_selected_scale=guard_scale,
        requested_to_safe_l2=float(np.linalg.norm(requested - safe_action)),
        valid=valid,
    )
    result.validate()
    return result


def squared_safeguard_intervention_penalty_v720(
    action: np.ndarray,
    safe_action: np.ndarray,
    valid: np.ndarray,
    *,
    coefficient: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``w ||u-u_safe||^2`` and its unweighted distance.

    Legacy rows have no recoverable pre-plant safe action.  Their ``valid``
    bit is false and they receive no fabricated action-aliasing penalty.
    """

    proposed = np.asarray(action, dtype=np.float32)
    projected = np.asarray(safe_action, dtype=np.float32)
    mask = np.asarray(valid, dtype=bool)
    count = len(proposed)
    if (
        proposed.shape != (count, 3)
        or projected.shape != proposed.shape
        or mask.shape != (count,)
        or not np.all(np.isfinite(proposed))
        or not np.all(np.isfinite(projected))
        or not np.isfinite(coefficient)
        or coefficient < 0.0
    ):
        raise ValueError("V720 safeguard penalty inputs are invalid")
    distance = np.linalg.norm(proposed - projected, axis=-1).astype(
        np.float32
    )
    penalty = (
        np.float32(coefficient)
        * np.square(distance, dtype=np.float32)
        * mask.astype(np.float32)
    )
    return penalty.astype(np.float32), distance


__all__ = [
    "SE_RL_SAFEGUARD_PROJECTION_FORMAT_V720",
    "SafeguardProjectedTaskActionV720",
    "safeguard_projected_task_action_v720",
    "squared_safeguard_intervention_penalty_v720",
]
