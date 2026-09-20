"""Outcome-only self-imitation for sustained contact transport.

The auxiliary target is the agent's own measured applied action on an
original-goal transition that was safe, in valid contact, and reduced the
object-to-target distance.  Hindsight-relabeled rows are excluded because the
recorded action was not selected for their substituted goal.  No expert,
scripted action, route, waypoint, or future policy target is used.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F


POSITIVE_EFFECT_TRANSPORT_SELF_IMITATION_FORMAT_V707 = (
    "edgearm-v707-positive-effect-contact-transport-self-imitation-v1"
)


@dataclass(frozen=True)
class PositiveEffectTransportSelfImitationConfigV707:
    coefficient: float = 0.0
    minimum_progress_m: float = 1.0e-5
    full_progress_weight_m: float = 7.5e-4

    def validate(self) -> None:
        values = np.asarray(
            [
                self.coefficient,
                self.minimum_progress_m,
                self.full_progress_weight_m,
            ],
            dtype=np.float64,
        )
        if (
            not np.all(np.isfinite(values))
            or self.coefficient < 0.0
            or self.minimum_progress_m <= 0.0
            or self.full_progress_weight_m < self.minimum_progress_m
        ):
            raise ValueError("V707 transport self-imitation config is invalid")


def positive_effect_transport_self_imitation_loss_v707(
    predicted_action: torch.Tensor,
    applied_action: torch.Tensor,
    original_task_progress_m: torch.Tensor,
    valid_contact: torch.Tensor,
    her_relabelled: torch.Tensor,
    action_feasible: torch.Tensor,
    safety_violation: torch.Tensor,
    importance_weight: torch.Tensor,
    *,
    config: PositiveEffectTransportSelfImitationConfigV707,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Return a differentiable contact-transport loss and selection audit."""

    if type(config) is not PositiveEffectTransportSelfImitationConfigV707:
        raise TypeError("V707 requires its exact transport self-imitation config")
    config.validate()
    count = predicted_action.shape[0]
    vectors = (predicted_action, applied_action)
    scalars = (
        original_task_progress_m,
        valid_contact,
        her_relabelled,
        action_feasible,
        safety_violation,
        importance_weight,
    )
    if (
        any(value.shape != (count, 3) for value in vectors)
        or any(value.shape != (count,) for value in scalars)
        or any(not bool(torch.isfinite(value).all().item()) for value in (*vectors, *scalars))
    ):
        raise ValueError("V707 transport self-imitation tensors are invalid")

    selected = (
        (original_task_progress_m >= config.minimum_progress_m)
        & (valid_contact > 0.5)
        & (her_relabelled < 0.5)
        & (safety_violation < 0.5)
    )
    target = torch.clamp(applied_action.detach(), min=-1.0, max=1.0)
    progress_weight = torch.clamp(
        original_task_progress_m / config.full_progress_weight_m,
        min=0.0,
        max=1.0,
    )
    weight = (
        selected.to(predicted_action.dtype)
        * progress_weight
        * importance_weight
    )
    per_row = F.smooth_l1_loss(
        predicted_action,
        target,
        reduction="none",
    ).mean(dim=-1)
    weight_sum = weight.sum()
    unscaled_loss = torch.where(
        weight_sum > 0.0,
        (weight * per_row).sum() / weight_sum.clamp_min(1.0e-8),
        predicted_action.sum() * 0.0,
    )
    scaled_loss = config.coefficient * unscaled_loss
    selected_count = int(selected.sum().detach().cpu().item())
    selected_infeasible_count = int(
        (selected & (action_feasible <= 0.5)).sum().detach().cpu().item()
    )
    selected_progress = original_task_progress_m[selected]
    audit = {
        "format": POSITIVE_EFFECT_TRANSPORT_SELF_IMITATION_FORMAT_V707,
        "configuration": asdict(config),
        "batch_size": count,
        "selected_transition_count": selected_count,
        "selected_transition_fraction": selected_count / count,
        "selected_originally_infeasible_transition_fraction": (
            selected_infeasible_count / selected_count
            if selected_count
            else 0.0
        ),
        "mean_selected_progress_m": (
            float(selected_progress.mean().detach().cpu().item())
            if selected_count
            else 0.0
        ),
        "unscaled_loss": float(unscaled_loss.detach().cpu().item()),
        "scaled_loss": float(scaled_loss.detach().cpu().item()),
        "target_is_clipped_measured_applied_action": True,
        "original_goal_transition_required": True,
        "valid_contact_required": True,
        "positive_measured_object_progress_required": True,
        "verified_safe_execution_required": True,
        "original_proposal_feasibility_required": False,
        "external_demonstration_used": False,
        "expert_action_used": False,
        "waypoint_or_path_used": False,
        "act_training_started": False,
        "production_admission": False,
    }
    return scaled_loss, audit


__all__ = [
    "POSITIVE_EFFECT_TRANSPORT_SELF_IMITATION_FORMAT_V707",
    "PositiveEffectTransportSelfImitationConfigV707",
    "positive_effect_transport_self_imitation_loss_v707",
]
