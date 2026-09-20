"""Projection-aware self-imitation from the agent's own positive effects.

Safety projection can map many policy proposals to substantially different
executed actions.  V706 uses replay transitions whose *measured* applied
action safely reduced the current tool-goal distance.  The acquisition actor
is softly pulled toward that clipped applied action on full-acquisition rows.
An originally infeasible proposal remains eligible when the projected action
was verified safe and effective: these are precisely the action-aliasing rows
that the auxiliary loss is intended to correct.

No external demonstration, expert action, route, waypoint, future state, or
scripted controller target is used.  The auxiliary loss is disabled by
default and leaves the V694 update bit-for-bit unchanged at coefficient zero.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F


POSITIVE_EFFECT_SELF_IMITATION_FORMAT_V706 = (
    "edgearm-v706-positive-effect-projection-aware-self-imitation-v1"
)


@dataclass(frozen=True)
class PositiveEffectSelfImitationConfigV706:
    coefficient: float = 0.0
    minimum_progress_m: float = 1.0e-4
    full_progress_weight_m: float = 1.5e-3
    minimum_acquisition_gate: float = 0.99

    def validate(self) -> None:
        values = np.asarray(
            [
                self.coefficient,
                self.minimum_progress_m,
                self.full_progress_weight_m,
                self.minimum_acquisition_gate,
            ],
            dtype=np.float64,
        )
        if (
            not np.all(np.isfinite(values))
            or self.coefficient < 0.0
            or self.minimum_progress_m <= 0.0
            or self.full_progress_weight_m < self.minimum_progress_m
            or not 0.5 <= self.minimum_acquisition_gate <= 1.0
        ):
            raise ValueError("V706 positive-effect self-imitation config is invalid")


def positive_effect_self_imitation_loss_v706(
    predicted_acquisition_action: torch.Tensor,
    applied_action: torch.Tensor,
    progress_m: torch.Tensor,
    acquisition_gate: torch.Tensor,
    action_feasible: torch.Tensor,
    safety_violation: torch.Tensor,
    importance_weight: torch.Tensor,
    action_absolute: torch.Tensor,
    *,
    config: PositiveEffectSelfImitationConfigV706,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Return a differentiable loss and an outcome-only selection audit."""

    if type(config) is not PositiveEffectSelfImitationConfigV706:
        raise TypeError("V706 requires its exact self-imitation config")
    config.validate()
    count = predicted_acquisition_action.shape[0]
    if (
        predicted_acquisition_action.shape != (count, 3)
        or applied_action.shape != (count, 3)
        or progress_m.shape != (count,)
        or acquisition_gate.shape != (count,)
        or action_feasible.shape != (count,)
        or safety_violation.shape != (count,)
        or importance_weight.shape != (count,)
        or action_absolute.shape != (3,)
        or not bool(torch.isfinite(predicted_acquisition_action).all().item())
        or not bool(torch.isfinite(applied_action).all().item())
        or not bool(torch.isfinite(progress_m).all().item())
        or not bool(torch.isfinite(acquisition_gate).all().item())
        or not bool(torch.isfinite(action_feasible).all().item())
        or not bool(torch.isfinite(safety_violation).all().item())
        or not bool(torch.isfinite(importance_weight).all().item())
        or not bool(torch.isfinite(action_absolute).all().item())
        or bool(torch.any(action_absolute <= 0.0).item())
    ):
        raise ValueError("V706 self-imitation tensors are invalid")

    selected = (
        (progress_m >= config.minimum_progress_m)
        & (acquisition_gate >= config.minimum_acquisition_gate)
        & (safety_violation < 0.5)
    )
    target = torch.maximum(
        torch.minimum(applied_action.detach(), action_absolute),
        -action_absolute,
    )
    progress_weight = torch.clamp(
        progress_m / config.full_progress_weight_m,
        min=0.0,
        max=1.0,
    )
    weight = (
        selected.to(predicted_acquisition_action.dtype)
        * progress_weight
        * importance_weight
    )
    per_row = F.smooth_l1_loss(
        predicted_acquisition_action,
        target,
        reduction="none",
    ).mean(dim=-1)
    weight_sum = weight.sum()
    unscaled_loss = torch.where(
        weight_sum > 0.0,
        (weight * per_row).sum() / weight_sum.clamp_min(1.0e-8),
        predicted_acquisition_action.sum() * 0.0,
    )
    scaled_loss = config.coefficient * unscaled_loss
    selected_count = int(selected.sum().detach().cpu().item())
    selected_infeasible_count = int(
        (selected & (action_feasible <= 0.5)).sum().detach().cpu().item()
    )
    selected_progress = progress_m[selected]
    audit = {
        "format": POSITIVE_EFFECT_SELF_IMITATION_FORMAT_V706,
        "configuration": asdict(config),
        "batch_size": count,
        "selected_transition_count": selected_count,
        "selected_transition_fraction": selected_count / count,
        "selected_originally_infeasible_transition_count": (
            selected_infeasible_count
        ),
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
        "positive_measured_effect_required": True,
        "full_acquisition_gate_required": True,
        "verified_safe_execution_required": True,
        "original_proposal_feasibility_required": False,
        "originally_infeasible_projected_effects_correct_action_aliasing": True,
        "external_demonstration_used": False,
        "expert_action_used": False,
        "waypoint_or_path_used": False,
        "act_training_started": False,
        "production_admission": False,
    }
    return scaled_loss, audit


__all__ = [
    "POSITIVE_EFFECT_SELF_IMITATION_FORMAT_V706",
    "PositiveEffectSelfImitationConfigV706",
    "positive_effect_self_imitation_loss_v706",
]
