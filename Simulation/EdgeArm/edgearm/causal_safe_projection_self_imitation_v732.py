"""Home-only self-imitation from the same-state V720 safe projection.

V731 showed a repeatable acquisition failure: exact-Home rollouts kept the
acquisition gate at one but never reached contact, while the V705 audit found
large proposal-to-execution aliasing.  The DLS/V4 safeguard already records a
same-state, pre-plant safe task action for every valid projection.  V732 uses
that action as an auxiliary target only when the resulting transition made
positive progress toward the original Home precontact goal.

Measured next-state progress is admission evidence, not an action label.  HER
rows, post-contact reacquisition rows, non-Home rows, invalid projections, and
safety violations are excluded.  No expert action, waypoint, path, inverse
dynamics target, or future simulator action is introduced.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

import torch
from torch.nn import functional as F

from .positive_effect_self_imitation_v706 import (
    PositiveEffectSelfImitationConfigV706,
)


CAUSAL_SAFE_PROJECTION_SELF_IMITATION_FORMAT_V732 = (
    "edgearm-v732-home-safe-projection-self-imitation-v1"
)
CAUSAL_SAFE_PROJECTION_SELF_IMITATION_MODE_V732 = (
    "safe_projection_action_v732"
)


def causal_safe_projection_home_acquisition_loss_v732(
    predicted_acquisition_action: torch.Tensor,
    safeguard_projected_action_v720: torch.Tensor,
    measured_progress_m: torch.Tensor,
    acquisition_gate: torch.Tensor,
    exact_home_source: torch.Tensor,
    tool_goal_her_relabelled: torch.Tensor,
    post_contact_reacquisition_mode_v730: torch.Tensor,
    safeguard_projection_valid_v720: torch.Tensor,
    action_feasible: torch.Tensor,
    safety_violation: torch.Tensor,
    importance_weight: torch.Tensor,
    action_absolute: torch.Tensor,
    *,
    config: PositiveEffectSelfImitationConfigV706,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Imitate effective same-state guard projections on exact-Home rows."""

    if type(config) is not PositiveEffectSelfImitationConfigV706:
        raise TypeError("V732 requires the exact V706 thresholds")
    config.validate()
    count = predicted_acquisition_action.shape[0]
    vectors = (
        predicted_acquisition_action,
        safeguard_projected_action_v720,
    )
    scalars = (
        measured_progress_m,
        acquisition_gate,
        exact_home_source,
        tool_goal_her_relabelled,
        post_contact_reacquisition_mode_v730,
        safeguard_projection_valid_v720,
        action_feasible,
        safety_violation,
        importance_weight,
    )
    if (
        any(value.shape != (count, 3) for value in vectors)
        or any(value.shape != (count,) for value in scalars)
        or action_absolute.shape != (3,)
        or any(not bool(torch.isfinite(value).all().item()) for value in vectors)
        or any(not bool(torch.isfinite(value).all().item()) for value in scalars)
        or not bool(torch.isfinite(action_absolute).all().item())
        or bool(torch.any(action_absolute <= 0.0).item())
    ):
        raise ValueError("V732 safe-projection self-imitation tensors are invalid")

    selected = (
        (measured_progress_m >= config.minimum_progress_m)
        & (acquisition_gate >= config.minimum_acquisition_gate)
        & (exact_home_source > 0.5)
        & (tool_goal_her_relabelled < 0.5)
        & (post_contact_reacquisition_mode_v730 < 0.5)
        & (safeguard_projection_valid_v720 > 0.5)
        & (safety_violation < 0.5)
    )
    target = torch.maximum(
        torch.minimum(
            safeguard_projected_action_v720.detach(),
            action_absolute,
        ),
        -action_absolute,
    )
    progress_weight = torch.clamp(
        measured_progress_m / config.full_progress_weight_m,
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
    selected_progress = measured_progress_m[selected]
    selected_infeasible_count = int(
        (selected & (action_feasible <= 0.5)).sum().detach().cpu().item()
    )
    return scaled_loss, {
        "format": CAUSAL_SAFE_PROJECTION_SELF_IMITATION_FORMAT_V732,
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
        "target_is_same_state_v720_safe_projection": True,
        "measured_next_state_progress_is_selector_not_action_label": True,
        "exact_home_source_required": True,
        "original_precontact_goal_required": True,
        "post_contact_reacquisition_excluded": True,
        "valid_safe_projection_required": True,
        "original_proposal_feasibility_required": False,
        "external_demonstration_used": False,
        "expert_action_used": False,
        "waypoint_or_path_used": False,
        "act_training_started": False,
        "bulk_vla_data_use_allowed": False,
        "production_admission": False,
    }


__all__ = [
    "CAUSAL_SAFE_PROJECTION_SELF_IMITATION_FORMAT_V732",
    "CAUSAL_SAFE_PROJECTION_SELF_IMITATION_MODE_V732",
    "causal_safe_projection_home_acquisition_loss_v732",
]
