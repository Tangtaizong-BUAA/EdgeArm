"""Causal self-imitation using the policy decision stored at the same state.

V706/V707 used the tool displacement measured after ``env.step`` as an action
label.  With servo lag, a latched joint target, and contact dynamics, that
quantity is an outcome of several decisions rather than the action chosen at
the current state.  V717 keeps outcome measurements only as admission
evidence and imitates the replay policy action that was actually selected at
that state.

The acquisition auxiliary admits only safe, unmodified policy decisions.  In
contact transport the V711 low-level impedance filter owns lateral and
vertical motion, so V717 imitates only the policy-controlled forward
component.  No expert, route, waypoint, inverse-dynamics label, or future
policy target is introduced.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

import torch
from torch.nn import functional as F

from .positive_effect_self_imitation_v706 import (
    PositiveEffectSelfImitationConfigV706,
)
from .positive_effect_transport_self_imitation_v707 import (
    PositiveEffectTransportSelfImitationConfigV707,
)
from .causal_safe_projection_self_imitation_v732 import (
    CAUSAL_SAFE_PROJECTION_SELF_IMITATION_MODE_V732,
)


CAUSAL_CHOSEN_ACTION_SELF_IMITATION_FORMAT_V717 = (
    "edgearm-v717-causal-chosen-policy-action-self-imitation-v1"
)
CAUSAL_CHOSEN_ACTION_TRANSPORT_SELF_IMITATION_FORMAT_V717 = (
    "edgearm-v717-causal-chosen-forward-transport-self-imitation-v1"
)
MEASURED_EFFECT_SELF_IMITATION_MODE_V717 = "measured_effect_v706_v707"
CHOSEN_POLICY_ACTION_SELF_IMITATION_MODE_V717 = "chosen_policy_action_v717"
SELF_IMITATION_TARGET_MODES_V717 = (
    MEASURED_EFFECT_SELF_IMITATION_MODE_V717,
    CHOSEN_POLICY_ACTION_SELF_IMITATION_MODE_V717,
    CAUSAL_SAFE_PROJECTION_SELF_IMITATION_MODE_V732,
)


def _validate_common_vectors_v717(
    predicted_action: torch.Tensor,
    chosen_policy_action: torch.Tensor,
) -> int:
    count = predicted_action.shape[0]
    if (
        predicted_action.shape != (count, 3)
        or chosen_policy_action.shape != (count, 3)
        or not bool(torch.isfinite(predicted_action).all().item())
        or not bool(torch.isfinite(chosen_policy_action).all().item())
    ):
        raise ValueError("V717 action tensors are invalid")
    return count


def causal_chosen_action_acquisition_loss_v717(
    predicted_acquisition_action: torch.Tensor,
    chosen_policy_action: torch.Tensor,
    measured_progress_m: torch.Tensor,
    acquisition_gate: torch.Tensor,
    action_feasible: torch.Tensor,
    safety_violation: torch.Tensor,
    importance_weight: torch.Tensor,
    action_absolute: torch.Tensor,
    *,
    config: PositiveEffectSelfImitationConfigV706,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Imitate safe same-state decisions selected by positive approach."""

    if type(config) is not PositiveEffectSelfImitationConfigV706:
        raise TypeError("V717 acquisition requires the exact V706 thresholds")
    config.validate()
    count = _validate_common_vectors_v717(
        predicted_acquisition_action,
        chosen_policy_action,
    )
    scalars = (
        measured_progress_m,
        acquisition_gate,
        action_feasible,
        safety_violation,
        importance_weight,
    )
    if (
        any(value.shape != (count,) for value in scalars)
        or any(not bool(torch.isfinite(value).all().item()) for value in scalars)
        or action_absolute.shape != (3,)
        or not bool(torch.isfinite(action_absolute).all().item())
        or bool(torch.any(action_absolute <= 0.0).item())
    ):
        raise ValueError("V717 acquisition tensors are invalid")

    selected = (
        (measured_progress_m >= config.minimum_progress_m)
        & (acquisition_gate >= config.minimum_acquisition_gate)
        & (action_feasible > 0.5)
        & (safety_violation < 0.5)
    )
    target = torch.maximum(
        torch.minimum(chosen_policy_action.detach(), action_absolute),
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
    return scaled_loss, {
        "format": CAUSAL_CHOSEN_ACTION_SELF_IMITATION_FORMAT_V717,
        "configuration": asdict(config),
        "batch_size": count,
        "selected_transition_count": selected_count,
        "selected_transition_fraction": selected_count / count,
        "selected_originally_infeasible_transition_count": 0,
        "selected_originally_infeasible_transition_fraction": 0.0,
        "mean_selected_progress_m": (
            float(selected_progress.mean().detach().cpu().item())
            if selected_count
            else 0.0
        ),
        "unscaled_loss": float(unscaled_loss.detach().cpu().item()),
        "scaled_loss": float(scaled_loss.detach().cpu().item()),
        "target_is_same_state_replay_policy_action": True,
        "measured_effect_is_selector_not_action_label": True,
        "unmodified_feasible_policy_decision_required": True,
        "positive_measured_approach_required": True,
        "full_acquisition_gate_required": True,
        "external_demonstration_used": False,
        "expert_action_used": False,
        "waypoint_or_path_used": False,
        "act_training_started": False,
        "production_admission": False,
    }


def causal_chosen_forward_transport_loss_v717(
    predicted_action: torch.Tensor,
    chosen_policy_action: torch.Tensor,
    measured_original_task_progress_m: torch.Tensor,
    valid_contact: torch.Tensor,
    her_relabelled: torch.Tensor,
    action_feasible: torch.Tensor,
    safety_violation: torch.Tensor,
    importance_weight: torch.Tensor,
    *,
    config: PositiveEffectTransportSelfImitationConfigV707,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Imitate the policy-controlled forward decision during safe transport."""

    if type(config) is not PositiveEffectTransportSelfImitationConfigV707:
        raise TypeError("V717 transport requires the exact V707 thresholds")
    config.validate()
    count = _validate_common_vectors_v717(
        predicted_action,
        chosen_policy_action,
    )
    scalars = (
        measured_original_task_progress_m,
        valid_contact,
        her_relabelled,
        action_feasible,
        safety_violation,
        importance_weight,
    )
    if any(value.shape != (count,) for value in scalars) or any(
        not bool(torch.isfinite(value).all().item()) for value in scalars
    ):
        raise ValueError("V717 transport tensors are invalid")

    selected = (
        (measured_original_task_progress_m >= config.minimum_progress_m)
        & (valid_contact > 0.5)
        & (her_relabelled < 0.5)
        & (safety_violation < 0.5)
        & (chosen_policy_action[:, 0] > 0.0)
    )
    target_forward = torch.clamp(
        chosen_policy_action[:, 0].detach(),
        min=0.0,
        max=1.0,
    )
    progress_weight = torch.clamp(
        measured_original_task_progress_m / config.full_progress_weight_m,
        min=0.0,
        max=1.0,
    )
    weight = (
        selected.to(predicted_action.dtype)
        * progress_weight
        * importance_weight
    )
    per_row = F.smooth_l1_loss(
        predicted_action[:, 0],
        target_forward,
        reduction="none",
    )
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
    selected_progress = measured_original_task_progress_m[selected]
    return scaled_loss, {
        "format": CAUSAL_CHOSEN_ACTION_TRANSPORT_SELF_IMITATION_FORMAT_V717,
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
        "target_is_same_state_replay_policy_forward_action": True,
        "measured_effect_is_selector_not_action_label": True,
        "forward_dimension_only": True,
        "low_level_impedance_owns_lateral_and_vertical_contact_motion": True,
        "original_goal_transition_required": True,
        "valid_contact_required": True,
        "positive_measured_object_progress_required": True,
        "external_demonstration_used": False,
        "expert_action_used": False,
        "waypoint_or_path_used": False,
        "act_training_started": False,
        "production_admission": False,
    }


__all__ = [
    "CAUSAL_CHOSEN_ACTION_SELF_IMITATION_FORMAT_V717",
    "CAUSAL_CHOSEN_ACTION_TRANSPORT_SELF_IMITATION_FORMAT_V717",
    "CHOSEN_POLICY_ACTION_SELF_IMITATION_MODE_V717",
    "MEASURED_EFFECT_SELF_IMITATION_MODE_V717",
    "SELF_IMITATION_TARGET_MODES_V717",
    "causal_chosen_action_acquisition_loss_v717",
    "causal_chosen_forward_transport_loss_v717",
]
