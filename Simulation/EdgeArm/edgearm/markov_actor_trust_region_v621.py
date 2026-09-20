"""Fail-closed actor trust region for controller-state-complete HER-SAC.

V618 showed that simply appending controller memory to the actor is not a
safe update rule: most of its held-out action drift came from overwriting the
already useful V43 observation policy, while the new controller branch added
another unconstrained residual.  V621 keeps the parent actor as the behavioral
prior, caps the controller hidden residual relative to the parent hidden norm,
and anchors deterministic actions most strongly when the persistent controller
has little backlog.

The trust region changes learning only.  It does not provide an expert action,
task phase, future state, or privileged route to the actor.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch


MARKOV_ACTOR_TRUST_REGION_FORMAT_V621 = (
    "edgearm-v621-markov-actor-conditional-trust-region-v1"
)


@dataclass(frozen=True)
class MarkovActorTrustRegionConfigV621:
    """Learning-only constraints around one exact V43 parent actor."""

    freeze_parent_base: bool = True
    maximum_context_over_base_hidden_norm: float = 0.05
    low_backlog_anchor_coefficient: float = 2.0
    low_backlog_scale: float = 0.35

    def validate(self) -> None:
        if type(self.freeze_parent_base) is not bool:
            raise TypeError("V621 parent-freeze selector must be boolean")
        values = (
            self.maximum_context_over_base_hidden_norm,
            self.low_backlog_anchor_coefficient,
            self.low_backlog_scale,
        )
        if any(not math.isfinite(value) for value in values):
            raise ValueError("V621 trust-region parameters must be finite")
        if not 0.0 < self.maximum_context_over_base_hidden_norm <= 0.25:
            raise ValueError("V621 context ratio must lie in (0, 0.25]")
        if self.low_backlog_anchor_coefficient < 0.0:
            raise ValueError("V621 anchor coefficient must be non-negative")
        if not 0.05 <= self.low_backlog_scale <= 2.0:
            raise ValueError("V621 backlog scale is outside [0.05, 2]")


def bounded_context_hidden_v621(
    base_hidden: torch.Tensor,
    context_hidden: torch.Tensor,
    *,
    maximum_ratio: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Cap every context residual by a row-local fraction of base norm."""

    if (
        base_hidden.ndim != 2
        or context_hidden.shape != base_hidden.shape
        or not 0.0 < maximum_ratio <= 0.25
    ):
        raise ValueError("V621 hidden trust-region inputs are invalid")
    base_norm = base_hidden.norm(dim=-1, keepdim=True).clamp_min(1.0e-8)
    context_norm = context_hidden.norm(dim=-1, keepdim=True).clamp_min(1.0e-8)
    maximum_norm = float(maximum_ratio) * base_norm
    scale = torch.clamp(maximum_norm / context_norm, max=1.0)
    bounded = context_hidden * scale
    realized_ratio = bounded.norm(dim=-1) / base_norm.squeeze(-1)
    return bounded, realized_ratio


def low_backlog_anchor_weight_v621(
    controller_state: torch.Tensor,
    *,
    backlog_scale: float,
) -> torch.Tensor:
    """Return a smooth strong-near-zero weight from the nine backlog fields."""

    if controller_state.ndim != 2 or controller_state.shape[-1] < 9:
        raise ValueError("V621 controller state must contain nine backlog fields")
    if not 0.05 <= backlog_scale <= 2.0:
        raise ValueError("V621 backlog scale is invalid")
    backlog = controller_state[:, :9].square().mean(dim=-1).sqrt()
    return torch.exp(-backlog / float(backlog_scale))


def deterministic_action_anchor_v621(
    candidate_action: torch.Tensor,
    parent_action: torch.Tensor,
    controller_state: torch.Tensor,
    *,
    backlog_scale: float,
    row_multiplier: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute the weighted action anchor and diagnostics."""

    if (
        candidate_action.ndim != 2
        or parent_action.shape != candidate_action.shape
        or controller_state.shape[0] != candidate_action.shape[0]
    ):
        raise ValueError("V621 action-anchor shapes disagree")
    weight = low_backlog_anchor_weight_v621(
        controller_state, backlog_scale=backlog_scale
    )
    if row_multiplier is not None:
        multiplier = torch.as_tensor(
            row_multiplier,
            device=weight.device,
            dtype=weight.dtype,
        )
        if (
            multiplier.shape != weight.shape
            or not bool(torch.isfinite(multiplier).all().item())
            or bool(torch.any(multiplier < 0.0).item())
            or bool(torch.any(multiplier > 1.0).item())
        ):
            raise ValueError("V621 anchor row multiplier is invalid")
        weight = weight * multiplier
    per_row = (candidate_action - parent_action).square().mean(dim=-1)
    loss = (weight * per_row).sum() / weight.sum().clamp_min(1.0e-8)
    mean_delta_l2 = (candidate_action - parent_action).norm(dim=-1).mean()
    return loss, weight.mean(), mean_delta_l2


__all__ = [
    "MARKOV_ACTOR_TRUST_REGION_FORMAT_V621",
    "MarkovActorTrustRegionConfigV621",
    "bounded_context_hidden_v621",
    "deterministic_action_anchor_v621",
    "low_backlog_anchor_weight_v621",
]
