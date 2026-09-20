"""Action-space option for learning Home-to-contact acquisition.

The V630 hidden residual preserved the frozen transport policy, but V638
showed that it still inherited a strong action bias from that parent: every
exact-Home probe moved substantially while making essentially no progress
toward the task-aligned pre-contact point.  V639 adds a separately trainable
action-space residual.  It is active only under the same audited acquisition
gate and is bit-exact zero at contact and on the transport manifold.

This module supplies capacity, not an expert.  It receives no waypoint path,
future state, expert action, or behavior-cloning target.  SAC must learn the
residual from environment interaction and the existing acquisition reward.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch


PHASE_ISOLATED_ACQUISITION_OPTION_FORMAT_V639 = (
    "edgearm-v639-phase-isolated-acquisition-action-option-v1"
)


@dataclass(frozen=True)
class PhaseIsolatedAcquisitionOptionConfigV639:
    """Bound a learned pre-tanh action residual by the acquisition gate."""

    maximum_pre_tanh_mean_residual: float = 4.0

    def validate(self) -> None:
        value = self.maximum_pre_tanh_mean_residual
        if not math.isfinite(value) or not 0.5 <= value <= 8.0:
            raise ValueError("V639 acquisition option residual bound is invalid")


def phase_isolated_mean_residual_v639(
    raw_residual: torch.Tensor,
    acquisition_gate: torch.Tensor,
    *,
    maximum_pre_tanh_mean_residual: float,
) -> torch.Tensor:
    """Return a smooth bounded residual that is exact zero at gate zero."""

    if (
        raw_residual.ndim != 2
        or acquisition_gate.shape != (raw_residual.shape[0],)
        or raw_residual.shape[1] < 1
        or not math.isfinite(maximum_pre_tanh_mean_residual)
        or not 0.5 <= maximum_pre_tanh_mean_residual <= 8.0
        or not bool(torch.isfinite(raw_residual).all().item())
        or not bool(torch.isfinite(acquisition_gate).all().item())
        or bool(torch.any(acquisition_gate < 0.0).item())
        or bool(torch.any(acquisition_gate > 1.0).item())
    ):
        raise ValueError("V639 acquisition option inputs are invalid")
    residual = (
        float(maximum_pre_tanh_mean_residual)
        * acquisition_gate.unsqueeze(-1)
        * torch.tanh(raw_residual)
    )
    zero_gate = acquisition_gate == 0.0
    if bool(torch.any(zero_gate).item()) and not torch.equal(
        residual[zero_gate], torch.zeros_like(residual[zero_gate])
    ):
        raise RuntimeError("V639 option changed the exact transport policy")
    return residual


__all__ = [
    "PHASE_ISOLATED_ACQUISITION_OPTION_FORMAT_V639",
    "PhaseIsolatedAcquisitionOptionConfigV639",
    "phase_isolated_mean_residual_v639",
]
