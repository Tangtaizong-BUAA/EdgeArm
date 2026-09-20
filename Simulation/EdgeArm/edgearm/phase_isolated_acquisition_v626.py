"""Geometry-conditioned phase isolation for Home-to-contact learning.

V625 proved that one global five-percent residual has the wrong compromise:
it is too small to overturn the contact-transport parent while the tool is far
from the block, yet large enough to degrade that parent after contact.  V626
turns the residual into a soft acquisition option.  Simulator geometry only
sets a continuous capacity gate; it never supplies an action, route, future
state, or expert label.

The gate is exactly zero once the tool is correctly positioned and aligned for
pre-contact, and it is also forced to zero whenever block contact already
exists.  In that region the actor is bit-for-bit the frozen transport parent.
Far from the pre-contact manifold the same learned residual may use a much
larger norm budget so SAC can discover Home-to-contact motion.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from .goal_conditioned_her_sac_v43 import (
    GOAL_DIM_V43,
    OBSERVATION_DIM_V43,
)
from .privileged_effect_state_v1 import (
    PRIVILEGED_EFFECT_STATE_DIM,
    privileged_effect_state_slices_v1,
)


PHASE_ISOLATED_ACQUISITION_FORMAT_V626 = (
    "edgearm-v630-controller-aligned-phase-isolated-acquisition-v1"
)

_SLICES_V626 = privileged_effect_state_slices_v1()


@dataclass(frozen=True)
class PhaseIsolatedAcquisitionConfigV626:
    """Continuous acquisition gate with an exact transport boundary."""

    precontact_standoff_m: float = 0.055
    precontact_tool_height_m: float = 0.055
    exact_transport_distance_m: float = 0.040
    full_acquisition_distance_m: float = 0.120
    exact_transport_alignment: float = 0.98
    full_acquisition_alignment: float = 0.65
    exact_transport_maximum_tool_height_m: float = 0.061
    full_acquisition_tool_height_m: float = 0.100
    maximum_acquisition_context_over_base_hidden_norm: float = 0.75
    exact_frozen_parent_transport: bool = True

    def validate(self) -> None:
        values = (
            self.precontact_standoff_m,
            self.precontact_tool_height_m,
            self.exact_transport_distance_m,
            self.full_acquisition_distance_m,
            self.exact_transport_alignment,
            self.full_acquisition_alignment,
            self.exact_transport_maximum_tool_height_m,
            self.full_acquisition_tool_height_m,
            self.maximum_acquisition_context_over_base_hidden_norm,
        )
        if any(not math.isfinite(value) for value in values):
            raise ValueError("V626 phase-isolation parameters must be finite")
        if not 0.030 <= self.precontact_standoff_m <= 0.100:
            raise ValueError("V626 pre-contact standoff is invalid")
        if not 0.045 <= self.precontact_tool_height_m <= 0.100:
            raise ValueError("V626 pre-contact height is invalid")
        if not (
            0.010
            <= self.exact_transport_distance_m
            < self.full_acquisition_distance_m
            <= 0.400
        ):
            raise ValueError("V626 distance transition is invalid")
        if not (
            0.0
            <= self.full_acquisition_alignment
            < self.exact_transport_alignment
            <= 1.0
        ):
            raise ValueError("V626 alignment transition is invalid")
        if not (
            0.045
            <= self.exact_transport_maximum_tool_height_m
            < self.full_acquisition_tool_height_m
            <= 0.400
        ):
            raise ValueError("V630 tool-height transition is invalid")
        if not (
            0.10
            <= self.maximum_acquisition_context_over_base_hidden_norm
            <= 1.50
        ):
            raise ValueError("V626 acquisition residual capacity is invalid")
        if self.exact_frozen_parent_transport is not True:
            raise ValueError("V626 requires exact frozen-parent transport")


def _smoothstep_unit_v626(value: torch.Tensor) -> torch.Tensor:
    selected = torch.clamp(value, 0.0, 1.0)
    return selected.square() * (3.0 - 2.0 * selected)


def acquisition_phase_gate_v626(
    observation: torch.Tensor,
    *,
    config: PhaseIsolatedAcquisitionConfigV626,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return gate, tool-to-precontact distance, alignment, and contact."""

    config.validate()
    if (
        observation.ndim != 2
        or observation.shape[-1] != OBSERVATION_DIM_V43
        or not bool(torch.isfinite(observation).all().item())
    ):
        raise ValueError("V626 observation shape or values are invalid")
    block_pose = observation[
        :, _SLICES_V626["block_pose_xyz_quaternion_wxyz"]
    ]
    tool_pose = observation[:, _SLICES_V626["tool_pose_position_rotation"]]
    desired_goal = observation[
        :, PRIVILEGED_EFFECT_STATE_DIM : (
            PRIVILEGED_EFFECT_STATE_DIM + GOAL_DIM_V43
        )
    ]
    block_xy = block_pose[:, :2]
    direction = desired_goal - block_xy
    direction_norm = direction.norm(dim=-1, keepdim=True)
    fallback = torch.zeros_like(direction)
    fallback[:, 0] = 1.0
    forward = torch.where(
        direction_norm > 1.0e-7,
        direction / direction_norm.clamp_min(1.0e-7),
        fallback,
    )
    height = torch.full_like(
        block_xy[:, :1], config.precontact_tool_height_m
    )
    precontact = torch.cat(
        (
            block_xy - config.precontact_standoff_m * forward,
            height,
        ),
        dim=-1,
    )
    distance = (tool_pose[:, :3] - precontact).norm(dim=-1)
    rotation = tool_pose[:, 3:12].reshape(-1, 3, 3)
    broad_face_normal_xy = rotation[:, :, 1][:, :2]
    horizontal_norm = broad_face_normal_xy.norm(dim=-1).clamp_min(1.0e-7)
    alignment = torch.abs(
        (broad_face_normal_xy * forward).sum(dim=-1)
    ) / horizontal_norm
    alignment = torch.clamp(alignment, 0.0, 1.0)

    distance_phase = _smoothstep_unit_v626(
        (distance - config.exact_transport_distance_m)
        / (
            config.full_acquisition_distance_m
            - config.exact_transport_distance_m
        )
    )
    alignment_phase = _smoothstep_unit_v626(
        (config.exact_transport_alignment - alignment)
        / (
            config.exact_transport_alignment
            - config.full_acquisition_alignment
        )
    )
    height_phase = _smoothstep_unit_v626(
        (
            tool_pose[:, 2]
            - config.exact_transport_maximum_tool_height_m
        )
        / (
            config.full_acquisition_tool_height_m
            - config.exact_transport_maximum_tool_height_m
        )
    )
    contact = (
        observation[:, _SLICES_V626["tool_block_contact_count"]][:, 0]
        > 0.0
    )
    gate = torch.maximum(
        torch.maximum(distance_phase, alignment_phase),
        height_phase,
    )
    gate = torch.where(contact, torch.zeros_like(gate), gate)
    if not bool(
        (
            torch.isfinite(gate).all()
            & torch.isfinite(distance).all()
            & torch.isfinite(alignment).all()
            & (gate >= 0.0).all()
            & (gate <= 1.0).all()
        ).item()
    ):
        raise RuntimeError("V626 acquisition phase gate is invalid")
    return gate, distance, alignment, contact


def phase_isolated_context_hidden_v626(
    base_hidden: torch.Tensor,
    context_hidden: torch.Tensor,
    acquisition_gate: torch.Tensor,
    *,
    maximum_acquisition_ratio: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Bound residual capacity rowwise and force exact zero in transport."""

    if (
        base_hidden.ndim != 2
        or context_hidden.shape != base_hidden.shape
        or acquisition_gate.shape != (base_hidden.shape[0],)
        or not math.isfinite(maximum_acquisition_ratio)
        or not 0.10 <= maximum_acquisition_ratio <= 1.50
        or not bool(torch.isfinite(acquisition_gate).all().item())
        or not bool(
            ((acquisition_gate >= 0.0) & (acquisition_gate <= 1.0))
            .all()
            .item()
        )
    ):
        raise ValueError("V626 phase-isolated residual inputs are invalid")
    base_norm = base_hidden.norm(dim=-1, keepdim=True).clamp_min(1.0e-8)
    context_norm = context_hidden.norm(dim=-1, keepdim=True)
    maximum_norm = (
        maximum_acquisition_ratio
        * acquisition_gate.unsqueeze(-1)
        * base_norm
    )
    scale = torch.clamp(
        maximum_norm / context_norm.clamp_min(1.0e-8), max=1.0
    )
    scale = torch.where(
        acquisition_gate.unsqueeze(-1) == 0.0,
        torch.zeros_like(scale),
        scale,
    )
    bounded = context_hidden * scale
    realized_ratio = bounded.norm(dim=-1) / base_norm.squeeze(-1)
    if bool(torch.any(acquisition_gate == 0.0).item()) and not torch.equal(
        bounded[acquisition_gate == 0.0],
        torch.zeros_like(bounded[acquisition_gate == 0.0]),
    ):
        raise RuntimeError("V626 exact transport isolation was violated")
    return bounded, realized_ratio


__all__ = [
    "PHASE_ISOLATED_ACQUISITION_FORMAT_V626",
    "PhaseIsolatedAcquisitionConfigV626",
    "acquisition_phase_gate_v626",
    "phase_isolated_context_hidden_v626",
]
