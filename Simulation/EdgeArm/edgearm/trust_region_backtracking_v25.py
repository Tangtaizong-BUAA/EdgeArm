"""Fail-closed parameter-space backtracking for EdgeArm V25.

PPO's sampled KL can remain small while a recurrent closed-loop controller
crosses a contact or safety boundary.  V25 therefore treats the optimizer
result as a direction, evaluates progressively smaller points on that exact
direction, and commits only the largest point that passes the paired
closed-loop gates.
"""

from __future__ import annotations

from collections.abc import Mapping

import torch


TRUST_REGION_BACKTRACKING_PPO_FORMAT_V25 = (
    "edgearm-v25-frontier-balanced-trust-region-backtracking-ppo-v1"
)
BACKTRACKING_SCALES_V25 = (1.0, 0.5, 0.25, 0.125, 0.0625)


def validate_backtracking_scales_v25(scales: tuple[float, ...]) -> None:
    """Require a deterministic largest-to-smallest positive search schedule."""

    if not isinstance(scales, tuple) or not scales:
        raise ValueError("V25 backtracking scales must be a non-empty tuple")
    if any(type(scale) is not float for scale in scales):
        raise TypeError("V25 backtracking scales must contain exact floats")
    if scales[0] != 1.0:
        raise ValueError("V25 backtracking must evaluate the full proposal first")
    if any(not 0.0 < scale <= 1.0 for scale in scales):
        raise ValueError("V25 backtracking scales must lie in (0,1]")
    if any(right >= left for left, right in zip(scales, scales[1:])):
        raise ValueError("V25 backtracking scales must be strictly decreasing")


def interpolate_state_dict_v25(
    parent: Mapping[str, torch.Tensor],
    optimizer_proposal: Mapping[str, torch.Tensor],
    scale: float,
) -> dict[str, torch.Tensor]:
    """Return ``parent + scale * (proposal - parent)`` without mutating inputs.

    Scale one and zero use exact clones so their state hashes are byte-identical
    to the corresponding endpoint. Non-floating buffers may not change along
    the proposed update and are copied from the parent.
    """

    if type(scale) is not float or not 0.0 <= scale <= 1.0:
        raise ValueError("V25 interpolation scale must be an exact float in [0,1]")
    if list(parent) != list(optimizer_proposal):
        raise ValueError("V25 parent/proposal state keys or ordering differ")
    interpolated: dict[str, torch.Tensor] = {}
    for name, parent_tensor in parent.items():
        proposal_tensor = optimizer_proposal[name]
        if not isinstance(parent_tensor, torch.Tensor) or not isinstance(
            proposal_tensor, torch.Tensor
        ):
            raise TypeError(f"V25 state entry is not a tensor: {name}")
        if (
            parent_tensor.shape != proposal_tensor.shape
            or parent_tensor.dtype != proposal_tensor.dtype
            or parent_tensor.device != proposal_tensor.device
        ):
            raise ValueError(f"V25 parent/proposal tensor metadata differs: {name}")
        if scale == 0.0:
            selected = parent_tensor.detach().clone()
        elif scale == 1.0:
            selected = proposal_tensor.detach().clone()
        elif parent_tensor.is_floating_point() or parent_tensor.is_complex():
            selected = torch.lerp(parent_tensor, proposal_tensor, scale).detach().clone()
        else:
            if not torch.equal(parent_tensor, proposal_tensor):
                raise ValueError(f"V25 cannot interpolate changed discrete state: {name}")
            selected = parent_tensor.detach().clone()
        if (selected.is_floating_point() or selected.is_complex()) and not torch.isfinite(
            selected
        ).all():
            raise RuntimeError(f"V25 interpolated state became non-finite: {name}")
        interpolated[name] = selected
    return interpolated


def scale_token_v25(scale: float) -> str:
    """Produce a stable filesystem token for a validated candidate scale."""

    if type(scale) is not float or not 0.0 < scale <= 1.0:
        raise ValueError("V25 scale token requires an exact float in (0,1]")
    return f"{scale:.6f}".replace(".", "p")


validate_backtracking_scales_v25(BACKTRACKING_SCALES_V25)


__all__ = [
    "BACKTRACKING_SCALES_V25",
    "TRUST_REGION_BACKTRACKING_PPO_FORMAT_V25",
    "interpolate_state_dict_v25",
    "scale_token_v25",
    "validate_backtracking_scales_v25",
]
