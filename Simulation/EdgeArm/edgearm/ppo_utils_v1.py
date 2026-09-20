"""Small, versioned PPO math primitives for the scratch full-action teacher.

The helpers in this module intentionally know nothing about experts, residual
policies, or the EdgeArm simulator.  Keeping the probability and return math
here makes it possible to test the two easy-to-miss PPO boundaries directly:

* actions are sampled in an unconstrained Gaussian space and transformed by
  exactly one ``tanh``;
* time-limit truncations bootstrap the value function, while true terminal
  transitions do not, and neither boundary leaks GAE into the next episode.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.nn import functional as functional


TANH_ACTION_EPSILON = 1.0e-6


@dataclass(frozen=True)
class SquashedGaussianSampleV1:
    """One sample and its exact density after a single tanh transform."""

    action: torch.Tensor
    pre_tanh: torch.Tensor
    log_prob: torch.Tensor


def squashed_gaussian_log_prob_v1(
    distribution: torch.distributions.Normal,
    pre_tanh: torch.Tensor,
) -> torch.Tensor:
    """Return ``log p(tanh(z))`` with the exact change-of-variables term.

    The softplus identity is numerically stable even when ``pre_tanh`` has a
    magnitude large enough for ``1 - tanh(z)**2`` to round to zero.
    """

    if distribution.loc.shape != pre_tanh.shape:
        raise ValueError(
            "distribution location and pre_tanh must have identical shapes: "
            f"{distribution.loc.shape} != {pre_tanh.shape}"
        )
    base_log_prob = distribution.log_prob(pre_tanh)
    log_tanh_jacobian = 2.0 * (
        math.log(2.0) - pre_tanh - functional.softplus(-2.0 * pre_tanh)
    )
    return (base_log_prob - log_tanh_jacobian).sum(dim=-1)


def squashed_gaussian_log_prob_from_action_v1(
    distribution: torch.distributions.Normal,
    action: torch.Tensor,
    *,
    epsilon: float = TANH_ACTION_EPSILON,
) -> torch.Tensor:
    """Invert a bounded action and evaluate its squashed Gaussian density."""

    if not 0.0 < epsilon < 0.1:
        raise ValueError("epsilon must be in (0, 0.1)")
    if distribution.loc.shape != action.shape:
        raise ValueError(
            "distribution location and action must have identical shapes: "
            f"{distribution.loc.shape} != {action.shape}"
        )
    bounded = action.clamp(-1.0 + epsilon, 1.0 - epsilon)
    pre_tanh = torch.atanh(bounded)
    return squashed_gaussian_log_prob_v1(distribution, pre_tanh)


def sample_squashed_gaussian_v1(
    distribution: torch.distributions.Normal,
    *,
    generator: torch.Generator | None = None,
) -> SquashedGaussianSampleV1:
    """Sample a Normal latent using an optional deterministic generator."""

    noise = torch.randn(
        distribution.loc.shape,
        dtype=distribution.loc.dtype,
        device=distribution.loc.device,
        generator=generator,
    )
    pre_tanh = distribution.loc + distribution.scale * noise
    action = torch.tanh(pre_tanh)
    return SquashedGaussianSampleV1(
        action=action,
        pre_tanh=pre_tanh,
        log_prob=squashed_gaussian_log_prob_v1(distribution, pre_tanh),
    )


def compute_gae_termination_truncation_v1(
    rewards: np.ndarray,
    values: np.ndarray,
    next_values: np.ndarray,
    terminated: np.ndarray,
    truncated: np.ndarray,
    *,
    gamma: float,
    gae_lambda: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute GAE without conflating termination and time-limit truncation.

    ``next_values`` contributes at a truncation boundary but not at a true
    terminal boundary.  The recursive advantage tail is cut at *both* kinds
    of episode boundary so samples from a reset episode can never leak back.
    """

    arrays = {
        "rewards": np.asarray(rewards),
        "values": np.asarray(values),
        "next_values": np.asarray(next_values),
        "terminated": np.asarray(terminated),
        "truncated": np.asarray(truncated),
    }
    shapes = {name: value.shape for name, value in arrays.items()}
    if len(set(shapes.values())) != 1 or arrays["rewards"].ndim != 1:
        raise ValueError(f"GAE inputs must be same-shape rank-one arrays, got {shapes}")
    if arrays["rewards"].size == 0:
        raise ValueError("GAE requires at least one transition")
    if not 0.0 < gamma <= 1.0 or not 0.0 <= gae_lambda <= 1.0:
        raise ValueError("gamma and gae_lambda are outside their valid ranges")
    for name in ("rewards", "values", "next_values"):
        if not np.all(np.isfinite(arrays[name])):
            raise ValueError(f"{name} contains non-finite values")

    reward = arrays["rewards"].astype(np.float64, copy=False)
    value = arrays["values"].astype(np.float64, copy=False)
    next_value = arrays["next_values"].astype(np.float64, copy=False)
    terminal = arrays["terminated"].astype(bool, copy=False)
    time_limit = arrays["truncated"].astype(bool, copy=False)
    advantage = np.zeros_like(reward, dtype=np.float64)
    recursive_tail = 0.0
    for index in range(reward.size - 1, -1, -1):
        bootstrap = 0.0 if terminal[index] else next_value[index]
        delta = reward[index] + gamma * bootstrap - value[index]
        episode_continues = not (terminal[index] or time_limit[index])
        recursive_tail = delta + gamma * gae_lambda * float(episode_continues) * recursive_tail
        advantage[index] = recursive_tail
    returns = advantage + value
    return advantage.astype(np.float32), returns.astype(np.float32)


def state_dict_sha256_v1(state_dict: Mapping[str, torch.Tensor]) -> str:
    """Hash a tensor state dict with names, dtypes, shapes, and raw bytes."""

    digest = hashlib.sha256()
    for name in sorted(state_dict):
        tensor = state_dict[name]
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"state_dict entry {name!r} is not a tensor")
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(b"\0")
        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
        digest.update(value.numpy().tobytes(order="C"))
    return digest.hexdigest()


def finite_module_parameters_v1(module: nn.Module) -> bool:
    """Return whether every parameter and buffer is finite."""

    return all(torch.isfinite(value).all().item() for value in module.state_dict().values())
