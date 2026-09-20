"""Markov controller state for the stock-gripper task-frame adapter.

The V12/V13/V22 adapter intentionally latches a Cartesian goal and an exact
joint target across control decisions.  Those two targets affect the next IK
translation, but they are not part of the simulator's 163-D privileged state.
Consequently, a policy that only observes the simulator state is solving a
partially observable problem even though it is advertised as a Markov SAC
teacher.

V614 exposes only the controller memory required to make action translation
state-complete.  It contains no expert action, route, phase label, or future
information.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from typing import Protocol

import numpy as np

from .sim2real_env_v10 import RealisticEdgeArmEnvV10


TASKFRAME_CONTROLLER_STATE_FORMAT_V614 = (
    "edgearm-v614-taskframe-controller-markov-state-v1"
)
TASKFRAME_CONTROLLER_STATE_DIM_V614 = 11


class TaskFrameControllerProtocolV614(Protocol):
    env: RealisticEdgeArmEnvV10
    _latched_joint_target: np.ndarray | None
    _cartesian_goal: np.ndarray | None
    config: object


@dataclass(frozen=True)
class TaskFrameControllerStateFieldV614:
    name: str
    size: int
    source: str
    normalization: str


TASKFRAME_CONTROLLER_STATE_FIELDS_V614 = (
    TaskFrameControllerStateFieldV614(
        "latched_joint_target_error_over_max_delta",
        6,
        "adapter._latched_joint_target-env.data.qpos[:6]",
        "divide_by_env.config.max_joint_delta_then_clip_-4_4",
    ),
    TaskFrameControllerStateFieldV614(
        "cartesian_goal_error_over_maximum_lead",
        3,
        "adapter._cartesian_goal-env.tool_xyz()",
        "divide_by_adapter.config.maximum_cartesian_goal_lead_m_then_clip_-4_4",
    ),
    TaskFrameControllerStateFieldV614(
        "latched_joint_target_valid",
        1,
        "adapter._latched_joint_target is not None",
        "bool_to_float",
    ),
    TaskFrameControllerStateFieldV614(
        "cartesian_goal_valid",
        1,
        "adapter._cartesian_goal is not None",
        "bool_to_float",
    ),
)


if (
    sum(field.size for field in TASKFRAME_CONTROLLER_STATE_FIELDS_V614)
    != TASKFRAME_CONTROLLER_STATE_DIM_V614
):  # pragma: no cover - import-time invariant
    raise RuntimeError("V614 task-frame state fields do not sum to 11")


def _schema_sha256_v614() -> str:
    payload = {
        "format": TASKFRAME_CONTROLLER_STATE_FORMAT_V614,
        "dimension": TASKFRAME_CONTROLLER_STATE_DIM_V614,
        "dtype": "float32",
        "fields": [
            asdict(field) for field in TASKFRAME_CONTROLLER_STATE_FIELDS_V614
        ],
    }
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


TASKFRAME_CONTROLLER_STATE_SCHEMA_SHA256_V614 = _schema_sha256_v614()


def taskframe_controller_state_slices_v614() -> dict[str, slice]:
    result: dict[str, slice] = {}
    start = 0
    for field in TASKFRAME_CONTROLLER_STATE_FIELDS_V614:
        result[field.name] = slice(start, start + field.size)
        start += field.size
    if start != TASKFRAME_CONTROLLER_STATE_DIM_V614:  # pragma: no cover
        raise RuntimeError("V614 task-frame state slice layout drifted")
    return result


def build_taskframe_controller_state_v614(
    adapter: TaskFrameControllerProtocolV614,
) -> np.ndarray:
    """Return the exact normalized controller memory before one decision."""

    env = adapter.env
    if type(env) is not RealisticEdgeArmEnvV10:
        raise TypeError("V614 controller state requires exact V10 environment")
    joint_target = adapter._latched_joint_target
    cartesian_goal = adapter._cartesian_goal
    if joint_target is None or cartesian_goal is None:
        raise RuntimeError("V614 controller state requires an active episode")
    joint = np.asarray(joint_target, dtype=np.float64)
    goal = np.asarray(cartesian_goal, dtype=np.float64)
    physical_joint = np.asarray(env.data.qpos[:6], dtype=np.float64)
    tool_position = np.asarray(env.tool_xyz(), dtype=np.float64)
    if (
        joint.shape != (6,)
        or goal.shape != (3,)
        or physical_joint.shape != (6,)
        or tool_position.shape != (3,)
        or not np.all(np.isfinite(np.r_[joint, goal, physical_joint, tool_position]))
    ):
        raise RuntimeError("V614 controller memory contains invalid values")
    maximum_joint_delta = float(env.config.max_joint_delta)
    maximum_cartesian_lead = float(
        getattr(adapter.config, "maximum_cartesian_goal_lead_m")
    )
    if (
        not np.isfinite(maximum_joint_delta)
        or maximum_joint_delta <= 0.0
        or not np.isfinite(maximum_cartesian_lead)
        or maximum_cartesian_lead <= 0.0
    ):
        raise RuntimeError("V614 controller-state normalizer is invalid")
    state = np.r_[
        np.clip(
            (joint - physical_joint) / maximum_joint_delta,
            -4.0,
            4.0,
        ),
        np.clip(
            (goal - tool_position) / maximum_cartesian_lead,
            -4.0,
            4.0,
        ),
        1.0,
        1.0,
    ].astype(np.float32)
    if (
        state.shape != (TASKFRAME_CONTROLLER_STATE_DIM_V614,)
        or not np.all(np.isfinite(state))
    ):
        raise RuntimeError("V614 controller state is invalid")
    return state


__all__ = [
    "TASKFRAME_CONTROLLER_STATE_DIM_V614",
    "TASKFRAME_CONTROLLER_STATE_FIELDS_V614",
    "TASKFRAME_CONTROLLER_STATE_FORMAT_V614",
    "TASKFRAME_CONTROLLER_STATE_SCHEMA_SHA256_V614",
    "build_taskframe_controller_state_v614",
    "taskframe_controller_state_slices_v614",
]
