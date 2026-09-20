"""Dynamically reachable reverse-curriculum resets for contact acquisition.

V622 interpolated joint coordinates between Home and a task-aligned
pre-contact pose.  Although every interpolated pose was statically safe, a
small interpolation change produced a large policy-distribution cliff.  V673
instead starts from the mastered V22 pre-contact state and executes a
deterministic, V4-guarded random walk.  Only contact-free outcomes in the
requested expansion band are installed as time-zero curriculum starts.

The random walk is reset generation only.  It is not an expert path, is never
added to replay, and can never be exported as VLA data.  This follows reverse
curriculum generation's feasible-neighbour expansion and BaRC's requirement
that harder starts be dynamically consistent.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
from typing import Any

import mujoco
import numpy as np

from .asymmetric_multiview_ppo_v1 import (
    MAX_RESET_ATTEMPTS_V12,
    RESET_RETRY_STRIDE_V12,
    SOURCE_TYPE,
    VIEW_NAMES,
    MultiViewRendererProtocolV1,
    StockGripperTaskSpaceActionConfigV12,
    canonical_sha256_v1,
)
from .guarded_joint_delta_action_v664 import (
    GuardedJointActionPreflightV672,
    GuardedJointDeltaActionV664,
    GuardedJointNoSafeActionV664,
)
from .planar_push_projected_joint_action_v668 import (
    project_planar_push_joint_action_v668,
)
from .sim2real_env_v10 import RealisticEdgeArmEnvV10
from .stock_gripper_push_face_contact_v22 import (
    restore_stock_gripper_distal_contact_v22,
)
from .stock_gripper_taskframe_v22 import (
    StockGripperTaskFrameAdapterV22,
    reset_stock_taskframe_episode_v22,
    transition_contact_telemetry_v22,
)
from .task_independent_home_reset_v597 import (
    StockGripperHomeTaskFrameAdapterV597,
)
from .taskframe_dls_joint_action_v688 import (
    TaskFrameDLSJointActionConfigV688,
    project_taskframe_action_v688,
)


DYNAMIC_REVERSE_CURRICULUM_RESET_FORMAT_V673 = "edgearm-v673-dynamic-reverse-random-walk-reset-v1"
DYNAMIC_REVERSE_CURRICULUM_SOURCE_V673 = "v4_guarded_contact_free_random_walk_from_mastered_precontact"
DYNAMIC_TASKSPACE_REVERSE_CURRICULUM_RESET_FORMAT_V698 = (
    "edgearm-v698-dynamic-taskspace-reverse-random-walk-reset-v1"
)
DYNAMIC_TASKSPACE_REVERSE_CURRICULUM_SOURCE_V698 = (
    "v4_guarded_3d_taskspace_random_walk_from_true_v22_precontact"
)
DYNAMIC_TASKSPACE_RADIAL_RETREAT_RESET_FORMAT_V745 = "edgearm-v745-v698-seed-shuffled-tabu-reset-v3"
DYNAMIC_TASKSPACE_RADIAL_RETREAT_SOURCE_V745 = (
    "v4_guarded_3d_taskspace_random_walk_with_bounded_tabu_local_escape"
)
_RANDOM_WALK_BLOCK_MOTION_TOLERANCE_M_V673 = 2.0e-5
_RANDOM_WALK_MINIMUM_ALIGNMENT_V673 = 0.97
_RANDOM_WALK_CORRELATION_V673 = 0.70
_RANDOM_WALK_ACTION_STD_V673 = 0.55
_RANDOM_WALK_CANDIDATE_COUNT_V673 = 16
_RANDOM_WALK_MINIMUM_OUTWARD_PROGRESS_M_V673 = 1.0e-4
_RANDOM_WALK_MINIMUM_OUTWARD_PROGRESS_M_V745 = 1.0e-6
_RANDOM_WALK_MAXIMUM_INWARD_ESCAPE_M_V745 = 1.0e-3
_RANDOM_WALK_MAXIMUM_CONSECUTIVE_ESCAPE_STEPS_V745 = 12
_RANDOM_WALK_MINIMUM_ESCAPE_JOINT_NOVELTY_RAD_V745 = 1.0e-4
_RANDOM_WALK_OUTWARD_PREFLIGHT_SHORTLIST_COUNT_V745 = 4
_RANDOM_WALK_MINIMUM_CONTACT_PART_CLEARANCE_M_V673 = 2.5e-4
_RANDOM_WALK_MINIMUM_ALIGNMENT_V698 = 0.90
_RANDOM_WALK_CANDIDATE_COUNT_V698 = 8
_RANDOM_WALK_ACTION_STD_V698 = 0.70

# Expansion is measured from the exact same-task V22 pre-contact distance.
# The first tier deliberately closes the observed 2.1 cm -> 3.7 cm cliff with
# a much narrower dynamically reached shell.  Later shells are only visited
# after the preceding shell demonstrates clean contact acquisition.
_DYNAMIC_REVERSE_SPECS_V673: dict[float, tuple[float, float, int]] = {
    0.99: (0.002, 0.008, 24),
    0.98: (0.006, 0.016, 36),
    0.95: (0.014, 0.030, 48),
    0.90: (0.026, 0.050, 60),
    0.80: (0.045, 0.080, 80),
    0.65: (0.075, 0.115, 100),
    0.45: (0.105, 0.155, 120),
    0.25: (0.145, 0.205, 120),
}

# V698 keeps the true V22 start but expands in the full 3-D task space used by
# the deployed DLS executor.  Each label is only a stable curriculum key; the
# actual difficulty is the measured pre-contact-distance expansion band.
_DYNAMIC_TASKSPACE_REVERSE_SPECS_V698: dict[float, tuple[float, float, int]] = {
    0.995: (0.002, 0.006, 24),
    0.990: (0.005, 0.012, 32),
    0.980: (0.010, 0.022, 40),
    0.950: (0.020, 0.040, 48),
    0.900: (0.035, 0.060, 60),
    0.800: (0.055, 0.090, 72),
    0.650: (0.085, 0.125, 84),
    0.450: (0.120, 0.165, 96),
    0.250: (0.160, 0.205, 108),
    0.100: (0.200, 0.245, 120),
    0.050: (0.235, 0.285, 120),
}


class StockGripperDynamicReverseTaskFrameAdapterV673(StockGripperHomeTaskFrameAdapterV597):
    """Begin the unchanged V22 runtime at one V673-audited state."""

    def __init__(
        self,
        env: RealisticEdgeArmEnvV10,
        config: StockGripperTaskSpaceActionConfigV12 | None = None,
    ) -> None:
        super().__init__(env, config)
        self.expected_reset_joint_position_v673: np.ndarray | None = None
        self.dynamic_reverse_reset_audit_v673: dict[str, Any] = {}

    def arm_expected_reset_v673(self, joint_position: np.ndarray) -> None:
        expected = np.asarray(joint_position, dtype=np.float64)
        if expected.shape != (6,) or not np.all(np.isfinite(expected)):
            raise ValueError("V673 expected reset must be finite [6]")
        self.expected_reset_joint_position_v673 = expected.copy()

    def begin_episode(self, seed: int) -> None:
        expected = self.expected_reset_joint_position_v673
        if expected is None:
            raise RuntimeError("V673 adapter was not armed by its reset")
        self._begin_episode_from_exact_joint_v622(
            seed,
            expected,
            reset_kind="privileged_dynamic_reverse_random_walk_curriculum",
        )


def dynamic_reverse_curriculum_spec_v673(
    home_to_precontact_fraction: float,
) -> dict[str, float | int]:
    """Return the deterministic feasible-neighbour shell for one old tier."""

    fraction = float(home_to_precontact_fraction)
    if not np.isfinite(fraction):
        raise ValueError("V673 curriculum fraction is not finite")
    matched = [
        key for key in _DYNAMIC_REVERSE_SPECS_V673 if np.isclose(fraction, key, rtol=0.0, atol=1.0e-12)
    ]
    if len(matched) != 1:
        raise ValueError("V673 curriculum fraction has no dynamic shell")
    key = matched[0]
    minimum, maximum, steps = _DYNAMIC_REVERSE_SPECS_V673[key]
    return {
        "home_to_precontact_fraction_label": key,
        "minimum_precontact_distance_expansion_m": minimum,
        "maximum_precontact_distance_expansion_m": maximum,
        "maximum_random_walk_steps": steps,
    }


def dynamic_taskspace_reverse_curriculum_spec_v698(
    home_to_precontact_fraction: float,
) -> dict[str, float | int | str]:
    """Return a measured 3-D feasible-neighbour shell for V698."""

    fraction = float(home_to_precontact_fraction)
    if not np.isfinite(fraction):
        raise ValueError("V698 curriculum fraction is not finite")
    matched = [
        key
        for key in _DYNAMIC_TASKSPACE_REVERSE_SPECS_V698
        if np.isclose(fraction, key, rtol=0.0, atol=1.0e-12)
    ]
    if len(matched) != 1:
        raise ValueError("V698 curriculum fraction has no 3-D dynamic shell")
    key = matched[0]
    minimum, maximum, steps = _DYNAMIC_TASKSPACE_REVERSE_SPECS_V698[key]
    return {
        "format": DYNAMIC_TASKSPACE_REVERSE_CURRICULUM_RESET_FORMAT_V698,
        "home_to_precontact_fraction_label": key,
        "minimum_precontact_distance_expansion_m": minimum,
        "maximum_precontact_distance_expansion_m": maximum,
        "maximum_random_walk_steps": steps,
        "expansion_coordinate": "measured_3d_tool_to_precontact_distance",
    }


def _precontact_geometry_v673(
    env: RealisticEdgeArmEnvV10,
) -> tuple[float, float]:
    block = env.block_xy().copy()
    target = env.target_xy.copy()
    direction = target - block
    norm = float(np.linalg.norm(direction))
    forward = np.asarray([1.0, 0.0], dtype=np.float64) if norm <= 1.0e-7 else direction / norm
    point = np.r_[block - 0.055 * forward, 0.055]
    distance = float(np.linalg.norm(env.tool_xyz() - point))
    rotation = env.data.site_xmat[env._ids["tool_site"]].reshape(3, 3)
    normal_xy = rotation[:, 1][:2]
    horizontal_norm = float(np.linalg.norm(normal_xy))
    heading = abs(float(np.dot(normal_xy, forward))) / max(horizontal_norm, 1.0e-7)
    return distance, float(np.clip(min(horizontal_norm, heading), 0.0, 1.0))


def _precontact_geometry_for_joint_v673(
    env: RealisticEdgeArmEnvV10,
    joint_position_rad: np.ndarray,
) -> tuple[float, float, float]:
    """Evaluate one forecast endpoint without changing the live simulator."""

    joint = np.asarray(joint_position_rad, dtype=np.float64)
    if joint.shape != (6,) or not np.all(np.isfinite(joint)):
        raise ValueError("V673 forecast joint endpoint is invalid")
    scratch = mujoco.MjData(env.model)
    mujoco.mj_copyData(scratch, env.model, env.data)
    scratch.qpos[:6] = joint
    scratch.qvel[:6] = 0.0
    mujoco.mj_forward(env.model, scratch)
    block = env.block_xy().copy()
    target = env.target_xy.copy()
    direction = target - block
    norm = float(np.linalg.norm(direction))
    forward = np.asarray([1.0, 0.0], dtype=np.float64) if norm <= 1.0e-7 else direction / norm
    point = np.r_[block - 0.055 * forward, 0.055]
    tool = np.asarray(scratch.site_xpos[env._ids["tool_site"]], dtype=np.float64)
    rotation = np.asarray(scratch.site_xmat[env._ids["tool_site"]], dtype=np.float64).reshape(3, 3)
    normal_xy = rotation[:, 1][:2]
    horizontal_norm = float(np.linalg.norm(normal_xy))
    heading = abs(float(np.dot(normal_xy, forward))) / max(horizontal_norm, 1.0e-7)
    return (
        float(np.linalg.norm(tool - point)),
        float(np.clip(min(horizontal_norm, heading), 0.0, 1.0)),
        float(tool[2]),
    )


def _task_aligned_radial_retreat_action_v745(
    env: RealisticEdgeArmEnvV10,
    config: TaskFrameDLSJointActionConfigV688,
) -> np.ndarray:
    """Point one reset-only task action away from the precontact point.

    V698 sampled eight random actions plus the six signed task axes.  A DLS
    projection can map all of those discrete directions to tangential or
    inward endpoints even while a combined task-space direction remains
    available.  This helper contributes that combined radial direction as one
    additional *candidate*.  It is still filtered by the unchanged V4 guard,
    contact-free forecast, alignment, height, and measured outward-progress
    gates before execution.
    """

    if type(env) is not RealisticEdgeArmEnvV10:
        raise TypeError("V745 radial retreat requires the exact V10 environment")
    if type(config) is not TaskFrameDLSJointActionConfigV688:
        raise TypeError("V745 radial retreat requires the exact V688 config")
    config.validate()
    block = env.block_xy().copy()
    target = env.target_xy.copy()
    direction = target - block
    norm = float(np.linalg.norm(direction))
    forward = np.asarray([1.0, 0.0], dtype=np.float64) if norm <= 1.0e-7 else direction / norm
    lateral = np.asarray([-forward[1], forward[0]], dtype=np.float64)
    point = np.r_[block - 0.055 * forward, 0.055]
    radial_world = np.asarray(env.tool_xyz(), dtype=np.float64) - point
    radial_local = np.asarray(
        [
            float(np.dot(radial_world[:2], forward)),
            float(np.dot(radial_world[:2], lateral)),
            float(radial_world[2]),
        ],
        dtype=np.float64,
    )
    task_scale = np.asarray(
        [
            config.forward_translation_step_m,
            config.lateral_translation_step_m,
            config.vertical_translation_step_m,
        ],
        dtype=np.float64,
    )
    normalized = radial_local / task_scale
    maximum = float(np.max(np.abs(normalized)))
    if not np.isfinite(maximum):
        raise RuntimeError("V745 radial retreat direction is not finite")
    if maximum <= 1.0e-9:
        # The exact precontact point has no radial gradient.  Negative task
        # forward is the task-frame retreat direction and remains only a
        # guarded candidate, never an installed action by itself.
        normalized = np.asarray([-1.0, 0.0, 0.0], dtype=np.float64)
    else:
        normalized /= maximum
    return np.clip(normalized, -1.0, 1.0).astype(np.float32)


def _preflight_endpoint_v673(
    preflight: GuardedJointActionPreflightV672,
) -> np.ndarray | None:
    if not preflight.safe_candidate_found:
        return None
    if bool(preflight.guard_report.get("selected_is_baseline_hold", False)):
        return None
    scale = preflight.guard_report.get("selected_scale")
    if scale is None or float(scale) <= 0.0:
        return None
    selected_forecast = preflight.guard_report.get("selected_forecast")
    if not isinstance(selected_forecast, dict):
        raise RuntimeError("V673 preflight lost selected forecast")
    rows = [
        row for row in selected_forecast.get("applications", []) if bool(row.get("candidate_command", False))
    ]
    if len(rows) != 1:
        raise RuntimeError("V673 preflight lost candidate endpoint identity")
    endpoint = np.asarray(rows[0]["physics_endpoint_rad"], dtype=np.float64)
    if endpoint.shape != (6,) or not np.all(np.isfinite(endpoint)):
        raise RuntimeError("V673 preflight endpoint is invalid")
    return endpoint


def _preflight_is_contact_free_v673(
    preflight: GuardedJointActionPreflightV672,
    *,
    initial_block_xy_m: np.ndarray,
) -> bool:
    """Reject a V4-safe candidate if its exact forecast touches the block."""

    initial = np.asarray(initial_block_xy_m, dtype=np.float64)
    if initial.shape != (2,) or not np.all(np.isfinite(initial)):
        raise ValueError("V673 initial block position is invalid")
    forecast = preflight.guard_report.get("selected_forecast")
    if not isinstance(forecast, dict):
        return False
    contact_by_role = forecast.get("minimum_forecast_contact_part_block_distance_by_role_m")
    applications = forecast.get("applications")
    if not isinstance(contact_by_role, dict) or not isinstance(applications, list):
        return False
    contact_distances = np.asarray(list(contact_by_role.values()), dtype=np.float64)
    if (
        contact_distances.size != 8
        or not np.all(np.isfinite(contact_distances))
        or float(np.min(contact_distances)) < _RANDOM_WALK_MINIMUM_CONTACT_PART_CLEARANCE_M_V673
    ):
        return False
    for application in applications:
        block = np.asarray(application.get("block_xy_m"), dtype=np.float64)
        trace = application.get("trace_audit")
        if (
            block.shape != (2,)
            or not np.all(np.isfinite(block))
            or float(np.linalg.norm(block - initial)) > _RANDOM_WALK_BLOCK_MOTION_TOLERANCE_M_V673
            or not isinstance(trace, dict)
            or int(trace.get("invalid_tool_block_contact_count", -1)) != 0
        ):
            return False
    return True


def reset_stock_dynamic_reverse_episode_v673(
    env: RealisticEdgeArmEnvV10,
    renderer: MultiViewRendererProtocolV1,
    action_adapter: StockGripperDynamicReverseTaskFrameAdapterV673,
    *,
    requested_seed: int,
    obstacle: bool,
    stress: bool,
    home_to_precontact_fraction: float,
    taskspace_3d_v698: bool = False,
    radial_retreat_candidate_v745: bool = False,
) -> dict[str, Any]:
    """Generate and install one contact-free dynamically reached start.

    ``taskspace_3d_v698`` expands with the same guarded 3-D DLS coordinate
    transform as online training.  The legacy default retains V673's planar
    joint-space reset contract for reproducibility of historical artifacts.
    """

    if type(env) is not RealisticEdgeArmEnvV10:
        raise TypeError("V673 reset requires exact V10 environment")
    if type(action_adapter) is not StockGripperDynamicReverseTaskFrameAdapterV673:
        raise TypeError("V673 reset requires exact V673 adapter")
    if action_adapter.env is not env:
        raise ValueError("V673 adapter belongs to another environment")
    if tuple(renderer.view_names) != VIEW_NAMES:
        raise ValueError("V673 reset renderer view order changed")
    if type(requested_seed) is not int or requested_seed < 0:
        raise ValueError("V673 reset seed must be non-negative")
    if type(obstacle) is not bool or type(stress) is not bool:
        raise TypeError("V673 reset conditions must be boolean")
    if type(taskspace_3d_v698) is not bool:
        raise TypeError("V698 task-space reset selector must be boolean")
    if (
        type(radial_retreat_candidate_v745) is not bool
        or radial_retreat_candidate_v745
        and not taskspace_3d_v698
    ):
        raise TypeError("V745 radial reset selector requires V698 task space")
    spec = (
        dynamic_taskspace_reverse_curriculum_spec_v698(home_to_precontact_fraction)
        if taskspace_3d_v698
        else dynamic_reverse_curriculum_spec_v673(home_to_precontact_fraction)
    )
    reset_format = (
        DYNAMIC_TASKSPACE_RADIAL_RETREAT_RESET_FORMAT_V745
        if radial_retreat_candidate_v745
        else (
            DYNAMIC_TASKSPACE_REVERSE_CURRICULUM_RESET_FORMAT_V698
            if taskspace_3d_v698
            else DYNAMIC_REVERSE_CURRICULUM_RESET_FORMAT_V673
        )
    )
    reset_source = (
        DYNAMIC_TASKSPACE_RADIAL_RETREAT_SOURCE_V745
        if radial_retreat_candidate_v745
        else (
            DYNAMIC_TASKSPACE_REVERSE_CURRICULUM_SOURCE_V698
            if taskspace_3d_v698
            else DYNAMIC_REVERSE_CURRICULUM_SOURCE_V673
        )
    )
    minimum_alignment = (
        _RANDOM_WALK_MINIMUM_ALIGNMENT_V698 if taskspace_3d_v698 else _RANDOM_WALK_MINIMUM_ALIGNMENT_V673
    )
    candidate_count = (
        _RANDOM_WALK_CANDIDATE_COUNT_V698 if taskspace_3d_v698 else _RANDOM_WALK_CANDIDATE_COUNT_V673
    )
    minimum_expansion = float(spec["minimum_precontact_distance_expansion_m"])
    maximum_expansion = float(spec["maximum_precontact_distance_expansion_m"])
    maximum_walk_steps = int(spec["maximum_random_walk_steps"])
    minimum_outward_progress = (
        _RANDOM_WALK_MINIMUM_OUTWARD_PROGRESS_M_V745
        if radial_retreat_candidate_v745
        else _RANDOM_WALK_MINIMUM_OUTWARD_PROGRESS_M_V673
    )
    rejected: list[dict[str, Any]] = []

    for attempt_index in range(MAX_RESET_ATTEMPTS_V12):
        proposal_seed = requested_seed + attempt_index * RESET_RETRY_STRIDE_V12
        walk_adapter = StockGripperTaskFrameAdapterV22(env, action_adapter.config)
        try:
            reset_stock_taskframe_episode_v22(
                env,
                renderer,
                walk_adapter,
                requested_seed=proposal_seed,
                obstacle=obstacle,
                stress=stress,
            )
        except RuntimeError as error:
            rejected.append(
                {
                    "attempt_index": attempt_index,
                    "seed": proposal_seed,
                    "failure_reasons": [f"precontact_reset:{error}"],
                }
            )
            continue
        probe_domain = deepcopy(env.episode_domain.get("realism_v7", {}))
        # The historical V12 wrapper can report its requested retry seed while
        # V7 has internally accepted a later task candidate.  Bind the final
        # time-zero reinstall to the actual accepted seed or block/target
        # identity will silently change.
        selected_seed = int(probe_domain.get("accepted_candidate_seed", proposal_seed))
        block = env.block_xy().copy()
        target = env.target_xy.copy()
        initial_block_target_distance = float(env.distance_to_target())
        initial_target_coverage = float(env.block_target_coverage())
        initial_precontact_distance, initial_alignment = _precontact_geometry_v673(env)
        lower = initial_precontact_distance + minimum_expansion
        upper = initial_precontact_distance + maximum_expansion
        guarded = GuardedJointDeltaActionV664(walk_adapter)
        rng = np.random.default_rng(
            selected_seed
            ^ (0x698D1A if taskspace_3d_v698 else 0x673D1A)
            ^ int(round(100 * float(home_to_precontact_fraction)))
        )
        correlated = np.zeros(
            3 if taskspace_3d_v698 else 5,
            dtype=np.float64,
        )
        taskspace_dls = TaskFrameDLSJointActionConfigV688(
            forward_translation_step_m=(action_adapter.config.forward_translation_step_m),
            lateral_translation_step_m=(action_adapter.config.lateral_translation_step_m),
            vertical_translation_step_m=(action_adapter.config.vertical_translation_step_m),
        )
        height_bounds = (
            tuple(float(value) for value in env.config.workspace_z)
            if taskspace_3d_v698
            else (
                float(action_adapter.config.minimum_tool_height_m),
                float(action_adapter.config.maximum_tool_height_m),
            )
        )
        walk_rows: list[dict[str, Any]] = []
        final_joint: np.ndarray | None = None
        failure_reason = "random_walk_did_not_reach_requested_shell"
        last_candidate_diagnostics: dict[str, Any] = {}
        consecutive_local_escape_steps_v745 = 0
        local_escape_step_count_v745 = 0
        best_precontact_distance_v745 = initial_precontact_distance
        visited_joint_endpoints_v745 = [np.asarray(env.data.qpos[:6], dtype=np.float64).copy()]

        for walk_step in range(maximum_walk_steps):
            current_distance, _current_alignment = _precontact_geometry_v673(env)
            candidates: list[dict[str, Any]] = []
            local_escape_candidates_v745: list[dict[str, Any]] = []
            basis_actions = (
                (
                    np.asarray([1.0, 0.0, 0.0], dtype=np.float32),
                    np.asarray([-1.0, 0.0, 0.0], dtype=np.float32),
                    np.asarray([0.0, 1.0, 0.0], dtype=np.float32),
                    np.asarray([0.0, -1.0, 0.0], dtype=np.float32),
                    np.asarray([0.0, 0.0, 1.0], dtype=np.float32),
                    np.asarray([0.0, 0.0, -1.0], dtype=np.float32),
                )
                if taskspace_3d_v698
                else ()
            )
            radial_actions = (
                (
                    _task_aligned_radial_retreat_action_v745(
                        env,
                        taskspace_dls,
                    ),
                )
                if radial_retreat_candidate_v745
                else ()
            )
            total_candidate_count = candidate_count + len(basis_actions) + len(radial_actions)
            random_candidate_rows: list[tuple[np.ndarray, np.ndarray]] = []
            for _random_candidate_index in range(candidate_count):
                innovation = rng.normal(
                    0.0,
                    (_RANDOM_WALK_ACTION_STD_V698 if taskspace_3d_v698 else _RANDOM_WALK_ACTION_STD_V673),
                    correlated.shape[0],
                )
                candidate_correlated = (
                    _RANDOM_WALK_CORRELATION_V673 * correlated
                    + np.sqrt(1.0 - _RANDOM_WALK_CORRELATION_V673**2) * innovation
                )
                random_candidate_rows.append(
                    (
                        candidate_correlated,
                        np.clip(candidate_correlated, -1.0, 1.0).astype(np.float32),
                    )
                )
            candidate_indices = (
                tuple(int(value) for value in rng.permutation(total_candidate_count))
                if radial_retreat_candidate_v745
                else tuple(range(total_candidate_count))
            )
            candidate_diagnostics = {
                "walk_step": walk_step,
                "current_precontact_distance_m": current_distance,
                "requested_shell_lower_m": lower,
                "requested_shell_upper_m": upper,
                "distance_shortfall_to_lower_m": max(
                    lower - current_distance,
                    0.0,
                ),
                "sampled_random": candidate_count,
                "task_independent_basis": len(basis_actions),
                "task_aligned_radial_retreat_v745": len(radial_actions),
                "sampled_total": total_candidate_count,
                "candidate_evaluation_order_v745": (
                    list(candidate_indices) if radial_retreat_candidate_v745 else None
                ),
                "preflight_evaluated": 0,
                "outward_preflight_shortlist_count_v745": (
                    _RANDOM_WALK_OUTWARD_PREFLIGHT_SHORTLIST_COUNT_V745
                    if radial_retreat_candidate_v745
                    else None
                ),
                "guard_preflight_safe": 0,
                "contact_free": 0,
                "endpoint_available": 0,
                "outward": 0,
                "inside_requested_shell_upper_v745": 0,
                "aligned": 0,
                "inside_height_bounds": 0,
                "accepted": 0,
                "local_escape_eligible_v745": 0,
                "novel_endpoint_v745": 0,
                "maximum_allowed_inward_escape_m_v745": (
                    _RANDOM_WALK_MAXIMUM_INWARD_ESCAPE_M_V745 if radial_retreat_candidate_v745 else None
                ),
                "consecutive_local_escape_steps_before_v745": (consecutive_local_escape_steps_v745),
                "initial_safe_precontact_distance_floor_m_v745": (initial_precontact_distance),
                "best_precontact_distance_m_v745": (best_precontact_distance_v745),
                "maximum_predicted_distance_m": None,
                "maximum_predicted_outward_delta_m": None,
                "radial_predicted_distance_m_v745": None,
                "radial_predicted_outward_delta_m_v745": None,
            }
            for candidate_index in candidate_indices:
                candidate_diagnostics["preflight_evaluated"] += 1
                if candidate_index < candidate_count:
                    candidate_correlated, candidate_raw = random_candidate_rows[candidate_index]
                    candidate_source = "correlated_random"
                elif candidate_index < candidate_count + len(basis_actions):
                    candidate_raw = basis_actions[candidate_index - candidate_count].copy()
                    candidate_correlated = candidate_raw.astype(np.float64)
                    candidate_source = "task_independent_axis_basis"
                else:
                    candidate_raw = radial_actions[
                        candidate_index - candidate_count - len(basis_actions)
                    ].copy()
                    candidate_correlated = candidate_raw.astype(np.float64)
                    candidate_source = "task_aligned_radial_retreat_v745"
                if taskspace_3d_v698:
                    candidate_taskspace = project_taskframe_action_v688(
                        env,
                        candidate_raw,
                        taskspace_dls,
                    )
                    projected_joint_action = candidate_taskspace.projected_joint_action
                    projection_l2 = float(candidate_taskspace.requested_to_predicted_task_l2)
                    projection_format = str(candidate_taskspace.format)
                else:
                    candidate_planar = project_planar_push_joint_action_v668(env, candidate_raw)
                    projected_joint_action = candidate_planar.projected_action
                    projection_l2 = float(candidate_planar.projection_l2)
                    projection_format = str(candidate_planar.format)
                try:
                    candidate_preflight = guarded.preflight(projected_joint_action)
                except GuardedJointNoSafeActionV664:
                    continue
                candidate_diagnostics["guard_preflight_safe"] += 1
                if not _preflight_is_contact_free_v673(
                    candidate_preflight,
                    initial_block_xy_m=env.block_xy(),
                ):
                    continue
                candidate_diagnostics["contact_free"] += 1
                endpoint = _preflight_endpoint_v673(candidate_preflight)
                if endpoint is None:
                    continue
                candidate_diagnostics["endpoint_available"] += 1
                predicted_distance, predicted_alignment, predicted_height = (
                    _precontact_geometry_for_joint_v673(env, endpoint)
                )
                outward = bool(predicted_distance >= current_distance + minimum_outward_progress)
                predicted_outward_delta = predicted_distance - current_distance
                previous_maximum = candidate_diagnostics["maximum_predicted_distance_m"]
                if previous_maximum is None or predicted_distance > float(previous_maximum):
                    candidate_diagnostics["maximum_predicted_distance_m"] = predicted_distance
                    candidate_diagnostics["maximum_predicted_outward_delta_m"] = predicted_outward_delta
                if candidate_source == "task_aligned_radial_retreat_v745":
                    candidate_diagnostics["radial_predicted_distance_m_v745"] = predicted_distance
                    candidate_diagnostics["radial_predicted_outward_delta_m_v745"] = predicted_outward_delta
                aligned = bool(predicted_alignment >= minimum_alignment)
                inside_height = bool(height_bounds[0] <= predicted_height <= height_bounds[1])
                inside_requested_shell_upper_v745 = bool(
                    not radial_retreat_candidate_v745 or predicted_distance <= upper
                )
                candidate_diagnostics["outward"] += int(outward)
                candidate_diagnostics["inside_requested_shell_upper_v745"] += int(
                    inside_requested_shell_upper_v745
                )
                candidate_diagnostics["aligned"] += int(aligned)
                candidate_diagnostics["inside_height_bounds"] += int(inside_height)
                novel_endpoint_v745 = bool(
                    all(
                        float(np.linalg.norm(endpoint - visited_endpoint))
                        >= _RANDOM_WALK_MINIMUM_ESCAPE_JOINT_NOVELTY_RAD_V745
                        for visited_endpoint in visited_joint_endpoints_v745
                    )
                )
                candidate_diagnostics["novel_endpoint_v745"] += int(novel_endpoint_v745)
                candidate = {
                    "candidate_index": candidate_index,
                    "candidate_source": candidate_source,
                    "correlated": candidate_correlated,
                    "raw_action": candidate_raw,
                    "projected_joint_action": projected_joint_action.copy(),
                    "projection_l2": projection_l2,
                    "projection_format": projection_format,
                    "preflight": candidate_preflight,
                    "predicted_joint_endpoint": endpoint.copy(),
                    "predicted_distance_m": predicted_distance,
                    "predicted_outward_delta_m": predicted_outward_delta,
                    "predicted_alignment": predicted_alignment,
                    "predicted_height_m": predicted_height,
                    "selected_local_escape_v745": False,
                }
                if outward and aligned and inside_height and inside_requested_shell_upper_v745:
                    candidates.append(candidate)
                    candidate_diagnostics["accepted"] += 1
                    if (
                        radial_retreat_candidate_v745
                        and len(candidates) >= _RANDOM_WALK_OUTWARD_PREFLIGHT_SHORTLIST_COUNT_V745
                    ):
                        break
                    continue
                local_escape_eligible_v745 = bool(
                    radial_retreat_candidate_v745
                    and not outward
                    and aligned
                    and inside_height
                    and novel_endpoint_v745
                    and predicted_outward_delta >= -_RANDOM_WALK_MAXIMUM_INWARD_ESCAPE_M_V745
                    and predicted_distance >= initial_precontact_distance
                    and current_distance < lower
                    and walk_step + 1 < maximum_walk_steps
                )
                if local_escape_eligible_v745:
                    candidate["selected_local_escape_v745"] = True
                    local_escape_candidates_v745.append(candidate)
                    candidate_diagnostics["local_escape_eligible_v745"] += 1
            last_candidate_diagnostics = candidate_diagnostics
            if candidates and taskspace_3d_v698:
                # Keep stochastic diversity while avoiding a 3-D Brownian
                # walk that spends the entire reset budget circling one shell.
                candidates.sort(
                    key=lambda row: float(row["predicted_distance_m"]),
                    reverse=True,
                )
                pool_size = max(1, (len(candidates) + 1) // 2)
                chosen = candidates[int(rng.integers(0, pool_size))]
                consecutive_local_escape_steps_v745 = 0
            elif candidates:
                chosen = candidates[int(rng.integers(0, len(candidates)))]
                consecutive_local_escape_steps_v745 = 0
            elif (
                radial_retreat_candidate_v745
                and local_escape_candidates_v745
                and consecutive_local_escape_steps_v745 < _RANDOM_WALK_MAXIMUM_CONSECUTIVE_ESCAPE_STEPS_V745
            ):
                # DLS can place the discrete action set at a one-step radial
                # local maximum.  Take the least-inward novel endpoint through
                # a bounded tabu search.  It may never cross the original safe
                # precontact distance floor; the identical V4 preflight,
                # contact, alignment, and height gates still apply, and the
                # final state must land inside the original exact shell.
                chosen = max(
                    local_escape_candidates_v745,
                    key=lambda row: (
                        float(row["predicted_distance_m"]),
                        -int(row["candidate_index"]),
                    ),
                )
                consecutive_local_escape_steps_v745 += 1
                local_escape_step_count_v745 += 1
            else:
                failure_reason = (
                    "random_walk_has_no_v4_safe_outward_or_bounded_local_escape_candidate_v745"
                    if radial_retreat_candidate_v745
                    else "random_walk_has_no_v4_safe_outward_candidate"
                )
                break
            correlated = np.asarray(chosen["correlated"], dtype=np.float64)
            raw_action = np.asarray(chosen["raw_action"], dtype=np.float32)
            projected_joint_action = np.asarray(
                chosen["projected_joint_action"],
                dtype=np.float32,
            )
            verified_preflight = chosen["preflight"]
            block_before = env.block_xy().copy()
            try:
                translated = guarded.translate(
                    projected_joint_action,
                    verified_preflight=verified_preflight,
                )
            except GuardedJointNoSafeActionV664:
                failure_reason = "random_walk_guard_has_no_safe_action"
                break
            _observation, _reward, terminated, truncated, info = env.step(translated.submitted_joint_action)
            telemetry = transition_contact_telemetry_v22(
                info,
                block_before_xy_m=block_before,
                block_after_xy_m=env.block_xy(),
            )
            distance, alignment = _precontact_geometry_v673(env)
            best_precontact_distance_v745 = max(
                best_precontact_distance_v745,
                distance,
            )
            row = {
                "walk_step": walk_step,
                "sampled_candidate_count": (total_candidate_count),
                "preflight_evaluated_candidate_count": int(candidate_diagnostics["preflight_evaluated"]),
                "v4_safe_outward_candidate_count": len(candidates),
                "v745_bounded_local_escape_candidate_count": len(local_escape_candidates_v745),
                "selected_random_candidate_index": int(chosen["candidate_index"]),
                "selected_candidate_source": str(chosen["candidate_source"]),
                "selected_local_escape_v745": bool(chosen["selected_local_escape_v745"]),
                "selected_predicted_outward_delta_m": float(chosen["predicted_outward_delta_m"]),
                "consecutive_local_escape_steps_v745": (consecutive_local_escape_steps_v745),
                "raw_action": raw_action.tolist(),
                "projected_action": projected_joint_action.tolist(),
                "projection_l2": float(chosen["projection_l2"]),
                "projection_format": str(chosen["projection_format"]),
                "taskspace_3d_v698": taskspace_3d_v698,
                "predicted_tool_precontact_distance_m": float(chosen["predicted_distance_m"]),
                "predicted_precontact_alignment": float(chosen["predicted_alignment"]),
                "guard_selected_scale": float(translated.guard_selected_scale),
                "guard_intervened": bool(translated.guard_intervened),
                "tool_precontact_distance_m": distance,
                "precontact_alignment": alignment,
                "tool_height_m": float(env.tool_xyz()[2]),
                "tool_block_contact_any": bool(telemetry["tool_block_contact_any"]),
                "invalid_tool_block_contact_any": bool(telemetry["invalid_tool_block_contact_any"]),
                "block_displacement_m": float(telemetry["step_block_displacement_m"]),
            }
            walk_rows.append(row)
            visited_joint_endpoints_v745.append(np.asarray(env.data.qpos[:6], dtype=np.float64).copy())
            forbidden_contact_or_motion = bool(
                row["tool_block_contact_any"]
                or row["invalid_tool_block_contact_any"]
                or row["block_displacement_m"] > _RANDOM_WALK_BLOCK_MOTION_TOLERANCE_M_V673
            )
            if forbidden_contact_or_motion:
                failure_reason = "random_walk_touched_or_moved_block"
                break
            if bool(info.get("safety_stop", False) or terminated or truncated):
                failure_reason = "random_walk_environment_terminal"
                break
            tool_height = float(env.tool_xyz()[2])
            if (
                lower <= distance <= upper
                and alignment >= minimum_alignment
                and height_bounds[0] <= tool_height <= height_bounds[1]
            ):
                final_joint = np.asarray(env.data.qpos[:6], dtype=np.float64).copy()
                final_joint[5] = float(env.tool_gripper_joint_position_rad)
                failure_reason = ""
                break

        if final_joint is None:
            rejected.append(
                {
                    "attempt_index": attempt_index,
                    "seed": selected_seed,
                    "failure_reasons": [failure_reason],
                    "random_walk_step_count": len(walk_rows),
                    "random_walk_local_escape_step_count_v745": (local_escape_step_count_v745),
                    "random_walk_maximum_consecutive_local_escape_steps_v745": max(
                        (
                            int(
                                row.get(
                                    "consecutive_local_escape_steps_v745",
                                    0,
                                )
                            )
                            for row in walk_rows
                        ),
                        default=0,
                    ),
                    "initial_precontact_distance_m": (initial_precontact_distance),
                    "requested_precontact_distance_band_m": [lower, upper],
                    "final_precontact_distance_m": (
                        walk_rows[-1]["tool_precontact_distance_m"]
                        if walk_rows
                        else initial_precontact_distance
                    ),
                    "last_candidate_diagnostics": (last_candidate_diagnostics),
                }
            )
            continue

        # Reinstall the dynamically reached physical joint state at time zero.
        # This clears command queues, counters, velocities, and reset-walk
        # history while preserving the same deterministic block/target task.
        restore_stock_gripper_distal_contact_v22(env)
        try:
            env.reset_task_independent_home_v597(
                final_joint,
                seed=selected_seed,
                obstacle=obstacle,
                stress=stress,
            )
        except RuntimeError as error:
            rejected.append(
                {
                    "attempt_index": attempt_index,
                    "seed": selected_seed,
                    "failure_reasons": [f"dynamic_state_install:{error}"],
                }
            )
            continue
        identity_checks = {
            "block_xy": bool(np.allclose(env.block_xy(), block, rtol=0.0, atol=1.0e-10)),
            "target_xy": bool(np.allclose(env.target_xy, target, rtol=0.0, atol=1.0e-12)),
            "joint_position": bool(np.array_equal(env.data.qpos[:6], final_joint)),
            "time_zero": bool(float(env.data.time) == 0.0),
        }
        if not all(identity_checks.values()):
            failed_identity = [name for name, passed in identity_checks.items() if not passed]
            rejected.append(
                {
                    "attempt_index": attempt_index,
                    "seed": selected_seed,
                    "failure_reasons": ["dynamic_install_changed_" + "_and_".join(failed_identity)],
                    "identity_checks": identity_checks,
                }
            )
            continue
        planning_distances = env._tool_planning_signed_distances_for_data(env._ids["block_geom"], env.data)
        safety_distances = env._tool_safety_signed_distances_for_data(env._ids["block_geom"], env.data)
        desk_clearance = float(env._minimum_tool_safety_signed_distance_for_data(env._desk_geom, env.data))
        final_distance, final_alignment = _precontact_geometry_v673(env)
        filtered, filter_reason = env._safety_filter(final_joint)
        final_failures: list[str] = []
        if not np.array_equal(filtered, final_joint) or filter_reason:
            final_failures.append("dynamic_state_not_safety_filter_fixed_point")
        if env._tool_block_contacts() != 0:
            final_failures.append("dynamic_state_has_tool_block_contact")
        if float(np.min(safety_distances)) < 0.0 or desk_clearance < 0.0:
            final_failures.append("dynamic_state_has_penetration")
        if not lower <= final_distance <= upper:
            final_failures.append("dynamic_state_left_requested_shell")
        if final_alignment < minimum_alignment:
            final_failures.append("dynamic_state_lost_precontact_alignment")
        if initial_target_coverage != 0.0 or env.block_target_coverage() != 0.0:
            final_failures.append("task_begins_with_nonzero_target_coverage")
        if final_failures:
            rejected.append(
                {
                    "attempt_index": attempt_index,
                    "seed": selected_seed,
                    "failure_reasons": final_failures,
                }
            )
            continue

        realism = env.episode_domain["realism_v7"]
        realism.update(
            {
                "reset_state_source": reset_source,
                "task_aligned_privileged_reset": True,
                "deployment_reset_equivalent": False,
                "reset_privileged_state_used": [
                    "block_xy",
                    "target_xy",
                    "task_aligned_precontact_joint_position",
                    "v4_guarded_random_walk_outcome",
                ],
                "reset_dynamic_reverse_joint_position_rad": (final_joint.tolist()),
                "home_to_precontact_fraction_label": float(home_to_precontact_fraction),
                "random_walk_physics_steps_before_policy": len(walk_rows),
                "final_installed_state_time_seconds": 0.0,
                "production_admission": False,
            }
        )
        env._reset_collision_audit = {
            "format": (
                "edgearm-v698-dynamic-taskspace-reset-collision-audit-v1"
                if taskspace_3d_v698
                else "edgearm-v673-dynamic-reset-collision-audit-v1"
            ),
            "reset_valid": True,
            "reset_failure_reasons": [],
            "tool_block_contact_count": int(env._tool_block_contacts()),
            "tool_block_signed_distance_m": float(np.min(planning_distances)),
            "tool_block_safety_signed_distance_m": float(np.min(safety_distances)),
            "tool_desk_signed_distance_m": desk_clearance,
            "dynamic_path_executed": True,
            "dynamic_path_contact_free": True,
        }
        realism["reset_collision_audit"] = deepcopy(env._reset_collision_audit)
        action_adapter.arm_expected_reset_v673(final_joint)
        try:
            action_adapter.begin_episode(selected_seed)
        except RuntimeError as error:
            rejected.append(
                {
                    "attempt_index": attempt_index,
                    "seed": selected_seed,
                    "failure_reasons": [f"guarded_runtime:{error}"],
                }
            )
            continue
        renderer.begin_episode(selected_seed)
        audit = {
            "format": reset_format,
            "requested_seed": requested_seed,
            "selected_seed": selected_seed,
            "selected_attempt_index": attempt_index,
            "maximum_attempts": MAX_RESET_ATTEMPTS_V12,
            "retry_stride": RESET_RETRY_STRIDE_V12,
            "rejected_attempts": rejected,
            "obstacle": obstacle,
            "stress": stress,
            "source_type": SOURCE_TYPE,
            "curriculum_spec": spec,
            "initial_exact_precontact_distance_m": (initial_precontact_distance),
            "initial_exact_precontact_alignment": initial_alignment,
            "requested_precontact_distance_band_m": [lower, upper],
            "installed_precontact_distance_m": final_distance,
            "installed_precontact_alignment": final_alignment,
            "installed_joint_position_rad": final_joint.tolist(),
            "initial_block_target_distance_m": (initial_block_target_distance),
            "initial_target_coverage": initial_target_coverage,
            "random_walk_step_count": len(walk_rows),
            "random_walk_trace": walk_rows,
            "random_walk_trace_sha256": canonical_sha256_v1(walk_rows),
            "random_walk_dynamically_executed": True,
            "random_walk_v4_guarded": True,
            "random_walk_taskspace_3d_dls_v698": taskspace_3d_v698,
            "random_walk_radial_retreat_candidate_v745": (radial_retreat_candidate_v745),
            "random_walk_bounded_local_escape_v745": (radial_retreat_candidate_v745),
            "random_walk_seed_shuffled_candidate_order_v745": (radial_retreat_candidate_v745),
            "random_walk_outward_preflight_shortlist_count_v745": (
                _RANDOM_WALK_OUTWARD_PREFLIGHT_SHORTLIST_COUNT_V745 if radial_retreat_candidate_v745 else None
            ),
            "random_walk_reset_distribution_changed_from_v698": (radial_retreat_candidate_v745),
            "random_walk_local_escape_step_count_v745": (local_escape_step_count_v745),
            "random_walk_monotonic_outward": bool(local_escape_step_count_v745 == 0),
            "random_walk_maximum_inward_escape_m_v745": (
                _RANDOM_WALK_MAXIMUM_INWARD_ESCAPE_M_V745 if radial_retreat_candidate_v745 else None
            ),
            "random_walk_maximum_consecutive_escape_steps_v745": (
                _RANDOM_WALK_MAXIMUM_CONSECUTIVE_ESCAPE_STEPS_V745 if radial_retreat_candidate_v745 else None
            ),
            "random_walk_minimum_escape_joint_novelty_rad_v745": (
                _RANDOM_WALK_MINIMUM_ESCAPE_JOINT_NOVELTY_RAD_V745 if radial_retreat_candidate_v745 else None
            ),
            "random_walk_initial_safe_distance_floor_m_v745": (
                initial_precontact_distance if radial_retreat_candidate_v745 else None
            ),
            "random_walk_planar_joint_projection_v673": (not taskspace_3d_v698),
            "random_walk_alignment_threshold": minimum_alignment,
            "random_walk_minimum_outward_progress_m": (minimum_outward_progress),
            "random_walk_height_bounds_m": list(height_bounds),
            "random_walk_contact_free": True,
            "random_walk_is_reset_generation_only": True,
            "random_walk_added_to_replay": False,
            "intermediate_state_is_reset_not_executed_trajectory": True,
            "task_aligned_privileged_reset": True,
            "eligible_for_final_data": False,
            "bulk_vla_data_use_allowed": False,
            "expert_calls": 0,
            "expert_actions": 0,
            "expert_paths": 0,
            "behavior_cloning_steps": 0,
            "deployment_reset_equivalent": False,
            "production_admission": False,
            "adapter_runtime": deepcopy(env.episode_domain["stock_gripper_taskframe_runtime_v22"]),
            "action_adapter_config": asdict(action_adapter.config),
        }
        action_adapter.dynamic_reverse_reset_audit_v673 = deepcopy(audit)
        env.episode_domain[
            (
                "dynamic_taskspace_reverse_curriculum_reset_v698"
                if taskspace_3d_v698
                else "dynamic_reverse_curriculum_reset_v673"
            )
        ] = deepcopy(audit)
        return audit

    reasons = "; ".join(
        (
            f"seed={row['seed']}:{','.join(row['failure_reasons'])}"
            + (
                f":candidate_counts={row['last_candidate_diagnostics']}"
                if row.get("last_candidate_diagnostics")
                else ""
            )
        )
        for row in rejected
    )
    raise RuntimeError(
        (
            "V698 dynamic task-space reverse reset exhausted deterministic attempts: "
            if taskspace_3d_v698
            else "V673 dynamic reverse reset exhausted deterministic attempts: "
        )
        + reasons
    )


__all__ = [
    "DYNAMIC_REVERSE_CURRICULUM_RESET_FORMAT_V673",
    "DYNAMIC_REVERSE_CURRICULUM_SOURCE_V673",
    "DYNAMIC_TASKSPACE_REVERSE_CURRICULUM_RESET_FORMAT_V698",
    "DYNAMIC_TASKSPACE_REVERSE_CURRICULUM_SOURCE_V698",
    "DYNAMIC_TASKSPACE_RADIAL_RETREAT_RESET_FORMAT_V745",
    "DYNAMIC_TASKSPACE_RADIAL_RETREAT_SOURCE_V745",
    "StockGripperDynamicReverseTaskFrameAdapterV673",
    "dynamic_taskspace_reverse_curriculum_spec_v698",
    "dynamic_reverse_curriculum_spec_v673",
    "reset_stock_dynamic_reverse_episode_v673",
]
