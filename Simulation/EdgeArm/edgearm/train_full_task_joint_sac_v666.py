"""Train the full-task guarded-joint RL data-generator teacher.

Unlike V654--V658, V666 never treats precontact acquisition as an episode
success or terminal.  Curriculum changes only the initial arm state.  Its
easiest tier puts the stock gripper at the audited V22 task-aligned
precontact reset while leaving the block at the full task start, so the same
policy must still continue through contact, full target-directed transport,
target coverage, and the environment's exact 90-frame / three-second strict
hold.

Fast training executes bounded five-joint deltas through the exact V10 plant
and its built-in joint/workspace filter.  These transitions are learning-only
and categorically ineligible for data export.  Guarded V4 execution is a
separate supported mode and is mandatory for final validation/generation.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from .asymmetric_multiview_ppo_v1 import (
    SOURCE_TYPE,
    canonical_sha256_v1,
    sha256_file_v1,
)
from .causal_smooth_relay_v646 import CausalSmoothRelayConfigV646
from .diagnose_exact_home_taskframe_reachability_v661 import (
    LONG_RANGE_STAGE_INDEX_V661,
    _NoImageResetRendererV661,
)
from .dynamic_reverse_curriculum_reset_v673 import (
    StockGripperDynamicReverseTaskFrameAdapterV673,
    reset_stock_dynamic_reverse_episode_v673,
)
from .full_task_start_contract_v669 import (
    FULL_TASK_MAXIMUM_INITIAL_COVERAGE_V669,
    FULL_TASK_MAXIMUM_INITIAL_DISTANCE_M_V669,
    FULL_TASK_MINIMUM_INITIAL_DISTANCE_M_V669,
    FULL_TASK_START_CONTRACT_FORMAT_V669,
    audit_full_task_start_v669,
)
from .guarded_joint_delta_action_v664 import (
    ARM_JOINT_ACTION_DIM_V664,
    GuardedJointDeltaActionConfigV664,
    GuardedJointDeltaActionV664,
    GuardedJointNoSafeActionV664,
)
from .joint_goal_sac_v665 import (
    JOINT_ACQUISITION_CONTEXT_FORMAT_V683,
    JOINT_ACQUISITION_PROGRESS_FORMAT_V686,
    JOINT_GOAL_OBSERVATION_DIM_V665,
    JOINT_GOAL_SAC_FORMAT_V665,
    JOINT_TASKFRAME_FEATURE_FORMAT_V682,
    JointGoalReplayV665,
    JointGoalSACBundleV665,
    JointGoalSACConfigV665,
    initialize_joint_goal_sac_v665,
    joint_goal_checkpoint_payload_v665,
    joint_goal_observation_v665,
    joint_teacher_reward_v665,
    pretrain_joint_acquisition_progress_v686,
    pretrain_joint_goal_acquisition_v684,
    update_joint_goal_sac_v665,
)
from .orientation_projected_joint_action_v667 import (
    ORIENTATION_PROJECTED_JOINT_ACTION_FORMAT_V667,
    orientation_constraint_state_v667,
    project_joint_action_v667,
)
from .planar_push_projected_joint_action_v668 import (
    PLANAR_PUSH_PROJECTED_JOINT_ACTION_FORMAT_V668,
    planar_push_constraint_state_v668,
    project_planar_push_joint_action_v668,
)
from .privileged_effect_state_v1 import build_privileged_effect_state_v1
from .sim2real_env_v10 import RealisticEdgeArmEnvV10
from .stock_gripper_taskframe_v22 import (
    StockGripperTaskFrameAdapterV22,
    reset_stock_taskframe_episode_v22,
    transition_contact_telemetry_v22,
)
from .task_independent_certified_long_range_reset_v649 import (
    reset_stock_home_certified_long_range_episode_v649,
)
from .task_independent_home_reset_v597 import (
    StockGripperHomeTaskFrameAdapterV597,
)
from .train_causal_smooth_long_range_relay_v650 import _stage_runtime_v650
from .train_goal_conditioned_her_sac_v43 import (
    _append_jsonl,
    _atomic_json,
    _load_collection_contract,
    _utc_now,
)


FULL_TASK_JOINT_SAC_TRAINING_FORMAT_V666 = "edgearm-v666-full-task-five-joint-sac-training-v23"
_JOINT_GOAL_SAC_FORMAT_V682 = "edgearm-v665-full-task-joint-goal-sac-v7"
_JOINT_GOAL_SAC_FORMAT_V683 = "edgearm-v665-full-task-joint-goal-sac-v8"
_JOINT_GOAL_SAC_FORMAT_V684 = "edgearm-v665-full-task-joint-goal-sac-v9"
_JOINT_GOAL_SAC_FORMAT_V685 = "edgearm-v665-full-task-joint-goal-sac-v10"
_JOINT_GOAL_SAC_FORMAT_V686 = "edgearm-v665-full-task-joint-goal-sac-v11"
_PRE_TASKFRAME_SAC_FORMATS_V682 = {
    "edgearm-v665-full-task-joint-goal-sac-v2",
    "edgearm-v665-full-task-joint-goal-sac-v3",
    "edgearm-v665-full-task-joint-goal-sac-v4",
    "edgearm-v665-full-task-joint-goal-sac-v5",
    "edgearm-v665-full-task-joint-goal-sac-v6",
}
CONTACT_ADAPTIVE_PROJECTION_FORMAT_V681 = "edgearm-v681-contact-latched-orientation-to-planar-projection-v1"
_CURRICULUM_FRACTIONS_V666 = (
    1.0,
    0.99,
    0.98,
    0.95,
    0.90,
    0.80,
    0.65,
    0.45,
    0.25,
    0.0,
)
_FAST_EXECUTION_MODES_V666 = {
    "fast_v10_learning_only",
    "fast_v10_orientation_projected_learning_only",
    "fast_v10_planar_push_projected_learning_only",
    "fast_v10_contact_adaptive_projected_learning_only",
}
_GUARDED_EXECUTION_MODES_V666 = {
    "guarded_v4",
    "guarded_v4_orientation_projected",
    "guarded_v4_planar_push_projected",
    "guarded_v4_contact_adaptive_projected",
}
_ORIENTATION_PROJECTED_MODES_V666 = {
    "fast_v10_orientation_projected_learning_only",
    "guarded_v4_orientation_projected",
}
_PLANAR_PUSH_PROJECTED_MODES_V666 = {
    "fast_v10_planar_push_projected_learning_only",
    "guarded_v4_planar_push_projected",
}
_CONTACT_ADAPTIVE_PROJECTED_MODES_V681 = {
    "fast_v10_contact_adaptive_projected_learning_only",
    "guarded_v4_contact_adaptive_projected",
}
_STRUCTURALLY_PROJECTED_MODES_V666 = (
    _ORIENTATION_PROJECTED_MODES_V666
    | _PLANAR_PUSH_PROJECTED_MODES_V666
    | _CONTACT_ADAPTIVE_PROJECTED_MODES_V681
)
_EXECUTION_MODES_V666 = _FAST_EXECUTION_MODES_V666 | _GUARDED_EXECUTION_MODES_V666
_UNATTRIBUTED_MOTION_COUPLING_CLEARANCE_M_V666 = 0.002
_WARM_START_COMPATIBLE_SAC_FORMATS_V670 = {
    *_PRE_TASKFRAME_SAC_FORMATS_V682,
    _JOINT_GOAL_SAC_FORMAT_V682,
    _JOINT_GOAL_SAC_FORMAT_V683,
    _JOINT_GOAL_SAC_FORMAT_V684,
    _JOINT_GOAL_SAC_FORMAT_V685,
    _JOINT_GOAL_SAC_FORMAT_V686,
    JOINT_GOAL_SAC_FORMAT_V665,
}
_VALUE_FUNCTION_COMPATIBLE_SAC_FORMATS_V680 = {
    "edgearm-v665-full-task-joint-goal-sac-v5",
    "edgearm-v665-full-task-joint-goal-sac-v6",
    _JOINT_GOAL_SAC_FORMAT_V682,
    _JOINT_GOAL_SAC_FORMAT_V683,
    _JOINT_GOAL_SAC_FORMAT_V684,
    _JOINT_GOAL_SAC_FORMAT_V685,
    _JOINT_GOAL_SAC_FORMAT_V686,
    JOINT_GOAL_SAC_FORMAT_V665,
}


@dataclass(frozen=True)
class FullTaskJointTrainingConfigV666:
    warmup_transitions: int = 512
    offline_acquisition_self_imitation_updates: int = 0
    offline_acquisition_progress_model_updates: int = 0
    updates_per_environment_step: int = 1
    random_action_standard_deviation: float = 0.35
    random_action_correlation: float = 0.85
    exact_home_probe_interval: int = 5
    frontier_retention_interval: int = 4
    mastery_recent_window: int = 6
    mastery_minimum_contact_episodes: int = 4
    mastery_minimum_effectful_episodes: int = 3
    mastery_minimum_net_progress_m: float = 0.020
    mastery_minimum_retention_episodes: int = 3
    mastery_minimum_retained_contact_steps: int = 60
    mastery_minimum_contact_retention_fraction: float = 0.60
    mastery_maximum_contact_loss_events: int = 6
    mastery_minimum_retained_progress_m: float = 0.005
    mastery_strict_success_override: int = 2
    replay_save_interval_episodes: int = 1
    policy_candidate_count: int = 16
    policy_feasibility_shortlist_fraction: float = 0.25
    exact_guard_candidate_preflight_count: int = 2

    def validate(self) -> None:
        for name in (
            "warmup_transitions",
            "offline_acquisition_self_imitation_updates",
            "offline_acquisition_progress_model_updates",
            "updates_per_environment_step",
            "exact_home_probe_interval",
            "frontier_retention_interval",
            "mastery_recent_window",
            "mastery_minimum_contact_episodes",
            "mastery_minimum_effectful_episodes",
            "mastery_minimum_retention_episodes",
            "mastery_minimum_retained_contact_steps",
            "mastery_maximum_contact_loss_events",
            "mastery_strict_success_override",
            "replay_save_interval_episodes",
            "policy_candidate_count",
            "exact_guard_candidate_preflight_count",
        ):
            value = getattr(self, name)
            minimum = (
                0
                if name
                in {
                    "warmup_transitions",
                    "offline_acquisition_self_imitation_updates",
                    "offline_acquisition_progress_model_updates",
                }
                else 1
            )
            if type(value) is not int or value < minimum:
                raise ValueError(f"V666 {name} is invalid")
        for name in (
            "random_action_standard_deviation",
            "random_action_correlation",
            "mastery_minimum_net_progress_m",
            "mastery_minimum_contact_retention_fraction",
            "mastery_minimum_retained_progress_m",
            "policy_feasibility_shortlist_fraction",
        ):
            value = float(getattr(self, name))
            minimum = (
                0.0
                if name
                in {
                    "random_action_standard_deviation",
                    "random_action_correlation",
                }
                else 1.0e-6
            )
            if not np.isfinite(value) or not minimum <= value <= 1.0:
                raise ValueError(f"V666 {name} is invalid")
        if self.mastery_minimum_contact_episodes > self.mastery_recent_window:
            raise ValueError("V666 contact mastery count exceeds its window")
        if self.mastery_minimum_effectful_episodes > self.mastery_recent_window:
            raise ValueError("V666 effect mastery count exceeds its window")
        if self.mastery_minimum_retention_episodes > self.mastery_recent_window:
            raise ValueError("V679 retention mastery count exceeds its window")
        if self.exact_guard_candidate_preflight_count > self.policy_candidate_count:
            raise ValueError("V666 exact preflight count exceeds candidate count")


def _tier_code_v666(fraction: float) -> str:
    if fraction == 1.0:
        return "precontact_v22"
    return "home" if fraction == 0.0 else f"approach_{int(round(100 * fraction)):02d}"


def select_start_fraction_v666(
    *,
    episode_index: int,
    records: Sequence[dict[str, Any]],
    config: FullTaskJointTrainingConfigV666 | None = None,
) -> tuple[float, dict[str, Any]]:
    """Choose a reverse-curriculum start without redefining task success."""

    selected = config or FullTaskJointTrainingConfigV666()
    selected.validate()
    if type(episode_index) is not int or episode_index < 1:
        raise ValueError("V666 episode index is invalid")
    evidence: dict[str, Any] = {}
    frontier = _CURRICULUM_FRACTIONS_V666[-2]
    mastered: list[str] = []
    for fraction in _CURRICULUM_FRACTIONS_V666[:-1]:
        code = _tier_code_v666(fraction)
        rows = [row for row in records if row.get("start_tier") == code]
        recent = rows[-selected.mastery_recent_window :]
        clean_rows = [
            row
            for row in recent
            if int(row.get("invalid_contact_steps", 0)) == 0
            and not bool(row.get("physical_contact_audit_failure", False))
        ]
        contacts = sum(int(bool(row.get("valid_contact_reached"))) for row in clean_rows)
        progress_qualified = sum(
            int(
                bool(row.get("effectful_block_motion"))
                and bool(row.get("valid_contact_reached"))
                and float(row.get("net_object_target_progress_m", 0.0))
                >= selected.mastery_minimum_net_progress_m
            )
            for row in clean_rows
        )
        retention_qualified = 0
        retention_fractions: list[float] = []
        for row in clean_rows:
            if not all(
                key in row
                for key in (
                    "valid_contact_steps",
                    "contact_absent_steps",
                    "contact_loss_event_steps",
                    "steps_executed",
                )
            ):
                continue
            valid_steps = int(row["valid_contact_steps"])
            absent_steps = int(row["contact_absent_steps"])
            loss_events = int(row["contact_loss_event_steps"])
            executed_steps = int(row["steps_executed"])
            transport_steps = valid_steps + absent_steps
            if (
                valid_steps < 0
                or absent_steps < 0
                or loss_events < 0
                or executed_steps < 1
                or transport_steps > executed_steps
            ):
                continue
            retention_fraction = valid_steps / max(transport_steps, 1)
            retention_fractions.append(retention_fraction)
            retention_qualified += int(
                bool(row.get("valid_contact_reached"))
                and bool(row.get("effectful_block_motion"))
                and valid_steps >= selected.mastery_minimum_retained_contact_steps
                and retention_fraction >= selected.mastery_minimum_contact_retention_fraction
                and loss_events <= selected.mastery_maximum_contact_loss_events
                and float(row.get("net_object_target_progress_m", 0.0))
                >= selected.mastery_minimum_retained_progress_m
            )
        successes = sum(int(bool(row.get("strict_success"))) for row in recent)
        # The closest start must prove transport, while each progressively
        # harder approach tier only has to prove clean acquisition.  Requiring
        # another two-centimetre push at every tiny interpolation step wastes
        # the rare contact signal and makes reverse curriculum unnecessarily
        # coarse.  Full strict success remains the only task success.
        acquisition_tier = bool(fraction < 1.0)
        evidence_mastered = bool(
            contacts >= selected.mastery_minimum_contact_episodes
            and retention_qualified >= selected.mastery_minimum_retention_episodes
            if acquisition_tier
            else (
                contacts >= selected.mastery_minimum_contact_episodes
                and progress_qualified >= selected.mastery_minimum_effectful_episodes
                and retention_qualified >= selected.mastery_minimum_retention_episodes
            )
        )
        tier_mastered = bool(
            len(recent) >= selected.mastery_recent_window
            and (evidence_mastered or successes >= selected.mastery_strict_success_override)
        )
        evidence[code] = {
            "episode_count": len(rows),
            "recent_episode_count": len(recent),
            "recent_contact_episode_count": contacts,
            "recent_progress_qualified_episode_count": progress_qualified,
            "recent_retention_qualified_episode_count": retention_qualified,
            "recent_mean_contact_retention_fraction": (
                float(np.mean(retention_fractions)) if retention_fractions else None
            ),
            "retention_requires_complete_v676_metrics": True,
            "minimum_retained_contact_steps": (selected.mastery_minimum_retained_contact_steps),
            "minimum_contact_retention_fraction": (selected.mastery_minimum_contact_retention_fraction),
            "maximum_contact_loss_events": (selected.mastery_maximum_contact_loss_events),
            "minimum_retained_progress_m": (selected.mastery_minimum_retained_progress_m),
            "minimum_retention_episodes": (selected.mastery_minimum_retention_episodes),
            "progress_qualified_requires_clean_valid_effectful_episode": True,
            "minimum_net_progress_m": selected.mastery_minimum_net_progress_m,
            "recent_strict_success_count": successes,
            "mastery_evidence_kind": (
                "clean_contact_acquisition_and_retention"
                if acquisition_tier
                else "clean_contact_retention_and_two_centimetre_transport"
            ),
            "mastered": tier_mastered,
        }
        if tier_mastered:
            mastered.append(code)
            continue
        frontier = fraction
        break
    else:
        # After every non-Home tier is mastered, ordinary episodes must finally
        # start from exact Home rather than remaining stuck at approach_25.
        frontier = 0.0
    if episode_index % selected.exact_home_probe_interval == 0:
        chosen = 0.0
        reason = "periodic_exact_home_full_task_probe"
    elif episode_index % selected.frontier_retention_interval == 0:
        chosen = max(
            frontier,
            _CURRICULUM_FRACTIONS_V666[max(_CURRICULUM_FRACTIONS_V666.index(frontier) - 1, 0)],
        )
        reason = "easier_frontier_retention"
    else:
        chosen = frontier
        reason = "joint_full_task_reverse_curriculum_frontier"
    return chosen, {
        "format": FULL_TASK_JOINT_SAC_TRAINING_FORMAT_V666,
        "episode_index": episode_index,
        "selected_fraction": chosen,
        "selected_tier": _tier_code_v666(chosen),
        "frontier_fraction": frontier,
        "frontier_tier": _tier_code_v666(frontier),
        "selection_reason": reason,
        "mastered_tiers": mastered,
        "evidence_before_selection": evidence,
        "precontact_is_terminal": False,
        "strict_three_second_success_is_only_positive_terminal": True,
        "non_home_final_data_eligible": False,
        "production_admission": False,
    }


def fast_joint_command_v666(
    *,
    policy_action: np.ndarray,
    reported_joint_position_rad: np.ndarray,
    zero_offset_rad: np.ndarray,
    fixed_gripper_joint_position_rad: float,
    plant_max_joint_delta_rad: float,
    policy_joint_target_step_rad: float,
) -> np.ndarray:
    """Map V664 semantics to a direct V10 learning-only plant command."""

    action = np.asarray(policy_action, dtype=np.float64)
    reported = np.asarray(reported_joint_position_rad, dtype=np.float64)
    zero = np.asarray(zero_offset_rad, dtype=np.float64)
    plant_step = float(plant_max_joint_delta_rad)
    policy_step = float(policy_joint_target_step_rad)
    gripper = float(fixed_gripper_joint_position_rad)
    if (
        action.shape != (ARM_JOINT_ACTION_DIM_V664,)
        or reported.shape != (6,)
        or zero.shape != (6,)
        or not np.all(np.isfinite(np.r_[action, reported, zero, plant_step, policy_step, gripper]))
        or plant_step <= 0.0
        or policy_step <= 0.0
        or policy_step > plant_step
    ):
        raise ValueError("V666 fast joint-command inputs are invalid")
    result = np.zeros(6, dtype=np.float64)
    result[:5] = np.clip(action, -1.0, 1.0) * (policy_step / plant_step)
    physical_reference = reported[5] - zero[5]
    result[5] = np.clip((gripper - physical_reference) / plant_step, -1.0, 1.0)
    return result.astype(np.float32)


def unattributed_motion_audit_v666(
    *,
    displacement_is_effectful: bool,
    contact_before: bool,
    valid_contact: bool,
    raw_tool_contact: bool,
    minimum_safety_only_clearance_m: float,
    coupling_clearance_m: float = (_UNATTRIBUTED_MOTION_COUPLING_CLEARANCE_M_V666),
) -> tuple[bool, bool]:
    """Separate passive object drift from robot-coupled untracked motion."""

    if any(
        type(value) is not bool
        for value in (
            displacement_is_effectful,
            contact_before,
            valid_contact,
            raw_tool_contact,
        )
    ):
        raise TypeError("V666 unattributed-motion flags must be booleans")
    clearance = float(minimum_safety_only_clearance_m)
    coupling = float(coupling_clearance_m)
    if not np.isfinite(clearance) or not np.isfinite(coupling) or coupling <= 0.0:
        raise ValueError("V666 unattributed-motion clearance is invalid")
    observed = bool(
        displacement_is_effectful and not contact_before and not valid_contact and not raw_tool_contact
    )
    robot_coupled_failure = bool(observed and clearance <= coupling)
    return observed, robot_coupled_failure


def applied_policy_action_from_plant_v666(
    applied_plant_action: np.ndarray,
    *,
    plant_max_joint_delta_rad: float,
    policy_joint_target_step_rad: float,
) -> np.ndarray:
    applied = np.asarray(applied_plant_action, dtype=np.float64)
    plant_step = float(plant_max_joint_delta_rad)
    policy_step = float(policy_joint_target_step_rad)
    if (
        applied.shape != (6,)
        or not np.all(np.isfinite(applied))
        or plant_step <= 0.0
        or policy_step <= 0.0
        or policy_step > plant_step
    ):
        raise ValueError("V666 applied plant-action inputs are invalid")
    return np.clip(applied[:5] * (plant_step / policy_step), -1.0, 1.0).astype(np.float32)


def _precontact_geometry_v666(env: RealisticEdgeArmEnvV10) -> tuple[float, float]:
    block = env.block_xy().copy()
    target = env.target_xy.copy()
    direction = target - block
    norm = float(np.linalg.norm(direction))
    forward = np.asarray([1.0, 0.0], dtype=np.float64) if norm <= 1.0e-7 else direction / norm
    goal = np.r_[block - 0.055 * forward, 0.055]
    distance = float(np.linalg.norm(env.tool_xyz() - goal))
    rotation = env.data.site_xmat[env._ids["tool_site"]].reshape(3, 3)
    normal_xy = rotation[:, 1][:2]
    horizontal_norm = float(np.linalg.norm(normal_xy))
    heading_alignment = abs(float(np.dot(normal_xy, forward))) / max(horizontal_norm, 1.0e-7)
    # A face aimed at the target but tilted toward the desk is not a valid
    # pushing pose.  The quality must therefore include both heading and
    # vertical-face horizontal norm, matching the contact semantic gate.
    orientation_quality = min(horizontal_norm, heading_alignment)
    return distance, float(np.clip(orientation_quality, 0.0, 1.0))


def _controller_target_v666(
    env: RealisticEdgeArmEnvV10,
    adapter: StockGripperTaskFrameAdapterV22,
    execution_mode: str,
) -> np.ndarray:
    if execution_mode in _GUARDED_EXECUTION_MODES_V666:
        value = adapter._latched_joint_target
        if value is None:
            raise RuntimeError("V666 guarded controller target is missing")
        return np.asarray(value, dtype=np.float32).copy()
    return np.asarray(env.data.qpos[:6], dtype=np.float32).copy()


def projection_regime_v681(
    execution_mode: str,
    *,
    admissible_contact_latched: bool,
) -> str:
    """Choose structural DOFs without prescribing a task-space route."""

    if execution_mode not in _EXECUTION_MODES_V666 or type(admissible_contact_latched) is not bool:
        raise ValueError("V681 projection-regime request is invalid")
    if execution_mode in _CONTACT_ADAPTIVE_PROJECTED_MODES_V681:
        return "planar" if admissible_contact_latched else "orientation"
    if execution_mode in _PLANAR_PUSH_PROJECTED_MODES_V666:
        return "planar"
    if execution_mode in _ORIENTATION_PROJECTED_MODES_V666:
        return "orientation"
    return "none"


def _structurally_project_action_v672(
    env: RealisticEdgeArmEnvV10,
    action: np.ndarray,
    execution_mode: str,
    *,
    admissible_contact_latched: bool,
) -> tuple[np.ndarray, float, bool]:
    regime = projection_regime_v681(
        execution_mode,
        admissible_contact_latched=admissible_contact_latched,
    )
    if regime == "planar":
        projected = project_planar_push_joint_action_v668(env, action)
        return (
            projected.projected_action,
            projected.projection_l2,
            projected.intervened,
        )
    if regime == "orientation":
        projected = project_joint_action_v667(env, action)
        return (
            projected.projected_action,
            projected.projection_l2,
            projected.intervened,
        )
    return np.asarray(action, dtype=np.float32).copy(), 0.0, False


def _policy_action_v666(
    bundle: JointGoalSACBundleV665,
    env: RealisticEdgeArmEnvV10,
    observation: np.ndarray,
    *,
    warmup: bool,
    random_state: np.ndarray,
    rng: np.random.Generator,
    config: FullTaskJointTrainingConfigV666,
    execution_mode: str,
    admissible_contact_latched: bool,
) -> tuple[
    np.ndarray,
    np.ndarray,
    dict[str, Any],
    np.ndarray,
    np.ndarray,
]:
    if warmup:
        innovation = rng.normal(
            0.0,
            config.random_action_standard_deviation,
            ARM_JOINT_ACTION_DIM_V664,
        )
        scale = np.sqrt(max(1.0 - config.random_action_correlation**2, 0.0))
        next_random = config.random_action_correlation * random_state + scale * innovation
        random_action = np.clip(next_random, -1.0, 1.0).astype(np.float32)
        return (
            random_action,
            next_random,
            {
                "format": "edgearm-v671-feasibility-ranked-policy-action-v1",
                "warmup_random_action": True,
                "candidate_count": 1,
                "shortlist_count": 1,
                "selected_candidate_index": 0,
                "selected_predicted_feasibility": None,
                "maximum_candidate_predicted_feasibility": None,
                "selected_q_value": None,
                "selection_regime": "warmup_random_action",
                "selected_acquisition_score": None,
                "selected_acquisition_axis_progress_m": None,
                "candidate_predicted_feasibility": [None],
                "candidate_q_value": [None],
                "candidate_acquisition_score": None,
                "candidate_acquisition_axis_progress_m": None,
                "learned_preference_order": [0],
                "production_admission": False,
            },
            random_action[None, :].copy(),
            np.asarray([0], dtype=np.int64),
        )
    device = next(bundle.actor.parameters()).device
    tensor = torch.from_numpy(observation).to(device).unsqueeze(0)
    candidate_count = config.policy_candidate_count
    repeated = tensor.repeat(candidate_count, 1)
    bundle.actor.eval()
    bundle.critic.eval()
    bundle.feasibility.eval()
    bundle.acquisition_progress_model.eval()
    with torch.no_grad():
        candidates, _log_probability = bundle.actor.sample(repeated)
        deterministic, _ = bundle.actor.sample(tensor, deterministic=True)
        candidates[0] = deterministic[0]
        feasibility = torch.sigmoid(bundle.feasibility(repeated, candidates))
        q1, q2 = bundle.critic(repeated, candidates)
        q_value = torch.minimum(q1, q2)
    candidate_actions_np = candidates.cpu().numpy().astype(np.float32)
    acquisition_before_contact = bool(not admissible_contact_latched)
    if acquisition_before_contact:
        candidate_execution_actions_np = np.stack(
            [
                _structurally_project_action_v672(
                    env,
                    candidate,
                    execution_mode,
                    admissible_contact_latched=False,
                )[0]
                for candidate in candidate_actions_np
            ]
        ).astype(np.float32)
        candidate_execution_actions = torch.from_numpy(candidate_execution_actions_np).to(device)
        with torch.no_grad():
            acquisition_progress1, acquisition_progress2 = bundle.acquisition_progress_model(
                repeated,
                candidate_execution_actions,
            )
        conservative_acquisition_progress = torch.minimum(
            acquisition_progress1,
            acquisition_progress2,
        )
        acquisition_progress_np = conservative_acquisition_progress.cpu().numpy().astype(np.float64)
    else:
        candidate_execution_actions_np = candidate_actions_np.copy()
        acquisition_progress_np = np.zeros((candidate_count, 3), dtype=np.float64)
    feasibility_np = feasibility.cpu().numpy().astype(np.float64)
    q_np = q_value.cpu().numpy().astype(np.float64)
    shortlist_count = max(
        1,
        int(np.ceil(candidate_count * config.policy_feasibility_shortlist_fraction)),
    )
    if acquisition_before_contact:
        preference_order, acquisition_score_np = acquisition_axis_ranked_candidate_order_v686(
            feasibility_np,
            acquisition_progress_np,
            shortlist_count=shortlist_count,
        )
        remaining_score = acquisition_score_np
        selection_regime = "exact_home_axis_progress_v686"
    else:
        preference_order = feasibility_ranked_candidate_order_v672(
            feasibility_np,
            q_np,
            shortlist_count=shortlist_count,
        )
        acquisition_score_np = np.full(candidate_count, np.nan, dtype=np.float64)
        remaining_score = feasibility_np
        selection_regime = "contact_latched_task_value_v672"
    full_preference_order = complete_candidate_preference_order_v677(
        preference_order,
        feasibility_np,
        remaining_score=remaining_score,
    )
    remaining_order = full_preference_order[preference_order.size :]
    selected_index = int(preference_order[0])
    return (
        candidates[selected_index].cpu().numpy().astype(np.float32),
        random_state,
        {
            "format": "edgearm-v671-feasibility-ranked-policy-action-v1",
            "warmup_random_action": False,
            "candidate_count": candidate_count,
            "shortlist_count": shortlist_count,
            "selected_candidate_index": selected_index,
            "deterministic_candidate_selected": selected_index == 0,
            "selected_predicted_feasibility": float(feasibility_np[selected_index]),
            "maximum_candidate_predicted_feasibility": float(np.max(feasibility_np)),
            "mean_candidate_predicted_feasibility": float(np.mean(feasibility_np)),
            "selected_q_value": float(q_np[selected_index]),
            "selection_regime": selection_regime,
            "selected_acquisition_score": (
                float(acquisition_score_np[selected_index]) if acquisition_before_contact else None
            ),
            "selected_acquisition_axis_progress_m": (
                acquisition_progress_np[selected_index].tolist() if acquisition_before_contact else None
            ),
            "candidate_predicted_feasibility": feasibility_np.tolist(),
            "candidate_q_value": q_np.tolist(),
            "candidate_acquisition_score": (
                acquisition_score_np.tolist() if acquisition_before_contact else None
            ),
            "candidate_acquisition_axis_progress_m": (
                acquisition_progress_np.tolist() if acquisition_before_contact else None
            ),
            "candidate_execution_action": (
                candidate_execution_actions_np.tolist() if acquisition_before_contact else None
            ),
            "selected_execution_action": (
                candidate_execution_actions_np[selected_index].tolist()
                if acquisition_before_contact
                else None
            ),
            "learned_preference_order": full_preference_order.tolist(),
            "learned_primary_preference_order": preference_order.tolist(),
            "learned_emergency_preference_order": remaining_order.tolist(),
            "production_admission": False,
        },
        candidate_actions_np,
        full_preference_order,
    )


def feasibility_ranked_candidate_index_v671(
    predicted_feasibility: np.ndarray,
    q_value: np.ndarray,
    *,
    shortlist_count: int,
) -> int:
    """Choose task value only inside the most feasible policy candidates."""

    order = feasibility_ranked_candidate_order_v672(
        predicted_feasibility,
        q_value,
        shortlist_count=shortlist_count,
    )
    return int(order[0])


def feasibility_ranked_candidate_order_v672(
    predicted_feasibility: np.ndarray,
    q_value: np.ndarray,
    *,
    shortlist_count: int,
) -> np.ndarray:
    """Rank the learned-feasibility shortlist by task value."""

    feasibility = np.asarray(predicted_feasibility, dtype=np.float64)
    value = np.asarray(q_value, dtype=np.float64)
    if (
        feasibility.ndim != 1
        or value.shape != feasibility.shape
        or feasibility.size < 1
        or not np.all(np.isfinite(feasibility))
        or not np.all(np.isfinite(value))
        or np.any(feasibility < 0.0)
        or np.any(feasibility > 1.0)
        or type(shortlist_count) is not int
        or not 1 <= shortlist_count <= feasibility.size
    ):
        raise ValueError("V671 candidate ranking inputs are invalid")
    feasibility_order = np.argsort(-feasibility, kind="stable")
    shortlist = feasibility_order[:shortlist_count]
    task_order = np.argsort(-value[shortlist], kind="stable")
    return shortlist[task_order].astype(np.int64)


def acquisition_axis_ranked_candidate_order_v686(
    predicted_feasibility: np.ndarray,
    predicted_axis_progress_m: np.ndarray,
    *,
    shortlist_count: int,
    maximum_lateral_regression_m: float = 2.5e-4,
    maximum_height_regression_m: float = 5.0e-4,
) -> tuple[np.ndarray, np.ndarray]:
    """Rank feasible Home actions by pessimistic three-axis progress.

    Positive values reduce absolute task-frame error.  A candidate that stays
    inside the one-step lateral/height regression envelope is always preferred
    to one that leaves it; the fallback score strongly penalizes regressions
    so a model cannot trade a large sideways excursion for forward progress.
    """

    feasibility = np.asarray(predicted_feasibility, dtype=np.float64)
    progress = np.asarray(predicted_axis_progress_m, dtype=np.float64)
    thresholds = np.asarray(
        (maximum_lateral_regression_m, maximum_height_regression_m),
        dtype=np.float64,
    )
    if (
        feasibility.ndim != 1
        or feasibility.size < 1
        or progress.shape != (feasibility.size, 3)
        or not np.all(np.isfinite(feasibility))
        or not np.all(np.isfinite(progress))
        or np.any(feasibility < 0.0)
        or np.any(feasibility > 1.0)
        or type(shortlist_count) is not int
        or not 1 <= shortlist_count <= feasibility.size
        or not np.all(np.isfinite(thresholds))
        or np.any(thresholds <= 0.0)
    ):
        raise ValueError("V686 acquisition candidate ranking inputs are invalid")
    feasibility_order = np.argsort(-feasibility, kind="stable")
    shortlist = feasibility_order[:shortlist_count]
    lateral = progress[:, 1]
    height = progress[:, 2]
    axis_consistent = (lateral >= -maximum_lateral_regression_m) & (height >= -maximum_height_regression_m)
    score = (
        progress[:, 0]
        + 0.50 * lateral
        + 0.25 * height
        + 8.0 * np.minimum(lateral, 0.0)
        + 4.0 * np.minimum(height, 0.0)
    )
    safe = shortlist[axis_consistent[shortlist]]
    unsafe = shortlist[~axis_consistent[shortlist]]
    safe_order = safe[np.argsort(-score[safe], kind="stable")]
    unsafe_order = unsafe[np.argsort(-score[unsafe], kind="stable")]
    return np.concatenate((safe_order, unsafe_order)).astype(np.int64), score


def adaptive_exact_preflight_schedule_v677(
    preference_order: np.ndarray,
    *,
    primary_count: int,
) -> tuple[tuple[int, bool], ...]:
    """Check the cheap shortlist first, then all remaining policy samples.

    The boolean marks emergency escalation.  The full order is still learned
    feasibility/Q order; V677 adds no planner or expert candidate.
    """

    order = np.asarray(preference_order)
    if (
        order.ndim != 1
        or order.size < 1
        or not np.issubdtype(order.dtype, np.integer)
        or not np.array_equal(np.sort(order), np.arange(order.size))
        or type(primary_count) is not int
        or not 1 <= primary_count <= order.size
    ):
        raise ValueError("V677 exact-preflight schedule is invalid")
    return tuple(
        (int(candidate_index), rank >= primary_count) for rank, candidate_index in enumerate(order.tolist())
    )


def complete_candidate_preference_order_v677(
    primary_order: np.ndarray,
    predicted_feasibility: np.ndarray,
    *,
    remaining_score: np.ndarray | None = None,
) -> np.ndarray:
    """Append every non-shortlisted actor sample by learned feasibility."""

    primary = np.asarray(primary_order)
    feasibility = np.asarray(predicted_feasibility, dtype=np.float64)
    secondary = feasibility if remaining_score is None else np.asarray(remaining_score, dtype=np.float64)
    if (
        primary.ndim != 1
        or primary.size < 1
        or not np.issubdtype(primary.dtype, np.integer)
        or feasibility.ndim != 1
        or primary.size > feasibility.size
        or not np.all(np.isfinite(feasibility))
        or np.any(feasibility < 0.0)
        or np.any(feasibility > 1.0)
        or secondary.shape != feasibility.shape
        or not np.all(np.isfinite(secondary))
        or np.unique(primary).size != primary.size
        or np.any(primary < 0)
        or np.any(primary >= feasibility.size)
    ):
        raise ValueError("V677 complete candidate preference is invalid")
    primary_members = set(primary.tolist())
    remaining = [
        int(candidate_index)
        for candidate_index in np.argsort(-secondary, kind="stable").tolist()
        if int(candidate_index) not in primary_members
    ]
    result = np.asarray([*primary.tolist(), *remaining], dtype=np.int64)
    if not np.array_equal(np.sort(result), np.arange(feasibility.size)):
        raise RuntimeError("V677 candidate preference lost an actor sample")
    return result


def _atomic_torch_save_v666(payload: dict[str, Any], path: Path) -> None:
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(destination)


def _load_augmented_state_v684(
    module: torch.nn.Module,
    source_state: dict[str, Any],
    *,
    source_format: str,
) -> tuple[bool, bool]:
    """Load legacy weights and audit each newly initialized policy branch."""

    if source_format in {
        _JOINT_GOAL_SAC_FORMAT_V683,
        _JOINT_GOAL_SAC_FORMAT_V684,
        _JOINT_GOAL_SAC_FORMAT_V685,
        _JOINT_GOAL_SAC_FORMAT_V686,
        JOINT_GOAL_SAC_FORMAT_V665,
    }:
        module.load_state_dict(source_state, strict=True)
        return False, False
    result = module.load_state_dict(source_state, strict=False)
    taskframe_missing = {
        name for name in module.state_dict() if name.endswith("taskframe_feature_projection.weight")
    }
    acquisition_missing = {name for name in module.state_dict() if name.startswith("acquisition_")}
    if source_format in _PRE_TASKFRAME_SAC_FORMATS_V682:
        expected_missing = taskframe_missing | acquisition_missing
    elif source_format == _JOINT_GOAL_SAC_FORMAT_V682:
        expected_missing = acquisition_missing
        taskframe_missing = set()
    else:
        raise ValueError("V684 legacy checkpoint format is not migratable")
    if set(result.missing_keys) != expected_missing or result.unexpected_keys:
        raise ValueError("V684 legacy checkpoint state migration is invalid")
    with torch.no_grad():
        for name, parameter in module.named_parameters():
            if name in taskframe_missing:
                parameter.zero_()
    return bool(taskframe_missing), bool(acquisition_missing)


def warm_start_joint_goal_bundle_v670(
    bundle: JointGoalSACBundleV665,
    checkpoint_path: Path,
    *,
    config: JointGoalSACConfigV665,
    load_value_functions: bool = False,
    load_optimizer_state: bool = False,
) -> dict[str, Any]:
    """Warm start a policy or exactly continue a current-format learner."""

    source = Path(checkpoint_path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"V670 warm-start checkpoint missing: {source}")
    payload = torch.load(source, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise ValueError("V670 warm-start checkpoint is not a mapping")
    source_format = payload.get("format")
    source_config = payload.get("configuration")
    if type(load_value_functions) is not bool or type(load_optimizer_state) is not bool:
        raise TypeError("V670 warm-start state flags must be boolean")
    if load_optimizer_state and not load_value_functions:
        raise ValueError("V678 optimizer continuation requires value-function continuation")
    if (
        source_format not in _WARM_START_COMPATIBLE_SAC_FORMATS_V670
        or source_format
        in {
            _JOINT_GOAL_SAC_FORMAT_V682,
            _JOINT_GOAL_SAC_FORMAT_V683,
            _JOINT_GOAL_SAC_FORMAT_V684,
            _JOINT_GOAL_SAC_FORMAT_V685,
            _JOINT_GOAL_SAC_FORMAT_V686,
            JOINT_GOAL_SAC_FORMAT_V665,
        }
        and payload.get("taskframe_feature_format") != JOINT_TASKFRAME_FEATURE_FORMAT_V682
        or source_format
        in {
            _JOINT_GOAL_SAC_FORMAT_V683,
            _JOINT_GOAL_SAC_FORMAT_V684,
            _JOINT_GOAL_SAC_FORMAT_V685,
            _JOINT_GOAL_SAC_FORMAT_V686,
            JOINT_GOAL_SAC_FORMAT_V665,
        }
        and payload.get("acquisition_context_format") != JOINT_ACQUISITION_CONTEXT_FORMAT_V683
        or source_format == JOINT_GOAL_SAC_FORMAT_V665
        and payload.get("acquisition_progress_format") != JOINT_ACQUISITION_PROGRESS_FORMAT_V686
        or payload.get("observation_dimension") != JOINT_GOAL_OBSERVATION_DIM_V665
        or payload.get("action_dimension") != ARM_JOINT_ACTION_DIM_V664
        or not isinstance(source_config, dict)
        or int(source_config.get("hidden_dim", -1)) != config.hidden_dim
        or not isinstance(payload.get("actor_state_dict"), dict)
        or not isinstance(payload.get("feasibility_state_dict"), dict)
        or source_format == JOINT_GOAL_SAC_FORMAT_V665
        and not isinstance(payload.get("acquisition_progress_state_dict"), dict)
        or load_value_functions
        and (
            source_format not in _VALUE_FUNCTION_COMPATIBLE_SAC_FORMATS_V680
            or not isinstance(payload.get("critic_state_dict"), dict)
            or not isinstance(payload.get("target_critic_state_dict"), dict)
        )
        or load_optimizer_state
        and (
            source_format != JOINT_GOAL_SAC_FORMAT_V665
            or not isinstance(payload.get("actor_optimizer_state_dict"), dict)
            or not isinstance(payload.get("critic_optimizer_state_dict"), dict)
            or not isinstance(payload.get("feasibility_optimizer_state_dict"), dict)
            or not isinstance(payload.get("acquisition_progress_optimizer_state_dict"), dict)
        )
    ):
        raise ValueError("V670 warm-start checkpoint contract is incompatible")
    actor_migrated, actor_acquisition_initialized = _load_augmented_state_v684(
        bundle.actor,
        payload["actor_state_dict"],
        source_format=source_format,
    )
    feasibility_migrated, feasibility_acquisition_initialized = _load_augmented_state_v684(
        bundle.feasibility,
        payload["feasibility_state_dict"],
        source_format=source_format,
    )
    value_functions_migrated = False
    value_acquisition_initialized = False
    if load_value_functions:
        critic_migrated, critic_acquisition_initialized = _load_augmented_state_v684(
            bundle.critic,
            payload["critic_state_dict"],
            source_format=source_format,
        )
        target_migrated, target_acquisition_initialized = _load_augmented_state_v684(
            bundle.target_critic,
            payload["target_critic_state_dict"],
            source_format=source_format,
        )
        value_functions_migrated = bool(critic_migrated or target_migrated)
        value_acquisition_initialized = bool(critic_acquisition_initialized or target_acquisition_initialized)
        bundle.update_index = int(payload.get("update_index", 0))
    acquisition_progress_model_initialized = source_format != JOINT_GOAL_SAC_FORMAT_V665
    if not acquisition_progress_model_initialized:
        bundle.acquisition_progress_model.load_state_dict(
            payload["acquisition_progress_state_dict"],
            strict=True,
        )
    if load_optimizer_state:
        bundle.actor_optimizer.load_state_dict(payload["actor_optimizer_state_dict"])
        bundle.critic_optimizer.load_state_dict(payload["critic_optimizer_state_dict"])
        bundle.feasibility_optimizer.load_state_dict(payload["feasibility_optimizer_state_dict"])
        bundle.acquisition_progress_optimizer.load_state_dict(
            payload["acquisition_progress_optimizer_state_dict"]
        )
    return {
        "format": "edgearm-v670-contact-acquisition-warm-start-v1",
        "source_checkpoint": str(source),
        "source_checkpoint_sha256": sha256_file_v1(source),
        "source_algorithm_format": source_format,
        "source_completed_episode_count": int(payload.get("completed_episode_count", 0)),
        "source_total_environment_steps": int(payload.get("total_environment_steps", 0)),
        "actor_loaded": True,
        "feasibility_loaded": True,
        "critic_loaded": load_value_functions,
        "target_critic_loaded": load_value_functions,
        "optimizer_state_loaded": load_optimizer_state,
        "learner_state_continuation": bool(load_value_functions and load_optimizer_state),
        "rng_state_loaded": False,
        "bitwise_exact_continuation": False,
        "obsolete_reward_value_functions_reinitialized": bool(not load_value_functions),
        "taskframe_feature_format": JOINT_TASKFRAME_FEATURE_FORMAT_V682,
        "taskframe_actor_projection_zero_migrated": actor_migrated,
        "taskframe_feasibility_projection_zero_migrated": (feasibility_migrated),
        "taskframe_value_projection_zero_migrated": (value_functions_migrated),
        "acquisition_context_format": JOINT_ACQUISITION_CONTEXT_FORMAT_V683,
        "acquisition_progress_format": JOINT_ACQUISITION_PROGRESS_FORMAT_V686,
        "acquisition_progress_model_initialized": (acquisition_progress_model_initialized),
        "contact_conditioned_acquisition_branch_initialized": (actor_acquisition_initialized),
        "unexpected_feasibility_acquisition_branch_initialized": (feasibility_acquisition_initialized),
        "unexpected_value_acquisition_branch_initialized": (value_acquisition_initialized),
        "source_update_index": int(payload.get("update_index", 0)),
        "production_admission": False,
    }


def _load_curriculum_history_chain_v674(
    run_dir: Path,
    *,
    visited: frozenset[Path],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Load an audited, recursively chained curriculum history.

    A continued run writes only its newly completed episodes.  Earlier V670
    loaded those local rows as though every run had to begin at episode one,
    which made a second continuation lose the curriculum evidence that
    preceded it.  V674 follows the immutable parent audit recorded in the run
    plan, verifies every hash and index boundary, and composes selection-only
    evidence without loading replay or treating any row as exportable data.
    """

    source = Path(run_dir).expanduser().resolve()
    if source in visited:
        raise ValueError("V674 curriculum history chain contains a cycle")
    next_visited = visited | {source}
    plan_path = source / "run_plan.json"
    episodes_path = source / "episodes.jsonl"
    if not plan_path.is_file() or not episodes_path.is_file():
        raise FileNotFoundError("V670 curriculum history artifacts are missing")
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if not isinstance(plan, dict):
        raise ValueError("V670 curriculum history plan is not a mapping")
    expected_hash = plan.get("run_plan_sha256")
    unhashed = dict(plan)
    unhashed.pop("run_plan_sha256", None)
    if expected_hash != canonical_sha256_v1(unhashed):
        raise ValueError("V670 curriculum history plan hash is invalid")
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(episodes_path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        record = json.loads(line)
        if (
            not isinstance(record, dict)
            or type(record.get("episode_index")) is not int
            or not isinstance(record.get("start_tier"), str)
            or bool(record.get("production_admission", True))
        ):
            raise ValueError(f"V670 curriculum history record {line_number} is invalid")
        records.append(record)
    if not records:
        raise ValueError("V670 curriculum history has no complete episodes")
    indices = [int(record["episode_index"]) for record in records]
    if indices != list(range(indices[0], indices[0] + len(indices))):
        raise ValueError("V674 local curriculum history indices are not contiguous")
    parent_records: list[dict[str, Any]] = []
    parent_chain: list[dict[str, Any]] = []
    if indices[0] != 1:
        parent_reference = plan.get("curriculum_history_audit_v670")
        if (
            not isinstance(parent_reference, dict)
            or parent_reference.get("enabled") is not True
            or not isinstance(parent_reference.get("source_run_dir"), str)
        ):
            raise ValueError("V674 continued curriculum history is missing its parent audit")
        parent_records, parent_audit = _load_curriculum_history_chain_v674(
            Path(parent_reference["source_run_dir"]),
            visited=next_visited,
        )
        expected_parent_count = len(parent_records)
        if (
            indices[0] != expected_parent_count + 1
            or parent_reference.get("source_run_plan_sha256") != parent_audit["source_run_plan_sha256"]
            or parent_reference.get("source_episodes_sha256") != parent_audit["source_episodes_sha256"]
            or int(parent_reference.get("source_episode_count", -1)) != expected_parent_count
            or int(parent_reference.get("first_episode_index", -1)) != 1
            or int(parent_reference.get("last_episode_index", -1)) != expected_parent_count
        ):
            raise ValueError("V674 curriculum history parent audit does not match its source")
        parent_chain = list(parent_audit["source_history_chain"])
    elif plan.get("curriculum_history_audit_v670", {}).get("enabled") is True:
        raise ValueError("V674 episode-one curriculum history unexpectedly declares a parent")
    combined = [*parent_records, *records]
    if [int(record["episode_index"]) for record in combined] != list(range(1, len(combined) + 1)):
        raise ValueError("V674 composed curriculum history is not contiguous from episode one")
    local_entry = {
        "source_run_dir": str(source),
        "source_run_plan_sha256": expected_hash,
        "source_episodes_sha256": sha256_file_v1(episodes_path),
        "first_local_episode_index": indices[0],
        "last_local_episode_index": indices[-1],
        "local_episode_count": len(records),
    }
    return combined, {
        "format": "edgearm-v674-chained-curriculum-history-audit-v1",
        "source_run_dir": str(source),
        "source_run_plan_sha256": expected_hash,
        "source_episodes_sha256": sha256_file_v1(episodes_path),
        "source_episode_count": len(combined),
        "local_source_episode_count": len(records),
        "first_episode_index": 1,
        "last_episode_index": len(combined),
        "source_history_chain": [*parent_chain, local_entry],
        "source_history_chain_depth": len(parent_chain) + 1,
        "selection_evidence_loaded": True,
        "replay_loaded": False,
        "critic_loaded_from_history": False,
        "trajectory_exported_from_history": False,
        "production_admission": False,
    }


def load_curriculum_history_v670(
    run_dir: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Load immutable selection evidence, including audited continuations."""

    return _load_curriculum_history_chain_v674(
        run_dir,
        visited=frozenset(),
    )


def _mark_failed_run_v666(output_dir: Path, error: Exception) -> None:
    """Make an unexpected CLI failure visible instead of leaving stale running state."""

    destination = Path(output_dir).expanduser().resolve()
    if not destination.exists():
        return
    state_path = destination / "run_state.json"
    existing: dict[str, Any] = {}
    if state_path.is_file():
        try:
            loaded = json.loads(state_path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                existing = loaded
        except (json.JSONDecodeError, OSError):
            existing = {}
    _atomic_json(
        state_path,
        {
            **existing,
            "status": "failed",
            "error_type": type(error).__name__,
            "error_message": str(error),
            "production_admission": False,
            "updated_at_utc": _utc_now(),
        },
    )


def _run_episode_v666(
    *,
    env: RealisticEdgeArmEnvV10,
    adapter: StockGripperTaskFrameAdapterV22,
    bundle: JointGoalSACBundleV665,
    replay: JointGoalReplayV665,
    sac_config: JointGoalSACConfigV665,
    training_config: FullTaskJointTrainingConfigV666,
    execution_mode: str,
    exact_home: bool,
    episode_index: int,
    maximum_steps: int,
    total_environment_steps: int,
    action_rng: np.random.Generator,
    update_rng: np.random.Generator,
    output_dir: Path,
) -> tuple[dict[str, Any], int, dict[str, Any] | None]:
    previous_applied = np.zeros(ARM_JOINT_ACTION_DIM_V664, dtype=np.float32)
    random_state = np.zeros(ARM_JOINT_ACTION_DIM_V664, dtype=np.float64)
    guarded = (
        GuardedJointDeltaActionV664(adapter, GuardedJointDeltaActionConfigV664())
        if execution_mode in _GUARDED_EXECUTION_MODES_V666
        else None
    )
    structurally_projected = bool(execution_mode in _STRUCTURALLY_PROJECTED_MODES_V666)
    planar_push_projected = bool(execution_mode in _PLANAR_PUSH_PROJECTED_MODES_V666)
    contact_adaptive_projected = bool(execution_mode in _CONTACT_ADAPTIVE_PROJECTED_MODES_V681)
    initial_object_distance = float(env.distance_to_target())
    initial_coverage = float(env.block_target_coverage())
    initial_tool_distance, _initial_alignment = _precontact_geometry_v666(env)
    minimum_tool_distance = initial_tool_distance
    maximum_alignment = 0.0
    maximum_coverage = initial_coverage
    maximum_hold_fraction = 0.0
    valid_contact_steps = 0
    invalid_contact_steps = 0
    effectful_steps = 0
    raw_tool_contact_steps = 0
    unattributed_block_motion_observed_steps = 0
    unattributed_block_motion_steps = 0
    safety_only_clearance_failure_steps = 0
    interventions = 0
    guarded_recovery_steps = 0
    acquisition_progress_steps = 0
    contact_acquisition_steps = 0
    contact_loss_event_steps = 0
    contact_absent_steps = 0
    contact_reacquisition_steps = 0
    contact_episode_replay_marked_steps = 0
    policy_candidate_selection_steps = 0
    deterministic_candidate_selected_steps = 0
    selected_predicted_feasibility_sum = 0.0
    maximum_candidate_predicted_feasibility_sum = 0.0
    minimum_selected_predicted_feasibility = float("inf")
    maximum_selected_predicted_feasibility = float("-inf")
    exact_guard_candidate_preflight_attempts = 0
    exact_guard_safe_candidate_steps = 0
    exact_guard_alternative_selected_steps = 0
    exact_guard_no_usable_candidate_steps = 0
    exact_guard_emergency_escalation_steps = 0
    exact_guard_emergency_preflight_attempts = 0
    exact_guard_emergency_recovered_steps = 0
    reward_sum = 0.0
    minimum_clearance = float("inf")
    latest_update: dict[str, Any] | None = None
    terminal_reason = "training_time_limit"
    strict_success = False
    failure_terminal = False
    executed_steps = 0
    block_path_length = 0.0
    qualified_block_path_length = 0.0
    admissible_contact_latched = False
    valid_contact_previous = False
    orientation_projection_steps = 0
    orientation_projection_l2_sum = 0.0
    orientation_only_projection_steps = 0
    planar_lock_projection_steps = 0
    initial_orientation = (
        planar_push_constraint_state_v668(env)
        if planar_push_projected
        else orientation_constraint_state_v667(env)
    )
    minimum_tool_face_horizontal_norm = initial_orientation.tool_face_horizontal_norm
    minimum_tool_face_push_alignment = initial_orientation.tool_face_push_alignment
    maximum_orientation_residual_norm = float(np.linalg.norm(initial_orientation.residual))
    minimum_tool_height_m = float(env.tool_xyz()[2])
    maximum_tool_height_m = minimum_tool_height_m
    maximum_absolute_tool_height_error_m: float | None = (
        abs(initial_orientation.tool_height_error_m) if planar_push_projected else None
    )

    for episode_step in range(maximum_steps):
        privileged_before = build_privileged_effect_state_v1(env)
        controller_before = _controller_target_v666(env, adapter, execution_mode)
        observation = joint_goal_observation_v665(
            privileged_before,
            np.asarray(env.target_xy, dtype=np.float32),
            controller_before,
            previous_applied,
            admissible_contact_latched,
        )
        object_distance_before = float(env.distance_to_target())
        precontact_before, alignment_before = _precontact_geometry_v666(env)
        coverage_before = float(env.block_target_coverage())
        hold_before = min(
            float(env._strict_success_streak) / float(env.config.strict_success_hold_steps),
            1.0,
        )
        # This is episode history, not an instantaneous raw MuJoCo contact.
        # It is also included in the policy observation above, preserving the
        # Markov contract for transport reward attribution.
        contact_before = admissible_contact_latched
        (
            action,
            random_state,
            candidate_audit,
            candidate_actions,
            candidate_preference_order,
        ) = _policy_action_v666(
            bundle,
            env,
            observation,
            warmup=total_environment_steps < training_config.warmup_transitions,
            random_state=random_state,
            rng=action_rng,
            config=training_config,
            execution_mode=execution_mode,
            admissible_contact_latched=contact_before,
        )
        block_before = env.block_xy().copy()
        guard_intervened = False
        guarded_recovery = False
        action_feasible = True
        applied_action = np.zeros(ARM_JOINT_ACTION_DIM_V664, np.float32)
        (
            control_action,
            orientation_projection_l2,
            orientation_projection_intervened,
        ) = _structurally_project_action_v672(
            env,
            action,
            execution_mode,
            admissible_contact_latched=contact_before,
        )
        verified_preflight = None
        step_terminal_without_physics = False
        try:
            if guarded is not None:
                learned_first_index = int(candidate_preference_order[0])
                first_preflight = None
                exact_attempts_this_step = 0
                exact_usable_candidate_found = False
                exact_selected_index = learned_first_index
                emergency_escalated_this_step = False
                emergency_attempts_this_step = 0
                emergency_recovered_this_step = False
                preflight_schedule = adaptive_exact_preflight_schedule_v677(
                    candidate_preference_order,
                    primary_count=min(
                        training_config.exact_guard_candidate_preflight_count,
                        int(candidate_preference_order.size),
                    ),
                )
                for candidate_index, emergency_candidate in preflight_schedule:
                    emergency_escalated_this_step |= emergency_candidate
                    emergency_attempts_this_step += int(emergency_candidate)
                    candidate_raw = candidate_actions[candidate_index]
                    (
                        candidate_control,
                        candidate_projection_l2,
                        candidate_projection_intervened,
                    ) = _structurally_project_action_v672(
                        env,
                        candidate_raw,
                        execution_mode,
                        admissible_contact_latched=contact_before,
                    )
                    preflight = guarded.preflight(candidate_control)
                    exact_attempts_this_step += 1
                    if candidate_index == learned_first_index:
                        first_preflight = preflight
                    selected_scale = preflight.guard_report.get("selected_scale")
                    usable = bool(
                        preflight.safe_candidate_found
                        and selected_scale is not None
                        and float(selected_scale) > 0.0
                        and not bool(preflight.guard_report.get("selected_is_baseline_hold", False))
                    )
                    if not usable:
                        continue
                    action = candidate_raw.copy()
                    control_action = candidate_control
                    orientation_projection_l2 = candidate_projection_l2
                    orientation_projection_intervened = candidate_projection_intervened
                    verified_preflight = preflight
                    exact_selected_index = candidate_index
                    exact_usable_candidate_found = True
                    emergency_recovered_this_step = emergency_candidate
                    break
                if not exact_usable_candidate_found:
                    verified_preflight = first_preflight
                exact_guard_candidate_preflight_attempts += exact_attempts_this_step
                exact_guard_safe_candidate_steps += int(exact_usable_candidate_found)
                exact_guard_alternative_selected_steps += int(
                    exact_usable_candidate_found and exact_selected_index != learned_first_index
                )
                exact_guard_no_usable_candidate_steps += int(not exact_usable_candidate_found)
                exact_guard_emergency_escalation_steps += int(emergency_escalated_this_step)
                exact_guard_emergency_preflight_attempts += emergency_attempts_this_step
                exact_guard_emergency_recovered_steps += int(emergency_recovered_this_step)
                candidate_audit.update(
                    {
                        "exact_guard_preflight_attempt_count": (exact_attempts_this_step),
                        "exact_guard_usable_candidate_found": (exact_usable_candidate_found),
                        "learned_first_candidate_index": (learned_first_index),
                        "selected_candidate_index": exact_selected_index,
                        "exact_guard_changed_candidate": bool(exact_selected_index != learned_first_index),
                        "exact_guard_is_final_execution_authority": True,
                        "exact_guard_emergency_escalated": (emergency_escalated_this_step),
                        "exact_guard_emergency_preflight_attempt_count": (emergency_attempts_this_step),
                        "exact_guard_emergency_recovered": (emergency_recovered_this_step),
                    }
                )
                candidate_audit["deterministic_candidate_selected"] = bool(exact_selected_index == 0)
                predicted_values = candidate_audit["candidate_predicted_feasibility"]
                q_values = candidate_audit["candidate_q_value"]
                candidate_audit["selected_predicted_feasibility"] = predicted_values[exact_selected_index]
                candidate_audit["selected_q_value"] = q_values[exact_selected_index]
                acquisition_scores = candidate_audit["candidate_acquisition_score"]
                acquisition_axes = candidate_audit["candidate_acquisition_axis_progress_m"]
                if acquisition_scores is not None and acquisition_axes is not None:
                    candidate_audit["selected_acquisition_score"] = acquisition_scores[exact_selected_index]
                    candidate_audit["selected_acquisition_axis_progress_m"] = acquisition_axes[
                        exact_selected_index
                    ]
                    candidate_audit["selected_execution_action"] = candidate_audit[
                        "candidate_execution_action"
                    ][exact_selected_index]
                translated = guarded.translate(
                    control_action,
                    verified_preflight=verified_preflight,
                )
                submitted = translated.submitted_joint_action
                applied_action = translated.applied_joint_action
                guard_intervened = translated.guard_intervened
                guarded_recovery = bool("v664_v22_guarded_recovery" in translated.guard_report)
                action_feasible = bool(
                    not translated.selected_is_hold
                    and translated.guard_selected_scale > 0.0
                    and np.linalg.norm(control_action.astype(np.float64) - applied_action.astype(np.float64))
                    <= 0.10
                )
            else:
                reported = np.asarray(
                    env._command_reference_reported_position(),
                    dtype=np.float64,
                )
                submitted = fast_joint_command_v666(
                    policy_action=control_action,
                    reported_joint_position_rad=reported,
                    zero_offset_rad=np.asarray(env._zero_offset),
                    fixed_gripper_joint_position_rad=(env.tool_gripper_joint_position_rad),
                    plant_max_joint_delta_rad=env.config.max_joint_delta,
                    policy_joint_target_step_rad=0.025,
                )
            selected_predicted_feasibility = candidate_audit["selected_predicted_feasibility"]
            if selected_predicted_feasibility is not None:
                selected_probability = float(selected_predicted_feasibility)
                policy_candidate_selection_steps += 1
                deterministic_candidate_selected_steps += int(
                    bool(candidate_audit["deterministic_candidate_selected"])
                )
                selected_predicted_feasibility_sum += selected_probability
                maximum_candidate_predicted_feasibility_sum += float(
                    candidate_audit["maximum_candidate_predicted_feasibility"]
                )
                minimum_selected_predicted_feasibility = min(
                    minimum_selected_predicted_feasibility,
                    selected_probability,
                )
                maximum_selected_predicted_feasibility = max(
                    maximum_selected_predicted_feasibility,
                    selected_probability,
                )
            _observation, _environment_reward, terminated, truncated, info = env.step(submitted)
            executed_steps += 1
            if guarded is None:
                transport = dict(info["sim2real_v2"])
                applied_action = applied_policy_action_from_plant_v666(
                    np.asarray(
                        transport["actually_applied_delayed_action"],
                        dtype=np.float32,
                    ),
                    plant_max_joint_delta_rad=env.config.max_joint_delta,
                    policy_joint_target_step_rad=0.025,
                )
                safety_reason = str(info.get("safety_reason", ""))
                guard_intervened = bool(safety_reason)
                action_feasible = bool(
                    not safety_reason
                    and np.linalg.norm(control_action.astype(np.float64) - applied_action.astype(np.float64))
                    <= 0.10
                )
            telemetry = transition_contact_telemetry_v22(
                info,
                block_before_xy_m=block_before,
                block_after_xy_m=env.block_xy(),
            )
            valid_contact = bool(telemetry["valid_push_side_contact_any"])
            invalid_contact = bool(telemetry["invalid_tool_block_contact_any"])
            raw_tool_contact = bool(telemetry["tool_block_contact_any"])
            block_displacement = float(telemetry["step_block_displacement_m"])
            clearance = float(telemetry["minimum_executed_safety_only_block_clearance_m"])
            minimum_clearance = min(minimum_clearance, clearance)
            hard_clearance = float(adapter.config.guard.hard_executed_safety_only_clearance_m)
            safety_only_clearance_failure = bool(clearance < hard_clearance)
            displacement_is_effectful = bool(block_displacement >= sac_config.effectful_block_displacement_m)
            (
                unattributed_block_motion_observed,
                unattributed_block_motion,
            ) = unattributed_motion_audit_v666(
                displacement_is_effectful=displacement_is_effectful,
                contact_before=contact_before,
                valid_contact=valid_contact,
                raw_tool_contact=raw_tool_contact,
                minimum_safety_only_clearance_m=clearance,
            )
            environment_strict_success = bool(info.get("success", False))
            success_without_admissible_contact = bool(
                environment_strict_success and not contact_before and not valid_contact
            )
            physical_contact_audit_failure = bool(
                unattributed_block_motion or success_without_admissible_contact
            )
            safety_violation = bool(
                info.get("safety_stop", False)
                or invalid_contact
                or safety_only_clearance_failure
                or physical_contact_audit_failure
            )
            strict_success = bool(environment_strict_success and not safety_violation)
            failure_terminal = bool(info.get("terminal_failure", False) or safety_violation)
            terminal = bool(terminated or truncated or strict_success or failure_terminal)
            terminal_reason = (
                "invalid_contact_failure"
                if invalid_contact
                else "safety_only_block_clearance_failure"
                if safety_only_clearance_failure
                else "unattributed_block_motion_failure"
                if unattributed_block_motion
                else "strict_success_without_admissible_contact_failure"
                if success_without_admissible_contact
                else "safety_failure"
                if safety_violation
                else str(info.get("terminal_reason", "nonterminal"))
            )
        except GuardedJointNoSafeActionV664:
            valid_contact = False
            invalid_contact = False
            raw_tool_contact = False
            block_displacement = 0.0
            displacement_is_effectful = False
            unattributed_block_motion_observed = False
            unattributed_block_motion = False
            safety_only_clearance_failure = False
            physical_contact_audit_failure = False
            safety_violation = True
            action_feasible = False
            guard_intervened = True
            guarded_recovery = False
            strict_success = False
            failure_terminal = True
            terminal = True
            terminated = True
            truncated = False
            terminal_reason = "guard_no_safe_action_before_physics"
            step_terminal_without_physics = True

        if episode_step + 1 >= maximum_steps and not terminal:
            terminal = True
            truncated = True
            terminal_reason = "training_horizon"

        privileged_after = build_privileged_effect_state_v1(env)
        controller_after = _controller_target_v666(env, adapter, execution_mode)
        contact_latched_after = bool(contact_before or valid_contact)
        next_observation = joint_goal_observation_v665(
            privileged_after,
            np.asarray(env.target_xy, dtype=np.float32),
            controller_after,
            applied_action,
            contact_latched_after,
        )
        object_distance_after = float(env.distance_to_target())
        precontact_after, alignment_after = _precontact_geometry_v666(env)
        coverage_after = float(env.block_target_coverage())
        hold_after = min(
            float(env._strict_success_streak) / float(env.config.strict_success_hold_steps),
            1.0,
        )
        projection_regime_after = projection_regime_v681(
            execution_mode,
            admissible_contact_latched=contact_latched_after,
        )
        orientation_after = (
            planar_push_constraint_state_v668(env)
            if projection_regime_after == "planar"
            else orientation_constraint_state_v667(env)
        )
        reward = joint_teacher_reward_v665(
            object_distance_before_m=object_distance_before,
            object_distance_after_m=object_distance_after,
            precontact_distance_before_m=precontact_before,
            precontact_distance_after_m=precontact_after,
            alignment_before=alignment_before,
            alignment_after=alignment_after,
            target_coverage_before=coverage_before,
            target_coverage_after=coverage_after,
            strict_hold_fraction_before=hold_before,
            strict_hold_fraction_after=hold_after,
            contact_before=contact_before,
            valid_contact_before=valid_contact_previous,
            valid_contact=valid_contact,
            block_step_displacement_m=block_displacement,
            # The orientation map defines the action-space semantics.  Only a
            # later plant/guard modification is a penalized projection.
            requested_action=control_action,
            applied_action=applied_action,
            guard_intervened=guard_intervened,
            invalid_contact=invalid_contact,
            safety_violation=safety_violation,
            strict_success=strict_success,
            failure_terminal=failure_terminal,
            config=sac_config,
        )
        effectful = bool(
            displacement_is_effectful
            and (contact_before or valid_contact)
            and not invalid_contact
            and not safety_violation
        )
        acquisition_progress = bool(
            reward.acquisition_progress_m >= sac_config.acquisition_progress_priority_threshold_m
        )
        replay.add(
            observation=observation,
            next_observation=next_observation,
            action=action,
            applied_action=applied_action,
            reward=reward.reward,
            terminal=terminal,
            action_feasible=action_feasible,
            strict_success=strict_success,
            valid_contact=valid_contact,
            effectful_block_motion=effectful,
            exact_home_start=exact_home,
            acquisition_progress=acquisition_progress,
            contact_acquired=reward.contact_acquired,
            contact_lost=reward.contact_lost,
            contact_absent=reward.contact_absent,
            contact_reacquired=reward.contact_reacquired,
            episode_index=episode_index,
            episode_step=episode_step,
        )
        if reward.contact_acquired:
            contact_episode_replay_marked_steps += replay.mark_contact_episode(episode_index)
        total_environment_steps += int(not step_terminal_without_physics)
        reward_sum += reward.reward
        previous_applied = applied_action.copy()
        admissible_contact_latched = contact_latched_after
        minimum_tool_distance = min(minimum_tool_distance, precontact_after)
        maximum_alignment = max(maximum_alignment, alignment_before, alignment_after)
        maximum_coverage = max(maximum_coverage, coverage_after)
        maximum_hold_fraction = max(maximum_hold_fraction, hold_after)
        valid_contact_steps += int(valid_contact)
        invalid_contact_steps += int(invalid_contact)
        effectful_steps += int(effectful)
        raw_tool_contact_steps += int(raw_tool_contact)
        unattributed_block_motion_observed_steps += int(unattributed_block_motion_observed)
        unattributed_block_motion_steps += int(unattributed_block_motion)
        safety_only_clearance_failure_steps += int(safety_only_clearance_failure)
        interventions += int(guard_intervened)
        guarded_recovery_steps += int(guarded_recovery)
        acquisition_progress_steps += int(acquisition_progress)
        contact_acquisition_steps += int(reward.contact_acquired)
        contact_loss_event_steps += int(reward.contact_lost)
        contact_absent_steps += int(reward.contact_absent)
        contact_reacquisition_steps += int(reward.contact_reacquired)
        valid_contact_previous = valid_contact
        orientation_projection_steps += int(orientation_projection_intervened)
        orientation_projection_l2_sum += orientation_projection_l2
        projection_regime_before = projection_regime_v681(
            execution_mode,
            admissible_contact_latched=contact_before,
        )
        orientation_only_projection_steps += int(projection_regime_before == "orientation")
        planar_lock_projection_steps += int(projection_regime_before == "planar")
        minimum_tool_face_horizontal_norm = min(
            minimum_tool_face_horizontal_norm,
            orientation_after.tool_face_horizontal_norm,
        )
        minimum_tool_face_push_alignment = min(
            minimum_tool_face_push_alignment,
            orientation_after.tool_face_push_alignment,
        )
        maximum_orientation_residual_norm = max(
            maximum_orientation_residual_norm,
            float(np.linalg.norm(orientation_after.residual)),
        )
        current_tool_height_m = float(env.tool_xyz()[2])
        minimum_tool_height_m = min(minimum_tool_height_m, current_tool_height_m)
        maximum_tool_height_m = max(maximum_tool_height_m, current_tool_height_m)
        if projection_regime_after == "planar":
            maximum_absolute_tool_height_error_m = max(
                float(maximum_absolute_tool_height_error_m or 0.0),
                abs(orientation_after.tool_height_error_m),
            )
        block_path_length += block_displacement
        qualified_block_path_length += block_displacement * int(
            bool(contact_before or valid_contact) and not invalid_contact and not safety_violation
        )

        if (
            replay.size >= sac_config.batch_size
            and total_environment_steps >= training_config.warmup_transitions
        ):
            for _ in range(training_config.updates_per_environment_step):
                metrics = update_joint_goal_sac_v665(
                    bundle,
                    replay,
                    config=sac_config,
                    rng=update_rng,
                )
                latest_update = asdict(metrics)
        if (episode_step + 1) % 30 == 0 or terminal:
            _atomic_json(
                output_dir / "run_state.json",
                {
                    "status": "running",
                    "episode_index": episode_index,
                    "episode_step": episode_step + 1,
                    "maximum_episode_steps": maximum_steps,
                    "execution_mode": execution_mode,
                    "exact_home_start": exact_home,
                    "current_object_target_distance_m": (object_distance_after),
                    "minimum_tool_precontact_distance_m": minimum_tool_distance,
                    "maximum_target_coverage": maximum_coverage,
                    "maximum_strict_hold_fraction": maximum_hold_fraction,
                    "valid_contact_steps": valid_contact_steps,
                    "orientation_projection_steps": (orientation_projection_steps),
                    "orientation_only_projection_steps": (orientation_only_projection_steps),
                    "planar_lock_projection_steps": (planar_lock_projection_steps),
                    "minimum_tool_face_horizontal_norm": (minimum_tool_face_horizontal_norm),
                    "minimum_tool_height_m": minimum_tool_height_m,
                    "maximum_tool_height_m": maximum_tool_height_m,
                    "planar_push_projection_enabled": planar_push_projected,
                    "contact_adaptive_projection_enabled": (contact_adaptive_projected),
                    "planar_lock_after_contact_enabled": (contact_adaptive_projected),
                    "unattributed_block_motion_steps": (unattributed_block_motion_steps),
                    "unattributed_block_motion_observed_steps": (unattributed_block_motion_observed_steps),
                    "guarded_recovery_steps": guarded_recovery_steps,
                    "acquisition_progress_steps": acquisition_progress_steps,
                    "contact_acquisition_steps": contact_acquisition_steps,
                    "contact_loss_event_steps": contact_loss_event_steps,
                    "contact_absent_steps": contact_absent_steps,
                    "contact_reacquisition_steps": (contact_reacquisition_steps),
                    "contact_episode_replay_marked_steps": (contact_episode_replay_marked_steps),
                    "policy_candidate_selection_steps": (policy_candidate_selection_steps),
                    "mean_selected_predicted_feasibility": (
                        selected_predicted_feasibility_sum / policy_candidate_selection_steps
                        if policy_candidate_selection_steps
                        else None
                    ),
                    "latest_policy_candidate_selection_v671": (candidate_audit),
                    "exact_guard_candidate_preflight_attempts": (exact_guard_candidate_preflight_attempts),
                    "exact_guard_safe_candidate_steps": (exact_guard_safe_candidate_steps),
                    "exact_guard_alternative_selected_steps": (exact_guard_alternative_selected_steps),
                    "exact_guard_no_usable_candidate_steps": (exact_guard_no_usable_candidate_steps),
                    "exact_guard_emergency_escalation_steps": (exact_guard_emergency_escalation_steps),
                    "exact_guard_emergency_preflight_attempts": (exact_guard_emergency_preflight_attempts),
                    "exact_guard_emergency_recovered_steps": (exact_guard_emergency_recovered_steps),
                    "safety_only_clearance_failure_steps": (safety_only_clearance_failure_steps),
                    "replay_size": replay.size,
                    "update_index": bundle.update_index,
                    "precontact_is_terminal": False,
                    "production_admission": False,
                    "updated_at_utc": _utc_now(),
                },
            )
        if terminal:
            break

    return (
        {
            "format": FULL_TASK_JOINT_SAC_TRAINING_FORMAT_V666,
            "episode_index": episode_index,
            "execution_mode": execution_mode,
            "exact_home_start": exact_home,
            "initial_target_coverage": initial_coverage,
            "initial_object_target_distance_m": initial_object_distance,
            "final_object_target_distance_m": float(env.distance_to_target()),
            "net_object_target_progress_m": (initial_object_distance - float(env.distance_to_target())),
            "initial_tool_precontact_distance_m": initial_tool_distance,
            "minimum_tool_precontact_distance_m": minimum_tool_distance,
            "maximum_precontact_alignment": maximum_alignment,
            "maximum_target_coverage": maximum_coverage,
            "maximum_strict_hold_fraction": maximum_hold_fraction,
            "valid_contact_steps": valid_contact_steps,
            "valid_contact_reached": valid_contact_steps > 0,
            "invalid_contact_steps": invalid_contact_steps,
            "effectful_block_motion_steps": effectful_steps,
            "effectful_block_motion": effectful_steps > 0,
            "effectful_motion_requires_admissible_contact_history": True,
            "raw_tool_contact_steps": raw_tool_contact_steps,
            "unattributed_block_motion_steps": (unattributed_block_motion_steps),
            "unattributed_block_motion_observed_steps": (unattributed_block_motion_observed_steps),
            "unattributed_motion_robot_coupling_clearance_m": (
                _UNATTRIBUTED_MOTION_COUPLING_CLEARANCE_M_V666
            ),
            "safety_only_clearance_failure_steps": (safety_only_clearance_failure_steps),
            "physical_contact_audit_failure": bool(
                unattributed_block_motion_steps > 0 or safety_only_clearance_failure_steps > 0
            ),
            "minimum_safety_only_block_clearance_m": minimum_clearance,
            "block_path_length_m": block_path_length,
            "qualified_block_path_length_m": qualified_block_path_length,
            "admissible_contact_latched": admissible_contact_latched,
            "guard_or_filter_intervention_steps": interventions,
            "guarded_recovery_steps": guarded_recovery_steps,
            "acquisition_progress_steps": acquisition_progress_steps,
            "contact_acquisition_steps": contact_acquisition_steps,
            "contact_loss_event_steps": contact_loss_event_steps,
            "contact_absent_steps": contact_absent_steps,
            "contact_reacquisition_steps": contact_reacquisition_steps,
            "contact_episode_replay_marked_steps": (contact_episode_replay_marked_steps),
            "policy_candidate_selection_steps": (policy_candidate_selection_steps),
            "deterministic_candidate_selected_steps": (deterministic_candidate_selected_steps),
            "mean_selected_predicted_feasibility": (
                selected_predicted_feasibility_sum / policy_candidate_selection_steps
                if policy_candidate_selection_steps
                else None
            ),
            "mean_maximum_candidate_predicted_feasibility": (
                maximum_candidate_predicted_feasibility_sum / policy_candidate_selection_steps
                if policy_candidate_selection_steps
                else None
            ),
            "minimum_selected_predicted_feasibility": (
                minimum_selected_predicted_feasibility if policy_candidate_selection_steps else None
            ),
            "maximum_selected_predicted_feasibility": (
                maximum_selected_predicted_feasibility if policy_candidate_selection_steps else None
            ),
            "exact_guard_candidate_preflight_attempts": (exact_guard_candidate_preflight_attempts),
            "exact_guard_safe_candidate_steps": (exact_guard_safe_candidate_steps),
            "exact_guard_alternative_selected_steps": (exact_guard_alternative_selected_steps),
            "exact_guard_no_usable_candidate_steps": (exact_guard_no_usable_candidate_steps),
            "exact_guard_emergency_escalation_steps": (exact_guard_emergency_escalation_steps),
            "exact_guard_emergency_preflight_attempts": (exact_guard_emergency_preflight_attempts),
            "exact_guard_emergency_recovered_steps": (exact_guard_emergency_recovered_steps),
            "orientation_projection_enabled": structurally_projected,
            "orientation_projection_format": (
                CONTACT_ADAPTIVE_PROJECTION_FORMAT_V681
                if contact_adaptive_projected
                else PLANAR_PUSH_PROJECTED_JOINT_ACTION_FORMAT_V668
                if planar_push_projected
                else ORIENTATION_PROJECTED_JOINT_ACTION_FORMAT_V667
                if structurally_projected
                else None
            ),
            "planar_push_projection_enabled": planar_push_projected,
            "contact_adaptive_projection_enabled": (contact_adaptive_projected),
            "planar_lock_after_contact_enabled": (contact_adaptive_projected),
            "orientation_projection_steps": orientation_projection_steps,
            "orientation_projection_l2_sum": orientation_projection_l2_sum,
            "orientation_only_projection_steps": (orientation_only_projection_steps),
            "planar_lock_projection_steps": planar_lock_projection_steps,
            "minimum_tool_face_horizontal_norm": (minimum_tool_face_horizontal_norm),
            "minimum_tool_face_push_alignment": (minimum_tool_face_push_alignment),
            "maximum_orientation_residual_norm": (maximum_orientation_residual_norm),
            "minimum_tool_height_m": minimum_tool_height_m,
            "maximum_tool_height_m": maximum_tool_height_m,
            "maximum_absolute_tool_height_error_m": (maximum_absolute_tool_height_error_m),
            "episode_reward_sum": reward_sum,
            "steps_executed": executed_steps,
            "terminal_reason": terminal_reason,
            "failure_terminal": failure_terminal,
            "strict_success": strict_success,
            "strict_three_second_hold_evaluated": True,
            "precontact_is_terminal": False,
            "curriculum_reset_is_action_or_path": False,
            "bulk_vla_data_use_allowed": False,
            "production_admission": False,
        },
        total_environment_steps,
        latest_update,
    )


def train_full_task_joint_sac_v666(
    *,
    source_training_run_dir: Path,
    output_dir: Path,
    episodes: int,
    maximum_steps: int,
    seed: int,
    execution_mode: str = "fast_v10_learning_only",
    warm_start_checkpoint: Path | None = None,
    warm_start_value_functions: bool = False,
    warm_start_optimizer_state: bool = False,
    warm_start_replay: Path | None = None,
    curriculum_history_run_dir: Path | None = None,
    sac_config: JointGoalSACConfigV665 | None = None,
    training_config: FullTaskJointTrainingConfigV666 | None = None,
) -> dict[str, Any]:
    selected_sac = sac_config or JointGoalSACConfigV665()
    selected_training = training_config or FullTaskJointTrainingConfigV666()
    selected_sac.validate()
    selected_training.validate()
    if (
        type(episodes) is not int
        or episodes < 1
        or type(maximum_steps) is not int
        or maximum_steps < 30
        or type(seed) is not int
        or seed < 0
        or type(warm_start_value_functions) is not bool
        or type(warm_start_optimizer_state) is not bool
        or execution_mode not in _EXECUTION_MODES_V666
    ):
        raise ValueError("V666 training request is invalid")
    if warm_start_optimizer_state and not warm_start_value_functions:
        raise ValueError("V678 optimizer continuation requires value-function continuation")
    if warm_start_replay is not None and (warm_start_checkpoint is None or not warm_start_value_functions):
        raise ValueError("V680 replay import requires its checkpoint value functions")
    source = Path(source_training_run_dir).expanduser().resolve()
    destination = Path(output_dir).expanduser().resolve()
    if destination.exists():
        raise FileExistsError(f"V666 output exists: {destination}")
    source_plan_path = source / "run_plan.json"
    source_plan = json.loads(source_plan_path.read_text(encoding="utf-8"))
    source_hash = source_plan.get("run_plan_sha256")
    unhashed = dict(source_plan)
    unhashed.pop("run_plan_sha256", None)
    if source_hash != canonical_sha256_v1(unhashed):
        raise ValueError("V666 source training run-plan hash is invalid")
    collection_plan_path = Path(source_plan["source_collection_plan"]).expanduser().resolve()
    _collection_plan, base_environment, base_action, scene_path = _load_collection_contract(
        collection_plan_path
    )
    causal_config = CausalSmoothRelayConfigV646(**source_plan["causal_motion_config_v646"])
    environment_config, action_config = _stage_runtime_v650(
        base_environment,
        base_action,
        stage_index=LONG_RANGE_STAGE_INDEX_V661,
        maximum_steps=maximum_steps,
        causal_config=causal_config,
    )

    policy_torch_seed = seed ^ 0x671AC710
    torch.manual_seed(policy_torch_seed)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    bundle = initialize_joint_goal_sac_v665(
        seed,
        device=device,
        config=selected_sac,
    )
    if warm_start_checkpoint is None:
        warm_start_audit: dict[str, Any] = {
            "format": "edgearm-v670-contact-acquisition-warm-start-v1",
            "enabled": False,
            "actor_loaded": False,
            "feasibility_loaded": False,
            "critic_loaded": False,
            "optimizer_state_loaded": False,
            "learner_state_continuation": False,
            "rng_state_loaded": False,
            "bitwise_exact_continuation": False,
            "production_admission": False,
        }
    else:
        warm_start_audit = {
            **warm_start_joint_goal_bundle_v670(
                bundle,
                warm_start_checkpoint,
                config=selected_sac,
                load_value_functions=warm_start_value_functions,
                load_optimizer_state=warm_start_optimizer_state,
            ),
            "enabled": True,
        }
    if curriculum_history_run_dir is None:
        curriculum_history: list[dict[str, Any]] = []
        curriculum_history_audit: dict[str, Any] = {
            "format": "edgearm-v670-curriculum-history-audit-v1",
            "enabled": False,
            "source_episode_count": 0,
            "selection_evidence_loaded": False,
            "replay_loaded": False,
            "production_admission": False,
        }
    else:
        curriculum_history, curriculum_history_audit = load_curriculum_history_v670(
            curriculum_history_run_dir
        )
        curriculum_history_audit = {
            **curriculum_history_audit,
            "enabled": True,
        }

    replay = JointGoalReplayV665(
        selected_sac.replay_capacity,
        acquisition_progress_priority_bonus=(selected_sac.acquisition_progress_priority_bonus),
        contact_acquisition_priority_bonus=(selected_sac.contact_acquisition_priority_bonus),
        contact_episode_priority_bonus=(selected_sac.contact_episode_priority_bonus),
        contact_loss_priority_bonus=(selected_sac.contact_loss_priority_bonus),
        contact_reacquisition_priority_bonus=(selected_sac.contact_reacquisition_priority_bonus),
    )
    if warm_start_replay is None:
        replay_continuation_audit: dict[str, Any] = {
            "format": "edgearm-v678-replay-continuation-audit-v1",
            "enabled": False,
            "restored_transition_count": 0,
            "schema_exact": False,
            "production_admission": False,
        }
    else:
        replay_source = Path(warm_start_replay).expanduser().resolve()
        checkpoint_source = Path(warm_start_checkpoint).expanduser().resolve()
        if replay_source.parent != checkpoint_source.parent.parent:
            raise ValueError("V678 checkpoint and replay do not belong to the same run")
        replay_continuation_audit = {
            **replay.restore(replay_source),
            "enabled": True,
            "source_replay_sha256": sha256_file_v1(replay_source),
            "source_checkpoint_sha256": warm_start_audit["source_checkpoint_sha256"],
            "checkpoint_and_replay_same_run": True,
            "optimizer_state_loaded": bool(warm_start_audit["optimizer_state_loaded"]),
            "actor_projection_distillation_objective_changed": bool(
                warm_start_audit["source_algorithm_format"]
                not in {
                    "edgearm-v665-full-task-joint-goal-sac-v6",
                    _JOINT_GOAL_SAC_FORMAT_V682,
                    _JOINT_GOAL_SAC_FORMAT_V683,
                    _JOINT_GOAL_SAC_FORMAT_V684,
                    _JOINT_GOAL_SAC_FORMAT_V685,
                    _JOINT_GOAL_SAC_FORMAT_V686,
                    JOINT_GOAL_SAC_FORMAT_V665,
                }
            ),
            "source_actor_optimizer_reinitialized_when_objective_changed": bool(
                warm_start_audit["source_algorithm_format"]
                not in {
                    "edgearm-v665-full-task-joint-goal-sac-v6",
                    _JOINT_GOAL_SAC_FORMAT_V682,
                    _JOINT_GOAL_SAC_FORMAT_V683,
                    _JOINT_GOAL_SAC_FORMAT_V684,
                    _JOINT_GOAL_SAC_FORMAT_V685,
                    _JOINT_GOAL_SAC_FORMAT_V686,
                    JOINT_GOAL_SAC_FORMAT_V665,
                }
                and not warm_start_audit["optimizer_state_loaded"]
            ),
            "taskframe_feature_architecture_changed": bool(
                warm_start_audit["source_algorithm_format"]
                not in {
                    _JOINT_GOAL_SAC_FORMAT_V682,
                    _JOINT_GOAL_SAC_FORMAT_V683,
                    _JOINT_GOAL_SAC_FORMAT_V684,
                    _JOINT_GOAL_SAC_FORMAT_V685,
                    _JOINT_GOAL_SAC_FORMAT_V686,
                    JOINT_GOAL_SAC_FORMAT_V665,
                }
            ),
            "contact_conditioned_acquisition_architecture_changed": bool(
                warm_start_audit["source_algorithm_format"]
                not in {
                    _JOINT_GOAL_SAC_FORMAT_V683,
                    _JOINT_GOAL_SAC_FORMAT_V684,
                    _JOINT_GOAL_SAC_FORMAT_V685,
                    _JOINT_GOAL_SAC_FORMAT_V686,
                    JOINT_GOAL_SAC_FORMAT_V665,
                }
            ),
            "acquisition_self_imitation_objective_changed": bool(
                warm_start_audit["source_algorithm_format"]
                not in {
                    _JOINT_GOAL_SAC_FORMAT_V685,
                    _JOINT_GOAL_SAC_FORMAT_V686,
                    JOINT_GOAL_SAC_FORMAT_V665,
                }
            ),
            "acquisition_progress_model_initialized": bool(
                warm_start_audit["acquisition_progress_model_initialized"]
            ),
            "source_optimizer_reinitialized_when_architecture_changed": bool(
                warm_start_audit["source_algorithm_format"]
                not in {
                    _JOINT_GOAL_SAC_FORMAT_V683,
                    _JOINT_GOAL_SAC_FORMAT_V684,
                    _JOINT_GOAL_SAC_FORMAT_V685,
                    _JOINT_GOAL_SAC_FORMAT_V686,
                    JOINT_GOAL_SAC_FORMAT_V665,
                }
                and not warm_start_audit["optimizer_state_loaded"]
            ),
            "source_optimizer_reinitialized_when_self_imitation_objective_changed": bool(
                warm_start_audit["source_algorithm_format"]
                not in {
                    _JOINT_GOAL_SAC_FORMAT_V685,
                    _JOINT_GOAL_SAC_FORMAT_V686,
                    JOINT_GOAL_SAC_FORMAT_V665,
                }
                and not warm_start_audit["optimizer_state_loaded"]
            ),
        }
    initial_replay_transition_count = replay.size
    if (
        selected_training.offline_acquisition_self_imitation_updates > 0
        or selected_training.offline_acquisition_progress_model_updates > 0
    ) and replay.size < selected_sac.batch_size:
        raise ValueError("V686 offline acquisition training requires a restored replay")

    destination.mkdir(parents=True)
    (destination / "checkpoints").mkdir()
    run_plan = {
        "format": FULL_TASK_JOINT_SAC_TRAINING_FORMAT_V666,
        "algorithm_format": JOINT_GOAL_SAC_FORMAT_V665,
        "created_at_utc": _utc_now(),
        "source_training_run_dir": str(source),
        "source_training_run_plan": str(source_plan_path),
        "source_training_run_plan_sha256": source_hash,
        "source_collection_plan": str(collection_plan_path),
        "source_collection_plan_sha256": sha256_file_v1(collection_plan_path),
        "scene_path": str(scene_path),
        "scene_sha256": sha256_file_v1(scene_path),
        "episodes": episodes,
        "maximum_steps": maximum_steps,
        "seed": seed,
        "policy_torch_seed": policy_torch_seed,
        "execution_mode": execution_mode,
        "sac_config_v665": asdict(selected_sac),
        "training_config_v666": asdict(selected_training),
        "warm_start_audit_v670": warm_start_audit,
        "replay_continuation_audit_v678": replay_continuation_audit,
        "curriculum_history_audit_v670": curriculum_history_audit,
        "environment_config": asdict(environment_config),
        "action_config": asdict(action_config),
        "curriculum_fractions": list(_CURRICULUM_FRACTIONS_V666),
        "exact_home_probe_interval_episodes": (selected_training.exact_home_probe_interval),
        "exact_home_probes_are_full_task_learning_not_export": True,
        "easiest_curriculum_tier": "precontact_v22",
        "precontact_v22_keeps_full_block_to_target_task": True,
        "precontact_v22_is_privileged_learning_only": True,
        "precontact_is_terminal": False,
        "strict_three_second_success_is_only_positive_terminal": True,
        "curriculum_changes_start_state_only": True,
        "reverse_curriculum_uses_fine_acquisition_frontier": True,
        "reverse_curriculum_start_states_are_dynamically_reachable": True,
        "reverse_curriculum_random_walk_is_reset_generation_only": True,
        "reverse_curriculum_random_walk_added_to_replay": False,
        "reverse_curriculum_research_basis": [
            "https://doi.org/10.48550/arxiv.1707.05300",
            "https://doi.org/10.1109/ICRA.2019.8794206",
        ],
        "approach_tier_mastery_requires_clean_contact_not_task_success": True,
        "policy_action_candidate_source": "same_sac_actor_only",
        "policy_candidate_feasibility_shortlist_enabled": bool(selected_training.policy_candidate_count > 1),
        "policy_candidate_task_value_ranking_inside_shortlist_after_contact": True,
        "policy_candidate_task_value_ranking_inside_shortlist_before_contact": False,
        "policy_candidate_axis_progress_ranking_inside_shortlist_before_contact": True,
        "candidate_selection_is_not_expert_or_planner": True,
        "candidate_selection_final_authority_remains_v4_guard": True,
        "exact_v4_preflight_applied_to_ranked_candidates": bool(
            execution_mode in _GUARDED_EXECUTION_MODES_V666
        ),
        "exact_v4_preflight_reuses_verified_forecast_for_execution": True,
        "exact_v4_preflight_adaptive_emergency_escalation_enabled": True,
        "exact_v4_preflight_emergency_candidates_are_actor_samples": True,
        "candidate_selection_research_basis": [
            "https://doi.org/10.48550/arxiv.2506.11033",
            "https://doi.org/10.1109/OJCSYS.2023.3256305",
        ],
        "fast_v10_transitions_are_learning_only": bool(execution_mode in _FAST_EXECUTION_MODES_V666),
        "orientation_projection_enabled": bool(execution_mode in _STRUCTURALLY_PROJECTED_MODES_V666),
        "orientation_projection_format": (
            CONTACT_ADAPTIVE_PROJECTION_FORMAT_V681
            if execution_mode in _CONTACT_ADAPTIVE_PROJECTED_MODES_V681
            else PLANAR_PUSH_PROJECTED_JOINT_ACTION_FORMAT_V668
            if execution_mode in _PLANAR_PUSH_PROJECTED_MODES_V666
            else ORIENTATION_PROJECTED_JOINT_ACTION_FORMAT_V667
            if execution_mode in _ORIENTATION_PROJECTED_MODES_V666
            else None
        ),
        "orientation_projection_constraints": (
            [
                "before_admissible_contact_tool_push_face_vertical",
                ("before_admissible_contact_tool_push_face_normal_parallel_to_block_target_direction"),
                "before_admissible_contact_xyz_motion_remains_policy_controlled",
                "after_admissible_contact_tool_site_height_equals_0.050_m",
            ]
            if execution_mode in _CONTACT_ADAPTIVE_PROJECTED_MODES_V681
            else [
                "tool_push_face_vertical",
                "tool_push_face_normal_parallel_to_block_target_direction",
                "tool_site_height_equals_0.050_m",
            ]
            if execution_mode in _PLANAR_PUSH_PROJECTED_MODES_V666
            else [
                "tool_push_face_vertical",
                "tool_push_face_normal_parallel_to_block_target_direction",
            ]
            if execution_mode in _ORIENTATION_PROJECTED_MODES_V666
            else []
        ),
        "orientation_projection_retained_local_nullspace_dimension": (
            None
            if execution_mode in _CONTACT_ADAPTIVE_PROJECTED_MODES_V681
            else 2
            if execution_mode in _PLANAR_PUSH_PROJECTED_MODES_V666
            else 3
            if execution_mode in _ORIENTATION_PROJECTED_MODES_V666
            else None
        ),
        "orientation_projection_retained_local_nullspace_by_regime": (
            {"before_admissible_contact": 3, "after_admissible_contact": 2}
            if execution_mode in _CONTACT_ADAPTIVE_PROJECTED_MODES_V681
            else None
        ),
        "planar_push_projection_enabled": bool(execution_mode in _PLANAR_PUSH_PROJECTED_MODES_V666),
        "contact_adaptive_projection_enabled": bool(execution_mode in _CONTACT_ADAPTIVE_PROJECTED_MODES_V681),
        "planar_lock_after_contact_enabled": bool(execution_mode in _CONTACT_ADAPTIVE_PROJECTED_MODES_V681),
        "planar_push_target_tool_height_m": (
            0.050
            if execution_mode in (_PLANAR_PUSH_PROJECTED_MODES_V666 | _CONTACT_ADAPTIVE_PROJECTED_MODES_V681)
            else None
        ),
        "contact_adaptive_projection_is_not_route_ik_or_expert": True,
        "orientation_projection_is_structural_action_semantics": True,
        "orientation_projection_is_not_a_safety_intervention": True,
        "orientation_projection_is_not_a_feasibility_failure": True,
        "final_generation_requires_guarded_v4_contact_adaptive_projection": (True),
        "object_reward_requires_latched_admissible_contact": True,
        "first_admissible_contact_is_rewarded": True,
        "first_admissible_contact_is_never_stalled_contact": True,
        "contact_episode_backward_replay_priority_enabled": True,
        "pre_target_contact_loss_penalty_enabled": True,
        "pre_target_contact_absence_penalty_enabled": True,
        "contact_recovery_precontact_objective_enabled": True,
        "contact_reacquisition_bonus_enabled": True,
        "contact_loss_target_coverage_exemption": (selected_sac.contact_loss_coverage_exemption),
        "contact_loss_priority_bonus": (selected_sac.contact_loss_priority_bonus),
        "contact_reacquisition_priority_bonus": (selected_sac.contact_reacquisition_priority_bonus),
        "event_replay_priority_survives_td_updates": True,
        "shield_executed_action_self_distillation_enabled": True,
        "shield_executed_action_self_distillation_coefficient": (
            selected_sac.actor_projection_distillation_coefficient
        ),
        "shield_executed_action_self_distillation_minimum_gap": (
            selected_sac.actor_projection_distillation_minimum_gap
        ),
        "shield_executed_action_self_distillation_is_not_expert_data": True,
        "off_policy_replay_survives_audited_continuations": bool(replay_continuation_audit["enabled"]),
        "failed_rl_transitions_are_replay_only_not_vla_demonstrations": True,
        "admissible_contact_latch_is_in_policy_observation": True,
        "taskframe_precontact_error_feature_enabled": True,
        "taskframe_precontact_error_feature_format": (JOINT_TASKFRAME_FEATURE_FORMAT_V682),
        "taskframe_precontact_error_feature_components": [
            "tool_to_precontact_forward_error",
            "tool_to_precontact_lateral_error",
            "tool_to_precontact_height_error",
        ],
        "taskframe_feature_uses_current_markov_state_only": True,
        "taskframe_feature_is_not_route_action_waypoint_or_expert": True,
        "taskframe_feature_projection_zero_initialized_on_v6_migration": bool(
            warm_start_audit.get(
                "taskframe_actor_projection_zero_migrated",
                False,
            )
        ),
        "contact_conditioned_acquisition_policy_enabled": True,
        "contact_conditioned_acquisition_context_format": (JOINT_ACQUISITION_CONTEXT_FORMAT_V683),
        "contact_conditioned_acquisition_context_components": [
            "current_taskframe_precontact_error",
            "current_physical_arm_joint_state",
            "current_arm_controller_target",
            "previous_applied_arm_action",
        ],
        "contact_conditioned_policy_gate": "current_admissible_contact_latch",
        "contact_conditioned_policy_uses_expert_phase_label": False,
        "contact_conditioned_policy_uses_expert_path_or_future_action": False,
        "contact_conditioned_acquisition_branch_initialized_on_legacy_migration": bool(
            warm_start_audit.get(
                "contact_conditioned_acquisition_branch_initialized",
                False,
            )
        ),
        "acquisition_self_imitation_enabled": bool(
            selected_sac.actor_acquisition_self_imitation_coefficient > 0.0
        ),
        "acquisition_self_imitation_coefficient": (selected_sac.actor_acquisition_self_imitation_coefficient),
        "acquisition_self_imitation_admission_requirements": [
            "exact_home_start",
            "before_admissible_contact",
            "exact_guard_action_feasible",
            "measured_3d_precontact_progress",
            "forward_error_reduced",
            "lateral_error_not_regressed",
            "height_error_not_materially_regressed",
        ],
        "acquisition_self_imitation_axis_consistency_contract": {
            "minimum_forward_improvement_m": (
                selected_sac.actor_acquisition_self_imitation_minimum_forward_improvement_m
            ),
            "maximum_lateral_regression_m": (
                selected_sac.actor_acquisition_self_imitation_maximum_lateral_regression_m
            ),
            "maximum_height_regression_m": (
                selected_sac.actor_acquisition_self_imitation_maximum_height_regression_m
            ),
        },
        "acquisition_self_imitation_target": "physically_applied_safe_action",
        "acquisition_self_imitation_uses_agent_experience_not_expert": True,
        "acquisition_self_imitation_research_basis": [
            "https://doi.org/10.48550/arxiv.1806.05635",
            "https://doi.org/10.48550/arxiv.1910.00177",
            "https://doi.org/10.48550/arxiv.2006.09359",
        ],
        "offline_acquisition_self_imitation_updates": (
            selected_training.offline_acquisition_self_imitation_updates
        ),
        "acquisition_progress_model_format": (JOINT_ACQUISITION_PROGRESS_FORMAT_V686),
        "acquisition_progress_model_training_scope": [
            "exact_home_start",
            "before_admissible_contact",
            "physically_applied_action_after_projection_and_guard",
            "guard_scaled_and_hold_outcomes_retained",
        ],
        "acquisition_progress_model_target": ("one_step_absolute_taskframe_error_reduction_m_by_axis"),
        "acquisition_progress_model_is_twin_and_pessimistic": True,
        "acquisition_candidate_axis_regression_envelope_m": {
            "lateral": 2.5e-4,
            "height": 5.0e-4,
        },
        "acquisition_candidate_uses_full_task_q_before_contact": False,
        "acquisition_candidate_scored_after_structural_projection": True,
        "acquisition_candidate_prediction_matches_executed_action_semantics": True,
        "offline_acquisition_progress_model_updates": (
            selected_training.offline_acquisition_progress_model_updates
        ),
        "passive_unattributed_object_drift_is_recorded_not_rewarded": True,
        "unattributed_motion_requires_robot_coupling_within_m": (
            _UNATTRIBUTED_MOTION_COUPLING_CLEARANCE_M_V666
        ),
        "robot_coupled_unattributed_block_motion_fails_closed": True,
        "safety_only_clearance_is_terminally_enforced": True,
        "guarded_v22_interior_recovery_enabled": bool(execution_mode in _GUARDED_EXECUTION_MODES_V666),
        "guarded_recovery_uses_same_v4_safety_proof": True,
        "guarded_recovery_is_not_an_expert_action": True,
        "full_task_start_contract_format": (FULL_TASK_START_CONTRACT_FORMAT_V669),
        "full_task_initial_distance_band_m": [
            FULL_TASK_MINIMUM_INITIAL_DISTANCE_M_V669,
            FULL_TASK_MAXIMUM_INITIAL_DISTANCE_M_V669,
        ],
        "full_task_maximum_initial_coverage": (FULL_TASK_MAXIMUM_INITIAL_COVERAGE_V669),
        "near_target_or_privileged_curriculum_is_export_forbidden": True,
        "expert_actions": 0,
        "expert_paths": 0,
        "behavior_cloning_steps": 0,
        "act_training_started": False,
        "wrist_multimodal_export_started": False,
        "source_type": SOURCE_TYPE,
        "production_admission": False,
    }
    run_plan["run_plan_sha256"] = canonical_sha256_v1(run_plan)
    _atomic_json(destination / "run_plan.json", run_plan)

    env = RealisticEdgeArmEnvV10(
        environment_config,
        seed=seed,
        model_scene_path=scene_path,
    )
    home_adapter = StockGripperHomeTaskFrameAdapterV597(env, action_config)
    approach_adapter = StockGripperDynamicReverseTaskFrameAdapterV673(env, action_config)
    precontact_adapter = StockGripperTaskFrameAdapterV22(env, action_config)
    renderer = _NoImageResetRendererV661()
    action_rng = np.random.default_rng(seed ^ 0x666A11)
    update_rng = np.random.default_rng(seed ^ 0x665AC)
    offline_acquisition_rng = np.random.default_rng(seed ^ 0x684AC)
    offline_acquisition_progress_rng = np.random.default_rng(seed ^ 0x686AC)
    records: list[dict[str, Any]] = []
    total_environment_steps = 0
    latest_update: dict[str, Any] | None = None
    offline_acquisition_latest_update: dict[str, Any] | None = None
    offline_acquisition_qualified_batch_count = 0
    offline_acquisition_qualified_sample_count = 0
    offline_acquisition_progress_latest_update: dict[str, Any] | None = None
    offline_acquisition_progress_qualified_batch_count = 0
    offline_acquisition_progress_qualified_sample_count = 0
    for offline_update_index in range(
        1,
        selected_training.offline_acquisition_progress_model_updates + 1,
    ):
        progress_metrics = pretrain_joint_acquisition_progress_v686(
            bundle,
            replay,
            config=selected_sac,
            rng=offline_acquisition_progress_rng,
        )
        offline_acquisition_progress_latest_update = asdict(progress_metrics)
        offline_acquisition_progress_qualified_batch_count += int(progress_metrics.qualified_sample_count > 0)
        offline_acquisition_progress_qualified_sample_count += int(progress_metrics.qualified_sample_count)
        if (
            offline_update_index == 1
            or offline_update_index % 100 == 0
            or offline_update_index == selected_training.offline_acquisition_progress_model_updates
        ):
            _atomic_json(
                destination / "run_state.json",
                {
                    "status": "offline_acquisition_progress_model",
                    "offline_update_index": offline_update_index,
                    "offline_update_count": (selected_training.offline_acquisition_progress_model_updates),
                    "qualified_batch_count": (offline_acquisition_progress_qualified_batch_count),
                    "qualified_sample_count": (offline_acquisition_progress_qualified_sample_count),
                    "latest_update": offline_acquisition_progress_latest_update,
                    "production_admission": False,
                    "updated_at_utc": _utc_now(),
                },
            )
    for offline_update_index in range(
        1,
        selected_training.offline_acquisition_self_imitation_updates + 1,
    ):
        offline_metrics = pretrain_joint_goal_acquisition_v684(
            bundle,
            replay,
            config=selected_sac,
            rng=offline_acquisition_rng,
        )
        offline_acquisition_latest_update = asdict(offline_metrics)
        offline_acquisition_qualified_batch_count += int(offline_metrics.qualified_sample_count > 0)
        offline_acquisition_qualified_sample_count += int(offline_metrics.qualified_sample_count)
        if (
            offline_update_index == 1
            or offline_update_index % 100 == 0
            or offline_update_index == selected_training.offline_acquisition_self_imitation_updates
        ):
            _atomic_json(
                destination / "run_state.json",
                {
                    "status": "offline_acquisition_self_imitation",
                    "offline_update_index": offline_update_index,
                    "offline_update_count": (selected_training.offline_acquisition_self_imitation_updates),
                    "qualified_batch_count": (offline_acquisition_qualified_batch_count),
                    "qualified_sample_count": (offline_acquisition_qualified_sample_count),
                    "latest_update": offline_acquisition_latest_update,
                    "production_admission": False,
                    "updated_at_utc": _utc_now(),
                },
            )
    if (
        selected_training.offline_acquisition_self_imitation_updates > 0
        or selected_training.offline_acquisition_progress_model_updates > 0
    ):
        _atomic_torch_save_v666(
            {
                **joint_goal_checkpoint_payload_v665(bundle, selected_sac),
                "training_format": FULL_TASK_JOINT_SAC_TRAINING_FORMAT_V666,
                "run_plan_sha256": run_plan["run_plan_sha256"],
                "completed_episode_count": len(curriculum_history),
                "total_environment_steps": 0,
                "offline_acquisition_self_imitation_updates": (
                    selected_training.offline_acquisition_self_imitation_updates
                ),
                "offline_acquisition_progress_model_updates": (
                    selected_training.offline_acquisition_progress_model_updates
                ),
            },
            destination / "checkpoints" / "offline_acquisition_pretrained.pt",
        )
    episode_index_offset = len(curriculum_history)
    for local_episode_index in range(1, episodes + 1):
        episode_index = episode_index_offset + local_episode_index
        fraction, selection = select_start_fraction_v666(
            episode_index=episode_index,
            records=[*curriculum_history, *records],
            config=selected_training,
        )
        requested_seed = seed + episode_index
        if fraction == 0.0:
            adapter: StockGripperTaskFrameAdapterV22 = home_adapter
            reset_audit = reset_stock_home_certified_long_range_episode_v649(
                env,
                renderer,
                adapter,
                requested_seed=requested_seed,
                obstacle=False,
                stress=False,
            )
            exact_home = True
        elif fraction == 1.0:
            adapter = precontact_adapter
            reset_audit = reset_stock_taskframe_episode_v22(
                env,
                renderer,
                precontact_adapter,
                requested_seed=requested_seed,
                obstacle=False,
                stress=False,
            )
            exact_home = False
        else:
            adapter = approach_adapter
            reset_audit = reset_stock_dynamic_reverse_episode_v673(
                env,
                renderer,
                approach_adapter,
                requested_seed=requested_seed,
                obstacle=False,
                stress=False,
                home_to_precontact_fraction=fraction,
            )
            exact_home = False
        record, total_environment_steps, episode_update = _run_episode_v666(
            env=env,
            adapter=adapter,
            bundle=bundle,
            replay=replay,
            sac_config=selected_sac,
            training_config=selected_training,
            execution_mode=execution_mode,
            exact_home=exact_home,
            episode_index=episode_index,
            maximum_steps=maximum_steps,
            total_environment_steps=total_environment_steps,
            action_rng=action_rng,
            update_rng=update_rng,
            output_dir=destination,
        )
        record.update(
            {
                "start_tier": _tier_code_v666(fraction),
                "home_to_precontact_fraction": fraction,
                "curriculum_reset_kind": (
                    "exact_home"
                    if exact_home
                    else "exact_task_aligned_precontact_v22"
                    if fraction == 1.0
                    else "v4_guarded_dynamic_reverse_random_walk_v673"
                ),
                "curriculum_selection_v666": selection,
                "reset_selected_seed": int(reset_audit["selected_seed"]),
                "task_aligned_privileged_reset": bool(not exact_home),
                "start_state_final_data_eligible": exact_home,
                "trajectory_data_role": (
                    "strict_home_candidate_not_yet_admitted"
                    if exact_home
                    else "learning_only_privileged_curriculum"
                ),
            }
        )
        start_audit = audit_full_task_start_v669(record)
        record["full_task_start_geometry_audit_v669"] = start_audit
        record["full_task_start_geometry_eligible"] = bool(start_audit["eligible"])
        records.append(record)
        _append_jsonl(destination / "episodes.jsonl", record)
        if episode_update is not None:
            latest_update = episode_update
        checkpoint_path = destination / "checkpoints" / "latest.pt"
        _atomic_torch_save_v666(
            {
                **joint_goal_checkpoint_payload_v665(bundle, selected_sac),
                "training_format": FULL_TASK_JOINT_SAC_TRAINING_FORMAT_V666,
                "run_plan_sha256": run_plan["run_plan_sha256"],
                "completed_episode_count": episode_index,
                "total_environment_steps": total_environment_steps,
            },
            checkpoint_path,
        )
        if (
            episode_index % selected_training.replay_save_interval_episodes == 0
            or local_episode_index == episodes
        ):
            replay.save(destination / "replay_latest.npz")

    home_records = [row for row in records if row["exact_home_start"]]
    curriculum_records = [row for row in records if not row["exact_home_start"]]
    home_successes = sum(int(row["strict_success"]) for row in home_records)
    curriculum_successes = sum(int(row["strict_success"]) for row in curriculum_records)
    contact_count = sum(int(row["valid_contact_reached"]) for row in records)
    audit_failure_count = sum(int(row["physical_contact_audit_failure"]) for row in records)
    strict_home_wrist_candidates = [
        row
        for row in home_records
        if execution_mode == "guarded_v4_contact_adaptive_projected"
        and row["strict_success"] is True
        and row["full_task_start_geometry_eligible"] is True
        and float(row["maximum_target_coverage"]) >= 0.95
        and float(row["maximum_strict_hold_fraction"]) >= 1.0 - 1.0e-6
        and int(row["invalid_contact_steps"]) == 0
        and row["physical_contact_audit_failure"] is False
    ]
    summary = {
        "format": FULL_TASK_JOINT_SAC_TRAINING_FORMAT_V666,
        "status": "complete",
        "created_at_utc": _utc_now(),
        "run_plan_sha256": run_plan["run_plan_sha256"],
        "algorithm_format": JOINT_GOAL_SAC_FORMAT_V665,
        "device": device,
        "execution_mode": execution_mode,
        "contact_adaptive_projection_enabled": bool(execution_mode in _CONTACT_ADAPTIVE_PROJECTED_MODES_V681),
        "episode_count": len(records),
        "new_episode_count": len(records),
        "curriculum_history_episode_count": len(curriculum_history),
        "combined_curriculum_episode_count": (len(curriculum_history) + len(records)),
        "exact_home_episode_count": len(home_records),
        "curriculum_episode_count": len(curriculum_records),
        "valid_contact_episode_count": contact_count,
        "physical_contact_audit_failure_episode_count": audit_failure_count,
        "unattributed_block_motion_episode_count": sum(
            int(row["unattributed_block_motion_steps"] > 0) for row in records
        ),
        "passive_unattributed_motion_observed_episode_count": sum(
            int(row["unattributed_block_motion_observed_steps"] > 0) for row in records
        ),
        "guarded_recovery_episode_count": sum(int(row["guarded_recovery_steps"] > 0) for row in records),
        "guarded_recovery_step_count": sum(int(row["guarded_recovery_steps"]) for row in records),
        "safety_only_clearance_failure_episode_count": sum(
            int(row["safety_only_clearance_failure_steps"] > 0) for row in records
        ),
        "strict_home_success_count": home_successes,
        "curriculum_strict_success_count": curriculum_successes,
        "replay_transition_count": replay.size,
        "initial_replay_transition_count": initial_replay_transition_count,
        "new_replay_transition_count": min(
            total_environment_steps,
            max(replay.size - initial_replay_transition_count, 0),
        ),
        "total_environment_steps": total_environment_steps,
        "update_index": bundle.update_index,
        "latest_update_metrics": latest_update,
        "offline_acquisition_self_imitation_update_count": (
            selected_training.offline_acquisition_self_imitation_updates
        ),
        "offline_acquisition_self_imitation_qualified_batch_count": (
            offline_acquisition_qualified_batch_count
        ),
        "offline_acquisition_self_imitation_qualified_sample_count": (
            offline_acquisition_qualified_sample_count
        ),
        "offline_acquisition_self_imitation_latest_metrics": (offline_acquisition_latest_update),
        "offline_acquisition_progress_model_update_count": (
            selected_training.offline_acquisition_progress_model_updates
        ),
        "offline_acquisition_progress_model_qualified_batch_count": (
            offline_acquisition_progress_qualified_batch_count
        ),
        "offline_acquisition_progress_model_qualified_sample_count": (
            offline_acquisition_progress_qualified_sample_count
        ),
        "offline_acquisition_progress_model_latest_metrics": (offline_acquisition_progress_latest_update),
        "warm_start_audit_v670": warm_start_audit,
        "replay_continuation_audit_v678": replay_continuation_audit,
        "curriculum_history_audit_v670": curriculum_history_audit,
        "acquisition_progress_step_count": sum(int(row["acquisition_progress_steps"]) for row in records),
        "contact_acquisition_step_count": sum(int(row["contact_acquisition_steps"]) for row in records),
        "contact_loss_event_step_count": sum(int(row["contact_loss_event_steps"]) for row in records),
        "contact_absent_step_count": sum(int(row["contact_absent_steps"]) for row in records),
        "contact_reacquisition_step_count": sum(int(row["contact_reacquisition_steps"]) for row in records),
        "contact_episode_replay_marked_step_count": sum(
            int(row["contact_episode_replay_marked_steps"]) for row in records
        ),
        "policy_candidate_selection_step_count": sum(
            int(row["policy_candidate_selection_steps"]) for row in records
        ),
        "orientation_only_projection_step_count": sum(
            int(row["orientation_only_projection_steps"]) for row in records
        ),
        "planar_lock_projection_step_count": sum(int(row["planar_lock_projection_steps"]) for row in records),
        "mean_selected_predicted_feasibility": (
            sum(
                float(row["mean_selected_predicted_feasibility"])
                * int(row["policy_candidate_selection_steps"])
                for row in records
                if row["mean_selected_predicted_feasibility"] is not None
            )
            / max(
                sum(int(row["policy_candidate_selection_steps"]) for row in records),
                1,
            )
        ),
        "exact_guard_candidate_preflight_attempt_count": sum(
            int(row["exact_guard_candidate_preflight_attempts"]) for row in records
        ),
        "exact_guard_safe_candidate_step_count": sum(
            int(row["exact_guard_safe_candidate_steps"]) for row in records
        ),
        "exact_guard_alternative_selected_step_count": sum(
            int(row["exact_guard_alternative_selected_steps"]) for row in records
        ),
        "exact_guard_no_usable_candidate_step_count": sum(
            int(row["exact_guard_no_usable_candidate_steps"]) for row in records
        ),
        "exact_guard_emergency_escalation_step_count": sum(
            int(row["exact_guard_emergency_escalation_steps"]) for row in records
        ),
        "exact_guard_emergency_preflight_attempt_count": sum(
            int(row["exact_guard_emergency_preflight_attempts"]) for row in records
        ),
        "exact_guard_emergency_recovered_step_count": sum(
            int(row["exact_guard_emergency_recovered_steps"]) for row in records
        ),
        "terminal_reason_counts": dict(Counter(str(row["terminal_reason"]) for row in records)),
        "precontact_terminal_count": 0,
        "precontact_is_terminal": False,
        "strict_three_second_success_is_only_positive_terminal": True,
        "object_reward_requires_latched_admissible_contact": True,
        "robot_coupled_unattributed_block_motion_fails_closed": True,
        "passive_unattributed_object_drift_is_recorded_not_rewarded": True,
        "curriculum_trajectory_export_count": 0,
        "full_task_start_geometry_eligible_episode_count": sum(
            int(row["full_task_start_geometry_eligible"]) for row in records
        ),
        "near_target_or_privileged_curriculum_export_count": 0,
        "strict_home_wrist_export_candidate_count": len(strict_home_wrist_candidates),
        "bulk_vla_data_use_allowed": False,
        "full_task_success_claimed": bool(strict_home_wrist_candidates),
        "act_training_started": False,
        "wrist_multimodal_export_started": False,
        "production_admission": False,
    }
    _atomic_json(destination / "summary.json", summary)
    _atomic_json(
        destination / "run_state.json",
        {
            "status": "complete",
            "episode_count": len(records),
            "replay_transition_count": replay.size,
            "initial_replay_transition_count": (initial_replay_transition_count),
            "new_replay_transition_count": min(
                total_environment_steps,
                max(replay.size - initial_replay_transition_count, 0),
            ),
            "update_index": bundle.update_index,
            "offline_acquisition_self_imitation_update_count": (
                selected_training.offline_acquisition_self_imitation_updates
            ),
            "offline_acquisition_self_imitation_qualified_sample_count": (
                offline_acquisition_qualified_sample_count
            ),
            "offline_acquisition_progress_model_update_count": (
                selected_training.offline_acquisition_progress_model_updates
            ),
            "offline_acquisition_progress_model_qualified_sample_count": (
                offline_acquisition_progress_qualified_sample_count
            ),
            "strict_home_success_count": home_successes,
            "strict_home_wrist_export_candidate_count": len(strict_home_wrist_candidates),
            "production_admission": False,
            "updated_at_utc": _utc_now(),
        },
    )
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train full-task five-joint SAC without precontact termination"
    )
    parser.add_argument("--source-training-run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=12)
    parser.add_argument("--maximum-steps", type=int, default=480)
    parser.add_argument("--seed", type=int, default=666_000_000)
    parser.add_argument(
        "--execution-mode",
        choices=sorted(_EXECUTION_MODES_V666),
        default="fast_v10_learning_only",
    )
    parser.add_argument("--warmup-transitions", type=int, default=512)
    parser.add_argument(
        "--offline-acquisition-self-imitation-updates",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--offline-acquisition-progress-model-updates",
        type=int,
        default=0,
    )
    parser.add_argument("--exact-home-probe-interval", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--replay-capacity", type=int, default=100_000)
    parser.add_argument("--replay-save-interval-episodes", type=int, default=1)
    parser.add_argument("--warm-start-checkpoint", type=Path)
    parser.add_argument("--warm-start-value-functions", action="store_true")
    parser.add_argument("--warm-start-optimizer-state", action="store_true")
    parser.add_argument("--warm-start-replay", type=Path)
    parser.add_argument("--curriculum-history-run-dir", type=Path)
    parser.add_argument("--policy-candidate-count", type=int, default=16)
    parser.add_argument(
        "--policy-feasibility-shortlist-fraction",
        type=float,
        default=0.25,
    )
    parser.add_argument("--exact-guard-candidate-preflight-count", type=int, default=2)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    sac_config = JointGoalSACConfigV665(
        batch_size=args.batch_size,
        hidden_dim=args.hidden_dim,
        replay_capacity=args.replay_capacity,
    )
    training_config = FullTaskJointTrainingConfigV666(
        warmup_transitions=args.warmup_transitions,
        offline_acquisition_self_imitation_updates=(args.offline_acquisition_self_imitation_updates),
        offline_acquisition_progress_model_updates=(args.offline_acquisition_progress_model_updates),
        exact_home_probe_interval=args.exact_home_probe_interval,
        policy_candidate_count=args.policy_candidate_count,
        policy_feasibility_shortlist_fraction=(args.policy_feasibility_shortlist_fraction),
        exact_guard_candidate_preflight_count=(args.exact_guard_candidate_preflight_count),
        replay_save_interval_episodes=args.replay_save_interval_episodes,
    )
    try:
        summary = train_full_task_joint_sac_v666(
            source_training_run_dir=args.source_training_run_dir,
            output_dir=args.output_dir,
            episodes=args.episodes,
            maximum_steps=args.maximum_steps,
            seed=args.seed,
            execution_mode=args.execution_mode,
            warm_start_checkpoint=args.warm_start_checkpoint,
            warm_start_value_functions=args.warm_start_value_functions,
            warm_start_optimizer_state=args.warm_start_optimizer_state,
            warm_start_replay=args.warm_start_replay,
            curriculum_history_run_dir=args.curriculum_history_run_dir,
            sac_config=sac_config,
            training_config=training_config,
        )
    except Exception as error:
        _mark_failed_run_v666(args.output_dir, error)
        raise
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CONTACT_ADAPTIVE_PROJECTION_FORMAT_V681",
    "FULL_TASK_JOINT_SAC_TRAINING_FORMAT_V666",
    "FullTaskJointTrainingConfigV666",
    "acquisition_axis_ranked_candidate_order_v686",
    "adaptive_exact_preflight_schedule_v677",
    "applied_policy_action_from_plant_v666",
    "complete_candidate_preference_order_v677",
    "fast_joint_command_v666",
    "feasibility_ranked_candidate_index_v671",
    "feasibility_ranked_candidate_order_v672",
    "select_start_fraction_v666",
    "train_full_task_joint_sac_v666",
    "unattributed_motion_audit_v666",
    "warm_start_joint_goal_bundle_v670",
    "load_curriculum_history_v670",
    "projection_regime_v681",
]
