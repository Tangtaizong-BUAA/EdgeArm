"""Diagnose exact-Home reachability through the deployed V597 action interface.

The learned acquisition policy currently approaches the block from exact Home
but stalls outside the contact-ready region.  This module separates two
possible causes without training or creating demonstration data:

* policy/exploration failure, when bounded direct goal feedback can traverse
  the unchanged guarded task-frame interface; or
* action-interface/IK failure, when the same interface stalls even under
  direct feedback on the identical fixed tasks.

The feedback probe uses simulator state and is therefore an oracle diagnostic.
Its actions, images, and transitions are forbidden from replay, imitation,
ACT/VLA training, production admission, and wrist-multimodal export.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .asymmetric_multiview_ppo_v1 import (
    SOURCE_TYPE,
    VIEW_NAMES,
    StockTaskFrameNoSafeRecoveryV13,
    canonical_sha256_v1,
    sha256_file_v1,
)
from .bounded_taskframe_acquisition_v654 import (
    BoundedAcquisitionActionConfigV654,
)
from .causal_smooth_relay_v646 import CausalSmoothRelayConfigV646
from .goal_conditioned_her_sac_v43 import (
    desired_goal_from_privileged_v43,
    goal_neutral_privileged_state_v43,
)
from .phase_isolated_acquisition_v626 import (
    PhaseIsolatedAcquisitionConfigV626,
)
from .privileged_effect_state_v1 import build_privileged_effect_state_v1
from .progressive_home_alignment_v662 import (
    install_progressive_home_alignment_v662,
)
from .relay_dual_goal_her_sac_v643 import precontact_goal_xyz_v643
from .sim2real_env_v10 import RealisticEdgeArmEnvV10
from .stock_gripper_taskframe_v22 import transition_contact_telemetry_v22
from .task_independent_certified_long_range_reset_v649 import (
    reset_stock_home_certified_long_range_episode_v649,
)
from .task_independent_home_reset_v597 import (
    HOME_ACQUISITION_TRANSPORT_ALIGNMENT_V597,
    StockGripperHomeTaskFrameAdapterV597,
)
from .train_causal_smooth_long_range_relay_v650 import _stage_runtime_v650
from .train_goal_conditioned_her_sac_v43 import (
    _append_jsonl,
    _atomic_json,
    _load_collection_contract,
    _utc_now,
)


EXACT_HOME_TASKFRAME_REACHABILITY_FORMAT_V661 = (
    "edgearm-v661-exact-home-taskframe-reachability-diagnostic-v1"
)
LONG_RANGE_STAGE_INDEX_V661 = 33


@dataclass(frozen=True)
class ExactHomeReachabilityProbeConfigV661:
    """Bounded diagnostic feedback with the same V654 actor support."""

    action_absolute: tuple[float, float, float] = (0.35, 0.35, 0.25)
    contact_ready_distance_m: float = 0.020
    contact_ready_alignment: float = HOME_ACQUISITION_TRANSPORT_ALIGNMENT_V597
    minimum_material_progress_m: float = 0.010

    def validate(self) -> None:
        action = np.asarray(self.action_absolute, dtype=np.float64)
        if (
            action.shape != (3,)
            or not np.all(np.isfinite(action))
            or np.any(action <= 0.0)
            or np.any(action > 1.0)
        ):
            raise ValueError("V661 action bounds are invalid")
        for name in (
            "contact_ready_distance_m",
            "contact_ready_alignment",
            "minimum_material_progress_m",
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"V661 {name} must be finite and positive")
        if self.contact_ready_distance_m > 0.05:
            raise ValueError("V661 contact-ready distance is too permissive")
        if not 0.9 <= self.contact_ready_alignment <= 1.0:
            raise ValueError("V661 contact-ready alignment is invalid")


class _NoImageResetRendererV661:
    view_names = VIEW_NAMES

    def __init__(self) -> None:
        self.episode_seed: int | None = None

    def begin_episode(self, seed: int) -> None:
        self.episode_seed = int(seed)


def taskframe_goal_feedback_action_v661(
    *,
    tool_xyz_m: np.ndarray,
    goal_xyz_m: np.ndarray,
    forward_xy: np.ndarray,
    lateral_xy: np.ndarray,
    translation_scale_xyz_m: np.ndarray,
    action_absolute: np.ndarray,
) -> np.ndarray:
    """Map instantaneous Cartesian goal error into bounded task-frame action."""

    tool = np.asarray(tool_xyz_m, dtype=np.float64)
    goal = np.asarray(goal_xyz_m, dtype=np.float64)
    forward = np.asarray(forward_xy, dtype=np.float64)
    lateral = np.asarray(lateral_xy, dtype=np.float64)
    scale = np.asarray(translation_scale_xyz_m, dtype=np.float64)
    bound = np.asarray(action_absolute, dtype=np.float64)
    if (
        tool.shape != (3,)
        or goal.shape != (3,)
        or forward.shape != (2,)
        or lateral.shape != (2,)
        or scale.shape != (3,)
        or bound.shape != (3,)
        or not np.all(np.isfinite(np.r_[tool, goal, forward, lateral, scale, bound]))
        or np.any(scale <= 0.0)
        or np.any(bound <= 0.0)
        or np.any(bound > 1.0)
    ):
        raise ValueError("V661 goal-feedback inputs are invalid")
    forward_norm = float(np.linalg.norm(forward))
    lateral_norm = float(np.linalg.norm(lateral))
    if (
        not np.isclose(forward_norm, 1.0, rtol=0.0, atol=1.0e-6)
        or not np.isclose(lateral_norm, 1.0, rtol=0.0, atol=1.0e-6)
        or not np.isclose(float(np.dot(forward, lateral)), 0.0, rtol=0.0, atol=1.0e-6)
    ):
        raise ValueError("V661 task-frame axes are not orthonormal")
    error = goal - tool
    local_error = np.asarray(
        [
            float(np.dot(error[:2], forward)),
            float(np.dot(error[:2], lateral)),
            float(error[2]),
        ],
        dtype=np.float64,
    )
    return np.clip(local_error / scale, -bound, bound).astype(np.float32)


def _precontact_goal_v661(
    env: RealisticEdgeArmEnvV10,
    phase_config: PhaseIsolatedAcquisitionConfigV626,
) -> np.ndarray:
    privileged = build_privileged_effect_state_v1(env)
    neutral = goal_neutral_privileged_state_v43(privileged)
    desired = desired_goal_from_privileged_v43(privileged)
    return precontact_goal_xyz_v643(
        neutral[None],
        desired[None],
        phase_config=phase_config,
    )[0].astype(np.float64)


def _face_alignment_v661(
    env: RealisticEdgeArmEnvV10,
    forward_xy: np.ndarray,
) -> float:
    normal = env.data.site_xmat[env._ids["tool_site"]].reshape(3, 3)[:, 1]
    horizontal_norm = float(np.linalg.norm(normal[:2]))
    return abs(float(np.dot(normal[:2], forward_xy))) / max(
        horizontal_norm,
        1.0e-12,
    )


def classify_exact_home_reachability_v661(
    episodes: Sequence[dict[str, Any]],
    *,
    config: ExactHomeReachabilityProbeConfigV661 | None = None,
) -> dict[str, Any]:
    """Classify the fixed-seed probe without promoting it into training data."""

    selected = config or ExactHomeReachabilityProbeConfigV661()
    selected.validate()
    rows = list(episodes)
    if not rows:
        raise ValueError("V661 classification requires episode evidence")
    if any(
        type(row) is not dict
        or row.get("diagnostic_only") is not True
        or row.get("production_admission") is not False
        for row in rows
    ):
        raise ValueError("V661 episode evidence is not fail-closed")
    contact_count = sum(int(bool(row.get("valid_contact_reached"))) for row in rows)
    ready_count = sum(int(bool(row.get("contact_ready_region_reached"))) for row in rows)
    material_count = sum(
        int(float(row["best_precontact_progress_m"]) >= selected.minimum_material_progress_m)
        for row in rows
    )
    if contact_count == len(rows):
        diagnosis = "action_interface_reachable_policy_learning_bottleneck"
    elif contact_count == 0 and ready_count == 0:
        diagnosis = "action_interface_or_local_ik_bottleneck"
    else:
        diagnosis = "seed_dependent_mixed_reachability"
    return {
        "format": EXACT_HOME_TASKFRAME_REACHABILITY_FORMAT_V661,
        "episode_count": len(rows),
        "valid_contact_episode_count": contact_count,
        "contact_ready_episode_count": ready_count,
        "material_progress_episode_count": material_count,
        "diagnosis": diagnosis,
        "probe_actions_are_privileged_diagnostic_only": True,
        "probe_transitions_admitted_to_replay": False,
        "expert_or_scripted_actions_used_for_training": False,
        "wrist_multimodal_export_started": False,
        "production_admission": False,
    }


def _run_probe_episode_v661(
    *,
    environment_config: Any,
    action_config: Any,
    scene_path: Path,
    phase_config: PhaseIsolatedAcquisitionConfigV626,
    requested_seed: int,
    maximum_steps: int,
    probe_config: ExactHomeReachabilityProbeConfigV661,
    step_log_path: Path,
    progressive_alignment_v662: bool,
) -> dict[str, Any]:
    env = RealisticEdgeArmEnvV10(
        environment_config,
        seed=requested_seed,
        model_scene_path=scene_path,
    )
    adapter = StockGripperHomeTaskFrameAdapterV597(env, action_config)
    progressive_alignment_audit = (
        install_progressive_home_alignment_v662(adapter)
        if progressive_alignment_v662
        else None
    )
    renderer = _NoImageResetRendererV661()
    reset = reset_stock_home_certified_long_range_episode_v649(
        env,
        renderer,
        adapter,
        requested_seed=requested_seed,
        obstacle=False,
        stress=False,
    )
    initial_block = env.block_xy().copy()
    initial_object_target_distance = float(env.distance_to_target())
    initial_coverage = float(env.block_target_coverage())
    initial_goal = _precontact_goal_v661(env, phase_config)
    initial_tool = env.tool_xyz().copy()
    initial_distance = float(np.linalg.norm(initial_goal - initial_tool))
    minimum_distance = initial_distance
    minimum_distance_step = 0
    maximum_alignment = 0.0
    valid_contact_steps = 0
    invalid_contact_steps = 0
    raw_contact_steps = 0
    ik_converged_steps = 0
    shield_projection_steps = 0
    shield_terminal = False
    strict_success = False
    contact_ready_step: int | None = None
    first_valid_contact_step: int | None = None
    terminal_reason = "diagnostic_time_limit"
    failure_reasons: Counter[str] = Counter()
    subphases: Counter[str] = Counter()
    minimum_safety_clearance = float("inf")
    executed_steps = 0
    scale = np.asarray(
        [
            action_config.forward_translation_step_m,
            action_config.lateral_translation_step_m,
            action_config.vertical_translation_step_m,
        ],
        dtype=np.float64,
    )
    action_bound = np.asarray(probe_config.action_absolute, dtype=np.float64)

    for step_index in range(maximum_steps):
        goal = _precontact_goal_v661(env, phase_config)
        tool_before = env.tool_xyz().copy()
        forward, lateral = adapter._task_axes(env)
        distance_before = float(np.linalg.norm(goal - tool_before))
        alignment_before = _face_alignment_v661(env, forward)
        action = taskframe_goal_feedback_action_v661(
            tool_xyz_m=tool_before,
            goal_xyz_m=goal,
            forward_xy=forward,
            lateral_xy=lateral,
            translation_scale_xyz_m=scale,
            action_absolute=action_bound,
        )
        block_before = env.block_xy().copy()
        try:
            translated = adapter.translate(action)
        except StockTaskFrameNoSafeRecoveryV13 as error:
            shield_terminal = True
            terminal_reason = "v22_action_shield_terminal"
            failure_reasons[type(error).__name__] += 1
            _append_jsonl(
                step_log_path,
                {
                    "step": step_index,
                    "tool_xyz_before_m": tool_before.tolist(),
                    "precontact_goal_xyz_m": goal.tolist(),
                    "tool_precontact_distance_before_m": distance_before,
                    "precontact_face_alignment_before": alignment_before,
                    "requested_task_action": action.tolist(),
                    "shield_terminal": True,
                    "exception_type": type(error).__name__,
                    "exception_message": str(error),
                    "diagnostic_only": True,
                    "production_admission": False,
                },
            )
            break

        failure_reasons[str(translated.failure_reason)] += 1
        ik_converged_steps += int(bool(translated.ik_converged))
        shield_projection_steps += int(not bool(translated.ik_converged))
        acquisition_report = dict(adapter.last_acquisition_report_v597)
        subphase = str(acquisition_report.get("acquisition_subphase", "parent_v22"))
        subphases[subphase] += 1
        _observation, _reward, terminated, truncated, info = env.step(
            translated.submitted_joint_action
        )
        executed_steps += 1
        telemetry = transition_contact_telemetry_v22(
            info,
            block_before_xy_m=block_before,
            block_after_xy_m=env.block_xy(),
        )
        tool_after = env.tool_xyz().copy()
        current_goal = _precontact_goal_v661(env, phase_config)
        current_forward, _current_lateral = adapter._task_axes(env)
        distance_after = float(np.linalg.norm(current_goal - tool_after))
        alignment_after = _face_alignment_v661(env, current_forward)
        minimum_safety_clearance = min(
            minimum_safety_clearance,
            float(telemetry["minimum_executed_safety_only_block_clearance_m"]),
        )
        if distance_after < minimum_distance:
            minimum_distance = distance_after
            minimum_distance_step = step_index + 1
        maximum_alignment = max(maximum_alignment, alignment_before, alignment_after)
        contact_ready = bool(
            distance_after <= probe_config.contact_ready_distance_m
            and alignment_after >= probe_config.contact_ready_alignment
        )
        if contact_ready and contact_ready_step is None:
            contact_ready_step = step_index + 1
        valid_contact = bool(telemetry["valid_push_side_contact_any"])
        invalid_contact = bool(telemetry["invalid_tool_block_contact_any"])
        raw_contact = bool(telemetry["tool_block_contact_any"])
        valid_contact_steps += int(valid_contact)
        invalid_contact_steps += int(invalid_contact)
        raw_contact_steps += int(raw_contact)
        if valid_contact and first_valid_contact_step is None:
            first_valid_contact_step = step_index + 1
        strict_success = strict_success or bool(info.get("success", False))
        _append_jsonl(
            step_log_path,
            {
                "step": step_index,
                "tool_xyz_before_m": tool_before.tolist(),
                "tool_xyz_after_m": tool_after.tolist(),
                "precontact_goal_xyz_m": current_goal.tolist(),
                "tool_precontact_distance_before_m": distance_before,
                "tool_precontact_distance_after_m": distance_after,
                "precontact_face_alignment_before": alignment_before,
                "precontact_face_alignment_after": alignment_after,
                "requested_task_action": action.tolist(),
                "applied_task_action": np.asarray(
                    translated.applied_task_action,
                    dtype=np.float64,
                ).tolist(),
                "submitted_joint_action": np.asarray(
                    translated.submitted_joint_action,
                    dtype=np.float64,
                ).tolist(),
                "joint_position_after_rad": np.asarray(
                    env.data.qpos[:6],
                    dtype=np.float64,
                ).tolist(),
                "application_scale": float(translated.application_scale),
                "ik_converged": bool(translated.ik_converged),
                "failure_reason": str(translated.failure_reason),
                "acquisition_subphase": subphase,
                "contact_ready": contact_ready,
                "raw_contact": raw_contact,
                "valid_contact": valid_contact,
                "invalid_contact": invalid_contact,
                "block_step_displacement_m": float(
                    telemetry["step_block_displacement_m"]
                ),
                "minimum_safety_only_clearance_m": float(
                    telemetry[
                        "minimum_executed_safety_only_block_clearance_m"
                    ]
                ),
                "terminated": bool(terminated),
                "truncated": bool(truncated),
                "diagnostic_only": True,
                "production_admission": False,
            },
        )
        if valid_contact:
            terminal_reason = "valid_contact_reached"
            break
        if terminated or truncated:
            terminal_reason = (
                "environment_terminated" if terminated else "environment_truncated"
            )
            break

    final_block = env.block_xy().copy()
    return {
        "format": EXACT_HOME_TASKFRAME_REACHABILITY_FORMAT_V661,
        "requested_seed": requested_seed,
        "selected_seed": int(reset["selected_seed"]),
        "exact_home_reset": True,
        "privileged_curriculum_reset": False,
        "initial_target_coverage": initial_coverage,
        "initial_object_target_distance_m": initial_object_target_distance,
        "initial_tool_xyz_m": initial_tool.tolist(),
        "initial_precontact_goal_xyz_m": initial_goal.tolist(),
        "initial_tool_precontact_distance_m": initial_distance,
        "minimum_tool_precontact_distance_m": minimum_distance,
        "minimum_distance_step": minimum_distance_step,
        "best_precontact_progress_m": initial_distance - minimum_distance,
        "maximum_precontact_face_alignment": maximum_alignment,
        "contact_ready_region_reached": contact_ready_step is not None,
        "first_contact_ready_step": contact_ready_step,
        "valid_contact_reached": first_valid_contact_step is not None,
        "first_valid_contact_step": first_valid_contact_step,
        "valid_contact_steps": valid_contact_steps,
        "invalid_contact_steps": invalid_contact_steps,
        "raw_contact_steps": raw_contact_steps,
        "ik_converged_steps": ik_converged_steps,
        "ik_failure_or_projection_steps": shield_projection_steps,
        "failure_reason_counts": dict(sorted(failure_reasons.items())),
        "acquisition_subphase_counts": dict(sorted(subphases.items())),
        "minimum_safety_only_clearance_m": (
            minimum_safety_clearance
            if np.isfinite(minimum_safety_clearance)
            else None
        ),
        "steps_executed": executed_steps,
        "shield_terminal": shield_terminal,
        "terminal_reason": terminal_reason,
        "strict_success_observed": strict_success,
        "net_object_target_progress_m": (
            initial_object_target_distance - float(env.distance_to_target())
        ),
        "cumulative_block_displacement_m": float(
            np.linalg.norm(final_block - initial_block)
        ),
        "probe_control_law": "instantaneous_privileged_precontact_goal_feedback",
        "progressive_home_alignment_v662": progressive_alignment_v662,
        "progressive_home_alignment_audit_v662": progressive_alignment_audit,
        "probe_uses_future_path_or_waypoints": False,
        "probe_action_used_for_training": False,
        "probe_transition_admitted_to_replay": False,
        "diagnostic_only": True,
        "act_training_started": False,
        "wrist_multimodal_export_started": False,
        "bulk_vla_data_use_allowed": False,
        "production_admission": False,
    }


def diagnose_exact_home_taskframe_reachability_v661(
    *,
    training_run_dir: Path,
    output_dir: Path,
    seeds: Sequence[int],
    maximum_steps: int,
    probe_config: ExactHomeReachabilityProbeConfigV661 | None = None,
    progressive_alignment_v662: bool = False,
) -> dict[str, Any]:
    selected = probe_config or ExactHomeReachabilityProbeConfigV661()
    selected.validate()
    seed_values = tuple(int(seed) for seed in seeds)
    if (
        not seed_values
        or len(set(seed_values)) != len(seed_values)
        or any(seed < 0 for seed in seed_values)
    ):
        raise ValueError("V661 seeds must be unique non-negative integers")
    if type(maximum_steps) is not int or maximum_steps < 30:
        raise ValueError("V661 maximum steps is invalid")
    if type(progressive_alignment_v662) is not bool:
        raise TypeError("V661 progressive alignment flag must be boolean")
    source = Path(training_run_dir).expanduser().resolve()
    destination = Path(output_dir).expanduser().resolve()
    if destination.exists():
        raise FileExistsError(f"V661 output exists: {destination}")
    source_plan_path = source / "run_plan.json"
    if not source_plan_path.is_file():
        raise FileNotFoundError("V661 source training run plan is missing")
    source_plan = json.loads(source_plan_path.read_text(encoding="utf-8"))
    source_hash = source_plan.get("run_plan_sha256")
    unhashed = dict(source_plan)
    unhashed.pop("run_plan_sha256", None)
    if source_hash != canonical_sha256_v1(unhashed):
        raise ValueError("V661 source training run-plan hash is invalid")
    collection_plan_path = Path(
        source_plan["source_collection_plan"]
    ).expanduser().resolve()
    _collection_plan, base_environment, base_action, scene_path = (
        _load_collection_contract(collection_plan_path)
    )
    causal_config = CausalSmoothRelayConfigV646(
        **source_plan["causal_motion_config_v646"]
    )
    phase_config = PhaseIsolatedAcquisitionConfigV626(
        **source_plan["phase_isolated_acquisition_v626"]
    )
    expected_action = BoundedAcquisitionActionConfigV654(
        **source_plan["bounded_action_config_v654"]
    )
    if tuple(expected_action.action_absolute) != tuple(selected.action_absolute):
        raise ValueError("V661 probe action bounds differ from the learned actor")
    environment_config, action_config = _stage_runtime_v650(
        base_environment,
        base_action,
        stage_index=LONG_RANGE_STAGE_INDEX_V661,
        maximum_steps=maximum_steps,
        causal_config=causal_config,
    )
    destination.mkdir(parents=True)
    steps_dir = destination / "steps"
    steps_dir.mkdir()
    run_plan = {
        "format": EXACT_HOME_TASKFRAME_REACHABILITY_FORMAT_V661,
        "created_at_utc": _utc_now(),
        "source_training_run_dir": str(source),
        "source_training_run_plan": str(source_plan_path),
        "source_training_run_plan_sha256": source_hash,
        "source_collection_plan": str(collection_plan_path),
        "source_collection_plan_sha256": sha256_file_v1(collection_plan_path),
        "scene_path": str(scene_path),
        "scene_sha256": sha256_file_v1(scene_path),
        "seeds": list(seed_values),
        "maximum_steps": maximum_steps,
        "long_range_stage": LONG_RANGE_STAGE_INDEX_V661,
        "probe_config": asdict(selected),
        "phase_isolated_acquisition_v626": asdict(phase_config),
        "environment_config": asdict(environment_config),
        "action_config": asdict(action_config),
        "source_type": SOURCE_TYPE,
        "same_policy_xyz_action_contract_as_training": True,
        "same_controller_schedule_as_source_training": bool(
            not progressive_alignment_v662
        ),
        "progressive_home_alignment_v662": progressive_alignment_v662,
        "probe_actions_are_privileged_diagnostic_only": True,
        "probe_actions_admitted_to_training": False,
        "probe_transitions_admitted_to_replay": False,
        "act_training_started": False,
        "wrist_multimodal_export_started": False,
        "production_admission": False,
    }
    run_plan["run_plan_sha256"] = canonical_sha256_v1(run_plan)
    _atomic_json(destination / "run_plan.json", run_plan)

    records: list[dict[str, Any]] = []
    for seed in seed_values:
        record = _run_probe_episode_v661(
            environment_config=environment_config,
            action_config=action_config,
            scene_path=scene_path,
            phase_config=phase_config,
            requested_seed=seed,
            maximum_steps=maximum_steps,
            probe_config=selected,
            step_log_path=steps_dir / f"seed_{seed}.jsonl",
            progressive_alignment_v662=progressive_alignment_v662,
        )
        records.append(record)
        _append_jsonl(destination / "episodes.jsonl", record)

    classification = classify_exact_home_reachability_v661(
        records,
        config=selected,
    )
    summary = {
        **classification,
        "status": "complete",
        "created_at_utc": _utc_now(),
        "run_plan_sha256": run_plan["run_plan_sha256"],
        "mean_initial_tool_precontact_distance_m": float(
            np.mean(
                [row["initial_tool_precontact_distance_m"] for row in records]
            )
        ),
        "mean_minimum_tool_precontact_distance_m": float(
            np.mean(
                [row["minimum_tool_precontact_distance_m"] for row in records]
            )
        ),
        "mean_best_precontact_progress_m": float(
            np.mean([row["best_precontact_progress_m"] for row in records])
        ),
        "total_ik_failure_or_projection_steps": int(
            sum(row["ik_failure_or_projection_steps"] for row in records)
        ),
        "shield_terminal_episode_count": int(
            sum(bool(row["shield_terminal"]) for row in records)
        ),
        "strict_success_claimed": False,
        "full_task_success_claimed": False,
        "next_decision": (
            "redesign_or_expand_action_interface_before_more_reward_tuning"
            if classification["diagnosis"]
            == "action_interface_or_local_ik_bottleneck"
            else "retain_action_interface_and_repair_policy_learning"
            if classification["diagnosis"]
            == "action_interface_reachable_policy_learning_bottleneck"
            else "stratify_failures_before_architecture_change"
        ),
        "episodes": records,
        "diagnostic_only": True,
        "act_training_started": False,
        "wrist_multimodal_export_started": False,
        "production_admission": False,
    }
    _atomic_json(destination / "summary.json", summary)
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Diagnose exact-Home task-frame reachability without training"
    )
    parser.add_argument("--training-run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, action="append", required=True)
    parser.add_argument("--maximum-steps", type=int, default=480)
    parser.add_argument("--progressive-alignment-v662", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    summary = diagnose_exact_home_taskframe_reachability_v661(
        training_run_dir=args.training_run_dir,
        output_dir=args.output_dir,
        seeds=args.seed,
        maximum_steps=args.maximum_steps,
        progressive_alignment_v662=args.progressive_alignment_v662,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "EXACT_HOME_TASKFRAME_REACHABILITY_FORMAT_V661",
    "ExactHomeReachabilityProbeConfigV661",
    "classify_exact_home_reachability_v661",
    "diagnose_exact_home_taskframe_reachability_v661",
    "taskframe_goal_feedback_action_v661",
]
