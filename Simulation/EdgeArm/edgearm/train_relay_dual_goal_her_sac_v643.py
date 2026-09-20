"""Train the V643 relay dual-goal Home-to-contact RL option.

This runner upgrades a completed V614/V640 lineage without importing its
failed acquisition residual.  The verified transport actor and feasibility
model are frozen.  A separate tool-goal actor/critic is first trained from the
persisted replay and is then refined using exact-Home, stage-32 (17--19 cm)
online interaction.  No expert action, path, waypoint, behavior cloning, ACT,
or wrist-image training is used here.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import torch

from .adaptive_start_curriculum_v622 import START_TIERS_V622
from .asymmetric_multiview_ppo_v1 import (
    SOURCE_TYPE,
    canonical_sha256_v1,
    sha256_file_v1,
)
from .goal_conditioned_her_sac_v43 import (
    ACTION_DIM_V43,
    GoalConditionedHerSACConfigV43,
    goal_neutral_privileged_state_v43,
    observation_with_goal_v43,
)
from .goal_conditioned_markov_her_sac_v614 import (
    GOAL_CONDITIONED_MARKOV_HER_CHECKPOINT_FORMAT_V614,
    GoalConditionedMarkovHerReplayV614,
)
from .phase_isolated_acquisition_v626 import (
    PHASE_ISOLATED_ACQUISITION_FORMAT_V626,
    PhaseIsolatedAcquisitionConfigV626,
)
from .relay_dual_goal_her_sac_v643 import (
    RELAY_DUAL_GOAL_CHECKPOINT_FORMAT_V643,
    RELAY_DUAL_GOAL_HER_SAC_FORMAT_V643,
    RelayDualGoalHerSACBundleV643,
    RelayDualGoalHerSACConfigV643,
    initialize_relay_dual_goal_her_sac_v643,
    precontact_goal_xyz_v643,
    relay_dual_goal_her_sac_update_v643,
    sample_relay_acquisition_batch_v643,
)
from .reverse_curriculum_v26 import wilson_lower_bound_v26
from .sim2real_env_v10 import RealisticEdgeArmEnvV10
from .stock_gripper_rollout_kernel_v22 import StockGripperRolloutKernelV22
from .stock_gripper_taskframe_v22 import (
    StockGripperTaskFrameAdapterV22,
    reset_stock_taskframe_episode_v22,
)
from .task_independent_home_reset_v597 import (
    StockGripperHomeTaskFrameAdapterV597,
    reset_stock_home_taskframe_episode_v597,
)
from .taskframe_controller_state_v614 import (
    TASKFRAME_CONTROLLER_STATE_SCHEMA_SHA256_V614,
    build_taskframe_controller_state_v614,
)
from .train_goal_conditioned_her_sac_v43 import (
    MINIMUM_FULL_TASK_INITIAL_DISTANCE_M_V607,
    MINIMUM_FULL_TASK_NET_PROGRESS_M_V607,
    AxisScaledColoredExplorationConfigV605,
    AxisScaledColoredExplorerV605,
    _append_jsonl,
    _atomic_json,
    _atomic_torch_save,
    _load_collection_contract,
    _resolve_device,
    _run_episode_v43,
    _training_selection_score_v610,
    _utc_now,
)
from .train_goal_conditioned_markov_her_sac_v614 import (
    _compact_hard_terminal_failure_v619,
    _failure_cohort_audit_v619,
)


RELAY_DUAL_GOAL_TRAIN_RUN_FORMAT_V643 = "edgearm-v643-relay-dual-goal-online-training-run-v1"
RELAY_DUAL_GOAL_EVALUATION_FORMAT_V643 = "edgearm-v643-relay-dual-goal-heldout-evaluation-v1"

_HOME_TIER_INDEX_V643 = next(tier.index for tier in START_TIERS_V622 if tier.code == "home")


def phase_preserving_random_exploration_v702(
    unit_action: np.ndarray,
    *,
    acquisition_gate: float,
    acquisition_action_absolute: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Match random exploration to the active blended-policy support.

    This is support matching only. It adds no task direction, waypoint,
    expert action, or future-state information.
    """

    unit = np.asarray(unit_action, dtype=np.float32)
    acquisition = np.asarray(
        acquisition_action_absolute,
        dtype=np.float32,
    )
    gate = float(acquisition_gate)
    if (
        unit.shape != (ACTION_DIM_V43,)
        or acquisition.shape != (ACTION_DIM_V43,)
        or not np.all(np.isfinite(np.r_[unit, acquisition, gate]))
        or np.any(np.abs(unit) > 1.0 + 1.0e-6)
        or np.any(acquisition <= 0.0)
        or np.any(acquisition > 1.0)
        or not 0.0 <= gate <= 1.0
    ):
        raise ValueError("V702 phase-preserving exploration inputs are invalid")
    support = np.add(
        np.float32(1.0 - gate),
        np.float32(gate) * acquisition,
        dtype=np.float32,
    )
    return np.multiply(unit, support, dtype=np.float32), support


def _relay_policy_action_v643(
    *,
    bundle: RelayDualGoalHerSACBundleV643,
    privileged_state: np.ndarray,
    desired_goal: np.ndarray,
    adapter: Any,
    deterministic: bool,
    rng: np.random.Generator,
    random_action_probability: float,
    random_explorer: AxisScaledColoredExplorerV605 | None,
) -> tuple[np.ndarray, dict[str, Any]]:
    neutral = goal_neutral_privileged_state_v43(privileged_state)
    object_goal = np.asarray(desired_goal, dtype=np.float32)
    transport_observation = observation_with_goal_v43(neutral, object_goal)
    controller_state = build_taskframe_controller_state_v614(adapter)
    acquisition_goal = precontact_goal_xyz_v643(
        neutral[None],
        object_goal[None],
        phase_config=bundle.policy.phase_config,
    )[0]
    device = next(bundle.policy.acquisition_actor.parameters()).device
    bundle.policy.eval()
    with torch.no_grad():
        policy_action, audit = bundle.policy.sample(
            torch.from_numpy(transport_observation).to(device).unsqueeze(0),
            torch.from_numpy(controller_state).to(device).unsqueeze(0),
            torch.from_numpy(acquisition_goal).to(device).unsqueeze(0),
            deterministic=deterministic,
        )
    acquisition_gate_value = float(audit["gate"][0].item())
    exploration_gate_eligible_v748 = bool(
        random_explorer is None
        or acquisition_gate_value >= random_explorer.config.acquisition_gate_minimum_v748
    )
    random_selected = bool(
        not deterministic
        and exploration_gate_eligible_v748
        and (
            random_explorer.select_v748(
                random_action_probability,
                rng=rng,
            )
            if random_explorer is not None
            else rng.random() < random_action_probability
        )
    )
    if random_selected:
        raw_random_action = (
            random_explorer.sample()
            if random_explorer is not None
            else rng.uniform(-1.0, 1.0, size=3).astype(np.float32)
        )
        bounded_config = getattr(
            bundle.policy.acquisition_actor,
            "action_config_v654",
            None,
        )
        acquisition_absolute = np.asarray(
            ((1.0, 1.0, 1.0) if bounded_config is None else bounded_config.action_absolute),
            dtype=np.float32,
        )
        action, random_action_support = phase_preserving_random_exploration_v702(
            raw_random_action,
            acquisition_gate=acquisition_gate_value,
            acquisition_action_absolute=acquisition_absolute,
        )
        source = "colored_random_exploration" if random_explorer else "uniform_random_exploration"
    else:
        action = policy_action.squeeze(0).cpu().numpy().astype(np.float32)
        raw_random_action = None
        random_action_support = None
        source = "relay_dual_goal_policy"
    runtime_audit = {
        "format": "edgearm-v643-runtime-relay-policy-audit-v1",
        "source": source,
        "acquisition_gate": acquisition_gate_value,
        "tool_precontact_distance_m": float(audit["distance_m"][0].item()),
        "precontact_alignment": float(audit["alignment"][0].item()),
        "contact_state": bool(audit["contact"][0].item()),
        "transport_action": audit["transport_action"][0].cpu().numpy().tolist(),
        "acquisition_action": audit["acquisition_action"][0].cpu().numpy().tolist(),
        "selected_action": np.asarray(action, dtype=np.float32).tolist(),
        "raw_random_action_v702": (
            None if raw_random_action is None else np.asarray(raw_random_action, dtype=np.float32).tolist()
        ),
        "random_action_support_v702": (
            None
            if random_action_support is None
            else np.asarray(random_action_support, dtype=np.float32).tolist()
        ),
        "random_exploration_matches_blended_policy_support_v702": True,
        "acquisition_goal_xyz_m": acquisition_goal.tolist(),
        "random_exploration_selected": random_selected,
        "random_exploration_gate_eligible_v748": (exploration_gate_eligible_v748),
        "random_exploration_burst_count_v748": (
            0 if random_explorer is None else random_explorer.burst_count_v748
        ),
        "random_exploration_burst_remaining_steps_v748": (
            0 if random_explorer is None else random_explorer.remaining_burst_steps_v748
        ),
        "expert_action_used": False,
        "waypoint_or_path_used": False,
        "production_admission": False,
    }
    taskframe_error = audit.get("taskframe_goal_error_features_v652")
    if taskframe_error is not None:
        runtime_audit["taskframe_goal_error_features_v652"] = taskframe_error[0].cpu().numpy().tolist()
        runtime_audit["acquisition_state_semantics"] = "normalized_tool_to_precontact_error_in_task_frame"
    return action, runtime_audit


def _run_episode_v643(
    env: RealisticEdgeArmEnvV10,
    adapter: Any,
    bundle: RelayDualGoalHerSACBundleV643,
    *,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
    runtime_policy_action_transformer: Callable[
        [np.ndarray, dict[str, Any]],
        tuple[np.ndarray, dict[str, Any]],
    ]
    | None = None,
    runtime_policy_transition_observer: Callable[[dict[str, Any]], None] | None = None,
    transition_audit_callback: Callable[[dict[str, Any]], None] | None = None,
    **kwargs: Any,
) -> tuple[dict[str, Any], dict[str, np.ndarray] | None]:
    controller_rows: list[np.ndarray] = []
    next_controller_rows: list[np.ndarray] = []
    policy_audits: list[dict[str, Any]] = []
    dls_causal_rows: list[dict[str, Any]] = []
    hard_terminal_failure: dict[str, Any] | None = None

    def policy_action(**policy_kwargs: Any) -> np.ndarray:
        action, audit = _relay_policy_action_v643(**policy_kwargs)
        if runtime_policy_action_transformer is not None:
            action, audit = runtime_policy_action_transformer(action, audit)
        policy_audits.append(audit)
        return action

    def pre_action(_env: RealisticEdgeArmEnvV10, _row: dict[str, Any]) -> None:
        controller_rows.append(build_taskframe_controller_state_v614(adapter))

    def transition(row: dict[str, Any]) -> None:
        nonlocal hard_terminal_failure
        if runtime_policy_transition_observer is not None:
            runtime_policy_transition_observer(row)
        if transition_audit_callback is not None:
            transition_audit_callback(row)
        next_controller_rows.append(build_taskframe_controller_state_v614(adapter))
        if progress_callback is not None:
            progress_callback(row)
        dls_projection = row.get("translation_audit", {}).get("dls_projection_v688")
        if isinstance(dls_projection, dict):
            dls_causal_rows.append(
                {
                    "requested": np.asarray(
                        dls_projection["requested_task_action"],
                        dtype=np.float32,
                    ),
                    "predicted": np.asarray(
                        dls_projection["predicted_normalized_task_action"],
                        dtype=np.float32,
                    ),
                    "applied": np.asarray(
                        row["applied_action"],
                        dtype=np.float32,
                    ),
                    "requested_to_predicted_l2": float(dls_projection["requested_to_predicted_task_l2"]),
                    "joint_limit_scale": float(dls_projection["joint_limit_scale"]),
                    "minimum_position_singular_value": float(
                        np.min(
                            np.asarray(
                                dls_projection["position_singular_values"],
                                dtype=np.float64,
                            )
                        )
                    ),
                    "orientation_residual_before_l2": float(
                        np.linalg.norm(
                            np.asarray(
                                dls_projection["orientation_residual_before"],
                                dtype=np.float64,
                            )
                        )
                    ),
                    "predicted_orientation_residual_after_l2": float(
                        np.linalg.norm(
                            np.asarray(
                                dls_projection["predicted_orientation_residual_after"],
                                dtype=np.float64,
                            )
                        )
                    ),
                    "valid_contact": bool(row["valid_contact"]),
                }
            )
        if bool(row.get("terminal", False)) and (
            bool(row.get("failure_terminal", False)) or bool(row.get("safety_violation", False))
        ):
            hard_terminal_failure = _compact_hard_terminal_failure_v619(row)

    record, episode = _run_episode_v43(
        env,
        adapter,
        bundle,  # type: ignore[arg-type]
        policy_action_callback=policy_action,
        pre_action_observation_callback=pre_action,
        transition_audit_callback=transition,
        **kwargs,
    )
    row_count = int(record["rows"])
    if not (len(controller_rows) == len(next_controller_rows) == len(policy_audits) == row_count):
        raise RuntimeError("V643 runtime audit row binding drifted")
    gates = np.asarray([row["acquisition_gate"] for row in policy_audits], dtype=np.float32)
    distances = np.asarray(
        [row["tool_precontact_distance_m"] for row in policy_audits],
        dtype=np.float32,
    )
    random_rows = np.asarray(
        [row["random_exploration_selected"] for row in policy_audits],
        dtype=bool,
    )
    exploration_gate_eligible_rows_v748 = np.asarray(
        [
            row["random_exploration_gate_eligible_v748"]
            for row in policy_audits
        ],
        dtype=bool,
    )
    burst_counts_v748 = np.asarray(
        [
            row["random_exploration_burst_count_v748"]
            for row in policy_audits
        ],
        dtype=np.int64,
    )
    record["relay_dual_goal_runtime_v643"] = {
        "format": "edgearm-v643-runtime-relay-episode-audit-v1",
        "row_count": row_count,
        "mean_acquisition_gate": float(np.mean(gates)),
        "full_acquisition_gate_fraction": float(np.mean(gates >= 0.999)),
        "exact_transport_gate_fraction": float(np.mean(gates == 0.0)),
        "initial_tool_precontact_distance_m": float(distances[0]),
        "minimum_tool_precontact_distance_m": float(np.min(distances)),
        "final_tool_precontact_distance_m": float(distances[-1]),
        "best_tool_precontact_progress_m": float(distances[0] - np.min(distances)),
        "net_tool_precontact_progress_m": float(distances[0] - distances[-1]),
        "random_exploration_fraction": float(np.mean(random_rows)),
        "random_exploration_gate_eligible_fraction_v748": float(
            np.mean(exploration_gate_eligible_rows_v748)
        ),
        "random_exploration_burst_count_v748": int(
            np.max(burst_counts_v748)
        ),
        "random_exploration_burst_remaining_steps_at_end_v748": int(
            policy_audits[-1][
                "random_exploration_burst_remaining_steps_v748"
            ]
        ),
        "random_exploration_matches_blended_policy_support_v702": all(
            bool(row["random_exploration_matches_blended_policy_support_v702"]) for row in policy_audits
        ),
        "support_limited_random_exploration_step_count_v702": sum(
            int(
                row["random_exploration_selected"]
                and row["random_action_support_v702"] is not None
                and np.any(
                    np.asarray(
                        row["random_action_support_v702"],
                        dtype=np.float32,
                    )
                    < np.float32(1.0)
                )
            )
            for row in policy_audits
        ),
        "frozen_transport_at_zero_gate": True,
        "tool_goal_execution_supervision_used": False,
        "expert_calls": 0,
        "behavior_cloning_steps": 0,
        "production_admission": False,
    }
    v714_rows = [
        row for row in policy_audits if bool(row.get("contact_latched_transport_option_active_v714", False))
    ]
    if v714_rows:
        record["contact_latched_transport_mode_runtime_v714"] = {
            "format": "edgearm-v714-contact-latched-transport-option-runtime-v1",
            "row_count": row_count,
            "active_step_count": len(v714_rows),
            "first_contact_policy_call": int(v714_rows[0]["first_contact_policy_call_v714"]),
            "active_without_current_contact_step_count": sum(
                int(not bool(row["current_contact_state_v714"])) for row in v714_rows
            ),
            "post_contact_random_exploration_suppressed_step_count": sum(
                int(bool(row["post_contact_random_exploration_suppressed_v714"])) for row in v714_rows
            ),
            "learned_frozen_transport_action_selected_on_every_active_step": all(
                bool(row["learned_frozen_transport_action_selected_v714"]) for row in v714_rows
            ),
            "expert_action_used": False,
            "waypoint_or_path_used": False,
            "production_admission": False,
        }
    if dls_causal_rows:
        requested = np.stack(
            [row["requested"] for row in dls_causal_rows],
            axis=0,
        )
        predicted = np.stack(
            [row["predicted"] for row in dls_causal_rows],
            axis=0,
        )
        applied = np.stack(
            [row["applied"] for row in dls_causal_rows],
            axis=0,
        )
        contact = np.asarray(
            [row["valid_contact"] for row in dls_causal_rows],
            dtype=bool,
        )

        def cosine_rows(left: np.ndarray, right: np.ndarray) -> np.ndarray:
            denominator = np.linalg.norm(left, axis=1) * np.linalg.norm(
                right,
                axis=1,
            )
            result = np.full(len(left), np.nan, dtype=np.float64)
            valid = denominator > 1.0e-8
            result[valid] = np.sum(left[valid] * right[valid], axis=1) / (denominator[valid])
            return result

        requested_predicted_cosine = cosine_rows(requested, predicted)
        requested_applied_cosine = cosine_rows(requested, applied)

        def finite_mean_or_zero(values: np.ndarray) -> float:
            finite = np.asarray(values, dtype=np.float64)
            finite = finite[np.isfinite(finite)]
            return float(np.mean(finite)) if len(finite) else 0.0

        record["dls_request_prediction_effect_audit_v715"] = {
            "format": "edgearm-v715-dls-request-prediction-effect-audit-v1",
            "row_count": len(dls_causal_rows),
            "mean_requested_action": requested.mean(axis=0).tolist(),
            "mean_predicted_normalized_action": predicted.mean(axis=0).tolist(),
            "mean_applied_action": applied.mean(axis=0).tolist(),
            "mean_requested_to_predicted_l2": float(
                np.mean([row["requested_to_predicted_l2"] for row in dls_causal_rows])
            ),
            "p95_requested_to_predicted_l2": float(
                np.percentile(
                    [row["requested_to_predicted_l2"] for row in dls_causal_rows],
                    95.0,
                )
            ),
            "mean_predicted_to_applied_l2": float(np.mean(np.linalg.norm(predicted - applied, axis=1))),
            "mean_requested_to_applied_l2": float(np.mean(np.linalg.norm(requested - applied, axis=1))),
            "mean_requested_predicted_cosine": finite_mean_or_zero(requested_predicted_cosine),
            "mean_requested_applied_cosine": finite_mean_or_zero(requested_applied_cosine),
            "mean_absolute_predicted_lateral_action": float(np.mean(np.abs(predicted[:, 1]))),
            "mean_absolute_applied_lateral_action": float(np.mean(np.abs(applied[:, 1]))),
            "contact_mean_absolute_applied_lateral_action": float(
                np.mean(np.abs(applied[contact, 1])) if np.any(contact) else 0.0
            ),
            "joint_limit_scaled_step_fraction": float(
                np.mean(
                    np.asarray(
                        [row["joint_limit_scale"] for row in dls_causal_rows],
                        dtype=np.float64,
                    )
                    < 1.0 - 1.0e-12
                )
            ),
            "minimum_position_singular_value": float(
                np.min([row["minimum_position_singular_value"] for row in dls_causal_rows])
            ),
            "mean_orientation_residual_before_l2": float(
                np.mean([row["orientation_residual_before_l2"] for row in dls_causal_rows])
            ),
            "mean_predicted_orientation_residual_after_l2": float(
                np.mean([row["predicted_orientation_residual_after_l2"] for row in dls_causal_rows])
            ),
            "audit_changes_actions": False,
            "expert_action_used": False,
            "waypoint_or_path_used": False,
            "production_admission": False,
        }
    relative_errors = [row.get("taskframe_goal_error_features_v652") for row in policy_audits]
    if all(value is not None for value in relative_errors):
        relative = np.asarray(relative_errors, dtype=np.float32)
        record["relay_taskframe_error_runtime_v652"] = {
            "format": "edgearm-v652-runtime-taskframe-error-audit-v1",
            "row_count": row_count,
            "initial_error": relative[0].tolist(),
            "final_error": relative[-1].tolist(),
            "minimum_error_l2": float(np.min(np.linalg.norm(relative, axis=-1))),
            "final_error_l2": float(np.linalg.norm(relative[-1])),
            "feature_is_action_or_path": False,
            "production_admission": False,
        }
    record["controller_state_complete"] = True
    record["controller_state_schema_sha256"] = TASKFRAME_CONTROLLER_STATE_SCHEMA_SHA256_V614
    record["relay_dual_goal_tool_her_v643"] = True
    record["bulk_vla_data_use_allowed"] = False
    record["production_admission"] = False
    if hard_terminal_failure is not None:
        record["hard_terminal_failure_audit_v619"] = hard_terminal_failure
    if episode is not None:
        episode["controller_state"] = np.asarray(controller_rows, dtype=np.float32)
        episode["next_controller_state"] = np.asarray(next_controller_rows, dtype=np.float32)
    return record, episode


def _evaluate_v643(
    environment_config: Any,
    action_config: Any,
    scene_path: Path,
    bundle: RelayDualGoalHerSACBundleV643,
    *,
    seed_base: int,
    episodes: int,
    maximum_steps: int,
    task_independent_home_reset: bool,
) -> dict[str, Any]:
    env = RealisticEdgeArmEnvV10(environment_config, seed=seed_base, model_scene_path=scene_path)
    if task_independent_home_reset:
        adapter = StockGripperHomeTaskFrameAdapterV597(env, action_config)
        reset = reset_stock_home_taskframe_episode_v597
    else:
        adapter = StockGripperTaskFrameAdapterV22(env, action_config)
        reset = reset_stock_taskframe_episode_v22
    records: list[dict[str, Any]] = []
    for index in range(episodes):
        record, _episode = _run_episode_v643(
            env,
            adapter,
            bundle,
            requested_seed=seed_base + index,
            maximum_steps=maximum_steps,
            deterministic=True,
            random_action_probability=0.0,
            action_seed=seed_base ^ (index + 0x643E),
            collect_replay=False,
            episode_reset=reset,
        )
        records.append(record)
    successes = sum(int(row["strict_success"]) for row in records)
    contacts = sum(int(int(row["valid_contact_steps"]) > 0) for row in records)
    acquisition_progress = [
        float(row["relay_dual_goal_runtime_v643"]["best_tool_precontact_progress_m"]) for row in records
    ]
    evaluation = {
        "format": RELAY_DUAL_GOAL_EVALUATION_FORMAT_V643,
        "created_at_utc": _utc_now(),
        "seed_base": seed_base,
        "episode_count": episodes,
        "strict_success_count": successes,
        "strict_success_rate": successes / episodes,
        "contact_episode_count": contacts,
        "contact_episode_rate": contacts / episodes,
        "mean_best_tool_precontact_progress_m": float(np.mean(acquisition_progress)),
        "mean_net_target_progress_m": float(np.mean([row["net_target_progress_m"] for row in records])),
        "mean_ik_failure_steps": float(np.mean([row["ik_failure_steps"] for row in records])),
        "total_ik_failure_steps": int(sum(int(row["ik_failure_steps"]) for row in records)),
        "episodes": records,
        "controller_state_complete": True,
        "exact_three_second_environment_success_only": True,
        "task_independent_home_reset": task_independent_home_reset,
        "object_target_stage": 32,
        "object_target_distance_range_m": [0.17, 0.19],
        "initial_target_overlap_allowed": False,
        "her_successes_in_numerator": 0,
        "production_admission": False,
    }
    evaluation["automatic_failure_cohort_audit_v619"] = _failure_cohort_audit_v619(records)
    return evaluation


def _checkpoint_payload_v643(
    bundle: RelayDualGoalHerSACBundleV643,
    replay: GoalConditionedMarkovHerReplayV614,
    *,
    run_plan_sha256: str,
    parent_v614_checkpoint_sha256: str,
    parent_replay_episode_count: int,
    phase: str,
    relay_online_episode_index: int,
    upgrade_audit: dict[str, Any],
) -> dict[str, Any]:
    return {
        "format": RELAY_DUAL_GOAL_CHECKPOINT_FORMAT_V643,
        "algorithm_format": RELAY_DUAL_GOAL_HER_SAC_FORMAT_V643,
        "created_at_utc": _utc_now(),
        "phase": phase,
        "relay_online_episode_index": relay_online_episode_index,
        "parent_replay_episode_count": parent_replay_episode_count,
        "update_index": bundle.update_index,
        "config": asdict(bundle.config),
        "phase_isolated_acquisition_v626": asdict(bundle.policy.phase_config),
        "run_plan_sha256": run_plan_sha256,
        "parent_v614_checkpoint_sha256": parent_v614_checkpoint_sha256,
        "upgrade_audit": upgrade_audit,
        "policy_state_dict": bundle.policy.state_dict(),
        "acquisition_critic_state_dict": (bundle.acquisition_critic.state_dict()),
        "target_acquisition_critic_state_dict": (bundle.target_acquisition_critic.state_dict()),
        "frozen_feasibility_state_dict": (bundle.frozen_feasibility.state_dict()),
        "actor_optimizer_state_dict": bundle.actor_optimizer.state_dict(),
        "critic_optimizer_state_dict": bundle.critic_optimizer.state_dict(),
        "replay_manifest": replay.manifest(),
        "controller_state_schema_sha256": (TASKFRAME_CONTROLLER_STATE_SCHEMA_SHA256_V614),
        "simulator_privileged_actor": True,
        "deployable_visual_policy": False,
        "expert_calls": 0,
        "behavior_cloning_steps": 0,
        "production_admission": False,
    }


def _restore_checkpoint_v643(
    bundle: RelayDualGoalHerSACBundleV643,
    payload: dict[str, Any],
    *,
    parent_v614_checkpoint_sha256: str,
) -> None:
    if (
        type(payload) is not dict
        or payload.get("format") != RELAY_DUAL_GOAL_CHECKPOINT_FORMAT_V643
        or payload.get("algorithm_format") != RELAY_DUAL_GOAL_HER_SAC_FORMAT_V643
        or payload.get("controller_state_schema_sha256") != TASKFRAME_CONTROLLER_STATE_SCHEMA_SHA256_V614
        or payload.get("parent_v614_checkpoint_sha256") != parent_v614_checkpoint_sha256
        or RelayDualGoalHerSACConfigV643(**payload["config"]) != bundle.config
        or PhaseIsolatedAcquisitionConfigV626(**payload["phase_isolated_acquisition_v626"])
        != bundle.policy.phase_config
        or payload.get("production_admission") is not False
    ):
        raise ValueError("V643 resume checkpoint identity changed")
    frozen_parent = {
        name: value.detach().cpu().clone()
        for name, value in bundle.policy.transport_actor.state_dict().items()
    }
    bundle.policy.load_state_dict(payload["policy_state_dict"], strict=True)
    if any(
        not torch.equal(value.detach().cpu(), frozen_parent[name])
        for name, value in bundle.policy.transport_actor.state_dict().items()
    ):
        raise ValueError("V643 resume changed the frozen transport actor")
    bundle.acquisition_critic.load_state_dict(payload["acquisition_critic_state_dict"], strict=True)
    bundle.target_acquisition_critic.load_state_dict(
        payload["target_acquisition_critic_state_dict"], strict=True
    )
    bundle.frozen_feasibility.load_state_dict(payload["frozen_feasibility_state_dict"], strict=True)
    bundle.actor_optimizer.load_state_dict(payload["actor_optimizer_state_dict"])
    bundle.critic_optimizer.load_state_dict(payload["critic_optimizer_state_dict"])
    bundle.update_index = int(payload["update_index"])


def run_relay_dual_goal_her_sac_v643(
    *,
    output_dir: Path,
    collection_plan: Path,
    parent_v614_checkpoint: Path,
    replay_npz: Path,
    resume_v643_checkpoint: Path | None,
    initialization_seed: int,
    sampling_seed_base: int,
    online_seed_base: int,
    evaluation_seed_base: int,
    offline_updates: int,
    relay_online_episodes: int,
    updates_per_episode: int,
    evaluation_every_episodes: int,
    evaluation_episodes: int,
    maximum_episode_steps: int,
    initial_random_action_probability: float,
    final_random_action_probability: float,
    device: str,
    config: RelayDualGoalHerSACConfigV643 | None = None,
    colored_random_exploration: (AxisScaledColoredExplorationConfigV605 | None) = None,
) -> dict[str, Any]:
    for name, value, minimum in (
        ("offline_updates", offline_updates, 0),
        ("relay_online_episodes", relay_online_episodes, 1),
        ("updates_per_episode", updates_per_episode, 1),
        ("evaluation_every_episodes", evaluation_every_episodes, 1),
        ("evaluation_episodes", evaluation_episodes, 1),
        ("maximum_episode_steps", maximum_episode_steps, 1),
    ):
        if type(value) is not int or value < minimum:
            raise ValueError(f"V643 {name} is invalid")
    probabilities = (
        initial_random_action_probability,
        final_random_action_probability,
    )
    if any(not np.isfinite(value) or not 0.0 <= value <= 1.0 for value in probabilities):
        raise ValueError("V643 exploration probability is invalid")
    destination = Path(output_dir).expanduser().resolve()
    if destination.exists():
        raise FileExistsError(f"V643 output already exists: {destination}")
    parent_path = Path(parent_v614_checkpoint).expanduser().resolve()
    replay_path = Path(replay_npz).expanduser().resolve()
    if not parent_path.is_file() or not replay_path.is_file():
        raise FileNotFoundError("V643 parent checkpoint or replay is missing")
    parent_sha256 = sha256_file_v1(parent_path)
    parent_payload = torch.load(parent_path, map_location="cpu", weights_only=True)
    if (
        type(parent_payload) is not dict
        or parent_payload.get("format") != GOAL_CONDITIONED_MARKOV_HER_CHECKPOINT_FORMAT_V614
    ):
        raise ValueError("V643 parent is not a V614 checkpoint")
    parent_config = GoalConditionedHerSACConfigV43(**parent_payload["config"])
    phase_config = PhaseIsolatedAcquisitionConfigV626(**parent_payload["phase_isolated_acquisition_v626"])
    selected_config = config or RelayDualGoalHerSACConfigV643(
        hidden_dim=parent_config.hidden_dim,
        batch_size=parent_config.batch_size,
    )
    selected_config.validate()
    if selected_config.hidden_dim != parent_config.hidden_dim:
        raise ValueError("V643 acquisition and parent hidden dimensions disagree")
    replay = GoalConditionedMarkovHerReplayV614.load_npz(replay_path, parent_config)
    selected_device = _resolve_device(device)
    bundle, upgrade_audit = initialize_relay_dual_goal_her_sac_v643(
        seed=initialization_seed,
        device=selected_device,
        parent_v614_payload=parent_payload,
        phase_config=phase_config,
        config=selected_config,
    )
    parent_replay_episode_count = replay.episode_count
    completed_relay_episodes = 0
    resume_path = None
    if resume_v643_checkpoint is not None:
        resume_path = Path(resume_v643_checkpoint).expanduser().resolve()
        resume_payload = torch.load(resume_path, map_location="cpu", weights_only=True)
        _restore_checkpoint_v643(
            bundle,
            resume_payload,
            parent_v614_checkpoint_sha256=parent_sha256,
        )
        completed_relay_episodes = int(resume_payload["relay_online_episode_index"])
        parent_replay_episode_count = int(resume_payload["parent_replay_episode_count"])
        if replay.episode_count != (parent_replay_episode_count + completed_relay_episodes):
            raise ValueError("V643 resume checkpoint and replay disagree")
        if relay_online_episodes <= completed_relay_episodes:
            raise ValueError("V643 resume target must exceed completed episodes")
        if offline_updates:
            raise ValueError("V643 resume cannot repeat offline pretraining")

    source_plan, environment_config, action_config, scene_path = _load_collection_contract(collection_plan)
    if environment_config.reverse_curriculum_stage_v26 != 32:
        environment_config = replace(
            environment_config,
            reverse_curriculum_stage_v26=32,
            strict_success_hold_steps=90,
        )
    if environment_config.max_steps < maximum_episode_steps:
        environment_config = replace(environment_config, max_steps=maximum_episode_steps)

    source_file = Path(__file__).resolve()
    run_plan = {
        "format": RELAY_DUAL_GOAL_TRAIN_RUN_FORMAT_V643,
        "algorithm_format": RELAY_DUAL_GOAL_HER_SAC_FORMAT_V643,
        "created_at_utc": _utc_now(),
        "output_dir": str(destination),
        "resolved_device": selected_device,
        "configuration": asdict(selected_config),
        "phase_isolated_acquisition_v626": {
            "format": PHASE_ISOLATED_ACQUISITION_FORMAT_V626,
            **asdict(phase_config),
        },
        "parent_v614_checkpoint": str(parent_path),
        "parent_v614_checkpoint_sha256": parent_sha256,
        "source_replay_npz": str(replay_path),
        "source_replay_npz_sha256": sha256_file_v1(replay_path),
        "parent_replay_episode_count": parent_replay_episode_count,
        "resume_v643_checkpoint": (None if resume_path is None else str(resume_path)),
        "resume_v643_checkpoint_sha256": (None if resume_path is None else sha256_file_v1(resume_path)),
        "upgrade_audit": upgrade_audit,
        "offline_updates": offline_updates,
        "relay_online_episodes": relay_online_episodes,
        "completed_relay_episodes_before_resume": completed_relay_episodes,
        "updates_per_episode": updates_per_episode,
        "evaluation_every_episodes": evaluation_every_episodes,
        "evaluation_episodes": evaluation_episodes,
        "maximum_episode_steps": maximum_episode_steps,
        "initial_random_action_probability": initial_random_action_probability,
        "final_random_action_probability": final_random_action_probability,
        "initialization_seed": initialization_seed,
        "sampling_seed_base": sampling_seed_base,
        "online_seed_base": online_seed_base,
        "evaluation_seed_base": evaluation_seed_base,
        "source_collection_plan": str(Path(collection_plan).resolve()),
        "source_collection_plan_sha256": source_plan["run_plan_sha256"],
        "scene_path": str(scene_path),
        "scene_sha256": sha256_file_v1(scene_path),
        "environment_config": asdict(environment_config),
        "action_adapter_config": asdict(action_config),
        "execution_kernel": StockGripperRolloutKernelV22.format,
        "new_online_object_target_stage": 32,
        "new_online_object_target_distance_range_m": [0.17, 0.19],
        "new_online_reset": "exact_task_independent_home_only",
        "initial_target_overlap_allowed": False,
        "legacy_stage31_replay_use": "curriculum_learning_only",
        "legacy_stage31_export_allowed": False,
        "tool_goal_future_her_learning_only": True,
        "tool_goal_her_rotates_action": False,
        "frozen_transport_at_zero_gate": True,
        "visual_actor_training": False,
        "act_training": False,
        "bulk_wrist_multimodal_generation_started": False,
        "source_type": SOURCE_TYPE,
        "simulator_privileged_actor": True,
        "expert_calls": 0,
        "behavior_cloning_steps": 0,
        "physical_samples": 0,
        "source_hashes": {
            "edgearm/train_relay_dual_goal_her_sac_v643.py": (sha256_file_v1(source_file)),
            "edgearm/relay_dual_goal_her_sac_v643.py": sha256_file_v1(
                source_file.with_name("relay_dual_goal_her_sac_v643.py")
            ),
            "edgearm/phase_isolated_acquisition_v626.py": sha256_file_v1(
                source_file.with_name("phase_isolated_acquisition_v626.py")
            ),
        },
        "production_admission": False,
    }
    run_plan["run_plan_sha256"] = canonical_sha256_v1(run_plan)
    destination.mkdir(parents=True)
    (destination / "checkpoints").mkdir()
    (destination / "evaluations").mkdir()
    _atomic_json(destination / "run_plan.json", run_plan)

    latest_metrics: dict[str, Any] | None = None
    evaluations: list[dict[str, Any]] = []
    online_records: list[dict[str, Any]] = []
    best_checkpoint: Path | None = None
    best_evaluation: dict[str, Any] | None = None
    best_score: tuple[int, float, int, int] | None = None

    _atomic_json(
        destination / "run_state.json",
        {
            "status": "running",
            "phase": "offline_relay_pretraining",
            "relay_online_episode_index": completed_relay_episodes,
            "update_index": bundle.update_index,
            "replay_transition_count": replay.transition_count,
            "updated_at_utc": _utc_now(),
        },
    )
    try:
        for local_update in range(1, offline_updates + 1):
            batch = sample_relay_acquisition_batch_v643(
                replay,
                batch_size=selected_config.batch_size,
                seed=sampling_seed_base + bundle.update_index,
                phase_config=phase_config,
                config=selected_config,
            )
            metrics = relay_dual_goal_her_sac_update_v643(bundle, batch)
            latest_metrics = {
                **asdict(metrics),
                "phase": "offline_relay_pretraining",
                "local_update": local_update,
                "relay_online_episode_index": completed_relay_episodes,
            }
            _append_jsonl(destination / "metrics.jsonl", latest_metrics)

        pre_online_evaluation = _evaluate_v643(
            environment_config,
            action_config,
            scene_path,
            bundle,
            seed_base=evaluation_seed_base,
            episodes=evaluation_episodes,
            maximum_steps=maximum_episode_steps,
            task_independent_home_reset=True,
        )
        pre_online_evaluation["evaluation_scope_v643"] = "post_offline_pretraining_exact_home_stage32"
        pre_online_evaluation["relay_online_episode_index"] = completed_relay_episodes
        pre_online_evaluation["update_index"] = bundle.update_index
        _atomic_json(
            destination / "evaluations" / "episode_000000.json",
            pre_online_evaluation,
        )
        evaluations.append(pre_online_evaluation)

        env = RealisticEdgeArmEnvV10(
            environment_config,
            seed=online_seed_base,
            model_scene_path=scene_path,
        )
        home_adapter = StockGripperHomeTaskFrameAdapterV597(env, action_config)
        for online_index in range(completed_relay_episodes + 1, relay_online_episodes + 1):
            fraction = (online_index - 1) / max(relay_online_episodes - 1, 1)
            random_probability = initial_random_action_probability + fraction * (
                final_random_action_probability - initial_random_action_probability
            )
            record, episode = _run_episode_v643(
                env,
                home_adapter,
                bundle,
                requested_seed=online_seed_base + online_index - 1,
                maximum_steps=maximum_episode_steps,
                deterministic=False,
                random_action_probability=float(random_probability),
                action_seed=online_seed_base ^ (online_index * 0x643A11),
                collect_replay=True,
                episode_reset=reset_stock_home_taskframe_episode_v597,
                colored_random_exploration=colored_random_exploration,
            )
            if episode is None:
                raise RuntimeError("V643 online collection lost replay")
            episode["start_tier_index_v622"] = np.full(
                len(episode["terminal"]),
                _HOME_TIER_INDEX_V643,
                dtype=np.int8,
            )
            record["start_tier_v622"] = "home"
            record["start_tier_final_data_eligible_v622"] = True
            record["object_target_stage_v643"] = 32
            record["object_target_distance_full_task_v643"] = bool(
                float(record["initial_block_target_distance_m"]) >= MINIMUM_FULL_TASK_INITIAL_DISTANCE_M_V607
            )
            record["relay_online_episode_index"] = online_index
            replay.add_episode(
                episode,
                source=f"online_v643_home_stage32_episode_{online_index:06d}",
            )
            record["replay_transition_count_after_append"] = replay.transition_count
            online_records.append(record)
            _append_jsonl(destination / "online_episodes.jsonl", record)

            for local_update in range(1, updates_per_episode + 1):
                batch = sample_relay_acquisition_batch_v643(
                    replay,
                    batch_size=selected_config.batch_size,
                    seed=sampling_seed_base + bundle.update_index,
                    phase_config=phase_config,
                    config=selected_config,
                )
                metrics = relay_dual_goal_her_sac_update_v643(bundle, batch)
                latest_metrics = {
                    **asdict(metrics),
                    "phase": "online_home_stage32_training",
                    "relay_online_episode_index": online_index,
                    "local_update_after_episode": local_update,
                }
                _append_jsonl(destination / "metrics.jsonl", latest_metrics)

            replay.save_npz(destination / "replay_latest.npz")
            checkpoint = destination / "checkpoints" / f"episode_{online_index:06d}.pt"
            _atomic_torch_save(
                checkpoint,
                _checkpoint_payload_v643(
                    bundle,
                    replay,
                    run_plan_sha256=run_plan["run_plan_sha256"],
                    parent_v614_checkpoint_sha256=parent_sha256,
                    parent_replay_episode_count=parent_replay_episode_count,
                    phase="online_home_stage32_training",
                    relay_online_episode_index=online_index,
                    upgrade_audit=upgrade_audit,
                ),
            )
            if online_index % evaluation_every_episodes == 0 or online_index == relay_online_episodes:
                evaluation = _evaluate_v643(
                    environment_config,
                    action_config,
                    scene_path,
                    bundle,
                    seed_base=evaluation_seed_base,
                    episodes=evaluation_episodes,
                    maximum_steps=maximum_episode_steps,
                    task_independent_home_reset=True,
                )
                evaluation["evaluation_scope_v643"] = "exact_home_full_acquisition_to_push_stage32"
                transport = _evaluate_v643(
                    environment_config,
                    action_config,
                    scene_path,
                    bundle,
                    seed_base=evaluation_seed_base,
                    episodes=evaluation_episodes,
                    maximum_steps=maximum_episode_steps,
                    task_independent_home_reset=False,
                )
                transport["evaluation_scope_v643"] = (
                    "privileged_precontact_frozen_transport_retention_stage32"
                )
                evaluation["transport_retention_evaluation_v643"] = transport
                evaluation["relay_online_episode_index"] = online_index
                evaluation["update_index"] = bundle.update_index
                _atomic_json(
                    destination / "evaluations" / f"episode_{online_index:06d}.json",
                    evaluation,
                )
                evaluations.append(evaluation)
                score = _training_selection_score_v610(evaluation)
                if best_score is None or score > best_score:
                    best_score = score
                    best_checkpoint = checkpoint
                    best_evaluation = evaluation
                    _atomic_json(
                        destination / "best_training_selection.json",
                        {
                            "format": "edgearm-v643-relay-checkpoint-selection-v1",
                            "selection_score": list(score),
                            "checkpoint": str(checkpoint),
                            "checkpoint_sha256": sha256_file_v1(checkpoint),
                            "evaluation": evaluation,
                            "production_admission": False,
                        },
                    )
            _atomic_json(
                destination / "run_state.json",
                {
                    "status": "running",
                    "phase": "online_home_stage32_training",
                    "relay_online_episode_index": online_index,
                    "relay_online_episodes": relay_online_episodes,
                    "update_index": bundle.update_index,
                    "latest_online_episode": record,
                    "latest_metrics": latest_metrics,
                    "latest_evaluation": evaluations[-1],
                    "replay_transition_count": replay.transition_count,
                    "replay_episode_count": replay.episode_count,
                    "updated_at_utc": _utc_now(),
                },
            )
    except Exception as error:
        failure = {
            "format": RELAY_DUAL_GOAL_TRAIN_RUN_FORMAT_V643,
            "status": "failed",
            "error_type": type(error).__name__,
            "error": str(error),
            "relay_online_episode_index": len(online_records) + completed_relay_episodes,
            "update_index": bundle.update_index,
            "replay_manifest": replay.manifest(),
            "failed_at_utc": _utc_now(),
            "production_admission": False,
        }
        _atomic_json(destination / "failure.json", failure)
        _atomic_json(destination / "run_state.json", failure)
        raise

    final_checkpoint = destination / "checkpoints" / "final.pt"
    _atomic_torch_save(
        final_checkpoint,
        _checkpoint_payload_v643(
            bundle,
            replay,
            run_plan_sha256=run_plan["run_plan_sha256"],
            parent_v614_checkpoint_sha256=parent_sha256,
            parent_replay_episode_count=parent_replay_episode_count,
            phase="complete",
            relay_online_episode_index=relay_online_episodes,
            upgrade_audit=upgrade_audit,
        ),
    )
    replay.save_npz(destination / "replay_final.npz")
    final_evaluation = evaluations[-1]
    success_count = int(final_evaluation["strict_success_count"])
    evaluation_count = int(final_evaluation["episode_count"])
    wilson = wilson_lower_bound_v26(success_count, evaluation_count)
    home_gate = bool(
        evaluation_count >= 48
        and float(final_evaluation["strict_success_rate"]) >= 0.80
        and wilson >= 0.65
        and all(
            row["task_aligned_privileged_reset"] is False
            and row["task_independent_final_home_reset"] is True
            and float(row["initial_target_coverage"]) == 0.0
            and float(row["initial_block_target_distance_m"]) >= MINIMUM_FULL_TASK_INITIAL_DISTANCE_M_V607
            and int(row["invalid_contact_steps"]) == 0
            and int(row["safety_steps"]) == 0
            and (
                not bool(row["strict_success"])
                or float(row["net_target_progress_m"]) >= MINIMUM_FULL_TASK_NET_PROGRESS_M_V607
            )
            for row in final_evaluation["episodes"]
        )
    )
    summary = {
        "format": RELAY_DUAL_GOAL_TRAIN_RUN_FORMAT_V643,
        "status": "complete",
        "completed_at_utc": _utc_now(),
        "relay_online_episode_count": relay_online_episodes,
        "update_index": bundle.update_index,
        "replay_manifest": replay.manifest(),
        "final_checkpoint": str(final_checkpoint),
        "final_checkpoint_sha256": sha256_file_v1(final_checkpoint),
        "best_training_checkpoint": (None if best_checkpoint is None else str(best_checkpoint)),
        "best_training_evaluation": best_evaluation,
        "final_evaluation": final_evaluation,
        "online_failure_cohort_audit_v619": _failure_cohort_audit_v619(online_records),
        "strict_success_count": success_count,
        "strict_success_rate": float(final_evaluation["strict_success_rate"]),
        "home_contact_episode_count": int(final_evaluation["contact_episode_count"]),
        "home_contact_episode_rate": float(final_evaluation["contact_episode_rate"]),
        "home_stage32_strict_gate_passed": home_gate,
        "home_stage32_wilson_lower_bound": wilson,
        "successful_rl_data_generation_ready": home_gate,
        "legacy_stage31_replay_export_allowed": False,
        "bulk_multimodal_generation_started": False,
        "remaining_gate": (
            "passed exact-Home stage32 strict gate; wrist collection may be planned"
            if home_gate
            else (
                "master exact-Home acquisition and strict transport on randomized "
                "17-19 cm stage32 tasks, then pass 48-task 80%/Wilson-0.65 gate"
            )
        ),
        "expert_calls": 0,
        "behavior_cloning_steps": 0,
        "physical_samples": 0,
        "production_admission": False,
    }
    _atomic_json(destination / "summary.json", summary)
    _atomic_json(
        destination / "run_state.json",
        {
            "status": "complete",
            "phase": "complete",
            "relay_online_episode_index": relay_online_episodes,
            "update_index": bundle.update_index,
            "strict_success_count": success_count,
            "strict_success_rate": summary["strict_success_rate"],
            "home_contact_episode_count": summary["home_contact_episode_count"],
            "replay_transition_count": replay.transition_count,
            "updated_at_utc": _utc_now(),
        },
    )
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--collection-plan", type=Path, required=True)
    parser.add_argument("--parent-v614-checkpoint", type=Path, required=True)
    parser.add_argument("--replay-npz", type=Path, required=True)
    parser.add_argument("--resume-v643-checkpoint", type=Path)
    parser.add_argument("--initialization-seed", type=int, default=643_000_000)
    parser.add_argument("--sampling-seed-base", type=int, default=643_100_000)
    parser.add_argument("--online-seed-base", type=int, default=643_200_000)
    parser.add_argument("--evaluation-seed-base", type=int, default=643_300_000)
    parser.add_argument("--offline-updates", type=int, default=512)
    parser.add_argument("--relay-online-episodes", type=int, default=8)
    parser.add_argument("--updates-per-episode", type=int, default=64)
    parser.add_argument("--evaluation-every-episodes", type=int, default=2)
    parser.add_argument("--evaluation-episodes", type=int, default=4)
    parser.add_argument("--maximum-episode-steps", type=int, default=480)
    parser.add_argument("--initial-random-action-probability", type=float, default=0.12)
    parser.add_argument("--final-random-action-probability", type=float, default=0.03)
    parser.add_argument(
        "--batch-size",
        type=int,
        help="Defaults to the parent checkpoint batch size.",
    )
    parser.add_argument(
        "--hidden-dim",
        type=int,
        help="Defaults to the parent checkpoint hidden dimension.",
    )
    parser.add_argument("--future-tool-goal-probability", type=float, default=0.50)
    parser.add_argument("--home-sampling-fraction", type=float, default=0.55)
    parser.add_argument("--colored-random-exploration", action="store_true")
    parser.add_argument("--colored-exploration-rho", type=float, default=0.94)
    parser.add_argument(
        "--colored-exploration-standard-deviation",
        type=float,
        nargs=3,
        default=(0.45, 0.28, 0.18),
    )
    parser.add_argument("--device", choices=("auto", "cpu", "mps"), default="auto")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    parent_payload = torch.load(
        Path(args.parent_v614_checkpoint).expanduser().resolve(),
        map_location="cpu",
        weights_only=True,
    )
    if (
        type(parent_payload) is not dict
        or parent_payload.get("format") != GOAL_CONDITIONED_MARKOV_HER_CHECKPOINT_FORMAT_V614
    ):
        raise ValueError("V643 parent is not a V614 checkpoint")
    parent_config = GoalConditionedHerSACConfigV43(**parent_payload["config"])
    config = RelayDualGoalHerSACConfigV643(
        hidden_dim=(parent_config.hidden_dim if args.hidden_dim is None else args.hidden_dim),
        batch_size=(parent_config.batch_size if args.batch_size is None else args.batch_size),
        future_tool_goal_probability=args.future_tool_goal_probability,
        home_sampling_fraction=args.home_sampling_fraction,
    )
    colored = (
        AxisScaledColoredExplorationConfigV605(
            standard_deviation=tuple(args.colored_exploration_standard_deviation),
            autoregressive_rho=args.colored_exploration_rho,
        )
        if args.colored_random_exploration
        else None
    )
    summary = run_relay_dual_goal_her_sac_v643(
        output_dir=args.output_dir,
        collection_plan=args.collection_plan,
        parent_v614_checkpoint=args.parent_v614_checkpoint,
        replay_npz=args.replay_npz,
        resume_v643_checkpoint=args.resume_v643_checkpoint,
        initialization_seed=args.initialization_seed,
        sampling_seed_base=args.sampling_seed_base,
        online_seed_base=args.online_seed_base,
        evaluation_seed_base=args.evaluation_seed_base,
        offline_updates=args.offline_updates,
        relay_online_episodes=args.relay_online_episodes,
        updates_per_episode=args.updates_per_episode,
        evaluation_every_episodes=args.evaluation_every_episodes,
        evaluation_episodes=args.evaluation_episodes,
        maximum_episode_steps=args.maximum_episode_steps,
        initial_random_action_probability=(args.initial_random_action_probability),
        final_random_action_probability=args.final_random_action_probability,
        device=args.device,
        config=config,
        colored_random_exploration=colored,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
