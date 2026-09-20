"""Train the V614 controller-state-complete HER-SAC RL data generator.

This runner accepts a frozen V43 checkpoint only as a zero-residual warm
start.  Every V614 update uses freshly collected transitions that include the
exact persistent task-frame controller state.  Curriculum rollouts remain RL
replay only; final wrist/VLA generation stays fail-closed.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, replace
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from .adaptive_start_curriculum_v622 import (
    ADAPTIVE_START_CURRICULUM_FORMAT_V622,
    START_TIERS_V622,
    curriculum_cohort_audit_v622,
    select_start_tier_v622,
)
from .asymmetric_multiview_ppo_v1 import (
    SOURCE_TYPE,
    canonical_sha256_v1,
    sha256_file_v1,
)
from .goal_conditioned_her_sac_v43 import (
    GoalConditionedHerSACConfigV43,
    goal_neutral_privileged_state_v43,
    observation_with_goal_v43,
)
from .goal_conditioned_markov_her_sac_v614 import (
    GOAL_CONDITIONED_MARKOV_HER_CHECKPOINT_FORMAT_V614,
    GOAL_CONDITIONED_MARKOV_HER_SAC_FORMAT_V614,
    GoalConditionedMarkovHerReplayV614,
    GoalConditionedMarkovHerSACBundleV614,
    goal_conditioned_markov_her_sac_update_v614,
    initialize_goal_conditioned_markov_her_sac_v614,
    warm_start_markov_her_sac_v614_from_v43,
)
from .interpolated_approach_reset_v622 import (
    StockGripperApproachTaskFrameAdapterV622,
    reset_stock_interpolated_approach_episode_v622,
)
from .markov_actor_trust_region_v621 import (
    MARKOV_ACTOR_TRUST_REGION_FORMAT_V621,
    MarkovActorTrustRegionConfigV621,
)
from .phase_isolated_acquisition_v626 import (
    PHASE_ISOLATED_ACQUISITION_FORMAT_V626,
    PhaseIsolatedAcquisitionConfigV626,
)
from .phase_isolated_acquisition_option_v639 import (
    PHASE_ISOLATED_ACQUISITION_OPTION_FORMAT_V639,
    PhaseIsolatedAcquisitionOptionConfigV639,
)
from .reverse_curriculum_v26 import wilson_lower_bound_v26
from .sim2real_env_v10 import RealisticEdgeArmEnvV10
from .stock_gripper_rollout_kernel_v22 import StockGripperRolloutKernelV22
from .stock_gripper_taskframe_v22 import (
    StockGripperTaskFrameAdapterV22,
    reset_stock_taskframe_episode_v22,
)
from .task_independent_home_reset_v597 import (
    HOME_ACQUISITION_TRANSPORT_ALIGNMENT_V597,
    HOME_ACQUISITION_TRANSPORT_HEIGHT_MARGIN_M_V597,
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


GOAL_CONDITIONED_MARKOV_TRAIN_RUN_FORMAT_V614 = (
    "edgearm-v614-controller-state-complete-online-training-run-v1"
)
GOAL_CONDITIONED_MARKOV_EVALUATION_FORMAT_V614 = (
    "edgearm-v614-controller-state-complete-heldout-evaluation-v1"
)


def _compact_hard_terminal_failure_v619(
    transition: dict[str, Any],
) -> dict[str, Any]:
    """Preserve causal shield evidence without copying huge forecast traces."""

    shield = transition.get("shield_failure_audit")
    guard = transition.get("execution_guard_audit") or {}
    recovery: dict[str, Any] = {}
    exception_type = None
    exception_message = None
    if isinstance(shield, dict):
        exception_type = shield.get("exception_type")
        exception_message = shield.get("exception_message")
        if isinstance(shield.get("last_guard_report"), dict):
            guard = shield["last_guard_report"]
        if isinstance(shield.get("last_recovery_report"), dict):
            recovery = shield["last_recovery_report"]
    attempts = list(guard.get("attempts", ())) if isinstance(guard, dict) else []
    task_candidate_rows: list[dict[str, Any]] = []
    if isinstance(guard, dict):
        task_candidate_rows.extend(
            row
            for row in guard.get("candidate_preflights", ())
            if isinstance(row, dict)
        )
        original_v689 = guard.get("original_v689_failure")
        if isinstance(original_v689, dict):
            task_candidate_rows.extend(
                row
                for row in original_v689.get("candidate_preflights", ())
                if isinstance(row, dict)
            )
        for lattice in guard.get("lattice_candidate_attempts", ()):
            if not isinstance(lattice, dict):
                continue
            if lattice.get("safe_motion_found") is False:
                task_candidate_rows.append(
                    {
                        "task_action_scale": None,
                        "safe_candidate_found": False,
                        "selected_is_baseline_hold": False,
                        "usable": False,
                        "exception_type": lattice.get("exception_type"),
                        "lattice_candidate_index": lattice.get("index"),
                    }
                )
    compact_attempts: list[dict[str, Any]] = []
    limiting_reasons: set[str] = set()
    for attempt in attempts[:8]:
        if not isinstance(attempt, dict):
            continue
        one_step_reasons = [
            str(value) for value in attempt.get("one_step_failure_reasons", ())
        ]
        if attempt.get("planning_margin_valid") is False:
            limiting_reasons.add("planning_margin_invalid")
        if attempt.get("forecast_infeasible") is True:
            limiting_reasons.add("forecast_infeasible")
        limiting_reasons.update(one_step_reasons)
        forecast_error = attempt.get("forecast_error")
        if forecast_error not in (None, "", "none"):
            limiting_reasons.add(f"forecast_error:{forecast_error}")
        compact_attempts.append(
            {
                "scale": attempt.get("scale"),
                "candidate_is_baseline_hold": attempt.get(
                    "candidate_is_baseline_hold"
                ),
                "forecast_infeasible": attempt.get("forecast_infeasible"),
                "forecast_error": forecast_error,
                "planning_margin_valid": attempt.get(
                    "planning_margin_valid"
                ),
                "minimum_one_step_clearance_m": attempt.get(
                    "minimum_one_step_clearance_m"
                ),
                "minimum_braking_clearance_m": attempt.get(
                    "minimum_braking_clearance_m"
                ),
                "one_step_failure_reasons": one_step_reasons,
            }
        )
    compact_task_candidates: list[dict[str, Any]] = []
    for row in task_candidate_rows[:16]:
        safe_candidate = row.get("safe_candidate_found")
        baseline_hold = bool(row.get("selected_is_baseline_hold", False))
        usable = bool(row.get("usable", False))
        if safe_candidate is False:
            limiting_reasons.add("task_action_preflight_rejected")
        if baseline_hold and not usable:
            limiting_reasons.add("verified_baseline_hold_only")
        if safe_candidate is True and not usable:
            limiting_reasons.add("task_action_motion_unusable")
        compact_task_candidates.append(
            {
                "task_action_scale": row.get("task_action_scale"),
                "safe_candidate_found": safe_candidate,
                "guard_selected_scale": row.get("guard_selected_scale"),
                "selected_is_baseline_hold": baseline_hold,
                "usable": usable,
                "exception_type": row.get("exception_type"),
                "lattice_candidate_index": row.get(
                    "lattice_candidate_index"
                ),
            }
        )
    if isinstance(guard, dict) and guard.get("selected_kind") == (
        "bounded_safe_backup_exhausted"
    ):
        limiting_reasons.add("bounded_safe_backup_exhausted")
    rejected = list(recovery.get("rejected_candidates", ()))
    if rejected:
        limiting_reasons.add("interior_recovery_candidates_rejected")
    if not limiting_reasons:
        limiting_reasons.add("no_guard_safe_candidate_unspecified")
    translation = transition.get("translation_audit") or {}
    return {
        "format": "edgearm-v619-compact-hard-terminal-failure-audit-v1",
        "episode_step": int(transition["episode_step"]),
        "terminal_reason": str(transition["terminal_reason"]),
        "exception_type": exception_type,
        "exception_message": exception_message,
        "selected_action": np.asarray(
            transition["selected_action"], dtype=np.float32
        ).tolist(),
        "applied_action": np.asarray(
            transition["applied_action"], dtype=np.float32
        ).tolist(),
        "translation_failure_reason": translation.get("failure_reason"),
        "translation_face_label": translation.get("face_label"),
        "guard_attempt_count": len(attempts),
        "guard_attempts_truncated": len(attempts) > len(compact_attempts),
        "guard_attempts": compact_attempts,
        "task_action_candidate_count": len(task_candidate_rows),
        "task_action_candidates_truncated": (
            len(task_candidate_rows) > len(compact_task_candidates)
        ),
        "task_action_candidates": compact_task_candidates,
        "recovery_rejected_candidate_count": len(rejected),
        "limiting_reasons": sorted(limiting_reasons),
        "production_admission": False,
    }


def _failure_cohort_audit_v619(
    records: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    """Turn per-episode failure evidence into one deterministic next focus."""

    phases = Counter(
        str(row.get("automatic_failure_audit_v604", {}).get(
            "first_unmet_phase", "missing_audit"
        ))
        for row in records
        if not bool(row.get("strict_success", False))
    )
    terminals = Counter(
        str(row.get("terminal_reason", "missing"))
        for row in records
        if not bool(row.get("strict_success", False))
    )
    no_contact = sum(
        int(not row.get("strict_success", False) and int(row.get("valid_contact_steps", 0)) == 0)
        for row in records
    )
    safety_terminal = terminals.get("v22_action_shield_terminal", 0)
    if safety_terminal:
        next_focus = "stabilize_goal_entry_and_settle_guard"
    elif no_contact:
        next_focus = "learn_home_to_contact_acquisition"
    elif phases.get("partial_target_entry_below_strict_coverage", 0):
        next_focus = "improve_contact_transport_to_full_coverage"
    elif phases.get("productive_transport_did_not_enter_target", 0):
        next_focus = "improve_productive_transport_distance"
    else:
        next_focus = "inspect_remaining_failure_cohort"
    hard_terminal_count = sum(
        int(row.get("terminal_reason") == "v22_action_shield_terminal")
        for row in records
    )
    hard_terminal_evidence = sum(
        int("hard_terminal_failure_audit_v619" in row) for row in records
    )
    return {
        "format": "edgearm-v619-automatic-failure-cohort-audit-v1",
        "episode_count": len(records),
        "strict_success_count": sum(
            int(bool(row.get("strict_success", False))) for row in records
        ),
        "failure_count": sum(
            int(not bool(row.get("strict_success", False))) for row in records
        ),
        "first_unmet_phase_counts": dict(sorted(phases.items())),
        "terminal_reason_counts": dict(sorted(terminals.items())),
        "no_valid_contact_failure_count": no_contact,
        "hard_terminal_count": hard_terminal_count,
        "hard_terminal_evidence_count": hard_terminal_evidence,
        "all_hard_terminals_have_compact_evidence": (
            hard_terminal_count == hard_terminal_evidence
        ),
        "recommended_next_focus": next_focus,
        "production_admission": False,
    }


def _curriculum_records_from_replay_v634(
    replay: GoalConditionedMarkovHerReplayV614,
) -> list[dict[str, Any]]:
    """Recover the exact evidence needed to continue a tiered curriculum."""

    records: list[dict[str, Any]] = []
    for source in replay.source_rows:
        start = int(source.get("start_row", -1))
        count = int(source.get("row_count", -1))
        stop = start + count
        if (
            start < 0
            or count < 1
            or stop > replay.transition_count
        ):
            raise ValueError("V634 adaptive resume replay row bounds changed")
        tier_rows = replay.start_tier_index_v622[start:stop]
        if (
            tier_rows.shape != (count,)
            or np.any(tier_rows < 0)
            or not np.all(tier_rows == tier_rows[0])
        ):
            raise ValueError(
                "V634 adaptive resume replay lacks one exact start tier"
            )
        tier = START_TIERS_V622[int(tier_rows[0])].code
        stored_tier = source.get("start_tier_v622")
        if stored_tier is not None and stored_tier != tier:
            raise ValueError(
                "V634 adaptive resume source and transition tiers disagree"
            )
        contact_steps = int(
            np.count_nonzero(replay.arrays["valid_contact"][start:stop])
        )
        strict_success = bool(source.get("strict_success", False))
        records.append(
            {
                "online_episode_index": int(
                    source.get("episode_index", len(records))
                )
                + 1,
                "start_tier_v622": tier,
                "strict_success": strict_success,
                "valid_contact_steps": contact_steps,
                "failure_terminal": bool(
                    source.get("failure_terminal", False)
                ),
                "terminal_reason": "resume_reconstructed_from_exact_replay",
                "automatic_failure_audit_v604": {
                    "first_unmet_phase": (
                        "strict_success"
                        if strict_success
                        else "resume_reconstructed_no_full_episode_audit"
                    )
                },
                "resume_reconstructed_from_exact_replay_v634": True,
                "production_admission": False,
            }
        )
    if len(records) != replay.episode_count:
        raise RuntimeError("V634 adaptive resume episode count drifted")
    curriculum_cohort_audit_v622(records)
    return records


def _matched_baseline_comparison_v616(
    candidate: dict[str, Any],
    baseline: dict[str, Any],
) -> dict[str, Any]:
    """Compare exact held-out task identities, never unmatched aggregates."""

    candidate_rows = list(candidate.get("episodes", ()))
    baseline_rows = list(baseline.get("episodes", ()))
    candidate_seeds = [int(row["requested_seed"]) for row in candidate_rows]
    baseline_seeds = [int(row["requested_seed"]) for row in baseline_rows]
    if (
        not candidate_rows
        or len(candidate_rows) != len(baseline_rows)
        or candidate_seeds != baseline_seeds
        or int(candidate.get("seed_base", -1))
        != int(baseline.get("seed_base", -2))
    ):
        raise ValueError("V616 matched baseline task identities disagree")
    candidate_ik = sum(int(row["ik_failure_steps"]) for row in candidate_rows)
    baseline_ik = sum(int(row["ik_failure_steps"]) for row in baseline_rows)
    return {
        "format": "edgearm-v616-v614-v43-matched-heldout-comparison-v1",
        "requested_seeds": candidate_seeds,
        "strict_success_count_delta": int(candidate["strict_success_count"])
        - int(baseline["strict_success_count"]),
        "strict_success_rate_delta": float(candidate["strict_success_rate"])
        - float(baseline["strict_success_rate"]),
        "mean_net_target_progress_m_delta": float(
            candidate["mean_net_target_progress_m"]
        )
        - float(baseline["mean_net_target_progress_m"]),
        "total_ik_failure_steps_delta": candidate_ik - baseline_ik,
        "candidate_total_ik_failure_steps": candidate_ik,
        "baseline_total_ik_failure_steps": baseline_ik,
        "production_admission": False,
    }


def _policy_action_v614(
    *,
    bundle: GoalConditionedMarkovHerSACBundleV614,
    privileged_state: np.ndarray,
    desired_goal: np.ndarray,
    adapter: StockGripperTaskFrameAdapterV22,
    deterministic: bool,
    rng: np.random.Generator,
    random_action_probability: float,
    random_explorer: AxisScaledColoredExplorerV605 | None,
) -> np.ndarray:
    if not deterministic and rng.random() < random_action_probability:
        if random_explorer is not None:
            return random_explorer.sample()
        return rng.uniform(-1.0, 1.0, size=3).astype(np.float32)
    observation = observation_with_goal_v43(
        goal_neutral_privileged_state_v43(privileged_state),
        np.asarray(desired_goal, dtype=np.float32),
    )
    controller_state = build_taskframe_controller_state_v614(adapter)
    device = next(bundle.actor.parameters()).device
    bundle.actor.eval()
    with torch.no_grad():
        action, _log_probability = bundle.actor.sample(
            torch.from_numpy(observation).to(device).unsqueeze(0),
            torch.from_numpy(controller_state).to(device).unsqueeze(0),
            deterministic=deterministic,
        )
    return action.squeeze(0).cpu().numpy().astype(np.float32)


def _run_episode_v614(
    env: RealisticEdgeArmEnvV10,
    adapter: StockGripperTaskFrameAdapterV22,
    bundle: GoalConditionedMarkovHerSACBundleV614,
    **kwargs: Any,
) -> tuple[dict[str, Any], dict[str, np.ndarray] | None]:
    """Run the exact V43 plant loop while binding controller state by row."""

    controller_rows: list[np.ndarray] = []
    next_controller_rows: list[np.ndarray] = []
    hard_terminal_failure: dict[str, Any] | None = None

    def pre_action(
        _env: RealisticEdgeArmEnvV10,
        _row: dict[str, Any],
    ) -> None:
        controller_rows.append(
            build_taskframe_controller_state_v614(adapter)
        )

    def transition(_row: dict[str, Any]) -> None:
        nonlocal hard_terminal_failure
        next_controller_rows.append(
            build_taskframe_controller_state_v614(adapter)
        )
        if bool(_row.get("terminal", False)) and (
            bool(_row.get("failure_terminal", False))
            or bool(_row.get("safety_violation", False))
        ):
            hard_terminal_failure = _compact_hard_terminal_failure_v619(
                _row
            )

    record, episode = _run_episode_v43(
        env,
        adapter,
        bundle,  # type: ignore[arg-type]
        policy_action_callback=_policy_action_v614,
        pre_action_observation_callback=pre_action,
        transition_audit_callback=transition,
        **kwargs,
    )
    if len(controller_rows) != int(record["rows"]) or len(
        next_controller_rows
    ) != int(record["rows"]):
        raise RuntimeError("V614 controller-state row binding drifted")
    record["controller_state_complete"] = True
    record["controller_state_schema_sha256"] = (
        TASKFRAME_CONTROLLER_STATE_SCHEMA_SHA256_V614
    )
    record["v43_hidden_controller_state_omission_fixed"] = True
    if hard_terminal_failure is not None:
        record["hard_terminal_failure_audit_v619"] = hard_terminal_failure
    record["bulk_vla_data_use_allowed"] = False
    record["production_admission"] = False
    if episode is not None:
        episode["controller_state"] = np.asarray(
            controller_rows, dtype=np.float32
        )
        episode["next_controller_state"] = np.asarray(
            next_controller_rows, dtype=np.float32
        )
    return record, episode


def _evaluate_v614(
    environment_config: Any,
    action_config: Any,
    scene_path: Path,
    bundle: GoalConditionedMarkovHerSACBundleV614,
    *,
    seed_base: int,
    episodes: int,
    maximum_steps: int,
    task_independent_home_reset: bool,
) -> dict[str, Any]:
    env = RealisticEdgeArmEnvV10(
        environment_config,
        seed=seed_base,
        model_scene_path=scene_path,
    )
    if task_independent_home_reset:
        adapter = StockGripperHomeTaskFrameAdapterV597(env, action_config)
        episode_reset = reset_stock_home_taskframe_episode_v597
    else:
        adapter = StockGripperTaskFrameAdapterV22(env, action_config)
        episode_reset = reset_stock_taskframe_episode_v22
    records: list[dict[str, Any]] = []
    for index in range(episodes):
        record, _episode = _run_episode_v614(
            env,
            adapter,
            bundle,
            requested_seed=seed_base + index,
            maximum_steps=maximum_steps,
            deterministic=True,
            random_action_probability=0.0,
            action_seed=seed_base ^ (index + 0x614E),
            collect_replay=False,
            episode_reset=episode_reset,
        )
        records.append(record)
    success_count = sum(int(row["strict_success"]) for row in records)
    evaluation = {
        "format": GOAL_CONDITIONED_MARKOV_EVALUATION_FORMAT_V614,
        "created_at_utc": _utc_now(),
        "seed_base": seed_base,
        "episode_count": episodes,
        "strict_success_count": success_count,
        "strict_success_rate": success_count / episodes,
        "contact_episode_count": sum(
            int(row["valid_contact_steps"] > 0) for row in records
        ),
        "mean_net_target_progress_m": float(
            np.mean([row["net_target_progress_m"] for row in records])
        ),
        "mean_ik_failure_steps": float(
            np.mean([row["ik_failure_steps"] for row in records])
        ),
        "total_ik_failure_steps": int(
            sum(row["ik_failure_steps"] for row in records)
        ),
        "episodes": records,
        "controller_state_complete": True,
        "exact_three_second_environment_success_only": True,
        "her_successes_in_numerator": 0,
        "task_independent_home_reset": task_independent_home_reset,
        "production_admission": False,
    }
    evaluation["automatic_failure_cohort_audit_v619"] = (
        _failure_cohort_audit_v619(records)
    )
    return evaluation


def _checkpoint_payload_v614(
    bundle: GoalConditionedMarkovHerSACBundleV614,
    config: GoalConditionedHerSACConfigV43,
    replay: GoalConditionedMarkovHerReplayV614,
    *,
    run_plan_sha256: str,
    phase: str,
    online_episode_index: int,
    warm_start_audit: dict[str, Any],
    adaptive_start_curriculum_v622: bool,
) -> dict[str, Any]:
    return {
        "format": GOAL_CONDITIONED_MARKOV_HER_CHECKPOINT_FORMAT_V614,
        "algorithm_format": GOAL_CONDITIONED_MARKOV_HER_SAC_FORMAT_V614,
        "created_at_utc": _utc_now(),
        "phase": phase,
        "online_episode_index": online_episode_index,
        "update_index": bundle.update_index,
        "config": asdict(config),
        "run_plan_sha256": run_plan_sha256,
        "warm_start_audit": warm_start_audit,
        "actor_trust_region_v621": (
            None
            if bundle.actor_trust_region_v621 is None
            else asdict(bundle.actor_trust_region_v621)
        ),
        "phase_isolated_acquisition_v626": (
            None
            if bundle.phase_isolated_acquisition_v626 is None
            else asdict(bundle.phase_isolated_acquisition_v626)
        ),
        "phase_isolated_acquisition_option_v639": (
            None
            if bundle.phase_isolated_acquisition_option_v639 is None
            else asdict(bundle.phase_isolated_acquisition_option_v639)
        ),
        "adaptive_start_curriculum_v622": adaptive_start_curriculum_v622,
        "actor_state_dict": bundle.actor.state_dict(),
        "critic_state_dict": bundle.critic.state_dict(),
        "target_critic_state_dict": bundle.target_critic.state_dict(),
        "feasibility_state_dict": bundle.feasibility.state_dict(),
        "actor_optimizer_state_dict": bundle.actor_optimizer.state_dict(),
        "critic_optimizer_state_dict": bundle.critic_optimizer.state_dict(),
        "feasibility_optimizer_state_dict": (
            bundle.feasibility_optimizer.state_dict()
        ),
        "replay_manifest": replay.manifest(),
        "controller_state_complete": True,
        "controller_state_schema_sha256": (
            TASKFRAME_CONTROLLER_STATE_SCHEMA_SHA256_V614
        ),
        "simulator_privileged_actor": True,
        "deployable_visual_policy": False,
        "expert_calls": 0,
        "behavior_cloning_steps": 0,
        "production_admission": False,
    }


def _restore_checkpoint_v614(
    bundle: GoalConditionedMarkovHerSACBundleV614,
    path: Path,
) -> dict[str, Any]:
    payload = torch.load(
        Path(path).expanduser().resolve(),
        map_location="cpu",
        weights_only=True,
    )
    if (
        type(payload) is not dict
        or payload.get("format")
        != GOAL_CONDITIONED_MARKOV_HER_CHECKPOINT_FORMAT_V614
        or payload.get("algorithm_format")
        != GOAL_CONDITIONED_MARKOV_HER_SAC_FORMAT_V614
        or payload.get("controller_state_schema_sha256")
        != TASKFRAME_CONTROLLER_STATE_SCHEMA_SHA256_V614
        or payload.get("production_admission") is not False
    ):
        raise ValueError("V614 resume checkpoint identity changed")
    stored_option = payload.get("phase_isolated_acquisition_option_v639")
    current_option = bundle.phase_isolated_acquisition_option_v639
    option_upgrade = current_option is not None and stored_option is None
    if option_upgrade:
        incompatible = bundle.actor.load_state_dict(
            payload["actor_state_dict"], strict=False
        )
        expected_missing = {
            name
            for name in bundle.actor.state_dict()
            if name.startswith("acquisition_option_encoder_v639.")
        }
        if (
            set(incompatible.missing_keys) != expected_missing
            or incompatible.unexpected_keys
        ):
            raise ValueError("V639 actor checkpoint upgrade keys changed")
        option_encoder = bundle.actor.acquisition_option_encoder_v639
        if option_encoder is None or bool(
            torch.count_nonzero(option_encoder[-1].weight).item()
            or torch.count_nonzero(option_encoder[-1].bias).item()
        ):
            raise RuntimeError("V639 upgraded option is not zero initialized")
    else:
        bundle.actor.load_state_dict(payload["actor_state_dict"], strict=True)
    bundle.critic.load_state_dict(payload["critic_state_dict"], strict=True)
    bundle.target_critic.load_state_dict(
        payload["target_critic_state_dict"], strict=True
    )
    bundle.feasibility.load_state_dict(
        payload["feasibility_state_dict"], strict=True
    )
    if not option_upgrade:
        bundle.actor_optimizer.load_state_dict(
            payload["actor_optimizer_state_dict"]
        )
    bundle.critic_optimizer.load_state_dict(
        payload["critic_optimizer_state_dict"]
    )
    bundle.feasibility_optimizer.load_state_dict(
        payload["feasibility_optimizer_state_dict"]
    )
    bundle.update_index = int(payload["update_index"])
    return {
        **payload,
        "phase_isolated_acquisition_option_upgrade_v639": {
            "upgraded_from_checkpoint_without_option": option_upgrade,
            "pre_upgrade_actor_behavior_preserved_exactly": True,
            "actor_optimizer_state_transferred": not option_upgrade,
            "critic_optimizer_state_transferred": True,
            "feasibility_optimizer_state_transferred": True,
            "expert_calls": 0,
            "behavior_cloning_steps": 0,
            "production_admission": False,
        },
    }


def run_goal_conditioned_markov_her_sac_v614(
    *,
    output_dir: Path,
    collection_plan: Path,
    parent_v43_checkpoint: Path | None,
    resume_v614_checkpoint: Path | None,
    resume_replay_npz: Path | None,
    initialization_seed: int,
    sampling_seed_base: int,
    online_seed_base: int,
    evaluation_seed_base: int,
    online_episodes: int,
    updates_per_episode: int,
    evaluation_every_episodes: int,
    evaluation_episodes: int,
    maximum_episode_steps: int,
    initial_random_action_probability: float,
    final_random_action_probability: float,
    device: str,
    task_independent_home_reset: bool = False,
    training_curriculum_stage: int | None = 31,
    colored_random_exploration: (
        AxisScaledColoredExplorationConfigV605 | None
    ) = None,
    config: GoalConditionedHerSACConfigV43 | None = None,
    matched_baseline_evaluation: Path | None = None,
    actor_trust_region_v621: MarkovActorTrustRegionConfigV621 | None = None,
    phase_isolated_acquisition_v626: (
        PhaseIsolatedAcquisitionConfigV626 | None
    ) = None,
    phase_isolated_acquisition_option_v639: (
        PhaseIsolatedAcquisitionOptionConfigV639 | None
    ) = None,
    adaptive_start_curriculum_v622: bool = False,
) -> dict[str, Any]:
    for name, value in (
        ("online_episodes", online_episodes),
        ("updates_per_episode", updates_per_episode),
        ("evaluation_every_episodes", evaluation_every_episodes),
        ("evaluation_episodes", evaluation_episodes),
        ("maximum_episode_steps", maximum_episode_steps),
    ):
        if type(value) is not int or value < 1:
            raise ValueError(f"V614 {name} must be positive")
    if training_curriculum_stage not in (None, 31, 32):
        raise ValueError("V614 curriculum stage must be 31 or 32")
    if task_independent_home_reset and training_curriculum_stage is not None:
        raise ValueError("V614 Home reset and curriculum reset are exclusive")
    if type(adaptive_start_curriculum_v622) is not bool:
        raise TypeError("V622 adaptive-start selector must be boolean")
    if adaptive_start_curriculum_v622 and (
        task_independent_home_reset or training_curriculum_stage is None
    ):
        raise ValueError(
            "V622 adaptive starts require a curriculum stage and own Home probes"
        )
    if actor_trust_region_v621 is not None:
        actor_trust_region_v621.validate()
    if phase_isolated_acquisition_v626 is not None:
        phase_isolated_acquisition_v626.validate()
        if (
            actor_trust_region_v621 is None
            or not actor_trust_region_v621.freeze_parent_base
        ):
            raise ValueError("V626 requires the frozen V621 parent actor")
    if phase_isolated_acquisition_option_v639 is not None:
        phase_isolated_acquisition_option_v639.validate()
        if phase_isolated_acquisition_v626 is None:
            raise ValueError("V639 requires V626 phase isolation")
    probabilities = (
        initial_random_action_probability,
        final_random_action_probability,
    )
    if any(
        not np.isfinite(value) or not 0.0 <= value <= 1.0
        for value in probabilities
    ):
        raise ValueError("V614 exploration probability is invalid")
    new_run = resume_v614_checkpoint is None
    if new_run != (resume_replay_npz is None):
        raise ValueError("V614 checkpoint and replay must resume together")
    if new_run and parent_v43_checkpoint is None:
        raise ValueError("V614 new run requires a frozen V43 parent")
    destination = Path(output_dir).expanduser().resolve()
    if destination.exists():
        raise FileExistsError(f"V614 output already exists: {destination}")
    source_plan, environment_config, action_config, scene_path = (
        _load_collection_contract(collection_plan)
    )
    runtime_transport_height = (
        float(action_config.maximum_tool_height_m)
        + HOME_ACQUISITION_TRANSPORT_HEIGHT_MARGIN_M_V597
    )
    if phase_isolated_acquisition_v626 is not None:
        if not np.isclose(
            phase_isolated_acquisition_v626
            .exact_transport_maximum_tool_height_m,
            runtime_transport_height,
            rtol=0.0,
            atol=1.0e-12,
        ):
            raise ValueError(
                "V630 actor gate and controller transport height disagree"
            )
        if not np.isclose(
            phase_isolated_acquisition_v626.exact_transport_alignment,
            HOME_ACQUISITION_TRANSPORT_ALIGNMENT_V597,
            rtol=0.0,
            atol=1.0e-12,
        ):
            raise ValueError(
                "V630 actor gate and controller transport alignment disagree"
            )
    requested_stage = (
        32
        if task_independent_home_reset
        else 30
        if training_curriculum_stage is None
        else training_curriculum_stage
    )
    if environment_config.reverse_curriculum_stage_v26 != requested_stage:
        environment_config = replace(
            environment_config,
            reverse_curriculum_stage_v26=requested_stage,
            strict_success_hold_steps=90,
        )
    if environment_config.max_steps < maximum_episode_steps:
        environment_config = replace(
            environment_config, max_steps=maximum_episode_steps
        )
    selected_device = _resolve_device(device)
    baseline_payload = None
    baseline_path = None
    if matched_baseline_evaluation is not None:
        baseline_path = Path(matched_baseline_evaluation).expanduser().resolve()
        baseline_payload = json.loads(baseline_path.read_text(encoding="utf-8"))
        if (
            type(baseline_payload) is not dict
            or not isinstance(baseline_payload.get("episodes"), list)
            or baseline_payload.get("production_admission") is not False
        ):
            raise ValueError("V616 matched baseline evaluation is invalid")

    parent_payload: dict[str, Any] | None = None
    resume_payload: dict[str, Any] | None = None
    if new_run:
        parent_path = Path(parent_v43_checkpoint).expanduser().resolve()  # type: ignore[arg-type]
        parent_payload = torch.load(
            parent_path, map_location="cpu", weights_only=True
        )
        parent_configuration = parent_payload.get("config")
        if type(parent_configuration) is not dict:
            raise ValueError("V614 V43 parent lacks configuration")
        selected_config = config or GoalConditionedHerSACConfigV43(
            **parent_configuration
        )
        replay = GoalConditionedMarkovHerReplayV614(selected_config)
    else:
        resume_path = Path(resume_v614_checkpoint).expanduser().resolve()  # type: ignore[arg-type]
        resume_payload = torch.load(
            resume_path, map_location="cpu", weights_only=True
        )
        resume_configuration = resume_payload.get("config")
        if type(resume_configuration) is not dict:
            raise ValueError("V614 resume checkpoint lacks configuration")
        selected_config = config or GoalConditionedHerSACConfigV43(
            **resume_configuration
        )
        replay = GoalConditionedMarkovHerReplayV614.load_npz(
            Path(resume_replay_npz), selected_config  # type: ignore[arg-type]
        )
    if adaptive_start_curriculum_v622:
        selected_config = replace(
            selected_config,
            phase_progress_reward_active=True,
            phase_retention_reward_active=True,
            home_acquisition_reward_active=True,
            home_acquisition_suppress_future_her_above_distance_m=(
                phase_isolated_acquisition_v626
                .exact_transport_distance_m
                if phase_isolated_acquisition_v626 is not None
                else selected_config
                .home_acquisition_suppress_future_her_above_distance_m
            ),
            home_acquisition_suppress_future_her_above_tool_height_m=(
                phase_isolated_acquisition_v626
                .exact_transport_maximum_tool_height_m
                if phase_isolated_acquisition_v626 is not None
                else selected_config
                .home_acquisition_suppress_future_her_above_tool_height_m
            ),
            home_acquisition_suppress_future_her_below_alignment=(
                phase_isolated_acquisition_v626.exact_transport_alignment
                if phase_isolated_acquisition_v626 is not None
                else selected_config
                .home_acquisition_suppress_future_her_below_alignment
            ),
        )
        if new_run:
            replay = GoalConditionedMarkovHerReplayV614(selected_config)
    selected_config.validate()
    if (task_independent_home_reset or adaptive_start_curriculum_v622) and not (
        selected_config.home_acquisition_reward_active
    ):
        raise ValueError("V614/V622 Home or approach reset requires acquisition reward")
    if phase_isolated_acquisition_v626 is not None and (
        phase_isolated_acquisition_v626.precontact_standoff_m
        != selected_config.home_acquisition_standoff_m
        or phase_isolated_acquisition_v626.precontact_tool_height_m
        != selected_config.home_acquisition_tool_height_m
    ):
        raise ValueError("V626 actor gate and acquisition reward geometry disagree")
    restored_trust_region = actor_trust_region_v621
    restored_phase_isolation = phase_isolated_acquisition_v626
    restored_acquisition_option = phase_isolated_acquisition_option_v639
    if not new_run:
        stored_trust_payload = resume_payload.get("actor_trust_region_v621")
        stored_trust_region = (
            None
            if stored_trust_payload is None
            else MarkovActorTrustRegionConfigV621(**stored_trust_payload)
        )
        if (
            actor_trust_region_v621 is not None
            and actor_trust_region_v621 != stored_trust_region
        ):
            raise ValueError("V621 resume trust-region configuration changed")
        restored_trust_region = stored_trust_region
        stored_phase_payload = resume_payload.get(
            "phase_isolated_acquisition_v626"
        )
        stored_phase_isolation = (
            None
            if stored_phase_payload is None
            else PhaseIsolatedAcquisitionConfigV626(
                **stored_phase_payload
            )
        )
        if (
            phase_isolated_acquisition_v626 is not None
            and phase_isolated_acquisition_v626
            != stored_phase_isolation
        ):
            raise ValueError("V626 resume phase-isolation configuration changed")
        restored_phase_isolation = stored_phase_isolation
        stored_option_payload = resume_payload.get(
            "phase_isolated_acquisition_option_v639"
        )
        stored_acquisition_option = (
            None
            if stored_option_payload is None
            else PhaseIsolatedAcquisitionOptionConfigV639(
                **stored_option_payload
            )
        )
        if (
            phase_isolated_acquisition_option_v639 is not None
            and stored_acquisition_option is not None
            and phase_isolated_acquisition_option_v639
            != stored_acquisition_option
        ):
            raise ValueError("V639 resume acquisition-option configuration changed")
        restored_acquisition_option = (
            phase_isolated_acquisition_option_v639
            if stored_acquisition_option is None
            else stored_acquisition_option
        )
        if bool(
            resume_payload.get("adaptive_start_curriculum_v622", False)
        ) != adaptive_start_curriculum_v622:
            raise ValueError("V622 resume adaptive-start mode changed")
    if restored_phase_isolation is not None and (
        restored_phase_isolation.precontact_standoff_m
        != selected_config.home_acquisition_standoff_m
        or restored_phase_isolation.precontact_tool_height_m
        != selected_config.home_acquisition_tool_height_m
    ):
        raise ValueError(
            "V626 restored actor gate and acquisition reward geometry disagree"
        )
    if restored_phase_isolation is not None and (
        not np.isclose(
            restored_phase_isolation
            .exact_transport_maximum_tool_height_m,
            runtime_transport_height,
            rtol=0.0,
            atol=1.0e-12,
        )
        or not np.isclose(
            restored_phase_isolation.exact_transport_alignment,
            HOME_ACQUISITION_TRANSPORT_ALIGNMENT_V597,
            rtol=0.0,
            atol=1.0e-12,
        )
    ):
        raise ValueError(
            "V630 restored actor gate and controller phase disagree"
        )
    bundle = initialize_goal_conditioned_markov_her_sac_v614(
        initialization_seed,
        device=selected_device,
        config=selected_config,
        actor_trust_region_v621=restored_trust_region,
        phase_isolated_acquisition_v626=restored_phase_isolation,
        phase_isolated_acquisition_option_v639=(
            restored_acquisition_option
        ),
    )
    option_upgrade_audit = {
        "upgraded_from_checkpoint_without_option": False,
        "pre_upgrade_actor_behavior_preserved_exactly": True,
        "actor_optimizer_state_transferred": False if new_run else True,
        "critic_optimizer_state_transferred": False if new_run else True,
        "feasibility_optimizer_state_transferred": False if new_run else True,
        "expert_calls": 0,
        "behavior_cloning_steps": 0,
        "production_admission": False,
    }
    if new_run:
        warm_start_audit = warm_start_markov_her_sac_v614_from_v43(
            bundle, parent_payload  # type: ignore[arg-type]
        )
        completed_online_episodes = 0
    else:
        restored = _restore_checkpoint_v614(
            bundle, Path(resume_v614_checkpoint)  # type: ignore[arg-type]
        )
        option_upgrade_audit = dict(
            restored["phase_isolated_acquisition_option_upgrade_v639"]
        )
        warm_start_audit = dict(restored["warm_start_audit"])
        completed_online_episodes = int(
            restored.get("online_episode_index", -1)
        )
        if (
            completed_online_episodes < 1
            or completed_online_episodes != replay.episode_count
        ):
            raise ValueError(
                "V634 resume checkpoint and replay episode counts disagree"
            )
        if online_episodes <= completed_online_episodes:
            raise ValueError(
                "V634 resume target must exceed completed online episodes"
            )

    warm_start_audit = {
        **warm_start_audit,
        "phase_isolated_acquisition_option_upgrade_v639": (
            option_upgrade_audit
        ),
    }

    source_file = Path(__file__).resolve()
    run_plan = {
        "format": GOAL_CONDITIONED_MARKOV_TRAIN_RUN_FORMAT_V614,
        "algorithm_format": GOAL_CONDITIONED_MARKOV_HER_SAC_FORMAT_V614,
        "created_at_utc": _utc_now(),
        "output_dir": str(destination),
        "resolved_device": selected_device,
        "configuration": asdict(selected_config),
        "online_episodes": online_episodes,
        "completed_online_episodes_before_resume": (
            completed_online_episodes
        ),
        "resume_curriculum_records_reconstructed_from_exact_replay_v634": (
            adaptive_start_curriculum_v622
            and completed_online_episodes > 0
        ),
        "updates_per_episode": updates_per_episode,
        "evaluation_every_episodes": evaluation_every_episodes,
        "evaluation_episodes": evaluation_episodes,
        "maximum_episode_steps": maximum_episode_steps,
        "initial_random_action_probability": (
            initial_random_action_probability
        ),
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
        "parent_v43_checkpoint": (
            None
            if parent_v43_checkpoint is None
            else str(Path(parent_v43_checkpoint).resolve())
        ),
        "parent_v43_checkpoint_sha256": (
            None
            if parent_v43_checkpoint is None
            else sha256_file_v1(parent_v43_checkpoint)
        ),
        "resume_v614_checkpoint": (
            None
            if resume_v614_checkpoint is None
            else str(Path(resume_v614_checkpoint).resolve())
        ),
        "resume_replay_npz": (
            None
            if resume_replay_npz is None
            else str(Path(resume_replay_npz).resolve())
        ),
        "matched_baseline_evaluation": (
            None if baseline_path is None else str(baseline_path)
        ),
        "matched_baseline_evaluation_sha256": (
            None if baseline_path is None else sha256_file_v1(baseline_path)
        ),
        "warm_start_audit": warm_start_audit,
        "phase_isolated_acquisition_option_upgrade_v639": (
            option_upgrade_audit
        ),
        "actor_trust_region_v621": (
            None
            if restored_trust_region is None
            else {
                "format": MARKOV_ACTOR_TRUST_REGION_FORMAT_V621,
                **asdict(restored_trust_region),
            }
        ),
        "phase_isolated_acquisition_v626": (
            None
            if restored_phase_isolation is None
            else {
                "format": PHASE_ISOLATED_ACQUISITION_FORMAT_V626,
                **asdict(restored_phase_isolation),
            }
        ),
        "phase_isolated_acquisition_option_v639": (
            None
            if restored_acquisition_option is None
            else {
                "format": PHASE_ISOLATED_ACQUISITION_OPTION_FORMAT_V639,
                **asdict(restored_acquisition_option),
            }
        ),
        "adaptive_start_curriculum_v622": (
            {
                "format": ADAPTIVE_START_CURRICULUM_FORMAT_V622,
                "active": True,
                "training_tiers": [
                    tier.code for tier in START_TIERS_V622
                ],
                "only_exact_home_eligible_for_final_data": True,
            }
            if adaptive_start_curriculum_v622
            else None
        ),
        "controller_state_schema_sha256": (
            TASKFRAME_CONTROLLER_STATE_SCHEMA_SHA256_V614
        ),
        "controller_state_complete_replay_required": True,
        "legacy_v43_replay_imported": False,
        "task_independent_home_reset": task_independent_home_reset,
        "training_curriculum_stage": training_curriculum_stage,
        "training_curriculum_only": training_curriculum_stage is not None,
        "training_curriculum_object_target_distance_is_full_task": bool(
            training_curriculum_stage == 32
        ),
        "training_curriculum_object_target_distance_is_bridge_only": bool(
            training_curriculum_stage == 31
        ),
        "only_stage32_meets_v607_full_task_distance_gate": True,
        "training_curriculum_trajectories_admitted_as_final_data": False,
        "colored_random_exploration": (
            None
            if colored_random_exploration is None
            else asdict(colored_random_exploration)
        ),
        "source_type": SOURCE_TYPE,
        "simulator_privileged_actor": True,
        "visual_actor_training": False,
        "act_training": False,
        "future_her_learning_only": True,
        "expert_calls": 0,
        "behavior_cloning_steps": 0,
        "physical_samples": 0,
        "source_hashes": {
            "edgearm/train_goal_conditioned_markov_her_sac_v614.py": (
                sha256_file_v1(source_file)
            ),
            "edgearm/goal_conditioned_markov_her_sac_v614.py": (
                sha256_file_v1(
                    source_file.with_name(
                        "goal_conditioned_markov_her_sac_v614.py"
                    )
                )
            ),
            "edgearm/taskframe_controller_state_v614.py": sha256_file_v1(
                source_file.with_name("taskframe_controller_state_v614.py")
            ),
            "edgearm/markov_actor_trust_region_v621.py": sha256_file_v1(
                source_file.with_name("markov_actor_trust_region_v621.py")
            ),
            "edgearm/phase_isolated_acquisition_v626.py": sha256_file_v1(
                source_file.with_name("phase_isolated_acquisition_v626.py")
            ),
            "edgearm/phase_isolated_acquisition_option_v639.py": (
                sha256_file_v1(
                    source_file.with_name(
                        "phase_isolated_acquisition_option_v639.py"
                    )
                )
            ),
            "edgearm/adaptive_start_curriculum_v622.py": sha256_file_v1(
                source_file.with_name("adaptive_start_curriculum_v622.py")
            ),
            "edgearm/interpolated_approach_reset_v622.py": sha256_file_v1(
                source_file.with_name("interpolated_approach_reset_v622.py")
            ),
        },
        "production_admission": False,
    }
    run_plan["run_plan_sha256"] = canonical_sha256_v1(run_plan)
    destination.mkdir(parents=True)
    (destination / "checkpoints").mkdir()
    (destination / "evaluations").mkdir()
    _atomic_json(destination / "run_plan.json", run_plan)
    _atomic_json(
        destination / "run_state.json",
        {
            "status": "running",
            "phase": "online_training",
            "online_episode_index": completed_online_episodes,
            "update_index": bundle.update_index,
            "replay_transition_count": replay.transition_count,
            "updated_at_utc": _utc_now(),
        },
    )

    env = RealisticEdgeArmEnvV10(
        environment_config,
        seed=online_seed_base,
        model_scene_path=scene_path,
    )
    if adaptive_start_curriculum_v622:
        precontact_adapter = StockGripperTaskFrameAdapterV22(
            env, action_config
        )
        approach_adapter = StockGripperApproachTaskFrameAdapterV622(
            env, action_config
        )
        home_adapter = StockGripperHomeTaskFrameAdapterV597(
            env, action_config
        )
        adapter = precontact_adapter
        episode_reset = reset_stock_taskframe_episode_v22
    elif task_independent_home_reset:
        adapter = StockGripperHomeTaskFrameAdapterV597(env, action_config)
        episode_reset = reset_stock_home_taskframe_episode_v597
    else:
        adapter = StockGripperTaskFrameAdapterV22(env, action_config)
        episode_reset = reset_stock_taskframe_episode_v22
    evaluations: list[dict[str, Any]] = []
    online_records: list[dict[str, Any]] = (
        _curriculum_records_from_replay_v634(replay)
        if adaptive_start_curriculum_v622
        and completed_online_episodes > 0
        else []
    )
    best_score: tuple[int, float, int, int] | None = None
    best_checkpoint: Path | None = None
    best_evaluation: dict[str, Any] | None = None
    latest_metrics: dict[str, Any] | None = None
    try:
        for online_index in range(
            completed_online_episodes + 1,
            online_episodes + 1,
        ):
            fraction = (online_index - 1) / max(online_episodes - 1, 1)
            random_probability = initial_random_action_probability + fraction * (
                final_random_action_probability
                - initial_random_action_probability
            )
            selected_start_tier = None
            start_selection_audit = None
            if adaptive_start_curriculum_v622:
                selected_start_tier, start_selection_audit = (
                    select_start_tier_v622(
                        episode_index=online_index,
                        records=online_records,
                        seed=sampling_seed_base ^ (online_index * 0x622A11),
                    )
                )
                if selected_start_tier.reset_kind == "task_aligned_precontact":
                    adapter = precontact_adapter
                    episode_reset = reset_stock_taskframe_episode_v22
                elif selected_start_tier.reset_kind == "task_independent_home":
                    adapter = home_adapter
                    episode_reset = reset_stock_home_taskframe_episode_v597
                else:
                    adapter = approach_adapter
                    selected_approach_fraction = float(
                        selected_start_tier.home_to_precontact_fraction
                    )

                    def episode_reset(
                        reset_env: RealisticEdgeArmEnvV10,
                        renderer: Any,
                        reset_adapter: Any,
                        *,
                        requested_seed: int,
                        obstacle: bool,
                        stress: bool,
                    ) -> dict[str, Any]:
                        return reset_stock_interpolated_approach_episode_v622(
                            reset_env,
                            renderer,
                            reset_adapter,
                            requested_seed=requested_seed,
                            obstacle=obstacle,
                            stress=stress,
                            home_to_precontact_fraction=(
                                selected_approach_fraction
                            ),
                            precontact_standoff_m=(
                                selected_config.home_acquisition_standoff_m
                            ),
                            precontact_tool_height_m=(
                                selected_config.home_acquisition_tool_height_m
                            ),
                        )
            record, episode = _run_episode_v614(
                env,
                adapter,
                bundle,
                requested_seed=online_seed_base + online_index - 1,
                maximum_steps=maximum_episode_steps,
                deterministic=False,
                random_action_probability=float(random_probability),
                action_seed=online_seed_base ^ (online_index * 0x614A11),
                collect_replay=True,
                episode_reset=episode_reset,
                colored_random_exploration=colored_random_exploration,
            )
            if episode is None:
                raise RuntimeError("V614 online collection lost replay")
            if selected_start_tier is not None:
                episode["start_tier_index_v622"] = np.full(
                    len(episode["terminal"]),
                    selected_start_tier.index,
                    dtype=np.int8,
                )
                record["start_tier_v622"] = selected_start_tier.code
                record["start_curriculum_selection_v622"] = (
                    start_selection_audit
                )
                record["start_tier_final_data_eligible_v622"] = bool(
                    selected_start_tier.eligible_for_final_data
                )
                if selected_start_tier.reset_kind == "interpolated_approach":
                    approach_audit = (
                        approach_adapter.approach_reset_audit_v622
                    )
                    record["approach_reset_evidence_v627"] = {
                        "format": approach_audit.get("format"),
                        "home_to_precontact_fraction": approach_audit.get(
                            "home_to_precontact_fraction"
                        ),
                        "initial_tool_precontact_distance_m": (
                            approach_audit.get(
                                "initial_tool_precontact_distance_m_v627"
                            )
                        ),
                        "initial_precontact_face_alignment": (
                            approach_audit.get(
                                "initial_precontact_face_alignment_v627"
                            )
                        ),
                        "static_path_valid": approach_audit.get(
                            "static_interpolation_path_audit", {}
                        ).get("path_valid"),
                        "privileged_training_reset": True,
                        "eligible_for_final_data": False,
                        "production_admission": False,
                    }
                record["bulk_vla_data_use_allowed"] = False
                record["production_admission"] = False
            replay.add_episode(
                episode,
                source=(
                    f"online_v614_episode_{online_index:06d}"
                    if selected_start_tier is None
                    else (
                        f"online_v622_{selected_start_tier.code}_"
                        f"episode_{online_index:06d}"
                    )
                ),
            )
            record["online_episode_index"] = online_index
            record["replay_transition_count_after_append"] = (
                replay.transition_count
            )
            online_records.append(record)
            _append_jsonl(destination / "online_episodes.jsonl", record)

            for local_update in range(1, updates_per_episode + 1):
                batch = replay.sample(
                    batch_size=selected_config.batch_size,
                    seed=sampling_seed_base + bundle.update_index,
                )
                metrics = goal_conditioned_markov_her_sac_update_v614(
                    bundle, batch, selected_config
                )
                latest_metrics = asdict(metrics)
                latest_metrics.update(
                    {
                        "phase": "online_training",
                        "online_episode_index": online_index,
                        "local_update_after_episode": local_update,
                    }
                )
                _append_jsonl(
                    destination / "metrics.jsonl", latest_metrics
                )

            replay.save_npz(destination / "replay_latest.npz")
            checkpoint = (
                destination
                / "checkpoints"
                / f"episode_{online_index:06d}.pt"
            )
            _atomic_torch_save(
                checkpoint,
                _checkpoint_payload_v614(
                    bundle,
                    selected_config,
                    replay,
                    run_plan_sha256=run_plan["run_plan_sha256"],
                    phase="online_training",
                    online_episode_index=online_index,
                    warm_start_audit=warm_start_audit,
                    adaptive_start_curriculum_v622=(
                        adaptive_start_curriculum_v622
                    ),
                ),
            )
            if (
                online_index % evaluation_every_episodes == 0
                or online_index == online_episodes
            ):
                transport_retention_evaluation = None
                if adaptive_start_curriculum_v622:
                    transport_retention_evaluation = _evaluate_v614(
                        environment_config,
                        action_config,
                        scene_path,
                        bundle,
                        seed_base=evaluation_seed_base,
                        episodes=evaluation_episodes,
                        maximum_steps=maximum_episode_steps,
                        task_independent_home_reset=False,
                    )
                    transport_retention_evaluation["evaluation_scope_v622"] = (
                        "privileged_precontact_transport_retention_only"
                    )
                    evaluation = _evaluate_v614(
                        environment_config,
                        action_config,
                        scene_path,
                        bundle,
                        seed_base=evaluation_seed_base,
                        episodes=evaluation_episodes,
                        maximum_steps=maximum_episode_steps,
                        task_independent_home_reset=True,
                    )
                    evaluation["evaluation_scope_v622"] = (
                        "exact_home_full_acquisition_to_push"
                    )
                    evaluation["transport_retention_evaluation_v622"] = (
                        transport_retention_evaluation
                    )
                else:
                    evaluation = _evaluate_v614(
                        environment_config,
                        action_config,
                        scene_path,
                        bundle,
                        seed_base=evaluation_seed_base,
                        episodes=evaluation_episodes,
                        maximum_steps=maximum_episode_steps,
                        task_independent_home_reset=(
                            task_independent_home_reset
                        ),
                    )
                evaluation["online_episode_index"] = online_index
                evaluation["update_index"] = bundle.update_index
                if baseline_payload is not None:
                    comparison_target = (
                        transport_retention_evaluation
                        if transport_retention_evaluation is not None
                        else evaluation
                    )
                    comparison = _matched_baseline_comparison_v616(
                        comparison_target, baseline_payload
                    )
                    comparison_target[
                        "matched_v43_baseline_comparison_v616"
                    ] = comparison
                    evaluation[
                        "matched_transport_baseline_comparison_v622"
                    ] = comparison
                _atomic_json(
                    destination
                    / "evaluations"
                    / f"episode_{online_index:06d}.json",
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
                            "format": (
                                "edgearm-v614-curriculum-checkpoint-selection-v1"
                            ),
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
                    "phase": "online_training",
                    "online_episode_index": online_index,
                    "online_episodes": online_episodes,
                    "update_index": bundle.update_index,
                    "latest_online_episode": record,
                    "latest_metrics": latest_metrics,
                    "latest_evaluation": (
                        None if not evaluations else evaluations[-1]
                    ),
                    "replay_transition_count": replay.transition_count,
                    "replay_episode_count": replay.episode_count,
                    "updated_at_utc": _utc_now(),
                },
            )
    except Exception as error:
        failure = {
            "format": GOAL_CONDITIONED_MARKOV_TRAIN_RUN_FORMAT_V614,
            "status": "failed",
            "error_type": type(error).__name__,
            "error": str(error),
            "online_episode_index": replay.episode_count,
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
        _checkpoint_payload_v614(
            bundle,
            selected_config,
            replay,
            run_plan_sha256=run_plan["run_plan_sha256"],
            phase="complete",
            online_episode_index=online_episodes,
            warm_start_audit=warm_start_audit,
            adaptive_start_curriculum_v622=(
                adaptive_start_curriculum_v622
            ),
        ),
    )
    replay.save_npz(destination / "replay_final.npz")
    final_evaluation = None if not evaluations else evaluations[-1]
    home_gate = False
    wilson = 0.0
    if final_evaluation is not None:
        successes = int(final_evaluation["strict_success_count"])
        count = int(final_evaluation["episode_count"])
        wilson = wilson_lower_bound_v26(successes, count)
        home_gate = bool(
            (task_independent_home_reset or adaptive_start_curriculum_v622)
            and count >= 48
            and float(final_evaluation["strict_success_rate"]) >= 0.80
            and wilson >= 0.65
            and all(
                row["task_aligned_privileged_reset"] is False
                and row["task_independent_final_home_reset"] is True
                and float(row["initial_target_coverage"]) == 0.0
                and float(row["initial_block_target_distance_m"])
                >= MINIMUM_FULL_TASK_INITIAL_DISTANCE_M_V607
                and int(row["invalid_contact_steps"]) == 0
                and int(row["safety_steps"]) == 0
                and (
                    not bool(row["strict_success"])
                    or float(row["net_target_progress_m"])
                    >= MINIMUM_FULL_TASK_NET_PROGRESS_M_V607
                )
                for row in final_evaluation["episodes"]
            )
        )
    summary = {
        "format": GOAL_CONDITIONED_MARKOV_TRAIN_RUN_FORMAT_V614,
        "status": "complete",
        "completed_at_utc": _utc_now(),
        "online_episode_count": online_episodes,
        "update_index": bundle.update_index,
        "replay_manifest": replay.manifest(),
        "final_checkpoint": str(final_checkpoint),
        "final_checkpoint_sha256": sha256_file_v1(final_checkpoint),
        "best_training_checkpoint": (
            None if best_checkpoint is None else str(best_checkpoint)
        ),
        "best_training_evaluation": best_evaluation,
        "final_evaluation": final_evaluation,
        "online_failure_cohort_audit_v619": _failure_cohort_audit_v619(
            online_records
        ),
        "strict_success_count": (
            0
            if final_evaluation is None
            else int(final_evaluation["strict_success_count"])
        ),
        "strict_success_rate": (
            0.0
            if final_evaluation is None
            else float(final_evaluation["strict_success_rate"])
        ),
        "controller_state_complete": True,
        "training_curriculum_only": training_curriculum_stage is not None,
        "training_curriculum_stage": training_curriculum_stage,
        "training_curriculum_trajectories_admitted_as_final_data": False,
        "actor_trust_region_v621": (
            None
            if bundle.actor_trust_region_v621 is None
            else {
                "format": MARKOV_ACTOR_TRUST_REGION_FORMAT_V621,
                **asdict(bundle.actor_trust_region_v621),
            }
        ),
        "phase_isolated_acquisition_v626": (
            None
            if bundle.phase_isolated_acquisition_v626 is None
            else {
                "format": PHASE_ISOLATED_ACQUISITION_FORMAT_V626,
                **asdict(bundle.phase_isolated_acquisition_v626),
            }
        ),
        "phase_isolated_acquisition_option_v639": (
            None
            if bundle.phase_isolated_acquisition_option_v639 is None
            else {
                "format": PHASE_ISOLATED_ACQUISITION_OPTION_FORMAT_V639,
                **asdict(bundle.phase_isolated_acquisition_option_v639),
            }
        ),
        "adaptive_start_curriculum_v622": (
            curriculum_cohort_audit_v622(online_records)
            if adaptive_start_curriculum_v622
            else None
        ),
        "home_strict_gate_passed": home_gate,
        "home_strict_gate_wilson_lower_bound": wilson,
        "successful_rl_data_generation_ready": home_gate,
        "bulk_multimodal_generation_started": False,
        "remaining_gate": (
            (
                "master Home acquisition on the 15-17 cm stage31 bridge, "
                "then pass the 48-task 17-19 cm stage32 Home-start strict "
                "gate before wrist-multimodal bulk collection"
            )
            if training_curriculum_stage == 31
            else (
                "48-task stage32 Home-start strict gate then "
                "wrist-multimodal bulk collection"
            )
            if task_independent_home_reset or adaptive_start_curriculum_v622
            else "transfer stable controller-state-complete transport to Home start"
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
            "online_episode_index": online_episodes,
            "update_index": bundle.update_index,
            "strict_success_count": summary["strict_success_count"],
            "strict_success_rate": summary["strict_success_rate"],
            "replay_transition_count": replay.transition_count,
            "updated_at_utc": _utc_now(),
        },
    )
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--collection-plan", type=Path, required=True)
    parser.add_argument("--parent-v43-checkpoint", type=Path)
    parser.add_argument("--resume-v614-checkpoint", type=Path)
    parser.add_argument("--resume-replay-npz", type=Path)
    parser.add_argument("--matched-baseline-evaluation", type=Path)
    parser.add_argument("--initialization-seed", type=int, default=614_000_000)
    parser.add_argument("--sampling-seed-base", type=int, default=614_100_000)
    parser.add_argument("--online-seed-base", type=int, default=614_200_000)
    parser.add_argument("--evaluation-seed-base", type=int, default=614_300_000)
    parser.add_argument("--online-episodes", type=int, default=8)
    parser.add_argument("--updates-per-episode", type=int, default=64)
    parser.add_argument("--evaluation-every-episodes", type=int, default=4)
    parser.add_argument("--evaluation-episodes", type=int, default=4)
    parser.add_argument("--maximum-episode-steps", type=int, default=480)
    parser.add_argument(
        "--initial-random-action-probability", type=float, default=0.10
    )
    parser.add_argument(
        "--final-random-action-probability", type=float, default=0.02
    )
    parser.add_argument("--task-independent-home-reset", action="store_true")
    parser.add_argument(
        "--training-curriculum-stage", type=int, choices=(31, 32), default=31
    )
    parser.add_argument("--colored-random-exploration", action="store_true")
    parser.add_argument(
        "--adaptive-start-curriculum-v622",
        action="store_true",
        help=(
            "Train across precontact, audited intermediate approach, and "
            "exact Home starts; evaluate exact Home as the primary task."
        ),
    )
    parser.add_argument(
        "--actor-trust-region-v621",
        action="store_true",
        help=(
            "Freeze the V43 actor base, cap controller residuals, and "
            "anchor actions when controller backlog is small."
        ),
    )
    parser.add_argument(
        "--v621-maximum-context-hidden-ratio", type=float, default=0.05
    )
    parser.add_argument(
        "--v621-low-backlog-anchor-coefficient", type=float, default=2.0
    )
    parser.add_argument(
        "--v621-low-backlog-scale", type=float, default=0.35
    )
    parser.add_argument(
        "--phase-isolated-acquisition-v626",
        action="store_true",
        help=(
            "Allow a large learned residual only while far from or "
            "misaligned with pre-contact, then restore the exact frozen "
            "transport parent."
        ),
    )
    parser.add_argument(
        "--v626-maximum-acquisition-context-ratio",
        type=float,
        default=0.75,
    )
    parser.add_argument(
        "--v626-exact-transport-distance-m",
        type=float,
        default=0.040,
    )
    parser.add_argument(
        "--v626-full-acquisition-distance-m",
        type=float,
        default=0.120,
    )
    parser.add_argument(
        "--v626-exact-transport-alignment",
        type=float,
        default=0.98,
    )
    parser.add_argument(
        "--v626-full-acquisition-alignment",
        type=float,
        default=0.65,
    )
    parser.add_argument(
        "--v626-exact-transport-maximum-tool-height-m",
        type=float,
        default=0.061,
    )
    parser.add_argument(
        "--v626-full-acquisition-tool-height-m",
        type=float,
        default=0.100,
    )
    parser.add_argument(
        "--phase-isolated-acquisition-option-v639",
        action="store_true",
        help=(
            "Add a separately trainable action-space option during "
            "Home-to-contact acquisition while preserving exact parent "
            "transport behavior."
        ),
    )
    parser.add_argument(
        "--v639-maximum-pre-tanh-mean-residual",
        type=float,
        default=4.0,
    )
    parser.add_argument("--colored-exploration-rho", type=float, default=0.95)
    parser.add_argument(
        "--colored-exploration-standard-deviation",
        type=float,
        nargs=3,
        default=(0.65, 0.22, 0.12),
    )
    parser.add_argument(
        "--device", choices=("auto", "cpu", "mps"), default="auto"
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    curriculum_stage = (
        None
        if args.task_independent_home_reset
        else args.training_curriculum_stage
    )
    colored = (
        AxisScaledColoredExplorationConfigV605(
            standard_deviation=tuple(
                args.colored_exploration_standard_deviation
            ),
            autoregressive_rho=args.colored_exploration_rho,
        )
        if args.colored_random_exploration
        else None
    )
    summary = run_goal_conditioned_markov_her_sac_v614(
        output_dir=args.output_dir,
        collection_plan=args.collection_plan,
        parent_v43_checkpoint=args.parent_v43_checkpoint,
        resume_v614_checkpoint=args.resume_v614_checkpoint,
        resume_replay_npz=args.resume_replay_npz,
        initialization_seed=args.initialization_seed,
        sampling_seed_base=args.sampling_seed_base,
        online_seed_base=args.online_seed_base,
        evaluation_seed_base=args.evaluation_seed_base,
        online_episodes=args.online_episodes,
        updates_per_episode=args.updates_per_episode,
        evaluation_every_episodes=args.evaluation_every_episodes,
        evaluation_episodes=args.evaluation_episodes,
        maximum_episode_steps=args.maximum_episode_steps,
        initial_random_action_probability=(
            args.initial_random_action_probability
        ),
        final_random_action_probability=(
            args.final_random_action_probability
        ),
        device=args.device,
        task_independent_home_reset=args.task_independent_home_reset,
        training_curriculum_stage=curriculum_stage,
        colored_random_exploration=colored,
        matched_baseline_evaluation=args.matched_baseline_evaluation,
        actor_trust_region_v621=(
            MarkovActorTrustRegionConfigV621(
                freeze_parent_base=True,
                maximum_context_over_base_hidden_norm=(
                    args.v621_maximum_context_hidden_ratio
                ),
                low_backlog_anchor_coefficient=(
                    args.v621_low_backlog_anchor_coefficient
                ),
                low_backlog_scale=args.v621_low_backlog_scale,
            )
            if args.actor_trust_region_v621
            else None
        ),
        phase_isolated_acquisition_v626=(
            PhaseIsolatedAcquisitionConfigV626(
                exact_transport_distance_m=(
                    args.v626_exact_transport_distance_m
                ),
                full_acquisition_distance_m=(
                    args.v626_full_acquisition_distance_m
                ),
                exact_transport_alignment=(
                    args.v626_exact_transport_alignment
                ),
                full_acquisition_alignment=(
                    args.v626_full_acquisition_alignment
                ),
                exact_transport_maximum_tool_height_m=(
                    args.v626_exact_transport_maximum_tool_height_m
                ),
                full_acquisition_tool_height_m=(
                    args.v626_full_acquisition_tool_height_m
                ),
                maximum_acquisition_context_over_base_hidden_norm=(
                    args.v626_maximum_acquisition_context_ratio
                ),
            )
            if args.phase_isolated_acquisition_v626
            else None
        ),
        phase_isolated_acquisition_option_v639=(
            PhaseIsolatedAcquisitionOptionConfigV639(
                maximum_pre_tanh_mean_residual=(
                    args.v639_maximum_pre_tanh_mean_residual
                )
            )
            if args.phase_isolated_acquisition_option_v639
            else None
        ),
        adaptive_start_curriculum_v622=(
            args.adaptive_start_curriculum_v622
        ),
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
