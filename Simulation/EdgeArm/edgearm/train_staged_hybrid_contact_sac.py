"""Train the frozen stage-one hybrid/RLPD EdgeArm pushing policy.

This entry point intentionally trains only task-aligned contact, transport and
the strict 90-step hold in the nominal no-obstacle plant.  It creates a fresh
successful-prior buffer, performs an explicit balanced warm start, then uses
an exact 50/50 prior/online SAC batch for every online update.

No result from this script is production-, wrist-export-, or ACT-eligible.
Exact-Home acquisition and unchanged V4 guarded evaluation are later gates.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from collections import Counter
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .hybrid_contact_sac import (
    CONTACT_MODE_NAMES,
    HybridContactSACBundle,
    HybridContactSACConfig,
    SymmetricPriorOnlineReplay,
    hybrid_bundle_state_dict,
    initialize_hybrid_contact_sac,
    load_hybrid_bundle_state_dict,
    pretrain_hybrid_actor_from_prior,
    sample_epsilon_mixed_contact_mode,
    synchronize_reference_actor,
    update_hybrid_contact_sac,
)
from .staged_push_rl import (
    STAGED_PUSH_MODE_MASK_START,
    STAGED_PUSH_OBSERVATION_DIM,
    StagedPushEpisode,
    StagedPushExecutorConfig,
    StagedPushRewardConfig,
    StagedPushStage,
    StagedPushStep,
    collect_successful_prior_episode,
)


STAGED_HYBRID_TRAINING_FORMAT = "edgearm-staged-hybrid-conservative-awac-1.0.0"


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(
            json.dumps(
                payload,
                sort_keys=True,
                ensure_ascii=False,
                allow_nan=False,
            )
            + "\n"
        )
        stream.flush()
        os.fsync(stream.fileno())


def _wilson_lower(successes: int, trials: int, z: float = 1.96) -> float:
    if trials < 1:
        return 0.0
    probability = successes / trials
    denominator = 1.0 + z * z / trials
    centre = probability + z * z / (2.0 * trials)
    radius = z * math.sqrt(probability * (1.0 - probability) / trials + z * z / (4.0 * trials**2))
    return float((centre - radius) / denominator)


def _online_actor_learning_eligibility(
    step: StagedPushStep,
) -> tuple[bool, float, str]:
    """Return explicit executor provenance for conservative actor learning."""

    execution_l2 = float(np.linalg.norm(step.proposed_action - step.executed_action))
    if step.intervention:
        return False, execution_l2, "shield_intervention"
    if step.control_steps < 1:
        return False, execution_l2, "no_executed_control_step"
    if step.command_provenance_observed_steps != step.control_steps:
        return False, execution_l2, "command_provenance_unobserved"
    if step.command_safety_rewrite_steps > 0:
        return False, execution_l2, "command_safety_rewrite"
    if step.invalid_contact_steps > 0:
        return False, execution_l2, "invalid_contact"
    mode_name = CONTACT_MODE_NAMES[step.mode]
    if mode_name == "separate_recontact":
        eligible = bool(
            step.valid_contact_before
            and not step.valid_contact_after
            and step.control_steps > 0
        )
        return (
            eligible,
            execution_l2,
            "eligible_safe_separation" if eligible else "separation_not_achieved",
        )
    if mode_name == "approach":
        eligible = step.reward > 0.0
        return (
            eligible,
            execution_l2,
            "eligible_approach_progress" if eligible else "no_approach_progress",
        )
    if mode_name == "settle_hold":
        eligible = step.target_coverage >= 0.95 and step.reward > 0.0
        return (
            eligible,
            execution_l2,
            "eligible_hold_progress" if eligible else "no_hold_progress",
        )
    eligible = (
        step.valid_contact_steps > 0
        and step.effectful_push_steps > 0
        and step.reward > 0.0
    )
    return (
        eligible,
        execution_l2,
        "eligible_effectful_contact" if eligible else "no_effectful_contact",
    )


def _evaluation_rank(evaluation: dict[str, Any]) -> tuple[float, ...]:
    """Lexicographic fail-closed policy rank for fixed-seed evaluations."""

    return (
        float(evaluation["safe_strict_success_count"]),
        float(evaluation["strict_success_count"]),
        -float(evaluation["intervention_terminal_count"]),
        -float(evaluation["invalid_contact_control_steps"]),
        float(evaluation["mean_maximum_strict_hold_fraction"]),
        float(evaluation["mean_maximum_target_coverage"]),
    )


def _reset_actor_optimizer(
    bundle: HybridContactSACBundle,
    config: HybridContactSACConfig,
) -> None:
    bundle.actor_optimizer = torch.optim.Adam(
        bundle.actor.parameters(),
        lr=config.online_actor_learning_rate,
    )


def _actor_action(
    bundle: HybridContactSACBundle,
    observation: np.ndarray,
    *,
    rng: np.random.Generator,
    device: str | torch.device,
    deterministic: bool,
    mode_exploration_fraction: float,
    sample_continuous_action: bool,
    sample_actor_categorical: bool,
) -> tuple[int, np.ndarray, np.ndarray, np.ndarray, str]:
    tensor = torch.as_tensor(observation[None, :], dtype=torch.float32, device=device)
    with torch.no_grad():
        probabilities = bundle.actor.mode_probabilities(tensor)[0]
        probabilities_numpy = probabilities.detach().cpu().numpy().astype(np.float64)
        probabilities_numpy = np.clip(probabilities_numpy, 0.0, None)
        probability_sum = float(probabilities_numpy.sum())
        if not np.isfinite(probability_sum) or probability_sum <= 0.0:
            raise RuntimeError("actor produced an invalid categorical probability vector")
        # NumPy's Generator.choice applies a stricter sum-to-one check than a
        # float32 softmax guarantees after CPU conversion.
        probabilities_numpy /= probability_sum
        allowed_mode_mask = (
            observation[
                STAGED_PUSH_MODE_MASK_START : STAGED_PUSH_MODE_MASK_START
                + len(CONTACT_MODE_NAMES)
            ]
            > 0.5
        )
        mode, behavior_probabilities, selection_source = (
            sample_epsilon_mixed_contact_mode(
                probabilities_numpy,
                allowed_mode_mask,
                exploration_fraction=mode_exploration_fraction,
                rng=rng,
                deterministic=deterministic,
                sample_actor_categorical=sample_actor_categorical,
            )
        )
        action, _ = bundle.actor.action_for_mode(
            tensor,
            torch.as_tensor([mode], dtype=torch.long, device=device),
            deterministic=deterministic or not sample_continuous_action,
        )
    return (
        mode,
        action[0].detach().cpu().numpy().astype(np.float32),
        probabilities_numpy,
        behavior_probabilities,
        selection_source,
    )


def run_learned_stage_one_episode(
    bundle: HybridContactSACBundle,
    *,
    seed: int,
    rng: np.random.Generator,
    device: str | torch.device,
    deterministic: bool,
    replay: SymmetricPriorOnlineReplay | None = None,
    sac_config: HybridContactSACConfig | None = None,
    updates_per_option: int = 0,
    mode_exploration_fraction: float = 0.0,
    sample_continuous_action: bool = True,
    sample_actor_categorical: bool = True,
    executor_config: StagedPushExecutorConfig | None = None,
    scene_mode: str = "single",
    trajectory_path: Path | None = None,
) -> tuple[dict[str, Any], list[dict[str, float | int | bool | str]]]:
    if (replay is None) != (sac_config is None):
        raise ValueError("online replay and SAC configuration must be supplied together")
    if type(updates_per_option) is not int or updates_per_option < 0:
        raise ValueError("updates per option is invalid")
    if (
        not np.isfinite(mode_exploration_fraction)
        or not 0.0 <= mode_exploration_fraction <= 1.0
    ):
        raise ValueError("mode exploration fraction must be in [0, 1]")
    if type(sample_continuous_action) is not bool:
        raise TypeError("sample continuous action flag must be boolean")
    if type(sample_actor_categorical) is not bool:
        raise TypeError("sample actor categorical flag must be boolean")
    episode = StagedPushEpisode(seed=seed, executor_config=executor_config, scene_mode=scene_mode)
    observation = episode.reset(seed=seed, stage=StagedPushStage.CONTACT_TRANSPORT_HOLD)
    trace = []
    if trajectory_path is not None:
        env = episode.env
        def capture(command):
            trace.append((float(env.data.time), env.data.qpos.copy(), env.data.qvel.copy(),
                          env.data.ctrl.copy(), np.asarray(command).copy()))
        capture(np.zeros(6))
        original_step = env.step
        def recorded_step(command):
            result = original_step(command)
            capture(command)
            return result
        env.step = recorded_step
    initial_block_xy = np.asarray(episode.env.block_xy(), dtype=np.float64).copy()
    initial_target_xy = np.asarray(episode.env.target_xy, dtype=np.float64).copy()
    initial_direction = initial_target_xy - initial_block_xy
    initial_direction /= max(float(np.linalg.norm(initial_direction)), 1.0e-12)
    initial_distance = float(episode.env.distance_to_target())
    initial_coverage = float(episode.env.block_target_coverage())
    mode_counts: Counter[str] = Counter()
    selection_source_counts: Counter[str] = Counter()
    exploration_selected_mode_counts: Counter[str] = Counter()
    valid_contact_control_steps_by_mode: Counter[str] = Counter()
    effectful_push_control_steps_by_mode: Counter[str] = Counter()
    actor_learning_eligible_options_by_mode: Counter[str] = Counter()
    reward_by_mode: Counter[str] = Counter()
    total_valid = 0
    total_invalid = 0
    total_effectful = 0
    valid_contact_events_by_role: Counter[str] = Counter()
    invalid_contact_events_by_role: Counter[str] = Counter()
    invalid_contact_substeps_by_role: Counter[str] = Counter()
    invalid_contact_control_steps_by_mode: Counter[str] = Counter()
    invalid_contact_option_indices: list[int] = []
    total_reward = 0.0
    maximum_coverage = initial_coverage
    maximum_hold = 0.0
    update_metrics: list[dict[str, float | int | bool | str]] = []
    actor_eligible_options = 0
    actor_rejection_reasons: Counter[str] = Counter()
    proposal_execution_l2_values: list[float] = []
    proposal_projection_l2_values: list[float] = []
    projection_tracking_l2_values: list[float] = []
    tracking_residual_alert_options = 0
    analytic_projection_control_steps = 0
    command_provenance_observed_steps = 0
    command_safety_rewrite_steps = 0
    recontact_active = False
    recontact_attempt_count = 0
    recontact_approach_option_count = 0
    current_recontact_approach_options = 0
    completed_recontact_approach_option_counts: list[int] = []
    for option_index in range(300):
        mode, action, probabilities, behavior_probabilities, selection_source = _actor_action(
            bundle,
            observation,
            rng=rng,
            device=device,
            deterministic=deterministic,
            mode_exploration_fraction=mode_exploration_fraction,
            sample_continuous_action=sample_continuous_action,
            sample_actor_categorical=sample_actor_categorical,
        )
        mode_name = CONTACT_MODE_NAMES[mode]
        mode_counts[mode_name] += 1
        selection_source_counts[selection_source] += 1
        if selection_source == "uniform_allowed_exploration":
            exploration_selected_mode_counts[mode_name] += 1
        step = episode.step_option(action, mode)
        actor_eligible, proposal_execution_l2, actor_eligibility_reason = (
            _online_actor_learning_eligibility(step)
        )
        actor_eligible_options += int(actor_eligible)
        actor_learning_eligible_options_by_mode[mode_name] += int(actor_eligible)
        if not actor_eligible:
            actor_rejection_reasons[actor_eligibility_reason] += 1
        proposal_execution_l2_values.append(proposal_execution_l2)
        proposal_projection_l2_values.append(step.proposal_projection_l2)
        projection_tracking_l2_values.append(step.projection_tracking_l2)
        tracking_residual_alert_options += int(step.tracking_residual_alert)
        analytic_projection_control_steps += step.analytic_projection_control_steps
        command_provenance_observed_steps += step.command_provenance_observed_steps
        command_safety_rewrite_steps += step.command_safety_rewrite_steps
        if replay is not None:
            replay.online.add(
                observation=observation,
                next_observation=step.next_observation,
                proposed_action=step.proposed_action,
                projected_action=step.projected_action,
                executed_action=step.executed_action,
                mode=step.mode,
                reward=step.reward,
                terminal=step.terminal,
                intervention=step.intervention,
                strict_success=step.strict_success,
                stage=int(StagedPushStage.CONTACT_TRANSPORT_HOLD),
                actor_learning_eligible=actor_eligible,
                proposal_execution_l2=proposal_execution_l2,
                proposal_projection_l2=step.proposal_projection_l2,
                projection_tracking_l2=step.projection_tracking_l2,
                projection_provenance_observed=True,
                analytic_projection_applied=(
                    step.analytic_projection_control_steps > 0
                ),
                command_provenance_observed=(
                    step.command_provenance_observed_steps == step.control_steps
                ),
                command_safety_rewrite=(step.command_safety_rewrite_steps > 0),
            )
            if replay.ready:
                for _ in range(updates_per_option):
                    update_metrics.append(
                        update_hybrid_contact_sac(
                            bundle,
                            replay,
                            config=sac_config,
                            rng=rng,
                            device=device,
                        )
                    )
        observation = step.next_observation
        total_reward += step.reward
        total_valid += step.valid_contact_steps
        total_invalid += step.invalid_contact_steps
        total_effectful += step.effectful_push_steps
        valid_contact_control_steps_by_mode[mode_name] += step.valid_contact_steps
        effectful_push_control_steps_by_mode[mode_name] += step.effectful_push_steps
        reward_by_mode[mode_name] += step.reward
        for role, count in zip(
            step.contact_role_names,
            step.valid_contact_event_count_by_role,
            strict=True,
        ):
            valid_contact_events_by_role[role] += count
        for role, event_count, substep_count in zip(
            step.contact_role_names,
            step.invalid_contact_event_count_by_role,
            step.invalid_contact_substep_count_by_role,
            strict=True,
        ):
            invalid_contact_events_by_role[role] += event_count
            invalid_contact_substeps_by_role[role] += substep_count
        if step.invalid_contact_steps > 0:
            invalid_contact_control_steps_by_mode[CONTACT_MODE_NAMES[mode]] += (
                step.invalid_contact_steps
            )
            invalid_contact_option_indices.append(option_index)
        if recontact_active and mode_name == "approach":
            recontact_approach_option_count += 1
            current_recontact_approach_options += 1
            if episode.valid_contact_latched:
                completed_recontact_approach_option_counts.append(
                    current_recontact_approach_options
                )
                current_recontact_approach_options = 0
                recontact_active = False
        if mode_name == "separate_recontact":
            recontact_attempt_count += 1
            recontact_active = True
            current_recontact_approach_options = 0
        maximum_coverage = max(maximum_coverage, step.target_coverage)
        maximum_hold = max(maximum_hold, step.strict_hold_fraction)
        if step.terminal:
            break
    summary = {
        "format": STAGED_HYBRID_TRAINING_FORMAT,
        "seed": seed,
        "deterministic": deterministic,
        "strict_success": bool(step.strict_success),
        "option_count": option_index + 1,
        "control_step_count": int(episode.env.step_count),
        "return": total_reward,
        "initial_object_target_distance_m": initial_distance,
        "initial_block_xy_m": initial_block_xy.tolist(),
        "initial_target_xy_m": initial_target_xy.tolist(),
        "initial_push_direction_xy": initial_direction.tolist(),
        "initial_target_coverage": initial_coverage,
        "maximum_target_coverage": maximum_coverage,
        "final_target_coverage": step.target_coverage,
        "maximum_strict_hold_fraction": maximum_hold,
        "final_strict_hold_fraction": step.strict_hold_fraction,
        "valid_contact_control_steps": total_valid,
        "invalid_contact_control_steps": total_invalid,
        "effectful_push_control_steps": total_effectful,
        "valid_contact_control_steps_by_mode": dict(valid_contact_control_steps_by_mode),
        "effectful_push_control_steps_by_mode": dict(
            effectful_push_control_steps_by_mode
        ),
        "valid_contact_event_count_by_role": dict(valid_contact_events_by_role),
        "invalid_contact_event_count_by_role": dict(invalid_contact_events_by_role),
        "invalid_contact_substep_count_by_role": dict(invalid_contact_substeps_by_role),
        "invalid_contact_control_steps_by_mode": dict(
            invalid_contact_control_steps_by_mode
        ),
        "invalid_contact_option_count": len(invalid_contact_option_indices),
        "invalid_contact_option_indices": invalid_contact_option_indices,
        "mode_option_counts": dict(mode_counts),
        "mode_selection_source_counts": dict(selection_source_counts),
        "uniform_exploration_selected_mode_counts": dict(
            exploration_selected_mode_counts
        ),
        "mode_exploration_fraction": mode_exploration_fraction,
        "continuous_action_sampling": (
            "gaussian_residual"
            if sample_continuous_action and not deterministic
            else "deterministic_mode_mean"
        ),
        "actor_mode_sampling": (
            "categorical" if sample_actor_categorical and not deterministic else "argmax"
        ),
        "final_mode_probabilities": probabilities.tolist(),
        "final_actor_mode_probabilities": probabilities.tolist(),
        "final_behavior_mode_probabilities": behavior_probabilities.tolist(),
        "terminal_reason": step.failure_reason,
        "intervention_terminal": bool(step.intervention),
        "sac_updates": len(update_metrics),
        "actor_learning_eligible_options": actor_eligible_options,
        "actor_learning_eligible_options_by_mode": dict(
            actor_learning_eligible_options_by_mode
        ),
        "reward_by_mode": {
            name: float(value) for name, value in reward_by_mode.items()
        },
        "recontact_attempt_count": recontact_attempt_count,
        "recontact_completed_count": len(completed_recontact_approach_option_counts),
        "recontact_unfinished_count": int(recontact_active),
        "recontact_approach_option_count": recontact_approach_option_count,
        "completed_recontact_approach_option_counts": (
            completed_recontact_approach_option_counts
        ),
        "mean_completed_recontact_approach_options": (
            float(np.mean(completed_recontact_approach_option_counts))
            if completed_recontact_approach_option_counts
            else 0.0
        ),
        "maximum_completed_recontact_approach_options": (
            max(completed_recontact_approach_option_counts)
            if completed_recontact_approach_option_counts
            else 0
        ),
        "actor_learning_eligibility_rate": actor_eligible_options / (option_index + 1),
        "actor_learning_rejection_reasons": dict(actor_rejection_reasons),
        "mean_proposal_execution_l2": float(np.mean(proposal_execution_l2_values)),
        "maximum_proposal_execution_l2": float(np.max(proposal_execution_l2_values)),
        "mean_proposal_projection_l2": float(
            np.mean(proposal_projection_l2_values)
        ),
        "maximum_proposal_projection_l2": float(
            np.max(proposal_projection_l2_values)
        ),
        "mean_projection_tracking_l2": float(
            np.mean(projection_tracking_l2_values)
        ),
        "maximum_projection_tracking_l2": float(
            np.max(projection_tracking_l2_values)
        ),
        "tracking_residual_alert_option_count": tracking_residual_alert_options,
        "analytic_projection_control_steps": analytic_projection_control_steps,
        "command_provenance_observed_steps": command_provenance_observed_steps,
        "command_provenance_missing_steps": (
            int(episode.env.step_count) - command_provenance_observed_steps
        ),
        "command_safety_rewrite_steps": command_safety_rewrite_steps,
        "production_admission": False,
        "wrist_multimodal_export_started": False,
        "act_training_started": False,
    }
    summary["scene_mode"] = scene_mode
    if episode.multichoice is not None:
        summary["multichoice_scene"] = episode.multichoice.audit()
    if trajectory_path is not None:
        np.savez_compressed(trajectory_path, time=np.asarray([x[0] for x in trace]),
                            qpos=np.stack([x[1] for x in trace]), qvel=np.stack([x[2] for x in trace]),
                            actuator_ctrl=np.stack([x[3] for x in trace]),
                            submitted_joint_command=np.stack([x[4] for x in trace]))
        summary["trajectory_path"] = str(trajectory_path)
        summary["trajectory_samples"] = len(trace)
    return summary, update_metrics


def evaluate_stage_one_policy(
    bundle: HybridContactSACBundle,
    *,
    seeds: list[int],
    rng: np.random.Generator,
    device: str | torch.device,
    scene_mode: str = "single",
) -> dict[str, Any]:
    episodes: list[dict[str, Any]] = []
    for seed in seeds:
        summary, _updates = run_learned_stage_one_episode(
            bundle,
            seed=seed,
            rng=rng,
            device=device,
            deterministic=True,
            scene_mode=scene_mode,
        )
        episodes.append(summary)
    successes = sum(int(row["strict_success"]) for row in episodes)
    invalid = sum(int(row["invalid_contact_control_steps"]) for row in episodes)
    interventions = sum(int(row["intervention_terminal"]) for row in episodes)
    safe_successes = sum(
        int(
            row["strict_success"]
            and row["invalid_contact_control_steps"] == 0
            and not row["intervention_terminal"]
        )
        for row in episodes
    )
    invalid_episodes = sum(
        int(row["invalid_contact_control_steps"] > 0) for row in episodes
    )
    invalid_events_by_role: Counter[str] = Counter()
    invalid_substeps_by_role: Counter[str] = Counter()
    invalid_steps_by_mode: Counter[str] = Counter()
    for row in episodes:
        invalid_events_by_role.update(row["invalid_contact_event_count_by_role"])
        invalid_substeps_by_role.update(row["invalid_contact_substep_count_by_role"])
        invalid_steps_by_mode.update(row["invalid_contact_control_steps_by_mode"])
    return {
        "format": STAGED_HYBRID_TRAINING_FORMAT,
        "evaluation_kind": "nominal_task_aligned_stage_one_only",
        "seeds": seeds,
        "episode_count": len(episodes),
        "strict_success_count": successes,
        "strict_success_rate": successes / len(episodes),
        "strict_success_wilson_lower_95": _wilson_lower(successes, len(episodes)),
        "safe_strict_success_count": safe_successes,
        "safe_strict_success_rate": safe_successes / len(episodes),
        "safe_strict_success_wilson_lower_95": _wilson_lower(
            safe_successes, len(episodes)
        ),
        "invalid_contact_control_steps": invalid,
        "invalid_contact_episode_count": invalid_episodes,
        "invalid_contact_event_count_by_role": dict(invalid_events_by_role),
        "invalid_contact_substep_count_by_role": dict(invalid_substeps_by_role),
        "invalid_contact_control_steps_by_mode": dict(invalid_steps_by_mode),
        "intervention_terminal_count": interventions,
        "mean_maximum_target_coverage": float(
            np.mean([row["maximum_target_coverage"] for row in episodes])
        ),
        "mean_maximum_strict_hold_fraction": float(
            np.mean([row["maximum_strict_hold_fraction"] for row in episodes])
        ),
        "episodes": episodes,
        "stage_one_smoke_mastered": bool(
            len(episodes) >= 8
            and successes / len(episodes) >= 0.75
            and invalid == 0
            and interventions == 0
        ),
        "exact_home_evaluated": False,
        "production_admission": False,
        "wrist_multimodal_export_started": False,
        "act_training_started": False,
    }


def train(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir).expanduser().resolve()
    initial_checkpoint = (
        Path(args.initial_checkpoint).expanduser().resolve()
        if args.initial_checkpoint
        else None
    )
    initial_prior_replay = (
        Path(args.initial_prior_replay).expanduser().resolve()
        if args.initial_prior_replay
        else None
    )
    importing_initial_state = initial_checkpoint is not None
    actor_only = bool(args.initial_actor_only)
    if initial_prior_replay is not None and initial_checkpoint is None:
        raise ValueError(
            "initial prior replay requires an initial checkpoint"
        )
    if actor_only and (initial_checkpoint is None or initial_prior_replay is not None):
        raise ValueError("actor-only restoration requires a checkpoint and newly collected prior")
    if initial_checkpoint is not None and initial_prior_replay is None and not actor_only:
        raise ValueError("checkpoint without prior requires explicit --initial-actor-only")
    if importing_initial_state:
        if not initial_checkpoint.is_file() or (initial_prior_replay is not None and not initial_prior_replay.is_file()):
            raise FileNotFoundError("initial checkpoint or prior replay does not exist")
        if args.bc_updates != 0:
            raise ValueError("imported checkpoint requires --bc-updates 0")
    output_dir.mkdir(parents=True, exist_ok=False)
    device = torch.device(args.device)
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    config = HybridContactSACConfig(
        observation_dim=STAGED_PUSH_OBSERVATION_DIM,
        mode_mask_start=STAGED_PUSH_MODE_MASK_START,
        hidden_dim=args.hidden_dim,
        batch_size=args.batch_size,
        prior_capacity=args.prior_capacity,
        online_capacity=args.online_capacity,
        actor_learning_rate=args.actor_learning_rate,
        online_actor_learning_rate=args.online_actor_learning_rate,
        online_mode_exploration_fraction=args.online_mode_exploration_fraction,
        online_sample_actor_categorical=not args.online_deterministic_actor_mode,
        online_sample_continuous_action=(
            not args.online_deterministic_continuous_action
        ),
        exploration_log_std=args.exploration_log_std,
        actor_update_delay=args.actor_update_delay,
        actor_update_interval=args.actor_update_interval,
        minimum_online_advantage=args.minimum_online_advantage,
        minimum_online_actor_samples=args.minimum_online_actor_samples,
        minimum_online_actor_evidence_per_mode=(
            args.minimum_online_actor_evidence_per_mode
        ),
        minimum_online_actor_recovery_modes=args.minimum_online_actor_recovery_modes,
        minimum_online_actor_effective_sample_size=args.minimum_online_actor_ess,
        online_advantage_regression_coefficient=args.online_actor_coefficient,
        maximum_actor_batch_action_delta=args.maximum_actor_batch_action_delta,
    )
    config.validate()
    run_plan = {
        "format": STAGED_HYBRID_TRAINING_FORMAT,
        "created_at_utc": _utc_now(),
        "seed": args.seed,
        "scene_mode": args.scene_mode,
        "device": str(device),
        "prior_success_episode_target": args.prior_successes,
        "maximum_prior_attempts": args.maximum_prior_attempts,
        "balanced_behavior_cloning_updates": args.bc_updates,
        "initial_actor_only": actor_only,
        "reward_configuration": asdict(StagedPushRewardConfig()),
        "executor_configuration": asdict(StagedPushExecutorConfig()),
        "source_code_sha256": {
            name: _file_sha256(Path(__file__).with_name(name))
            for name in ("staged_push_rl.py", "hybrid_contact_sac.py", "train_staged_hybrid_contact_sac.py")
        },
        "initial_checkpoint": (
            str(initial_checkpoint) if initial_checkpoint is not None else None
        ),
        "initial_checkpoint_sha256": (
            _file_sha256(initial_checkpoint) if initial_checkpoint is not None else None
        ),
        "initial_prior_replay": (
            str(initial_prior_replay) if initial_prior_replay is not None else None
        ),
        "initial_prior_replay_sha256": (
            _file_sha256(initial_prior_replay)
            if initial_prior_replay is not None
            else None
        ),
        "online_episodes": args.online_episodes,
        "updates_per_option": args.updates_per_option,
        "evaluation_interval_episodes": args.evaluation_interval,
        "evaluation_seed_count": args.evaluation_count,
        "sac_configuration": asdict(config),
        "training_stage": StagedPushStage.CONTACT_TRANSPORT_HOLD.name,
        "nominal_no_obstacle_only": True,
        "strict_hold_control_steps": 90,
        "option_control_steps": 4,
        "exact_symmetric_prior_online_sampling": True,
        "critic_action_source": "policy_proposal",
        "shield_intervention_transition": "terminal_surrogate_mdp",
        "actor_update_rule": "delayed_positive_advantage_weighted_regression",
        "actor_advantage_authority": "frozen_reference_actor_and_target_twin_critic",
        "online_actor_rows_require_executor_provenance": True,
        "fixed_seed_policy_regression_gate": "rollback_actor_unless_lexicographically_better",
        "full_v4_guard_used_during_training": False,
        "full_v4_guard_reserved_for_strict_evaluation": True,
        "exact_home_evaluated": False,
        "production_admission": False,
        "wrist_multimodal_export_started": False,
        "act_training_started": False,
    }
    _atomic_json(output_dir / "run_plan.json", run_plan)
    _atomic_json(
        output_dir / "run_state.json",
        {
            "format": STAGED_HYBRID_TRAINING_FORMAT,
            "status": (
                "importing_prior_and_checkpoint"
                if importing_initial_state
                else "collecting_successful_prior"
            ),
            "updated_at_utc": _utc_now(),
            "production_admission": False,
        },
    )

    replay = SymmetricPriorOnlineReplay(config)
    if initial_prior_replay is not None:
        prior_payload = torch.load(
            initial_prior_replay,
            map_location="cpu",
            weights_only=False,
        )
        if (
            not isinstance(prior_payload, dict)
            or prior_payload.get("format") != STAGED_HYBRID_TRAINING_FORMAT
            or prior_payload.get("production_admission") is not False
            or not isinstance(prior_payload.get("replay"), dict)
        ):
            raise ValueError("initial prior replay wrapper is invalid")
        replay.prior.load_state_dict(prior_payload["replay"])
        prior_successes = prior_payload.get("prior_success_count")
        prior_attempts = prior_payload.get("prior_attempt_count")
        if (
            type(prior_successes) is not int
            or prior_successes < 1
            or type(prior_attempts) is not int
            or prior_attempts < prior_successes
            or not replay.prior.frozen
            or replay.prior.size < config.batch_size // 2
        ):
            raise ValueError("initial prior replay evidence is insufficient")
        progress = {
            "phase": "initial_state_import",
            "prior_successes": prior_successes,
            "prior_attempts": prior_attempts,
            "prior_transition_count": replay.prior.size,
            "legacy_actor_rows_quarantined": int(
                np.count_nonzero(~replay.prior.actor_learning_eligible[: replay.prior.size])
            ),
            "updated_at_utc": _utc_now(),
        }
        _atomic_json(output_dir / "episode_progress.json", progress)
        print(json.dumps(progress, sort_keys=True, allow_nan=False), flush=True)
    else:
        prior_successes = 0
        prior_attempts = 0
        while (
            prior_successes < args.prior_successes
            and prior_attempts < args.maximum_prior_attempts
        ):
            episode_seed = args.seed + 10_000 + prior_attempts
            summary = collect_successful_prior_episode(replay.prior, seed=episode_seed, scene_mode=args.scene_mode)
            prior_attempts += 1
            prior_successes += int(summary["episode_admitted_to_successful_prior"])
            _append_jsonl(output_dir / "prior_episodes.jsonl", summary)
            progress = {
                "phase": "prior_collection",
                "attempts": prior_attempts,
                "strict_successes": prior_successes,
                "prior_transition_count": replay.prior.size,
                "latest": summary,
                "updated_at_utc": _utc_now(),
            }
            _atomic_json(output_dir / "episode_progress.json", progress)
            print(json.dumps(progress, sort_keys=True, allow_nan=False), flush=True)
        if prior_successes < args.prior_successes:
            raise RuntimeError(
                f"successful prior collection exhausted: {prior_successes}/{args.prior_successes}"
            )
        replay.prior.freeze()
    torch.save(
        {
            "format": STAGED_HYBRID_TRAINING_FORMAT,
            "prior_success_count": prior_successes,
            "prior_attempt_count": prior_attempts,
            "replay": replay.prior.state_dict(),
            "production_admission": False,
        },
        output_dir / "prior_replay.pt",
    )
    bundle = initialize_hybrid_contact_sac(args.seed, device=device, config=config)
    if importing_initial_state:
        checkpoint_payload = torch.load(
            initial_checkpoint,
            map_location=device,
            weights_only=False,
        )
        if (
            not isinstance(checkpoint_payload, dict)
            or checkpoint_payload.get("production_admission") is not False
            or checkpoint_payload.get("wrist_multimodal_export_started") is not False
            or checkpoint_payload.get("act_training_started") is not False
        ):
            raise ValueError("initial checkpoint safety gates are invalid")
        restore_bundle = (
            initialize_hybrid_contact_sac(args.seed, device=device, config=config)
            if actor_only else bundle
        )
        load_hybrid_bundle_state_dict(
            restore_bundle,
            checkpoint_payload,
            config=config,
            load_actor_optimizer=False,
            load_critic_optimizer=not actor_only,
        )
        if actor_only:
            # Preserve the known policy, but relearn Q under the current reward
            # and fresh replay contract.  Never mix old rewards with new ones.
            bundle.actor.load_state_dict(restore_bundle.actor.state_dict(), strict=True)
            synchronize_reference_actor(bundle)
    _atomic_json(
        output_dir / "run_state.json",
        {
            "format": STAGED_HYBRID_TRAINING_FORMAT,
            "status": (
                "imported_checkpoint_ready"
                if importing_initial_state
                else "balanced_prior_warm_start"
            ),
            "prior_successes": prior_successes,
            "prior_attempts": prior_attempts,
            "prior_transition_count": replay.prior.size,
            "updated_at_utc": _utc_now(),
            "production_admission": False,
        },
    )
    latest_bc: dict[str, float] = {}
    for update_index in range(args.bc_updates):
        latest_bc = pretrain_hybrid_actor_from_prior(
            bundle,
            replay.prior,
            config=config,
            rng=rng,
            device=device,
        )
        if (update_index + 1) % args.progress_interval == 0 or update_index + 1 == args.bc_updates:
            progress = {
                "phase": "balanced_prior_warm_start",
                "update": update_index + 1,
                "target_updates": args.bc_updates,
                "metrics": latest_bc,
                "updated_at_utc": _utc_now(),
            }
            _atomic_json(output_dir / "episode_progress.json", progress)
            print(json.dumps(progress, sort_keys=True, allow_nan=False), flush=True)

    # Online policy improvement starts from a frozen accepted reference.  The
    # warm-start optimizer moments are intentionally discarded so they cannot
    # leak into the delayed conservative update phase.
    synchronize_reference_actor(bundle)
    _reset_actor_optimizer(bundle, config)

    evaluation_seeds = [args.seed + 1_000_000 + index for index in range(args.evaluation_count)]
    preonline_evaluation = evaluate_stage_one_policy(
        bundle,
        seeds=evaluation_seeds,
        rng=rng,
        device=device,
        scene_mode=args.scene_mode,
    )
    _atomic_json(output_dir / "evaluation_preonline.json", preonline_evaluation)
    best_rank = _evaluation_rank(preonline_evaluation)
    best_evaluation = preonline_evaluation
    torch.save(hybrid_bundle_state_dict(bundle, config), output_dir / "checkpoint_best.pt")

    _atomic_json(
        output_dir / "run_state.json",
        {
            "format": STAGED_HYBRID_TRAINING_FORMAT,
            "status": "online_rlpd_training",
            "prior_successes": prior_successes,
            "prior_transition_count": replay.prior.size,
            "preonline_strict_success_count": preonline_evaluation["strict_success_count"],
            "updated_at_utc": _utc_now(),
            "production_admission": False,
        },
    )
    total_sac_updates = 0
    latest_update: dict[str, float | int | bool | str] = {}
    actor_policy_gate_rejections = 0
    initial_actor_update_count = bundle.actor_update_count
    last_gated_actor_update_count = initial_actor_update_count
    for episode_index in range(args.online_episodes):
        episode_seed = args.seed + 2_000_000 + episode_index
        summary, updates = run_learned_stage_one_episode(
            bundle,
            seed=episode_seed,
            rng=rng,
            device=device,
            deterministic=False,
            replay=replay,
            sac_config=config,
            updates_per_option=args.updates_per_option,
            mode_exploration_fraction=config.online_mode_exploration_fraction,
            sample_continuous_action=config.online_sample_continuous_action,
            sample_actor_categorical=config.online_sample_actor_categorical,
            scene_mode=args.scene_mode,
        )
        total_sac_updates += len(updates)
        if updates:
            latest_update = updates[-1]
        for update in updates:
            if bool(update["actor_update_scheduled"]):
                _append_jsonl(
                    output_dir / "actor_updates.jsonl",
                    {
                        **update,
                        "online_episode": episode_index + 1,
                        "logged_at_utc": _utc_now(),
                    },
                )
        summary["episode_index"] = episode_index + 1
        summary["total_sac_updates"] = total_sac_updates
        _append_jsonl(output_dir / "online_episodes.jsonl", summary)
        progress = {
            "phase": "online_rlpd_training",
            "episode": episode_index + 1,
            "target_episodes": args.online_episodes,
            "online_transition_count": replay.online.size,
            "total_sac_updates": total_sac_updates,
            "latest_update": latest_update,
            "latest_episode": summary,
            "updated_at_utc": _utc_now(),
        }
        _atomic_json(output_dir / "episode_progress.json", progress)
        print(json.dumps(progress, sort_keys=True, allow_nan=False), flush=True)
        if (episode_index + 1) % args.evaluation_interval == 0:
            candidate_checkpoint_name = f"checkpoint_candidate_episode_{episode_index + 1:04d}.pt"
            torch.save(
                hybrid_bundle_state_dict(bundle, config),
                output_dir / candidate_checkpoint_name,
            )
            torch.save(
                {
                    "format": STAGED_HYBRID_TRAINING_FORMAT,
                    "after_online_episode": episode_index + 1,
                    "replay": replay.online.state_dict(),
                    "production_admission": False,
                },
                output_dir / "online_replay_latest.pt",
            )
            evaluation = evaluate_stage_one_policy(
                bundle,
                seeds=evaluation_seeds,
                rng=rng,
                device=device,
                scene_mode=args.scene_mode,
            )
            evaluation["after_online_episode"] = episode_index + 1
            _atomic_json(
                output_dir / f"evaluation_episode_{episode_index + 1:04d}.json",
                evaluation,
            )
            candidate_rank = _evaluation_rank(evaluation)
            actor_changed_since_gate = bundle.actor_update_count > last_gated_actor_update_count
            accepted = actor_changed_since_gate and candidate_rank > best_rank
            gate_record = {
                "format": STAGED_HYBRID_TRAINING_FORMAT,
                "after_online_episode": episode_index + 1,
                "candidate_rank": list(candidate_rank),
                "incumbent_rank": list(best_rank),
                "accepted": accepted,
                "actor_changed_since_previous_gate": actor_changed_since_gate,
                "actor_update_count": bundle.actor_update_count,
                "candidate_checkpoint": candidate_checkpoint_name,
                "updated_at_utc": _utc_now(),
                "production_admission": False,
            }
            if not actor_changed_since_gate:
                gate_record["decision"] = "no_actor_change_keep_incumbent"
            elif accepted:
                best_rank = candidate_rank
                best_evaluation = evaluation
                synchronize_reference_actor(bundle)
                _reset_actor_optimizer(bundle, config)
                torch.save(
                    hybrid_bundle_state_dict(bundle, config),
                    output_dir / "checkpoint_best.pt",
                )
                gate_record["decision"] = "accept_and_advance_reference_actor"
            else:
                actor_policy_gate_rejections += 1
                best_state = torch.load(
                    output_dir / "checkpoint_best.pt",
                    map_location=device,
                    weights_only=False,
                )
                bundle.actor.load_state_dict(best_state["actor"], strict=True)
                synchronize_reference_actor(bundle)
                _reset_actor_optimizer(bundle, config)
                gate_record["decision"] = "reject_and_restore_best_actor"
            last_gated_actor_update_count = bundle.actor_update_count
            _append_jsonl(output_dir / "policy_gates.jsonl", gate_record)
    torch.save(
        {
            "format": STAGED_HYBRID_TRAINING_FORMAT,
            "after_online_episode": args.online_episodes,
            "replay": replay.online.state_dict(),
            "production_admission": False,
        },
        output_dir / "online_replay_latest.pt",
    )
    torch.save(hybrid_bundle_state_dict(bundle, config), output_dir / "checkpoint_latest.pt")
    final_evaluation = evaluate_stage_one_policy(
        bundle,
        seeds=evaluation_seeds,
        rng=rng,
        device=device,
        scene_mode=args.scene_mode,
    )
    _atomic_json(output_dir / "evaluation_final.json", final_evaluation)
    _atomic_json(output_dir / "evaluation_best.json", best_evaluation)
    summary = {
        "format": STAGED_HYBRID_TRAINING_FORMAT,
        "status": "complete",
        "completed_at_utc": _utc_now(),
        "prior_success_count": prior_successes,
        "prior_attempt_count": prior_attempts,
        "prior_transition_count": replay.prior.size,
        "online_episode_count": args.online_episodes,
        "online_transition_count": replay.online.size,
        "total_sac_updates": total_sac_updates,
        "total_actor_updates": bundle.actor_update_count - initial_actor_update_count,
        "cumulative_actor_updates": bundle.actor_update_count,
        "initial_actor_updates": initial_actor_update_count,
        "actor_policy_gate_rejections": actor_policy_gate_rejections,
        "preonline_evaluation": {
            key: preonline_evaluation[key]
            for key in (
                "strict_success_count",
                "safe_strict_success_count",
                "invalid_contact_control_steps",
                "episode_count",
                "strict_success_rate",
                "mean_maximum_target_coverage",
            )
        },
        "best_evaluation": {
            key: best_evaluation[key]
            for key in (
                "strict_success_count",
                "safe_strict_success_count",
                "invalid_contact_control_steps",
                "episode_count",
                "strict_success_rate",
                "strict_success_wilson_lower_95",
                "mean_maximum_target_coverage",
                "mean_maximum_strict_hold_fraction",
                "stage_one_smoke_mastered",
            )
        },
        "final_evaluation": {
            key: final_evaluation[key]
            for key in (
                "strict_success_count",
                "safe_strict_success_count",
                "invalid_contact_control_steps",
                "episode_count",
                "strict_success_rate",
                "strict_success_wilson_lower_95",
                "mean_maximum_target_coverage",
                "mean_maximum_strict_hold_fraction",
                "stage_one_smoke_mastered",
            )
        },
        "stage_one_only": True,
        "exact_home_evaluated": False,
        "production_admission": False,
        "wrist_multimodal_export_started": False,
        "act_training_started": False,
    }
    _atomic_json(output_dir / "summary.json", summary)
    _atomic_json(
        output_dir / "run_state.json",
        {
            **summary,
            "updated_at_utc": _utc_now(),
        },
    )
    print(json.dumps(summary, sort_keys=True, allow_nan=False), flush=True)
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--scene-mode", choices=("single", "multichoice_v1"), default="single")
    parser.add_argument("--seed", type=int, default=830_000)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--prior-capacity", type=int, default=200_000)
    parser.add_argument("--online-capacity", type=int, default=500_000)
    parser.add_argument("--actor-learning-rate", type=float, default=3.0e-4)
    parser.add_argument("--online-actor-learning-rate", type=float, default=3.0e-5)
    parser.add_argument("--online-mode-exploration-fraction", type=float, default=0.10)
    parser.add_argument("--online-deterministic-actor-mode", action="store_true")
    parser.add_argument("--online-deterministic-continuous-action", action="store_true")
    parser.add_argument("--exploration-log-std", type=float, default=-2.75)
    parser.add_argument("--actor-update-delay", type=int, default=256)
    parser.add_argument("--actor-update-interval", type=int, default=4)
    parser.add_argument("--minimum-online-advantage", type=float, default=0.05)
    parser.add_argument("--minimum-online-actor-samples", type=int, default=8)
    parser.add_argument("--minimum-online-actor-evidence-per-mode", type=int, default=4)
    parser.add_argument("--minimum-online-actor-recovery-modes", type=int, default=0)
    parser.add_argument("--minimum-online-actor-ess", type=float, default=4.0)
    parser.add_argument("--online-actor-coefficient", type=float, default=0.25)
    parser.add_argument("--maximum-actor-batch-action-delta", type=float, default=0.02)
    parser.add_argument("--initial-checkpoint")
    parser.add_argument("--initial-actor-only", action="store_true")
    parser.add_argument("--initial-prior-replay")
    parser.add_argument("--prior-successes", type=int, default=32)
    parser.add_argument("--maximum-prior-attempts", type=int, default=64)
    parser.add_argument("--bc-updates", type=int, default=2_000)
    parser.add_argument("--online-episodes", type=int, default=64)
    parser.add_argument("--updates-per-option", type=int, default=1)
    parser.add_argument("--evaluation-interval", type=int, default=8)
    parser.add_argument("--evaluation-count", type=int, default=8)
    parser.add_argument("--progress-interval", type=int, default=100)
    return parser


def main() -> None:
    args = _parser().parse_args()
    try:
        train(args)
    except KeyboardInterrupt:
        destination = Path(args.output_dir).expanduser().resolve()
        if destination.exists():
            _atomic_json(
                destination / "manual_stop_audit.json",
                {
                    "format": STAGED_HYBRID_TRAINING_FORMAT,
                    "status": "interrupted",
                    "reason": "operator_keyboard_interrupt",
                    "updated_at_utc": _utc_now(),
                    "production_admission": False,
                    "wrist_multimodal_export_started": False,
                    "act_training_started": False,
                },
            )
        raise
    except Exception as error:
        destination = Path(args.output_dir).expanduser().resolve()
        if destination.exists():
            _atomic_json(
                destination / "run_state.json",
                {
                    "format": STAGED_HYBRID_TRAINING_FORMAT,
                    "status": "failed",
                    "error_type": type(error).__name__,
                    "error_message": str(error),
                    "updated_at_utc": _utc_now(),
                    "production_admission": False,
                    "wrist_multimodal_export_started": False,
                    "act_training_started": False,
                },
            )
        raise


if __name__ == "__main__":
    main()


__all__ = [
    "STAGED_HYBRID_TRAINING_FORMAT",
    "evaluate_stage_one_policy",
    "run_learned_stage_one_episode",
    "train",
]
