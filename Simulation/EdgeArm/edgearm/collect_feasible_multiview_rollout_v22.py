"""Collect and admission-audit one true RGB V22 simulator RL rollout.

The actor is sampled online from its own causal state distribution.  Wrist,
front, angled, and overhead RGB plus joint/action histories are persisted by
the shared V13 writer, then relabelled with the exact V22 96/8/88 execution
contract.  The rollout is always retained as evidence; it becomes PPO-update
eligible only if the V21 contact, motion, executability, and projection gates
all pass.  This command performs no optimizer step.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
import math
from pathlib import Path
from typing import Any, Sequence

import h5py
import numpy as np
import torch

from .asymmetric_multiview_ppo_v1 import (
    CAUSAL_VISUAL_INFERENCE_CACHE_FORMAT_V24,
    H5_FORMAT,
    SOURCE_TYPE,
    MultiViewRGBRendererV1,
    canonical_sha256_v1,
    collect_asymmetric_multiview_rollout_v1,
    initialize_asymmetric_multiview_ppo_v1,
    parameter_counts_v1,
    sha256_file_v1,
    write_online_rollout_h5_v1,
)
from .evaluate_asymmetric_drq_sac_v1 import _configs_from_parent_plan
from .dynamic_language_v26 import persist_dynamic_language_v26
from .feasible_on_policy_ppo_v21 import (
    RolloutEligibilityConfigV21,
    build_batch_projection_audit_v21,
    rollout_update_gates_v21,
)
from .ppo_utils_v1 import state_dict_sha256_v1
from .reverse_curriculum_v26 import (
    REVERSE_CURRICULUM_FORMAT_V26,
    STRICT_SUCCESS_HOLD_SECONDS_V26,
    reverse_curriculum_stage_v26,
    strict_success_hold_steps_v26,
)
from .sim2real_env_v10 import RealisticEdgeArmEnvV10
from .source_closure_v26 import build_behavior_source_closure_v26
from .stock_gripper_reward_v22 import (
    STOCK_GRIPPER_REWARD_FORMAT_V22,
    StockGripperPotentialRewardV22,
)
from .stock_gripper_rollout_kernel_v22 import (
    STOCK_GRIPPER_ROLLOUT_FORMAT_V22,
    STOCK_GRIPPER_ROLLOUT_KERNEL_FORMAT_V22,
    StockGripperRolloutKernelV22,
)
from .stock_gripper_taskframe_v22 import StockGripperTaskFrameAdapterV22
from .train_on_policy_recurrent_ppo_v20 import (
    _atomic_json,
    _mean_network_sha256,
    _resolve_device,
    _utc_now,
)
from .v23_checkpoint_lineage import load_collection_parent_actor_critic_v23


FEASIBLE_MULTIVIEW_ROLLOUT_RUN_FORMAT_V22 = "edgearm-v22-feasible-multiview-rollout-admission-run-v1"
FEASIBLE_MULTIVIEW_H5_FORMAT_V22 = "edgearm-v22-feasible-multiview-trajectory-h5-v1"
CAUSAL_HISTORY_DIRECT_PERSISTENCE_FORMAT_V22 = "edgearm-v22-direct-causal-history-persistence-v1"
CAUSAL_HISTORY_DATASET_NAMES_V22 = (
    "policy_rgb_history",
    "policy_joint_history",
    "policy_action_history",
    "history_valid",
    "policy_view_history_valid",
)
CONDITION_SCHEDULES_V24: dict[str, tuple[tuple[bool, bool], ...] | None] = {
    "random": None,
    "ordinary_obstacle": ((False, False), (True, False)),
    "four_way": ((False, False), (False, True), (True, False), (True, True)),
}
PPO_PROPOSAL_STRENGTH_FORMAT_V40 = "edgearm-v40-audited-ppo-proposal-strength-v1"
CONSERVATIVE_PPO_LEARNING_RATE_V22 = 2.0e-6
CONSERVATIVE_PPO_UPDATE_EPOCHS_V22 = 1
MAXIMUM_PPO_LEARNING_RATE_V40 = 2.0e-4
MAXIMUM_PPO_UPDATE_EPOCHS_V40 = 8


def audited_ppo_proposal_strength_v40(
    *,
    learning_rate: float,
    update_epochs: int,
) -> dict[str, object]:
    """Validate proposal strength while retaining every closed-loop gate."""

    if isinstance(learning_rate, bool) or not isinstance(learning_rate, (int, float)):
        raise ValueError("V40 PPO learning rate must be a real scalar")
    learning_rate = float(learning_rate)
    if (
        not math.isfinite(learning_rate)
        or learning_rate <= 0.0
        or learning_rate > MAXIMUM_PPO_LEARNING_RATE_V40
    ):
        raise ValueError("V40 PPO learning rate is outside the audited range")
    if (
        type(update_epochs) is not int
        or update_epochs < 1
        or update_epochs > MAXIMUM_PPO_UPDATE_EPOCHS_V40
    ):
        raise ValueError("V40 PPO update epochs are outside the audited range")
    nominal_ratio = (
        learning_rate
        * update_epochs
        / (
            CONSERVATIVE_PPO_LEARNING_RATE_V22
            * CONSERVATIVE_PPO_UPDATE_EPOCHS_V22
        )
    )
    return {
        "format": PPO_PROPOSAL_STRENGTH_FORMAT_V40,
        "learning_rate": learning_rate,
        "update_epochs": update_epochs,
        "conservative_reference_learning_rate": CONSERVATIVE_PPO_LEARNING_RATE_V22,
        "conservative_reference_update_epochs": CONSERVATIVE_PPO_UPDATE_EPOCHS_V22,
        "nominal_optimizer_exposure_ratio": nominal_ratio,
        "ratio_is_not_a_closed_loop_effect_claim": True,
        "paired_same_seed_gate_retained": True,
        "meaningful_effect_gate_retained": True,
        "safety_gate_retained": True,
        "production_admission": False,
    }


def _validate_configuration_v22(
    *,
    rollout_seed: int,
    initialization_seed: int,
    rollout_steps: int,
    batch_size: int,
    exploration_standard_deviation: float,
    action_autoregressive_rho: float,
    reset_height_m: float,
    curriculum_tip_gap_band_m: tuple[float, float],
    obstacle_probability: float,
    stress_probability: float,
) -> None:
    if type(rollout_seed) is not int or rollout_seed < 0:
        raise ValueError("V22 rollout seed must be non-negative")
    if type(initialization_seed) is not int or initialization_seed < 0:
        raise ValueError("V22 initialization seed must be non-negative")
    if type(rollout_steps) is not int or rollout_steps < 32:
        raise ValueError("V22 rollout must contain at least 32 transitions")
    if type(batch_size) is not int or not 16 <= batch_size <= rollout_steps:
        raise ValueError("V22 batch size must be in [16,rollout_steps]")
    numeric = np.asarray(
        [
            exploration_standard_deviation,
            action_autoregressive_rho,
            reset_height_m,
            obstacle_probability,
            stress_probability,
            *curriculum_tip_gap_band_m,
        ],
        dtype=np.float64,
    )
    if not np.all(np.isfinite(numeric)):
        raise ValueError("V22 rollout configuration is non-finite")
    if not 0.05 <= exploration_standard_deviation <= 0.50:
        raise ValueError("V22 exploration standard deviation is outside [0.05,0.50]")
    if not 0.0 <= action_autoregressive_rho < 0.99:
        raise ValueError("V22 autoregressive rho is outside [0,0.99)")
    if not 0.047 <= reset_height_m <= 0.056:
        raise ValueError("V22 reset height escaped the audited stock band")
    if (
        curriculum_tip_gap_band_m[0] < 0.00025
        or curriculum_tip_gap_band_m[0] >= curriculum_tip_gap_band_m[1]
        or curriculum_tip_gap_band_m[1] > 0.020
    ):
        raise ValueError("V22 curriculum tip-gap band is invalid")
    if not 0.0 <= obstacle_probability <= 1.0:
        raise ValueError("V22 obstacle probability is outside [0,1]")
    if not 0.0 <= stress_probability <= 1.0:
        raise ValueError("V22 stress probability is outside [0,1]")


def _write_collection_state_v26(
    destination: Path,
    phase: str,
    **payload: Any,
) -> None:
    _atomic_json(
        destination / "run_state.json",
        {
            "format": FEASIBLE_MULTIVIEW_ROLLOUT_RUN_FORMAT_V22,
            "status": (
                "complete"
                if phase == "complete"
                else ("failed" if phase == "failed" else "running")
            ),
            "phase": phase,
            "updated_at_utc": _utc_now(),
            "production_admission": False,
            **payload,
        },
    )


def _relabel_h5_v22(
    path: Path,
    *,
    run_plan_sha256: str,
    actor_state_sha256: str,
    parent_checkpoint: Path,
    exploration_standard_deviation: float,
    projection_audit: dict[str, Any],
    eligibility_checks: dict[str, bool],
    condition_schedule: str,
    strict_success_contract: dict[str, Any],
) -> dict[str, Any]:
    eligible = all(eligibility_checks.values())
    with h5py.File(path, "r+") as stream:
        if str(stream.attrs.get("format", "")) != H5_FORMAT:
            raise ValueError("V22 relabel input is not the shared V13 writer")
        execution = stream.get("execution")
        policy = stream.get("policy_observation")
        if not isinstance(execution, h5py.Group):
            raise RuntimeError("V22 rollout lost its execution group")
        if not isinstance(policy, h5py.Group):
            raise RuntimeError("V22 rollout lost its policy-observation group")
        for name in CAUSAL_HISTORY_DATASET_NAMES_V22:
            if name not in policy or not bool(policy[name].attrs.get("policy_input_eligible", False)):
                raise RuntimeError(f"V22 rollout lost directly persisted causal history: {name}")
        neutral = "minimum_executed_safety_only_block_clearance_m"
        compatibility = "minimum_executed_94_safety_only_block_clearance_m"
        if neutral not in execution or compatibility not in execution:
            raise RuntimeError("V22 rollout lost neutral/compatibility clearance fields")
        if not np.array_equal(execution[neutral][...], execution[compatibility][...]):
            raise RuntimeError("V22 neutral and compatibility clearance values diverged")
        language_audit = persist_dynamic_language_v26(stream)
        stream.attrs.update(
            {
                "format": FEASIBLE_MULTIVIEW_H5_FORMAT_V22,
                "base_writer_format": H5_FORMAT,
                "rollout_format": STOCK_GRIPPER_ROLLOUT_FORMAT_V22,
                "execution_kernel_format": (STOCK_GRIPPER_ROLLOUT_KERNEL_FORMAT_V22),
                "reward_format": STOCK_GRIPPER_REWARD_FORMAT_V22,
                "run_plan_sha256": run_plan_sha256,
                "actor_state_sha256_at_collection": actor_state_sha256,
                "parent_checkpoint_path": str(parent_checkpoint),
                "parent_checkpoint_sha256": sha256_file_v1(parent_checkpoint),
                "exploration_standard_deviation": (exploration_standard_deviation),
                "contact_candidate_geom_count": 8,
                "safety_only_geom_count": 88,
                "historical_94_dataset_is_compatibility_alias": True,
                "neutral_safety_only_clearance_dataset": neutral,
                "rollout_projection_audit_json": json.dumps(
                    projection_audit,
                    sort_keys=True,
                ),
                "rollout_update_gate_checks_json": json.dumps(
                    eligibility_checks,
                    sort_keys=True,
                ),
                "rollout_actor_update_eligible": eligible,
                "ppo_training_eligible": eligible,
                "ppo_batch_reconstructable": True,
                "causal_history_materialization_format": (CAUSAL_HISTORY_DIRECT_PERSISTENCE_FORMAT_V22),
                "causal_history_reconstructed_from_current_rows": False,
                "causal_history_persisted_during_collection": True,
                "causal_history_dataset_names_json": json.dumps(CAUSAL_HISTORY_DATASET_NAMES_V22),
                "optimizer_steps": 0,
                "ineligible_rollout_retained_for_audit": True,
                "closed_loop_on_policy": True,
                "autoregressive_feedback_mode": ("applied_task_action_inverse_tanh"),
                "actor_inference_cache_format": (
                    CAUSAL_VISUAL_INFERENCE_CACHE_FORMAT_V24
                ),
                "episode_condition_schedule": condition_schedule,
                "strict_success_contract_json": json.dumps(
                    strict_success_contract,
                    sort_keys=True,
                ),
                "strict_success_hold_seconds": strict_success_contract[
                    "hold_seconds"
                ],
                "strict_success_hold_steps": strict_success_contract[
                    "hold_steps"
                ],
                "control_fps": strict_success_contract["fps"],
                "strict_success_realized_hold_seconds": strict_success_contract[
                    "realized_hold_seconds"
                ],
                "wrist_rgb_present": True,
                "multiview_rgb_present": True,
                "online_depth_present": False,
                "online_segmentation_present": False,
                "expert_calls": 0,
                "behavior_cloning_steps": 0,
                "physical_samples": 0,
                "production_admission": False,
            }
        )
        stream.flush()
        rows = int(stream.attrs["rollout_rows"])
        episodes = int(stream.attrs["completed_episodes"])
        contacts = int(stream.attrs["valid_push_side_contact_transitions"])
    return {
        "path": str(path.resolve()),
        "sha256": sha256_file_v1(path),
        "byte_count": path.stat().st_size,
        "rows": rows,
        "episodes": episodes,
        "valid_push_side_contact_transitions": contacts,
        "dynamic_language_audit": language_audit,
        "strict_success_contract": strict_success_contract,
        "actor_update_eligible": eligible,
        "ppo_batch_reconstructable": True,
        "format": FEASIBLE_MULTIVIEW_H5_FORMAT_V22,
        "production_admission": False,
    }


def run_feasible_multiview_rollout_v22(
    *,
    parent_checkpoint: Path,
    output_dir: Path,
    rollout_seed: int,
    initialization_seed: int,
    rollout_steps: int,
    batch_size: int,
    exploration_standard_deviation: float,
    action_autoregressive_rho: float,
    reset_height_m: float,
    curriculum_tip_gap_band_m: tuple[float, float],
    obstacle_probability: float,
    stress_probability: float,
    device: str,
    eligibility_config: RolloutEligibilityConfigV21 | None = None,
    condition_schedule: str = "random",
    reverse_curriculum_stage: int | None = None,
    ppo_learning_rate: float = CONSERVATIVE_PPO_LEARNING_RATE_V22,
    ppo_update_epochs: int = CONSERVATIVE_PPO_UPDATE_EPOCHS_V22,
) -> dict[str, Any]:
    proposal_strength = audited_ppo_proposal_strength_v40(
        learning_rate=ppo_learning_rate,
        update_epochs=ppo_update_epochs,
    )
    requested_curriculum = {
        "tip_gap_band_m": list(curriculum_tip_gap_band_m),
        "obstacle_probability": obstacle_probability,
        "stress_probability": stress_probability,
    }
    curriculum_stage = None
    if reverse_curriculum_stage is not None:
        curriculum_stage = reverse_curriculum_stage_v26(reverse_curriculum_stage)
        if condition_schedule != "random":
            raise ValueError("V26 reverse curriculum requires stochastic stage conditions")
        curriculum_tip_gap_band_m = curriculum_stage.tip_gap_range_m
        obstacle_probability = curriculum_stage.obstacle_probability
        stress_probability = curriculum_stage.stress_probability
    _validate_configuration_v22(
        rollout_seed=rollout_seed,
        initialization_seed=initialization_seed,
        rollout_steps=rollout_steps,
        batch_size=batch_size,
        exploration_standard_deviation=exploration_standard_deviation,
        action_autoregressive_rho=action_autoregressive_rho,
        reset_height_m=reset_height_m,
        curriculum_tip_gap_band_m=curriculum_tip_gap_band_m,
        obstacle_probability=obstacle_probability,
        stress_probability=stress_probability,
    )
    eligibility = eligibility_config or RolloutEligibilityConfigV21()
    eligibility.validate()
    if condition_schedule not in CONDITION_SCHEDULES_V24:
        raise ValueError("V24 condition schedule is unsupported")
    scheduled_conditions = CONDITION_SCHEDULES_V24[condition_schedule]
    destination = Path(output_dir).expanduser().resolve()
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"V22 output directory is not empty: {destination}")
    resolved_device = _resolve_device(device)
    actor_state, critic_state, root_plan, inherited_lineage = load_collection_parent_actor_critic_v23(
        parent_checkpoint
    )
    environment_config, parent_policy, parent_action, scene_path = _configs_from_parent_plan(root_plan)
    base_environment_sha256 = canonical_sha256_v1(asdict(environment_config))
    environment_lineage_override_v26: dict[str, Any] | None = None
    if curriculum_stage is not None:
        environment_overrides = {
            "reverse_curriculum_stage_v26": curriculum_stage.index,
            "strict_success_hold_steps": strict_success_hold_steps_v26(
                environment_config.fps
            ),
        }
        environment_config = replace(environment_config, **environment_overrides)
        environment_lineage_override_v26 = {
            "format": REVERSE_CURRICULUM_FORMAT_V26,
            "allowed_override_fields": sorted(environment_overrides),
            "overrides": environment_overrides,
            "base_environment_config_sha256": base_environment_sha256,
            "result_environment_config_sha256": canonical_sha256_v1(
                asdict(environment_config)
            ),
        }
    policy_config = replace(
        parent_policy,
        rollout_steps=rollout_steps,
        update_epochs=ppo_update_epochs,
        batch_size=batch_size,
        learning_rate=float(ppo_learning_rate),
        action_autoregressive_rho=action_autoregressive_rho,
        obstacle_probability=obstacle_probability,
        stress_probability=stress_probability,
        auxiliary_view_dropout_probability=0.10,
        seed=initialization_seed,
    )
    policy_config.validate()
    action_config = replace(
        parent_action,
        reset_height_candidates_m=(float(reset_height_m),),
        curriculum_reset_tip_gap_band_m=curriculum_tip_gap_band_m,
    )
    action_config.validate()
    bundle = initialize_asymmetric_multiview_ppo_v1(
        initialization_seed,
        device=resolved_device,
        scene_path=scene_path,
    )
    bundle.actor.load_state_dict(actor_state, strict=True)
    bundle.critic.load_state_dict(critic_state, strict=True)
    parent_actor_hash = state_dict_sha256_v1(actor_state)
    if state_dict_sha256_v1(bundle.actor.state_dict()) != parent_actor_hash:
        raise RuntimeError("V22 failed to load the exact parent actor")
    with torch.no_grad():
        bundle.actor.log_std.fill_(math.log(exploration_standard_deviation))
    actor_hash = state_dict_sha256_v1(bundle.actor.state_dict())
    if _mean_network_sha256(bundle.actor.state_dict()) != inherited_lineage["parent_actor_mean_state_sha256"]:
        raise RuntimeError("V22 exploration override changed actor means")
    reward = StockGripperPotentialRewardV22()
    kernel = StockGripperRolloutKernelV22()
    behavior_source_closure_v26 = build_behavior_source_closure_v26(
        Path(__file__).parent
    )
    source_hashes = {
        **behavior_source_closure_v26["files"],
        "scene": sha256_file_v1(scene_path),
    }
    strict_success_hold_steps = int(environment_config.strict_success_hold_steps)
    strict_success_contract = {
        "format": REVERSE_CURRICULUM_FORMAT_V26,
        "fps": int(environment_config.fps),
        "hold_seconds": STRICT_SUCCESS_HOLD_SECONDS_V26,
        "hold_steps": strict_success_hold_steps,
        "realized_hold_seconds": (
            strict_success_hold_steps / environment_config.fps
        ),
        "exact_three_second_contract": (
            strict_success_hold_steps
            == strict_success_hold_steps_v26(environment_config.fps)
        ),
    }
    plan = {
        "format": FEASIBLE_MULTIVIEW_ROLLOUT_RUN_FORMAT_V22,
        "created_at_utc": _utc_now(),
        "output_dir": str(destination),
        "source_type": SOURCE_TYPE,
        "rollout_seed": rollout_seed,
        "initialization_seed": initialization_seed,
        "resolved_device": resolved_device,
        "environment_config": asdict(environment_config),
        "ppo_config": asdict(policy_config),
        "ppo_proposal_strength_v40": proposal_strength,
        "action_adapter_config": asdict(action_config),
        "reward_config": asdict(reward.config),
        "reward_config_sha256": reward.config_sha256,
        "rollout_eligibility_config": asdict(eligibility),
        "execution_kernel_format": kernel.format,
        "scene_path": str(scene_path),
        "source_hashes": source_hashes,
        "behavior_source_closure_v26": behavior_source_closure_v26,
        "environment_lineage_override_v26": environment_lineage_override_v26,
        "reverse_curriculum_stage_v26": (
            asdict(curriculum_stage) if curriculum_stage is not None else None
        ),
        "requested_curriculum_arguments": requested_curriculum,
        "effective_curriculum": {
            "tip_gap_band_m": list(curriculum_tip_gap_band_m),
            "obstacle_probability": obstacle_probability,
            "stress_probability": stress_probability,
        },
        "strict_success_contract": strict_success_contract,
        "lineage": inherited_lineage,
        "parent_actor_state_sha256": parent_actor_hash,
        "actor_state_sha256_after_std_override": actor_hash,
        "exploration_standard_deviation": exploration_standard_deviation,
        "action_autoregressive_rho": action_autoregressive_rho,
        "episode_condition_schedule": condition_schedule,
        "episode_condition_pairs": scheduled_conditions,
        "actor_inference_cache_format": CAUSAL_VISUAL_INFERENCE_CACHE_FORMAT_V24,
        "closed_loop_on_policy_collection": True,
        "optimizer_steps": 0,
        "wrist_rgb_primary": True,
        "auxiliary_views": ["front", "angled", "overhead"],
        "online_depth_present": False,
        "online_segmentation_present": False,
        "parameter_counts": parameter_counts_v1(bundle),
        "expert_calls": 0,
        "behavior_cloning_steps": 0,
        "physical_samples": 0,
        "production_admission": False,
    }
    plan["run_plan_sha256"] = canonical_sha256_v1(plan)
    destination.mkdir(parents=True, exist_ok=True)
    _atomic_json(destination / "run_plan.json", plan)
    _write_collection_state_v26(
        destination,
        "preflight_complete",
        reverse_curriculum_stage_v26=(
            curriculum_stage.index if curriculum_stage is not None else None
        ),
        strict_success_contract=strict_success_contract,
    )
    env = RealisticEdgeArmEnvV10(
        environment_config,
        seed=rollout_seed,
        model_scene_path=scene_path,
    )
    renderer: MultiViewRGBRendererV1 | None = None
    try:
        _write_collection_state_v26(destination, "collecting_rollout")
        renderer = MultiViewRGBRendererV1(
            env,
            height=policy_config.image_height,
            width=policy_config.image_width,
        )
        rollout = collect_asymmetric_multiview_rollout_v1(
            env,
            renderer,
            StockGripperTaskFrameAdapterV22(env, action_config),
            bundle.actor,
            bundle.critic,
            policy_config,
            seed=rollout_seed,
            potential_reward=reward,
            autoregressive_applied_action_feedback=True,
            execution_kernel=kernel,
            episode_condition_schedule=scheduled_conditions,
        )
        audit = build_batch_projection_audit_v21(rollout, eligibility)
        checks = rollout_update_gates_v21(audit, eligibility)
        rollout_path = destination / "rollout.h5"
        write_online_rollout_h5_v1(
            rollout_path,
            rollout,
            update_index=0,
            provenance=bundle.provenance,
            config=policy_config,
            scene_sha256=source_hashes["scene"],
            resume_lineage=inherited_lineage,
        )
        artifact = _relabel_h5_v22(
            rollout_path,
            run_plan_sha256=plan["run_plan_sha256"],
            actor_state_sha256=actor_hash,
            parent_checkpoint=Path(parent_checkpoint).expanduser().resolve(),
            exploration_standard_deviation=exploration_standard_deviation,
            projection_audit=asdict(audit),
            eligibility_checks=checks,
            condition_schedule=condition_schedule,
            strict_success_contract=strict_success_contract,
        )
    except Exception as error:
        _write_collection_state_v26(
            destination,
            "failed",
            error_type=type(error).__name__,
            error=str(error),
        )
        _atomic_json(
            destination / "failure.json",
            {
                "format": FEASIBLE_MULTIVIEW_ROLLOUT_RUN_FORMAT_V22,
                "status": "failed",
                "error_type": type(error).__name__,
                "error": str(error),
                "failed_at_utc": _utc_now(),
                "production_admission": False,
            },
        )
        raise
    finally:
        if renderer is not None:
            renderer.close()
    summary = {
        "format": FEASIBLE_MULTIVIEW_ROLLOUT_RUN_FORMAT_V22,
        "status": "complete",
        "completed_at_utc": _utc_now(),
        "rollout_update_eligible": all(checks.values()),
        "rollout_update_eligibility_checks": checks,
        "rollout_projection_audit": asdict(audit),
        "invalid_contact_transition_count": int(np.count_nonzero(rollout.invalid_tool_block_contact_any)),
        "strict_success_transition_count": int(np.count_nonzero(rollout.strict_success)),
        "strict_success_contract": strict_success_contract,
        "reverse_curriculum_stage_v26": (
            asdict(curriculum_stage) if curriculum_stage is not None else None
        ),
        "minimum_executed_88_safety_only_block_clearance_m": float(
            np.min(rollout.minimum_executed_94_safety_only_block_clearance_m)
        ),
        "rollout_artifact": artifact,
        "actor_state_sha256": actor_hash,
        "actor_state_unchanged_by_collection": (
            state_dict_sha256_v1(bundle.actor.state_dict()) == actor_hash
        ),
        "optimizer_steps": 0,
        "expert_calls": 0,
        "behavior_cloning_steps": 0,
        "physical_samples": 0,
        "production_admission": False,
        "remaining_gates": [
            "eligible rollout before optimizer",
            "paired same-seed pre/post-update evaluation",
            "multi-seed obstacle and stress validation",
            "statistically validated 3-second stable target success",
            "depth, segmentation, causal 4D, and physical calibration",
        ],
    }
    _atomic_json(destination / "summary.json", summary)
    _write_collection_state_v26(
        destination,
        "complete",
        rollout_update_eligible=all(checks.values()),
        rollout_artifact=artifact,
    )
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--rollout-seed", type=int, default=25_900_000)
    parser.add_argument("--initialization-seed", type=int, default=25_950_000)
    parser.add_argument("--rollout-steps", type=int, default=360)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--exploration-standard-deviation", type=float, default=0.20)
    parser.add_argument("--action-autoregressive-rho", type=float, default=0.80)
    parser.add_argument(
        "--ppo-learning-rate",
        type=float,
        default=CONSERVATIVE_PPO_LEARNING_RATE_V22,
    )
    parser.add_argument(
        "--ppo-update-epochs",
        type=int,
        default=CONSERVATIVE_PPO_UPDATE_EPOCHS_V22,
    )
    parser.add_argument("--reset-height-m", type=float, default=0.050)
    parser.add_argument("--curriculum-min-tip-gap-m", type=float, default=0.00025)
    parser.add_argument("--curriculum-max-tip-gap-m", type=float, default=0.0048)
    parser.add_argument("--obstacle-probability", type=float, default=0.0)
    parser.add_argument("--stress-probability", type=float, default=0.0)
    parser.add_argument("--reverse-curriculum-stage", type=int)
    parser.add_argument(
        "--condition-schedule",
        choices=tuple(CONDITION_SCHEDULES_V24),
        default="random",
    )
    parser.add_argument("--device", choices=("auto", "cpu", "mps"), default="auto")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    summary = run_feasible_multiview_rollout_v22(
        parent_checkpoint=args.parent_checkpoint,
        output_dir=args.output_dir,
        rollout_seed=args.rollout_seed,
        initialization_seed=args.initialization_seed,
        rollout_steps=args.rollout_steps,
        batch_size=args.batch_size,
        exploration_standard_deviation=(args.exploration_standard_deviation),
        action_autoregressive_rho=args.action_autoregressive_rho,
        reset_height_m=args.reset_height_m,
        curriculum_tip_gap_band_m=(
            args.curriculum_min_tip_gap_m,
            args.curriculum_max_tip_gap_m,
        ),
        obstacle_probability=args.obstacle_probability,
        stress_probability=args.stress_probability,
        device=args.device,
        condition_schedule=args.condition_schedule,
        reverse_curriculum_stage=args.reverse_curriculum_stage,
        ppo_learning_rate=args.ppo_learning_rate,
        ppo_update_epochs=args.ppo_update_epochs,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
