"""Attempt one V40 backtracked PPO update from an exact V22 rollout.

The input is a single previously collected on-policy V22 rollout whose actor
hash, causal histories, action likelihoods, critic values, projection audit,
and eligibility gates are replayed before optimization.  The rollout may be
consumed once. Deterministic paired simulation evaluates progressively smaller
points on the optimizer direction, then either commits the largest candidate
that passes every gate or restores the actor and critic byte-for-byte. Failed
simulated episodes may inform PPO under the bounded V40 learning-source gate;
they never count as demonstrations and never relax candidate promotion.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from datetime import datetime, timezone
import copy
import json
import math
from pathlib import Path
from typing import Any, Sequence

import h5py
import torch

from .asymmetric_multiview_ppo_v1 import (
    SOURCE_TYPE,
    AsymmetricMultiViewPPOConfigV1,
    AsymmetricMultiViewRolloutBatchV1,
    AsymmetricPrivilegedCriticV1,
    CausalVisualInferenceCacheV24,
    MultiViewRGBRendererV1,
    SelectedViewRecurrentActorV1,
    autoregressive_action_distribution_v1,
    canonical_sha256_v1,
    evaluate_asymmetric_multiview_policy_v1,
    initialize_asymmetric_multiview_ppo_v1,
    parameter_counts_v1,
    sha256_file_v1,
)
from .audit_goal_directed_rollout_v25 import (
    audit_goal_directed_rollout_v25,
    frontier_condition_admission_v25,
)
from .evaluate_asymmetric_drq_sac_v1 import (
    _configs_from_parent_plan,
    build_paired_comparison_v1,
)
from .dynamic_language_v26 import verify_dynamic_language_v26
from .feasible_on_policy_ppo_v21 import (
    RolloutEligibilityConfigV21,
    build_batch_projection_audit_v21,
    rollout_update_gates_v21,
)
from .goal_directed_feasible_on_policy_ppo_v23 import (
    GoalDirectedFeasibleRewardConfigV23,
    LOG_PROBABILITY_ALIGNMENT_MODE_V25,
    MAXIMUM_BATCHED_REPLAY_LOG_PROBABILITY_DELTA_V25,
    align_cached_collection_log_probabilities_for_batched_ppo_v25,
    goal_directed_feasible_on_policy_ppo_update_v23,
)
from .meaningful_effect_backtracking_v26 import (
    MEANINGFUL_EFFECT_BACKTRACKING_FORMAT_V26,
    meaningful_effect_audit_v26,
    meaningful_effect_closed_loop_gates_v26,
    meaningful_effect_contract_v26,
)
from .ppo_utils_v1 import (
    squashed_gaussian_log_prob_v1,
    state_dict_sha256_v1,
)
from .reverse_curriculum_v26 import (
    REVERSE_CURRICULUM_FORMAT_V26,
    curriculum_promotion_gate_v26,
    evaluation_condition_schedule_v26,
    reverse_curriculum_stage_v26,
    strict_success_hold_steps_v26,
)
from .rl_learning_admission_v40 import rl_learning_source_admission_v40
from .sim2real_env_v10 import RealisticEdgeArmEnvV10
from .source_closure_v26 import verify_behavior_source_closure_v26
from .stock_gripper_reward_v22 import StockGripperPotentialRewardV22
from .stock_gripper_evaluation_kernel_v25 import (
    STOCK_GRIPPER_EVALUATION_FORMAT_V25,
    StockGripperEvaluationKernelV25,
)
from .stock_gripper_taskframe_v22 import StockGripperTaskFrameAdapterV22
from .trust_region_backtracking_v25 import (
    BACKTRACKING_SCALES_V25,
    interpolate_state_dict_v25,
    scale_token_v25,
)
from .train_on_policy_recurrent_ppo_v20 import (
    _mean_network_sha256,
    _resolve_device,
)
from .v23_checkpoint_lineage import (
    FEASIBLE_MULTIVIEW_PPO_CHECKPOINT_FORMAT_V23,
    FEASIBLE_MULTIVIEW_PPO_RUN_FORMAT_V23,
    load_collection_parent_actor_critic_v23,
)
from .v22_rollout_h5 import (
    CAUSAL_HISTORY_PERSISTENCE_FORMATS_V22,
    load_feasible_multiview_rollout_v22,
    load_verified_run_plan_v22,
)


FEASIBLE_MULTIVIEW_PPO_RUN_FORMAT_V22 = FEASIBLE_MULTIVIEW_PPO_RUN_FORMAT_V23
FEASIBLE_MULTIVIEW_PPO_CHECKPOINT_FORMAT_V22 = FEASIBLE_MULTIVIEW_PPO_CHECKPOINT_FORMAT_V23

SHARED_COLLECTION_SOURCE_FILES_V25 = (
    "edgearm/asymmetric_multiview_ppo_v1.py",
    "edgearm/stock_gripper_taskframe_v22.py",
    "edgearm/stock_gripper_reward_v22.py",
    "edgearm/stock_gripper_rollout_kernel_v22.py",
    "edgearm/v23_checkpoint_lineage.py",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    partial.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    partial.replace(path)


def _atomic_torch_save(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    torch.save(payload, partial)
    partial.replace(path)


def _attribute_text(attributes: h5py.AttributeManager, name: str) -> str:
    if name not in attributes:
        raise ValueError(f"V22 training source is missing attribute: {name}")
    value = attributes[name]
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def model_replay_audit_v22(
    actor: SelectedViewRecurrentActorV1,
    critic: AsymmetricPrivilegedCriticV1,
    batch: AsymmetricMultiViewRolloutBatchV1,
    config: AsymmetricMultiViewPPOConfigV1,
) -> dict[str, Any]:
    """Verify persisted policy likelihoods and values against current weights."""

    batch.validate()
    config.validate()
    try:
        device = next(actor.parameters()).device
    except StopIteration as error:  # pragma: no cover
        raise ValueError("V22 actor has no parameters") from error
    try:
        critic_device = next(critic.parameters()).device
    except StopIteration as error:  # pragma: no cover
        raise ValueError("V22 critic has no parameters") from error
    if critic_device != device:
        raise ValueError("V22 actor and critic devices differ")

    actor.eval()
    critic.eval()
    with torch.no_grad():
        inference_cache = CausalVisualInferenceCacheV24(
            actor,
            history_steps=config.history_steps,
        )
        replayed_log_prob_rows: list[torch.Tensor] = []
        for row in range(len(batch.rewards)):
            if int(batch.episode_step_ids[row]) == 0:
                inference_cache.reset()
            base_distribution = inference_cache.distribution(
                batch.rgb_frames[row],
                batch.joint_state[row],
                batch.previous_executed_action[row],
                batch.view_valid[row],
            )
            distribution = autoregressive_action_distribution_v1(
                base_distribution,
                torch.from_numpy(batch.previous_policy_pre_tanh[row])
                .to(device)
                .unsqueeze(0),
                config.action_autoregressive_rho,
            )
            replayed_log_prob_rows.append(
                squashed_gaussian_log_prob_v1(
                    distribution,
                    torch.from_numpy(batch.pre_tanh[row]).to(device).unsqueeze(0),
                ).cpu()
            )
        replayed_log_probs = torch.cat(replayed_log_prob_rows)
        replayed_values = critic(torch.from_numpy(batch.privileged_state).to(device)).cpu()
        replayed_next_values = critic(torch.from_numpy(batch.next_privileged_state).to(device)).cpu()
    old_log_probs = torch.from_numpy(batch.old_log_probs)
    old_values = torch.from_numpy(batch.values)
    old_next_values = torch.from_numpy(batch.next_values)
    likelihood_matches = torch.allclose(
        replayed_log_probs,
        old_log_probs,
        rtol=1.0e-5,
        atol=1.0e-5,
    )
    value_matches = torch.allclose(
        replayed_values,
        old_values,
        rtol=1.0e-5,
        atol=1.0e-5,
    )
    next_value_matches = torch.allclose(
        replayed_next_values,
        old_next_values,
        rtol=1.0e-5,
        atol=1.0e-5,
    )
    audit = {
        "old_log_probability_replay_matches": bool(likelihood_matches),
        "old_log_probability_replay_mode": "causal_visual_inference_cache_v24",
        "overlapping_visual_history_reencoded": False,
        "critic_value_replay_matches": bool(value_matches),
        "critic_next_value_replay_matches": bool(next_value_matches),
        "maximum_old_log_probability_absolute_error": float(
            torch.max(torch.abs(replayed_log_probs - old_log_probs)).item()
        ),
        "maximum_critic_value_absolute_error": float(
            torch.max(torch.abs(replayed_values - old_values)).item()
        ),
        "maximum_critic_next_value_absolute_error": float(
            torch.max(torch.abs(replayed_next_values - old_next_values)).item()
        ),
    }
    audit["all_model_outputs_replay"] = all(
        audit[name]
        for name in (
            "old_log_probability_replay_matches",
            "critic_value_replay_matches",
            "critic_next_value_replay_matches",
        )
    )
    if not audit["all_model_outputs_replay"]:
        raise ValueError(
            "V22 stored rollout does not belong to the loaded actor/critic: "
            + json.dumps(audit, sort_keys=True)
        )
    return audit


def _derive_exact_configs(
    plan: dict[str, Any],
    root_plan: dict[str, Any],
) -> tuple[Any, AsymmetricMultiViewPPOConfigV1, Any, Path]:
    root_environment, _parent_policy, parent_action, scene = _configs_from_parent_plan(root_plan)
    environment_payload = plan.get("environment_config")
    policy_payload = plan.get("ppo_config")
    action_payload = plan.get("action_adapter_config")
    if not all(
        isinstance(payload, dict) for payload in (environment_payload, policy_payload, action_payload)
    ):
        raise TypeError("V22 collection plan lost one of its exact configurations")
    environment_override = plan.get("environment_lineage_override_v26")
    if environment_override is None:
        environment = root_environment
        if canonical_sha256_v1(asdict(environment)) != canonical_sha256_v1(
            environment_payload
        ):
            raise ValueError("V22 collection environment diverged from its root lineage")
    else:
        if type(environment_override) is not dict:
            raise TypeError("V26 environment-lineage override must be a dictionary")
        if environment_override.get("format") != REVERSE_CURRICULUM_FORMAT_V26:
            raise ValueError("V26 environment-lineage override format changed")
        expected_fields = [
            "reverse_curriculum_stage_v26",
            "strict_success_hold_steps",
        ]
        if environment_override.get("allowed_override_fields") != expected_fields:
            raise ValueError("V26 environment-lineage override authority expanded")
        overrides = environment_override.get("overrides")
        if type(overrides) is not dict or sorted(overrides) != expected_fields:
            raise ValueError("V26 environment-lineage override fields changed")
        stage = reverse_curriculum_stage_v26(
            overrides.get("reverse_curriculum_stage_v26")
        )
        required_hold_steps = strict_success_hold_steps_v26(root_environment.fps)
        if overrides.get("strict_success_hold_steps") != required_hold_steps:
            raise ValueError("V26 environment override is not an exact three-second hold")
        if environment_override.get("base_environment_config_sha256") != canonical_sha256_v1(
            asdict(root_environment)
        ):
            raise ValueError("V26 base environment lineage changed")
        environment = replace(root_environment, **overrides)
        if environment.reverse_curriculum_stage_v26 != stage.index:
            raise RuntimeError("V26 reconstructed a different curriculum stage")
        result_hash = canonical_sha256_v1(asdict(environment))
        if environment_override.get("result_environment_config_sha256") != result_hash:
            raise ValueError("V26 resulting environment lineage changed")
        if result_hash != canonical_sha256_v1(environment_payload):
            raise ValueError("V26 collection environment cannot be exactly reconstructed")
    policy = AsymmetricMultiViewPPOConfigV1(**policy_payload)
    policy.validate()
    reset_heights = action_payload.get("reset_height_candidates_m")
    gap_band = action_payload.get("curriculum_reset_tip_gap_band_m")
    if not isinstance(reset_heights, list) or not isinstance(gap_band, list):
        raise TypeError("V22 action curriculum tuples are missing")
    action = replace(
        parent_action,
        reset_height_candidates_m=tuple(float(value) for value in reset_heights),
        curriculum_reset_tip_gap_band_m=tuple(float(value) for value in gap_band),
    )
    action.validate()
    if canonical_sha256_v1(asdict(action)) != canonical_sha256_v1(action_payload):
        raise ValueError("V22 action configuration cannot be exactly reconstructed")
    if str(scene) != str(Path(plan["scene_path"]).expanduser().resolve()):
        raise ValueError("V22 scene path diverged from its root lineage")
    expected_scene_hash = plan.get("source_hashes", {}).get("scene")
    if not isinstance(expected_scene_hash, str) or sha256_file_v1(scene) != expected_scene_hash:
        raise ValueError("V22 scene bytes changed after collection")
    return environment, policy, action, scene


def verify_shared_collection_sources_v25(plan: dict[str, Any]) -> dict[str, str]:
    """Prove that collection/training share behavior-critical source bytes."""

    source_hashes = plan.get("source_hashes")
    if not isinstance(source_hashes, dict):
        raise TypeError("V25 collection plan lost source hashes")
    closure = plan.get("behavior_source_closure_v26")
    if closure is not None:
        verified = verify_behavior_source_closure_v26(
            closure,
            Path(__file__).parent,
        )
        if any(source_hashes.get(name) != digest for name, digest in verified.items()):
            raise ValueError("V26 source closure and flat source hashes disagree")
        return verified
    verified: dict[str, str] = {}
    for relative_name in SHARED_COLLECTION_SOURCE_FILES_V25:
        expected = source_hashes.get(relative_name)
        current_path = Path(__file__).with_name(Path(relative_name).name)
        current = sha256_file_v1(current_path)
        if not isinstance(expected, str) or expected != current:
            raise ValueError(
                "V25 collection/training source bytes differ: " + relative_name
            )
        verified[relative_name] = current
    return verified


def verify_v26_collection_contracts(
    plan: dict[str, Any],
    rollout_path: Path,
    environment: Any,
    batch: AsymmetricMultiViewRolloutBatchV1,
) -> dict[str, Any]:
    """Fail closed on V26 language and three-second curriculum metadata."""

    if plan.get("behavior_source_closure_v26") is None:
        return {"v26_contract_present": False, "production_admission": False}
    strict_contract = plan.get("strict_success_contract")
    if type(strict_contract) is not dict:
        raise TypeError("V26 collection plan lost strict-success contract")
    with h5py.File(rollout_path, "r") as stream:
        h5_contract = json.loads(
            _attribute_text(stream.attrs, "strict_success_contract_json")
        )
        language_audit = verify_dynamic_language_v26(stream)
        execution = stream.get("execution")
        if not isinstance(execution, h5py.Group):
            raise RuntimeError("V26 rollout lost execution identity")
        if not torch.equal(
            torch.from_numpy(execution["episode_ids"][...]),
            torch.from_numpy(batch.episode_ids),
        ):
            raise ValueError("V26 language/execution row identity differs from PPO batch")
    if h5_contract != strict_contract:
        raise ValueError("V26 H5 and run-plan strict-success contracts differ")
    stage_payload = plan.get("reverse_curriculum_stage_v26")
    stage_index = environment.reverse_curriculum_stage_v26
    if stage_index is None or type(stage_payload) is not dict:
        raise ValueError("V26 collection lacks an active reverse-curriculum stage")
    stage = reverse_curriculum_stage_v26(stage_index)
    if canonical_sha256_v1(stage_payload) != canonical_sha256_v1(
        asdict(stage)
    ):
        raise ValueError("V26 collection stage payload changed")
    required_hold_steps = strict_success_hold_steps_v26(environment.fps)
    expected_contract = {
        "format": REVERSE_CURRICULUM_FORMAT_V26,
        "fps": int(environment.fps),
        "hold_seconds": 3.0,
        "hold_steps": required_hold_steps,
        "realized_hold_seconds": required_hold_steps / environment.fps,
        "exact_three_second_contract": True,
    }
    if strict_contract != expected_contract:
        raise ValueError("V26 strict-success contract is not exactly three seconds")
    return {
        "v26_contract_present": True,
        "reverse_curriculum_stage": asdict(stage),
        "strict_success_contract": strict_contract,
        "dynamic_language_audit": language_audit,
        "rollout_rows": len(batch.rewards),
        "expert_calls": 0,
        "behavior_cloning_steps": 0,
        "production_admission": False,
    }


def curriculum_promotion_from_evaluation_v26(
    evaluation: dict[str, Any],
    stage_index: int,
) -> dict[str, Any]:
    stage = reverse_curriculum_stage_v26(stage_index)
    required_integer_fields = (
        "episode_count",
        "strict_success_count",
        "contact_episode_count",
        "invalid_tool_block_contact_transition_count",
        "valid_push_side_contact_transition_count",
        "shield_terminal_episode_count",
        "safety_violation_episode_count",
        "control_fps",
        "strict_success_hold_steps",
        "expert_calls",
        "behavior_cloning_steps",
    )
    if any(type(evaluation.get(name)) is not int for name in required_integer_fields):
        raise TypeError("V26 evaluation lacks integer promotion evidence")
    valid_contacts = int(evaluation["valid_push_side_contact_transition_count"])
    invalid_contacts = int(evaluation["invalid_tool_block_contact_transition_count"])
    return curriculum_promotion_gate_v26(
        stage,
        evaluation_episodes=int(evaluation["episode_count"]),
        strict_success_episodes=int(evaluation["strict_success_count"]),
        contact_episodes=int(evaluation["contact_episode_count"]),
        invalid_contact_transitions=invalid_contacts,
        contact_transitions=valid_contacts + invalid_contacts,
        shield_terminal_episodes=int(evaluation["shield_terminal_episode_count"]),
        safety_violation_episodes=int(evaluation["safety_violation_episode_count"]),
        observed_fps=int(evaluation["control_fps"]),
        observed_hold_steps=int(evaluation["strict_success_hold_steps"]),
        expert_calls=int(evaluation["expert_calls"]),
        behavior_cloning_steps=int(evaluation["behavior_cloning_steps"]),
    )


def _write_run_state(destination: Path, phase: str, **payload: Any) -> None:
    _atomic_json(
        destination / "run_state.json",
        {
            "format": FEASIBLE_MULTIVIEW_PPO_RUN_FORMAT_V22,
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


def run_feasible_multiview_ppo_update_v22(
    *,
    rollout_path: Path,
    collection_run_plan_path: Path,
    output_dir: Path,
    evaluation_seed_base: int,
    evaluation_episodes: int,
    device: str,
    reward_config: GoalDirectedFeasibleRewardConfigV23 | None = None,
) -> dict[str, Any]:
    if type(evaluation_seed_base) is not int or evaluation_seed_base < 0:
        raise ValueError("V22 evaluation seed base must be non-negative")
    if type(evaluation_episodes) is not int or evaluation_episodes < 1:
        raise ValueError("V22 evaluation episode count must be positive")
    destination = Path(output_dir).expanduser().resolve()
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"V22 PPO output directory is not empty: {destination}")
    rollout = Path(rollout_path).expanduser().resolve()
    collection_plan_path = Path(collection_run_plan_path).expanduser().resolve()
    rollout_sha256 = sha256_file_v1(rollout)
    with h5py.File(rollout, "r") as stream:
        collection_plan_sha256 = _attribute_text(stream.attrs, "run_plan_sha256")
        collection_actor_sha256 = _attribute_text(stream.attrs, "actor_state_sha256_at_collection")
        parent_checkpoint = (
            Path(_attribute_text(stream.attrs, "parent_checkpoint_path")).expanduser().resolve()
        )
        if sha256_file_v1(parent_checkpoint) != _attribute_text(stream.attrs, "parent_checkpoint_sha256"):
            raise ValueError("V22 parent checkpoint bytes changed")
        if (
            _attribute_text(stream.attrs, "causal_history_materialization_format")
            not in CAUSAL_HISTORY_PERSISTENCE_FORMATS_V22
        ):
            raise ValueError("V22 training source lacks exact causal materialization")
        stored_projection_audit = json.loads(_attribute_text(stream.attrs, "rollout_projection_audit_json"))
        stored_eligibility_checks = json.loads(
            _attribute_text(stream.attrs, "rollout_update_gate_checks_json")
        )
    collection_plan = load_verified_run_plan_v22(
        collection_plan_path,
        expected_sha256=collection_plan_sha256,
    )
    shared_collection_source_hashes = verify_shared_collection_sources_v25(
        collection_plan
    )
    batch = load_feasible_multiview_rollout_v22(rollout)
    goal_directed_rollout_audit = audit_goal_directed_rollout_v25(rollout)
    if goal_directed_rollout_audit.get("rollout_sha256") != rollout_sha256:
        raise ValueError("V25 goal-directed audit hashed different rollout bytes")
    frontier_condition_admission = frontier_condition_admission_v25(
        goal_directed_rollout_audit
    )
    actor_state, critic_state, root_plan, lineage = load_collection_parent_actor_critic_v23(parent_checkpoint)
    if canonical_sha256_v1(lineage) != canonical_sha256_v1(collection_plan.get("lineage")):
        raise ValueError("V22 actor/critic lineage differs from collection")
    environment_config, policy_config, action_config, scene_path = _derive_exact_configs(
        collection_plan, root_plan
    )
    v26_collection_contract_audit = verify_v26_collection_contracts(
        collection_plan,
        rollout,
        environment_config,
        batch,
    )
    curriculum_stage_index = environment_config.reverse_curriculum_stage_v26
    evaluation_condition_schedule = (
        evaluation_condition_schedule_v26(
            reverse_curriculum_stage_v26(curriculum_stage_index),
            evaluation_episodes,
        )
        if curriculum_stage_index is not None
        else None
    )
    recorded_device = collection_plan.get("resolved_device")
    resolved_device = _resolve_device(device)
    if resolved_device != recorded_device:
        raise ValueError(
            f"V22 update device {resolved_device} differs from collection device {recorded_device}"
        )
    if batch.rollout_seed != int(collection_plan["rollout_seed"]):
        raise ValueError("V22 loaded batch seed differs from its collection plan")
    eligibility_payload = collection_plan.get("rollout_eligibility_config")
    if not isinstance(eligibility_payload, dict):
        raise TypeError("V22 collection plan lost rollout eligibility configuration")
    eligibility = RolloutEligibilityConfigV21(**eligibility_payload)
    eligibility.validate()
    projection_audit = build_batch_projection_audit_v21(batch, eligibility)
    eligibility_checks = rollout_update_gates_v21(projection_audit, eligibility)
    if asdict(projection_audit) != stored_projection_audit:
        raise ValueError("V22 recomputed projection audit differs from H5 admission")
    if eligibility_checks != stored_eligibility_checks or not all(eligibility_checks.values()):
        raise ValueError("V22 recomputed rollout eligibility differs or fails")
    rl_learning_source_admission = rl_learning_source_admission_v40(
        goal_directed_rollout_audit,
        eligibility_checks,
    )
    if rl_learning_source_admission.get("learning_update_allowed") is not True:
        raise ValueError("V40 rollout lacks bounded, task-active RL learning evidence")

    bundle = initialize_asymmetric_multiview_ppo_v1(
        int(collection_plan["initialization_seed"]),
        device=resolved_device,
        scene_path=scene_path,
    )
    bundle.actor.load_state_dict(actor_state, strict=True)
    bundle.critic.load_state_dict(critic_state, strict=True)
    with torch.no_grad():
        bundle.actor.log_std.fill_(math.log(float(collection_plan["exploration_standard_deviation"])))
    if state_dict_sha256_v1(bundle.actor.state_dict()) != collection_actor_sha256:
        raise ValueError("V22 loaded actor is not byte-identical to collection actor")
    if _mean_network_sha256(bundle.actor.state_dict()) != lineage["parent_actor_mean_state_sha256"]:
        raise ValueError("V22 actor mean network changed after collection")
    replay_audit = model_replay_audit_v22(
        bundle.actor,
        bundle.critic,
        batch,
        policy_config,
    )
    _aligned_replay_batch, batched_log_probability_alignment_audit = (
        align_cached_collection_log_probabilities_for_batched_ppo_v25(
            bundle.actor,
            batch,
            policy_config,
        )
    )
    intrinsic_reward = reward_config or GoalDirectedFeasibleRewardConfigV23()
    intrinsic_reward.validate()
    task_reward = StockGripperPotentialRewardV22()
    if task_reward.config_sha256 != batch.potential_reward_config_sha256:
        raise ValueError("V22 task reward configuration changed after collection")
    kernel = StockGripperEvaluationKernelV25()

    training_plan = {
        "format": FEASIBLE_MULTIVIEW_PPO_RUN_FORMAT_V22,
        "created_at_utc": _utc_now(),
        "source_type": SOURCE_TYPE,
        "output_dir": str(destination),
        "rollout_path": str(rollout),
        "rollout_sha256": rollout_sha256,
        "collection_run_plan_path": str(collection_plan_path),
        "collection_run_plan_sha256": collection_plan_sha256,
        "collection_actor_state_sha256": collection_actor_sha256,
        "parent_checkpoint_path": str(parent_checkpoint),
        "parent_checkpoint_sha256": sha256_file_v1(parent_checkpoint),
        "algorithm_format": MEANINGFUL_EFFECT_BACKTRACKING_FORMAT_V26,
        "execution_kernel_format": kernel.format,
        "ppo_config": asdict(policy_config),
        "intrinsic_reward_config": asdict(intrinsic_reward),
        "rollout_eligibility_config": asdict(eligibility),
        "rollout_projection_audit": asdict(projection_audit),
        "rollout_eligibility_checks": eligibility_checks,
        "goal_directed_rollout_audit": goal_directed_rollout_audit,
        "frontier_condition_admission": frontier_condition_admission,
        "rl_learning_source_admission_v40": rl_learning_source_admission,
        "collection_frontier_is_diagnostic_not_optimizer_admission": True,
        "model_replay_audit": replay_audit,
        "batched_ppo_log_probability_alignment_preflight": (
            batched_log_probability_alignment_audit
        ),
        "shared_collection_source_hashes_verified": (
            shared_collection_source_hashes
        ),
        "v26_collection_contract_audit": v26_collection_contract_audit,
        "reverse_curriculum_stage_v26": curriculum_stage_index,
        "evaluation_condition_schedule_v26": (
            [list(item) for item in evaluation_condition_schedule]
            if evaluation_condition_schedule is not None
            else None
        ),
        "evaluation_seed_base": evaluation_seed_base,
        "evaluation_episodes": evaluation_episodes,
        "resolved_device": resolved_device,
        "deferred_on_policy_update": True,
        "rollout_actor_hash_verified_unchanged": True,
        "rollout_consumption_limit": 1,
        "rollout_consumption_index": 1,
        "recollect_required_before_any_next_update": True,
        "paired_closed_loop_gate": True,
        "failed_proposal_rolls_back_actor_critic": True,
        "v21_euclidean_projection_penalty_retired": True,
        "same_direction_safe_attenuation_penalty": 0.0,
        "absolute_block_motion_intrinsic_bonus_retired": True,
        "goal_directed_signed_target_distance_progress": True,
        "contact_is_small_curriculum_bonus": True,
        "advantage_normalization": "per_complete_episode_v24",
        "cross_condition_negative_transfer_gate": True,
        "aggregate_shield_rejection_gate": True,
        "parameter_backtracking_scales": list(BACKTRACKING_SCALES_V25),
        "parameter_backtracking_selection": "largest_passing_same_seed_candidate",
        "meaningful_effect_contract_v26": meaningful_effect_contract_v26(),
        "optimizer_proposal_is_only_a_direction_until_closed_loop_acceptance": True,
        "batched_ppo_log_probability_alignment_mode": (
            LOG_PROBABILITY_ALIGNMENT_MODE_V25
        ),
        "batched_ppo_log_probability_alignment_tolerance": (
            MAXIMUM_BATCHED_REPLAY_LOG_PROBABILITY_DELTA_V25
        ),
        "exact_causal_collection_replay_precedes_numeric_alignment": True,
        "confirmed_v21_failure_mode": ("projection_penalty_dominated_sparse_task_reward_and_induced_retreat"),
        "confirmed_v22_failure_mode": ("contact_and_absolute_motion_improved_without_target_progress"),
        "confirmed_v24_failure_mode": (
            "small_sampled_kl_still_crossed_a_recurrent_obstacle_stress_safety_boundary"
        ),
        "confirmed_v26_failure_mode": (
            "micrometre_scale_task_delta_was_not_a_physically_meaningful_update_effect"
        ),
        "source_hashes": {
            "edgearm/train_feasible_multiview_ppo_v22.py": sha256_file_v1(Path(__file__).resolve()),
            "edgearm/directional_feasible_on_policy_ppo_v22.py": sha256_file_v1(
                Path(__file__).with_name("directional_feasible_on_policy_ppo_v22.py")
            ),
            "edgearm/goal_directed_feasible_on_policy_ppo_v23.py": sha256_file_v1(
                Path(__file__).with_name("goal_directed_feasible_on_policy_ppo_v23.py")
            ),
            "edgearm/evaluate_asymmetric_drq_sac_v1.py": sha256_file_v1(
                Path(__file__).with_name("evaluate_asymmetric_drq_sac_v1.py")
            ),
            "edgearm/audit_goal_directed_rollout_v25.py": sha256_file_v1(
                Path(__file__).with_name("audit_goal_directed_rollout_v25.py")
            ),
            "edgearm/rl_learning_admission_v40.py": sha256_file_v1(
                Path(__file__).with_name("rl_learning_admission_v40.py")
            ),
            "edgearm/asymmetric_multiview_ppo_v1.py": sha256_file_v1(
                Path(__file__).with_name("asymmetric_multiview_ppo_v1.py")
            ),
            "edgearm/v23_checkpoint_lineage.py": sha256_file_v1(
                Path(__file__).with_name("v23_checkpoint_lineage.py")
            ),
            "edgearm/on_policy_recurrent_ppo_v20.py": sha256_file_v1(
                Path(__file__).with_name("on_policy_recurrent_ppo_v20.py")
            ),
            "edgearm/feasible_on_policy_ppo_v21.py": sha256_file_v1(
                Path(__file__).with_name("feasible_on_policy_ppo_v21.py")
            ),
            "edgearm/stock_gripper_taskframe_v22.py": sha256_file_v1(
                Path(__file__).with_name("stock_gripper_taskframe_v22.py")
            ),
            "edgearm/stock_gripper_reward_v22.py": sha256_file_v1(
                Path(__file__).with_name("stock_gripper_reward_v22.py")
            ),
            "edgearm/stock_gripper_rollout_kernel_v22.py": sha256_file_v1(
                Path(__file__).with_name("stock_gripper_rollout_kernel_v22.py")
            ),
            "edgearm/stock_gripper_evaluation_kernel_v25.py": sha256_file_v1(
                Path(__file__).with_name("stock_gripper_evaluation_kernel_v25.py")
            ),
            "edgearm/ppo_utils_v1.py": sha256_file_v1(
                Path(__file__).with_name("ppo_utils_v1.py")
            ),
            "edgearm/trust_region_backtracking_v25.py": sha256_file_v1(
                Path(__file__).with_name("trust_region_backtracking_v25.py")
            ),
            "edgearm/meaningful_effect_backtracking_v26.py": sha256_file_v1(
                Path(__file__).with_name("meaningful_effect_backtracking_v26.py")
            ),
            "scene": sha256_file_v1(scene_path),
        },
        "parameter_counts": parameter_counts_v1(bundle),
        "expert_calls": 0,
        "behavior_cloning_steps": 0,
        "physical_samples": 0,
        "production_admission": False,
    }
    training_plan["run_plan_sha256"] = canonical_sha256_v1(training_plan)
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "evaluations").mkdir()
    (destination / "checkpoints").mkdir()
    _atomic_json(destination / "run_plan.json", training_plan)
    _write_run_state(
        destination,
        "preflight_complete",
        rollout_rows=len(batch.rewards),
        all_model_outputs_replay=True,
    )

    env = RealisticEdgeArmEnvV10(
        environment_config,
        seed=evaluation_seed_base,
        model_scene_path=scene_path,
    )
    adapter = StockGripperTaskFrameAdapterV22(env, action_config)
    renderer: MultiViewRGBRendererV1 | None = None
    actor_state_before = copy.deepcopy(bundle.actor.state_dict())
    critic_state_before = copy.deepcopy(bundle.critic.state_dict())
    pre_update_actor_sha256 = state_dict_sha256_v1(actor_state_before)
    pre_update_critic_sha256 = state_dict_sha256_v1(critic_state_before)
    started_at = _utc_now()
    try:
        renderer = MultiViewRGBRendererV1(
            env,
            height=policy_config.image_height,
            width=policy_config.image_width,
        )
        _write_run_state(destination, "baseline_evaluation")
        baseline = evaluate_asymmetric_multiview_policy_v1(
            env,
            renderer,
            adapter,
            bundle.actor,
            policy_config,
            seed_base=evaluation_seed_base,
            episodes=evaluation_episodes,
            potential_reward=task_reward,
            autoregressive_applied_action_feedback=True,
            execution_kernel=kernel,
            episode_condition_schedule=evaluation_condition_schedule,
        )
        baseline.update(
            {
                "evaluation_role": "pre_update_same_seed_baseline",
                "actor_state_sha256": pre_update_actor_sha256,
                "production_admission": False,
            }
        )
        baseline["payload_sha256"] = canonical_sha256_v1(baseline)
        baseline_curriculum_promotion = (
            curriculum_promotion_from_evaluation_v26(
                baseline,
                curriculum_stage_index,
            )
            if curriculum_stage_index is not None
            else None
        )
        baseline_path = destination / "evaluations" / "baseline.json"
        _atomic_json(baseline_path, baseline)
        _write_run_state(
            destination,
            "optimizer_update",
            baseline_path=str(baseline_path),
            baseline_strict_success_rate=baseline["strict_success_rate"],
        )
        ppo_metrics, reward_audit, optimizer = goal_directed_feasible_on_policy_ppo_update_v23(
            bundle.actor,
            bundle.critic,
            batch,
            policy_config,
            intrinsic_reward,
            eligibility,
            stratify_advantages_by_episode=True,
            align_cached_collection_numerics_v25=True,
        )
        optimizer_proposal_actor_state = copy.deepcopy(bundle.actor.state_dict())
        optimizer_proposal_critic_state = copy.deepcopy(bundle.critic.state_dict())
        optimizer_proposal_actor_sha256 = state_dict_sha256_v1(
            optimizer_proposal_actor_state
        )
        optimizer_proposal_critic_sha256 = state_dict_sha256_v1(
            optimizer_proposal_critic_state
        )
        backtracking_trials: list[dict[str, Any]] = []
        proposal: dict[str, Any] | None = None
        comparison: dict[str, Any] | None = None
        closed_loop_checks: dict[str, bool] | None = None
        proposal_actor_sha256: str | None = None
        proposal_critic_sha256: str | None = None
        accepted_parameter_scale: float | None = None

        for candidate_index, parameter_scale in enumerate(BACKTRACKING_SCALES_V25):
            candidate_actor_state = interpolate_state_dict_v25(
                actor_state_before,
                optimizer_proposal_actor_state,
                parameter_scale,
            )
            candidate_critic_state = interpolate_state_dict_v25(
                critic_state_before,
                optimizer_proposal_critic_state,
                parameter_scale,
            )
            bundle.actor.load_state_dict(candidate_actor_state, strict=True)
            bundle.critic.load_state_dict(candidate_critic_state, strict=True)
            candidate_actor_sha256 = state_dict_sha256_v1(bundle.actor.state_dict())
            candidate_critic_sha256 = state_dict_sha256_v1(bundle.critic.state_dict())
            _write_run_state(
                destination,
                "backtracking_candidate_evaluation",
                attempted_optimizer_steps=ppo_metrics.optimizer_steps,
                candidate_index=candidate_index,
                candidate_parameter_scale=parameter_scale,
                candidate_actor_state_sha256=candidate_actor_sha256,
            )
            candidate = evaluate_asymmetric_multiview_policy_v1(
                env,
                renderer,
                adapter,
                bundle.actor,
                policy_config,
                seed_base=evaluation_seed_base,
                episodes=evaluation_episodes,
                potential_reward=task_reward,
                autoregressive_applied_action_feedback=True,
                execution_kernel=kernel,
                episode_condition_schedule=evaluation_condition_schedule,
            )
            candidate.update(
                {
                    "evaluation_role": "v25_backtracking_same_seed_candidate",
                    "backtracking_candidate_index": candidate_index,
                    "parameter_scale": parameter_scale,
                    "actor_state_sha256": candidate_actor_sha256,
                    "production_admission": False,
                }
            )
            candidate["payload_sha256"] = canonical_sha256_v1(candidate)
            candidate_curriculum_promotion = (
                curriculum_promotion_from_evaluation_v26(
                    candidate,
                    curriculum_stage_index,
                )
                if curriculum_stage_index is not None
                else None
            )
            candidate_path = (
                destination
                / "evaluations"
                / f"candidate_scale_{scale_token_v25(parameter_scale)}.json"
            )
            _atomic_json(candidate_path, candidate)
            candidate_comparison = build_paired_comparison_v1(
                candidate,
                baseline,
                expected_evaluation_format=STOCK_GRIPPER_EVALUATION_FORMAT_V25,
            )
            candidate_effect_audit = meaningful_effect_audit_v26(
                candidate_comparison
            )
            candidate_checks = meaningful_effect_closed_loop_gates_v26(
                candidate_comparison
            )
            candidate_passed = all(candidate_checks.values())
            backtracking_trials.append(
                {
                    "candidate_index": candidate_index,
                    "parameter_scale": parameter_scale,
                    "actor_state_sha256": candidate_actor_sha256,
                    "critic_state_sha256": candidate_critic_sha256,
                    "evaluation_path": str(candidate_path),
                    "evaluation_sha256": sha256_file_v1(candidate_path),
                    "evaluation_payload_sha256": candidate["payload_sha256"],
                    "strict_success_rate": candidate["strict_success_rate"],
                    "curriculum_promotion": candidate_curriculum_promotion,
                    "closed_loop_gate_checks": candidate_checks,
                    "closed_loop_comparison": candidate_comparison,
                    "meaningful_effect_audit_v26": candidate_effect_audit,
                    "passed_all_gates": candidate_passed,
                }
            )
            proposal = candidate
            comparison = candidate_comparison
            closed_loop_checks = candidate_checks
            proposal_actor_sha256 = candidate_actor_sha256
            proposal_critic_sha256 = candidate_critic_sha256
            if candidate_passed:
                accepted_parameter_scale = parameter_scale
                break

        if not all(
            value is not None
            for value in (
                proposal,
                comparison,
                closed_loop_checks,
                proposal_actor_sha256,
                proposal_critic_sha256,
            )
        ):
            raise RuntimeError("V25 backtracking did not evaluate any candidate")
        assert proposal is not None
        assert comparison is not None
        assert closed_loop_checks is not None
        assert proposal_actor_sha256 is not None
        assert proposal_critic_sha256 is not None
        update_accepted = accepted_parameter_scale is not None
        proposal_path = destination / "evaluations" / "proposal.json"
        _atomic_json(proposal_path, proposal)
        if not update_accepted:
            bundle.actor.load_state_dict(actor_state_before, strict=True)
            bundle.critic.load_state_dict(critic_state_before, strict=True)
            optimizer = None
        committed_actor_sha256 = state_dict_sha256_v1(bundle.actor.state_dict())
        committed_critic_sha256 = state_dict_sha256_v1(bundle.critic.state_dict())
        if update_accepted and (
            committed_actor_sha256 != proposal_actor_sha256
            or committed_critic_sha256 != proposal_critic_sha256
        ):
            raise RuntimeError("accepted V25 backtracking candidate changed during commit")
        if not update_accepted and (
            committed_actor_sha256 != pre_update_actor_sha256
            or committed_critic_sha256 != pre_update_critic_sha256
        ):
            raise RuntimeError("rejected V25 search did not restore parent weights")
        record = {
            "format": FEASIBLE_MULTIVIEW_PPO_RUN_FORMAT_V22,
            "started_at_utc": started_at,
            "completed_at_utc": _utc_now(),
            "rollout_sha256": rollout_sha256,
            "pre_update_actor_state_sha256": pre_update_actor_sha256,
            "pre_update_critic_state_sha256": pre_update_critic_sha256,
            "optimizer_proposal_actor_state_sha256": (
                optimizer_proposal_actor_sha256
            ),
            "optimizer_proposal_critic_state_sha256": (
                optimizer_proposal_critic_sha256
            ),
            "proposal_actor_state_sha256": proposal_actor_sha256,
            "proposal_critic_state_sha256": proposal_critic_sha256,
            "committed_actor_state_sha256": committed_actor_sha256,
            "committed_critic_state_sha256": committed_critic_sha256,
            "update_accepted": update_accepted,
            "accepted_parameter_scale": accepted_parameter_scale,
            "backtracking_scale_schedule": list(BACKTRACKING_SCALES_V25),
            "backtracking_candidate_evaluation_count": len(backtracking_trials),
            "backtracking_trials": backtracking_trials,
            "attempted_optimizer_steps": ppo_metrics.optimizer_steps,
            "committed_optimizer_steps": (ppo_metrics.optimizer_steps if update_accepted else 0),
            "ppo_metrics": asdict(ppo_metrics),
            "reward_audit": asdict(reward_audit),
            "frontier_condition_admission": frontier_condition_admission,
            "rl_learning_source_admission_v40": rl_learning_source_admission,
            "v26_collection_contract_audit": v26_collection_contract_audit,
            "baseline_curriculum_promotion": baseline_curriculum_promotion,
            "proposal_curriculum_promotion": (
                backtracking_trials[-1]["curriculum_promotion"]
            ),
            "closed_loop_gate_checks": closed_loop_checks,
            "closed_loop_comparison": comparison,
            "baseline_evaluation": {
                "path": str(baseline_path),
                "sha256": sha256_file_v1(baseline_path),
                "payload_sha256": baseline["payload_sha256"],
                "strict_success_rate": baseline["strict_success_rate"],
            },
            "proposal_evaluation": {
                "path": str(proposal_path),
                "sha256": sha256_file_v1(proposal_path),
                "payload_sha256": proposal["payload_sha256"],
                "strict_success_rate": proposal["strict_success_rate"],
            },
            "recollect_required_before_any_next_update": True,
            "expert_calls": 0,
            "behavior_cloning_steps": 0,
            "physical_samples": 0,
            "production_admission": False,
        }
        _atomic_json(destination / "metrics.json", record)
        checkpoint = {
            "format": FEASIBLE_MULTIVIEW_PPO_CHECKPOINT_FORMAT_V22,
            "algorithm_format": MEANINGFUL_EFFECT_BACKTRACKING_FORMAT_V26,
            "source_type": SOURCE_TYPE,
            "created_at_utc": _utc_now(),
            "run_plan_sha256": training_plan["run_plan_sha256"],
            "source_rollout_sha256": rollout_sha256,
            "actor_state_dict": bundle.actor.state_dict(),
            "critic_state_dict": bundle.critic.state_dict(),
            "optimizer_state_dict": (None if optimizer is None else optimizer.state_dict()),
            "optimizer_state_origin": "unscaled_direction_proposal_v25",
            "optimizer_state_resume_supported": False,
            "actor_state_sha256": committed_actor_sha256,
            "critic_state_sha256": committed_critic_sha256,
            "ppo_config": asdict(policy_config),
            "intrinsic_reward_config": asdict(intrinsic_reward),
            "lineage": lineage,
            "metrics": record,
            "closed_loop_update_accepted": update_accepted,
            "accepted_parameter_scale": accepted_parameter_scale,
            "recollect_required_before_any_next_update": True,
            "expert_calls": 0,
            "behavior_cloning_steps": 0,
            "physical_samples": 0,
            "production_admission": False,
        }
        checkpoint_path = destination / "checkpoints" / "update_0001.pt"
        _atomic_torch_save(checkpoint_path, checkpoint)
        summary = {
            "format": FEASIBLE_MULTIVIEW_PPO_RUN_FORMAT_V22,
            "status": "complete",
            "started_at_utc": started_at,
            "completed_at_utc": _utc_now(),
            "update_accepted": update_accepted,
            "accepted_parameter_scale": accepted_parameter_scale,
            "backtracking_candidate_evaluation_count": len(backtracking_trials),
            "backtracking_trials": [
                {
                    "candidate_index": trial["candidate_index"],
                    "parameter_scale": trial["parameter_scale"],
                    "passed_all_gates": trial["passed_all_gates"],
                    "strict_success_rate": trial["strict_success_rate"],
                    "evaluation_path": trial["evaluation_path"],
                    "evaluation_sha256": trial["evaluation_sha256"],
                    "closed_loop_gate_checks": trial["closed_loop_gate_checks"],
                    "meaningful_effect_audit_v26": trial[
                        "meaningful_effect_audit_v26"
                    ],
                }
                for trial in backtracking_trials
            ],
            "attempted_optimizer_steps": ppo_metrics.optimizer_steps,
            "committed_optimizer_steps": (ppo_metrics.optimizer_steps if update_accepted else 0),
            "checkpoint_path": str(checkpoint_path),
            "checkpoint_sha256": sha256_file_v1(checkpoint_path),
            "metrics_path": str(destination / "metrics.json"),
            "metrics_sha256": sha256_file_v1(destination / "metrics.json"),
            "baseline_strict_success_rate": baseline["strict_success_rate"],
            "proposal_strict_success_rate": proposal["strict_success_rate"],
            "baseline_curriculum_promotion": baseline_curriculum_promotion,
            "proposal_curriculum_promotion": (
                backtracking_trials[-1]["curriculum_promotion"]
            ),
            "closed_loop_gate_checks": closed_loop_checks,
            "rl_learning_source_admission_v40": rl_learning_source_admission,
            "recollect_required_before_any_next_update": True,
            "expert_calls": 0,
            "behavior_cloning_steps": 0,
            "physical_samples": 0,
            "production_admission": False,
            "remaining_gates": [
                "fresh on-policy recollection before another optimizer update",
                "multi-seed obstacle and stress evaluation",
                "strict 3-second stable target success",
                "depth, segmentation, causal 4D, and physical calibration",
            ],
        }
        _atomic_json(destination / "summary.json", summary)
        _write_run_state(
            destination,
            "complete",
            update_accepted=update_accepted,
            accepted_parameter_scale=accepted_parameter_scale,
            backtracking_candidate_evaluation_count=len(backtracking_trials),
            checkpoint_path=str(checkpoint_path),
            checkpoint_sha256=summary["checkpoint_sha256"],
        )
        return summary
    except Exception as error:
        _write_run_state(
            destination,
            "failed",
            error_type=type(error).__name__,
            error=str(error),
        )
        _atomic_json(
            destination / "failure.json",
            {
                "format": FEASIBLE_MULTIVIEW_PPO_RUN_FORMAT_V22,
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


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rollout", type=Path, required=True)
    parser.add_argument("--collection-run-plan", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--evaluation-seed-base", type=int, default=25_980_000)
    parser.add_argument("--evaluation-episodes", type=int, default=1)
    parser.add_argument(
        "--ik-failure-penalty",
        type=float,
        default=GoalDirectedFeasibleRewardConfigV23().ik_failure_penalty,
        help=(
            "Explicit intrinsic penalty for non-converged IK transitions; the "
            "reward contract still enforces its bounded negative budget."
        ),
    )
    parser.add_argument("--device", choices=("auto", "cpu", "mps"), default="cpu")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    reward_config = replace(
        GoalDirectedFeasibleRewardConfigV23(),
        ik_failure_penalty=args.ik_failure_penalty,
    )
    reward_config.validate()
    result = run_feasible_multiview_ppo_update_v22(
        rollout_path=args.rollout,
        collection_run_plan_path=args.collection_run_plan,
        output_dir=args.output_dir,
        evaluation_seed_base=args.evaluation_seed_base,
        evaluation_episodes=args.evaluation_episodes,
        device=args.device,
        reward_config=reward_config,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "FEASIBLE_MULTIVIEW_PPO_CHECKPOINT_FORMAT_V22",
    "FEASIBLE_MULTIVIEW_PPO_RUN_FORMAT_V22",
    "model_replay_audit_v22",
    "run_feasible_multiview_ppo_update_v22",
]
