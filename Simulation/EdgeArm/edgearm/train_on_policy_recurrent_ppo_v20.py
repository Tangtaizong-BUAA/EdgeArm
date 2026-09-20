"""Train V20 by alternating exact closed-loop collection and proximal updates."""

from __future__ import annotations

import argparse
import copy
from dataclasses import asdict, replace
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from typing import Any, Sequence

import h5py
import numpy as np
import torch

from .asymmetric_multiview_ppo_v1 import (
    H5_FORMAT,
    SOURCE_TYPE,
    MultiViewRGBRendererV1,
    StockGripperTaskFrameAdapterV13,
    canonical_sha256_v1,
    collect_asymmetric_multiview_rollout_v1,
    evaluate_asymmetric_multiview_policy_v1,
    initialize_asymmetric_multiview_ppo_v1,
    parameter_counts_v1,
    sha256_file_v1,
    write_online_rollout_h5_v1,
)
from .evaluate_asymmetric_drq_sac_v1 import (
    _configs_from_parent_plan,
    build_paired_comparison_v1,
)
from .evaluate_asymmetric_shield_aware_drq_sac_v1 import (
    _load_v15_and_v14_actor_states,
)
from .on_policy_recurrent_ppo_v20 import (
    ON_POLICY_RECURRENT_CHECKPOINT_FORMAT_V20,
    ON_POLICY_RECURRENT_H5_FORMAT_V20,
    ON_POLICY_RECURRENT_PPO_FORMAT_V20,
    OnPolicyIntrinsicRewardConfigV20,
    closed_loop_update_gates_v20,
    on_policy_recurrent_ppo_update_v20,
)
from .ppo_utils_v1 import state_dict_sha256_v1
from .scratch_ppo_v6_candidate import ScratchPotentialRewardV6Candidate
from .sim2real_env_v10 import RealisticEdgeArmEnvV10


ON_POLICY_RECURRENT_RUN_FORMAT_V20 = (
    "edgearm-v20-closed-loop-on-policy-recurrent-ppo-run-v1"
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


def _append_jsonl(path: Path, payload: object) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
        stream.flush()


def _atomic_torch_save(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    torch.save(payload, partial)
    partial.replace(path)


def _resolve_device(requested: str) -> str:
    if requested == "cpu":
        return "cpu"
    if requested == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested but is unavailable")
        return "mps"
    if requested != "auto":
        raise ValueError(f"unsupported V20 device: {requested}")
    return "mps" if torch.backends.mps.is_available() else "cpu"


def _mean_network_sha256(state: dict[str, torch.Tensor]) -> str:
    mean_state = {name: value for name, value in state.items() if name != "log_std"}
    if len(mean_state) + 1 != len(state) or "log_std" not in state:
        raise ValueError("V20 actor state has no unique log_std parameter")
    return state_dict_sha256_v1(mean_state)


def _load_v15_actor_and_root_critic(
    parent_checkpoint: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    actor_state, _v14, root_plan, identities = _load_v15_and_v14_actor_states(
        parent_checkpoint
    )
    root_checkpoint_path = Path(
        identities["root_v13_checkpoint_path"]
    ).expanduser().resolve()
    root_checkpoint = torch.load(
        root_checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )
    critic_state = root_checkpoint.get("critic_state_dict")
    if (
        type(root_checkpoint) is not dict
        or not isinstance(critic_state, dict)
        or root_checkpoint.get("source_type") != SOURCE_TYPE
        or root_checkpoint.get("production_admission") is not False
        or sha256_file_v1(root_checkpoint_path)
        != identities["root_v13_checkpoint_sha256"]
    ):
        raise ValueError("V20 root V13 critic lineage is invalid")
    lineage = {
        "format": "edgearm-v15-to-v20-on-policy-lineage-v1",
        **identities,
        "parent_actor_mean_state_sha256": _mean_network_sha256(actor_state),
        "root_v13_critic_state_sha256": state_dict_sha256_v1(critic_state),
        "v17_weights_inherited": False,
        "v18_weights_inherited": False,
        "v19_weights_inherited": False,
        "offline_replay_actor_updates": 0,
        "expert_calls": 0,
        "behavior_cloning_steps": 0,
        "physical_samples": 0,
        "production_admission": False,
    }
    return actor_state, critic_state, root_plan, lineage


def _relabel_rollout_h5_v20(
    path: Path,
    *,
    actor_state_sha256: str,
    parent_checkpoint_path: str,
    parent_checkpoint_sha256: str,
    run_plan_sha256: str,
    update_index: int,
    exploration_standard_deviation: float,
) -> dict[str, Any]:
    with h5py.File(path, "r+") as stream:
        if str(stream.attrs.get("format", "")) != H5_FORMAT:
            raise ValueError("V20 relabel input is not the exact V13 writer")
        stream.attrs.update(
            {
                "format": ON_POLICY_RECURRENT_H5_FORMAT_V20,
                "base_writer_format": H5_FORMAT,
                "policy_format": ON_POLICY_RECURRENT_PPO_FORMAT_V20,
                "run_plan_sha256": run_plan_sha256,
                "v20_update_index": update_index,
                "actor_state_sha256_at_collection": actor_state_sha256,
                "parent_v15_checkpoint_path": parent_checkpoint_path,
                "parent_v15_checkpoint_sha256": parent_checkpoint_sha256,
                "closed_loop_on_policy": True,
                "policy_state_distribution": (
                    "exact pre-update actor AR and causal observation history"
                ),
                "autoregressive_feedback_mode": (
                    "requested_pre_tanh_zeroed_on_ik_failure"
                ),
                "infeasible_action_enters_next_ar_state": False,
                "recollect_after_every_actor_update": True,
                "ppo_training_eligible": True,
                "off_policy_replay_eligible": False,
                "exploration_standard_deviation": exploration_standard_deviation,
                "intrinsic_learning_reward_persisted": False,
                "persisted_shaped_reward_is_environment_source": True,
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
        "format": ON_POLICY_RECURRENT_H5_FORMAT_V20,
        "production_admission": False,
    }


def run_on_policy_recurrent_ppo_v20(
    *,
    parent_checkpoint: Path,
    output_dir: Path,
    updates: int,
    rollout_seed_base: int,
    initialization_seed: int,
    rollout_steps: int,
    learning_rate: float,
    batch_size: int,
    exploration_standard_deviation: float,
    action_autoregressive_rho: float,
    curriculum_tip_gap_band_m: tuple[float, float],
    obstacle_probability: float,
    stress_probability: float,
    evaluation_seed_base: int,
    evaluation_episodes: int,
    device: str,
    reward_config: OnPolicyIntrinsicRewardConfigV20 | None = None,
) -> dict[str, Any]:
    if type(updates) is not int or updates < 1:
        raise ValueError("V20 updates must be positive")
    if type(rollout_steps) is not int or rollout_steps < 32:
        raise ValueError("V20 rollout steps must be at least 32")
    if type(batch_size) is not int or batch_size < 16:
        raise ValueError("V20 batch size must be at least 16")
    numeric = np.asarray(
        [
            learning_rate,
            exploration_standard_deviation,
            action_autoregressive_rho,
            obstacle_probability,
            stress_probability,
            *curriculum_tip_gap_band_m,
        ],
        dtype=np.float64,
    )
    if not np.all(np.isfinite(numeric)):
        raise ValueError("V20 numeric configuration is non-finite")
    if not 0.0 < learning_rate <= 5.0e-5:
        raise ValueError("V20 learning rate must be in (0,5e-5]")
    if not 0.05 <= exploration_standard_deviation <= 0.50:
        raise ValueError("V20 exploration standard deviation is outside [0.05,0.50]")
    if not 0.0 <= action_autoregressive_rho < 0.99:
        raise ValueError("V20 action autoregressive rho must be in [0,0.99)")
    if (
        curriculum_tip_gap_band_m[0] < 0.00025
        or curriculum_tip_gap_band_m[0] >= curriculum_tip_gap_band_m[1]
        or curriculum_tip_gap_band_m[1] > 0.020
    ):
        raise ValueError("V20 near-contact curriculum band is invalid")
    if not 0.0 <= obstacle_probability <= 1.0 or not 0.0 <= stress_probability <= 1.0:
        raise ValueError("V20 domain probabilities must be in [0,1]")
    if type(evaluation_episodes) is not int or evaluation_episodes < 1:
        raise ValueError("V20 evaluation episode count must be positive")
    destination = Path(output_dir).expanduser().resolve()
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"V20 output directory is not empty: {destination}")

    resolved_device = _resolve_device(device)
    reward = reward_config or OnPolicyIntrinsicRewardConfigV20()
    reward.validate()
    actor_state, critic_state, root_plan, lineage = (
        _load_v15_actor_and_root_critic(parent_checkpoint)
    )
    environment_config, parent_policy_config, parent_action_config, scene_path = (
        _configs_from_parent_plan(root_plan)
    )
    policy_config = replace(
        parent_policy_config,
        rollout_steps=rollout_steps,
        update_epochs=1,
        batch_size=batch_size,
        learning_rate=learning_rate,
        action_autoregressive_rho=action_autoregressive_rho,
        clip_ratio=0.05,
        value_clip_ratio=0.10,
        entropy_coef=0.002,
        target_kl=0.003,
        obstacle_probability=obstacle_probability,
        stress_probability=stress_probability,
        auxiliary_view_dropout_probability=0.10,
        seed=initialization_seed,
    )
    policy_config.validate()
    action_config = replace(
        parent_action_config,
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
        raise RuntimeError("V20 failed to load the exact V15 parent actor")
    if _mean_network_sha256(bundle.actor.state_dict()) != lineage[
        "parent_actor_mean_state_sha256"
    ]:
        raise RuntimeError("V20 parent mean network changed while loading")
    with torch.no_grad():
        bundle.actor.log_std.fill_(math.log(exploration_standard_deviation))
    initial_actor_hash = state_dict_sha256_v1(bundle.actor.state_dict())
    if _mean_network_sha256(bundle.actor.state_dict()) != lineage[
        "parent_actor_mean_state_sha256"
    ]:
        raise RuntimeError("V20 exploration override changed deterministic actor means")

    source_hashes = {
        "edgearm/on_policy_recurrent_ppo_v20.py": sha256_file_v1(
            Path(__file__).with_name("on_policy_recurrent_ppo_v20.py")
        ),
        "edgearm/train_on_policy_recurrent_ppo_v20.py": sha256_file_v1(
            Path(__file__).resolve()
        ),
        "scene": sha256_file_v1(scene_path),
    }
    plan = {
        "format": ON_POLICY_RECURRENT_RUN_FORMAT_V20,
        "algorithm_format": ON_POLICY_RECURRENT_PPO_FORMAT_V20,
        "created_at_utc": _utc_now(),
        "output_dir": str(destination),
        "source_type": SOURCE_TYPE,
        "updates": updates,
        "rollout_seed_base": rollout_seed_base,
        "initialization_seed": initialization_seed,
        "resolved_device": resolved_device,
        "environment_config": asdict(environment_config),
        "ppo_config": asdict(policy_config),
        "action_adapter_config": asdict(action_config),
        "intrinsic_reward_config": asdict(reward),
        "scene_path": str(scene_path),
        "source_hashes": source_hashes,
        "lineage": lineage,
        "parent_actor_state_sha256": parent_actor_hash,
        "initial_actor_state_sha256_after_std_override": initial_actor_hash,
        "exploration_standard_deviation": exploration_standard_deviation,
        "action_autoregressive_rho": action_autoregressive_rho,
        "parent_deterministic_mean_preserved_before_first_update": True,
        "closed_loop_on_policy_collection": True,
        "recollect_after_every_actor_update": True,
        "autoregressive_feedback_mode": (
            "requested_pre_tanh_zeroed_on_ik_failure"
        ),
        "infeasible_action_enters_next_ar_state": False,
        "closed_loop_update_gate": True,
        "closed_loop_gate_seed_base": evaluation_seed_base,
        "closed_loop_gate_requires_primary_improvement": True,
        "closed_loop_gate_requires_contact_distance_progress_ik_non_regression": True,
        "failed_proposal_rolls_back_actor_critic_and_optimizer": True,
        "replay_previous_ar_state_for_actor_update": False,
        "offline_replay_actor_updates": 0,
        "evaluation_seed_base": evaluation_seed_base,
        "evaluation_episodes_per_update": evaluation_episodes,
        "parameter_counts": parameter_counts_v1(bundle),
        "expert_calls": 0,
        "behavior_cloning_steps": 0,
        "physical_samples": 0,
        "production_admission": False,
    }
    plan["run_plan_sha256"] = canonical_sha256_v1(plan)
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "rollouts").mkdir()
    (destination / "checkpoints").mkdir()
    (destination / "evaluations").mkdir()
    _atomic_json(destination / "run_plan.json", plan)

    env = RealisticEdgeArmEnvV10(
        environment_config,
        seed=rollout_seed_base,
        model_scene_path=scene_path,
    )
    renderer: MultiViewRGBRendererV1 | None = None
    adapter = StockGripperTaskFrameAdapterV13(env, action_config)
    optimizer: torch.optim.Optimizer | None = None
    records: list[dict[str, Any]] = []
    started_at = _utc_now()
    try:
        renderer = MultiViewRGBRendererV1(
            env,
            height=policy_config.image_height,
            width=policy_config.image_width,
        )
        potential_reward = ScratchPotentialRewardV6Candidate()
        for update_index in range(1, updates + 1):
            rollout_seed = rollout_seed_base + (update_index - 1) * 100_000
            update_config = replace(
                policy_config,
                seed=initialization_seed + update_index - 1,
            )
            pre_update_actor_hash = state_dict_sha256_v1(bundle.actor.state_dict())
            actor_state_before = copy.deepcopy(bundle.actor.state_dict())
            critic_state_before = copy.deepcopy(bundle.critic.state_dict())
            optimizer_state_before = (
                None if optimizer is None else copy.deepcopy(optimizer.state_dict())
            )
            rollout = collect_asymmetric_multiview_rollout_v1(
                env,
                renderer,
                adapter,
                bundle.actor,
                bundle.critic,
                update_config,
                seed=rollout_seed,
                potential_reward=potential_reward,
                autoregressive_zero_on_ik_failure=True,
            )
            base_rollout_path = destination / "rollouts" / f"update_{update_index:04d}.h5"
            write_online_rollout_h5_v1(
                base_rollout_path,
                rollout,
                update_index=update_index,
                provenance=bundle.provenance,
                config=update_config,
                scene_sha256=source_hashes["scene"],
                resume_lineage=lineage,
            )
            rollout_artifact = _relabel_rollout_h5_v20(
                base_rollout_path,
                actor_state_sha256=pre_update_actor_hash,
                parent_checkpoint_path=lineage["candidate_checkpoint_path"],
                parent_checkpoint_sha256=lineage["candidate_checkpoint_sha256"],
                run_plan_sha256=plan["run_plan_sha256"],
                update_index=update_index,
                exploration_standard_deviation=exploration_standard_deviation,
            )
            baseline = evaluate_asymmetric_multiview_policy_v1(
                env,
                renderer,
                adapter,
                bundle.actor,
                update_config,
                seed_base=evaluation_seed_base,
                episodes=evaluation_episodes,
                potential_reward=potential_reward,
                autoregressive_zero_on_ik_failure=True,
            )
            baseline.update(
                {
                    "v20_update_index": update_index,
                    "evaluation_role": "pre_update_closed_loop_gate_baseline",
                    "actor_state_sha256": pre_update_actor_hash,
                    "production_admission": False,
                }
            )
            baseline["payload_sha256"] = canonical_sha256_v1(baseline)
            baseline_path = (
                destination
                / "evaluations"
                / f"baseline_update_{update_index:04d}.json"
            )
            _atomic_json(baseline_path, baseline)
            ppo_metrics, reward_audit, optimizer = (
                on_policy_recurrent_ppo_update_v20(
                    bundle.actor,
                    bundle.critic,
                    rollout,
                    update_config,
                    reward,
                    optimizer=optimizer,
                )
            )
            proposal_actor_hash = state_dict_sha256_v1(bundle.actor.state_dict())
            heldout = evaluate_asymmetric_multiview_policy_v1(
                env,
                renderer,
                adapter,
                bundle.actor,
                update_config,
                seed_base=evaluation_seed_base,
                episodes=evaluation_episodes,
                potential_reward=potential_reward,
                autoregressive_zero_on_ik_failure=True,
            )
            heldout.update(
                {
                    "v20_update_index": update_index,
                    "evaluation_role": "post_update_closed_loop_gate_proposal",
                    "actor_state_sha256": proposal_actor_hash,
                    "production_admission": False,
                }
            )
            heldout["payload_sha256"] = canonical_sha256_v1(heldout)
            heldout_path = destination / "evaluations" / f"update_{update_index:04d}.json"
            _atomic_json(heldout_path, heldout)
            gate_comparison = build_paired_comparison_v1(heldout, baseline)
            gate_checks = closed_loop_update_gates_v20(gate_comparison)
            update_accepted = all(gate_checks.values())
            if not update_accepted:
                bundle.actor.load_state_dict(actor_state_before, strict=True)
                bundle.critic.load_state_dict(critic_state_before, strict=True)
                if optimizer_state_before is None:
                    optimizer = None
                else:
                    if optimizer is None:  # pragma: no cover - update creates it
                        raise RuntimeError("V20 lost its optimizer before rollback")
                    optimizer.load_state_dict(optimizer_state_before)
            post_update_actor_hash = state_dict_sha256_v1(bundle.actor.state_dict())
            if update_accepted and post_update_actor_hash != proposal_actor_hash:
                raise RuntimeError("V20 accepted proposal changed during gate handling")
            if not update_accepted and post_update_actor_hash != pre_update_actor_hash:
                raise RuntimeError("V20 rejected proposal did not restore the actor")
            record = {
                "format": ON_POLICY_RECURRENT_PPO_FORMAT_V20,
                "update_index": update_index,
                "rollout_seed": rollout_seed,
                "pre_update_actor_state_sha256": pre_update_actor_hash,
                "proposal_actor_state_sha256": proposal_actor_hash,
                "post_update_actor_state_sha256": post_update_actor_hash,
                "closed_loop_update_accepted": update_accepted,
                "closed_loop_gate_checks": gate_checks,
                "closed_loop_gate_comparison": gate_comparison,
                "recollected_after_previous_update": update_index > 1,
                "rollout_transition_count": len(rollout.rewards),
                "rollout_episode_count": rollout.completed_episode_count,
                "rollout_valid_contact_transition_count": int(
                    np.count_nonzero(rollout.valid_push_side_contact_any)
                ),
                "rollout_ik_failure_transition_count": int(
                    np.count_nonzero(~rollout.ik_converged)
                ),
                "rollout_mean_policy_task_action": (
                    rollout.policy_action.mean(axis=0).tolist()
                ),
                "rollout_mean_applied_task_action": (
                    rollout.applied_task_action.mean(axis=0).tolist()
                ),
                "reward_audit": asdict(reward_audit),
                "ppo": asdict(ppo_metrics),
                "rollout_artifact": rollout_artifact,
                "baseline_evaluation_artifact": {
                    "path": str(baseline_path),
                    "sha256": sha256_file_v1(baseline_path),
                    "payload_sha256": baseline["payload_sha256"],
                    "strict_success_rate": baseline["strict_success_rate"],
                    "mean_final_block_target_distance_m": baseline[
                        "mean_final_block_target_distance_m"
                    ],
                },
                "heldout_evaluation_artifact": {
                    "path": str(heldout_path),
                    "sha256": sha256_file_v1(heldout_path),
                    "payload_sha256": heldout["payload_sha256"],
                    "strict_success_rate": heldout["strict_success_rate"],
                    "mean_final_block_target_distance_m": heldout[
                        "mean_final_block_target_distance_m"
                    ],
                    "valid_contact_transition_count": int(
                        sum(
                            int(row["valid_push_side_contact_transition_count"])
                            for row in heldout["episodes"]
                        )
                    ),
                    "ik_failure_step_count": int(
                        sum(int(row["ik_failure_step_count"]) for row in heldout["episodes"])
                    ),
                },
                "expert_calls": 0,
                "behavior_cloning_steps": 0,
                "physical_samples": 0,
                "production_admission": False,
            }
            records.append(record)
            _append_jsonl(destination / "metrics.jsonl", record)
            checkpoint = {
                "format": ON_POLICY_RECURRENT_CHECKPOINT_FORMAT_V20,
                "algorithm_format": ON_POLICY_RECURRENT_PPO_FORMAT_V20,
                "source_type": SOURCE_TYPE,
                "created_at_utc": _utc_now(),
                "update_index": update_index,
                "run_plan_sha256": plan["run_plan_sha256"],
                "actor_state_dict": bundle.actor.state_dict(),
                "critic_state_dict": bundle.critic.state_dict(),
                "optimizer_state_dict": (
                    None if optimizer is None else optimizer.state_dict()
                ),
                "actor_state_sha256": post_update_actor_hash,
                "critic_state_sha256": state_dict_sha256_v1(
                    bundle.critic.state_dict()
                ),
                "ppo_config": asdict(update_config),
                "intrinsic_reward_config": asdict(reward),
                "lineage": lineage,
                "metrics": record,
                "closed_loop_update_accepted": update_accepted,
                "expert_calls": 0,
                "behavior_cloning_steps": 0,
                "physical_samples": 0,
                "production_admission": False,
            }
            checkpoint_path = destination / "checkpoints" / f"update_{update_index:04d}.pt"
            _atomic_torch_save(checkpoint_path, checkpoint)
            _atomic_json(
                destination / "run_state.json",
                {
                    "format": ON_POLICY_RECURRENT_RUN_FORMAT_V20,
                    "status": "running" if update_index < updates else "complete",
                    "last_completed_update": update_index,
                    "last_closed_loop_update_accepted": update_accepted,
                    "last_checkpoint": str(checkpoint_path),
                    "last_checkpoint_sha256": sha256_file_v1(checkpoint_path),
                    "updated_at_utc": _utc_now(),
                    "production_admission": False,
                },
            )
    except Exception as error:
        _atomic_json(
            destination / "failure.json",
            {
                "format": ON_POLICY_RECURRENT_RUN_FORMAT_V20,
                "status": "failed",
                "error_type": type(error).__name__,
                "error": str(error),
                "completed_updates": len(records),
                "failed_at_utc": _utc_now(),
                "production_admission": False,
            },
        )
        raise
    finally:
        if renderer is not None:
            renderer.close()
    final_checkpoint = destination / "checkpoints" / f"update_{updates:04d}.pt"
    summary = {
        "format": ON_POLICY_RECURRENT_RUN_FORMAT_V20,
        "status": "complete",
        "started_at_utc": started_at,
        "completed_at_utc": _utc_now(),
        "updates": updates,
        "accepted_updates": sum(
            int(row["closed_loop_update_accepted"]) for row in records
        ),
        "rejected_updates": sum(
            int(not row["closed_loop_update_accepted"]) for row in records
        ),
        "total_transitions": sum(row["rollout_transition_count"] for row in records),
        "total_episodes": sum(row["rollout_episode_count"] for row in records),
        "total_valid_contact_transitions": sum(
            row["rollout_valid_contact_transition_count"] for row in records
        ),
        "total_ik_failure_transitions": sum(
            row["rollout_ik_failure_transition_count"] for row in records
        ),
        "final_checkpoint": str(final_checkpoint),
        "final_checkpoint_sha256": sha256_file_v1(final_checkpoint),
        "final_checkpoint_closed_loop_update_accepted": records[-1][
            "closed_loop_update_accepted"
        ],
        "last_metrics": records[-1],
        "lineage": lineage,
        "expert_calls": 0,
        "behavior_cloning_steps": 0,
        "physical_samples": 0,
        "production_admission": False,
        "remaining_gates": [
            "same-seed paired task and IK improvement over V15",
            "multi-seed near-contact validation",
            "obstacle and stress held-out evaluation",
            "strict-success evaluation",
            "depth, segmentation, causal 4D, and physical calibration",
        ],
    }
    _atomic_json(destination / "summary.json", summary)
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--updates", type=int, default=1)
    parser.add_argument("--rollout-seed-base", type=int, default=25_700_000)
    parser.add_argument("--initialization-seed", type=int, default=25_600_000)
    parser.add_argument("--rollout-steps", type=int, default=360)
    parser.add_argument("--learning-rate", type=float, default=1.0e-5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--exploration-standard-deviation", type=float, default=0.20)
    parser.add_argument("--action-autoregressive-rho", type=float, default=0.80)
    parser.add_argument("--curriculum-min-tip-gap-m", type=float, default=0.00025)
    parser.add_argument("--curriculum-max-tip-gap-m", type=float, default=0.0048)
    parser.add_argument("--obstacle-probability", type=float, default=0.0)
    parser.add_argument("--stress-probability", type=float, default=0.0)
    parser.add_argument("--evaluation-seed-base", type=int, default=95_000_000)
    parser.add_argument("--evaluation-episodes", type=int, default=1)
    parser.add_argument("--device", choices=("auto", "cpu", "mps"), default="auto")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    summary = run_on_policy_recurrent_ppo_v20(
        parent_checkpoint=args.parent_checkpoint,
        output_dir=args.output_dir,
        updates=args.updates,
        rollout_seed_base=args.rollout_seed_base,
        initialization_seed=args.initialization_seed,
        rollout_steps=args.rollout_steps,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        exploration_standard_deviation=args.exploration_standard_deviation,
        action_autoregressive_rho=args.action_autoregressive_rho,
        curriculum_tip_gap_band_m=(
            args.curriculum_min_tip_gap_m,
            args.curriculum_max_tip_gap_m,
        ),
        obstacle_probability=args.obstacle_probability,
        stress_probability=args.stress_probability,
        evaluation_seed_base=args.evaluation_seed_base,
        evaluation_episodes=args.evaluation_episodes,
        device=args.device,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
