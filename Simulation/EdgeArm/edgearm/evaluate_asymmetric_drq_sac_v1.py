"""Paired closed-loop evaluation for a V14 DrQ-SAC actor and its V13 parent.

The two actors run deterministic policies in fresh copies of the exact V13
plant, scene, action adapter, and held-out seeds.  This is a diagnostic gate:
it reports paired task and safety deltas but never grants production admission.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from .asymmetric_drq_sac_v1 import (
    ASYMMETRIC_DRQ_SAC_CHECKPOINT_FORMAT_V1,
    ASYMMETRIC_DRQ_SAC_FORMAT_V1,
)
from .asymmetric_multiview_ppo_v1 import (
    EVALUATION_FORMAT,
    POLICY_FORMAT,
    SOURCE_TYPE,
    AsymmetricMultiViewPPOConfigV1,
    MultiViewRGBRendererV1,
    SelectedViewRecurrentActorV1,
    StockGripperTaskFrameAdapterV13,
    StockGripperTaskSpaceActionConfigV12,
    canonical_sha256_v1,
    evaluate_asymmetric_multiview_policy_v1,
    sha256_file_v1,
)
from .ppo_utils_v1 import state_dict_sha256_v1
from .sim2real_env_v10 import RealisticEdgeArmEnvV10, RealisticEnvV10Config
from .stock_gripper_action_guard_v3 import StockGripperActionGuardConfigV3
from .train_asymmetric_drq_sac_v1 import (
    OFFLINE_RUN_FORMAT_V1,
    _load_v13_parent_actor,
)


V14_EVALUATION_FORMAT_V1 = "edgearm-v14-asymmetric-drq-sac-heldout-evaluation-v1"
V14_PAIRED_COMPARISON_FORMAT_V1 = "edgearm-v14-v13-same-seed-closed-loop-comparison-v1"


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


def _resolve_device(requested: str) -> str:
    if requested == "cpu":
        return "cpu"
    if requested == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested but is unavailable")
        return "mps"
    if requested != "auto":
        raise ValueError(f"unsupported evaluation device: {requested}")
    return "mps" if torch.backends.mps.is_available() else "cpu"


def _load_json_dictionary(path: Path, *, label: str) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if type(payload) is not dict:
        raise TypeError(f"{label} must be a JSON dictionary")
    return payload


def _validate_run_plan_hash(plan: dict[str, Any], *, label: str) -> str:
    stored = plan.get("run_plan_sha256")
    unhashed = dict(plan)
    unhashed.pop("run_plan_sha256", None)
    if not isinstance(stored, str) or stored != canonical_sha256_v1(unhashed):
        raise ValueError(f"{label} run-plan hash chain is invalid")
    return stored


def _load_v14_and_parent_actor_states(
    checkpoint_path: Path,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
]:
    path = Path(checkpoint_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"V14 checkpoint is missing: {path}")
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if type(checkpoint) is not dict:
        raise TypeError("V14 checkpoint must be a dictionary")
    required = {
        "format": ASYMMETRIC_DRQ_SAC_CHECKPOINT_FORMAT_V1,
        "algorithm_format": ASYMMETRIC_DRQ_SAC_FORMAT_V1,
        "source_type": SOURCE_TYPE,
        "expert_calls": 0,
        "behavior_cloning_steps": 0,
        "physical_samples": 0,
        "production_admission": False,
    }
    for key, expected in required.items():
        if checkpoint.get(key) != expected:
            raise ValueError(f"V14 checkpoint mismatch for {key}")

    run_plan_path = path.parent.parent / "run_plan.json"
    if not run_plan_path.is_file():
        raise FileNotFoundError("V14 checkpoint run_plan.json is missing")
    run_plan = _load_json_dictionary(run_plan_path, label="V14")
    run_plan_hash = _validate_run_plan_hash(run_plan, label="V14")
    if (
        run_plan.get("format") != OFFLINE_RUN_FORMAT_V1
        or run_plan.get("algorithm_format") != ASYMMETRIC_DRQ_SAC_FORMAT_V1
        or checkpoint.get("run_plan_sha256") != run_plan_hash
    ):
        raise ValueError("V14 checkpoint and run plan do not form one lineage")

    candidate_state = checkpoint.get("actor_state_dict")
    if not isinstance(candidate_state, dict):
        raise TypeError("V14 candidate actor state is missing")
    candidate_probe = SelectedViewRecurrentActorV1(task_count=1)
    candidate_probe.load_state_dict(candidate_state, strict=True)
    candidate_hash = state_dict_sha256_v1(candidate_probe.state_dict())

    parent_lineage = checkpoint.get("parent_lineage")
    if type(parent_lineage) is not dict:
        raise TypeError("V14 parent lineage is missing")
    parent_path_value = parent_lineage.get("parent_checkpoint_path")
    if not isinstance(parent_path_value, str):
        raise TypeError("V14 parent checkpoint path is missing")
    parent_path = Path(parent_path_value).expanduser().resolve()
    parent_state, verified_lineage = _load_v13_parent_actor(parent_path)
    for key in (
        "parent_checkpoint_sha256",
        "parent_actor_state_sha256",
        "parent_run_plan_sha256",
    ):
        if parent_lineage.get(key) != verified_lineage.get(key):
            raise ValueError(f"V14 parent lineage mismatch for {key}")
    if sha256_file_v1(parent_path) != parent_lineage["parent_checkpoint_sha256"]:
        raise ValueError("V13 parent checkpoint bytes changed")

    parent_plan_path = Path(verified_lineage["parent_run_plan_path"])
    parent_plan = _load_json_dictionary(parent_plan_path, label="V13 parent")
    if (
        _validate_run_plan_hash(parent_plan, label="V13 parent")
        != (verified_lineage["parent_run_plan_sha256"])
    ):
        raise ValueError("V13 parent run-plan hash changed")
    if parent_plan.get("policy_format") != POLICY_FORMAT:
        raise ValueError("V13 parent policy format changed")

    identities = {
        "candidate_checkpoint_path": str(path),
        "candidate_checkpoint_sha256": sha256_file_v1(path),
        "candidate_update_index": int(checkpoint["update_index"]),
        "candidate_actor_state_sha256": candidate_hash,
        "candidate_run_plan_path": str(run_plan_path.resolve()),
        "candidate_run_plan_sha256": run_plan_hash,
        "parent_checkpoint_path": str(parent_path),
        "parent_checkpoint_sha256": verified_lineage["parent_checkpoint_sha256"],
        "parent_update_index": verified_lineage["parent_update_index"],
        "parent_actor_state_sha256": verified_lineage["parent_actor_state_sha256"],
        "parent_run_plan_path": str(parent_plan_path.resolve()),
        "parent_run_plan_sha256": verified_lineage["parent_run_plan_sha256"],
    }
    return candidate_state, parent_state, run_plan, parent_plan, identities


def _configs_from_parent_plan(
    parent_plan: dict[str, Any],
) -> tuple[
    RealisticEnvV10Config,
    AsymmetricMultiViewPPOConfigV1,
    StockGripperTaskSpaceActionConfigV12,
    Path,
]:
    environment_payload = parent_plan.get("environment_config")
    policy_payload = parent_plan.get("ppo_config")
    action_contract = parent_plan.get("action_contract")
    if type(environment_payload) is not dict or type(policy_payload) is not dict:
        raise TypeError("V13 parent environment or PPO config is missing")
    if type(action_contract) is not dict:
        raise TypeError("V13 parent action contract is missing")
    action_payload = action_contract.get("adapter_config")
    if type(action_payload) is not dict:
        raise TypeError("V13 parent action-adapter config is missing")
    action_payload = dict(action_payload)
    guard_payload = action_payload.pop("guard", None)
    if type(guard_payload) is not dict:
        raise TypeError("V13 parent guard config is missing")

    environment = RealisticEnvV10Config(**environment_payload)
    policy = AsymmetricMultiViewPPOConfigV1(**policy_payload)
    guard = StockGripperActionGuardConfigV3(**guard_payload)
    action = StockGripperTaskSpaceActionConfigV12(
        **action_payload,
        guard=guard,
    )
    policy.validate()
    action.validate()

    scene_value = parent_plan.get("scene_path")
    if not isinstance(scene_value, str):
        raise TypeError("V13 parent scene path is missing")
    scene = Path(scene_value).expanduser().resolve()
    if not scene.is_file() or sha256_file_v1(scene) != parent_plan.get("scene_sha256"):
        raise ValueError("V13 parent scene bytes changed or are missing")
    return environment, policy, action, scene


def _evaluate_actor_state(
    actor_state: dict[str, Any],
    *,
    environment_config: RealisticEnvV10Config,
    policy_config: AsymmetricMultiViewPPOConfigV1,
    action_config: StockGripperTaskSpaceActionConfigV12,
    scene_path: Path,
    seed_base: int,
    episodes: int,
    device: str,
    autoregressive_applied_action_feedback: bool = False,
    autoregressive_zero_on_ik_failure: bool = False,
) -> dict[str, Any]:
    actor = SelectedViewRecurrentActorV1(task_count=1).to(device)
    actor.load_state_dict(actor_state, strict=True)
    actor.eval()
    env = RealisticEdgeArmEnvV10(
        environment_config,
        seed=seed_base,
        model_scene_path=scene_path,
    )
    adapter = StockGripperTaskFrameAdapterV13(env, action_config)
    renderer = MultiViewRGBRendererV1(
        env,
        height=policy_config.image_height,
        width=policy_config.image_width,
    )
    try:
        return evaluate_asymmetric_multiview_policy_v1(
            env,
            renderer,
            adapter,
            actor,
            policy_config,
            seed_base=seed_base,
            episodes=episodes,
            autoregressive_applied_action_feedback=(autoregressive_applied_action_feedback),
            autoregressive_zero_on_ik_failure=(autoregressive_zero_on_ik_failure),
        )
    finally:
        renderer.close()


def _mean_episode_metric(evaluation: dict[str, Any], name: str) -> float:
    episodes = evaluation.get("episodes")
    if not isinstance(episodes, list) or not episodes:
        raise ValueError("paired evaluation has no episodes")
    values = [float(item[name]) for item in episodes]
    if not np.all(np.isfinite(values)):
        raise ValueError(f"paired evaluation metric {name} is non-finite")
    return float(np.mean(values))


def build_paired_comparison_v1(
    candidate: dict[str, Any],
    parent: dict[str, Any],
    *,
    expected_evaluation_format: str = EVALUATION_FORMAT,
) -> dict[str, Any]:
    """Build same-seed candidate-minus-parent task and safety deltas."""

    if not isinstance(expected_evaluation_format, str) or not expected_evaluation_format:
        raise ValueError("paired comparison evaluation format is missing")

    for label, evaluation in (("candidate", candidate), ("parent", parent)):
        if evaluation.get("format") != expected_evaluation_format:
            raise ValueError(f"{label} did not use the requested evaluator kernel")
        if evaluation.get("production_admission") is not False:
            raise ValueError(f"{label} evaluation cannot be production-admitted")
    candidate_episodes = candidate.get("episodes")
    parent_episodes = parent.get("episodes")
    if not isinstance(candidate_episodes, list) or not isinstance(parent_episodes, list):
        raise TypeError("paired evaluations must contain episode lists")
    if len(candidate_episodes) != len(parent_episodes) or not candidate_episodes:
        raise ValueError("paired evaluations must have equal nonzero episode counts")
    candidate_seeds = [int(item["seed"]) for item in candidate_episodes]
    parent_seeds = [int(item["seed"]) for item in parent_episodes]
    if candidate_seeds != parent_seeds:
        raise ValueError("candidate and parent did not execute identical seeds")
    candidate_conditions = [
        (
            int(item.get("episode_index", index)),
            item.get("obstacle"),
            item.get("stress"),
        )
        for index, item in enumerate(candidate_episodes)
    ]
    parent_conditions = [
        (
            int(item.get("episode_index", index)),
            item.get("obstacle"),
            item.get("stress"),
        )
        for index, item in enumerate(parent_episodes)
    ]
    if candidate_conditions != parent_conditions:
        raise ValueError("candidate and parent did not execute identical conditions")

    def total(name: str, rows: list[dict[str, Any]]) -> int:
        return int(sum(int(item[name]) for item in rows))

    candidate_safety = total("episode_safety_violation", candidate_episodes)
    parent_safety = total("episode_safety_violation", parent_episodes)
    candidate_contact = total("valid_push_side_contact_transition_count", candidate_episodes)
    parent_contact = total("valid_push_side_contact_transition_count", parent_episodes)
    candidate_shield = total("action_shield_rejection_step_count", candidate_episodes)
    parent_shield = total("action_shield_rejection_step_count", parent_episodes)
    candidate_ik_failure = total("ik_failure_step_count", candidate_episodes)
    parent_ik_failure = total("ik_failure_step_count", parent_episodes)
    distance_delta = float(
        candidate["mean_final_block_target_distance_m"] - parent["mean_final_block_target_distance_m"]
    )
    coverage_delta = float(candidate["mean_final_target_coverage"] - parent["mean_final_target_coverage"])
    block_progress_delta = _mean_episode_metric(candidate, "final_block_progress") - _mean_episode_metric(
        parent, "final_block_progress"
    )
    final_potential_delta = _mean_episode_metric(candidate, "final_potential") - _mean_episode_metric(
        parent, "final_potential"
    )
    application_scale_delta = _mean_episode_metric(
        candidate, "mean_ik_application_scale"
    ) - _mean_episode_metric(parent, "mean_ik_application_scale")
    success_delta = float(candidate["strict_success_rate"] - parent["strict_success_rate"])
    safety_regressed = candidate_safety > parent_safety
    improvement_signals = {
        "strict_success_rate_increased": success_delta > 0.0,
        "valid_contact_transitions_increased": (candidate_contact > parent_contact),
        "mean_final_distance_decreased": distance_delta < -1.0e-6,
        "mean_final_block_progress_increased": block_progress_delta > 1.0e-6,
        "mean_final_potential_increased": final_potential_delta > 1.0e-6,
    }
    primary_task_improved = any(
        improvement_signals[name]
        for name in (
            "strict_success_rate_increased",
            "valid_contact_transitions_increased",
            "mean_final_distance_decreased",
            "mean_final_block_progress_increased",
        )
    )
    executability_signals = {
        "ik_failure_steps_decreased": (candidate_ik_failure < parent_ik_failure),
        "mean_ik_application_scale_increased": (application_scale_delta > 1.0e-6),
    }
    paired_rows = []
    for candidate_row, parent_row in zip(candidate_episodes, parent_episodes):
        paired_rows.append(
            {
                "episode_index": int(
                    candidate_row.get("episode_index", len(paired_rows))
                ),
                "seed": int(candidate_row["seed"]),
                "obstacle": candidate_row.get("obstacle"),
                "stress": candidate_row.get("stress"),
                "candidate_strict_success": bool(candidate_row["strict_success"]),
                "parent_strict_success": bool(parent_row["strict_success"]),
                "final_distance_delta_m": float(
                    candidate_row["final_block_target_distance_m"]
                    - parent_row["final_block_target_distance_m"]
                ),
                "final_block_progress_delta": float(
                    candidate_row["final_block_progress"] - parent_row["final_block_progress"]
                ),
                "valid_contact_transition_delta": int(
                    candidate_row["valid_push_side_contact_transition_count"]
                    - parent_row["valid_push_side_contact_transition_count"]
                ),
                "shield_rejection_step_delta": int(
                    candidate_row["action_shield_rejection_step_count"]
                    - parent_row["action_shield_rejection_step_count"]
                ),
                "ik_failure_step_delta": int(
                    candidate_row["ik_failure_step_count"] - parent_row["ik_failure_step_count"]
                ),
                "mean_ik_application_scale_delta": float(
                    candidate_row["mean_ik_application_scale"] - parent_row["mean_ik_application_scale"]
                ),
                "candidate_safety_violation": bool(candidate_row["episode_safety_violation"]),
                "parent_safety_violation": bool(parent_row["episode_safety_violation"]),
            }
        )
    return {
        "format": V14_PAIRED_COMPARISON_FORMAT_V1,
        "paired_seed_identity": True,
        "paired_condition_identity": True,
        "episode_count": len(candidate_episodes),
        "seeds": candidate_seeds,
        "candidate_minus_parent": {
            "strict_success_rate": success_delta,
            "valid_contact_transition_count": (candidate_contact - parent_contact),
            "mean_final_block_target_distance_m": distance_delta,
            "mean_final_target_coverage": coverage_delta,
            "mean_final_block_progress": block_progress_delta,
            "mean_final_potential": final_potential_delta,
            "action_shield_rejection_step_count": (candidate_shield - parent_shield),
            "ik_failure_step_count": (candidate_ik_failure - parent_ik_failure),
            "mean_ik_application_scale": application_scale_delta,
            "safety_violation_episode_count": (candidate_safety - parent_safety),
        },
        "candidate_totals": {
            "valid_contact_transition_count": candidate_contact,
            "action_shield_rejection_step_count": candidate_shield,
            "ik_failure_step_count": candidate_ik_failure,
            "safety_violation_episode_count": candidate_safety,
        },
        "parent_totals": {
            "valid_contact_transition_count": parent_contact,
            "action_shield_rejection_step_count": parent_shield,
            "ik_failure_step_count": parent_ik_failure,
            "safety_violation_episode_count": parent_safety,
        },
        "improvement_signals": improvement_signals,
        "executability_signals": executability_signals,
        "online_executability_improved": any(executability_signals.values()),
        "primary_task_improved": primary_task_improved,
        "safety_regressed": safety_regressed,
        "diagnostic_net_improvement": (not safety_regressed and primary_task_improved),
        "paired_episodes": paired_rows,
        "production_admission": False,
    }


def run_paired_closed_loop_evaluation_v1(
    *,
    checkpoint_path: Path,
    output_json: Path,
    seed_base: int,
    episodes: int,
    device: str,
) -> dict[str, Any]:
    if type(seed_base) is not int or seed_base < 0:
        raise ValueError("paired evaluation seed base must be non-negative")
    if type(episodes) is not int or episodes < 1:
        raise ValueError("paired evaluation episode count must be positive")
    output = Path(output_json).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"paired evaluation output exists: {output}")
    resolved_device = _resolve_device(device)
    (
        candidate_state,
        parent_state,
        candidate_plan,
        parent_plan,
        identities,
    ) = _load_v14_and_parent_actor_states(checkpoint_path)
    environment, policy, action, scene = _configs_from_parent_plan(parent_plan)

    parent_evaluation = _evaluate_actor_state(
        parent_state,
        environment_config=environment,
        policy_config=policy,
        action_config=action,
        scene_path=scene,
        seed_base=seed_base,
        episodes=episodes,
        device=resolved_device,
    )
    candidate_evaluation = _evaluate_actor_state(
        candidate_state,
        environment_config=environment,
        policy_config=policy,
        action_config=action,
        scene_path=scene,
        seed_base=seed_base,
        episodes=episodes,
        device=resolved_device,
    )
    comparison = build_paired_comparison_v1(
        candidate_evaluation,
        parent_evaluation,
    )
    payload = {
        "format": V14_EVALUATION_FORMAT_V1,
        "created_at_utc": _utc_now(),
        "algorithm_format": ASYMMETRIC_DRQ_SAC_FORMAT_V1,
        "source_type": SOURCE_TYPE,
        "evaluation_kernel_format": EVALUATION_FORMAT,
        "evaluation_kernel_policy_format": POLICY_FORMAT,
        "same_seed_fresh_environment_comparison": True,
        "deterministic_policy": True,
        "seed_base": seed_base,
        "episode_count": episodes,
        "resolved_device": resolved_device,
        "identities": identities,
        "candidate_offline_run_format": candidate_plan["format"],
        "candidate": candidate_evaluation,
        "parent": parent_evaluation,
        "comparison": comparison,
        "expert_calls": 0,
        "behavior_cloning_steps": 0,
        "physical_samples": 0,
        "production_admission": False,
        "remaining_gates": [
            "multi-seed V14 online data collection and replay refresh",
            "strict-success held-out improvement at meaningful scale",
            "online depth, segmentation, and causal 4D reconstruction",
            "physical AQ16 and UNO Q calibration plus validation",
        ],
    }
    payload["payload_sha256"] = canonical_sha256_v1(payload)
    _atomic_json(output, payload)
    return payload


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--seed-base", type=int, default=93_000_000)
    parser.add_argument("--episodes", type=int, default=4)
    parser.add_argument("--device", choices=("auto", "cpu", "mps"), default="auto")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = run_paired_closed_loop_evaluation_v1(
        checkpoint_path=args.checkpoint,
        output_json=args.output_json,
        seed_base=args.seed_base,
        episodes=args.episodes,
        device=args.device,
    )
    compact = {
        "format": result["format"],
        "output_json": str(args.output_json.expanduser().resolve()),
        "comparison": result["comparison"],
        "production_admission": False,
    }
    print(json.dumps(compact, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
