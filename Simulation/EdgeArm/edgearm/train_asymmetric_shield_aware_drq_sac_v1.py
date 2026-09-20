"""Train V15 shield-aware DrQ-SAC from an exact V14 failed-task lineage."""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from .asymmetric_drq_sac_v1 import (
    ASYMMETRIC_DRQ_SAC_CHECKPOINT_FORMAT_V1,
    ASYMMETRIC_DRQ_SAC_FORMAT_V1,
    TwinPrivilegedActionCriticV1,
)
from .asymmetric_multiview_ppo_v1 import (
    SOURCE_TYPE,
    SelectedViewRecurrentActorV1,
    canonical_sha256_v1,
    sha256_file_v1,
)
from .asymmetric_shield_aware_drq_sac_v1 import (
    ACTION_FEASIBILITY_ARCHITECTURE_V1,
    ASYMMETRIC_SHIELD_AWARE_DRQ_SAC_CHECKPOINT_FORMAT_V1,
    ASYMMETRIC_SHIELD_AWARE_DRQ_SAC_FORMAT_V1,
    AsymmetricShieldAwareDrQSACConfigV1,
    asymmetric_shield_aware_drq_sac_update_v1,
    initialize_asymmetric_shield_aware_drq_sac_v1,
)
from .contact_prioritized_replay_v1 import ContactPrioritizedReplayConfigV1
from .evaluate_asymmetric_drq_sac_v1 import V14_EVALUATION_FORMAT_V1
from .ppo_utils_v1 import state_dict_sha256_v1
from .privileged_effect_state_v1 import PRIVILEGED_EFFECT_STATE_DIM
from .shield_aware_replay_v1 import ShieldAwareReplayStoreV1
from .train_asymmetric_drq_sac_v1 import OFFLINE_RUN_FORMAT_V1


SHIELD_AWARE_OFFLINE_RUN_FORMAT_V1 = (
    "edgearm-v15-asymmetric-shield-aware-drq-sac-offline-run-v1"
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
        raise ValueError(f"unsupported device: {requested}")
    return "mps" if torch.backends.mps.is_available() else "cpu"


def _load_json_dictionary(path: Path, *, label: str) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if type(payload) is not dict:
        raise TypeError(f"{label} must be a JSON dictionary")
    return payload


def _validate_run_plan(plan: dict[str, Any], *, label: str) -> str:
    stored = plan.get("run_plan_sha256")
    unhashed = dict(plan)
    unhashed.pop("run_plan_sha256", None)
    if not isinstance(stored, str) or stored != canonical_sha256_v1(unhashed):
        raise ValueError(f"{label} run-plan hash chain is invalid")
    return stored


def _load_v14_parent(
    checkpoint_path: Path,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
]:
    path = Path(checkpoint_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"V14 parent checkpoint is missing: {path}")
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if type(checkpoint) is not dict:
        raise TypeError("V14 parent checkpoint must be a dictionary")
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
            raise ValueError(f"V14 parent mismatch for {key}")
    plan_path = path.parent.parent / "run_plan.json"
    if not plan_path.is_file():
        raise FileNotFoundError("V14 parent run plan is missing")
    plan = _load_json_dictionary(plan_path, label="V14 parent")
    plan_hash = _validate_run_plan(plan, label="V14 parent")
    if (
        plan.get("format") != OFFLINE_RUN_FORMAT_V1
        or plan.get("algorithm_format") != ASYMMETRIC_DRQ_SAC_FORMAT_V1
        or checkpoint.get("run_plan_sha256") != plan_hash
    ):
        raise ValueError("V14 parent checkpoint and run plan disagree")

    actor_state = checkpoint.get("actor_state_dict")
    critic_state = checkpoint.get("critic_state_dict")
    target_state = checkpoint.get("target_critic_state_dict")
    if not all(
        isinstance(value, dict)
        for value in (actor_state, critic_state, target_state)
    ):
        raise TypeError("V14 parent model states are missing")
    actor = SelectedViewRecurrentActorV1(task_count=1)
    critic = TwinPrivilegedActionCriticV1()
    target = TwinPrivilegedActionCriticV1()
    actor.load_state_dict(actor_state, strict=True)
    critic.load_state_dict(critic_state, strict=True)
    target.load_state_dict(target_state, strict=True)
    lineage = {
        "format": "edgearm-v14-to-v15-shield-aware-lineage-v1",
        "parent_checkpoint_path": str(path),
        "parent_checkpoint_sha256": sha256_file_v1(path),
        "parent_checkpoint_format": ASYMMETRIC_DRQ_SAC_CHECKPOINT_FORMAT_V1,
        "parent_algorithm_format": ASYMMETRIC_DRQ_SAC_FORMAT_V1,
        "parent_update_index": int(checkpoint["update_index"]),
        "parent_actor_state_sha256": state_dict_sha256_v1(
            actor.state_dict()
        ),
        "parent_critic_state_sha256": state_dict_sha256_v1(
            critic.state_dict()
        ),
        "parent_target_critic_state_sha256": state_dict_sha256_v1(
            target.state_dict()
        ),
        "parent_run_plan_path": str(plan_path.resolve()),
        "parent_run_plan_sha256": plan_hash,
        "expert_calls": 0,
        "behavior_cloning_steps": 0,
        "physical_samples": 0,
        "production_admission": False,
    }
    states = {
        "actor": actor_state,
        "critic": critic_state,
        "target_critic": target_state,
    }
    return states, checkpoint, plan, lineage


def _load_parent_evaluation(
    path: Path,
    *,
    parent_lineage: dict[str, Any],
) -> dict[str, Any]:
    evaluation_path = Path(path).expanduser().resolve()
    if not evaluation_path.is_file():
        raise FileNotFoundError("V14 parent evaluation is missing")
    evaluation = _load_json_dictionary(
        evaluation_path,
        label="V14 parent evaluation",
    )
    if (
        evaluation.get("format") != V14_EVALUATION_FORMAT_V1
        or evaluation.get("production_admission") is not False
    ):
        raise ValueError("V14 parent evaluation identity changed")
    stored_hash = evaluation.get("payload_sha256")
    unhashed = dict(evaluation)
    unhashed.pop("payload_sha256", None)
    if not isinstance(stored_hash, str) or stored_hash != canonical_sha256_v1(
        unhashed
    ):
        raise ValueError("V14 parent evaluation payload hash is invalid")
    identities = evaluation.get("identities")
    if type(identities) is not dict or identities.get(
        "candidate_checkpoint_sha256"
    ) != parent_lineage["parent_checkpoint_sha256"]:
        raise ValueError("V14 evaluation belongs to another checkpoint")
    candidate = evaluation.get("candidate")
    comparison = evaluation.get("comparison")
    if type(candidate) is not dict or type(comparison) is not dict:
        raise TypeError("V14 parent evaluation metrics are missing")
    episodes = candidate.get("episodes")
    if not isinstance(episodes, list) or not episodes:
        raise ValueError("V14 parent evaluation contains no episodes")
    return {
        "format": "edgearm-v15-parent-failure-evidence-v1",
        "path": str(evaluation_path),
        "sha256": sha256_file_v1(evaluation_path),
        "payload_sha256": stored_hash,
        "episode_count": len(episodes),
        "strict_success_count": int(candidate["strict_success_count"]),
        "valid_contact_transition_count": int(
            sum(
                int(item["valid_push_side_contact_transition_count"])
                for item in episodes
            )
        ),
        "mean_ik_failure_steps": float(
            np.mean([int(item["ik_failure_step_count"]) for item in episodes])
        ),
        "mean_final_block_target_distance_m": float(
            candidate["mean_final_block_target_distance_m"]
        ),
        "primary_task_improved_over_v13": bool(
            comparison["primary_task_improved"]
        ),
        "safety_regressed": bool(comparison["safety_regressed"]),
        "diagnosis": (
            "zero contact and high online IK/guard hold rate after V14 offline "
            "training"
        ),
    }


def _parameter_counts(bundle: Any) -> dict[str, int]:
    actor = sum(parameter.numel() for parameter in bundle.actor.parameters())
    critic = sum(parameter.numel() for parameter in bundle.critic.parameters())
    feasibility = sum(
        parameter.numel() for parameter in bundle.feasibility.parameters()
    )
    return {
        "actor": actor,
        "twin_q_critic": critic,
        "action_feasibility_classifier": feasibility,
        "trainable_total": actor + critic + feasibility,
        "target_twin_q_critic_nontrainable": sum(
            parameter.numel() for parameter in bundle.target_critic.parameters()
        ),
    }


def run_offline_asymmetric_shield_aware_drq_sac_v1(
    *,
    output_dir: Path,
    replay_h5_paths: Sequence[Path],
    parent_checkpoint: Path,
    parent_evaluation_json: Path,
    gradient_updates: int,
    batch_size: int,
    initialization_seed: int,
    sampling_seed_base: int,
    device: str,
    replay_config: ContactPrioritizedReplayConfigV1 | None = None,
    sac_config: AsymmetricShieldAwareDrQSACConfigV1 | None = None,
    checkpoint_every_updates: int = 50,
) -> dict[str, Any]:
    for name, value in (
        ("gradient_updates", gradient_updates),
        ("batch_size", batch_size),
        ("checkpoint_every_updates", checkpoint_every_updates),
    ):
        if type(value) is not int or value < 1:
            raise ValueError(f"V15 {name} must be positive")
    for name, value in (
        ("initialization_seed", initialization_seed),
        ("sampling_seed_base", sampling_seed_base),
    ):
        if type(value) is not int or value < 0:
            raise ValueError(f"V15 {name} must be non-negative")
    output = Path(output_dir).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"V15 output already exists: {output}")
    selected_replay = replay_config or ContactPrioritizedReplayConfigV1()
    selected_sac = sac_config or AsymmetricShieldAwareDrQSACConfigV1()
    selected_replay.validate()
    selected_sac.validate()
    resolved_device = _resolve_device(device)

    parent_states, _parent_checkpoint, _parent_plan, parent_lineage = (
        _load_v14_parent(parent_checkpoint)
    )
    failure_evidence = _load_parent_evaluation(
        parent_evaluation_json,
        parent_lineage=parent_lineage,
    )
    store = ShieldAwareReplayStoreV1.from_h5(
        tuple(replay_h5_paths),
        selected_replay,
    )
    bundle = initialize_asymmetric_shield_aware_drq_sac_v1(
        initialization_seed,
        device=resolved_device,
        config=selected_sac,
        parent_actor_state_dict=parent_states["actor"],
        parent_critic_state_dict=parent_states["critic"],
        parent_target_critic_state_dict=parent_states["target_critic"],
    )
    if (
        bundle.actor_initial_state_sha256
        != parent_lineage["parent_actor_state_sha256"]
        or bundle.critic_initial_state_sha256
        != parent_lineage["parent_critic_state_sha256"]
    ):
        raise RuntimeError("V15 model does not match its verified V14 parent")
    counts = _parameter_counts(bundle)
    source = Path(__file__).resolve()
    run_plan = {
        "format": SHIELD_AWARE_OFFLINE_RUN_FORMAT_V1,
        "created_at_utc": _utc_now(),
        "algorithm_format": ASYMMETRIC_SHIELD_AWARE_DRQ_SAC_FORMAT_V1,
        "checkpoint_format": (
            ASYMMETRIC_SHIELD_AWARE_DRQ_SAC_CHECKPOINT_FORMAT_V1
        ),
        "source_type": SOURCE_TYPE,
        "offline_only": True,
        "gradient_updates": gradient_updates,
        "batch_size": batch_size,
        "initialization_seed": initialization_seed,
        "sampling_seed_base": sampling_seed_base,
        "resolved_device": resolved_device,
        "sac_config": asdict(selected_sac),
        "replay_config": asdict(selected_replay),
        "replay_manifest": store.manifest(),
        "parent_lineage": parent_lineage,
        "parent_failure_evidence": failure_evidence,
        "parameter_counts": counts,
        "action_feasibility_architecture": (
            ACTION_FEASIBILITY_ARCHITECTURE_V1
        ),
        "actor_inputs": (
            "causal four-view RGB, joint history, executed-action history, "
            "masks, task id, previous AR latent"
        ),
        "actor_privileged_state_inputs": 0,
        "critic_privileged_state_inputs": PRIVILEGED_EFFECT_STATE_DIM,
        "feasibility_privileged_state_inputs": (
            PRIVILEGED_EFFECT_STATE_DIM
        ),
        "feasibility_actor_input": False,
        "feasibility_deployment_component": False,
        "expert_calls": 0,
        "behavior_cloning_steps": 0,
        "physical_samples": 0,
        "production_admission": False,
        "source_hashes": {
            "edgearm/train_asymmetric_shield_aware_drq_sac_v1.py": (
                sha256_file_v1(source)
            ),
            "edgearm/asymmetric_shield_aware_drq_sac_v1.py": (
                sha256_file_v1(
                    source.with_name(
                        "asymmetric_shield_aware_drq_sac_v1.py"
                    )
                )
            ),
            "edgearm/shield_aware_replay_v1.py": sha256_file_v1(
                source.with_name("shield_aware_replay_v1.py")
            ),
            "edgearm/asymmetric_drq_sac_v1.py": sha256_file_v1(
                source.with_name("asymmetric_drq_sac_v1.py")
            ),
        },
        "remaining_gates": [
            "same-seed closed-loop reduction in IK/guard hold rate",
            "multi-seed increase in valid contact and block motion",
            "fresh online RL trajectory collection through V13 shield",
            "strict-success held-out evaluation",
            "depth, segmentation, causal 4D, and physical calibration",
        ],
    }
    run_plan["run_plan_sha256"] = canonical_sha256_v1(run_plan)
    output.mkdir(parents=True)
    (output / "checkpoints").mkdir()
    _atomic_json(output / "run_plan.json", run_plan)
    started_at = _utc_now()
    latest: dict[str, Any] | None = None
    for update_index in range(1, gradient_updates + 1):
        sampling_seed = sampling_seed_base + update_index - 1
        batch = store.sample(batch_size=batch_size, seed=sampling_seed)
        metrics = asymmetric_shield_aware_drq_sac_update_v1(
            bundle,
            batch,
            selected_sac,
            update_index=update_index,
            seed=sampling_seed ^ 0xD15A,
        )
        latest = {
            **asdict(metrics),
            "sampling_seed": sampling_seed,
            "sampled_stratum_counts": {
                str(name): int(count)
                for name, count in zip(
                    *np.unique(
                        batch.base.sampled_stratum,
                        return_counts=True,
                    )
                )
            },
            "sampled_action_feasible_count": int(
                np.count_nonzero(batch.source_action_feasible)
            ),
            "sampled_action_infeasible_count": int(
                np.count_nonzero(~batch.source_action_feasible)
            ),
            "expert_calls": 0,
            "behavior_cloning_steps": 0,
            "physical_samples": 0,
            "production_admission": False,
        }
        _append_jsonl(output / "metrics.jsonl", latest)
        due = (
            update_index % checkpoint_every_updates == 0
            or update_index == gradient_updates
        )
        checkpoint_path: Path | None = None
        if due:
            checkpoint_path = (
                output / "checkpoints" / f"update_{update_index:06d}.pt"
            )
            _atomic_torch_save(
                checkpoint_path,
                {
                    "format": (
                        ASYMMETRIC_SHIELD_AWARE_DRQ_SAC_CHECKPOINT_FORMAT_V1
                    ),
                    "algorithm_format": (
                        ASYMMETRIC_SHIELD_AWARE_DRQ_SAC_FORMAT_V1
                    ),
                    "source_type": SOURCE_TYPE,
                    "update_index": update_index,
                    "created_at_utc": _utc_now(),
                    "run_plan_sha256": run_plan["run_plan_sha256"],
                    "actor_state_dict": bundle.actor.state_dict(),
                    "critic_state_dict": bundle.critic.state_dict(),
                    "target_critic_state_dict": (
                        bundle.target_critic.state_dict()
                    ),
                    "feasibility_state_dict": (
                        bundle.feasibility.state_dict()
                    ),
                    "actor_optimizer_state_dict": (
                        bundle.actor_optimizer.state_dict()
                    ),
                    "critic_optimizer_state_dict": (
                        bundle.critic_optimizer.state_dict()
                    ),
                    "feasibility_optimizer_state_dict": (
                        bundle.feasibility_optimizer.state_dict()
                    ),
                    "sac_config": asdict(selected_sac),
                    "replay_config": asdict(selected_replay),
                    "replay_manifest": store.manifest(),
                    "parent_lineage": parent_lineage,
                    "parent_failure_evidence": failure_evidence,
                    "metrics": latest,
                    "expert_calls": 0,
                    "behavior_cloning_steps": 0,
                    "physical_samples": 0,
                    "production_admission": False,
                },
            )
        _atomic_json(
            output / "run_state.json",
            {
                "format": SHIELD_AWARE_OFFLINE_RUN_FORMAT_V1,
                "status": (
                    "complete"
                    if update_index == gradient_updates
                    else "running"
                ),
                "last_completed_update": update_index,
                "gradient_updates": gradient_updates,
                "last_checkpoint": (
                    str(checkpoint_path)
                    if checkpoint_path is not None
                    else None
                ),
                "updated_at_utc": _utc_now(),
                "production_admission": False,
            },
        )
    if latest is None:  # pragma: no cover
        raise RuntimeError("V15 training produced no update")
    final_checkpoint = (
        output / "checkpoints" / f"update_{gradient_updates:06d}.pt"
    )
    if not final_checkpoint.is_file():
        raise RuntimeError("V15 final checkpoint is missing")
    summary = {
        "format": SHIELD_AWARE_OFFLINE_RUN_FORMAT_V1,
        "status": "complete",
        "started_at_utc": started_at,
        "completed_at_utc": _utc_now(),
        "gradient_updates": gradient_updates,
        "batch_size": batch_size,
        "replay_transition_count": store.transition_count,
        "replay_manifest": store.manifest(),
        "parameter_counts": counts,
        "final_metrics": latest,
        "final_checkpoint": str(final_checkpoint),
        "final_checkpoint_sha256": sha256_file_v1(final_checkpoint),
        "parent_lineage": parent_lineage,
        "parent_failure_evidence": failure_evidence,
        "expert_calls": 0,
        "behavior_cloning_steps": 0,
        "physical_samples": 0,
        "offline_only": True,
        "closed_loop_evaluated": False,
        "production_admission": False,
        "remaining_gates": run_plan["remaining_gates"],
    }
    _atomic_json(output / "summary.json", summary)
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--replay-h5", type=Path, action="append", required=True)
    parser.add_argument("--parent-checkpoint", type=Path, required=True)
    parser.add_argument("--parent-evaluation-json", type=Path, required=True)
    parser.add_argument("--gradient-updates", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--initialization-seed", type=int, default=22_600_000)
    parser.add_argument("--sampling-seed-base", type=int, default=22_700_000)
    parser.add_argument("--checkpoint-every-updates", type=int, default=25)
    parser.add_argument("--device", choices=("auto", "cpu", "mps"), default="auto")
    parser.add_argument(
        "--actor-infeasibility-coefficient",
        type=float,
        default=0.75,
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    config = replace(
        AsymmetricShieldAwareDrQSACConfigV1(),
        actor_infeasibility_coefficient=(
            args.actor_infeasibility_coefficient
        ),
    )
    summary = run_offline_asymmetric_shield_aware_drq_sac_v1(
        output_dir=args.output_dir,
        replay_h5_paths=args.replay_h5,
        parent_checkpoint=args.parent_checkpoint,
        parent_evaluation_json=args.parent_evaluation_json,
        gradient_updates=args.gradient_updates,
        batch_size=args.batch_size,
        initialization_seed=args.initialization_seed,
        sampling_seed_base=args.sampling_seed_base,
        device=args.device,
        sac_config=config,
        checkpoint_every_updates=args.checkpoint_every_updates,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
