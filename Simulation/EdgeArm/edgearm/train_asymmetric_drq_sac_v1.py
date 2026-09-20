"""Offline V14 DrQ-SAC bootstrap from source-closed V13 scratch rollouts.

This executable is an intermediate training gate, not production admission.
It can continue only an actor from this project's own random-initialized V13
``sim_rl_scratch`` lineage, initializes new twin Q critics from random weights,
and trains solely from exact V13 H5 artifacts.  It does not call an expert,
perform behavior cloning, or consume physical samples.
"""

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
    ASYMMETRIC_DRQ_SAC_CRITIC_ARCHITECTURE_V1,
    ASYMMETRIC_DRQ_SAC_FORMAT_V1,
    AsymmetricDrQSACConfigV1,
    asymmetric_drq_sac_update_v1,
    initialize_asymmetric_drq_sac_v1,
)
from .asymmetric_multiview_ppo_v1 import (
    CHECKPOINT_FORMAT,
    POLICY_FORMAT,
    SOURCE_TYPE,
    SelectedViewRecurrentActorV1,
    canonical_sha256_v1,
    sha256_file_v1,
)
from .contact_prioritized_replay_v1 import (
    ContactPrioritizedReplayConfigV1,
    ContactPrioritizedReplayStoreV1,
)
from .ppo_utils_v1 import state_dict_sha256_v1
from .privileged_effect_state_v1 import PRIVILEGED_EFFECT_STATE_DIM


OFFLINE_RUN_FORMAT_V1 = "edgearm-v14-asymmetric-drq-sac-offline-run-v1"


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
    line = json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n"
    with path.open("a", encoding="utf-8") as stream:
        stream.write(line)
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


def _parent_run_plan_path(checkpoint_path: Path) -> Path:
    direct = checkpoint_path.parent / "run_plan.json"
    nested = checkpoint_path.parent.parent / "run_plan.json"
    if direct.is_file():
        return direct
    if nested.is_file():
        return nested
    raise FileNotFoundError("V13 parent run_plan.json is missing")


def _load_v13_parent_actor(
    checkpoint_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    path = Path(checkpoint_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"V13 parent checkpoint is missing: {path}")
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if type(checkpoint) is not dict:
        raise TypeError("V13 parent checkpoint must be a dictionary")
    required = {
        "format": CHECKPOINT_FORMAT,
        "source_type": SOURCE_TYPE,
        "production_admission": False,
        "physical_samples": 0,
        "expert_warm_start": False,
        "external_pretraining": False,
    }
    for key, expected in required.items():
        if checkpoint.get(key) != expected:
            raise ValueError(
                f"V13 parent checkpoint mismatch for {key}: "
                f"{checkpoint.get(key)!r} != {expected!r}"
            )
    provenance = checkpoint.get("provenance")
    if type(provenance) is not dict:
        raise TypeError("V13 parent provenance is missing")
    provenance_required = {
        "source_type": SOURCE_TYPE,
        "policy_format": POLICY_FORMAT,
        "random_initialization": True,
        "expert_calls": 0,
        "warm_start": False,
        "behavior_cloning_steps": 0,
        "actor_privileged_state_inputs": 0,
    }
    for key, expected in provenance_required.items():
        if provenance.get(key) != expected:
            raise ValueError(f"V13 parent provenance mismatch for {key}")
    actor_state = checkpoint.get("actor_state_dict")
    if not isinstance(actor_state, dict):
        raise TypeError("V13 parent actor state is missing")
    probe = SelectedViewRecurrentActorV1(task_count=1)
    probe.load_state_dict(actor_state, strict=True)
    actor_hash = state_dict_sha256_v1(probe.state_dict())

    plan_path = _parent_run_plan_path(path)
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if type(plan) is not dict:
        raise TypeError("V13 parent run plan must be a dictionary")
    stored_hash = plan.get("run_plan_sha256")
    plan_payload = dict(plan)
    plan_payload.pop("run_plan_sha256", None)
    if (
        not isinstance(stored_hash, str)
        or stored_hash != canonical_sha256_v1(plan_payload)
        or checkpoint.get("run_plan_sha256") != stored_hash
    ):
        raise ValueError("V13 parent run-plan hash chain is invalid")
    if plan.get("policy_format") != POLICY_FORMAT:
        raise ValueError("V13 parent run-plan policy format changed")
    lineage = {
        "format": "edgearm-v13-to-v14-own-scratch-actor-lineage-v1",
        "parent_checkpoint_path": str(path),
        "parent_checkpoint_sha256": sha256_file_v1(path),
        "parent_checkpoint_format": CHECKPOINT_FORMAT,
        "parent_policy_format": POLICY_FORMAT,
        "parent_update_index": int(checkpoint["update_index"]),
        "parent_actor_state_sha256": actor_hash,
        "parent_run_plan_path": str(plan_path.resolve()),
        "parent_run_plan_sha256": stored_hash,
        "parent_random_initialization": True,
        "expert_calls": 0,
        "behavior_cloning_steps": 0,
        "physical_samples": 0,
        "continuation_algorithm": ASYMMETRIC_DRQ_SAC_FORMAT_V1,
        "production_admission": False,
    }
    return actor_state, lineage


def _parameter_counts(bundle: Any) -> dict[str, int]:
    actor = sum(parameter.numel() for parameter in bundle.actor.parameters())
    critic = sum(parameter.numel() for parameter in bundle.critic.parameters())
    return {
        "actor": actor,
        "twin_q_critic": critic,
        "trainable_total": actor + critic,
        "target_twin_q_critic_nontrainable": sum(
            parameter.numel() for parameter in bundle.target_critic.parameters()
        ),
    }


def run_offline_asymmetric_drq_sac_v1(
    *,
    output_dir: Path,
    replay_h5_paths: Sequence[Path],
    parent_checkpoint: Path,
    gradient_updates: int,
    batch_size: int,
    initialization_seed: int,
    sampling_seed_base: int,
    device: str,
    replay_config: ContactPrioritizedReplayConfigV1 | None = None,
    sac_config: AsymmetricDrQSACConfigV1 | None = None,
    checkpoint_every_updates: int = 50,
) -> dict[str, Any]:
    if type(gradient_updates) is not int or gradient_updates < 1:
        raise ValueError("offline DrQ-SAC gradient_updates must be positive")
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("offline DrQ-SAC batch_size must be positive")
    if type(initialization_seed) is not int or initialization_seed < 0:
        raise ValueError("offline DrQ-SAC initialization_seed is invalid")
    if type(sampling_seed_base) is not int or sampling_seed_base < 0:
        raise ValueError("offline DrQ-SAC sampling_seed_base is invalid")
    if (
        type(checkpoint_every_updates) is not int
        or checkpoint_every_updates < 1
    ):
        raise ValueError("offline checkpoint interval must be positive")
    output = Path(output_dir).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"offline DrQ-SAC output already exists: {output}")
    selected_replay = replay_config or ContactPrioritizedReplayConfigV1()
    selected_sac = sac_config or AsymmetricDrQSACConfigV1()
    selected_replay.validate()
    selected_sac.validate()
    resolved_device = _resolve_device(device)

    parent_actor, parent_lineage = _load_v13_parent_actor(parent_checkpoint)
    store = ContactPrioritizedReplayStoreV1.from_h5(
        tuple(replay_h5_paths), selected_replay
    )
    bundle = initialize_asymmetric_drq_sac_v1(
        initialization_seed,
        device=resolved_device,
        config=selected_sac,
        parent_actor_state_dict=parent_actor,
    )
    if bundle.actor_initial_state_sha256 != parent_lineage[
        "parent_actor_state_sha256"
    ]:
        raise RuntimeError("V14 actor differs from its verified V13 parent")
    counts = _parameter_counts(bundle)
    source_path = Path(__file__).resolve()
    run_plan = {
        "format": OFFLINE_RUN_FORMAT_V1,
        "created_at_utc": _utc_now(),
        "algorithm_format": ASYMMETRIC_DRQ_SAC_FORMAT_V1,
        "checkpoint_format": ASYMMETRIC_DRQ_SAC_CHECKPOINT_FORMAT_V1,
        "critic_architecture": ASYMMETRIC_DRQ_SAC_CRITIC_ARCHITECTURE_V1,
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
        "parameter_counts": counts,
        "actor_inputs": (
            "causal four-view RGB, joint history, previously executed action "
            "history, masks, task id, previous AR latent"
        ),
        "actor_privileged_state_inputs": 0,
        "critic_privileged_state_inputs": PRIVILEGED_EFFECT_STATE_DIM,
        "expert_calls": 0,
        "behavior_cloning_steps": 0,
        "physical_samples": 0,
        "production_admission": False,
        "source_hashes": {
            "edgearm/train_asymmetric_drq_sac_v1.py": sha256_file_v1(
                source_path
            ),
            "edgearm/asymmetric_drq_sac_v1.py": sha256_file_v1(
                source_path.with_name("asymmetric_drq_sac_v1.py")
            ),
            "edgearm/contact_prioritized_replay_v1.py": sha256_file_v1(
                source_path.with_name("contact_prioritized_replay_v1.py")
            ),
            "edgearm/asymmetric_multiview_ppo_v1.py": sha256_file_v1(
                source_path.with_name("asymmetric_multiview_ppo_v1.py")
            ),
        },
        "remaining_gates": [
            "V14 actor closed-loop online collection through the V13 shield",
            "multi-seed contact and block-motion improvement",
            "strict-success held-out evaluation",
            "online depth/segmentation and causal 4D reconstruction",
            "physical AQ16/UNO Q calibration and validation",
        ],
    }
    run_plan["run_plan_sha256"] = canonical_sha256_v1(run_plan)
    output.mkdir(parents=True)
    (output / "checkpoints").mkdir()
    _atomic_json(output / "run_plan.json", run_plan)
    started_at = _utc_now()
    latest_metrics: dict[str, Any] | None = None
    for update_index in range(1, gradient_updates + 1):
        sampling_seed = sampling_seed_base + update_index - 1
        batch = store.sample(batch_size=batch_size, seed=sampling_seed)
        metrics = asymmetric_drq_sac_update_v1(
            bundle,
            batch,
            selected_sac,
            update_index=update_index,
            seed=sampling_seed ^ 0xD14A,
        )
        latest_metrics = {
            **asdict(metrics),
            "sampling_seed": sampling_seed,
            "sampled_stratum_counts": {
                str(name): int(count)
                for name, count in zip(
                    *np.unique(batch.sampled_stratum, return_counts=True)
                )
            },
            "source_shield_rejection_samples": int(
                np.count_nonzero(batch.source_shield_rejected_before_step)
            ),
            "terminal_within_horizon_samples": int(
                np.count_nonzero(batch.terminal_within_horizon)
            ),
            "expert_calls": 0,
            "behavior_cloning_steps": 0,
            "physical_samples": 0,
            "production_admission": False,
        }
        _append_jsonl(output / "metrics.jsonl", latest_metrics)
        checkpoint_due = (
            update_index % checkpoint_every_updates == 0
            or update_index == gradient_updates
        )
        checkpoint_path: Path | None = None
        if checkpoint_due:
            checkpoint_path = (
                output / "checkpoints" / f"update_{update_index:06d}.pt"
            )
            _atomic_torch_save(
                checkpoint_path,
                {
                    "format": ASYMMETRIC_DRQ_SAC_CHECKPOINT_FORMAT_V1,
                    "algorithm_format": ASYMMETRIC_DRQ_SAC_FORMAT_V1,
                    "source_type": SOURCE_TYPE,
                    "update_index": update_index,
                    "created_at_utc": _utc_now(),
                    "run_plan_sha256": run_plan["run_plan_sha256"],
                    "actor_state_dict": bundle.actor.state_dict(),
                    "critic_state_dict": bundle.critic.state_dict(),
                    "target_critic_state_dict": (
                        bundle.target_critic.state_dict()
                    ),
                    "actor_optimizer_state_dict": (
                        bundle.actor_optimizer.state_dict()
                    ),
                    "critic_optimizer_state_dict": (
                        bundle.critic_optimizer.state_dict()
                    ),
                    "sac_config": asdict(selected_sac),
                    "replay_config": asdict(selected_replay),
                    "replay_manifest": store.manifest(),
                    "parent_lineage": parent_lineage,
                    "metrics": latest_metrics,
                    "expert_calls": 0,
                    "behavior_cloning_steps": 0,
                    "physical_samples": 0,
                    "production_admission": False,
                },
            )
        _atomic_json(
            output / "run_state.json",
            {
                "format": OFFLINE_RUN_FORMAT_V1,
                "status": (
                    "complete" if update_index == gradient_updates else "running"
                ),
                "last_completed_update": update_index,
                "gradient_updates": gradient_updates,
                "last_checkpoint": (
                    str(checkpoint_path) if checkpoint_path is not None else None
                ),
                "updated_at_utc": _utc_now(),
                "production_admission": False,
            },
        )
    if latest_metrics is None:  # pragma: no cover - guarded above
        raise RuntimeError("offline DrQ-SAC produced no updates")
    final_checkpoint = output / "checkpoints" / f"update_{gradient_updates:06d}.pt"
    if not final_checkpoint.is_file():
        raise RuntimeError("offline DrQ-SAC final checkpoint is missing")
    summary = {
        "format": OFFLINE_RUN_FORMAT_V1,
        "status": "complete",
        "started_at_utc": started_at,
        "completed_at_utc": _utc_now(),
        "gradient_updates": gradient_updates,
        "batch_size": batch_size,
        "replay_transition_count": store.transition_count,
        "replay_stratum_counts": store.stratum_counts(),
        "parameter_counts": counts,
        "final_metrics": latest_metrics,
        "final_checkpoint": str(final_checkpoint),
        "final_checkpoint_sha256": sha256_file_v1(final_checkpoint),
        "parent_lineage": parent_lineage,
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
    parser.add_argument("--gradient-updates", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--initialization-seed", type=int, default=22_400_000)
    parser.add_argument("--sampling-seed-base", type=int, default=22_500_000)
    parser.add_argument("--checkpoint-every-updates", type=int, default=50)
    parser.add_argument("--device", choices=("auto", "cpu", "mps"), default="auto")
    parser.add_argument("--random-shift-padding", type=int, default=4)
    parser.add_argument("--entropy-temperature", type=float, default=0.05)
    parser.add_argument("--conservative-q-coefficient", type=float, default=0.10)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    sac_config = replace(
        AsymmetricDrQSACConfigV1(),
        random_shift_padding_pixels=args.random_shift_padding,
        entropy_temperature=args.entropy_temperature,
        conservative_q_coefficient=args.conservative_q_coefficient,
    )
    summary = run_offline_asymmetric_drq_sac_v1(
        output_dir=args.output_dir,
        replay_h5_paths=args.replay_h5,
        parent_checkpoint=args.parent_checkpoint,
        gradient_updates=args.gradient_updates,
        batch_size=args.batch_size,
        initialization_seed=args.initialization_seed,
        sampling_seed_base=args.sampling_seed_base,
        device=args.device,
        sac_config=sac_config,
        checkpoint_every_updates=args.checkpoint_every_updates,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
