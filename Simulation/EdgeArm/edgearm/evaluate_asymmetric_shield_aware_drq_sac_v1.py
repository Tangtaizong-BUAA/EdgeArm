"""Same-seed closed-loop evaluation of V15 against its direct V14 parent."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import torch

from .asymmetric_drq_sac_v1 import ASYMMETRIC_DRQ_SAC_FORMAT_V1
from .asymmetric_multiview_ppo_v1 import (
    EVALUATION_FORMAT,
    POLICY_FORMAT,
    SOURCE_TYPE,
    SelectedViewRecurrentActorV1,
    canonical_sha256_v1,
    sha256_file_v1,
)
from .asymmetric_shield_aware_drq_sac_v1 import (
    ASYMMETRIC_SHIELD_AWARE_DRQ_SAC_CHECKPOINT_FORMAT_V1,
    ASYMMETRIC_SHIELD_AWARE_DRQ_SAC_FORMAT_V1,
)
from .evaluate_asymmetric_drq_sac_v1 import (
    _atomic_json,
    _configs_from_parent_plan,
    _evaluate_actor_state,
    _resolve_device,
    _utc_now,
    build_paired_comparison_v1,
)
from .ppo_utils_v1 import state_dict_sha256_v1
from .train_asymmetric_drq_sac_v1 import _load_v13_parent_actor
from .train_asymmetric_shield_aware_drq_sac_v1 import (
    SHIELD_AWARE_OFFLINE_RUN_FORMAT_V1,
    _load_json_dictionary,
    _load_v14_parent,
    _validate_run_plan,
)


V15_EVALUATION_FORMAT_V1 = (
    "edgearm-v15-shield-aware-v14-parent-heldout-evaluation-v1"
)


def _load_v15_and_v14_actor_states(
    checkpoint_path: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    path = Path(checkpoint_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"V15 checkpoint is missing: {path}")
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if type(checkpoint) is not dict:
        raise TypeError("V15 checkpoint must be a dictionary")
    required = {
        "format": ASYMMETRIC_SHIELD_AWARE_DRQ_SAC_CHECKPOINT_FORMAT_V1,
        "algorithm_format": ASYMMETRIC_SHIELD_AWARE_DRQ_SAC_FORMAT_V1,
        "source_type": SOURCE_TYPE,
        "expert_calls": 0,
        "behavior_cloning_steps": 0,
        "physical_samples": 0,
        "production_admission": False,
    }
    for key, expected in required.items():
        if checkpoint.get(key) != expected:
            raise ValueError(f"V15 checkpoint mismatch for {key}")
    plan_path = path.parent.parent / "run_plan.json"
    if not plan_path.is_file():
        raise FileNotFoundError("V15 run plan is missing")
    plan = _load_json_dictionary(plan_path, label="V15")
    plan_hash = _validate_run_plan(plan, label="V15")
    if (
        plan.get("format") != SHIELD_AWARE_OFFLINE_RUN_FORMAT_V1
        or plan.get("algorithm_format")
        != ASYMMETRIC_SHIELD_AWARE_DRQ_SAC_FORMAT_V1
        or checkpoint.get("run_plan_sha256") != plan_hash
    ):
        raise ValueError("V15 checkpoint and run plan disagree")
    candidate_state = checkpoint.get("actor_state_dict")
    if not isinstance(candidate_state, dict):
        raise TypeError("V15 actor state is missing")
    candidate_probe = SelectedViewRecurrentActorV1(task_count=1)
    candidate_probe.load_state_dict(candidate_state, strict=True)
    candidate_hash = state_dict_sha256_v1(candidate_probe.state_dict())

    parent_lineage = checkpoint.get("parent_lineage")
    if type(parent_lineage) is not dict:
        raise TypeError("V15 parent lineage is missing")
    parent_path_value = parent_lineage.get("parent_checkpoint_path")
    if not isinstance(parent_path_value, str):
        raise TypeError("V15 parent checkpoint path is missing")
    parent_path = Path(parent_path_value).expanduser().resolve()
    parent_states, parent_checkpoint, _parent_plan, verified = (
        _load_v14_parent(parent_path)
    )
    for key in (
        "parent_checkpoint_sha256",
        "parent_actor_state_sha256",
        "parent_critic_state_sha256",
        "parent_run_plan_sha256",
    ):
        if parent_lineage.get(key) != verified.get(key):
            raise ValueError(f"V15-to-V14 lineage mismatch for {key}")

    root_lineage = parent_checkpoint.get("parent_lineage")
    if type(root_lineage) is not dict:
        raise TypeError("V14-to-V13 root lineage is missing")
    root_path_value = root_lineage.get("parent_checkpoint_path")
    if not isinstance(root_path_value, str):
        raise TypeError("V13 root checkpoint path is missing")
    _root_state, verified_root = _load_v13_parent_actor(
        Path(root_path_value)
    )
    root_plan_path = Path(verified_root["parent_run_plan_path"])
    root_plan = _load_json_dictionary(root_plan_path, label="V13 root")
    identities = {
        "candidate_checkpoint_path": str(path),
        "candidate_checkpoint_sha256": sha256_file_v1(path),
        "candidate_update_index": int(checkpoint["update_index"]),
        "candidate_actor_state_sha256": candidate_hash,
        "candidate_run_plan_path": str(plan_path.resolve()),
        "candidate_run_plan_sha256": plan_hash,
        "direct_parent_checkpoint_path": str(parent_path),
        "direct_parent_checkpoint_sha256": verified[
            "parent_checkpoint_sha256"
        ],
        "direct_parent_update_index": verified["parent_update_index"],
        "direct_parent_actor_state_sha256": verified[
            "parent_actor_state_sha256"
        ],
        "root_v13_checkpoint_path": verified_root[
            "parent_checkpoint_path"
        ],
        "root_v13_checkpoint_sha256": verified_root[
            "parent_checkpoint_sha256"
        ],
        "root_v13_run_plan_path": str(root_plan_path.resolve()),
        "root_v13_run_plan_sha256": verified_root[
            "parent_run_plan_sha256"
        ],
    }
    return candidate_state, parent_states["actor"], root_plan, identities


def run_v15_parent_paired_evaluation_v1(
    *,
    checkpoint_path: Path,
    output_json: Path,
    seed_base: int,
    episodes: int,
    device: str,
) -> dict[str, Any]:
    if type(seed_base) is not int or seed_base < 0:
        raise ValueError("V15 evaluation seed base must be non-negative")
    if type(episodes) is not int or episodes < 1:
        raise ValueError("V15 evaluation episodes must be positive")
    output = Path(output_json).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"V15 evaluation output exists: {output}")
    resolved_device = _resolve_device(device)
    candidate_state, parent_state, root_plan, identities = (
        _load_v15_and_v14_actor_states(checkpoint_path)
    )
    environment, policy, action, scene = _configs_from_parent_plan(root_plan)
    parent = _evaluate_actor_state(
        parent_state,
        environment_config=environment,
        policy_config=policy,
        action_config=action,
        scene_path=scene,
        seed_base=seed_base,
        episodes=episodes,
        device=resolved_device,
    )
    candidate = _evaluate_actor_state(
        candidate_state,
        environment_config=environment,
        policy_config=policy,
        action_config=action,
        scene_path=scene,
        seed_base=seed_base,
        episodes=episodes,
        device=resolved_device,
    )
    comparison = build_paired_comparison_v1(candidate, parent)
    payload = {
        "format": V15_EVALUATION_FORMAT_V1,
        "created_at_utc": _utc_now(),
        "algorithm_format": ASYMMETRIC_SHIELD_AWARE_DRQ_SAC_FORMAT_V1,
        "direct_parent_algorithm_format": ASYMMETRIC_DRQ_SAC_FORMAT_V1,
        "source_type": SOURCE_TYPE,
        "evaluation_kernel_format": EVALUATION_FORMAT,
        "evaluation_kernel_policy_format": POLICY_FORMAT,
        "same_seed_fresh_environment_comparison": True,
        "deterministic_policy": True,
        "seed_base": seed_base,
        "episode_count": episodes,
        "resolved_device": resolved_device,
        "identities": identities,
        "candidate": candidate,
        "direct_v14_parent": parent,
        "comparison": comparison,
        "expert_calls": 0,
        "behavior_cloning_steps": 0,
        "physical_samples": 0,
        "production_admission": False,
        "remaining_gates": [
            "valid contact and block-motion improvement",
            "fresh online RL trajectory collection",
            "strict-success held-out evaluation",
            "depth, segmentation, causal 4D, and physical calibration",
        ],
    }
    payload["payload_sha256"] = canonical_sha256_v1(payload)
    _atomic_json(output, payload)
    return payload


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--seed-base", type=int, default=94_000_000)
    parser.add_argument("--episodes", type=int, default=4)
    parser.add_argument("--device", choices=("auto", "cpu", "mps"), default="auto")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = run_v15_parent_paired_evaluation_v1(
        checkpoint_path=args.checkpoint,
        output_json=args.output_json,
        seed_base=args.seed_base,
        episodes=args.episodes,
        device=args.device,
    )
    print(
        json.dumps(
            {
                "format": result["format"],
                "output_json": str(args.output_json.expanduser().resolve()),
                "comparison": result["comparison"],
                "production_admission": False,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
