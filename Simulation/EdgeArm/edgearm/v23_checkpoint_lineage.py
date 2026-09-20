"""Verify accepted V23 checkpoints and expose them as collection parents."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from .asymmetric_multiview_ppo_v1 import (
    SOURCE_TYPE,
    canonical_sha256_v1,
    sha256_file_v1,
)
from .goal_directed_feasible_on_policy_ppo_v23 import (
    FRONTIER_BALANCED_GOAL_DIRECTED_PPO_FORMAT_V24,
    GOAL_DIRECTED_FEASIBLE_ON_POLICY_PPO_FORMAT_V23,
)
from .meaningful_effect_backtracking_v26 import (
    MEANINGFUL_EFFECT_BACKTRACKING_FORMAT_V26,
)
from .ppo_utils_v1 import state_dict_sha256_v1
from .trust_region_backtracking_v25 import TRUST_REGION_BACKTRACKING_PPO_FORMAT_V25
from .train_on_policy_recurrent_ppo_v20 import (
    _load_v15_actor_and_root_critic,
    _mean_network_sha256,
)
from .v26_reverse_curriculum_genesis import (
    V26_REVERSE_CURRICULUM_GENESIS_CHECKPOINT_FORMAT,
    load_reverse_curriculum_genesis_v26,
)


FEASIBLE_MULTIVIEW_PPO_RUN_FORMAT_V23 = "edgearm-v23-goal-directed-multiview-ppo-update-run-v1"
FEASIBLE_MULTIVIEW_PPO_CHECKPOINT_FORMAT_V23 = "edgearm-v23-goal-directed-multiview-ppo-checkpoint-v1"
ACCEPTED_V23_PARENT_LINEAGE_FORMAT = "edgearm-v23-accepted-checkpoint-parent-lineage-v1"
V28_SPLIT_CRITIC_SMOKE_CHECKPOINT_FORMAT = (
    "edgearm-v28-split-component-critic-only-smoke-checkpoint-v1"
)


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"V23 expected a JSON object: {path}")
    return value


def _load_accepted_v23_checkpoint(
    checkpoint_path: Path,
    checkpoint: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    if (
        checkpoint.get("format") != FEASIBLE_MULTIVIEW_PPO_CHECKPOINT_FORMAT_V23
        or checkpoint.get("algorithm_format")
        not in {
            GOAL_DIRECTED_FEASIBLE_ON_POLICY_PPO_FORMAT_V23,
            FRONTIER_BALANCED_GOAL_DIRECTED_PPO_FORMAT_V24,
            TRUST_REGION_BACKTRACKING_PPO_FORMAT_V25,
            MEANINGFUL_EFFECT_BACKTRACKING_FORMAT_V26,
        }
        or checkpoint.get("source_type") != SOURCE_TYPE
        or checkpoint.get("closed_loop_update_accepted") is not True
        or checkpoint.get("production_admission") is not False
        or checkpoint.get("expert_calls") != 0
        or checkpoint.get("behavior_cloning_steps") != 0
        or checkpoint.get("physical_samples") != 0
    ):
        raise ValueError("V23 checkpoint is not an accepted scratch-RL parent")
    actor_state = checkpoint.get("actor_state_dict")
    critic_state = checkpoint.get("critic_state_dict")
    metrics = checkpoint.get("metrics")
    stored_lineage = checkpoint.get("lineage")
    if not all(isinstance(value, dict) for value in (actor_state, critic_state, metrics, stored_lineage)):
        raise TypeError("V23 accepted checkpoint payload is incomplete")
    assert isinstance(actor_state, dict)
    assert isinstance(critic_state, dict)
    assert isinstance(metrics, dict)
    assert isinstance(stored_lineage, dict)
    actor_sha256 = state_dict_sha256_v1(actor_state)
    critic_sha256 = state_dict_sha256_v1(critic_state)
    gate_checks = metrics.get("closed_loop_gate_checks")
    if (
        actor_sha256 != checkpoint.get("actor_state_sha256")
        or critic_sha256 != checkpoint.get("critic_state_sha256")
        or actor_sha256 != metrics.get("committed_actor_state_sha256")
        or critic_sha256 != metrics.get("committed_critic_state_sha256")
        or metrics.get("update_accepted") is not True
        or int(metrics.get("committed_optimizer_steps", 0)) < 1
        or not isinstance(gate_checks, dict)
        or not gate_checks
        or not all(gate_checks.values())
        or checkpoint.get("optimizer_state_dict") is None
    ):
        raise ValueError("V23 accepted checkpoint weights or gate evidence changed")

    run_plan_path = checkpoint_path.parents[1] / "run_plan.json"
    run_plan = _load_json(run_plan_path)
    run_plan_sha256 = checkpoint.get("run_plan_sha256")
    unhashed_plan = dict(run_plan)
    unhashed_plan.pop("run_plan_sha256", None)
    if (
        run_plan.get("format") != FEASIBLE_MULTIVIEW_PPO_RUN_FORMAT_V23
        or run_plan.get("run_plan_sha256") != run_plan_sha256
        or canonical_sha256_v1(unhashed_plan) != run_plan_sha256
        or run_plan.get("algorithm_format") != checkpoint.get("algorithm_format")
        or run_plan.get("rollout_consumption_limit") != 1
        or run_plan.get("rollout_consumption_index") != 1
        or run_plan.get("production_admission") is not False
    ):
        raise ValueError("V23 accepted checkpoint run plan failed verification")
    source_rollout = Path(str(run_plan["rollout_path"])).expanduser().resolve()
    source_rollout_sha256 = sha256_file_v1(source_rollout)
    if (
        source_rollout_sha256 != run_plan.get("rollout_sha256")
        or source_rollout_sha256 != checkpoint.get("source_rollout_sha256")
        or source_rollout_sha256 != metrics.get("rollout_sha256")
    ):
        raise ValueError("V23 accepted checkpoint source rollout changed")
    parent_checkpoint = Path(str(run_plan["parent_checkpoint_path"])).expanduser().resolve()
    if sha256_file_v1(parent_checkpoint) != run_plan.get("parent_checkpoint_sha256"):
        raise ValueError("V23 accepted checkpoint parent bytes changed")
    _parent_actor, _parent_critic, root_plan, parent_lineage = load_collection_parent_actor_critic_v23(
        parent_checkpoint
    )
    if canonical_sha256_v1(parent_lineage) != canonical_sha256_v1(stored_lineage):
        raise ValueError("V23 accepted checkpoint parent lineage changed")
    accepted_updates = int(parent_lineage.get("accepted_goal_directed_updates", 0)) + 1
    lineage = {
        "format": ACCEPTED_V23_PARENT_LINEAGE_FORMAT,
        "parent_checkpoint_path": str(checkpoint_path),
        "parent_checkpoint_sha256": sha256_file_v1(checkpoint_path),
        "parent_run_plan_path": str(run_plan_path),
        "parent_run_plan_sha256": run_plan_sha256,
        "parent_source_rollout_path": str(source_rollout),
        "parent_source_rollout_sha256": source_rollout_sha256,
        "parent_actor_state_sha256": actor_sha256,
        "parent_actor_mean_state_sha256": _mean_network_sha256(actor_state),
        "parent_critic_state_sha256": critic_sha256,
        "accepted_goal_directed_updates": accepted_updates,
        "previous_parent_lineage": parent_lineage,
        "offline_replay_actor_updates": 0,
        "expert_calls": 0,
        "behavior_cloning_steps": 0,
        "physical_samples": 0,
        "production_admission": False,
    }
    return actor_state, critic_state, root_plan, lineage


def _load_v28_critic_only_collection_parent(
    checkpoint_path: Path,
    checkpoint: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Expose the unchanged actor and legacy scalar anchor after V28 critic smoke."""

    metrics = checkpoint.get("metrics")
    actor_state = checkpoint.get("actor_state_dict")
    component_state = checkpoint.get("component_critic_state_dict")
    if not all(isinstance(value, dict) for value in (metrics, actor_state, component_state)):
        raise TypeError("V28 critic-only collection parent is incomplete")
    assert isinstance(metrics, dict)
    assert isinstance(actor_state, dict)
    assert isinstance(component_state, dict)
    if (
        checkpoint.get("format") != V28_SPLIT_CRITIC_SMOKE_CHECKPOINT_FORMAT
        or checkpoint.get("source_type") != SOURCE_TYPE
        or checkpoint.get("critic_only_smoke_passed") is not True
        or checkpoint.get("actor_update_committed") is not False
        or checkpoint.get("closed_loop_update_accepted") is not False
        or checkpoint.get("optimizer_state_resume_supported") is not False
        or checkpoint.get("fresh_on_policy_rollout_required") is not True
        or checkpoint.get("expert_calls") != 0
        or checkpoint.get("behavior_cloning_steps") != 0
        or checkpoint.get("physical_samples") != 0
        or checkpoint.get("production_admission") is not False
        or metrics.get("actor_unchanged") is not True
        or metrics.get("immutable_modules_unchanged") is not True
        or metrics.get("copied_component_rows_unchanged") is not True
        or metrics.get("split_heads_changed") is not True
        or metrics.get("smoke_passed") is not True
    ):
        raise ValueError("V28 critic-only checkpoint is not a collection parent")
    if state_dict_sha256_v1(actor_state) != checkpoint.get("actor_state_sha256"):
        raise ValueError("V28 critic-only parent actor changed")
    if state_dict_sha256_v1(component_state) != checkpoint.get(
        "component_critic_state_sha256"
    ):
        raise ValueError("V28 critic-only parent component critic changed")
    required_component_keys = {
        "trunk.0.weight",
        "trunk.0.bias",
        "trunk.2.weight",
        "trunk.2.bias",
        "legacy_total_anchor.weight",
        "legacy_total_anchor.bias",
    }
    if not required_component_keys.issubset(component_state):
        raise ValueError("V28 critic-only parent lost its scalar anchor state")
    scalar_critic_state = {
        "network.0.weight": component_state["trunk.0.weight"],
        "network.0.bias": component_state["trunk.0.bias"],
        "network.2.weight": component_state["trunk.2.weight"],
        "network.2.bias": component_state["trunk.2.bias"],
        "network.4.weight": component_state["legacy_total_anchor.weight"],
        "network.4.bias": component_state["legacy_total_anchor.bias"],
    }
    run_plan_path = checkpoint_path.parents[1] / "run_plan.json"
    run_plan = _load_json(run_plan_path)
    unhashed_plan = dict(run_plan)
    run_plan_sha256 = unhashed_plan.pop("run_plan_sha256", None)
    if (
        run_plan_sha256 != checkpoint.get("run_plan_sha256")
        or canonical_sha256_v1(unhashed_plan) != run_plan_sha256
        or run_plan.get("fresh_on_policy_rollout_required_after_smoke") is not True
        or run_plan.get("production_admission") is not False
    ):
        raise ValueError("V28 critic-only run plan failed verification")
    migration_path = Path(
        str(checkpoint["parent_migration_checkpoint_path"])
    ).expanduser().resolve()
    if sha256_file_v1(migration_path) != checkpoint.get(
        "parent_migration_checkpoint_sha256"
    ):
        raise ValueError("V28 critic-only migration parent bytes changed")
    migration = torch.load(migration_path, map_location="cpu", weights_only=True)
    if not isinstance(migration, dict) or not isinstance(migration.get("root_plan"), dict):
        raise TypeError("V28 critic-only parent root plan is missing")
    parent_lineage = migration.get("parent_lineage")
    if not isinstance(parent_lineage, dict):
        raise TypeError("V28 critic-only parent lineage is missing")
    rollout_path = Path(str(run_plan["rollout_path"])).expanduser().resolve()
    if sha256_file_v1(rollout_path) != run_plan.get("rollout_sha256"):
        raise ValueError("V28 critic-only historical smoke rollout changed")
    actor_sha256 = state_dict_sha256_v1(actor_state)
    critic_sha256 = state_dict_sha256_v1(scalar_critic_state)
    lineage = {
        "format": ACCEPTED_V23_PARENT_LINEAGE_FORMAT,
        "parent_checkpoint_path": str(checkpoint_path),
        "parent_checkpoint_sha256": sha256_file_v1(checkpoint_path),
        "parent_run_plan_path": str(run_plan_path),
        "parent_run_plan_sha256": run_plan_sha256,
        "parent_source_rollout_path": str(rollout_path),
        "parent_source_rollout_sha256": run_plan["rollout_sha256"],
        "parent_source_rollout_role": (
            "historical_critic_only_target_smoke_not_actor_update"
        ),
        "parent_actor_state_sha256": actor_sha256,
        "parent_actor_mean_state_sha256": _mean_network_sha256(actor_state),
        "parent_critic_state_sha256": critic_sha256,
        "accepted_goal_directed_updates": int(
            parent_lineage.get("accepted_goal_directed_updates", 0)
        ),
        "v28_split_critic_only_migrations": int(
            parent_lineage.get("v28_split_critic_only_migrations", 0)
        )
        + 1,
        "previous_parent_lineage": parent_lineage,
        "offline_replay_actor_updates": 0,
        "expert_calls": 0,
        "behavior_cloning_steps": 0,
        "physical_samples": 0,
        "production_admission": False,
    }
    return actor_state, scalar_critic_state, migration["root_plan"], lineage


def load_collection_parent_actor_critic_v23(
    parent_checkpoint: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Load a verified scratch genesis or an accepted on-policy child."""

    path = Path(parent_checkpoint).expanduser().resolve()
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict):
        raise TypeError("collection parent checkpoint must be a dictionary")
    if checkpoint.get("format") == V26_REVERSE_CURRICULUM_GENESIS_CHECKPOINT_FORMAT:
        return load_reverse_curriculum_genesis_v26(path)
    if checkpoint.get("format") == FEASIBLE_MULTIVIEW_PPO_CHECKPOINT_FORMAT_V23:
        return _load_accepted_v23_checkpoint(path, checkpoint)
    if checkpoint.get("format") == V28_SPLIT_CRITIC_SMOKE_CHECKPOINT_FORMAT:
        return _load_v28_critic_only_collection_parent(path, checkpoint)
    return _load_v15_actor_and_root_critic(path)


__all__ = [
    "ACCEPTED_V23_PARENT_LINEAGE_FORMAT",
    "FEASIBLE_MULTIVIEW_PPO_CHECKPOINT_FORMAT_V23",
    "FEASIBLE_MULTIVIEW_PPO_RUN_FORMAT_V23",
    "V28_SPLIT_CRITIC_SMOKE_CHECKPOINT_FORMAT",
    "load_collection_parent_actor_critic_v23",
]
