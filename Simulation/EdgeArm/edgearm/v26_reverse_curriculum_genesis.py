"""Create and verify a fresh zero-mean scratch genesis for V26 curriculum RL."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Sequence

import torch

from .asymmetric_multiview_ppo_v1 import (
    SOURCE_TYPE,
    AsymmetricMultiViewPPOConfigV1,
    StockGripperTaskSpaceActionConfigV12,
    canonical_sha256_v1,
    initialize_asymmetric_multiview_ppo_v1,
    parameter_counts_v1,
    sha256_file_v1,
)
from .ppo_utils_v1 import state_dict_sha256_v1
from .reverse_curriculum_v26 import strict_success_hold_steps_v26
from .sim2real_env_v10 import RealisticEnvV10Config
from .stock_gripper_action_guard_v3 import StockGripperActionGuardConfigV3


V26_REVERSE_CURRICULUM_GENESIS_RUN_FORMAT = (
    "edgearm-v26-reverse-curriculum-scratch-genesis-run-v1"
)
V26_REVERSE_CURRICULUM_GENESIS_CHECKPOINT_FORMAT = (
    "edgearm-v26-reverse-curriculum-scratch-genesis-checkpoint-v1"
)
V26_REVERSE_CURRICULUM_GENESIS_LINEAGE_FORMAT = (
    "edgearm-v26-reverse-curriculum-scratch-genesis-lineage-v1"
)
SCENE_PATH_V26 = (
    Path(__file__).resolve().parents[2]
    / "SO101"
    / "edgearm_multiview_rl_scene_v1.xml"
)
V26_REVERSE_CURRICULUM_EPISODE_MAX_STEPS = 360


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


def _atomic_torch(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    torch.save(payload, partial)
    partial.replace(path)


def _mean_state_sha256(state: dict[str, Any]) -> str:
    if "log_std" not in state:
        raise ValueError("V26 genesis actor state lost log_std")
    return state_dict_sha256_v1(
        {name: value for name, value in state.items() if name != "log_std"}
    )


def _root_configs_from_plan(
    plan: dict[str, Any],
) -> tuple[
    RealisticEnvV10Config,
    AsymmetricMultiViewPPOConfigV1,
    StockGripperTaskSpaceActionConfigV12,
    Path,
]:
    environment_payload = plan.get("environment_config")
    policy_payload = plan.get("ppo_config")
    action_contract = plan.get("action_contract")
    if not all(
        isinstance(value, dict)
        for value in (environment_payload, policy_payload, action_contract)
    ):
        raise TypeError("V26 genesis root configuration is incomplete")
    assert isinstance(environment_payload, dict)
    assert isinstance(policy_payload, dict)
    assert isinstance(action_contract, dict)
    action_payload_value = action_contract.get("adapter_config")
    if not isinstance(action_payload_value, dict):
        raise TypeError("V26 genesis action-adapter configuration is missing")
    action_payload = dict(action_payload_value)
    guard_payload = action_payload.pop("guard", None)
    if not isinstance(guard_payload, dict):
        raise TypeError("V26 genesis guard configuration is missing")
    environment = RealisticEnvV10Config(**environment_payload)
    policy = AsymmetricMultiViewPPOConfigV1(**policy_payload)
    action = StockGripperTaskSpaceActionConfigV12(
        **action_payload,
        guard=StockGripperActionGuardConfigV3(**guard_payload),
    )
    policy.validate()
    action.validate()
    scene = Path(str(plan.get("scene_path", ""))).expanduser().resolve()
    if not scene.is_file() or sha256_file_v1(scene) != plan.get("scene_sha256"):
        raise ValueError("V26 genesis scene identity changed")
    return environment, policy, action, scene


def create_reverse_curriculum_genesis_v26(
    *,
    output_directory: Path,
    initialization_seed: int,
) -> dict[str, Any]:
    if type(initialization_seed) is not int or initialization_seed < 0:
        raise ValueError("V26 genesis initialization seed must be non-negative")
    output = Path(output_directory).expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"V26 genesis output directory is non-empty: {output}")
    if not SCENE_PATH_V26.is_file():
        raise FileNotFoundError(f"V26 genesis scene is missing: {SCENE_PATH_V26}")
    output.mkdir(parents=True, exist_ok=True)
    environment = RealisticEnvV10Config(
        max_steps=V26_REVERSE_CURRICULUM_EPISODE_MAX_STEPS,
        strict_success_hold_steps=strict_success_hold_steps_v26(
            RealisticEnvV10Config().fps
        ),
        command_delay_steps_range=(0, 0),
        command_loss_probability=0.0,
        command_burst_start_probability=0.0,
    )
    policy = AsymmetricMultiViewPPOConfigV1()
    action = StockGripperTaskSpaceActionConfigV12()
    bundle = initialize_asymmetric_multiview_ppo_v1(
        initialization_seed,
        device="cpu",
        scene_path=SCENE_PATH_V26,
    )
    actor_state = bundle.actor.state_dict()
    critic_state = bundle.critic.state_dict()
    if not torch.equal(
        actor_state["mean_head.weight"],
        torch.zeros_like(actor_state["mean_head.weight"]),
    ) or not torch.equal(
        actor_state["mean_head.bias"],
        torch.zeros_like(actor_state["mean_head.bias"]),
    ):
        raise RuntimeError("V26 scratch genesis action mean is not exactly zero")
    plan = {
        "format": V26_REVERSE_CURRICULUM_GENESIS_RUN_FORMAT,
        "created_at_utc": _utc_now(),
        "source_type": SOURCE_TYPE,
        "initialization_seed": initialization_seed,
        "random_initialization": True,
        "zero_mean_action_head_initialization": True,
        "reverse_curriculum_episode_horizon": {
            "max_steps": environment.max_steps,
            "strict_success_hold_steps": environment.strict_success_hold_steps,
            "minimum_pre_hold_action_steps": (
                environment.max_steps - environment.strict_success_hold_steps
            ),
            "deterministic_initial_transport": True,
            "command_delay_steps_range": list(
                environment.command_delay_steps_range
            ),
            "command_loss_probability": environment.command_loss_probability,
            "command_burst_start_probability": (
                environment.command_burst_start_probability
            ),
        },
        "environment_config": asdict(environment),
        "ppo_config": asdict(policy),
        "action_contract": {
            "adapter": "StockGripperTaskFrameAdapterV22",
            "adapter_config": asdict(action),
            "policy_action_space": "forward_lateral_vertical_taskframe",
        },
        "scene_path": str(SCENE_PATH_V26),
        "scene_sha256": sha256_file_v1(SCENE_PATH_V26),
        "parameter_counts": parameter_counts_v1(bundle),
        "provenance": asdict(bundle.provenance),
        "optimizer_steps": 0,
        "expert_calls": 0,
        "behavior_cloning_steps": 0,
        "external_pretraining": False,
        "physical_samples": 0,
        "production_admission": False,
    }
    plan["run_plan_sha256"] = canonical_sha256_v1(plan)
    _atomic_json(output / "run_plan.json", plan)
    checkpoint = {
        "format": V26_REVERSE_CURRICULUM_GENESIS_CHECKPOINT_FORMAT,
        "created_at_utc": _utc_now(),
        "source_type": SOURCE_TYPE,
        "initialization_seed": initialization_seed,
        "run_plan_sha256": plan["run_plan_sha256"],
        "actor_state_dict": actor_state,
        "critic_state_dict": critic_state,
        "actor_state_sha256": state_dict_sha256_v1(actor_state),
        "actor_mean_state_sha256": _mean_state_sha256(actor_state),
        "critic_state_sha256": state_dict_sha256_v1(critic_state),
        "provenance": asdict(bundle.provenance),
        "optimizer_state_dict": None,
        "optimizer_steps": 0,
        "random_initialization": True,
        "zero_mean_action_head_initialization": True,
        "expert_calls": 0,
        "behavior_cloning_steps": 0,
        "external_pretraining": False,
        "physical_samples": 0,
        "production_admission": False,
    }
    checkpoint_path = output / "checkpoints" / "genesis.pt"
    _atomic_torch(checkpoint_path, checkpoint)
    summary = {
        "format": V26_REVERSE_CURRICULUM_GENESIS_RUN_FORMAT,
        "status": "complete",
        "completed_at_utc": _utc_now(),
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": sha256_file_v1(checkpoint_path),
        "actor_state_sha256": checkpoint["actor_state_sha256"],
        "critic_state_sha256": checkpoint["critic_state_sha256"],
        "zero_mean_action_head_initialization": True,
        "optimizer_steps": 0,
        "expert_calls": 0,
        "behavior_cloning_steps": 0,
        "physical_samples": 0,
        "production_admission": False,
    }
    _atomic_json(output / "summary.json", summary)
    return summary


def load_reverse_curriculum_genesis_v26(
    checkpoint_path: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    path = Path(checkpoint_path).expanduser().resolve()
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict):
        raise TypeError("V26 genesis checkpoint must be a dictionary")
    required = {
        "format": V26_REVERSE_CURRICULUM_GENESIS_CHECKPOINT_FORMAT,
        "source_type": SOURCE_TYPE,
        "random_initialization": True,
        "zero_mean_action_head_initialization": True,
        "optimizer_steps": 0,
        "expert_calls": 0,
        "behavior_cloning_steps": 0,
        "external_pretraining": False,
        "physical_samples": 0,
        "production_admission": False,
    }
    if any(checkpoint.get(name) != expected for name, expected in required.items()):
        raise ValueError("V26 genesis checkpoint contract changed")
    actor_state = checkpoint.get("actor_state_dict")
    critic_state = checkpoint.get("critic_state_dict")
    if not isinstance(actor_state, dict) or not isinstance(critic_state, dict):
        raise TypeError("V26 genesis model states are missing")
    if checkpoint.get("optimizer_state_dict") is not None:
        raise ValueError("V26 genesis unexpectedly contains optimizer history")
    plan_path = path.parents[1] / "run_plan.json"
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if not isinstance(plan, dict):
        raise TypeError("V26 genesis run plan must be an object")
    stored_plan_hash = plan.get("run_plan_sha256")
    unhashed = dict(plan)
    unhashed.pop("run_plan_sha256", None)
    if (
        plan.get("format") != V26_REVERSE_CURRICULUM_GENESIS_RUN_FORMAT
        or stored_plan_hash != checkpoint.get("run_plan_sha256")
        or stored_plan_hash != canonical_sha256_v1(unhashed)
    ):
        raise ValueError("V26 genesis run-plan hash chain changed")
    _environment, _policy, _action, scene = _root_configs_from_plan(plan)
    seed = checkpoint.get("initialization_seed")
    if type(seed) is not int or seed != plan.get("initialization_seed"):
        raise ValueError("V26 genesis initialization seed changed")
    expected = initialize_asymmetric_multiview_ppo_v1(
        seed,
        device="cpu",
        scene_path=scene,
    )
    expected_actor_hash = state_dict_sha256_v1(expected.actor.state_dict())
    expected_critic_hash = state_dict_sha256_v1(expected.critic.state_dict())
    actor_hash = state_dict_sha256_v1(actor_state)
    critic_hash = state_dict_sha256_v1(critic_state)
    if (
        actor_hash != expected_actor_hash
        or critic_hash != expected_critic_hash
        or actor_hash != checkpoint.get("actor_state_sha256")
        or critic_hash != checkpoint.get("critic_state_sha256")
        or _mean_state_sha256(actor_state) != checkpoint.get("actor_mean_state_sha256")
        or checkpoint.get("provenance") != asdict(expected.provenance)
    ):
        raise ValueError("V26 genesis weights are not the exact seeded initialization")
    lineage = {
        "format": V26_REVERSE_CURRICULUM_GENESIS_LINEAGE_FORMAT,
        "parent_checkpoint_path": str(path),
        "parent_checkpoint_sha256": sha256_file_v1(path),
        "parent_run_plan_path": str(plan_path),
        "parent_run_plan_sha256": stored_plan_hash,
        "parent_actor_state_sha256": actor_hash,
        "parent_actor_mean_state_sha256": _mean_state_sha256(actor_state),
        "parent_critic_state_sha256": critic_hash,
        "accepted_goal_directed_updates": 0,
        "random_initialization": True,
        "zero_mean_action_head_initialization": True,
        "offline_replay_actor_updates": 0,
        "expert_calls": 0,
        "behavior_cloning_steps": 0,
        "physical_samples": 0,
        "production_admission": False,
    }
    return actor_state, critic_state, plan, lineage


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--initialization-seed", type=int, default=27_000_000)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    summary = create_reverse_curriculum_genesis_v26(
        output_directory=args.output_directory,
        initialization_seed=args.initialization_seed,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "V26_REVERSE_CURRICULUM_GENESIS_CHECKPOINT_FORMAT",
    "V26_REVERSE_CURRICULUM_GENESIS_LINEAGE_FORMAT",
    "V26_REVERSE_CURRICULUM_GENESIS_RUN_FORMAT",
    "V26_REVERSE_CURRICULUM_EPISODE_MAX_STEPS",
    "create_reverse_curriculum_genesis_v26",
    "load_reverse_curriculum_genesis_v26",
]
