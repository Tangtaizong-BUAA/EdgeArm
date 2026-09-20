"""Train the causal smooth relay on certified 20.5--25.5 cm tasks.

V650 composes four independently audited changes:

* the frozen V640 transport actor and feasibility model;
* only the V645 acquisition actor as a warm start, never its polluted replay;
* V646 smooth acquisition and causal object-motion enforcement; and
* V648/V649 long-range task bands with a final exact-Home reset.

No expert action, scripted route, behavior cloning, ACT update, wrist-image
bulk export, or success inferred from HER is permitted.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import torch

from .adaptive_start_curriculum_v622 import START_TIERS_V622
from .asymmetric_multiview_ppo_v1 import (
    SOURCE_TYPE,
    canonical_sha256_v1,
    sha256_file_v1,
)
from .causal_smooth_relay_v646 import (
    AcquisitionActionSlewFilterV646,
    CausalSmoothRelayConfigV646,
    PrecontactBlockCausalStabilizerV646,
    precontact_causal_motion_audit_v646,
    sample_causal_smooth_relay_batch_v646,
    strengthened_relay_config_v646,
)
from .goal_conditioned_her_sac_v43 import GoalConditionedHerSACConfigV43
from .goal_conditioned_markov_her_sac_v614 import (
    GOAL_CONDITIONED_MARKOV_HER_CHECKPOINT_FORMAT_V614,
    GoalConditionedMarkovHerReplayV614,
)
from .phase_isolated_acquisition_v626 import (
    PhaseIsolatedAcquisitionConfigV626,
)
from .ppo_utils_v1 import state_dict_sha256_v1
from .relay_dual_goal_her_sac_v643 import (
    RELAY_DUAL_GOAL_CHECKPOINT_FORMAT_V643,
    RELAY_DUAL_GOAL_HER_SAC_FORMAT_V643,
    RelayDualGoalHerSACBundleV643,
    RelayDualGoalHerSACConfigV643,
    initialize_relay_dual_goal_her_sac_v643,
    relay_dual_goal_her_sac_update_v643,
)
from .relay_taskframe_error_her_sac_v652 import (
    RELAY_TASKFRAME_ERROR_HER_SAC_FORMAT_V652,
    RelayTaskframeErrorHerSACBundleV652,
    initialize_relay_taskframe_error_her_sac_v652,
    relay_taskframe_error_her_sac_update_v652,
    sample_taskframe_error_causal_batch_v652,
)
from .reverse_curriculum_v26 import (
    EXTENDED_RANGE_STAGE_INDEX_V648,
    LONG_RANGE_STAGE_INDEX_V648,
    reverse_curriculum_stage_v26,
    wilson_lower_bound_v26,
)
from .sim2real_env_v10 import RealisticEdgeArmEnvV10
from .task_independent_certified_long_range_reset_v649 import (
    CERTIFIED_LONG_RANGE_HOME_RESET_FORMAT_V649,
    reset_stock_home_certified_long_range_episode_v649,
)
from .task_independent_home_reset_v597 import (
    StockGripperHomeTaskFrameAdapterV597,
)
from .taskframe_controller_state_v614 import (
    TASKFRAME_CONTROLLER_STATE_SCHEMA_SHA256_V614,
)
from .train_goal_conditioned_her_sac_v43 import (
    AxisScaledColoredExplorationConfigV605,
    _append_jsonl,
    _atomic_json,
    _atomic_torch_save,
    _load_collection_contract,
    _resolve_device,
    _utc_now,
)
from .train_relay_dual_goal_her_sac_v643 import (
    _run_episode_v643,
)


CAUSAL_LONG_RANGE_RELAY_FORMAT_V650 = (
    "edgearm-v650-causal-smooth-certified-long-range-relay-v1"
)
CAUSAL_LONG_RANGE_CHECKPOINT_FORMAT_V650 = (
    "edgearm-v650-causal-smooth-certified-long-range-checkpoint-v1"
)
CAUSAL_LONG_RANGE_EVALUATION_FORMAT_V650 = (
    "edgearm-v650-causal-smooth-certified-long-range-evaluation-v1"
)
TASKFRAME_ERROR_LONG_RANGE_RELAY_FORMAT_V653 = (
    "edgearm-v653-taskframe-error-certified-long-range-relay-v1"
)
TASKFRAME_ERROR_LONG_RANGE_CHECKPOINT_FORMAT_V653 = (
    "edgearm-v653-taskframe-error-certified-long-range-checkpoint-v1"
)
TASKFRAME_ERROR_LONG_RANGE_EVALUATION_FORMAT_V653 = (
    "edgearm-v653-taskframe-error-certified-long-range-evaluation-v1"
)
_STAGES_V650 = (
    LONG_RANGE_STAGE_INDEX_V648,
    EXTENDED_RANGE_STAGE_INDEX_V648,
)
_HOME_TIER_INDEX_V650 = next(
    tier.index for tier in START_TIERS_V622 if tier.code == "home"
)


def _import_acquisition_actor_v650(
    bundle: RelayDualGoalHerSACBundleV643,
    warm_payload: dict[str, Any],
    *,
    parent_checkpoint_sha256: str,
) -> dict[str, Any]:
    """Import only V645 acquisition weights into fresh V650 optimizers/Qs."""

    if (
        type(warm_payload) is not dict
        or warm_payload.get("format")
        != RELAY_DUAL_GOAL_CHECKPOINT_FORMAT_V643
        or warm_payload.get("algorithm_format")
        != RELAY_DUAL_GOAL_HER_SAC_FORMAT_V643
        or warm_payload.get("parent_v614_checkpoint_sha256")
        != parent_checkpoint_sha256
        or warm_payload.get("production_admission") is not False
    ):
        raise ValueError("V650 warm acquisition checkpoint identity changed")
    warm_config = RelayDualGoalHerSACConfigV643(
        **warm_payload["config"]
    )
    if (
        warm_config.hidden_dim != bundle.config.hidden_dim
        or PhaseIsolatedAcquisitionConfigV626(
            **warm_payload["phase_isolated_acquisition_v626"]
        )
        != bundle.policy.phase_config
    ):
        raise ValueError("V650 warm acquisition architecture changed")

    transport_before = {
        name: value.detach().cpu().clone()
        for name, value in bundle.policy.transport_actor.state_dict().items()
    }
    feasibility_before = {
        name: value.detach().cpu().clone()
        for name, value in bundle.frozen_feasibility.state_dict().items()
    }
    merged = bundle.policy.state_dict()
    source = warm_payload["policy_state_dict"]
    acquisition_keys = {
        name for name in merged if name.startswith("acquisition_actor.")
    }
    if not acquisition_keys or acquisition_keys != {
        name for name in source if name.startswith("acquisition_actor.")
    }:
        raise ValueError("V650 warm acquisition tensor identity changed")
    for name in acquisition_keys:
        value = source[name]
        if value.shape != merged[name].shape:
            raise ValueError("V650 warm acquisition tensor shape changed")
        merged[name] = value.detach().clone()
    bundle.policy.load_state_dict(merged, strict=True)
    if any(
        not torch.equal(value.detach().cpu(), transport_before[name])
        for name, value in bundle.policy.transport_actor.state_dict().items()
    ):
        raise RuntimeError("V650 warm import changed frozen transport")
    if any(
        not torch.equal(value.detach().cpu(), feasibility_before[name])
        for name, value in bundle.frozen_feasibility.state_dict().items()
    ):
        raise RuntimeError("V650 warm import changed frozen feasibility")
    return {
        "format": "edgearm-v650-acquisition-only-warm-import-v1",
        "source_update_index": int(warm_payload["update_index"]),
        "imported_tensor_count": len(acquisition_keys),
        "acquisition_actor_sha256": state_dict_sha256_v1(
            bundle.policy.acquisition_actor.state_dict()
        ),
        "transport_actor_sha256": state_dict_sha256_v1(
            bundle.policy.transport_actor.state_dict()
        ),
        "frozen_feasibility_sha256": state_dict_sha256_v1(
            bundle.frozen_feasibility.state_dict()
        ),
        "warm_critic_imported": False,
        "warm_optimizer_imported": False,
        "warm_replay_imported": False,
        "v645_online_drift_episodes_imported": False,
        "production_admission": False,
    }


def _stage_runtime_v650(
    base_environment: Any,
    base_action: Any,
    *,
    stage_index: int,
    maximum_steps: int,
    causal_config: CausalSmoothRelayConfigV646,
) -> tuple[Any, Any]:
    stage = reverse_curriculum_stage_v26(stage_index)
    environment = replace(
        base_environment,
        reverse_curriculum_stage_v26=stage_index,
        obstacle_probability=0.0,
        strict_success_hold_steps=90,
        max_steps=max(maximum_steps, int(base_environment.max_steps)),
        physics_substeps=max(
            causal_config.minimum_physics_substeps,
            int(base_environment.physics_substeps),
        ),
    )
    action = replace(
        base_action,
        curriculum_reset_tip_gap_band_m=stage.tip_gap_range_m,
    )
    return environment, action


def _run_causal_episode_v650(
    env: RealisticEdgeArmEnvV10,
    adapter: StockGripperHomeTaskFrameAdapterV597,
    bundle: (
        RelayDualGoalHerSACBundleV643
        | RelayTaskframeErrorHerSACBundleV652
    ),
    *,
    stage_index: int,
    requested_seed: int,
    maximum_steps: int,
    deterministic: bool,
    random_action_probability: float,
    action_seed: int,
    collect_replay: bool,
    causal_config: CausalSmoothRelayConfigV646,
    colored_random_exploration: (
        AxisScaledColoredExplorationConfigV605 | None
    ),
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> tuple[dict[str, Any], dict[str, np.ndarray] | None]:
    action_filter = AcquisitionActionSlewFilterV646(
        phase_config=bundle.policy.phase_config,
        config=causal_config,
    )
    stabilizer = PrecontactBlockCausalStabilizerV646()
    record, episode = _run_episode_v643(
        env,
        adapter,
        bundle,
        requested_seed=requested_seed,
        maximum_steps=maximum_steps,
        deterministic=deterministic,
        random_action_probability=random_action_probability,
        action_seed=action_seed,
        collect_replay=True,
        policy_action_filter=action_filter,
        post_step_causal_stabilizer=stabilizer,
        episode_reset=reset_stock_home_certified_long_range_episode_v649,
        colored_random_exploration=colored_random_exploration,
        progress_callback=progress_callback,
    )
    if episode is None:
        raise RuntimeError("V650 causal episode lost transition evidence")
    causal_audit = precontact_causal_motion_audit_v646(
        episode, config=causal_config
    )
    reset_audit = env.episode_domain.get(
        "certified_long_range_home_reset_v649"
    )
    if (
        type(reset_audit) is not dict
        or reset_audit.get("format")
        != CERTIFIED_LONG_RANGE_HOME_RESET_FORMAT_V649
    ):
        raise RuntimeError("V650 lost certified long-range reset evidence")
    stage = reverse_curriculum_stage_v26(stage_index)
    initial_distance = float(record["initial_block_target_distance_m"])
    reset_valid = bool(
        record["task_independent_final_home_reset"] is True
        and record["task_aligned_privileged_reset"] is False
        and float(record["initial_target_coverage"]) == 0.0
        and reset_audit["privileged_probe_pose_discarded"] is True
        and reset_audit["physics_steps_before_final_policy"] == 0
        and stage.block_target_distance_range_m[0]
        <= initial_distance
        <= stage.block_target_distance_range_m[1]
    )
    replay_admission = bool(
        reset_valid and causal_audit["replay_admission"]
    )
    record.update(
        {
            "object_target_stage_v650": stage_index,
            "object_target_stage_name_v650": stage.name,
            "object_target_distance_range_m_v650": list(
                stage.block_target_distance_range_m
            ),
            "certified_long_range_reset_v649": {
                "format": reset_audit["format"],
                "selected_seed": reset_audit["selected_seed"],
                "selected_attempt_index": reset_audit[
                    "selected_attempt_index"
                ],
                "rejected_candidate_count": len(
                    reset_audit["rejected_candidates"]
                ),
                "task_identity": reset_audit["task_identity"],
                "privileged_probe_pose_discarded": True,
                "privileged_probe_used_as_action_label": False,
            },
            "causal_motion_audit_v646": causal_audit,
            "precontact_stabilizer_summary_v646": stabilizer.summary(),
            "v650_reset_admission": reset_valid,
            "v650_replay_admission": replay_admission,
            "v650_strict_success_admitted": bool(
                record["strict_success"] and replay_admission
            ),
            "start_tier_v622": "home",
            "start_tier_final_data_eligible_v622": True,
            "expert_calls": 0,
            "expert_paths": 0,
            "behavior_cloning_steps": 0,
            "act_training_started": False,
            "bulk_vla_data_use_allowed": False,
            "production_admission": False,
        }
    )
    episode["start_tier_index_v622"] = np.full(
        len(episode["terminal"]),
        _HOME_TIER_INDEX_V650,
        dtype=np.int8,
    )
    return record, episode if collect_replay else None


def _evaluate_stage_v650(
    environment_config: Any,
    action_config: Any,
    scene_path: Path,
    bundle: (
        RelayDualGoalHerSACBundleV643
        | RelayTaskframeErrorHerSACBundleV652
    ),
    *,
    stage_index: int,
    seed_base: int,
    episodes: int,
    maximum_steps: int,
    causal_config: CausalSmoothRelayConfigV646,
    evaluation_format: str = CAUSAL_LONG_RANGE_EVALUATION_FORMAT_V650,
) -> dict[str, Any]:
    env = RealisticEdgeArmEnvV10(
        environment_config,
        seed=seed_base,
        model_scene_path=scene_path,
    )
    adapter = StockGripperHomeTaskFrameAdapterV597(env, action_config)
    records: list[dict[str, Any]] = []
    for index in range(episodes):
        record, _episode = _run_causal_episode_v650(
            env,
            adapter,
            bundle,
            stage_index=stage_index,
            requested_seed=seed_base + index,
            maximum_steps=maximum_steps,
            deterministic=True,
            random_action_probability=0.0,
            action_seed=seed_base ^ (index + 0x650E),
            collect_replay=False,
            causal_config=causal_config,
            colored_random_exploration=None,
        )
        records.append(record)
    strict = sum(
        int(record["v650_strict_success_admitted"])
        for record in records
    )
    contacts = sum(
        int(int(record["valid_contact_steps"]) > 0)
        for record in records
    )
    safety = sum(int(record["safety_steps"] > 0) for record in records)
    invalid = sum(
        int(record["invalid_contact_steps"] > 0) for record in records
    )
    causal_valid = sum(
        int(record["causal_motion_audit_v646"]["causal_motion_valid"])
        for record in records
    )
    return {
        "format": evaluation_format,
        "created_at_utc": _utc_now(),
        "stage": asdict(reverse_curriculum_stage_v26(stage_index)),
        "seed_base": seed_base,
        "episode_count": episodes,
        "strict_success_count": strict,
        "strict_success_rate": strict / episodes,
        "strict_success_wilson_lower_bound": wilson_lower_bound_v26(
            strict, episodes
        ),
        "contact_episode_count": contacts,
        "contact_episode_rate": contacts / episodes,
        "causal_motion_valid_count": causal_valid,
        "causal_motion_valid_rate": causal_valid / episodes,
        "safety_episode_count": safety,
        "invalid_contact_episode_count": invalid,
        "mean_initial_center_distance_m": float(
            np.mean(
                [
                    record["initial_block_target_distance_m"]
                    for record in records
                ]
            )
        ),
        "mean_best_tool_precontact_progress_m": float(
            np.mean(
                [
                    record["relay_dual_goal_runtime_v643"][
                        "best_tool_precontact_progress_m"
                    ]
                    for record in records
                ]
            )
        ),
        "mean_net_target_progress_m": float(
            np.mean([record["net_target_progress_m"] for record in records])
        ),
        "mean_ik_failure_steps": float(
            np.mean([record["ik_failure_steps"] for record in records])
        ),
        "episodes": records,
        "her_successes_in_numerator": 0,
        "exact_home_only": True,
        "three_second_hold_required": True,
        "expert_calls": 0,
        "behavior_cloning_steps": 0,
        "bulk_vla_data_use_allowed": False,
        "production_admission": False,
    }


def _checkpoint_payload_v650(
    bundle: (
        RelayDualGoalHerSACBundleV643
        | RelayTaskframeErrorHerSACBundleV652
    ),
    replay: GoalConditionedMarkovHerReplayV614,
    *,
    run_plan_sha256: str,
    parent_checkpoint_sha256: str,
    warm_checkpoint_sha256: str,
    online_episode_index: int,
    admitted_online_episode_count: int,
    causal_config: CausalSmoothRelayConfigV646,
    warm_import_audit: dict[str, Any],
    checkpoint_format: str = CAUSAL_LONG_RANGE_CHECKPOINT_FORMAT_V650,
    algorithm_format: str = CAUSAL_LONG_RANGE_RELAY_FORMAT_V650,
    acquisition_state_audit: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "format": checkpoint_format,
        "algorithm_format": algorithm_format,
        "created_at_utc": _utc_now(),
        "run_plan_sha256": run_plan_sha256,
        "parent_v614_checkpoint_sha256": parent_checkpoint_sha256,
        "warm_v643_checkpoint_sha256": warm_checkpoint_sha256,
        "online_episode_index": online_episode_index,
        "admitted_online_episode_count": admitted_online_episode_count,
        "update_index": bundle.update_index,
        "config": asdict(bundle.config),
        "causal_smooth_config_v646": asdict(causal_config),
        "phase_isolated_acquisition_v626": asdict(
            bundle.policy.phase_config
        ),
        "long_range_stage_indices": list(_STAGES_V650),
        "warm_import_audit": warm_import_audit,
        "acquisition_state_audit": acquisition_state_audit,
        "policy_state_dict": bundle.policy.state_dict(),
        "acquisition_critic_state_dict": (
            bundle.acquisition_critic.state_dict()
        ),
        "target_acquisition_critic_state_dict": (
            bundle.target_acquisition_critic.state_dict()
        ),
        "frozen_feasibility_state_dict": (
            bundle.frozen_feasibility.state_dict()
        ),
        "actor_optimizer_state_dict": bundle.actor_optimizer.state_dict(),
        "critic_optimizer_state_dict": bundle.critic_optimizer.state_dict(),
        "replay_manifest": replay.manifest(),
        "controller_state_schema_sha256": (
            TASKFRAME_CONTROLLER_STATE_SCHEMA_SHA256_V614
        ),
        "expert_calls": 0,
        "behavior_cloning_steps": 0,
        "act_training_started": False,
        "bulk_wrist_multimodal_generation_started": False,
        "production_admission": False,
    }


def run_causal_smooth_long_range_relay_v650(
    *,
    output_dir: Path,
    collection_plan: Path,
    parent_v614_checkpoint: Path,
    source_replay_npz: Path,
    warm_v643_checkpoint: Path,
    initialization_seed: int,
    sampling_seed_base: int,
    online_seed_base: int,
    evaluation_seed_base: int,
    offline_updates: int,
    online_episodes: int,
    updates_per_episode: int,
    evaluation_every_episodes: int,
    evaluation_episodes_per_stage: int,
    maximum_episode_steps: int,
    initial_random_action_probability: float,
    final_random_action_probability: float,
    device: str,
    colored_random_exploration: (
        AxisScaledColoredExplorationConfigV605 | None
    ) = None,
    acquisition_state_mode: str = "v643_absolute_world_goal",
) -> dict[str, Any]:
    if acquisition_state_mode not in {
        "v643_absolute_world_goal",
        "v652_taskframe_relative_error",
    }:
        raise ValueError("V650 acquisition state mode is invalid")
    taskframe_error_mode = bool(
        acquisition_state_mode == "v652_taskframe_relative_error"
    )
    run_format = (
        TASKFRAME_ERROR_LONG_RANGE_RELAY_FORMAT_V653
        if taskframe_error_mode
        else CAUSAL_LONG_RANGE_RELAY_FORMAT_V650
    )
    checkpoint_format = (
        TASKFRAME_ERROR_LONG_RANGE_CHECKPOINT_FORMAT_V653
        if taskframe_error_mode
        else CAUSAL_LONG_RANGE_CHECKPOINT_FORMAT_V650
    )
    evaluation_format = (
        TASKFRAME_ERROR_LONG_RANGE_EVALUATION_FORMAT_V653
        if taskframe_error_mode
        else CAUSAL_LONG_RANGE_EVALUATION_FORMAT_V650
    )
    offline_phase = (
        "offline_taskframe_error_recalibration_v653"
        if taskframe_error_mode
        else "offline_causal_recalibration_v650"
    )
    online_phase = (
        "online_taskframe_error_long_range_v653"
        if taskframe_error_mode
        else "online_certified_long_range_v650"
    )
    for name, value, minimum in (
        ("offline_updates", offline_updates, 0),
        ("online_episodes", online_episodes, 1),
        ("updates_per_episode", updates_per_episode, 1),
        ("evaluation_every_episodes", evaluation_every_episodes, 1),
        (
            "evaluation_episodes_per_stage",
            evaluation_episodes_per_stage,
            0,
        ),
        ("maximum_episode_steps", maximum_episode_steps, 180),
    ):
        if type(value) is not int or value < minimum:
            raise ValueError(f"V650 {name} is invalid")
    probabilities = (
        initial_random_action_probability,
        final_random_action_probability,
    )
    if any(
        not np.isfinite(value) or not 0.0 <= value <= 1.0
        for value in probabilities
    ):
        raise ValueError("V650 exploration probability is invalid")
    destination = Path(output_dir).expanduser().resolve()
    if destination.exists():
        raise FileExistsError(f"V650 output already exists: {destination}")
    parent_path = Path(parent_v614_checkpoint).expanduser().resolve()
    replay_path = Path(source_replay_npz).expanduser().resolve()
    warm_path = Path(warm_v643_checkpoint).expanduser().resolve()
    plan_path = Path(collection_plan).expanduser().resolve()
    if not all(
        path.is_file()
        for path in (parent_path, replay_path, warm_path, plan_path)
    ):
        raise FileNotFoundError("V650 source checkpoint, replay, or plan is missing")

    parent_sha256 = sha256_file_v1(parent_path)
    warm_sha256 = sha256_file_v1(warm_path)
    parent_payload = torch.load(
        parent_path, map_location="cpu", weights_only=True
    )
    warm_payload = torch.load(
        warm_path, map_location="cpu", weights_only=True
    )
    if (
        type(parent_payload) is not dict
        or parent_payload.get("format")
        != GOAL_CONDITIONED_MARKOV_HER_CHECKPOINT_FORMAT_V614
    ):
        raise ValueError("V650 parent is not a V614 checkpoint")
    if (
        type(warm_payload) is not dict
        or warm_payload.get("format")
        != RELAY_DUAL_GOAL_CHECKPOINT_FORMAT_V643
        or warm_payload.get("algorithm_format")
        != RELAY_DUAL_GOAL_HER_SAC_FORMAT_V643
    ):
        raise ValueError("V650 warm source is not a V643 checkpoint")
    parent_config = GoalConditionedHerSACConfigV43(
        **parent_payload["config"]
    )
    phase_config = PhaseIsolatedAcquisitionConfigV626(
        **parent_payload["phase_isolated_acquisition_v626"]
    )
    warm_config = RelayDualGoalHerSACConfigV643(
        **warm_payload["config"]
    )
    base_config = RelayDualGoalHerSACConfigV643(
        **{
            **asdict(warm_config),
            "hidden_dim": parent_config.hidden_dim,
            "batch_size": parent_config.batch_size,
        }
    )
    selected_config = strengthened_relay_config_v646(base_config)
    causal_config = CausalSmoothRelayConfigV646()
    causal_config.validate()
    replay = GoalConditionedMarkovHerReplayV614.load_npz(
        replay_path, parent_config
    )
    parent_replay_episode_count = replay.episode_count
    selected_device = _resolve_device(device)
    if taskframe_error_mode:
        bundle, upgrade_audit = (
            initialize_relay_taskframe_error_her_sac_v652(
                seed=initialization_seed,
                device=selected_device,
                parent_v614_payload=parent_payload,
                phase_config=phase_config,
                config=selected_config,
            )
        )
        warm_import_audit = {
            "format": "edgearm-v653-v643-warm-exclusion-v1",
            "source_checkpoint_sha256": warm_sha256,
            "source_used_for_hyperparameter_shape_only": True,
            "source_acquisition_actor_imported": False,
            "source_critic_imported": False,
            "source_optimizer_imported": False,
            "source_replay_imported": False,
            "reason": (
                "v643_absolute_world_goal_semantics_incompatible_with_"
                "v652_taskframe_relative_error_semantics"
            ),
            "production_admission": False,
        }
        acquisition_state_audit = {
            "format": RELAY_TASKFRAME_ERROR_HER_SAC_FORMAT_V652,
            "mode": acquisition_state_mode,
            "raw_world_goal_at_acquisition_head": False,
            "normalized_taskframe_relative_error_at_head": True,
            "feature_is_action_or_path": False,
            "expert_action_used": False,
            "behavior_cloning_steps": 0,
            "production_admission": False,
        }
    else:
        bundle, upgrade_audit = initialize_relay_dual_goal_her_sac_v643(
            seed=initialization_seed,
            device=selected_device,
            parent_v614_payload=parent_payload,
            phase_config=phase_config,
            config=selected_config,
        )
        warm_import_audit = _import_acquisition_actor_v650(
            bundle,
            warm_payload,
            parent_checkpoint_sha256=parent_sha256,
        )
        acquisition_state_audit = {
            "format": RELAY_DUAL_GOAL_HER_SAC_FORMAT_V643,
            "mode": acquisition_state_mode,
            "raw_world_goal_at_acquisition_head": True,
            "normalized_taskframe_relative_error_at_head": False,
            "production_admission": False,
        }

    source_plan, base_environment, base_action, scene_path = (
        _load_collection_contract(plan_path)
    )
    runtimes = {
        stage_index: _stage_runtime_v650(
            base_environment,
            base_action,
            stage_index=stage_index,
            maximum_steps=maximum_episode_steps,
            causal_config=causal_config,
        )
        for stage_index in _STAGES_V650
    }
    source_file = Path(__file__).resolve()
    run_plan = {
        "format": run_format,
        "created_at_utc": _utc_now(),
        "output_dir": str(destination),
        "resolved_device": selected_device,
        "configuration": asdict(selected_config),
        "causal_smooth_config_v646": asdict(causal_config),
        "phase_isolated_acquisition_v626": asdict(phase_config),
        "parent_v614_checkpoint": str(parent_path),
        "parent_v614_checkpoint_sha256": parent_sha256,
        "source_replay_npz": str(replay_path),
        "source_replay_npz_sha256": sha256_file_v1(replay_path),
        "source_replay_scope": (
            "legacy_15_19cm_curriculum_learning_only_never_export"
        ),
        "parent_replay_episode_count": parent_replay_episode_count,
        "warm_v643_checkpoint": str(warm_path),
        "warm_v643_checkpoint_sha256": warm_sha256,
        "warm_import_audit": warm_import_audit,
        "acquisition_state_audit": acquisition_state_audit,
        "frozen_parent_upgrade_audit": upgrade_audit,
        "v645_replay_imported": False,
        "v645_online_drift_episodes_imported": False,
        "v645_acquisition_actor_imported": not taskframe_error_mode,
        "offline_updates": offline_updates,
        "online_episodes": online_episodes,
        "updates_per_episode": updates_per_episode,
        "evaluation_every_episodes": evaluation_every_episodes,
        "evaluation_episodes_per_stage": evaluation_episodes_per_stage,
        "maximum_episode_steps": maximum_episode_steps,
        "initial_random_action_probability": (
            initial_random_action_probability
        ),
        "final_random_action_probability": final_random_action_probability,
        "initialization_seed": initialization_seed,
        "sampling_seed_base": sampling_seed_base,
        "online_seed_base": online_seed_base,
        "evaluation_seed_base": evaluation_seed_base,
        "long_range_stages": [
            asdict(reverse_curriculum_stage_v26(index))
            for index in _STAGES_V650
        ],
        "new_online_distance_union_m": [0.205, 0.255],
        "new_online_reset": (
            "v649_certified_task_then_identical_exact_home"
        ),
        "new_online_initial_target_overlap_allowed": False,
        "source_collection_plan": str(plan_path),
        "source_collection_plan_sha256": source_plan["run_plan_sha256"],
        "scene_path": str(scene_path),
        "scene_sha256": sha256_file_v1(scene_path),
        "stage_environment_configs": {
            str(index): asdict(runtimes[index][0])
            for index in _STAGES_V650
        },
        "stage_action_configs": {
            str(index): asdict(runtimes[index][1])
            for index in _STAGES_V650
        },
        "frozen_transport_at_zero_gate": True,
        "simulator_privileged_actor": True,
        "visual_actor_training": False,
        "act_training": False,
        "bulk_wrist_multimodal_generation_started": False,
        "expert_calls": 0,
        "behavior_cloning_steps": 0,
        "source_type": SOURCE_TYPE,
        "source_hashes": {
            "edgearm/train_causal_smooth_long_range_relay_v650.py": (
                sha256_file_v1(source_file)
            ),
            "edgearm/causal_smooth_relay_v646.py": sha256_file_v1(
                source_file.with_name("causal_smooth_relay_v646.py")
            ),
            "edgearm/reverse_curriculum_v26.py": sha256_file_v1(
                source_file.with_name("reverse_curriculum_v26.py")
            ),
            "edgearm/relay_taskframe_error_her_sac_v652.py": (
                sha256_file_v1(
                    source_file.with_name(
                        "relay_taskframe_error_her_sac_v652.py"
                    )
                )
            ),
            (
                "edgearm/task_independent_certified_long_range_reset_v649.py"
            ): sha256_file_v1(
                source_file.with_name(
                    "task_independent_certified_long_range_reset_v649.py"
                )
            ),
        },
        "production_admission": False,
    }
    run_plan["run_plan_sha256"] = canonical_sha256_v1(run_plan)
    destination.mkdir(parents=True)
    (destination / "checkpoints").mkdir()
    (destination / "evaluations").mkdir()
    _atomic_json(destination / "run_plan.json", run_plan)

    latest_metrics: dict[str, Any] | None = None
    evaluations: list[dict[str, Any]] = []
    online_records: list[dict[str, Any]] = []
    admitted_online_episodes = 0
    best_score: tuple[float, ...] | None = None
    best_checkpoint: Path | None = None

    def sample_training_batch(seed: int) -> dict[str, np.ndarray]:
        if taskframe_error_mode:
            if type(bundle) is not RelayTaskframeErrorHerSACBundleV652:
                raise RuntimeError("V653 task-frame bundle identity changed")
            return sample_taskframe_error_causal_batch_v652(
                replay,
                batch_size=selected_config.batch_size,
                seed=seed,
                phase_config=phase_config,
                relay_config=selected_config,
                causal_config=causal_config,
            )
        return sample_causal_smooth_relay_batch_v646(
            replay,
            batch_size=selected_config.batch_size,
            seed=seed,
            phase_config=phase_config,
            relay_config=selected_config,
            causal_config=causal_config,
        )

    def update_bundle(batch: dict[str, np.ndarray]) -> Any:
        if taskframe_error_mode:
            if type(bundle) is not RelayTaskframeErrorHerSACBundleV652:
                raise RuntimeError("V653 task-frame bundle identity changed")
            return relay_taskframe_error_her_sac_update_v652(bundle, batch)
        if type(bundle) is not RelayDualGoalHerSACBundleV643:
            raise RuntimeError("V650 absolute-goal bundle identity changed")
        return relay_dual_goal_her_sac_update_v643(bundle, batch)

    try:
        for offline_index in range(1, offline_updates + 1):
            batch = sample_training_batch(
                sampling_seed_base + bundle.update_index
            )
            metrics = update_bundle(batch)
            latest_metrics = {
                **asdict(metrics),
                "phase": offline_phase,
                "offline_update_index": offline_index,
                "uncausal_batch_fraction": float(
                    np.mean(batch["uncausal_precontact_motion"])
                ),
                "mean_recorded_action_rate_l2": float(
                    np.mean(batch["recorded_action_rate_l2"])
                ),
            }
            _append_jsonl(destination / "metrics.jsonl", latest_metrics)
            if offline_index % 64 == 0 or offline_index == offline_updates:
                _atomic_json(
                    destination / "run_state.json",
                    {
                        "status": "running",
                        "phase": offline_phase,
                        "offline_update_index": offline_index,
                        "offline_updates": offline_updates,
                        "update_index": bundle.update_index,
                        "latest_metrics": latest_metrics,
                        "updated_at_utc": _utc_now(),
                        "production_admission": False,
                    },
                )

        environments: dict[int, RealisticEdgeArmEnvV10] = {}
        adapters: dict[int, StockGripperHomeTaskFrameAdapterV597] = {}
        for stage_index in _STAGES_V650:
            environment, action = runtimes[stage_index]
            env = RealisticEdgeArmEnvV10(
                environment,
                seed=online_seed_base + stage_index * 1_000_000,
                model_scene_path=scene_path,
            )
            environments[stage_index] = env
            adapters[stage_index] = StockGripperHomeTaskFrameAdapterV597(
                env, action
            )

        for online_index in range(1, online_episodes + 1):
            stage_index = _STAGES_V650[
                (online_index - 1) % len(_STAGES_V650)
            ]
            fraction = (online_index - 1) / max(online_episodes - 1, 1)
            random_probability = initial_random_action_probability + fraction * (
                final_random_action_probability
                - initial_random_action_probability
            )
            requested_seed = (
                online_seed_base
                + stage_index * 1_000_000
                + online_index
            )

            def write_episode_heartbeat(
                row: dict[str, Any],
                *,
                active_online_index: int = online_index,
                active_stage_index: int = stage_index,
            ) -> None:
                completed_steps = int(row["episode_step"]) + 1
                if completed_steps % 30 != 0 and not bool(row["terminal"]):
                    return
                current_distance = float(
                    np.linalg.norm(
                        np.asarray(row["next_achieved_goal"], dtype=np.float64)
                        - np.asarray(row["desired_goal"], dtype=np.float64)
                    )
                )
                _atomic_json(
                    destination / "run_state.json",
                    {
                        "status": "running",
                        "phase": (
                            "online_episode_execution_v653"
                            if taskframe_error_mode
                            else "online_episode_execution_v650"
                        ),
                        "v650_online_episode_index": active_online_index,
                        "online_episodes": online_episodes,
                        "stage_index": active_stage_index,
                        "episode_step": completed_steps,
                        "maximum_episode_steps": maximum_episode_steps,
                        "current_block_target_distance_m": current_distance,
                        "latest_step_valid_contact": bool(
                            row["valid_contact"]
                        ),
                        "latest_step_block_displacement_m": float(
                            row["step_block_displacement_m"]
                        ),
                        "latest_step_action_feasible": bool(
                            row["action_feasible"]
                        ),
                        "latest_step_terminal_reason": str(
                            row["terminal_reason"]
                        ),
                        "admitted_online_episode_count": (
                            admitted_online_episodes
                        ),
                        "update_index": bundle.update_index,
                        "updated_at_utc": _utc_now(),
                        "production_admission": False,
                    },
                )

            record, episode = _run_causal_episode_v650(
                environments[stage_index],
                adapters[stage_index],
                bundle,
                stage_index=stage_index,
                requested_seed=requested_seed,
                maximum_steps=maximum_episode_steps,
                deterministic=False,
                random_action_probability=float(random_probability),
                action_seed=online_seed_base ^ (online_index * 0x650A11),
                collect_replay=True,
                causal_config=causal_config,
                colored_random_exploration=colored_random_exploration,
                progress_callback=write_episode_heartbeat,
            )
            if episode is None:
                raise RuntimeError("V650 online episode lost replay")
            record["v650_online_episode_index"] = online_index
            record["acquisition_state_mode"] = acquisition_state_mode
            record["taskframe_error_acquisition_v652"] = (
                taskframe_error_mode
            )
            if record["v650_replay_admission"]:
                replay.add_episode(
                    episode,
                    source=(
                        f"online_{'v653' if taskframe_error_mode else 'v650'}_"
                        f"certified_home_stage{stage_index}_"
                        f"episode_{online_index:06d}"
                    ),
                )
                admitted_online_episodes += 1
            record["replay_transition_count_after_episode"] = (
                replay.transition_count
            )
            record["admitted_online_episode_count"] = (
                admitted_online_episodes
            )
            online_records.append(record)
            _append_jsonl(destination / "online_episodes.jsonl", record)

            for local_update in range(1, updates_per_episode + 1):
                batch = sample_training_batch(
                    sampling_seed_base + bundle.update_index
                )
                metrics = update_bundle(batch)
                latest_metrics = {
                    **asdict(metrics),
                    "phase": online_phase,
                    "v650_online_episode_index": online_index,
                    "stage_index": stage_index,
                    "local_update_after_episode": local_update,
                    "uncausal_batch_fraction": float(
                        np.mean(batch["uncausal_precontact_motion"])
                    ),
                    "mean_recorded_action_rate_l2": float(
                        np.mean(batch["recorded_action_rate_l2"])
                    ),
                }
                _append_jsonl(destination / "metrics.jsonl", latest_metrics)

            replay.save_npz(destination / "replay_latest.npz")
            checkpoint = (
                destination
                / "checkpoints"
                / f"episode_{online_index:06d}.pt"
            )
            _atomic_torch_save(
                checkpoint,
                _checkpoint_payload_v650(
                    bundle,
                    replay,
                    run_plan_sha256=run_plan["run_plan_sha256"],
                    parent_checkpoint_sha256=parent_sha256,
                    warm_checkpoint_sha256=warm_sha256,
                    online_episode_index=online_index,
                    admitted_online_episode_count=admitted_online_episodes,
                    causal_config=causal_config,
                    warm_import_audit=warm_import_audit,
                    checkpoint_format=checkpoint_format,
                    algorithm_format=run_format,
                    acquisition_state_audit=acquisition_state_audit,
                ),
            )
            if (
                evaluation_episodes_per_stage > 0
                and (
                    online_index % evaluation_every_episodes == 0
                    or online_index == online_episodes
                )
            ):
                by_stage: dict[str, Any] = {}
                for eval_stage in _STAGES_V650:
                    environment, action = runtimes[eval_stage]
                    by_stage[str(eval_stage)] = _evaluate_stage_v650(
                        environment,
                        action,
                        scene_path,
                        bundle,
                        stage_index=eval_stage,
                        seed_base=(
                            evaluation_seed_base
                            + eval_stage * 1_000_000
                            + online_index * 10_000
                        ),
                        episodes=evaluation_episodes_per_stage,
                        maximum_steps=maximum_episode_steps,
                        causal_config=causal_config,
                        evaluation_format=evaluation_format,
                    )
                total_episodes = sum(
                    row["episode_count"] for row in by_stage.values()
                )
                total_strict = sum(
                    row["strict_success_count"]
                    for row in by_stage.values()
                )
                evaluation = {
                    "format": evaluation_format,
                    "created_at_utc": _utc_now(),
                    "v650_online_episode_index": online_index,
                    "update_index": bundle.update_index,
                    "by_stage": by_stage,
                    "episode_count": total_episodes,
                    "strict_success_count": total_strict,
                    "strict_success_rate": total_strict / total_episodes,
                    "all_causal_motion_valid": all(
                        row["causal_motion_valid_count"]
                        == row["episode_count"]
                        for row in by_stage.values()
                    ),
                    "formal_gate_passed": False,
                    "formal_gate_block_reason": (
                        "requires at least 24 held-out tasks per stage, "
                        "80% aggregate strict success, Wilson >= 0.65, "
                        "zero safety/invalid contact, and causal validity"
                    ),
                    "bulk_vla_data_use_allowed": False,
                    "production_admission": False,
                }
                enough = all(
                    row["episode_count"] >= 24
                    for row in by_stage.values()
                )
                no_safety = all(
                    row["safety_episode_count"] == 0
                    and row["invalid_contact_episode_count"] == 0
                    for row in by_stage.values()
                )
                stage_wilson = all(
                    row["strict_success_wilson_lower_bound"] >= 0.65
                    for row in by_stage.values()
                )
                evaluation["formal_gate_passed"] = bool(
                    enough
                    and evaluation["strict_success_rate"] >= 0.80
                    and stage_wilson
                    and no_safety
                    and evaluation["all_causal_motion_valid"]
                )
                if evaluation["formal_gate_passed"]:
                    evaluation["formal_gate_block_reason"] = None
                evaluations.append(evaluation)
                _atomic_json(
                    destination
                    / "evaluations"
                    / f"episode_{online_index:06d}.json",
                    evaluation,
                )
                score = (
                    float(evaluation["strict_success_rate"]),
                    float(
                        np.mean(
                            [
                                row["contact_episode_rate"]
                                for row in by_stage.values()
                            ]
                        )
                    ),
                    float(
                        np.mean(
                            [
                                row["mean_best_tool_precontact_progress_m"]
                                for row in by_stage.values()
                            ]
                        )
                    ),
                    -float(
                        np.mean(
                            [
                                row["mean_ik_failure_steps"]
                                for row in by_stage.values()
                            ]
                        )
                    ),
                )
                if best_score is None or score > best_score:
                    best_score = score
                    best_checkpoint = checkpoint
                    _atomic_json(
                        destination / "best_training_selection.json",
                        {
                            "format": (
                                "edgearm-v650-diagnostic-checkpoint-"
                                "selection-v1"
                            ),
                            "selection_score": list(score),
                            "checkpoint": str(checkpoint),
                            "checkpoint_sha256": sha256_file_v1(checkpoint),
                            "evaluation": evaluation,
                            "production_admission": False,
                        },
                    )
            _atomic_json(
                destination / "run_state.json",
                {
                    "status": "running",
                    "phase": online_phase,
                    "v650_online_episode_index": online_index,
                    "online_episodes": online_episodes,
                    "admitted_online_episode_count": admitted_online_episodes,
                    "update_index": bundle.update_index,
                    "latest_online_episode": record,
                    "latest_metrics": latest_metrics,
                    "latest_evaluation": (
                        None if not evaluations else evaluations[-1]
                    ),
                    "replay_transition_count": replay.transition_count,
                    "replay_episode_count": replay.episode_count,
                    "updated_at_utc": _utc_now(),
                    "production_admission": False,
                },
            )
    except Exception as error:
        failure = {
            "format": run_format,
            "status": "failed",
            "error_type": type(error).__name__,
            "error": str(error),
            "online_episode_index": len(online_records),
            "admitted_online_episode_count": admitted_online_episodes,
            "update_index": bundle.update_index,
            "replay_manifest": replay.manifest(),
            "failed_at_utc": _utc_now(),
            "production_admission": False,
        }
        _atomic_json(destination / "failure.json", failure)
        _atomic_json(destination / "run_state.json", failure)
        raise

    replay.save_npz(destination / "replay_final.npz")
    final_checkpoint = destination / "checkpoints" / "final.pt"
    _atomic_torch_save(
        final_checkpoint,
        _checkpoint_payload_v650(
            bundle,
            replay,
            run_plan_sha256=run_plan["run_plan_sha256"],
            parent_checkpoint_sha256=parent_sha256,
            warm_checkpoint_sha256=warm_sha256,
            online_episode_index=online_episodes,
            admitted_online_episode_count=admitted_online_episodes,
            causal_config=causal_config,
            warm_import_audit=warm_import_audit,
            checkpoint_format=checkpoint_format,
            algorithm_format=run_format,
            acquisition_state_audit=acquisition_state_audit,
        ),
    )
    latest_evaluation = None if not evaluations else evaluations[-1]
    summary = {
        "format": run_format,
        "status": "complete",
        "created_at_utc": _utc_now(),
        "run_plan_sha256": run_plan["run_plan_sha256"],
        "update_index": bundle.update_index,
        "parent_replay_episode_count": parent_replay_episode_count,
        "online_episode_count": len(online_records),
        "admitted_online_episode_count": admitted_online_episodes,
        "rejected_online_episode_count": (
            len(online_records) - admitted_online_episodes
        ),
        "strict_online_success_count": sum(
            int(record["v650_strict_success_admitted"])
            for record in online_records
        ),
        "latest_evaluation": latest_evaluation,
        "best_checkpoint": (
            None if best_checkpoint is None else str(best_checkpoint)
        ),
        "final_checkpoint": str(final_checkpoint),
        "final_checkpoint_sha256": sha256_file_v1(final_checkpoint),
        "replay_final": str(destination / "replay_final.npz"),
        "replay_episode_count": replay.episode_count,
        "replay_transition_count": replay.transition_count,
        "formal_gate_passed": bool(
            latest_evaluation is not None
            and latest_evaluation["formal_gate_passed"]
        ),
        "act_training_started": False,
        "bulk_wrist_multimodal_generation_started": False,
        "bulk_vla_data_use_allowed": False,
        "expert_calls": 0,
        "behavior_cloning_steps": 0,
        "production_admission": False,
    }
    _atomic_json(destination / "summary.json", summary)
    _atomic_json(
        destination / "run_state.json",
        {
            "status": "complete",
            "phase": "complete",
            "online_episode_index": online_episodes,
            "admitted_online_episode_count": admitted_online_episodes,
            "update_index": bundle.update_index,
            "latest_evaluation": latest_evaluation,
            "formal_gate_passed": summary["formal_gate_passed"],
            "bulk_vla_data_use_allowed": False,
            "updated_at_utc": _utc_now(),
            "production_admission": False,
        },
    )
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--collection-plan", type=Path, required=True)
    parser.add_argument("--parent-v614-checkpoint", type=Path, required=True)
    parser.add_argument("--source-replay-npz", type=Path, required=True)
    parser.add_argument("--warm-v643-checkpoint", type=Path, required=True)
    parser.add_argument("--initialization-seed", type=int, default=650_000_000)
    parser.add_argument("--sampling-seed-base", type=int, default=650_100_000)
    parser.add_argument("--online-seed-base", type=int, default=650_200_000)
    parser.add_argument("--evaluation-seed-base", type=int, default=650_300_000)
    parser.add_argument("--offline-updates", type=int, default=512)
    parser.add_argument("--online-episodes", type=int, default=8)
    parser.add_argument("--updates-per-episode", type=int, default=96)
    parser.add_argument("--evaluation-every-episodes", type=int, default=2)
    parser.add_argument(
        "--evaluation-episodes-per-stage", type=int, default=2
    )
    parser.add_argument("--maximum-episode-steps", type=int, default=720)
    parser.add_argument(
        "--initial-random-action-probability", type=float, default=0.10
    )
    parser.add_argument(
        "--final-random-action-probability", type=float, default=0.03
    )
    parser.add_argument("--colored-random-exploration", action="store_true")
    parser.add_argument("--colored-exploration-rho", type=float, default=0.94)
    parser.add_argument(
        "--colored-exploration-standard-deviation",
        type=float,
        nargs=3,
        default=(0.35, 0.22, 0.14),
    )
    parser.add_argument("--device", choices=("auto", "cpu", "mps"), default="auto")
    parser.add_argument(
        "--acquisition-state-mode",
        choices=(
            "v643_absolute_world_goal",
            "v652_taskframe_relative_error",
        ),
        default="v643_absolute_world_goal",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    colored = (
        AxisScaledColoredExplorationConfigV605(
            standard_deviation=tuple(
                arguments.colored_exploration_standard_deviation
            ),
            autoregressive_rho=arguments.colored_exploration_rho,
        )
        if arguments.colored_random_exploration
        else None
    )
    summary = run_causal_smooth_long_range_relay_v650(
        output_dir=arguments.output_dir,
        collection_plan=arguments.collection_plan,
        parent_v614_checkpoint=arguments.parent_v614_checkpoint,
        source_replay_npz=arguments.source_replay_npz,
        warm_v643_checkpoint=arguments.warm_v643_checkpoint,
        initialization_seed=arguments.initialization_seed,
        sampling_seed_base=arguments.sampling_seed_base,
        online_seed_base=arguments.online_seed_base,
        evaluation_seed_base=arguments.evaluation_seed_base,
        offline_updates=arguments.offline_updates,
        online_episodes=arguments.online_episodes,
        updates_per_episode=arguments.updates_per_episode,
        evaluation_every_episodes=arguments.evaluation_every_episodes,
        evaluation_episodes_per_stage=(
            arguments.evaluation_episodes_per_stage
        ),
        maximum_episode_steps=arguments.maximum_episode_steps,
        initial_random_action_probability=(
            arguments.initial_random_action_probability
        ),
        final_random_action_probability=(
            arguments.final_random_action_probability
        ),
        device=arguments.device,
        colored_random_exploration=colored,
        acquisition_state_mode=arguments.acquisition_state_mode,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "CAUSAL_LONG_RANGE_CHECKPOINT_FORMAT_V650",
    "CAUSAL_LONG_RANGE_EVALUATION_FORMAT_V650",
    "CAUSAL_LONG_RANGE_RELAY_FORMAT_V650",
    "TASKFRAME_ERROR_LONG_RANGE_CHECKPOINT_FORMAT_V653",
    "TASKFRAME_ERROR_LONG_RANGE_EVALUATION_FORMAT_V653",
    "TASKFRAME_ERROR_LONG_RANGE_RELAY_FORMAT_V653",
    "run_causal_smooth_long_range_relay_v650",
]
