"""Task-frame relative-error acquisition RL for exact-Home starts.

V643 supplied the acquisition goal as an absolute world XYZ coordinate while
the policy action is expressed as task-frame forward/lateral/vertical motion.
That representation forces a small MLP to discover both a subtraction and a
per-task rotation from sparse Home data.  V652 keeps the same action space and
SAC/HER update, but replaces the three raw goal coordinates at the acquisition
head with the normalized tool-to-goal error in that exact task frame.

The feature is current Markov state only.  It is not an action, waypoint,
route, expert label, behavior-cloning target, or future observation.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch

from .causal_smooth_relay_v646 import (
    CausalSmoothRelayConfigV646,
    sample_causal_smooth_relay_batch_v646,
)
from .goal_conditioned_her_sac_v43 import (
    OBSERVATION_DIM_V43,
    observation_with_goal_v43,
)
from .goal_conditioned_markov_her_sac_v614 import (
    GoalConditionedMarkovHerReplayV614,
)
from .phase_isolated_acquisition_v626 import (
    PhaseIsolatedAcquisitionConfigV626,
    acquisition_phase_gate_v626,
)
from .privileged_effect_state_v1 import (
    PRIVILEGED_EFFECT_STATE_DIM,
    privileged_effect_state_slices_v1,
)
from .relay_dual_goal_her_sac_v643 import (
    ACQUISITION_GOAL_DIM_V643,
    ACQUISITION_OBSERVATION_DIM_V643,
    RELAY_DUAL_GOAL_HER_SAC_FORMAT_V643,
    RelayDualGoalHerSACBundleV643,
    RelayDualGoalHerSACConfigV643,
    RelayDualGoalHerSACMetricsV643,
    RelayDualGoalPolicyV643,
    initialize_relay_dual_goal_her_sac_v643,
    relay_dual_goal_her_sac_update_v643,
)
from .taskframe_controller_state_v614 import (
    TASKFRAME_CONTROLLER_STATE_DIM_V614,
)


RELAY_TASKFRAME_ERROR_HER_SAC_FORMAT_V652 = "edgearm-v652-taskframe-relative-error-acquisition-her-sac-v1"
TASKFRAME_ERROR_FEATURE_FORMAT_V652 = "edgearm-v652-normalized-taskframe-tool-goal-error-v1"
TASKFRAME_ERROR_FEATURE_DIM_V652 = 3
TASKFRAME_ERROR_SCALE_V652 = np.asarray((0.25, 0.20, 0.15), dtype=np.float32)
TASKFRAME_ERROR_CLIP_V652 = 2.0
_SLICES_V652 = privileged_effect_state_slices_v1()


def taskframe_goal_error_features_v652(
    transport_observation: np.ndarray,
    acquisition_goal_xyz: np.ndarray,
) -> np.ndarray:
    """Encode tool-to-goal error in policy action coordinates."""

    observation = np.asarray(transport_observation, dtype=np.float32)
    goal = np.asarray(acquisition_goal_xyz, dtype=np.float32)
    count = len(observation)
    if (
        observation.shape != (count, OBSERVATION_DIM_V43)
        or goal.shape != (count, ACQUISITION_GOAL_DIM_V643)
        or not np.all(np.isfinite(observation))
        or not np.all(np.isfinite(goal))
    ):
        raise ValueError("V652 task-frame feature inputs are invalid")
    neutral = observation[:, :PRIVILEGED_EFFECT_STATE_DIM]
    object_goal = observation[:, PRIVILEGED_EFFECT_STATE_DIM:]
    block = neutral[:, _SLICES_V652["block_pose_xyz_quaternion_wxyz"]][:, :2]
    tool = neutral[:, _SLICES_V652["tool_pose_position_rotation"]][:, :3]
    direction = object_goal - block
    norm = np.linalg.norm(direction, axis=-1, keepdims=True)
    fallback = np.zeros_like(direction)
    fallback[:, 0] = 1.0
    forward = np.where(
        norm > 1.0e-7,
        direction / np.maximum(norm, 1.0e-7),
        fallback,
    )
    lateral = np.stack((-forward[:, 1], forward[:, 0]), axis=-1)
    error = goal - tool
    task_error = np.stack(
        (
            np.sum(error[:, :2] * forward, axis=-1),
            np.sum(error[:, :2] * lateral, axis=-1),
            error[:, 2],
        ),
        axis=-1,
    )
    result = np.clip(
        task_error / TASKFRAME_ERROR_SCALE_V652,
        -TASKFRAME_ERROR_CLIP_V652,
        TASKFRAME_ERROR_CLIP_V652,
    ).astype(np.float32)
    if result.shape != (count, TASKFRAME_ERROR_FEATURE_DIM_V652):
        raise RuntimeError("V652 task-frame feature dimension drifted")
    return result


def _taskframe_goal_error_features_torch_v652(
    transport_observation: torch.Tensor,
    acquisition_goal_xyz: torch.Tensor,
) -> torch.Tensor:
    count = transport_observation.shape[0]
    if transport_observation.shape != (count, OBSERVATION_DIM_V43) or acquisition_goal_xyz.shape != (
        count,
        ACQUISITION_GOAL_DIM_V643,
    ):
        raise ValueError("V652 torch task-frame feature inputs are invalid")
    neutral = transport_observation[:, :PRIVILEGED_EFFECT_STATE_DIM]
    object_goal = transport_observation[:, PRIVILEGED_EFFECT_STATE_DIM:]
    block_slice = _SLICES_V652["block_pose_xyz_quaternion_wxyz"]
    tool_slice = _SLICES_V652["tool_pose_position_rotation"]
    block = neutral[:, block_slice.start : block_slice.start + 2]
    tool = neutral[:, tool_slice.start : tool_slice.start + 3]
    direction = object_goal - block
    norm = torch.linalg.vector_norm(direction, dim=-1, keepdim=True)
    fallback = torch.zeros_like(direction)
    fallback[:, 0] = 1.0
    forward = torch.where(
        norm > 1.0e-7,
        direction / norm.clamp_min(1.0e-7),
        fallback,
    )
    lateral = torch.stack((-forward[:, 1], forward[:, 0]), dim=-1)
    error = acquisition_goal_xyz - tool
    task_error = torch.stack(
        (
            torch.sum(error[:, :2] * forward, dim=-1),
            torch.sum(error[:, :2] * lateral, dim=-1),
            error[:, 2],
        ),
        dim=-1,
    )
    scale = torch.as_tensor(
        TASKFRAME_ERROR_SCALE_V652,
        dtype=task_error.dtype,
        device=task_error.device,
    )
    return torch.clamp(
        task_error / scale,
        -TASKFRAME_ERROR_CLIP_V652,
        TASKFRAME_ERROR_CLIP_V652,
    )


def acquisition_observation_v652(
    neutral_state: np.ndarray,
    original_object_goal: np.ndarray,
    controller_state: np.ndarray,
    acquisition_goal_xyz: np.ndarray,
) -> np.ndarray:
    """Build V652 state without changing the network input dimension."""

    neutral = np.asarray(neutral_state, dtype=np.float32)
    object_goal = np.asarray(original_object_goal, dtype=np.float32)
    controller = np.asarray(controller_state, dtype=np.float32)
    acquisition_goal = np.asarray(acquisition_goal_xyz, dtype=np.float32)
    count = len(neutral)
    if (
        neutral.shape != (count, PRIVILEGED_EFFECT_STATE_DIM)
        or object_goal.shape != (count, 2)
        or controller.shape != (count, TASKFRAME_CONTROLLER_STATE_DIM_V614)
        or acquisition_goal.shape != (count, ACQUISITION_GOAL_DIM_V643)
    ):
        raise ValueError("V652 acquisition observation inputs are invalid")
    transport = observation_with_goal_v43(neutral, object_goal)
    relative_error = taskframe_goal_error_features_v652(transport, acquisition_goal)
    result = np.concatenate(
        (transport, controller, relative_error),
        axis=-1,
        dtype=np.float32,
    )
    if result.shape != (count, ACQUISITION_OBSERVATION_DIM_V643) or not np.all(np.isfinite(result)):
        raise RuntimeError("V652 acquisition observation is invalid")
    return result


class RelayTaskframeErrorPolicyV652(RelayDualGoalPolicyV643):
    """V643 option boundary with a task-frame relative acquisition state."""

    def sample(
        self,
        transport_observation: torch.Tensor,
        controller_state: torch.Tensor,
        acquisition_goal_xyz: torch.Tensor,
        *,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        count = transport_observation.shape[0]
        if (
            transport_observation.shape != (count, OBSERVATION_DIM_V43)
            or controller_state.shape != (count, TASKFRAME_CONTROLLER_STATE_DIM_V614)
            or acquisition_goal_xyz.shape != (count, ACQUISITION_GOAL_DIM_V643)
        ):
            raise ValueError("V652 policy input shapes changed")
        relative_error = _taskframe_goal_error_features_torch_v652(
            transport_observation,
            acquisition_goal_xyz,
        )
        acquisition_observation = torch.cat(
            (transport_observation, controller_state, relative_error),
            dim=-1,
        )
        transport_action, _ = self.transport_actor.sample(transport_observation, deterministic=deterministic)
        acquisition_action, acquisition_log_probability = self.acquisition_actor.sample(
            acquisition_observation, deterministic=deterministic
        )
        gate, distance, alignment, contact = acquisition_phase_gate_v626(
            transport_observation, config=self.phase_config
        )
        action = self.blend_action(transport_action, acquisition_action, gate)
        return action, {
            "gate": gate,
            "distance_m": distance,
            "alignment": alignment,
            "contact": contact,
            "transport_action": transport_action,
            "acquisition_action": acquisition_action,
            "acquisition_log_probability": acquisition_log_probability,
            "taskframe_goal_error_features_v652": relative_error,
        }


@dataclass
class RelayTaskframeErrorHerSACBundleV652:
    """External V652 identity around the unchanged audited V643 SAC kernel."""

    kernel: RelayDualGoalHerSACBundleV643
    format: str = RELAY_TASKFRAME_ERROR_HER_SAC_FORMAT_V652

    @property
    def policy(self) -> RelayTaskframeErrorPolicyV652:
        policy = self.kernel.policy
        if type(policy) is not RelayTaskframeErrorPolicyV652:
            raise RuntimeError("V652 kernel policy identity changed")
        return policy

    @property
    def acquisition_critic(self) -> torch.nn.Module:
        return self.kernel.acquisition_critic

    @property
    def target_acquisition_critic(self) -> torch.nn.Module:
        return self.kernel.target_acquisition_critic

    @property
    def frozen_feasibility(self) -> torch.nn.Module:
        return self.kernel.frozen_feasibility

    @property
    def actor_optimizer(self) -> torch.optim.Optimizer:
        return self.kernel.actor_optimizer

    @property
    def critic_optimizer(self) -> torch.optim.Optimizer:
        return self.kernel.critic_optimizer

    @property
    def config(self) -> RelayDualGoalHerSACConfigV643:
        return self.kernel.config

    @property
    def update_index(self) -> int:
        return self.kernel.update_index


def initialize_relay_taskframe_error_her_sac_v652(
    *,
    seed: int,
    device: str | torch.device,
    parent_v614_payload: dict[str, Any],
    phase_config: PhaseIsolatedAcquisitionConfigV626,
    config: RelayDualGoalHerSACConfigV643,
) -> tuple[RelayTaskframeErrorHerSACBundleV652, dict[str, Any]]:
    """Create fresh V652 acquisition networks and frozen V614 transport."""

    kernel, parent_audit = initialize_relay_dual_goal_her_sac_v643(
        seed=seed,
        device=device,
        parent_v614_payload=parent_v614_payload,
        phase_config=phase_config,
        config=config,
    )
    policy = RelayTaskframeErrorPolicyV652(
        hidden_dim=config.hidden_dim,
        phase_config=phase_config,
    ).to(device)
    policy.load_state_dict(kernel.policy.state_dict(), strict=True)
    for parameter in policy.transport_actor.parameters():
        parameter.requires_grad_(False)
    kernel.policy = policy
    kernel.actor_optimizer = torch.optim.Adam(
        policy.acquisition_actor.parameters(),
        lr=config.actor_learning_rate,
    )
    return RelayTaskframeErrorHerSACBundleV652(kernel), {
        **parent_audit,
        "format": "edgearm-v652-v614-taskframe-error-upgrade-v1",
        "algorithm_format": RELAY_TASKFRAME_ERROR_HER_SAC_FORMAT_V652,
        "acquisition_state_semantics": TASKFRAME_ERROR_FEATURE_FORMAT_V652,
        "acquisition_state_dimension_changed": False,
        "raw_world_acquisition_goal_at_head": False,
        "taskframe_relative_error_at_head": True,
        "acquisition_actor_initialized_fresh": True,
        "warm_v643_acquisition_actor_imported": False,
        "expert_action_used": False,
        "waypoint_or_path_used": False,
        "behavior_cloning_steps": 0,
        "production_admission": False,
    }


def sample_taskframe_error_causal_batch_v652(
    replay: GoalConditionedMarkovHerReplayV614,
    *,
    batch_size: int,
    seed: int,
    phase_config: PhaseIsolatedAcquisitionConfigV626,
    relay_config: RelayDualGoalHerSACConfigV643,
    causal_config: CausalSmoothRelayConfigV646,
    episode_mass_multiplier: np.ndarray | None = None,
    transition_mass_multiplier: np.ndarray | None = None,
    first_contact_prefix_learning_mask_v738: np.ndarray | None = None,
    future_goal_strategy: str = "uniform",
    true_goal_minimum_direction_cosine: float | None = None,
    true_goal_minimum_progress_m: float | None = None,
    post_contact_reacquisition_learning_v730: bool = False,
) -> dict[str, np.ndarray]:
    """Reuse V646 causal sampling and replace only acquisition state encoding."""

    batch = sample_causal_smooth_relay_batch_v646(
        replay,
        batch_size=batch_size,
        seed=seed,
        phase_config=phase_config,
        relay_config=relay_config,
        causal_config=causal_config,
        episode_mass_multiplier=episode_mass_multiplier,
        transition_mass_multiplier=transition_mass_multiplier,
        first_contact_prefix_learning_mask_v738=(
            first_contact_prefix_learning_mask_v738
        ),
        future_goal_strategy=future_goal_strategy,
        true_goal_minimum_direction_cosine=(true_goal_minimum_direction_cosine),
        true_goal_minimum_progress_m=true_goal_minimum_progress_m,
        post_contact_reacquisition_learning_v730=(
            post_contact_reacquisition_learning_v730
        ),
    )
    acquisition_goal = batch["acquisition_goal_xyz"]
    batch["acquisition_observation"] = np.concatenate(
        (
            batch["transport_observation"],
            batch["controller_state"],
            taskframe_goal_error_features_v652(batch["transport_observation"], acquisition_goal),
        ),
        axis=-1,
        dtype=np.float32,
    )
    batch["next_acquisition_observation"] = np.concatenate(
        (
            batch["next_transport_observation"],
            batch["next_controller_state"],
            taskframe_goal_error_features_v652(batch["next_transport_observation"], acquisition_goal),
        ),
        axis=-1,
        dtype=np.float32,
    )
    batch["taskframe_goal_error_features_v652"] = batch["acquisition_observation"][
        :, -TASKFRAME_ERROR_FEATURE_DIM_V652:
    ].copy()
    if batch["acquisition_observation"].shape != (batch_size, ACQUISITION_OBSERVATION_DIM_V643) or batch[
        "next_acquisition_observation"
    ].shape != (batch_size, ACQUISITION_OBSERVATION_DIM_V643):
        raise RuntimeError("V652 sampled observation dimension drifted")
    return batch


@dataclass(frozen=True)
class RelayTaskframeErrorHerSACMetricsV652:
    update_index: int
    critic_loss: float
    actor_loss: float
    mean_q_target: float
    mean_q_data: float
    predicted_policy_feasibility: float
    mean_acquisition_gate: float
    exact_home_source_fraction: float
    tool_goal_her_fraction: float
    original_goal_success_fraction: float
    hindsight_goal_success_fraction: float
    mean_acquisition_progress_m: float
    mean_learning_reward: float
    mean_transport_acquisition_action_delta_l2: float
    actor_updated: bool
    mean_taskframe_goal_error_l2: float
    format: str = RELAY_TASKFRAME_ERROR_HER_SAC_FORMAT_V652


def relay_taskframe_error_her_sac_update_v652(
    bundle: RelayTaskframeErrorHerSACBundleV652,
    batch: dict[str, np.ndarray],
) -> RelayTaskframeErrorHerSACMetricsV652:
    """Run the unchanged SAC optimizer on the corrected Markov features."""

    if bundle.format != RELAY_TASKFRAME_ERROR_HER_SAC_FORMAT_V652:
        raise ValueError("V652 bundle identity changed")
    if bundle.kernel.format != RELAY_DUAL_GOAL_HER_SAC_FORMAT_V643:
        raise ValueError("V652 audited V643 optimizer kernel changed")
    features = np.asarray(
        batch.get("taskframe_goal_error_features_v652"),
        dtype=np.float32,
    )
    if features.shape != (bundle.config.batch_size, TASKFRAME_ERROR_FEATURE_DIM_V652) or not np.all(
        np.isfinite(features)
    ):
        raise ValueError("V652 update lacks task-frame error evidence")
    base: RelayDualGoalHerSACMetricsV643 = relay_dual_goal_her_sac_update_v643(bundle.kernel, batch)
    values = asdict(base)
    values.pop("format")
    return RelayTaskframeErrorHerSACMetricsV652(
        **values,
        mean_taskframe_goal_error_l2=float(np.mean(np.linalg.norm(features, axis=-1))),
    )


__all__ = [
    "RELAY_TASKFRAME_ERROR_HER_SAC_FORMAT_V652",
    "TASKFRAME_ERROR_FEATURE_FORMAT_V652",
    "TASKFRAME_ERROR_FEATURE_DIM_V652",
    "RelayTaskframeErrorHerSACBundleV652",
    "RelayTaskframeErrorHerSACMetricsV652",
    "RelayTaskframeErrorPolicyV652",
    "acquisition_observation_v652",
    "initialize_relay_taskframe_error_her_sac_v652",
    "relay_taskframe_error_her_sac_update_v652",
    "sample_taskframe_error_causal_batch_v652",
    "taskframe_goal_error_features_v652",
]
