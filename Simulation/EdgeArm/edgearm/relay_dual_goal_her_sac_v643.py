"""Relay dual-goal SAC for Home-to-contact acquisition.

The transport teacher uses block XY as its achieved goal.  Before contact the
block is stationary, so object-goal HER cannot credit useful free-space tool
motion.  V643 keeps the verified transport actor frozen and learns a separate
acquisition option whose achieved goal is the controllable tool XYZ position.

The option receives no expert action, waypoint sequence, controller override,
or future simulator state at execution.  Future tool positions are used only
as hindsight goals during replay.  The original block-to-language-target goal
continues to define the task-frame action axes, therefore acquisition HER never
rotates the recorded action.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
import math
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .adaptive_start_curriculum_v622 import START_TIERS_V622
from .goal_conditioned_her_sac_v43 import (
    ACTION_DIM_V43,
    OBSERVATION_DIM_V43,
    GoalConditionedActorV43,
    observation_with_goal_v43,
)
from .goal_conditioned_markov_her_sac_v614 import (
    GOAL_CONDITIONED_MARKOV_HER_CHECKPOINT_FORMAT_V614,
    GOAL_CONDITIONED_MARKOV_HER_SAC_FORMAT_V614,
    ActionFeasibilityMarkovV614,
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
from .post_contact_reacquisition_replay_v730 import (
    PostContactReacquisitionReplayConfigV730,
    derive_post_contact_reacquisition_labels_v730,
)
from .taskframe_controller_state_v614 import (
    TASKFRAME_CONTROLLER_STATE_DIM_V614,
    TASKFRAME_CONTROLLER_STATE_SCHEMA_SHA256_V614,
)
from .se_rl_safeguard_projection_v720 import (
    squared_safeguard_intervention_penalty_v720,
)


RELAY_DUAL_GOAL_HER_SAC_FORMAT_V643 = "edgearm-v643-relay-dual-goal-tool-her-sac-v1"
RELAY_DUAL_GOAL_CHECKPOINT_FORMAT_V643 = "edgearm-v643-relay-dual-goal-tool-her-sac-checkpoint-v1"
RELAY_ACQUISITION_BATCH_FORMAT_V643 = "edgearm-v643-phase-balanced-tool-goal-her-batch-v1"
ACQUISITION_GOAL_DIM_V643 = 3
ACQUISITION_OBSERVATION_DIM_V643 = (
    OBSERVATION_DIM_V43 + TASKFRAME_CONTROLLER_STATE_DIM_V614 + ACQUISITION_GOAL_DIM_V643
)

_SLICES_V643 = privileged_effect_state_slices_v1()
_HOME_TIER_INDEX_V643 = next(tier.index for tier in START_TIERS_V622 if tier.code == "home")


@dataclass(frozen=True)
class RelayDualGoalHerSACConfigV643:
    """Independent acquisition SAC and phase-balanced replay contract."""

    hidden_dim: int = 256
    batch_size: int = 256
    gamma: float = 0.99
    target_tau: float = 0.005
    entropy_temperature: float = 0.06
    actor_learning_rate: float = 3.0e-4
    critic_learning_rate: float = 3.0e-4
    policy_update_period: int = 2
    future_tool_goal_probability: float = 0.50
    minimum_future_tool_displacement_m: float = 0.0010
    acquisition_success_distance_m: float = 0.010
    original_goal_transport_gate_threshold: float = 0.005
    minimum_replay_acquisition_gate: float = 0.01
    home_sampling_fraction: float = 0.55
    progress_scale_m: float = 0.0015
    maximum_progress_reward: float = 1.0
    success_bonus: float = 2.0
    valid_contact_bonus: float = 3.0
    action_penalty: float = 0.008
    projection_penalty: float = 0.10
    safeguard_projection_penalty_v720: bool = False
    infeasible_action_penalty: float = 0.60
    safety_penalty: float = 3.0
    actor_infeasibility_coefficient: float = 0.30
    critic_conservative_coefficient: float = 0.01
    positive_approach_priority: float = 5.0
    contact_credit_priority: float = 8.0
    maximum_gradient_norm: float = 10.0

    def validate(self) -> None:
        probabilities = (
            self.gamma,
            self.target_tau,
            self.future_tool_goal_probability,
            self.home_sampling_fraction,
        )
        if any(not math.isfinite(value) or not 0.0 < value <= 1.0 for value in probabilities):
            raise ValueError("V643 probability configuration is invalid")
        positive = (
            self.entropy_temperature,
            self.actor_learning_rate,
            self.critic_learning_rate,
            self.minimum_future_tool_displacement_m,
            self.acquisition_success_distance_m,
            self.progress_scale_m,
            self.maximum_progress_reward,
            self.success_bonus,
            self.valid_contact_bonus,
            self.maximum_gradient_norm,
        )
        if any(not math.isfinite(value) or value <= 0.0 for value in positive):
            raise ValueError("V643 positive configuration is invalid")
        nonnegative = (
            self.action_penalty,
            self.projection_penalty,
            self.infeasible_action_penalty,
            self.safety_penalty,
            self.actor_infeasibility_coefficient,
            self.critic_conservative_coefficient,
            self.positive_approach_priority,
            self.contact_credit_priority,
        )
        if any(not math.isfinite(value) or value < 0.0 for value in nonnegative):
            raise ValueError("V643 nonnegative configuration is invalid")
        if type(self.safeguard_projection_penalty_v720) is not bool:
            raise ValueError("V720 safeguard projection reward flag is invalid")
        if not (
            0.0 <= self.original_goal_transport_gate_threshold < self.minimum_replay_acquisition_gate <= 0.25
        ):
            raise ValueError("V643 acquisition gate thresholds are invalid")
        if not 0.002 <= self.acquisition_success_distance_m <= 0.030:
            raise ValueError("V643 acquisition success distance is invalid")
        for name in ("hidden_dim", "batch_size", "policy_update_period"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"V643 {name} must be a positive integer")


def tool_xyz_from_neutral_v643(neutral_state: np.ndarray) -> np.ndarray:
    state = np.asarray(neutral_state, dtype=np.float32)
    if state.ndim != 2 or state.shape[-1] != PRIVILEGED_EFFECT_STATE_DIM or not np.all(np.isfinite(state)):
        raise ValueError("V643 neutral state is invalid")
    tool = state[:, _SLICES_V643["tool_pose_position_rotation"]]
    return tool[:, :3].astype(np.float32, copy=True)


def precontact_goal_xyz_v643(
    neutral_state: np.ndarray,
    original_object_goal: np.ndarray,
    *,
    phase_config: PhaseIsolatedAcquisitionConfigV626,
) -> np.ndarray:
    """Return the task-aligned precontact point, never an action or path."""

    phase_config.validate()
    state = np.asarray(neutral_state, dtype=np.float32)
    goal = np.asarray(original_object_goal, dtype=np.float32)
    count = len(state)
    if (
        state.shape != (count, PRIVILEGED_EFFECT_STATE_DIM)
        or goal.shape != (count, 2)
        or not np.all(np.isfinite(state))
        or not np.all(np.isfinite(goal))
    ):
        raise ValueError("V643 precontact-goal inputs are invalid")
    block = state[:, _SLICES_V643["block_pose_xyz_quaternion_wxyz"]][:, :2]
    delta = goal - block
    norm = np.linalg.norm(delta, axis=-1, keepdims=True)
    fallback = np.zeros_like(delta)
    fallback[:, 0] = 1.0
    forward = np.where(norm > 1.0e-7, delta / np.maximum(norm, 1.0e-7), fallback)
    result = np.concatenate(
        (
            block - np.float32(phase_config.precontact_standoff_m) * forward,
            np.full(
                (count, 1),
                phase_config.precontact_tool_height_m,
                dtype=np.float32,
            ),
        ),
        axis=-1,
    )
    return result.astype(np.float32)


def acquisition_observation_v643(
    neutral_state: np.ndarray,
    original_object_goal: np.ndarray,
    controller_state: np.ndarray,
    acquisition_goal_xyz: np.ndarray,
) -> np.ndarray:
    """Keep the language-selected object goal and add a controllable tool goal."""

    state = np.asarray(neutral_state, dtype=np.float32)
    object_goal = np.asarray(original_object_goal, dtype=np.float32)
    controller = np.asarray(controller_state, dtype=np.float32)
    acquisition_goal = np.asarray(acquisition_goal_xyz, dtype=np.float32)
    count = len(state)
    if (
        state.shape != (count, PRIVILEGED_EFFECT_STATE_DIM)
        or object_goal.shape != (count, 2)
        or controller.shape != (count, TASKFRAME_CONTROLLER_STATE_DIM_V614)
        or acquisition_goal.shape != (count, ACQUISITION_GOAL_DIM_V643)
        or not all(np.all(np.isfinite(value)) for value in (state, object_goal, controller, acquisition_goal))
    ):
        raise ValueError("V643 acquisition observation inputs are invalid")
    transport_observation = observation_with_goal_v43(state, object_goal)
    result = np.concatenate(
        (transport_observation, controller, acquisition_goal),
        axis=-1,
        dtype=np.float32,
    )
    if result.shape != (count, ACQUISITION_OBSERVATION_DIM_V643):
        raise RuntimeError("V643 acquisition observation dimension drifted")
    return result


def acquisition_gate_numpy_v643(
    neutral_state: np.ndarray,
    original_object_goal: np.ndarray,
    *,
    phase_config: PhaseIsolatedAcquisitionConfigV626,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    observation = observation_with_goal_v43(
        np.asarray(neutral_state, dtype=np.float32),
        np.asarray(original_object_goal, dtype=np.float32),
    )
    with torch.no_grad():
        gate, distance, alignment, contact = acquisition_phase_gate_v626(
            torch.from_numpy(observation), config=phase_config
        )
    return (
        gate.numpy().astype(np.float32),
        distance.numpy().astype(np.float32),
        alignment.numpy().astype(np.float32),
        contact.numpy().astype(bool),
    )


class RelayAcquisitionActorV643(nn.Module):
    def __init__(self, hidden_dim: int = 256) -> None:
        super().__init__()
        self.trunk = nn.Sequential(
            nn.LayerNorm(ACQUISITION_OBSERVATION_DIM_V643),
            nn.Linear(ACQUISITION_OBSERVATION_DIM_V643, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.mean = nn.Linear(hidden_dim, ACTION_DIM_V43)
        self.log_std = nn.Linear(hidden_dim, ACTION_DIM_V43)

    def distribution(self, observation: torch.Tensor) -> torch.distributions.Normal:
        if observation.ndim != 2 or observation.shape[-1] != ACQUISITION_OBSERVATION_DIM_V643:
            raise ValueError("V643 acquisition actor input shape changed")
        hidden = self.trunk(observation)
        return torch.distributions.Normal(
            self.mean(hidden), torch.clamp(self.log_std(hidden), -5.0, 1.0).exp()
        )

    def sample(
        self,
        observation: torch.Tensor,
        *,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        distribution = self.distribution(observation)
        pre_tanh = distribution.mean if deterministic else distribution.rsample()
        action = torch.tanh(pre_tanh)
        if deterministic:
            log_probability = torch.zeros(action.shape[0], device=action.device, dtype=action.dtype)
        else:
            correction = torch.log(1.0 - action.square() + 1.0e-6)
            log_probability = (distribution.log_prob(pre_tanh) - correction).sum(dim=-1)
        return action, log_probability


class _RelayAcquisitionActionValueV643(nn.Module):
    def __init__(self, hidden_dim: int = 256) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.LayerNorm(ACQUISITION_OBSERVATION_DIM_V643 + ACTION_DIM_V43),
            nn.Linear(
                ACQUISITION_OBSERVATION_DIM_V643 + ACTION_DIM_V43,
                hidden_dim,
            ),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, observation: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        if (
            observation.ndim != 2
            or observation.shape[-1] != ACQUISITION_OBSERVATION_DIM_V643
            or action.shape != (observation.shape[0], ACTION_DIM_V43)
        ):
            raise ValueError("V643 acquisition critic input shape changed")
        return self.network(torch.cat((observation, action), dim=-1)).squeeze(-1)


class TwinRelayAcquisitionCriticV643(nn.Module):
    def __init__(self, hidden_dim: int = 256) -> None:
        super().__init__()
        self.q1 = _RelayAcquisitionActionValueV643(hidden_dim)
        self.q2 = _RelayAcquisitionActionValueV643(hidden_dim)

    def forward(self, observation: torch.Tensor, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.q1(observation, action), self.q2(observation, action)


class RelayDualGoalPolicyV643(nn.Module):
    """Frozen transport actor plus a separately learned acquisition option."""

    def __init__(
        self,
        *,
        hidden_dim: int,
        phase_config: PhaseIsolatedAcquisitionConfigV626,
    ) -> None:
        super().__init__()
        phase_config.validate()
        self.transport_actor = GoalConditionedActorV43(hidden_dim)
        self.acquisition_actor = RelayAcquisitionActorV643(hidden_dim)
        self.phase_config = phase_config

    @staticmethod
    def blend_action(
        transport_action: torch.Tensor,
        acquisition_action: torch.Tensor,
        gate: torch.Tensor,
    ) -> torch.Tensor:
        if (
            transport_action.shape != acquisition_action.shape
            or transport_action.ndim != 2
            or gate.shape != (transport_action.shape[0],)
            or not bool(torch.isfinite(gate).all().item())
            or bool(torch.any(gate < 0.0).item())
            or bool(torch.any(gate > 1.0).item())
        ):
            raise ValueError("V643 action blend inputs are invalid")
        blended = transport_action + gate.unsqueeze(-1) * (acquisition_action - transport_action)
        zero_gate = gate == 0.0
        blended = torch.where(zero_gate.unsqueeze(-1), transport_action, blended)
        if bool(torch.any(zero_gate).item()) and not torch.equal(
            blended[zero_gate], transport_action[zero_gate]
        ):
            raise RuntimeError("V643 changed the exact transport policy")
        return blended

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
            raise ValueError("V643 policy input shapes changed")
        acquisition_observation = torch.cat(
            (transport_observation, controller_state, acquisition_goal_xyz),
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
        }


@dataclass
class RelayDualGoalHerSACBundleV643:
    policy: RelayDualGoalPolicyV643
    acquisition_critic: TwinRelayAcquisitionCriticV643
    target_acquisition_critic: TwinRelayAcquisitionCriticV643
    frozen_feasibility: ActionFeasibilityMarkovV614
    actor_optimizer: torch.optim.Optimizer
    critic_optimizer: torch.optim.Optimizer
    config: RelayDualGoalHerSACConfigV643
    update_index: int = 0
    format: str = RELAY_DUAL_GOAL_HER_SAC_FORMAT_V643


def initialize_relay_dual_goal_her_sac_v643(
    *,
    seed: int,
    device: str | torch.device,
    parent_v614_payload: dict[str, Any],
    phase_config: PhaseIsolatedAcquisitionConfigV626,
    config: RelayDualGoalHerSACConfigV643 | None = None,
) -> tuple[RelayDualGoalHerSACBundleV643, dict[str, Any]]:
    """Load only verified transport behavior; initialize acquisition fresh."""

    if type(seed) is not int or seed < 0:
        raise ValueError("V643 initialization seed must be non-negative")
    selected = config or RelayDualGoalHerSACConfigV643()
    selected.validate()
    phase_config.validate()
    payload = parent_v614_payload
    if (
        type(payload) is not dict
        or payload.get("format") != GOAL_CONDITIONED_MARKOV_HER_CHECKPOINT_FORMAT_V614
        or payload.get("algorithm_format") != GOAL_CONDITIONED_MARKOV_HER_SAC_FORMAT_V614
        or payload.get("controller_state_schema_sha256") != TASKFRAME_CONTROLLER_STATE_SCHEMA_SHA256_V614
        or int(payload.get("expert_calls", -1)) != 0
        or int(payload.get("behavior_cloning_steps", -1)) != 0
        or payload.get("production_admission") is not False
    ):
        raise ValueError("V643 parent V614 checkpoint identity is invalid")
    parent_config = payload.get("config")
    if type(parent_config) is not dict or int(parent_config.get("hidden_dim", -1)) != selected.hidden_dim:
        raise ValueError("V643 parent hidden dimension changed")
    stored_phase = payload.get("phase_isolated_acquisition_v626")
    if type(stored_phase) is not dict:
        raise ValueError("V643 parent lacks the audited acquisition gate")
    if PhaseIsolatedAcquisitionConfigV626(**stored_phase) != phase_config:
        raise ValueError("V643 parent and relay phase gates disagree")

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        policy = RelayDualGoalPolicyV643(hidden_dim=selected.hidden_dim, phase_config=phase_config)
        critic = TwinRelayAcquisitionCriticV643(selected.hidden_dim)
        target = TwinRelayAcquisitionCriticV643(selected.hidden_dim)
        feasibility = ActionFeasibilityMarkovV614(selected.hidden_dim)

    actor_state = payload.get("actor_state_dict")
    if not isinstance(actor_state, Mapping):
        raise ValueError("V643 parent actor state is missing")
    transport_state = {
        key[len("base.") :]: value for key, value in actor_state.items() if key.startswith("base.")
    }
    if not transport_state:
        raise ValueError("V643 parent actor lacks frozen transport weights")
    policy.transport_actor.load_state_dict(transport_state, strict=True)
    feasibility.load_state_dict(payload["feasibility_state_dict"], strict=True)
    target.load_state_dict(critic.state_dict(), strict=True)

    for module in (policy.transport_actor, feasibility, target):
        for parameter in module.parameters():
            parameter.requires_grad_(False)

    policy = policy.to(device)
    critic = critic.to(device)
    target = target.to(device)
    feasibility = feasibility.to(device)
    bundle = RelayDualGoalHerSACBundleV643(
        policy=policy,
        acquisition_critic=critic,
        target_acquisition_critic=target,
        frozen_feasibility=feasibility,
        actor_optimizer=torch.optim.Adam(
            policy.acquisition_actor.parameters(),
            lr=selected.actor_learning_rate,
        ),
        critic_optimizer=torch.optim.Adam(critic.parameters(), lr=selected.critic_learning_rate),
        config=selected,
    )
    return bundle, {
        "format": "edgearm-v643-v614-frozen-transport-upgrade-v1",
        "parent_algorithm_format": payload["algorithm_format"],
        "parent_update_index": int(payload["update_index"]),
        "transport_actor_copied": True,
        "transport_actor_frozen": True,
        "feasibility_model_copied": True,
        "feasibility_model_frozen": True,
        "acquisition_actor_initialized_fresh": True,
        "acquisition_critic_initialized_fresh": True,
        "old_v639_acquisition_residual_imported": False,
        "action_or_waypoint_supervision_used": False,
        "expert_calls": 0,
        "behavior_cloning_steps": 0,
        "production_admission": False,
    }


def _episode_end_rows_v643(episode_index: np.ndarray) -> np.ndarray:
    result = np.empty(len(episode_index), dtype=np.int64)
    for value in np.unique(episode_index):
        rows = np.flatnonzero(episode_index == value)
        result[rows] = rows[-1]
    return result


def _pool_probability_v643(
    rows: np.ndarray,
    weights: np.ndarray,
    episode_index: np.ndarray,
    episode_mass_multiplier: np.ndarray | None = None,
) -> np.ndarray:
    """Equal episode mass, then preserve causal priorities within an episode."""

    probability = np.zeros(len(weights), dtype=np.float64)
    episodes = np.unique(episode_index[rows])
    if episode_mass_multiplier is None:
        episode_mass = np.ones(len(episodes), dtype=np.float64)
    else:
        multiplier = np.asarray(episode_mass_multiplier, dtype=np.float64)
        if (
            multiplier.shape != weights.shape
            or not np.all(np.isfinite(multiplier))
            or np.any(multiplier <= 0.0)
        ):
            raise ValueError("V643 episode mass multiplier is invalid")
        episode_mass = np.empty(len(episodes), dtype=np.float64)
        for local_index, episode in enumerate(episodes):
            selected = rows[episode_index[rows] == episode]
            local = multiplier[selected]
            if not np.allclose(local, local[0], atol=1.0e-12, rtol=0.0):
                raise ValueError("V643 episode mass multiplier changed within an episode")
            episode_mass[local_index] = float(local[0])
    episode_mass /= float(np.sum(episode_mass))
    for local_index, episode in enumerate(episodes):
        selected = rows[episode_index[rows] == episode]
        local = weights[selected]
        probability[selected] = episode_mass[local_index] * local / float(np.sum(local))
    return probability


def _joint_mode_home_quota_v730(
    *,
    batch_size: int,
    reacquisition_fraction: float,
    home_fraction: float,
    pool_available: tuple[bool, bool, bool, bool],
) -> tuple[int, int, int, int] | None:
    """Find integer RH/RO/SH/SO counts preserving both target marginals."""

    if (
        type(batch_size) is not int
        or batch_size < 2
        or not np.isfinite(reacquisition_fraction)
        or not 0.0 < reacquisition_fraction < 1.0
        or not np.isfinite(home_fraction)
        or not 0.0 < home_fraction <= 1.0
        or len(pool_available) != 4
        or any(type(value) is not bool for value in pool_available)
    ):
        raise ValueError("V730 joint sampling quota is invalid")
    reacquisition_count = int(
        np.clip(round(batch_size * reacquisition_fraction), 1, batch_size - 1)
    )
    home_count = int(
        np.clip(round(batch_size * home_fraction), 1, batch_size - 1)
    )
    target_intersection = (
        reacquisition_count * home_count / batch_size
    )
    candidates: list[tuple[float, tuple[int, int, int, int]]] = []
    for reacquisition_home in range(
        max(0, reacquisition_count + home_count - batch_size),
        min(reacquisition_count, home_count) + 1,
    ):
        counts = (
            reacquisition_home,
            reacquisition_count - reacquisition_home,
            home_count - reacquisition_home,
            batch_size
            - reacquisition_count
            - home_count
            + reacquisition_home,
        )
        if any(
            count > 0 and not available
            for count, available in zip(
                counts,
                pool_available,
                strict=True,
            )
        ):
            continue
        candidates.append(
            (abs(reacquisition_home - target_intersection), counts)
        )
    if not candidates:
        return None
    return min(candidates, key=lambda item: item[0])[1]


def sample_relay_acquisition_batch_v643(
    replay: GoalConditionedMarkovHerReplayV614,
    *,
    batch_size: int,
    seed: int,
    phase_config: PhaseIsolatedAcquisitionConfigV626,
    config: RelayDualGoalHerSACConfigV643,
    episode_mass_multiplier: np.ndarray | None = None,
    transition_mass_multiplier: np.ndarray | None = None,
    first_contact_prefix_learning_mask_v738: np.ndarray | None = None,
    future_goal_strategy: str = "uniform",
    true_goal_minimum_direction_cosine: float | None = None,
    true_goal_minimum_progress_m: float | None = None,
    post_contact_reacquisition_learning_v730: bool = False,
) -> dict[str, np.ndarray]:
    """Sample Home-balanced acquisition rows and apply tool-position HER."""

    config.validate()
    phase_config.validate()
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("V643 replay batch size must be positive")
    if type(seed) is not int or seed < 0:
        raise ValueError("V643 replay seed must be non-negative")
    if type(post_contact_reacquisition_learning_v730) is not bool:
        raise ValueError("V730 replay learning flag must be an exact boolean")
    if replay.transition_count < 1:
        raise RuntimeError("V643 replay is empty")
    if future_goal_strategy not in {
        "uniform",
        "contact_frontier_v657",
        "true_goal_consistent_v658",
    }:
        raise ValueError("V643 future-goal strategy is invalid")
    if future_goal_strategy == "true_goal_consistent_v658":
        if (
            true_goal_minimum_direction_cosine is None
            or not np.isfinite(true_goal_minimum_direction_cosine)
            or not 0.0 < true_goal_minimum_direction_cosine < 1.0
            or true_goal_minimum_progress_m is None
            or not np.isfinite(true_goal_minimum_progress_m)
            or true_goal_minimum_progress_m <= 0.0
        ):
            raise ValueError("V658 true-goal hindsight thresholds are invalid")
    if replay.start_tier_index_v622.shape != (replay.transition_count,) or np.any(
        replay.start_tier_index_v622 < 0
    ):
        raise ValueError("V643 requires complete V622 start-tier labels")

    arrays = replay.arrays
    gate, distance, alignment, contact_before = acquisition_gate_numpy_v643(
        arrays["neutral_state"],
        arrays["desired_goal"],
        phase_config=phase_config,
    )
    next_gate, next_distance, next_alignment, next_contact_state = acquisition_gate_numpy_v643(
        arrays["next_neutral_state"],
        arrays["desired_goal"],
        phase_config=phase_config,
    )
    geometric_gate_v730 = gate.copy()
    geometric_next_gate_v730 = next_gate.copy()
    reacquisition_mode_v730 = np.zeros(
        replay.transition_count,
        dtype=bool,
    )
    next_reacquisition_mode_v730 = np.zeros_like(
        reacquisition_mode_v730
    )
    replay_config_v730 = PostContactReacquisitionReplayConfigV730()
    if post_contact_reacquisition_learning_v730:
        labels_v730 = derive_post_contact_reacquisition_labels_v730(
            episode_index=arrays["episode_index"],
            episode_step=arrays["episode_step"],
            instantaneous_contact=contact_before,
            valid_contact=arrays["valid_contact"],
            config=replay_config_v730,
        )
        reacquisition_mode_v730 = labels_v730["current_mode"]
        next_reacquisition_mode_v730 = labels_v730["next_mode"]
        gate = np.where(
            reacquisition_mode_v730,
            np.float32(replay_config_v730.effective_acquisition_gate),
            gate,
        ).astype(np.float32)
        next_gate = np.where(
            next_reacquisition_mode_v730,
            np.float32(replay_config_v730.effective_acquisition_gate),
            next_gate,
        ).astype(np.float32)
    first_contact_prefix_v738 = np.zeros(
        replay.transition_count,
        dtype=bool,
    )
    next_first_contact_prefix_v738 = np.zeros_like(
        first_contact_prefix_v738
    )
    if first_contact_prefix_learning_mask_v738 is not None:
        first_contact_prefix_v738 = np.asarray(
            first_contact_prefix_learning_mask_v738
        )
        if (
            first_contact_prefix_v738.shape
            != (replay.transition_count,)
            or first_contact_prefix_v738.dtype != np.dtype(bool)
            or np.any(first_contact_prefix_v738 & contact_before)
            or np.any(
                first_contact_prefix_v738 & reacquisition_mode_v730
            )
        ):
            raise ValueError("V738 first-contact prefix mask is invalid")
        for episode_v738 in np.unique(arrays["episode_index"]):
            rows_v738 = np.flatnonzero(
                arrays["episode_index"] == episode_v738
            )
            if len(rows_v738) > 1:
                next_first_contact_prefix_v738[rows_v738[:-1]] = (
                    first_contact_prefix_v738[rows_v738[1:]]
                )
        gate = np.where(
            first_contact_prefix_v738,
            np.float32(1.0),
            gate,
        ).astype(np.float32)
        next_gate = np.where(
            next_first_contact_prefix_v738,
            np.float32(1.0),
            next_gate,
        ).astype(np.float32)
    eligible = (
        gate >= np.float32(config.minimum_replay_acquisition_gate)
    ) & (~contact_before | reacquisition_mode_v730)
    if config.safeguard_projection_penalty_v720:
        projection_valid = getattr(
            replay,
            "safeguard_projection_valid_v720",
            None,
        )
        if (
            not isinstance(projection_valid, np.ndarray)
            or projection_valid.shape != (replay.transition_count,)
            or projection_valid.dtype != np.dtype(bool)
        ):
            raise RuntimeError(
                "V720 acquisition replay lacks projection validity"
            )
        eligible &= projection_valid
    eligible_rows = np.flatnonzero(eligible)
    if not len(eligible_rows):
        raise RuntimeError("V643 replay has no acquisition-phase transitions")

    progress = (distance - next_distance).astype(np.float32)
    profiles = replay._contact_learning_profiles()
    weights = np.ones(replay.transition_count, dtype=np.float64)
    weights += config.positive_approach_priority * (
        progress >= np.float32(0.05 * config.progress_scale_m)
    ).astype(np.float64)
    weights += config.contact_credit_priority * profiles["contact_credit_source"].astype(np.float64)
    weights += config.valid_contact_bonus * arrays["valid_contact"].astype(np.float64)
    source_transition_mass = np.ones(
        replay.transition_count,
        dtype=np.float64,
    )
    if transition_mass_multiplier is not None:
        source_transition_mass = np.asarray(
            transition_mass_multiplier,
            dtype=np.float64,
        )
        if (
            source_transition_mass.shape != (replay.transition_count,)
            or not np.all(np.isfinite(source_transition_mass))
            or np.any(source_transition_mass <= 0.0)
        ):
            raise ValueError("V643 transition mass multiplier is invalid")
        weights *= source_transition_mass
        if not np.all(np.isfinite(weights)) or np.any(weights <= 0.0):
            raise RuntimeError("V643 weighted transition mass is invalid")
    if post_contact_reacquisition_learning_v730:
        weights += (
            replay_config_v730.replay_priority_increment
            * reacquisition_mode_v730.astype(np.float64)
        )

    home_mask = replay.start_tier_index_v622 == _HOME_TIER_INDEX_V643
    home_rows = np.flatnonzero(eligible & home_mask)
    other_rows = np.flatnonzero(eligible & ~home_mask)
    rng = np.random.default_rng(seed)
    reacquisition_rows_v730 = np.flatnonzero(
        eligible & reacquisition_mode_v730
    )
    standard_rows_v730 = np.flatnonzero(
        eligible & ~reacquisition_mode_v730
    )
    joint_pools_v730 = (
        np.flatnonzero(
            eligible & reacquisition_mode_v730 & home_mask
        ),
        np.flatnonzero(
            eligible & reacquisition_mode_v730 & ~home_mask
        ),
        np.flatnonzero(
            eligible & ~reacquisition_mode_v730 & home_mask
        ),
        np.flatnonzero(
            eligible & ~reacquisition_mode_v730 & ~home_mask
        ),
    )
    joint_counts_v730 = (
        _joint_mode_home_quota_v730(
            batch_size=batch_size,
            reacquisition_fraction=(
                replay_config_v730.reacquisition_sampling_fraction
            ),
            home_fraction=config.home_sampling_fraction,
            pool_available=tuple(
                bool(len(pool)) for pool in joint_pools_v730
            ),
        )
        if post_contact_reacquisition_learning_v730
        else None
    )

    def sample_pool_v730(
        rows_v730: np.ndarray,
        count_v730: int,
        mass_v730: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        probability_v730 = _pool_probability_v643(
            rows_v730,
            weights,
            arrays["episode_index"],
            episode_mass_multiplier,
        )
        sampled_v730 = rng.choice(
            rows_v730,
            size=count_v730,
            replace=len(rows_v730) < count_v730,
            p=probability_v730[rows_v730],
        ).astype(np.int64)
        return (
            sampled_v730,
            mass_v730 * probability_v730[sampled_v730],
        )

    if (
        post_contact_reacquisition_learning_v730
        and len(reacquisition_rows_v730)
        and len(standard_rows_v730)
        and joint_counts_v730 is not None
    ):
        sampled_parts_v730: list[np.ndarray] = []
        probability_parts_v730: list[np.ndarray] = []
        for pool_v730, count_v730 in zip(
            joint_pools_v730,
            joint_counts_v730,
            strict=True,
        ):
            if count_v730 == 0:
                continue
            sampled_v730, probability_v730 = sample_pool_v730(
                pool_v730,
                count_v730,
                count_v730 / batch_size,
            )
            sampled_parts_v730.append(sampled_v730)
            probability_parts_v730.append(probability_v730)
        roots = np.concatenate(sampled_parts_v730)
        sampled_probability = np.concatenate(probability_parts_v730)
        permutation = rng.permutation(batch_size)
        roots = roots[permutation]
        sampled_probability = sampled_probability[permutation]
    elif (
        post_contact_reacquisition_learning_v730
        and len(reacquisition_rows_v730)
        and len(standard_rows_v730)
    ):
        reacquisition_count_v730 = int(
            np.clip(
                round(
                    batch_size
                    * replay_config_v730.reacquisition_sampling_fraction
                ),
                1,
                batch_size - 1,
            )
        )
        standard_count_v730 = batch_size - reacquisition_count_v730
        sampled_reacquisition_v730, probability_reacquisition_v730 = (
            sample_pool_v730(
                reacquisition_rows_v730,
                reacquisition_count_v730,
                reacquisition_count_v730 / batch_size,
            )
        )
        sampled_standard_v730, probability_standard_v730 = (
            sample_pool_v730(
                standard_rows_v730,
                standard_count_v730,
                standard_count_v730 / batch_size,
            )
        )
        roots = np.concatenate(
            (sampled_reacquisition_v730, sampled_standard_v730)
        )
        sampled_probability = np.concatenate(
            (probability_reacquisition_v730, probability_standard_v730)
        )
        permutation = rng.permutation(batch_size)
        roots = roots[permutation]
        sampled_probability = sampled_probability[permutation]
    elif len(home_rows) and len(other_rows):
        home_count = int(np.clip(round(batch_size * config.home_sampling_fraction), 1, batch_size - 1))
        other_count = batch_size - home_count
        home_probability = _pool_probability_v643(
            home_rows,
            weights,
            arrays["episode_index"],
            episode_mass_multiplier,
        )
        other_probability = _pool_probability_v643(
            other_rows,
            weights,
            arrays["episode_index"],
            episode_mass_multiplier,
        )
        sampled_home = rng.choice(
            home_rows,
            size=home_count,
            replace=len(home_rows) < home_count,
            p=home_probability[home_rows],
        ).astype(np.int64)
        sampled_other = rng.choice(
            other_rows,
            size=other_count,
            replace=len(other_rows) < other_count,
            p=other_probability[other_rows],
        ).astype(np.int64)
        roots = np.concatenate((sampled_home, sampled_other))
        sampled_probability = np.concatenate(
            (
                config.home_sampling_fraction * home_probability[sampled_home],
                (1.0 - config.home_sampling_fraction) * other_probability[sampled_other],
            )
        )
        permutation = rng.permutation(batch_size)
        roots = roots[permutation]
        sampled_probability = sampled_probability[permutation]
    else:
        pool = home_rows if len(home_rows) else other_rows
        probability = _pool_probability_v643(
            pool,
            weights,
            arrays["episode_index"],
            episode_mass_multiplier,
        )
        roots = rng.choice(
            pool,
            size=batch_size,
            replace=len(pool) < batch_size,
            p=probability[pool],
        ).astype(np.int64)
        sampled_probability = probability[roots]

    original_object_goal = arrays["desired_goal"][roots].copy()
    acquisition_goal = precontact_goal_xyz_v643(
        arrays["neutral_state"][roots],
        original_object_goal,
        phase_config=phase_config,
    )
    original_acquisition_goal = acquisition_goal.copy()
    relabelled = np.zeros(batch_size, dtype=bool)
    future_row_selected = np.full(batch_size, -1, dtype=np.int64)
    contact_frontier_selected = np.zeros(batch_size, dtype=bool)
    contact_goal_from_contact = np.zeros(batch_size, dtype=bool)
    future_tool_goal_displacement = np.zeros(batch_size, dtype=np.float32)
    future_tool_goal_target_gap = np.zeros(batch_size, dtype=np.float32)
    true_goal_attempted = np.zeros(batch_size, dtype=bool)
    true_goal_selected = np.zeros(batch_size, dtype=bool)
    true_goal_from_contact = np.zeros(batch_size, dtype=bool)
    true_goal_direction_cosine = np.zeros(batch_size, dtype=np.float32)
    true_goal_real_progress = np.zeros(batch_size, dtype=np.float32)
    true_goal_candidate_count = np.zeros(batch_size, dtype=np.int32)
    true_goal_rejected_candidate_count = np.zeros(batch_size, dtype=np.int32)
    tool = tool_xyz_from_neutral_v643(arrays["neutral_state"])
    next_tool = tool_xyz_from_neutral_v643(arrays["next_neutral_state"])
    episode_end = _episode_end_rows_v643(arrays["episode_index"])
    for batch_index, row in enumerate(roots):
        if rng.random() >= config.future_tool_goal_probability:
            continue
        if future_goal_strategy == "true_goal_consistent_v658":
            true_goal_attempted[batch_index] = True
        final = int(episode_end[row])
        candidates = np.arange(row, final + 1, dtype=np.int64)
        if future_goal_strategy == "uniform":
            candidates = candidates[eligible[candidates]]
        else:
            candidates = candidates[~contact_before[candidates] | arrays["valid_contact"][candidates]]
        if not len(candidates):
            continue
        if future_goal_strategy == "uniform":
            future_row = int(candidates[int(rng.integers(0, len(candidates)))])
            future_goal = next_tool[future_row]
            displacement = float(np.linalg.norm(future_goal - tool[row]))
            if displacement < config.minimum_future_tool_displacement_m:
                continue
        else:
            candidate_goal = next_tool[candidates]
            displacement_by_candidate = np.linalg.norm(candidate_goal - tool[row], axis=-1)
            displaced = displacement_by_candidate >= config.minimum_future_tool_displacement_m
            candidates = candidates[displaced]
            candidate_goal = candidate_goal[displaced]
            displacement_by_candidate = displacement_by_candidate[displaced]
            if not len(candidates):
                continue
            if future_goal_strategy == "true_goal_consistent_v658":
                original_direction = original_acquisition_goal[batch_index] - tool[row]
                original_distance = float(np.linalg.norm(original_direction))
                candidate_direction = candidate_goal - tool[row]
                candidate_direction_norm = np.linalg.norm(candidate_direction, axis=-1)
                direction_cosine_by_candidate = (candidate_direction @ original_direction) / np.maximum(
                    candidate_direction_norm * original_distance,
                    np.float32(1.0e-12),
                )
                real_progress_by_candidate = original_distance - np.linalg.norm(
                    candidate_goal - original_acquisition_goal[batch_index],
                    axis=-1,
                )
                consistent = (
                    direction_cosine_by_candidate >= np.float32(true_goal_minimum_direction_cosine)
                ) & (real_progress_by_candidate >= np.float32(true_goal_minimum_progress_m))
                true_goal_candidate_count[batch_index] = len(candidates)
                true_goal_rejected_candidate_count[batch_index] = int(np.count_nonzero(~consistent))
                candidates = candidates[consistent]
                candidate_goal = candidate_goal[consistent]
                displacement_by_candidate = displacement_by_candidate[consistent]
                direction_cosine_by_candidate = direction_cosine_by_candidate[consistent]
                real_progress_by_candidate = real_progress_by_candidate[consistent]
                if not len(candidates):
                    continue
            contact_candidates = np.flatnonzero(arrays["valid_contact"][candidates])
            if len(contact_candidates):
                local_future_index = int(contact_candidates[0])
                if future_goal_strategy == "contact_frontier_v657":
                    contact_goal_from_contact[batch_index] = True
                else:
                    true_goal_from_contact[batch_index] = True
            else:
                target_gap_by_candidate = np.linalg.norm(
                    candidate_goal - original_acquisition_goal[batch_index],
                    axis=-1,
                )
                local_future_index = int(np.argmin(target_gap_by_candidate))
            future_row = int(candidates[local_future_index])
            future_goal = candidate_goal[local_future_index]
            displacement = float(displacement_by_candidate[local_future_index])
            if future_goal_strategy == "contact_frontier_v657":
                contact_frontier_selected[batch_index] = True
                future_tool_goal_displacement[batch_index] = displacement
                future_tool_goal_target_gap[batch_index] = float(
                    np.linalg.norm(future_goal - original_acquisition_goal[batch_index])
                )
            else:
                true_goal_selected[batch_index] = True
                true_goal_direction_cosine[batch_index] = float(
                    direction_cosine_by_candidate[local_future_index]
                )
                true_goal_real_progress[batch_index] = float(real_progress_by_candidate[local_future_index])
        acquisition_goal[batch_index] = future_goal
        relabelled[batch_index] = True
        future_row_selected[batch_index] = future_row

    achieved = tool[roots]
    next_achieved = next_tool[roots]
    before_distance = np.linalg.norm(achieved - acquisition_goal, axis=-1)
    after_distance = np.linalg.norm(next_achieved - acquisition_goal, axis=-1)
    acquisition_progress = before_distance - after_distance
    reward = np.clip(
        acquisition_progress / config.progress_scale_m,
        -config.maximum_progress_reward,
        config.maximum_progress_reward,
    ).astype(np.float32)
    original_success = ~relabelled & (
        arrays["valid_contact"][roots]
        | (next_gate[roots] <= np.float32(config.original_goal_transport_gate_threshold))
    )
    hindsight_success = relabelled & (after_distance <= config.acquisition_success_distance_m)
    success = original_success | hindsight_success
    reward += config.success_bonus * success.astype(np.float32)
    reward += config.valid_contact_bonus * (~relabelled & arrays["valid_contact"][roots]).astype(np.float32)
    action = arrays["action"][roots].copy().astype(np.float32)
    applied_action = arrays["applied_action"][roots].copy().astype(np.float32)
    actual_effect_projection_l2_v720 = np.linalg.norm(
        action - applied_action,
        axis=-1,
    ).astype(np.float32)
    safeguard_projected_action_v720 = applied_action.copy()
    safeguard_projection_valid_v720 = np.zeros(batch_size, dtype=bool)
    if config.safeguard_projection_penalty_v720:
        projected_source = getattr(
            replay,
            "safeguard_projected_action_v720",
            None,
        )
        projection_valid_source = getattr(
            replay,
            "safeguard_projection_valid_v720",
            None,
        )
        if (
            not isinstance(projected_source, np.ndarray)
            or projected_source.shape != (replay.transition_count, 3)
            or not np.all(np.isfinite(projected_source))
            or not isinstance(projection_valid_source, np.ndarray)
            or projection_valid_source.shape != (replay.transition_count,)
            or projection_valid_source.dtype != np.dtype(bool)
        ):
            raise RuntimeError(
                "V720 acquisition replay lacks valid safeguard projections"
            )
        safeguard_projected_action_v720 = projected_source[roots].copy()
        safeguard_projection_valid_v720 = projection_valid_source[roots].copy()
        (
            projection_penalty_component_v720,
            safeguard_projection_l2_v720,
        ) = squared_safeguard_intervention_penalty_v720(
            action,
            safeguard_projected_action_v720,
            safeguard_projection_valid_v720,
            coefficient=config.projection_penalty,
        )
        projection_l2 = np.where(
            safeguard_projection_valid_v720,
            safeguard_projection_l2_v720,
            np.float32(0.0),
        ).astype(np.float32)
    else:
        projection_l2 = actual_effect_projection_l2_v720
        projection_penalty_component_v720 = (
            np.float32(config.projection_penalty)
            * np.square(projection_l2, dtype=np.float32)
        )
    reward -= config.action_penalty * np.sum(np.square(action), axis=-1)
    reward -= projection_penalty_component_v720
    reward -= config.infeasible_action_penalty * (~arrays["action_feasible"][roots]).astype(np.float32)
    reward -= config.safety_penalty * arrays["safety_violation"][roots].astype(np.float32)
    done = arrays["terminal"][roots] | arrays["failure_terminal"][roots] | success
    importance = np.power(replay.transition_count * sampled_probability, -0.4)
    importance /= max(float(np.max(importance)), 1.0e-12)
    controller = replay.controller_state[roots]
    next_controller = replay.next_controller_state[roots]
    batch = {
        "acquisition_observation": acquisition_observation_v643(
            arrays["neutral_state"][roots],
            original_object_goal,
            controller,
            acquisition_goal,
        ),
        "next_acquisition_observation": acquisition_observation_v643(
            arrays["next_neutral_state"][roots],
            original_object_goal,
            next_controller,
            acquisition_goal,
        ),
        "transport_observation": observation_with_goal_v43(
            arrays["neutral_state"][roots], original_object_goal
        ),
        "next_transport_observation": observation_with_goal_v43(
            arrays["next_neutral_state"][roots], original_object_goal
        ),
        "controller_state": controller.copy(),
        "next_controller_state": next_controller.copy(),
        "action": action,
        "applied_action": applied_action,
        "projection_l2": projection_l2,
        "actual_effect_projection_l2_v720": (
            actual_effect_projection_l2_v720
        ),
        "safeguard_projected_action_v720": (
            safeguard_projected_action_v720.astype(np.float32)
        ),
        "safeguard_projection_valid_v720": (
            safeguard_projection_valid_v720
        ),
        "projection_penalty_component_v720": (
            projection_penalty_component_v720
        ),
        "safeguard_projection_penalty_active_v720": np.full(
            batch_size,
            config.safeguard_projection_penalty_v720,
            dtype=bool,
        ),
        "reward": reward.astype(np.float32),
        "done": done.astype(np.float32),
        "action_feasible": arrays["action_feasible"][roots].astype(np.float32),
        "safety_violation": arrays["safety_violation"][roots].astype(np.float32),
        "importance_weight": importance.astype(np.float32),
        "acquisition_gate": gate[roots].copy(),
        "next_acquisition_gate": next_gate[roots].copy(),
        "acquisition_distance_m": distance[roots].copy(),
        "next_acquisition_distance_m": next_distance[roots].copy(),
        "acquisition_alignment": alignment[roots].copy(),
        "next_acquisition_alignment": next_alignment[roots].copy(),
        "acquisition_goal_xyz": acquisition_goal.astype(np.float32),
        "tool_achieved_goal_xyz": achieved.astype(np.float32),
        "next_tool_achieved_goal_xyz": next_achieved.astype(np.float32),
        "acquisition_progress_m": acquisition_progress.astype(np.float32),
        "tool_goal_her_relabelled": relabelled,
        "original_goal_success": original_success,
        "hindsight_goal_success": hindsight_success,
        "source_row_index": roots,
        "source_episode_index": arrays["episode_index"][roots].copy(),
        "source_start_tier_index_v622": replay.start_tier_index_v622[roots].copy(),
        "exact_home_source": home_mask[roots].copy(),
        "future_source_row_index": future_row_selected,
        "contact_frontier_future_goal_selected_v657": (contact_frontier_selected),
        "contact_frontier_goal_from_contact_v657": (contact_goal_from_contact),
        "future_tool_goal_displacement_m_v657": (future_tool_goal_displacement),
        "future_tool_goal_target_gap_m_v657": future_tool_goal_target_gap,
        "true_goal_consistent_future_goal_attempted_v658": true_goal_attempted,
        "true_goal_consistent_future_goal_selected_v658": true_goal_selected,
        "true_goal_consistent_goal_from_contact_v658": true_goal_from_contact,
        "future_tool_goal_direction_cosine_v658": true_goal_direction_cosine,
        "future_tool_goal_real_progress_m_v658": true_goal_real_progress,
        "future_tool_goal_candidate_count_v658": true_goal_candidate_count,
        "future_tool_goal_rejected_candidate_count_v658": (true_goal_rejected_candidate_count),
        "recorded_action_rotated_for_tool_her": np.zeros(batch_size, dtype=bool),
    }
    if transition_mass_multiplier is not None:
        batch["source_transition_mass_multiplier"] = (
            source_transition_mass[roots].copy()
        )
        batch["source_sampling_probability_v738"] = (
            sampled_probability.copy().astype(np.float64)
        )
    if post_contact_reacquisition_learning_v730:
        batch.update(
            {
                "post_contact_reacquisition_mode_v730": (
                    reacquisition_mode_v730[roots].copy()
                ),
                "next_post_contact_reacquisition_mode_v730": (
                    next_reacquisition_mode_v730[roots].copy()
                ),
                "geometric_acquisition_gate_v730": (
                    geometric_gate_v730[roots].copy()
                ),
                "geometric_next_acquisition_gate_v730": (
                    geometric_next_gate_v730[roots].copy()
                ),
                "effective_gate_overridden_v730": (
                    reacquisition_mode_v730[roots].copy()
                ),
                "next_effective_gate_overridden_v730": (
                    next_reacquisition_mode_v730[roots].copy()
                ),
                "post_contact_reacquisition_learning_enabled_v730": (
                    np.ones(batch_size, dtype=bool)
                ),
                "v730_batch_reacquisition_quota_count": np.full(
                    batch_size,
                    np.count_nonzero(reacquisition_mode_v730[roots]),
                    dtype=np.int32,
                ),
                "v730_batch_home_quota_count": np.full(
                    batch_size,
                    np.count_nonzero(home_mask[roots]),
                    dtype=np.int32,
                ),
            }
        )
    if first_contact_prefix_learning_mask_v738 is not None:
        batch.update(
            {
                "first_contact_prefix_learning_mode_v738": (
                    first_contact_prefix_v738[roots].copy()
                ),
                "next_first_contact_prefix_learning_mode_v738": (
                    next_first_contact_prefix_v738[roots].copy()
                ),
                "geometric_acquisition_gate_v738": (
                    geometric_gate_v730[roots].copy()
                ),
                "geometric_next_acquisition_gate_v738": (
                    geometric_next_gate_v730[roots].copy()
                ),
                "effective_gate_overridden_v738": (
                    first_contact_prefix_v738[roots].copy()
                ),
                "next_effective_gate_overridden_v738": (
                    next_first_contact_prefix_v738[roots].copy()
                ),
            }
        )
    if not all(len(value) == batch_size for value in batch.values()):
        raise RuntimeError("V643 acquisition batch row counts disagree")
    return batch


@dataclass(frozen=True)
class RelayDualGoalHerSACMetricsV643:
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
    format: str = RELAY_DUAL_GOAL_HER_SAC_FORMAT_V643


def relay_dual_goal_her_sac_update_v643(
    bundle: RelayDualGoalHerSACBundleV643,
    batch: dict[str, np.ndarray],
) -> RelayDualGoalHerSACMetricsV643:
    config = bundle.config
    config.validate()
    if bundle.format != RELAY_DUAL_GOAL_HER_SAC_FORMAT_V643:
        raise ValueError("V643 bundle identity changed")
    device = next(bundle.policy.acquisition_actor.parameters()).device

    def tensor(name: str) -> torch.Tensor:
        if name not in batch:
            raise KeyError(f"V643 batch lacks {name}")
        return torch.from_numpy(np.asarray(batch[name])).to(device)

    observation = tensor("acquisition_observation")
    next_observation = tensor("next_acquisition_observation")
    transport_observation = tensor("transport_observation")
    next_transport_observation = tensor("next_transport_observation")
    controller = tensor("controller_state")
    next_controller = tensor("next_controller_state")
    action = tensor("action")
    reward = tensor("reward")
    done = tensor("done")
    importance = tensor("importance_weight")
    gate = tensor("acquisition_gate")
    next_gate = tensor("next_acquisition_gate")
    count = observation.shape[0]
    if (
        observation.shape != (count, ACQUISITION_OBSERVATION_DIM_V643)
        or next_observation.shape != observation.shape
        or transport_observation.shape != (count, OBSERVATION_DIM_V43)
        or next_transport_observation.shape != transport_observation.shape
        or controller.shape != (count, TASKFRAME_CONTROLLER_STATE_DIM_V614)
        or next_controller.shape != controller.shape
        or action.shape != (count, ACTION_DIM_V43)
        or gate.shape != (count,)
        or next_gate.shape != (count,)
    ):
        raise ValueError("V643 update batch shape changed")

    bundle.update_index += 1
    update_index = bundle.update_index
    with torch.no_grad():
        next_acquisition_action, next_log_probability = bundle.policy.acquisition_actor.sample(
            next_observation
        )
        next_transport_action, _ = bundle.policy.transport_actor.sample(
            next_transport_observation, deterministic=True
        )
        next_policy_action = bundle.policy.blend_action(
            next_transport_action, next_acquisition_action, next_gate
        )
        target_q1, target_q2 = bundle.target_acquisition_critic(next_observation, next_policy_action)
        target_value = torch.minimum(target_q1, target_q2) - (
            config.entropy_temperature * next_gate * next_log_probability
        )
        q_target = reward + config.gamma * (1.0 - done) * target_value

    q1, q2 = bundle.acquisition_critic(observation, action)
    td = F.smooth_l1_loss(q1, q_target, reduction="none") + F.smooth_l1_loss(q2, q_target, reduction="none")
    random_action = torch.empty_like(action).uniform_(-1.0, 1.0)
    random_q1, random_q2 = bundle.acquisition_critic(observation, random_action)
    conservative = (
        torch.logsumexp(torch.stack((random_q1, q1), dim=0), dim=0)
        - q1
        + torch.logsumexp(torch.stack((random_q2, q2), dim=0), dim=0)
        - q2
    ).mean()
    critic_loss = (importance * td).mean() + (config.critic_conservative_coefficient * conservative)
    bundle.critic_optimizer.zero_grad(set_to_none=True)
    critic_loss.backward()
    nn.utils.clip_grad_norm_(bundle.acquisition_critic.parameters(), config.maximum_gradient_norm)
    bundle.critic_optimizer.step()

    actor_updated = update_index % config.policy_update_period == 0
    actor_loss_value = 0.0
    predicted_feasibility_value = 0.0
    mean_action_delta = 0.0
    if actor_updated:
        for parameter in bundle.acquisition_critic.parameters():
            parameter.requires_grad_(False)
        acquisition_action, log_probability = bundle.policy.acquisition_actor.sample(observation)
        with torch.no_grad():
            transport_action, _ = bundle.policy.transport_actor.sample(
                transport_observation, deterministic=True
            )
        policy_action = bundle.policy.blend_action(transport_action, acquisition_action, gate)
        policy_q1, policy_q2 = bundle.acquisition_critic(observation, policy_action)
        policy_feasibility = torch.sigmoid(
            bundle.frozen_feasibility(transport_observation, controller, policy_action)
        )
        per_row_loss = (
            config.entropy_temperature * gate * log_probability
            - torch.minimum(policy_q1, policy_q2)
            + config.actor_infeasibility_coefficient * (1.0 - policy_feasibility)
        )
        actor_loss = (importance * per_row_loss).sum() / importance.sum().clamp_min(1.0e-8)
        bundle.actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        nn.utils.clip_grad_norm_(
            bundle.policy.acquisition_actor.parameters(),
            config.maximum_gradient_norm,
        )
        bundle.actor_optimizer.step()
        for parameter in bundle.acquisition_critic.parameters():
            parameter.requires_grad_(True)
        actor_loss_value = float(actor_loss.item())
        predicted_feasibility_value = float(policy_feasibility.mean().item())
        mean_action_delta = float((acquisition_action - transport_action).norm(dim=-1).mean().item())

    with torch.no_grad():
        for target_parameter, parameter in zip(
            bundle.target_acquisition_critic.parameters(),
            bundle.acquisition_critic.parameters(),
            strict=True,
        ):
            target_parameter.mul_(1.0 - config.target_tau).add_(parameter, alpha=config.target_tau)

    metrics = RelayDualGoalHerSACMetricsV643(
        update_index=update_index,
        critic_loss=float(critic_loss.item()),
        actor_loss=actor_loss_value,
        mean_q_target=float(q_target.mean().item()),
        mean_q_data=float(torch.minimum(q1, q2).mean().item()),
        predicted_policy_feasibility=predicted_feasibility_value,
        mean_acquisition_gate=float(gate.mean().item()),
        exact_home_source_fraction=float(np.mean(batch["exact_home_source"])),
        tool_goal_her_fraction=float(np.mean(batch["tool_goal_her_relabelled"])),
        original_goal_success_fraction=float(np.mean(batch["original_goal_success"])),
        hindsight_goal_success_fraction=float(np.mean(batch["hindsight_goal_success"])),
        mean_acquisition_progress_m=float(np.mean(batch["acquisition_progress_m"])),
        mean_learning_reward=float(reward.mean().item()),
        mean_transport_acquisition_action_delta_l2=mean_action_delta,
        actor_updated=actor_updated,
    )
    for name, value in asdict(metrics).items():
        if isinstance(value, float) and not np.isfinite(value):
            raise RuntimeError(f"V643 update metric {name} is non-finite")
    return metrics


__all__ = [
    "ACQUISITION_GOAL_DIM_V643",
    "ACQUISITION_OBSERVATION_DIM_V643",
    "RELAY_ACQUISITION_BATCH_FORMAT_V643",
    "RELAY_DUAL_GOAL_CHECKPOINT_FORMAT_V643",
    "RELAY_DUAL_GOAL_HER_SAC_FORMAT_V643",
    "RelayAcquisitionActorV643",
    "RelayDualGoalHerSACBundleV643",
    "RelayDualGoalHerSACConfigV643",
    "RelayDualGoalHerSACMetricsV643",
    "RelayDualGoalPolicyV643",
    "TwinRelayAcquisitionCriticV643",
    "acquisition_gate_numpy_v643",
    "acquisition_observation_v643",
    "initialize_relay_dual_goal_her_sac_v643",
    "precontact_goal_xyz_v643",
    "relay_dual_goal_her_sac_update_v643",
    "sample_relay_acquisition_batch_v643",
    "tool_xyz_from_neutral_v643",
]
