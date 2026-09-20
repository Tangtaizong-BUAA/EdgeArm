"""Controller-state-complete goal-conditioned HER-SAC teacher.

V43 observes the physical simulator state but omits the persistent Cartesian
and joint targets retained by its task-frame adapter.  V614 keeps the same
task, reward, replay/HER semantics, and three-dimensional action space while
conditioning actor, critic, and feasibility model on that exact controller
memory.  Historical V43 weights can be warm-started exactly: each new context
branch is zero-residual initialized, so the initial V614 policy is identical
to its frozen parent before learning from fresh state-complete transitions.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .adaptive_start_curriculum_v622 import START_TIERS_V622
from .causal_chosen_action_self_imitation_v717 import (
    CHOSEN_POLICY_ACTION_SELF_IMITATION_MODE_V717,
    MEASURED_EFFECT_SELF_IMITATION_MODE_V717,
    SELF_IMITATION_TARGET_MODES_V717,
    causal_chosen_forward_transport_loss_v717,
)
from .goal_conditioned_her_sac_v43 import (
    ACTION_DIM_V43,
    GOAL_CONDITIONED_HER_CHECKPOINT_FORMAT_V43,
    GOAL_CONDITIONED_HER_SAC_FORMAT_V43,
    OBSERVATION_DIM_V43,
    GoalConditionedActorV43,
    GoalConditionedHerReplayV43,
    GoalConditionedHerSACConfigV43,
    _ActionValueV43,
    _balanced_binary_loss,
)
from .markov_actor_trust_region_v621 import (
    MarkovActorTrustRegionConfigV621,
    bounded_context_hidden_v621,
    deterministic_action_anchor_v621,
)
from .phase_isolated_acquisition_v626 import (
    PhaseIsolatedAcquisitionConfigV626,
    acquisition_phase_gate_v626,
    phase_isolated_context_hidden_v626,
)
from .phase_isolated_acquisition_option_v639 import (
    PhaseIsolatedAcquisitionOptionConfigV639,
    phase_isolated_mean_residual_v639,
)
from .positive_effect_transport_self_imitation_v707 import (
    PositiveEffectTransportSelfImitationConfigV707,
    positive_effect_transport_self_imitation_loss_v707,
)
from .taskframe_controller_state_v614 import (
    TASKFRAME_CONTROLLER_STATE_DIM_V614,
    TASKFRAME_CONTROLLER_STATE_SCHEMA_SHA256_V614,
)


GOAL_CONDITIONED_MARKOV_HER_SAC_FORMAT_V614 = (
    "edgearm-v614-controller-state-complete-goal-conditioned-her-sac-v1"
)
GOAL_CONDITIONED_MARKOV_HER_REPLAY_FORMAT_V614 = (
    "edgearm-v614-controller-state-complete-her-replay-v1"
)
GOAL_CONDITIONED_MARKOV_HER_CHECKPOINT_FORMAT_V614 = (
    "edgearm-v614-controller-state-complete-her-sac-checkpoint-v1"
)


def _zero_last_linear_v614(module: nn.Sequential) -> None:
    last = module[-1]
    if not isinstance(last, nn.Linear):  # pragma: no cover - construction invariant
        raise TypeError("V614 context residual must end with Linear")
    nn.init.zeros_(last.weight)
    nn.init.zeros_(last.bias)


class GoalConditionedMarkovActorV614(nn.Module):
    """V43 actor plus a zero-initialized controller-state residual."""

    def __init__(self, hidden_dim: int = 256) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.base = GoalConditionedActorV43(hidden_dim)
        self.trust_region_v621: MarkovActorTrustRegionConfigV621 | None = None
        self.phase_isolated_acquisition_v626: (
            PhaseIsolatedAcquisitionConfigV626 | None
        ) = None
        self.observation_controller_encoder_v621: nn.Sequential | None = None
        self.acquisition_option_encoder_v639: nn.Sequential | None = None
        self.acquisition_option_v639: (
            PhaseIsolatedAcquisitionOptionConfigV639 | None
        ) = None
        self.controller_encoder = nn.Sequential(
            nn.LayerNorm(TASKFRAME_CONTROLLER_STATE_DIM_V614),
            nn.Linear(TASKFRAME_CONTROLLER_STATE_DIM_V614, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        _zero_last_linear_v614(self.controller_encoder)

    def install_observation_controller_residual_v621(self) -> None:
        """Add a zero-residual branch that can learn Home acquisition."""

        if self.observation_controller_encoder_v621 is not None:
            raise RuntimeError("V621 observation/controller residual already installed")
        encoder = nn.Sequential(
            nn.LayerNorm(
                OBSERVATION_DIM_V43 + TASKFRAME_CONTROLLER_STATE_DIM_V614
            ),
            nn.Linear(
                OBSERVATION_DIM_V43 + TASKFRAME_CONTROLLER_STATE_DIM_V614,
                self.hidden_dim,
            ),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        _zero_last_linear_v614(encoder)
        self.observation_controller_encoder_v621 = encoder

    def install_acquisition_option_v639(self) -> None:
        """Install a zero-residual action head for Home acquisition."""

        if self.acquisition_option_encoder_v639 is not None:
            raise RuntimeError("V639 acquisition option is already installed")
        encoder = nn.Sequential(
            nn.LayerNorm(
                OBSERVATION_DIM_V43 + TASKFRAME_CONTROLLER_STATE_DIM_V614
            ),
            nn.Linear(
                OBSERVATION_DIM_V43 + TASKFRAME_CONTROLLER_STATE_DIM_V614,
                self.hidden_dim,
            ),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, ACTION_DIM_V43),
        )
        _zero_last_linear_v614(encoder)
        self.acquisition_option_encoder_v639 = encoder

    def context_hidden_v621(
        self,
        observation: torch.Tensor,
        controller_state: torch.Tensor,
    ) -> torch.Tensor:
        context = self.controller_encoder(controller_state)
        observation_controller = self.observation_controller_encoder_v621
        if observation_controller is not None:
            context = context + observation_controller(
                torch.cat((observation, controller_state), dim=-1)
            )
        return context

    def acquisition_gate_v626(
        self,
        observation: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        config = self.phase_isolated_acquisition_v626
        if config is None:
            raise RuntimeError("V626 acquisition gate is not installed")
        return acquisition_phase_gate_v626(observation, config=config)

    def acquisition_option_mean_residual_v639(
        self,
        observation: torch.Tensor,
        controller_state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        config = self.acquisition_option_v639
        encoder = self.acquisition_option_encoder_v639
        if config is None or encoder is None:
            raise RuntimeError("V639 acquisition option is not installed")
        gate, _distance, _alignment, _contact = self.acquisition_gate_v626(
            observation
        )
        raw = encoder(torch.cat((observation, controller_state), dim=-1))
        residual = phase_isolated_mean_residual_v639(
            raw,
            gate,
            maximum_pre_tanh_mean_residual=(
                config.maximum_pre_tanh_mean_residual
            ),
        )
        return residual, gate

    def distribution(
        self,
        observation: torch.Tensor,
        controller_state: torch.Tensor,
    ) -> torch.distributions.Normal:
        if (
            observation.ndim != 2
            or observation.shape[-1] != OBSERVATION_DIM_V43
            or controller_state.shape
            != (observation.shape[0], TASKFRAME_CONTROLLER_STATE_DIM_V614)
        ):
            raise ValueError("V614 actor input shape changed")
        base_hidden = self.base.trunk(observation)
        context_hidden = self.context_hidden_v621(
            observation, controller_state
        )
        if self.phase_isolated_acquisition_v626 is not None:
            gate, _distance, _alignment, _contact = (
                self.acquisition_gate_v626(observation)
            )
            context_hidden, _realized_ratio = (
                phase_isolated_context_hidden_v626(
                    base_hidden,
                    context_hidden,
                    gate,
                    maximum_acquisition_ratio=(
                        self.phase_isolated_acquisition_v626
                        .maximum_acquisition_context_over_base_hidden_norm
                    ),
                )
            )
        elif self.trust_region_v621 is not None:
            context_hidden, _realized_ratio = bounded_context_hidden_v621(
                base_hidden,
                context_hidden,
                maximum_ratio=(
                    self.trust_region_v621
                    .maximum_context_over_base_hidden_norm
                ),
            )
        hidden = base_hidden + context_hidden
        mean = self.base.mean(hidden)
        if self.acquisition_option_v639 is not None:
            option_residual, _gate = (
                self.acquisition_option_mean_residual_v639(
                    observation, controller_state
                )
            )
            mean = mean + option_residual
        log_std = torch.clamp(self.base.log_std(hidden), -5.0, 1.0)
        return torch.distributions.Normal(mean, log_std.exp())

    def sample(
        self,
        observation: torch.Tensor,
        controller_state: torch.Tensor,
        *,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        distribution = self.distribution(observation, controller_state)
        pre_tanh = (
            distribution.mean if deterministic else distribution.rsample()
        )
        action = torch.tanh(pre_tanh)
        if deterministic:
            log_probability = torch.zeros(
                action.shape[0], device=action.device, dtype=action.dtype
            )
        else:
            correction = torch.log(1.0 - action.square() + 1.0e-6)
            log_probability = (
                distribution.log_prob(pre_tanh) - correction
            ).sum(dim=-1)
        return action, log_probability


class _MarkovActionValueV614(nn.Module):
    """Exact V43 base Q plus a context/action residual."""

    def __init__(self, hidden_dim: int = 256) -> None:
        super().__init__()
        self.base = _ActionValueV43(hidden_dim)
        context_action_dim = (
            TASKFRAME_CONTROLLER_STATE_DIM_V614 + ACTION_DIM_V43
        )
        self.controller_residual = nn.Sequential(
            nn.LayerNorm(context_action_dim),
            nn.Linear(context_action_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )
        _zero_last_linear_v614(self.controller_residual)

    def forward(
        self,
        observation: torch.Tensor,
        controller_state: torch.Tensor,
        action: torch.Tensor,
    ) -> torch.Tensor:
        if (
            observation.ndim != 2
            or observation.shape[-1] != OBSERVATION_DIM_V43
            or controller_state.shape
            != (observation.shape[0], TASKFRAME_CONTROLLER_STATE_DIM_V614)
            or action.shape != (observation.shape[0], ACTION_DIM_V43)
        ):
            raise ValueError("V614 action-value input shape changed")
        residual_input = torch.cat((controller_state, action), dim=-1)
        return self.base(observation, action) + self.controller_residual(
            residual_input
        ).squeeze(-1)


class TwinGoalConditionedMarkovCriticV614(nn.Module):
    def __init__(self, hidden_dim: int = 256) -> None:
        super().__init__()
        self.q1 = _MarkovActionValueV614(hidden_dim)
        self.q2 = _MarkovActionValueV614(hidden_dim)

    def forward(
        self,
        observation: torch.Tensor,
        controller_state: torch.Tensor,
        action: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            self.q1(observation, controller_state, action),
            self.q2(observation, controller_state, action),
        )


class ActionFeasibilityMarkovV614(_MarkovActionValueV614):
    pass


@dataclass
class GoalConditionedMarkovHerSACBundleV614:
    actor: GoalConditionedMarkovActorV614
    critic: TwinGoalConditionedMarkovCriticV614
    target_critic: TwinGoalConditionedMarkovCriticV614
    feasibility: ActionFeasibilityMarkovV614
    actor_optimizer: torch.optim.Optimizer
    critic_optimizer: torch.optim.Optimizer
    feasibility_optimizer: torch.optim.Optimizer
    actor_trust_region_v621: MarkovActorTrustRegionConfigV621 | None = None
    phase_isolated_acquisition_v626: (
        PhaseIsolatedAcquisitionConfigV626 | None
    ) = None
    phase_isolated_acquisition_option_v639: (
        PhaseIsolatedAcquisitionOptionConfigV639 | None
    ) = None
    update_index: int = 0
    format: str = GOAL_CONDITIONED_MARKOV_HER_SAC_FORMAT_V614


def initialize_goal_conditioned_markov_her_sac_v614(
    seed: int,
    *,
    device: str | torch.device,
    config: GoalConditionedHerSACConfigV43 | None = None,
    actor_trust_region_v621: MarkovActorTrustRegionConfigV621 | None = None,
    phase_isolated_acquisition_v626: (
        PhaseIsolatedAcquisitionConfigV626 | None
    ) = None,
    phase_isolated_acquisition_option_v639: (
        PhaseIsolatedAcquisitionOptionConfigV639 | None
    ) = None,
) -> GoalConditionedMarkovHerSACBundleV614:
    if type(seed) is not int or seed < 0:
        raise ValueError("V614 initialization seed must be non-negative")
    selected = config or GoalConditionedHerSACConfigV43()
    selected.validate()
    if actor_trust_region_v621 is not None:
        actor_trust_region_v621.validate()
    if phase_isolated_acquisition_v626 is not None:
        phase_isolated_acquisition_v626.validate()
        if (
            actor_trust_region_v621 is None
            or not actor_trust_region_v621.freeze_parent_base
        ):
            raise ValueError("V626 requires the frozen V621 parent actor")
    if phase_isolated_acquisition_option_v639 is not None:
        phase_isolated_acquisition_option_v639.validate()
        if phase_isolated_acquisition_v626 is None:
            raise ValueError("V639 requires the V626 acquisition gate")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        actor = GoalConditionedMarkovActorV614(selected.hidden_dim)
        if actor_trust_region_v621 is not None:
            actor.install_observation_controller_residual_v621()
        if phase_isolated_acquisition_option_v639 is not None:
            actor.install_acquisition_option_v639()
        critic = TwinGoalConditionedMarkovCriticV614(selected.hidden_dim)
        target = TwinGoalConditionedMarkovCriticV614(selected.hidden_dim)
        feasibility = ActionFeasibilityMarkovV614(selected.hidden_dim)
    target.load_state_dict(critic.state_dict(), strict=True)
    for parameter in target.parameters():
        parameter.requires_grad_(False)
    actor = actor.to(device)
    critic = critic.to(device)
    target = target.to(device)
    feasibility = feasibility.to(device)
    actor.trust_region_v621 = actor_trust_region_v621
    actor.phase_isolated_acquisition_v626 = (
        phase_isolated_acquisition_v626
    )
    actor.acquisition_option_v639 = (
        phase_isolated_acquisition_option_v639
    )
    if (
        actor_trust_region_v621 is not None
        and actor_trust_region_v621.freeze_parent_base
    ):
        for parameter in actor.base.parameters():
            parameter.requires_grad_(False)
    trainable_actor_parameters = [
        parameter for parameter in actor.parameters() if parameter.requires_grad
    ]
    if not trainable_actor_parameters:
        raise RuntimeError("V621 trust region froze the entire actor")
    return GoalConditionedMarkovHerSACBundleV614(
        actor=actor,
        critic=critic,
        target_critic=target,
        feasibility=feasibility,
        actor_optimizer=torch.optim.Adam(
            trainable_actor_parameters, lr=selected.actor_learning_rate
        ),
        critic_optimizer=torch.optim.Adam(
            critic.parameters(), lr=selected.critic_learning_rate
        ),
        feasibility_optimizer=torch.optim.Adam(
            feasibility.parameters(), lr=selected.feasibility_learning_rate
        ),
        actor_trust_region_v621=actor_trust_region_v621,
        phase_isolated_acquisition_v626=phase_isolated_acquisition_v626,
        phase_isolated_acquisition_option_v639=(
            phase_isolated_acquisition_option_v639
        ),
    )


def _parent_action_value_state_v614(
    state: dict[str, torch.Tensor],
    prefix: str,
) -> dict[str, torch.Tensor]:
    selected = {
        key[len(prefix) :]: value
        for key, value in state.items()
        if key.startswith(prefix)
    }
    if not selected:
        raise ValueError(f"V614 parent action-value state lacks {prefix}")
    return selected


def warm_start_markov_her_sac_v614_from_v43(
    bundle: GoalConditionedMarkovHerSACBundleV614,
    payload: dict[str, Any],
) -> dict[str, Any]:
    """Copy V43 behavior exactly and leave every context residual at zero."""

    if (
        type(payload) is not dict
        or payload.get("format")
        != GOAL_CONDITIONED_HER_CHECKPOINT_FORMAT_V43
        or payload.get("algorithm_format") != GOAL_CONDITIONED_HER_SAC_FORMAT_V43
        or int(payload.get("expert_calls", -1)) != 0
        or int(payload.get("behavior_cloning_steps", -1)) != 0
        or payload.get("production_admission") is not False
    ):
        raise ValueError("V614 parent V43 identity or provenance is invalid")
    bundle.actor.base.load_state_dict(payload["actor_state_dict"], strict=True)
    critic_state = payload["critic_state_dict"]
    target_state = payload["target_critic_state_dict"]
    bundle.critic.q1.base.load_state_dict(
        _parent_action_value_state_v614(critic_state, "q1."), strict=True
    )
    bundle.critic.q2.base.load_state_dict(
        _parent_action_value_state_v614(critic_state, "q2."), strict=True
    )
    bundle.target_critic.q1.base.load_state_dict(
        _parent_action_value_state_v614(target_state, "q1."), strict=True
    )
    bundle.target_critic.q2.base.load_state_dict(
        _parent_action_value_state_v614(target_state, "q2."), strict=True
    )
    bundle.feasibility.base.load_state_dict(
        payload["feasibility_state_dict"], strict=True
    )
    context_parameters = [
        bundle.actor.controller_encoder[-1].weight,
        bundle.actor.controller_encoder[-1].bias,
        bundle.critic.q1.controller_residual[-1].weight,
        bundle.critic.q1.controller_residual[-1].bias,
        bundle.critic.q2.controller_residual[-1].weight,
        bundle.critic.q2.controller_residual[-1].bias,
        bundle.target_critic.q1.controller_residual[-1].weight,
        bundle.target_critic.q1.controller_residual[-1].bias,
        bundle.target_critic.q2.controller_residual[-1].weight,
        bundle.target_critic.q2.controller_residual[-1].bias,
        bundle.feasibility.controller_residual[-1].weight,
        bundle.feasibility.controller_residual[-1].bias,
    ]
    observation_controller = (
        bundle.actor.observation_controller_encoder_v621
    )
    if observation_controller is not None:
        context_parameters.extend(
            [
                observation_controller[-1].weight,
                observation_controller[-1].bias,
            ]
        )
    acquisition_option = bundle.actor.acquisition_option_encoder_v639
    if acquisition_option is not None:
        context_parameters.extend(
            [
                acquisition_option[-1].weight,
                acquisition_option[-1].bias,
            ]
        )
    if any(bool(torch.count_nonzero(value).item()) for value in context_parameters):
        raise RuntimeError("V614 context residual did not remain zero initialized")
    return {
        "format": "edgearm-v614-v43-exact-behavior-warm-start-v1",
        "parent_update_index": int(payload["update_index"]),
        "parent_actor_behavior_copied": True,
        "parent_critic_behavior_copied": True,
        "parent_feasibility_behavior_copied": True,
        "controller_context_residual_zero_initialized": True,
        "phase_isolated_acquisition_v626": (
            bundle.phase_isolated_acquisition_v626 is not None
        ),
        "phase_isolated_acquisition_option_v639": (
            bundle.phase_isolated_acquisition_option_v639 is not None
        ),
        "exact_frozen_parent_transport_at_zero_gate_v626": (
            bundle.phase_isolated_acquisition_v626 is not None
        ),
        "optimizer_state_transferred": False,
        "controller_state_schema_sha256": (
            TASKFRAME_CONTROLLER_STATE_SCHEMA_SHA256_V614
        ),
        "expert_calls": 0,
        "behavior_cloning_steps": 0,
        "production_admission": False,
    }


class GoalConditionedMarkovHerReplayV614(GoalConditionedHerReplayV43):
    """Fresh V43 replay with exact controller state on every transition."""

    def __init__(self, config: GoalConditionedHerSACConfigV43 | None = None) -> None:
        super().__init__(config)
        self.controller_state = np.empty(
            (0, TASKFRAME_CONTROLLER_STATE_DIM_V614), dtype=np.float32
        )
        self.next_controller_state = np.empty_like(self.controller_state)
        self.start_tier_index_v622 = np.empty(0, dtype=np.int8)
        self.home_to_precontact_fraction_v750 = np.empty(
            0,
            dtype=np.float32,
        )
        self.direct_precontact_reset_v750 = np.empty(0, dtype=bool)
        self.reset_fraction_exact_v750 = np.empty(0, dtype=bool)
        self.execution_action_feasible_v750 = np.empty(0, dtype=bool)
        self.execution_action_feasibility_exact_v750 = np.empty(
            0,
            dtype=bool,
        )
        self.safeguard_projected_action_v720 = np.empty(
            (0, ACTION_DIM_V43),
            dtype=np.float32,
        )
        self.safeguard_projection_valid_v720 = np.empty(0, dtype=bool)

    def add_episode(self, episode: dict[str, np.ndarray], *, source: str) -> None:
        count = int(len(np.asarray(episode.get("terminal", ()))))
        controller = np.asarray(
            episode.get("controller_state"), dtype=np.float32
        )
        next_controller = np.asarray(
            episode.get("next_controller_state"), dtype=np.float32
        )
        expected = (count, TASKFRAME_CONTROLLER_STATE_DIM_V614)
        if (
            count < 1
            or controller.shape != expected
            or next_controller.shape != expected
            or not np.all(np.isfinite(controller))
            or not np.all(np.isfinite(next_controller))
            or not np.all(controller[:, -2:] == 1.0)
            or not np.all(next_controller[:, -2:] == 1.0)
        ):
            raise ValueError("V614 replay episode lacks valid controller state")
        tier = episode.get("start_tier_index_v622")
        if tier is None:
            start_tier = np.full(count, -1, dtype=np.int8)
        else:
            start_tier = np.asarray(tier, dtype=np.int8)
            if (
                start_tier.shape != (count,)
                or np.any(start_tier < 0)
                or np.any(start_tier >= len(START_TIERS_V622))
                or not np.all(start_tier == start_tier[0])
            ):
                raise ValueError("V622 replay start tier is invalid")
        reset_fraction_value = episode.get(
            "home_to_precontact_fraction_v750"
        )
        direct_precontact_value = episode.get(
            "direct_precontact_reset_v750"
        )
        if (reset_fraction_value is None) != (
            direct_precontact_value is None
        ):
            raise ValueError(
                "V750 replay reset-provenance fields must be supplied together"
            )
        if reset_fraction_value is None:
            reset_fraction = np.empty(count, dtype=np.float32)
            direct_precontact = np.empty(count, dtype=bool)
            for row, tier_index in enumerate(start_tier):
                if int(tier_index) < 0:
                    # Unlabelled legacy rows are fail-closed as possibly direct
                    # precontact.  No geometry is fabricated for V748 learning.
                    reset_fraction[row] = np.float32(1.0)
                    direct_precontact[row] = True
                else:
                    tier_spec = START_TIERS_V622[int(tier_index)]
                    reset_fraction[row] = np.float32(
                        tier_spec.home_to_precontact_fraction
                    )
                    direct_precontact[row] = bool(tier_spec.index == 0)
            reset_fraction_exact = np.zeros(count, dtype=bool)
        else:
            reset_fraction = np.asarray(
                reset_fraction_value,
                dtype=np.float32,
            )
            direct_precontact = np.asarray(
                direct_precontact_value,
                dtype=bool,
            )
            if (
                reset_fraction.shape != (count,)
                or direct_precontact.shape != (count,)
                or not np.all(np.isfinite(reset_fraction))
                or np.any(reset_fraction < 0.0)
                or np.any(reset_fraction > 1.0)
                or not np.all(reset_fraction == reset_fraction[0])
                or not np.all(direct_precontact == direct_precontact[0])
                or bool(direct_precontact[0])
                != bool(np.isclose(reset_fraction[0], 1.0, atol=1.0e-7))
            ):
                raise ValueError("V750 replay reset provenance is invalid")
            reset_fraction_exact = np.ones(count, dtype=bool)
            if int(start_tier[0]) == START_TIERS_V622[-1].index and not bool(
                np.isclose(reset_fraction[0], 0.0, atol=1.0e-7)
            ):
                raise ValueError(
                    "V750 exact-Home tier must carry zero reset fraction"
                )
        execution_feasible_value = episode.get(
            "execution_action_feasible_v741"
        )
        if execution_feasible_value is None:
            execution_feasible = np.asarray(
                episode["action_feasible"],
                dtype=bool,
            )
            execution_feasibility_exact = np.zeros(count, dtype=bool)
        else:
            execution_feasible = np.asarray(
                execution_feasible_value,
                dtype=bool,
            )
            execution_feasibility_exact = np.ones(count, dtype=bool)
        if execution_feasible.shape != (count,):
            raise ValueError(
                "V750 replay execution feasibility is invalid"
            )
        projected_value = episode.get("safeguard_projected_action_v720")
        projection_valid_value = episode.get(
            "safeguard_projection_valid_v720"
        )
        if (projected_value is None) != (projection_valid_value is None):
            raise ValueError(
                "V720 replay safeguard projection fields must be supplied together"
            )
        if projected_value is None:
            safeguard_projected = np.zeros(
                (count, ACTION_DIM_V43),
                dtype=np.float32,
            )
            safeguard_valid = np.zeros(count, dtype=bool)
        else:
            safeguard_projected = np.asarray(
                projected_value,
                dtype=np.float32,
            )
            safeguard_valid = np.asarray(
                projection_valid_value,
                dtype=bool,
            )
            if (
                safeguard_projected.shape != (count, ACTION_DIM_V43)
                or safeguard_valid.shape != (count,)
                or not np.all(np.isfinite(safeguard_projected))
                or np.any(np.abs(safeguard_projected) > 1.0 + 1.0e-6)
            ):
                raise ValueError("V720 replay safeguard projection is invalid")
        base_episode = {
            key: value
            for key, value in episode.items()
            if key
            not in {
                "controller_state",
                "next_controller_state",
                "home_to_precontact_fraction_v750",
                "direct_precontact_reset_v750",
                "execution_action_feasible_v741",
                "safeguard_projected_action_v720",
                "safeguard_projection_valid_v720",
            }
        }
        super().add_episode(base_episode, source=source)
        self.controller_state = np.concatenate(
            (self.controller_state, controller), axis=0
        )
        self.next_controller_state = np.concatenate(
            (self.next_controller_state, next_controller), axis=0
        )
        self.start_tier_index_v622 = np.concatenate(
            (self.start_tier_index_v622, start_tier), axis=0
        )
        self.home_to_precontact_fraction_v750 = np.concatenate(
            (self.home_to_precontact_fraction_v750, reset_fraction),
            axis=0,
        )
        self.direct_precontact_reset_v750 = np.concatenate(
            (self.direct_precontact_reset_v750, direct_precontact),
            axis=0,
        )
        self.reset_fraction_exact_v750 = np.concatenate(
            (self.reset_fraction_exact_v750, reset_fraction_exact),
            axis=0,
        )
        self.execution_action_feasible_v750 = np.concatenate(
            (self.execution_action_feasible_v750, execution_feasible),
            axis=0,
        )
        self.execution_action_feasibility_exact_v750 = np.concatenate(
            (
                self.execution_action_feasibility_exact_v750,
                execution_feasibility_exact,
            ),
            axis=0,
        )
        self.safeguard_projected_action_v720 = np.concatenate(
            (
                self.safeguard_projected_action_v720,
                safeguard_projected,
            ),
            axis=0,
        )
        self.safeguard_projection_valid_v720 = np.concatenate(
            (self.safeguard_projection_valid_v720, safeguard_valid),
            axis=0,
        )
        if int(start_tier[0]) >= 0:
            self.source_rows[-1]["start_tier_v622"] = START_TIERS_V622[
                int(start_tier[0])
            ].code
        self.source_rows[-1]["home_to_precontact_fraction_v750"] = float(
            reset_fraction[0]
        )
        self.source_rows[-1]["direct_precontact_reset_v750"] = bool(
            direct_precontact[0]
        )
        self.source_rows[-1]["reset_fraction_exact_v750"] = bool(
            reset_fraction_exact[0]
        )
        self.source_rows[-1]["execution_feasibility_exact_v750"] = bool(
            np.all(execution_feasibility_exact)
        )
        if (
            len(self.controller_state) != self.transition_count
            or len(self.home_to_precontact_fraction_v750)
            != self.transition_count
            or len(self.direct_precontact_reset_v750)
            != self.transition_count
            or len(self.reset_fraction_exact_v750)
            != self.transition_count
            or len(self.execution_action_feasible_v750)
            != self.transition_count
            or len(self.execution_action_feasibility_exact_v750)
            != self.transition_count
            or len(self.safeguard_projected_action_v720)
            != self.transition_count
            or len(self.safeguard_projection_valid_v720)
            != self.transition_count
        ):
            raise RuntimeError("V614 replay controller rows drifted")

    def sample(self, **kwargs: Any) -> dict[str, np.ndarray]:
        batch = super().sample(**kwargs)
        rows = np.asarray(batch["source_row_index"], dtype=np.int64)
        batch["controller_state"] = self.controller_state[rows].copy()
        batch["next_controller_state"] = (
            self.next_controller_state[rows].copy()
        )
        batch["start_tier_index_v622"] = (
            self.start_tier_index_v622[rows].copy()
        )
        batch["home_to_precontact_fraction_v750"] = (
            self.home_to_precontact_fraction_v750[rows].copy()
        )
        batch["direct_precontact_reset_v750"] = (
            self.direct_precontact_reset_v750[rows].copy()
        )
        batch["reset_fraction_exact_v750"] = (
            self.reset_fraction_exact_v750[rows].copy()
        )
        batch["execution_action_feasible_v750"] = (
            self.execution_action_feasible_v750[rows].copy()
        )
        batch["execution_action_feasibility_exact_v750"] = (
            self.execution_action_feasibility_exact_v750[rows].copy()
        )
        batch["safeguard_projected_action_v720"] = (
            self.safeguard_projected_action_v720[rows].copy()
        )
        batch["safeguard_projection_valid_v720"] = (
            self.safeguard_projection_valid_v720[rows].copy()
        )
        return batch

    def _sampling_weights(self) -> np.ndarray:
        """Preserve within-tier priorities while balancing observed V622 tiers."""

        weights = super()._sampling_weights()
        tier = self.start_tier_index_v622
        if self.config.safeguard_projection_penalty_v720:
            valid = self.safeguard_projection_valid_v720
            if (
                valid.shape != (self.transition_count,)
                or valid.dtype != np.dtype(bool)
            ):
                raise RuntimeError("V720 replay projection validity changed")
            weights = np.where(valid, weights, 0.0)
            mass = float(np.sum(weights))
            if mass <= 0.0:
                raise RuntimeError(
                    "V720 replay has no reward-semantics-complete transitions"
                )
            weights = weights / mass
        positive_weight = weights > 0.0
        observed = sorted(
            int(value)
            for value in np.unique(tier[positive_weight])
            if value >= 0
        )
        if not observed:
            return weights
        adaptive_rows = (tier >= 0) | ~positive_weight
        if bool(np.any(~adaptive_rows)):
            raise RuntimeError(
                "V622 adaptive replay cannot mix unlabelled legacy transitions"
            )
        balanced = weights.copy()
        target_mass = 1.0 / len(observed)
        for index in observed:
            selected = tier == index
            mass = float(np.sum(weights[selected]))
            if mass <= 0.0:
                raise RuntimeError("V622 replay tier has zero sampling mass")
            balanced[selected] *= target_mass / mass
        return balanced / float(np.sum(balanced))

    def manifest(self) -> dict[str, Any]:
        base = super().manifest()
        tier_counts = {
            tier.code: int(np.sum(self.start_tier_index_v622 == tier.index))
            for tier in START_TIERS_V622
        }
        return {
            **base,
            "format": GOAL_CONDITIONED_MARKOV_HER_REPLAY_FORMAT_V614,
            "controller_state_dimension": (
                TASKFRAME_CONTROLLER_STATE_DIM_V614
            ),
            "controller_state_schema_sha256": (
                TASKFRAME_CONTROLLER_STATE_SCHEMA_SHA256_V614
            ),
            "controller_state_transition_count": len(self.controller_state),
            "controller_state_complete": bool(
                len(self.controller_state) == self.transition_count
            ),
            "start_tier_schema_v622": [tier.code for tier in START_TIERS_V622],
            "start_tier_transition_counts_v622": tier_counts,
            "start_tier_complete_v622": bool(
                self.transition_count > 0
                and np.all(self.start_tier_index_v622 >= 0)
            ),
            "start_tier_stratified_sampling_v622": bool(
                self.transition_count > 0
                and np.all(self.start_tier_index_v622 >= 0)
            ),
            "reset_fraction_provenance_v750": (
                "explicit_curriculum_fraction_or_fail_closed_legacy_tier"
            ),
            "reset_fraction_exact_transition_count_v750": int(
                np.count_nonzero(self.reset_fraction_exact_v750)
            ),
            "reset_fraction_legacy_backfill_transition_count_v750": int(
                self.transition_count
                - np.count_nonzero(self.reset_fraction_exact_v750)
            ),
            "direct_precontact_transition_count_v750": int(
                np.count_nonzero(self.direct_precontact_reset_v750)
            ),
            "ambiguous_legacy_tier_zero_excluded_from_v748_count_v750": int(
                np.count_nonzero(
                    self.direct_precontact_reset_v750
                    & ~self.reset_fraction_exact_v750
                )
            ),
            "execution_feasibility_exact_transition_count_v750": int(
                np.count_nonzero(
                    self.execution_action_feasibility_exact_v750
                )
            ),
            "execution_feasibility_legacy_fallback_transition_count_v750": int(
                self.transition_count
                - np.count_nonzero(
                    self.execution_action_feasibility_exact_v750
                )
            ),
            "safeguard_projection_semantics_v720": (
                "dls_predicted_guard_scaled_pre_plant_safe_task_action"
            ),
            "safeguard_projection_valid_transition_count_v720": int(
                np.count_nonzero(self.safeguard_projection_valid_v720)
            ),
            "legacy_actual_effect_projection_excluded_count_v720": int(
                self.transition_count
                - np.count_nonzero(self.safeguard_projection_valid_v720)
            ),
            "v720_value_update_eligible_transition_count": int(
                np.count_nonzero(self.safeguard_projection_valid_v720)
                if self.config.safeguard_projection_penalty_v720
                else self.transition_count
            ),
            "legacy_rows_excluded_from_v720_value_updates": bool(
                self.config.safeguard_projection_penalty_v720
            ),
            "production_admission": False,
        }

    def save_npz(self, path: Path) -> None:
        destination = Path(path).expanduser().resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        partial = destination.with_suffix(destination.suffix + ".partial")
        with partial.open("wb") as stream:
            np.savez_compressed(
                stream,
                **self.arrays,
                controller_state=self.controller_state,
                next_controller_state=self.next_controller_state,
                start_tier_index_v622=self.start_tier_index_v622,
                home_to_precontact_fraction_v750=(
                    self.home_to_precontact_fraction_v750
                ),
                direct_precontact_reset_v750=(
                    self.direct_precontact_reset_v750
                ),
                reset_fraction_exact_v750=(
                    self.reset_fraction_exact_v750
                ),
                execution_action_feasible_v750=(
                    self.execution_action_feasible_v750
                ),
                execution_action_feasibility_exact_v750=(
                    self.execution_action_feasibility_exact_v750
                ),
                safeguard_projected_action_v720=(
                    self.safeguard_projected_action_v720
                ),
                safeguard_projection_valid_v720=(
                    self.safeguard_projection_valid_v720
                ),
                next_episode_index=np.asarray(
                    [self._next_episode_index], np.int64
                ),
            )
        partial.replace(destination)

    @classmethod
    def load_npz(
        cls,
        path: Path,
        config: GoalConditionedHerSACConfigV43 | None = None,
    ) -> "GoalConditionedMarkovHerReplayV614":
        result = super().load_npz(path, config)
        if not isinstance(result, cls):  # pragma: no cover - cls invariant
            raise TypeError("V614 replay loader returned the wrong class")
        source = Path(path).expanduser().resolve()
        with np.load(source, allow_pickle=False) as archive:
            if (
                "controller_state" not in archive
                or "next_controller_state" not in archive
            ):
                raise ValueError("V614 replay file lacks controller state")
            result.controller_state = np.asarray(
                archive["controller_state"], dtype=np.float32
            )
            result.next_controller_state = np.asarray(
                archive["next_controller_state"], dtype=np.float32
            )
            result.start_tier_index_v622 = (
                np.asarray(archive["start_tier_index_v622"], dtype=np.int8)
                if "start_tier_index_v622" in archive
                else np.full(result.transition_count, -1, dtype=np.int8)
            )
            reset_provenance_fields_v750 = {
                "home_to_precontact_fraction_v750",
                "direct_precontact_reset_v750",
                "reset_fraction_exact_v750",
            }
            present_reset_provenance_fields_v750 = (
                reset_provenance_fields_v750.intersection(archive.files)
            )
            if present_reset_provenance_fields_v750 and (
                present_reset_provenance_fields_v750
                != reset_provenance_fields_v750
            ):
                raise ValueError(
                    "V750 persisted reset provenance is incomplete"
                )
            if present_reset_provenance_fields_v750:
                result.home_to_precontact_fraction_v750 = np.asarray(
                    archive["home_to_precontact_fraction_v750"],
                    dtype=np.float32,
                )
                result.direct_precontact_reset_v750 = np.asarray(
                    archive["direct_precontact_reset_v750"],
                    dtype=bool,
                )
                result.reset_fraction_exact_v750 = np.asarray(
                    archive["reset_fraction_exact_v750"],
                    dtype=bool,
                )
            else:
                result.home_to_precontact_fraction_v750 = np.empty(
                    result.transition_count,
                    dtype=np.float32,
                )
                result.direct_precontact_reset_v750 = np.empty(
                    result.transition_count,
                    dtype=bool,
                )
                for row, tier_index in enumerate(
                    result.start_tier_index_v622
                ):
                    if int(tier_index) < 0:
                        result.home_to_precontact_fraction_v750[row] = 1.0
                        result.direct_precontact_reset_v750[row] = True
                    else:
                        tier_spec = START_TIERS_V622[int(tier_index)]
                        result.home_to_precontact_fraction_v750[row] = (
                            tier_spec.home_to_precontact_fraction
                        )
                        result.direct_precontact_reset_v750[row] = bool(
                            tier_spec.index == 0
                        )
                result.reset_fraction_exact_v750 = np.zeros(
                    result.transition_count,
                    dtype=bool,
                )
            execution_provenance_fields_v750 = {
                "execution_action_feasible_v750",
                "execution_action_feasibility_exact_v750",
            }
            present_execution_provenance_fields_v750 = (
                execution_provenance_fields_v750.intersection(
                    archive.files
                )
            )
            if present_execution_provenance_fields_v750 and (
                present_execution_provenance_fields_v750
                != execution_provenance_fields_v750
            ):
                raise ValueError(
                    "V750 persisted execution provenance is incomplete"
                )
            if present_execution_provenance_fields_v750:
                result.execution_action_feasible_v750 = np.asarray(
                    archive["execution_action_feasible_v750"],
                    dtype=bool,
                )
                result.execution_action_feasibility_exact_v750 = (
                    np.asarray(
                        archive[
                            "execution_action_feasibility_exact_v750"
                        ],
                        dtype=bool,
                    )
                )
            else:
                result.execution_action_feasible_v750 = np.asarray(
                    result.arrays["action_feasible"],
                    dtype=bool,
                ).copy()
                result.execution_action_feasibility_exact_v750 = (
                    np.zeros(result.transition_count, dtype=bool)
                )
            result.safeguard_projected_action_v720 = (
                np.asarray(
                    archive["safeguard_projected_action_v720"],
                    dtype=np.float32,
                )
                if "safeguard_projected_action_v720" in archive
                else np.zeros(
                    (result.transition_count, ACTION_DIM_V43),
                    dtype=np.float32,
                )
            )
            result.safeguard_projection_valid_v720 = (
                np.asarray(
                    archive["safeguard_projection_valid_v720"],
                    dtype=bool,
                )
                if "safeguard_projection_valid_v720" in archive
                else np.zeros(result.transition_count, dtype=bool)
            )
        expected = (
            result.transition_count,
            TASKFRAME_CONTROLLER_STATE_DIM_V614,
        )
        if (
            result.controller_state.shape != expected
            or result.next_controller_state.shape != expected
            or not np.all(np.isfinite(result.controller_state))
            or not np.all(np.isfinite(result.next_controller_state))
            or not np.all(result.controller_state[:, -2:] == 1.0)
            or not np.all(result.next_controller_state[:, -2:] == 1.0)
            or result.start_tier_index_v622.shape
            != (result.transition_count,)
            or np.any(result.start_tier_index_v622 < -1)
            or np.any(
                result.start_tier_index_v622 >= len(START_TIERS_V622)
            )
            or result.home_to_precontact_fraction_v750.shape
            != (result.transition_count,)
            or result.direct_precontact_reset_v750.shape
            != (result.transition_count,)
            or result.reset_fraction_exact_v750.shape
            != (result.transition_count,)
            or not np.all(
                np.isfinite(result.home_to_precontact_fraction_v750)
            )
            or np.any(result.home_to_precontact_fraction_v750 < 0.0)
            or np.any(result.home_to_precontact_fraction_v750 > 1.0)
            or np.any(
                result.direct_precontact_reset_v750
                & ~np.isclose(
                    result.home_to_precontact_fraction_v750,
                    1.0,
                    atol=1.0e-7,
                )
            )
            or result.execution_action_feasible_v750.shape
            != (result.transition_count,)
            or result.execution_action_feasibility_exact_v750.shape
            != (result.transition_count,)
            or result.safeguard_projected_action_v720.shape
            != (result.transition_count, ACTION_DIM_V43)
            or result.safeguard_projection_valid_v720.shape
            != (result.transition_count,)
            or not np.all(
                np.isfinite(result.safeguard_projected_action_v720)
            )
            or np.any(
                np.abs(result.safeguard_projected_action_v720)
                > 1.0 + 1.0e-6
            )
        ):
            raise ValueError("V614 persisted controller state is invalid")
        return result


@dataclass(frozen=True)
class GoalConditionedMarkovHerSACMetricsV614:
    update_index: int
    critic_loss: float
    actor_loss: float
    feasibility_loss: float
    feasibility_accuracy: float
    mean_q_target: float
    mean_q_data: float
    mean_policy_action_abs: float
    mean_policy_forward_action: float
    predicted_policy_feasibility: float
    mean_controller_state_abs: float
    her_relabel_fraction: float
    strict_success_source_fraction: float
    mean_learning_reward: float
    actor_updated: bool
    actor_anchor_loss_v621: float
    mean_anchor_weight_v621: float
    mean_parent_action_delta_l2_v621: float
    maximum_context_over_base_hidden_norm_v621: float
    actor_parent_base_frozen_v621: bool
    mean_acquisition_gate_v626: float
    exact_transport_parent_row_fraction_v626: float
    mean_acquisition_parent_action_delta_l2_v626: float
    mean_acquisition_option_residual_l2_v639: float
    maximum_acquisition_option_residual_l2_v639: float
    positive_effect_transport_self_imitation_enabled_v707: bool = False
    positive_effect_transport_self_imitation_loss_v707: float = 0.0
    positive_effect_transport_self_imitation_selected_fraction_v707: float = 0.0
    positive_effect_transport_self_imitation_mean_progress_m_v707: float = 0.0
    positive_effect_transport_self_imitation_originally_infeasible_fraction_v707: float = 0.0
    causal_chosen_forward_self_imitation_enabled_v717: bool = False
    causal_chosen_forward_self_imitation_loss_v717: float = 0.0
    causal_chosen_forward_self_imitation_selected_fraction_v717: float = 0.0
    causal_chosen_forward_self_imitation_mean_progress_m_v717: float = 0.0
    causal_chosen_forward_self_imitation_filtered_mdp_fraction_v717: float = 0.0
    actor_update_enabled_v721: bool = True
    format: str = GOAL_CONDITIONED_MARKOV_HER_SAC_FORMAT_V614


def goal_conditioned_markov_her_sac_update_v614(
    bundle: GoalConditionedMarkovHerSACBundleV614,
    batch: dict[str, np.ndarray],
    config: GoalConditionedHerSACConfigV43,
    *,
    transport_self_imitation_config_v707: (
        PositiveEffectTransportSelfImitationConfigV707 | None
    ) = None,
    transport_self_imitation_target_mode: str = (
        MEASURED_EFFECT_SELF_IMITATION_MODE_V717
    ),
    actor_update_enabled_v721: bool = True,
) -> GoalConditionedMarkovHerSACMetricsV614:
    config.validate()
    if type(actor_update_enabled_v721) is not bool:
        raise ValueError("V721 actor update gate must be an exact boolean")
    transport_self_imitation = (
        transport_self_imitation_config_v707
        or PositiveEffectTransportSelfImitationConfigV707()
    )
    if (
        type(transport_self_imitation)
        is not PositiveEffectTransportSelfImitationConfigV707
    ):
        raise TypeError("V614 requires the exact V707 transport self-imitation config")
    transport_self_imitation.validate()
    if transport_self_imitation_target_mode not in SELF_IMITATION_TARGET_MODES_V717:
        raise ValueError("V614 transport self-imitation target mode is invalid")
    if bundle.format != GOAL_CONDITIONED_MARKOV_HER_SAC_FORMAT_V614:
        raise ValueError("V614 bundle identity changed")
    device = next(bundle.actor.parameters()).device

    def tensor(name: str) -> torch.Tensor:
        if name not in batch:
            raise KeyError(f"V614 batch lacks {name}")
        return torch.from_numpy(batch[name]).to(device)

    observation = tensor("observation")
    next_observation = tensor("next_observation")
    controller = tensor("controller_state")
    next_controller = tensor("next_controller_state")
    action = tensor("action")
    reward = tensor("reward")
    done = tensor("done")
    feasible = tensor("action_feasible")
    importance = tensor("importance_weight")
    batch_size = observation.shape[0]
    if (
        observation.shape != (batch_size, OBSERVATION_DIM_V43)
        or next_observation.shape != observation.shape
        or controller.shape
        != (batch_size, TASKFRAME_CONTROLLER_STATE_DIM_V614)
        or next_controller.shape != controller.shape
        or action.shape != (batch_size, ACTION_DIM_V43)
    ):
        raise ValueError("V614 update batch shape changed")

    bundle.update_index += 1
    update_index = bundle.update_index
    feasibility_logit = bundle.feasibility(
        observation, controller, action
    )
    feasibility_loss = _balanced_binary_loss(feasibility_logit, feasible)
    bundle.feasibility_optimizer.zero_grad(set_to_none=True)
    feasibility_loss.backward()
    nn.utils.clip_grad_norm_(
        bundle.feasibility.parameters(), config.maximum_gradient_norm
    )
    bundle.feasibility_optimizer.step()

    with torch.no_grad():
        next_action, next_log_probability = bundle.actor.sample(
            next_observation, next_controller
        )
        target_q1, target_q2 = bundle.target_critic(
            next_observation, next_controller, next_action
        )
        target_value = torch.minimum(target_q1, target_q2) - (
            config.entropy_temperature * next_log_probability
        )
        q_target = reward + config.gamma * (1.0 - done) * target_value

    q1, q2 = bundle.critic(observation, controller, action)
    critic_td = F.smooth_l1_loss(
        q1, q_target, reduction="none"
    ) + F.smooth_l1_loss(q2, q_target, reduction="none")
    random_action = torch.empty_like(action).uniform_(-1.0, 1.0)
    random_q1, random_q2 = bundle.critic(
        observation, controller, random_action
    )
    conservative = (
        torch.logsumexp(torch.stack((random_q1, q1), dim=0), dim=0) - q1
        + torch.logsumexp(torch.stack((random_q2, q2), dim=0), dim=0)
        - q2
    ).mean()
    critic_loss = (importance * critic_td).mean() + (
        config.critic_conservative_coefficient * conservative
    )
    bundle.critic_optimizer.zero_grad(set_to_none=True)
    critic_loss.backward()
    nn.utils.clip_grad_norm_(
        bundle.critic.parameters(), config.maximum_gradient_norm
    )
    bundle.critic_optimizer.step()

    actor_updated = bool(
        actor_update_enabled_v721
        and update_index % config.policy_update_period == 0
    )
    actor_loss_value = 0.0
    mean_policy_action_abs = 0.0
    mean_policy_forward = 0.0
    predicted_policy_feasibility = 0.0
    actor_anchor_loss_v621 = 0.0
    mean_anchor_weight_v621 = 0.0
    mean_parent_action_delta_l2_v621 = 0.0
    maximum_context_over_base_hidden_norm_v621 = 0.0
    mean_acquisition_gate_v626 = 0.0
    exact_transport_parent_row_fraction_v626 = 0.0
    mean_acquisition_parent_action_delta_l2_v626 = 0.0
    mean_acquisition_option_residual_l2_v639 = 0.0
    maximum_acquisition_option_residual_l2_v639 = 0.0
    transport_self_imitation_audit: dict[str, Any] = {
        "selected_transition_fraction": 0.0,
        "mean_selected_progress_m": 0.0,
        "selected_originally_infeasible_transition_fraction": 0.0,
        "scaled_loss": 0.0,
    }
    if actor_updated:
        for module in (bundle.critic, bundle.feasibility):
            for parameter in module.parameters():
                parameter.requires_grad_(False)
        policy_action, log_probability = bundle.actor.sample(
            observation, controller
        )
        policy_q1, policy_q2 = bundle.critic(
            observation, controller, policy_action
        )
        policy_feasibility = torch.sigmoid(
            bundle.feasibility(observation, controller, policy_action)
        )
        actor_loss = (
            config.entropy_temperature * log_probability
            - torch.minimum(policy_q1, policy_q2)
            + config.actor_infeasibility_coefficient
            * (1.0 - policy_feasibility)
        ).mean()
        trust_region = bundle.actor_trust_region_v621
        if trust_region is not None:
            base_hidden = bundle.actor.base.trunk(observation)
            raw_context_hidden = bundle.actor.context_hidden_v621(
                observation, controller
            )
            phase_isolation = bundle.phase_isolated_acquisition_v626
            anchor_row_multiplier = None
            acquisition_gate = None
            if phase_isolation is not None:
                acquisition_gate, _distance, _alignment, _contact = (
                    bundle.actor.acquisition_gate_v626(observation)
                )
                _bounded_context, realized_ratio = (
                    phase_isolated_context_hidden_v626(
                        base_hidden,
                        raw_context_hidden,
                        acquisition_gate,
                        maximum_acquisition_ratio=(
                            phase_isolation
                            .maximum_acquisition_context_over_base_hidden_norm
                        ),
                    )
                )
                anchor_row_multiplier = 1.0 - acquisition_gate
                mean_acquisition_gate_v626 = float(
                    acquisition_gate.mean().item()
                )
                exact_transport_parent_row_fraction_v626 = float(
                    (acquisition_gate == 0.0).float().mean().item()
                )
            else:
                _bounded_context, realized_ratio = (
                    bounded_context_hidden_v621(
                        base_hidden,
                        raw_context_hidden,
                        maximum_ratio=(
                            trust_region
                            .maximum_context_over_base_hidden_norm
                        ),
                    )
                )
            candidate_deterministic_action = torch.tanh(
                bundle.actor.distribution(observation, controller).mean
            )
            with torch.no_grad():
                parent_action, _ = bundle.actor.base.sample(
                    observation, deterministic=True
                )
            anchor_loss, anchor_weight, parent_delta = (
                deterministic_action_anchor_v621(
                    candidate_deterministic_action,
                    parent_action,
                    controller,
                    backlog_scale=trust_region.low_backlog_scale,
                    row_multiplier=anchor_row_multiplier,
                )
            )
            if acquisition_gate is not None:
                per_row_parent_delta = (
                    candidate_deterministic_action - parent_action
                ).norm(dim=-1)
                mean_acquisition_parent_action_delta_l2_v626 = float(
                    (
                        acquisition_gate * per_row_parent_delta
                    ).sum().div(acquisition_gate.sum().clamp_min(1.0e-8)).item()
                )
            if bundle.phase_isolated_acquisition_option_v639 is not None:
                option_residual, option_gate = (
                    bundle.actor.acquisition_option_mean_residual_v639(
                        observation, controller
                    )
                )
                option_norm = option_residual.norm(dim=-1)
                mean_acquisition_option_residual_l2_v639 = float(
                    (option_gate * option_norm)
                    .sum()
                    .div(option_gate.sum().clamp_min(1.0e-8))
                    .item()
                )
                maximum_acquisition_option_residual_l2_v639 = float(
                    option_norm.max().item()
                )
            actor_loss = actor_loss + (
                trust_region.low_backlog_anchor_coefficient * anchor_loss
            )
            actor_anchor_loss_v621 = float(anchor_loss.item())
            mean_anchor_weight_v621 = float(anchor_weight.item())
            mean_parent_action_delta_l2_v621 = float(parent_delta.item())
            maximum_context_over_base_hidden_norm_v621 = float(
                realized_ratio.max().item()
            )
        if transport_self_imitation.coefficient > 0.0:
            deterministic_transport_action, _ = bundle.actor.sample(
                observation,
                controller,
                deterministic=True,
            )
            if (
                transport_self_imitation_target_mode
                == CHOSEN_POLICY_ACTION_SELF_IMITATION_MODE_V717
            ):
                (
                    transport_self_imitation_loss,
                    transport_self_imitation_audit,
                ) = causal_chosen_forward_transport_loss_v717(
                    deterministic_transport_action,
                    action,
                    tensor("original_task_progress_m"),
                    tensor("valid_contact"),
                    tensor("her_relabelled"),
                    feasible,
                    tensor("safety_violation"),
                    importance,
                    config=transport_self_imitation,
                )
            else:
                (
                    transport_self_imitation_loss,
                    transport_self_imitation_audit,
                ) = positive_effect_transport_self_imitation_loss_v707(
                    deterministic_transport_action,
                    tensor("applied_action"),
                    tensor("original_task_progress_m"),
                    tensor("valid_contact"),
                    tensor("her_relabelled"),
                    feasible,
                    tensor("safety_violation"),
                    importance,
                    config=transport_self_imitation,
                )
            actor_loss = actor_loss + transport_self_imitation_loss
        bundle.actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        nn.utils.clip_grad_norm_(
            bundle.actor.parameters(), config.maximum_gradient_norm
        )
        bundle.actor_optimizer.step()
        for module in (bundle.critic, bundle.feasibility):
            for parameter in module.parameters():
                parameter.requires_grad_(True)
        actor_loss_value = float(actor_loss.item())
        mean_policy_action_abs = float(policy_action.abs().mean().item())
        mean_policy_forward = float(policy_action[:, 0].mean().item())
        predicted_policy_feasibility = float(
            policy_feasibility.mean().item()
        )

    with torch.no_grad():
        for target_parameter, parameter in zip(
            bundle.target_critic.parameters(),
            bundle.critic.parameters(),
            strict=True,
        ):
            target_parameter.mul_(1.0 - config.target_tau).add_(
                parameter, alpha=config.target_tau
            )
        accuracy = (
            (feasibility_logit >= 0.0) == (feasible > 0.5)
        ).float().mean()

    metrics = GoalConditionedMarkovHerSACMetricsV614(
        update_index=update_index,
        critic_loss=float(critic_loss.item()),
        actor_loss=actor_loss_value,
        feasibility_loss=float(feasibility_loss.item()),
        feasibility_accuracy=float(accuracy.item()),
        mean_q_target=float(q_target.mean().item()),
        mean_q_data=float(torch.minimum(q1, q2).mean().item()),
        mean_policy_action_abs=mean_policy_action_abs,
        mean_policy_forward_action=mean_policy_forward,
        predicted_policy_feasibility=predicted_policy_feasibility,
        mean_controller_state_abs=float(controller.abs().mean().item()),
        her_relabel_fraction=float(np.mean(batch["her_relabelled"])),
        strict_success_source_fraction=float(
            np.mean(batch["strict_success_source"])
        ),
        mean_learning_reward=float(reward.mean().item()),
        actor_updated=actor_updated,
        actor_anchor_loss_v621=actor_anchor_loss_v621,
        mean_anchor_weight_v621=mean_anchor_weight_v621,
        mean_parent_action_delta_l2_v621=(
            mean_parent_action_delta_l2_v621
        ),
        maximum_context_over_base_hidden_norm_v621=(
            maximum_context_over_base_hidden_norm_v621
        ),
        actor_parent_base_frozen_v621=bool(
            bundle.actor_trust_region_v621 is not None
            and bundle.actor_trust_region_v621.freeze_parent_base
        ),
        mean_acquisition_gate_v626=mean_acquisition_gate_v626,
        exact_transport_parent_row_fraction_v626=(
            exact_transport_parent_row_fraction_v626
        ),
        mean_acquisition_parent_action_delta_l2_v626=(
            mean_acquisition_parent_action_delta_l2_v626
        ),
        mean_acquisition_option_residual_l2_v639=(
            mean_acquisition_option_residual_l2_v639
        ),
        maximum_acquisition_option_residual_l2_v639=(
            maximum_acquisition_option_residual_l2_v639
        ),
        positive_effect_transport_self_imitation_enabled_v707=bool(
            transport_self_imitation.coefficient > 0.0
            and transport_self_imitation_target_mode
            == MEASURED_EFFECT_SELF_IMITATION_MODE_V717
        ),
        positive_effect_transport_self_imitation_loss_v707=float(
            transport_self_imitation_audit["scaled_loss"]
            if transport_self_imitation_target_mode
            == MEASURED_EFFECT_SELF_IMITATION_MODE_V717
            else 0.0
        ),
        positive_effect_transport_self_imitation_selected_fraction_v707=float(
            transport_self_imitation_audit["selected_transition_fraction"]
            if transport_self_imitation_target_mode
            == MEASURED_EFFECT_SELF_IMITATION_MODE_V717
            else 0.0
        ),
        positive_effect_transport_self_imitation_mean_progress_m_v707=float(
            transport_self_imitation_audit["mean_selected_progress_m"]
            if transport_self_imitation_target_mode
            == MEASURED_EFFECT_SELF_IMITATION_MODE_V717
            else 0.0
        ),
        positive_effect_transport_self_imitation_originally_infeasible_fraction_v707=float(
            transport_self_imitation_audit[
                "selected_originally_infeasible_transition_fraction"
            ]
            if transport_self_imitation_target_mode
            == MEASURED_EFFECT_SELF_IMITATION_MODE_V717
            else 0.0
        ),
        causal_chosen_forward_self_imitation_enabled_v717=bool(
            transport_self_imitation.coefficient > 0.0
            and transport_self_imitation_target_mode
            == CHOSEN_POLICY_ACTION_SELF_IMITATION_MODE_V717
        ),
        causal_chosen_forward_self_imitation_loss_v717=float(
            transport_self_imitation_audit["scaled_loss"]
            if transport_self_imitation_target_mode
            == CHOSEN_POLICY_ACTION_SELF_IMITATION_MODE_V717
            else 0.0
        ),
        causal_chosen_forward_self_imitation_selected_fraction_v717=float(
            transport_self_imitation_audit["selected_transition_fraction"]
            if transport_self_imitation_target_mode
            == CHOSEN_POLICY_ACTION_SELF_IMITATION_MODE_V717
            else 0.0
        ),
        causal_chosen_forward_self_imitation_mean_progress_m_v717=float(
            transport_self_imitation_audit["mean_selected_progress_m"]
            if transport_self_imitation_target_mode
            == CHOSEN_POLICY_ACTION_SELF_IMITATION_MODE_V717
            else 0.0
        ),
        causal_chosen_forward_self_imitation_filtered_mdp_fraction_v717=float(
            transport_self_imitation_audit[
                "selected_originally_infeasible_transition_fraction"
            ]
            if transport_self_imitation_target_mode
            == CHOSEN_POLICY_ACTION_SELF_IMITATION_MODE_V717
            else 0.0
        ),
        actor_update_enabled_v721=actor_update_enabled_v721,
    )
    for name, value in asdict(metrics).items():
        if isinstance(value, float) and not np.isfinite(value):
            raise RuntimeError(f"V614 update metric {name} is non-finite")
    return metrics


__all__ = [
    "GOAL_CONDITIONED_MARKOV_HER_CHECKPOINT_FORMAT_V614",
    "GOAL_CONDITIONED_MARKOV_HER_REPLAY_FORMAT_V614",
    "GOAL_CONDITIONED_MARKOV_HER_SAC_FORMAT_V614",
    "ActionFeasibilityMarkovV614",
    "GoalConditionedMarkovActorV614",
    "GoalConditionedMarkovHerReplayV614",
    "GoalConditionedMarkovHerSACBundleV614",
    "GoalConditionedMarkovHerSACMetricsV614",
    "TwinGoalConditionedMarkovCriticV614",
    "goal_conditioned_markov_her_sac_update_v614",
    "initialize_goal_conditioned_markov_her_sac_v614",
    "warm_start_markov_her_sac_v614_from_v43",
]
