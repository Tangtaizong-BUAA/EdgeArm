"""Markov bounded-action curriculum primitives for Home acquisition.

V646 placed a stateful slew limiter after the policy.  Its previous selected
action was not part of the policy state, so the resulting process was not
Markov from the acquisition actor's perspective.  V654 removes that hidden
state from acquisition learning: the stochastic actor itself emits a small,
memoryless task-frame action.  Safety projection remains inside the audited
V22/V597 plant and is fully represented by the existing controller state.

The reverse start curriculum below supplies reset states only.  It never
supplies an action, route, waypoint, future observation, or behavior-cloning
target.  Every non-Home tier is learning-only and is categorically ineligible
for wrist/VLA data export.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Sequence

import numpy as np
import torch

from .causal_smooth_relay_v646 import PrecontactBlockCausalStabilizerV646
from .goal_conditioned_her_sac_v43 import goal_neutral_privileged_state_v43
from .privileged_effect_state_v1 import build_privileged_effect_state_v1
from .relay_dual_goal_her_sac_v643 import (
    ACTION_DIM_V43,
    RelayAcquisitionActorV643,
    acquisition_gate_numpy_v643,
)
from .relay_taskframe_error_her_sac_v652 import (
    RelayTaskframeErrorHerSACBundleV652,
)
from .sim2real_env_v10 import RealisticEdgeArmEnvV10


BOUNDED_TASKFRAME_ACQUISITION_FORMAT_V654 = (
    "edgearm-v654-markov-bounded-taskframe-acquisition-v1"
)
ACQUISITION_CURRICULUM_FORMAT_V654 = (
    "edgearm-v654-fine-sequential-home-acquisition-curriculum-v1"
)
ACQUISITION_TERMINATOR_FORMAT_V654 = (
    "edgearm-v654-stable-transport-ready-acquisition-terminator-v1"
)


@dataclass(frozen=True)
class BoundedAcquisitionActionConfigV654:
    action_absolute: tuple[float, float, float] = (0.35, 0.35, 0.25)

    def validate(self) -> None:
        values = np.asarray(self.action_absolute, dtype=np.float64)
        if (
            values.shape != (ACTION_DIM_V43,)
            or not np.all(np.isfinite(values))
            or np.any(values <= 0.0)
            or np.any(values > 1.0)
        ):
            raise ValueError("V654 acquisition action bounds are invalid")


class BoundedRelayAcquisitionActorV654(RelayAcquisitionActorV643):
    """V643 network with an explicit memoryless action support."""

    def __init__(
        self,
        hidden_dim: int = 256,
        *,
        action_config: BoundedAcquisitionActionConfigV654 | None = None,
    ) -> None:
        super().__init__(hidden_dim)
        self.action_config_v654 = (
            action_config or BoundedAcquisitionActionConfigV654()
        )
        self.action_config_v654.validate()

    def sample(
        self,
        observation: torch.Tensor,
        *,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        distribution = self.distribution(observation)
        pre_tanh = (
            distribution.mean if deterministic else distribution.rsample()
        )
        unit_action = torch.tanh(pre_tanh)
        scale = torch.as_tensor(
            self.action_config_v654.action_absolute,
            dtype=unit_action.dtype,
            device=unit_action.device,
        )
        action = unit_action * scale
        if deterministic:
            log_probability = torch.zeros(
                action.shape[0], device=action.device, dtype=action.dtype
            )
        else:
            tanh_correction = torch.log(
                1.0 - unit_action.square() + 1.0e-6
            )
            scale_correction = torch.log(scale).sum()
            log_probability = (
                distribution.log_prob(pre_tanh) - tanh_correction
            ).sum(dim=-1) - scale_correction
        return action, log_probability


def install_bounded_acquisition_actor_v654(
    bundle: RelayTaskframeErrorHerSACBundleV652,
    *,
    action_config: BoundedAcquisitionActionConfigV654 | None = None,
) -> dict[str, Any]:
    """Replace only the V652 acquisition action parametrization."""

    if type(bundle) is not RelayTaskframeErrorHerSACBundleV652:
        raise TypeError("V654 requires the exact V652 bundle")
    selected = action_config or BoundedAcquisitionActionConfigV654()
    selected.validate()
    old_actor = bundle.policy.acquisition_actor
    device = next(old_actor.parameters()).device
    bounded = BoundedRelayAcquisitionActorV654(
        bundle.config.hidden_dim,
        action_config=selected,
    ).to(device)
    bounded.load_state_dict(old_actor.state_dict(), strict=True)
    bundle.policy.acquisition_actor = bounded
    bundle.kernel.actor_optimizer = torch.optim.Adam(
        bounded.parameters(), lr=bundle.config.actor_learning_rate
    )
    return {
        "format": BOUNDED_TASKFRAME_ACQUISITION_FORMAT_V654,
        "action_absolute": list(selected.action_absolute),
        "stateful_slew_filter_present": False,
        "previous_selected_action_required_by_policy": False,
        "critic_and_actor_action_support_match_online_execution": True,
        "transport_actor_changed": False,
        "expert_action_used": False,
        "waypoint_or_path_used": False,
        "behavior_cloning_steps": 0,
        "production_admission": False,
    }


@dataclass(frozen=True)
class AcquisitionStartTierV654:
    index: int
    code: str
    home_to_precontact_fraction: float
    exact_home: bool
    privileged_training_reset: bool
    eligible_for_final_data: bool

    def validate(self) -> None:
        if type(self.index) is not int or self.index < 0:
            raise ValueError("V654 tier index is invalid")
        if not self.code or not self.code.replace("_", "").isalnum():
            raise ValueError("V654 tier code is invalid")
        if not 0.0 <= self.home_to_precontact_fraction <= 0.95:
            raise ValueError("V654 tier fraction is invalid")
        if self.exact_home != (self.home_to_precontact_fraction == 0.0):
            raise ValueError("V654 Home identity and fraction disagree")
        if self.eligible_for_final_data != self.exact_home:
            raise ValueError("V654 only exact Home may be final-data eligible")
        if self.privileged_training_reset == self.exact_home:
            raise ValueError("V654 reset privilege identity disagrees")


_FRACTIONS_V654 = (
    0.95,
    0.90,
    0.85,
    0.80,
    0.75,
    0.70,
    0.65,
    0.55,
    0.45,
    0.35,
    0.25,
    0.15,
    0.05,
    0.0,
)


def _tier_code_v654(fraction: float) -> str:
    return "home" if fraction == 0.0 else f"approach_{int(round(100 * fraction)):02d}"


ACQUISITION_START_TIERS_V654 = tuple(
    AcquisitionStartTierV654(
        index=index,
        code=_tier_code_v654(fraction),
        home_to_precontact_fraction=fraction,
        exact_home=bool(fraction == 0.0),
        privileged_training_reset=bool(fraction != 0.0),
        eligible_for_final_data=bool(fraction == 0.0),
    )
    for index, fraction in enumerate(_FRACTIONS_V654)
)
for _tier in ACQUISITION_START_TIERS_V654:
    _tier.validate()


def _tier_records_v654(
    records: Sequence[dict[str, Any]], tier: AcquisitionStartTierV654
) -> list[dict[str, Any]]:
    return [
        record
        for record in records
        if record.get("acquisition_start_tier_v654") == tier.code
    ]


def acquisition_tier_mastered_v654(
    records: Sequence[dict[str, Any]],
    tier: AcquisitionStartTierV654,
) -> tuple[bool, dict[str, Any]]:
    """Require repeatable success and executable actions before promotion."""

    rows = _tier_records_v654(records, tier)
    recent = rows[-5:]
    successes = sum(
        int(bool(row.get("acquisition_curriculum_success_v654")))
        for row in recent
    )
    feasible = [
        float(row.get("action_feasible_fraction_v654", 0.0))
        for row in recent
    ]
    median_feasible = float(np.median(feasible)) if feasible else 0.0
    mastered = bool(
        len(recent) >= 5 and successes >= 4 and median_feasible >= 0.75
    )
    return mastered, {
        "episode_count": len(rows),
        "recent_window_size": 5,
        "recent_episode_count": len(recent),
        "recent_success_count": successes,
        "recent_success_rate": successes / len(recent) if recent else 0.0,
        "recent_median_action_feasible_fraction": median_feasible,
        "mastered": mastered,
    }


def select_acquisition_start_tier_v654(
    *,
    episode_index: int,
    records: Sequence[dict[str, Any]],
) -> tuple[AcquisitionStartTierV654, dict[str, Any]]:
    """Advance one fine acquisition frontier at a time."""

    if type(episode_index) is not int or episode_index < 1:
        raise ValueError("V654 episode index must be positive")
    evidence: dict[str, Any] = {}
    frontier = ACQUISITION_START_TIERS_V654[-1]
    mastered_codes: list[str] = []
    for tier in ACQUISITION_START_TIERS_V654[:-1]:
        mastered, row = acquisition_tier_mastered_v654(records, tier)
        evidence[tier.code] = row
        if mastered:
            mastered_codes.append(tier.code)
            continue
        frontier = tier
        break
    home = ACQUISITION_START_TIERS_V654[-1]
    _, evidence[home.code] = acquisition_tier_mastered_v654(records, home)

    if episode_index % 6 == 0:
        selected = home
        reason = "periodic_exact_home_probe"
    elif episode_index % 5 == 0 and frontier.index > 0:
        selected = ACQUISITION_START_TIERS_V654[frontier.index - 1]
        reason = "mastered_predecessor_retention"
    else:
        selected = frontier
        reason = "fine_sequential_acquisition_frontier"
    return selected, {
        "format": ACQUISITION_CURRICULUM_FORMAT_V654,
        "episode_index": episode_index,
        "selected_tier": asdict(selected),
        "selection_reason": reason,
        "frontier_tier": frontier.code,
        "mastered_tiers": mastered_codes,
        "evidence_before_selection": evidence,
        "promotion_rule": {
            "recent_window": 5,
            "minimum_successes": 4,
            "minimum_median_action_feasible_fraction": 0.75,
        },
        "exact_home_probe_interval": 6,
        "predecessor_retention_interval": 5,
        "non_home_tiers_are_reset_states_not_demonstrations": True,
        "non_home_tiers_final_data_eligible": False,
        "production_admission": False,
    }


@dataclass(frozen=True)
class AcquisitionTerminationConfigV654:
    stable_steps: int = 5
    maximum_gate: float = 0.005

    def validate(self) -> None:
        if type(self.stable_steps) is not int or self.stable_steps < 2:
            raise ValueError("V654 acquisition stable-step count is invalid")
        if not math.isfinite(self.maximum_gate) or not 0.0 <= self.maximum_gate <= 0.02:
            raise ValueError("V654 acquisition gate threshold is invalid")


class AcquisitionCurriculumTerminatorV654:
    """Compose causal stabilization with stable transport-ready termination."""

    def __init__(
        self,
        bundle: RelayTaskframeErrorHerSACBundleV652,
        *,
        config: AcquisitionTerminationConfigV654 | None = None,
    ) -> None:
        self.bundle = bundle
        self.config = config or AcquisitionTerminationConfigV654()
        self.config.validate()
        self.stabilizer = PrecontactBlockCausalStabilizerV646()
        self._streak = 0
        self._success = False
        self._decision_count = 0
        self._distance_values: list[float] = []
        self._gate_values: list[float] = []
        self._alignment_values: list[float] = []

    def __call__(
        self,
        env: RealisticEdgeArmEnvV10,
        info: dict[str, Any],
        block_before_xy_m: np.ndarray,
        terminated: bool,
        truncated: bool,
    ) -> tuple[bool, bool, dict[str, Any]]:
        terminated, truncated, causal = self.stabilizer(
            env,
            info,
            block_before_xy_m,
            terminated,
            truncated,
        )
        privileged = build_privileged_effect_state_v1(env)
        neutral = goal_neutral_privileged_state_v43(privileged)
        gate, distance, alignment, contact = acquisition_gate_numpy_v643(
            neutral[None],
            np.asarray(env.target_xy, dtype=np.float32)[None],
            phase_config=self.bundle.policy.phase_config,
        )
        gate_value = float(gate[0])
        distance_value = float(distance[0])
        alignment_value = float(alignment[0])
        # Contact forces the V626 gate to zero.  For curriculum promotion we
        # additionally require the explicit distance/alignment/height
        # transport-ready geometry, so an arbitrary collision cannot pass.
        phase = self.bundle.policy.phase_config
        geometrically_ready = bool(
            distance_value <= phase.exact_transport_distance_m
            and alignment_value >= phase.exact_transport_alignment
            and float(env.tool_xyz()[2])
            <= phase.exact_transport_maximum_tool_height_m
            and gate_value <= self.config.maximum_gate
        )
        if geometrically_ready and not bool(info.get("safety_stop", False)):
            self._streak += 1
        else:
            self._streak = 0
        if self._streak >= self.config.stable_steps:
            self._success = True
            terminated = True
            truncated = False
            info["success"] = False
            info["terminal_failure"] = False
            info["terminated"] = True
            info["truncated"] = False
            info["terminal_reason"] = "v654_stable_transport_ready"
        self._decision_count += 1
        self._distance_values.append(distance_value)
        self._gate_values.append(gate_value)
        self._alignment_values.append(alignment_value)
        audit = {
            **causal,
            "format": ACQUISITION_TERMINATOR_FORMAT_V654,
            "decision_index": self._decision_count - 1,
            "acquisition_gate": gate_value,
            "tool_precontact_distance_m": distance_value,
            "precontact_alignment": alignment_value,
            "contact_state": bool(contact[0]),
            "geometrically_transport_ready": geometrically_ready,
            "stable_ready_streak": self._streak,
            "acquisition_curriculum_success": self._success,
            "full_task_success_claimed": False,
            "production_admission": False,
        }
        info["acquisition_curriculum_terminator_v654"] = dict(audit)
        return bool(terminated), bool(truncated), audit

    def summary(self) -> dict[str, Any]:
        if not self._distance_values:
            raise RuntimeError("V654 acquisition terminator has no decisions")
        return {
            "format": ACQUISITION_TERMINATOR_FORMAT_V654,
            "decision_count": self._decision_count,
            "success": self._success,
            "required_stable_steps": self.config.stable_steps,
            "initial_tool_precontact_distance_m": self._distance_values[0],
            "minimum_tool_precontact_distance_m": min(self._distance_values),
            "final_tool_precontact_distance_m": self._distance_values[-1],
            "minimum_acquisition_gate": min(self._gate_values),
            "final_acquisition_gate": self._gate_values[-1],
            "maximum_precontact_alignment": max(self._alignment_values),
            "causal_stabilizer": self.stabilizer.summary(),
            "full_task_success_claimed": False,
            "curriculum_only": True,
            "bulk_vla_data_use_allowed": False,
            "production_admission": False,
        }


__all__ = [
    "ACQUISITION_CURRICULUM_FORMAT_V654",
    "ACQUISITION_START_TIERS_V654",
    "ACQUISITION_TERMINATOR_FORMAT_V654",
    "BOUNDED_TASKFRAME_ACQUISITION_FORMAT_V654",
    "AcquisitionCurriculumTerminatorV654",
    "AcquisitionStartTierV654",
    "AcquisitionTerminationConfigV654",
    "BoundedAcquisitionActionConfigV654",
    "BoundedRelayAcquisitionActorV654",
    "acquisition_tier_mastered_v654",
    "install_bounded_acquisition_actor_v654",
    "select_acquisition_start_tier_v654",
]
