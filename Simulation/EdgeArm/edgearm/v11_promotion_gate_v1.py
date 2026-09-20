"""Single-source promotion gate for V11 synthetic demonstrations.

The evaluator, collector and dataset auditor must agree on whether an episode
is eligible for ACT action supervision.  This module is deliberately pure: it
accepts measured episode evidence and returns a deterministic decision without
reading simulator state or trusting a precomputed success attribute.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from typing import Any

import numpy as np


V11_PROMOTION_GATE_FORMAT = "edgearm-v11-promotion-gate-v2"
CAUSAL_EXECUTION_PHASES = frozenset(
    {"sustained_push", "fine_correction", "re_push"}
)


@dataclass(frozen=True)
class V11PromotionGateConfig:
    minimum_effective_block_displacement_m: float = 0.010
    minimum_causal_valid_push_transitions: int = 1
    minimum_causal_valid_push_directional_displacement_m: float = 0.001
    minimum_causal_valid_push_target_progress_m: float = 0.001

    def __post_init__(self) -> None:
        if (
            not isinstance(self.minimum_causal_valid_push_transitions, int)
            or self.minimum_causal_valid_push_transitions < 1
        ):
            raise ValueError(
                "minimum_causal_valid_push_transitions must be a positive integer"
            )
        for name in (
            "minimum_effective_block_displacement_m",
            "minimum_causal_valid_push_directional_displacement_m",
            "minimum_causal_valid_push_target_progress_m",
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")

    @property
    def profile_hash(self) -> str:
        encoded = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class V11PromotionEvidence:
    strict_success: bool
    terminated: bool
    truncated: bool
    collision_failure_any: bool
    tool_block_contact_steps: int
    valid_push_side_contact_steps: int
    episode_net_block_displacement_m: float
    minimum_pusher_desk_signed_distance_m: float
    pusher_desk_hard_floor_m: float
    runtime_guard_static_infeasible_events: int
    runtime_guard_dynamic_infeasible_events: int
    runtime_guard_safety_stop: bool
    precontact_plan_infeasible_steps_total: int
    precontact_plan_infeasible_steps_before_first_causal_push: int
    precontact_completed_once: bool
    causal_valid_push_transition_steps: int
    causal_valid_push_directional_displacement_m: float
    causal_valid_push_target_progress_m: float
    first_causal_valid_push_effect_step: int
    strict_success_effect_step: int
    causal_push_applied_parent_after_precontact_completion: bool
    invalid_geometry_contact_steps: int
    forbidden_pusher_desk_penetration_substeps: int = 0
    forbidden_non_tool_robot_desk_penetration_any: bool = False

    def __post_init__(self) -> None:
        integer_fields = (
            "tool_block_contact_steps",
            "valid_push_side_contact_steps",
            "runtime_guard_static_infeasible_events",
            "runtime_guard_dynamic_infeasible_events",
            "precontact_plan_infeasible_steps_total",
            "precontact_plan_infeasible_steps_before_first_causal_push",
            "causal_valid_push_transition_steps",
            "forbidden_pusher_desk_penetration_substeps",
            "invalid_geometry_contact_steps",
        )
        for name in integer_fields:
            value = getattr(self, name)
            if not isinstance(value, (int, np.integer)) or int(value) < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        for name in (
            "episode_net_block_displacement_m",
            "minimum_pusher_desk_signed_distance_m",
            "pusher_desk_hard_floor_m",
            "causal_valid_push_directional_displacement_m",
            "causal_valid_push_target_progress_m",
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value):
                raise ValueError(f"{name} must be finite")
        if self.episode_net_block_displacement_m < 0.0:
            raise ValueError("episode_net_block_displacement_m must be non-negative")
        for name in (
            "first_causal_valid_push_effect_step",
            "strict_success_effect_step",
        ):
            value = getattr(self, name)
            if not isinstance(value, (int, np.integer)) or int(value) < -1:
                raise ValueError(f"{name} must be an integer greater than or equal to -1")
        if self.causal_valid_push_directional_displacement_m < 0.0:
            raise ValueError(
                "causal_valid_push_directional_displacement_m must be non-negative"
            )
        if self.causal_valid_push_target_progress_m < 0.0:
            raise ValueError(
                "causal_valid_push_target_progress_m must be non-negative"
            )
        if self.pusher_desk_hard_floor_m < 0.0:
            raise ValueError("pusher_desk_hard_floor_m must be non-negative")


def evaluate_v11_promotion(
    evidence: V11PromotionEvidence,
    config: V11PromotionGateConfig | None = None,
) -> dict[str, Any]:
    """Return a fail-closed, recomputable action-supervision decision."""

    profile = config or V11PromotionGateConfig()
    checks = {
        "strict_success": bool(evidence.strict_success),
        "terminated": bool(evidence.terminated),
        "not_truncated": not bool(evidence.truncated),
        "no_collision_failure": not bool(evidence.collision_failure_any),
        "tool_block_contact_observed": evidence.tool_block_contact_steps > 0,
        "valid_push_side_contact_observed": (
            evidence.valid_push_side_contact_steps > 0
        ),
        "effective_block_displacement": (
            evidence.episode_net_block_displacement_m
            >= profile.minimum_effective_block_displacement_m
        ),
        "pusher_desk_hard_floor_preserved": (
            evidence.minimum_pusher_desk_signed_distance_m
            >= evidence.pusher_desk_hard_floor_m
        ),
        "runtime_static_guard_feasible": (
            evidence.runtime_guard_static_infeasible_events == 0
        ),
        "runtime_dynamic_guard_feasible": (
            evidence.runtime_guard_dynamic_infeasible_events == 0
        ),
        "no_runtime_guard_safety_stop": not bool(
            evidence.runtime_guard_safety_stop
        ),
        "no_precausal_precontact_plan_infeasible_step": (
            evidence.precontact_plan_infeasible_steps_before_first_causal_push == 0
        ),
        "precontact_completed_once": bool(
            evidence.precontact_completed_once
        ),
        "causal_valid_push_effect_observed": (
            evidence.causal_valid_push_transition_steps
            >= profile.minimum_causal_valid_push_transitions
            and evidence.causal_valid_push_directional_displacement_m
            >= profile.minimum_causal_valid_push_directional_displacement_m
            and evidence.causal_valid_push_target_progress_m
            >= profile.minimum_causal_valid_push_target_progress_m
            and evidence.causal_push_applied_parent_after_precontact_completion
        ),
        "causal_push_precedes_strict_success": (
            evidence.first_causal_valid_push_effect_step >= 0
            and evidence.strict_success_effect_step >= 0
            and evidence.first_causal_valid_push_effect_step
            < evidence.strict_success_effect_step
        ),
        "no_invalid_geometry_contact": evidence.invalid_geometry_contact_steps == 0,
        "no_forbidden_pusher_desk_penetration": (
            evidence.forbidden_pusher_desk_penetration_substeps == 0
        ),
        "no_forbidden_non_tool_robot_desk_penetration": not bool(
            evidence.forbidden_non_tool_robot_desk_penetration_any
        ),
    }
    failed = [name for name, passed in checks.items() if not passed]
    return {
        "format": V11_PROMOTION_GATE_FORMAT,
        "profile_hash": profile.profile_hash,
        "configuration": asdict(profile),
        "passed": not failed,
        "failed_checks": failed,
        "checks": checks,
        "evidence": asdict(evidence),
        "action_label_eligible": not failed,
        "physical_samples": 0,
        "physical_trials": 0,
        "physical_validation": False,
    }


class V11PromotionAccumulator:
    """Accumulate command-lineage-aware causal push evidence.

    A post-step push effect is eligible only when its actually applied delayed
    command was submitted after the controller had completed pre-contact.  The
    class is shared by the evaluator and collector so their episode evidence
    cannot drift; the HDF5 auditor still recomputes the same facts independently.
    """

    def __init__(self) -> None:
        self.precontact_plan_infeasible_steps_total = 0
        self.precontact_plan_infeasible_steps_before_first_causal_push = 0
        self.precontact_completed_once = False
        self.causal_valid_push_transition_steps = 0
        self.causal_valid_push_directional_displacement_m = 0.0
        self.causal_valid_push_target_progress_m = 0.0
        self.first_causal_valid_push_effect_step = -1
        self.strict_success_effect_step = -1
        self._command_precontact_completed: dict[int, bool] = {}

    def update(
        self,
        *,
        submitted_command_id: int,
        applied_command_id: int,
        decision_precontact_complete: bool,
        decision_precontact_plan_feasible: bool,
        effect_step: int,
        effect_execution_phase_name: str,
        effect_execution_phase_valid: bool,
        effect_valid_push_side_contact_any: bool,
        effect_push_directional_block_displacement_m: float,
        effect_progress_toward_target_m: float,
        effect_strict_success: bool,
    ) -> dict[str, bool]:
        directional = float(effect_push_directional_block_displacement_m)
        progress = float(effect_progress_toward_target_m)
        if not np.isfinite(directional) or not np.isfinite(progress):
            raise ValueError("causal push displacement and progress must be finite")
        if effect_step < 0:
            raise ValueError("effect_step must be non-negative")

        completion_event = bool(
            decision_precontact_complete and not self.precontact_completed_once
        )
        self.precontact_completed_once = bool(
            self.precontact_completed_once or decision_precontact_complete
        )
        plan_infeasible = not decision_precontact_plan_feasible
        self.precontact_plan_infeasible_steps_total += int(plan_infeasible)
        if self.causal_valid_push_transition_steps == 0:
            self.precontact_plan_infeasible_steps_before_first_causal_push += int(
                plan_infeasible
            )
        self._command_precontact_completed[int(submitted_command_id)] = (
            self.precontact_completed_once
        )
        applied_parent_completed = bool(
            self._command_precontact_completed.get(int(applied_command_id), False)
        )
        effect_push_signature = bool(
            effect_execution_phase_valid
            and effect_execution_phase_name in CAUSAL_EXECUTION_PHASES
            and effect_valid_push_side_contact_any
            and directional > 0.0
            and progress > 0.0
        )
        causal_transition = bool(effect_push_signature and applied_parent_completed)
        if causal_transition:
            self.causal_valid_push_transition_steps += 1
            self.causal_valid_push_directional_displacement_m += directional
            self.causal_valid_push_target_progress_m += progress
            if self.first_causal_valid_push_effect_step < 0:
                self.first_causal_valid_push_effect_step = int(effect_step)
        if effect_strict_success and self.strict_success_effect_step < 0:
            self.strict_success_effect_step = int(effect_step)
        return {
            "diagnostic_teacher_precontact_completion_event": completion_event,
            "diagnostic_teacher_precontact_completed_once": (
                self.precontact_completed_once
            ),
            "effect_applied_parent_precontact_completed_once": (
                applied_parent_completed
            ),
            "effect_causal_valid_push_transition": causal_transition,
        }

    def evidence_fields(self) -> dict[str, Any]:
        return {
            "precontact_plan_infeasible_steps_total": (
                self.precontact_plan_infeasible_steps_total
            ),
            "precontact_plan_infeasible_steps_before_first_causal_push": (
                self.precontact_plan_infeasible_steps_before_first_causal_push
            ),
            "precontact_completed_once": self.precontact_completed_once,
            "causal_valid_push_transition_steps": (
                self.causal_valid_push_transition_steps
            ),
            "causal_valid_push_directional_displacement_m": (
                self.causal_valid_push_directional_displacement_m
            ),
            "causal_valid_push_target_progress_m": (
                self.causal_valid_push_target_progress_m
            ),
            "first_causal_valid_push_effect_step": (
                self.first_causal_valid_push_effect_step
            ),
            "strict_success_effect_step": self.strict_success_effect_step,
            "causal_push_applied_parent_after_precontact_completion": bool(
                self.causal_valid_push_transition_steps > 0
            ),
        }


__all__ = [
    "V11_PROMOTION_GATE_FORMAT",
    "V11PromotionAccumulator",
    "V11PromotionEvidence",
    "V11PromotionGateConfig",
    "evaluate_v11_promotion",
]
