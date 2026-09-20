"""Causal-motion admission and smooth acquisition control for V643.

V645 exposed a simulator failure that ordinary task reward cannot distinguish:
the block moved centimetres before any robot/block contact.  V646 therefore
adds two independent safeguards without supplying an expert path:

* a stateful, task-action slew limiter for the acquisition option; and
* a fail-closed causal audit that rejects pre-contact block motion.

The frozen transport actor remains bit-exact whenever the V626 acquisition
gate is zero.  The filter is part of the low-level execution environment, not
an action label or privileged waypoint source.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Any

import mujoco
import numpy as np

from .goal_conditioned_her_sac_v43 import goal_neutral_privileged_state_v43
from .goal_conditioned_markov_her_sac_v614 import (
    GoalConditionedMarkovHerReplayV614,
)
from .phase_isolated_acquisition_v626 import (
    PhaseIsolatedAcquisitionConfigV626,
)
from .privileged_effect_state_v1 import privileged_effect_state_slices_v1
from .relay_dual_goal_her_sac_v643 import (
    RelayDualGoalHerSACConfigV643,
    acquisition_gate_numpy_v643,
    sample_relay_acquisition_batch_v643,
)
from .sim2real_env_v10 import RealisticEdgeArmEnvV10


CAUSAL_SMOOTH_RELAY_FORMAT_V646 = "edgearm-v646-causal-motion-smooth-acquisition-relay-v1"
CAUSAL_MOTION_AUDIT_FORMAT_V646 = "edgearm-v646-precontact-causal-motion-audit-v1"

_SLICES_V646 = privileged_effect_state_slices_v1()


@dataclass(frozen=True)
class CausalSmoothRelayConfigV646:
    """Execution and replay safeguards inferred from the V645 failure."""

    maximum_acquisition_action_absolute: tuple[float, float, float] = (
        0.35,
        0.35,
        0.25,
    )
    maximum_acquisition_action_delta: tuple[float, float, float] = (
        0.06,
        0.06,
        0.04,
    )
    unexplained_precontact_step_displacement_m: float = 0.00025
    unexplained_precontact_episode_displacement_m: float = 0.00100
    uncausal_motion_penalty: float = 8.0
    minimum_physics_substeps: int = 16

    def validate(self) -> None:
        absolute = np.asarray(self.maximum_acquisition_action_absolute, dtype=np.float64)
        delta = np.asarray(self.maximum_acquisition_action_delta, dtype=np.float64)
        if (
            absolute.shape != (3,)
            or delta.shape != (3,)
            or not np.all(np.isfinite(np.r_[absolute, delta]))
            or np.any(absolute <= 0.0)
            or np.any(absolute > 1.0)
            or np.any(delta <= 0.0)
            or np.any(delta > absolute)
        ):
            raise ValueError("V646 acquisition action bounds are invalid")
        positive = (
            self.unexplained_precontact_step_displacement_m,
            self.unexplained_precontact_episode_displacement_m,
            self.uncausal_motion_penalty,
        )
        if any(not math.isfinite(value) or value <= 0.0 for value in positive):
            raise ValueError("V646 causal thresholds must be positive")
        if (
            self.unexplained_precontact_episode_displacement_m
            < self.unexplained_precontact_step_displacement_m
        ):
            raise ValueError("V646 cumulative causal threshold is too small")
        if type(self.minimum_physics_substeps) is not int or self.minimum_physics_substeps < 8:
            raise ValueError("V646 physics substep floor is invalid")


def strengthened_relay_config_v646(
    base: RelayDualGoalHerSACConfigV643,
) -> RelayDualGoalHerSACConfigV643:
    """Increase penalties for exactly the projection/IK failure seen in V645."""

    base.validate()
    result = replace(
        base,
        action_penalty=max(base.action_penalty, 0.020),
        projection_penalty=max(base.projection_penalty, 0.60),
        infeasible_action_penalty=max(base.infeasible_action_penalty, 1.50),
        actor_infeasibility_coefficient=max(base.actor_infeasibility_coefficient, 1.00),
    )
    result.validate()
    return result


class AcquisitionActionSlewFilterV646:
    """Bound acquisition commands while preserving exact transport at gate 0."""

    def __init__(
        self,
        *,
        phase_config: PhaseIsolatedAcquisitionConfigV626,
        config: CausalSmoothRelayConfigV646 | None = None,
    ) -> None:
        phase_config.validate()
        self.phase_config = phase_config
        self.config = config or CausalSmoothRelayConfigV646()
        self.config.validate()
        self._previous_selected_action = np.zeros(3, dtype=np.float32)

    def __call__(
        self,
        proposed_action: np.ndarray,
        privileged_state: np.ndarray,
        desired_goal: np.ndarray,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        proposed = np.asarray(proposed_action, dtype=np.float32)
        state = np.asarray(privileged_state, dtype=np.float32)
        goal = np.asarray(desired_goal, dtype=np.float32)
        if (
            proposed.shape != (3,)
            or state.ndim != 1
            or goal.shape != (2,)
            or not np.all(np.isfinite(np.r_[proposed, state, goal]))
        ):
            raise ValueError("V646 action-filter input is invalid")
        neutral = goal_neutral_privileged_state_v43(state)
        gate, distance, alignment, contact = acquisition_gate_numpy_v643(
            neutral[None],
            goal[None],
            phase_config=self.phase_config,
        )
        acquisition_gate = float(gate[0])
        if acquisition_gate == 0.0:
            selected = proposed.copy()
            bounded = proposed.copy()
            slewed = proposed.copy()
            source = "v646_exact_frozen_transport_passthrough"
        else:
            maximum = np.asarray(
                self.config.maximum_acquisition_action_absolute,
                dtype=np.float32,
            )
            maximum_delta = np.asarray(
                self.config.maximum_acquisition_action_delta,
                dtype=np.float32,
            )
            bounded = np.clip(proposed, -maximum, maximum).astype(np.float32)
            slewed = self._previous_selected_action + np.clip(
                bounded - self._previous_selected_action,
                -maximum_delta,
                maximum_delta,
            )
            selected = proposed + np.float32(acquisition_gate) * (slewed - proposed)
            selected = np.clip(selected, -1.0, 1.0).astype(np.float32)
            source = "v646_acquisition_action_slew_filter"
        previous = self._previous_selected_action.copy()
        self._previous_selected_action = selected.copy()
        intervened = not np.array_equal(selected, proposed)
        return selected, {
            "format": CAUSAL_SMOOTH_RELAY_FORMAT_V646,
            "selected_source": source,
            "intervened": intervened,
            "acquisition_gate": acquisition_gate,
            "tool_precontact_distance_m": float(distance[0]),
            "precontact_alignment": float(alignment[0]),
            "contact_state": bool(contact[0]),
            "proposed_action": proposed.tolist(),
            "bounded_action": bounded.tolist(),
            "slewed_action": slewed.tolist(),
            "previous_selected_action": previous.tolist(),
            "selected_action": selected.tolist(),
            "transport_gate_zero_bit_exact": bool(
                acquisition_gate != 0.0 or np.array_equal(selected, proposed)
            ),
            "expert_action_used": False,
            "waypoint_or_path_used": False,
            "production_admission": False,
        }


class PrecontactBlockCausalStabilizerV646:
    """Preserve the flat-table invariant until real contact is observed.

    MuJoCo's soft friction constraints can move the free block even when every
    robot geometry has positive clearance.  In the current obstacle-free task
    there is no physical cause for pre-contact XY motion.  This stabilizer
    restores only that unforced XY drift; the first raw contact permanently
    releases the block and all contact dynamics remain untouched.
    """

    def __init__(self) -> None:
        self._contact_seen = False
        self._decision_count = 0
        self._restore_count = 0
        self._cumulative_restored_displacement_m = 0.0
        self._maximum_restored_step_displacement_m = 0.0

    def __call__(
        self,
        env: RealisticEdgeArmEnvV10,
        info: dict[str, Any],
        block_before_xy_m: np.ndarray,
        terminated: bool,
        truncated: bool,
    ) -> tuple[bool, bool, dict[str, Any]]:
        if type(env) is not RealisticEdgeArmEnvV10:
            raise TypeError("V646 stabilizer requires the exact V10 plant")
        if env.obstacle_enabled:
            raise RuntimeError("V646 precontact block stabilization is obstacle-free only")
        trace = info.get("physics_substep_contact_v1")
        before = np.asarray(block_before_xy_m, dtype=np.float64)
        if type(info) is not dict or type(trace) is not dict or before.shape != (2,):
            raise ValueError("V646 stabilizer lost transition evidence")
        raw_counts = np.asarray(trace.get("tool_block_contact_count", ()), dtype=np.int64)
        if raw_counts.shape != (env.config.physics_substeps,):
            raise RuntimeError("V646 stabilizer contact trace shape changed")
        raw_contact = bool(np.any(raw_counts > 0))
        self._contact_seen = bool(self._contact_seen or raw_contact)
        after_before_restore = env.block_xy()
        raw_displacement = float(np.linalg.norm(after_before_restore - before))
        restored = False
        if not self._contact_seen:
            joint_address = int(env.model.jnt_qposadr[env._ids["block_joint"]])
            dof_address = int(env.model.jnt_dofadr[env._ids["block_joint"]])
            env.data.qpos[joint_address : joint_address + 2] = before
            env.data.qvel[dof_address : dof_address + 2] = 0.0
            mujoco.mj_forward(env.model, env.data)
            if not np.allclose(env.block_xy(), before, rtol=0.0, atol=1.0e-12):
                raise RuntimeError("V646 failed to restore unforced block XY")
            env.last_distance = env.distance_to_target()
            env._strict_success_streak = 0
            restored = bool(raw_displacement > 0.0)
            if restored:
                self._restore_count += 1
                self._cumulative_restored_displacement_m += raw_displacement
                self._maximum_restored_step_displacement_m = max(
                    self._maximum_restored_step_displacement_m,
                    raw_displacement,
                )
            reason = str(info.get("terminal_reason", "nonterminal"))
            if reason in {"strict_success", "block_out_of_bounds"}:
                terminated = False
                truncated = bool(env.step_count >= env.config.max_steps)
                info["success"] = False
                info["terminal_failure"] = False
                info["terminated"] = False
                info["truncated"] = truncated
                info["terminal_reason"] = "time_limit" if truncated else "nonterminal"
            info["distance"] = env.distance_to_target()
        self._decision_count += 1
        audit = {
            "format": "edgearm-v646-precontact-block-causal-stabilizer-step-v1",
            "decision_index": self._decision_count - 1,
            "raw_robot_block_contact_this_step": raw_contact,
            "contact_seen_latched": self._contact_seen,
            "raw_unforced_block_xy_displacement_m": raw_displacement,
            "unforced_block_xy_restored": restored,
            "block_xy_after_stabilization_m": env.block_xy().tolist(),
            "expert_action_used": False,
            "waypoint_or_path_used": False,
            "production_admission": False,
        }
        info["causal_precontact_stabilizer_v646"] = dict(audit)
        return bool(terminated), bool(truncated), audit

    def summary(self) -> dict[str, Any]:
        return {
            "format": "edgearm-v646-precontact-block-causal-stabilizer-summary-v1",
            "decision_count": self._decision_count,
            "contact_seen": self._contact_seen,
            "restore_step_count": self._restore_count,
            "cumulative_restored_displacement_m": (self._cumulative_restored_displacement_m),
            "maximum_restored_step_displacement_m": (self._maximum_restored_step_displacement_m),
            "unforced_object_motion_used_as_task_progress": False,
            "expert_action_used": False,
            "waypoint_or_path_used": False,
            "production_admission": False,
        }


def _raw_contact_evidence_v646(
    episode: dict[str, np.ndarray],
) -> np.ndarray:
    neutral = np.asarray(episode["neutral_state"], dtype=np.float32)
    next_neutral = np.asarray(episode["next_neutral_state"], dtype=np.float32)
    contact_slice = _SLICES_V646["tool_block_contact_count"]
    evidence = (
        (neutral[:, contact_slice][:, 0] > 0.0)
        | (next_neutral[:, contact_slice][:, 0] > 0.0)
        | np.asarray(episode["valid_contact"], dtype=bool)
        | np.asarray(episode["invalid_contact"], dtype=bool)
    )
    raw_substep = np.asarray(
        episode.get(
            "raw_contact_evidence_v646",
            np.zeros(len(evidence), dtype=bool),
        ),
        dtype=bool,
    )
    if raw_substep.shape != evidence.shape:
        raise ValueError("V646 raw substep contact evidence shape changed")
    return evidence | raw_substep


def precontact_causal_motion_audit_v646(
    episode: dict[str, np.ndarray],
    *,
    config: CausalSmoothRelayConfigV646 | None = None,
) -> dict[str, Any]:
    """Reject object motion that precedes every observed robot/object contact."""

    selected = config or CausalSmoothRelayConfigV646()
    selected.validate()
    displacement = np.asarray(episode["step_block_displacement_m"], dtype=np.float64)
    evidence = _raw_contact_evidence_v646(episode)
    if (
        displacement.ndim != 1
        or evidence.shape != displacement.shape
        or not np.all(np.isfinite(displacement))
        or np.any(displacement < 0.0)
    ):
        raise ValueError("V646 episode causal arrays are invalid")
    contact_seen = np.maximum.accumulate(evidence)
    unexplained_step = ~contact_seen & (displacement > selected.unexplained_precontact_step_displacement_m)
    precontact_displacement = np.where(~contact_seen, displacement, 0.0)
    cumulative = np.cumsum(precontact_displacement)
    unexplained_cumulative = ~contact_seen & (
        cumulative > selected.unexplained_precontact_episode_displacement_m
    )
    rejected = unexplained_step | unexplained_cumulative
    rejected_rows = np.flatnonzero(rejected)
    first = None if not len(rejected_rows) else int(rejected_rows[0])
    return {
        "format": CAUSAL_MOTION_AUDIT_FORMAT_V646,
        "row_count": int(len(displacement)),
        "contact_evidence_observed": bool(np.any(evidence)),
        "first_contact_evidence_step": (None if not np.any(evidence) else int(np.flatnonzero(evidence)[0])),
        "unexplained_precontact_motion_step_count": int(np.count_nonzero(rejected)),
        "first_unexplained_precontact_motion_step": first,
        "maximum_precontact_step_displacement_m": float(np.max(precontact_displacement, initial=0.0)),
        "cumulative_precontact_displacement_m": float(cumulative[-1] if len(cumulative) else 0.0),
        "causal_motion_valid": bool(not np.any(rejected)),
        "replay_admission": bool(not np.any(rejected)),
        "bulk_vla_data_admission": False,
        "production_admission": False,
    }


def sample_causal_smooth_relay_batch_v646(
    replay: GoalConditionedMarkovHerReplayV614,
    *,
    batch_size: int,
    seed: int,
    phase_config: PhaseIsolatedAcquisitionConfigV626,
    relay_config: RelayDualGoalHerSACConfigV643,
    causal_config: CausalSmoothRelayConfigV646 | None = None,
    episode_mass_multiplier: np.ndarray | None = None,
    transition_mass_multiplier: np.ndarray | None = None,
    first_contact_prefix_learning_mask_v738: np.ndarray | None = None,
    future_goal_strategy: str = "uniform",
    true_goal_minimum_direction_cosine: float | None = None,
    true_goal_minimum_progress_m: float | None = None,
    post_contact_reacquisition_learning_v730: bool = False,
) -> dict[str, np.ndarray]:
    """Apply fail-closed causal penalties to an otherwise unchanged V643 batch."""

    selected = causal_config or CausalSmoothRelayConfigV646()
    selected.validate()
    batch = sample_relay_acquisition_batch_v643(
        replay,
        batch_size=batch_size,
        seed=seed,
        phase_config=phase_config,
        config=relay_config,
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
    roots = np.asarray(batch["source_row_index"], dtype=np.int64)
    arrays = replay.arrays
    evidence = _raw_contact_evidence_v646(arrays)
    contact_seen = np.zeros(replay.transition_count, dtype=bool)
    for episode_index in np.unique(arrays["episode_index"]):
        rows = np.flatnonzero(arrays["episode_index"] == episode_index)
        contact_seen[rows] = np.maximum.accumulate(evidence[rows])
    displacement = np.asarray(arrays["step_block_displacement_m"], dtype=np.float64)
    uncausal = ~contact_seen[roots] & (
        displacement[roots] > selected.unexplained_precontact_step_displacement_m
    )
    reward = np.asarray(batch["reward"], dtype=np.float32).copy()
    done = np.asarray(batch["done"], dtype=np.float32).copy()
    reward -= np.float32(selected.uncausal_motion_penalty) * uncausal.astype(np.float32)
    done[uncausal] = 1.0
    previous_applied = np.zeros((len(roots), 3), dtype=np.float32)
    for index, row in enumerate(roots):
        previous = int(row) - 1
        if previous >= 0 and arrays["episode_index"][previous] == arrays["episode_index"][row]:
            previous_applied[index] = arrays["applied_action"][previous]
    batch["reward"] = reward
    batch["done"] = done
    batch["uncausal_precontact_motion"] = uncausal.astype(bool)
    batch["previous_applied_action"] = previous_applied
    batch["recorded_action_rate_l2"] = np.linalg.norm(
        np.asarray(batch["action"], dtype=np.float32) - previous_applied,
        axis=-1,
    ).astype(np.float32)
    return batch


__all__ = [
    "CAUSAL_MOTION_AUDIT_FORMAT_V646",
    "CAUSAL_SMOOTH_RELAY_FORMAT_V646",
    "AcquisitionActionSlewFilterV646",
    "CausalSmoothRelayConfigV646",
    "PrecontactBlockCausalStabilizerV646",
    "precontact_causal_motion_audit_v646",
    "sample_causal_smooth_relay_batch_v646",
    "strengthened_relay_config_v646",
]
