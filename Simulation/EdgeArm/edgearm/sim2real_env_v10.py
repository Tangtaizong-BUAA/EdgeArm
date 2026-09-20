"""Joint-bounded workspace-filter successor to the stock-gripper V9 plant.

V9 inherits the historical production workspace filter.  That filter clips a
requested target to the configured joint limits, but, when the clipped target
is outside the Cartesian workspace, it interpolates back toward the *live*
joint state.  A force-limited live state can sit microscopically outside a
joint limit.  Interpolation toward that state can therefore re-introduce a
joint-limit violation after clipping and make controller preflight
non-idempotent.

V10 leaves V9 and every source-bound V9 artifact unchanged.  It clips the live
workspace anchor before interpolation, clips every candidate again, and
fails closed if no target satisfies both the joint and Cartesian workspace
contracts.  The wrist camera and stock SO-101 gripper geometry are otherwise
unchanged synthetic, uncalibrated priors.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from .production_env import ProductionMjcfBundleV1
from .reverse_curriculum_v26 import (
    REVERSE_CURRICULUM_FORMAT_V26,
    STRICT_SUCCESS_HOLD_SECONDS_V26,
    reverse_curriculum_stage_v26,
    sample_reverse_curriculum_task_v26,
    strict_success_hold_steps_v26,
)
from .sim2real_env_v6 import RealisticEdgeArmEnvV6
from .sim2real_env_v7 import (
    CONTACT_FEASIBLE_GEOMETRY_VERSION,
    REALISM_CLAIM_LEVEL,
    REALISTIC_DYNAMICS_PROFILE_VERSION,
)
from .sim2real_env_v9 import RealisticEdgeArmEnvV9, RealisticEnvV9Config


JOINT_BOUNDED_DYNAMICS_PROFILE_V10 = (
    "edgearm-sim2real-dynamics-v10-joint-bounded-workspace-filter"
)
JOINT_BOUNDED_SAFETY_FILTER_V10 = (
    "edgearm-joint-bounded-idempotent-workspace-filter-v1"
)
JOINT_BOUNDED_PARAMETER_SOURCE_V10 = (
    "v9_exact_stock_gripper_plus_joint_clipped_live_workspace_anchor"
)
TARGET_CHANGED_ABSOLUTE_TOLERANCE_RAD_V10 = 1.0e-12
TASK_INDEPENDENT_HOME_RESET_SOURCE_V597 = (
    "authored_home_keyframe_installed_before_first_forward_no_physics_step"
)


class V10SafetyFilterInfeasible(RuntimeError):
    """Raised after V10 proves that no joint/workspace target is available."""


class V10ResetExactAnchorInfeasible(V10SafetyFilterInfeasible):
    """Raised when one sampled reset is not itself an exact safe anchor.

    A task-aligned reset candidate that needs workspace projection is not a
    simulator/programming failure.  It is one infeasible sample and must be
    rejected by the bounded reset-retry loop just like an IK-infeasible
    candidate.  Giving it a dedicated type keeps that distinction fail-closed
    without converting unrelated ``RuntimeError`` failures into retries.
    """


@dataclass(frozen=True)
class RealisticEnvV10Config(RealisticEnvV9Config):
    """Exact V9 plant parameters with the V10 safety-filter contract."""

    reverse_curriculum_stage_v26: int | None = None
    multichoice_blocks: bool = False

    def __post_init__(self) -> None:
        super().__post_init__()
        stage_index = self.reverse_curriculum_stage_v26
        if stage_index is None:
            return
        stage = reverse_curriculum_stage_v26(stage_index)
        required_hold_steps = strict_success_hold_steps_v26(self.fps)
        if self.strict_success_hold_steps != required_hold_steps:
            raise ValueError(
                "V26 reverse curriculum requires an exact three-second strict-success hold"
            )
        if self.max_steps < required_hold_steps + 32:
            raise ValueError(
                "V26 episode horizon must leave at least 32 action steps before the hold window"
            )
        stage.validate()


class RealisticEdgeArmEnvV10(RealisticEdgeArmEnvV9):
    """V9 stock-gripper plant with an idempotent joint/workspace filter."""

    profile_version = JOINT_BOUNDED_DYNAMICS_PROFILE_V10

    def __init__(
        self,
        config: RealisticEnvV10Config | None = None,
        seed: int = 0,
        *,
        model_scene_path: Path | None = None,
        model_scene_bundle: ProductionMjcfBundleV1 | None = None,
    ) -> None:
        if config is not None and type(config) is not RealisticEnvV10Config:
            raise TypeError("RealisticEdgeArmEnvV10 requires exact RealisticEnvV10Config")
        self.joint_bounded_config = config or RealisticEnvV10Config()
        self._workspace_recovery_anchor_v10: np.ndarray | None = None
        self._reverse_curriculum_task_v26: dict[str, Any] | None = None
        self._task_independent_reset_joint_position_v597: (
            np.ndarray | None
        ) = None
        super().__init__(
            self.joint_bounded_config,
            seed=seed,
            model_scene_path=model_scene_path,
            model_scene_bundle=model_scene_bundle,
        )

    def _initial_reset_joint_position(
        self,
        block: np.ndarray,
        target: np.ndarray,
    ) -> np.ndarray:
        """Select the normal V7 reset or an explicitly armed Home reset."""

        override = self._task_independent_reset_joint_position_v597
        if override is None:
            return super()._initial_reset_joint_position(block, target)
        return np.asarray(override, dtype=np.float64).copy()

    def _install_reverse_curriculum_profile_v26(self) -> None:
        stage_index = self.joint_bounded_config.reverse_curriculum_stage_v26
        if stage_index is None:
            return
        if self._reverse_curriculum_task_v26 is None:
            raise RuntimeError("V26 reset lost its sampled curriculum task")
        stage = reverse_curriculum_stage_v26(stage_index)
        required_hold_steps = strict_success_hold_steps_v26(self.config.fps)
        self.episode_domain["reverse_curriculum_v26"] = {
            "format": REVERSE_CURRICULUM_FORMAT_V26,
            "stage": asdict(stage),
            "sampled_task": deepcopy(self._reverse_curriculum_task_v26),
            "strict_success_hold_seconds": STRICT_SUCCESS_HOLD_SECONDS_V26,
            "strict_success_hold_steps": required_hold_steps,
            "fps": int(self.config.fps),
            "realized_hold_seconds": required_hold_steps / self.config.fps,
            "expert_calls": 0,
            "behavior_cloning_steps": 0,
            "production_admission": False,
        }

    def reset_task_independent_home_v597(
        self,
        joint_position: np.ndarray,
        *,
        seed: int,
        obstacle: bool,
        stress: bool,
    ) -> dict[str, Any]:
        """Reset directly to Home without a task-aligned transient or step.

        This is intentionally separate from :meth:`reset`: existing V10
        artifacts retain their source-bound V7 task-aligned reset.  The V597
        caller must complete its Home collision audit before any policy step.
        """

        home = np.asarray(joint_position, dtype=np.float64)
        if (
            home.shape != (6,)
            or not np.all(np.isfinite(home))
            or np.any(home < self.joint_ranges[:, 0])
            or np.any(home > self.joint_ranges[:, 1])
        ):
            raise ValueError("task-independent Home must be joint-bounded [6]")
        if self._task_independent_reset_joint_position_v597 is not None:
            raise RuntimeError("nested task-independent Home reset is forbidden")

        self._task_independent_reset_joint_position_v597 = home.copy()
        try:
            observation = RealisticEdgeArmEnvV6.reset(
                self,
                seed=seed,
                obstacle=obstacle,
                stress=stress,
            )
        finally:
            self._task_independent_reset_joint_position_v597 = None

        if not np.array_equal(self.data.qpos[:6], home):
            raise RuntimeError("task-independent reset did not retain exact Home")
        if float(self.data.time) != 0.0:
            raise RuntimeError("task-independent reset advanced physics before policy")

        self._reset_collision_audit = {
            "format": "edgearm-v597-home-reset-audit-pending-v1",
            "reset_valid": None,
            "reset_failure_reasons": ["pending_v597_home_collision_audit"],
            "task_aligned_ik_calls_before_policy": 0,
            "physics_steps_before_policy": 0,
        }
        self.episode_domain["realism_v7"] = {
            "profile_version": self.profile_version,
            "base_dynamics_profile_version": (
                REALISTIC_DYNAMICS_PROFILE_VERSION
            ),
            "effective_profile_version": self.profile_version,
            "geometry_version": CONTACT_FEASIBLE_GEOMETRY_VERSION,
            "claim_level": REALISM_CLAIM_LEVEL,
            "parameter_source": (
                "v6_randomized_dynamics_plus_v7_workbench_geometry_"
                "plus_authored_task_independent_home"
            ),
            "physical_samples": 0,
            "physical_trials": 0,
            "physically_calibrated": False,
            "physical_hardware_connected": False,
            "state_update": (
                "mujoco_force_limited_actuator_with_workbench_mount_clearance"
            ),
            "reset_state_source": TASK_INDEPENDENT_HOME_RESET_SOURCE_V597,
            "task_aligned_privileged_reset": False,
            "deployment_reset_equivalent": False,
            "reset_privileged_state_used": [],
            "requested_seed": int(seed),
            "accepted_candidate_seed": int(seed),
            "reset_attempt_count": 1,
            "rejected_reset_attempts": [],
            "desk_original_x_bounds_m": self._desk_original_x_bounds.tolist(),
            "desk_corrected_x_bounds_m": self._desk_corrected_x_bounds.tolist(),
            "reset_home_joint_position_rad": home.tolist(),
            "task_aligned_ik_calls_before_policy": 0,
            "physics_steps_before_policy": 0,
            "obstacle_placement_audit": deepcopy(
                self._obstacle_placement_audit
            ),
            "reset_collision_audit": deepcopy(self._reset_collision_audit),
        }
        self._physics_substep_contact_v1 = None
        self._runtime_pusher_desk_guard_v1 = {}
        self._controller_preflight_v1 = {}
        self._terminal_viability_gate_evaluated_decisions = 0
        self._terminal_viability_gate_skipped_decisions = 0
        self._terminal_viability_gate_reason_counts = {
            "selected_search_index_positive": 0,
            "bisection_lambda_below_one": 0,
            "forecast_minimum_within_reserve": 0,
        }
        self._terminal_viability_exact_evaluation_count = 0
        self._terminal_viability_exact_evaluation_wall_seconds = 0.0
        self.episode_domain["realism_v9"] = self._stock_gripper_profile()

        filtered_home, reason = self._safety_filter(home)
        if not np.array_equal(filtered_home, home) or reason:
            raise RuntimeError(
                "task-independent Home is not an exact V10 recovery anchor"
            )
        self._workspace_recovery_anchor_v10 = home.copy()
        self.episode_domain["realism_v10"] = self._joint_bounded_profile()
        self._install_reverse_curriculum_profile_v26()
        return self.observation() if observation is not None else observation

    def _sample_task(self, stress: bool) -> tuple[np.ndarray, np.ndarray]:
        """Use production sampling unless an explicit V26 stage is active."""

        stage_index = self.joint_bounded_config.reverse_curriculum_stage_v26
        if stage_index is None:
            self._reverse_curriculum_task_v26 = None
            return super()._sample_task(stress)
        task = sample_reverse_curriculum_task_v26(
            self.rng,
            reverse_curriculum_stage_v26(stage_index),
            stress=stress,
        )
        self._reverse_curriculum_task_v26 = asdict(task)
        return (
            np.asarray(task.block_xy_m, dtype=np.float64),
            np.asarray(task.target_xy_m, dtype=np.float64),
        )

    def _safety_filter(self, requested: np.ndarray) -> tuple[np.ndarray, str]:
        """Return one target satisfying joint and Cartesian bounds exactly.

        The selected candidate is a mathematical fixed point of this filter
        for the current plant state: it is already joint-bounded and its tool
        site is already inside the configured workspace, so a second call
        returns the same float64 vector bit-for-bit.
        """

        requested = np.asarray(requested, dtype=np.float64)
        if requested.shape != (6,) or not np.all(np.isfinite(requested)):
            raise ValueError("V10 safety target must be a finite six-joint vector")
        lower = np.asarray(self.joint_ranges[:, 0], dtype=np.float64)
        upper = np.asarray(self.joint_ranges[:, 1], dtype=np.float64)
        selected = np.clip(requested, lower, upper)
        reasons: list[str] = []
        if np.any(self._target_changed_mask(requested, selected)):
            reasons.append("joint_limit")

        trial = mujoco.MjData(self.model)

        def workspace_valid(candidate: np.ndarray) -> bool:
            trial.qpos[:] = self.data.qpos
            trial.qvel[:] = self.data.qvel
            trial.qpos[:6] = candidate
            mujoco.mj_forward(self.model, trial)
            xyz = trial.site_xpos[self._ids["tool_site"]]
            bounds = (
                self.config.workspace_x,
                self.config.workspace_y,
                self.config.workspace_z,
            )
            return all(
                low <= value <= high
                for value, (low, high) in zip(xyz, bounds, strict=True)
            )

        if not workspace_valid(selected):
            live = np.asarray(self.data.qpos[:6], dtype=np.float64).copy()
            bounded_live = np.clip(live, lower, upper)
            if not np.array_equal(bounded_live, live):
                reasons.append("joint_bounded_workspace_anchor")
            found = False
            for scale in (0.5, 0.25, 0.1, 0.05, 0.0):
                candidate = bounded_live + scale * (selected - bounded_live)
                candidate = np.clip(candidate, lower, upper)
                if workspace_valid(candidate):
                    selected = candidate
                    found = True
                    break
            if not found and self._workspace_recovery_anchor_v10 is not None:
                recovery_anchor = np.clip(
                    np.asarray(
                        self._workspace_recovery_anchor_v10,
                        dtype=np.float64,
                    ),
                    lower,
                    upper,
                )
                if workspace_valid(recovery_anchor):
                    # Find the first valid point on a deterministic dense path
                    # away from the invalid live state, then refine only the
                    # bracketing interval.  The retained upper endpoint is
                    # always explicitly workspace-valid; monotonicity outside
                    # that local bracket is neither assumed nor required.
                    previous_lambda = 0.0
                    for candidate_lambda in np.linspace(0.0, 1.0, 65)[1:]:
                        candidate = bounded_live + candidate_lambda * (
                            recovery_anchor - bounded_live
                        )
                        candidate = np.clip(candidate, lower, upper)
                        if workspace_valid(candidate):
                            lower_lambda = previous_lambda
                            upper_lambda = float(candidate_lambda)
                            upper_candidate = candidate
                            for _ in range(12):
                                midpoint_lambda = 0.5 * (
                                    lower_lambda + upper_lambda
                                )
                                midpoint = bounded_live + midpoint_lambda * (
                                    recovery_anchor - bounded_live
                                )
                                midpoint = np.clip(midpoint, lower, upper)
                                if workspace_valid(midpoint):
                                    upper_lambda = midpoint_lambda
                                    upper_candidate = midpoint
                                else:
                                    lower_lambda = midpoint_lambda
                            selected = upper_candidate
                            reasons.append("workspace_recovery_anchor")
                            found = True
                            break
                        previous_lambda = float(candidate_lambda)
                    if not found:
                        # ``bounded_live + 1 * (anchor - bounded_live)`` can
                        # differ from ``anchor`` by a final rounding bit.  The
                        # stored anchor itself was validated at reset, so keep
                        # it as the explicit last authority.
                        selected = recovery_anchor.copy()
                        reasons.append("workspace_recovery_anchor")
                        found = True
            reasons.append("workspace_scaled")
            if not found:
                raise V10SafetyFilterInfeasible(
                    "V10 safety filter found no joint-bounded workspace-feasible target"
                )

        if (
            not np.all((selected >= lower) & (selected <= upper))
            or not workspace_valid(selected)
        ):  # pragma: no cover - defensive postcondition
            raise RuntimeError("V10 safety filter postcondition failed")
        return selected.copy(), "+".join(reasons)

    @staticmethod
    def _target_changed_mask(
        before: np.ndarray,
        after: np.ndarray,
    ) -> np.ndarray:
        """Return the versioned exact-tolerance V10 provenance mask."""

        return np.logical_not(
            np.isclose(
                np.asarray(before, dtype=np.float64),
                np.asarray(after, dtype=np.float64),
                rtol=0.0,
                atol=TARGET_CHANGED_ABSOLUTE_TOLERANCE_RAD_V10,
            )
        )

    def _joint_bounded_profile(self) -> dict[str, Any]:
        return {
            "profile_version": JOINT_BOUNDED_DYNAMICS_PROFILE_V10,
            "safety_filter_version": JOINT_BOUNDED_SAFETY_FILTER_V10,
            "parameter_source": JOINT_BOUNDED_PARAMETER_SOURCE_V10,
            "stock_follower_unmodified": True,
            "added_contact_tool": False,
            "joint_clip_before_workspace_projection": True,
            "workspace_anchor_joint_clipped": True,
            "workspace_candidates_joint_clipped": True,
            "exact_idempotence_postcondition": True,
            "target_changed_rtol": 0.0,
            "target_changed_atol_rad": (
                TARGET_CHANGED_ABSOLUTE_TOLERANCE_RAD_V10
            ),
            "infeasible_step_transaction_restored_before_estop": True,
            "wrist_camera_physically_calibrated": False,
            "physical_samples": 0,
            "physical_trials": 0,
            "physical_hardware_connected": False,
            "configuration": asdict(self.joint_bounded_config),
        }

    def reset(
        self,
        seed: int | None = None,
        *,
        obstacle: bool | None = None,
        stress: bool = False,
    ) -> dict[str, Any]:
        observation = super().reset(seed=seed, obstacle=obstacle, stress=stress)
        recovery_anchor = np.clip(
            np.asarray(self.data.qpos[:6], dtype=np.float64),
            self.joint_ranges[:, 0],
            self.joint_ranges[:, 1],
        )
        filtered_anchor, _reason = self._safety_filter(recovery_anchor)
        if not np.array_equal(filtered_anchor, recovery_anchor):
            raise V10ResetExactAnchorInfeasible(
                "V10 sampled reset did not provide an exact recovery anchor"
            )
        self._workspace_recovery_anchor_v10 = recovery_anchor.copy()
        self.episode_domain["realism_v10"] = self._joint_bounded_profile()
        self._install_reverse_curriculum_profile_v26()
        return observation

    def step(
        self,
        action: Any,
    ) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        snapshot = self._snapshot_terminal_viability_state()
        v6_rng_state = deepcopy(self._v6_rng.bit_generator.state)
        scalar_snapshot = {
            name: deepcopy(getattr(self, name))
            for name in (
                "step_count",
                "success_streak",
                "_strict_success_streak",
                "last_distance",
                "_command_burst_remaining",
                "_last_command_lost",
                "_submission_ingress_lost",
                "_last_command_feedback_v1",
                "_terminal_viability_gate_evaluated_decisions",
                "_terminal_viability_gate_skipped_decisions",
                "_terminal_viability_gate_reason_counts",
                "_terminal_viability_exact_evaluation_count",
                "_terminal_viability_exact_evaluation_wall_seconds",
            )
        }
        try:
            observation, reward, terminated, truncated, info = super().step(action)
        except V10SafetyFilterInfeasible as error:
            self._restore_terminal_viability_state(snapshot)
            self._v6_rng.bit_generator.state = deepcopy(v6_rng_state)
            for name, value in scalar_snapshot.items():
                setattr(self, name, value)
            self.estop = True
            raise V10SafetyFilterInfeasible(
                "V10 restored the failed command transaction and latched estop"
            ) from error
        info["realism_v10"] = dict(self.episode_domain["realism_v10"])
        return observation, reward, terminated, truncated, info


__all__ = [
    "JOINT_BOUNDED_DYNAMICS_PROFILE_V10",
    "JOINT_BOUNDED_PARAMETER_SOURCE_V10",
    "JOINT_BOUNDED_SAFETY_FILTER_V10",
    "TARGET_CHANGED_ABSOLUTE_TOLERANCE_RAD_V10",
    "TASK_INDEPENDENT_HOME_RESET_SOURCE_V597",
    "RealisticEdgeArmEnvV10",
    "RealisticEnvV10Config",
    "V10ResetExactAnchorInfeasible",
    "V10SafetyFilterInfeasible",
]
