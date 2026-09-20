"""Task-independent Home reset for complete acquisition-to-push rollouts.

Earlier reverse-curriculum rollouts place the stock gripper a few millimetres
behind the block with privileged block/target state.  Those trajectories are
useful for learning contact transport, but they do not test visual discovery
or approach from the robot's ordinary Home pose.  V597 restores the authored
Home joint state after task sampling and starts the unchanged V22 guarded
task-frame controller there.

The final simulated arm state is independent of block and target pose.  It is
not called deployment-equivalent until the authored Home pose and wrist camera
have been calibrated against physical hardware.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, replace
from typing import Any

import numpy as np
from scipy.optimize import least_squares

from .asymmetric_multiview_ppo_v1 import (
    MAX_RESET_ATTEMPTS_V12,
    RESET_RETRY_STRIDE_V12,
    SOURCE_TYPE,
    VIEW_NAMES,
    MultiViewRendererProtocolV1,
    SideContactIKPlannerV1,
    StockGripperTaskSpaceActionConfigV12,
)
from .sim2real_env_v10 import RealisticEdgeArmEnvV10
from .stock_gripper_action_guard_v4 import (
    STOCK_GRIPPER_ACTION_GUARD_FORMAT_V4,
    StockGripperActionGuardV4,
)
from .stock_gripper_push_face_contact_v22 import (
    STOCK_GRIPPER_PUSH_FACE_CONTACT_FORMAT_V22,
    configure_stock_gripper_push_face_contact_v22,
    restore_stock_gripper_distal_contact_v22,
)
from .stock_gripper_taskframe_v22 import (
    StockGripperTaskFrameAdapterV22,
)


TASK_INDEPENDENT_HOME_RESET_FORMAT_V597 = (
    "edgearm-v597-task-independent-stock-gripper-home-reset-v1"
)
TASK_INDEPENDENT_HOME_RUNTIME_FORMAT_V597 = (
    "edgearm-v597-task-independent-stock-gripper-home-runtime-v1"
)
TASK_INDEPENDENT_HOME_ACQUISITION_FORMAT_V597 = (
    "edgearm-v597-bounded-local-home-acquisition-v1"
)

_ACQUISITION_POSITION_SCALE_M_V597 = 0.010
_ACQUISITION_FREE_SPACE_NORMAL_SCALE_V597 = 0.25
_ACQUISITION_ALIGNMENT_NORMAL_SCALE_V597 = 0.35
_ACQUISITION_ALIGNMENT_MAXIMUM_HEIGHT_M_V597 = 0.080
_ACQUISITION_ALIGNMENT_MAXIMUM_BLOCK_XY_DISTANCE_M_V597 = 0.120
_ACQUISITION_JOINT_REGULARIZATION_V597 = 0.03
_ACQUISITION_MAXIMUM_FUNCTION_EVALUATIONS_V597 = 80
HOME_ACQUISITION_TRANSPORT_ALIGNMENT_V597 = 0.98
HOME_ACQUISITION_TRANSPORT_HEIGHT_MARGIN_M_V597 = 0.003


@dataclass(frozen=True)
class HomeAcquisitionOrientationScheduleV597:
    """Select the automatic broad-face orientation schedule during Home reach."""

    mode: str = "hard_switch"
    alignment_maximum_height_m: float = (
        _ACQUISITION_ALIGNMENT_MAXIMUM_HEIGHT_M_V597
    )
    alignment_start_block_xy_distance_m: float = (
        _ACQUISITION_ALIGNMENT_MAXIMUM_BLOCK_XY_DISTANCE_M_V597
    )
    full_alignment_block_xy_distance_m: float = (
        _ACQUISITION_ALIGNMENT_MAXIMUM_BLOCK_XY_DISTANCE_M_V597
    )
    free_space_normal_scale: float = _ACQUISITION_FREE_SPACE_NORMAL_SCALE_V597
    alignment_normal_scale: float = _ACQUISITION_ALIGNMENT_NORMAL_SCALE_V597
    release_precontact_distance_m: float | None = None
    precontact_standoff_m: float = 0.055
    precontact_tool_height_m: float = 0.055

    def validate(self) -> None:
        if self.mode not in {"hard_switch", "progressive"}:
            raise ValueError("V597 orientation schedule mode is invalid")
        values = np.asarray(
            [
                self.alignment_maximum_height_m,
                self.alignment_start_block_xy_distance_m,
                self.full_alignment_block_xy_distance_m,
                self.free_space_normal_scale,
                self.alignment_normal_scale,
                self.precontact_standoff_m,
                self.precontact_tool_height_m,
            ],
            dtype=np.float64,
        )
        if not np.all(np.isfinite(values)) or np.any(values <= 0.0):
            raise ValueError("V597 orientation schedule values must be positive")
        if (
            self.full_alignment_block_xy_distance_m
            > self.alignment_start_block_xy_distance_m
        ):
            raise ValueError("V597 full alignment must be inside its start radius")
        if self.mode == "hard_switch" and not np.isclose(
            self.full_alignment_block_xy_distance_m,
            self.alignment_start_block_xy_distance_m,
            rtol=0.0,
            atol=1.0e-12,
        ):
            raise ValueError("V597 hard switch cannot have a blend interval")
        if self.mode == "progressive" and not (
            self.full_alignment_block_xy_distance_m
            < self.alignment_start_block_xy_distance_m
        ):
            raise ValueError("V597 progressive schedule requires a blend interval")
        release = self.release_precontact_distance_m
        if release is not None and (
            not np.isfinite(float(release))
            or not 0.0 < float(release) <= 0.08
        ):
            raise ValueError("V597 release precontact distance is invalid")


def home_acquisition_alignment_weight_v597(
    schedule: HomeAcquisitionOrientationScheduleV597,
    *,
    tool_height_m: float,
    tool_block_xy_distance_m: float,
) -> float:
    """Return the automatic task-axis alignment weight without hidden state."""

    if type(schedule) is not HomeAcquisitionOrientationScheduleV597:
        raise TypeError("V597 alignment weight requires its exact schedule")
    schedule.validate()
    height = float(tool_height_m)
    distance = float(tool_block_xy_distance_m)
    if (
        not np.isfinite(height)
        or not np.isfinite(distance)
        or distance < 0.0
    ):
        raise ValueError("V597 alignment geometry is invalid")
    if (
        height > schedule.alignment_maximum_height_m
        or distance > schedule.alignment_start_block_xy_distance_m
    ):
        return 0.0
    if (
        schedule.mode == "hard_switch"
        or distance <= schedule.full_alignment_block_xy_distance_m
    ):
        return 1.0
    return float(
        (schedule.alignment_start_block_xy_distance_m - distance)
        / (
            schedule.alignment_start_block_xy_distance_m
            - schedule.full_alignment_block_xy_distance_m
        )
    )


def authored_stock_gripper_home_q_v597(
    env: RealisticEdgeArmEnvV10,
) -> np.ndarray:
    """Return the task-independent authored Home pose with stock gripper set."""

    if type(env) is not RealisticEdgeArmEnvV10:
        raise TypeError("V597 Home pose requires exact RealisticEdgeArmEnvV10")
    if int(env.model.nkey) < 1:
        raise RuntimeError("V597 model has no authored Home keyframe")
    home = np.asarray(env.model.key_qpos[0, :6], dtype=np.float64).copy()
    home[5] = float(env.tool_gripper_joint_position_rad)
    if (
        home.shape != (6,)
        or not np.all(np.isfinite(home))
        or np.any(home < env.joint_ranges[:, 0])
        or np.any(home > env.joint_ranges[:, 1])
    ):
        raise RuntimeError("V597 authored Home pose violates joint limits")
    return home


class StockGripperHomeTaskFrameAdapterV597(
    StockGripperTaskFrameAdapterV22
):
    """Start V22 guarded task-frame control without task-aligned prepositioning."""

    def __init__(
        self,
        env: RealisticEdgeArmEnvV10,
        config: StockGripperTaskSpaceActionConfigV12 | None = None,
        *,
        orientation_schedule: HomeAcquisitionOrientationScheduleV597 | None = None,
    ) -> None:
        super().__init__(env, config)
        self.orientation_schedule_v597 = (
            orientation_schedule or HomeAcquisitionOrientationScheduleV597()
        )
        if type(self.orientation_schedule_v597) is not HomeAcquisitionOrientationScheduleV597:
            raise TypeError("V597 orientation schedule must use its exact config")
        self.orientation_schedule_v597.validate()
        self.home_reset_audit_v597: dict[str, Any] = {}
        self.last_acquisition_report_v597: dict[str, Any] = {}
        self.acquisition_step_count_v597 = 0

    def begin_episode(self, seed: int) -> None:
        if type(seed) is not int or seed < 0:
            raise ValueError("V597 Home episode seed must be non-negative")
        expected_home = authored_stock_gripper_home_q_v597(self.env)
        self._begin_episode_from_exact_joint_v622(
            seed,
            expected_home,
            reset_kind="task_independent_home",
        )

    def _begin_episode_from_exact_joint_v622(
        self,
        seed: int,
        expected_joint_position: np.ndarray,
        *,
        reset_kind: str,
    ) -> None:
        """Initialize the unchanged guarded runtime at one audited reset.

        V597 calls this only for exact authored Home.  V622 uses the narrow
        extension for explicitly privileged intermediate curriculum states.
        """

        if type(seed) is not int or seed < 0:
            raise ValueError("V597/V622 episode seed must be non-negative")
        expected = np.asarray(expected_joint_position, dtype=np.float64)
        if (
            expected.shape != (6,)
            or not np.all(np.isfinite(expected))
            or np.any(expected < self.env.joint_ranges[:, 0])
            or np.any(expected > self.env.joint_ranges[:, 1])
        ):
            raise ValueError("V597/V622 expected reset joint state is invalid")
        if reset_kind not in {
            "task_independent_home",
            "privileged_interpolated_approach_curriculum",
            "privileged_dynamic_reverse_random_walk_curriculum",
        }:
            raise ValueError("V597/V622 reset kind is invalid")
        live = np.asarray(self.env.data.qpos[:6], dtype=np.float64).copy()
        if not np.array_equal(live, expected):
            raise RuntimeError("V597/V622 episode did not begin at audited reset")

        restore_stock_gripper_distal_contact_v22(self.env)
        self.tracking = SideContactIKPlannerV1(self.env, self._ik_config)
        self._cartesian_goal = self.env.tool_xyz().copy()
        self._latched_joint_target = live.copy()
        self.env._workspace_recovery_anchor_v10 = live.copy()
        self.last_guard_report = {}
        self.last_recovery_report = {}
        self.last_policy_target_encoding_report = {}
        self.recovery_count = 0
        self.last_acquisition_report_v597 = {}
        self.acquisition_step_count_v597 = 0

        profile = configure_stock_gripper_push_face_contact_v22(self.env)
        guard = StockGripperActionGuardV4(self.env, self.config.guard)
        hold_action, hold_identity = guard.action_for_absolute_target(live)
        selected_hold, hold_report = guard.select(
            hold_action,
            baseline_action=hold_action,
            require_hold_tail=True,
        )
        if selected_hold is None or not np.array_equal(
            selected_hold, hold_action
        ):
            restore_stock_gripper_distal_contact_v22(self.env)
            raise RuntimeError("V597/V622 reset target failed exact V22 hold proof")
        hold_tail = dict(hold_report["selected_hold_tail_forecast"])
        if not (
            bool(hold_tail["hard_valid"])
            and bool(hold_tail["planning_margin_valid"])
            and bool(hold_tail["remaining_submissions_target_latched_holds"])
        ):
            restore_stock_gripper_distal_contact_v22(self.env)
            raise RuntimeError("V597/V622 reset hold-tail proof is incomplete")

        self.guard = guard
        self.last_guard_report = hold_report
        self.push_face_profile_v22 = deepcopy(profile)
        self._episode_active = True
        self.env.episode_domain["stock_gripper_taskframe_runtime_v22"] = {
            "format": TASK_INDEPENDENT_HOME_RUNTIME_FORMAT_V597,
            "seed": seed,
            "source_type": SOURCE_TYPE,
            "base_runtime_contract": (
                "edgearm-stock-gripper-semantic-push-face-taskframe-runtime-v22"
            ),
            "contact_identity_format": (
                STOCK_GRIPPER_PUSH_FACE_CONTACT_FORMAT_V22
            ),
            "contact_candidate_geom_count": int(
                profile["contact_candidate_geom_count"]
            ),
            "safety_only_geom_count": int(profile["safety_only_geom_count"]),
            "collision_geometry_changed": False,
            "policy_action_space_changed_from_v22": False,
            "task_independent_home_start": bool(
                reset_kind == "task_independent_home"
            ),
            "start_reset_kind_v622": reset_kind,
            "intermediate_curriculum_start": bool(
                reset_kind != "task_independent_home"
            ),
            "dynamic_reverse_random_walk_curriculum": bool(
                reset_kind
                == "privileged_dynamic_reverse_random_walk_curriculum"
            ),
            "safety_guard_format": STOCK_GRIPPER_ACTION_GUARD_FORMAT_V4,
            "safety_guard_config": asdict(self.config.guard),
            "initial_hold_target_identity": hold_identity,
            "initial_hold_tail_minimum_clearance_m": float(
                hold_tail[
                    "minimum_forecast_safety_only_block_distance_m"
                ]
            ),
            "stock_follower_unmodified": True,
            "added_contact_tool": False,
            "simulator_privileged_actor": True,
            "bounded_local_home_acquisition_active": True,
            "home_acquisition_format": (
                TASK_INDEPENDENT_HOME_ACQUISITION_FORMAT_V597
            ),
            "home_acquisition_policy_controls_xyz": True,
            "home_acquisition_automatic_orientation_constraint": True,
            "home_acquisition_orientation_schedule": asdict(
                self.orientation_schedule_v597
            ),
            "home_acquisition_automatic_route_or_path": False,
            "home_acquisition_uses_privileged_task_axis": True,
            "physical_task_axis_source_required": (
                "wrist_camera_perception_or_3d_reconstruction"
            ),
            "physical_home_calibrated": False,
            "production_admission": False,
        }

    def _acquisition_pose_v597(
        self,
        movable_joint_position_rad: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        movable = np.asarray(movable_joint_position_rad, dtype=np.float64)
        if movable.shape != (5,) or not np.all(np.isfinite(movable)):
            raise ValueError("V597 acquisition joint position must be finite [5]")
        complete = np.r_[
            movable,
            float(self.env.tool_gripper_joint_position_rad),
        ]
        return self._pose_for_joint_position(complete)

    def _home_acquisition_active_v597(
        self,
        forward: np.ndarray,
    ) -> tuple[bool, float, float, float]:
        direction = np.asarray(forward, dtype=np.float64)
        direction /= max(float(np.linalg.norm(direction)), 1.0e-12)
        position = self.env.tool_xyz().copy()
        normal = self.env.data.site_xmat[
            self.env._ids["tool_site"]
        ].reshape(3, 3)[:, 1].copy()
        horizontal_norm = float(np.linalg.norm(normal[:2]))
        alignment = abs(float(np.dot(normal[:2], direction))) / max(
            horizontal_norm,
            1.0e-12,
        )
        maximum_transport_height = (
            float(self.config.maximum_tool_height_m)
            + HOME_ACQUISITION_TRANSPORT_HEIGHT_MARGIN_M_V597
        )
        schedule = self.orientation_schedule_v597
        precontact_goal = np.r_[
            self.env.block_xy()
            - schedule.precontact_standoff_m * direction,
            schedule.precontact_tool_height_m,
        ]
        precontact_distance = float(np.linalg.norm(position - precontact_goal))
        release_distance = schedule.release_precontact_distance_m
        active = bool(
            float(position[2]) > maximum_transport_height
            or alignment < HOME_ACQUISITION_TRANSPORT_ALIGNMENT_V597
            or (
                release_distance is not None
                and precontact_distance > float(release_distance)
            )
        )
        return active, alignment, float(position[2]), precontact_distance

    def _policy_tool_height_bounds(self) -> tuple[float, float]:
        """Preserve policy Z authority throughout free-space acquisition."""

        forward, _lateral = self._task_axes(self.env)
        acquisition_active, _alignment, _height, _distance = (
            self._home_acquisition_active_v597(forward)
        )
        if acquisition_active:
            return (
                float(self.config.minimum_tool_height_m),
                float(self.env.config.workspace_z[1]),
            )
        return super()._policy_tool_height_bounds()

    def _track_policy_target(
        self,
        candidate_goal: np.ndarray,
        forward: np.ndarray,
        *,
        initial: np.ndarray,
    ) -> dict[str, Any]:
        """Take one bounded local acquisition step before contact transport.

        V22's tracker is intentionally strict on the contact manifold: it
        requires the broad face to be aligned with the push direction at the
        requested Cartesian endpoint.  The authored Home pose is far above
        that manifold and has an approximately orthogonal face normal, so an
        instantaneous solve produces no executable local action.  Here the
        policy still chooses XYZ; only the user-required broad-face
        orientation constraint is progressed automatically.  The candidate
        remains inside one ordinary V12 joint-target step and then passes the
        unchanged V22 dynamic guard in the parent controller.
        """

        (
            acquisition_active,
            alignment_before,
            height_before,
            precontact_distance_before,
        ) = (
            self._home_acquisition_active_v597(forward)
        )
        if not acquisition_active:
            tracked = super()._track_policy_target(
                candidate_goal,
                forward,
                initial=initial,
            )
            self.last_acquisition_report_v597 = {
                "format": TASK_INDEPENDENT_HOME_ACQUISITION_FORMAT_V597,
                "mode": "contact_transport_tracker_v22",
                "acquisition_active": False,
                "normal_heading_alignment_before": alignment_before,
                "tool_height_before_m": height_before,
                "tool_precontact_distance_before_m": (
                    precontact_distance_before
                ),
                "release_precontact_distance_m": (
                    self.orientation_schedule_v597.release_precontact_distance_m
                ),
                "parent_tracker_feasible": bool(tracked["feasible"]),
                "automatic_route_or_path": False,
                "production_admission": False,
            }
            return tracked

        target = np.asarray(candidate_goal, dtype=np.float64)
        direction = np.asarray(forward, dtype=np.float64)
        if (
            target.shape != (3,)
            or direction.shape != (2,)
            or not np.all(np.isfinite(np.r_[target, direction]))
        ):
            raise ValueError("V597 acquisition target and direction are invalid")
        direction /= max(float(np.linalg.norm(direction)), 1.0e-12)
        live_q = np.asarray(self.env.data.qpos[:6], dtype=np.float64).copy()
        current_position, current_normal = self._acquisition_pose_v597(
            live_q[:5]
        )
        tool_block_xy_distance = float(
            np.linalg.norm(current_position[:2] - self.env.block_xy())
        )
        schedule = self.orientation_schedule_v597
        alignment_weight = home_acquisition_alignment_weight_v597(
            schedule,
            tool_height_m=height_before,
            tool_block_xy_distance_m=tool_block_xy_distance,
        )
        current_unit_normal = current_normal / max(
            float(np.linalg.norm(current_normal)),
            1.0e-12,
        )
        aligned_normal = np.r_[direction, 0.0]
        if float(np.dot(current_unit_normal, aligned_normal)) < 0.0:
            aligned_normal = -aligned_normal
        desired_normal = (
            (1.0 - alignment_weight) * current_unit_normal
            + alignment_weight * aligned_normal
        )
        desired_normal /= max(float(np.linalg.norm(desired_normal)), 1.0e-12)
        normal_scale = float(
            (1.0 - alignment_weight) * schedule.free_space_normal_scale
            + alignment_weight * schedule.alignment_normal_scale
        )
        if alignment_weight <= 0.0:
            acquisition_subphase = "free_space_xyz_preserve_face"
        elif alignment_weight >= 1.0:
            acquisition_subphase = "near_task_axis_alignment"
        else:
            acquisition_subphase = "progressive_task_axis_alignment"

        maximum_delta = float(self.config.maximum_joint_target_delta_rad)
        lower = np.maximum(
            np.asarray(self.env.joint_ranges[:5, 0], dtype=np.float64),
            live_q[:5] - maximum_delta,
        )
        upper = np.minimum(
            np.asarray(self.env.joint_ranges[:5, 1], dtype=np.float64),
            live_q[:5] + maximum_delta,
        )
        start = np.clip(
            np.asarray(initial[:5], dtype=np.float64),
            lower,
            upper,
        )

        def residual(joints: np.ndarray) -> np.ndarray:
            position, normal = self._acquisition_pose_v597(joints)
            return np.r_[
                (target - position) / _ACQUISITION_POSITION_SCALE_M_V597,
                (desired_normal - normal) / normal_scale,
                _ACQUISITION_JOINT_REGULARIZATION_V597
                * (joints - live_q[:5])
                / maximum_delta,
            ]

        current_residual = residual(live_q[:5])
        solution = least_squares(
            residual,
            start,
            bounds=(lower, upper),
            max_nfev=_ACQUISITION_MAXIMUM_FUNCTION_EVALUATIONS_V597,
            ftol=1.0e-10,
            xtol=1.0e-10,
            gtol=1.0e-10,
        )
        proposed = np.r_[
            np.asarray(solution.x, dtype=np.float64),
            float(self.env.tool_gripper_joint_position_rad),
        ]
        proposed_position, proposed_normal = self._acquisition_pose_v597(
            proposed[:5]
        )
        proposed_residual = residual(proposed[:5])
        objective_before = float(np.dot(current_residual, current_residual))
        objective_after = float(np.dot(proposed_residual, proposed_residual))
        horizontal_norm_after = float(np.linalg.norm(proposed_normal[:2]))
        alignment_after = abs(
            float(np.dot(proposed_normal[:2], direction))
        ) / max(horizontal_norm_after, 1.0e-12)
        joint_delta_linf = float(
            np.max(np.abs(proposed[:5] - live_q[:5]))
        )
        feasible = bool(
            np.all(np.isfinite(proposed))
            and objective_after + 1.0e-10 < objective_before
            and joint_delta_linf > 1.0e-8
            and joint_delta_linf <= maximum_delta + 1.0e-10
        )
        self.last_acquisition_report_v597 = {
            "format": TASK_INDEPENDENT_HOME_ACQUISITION_FORMAT_V597,
            "mode": "bounded_local_partial_alignment",
            "acquisition_subphase": acquisition_subphase,
            "acquisition_active": True,
            "policy_controls_xyz": True,
            "automatic_orientation_constraint": True,
            "automatic_route_or_path": False,
            "privileged_task_axis_used": True,
            "candidate_goal_world_m": target.tolist(),
            "current_tool_position_world_m": current_position.tolist(),
            "proposed_tool_position_world_m": proposed_position.tolist(),
            "desired_face_normal_world": desired_normal.tolist(),
            "current_face_normal_world": current_normal.tolist(),
            "proposed_face_normal_world": proposed_normal.tolist(),
            "normal_heading_alignment_before": alignment_before,
            "normal_heading_alignment_after": alignment_after,
            "tool_height_before_m": height_before,
            "tool_height_after_m": float(proposed_position[2]),
            "tool_precontact_distance_before_m": precontact_distance_before,
            "release_precontact_distance_m": (
                schedule.release_precontact_distance_m
            ),
            "tool_block_xy_distance_before_m": tool_block_xy_distance,
            "automatic_alignment_weight": alignment_weight,
            "orientation_schedule": asdict(schedule),
            "normal_objective_scale": normal_scale,
            "objective_before": objective_before,
            "objective_after": objective_after,
            "joint_delta_linf_rad": joint_delta_linf,
            "joint_target_limit_rad": maximum_delta,
            "solver_success": bool(solution.success),
            "solver_status": int(solution.status),
            "solver_function_evaluations": int(solution.nfev),
            "locally_feasible": feasible,
            "unchanged_v22_guard_still_required": True,
            "production_admission": False,
        }
        if feasible:
            self.acquisition_step_count_v597 += 1
        return {
            "format": TASK_INDEPENDENT_HOME_ACQUISITION_FORMAT_V597,
            "live_state_written": False,
            "feasible": feasible,
            "best": {
                "joint_position_rad": proposed.tolist(),
                "target_xyz_m": target.tolist(),
                "actual_xyz_m": proposed_position.tolist(),
                "tool_face_normal_world": proposed_normal.tolist(),
                "normal_heading_alignment": alignment_after,
                "solver": {
                    "method": "bounded_local_least_squares",
                    "success": bool(solution.success),
                    "status": int(solution.status),
                    "function_evaluations": int(solution.nfev),
                },
            },
            "candidate_count": 1,
        }

    def translate(
        self,
        policy_action: np.ndarray,
        *,
        preserve_latched_target: bool = False,
    ) -> Any:
        translated = super().translate(
            policy_action,
            preserve_latched_target=preserve_latched_target,
        )
        report = deepcopy(self.last_acquisition_report_v597)
        if report:
            self.last_guard_report["home_acquisition_v597"] = report
        if (
            translated.ik_converged
            and report.get("mode") == "bounded_local_partial_alignment"
        ):
            return replace(
                translated,
                face_label=(
                    "task_independent_home_bounded_acquisition_v597"
                ),
            )
        return translated


def reset_stock_home_taskframe_episode_v597(
    env: RealisticEdgeArmEnvV10,
    renderer: MultiViewRendererProtocolV1,
    action_adapter: StockGripperHomeTaskFrameAdapterV597,
    *,
    requested_seed: int,
    obstacle: bool,
    stress: bool,
) -> dict[str, Any]:
    """Sample a task directly at Home, before any physics transition."""

    if type(env) is not RealisticEdgeArmEnvV10:
        raise TypeError("V597 Home reset requires exact RealisticEdgeArmEnvV10")
    if type(action_adapter) is not StockGripperHomeTaskFrameAdapterV597:
        raise TypeError("V597 Home reset requires exact V597 adapter")
    if action_adapter.env is not env:
        raise ValueError("V597 Home adapter belongs to another environment")
    if tuple(renderer.view_names) != VIEW_NAMES:
        raise ValueError("V597 Home reset renderer view order changed")
    if type(requested_seed) is not int or requested_seed < 0:
        raise ValueError("V597 Home reset seed must be non-negative")
    if type(obstacle) is not bool or type(stress) is not bool:
        raise TypeError("V597 Home reset conditions must be booleans")

    rejected: list[dict[str, Any]] = []
    for attempt_index in range(MAX_RESET_ATTEMPTS_V12):
        selected_seed = requested_seed + attempt_index * RESET_RETRY_STRIDE_V12
        restore_stock_gripper_distal_contact_v22(env)
        home = authored_stock_gripper_home_q_v597(env)
        env.reset_task_independent_home_v597(
            home,
            seed=selected_seed,
            obstacle=obstacle,
            stress=stress,
        )
        native_home_reset = deepcopy(env.episode_domain.get("realism_v7", {}))
        initial_distance = float(env.distance_to_target())
        initial_coverage = float(env.block_target_coverage())
        block_before_home = env.block_xy().copy()
        env.last_distance = env.distance_to_target()
        filtered_home, filter_reason = env._safety_filter(home)
        planning_distances = env._tool_planning_signed_distances_for_data(
            env._ids["block_geom"], env.data
        )
        safety_distances = env._tool_safety_signed_distances_for_data(
            env._ids["block_geom"], env.data
        )
        desk_clearance = float(
            env._minimum_tool_safety_signed_distance_for_data(
                env._desk_geom, env.data
            )
        )
        failure_reasons: list[str] = []
        forbidden_home_contacts: list[dict[str, Any]] = []
        penetration_tolerance = float(
            env.contact_feasible_config.reset_penetration_tolerance_m
        )
        for contact_index in range(env.data.ncon):
            contact = env.data.contact[contact_index]
            first = int(contact.geom1)
            second = int(contact.geom2)
            pair = {first, second}
            touches_desk = env._desk_geom in pair
            touches_obstacle = (
                env.obstacle_enabled
                and env._ids["obstacle_geom"] in pair
            )
            other = None
            if touches_desk:
                other = second if first == env._desk_geom else first
            elif touches_obstacle:
                other = (
                    second
                    if first == env._ids["obstacle_geom"]
                    else first
                )
            if (
                other is not None
                and other in env._robot_geoms
                and float(contact.dist) < -penetration_tolerance
            ):
                forbidden_home_contacts.append(
                    {
                        "contact_index": contact_index,
                        "geom1": first,
                        "geom2": second,
                        "distance_m": float(contact.dist),
                    }
                )
        if not np.array_equal(filtered_home, home) or filter_reason:
            failure_reasons.append("home_not_exact_workspace_filter_fixed_point")
        if float(env.data.time) != 0.0:
            failure_reasons.append("physics_advanced_before_policy")
        if int(native_home_reset.get("task_aligned_ik_calls_before_policy", -1)) != 0:
            failure_reasons.append("task_aligned_ik_called_before_policy")
        if int(native_home_reset.get("physics_steps_before_policy", -1)) != 0:
            failure_reasons.append("physics_step_recorded_before_policy")
        if env._tool_block_contacts() != 0:
            failure_reasons.append("home_tool_block_contact")
        if float(np.min(safety_distances)) < 0.0:
            failure_reasons.append("home_tool_block_penetration")
        if desk_clearance < 0.0:
            failure_reasons.append("home_tool_desk_penetration")
        if forbidden_home_contacts:
            failure_reasons.append("home_robot_desk_or_obstacle_penetration")
        if not np.array_equal(env.block_xy(), block_before_home):
            failure_reasons.append("home_install_moved_block")
        if initial_coverage != 0.0:
            failure_reasons.append("task_begins_with_nonzero_target_coverage")
        if failure_reasons:
            rejected.append(
                {
                    "attempt_index": attempt_index,
                    "seed": selected_seed,
                    "failure_reasons": failure_reasons,
                }
            )
            continue

        home_collision_audit = {
            "format": "edgearm-v597-native-home-collision-audit-v1",
            "reset_valid": True,
            "reset_failure_reasons": [],
            "task_aligned_ik_calls_before_policy": 0,
            "physics_steps_before_policy": 0,
            "forbidden_penetration_count": len(forbidden_home_contacts),
            "forbidden_contacts": forbidden_home_contacts,
            "tool_block_contact_count": int(env._tool_block_contacts()),
            "tool_block_signed_distance_m": float(np.min(planning_distances)),
            "tool_block_safety_signed_distance_m": float(np.min(safety_distances)),
            "tool_desk_signed_distance_m": desk_clearance,
            "block_xy_displacement_before_policy_m": 0.0,
        }
        env._reset_collision_audit = deepcopy(home_collision_audit)
        env.episode_domain["realism_v7"]["reset_collision_audit"] = deepcopy(
            home_collision_audit
        )
        try:
            action_adapter.begin_episode(selected_seed)
        except RuntimeError as error:
            rejected.append(
                {
                    "attempt_index": attempt_index,
                    "seed": selected_seed,
                    "failure_reasons": [str(error)],
                }
            )
            continue

        renderer.begin_episode(selected_seed)
        audit = {
            "format": TASK_INDEPENDENT_HOME_RESET_FORMAT_V597,
            "requested_seed": requested_seed,
            "selected_seed": selected_seed,
            "selected_attempt_index": attempt_index,
            "maximum_attempts": MAX_RESET_ATTEMPTS_V12,
            "retry_stride": RESET_RETRY_STRIDE_V12,
            "rejected_attempts": rejected,
            "obstacle": obstacle,
            "stress": stress,
            "source_type": SOURCE_TYPE,
            "task_aligned_privileged_reset": False,
            "task_independent_final_home_reset": True,
            "parent_task_aligned_reset_was_transient_and_discarded": False,
            "native_home_reset_before_first_forward": True,
            "native_home_reset_source": native_home_reset.get(
                "reset_state_source"
            ),
            "task_aligned_ik_calls_before_policy": 0,
            "physics_steps_before_policy": 0,
            "privileged_reset_state_used_for_final_arm_pose": [],
            "deployment_reset_equivalent": False,
            "deployment_reset_block_reason": (
                "physical_home_and_wrist_camera_not_calibrated"
            ),
            "physical_home_calibrated": False,
            "curriculum_reset_approach_actions": 0,
            "policy_must_learn_visual_approach": True,
            "authored_home_joint_position_rad": home.tolist(),
            "initial_tool_position_world_m": env.tool_xyz().tolist(),
            "initial_tool_block_xy_distance_m": float(
                np.linalg.norm(env.tool_xyz()[:2] - env.block_xy())
            ),
            "initial_tool_block_tip_gap_m": float(
                np.min(planning_distances)
            ),
            "initial_tool_block_safety_clearance_m": float(
                np.min(safety_distances)
            ),
            "initial_tool_desk_clearance_m": desk_clearance,
            "initial_block_target_distance_m": initial_distance,
            "initial_target_coverage": initial_coverage,
            "reset_collision_audit": home_collision_audit,
            "stock_follower_unmodified": True,
            "added_contact_tool": False,
            "expert_calls": 0,
            "expert_paths": 0,
            "behavior_cloning_steps": 0,
            "production_admission": False,
            "bulk_vla_data_use_allowed": False,
        }
        action_adapter.home_reset_audit_v597 = deepcopy(audit)
        env.episode_domain["task_independent_home_reset_v597"] = deepcopy(
            audit
        )
        return audit

    reasons = "; ".join(
        f"seed={row['seed']}:{','.join(row['failure_reasons'])}"
        for row in rejected
    )
    raise RuntimeError(
        "V597 task-independent Home reset exhausted deterministic attempts: "
        + reasons
    )


__all__ = [
    "HOME_ACQUISITION_TRANSPORT_ALIGNMENT_V597",
    "HOME_ACQUISITION_TRANSPORT_HEIGHT_MARGIN_M_V597",
    "TASK_INDEPENDENT_HOME_RESET_FORMAT_V597",
    "TASK_INDEPENDENT_HOME_RUNTIME_FORMAT_V597",
    "HomeAcquisitionOrientationScheduleV597",
    "home_acquisition_alignment_weight_v597",
    "StockGripperHomeTaskFrameAdapterV597",
    "authored_stock_gripper_home_q_v597",
    "reset_stock_home_taskframe_episode_v597",
]
