"""Legacy statically audited joint-interpolation curriculum resets.

V622 interpolates from authored Home to the ordinary environment reset pose.
That endpoint was historically described as task-aligned pre-contact, but an
exact V22 comparison proved that claim false: the two reset contracts install
different joint states and a direct Home-to-V22 interpolation is collision
invalid.  V622 v3 therefore records the endpoint mismatch explicitly.  New
training must use a dynamically reachable bridge instead of treating V622 as
continuous with V22 fraction 1.0.

The states remain privileged learning-only resets and are never robot
trajectories, demonstrations, or final VLA data.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
from typing import Any

import mujoco
import numpy as np

from .asymmetric_multiview_ppo_v1 import (
    MAX_RESET_ATTEMPTS_V12,
    RESET_RETRY_STRIDE_V12,
    SOURCE_TYPE,
    VIEW_NAMES,
    MultiViewRendererProtocolV1,
    StockGripperTaskSpaceActionConfigV12,
)
from .sim2real_env_v10 import RealisticEdgeArmEnvV10
from .stock_gripper_push_face_contact_v22 import (
    restore_stock_gripper_distal_contact_v22,
)
from .task_independent_home_reset_v597 import (
    StockGripperHomeTaskFrameAdapterV597,
    authored_stock_gripper_home_q_v597,
)


INTERPOLATED_APPROACH_RESET_FORMAT_V622 = (
    "edgearm-v622-privileged-interpolated-approach-reset-v3"
)
INTERPOLATED_APPROACH_RESET_SOURCE_V622 = (
    "legacy_same_task_home_to_default_reset_joint_interpolation_static_path_audited"
)


class StockGripperApproachTaskFrameAdapterV622(
    StockGripperHomeTaskFrameAdapterV597
):
    """Use the V597 acquisition tracker from an audited intermediate state."""

    def __init__(
        self,
        env: RealisticEdgeArmEnvV10,
        config: StockGripperTaskSpaceActionConfigV12 | None = None,
    ) -> None:
        super().__init__(env, config)
        self.expected_reset_joint_position_v622: np.ndarray | None = None
        self.approach_reset_audit_v622: dict[str, Any] = {}

    def arm_expected_reset_v622(self, joint_position: np.ndarray) -> None:
        expected = np.asarray(joint_position, dtype=np.float64)
        if expected.shape != (6,) or not np.all(np.isfinite(expected)):
            raise ValueError("V622 expected approach reset must be finite [6]")
        self.expected_reset_joint_position_v622 = expected.copy()

    def begin_episode(self, seed: int) -> None:
        expected = self.expected_reset_joint_position_v622
        if expected is None:
            raise RuntimeError("V622 approach adapter was not armed by reset")
        self._begin_episode_from_exact_joint_v622(
            seed,
            expected,
            reset_kind="privileged_interpolated_approach_curriculum",
        )


def _static_interpolation_path_audit_v622(
    env: RealisticEdgeArmEnvV10,
    *,
    home: np.ndarray,
    precontact: np.ndarray,
    selected_fraction: float,
    sample_count: int = 33,
) -> dict[str, Any]:
    if sample_count < 17 or sample_count % 2 != 1:
        raise ValueError("V622 static path audit needs an odd count >= 17")
    fractions = np.linspace(0.0, selected_fraction, sample_count)
    scratch = mujoco.MjData(env.model)
    minimum_block_clearance = float("inf")
    minimum_desk_clearance = float("inf")
    minimum_workspace_margin = float("inf")
    penetrations: list[dict[str, Any]] = []
    penetration_tolerance = float(
        env.contact_feasible_config.reset_penetration_tolerance_m
    )
    for sample_index, fraction in enumerate(fractions):
        mujoco.mj_copyData(scratch, env.model, env.data)
        joint = home + float(fraction) * (precontact - home)
        joint[5] = float(env.tool_gripper_joint_position_rad)
        scratch.qpos[:6] = joint
        scratch.qvel[:6] = 0.0
        mujoco.mj_forward(env.model, scratch)
        block_clearance = float(
            env._minimum_tool_safety_signed_distance_for_data(
                env._ids["block_geom"], scratch
            )
        )
        desk_clearance = float(
            env._minimum_tool_safety_signed_distance_for_data(
                env._desk_geom, scratch
            )
        )
        tool = np.asarray(
            scratch.site_xpos[env._ids["tool_site"]], dtype=np.float64
        )
        margins = np.asarray(
            [
                tool[0] - env.config.workspace_x[0],
                env.config.workspace_x[1] - tool[0],
                tool[1] - env.config.workspace_y[0],
                env.config.workspace_y[1] - tool[1],
                tool[2] - env.config.workspace_z[0],
                env.config.workspace_z[1] - tool[2],
            ],
            dtype=np.float64,
        )
        minimum_block_clearance = min(minimum_block_clearance, block_clearance)
        minimum_desk_clearance = min(minimum_desk_clearance, desk_clearance)
        minimum_workspace_margin = min(
            minimum_workspace_margin, float(np.min(margins))
        )
        for contact_index in range(scratch.ncon):
            contact = scratch.contact[contact_index]
            first = int(contact.geom1)
            second = int(contact.geom2)
            pair = {first, second}
            robot_environment_pair = bool(
                pair.intersection(env._robot_geoms)
                and pair.intersection(
                    {
                        env._desk_geom,
                        env._ids["block_geom"],
                        env._ids["obstacle_geom"],
                    }
                )
            )
            if (
                robot_environment_pair
                and float(contact.dist) < -penetration_tolerance
            ):
                penetrations.append(
                    {
                        "sample_index": sample_index,
                        "fraction": float(fraction),
                        "geom1": first,
                        "geom2": second,
                        "distance_m": float(contact.dist),
                    }
                )
    valid = bool(
        minimum_block_clearance >= 0.0
        and minimum_desk_clearance >= 0.0
        and minimum_workspace_margin >= 0.0
        and not penetrations
    )
    return {
        "format": "edgearm-v622-static-joint-interpolation-path-audit-v1",
        "sample_count": sample_count,
        "selected_home_to_precontact_fraction": selected_fraction,
        "minimum_tool_block_safety_clearance_m": minimum_block_clearance,
        "minimum_tool_desk_safety_clearance_m": minimum_desk_clearance,
        "minimum_tool_workspace_margin_m": minimum_workspace_margin,
        "penetration_count": len(penetrations),
        "penetrations": penetrations[:16],
        "path_valid": valid,
        "dynamic_path_executed": False,
        "training_reset_only": True,
        "production_admission": False,
    }


def reset_stock_interpolated_approach_episode_v622(
    env: RealisticEdgeArmEnvV10,
    renderer: MultiViewRendererProtocolV1,
    action_adapter: StockGripperApproachTaskFrameAdapterV622,
    *,
    requested_seed: int,
    obstacle: bool,
    stress: bool,
    home_to_precontact_fraction: float,
    precontact_standoff_m: float = 0.055,
    precontact_tool_height_m: float = 0.055,
) -> dict[str, Any]:
    """Install a same-task, statically audited intermediate approach state."""

    if type(env) is not RealisticEdgeArmEnvV10:
        raise TypeError("V622 approach reset requires exact V10 environment")
    if type(action_adapter) is not StockGripperApproachTaskFrameAdapterV622:
        raise TypeError("V622 approach reset requires exact V622 adapter")
    if action_adapter.env is not env:
        raise ValueError("V622 approach adapter belongs to another environment")
    if tuple(renderer.view_names) != VIEW_NAMES:
        raise ValueError("V622 reset renderer view order changed")
    if type(requested_seed) is not int or requested_seed < 0:
        raise ValueError("V622 reset seed must be non-negative")
    if type(obstacle) is not bool or type(stress) is not bool:
        raise TypeError("V622 reset conditions must be boolean")
    # Values just below one are intentionally supported for fine reverse
    # curriculum.  Fraction 1.0 remains reserved for the independently
    # audited exact V22 precontact reset, so the two reset contracts cannot be
    # silently conflated.
    if not 0.05 <= home_to_precontact_fraction <= 0.995:
        raise ValueError("V622 approach fraction must lie in [0.05, 0.995]")
    if not 0.030 <= precontact_standoff_m <= 0.100:
        raise ValueError("V627 pre-contact standoff is invalid")
    if not 0.045 <= precontact_tool_height_m <= 0.100:
        raise ValueError("V627 pre-contact height is invalid")

    rejected: list[dict[str, Any]] = []
    for attempt_index in range(MAX_RESET_ATTEMPTS_V12):
        proposal_seed = requested_seed + attempt_index * RESET_RETRY_STRIDE_V12
        restore_stock_gripper_distal_contact_v22(env)
        try:
            env.reset(seed=proposal_seed, obstacle=obstacle, stress=stress)
        except RuntimeError as error:
            rejected.append(
                {
                    "attempt_index": attempt_index,
                    "seed": proposal_seed,
                    "failure_reasons": [f"legacy_reference_probe:{error}"],
                }
            )
            continue
        probe_domain = deepcopy(env.episode_domain.get("realism_v7", {}))
        selected_seed = int(
            probe_domain.get("accepted_candidate_seed", proposal_seed)
        )
        precontact = np.asarray(env.data.qpos[:6], dtype=np.float64).copy()
        home = authored_stock_gripper_home_q_v597(env)
        block = env.block_xy().copy()
        target = env.target_xy.copy()
        initial_distance = float(env.distance_to_target())
        initial_coverage = float(env.block_target_coverage())
        selected = home + float(home_to_precontact_fraction) * (
            precontact - home
        )
        selected[5] = float(env.tool_gripper_joint_position_rad)
        path_audit = _static_interpolation_path_audit_v622(
            env,
            home=home,
            precontact=precontact,
            selected_fraction=float(home_to_precontact_fraction),
        )
        if not bool(path_audit["path_valid"]):
            rejected.append(
                {
                    "attempt_index": attempt_index,
                    "seed": selected_seed,
                    "failure_reasons": ["static_interpolation_path_invalid"],
                    "path_audit": path_audit,
                }
            )
            continue

        restore_stock_gripper_distal_contact_v22(env)
        try:
            env.reset_task_independent_home_v597(
                selected,
                seed=selected_seed,
                obstacle=obstacle,
                stress=stress,
            )
        except RuntimeError as error:
            rejected.append(
                {
                    "attempt_index": attempt_index,
                    "seed": selected_seed,
                    "failure_reasons": [f"intermediate_install:{error}"],
                }
            )
            continue
        if not (
            np.allclose(env.block_xy(), block, rtol=0.0, atol=1.0e-10)
            and np.allclose(env.target_xy, target, rtol=0.0, atol=1.0e-12)
            and np.array_equal(env.data.qpos[:6], selected)
            and float(env.data.time) == 0.0
        ):
            rejected.append(
                {
                    "attempt_index": attempt_index,
                    "seed": selected_seed,
                    "failure_reasons": ["same_task_or_exact_state_identity_changed"],
                }
            )
            continue
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
        filtered, filter_reason = env._safety_filter(selected)
        failure_reasons: list[str] = []
        if not np.array_equal(filtered, selected) or filter_reason:
            failure_reasons.append("intermediate_not_workspace_filter_fixed_point")
        if env._tool_block_contacts() != 0:
            failure_reasons.append("intermediate_tool_block_contact")
        if float(np.min(safety_distances)) < 0.0:
            failure_reasons.append("intermediate_tool_block_penetration")
        if desk_clearance < 0.0:
            failure_reasons.append("intermediate_tool_desk_penetration")
        if initial_coverage != 0.0 or env.block_target_coverage() != 0.0:
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

        tool_position = env.tool_xyz().copy()
        tool_pose = np.asarray(env.observation()["tool_pose"], dtype=np.float64)
        if tool_pose.shape != (12,) or not np.all(np.isfinite(tool_pose)):
            raise RuntimeError("V627 installed tool pose is invalid")
        task_direction = target - block
        task_direction_norm = float(np.linalg.norm(task_direction))
        if task_direction_norm <= 1.0e-7:
            task_forward = np.asarray((1.0, 0.0), dtype=np.float64)
        else:
            task_forward = task_direction / task_direction_norm
        precontact_point = np.asarray(
            (
                block[0] - precontact_standoff_m * task_forward[0],
                block[1] - precontact_standoff_m * task_forward[1],
                precontact_tool_height_m,
            ),
            dtype=np.float64,
        )
        precontact_distance = float(
            np.linalg.norm(tool_position - precontact_point)
        )
        tool_rotation = tool_pose[3:12].reshape(3, 3)
        broad_face_normal_xy = tool_rotation[:, 1][:2]
        broad_face_norm = float(np.linalg.norm(broad_face_normal_xy))
        precontact_alignment = float(
            abs(np.dot(broad_face_normal_xy, task_forward))
            / max(broad_face_norm, 1.0e-7)
        )
        precontact_alignment = float(
            np.clip(precontact_alignment, 0.0, 1.0)
        )

        realism = env.episode_domain["realism_v7"]
        realism.update(
            {
                "reset_state_source": INTERPOLATED_APPROACH_RESET_SOURCE_V622,
                "task_aligned_privileged_reset": True,
                "deployment_reset_equivalent": False,
                "reset_privileged_state_used": [
                    "block_xy",
                    "target_xy",
                    "task_aligned_precontact_joint_position",
                ],
                "reset_approach_joint_position_rad": selected.tolist(),
                "reset_home_joint_position_rad": home.tolist(),
                "reset_precontact_joint_position_rad": precontact.tolist(),
                "task_aligned_precontact_endpoint_verified": False,
                "legacy_reference_is_true_v22_precontact": False,
                "home_to_precontact_fraction": float(
                    home_to_precontact_fraction
                ),
                "task_aligned_ik_calls_before_policy": int(
                    probe_domain.get("task_aligned_ik_calls_before_policy", 1)
                ),
                "physics_steps_before_policy": int(
                    probe_domain.get("physics_steps_before_policy", 0)
                ),
                "final_installed_state_time_seconds": 0.0,
                "production_admission": False,
            }
        )
        env._reset_collision_audit = {
            "format": "edgearm-v622-intermediate-reset-collision-audit-v1",
            "reset_valid": True,
            "reset_failure_reasons": [],
            "tool_block_contact_count": int(env._tool_block_contacts()),
            "tool_block_signed_distance_m": float(np.min(planning_distances)),
            "tool_block_safety_signed_distance_m": float(
                np.min(safety_distances)
            ),
            "tool_desk_signed_distance_m": desk_clearance,
            "static_interpolation_path_audit": path_audit,
        }
        realism["reset_collision_audit"] = deepcopy(env._reset_collision_audit)
        action_adapter.arm_expected_reset_v622(selected)
        try:
            action_adapter.begin_episode(selected_seed)
        except RuntimeError as error:
            rejected.append(
                {
                    "attempt_index": attempt_index,
                    "seed": selected_seed,
                    "failure_reasons": [f"guarded_runtime:{error}"],
                }
            )
            continue
        renderer.begin_episode(selected_seed)
        audit = {
            "format": INTERPOLATED_APPROACH_RESET_FORMAT_V622,
            "requested_seed": requested_seed,
            "selected_seed": selected_seed,
            "selected_attempt_index": attempt_index,
            "maximum_attempts": MAX_RESET_ATTEMPTS_V12,
            "retry_stride": RESET_RETRY_STRIDE_V12,
            "rejected_attempts": rejected,
            "obstacle": obstacle,
            "stress": stress,
            "source_type": SOURCE_TYPE,
            "home_to_precontact_fraction": float(
                home_to_precontact_fraction
            ),
            "authored_home_joint_position_rad": home.tolist(),
            "task_aligned_precontact_joint_position_rad": precontact.tolist(),
            "legacy_reference_joint_position_rad": precontact.tolist(),
            "task_aligned_precontact_endpoint_verified": False,
            "legacy_reference_is_true_v22_precontact": False,
            "installed_joint_position_rad": selected.tolist(),
            "initial_tool_position_world_m": env.tool_xyz().tolist(),
            "initial_tool_precontact_distance_m_v627": precontact_distance,
            "initial_precontact_face_alignment_v627": precontact_alignment,
            "precontact_point_world_m_v627": precontact_point.tolist(),
            "precontact_standoff_m_v627": float(precontact_standoff_m),
            "precontact_tool_height_m_v627": float(
                precontact_tool_height_m
            ),
            "initial_tool_block_xy_distance_m": float(
                np.linalg.norm(env.tool_xyz()[:2] - env.block_xy())
            ),
            "initial_tool_block_tip_gap_m": float(
                np.min(planning_distances)
            ),
            "initial_block_target_distance_m": initial_distance,
            "initial_target_coverage": initial_coverage,
            "static_interpolation_path_audit": path_audit,
            "task_aligned_privileged_reset": True,
            "task_independent_final_home_reset": False,
            "policy_must_learn_visual_approach": True,
            "curriculum_reset_approach_actions": 0,
            "intermediate_state_is_reset_not_executed_trajectory": True,
            "eligible_for_final_data": False,
            "bulk_vla_data_use_allowed": False,
            "expert_calls": 0,
            "expert_paths": 0,
            "behavior_cloning_steps": 0,
            "deployment_reset_equivalent": False,
            "production_admission": False,
            "adapter_runtime": deepcopy(
                env.episode_domain["stock_gripper_taskframe_runtime_v22"]
            ),
            "action_adapter_config": asdict(action_adapter.config),
        }
        action_adapter.approach_reset_audit_v622 = deepcopy(audit)
        env.episode_domain["interpolated_approach_reset_v622"] = deepcopy(audit)
        return audit

    reasons = "; ".join(
        f"seed={row['seed']}:{','.join(row['failure_reasons'])}"
        for row in rejected
    )
    raise RuntimeError(
        "V622 interpolated approach reset exhausted deterministic attempts: "
        + reasons
    )


__all__ = [
    "INTERPOLATED_APPROACH_RESET_FORMAT_V622",
    "INTERPOLATED_APPROACH_RESET_SOURCE_V622",
    "StockGripperApproachTaskFrameAdapterV622",
    "reset_stock_interpolated_approach_episode_v622",
]
