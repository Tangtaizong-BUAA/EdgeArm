"""Certified long-range task sampling with a final exact-Home arm state.

Long-range geometry can place an otherwise valid block/target pair on an IK
branch that the stock follower cannot safely use.  V649 rejects those tasks at
reset time with the exact V22 precontact solver, discards the privileged probe
pose, and recreates the accepted task at the unchanged V597 Home state.

The probe supplies no policy action, route, waypoint, demonstration, reward,
or replay row.  It is a task-feasibility constraint only.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
from typing import Any

import numpy as np

from .asymmetric_multiview_ppo_v1 import (
    MAX_CURRICULUM_RESET_ATTEMPTS_V16,
    RESET_RETRY_STRIDE_V12,
    SOURCE_TYPE,
    VIEW_NAMES,
    MultiViewRendererProtocolV1,
    canonical_sha256_v1,
)
from .reverse_curriculum_v26 import (
    EXTENDED_RANGE_STAGE_INDEX_V648,
    LONG_RANGE_STAGE_INDEX_V648,
    reverse_curriculum_stage_v26,
)
from .sim2real_env_v10 import RealisticEdgeArmEnvV10
from .stock_gripper_taskframe_v22 import StockGripperTaskFrameAdapterV22
from .task_independent_home_reset_v597 import (
    StockGripperHomeTaskFrameAdapterV597,
    authored_stock_gripper_home_q_v597,
    reset_stock_home_taskframe_episode_v597,
)


CERTIFIED_LONG_RANGE_HOME_RESET_FORMAT_V649 = (
    "edgearm-v649-certified-long-range-final-home-reset-v1"
)
_ALLOWED_STAGES_V649 = {
    LONG_RANGE_STAGE_INDEX_V648,
    EXTENDED_RANGE_STAGE_INDEX_V648,
}


class CertifiedLongRangeResetInfeasibleV649(RuntimeError):
    """Raised when no sampled long-range task passes the stock reset gate."""


def _task_identity_v649(env: RealisticEdgeArmEnvV10) -> dict[str, Any]:
    block = np.asarray(env.block_xy(), dtype=np.float64)
    target = np.asarray(env.target_xy, dtype=np.float64)
    payload = {
        "block_xy_m": block.tolist(),
        "target_xy_m": target.tolist(),
        "center_distance_m": float(np.linalg.norm(target - block)),
        "initial_target_coverage": float(env.block_target_coverage()),
    }
    return {**payload, "sha256": canonical_sha256_v1(payload)}


def reset_stock_home_certified_long_range_episode_v649(
    env: RealisticEdgeArmEnvV10,
    renderer: MultiViewRendererProtocolV1,
    action_adapter: StockGripperHomeTaskFrameAdapterV597,
    *,
    requested_seed: int,
    obstacle: bool,
    stress: bool,
) -> dict[str, Any]:
    """Return a reachable long-range task at the exact authored Home pose."""

    if type(env) is not RealisticEdgeArmEnvV10:
        raise TypeError("V649 reset requires the exact V10 plant")
    if type(action_adapter) is not StockGripperHomeTaskFrameAdapterV597:
        raise TypeError("V649 reset requires the exact V597 Home adapter")
    if action_adapter.env is not env:
        raise ValueError("V649 Home adapter belongs to another environment")
    if tuple(renderer.view_names) != VIEW_NAMES:
        raise ValueError("V649 reset renderer view order changed")
    if type(requested_seed) is not int or requested_seed < 0:
        raise ValueError("V649 requested seed must be non-negative")
    if obstacle or stress:
        raise ValueError("V649 long-range reset is obstacle-free nominal only")
    stage_index = env.joint_bounded_config.reverse_curriculum_stage_v26
    if stage_index not in _ALLOWED_STAGES_V649:
        raise ValueError("V649 requires an exact V648 long-range stage")
    stage = reverse_curriculum_stage_v26(int(stage_index))
    if tuple(action_adapter.config.curriculum_reset_tip_gap_band_m or ()) != (
        stage.tip_gap_range_m
    ):
        raise ValueError("V649 adapter tip-gap band disagrees with its stage")

    rejected: list[dict[str, Any]] = []
    for attempt_index in range(MAX_CURRICULUM_RESET_ATTEMPTS_V16):
        candidate_seed = (
            requested_seed + attempt_index * RESET_RETRY_STRIDE_V12
        )
        initial_home = reset_stock_home_taskframe_episode_v597(
            env,
            renderer,
            action_adapter,
            requested_seed=candidate_seed,
            obstacle=False,
            stress=False,
        )
        selected_seed = int(initial_home["selected_seed"])
        identity_before = _task_identity_v649(env)
        try:
            probe_adapter = StockGripperTaskFrameAdapterV22(
                env, action_adapter.config
            )
            probe_adapter.begin_episode(selected_seed)
        except RuntimeError as error:
            rejected.append(
                {
                    "attempt_index": attempt_index,
                    "candidate_requested_seed": candidate_seed,
                    "candidate_selected_seed": selected_seed,
                    "task_identity": identity_before,
                    "error_type": type(error).__name__,
                    "reason": str(error),
                }
            )
            continue

        probe = {
            "format": "edgearm-v649-discarded-v22-precontact-probe-v1",
            "task_identity": identity_before,
            "tool_xyz_m": env.tool_xyz().tolist(),
            "joint_position_rad": np.asarray(
                env.data.qpos[:6], dtype=np.float64
            ).tolist(),
            "minimum_tool_block_planning_distance_m": float(
                np.min(
                    env._tool_planning_signed_distances_for_data(
                        env._ids["block_geom"], env.data
                    )
                )
            ),
            "reset_runtime_audit": deepcopy(
                env.episode_domain.get(
                    "stock_gripper_taskframe_reset_v12", {}
                )
            ),
            "policy_action_used": False,
            "expert_action_used": False,
            "waypoint_or_route_used": False,
            "replay_rows_created": 0,
            "final_arm_state_used": False,
            "production_admission": False,
        }

        final_home = reset_stock_home_taskframe_episode_v597(
            env,
            renderer,
            action_adapter,
            requested_seed=selected_seed,
            obstacle=False,
            stress=False,
        )
        identity_after = _task_identity_v649(env)
        authored_home = authored_stock_gripper_home_q_v597(env)
        if (
            int(final_home["selected_seed"]) != selected_seed
            or identity_after != identity_before
            or not np.array_equal(env.data.qpos[:6], authored_home)
            or float(env.data.time) != 0.0
            or identity_after["initial_target_coverage"] != 0.0
        ):
            raise RuntimeError(
                "V649 accepted probe did not restore the identical exact-Home task"
            )
        audit = {
            "format": CERTIFIED_LONG_RANGE_HOME_RESET_FORMAT_V649,
            "requested_seed": requested_seed,
            "selected_seed": selected_seed,
            "selected_attempt_index": attempt_index,
            "retry_stride": RESET_RETRY_STRIDE_V12,
            "maximum_attempts": MAX_CURRICULUM_RESET_ATTEMPTS_V16,
            "rejected_candidates": rejected,
            "stage": asdict(stage),
            "task_identity": identity_after,
            "initial_home_reset_audit": initial_home,
            "discarded_precontact_probe": probe,
            "final_home_reset_audit": final_home,
            "task_aligned_privileged_reset": False,
            "task_independent_final_home_reset": True,
            "initial_tool_block_xy_distance_m": final_home[
                "initial_tool_block_xy_distance_m"
            ],
            "initial_tool_block_tip_gap_m": final_home[
                "initial_tool_block_tip_gap_m"
            ],
            "initial_tool_block_safety_clearance_m": final_home[
                "initial_tool_block_safety_clearance_m"
            ],
            "initial_tool_desk_clearance_m": final_home[
                "initial_tool_desk_clearance_m"
            ],
            "policy_must_learn_visual_approach": True,
            "deployment_reset_equivalent": False,
            "curriculum_reset_approach_actions": 0,
            "final_arm_pose_source": (
                "authored_task_independent_home_recreated_after_probe"
            ),
            "privileged_probe_pose_discarded": True,
            "privileged_probe_used_as_policy_input": False,
            "privileged_probe_used_as_action_label": False,
            "policy_must_learn_complete_home_to_contact_motion": True,
            "physics_steps_before_final_policy": 0,
            "expert_calls": 0,
            "expert_paths": 0,
            "behavior_cloning_steps": 0,
            "source_type": SOURCE_TYPE,
            "bulk_vla_data_use_allowed": False,
            "production_admission": False,
        }
        env.episode_domain[
            "certified_long_range_home_reset_v649"
        ] = deepcopy(audit)
        action_adapter.home_reset_audit_v597 = deepcopy(audit)
        return audit

    reasons = "; ".join(
        f"seed={row['candidate_selected_seed']}:{row['error_type']}"
        for row in rejected
    )
    raise CertifiedLongRangeResetInfeasibleV649(
        "V649 exhausted certified long-range task candidates: " + reasons
    )


__all__ = [
    "CERTIFIED_LONG_RANGE_HOME_RESET_FORMAT_V649",
    "CertifiedLongRangeResetInfeasibleV649",
    "reset_stock_home_certified_long_range_episode_v649",
]
