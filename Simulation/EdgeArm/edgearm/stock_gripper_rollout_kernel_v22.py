"""Execution-kernel adapter for collecting V22 simulator RL rollouts."""

from __future__ import annotations

from typing import Any

import mujoco
import numpy as np

from .asymmetric_multiview_ppo_v1 import (
    MultiViewRendererProtocolV1,
    StockGripperTaskFrameAdapterV13,
)
from .scratch_ppo_v6_candidate import (
    ScratchPotentialRewardV6Candidate,
    ScratchSafetyEvidenceV6Candidate,
)
from .sim2real_env_v10 import RealisticEdgeArmEnvV10
from .stock_gripper_action_guard_v4 import (
    STOCK_GRIPPER_ACTION_GUARD_FORMAT_V4,
)
from .stock_gripper_push_face_contact_v22 import (
    STOCK_GRIPPER_PUSH_FACE_CONTACT_FORMAT_V22,
)
from .stock_gripper_reward_v22 import StockGripperPotentialRewardV22
from .stock_gripper_taskframe_v22 import (
    StockGripperTaskFrameAdapterV22,
    reset_stock_taskframe_episode_v22,
    transition_contact_telemetry_v22,
)


STOCK_GRIPPER_ROLLOUT_KERNEL_FORMAT_V22 = "edgearm-stock-gripper-semantic-push-face-rollout-kernel-v22"
STOCK_GRIPPER_ROLLOUT_FORMAT_V22 = "edgearm-sim-rl-scratch-online-semantic-push-face-rollout-v22"
STOCK_GRIPPER_EVALUATION_FORMAT_V22 = "edgearm-semantic-push-face-heldout-evaluation-v22"


class StockGripperRolloutKernelV22:
    """Bind the shared visual collector to the complete V22 contract."""

    format = STOCK_GRIPPER_ROLLOUT_KERNEL_FORMAT_V22
    rollout_format = STOCK_GRIPPER_ROLLOUT_FORMAT_V22
    evaluation_format = STOCK_GRIPPER_EVALUATION_FORMAT_V22
    safety_only_geom_count = 88
    contact_candidate_geom_count = 8
    contact_identity_format = STOCK_GRIPPER_PUSH_FACE_CONTACT_FORMAT_V22
    safety_guard_format = STOCK_GRIPPER_ACTION_GUARD_FORMAT_V4

    def validate(
        self,
        env: RealisticEdgeArmEnvV10,
        action_adapter: StockGripperTaskFrameAdapterV13,
        reward: ScratchPotentialRewardV6Candidate,
    ) -> None:
        if type(action_adapter) is not StockGripperTaskFrameAdapterV22:
            raise TypeError("V22 rollout kernel requires exact adapter V22")
        if action_adapter.env is not env:
            raise ValueError("V22 rollout adapter belongs to another environment")
        if type(reward) is not StockGripperPotentialRewardV22:
            raise TypeError("V22 rollout kernel requires exact V22 reward")

    def reset_episode(
        self,
        env: RealisticEdgeArmEnvV10,
        renderer: MultiViewRendererProtocolV1,
        action_adapter: StockGripperTaskFrameAdapterV13,
        *,
        requested_seed: int,
        obstacle: bool,
        stress: bool,
    ) -> dict[str, Any]:
        if type(action_adapter) is not StockGripperTaskFrameAdapterV22:
            raise TypeError("V22 reset requires exact adapter V22")
        return reset_stock_taskframe_episode_v22(
            env,
            renderer,
            action_adapter,
            requested_seed=requested_seed,
            obstacle=obstacle,
            stress=stress,
        )

    def transition_contact(
        self,
        info: dict[str, Any],
        *,
        block_before_xy_m: np.ndarray,
        block_after_xy_m: np.ndarray,
    ) -> dict[str, Any]:
        return transition_contact_telemetry_v22(
            info,
            block_before_xy_m=block_before_xy_m,
            block_after_xy_m=block_after_xy_m,
        )

    def transition_safety(
        self,
        reward: ScratchPotentialRewardV6Candidate,
        env: RealisticEdgeArmEnvV10,
        info: dict[str, Any],
    ) -> ScratchSafetyEvidenceV6Candidate:
        if type(reward) is not StockGripperPotentialRewardV22:
            raise TypeError("V22 safety requires exact V22 reward")
        return reward.evaluate_transition_safety(env, info)

    def static_safety(
        self,
        reward: ScratchPotentialRewardV6Candidate,
        env: RealisticEdgeArmEnvV10,
    ) -> tuple[ScratchSafetyEvidenceV6Candidate, float]:
        """Fail-closed safety evidence when no command/trace was executed."""

        if type(reward) is not StockGripperPotentialRewardV22:
            raise TypeError("V22 static safety requires exact V22 reward")
        mujoco.mj_forward(env.model, env.data)
        safety_ids = tuple(int(value) for value in env._ids["tool_safety_geoms"])
        contact_ids = tuple(int(value) for value in env._ids["tool_contact_geoms"])
        if (
            len(safety_ids) != 96
            or len(set(safety_ids)) != 96
            or len(contact_ids) != 8
            or len(set(contact_ids)) != 8
            or not set(contact_ids).issubset(safety_ids)
        ):
            raise RuntimeError("V22 static safety lost the 96/8 identity")
        contact_indices = tuple(safety_ids.index(value) for value in contact_ids)
        safety_only_indices = tuple(index for index in range(96) if index not in set(contact_indices))
        if len(safety_only_indices) != 88:
            raise RuntimeError("V22 static safety did not resolve 88 parts")
        block = np.asarray(
            env._tool_safety_signed_distances_for_data(env._ids["block_geom"], env.data),
            dtype=np.float64,
        )
        desk = np.asarray(
            env._tool_safety_signed_distances_for_data(env._desk_geom, env.data),
            dtype=np.float64,
        )
        if (
            block.shape != (96,)
            or desk.shape != (96,)
            or not np.all(np.isfinite(block))
            or not np.all(np.isfinite(desk))
        ):
            raise RuntimeError("V22 static safety geometry is invalid")
        safety_only = block[np.asarray(safety_only_indices, dtype=np.int64)]
        limiting_local = int(np.argmin(safety_only))
        limiting_index = int(safety_only_indices[limiting_local])
        minimum_safety = float(safety_only[limiting_local])
        minimum_desk = float(np.min(desk))
        contact_minimums = tuple(float(block[index]) for index in contact_indices)
        unauthorized_depth = max(
            (-value for value in contact_minimums if value < -reward.config.penetration_tolerance_m),
            default=0.0,
        )
        unauthorized_count = int(
            np.count_nonzero(np.asarray(contact_minimums) < -reward.config.penetration_tolerance_m)
        )
        safety_depth = max(
            reward.config.safety_only_block_clearance_m - minimum_safety,
            0.0,
        )
        unauthorized_cost_depth = max(
            unauthorized_depth - reward.config.penetration_tolerance_m,
            0.0,
        )
        desk_depth = max(
            -minimum_desk - reward.config.penetration_tolerance_m,
            0.0,
        )
        normalized_safety = float(
            np.clip(
                safety_depth / reward.config.safety_depth_scale_m,
                0.0,
                1.0,
            )
        )
        normalized_unauthorized = float(
            np.clip(
                unauthorized_cost_depth / reward.config.safety_depth_scale_m,
                0.0,
                1.0,
            )
        )
        normalized_desk = float(
            np.clip(
                desk_depth / reward.config.safety_depth_scale_m,
                0.0,
                1.0,
            )
        )
        normalized = max(
            normalized_safety,
            normalized_unauthorized,
            normalized_desk,
        )
        safety_violation = bool(minimum_safety < reward.config.safety_only_block_clearance_m)
        desk_penetration = bool(minimum_desk < -reward.config.penetration_tolerance_m)
        reasons: list[str] = []
        if safety_violation:
            reasons.append("safety_only_block_clearance_violation")
        if unauthorized_count:
            reasons.append("unauthorized_contact_part_penetration")
        if desk_penetration:
            reasons.append("full_safety_union_desk_penetration")
        evidence = ScratchSafetyEvidenceV6Candidate(
            physics_substeps=0,
            safety_geom_count=96,
            safety_only_geom_count=88,
            contact_safety_geom_indices=contact_indices,  # type: ignore[arg-type]
            minimum_safety_only_block_signed_distance_m=minimum_safety,
            limiting_safety_only_geom_index=limiting_index,
            minimum_contact_part_block_signed_distance_by_role_m=(
                contact_minimums  # type: ignore[arg-type]
            ),
            minimum_full_safety_desk_signed_distance_m=minimum_desk,
            safety_only_clearance_violation=safety_violation,
            safety_only_penetration=bool(minimum_safety < -reward.config.penetration_tolerance_m),
            unauthorized_contact_part_penetration=bool(unauthorized_count),
            full_safety_desk_penetration=desk_penetration,
            unauthorized_contact_part_penetration_count=unauthorized_count,
            authorized_contact_part_penetration_count=0,
            normalized_safety_only_block_cost=normalized_safety,
            normalized_unauthorized_contact_cost=normalized_unauthorized,
            normalized_full_safety_desk_cost=normalized_desk,
            normalized_safety_cost=normalized,
            safety_penalty=float(-reward.config.privileged_safety_penalty_coefficient * normalized),
            hard_safety_violation=bool(reasons),
            hard_safety_reason="+".join(reasons),
            privileged_reward_input=True,
            maximum_unauthorized_contact_penetration_depth_m=(unauthorized_depth),
        )
        return evidence, minimum_safety

    def minimum_safety_only_clearance(
        self,
        telemetry: dict[str, Any],
    ) -> float:
        return float(telemetry["minimum_executed_safety_only_block_clearance_m"])


__all__ = [
    "STOCK_GRIPPER_EVALUATION_FORMAT_V22",
    "STOCK_GRIPPER_ROLLOUT_FORMAT_V22",
    "STOCK_GRIPPER_ROLLOUT_KERNEL_FORMAT_V22",
    "StockGripperRolloutKernelV22",
]
