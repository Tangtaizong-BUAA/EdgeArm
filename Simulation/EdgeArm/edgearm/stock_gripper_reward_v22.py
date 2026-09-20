"""V22 task reward and 96/8/88 privileged safety adjudication.

The bounded V6 task potential and canonical float32 shaping arithmetic are
retained.  Only the safety-evidence parser is versioned: it recognizes the
eight conditional broad-face contact candidates and treats penetration as
authorized only on a same-role raw contact whose complete geometric semantic
classification is valid.  The other 88 CAD-derived parts keep the original
positive-clearance hard gate.
"""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from typing import Any

import mujoco
import numpy as np

from .contact_telemetry_v1 import (
    PHYSICS_SUBSTEP_CONTACT_FORMAT,
    TOOL_CONTACT_IDENTITY_FORMAT,
    TOOL_SAFETY_GEOM_ORDER_FORMAT,
    stable_tool_safety_geom_order_sha256,
)
from .scratch_ppo_v6_candidate import (
    ScratchPotentialEvaluationV6Candidate,
    ScratchPotentialRewardV6Candidate,
    ScratchPotentialRewardV6CandidateConfig,
    ScratchSafetyEvidenceV6Candidate,
)
from .sim2real_env_v10 import RealisticEdgeArmEnvV10
from .stock_gripper_push_face_contact_v22 import (
    STOCK_GRIPPER_PUSH_FACE_CONTACT_FORMAT_V22,
    stock_gripper_distal_reference_identity_v22,
    stock_gripper_push_face_profile_v22,
)


STOCK_GRIPPER_REWARD_FORMAT_V22 = "edgearm-v22-semantic-push-face-canonical-f32-task-reward-v1"


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _finite_matrix(
    value: object,
    *,
    name: str,
    shape: tuple[int, int],
) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != shape or not np.all(np.isfinite(array)):
        raise RuntimeError(f"V22 {name} must have finite shape {shape}")
    return array


def _count_matrix(
    value: object,
    *,
    name: str,
    shape: tuple[int, int],
) -> np.ndarray:
    raw = np.asarray(value)
    if raw.shape != shape or not np.issubdtype(raw.dtype, np.integer):
        raise RuntimeError(f"V22 {name} must have integer shape {shape}")
    array = raw.astype(np.int64, copy=False)
    if np.any(array < 0):
        raise RuntimeError(f"V22 {name} contains a negative count")
    return array


def _binary_matrix(
    value: object,
    *,
    name: str,
    shape: tuple[int, int],
) -> np.ndarray:
    raw = np.asarray(value)
    if raw.shape != shape:
        raise RuntimeError(f"V22 {name} must have shape {shape}")
    if not np.all((raw == 0) | (raw == 1)):
        raise RuntimeError(f"V22 {name} must be binary")
    return raw.astype(bool, copy=False)


class StockGripperPotentialRewardV22(ScratchPotentialRewardV6Candidate):
    """V6 potential/arithmetic with V22 role-preserving safety evidence."""

    version = STOCK_GRIPPER_REWARD_FORMAT_V22

    def __init__(
        self,
        config: ScratchPotentialRewardV6CandidateConfig | None = None,
    ) -> None:
        super().__init__(config)
        self.config_sha256 = _canonical_sha256(
            {
                "format": STOCK_GRIPPER_REWARD_FORMAT_V22,
                "base_reward_version": super().version,
                "base_reward_config": asdict(self.config),
                "contact_identity_format": (STOCK_GRIPPER_PUSH_FACE_CONTACT_FORMAT_V22),
                "safety_geom_count": 96,
                "contact_candidate_geom_count": 8,
                "safety_only_geom_count": 88,
                "authorization": ("same_role_raw_contact_and_all_geometric_semantics_valid"),
            }
        )

    def evaluate(
        self,
        env: RealisticEdgeArmEnvV10,
    ) -> ScratchPotentialEvaluationV6Candidate:
        """Evaluate the unchanged two-jaw task potential under V22 safety.

        V6's bounded contact-progress term is intentionally defined on the two
        distal planning references, not on collision authorization.  Present
        that immutable metadata view only while computing the pure potential;
        no MuJoCo state, geometry, controller, or episode record is changed.
        """

        exact = self._require_exact_v10(env)
        active_ids = tuple(int(value) for value in exact._ids["tool_contact_geoms"])
        active_roles = tuple(str(value) for value in exact._ids["tool_contact_geom_roles"])
        if len(active_ids) != 8 or len(active_roles) != 8:
            raise RuntimeError("V22 task potential requires the active 8-part identity")
        distal_ids, distal_roles = stock_gripper_distal_reference_identity_v22(exact)
        try:
            exact._ids["tool_contact_geoms"] = distal_ids
            exact._ids["tool_contact_geom_roles"] = distal_roles
            return super().evaluate(exact)
        finally:
            exact._ids["tool_contact_geoms"] = active_ids
            exact._ids["tool_contact_geom_roles"] = active_roles

    def evaluate_transition_safety(
        self,
        env: RealisticEdgeArmEnvV10,
        info: dict[str, Any],
    ) -> ScratchSafetyEvidenceV6Candidate:
        exact = self._require_exact_v10(env)
        profile = stock_gripper_push_face_profile_v22(exact)
        if profile["contact_candidate_geom_count"] != 8 or profile["safety_only_geom_count"] != 88:
            raise RuntimeError("V22 reward lost its 96/8/88 contact profile")
        trace = info.get("physics_substep_contact_v1")
        if not isinstance(trace, dict):
            raise RuntimeError("V22 reward requires physics-substep telemetry")
        if trace.get("format") != PHYSICS_SUBSTEP_CONTACT_FORMAT:
            raise RuntimeError("V22 reward contact telemetry format changed")
        if (
            trace.get("contact_identity_format") != STOCK_GRIPPER_PUSH_FACE_CONTACT_FORMAT_V22
            or trace.get("base_contact_identity_format") != TOOL_CONTACT_IDENTITY_FORMAT
        ):
            raise RuntimeError("V22 reward contact identity format changed")
        if trace.get("simulator_privileged_truth") is not True:
            raise RuntimeError("V22 reward input must declare simulator privilege")
        if trace.get("tool_safety_geom_order_format") != TOOL_SAFETY_GEOM_ORDER_FORMAT:
            raise RuntimeError("V22 safety geometry order format changed")

        expected_safety_ids = tuple(int(value) for value in exact._ids["tool_safety_geoms"])
        expected_contact_ids = tuple(int(value) for value in exact._ids["tool_contact_geoms"])
        expected_roles = tuple(str(value) for value in exact._ids["tool_contact_geom_roles"])
        trace_safety_ids = tuple(int(value) for value in trace.get("tool_safety_geom_ids", ()))
        trace_contact_ids = tuple(int(value) for value in trace.get("tool_contact_geom_ids", ()))
        trace_roles = tuple(str(value) for value in trace.get("tool_contact_role_names", ()))
        trace_safety_names = tuple(str(value) for value in trace.get("tool_safety_geom_names", ()))
        expected_safety_names = tuple(
            mujoco.mj_id2name(
                exact.model,
                mujoco.mjtObj.mjOBJ_GEOM,
                geom_id,
            )
            or ""
            for geom_id in expected_safety_ids
        )
        if (
            len(expected_safety_ids) != 96
            or len(set(expected_safety_ids)) != 96
            or len(expected_contact_ids) != 8
            or len(set(expected_contact_ids)) != 8
            or len(expected_roles) != 8
            or len(set(expected_roles)) != 8
            or not set(expected_contact_ids).issubset(expected_safety_ids)
            or trace_safety_ids != expected_safety_ids
            or trace_contact_ids != expected_contact_ids
            or trace_roles != expected_roles
            or trace_safety_names != expected_safety_names
            or trace.get("tool_safety_geom_count") != 96
            or trace.get("tool_safety_distance_sampled_each_substep") is not True
            or trace.get("tool_safety_geom_order_sha256")
            != stable_tool_safety_geom_order_sha256(trace_safety_names)
        ):
            raise RuntimeError("V22 reward requires the exact ordered 96/8 union")
        contact_indices = tuple(expected_safety_ids.index(value) for value in expected_contact_ids)
        safety_only_indices = tuple(index for index in range(96) if index not in set(contact_indices))
        if len(safety_only_indices) != 88:
            raise RuntimeError("V22 reward did not resolve 88 safety-only parts")

        physics_substeps = int(trace.get("physics_substeps", -1))
        if physics_substeps < 1:
            raise RuntimeError("V22 reward physics substeps must be positive")
        block = _finite_matrix(
            trace.get("tool_safety_block_signed_distance_m"),
            name="tool_safety_block_signed_distance_m",
            shape=(physics_substeps, 96),
        )
        desk = _finite_matrix(
            trace.get("tool_safety_desk_signed_distance_m"),
            name="tool_safety_desk_signed_distance_m",
            shape=(physics_substeps, 96),
        )
        raw = _count_matrix(
            trace.get("tool_block_contact_count_by_role"),
            name="tool_block_contact_count_by_role",
            shape=(physics_substeps, 8),
        )
        invalid = _count_matrix(
            trace.get("invalid_tool_block_contact_count_by_role"),
            name="invalid_tool_block_contact_count_by_role",
            shape=(physics_substeps, 8),
        )
        all_valid = _binary_matrix(
            trace.get("all_tool_block_contacts_geometrically_valid_by_role"),
            name="all_tool_block_contacts_geometrically_valid_by_role",
            shape=(physics_substeps, 8),
        )
        if np.any(invalid > raw) or not np.array_equal(
            all_valid,
            invalid == 0,
        ):
            raise RuntimeError("V22 geometric-valid role evidence is inconsistent")

        safety_only = block[
            :,
            np.asarray(safety_only_indices, dtype=np.int64),
        ]
        flat_limiting = int(np.argmin(safety_only))
        _, limiting_local = np.unravel_index(
            flat_limiting,
            safety_only.shape,
        )
        limiting_index = int(safety_only_indices[int(limiting_local)])
        minimum_safety_only = float(np.min(safety_only))
        minimum_desk = float(np.min(desk))

        contact_minimums: list[float] = []
        unauthorized_count = 0
        authorized_count = 0
        maximum_unauthorized_depth = 0.0
        for role_index, contact_index in enumerate(contact_indices):
            distances = block[:, contact_index]
            contact_minimums.append(float(np.min(distances)))
            penetrating = distances < -self.config.penetration_tolerance_m
            authorized = (raw[:, role_index] > 0) & all_valid[:, role_index] & (invalid[:, role_index] == 0)
            unauthorized = penetrating & ~authorized
            unauthorized_count += int(np.count_nonzero(unauthorized))
            authorized_count += int(np.count_nonzero(penetrating & authorized))
            if np.any(unauthorized):
                maximum_unauthorized_depth = max(
                    maximum_unauthorized_depth,
                    float(np.max(-distances[unauthorized])),
                )

        safety_only_clearance_violation = bool(
            minimum_safety_only < self.config.safety_only_block_clearance_m
        )
        safety_only_penetration = bool(minimum_safety_only < -self.config.penetration_tolerance_m)
        unauthorized_penetration = unauthorized_count > 0
        desk_penetration = bool(minimum_desk < -self.config.penetration_tolerance_m)
        safety_only_depth = max(
            self.config.safety_only_block_clearance_m - minimum_safety_only,
            0.0,
        )
        unauthorized_depth = max(
            maximum_unauthorized_depth - self.config.penetration_tolerance_m,
            0.0,
        )
        desk_depth = max(
            -minimum_desk - self.config.penetration_tolerance_m,
            0.0,
        )
        normalized_safety_only = float(
            np.clip(
                safety_only_depth / self.config.safety_depth_scale_m,
                0.0,
                1.0,
            )
        )
        normalized_unauthorized = float(
            np.clip(
                unauthorized_depth / self.config.safety_depth_scale_m,
                0.0,
                1.0,
            )
        )
        normalized_desk = float(
            np.clip(
                desk_depth / self.config.safety_depth_scale_m,
                0.0,
                1.0,
            )
        )
        normalized_safety = max(
            normalized_safety_only,
            normalized_unauthorized,
            normalized_desk,
        )
        hard_reasons: list[str] = []
        if safety_only_clearance_violation:
            hard_reasons.append("safety_only_block_clearance_violation")
        if unauthorized_penetration:
            hard_reasons.append("unauthorized_contact_part_penetration")
        if desk_penetration:
            hard_reasons.append("full_safety_union_desk_penetration")
        return ScratchSafetyEvidenceV6Candidate(
            physics_substeps=physics_substeps,
            safety_geom_count=96,
            safety_only_geom_count=88,
            contact_safety_geom_indices=contact_indices,  # type: ignore[arg-type]
            minimum_safety_only_block_signed_distance_m=minimum_safety_only,
            limiting_safety_only_geom_index=limiting_index,
            minimum_contact_part_block_signed_distance_by_role_m=tuple(contact_minimums),  # type: ignore[arg-type]
            minimum_full_safety_desk_signed_distance_m=minimum_desk,
            safety_only_clearance_violation=safety_only_clearance_violation,
            safety_only_penetration=safety_only_penetration,
            unauthorized_contact_part_penetration=unauthorized_penetration,
            full_safety_desk_penetration=desk_penetration,
            unauthorized_contact_part_penetration_count=unauthorized_count,
            authorized_contact_part_penetration_count=authorized_count,
            normalized_safety_only_block_cost=normalized_safety_only,
            normalized_unauthorized_contact_cost=normalized_unauthorized,
            normalized_full_safety_desk_cost=normalized_desk,
            normalized_safety_cost=normalized_safety,
            safety_penalty=float(-self.config.privileged_safety_penalty_coefficient * normalized_safety),
            hard_safety_violation=bool(hard_reasons),
            hard_safety_reason="+".join(hard_reasons),
            privileged_reward_input=True,
            maximum_unauthorized_contact_penetration_depth_m=(maximum_unauthorized_depth),
        )


__all__ = [
    "STOCK_GRIPPER_REWARD_FORMAT_V22",
    "StockGripperPotentialRewardV22",
]
