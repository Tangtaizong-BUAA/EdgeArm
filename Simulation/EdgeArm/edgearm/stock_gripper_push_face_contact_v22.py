"""Versioned semantic push-face contact identity for the stock SO-101 jaws.

V9 labels only the single most distal CoACD part of each jaw as contact
capable.  That is appropriate for a fingertip-end contact, but the EdgeArm
task aligns the *broad closed-gripper face* with the near face of the block.
An execution sweep proved that several adjacent, CAD-derived convex parts are
nearly coplanar with that face; requiring every one except the two distal
parts to retain a positive 0.35 mm gap makes physical contact impossible.

V22 does not add or reshape collision geometry.  It exposes a frozen subset
of the existing 96 stock-CAD parts as conditional contact candidates.  A part
is not automatically authorized merely because it appears here: every raw
contact must still pass the existing normal, side-plane, central-band/edge,
rear-support, and force gates.  The remaining parts retain positive-clearance
authority in the online forecast guard.

The mutation is deliberately opt-in and reversible so all historical V9-V21
artifacts and tests keep their exact two-part identity.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import mujoco

from .production_env import (
    STOCK_GRIPPER_FIXED_COLLISION_GEOM,
    STOCK_GRIPPER_FIXED_SAFETY_CONVEX_PREFIX,
    STOCK_GRIPPER_MOVING_COLLISION_GEOM,
    STOCK_GRIPPER_MOVING_SAFETY_CONVEX_PREFIX,
)
from .sim2real_env_v10 import RealisticEdgeArmEnvV10


STOCK_GRIPPER_PUSH_FACE_CONTACT_FORMAT_V22 = "edgearm-stock-gripper-semantic-broad-push-face-contact-v22"
STOCK_GRIPPER_PUSH_FACE_CONTACT_EVIDENCE_V22 = (
    "same-seed-height-sweep-plus-all-part-slow-contact-manifold-diagnostic"
)

# Frozen union observed on the broad pushing face over the admitted V12 height
# band.  The original distal pieces remain present for backward physical
# coverage.  Candidate status only removes the pre-contact positive-margin
# contradiction; actual contact remains fail-closed under semantic telemetry.
STOCK_GRIPPER_PUSH_FACE_CONTACT_NAMES_V22 = (
    STOCK_GRIPPER_FIXED_COLLISION_GEOM,
    f"{STOCK_GRIPPER_FIXED_SAFETY_CONVEX_PREFIX}001",
    STOCK_GRIPPER_MOVING_COLLISION_GEOM,
    f"{STOCK_GRIPPER_MOVING_SAFETY_CONVEX_PREFIX}003",
    f"{STOCK_GRIPPER_MOVING_SAFETY_CONVEX_PREFIX}022",
    f"{STOCK_GRIPPER_MOVING_SAFETY_CONVEX_PREFIX}028",
    f"{STOCK_GRIPPER_MOVING_SAFETY_CONVEX_PREFIX}029",
    f"{STOCK_GRIPPER_MOVING_SAFETY_CONVEX_PREFIX}031",
)

_BASE_CONTACT_IDS_KEY = "v22_base_distal_tool_contact_geoms"
_BASE_CONTACT_ROLES_KEY = "v22_base_distal_tool_contact_geom_roles"


def _geom_name(env: RealisticEdgeArmEnvV10, geom_id: int) -> str:
    return (
        mujoco.mj_id2name(
            env.model,
            mujoco.mjtObj.mjOBJ_GEOM,
            int(geom_id),
        )
        or f"geom_{int(geom_id)}"
    )


def _role_for_name(name: str) -> str:
    if name == STOCK_GRIPPER_FIXED_COLLISION_GEOM:
        return "fixed_push_face_part_000"
    if name == STOCK_GRIPPER_MOVING_COLLISION_GEOM:
        return "moving_push_face_part_004"
    if name.startswith(STOCK_GRIPPER_FIXED_SAFETY_CONVEX_PREFIX):
        suffix = name.removeprefix(STOCK_GRIPPER_FIXED_SAFETY_CONVEX_PREFIX)
        return f"fixed_push_face_part_{suffix}"
    if name.startswith(STOCK_GRIPPER_MOVING_SAFETY_CONVEX_PREFIX):
        suffix = name.removeprefix(STOCK_GRIPPER_MOVING_SAFETY_CONVEX_PREFIX)
        return f"moving_push_face_part_{suffix}"
    raise RuntimeError(f"V22 push-face candidate has unexpected name: {name}")


def stock_gripper_push_face_profile_v22(
    env: RealisticEdgeArmEnvV10,
) -> dict[str, Any]:
    """Return the current V22 identity without mutating simulator state."""

    if type(env) is not RealisticEdgeArmEnvV10:
        raise TypeError("V22 push-face profile requires exact RealisticEdgeArmEnvV10")
    safety = tuple(int(value) for value in env._ids["tool_safety_geoms"])
    contacts = tuple(int(value) for value in env._ids["tool_contact_geoms"])
    roles = tuple(str(value) for value in env._ids["tool_contact_geom_roles"])
    names = tuple(_geom_name(env, value) for value in contacts)
    safety_only = tuple(value for value in safety if value not in set(contacts))
    return {
        "format": STOCK_GRIPPER_PUSH_FACE_CONTACT_FORMAT_V22,
        "evidence_source": STOCK_GRIPPER_PUSH_FACE_CONTACT_EVIDENCE_V22,
        "stock_follower_unmodified": True,
        "added_contact_tool": False,
        "collision_geometry_changed": False,
        "contact_candidate_geom_count": len(contacts),
        "contact_candidate_geom_names": list(names),
        "contact_candidate_role_names": list(roles),
        "safety_union_geom_count": len(safety),
        "safety_only_geom_count": len(safety_only),
        "actual_contact_requires_geometric_semantic_gate": True,
        "simulator_privileged": True,
        "physical_samples": 0,
        "production_admission": False,
    }


def stock_gripper_distal_reference_identity_v22(
    env: RealisticEdgeArmEnvV10,
) -> tuple[tuple[int, int], tuple[str, str]]:
    """Return the frozen two-jaw references retained by the task potential."""

    if type(env) is not RealisticEdgeArmEnvV10:
        raise TypeError("V22 distal reference identity requires exact V10")
    ids = env._ids.get(_BASE_CONTACT_IDS_KEY)
    roles = env._ids.get(_BASE_CONTACT_ROLES_KEY)
    if ids is None or roles is None:
        current_ids = tuple(int(value) for value in env._ids["tool_contact_geoms"])
        current_roles = tuple(str(value) for value in env._ids["tool_contact_geom_roles"])
        if len(current_ids) != 2 or current_roles != ("fixed_tip", "moving_tip"):
            raise RuntimeError("V22 distal reference identity is unavailable")
        return current_ids, current_roles
    selected_ids = tuple(int(value) for value in ids)
    selected_roles = tuple(str(value) for value in roles)
    if len(selected_ids) != 2 or selected_roles != ("fixed_tip", "moving_tip"):
        raise RuntimeError("V22 distal reference identity changed")
    return selected_ids, selected_roles


def configure_stock_gripper_push_face_contact_v22(
    env: RealisticEdgeArmEnvV10,
) -> dict[str, Any]:
    """Opt an exact V10 instance into the V22 contact-candidate identity."""

    if type(env) is not RealisticEdgeArmEnvV10:
        raise TypeError("V22 push-face configuration requires exact V10")
    safety = tuple(int(value) for value in env._ids["tool_safety_geoms"])
    if len(safety) != 96 or len(set(safety)) != 96:
        raise RuntimeError("V22 requires the exact 96-part stock safety union")
    safety_by_name = {_geom_name(env, value): value for value in safety}
    missing = [name for name in STOCK_GRIPPER_PUSH_FACE_CONTACT_NAMES_V22 if name not in safety_by_name]
    if missing:
        raise RuntimeError(f"V22 push-face parts are missing: {missing}")

    if _BASE_CONTACT_IDS_KEY not in env._ids:
        base_ids = tuple(int(value) for value in env._ids["tool_contact_geoms"])
        base_roles = tuple(str(value) for value in env._ids["tool_contact_geom_roles"])
        if len(base_ids) != 2 or base_roles != ("fixed_tip", "moving_tip"):
            raise RuntimeError("V22 activation requires the historical V9 96/2 identity")
        env._ids[_BASE_CONTACT_IDS_KEY] = base_ids
        env._ids[_BASE_CONTACT_ROLES_KEY] = base_roles

    selected_names = set(STOCK_GRIPPER_PUSH_FACE_CONTACT_NAMES_V22)
    contacts = tuple(value for value in safety if _geom_name(env, value) in selected_names)
    names = tuple(_geom_name(env, value) for value in contacts)
    if len(contacts) != len(STOCK_GRIPPER_PUSH_FACE_CONTACT_NAMES_V22):
        raise RuntimeError("V22 push-face candidate cardinality changed")
    if set(names) != selected_names:
        raise RuntimeError("V22 push-face candidate identity changed")
    roles = tuple(_role_for_name(name) for name in names)
    if len(set(roles)) != len(roles):
        raise RuntimeError("V22 push-face roles must be unique")

    env._ids["tool_contact_geoms"] = contacts
    env._ids["tool_contact_geom_roles"] = roles
    env._ids["tool_contact_geometry_mode"] = "semantic_broad_push_face_candidate_union_v22"
    env._ids["tool_contact_identity_format"] = STOCK_GRIPPER_PUSH_FACE_CONTACT_FORMAT_V22
    profile = stock_gripper_push_face_profile_v22(env)
    if profile["contact_candidate_geom_count"] != 8:
        raise RuntimeError("V22 push-face profile must expose eight candidates")
    if profile["safety_only_geom_count"] != 88:
        raise RuntimeError("V22 push-face profile must retain 88 safety-only parts")
    env.episode_domain["stock_gripper_push_face_contact_v22"] = deepcopy(profile)
    return profile


def restore_stock_gripper_distal_contact_v22(
    env: RealisticEdgeArmEnvV10,
) -> None:
    """Restore the historical two-distal-part identity before a V12 reset."""

    if type(env) is not RealisticEdgeArmEnvV10:
        raise TypeError("V22 distal-contact restore requires exact V10")
    base_ids = env._ids.get(_BASE_CONTACT_IDS_KEY)
    base_roles = env._ids.get(_BASE_CONTACT_ROLES_KEY)
    if base_ids is None or base_roles is None:
        contacts = tuple(int(value) for value in env._ids["tool_contact_geoms"])
        roles = tuple(str(value) for value in env._ids["tool_contact_geom_roles"])
        if len(contacts) != 2 or roles != ("fixed_tip", "moving_tip"):
            raise RuntimeError("V22 cannot infer the historical distal identity")
        return
    env._ids["tool_contact_geoms"] = tuple(int(value) for value in base_ids)
    env._ids["tool_contact_geom_roles"] = tuple(str(value) for value in base_roles)
    env._ids.pop("tool_contact_geometry_mode", None)
    env._ids.pop("tool_contact_identity_format", None)


__all__ = [
    "STOCK_GRIPPER_PUSH_FACE_CONTACT_EVIDENCE_V22",
    "STOCK_GRIPPER_PUSH_FACE_CONTACT_FORMAT_V22",
    "STOCK_GRIPPER_PUSH_FACE_CONTACT_NAMES_V22",
    "configure_stock_gripper_push_face_contact_v22",
    "restore_stock_gripper_distal_contact_v22",
    "stock_gripper_distal_reference_identity_v22",
    "stock_gripper_push_face_profile_v22",
]
