"""Progressive broad-face alignment for exact-Home acquisition.

V597 used a hard 12 cm switch: it preserved the current face outside that
radius and requested full task-axis alignment inside it.  Same-seed V661
evidence showed the guarded local IK repeatedly stalling just outside this
boundary.  V662 begins a smooth alignment at 18 cm and reaches full alignment
by 10 cm, while leaving policy XYZ authority, action bounds, safety guards,
robot geometry, and the exact Home reset unchanged.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from .task_independent_home_reset_v597 import (
    HomeAcquisitionOrientationScheduleV597,
    StockGripperHomeTaskFrameAdapterV597,
)


PROGRESSIVE_HOME_ALIGNMENT_FORMAT_V662 = (
    "edgearm-v662-progressive-home-task-axis-alignment-v1"
)


def progressive_home_alignment_schedule_v662(
) -> HomeAcquisitionOrientationScheduleV597:
    schedule = HomeAcquisitionOrientationScheduleV597(
        mode="progressive",
        alignment_maximum_height_m=0.120,
        alignment_start_block_xy_distance_m=0.180,
        full_alignment_block_xy_distance_m=0.100,
        free_space_normal_scale=0.25,
        alignment_normal_scale=0.35,
        release_precontact_distance_m=0.040,
        precontact_standoff_m=0.055,
        precontact_tool_height_m=0.055,
    )
    schedule.validate()
    return schedule


def install_progressive_home_alignment_v662(
    adapter: StockGripperHomeTaskFrameAdapterV597,
) -> dict[str, Any]:
    if type(adapter) is not StockGripperHomeTaskFrameAdapterV597:
        raise TypeError("V662 requires the exact V597 adapter")
    schedule = progressive_home_alignment_schedule_v662()
    adapter.orientation_schedule_v597 = schedule
    audit = {
        "format": PROGRESSIVE_HOME_ALIGNMENT_FORMAT_V662,
        "orientation_schedule": asdict(schedule),
        "policy_action_dimension": 3,
        "policy_xyz_authority_unchanged": True,
        "automatic_orientation_constraint_retained": True,
        "hard_distance_switch_removed": True,
        "premature_contact_tracker_handoff_removed": True,
        "research_basis": [
            {
                "title": "Redundancy-aware Action Spaces for Robot Learning",
                "arxiv": "2406.04144",
                "use": (
                    "action validity and discriminability motivate removing "
                    "a locally invalid hard switch"
                ),
                "implementation_copied": False,
            },
            {
                "title": (
                    "Analytically Informed Inverse Kinematics Solution at "
                    "Singularities"
                ),
                "arxiv": "2412.20409",
                "use": (
                    "singularity analysis motivates avoiding discontinuous "
                    "orientation demands before changing solvers"
                ),
                "implementation_copied": False,
            },
        ],
        "expert_action_used": False,
        "waypoint_or_path_used": False,
        "behavior_cloning_steps": 0,
        "act_training_started": False,
        "wrist_multimodal_export_started": False,
        "production_admission": False,
    }
    return audit


__all__ = [
    "PROGRESSIVE_HOME_ALIGNMENT_FORMAT_V662",
    "install_progressive_home_alignment_v662",
    "progressive_home_alignment_schedule_v662",
]
