"""Simulation-only task-direction ablation. Original telemetry remains intact."""

from dataclasses import replace
from unittest.mock import patch
from . import contact_telemetry_v1 as telemetry


class ContactAudit:
    def __init__(self, profile="strict"):
        if profile not in ("strict", "direction_tolerant_v1", "task_goal_v1"):
            raise ValueError("unknown contact profile")
        self.profile = profile
        self.physical_geometry_invalid = False

    def clear(self):
        self.physical_geometry_invalid = False

    def __enter__(self):
        original = telemetry.push_side_contact_metrics

        def capture(model, data, geometry, intended_push_direction_xy, thresholds=None):
            strict = original(model, data, geometry, intended_push_direction_xy, thresholds)
            if self.profile == "task_goal_v1":
                # This profile retains the original diagnostic ledger but does
                # not use object-contact geometry as a terminal condition.
                return strict
            if self.profile == "strict":
                self.physical_geometry_invalid |= bool(strict["invalid_tool_block_contact_count"])
                return strict
            limits = thresholds or telemetry.PushSideContactThresholds()
            # Remove only goal-direction preferences, not side-plane/height or
            # horizontal-contact tests. Reverse-facing contact still fails at 0.
            if self.profile != "strict":
                limits = replace(
                    limits,
                    minimum_contact_normal_push_alignment=0.0,
                    minimum_tool_face_push_alignment=0.0,
                    minimum_rear_support_ratio=0.0,
                    minimum_edge_normal_push_alignment=0.0,
                    minimum_edge_tool_face_push_alignment=0.0,
                    minimum_edge_rear_support_ratio=0.0,
                )
            physical = original(model, data, geometry, intended_push_direction_xy, limits)
            self.physical_geometry_invalid |= bool(physical["invalid_tool_block_contact_count"])
            return strict

        self.context = patch.object(telemetry, "push_side_contact_metrics", capture)
        self.context.__enter__()
        return self

    def __exit__(self, *args):
        return self.context.__exit__(*args)


def hard_failure(trace, multichoice_failed, geometry_invalid):
    required = ("forbidden_tool_desk_penetration_any", "forbidden_non_tool_robot_desk_penetration_any")
    if any(k not in trace for k in required):
        raise ValueError("missing physical safety telemetry")
    return bool(multichoice_failed or geometry_invalid or any(trace[k] for k in required))


def profile_failure(profile, trace, multichoice_failed, geometry_invalid):
    if profile == "task_goal_v1":
        return hard_failure(trace, False, False)
    return hard_failure(trace, multichoice_failed, geometry_invalid)
