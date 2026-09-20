"""Run99 joint projection around AST-equivalent frozen Run93/94 inference.

Only cohort admission and result paths differ. All commands use the same
reported-state/static-geometry projection as completed Run101 development.
"""
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from . import run94_frozen_fast_policy as policy
from .run42.session import DomainSession
from .run85_frozen_policy import validate_cohort
from .run99_joint_feasibility import JointFeasibilityProjection, NOMINAL_WORKSPACE
from .train_staged_hybrid_contact_sac import _atomic_json


class CheckedSession(DomainSession):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        cfg = self.env.config
        if not np.array_equal([cfg.workspace_x, cfg.workspace_y, cfg.workspace_z], NOMINAL_WORKSPACE):
            raise ValueError("controller calibration must match unchanged plant workspace")


def episode(job):
    seed, checkpoint, vision, control, keypoint, output, condition, groups = job
    if condition != "camera_clearance":
        raise ValueError("only the frozen joint-feasible condition is admitted")
    validate_cohort(seed, groups)
    holder = {}
    changes = []
    original_factor = policy.factor_command

    class CalibratedSession(CheckedSession):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            holder["projection"] = JointFeasibilityProjection(self.env.model, self.env._ids["tool_site"])

    def constrained_factor(mode, proprio, selected, world, memory, anchor, learned):
        prediction = original_factor(mode, proprio, selected, world, memory, anchor, learned)
        if condition == "baseline":
            return prediction
        if prediction.shape != (1, 6):
            raise ValueError("single causal command required")
        action, audit = holder["projection"].project(
            proprio[0, :12].detach().cpu().numpy(), prediction[0].detach().cpu().numpy()
        )
        changes.append(audit)
        return torch.from_numpy(action).to(prediction.device)[None]

    directory = Path(output) / "joint_feasible" / condition
    with (
        patch.object(policy, "DomainSession", CalibratedSession),
        patch.object(policy, "factor_command", constrained_factor),
    ):
        result = policy.episode(
            (seed, checkpoint, vision, control, keypoint, str(directory), "memory_fast_keypoints", groups)
        )
    result.update(
        condition="joint_feasible", independent_acceptance=True,
        camera_clearance_constraint=True, clearance_m=.002,
        projection_calls=len(changes), projection_active=sum(r["active"] for r in changes),
        projection_only_uses_reported_joints_and_static_calibration=True,
        projection_is_learned=False, projection_counts_are_proposals_before_scripted_completion_override=True,
        joint_workspace_camera_constraint=True, workspace_inset_m=.0005,
        controller_is_learned=False, plant_unchanged=True,
    )
    folder = directory / "memory_fast_keypoints" / f"episode_{seed}"
    result["result_path"] = str(folder / "result.json")
    _atomic_json(folder / "result.json", result)
    _atomic_json(folder / "clearance_projection.json", dict(predictions_only=True, events=changes))
    return result
