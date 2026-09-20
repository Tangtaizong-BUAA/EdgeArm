"""Confidence-gated learned keypoints correct an existing learned spatial map.

Fusion thresholds are explicit task-specific heuristics, not learned SLAM or
calibrated uncertainty. Neither simulator geometry nor future frames enter.
"""

import torch


class GeometricMemoryCorrection:
    def __init__(self):
        self.previous = None
        self.previous_step = None
        self.last_accepted = None
        self.count = None
        self.last_step = -1

    def update(self, memory, measurement, step):
        if step <= self.last_step:
            raise ValueError("strictly increasing current image observations required")
        self.last_step = step
        xyz = measurement["xyz"]
        if xyz.shape != memory["xyz"].shape or xyz.shape[-2:] != (7, 3):
            raise ValueError("seven learned geometric landmarks required")
        if self.previous is None:
            self.previous = torch.zeros_like(xyz)
            self.previous_step = torch.full_like(xyz[..., 0], -1, dtype=torch.long)
            self.last_accepted = self.previous_step.clone()
            self.count = torch.zeros_like(self.previous_step)
        valid = (measurement["confidence"] >= 0.95) & measurement["valid"] & torch.isfinite(xyz).all(-1)
        consistent = (
            (self.previous_step >= 0)
            & ((step - self.previous_step) <= 16)
            & ((xyz - self.previous).norm(dim=-1) <= 0.025)
        )
        near = (xyz - memory["xyz"]).norm(dim=-1) <= 0.05
        accepted = valid & (near | consistent)
        corrected = dict(memory)
        corrected["xyz"] = torch.where(accepted[..., None], 0.15 * memory["xyz"] + 0.85 * xyz, memory["xyz"])
        corrected["variance"] = torch.where(
            accepted[..., None],
            torch.minimum(memory["variance"], torch.full_like(xyz, 0.004**2)),
            memory["variance"],
        )
        self.previous = xyz.detach().clone()
        self.previous_step = torch.where(
            valid, torch.full_like(self.previous_step, step), torch.full_like(self.previous_step, -1)
        )
        self.last_accepted = torch.where(
            accepted, torch.full_like(self.last_accepted, step), self.last_accepted
        )
        self.count += accepted.long()
        return corrected, accepted

    def completion_observed(self, selected, step):
        if self.last_accepted is None:
            return False
        block, goal = selected
        return bool(
            self.last_accepted[0, block] >= step - 16
            and self.count[0, block] >= 2
            and self.count[0, goal] >= 2
        )
