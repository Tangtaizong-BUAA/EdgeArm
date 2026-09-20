"""Retain image-measured static regions while dynamic spatial memory evolves.

The median and locking rule are declared causal fusion heuristics. Landmarks
come only from the learned keypoint model; static goals are task priors.
"""

import torch

from .run89_geometric_memory import GeometricMemoryCorrection


class StaticSurveyMemory(GeometricMemoryCorrection):
    def __init__(self):
        super().__init__()
        self.survey = []
        self.survey_valid = []
        self.static = None
        self.locked = None
        self.finalized = False

    def update(self, memory, measurement, step):
        out, accepted = super().update(memory, measurement, step)
        if self.static is None:
            self.static = torch.zeros_like(measurement["xyz"][:, 4:])
            self.locked = torch.zeros_like(measurement["valid"][:, 4:])
        if step <= 220:
            self.survey.append(measurement["xyz"][:, 4:].clone())
            self.survey_valid.append((measurement["confidence"][:, 4:] >= 0.95) & measurement["valid"][:, 4:])
        if step >= 220 and not self.finalized:
            points = torch.stack(self.survey)
            valid = torch.stack(self.survey_valid)
            median = (
                torch.where(valid[..., None], points, torch.full_like(points, float("nan")))
                .cpu()
                .nanmedian(0)
                .values.to(points.device)
            )
            supported = (valid.sum(0) >= 3) & torch.isfinite(median).all(-1)
            newly = supported & ~self.locked
            self.static = torch.where(newly[..., None], median, self.static)
            self.locked |= newly
            self.finalized = True
        if step >= 220:
            xyz = out["xyz"].clone()
            xyz[:, 4:] = torch.where(self.locked[..., None], self.static, xyz[:, 4:])
            out = dict(out, xyz=xyz)
        return out, accepted
