"""Image-localized landmarks with calibrated rays and declared height priors.

Deployment uses wrist RGB, estimated camera extrinsics/intrinsics only. Object
heights are nominal task priors, not simulator measurements or learned depth.
Training labels are projected only in the loss, never passed to forward().
"""

import math

import torch
from torch import nn
from torch.nn import functional as F


def project_labels(xyz, pose, K):
    camera = torch.einsum("bnj,bjk->bnk", xyz - pose[:, None, :3], pose[:, 3:].reshape(-1, 3, 3))
    depth = -camera[..., 2]
    safe = depth.clamp_min(0.001)
    uv = torch.stack(
        (
            K[:, None, 0, 0] * camera[..., 0] / safe + K[:, None, 0, 2],
            K[:, None, 1, 2] - K[:, None, 1, 1] * camera[..., 1] / safe,
        ),
        -1,
    )
    valid = (depth > 0.015) & (uv[..., 0] >= 1) & (uv[..., 0] < 159) & (uv[..., 1] >= 1) & (uv[..., 1] < 119)
    return uv, valid


def backproject(uv, pose, K, heights):
    rays = torch.stack(
        (
            (uv[..., 0] - K[:, None, 0, 2]) / K[:, None, 0, 0],
            -(uv[..., 1] - K[:, None, 1, 2]) / K[:, None, 1, 1],
            -torch.ones_like(uv[..., 0]),
        ),
        -1,
    )
    direction = torch.einsum("bij,bnj->bni", pose[:, 3:].reshape(-1, 3, 3), rays)
    denominator = direction[..., 2]
    safe = torch.where(denominator.abs() > 0.001, denominator, torch.ones_like(denominator) * 0.001)
    distance = (heights[None] - pose[:, None, 2]) / safe
    xyz = pose[:, None, :3] + distance[..., None] * direction
    valid = (denominator.abs() > 0.001) & (distance > 0.015) & (distance < 2) & torch.isfinite(xyz).all(-1)
    return xyz, valid


class WristKeypoints(nn.Module):
    kind = "image_keypoints_calibrated_ray_nominal_height_v88"

    def __init__(self):
        super().__init__()
        self.early = nn.Sequential(
            nn.Conv2d(3, 32, 5, 2, 2),
            nn.GroupNorm(4, 32),
            nn.SiLU(),
            nn.Conv2d(32, 64, 3, 2, 1),
            nn.GroupNorm(8, 64),
            nn.SiLU(),
        )
        self.context = nn.Sequential(
            nn.Conv2d(64, 96, 3, 2, 1),
            nn.GroupNorm(8, 96),
            nn.SiLU(),
            nn.Conv2d(96, 96, 3, 1, 1),
            nn.GroupNorm(8, 96),
            nn.SiLU(),
        )
        self.heatmap = nn.Sequential(
            nn.Conv2d(160, 64, 3, 1, 1), nn.GroupNorm(8, 64), nn.SiLU(), nn.Conv2d(64, 7, 1)
        )
        self.absent = nn.Linear(96, 7)
        self.register_buffer("heights", torch.tensor([0.051] * 4 + [0.026] * 3))

    def forward(self, rgb, pose, K):
        if (
            rgb.ndim != 4
            or rgb.shape[1:] != (120, 160, 3)
            or pose.shape[1:] != (12,)
            or K.shape[1:] != (3, 3)
        ):
            raise ValueError("single current wrist image plus estimated calibration required")
        early = self.early(rgb.permute(0, 3, 1, 2).float() / 255)
        context = self.context(early)
        up = F.interpolate(context, size=early.shape[-2:], mode="bilinear", align_corners=False)
        logits = self.heatmap(torch.cat((early, up), 1)).float().flatten(2)
        absent = self.absent(context.mean((-1, -2))).float() + math.log(logits.shape[-1])
        logp = torch.cat((logits, absent[..., None]), -1).log_softmax(-1)
        h, w = early.shape[-2:]
        yy, xx = torch.meshgrid(
            torch.arange(h, device=rgb.device), torch.arange(w, device=rgb.device), indexing="ij"
        )
        grid = torch.stack(
            ((xx.flatten().float() + 0.5) * 160 / w - 0.5, (yy.flatten().float() + 0.5) * 120 / h - 0.5), -1
        )
        uv = logits.softmax(-1) @ grid
        xyz, valid = backproject(uv, pose.float(), K.float(), self.heights)
        return dict(logp=logp, uv=uv, xyz=xyz, valid=valid, confidence=1 - logp[..., -1].exp(), grid=grid)


def keypoint_loss(output, xyz, present, visible, pose, K):
    target, inside = project_labels(xyz, pose, K)
    known = present.bool() & visible.bool() & inside
    distance = (output["grid"][None, None] - target[:, :, None]).square().sum(-1)
    heat = (-distance / (2 * 4.0**2)).softmax(-1)
    target_heat = heat * known[..., None]
    distribution = torch.cat((target_heat, (~known)[..., None].float()), -1)
    ce = -(distribution * output["logp"]).sum(-1)
    # Both observed and absent labels matter, independent of their prevalence.
    loss_pos = (ce * known).sum() / known.sum().clamp_min(1)
    loss_neg = (ce * (~known)).sum() / (~known).sum().clamp_min(1)
    coordinate = F.smooth_l1_loss(output["uv"] / 8, target / 8, reduction="none").mean(-1)
    pixel = (coordinate * known).sum() / known.sum().clamp_min(1)
    loss = loss_pos + loss_neg + 0.5 * pixel
    return loss, dict(heatmap=float((loss_pos + loss_neg).detach()), pixel=float(pixel.detach()))
