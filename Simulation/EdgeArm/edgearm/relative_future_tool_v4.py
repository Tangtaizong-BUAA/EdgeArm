"""Bounded relative future-motion auxiliary head. No future inputs; no action override."""

import torch
from torch import nn


class RelativeFutureToolV4(nn.Module):
    def __init__(self, feature_dim=256, max_displacement_m=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(feature_dim)
        self.limit = max_displacement_m
        self.net = nn.Sequential(nn.Linear(feature_dim + 7, 128), nn.GELU(), nn.Linear(128, 3))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, decoded, current_xyz, current_twist):
        b, k, _ = decoded.shape
        horizon = (
            torch.arange(1, k + 1, device=decoded.device, dtype=decoded.dtype)[None, :, None].expand(b, k, 1)
            / k
        )
        x = torch.cat((self.norm(decoded), current_twist[:, None, :].expand(b, k, 6), horizon), -1)
        return current_xyz[:, None, :] + self.limit * torch.tanh(self.net(x))
