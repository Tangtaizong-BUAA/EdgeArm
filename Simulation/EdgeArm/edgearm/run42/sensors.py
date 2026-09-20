"""Causal RGB corruption and uncertainty on RGB-estimated depth, not truth."""

import math

import numpy as np
from PIL import Image, ImageFilter
import torch
from torch import nn


def corrupt_rgb(rgb, parameters, time_s):
    p = parameters
    if p['stage'] < 2:
        return np.asarray(rgb, np.uint8).copy()
    frame = round(time_s * 30)
    rng = np.random.default_rng(np.random.SeedSequence([p['seed'], frame, 4201]))
    image = Image.fromarray(rgb)
    if p['blur_radius_px']:
        image = image.filter(ImageFilter.GaussianBlur(p['blur_radius_px']))
    x = np.asarray(image, np.float32) / 255
    ev = p['exposure_ev'] + p['exposure_drift_ev'] * math.sin(.7 * time_s + p['seed'] % 101)
    x = x * 2 ** ev * np.asarray(p['white_balance'])
    sigma = np.sqrt((p['noise_std_255'] / 255) ** 2 + p['shot_noise'] ** 2 * np.maximum(x, 0))
    x += rng.normal(size=x.shape) * sigma
    return np.rint(np.clip(x, 0, 1) * 255).astype(np.uint8)


def hashed_uniform(indices):
    # Integer hash avoids redrawing a historical frame when it is observed
    # again or moved to another inference batch. No simulator labels involved.
    x = indices.to(torch.int64)
    x = (x ^ (x >> 16)) * 0x45D9F3B
    x = (x ^ (x >> 16)) * 0x45D9F3B
    x = x ^ (x >> 16)
    return ((x & 0xFFFFFF).float() + .5) / 0x1000000


class EstimatedDepthPerturbation(nn.Module):
    def __init__(self, depth_student):
        super().__init__()
        self.student = depth_student
        self.context = None

    def configure(self, contexts, inputs):
        if all(c['domain']['stage'] < 2 for c in contexts):
            self.context = None
            return inputs
        device = inputs['robot_state'].device
        times = inputs['camera_device_time_delta_s_window']
        current = torch.tensor([c['time_s'] for c in contexts], device=device)[:, None]
        frames = torch.round((times + current) * 30).to(torch.int64)
        seeds = torch.tensor([c['domain']['seed'] for c in contexts], device=device, dtype=torch.int64)[:, None]
        keys = frames * 104729 + seeds * 15485863
        values = {}
        for name in ('depth_scale_error', 'depth_bias_m', 'depth_noise_relative', 'uncertainty_scale'):
            values[name] = torch.tensor([c['domain'][name] for c in contexts], device=device)[:, None].expand_as(times).reshape(-1, 1, 1)
        active = torch.tensor([c['domain']['stage'] >= 2 for c in contexts], device=device)[:, None].expand_as(times)
        self.context = dict(keys=keys.reshape(-1, 1, 1), active=active.reshape(-1, 1, 1), **values)
        drop = torch.tensor([c['domain']['depth_dropout'] for c in contexts], device=device)[:, None]
        # Remove estimated geometry on missing frames; never claim calibrated
        # exact RGB-D just because the simulator has an exact camera matrix.
        result = dict(inputs)
        result['geometry_available_mask'] = inputs['geometry_available_mask'] & (hashed_uniform(keys) >= drop)
        return result

    def forward(self, rgb):
        depth, uncertainty = self.student(rgb)
        if self.context is None:
            return depth, uncertainty
        c = self.context
        if len(c['keys']) != len(depth):
            raise ValueError('depth uncertainty context/history alignment mismatch')
        grid = torch.arange(depth.shape[-2] * depth.shape[-1], device=depth.device).reshape(1, *depth.shape[-2:])
        u1 = hashed_uniform(c['keys'] + grid * 97).clamp_min(1e-7)
        u2 = hashed_uniform(c['keys'] + grid * 193 + 8191)
        normal = torch.sqrt(-2 * torch.log(u1)) * torch.cos(2 * math.pi * u2)
        sigma = depth * c['depth_noise_relative']
        estimate = (depth * (1 + c['depth_scale_error']) + c['depth_bias_m'] + sigma * normal).clamp(.01, 2.)
        error_scale = (depth * c['depth_scale_error']).square() + c['depth_bias_m'].square() + sigma.square()
        noisy_uncertainty = torch.sqrt(uncertainty.square() + error_scale) * c['uncertainty_scale']
        return torch.where(c['active'], estimate, depth), torch.where(c['active'], noisy_uncertainty, uncertainty)
