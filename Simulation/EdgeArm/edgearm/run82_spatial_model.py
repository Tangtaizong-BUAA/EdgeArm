"""Trainable action-conditioned sparse object-space memory, never oracle-fed.

Seven semantic state tokens predict 56 surface points over time. This is a
task-specific sparse 3D+time representation, not general dense reconstruction.
Only the loss function accepts privileged labels; step() has an explicit
deployment-observation interface. Previous executed commands are contained in
the 114-D proprioceptive history, never the current teacher action.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F

from .run63_control_probe import TinyTarget
from .run67_visual_state import VisualState
from .run68_visual_control import estimated_inputs


def detach_state(state):
    return {k: v.detach() for k, v in state.items()}


class SparseSpatialPolicy(nn.Module):
    kind = 'action_conditioned_sparse_spatial_memory_v82'

    def __init__(self):
        super().__init__()
        self.visual = VisualState(recent_block_seconds=.7)
        self.control = TinyTarget(118, 512)
        self.context = nn.Sequential(nn.Linear(114, 128), nn.SiLU(), nn.Linear(128, 128))
        self.motion = nn.Sequential(nn.Linear(128*2+3, 128), nn.SiLU(), nn.Linear(128, 3))
        self.key = nn.Linear(128, 64)
        self.query = nn.Linear(128, 64)
        self.spatial = nn.Sequential(nn.Linear(3, 64), nn.SiLU(), nn.Linear(64, 128))
        self.update = nn.GRUCell(128*3, 128)
        self.visibility = nn.Sequential(nn.Linear(256, 64), nn.SiLU(), nn.Linear(64, 1))
        self.write_gate = nn.Sequential(nn.Linear(256, 64), nn.SiLU(), nn.Linear(64, 1))
        self.geometry = nn.Sequential(nn.Linear(128, 128), nn.SiLU(), nn.Linear(128, 3))
        self.noise = nn.Linear(128, 3)
        self.surface = nn.Sequential(nn.Linear(128, 128), nn.SiLU(), nn.Linear(128, 8*3))
        self.action_memory = nn.Sequential(nn.Linear(128*2+12, 256), nn.SiLU(), nn.Linear(256, 6))
        self.register_buffer('heights', torch.tensor([.051]*4+[.026]*3))
        self.register_buffer('dynamic', torch.tensor([1.]*4+[0.]*3)[None, :, None])
        # Start with the existing image estimate, then learn motion/memory.
        nn.init.zeros_(self.motion[-1].weight); nn.init.zeros_(self.motion[-1].bias)
        nn.init.zeros_(self.geometry[-1].weight); nn.init.zeros_(self.geometry[-1].bias)
        nn.init.zeros_(self.action_memory[-1].weight); nn.init.zeros_(self.action_memory[-1].bias)
        nn.init.zeros_(self.noise.weight); nn.init.constant_(self.noise.bias, -1.)
        nn.init.constant_(self.visibility[-1].bias, 1.)
        nn.init.constant_(self.write_gate[-1].bias, 2.)

    def initialize_from(self, vision, control):
        self.visual.load_state_dict(vision['model'])
        self.control.load_state_dict(control['model'])

    def initial_state(self, batch, device):
        xyz = torch.zeros(batch, 7, 3, device=device)
        xyz[..., 0] = .3
        xyz[..., 2] = self.heights
        return dict(hidden=torch.zeros(batch, 7, 128, device=device), xyz=xyz,
                    variance=torch.full_like(xyz, .08**2),
                    seen=torch.zeros(batch, 7, 1, device=device))

    def observation(self, rgb, pose, K, age):
        return self.visual(rgb, pose, K, age, return_features=True)

    def step(self, rgb, pose, K, age, proprio, selected, dt, state=None,
             *, encoded=None, erase_memory=False, erase_action=False):
        if proprio.ndim != 2 or proprio.shape[-1] != 114:
            raise ValueError('114-D deployable proprioceptive history required')
        batch = len(proprio)
        if selected.shape != (batch, 2) or dt.shape != (batch,):
            raise ValueError('named selection and elapsed time required')
        if torch.any(dt < 0) or torch.any(dt > .5):
            raise ValueError('bounded causal inter-observation time required')
        if state is None or erase_memory:
            state = self.initial_state(batch, proprio.device)
        observed_xy, feature = self.observation(rgb, pose, K, age) if encoded is None else encoded
        # Normalization is copied from the old control model, excluding all XY.
        mean = torch.cat((self.control.xmean[:108], self.control.xmean[112:]))
        scale = torch.cat((self.control.xscale[:108], self.control.xscale[112:])).clamp_min(.02)
        context_input = ((proprio-mean)/scale).clamp(-20, 20)
        if erase_action:
            context_input = context_input.clone()
            context_input[:, 78:108] = 0.  # four completed actions + applied target
        context = self.context(context_input)[:, None].expand(-1, 7, -1)
        motion = .025*torch.tanh(self.motion(torch.cat((state['hidden'], context, state['xyz']), -1)))
        prior = state['xyz'] + self.dynamic*motion*(dt[:, None, None]/(4/30))
        process = (.003+.012*dt[:, None, None])*self.dynamic + .00005
        prior_var = state['variance'] + process.square()
        # Sparse exchange: each query reads three of the seven spatial tokens.
        keys = self.key(state['hidden']+self.spatial(prior))
        queries = self.query(feature+context)
        scores = queries@keys.transpose(-1, -2)/math.sqrt(64)
        distance = torch.cdist(prior, prior).square()
        scores = scores-distance/.15**2
        values, indices = scores.topk(3, dim=-1)
        neighbors = state['hidden'][:, None].expand(-1, 7, -1, -1).gather(
            2, indices[..., None].expand(-1, -1, -1, 128))
        sparse_context = (values.softmax(-1)[..., None]*neighbors).sum(2)
        candidate = self.update(torch.cat((feature, context, sparse_context), -1).reshape(-1, 384),
                                state['hidden'].reshape(-1, 128)).reshape(batch, 7, 128)
        pair = torch.cat((feature, state['hidden']), -1)
        visible_logit = self.visibility(pair)
        visible = visible_logit.sigmoid()
        gate = visible*self.write_gate(pair).sigmoid()
        # No teacher forcing: hidden and spatial state always come from model.
        hidden = gate*candidate+(1-gate)*state['hidden']
        measured = torch.cat((observed_xy, self.heights[None, :, None].expand(batch, -1, -1)), -1)
        measured = measured + .025*torch.tanh(self.geometry(candidate))
        obs_var = (.002+.04*torch.sigmoid(self.noise(candidate))).square()
        gain = gate*prior_var/(prior_var+obs_var)
        # At the first observation there is no past map to trust.
        gain = torch.where(state['seen'] > .01, gain, torch.ones_like(gain))
        xyz = prior+gain*(measured-prior)
        variance = (1-gain).square()*prior_var+gain.square()*obs_var
        seen = torch.maximum(state['seen'], visible)
        points = xyz[:, :, None]+.07*torch.tanh(self.surface(hidden).reshape(batch, 7, 8, 3))
        new_state = dict(hidden=hidden, xyz=xyz, variance=variance, seen=seen)
        action = self.command(proprio, selected, new_state)
        result = dict(action=action, xyz=xyz, points=points, prior=prior,
                      variance=variance, visible_logit=visible_logit.squeeze(-1),
                      write_gate=gate.squeeze(-1), neighbors=indices)
        return result, new_state

    def command(self, proprio, selected, state):
        """30 Hz feedback control can read the latest sparse map between images."""
        with torch.autocast(device_type=proprio.device.type, enabled=False):
            hidden, xyz, variance = (state[k].float() for k in ('hidden', 'xyz', 'variance'))
            x = estimated_inputs(proprio.float(), xyz[..., :2], selected)
            selected_hidden = hidden.gather(1, selected[..., None].expand(-1, -1, 128)).flatten(1)
            selected_xyz = xyz.gather(1, selected[..., None].expand(-1, -1, 3)).flatten(1)
            selected_var = variance.gather(1, selected[..., None].expand(-1, -1, 3)).sqrt().flatten(1)
            action = self.control(x)+.15*torch.tanh(self.action_memory(
                torch.cat((selected_hidden, selected_xyz, selected_var), -1)))
        return action


def spatial_losses(output, labels, action_scale, action_weight=1.):
    """Only here do simulator-derived supervision labels enter computation."""
    present = labels['present'].float()
    norm = present.sum().clamp_min(1)
    xyz_error = (output['xyz']-labels['xyz'])/.02
    location = (F.smooth_l1_loss(xyz_error, torch.zeros_like(xyz_error), reduction='none').mean(-1)*present).sum()/norm
    prior_error = (output['prior']-labels['xyz'])/.02
    prior_mask = labels.get('prior_valid', present).float()
    prior = (F.smooth_l1_loss(prior_error, torch.zeros_like(prior_error), reduction='none').mean(-1)*prior_mask).sum()/prior_mask.sum().clamp_min(1)
    # Unordered point-set reconstruction avoids assigning an arbitrary cube corner.
    distance = torch.cdist(output['points'].flatten(0, 1).float(), labels['points'].flatten(0, 1).float())/.02
    chamfer = (distance.min(-1).values.mean(-1)+distance.min(-2).values.mean(-1)).reshape_as(present)
    reconstruction = (chamfer*present).sum()/norm
    valid_frames = labels.get('frame_valid', torch.ones_like(present[:, 0])).float()
    visibility_rows = F.binary_cross_entropy_with_logits(output['visible_logit'], labels['visible'].float(), reduction='none').mean(-1)
    visibility = (visibility_rows*valid_frames).sum()/valid_frames.sum().clamp_min(1)
    variance = output['variance'].clamp_min(.002**2)
    calibration = (((output['xyz']-labels['xyz']).square().detach()/variance + torch.log(variance/.02**2)).mean(-1)*present).sum()/norm
    action_rows = labels['action_valid'].float()
    error = F.smooth_l1_loss(output['action']/action_scale, labels['command']/action_scale, reduction='none').mean(-1)
    action = (error*action_rows).sum()/action_rows.sum().clamp_min(1)
    total = location+.25*prior+.15*reconstruction+.15*visibility+.01*calibration+action_weight*action
    return total, dict(location=location, motion_prior=prior, reconstruction=reconstruction,
                       visibility=visibility, uncertainty=calibration, action=action,
                       position_mm=(xyz_error.norm(dim=-1)*present).sum()/norm*20)
