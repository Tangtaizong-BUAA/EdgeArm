"""Prior-first incremental ACT objective with bounded old-behavior anchoring."""
from copy import deepcopy
import math
import torch
from torch import nn


def configure_trainable(model, rgb_unfreeze):
    if rgb_unfreeze not in ('none','last_conv','all'):
        raise ValueError('unknown RGB adaptation stage')
    model.requires_grad_(True)
    model.policy.depth_student.requires_grad_(False)
    model.policy.next_tool_head.requires_grad_(False)
    model.policy.core.rgbd_encoder.requires_grad_(False)
    if rgb_unfreeze=='last_conv':
        model.policy.core.rgbd_encoder[4].requires_grad_(True)
    elif rgb_unfreeze=='all':
        model.policy.core.rgbd_encoder.requires_grad_(True)


def per_window_l1(prediction, target, mask):
    target = torch.where(mask[...,None],target,0.)
    err = (prediction.float()-target.float()).abs().mean(-1)
    return (err*mask).sum(-1)/mask.sum(-1).clamp_min(1), err[:,0]


class Objective(nn.Module):
    def __init__(self, model, profile):
        super().__init__()
        self.model, self.profile = model, profile
        self.reference = deepcopy(model).eval().requires_grad_(False)

    def train(self, mode=True):
        super().train(mode)
        self.reference.eval()
        self.model.policy.depth_student.eval()
        return self

    def forward(self, batch, update):
        p = self.profile
        out = self.model(batch['inputs'],return_aux=True,
                         teacher_actions=batch['target'],teacher_mask=batch['mask'])
        chunk, first = per_window_l1(out['action'],batch['target'],batch['mask'])
        posterior, pfirst = per_window_l1(out['posterior_action'],batch['target'],batch['mask'])
        prior_loss = (chunk+p['prior_first_weight']*first).mean()
        posterior_loss = (posterior+p['prior_first_weight']*pfirst).mean()
        fm = batch['future_tool_mask']
        future_error = (out['next_tool_xyz'].float()-batch['future_tool_xyz'].float()).abs().mean(-1)/.1
        future = (future_error*fm).sum()/fm.sum().clamp_min(1)
        with torch.no_grad():
            reference = self.reference(batch['inputs'])
            _, reference_error = per_window_l1(reference,batch['target'],batch['mask'])
            eligible = (batch['cohort']<2) & (reference_error<=p['old_anchor_max_teacher_mae'])
        anchor_error = (out['action'].float()-reference.float()).abs().mean(-1)
        valid = batch['mask'] & eligible[:,None]
        # Mean over ALL old/new windows bounds the term as eligible count falls.
        anchor = ((anchor_error*valid).sum(-1)/batch['mask'].sum(-1).clamp_min(1)).mean()
        kl_scale = p['kl_weight']*min((update+1)/p['kl_warmup_updates'],1.)
        loss = (prior_loss+p['posterior_weight']*posterior_loss+kl_scale*out['kl'].float()
                +p['future_tool_weight']*future+p['old_anchor_weight']*anchor)
        # Only loss stays attached. DDP unused-parameter discovery must not
        # mistake non-loss routing/debug outputs for differentiable objectives.
        metrics = torch.stack([loss.detach(),prior_loss.detach(),posterior_loss.detach(),
            out['kl'].float().detach(),future.detach(),anchor.detach(),first.mean().detach(),
            eligible.float().mean().detach()])
        return loss, metrics


def parameter_groups(model, p):
    groups = {}
    for name,param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith('policy.core.rgbd_encoder.'):
            scale=p['rgb_lr_scale']
        elif name.startswith(('posterior.','latent_projection.','future_head.','policy.core.action_')):
            scale=1.
        else:
            scale=p['representation_lr_scale']
        decay=p['weight_decay'] if param.ndim>=2 and not name.endswith('bias') else 0.
        groups.setdefault((scale,decay),[]).append(param)
    return [dict(params=params,lr=p['learning_rate']*scale,initial_lr=p['learning_rate']*scale,
                 weight_decay=decay) for (scale,decay),params in groups.items()]


def lr_multiplier(update,total,warmup):
    if update<warmup:
        return (update+1)/max(warmup,1)
    fraction=min(1.,(update-warmup)/max(total-warmup,1))
    return .1+.9*.5*(1+math.cos(math.pi*fraction))
