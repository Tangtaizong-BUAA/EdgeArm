"""Multimodal ACT-style behavior distillation with a training-only action posterior.

Extends the existing causal observation encoder. The prior is z=0. Latent
conditioning shifts action queries to preserve the old warm-start exactly;
this is an ACT-inspired variant, not an identical reproduction of official ACT.
"""

import torch
from torch import nn
from .temporal_spatial_policy_v3 import TemporalSpatialPolicyV3
from .relative_future_tool_v4 import RelativeFutureToolV4


class ActionPosteriorV5(nn.Module):
    def __init__(self, dim, heads, chunk, latent_dim=32):
        super().__init__()
        self.chunk = chunk
        self.latent_dim = latent_dim
        self.cls = nn.Parameter(torch.zeros(1, 1, dim))
        self.positions = nn.Parameter(torch.randn(1, chunk + 2, dim) * 0.01)
        self.state = nn.Linear(12, dim)
        self.actions = nn.Linear(6, dim)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(dim, heads, dim * 2, dropout=0, batch_first=True, activation="gelu"),
            2,
            enable_nested_tensor=False,
        )
        self.output = nn.Linear(dim, latent_dim * 2)

    def forward(self, state, actions, mask):
        b = len(state)
        if (
            actions.shape != (b, self.chunk, 6)
            or mask.shape != (b, self.chunk)
            or mask.dtype != torch.bool
            or not mask.any(1).all()
        ):
            raise ValueError("invalid training action posterior labels")
        if not torch.isfinite(actions[mask]).all() or (actions[mask].abs() > 1.000001).any():
            raise ValueError("invalid normalized teacher action")
        encoded = (
            torch.cat(
                (
                    self.cls.expand(b, -1, -1),
                    self.state(state)[:, None, :],
                    self.actions(torch.where(mask[:, :, None], actions, 0.0)),
                ),
                1,
            )
            + self.positions
        )
        padding = torch.cat((torch.zeros(b, 2, dtype=torch.bool, device=mask.device), ~mask), 1)
        values = self.output(self.encoder(encoded, src_key_padding_mask=padding)[:, 0])
        mu, logvar = values.chunk(2, -1)
        logvar = logvar.clamp(-8, 8)
        kl = 0.5 * (mu.square() + logvar.exp() - 1 - logvar).sum(-1).mean()
        return mu, logvar, kl


class MultimodalACTV5(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.policy = TemporalSpatialPolicyV3(config)
        self.config = self.policy.config
        c = self.config
        self.posterior = ActionPosteriorV5(c.model_dim, c.attention_heads, c.action_chunk_size)
        self.latent_projection = nn.Linear(32, c.model_dim, bias=False)
        self.future_head = RelativeFutureToolV4(c.model_dim)

    def forward(self, inputs, return_aux=False, teacher_actions=None, teacher_mask=None):
        if (teacher_actions is None) != (teacher_mask is None):
            raise ValueError("posterior requires both actions and valid mask")
        if teacher_actions is not None and not self.training:
            raise ValueError("deployment cannot consume future teacher actions")
        memory, padding, audit = self.policy.encode_observations(inputs)
        prior, decoded = self.policy.decode_observations(memory, padding)
        if not return_aux and teacher_actions is None:
            return prior
        current = inputs["kinematic_history"][:, -1]
        result = dict(
            action=prior,
            next_tool_xyz=self.future_head(decoded, current[:, :3], current[:, 12:18]),
            sparse_audit=audit,
            **self.policy.core.depth_aux,
        )
        if teacher_actions is not None:
            mu, logvar, kl = self.posterior(inputs["robot_state"], teacher_actions, teacher_mask)
            z = mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)
            posterior, _ = self.policy.decode_observations(memory, padding, self.latent_projection(z))
            result.update(posterior_action=posterior, kl=kl, posterior_mu=mu, posterior_logvar=logvar)
        return result
