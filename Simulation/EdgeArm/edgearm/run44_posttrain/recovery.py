"""Fresh post-ACT correction components, not a launched online RL workflow.

New ACT features MUST invalidate old residual/critic/replay state. The readout
has no old residual anchor, no simulator truth, and a zero residual starts at
exactly the new ACT prior. Environment/reward adapters remain a separate gate.
"""
import hashlib
import json
import torch
from torch import nn
from ..candidate_command_contract_v2 import ACTION_CONTRACT
from ..constrained_recovery_run40 import CalibratedLearner
from ..run42.learner import DistributedLearner
from .data import digest


class ACTPriorReadout(nn.Module):
    def __init__(self, base, *, action_query_index=0):
        super().__init__()
        self.base=base.eval().requires_grad_(False)
        self.config=base.config
        if not isinstance(action_query_index, int) or not 0 <= action_query_index < self.config.action_chunk_size:
            raise ValueError('invalid ACT execution query')
        if action_query_index and self.config.action_head_space not in (
                'absolute_joint_target_v57', 'applied_target_delta_v56'):
            raise ValueError('lookahead requires absolute joint targets, not future relative commands')
        self.action_query_index=action_query_index
        self.feature_dim=2*self.config.model_dim+37

    def train(self, mode=True):
        super().train(mode)
        self.base.eval()
        return self

    @torch.no_grad()
    def forward(self, inputs):
        memory,padding,_=self.base.policy.encode_observations(inputs)
        actions,decoded=self.base.policy.decode_observations(memory,padding)
        base=actions[:,self.action_query_index]
        h=torch.cat((decoded[:,self.action_query_index],inputs['robot_state'],base),-1)
        valid=~padding
        pooled=(memory*valid[...,None]).sum(1)/valid.sum(1,keepdim=True).clamp_min(1)
        mask=inputs['command_feedback_mask'][:,-1,None]
        feedback=torch.where(mask,inputs['tracking_error_history'][:,-1],0.)
        command=torch.where(mask,inputs['command_history'][:,-1],0.)
        x=torch.cat((h,pooled,base,feedback,command,mask.to(base.dtype)),-1)
        return x.detach(),base.detach()


class FreshCorrector(DistributedLearner):
    """Reuse tested gradient AllReduce, but NEVER load the old ACT's Q/residual."""
    def __init__(self, feature_dim, privileged_dim, device, *, residual_limit=.12):
        # privileged_dim already includes any training-only domain context.
        CalibratedLearner.__init__(self,feature_dim,privileged_dim,device,residual_limit=residual_limit)
        self.q_forward,self.actor_forward=self.q,self.actor


def replay_identity(base_checkpoint, feature_schema='run44-pure-act-prior-v1',
                    reward_schema='run37-potential-gamma0p999-v1'):
    return dict(act_sha256=digest(base_checkpoint),feature_schema=feature_schema,
        action_contract_sha256=hashlib.sha256(json.dumps(ACTION_CONTRACT,sort_keys=True).encode()).hexdigest(),
        reward_schema=reward_schema,contact_profile='task_goal_v1',old_run33_anchor_used=False)


def validate_replay_identity(recorded, expected):
    if recorded!=expected:
        raise ValueError('re-encode raw observations and recollect reward/critic transitions for this ACT lineage')


def sampled_rows_budget(new_transitions, batch_per_gpu=1024, world_size=2, rows_per_transition=64):
    """Define replay reuse in sampled rows, independent of GPU/global batch size."""
    if min(new_transitions,batch_per_gpu,world_size,rows_per_transition)<=0:
        raise ValueError('positive replay budget required')
    return max(1,new_transitions*rows_per_transition//(batch_per_gpu*world_size))
