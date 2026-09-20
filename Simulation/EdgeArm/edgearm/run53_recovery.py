"""DAgger-style recovery labels and full behavior-layer adaptation.

The geometric teacher is simulation-only. Neither route IDs nor simulator
geometry are passed to the deployed actor. This module does not call hardware.
"""
from copy import deepcopy
from dataclasses import replace

import mujoco
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def recovery_parameters(route, variant=0):
    if route not in range(9) or variant not in (0, 1):
        raise ValueError('nine training routes and two bounded teacher variants')
    p = dict(height=.065, orientation_gain=.03, offset=.008, speed=.7,
             directional=True, lateral=0., free_orientation=False, retreat_on_goal=True)
    if route == 6:
        p.update(height=.067, orientation_gain=.003, offset=-.015, speed=.3,
                 free_orientation=True, directional=False, lateral=.022)
    elif route == 7:
        p.update(height=.055, orientation_gain=.01, offset=-.03, speed=.6,
                 directional=False)
    if variant:
        p.update(height=p['height']+.006, speed=p['speed']*.8)
    return p


class RecoveryTeacher:
    """Live-state geometric continuation, with physical lift/reposition/hold.

    Only generates a normalized submitted command. Never edits qpos, object
    state, workspace bounds, force limits, or success/contact definitions.
    """
    def __init__(self, session, route, variant=0, *, goal_retreat_coverage=.95):
        if not .95 <= goal_retreat_coverage <= 1.:
            raise ValueError('teacher may add completion margin, never relax success')
        self.session, self.p = session, recovery_parameters(route, variant)
        self.goal_retreat_coverage = float(goal_retreat_coverage)
        session.episode.dls_config = replace(session.episode.dls_config,
            orientation_correction_gain=self.p['orientation_gain'],
            directional_face_yaw_v778=self.p['directional'])
        self.phase = 'push'
        self.hold_anchor = None

    def cartesian(self, world):
        e = self.session.env
        jp = np.zeros((3, e.model.nv)); jr = np.zeros_like(jp)
        mujoco.mj_jacSite(e.model, e.data, jp, jr, e._ids['tool_site'])
        j = jp[:, :5]
        dq = j.T @ np.linalg.solve(j @ j.T + 1e-4*np.eye(3), np.asarray(world))
        desired = e.data.qpos[:6].copy()
        desired[:5] += np.clip(dq, -.025, .025)
        desired[5] = e.tool_gripper_joint_position_rad
        return np.clip((desired-e._command_reference_reported_position()) /
                       e.config.max_joint_delta, -1, 1).astype(np.float32)

    def command(self):
        e, ep, p = self.session.env, self.session.episode, self.p
        block, tool = e.block_xy(), e.tool_xyz()
        delta = e.target_xy-block
        forward = delta/max(np.linalg.norm(delta), 1e-8)
        lateral = np.array([-forward[1], forward[0]])
        if e.block_target_coverage() >= self.goal_retreat_coverage:
            self.phase = 'hold'
            if tool[2] < .12:
                self.hold_anchor = None
                return self.cartesian([0, 0, .0015])
            if self.hold_anchor is None:
                self.hold_anchor = e.data.ctrl[:6].copy()
            return np.clip((self.hold_anchor-e._command_reference_reported_position()) /
                           e.config.max_joint_delta, -1, 1).astype(np.float32)
        self.hold_anchor = None
        if self.phase == 'hold':
            self.phase = 'push'
        behind = float(np.dot(block-tool[:2], forward))
        side = float(np.dot(tool[:2]-block, lateral))-p['lateral']
        if self.phase == 'push' and (behind < -.025 or abs(side) > .065 or behind > .12):
            self.phase = 'lift'
        staging = block-.035*forward+p['lateral']*lateral
        if self.phase == 'lift':
            if tool[2] < .12:
                return self.cartesian([0, 0, .002])
            self.phase = 'align'
        if self.phase == 'align':
            if np.linalg.norm(staging-tool[:2]) > .012:
                return self.cartesian(np.r_[np.clip((staging-tool[:2])*.12, -.0015, .0015),
                                             np.clip(.12-tool[2], -.002, .002)])
            self.phase = 'descend'
        if self.phase == 'descend':
            if tool[2] > p['height']+.005:
                return self.cartesian([0, 0, -.0015])
            self.phase = 'push'
        desired = block+p['offset']*forward+p['lateral']*lateral
        error = desired-tool[:2]
        action = np.array([np.dot(error, forward)/.0015, np.dot(error, lateral)/.001,
                           (p['height']-tool[2])/.004], np.float32)
        action = np.clip(action, [-p['speed'], -1, -.5], [p['speed'], 1, .5])
        if p['free_orientation']:
            return self.cartesian(np.r_[forward*action[0]*.0015+lateral*action[1]*.001, action[2]*.004])
        return ep._submitted_action(action, 1).joint_command.copy()


def teacher_phase(step, start):
    if not isinstance(start, int) or not 0 <= start <= 300 or step < 0:
        raise ValueError('bounded real policy roll-in required')
    return step >= start


class RecoveryBehaviorLearner:
    """Success-only supervised improvement; not mislabeled as RL updates."""
    def __init__(self, actor, *, lr=3e-5):
        self.actor = actor
        self.actor.requires_grad_(True)
        self.reference = deepcopy(actor).eval().requires_grad_(False)
        self.optimizer = torch.optim.AdamW(actor.parameters(), lr=lr, weight_decay=1e-4,
                                          fused=next(actor.parameters()).is_cuda)
        self.steps = 0

    def update(self, old, recovery):
        for batch in (old, recovery):
            if not torch.all(batch['bc_mask'] == 1):
                raise ValueError('only actually successful teacher continuations are BC labels')
        old_action = self.actor(old['x'], old['base'])
        new_action = self.actor(recovery['x'], recovery['base'])
        old_loss = F.smooth_l1_loss(old_action/.05, old['action']/.05)
        recovery_loss = F.smooth_l1_loss(new_action/.05, recovery['action']/.05)
        with torch.no_grad():
            anchor = self.reference(old['x'], old['base'])
        retention = F.smooth_l1_loss(old_action/.05, anchor/.05)
        loss = .5*old_loss + .5*recovery_loss + .02*retention
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        norm = nn.utils.clip_grad_norm_(self.actor.parameters(), 1.)
        if not torch.isfinite(loss) or not torch.isfinite(norm):
            raise FloatingPointError('nonfinite recovery behavior update')
        self.optimizer.step()
        self.steps += 1
        return dict(loss=float(loss.detach()), old_loss=float(old_loss.detach()),
                    recovery_loss=float(recovery_loss.detach()), retention=float(retention.detach()))
