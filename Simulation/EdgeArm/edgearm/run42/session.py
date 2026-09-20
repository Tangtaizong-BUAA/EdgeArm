"""New reset path isolated from the frozen Run41 evaluator."""

import mujoco
import numpy as np
from PIL import Image

from ..branch_session_run37 import BranchSession
from ..contact_audit_run33 import ContactAudit
from ..materialize_temporal_packets_v3 import kinematics
from ..staged_push_rl import StagedPushEpisode, StagedPushStage
from ..temporal_online_v3 import TemporalOnlineBuffer
from .domain import DomainScene, environment_config
from .sensors import corrupt_rgb


class DomainSession(BranchSession):
    def __init__(self, config, action_contract, seed, parameters, *, render=True):
        self.parameters = parameters
        self.episode = StagedPushEpisode(seed=int(seed), environment_config=environment_config(parameters),
                                        scene_mode='multichoice_v1')
        self.episode.multichoice = DomainScene(self.episode.env, parameters)
        self.episode.reset(seed=int(seed), stage=StagedPushStage.CONTACT_TRANSPORT_HOLD)
        self.env = self.episode.env
        self.device, self.action_contract = 'cpu', action_contract
        cam = self.env._ids['cameras']['wrist']
        f = 60 / np.tan(np.deg2rad(self.env.model.cam_fovy[cam]) / 2)
        true_k = np.array([[f, 0, 79.5], [0, f, 59.5], [0, 0, 1]], np.float32)
        estimated_k = true_k.copy()
        estimated_k[:2, :2] *= 1 + parameters['intrinsics_scale_error']
        self.camera_audit = dict(actual_K=true_k.tolist(), estimated_K=estimated_k.tolist(),
                                 actual_fovy_deg=float(self.env.model.cam_fovy[cam]),
                                 focal_equivalent_mm=parameters['focal_equivalent_mm'],
                                 depth_source='RGBDepthStudent prediction, never simulator depth')
        self.buffer = TemporalOnlineBuffer(config, self.episode.multichoice.contract['instruction'], estimated_k)
        # Preserve Run40's render then resize convention for nominal equivalence.
        self.renderer = mujoco.Renderer(self.env.model, width=640, height=480) if render else None
        self.audit = ContactAudit('task_goal_v1')
        self.audit.__enter__()
        self.inputs = self.reported = None
        self.end_kind, self.reason = 'sampler_cut', 'nonterminal'
        self.view_counts = dict(frames=0, selected_block_center=0, selected_target_center=0)

    def _audit_view(self):
        # Diagnostic only: true geometry never enters the actor input. This is
        # a frustum test, NOT an occlusion/segmentation or RGB-recognition claim.
        cam = self.env._ids['cameras']['wrist']
        rotation = self.env.data.cam_xmat[cam].reshape(3, 3)
        position = self.env.data.cam_xpos[cam]
        k = np.asarray(self.camera_audit['actual_K'])
        self.view_counts['frames'] += 1
        for name, key in [('selected_block_center', 'block_geom'), ('selected_target_center', 'target_geom')]:
            point = rotation.T @ (self.env.data.geom_xpos[self.env._ids[key]] - position)
            depth = -point[2]
            if depth <= 0:
                continue
            u = k[0, 0] * point[0] / depth + k[0, 2]
            v = k[1, 2] - k[1, 1] * point[1] / depth
            self.view_counts[name] += int(0 <= u < 160 and 0 <= v < 120)

    def visibility_summary(self):
        n = self.view_counts['frames']
        return dict(frames=n, block_center_in_frustum_fraction=self.view_counts['selected_block_center'] / max(n, 1),
                    target_center_in_frustum_fraction=self.view_counts['selected_target_center'] / max(n, 1),
                    occlusion_checked=False, diagnostic_only=True)

    def observe(self):
        if self.inputs is not None:
            return self.inputs
        if self.renderer is None:
            raise RuntimeError('no-render mechanical smoke cannot provide actor observations')
        self.renderer.update_scene(self.env.data, camera='edgearm_wrist')
        self._audit_view()
        rgb = np.asarray(Image.fromarray(self.renderer.render()).resize((160, 120)))
        rgb = corrupt_rgb(rgb, self.parameters, float(self.env.data.time))
        self.reported = np.asarray(self.env.observation()['joint_state'], np.float32)
        tool, pose = kinematics(self.env, self.reported[:6], self.reported[6:])
        self.buffer.observe(rgb=rgb, joint=self.reported, tool=tool, camera_pose=pose,
                            time_s=float(self.env.data.time), geometry_valid=True)
        self.inputs = {k: v.to(self.device) for k, v in self.buffer.tensors().items()}
        return self.inputs

    def close(self):
        self.audit.__exit__(None, None, None)
        if self.renderer is not None:
            self.renderer.close()
