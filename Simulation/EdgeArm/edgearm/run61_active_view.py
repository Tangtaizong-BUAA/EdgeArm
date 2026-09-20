"""Bounded wrist-view feasibility diagnostic, NOT a learned push policy.

The controller consumes reported joints and the static arm/camera calibration
only. Object state is reserved for the caller's diagnostic measurements.
All motion goes through the existing Command-V2 plant and safety filter.
"""
import mujoco
import numpy as np
from scipy.optimize import least_squares


class WristSurvey:
    def __init__(self, model, tool_site, camera_id, joint_delta=.055):
        self.model = model
        self.data = mujoco.MjData(model)
        self.tool_site, self.camera_id = tool_site, camera_id
        self.joint_delta = float(joint_delta)
        self.phase = 'lift'
        self.steps = 0
        self.target = None
        self.fit = None

    def fk(self, q):
        q = np.asarray(q, np.float64)
        if q.shape != (6,) or not np.isfinite(q).all():
            raise ValueError('six finite reported joint positions required')
        # Fresh model-default object coordinates are irrelevant to arm FK.
        # In particular, no live environment data, goal, or route is read.
        self.data.qpos[:6] = q
        self.data.qvel[:] = 0
        mujoco.mj_forward(self.model, self.data)
        tool = self.data.site_xpos[self.tool_site].copy()
        camera = self.data.cam_xpos[self.camera_id].copy()
        forward = -self.data.cam_xmat[self.camera_id].reshape(3, 3)[:, 2].copy()
        return tool, camera, forward

    def inspection_target(self, reported):
        """One fixed table survey pose, not a selected-object target."""
        q = np.asarray(reported, np.float64).copy()
        bounds = self.model.jnt_range[:5].copy()
        lower, upper = bounds[:, 0]+.002, bounds[:, 1]-.002
        def error(x):
            candidate = np.r_[x, q[5]]
            tool, camera, forward = self.fk(candidate)
            direction = np.array([.31, 0., .026])-camera
            direction /= max(np.linalg.norm(direction), 1e-8)
            return np.r_[(tool-np.array([.20, 0., .22]))*4,
                         (forward-direction)*.6, (x-q[:5])*.005]
        fit = least_squares(error, np.clip(q[:5], lower, upper),
                            bounds=(lower, upper), max_nfev=150,
                            ftol=1e-8, xtol=1e-8, gtol=1e-8)
        target = np.r_[fit.x, q[5]]
        tool, camera, forward = self.fk(target)
        self.fit = dict(cost=float(fit.cost), tool=tool.tolist(),
                        camera=camera.tolist(), optical_forward=forward.tolist())
        return target

    def command(self, reported):
        q = np.asarray(reported, np.float64)[:6].copy()
        tool, _, _ = self.fk(q)
        self.steps += 1
        if self.phase == 'lift' and (tool[2] >= .175 or self.steps > 110):
            self.phase = 'survey'
            self.target = self.inspection_target(q)
        if self.phase == 'lift':
            jp = np.zeros((3, self.model.nv))
            jr = np.zeros_like(jp)
            mujoco.mj_jacSite(self.model, self.data, jp, jr, self.tool_site)
            j = jp[:, :5]
            dq = j.T @ np.linalg.solve(j @ j.T + 1e-4*np.eye(3), [0, 0, .002])
            command = np.r_[np.clip(dq, -.018, .018), 0.]/self.joint_delta
        else:
            command = np.clip(self.target-q, -.012, .012)/self.joint_delta
        return np.clip(command, -1, 1).astype(np.float32)


class SurveyAndReturn:
    """Observe, retrace OWN reported joint history, then settle at initial q.

    The prefix is 220 real control steps, included in the unchanged 900-step
    episode limit. This is not a reference demonstration or a teleport reset.
    """
    outbound_steps = 90
    prefix_steps = 220

    def __init__(self, model, tool_site, camera_id, joint_delta=.055):
        self.survey = WristSurvey(model, tool_site, camera_id, joint_delta)
        self.path = []
        self.steps = 0
        self.initial_q = None

    def command(self, reported):
        q = np.asarray(reported, np.float64)[:6].copy()
        if q.shape != (6,) or not np.isfinite(q).all():
            raise ValueError('finite reported joints required')
        if self.steps >= self.prefix_steps:
            raise RuntimeError('observation prefix already finished')
        if self.initial_q is None:
            self.initial_q = q.copy()
        if self.steps < self.outbound_steps:
            self.path.append(q.copy())
            command = self.survey.command(q)
        else:
            index = self.steps-self.outbound_steps
            target = self.path[-1-index] if index < len(self.path) else self.initial_q
            command = np.clip(target-q, -.018, .018)/self.survey.joint_delta
        self.steps += 1
        return np.clip(command, -1, 1).astype(np.float32)


def validate_survey_job(job):
    if not job.get('active_wrist_survey', False):
        return False
    if job['active_wrist_survey'] is not True:
        raise ValueError('explicit, fixed observation prefix required')
    if any(job.get(k) for k in ('reference_store', 'workspace_adapter')):
        raise ValueError('survey cannot mix reference trajectories or plant changes')
    if job.get('recovery_teacher'):
        if (job.get('survey_training_collection') is not True or
                job.get('source') != 'run62_active_view_recovery_collection' or
                job['recovery_teacher'].get('start') not in (220, 250)):
            raise ValueError('teacher allowed only in explicitly marked Run62 training collection')
    elif job.get('survey_training_collection'):
        raise ValueError('training collection needs a marked teacher suffix')
    if job.get('max_steps', 900) != 900 or job.get('noise_std', 0.) != 0.:
        raise ValueError('preserve the 900-step deterministic evaluation protocol')
    return True
