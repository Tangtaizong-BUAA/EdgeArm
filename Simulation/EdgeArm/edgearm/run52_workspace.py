"""Opt-in simulation ablation: preserve Cartesian tangential motion at bounds.

No workspace/joint limits or collision rules are relaxed. The legacy filter
remains the fallback. This changes the controller, never the actor input.
"""
import types

import mujoco
import numpy as np


def project_workspace(model, live_data, site, ranges, bounds, requested, *,
                      margin=.0003, correction_limit=.04, iterations=8):
    requested = np.asarray(requested, np.float64)
    ranges, bounds = np.asarray(ranges), np.asarray(bounds)
    if requested.shape != (6,) or not np.isfinite(requested).all():
        raise ValueError('finite six-joint request required')
    if ranges.shape != (6, 2) or bounds.shape != (3, 2) or not (ranges[:, 1]>ranges[:, 0]).all():
        raise ValueError('invalid joint/workspace bounds')
    if not 0 < margin < .005 or not 0 < correction_limit <= .055:
        raise ValueError('bounded simulation projection required')
    lower, upper = ranges.T
    anchor = np.clip(requested, lower, upper)
    q = anchor.copy()
    data = mujoco.MjData(model)
    mujoco.mj_copyData(data, model, live_data)
    jac = np.zeros((3, model.nv))
    rot = np.zeros_like(jac)

    def position(value):
        data.qpos[:6] = value
        mujoco.mj_forward(model, data)
        return data.site_xpos[site].copy()

    def valid(xyz):
        return bool(((xyz >= bounds[:, 0]) & (xyz <= bounds[:, 1])).all())

    start = position(q)
    if valid(start):
        return q, dict(kind='unchanged_or_joint_clip', iterations=0)
    # Keep valid x/y/z coordinates; correct only the violated coordinates.
    goal = np.clip(start, bounds[:, 0]+margin, bounds[:, 1]-margin)
    for i in range(iterations):
        xyz = position(q)
        if valid(xyz):
            return q, dict(kind='tangent_projected', iterations=i,
                           xyz_before=start.tolist(), xyz_after=xyz.tolist(),
                           correction_rad=float(np.max(np.abs(q-anchor))))
        mujoco.mj_jacSite(model, data, jac, rot, site)
        j = jac[:, :6]
        dq = j.T @ np.linalg.solve(j @ j.T + 1e-8*np.eye(3), goal-xyz)
        norm = np.max(np.abs(dq))
        if norm > .015:
            dq *= .015/norm
        best = q
        error = np.linalg.norm(goal-xyz)
        for scale in (1., .5, .25):
            candidate = np.clip(q+scale*dq, np.maximum(lower, anchor-correction_limit),
                                np.minimum(upper, anchor+correction_limit))
            candidate_error = np.linalg.norm(goal-position(candidate))
            if candidate_error < error:
                best, error = candidate, candidate_error
        if np.array_equal(best, q):
            break
        q = best
    xyz = position(q)
    if valid(xyz):
        return q, dict(kind='tangent_projected', iterations=iterations,
                       xyz_before=start.tolist(), xyz_after=xyz.tolist(),
                       correction_rad=float(np.max(np.abs(q-anchor))))
    return None, dict(kind='legacy_fallback', iterations=iterations)


def install(session, mode):
    if mode != 'tangent_same_bounds_v1':
        raise ValueError('unknown Run52 simulation controller ablation')
    env = session.env
    original = env._safety_filter
    bounds = np.asarray([env.config.workspace_x, env.config.workspace_y, env.config.workspace_z])
    stats = dict(mode=mode, original_workspace_bounds=bounds.tolist(), limits_relaxed=False,
                 calls=0, tangent_projected=0, legacy_fallback=0, maximum_correction_rad=0.)

    def wrapped(self, requested):
        stats['calls'] += 1
        q, diagnostic = project_workspace(self.model, self.data, self._ids['tool_site'],
                                         self.joint_ranges, bounds, requested)
        if q is None:
            stats['legacy_fallback'] += 1
            return original(requested)
        reasons = []
        if np.any(self._target_changed_mask(requested, np.clip(requested, self.joint_ranges[:, 0], self.joint_ranges[:, 1]))):
            reasons.append('joint_limit')
        if diagnostic['kind'] == 'tangent_projected':
            stats['tangent_projected'] += 1
            stats['maximum_correction_rad'] = max(stats['maximum_correction_rad'], diagnostic['correction_rad'])
            reasons.append('run52_tangent_same_workspace')
        return q.copy(), '+'.join(reasons)

    env._safety_filter = types.MethodType(wrapped, env)
    session.run52_workspace_stats = stats
