"""Camera-housing clearance from reported joints and static calibration only.

This is an explicit kinematic controller constraint, not a learned ability or
a change to the simulator's physical collision/success criteria.
"""

import itertools

import mujoco
import numpy as np


class CameraClearanceProjection:
    def __init__(self, model, tool_site, *, clearance_m=0.002, joint_delta=0.055):
        if not 0 < clearance_m <= 0.005:
            raise ValueError("bounded camera clearance required")
        self.model = model
        self.data = mujoco.MjData(model)  # No live physical data is accepted.
        self.tool_site = int(tool_site)
        self.margin = float(clearance_m)
        self.joint_delta = float(joint_delta)
        self.housing = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "production_wrist_camera_housing")
        self.desk = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "edgearm_desk")
        if min(self.housing, self.desk) < 0:
            raise ValueError("declared camera and calibrated desk geometry required")
        box = mujoco.mjtGeom.mjGEOM_BOX
        if model.geom_type[self.housing] != box or model.geom_type[self.desk] != box:
            raise ValueError("box camera and desk calibration required")
        if model.geom_bodyid[self.desk] != 0 or not np.allclose(model.geom_quat[self.desk], [1, 0, 0, 0]):
            raise ValueError("static horizontal desk calibration required")
        self.desk_top = float(model.geom_pos[self.desk, 2] + model.geom_size[self.desk, 2])
        signs = np.asarray(list(itertools.product((-1.0, 1.0), repeat=3)))
        self.corners = signs * model.geom_size[self.housing]
        self.body = int(model.geom_bodyid[self.housing])
        self.limits = model.jnt_range[:6].copy()

    def _geometry(self, q, *, jacobian=False):
        self.data.qpos[:6] = q
        mujoco.mj_kinematics(self.model, self.data)
        rotation = self.data.geom_xmat[self.housing].reshape(3, 3)
        points = self.corners @ rotation.T + self.data.geom_xpos[self.housing]
        point = points[int(np.argmin(points[:, 2]))]
        height = float(point[2] - self.desk_top)
        if not jacobian:
            return height
        mujoco.mj_comPos(self.model, self.data)
        jp = np.zeros((3, self.model.nv))
        jr = np.zeros_like(jp)
        mujoco.mj_jac(self.model, self.data, jp, jr, point, self.body)
        gradient = jp[2, :5].copy()
        mujoco.mj_jacSite(self.model, self.data, jp, jr, self.tool_site)
        return height, gradient, jp[:2, :5].copy()

    def project(self, reported, normalized_command):
        reported = np.asarray(reported, dtype=np.float64)
        q = reported[:6]
        velocity = reported[6:12] if reported.shape == (12,) else np.zeros(6)
        original = np.asarray(normalized_command, dtype=np.float64)
        if (
            reported.shape not in ((6,), (12,))
            or original.shape != (6,)
            or not np.isfinite(np.r_[reported, original]).all()
        ):
            raise ValueError("finite reported joints and six motor commands required")
        nominal = np.clip(original, -1, 1)
        initial_clearance, gradient, _ = self._geometry(q, jacobian=True)
        # Three control frames of braking reserve use reported velocity, not
        # actual simulator velocity. A position-only endpoint check missed the
        # observed collisions while the camera was still moving downward.
        downward_speed = max(0.0, -float(gradient @ velocity[:5]))
        required = self.margin + min(0.008, 0.1 * downward_speed + max(0.0, self.margin - initial_clearance))
        candidate = np.clip(q + self.joint_delta * nominal, self.limits[:, 0], self.limits[:, 1])
        requested_clearance = self._geometry(candidate)
        if min(initial_clearance, requested_clearance) >= required:
            return original.astype(np.float32), dict(
                active=False,
                before_m=initial_clearance,
                requested_m=requested_clearance,
                after_m=requested_clearance,
            )
        # Preserve tool XY to first order while lifting the camera's lowest
        # corner; fall back to the full gradient if the nullspace is singular.
        lower = np.maximum(q - self.joint_delta, self.limits[:, 0])
        upper = np.minimum(q + self.joint_delta, self.limits[:, 1])
        for _ in range(6):
            height, gradient, xy_jac = self._geometry(candidate, jacobian=True)
            deficit = required - height
            if deficit <= 1e-5:
                break
            null_gradient = gradient - xy_jac.T @ np.linalg.solve(
                xy_jac @ xy_jac.T + 1e-7 * np.eye(2), xy_jac @ gradient
            )
            direction = null_gradient if gradient @ null_gradient > 1e-7 else gradient
            increment = direction * min(deficit / max(float(gradient @ direction), 1e-10), 1000.0)
            candidate[:5] += np.clip(increment, -0.025, 0.025)
            candidate = np.clip(candidate, lower, upper)
        result = np.clip((candidate - q) / self.joint_delta, -1, 1).astype(np.float32)
        return result, dict(
            active=bool(np.max(np.abs(result - nominal)) > 1e-6),
            before_m=initial_clearance,
            requested_m=requested_clearance,
            required_m=required,
            after_m=self._geometry(q + self.joint_delta * result),
            correction_max_rad=float(np.max(np.abs(result - nominal)) * self.joint_delta),
        )
