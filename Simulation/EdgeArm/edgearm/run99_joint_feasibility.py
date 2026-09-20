"""Joint camera/workspace constraint on a reported-state command.

The existing simulator safeguards remain unchanged. This controller solves
both geometric requirements together so a camera correction is not discarded
by a subsequent workspace projection. It is not a learned policy component.
"""

import mujoco
import numpy as np
from scipy.optimize import minimize

from .run96_camera_clearance import CameraClearanceProjection


NOMINAL_WORKSPACE = ((0.07, 0.43), (-0.27, 0.27), (0.045, 0.26))


class JointFeasibilityProjection(CameraClearanceProjection):
    def __init__(self, model, tool_site, *, workspace=NOMINAL_WORKSPACE, inset_m=0.0005):
        super().__init__(model, tool_site)
        self.workspace = np.asarray(workspace, np.float64)
        if self.workspace.shape != (3, 2) or not 0 <= inset_m <= .002:
            raise ValueError("calibrated bounded workspace required")
        self.low = self.workspace[:, 0] + inset_m
        self.high = self.workspace[:, 1] - inset_m
        if not np.all(np.isfinite(self.workspace)) or np.any(self.low >= self.high):
            raise ValueError("invalid workspace")

    def constraints(self, q, required):
        height, gradient, _ = self._geometry(q, jacobian=True)
        xyz = self.data.site_xpos[self.tool_site].copy()
        jp = np.zeros((3, self.model.nv))
        jr = np.zeros_like(jp)
        mujoco.mj_jacSite(self.model, self.data, jp, jr, self.tool_site)
        # Return metres and metres/radian; only five arm joints are solved.
        return np.r_[height - required, xyz - self.low, self.high - xyz], np.vstack(
            (gradient, jp[:, :5], -jp[:, :5])
        )

    def project(self, reported, normalized_command):
        reported, original = np.asarray(reported, np.float64), np.asarray(normalized_command, np.float64)
        if reported.shape not in ((6,), (12,)) or original.shape != (6,):
            raise ValueError("reported joints and six commands required")
        if not np.isfinite(np.r_[reported, original]).all():
            raise ValueError("finite reported state required")
        q = reported[:6]
        nominal = np.clip(original, -1, 1)
        target = np.clip(q + self.joint_delta * nominal, self.limits[:, 0], self.limits[:, 1])
        height, gradient, _ = self._geometry(q, jacobian=True)
        speed = max(0.0, -float(gradient @ reported[6:11])) if len(reported) == 12 else 0.0
        required = self.margin + min(.008, .1 * speed + max(0.0, self.margin - height))
        values, _ = self.constraints(target, required)
        if np.min(values) >= 0:
            return original.astype(np.float32), dict(active=False, feasible=True, before_m=height,
                requested_m=self._geometry(target), after_m=self._geometry(target), required_m=required)
        lower = np.maximum(q - self.joint_delta, self.limits[:, 0])
        upper = np.minimum(q + self.joint_delta, self.limits[:, 1])

        def full(x):
            return np.r_[x, target[5]]

        # Small static nonlinear solve; never calls live physics or reads an
        # object pose. Joint range and command bounds remain the original ones.
        result = minimize(
            lambda x: .5 * float(np.sum((x - target[:5]) ** 2)), target[:5],
            jac=lambda x: x - target[:5], method="SLSQP",
            bounds=list(zip(lower[:5], upper[:5], strict=True)),
            constraints=[dict(type="ineq", fun=lambda x: self.constraints(full(x), required)[0] * 100,
                              jac=lambda x: self.constraints(full(x), required)[1] * 100)],
            options=dict(maxiter=32, ftol=1e-12, disp=False),
        )
        candidate = full(result.x)
        feasible = bool(np.min(self.constraints(candidate, required)[0]) >= -1e-7)
        if not feasible:
            # Explicit fallback preserves the prior constraint and original
            # downstream safety. It is recorded, never declared safe/feasible.
            output, details = super().project(reported, original)
            details.update(feasible=False, joint_solver_success=bool(result.success),
                           joint_solver_iterations=int(result.nit), fallback="camera_only")
            return output, details
        output = np.clip((candidate - q) / self.joint_delta, -1, 1).astype(np.float32)
        realized = q + self.joint_delta * output
        values, _ = self.constraints(realized, required)
        return output, dict(
            active=bool(np.max(np.abs(output - nominal)) > 1e-6), feasible=bool(np.min(values) >= -1e-7),
            before_m=height, requested_m=self._geometry(target), after_m=self._geometry(realized),
            required_m=required, minimum_constraint_slack_m=float(np.min(values)),
            correction_max_rad=float(np.max(np.abs(output - nominal)) * self.joint_delta),
            joint_solver_success=bool(result.success), joint_solver_iterations=int(result.nit),
        )
