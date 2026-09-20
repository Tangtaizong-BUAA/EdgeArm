import numpy as np
import pytest

from edgearm.candidate_command_contract_v2 import ACTION_CONTRACT
from edgearm.run42.domain import sample_domain
from edgearm.run42.session import DomainSession
from edgearm.run99_joint_feasibility import JointFeasibilityProjection, NOMINAL_WORKSPACE
from edgearm.run99_joint_probe import episode
from edgearm.sparse_4d_vla_act_v26 import Sparse4DVLAConfigV26


def test_probe_never_consumes_independent_scenes():
    with pytest.raises(ValueError):
        episode((98000100 * 9, None, None, None, None, None, "joint_feasible"))


def test_workspace_and_camera_are_satisfied_together_without_live_state():
    seed = 873900002
    session = DomainSession(
        Sparse4DVLAConfigV26(language_max_tokens=128, visual_memory_mode="episode_anchors_v54"),
        ACTION_CONTRACT, seed, sample_domain(seed + 6001, 0), render=False)
    try:
        p = JointFeasibilityProjection(session.env.model, session.env._ids["tool_site"])
        q = np.array([.227029160, .762388468, -.427980661, .056757290, .934194326, -.161067992])
        command = np.array([-.067012988, -.103999175, -.170865923, .015092226, .008514228, .005364413])
        live = session.env.data.qpos.copy()
        original = q + .055 * command
        p._geometry(original)
        assert p.data.site_xpos[p.tool_site, 0] > NOMINAL_WORKSPACE[0][1]
        output, audit = p.project(q, command)
        assert audit["feasible"] and audit["active"]
        target = q + .055 * output
        values, jac = p.constraints(target, audit["required_m"])
        assert values.min() >= -1e-7
        assert np.max(np.abs(output)) <= 1
        np.testing.assert_array_equal(session.env.data.qpos, live)
        # Nonlinear constraint gradients, including all workspace axes.
        eps = 1e-6
        numerical = np.column_stack([(p.constraints(target + np.eye(6)[i] * eps, audit["required_m"])[0]
                    - p.constraints(target - np.eye(6)[i] * eps, audit["required_m"])[0]) / (2 * eps)
                    for i in range(5)])
        np.testing.assert_allclose(jac, numerical, atol=1e-7, rtol=1e-4)
        # A corrected feasible target is a fixed point of the controller.
        again, _ = p.project(q, output)
        np.testing.assert_allclose(again, output, atol=3e-5)
    finally:
        session.close()
