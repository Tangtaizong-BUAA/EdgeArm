import ast
import hashlib
import inspect

import pytest

from edgearm import run96_clearance_probe, run99_joint_probe, run102_frozen_joint_policy
from edgearm.run102_policy_acceptance import development_gate


def nested_function(callable_, name):
    return next(node for node in ast.walk(ast.parse(inspect.getsource(callable_)))
                if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name == name)


def test_projection_invocation_and_workspace_check_match_development():
    assert ast.dump(nested_function(run96_clearance_probe.episode, "constrained_factor")) == ast.dump(
        nested_function(run102_frozen_joint_policy.episode, "constrained_factor"))
    old = nested_function(run99_joint_probe.episode, "CheckedSession")
    new = ast.parse(inspect.getsource(run102_frozen_joint_policy.CheckedSession)).body[0]
    assert ast.dump(old) == ast.dump(new)


def test_specific_visual_checkpoint_and_all_static_weights_are_bound(tmp_path):
    keypoint = tmp_path / "keypoints_step_600.pt"
    keypoint.write_bytes(b"frozen-test-weight")
    digest = hashlib.sha256(keypoint.read_bytes()).hexdigest()
    hashes = dict(checkpoint="map", vision="vision", control="action", keypoint=digest)
    row = dict(keypoint_checkpoint=str(keypoint), keypoint_sha256=digest,
               episodes=36, successes=26, hard_failures=1, block_out_of_bounds=0)
    state = dict(run="Run101", status="complete_pending_review", evaluations=[row],
                 checkpoint_hashes=dict(hashes, keypoint="original-perception-anchor"))
    development_gate(state, hashes, keypoint)
    for fields in (dict(successes=25), dict(episodes=35), dict(hard_failures=2),
                   dict(block_out_of_bounds=1), dict(keypoint_sha256="changed")):
        with pytest.raises(ValueError):
            development_gate(dict(state, evaluations=[dict(row, **fields)]), hashes, keypoint)
    for name in ("checkpoint", "vision", "control"):
        with pytest.raises(ValueError):
            development_gate(state, dict(hashes, **{name: "changed"}), keypoint)
    with pytest.raises(ValueError):
        development_gate(dict(state, status="running"), hashes, keypoint)


def test_independent_wrapper_rejects_training_and_development_before_model_loading():
    for seed in (97100000 * 9, 100500000 * 9, 98000108 * 9):
        with pytest.raises(ValueError):
            run102_frozen_joint_policy.episode((seed, "", "", "", "", "", "camera_clearance",
                                               tuple(range(98000100, 98000108))))
