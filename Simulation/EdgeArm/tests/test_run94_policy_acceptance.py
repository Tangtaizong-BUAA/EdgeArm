import ast
import hashlib
import inspect

import pytest

from edgearm import run93_fast_observation as development
from edgearm import run94_frozen_fast_policy as frozen
from edgearm.run94_policy_acceptance import development_gate


def test_inference_body_and_helpers_identical_after_guard():
    for name in ("factor_command", "image_indices", "keypoint_due", "FreshCompletionLatch"):
        assert ast.dump(ast.parse(inspect.getsource(getattr(development, name)))) == ast.dump(
            ast.parse(inspect.getsource(getattr(frozen, name)))
        )
    a = ast.parse(inspect.getsource(development.episode)).body[0].body
    b = ast.parse(inspect.getsource(frozen.episode)).body[0].body
    assert ast.dump(ast.Module(body=a[2:], type_ignores=[])) == ast.dump(
        ast.Module(body=b[3:], type_ignores=[])
    )


def test_only_selected_completed_checkpoint_can_pass(tmp_path):
    control = tmp_path / "control_round_2.pt"
    control.write_bytes(b"frozen-test-weight")
    hashes = dict(
        checkpoint="map",
        vision="vision",
        keypoint="keypoint",
        control=hashlib.sha256(control.read_bytes()).hexdigest(),
    )
    row = dict(checkpoint=str(control), episodes=36, successes=26, hard_failures=1, block_out_of_bounds=0)
    state = dict(
        run="Run94",
        status="complete_pending_review",
        evaluations=[row],
        checkpoint_hashes=dict(hashes, control="initial-control-not-selected"),
    )
    development_gate(state, hashes, control)
    for override in (
        dict(successes=25),
        dict(hard_failures=2),
        dict(block_out_of_bounds=1),
        dict(episodes=35),
        dict(checkpoint=str(tmp_path / "another.pt")),
    ):
        with pytest.raises(ValueError):
            development_gate(dict(state, evaluations=[dict(row, **override)]), hashes, control)
    with pytest.raises(ValueError):
        development_gate(state, dict(hashes, keypoint="changed"), control)
    with pytest.raises(ValueError):
        development_gate(dict(state, status="running"), hashes, control)
    control.write_bytes(b"modified-weight")
    with pytest.raises(ValueError):
        development_gate(state, hashes, control)


def test_independent_guard_excludes_development_and_training():
    groups = tuple(range(98000100, 98000108))
    frozen.validate_cohort(groups[0] * 9, groups)
    for seed in (97100000 * 9, 100500000 * 9, 98000108 * 9):
        with pytest.raises(ValueError):
            frozen.validate_cohort(seed, groups)
