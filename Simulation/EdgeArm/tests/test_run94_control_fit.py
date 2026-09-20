from copy import deepcopy

import numpy as np
import pytest
import torch

from edgearm.run63_control_probe import TinyTarget
from edgearm.run94_collect import validate_collection
from edgearm.run94_control_fit import route_pools, train_step


def test_collection_never_accepts_development_or_independent_seeds():
    validate_collection(100500000 * 9, 0.5, "memory_fast_keypoints")
    for group in (97100000, 98000100, 100200000, 100500012):
        with pytest.raises(ValueError):
            validate_collection(group * 9, 0.5, "memory_fast_keypoints")
    with pytest.raises(ValueError):
        validate_collection(100500000 * 9, 1.0, "memory_fast_keypoints")


def test_route_sampler_requires_all_nine_routes():
    with pytest.raises(ValueError):
        route_pools(np.arange(8), "cpu")
    assert len(route_pools(np.arange(9), "cpu")) == 9


def test_update_changes_only_student_control_and_uses_finite_loss():
    torch.manual_seed(94)
    anchor = TinyTarget(118, 32).eval().requires_grad_(False)
    model = deepcopy(anchor).requires_grad_(True)
    x = torch.randn(36, 118)
    with torch.no_grad():
        old_y = anchor(x)
    new_y = old_y + 0.02
    pools = route_pools(np.tile(np.arange(9), 4), "cpu")
    old = (x, old_y, pools)
    new = (x, new_y, pools)
    before = {k: v.detach().clone() for k, v in anchor.state_dict().items()}
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    metrics = train_step(model, anchor, optimizer, old, new)
    assert np.isfinite(list(metrics.values())).all()
    assert all(torch.equal(before[k], v) for k, v in anchor.state_dict().items())
    assert any(not torch.equal(before[k], v) for k, v in model.state_dict().items())
