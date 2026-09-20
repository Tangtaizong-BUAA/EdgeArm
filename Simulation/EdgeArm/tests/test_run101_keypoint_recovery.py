import numpy as np
import pytest
import torch

from edgearm.run101_replay_keypoint_labels import check_reproduction, training_split
from edgearm.run101_keypoint_recovery import draw_balanced


def test_perception_validation_split_never_admits_policy_eval():
    assert training_split(100500009 * 9) == "train"
    assert training_split(100500010 * 9) == "validation"
    for group in (97100000, 98000000, 98000100, 100500012):
        with pytest.raises(ValueError):
            training_split(group * 9)


def test_replay_rejects_drift_in_report_object_or_images():
    check_reproduction(0, 0, 0, 0)
    for bad in ((1e-4, 0, 0, 0), (0, 1e-5, 0, 0), (0, 0, 3, 0), (0, 0, 1, .02)):
        with pytest.raises(ValueError):
            check_reproduction(*bad)


def test_route_draw_is_equal_and_validation_excluded():
    route = torch.tensor(np.repeat(np.arange(9), 10))
    validation = torch.arange(90) % 2 == 0
    ids = draw_balanced(dict(route=route, validation=validation), per_route=8)
    assert not validation[ids].any()
    assert torch.equal(torch.bincount(route[ids]), torch.full((9,), 8))
