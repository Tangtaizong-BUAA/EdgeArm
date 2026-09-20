import numpy as np
import pytest

from edgearm.run82_prepare_sequences import training_split
from edgearm.run82_train_spatial import sequence_chunk


def item():
    times = np.array([0, 8, 16, 24, 32])
    n = len(times)
    obs = dict(time_step=times, rgb=np.stack([np.full((120, 160, 3), t, np.uint8) for t in times]),
               pose=np.zeros((n, 12), np.float32), K=np.eye(3, dtype=np.float32),
               proprio=np.zeros((n, 114), np.float32), selected=np.array([0, 4]))
    labs = dict(xyz=np.zeros((n, 7, 3), np.float32), points=np.zeros((n, 7, 8, 3), np.float32),
                present=np.ones((n, 7), bool), visible=np.ones((n, 7), bool),
                command=np.zeros((n, 6), np.float32), action_valid=np.ones(n, bool))
    return obs, labs


def test_training_groups_do_not_accept_development_or_holdout():
    assert training_split(100200000*9) == 'train'
    assert training_split(100200011*9+8) == 'validation'
    for group in (97100000, 98000000, 98000100, 100300000):
        with pytest.raises(ValueError): training_split(group*9)


def test_sequence_images_are_causal_and_padding_excluded_from_losses():
    ins, labs, valid = sequence_chunk([item()], 3, 4, device='cpu')
    assert valid.tolist() == [[True, True, False, False]]
    assert (ins['age'] <= 0).all()
    assert ins['rgb'][0, 0].max() <= 24
    assert not labs['present'][0, 2:].any()
    assert not labs['action_valid'][0, 2:].any()
    assert not labs['frame_valid'][0, 2:].any()


def test_blind_augmentation_removes_that_image_from_later_history():
    blind = [np.array([False, False, True, True, False])]
    ins, labs, _ = sequence_chunk([item()], 0, 5, blind=blind, device='cpu')
    assert 16 not in np.unique(ins['rgb'].numpy())
    assert 24 not in np.unique(ins['rgb'].numpy())
    assert not labs['visible'][0, 2:4].any()
    assert labs['present'][0, 2:4].all()  # memory supervision survives occlusion
