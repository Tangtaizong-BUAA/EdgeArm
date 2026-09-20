import pytest
import numpy as np

from edgearm.run100_feasible_labels import align_episode, validate_training_file


def test_only_original_108_training_episode_labels_are_accepted():
    assert validate_training_file("collection_1/memory_fast_keypoints/episode_904500000/action_labels.npz") == 904500000
    for group in (97100000, 98000100, 100500012, 100500020):
        with pytest.raises(ValueError):
            validate_training_file(f"episode_{group * 9}/action_labels.npz")
    with pytest.raises(ValueError):
        validate_training_file("episode_904500000/trace.npz")


def test_alignment_keeps_causal_input_and_records_label_change(tmp_path):
    folder = tmp_path / "episode_904500000"
    folder.mkdir()
    x = np.zeros((2, 118), np.float32)
    x[:, :6] = [.227029160, .762388468, -.427980661, .056757290, .934194326, -.161067992]
    command = np.tile([-.067012988, -.103999175, -.170865923, .015092226, .008514228, .005364413], (2, 1)).astype(np.float32)
    path = folder / "action_labels.npz"
    np.savez(path, x=x, command=command, time_step=np.array([220, 221]))
    observed, labels, route, audit = align_episode(path)
    np.testing.assert_array_equal(observed, x)
    assert labels.shape == (2, 6) and np.all(np.abs(labels) <= 1)
    assert audit["changed_over_1mrad"] == 2 and audit["rejected_rows"] == 0
    assert not audit["uses_heldout"] and not audit["certified_success_demonstrations"]
    assert np.all(route == 0)
