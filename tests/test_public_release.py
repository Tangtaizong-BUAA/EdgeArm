import json
from pathlib import Path
import numpy as np
import pytest
from edgearm_release.cli import ROOT, STAGES, recipe_command


def test_recipes_only_launch_explicit_supported_modules():
    for path in (ROOT / 'recipes').glob('*.json'):
        command = recipe_command(json.loads(path.read_text()))
        assert command[1] == '-m'
        assert command[2] in STAGES.values()
    with pytest.raises(ValueError):
        recipe_command({'stage': 'arbitrary.shell', 'arguments': []})
    with pytest.raises(ValueError):
        recipe_command({'stage': 'dagger', 'arguments': 'not a list'})


def test_synthetic_causal_packet_and_privileged_label_separation(tmp_path):
    # Synthetic values only; no recorded image or training sample is embedded.
    from edgearm.run82_train_spatial import Sequences
    folder = tmp_path / 'sequence'
    folder.mkdir()
    np.savez(folder / 'inputs.npz', rgb=np.zeros((2, 120, 160, 3), np.uint8),
             pose=np.zeros((2, 4, 4)), K=np.eye(3), proprio=np.zeros((2, 114)),
             time_step=np.array([0, 8]), selected=np.array([0, 4]))
    np.savez(folder / 'labels.npz', xyz=np.zeros((2, 7, 3)))
    row = dict(seed=100200000 * 9, split='train', route=0, variant=-1, folder=str(folder))
    manifest = tmp_path / 'manifest.json'
    manifest.write_text(json.dumps({'records': [row]}))
    inputs, _ = Sequences(manifest).load(row)
    assert 'xyz' not in inputs
    np.savez(folder / 'inputs.npz', **inputs, simulator_xyz=np.zeros((2, 7, 3)))
    with pytest.raises(ValueError, match='undeclared'):
        Sequences(manifest).load(row)


def test_assets_and_model_manifest():
    from edgearm_release.cli import checkpoint_manifest
    assert (ROOT / 'Simulation/SO101/edgearm_m2_m4_scene.xml').is_file()
    assert len(checkpoint_manifest()['files']) == 4
    assert checkpoint_manifest()['training_data_public'] is False
