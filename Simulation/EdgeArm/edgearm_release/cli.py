"""Print-first recipes, verified model downloads, and portable benchmark replay."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]
STAGES = {
    'prepare-sequences': 'edgearm.run82_prepare_sequences',
    'spatial-memory': 'edgearm.run82_train_spatial',
    'keypoints': 'edgearm.run88_train_keypoints',
    'dagger': 'edgearm.run94_control_fit',
    'align-actions': 'edgearm.run100_feasible_labels',
    'adapt-perception': 'edgearm.run101_keypoint_recovery',
    'independent-evaluation': 'edgearm.run102_policy_acceptance',
    'historical-act': 'edgearm.train_sparse_4d_vla_act_v26',
    'residual-rl': 'edgearm.run80_group_residual_rl',
}


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def checkpoint_manifest():
    return json.loads((ROOT / 'provenance/checkpoints.json').read_text())


def verify_weights(folder):
    for name, record in checkpoint_manifest()['files'].items():
        path = Path(folder) / name
        if not path.is_file() or sha256(path) != record['sha256']:
            raise ValueError('missing or mismatched model: ' + name)


def recipe_command(recipe):
    if set(recipe) != {'stage', 'arguments'} or recipe['stage'] not in STAGES:
        raise ValueError('recipe must declare one supported stage and arguments')
    args = recipe['arguments']
    if not isinstance(args, list) or not all(isinstance(x, str) for x in args):
        raise ValueError('arguments must be a list of strings')
    return [sys.executable, '-m', STAGES[recipe['stage']], *args]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    sub.add_parser('doctor')
    sub.add_parser('stages')
    run = sub.add_parser('run')
    run.add_argument('recipe', type=Path)
    run.add_argument('--execute', action='store_true')
    down = sub.add_parser('download')
    down.add_argument('--output', type=Path, required=True)
    ev = sub.add_parser('evaluate')
    ev.add_argument('--weights', type=Path, required=True)
    ev.add_argument('--output', type=Path, required=True)
    ev.add_argument('--workers', type=int, choices=range(1, 10), default=4)
    ev.add_argument('--groups', type=int, choices=range(1, 17), default=8)
    ev.add_argument('--group-start', type=int, default=98000100)
    a = p.parse_args()
    if a.command == 'doctor':
        import torch
        print(json.dumps({'python': sys.version.split()[0],
            'dependencies': {n: importlib.metadata.version(n) for n in ('torch', 'numpy', 'mujoco', 'scipy')},
            'cuda_available': torch.cuda.is_available(),
            'scene_exists': (ROOT / 'Simulation/SO101/edgearm_m2_m4_scene.xml').is_file(),
            'training_data_public': False, 'hardware_deployment_validated': False}, indent=2))
    elif a.command == 'stages':
        print(json.dumps(STAGES, indent=2))
    elif a.command == 'run':
        cmd = recipe_command(json.loads(a.recipe.read_text()))
        print(json.dumps({'argv': cmd, 'cwd': str(ROOT), 'execute': a.execute}, indent=2))
        if a.execute:
            subprocess.run(cmd, cwd=ROOT, check=True)
    elif a.command == 'download':
        from huggingface_hub import snapshot_download
        m = checkpoint_manifest()
        snapshot_download(repo_id=m['repo_id'], revision=m['revision'],
            allow_patterns=list(m['files']) + ['README.md', 'LICENSE'], local_dir=a.output)
        verify_weights(a.output)
        print('Verified four checkpoint hashes.')
    elif a.command == 'evaluate':
        if not 98000000 <= a.group_start < a.group_start + a.groups <= 99000000:
            p.error('benchmark group range must stay within [98000000, 99000000)')
        verify_weights(a.weights)
        import torch
        if not torch.cuda.is_available():
            p.error('the frozen evaluation policy requires an NVIDIA CUDA device')
        if sys.platform == 'linux':
            os.environ.setdefault('MUJOCO_GL', 'egl')
            os.environ.setdefault('PYOPENGL_PLATFORM', 'egl')
        from .evaluate import evaluate
        evaluate(a.weights.resolve(), a.output.resolve(), a.group_start, a.groups, a.workers)


if __name__ == '__main__':
    main()
