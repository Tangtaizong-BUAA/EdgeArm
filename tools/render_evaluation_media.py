"""Export a disclosed evaluation-only wrist replay, never a training dataset.

Requires numpy, Pillow and ffmpeg. The input trace stays private. RGB was archived
every eight control steps at 30 Hz; repeat-free playback is therefore 3.75 fps.
No interpolation, policy inference, physics reconstruction, or added frames.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np
from PIL import Image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trace', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--seed', required=True, type=int)
    parser.add_argument('--outcome', choices=['success', 'timeout'], required=True)
    parser.add_argument('--selection', default='lowest seed within outcome category; post-hoc illustration')
    parser.add_argument('--route', type=int)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    with np.load(args.trace, allow_pickle=False) as archive:
        rgb = archive['wrist_rgb'].copy()
        steps = len(archive['command'])
    assert rgb.dtype == np.uint8 and rgb.shape[1:] == (120, 160, 3)
    assert len(rgb) == (steps + 7) // 8
    stem = f'run102-{args.outcome}-{args.seed}'
    video = args.output / f'{stem}.mp4'
    cmd = ['ffmpeg', '-y', '-v', 'error', '-f', 'rawvideo', '-pixel_format', 'rgb24',
           '-video_size', '160x120', '-framerate', '15/4', '-i', '-', '-an',
           '-vf', 'scale=640:480:flags=neighbor', '-c:v', 'libx264', '-crf', '18',
           '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(video)]
    subprocess.run(cmd, input=rgb.tobytes(), check=True)
    poster_index = min(len(rgb)-1, 34)  # same step-272 rule for either outcome
    Image.fromarray(rgb[poster_index]).resize((640, 480), Image.Resampling.NEAREST).save(
        args.output / f'{stem}.png')
    subprocess.run(['ffmpeg', '-y', '-v', 'error', '-i', str(video), '-filter_complex',
                    '[0:v]scale=320:240:flags=neighbor,split[a][b];'
                    '[a]palettegen[p];[b][p]paletteuse', '-loop', '0',
                    str(args.output / f'{stem}.gif')], check=True)
    metadata = {
        'seed': args.seed, 'outcome_from_archived_result': args.outcome,
        'source_trace_sha256': hashlib.sha256(args.trace.read_bytes()).hexdigest(),
        'source': 'Run102 frozen independent evaluation; not a training sample',
        'selection': args.selection,
        'route': args.route,
        'steps': steps, 'frames': len(rgb), 'native_shape': [120, 160, 3],
        'control_hz': 30, 'archived_frame_stride': 8, 'playback_fps': 3.75,
        'presentation': 'nearest-neighbor enlargement; MP4 compression; no interpolation',
        'duration_seconds': len(rgb)/3.75,
        'new_evaluation': False, 'training_data_public': False,
        'files': {p.name: {'sha256': hashlib.sha256(p.read_bytes()).hexdigest(),
                          'bytes': p.stat().st_size}
                  for p in args.output.glob(f'{stem}.*') if p.suffix != '.json'},
    }
    (args.output / f'{stem}.json').write_text(json.dumps(metadata, indent=2)+'\n')
    print(json.dumps(metadata, indent=2))


if __name__ == '__main__':
    main()
