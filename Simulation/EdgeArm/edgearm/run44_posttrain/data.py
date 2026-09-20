"""Validated mmap packets; completed trajectories are the split boundary."""
from collections import OrderedDict
import hashlib
import json
from pathlib import Path

import numpy as np
from torch.utils.data import Dataset

from ..temporal_input_contract_v3 import make_sample

FIELDS = ('rgb', 'joint', 'previous_motion', 'tool', 'camera_pose', 'time',
          'geometry_valid', 'command', 'command_valid', 'applied_target',
          'tracking_error', 'feedback_valid', 'K')
COHORTS = ('old_rl', 'old_human', 'new_familiar', 'new_edge', 'new_perturbed')


def digest(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def validate_arrays(z):
    if set(z) != set(FIELDS):
        raise ValueError('raw packet field mismatch')
    n = len(z['time'])
    if n < 2 or z['rgb'].shape != (n, 120, 160, 3) or z['rgb'].dtype != np.uint8:
        raise ValueError('invalid wrist RGB dimensions')
    shapes = {'joint': (n, 12), 'previous_motion': (n, 6), 'tool': (n, 18),
        'camera_pose': (n, 12), 'command': (n, 6), 'applied_target': (n, 6),
        'tracking_error': (n, 6), 'K': (3, 3)}
    for key, shape in shapes.items():
        if z[key].shape != shape or not np.isfinite(z[key]).all():
            raise ValueError('invalid '+key)
    for k in ('geometry_valid', 'command_valid', 'feedback_valid'):
        if z[k].shape != (n,) or z[k].dtype != bool:
            raise ValueError('invalid mask '+k)
    if not np.isfinite(z['time']).all() or not (np.diff(z['time']) > 0).all():
        raise ValueError('nonmonotonic physical time')
    valid = z['command_valid']
    if not valid.any() or np.max(np.abs(z['command'][valid])) > 1.000001:
        raise ValueError('missing or out-of-range submitted commands')
    if min(z['K'][0, 0], z['K'][1, 1]) <= 0:
        raise ValueError('invalid camera calibration')
    rotation = z['camera_pose'][z['geometry_valid'], 3:].reshape(-1, 3, 3)
    if len(rotation) and (not np.allclose(rotation @ rotation.transpose(0, 2, 1), np.eye(3), atol=.002)
                         or not np.allclose(np.linalg.det(rotation), 1., atol=.002)):
        raise ValueError('invalid camera rotation')
    return dict(length=n, commands=int(valid.sum()), rgb_bytes=int(z['rgb'].nbytes))


def read_parts(path):
    path = Path(path).resolve()
    manifest = json.loads((path/'episode_manifest.json').read_text())
    if not manifest['complete'] or manifest['schema'] != 'run39-wrist-parts-v1':
        raise ValueError('incomplete or unknown collection')
    segments, previous, expected = [], None, 0
    for part in manifest['parts']:
        file = (path/part['path']).resolve()
        if not file.is_relative_to(path) or digest(file) != part['sha256']:
            raise ValueError('part provenance mismatch')
        with np.load(file, allow_pickle=False) as data:
            z = {k: data[k] for k in FIELDS}
        n = part['action_end'] - part['action_begin']
        if part['action_begin'] != expected or n < 1 or len(z['time']) != n+1:
            raise ValueError('noncontiguous part')
        if previous is not None:
            for key in ('rgb', 'joint', 'time', 'camera_pose', 'tool'):
                if not np.array_equal(previous[key][-1], z[key][0]):
                    raise ValueError('mismatched endpoint '+key)
            if not np.array_equal(previous['K'], z['K']):
                raise ValueError('camera intrinsics changed without frame calibration')
        if not z['command_valid'][:-1].all() or z['command_valid'][-1] or not z['feedback_valid'][:-1].all():
            raise ValueError('incomplete command/feedback transaction')
        segments.append({k: z[k][:-1] for k in FIELDS if k != 'K'})
        expected, previous = part['action_end'], z
    if previous is None:
        raise ValueError('empty episode')
    arrays = {k: np.concatenate([s[k] for s in segments]+[previous[k][-1:]], axis=0)
              for k in FIELDS if k != 'K'}
    arrays['K'] = previous['K']
    validate_arrays(arrays)
    return arrays, manifest


def cohort_for_new(row):
    if row['stratum'] == 'familiar_center':
        return 'new_familiar'
    if row['stratum'].startswith('center_jitter_'):
        return 'new_perturbed'
    if row['pair'] != [1, 1]:
        return 'new_edge'
    raise ValueError('unknown collection stratum')


def check_disjoint(records):
    if any(r['split'] not in ('train','validation') for r in records):
        raise ValueError('unknown dataset split')
    train, validation = ([r for r in records if r['split'] == split] for split in ('train', 'validation'))
    if not train or not validation:
        raise ValueError('need isolated train and validation trajectories')
    for key in ('group', 'parent_source_path', 'source_sha256'):
        if {r[key] for r in train} & {r[key] for r in validation}:
            raise ValueError('train/validation leakage: '+key)
    if len({r['id'] for r in records}) != len(records):
        raise ValueError('duplicate trajectory')


class MMapEpisodes(Dataset):
    def __init__(self, records, cfg, cache_size=8):
        self.records, self.cfg, self.cache_size = records, cfg, cache_size
        self.cache = OrderedDict()

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        ri, t = index
        r = self.records[ri]
        if ri not in self.cache:
            self.cache[ri] = {k: np.load(Path(r['store'])/(k+'.npy'), mmap_mode='r', allow_pickle=False) for k in FIELDS}
            if len(self.cache) > self.cache_size:
                self.cache.popitem(last=False)
        self.cache.move_to_end(ri)
        sample = make_sample(self.cache[ri], int(t), r['instruction'], self.cfg, include_auxiliary_depth=False)
        sample.update(cohort=np.int64(COHORTS.index(r['cohort'])), record_id=np.int64(ri), time_index=np.int64(t))
        return sample
