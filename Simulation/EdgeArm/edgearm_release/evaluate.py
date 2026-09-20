"""Replay the published benchmark without claiming a new independent acceptance."""
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import multiprocessing as mp
from pathlib import Path
import time

from .cli import ROOT, sha256


def job(args):
    from edgearm.run102_frozen_joint_policy import episode
    result = episode(args)
    result['independent_acceptance'] = False
    result['evaluation_kind'] = 'public_benchmark_replay'
    Path(result['result_path']).write_text(json.dumps(result, indent=2) + '\n')
    return result


def evaluate(weights, output, group_start, count, workers):
    from edgearm.visual_policy_acceptance import report
    output.mkdir(parents=True, exist_ok=False)
    groups = tuple(range(group_start, group_start + count))
    source = {str(p.relative_to(ROOT)): sha256(p)
              for p in sorted((ROOT / 'Simulation/EdgeArm').rglob('*.py'))}
    freeze = {'weights': {n: sha256(weights / (n + '.pt'))
              for n in ('checkpoint', 'vision', 'control', 'keypoint')},
              'source_sha256': source, 'groups': groups, 'created': time.time(),
              'evaluation_kind': 'public_benchmark_replay', 'independent_acceptance': False}
    (output / 'freeze.json').write_text(json.dumps(freeze, indent=2) + '\n')
    jobs = [(g * 9 + r, *(str(weights / (n + '.pt')) for n in
             ('checkpoint', 'vision', 'control', 'keypoint')), str(output / 'episodes'),
             'camera_clearance', groups) for g in groups for r in range(9)]
    results = []
    with ProcessPoolExecutor(workers, mp_context=mp.get_context('spawn')) as pool:
        for future in as_completed([pool.submit(job, x) for x in jobs]):
            results.append(future.result())
            print(json.dumps({'completed': len(results), 'total': len(jobs)}), flush=True)
    current = {str(p.relative_to(ROOT)): sha256(p)
               for p in sorted((ROOT / 'Simulation/EdgeArm').rglob('*.py'))}
    if current != source:
        raise RuntimeError('source changed during evaluation')
    summary = report(results, groups)
    summary.update(independent_acceptance=False, evaluation_kind='public_benchmark_replay')
    summary.pop('target_point_estimate_met', None)
    summary.pop('statistical_lower_bound_at_least_80', None)
    (output / 'summary.json').write_text(json.dumps({'summary': summary, 'results': results}, indent=2) + '\n')
    print(json.dumps(summary, indent=2))

