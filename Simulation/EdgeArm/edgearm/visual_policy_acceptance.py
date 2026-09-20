"""One-shot, frozen-candidate evaluation on previously unused scene groups.

This module never trains or selects a checkpoint. A group is claimed before
execution, including if the evaluation is interrupted. Task-level acceptance
does not imply original ACT, Home, real-world or production acceptance.
"""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import shutil
import time

import numpy as np
import torch

from .constrained_recovery_run40 import summarize
from .run67_visual_state import evaluate_episode
from .train_staged_hybrid_contact_sac import _atomic_json


def wilson(successes,total):
    if not 0<=successes<=total or total<=0:raise ValueError('valid binomial counts required')
    z=1.959963984540054;p=successes/total;d=1+z*z/total
    center=(p+z*z/(2*total))/d
    half=z*math.sqrt(p*(1-p)/total+z*z/(4*total*total))/d
    return [max(0.,center-half),min(1.,center+half)]


def report(results,groups):
    if len(results)!=9*len(groups) or len({r['seed'] for r in results})!=len(results):
        raise ValueError('complete, unique, nine-route evaluation required')
    expected={g*9+r for g in groups for r in range(9)}
    if {r['seed'] for r in results}!=expected:raise ValueError('frozen seed contract mismatch')
    summary=summarize(results)
    summary['success_rate']=summary['successes']/summary['episodes']
    summary['wilson_95_episode_approximation']=wilson(summary['successes'],summary['episodes'])
    # Nine routes within one generated scene can be dependent. Resample whole
    # scene groups as a second descriptive interval, not individual episodes.
    means=np.asarray([np.mean([r['safe_success'] for r in results if r['seed']//9==g]) for g in groups])
    rng=np.random.default_rng(713)
    boot=means[rng.integers(len(groups),size=(20000,len(groups)))].mean(1)
    summary['scene_group_bootstrap_95']=np.quantile(boot,[.025,.975]).tolist()
    summary['independent_scene_groups']=len(groups)
    summary['end_kind_counts']=dict(Counter(r['end_kind'] for r in results))
    summary['terminal_reason_counts']=dict(Counter(r.get('terminal_reason',r['end_kind']) for r in results))
    summary['out_of_bounds_episodes']=sum(r.get('terminal_reason')=='block_out_of_bounds'
        or r['end_kind']=='block_out_of_bounds' for r in results)
    summary['physical_failure_episodes']=sum(r['end_kind']=='hard_failure'
        or r.get('terminal_reason')=='block_out_of_bounds' or r['end_kind']=='block_out_of_bounds' for r in results)
    summary['pairs']={str(route):summarize([r for r in results if r['seed']%9==route]) for route in range(9)}
    for value in summary['pairs'].values():
        value['success_rate']=value['successes']/value['episodes']
        value['wilson_95']=wilson(value['successes'],value['episodes'])
    summary.update(target_point_estimate_met=summary['success_rate']>=.8,
        statistical_lower_bound_at_least_80=summary['wilson_95_episode_approximation'][0]>=.8,
        independent_acceptance=True,actor_uses_simulator_state=False,teacher_assisted=False,
        original_ACT_checkpoint=False,start_stage='CONTACT_TRANSPORT_HOLD',
        exact_home_evaluated=False,real_robot_evaluated=False,language_parser='seven_named_colors',
        temporal_tracker=any(r.get('temporal_tracker',False) for r in results),
        fixed_wrist_survey=any(r.get('fixed_wrist_survey',False) for r in results),
        retain_overview=any(r.get('retain_overview',False) for r in results),
        learned_observation_controller=False,
        geometry_gated=any(r.get('geometry_gated',False) for r in results),
        task_conditioned_initial_pose=True,deployment_route_id_input=False,
        production_admission=False,export_admission=False,final_vla_acceptance=False)
    return summary


def claim_groups(registry,groups,freeze):
    registry=Path(registry);registry.mkdir(parents=True,exist_ok=True)
    if any((registry/f'group_{g}.json').exists() for g in groups):
        raise ValueError('held-out group already consumed or claimed')
    for group in groups:
        with (registry/f'group_{group}.json').open('x') as f:
            json.dump(dict(group=group,claimed=time.time(),freeze=str(freeze)),f)


def evaluation_job(seed,checkpoints,output,options=None):
    job=(seed,checkpoints['vision']['frozen'],checkpoints['control']['frozen'],str(output))
    if options:
        return job+(checkpoints.get('tracker',{}).get('frozen'),dict(options))
    return job+(checkpoints['tracker']['frozen'],) if 'tracker' in checkpoints else job


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--vision',type=Path,required=True);p.add_argument('--control',type=Path,required=True)
    p.add_argument('--tracker',type=Path)
    p.add_argument('--fixed-wrist-survey',action='store_true')
    p.add_argument('--retain-overview',action='store_true')
    p.add_argument('--output',type=Path,required=True);p.add_argument('--registry',type=Path,required=True)
    p.add_argument('--group-start',type=int,default=98000000);p.add_argument('--groups',type=int,default=8)
    p.add_argument('--workers',type=int,default=6)
    a=p.parse_args()
    if a.retain_overview and not a.fixed_wrist_survey:
        raise ValueError('retained overview requires the real 220-command survey')
    options=dict(fixed_wrist_survey=a.fixed_wrist_survey,retain_overview=a.retain_overview)
    if not 98000000<=a.group_start<99000000 or not 8<=a.groups<=16 or not 1<=a.workers<=9:
        raise ValueError('bounded independent evaluation required')
    if a.group_start+a.groups>99000000:raise ValueError('evaluation groups out of range')
    a.output.mkdir(parents=True,exist_ok=False)
    checkpoints={}
    required_tracker_hash=None
    paths=[('vision',a.vision),('control',a.control)]+([('tracker',a.tracker)] if a.tracker else [])
    if a.tracker:os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    for name,path in paths:
        saved=torch.load(path,map_location='cpu',weights_only=True)
        if saved.get('actor_uses_simulator_state') is not False:
            raise ValueError('only a deployable non-oracle candidate may be tested')
        if name=='tracker' and saved.get('vision_sha256')!=checkpoints['vision']['sha256']:
            raise ValueError('tracker visual representation mismatch')
        if name=='control':required_tracker_hash=saved.get('tracker_sha256')
        dest=a.output/(name+'.pt');shutil.copy2(path,dest)
        checkpoints[name]=dict(source=str(path),frozen=str(dest),sha256=hashlib.sha256(dest.read_bytes()).hexdigest())
        del saved
    if required_tracker_hash and required_tracker_hash!=checkpoints.get('tracker',{}).get('sha256'):
        raise ValueError('action policy requires its matching temporal tracker')
    source_root=Path(__file__).parent
    source_names=('visual_policy_acceptance.py','run67_visual_state.py','run63_control_probe.py',
        'run42/session.py','run42/domain.py','production_env.py','multichoice_scene_v1.py','candidate_command_contract_v2.py',
        'run71_temporal_tracking.py','run73_anchored_specialist.py','run74_observation_probe.py','run61_active_view.py')
    source_hashes={name:hashlib.sha256((source_root/name).read_bytes()).hexdigest() for name in source_names}
    for name in source_names:
        dest=a.output/'frozen_source'/name;dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(source_root/name,dest)
    groups=list(range(a.group_start,a.group_start+a.groups))
    seeds=[g*9+r for g in groups for r in range(9)]
    freeze=dict(created=time.time(),checkpoints=checkpoints,source_hashes=source_hashes,groups=groups,seeds=seeds,
        selection_role='one_frozen_candidate_no_test_selection',max_steps=900,hold_seconds=3,
        temporal_tracker=a.tracker is not None,
        evaluation_options=options,scripted_survey_commands=220 if a.fixed_wrist_survey else 0,
        learned_observation_controller=False,
        initial_coverage=0,success_definition_changed=False,physics_changed=False,
        production_admission=False,export_admission=False,final_vla_acceptance=False)
    _atomic_json(a.output/'candidate_freeze.json',freeze)
    claim_groups(a.registry,groups,a.output/'candidate_freeze.json')
    state=dict(run='VisualIndependent',status='running',started=time.time(),completed=0,total=len(seeds),
        phase='independent_frozen_evaluation',groups=groups,independent_acceptance=True,evaluations=[],
        evaluation_options=options,
        actor_uses_simulator_state=False,production_admission=False,export_admission=False,final_vla_acceptance=False)
    def publish():
        state.update(updated=time.time(),elapsed_seconds=time.time()-state['started'])
        _atomic_json(a.output/'run_state.json',state)
    publish();results=[]
    try:
        with ProcessPoolExecutor(a.workers,mp_context=mp.get_context('spawn')) as pool:
            futures=[pool.submit(evaluate_episode,evaluation_job(s,checkpoints,a.output/'episodes',options)) for s in seeds]
            for future in as_completed(futures):
                row=future.result();row['independent_acceptance']=True;results.append(row)
                state.update(completed=len(results),partial=summarize(results));publish()
        current={name:hashlib.sha256((source_root/name).read_bytes()).hexdigest() for name in source_names}
        if current!=source_hashes:raise ValueError('evaluation source changed while running')
        summary=report(results,groups);_atomic_json(a.output/'summary.json',dict(summary=summary,results=results))
        state.update(status='complete',summary=summary,target_point_estimate_met=summary['target_point_estimate_met']);publish()
    except BaseException as exc:
        state.update(status='failed',error=repr(exc),independent_result_incomplete=True);publish();raise


if __name__=='__main__':main()
