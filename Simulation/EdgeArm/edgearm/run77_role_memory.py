"""Bounded role-separated temporal perception and matched control adaptation.

Static goals retain the survey anchors. Moving blocks cannot attend to frames
older than 0.7 seconds, including indirectly through the goal queries. This is
a supervised mechanism experiment, not RL or original ACT.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import time

from .run76_color_dagger import ColorFrames, mixture_loader, recode
from .run75_observation_training import SurveyFrames
from .run67_visual_state import VisualState, visual_batch, evaluate_episode
from .run63_control_probe import TinyTarget
from .run68_visual_control import fit
from .run34_repeat_eval import deterministic_runtime
from .constrained_recovery_run40 import summarize
from .train_staged_hybrid_contact_sac import _atomic_json
import torch
from torch import nn
from torch.nn import functional as F


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('source-state','old-collection','output'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--wall-seconds',type=int,default=650)
    p.add_argument('--visual-updates',type=int,default=1000)
    p.add_argument('--control-updates',type=int,default=2000)
    p.add_argument('--workers',type=int,default=9)
    a=p.parse_args()
    if not 300<=a.wall_seconds<=900 or not 50<=a.visual_updates<=1500 or not 1<=a.workers<=9:
        raise ValueError('bounded final mechanism experiment')
    source=json.loads(a.source_state.read_text())
    if source['status'] not in ('complete_pending_review','failed') or not source.get('best_vision'):
        raise ValueError('completed Run76 development selection required')
    roots=[Path(source['reused_first_collection'])]
    for folder in sorted(a.source_state.parent.glob('collection_*')):
        if (folder/'summary.json').is_file():roots.append(folder)
    a.output.mkdir(parents=True,exist_ok=False);deterministic_runtime();torch.set_num_threads(2)
    started=time.time();deadline=started+a.wall_seconds
    state=dict(run='Run77',status='running',started=started,step=0,round=0,evaluations=[],
        total_updates=a.visual_updates+a.control_updates,recent_block_seconds=.7,
        source_state=str(a.source_state),source_vision=source['best_vision'],source_control=source['best_control'],
        training_roots=[str(x) for x in roots],training_kind='role_separated_visual_supervision_not_RL',
        static_goal_memory='all_past_frames_including_survey_anchors',moving_block_memory='last_0.7_seconds',
        fixed_wrist_survey=True,retain_overview=True,fixed_survey_steps=220,max_steps=900,
        actor_uses_simulator_state=False,original_ACT_checkpoint=False,independent_acceptance=False,
        production_admission=False,export_admission=False,final_vla_acceptance=False)
    def publish(phase=None):
        if phase:state['phase']=phase
        state.update(updated=time.time(),elapsed_seconds=time.time()-started)
        _atomic_json(a.output/'run_state.json',state)
    try:
        publish('role_visual_fit')
        old=SurveyFrames(a.old_collection);new=ColorFrames(roots)
        loader=mixture_loader(old,new,a.visual_updates,64)
        visual=VisualState(recent_block_seconds=.7).cuda()
        visual.load_state_dict(torch.load(source['best_vision'],map_location='cpu',weights_only=True)['model'])
        optimizer=torch.optim.AdamW(visual.parameters(),lr=.00003,weight_decay=.0001,fused=True)
        state.update(old_training_frames=len(old),new_training_frames=len(new))
        for batch in loader:
            if time.time()>deadline-240:raise TimeoutError('reserve development time')
            b=visual_batch(batch)
            with torch.autocast('cuda',dtype=torch.bfloat16):
                world=visual(b['rgb'],b['pose'],b['K'],b['age'])
                error=F.smooth_l1_loss(world/.02,b['xy']/.02,beta=.05,reduction='none').mean(-1)
                loss=((error*b['mask']).sum(-1)/b['mask'].sum(-1).clamp_min(1)).mean()
            optimizer.zero_grad(set_to_none=True);loss.backward()
            nn.utils.clip_grad_norm_(visual.parameters(),5,error_if_nonfinite=True);optimizer.step()
            state['step']+=1
            if state['step']%50==0:state['metrics']=dict(loss=float(loss.detach()));publish()
        a.vision=a.output/'vision.pt'
        torch.save(dict(model={k:v.detach().cpu().clone() for k,v in visual.state_dict().items()},
            recent_block_seconds=.7,actor_uses_simulator_state=False,original_ACT_checkpoint=False,
            production_admission=False,export_admission=False,final_vla_acceptance=False),a.vision)
        del loader,old,new,optimizer;visual.eval();publish('role_control_recode')
        old_data=recode(visual,SurveyFrames(a.old_collection,action_only=True))
        online_data=recode(visual,ColorFrames(roots,action_only=True))
        del visual;torch.cuda.empty_cache()
        control=TinyTarget(118,512).cuda()
        control.load_state_dict(torch.load(source['best_control'],map_location='cpu',weights_only=True)['model'])
        state.update(vision_sha256=hashlib.sha256(a.vision.read_bytes()).hexdigest(),
            recoded_rows=len(old_data[0]),online_rows=len(online_data[0]))
        a.updates=a.control_updates;a.learning_rate=.00005
        checkpoint=fit(control,old_data,online_data,a,state,publish,deadline)
        del control;torch.cuda.empty_cache()
        results=[];state.update(phase_episodes_total=36,phase_episodes_completed=0)
        publish('role_autonomous_development')
        with ProcessPoolExecutor(a.workers,mp_context=mp.get_context('spawn')) as pool:
            jobs=[(g*9+r,str(a.vision),str(checkpoint),str(a.output/'development'),None,
                dict(fixed_wrist_survey=True,retain_overview=True))
                for g in range(97100000,97100004) for r in range(9)]
            for future in as_completed([pool.submit(evaluate_episode,j) for j in jobs]):
                results.append(future.result());state.update(phase_episodes_completed=len(results),partial=summarize(results));publish()
        summary=summarize(results)|dict(pairs={str(r):summarize([x for x in results if x['seed']%9==r]) for r in range(9)})
        _atomic_json(a.output/'development/summary.json',dict(summary=summary,results=results))
        state.update(status='complete_pending_review',evaluations=[summary],vision=str(a.vision),control=str(checkpoint))
        publish('finished')
    except BaseException as exc:state.update(status='failed',error=repr(exc));publish();raise


if __name__=='__main__':main()
