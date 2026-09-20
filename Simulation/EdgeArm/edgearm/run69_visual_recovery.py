"""Correct wrist perception on newly visited training states, freezing control.

Only collection folders supply additional labels. Development and held-out
rollouts are never training inputs. This is supervised perception adaptation,
not RL, and neither the original ACT checkpoint nor task rules are changed.
"""
import argparse
from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset, WeightedRandomSampler

from .constrained_recovery_run40 import summarize
from .run34_repeat_eval import deterministic_runtime
from .run67_visual_state import (OnlineWristStates, VisualState, WristStates,
    evaluate_episode, history_indices, validate, visual_batch)
from .train_staged_hybrid_contact_sac import _atomic_json


class CorrectionWristStates(Dataset):
    """Saved visual labels are separate from the student's estimated x fields."""
    keys=('wrist_rgb','camera_pose','camera_K','frame_steps','visual_labels_xy','visual_label_mask','selected')
    def __init__(self,root,cache_root,*,group_start=99700000):
        if group_start not in (99700000,99800000):
            raise ValueError('dedicated collection groups only')
        self.rows,self.index,self.cache=[],[],OrderedDict()
        cache_root=Path(cache_root);cache_root.mkdir(parents=True,exist_ok=False)
        for path in sorted(Path(root).glob('collection_*/episode_*/perceived_commands.npz')):
            result=json.loads((path.parent/'result.json').read_text());seed=result['seed']
            if not result.get('collection') or not group_start<=seed//9<group_start+100000:
                raise ValueError('only dedicated training collection is allowed')
            dest=cache_root/f'episode_{seed}';dest.mkdir(exist_ok=False)
            with np.load(path,allow_pickle=False) as z:
                n=len(z['frame_steps'])
                if not np.all(np.diff(z['frame_steps'])>0):raise ValueError('strict frame order required')
                if not z['visual_label_mask'][:,z['selected']].all():raise ValueError('selected labels required')
                for key in self.keys:np.save(dest/(key+'.npy'),z[key],allow_pickle=False)
            _atomic_json(dest/'source.json',dict(source=str(path),seed=seed,training_only=True))
            row_id=len(self.rows);self.rows.append((dest,seed));self.index.extend((row_id,k) for k in range(n))
        if not self.index:raise ValueError('empty correction set')

    def __len__(self):return len(self.index)

    def __getitem__(self,index):
        row,k=self.index[index];path,_=self.rows[row]
        if row not in self.cache:
            self.cache[row]={key:np.load(path/(key+'.npy'),mmap_mode='r') for key in self.keys}
            while len(self.cache)>24:self.cache.popitem(last=False)
        else:self.cache.move_to_end(row)
        a=self.cache[row];t=int(a['frame_steps'][k]);requested=history_indices(t)
        ids=np.maximum(0,np.searchsorted(a['frame_steps'],requested,side='right')-1)
        actual=a['frame_steps'][ids]
        return dict(rgb=a['wrist_rgb'][ids].copy(),pose=a['camera_pose'][ids].copy(),
            K=a['camera_K'].copy(),age=np.asarray((actual-t)/30,np.float32),
            xy=a['visual_labels_xy'][k].copy(),mask=a['visual_label_mask'][k].copy(),
            selected=a['selected'].copy(),time_step=np.int64(t))


class TaggedDataset(Dataset):
    def __init__(self,dataset,routes,source):
        if len(dataset)!=len(routes):raise ValueError('route tag length mismatch')
        self.dataset,self.routes,self.source=dataset,np.asarray(routes,np.int64),source
    def __len__(self):return len(self.dataset)
    def __getitem__(self,i):
        return self.dataset[i]|dict(route=self.routes[i],source=np.int64(self.source))


def equal_route_weights(routes,source_weight):
    routes=np.asarray(routes);counts=np.bincount(routes,minlength=9)
    if len(counts)!=9 or np.any(counts==0):raise ValueError('all nine routes required')
    return source_weight/(9*counts[routes])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',type=Path,required=True);p.add_argument('--vision',type=Path,required=True)
    p.add_argument('--control',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--old-online-root',type=Path,required=True);p.add_argument('--correction-root',type=Path,required=True)
    p.add_argument('--updates',type=int,default=6000);p.add_argument('--batch',type=int,default=64)
    p.add_argument('--evaluate-every',type=int,default=1500);p.add_argument('--workers',type=int,default=6)
    p.add_argument('--wall-seconds',type=int,default=2300);p.add_argument('--learning-rate',type=float,default=.0001)
    p.add_argument('--smoke-only',action='store_true')
    a=p.parse_args()
    if not 100<=a.updates<=12000 or not 8<=a.batch<=128 or not 1<=a.workers<=9:
        raise ValueError('bounded perception pilot required')
    if not 300<=a.wall_seconds<=3500 or not .00001<=a.learning_rate<=.0003 or a.evaluate_every<100:
        raise ValueError('finite pilot required')
    a.output.mkdir(parents=True,exist_ok=False);deterministic_runtime();torch.set_num_threads(2)
    state=dict(run='Run69',status='running',phase='prepare_visual_recovery',started=time.time(),step=0,
        total_updates=a.updates,batch=a.batch,evaluations=[],target_rate=.8,
        control_checkpoint=str(a.control),control_sha256=hashlib.sha256(a.control.read_bytes()).hexdigest(),
        initial_vision_checkpoint=str(a.vision),actor_uses_simulator_state=False,visual_grounding=True,
        original_ACT_checkpoint=False,action_policy_changed=False,perception_policy_changed=True,
        training_kind='on_policy_visual_label_supervision_not_RL',independent_acceptance=False,
        production_admission=False,export_admission=False,final_vla_acceptance=False,
        start_stage='CONTACT_TRANSPORT_HOLD',sampled_source_routes=[[0]*9,[0]*9])
    def publish(phase=None):
        if phase:state['phase']=phase
        state.update(updated=time.time(),elapsed_seconds=time.time()-state['started'])
        _atomic_json(a.output/'run_state.json',state)
    publish()
    try:
        records=[r for r in json.loads(a.manifest.read_text())['records'] if r['split']=='train' and r['run54_pool']=='recovery']
        val_groups=set(sorted({r['seed']//9 for r in records})[::5])
        old=WristStates([r for r in records if r['seed']//9 not in val_groups])
        old_val=WristStates([r for r in records if r['seed']//9 in val_groups],stride=12)
        prior=OnlineWristStates([a.old_online_root],a.output/'old_online_cache')
        correction=CorrectionWristStates(a.correction_root,a.output/'correction_cache')
        groups=sorted({seed//9 for _,seed in correction.rows})
        if len(groups)<4:raise ValueError('at least four independent collection groups required')
        correction_val_groups=set(groups[-2:])
        train_ids=[i for i,(row,k) in enumerate(correction.index) if correction.rows[row][1]//9 not in correction_val_groups]
        val_ids=[i for i,(row,k) in enumerate(correction.index) if correction.rows[row][1]//9 in correction_val_groups]
        old_routes=[old.records[row]['seed']%9 for row,t in old.index]+[prior.rows[row][1]%9 for row,t in prior.index]
        new_routes=[correction.rows[correction.index[i][0]][1]%9 for i in train_ids]
        old_train=ConcatDataset([old,prior]);new_train=Subset(correction,train_ids)
        train=ConcatDataset([TaggedDataset(old_train,old_routes,0),TaggedDataset(new_train,new_routes,1)])
        weights=np.r_[equal_route_weights(old_routes,.5),equal_route_weights(new_routes,.5)]
        sampler=WeightedRandomSampler(torch.from_numpy(weights),a.updates*a.batch,replacement=True)
        loader=DataLoader(train,batch_size=a.batch,sampler=sampler,num_workers=6,pin_memory=True,persistent_workers=True)
        validators=[DataLoader(x,batch_size=a.batch,num_workers=2,pin_memory=True) for x in (old_val,Subset(correction,val_ids))]
        state.update(training_frames=len(train),old_training_frames=len(old_train),new_training_frames=len(new_train),
            validation_frames=len(old_val)+len(val_ids),validation_groups=sorted(val_groups),
            correction_validation_groups=sorted(correction_val_groups),
            correction_training_episodes=sum(seed//9 not in correction_val_groups for _,seed in correction.rows),
            correction_validation_episodes=sum(seed//9 in correction_val_groups for _,seed in correction.rows))
        model=VisualState().cuda();model.load_state_dict(torch.load(a.vision,map_location='cpu',weights_only=True)['model'])
        optimizer=torch.optim.AdamW(model.parameters(),lr=a.learning_rate,weight_decay=.0001,fused=True)
        state['validation_before']={'old':validate(model,validators[0],limit=10000),'new':validate(model,validators[1],limit=10000)}
        publish('visual_recovery_training')
        for step,batch in enumerate(loader):
            if time.time()>state['started']+a.wall_seconds-180:
                state['budget_stopped']=True;break
            b=visual_batch(batch)
            with torch.autocast('cuda',dtype=torch.bfloat16):
                prediction=model(b['rgb'],b['pose'],b['K'],b['age'])
                error=F.smooth_l1_loss(prediction/.02,b['xy']/.02,beta=.05,reduction='none').mean(-1)
                # A six-label legacy frame must not outweigh a two-label
                # correction frame merely because more colors are annotated.
                loss=((error*b['mask']).sum(-1)/b['mask'].sum(-1).clamp_min(1)).mean()
            optimizer.zero_grad(set_to_none=True);loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(),5,error_if_nonfinite=True);optimizer.step()
            for group in optimizer.param_groups:group['lr']=.00001+(a.learning_rate-.00001)*.5*(1+np.cos(np.pi*(step+1)/a.updates))
            for source in range(2):
                counts=np.bincount(batch['route'][batch['source']==source].numpy(),minlength=9)
                state['sampled_source_routes'][source]=[int(x+y) for x,y in zip(state['sampled_source_routes'][source],counts)]
            state['step']=step+1
            if (step+1)%50==0:
                state['metrics']=dict(loss=float(loss.detach()),position_training_mean_mm=float(
                    torch.linalg.vector_norm(prediction.detach()-b['xy'],dim=-1)[b['mask']].mean()*1000))
                publish('visual_recovery_training')
            if (step+1)%a.evaluate_every==0 or step+1==a.updates:
                state['visual_validation']=validate(model,validators[0],limit=10000)
                state['correction_validation']=validate(model,validators[1],limit=10000)
                checkpoint=a.output/f'visual_step_{step+1}.pt'
                torch.save(dict(model={k:v.detach().cpu().clone() for k,v in model.state_dict().items()},
                    actor_uses_simulator_state=False,original_ACT_checkpoint=False,
                    production_admission=False,export_admission=False,final_vla_acceptance=False),checkpoint)
                publish('visual_validation')
                if not a.smoke_only:
                    results=[];folder=a.output/f'development_{step+1}'
                    state.update(phase_episodes_completed=0,phase_episodes_total=18);publish('visual_development')
                    torch.cuda.empty_cache()
                    with ProcessPoolExecutor(a.workers,mp_context=mp.get_context('spawn')) as pool:
                        futures=[pool.submit(evaluate_episode,(g*9+r,str(checkpoint),str(a.control),str(folder)))
                                 for g in range(97000000,97000002) for r in range(9)]
                        for future in as_completed(futures):
                            results.append(future.result());state.update(phase_episodes_completed=len(results),partial=summarize(results));publish()
                    summary=summarize(results)|dict(label=f'development_{step+1}',visual_grounding=True,
                        actor_uses_simulator_state=False,independent_acceptance=False,
                        pairs={str(r):summarize([x for x in results if x['seed']%9==r]) for r in range(9)})
                    _atomic_json(folder/'summary.json',dict(summary=summary,results=results));state['evaluations'].append(summary)
                    if summary['successes']>=15:state['development_gate_passed']=True;break
                model.train()
        state['status']='smoke_complete' if a.smoke_only else 'complete_pending_review';publish('finished')
    except BaseException as exc:
        state.update(status='failed',error=repr(exc));publish();raise


if __name__=='__main__':main()
