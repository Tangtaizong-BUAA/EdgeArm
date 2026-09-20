"""Learn a persistent causal spatial correction, without changing the action net.

Frozen wrist-visual features and deployable proprioception enter a GRU every
four commands. Only training labels contain object truth. Whole collection
episodes, including unsuccessful ones, supply perception labels, not success
demonstrations. This is supervised tracking, not RL or original ACT training.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import time

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from .constrained_recovery_run40 import summarize
from .run34_repeat_eval import deterministic_runtime
from .run67_visual_state import OnlineWristStates, VisualState, evaluate_episode, visual_batch
from .run68_visual_control import CommandImages, estimated_inputs
from .run69_visual_recovery import CorrectionWristStates
from .train_staged_hybrid_contact_sac import _atomic_json


def tracker_features(x, visual_tokens, selection):
    if x.shape[-1] != 118 or visual_tokens.shape[-2:] != (7, 128):
        raise ValueError('deployable spatial/proprioceptive schema required')
    token=visual_tokens[torch.arange(len(x),device=x.device),selection[:,0]]
    return torch.cat((x,token),-1)


class TemporalTracker(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('mean',torch.zeros(246))
        self.register_buffer('scale',torch.ones(246))
        self.project=nn.Sequential(nn.Linear(246,192),nn.LayerNorm(192),nn.SiLU())
        self.memory=nn.GRU(192,192,num_layers=2,batch_first=True)
        self.head=nn.Sequential(nn.Linear(192,96),nn.SiLU(),nn.Linear(96,2))
        nn.init.zeros_(self.head[-1].weight);nn.init.zeros_(self.head[-1].bias)

    def forward(self,x,hidden=None):
        z=self.project((x-self.mean)/self.scale)
        z,hidden=self.memory(z,hidden)
        return .2*torch.tanh(self.head(z)),hidden


def load_tracker(path,vision_path):
    saved=torch.load(path,map_location='cpu',weights_only=True)
    if saved['vision_sha256']!=hashlib.sha256(Path(vision_path).read_bytes()).hexdigest():
        raise ValueError('tracker requires its fixed visual representation')
    if saved.get('actor_uses_simulator_state') is not False:
        raise ValueError('non-oracle tracker required')
    model=TemporalTracker().cuda().eval();model.load_state_dict(saved['model'])
    return model


class SequenceFrames(Dataset):
    def __init__(self,images):self.images,self.commands=images,CommandImages(images)
    def __len__(self):return len(self.images)
    def __getitem__(self,i):
        row,_=self.images.index[i]
        return self.commands[i]|dict(episode=np.int64(row))


def encode(a,state,publish):
    model=VisualState().cuda().eval()
    model.load_state_dict(torch.load(a.vision,map_location='cpu',weights_only=True)['model'])
    sources=[OnlineWristStates([a.online_root],a.output/'online_cache')]
    sources += [CorrectionWristStates(root,a.output/f'correction_cache_{i}',group_start=start)
                for i,(root,start) in enumerate(zip(a.correction_roots,(99700000,99800000)))]
    sequences=[]
    for source,images in enumerate(sources):
        validation=set(sorted({seed//9 for _,seed in images.rows})[-2:]) if source else set()
        parts={}
        loader=DataLoader(SequenceFrames(images),batch_size=96,num_workers=6,pin_memory=True,shuffle=False)
        with torch.inference_mode():
            for batch in loader:
                b=visual_batch(batch);world,tokens=model(b['rgb'],b['pose'],b['K'],b['age'],return_features=True)
                x=estimated_inputs(b['proprio'],world,b['selected'])
                features=tracker_features(x,tokens,b['selected']).cpu().numpy()
                rows=torch.arange(len(world),device='cuda');index=b['selected'][:,0]
                raw=world[rows,index].cpu().numpy();target=b['xy'][rows,index].cpu().numpy()
                if not b['mask'][rows,index].all():raise ValueError('training position labels required')
                for j,episode in enumerate(batch['episode'].tolist()):
                    parts.setdefault(episode,[]).append((int(batch['time_step'][j]),features[j],raw[j],target[j]))
                state['encoded_frames']=state.get('encoded_frames',0)+len(features);publish('encoding_causal_sequences')
        for episode,items in sorted(parts.items()):
            seed=images.rows[episode][1];items.sort(key=lambda item:item[0])
            times=np.array([x[0] for x in items])
            if times[0]!=0 or not np.all(np.diff(times)==4):
                raise ValueError('complete causal four-step sequence required')
            sequences.append(dict(seed=seed,source=source,route=seed%9,validation=seed//9 in validation,
                x=np.stack([x[1] for x in items]),raw=np.stack([x[2] for x in items]),
                target=np.stack([x[3] for x in items])))
    del model;torch.cuda.empty_cache()
    torch.save(sequences,a.output/'encoded_sequences.pt')
    state.update(training_episodes=sum(not r['validation'] for r in sequences),
        validation_episodes=sum(r['validation'] for r in sequences),
        validation_groups=sorted({r['seed']//9 for r in sequences if r['validation']}))
    return sequences


def pack(sequences):
    n,maxlen=len(sequences),max(len(s['x']) for s in sequences)
    x=np.zeros((n,maxlen,246),np.float32);raw=np.zeros((n,maxlen,2),np.float32)
    target=np.zeros_like(raw);mask=np.zeros((n,maxlen),bool)
    for i,s in enumerate(sequences):
        count=len(s['x']);x[i,:count]=s['x'];raw[i,:count]=s['raw'];target[i,:count]=s['target'];mask[i,:count]=True
    return [torch.from_numpy(v).cuda() for v in (x,raw,target,mask)]


def sequence_weights(sequences):
    weights=np.zeros(len(sequences),np.float64)
    for source in sorted({s['source'] for s in sequences}):
        for route in range(9):
            ids=[i for i,s in enumerate(sequences) if s['source']==source and s['route']==route]
            if not ids:raise ValueError('every source must cover all nine routes')
            weights[ids]=1/len(ids)
    return torch.from_numpy(weights/weights.sum()).cuda()


def tracking_metrics(model,tensors):
    model.eval();values=[];raw_values=[]
    with torch.inference_mode():
        for start in range(0,len(tensors[0]),32):
            x,raw,target,mask=[v[start:start+32] for v in tensors]
            corrected=raw+model(x)[0]
            values.append(torch.linalg.vector_norm(corrected-target,dim=-1)[mask].cpu())
            raw_values.append(torch.linalg.vector_norm(raw-target,dim=-1)[mask].cpu())
    v,r=torch.cat(values),torch.cat(raw_values);model.train()
    return dict(block_mean_mm=float(v.mean()*1000),block_p90_mm=float(v.quantile(.9)*1000),
        raw_block_mean_mm=float(r.mean()*1000),frames=len(v))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('vision','control','online-root','output'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--correction-roots',type=Path,nargs=2,required=True)
    p.add_argument('--encoded',type=Path)
    p.add_argument('--updates',type=int,default=3000);p.add_argument('--batch',type=int,default=32)
    p.add_argument('--evaluate-every',type=int,default=1000);p.add_argument('--workers',type=int,default=6)
    p.add_argument('--wall-seconds',type=int,default=2300);p.add_argument('--learning-rate',type=float,default=.0002)
    p.add_argument('--smoke-only',action='store_true');a=p.parse_args()
    if not (100<=a.updates<=6000 and 8<=a.batch<=64 and 100<=a.evaluate_every<=a.updates
            and 1<=a.workers<=9 and 300<=a.wall_seconds<=3500 and .00001<=a.learning_rate<=.001):
        raise ValueError('bounded tracking experiment required')
    a.output.mkdir(parents=True,exist_ok=False);deterministic_runtime();torch.set_num_threads(2)
    state=dict(run='Run71',started=time.time(),status='running',phase='initializing',step=0,
        total_updates=a.updates,evaluations=[],target_rate=.8,
        vision_sha256=hashlib.sha256(a.vision.read_bytes()).hexdigest(),
        control_sha256=hashlib.sha256(a.control.read_bytes()).hexdigest(),
        training_kind='causal_recurrent_position_supervision_not_RL',
        actor_uses_simulator_state=False,original_ACT_checkpoint=False,visual_grounding=True,
        action_policy_changed=False,perception_policy_changed=True,tracker_update_interval=4,
        start_stage='CONTACT_TRANSPORT_HOLD',independent_acceptance=False,
        production_admission=False,export_admission=False,final_vla_acceptance=False)
    def publish(phase=None):
        if phase:state['phase']=phase
        state.update(updated=time.time(),elapsed_seconds=time.time()-state['started'])
        _atomic_json(a.output/'run_state.json',state)
    publish();deadline=state['started']+a.wall_seconds
    try:
        if a.encoded:
            parent=json.loads((a.encoded.parent/'run_state.json').read_text())
            if parent['vision_sha256']!=state['vision_sha256']:raise ValueError('encoded visual representation mismatch')
            sequences=torch.load(a.encoded,weights_only=False,map_location='cpu')
        else:sequences=encode(a,state,publish)
        if any(97000000<=s['seed']//9<99000000 for s in sequences):raise ValueError('evaluation leak')
        train=[s for s in sequences if not s['validation']];validation=[s for s in sequences if s['validation']]
        state.update(training_episodes=len(train),validation_episodes=len(validation),
            training_frames=sum(len(s['x']) for s in train),validation_frames=sum(len(s['x']) for s in validation),
            validation_groups=sorted({s['seed']//9 for s in validation}),sampled_source_routes=[[0]*9 for _ in range(3)])
        train_tensors=pack(train);val_tensors=pack(validation);weights=sequence_weights(train)
        model=TemporalTracker().cuda()
        normal=train_tensors[0][train_tensors[3]]
        model.mean.copy_(normal.mean(0));model.scale.copy_(normal.std(0).clamp_min(.02));del normal
        optimizer=torch.optim.AdamW(model.parameters(),lr=a.learning_rate,weight_decay=.0001,fused=True)
        state['validation_before']=tracking_metrics(model,val_tensors);publish('tracking_training')
        if not a.smoke_only:
            folder=a.output/'development_baseline';results=[]
            state.update(phase_episodes_completed=0,phase_episodes_total=18);publish('development_baseline')
            with ProcessPoolExecutor(a.workers,mp_context=mp.get_context('spawn')) as pool:
                jobs=[(g*9+r,str(a.vision),str(a.control),str(folder))
                      for g in range(97000000,97000002) for r in range(9)]
                for future in as_completed([pool.submit(evaluate_episode,j) for j in jobs]):
                    results.append(future.result());state.update(phase_episodes_completed=len(results),partial=summarize(results));publish()
            summary=summarize(results)|dict(label='development_baseline',temporal_tracker=False,
                actor_uses_simulator_state=False,teacher_assisted=False,independent_acceptance=False,
                pairs={str(r):summarize([x for x in results if x['seed']%9==r]) for r in range(9)})
            _atomic_json(folder/'summary.json',dict(summary=summary,results=results));state['evaluations'].append(summary)
            publish('tracking_training')
        best_rank=None
        for step in range(a.updates):
            if time.time()>deadline-180:state['budget_stopped']=True;break
            indices=torch.multinomial(weights,a.batch,replacement=True)
            x,raw,target,mask=[v[indices] for v in train_tensors]
            corrected=raw+model(x)[0]
            error=F.smooth_l1_loss(corrected/.02,target/.02,beta=.1,reduction='none').mean(-1)
            loss=((error*mask).sum(1)/mask.sum(1)).mean()
            movement=F.smooth_l1_loss((corrected[:,1:]-corrected[:,:-1])/.02,
                (target[:,1:]-target[:,:-1])/.02,beta=.1,reduction='none').mean(-1)
            movement_mask=mask[:,1:]&mask[:,:-1]
            loss=loss+.05*((movement*movement_mask).sum(1)/movement_mask.sum(1).clamp_min(1)).mean()
            optimizer.zero_grad(set_to_none=True);loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(),5,error_if_nonfinite=True);optimizer.step()
            for i in indices.tolist():state['sampled_source_routes'][train[i]['source']][train[i]['route']]+=1
            state['step']=step+1
            if (step+1)%50==0:
                state['metrics']=dict(loss=float(loss.detach()),position_training_mean_mm=float(
                    torch.linalg.vector_norm(corrected.detach()-target,dim=-1)[mask].mean()*1000))
                publish('tracking_training')
            if (step+1)%a.evaluate_every==0 or step+1==a.updates:
                state['tracking_validation']=tracking_metrics(model,val_tensors)
                checkpoint=a.output/f'tracker_step_{step+1}.pt'
                torch.save(dict(model={k:v.detach().cpu().clone() for k,v in model.state_dict().items()},
                    vision_sha256=state['vision_sha256'],actor_uses_simulator_state=False,
                    original_ACT_checkpoint=False,tracker_update_interval=4,
                    production_admission=False,export_admission=False,final_vla_acceptance=False),checkpoint)
                state['checkpoint']=str(checkpoint);publish('tracking_validation')
                if a.smoke_only:continue
                if time.time()>deadline-180:state['budget_stopped']=True;break
                folder=a.output/f'development_{step+1}';results=[]
                state.update(phase_episodes_completed=0,phase_episodes_total=18);publish('visual_development')
                torch.cuda.empty_cache()
                with ProcessPoolExecutor(a.workers,mp_context=mp.get_context('spawn')) as pool:
                    jobs=[(g*9+r,str(a.vision),str(a.control),str(folder),str(checkpoint))
                          for g in range(97000000,97000002) for r in range(9)]
                    for future in as_completed([pool.submit(evaluate_episode,j) for j in jobs]):
                        results.append(future.result());state.update(phase_episodes_completed=len(results),partial=summarize(results));publish()
                summary=summarize(results)|dict(label=f'development_{step+1}',
                    actor_uses_simulator_state=False,teacher_assisted=False,independent_acceptance=False,
                    pairs={str(r):summarize([x for x in results if x['seed']%9==r]) for r in range(9)})
                _atomic_json(folder/'summary.json',dict(summary=summary,results=results));state['evaluations'].append(summary)
                rank=(summary['successes'],-summary['hard_failures'],summary['mean_coverage'])
                if best_rank is None or rank>best_rank:
                    best_rank=rank;state.update(best_checkpoint=str(checkpoint),best_development_rank=list(rank))
                publish()
                if summary['successes']>=15:state['development_gate_passed']=True;break
        state['status']='smoke_complete' if a.smoke_only else 'complete_pending_review';publish('finished')
    except BaseException as exc:state.update(status='failed',error=repr(exc));publish();raise


if __name__=='__main__':main()
