"""Causal wrist RGB -> color-keyed spatial state -> learned command policy.

This is a modular visual-language control experiment, not the original ACT
checkpoint. All object coordinates below are training labels or audit outputs;
the rollout actor receives only RGB history, camera calibration, instruction,
reported joints/FK and completed commands. No seed/route is an actor input.
"""
import argparse
from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import multiprocessing as mp
import os
from pathlib import Path
import time

for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ.setdefault(key, '1')
os.environ.setdefault('MUJOCO_GL', 'egl')
os.environ.setdefault('PYOPENGL_PLATFORM', 'egl')

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import ConcatDataset, DataLoader, Dataset

from .candidate_command_contract_v2 import ACTION_CONTRACT
from .constrained_recovery_run40 import summarize
from .multichoice_scene_v1 import scene_contract
from .production_env import BLOCK_COLORS, TARGET_COLORS
from .run34_repeat_eval import deterministic_runtime
from .run42.domain import sample_domain
from .run42.session import DomainSession
from .run64_state_probe import history4
from .sparse_4d_vla_act_v26 import Sparse4DVLAConfigV26
from .train_staged_hybrid_contact_sac import _atomic_json

COLORS = tuple(BLOCK_COLORS)+tuple(TARGET_COLORS)


def language_indices(instruction):
    blocks = [i for i, k in enumerate(BLOCK_COLORS) if BLOCK_COLORS[k][1]+'方块' in instruction]
    goals = [i+4 for i, k in enumerate(TARGET_COLORS) if TARGET_COLORS[k][1]+'目标' in instruction]
    if len(blocks) != 1 or len(goals) != 1:
        raise ValueError('one named block and one named goal required')
    return blocks[0], goals[0]


def history_indices(t):
    if t < 0:
        raise ValueError('negative time')
    return np.array([0, min(20,t), max(0,t-80), max(0,t-40), max(0,t-20),
                     max(0,t-10), max(0,t-4), t], np.int64)


def color_labels(contract, privileged):
    """Training only: full physical object positions, keyed by named color."""
    xy = np.zeros((7,2), np.float32)
    mask = np.zeros(7, bool)
    selected = contract['selected_block']
    others = [i for i in range(3) if i != selected]
    positions = {selected:privileged[6:8], others[0]:privileged[13:15], others[1]:privileged[20:22]}
    for slot, color in enumerate(contract['block_colors']):
        index = COLORS.index(color)
        xy[index] = positions[slot]; mask[index] = True
    for slot, color in enumerate(contract['target_colors']):
        index = COLORS.index(color)
        xy[index] = contract['target_positions'][slot]; mask[index] = True
    return xy, mask


class WristStates(Dataset):
    def __init__(self, records, *, stride=4):
        self.records, self.cache = records, OrderedDict()
        self.index = [(i,t) for i,r in enumerate(records) for t in r['valid_times'][::stride]]
        if not self.index:
            raise ValueError('empty visual dataset')

    def __len__(self):
        return len(self.index)

    def _arrays(self, index):
        if index not in self.cache:
            row = self.records[index]; folder = Path(row['store'])
            arrays = {k:np.load(folder/(k+'.npy'), mmap_mode='r') for k in ('rgb','camera_pose','time','K')}
            with np.load(Path(row['parent_source_path'])/'transitions.npz', allow_pickle=False) as z:
                arrays['truth'] = z['privileged'].copy()
            arrays['contract'] = scene_contract(row['seed'])
            self.cache[index] = arrays
            while len(self.cache) > 24:
                self.cache.popitem(last=False)
        else:
            self.cache.move_to_end(index)
        return self.cache[index]

    def __getitem__(self, index):
        record_index, t = self.index[index]; row = self.records[record_index]
        a = self._arrays(record_index); ids = history_indices(t)
        xy, mask = color_labels(a['contract'], a['truth'][t-row['teacher_start_step']])
        return dict(rgb=np.asarray(a['rgb'][ids]).copy(), pose=np.asarray(a['camera_pose'][ids]).copy(),
            age=np.asarray(a['time'][ids]-a['time'][t],np.float32), K=np.asarray(a['K']).copy(),
            xy=xy, mask=mask, selected=np.asarray(language_indices(row['instruction']),np.int64),time_step=np.int64(t))


class OnlineWristStates(Dataset):
    """Only named collection folders, never development/held-out rollouts."""
    def __init__(self, roots, cache_root):
        self.rows,self.index,self.cache=[],[],OrderedDict()
        cache_root=Path(cache_root);cache_root.mkdir(parents=True,exist_ok=False)
        for root in roots:
            for path in sorted(Path(root).glob('collection_*/episode_*/states.npz')):
                result=json.loads((path.parent/'result.json').read_text())
                if not result.get('collection') or 97000000<=result['seed']//9<99000000:
                    raise ValueError('evaluation data must never enter training')
                dest=cache_root/f"episode_{result['seed']}"
                dest.mkdir(exist_ok=False)
                with np.load(path,allow_pickle=False) as z:
                    n=len(z['frame_steps'])
                    for key in ('x','wrist_rgb','camera_pose','frame_steps','camera_K'):
                        np.save(dest/(key+'.npy'),z[key],allow_pickle=False)
                _atomic_json(dest/'source.json',dict(source=str(path),seed=result['seed'],training_only=True))
                index=len(self.rows);self.rows.append((dest,result['seed']))
                self.index.extend((index,i) for i in range(n))
        if not self.index:raise ValueError('empty online collection input')

    def __len__(self):return len(self.index)

    def __getitem__(self,index):
        row_id,k=self.index[index];path,seed=self.rows[row_id]
        if row_id not in self.cache:
            self.cache[row_id]={key:np.load(path/(key+'.npy'),mmap_mode='r')
                                for key in ('x','wrist_rgb','camera_pose','frame_steps','camera_K')}
            while len(self.cache)>24:self.cache.popitem(last=False)
        else:self.cache.move_to_end(row_id)
        a=self.cache[row_id];t=int(a['frame_steps'][k]);ids=history_indices(t)
        positions=np.maximum(0,np.searchsorted(a['frame_steps'],ids,side='right')-1)
        actual_times=a['frame_steps'][positions]
        c=scene_contract(seed);selection=language_indices(c['instruction'])
        xy=np.zeros((7,2),np.float32);mask=np.zeros(7,bool)
        xy[selection[0]]=a['x'][t,108:110]+a['x'][t,12:14];mask[selection[0]]=True
        for slot,color in enumerate(c['target_colors']):
            i=COLORS.index(color);xy[i]=c['target_positions'][slot];mask[i]=True
        return dict(rgb=a['wrist_rgb'][positions].copy(),pose=a['camera_pose'][positions].copy(),
            age=np.asarray((actual_times-t)/30,np.float32),K=a['camera_K'].copy(),xy=xy,mask=mask,
            selected=np.asarray(selection,np.int64),time_step=np.int64(t))


def ray_geometry(pose, K, age, height, width):
    """Deployable FK/intrinsics only. Two known tabletop-height ray planes."""
    b,t,_ = pose.shape
    yy,xx = torch.meshgrid(torch.arange(height,device=pose.device),torch.arange(width,device=pose.device),indexing='ij')
    u=(xx.flatten().float()+.5)*(160/width)-.5
    v=(yy.flatten().float()+.5)*(120/height)-.5
    cx,cy = K[:,0,2],K[:,1,2]
    fx,fy = K[:,0,0],K[:,1,1]
    rays = torch.stack(((u[None]-cx[:,None])/fx[:,None],
                        -(v[None]-cy[:,None])/fy[:,None],-torch.ones((b,len(u)),device=pose.device)),dim=-1)
    direction = torch.einsum('btij,bnj->btni',pose[...,3:].reshape(b,t,3,3),rays)
    origin = pose[...,:3,None].transpose(-1,-2)
    pieces=[]
    for z in (.026,.051):
        dz=direction[...,2:3]
        distance=(z-origin[...,2:3])/torch.where(dz.abs()>.001,dz,torch.ones_like(dz)*.001)
        valid=(distance>0)&(distance<2)
        point=origin+distance*direction
        pieces.extend((point[...,:2].clamp(-1,1),valid.float()))
    pieces += [origin.expand(-1,-1,len(u),-1),age[:,:,None,None].expand(-1,-1,len(u),1)/30,
               torch.stack((u/160,v/120),-1)[None,None].expand(b,t,-1,-1)]
    return torch.cat(pieces,-1)


class VisualState(nn.Module):
    def __init__(self, width=128, recent_block_seconds=None):
        super().__init__()
        if recent_block_seconds is not None and not .1 <= recent_block_seconds <= 1.:
            raise ValueError('bounded causal moving-object memory window required')
        self.recent_block_seconds=recent_block_seconds
        self.encoder=nn.Sequential(nn.Conv2d(3,32,5,2,2),nn.GroupNorm(4,32),nn.SiLU(),
            nn.Conv2d(32,64,3,2,1),nn.GroupNorm(8,64),nn.SiLU(),
            nn.Conv2d(64,width,3,2,1),nn.GroupNorm(8,width),nn.SiLU())
        self.geometry=nn.Sequential(nn.Linear(12,width),nn.SiLU(),nn.Linear(width,width))
        self.queries=nn.Parameter(torch.randn(7,width)*.02)
        layer=nn.TransformerDecoderLayer(width,4,width*3,dropout=0.,batch_first=True,norm_first=True)
        self.decoder=nn.TransformerDecoder(layer,2,norm=nn.LayerNorm(width))
        self.readout=nn.Sequential(nn.Linear(width,width),nn.SiLU(),nn.Linear(width,2))
        self.register_buffer('center',torch.tensor([.3,0.]))
        self.register_buffer('extent',torch.tensor([.25,.25]))

    def forward(self,rgb,pose,K,age,return_features=False):
        b,t,h,w,c=rgb.shape
        encoded=self.encoder(rgb.permute(0,1,4,2,3).reshape(b*t,c,h,w).float()/255)
        hh,ww=encoded.shape[-2:]
        visual=encoded.flatten(2).transpose(1,2).reshape(b,t,hh*ww,-1)
        geometry=ray_geometry(pose.float(),K.float(),age.float(),hh,ww)
        memory=(visual+self.geometry(geometry)).flatten(1,2)
        memory_mask=target_mask=None
        if self.recent_block_seconds is not None:
            stale=age < -self.recent_block_seconds
            if torch.any(stale.all(-1)):raise ValueError('moving object requires a recent observation')
            memory_mask=torch.zeros((b,7,t,hh*ww),device=pose.device,dtype=torch.bool)
            memory_mask[:,:4]=stale[:,None,:,None]
            memory_mask=memory_mask.flatten(2).repeat_interleave(4,dim=0)
            # Prevent an indirect old-frame path through static-goal queries.
            target_mask=torch.zeros((7,7),device=pose.device,dtype=torch.bool)
            target_mask[:4,4:]=True
        hidden=self.decoder(self.queries[None].expand(b,-1,-1),memory,
                            memory_mask=memory_mask,tgt_mask=target_mask)
        # World-coordinate output is FP32 even under BF16 visual training.
        with torch.autocast(device_type=rgb.device.type,enabled=False):
            world=self.center+self.extent*torch.tanh(self.readout(hidden.float()))
        return (world,hidden.float()) if return_features else world


def actor_input(reported,tool,q_history,commands,previous_target,initial,world_xy,selection):
    block,goal=world_xy[list(selection)]
    return np.r_[reported,tool,history4(q_history,12),history4(commands,6),previous_target,
                 block-tool[:2],goal-tool[:2],initial].astype(np.float32)


def visual_batch(batch,device='cuda'):
    return {k:v.to(device,non_blocking=True) for k,v in batch.items()}


def validate(model,loader,limit=24):
    errors,selected,early=[] ,[],[]
    visual_dependence={}
    model.eval()
    with torch.inference_mode():
        for i,batch in enumerate(loader):
            b=visual_batch(batch)
            prediction=model(b['rgb'],b['pose'],b['K'],b['age'])
            error=torch.linalg.vector_norm(prediction-b['xy'],dim=-1)
            errors.append(error[b['mask']].cpu())
            selected_error=error.gather(1,b['selected']).cpu()
            selected.append(selected_error)
            early.append(selected_error[(b['time_step']<30).cpu()])
            if i==0:
                blank=model(torch.zeros_like(b['rgb']),b['pose'],b['K'],b['age'])
                blank_error=torch.linalg.vector_norm(blank-b['xy'],dim=-1)
                visual_dependence=dict(real_image_mean_mm=float(error[b['mask']].mean()*1000),
                    blank_image_mean_mm=float(blank_error[b['mask']].mean()*1000),samples=len(error))
            if i+1>=limit:break
    model.train()
    errors=torch.cat(errors); selected=torch.cat(selected);early=torch.cat(early)
    return dict(position_mean_mm=float(errors.mean()*1000),position_p90_mm=float(errors.quantile(.9)*1000),
                selected_block_mean_mm=float(selected[:,0].mean()*1000),
                selected_goal_mean_mm=float(selected[:,1].mean()*1000),samples=len(selected),
                early_block_mean_mm=float(early[:,0].mean()*1000) if len(early) else None,
                early_goal_mean_mm=float(early[:,1].mean()*1000) if len(early) else None,
                early_samples=len(early),visual_dependence_audit=visual_dependence)


def evaluate_episode(job):
    seed,vision_path,control_path,output=job[:4]
    tracker_path=job[4] if len(job)>4 else None
    options=job[5] if len(job)>5 else {}
    if set(options)-{'fixed_wrist_survey','retain_overview'}:raise ValueError('unknown evaluation options')
    if options.get('retain_overview') and not options.get('fixed_wrist_survey'):
        raise ValueError('overview anchors require physical survey')
    deterministic_runtime(); torch.set_num_threads(1)
    visual_saved=torch.load(vision_path,map_location='cpu',weights_only=True)
    visual=VisualState(recent_block_seconds=visual_saved.get('recent_block_seconds')).cuda().eval()
    visual.load_state_dict(visual_saved['model'])
    from .run73_anchored_specialist import load_visual_control
    control=load_visual_control(control_path)
    tracker=None
    if tracker_path is not None:
        from .run71_temporal_tracking import load_tracker, tracker_features
        tracker=load_tracker(tracker_path,vision_path)
    tracker_hidden=None;tracker_residual=np.zeros(2,np.float32)
    cfg=Sparse4DVLAConfigV26(language_max_tokens=128,visual_memory_mode='episode_anchors_v54')
    session=DomainSession(cfg,ACTION_CONTRACT,seed,sample_domain(seed+6001,0))
    folder=Path(output)/f'episode_{seed}';folder.mkdir(parents=True,exist_ok=False)
    selection=language_indices(session.buffer.instruction)
    q_history,commands,positions,errors,frames=[],[],[],[],[]
    maximum_coverage=maximum_hold=0.;contacts=rewrites=0
    started=time.time();use_specialist=None;specialist_probability=None
    try:
        session.observe(); initial=session.reported[:6].copy()
        if session.reward_state()[1]!=0:raise ValueError('initial overlap')
        survey_audit=None
        if options.get('fixed_wrist_survey'):
            from .run74_observation_probe import observation_prefix
            survey_audit=observation_prefix(session,initial,q_history,commands,frames)
        policy_start=len(commands)
        for step in range(policy_start,900):
            if session.end_kind!='sampler_cut':break
            session.observe(); rows=session.buffer.rows
            if options.get('retain_overview'):
                from .run74_observation_probe import observation_history_indices
                ids=observation_history_indices(step,True)
            else:ids=history_indices(step)
            rgb=np.stack([rows[int(i)]['rgb'] for i in ids])
            pose=np.stack([rows[int(i)]['camera_pose'] for i in ids])
            age=np.asarray([rows[int(i)]['time']-rows[-1]['time'] for i in ids],np.float32)
            with torch.inference_mode():
                visual_output=visual(torch.from_numpy(rgb)[None].cuda(),torch.from_numpy(pose)[None].cuda(),
                    torch.from_numpy(session.buffer.K)[None].cuda(),torch.from_numpy(age)[None].cuda(),
                    return_features=tracker is not None)
                world=(visual_output[0] if tracker is not None else visual_output)[0].cpu().numpy()
                previous=rows[-2]['applied_target'] if step else initial
                x=actor_input(session.reported,rows[-1]['tool'],q_history,commands,previous,initial,world,selection)
                raw_x=torch.from_numpy(x.copy())[None].cuda()
                if getattr(control,'geometry_gated',False):
                    chosen,probability=control.routing(raw_x)
                    use_specialist=bool(chosen.item());specialist_probability=float(probability.item())
                    if use_specialist and tracker is None:raise ValueError('specialist requires its spatial tracker')
                if tracker is not None and use_specialist is not False:
                    if step%4==0:
                        features=tracker_features(torch.from_numpy(x)[None].cuda(),visual_output[1],
                            torch.tensor([selection],device='cuda'))
                        correction,tracker_hidden=tracker(features[:,None],tracker_hidden)
                        tracker_residual=correction[0,0].cpu().numpy()
                    world[selection[0]]+=tracker_residual
                    x=actor_input(session.reported,rows[-1]['tool'],q_history,commands,previous,initial,world,selection)
                control_input=torch.from_numpy(x)[None].cuda()
                action=(control(control_input,raw_x) if getattr(control,'geometry_gated',False)
                        else control(control_input))[0].cpu().numpy().clip(-1,1)
            # Audit only, AFTER the action is fixed. These values are never fed back.
            truth=np.stack((session.env.block_xy(),session.env.target_xy))
            errors.append(np.linalg.norm(world[list(selection)]-truth,axis=1))
            positions.append(world);q_history.append(session.reported.copy());commands.append(action.copy())
            if step%8==0:frames.append(rows[-1]['rgb'].copy())
            result=session.advance(action);metrics=session.reward_state()
            maximum_coverage=max(maximum_coverage,metrics[1]);maximum_hold=max(maximum_hold,metrics[2])
            contacts+=int(result.get('valid_contact',False));rewrites+=int(result.get('safety_rewrite',False))
            if result['kind']!='sampler_cut':break
        kind=session.end_kind if session.end_kind!='sampler_cut' else 'finite_timeout'
        result=dict(seed=seed,safe_success=kind=='success',end_kind=kind,terminal_reason=session.reason,
            steps=len(commands),seconds=time.time()-started,maximum_coverage=float(maximum_coverage),
            maximum_hold_s=float(maximum_hold),valid_contact_steps=contacts,safety_rewrite_steps=rewrites,
            mean_block_localization_mm=float(np.mean(np.asarray(errors)[:,0])*1000) if errors else None,
            mean_goal_localization_mm=float(np.mean(np.asarray(errors)[:,1])*1000) if errors else None,
            actor_uses_simulator_state=False,teacher_assisted=False,visual_grounding=True,
            instruction=session.buffer.instruction,language_parser='seven_named_colors',
            original_ACT_checkpoint=False,independent_acceptance=False,initial_coverage=0.,
            temporal_tracker=tracker_path is not None,
            geometry_gated=getattr(control,'geometry_gated',False),
            learned_specialist_selected=use_specialist,specialist_probability=specialist_probability,
            fixed_wrist_survey=bool(options.get('fixed_wrist_survey')),survey_audit=survey_audit,
            retain_overview=bool(options.get('retain_overview')),policy_start_step=policy_start,
            recent_block_seconds=visual.recent_block_seconds,
            production_admission=False,export_admission=False,final_vla_acceptance=False)
        np.savez_compressed(folder/'trace.npz',reported=np.asarray(q_history),command=np.asarray(commands),
            predicted_world_xy=np.asarray(positions),audit_position_errors=np.asarray(errors),wrist_rgb=np.asarray(frames),
            audit_policy_steps=np.arange(policy_start,policy_start+len(errors)))
        _atomic_json(folder/'result.json',result)
        return result
    finally:session.close()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',type=Path,required=True);p.add_argument('--control',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--updates',type=int,default=6000)
    p.add_argument('--batch',type=int,default=48);p.add_argument('--loader-workers',type=int,default=6)
    p.add_argument('--eval-workers',type=int,default=6);p.add_argument('--wall-seconds',type=int,default=3500)
    p.add_argument('--evaluate-every',type=int,default=2000)
    p.add_argument('--online-root',type=Path,action='append',default=[])
    a=p.parse_args()
    if not 100<=a.updates<=12000 or not 4<=a.batch<=128 or not 1<=a.eval_workers<=9:
        raise ValueError('bounded visual pilot required')
    a.output.mkdir(parents=True,exist_ok=False);deterministic_runtime();torch.set_num_threads(2)
    rows=[r for r in json.loads(a.manifest.read_text())['records'] if r['split']=='train' and r['run54_pool']=='recovery']
    groups=sorted({r['seed']//9 for r in rows});validation_groups=set(groups[::5])
    train_rows=[r for r in rows if r['seed']//9 not in validation_groups]
    val_rows=[r for r in rows if r['seed']//9 in validation_groups]
    train=WristStates(train_rows); val=WristStates(val_rows,stride=12)
    online=OnlineWristStates(a.online_root,a.output/'online_cache') if a.online_root else None
    if online is not None:train=ConcatDataset([train,online])
    loader=DataLoader(train,batch_size=a.batch,shuffle=True,num_workers=a.loader_workers,
        pin_memory=True,persistent_workers=a.loader_workers>0,drop_last=True)
    val_loader=DataLoader(val,batch_size=a.batch,shuffle=False,num_workers=2,pin_memory=True,persistent_workers=True)
    model=VisualState().cuda();optimizer=torch.optim.AdamW(model.parameters(),lr=.0003,weight_decay=.0001,fused=True)
    state=dict(run='Run67',status='running',phase='visual_spatial_training',started=time.time(),step=0,
        total_updates=a.updates,batch=a.batch,training_episodes=len(train_rows),validation_episodes=len(val_rows),
        online_training_episodes=len(online.rows) if online is not None else 0,
        training_frames=len(train),validation_frames=len(val),
        validation_groups=sorted(validation_groups),control_checkpoint=str(a.control),evaluations=[],
        parameters=sum(p.numel() for p in model.parameters()),
        visual_grounding=True,actor_uses_simulator_state=False,original_ACT_checkpoint=False,
        training_kind='color_keyed_causal_wrist_spatial_supervision',target_rate=.8,
        independent_acceptance=False,production_admission=False,export_admission=False,final_vla_acceptance=False)
    def publish(phase=None):
        if phase:state['phase']=phase
        state.update(updated=time.time(),elapsed_seconds=time.time()-state['started'])
        _atomic_json(a.output/'run_state.json',state)
    publish();iterator=iter(loader)
    try:
        for step in range(a.updates):
            if time.time()>state['started']+a.wall_seconds-180:
                state['budget_stopped']=True;break
            try:batch=next(iterator)
            except StopIteration:iterator=iter(loader);batch=next(iterator)
            b=visual_batch(batch)
            with torch.autocast('cuda',dtype=torch.bfloat16):
                prediction=model(b['rgb'],b['pose'],b['K'],b['age'])
                loss=F.smooth_l1_loss(prediction[b['mask']]/.02,b['xy'][b['mask']]/.02,beta=.05)
            optimizer.zero_grad(set_to_none=True);loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(),5,error_if_nonfinite=True);optimizer.step()
            for g in optimizer.param_groups:g['lr']=.00002+.00028*.5*(1+np.cos(np.pi*(step+1)/a.updates))
            state['step']=step+1
            if (step+1)%50==0:
                state['metrics']=dict(loss=float(loss.detach()),position_training_mean_mm=float(
                    torch.linalg.vector_norm(prediction.detach()-b['xy'],dim=-1)[b['mask']].mean()*1000))
                state['samples_per_second']=a.batch*(step+1)/max(time.time()-state['started'],1.)
                publish('visual_spatial_training')
            if (step+1)%a.evaluate_every==0 or step+1==a.updates:
                state['visual_validation']=validate(model,val_loader)
                checkpoint=a.output/f'visual_step_{step+1}.pt'
                torch.save(dict(model={k:v.detach().cpu().clone() for k,v in model.state_dict().items()},
                    actor_uses_simulator_state=False,original_ACT_checkpoint=False,
                    production_admission=False,export_admission=False,final_vla_acceptance=False),checkpoint)
                publish('visual_validation')
                if state['visual_validation']['selected_block_mean_mm']<25 and state['visual_validation']['selected_goal_mean_mm']<25:
                    torch.cuda.empty_cache()
                    results=[];folder=a.output/f'development_{step+1}'
                    state.update(phase_episodes_completed=0,phase_episodes_total=18);publish('visual_development')
                    with ProcessPoolExecutor(a.eval_workers,mp_context=mp.get_context('spawn')) as pool:
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
        state['status']='complete_pending_review';publish('finished')
    except BaseException as exc:
        state.update(status='failed',error=repr(exc));publish();raise


if __name__=='__main__':main()
