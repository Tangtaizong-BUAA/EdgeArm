"""Adapt a learned controller to estimated, never oracle, spatial inputs.

Frozen wrist perception is used both when recoding training demonstrations and
during physical online rollouts. Teachers are queried only in collection, and
independent evaluation is deliberately not implemented in this pilot module.
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
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset

from .candidate_command_contract_v2 import ACTION_CONTRACT
from .constrained_recovery_run40 import summarize
from .run34_repeat_eval import deterministic_runtime
from .run42.domain import sample_domain
from .run42.session import DomainSession
from .run53_recovery import RecoveryTeacher
from .run63_control_probe import TinyTarget
from .run64_state_probe import history4
from .run67_visual_state import (OnlineWristStates, VisualState, WristStates, actor_input,
    evaluate_episode, history_indices, language_indices, visual_batch)
from .sparse_4d_vla_act_v26 import Sparse4DVLAConfigV26
from .train_staged_hybrid_contact_sac import _atomic_json


def estimated_inputs(proprio, world, selection):
    """The 114 proprioceptive values contain no object/goal truth."""
    if proprio.shape[-1]!=114 or world.shape[-2:]!=(7,2):
        raise ValueError('fixed deployable input schema required')
    chosen=world.gather(1,selection[:,:,None].expand(-1,-1,2))
    relative=(chosen-proprio[:,None,12:14]).flatten(1)
    return torch.cat((proprio[:,:108],relative,proprio[:,108:]),1)


class CommandImages(Dataset):
    def __init__(self,images):self.images,self.cache=images,OrderedDict()
    def __len__(self):return len(self.images)

    def __getitem__(self,i):
        b=self.images[i];row_id,t=self.images.index[i]
        if isinstance(self.images,WristStates):
            row=self.images.records[row_id]
            if row_id not in self.cache:
                self.cache[row_id]={k:np.load(Path(row['store'])/(k+'.npy'),mmap_mode='r')
                                   for k in ('joint','tool','command','applied_target')}
                while len(self.cache)>24:self.cache.popitem(last=False)
            z=self.cache[row_id];q=z['joint'];commands=z['command'];seed=row['seed']
            previous=z['applied_target'][t-1] if t else q[0,:6]
            proprio=np.r_[q[t],z['tool'][t],history4(q[max(0,t-4):t],12),
                history4(commands[max(0,t-4):t],6),previous,q[0,:6]].astype(np.float32)
            label=commands[t].copy()
        else:
            path,seed=self.images.rows[row_id];z=self.images.cache[row_id]
            t=int(z['frame_steps'][t])
            if row_id not in self.cache:
                source=json.loads((path/'source.json').read_text())['source']
                with np.load(source,allow_pickle=False) as original:
                    self.cache[row_id]={'teacher_action':original['teacher_action'].copy()}
                    if 'x' not in z:self.cache[row_id]['x']=original['x'].copy()
            x=z['x'][t] if 'x' in z else self.cache[row_id]['x'][t]
            proprio=np.r_[x[:108],x[112:118]].astype(np.float32)
            label=self.cache[row_id]['teacher_action'][t].copy()
        b.update(proprio=proprio,command=label,route=np.int64(seed%9))
        return b


def recode(a,output,state,publish):
    if getattr(a,'tracker',None):
        from .run72_tracker_control import recode_with_tracker
        return recode_with_tracker(a,output,state,publish)
    rows=[r for r in json.loads(a.manifest.read_text())['records']
          if r['split']=='train' and r['run54_pool']=='recovery']
    # Preserve Run67's visual-validation groups even during action adaptation.
    validation_groups=set(sorted({r['seed']//9 for r in rows})[::5])
    rows=[r for r in rows if r['seed']//9 not in validation_groups]
    datasets=[CommandImages(WristStates(rows))]
    if a.online_root:
        datasets.append(CommandImages(OnlineWristStates(a.online_root,output/'online_cache')))
    if getattr(a,'correction_root',None):
        from .run69_visual_recovery import CorrectionWristStates
        corrected=CorrectionWristStates(a.correction_root,output/'correction_cache')
        validation=set(sorted({seed//9 for _,seed in corrected.rows})[-2:])
        indices=[i for i,(row,t) in enumerate(corrected.index) if corrected.rows[row][1]//9 not in validation]
        datasets.append(Subset(CommandImages(corrected),indices))
        state.update(correction_recode_frames=len(indices),correction_validation_groups=sorted(validation))
    dataset=ConcatDataset(datasets)
    loader=DataLoader(dataset,batch_size=96,num_workers=6,pin_memory=True,shuffle=False)
    visual=VisualState().cuda().eval()
    visual.load_state_dict(torch.load(a.vision,map_location='cpu',weights_only=True)['model'])
    xs,ys,routes=[],[],[]
    with torch.inference_mode():
        for i,batch in enumerate(loader):
            b=visual_batch(batch)
            world=visual(b['rgb'],b['pose'],b['K'],b['age'])
            xs.append(estimated_inputs(b['proprio'],world,b['selected']).cpu().numpy())
            ys.append(batch['command'].numpy());routes.append(batch['route'].numpy())
            if i%20==0:
                state.update(recode_completed=sum(len(x) for x in xs),recode_total=len(dataset));publish('recode_perceived_inputs')
    result=tuple(np.concatenate(x) for x in (xs,ys,routes))
    np.savez_compressed(output/'recoded_commands.npz',x=result[0],action=result[1],route=result[2])
    del visual;torch.cuda.empty_cache()
    return result


def collect_episode(job):
    seed,vision_path,control_path,output,beta=job[:5]
    tracker_path=job[5] if len(job)>5 else None
    if not 99700000<=seed//9<100000000 or not 0<=beta<=1:
        raise ValueError('training-only collection seeds and bounded teacher mixing')
    deterministic_runtime();torch.set_num_threads(1)
    visual=VisualState().cuda().eval();visual.load_state_dict(torch.load(vision_path,map_location='cpu',weights_only=True)['model'])
    control=TinyTarget(118,512).cuda().eval();control.load_state_dict(torch.load(control_path,map_location='cpu',weights_only=True)['model'])
    tracker=None
    if tracker_path is not None:
        from .run71_temporal_tracking import load_tracker,tracker_features
        tracker=load_tracker(tracker_path,vision_path)
    tracker_hidden=None;tracker_residual=np.zeros(2,np.float32)
    cfg=Sparse4DVLAConfigV26(language_max_tokens=128,visual_memory_mode='episode_anchors_v54')
    session=DomainSession(cfg,ACTION_CONTRACT,seed,sample_domain(seed+6001,0))
    teacher=RecoveryTeacher(session,seed%9,goal_retreat_coverage=.995)
    selection=language_indices(session.buffer.instruction);rng=np.random.default_rng(seed+6801)
    folder=Path(output)/f'episode_{seed}';folder.mkdir(parents=True,exist_ok=False)
    q_history,commands,xs,labels,teacher_flags=[],[],[],[],[]
    frames,frame_steps,cameras,visual_labels,visual_masks=[],[],[],[],[]
    coverage=hold=0.;contacts=rewrites=0;started=time.time()
    try:
        session.observe();initial=session.reported[:6].copy()
        if session.reward_state()[1]!=0:raise ValueError('initial overlap')
        for step in range(900):
            session.observe();rows=session.buffer.rows;ids=history_indices(step)
            rgb=np.stack([rows[int(i)]['rgb'] for i in ids]);pose=np.stack([rows[int(i)]['camera_pose'] for i in ids])
            age=np.asarray([rows[int(i)]['time']-rows[-1]['time'] for i in ids],np.float32)
            with torch.inference_mode():
                visual_output=visual(torch.from_numpy(rgb)[None].cuda(),torch.from_numpy(pose)[None].cuda(),
                    torch.from_numpy(session.buffer.K)[None].cuda(),torch.from_numpy(age)[None].cuda(),
                    return_features=tracker is not None)
                world=(visual_output[0] if tracker is not None else visual_output)[0].cpu().numpy()
                previous=rows[-2]['applied_target'] if step else initial
                x=actor_input(session.reported,rows[-1]['tool'],q_history,commands,previous,initial,world,selection)
                if tracker is not None:
                    if step%4==0:
                        features=tracker_features(torch.from_numpy(x)[None].cuda(),visual_output[1],
                            torch.tensor([selection],device='cuda'))
                        correction,tracker_hidden=tracker(features[:,None],tracker_hidden)
                        tracker_residual=correction[0,0].cpu().numpy()
                    world[selection[0]]+=tracker_residual
                    x=actor_input(session.reported,rows[-1]['tool'],q_history,commands,previous,initial,world,selection)
                prediction=control(torch.from_numpy(x)[None].cuda())[0].cpu().numpy().clip(-1,1)
            # Privileged teacher query is separate from the already-fixed student prediction.
            label=teacher.command()
            if step%15==0:use_teacher=rng.random()<beta
            action=label.copy() if use_teacher else prediction.copy()
            xs.append(x);labels.append(label);teacher_flags.append(use_teacher)
            q_history.append(session.reported.copy());commands.append(action.copy())
            if step%4==0:
                frames.append(rows[-1]['rgb'].copy());frame_steps.append(step);cameras.append(rows[-1]['camera_pose'].copy())
                # Side-channel training labels, recorded after prediction only.
                # They are never appended to the actor input or its history.
                truth=np.zeros((7,2),np.float32);mask=np.zeros(7,bool)
                truth[selection[0]]=session.env.block_xy();truth[selection[1]]=session.env.target_xy
                mask[list(selection)]=True;visual_labels.append(truth);visual_masks.append(mask)
            result=session.advance(action);metrics=session.reward_state()
            coverage=max(coverage,metrics[1]);hold=max(hold,metrics[2])
            contacts+=int(result.get('valid_contact',False));rewrites+=int(result.get('safety_rewrite',False))
            if result['kind']!='sampler_cut':break
        kind=session.end_kind if session.end_kind!='sampler_cut' else 'finite_timeout'
        np.savez_compressed(folder/'perceived_commands.npz',x=np.asarray(xs,np.float32),
            teacher_action=np.asarray(labels,np.float32),executed_action=np.asarray(commands,np.float32),
            teacher_executed=np.asarray(teacher_flags,bool),wrist_rgb=np.asarray(frames,np.uint8),frame_steps=np.asarray(frame_steps),
            camera_pose=np.asarray(cameras,np.float32),camera_K=np.asarray(session.buffer.K,np.float32),
            visual_labels_xy=np.asarray(visual_labels,np.float32),visual_label_mask=np.asarray(visual_masks,bool),
            selected=np.asarray(selection,np.int64))
        result=dict(seed=seed,safe_success=kind=='success',end_kind=kind,terminal_reason=session.reason,
            steps=step+1,seconds=time.time()-started,maximum_coverage=float(coverage),maximum_hold_s=float(hold),
            valid_contact_steps=contacts,safety_rewrite_steps=rewrites,initial_coverage=0.,
            teacher_assisted=bool(any(teacher_flags)),teacher_execution_fraction=float(np.mean(teacher_flags)),
            collection=True,actor_uses_simulator_state=False,teacher_uses_simulator_state=True,
            temporal_tracker=tracker_path is not None,
            visual_grounding=True,independent_acceptance=False,production_admission=False,
            export_admission=False,final_vla_acceptance=False,data=str(folder/'perceived_commands.npz'))
        _atomic_json(folder/'result.json',result);return result
    finally:session.close()


def seed_schedule(round_index,group_start=99700000):
    if not 0<=round_index<4 or group_start not in (99700000,99800000,99900000):
        raise ValueError('bounded training rounds and dedicated groups required')
    return [g*9+r for g in range(group_start+2*round_index,group_start+2+2*round_index) for r in range(9)]


def development_rank(results):
    summary=summarize(results)
    return (summary['successes'],-summary['hard_failures'],summary['mean_coverage'],summary['mean_hold_s'])


def fit(model,old,online,a,state,publish,deadline):
    sources=[old]+([online] if online is not None else [])
    tensors=[]
    for x,y,routes in sources:
        if x.shape!=(len(y),118) or y.shape[1]!=6 or not np.isfinite(x).all() or not np.isfinite(y).all():
            raise ValueError('finite perceived state / action labels required')
        pools=[torch.from_numpy(np.flatnonzero(routes==r)).cuda() for r in range(9)]
        if any(len(pool)==0 for pool in pools):raise ValueError('all nine routes required in each source')
        tensors.append((torch.from_numpy(x).cuda(),torch.from_numpy(y).cuda(),pools))
    optimizer=torch.optim.AdamW(model.parameters(),lr=a.learning_rate,weight_decay=1e-6,fused=True)
    model.train()
    for i in range(a.updates):
        if time.time()>deadline-180:raise TimeoutError('visual adaptation budget reached')
        loss=0.
        for x,y,pools in tensors:
            ids=torch.cat([pool[torch.randint(len(pool),(114,),device='cuda')] for pool in pools])
            loss=loss+((model(x[ids])-y[ids])/model.yscale).square().mean()/len(tensors)
        optimizer.zero_grad(set_to_none=True);loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(),5,error_if_nonfinite=True);optimizer.step()
        for group in optimizer.param_groups:
            group['lr']=.00002+(a.learning_rate-.00002)*.5*(1+np.cos(np.pi*(i+1)/a.updates))
        state['step']+=1
        if (i+1)%100==0:
            with torch.inference_mode():
                errors=[float((model(x[::8])-y[::8]).abs().mean()) for x,y,_ in tensors]
            state['metrics']=dict(loss=float(loss.detach()),recoded_command_mae=errors[0],
                                  online_command_mae=errors[-1] if online is not None else None)
            publish('visual_online_supervised_fit' if online is not None else 'perceived_input_fit')
    checkpoint=a.output/f"round_{state['round']}.pt"
    torch.save(dict(model={k:v.detach().cpu().clone() for k,v in model.state_dict().items()},
        input_dim=118,width=512,vision_checkpoint=str(a.vision),vision_sha256=state['vision_sha256'],
        tracker_sha256=state.get('tracker_sha256'),
        actor_uses_simulator_state=False,visual_grounding=True,original_ACT_checkpoint=False,
        training_kind='perceived_input_DAgger_supervision_not_RL',
        production_admission=False,export_admission=False,final_vla_acceptance=False),checkpoint)
    model.eval();return checkpoint


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',type=Path,required=True);p.add_argument('--vision',type=Path,required=True)
    p.add_argument('--control',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--online-root',type=Path,action='append',default=[])
    p.add_argument('--correction-root',type=Path)
    p.add_argument('--tracker',type=Path);p.add_argument('--encoded-sequences',type=Path)
    p.add_argument('--tracked-command-roots',type=Path,nargs='+')
    p.add_argument('--workers',type=int,default=6);p.add_argument('--rounds',type=int,default=3)
    p.add_argument('--updates',type=int,default=3000);p.add_argument('--wall-seconds',type=int,default=3500)
    p.add_argument('--learning-rate',type=float,default=.00015)
    p.add_argument('--run-label',choices=('Run68','Run70','Run72'),default='Run68')
    p.add_argument('--group-start',type=int,choices=(99700000,99800000,99900000),default=99700000)
    p.add_argument('--select-best-after-each-round',action='store_true')
    p.add_argument('--teacher-betas',type=float,nargs=4,default=(.75,.5,.25,0.))
    p.add_argument('--smoke-only',action='store_true')
    a=p.parse_args()
    if not 1<=a.workers<=9 or not 0<=a.rounds<=4 or not 100<=a.updates<=6000:
        raise ValueError('bounded visual adaptation required')
    if not 300<=a.wall_seconds<=3500 or not .00002<=a.learning_rate<=.0003:
        raise ValueError('finite time and learning-rate bounds required')
    if any(not 0<=x<=1 for x in a.teacher_betas):raise ValueError('bounded teacher execution probabilities')
    if any((a.tracker,a.encoded_sequences,a.tracked_command_roots)) and not all((a.tracker,a.encoded_sequences,a.tracked_command_roots)):
        raise ValueError('tracker, causal encoded sequences and training command roots are required together')
    if a.tracker:
        import os
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    a.output.mkdir(parents=True,exist_ok=False);deterministic_runtime();torch.set_num_threads(2)
    state=dict(run=a.run_label,status='running',phase='initializing',started=time.time(),step=0,round=0,
        rounds=a.rounds,total_updates=a.updates*(1+a.rounds),evaluations=[],collections=[],
        vision_checkpoint=str(a.vision),control_checkpoint=str(a.control),
        vision_sha256=hashlib.sha256(a.vision.read_bytes()).hexdigest(),
        tracker_sha256=hashlib.sha256(a.tracker.read_bytes()).hexdigest() if a.tracker else None,
        temporal_tracker=a.tracker is not None,
        actor_uses_simulator_state=False,visual_grounding=True,original_ACT_checkpoint=False,
        training_kind='perceived_input_DAgger_supervision_not_RL',target_rate=.8,
        start_stage='CONTACT_TRANSPORT_HOLD',language_parser='seven_named_colors',
        teacher_uses_simulator_state=True,independent_acceptance=False,
        select_best_after_each_round=a.select_best_after_each_round,selection_history=[],
        teacher_betas=list(a.teacher_betas),collection_group_start=a.group_start,
        production_admission=False,export_admission=False,final_vla_acceptance=False)
    def publish(phase=None):
        if phase:state['phase']=phase
        state.update(updated=time.time(),elapsed_seconds=time.time()-state['started'])
        _atomic_json(a.output/'run_state.json',state)

    def rollout(checkpoint,seeds,name,beta=None):
        if time.time()>deadline-180:raise TimeoutError('insufficient time for bounded rollout')
        collect=beta is not None;results=[]
        state.update(phase_episodes_total=len(seeds),phase_episodes_completed=0);publish(name)
        torch.cuda.empty_cache()
        with ProcessPoolExecutor(a.workers,mp_context=mp.get_context('spawn')) as pool:
            suffix=(str(a.tracker),) if a.tracker else ()
            if collect:
                futures=[pool.submit(collect_episode,(s,str(a.vision),str(checkpoint),str(a.output/name),beta)+suffix) for s in seeds]
            else:
                futures=[pool.submit(evaluate_episode,(s,str(a.vision),str(checkpoint),str(a.output/name))+suffix) for s in seeds]
            for future in as_completed(futures):
                results.append(future.result());state.update(phase_episodes_completed=len(results),partial=summarize(results));publish()
        summary=summarize(results)|dict(label=name,round=state['round'],teacher_assisted=any(r.get('teacher_assisted',False) for r in results),
            visual_grounding=True,actor_uses_simulator_state=False,independent_acceptance=False,
            pairs={str(r):summarize([x for x in results if x['seed']%9==r]) for r in range(9)})
        _atomic_json(a.output/name/'summary.json',dict(summary=summary,results=results))
        state['collections' if collect else 'evaluations'].append(summary);publish();return results

    publish();deadline=state['started']+a.wall_seconds
    try:
        old=recode(a,a.output,state,publish);state['recoded_rows']=len(old[0])
        state['recoded_route_counts']={str(r):int((old[2]==r).sum()) for r in range(9)}
        model=TinyTarget(118,512).cuda()
        model.load_state_dict(torch.load(a.control,map_location='cpu',weights_only=True)['model'])
        best_checkpoint,best_rank=a.control,None
        def select(checkpoint,results):
            nonlocal best_checkpoint,best_rank
            rank=development_rank(results)
            improved=best_rank is None or rank>best_rank
            if improved:best_checkpoint,best_rank=checkpoint,rank
            selected=best_checkpoint if a.select_best_after_each_round else checkpoint
            if a.select_best_after_each_round:
                model.load_state_dict(torch.load(selected,map_location='cpu',weights_only=True)['model'])
            state['selection_history'].append(dict(candidate=str(checkpoint),rank=list(rank),improved=improved,
                selected=str(selected),rolled_back=selected!=checkpoint))
            state.update(best_checkpoint=str(best_checkpoint),best_development_rank=list(best_rank))
            publish();return selected
        dev=[g*9+r for g in range(97000000,97000002) for r in range(9)]
        if a.select_best_after_each_round and not a.smoke_only:
            baseline=rollout(a.control,dev,'development_baseline')
            select(a.control,baseline)
            if best_rank[0]>=15:
                state.update(status='complete_pending_review',checkpoint=str(a.control),development_gate_passed=True)
                publish('finished');return
        checkpoint=fit(model,old,None,a,state,publish,deadline)
        if a.smoke_only:
            state.update(status='smoke_complete',checkpoint=str(checkpoint));publish('finished');return
        results=rollout(checkpoint,dev,'development_0');checkpoint=select(checkpoint,results);online_chunks=[]
        for round_index in range(a.rounds):
            if (best_rank[0] if a.select_best_after_each_round else sum(r['safe_success'] for r in results))>=15:
                state['development_gate_passed']=True;break
            if time.time()>deadline-360:
                state['budget_stopped']=True;break
            state['round']=round_index+1
            collected=rollout(checkpoint,seed_schedule(round_index,a.group_start),f'collection_{round_index+1}',a.teacher_betas[round_index])
            for result in collected:
                with np.load(result['data'],allow_pickle=False) as data:
                    online_chunks.append((data['x'].copy(),data['teacher_action'].copy(),
                                          np.full(len(data['x']),result['seed']%9,np.int64)))
            online=tuple(np.concatenate([c[i] for c in online_chunks]) for i in range(3))
            state['online_rows']=len(online[0]);state['online_route_counts']={str(r):int((online[2]==r).sum()) for r in range(9)}
            checkpoint=fit(model,old,online,a,state,publish,deadline)
            results=rollout(checkpoint,dev,f'development_{round_index+1}')
            checkpoint=select(checkpoint,results)
        if (best_rank[0] if a.select_best_after_each_round else sum(r['safe_success'] for r in results))>=15:
            state['development_gate_passed']=True
        state.update(status='complete_pending_review',checkpoint=str(checkpoint));publish('finished')
    except BaseException as exc:
        state.update(status='failed',error=repr(exc));publish();raise


if __name__=='__main__':main()
