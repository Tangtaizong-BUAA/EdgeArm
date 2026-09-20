"""On-policy correction plus exact rendered color counterfactuals.

Color variants replay recorded physical states only for training images. They
are not additional physical trajectories and never contribute success counts.
The deployment policy receives only rendered RGB, FK, and completed actions.
"""
import argparse
from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import time

for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'): os.environ.setdefault(key,'1')
os.environ.setdefault('MUJOCO_GL','egl')
os.environ.setdefault('PYOPENGL_PLATFORM','egl')

import mujoco
import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import ConcatDataset, DataLoader, WeightedRandomSampler

from .constrained_recovery_run40 import summarize
from .production_env import BLOCK_COLORS, TARGET_COLORS
from .run34_repeat_eval import deterministic_runtime
from .run63_control_probe import TinyTarget
from .run67_visual_state import COLORS, VisualState, evaluate_episode, visual_batch
from .run68_visual_control import estimated_inputs, fit
from .run75_observation_training import SurveyFrames, collect
from .train_staged_hybrid_contact_sac import _atomic_json


def color_permutation(seed, variant):
    if variant not in (0,1,2): raise ValueError('bounded counterfactual variant')
    rng=np.random.default_rng(np.random.SeedSequence([seed,variant,7601]))
    return np.r_[rng.permutation(4),4+rng.permutation(3)]


def relabel(xy,mask,selected,permutation):
    p=np.asarray(permutation)
    if sorted(p[:4].tolist())!=list(range(4)) or sorted(p[4:].tolist())!=list(range(4,7)):
        raise ValueError('within-role color bijection required')
    target=np.empty_like(xy);valid=np.empty_like(mask)
    target[...,p,:]=xy;valid[...,p]=mask
    return target,valid,p[np.asarray(selected)]


def recolor_episode(session,folder,render_states,variants):
    """Only after physical collection ends; no generated image goes to actor."""
    env=session.env;scene=session.episode.multichoice.contract
    with np.load(folder/'frames.npz',allow_pickle=False) as z: data={k:z[k].copy() for k in z.files}
    if len(render_states)!=len(data['time_step']): raise ValueError('render-state/frame alignment')
    seed=json.loads((folder/'result.json').read_text())['seed']
    others=[i for i in range(3) if i!=scene['selected_block']]
    geometries=[(env._ids['block_geom'],COLORS.index(scene['block_colors'][scene['selected_block']]))]
    geometries += [(g,COLORS.index(scene['block_colors'][slot])) for g,slot in zip(session.episode.multichoice.geom_ids,others)]
    geometries += [(env._ids['target_geom'],COLORS.index(scene['target_colors'][scene['selected_target']]))]
    for i,slot in enumerate(j for j in range(3) if j!=scene['selected_target']):
        geom=mujoco.mj_name2id(env.model,mujoco.mjtObj.mjOBJ_GEOM,f'choice_target_{i}')
        geometries.append((geom,COLORS.index(scene['target_colors'][slot])))
    rgba=[*BLOCK_COLORS.values(),*TARGET_COLORS.values()]
    original=env.model.geom_rgba.copy();final_q=env.data.qpos.copy()
    np.savez_compressed(folder/'render_audit.npz',qpos=render_states,frame_steps=data['time_step'])
    try:
        # Verify saved poses reproduce the original RGB before recoloring.
        env.data.qpos[:]=render_states[0];mujoco.mj_forward(env.model,env.data)
        session.renderer.update_scene(env.data,camera='edgearm_wrist')
        probe=np.asarray(Image.fromarray(session.renderer.render()).resize((160,120)))
        delta=np.abs(probe.astype(np.int16)-data['rgb'][0].astype(np.int16))
        if delta.max()>2 or delta.mean()>.01: raise ValueError('physical state replay does not reproduce source RGB')
        for variant in range(variants):
            perm=color_permutation(seed,variant)
            for geom,index in geometries: env.model.geom_rgba[geom]=rgba[int(perm[index])][0]
            images=[]
            for q in render_states:
                env.data.qpos[:]=q;mujoco.mj_forward(env.model,env.data)
                session.renderer.update_scene(env.data,camera='edgearm_wrist')
                images.append(np.asarray(Image.fromarray(session.renderer.render()).resize((160,120))))
            xy,mask,selected=relabel(data['xy'],data['mask'],data['selected'],perm)
            out=folder/f'variant_{variant}';out.mkdir(exist_ok=False)
            np.savez_compressed(out/'frames.npz',**(data|dict(rgb=np.asarray(images),xy=xy,mask=mask,selected=selected)))
            _atomic_json(out/'result.json',dict(seed=seed,collection=True,counterfactual_render=True,
                physical_parent=str(folder),color_old_to_new=perm.tolist(),additional_physical_episodes=0,
                instruction='把'+BLOCK_COLORS[COLORS[int(selected[0])]][1]+'方块推到'+
                    TARGET_COLORS[COLORS[int(selected[1])]][1]+'目标区域。',
                actor_receives_render_qpos=False,source_replay_mean_pixel_error=float(delta.mean())))
    finally:
        env.model.geom_rgba[:]=original;env.data.qpos[:]=final_q;mujoco.mj_forward(env.model,env.data)


class ColorFrames(SurveyFrames):
    def __init__(self,roots,action_only=False):
        self.rows=[];self.index=[];self.cache=OrderedDict();self.action_only=action_only
        for root in roots:
            paths=sorted(Path(root).glob('episode_*/frames.npz'))+sorted(Path(root).glob('episode_*/variant_*/frames.npz'))
            for path in paths:
                meta=json.loads((path.parent/'result.json').read_text());seed=meta['seed']
                if not meta.get('collection') or not 100200000<=seed//9<100200016:
                    raise ValueError('new training collection only; no development or holdout')
                cache=path.parent/'numpy_frames'
                if not cache.exists():
                    cache.mkdir()
                    with np.load(path,allow_pickle=False) as z:
                        for key in z.files: np.save(cache/(key+'.npy'),z[key],allow_pickle=False)
                times=np.load(cache/'time_step.npy',mmap_mode='r')
                row=len(self.rows);self.rows.append((cache,seed))
                self.index.extend((row,k) for k,t in enumerate(times) if not action_only or t>=220)
        if not self.index: raise ValueError('empty color collection')


def mixture_loader(old,new,updates,batch):
    weights=[]
    for dataset in (old,new):
        routes=np.asarray([dataset.rows[row][1]%9 for row,_ in dataset.index])
        counts=np.bincount(routes,minlength=9)
        if np.any(counts==0): raise ValueError('all nine routes required in both sources')
        weights.extend((.5/(9*counts[routes])).tolist())
    sampler=WeightedRandomSampler(torch.tensor(weights,dtype=torch.double),updates*batch,replacement=True)
    return DataLoader(ConcatDataset([old,new]),batch_size=batch,sampler=sampler,num_workers=6,
                      pin_memory=True,persistent_workers=True)


def recode(visual,dataset):
    loader=DataLoader(dataset,batch_size=96,shuffle=False,num_workers=6,pin_memory=True)
    xs,ys,routes=[],[],[]
    with torch.inference_mode():
        for batch in loader:
            b=visual_batch(batch);world=visual(b['rgb'],b['pose'],b['K'],b['age'])
            xs.append(estimated_inputs(b['proprio'],world,b['selected']).cpu().numpy())
            ys.append(batch['command'].numpy());routes.append(batch['route'].numpy())
    return tuple(np.concatenate(x) for x in (xs,ys,routes))


def reused_collection_results(folder):
    """Reuse only the completed first training batch, never development data."""
    rows=[]
    expected={g*9+r for g in range(100200000,100200004) for r in range(9)}
    for path in sorted(Path(folder).glob('episode_*/result.json')):
        row=json.loads(path.read_text())
        if (row.get('seed') not in expected or row.get('collection') is not True
                or row.get('teacher_beta') != .75 or row.get('recolor_variants') != 3):
            raise ValueError('only the complete first Run76 collection can be reused')
        for variant in range(3):
            variant_path=path.parent/f'variant_{variant}'
            meta=json.loads((variant_path/'result.json').read_text())
            if (meta.get('seed') != row['seed'] or meta.get('counterfactual_render') is not True
                    or not (variant_path/'frames.npz').is_file()):
                raise ValueError('incomplete training color variant')
        if not (path.parent/'frames.npz').is_file(): raise ValueError('missing source frames')
        rows.append(row)
    if len(rows)!=36 or {r['seed'] for r in rows}!=expected:
        raise ValueError('all 36 first-round physical episodes are required')
    return rows


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('vision','control','old-collection','output'): p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--workers',type=int,default=9);p.add_argument('--rounds',type=int,default=3)
    p.add_argument('--visual-updates',type=int,default=1000);p.add_argument('--control-updates',type=int,default=3000)
    p.add_argument('--wall-seconds',type=int,default=2300);p.add_argument('--smoke-only',action='store_true')
    p.add_argument('--reuse-first-collection',type=Path)
    a=p.parse_args()
    if not 1<=a.workers<=9 or not 1<=a.rounds<=4 or not 50<=a.visual_updates<=1500:
        raise ValueError('bounded on-policy correction experiment')
    a.output.mkdir(parents=True,exist_ok=False);deterministic_runtime();torch.set_num_threads(2)
    current_vision,current_control=a.vision,a.control;roots=[]
    state=dict(run='Run76',status='running',started=time.time(),step=0,round=0,evaluations=[],collections=[],
        total_updates=a.rounds*(a.visual_updates+a.control_updates),
        training_kind='on_policy_DAgger_with_rendered_color_counterfactuals_not_RL',
        actor_uses_simulator_state=False,original_ACT_checkpoint=False,fixed_wrist_survey=True,retain_overview=True,
        fixed_survey_steps=220,max_steps=900,physical_episodes_per_round=36,color_variants_per_episode=3,
        teacher_route6_speed=.6,teacher_betas=[.75,.5,.25,0.],independent_acceptance=False,
        production_admission=False,export_admission=False,final_vla_acceptance=False)
    def publish(phase=None):
        if phase:state['phase']=phase
        state.update(updated=time.time(),elapsed_seconds=time.time()-state['started'])
        _atomic_json(a.output/'run_state.json',state)
    publish('initializing');deadline=time.time()+a.wall_seconds
    best_rank=(5,-2,.4068627450980392,.4361111111111111)
    try:
        for round_index in range(a.rounds):
            if time.time()>deadline-300:state['budget_stopped']=True;break
            state['round']=round_index+1;folder=a.output/f'collection_{round_index}';results=[]
            groups=1 if a.smoke_only else 4
            options=dict(beta=state['teacher_betas'][round_index],recolor_variants=3,route6_speed=.6,run76=True)
            state.update(phase_episodes_completed=0,phase_episodes_total=groups*9);publish('on_policy_color_collection')
            if round_index==0 and a.reuse_first_collection is not None:
                folder=a.reuse_first_collection;results=reused_collection_results(folder)
                state.update(reused_first_collection=str(folder),phase_episodes_completed=len(results),partial=summarize(results))
                publish('reused_training_collection')
            else:
                with ProcessPoolExecutor(a.workers,mp_context=mp.get_context('spawn')) as pool:
                    jobs=[(g*9+r,str(current_vision),str(current_control),str(folder),options)
                        for g in range(100200000+4*round_index,100200000+4*round_index+groups) for r in range(9)]
                    for future in as_completed([pool.submit(collect,j) for j in jobs]):
                        results.append(future.result());state.update(phase_episodes_completed=len(results),partial=summarize(results));publish()
            summary=summarize(results)|dict(physical_episodes=len(results),render_variants=3*len(results))
            if not (round_index==0 and a.reuse_first_collection is not None):
                _atomic_json(folder/'summary.json',dict(summary=summary,results=results))
            state['collections'].append(summary);roots.append(folder)
            if a.smoke_only:
                data=ColorFrames(roots,action_only=True);sample=data[0]
                state.update(smoke_frames=len(data),smoke_input_shape=list(sample['proprio'].shape));break
            old=SurveyFrames(a.old_collection);new=ColorFrames(roots)
            loader=mixture_loader(old,new,a.visual_updates,64)
            visual=VisualState().cuda();visual.load_state_dict(torch.load(current_vision,map_location='cpu',weights_only=True)['model'])
            optimizer=torch.optim.AdamW(visual.parameters(),lr=.00003,weight_decay=.0001,fused=True)
            state.update(old_training_frames=len(old),new_training_frames=len(new));publish('color_visual_adaptation')
            for batch in loader:
                if time.time()>deadline-240:raise TimeoutError('reserve evaluation time')
                b=visual_batch(batch)
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    world=visual(b['rgb'],b['pose'],b['K'],b['age'])
                    err=F.smooth_l1_loss(world/.02,b['xy']/.02,beta=.05,reduction='none').mean(-1)
                    loss=((err*b['mask']).sum(-1)/b['mask'].sum(-1).clamp_min(1)).mean()
                optimizer.zero_grad(set_to_none=True);loss.backward();nn.utils.clip_grad_norm_(visual.parameters(),5,error_if_nonfinite=True);optimizer.step()
                state['step']+=1
                if state['step']%50==0:
                    state['metrics']=dict(loss=float(loss.detach()),position_training_mean_mm=float(
                        torch.linalg.vector_norm(world.detach()-b['xy'],dim=-1)[b['mask']].mean()*1000));publish()
            vision_path=a.output/f'vision_round_{round_index+1}.pt'
            torch.save(dict(model={k:v.detach().cpu().clone() for k,v in visual.state_dict().items()},
                actor_uses_simulator_state=False,original_ACT_checkpoint=False,production_admission=False,
                export_admission=False,final_vla_acceptance=False),vision_path)
            del loader,old,new,optimizer;visual.eval();publish('color_control_recode')
            old_data=recode(visual,SurveyFrames(a.old_collection,action_only=True))
            online_data=recode(visual,ColorFrames(roots,action_only=True))
            del visual;torch.cuda.empty_cache()
            model=TinyTarget(118,512).cuda();model.load_state_dict(torch.load(current_control,map_location='cpu',weights_only=True)['model'])
            a.vision=vision_path;a.updates=a.control_updates;a.learning_rate=.00005
            state.update(vision_sha256=hashlib.sha256(vision_path.read_bytes()).hexdigest(),
                recoded_rows=len(old_data[0]),online_rows=len(online_data[0]))
            control_path=fit(model,old_data,online_data,a,state,publish,deadline)
            del model;torch.cuda.empty_cache();results=[];folder=a.output/f'development_{round_index+1}'
            state.update(phase_episodes_completed=0,phase_episodes_total=36);publish('color_autonomous_development')
            with ProcessPoolExecutor(a.workers,mp_context=mp.get_context('spawn')) as pool:
                jobs=[(g*9+r,str(vision_path),str(control_path),str(folder),None,
                    dict(fixed_wrist_survey=True,retain_overview=True)) for g in range(97100000,97100004) for r in range(9)]
                for future in as_completed([pool.submit(evaluate_episode,j) for j in jobs]):
                    results.append(future.result());state.update(phase_episodes_completed=len(results),partial=summarize(results));publish()
            summary=summarize(results)|dict(pairs={str(r):summarize([x for x in results if x['seed']%9==r]) for r in range(9)})
            _atomic_json(folder/'summary.json',dict(summary=summary,results=results));state['evaluations'].append(summary)
            rank=(summary['successes'],-summary['hard_failures'],summary['mean_coverage'],summary['mean_hold_s'])
            if rank>best_rank:
                best_rank=rank;current_vision,current_control=vision_path,control_path
            state.update(best_development_rank=list(best_rank),best_vision=str(current_vision),best_control=str(current_control))
            publish()
            if summary['successes']>=30:state['development_gate_passed']=True;break
        state['status']='smoke_complete' if a.smoke_only else 'complete_pending_review';publish('finished')
    except BaseException as exc:state.update(status='failed',error=repr(exc));publish();raise


if __name__=='__main__':main()
