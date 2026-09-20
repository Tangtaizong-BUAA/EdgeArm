"""Recode complete causal tracking sequences for an action-adaptation probe."""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch


def corrected_actor_inputs(features,correction):
    if features.shape[-1]!=246 or correction.shape!=features.shape[:-1]+(2,):
        raise ValueError('causal encoded schema mismatch')
    # features contain raw perceptual coordinates, not the position labels.
    x=features[...,:118].clone()
    x[...,108:110]+=correction
    return x


def command_sources(roots):
    sources={}
    for root in roots:
        for name in ('states.npz','perceived_commands.npz'):
            for path in sorted(Path(root).glob('collection_*/episode_*/'+name)):
                metadata=json.loads((path.parent/'result.json').read_text());seed=metadata['seed']
                if not metadata.get('collection') or not 99500000<=seed//9<100000000:
                    raise ValueError('dedicated training command sources required')
                if seed in sources:raise ValueError('duplicate training seed')
                sources[seed]=path
    return sources


def recode_with_tracker(a,output,state,publish):
    from .run71_temporal_tracking import load_tracker
    parent=json.loads((a.encoded_sequences.parent/'run_state.json').read_text())
    vision_hash=hashlib.sha256(a.vision.read_bytes()).hexdigest()
    if parent['vision_sha256']!=vision_hash:raise ValueError('encoded visual representation mismatch')
    sequences=torch.load(a.encoded_sequences,map_location='cpu',weights_only=False)
    sources=command_sources(a.tracked_command_roots);tracker=load_tracker(a.tracker,a.vision)
    xs,ys,routes=[],[],[];validation=set()
    with torch.inference_mode():
        for sequence in sequences:
            seed=sequence['seed']
            if not 99500000<=seed//9<100000000:raise ValueError('non-training seed in encoded data')
            if sequence['validation']:
                validation.add(seed//9);continue
            features=torch.from_numpy(sequence['x'])[None].cuda()
            correction,_=tracker(features)
            x=corrected_actor_inputs(features,correction)[0].cpu().numpy()
            with np.load(sources[seed],allow_pickle=False) as z:
                times=z['frame_steps']
                if len(times)!=len(x) or not np.array_equal(times,np.arange(len(x))*4):
                    raise ValueError('tracking/action times must match exactly')
                y=z['teacher_action'][times].copy()
            xs.append(x);ys.append(y);routes.append(np.full(len(x),seed%9,np.int64))
            state['recode_completed']=sum(len(x) for x in xs);publish('recode_tracked_inputs')
    result=tuple(np.concatenate(parts) for parts in (xs,ys,routes))
    np.savez_compressed(output/'recoded_commands.npz',x=result[0],action=result[1],route=result[2])
    state.update(correction_validation_groups=sorted(validation),recode_total=len(result[0]))
    del tracker;torch.cuda.empty_cache()
    return result
