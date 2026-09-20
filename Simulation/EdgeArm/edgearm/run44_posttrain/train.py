"""One or two GPUs train ONE incremental ACT model; no deployment promotion."""
import argparse
from collections import defaultdict
from contextlib import nullcontext
from dataclasses import asdict
from datetime import timedelta
import hashlib
import json
import os
from pathlib import Path
import signal
import time

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

from ..candidate_command_contract_v2 import ACTION_CONTRACT
from ..evaluate_multimodal_act_v5 import load_model
from ..temporal_input_contract_v3 import INPUT_KEYS
from ..train_multimodal_act_v5 import FORMAT
from ..train_staged_hybrid_contact_sac import _atomic_json
from .data import COHORTS, MMapEpisodes, check_disjoint, digest
from .objective import Objective, configure_trainable, parameter_groups, lr_multiplier, per_window_l1
from .sampling import BalancedBatches, ValidationBatches

METRIC_NAMES=('loss','prior_loss','posterior_loss','kl','future_tool_per_0p1m',
              'old_anchor_loss','first_command_mae','old_anchor_eligible_fraction')


def runtime_ranks(profile, environment=None):
    env=os.environ if environment is None else environment
    size=int(env.get('WORLD_SIZE',1));rank=int(env.get('RANK',0))
    local=int(env.get('LOCAL_RANK',rank))
    if (profile['schema']!='run44-incremental-act-v1' or size!=profile['world_size']
            or size not in (1,2) or not 0<=rank<size or not 0<=local<size):
        raise ValueError('profile must match one or two synchronized GPU ranks')
    return rank,size,local


def gather(value, size):
    if size==1:
        return [value]
    result=[None]*size
    dist.all_gather_object(result,value)
    return result


def reduce(value, size, op=dist.ReduceOp.SUM):
    if size>1:
        dist.all_reduce(value,op=op)


def cpu_worker(_):
    torch.set_num_threads(1)


def move(batch,device):
    if isinstance(batch,dict):
        return {k:move(v,device) for k,v in batch.items()}
    return batch.to(device,non_blocking=True)


def loader(records,cfg,sampler,p,workers):
    kwargs=dict(batch_sampler=sampler,pin_memory=True,num_workers=workers,
                worker_init_fn=cpu_worker,
                generator=torch.Generator().manual_seed(p['seed']+90001))
    if workers:
        kwargs.update(persistent_workers=True,prefetch_factor=p['prefetch_factor'],
                      multiprocessing_context='spawn')
    return DataLoader(MMapEpisodes(records,cfg,p['mmap_episode_cache_per_worker']),**kwargs)


def fingerprint(model):
    h=hashlib.sha256()
    for name,value in model.state_dict().items():
        h.update(name.encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def evaluate(model, batches, records, p, device, world_size):
    model.eval()
    local=defaultdict(lambda:[0.,0.,0.])
    with torch.inference_mode():
        for host in batches:
            b=move(host,device)
            with torch.autocast('cuda',dtype=torch.bfloat16,enabled=p['precision']=='bf16'):
                predicted=model(b['inputs'])
            if not torch.isfinite(predicted).all():
                raise ValueError('nonfinite prior validation')
            chunk,first=per_window_l1(predicted,b['target'],b['mask'])
            values=torch.stack([chunk,first],-1).cpu().tolist()
            for ri,(ce,fe) in zip(host['record_id'].tolist(),values):
                row=records[ri]
                for key in (row['cohort'],row['cohort']+'/'+str(row.get('pair')),'pooled'):
                    local[key][0]+=ce
                    local[key][1]+=fe
                    local[key][2]+=1
    shards=gather(dict(local),world_size)
    total=defaultdict(lambda:[0.,0.,0.])
    for shard in shards:
        for k,values in shard.items():
            total[k]=[a+b for a,b in zip(total[k],values)]
    result={k:dict(chunk_mae=v[0]/v[2],first_mae=v[1]/v[2],windows=int(v[2])) for k,v in total.items()}
    if set(COHORTS)-set(result):
        raise ValueError('validation lacks one or more rehearsal cohorts')
    result['selection_score']=sum(p['cohort_mass'][c]*result[c]['first_mae'] for c in COHORTS)
    result['posterior_used']=False
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile',type=Path,required=True)
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--initial-checkpoint',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--resume',type=Path)
    parser.add_argument('--readiness',type=Path,
                        help='Current Run44 handoff plan required for full training')
    parser.add_argument('--smoke-updates',type=int,default=0)
    parser.add_argument('--batch-per-gpu',type=int)
    parser.add_argument('--stop-after-update',type=int,
                        help='Engineering interruption test; save a recoverable boundary without changing schedule')
    parser.add_argument('--deterministic-audit',action='store_true',
                        help='Smoke-only deterministic kernels for restart equivalence, not throughput measurement')
    args=parser.parse_args()
    p=json.loads(args.profile.read_text())
    rank,size,local_rank=runtime_ranks(p)
    if p['automatic_rl_launch'] or any(p[k] for k in ('production_admission','export_admission','final_vla_acceptance')):
        raise ValueError('offline fitting cannot enable RL/deployment acceptance')
    if not 0<=args.smoke_updates<=32:
        raise ValueError('smoke is bounded to 32 updates')
    if args.deterministic_audit and not args.smoke_updates:
        raise ValueError('deterministic restart audit requires bounded smoke')
    p['deterministic_audit']=args.deterministic_audit
    if args.batch_per_gpu:
        p['batch_per_gpu']=args.batch_per_gpu
    if args.smoke_updates:
        p.update(epochs=1,steps_per_epoch=args.smoke_updates,
                 validation_windows_per_episode=2,
                 log_every_updates=1,max_wall_seconds=900)
    else:
        if args.readiness is None:
            raise ValueError('full training requires a current phase/resource handoff')
        handoff=json.loads(args.readiness.read_text())
        if (handoff.get('schema')!='run44-posttrain-handoff-v1' or not handoff['data']['ready']
                or handoff['data_sha256']!=digest(args.manifest)
                or handoff['base_sha256']!=digest(args.initial_checkpoint)
                or handoff['profile_sha256']!=digest(args.profile)):
            raise ValueError('full training handoff not ready or stale')
        if args.batch_per_gpu is not None and args.batch_per_gpu!=json.loads(args.profile.read_text())['batch_per_gpu']:
            raise ValueError('benchmark a changed batch in the profile and regenerate handoff')
        for path in handoff['collector_states']:
            if json.loads(Path(path).read_text())['status'] not in (
                'quota_complete','partial_quota_complete','pilot_complete','time_budget_stopped','completed','snapshot_complete'):
                raise ValueError('collection phase not cleanly complete; do not oversubscribe both stages')
    torch.cuda.set_device(local_rank)
    device=torch.device('cuda',local_rank)
    if p['precision']=='bf16' and not torch.cuda.is_bf16_supported():
        raise ValueError('BF16 requested but unsupported')
    torch.set_num_threads(p['cpu_threads_per_rank'])
    torch.set_num_interop_threads(1)
    torch.manual_seed(p['seed'])
    torch.backends.cuda.matmul.allow_tf32=True
    if args.deterministic_audit:
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark=False
        torch.backends.cudnn.deterministic=True
        torch.backends.cuda.matmul.allow_tf32=False
        torch.backends.cudnn.allow_tf32=False
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
    if size>1:
        dist.init_process_group('nccl',timeout=timedelta(minutes=15),device_id=device)
    if not args.deterministic_audit:
        torch.backends.cudnn.benchmark=p.get('cudnn_benchmark',False)
    started=time.time()
    stop_requested=False
    def stop(signum,frame):
        nonlocal stop_requested
        stop_requested=True
    signal.signal(signal.SIGTERM,stop)
    manifest=json.loads(args.manifest.read_text())
    if manifest['schema']!='run44-mmap-causal-v1' or manifest['action_contract']!=ACTION_CONTRACT:
        raise ValueError('data contract mismatch')
    if manifest['smoke_only'] and not args.smoke_updates:
        raise ValueError('smoke subset cannot be used as formal data')
    rows=manifest['records']
    check_disjoint(rows)
    tr=[r for r in rows if r['split']=='train']
    va=[r for r in rows if r['split']=='validation']
    if not args.smoke_updates:
        if sum(r['cohort'].startswith('new_') for r in tr)<p['minimum_new_train_episodes']:
            raise ValueError('not enough completed new training episodes')
        if sum(r['cohort']=='new_edge' for r in tr)<p['minimum_new_edge_train_episodes']:
            raise ValueError('need edge evidence, not just familiar duplicates')
    inputs=dict(manifest=digest(args.manifest),initial_checkpoint=digest(args.initial_checkpoint))
    all_inputs=gather(inputs,size)
    if any(x!=inputs for x in all_inputs):
        raise ValueError('ranks loaded different data/model snapshots')
    state=dict(schema='run44-incremental-act-v1',status='initializing',started=started,
        smoke_only=bool(args.smoke_updates),profile=p,input_sha256=inputs,world_size=size,
        single_shared_model=True,closed_loop_evaluated=False,production_admission=False,
        export_admission=False,final_vla_acceptance=False,exact_home_evaluated=False)
    def publish():
        state.update(updated=time.time(),elapsed_seconds=time.time()-started)
        if rank==0:
            _atomic_json(args.output/'run_state.json',state)
    try:
        if rank==0:
            args.output.mkdir(parents=True,exist_ok=False)
        if size>1:
            dist.barrier()
        publish()
        model,base=load_model(args.initial_checkpoint)
        if base['format']!=FORMAT:
            raise ValueError('warm start must be the original ACT architecture')
        configure_trainable(model,p['rgb_unfreeze'])
        model.to(device)
        objective=Objective(model,p).to(device)
        optimizer=torch.optim.AdamW(parameter_groups(model,p),fused=True)
        start_update=0
        resume=None
        if args.resume:
            resume=torch.load(args.resume,map_location='cpu',weights_only=False)
            if resume['input_sha256']!=inputs or resume['profile']!=p or resume['world_size']!=size:
                raise ValueError('exact resume requires identical data/profile/world size')
            model.load_state_dict(resume['model'])
            optimizer.load_state_dict(resume['optimizer'])
            start_update=resume['step']
        ddp=(DDP(objective,device_ids=[local_rank],find_unused_parameters=True,
                gradient_as_bucket_view=True,broadcast_buffers=False) if size>1 else objective)
        initial_hash=fingerprint(model)
        hashes=gather(initial_hash,size)
        if len(set(hashes))!=1:
            raise ValueError('DDP initial weights differ')
        train_sampler=BalancedBatches(tr,p['cohort_mass'],batch_size=p['batch_per_gpu'],
            batches=p['steps_per_epoch']*p['gradient_accumulation'],rank=rank,world_size=size,
            seed=p['seed'],stride=p['window_stride'],uniform_mix=p['pair_balance_uniform_mix'],
            max_oversampling=p['pair_max_oversampling'])
        train_loader=loader(tr,model.config,train_sampler,p,p['loader_workers_per_gpu'])
        val_sampler=ValidationBatches(va,p['batch_per_gpu'],rank,size,p['validation_windows_per_episode'])
        val_loader=loader(va,model.config,val_sampler,p,min(2,p['loader_workers_per_gpu']))
        state.update(parameters=sum(x.numel() for x in model.parameters()),
            trainable_parameters=sum(x.numel() for x in model.parameters() if x.requires_grad),
            frozen_parameters=[n for n,x in model.named_parameters() if not x.requires_grad],
            global_batch=p['batch_per_gpu']*size*p['gradient_accumulation'],
            training_episodes=len(tr),validation_episodes=len(va),status='baseline_validation')
        publish()
        current_validation=evaluate(model,val_loader,va,p,device,size)
        baseline=current_validation if resume is None else resume['baseline_validation']
        best=baseline['selection_score'] if resume is None else resume['best_score']
        if rank==0:
            _atomic_json(args.output/'baseline_validation.json',baseline)
            if resume:
                _atomic_json(args.output/'resume_validation.json',current_validation)
        if resume:
            torch.set_rng_state(resume['rng_by_rank'][rank]['cpu'])
            torch.cuda.set_rng_state(resume['rng_by_rank'][rank]['cuda'],device)
        else:
            torch.manual_seed(p['seed']+rank)
        observed=torch.zeros(len(COHORTS),device=device,dtype=torch.int64)
        epoch_stats=[]
        update=start_update
        total_updates=p['epochs']*p['steps_per_epoch']
        metrics_sum=torch.zeros(len(METRIC_NAMES),device=device)
        timing=dict(data_wait_seconds=0.,gpu_update_seconds=0.,wall_window_start=time.monotonic())
        last_logged=update
        unique=set()
        pending_events=[]
        def save(name,epoch):
            rng=gather(dict(cpu=torch.get_rng_state(),cuda=torch.cuda.get_rng_state(device)),size)
            if rank==0:
                value=dict(format=FORMAT,posttrain_schema='run44-incremental-act-v1',
                    config=asdict(model.config),model={k:v.detach().cpu() for k,v in model.state_dict().items()},
                    optimizer=optimizer.state_dict(),step=update,epoch=epoch,profile=p,world_size=size,
                    action_contract=ACTION_CONTRACT,input_keys=sorted(INPUT_KEYS),input_sha256=inputs,
                    initial_checkpoint_sha256=inputs['initial_checkpoint'],rng_by_rank=rng,best_score=best,
                    baseline_validation=baseline,smoke_only=bool(args.smoke_updates),
                    production_admission=False,export_admission=False,final_vla_acceptance=False)
                tmp=args.output/(name+'.tmp')
                torch.save(value,tmp)
                tmp.replace(args.output/name)
        for epoch in range(start_update//p['steps_per_epoch'],p['epochs']):
            train_sampler.set_epoch(epoch)
            iterator=iter(train_loader)
            skipped=(start_update%p['steps_per_epoch'])*p['gradient_accumulation'] if epoch==start_update//p['steps_per_epoch'] else 0
            for _ in range(skipped):
                next(iterator)
            objective.train()
            updates_this_epoch=p['steps_per_epoch']-(start_update%p['steps_per_epoch'] if epoch==start_update//p['steps_per_epoch'] else 0)
            for _ in range(updates_this_epoch):
                scale=lr_multiplier(update,total_updates,p['warmup_updates'])
                for group in optimizer.param_groups:
                    group['lr']=group['initial_lr']*scale
                optimizer.zero_grad(set_to_none=True)
                events=[]
                for micro in range(p['gradient_accumulation']):
                    tick=time.monotonic()
                    host=next(iterator)
                    timing['data_wait_seconds']+=time.monotonic()-tick
                    unique.update(zip(host['record_id'].tolist(),host['time_index'].tolist()))
                    batch=move(host,device)
                    observed+=torch.bincount(batch['cohort'],minlength=len(COHORTS))
                    begin,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                    begin.record()
                    context=(ddp.no_sync() if size>1 and micro+1<p['gradient_accumulation'] else nullcontext())
                    with context:
                        with torch.autocast('cuda',dtype=torch.bfloat16,enabled=p['precision']=='bf16'):
                            loss,metrics=ddp(batch,update)
                        if not torch.isfinite(loss):
                            raise ValueError('nonfinite training loss')
                        (loss/p['gradient_accumulation']).backward()
                    end.record()
                    events.append((begin,end))
                    metrics_sum+=metrics.detach()/p['gradient_accumulation']
                norm=torch.nn.utils.clip_grad_norm_(model.parameters(),p['gradient_clip'],error_if_nonfinite=True)
                optimizer.step()
                # Read timing events at log boundaries, not a forced device fence
                # after every optimizer update. Keep finite-loss/gradient guards.
                pending_events.extend(events)
                update+=1
                if update%p['log_every_updates']==0 or update==total_updates:
                    sums=metrics_sum.clone()
                    reduce(sums,size)
                    n=update-last_logged
                    means=(sums/(size*n)).cpu().tolist()
                    mix=observed.clone()
                    reduce(mix,size)
                    pending_events[-1][1].synchronize()
                    timing['gpu_update_seconds']+=sum(a.elapsed_time(b) for a,b in pending_events)/1000
                    pending_events.clear()
                    elapsed=time.monotonic()-timing['wall_window_start']
                    state.update(status='smoke_training' if args.smoke_updates else 'act_finetuning',
                        step=update,total_updates=total_updates,epoch=epoch+1,
                        metrics=dict(zip(METRIC_NAMES,means)),gradient_norm=float(norm),
                        samples_per_second=n*state['global_batch']/max(elapsed,1e-6),
                        actual_sample_counts=dict(zip(COHORTS,mix.cpu().tolist())),
                        data_wait_seconds_rank0=timing['data_wait_seconds'],
                        gpu_update_seconds_rank0=timing['gpu_update_seconds'],
                        window_wall_seconds=elapsed,unique_windows_rank0=len(unique),
                        cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
                        cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved(device))
                    publish()
                    if rank==0:
                        with (args.output/'training_metrics.jsonl').open('a') as f:
                            f.write(json.dumps(state)+'\n')
                    metrics_sum.zero_()
                    last_logged=update
                    timing=dict(data_wait_seconds=0.,gpu_update_seconds=0.,wall_window_start=time.monotonic())
                should_stop=bool(stop_requested or time.time()-started>p['max_wall_seconds']
                    or (args.stop_after_update is not None and update>=args.stop_after_update))
                if size>1:
                    flag=torch.tensor(int(should_stop),device=device)
                    reduce(flag,size,op=dist.ReduceOp.MAX)
                    should_stop=bool(flag.item())
                if update%p['checkpoint_every_updates']==0 or should_stop:
                    save('checkpoint_last.pt',epoch)
                if should_stop:
                    state['status']='budget_or_signal_stopped'
                    publish()
                    return
            state['status']='prior_validation'
            publish()
            validation=evaluate(model,val_loader,va,p,device,size)
            epoch_stats.append(dict(epoch=epoch+1,step=update,**validation))
            if rank==0:
                _atomic_json(args.output/'validation.json',dict(baseline=baseline,epochs=epoch_stats))
            # An offline candidate is not deployment admission. Keep the last
            # model as well so a small validation fluctuation never loses work.
            old_ok=all(validation[c]['first_mae']<=baseline[c]['first_mae']*1.10+.005 for c in ('old_rl','old_human'))
            if validation['selection_score']<best and old_ok:
                best=validation['selection_score']
                save('checkpoint_best_offline.pt',epoch)
            save('checkpoint_last.pt',epoch)
        final_hash=fingerprint(model)
        hashes=gather(final_hash,size)
        if len(set(hashes))!=1:
            raise ValueError('ranks ended with different parameters')
        state.update(status='smoke_complete' if args.smoke_updates else 'act_complete_pending_closed_loop',
            step=update,shared_parameter_sha256=final_hash,parameter_update_observed=final_hash!=initial_hash,
            best_offline_score=best,baseline_score=baseline['selection_score'])
        publish()
    except BaseException as error:
        state.update(status='failed',error=repr(error))
        publish()
        raise
    finally:
        if size>1:
            dist.destroy_process_group()


if __name__=='__main__':
    main()
