"""Bounded difficult-route collection using the unchanged frozen RL teacher."""
import os
os.environ.setdefault('MUJOCO_GL','egl')
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='1'
import argparse
from concurrent.futures import ProcessPoolExecutor,wait,FIRST_COMPLETED
import multiprocessing as mp
from pathlib import Path
import shutil
import time
from .run43_teacher_collection import initialize_teacher,collect_teacher
from .run44_posttrain.data import digest
from .run42.domain import sample_domain
from .train_staged_hybrid_contact_sac import _atomic_json

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--workers',type=int,default=28)
    p.add_argument('--attempts',type=int,default=180)
    p.add_argument('--seed-base',type=int,default=66000000)
    a=p.parse_args()
    if not 1<=a.workers<=28 or not 1<=a.attempts<=240:raise ValueError('bounded batch required')
    a.output.mkdir(parents=True,exist_ok=False)
    ctx=mp.get_context('spawn');devices=ctx.Queue()
    for i in range(a.workers):devices.put(i%2)
    rows=[];pending={};submitted=0;started=time.time();sha=digest(a.checkpoint)
    def save(status):
        _atomic_json(a.output/'run_state.json',dict(status=status,started=started,updated=time.time(),
            workers=a.workers,submitted=submitted,completed=len(rows),accepted=sum(r['accepted'] for r in rows),
            records=rows,pairs=[0,6,7],seed_base=a.seed_base,model_updates=0,
            production_admission=False,export_admission=False))
        _atomic_json(a.output/'success_manifest.json',dict(records=[r for r in rows if r['accepted']],
            complete=status!='running',production_admission=False,export_admission=False))
    with ProcessPoolExecutor(a.workers,mp_context=ctx,initializer=initialize_teacher,
            initargs=(str(a.checkpoint),devices)) as pool:
        while submitted<a.attempts or pending:
            while submitted<a.attempts and len(pending)<a.workers:
                if shutil.disk_usage(a.output).free<20*1024**3:raise RuntimeError('disk reserve')
                i=submitted;pair=[0,6,7][i%3];group=a.seed_base+i
                job=dict(output=str(a.output/f'episode_{i:05d}'),seed=group*9+pair,
                    parameters=sample_domain(group+43,0),teacher_sha256=sha,
                    stratum=f'pair_{pair//3}_{pair%3}',split='validation' if group%10==0 else 'train')
                pending[pool.submit(collect_teacher,job)]=i;submitted+=1
            save('running')
            done,_=wait(pending,timeout=20,return_when=FIRST_COMPLETED)
            for f in done:rows.append(f.result());del pending[f]
        save('completed')

if __name__=='__main__':main()
