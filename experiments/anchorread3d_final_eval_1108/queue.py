"""Resource-aware isolated subprocess queue; never signals foreign processes."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import time

from .common import OUT,ROOT,read,write


def gpu_snapshot():
    raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True)
    processes=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],text=True)
    busy={line.split(',')[0].strip() for line in processes.splitlines() if ',' in line}
    result=[]
    for line in raw.splitlines():
        i,uuid,used,free,util=[v.strip() for v in line.split(',')]
        result.append(dict(index=i,uuid=uuid,used=int(used),free=int(free),util=int(util),has_process=uuid in busy))
    return result


def available_gpus(snapshot,green=False):
    free=[s['index'] for s in snapshot if not s['has_process'] and s['used']<750 and s['free']>=21000 and s['util']<=5]
    # User policy: <=2 free -> use all; >2 free -> leave one unused.
    return free if len(free)<=2 else free[:-1]


def launchable_gpus(snapshot,reserved):
    # Reservations count even before a CPU-loading worker creates a CUDA
    # context. Otherwise the same card can be counted as both free and active.
    free=[s['index'] for s in snapshot if s['index'] not in reserved and not s['has_process']
          and s['used']<750 and s['free']>=21000 and s['util']<=5]
    capacity=len(free)+len(reserved)
    target=capacity if capacity<=2 else capacity-1
    return free[:max(0,target-len(reserved))]


def environment(gpu):
    env=dict(os.environ)
    env.update(CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1',
               PYTHONNOUSERSITE='1',TOKENIZERS_PARALLELISM='false',CUBLAS_WORKSPACE_CONFIG=':4096:8',
               PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True',PYTHONUNBUFFERED='1')
    return env


def run_queue(jobs,name):
    from filelock import FileLock
    directory=OUT/'queues'/name;directory.mkdir(parents=True,exist_ok=True)
    # The automatic workflow and standalone GREEN launcher share this lock.
    with FileLock(str(directory/'.queue.lock'),timeout=0):
        return _run_queue(jobs,name)


def _run_queue(jobs,name):
    directory=OUT/'queues'/name;directory.mkdir(parents=True,exist_ok=True)
    pending=list(jobs);running={};completed=[];failed=[]
    while pending or running:
        for gpu,(proc,job,handle,started) in list(running.items()):
            rc=proc.poll()
            if rc is None:continue
            handle.close();record=dict(job=job,returncode=rc,seconds=time.time()-started)
            (completed if rc==0 else failed).append(record)
            print('DONE' if rc==0 else 'FAILED',job['id'],'gpu',gpu,'code',rc,flush=True)
            del running[gpu]
        snap=gpu_snapshot()
        # Account for our active workers too, so a later poll cannot consume the
        # one GPU deliberately left free at the previous poll.
        # A worker loading on CPU may not yet appear in nvidia-smi; its assigned
        # GPU is nevertheless reserved and must not be counted twice.
        for gpu in launchable_gpus(snap,running):
            if gpu in running or not pending:continue
            job=pending.pop(0);log=directory/(job['id'].replace('/','__')+'.log')
            handle=log.open('a');proc=subprocess.Popen(job['command'],cwd=ROOT,env=environment(gpu),stdout=handle,stderr=subprocess.STDOUT)
            running[gpu]=(proc,job,handle,time.time());print('START',job['id'],'gpu',gpu,'pid',proc.pid,flush=True)
        write(directory/'status.json',dict(time=time.time(),pending=len(pending),
              running={g:dict(pid=v[0].pid,id=v[1]['id'],elapsed=time.time()-v[3]) for g,v in running.items()},
              completed=completed,failed=failed,gpus=snap))
        if pending or running:time.sleep(10)
    write(directory/'complete.json',dict(status='complete' if not failed else 'failed',completed=completed,failed=failed))
    return not failed


def main():
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['ce','jobs'],required=True);p.add_argument('--jobs',type=Path);p.add_argument('--name',default='custom');a=p.parse_args()
    if a.phase=='ce':
        missing=read(OUT/'significance/ce_missing.json')
        missing.sort(key=lambda e:(e['key'].split('/')[1]!='A0',int(e['key'].split('/r')[-1]),e['key']))
        jobs=[dict(id=e['key'],command=[sys.executable,'-m','experiments.anchorread3d_final_eval_1108.inference','--setting',e['key']]) for e in missing]
        name='ce'
    else:jobs=read(a.jobs);name=a.name
    if not run_queue(jobs,name):raise SystemExit(1)


if __name__=='__main__':main()
