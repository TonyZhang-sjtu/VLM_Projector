"""Supervise the unchanged production scorer and stop on runtime failures.

Only the child launched by this process may be terminated. Benign protocol
statuses (e.g. GREEN format repair failure) remain explicit scored outcomes.
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from .common import ROOT,write


def supervise(command,cache,status_path,interval=10,stall_seconds=1800):
    cache=Path(cache);status_path=Path(status_path)
    before={str(p):p.stat().st_mtime_ns for p in cache.glob('*/*.json')}
    known={};started=time.time();last_progress=started
    child=subprocess.Popen(command,cwd=ROOT)
    try:
        while True:
            counts={};errors=[]
            for p in cache.glob('*/*.json'):
                modified=p.stat().st_mtime_ns;key=str(p)
                if key not in known or known[key][0]!=modified:
                    result=json.loads(p.read_text());known[key]=(modified,result['status'])
                    if modified!=before.get(key):last_progress=time.time()
                state=known[key][1];group=p.parent.name
                counts.setdefault(group,{})[state]=counts.setdefault(group,{}).get(state,0)+1
                if state=='scoring_error' and modified!=before.get(key):errors.append(str(p))
            code=child.poll()
            progress=dict(status='running' if code is None else 'complete' if code==0 else 'failed',
                          pid=child.pid,elapsed=time.time()-started,cache_counts=counts,time=time.time())
            write(status_path,progress)
            if errors:raise RuntimeError(f'Production scorer runtime error; inspect cache entries: {errors[:5]}')
            if code is not None:
                if code:raise RuntimeError(f'Production scorer exited with status {code}')
                return
            if time.time()-last_progress>stall_seconds:
                raise RuntimeError(f'No scoring-cache progress for {stall_seconds} seconds; child PID {child.pid}')
            time.sleep(interval)
    except BaseException as exc:
        if child.poll() is None:
            child.terminate()
            try:child.wait(timeout=30)
            except subprocess.TimeoutExpired:child.kill();child.wait()
        write(status_path,dict(status='failed',error=str(exc),pid=child.pid,
              returncode=child.returncode,time=time.time(),elapsed=time.time()-started))
        raise


def main():
    parser=argparse.ArgumentParser(add_help=False)
    parser.add_argument('--cache-dir',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    args,_=parser.parse_known_args()
    command=[sys.executable,'-m','src.cli.score_reports',*sys.argv[1:]]
    supervise(command,args.cache_dir,args.output_dir/'worker_monitor.json')


if __name__=='__main__':main()
