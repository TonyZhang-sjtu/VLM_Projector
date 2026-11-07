"""Observe specific fresh jobs until every worker has at least five valid updates.
No GPU allocation, checkpoint loading, resubmission, or cancellation is performed.
"""
import argparse
from datetime import datetime
import json
from pathlib import Path
import subprocess
import sys
import time
from zoneinfo import ZoneInfo
import yaml

ROOT = Path(__file__).resolve().parents[2]
TERMINAL_FAILURE = {'FAILED','CANCELLED','TIMEOUT','NODE_FAIL','OUT_OF_MEMORY','PREEMPTED','BOOT_FAIL','DEADLINE'}


def inspect_worker(task):
    path=Path(task['train_log'])
    rows=[]
    if path.exists():
        for line in path.read_text().splitlines():
            try: rows.append(json.loads(line))
            except json.JSONDecodeError: pass
    if not rows: return {'label':task['label'],'ready':False,'reason':'waiting for first update'}
    train_dir=path.parent
    config_path=train_dir/('config.yaml' if task['task']=='vqa' else 'resolved_config.yaml')
    if not config_path.exists():
        config_path=Path(task['run'])/('launch_config.yaml' if task['task']=='vqa' else 'training_config.yaml')
    if not config_path.exists():
        return {'label':task['label'],'ready':False,'reason':'waiting for saved config'}
    config=yaml.safe_load(config_path.read_text())
    training=config['training']
    check={'label':task['label'],'first_step':rows[0].get('step'),'last_step':rows[-1].get('step'),
           'steps':len(rows),'resume':training.get('resume'),'init_checkpoint':training.get('init_checkpoint'),
           'finite_loss':task['finite_loss'],'config':str(config_path)}
    errors=[]
    if check['first_step']!=1:errors.append('first logged step is not 1')
    if check['resume'] or check['init_checkpoint']:errors.append('training checkpoint resume/init set')
    if not task['finite_loss']:errors.append('non-finite loss')
    if task['error_lines']:errors.extend(task['error_lines'])
    if task['task']=='vqa' and not task['balanced_all_steps']:errors.append('unbalanced T1/T2 training window')
    check['errors']=errors
    check['ready']=not errors and len(rows)>=5 and task['state'] in ('running','completed')
    return check


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--job-ids',nargs='+',required=True)
    parser.add_argument('--hours',type=float,default=36)
    args=parser.parse_args()
    ids=set(args.job_ids)
    directory=ROOT/'.marlowe/m06_m09'
    deadline=time.monotonic()+args.hours*3600
    while True:
        report={'checked_at':datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),'job_ids':sorted(ids)}
        try:
            proc=subprocess.run([sys.executable,str(ROOT/'scripts/marlowe/check_rerun_waves.py')],capture_output=True,text=True,check=True,timeout=45)
            snapshot=json.loads(proc.stdout)
            jobs=[j for j in snapshot['jobs'] if j['job_id'] in ids]
            if {j['job_id'] for j in jobs}!=ids:raise ValueError('Current submissions no longer match monitored job IDs')
            snapshot_path=directory/'fresh_start_status.json'
            temp=snapshot_path.with_suffix('.tmp');temp.write_text(json.dumps(snapshot,indent=2)+'\n');temp.replace(snapshot_path)
            workers=[inspect_worker(t) for j in jobs for t in j['tasks']]
            report.update(scheduler={j['job_id']:j['state'] for j in jobs},workers=workers)
            failed=[j['job_id'] for j in jobs if j['state'].split('+')[0] in TERMINAL_FAILURE]
            if failed or any(w.get('errors') for w in workers):
                report.update(state='needs_attention',failed_jobs=failed)
            elif len(workers)==11 and all(w['ready'] for w in workers):
                report['state']='verified_11_workers_fresh_at_least_5_steps'
            else:report['state']='waiting'
        except Exception as exc:
            report.update(state='monitor_error',error=str(exc))
        if time.monotonic()>=deadline and report['state'] in ('waiting','monitor_error'):
            report['state']='monitor_timeout'
        temp=directory/'fresh_start_watch.tmp'
        temp.write_text(json.dumps(report,indent=2)+'\n');temp.replace(directory/'fresh_start_watch.json')
        with (directory/'fresh_start_watch.jsonl').open('a') as f:f.write(json.dumps(report)+'\n')
        print(json.dumps(report),flush=True)
        if report['state']=='verified_11_workers_fresh_at_least_5_steps':return 0
        if report['state'] in ('needs_attention','monitor_timeout'):return 1
        time.sleep(60)

if __name__=='__main__':
    raise SystemExit(main())
