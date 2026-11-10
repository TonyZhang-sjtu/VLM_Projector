"""Persistent progress/exception observer for our isolated evaluation only."""
import json
import math
import os
from pathlib import Path
import time

from .common import OUT,read,write


def main():
    from filelock import FileLock
    OUT.mkdir(parents=True,exist_ok=True)
    with FileLock(str(OUT/'monitor.lock'),timeout=0):
        known={};alerts_seen=set();started=time.time()
        history=OUT/'alerts.jsonl'
        if history.exists():
            for line in history.read_text().splitlines():
                item=json.loads(line);item.pop('time',None)
                alerts_seen.add(json.dumps(item,sort_keys=True))
        while True:
            records=[];alerts=[]
            for path in sorted((OUT/'queues').glob('*/status.json')):
                s=read(path);phase=path.parent.name
                row=dict(phase=phase,pending=s['pending'],completed=len(s['completed']),failed=len(s['failed']),running=[])
                for gpu,item in s['running'].items():
                    value=dict(item,gpu=gpu)
                    try:os.kill(item['pid'],0);value['process_alive']=True
                    except ProcessLookupError:value['process_alive']=False
                    # Stale entries after process completion are harmless until
                    # the queue's next 10-second update.
                    if not value['process_alive'] and time.time()-s['time']>60:
                        alerts.append(dict(kind='stale_worker',phase=phase,id=item['id']))
                    row['running'].append(value)
                for failure in s['failed']:
                    alerts.append(dict(kind='subprocess_failure',phase=phase,id=failure['job']['id'],returncode=failure['returncode']))
                records.append(row)
            ce=[]
            for folder in sorted((OUT/'ce').glob('*')):
                if not folder.is_dir():continue
                p=folder/'samples.jsonl'
                if not p.exists():p=folder/'samples.jsonl.tmp'
                if not p.exists():continue
                # The writer flushes full JSON lines. Ignore a transient partial
                # last line, which is never considered a complete observation.
                try:data=p.read_text()
                except FileNotFoundError:continue  # Atomic completion rename raced this observation.
                lines=data.splitlines()
                if not data.endswith('\n'):lines=lines[:-1]
                rr=[json.loads(line) for line in lines]
                assert len(rr)==len({r['id'] for r in rr}),f'Duplicate CE samples: {folder}'
                assert all(math.isfinite(r['loss_sum']) and r['loss_sum']>=0 and r['answer_tokens']>0 for r in rr)
                ce.append(dict(setting=folder.name,samples=len(rr),complete=(folder/'summary.json').exists()))
            scoring={}
            for cache in sorted((OUT/'visual/scoring').glob('cache_*')):
                counts={}
                for p in cache.glob('*/*.json'):
                    st=p.stat();old=known.get(str(p))
                    if old is None or old[0]!=st.st_mtime_ns:
                        d=read(p);known[str(p)]=(st.st_mtime_ns,d['status'])
                    status=known[str(p)][1];metric=p.parent.name
                    counts.setdefault(metric,{})[status]=counts.setdefault(metric,{}).get(status,0)+1
                    if status=='scoring_error':alerts.append(dict(kind='scoring_error',path=str(p)))
                scoring[cache.name]=counts
            for alert in alerts:
                key=json.dumps(alert,sort_keys=True)
                if key not in alerts_seen:
                    alerts_seen.add(key)
                    with (OUT/'alerts.jsonl').open('a') as f:f.write(json.dumps(dict(time=time.time(),**alert))+'\n')
                    print('ALERT',key,flush=True)
            state=read(OUT/'workflow_status.json') if (OUT/'workflow_status.json').exists() else {}
            reuse=len(list((OUT/'ce_reuse').glob('*.json')))
            resolutions=read(OUT/'alert_resolutions.json').get('alerts',[]) if (OUT/'alert_resolutions.json').exists() else []
            resolved={json.dumps(r['alert'],sort_keys=True) for r in resolutions}
            write(OUT/'monitor_status.json',dict(time=time.time(),elapsed=time.time()-started,workflow=state,
                  queues=records,ce_exports=ce,reused_ce_exports=reuse,scoring_cache_progress=scoring,
                  alerts=len(alerts_seen),resolved_alerts=len(alerts_seen & resolved),unresolved_alerts=len(alerts_seen-resolved)))
            print('MONITOR',state.get('stage',state.get('status')), 'CE',sum(r['complete'] for r in ce),
                  'partial',sum(not r['complete'] for r in ce),'unresolved_alerts',len(alerts_seen-resolved),flush=True)
            if state.get('status') in ('complete','failed'):return
            time.sleep(30)


if __name__=='__main__':main()
