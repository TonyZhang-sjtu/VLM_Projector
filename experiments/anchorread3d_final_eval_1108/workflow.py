"""Monitored, gated continuation: p-values -> RG controls -> scoring -> GREEN.

No training commands, no original result writes, and no process-group signals.
Each next stage requires both process success and independent output validation.
"""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

from .common import ROOT,OUT,INDEX,read,write
from .queue import run_queue


def stage(name,command):
    logs=OUT/'workflow_logs';logs.mkdir(parents=True,exist_ok=True)
    path=logs/f'{name}.log';start=time.time()
    print('STAGE_START',name,flush=True)
    with path.open('a') as f:
        p=subprocess.Popen(command,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
        while p.poll() is None:
            write(OUT/'workflow_status.json',dict(status='running',stage=name,pid=p.pid,
                  elapsed=time.time()-start,time=time.time(),log=str(path)))
            time.sleep(10)
    if p.returncode:raise RuntimeError(f'{name} failed with {p.returncode}; inspect {path}')
    print('STAGE_DONE',name,round(time.time()-start,1),flush=True)


def main():
    from filelock import FileLock
    OUT.mkdir(parents=True,exist_ok=True)
    with FileLock(str(OUT/'workflow.lock'),timeout=0):
        try:
            while not (OUT/'queues/ce/complete.json').exists():
                status=read(OUT/'queues/ce/status.json') if (OUT/'queues/ce/status.json').exists() else {}
                write(OUT/'workflow_status.json',dict(status='running',stage='ce_exports',time=time.time(),
                      completed=len(status.get('completed',[])),pending=status.get('pending'),running=status.get('running'),
                      failed=len(status.get('failed',[]))))
                time.sleep(20)
            assert read(OUT/'queues/ce/complete.json')['status']=='complete','CE queue failure blocks visual stage'
            python=sys.executable;module='experiments.anchorread3d_final_eval_1108'
            from src.evaluation.clinical import file_digest
            previous=read(OUT/'significance/completion.json') if (OUT/'significance/completion.json').exists() else {}
            index=read(INDEX)
            pvalues_complete=(previous.get('status')=='complete' and previous.get('completed')==1050
                and previous.get('source_workbook_sha256')==file_digest(index['master'])
                and previous.get('source_index_sha256')==file_digest(INDEX)
                and file_digest(previous['workbook']['path'])==previous['workbook']['sha256'])
            if not pvalues_complete:stage('final_pvalues',[python,'-m',f'{module}.significance'])
            else:print('REUSE_VERIFIED_FINAL_PVALUES',flush=True)
            pv=read(OUT/'significance/completion.json')
            assert pv['status']=='complete' and pv['completed']==pv['planned']==1050
            stage('visual_prepare',[python,'-m',f'{module}.visual','--prepare'])
            smoke=[dict(id='rg_A0_r32_smoke',command=[python,'-m',f'{module}.visual','--setting','rg/A0/r32','--limit','4'])]
            write(OUT/'workflow_status.json',dict(status='running',stage='visual_smoke',time=time.time()))
            smoke_path=OUT/'visual/rg__A0__r32/smoke.json'
            smoke_saved=read(smoke_path) if smoke_path.exists() else {}
            if smoke_saved.get('status')!='passed':assert run_queue(smoke,'visual_smoke')
            else:
                assert smoke_saved['zero_and_text_equal_production_controls']
                assert len(smoke_saved['checks'])==4 and all(x['same_generated_tokens'] for x in smoke_saved['checks'])
                print('REUSE_VERIFIED_VISUAL_SMOKE',flush=True)
            assert read(OUT/'visual/rg__A0__r32/smoke.json')['status']=='passed'
            jobs=[dict(id=k,command=[python,'-m',f'{module}.visual','--setting',k]) for k in read(OUT/'visual/plan.json')['settings']]
            write(OUT/'workflow_status.json',dict(status='running',stage='visual_inference',time=time.time()))
            assert run_queue(jobs,'visual_inference')
            stage('validate_visual_inference',[python,'-m',f'{module}.validate_visual'])
            stage('visual_pool',[python,'-m',f'{module}.score_visual','pool'])
            stage('visual_graphs',[python,'-m',f'{module}.score_visual','graphs'])
            write(OUT/'workflow_status.json',dict(status='running',stage='clinical_non_green',time=time.time()))
            assert run_queue(read(OUT/'visual/scoring/clinical_jobs.json'),'clinical_non_green')
            stage('verify_non_green',[python,'-m',f'{module}.score_visual','verify'])
            stage('export_before_green',[python,'-m',f'{module}.export_visual','--allow-pending-green'])
            # GREEN is last, and this explicit gate is reused by the standalone shell launcher.
            stage('green_gate',[python,'-m',f'{module}.green_gate'])
            write(OUT/'workflow_status.json',dict(status='running',stage='green',time=time.time()))
            assert run_queue(read(OUT/'visual/scoring/green_jobs.json'),'green')
            stage('verify_green',[python,'-m',f'{module}.score_visual','verify-green'])
            stage('export_final',[python,'-m',f'{module}.export_visual'])
            from src.evaluation.clinical import file_digest
            index=read(INDEX);assert file_digest(index['master'])==index['master_sha256']
            write(OUT/'workflow_status.json',dict(status='complete',time=time.time(),
                  pvalues=read(OUT/'significance/completion.json'),visual=read(OUT/'visual/export.json')))
        except BaseException as exc:
            write(OUT/'workflow_status.json',dict(status='failed',error=str(exc),traceback=traceback.format_exc(),time=time.time()))
            raise


if __name__=='__main__':main()
