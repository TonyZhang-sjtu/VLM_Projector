#!/usr/bin/env python3
"""Supervise authorized v2 preparation; stop on any unresolved QC failure.

This job never starts formal QA training. It waits for the frozen pilot, checks
the two-case integration and reproducibility, freezes the protocol only if the
train-only gates pass, then runs the full offline preprocessing and audit.
"""
import argparse
from datetime import datetime, timezone
import json
import os
import signal
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from src.data.neuroqa_manifest import write_json

PLAN = ROOT/'results/vqa/neuroqa/preprocessing_v2_plan'
STORE = ROOT/'assets/preprocessing/neuroqa_mni1mm_v2'
STATE = PLAN/'pipeline_status.json'


def report(phase, **fields):
    data = {'utc':datetime.now(timezone.utc).isoformat(),'phase':phase,**fields}
    write_json(STATE,data)
    print(json.dumps(data),flush=True)


def alive(pid):
    try:
        os.kill(pid,0)
        return True
    except ProcessLookupError:
        return False


def batch_done(path, expected):
    if not path.exists():
        return False
    data = json.loads(path.read_text())
    count = len(data.get('successful',data.get('success',[]))) + len(data.get('failed',[]))
    if data['requested'] != expected:
        raise RuntimeError(f'Unexpected batch size: {path}')
    if count == expected and data.get('failed'):
        raise RuntimeError(f'Batch has failures: {path}')
    return count == expected


def execute(label, args, *, allow_failure=False, env=None):
    report(label,command=args)
    with (PLAN/f'{label}.log').open('a' if label == 'full_preprocessing' else 'w') as out:
        child = subprocess.Popen(args,cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                                 env=env,text=True,bufsize=1,start_new_session=True)
        try:
            for line in child.stdout:
                out.write(line)
                out.flush()
                print(line,end='',flush=True)
            returncode = child.wait()
        finally:
            if child.poll() is None:
                try:
                    os.killpg(child.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                child.wait()
    if returncode and not allow_failure:
        raise RuntimeError(f'{label} exited {returncode}; inspect {label}.log')
    return returncode


def wait_gpu(required_mib):
    deadline = time.monotonic() + 12*3600
    while time.monotonic() < deadline:
        query = subprocess.run(['nvidia-smi','-i','0','--query-gpu=memory.free',
                                '--format=csv,noheader,nounits'],capture_output=True,text=True,check=True)
        available = int(query.stdout.strip().splitlines()[0])
        if available >= required_mib:
            return
        report('waiting_for_gpu_0',available_mib=available,required_mib=required_mib)
        time.sleep(30)
    raise RuntimeError('GPU 0 did not have enough free memory for the requested smoke within 12 hours')


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--pilot-pid',type=int,default=0)
    p.add_argument('--sandbox-pid',type=int,default=0)
    p.add_argument('--repro-pid',type=int,default=0)
    p.add_argument('--workers',type=int,default=16)
    p.add_argument('--resume-frozen',action='store_true',help='Resume the already frozen protocol and preserve completed cases')
    p.add_argument('--resume-from-mlp',type=Path,help='Resume after a completed 55-configuration projector smoke report')
    args = p.parse_args()
    py = sys.executable
    if args.resume_from_mlp:
        smoke = args.resume_from_mlp.resolve()
        measurements = json.loads((smoke/'report.json').read_text())
        audit = json.loads((STORE/'independent_audit.json').read_text())
        final = json.loads((STORE/'finalization.json').read_text())
        if (len(measurements) != 55 or any(x.get('status') != 'passed' for x in measurements)
                or not audit.get('summary',{}).get('ready_for_full_training')
                or len(final.get('verified_volumes',[])) != 828 or final.get('failed')
                or not (STORE/'qc_review_queue.json').is_file()):
            raise RuntimeError('Cannot resume at MLP: completed matrix or preprocessing audit is missing or failed')
        gpu_env = {**os.environ,'CUDA_VISIBLE_DEVICES':'0'}
        gpu_env.setdefault('PYTORCH_CUDA_ALLOC_CONF','expandable_segments:True')
        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
        mlp = ROOT/'runs/vqa/neuroqa/smoke'/f'v2_mlp_{stamp}'
        wait_gpu(21000)
        execute('mlp_smoke',[py,'scripts/vqa/neuroqa/smoke_gpu.py','--config',
            'configs/vqa/neuroqa/mixed_v2.yaml','--methods','mlp2x_gelu','--ratios','1','--output',str(mlp)],env=gpu_env)
        trainer = ROOT/'runs/vqa/neuroqa/smoke'/f'v2_trainer_resume_{stamp}'
        wait_gpu(20000)
        execute('trainer_resume_smoke',[py,'scripts/vqa/neuroqa/smoke_training_resume.py','--config',
            'configs/vqa/neuroqa/mixed_v2.yaml','--output',str(trainer)],env=gpu_env)
        report('awaiting_qc_review',audited_cases=audit['summary']['audited_cases'],
            reviewed=False,ready_for_full_training=False,projector_smoke=str(smoke),
            mlp_smoke=str(mlp),trainer_resume_smoke=str(trainer),
            message='Independent audit and engineering smokes passed; review flagged cases before finalization.')
        return
    if not args.resume_frozen:
        if not all((args.pilot_pid, args.sandbox_pid, args.repro_pid)):
            p.error('pilot, sandbox and reproducibility PIDs are required unless --resume-frozen is set')
        start = time.monotonic()
        pilot = PLAN/'pilot/batch.json'
        repro = PLAN/'reproducibility/batch.json'
        sandbox = PLAN/'sandbox/neuroqa_mni1mm_v2/cases'
        cases = ('BraTS-GLI-00017-000','BraTS-GLI-00540-000')
        while True:
            p_done = batch_done(pilot,96)
            r_done = batch_done(repro,4)
            s_done = all((sandbox/c/'complete.json').is_file() for c in cases)
            report('waiting_for_pilot',pilot_complete=p_done,repro_complete=r_done,sandbox_complete=s_done)
            for done,pid,name in ((p_done,args.pilot_pid,'pilot'),(r_done,args.repro_pid,'reproducibility'),
                                   (s_done,args.sandbox_pid,'sandbox')):
                if not done and not alive(pid):
                    raise RuntimeError(f'{name} worker exited before completing')
            if p_done and r_done:
                break
            if time.monotonic()-start > 12*3600:
                raise RuntimeError('Pilot/integration did not finish within 12 hours')
            time.sleep(20)
        execute('pilot_summary_complete',[py,'scripts/vqa/neuroqa/summarize_pilot_v2.py'])
        execute('prepare_pilot_review',[py,'scripts/vqa/neuroqa/make_pilot_review_v2.py'])
        visual_review = PLAN/'pilot_visual_review/review.json'
        while not visual_review.exists():
            report('awaiting_pilot_visual_review',review_queue=str(PLAN/'pilot_visual_review/queue.json'))
            time.sleep(20)
        while not all((sandbox/c/'complete.json').is_file() for c in cases):
            if not alive(args.sandbox_pid):
                raise RuntimeError('Sandbox worker exited before completing')
            report('waiting_for_sandbox_n4')
            time.sleep(20)
        execute('sandbox_verify',[py,'scripts/vqa/neuroqa/verify_v2_sandbox.py'])
        execute('protocol_selection',[py,'scripts/vqa/neuroqa/select_v2_protocol.py','--apply'])
    else:
        if not (STORE/'protocol.json').is_file():
            raise RuntimeError('Cannot resume: frozen protocol is missing')
        report('resuming_frozen',workers=args.workers)
    execute('full_preprocessing',[py,'scripts/vqa/neuroqa/preprocess_smri_v2.py','--workers',str(args.workers)])
    # The first finalization builds all caches but deliberately cannot mark ready
    # before an independently verified QC review exists.
    execute('cache_build',[py,'scripts/vqa/neuroqa/finalize_preprocessing.py','--config',
                            'configs/vqa/neuroqa/mixed_v2.yaml'],allow_failure=True)
    final = json.loads((STORE/'finalization.json').read_text())
    if len(final['verified_volumes']) != final['expected_volumes'] or final['failed']:
        raise RuntimeError('Model patch cache finalization failed')
    execute('independent_audit',[py,'scripts/vqa/neuroqa/audit_preprocessing.py','--root',str(STORE),
        '--config','configs/vqa/neuroqa/mixed_v2.yaml','--output',str(STORE/'independent_audit.json'),'--workers','4'])
    audit = json.loads((STORE/'independent_audit.json').read_text())
    if not audit.get('summary',{}).get('ready_for_full_training'):
        raise RuntimeError('Independent full-data audit failed; inspect independent_audit.json')
    execute('intensity_audit',[py,'scripts/vqa/neuroqa/audit_intensity_v2.py','--workers','4'])
    execute('lesion_coverage_audit',[py,'scripts/vqa/neuroqa/audit_lesion_coverage_v2.py','--workers','4'])
    execute('prepare_review_queue',[py,'scripts/vqa/neuroqa/prepare_v2_review.py'])
    gpu_env = {**os.environ,'CUDA_VISIBLE_DEVICES':'0'}
    gpu_env.setdefault('PYTORCH_CUDA_ALLOC_CONF','expandable_segments:True')
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    smoke = ROOT/'runs/vqa/neuroqa/smoke'/f'v2_projector_matrix_{stamp}'
    wait_gpu(20000)
    execute('projector_smoke',[py,'scripts/vqa/neuroqa/smoke_gpu.py','--config',
        'configs/vqa/neuroqa/mixed_v2.yaml','--output',str(smoke)],env=gpu_env)
    mlp = ROOT/'runs/vqa/neuroqa/smoke'/f'v2_mlp_{stamp}'
    wait_gpu(21000)
    execute('mlp_smoke',[py,'scripts/vqa/neuroqa/smoke_gpu.py','--config',
        'configs/vqa/neuroqa/mixed_v2.yaml','--methods','mlp2x_gelu','--ratios','1','--output',str(mlp)],env=gpu_env)
    trainer = ROOT/'runs/vqa/neuroqa/smoke'/f'v2_trainer_resume_{stamp}'
    wait_gpu(20000)
    execute('trainer_resume_smoke',[py,'scripts/vqa/neuroqa/smoke_training_resume.py','--config',
        'configs/vqa/neuroqa/mixed_v2.yaml','--output',str(trainer)],env=gpu_env)
    # Never auto-claim visual review. Review report and subsequent finalization
    # are a deliberate separate phase once every flagged case is assessed.
    report('awaiting_qc_review',audited_cases=audit['summary']['audited_cases'],
        reviewed=False,ready_for_full_training=False,
        projector_smoke=str(smoke),mlp_smoke=str(mlp),trainer_resume_smoke=str(trainer),
        message='Independent audit and engineering smokes passed; review flagged cases before finalization.')


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        report('stopped',error=repr(exc),ready_for_full_training=False)
        raise
