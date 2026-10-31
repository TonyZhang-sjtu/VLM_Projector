"""Wait for sufficient GPU memory, export CE, then refresh statistics."""
import argparse
from datetime import datetime
import fcntl
import os
from pathlib import Path
import subprocess
import sys
import time
from zoneinfo import ZoneInfo

from src.evaluation.clinical import write_json
from src.evaluation.ce_runtime import ce_replay_environment
from src.evaluation.significance_io import ROOT, read_json


def eligible_gpus():
    """Require only at least 20,000 MiB free; active jobs/utilization are allowed."""
    command = ['nvidia-smi', '--query-gpu=index,memory.free',
               '--format=csv,noheader,nounits']
    info = subprocess.check_output(command, text=True)
    available = []
    for line in info.splitlines():
        index, free = [s.strip() for s in line.split(',')]
        if int(free) >= 20000:
            available.append(index)
    return available


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--manifest', type=Path, required=True)
    cli.add_argument('--output-dir', type=Path, required=True)
    cli.add_argument('--projector-python', default='/mnt/sda/mingjie/conda_envs/projector/bin/python')
    cli.add_argument('--wait', action='store_true')
    cli.add_argument('--max-wait-hours', type=float, default=24)
    cli.add_argument('--poll-seconds', type=float, default=30)
    args = cli.parse_args()
    output = args.output_dir.resolve()
    ce_dir = output / 'ce'
    ce_dir.mkdir(parents=True, exist_ok=True)
    queue_lock = (ce_dir / 'queue.lock').open('w')
    fcntl.flock(queue_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    manifest = read_json(args.manifest)
    audits = {r['method_id']: r for r in read_json(output/'run_audit.json')}
    entries = [m for m in manifest['methods'] if m.get('run_dir')
               and audits.get(m['id'], {}).get('status') == 'verified'
               and audits[m['id']].get('checkpoint_bytes_verified')]
    # The uncompressed reference is slowest; run shorter CE exports first.
    entries.sort(key=lambda m: m['ratio'] == 1)
    done, failed = [], []
    wait_started = time.monotonic()

    def status(state, **extra):
        record = dict(status=state, pid=os.getpid(), completed=done, failed=failed,
                      methods=[m['id'] for m in entries], manifest=str(args.manifest.resolve()),
                      updated_at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(), **extra)
        write_json(ce_dir/'queue_status.json', record)
        print(record, flush=True)

    for entry in entries:
        gpu_lock = None
        if (ce_dir/entry['id']/'summary.json').exists():
            done.append(entry['id'])
            continue
        while gpu_lock is None:
            for index in eligible_gpus():
                lock_path = ROOT / '.tmp' / f'significance_gpu_{index}.lock'
                lock_path.parent.mkdir(exist_ok=True)
                candidate_lock = lock_path.open('w')
                try:
                    fcntl.flock(candidate_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    candidate_lock.close()
                    continue
                if index not in eligible_gpus():
                    candidate_lock.close()
                    continue
                gpu_lock, gpu = candidate_lock, index
                break
            if gpu_lock is None:
                status('waiting_for_gpu_memory', next_method=entry['id'])
                if not args.wait:
                    return
                if time.monotonic() - wait_started > args.max_wait_hours * 3600:
                    status('wait_timeout', next_method=entry['id'])
                    return
                time.sleep(min(60, max(5, args.poll_seconds)))
        method_output = ce_dir / entry['id']
        method_output.mkdir(exist_ok=True)
        status('running_ce', active_method=entry['id'], gpu=gpu)
        env = ce_replay_environment()
        env['CUDA_VISIBLE_DEVICES'] = gpu
        command = [args.projector_python, '-m', 'src.cli.export_answer_ce',
                   '--manifest', str(args.manifest.resolve()), '--method', entry['id'],
                   '--output-dir', str(method_output), '--device', 'cuda:0']
        try:
            with (method_output/'console.log').open('a') as log:
                result = subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
            (done if result.returncode == 0 else failed).append(entry['id'])
        finally:
            gpu_lock.close()
        wait_started = time.monotonic()
    status('refreshing_statistics')
    result = subprocess.run([sys.executable, '-m', 'src.cli.significance', '--manifest',
                             str(args.manifest.resolve()), '--output-dir', str(output)], cwd=ROOT)
    status('completed_available_sources' if not failed and result.returncode == 0 else 'finished_with_errors',
           report_returncode=result.returncode,
           missing_sources=[m['id'] for m in manifest['methods'] if not m.get('run_dir')])


if __name__ == '__main__':
    main()
