"""M06--M09: one concurrent wave per single-GPU Slurm allocation."""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import threading
import time
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[2]
OUR = 'bindformer_adirect_localread'
GROUPS = {
    'm06': [('vqa', 'concat3d', 256), ('vqa', 'concat3d', 128)],
    'm07': [('vqa', 'adaptive_merge3d', 256), ('vqa', 'token_select3d', 64),
            ('vqa', 'token_select3d', 32), ('vqa', 'token_select3d', 16)],
    'm08': [('vqa', 'token_select3d', 128), ('vqa', OUR, 128), ('vqa', OUR, 32)],
    'm09': [('rg', OUR, 128), ('rg', OUR, 256), ('vqa', OUR, 64), ('vqa', OUR, 16)],
}


def now():
    return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(timespec='seconds')


def attempt_tag(base_tag, job_id, restart_count):
    """Each Slurm restart starts fresh in its own namespace, never from last.pt."""
    if restart_count < 0:
        raise ValueError('SLURM_RESTART_COUNT must not be negative')
    if restart_count:
        if not job_id or not re.fullmatch(r'[0-9]+', job_id):
            raise ValueError('A Slurm restart requires a numeric SLURM_JOB_ID')
        return f'{base_tag}_j{job_id}_restart{restart_count}'
    return base_tag


def assert_fresh(specs):
    existing = [spec['run'] for spec in specs if Path(spec['run']).exists()]
    if existing:
        raise FileExistsError(f'Refusing to overwrite existing training runs: {existing}')


def tasks(group, tag):
    result = []
    for task, method, ratio in GROUPS[group]:
        label = f'{task}_{method}_r{ratio}'
        if task == 'vqa':
            script = 'vqa_baseline_adirect_localread.sh' if method == OUR else f'vqa_{method}.sh'
            command = ['bash', str(ROOT / 'scripts/Exp_VQA' / script), '--ratios', str(ratio),
                       '--run-tag', tag]
            if method == 'concat3d':
                command += ['--concat-device', 'cuda:0']
            run = ROOT / 'runs/vqa/neuroqa/official_holdout_patient_guard_v1/v2/unreviewed_training/mixed' / method / f'r{ratio}' / 's42' / tag
            train = run / 'train/train.jsonl'
        else:
            command = ['bash', str(ROOT / 'scripts/Exp1026/run_baseline_adirect_localread.sh'),
                       '--ratios', str(ratio), '--run-tag', f'{tag}_{label}']
            run = ROOT / 'runs/baselines' / f'{tag}_{label}' / f'r{ratio}'
            train = run / 'joint_lora/train.jsonl'
        result.append(dict(label=label, task=task, method=method, ratio=ratio,
                           command=command, run=str(run), train_log=str(train)))
    return result


def stop_children(children):
    # Kill complete process groups, including sweep -> trainer/evaluator children.
    for process in children:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    deadline = time.monotonic() + 5
    while any(p.poll() is None for p in children) and time.monotonic() < deadline:
        time.sleep(0.1)
    for process in children:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()


def gpu_monitor(path, stop):
    with path.open('a', buffering=1) as out:
        while not stop.is_set():
            row = {'time': now()}
            for key, query in [('gpu', '--query-gpu=uuid,name,memory.used,memory.total,utilization.gpu'),
                               ('processes', '--query-compute-apps=pid,process_name,used_gpu_memory')]:
                try:
                    r = subprocess.run(['nvidia-smi', query, '--format=csv,noheader,nounits'],
                                       capture_output=True, text=True, timeout=10)
                    row[key] = r.stdout.strip().splitlines()
                    if r.returncode:
                        row[key + '_error'] = r.stderr.strip()
                except Exception as exc:
                    row[key + '_error'] = str(exc)
            out.write(json.dumps(row) + '\n')
            stop.wait(5)


def run_wave(specs, directory):
    children, handles = [], []
    stop = threading.Event()
    monitor = threading.Thread(target=gpu_monitor, args=(directory / 'gpu.jsonl', stop), daemon=True)
    monitor.start()
    def interrupted(signum, frame):
        raise KeyboardInterrupt(f'signal {signum}')
    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(sig, interrupted)
    status = dict(started_at=now(), job_id=os.environ['SLURM_JOB_ID'], state='running',
                  restart_count=int(os.environ.get('SLURM_RESTART_COUNT', '0')),
                  training_policy='fresh_no_checkpoint_resume', tasks=[])
    def save():
        temporary = directory / 'status.tmp'
        temporary.write_text(json.dumps(status, indent=2) + '\n')
        temporary.replace(directory / 'status.json')
    try:
        for spec in specs:
            handle = (directory / f"{spec['label']}.log").open('a')
            handles.append(handle)
            process = subprocess.Popen(spec['command'], cwd=ROOT, stdout=handle,
                                       stderr=subprocess.STDOUT, start_new_session=True)
            children.append(process)
            status['tasks'].append(dict(**spec, pid=process.pid, state='running', started_at=now()))
            print(f"START {spec['label']} pid={process.pid}", flush=True)
            save()
        pending = set(range(len(children)))
        while pending:
            for index in list(pending):
                code = children[index].poll()
                if code is None:
                    continue
                pending.remove(index)
                row = status['tasks'][index]
                row.update(state='completed' if code == 0 else 'failed', exit_code=code, ended_at=now())
                save()
                print(f"EXIT {row['label']} code={code}", flush=True)
                if code:
                    raise RuntimeError(f"{row['label']} exited {code}; stopping allocation's other workers")
            if pending:
                time.sleep(1)
        status.update(state='completed', ended_at=now())
    except BaseException as exc:
        status.update(state='failed', error=str(exc), ended_at=now())
        raise
    finally:
        stop_children(children)
        for index, process in enumerate(children):
            if status['tasks'][index]['state'] == 'running':
                status['tasks'][index].update(state='stopped', exit_code=process.returncode, ended_at=now())
        save()
        stop.set()
        monitor.join(timeout=22)
        for handle in handles:
            handle.close()


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('group', choices=GROUPS)
    parser.add_argument('--run-tag', default=os.environ.get('MARLOWE_RUN_TAG'))
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    base_tag = args.run_tag or f'marlowe_{args.group}_' + datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%dT%H%M%S')
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', base_tag):
        parser.error('Invalid run tag')
    job_id = os.environ.get('SLURM_JOB_ID')
    restart_count = int(os.environ.get('SLURM_RESTART_COUNT', '0'))
    tag = attempt_tag(base_tag, job_id, restart_count)
    specs = tasks(args.group, tag)
    if not args.dry_run and not os.environ.get('SLURM_JOB_ID'):
        parser.error('Submit this GPU workload through Slurm; use --dry-run on login nodes')
    assert_fresh(specs)
    directory = ROOT / ('.marlowe/m06_m09/dryrun' if args.dry_run else 'logs/marlowe_waves') / tag
    directory.mkdir(parents=True, exist_ok=args.dry_run)
    plan = dict(group=args.group, tag=tag, base_tag=base_tag, job_id=job_id,
                restart_count=restart_count, training_policy='fresh_no_checkpoint_resume',
                created_at=now(), tasks=specs)
    (directory / 'plan.json').write_text(json.dumps(plan, indent=2) + '\n')
    if not args.dry_run:
        attempts = ROOT / '.marlowe/m06_m09/attempts'
        attempts.mkdir(parents=True, exist_ok=True)
        temporary = attempts / f'{job_id}.tmp'
        temporary.write_text(json.dumps(plan, indent=2) + '\n')
        temporary.replace(attempts / f'{job_id}.json')
    print(f'FRESH TRAINING tag={tag} restart_count={restart_count}; no checkpoint resume', flush=True)
    # Dry-run every child before launching any worker; only one ratio per child.
    for spec in specs:
        with (directory / f"{spec['label']}.plan.log").open('w') as handle:
            subprocess.run(spec['command'] + ['--dry-run'], cwd=ROOT, stdout=handle,
                           stderr=subprocess.STDOUT, check=True)
        print(f"CHECKED {spec['label']}", flush=True)
    if args.dry_run:
        print(directory / 'plan.json')
        return
    result = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total,memory.free',
                             '--format=csv,noheader,nounits'], capture_output=True, text=True, check=True)
    lines = result.stdout.strip().splitlines()
    if len(lines) != 1 or 'H100' not in lines[0] or float(lines[0].split(',')[-1]) < 80000:
        raise RuntimeError(f'Expected one free H100 80GB; got {result.stdout}')
    print(f'GPU {lines[0]}', flush=True)
    run_wave(specs, directory)


if __name__ == '__main__':
    main()
