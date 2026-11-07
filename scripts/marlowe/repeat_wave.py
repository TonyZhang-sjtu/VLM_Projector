"""M13: four fresh ADirect+LocalRead repeats, started ten seconds apart."""
import argparse
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import threading
import time
from uuid import uuid4

from convpool_wave import assert_fresh, preflight
from rerun_wave import ROOT, OUR, gpu_monitor, now, stop_children

LAUNCH_INTERVAL = 10.0


def unique_tag(base, job_id=None, restart_count=0):
    if restart_count < 0 or (restart_count and not job_id):
        raise ValueError('A restart requires a job ID and nonnegative restart count')
    if job_id and not re.fullmatch(r'[0-9]+', job_id):
        raise ValueError('Expected a numeric Slurm job ID')
    return base + (f'_j{job_id}' if job_id else '') + (f'_restart{restart_count}' if restart_count else '')


def tasks(tag):
    specs = []
    for task, ratio, repeat in [('rg', 16, 1), ('rg', 16, 2), ('vqa', 64, 1), ('vqa', 64, 2)]:
        label = f'{task}_{OUR}_r{ratio}_rep{repeat:02d}'
        run_tag = f'{tag}_{label}'
        if task == 'rg':
            script = ROOT / 'scripts/Exp1026/run_baseline_adirect_localread.sh'
            run = ROOT / 'runs/baselines' / run_tag / f'r{ratio}'
            metric = run / 'evaluation/best_test/metric.json'
            train = run / 'joint_lora/train.jsonl'
            recipe = ['--set', 'seed=42', '--set', 'training.deterministic=true']
        else:
            script = ROOT / 'scripts/Exp_VQA/vqa_baseline_adirect_localread.sh'
            relative = Path('vqa/neuroqa/official_holdout_patient_guard_v1/v2/unreviewed_training/mixed') / OUR / f'r{ratio}' / 's42' / run_tag
            run = ROOT / 'runs' / relative
            metric = ROOT / 'results' / relative / 'test/normal/metrics.json'
            train = run / 'train/train.jsonl'
            recipe = ['--seed', '42', '--set', 'training.deterministic=false']
        specs.append(dict(label=label, task=task, method=OUR, ratio=ratio, repeat=repeat,
                          seed=42, deterministic=task == 'rg', run_tag=run_tag,
                          launch_order=len(specs), initial=True,
                          command=['bash', str(script), '--ratios', str(ratio), '--run-tag', run_tag, *recipe],
                          run=str(run), train_log=str(train), metrics=str(metric)))
    return specs


def assert_unique(specs):
    for key in ['label', 'run_tag', 'run', 'train_log', 'metrics']:
        values = [spec[key] for spec in specs]
        if len(set(values)) != len(values):
            raise ValueError(f'Duplicate {key}; refusing to start colliding experiments')
    roots = [Path(spec['run']).resolve() for spec in specs]
    for i, left in enumerate(roots):
        for right in roots[i+1:]:
            if left in right.parents or right in left.parents:
                raise ValueError('Overlapping training directories')


def run_wave(specs, directory, interval=LAUNCH_INTERVAL):
    """Launch each branch once; keep its slot through final evaluation."""
    children, handles, active = [], [], {}
    stop = threading.Event()
    monitor = threading.Thread(target=gpu_monitor, args=(directory / 'gpu.jsonl', stop), daemon=True)
    status = dict(started_at=now(), job_id=os.environ['SLURM_JOB_ID'], state='running',
                  launch_interval_seconds=interval, max_workers=len(specs),
                  restart_count=int(os.environ.get('SLURM_RESTART_COUNT', '0')),
                  training_policy='fresh_no_checkpoint_resume',
                  tasks=[dict(**spec, state='waiting') for spec in specs])
    previous_handlers = {}

    def save():
        temporary = directory / 'status.tmp'
        temporary.write_text(json.dumps(status, indent=2) + '\n')
        temporary.replace(directory / 'status.json')

    def interrupted(signum, frame):
        raise KeyboardInterrupt(f'signal {signum}')

    try:
        assert_unique(specs)
        assert_fresh(specs)
        for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
            previous_handlers[sig] = signal.signal(sig, interrupted)
        monitor.start()
        save()
        next_index, next_launch = 0, time.monotonic()
        while active or next_index < len(specs):
            for index, process in list(active.items()):
                code = process.poll()
                if code is None:
                    continue
                del active[index]
                row = status['tasks'][index]
                row.update(state='completed' if code == 0 else 'failed', exit_code=code, ended_at=now())
                save()
                print(f"EXIT {row['label']} code={code}", flush=True)
                if code:
                    raise RuntimeError(f"{row['label']} exited {code}; stopping this allocation's workers")
            if next_index < len(specs) and time.monotonic() >= next_launch:
                spec = specs[next_index]
                assert_fresh([spec])
                handle = (directory / f"{spec['label']}.log").open('x')
                handles.append(handle)
                process = subprocess.Popen(spec['command'], cwd=ROOT, stdout=handle,
                                           stderr=subprocess.STDOUT, start_new_session=True)
                children.append(process)
                active[next_index] = process
                status['tasks'][next_index].update(state='running', pid=process.pid, started_at=now())
                print(f"START {spec['label']} pid={process.pid} active={len(active)}/{len(specs)}", flush=True)
                save()
                next_index += 1
                next_launch = time.monotonic() + interval
            if active or next_index < len(specs):
                time.sleep(0.1)
        status.update(state='completed', ended_at=now())
    except BaseException as exc:
        status.update(state='failed', error=str(exc), ended_at=now())
        raise
    finally:
        stop_children(children)
        for index, process in active.items():
            status['tasks'][index].update(state='stopped', exit_code=process.returncode, ended_at=now())
        for row in status['tasks']:
            if row['state'] == 'waiting':
                row.update(state='not_started', reason='allocation stopped before this task')
        save()
        stop.set()
        if monitor.ident is not None:
            monitor.join(timeout=22)
        for handle in handles:
            handle.close()
        for sig, handler in previous_handlers.items():
            signal.signal(sig, handler)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--run-tag', default=os.environ.get('MARLOWE_RUN_TAG'))
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    base = args.run_tag or 'marlowe_m13_' + re.sub(r'[-:]', '', now()[:19]) + '_' + uuid4().hex[:8]
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', base):
        parser.error('Invalid run tag')
    job_id = os.environ.get('SLURM_JOB_ID')
    restart = int(os.environ.get('SLURM_RESTART_COUNT', '0'))
    if not args.dry_run and not job_id:
        parser.error('Submit through Slurm; only --dry-run is allowed on login nodes')
    tag = unique_tag(base, job_id, restart)
    specs = tasks(tag)
    assert_unique(specs)
    assert_fresh(specs)
    directory = ROOT / ('.marlowe/m13/dryrun' if args.dry_run else 'logs/marlowe_waves') / tag
    directory.mkdir(parents=True, exist_ok=args.dry_run)
    plan = dict(group='m13', tag=tag, base_tag=base, job_id=job_id, restart_count=restart,
                training_policy='fresh_no_checkpoint_resume', seed_policy='retain_seed42_and_previous_determinism',
                launch_interval_seconds=LAUNCH_INTERVAL, max_workers=4, created_at=now(), tasks=specs)
    (directory / 'plan.json').write_text(json.dumps(plan, indent=2) + '\n')
    preflight(specs, directory)
    assert_fresh(specs)  # Dry-run/check-data must never create experiment outputs.
    if args.dry_run:
        print(directory / 'plan.json')
        return
    result = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total,memory.free',
                             '--format=csv,noheader,nounits'], capture_output=True, text=True, check=True)
    lines = result.stdout.strip().splitlines()
    if len(lines) != 1 or 'H100' not in lines[0] or float(lines[0].split(',')[-1]) < 80000:
        raise RuntimeError(f'Expected one free H100 80GB; got {result.stdout}')
    attempts = ROOT / '.marlowe/m13/attempts'
    attempts.mkdir(parents=True, exist_ok=True)
    temporary = attempts / f'{job_id}.tmp'
    temporary.write_text(json.dumps(plan, indent=2) + '\n')
    temporary.replace(attempts / f'{job_id}.json')
    print(f'GPU {lines[0]} FRESH tag={tag} restart={restart}', flush=True)
    run_wave(specs, directory)


if __name__ == '__main__':
    main()
