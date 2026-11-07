"""M10--M12: at most four independent train-and-test workers on one H100."""
import argparse
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import threading
import time

from rerun_wave import ROOT, OUR, attempt_tag, gpu_monitor, now, stop_children

MAX_WORKERS = 4
GROUPS = {
    'm10': [('rg', 'convpool3d', 8), ('rg', 'convpool3d', 16),
            ('rg', OUR, 8), ('rg', OUR, 16)],
    'm11': [('vqa', 'convpool3d', 256), ('vqa', 'convpool3d', 128),
            ('vqa', 'convpool3d', 64), ('rg', 'convpool3d', 256),
            ('rg', 'convpool3d', 128)],
    'm12': [('vqa', 'convpool3d', 32), ('vqa', 'convpool3d', 16),
            ('vqa', 'convpool3d', 8), ('rg', 'convpool3d', 64),
            ('rg', 'convpool3d', 32)],
}


def tasks(group, tag):
    specs = []
    for index, (task, method, ratio) in enumerate(GROUPS[group]):
        label = f'{task}_{method}_r{ratio}'
        if task == 'vqa':
            script = ROOT / 'scripts/Exp_VQA/vqa_convpool3d.sh'
            run_tag = tag
            relative = Path('vqa/neuroqa/official_holdout_patient_guard_v1/v2/unreviewed_training/mixed') / method / f'r{ratio}' / 's42' / tag
            run = ROOT / 'runs' / relative
            metrics = ROOT / 'results' / relative / 'test/normal/metrics.json'
            train = run / 'train/train.jsonl'
        else:
            script = ROOT / 'scripts/Exp1026' / (
                'run_baseline_adirect_localread.sh' if method == OUR else 'run_convpool3d.sh')
            run_tag = f'{tag}_{label}'
            run = ROOT / 'runs/baselines' / run_tag / f'r{ratio}'
            metrics = run / 'evaluation/best_test/metric.json'
            train = run / 'joint_lora/train.jsonl'
        specs.append(dict(label=label, task=task, method=method, ratio=ratio,
                          launch_order=index, initial=index < MAX_WORKERS,
                          command=['bash', str(script), '--ratios', str(ratio), '--run-tag', run_tag],
                          run=str(run), train_log=str(train), metrics=str(metrics)))
    return specs


def assert_fresh(specs):
    for spec in specs:
        # Also reject an existing RG sweep root or VQA evaluation directory.
        run = Path(spec['run'])
        output = run.parent if spec['task'] == 'rg' else Path(spec['metrics']).parents[2]
        for path in (run, output):
            if path.exists():
                raise FileExistsError(f'Refusing to overwrite existing experiment: {path}')


def preflight(specs, directory):
    checked_data = False
    summary = []
    for spec in specs:
        log = directory / f"{spec['label']}.plan.log"
        with log.open('w') as handle:
            subprocess.run(spec['command'] + ['--dry-run'], cwd=ROOT, stdout=handle,
                           stderr=subprocess.STDOUT, check=True)
        content = log.read_text()
        if spec['task'] == 'rg':
            plan = json.loads(content)
            assert plan['projector'] == spec['method'] and plan['ratios'] == [spec['ratio']]
            assert plan['require_full_test'] and len(plan['runs']) == 1
            run = plan['runs'][0]
            assert run['directory'] == spec['run'] and run['metric_path'] == spec['metrics']
            config = run['config']
            assert config['training']['resume'] is None and config['training']['init_checkpoint'] is None
            assert config['training']['effective_batch_size'] == 16
            assert config['training']['microbatch_per_device'] == 1
            summary.append(dict(label=spec['label'], tokens=run['budget']['expected_output_tokens'],
                                train_samples=run['train_samples'], val_samples=run['val_samples'],
                                test_samples=run['evaluation_samples'], run=spec['run'], metrics=spec['metrics']))
        else:
            header, _ = json.JSONDecoder().raw_decode(content.lstrip())
            assert header['run'] == spec['run']
            assert str(Path(spec['metrics']).parent) in content
            assert header['preprocessing'] == 'v2'
            assert header['invalid_sample_policy'] == 'include_unreviewed'
            for key in ('cache', 'manifest'):
                path = Path(header[key])
                assert (path if path.is_absolute() else ROOT / path).is_dir(), path
            if not checked_data:
                with (directory / 'vqa_data_check.log').open('w') as handle:
                    subprocess.run(spec['command'] + ['--check-data'], cwd=ROOT, stdout=handle,
                                   stderr=subprocess.STDOUT, check=True)
                checked_data = True
            summary.append(dict(label=spec['label'], tokens=header['M'], run=spec['run'],
                                metrics=spec['metrics'], manifest=header['manifest'], cache=header['cache']))
        print(f"CHECKED {spec['label']}", flush=True)
    (directory / 'preflight_summary.json').write_text(json.dumps(summary, indent=2) + '\n')


def run_queue(specs, directory):
    """A slot is occupied until the entire child sweep (including test) exits."""
    children, handles, active = [], [], {}
    stop = threading.Event()
    monitor = threading.Thread(target=gpu_monitor, args=(directory / 'gpu.jsonl', stop), daemon=True)
    status = dict(started_at=now(), job_id=os.environ['SLURM_JOB_ID'], state='running',
                  max_workers=MAX_WORKERS, restart_count=int(os.environ.get('SLURM_RESTART_COUNT', '0')),
                  training_policy='fresh_no_checkpoint_resume',
                  tasks=[dict(**spec, state='waiting') for spec in specs])
    previous_handlers = {}

    def save():
        temporary = directory / 'status.tmp'
        temporary.write_text(json.dumps(status, indent=2) + '\n')
        temporary.replace(directory / 'status.json')

    def interrupted(signum, frame):
        raise KeyboardInterrupt(f'signal {signum}')

    def launch(index):
        spec = specs[index]
        assert_fresh([spec])
        handle = (directory / f"{spec['label']}.log").open('a')
        handles.append(handle)
        process = subprocess.Popen(spec['command'], cwd=ROOT, stdout=handle,
                                   stderr=subprocess.STDOUT, start_new_session=True)
        children.append(process)
        active[index] = process
        status['tasks'][index].update(state='running', pid=process.pid, started_at=now())
        print(f"START {spec['label']} pid={process.pid} active={len(active)}/{MAX_WORKERS}", flush=True)
        save()

    try:
        for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
            previous_handlers[sig] = signal.signal(sig, interrupted)
        monitor.start()
        save()
        next_index = 0
        while active or next_index < len(specs):
            # Reap ALL exits and handle failures before refilling any slot.
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
            while len(active) < MAX_WORKERS and next_index < len(specs):
                launch(next_index)
                next_index += 1
            if active:
                time.sleep(1)
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
    parser.add_argument('group', choices=GROUPS)
    parser.add_argument('--run-tag', default=os.environ.get('MARLOWE_RUN_TAG'))
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    base_tag = args.run_tag or f'marlowe_{args.group}_' + re.sub(r'[-:]', '', now()[:19])
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', base_tag):
        parser.error('Invalid run tag')
    job_id = os.environ.get('SLURM_JOB_ID')
    restart_count = int(os.environ.get('SLURM_RESTART_COUNT', '0'))
    if not args.dry_run and not job_id:
        parser.error('Submit through Slurm; only --dry-run is allowed on login nodes')
    tag = attempt_tag(base_tag, job_id, restart_count)
    specs = tasks(args.group, tag)
    assert_fresh(specs)
    directory = ROOT / ('.marlowe/m10_m12/dryrun' if args.dry_run else 'logs/marlowe_waves') / tag
    directory.mkdir(parents=True, exist_ok=args.dry_run)
    plan = dict(group=args.group, tag=tag, base_tag=base_tag, job_id=job_id,
                restart_count=restart_count, training_policy='fresh_no_checkpoint_resume',
                max_workers=MAX_WORKERS, created_at=now(), tasks=specs)
    (directory / 'plan.json').write_text(json.dumps(plan, indent=2) + '\n')
    preflight(specs, directory)
    if args.dry_run:
        print(directory / 'plan.json')
        return
    result = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total,memory.free',
                             '--format=csv,noheader,nounits'], capture_output=True, text=True, check=True)
    lines = result.stdout.strip().splitlines()
    if len(lines) != 1 or 'H100' not in lines[0] or float(lines[0].split(',')[-1]) < 80000:
        raise RuntimeError(f'Expected one free H100 80GB; got {result.stdout}')
    attempts = ROOT / '.marlowe/m10_m12/attempts'
    attempts.mkdir(parents=True, exist_ok=True)
    temporary = attempts / f'{job_id}.tmp'
    temporary.write_text(json.dumps(plan, indent=2) + '\n')
    temporary.replace(attempts / f'{job_id}.json')
    print(f'GPU {lines[0]} FRESH tag={tag} restart={restart_count}', flush=True)
    run_queue(specs, directory)


if __name__ == '__main__':
    main()
