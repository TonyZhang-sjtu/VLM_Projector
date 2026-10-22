"""Run-local evaluation overrides, including handoff from an already running launcher.

Training never reads this policy. A legacy launcher has its evaluation list in
memory; its freshly started evaluation child can decline a removed checkpoint
before model loading. A detached supervisor then waits for that launcher to exit
and resumes *evaluation only*. No signals are sent to training or other processes.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

from src.config import ROOT

POLICY_FILE = 'evaluation_policy.json'


def write_atomic(path, value):
    path = Path(path)
    with tempfile.NamedTemporaryFile(mode='w', dir=path.parent, delete=False) as f:
        json.dump(value, f, indent=2, ensure_ascii=False)
        f.write('\n')
        temporary = Path(f.name)
    temporary.replace(path)


def read_policy(root):
    root = Path(root).resolve()
    path = root / POLICY_FILE
    if not path.exists():
        return None
    policy = json.loads(path.read_text())
    if (policy.get('version') != 1 or policy.get('run_root') != str(root)
            or policy.get('evaluation_checkpoint_epochs') != []
            or policy.get('allowed_checkpoint_labels') != ['best']):
        raise ValueError(f'Invalid best-only evaluation policy: {path}')
    actual = hashlib.sha256((root / 'sweep_plan.json').read_bytes()).hexdigest()
    if policy.get('training_plan_sha256') != actual:
        raise ValueError('Evaluation override does not match this saved training plan')
    return policy


def apply_policy(plan):
    if not read_policy(plan['run_root']):
        return False
    from src.cli.ft_experiments import select_evaluations
    select_evaluations(plan, [])
    return True


def process_identity(pid):
    """Linux PID plus start ticks prevents confusing PID reuse with the launcher."""
    try:
        fields = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
        return {'state': fields[0], 'start_ticks': fields[19]}
    except FileNotFoundError:
        return None


def legacy_handoff(args):
    """Called before any checkpoint/model loading in evaluate_checkpoint.main."""
    checkpoint = args.checkpoint.expanduser().resolve()
    if len(checkpoint.parents) < 4:
        return
    root = checkpoint.parents[3]
    policy = read_policy(root)
    if policy is None or checkpoint.name == 'best.pt':
        return
    plan = json.loads((root / 'sweep_plan.json').read_text())
    if plan.get('runner') != 'ft_experiments' or len(plan['runs']) != 1:
        raise ValueError('Runtime evaluation policy requires an N-series run')
    run = plan['runs'][0]
    matching = [job for job in run['evaluations'] if Path(job['checkpoint']).resolve() == checkpoint
                and Path(job['directory']).resolve() == args.output_dir.expanduser().resolve()]
    if len(matching) != 1 or args.split != plan['eval_split']:
        raise ValueError('Removed evaluation request does not match the original run plan')
    if args.dry_run:
        print(json.dumps({'status': 'skipped_by_runtime_policy', 'checkpoint': str(checkpoint),
                          'allowed_checkpoint_labels': ['best']}, indent=2))
        raise SystemExit(0)
    from src.cli.baseline_sweep import training_finished
    if not training_finished(Path(run['checkpoint'])):
        raise ValueError('Refusing evaluation handoff before training completion')
    parent = os.getppid()
    identity = process_identity(parent)
    argv = Path(f'/proc/{parent}/cmdline').read_bytes().decode().split('\0')
    if ('src.cli.ft_experiments' not in argv
            or not (str(root) in argv or root.name in argv)
            or identity is None):
        raise ValueError('Best-only policy: resume via the matching nN.sh instead of this removed checkpoint')
    log_path = root / 'evaluation_best_only_handoff.log'
    command = [sys.executable, '-u', '-m', 'src.cli.ft_evaluation_policy',
               '--handoff', str(root), '--wait-pid', str(parent),
               '--wait-start-ticks', identity['start_ticks']]
    print(f'User requested best only: skipping {checkpoint.name} before model loading. '
          f'Training is complete. Evaluation will resume in {log_path}.', flush=True)
    with log_path.open('a') as log:
        log.write('\n# Best-only handoff from a legacy in-memory evaluation list\n')
        log.flush()
        process = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL,
                                   stdout=log, stderr=subprocess.STDOUT, close_fds=True,
                                   start_new_session=True)
    # The old launcher's run_logged raises on 75, then records its exit. Its
    # training child has already finished. The new supervisor waits for that
    # final status write before starting a best-only resume, avoiding races.
    print(f'Intentional evaluation handoff (exit 75), supervisor PID={process.pid}.', flush=True)
    raise SystemExit(75)


def handoff_main(args):
    root = args.handoff.expanduser().resolve(strict=True)
    lock = (root / '.evaluation_handoff.lock').open('a')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print('Another best-only handoff supervisor already owns this run.', flush=True)
        return
    status_path = root / 'evaluation_handoff.json'
    status = {'status': 'waiting_for_previous_launcher', 'supervisor_pid': os.getpid(),
              'previous_launcher_pid': args.wait_pid, 'allowed_checkpoint_labels': ['best'],
              'started_at': time.strftime('%Y-%m-%dT%H:%M:%S%z')}
    write_atomic(status_path, status)
    try:
        read_policy(root)
        deadline = time.monotonic() + 180
        while True:
            identity = process_identity(args.wait_pid)
            if (identity is None or identity['start_ticks'] != args.wait_start_ticks
                    or identity['state'] == 'Z'):
                break
            if time.monotonic() >= deadline:
                raise TimeoutError('Previous launcher has not exited; refusing concurrent evaluation')
            time.sleep(.5)
        from src.cli.baseline_sweep import training_finished
        plan = json.loads((root / 'sweep_plan.json').read_text())
        run = plan['runs'][0]
        if not training_finished(Path(run['checkpoint'])):
            raise ValueError('Training completion is missing; no training restart is permitted')
        command = [sys.executable, '-u', '-m', 'src.cli.ft_experiments',
                   '--projector', 'avgpool3d', '--resume-run', str(root),
                   '--eval-checkpoint-epochs', '--require-full-test']
        status.update(status='evaluating_best', command=command,
                      cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'))
        write_atomic(status_path, status)
        print('Training already completed; starting only final best.pt evaluation.', flush=True)
        result = subprocess.run(command, cwd=ROOT)
        if result.returncode:
            raise subprocess.CalledProcessError(result.returncode, command)
        status.update(status='complete', finished_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'))
        write_atomic(status_path, status)
    except BaseException as exc:
        status.update(status='failed', error_type=type(exc).__name__, error=str(exc))
        write_atomic(status_path, status)
        raise
    finally:
        lock.close()


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--handoff', type=Path, required=True)
    cli.add_argument('--wait-pid', type=int, required=True)
    cli.add_argument('--wait-start-ticks', required=True)
    handoff_main(cli.parse_args())
