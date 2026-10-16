"""Sequential BraTS joint-LoRA training and best-checkpoint evaluation.

All budgets are checked before the first model is allocated. Each ratio starts
from the pretrained assets independently; no checkpoint crosses ratio runs.
"""
import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys

from omegaconf import OmegaConf

from src.config import ROOT, load_config, save_config, validate
from src.models.projectors.specs import DEFAULTS


RATIOS = (4, 8, 16, 32)
RESERVED = (
    'project_root', 'projector.name', 'projector.target_compression_ratio',
    'projector.resolved_budget', 'projector.params.num_tokens',
    'projector.params.group_size', 'training.run_dir', 'training.resume',
    'training.init_checkpoint', 'training.stage', 'training.distributed',
    'training.device', 'data.datasets', 'evaluation.datasets',
)


def parser():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--projector', required=True, choices=sorted(DEFAULTS))
    cli.add_argument('--run-tag', help='New directory name; default: method + UTC timestamp')
    cli.add_argument('--output-root', type=Path, default=ROOT / 'runs/baselines')
    cli.add_argument('--ratios', nargs='+', type=int,
                     help='Default: 4 8 16 32; MLP supports only 1 and runs once')
    cli.add_argument('--device', help='Training and generation device; default from config (cuda:0), relative to CUDA_VISIBLE_DEVICES')
    cli.add_argument('--scoring-device', help='Offline metrics device; default: same as training')
    cli.add_argument('--eval-split', choices=['val', 'test'], default='test')
    cli.add_argument('--metrics', nargs='+', default=['all'])
    cli.add_argument('--repeats', type=int, default=1000, help='Evaluation bootstrap repeats')
    cli.add_argument('--eval-sample-fraction', type=float,
                     help='Optional fixed random evaluation fraction; default: full split')
    cli.add_argument('--config', action='append', default=[], type=Path,
                     help='Additional local YAML overlay, applied after shared presets and before --set')
    cli.add_argument('--set', action='append', default=[], dest='overrides',
                     metavar='KEY=VALUE', help='Training/config override, e.g. training.lora.rank=32')
    cli.add_argument('--dry-run', action='store_true',
                     help='Read manifests/headers, print configs and commands; no models or run files')
    cli.add_argument('--resume-run', type=Path,
                     help='Resume an existing sweep from its saved plan; completed ratios are skipped')
    return cli


def checked_ratios(args):
    ratios = args.ratios if args.ratios is not None else (
        [1] if args.projector == 'mlp2x_gelu' else list(RATIOS))
    allowed = {1} if args.projector == 'mlp2x_gelu' else set(RATIOS)
    if not ratios or len(ratios) != len(set(ratios)) or not set(ratios) <= allowed:
        raise ValueError(f'{args.projector} ratios must be unique values from {sorted(allowed)}')
    return sorted(ratios)


def check_overrides(overrides):
    for override in overrides:
        key, sep, _ = override.partition('=')
        if not sep or key != key.strip() or not key:
            raise ValueError('Use --set KEY=VALUE')
        if any(key == item or key.startswith(item + '.') or item.startswith(key + '.')
               for item in RESERVED):
            raise ValueError(f'{key} is controlled by this sweep; use its dedicated options or the generic launcher')


def build_plan(args):
    from src.cli.evaluate_checkpoint import metric_selection, select_samples
    from src.data.manifest import load_manifests
    from src.models.projectors.budget import prepare_budget, resolve_budget
    from src.training.schedule import resolve_joint_schedule
    from src.training.trainer import select_training_samples, training_budget

    if int(os.environ.get('WORLD_SIZE', '1')) != 1:
        raise ValueError('Baseline sweeps are single-GPU launchers; use launch_ddp.sh for DDP')
    ratios = checked_ratios(args)
    check_overrides(args.overrides)
    metrics = metric_selection(args.metrics)
    if args.repeats < 0:
        raise ValueError('--repeats must be nonnegative')
    fraction = args.eval_sample_fraction
    if fraction is not None and (not math.isfinite(fraction) or not 0 < fraction <= 1):
        raise ValueError('--eval-sample-fraction must be greater than 0 and at most 1')
    tag = args.run_tag or f'{args.projector}_joint_lora_{datetime.now(timezone.utc):%Y%m%dT%H%M%S%fZ}'
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', tag):
        raise ValueError('--run-tag must be a single name using letters, digits, underscore, dot or hyphen')
    run_root = args.output_root.expanduser().resolve() / tag
    if run_root.exists():
        raise FileExistsError(f'Refusing to overwrite an existing sweep: {run_root}')

    presets = [ROOT / 'configs/training/joint_lora.yaml',
               ROOT / 'configs/training/deterministic.yaml',
               ROOT / 'configs/preprocessing/native_grid.yaml',
               ROOT / f'configs/projector/{args.projector}.yaml']
    presets.extend(args.config)
    runs, samples, header_report = [], None, None
    for ratio in ratios:
        ratio_dir = run_root / f'r{ratio}'
        train_dir = ratio_dir / 'joint_lora'
        evaluation_dir = ratio_dir / 'evaluation' / f'best_{args.eval_split}'
        config_path = ratio_dir / 'training_config.yaml'
        overrides = [*args.overrides,
                     f'projector.target_compression_ratio={ratio}']
        config = load_config(presets, overrides)
        # Set paths directly instead of parsing user paths as YAML expressions.
        config.training.run_dir = str(train_dir)
        if args.device is not None:
            config.training.device = args.device
        device = config.training.device
        scoring_device = args.scoring_device or device
        if not re.fullmatch(r'cuda:\d+', device):
            raise ValueError('--device must be cuda:N')
        if not re.fullmatch(r'cpu|cuda:\d+', scoring_device):
            raise ValueError('--scoring-device must be cpu or cuda:N')
        if samples is None:
            samples, _ = load_manifests(config)
            train = select_training_samples(config, samples['train'], 'train')
            val = select_training_samples(config, samples['val'], 'val')
            if not train or not val:
                raise ValueError('Training and validation populations must both be nonempty')
            # Every ratio uses the same input/data settings. Read all selected
            # train/val headers once, then reuse only their checked geometry.
            header_report = prepare_budget(config, [*train, *val])
        else:
            geometry = header_report['budget']
            resolve_budget(config.projector, geometry['input_grid_dhw'],
                           geometry['input_extent_mm_dhw'])
        validate(config)
        windows, total = training_budget(config, len(train))
        schedule = resolve_joint_schedule(config.training, total).to_dict()
        selected = select_samples(samples[args.eval_split], sample_fraction=fraction,
                                  seed=config.seed)
        if not selected:
            raise ValueError('Evaluation population is empty')
        checkpoint = train_dir / 'checkpoints/best.pt'
        train_command = [sys.executable, '-u', '-m', 'src.cli.train', '--config', str(config_path)]
        eval_command = [sys.executable, '-u', '-m', 'src.cli.evaluate_checkpoint',
                        '--checkpoint', str(checkpoint), '--output-dir', str(evaluation_dir),
                        '--split', args.eval_split, '--device', device,
                        '--scoring-device', scoring_device, '--metrics', *metrics,
                        '--repeats', str(args.repeats)]
        if fraction is not None:
            eval_command += ['--sample-fraction', str(fraction), '--sample-seed', str(config.seed)]
        runs.append({
            'ratio': ratio, 'directory': str(ratio_dir), 'config_path': str(config_path),
            'config': OmegaConf.to_container(config, resolve=True),
            'budget': OmegaConf.to_container(config.projector.resolved_budget, resolve=True),
            'train_samples': len(train), 'val_samples': len(val),
            'updates_per_epoch': windows, 'schedule': schedule,
            'evaluation_samples': len(selected), 'evaluation_population': len(samples[args.eval_split]),
            'evaluation_scope': 'full_split' if len(selected) == len(samples[args.eval_split]) else 'fractional_sample',
            'checkpoint': str(checkpoint), 'metric_path': str(evaluation_dir / 'metric.json'),
            'train_command': train_command, 'eval_command': eval_command,
        })
    return {'version': 1, 'projector': args.projector, 'run_root': str(run_root),
            'ratios': ratios, 'eval_split': args.eval_split, 'metrics': metrics,
            'input_header_preflight': {k: v for k, v in header_report.items() if k != 'budget'},
            'runs': runs}


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def run_logged(command, log_path, append=False):
    print(shlex.join(command), flush=True)
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open('a' if append else 'w') as log:
        if append:
            log.write('\n# resumed command\n')
        log.write(shlex.join(command) + '\n')
        log.flush()
        with subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, text=True, bufsize=1) as process:
            try:
                for line in process.stdout:
                    print(line, end='', flush=True)
                    log.write(line)
                    log.flush()
                code = process.wait()
            except BaseException:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                raise
        if code:
            raise subprocess.CalledProcessError(code, command)


def load_resume_plan(path, projector):
    root = path.expanduser().resolve(strict=True)
    if not root.is_dir():
        raise ValueError(f'--resume-run must be a sweep directory: {root}')
    plan_path = root / 'sweep_plan.json'
    if not plan_path.is_file():
        raise ValueError(f'Missing saved sweep plan: {plan_path}')
    plan = json.loads(plan_path.read_text())
    if plan.get('version') != 1 or plan.get('projector') != projector:
        raise ValueError('Saved sweep plan version/projector does not match this launcher')
    if Path(plan.get('run_root', '')).resolve() != root:
        raise ValueError('Saved sweep plan run_root does not match --resume-run')
    if not plan.get('runs') or [run.get('ratio') for run in plan['runs']] != plan.get('ratios'):
        raise ValueError('Saved sweep plan has an invalid run list')
    for run in plan['runs']:
        for key in ('directory', 'config_path', 'checkpoint', 'metric_path'):
            try:
                Path(run[key]).resolve().relative_to(root)
            except (KeyError, ValueError):
                raise ValueError(f'Saved sweep {key} is outside the run directory') from None
    return plan


def training_finished(checkpoint):
    state_path = checkpoint.parent.parent / 'training_state.json'
    if not checkpoint.is_file() or not state_path.is_file():
        return False
    state = json.loads(state_path.read_text())
    return state.get('stop_reason') in ('max_epochs', 'max_steps', 'early_stopping')


def execute_plan(plan, resume=False):
    root = Path(plan['run_root'])
    if resume:
        if not root.is_dir():
            raise FileNotFoundError(f'Cannot resume missing sweep: {root}')
        state_path = root / 'sweep_status.json'
        previous = json.loads(state_path.read_text()) if state_path.is_file() else {}
        completed = [ratio for ratio in previous.get('completed_ratios', [])
                     if ratio in plan['ratios']]
        partial = [item for item in previous.get('partial_ratios', [])
                   if item.get('ratio') in completed]
        state = {'status': 'running', 'projector': plan['projector'],
                 'completed_ratios': completed, 'partial_ratios': partial,
                 'active_ratio': None, 'phase': 'resume_preflight_complete'}
    else:
        root.parent.mkdir(parents=True, exist_ok=True)
        root.mkdir()  # Atomic refusal if another process claimed this run name.
        write_json(root / 'sweep_plan.json', plan)
        for run in plan['runs']:
            save_config(OmegaConf.create(run['config']), run['config_path'])
        state = {'status': 'running', 'projector': plan['projector'], 'completed_ratios': [],
                 'partial_ratios': [], 'active_ratio': None, 'phase': 'preflight_complete'}
    try:
        for run in plan['runs']:
            if run['ratio'] in state['completed_ratios']:
                print(f'Skipping completed ratio r{run["ratio"]}.', flush=True)
                continue
            state.update(active_ratio=run['ratio'], phase='training')
            write_json(root / 'sweep_status.json', state)
            checkpoint = Path(run['checkpoint'])
            if resume and checkpoint.is_file():
                if not training_finished(checkpoint):
                    raise ValueError(
                        f'Checkpoint exists but training completion is not recorded: {checkpoint}')
                print(f'Reusing completed checkpoint for r{run["ratio"]}: {checkpoint}', flush=True)
            else:
                run_logged(run['train_command'], Path(run['directory']) / 'train_console.log',
                           append=resume)
            if not Path(run['checkpoint']).is_file():
                raise FileNotFoundError(f'Training returned without best checkpoint: {run["checkpoint"]}')
            state['phase'] = 'evaluation'
            write_json(root / 'sweep_status.json', state)
            eval_command = list(run['eval_command'])
            evaluation_dir = Path(run['metric_path']).parent
            if resume and evaluation_dir.exists() and '--resume' not in eval_command:
                eval_command.append('--resume')
            run_logged(eval_command, Path(run['directory']) / 'evaluation_console.log',
                       append=resume)
            result = json.loads(Path(run['metric_path']).read_text())
            if result.get('status') not in ('complete', 'partial'):
                raise ValueError(f'Evaluation did not complete for ratio {run["ratio"]}')
            if result['status'] == 'partial':
                partial_result = {
                    'ratio': run['ratio'],
                    'skipped_metrics': result.get('skipped_metrics', {}),
                }
                if result.get('partial_metrics'):
                    partial_result['partial_metrics'] = result['partial_metrics']
                state['partial_ratios'].append(partial_result)
            state['completed_ratios'].append(run['ratio'])
            write_json(root / 'sweep_status.json', state)
        state.update(status='partial' if state['partial_ratios'] else 'complete',
                     active_ratio=None, phase='complete')
    except BaseException as exc:
        state.update(status='failed', error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        write_json(root / 'sweep_status.json', state)
    print(f'Sweep complete: {root}', flush=True)


def main():
    cli = parser()
    args = cli.parse_args()
    try:
        plan = (load_resume_plan(args.resume_run, args.projector)
                if args.resume_run is not None else build_plan(args))
    except (ValueError, FileExistsError, FileNotFoundError) as exc:
        cli.error(str(exc))
    if args.dry_run:
        print(json.dumps(plan, indent=2))
        return
    for run in plan['runs']:
        budget = run['budget']
        print(f'{args.projector} target={run["ratio"]}x: '
              f'N={budget["input_tokens"]}, M={budget["expected_output_tokens"]}, '
              f'actual={budget["actual_ratio"]:.6f}x; schedule={run["schedule"]}', flush=True)
    execute_plan(plan, resume=args.resume_run is not None)


if __name__ == '__main__':
    main()
