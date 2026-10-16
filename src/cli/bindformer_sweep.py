"""BraTS BindFormer joint-LoRA training and best-checkpoint evaluation sweep.

Each ratio follows the same independent projector-warmup, joint-warmup, and
joint-cosine schedule as the other baseline launchers. Every ratio starts from
pretrained assets and is evaluated before the next ratio begins.
"""
import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import re
import sys

from omegaconf import OmegaConf

from src.cli.baseline_sweep import (
    check_overrides,
    execute_plan,
    load_resume_plan,
)
from src.cli.evaluate_checkpoint import metric_selection, select_samples
from src.config import ROOT, load_config, validate
from src.models.projectors.budget import prepare_budget, resolve_budget
from src.training.schedule import resolve_joint_schedule
from src.training.trainer import training_budget, select_training_samples


def parser():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--ratios', nargs='+', type=int, choices=(4, 8, 16, 32),
                     default=[4, 8, 16, 32])
    cli.add_argument('--global-fraction', type=float, default=0.25)
    cli.add_argument('--slot-iterations', type=int, default=3)
    for role in ('finding','location_laterality','signal','morphology_extent'):
        cli.add_argument('--'+role.replace('_','-')+'-queries',type=int,default=None,
                         help='Queries per slot for this role (default 1)')
    cli.add_argument('--knowledge-mode', choices=('name_definition', 'none'),
                     default='name_definition')
    cli.add_argument('--support-beta', nargs='+', type=float, default=[1.0])
    cli.add_argument('--device', default='cuda:0',
                     help='Logical CUDA device inside CUDA_VISIBLE_DEVICES')
    cli.add_argument('--scoring-device', help='Evaluation device; default matches --device')
    cli.add_argument('--run-tag')
    cli.add_argument('--output-root', type=Path, default=ROOT / 'runs/baselines')
    cli.add_argument('--eval-split', choices=('val', 'test'), default='test')
    cli.add_argument('--metrics', nargs='+', default=['all'])
    cli.add_argument('--repeats', type=int, default=1000)
    cli.add_argument('--eval-sample-fraction', type=float)
    cli.add_argument('--config', action='append', default=[], type=Path,
                     help='Additional local YAML overlay, applied after shared presets and before --set')
    cli.add_argument('--set', action='append', default=[], dest='overrides',
                     metavar='KEY=VALUE')
    cli.add_argument('--dry-run', action='store_true')
    cli.add_argument('--resume-run', type=Path)
    return cli


def build_plan(args):
    from src.data.manifest import load_manifests

    if int(os.environ.get('WORLD_SIZE', '1')) != 1:
        raise ValueError('BindFormer baseline sweeps are single-GPU launchers; use launch_ddp.sh for DDP')
    ratios = sorted(args.ratios)
    if not ratios or len(ratios) != len(set(ratios)):
        raise ValueError('--ratios must contain unique values')
    if not math.isfinite(args.global_fraction) or not 0 <= args.global_fraction < 1:
        raise ValueError('--global-fraction must be in [0,1)')
    if len(args.support_beta) not in (1, 4) or any(
            not math.isfinite(x) or x < 0 for x in args.support_beta):
        raise ValueError('--support-beta must contain one or four nonnegative finite values')
    if args.repeats < 0:
        raise ValueError('--repeats must be nonnegative')
    fraction = args.eval_sample_fraction
    if fraction is not None and (not math.isfinite(fraction) or not 0 < fraction <= 1):
        raise ValueError('--eval-sample-fraction must be greater than 0 and at most 1')
    if not re.fullmatch(r'cuda:\d+', args.device):
        raise ValueError('--device must be cuda:N')
    scoring_device = args.scoring_device or args.device
    if not re.fullmatch(r'cpu|cuda:\d+', scoring_device):
        raise ValueError('--scoring-device must be cpu or cuda:N')

    metrics = metric_selection(args.metrics)
    check_overrides(args.overrides)
    custom_overrides = [
        f'projector.params.global_fraction={args.global_fraction}',
        f'projector.params.slot_iterations={args.slot_iterations}',
        f'projector.params.knowledge_mode={args.knowledge_mode}',
        f'projector.params.support_beta={args.support_beta[0] if len(args.support_beta) == 1 else args.support_beta}',
    ]
    for role in ('finding','location_laterality','signal','morphology_extent'):
        count=getattr(args,role+'_queries')
        if count is not None:
            if count < 1:
                raise ValueError('Role query counts must be positive integers')
            custom_overrides.append(f'projector.params.role_query_counts.{role}={count}')
    tag = args.run_tag or (
        f'bindformer3d_joint_lora_{datetime.now(timezone.utc):%Y%m%dT%H%M%S%fZ}')
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', tag):
        raise ValueError('--run-tag must be a single plain directory name')
    run_root = args.output_root.expanduser().resolve() / tag
    if run_root.exists():
        raise FileExistsError(f'Refusing to overwrite an existing sweep: {run_root}')

    presets = [
        ROOT / 'configs/training/joint_lora.yaml',
        ROOT / 'configs/training/deterministic.yaml',
        ROOT / 'configs/preprocessing/native_grid.yaml',
        ROOT / 'configs/projector/bindformer3d.yaml',
    ]
    presets.extend(args.config)
    samples = None
    header_report = None
    runs = []
    for ratio in ratios:
        ratio_dir = run_root / f'r{ratio}'
        train_dir = ratio_dir / 'joint_lora'
        evaluation_dir = ratio_dir / 'evaluation' / f'best_{args.eval_split}'
        config_path = ratio_dir / 'training_config.yaml'
        config = load_config(
            presets,
            [*custom_overrides, *args.overrides,
             f'projector.target_compression_ratio={ratio}'],
        )
        config.training.run_dir = str(train_dir)
        if config.projector.params.vocabulary_scope != 'brats_train':
            raise ValueError('This BraTS sweep requires vocabulary_scope=brats_train')
        config.training.device = args.device

        if samples is None:
            samples, _ = load_manifests(config)
            train = select_training_samples(config, samples['train'], 'train')
            val = select_training_samples(config, samples['val'], 'val')
            if not train or not val:
                raise ValueError('Training and validation populations must both be nonempty')
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
        train_command = [sys.executable, '-u', '-m', 'src.cli.train',
                         '--config', str(config_path)]
        eval_command = [sys.executable, '-u', '-m', 'src.cli.evaluate_checkpoint',
                        '--checkpoint', str(checkpoint), '--output-dir', str(evaluation_dir),
                        '--split', args.eval_split, '--device', args.device,
                        '--scoring-device', scoring_device, '--metrics', *metrics,
                        '--repeats', str(args.repeats)]
        if fraction is not None:
            eval_command += ['--sample-fraction', str(fraction),
                             '--sample-seed', str(config.seed)]
        runs.append({
            'ratio': ratio,
            'directory': str(ratio_dir),
            'config_path': str(config_path),
            'config': OmegaConf.to_container(config, resolve=True),
            'budget': OmegaConf.to_container(config.projector.resolved_budget, resolve=True),
            'train_samples': len(train),
            'val_samples': len(val),
            'updates_per_epoch': windows,
            'schedule': schedule,
            'evaluation_samples': len(selected),
            'evaluation_population': len(samples[args.eval_split]),
            'evaluation_scope': 'full_split' if len(selected) == len(samples[args.eval_split])
            else 'fractional_sample',
            'checkpoint': str(checkpoint),
            'metric_path': str(evaluation_dir / 'metric.json'),
            'train_command': train_command,
            'eval_command': eval_command,
        })

    return {
        'version': 1,
        'projector': 'bindformer3d',
        'run_root': str(run_root),
        'ratios': ratios,
        'eval_split': args.eval_split,
        'metrics': metrics,
        'input_header_preflight': {k: v for k, v in header_report.items() if k != 'budget'},
        'bindformer_training_protocol': 'joint_lora',
        'runs': runs,
    }


def main():
    cli = parser()
    args = cli.parse_args()
    try:
        if args.resume_run is not None:
            plan = load_resume_plan(args.resume_run, 'bindformer3d')
        else:
            plan = build_plan(args)
    except (ValueError, FileExistsError, FileNotFoundError) as exc:
        cli.error(str(exc))
    if args.dry_run:
        print(json.dumps(plan, indent=2))
        return
    for run in plan['runs']:
        budget = run['budget']
        print(f"bindformer3d target={run['ratio']}x: "
              f"N={budget['input_tokens']}, M={budget['expected_output_tokens']}, "
              f"actual={budget['actual_ratio']:.6f}x; schedule={run['schedule']}", flush=True)
    execute_plan(plan, resume=args.resume_run is not None)


if __name__ == '__main__':
    main()
