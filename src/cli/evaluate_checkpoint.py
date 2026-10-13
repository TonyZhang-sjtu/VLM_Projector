"""One command: restore checkpoint configuration, generate reports, and score.

The generator and offline scorers run in separate, sequential processes so
their dependencies and GPU allocations do not overlap.
"""
import argparse
import json
import math
import os
from pathlib import Path
import random
import re
import shlex
import shutil
import subprocess
import sys

from src.evaluation.clinical import file_digest, read_jsonl, validate_rows, write_json
from src.data.datasets import ALL_DATASETS, normalize_datasets


ALL_METRICS = ('bleu4', 'rougeL', 'green', 'radgraph', 'rate', 'bertscore', 'answer_ce')
OFFLINE_METRICS = ('green', 'radgraph', 'rate', 'bertscore')


def metric_selection(values):
    if values == ['all']:
        return list(ALL_METRICS)
    if not values or 'all' in values or len(values) != len(set(values)) or any(v not in ALL_METRICS for v in values):
        raise ValueError('Use --metrics all or a unique space-separated list: ' + ' '.join(ALL_METRICS))
    return values


def select_samples(samples, ids=None, limit=None, sample_fraction=None, seed=42):
    if limit is not None and limit < 1:
        raise ValueError('--limit must be positive')
    if limit is not None and sample_fraction is not None:
        raise ValueError('--limit and --sample-fraction are mutually exclusive')
    if sample_fraction is not None and (
            not math.isfinite(sample_fraction) or not 0 < sample_fraction <= 1):
        raise ValueError('--sample-fraction must be greater than 0 and at most 1')
    if ids is not None:
        lookup = {s.id: s for s in samples}
        if (not isinstance(ids, list) or not ids or not all(isinstance(i, str) for i in ids)
                or len(ids) != len(set(ids)) or any(i not in lookup for i in ids)):
            raise ValueError('IDs must be unique and belong to the selected split')
        samples = [lookup[i] for i in ids]
    if limit is not None:
        return samples[:limit]
    if sample_fraction is not None and samples:
        sample_count = max(1, math.floor(len(samples) * sample_fraction + 0.5))
        selected = set(random.Random(seed).sample(range(len(samples)), sample_count))
        # Keep manifest/ID-file ordering stable while the selected subset is random.
        samples = [sample for index, sample in enumerate(samples) if index in selected]
    return samples


def checkpoint_config(path, args):
    import torch
    from omegaconf import OmegaConf
    from src.config import validate
    # Only project-generated local checkpoints: NumPy RNG/optimizer metadata
    # require the project's existing full-pickle loading convention.
    saved = torch.load(path, map_location='cpu', weights_only=False, mmap=True)
    expected_format = 2 if saved.get('stage') == 'joint_lora' else 1
    if saved.get('format_version') != expected_format or saved['stage'] != saved['config']['training']['stage']:
        raise ValueError('Unsupported or inconsistent project checkpoint')
    config = OmegaConf.create(saved['config'])
    # Pre-selector checkpoints were trained against all three source datasets.
    if config.data.get('datasets') is None:
        config.data.datasets = list(ALL_DATASETS)
    else:
        config.data.datasets = normalize_datasets(config.data.datasets)
    if not config.evaluation.get('datasets'):
        config.evaluation.datasets = None
    if getattr(args, 'datasets', None) is not None:
        config.evaluation.datasets = normalize_datasets(args.datasets)
    config.training.device = args.device
    config.training.distributed = 'none'
    config.evaluation.split = args.split
    config.evaluation.checkpoint = str(path)
    config.evaluation.output_dir = str(args.output_dir / 'generation')
    # If the clinical step is present it computes corpus bootstrap once; avoid
    # doing the same expensive lexical bootstrap in both subprocesses.
    config.evaluation.bootstrap_repeats = 0 if any(m in OFFLINE_METRICS for m in args.metrics) else args.repeats
    validate(config)
    return config


def make_commands(args, config_path, ids_path):
    generation = [sys.executable, '-m', 'src.cli.evaluate', '--config', str(config_path),
                  '--ids-file', str(ids_path)]
    if 'answer_ce' not in args.metrics:
        generation.append('--skip-answer-ce')
    metrics = [m for m in args.metrics if m in OFFLINE_METRICS]
    scoring = None
    if metrics:
        scoring = [sys.executable, '-m', 'src.cli.score_reports',
                   '--predictions', str(args.output_dir / 'generation/predictions.jsonl'),
                   '--expected-ids', str(ids_path), '--output-dir', str(args.output_dir / 'scoring'),
                   '--device', args.scoring_device or args.device, '--repeats', str(args.repeats),
                   '--metrics', *metrics]
    return generation, scoring


def combine_metrics(generated, scored, metrics):
    overall = {}
    for metric in metrics:
        if metric in ('bleu4', 'rougeL'):
            overall[metric] = generated['overall'][metric]
        elif metric == 'answer_ce':
            overall[metric] = generated['answer_ce']['answer_ce']
        else:
            from src.evaluation.clinical import METRIC_KEYS
            for key in METRIC_KEYS[metric]:
                overall[key] = scored['clinical']['overall'][key]['mean']
    if generated['status'] != 'complete' or (scored is not None and scored['status'] != 'complete'):
        raise ValueError('Cannot publish complete metrics from a failed evaluation stage')
    if any(v is None for v in overall.values()):
        raise ValueError('Selected metric missing')
    return overall


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--checkpoint', type=Path, required=True)
    cli.add_argument('--output-dir', type=Path, required=True, help='New output directory; never overwrite an existing experiment')
    cli.add_argument('--split', choices=['train', 'val', 'test'], default='test')
    cli.add_argument('--datasets', nargs='+', metavar='DATASET',
                     help='Evaluation scope: brats, isles, wmh, all, or a combination; defaults to checkpoint training scope')
    cli.add_argument('--device', required=True, help='VLM visible GPU index, e.g. cuda:2')
    cli.add_argument('--scoring-device', help='Optional different device for offline scorers, e.g. cuda:1 or cpu')
    cli.add_argument('--metrics', nargs='+', default=['all'],
                     help='Default: all (including BERTScore); or any of: ' + ' '.join(ALL_METRICS))
    cli.add_argument('--repeats', type=int, default=1000)
    subset = cli.add_mutually_exclusive_group()
    subset.add_argument('--limit', type=int, help='Evaluate the first N samples')
    subset.add_argument('--sample-fraction', type=float,
                        help='Deterministically sample this fraction of the selected split (for example 0.3)')
    cli.add_argument('--sample-seed', type=int,
                     help='Override the checkpoint seed used for --sample-fraction')
    cli.add_argument('--ids-file', type=Path, help='Optional JSON array of IDs in the requested split')
    cli.add_argument('--dry-run', action='store_true', help='Check checkpoint metadata/manifests and display plan without loading VLM/scorers')
    args = cli.parse_args()
    try:
        args.metrics = metric_selection(args.metrics)
    except ValueError as exc:
        cli.error(str(exc))
    if not re.fullmatch(r'cuda:\d+', args.device):
        cli.error('VLM evaluation requires --device cuda:N')
    if args.scoring_device and not re.fullmatch(r'cpu|cuda:\d+', args.scoring_device):
        cli.error('--scoring-device must be cpu or cuda:N')
    if args.repeats < 0 or (args.limit is not None and args.limit < 1):
        cli.error('Invalid repeats or limit')
    if args.sample_seed is not None and args.sample_fraction is None:
        cli.error('--sample-seed requires --sample-fraction')
    if args.sample_fraction is not None and (
            not math.isfinite(args.sample_fraction) or not 0 < args.sample_fraction <= 1):
        cli.error('--sample-fraction must be greater than 0 and at most 1')
    args.checkpoint = args.checkpoint.resolve(strict=True)
    args.output_dir = args.output_dir.resolve()
    if args.output_dir.exists():
        raise FileExistsError(f'Use a new output directory: {args.output_dir}')
    checkpoint = args.checkpoint
    if not args.dry_run:
        args.output_dir.mkdir(parents=True)
        checkpoint = args.output_dir / 'checkpoint.pt'
        # Training writes best/last atomically. A hard link preserves the selected
        # inode if training subsequently replaces it, without duplicating storage.
        try:
            os.link(args.checkpoint, checkpoint)
        except OSError:
            shutil.copy2(args.checkpoint, checkpoint)
    try:
        config = checkpoint_config(checkpoint, args)
        from src.config import save_config
        from src.data.manifest import load_manifests
        populations, manifest_summary = load_manifests(config, datasets=config.evaluation.datasets)
        ids = json.loads(args.ids_file.read_text()) if args.ids_file else None
        sample_seed = args.sample_seed if args.sample_seed is not None else config.seed
        samples = select_samples(populations[args.split], ids, args.limit,
                                 args.sample_fraction, sample_seed)
        config_path = args.output_dir / 'evaluation_config.yaml'
        ids_path = args.output_dir / 'expected_ids.json'
        commands = make_commands(args, config_path, ids_path)
        if commands[1] is not None:
            commands[1].extend(['--seed', str(config.seed)])
        plan = {'status': 'planned', 'checkpoint': str(args.checkpoint),
                'checkpoint_sha256': file_digest(checkpoint), 'stage': config.training.stage,
                'split': args.split, 'split_population': len(populations[args.split]),
                'training_dataset_scope': list(config.data.datasets),
                'evaluation_dataset_scope': manifest_summary[args.split]['datasets'],
                'split_dataset_counts': manifest_summary[args.split]['dataset_counts'],
                'evaluated_samples': len(samples), 'patients': len({s.patient_id for s in samples}),
                'scope': ('full_split' if len(samples) == len(populations[args.split]) else
                          'fractional_sample' if args.sample_fraction is not None else 'subset_smoke'),
                'sample_fraction': args.sample_fraction,
                'sample_seed': sample_seed if args.sample_fraction is not None else None,
                'metrics': args.metrics, 'device': args.device,
                'scoring_device': args.scoring_device or args.device, 'manifest_summary': manifest_summary,
                'commands': [cmd for cmd in commands if cmd is not None]}
        print(json.dumps({k:v for k,v in plan.items() if k not in ('manifest_summary', 'commands')}, indent=2), flush=True)
        for command in plan['commands']:
            print(shlex.join(command), flush=True)
        if args.dry_run:
            return
        write_json(args.output_dir / 'pipeline.json', dict(plan, status='running'))
        save_config(config, config_path)
        write_json(ids_path, [s.id for s in samples])
        env = dict(os.environ)
        env['PYTHONNOUSERSITE'] = '1'
        env.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
        env.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
        for name, command in zip(('generation', 'scoring'), commands):
            if command is None:
                continue
            print(f'Running {name}; GPU memory is released before the next stage.', flush=True)
            subprocess.run(command, check=True, env=env)
            if name == 'generation':
                if file_digest(checkpoint) != plan['checkpoint_sha256']:
                    raise ValueError('Checkpoint changed in place during evaluation')
                rows = read_jsonl(args.output_dir / 'generation/predictions.jsonl')
                validate_rows(rows, [s.id for s in samples])
                lookup = {s.id:s for s in samples}
                for row in rows:
                    sample = lookup[row['id']]
                    if (row['reference'] != sample.finding or row['checkpoint_id'] != plan['checkpoint_sha256']
                            or any(row[k] != getattr(sample, k) for k in ('patient_id', 'case_id', 'cohort', 'sequence', 'split'))):
                        raise ValueError('Generated prediction metadata does not match checkpoint/manifest')
        generated = json.loads((args.output_dir / 'generation/metrics.json').read_text())
        scored = json.loads((args.output_dir / 'scoring/metrics.json').read_text()) if commands[1] else None
        result = dict(plan, status='complete', overall=combine_metrics(generated, scored, args.metrics))
        result.pop('commands')
        result['details'] = {'generation': 'generation/metrics.json',
                             'scoring': 'scoring/metrics.json' if scored else None,
                             'predictions': 'generation/predictions.jsonl'}
        result['bootstrap'] = {'lexical': (scored or generated)['bootstrap'],
                               'clinical': scored['clinical']['bootstrap'] if scored else None}
        write_json(args.output_dir / 'metric.json', result)
        write_json(args.output_dir / 'pipeline.json', dict(plan, status='complete'))
        print(json.dumps({'status': 'complete', 'overall': result['overall'],
                          'output': str(args.output_dir / 'metric.json')}, indent=2))
    except Exception as exc:
        if not args.dry_run and args.output_dir.exists():
            write_json(args.output_dir / 'pipeline.json', {'status':'failed', 'checkpoint':str(args.checkpoint),
                       'error_type':type(exc).__name__, 'error':str(exc)})
        raise


if __name__ == '__main__':
    main()
