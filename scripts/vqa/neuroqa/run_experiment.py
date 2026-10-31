#!/usr/bin/env python3
"""Compose a separately named VQA run. Preview by default; --execute starts training."""
import argparse
from datetime import datetime, timezone
from pathlib import Path
import shlex
import subprocess
import sys
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from src.config import load_config
from src.models.projectors.budget import resolve_budget


def main():
    cli = argparse.ArgumentParser(__doc__)
    cli.add_argument('--projector', required=True)
    cli.add_argument('--config', default='configs/vqa/neuroqa/mixed.yaml')
    cli.add_argument('--ratio', type=int, required=True)
    cli.add_argument('--seed', type=int, default=42)
    cli.add_argument('--run-tag')
    cli.add_argument('--concat-device', default='cpu', help='cpu on a single 24GB GPU; cuda:1 when two GPUs are assigned')
    cli.add_argument('--set', action='append', default=[], dest='overrides')
    cli.add_argument('--execute', action='store_true')
    cli.add_argument('--evaluate-test', action='store_true')
    args = cli.parse_args()
    tag = args.run_tag or datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ_') + uuid4().hex[:6]
    if Path(tag).name != tag or tag in ('.','..'):
        cli.error('run-tag must be a single directory name')
    overrides = [f'projector.name={args.projector}', f'projector.target_compression_ratio={args.ratio}', f'seed={args.seed}']
    if args.projector == 'concat3d':
        overrides.append(f'training.projector_device={args.concat_device}')
    if args.projector == 'bindformer3d':
        overrides.append('projector.params.knowledge_mode=none')
    overrides += args.overrides
    config = load_config([ROOT / args.config], overrides)
    data_run = Path(config.data.target_version)
    if config.preprocessing.name == 'neuroqa_mni1mm_v2':
        data_run /= 'mni_v2'
    run = Path('runs/vqa/neuroqa') / data_run / 'mixed' / config.projector.name / f'r{config.projector.target_compression_ratio}' / f's{config.seed}' / tag
    overrides.append(f'training.run_dir={run}')
    budget = resolve_budget(config.projector, [49,15,13], [196,240,208])
    print(f'Run: {run}\nN={budget["input_tokens"]}, M={budget["expected_output_tokens"]}, actual ratio={budget["actual_ratio"]:.5f}', flush=True)
    command = [sys.executable, '-m', 'src.cli.vqa_train', '--config', args.config]
    for value in overrides:
        command.extend(['--set', value])
    print(shlex.join(command), flush=True)
    if args.execute:
        subprocess.run(command, cwd=ROOT, check=True)
        if args.evaluate_test:
            results = Path('results/vqa/neuroqa') / data_run / 'mixed' / config.projector.name / f'r{config.projector.target_compression_ratio}' / f's{config.seed}' / tag / 'test'
            subprocess.run([sys.executable, '-m', 'src.cli.vqa_evaluate', '--checkpoint', str(run / 'checkpoints/best.pt'),
                            '--output', str(results)], cwd=ROOT, check=True)


if __name__ == '__main__':
    main()
