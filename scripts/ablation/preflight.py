#!/usr/bin/env python3
"""Run all eight shell dry-runs and archive their actual budgets, without GPUs."""
import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import yaml

ROOT = Path(__file__).resolve().parents[2]
SUFFIXES = ('a1_no_localread', 'a2_no_mean_bypass', 'a3_global_only', 'a4_local_only')


def vqa_configs(output):
    """Extract the YAML blocks printed by the unmodified VQA dry-run CLI."""
    block = None
    configs = []
    for line in output.splitlines():
        if line == 'projector:':
            block = [line]
        elif block is not None:
            if ' -u -m src.cli.' in line:
                configs.append(yaml.safe_load('\n'.join(block)))
                block = None
            else:
                block.append(line)
    if len(configs) != 2:
        raise ValueError(f'Expected two VQA dry-run configurations, got {len(configs)}')
    return configs


def main():
    cli = argparse.ArgumentParser(__doc__)
    cli.add_argument('--output-dir', required=True, type=Path)
    cli.add_argument('--check-vqa-data', action='store_true', help='Also run one read-only v2 population/header check')
    args = cli.parse_args()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ, PROJECTOR_PYTHON=sys.executable, CUDA_VISIBLE_DEVICES='0',
               OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2')
    rows, statuses = [], []
    for task in ('rg', 'vqa'):
        for suffix in SUFFIXES:
            command = ['bash', str(ROOT/f'scripts/ablation/{task}_{suffix}.sh'), '--dry-run',
                       '--run-tag', f'preflight_{out.name}_{suffix}']
            result = subprocess.run(command, cwd=ROOT, env=env, text=True, capture_output=True)
            stem = f'{task}_{suffix}'
            (out/f'{stem}.stdout').write_text(result.stdout)
            (out/f'{stem}.stderr').write_text(result.stderr)
            statuses.append(dict(script=stem, returncode=result.returncode, command=command))
            (out/'status.json').write_text(json.dumps(statuses, indent=2)+'\n')
            if result.returncode:
                raise RuntimeError(f'{stem} failed; see {out}/{stem}.stderr')
            if task == 'rg':
                plan = json.loads(result.stdout)
                configs = [run['config'] for run in plan['runs']]
                assert plan['require_full_test']
            else:
                configs = vqa_configs(result.stdout)
            for config in configs:
                p, t = config['projector'], config['training']
                b = p['resolved_budget']
                ratio = int(p['target_compression_ratio'])
                expected = {('rg', 32): 275, ('rg', 256): 35, ('vqa', 32): 288, ('vqa', 256): 36}[task, ratio]
                assert b['expected_output_tokens'] == expected == b['global_tokens'] + b['local_tokens']
                assert t['projector_warmup_steps'] == t['joint_warmup_steps'] == 77
                assert t['early_stopping_min_delta'] == .005 and t['cosine_min_lr_ratio'] == 1
                assert t['projector_lr'] == t['encoder_lr'] == 2e-4
                assert t['effective_batch_size'] == 16 and t['microbatch_per_device'] == 1
                (out/f'{stem}_r{ratio}.yaml').write_text(yaml.safe_dump(config, sort_keys=False))
                rows.append(dict(task=task, variant=b['ablation_id'], projector=p['name'], ratio=ratio,
                                 N=b['input_tokens'], M=expected, G=b['global_tokens'], L=b['local_tokens'],
                                 actual_ratio=b['actual_ratio'], actual_global_fraction=b['actual_global_fraction'],
                                 partition=b['allocation_version'],
                                 children_per_parent=json.dumps(b['children_per_parent'])))
            print(f'PASS {stem}: r32, r256', flush=True)
    with (out/'budgets.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    if args.check_vqa_data:
        command = ['bash', str(ROOT/'scripts/ablation/vqa_a1_no_localread.sh'), '--check-data',
                   '--run-tag', f'preflight_{out.name}_data']
        result = subprocess.run(command, cwd=ROOT, env=env, text=True, capture_output=True)
        (out/'vqa_data.stdout').write_text(result.stdout)
        (out/'vqa_data.stderr').write_text(result.stderr)
        statuses.append(dict(check='vqa_data', returncode=result.returncode, command=command))
        (out/'status.json').write_text(json.dumps(statuses, indent=2)+'\n')
        if result.returncode:
            raise RuntimeError(f'VQA data readiness failed; see {out}/vqa_data.stdout')
        print('PASS VQA data readiness', flush=True)
    (out/'summary.json').write_text(json.dumps(dict(status='passed', configurations=len(rows),
                                                    gpu_allocated=False, checks=statuses), indent=2)+'\n')


if __name__ == '__main__':
    main()
