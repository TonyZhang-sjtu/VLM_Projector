#!/usr/bin/env python3
"""Snapshot only AnchorRead3D ablation runs into new JSON/CSV/XLSX files.

Run tags remain separate rows. Missing/incomplete evaluations are never filled
with scores from another run or treated as completed test results.
"""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.models.projectors.anchorread3d_ablation.budget import METHODS


def read_json(path):
    return json.loads(path.read_text())


def scalars(value, prefix='metric'):
    result = {}
    for key, item in value.items():
        name = f'{prefix}.{key}'
        if isinstance(item, dict):
            result.update(scalars(item, name))
        elif item is None or type(item) in (int, float, bool, str):
            result[name] = item
    return result


def base_row(task, config, run, metric_path):
    p = config['projector']
    b = p['resolved_budget']
    recipe = {k: v for k, v in config['training'].items()
              if k not in ('run_dir', 'feature_cache_dir', 'resume')}
    digest = hashlib.sha256(json.dumps(recipe, sort_keys=True).encode()).hexdigest()
    return dict(task=task, variant=b['ablation_id'], projector=p['name'],
                ratio=p['target_compression_ratio'], seed=config['seed'],
                N=b['input_tokens'], M=b['expected_output_tokens'], G=b['global_tokens'], L=b['local_tokens'],
                actual_ratio=b['actual_ratio'], actual_global_fraction=b['actual_global_fraction'],
                allocation_version=b['allocation_version'],
                children_per_parent=json.dumps(b['children_per_parent']),
                training_recipe_sha256=digest, run=str(run), metrics=str(metric_path),
                status='pending', scope=None, coverage=None, error=None)


def collect(runs_root):
    rows, manifest = [], []
    for task, filename in (('rg', 'sweep_plan.json'), ('vqa', 'launch.json')):
        for source in sorted((runs_root/task).rglob(filename)):
            spec = read_json(source)
            runs = spec['runs'] if task == 'rg' else [spec]
            for run in runs:
                config = run['config']
                if config['projector']['name'] not in METHODS:
                    continue
                if task == 'rg':
                    run_dir, path = Path(run['directory']), Path(run['metric_path'])
                else:
                    ev = run['evaluation']
                    run_dir = Path(run['run'])
                    path = Path(ev['root'])/ev['tag']/'normal'/'metrics.json'
                row = base_row(task, config, run_dir, path)
                record = dict(task=task, source=str(source), run=str(run_dir), config=config,
                              metric_path=str(path), metrics=None)
                if path.is_file():
                    try:
                        metrics = read_json(path)
                        record['metrics'] = metrics
                        row['scope'] = metrics.get('scope')
                        if task == 'rg':
                            from src.cli.baseline_sweep import validate_full_test_result
                            validate_full_test_result(metrics, run)
                            if metrics.get('status') not in ('complete', 'partial'):
                                raise ValueError('RG final evaluation has not completed')
                            row.update(status=metrics['status'], coverage=metrics['scope'],
                                       evaluated_samples=metrics['evaluated_samples'],
                                       population_size=metrics['split_population'])
                        else:
                            required = ('evaluation_coverage', 'scope', 'evaluated_subset', 'population_size',
                                        'evaluation_selection_fingerprint', 'training_scope')
                            if (metrics.get('split') != 'test' or metrics.get('control') != 'normal'
                                    or not all(key in metrics for key in required)):
                                raise ValueError('VQA test/normal evaluation metadata is missing')
                            if not metrics.get('overall'):
                                raise ValueError('VQA overall metrics are empty')
                            row.update(status='finished', coverage=metrics['evaluation_coverage'],
                                       evaluated_samples=metrics['evaluated_subset'],
                                       population_size=metrics['population_size'],
                                       selection_fingerprint=metrics['evaluation_selection_fingerprint'])
                        row.update(scalars(metrics['overall']))
                    except (ValueError, KeyError, TypeError) as error:
                        row.update(status='incomplete_or_invalid', error=str(error))
                rows.append(row)
                manifest.append(record)
    return rows, manifest


def write_snapshot(output, rows, manifest):
    from openpyxl import Workbook
    output.mkdir(parents=True, exist_ok=False)
    fields = list(dict.fromkeys(key for row in rows for key in row)) or ['task', 'variant', 'ratio', 'status']
    (output/'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+'\n')
    with (output/'results.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)
    book = Workbook()
    sheet = book.active
    sheet.title = 'Ablations'
    sheet.append(fields)
    for row in rows:
        sheet.append([row.get(key) for key in fields])
    sheet.freeze_panes = 'E2'
    sheet.auto_filter.ref = sheet.dimensions
    book.save(output/'results.xlsx')


def main():
    cli = argparse.ArgumentParser(__doc__)
    cli.add_argument('--runs-root', type=Path, default=ROOT/'runs/ablations/anchorread3d')
    cli.add_argument('--output-dir', type=Path, default=ROOT/'results/ablations/anchorread3d/summaries'/
                     datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    args = cli.parse_args()
    rows, manifest = collect(args.runs_root.resolve())
    write_snapshot(args.output_dir.resolve(), rows, manifest)
    print(f'Saved {len(rows)} run rows to {args.output_dir.resolve()}')


if __name__ == '__main__':
    main()
