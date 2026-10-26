"""Compare Exp1024 dry-run plans, allowing only server-local path changes."""
import argparse
from copy import deepcopy
import json
from pathlib import Path


LOCAL_PATH_KEYS = (
    'project_root', 'data.root', 'data.annotation_root', 'data.manifest_dir',
    'encoder.path', 'decoder.path', 'preprocessing.cache_dir',
    'training.feature_cache_dir', 'training.run_dir',
    'projector.params.concept_bank', 'projector.params.feature_target_cache',
    'projector.params.report_facts.path', 'projector.params.report_output.path',
)
ENTRIES = (
    'experiments/report_facts_v1/entry.py',
    'experiments/report_output_v1/entry.py',
    'results/BindFormer_role_value_adapter_r16/entry.py',
)


def normalized_config(config):
    value = deepcopy(config)
    for key in LOCAL_PATH_KEYS:
        parts = key.split('.')
        node = value
        for part in parts[:-1]:
            node = node.get(part, {})
        if parts[-1] in node:
            node[parts[-1]] = '<LOCAL_PATH>'
    return value


def normalized_command(command):
    command = list(command)
    command[0] = '<PYTHON>'
    for i, arg in enumerate(command[1:], 1):
        for entry in ENTRIES:
            if arg == entry or arg.endswith('/' + entry):
                command[i] = entry
        if command[i - 1] in ('--config', '--checkpoint', '--output-dir'):
            command[i] = '<LOCAL_PATH>'
    return command


def read_plans(directory):
    decoder = json.JSONDecoder()
    result = {'format': 'exp1024_sync_plans_v1', 'queues': {}}
    for n in range(1, 5):
        remaining = (Path(directory) / f'queue{n}.jsons').read_text()
        runs = []
        while remaining.strip():
            plan, end = decoder.raw_decode(remaining.lstrip())
            remaining = remaining.lstrip()[end:]
            for run in plan['runs']:
                record = {key: run[key] for key in (
                    'ratio', 'budget', 'train_samples', 'val_samples',
                    'manifest_summary', 'data_identity', 'updates_per_epoch',
                    'schedule', 'evaluation_samples', 'evaluation_population', 'evaluation_scope')}
                record['config'] = normalized_config(run['config'])
                record['train_command'] = normalized_command(run['train_command'])
                record['eval_command'] = normalized_command(run['eval_command'])
                record['evaluation_protocol'] = {key: plan[key] for key in
                    ('require_full_test', 'eval_split', 'metrics')}
                runs.append(record)
        if len(runs) != 3:
            raise ValueError(f'Queue {n}: expected 3 experiments, got {len(runs)}')
        result['queues'][str(n)] = runs
    return result


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--plans', type=Path, required=True)
    cli.add_argument('--reference', type=Path)
    cli.add_argument('--output', type=Path, required=True)
    args = cli.parse_args()
    result = read_plans(args.plans)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
        stream.write('\n')
    if args.reference:
        expected = json.loads(args.reference.read_text())
        if result != expected:
            raise SystemExit('Plan mismatch: compare output JSON with reference; do not start training')
        print('All 12 configurations, sample identities and train/evaluation commands match.')
    else:
        print('Recorded all 12 configurations and train/evaluation commands.')


if __name__ == '__main__':
    main()
