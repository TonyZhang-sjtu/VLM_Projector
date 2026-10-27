"""Compare every Exp1026 launcher's resolved plan across server-local paths.

Generate one <launcher_stem>.json per shell using --dry-run first. This tool
loads no model and includes no sample IDs or report text in its output.
"""
import argparse
import hashlib
import json
from pathlib import Path

from check_sync_plans import normalized_command, normalized_config


ROOT = Path(__file__).resolve().parents[1]


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def summarize(plans):
    result = {'format': 'exp1026_sync_v1', 'launchers': {}}
    scripts = sorted((ROOT / 'scripts/Exp1026').glob('*.sh'))
    if not scripts:
        raise ValueError('No Exp1026 launchers found')
    for script in scripts:
        plan = json.loads((plans / f'{script.stem}.json').read_text())
        identity = {key: plan[key] for key in (
            'projector', 'ratios', 'require_full_test', 'eval_split', 'metrics')}
        identity['runs'] = []
        for run in plan['runs']:
            record = {key: run[key] for key in (
                'ratio', 'budget', 'train_samples', 'val_samples', 'manifest_summary',
                'data_identity', 'updates_per_epoch', 'schedule', 'evaluation_samples',
                'evaluation_population', 'evaluation_scope')}
            record['config'] = normalized_config(run['config'])
            record['train_command'] = normalized_command(run['train_command'])
            record['eval_command'] = normalized_command(run['eval_command'])
            record['checkpoint_filename'] = Path(run['checkpoint']).name
            identity['runs'].append(record)
        result['launchers'][script.name] = {
            'launcher_sha256': hashlib.sha256(script.read_bytes()).hexdigest(),
            'normalized_plan_sha256': digest(identity),
            'projector': plan['projector'], 'ratios': plan['ratios'],
            'tokens': [run['budget']['expected_output_tokens'] for run in plan['runs']],
            'train_val_test': [[run['train_samples'], run['val_samples'], run['evaluation_samples']]
                               for run in plan['runs']],
        }
    return result


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--plans', type=Path, required=True)
    cli.add_argument('--reference', type=Path)
    cli.add_argument('--output', type=Path, required=True, help='New local evidence JSON')
    args = cli.parse_args()
    result = summarize(args.plans)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
        stream.write('\n')
    if args.reference:
        expected = json.loads(args.reference.read_text())
        if expected.get('format') != result['format']:
            raise SystemExit('Unsupported reference format')
        names = set(result['launchers']) | set(expected['launchers'])
        different = sorted(name for name in names
                           if result['launchers'].get(name) != expected['launchers'].get(name))
        if different:
            raise SystemExit('Launcher/plan mismatch: ' + ', '.join(different))
    count = sum(len(item['ratios']) for item in result['launchers'].values())
    verb = 'match the reference' if args.reference else 'recorded'
    print(f'{len(result["launchers"])} launchers / {count} experiments {verb}.')


if __name__ == '__main__':
    main()
