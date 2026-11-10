"""M18: four RG16 repeats; retain M13's recipe except deterministic=false."""
import argparse
from copy import deepcopy
import json
import os
import re
import subprocess
from uuid import uuid4

import yaml

from convpool_wave import assert_fresh, preflight
from repeat_wave import LAUNCH_INTERVAL, assert_unique, run_wave, unique_tag
from rerun_wave import ROOT, OUR, now

REFERENCE = ROOT / (
    'runs/baselines/marlowe_m13_20261107T163051_1ff34688_j560041_'
    'rg_bindformer_adirect_localread_r16_rep01/r16/joint_lora/resolved_config.yaml'
)


def tasks(tag):
    specs = []
    for repeat in range(1, 5):
        label = f'rg_{OUR}_r16_rep{repeat:02d}'
        run_tag = f'{tag}_{label}'
        run = ROOT / 'runs/baselines' / run_tag / 'r16'
        specs.append(dict(
            label=label, task='rg', method=OUR, ratio=16, repeat=repeat,
            seed=42, deterministic=False, run_tag=run_tag,
            launch_order=repeat - 1, initial=True,
            command=['bash', str(ROOT / 'scripts/Exp1026/run_baseline_adirect_localread.sh'),
                     '--ratios', '16', '--run-tag', run_tag,
                     '--set', 'seed=42', '--set', 'training.deterministic=false'],
            run=str(run), train_log=str(run / 'joint_lora/train.jsonl'),
            metrics=str(run / 'evaluation/best_test/metric.json')))
    return specs


def verify_recipe(specs, directory):
    reference = yaml.safe_load(REFERENCE.read_text())
    if reference['seed'] != 42 or reference['training']['deterministic'] is not True:
        raise ValueError('Unexpected historical M13 reference recipe')
    verification = []
    for spec in specs:
        plan = json.loads((directory / f"{spec['label']}.plan.log").read_text())
        child = plan['runs'][0]
        actual = child['config']
        expected = deepcopy(reference)
        expected['training']['deterministic'] = False
        expected['training']['run_dir'] = actual['training']['run_dir']
        if actual != expected:
            sections = [k for k in set(actual) | set(expected)
                        if actual.get(k) != expected.get(k)]
            raise ValueError(f"{spec['label']}: unexpected recipe change in {sections}")
        if actual['training']['run_dir'] != str(ROOT / spec['run'] / 'joint_lora'):
            raise ValueError('Training output differs from planned output')
        if child['budget']['expected_output_tokens'] != 549:
            raise ValueError('RG16 must retain 549 visual tokens')
        if child['evaluation_samples'] != 560 or plan['eval_split'] != 'test':
            raise ValueError('Expected the original full 560-report test population')
        verification.append(dict(label=spec['label'], seed=actual['seed'],
            deterministic=actual['training']['deterministic'],
            matches_m13_except_deterministic_and_run_dir=True,
            resume=actual['training']['resume'], init_checkpoint=actual['training']['init_checkpoint'],
            train_samples=child['train_samples'], val_samples=child['val_samples'],
            test_samples=child['evaluation_samples'], visual_tokens=549,
            run=spec['run'], metrics=spec['metrics']))
    (directory / 'recipe_verification.json').write_text(json.dumps(dict(
        reference=str(REFERENCE), changed_setting={'training.deterministic': [True, False]},
        tasks=verification), indent=2) + '\n')


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--run-tag', default=os.environ.get('MARLOWE_RUN_TAG'))
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    base = args.run_tag or 'marlowe_m18_' + re.sub(r'[-:]', '', now()[:19]) + '_' + uuid4().hex[:8]
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', base):
        parser.error('Invalid run tag')
    job_id = os.environ.get('SLURM_JOB_ID')
    restart = int(os.environ.get('SLURM_RESTART_COUNT', '0'))
    if not args.dry_run and not job_id:
        parser.error('Submit through Slurm; only --dry-run is allowed on login nodes')
    tag = unique_tag(base, job_id, restart)
    specs = tasks(tag)
    assert_unique(specs)
    assert_fresh(specs)
    directory = ROOT / ('.marlowe/m18/dryrun' if args.dry_run else 'logs/marlowe_waves') / tag
    directory.mkdir(parents=True, exist_ok=args.dry_run)
    plan = dict(group='m18', tag=tag, base_tag=base, job_id=job_id, restart_count=restart,
        training_policy='fresh_no_checkpoint_resume', seed=42, deterministic=False,
        launch_interval_seconds=LAUNCH_INTERVAL, max_workers=4,
        reference_config=str(REFERENCE), created_at=now(), tasks=specs)
    (directory / 'plan.json').write_text(json.dumps(plan, indent=2) + '\n')
    preflight(specs, directory)
    verify_recipe(specs, directory)
    assert_fresh(specs)
    if args.dry_run:
        print(directory / 'plan.json')
        return
    result = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total,memory.free',
        '--format=csv,noheader,nounits'], capture_output=True, text=True, check=True)
    lines = result.stdout.strip().splitlines()
    if len(lines) != 1 or 'H100' not in lines[0] or float(lines[0].split(',')[-1]) < 80000:
        raise RuntimeError(f'Expected one free H100 80GB; got {result.stdout}')
    attempts = ROOT / '.marlowe/m18/attempts'
    attempts.mkdir(parents=True, exist_ok=True)
    temporary = attempts / f'{job_id}.tmp'
    temporary.write_text(json.dumps(plan, indent=2) + '\n')
    temporary.replace(attempts / f'{job_id}.json')
    print(f'GPU {lines[0]} FRESH tag={tag} restart={restart}', flush=True)
    run_wave(specs, directory, interval=LAUNCH_INTERVAL)


if __name__ == '__main__':
    main()
