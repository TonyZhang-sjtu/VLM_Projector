"""One entry point for Baseline A vs six baselines at 16/32/64/128/256x."""
import argparse
from contextlib import contextmanager
import csv
from datetime import datetime
import fcntl
from importlib.metadata import version
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from zoneinfo import ZoneInfo

from src.cli.significance import export_report
from src.cli.significance_queue import eligible_gpus
from src.evaluation.clinical import file_digest, write_json
from src.evaluation.ce_runtime import DEFAULT_PROFILES, ce_replay_environment, resolve_ce_runtime
from src.evaluation.significance import METRICS, holm_adjust, paired_test
from src.evaluation.significance_io import ROOT, read_json, load_run, attach_ce, validate_run_pair
from src.evaluation.significance_study import (
    RATIOS, METHODS, read_targets, inventory, matching_candidates, candidate_identity, select_candidate,
)


def stamp():
    return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()


def key_for(target):
    return f'r{target["ratio"]}/{target["id"]}'


def prepare_study(args, output):
    targets, source = read_targets(args.workbook)
    frozen_path = output / 'run_bindings.json'
    seeds = read_json(args.bindings)
    frozen = read_json(frozen_path) if frozen_path.is_file() else {'bindings': {}}
    for previous in (seeds, frozen):
        if previous.get('targets_sha256', source['targets_sha256']) != source['targets_sha256']:
            raise ValueError('Excel target cells changed since binding. Use a new output directory and updated --bindings; no silent remapping.')
    pins = dict(seeds['bindings'])
    for key, pin in frozen['bindings'].items():
        if key in pins and pins[key] != pin:
            raise ValueError(f'Conflicting frozen bindings: {key}')
        pins[key] = pin
    items, inventory_errors = inventory(args.search_roots)
    runs, entries, audits, errors, candidates = {}, {}, [], {}, {}
    verified_pins = dict(frozen['bindings'])
    for target in targets:
        key = key_for(target)
        matches = matching_candidates(target, items)
        candidates[key] = [dict(candidate_identity(c), seed=c['config']['seed'],
                                training_recipe={k: c['config']['training'].get(k) for k in (
                                    'projector_lr', 'encoder_lr', 'max_epochs', 'cosine_min_lr_ratio')})
                           for c in matches]
        entries[key] = dict(target, run_dir=None)
        print(f'AUDIT {key} ({target["excel_range"]})', flush=True)
        try:
            if key not in pins:
                raise ValueError('Explicit binding required; inspect mapping_candidates.json. No automatic binding from rounded scores.')
            candidate = select_candidate(target, matches, pins[key])
            entry = dict(target, **candidate_identity(candidate))
            entries[key] = entry
            run = load_run(entry)
            if not run['audit']['checkpoint_bytes_verified']:
                raise ValueError('best.pt missing; transfer the full selected run before starting')
            attach_ce(run, output / key.split('/')[0] / 'ce' / target['id'], policy=args.ce_policy)
            run['audit'].update(status='verified', excel_range=target['excel_range'])
            runs[key] = run
        except Exception as exc:
            errors[key] = f'{type(exc).__name__}: {exc}'
            print(f'NOT READY {key}: {errors[key]}', flush=True)
    # Test all pair contracts before launching any CE inference or permutations.
    for ratio in RATIOS:
        ours_key = f'r{ratio}/ours'
        for method in METHODS:
            key = f'r{ratio}/{method}'
            if key not in runs:
                continue
            try:
                if ours_key not in runs:
                    raise ValueError('Baseline A unavailable for pair validation')
                validate_run_pair(runs[ours_key], runs[key])
                verified_pins[key] = candidate_identity(entries[key])
            except Exception as exc:
                errors[key] = f'Pair mismatch: {exc}'
        ratio_entries = [entries[f'r{ratio}/{m}'] for m in METHODS]
        manifest = dict(schema_version=1, study=f'baseline_a_r{ratio}_no_mlp', ours='ours',
                        family_size=66, permutations=10000, bootstraps=10000, resampling_seed=20261030,
                        source=source, methods=ratio_entries, ce_policy=args.ce_policy)
        (output / f'r{ratio}').mkdir(exist_ok=True)
        write_json(output / f'r{ratio}' / 'run_manifest.json', manifest)
    for target in targets:
        key = key_for(target)
        audit = dict(runs[key]['audit']) if key in runs else dict(method_id=target['id'], ratio=target['ratio'])
        if key in errors:
            audit.update(status='not_ready', reason=errors[key])
        audits.append(audit)
    if file_digest(args.workbook) != source['workbook_sha256']:
        raise ValueError('Workbook changed during audit; rerun with a stable input')
    shutil.copy2(args.workbook, output / 'source_workbook.xlsx')
    write_json(output / 'run_bindings.json', dict(targets_sha256=source['targets_sha256'], bindings=verified_pins))
    write_json(output / 'mapping_candidates.json', dict(candidates=candidates, unreadable_files=inventory_errors))
    write_json(output / 'run_audit.json', audits)
    write_json(output / 'study_snapshot.json', dict(
        source=source, targets=targets, created_at=stamp(), ratios=list(RATIOS),
        methods=list(METHODS), global_family_size=330, per_ratio_family_size=66,
        permutations=10000, bootstraps=10000, resampling_seed=20261030,
        ce_policy=args.ce_policy, runtime_profiles_sha256=file_digest(args.runtime_profiles),
        package_versions={name: version(name) for name in ('numpy', 'sacrebleu', 'rouge-score', 'torch', 'openpyxl')},
        source_hashes={str(p.relative_to(ROOT)): file_digest(p) for p in (
            Path(__file__), ROOT/'src/evaluation/significance_study.py',
            ROOT/'src/evaluation/significance_io.py', ROOT/'src/evaluation/significance.py',
            ROOT/'src/cli/export_answer_ce.py', ROOT/'src/evaluation/ce_runtime.py',
            ROOT/'src/evaluation/ce_validation.py')},
    ))
    write_json(output / 'preflight.json', dict(status='ready' if not errors else 'not_ready',
        updated_at=stamp(), expected_runs=35, ready_runs=35-len(errors), issues=errors,
        inference_started=False, statistical_tests_started=False))
    return targets, source, entries, runs, errors


def run_export(args, output, target, *, dry_run=False, gpu=None):
    ratio, method = target['ratio'], target['id']
    dest = output / f'r{ratio}' / 'ce' / method
    dest.mkdir(parents=True, exist_ok=True)
    manifest_path = output / f'r{ratio}' / 'run_manifest.json'
    entry = next(m for m in read_json(manifest_path)['methods'] if m['id'] == method)
    runtime = resolve_ce_runtime(entry, args.runtime_profiles)
    command = [args.projector_python, '-m', 'src.cli.export_answer_ce', '--manifest',
               str(output / f'r{ratio}' / 'run_manifest.json'), '--method', method,
               '--output-dir', str(dest), '--device', 'cuda:0',
               '--runtime-profiles', str(args.runtime_profiles), '--ce-policy', args.ce_policy]
    if dry_run:
        command.append('--dry-run')
    env = ce_replay_environment(settings=runtime['environment'])
    if gpu is not None:
        env['CUDA_VISIBLE_DEVICES'] = str(gpu)
    log = dest / ('preflight.log' if dry_run else 'console.log')
    with log.open('a') as stream:
        result = subprocess.run(command, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f'CE {key_for(target)} failed (exit {result.returncode}); see {log}')


def export_missing_ce(args, output, targets, runs):
    """Quarantine individual CE failures; preserve progress and finish other jobs."""
    errors, ready = {}, []
    def progress():
        done = [key_for(t) for t in targets if 'answer_ce' in runs[key_for(t)]['vectors']]
        write_json(output / 'ce_progress.json', dict(updated_at=stamp(), completed=done,
            failed=errors, pending=[key_for(t) for t in targets if key_for(t) not in done
                                   and key_for(t) not in errors], ce_policy=args.ce_policy))
    for target in targets:
        key = key_for(target)
        if 'answer_ce' in runs[key]['vectors']:
            continue
        try:
            print(f'CE PREFLIGHT {key}', flush=True)
            run_export(args, output, target, dry_run=True)
            ready.append(target)
        except Exception as exc:
            errors[key] = str(exc)
            print(f'CE UNAVAILABLE {key}: {exc}; continuing other verified runs', flush=True)
    progress()
    for target in ready:
        key = key_for(target)
        try:
            with available_gpu(args, output, key) as gpu:
                print(f'CE EXPORT {key} GPU {gpu}', flush=True)
                write_json(output / 'status.json', dict(status='exporting_ce', active_method=key,
                           gpu=gpu, ce_policy=args.ce_policy, updated_at=stamp()))
                run_export(args, output, target, gpu=gpu)
            if not attach_ce(runs[key], output / f'r{target["ratio"]}' / 'ce' / target['id'], policy=args.ce_policy):
                raise RuntimeError(f'CE export missing for {key}')
        except Exception as exc:
            errors[key] = str(exc)
            print(f'CE FAILED {key}: {exc}; continuing other verified runs', flush=True)
        progress()
    return errors


@contextmanager
def available_gpu(args, output, method):
    started, gpu_lock = time.monotonic(), None
    try:
        while gpu_lock is None:
            choices = eligible_gpus()
            if args.gpu != 'auto':
                choices = [g for g in choices if g == args.gpu]
            for gpu in choices:
                path = ROOT / '.tmp' / f'significance_gpu_{gpu}.lock'
                path.parent.mkdir(exist_ok=True)
                lock = path.open('w')
                try:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    lock.close()
                    continue
                if gpu not in eligible_gpus():
                    lock.close()
                    continue
                gpu_lock = lock
                break
            if gpu_lock is None:
                write_json(output / 'status.json', dict(status='waiting_for_gpu_memory', active_method=method, updated_at=stamp()))
                if time.monotonic()-started >= args.max_wait_hours*3600:
                    raise RuntimeError('No eligible GPU available before timeout (requires >= 20000 MiB free); rerun to resume, or use --skip-ce for CPU-only analysis')
                print(f'WAIT GPU with >= 20000 MiB free for {method}', flush=True)
                time.sleep(30)
        yield gpu
    finally:
        if gpu_lock:
            gpu_lock.close()


def apply_suite_holm(rows):
    complete = [r for r in rows if r['status'] == 'complete']
    adjusted = holm_adjust([r['p_raw'] for r in complete], 330)
    state = 'final_full_family' if len(complete) == 330 else 'conservative_incomplete_family'
    for result, p in zip(complete, adjusted):
        result.update(p_holm_330=p, holm_330_status=state, significant_holm_330_005=p < .05)
    return state


def export_suite(output, source, results):
    state = apply_suite_holm(results)
    complete = sum(r['status'] == 'complete' for r in results)
    report = dict(status='complete' if complete == 330 else 'partial', completed_tests=complete,
                  planned_tests=330, holm_330_status=state, source=source, comparisons=results)
    write_json(output / 'comparisons.json', report)
    fields = ['ratio', 'baseline', 'metric', 'status', 'score_ours', 'score_baseline',
              'delta_ours_minus_baseline', 'improvement', 'ci95_low', 'ci95_high',
              'p_raw', 'p_holm_66', 'p_holm_330', 'holm_66_status', 'holm_330_status',
              'significant_holm_330_005', 'patients', 'common_valid_samples',
              'expected_samples', 'valid_ours', 'valid_baseline', 'p_mc_se', 'reason']
    fields += ['score_basis', 'historical_ce_ours', 'historical_ce_baseline',
               'ce_replay_difference_ours', 'ce_replay_difference_baseline']
    temporary = output / 'comparisons.csv.tmp'
    with temporary.open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(results)
    temporary.replace(output / 'comparisons.csv')
    from openpyxl import Workbook
    from openpyxl.styles import Font, PatternFill
    book = Workbook()
    sheet = book.active
    sheet.title = 'comparisons'
    sheet.append(fields)
    for row in results:
        sheet.append([row.get(k) for k in fields])
    for key in ('p_raw', 'p_holm_66', 'p_holm_330'):
        sheet = book.create_sheet(key)
        sheet.append(['Compression', 'Baseline', *METRICS])
        for ratio in RATIOS:
            for method, (label, _) in METHODS.items():
                if method == 'ours':
                    continue
                lookup = {r['metric']: r for r in results if r['ratio'] == ratio and r['baseline_id'] == method}
                sheet.append([ratio, label, *[lookup[m].get(key) for m in METRICS]])
    sheet = book.create_sheet('run_identity')
    columns = ['ratio', 'method_id', 'excel_range', 'status', 'run_dir', 'checkpoint_sha256',
               'seed', 'config_sha256', 'test_identity_sha256', 'selected_step', 'reason']
    sheet.append(columns)
    for audit in read_json(output / 'run_audit.json'):
        sheet.append([audit.get(k) for k in columns])
    sheet = book.create_sheet('ce_replay_audit')
    sheet.append(['ratio', 'method', 'recomputed_ce', 'historical_ce', 'difference',
                  'ce_policy', 'runtime_profile', 'replay_gpu', 'environment'])
    for audit in read_json(output / 'run_audit.json'):
        replay = audit.get('ce_replay', {})
        profile = replay.get('runtime_profile') or {}
        sheet.append([audit.get('ratio'), audit.get('method_id'), replay.get('answer_ce'),
            replay.get('historical_answer_ce'), replay.get('difference'), replay.get('ce_policy'),
            profile.get('id'), replay.get('device_name'), json.dumps(replay.get('runtime_environment'))])
    sheet = book.create_sheet('protocol')
    sheet.append(['Key', 'Value'])
    for k, v in {
        'source': json.dumps(source, ensure_ascii=False), 'scope': 'Fixed checkpoints, not training seed uncertainty',
        'test': '10000 paired patient-cluster label exchanges, two-sided absolute difference; (extreme+1)/(B+1)',
        'CI': '10000 paired patient-cluster percentile bootstrap; marginal 95%, not multiplicity-adjusted',
        'p_holm_66': '6 baselines x 11 metrics within each compression ratio',
        'p_holm_330': 'Primary family: all 5 ratios x 6 baselines x 11 metrics; excludes MLP',
        'missing': 'Pairwise common valid samples; no imputation. Missing CE stays blank.',
        'partial': 'Incomplete families use conservative full-family adjustment; recompute when complete.',
        'aggregation': 'Corpus BLEU; original report means; CE=sum(loss_sum)/sum(answer_tokens)',
        'ce_replay': 'CE scores/p-values use per-sample teacher-forced replay. Recompute mode preserves historical differences without claiming historical numerical equivalence; report the recomputed CE with its p-value. See ce_replay_audit. Strict mode additionally requires historical total and cohort agreement.',
        'interpretation': 'Check effect direction as well as p; nonsignificance is not equivalence.',
        'cross_server_scorers': 'Require identical model/code/packages/settings/seed/TF32/backend; record but exclude executable path, GPU model label and CUDA ordinal. Not a claim of bit-identical hardware results.',
    }.items():
        sheet.append([k, v])
    for sheet in book:
        sheet.freeze_panes = 'C2'
        sheet.auto_filter.ref = sheet.dimensions
        for cell in sheet[1]:
            cell.font = Font(bold=True, color='FFFFFF')
            cell.fill = PatternFill('solid', fgColor='234B70')
        for col in sheet.columns:
            sheet.column_dimensions[col[0].column_letter].width = min(60, max(18, len(str(col[0].value))+2))
        for row in sheet.iter_rows(min_row=2):
            for cell in row:
                if isinstance(cell.value, float):
                    cell.number_format = '0.000000'
    book.save(output / 'significance.tmp.xlsx')
    (output / 'significance.tmp.xlsx').replace(output / 'significance.xlsx')
    return report


def run_study(args, output):
    targets, source, entries, runs, errors = prepare_study(args, output)
    if errors:
        raise RuntimeError(f'{len(errors)}/35 runs not ready; see {output / "preflight.json"}. No CE or statistical tests started.')
    if args.audit_only:
        print('READY: 35 identities and all paired contracts verified; no CE or statistical tests started.', flush=True)
        return 'audit_passed'
    ce_errors = {} if args.skip_ce else export_missing_ce(args, output, targets, runs)
    all_results = []
    for ratio in RATIOS:
        ratio_output = output / f'r{ratio}'
        manifest = read_json(ratio_output / 'run_manifest.json')
        ours, results = runs[f'r{ratio}/ours'], []
        for entry in manifest['methods']:
            if entry['id'] == 'ours':
                continue
            other = runs[f'r{ratio}/{entry["id"]}']
            validate_run_pair(ours, other)
            for metric in METRICS:
                row = dict(baseline_id=entry['id'], baseline=entry['label'], metric=metric,
                           status='pending', p_raw=None, p_holm=None,
                           score_basis='recomputed_teacher_forced_ce' if metric == 'answer_ce' else 'historical_per_sample_outputs')
                if metric == 'answer_ce':
                    row.update(historical_ce_ours=ours['audit']['original_metrics']['answer_ce'],
                        historical_ce_baseline=other['audit']['original_metrics']['answer_ce'],
                        ce_replay_difference_ours=ours['audit'].get('ce_replay', {}).get('difference'),
                        ce_replay_difference_baseline=other['audit'].get('ce_replay', {}).get('difference'))
                if metric not in ours['vectors'] or metric not in other['vectors']:
                    row['reason'] = 'Per-sample CE absent; ' + ('--skip-ce selected' if args.skip_ce else
                        '; '.join(ce_errors.get(k, '') for k in (f'r{ratio}/ours', f'r{ratio}/{entry["id"]}') if k in ce_errors))
                else:
                    print(f'TEST r{ratio} {entry["id"]} {metric}', flush=True)
                    row.update(paired_test(ours['rows'], other['rows'], ours['vectors'][metric],
                        other['vectors'][metric], metric, permutations=manifest['permutations'],
                        bootstraps=manifest['bootstraps'], seed=manifest['resampling_seed']))
                    row['status'] = 'complete'
                results.append(row)
        completed = [r for r in results if r['status'] == 'complete']
        state = 'final_full_family' if len(completed) == 66 else 'conservative_incomplete_family'
        for row, p in zip(completed, holm_adjust([r['p_raw'] for r in completed], 66)):
            row.update(p_holm=p, holm_status=state, significant_holm_005=p < .05)
        audits = [runs[f'r{ratio}/{m}']['audit'] for m in METHODS]
        write_json(ratio_output / 'run_audit.json', audits)
        write_json(ratio_output / 'comparisons.json', dict(
            status='complete' if len(completed) == 66 else 'partial', completed_tests=len(completed),
            planned_tests=66, holm_status=state, comparisons=results))
        export_report(ratio_output, manifest, audits, results)
        all_results.extend(dict(r, ratio=ratio, p_holm_66=r.get('p_holm'), holm_66_status=r.get('holm_status')) for r in results)
    write_json(output / 'run_audit.json', [runs[key_for(t)]['audit'] for t in targets])
    report = export_suite(output, source, all_results)
    print(f'{report["status"]}: {report["completed_tests"]}/330 tests; {output / "significance.xlsx"}', flush=True)
    return report['status']


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workbook', type=Path, default=ROOT/'results/Experiments for VLM Projector.xlsx')
    parser.add_argument('--output-dir', type=Path, default=ROOT/'results/significance_baseline_a_5ratios')
    parser.add_argument('--bindings', type=Path, default=ROOT/'scripts/significance/baseline_a_bindings.json')
    parser.add_argument('--search-roots', type=Path, nargs='+', default=[ROOT/'runs/baselines', ROOT/'runs/imported/server_b'])
    parser.add_argument('--audit-only', action='store_true', help='Verify all 35 run identities and pair contracts; no model inference/tests')
    parser.add_argument('--skip-ce', action='store_true', help='CPU analysis only; missing CE stays blank (planned families remain 66/330)')
    parser.add_argument('--gpu', default='auto', help='Physical GPU index or auto; requires only >= 20000 MiB free memory')
    parser.add_argument('--max-wait-hours', type=float, default=24)
    parser.add_argument('--projector-python', default=os.environ.get('PROJECTOR_PYTHON', '/mnt/sda/mingjie/conda_envs/projector/bin/python'))
    parser.add_argument('--runtime-profiles', type=Path, default=DEFAULT_PROFILES)
    parser.add_argument('--ce-policy', choices=('strict', 'recompute'), default='strict',
                        help='recompute uses new CE measurements and reports historical differences explicitly')
    args = parser.parse_args(argv)
    if args.max_wait_hours < 0 or (args.gpu != 'auto' and not args.gpu.isdecimal()):
        parser.error('Use nonnegative --max-wait-hours and --gpu auto or a physical GPU index')
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    with (output / 'suite.lock').open('w') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            parser.exit(2, f'Another suite is using {output}\n')
        write_json(output / 'status.json', dict(status='auditing', updated_at=stamp(), pid=os.getpid()))
        try:
            state = run_study(args, output)
        except KeyboardInterrupt:
            write_json(output / 'status.json', dict(status='interrupted', updated_at=stamp()))
            return 130
        except Exception as exc:
            write_json(output / 'status.json', dict(status='stopped', reason=str(exc), updated_at=stamp()))
            print(f'STOPPED: {exc}', file=sys.stderr, flush=True)
            return 2
        write_json(output / 'status.json', dict(status=state, updated_at=stamp()))
    return 0


if __name__ == '__main__':
    sys.exit(main())
