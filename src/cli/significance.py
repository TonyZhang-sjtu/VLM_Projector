"""Independent patient-paired significance report; never launches training."""
import argparse
import csv
from datetime import datetime
from importlib.metadata import version
import json
import os
from pathlib import Path
from zoneinfo import ZoneInfo

import yaml

from src.evaluation.clinical import file_digest, validate_pair, write_json
from src.evaluation.significance import METRICS, GROUPS, holm_adjust, paired_test
from src.evaluation.significance_io import ROOT, read_json, resolve, load_run, attach_ce, validate_run_pair, find_metric_files


def discover(manifest):
    """Candidate retrieval only. Rounded table values never authorize auto-binding."""
    found = find_metric_files(manifest.get('search_roots', ['runs']))
    candidates = {m['id']: [] for m in manifest['methods']}
    for path in sorted(set(found)):
        if path.parent.name != 'best_test':
            continue
        run = path.parents[2]
        metric = read_json(path)
        cfg_path = run / metric.get('stage', 'joint_lora') / 'resolved_config.yaml'
        if not cfg_path.is_file():
            continue
        cfg = yaml.safe_load(cfg_path.read_text())
        for target in manifest['methods']:
            if (cfg['projector']['name'] != target['projector']
                    or cfg['projector'].get('target_compression_ratio') != target['ratio']):
                continue
            differences = {k: {'table': v, 'recorded': metric.get('overall', {}).get(k)}
                           for k, v in target['expected_rounded_metrics'].items()
                           if k not in metric.get('overall', {}) or abs(metric['overall'][k]-v) > .000050001}
            candidates[target['id']].append({
                'run_dir': str(run), 'metric_file': str(path),
                'checkpoint_sha256': metric.get('checkpoint_sha256'),
                'all_11_rounded_metrics_match': not differences, 'differences': differences,
                'seed': cfg['seed'], 'projector_lr': cfg['training'].get('projector_lr'),
                'encoder_lr': cfg['training'].get('encoder_lr'),
                'cosine_min_lr_ratio': cfg['training'].get('cosine_min_lr_ratio'),
                'max_epochs': cfg['training'].get('max_epochs'),
            })
    return candidates


def export_report(output, manifest, audits, results):
    """CSV contains both p columns; Excel adds matrices for easy comparison."""
    fields = ['baseline', 'metric', 'status', 'score_ours', 'score_baseline',
              'delta_ours_minus_baseline', 'improvement', 'ci95_low', 'ci95_high',
              'p_raw', 'p_holm', 'holm_status', 'significant_holm_005',
              'patients', 'common_valid_samples', 'expected_samples', 'valid_ours',
              'valid_baseline', 'p_mc_se', 'reason']
    temp = output / 'comparisons.csv.tmp'
    with temp.open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(results)
    temp.replace(output / 'comparisons.csv')
    try:
        from openpyxl import Workbook
        from openpyxl.styles import Font, PatternFill
    except ImportError:
        print('openpyxl unavailable: CSV/JSON written; Excel export skipped.', flush=True)
        return
    wb = Workbook()
    ws = wb.active
    ws.title = 'comparisons'
    ws.append(fields)
    for row in results:
        ws.append([row.get(k) for k in fields])
    for key, title in [('p_raw', 'p_raw'), ('p_holm', 'p_Holm'),
                       ('delta_ours_minus_baseline', 'delta_ours_minus_baseline')]:
        ws = wb.create_sheet(title)
        ws.append(['Baseline', *METRICS])
        for entry in manifest['methods']:
            if entry['id'] == manifest['ours']:
                continue
            lookup = {r['metric']: r for r in results if r['baseline_id'] == entry['id']}
            ws.append([entry['label'], *[lookup[m].get(key) for m in METRICS]])
    ws = wb.create_sheet('run_identity')
    keys = ['method_id', 'label', 'status', 'run_dir', 'checkpoint_sha256',
            'checkpoint_bytes_verified', 'config_sha256', 'seed', 'ratio',
            'selected_step', 'selected_validation_ce', 'samples', 'patients',
            'test_identity_sha256', 'training_selection_sha256', 'reason']
    ws.append(keys)
    for audit in audits:
        ws.append([audit.get(k) for k in keys])
    ws = wb.create_sheet('protocol')
    ws.append(['Key', 'Value'])
    protocol = {
        'family_size': manifest['family_size'],
        'test': 'two-sided paired patient-cluster label exchange; |A-B| statistic',
        'CI': 'paired patient-cluster percentile bootstrap; marginal, unadjusted 95%',
        'missing': 'pairwise common-valid per metric; never impute scores',
        'scope': 'fixed checkpoints; excludes training-seed uncertainty',
        'Holm': f"Full planned {manifest['family_size']} hypotheses. While incomplete, available p-values use conservative full-family adjustment; missing raw p-values remain blank.",
        'significance': 'Use p_Holm < .05 AND effect direction; not evidence of equivalence when nonsignificant.',
        'BLEU': 'corpus BLEU from aggregated clipped n-gram counts; no cross-report n-grams',
        'CE': 'sum loss_sum / sum answer_tokens; replay must match historical CE within 1e-5',
        'source_snapshot': json.dumps(manifest['source'], ensure_ascii=False),
    }
    for key, value in protocol.items():
        ws.append([key, value])
    for ws in wb:
        ws.freeze_panes = 'B2'
        ws.auto_filter.ref = ws.dimensions
        for cell in ws[1]:
            cell.font = Font(bold=True, color='FFFFFF')
            cell.fill = PatternFill('solid', fgColor='234B70')
        for col in ws.columns:
            ws.column_dimensions[col[0].column_letter].width = min(65, max(18, len(str(col[0].value))+2))
        for row in ws.iter_rows(min_row=2):
            for cell in row:
                if isinstance(cell.value, float):
                    cell.number_format = '0.000000'
    temp = output / 'significance.tmp.xlsx'
    wb.save(temp)
    temp.replace(output / 'significance.xlsx')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--ce-dir', type=Path)
    parser.add_argument('--audit-only', action='store_true')
    parser.add_argument('--discover', action='store_true')
    args = parser.parse_args()
    manifest = read_json(args.manifest)
    if len(manifest['methods']) != len({m['id'] for m in manifest['methods']}):
        raise ValueError('Duplicate manifest method IDs')
    if manifest['family_size'] != (len(manifest['methods'])-1) * len(METRICS):
        raise ValueError('Family size must include every declared method/metric pair')
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if args.discover:
        write_json(output / 'mapping_candidates.json', discover(manifest))
    ce_dir = args.ce_dir or output / 'ce'
    audits, runs, errors = [], {}, {}
    for entry in manifest['methods']:
        identity = entry['id']
        if not entry.get('run_dir'):
            reason = 'Source run not supplied; screenshot totals alone are insufficient'
            audits.append(dict(method_id=identity, label=entry['label'], status='missing_source', reason=reason))
            errors[identity] = reason
            continue
        print(f'AUDIT {identity}', flush=True)
        try:
            run = load_run(entry)
            attach_ce(run, ce_dir / identity)
            run['audit']['status'] = 'verified'
            audits.append(run['audit'])
            runs[identity] = run
        except Exception as exc:
            reason = f'{type(exc).__name__}: {exc}'
            print(f'AUDIT FAILED {identity}: {reason}', flush=True)
            audits.append(dict(method_id=identity, label=entry['label'], status='audit_failed',
                               run_dir=entry['run_dir'], reason=reason))
            errors[identity] = reason
    write_json(output / 'run_audit.json', audits)
    write_json(output / 'study_snapshot.json', {
        'manifest': manifest, 'manifest_sha256': file_digest(args.manifest),
        'generated_at': datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
        'package_versions': {name: version(name) for name in ('numpy', 'sacrebleu', 'rouge-score', 'torch')},
        'analysis_source_hashes': {str(p.relative_to(ROOT)): file_digest(p) for p in (
            Path(__file__), ROOT/'src/evaluation/significance.py', ROOT/'src/evaluation/significance_io.py')},
    })
    if args.audit_only:
        return
    ours = runs.get(manifest['ours'])
    results = []
    for entry in manifest['methods']:
        if entry['id'] == manifest['ours']:
            continue
        other = runs.get(entry['id'])
        pair_error = None
        if ours and other:
            try:
                validate_run_pair(ours, other)
            except Exception as exc:
                pair_error = str(exc)
        for metric in METRICS:
            result = dict(baseline_id=entry['id'], baseline=entry['label'], metric=metric,
                          status='pending', p_raw=None, p_holm=None)
            reason = pair_error or errors.get(manifest['ours']) or errors.get(entry['id'])
            if not reason and (metric not in ours['vectors'] or metric not in other['vectors']):
                reason = 'Per-sample CE export pending'
            if reason:
                result['reason'] = reason
            else:
                print(f'TEST {entry["id"]} {metric}', flush=True)
                result.update(paired_test(ours['rows'], other['rows'], ours['vectors'][metric],
                    other['vectors'][metric], metric, permutations=manifest['permutations'],
                    bootstraps=manifest['bootstraps'], seed=manifest['resampling_seed']))
                result['status'] = 'complete'
            results.append(result)
    complete = [r for r in results if r['status'] == 'complete']
    adjusted = holm_adjust([r['p_raw'] for r in complete], manifest['family_size'])
    holm_status = 'final_full_family' if len(complete) == manifest['family_size'] else 'conservative_incomplete_family'
    for result, p in zip(complete, adjusted):
        result.update(p_holm=p, holm_status=holm_status, significant_holm_005=p < .05)
    report = {'status': 'complete' if len(complete) == manifest['family_size'] else 'partial',
              'completed_tests': len(complete), 'planned_tests': manifest['family_size'],
              'holm_status': holm_status, 'comparisons': results}
    write_json(output / 'comparisons.json', report)
    export_report(output, manifest, audits, results)
    print(json.dumps({k: v for k, v in report.items() if k != 'comparisons'}), flush=True)


if __name__ == '__main__':
    main()
