"""Compare frozen graph-v2 scores at 32x/256x; no model/GPU inference.

Patient-cluster paired percentile bootstrap quantifies sampling uncertainty only.
It does not validate automatic labels or correct for multiple comparisons.
GREEN category (c) is a report-wide error count, not lesion-location accuracy.
"""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import openpyxl

from .prepare import add_sheet, green_location_count
from .protocol_v2 import VERSION

ROOT = Path(__file__).resolve().parents[2]
BASELINES = ['AvgPool3D', 'Concat3D', 'ConvPool3D', 'LocalPacker3D',
             'Perceiver_NV', 'Adaptive_Merge3D', 'Token_Select3D']
DIMS = ['side', 'site', 'joint']


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fractions(s):
    n, exact, tp, fp, fn = np.moveaxis(s, -1, 0)
    def div(a, b):
        return np.divide(a, b, out=np.full(np.shape(a), np.nan), where=b > 0)
    return {'em': div(exact, n), 'precision': div(tp, tp + fp),
            'recall': div(tp, tp + fn), 'micro_f1': div(2 * tp, 2 * tp + fp + fn)}


def interval(x):
    lo, hi = np.nanpercentile(x, [2.5, 97.5])
    return float(lo), float(hi)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--resamples', default=10000, type=int)
    p.add_argument('--seed', default=20261108, type=int)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    source = ROOT/'results/radmatch_comparison_1108/full_inputs/comparison_inputs.json'
    index_path = ROOT/'results/anchorread3d_analysis_1108/artifact_reconciliation_1108.json'
    index = json.loads(index_path.read_text())
    entries = {e['key']: e for e in index['paper_settings']}
    groups = defaultdict(dict)
    for r in json.loads(source.read_text()):
        if r['setting'].endswith(('/r32', '/r256')) or r['setting'] == 'rg/MLP/r1':
            assert r['graph_protocol'] == VERSION
            groups[r['setting']][r['id']] = r
    assert len(groups) == 25
    canonical = groups['rg/A0/r32']
    patients = sorted({r['patient_id'] for r in canonical.values()})
    assert len(canonical) == 560 and len(patients) == 140
    pi = {v: i for i, v in enumerate(patients)}
    counts = np.random.default_rng(a.seed).multinomial(
        len(patients), np.full(len(patients), 1 / len(patients)), size=a.resamples)
    scores, boot, green_boot, green_scores = {}, {}, {}, {}
    green_observations = defaultdict(dict)
    summary, green_summary, cases, provenance = [], [], [], []
    prior = {(r['setting'], r['dimension']): r for r in csv.DictReader(
        (ROOT/'results/radmatch_comparison_1108/version_comparison/all_versions.csv').open())
        if r['version'] == 'graph_v2'}
    for key, rows in sorted(groups.items()):
        assert set(rows) == set(canonical)
        for rid, r in rows.items():
            ref = canonical[rid]
            assert r['patient_id'] == ref['patient_id']
            assert r['reference_extraction'] == ref['reference_extraction']
            assert r['reference_sha256'] == ref['reference_sha256']
        for dim in DIMS:
            stats = np.zeros((len(patients), 5), dtype=float)
            eligible_patients = set()
            for rid, r in rows.items():
                sc = r['graph_scores'][dim]
                if not sc['eligible']:
                    continue
                eligible_patients.add(r['patient_id'])
                stats[pi[r['patient_id']]] += [1, sc['exact'], sc['tp'], sc['fp'], sc['fn']]
            totals = stats.sum(axis=0)
            values = fractions(totals)
            samples = fractions(counts @ stats)
            assert int(totals[0]) == int(prior[(key, dim)]['n'])
            assert np.isclose(values['micro_f1'], float(prior[(key, dim)]['micro_f1']))
            assert np.isclose(values['em'], float(prior[(key, dim)]['exact']))
            for metric, value in values.items():
                scores[key, dim, metric] = float(value)
                boot[key, dim, metric] = samples[metric]
                lo, hi = interval(samples[metric])
                summary.append(dict(setting=key, dimension=dim, metric=metric,
                    n=int(totals[0]), patients=len(eligible_patients),
                    exact_count=int(totals[1]), tp=int(totals[2]), fp=int(totals[3]), fn=int(totals[4]),
                    value=float(value), ci_low=lo, ci_high=hi))
        clinical = Path(entries[key]['paths']['metric']).parent/'scoring/clinical_scores.jsonl'
        clinical_rows = {r['id']: r for r in map(json.loads, clinical.read_text().splitlines())}
        assert set(clinical_rows) == set(rows)
        gs = np.zeros((len(patients), 6), dtype=float)
        status = Counter()
        fingerprints = Counter()
        repairs = 0
        for rid, r in clinical_rows.items():
            frozen = rows[rid]
            assert hashlib.sha256(r['prediction'].encode()).hexdigest() == frozen['prediction_sha256']
            assert hashlib.sha256(r['reference'].encode()).hexdigest() == frozen['reference_sha256']
            g = r['clinical']['green']
            status[g['status']] += 1
            fingerprints[g.get('scorer_fingerprint', '')] += 1
            details = g.get('details', {})
            loc = green_location_count(details) if g['status'] == 'ok' else None
            repairs += int(bool(details.get('format_repair')))
            case = dict(setting=key, id=rid, patient_id=r['patient_id'],
                green_status=g['status'], green_location_errors=loc)
            for dim in DIMS:
                sc = frozen['graph_scores'][dim]
                case[dim+'_eligible'] = sc['eligible']
                case[dim+'_em'] = sc.get('exact')
                case[dim+'_f1'] = sc.get('f1')
            cases.append(case)
            if loc is not None:
                errors = details['significant_errors']
                assert len(errors) == 6 and all(isinstance(x, (int, float)) and x >= 0 for x in errors)
                observation = [1, loc, int(loc > 0), errors[0], errors[1], g['values']['green']]
                green_observations[key][rid] = observation
                gs[pi[r['patient_id']]] += observation
        totals = gs.sum(axis=0)
        assert totals[0] > 0
        draws = counts @ gs
        row = dict(setting=key, n=int(totals[0]), location_errors_total=float(totals[1]),
            reports_with_location_error=int(totals[2]), format_repaired_reports=repairs,
            scorer_fingerprints=dict(fingerprints), status_counts=dict(status))
        for j, name in enumerate(['location_errors_per_report', 'location_error_report_rate',
                                  'false_findings_per_report', 'missing_findings_per_report', 'green_mean'], start=1):
            val = float(totals[j] / totals[0])
            samples = draws[:, j] / draws[:, 0]
            lo, hi = interval(samples)
            row.update({name: val, name+'_ci_low': lo, name+'_ci_high': hi})
            green_scores[key, name] = val
            green_boot[key, name] = samples
        green_summary.append(row)
        provenance.append(dict(setting=key, clinical_path=str(clinical), clinical_sha256=sha(clinical)))
    # GREEN failures are retained in coverage. Use one common valid intersection
    # across ALL 25 settings for every comparative GREEN statistic.
    green_common_ids = set.intersection(*(set(v) for v in green_observations.values()))
    assert green_common_ids
    green_names = ['location_errors_per_report', 'location_error_report_rate',
                   'false_findings_per_report', 'missing_findings_per_report', 'green_mean']
    for row in green_summary:
        key = row['setting']
        gs = np.zeros((len(patients), 6), dtype=float)
        for rid in green_common_ids:
            gs[pi[canonical[rid]['patient_id']]] += green_observations[key][rid]
        totals, draws = gs.sum(axis=0), counts @ gs
        row['common_n'] = int(totals[0])
        for j, name in enumerate(green_names, start=1):
            val = float(totals[j] / totals[0])
            samples = draws[:, j] / draws[:, 0]
            lo, hi = interval(samples)
            row.update({'common_'+name: val, 'common_'+name+'_ci_low': lo,
                        'common_'+name+'_ci_high': hi})
            green_scores[key, name] = val
            green_boot[key, name] = samples
    paired, best = [], []
    for ratio in [32, 256]:
        ours = f'rg/A0/r{ratio}'
        controls = [f'rg/{name}/r{ratio}' for name in BASELINES + ['A1', 'A2', 'A3', 'A4']] + ['rg/MLP/r1']
        for dim in DIMS:
            for metric in ['em', 'micro_f1', 'precision', 'recall']:
                winner = max((f'rg/{b}/r{ratio}' for b in BASELINES), key=lambda k: scores[k, dim, metric])
                for other in controls:
                    lo, hi = interval(boot[ours, dim, metric] - boot[other, dim, metric])
                    row = dict(ratio=ratio, ours=ours, comparator=other, family='graph_v2', dimension=dim,
                        metric=metric, ours_value=scores[ours, dim, metric], comparator_value=scores[other, dim, metric],
                        delta=scores[ours, dim, metric] - scores[other, dim, metric],
                        delta_ci_low=lo, delta_ci_high=hi, best_main_baseline=(other == winner))
                    paired.append(row)
                    if other == winner:
                        best.append(row)
        for metric in ['location_errors_per_report', 'location_error_report_rate', 'false_findings_per_report', 'missing_findings_per_report', 'green_mean']:
            for other in controls:
                lo, hi = interval(green_boot[ours, metric] - green_boot[other, metric])
                paired.append(dict(ratio=ratio, ours=ours, comparator=other, family='green', dimension='report_common_valid',
                    metric=metric, ours_value=green_scores[ours, metric], comparator_value=green_scores[other, metric],
                    delta=green_scores[ours, metric] - green_scores[other, metric],
                    delta_ci_low=lo, delta_ci_high=hi, best_main_baseline=False))
    for name, data in [('graph_summary', summary), ('paired_differences', paired),
                       ('best_baseline_differences', best), ('green_summary', green_summary), ('cases', cases)]:
        with (a.output/(name+'.csv')).open('w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(data[0])); w.writeheader(); w.writerows(data)
    metadata = dict(graph_protocol=VERSION, source=str(source), source_sha256=sha(source),
        index_sha256=sha(index_path), bootstrap='paired patient-cluster percentile',
        resamples=a.resamples, seed=a.seed, independent_patients=len(patients), settings=len(groups),
        green_common_reports=len(green_common_ids), green_excluded_ids=sorted(set(canonical)-green_common_ids),
        note='Automatic labels, not clinical gold. CIs reflect sampling only; no multiplicity correction. Best baseline selected from observed scores. GREEN location includes all findings, including midline shift.',
        provenance=provenance)
    (a.output/'metadata.json').write_text(json.dumps(metadata, indent=2)+'\n')
    wb = openpyxl.Workbook(); wb.remove(wb.active)
    add_sheet(wb, 'Interpretation', [{'note': metadata['note']}])
    add_sheet(wb, 'Graph_V2', summary); add_sheet(wb, 'Best_baseline', best)
    add_sheet(wb, 'Paired_differences', paired); add_sheet(wb, 'GREEN', green_summary)
    add_sheet(wb, 'Cases', cases)
    wb.save(a.output/'spatial_comparison_r32_r256.xlsx')
    for r in best:
        if r['metric'] in ['em', 'micro_f1']:
            print(json.dumps(r), flush=True)
    for r in green_summary:
        print('GREEN', r['setting'], 'valid', r['n'], 'common', r['common_n'],
              r['common_location_errors_per_report'], r['common_location_error_report_rate'], flush=True)


if __name__ == '__main__':
    main()
