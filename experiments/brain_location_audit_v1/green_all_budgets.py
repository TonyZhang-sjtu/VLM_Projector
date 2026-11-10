"""Read-only GREEN component audit for the 1108 main RG benchmark.

Exports plain CSV/JSON, never overwrites original results. All-budget common
reports are primary; within-budget common and all-valid sets are sensitivity
views. CIs are paired patient-cluster percentile intervals, not equivalence tests.
"""
import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path

import numpy as np

from src.evaluation.clinical import digest, file_digest
from src.evaluation.significance_io import scorer_contract

ROOT = Path(__file__).resolve().parents[2]
RATIOS = [8, 16, 32, 64, 128, 256]
BASELINES = ['AvgPool3D', 'Concat3D', 'ConvPool3D', 'LocalPacker3D',
             'Perceiver_NV', 'Adaptive_Merge3D', 'Token_Select3D']
METRICS = ['a_false_finding', 'b_missing_finding', 'c_location',
           'd_severity', 'e_spurious_comparison', 'f_missing_comparison',
           'matched_findings', 'green', 'location_error_report_rate']


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--seed', type=int, default=20261108)
    p.add_argument('--resamples', type=int, default=10000)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    ip = ROOT/'results/anchorread3d_analysis_1108/artifact_reconciliation_1108.json'
    idx = json.loads(ip.read_text())
    assert file_digest(idx['master']) == idx['master_sha256']
    entries = {e['key']: e for e in idx['paper_settings'] if e['key'].startswith('rg/')
               and e['key'].split('/')[1] in BASELINES + ['A0', 'MLP']}
    assert len(entries) == 49
    observations = {}; coverage = []; provenance = []; cases = []
    ref_ids = None; reference_identity = None; canonical = None; contract_set = set()
    for key, e in sorted(entries.items()):
        cp = Path(e['paths']['metric']).parent/'scoring/clinical_scores.jsonl'
        rows = [json.loads(line) for line in cp.open() if line.strip()]
        pp = cp.parent/'scoring_provenance.json'
        prov = json.loads(pp.read_text()); spec = prov['scorers']['green']
        fp = digest(spec); contract_set.add(digest(scorer_contract(spec)))
        ids = {r['id'] for r in rows}
        assert len(rows) == len(ids) == 560
        identity = sorted((r['id'], r['patient_id'], r['reference']) for r in rows)
        if ref_ids is None:
            ref_ids, reference_identity = ids, identity
            canonical = {r['id']: r for r in rows}
        assert ids == ref_ids and identity == reference_identity
        values = {}; status = Counter(); repaired = 0; categories = set()
        for r in rows:
            g = r['clinical']['green']; assert g['scorer_fingerprint'] == fp
            status[g['status']] += 1
            d = g.get('details', {}); repaired += int(bool(d.get('format_repair')))
            if g['status'] != 'ok':
                cases.append(dict(setting=key, id=r['id'], status=g['status'], values=None))
                continue
            errors, names = d['significant_errors'], d['error_categories']
            assert len(errors) == len(names) == 6
            assert [name[:3] for name in names] == [f'({c})' for c in 'abcdef']
            assert 'anatomic location/position' in names[2].lower()
            assert all(isinstance(x, (int, float)) and x >= 0 and np.isfinite(x) for x in errors)
            categories.add(tuple(names))
            matched, green = d['matched_findings'], g['values']['green']
            den = matched + sum(errors)
            if den > 0:
                assert np.isclose(matched / den, green), (key, r['id'], matched, errors, green)
            obs = list(errors) + [matched, green, int(errors[2] > 0)]
            values[r['id']] = obs
            cases.append(dict(setting=key, id=r['id'], status='ok', values=obs))
        assert len(categories) == 1
        main_metric = json.loads(Path(e['paths']['metric']).read_text())
        assert np.isclose(np.mean([v[7] for v in values.values()]), main_metric['overall']['green'])
        observations[key] = values
        coverage.append(dict(setting=key, expected=560, valid=len(values),
            failed=560-len(values), statuses=dict(status), format_repairs=repaired,
            failed_ids=sorted(ids-set(values))))
        provenance.append(dict(setting=key, clinical_path=str(cp), clinical_sha256=file_digest(cp),
            provenance_sha256=file_digest(pp), scorer_fingerprint=fp,
            scorer_contract=digest(scorer_contract(spec)), device=spec.get('device_name')))
    assert len(contract_set) == 1
    common = set.intersection(*(set(v) for v in observations.values()))
    within = {ratio: set.intersection(*(set(observations[f'rg/{m}/r{ratio}'])
                                       for m in BASELINES + ['A0'])) for ratio in RATIOS}
    patients = sorted({r['patient_id'] for r in canonical.values()})
    assert len(patients) == 140
    pi = {p: i for i, p in enumerate(patients)}
    draws = np.random.default_rng(a.seed).multinomial(
        len(patients), np.full(len(patients), 1/len(patients)), size=a.resamples)
    summary, paired, best, ranks = [], [], [], []
    metrics_cache = {}; bootstrap_cache = {}
    for key, vals in sorted(observations.items()):
        ratio = int(key.split('/r')[-1])
        populations = {'all_valid': set(vals), 'all_budgets_common': common}
        if ratio != 1:
            populations['within_budget_common'] = within[ratio]
        for scope, selected in populations.items():
            stats = np.zeros((len(patients), 1+len(METRICS)), dtype=float)
            for rid in selected:
                stats[pi[canonical[rid]['patient_id']]] += [1]+vals[rid]
            totals = stats.sum(axis=0)
            sampled = draws @ stats
            for i, metric in enumerate(METRICS, start=1):
                value = float(totals[i]/totals[0])
                samples = sampled[:, i]/sampled[:, 0]
                lo, hi = np.percentile(samples, [2.5, 97.5])
                metrics_cache[scope, key, metric] = value
                bootstrap_cache[scope, key, metric] = samples
                summary.append(dict(scope=scope, setting=key, metric=metric, n=len(selected),
                    patients=len({canonical[rid]['patient_id'] for rid in selected}),
                    total=float(totals[i]), value=value, ci_low=float(lo), ci_high=float(hi)))
    for scope in ['all_budgets_common', 'within_budget_common']:
        for ratio in RATIOS:
            ours = f'rg/A0/r{ratio}'
            for metric in METRICS:
                higher = metric in ['matched_findings', 'green']
                keys = [f'rg/{b}/r{ratio}' for b in BASELINES]
                winner = (max if higher else min)(keys, key=lambda k: metrics_cache[scope, k, metric])
                ov = metrics_cache[scope, ours, metric]
                rank = 1+sum((metrics_cache[scope, k, metric]>ov if higher else
                              metrics_cache[scope, k, metric]<ov) for k in keys)
                ties = sum(np.isclose(metrics_cache[scope, k, metric],ov,atol=1e-12,rtol=0) for k in keys)
                ranks.append(dict(scope=scope, ratio=ratio, metric=metric, ours_value=ov,
                                  rank=rank, tied_baselines=ties))
                for key in keys:
                    other = metrics_cache[scope, key, metric]
                    delta = bootstrap_cache[scope, ours, metric]-bootstrap_cache[scope, key, metric]
                    lo, hi = np.percentile(delta, [2.5,97.5])
                    effect = (ov-other) * (1 if higher else -1)
                    row = dict(scope=scope, ratio=ratio, metric=metric, comparator=key,
                        ours_value=ov, baseline_value=other, ours_minus_baseline=ov-other,
                        ci_low=float(lo), ci_high=float(hi),
                        direction='better' if effect>1e-12 else 'worse' if effect < -1e-12 else 'tied',
                        interval_crosses_zero=bool(lo<=0<=hi), best_baseline=(key==winner))
                    paired.append(row)
                    if key==winner:best.append(row)
    def save(name, data):
        with (a.output/name).open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
    save('summary.csv',summary);save('paired_differences.csv',paired)
    save('best_baseline.csv',best);save('ranks.csv',ranks);save('coverage.csv',coverage)
    metadata=dict(index_sha256=file_digest(ip),master_sha256=idx['master_sha256'],
        settings=len(observations),ratios=RATIOS,metrics=METRICS,primary_scope='all_budgets_common',
        common_n=len(common),common_patients=len({canonical[rid]['patient_id'] for rid in common}),
        common_ids=sorted(common),excluded_ids=sorted(ref_ids-common),
        within_budget_n={r:len(v) for r,v in within.items()},resamples=a.resamples,seed=a.seed,
        bootstrap='paired patient-cluster percentile; no multiplicity adjustment',
        notes=['All error components are clinically significant errors per report, lower is better.',
               'No missing/invalid output is treated as zero. Same scoring contract across all settings.',
               'Intervals crossing zero do not establish equivalence.',
               'Counts reflect GREEN judgments, not human-validated lesion locations.',
               '16x uses the frozen 1108 table checkpoint; it may be superseded by ongoing training.'],
        provenance=provenance)
    (a.output/'metadata.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2)+'\n')
    (a.output/'cases.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in cases))
    print(json.dumps({k:v for k,v in metadata.items() if k not in ['provenance','common_ids','excluded_ids']},indent=2))
    for metric in METRICS:
        print('\nMETRIC',metric)
        for method in BASELINES+['A0']:
            print(method,' '.join(f'{metrics_cache["all_budgets_common",f"rg/{method}/r{ratio}",metric]:.4f}' for ratio in RATIOS))
        print('MLP 1x',f'{metrics_cache["all_budgets_common","rg/MLP/r1",metric]:.4f}')
    print('\nBEST BASELINE COMPARISONS')
    for r in best:
        if r['scope']=='all_budgets_common':print(json.dumps(r))


if __name__=='__main__':
    main()
