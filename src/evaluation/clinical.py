"""Offline clinical scoring, resumable per-pair cache, and patient statistics.

This module deliberately does not import the scorers or load GPU models.
"""
import hashlib
import json
import math
import os
import time
from collections import Counter, defaultdict
from pathlib import Path


METRIC_KEYS = {
    'green': ('green',),
    'radgraph': ('radgraph_simple', 'radgraph_partial', 'radgraph_complete'),
    'rate': ('rate',),
    'bertscore': ('bertscore_precision', 'bertscore_recall', 'bertscore_f1'),
}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     allow_nan=False).encode()).hexdigest()


def file_digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def write_json(path, value):
    """Replace atomically; an interrupted write cannot become a valid cache entry."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n',
                    encoding='utf-8')
    temp.replace(path)


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()]


def validate_rows(rows, expected_ids=None):
    if not rows:
        raise ValueError('Cannot score an empty prediction set')
    for row in rows:
        for field in ('id', 'patient_id', 'case_id', 'cohort', 'sequence', 'reference', 'prediction'):
            if not isinstance(row.get(field), str):
                raise ValueError(f'{field} must be a string: {row.get("id")}')
            if field != 'prediction' and not row[field].strip():
                raise ValueError(f'{field} is empty: {row.get("id")}')
    ids = [r['id'] for r in rows]
    if len(set(ids)) != len(ids):
        raise ValueError('Duplicate prediction IDs')
    if expected_ids is not None:
        if not isinstance(expected_ids, list) or not all(isinstance(i, str) for i in expected_ids):
            raise ValueError('Expected IDs must be a JSON array of strings')
        if len(expected_ids) != len(set(expected_ids)) or set(ids) != set(expected_ids):
            raise ValueError('Prediction coverage mismatch')


def validate_result(metric, result):
    if result.get('status') != 'ok':
        if result.get('values'):
            raise ValueError('Failed scoring must not contribute numeric values')
        return
    if set(result['values']) != set(METRIC_KEYS[metric]):
        raise ValueError(f'Wrong score keys for {metric}')
    for key, value in result['values'].items():
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f'Non-finite/non-numeric score: {key}')
        # Raw BERTScore is based on cosine similarity, can be negative, and can
        # exceed 1 by floating-point roundoff. Preserve the official values.
        if metric not in ('rate', 'bertscore') and not 0 <= value <= 1:
            raise ValueError(f'Out-of-range score: {key}')


def score_predictions(rows, output_dir, specs, factory, cache_dir=None):
    """Load one scorer at a time, and only if an uncached valid pair needs it.

    Cache identity is text + full scorer specification, independent of row ID.
    Failures remain visible and are retried on subsequent invocations.
    """
    validate_rows(rows)
    output = Path(output_dir)
    cache = Path(cache_dir) if cache_dir else output / 'cache'
    scored = [dict(row, clinical={}) for row in rows]
    run = {}
    for metric, spec in specs.items():
        fingerprint = digest(spec)
        scorer = None
        counts = Counter()
        start = time.perf_counter()
        try:
            for row in scored:
                pair = {'reference': row['reference'], 'prediction': row['prediction']}
                key = digest({'pair': pair, 'scorer_fingerprint': fingerprint})
                path = cache / metric / f'{key}.json'
                entry = None
                if path.exists():
                    try:
                        entry = json.loads(path.read_text(encoding='utf-8'))
                        if (entry['cache_key'] != key or entry['scorer_fingerprint'] != fingerprint
                                or entry['pair_hash'] != digest(pair)):
                            raise ValueError('Cache identity mismatch')
                        validate_result(metric, entry)
                        if entry['status'] != 'ok':
                            entry = None
                    except (ValueError, KeyError, TypeError):
                        entry = None
                        counts['invalid_cache'] += 1
                if entry is not None:
                    counts['cache_hits'] += 1
                else:
                    if not row['prediction'].strip():
                        entry = {'status': 'empty_prediction', 'values': {}, 'details': {}}
                    else:
                        # Initialization/dependency errors are fatal, not hundreds of missing scores.
                        if scorer is None:
                            scorer = factory(metric)
                        tick = time.perf_counter()
                        try:
                            entry = scorer.score(row['reference'], row['prediction'])
                            validate_result(metric, entry)
                        except Exception as exc:
                            entry = {'status': 'scoring_error', 'values': {},
                                     'details': {'error_type': type(exc).__name__, 'error': str(exc)}}
                        entry['seconds'] = time.perf_counter() - tick
                        counts['model_calls'] += 1
                    entry.update(cache_key=key, scorer_fingerprint=fingerprint, pair_hash=digest(pair))
                    write_json(path, entry)
                row['clinical'][metric] = entry
                counts['rows'] += 1
                counts[entry['status']] += 1
        finally:
            if scorer is not None:
                scorer.close()
        run[metric] = dict(counts, wall_seconds=time.perf_counter() - start,
                           scorer_fingerprint=fingerprint)
        # Checkpoint after each metric as well as after each pair in the cache.
        output.mkdir(parents=True, exist_ok=True)
        temp = output / f'clinical_scores.{os.getpid()}.tmp'
        temp.write_text(''.join(json.dumps(r, ensure_ascii=False, allow_nan=False) + '\n' for r in scored),
                        encoding='utf-8')
        temp.replace(output / 'clinical_scores.jsonl')
    return scored, run


def values_for(rows, metric, key):
    return [r['clinical'][metric]['values'].get(key)
            if r['clinical'][metric]['status'] == 'ok' else None for r in rows]


def summarize(rows, metrics):
    result = {}
    for metric in metrics:
        statuses = dict(Counter(r['clinical'][metric]['status'] for r in rows))
        for key in METRIC_KEYS[metric]:
            values = values_for(rows, metric, key)
            valid = [v for v in values if v is not None]
            mean = sum(valid) / len(valid) if valid else None
            result[key] = {
                'mean': mean if len(valid) == len(rows) else None,
                'valid_only_mean': mean, 'valid': len(valid), 'expected': len(rows),
                'status': 'complete' if len(valid) == len(rows) else 'incomplete',
                'statuses': statuses,
            }
    return result


def clinical_bootstrap(rows, metrics, repeats=1000, seed=42, paired_rows=None):
    import numpy as np
    if repeats < 0:
        raise ValueError('repeats must be nonnegative')
    if paired_rows is not None:
        validate_pair(rows, paired_rows, metrics)
        lookup = {r['id']: r for r in paired_rows}
        paired_rows = [lookup[r['id']] for r in rows]
    patients = sorted({r['patient_id'] for r in rows})
    result = {'unit': 'patient_id', 'patients': len(patients), 'repeats': repeats,
              'seed': seed, 'quantity': 'paired B minus A' if paired_rows is not None else 'score',
              'status': 'complete' if repeats else 'not_run', 'metrics': {}}
    if not repeats:
        return result
    indices = [[i for i, r in enumerate(rows) if r['patient_id'] == p] for p in patients]
    rng = np.random.default_rng(seed)
    # Reuse the same patient draws for all metrics and both compared systems.
    draws = rng.integers(0, len(patients), size=(repeats, len(patients)))
    sizes = np.array([len(i) for i in indices])
    for metric in metrics:
        for key in METRIC_KEYS[metric]:
            values = values_for(rows, metric, key)
            other = values_for(paired_rows, metric, key) if paired_rows is not None else None
            if None in values or (other is not None and None in other):
                result['metrics'][key] = {'status': 'incomplete', 'ci95': None}
                result['status'] = 'incomplete'
                continue
            vector = np.asarray(values)
            if other is not None:
                vector = np.asarray(other) - vector
            sums = np.array([vector[i].sum() for i in indices])
            report_draws = sums[draws].sum(axis=1) / sizes[draws].sum(axis=1)
            macro_draws = (sums / sizes)[draws].mean(axis=1)
            result['metrics'][key] = {
                'status': 'complete', 'point': float(vector.mean()),
                'ci95': np.percentile(report_draws, [2.5, 97.5]).tolist(),
                'patient_macro_point': float((sums / sizes).mean()),
                'patient_macro_ci95': np.percentile(macro_draws, [2.5, 97.5]).tolist(),
            }
    return result


def aggregate_clinical(rows, metrics, repeats=1000, seed=42):
    validate_rows(rows)
    overall = summarize(rows, metrics)
    strata = {}
    for field in ('cohort', 'sequence', 'source'):
        groups = defaultdict(list)
        for row in rows:
            value = ('BraTS' if row['cohort'].startswith('BraTS_') else row['cohort']) if field == 'source' else row[field]
            groups[value].append(row)
        strata[field] = {name: summarize(group, metrics) for name, group in sorted(groups.items())}
    hierarchy = {}
    for field in ('case_id', 'patient_id'):
        groups = defaultdict(list)
        for row in rows:
            groups[row[field]].append(row)
        summaries = [summarize(group, metrics) for group in groups.values()]
        hierarchy[field] = {'units': len(groups), 'macro': {
            key: sum(s[key]['mean'] for s in summaries) / len(summaries)
            if all(s[key]['mean'] is not None for s in summaries) else None for key in overall}}
    anomalies = {}
    for metric in metrics:
        anomalies[metric] = {
            'statuses': dict(Counter(r['clinical'][metric]['status'] for r in rows)),
            'warnings': dict(Counter(w for r in rows for w in r['clinical'][metric].get('details', {}).get('warnings', []))),
        }
    return {'status': 'complete' if all(v['status'] == 'complete' for v in overall.values()) else 'incomplete',
            'samples': len(rows), 'overall': overall, 'strata': strata, 'hierarchy': hierarchy,
            'coverage': {'empty_predictions': sum(not r['prediction'].strip() for r in rows),
                         'generation_limit': sum(bool(r.get('hit_generation_limit')) for r in rows)},
            'anomalies': anomalies, 'bootstrap': clinical_bootstrap(rows, metrics, repeats, seed)}


def validate_pair(a, b, metrics):
    validate_rows(a)
    validate_rows(b, [r['id'] for r in a])
    lookup = {r['id']: r for r in b}
    for row in a:
        other = lookup[row['id']]
        for field in ('reference', 'patient_id', 'case_id', 'cohort', 'sequence'):
            if row[field] != other[field]:
                raise ValueError(f'Paired {field} mismatch: {row["id"]}')
        for metric in metrics:
            for item in (row, other):
                entry = item['clinical'][metric]
                validate_result(metric, entry)
                if entry['pair_hash'] != digest({k: item[k] for k in ('reference', 'prediction')}):
                    raise ValueError('Clinical scores do not belong to these prediction texts')
            if row['clinical'][metric]['scorer_fingerprint'] != other['clinical'][metric]['scorer_fingerprint']:
                raise ValueError(f'Paired scorer configuration mismatch: {metric}')
