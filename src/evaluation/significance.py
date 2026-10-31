"""Fixed-checkpoint, patient-cluster paired inference (no model execution).

The statistic always retains the report metric's original aggregation. Patients
are the independent exchange/resampling units, not individual MRI sequences.
"""
from itertools import product
import math

import numpy as np
from sacrebleu.metrics import BLEU
from rouge_score.rouge_scorer import RougeScorer

METRICS = ('bleu4', 'rougeL', 'green', 'radgraph_simple', 'radgraph_partial',
           'radgraph_complete', 'rate', 'bertscore_precision', 'bertscore_recall',
           'bertscore_f1', 'answer_ce')
GROUPS = {k: g for g, keys in {
    'green': ('green',), 'radgraph': METRICS[3:6], 'rate': ('rate',),
    'bertscore': METRICS[7:10],
}.items() for k in keys}


def bleu_metric():
    return BLEU(tokenize='13a', lowercase=False, smooth_method='exp', effective_order=False)


def lexical_vectors(rows):
    """Cache sufficient statistics, never substitute averaged sentence BLEU."""
    predictions = [' '.join(r['prediction'].split()) for r in rows]
    references = [' '.join(r['reference'].split()) for r in rows]
    bleu = bleu_metric()
    statistics = bleu._extract_corpus_statistics(predictions, [references])
    rouge = RougeScorer(['rougeL'], use_stemmer=True)
    return {
        'bleu4': {r['id']: s for r, s in zip(rows, statistics)},
        'rougeL': {r['id']: [100 * rouge.score(ref, pred)['rougeL'].fmeasure, 1]
                   for r, ref, pred in zip(rows, references, predictions)},
    }


def statistic(metric, totals):
    totals = np.asarray(totals, dtype=float)
    if metric == 'bleu4':
        return float(bleu_metric()._compute_score_from_stats(totals.tolist()).score)
    if totals[1] <= 0:
        raise ValueError('Nonpositive metric denominator')
    return float(totals[0] / totals[1])


def holm_adjust(pvalues, family_size=None):
    """Holm; untested members of a fixed family cannot be rejected.

Using the full family size with only completed p-values gives conservative
adjustments, equivalent to padding the family with p=1 for adjustment only.
Missing observations/results are never filled with numeric scores or raw p's.
"""
    p = np.asarray(pvalues, dtype=float)
    m = len(p) if family_size is None else family_size
    if m < len(p) or not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError('Invalid p-values/family size')
    order = np.argsort(p, kind='stable')
    adjusted = np.empty(len(p))
    adjusted[order] = np.minimum(1, np.maximum.accumulate(p[order] * (m - np.arange(len(p)))))
    return adjusted.tolist()


def paired_test(rows_a, rows_b, vectors_a, vectors_b, metric, *,
                permutations=10000, bootstraps=10000, seed=20261030):
    """A minus B; two-sided |delta| permutation test and percentile 95% CI.

Requires independent patient clusters and exchangeability of the whole paired
patient vectors under the permutation null. CI is marginal, not simultaneous.
"""
    if permutations < 1 or bootstraps < 1:
        raise ValueError('Positive permutation/bootstrap counts required')
    a = {r['id']: r for r in rows_a}
    b = {r['id']: r for r in rows_b}
    if len(a) != len(rows_a) or len(b) != len(rows_b) or a.keys() != b.keys():
        raise ValueError('Duplicate IDs or unequal test-set coverage')
    for key in a:
        for field in ('patient_id', 'reference', 'case_id', 'sequence'):
            if a[key][field] != b[key][field]:
                raise ValueError(f'Paired {field} mismatch: {key}')
    if set(vectors_a) - a.keys() or set(vectors_b) - b.keys():
        raise ValueError('Score IDs outside the declared test set')
    ids = sorted(set(vectors_a) & set(vectors_b))
    if not ids:
        raise ValueError('No common valid samples')
    patients = sorted({a[i]['patient_id'] for i in ids})
    if len(patients) < 2:
        raise ValueError('At least two independent patient clusters required')
    patient_index = {p: i for i, p in enumerate(patients)}
    x = np.asarray([vectors_a[i] for i in ids], dtype=float)
    y = np.asarray([vectors_b[i] for i in ids], dtype=float)
    if x.shape != y.shape or x.ndim != 2 or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError('Nonfinite scores or incompatible sufficient statistics')
    xa = np.zeros((len(patients), x.shape[1]))
    xb = np.zeros_like(xa)
    for key, vx, vy in zip(ids, x, y):
        j = patient_index[a[key]['patient_id']]
        xa[j] += vx
        xb[j] += vy
    score_a, score_b = statistic(metric, xa.sum(0)), statistic(metric, xb.sum(0))
    observed = score_a - score_b
    midpoint = (xa + xb).sum(0) / 2
    half_diff = (xa - xb) / 2
    rng = np.random.default_rng(seed)
    exact = len(patients) <= 20 and 2 ** len(patients) <= permutations
    count = 2 ** len(patients) if exact else permutations
    extreme = 0
    tolerance = 64 * np.finfo(float).eps * max(1, abs(observed))
    exact_signs = iter(product((-1., 1.), repeat=len(patients))) if exact else None
    for start in range(0, count, 256):
        size = min(256, count - start)
        signs = (np.array([next(exact_signs) for _ in range(size)]) if exact else
                 rng.integers(0, 2, size=(size, len(patients))) * 2 - 1)
        shifts = signs @ half_diff
        for shift in shifts:
            delta = statistic(metric, midpoint + shift) - statistic(metric, midpoint - shift)
            extreme += abs(delta) >= abs(observed) - tolerance
    p = extreme / count if exact else (extreme + 1) / (count + 1)
    # Separate deterministic streams keep CI unchanged if permutation budget changes.
    bootstrap_rng = np.random.default_rng(np.random.SeedSequence([seed, 1]))
    draws = []
    for start in range(0, bootstraps, 256):
        ix = bootstrap_rng.integers(0, len(patients), size=(min(256, bootstraps-start), len(patients)))
        for sa, sb in zip(xa[ix].sum(1), xb[ix].sum(1)):
            draws.append(statistic(metric, sa) - statistic(metric, sb))
    low, high = np.quantile(draws, [.025, .975])
    return {
        'score_ours': score_a, 'score_baseline': score_b,
        'delta_ours_minus_baseline': observed,
        'improvement': -observed if metric == 'answer_ce' else observed,
        'ci95_low': float(low), 'ci95_high': float(high), 'p_raw': float(p),
        'p_mc_se': 0. if exact else math.sqrt(p * (1-p) / (count+1)),
        'patients': len(patients), 'common_valid_samples': len(ids),
        'expected_samples': len(a), 'valid_ours': len(vectors_a), 'valid_baseline': len(vectors_b),
        'excluded_ids': sorted(a.keys() - set(ids)),
        'permutations': count, 'permutation_mode': 'exact' if exact else 'monte_carlo',
        'bootstrap_repeats': bootstraps, 'resampling_seed': seed,
        'scope': 'fixed_checkpoints_patient_sampling',
        'missing_policy': 'pairwise_common_valid',
        'ci_scope': 'marginal_percentile_95_not_multiplicity_adjusted',
    }
