import numpy as np
import pytest

from src.evaluation.significance import (
    bleu_metric, lexical_vectors, statistic, paired_test, holm_adjust,
)
from src.cli.significance_queue import eligible_gpus


def rows(n=6):
    return [dict(id=str(i), patient_id=str(i//2), case_id=str(i//2), sequence=str(i%2),
                 reference='the cat sat on the mat', prediction='the cat sat on the mat')
            for i in range(n)]


def test_exchange_unit_is_patient_not_sequence_and_zero_difference():
    r = rows()
    a = {x['id']: [1., 1] for x in r}
    b = {x['id']: [0., 1] for x in r}
    result = paired_test(r, r, a, b, 'green', permutations=100, bootstraps=50)
    # Three clusters -> 8 exact exchanges, 2 as extreme, despite six reports.
    assert result['patients'] == 3
    assert result['permutations'] == 8
    assert result['p_raw'] == .25
    assert result['ci95_low'] == result['ci95_high'] == 1
    null = paired_test(r, r, a, a, 'green', permutations=100, bootstraps=50)
    assert null['p_raw'] == 1 and null['ci95_low'] == null['ci95_high'] == 0


def test_common_valid_pairing_does_not_impute_or_drop_other_valid_sequences():
    r = rows()
    a = {x['id']: [1., 1] for x in r if x['id'] != '0'}
    b = {x['id']: [.5, 1] for x in r if x['id'] != '2'}
    result = paired_test(r, list(reversed(r)), a, b, 'green', permutations=100, bootstraps=30)
    assert result['common_valid_samples'] == 4
    assert result['patients'] == 3
    assert result['excluded_ids'] == ['0', '2']
    assert result['delta_ours_minus_baseline'] == .5


def test_patient_and_reference_mismatches_are_rejected():
    r = rows()
    a = {x['id']: [1., 1] for x in r}
    for field in ('patient_id', 'reference'):
        broken = [dict(x) for x in r]
        broken[0][field] = 'changed'
        with pytest.raises(ValueError, match='mismatch'):
            paired_test(r, broken, a, a, 'green')
    with pytest.raises(ValueError, match='coverage'):
        paired_test(r, r[:-1], a, a, 'green')


def test_ce_retains_token_weighting_and_effect_direction():
    r = rows(2)
    r[1]['patient_id'] = 'second'
    a = {'0': [1., 1], '1': [18., 9]}
    b = {'0': [2., 1], '1': [18., 9]}
    result = paired_test(r, r, a, b, 'answer_ce', permutations=10, bootstraps=20)
    assert result['score_ours'] == 1.9  # Not the average of per-report CEs (1.5).
    assert result['score_baseline'] == 2.
    assert result['improvement'] == pytest.approx(.1)


def test_bleu_sufficient_statistics_match_corpus_not_sentence_average():
    r = rows(2)
    r[1].update(prediction='z ' * 30, reference='a ' * 30, patient_id='second')
    vectors = lexical_vectors(r)['bleu4']
    aggregated = statistic('bleu4', np.asarray(list(vectors.values())).sum(0))
    direct = bleu_metric().corpus_score([x['prediction'].strip() for x in r],
                                        [[x['reference'].strip() for x in r]]).score
    per_report_mean = np.mean([statistic('bleu4', v) for v in vectors.values()])
    assert aggregated == pytest.approx(direct)
    assert aggregated != pytest.approx(per_report_mean)
    # Duplicate resampled patients require duplicate sufficient statistics too.
    duplicated = statistic('bleu4', np.asarray([vectors['0'], vectors['0'], vectors['1']]).sum(0))
    subset = [r[0], r[0], r[1]]
    direct_duplicate = bleu_metric().corpus_score([x['prediction'].strip() for x in subset],
                         [[x['reference'].strip() for x in subset]]).score
    assert duplicated == pytest.approx(direct_duplicate)


def test_randomization_is_reproducible_and_never_returns_zero():
    r = rows(40)
    a = {x['id']: [1., 1] for x in r}
    b = {x['id']: [0., 1] for x in r}
    one = paired_test(r, r, a, b, 'green', permutations=50, bootstraps=50, seed=12)
    two = paired_test(r, r, a, b, 'green', permutations=50, bootstraps=50, seed=12)
    assert one == two
    assert one['p_raw'] >= 1/51


def test_holm_full_and_prespecified_incomplete_family():
    assert holm_adjust([.01, .04, .03]) == pytest.approx([.03, .06, .06])
    assert holm_adjust([.0001, .01, .3], 77) == pytest.approx([.0077, .76, 1])
    assert holm_adjust([], 77) == []
    with pytest.raises(ValueError):
        holm_adjust([.01, .02], 1)


def test_gpu_selection_requires_only_free_memory_and_includes_exact_threshold(monkeypatch):
    calls = []
    def query(command, **kwargs):
        calls.append(command)
        # Process lists, utilization, and used memory must not be queried.
        assert command == ['nvidia-smi', '--query-gpu=index,memory.free',
                           '--format=csv,noheader,nounits']
        return '0, 20397\n1, 20000\n2, 19999\n3, 24000\n'
    monkeypatch.setattr('src.cli.significance_queue.subprocess.check_output', query)
    assert eligible_gpus() == ['0', '1', '3']
    assert len(calls) == 1
