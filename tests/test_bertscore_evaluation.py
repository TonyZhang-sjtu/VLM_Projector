"""BERTScore routing, provenance, aggregation and failure contracts (no downloads)."""
from argparse import Namespace
import copy
import math

import pytest

from src.cli.evaluate_checkpoint import combine_metrics, make_commands, metric_selection
from src.evaluation.clinical import (
    METRIC_KEYS, aggregate_clinical, clinical_bootstrap, score_predictions, validate_result,
)
from src.evaluation import clinical_runtime as runtime


def reports():
    return [dict(id=str(i), patient_id='p0' if i < 2 else 'p1', case_id=str(i // 2),
                 cohort='WMH', sequence='FLAIR', reference=f'Lesion {i}.', prediction=f'Report {i}.')
            for i in range(3)]


class FakeBert:
    def __init__(self, calls):
        self.calls = calls

    def score(self, ref, pred):
        self.calls.append((ref, pred))
        p, r = (.2, .8) if pred.endswith('0.') else (.8, .2)
        return {'status': 'ok', 'values': dict(zip(METRIC_KEYS['bertscore'], (p, r, 2*p*r/(p+r)))),
                'details': {}}

    def close(self):
        pass


def test_selected_and_all_routes_and_combined_output(tmp_path):
    args = Namespace(metrics=['bertscore'], output_dir=tmp_path, device='cuda:2',
                     scoring_device='cpu', repeats=20)
    gen, scoring = make_commands(args, tmp_path/'config.yaml', tmp_path/'ids.json')
    assert '--skip-answer-ce' in gen
    assert scoring[scoring.index('--metrics')+1:] == ['bertscore']
    assert scoring[scoring.index('--device')+1] == 'cpu'
    args.metrics = metric_selection(['all'])
    gen, scoring = make_commands(args, tmp_path/'config.yaml', tmp_path/'ids.json')
    assert '--skip-answer-ce' not in gen
    assert scoring[scoring.index('--metrics')+1:] == ['green', 'radgraph', 'rate', 'bertscore']
    scores = {key: {'mean': .8} for key in METRIC_KEYS['bertscore']}
    merged = combine_metrics({'status': 'complete'}, {'status': 'complete', 'clinical': {'overall': scores}},
                             ['bertscore'])
    assert merged == {key: .8 for key in METRIC_KEYS['bertscore']}


def test_only_requested_dependencies_and_assets(monkeypatch, tmp_path):
    inspected = []
    def version(name):
        inspected.append(name)
        assert name not in ('green_score', 'radgraph', 'RaTEScore', 'medspacy')
        return 'test'
    monkeypatch.setattr(runtime.metadata, 'version', version)
    paths = []
    def verify(path):
        paths.append(path.name)
        return {'revision': 'test'}
    monkeypatch.setattr(runtime, 'verify_asset', verify)
    monkeypatch.setattr(runtime, 'package_digest', lambda name: name)
    specs = runtime.specifications(['bertscore'], tmp_path, 'cpu')
    assert 'bert-score' in inspected and paths == ['bert-score']
    assert specs['bertscore']['settings']['num_layers'] == 9


def test_raw_scores_not_clipped_and_nonfinite_rejected():
    entry = {'status': 'ok', 'values': {key: -.01 for key in METRIC_KEYS['bertscore']}}
    validate_result('bertscore', entry)
    entry['values']['bertscore_f1'] = 1.00000012
    validate_result('bertscore', entry)
    for invalid in (math.nan, math.inf, True):
        entry['values']['bertscore_f1'] = invalid
        with pytest.raises(ValueError):
            validate_result('bertscore', entry)


def test_per_report_f1_cache_and_patient_intervals(tmp_path):
    calls = []
    spec = {'bertscore': {'settings': dict(runtime.BERTSCORE_SETTINGS), 'revision': 'a'}}
    first, run = score_predictions(reports(), tmp_path, spec, lambda _: FakeBert(calls))
    assert run['bertscore']['model_calls'] == 3
    _, replay = score_predictions(reports(), tmp_path, spec, lambda _: pytest.fail('Unexpected model load'))
    assert replay['bertscore']['cache_hits'] == 3
    summary = aggregate_clinical(first, ['bertscore'], 30)
    assert summary['overall']['bertscore_f1']['mean'] == pytest.approx(.32)
    # The harmonic mean of aggregate P/R is .48 here, NOT the mean report F1.
    assert summary['overall']['bertscore_f1']['mean'] != pytest.approx(.48)
    paired = clinical_bootstrap(first, ['bertscore'], 30, paired_rows=list(reversed(first)))
    assert all(v['ci95'] == [0., 0.] for v in paired['metrics'].values())
    changed = copy.deepcopy(spec)
    changed['bertscore']['settings']['num_layers'] = 8
    score_predictions(reports(), tmp_path, changed, lambda _: FakeBert(calls))
    assert len(calls) == 6
    bad = copy.deepcopy(first)
    bad[0]['clinical']['bertscore'].update(status='input_too_long', values={})
    incomplete = aggregate_clinical(bad, ['bertscore'], 20)
    assert incomplete['status'] == 'incomplete'
    assert incomplete['overall']['bertscore_f1']['mean'] is None
    assert incomplete['overall']['bertscore_f1']['valid'] == 2
    assert incomplete['bootstrap']['metrics']['bertscore_f1']['ci95'] is None


def test_empty_prediction_does_not_load_encoder(tmp_path):
    data = reports()[:1]
    data[0]['prediction'] = '  '
    scored, _ = score_predictions(data, tmp_path, {'bertscore': {}},
                                  lambda _: pytest.fail('Empty prediction loaded encoder'))
    assert scored[0]['clinical']['bertscore']['status'] == 'empty_prediction'
    assert scored[0]['clinical']['bertscore']['values'] == {}
