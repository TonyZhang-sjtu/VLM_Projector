"""Contract tests: alignment, cache invalidation, failure denominators, patient clusters."""
import copy
import json
import pytest
from src.evaluation.clinical import (
    aggregate_clinical, clinical_bootstrap, digest, read_jsonl, score_predictions,
    validate_pair, validate_rows,
)
from src.evaluation.clinical_scorers import green_structure_error


def rows():
    return [dict(id=str(i),patient_id='p0' if i<2 else 'p1',case_id=str(i//2),
                 cohort='WMH',sequence='FLAIR',reference=f'Lesion {i}.',prediction=f'Report {i}.') for i in range(3)]


class Fake:
    def __init__(self, calls): self.calls=calls
    def score(self, ref, pred):
        self.calls.append((ref,pred))
        return {'status':'ok','values':{'green':1. if pred.endswith('2.') else 0.},'details':{}}
    def close(self): pass


def test_cache_text_model_and_reference_invalidation(tmp_path):
    data=rows(); calls=[];factory=lambda name:Fake(calls)
    first,run=score_predictions(data,tmp_path,{'green':{'revision':'a'}},factory)
    assert len(calls)==3 and run['green']['model_calls']==3
    second,run=score_predictions(list(reversed(data)),tmp_path,{'green':{'revision':'a'}},lambda _:pytest.fail('cached run loaded model'))
    assert run['green']['cache_hits']==3 and second[0]['id']=='2'
    changed=copy.deepcopy(data);changed[0]['prediction']='Changed'
    score_predictions(changed,tmp_path,{'green':{'revision':'a'}},factory)
    assert len(calls)==4
    changed[1]['reference']='Changed reference'
    score_predictions(changed,tmp_path,{'green':{'revision':'a'}},factory)
    assert len(calls)==5
    score_predictions(data,tmp_path,{'green':{'revision':'b'}},factory)
    assert len(calls)==8


def test_coverage_and_failure_denominator(tmp_path):
    data=rows()
    with pytest.raises(ValueError,match='Duplicate'):validate_rows(data+data[:1])
    with pytest.raises(ValueError,match='coverage'):validate_rows(data,['0','1'])
    data[0]['prediction']=''
    scored,_=score_predictions(data,tmp_path,{'green':{}},lambda _:Fake([]))
    summary=aggregate_clinical(scored,['green'],20)
    assert summary['status']=='incomplete'
    assert summary['overall']['green']['mean'] is None
    assert summary['overall']['green']['valid']==2
    assert summary['overall']['green']['valid_only_mean']==.5
    assert summary['bootstrap']['metrics']['green']['ci95'] is None


def test_patient_bootstrap_and_pair_guard(tmp_path):
    scored,_=score_predictions(rows(),tmp_path,{'green':{}},lambda _:Fake([]))
    summary=aggregate_clinical(scored,['green'],100,11)
    assert summary['overall']['green']['mean']==pytest.approx(1/3)
    assert summary['hierarchy']['patient_id']['macro']['green']==.5
    paired=clinical_bootstrap(scored,['green'],100,11,list(reversed(scored)))
    assert paired['metrics']['green']['ci95']==[0.,0.]
    assert paired['metrics']['green']['patient_macro_ci95']==[0.,0.]
    # Reproduce cluster draws independently, so sample-level resampling cannot pass.
    import numpy as np
    draws=np.random.default_rng(11).integers(0,2,size=(100,2))
    expected=np.array([sum(d)/(2*sum(d==0)+sum(d==1)) for d in draws])
    assert summary['bootstrap']['metrics']['green']['ci95']==pytest.approx(np.percentile(expected,[2.5,97.5]))
    corrupted=copy.deepcopy(scored);corrupted[0]['clinical']['green']['scorer_fingerprint']='other'
    with pytest.raises(ValueError,match='configuration'):validate_pair(scored,corrupted,['green'])
    corrupted=copy.deepcopy(scored);corrupted[0]['prediction']='Different prediction'
    with pytest.raises(ValueError,match='texts'):validate_pair(scored,corrupted,['green'])


def test_corrupt_cache_is_recomputed_and_failed_scores_retried(tmp_path):
    scored,_=score_predictions(rows()[:1],tmp_path,{'green':{}},lambda _:Fake([]))
    path=next((tmp_path/'cache/green').glob('*.json'))
    path.write_text('{interrupted')
    calls=[]
    _,run=score_predictions(rows()[:1],tmp_path,{'green':{}},lambda _:Fake(calls))
    assert len(calls)==1 and run['green']['invalid_cache']==1
    entry=json.loads(path.read_text());entry['status']='parse_error';entry['values']={}
    path.write_text(json.dumps(entry))
    score_predictions(rows()[:1],tmp_path,{'green':{}},lambda _:Fake(calls))
    assert len(calls)==2


def test_unicode_reports_round_trip_to_utf8_scored_jsonl(tmp_path):
    row=rows()[:1][0]
    row['reference']='Patient’s MRI shows a lesion.'
    row['prediction']='Patient’s MRI shows no lesion.'
    scored,_=score_predictions([row],tmp_path,{'green':{}},lambda _:Fake([]))
    saved=read_jsonl(tmp_path/'clinical_scores.jsonl')
    assert saved[0]['reference']==row['reference']
    assert saved[0]['prediction']==row['prediction']
    assert scored[0]['prediction']==row['prediction']


def test_green_parser_rejects_silent_zero_and_invalid_counts():
    assert green_structure_error('Unstructured output')
    assert green_structure_error('[Clinically Significant Errors]:\nNo errors.\n\n[Matched Findings]:\n1. Normal brain.') is None
    assert green_structure_error('[Clinically Significant Errors]:\n(a) False finding: two.\n\n[Matched Findings]:\n1. Normal brain.')
    assert green_structure_error('[Clinically Significant Errors]:\n(a) False finding: 2. Lesions.\n\n[Matched Findings]:\n0. None.') is None
