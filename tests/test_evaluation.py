import pytest
from src.evaluation.metrics import evaluate_rows,patient_bootstrap


def rows():
    return [{'id':str(i),'patient_id':str(i//2),'case_id':str(i//2),'cohort':'BraTS_GLI' if i<2 else 'WMH',
        'sequence':'FLAIR','prediction':'A small lesion is visible in the left frontal lobe.',
        'reference':'A small lesion is visible in the left frontal lobe.','hit_generation_limit':False} for i in range(4)]


def test_metrics_coverage_strata_and_bootstrap():
    r=rows();m=evaluate_rows(r,[x['id'] for x in r],repeats=12,excluded_ids=['0'])
    assert abs(m['overall']['bleu4']-100)<1e-6 and m['overall']['rougeL']==100
    assert m['hierarchy']['patient_id']['units']==2 and len(m['strata']['source'])==2
    assert m['review_sensitivity']['remaining_count']==3
    ci=patient_bootstrap(r,12,paired_rows=r)
    assert ci['bleu4_95ci']==[0.,0.] and ci['rougeL_95ci']==[0.,0.]
    with pytest.raises(ValueError):evaluate_rows(r+r[:1],[x['id'] for x in r],0)
    with pytest.raises(ValueError):evaluate_rows(r,['missing'],0)


def test_empty_predictions_are_counted():
    r=rows();r[0]['prediction']=''
    m=evaluate_rows(r,[x['id'] for x in r],0)
    assert m['coverage']['empty']==1 and m['status']=='generation_failures'
