from types import SimpleNamespace
import numpy as np
import pytest
from src.evaluation.significance import paired_test, statistic, lexical_vectors
from src.evaluation.neuroqa import evaluate_rows
from .significance import paired_p, vqa_vectors


def toy():
    return [dict(id=str(i),patient_id=str(i//2),case_id=str(i//2),sequence=str(i%2),
                 reference='the cat sat on the mat',prediction='the cat sat on the mat') for i in range(6)]


def test_exact_patient_exchange_and_symmetry():
    rr=toy();a={r['id']:[1.,1] for r in rr};b={r['id']:[0.,1] for r in rr}
    assert paired_p(rr,rr,a,b,'green',100)['p_raw']==.25
    assert paired_p(rr,rr,a,a,'green',100)['p_raw']==1
    assert paired_p(rr,rr,b,a,'green',100)['p_raw']==.25
    assert paired_p(rr,rr,a,b,'green',100)['p_raw']==paired_test(rr,rr,a,b,'green',permutations=100,bootstraps=10)['p_raw']


def test_corpus_bleu_and_missing_are_not_sentence_average_or_imputation():
    a=toy();b=[{**r,'prediction':'wrong '*(i+1)} for i,r in enumerate(a)]
    va=lexical_vectors(a)['bleu4'];vb=lexical_vectors(b)['bleu4']
    actual=paired_p(a,b,va,vb,'bleu4',100)
    old=paired_test(a,b,va,vb,'bleu4',permutations=100,bootstraps=10)
    assert actual['p_raw']==old['p_raw']
    del va['0'];del vb['2']
    assert paired_p(a,b,va,vb,'bleu4',100)['common_valid_samples']==4


def test_vqa_macro_and_question_ce_reproduce_official_wrapper():
    samples=[];rr=[]
    # Unequal question counts per patient, category, and modality; duplicate QA views.
    for i,(patient,qa,seq,kind,cat,pred,gold) in enumerate([
        ('p0','q0','T1W','YN','loc','yes','yes'),
        ('p0','q0','T2W','YN','loc','no','yes'),
        ('p0','q1','T1W','YN','sig','no','no'),
        ('p1','q2','T2W','MCQ','loc','a','right'),
        ('p2','q3','T2W','Open','size','large lesion','small lesion'),
        ('p2','q3','T1W','Open','size','small lesion','small lesion')]):
        row=dict(neuroqa_id=qa,question_type=kind,category=cat,answer=gold,options=['right','left'],
                 dataset='toy',human_assessable=True,image_dependency='image-grounded',question='Where?')
        samples.append(SimpleNamespace(id=str(i),qa_id=qa,patient_id=patient,sequence=seq,row=row))
        rr.append(dict(id=str(i),prediction=pred,qa_ce=float(i+1)))
    metrics,scored=evaluate_rows(samples,rr)
    vv=vqa_vectors(scored)
    for name in ('patient_macro_accuracy','category_macro_accuracy','modality_macro_accuracy'):
        assert statistic(name,np.array(list(vv[name].values())).sum(0))==pytest.approx(metrics[name])
    assert statistic('qa_ce',np.array(list(vv['qa_ce'].values())).sum(0))==pytest.approx(metrics['overall']['qa_ce'])


def test_cluster_metadata_mismatch_rejected():
    rr=toy();bad=[dict(r) for r in rr];bad[0]['patient_id']='wrong'
    vv={r['id']:[1.,1] for r in rr}
    with pytest.raises(AssertionError):paired_p(rr,bad,vv,vv,'green',10)
