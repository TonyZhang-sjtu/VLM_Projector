"""Semantic regression tests; no model loading or GPU use."""
import copy
import pytest
from .protocol import anatomy_pairs, extract_graph, label_sets, score_sets


def facts(*xs):
    return {'facts':[dict(side=s,site=t,status=a) for s,t,a in xs]}


def graph(text, finding='lesion', location=None, status='definitely present'):
    words=text.split()
    def node(phrase,label,relations):
        p=phrase.split();i=next(i for i in range(len(words)) if words[i:i+len(p)]==p)
        return dict(tokens=phrase,label=label,start_ix=i,end_ix=i+len(p)-1,relations=relations)
    entities={'1':node(finding,'Observation::'+status,[['located_at','2']] if location else [])}
    if location:
        entities['2']=node(location,'Anatomy::definitely present',[])
    return dict(text=text,entities=entities)


@pytest.mark.parametrize('text,expected',[
 ('left frontal and parietal lobes',{('left','frontal_lobe'),('left','parietal_lobe')}),
 ('left frontoparietal junction',{('left','frontal_lobe'),('left','parietal_lobe')}),
 ('left frontal and right temporal lobes',{('left','frontal_lobe'),('right','temporal_lobe')}),
 ('left temporal lobe , insular region , and basal ganglia area',
  {('left','temporal_lobe'),(None,'insula'),(None,'basal_ganglia')}),
 ('left frontal lobe and parietal lobe',{('left','frontal_lobe'),(None,'parietal_lobe')}),
 ('left and right frontal lobes',{('bilateral','frontal_lobe')}),
 ('both frontal and parietal lobes',{('bilateral','frontal_lobe'),('bilateral','parietal_lobe')}),
 ('frontal lobe on the left',{('left','frontal_lobe')}),
 ('base of the right middle cranial fossa',{('right','middle_cranial_fossa')}),
 ('left temporal pole',{('left','temporal_lobe')}),
 ('brain stem',{(None,'brainstem')}),
 ('midline corpus callosum',{('midline','corpus_callosum')}),
 ('right frontal horn of the ventricle',{('right','ventricle')}),
 ('right temporal bone',{('right',None)}),
])
def test_explicit_anatomy_scope(text,expected):
    assert set(anatomy_pairs(text))==expected


@pytest.mark.parametrize('finding',['abnormal signal','signal focus','abnormality','lesion','mass','nodule'])
def test_finding_aliases_and_junction(finding):
    text=f'A {finding} is present at the junction of the left frontal and parietal lobes .'
    g=graph(text,finding,location='junction')
    assert label_sets(extract_graph(g))['joint']=={('left','frontal_lobe'),('left','parietal_lobe')}


def test_no_graph_relation_explicit_attributive_recovery():
    g=graph('A left frontal lesion displaces the right lateral ventricle .')
    assert label_sets(extract_graph(g))['joint']=={('left','frontal_lobe')}


def test_missing_edge_fronted_anatomy():
    g=graph('At the left frontal and parietal junction , there is an abnormal signal focus .','abnormal signal focus')
    assert label_sets(extract_graph(g))['joint']=={('left','frontal_lobe'),('left','parietal_lobe')}


@pytest.mark.parametrize('tail',[
 'There is rightward midline shift .',
 'There is compression of both lateral ventricles .',
 'Edema is present in the right temporal lobe .',
])
def test_do_not_borrow_other_finding_or_other_sentence(tail):
    g=graph('A lesion is in the left frontal lobe . '+tail,location='frontal')
    assert label_sets(extract_graph(g))['joint']=={('left','frontal_lobe')}


@pytest.mark.parametrize('finding',['edema','mass effect','peritumoral edema'])
def test_nonlesion_findings_not_promoted(finding):
    g=graph(f'There is {finding} in the left frontal lobe .',finding,location='frontal')
    assert extract_graph(g)['facts']==[]


@pytest.mark.parametrize('status',['definitely absent','uncertain'])
def test_assertion_and_missing_predictions_stay_in_denominator(status):
    t='A lesion is in the left frontal lobe .'
    ref=extract_graph(graph(t,location='frontal'))
    pred=extract_graph(graph(t,location='frontal',status=status))
    assert all(r['eligible'] and r['fn']==1 and r['tp']==0 for r in score_sets(ref,pred).values())


def test_no_cross_sentence_location_guess():
    g=graph('The lesion enhances . The right frontal lobe is normal .')
    assert extract_graph(g)['facts']==[]


def test_unknown_side_keeps_region_but_not_joint():
    r=facts(('left','temporal_lobe','present'),(None,'basal_ganglia','present'))
    labels=label_sets(r)
    assert labels['eligible']==dict(side=True,site=True,joint=True)
    assert labels['joint']=={('left','temporal_lobe')}
    assert labels['site']=={'temporal_lobe','basal_ganglia'}


def test_unknown_reference_side_cannot_create_false_positive():
    r=facts(('left','temporal_lobe','present'),(None,'basal_ganglia','present'))
    p=facts(('left','temporal_lobe','present'),('right','basal_ganglia','present'))
    score=score_sets(r,p)
    assert all(v['exact']==1 for v in score.values())
    assert score['joint']['ignored_prediction_locations']==[('right','basal_ganglia')]


def test_known_wrong_side_not_excused_by_unknown_other_site():
    r=facts(('left','temporal_lobe','present'),(None,'basal_ganglia','present'))
    p=facts(('right','temporal_lobe','present'),('left','basal_ganglia','present'))
    score=score_sets(r,p)
    assert score['side']['tp']==score['joint']['tp']==0
    assert score['side']['fp']==score['joint']['fp']==1


def test_predicted_missing_side_still_false_negative():
    r=facts(('left','frontal_lobe','present'))
    p=facts((None,'frontal_lobe','present'))
    s=score_sets(r,p)
    assert s['site']['exact']==1
    assert s['joint']['fn']==s['side']['fn']==1


def test_extra_known_location_penalized():
    r=facts(('left','frontal_lobe','present'))
    p=facts(('left','frontal_lobe','present'),('right','temporal_lobe','present'))
    assert score_sets(r,p)['joint']['f1']==pytest.approx(2/3)


def test_side_swap_preserves_separate_sets_but_fails_binding():
    r=facts(('left','frontal_lobe','present'),('right','temporal_lobe','present'))
    p=facts(('right','frontal_lobe','present'),('left','temporal_lobe','present'))
    s=score_sets(r,p)
    assert s['side']['exact']==s['site']['exact']==1 and s['joint']['tp']==0


def test_adjacent_is_retained_as_distinct_relation():
    p=extract_graph(graph('A lesion is adjacent to the left lateral ventricle .',location='ventricle'))
    assert label_sets(p)['joint']=={('left','adjacent_to/ventricle')}
    r=facts(('left','ventricle','present'))
    assert score_sets(r,p)['joint']['tp']==0


def test_input_graph_not_mutated():
    g=graph('A lesion at the left frontal and parietal junction .',location='junction');before=copy.deepcopy(g)
    assert extract_graph(g)==extract_graph(g)
    assert g==before


def test_fully_unspecified_reference_is_not_wildcard():
    r=facts(('left','frontal_lobe','present'),(None,None,'present'))
    p=facts(('right','temporal_lobe','present'))
    assert score_sets(r,p)['joint']['fp']==1


@pytest.mark.parametrize('text,finding,target,expected',[
 ('An abnormality in the right middle cranial fossa .','abnormality','fossa',('right','middle_cranial_fossa')),
 ('A lesion beneath the skull plate in the right frontal region .','lesion','plate',('right','frontal_lobe')),
 ('The left frontal lobe contains the largest lesion .','lesion',None,('left','frontal_lobe')),
 ('A nodule with low signal intensity located in the left parietal lobe .','nodule',None,('left','parietal_lobe')),
 ('The nodules , particularly the largest one in the right parietal lobe .','nodules',None,('right','parietal_lobe')),
 ('A lesion on the left side of the falx cerebri .','lesion','falx',('left','falx')),
 ('A nodule adjacent to the cerebellar tentorium on the right side .','nodule','tentorium',('right','adjacent_to/tentorium')),
 ('A lesion adjacent to the falx cerebri in the right frontal lobe .','lesion','falx',('right','frontal_lobe')),
])
def test_extended_explicit_locations(text,finding,target,expected):
    assert expected in label_sets(extract_graph(graph(text,finding,target)))['joint']


def test_tentorium_not_silently_cerebellum():
    assert anatomy_pairs('right cerebellar tentorium')==[('right','tentorium')]


def test_signal_bridge_cannot_borrow_edema_location():
    g=graph('A lesion causes edema located in the right temporal lobe .')
    assert extract_graph(g)['facts']==[]


def test_bilateral_same_site_equivalence_and_unilateral_contradiction():
    r=facts(('bilateral','frontal_lobe','present'))
    p=facts(('left','frontal_lobe','present'),('right','frontal_lobe','present'))
    assert score_sets(r,p)['joint']['exact']==1
    assert score_sets(r,facts(('left','frontal_lobe','present')))['joint']['tp']==0


def test_duplication_and_order_invariance():
    a=('left','frontal_lobe','present');b=('right','temporal_lobe','present')
    assert score_sets(facts(a,b),facts(b,a,a))['joint']['exact']==1


@pytest.mark.parametrize('bridge',['is not located','is never situated','may be located','might be situated','is possibly located'])
def test_location_assertion_can_differ_from_lesion_presence(bridge):
    # Deliberately present graph finding: position itself is negated/uncertain.
    g=graph(f'A lesion {bridge} in the left frontal lobe .',location='frontal')
    assert not label_sets(extract_graph(g))['eligible']['joint']
    ref=facts(('left','frontal_lobe','present'))
    assert score_sets(ref,extract_graph(g))['joint']['fn']==1


def test_uncertain_diagnosis_does_not_erase_definite_location():
    g=graph('A lesion in the left frontal lobe may represent a glioma .',location='frontal')
    assert label_sets(extract_graph(g))['joint']=={('left','frontal_lobe')}


def test_do_not_borrow_cyst_position_across_predicates():
    g=graph('A lesion is visible and a cyst is located in the right temporal lobe .')
    assert not label_sets(extract_graph(g))['eligible']['joint']
