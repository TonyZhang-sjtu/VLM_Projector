"""Spatial protocol regressions; no clinical model, GPU, or network required."""
import pytest
from experiments.brain_location_audit_v1.protocol import (
    extract_graph, label_sets, normalize_anatomy, score_sets,
)


def facts(*items):
    return {'facts': [dict(side=s, site=t, status=a) for s, t, a in items]}


def graph(location='left frontal lobe', status='definitely present', finding='lesion'):
    return {'entities': {
        '1': dict(tokens=finding, label='Observation::'+status, start_ix=1,
                  relations=[['located_at', '2']]),
        '2': dict(tokens=location, label='Anatomy::definitely present', start_ix=4, relations=[]),
    }}


@pytest.mark.parametrize('text,sites,sides', [
    ('left frontal lobe', ['frontal_lobe'], ['left']),
    ('frontal lobe on the left', ['frontal_lobe'], ['left']),
    ('frontal lobe', ['frontal_lobe'], []),
    ('brain stem', ['brainstem'], []),
    ('left frontoparietal lobes', ['frontal_lobe', 'parietal_lobe'], ['left']),
    ('bilateral temporal regions', ['temporal_lobe'], ['bilateral']),
    ('left and right frontal lobes', ['frontal_lobe'], ['bilateral']),
    ('midline corpus callosum', ['corpus_callosum'], ['midline']),
    ('leftward deviation', [], []),
    ('right frontal horn of ventricle', ['ventricle'], ['right']),
    ('right temporal bone', [], ['right']),
    ('left insular cortex', ['insula'], ['left']),
    ('right hippocampal region', ['hippocampus'], ['right']),
    ('cerebellopontine angle', ['cerebellopontine_angle'], []),
])
def test_anatomy(text, sites, sides):
    assert normalize_anatomy(text) == dict(sites=sites, sides=sides)


def test_same_graph_has_identical_reference_and_prediction_parsing():
    ref = graph('frontal lobe on the left')
    assert score_sets(extract_graph(ref), extract_graph(ref))['joint']['exact'] == 1


@pytest.mark.parametrize('status', ['definitely absent', 'uncertain'])
def test_negated_or_uncertain_predictions_are_not_removed_from_denominator(status):
    score = score_sets(extract_graph(graph()), extract_graph(graph(status=status)))
    for field in score.values():
        assert field['eligible'] and field['f1'] == 0 and field['exact'] == 0


def test_missing_side_does_not_erase_correct_site():
    score = score_sets(extract_graph(graph()), extract_graph(graph('frontal lobe')))
    assert score['site']['exact'] == 1
    assert score['side']['exact'] == score['joint']['exact'] == 0


def test_unknown_reference_does_not_mean_negative_reference():
    result = score_sets(facts(), extract_graph(graph()))
    assert all(r == {'eligible': False} for r in result.values())


def test_side_swap_cannot_be_hidden_by_same_anatomy_set():
    ref = facts(('left','frontal_lobe','present'), ('right','temporal_lobe','present'))
    pred = facts(('right','frontal_lobe','present'), ('left','temporal_lobe','present'))
    score = score_sets(ref, pred)
    assert score['site']['exact'] == score['side']['exact'] == 1
    assert score['joint']['f1'] == 0


def test_bilateral_is_not_a_correct_unilateral_match():
    score = score_sets(extract_graph(graph()), extract_graph(graph('bilateral frontal lobes')))
    assert score['side']['f1'] == score['joint']['f1'] == 0
    assert score['site']['exact'] == 1


def test_bilateral_and_explicit_two_sides_same_site_are_equivalent():
    ref = facts(('bilateral','frontal_lobe','present'))
    pred = facts(('left','frontal_lobe','present'), ('right','frontal_lobe','present'))
    assert score_sets(ref, pred)['joint']['exact'] == 1


def test_duplicates_and_sentence_order_do_not_change_score():
    one = ('left','frontal_lobe','present'); two = ('right','temporal_lobe','present')
    assert score_sets(facts(one, two), facts(two, one, one))['joint']['exact'] == 1


def test_omissions_and_extra_sites_both_reduce_f1():
    one = ('left','frontal_lobe','present'); two = ('right','temporal_lobe','present')
    assert score_sets(facts(one,two), facts(one))['joint']['f1'] == pytest.approx(2/3)
    assert score_sets(facts(one), facts(one,two))['joint']['f1'] == pytest.approx(2/3)


def test_no_invented_midline_side_from_brainstem_anatomy():
    parsed = label_sets(extract_graph(graph('brainstem')))
    assert parsed['site'] == {'brainstem'} and parsed['side'] == set()
    assert parsed['eligible']['site'] and not parsed['eligible']['joint']


def test_modifiers_are_followed_but_unrelated_ventricles_are_not():
    g = graph('frontal')
    g['entities']['3'] = dict(tokens='left',label='Anatomy::definitely present',start_ix=3,
                              relations=[['modify','2']])
    g['entities']['4'] = dict(tokens='both lateral ventricles',label='Anatomy::definitely present',
                              start_ix=20,relations=[])
    assert label_sets(extract_graph(g))['joint'] == {('left','frontal_lobe')}


def test_midline_shift_is_separate_from_lesion_laterality():
    g = graph()
    g['entities']['3'] = dict(tokens='rightward deviation',label='Observation::definitely present',
                              start_ix=20,relations=[['located_at','4']])
    g['entities']['4'] = dict(tokens='midline',label='Anatomy::definitely present',start_ix=24,relations=[])
    out = extract_graph(g)
    assert label_sets(out)['side'] == {'left'}
    assert out['midline_shift'][0]['direction'] == 'right'


def test_mass_effect_is_not_a_mass():
    assert extract_graph(graph(finding='mass effect'))['facts'] == []


def test_ambiguous_merged_anatomy_does_not_invent_joint_pairs():
    out = extract_graph(graph('left frontal and right temporal lobes'))
    assert 'ambiguous_side_site_binding' in out['flags']
    assert not label_sets(out)['eligible']['joint']


def test_graph_cycles_terminate():
    g=graph('frontal')
    g['entities']['2']['relations']=[['modify','3']]
    g['entities']['3']=dict(tokens='left',label='Anatomy::definitely present',start_ix=3,
                            relations=[['modify','2']])
    assert label_sets(extract_graph(g))['site'] == {'frontal_lobe'}


def test_green_uses_category_identity_not_an_assumed_index():
    from experiments.brain_location_audit_v1.prepare import green_location_count
    d={'error_categories':['(c) Misidentification of a finding\'s anatomic location/position','severity'],
       'significant_errors':[2,7]}
    assert green_location_count(d)==2
    assert green_location_count(dict(d,significant_errors=[0,7]))==0
    assert green_location_count(dict(d,significant_errors=[2])) is None
    assert green_location_count({}) is None


def test_pilot_selection_independent_of_iteration_order():
    from experiments.brain_location_audit_v1.prepare import choose_pilot
    rows={f'{p}_{s}':dict(id=f'{p}_{s}',patient_id=str(p),sequence=s)
          for p in range(30) for s in ['T1W','T1C','T2W','FLAIR']}
    selected=choose_pilot(rows)
    assert selected==choose_pilot(dict(reversed(list(rows.items()))))
    assert len(selected)==len({rows[k]['patient_id'] for k in selected})==24


def test_stress_pack_has_unique_ids_and_targeted_spatial_cases():
    from experiments.brain_location_audit_v1.prepare import stress_cases
    cases=stress_cases()
    assert len(cases)==len({c['id'] for c in cases})==28
    assert all(c['reference'] and c['prediction'] and c['expected_property'] for c in cases)
    assert any('multisite_binding' in c['id'] for c in cases)
