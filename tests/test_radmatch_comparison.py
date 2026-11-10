"""Comparison orchestration tests; do not pretend mocked calls are real scoring."""
import json
from pathlib import Path

import pytest

from experiments.brain_location_audit_v1.protocol_v2 import extract_graph
from experiments.brain_location_audit_v1.radmatch_compare import read_case, verify_inputs, digest


def node(text, relations=(), observation=False):
    return dict(tokens=text, label=('Observation' if observation else 'Anatomy')+'::definitely present',
                relations=list(relations))


def redundant_graph():
    return dict(entities={
        '1': node('lesion', [('located_at','2'),('located_at','3')], True),
        '2': node('frontal'),
        '3': node('lobe', [('modify','2')]),
        '4': node('left', [('modify','2'),('modify','3')]),
    })


def test_contained_partial_relation_is_not_an_extra_location():
    r=extract_graph(redundant_graph())
    assert len(r['facts'])==1
    assert r['facts'][0]['side']=='left' and r['facts'][0]['site']=='frontal_lobe'
    assert len(r['contained_redundant_facts'])==1
    assert 'positive_location_without_site' not in r['flags']


def test_other_finding_partial_location_is_preserved():
    g=redundant_graph()
    g['entities']['1']['relations']=[['located_at','2']]
    g['entities']['5']=node('mass',[('located_at','4')],True)
    r=extract_graph(g)
    assert len(r['facts'])==2 and not r['contained_redundant_facts']


def test_different_mention_partial_location_is_preserved():
    g=redundant_graph()
    g['entities']['1']['relations']=[['located_at','2'],['located_at','5']]
    g['entities']['5']=node('left')
    assert len(extract_graph(g)['facts'])==2


def test_independent_brain_regions_are_not_deduplicated():
    g=redundant_graph()
    g['entities']['1']['relations']=[['located_at','2'],['located_at','5']]
    g['entities']['5']=node('left temporal lobe')
    r=extract_graph(g)
    assert {f['site'] for f in r['facts']}=={'frontal_lobe','temporal_lobe'}


def test_missing_output_is_not_zero_error(tmp_path):
    result,_=read_case(tmp_path,'case_000')
    assert result['status']=='missing_or_invalid'
    assert 'actionable_errors' not in result and 'location_major' not in result


def write_result(tmp_path, fallback=False):
    values={
        'findings_gt':[], 'findings_pred':[],
        'matching':{'validation_fallback':fallback},
        'attribute_errors':{'muc_records':[]},
        'per_report_metrics':dict(actionable_errors_total=0,
             muc_counts={'MIS':0,'SPU':0},attribute_breakdown={'location':{'major':0,'minor':0,'evaluated':0}}),
    }
    for name,value in values.items():
        (tmp_path/name).mkdir()
        (tmp_path/name/'case_000.json').write_text(json.dumps(value))


def test_degraded_upstream_matching_stays_visible(tmp_path):
    write_result(tmp_path,True)
    r,_=read_case(tmp_path,'case_000')
    assert r['status']=='degraded' and r['location_matched_pairs']==0


def test_partial_stage_failure_is_not_scored(tmp_path):
    write_result(tmp_path)
    (tmp_path/'attribute_errors/case_000.json').write_text('broken json')
    r,_=read_case(tmp_path,'case_000')
    assert r['status']=='missing_or_invalid'


def test_frozen_input_mutation_is_rejected(tmp_path):
    (tmp_path/'report.txt').write_text('left frontal lesion')
    (tmp_path/'manifest.json').write_text(json.dumps({'file_sha256':{'report.txt':digest('left frontal lesion')}}))
    verify_inputs(tmp_path)
    (tmp_path/'report.txt').write_text('right frontal lesion')
    with pytest.raises(ValueError,match='Input changed'):
        verify_inputs(tmp_path)


def test_collect_follows_official_radmatch_results_directory(tmp_path):
    from argparse import Namespace
    from experiments.brain_location_audit_v1.radmatch_compare import collect
    bundle=tmp_path/'inputs';bundle.mkdir()
    (bundle/'comparison_inputs.json').write_text('[]')
    (bundle/'manifest.json').write_text(json.dumps(dict(file_sha256={},groups=[
        dict(name='stress',kind='synthetic',expected_ids=['case_000'])])))
    output=tmp_path/'output'
    official=output/'stress'/'radmatch_results';official.mkdir(parents=True)
    write_result(official)
    collect(Namespace(bundle=bundle,output=output))
    coverage=json.loads((output/'coverage.json').read_text())
    assert coverage['status_counts']=={'scored':1}
