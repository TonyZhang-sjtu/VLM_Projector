"""Semantic extraction regressions: no model weights or external services needed."""
import json
from types import SimpleNamespace

import pytest

from scripts.bindformer_v2_spec import specs
from scripts.build_bindformer_knowledge_v2 import extract, inputs, segments


def run_extract(tmp_path, text):
    audit = tmp_path / 'audit'
    audit.mkdir()
    (audit / 'ner_spans.jsonl').write_text('')
    output = tmp_path / 'output'
    output.mkdir()
    rows = [dict(id='synthetic-train-example', cohort='BraTS_GLI', sequence='T1C', finding=text)]
    return extract(SimpleNamespace(audit_input=audit, output=output), rows, specs())


@pytest.mark.parametrize('text,expected,forbidden', [
    ('A lesion under the left frontal bone.', 'frontal_bone', 'frontal_lobe'),
    ('A lesion in the right cerebellopontine angle.', 'cerebellopontine_angle', 'cerebellum'),
    ('A lesion in the precentral cerebellar cistern.', 'precentral_cerebellar_cistern', 'precentral_gyrus'),
    ('A lesion in the hippocampal region.', 'hippocampal_region', 'hippocampus'),
    ('The lesion causes a shift to the left.', 'leftward', 'left'),
])
def test_preserve_anatomical_and_direction_distinctions(tmp_path, text, expected, forbidden):
    mentions, _, _ = run_extract(tmp_path, text)
    keys = {m['key'] for m in mentions}
    assert expected in keys
    assert forbidden not in keys
    assert all(text[m['start']:m['end']] == m['text'] for m in mentions)


@pytest.mark.parametrize('text,expected,forbidden', [
    ('Lesions involve the frontal and parietal lobes.', {'frontal_lobe','parietal_lobe'}, {'frontal_bone'}),
    ('Lesions involve the temporal-occipital region.', {'temporal_region','occipital_region'}, {'temporal_lobe','occipital_lobe'}),
])
def test_split_coordinated_anatomy_using_head(tmp_path, text, expected, forbidden):
    mentions, _, _ = run_extract(tmp_path, text)
    keys = {m['key'] for m in mentions}
    assert expected <= keys
    assert not (forbidden & keys)
    assert any(m['extraction_method']=='head_supported_anatomical_coordination' for m in mentions)


@pytest.mark.parametrize('text', [
    'There is no mention of edema.',
    'There is no detail provided about edema.',
    'Edema is not explicitly described.',
])
def test_missing_description_is_not_a_negative_label(tmp_path, text):
    mentions, _, _ = run_extract(tmp_path, text)
    edema = next(m for m in mentions if m['key']=='edema')
    kinds = {c['kind'] for c in edema['clause']['assertion_cues']}
    assert 'not_mentioned' in kinds
    assert 'negative' not in kinds
    assert edema['usable_as_patient_supervision'] is False


def test_uncertain_interpretation_and_measurement_remain_context(tmp_path):
    text = 'A lesion measures 12*14*15 mm with possible bleeding or calcification.'
    mentions, context, _ = run_extract(tmp_path, text)
    assert not any(m['key']=='calcification' for m in mentions)
    calc = next(c for c in context if c.get('key')=='calcification')
    assert any(c['kind']=='uncertain' for c in calc['clause']['assertion_cues'])
    measurement = next(c for c in context if c['kind']=='measurement')
    assert measurement['values']==[12.0,14.0,15.0]
    assert measurement['unit']=='mm'
    assert measurement['vector_bank_inclusion'] is False


def test_ambiguous_objects_do_not_become_instance_labels(tmp_path):
    mentions, _, _ = run_extract(tmp_path, 'One lesion and another lesion show heterogeneous signal.')
    texture = next(m for m in mentions if m['key']=='heterogeneous')
    assert len(texture['object_anchor_candidates'])==2
    assert texture['attachment_status']=='ambiguous_multiple_text_anchors'
    assert texture['usable_as_patient_supervision'] is False


@pytest.mark.parametrize('text,key',[
    ('The lesion exhibits mixed high and low signals.', 'mixed_signal'),
    ('The boundaries are indistinct.', 'ill_defined'),
    ('No shift in midline structures is described.', 'midline_shift'),
    ('Narrowed bilateral lateral ventricles are observed.', 'ventricular_compression'),
])
def test_corpus_wording_variants(tmp_path,text,key):
    mentions,_,_=run_extract(tmp_path,text)
    assert any(m['key']==key for m in mentions)


def test_bare_genu_is_not_automatically_callosal(tmp_path):
    mentions, _, unresolved = run_extract(tmp_path, 'An abnormality involves the genu of the internal capsule.')
    assert not any(m['key']=='callosal_genu' for m in mentions)
    assert any(m['key']=='callosal_genu' for m in unresolved)


def test_sentence_boundaries_preserve_decimals():
    text='A lesion measures 12.5 mm. No edema.'
    assert [text[a:b] for a,b in segments(text)] == ['A lesion measures 12.5 mm.', 'No edema.']


def test_nontraining_inputs_rejected_before_audit_access(tmp_path):
    manifest=tmp_path/'train.jsonl'
    manifest.write_text(json.dumps({'split':'test'})+'\n')
    with pytest.raises(ValueError, match='Non-training'):
        inputs(SimpleNamespace(manifest=manifest))
    with pytest.raises(ValueError, match='Only train'):
        inputs(SimpleNamespace(manifest=tmp_path/'val.jsonl'))
