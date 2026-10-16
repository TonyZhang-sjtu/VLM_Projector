"""Validate v2 provenance, exact source spans, bank alignment and vector values."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from safetensors.numpy import load_file

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def validate(output):
    output = Path(output)
    build = json.loads((output/'build_manifest.json').read_text())
    encoded = json.loads((output/'embedding_manifest.json').read_text())
    bank = read(output/'concepts.jsonl')
    candidates = {c['key']:c for c in read(output/'candidates.jsonl')}
    source = Path(build['source_manifest'])
    require(source.name=='train.jsonl' and sha(source)==build['source_manifest_sha256'], 'Wrong corpus fingerprint')
    rows = {r['id']:r for r in read(source)}
    require(len(rows)==build['reports'] and all(r['split']=='train' for r in rows.values()), 'Non-train or missing source row')
    require(build['script_sha256']==sha(ROOT/'scripts/build_bindformer_knowledge_v2.py'), 'Stale builder')
    require(build['spec_sha256']==sha(ROOT/'scripts/bindformer_v2_spec.py'), 'Stale vocabulary specification')
    require(build['decisions_sha256']==sha(ROOT/'Pipeline_0929/bindformer_v2_cui_decisions.json'), 'Stale CUI decisions')
    require(encoded['build_manifest_sha256']==sha(output/'build_manifest.json'), 'Vectors from another build')
    for manifest in [build, encoded]:
        for filename, expected in manifest['files'].items():
            require(sha(output/filename)==expected, f'File digest mismatch: {filename}')
    reports = read(output/'reports_context.jsonl')
    cues = 0
    for report in reports:
        text = rows[report['report_id']]['finding']
        require(report['finding']==text, 'Report text changed')
        require(report['patient_instance_labels_created'] is False, 'Unexpected patient labels')
        for clause in report['clauses']:
            require(text[clause['start']:clause['end']]==clause['text'], 'Invalid clause span')
            for part in clause['object_anchors'] + clause['assertion_cues']:
                require(text[part['start']:part['end']]==part['text'], 'Invalid cue/anchor offset')
            cues += len(clause['assertion_cues'])
    span_counts = {}
    for filename in ['mentions.jsonl','context_fields.jsonl','ambiguous_mentions.jsonl','suppressed_components.jsonl']:
        records = read(output/filename)
        for item in records:
            require(item['report_id'] in rows, 'Unknown report in extracted spans')
            text = rows[item['report_id']]['finding']
            require(text[item['start']:item['end']]==item['text'], f'Invalid raw source span: {filename}')
            if filename=='mentions.jsonl':
                require(item['usable_as_patient_supervision'] is False, 'Heuristic mention promoted to label')
                require(item['key'] in candidates, 'Mention lacks a concept specification')
                anchors = {a['anchor_id'] for a in item['clause']['object_anchors']}
                require(set(item['object_anchor_candidates'])<=anchors, 'Dangling object reference')
        span_counts[filename]=len(records)
    index = json.loads((output/'concept_index.json').read_text())
    require(index['concept_ids']==[c['concept_id'] for c in bank], 'Concept ID order mismatch')
    require([c['embedding_index'] for c in bank]==list(range(len(bank))), 'Embedding indices not contiguous')
    require(len(set(index['concept_ids']))==len(bank), 'Duplicate concept identity')
    inputs = read(output/'encoding_inputs.jsonl')
    for i, (c, t) in enumerate(zip(bank,inputs)):
        require(t['index']==i and t['concept_id']==c['concept_id'], 'Encoding input ID mismatch')
        require(t['text']==c['name']+'. '+c['short_definition']==c['vector_input'], 'Wrong encoding text template')
        require(t['text_sha256']==hashlib.sha256(t['text'].encode()).hexdigest(), 'Text digest mismatch')
        require(0<t['tokens']<=encoded['max_tokens'], 'Truncated/invalid text length')
        if c['field']!='control':
            require(c['train_evidence']['compatible_reports']>0, 'Unobserved term in active bank')
    require(len(inputs)==len(bank), 'Encoding input count mismatch')
    tensors = load_file(str(output/'concept_bank.safetensors'))
    for name in ['embeddings','role_mask','sequence_mask','scope_mask']:
        array = np.load(output/(name+'.npy'))
        require(np.array_equal(array,tensors[name]), f'NPY/safetensors disagreement: {name}')
    e = tensors['embeddings']; r=tensors['role_mask']; s=tensors['sequence_mask']; v=tensors['scope_mask']
    require(e.shape==(len(bank),768) and e.dtype==np.float32 and np.isfinite(e).all(), 'Invalid embedding matrix')
    norms = np.linalg.norm(e,axis=1)
    require(np.allclose(norms,1,atol=1e-6,rtol=0), 'Vectors are not L2-normalized')
    for name,array,labels,field in [('role',r,index['roles'],'role_memberships'),('sequence',s,index['sequences'],'sequence_scope')]:
        expected=np.array([[label in c[field] for c in bank] for label in labels],dtype=bool)
        require(array.dtype==np.bool_ and np.array_equal(array,expected), f'Incorrect {name} mask')
    expected_scope=np.array([[c['field']=='control' or any(k.startswith('BraTS_') for k in c['train_evidence']['by_cohort']) for c in bank],[True]*len(bank)],dtype=bool)
    require(index['vocabulary_scopes']==['brats_train','all_train'] and np.array_equal(v,expected_scope), 'Wrong vocabulary scope mask')
    umls={c['cui']:c for c in read(output/'umls_evidence.jsonl')}
    definition_count=0
    for c in bank:
        if c['selected_cui']:
            require(c['selected_cui'] in umls, 'Missing CUI evidence')
            d=c['source']['definition_source']
            if d:
                require(d in umls[c['selected_cui']]['definitions'], 'Selected definition not in source evidence')
                definition_count+=1
        else:
            require(c['identity']=='local_identity', 'Null CUI with wrong identity type')
    source_audit=ROOT/'assets/knowledge/bindformer_brain_corpus_audit_v1'
    residuals={}
    for original,current in [('ner_phrase_candidates.jsonl','ner_reconciliation.jsonl'),
                             ('umls_phrase_candidates.jsonl','umls_reconciliation.jsonl'),
                             ('word_inventory.jsonl','word_reconciliation.jsonl')]:
        a={x['normalized'] for x in read(source_audit/original)}
        b={x['normalized'] for x in read(output/current)}
        require(a==b, f'Lost residual input strings: {current}')
        residuals[current]=len(b)
    result=dict(status='passed',validator_sha256=sha(__file__),build_manifest_sha256=sha(output/'build_manifest.json'),
        embedding_manifest_sha256=sha(output/'embedding_manifest.json'),source_reports=len(rows),
        exact_span_counts=span_counts,assertion_cues_checked=cues,concepts=len(bank),
        selected_original_definitions_checked=definition_count,vector_shape=list(e.shape),
        norm_range=[float(norms.min()),float(norms.max())],scope_counts=dict(zip(index['vocabulary_scopes'],v.sum(-1).tolist())),
        residual_strings_preserved=residuals,patient_label_assertions_created=False,
        limits='Validation checks provenance, extraction contracts and numerical consistency; it is not clinical accuracy certification.')
    (output/'validation.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(result,ensure_ascii=False),flush=True)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'assets/knowledge/bindformer_brain_candidates_v2')
    validate(parser.parse_args().output)
