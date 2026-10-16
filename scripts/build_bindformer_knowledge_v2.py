"""Build the corpus-derived BindFormer v2 bank and offline SapBERT CLS cache.

Stages preserve uncertain text attachments and never export patient fact labels.
All source reports must be training rows from the audited manifest.
"""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data.sequence_mapping import sequence_of
from scripts.bindformer_v2_spec import ROLES, SEQUENCES, literal_pattern, specs


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def norm(text):
    return ' '.join(re.sub(r'[-‐‑–—]', ' ', text.casefold()).split())


def read_lines(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def write_lines(path, rows):
    with Path(path).open('w') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')


def write_json(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')


def write_tsv(path, rows, fields):
    with Path(path).open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields, delimiter='\t', extrasaction='ignore')
        w.writeheader()
        for row in rows:
            w.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v
                        for k, v in row.items() if k in fields})


def inputs(args):
    if args.manifest.name != 'train.jsonl':
        raise ValueError('Only train.jsonl accepted')
    rows = read_lines(args.manifest)
    if not rows or any(r['split'] != 'train' for r in rows):
        raise ValueError('Non-training or empty report input')
    if len({r['id'] for r in rows}) != len(rows):
        raise ValueError('Duplicate report IDs')
    audit = json.loads((args.audit_input / 'audit.json').read_text())
    if audit['train_sha256'] != sha(args.manifest):
        raise ValueError('Source report audit is stale')
    for row in rows:
        row['sequence'] = sequence_of(row['cohort'], row['id'])
    args.output.mkdir(parents=True, exist_ok=True)
    vocabulary = specs()
    if len({x['key'] for x in vocabulary}) != len(vocabulary):
        raise ValueError('Duplicate concept keys')
    return rows, vocabulary


def link(args, vocabulary):
    terms = defaultdict(set)
    for item in vocabulary:
        for term in item['search_terms']:
            terms[norm(term)].add(item['key'])
    matches = defaultdict(lambda: defaultdict(list))
    print('Searching UMLS using corpus-normalized concept names and aliases...', flush=True)
    with (args.meta / 'MRCONSO.RRF').open() as f:
        for line in f:
            row = line.rstrip('\n').split('|')
            if row[1] != 'ENG' or row[16] != 'N':
                continue
            for key in terms.get(norm(row[14]), ()):
                matches[key][row[0]].append(dict(text=row[14], aui=row[7], source=row[11], tty=row[12]))
    # Also inspect preselected identities even when the task-specific name is not an exact synonym.
    cuis = {cui for m in matches.values() for cui in m} | {s['preferred_cui'] for s in vocabulary if s['preferred_cui']}
    details = {c: dict(cui=c, synonyms=set(), active_english_auis=set(), sources=set(), semantic_types=[], definitions=[])
               for c in cuis}
    print(f'Collecting source definitions and semantic types for {len(cuis)} candidate CUIs...', flush=True)
    with (args.meta / 'MRCONSO.RRF').open() as f:
        for line in f:
            cui = line.split('|', 1)[0]
            if cui not in details:
                continue
            row = line.rstrip('\n').split('|')
            if row[1] == 'ENG' and row[16] == 'N':
                d = details[cui]
                d['synonyms'].add(row[14]); d['active_english_auis'].add(row[7]); d['sources'].add(row[11])
    with (args.meta / 'MRSTY.RRF').open() as f:
        for line in f:
            row = line.rstrip('\n').split('|')
            if row[0] in details:
                details[row[0]]['semantic_types'].append(dict(tui=row[1], name=row[3]))
    with (args.meta / 'MRDEF.RRF').open() as f:
        for line in f:
            row = line.rstrip('\n').split('|')
            if row[0] in details and row[6] == 'N' and row[1] in details[row[0]]['active_english_auis']:
                details[row[0]]['definitions'].append(dict(aui=row[1], source=row[4], text=row[5]))
    sources = {s for d in details.values() for s in d['sources']}
    source_versions = defaultdict(list)
    with (args.meta / 'MRSAB.RRF').open() as f:
        for line in f:
            row = line.rstrip('\n').split('|')
            if row[3] in sources:
                source_versions[row[3]].append(dict(versioned_source=row[2], name=row[4], version=row[6]))
    for d in details.values():
        d['synonyms'] = sorted(d['synonyms']); d['sources'] = sorted(d['sources'])
        del d['active_english_auis']
    candidates = [dict(key=s['key'], name=s['name'], role=s['role'], field=s['field'],
                       search_terms=s['search_terms'], preferred_cui=s['preferred_cui'],
                       matches=[dict(cui=c, atoms=a) for c, a in sorted(matches[s['key']].items())]) for s in vocabulary]
    write_lines(args.output / 'umls_candidates.jsonl', candidates)
    write_lines(args.output / 'umls_evidence.jsonl', [details[c] for c in sorted(details)])
    write_json(args.output / 'source_versions.json', source_versions)
    write_json(args.output / 'link_audit.json', dict(umls_version='2026AA', script_sha256=sha(__file__),
        spec_sha256=sha(ROOT / 'scripts/bindformer_v2_spec.py'),
        raw_files={p.name: dict(size=p.stat().st_size, mtime_ns=p.stat().st_mtime_ns)
                   for p in [args.meta / n for n in ['MRCONSO.RRF', 'MRSTY.RRF', 'MRDEF.RRF', 'MRSAB.RRF']]},
        concepts=len(vocabulary), candidate_cuis=len(cuis)))
    print(f'Linked candidate search: {len(vocabulary)} specifications / {len(cuis)} CUIs', flush=True)


CUES = {
    'not_mentioned': r'\b(?:no mention(?: of)?|not (?:explicitly )?(?:mentioned|described|reported|provided|specified)|no (?:additional )?(?:details?|information|description|comments?)(?: (?:are |is )?provided)?|does not (?:explicitly )?(?:mention|report|describe)|do not (?:explicitly )?(?:mention|report|describe))\b',
    'negative': r'\b(?:no|not|without|neither|nor|absence of|absent|lack of)\b',
    'uncertain': r'\b(?:possible|possibly|potential|potentially|may|might|could|likely|suggests?|suggesting|probable|perhaps)\b',
    'normality': r'\b(?:normal|unremarkable|intact|preserved|maintained|unaffected|unshifted|unchanged|unaltered|undisturbed)\b',
    'interpretation': r'\b(?:indicating|indicative of|suggests?|suggesting|consistent with|characteristic of|associated with|often seen|commonly associated)\b',
}
INTERPRETATIONS = {
    'gliosis': r'\b(?:gliosis|gliotic)\b', 'calcification': r'\bcalcification\b',
    'vascular_mechanism': r'\b(?:vascularity|neovascularization|vascularization|neovascularity|angiogenesis|vascularized)\b',
    'blood_brain_barrier': r'\b(?:blood[- ]brain barrier|bbb)\b',
    'diagnostic_interpretation': r'\b(?:gliomas?|meningiomas?|metastatic|infarction|demyelination|inflammation|malignan\w*|cytotoxic|vasogenic|infiltration)\b',
}


def segments(text):
    boundaries = [0] + [m.end() for m in re.finditer(r'[.!?](?=\s|$)|;', text)]
    if boundaries[-1] != len(text):
        boundaries.append(len(text))
    result = []
    for a, b in zip(boundaries, boundaries[1:]):
        cuts = [a] + [a + m.end() for m in re.finditer(r'\b(?:but|however|whereas)\b', text[a:b], re.I)] + [b]
        for start, end in zip(cuts, cuts[1:]):
            while start < end and text[start].isspace():
                start += 1
            if end > start:
                result.append((start, end))
    return result


def extract(args, rows, vocabulary):
    patterns = {s['key']: [re.compile(literal_pattern(a), re.I) for a in s['aliases']] +
                          [re.compile(p, re.I) for p in s.get('extra_patterns', [])] for s in vocabulary}
    by_key = {s['key']: s for s in vocabulary}
    source_spans = defaultdict(list)
    for span in read_lines(args.audit_input / 'ner_spans.jsonl'):
        source_spans[span['report_id']].append(span)
    mentions, suppressed, contexts, unresolved = [], [], [], []
    report_records = []
    # Head-bearing coordination only. The adjectives do not decide CUI identities themselves.
    lobe_map = {'frontal':'frontal_lobe', 'fronto':'frontal_lobe', 'parietal':'parietal_lobe',
                'parieto':'parietal_lobe', 'temporal':'temporal_lobe', 'temporo':'temporal_lobe',
                'occipital':'occipital_lobe', 'occipito':'occipital_lobe', 'insular':'insula', 'insula':'insula'}
    atom = '(?:' + '|'.join(sorted(lobe_map, key=len, reverse=True)) + ')'
    coord_pattern = re.compile(r'\b' + atom + r'(?:(?:[-, /]+|\s+(?:and\s+)?)' + atom + r')+\s+(?:lobes?|regions?)\b', re.I)
    anchor_pattern = re.compile(r'\b(?:lesions?|foci|focus|nodules?|masses|mass)\b', re.I)
    for row in rows:
        text = row['finding']; rid = row['id']; clauses = segments(text)
        clause_records = []
        for ci, (start, end) in enumerate(clauses):
            anchors = []
            for m in anchor_pattern.finditer(text, start, end):
                if m[0].lower() == 'mass' and re.match(r'[- ](?:like|effect)\b', text[m.end():], re.I):
                    continue
                anchors.append(dict(anchor_id=f'{rid}:object:{m.start()}', start=m.start(), end=m.end(), text=m[0],
                                    kind='report_text_group_not_verified_instance'))
            cues = []
            for kind, expression in CUES.items():
                for m in re.finditer(expression, text[start:end], re.I):
                    cues.append(dict(kind=kind, start=start + m.start(), end=start + m.end(), text=m[0]))
            # "no mention" is a missingness cue, not a separate negative cue.
            missing = [c for c in cues if c['kind'] == 'not_mentioned']
            cues = [c for c in cues if not (c['kind'] == 'negative' and
                     any(x['start'] <= c['start'] and c['end'] <= x['end'] for x in missing))]
            clause_records.append(dict(clause_id=f'{rid}:clause:{ci}', start=start, end=end, text=text[start:end],
                object_anchors=anchors, assertion_cues=sorted(cues, key=lambda x: (x['start'], x['kind'])),
                attachment_policy='textual_candidates_only_no_instance_or_assertion_ground_truth'))
        found = []
        def append(key, start, end, method, support=None):
            if not any(x['key'] == key and x['start'] == start and x['end'] == end for x in found):
                found.append(dict(key=key, start=start, end=end, text=text[start:end], extraction_method=method,
                                  supporting_phrase=support))
        for key, ps in patterns.items():
            for pattern in ps:
                for m in pattern.finditer(text):
                    append(key, m.start(), m.end(), 'explicit_alias_or_bounded_relation')
        for m in coord_pattern.finditer(text):
            for part in re.finditer(atom, m[0], re.I):
                key = lobe_map[part[0].lower()]
                if re.search(r'regions?$', m[0], re.I) and key.endswith('_lobe'):
                    key = key.replace('_lobe', '_region')
                append(key, m.start() + part.start(), m.start() + part.end(),
                       'head_supported_anatomical_coordination',
                       dict(start=m.start(), end=m.end(), text=m[0], head='lobe_or_region'))
        # Short anatomical forms whose CUI cannot be inferred from the word alone.
        context_requirements = {
            'callosal_genu': ('genu', r'corpus callosum|callosal'),
            'midbrain_tegmentum': ('tegmentum', r'midbrain|mesencephalon'),
            'superior_sagittal_sinus': ('sagittal sinus', r'superior sagittal sinus'),
        }
        retained = []
        for item in found:
            ci = next(i for i, (a, b) in enumerate(clauses) if a <= item['start'] < b)
            clause = clause_records[ci]
            key = item['key']; skip = None
            if key == 'mass' and re.match(r'[- ](?:like|effect)\b', text[item['end']:], re.I):
                skip = 'mass_word_inside_shape_or_effect_is_not_separate_mass_evidence'
            if key == 'enhancement' and re.search(r'(?:contrast|gadolinium)[- ]$', text[max(0,item['start']-20):item['start']], re.I):
                if re.match(r'\s+(?:images?|imaging|findings|sequences?|T1)\b', text[item['end']:], re.I):
                    skip = 'acquisition_description_not_lesion_enhancement'
            if key in context_requirements:
                short, required = context_requirements[key]
                if norm(item['text']) == short and not re.search(required, clause['text'], re.I):
                    skip = 'ambiguous_anatomical_short_form_needs_review'
            if skip:
                unresolved.append(dict(report_id=rid, cohort=row['cohort'], sequence=row['sequence'], **item,
                                       reason=skip, clause=clause))
                continue
            item['clause_index'] = ci
            retained.append(item)
        # Retain the longest named anatomy; record suppressed generic components for audit.
        chosen = []
        for item in sorted(retained, key=lambda x: (-(x['end'] - x['start']), x['start'], x['key'])):
            spec = by_key[item['key']]
            containers = [x for x in chosen if x['start'] <= item['start'] and item['end'] <= x['end'] and
                ((by_key[x['key']]['role'], by_key[x['key']]['field']) == (spec['role'], spec['field']) or
                 (spec['field'] == 'laterality' and by_key[x['key']]['field'] == 'displacement_direction'))]
            if containers:
                suppressed.append(dict(report_id=rid, **item, contained_in_key=containers[0]['key'],
                                       reason='more_specific_same_field_phrase'))
            else:
                chosen.append(item)
        local_mentions = []
        for item in sorted(chosen, key=lambda x: (x['start'], x['end'], x['key'])):
            spec = by_key[item['key']]; clause = clause_records[item.pop('clause_index')]
            anchors = clause['object_anchors']
            related_ner = [dict(start=s['start'], end=s['end'], text=s['text'], predicted_label=s['ner_label'])
                           for s in source_spans[rid] if s['start'] < item['end'] and item['start'] < s['end']]
            mention = dict(mention_id=f'{rid}:{item["start"]}:{item["end"]}:{item["key"]}', report_id=rid,
                cohort=row['cohort'], sequence=row['sequence'], concept_id=spec['concept_id'],
                role=spec['role'], field=spec['field'], **item, clause=clause,
                object_anchor_candidates=[a['anchor_id'] for a in anchors],
                attachment_status=('no_explicit_local_object' if not anchors else
                                   'single_local_text_anchor_unverified' if len(anchors) == 1 else 'ambiguous_multiple_text_anchors'),
                assertion_status='context_preserved_not_resolved_as_label',
                sequence_compatible=row['sequence'] in spec['sequence_scope'],
                overlapping_ner_candidates=related_ner, usable_as_patient_supervision=False)
            mentions.append(mention); local_mentions.append(mention)
        for key, expression in INTERPRETATIONS.items():
            for m in re.finditer(expression, text, re.I):
                ci = next(i for i,(a,b) in enumerate(clauses) if a <= m.start() < b)
                contexts.append(dict(report_id=rid, cohort=row['cohort'], sequence=row['sequence'],
                    kind='interpretation_review', key=key, start=m.start(), end=m.end(), text=m[0],
                    clause=clause_records[ci], vector_bank_inclusion=False,
                    reason='etiology_mechanism_or_differential_requires_separate_review'))
        for m in re.finditer(r'\b\d+(?:\.\d+)?(?:\s*[*x×]\s*\d+(?:\.\d+)?){0,2}\s*(?:mm|millimeters)\b', text, re.I):
            ci = next(i for i,(a,b) in enumerate(clauses) if a <= m.start() < b)
            contexts.append(dict(report_id=rid, cohort=row['cohort'], sequence=row['sequence'], kind='measurement',
                start=m.start(), end=m.end(), text=m[0], values=[float(n) for n in re.findall(r'\d+(?:\.\d+)?',m[0])],
                unit='mm', dimension_axes='unspecified_in_text', clause=clause_records[ci], vector_bank_inclusion=False))
        for m in re.finditer(r'\b(one|two|three|four|five|[1-9])\s+(?:(?:small|large|punctate|nodular|irregular)\s+){0,2}(lesions?|foci|nodules?|masses)\b',text,re.I):
            number = {'one':1,'two':2,'three':3,'four':4,'five':5}.get(m[1].lower())
            if number is None:
                number = int(m[1])
            ci = next(i for i,(a,b) in enumerate(clauses) if a <= m.start() < b)
            contexts.append(dict(report_id=rid, cohort=row['cohort'], sequence=row['sequence'], kind='count',
                start=m.start(), end=m.end(), text=m[0], value=number, object_name=m[2],
                clause=clause_records[ci], vector_bank_inclusion=False))
        for m in re.finditer(r'\b(?:anterior|posterior|superior|inferior|adjacent to|beneath|under|above|between|crossing|crosses|compressing|displacing)\b', text, re.I):
            ci = next(i for i,(a,b) in enumerate(clauses) if a <= m.start() < b)
            contexts.append(dict(report_id=rid, cohort=row['cohort'], sequence=row['sequence'], kind='spatial_relation_cue',
                start=m.start(), end=m.end(), text=m[0], clause=clause_records[ci],
                relation_arguments='preserved_in_clause_not_resolved_as_ground_truth', vector_bank_inclusion=False))
        report_records.append(dict(report_id=rid, cohort=row['cohort'], sequence=row['sequence'],
            finding=text, clauses=clause_records, mentions=[m['mention_id'] for m in local_mentions],
            cross_clause_coreference='not_resolved', patient_instance_labels_created=False))
    write_lines(args.output / 'mentions.jsonl', mentions)
    write_lines(args.output / 'reports_context.jsonl', report_records)
    write_lines(args.output / 'context_fields.jsonl', contexts)
    write_lines(args.output / 'ambiguous_mentions.jsonl', unresolved)
    write_lines(args.output / 'suppressed_components.jsonl', suppressed)
    return mentions, contexts, unresolved


def reconcile(args, rows, mentions, contexts):
    """Keep all independent audit inputs, including residual strings and rare terms."""
    by_report = defaultdict(list)
    for m in mentions:
        by_report[m['report_id']].append(m)
    ner_groups = defaultdict(lambda: dict(concept_ids=set(), reports=set(), uncovered_reports=set()))
    for span in read_lines(args.audit_input / 'ner_spans.jsonl'):
        ids = {m['concept_id'] for m in by_report[span['report_id']]
               if span['start'] < m['end'] and m['start'] < span['end']}
        group = ner_groups[span['normalized']]
        group['concept_ids'].update(ids); group['reports'].add(span['report_id'])
        if not ids:
            group['uncovered_reports'].add(span['report_id'])
    ner = []
    for original in read_lines(args.audit_input / 'ner_phrase_candidates.jsonl'):
        g = ner_groups[original['normalized']]
        ner.append(dict(original, v2_overlapping_concepts=sorted(g['concept_ids']),
            reports_without_concept_overlap=len(g['uncovered_reports']),
            disposition='component_overlap_not_semantic_coverage' if g['concept_ids'] else 'residual_review',
            patient_label=False))
    write_lines(args.output / 'ner_reconciliation.jsonl', ner)
    observed = defaultdict(set)
    for m in mentions:
        observed[norm(m['text'])].add(m['concept_id'])
    lexical = []
    for original in read_lines(args.audit_input / 'umls_phrase_candidates.jsonl'):
        ids = sorted(observed.get(original['normalized'], set()))
        lexical.append(dict(original, v2_surface_form_concepts=ids,
            disposition='observed_surface_form_not_all_CUIs_accepted' if ids else 'residual_context_or_nonconcept_review'))
    write_lines(args.output / 'umls_reconciliation.jsonl', lexical)
    # Unlike the independent mining pass, this pass measures string containment only.
    all_words = defaultdict(lambda: dict(total=0, in_concept_span=0, in_context_field=0))
    context_index = defaultdict(list)
    for c in contexts:
        context_index[c['report_id']].append((c['start'], c['end']))
    for report in read_lines(args.output / 'reports_context.jsonl'):
        for clause in report['clauses']:
            context_index[report['report_id']].extend((c['start'], c['end']) for c in clause['assertion_cues'])
    for row in rows:
        for match in re.finditer(r"[A-Za-z]+(?:[-'][A-Za-z]+)*", row['finding']):
            w = all_words[match[0].lower()]; w['total'] += 1
            if any(m['start'] <= match.start() and match.end() <= m['end'] for m in by_report[row['id']]):
                w['in_concept_span'] += 1
            if any(a <= match.start() and match.end() <= b for a,b in context_index[row['id']]):
                w['in_context_field'] += 1
    words = []
    for original in read_lines(args.audit_input / 'word_inventory.jsonl'):
        stats = all_words[original['normalized']]
        words.append(dict(original, v2_string_containment=stats,
            disposition='has_concept_or_context_span' if stats['in_concept_span'] or stats['in_context_field'] else
                        'unassigned_word_review_includes_ordinary_language'))
    write_lines(args.output / 'word_reconciliation.jsonl', words)
    write_tsv(args.output / 'word_reconciliation.tsv', [dict(r, **r['v2_string_containment']) for r in words],
              ['normalized', 'reports', 'brats_reports', 'total', 'in_concept_span', 'in_context_field', 'disposition'])
    return dict(ner_strings_retained=len(ner), umls_strings_retained=len(lexical), word_forms_retained=len(words),
                ner_strings_without_any_concept_overlap=sum(not r['v2_overlapping_concepts'] for r in ner),
                warning='Overlap and word containment are not clinical concept recall; no residual is silently dropped.')


def build(args, rows, vocabulary):
    link_audit = json.loads((args.output / 'link_audit.json').read_text())
    if link_audit['spec_sha256'] != sha(ROOT / 'scripts/bindformer_v2_spec.py'):
        raise ValueError('Vocabulary changed: rerun --stage link before building')
    decisions = json.loads(args.decisions.read_text())
    if set(decisions) != {s['key'] for s in vocabulary}:
        raise ValueError('Each specification needs an explicit CUI/local-ID decision')
    umls = {r['cui']: r for r in read_lines(args.output / 'umls_evidence.jsonl')}
    candidates = {r['key']: r for r in read_lines(args.output / 'umls_candidates.jsonl')}
    source_versions = json.loads((args.output / 'source_versions.json').read_text())
    mentions, contexts, unresolved = extract(args, rows, vocabulary)
    by_key = defaultdict(list)
    for m in mentions:
        by_key[m['key']].append(m)
    contained = defaultdict(list)
    row_map = {r['id']:r for r in rows}
    for m in read_lines(args.output / 'suppressed_components.jsonl'):
        contained[m['key']].append(m)
    bank, pending, all_candidates = [], [], []
    for spec in vocabulary:
        key = spec['key']; decision = decisions[key]; cui = decision['selected_cui']
        support = by_key[key]
        reference_ids = {m['report_id'] for m in support} | {m['report_id'] for m in contained[key]}
        compatible_ids = {rid for rid in reference_ids if row_map[rid]['sequence'] in spec['sequence_scope']}
        evidence = umls[cui] if cui else None
        if cui and not evidence['synonyms']:
            raise ValueError(f'No valid English source names for {key}/{cui}')
        original_definition = None
        if decision['selected_definition_aui']:
            choices = [d for d in evidence['definitions'] if d['aui'] == decision['selected_definition_aui'] and
                       d['source'] == decision['selected_definition_source']]
            if len(choices) != 1:
                raise ValueError(f'Definition identity is not unique: {key}')
            original_definition = choices[0]
        if not spec['short_definition'].strip():
            raise ValueError(f'Missing description: {key}')
        membership = [spec['role']]
        if spec['field'] == 'degree':
            membership = ['finding', 'signal', 'morphology_extent']
        item = dict(spec, role_memberships=membership, selected_cui=cui,
            identity=decision['identity_status'], cui_review=decision,
            observed_aliases=dict(sorted(Counter(m['text'] for m in support).items())),
            train_evidence=dict(reports=len(reference_ids), retained_mention_reports=len({m['report_id'] for m in support}),
                compatible_reports=len(compatible_ids), retained_mentions=len(support),
                contained_component_mentions=len(contained[key]),
                by_cohort=dict(Counter(row_map[r]['cohort'] for r in sorted(reference_ids))),
                by_sequence=dict(Counter(row_map[r]['sequence'] for r in sorted(reference_ids))),
                examples=[dict(report_id=m['report_id'], start=m['start'], end=m['end'], text=m['text'],
                               clause=m['clause']['text'], sequence=m['sequence']) for m in support[:3]]),
            source=dict(umls_release='2026AA' if cui else None, definition_source=original_definition,
                definition_source_versions=source_versions.get(original_definition['source'], []) if original_definition else [],
                terminology_sources=evidence['sources'] if evidence else [],
                semantic_types=evidence['semantic_types'] if evidence else [],
                text_provenance='project_authored_short_description_not_verbatim_UMLS',
                related_concept_evidence=decision.get('related_concept_evidence',[]),
                local_schema_version='bindformer_brain_v2.0',
                annotation_status='vocabulary_review_only_not_patient_annotation'),
            candidate_cuis=[m['cui'] for m in candidates[key]['matches']],
            vector_input=f"{spec['name']}. {spec['short_definition']}")
        if spec['inclusion'] == 'schema_control_not_observed_term' or compatible_ids:
            item['bank_status'] = 'included_research_vocabulary'
            bank.append(item)
        else:
            item['bank_status'] = 'deferred_no_compatible_training_phrase'
            pending.append(item)
        all_candidates.append(item)
    # A fixed role/key order prevents embedding and mask row drift.
    bank.sort(key=lambda x:(ROLES.index(x['role']), x['field'], x['key']))
    for i, item in enumerate(bank):
        item['embedding_index'] = i
    interpretation_groups = defaultdict(list)
    for c in contexts:
        if c['kind'] == 'interpretation_review':
            interpretation_groups[c['key']].append(c)
    for key, records in sorted(interpretation_groups.items()):
        pending.append(dict(key=key, bank_status='deferred_interpretation_or_etiology_review',
                            report_count=len({r['report_id'] for r in records}), examples=records[:3],
                            reason='Retained in context_fields; not silently promoted to visual fact concepts.'))
    write_lines(args.output / 'concepts.jsonl', bank)
    write_lines(args.output / 'candidates.jsonl', all_candidates)
    write_lines(args.output / 'pending_candidates.jsonl', pending)
    write_tsv(args.output / 'concepts.tsv', [dict(r,
        train_reports=r['train_evidence']['reports'], compatible_reports=r['train_evidence']['compatible_reports'],
        definition_source=r['source']['definition_source']['source'] if r['source']['definition_source'] else '',
        decision_reason=r['cui_review']['decision_reason']) for r in bank],
        ['embedding_index','concept_id','role','field','role_memberships','name','short_definition','selected_cui','identity',
         'aliases','observed_aliases','sequence_scope','train_reports','compatible_reports','definition_source','decision_reason'])
    # Trace old seeds without treating their former unchecked CUIs as accepted identities.
    migration = []
    bank_by_key = {s['key']:s for s in bank}
    rename = {'hemisphere':'cerebral_hemisphere'}
    for old in read_lines(args.old_candidates):
        key = old['candidate_id'].split(':')[-1]; new = bank_by_key.get(rename.get(key,key))
        migration.append(dict(v1_key=key, v2_concept_id=new['concept_id'] if new else None,
            status='retained_with_corpus_alias_and_CUI_review' if new else 'not_in_active_v2_bank_retained_as_extension_candidate',
            old_candidate_cuis=[x['cui'] for x in old['umls_candidates']],
            selected_cui=new['selected_cui'] if new else None,
            reason=new['cui_review']['decision_reason'] if new else 'No corresponding active reviewed entry; old record remains in v1.'))
    write_lines(args.output / 'v1_migration.jsonl', migration)
    reconciliation = reconcile(args, rows, mentions, contexts)
    schema = dict(version='bindformer_brain_v2.0', roles=ROLES, sequences=SEQUENCES,
        fields={role:sorted({s['field'] for s in bank if role in s['role_memberships']}) for role in ROLES},
        context_only_fields=['assertion_cues','object_anchor_candidates','spatial_relation_cue','measurement','count','interpretation_review'],
        relation_policy='Objects are textual candidate references. No patient-instance or tuple truth is asserted.',
        missingness_policy='not_mentioned, negative, uncertain, normality and explicit unknown stay distinct.',
        numbers_policy='Measurements and counts are structured values, not one concept per numeric token.',
        alias_policy='Full phrases, bounded explicit aliases and head-supported coordination; no arbitrary adjective-prefix CUI mapping.',
        general_degree_roles=['finding','signal','morphology_extent'])
    write_json(args.output / 'schema.json', schema)
    manifest = dict(version='bindformer_brain_v2.0', reports=len(rows), cohorts=dict(Counter(r['cohort'] for r in rows)),
        source_manifest=str(args.manifest), source_manifest_sha256=sha(args.manifest),
        script_sha256=sha(__file__), spec_sha256=sha(ROOT / 'scripts/bindformer_v2_spec.py'),
        decisions_sha256=sha(args.decisions), independent_audit_sha256=sha(args.audit_input / 'audit.json'),
        concepts=len(bank), umls_linked=sum(bool(s['selected_cui']) for s in bank),
        local_id=sum(s['selected_cui'] is None for s in bank), controls=sum(s['field']=='control' for s in bank),
        primary_roles=dict(Counter(s['role'] for s in bank)),
        role_memberships={r:sum(r in s['role_memberships'] for s in bank) for r in ROLES},
        mentions=len(mentions), unresolved_mentions=len(unresolved), context_fields=len(contexts),
        context_field_types=dict(Counter(c['kind'] for c in contexts)),
        assertion_cue_types=dict(Counter(c['kind'] for report in read_lines(args.output/'reports_context.jsonl')
                                        for clause in report['clauses'] for c in clause['assertion_cues'])),
        deferred_specifications=sum(c['bank_status'].startswith('deferred') for c in all_candidates),
        vocabulary_scope_counts={'brats_train':sum(s['field']=='control' or any(c.startswith('BraTS_')
                                      for c in s['train_evidence']['by_cohort']) for s in bank),'all_train':len(bank)},
        scope_note='The full manifest contains 2177 train reports; current default data.datasets=[brats] selects 1960.',
        reconciliation=reconciliation, clinical_annotation=False, validation_or_test_reports_used=False,
        files={name:sha(args.output/name) for name in ['concepts.jsonl','candidates.jsonl','mentions.jsonl',
                    'reports_context.jsonl','context_fields.jsonl','schema.json','pending_candidates.jsonl']})
    write_json(args.output / 'build_manifest.json', manifest)
    print(json.dumps({k:manifest[k] for k in ['concepts','umls_linked','local_id','controls','primary_roles','mentions','deferred_specifications']}),flush=True)


def encode(args):
    from src.evaluation.clinical_runtime import activate
    activate()
    import numpy as np
    import torch
    import torch.nn.functional as F
    from transformers import AutoModel, AutoTokenizer
    from safetensors.torch import save_file
    torch.set_num_threads(args.threads)
    model_manifest = json.loads((args.encoder / 'asset_manifest.json').read_text())
    expected = ('cambridgeltl/SapBERT-from-PubMedBERT-fulltext','090663c3ae57bf35ffe4d0d468a2a88d03051a4d')
    if (model_manifest['repo_id'],model_manifest['revision']) != expected:
        raise ValueError('Unexpected encoder identity/revision')
    for f in model_manifest['files']:
        if sha(args.encoder / f['path']) != f['sha256']:
            raise ValueError(f'Encoder asset checksum mismatch: {f["path"]}')
    bank = read_lines(args.output / 'concepts.jsonl')
    build_manifest = json.loads((args.output / 'build_manifest.json').read_text())
    if build_manifest['files']['concepts.jsonl'] != sha(args.output / 'concepts.jsonl'):
        raise ValueError('Concept bank changed since build')
    tokenizer = AutoTokenizer.from_pretrained(args.encoder, local_files_only=True)
    model = AutoModel.from_pretrained(args.encoder, local_files_only=True)
    model.eval(); model.requires_grad_(False)
    texts = [r['vector_input'] for r in bank]
    lengths = [len(tokenizer(t, truncation=False)['input_ids']) for t in texts]
    if not bank or max(lengths) > args.max_tokens:
        raise ValueError('Empty bank or description exceeds token limit; review text, never silently truncate')
    chunks = []
    with torch.inference_mode():
        for start in range(0,len(texts),args.batch_size):
            batch = tokenizer(texts[start:start+args.batch_size],padding=True,truncation=False,return_tensors='pt')
            cls = model(**batch).last_hidden_state[:,0,:].float()
            if not torch.isfinite(cls).all() or (cls.norm(dim=-1) < 1e-8).any():
                raise ValueError('Nonfinite/zero CLS vector')
            chunks.append(F.normalize(cls,p=2,dim=-1))
    embeddings = torch.cat(chunks).contiguous()
    if embeddings.shape != (len(bank),768):
        raise ValueError('Unexpected SapBERT width')
    role_mask = torch.tensor([[r in s['role_memberships'] for s in bank] for r in ROLES],dtype=torch.bool)
    sequence_mask = torch.tensor([[seq in s['sequence_scope'] for s in bank] for seq in SEQUENCES],dtype=torch.bool)
    scopes = ['brats_train','all_train']
    scope_mask = torch.tensor([[s['field']=='control' or any(c.startswith('BraTS_')
                               for c in s['train_evidence']['by_cohort']) for s in bank],
                              [True for s in bank]],dtype=torch.bool)
    np.save(args.output/'embeddings.npy',embeddings.numpy())
    np.save(args.output/'role_mask.npy',role_mask.numpy())
    np.save(args.output/'sequence_mask.npy',sequence_mask.numpy())
    np.save(args.output/'scope_mask.npy',scope_mask.numpy())
    save_file(dict(embeddings=embeddings,role_mask=role_mask,sequence_mask=sequence_mask,scope_mask=scope_mask),
              str(args.output/'concept_bank.safetensors'))
    ids = [s['concept_id'] for s in bank]
    write_json(args.output/'concept_index.json',dict(concept_ids=ids,roles=ROLES,sequences=SEQUENCES,
               vocabulary_scopes=scopes,default_vocabulary_scope='brats_train'))
    write_lines(args.output/'encoding_inputs.jsonl',[dict(index=i,concept_id=ids[i],text=texts[i],tokens=lengths[i],
                text_sha256=hashlib.sha256(texts[i].encode()).hexdigest()) for i in range(len(bank))])
    norms = embeddings.norm(dim=-1)
    manifest = dict(version='bindformer_brain_v2.0', encoder=model_manifest['repo_id'],revision=model_manifest['revision'],
        pooling='last_hidden_state[:, 0, :] (final-layer CLS)',normalize='L2 along embedding dimension',
        template='{name}. {short_definition}',dtype='float32',device='cpu',frozen=True,
        max_tokens=args.max_tokens,max_observed_tokens=max(lengths),truncated_concepts=0,shape=list(embeddings.shape),
        role_mask_shape=list(role_mask.shape),sequence_mask_shape=list(sequence_mask.shape),
        scope_mask_shape=list(scope_mask.shape),scope_concept_counts=dict(zip(scopes,scope_mask.sum(-1).tolist())),
        norm_range=[norms.min().item(),norms.max().item()],concepts_sha256=sha(args.output/'concepts.jsonl'),
        build_manifest_sha256=sha(args.output/'build_manifest.json'),encoder_asset_manifest_sha256=sha(args.encoder/'asset_manifest.json'),
        script_sha256=sha(__file__),files={name:sha(args.output/name) for name in ['embeddings.npy','role_mask.npy',
            'sequence_mask.npy','scope_mask.npy','concept_bank.safetensors','concept_index.json','encoding_inputs.jsonl']},
        note='Offline concept memory only. No report text, patient features or extra LLM calls enter these vectors.')
    write_json(args.output/'embedding_manifest.json',manifest)
    print(json.dumps({k:manifest[k] for k in ['shape','role_mask_shape','sequence_mask_shape','max_observed_tokens','norm_range']}),flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage', choices=['link', 'extract', 'build', 'encode', 'validate', 'all'], default='all')
    p.add_argument('--manifest', type=Path, default=ROOT / 'assets/data/RadGenome-Brain/training_manifests/single_sequence_findings_coref_v2/train.jsonl')
    p.add_argument('--audit-input', type=Path, default=ROOT / 'assets/knowledge/bindformer_brain_corpus_audit_v1')
    p.add_argument('--meta', type=Path, default=ROOT / 'assets/knowledge/umls/META')
    p.add_argument('--output', type=Path, default=ROOT / 'assets/knowledge/bindformer_brain_candidates_v2')
    p.add_argument('--decisions', type=Path, default=ROOT / 'Pipeline_0929/bindformer_v2_cui_decisions.json')
    p.add_argument('--old-candidates', type=Path, default=ROOT / 'assets/knowledge/bindformer_brain_candidates_v1/candidates.jsonl')
    p.add_argument('--encoder', type=Path, default=ROOT / 'assets/models/knowledge/sapbert')
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--max-tokens', type=int, default=128)
    args = p.parse_args()
    if min(args.threads,args.batch_size,args.max_tokens) <= 0:
        raise ValueError('Thread count, batch size and token limit must be positive')
    rows, vocabulary = inputs(args)
    if args.stage in ['link', 'all']:
        link(args, vocabulary)
    if args.stage == 'extract':
        mentions, contexts, unresolved = extract(args, rows, vocabulary)
        counts = Counter(m['key'] for m in mentions)
        print(json.dumps(dict(mentions=len(mentions), context_fields=len(contexts), ambiguous_mentions=len(unresolved),
                             missing=[s['key'] for s in vocabulary if s['inclusion']=='corpus_observed' and not counts[s['key']]])))
    if args.stage in ['build','all']:
        build(args, rows, vocabulary)
    if args.stage in ['encode','all']:
        encode(args)
    if args.stage in ['validate','all']:
        from scripts.validate_bindformer_knowledge_v2 import validate
        validate(args.output)


if __name__ == '__main__':
    main()
