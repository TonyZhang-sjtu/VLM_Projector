"""Build immutable report-output labels; never reads images or segmentation.

Train labels inherit the reviewed v1 candidates, with extra explicit exclusions.
Validation labels apply the frozen train-derived parser, never train the parser.
"""
import argparse
from collections import Counter, defaultdict
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re

from src.config import ROOT, file_hash
from src.data.sequence_mapping import sequence_of
from experiments.report_facts_v1.extract import ANCHOR
from .extract import extract, evidence_span
from .schema import VERSION, FIELDS


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def text_hash(text):
    return hashlib.sha256(text.encode()).hexdigest()


def write_rows(path, rows):
    Path(path).write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows))


def refine_facts(source, facts):
    """Do not promote edema mentioned as an alternative explanation to a fact."""
    accepted, rejected = {}, []
    for field, fact in facts.items():
        if field not in FIELDS:
            continue
        fact = deepcopy(fact)
        evidence = []
        for item in fact['evidence']:
            clause = item['evidence']
            if field == 'edema' and fact['value'] == 'present' and re.search(
                r'\b(?:suggestive|suggesting|indicating|indicative|reflecting|consistent with)\b|'
                r'\bedema\s+or\b|\bor\s+(?:\w+\s+){0,2}edema\b', clause, re.I
            ):
                rejected.append(dict(field=field, reason='edema_inference_or_alternative', evidence=clause))
                continue
            evidence.append(item)
        if evidence:
            fact['evidence'] = evidence
            accepted[field] = fact
    return accepted, rejected


def replacement(text, field, fact, evidence, value):
    """Return a local edit, or refuse an unsafe/unsupported grammatical edit."""
    start, end = evidence['start'], evidence['end']
    if field == 'edema' and text[start:end].lower() == 'edematous':
        noun = re.match(r'\s+changes\b', text[end:], re.I)
        if noun:
            end += noun.end()
    old = text[start:end]
    if field == 'laterality':
        new = value
    elif field == 'site':
        # The inherited inventory can identify 'cerebellar' inside 'cerebellar
        # hemisphere'; replace the full noun phrase, not 'frontal lobe hemisphere'.
        if old.lower() == 'cerebellar':
            tail = re.match(r'\s+hemisphere', text[end:], re.I)
            if tail is None:
                return None
            end += tail.end()
        new = value.replace('_', ' ')
    elif field == 'signal':
        if 'mixed_signal' in (value, fact['value']):
            return None
        stem = dict(hyperintense='hyper', hypointense='hypo', isointense='iso')[value]
        if re.search(r'(?:hyper|hypo|iso)[- ]?intens', old, re.I):
            new = re.sub(r'(?:hyper|hypo|iso)([- ]?intens)', stem + r'\1', old, flags=re.I)
        elif re.search(r'\b(?:high|low|intermediate)\b', old, re.I):
            adjective = dict(hyperintense='high', hypointense='low', isointense='isointense')[value]
            new = re.sub(r'\b(?:high|low|intermediate)\b', adjective, old, flags=re.I)
        else:
            return None
        if value == 'isointense':
            degree = re.search(r'\b(?:slightly|mildly|markedly|very)\s+$', text[:start], re.I)
            if degree:
                start = degree.start()
    elif field == 'edema':
        span = evidence_span(text, field, fact, evidence)
        a = span['clause_start']
        prefix, tail = text[a:start], text[end:span['clause_end']]
        if fact['value'] == 'absent':
            match = re.search(r'\b(no|without|absence of|lack of|not associated with|does not show)\b[^,;]*$', prefix, re.I)
            if match is None:
                match = re.match(r'\s+(?:is|was)\s+(absent|not present)\b', tail, re.I)
                if match is None:
                    return None
                end += match.end()
                new = re.sub(r'\b(?:absent|not present)\b', 'present', text[start:end], flags=re.I)
            else:
                start = a + match.start()
                old = text[start:end]
                cue = match[1].lower()
                if cue == 'without' and re.match(r'\s+causing\b', old[len(match[1]):], re.I):
                    return None
                cue_replacement = {'no': '', 'without': 'with', 'absence of': 'presence of',
                                   'lack of': 'presence of', 'not associated with': 'associated with',
                                   'does not show': 'shows'}[cue]
                new = cue_replacement + old[len(match[1]):]
                # 'without any edema' -> 'with edema', not 'with any edema'.
                new = re.sub(r'\bany\s+', '', new, flags=re.I).lstrip()
        else:
            # Only edit a recognized local predicate; do not produce e.g.
            # 'peritumoral no edema' or negate a whole list of other findings.
            modifiers = r'(?:(?:associated|surrounding|perilesional|peritumoral|brain|cerebral|parenchymal|mild|marked|significant|extensive|large|adjacent|accompanying)\s+)*'
            match = re.search(r'\b(with|shows?|exhibits?|has|causes?)\s+(' + modifiers + r')$', prefix, re.I)
            if match:
                start = a + match.start()
                verb = match[1]
                new = ('without' if verb.lower() == 'with' else verb + ' no') + ' ' + text[a+match.start(2):end]
            else:
                match = re.search(r'\b(There\s+is|There\s+is\s+a(?:\s+large)?\s+area\s+of)\s+(' + modifiers + r')$', prefix, re.I)
                if match:
                    start = a + match.start()
                    new = 'There is no ' + text[a+match.start(2):end]
                else:
                    match = re.match(r'\s+(?:is\s+)?(?:also\s+)?(?:present|noted|evident|seen|observed)\b', tail, re.I)
                    if match is None:
                        return None
                    start = span['start']
                    noun = text[start:evidence['end']]
                    end += match.end()
                    new = noun + ' is absent'
            # 'No mild/significant edema' is NOT equivalent to absent edema.
            # Remove severity modifiers when creating an explicit absence.
            new = re.sub(r'\b(?:mild|marked|significant|extensive|large)\s+', '', new, flags=re.I)
    else:
        raise ValueError(field)
    if text[start:start+1].isupper() and new:
        new = new[0].upper() + new[1:]
    if new == text[start:end] or not new.strip():
        return None
    return dict(start=start, end=end, original=text[start:end], replacement=new)


def edit_spans(spans, edit, text):
    """Project known evidence through one edit, preserving all scored fields."""
    a, b, new = edit['start'], edit['end'], edit['replacement']
    delta = len(new) - (b-a)
    output = {}
    for field, values in spans.items():
        converted = []
        for span in values:
            start, end = span['start'], span['end']
            if end <= a:
                pass
            elif start >= b:
                start += delta; end += delta
            elif start <= a and end >= b:
                end += delta
            elif a <= start and end <= b:
                start, end = a, a+len(new)
            else:
                raise ValueError('Partially overlapping attribute edit')
            converted.append(dict(start=start, end=end, text=text[start:end]))
        output[field] = converted
    return output


def make_row(source, inherited, pairs=()):
    text = source['finding']
    facts, rejected = refine_facts(source, inherited['facts'])
    row = {k: source[k] for k in ('id', 'patient_id', 'cohort', 'sequence', 'split')}
    row.update(version=VERSION, finding_sha256=text_hash(text), facts={}, groups=[], rejected=rejected,
               attachment='single_lesion_text_heuristic_not_verified_instance')
    row['skip_reason'] = inherited.get('skip_reason')
    groups = {}
    for field, fact in facts.items():
        spans = [evidence_span(text, field, fact, e) for e in fact['evidence']]
        row['facts'][field] = dict(value=fact['value'], spans=spans)
        first = min(fact['evidence'], key=lambda e: e['start'])
        span = evidence_span(text, field, fact, first)
        key = (span['clause_start'], span['clause_end'])
        group = groups.setdefault(key, dict(group_id=f'{source["id"]}:clause:{key[0]}',
                  entity_id='L1_text_heuristic', start=key[0], end=key[1], spans={}, candidates=[]))
        group['spans'][field] = [span]
    for group in groups.values():
        group['fields'] = sorted(group['spans'])
        group['joint_eligible'] = len(group['fields']) >= 2 and bool(ANCHOR.search(text[group['start']:group['end']]))
    for pair in pairs:
        field = pair['field']
        if field not in facts:
            row['rejected'].append(dict(pair_id=pair['pair_id'], field=field, reason='fact_not_retained'))
            continue
        fact = facts[field]
        if pair['positive'] != fact['value']:
            raise ValueError('Pair target disagrees with reference')
        evidence = min(fact['evidence'], key=lambda e: e['start'])
        span = evidence_span(text, field, fact, evidence)
        group = groups[(span['clause_start'], span['clause_end'])]
        edit = replacement(text, field, fact, evidence, pair['negative'])
        if edit is None:
            row['rejected'].append(dict(pair_id=pair['pair_id'], field=field, reason='no_safe_local_edit'))
            continue
        # Stop after this group's evidence, excluding later clauses that could
        # repeat the original answer. No generated EOT is included in the score.
        stop = max(edit['end'], max(s['end'] for ss in group['spans'].values() for s in ss))
        candidate = text[:edit['start']] + edit['replacement'] + text[edit['end']:stop]
        candidate_spans = edit_spans(group['spans'], edit, candidate)
        group['candidates'].append(dict(**pair, edit=edit, text=candidate, spans=candidate_spans))
    row['groups'] = sorted(groups.values(), key=lambda g: g['start'])
    return row


def build(manifest_dir, output, facts_path, pairs_path):
    manifest_dir, output, facts_path, pairs_path = map(Path, (manifest_dir, output, facts_path, pairs_path))
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite labels: {output}')
    inherited = {r['id']: r for r in read_rows(facts_path)}
    pairs = {r['id']: r['pairs'] for r in read_rows(pairs_path)}
    all_rows = {}
    for split in ('train', 'val'):
        rows = read_rows(manifest_dir/f'{split}.jsonl')
        all_rows[split] = []
        for source in rows:
            if source['split'] != split:
                raise ValueError('Manifest split mismatch')
            if not source['cohort'].startswith('BraTS_'):
                continue
            source['sequence'] = sequence_of(source['cohort'], source['id'])
            if split == 'train':
                labels = inherited[source['id']]
                if labels['finding_sha256'] != text_hash(source['finding']):
                    raise ValueError('Stale inherited facts')
            else:
                labels = extract(source)
            all_rows[split].append(make_row(source, labels, pairs.get(source['id'], ()) if split == 'train' else ()))
    train_patients = {r['patient_id'] for r in all_rows['train']}
    if train_patients & {r['patient_id'] for r in all_rows['val']}:
        raise ValueError('Patient leakage between train and validation')
    # A source pair is a reciprocal two-patient decision. If one report cannot
    # be edited reliably, retain its counterpart's positive span but exclude
    # the now one-sided ranking pair. Do not create an answer-class imbalance
    # simply because one polarity is easier to rewrite grammatically.
    pair_counts = Counter(c['pair_id'] for r in all_rows['train'] for g in r['groups'] for c in g['candidates'])
    for row in all_rows['train']:
        for group in row['groups']:
            kept = []
            for candidate in group['candidates']:
                if pair_counts[candidate['pair_id']] == 2:
                    kept.append(candidate)
                else:
                    row['rejected'].append(dict(pair_id=candidate['pair_id'], field=candidate['field'],
                                               reason='reciprocal_partner_has_no_reliable_edit'))
            group['candidates'] = kept
    # Assert every source pair is either retained or explicitly accounted for.
    expected = sum(len(ps) for ps in pairs.values())
    retained = sum(len(g['candidates']) for r in all_rows['train'] for g in r['groups'])
    rejected = sum('pair_id' in x for r in all_rows['train'] for x in r['rejected'])
    if retained + rejected != expected:
        raise ValueError('Silent loss of source pairs')
    output.mkdir(parents=True)
    for split, rows in all_rows.items():
        write_rows(output/f'{split}.jsonl', rows)
    audit = [dict(id=r['id'], split=split, **reason) for split, rows in all_rows.items() for r in rows for reason in r['rejected']]
    write_rows(output/'audit.jsonl', audit)
    schema = dict(version=VERSION, fields=FIELDS, missing_label_policy='ignore', segmentation_used=False,
                  group_policy='same_clause_single_lesion_heuristic', multiple_lesions='not_supervised',
                  negative_policy='all_validated_source_pairs_no_per_report_cap',
                  review_status='rule_derived_requires_expert_review')
    (output/'schema.json').write_text(json.dumps(schema, indent=2)+'\n')
    metadata = dict(version=VERSION, manifest_hashes={s: file_hash(manifest_dir/f'{s}.jsonl') for s in all_rows},
                    input_hashes={str(p): file_hash(p) for p in (facts_path, pairs_path)},
                    source_pairs=expected, retained_pairs=retained, rejected_pairs=rejected,
                    source_hashes={str(p.relative_to(ROOT)): file_hash(p) for p in [Path(__file__),
                        Path(__file__).with_name('extract.py'), ROOT/'experiments/report_facts_v1/extract.py',
                        ROOT/'scripts/bindformer_v2_spec.py', ROOT/'scripts/build_bindformer_knowledge_v2.py']},
                    files={p.name: file_hash(p) for p in output.iterdir()}, splits={})
    for split, rows in all_rows.items():
        metadata['splits'][split] = dict(reports=len(rows), labelled_reports=sum(bool(r['facts']) for r in rows),
            fields=dict(Counter(k for r in rows for k in r['facts'])),
            joint_groups=sum(g['joint_eligible'] for r in rows for g in r['groups']),
            candidates_per_report=dict(Counter(sum(len(g['candidates']) for g in r['groups']) for r in rows)))
    (output/'metadata.json').write_text(json.dumps(metadata, indent=2)+'\n')
    return metadata


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--manifest-dir', type=Path, default=ROOT/'assets/data/RadGenome-Brain/training_manifests/single_sequence_findings_coref_v2')
    cli.add_argument('--output', type=Path, default=ROOT/'assets/knowledge/report_output_v1')
    source = ROOT/'results/ReportFacts_compression_diagnosis_20261023/report_output_supervision_design'
    cli.add_argument('--facts', type=Path, default=source/'known_train_facts.jsonl')
    cli.add_argument('--pairs', type=Path, default=source/'full_attribute_pairs.jsonl')
    args = cli.parse_args()
    print(json.dumps(build(args.manifest_dir, args.output, args.facts, args.pairs), indent=2))


if __name__ == '__main__':
    main()
