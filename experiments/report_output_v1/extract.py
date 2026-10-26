"""Deterministic text-only extraction. No reference answers enter prediction parsing.

Reuses the existing train-derived concept aliases and assertion rules. Conservative
single-lesion discourse labels are NOT spatial or multi-lesion instance truth.
"""
from functools import lru_cache
import re

from scripts.bindformer_v2_spec import specs, literal_pattern
from scripts.build_bindformer_knowledge_v2 import segments
from experiments.report_facts_v1.extract import extract_report
from .schema import FIELDS


@lru_cache(maxsize=1)
def vocabulary():
    items = specs()
    patterns = {
        s['key']: [re.compile(literal_pattern(a), re.I) for a in s['aliases']]
        + [re.compile(p, re.I) for p in s.get('extra_patterns', [])]
        for s in items
    }
    return {s['key']: s for s in items}, patterns


def mentions(row):
    text, rid = row['finding'], row['id']
    by_key, patterns = vocabulary()
    clauses = segments(text)
    found = {}
    for key, expressions in patterns.items():
        for expression in expressions:
            for match in expression.finditer(text):
                found[(key, match.start(), match.end())] = None
    # Preserve coordinated anatomy, so 'frontal-parietal lobes' is not reduced
    # to whichever full lobe name happens to match the lexical inventory.
    lobes = dict(frontal='frontal_lobe', fronto='frontal_lobe', parietal='parietal_lobe',
                 parieto='parietal_lobe', temporal='temporal_lobe', temporo='temporal_lobe',
                 occipital='occipital_lobe', occipito='occipital_lobe', insular='insula', insula='insula')
    atom = '(?:' + '|'.join(sorted(lobes, key=len, reverse=True)) + ')'
    coord = re.compile(r'\b' + atom + r'(?:(?:[-, /]+|\s+(?:and\s+)?)' + atom + r')+\s+(?:lobes?|regions?)\b', re.I)
    for match in coord.finditer(text):
        for part in re.finditer(atom, match[0], re.I):
            key = lobes[part[0].lower()]
            if re.search(r'regions?$', match[0], re.I) and key.endswith('_lobe'):
                key = key.replace('_lobe', '_region')
            found[(key, match.start() + part.start(), match.start() + part.end())] = None
    chosen = []
    for key, start, end in sorted(found, key=lambda x: (-(x[2]-x[1]), x[1], x[0])):
        spec = by_key[key]
        if key == 'mass' and re.match(r'[- ](?:like|effect)\b', text[end:], re.I):
            continue
        if any(a <= start and end <= b and
               ((by_key[k]['role'], by_key[k]['field']) == (spec['role'], spec['field']) or
                (spec['field'] == 'laterality' and by_key[k]['field'] == 'displacement_direction'))
               for k, a, b in chosen):
            continue
        chosen.append((key, start, end))
    result = []
    for key, start, end in sorted(chosen, key=lambda x: (x[1], x[2], x[0])):
        ci, (a, b) = next((i, ab) for i, ab in enumerate(clauses) if ab[0] <= start < ab[1])
        spec = by_key[key]
        result.append(dict(key=key, start=start, end=end,
                           mention_id=f'{rid}:{start}:{end}:{key}', field=spec['field'],
                           sequence_compatible=row['sequence'] in spec['sequence_scope'],
                           clause=dict(start=a, end=b, text=text[a:b], clause_id=f'{rid}:clause:{ci}')))
    return result


def extract(row):
    result = extract_report(row, mentions(row))
    result['facts'] = {k: v for k, v in result['facts'].items() if k in FIELDS}
    return result


def clause_bounds(text, start):
    return next((a, b) for a, b in segments(text) if a <= start < b)


def evidence_span(text, field, fact, evidence):
    """Retain the assertion, especially the negation of edema, in the CE mask."""
    start, end = evidence['start'], evidence['end']
    if not 0 <= start < end <= len(text):
        raise ValueError('Evidence outside report')
    a, b = clause_bounds(text, start)
    if field == 'edema':
        if text[start:end].lower() == 'edematous':
            noun = re.match(r'\s+changes\b', text[end:b], re.I)
            if noun:
                end += noun.end()
        prefix, tail = text[a:start], text[end:b]
        cue = re.search(r'\b(?:no|without|absence of|lack of|not associated with|does not show)\b[^,;]*$', prefix, re.I)
        if fact['value'] == 'absent' and cue:
            start = a + cue.start()
        else:
            modifiers = r'(?:(?:associated|surrounding|perilesional|peritumoral|brain|cerebral|parenchymal|mild|marked|significant|extensive|large|adjacent|accompanying)\s+)*'
            predicate = re.search(r'\b(?:with|shows?|exhibits?|has|causes?|There\s+is(?:\s+a(?:\s+large)?\s+area\s+of)?)\s+' + modifiers + '$', prefix, re.I)
            if predicate:
                start = a + predicate.start()
            else:
                modifier = re.search(modifiers + '$', prefix, re.I)
                start = a + modifier.start()
        suffix = re.match(r'\s+(?:is|was)\s+(?:also\s+)?(?:absent|not present|present|noted|evident|seen|observed)\b', tail, re.I)
        if suffix:
            end += suffix.end()
        if fact['value'] == 'absent' and not cue and not suffix:
            raise ValueError('Negative edema span has no explicit assertion cue')
    return dict(start=start, end=end, text=text[start:end], clause_start=a, clause_end=b)
