"""Versioned report-text location agreement, shared by all report evaluators.

The reference rule retains the historical single-lesion eligibility criteria.
These are text heuristics, not image-grounded or manually verified labels.
"""
import hashlib
import json
from pathlib import Path
import re

from src.evaluation.location_reference import reference_location


VERSION = 'location_phrase_v2'
REFERENCE_VERSION = 'single_lesion_reference_v1'
ANCHOR = re.compile(r'\b(?:lesion|mass|nodule|tumou?r|abnormal signal focus)\b(?![- ](?:effect|like))', re.I)
LOCATION = re.compile(
    r'\b(left|right)\s+(frontal(?:[- ](?:parietal|temporal|occipital))?'
    r'|parietal(?:[- ](?:occipital|temporal))?|temporal(?:[- ]occipital)?'
    r'|occipital|cerebellar|cerebellum|cerebellopontine|basal ganglia|thalamus|thalamic)'
    r'\b(?:\s+(lobe|region|hemisphere|angle))?', re.I)
SITE_NAMES = {'cerebellar': 'cerebellum', 'cerebellum': 'cerebellum',
              'cerebellopontine': 'cerebellopontine_angle',
              'basal ganglia': 'basal_ganglia', 'thalamus': 'thalamus', 'thalamic': 'thalamus'}


def direct_location(text):
    """Historical phrase rule plus abnormal signal focus and cerebellum aliases."""
    candidates = []
    for sentence in re.split(r'[.;\n]', text):
        anchors = list(ANCHOR.finditer(sentence))
        if not anchors:
            continue
        for match in LOCATION.finditer(sentence):
            prefix = sentence[max(0, match.start() - 60):match.start()].lower()
            if re.search('compress|displac|shift|deviat|edema|adjacent|surrounding', prefix):
                continue
            if min(abs(anchor.start() - match.start()) for anchor in anchors) > 180:
                continue
            name = match[2].lower()
            candidates.append(dict(side=match[1].lower(),
                                   site=SITE_NAMES.get(name, name + '_lobe'), text=match[0]))
    sides = {item['side'] for item in candidates}
    sites = {item['site'] for item in candidates}
    return dict(candidates=candidates, side=next(iter(sides)) if len(sides) == 1 else None,
                site=next(iter(sites)) if len(sites) == 1 else None)


def evaluate_locations(rows):
    """Score eligible references; unresolved predictions remain in denominators."""
    if len({row['id'] for row in rows}) != len(rows):
        raise ValueError('Duplicate prediction IDs in location evaluation')
    records = []
    for row in rows:
        truth = reference_location(row['reference'], row.get('sequence', ''),
                                   row.get('text_audit_status', ''))
        predicted = direct_location(row['prediction'])
        records.append(dict(id=row['id'], patient_id=row['patient_id'], truth=truth,
                            predicted=predicted,
                            reference_sha256=hashlib.sha256(row['reference'].encode()).hexdigest()))
    overall = {}
    for name, fields in (('laterality', ('side',)), ('site', ('site',)), ('side_site', ('side', 'site'))):
        eligible = [r for r in records if all(r['truth'][f] is not None for f in fields)]
        correct = sum(all(r['predicted'][f] == r['truth'][f] for f in fields) for r in eligible)
        unresolved = sum(any(r['predicted'][f] is None for f in fields) for r in eligible)
        overall[name] = dict(correct=correct, n=len(eligible),
                             accuracy=correct / len(eligible) if eligible else None,
                             patients=len({r['patient_id'] for r in eligible}),
                             unresolved=unresolved, wrong=len(eligible) - correct - unresolved)
    reference_identity = [{k: r[k] for k in ('id', 'patient_id', 'truth', 'reference_sha256')}
                          for r in sorted(records, key=lambda r: r['id'])]
    return dict(version=VERSION, reference_version=REFERENCE_VERSION,
                parser_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                reference_parser_sha256=hashlib.sha256(Path(__file__).with_name('location_reference.py').read_bytes()).hexdigest(),
                reference_labels_sha256=hashlib.sha256(json.dumps(reference_identity, sort_keys=True).encode()).hexdigest(),
                interpretation='Report-text agreement with conservative automatic reference labels; not clinical or image-grounded accuracy.',
                missing_prediction_policy='Unresolved or conflicting predicted fields count as incorrect on eligible references.',
                samples=len(rows), patients=len({r['patient_id'] for r in rows}), overall=overall)
