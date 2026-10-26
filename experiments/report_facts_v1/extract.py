"""Conservative, auditable promotion of existing candidate mentions to weak labels.

Only explicit statements in a single lesion discourse are promoted. This is a
text attachment heuristic, not clinical certification or spatial grounding.
"""
import re
from collections import defaultdict
from .schema import FIELDS, VERSION

MULTIPLE = re.compile(r'\b(?:lesions|nodules|masses|foci|multiple|numerous|bilateral|both|two|three|four|another|largest|larger|dominant|additional lesion)\b', re.I)
MISSING = re.compile(r'\b(?:no (?:mention(?:ed)?|reported)|(?:without|nor) mention|not (?:explicitly )?(?:mentioned|reported|described|specified|provided|discussed|assessed|addressed)|(?:does|do) not (?:mention|report|describe|elaborate)|no (?:(?:additional|further|explicit|detailed) )?(?:details?|information|description|comments?|observations?|specifics)|(?:lacks?|without) (?:any )?(?:mention|description|information)|unknown|indeterminate|unreported|unspecified)\b', re.I)
UNCERTAIN = re.compile(r'\b(?:may|might|could|possible|possibly|potential|likely|probable|suggests?|suggesting|suspicious)\b', re.I)
NEGATIVE = re.compile(r'\b(?:no|not|without|absence of|absent|neither|nor|lack of)\b', re.I)
LOCAL_ABSENCE = re.compile(r'\b(?:(?:no|without|absence of|lack of)\s+(?:(?:any|associated|surrounding|perilesional|peritumoral|accompanying|observable|observed|evident|apparent|peripheral|brain|tissue|cerebral|parenchymal|evidence|signs|indication|of|causing)\s+){0,8}|not (?:associated with|showing|show|exhibiting|exhibit)\s+)$', re.I)
ANCHOR = re.compile(r'\b(?:lesion|mass|nodule|tumou?r)\b(?![- ](?:effect|like))', re.I)
KEY_FIELD = {value: field for field in ('site', 'signal', 'margin') for value in FIELDS[field]['values']}
KEY_FIELD.update(left='laterality', right='laterality', edema='edema', enhancement='enhancement',
                 nonenhancing='enhancement', ring_enhancement='enhancement', nodular_enhancement='enhancement',
                 garland_enhancement='enhancement')


def extract_report(row, mentions):
    text = row['finding']
    result = dict(version=VERSION, id=row['id'], patient_id=row['patient_id'], cohort=row['cohort'],
                  sequence=row['sequence'], split=row['split'], facts={}, rejected=[],
                  attachment='single_lesion_discourse_heuristic_not_verified_instance')
    if MULTIPLE.search(text) or not ANCHOR.search(text) or 'requires_review' in row.get('text_audit_status',''):
        result['skip_reason'] = 'multiple_or_missing_lesion_anchor_or_source_requires_review'
        return result
    candidates = defaultdict(list)
    lesion_sides={m['key'] for m in mentions if m['key'] in ('left','right') and ANCHOR.search(m['clause']['text'])
                  and not re.search(r'midline|shift|displac|deviat|ventricle',m['clause']['text'],re.I)}
    if len(lesion_sides)>1:
        result['skip_reason']='multiple_lesion_sides'
        return result
    for m in mentions:
        field = KEY_FIELD.get(m['key'])
        if field is None:
            continue
        clause = m['clause']['text']
        reason = None
        if not m['sequence_compatible'] or (field == 'enhancement' and row['sequence'] != 'T1C'):
            reason = 'sequence_incompatible'
        elif MISSING.search(clause):
            reason = 'not_mentioned_or_unknown'
        elif UNCERTAIN.search(clause):
            reason = 'uncertain_or_interpretive'
        # Lesion attributes need a local explicit lesion anchor. Only edema may
        # use a separate explicit sentence (e.g. "No associated edema ...").
        elif field != 'edema' and not ANCHOR.search(clause):
            reason = 'no_local_lesion_anchor'
        elif field == 'laterality' and re.search(r'midline|shift|displac|deviat|ventricle|falx', clause, re.I):
            reason = 'direction_or_nonlesion_anatomy_ambiguous'
        elif field == 'site' and re.search(r'compress|displac|shift|adjacent|abuts?|edema|oedema', clause, re.I):
            reason = 'site_may_belong_to_secondary_effect'
        elif field == 'site' and any(n['field']=='anatomy' and n['key']!=m['key'] and
                                     n['clause']['clause_id']==m['clause']['clause_id'] for n in mentions):
            reason = 'multiple_anatomical_sites'
        elif field == 'signal' and re.search(r'edema|oedema|surrounding|necrotic|cystic|component|central|peripheral|at the center|T1.*T2|T2.*T1', clause, re.I):
            reason = 'signal_subject_or_sequence_ambiguous'
        if reason:
            result['rejected'].append(dict(mention_id=m['mention_id'], field=field, reason=reason))
            continue
        negative = bool(NEGATIVE.search(clause))
        value = m['key']
        if field in ('edema','enhancement'):
            # Limit negation to the local predicate instead of crossing commas
            # or unrelated coordinated effects. Ambiguous scopes are skipped.
            offset = m['start'] - m['clause']['start']
            prefix = re.split(r',|\band\b|\bbut\b', clause[:offset], flags=re.I)[-1]
            suffix = clause[m['end']-m['clause']['start']:]
            local_neg = bool(LOCAL_ABSENCE.search(prefix) or re.match(r'\s+(?:is|was) (?:absent|not present)\b', suffix, re.I))
            if negative and re.search(r'\b(?:significant|obvious|appreciable|marked)\b',prefix,re.I):
                result['rejected'].append(dict(mention_id=m['mention_id'],field=field,reason='qualified_negative_not_absence'))
                continue
            if negative and not local_neg:
                result['rejected'].append(dict(mention_id=m['mention_id'], field=field, reason='negation_scope_ambiguous'))
                continue
            value = 'absent' if local_neg or m['key']=='nonenhancing' else 'present'
        elif negative:
            result['rejected'].append(dict(mention_id=m['mention_id'], field=field, reason='negated_attribute'))
            continue
        candidates[field].append(dict(value=value, mention_id=m['mention_id'], start=m['start'], end=m['end'],
                                      evidence=clause, clause_id=m['clause']['clause_id'],
                                      assertion='explicit_negative' if value=='absent' else 'affirmed'))
    for field, facts in candidates.items():
        if any(r.get('field')==field and r['reason'] in ('negation_scope_ambiguous','qualified_negative_not_absence') for r in result['rejected']):
            continue
        values = {f['value'] for f in facts}
        if field == 'signal' and 'mixed_signal' in values:
            # Explicit mixed is retained; do not manufacture mixed from two
            # incompatible statements or two separate components.
            facts = [f for f in facts if f['value']=='mixed_signal']
            values = {'mixed_signal'}
        if len(values) == 1:
            result['facts'][field] = dict(value=next(iter(values)), evidence=facts)
        else:
            result['rejected'].append(dict(field=field, reason='conflicting_or_multiple_values', values=sorted(values)))
    return result
