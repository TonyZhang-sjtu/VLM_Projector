"""Normalize cached RadGraph-XL relations into reviewable location candidates.

This is an adaptation/diagnostic protocol, not a new validated metric or a new
clinical extractor. Reference and prediction graphs use exactly the same code.
No text is sent to a service and no model is loaded. Graph mistakes remain
possible: candidates must not be promoted to reference gold without review.
"""
import re

VERSION = 'brain_location_graph_audit_v1'
LESION = re.compile(r'\b(?:lesions?|masses|mass|nodules?|tumou?rs?|neoplasms?|'
                    r'abnormal signal (?:focus|foci)|signal abnormalit(?:y|ies))\b', re.I)
NON_LESION = re.compile(r'\bmass\s*[- ]?\s*effect\b|\b(?:perilesional|peritumoral)\b', re.I)
# Explicit aliases only. In particular, hippocampus is not silently converted
# to temporal lobe, and brainstem is not silently assigned a midline side.
SITES = {
    'frontal_lobe': r'\b(?:frontal|fronto)(?!\s+(?:horn|bone))\b',
    'parietal_lobe': r'\b(?:parietal|parieto)(?!\s+bone)\b',
    'temporal_lobe': r'\b(?:temporal|temporo)(?!\s+(?:horn|bone))\b',
    'occipital_lobe': r'\b(?:occipital|occipito)(?!\s+bone)\b',
    'insula': r'\b(?:insula|insular)\b',
    'basal_ganglia': r'\bbasal\s+ganglia\b',
    'thalamus': r'\b(?:thalamus|thalami|thalamic)\b',
    'hippocampus': r'\b(?:hippocampus|hippocampal)\b',
    'corpus_callosum': r'\b(?:corpus\s+callosum|callosal)\b',
    'brainstem': r'\bbrain\s*stem\b',
    'pons': r'\b(?:pons|pontine)\b',
    'midbrain': r'\b(?:midbrain|mesencephalon)\b',
    'medulla': r'\bmedulla(?:\s+oblongata)?\b',
    'cerebellum': r'\b(?:cerebellum|cerebellar)\b',
    'cerebellopontine_angle': r'\bcerebellopontine\s+angle\b',
    'falx': r'\b(?:falx|parafalcine)\b',
    'skull_base': r'\b(?:skull\s+base|base\s+of\s+(?:the\s+)?skull)\b',
    'internal_capsule': r'\binternal\s+capsule\b',
    'external_capsule': r'\bexternal\s+capsule\b',
    'centrum_semiovale': r'\bcentrum\s+semiovale\b',
    'corona_radiata': r'\bcorona\s+radiata\b',
    'suprasellar_region': r'\bsuprasellar\b',
    'ventricle': r'\bventricles?\b',
    'cerebral_hemisphere': r'\bcerebral\s+hemispheres?\b',
}
COARSE = {'pons': 'brainstem', 'midbrain': 'brainstem', 'medulla': 'brainstem'}


def normalize_anatomy(text):
    text = text.lower().replace('–', '-').replace('—', '-')
    text = re.sub(r'\bfronto(?=parietal|temporal|occipital)', 'fronto-', text)
    text = re.sub(r'\btemporo(?=parietal|occipital)', 'temporo-', text)
    text = re.sub(r'\bparieto(?=occipital|temporal)', 'parieto-', text)
    sites = sorted(k for k, pattern in SITES.items() if re.search(pattern, text))
    if len(sites) > 1 and 'cerebral_hemisphere' in sites:
        sites.remove('cerebral_hemisphere')
    left, right = bool(re.search(r'\bleft\b', text)), bool(re.search(r'\bright\b', text))
    bilateral = bool(re.search(r'\b(?:bilateral(?:ly)?|both)\b', text))
    sides = ['bilateral'] if bilateral or (left and right) else ['left'] if left else ['right'] if right else []
    if re.search(r'\bmid\s*[- ]?\s*line\b', text):
        sides.append('midline')
    return {'sites': sites, 'sides': sides}


def assertion(label):
    label = label.lower()
    if 'uncertain' in label:
        return 'uncertain'
    if 'absent' in label:
        return 'absent'
    return 'present' if 'present' in label else 'unknown'


def extract_graph(graph):
    """Return explicit located_at relations, source IDs, and token evidence.

    Only incoming Anatomy::modify edges are followed. Connecting all anatomy
    nodes as an undirected component can accidentally borrow a different site's
    side, so disconnected modifiers are deliberately left for human review.
    """
    if not isinstance(graph, dict) or not isinstance(graph.get('entities'), dict):
        return {'version': VERSION, 'facts': [], 'midline_shift': [], 'flags': ['graph_unavailable']}
    entities = graph['entities']
    incoming = {}
    for source, e in entities.items():
        for relation, target in e.get('relations', []):
            if relation == 'modify' and e.get('label', '').lower().startswith('anatomy'):
                incoming.setdefault(str(target), []).append(source)

    def anatomy_context(target):
        seen, todo = set(), [target]
        while todo:
            node = todo.pop()
            if node in seen or node not in entities:
                continue
            if not entities[node].get('label', '').lower().startswith('anatomy'):
                continue
            seen.add(node)
            todo.extend(incoming.get(node, []))
        ids = sorted(seen, key=lambda k: (entities[k].get('start_ix', 0), k))
        return ids, ' '.join(entities[k].get('tokens', '') for k in ids)

    facts, shifts, flags = [], [], set()
    for source, e in entities.items():
        if not e.get('label', '').lower().startswith('observation'):
            continue
        finding = e.get('tokens', '')
        status = assertion(e.get('label', ''))
        targets = [str(t) for rel, t in e.get('relations', []) if rel == 'located_at']
        # Midline displacement is a different finding from a midline lesion.
        for target in targets:
            ids, context = anatomy_context(target)
            if re.search(r'\b(?:shift|deviation|displacement)\b', finding, re.I) and re.search(r'\bmidline\b', context, re.I):
                direction = ('left' if re.search(r'\bleft(?:ward)?\b', finding, re.I) else
                             'right' if re.search(r'\bright(?:ward)?\b', finding, re.I) else None)
                shifts.append({'status': status, 'direction': direction, 'finding': finding,
                               'finding_entity_id': source, 'anatomy_entity_ids': ids})
        if not LESION.search(finding) or NON_LESION.search(finding):
            continue
        if not targets:
            flags.add('lesion_without_explicit_located_at')
        for target in targets:
            ids, context = anatomy_context(target)
            normalized = normalize_anatomy(context)
            if (len(normalized['sites']) > 1 and re.search(r'\bleft\b', context, re.I)
                    and re.search(r'\bright\b', context, re.I)
                    and not re.search(r'\b(?:both|bilateral)\b', context, re.I)):
                # A malformed/merged graph must not invent a Cartesian product.
                normalized['sides'] = []
                flags.add('ambiguous_side_site_binding')
            if not normalized['sites'] and not normalized['sides']:
                flags.add('unmapped_anatomy')
            if len(normalized['sides']) > 1:
                flags.add('multiple_side_states_one_relation')
            for site in normalized['sites'] or [None]:
                facts.append({'side': normalized['sides'][0] if len(normalized['sides']) == 1 else None,
                              'site': site, 'status': status, 'finding': finding,
                              'finding_entity_id': source, 'anatomy_entity_ids': ids,
                              'evidence': context, 'source': 'cached_radgraph_xl'})
    if any(f['status'] == 'uncertain' for f in facts):
        flags.add('uncertain_location_fact')
    if any(f['status'] == 'present' and f['side'] is None for f in facts):
        flags.add('positive_location_without_side')
    if any(f['status'] == 'present' and f['site'] is None for f in facts):
        flags.add('positive_location_without_site')
    return {'version': VERSION, 'facts': facts, 'midline_shift': shifts, 'flags': sorted(flags)}


def label_sets(extraction, coarse=False):
    """Deduplicate location facts, not lesion instances or lesion counts.

    Bilateral and explicit left+right at the SAME site are equivalent. A left
    frontal and right temporal lesion must not become bilateral at both sites.
    Unknown sides remain unknown; a site can be scored without a side.
    """
    positive = [f for f in extraction['facts'] if f['status'] == 'present']
    locations = set()
    for f in positive:
        site = COARSE.get(f['site'], f['site']) if coarse else f['site']
        if f['side'] is not None or site is not None:
            locations.add((f['side'], site))
    for site in {p[1] for p in locations}:
        if site is not None and ('left', site) in locations and ('right', site) in locations:
            locations.difference_update({('left', site), ('right', site)})
            locations.add(('bilateral', site))
        if ('bilateral', site) in locations:
            locations.difference_update({('left', site), ('right', site)})
    sides = {side for side, _ in locations if side is not None}
    if 'bilateral' in sides or {'left', 'right'} <= sides:
        sides.difference_update({'left', 'right'})
        sides.add('bilateral')
    sites = {site for _, site in locations if site is not None}
    # Partial information is represented in the joint output as well; it cannot
    # match a complete pair and cannot disappear from precision penalties.
    joint = {(side or 'unspecified', site or 'unspecified') for side, site in locations}
    eligible = {'side': bool(locations) and all(s is not None for s, _ in locations),
                'site': bool(locations) and all(t is not None for _, t in locations),
                'joint': bool(locations) and all(s is not None and t is not None for s, t in locations)}
    return {'side': sides, 'site': sites, 'joint': joint, 'eligible': eligible}


def score_sets(reference, prediction):
    """Standard exact match and set P/R/F1 on reference-defined denominators.

    This is provisional agreement with automatic labels, never image-grounded
    accuracy. Empty/negated/uncertain predictions do not remove eligible refs.
    """
    truth, pred = label_sets(reference), label_sets(prediction)
    result = {}
    for field in ('side', 'site', 'joint'):
        if not truth['eligible'][field]:
            result[field] = {'eligible': False}
            continue
        t, p = truth[field], pred[field]
        tp, fp, fn = len(t & p), len(p - t), len(t - p)
        result[field] = {'eligible': True, 'exact': int(t == p), 'tp': tp, 'fp': fp, 'fn': fn,
                         'precision': tp / len(p) if p else 0.0, 'recall': tp / len(t),
                         'f1': 2 * tp / (len(t) + len(p)), 'missing': not bool(p)}
    return result
