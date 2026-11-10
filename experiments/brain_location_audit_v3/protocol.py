"""Graph-anchored explicit location extraction and partial-reference evaluation.

The same extractor is used on both sides. Text recovery is bounded to anatomical
noun phrases attached to a graph finding, not a full-report keyword bag. No
cross-sentence, cross-sequence, atlas-based or patient-based label completion.
This is report-level fact-set agreement, NOT lesion-instance matching.
"""
import re
from experiments.brain_location_audit_v1 import protocol as old

VERSION = 'brain_location_graph_v3_explicit_partial_r2'
DIMS = ('side', 'site', 'joint')
LESION = re.compile(
    r'\b(?:lesions?|masses|mass|nodules?|tumou?rs?|neoplasms?|'
    r'abnormalit(?:y|ies)|abnormal\s+signals?|signal\s+abnormalit(?:y|ies)|'
    r'(?:signal\s+)?(?:focus|foci))\b', re.I)
NON_LESION = re.compile(r'\bmass\s*[- ]?\s*effect\b|\b(?:perilesional|peritumoral|edema|oedema)\b', re.I)
SITES = dict(old.SITES, **{
    'cerebellum': r'\b(?:cerebellum|cerebellar(?!\s+tentorium))\b',
    'tentorium': r'\btentorium(?:\s+cerebelli)?\b',
    'anterior_cranial_fossa': r'\banterior\s+cranial\s+fossa\b',
    'middle_cranial_fossa': r'\bmiddle\s+cranial\s+fossa\b',
    'posterior_cranial_fossa': r'\bposterior\s+(?:cranial\s+)?fossa\b',
    'sellar_region': r'\bsellar\b',
    'parasellar_region': r'\bparasellar\b',
    'pineal_region': r'\bpineal\b',
})
SIDE = re.compile(r'\b(?:left|right|bilateral(?:ly)?|both|mid\s*-?\s*line)\b', re.I)
# A completed head ends shared-side scope: left temporal lobe, basal ganglia
# does NOT give the basal ganglia a left label. A shared head does: left frontal
# and parietal lobes. Commas also end side propagation conservatively.
HEAD = re.compile(r'\b(?:lobes?|regions?|areas?|poles?|ganglia|thalamus|thalami|insula|'
                  r'cerebellum|brainstem|pons|midbrain|medulla|hippocampus|fossa|'
                  r'ventricles?|hemispheres?|capsule|angle|callosum)\b|[,;]', re.I)
WORDS = set('''left right bilateral bilaterally both midline mid line frontal fronto
parietal parieto temporal temporo occipital occipito insula insular basal ganglia
thalamus thalami thalamic hippocampus hippocampal corpus callosum callosal brain
stem brainstem pons pontine midbrain mesencephalon medulla oblongata cerebellum
cerebellar cerebellopontine angle falx parafalcine skull base internal external
capsule centrum semiovale corona radiata suprasellar sellar parasellar pineal
ventricle ventricles cerebral hemisphere hemispheres anterior middle posterior
cranial fossa lobe lobes region regions area areas junction pole poles cortex
cortical subcortical white matter convexity parasagittal medial lateral superior
inferior central precentral postcentral gyrus gyri sulcus sulci horn horns the
of and on at in within along to side tentorium cerebelli cerebri plate calvarial
skullcap under beneath part'''.split())


def normalized_text(text):
    text = text.lower().replace('–', '-').replace('—', '-')
    text = re.sub(r'\bfronto(?=parietal|temporal|occipital)', 'fronto-', text)
    text = re.sub(r'\btemporo(?=parietal|occipital)', 'temporo-', text)
    return re.sub(r'\bparieto(?=occipital|temporal)', 'parieto-', text)


def anatomy_pairs(text):
    """Explicit side scope inside ONE anatomical phrase, with shared-head lists."""
    text = normalized_text(text)
    sites = sorted((m.start(), m.end(), site) for site, rx in SITES.items()
                   for m in re.finditer(rx, text))
    if len(sites) > 1:
        sites = [s for s in sites if s[2] != 'cerebral_hemisphere']
    sides = list(SIDE.finditer(text))

    def name(m):
        t = m.group().lower()
        return 'bilateral' if t.startswith('bilateral') or t == 'both' else 'midline' if t.startswith('mid') else t

    pairs = []
    for start, end, site in sites:
        before = [s for s in sides if s.end() <= start]
        side = None
        if before:
            last = before[-1]
            if not HEAD.search(text[last.end():start]):
                side = name(last)
                if len(before) >= 2 and {name(before[-2]), name(last)} == {'left', 'right'}:
                    between = text[before[-2].end():last.start()]
                    if re.fullmatch(r'\s*(?:and|&|/)\s*', between):
                        side = 'bilateral'
        # Postposed side is explicit only when no conflicting preceding side and
        # the entire phrase names a single region: frontal lobe on the left.
        if side is None and len(sites) == 1 and len(sides) == 1 and sides[0].start() >= end:
            if re.fullmatch(r'\s*(?:(?:lobe|region|area|located|at|on|the|is)\s*)*', text[end:sides[0].start()]):
                side = name(sides[0])
        pairs.append((side, site))
    if not sites and sides:
        ns = {name(m) for m in sides}
        if ns == {'left', 'right'}:
            ns = {'bilateral'}
        if len(ns) == 1:
            pairs.append((next(iter(ns)), None))
    return sorted(set(pairs), key=str)


def _allowed(token):
    if token in {',', '-', '/', '&', '(' , ')'}:
        return True
    words = re.findall(r'[a-z]+', normalized_text(token))
    return bool(words) and all(w in WORDS for w in words) and not any(c in token for c in '.;:')


def _phrase(tokens, start, end):
    """Expand a target within a bounded lexical anatomical noun phrase."""
    if not tokens or not (0 <= start <= end < len(tokens)):
        return None
    if not all(_allowed(t) for t in tokens[start:end+1]):
        return None
    a, b = start, end
    while a > 0 and start-a < 28 and _allowed(tokens[a-1]):
        a -= 1
    while b+1 < len(tokens) and b-end < 28 and _allowed(tokens[b+1]):
        b += 1
    while a < b and tokens[a] in {',', '-', '/', '&', 'and'}:
        a += 1
    while b > a and tokens[b] in {',', '-', '/', '&', 'and'}:
        b -= 1
    return a, b, ' '.join(tokens[a:b+1])


def extract_graph(graph):
    if not isinstance(graph, dict) or not isinstance(graph.get('entities'), dict):
        return dict(version=VERSION, facts=[], flags=['graph_unavailable'], midline_shift=[])
    entities = {str(k): v for k, v in graph['entities'].items()}
    tokens = graph.get('text', '').split()
    incoming = {}
    for k, e in entities.items():
        if e.get('label', '').lower().startswith('anatomy'):
            for rel, target in e.get('relations', []):
                if rel == 'modify':
                    incoming.setdefault(str(target), []).append(k)
    facts, flags = [], set()
    legacy = old.extract_graph(graph)
    # All lexical anatomical phrases for tightly bounded missing-edge recovery.
    phrases = {}
    for i, t in enumerate(tokens):
        if _allowed(t):
            ph = _phrase(tokens, i, i)
            if ph and anatomy_pairs(ph[2]):
                phrases[ph[:2]] = ph

    def graph_context(target):
        todo, seen = [target], set()
        while todo:
            k = todo.pop()
            if k in seen or k not in entities or not entities[k].get('label', '').lower().startswith('anatomy'):
                continue
            seen.add(k)
            todo.extend(incoming.get(k, []))
        ids = sorted(seen, key=lambda k: (entities[k].get('start_ix', 0), k))
        return ids, ' '.join(entities[k].get('tokens', '') for k in ids)

    for source, e in entities.items():
        finding = e.get('tokens', '')
        if not e.get('label', '').lower().startswith('observation') or not LESION.search(finding) or NON_LESION.search(finding):
            continue
        status = old.assertion(e.get('label', ''))
        start, end = e.get('start_ix', -1), e.get('end_ix', e.get('start_ix', -1))
        prefix = ' '.join(tokens[max(0,start-5):start]).lower()
        if re.search(r'\b(?:no|without|absence of|no evidence of)\s*(?:(?:a|an|the)\s*)?$', prefix):
            status = 'absent'
        elif re.search(r'\b(?:possible|suspected|questionable)\s*$', prefix):
            status = 'uncertain'
        candidates = []
        for rel, target in e.get('relations', []):
            target = str(target)
            if rel != 'located_at' or target not in entities:
                continue
            ids, context = graph_context(target)
            node = entities[target]
            ph = _phrase(tokens, node.get('start_ix', -1), node.get('end_ix', node.get('start_ix', -1)))
            relation_prefix = ''
            if ph:
                # 'adjacent to the ventricle' is not an intraventricular lesion.
                pre = ' '.join(tokens[max(0,ph[0]-4):ph[0]]).lower()
                if re.search(r'\b(?:compressing|compression of)\s*$', pre):
                    flags.add('excluded_compressed_anatomy')
                    continue
                if re.search(r'\b(?:adjacent|abutting)\s*$', pre):
                    relation_prefix = 'adjacent_to/'
                context = ph[2]
                ids = sorted({k for k,n in entities.items() if n.get('label','').lower().startswith('anatomy')
                              and ph[0] <= n.get('start_ix', -1) <= ph[1]}, key=lambda k: entities[k].get('start_ix',0))
            candidates.append((context, ids, ph[:2] if ph else None, 'located_at_bounded_phrase', relation_prefix))
        # Recover only explicit attributive or locative syntax in the same clause.
        # No cross-sentence pronoun resolution or general nearest-anatomy rule.
        for (a,b), ph in phrases.items():
            ok = False
            if end < a and a-end <= 40:
                gap = ' '.join(tokens[end+1:a]).lower()
                lead = ph[2].lower()
                ok = bool(re.fullmatch(r'\s*(?:(?:is|are|was|were|located|situated|seen|observed|identified|present|detected|noted|visible|centered)\s*)*(?:in|within|at|on|under|beneath|involving|occupying)?\s*', gap))
                ok &= bool(re.search(r'\b(?:in|within|at|on|under|beneath|involving|occupying)\b', gap+' '+lead[:20]))
                # Transparent signal descriptions may intervene before an
                # explicit 'located' predicate, still in this finding's clause.
                if re.search(r'\b(?:located|situated)\s*$',gap):
                    forbidden = re.search(r'[.;:]|\b(?:edema|oedema|shift|compression|ventricles?|cysts?|hemorrhage|bleeding|infarct|atrophy|without|no|possible|suspected)\b',gap)
                    ok |= not forbidden and not LESION.search(gap)
                # A stated largest member of this same lesion group. This does
                # not create lesion counts or infer unstated locations.
                if re.search(r'\b(?:with|particularly|especially)\s+the\s+(?:largest|larger)\s*(?:one\s*)?$',gap):
                    ok |= not re.search(r'[.;:]|\b(?:edema|oedema|shift|compression|without|no)\b',gap) and not LESION.search(gap)
            elif b < start and start-b <= 7:
                gap = ' '.join(tokens[b+1:start]).lower()
                ok = bool(re.fullmatch(r'\s*(?:,\s*)?(?:(?:there|is|are|an|a|the|large|largest|larger|small|enhancing|enhanced|contains|has|hosts|mass|like|-)\s*)*',gap))
            if ok:
                ids = sorted({k for k,n in entities.items() if n.get('label','').lower().startswith('anatomy')
                              and a <= n.get('start_ix',-1) <= b}, key=lambda k: entities[k].get('start_ix',0))
                candidates.append((ph[2],ids,(a,b),'explicit_locative_or_attributive_recovery',''))
        if not candidates:
            flags.add('lesion_without_recoverable_location')
        for context, ids, span, rule, relation_prefix in candidates:
            local_status=status
            if span and end<span[0] and span[0]-end<=40:
                bridge=' '.join(tokens[end+1:span[0]]).lower()
                if re.search(r'\b(?:not|never)\s*(?:(?:located|situated|present|seen|identified)\s*)?$',bridge):
                    local_status='absent';flags.add('explicit_location_negation')
                elif re.search(r'\b(?:(?:may|might|could)\s+(?:be\s+)?(?:located|situated|present|seen|identified)|(?:possibly|probably|potentially)\s*(?:located|situated)?)\s*$',bridge):
                    local_status='uncertain';flags.add('explicit_location_uncertainty')
            pairs = [(s,relation_prefix+t if t else t) for s,t in anatomy_pairs(context)]
            if relation_prefix:
                # 'adjacent to the tentorium IN the right occipital region':
                # retain the adjacency qualifier only before the explicit in-PP.
                boundary=re.search(r'\b(?:in|within)\s+(?:the\s+)?(?:left|right|bilateral)\b',context,re.I)
                if boundary:
                    pairs=[(s,relation_prefix+t if t else t) for s,t in anatomy_pairs(context[:boundary.start()])]
                    pairs+=anatomy_pairs(context[boundary.start():])
            if not pairs:
                flags.add('unmapped_anatomy')
            for side, site in pairs:
                facts.append(dict(side=side,site=site,status=local_status,finding=finding,
                                  finding_entity_id=source,anatomy_entity_ids=ids,evidence=context,
                                  evidence_token_span=span,source='cached_radgraph_xl_v3',rule=rule))
    # Deduplicate only the same finding/fact. Report-level label sets below also
    # deduplicate repeated mentions without pretending to count lesion instances.
    unique = {}
    for f in facts:
        key = (f['finding_entity_id'],f['status'],f['side'],f['site'])
        unique.setdefault(key,f)
    facts = list(unique.values())
    # Explicit predication of the SAME finding: 'lesion in the corpus callosum
    # is located at the midline'. Only one site-only and one side-only fact,
    # joined by this bounded copular construction; no arbitrary fact fusion.
    for source in {f['finding_entity_id'] for f in facts}:
        group=[f for f in facts if f['finding_entity_id']==source]
        sites=[f for f in group if f['side'] is None and f['site'] is not None]
        sides=[f for f in group if f['side'] is not None and f['site'] is None]
        if len(group)==2 and len(sites)==len(sides)==1:
            a,b=sites[0],sides[0]
            if a['status']==b['status'] and a['evidence_token_span'] and b['evidence_token_span']:
                end=a['evidence_token_span'][1];start=b['evidence_token_span'][0]
                between=' '.join(tokens[end+1:start]).lower()
                if start>end and re.fullmatch(r'(?:is|are)\s+(?:located|situated)',between):
                    combined=dict(a,side=b['side'],evidence=a['evidence']+' '+between+' '+b['evidence'],
                                  anatomy_entity_ids=sorted(set(a['anatomy_entity_ids']+b['anatomy_entity_ids'])),
                                  evidence_token_span=(a['evidence_token_span'][0],b['evidence_token_span'][1]),
                                  rule='same_finding_explicit_side_predication')
                    facts=[f for f in facts if f is not a and f is not b]+[combined]
    for key in ('side','site'):
        if any(f['status']=='present' and f[key] is None for f in facts):
            flags.add('positive_location_without_'+key)
    return dict(version=VERSION,facts=facts,flags=sorted(flags),midline_shift=legacy['midline_shift'])


def locations(extraction):
    loc = {(f['side'],f['site']) for f in extraction['facts'] if f['status']=='present'
           and (f['side'] is not None or f['site'] is not None)}
    for site in {t for _,t in loc if t is not None}:
        if ('bilateral',site) in loc or {('left',site),('right',site)} <= loc:
            loc.discard(('left',site));loc.discard(('right',site));loc.add(('bilateral',site))
    return loc


def _labels(loc, dim):
    if dim == 'joint':
        return {(s,t) for s,t in loc if s is not None and t is not None}
    vals = {p[0 if dim=='side' else 1] for p in loc if p[0 if dim=='side' else 1] is not None}
    if dim=='side' and ('bilateral' in vals or {'left','right'} <= vals):
        vals.difference_update({'left','right'});vals.add('bilateral')
    return vals


def label_sets(extraction):
    loc=locations(extraction)
    labels={d:_labels(loc,d) for d in DIMS}
    labels['eligible']={d:bool(labels[d]) for d in DIMS}
    return labels


def score_sets(reference, prediction):
    """Known-fact set F1; unmatched predictions compatible with a reference's
    explicit partial location are unassessable on that missing dimension.
    Missing predictions NEVER remove known reference facts from the denominator.
    A (None,None) reference is never a wildcard that excuses arbitrary claims.
    """
    truthloc, predloc = locations(reference), locations(prediction)
    result={}
    for dim in DIMS:
        truth=_labels(truthloc,dim)
        if not truth:
            result[dim]={'eligible':False};continue
        ignored=set()
        for s,t in predloc:
            if (s,t) in truthloc:
                continue
            if dim in ('side','joint') and t is not None and (None,t) in truthloc:
                # A known side at this same site takes precedence over a partial
                # duplicate; do not mask an explicit side contradiction.
                if not any(rs is not None and rt==t for rs,rt in truthloc):
                    ignored.add((s,t))
            if dim in ('site','joint') and s is not None and (s,None) in truthloc:
                if (s,t) not in truthloc:
                    ignored.add((s,t))
        pred=_labels(predloc-ignored,dim)
        tp,fp,fn=len(truth&pred),len(pred-truth),len(truth-pred)
        result[dim]=dict(eligible=True,exact=int(truth==pred),tp=tp,fp=fp,fn=fn,
                         precision=tp/(tp+fp) if tp+fp else 0.,recall=tp/(tp+fn),
                         f1=2*tp/(2*tp+fp+fn),missing=not bool(pred),
                         ignored_prediction_locations=sorted(ignored,key=str))
    return result
