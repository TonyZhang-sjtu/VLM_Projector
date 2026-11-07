"""Frozen reference-only location eligibility from report_output_v1.

Only the location-related vocabulary and decisions are retained here, so the
standard evaluator has no runtime dependency on experimental training scripts.
The prediction vocabulary lives in location.py and may be versioned separately.
Reference labels retain the historical conservative single-lesion protocol.
"""
import re

# Reference sources at promotion into the evaluation pipeline:
# experiments/report_output_v1/extract.py: 30744a1acbfe7847618222f1d7dae7c1087035cd7e83df8f2db63e569ebf0c75
# experiments/report_facts_v1/extract.py: fc85782114a1ee4939da7b12179eb101afb4e15f0bcea3a9859b31df14993a0b
# scripts/bindformer_v2_spec.py: 8c7543a4cad52e678f99c76aa69e1f395a73a5baacac9846bc4dde4926c262f4
# scripts/build_bindformer_knowledge_v2.py: eca4bc644dd87ae0b832864d0de4bd762e6c91986825de54aa7da7c53d59c157

MULTIPLE = re.compile(r'\b(?:lesions|nodules|masses|foci|multiple|numerous|bilateral|both|two|three|four|another|largest|larger|dominant|additional lesion)\b', re.I)
MISSING = re.compile(r'\b(?:no (?:mention(?:ed)?|reported)|(?:without|nor) mention|not (?:explicitly )?(?:mentioned|reported|described|specified|provided|discussed|assessed|addressed)|(?:does|do) not (?:mention|report|describe|elaborate)|no (?:(?:additional|further|explicit|detailed) )?(?:details?|information|description|comments?|observations?|specifics)|(?:lacks?|without) (?:any )?(?:mention|description|information)|unknown|indeterminate|unreported|unspecified)\b', re.I)
UNCERTAIN = re.compile(r'\b(?:may|might|could|possible|possibly|potential|likely|probable|suggests?|suggesting|suspicious)\b', re.I)
NEGATIVE = re.compile(r'\b(?:no|not|without|absence of|absent|neither|nor|lack of)\b', re.I)
ANCHOR = re.compile(r'\b(?:lesion|mass|nodule|tumou?r)\b(?![- ](?:effect|like))', re.I)

SEQUENCES = ('T1W', 'T1C', 'T2W', 'FLAIR', 'ADC')
SITES = ('frontal_lobe', 'parietal_lobe', 'temporal_lobe', 'occipital_lobe', 'cerebellum', 'thalamus', 'basal_ganglia', 'brainstem')

# (concept, field, frozen alias patterns)
_VOCABULARY = [
    ('brain', 'anatomy', ['(?<!\\w)brain(?!\\w)']),
    ('brain_parenchyma', 'anatomy', ['(?<!\\w)brain[\\s\\-‐‑–—]+parenchyma(?!\\w)', '(?<!\\w)brain[\\s\\-‐‑–—]+parenchymal(?!\\w)', '(?<!\\w)brain[\\s\\-‐‑–—]+tissue(?!\\w)', '(?<!\\w)brain[\\s\\-‐‑–—]+tissues(?!\\w)', '(?<!\\w)cerebral[\\s\\-‐‑–—]+parenchyma(?!\\w)']),
    ('frontal_lobe', 'anatomy', ['(?<!\\w)frontal[\\s\\-‐‑–—]+lobe(?!\\w)', '(?<!\\w)frontal[\\s\\-‐‑–—]+lobes(?!\\w)']),
    ('parietal_lobe', 'anatomy', ['(?<!\\w)parietal[\\s\\-‐‑–—]+lobe(?!\\w)', '(?<!\\w)parietal[\\s\\-‐‑–—]+lobes(?!\\w)']),
    ('temporal_lobe', 'anatomy', ['(?<!\\w)temporal[\\s\\-‐‑–—]+lobe(?!\\w)', '(?<!\\w)temporal[\\s\\-‐‑–—]+lobes(?!\\w)']),
    ('occipital_lobe', 'anatomy', ['(?<!\\w)occipital[\\s\\-‐‑–—]+lobe(?!\\w)', '(?<!\\w)occipital[\\s\\-‐‑–—]+lobes(?!\\w)']),
    ('insula', 'anatomy', ['(?<!\\w)insula(?!\\w)', '(?<!\\w)insular[\\s\\-‐‑–—]+lobe(?!\\w)', '(?<!\\w)insular[\\s\\-‐‑–—]+cortex(?!\\w)', '(?<!\\w)insular[\\s\\-‐‑–—]+region(?!\\w)']),
    ('thalamus', 'anatomy', ['(?<!\\w)thalamus(?!\\w)', '(?<!\\w)thalami(?!\\w)', '(?<!\\w)thalamic(?!\\w)']),
    ('basal_ganglia', 'anatomy', ['(?<!\\w)basal[\\s\\-‐‑–—]+ganglia(?!\\w)']),
    ('corpus_callosum', 'anatomy', ['(?<!\\w)corpus[\\s\\-‐‑–—]+callosum(?!\\w)', '(?<!\\w)callosal(?!\\w)']),
    ('callosal_splenium', 'anatomy', ['(?<!\\w)splenium(?!\\w)', '(?<!\\w)splenium[\\s\\-‐‑–—]+of[\\s\\-‐‑–—]+the[\\s\\-‐‑–—]+corpus[\\s\\-‐‑–—]+callosum(?!\\w)']),
    ('callosal_genu', 'anatomy', ['(?<!\\w)genu(?!\\w)', '(?<!\\w)genu[\\s\\-‐‑–—]+of[\\s\\-‐‑–—]+the[\\s\\-‐‑–—]+corpus[\\s\\-‐‑–—]+callosum(?!\\w)', '(?<!\\w)knee[\\s\\-‐‑–—]+of[\\s\\-‐‑–—]+the[\\s\\-‐‑–—]+corpus[\\s\\-‐‑–—]+callosum(?!\\w)']),
    ('callosal_body', 'anatomy', ['(?<!\\w)body[\\s\\-‐‑–—]+of[\\s\\-‐‑–—]+the[\\s\\-‐‑–—]+corpus[\\s\\-‐‑–—]+callosum(?!\\w)', '(?<!\\w)body[\\s\\-‐‑–—]+of[\\s\\-‐‑–—]+corpus[\\s\\-‐‑–—]+callosum(?!\\w)']),
    ('white_matter', 'anatomy', ['(?<!\\w)white[\\s\\-‐‑–—]+matter(?!\\w)']),
    ('gray_matter', 'anatomy', ['(?<!\\w)gray[\\s\\-‐‑–—]+matter(?!\\w)', '(?<!\\w)grey[\\s\\-‐‑–—]+matter(?!\\w)']),
    ('cortex', 'anatomy', ['(?<!\\w)cerebral[\\s\\-‐‑–—]+cortex(?!\\w)', '(?<!\\w)cortex(?!\\w)', '(?<!\\w)cortical(?!\\w)']),
    ('cerebral_hemisphere', 'anatomy', ['(?<!\\w)cerebral[\\s\\-‐‑–—]+hemisphere(?!\\w)', '(?<!\\w)cerebral[\\s\\-‐‑–—]+hemispheres(?!\\w)']),
    ('cerebellum', 'anatomy', ['(?<!\\w)cerebellum(?!\\w)', '(?<!\\w)cerebella(?!\\w)', '(?<!\\w)cerebellar(?!\\w)']),
    ('cerebellar_hemisphere', 'anatomy', ['(?<!\\w)cerebellar[\\s\\-‐‑–—]+hemisphere(?!\\w)', '(?<!\\w)cerebellar[\\s\\-‐‑–—]+hemispheres(?!\\w)']),
    ('cerebellar_vermis', 'anatomy', ['(?<!\\w)vermis(?!\\w)', '(?<!\\w)vermian(?!\\w)', '(?<!\\w)cerebellar[\\s\\-‐‑–—]+vermis(?!\\w)', '(?<!\\w)vermis[\\s\\-‐‑–—]+of[\\s\\-‐‑–—]+the[\\s\\-‐‑–—]+cerebellum(?!\\w)']),
    ('brainstem', 'anatomy', ['(?<!\\w)brainstem(?!\\w)', '(?<!\\w)brain[\\s\\-‐‑–—]+stem(?!\\w)']),
    ('midbrain', 'anatomy', ['(?<!\\w)midbrain(?!\\w)', '(?<!\\w)mesencephalon(?!\\w)']),
    ('pons', 'anatomy', ['(?<!\\w)pons(?!\\w)', '(?<!\\w)pontine(?!\\w)']),
    ('midbrain_tegmentum', 'anatomy', ['(?<!\\w)midbrain[\\s\\-‐‑–—]+tegmentum(?!\\w)', '(?<!\\w)tegmentum(?!\\w)']),
    ('cerebral_peduncle', 'anatomy', ['(?<!\\w)cerebral[\\s\\-‐‑–—]+peduncle(?!\\w)', '(?<!\\w)cerebral[\\s\\-‐‑–—]+peduncles(?!\\w)']),
    ('internal_capsule', 'anatomy', ['(?<!\\w)internal[\\s\\-‐‑–—]+capsule(?!\\w)']),
    ('external_capsule', 'anatomy', ['(?<!\\w)external[\\s\\-‐‑–—]+capsule(?!\\w)']),
    ('hippocampus', 'anatomy', ['(?<!\\w)hippocampus(?!\\w)']),
    ('hippocampal_region', 'anatomy', ['(?<!\\w)hippocampal(?!\\w)', '(?<!\\w)hippocampal[\\s\\-‐‑–—]+region(?!\\w)']),
    ('parahippocampal_gyrus', 'anatomy', ['(?<!\\w)parahippocampal[\\s\\-‐‑–—]+gyrus(?!\\w)']),
    ('precentral_gyrus', 'anatomy', ['(?<!\\w)precentral[\\s\\-‐‑–—]+gyrus(?!\\w)']),
    ('precentral_cerebellar_cistern', 'anatomy', ['(?<!\\w)precentral[\\s\\-‐‑–—]+cerebellar[\\s\\-‐‑–—]+cistern(?!\\w)']),
    ('precuneus', 'anatomy', ['(?<!\\w)precuneus(?!\\w)', '(?<!\\w)pre[\\s\\-‐‑–—]+cuneus(?!\\w)']),
    ('ventricular_system', 'anatomy', ['(?<!\\w)ventricle(?!\\w)', '(?<!\\w)ventricles(?!\\w)', '(?<!\\w)cerebral[\\s\\-‐‑–—]+ventricles(?!\\w)', '(?<!\\w)ventricular[\\s\\-‐‑–—]+system(?!\\w)']),
    ('lateral_ventricle', 'anatomy', ['(?<!\\w)lateral[\\s\\-‐‑–—]+ventricle(?!\\w)', '(?<!\\w)lateral[\\s\\-‐‑–—]+ventricles(?!\\w)']),
    ('fourth_ventricle', 'anatomy', ['(?<!\\w)fourth[\\s\\-‐‑–—]+ventricle(?!\\w)']),
    ('anterior_horn', 'anatomy', ['(?<!\\w)anterior[\\s\\-‐‑–—]+horn(?!\\w)', '(?<!\\w)anterior[\\s\\-‐‑–—]+horns(?!\\w)', '(?<!\\w)frontal[\\s\\-‐‑–—]+horn(?!\\w)', '(?<!\\w)frontal[\\s\\-‐‑–—]+horns(?!\\w)']),
    ('posterior_horn', 'anatomy', ['(?<!\\w)posterior[\\s\\-‐‑–—]+horn(?!\\w)', '(?<!\\w)posterior[\\s\\-‐‑–—]+horns(?!\\w)']),
    ('temporal_horn', 'anatomy', ['(?<!\\w)temporal[\\s\\-‐‑–—]+horn(?!\\w)', '(?<!\\w)temporal[\\s\\-‐‑–—]+horns(?!\\w)']),
    ('septum_pellucidum', 'anatomy', ['(?<!\\w)septum[\\s\\-‐‑–—]+pellucidum(?!\\w)']),
    ('cerebral_sulcus', 'anatomy', ['(?<!\\w)sulci(?!\\w)', '(?<!\\w)sulcus(?!\\w)', '(?<!\\w)sulcal(?!\\w)']),
    ('cerebral_gyrus', 'anatomy', ['(?<!\\w)gyri(?!\\w)', '(?<!\\w)gyrus(?!\\w)', '(?<!\\w)gyral(?!\\w)']),
    ('subarachnoid_cistern', 'anatomy', ['(?<!\\w)cistern(?!\\w)', '(?<!\\w)cisterns(?!\\w)', '(?<!\\w)cisternal(?!\\w)']),
    ('longitudinal_fissure', 'anatomy', ['(?<!\\w)interhemispheric[\\s\\-‐‑–—]+fissure(?!\\w)', '(?<!\\w)longitudinal[\\s\\-‐‑–—]+fissure(?!\\w)', '(?<!\\w)longitudinal[\\s\\-‐‑–—]+fissure[\\s\\-‐‑–—]+of[\\s\\-‐‑–—]+the[\\s\\-‐‑–—]+cerebrum(?!\\w)']),
    ('cerebral_fissure', 'anatomy', ['(?<!\\w)cerebral[\\s\\-‐‑–—]+fissure(?!\\w)']),
    ('sylvian_fissure', 'anatomy', ['(?<!\\w)sylvian[\\s\\-‐‑–—]+fissure(?!\\w)', '(?<!\\w)lateral[\\s\\-‐‑–—]+cerebral[\\s\\-‐‑–—]+fissure(?!\\w)']),
    ('falx', 'anatomy', ['(?<!\\w)falx(?!\\w)', '(?<!\\w)falx[\\s\\-‐‑–—]+cerebri(?!\\w)']),
    ('skull', 'anatomy', ['(?<!\\w)skull(?!\\w)', '(?<!\\w)cranial[\\s\\-‐‑–—]+bone(?!\\w)']),
    ('calvarium', 'anatomy', ['(?<!\\w)calvarium(?!\\w)', '(?<!\\w)calvaria(?!\\w)', '(?<!\\w)calvarial(?!\\w)', '(?<!\\w)cranial[\\s\\-‐‑–—]+vault(?!\\w)']),
    ('skull_base', 'anatomy', ['(?<!\\w)skull[\\s\\-‐‑–—]+base(?!\\w)', '(?<!\\w)cranial[\\s\\-‐‑–—]+base(?!\\w)', '\\bbase of (?:the )?(?:left |right )?skull\\b']),
    ('frontal_bone', 'anatomy', ['(?<!\\w)frontal[\\s\\-‐‑–—]+bone(?!\\w)', '(?<!\\w)frontal[\\s\\-‐‑–—]+bone[\\s\\-‐‑–—]+plate(?!\\w)']),
    ('parietal_bone', 'anatomy', ['(?<!\\w)parietal[\\s\\-‐‑–—]+bone(?!\\w)', '(?<!\\w)parietal[\\s\\-‐‑–—]+bone[\\s\\-‐‑–—]+plate(?!\\w)']),
    ('temporal_bone', 'anatomy', ['(?<!\\w)temporal[\\s\\-‐‑–—]+bone(?!\\w)', '(?<!\\w)temporal[\\s\\-‐‑–—]+bone[\\s\\-‐‑–—]+plate(?!\\w)']),
    ('occipital_bone', 'anatomy', ['(?<!\\w)occipital[\\s\\-‐‑–—]+bone(?!\\w)', '(?<!\\w)occipital[\\s\\-‐‑–—]+bone[\\s\\-‐‑–—]+plate(?!\\w)']),
    ('tentorium', 'anatomy', ['(?<!\\w)tentorium(?!\\w)', '(?<!\\w)cerebellar[\\s\\-‐‑–—]+tentorium(?!\\w)', '(?<!\\w)tentorium[\\s\\-‐‑–—]+cerebelli(?!\\w)']),
    ('superior_sagittal_sinus', 'anatomy', ['(?<!\\w)superior[\\s\\-‐‑–—]+sagittal[\\s\\-‐‑–—]+sinus(?!\\w)']),
    ('sagittal_sinus', 'anatomy', ['(?<!\\w)sagittal[\\s\\-‐‑–—]+sinus(?!\\w)']),
    ('cerebellopontine_angle', 'anatomy', ['(?<!\\w)cerebellopontine[\\s\\-‐‑–—]+angle(?!\\w)', '(?<!\\w)pontine[\\s\\-‐‑–—]+cerebellar[\\s\\-‐‑–—]+angle(?!\\w)']),
    ('middle_cranial_fossa', 'anatomy', ['(?<!\\w)middle[\\s\\-‐‑–—]+cranial[\\s\\-‐‑–—]+fossa(?!\\w)']),
    ('suprasellar_region', 'anatomy', ['(?<!\\w)suprasellar(?!\\w)', '(?<!\\w)suprasellar[\\s\\-‐‑–—]+region(?!\\w)']),
    ('frontal_operculum', 'anatomy', ['(?<!\\w)frontal[\\s\\-‐‑–—]+operculum(?!\\w)', '(?<!\\w)frontal[\\s\\-‐‑–—]+opercular(?!\\w)']),
    ('centrum_semiovale', 'anatomy', ['(?<!\\w)centrum[\\s\\-‐‑–—]+semiovale(?!\\w)', '(?<!\\w)center[\\s\\-‐‑–—]+semiovale(?!\\w)']),
    ('corona_radiata', 'anatomy', ['(?<!\\w)corona[\\s\\-‐‑–—]+radiata(?!\\w)']),
    ('periventricular', 'anatomy', ['(?<!\\w)periventricular(?!\\w)', '(?<!\\w)periventricular[\\s\\-‐‑–—]+region(?!\\w)']),
    ('frontal_region', 'regional_descriptor', ['(?<!\\w)frontal[\\s\\-‐‑–—]+region(?!\\w)']),
    ('parietal_region', 'regional_descriptor', ['(?<!\\w)parietal[\\s\\-‐‑–—]+region(?!\\w)']),
    ('temporal_region', 'regional_descriptor', ['(?<!\\w)temporal[\\s\\-‐‑–—]+region(?!\\w)', '(?<!\\w)temporal[\\s\\-‐‑–—]+regions(?!\\w)']),
    ('occipital_region', 'regional_descriptor', ['(?<!\\w)occipital[\\s\\-‐‑–—]+region(?!\\w)']),
    ('left', 'laterality', ['(?<!\\w)left(?!\\w)', '(?<!\\w)left[\\s\\-‐‑–—]+sided(?!\\w)']),
    ('right', 'laterality', ['(?<!\\w)right(?!\\w)', '(?<!\\w)right[\\s\\-‐‑–—]+sided(?!\\w)']),
    ('bilateral', 'laterality', ['(?<!\\w)bilateral(?!\\w)', '(?<!\\w)bilaterally(?!\\w)']),
    ('midline', 'laterality', ['(?<!\\w)midline(?!\\w)', '(?<!\\w)mid[\\s\\-‐‑–—]+line(?!\\w)']),
    ('leftward', 'displacement_direction', ['(?<!\\w)leftward(?!\\w)', '\\b(?:to|toward|towards)(?: the)? left\\b']),
    ('rightward', 'displacement_direction', ['(?<!\\w)rightward(?!\\w)', '\\b(?:to|toward|towards)(?: the)? right\\b']),
]
_FIELDS = {key: field for key, field, _ in _VOCABULARY}
_PATTERNS = {key: [re.compile(p, re.I) for p in patterns] for key, _, patterns in _VOCABULARY}

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

def _mentions(text):
    clauses = segments(text)
    found = set()
    for key, patterns in _PATTERNS.items():
        for pattern in patterns:
            found.update((key, m.start(), m.end()) for m in pattern.finditer(text))
    # Coordinated sites must not collapse to just the last named lobe.
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
            found.add((key, match.start() + part.start(), match.start() + part.end()))
    chosen = []
    for key, start, end in sorted(found, key=lambda x: (-(x[2] - x[1]), x[1], x[0])):
        if any(a <= start and end <= b and
               (_FIELDS[k] == _FIELDS[key] or
                (_FIELDS[key] == 'laterality' and _FIELDS[k] == 'displacement_direction'))
               for k, a, b in chosen):
            continue
        chosen.append((key, start, end))
    return [(key, next((a, b) for a, b in clauses if a <= start < b))
            for key, start, end in sorted(chosen, key=lambda x: (x[1], x[2], x[0]))]


def reference_location(text, sequence, text_audit_status=''):
    """Extract only explicit, unambiguous reference attributes; never predictions."""
    empty = dict(side=None, site=None)
    if (MULTIPLE.search(text) or not ANCHOR.search(text)
            or 'requires_review' in text_audit_status):
        return empty
    mentions = _mentions(text)
    lesion_sides = {key for key, (a, b) in mentions if key in ('left', 'right')
                    and ANCHOR.search(text[a:b])
                    and not re.search(r'midline|shift|displac|deviat|ventricle', text[a:b], re.I)}
    if len(lesion_sides) > 1:
        return empty
    candidates = dict(side=set(), site=set())
    for key, (a, b) in mentions:
        field = 'side' if key in ('left', 'right') else 'site' if key in SITES else None
        if field is None:
            continue
        clause = text[a:b]
        if (sequence not in SEQUENCES or MISSING.search(clause) or UNCERTAIN.search(clause)
                or not ANCHOR.search(clause) or NEGATIVE.search(clause)):
            continue
        if field == 'side' and re.search(r'midline|shift|displac|deviat|ventricle|falx', clause, re.I):
            continue
        if field == 'site':
            if re.search(r'compress|displac|shift|adjacent|abuts?|edema|oedema', clause, re.I):
                continue
            if any(_FIELDS[other] == 'anatomy' and other != key and bounds == (a, b)
                   for other, bounds in mentions):
                continue
        candidates[field].add(key)
    return {field: next(iter(values)) if len(values) == 1 else None
            for field, values in candidates.items()}
