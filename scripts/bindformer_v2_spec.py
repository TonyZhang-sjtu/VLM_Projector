"""Corpus-reviewed v2 vocabulary specification; project-authored short descriptions.

These are research vocabulary decisions, not patient labels or clinical certification.
Every non-control entry must have TRAIN evidence before entering the encoded bank.
Unresolved CUI identities use stable local IDs instead of a guessed mapping.
"""
import re

ROLES = ['finding', 'location_laterality', 'signal', 'morphology_extent']
SEQUENCES = ['T1W', 'T1C', 'T2W', 'FLAIR', 'ADC']


def specs():
    result = []
    def add(role, field, data):
        for line in data.strip().splitlines():
            key, name, aliases, description, cui = line.split('|')
            aliases = aliases.split(';')
            result.append(dict(key=key, concept_id='bindformer:' + key, role=role, field=field,
                name=name, aliases=aliases, short_definition=description, preferred_cui=cui or None,
                search_terms=list(dict.fromkeys([name] + aliases)), sequence_scope=SEQUENCES.copy(),
                definition_edit_method='project_authored_task_description',
                review_status='research_vocabulary_reviewed_not_clinically_certified',
                inclusion='corpus_observed', parent_key=None, note=''))
    add('finding', 'observation', '''
lesion|Brain lesion|lesion;lesions;focus;foci|A localized abnormal area in brain tissue, without specifying its cause or disease type.|C0221505
mass|Intracranial mass|mass;masses|A space-occupying abnormal structure within the cranium; this term alone does not establish a tumor diagnosis.|
abnormal_signal|Abnormal MRI signal|abnormal signal;abnormal signals;abnormal signal intensity|MRI signal described as abnormal relative to the expected appearance for the sequence and tissue.|
edema|Brain edema|edema;oedema;edematous|Excess fluid accumulation in brain tissue, described separately from its suspected cause.|C0006114
hemorrhage|Intracranial hemorrhage|hemorrhage;hemorrhagic;bleeding|Bleeding within the cranium; its presence and certainty must be determined from the report context.|C0151699
necrosis|Tissue necrosis|necrosis;necrotic|Death of cells or tissue, distinct from a tissue specimen label or an inferred tumor diagnosis.|C0027540
cystic_change|Cystic appearance|cystic;cystic change;cystic changes|An imaging appearance containing a cavity or cyst-like component, without asserting its tissue contents.|
dural_tail|Dural tail sign|dural tail;dural tail sign;meningeal tail;meningeal tail sign|A tail-like extension of enhancement along adjacent dura; it is an imaging sign rather than a specific diagnosis.|
''')
    add('finding', 'structural_effect', '''
mass_effect|Mass effect|mass effect;mass-effect;pressure effect|Pressure or displacement of adjacent structures caused by a space-occupying process.|C4086564
midline_shift|Brain midline shift|midline shift;midline displacement;shift of midline structures;displacement of midline structures;deviation of midline structures|Displacement of brain midline structures away from their expected central position.|C4086580
ventricular_compression|Cerebral ventricular compression|ventricular compression;compression of the ventricle;compression of the ventricles|Narrowing or deformation of a cerebral ventricle due to pressure from adjacent structures.|
ventricular_dilatation|Cerebral ventricular enlargement|ventricular dilatation;ventricular dilation;ventriculomegaly;ventricular enlargement|Enlargement of the cerebral ventricular spaces; this does not by itself specify hydrocephalus.|C1531647
sulcal_effacement|Sulcal effacement|sulcal effacement;effacement of sulci;effacement of the sulci|Reduction or loss of the visible spaces of cerebral sulci on imaging.|C5442488
''')
    add('location_laterality', 'anatomy', '''
brain|Brain|brain|The intracranial part of the central nervous system, providing a whole-brain anatomical reference.|
brain_parenchyma|Brain parenchyma|brain parenchyma;brain parenchymal;brain tissue;brain tissues;cerebral parenchyma|The functional tissue of the brain, as distinct from ventricular fluid spaces and surrounding coverings.|
frontal_lobe|Frontal lobe|frontal lobe;frontal lobes|The anterior lobe of a cerebral hemisphere, in front of the central sulcus; distinct from the frontal bone.|C0016733
parietal_lobe|Parietal lobe|parietal lobe;parietal lobes|The cerebral lobe behind the central sulcus and above the temporal lobe; distinct from the parietal bone.|C0030560
temporal_lobe|Temporal lobe|temporal lobe;temporal lobes|The lower lateral lobe of a cerebral hemisphere, below the lateral fissure; distinct from the temporal bone.|C0039485
occipital_lobe|Occipital lobe|occipital lobe;occipital lobes|The posterior lobe of a cerebral hemisphere; distinct from the occipital bone.|C0028785
insula|Insula|insula;insular lobe;insular cortex;insular region|The cerebral cortical region deep within the lateral fissure, covered by adjacent opercula.|C0021640
thalamus|Thalamus|thalamus;thalami;thalamic|A paired deep gray matter structure along the sides of the third ventricle.|C0039729
basal_ganglia|Basal ganglia|basal ganglia|A group of deep subcortical gray matter nuclei in the cerebral hemispheres.|C0004781
corpus_callosum|Corpus callosum|corpus callosum;callosal|The major white matter commissure connecting the two cerebral hemispheres.|C0010090
callosal_splenium|Splenium of corpus callosum|splenium;splenium of the corpus callosum|The posterior thickened part of the corpus callosum.|
callosal_genu|Genu of corpus callosum|genu;genu of the corpus callosum;knee of the corpus callosum|The anterior curved portion of the corpus callosum; distinct from the genu of the internal capsule.|
callosal_body|Body of corpus callosum|body of the corpus callosum;body of corpus callosum|The central elongated portion of the corpus callosum between its anterior and posterior parts.|
white_matter|Cerebral white matter|white matter|Brain tissue composed predominantly of nerve fiber pathways rather than neuronal cell bodies.|C0682708
gray_matter|Gray matter|gray matter;grey matter|Central nervous system tissue rich in neuronal cell bodies; it includes cortex and deep gray nuclei.|C0018220
cortex|Cerebral cortex|cerebral cortex;cortex;cortical|The folded layer of gray matter covering the cerebral hemispheres.|C0007776
cerebral_hemisphere|Cerebral hemisphere|cerebral hemisphere;cerebral hemispheres|One of the two major halves of the cerebrum, distinguished from a cerebellar hemisphere.|C0228174
cerebellum|Cerebellum|cerebellum;cerebella;cerebellar|The posterior fossa brain structure behind the brainstem; distinct from the adjacent cerebellopontine angle.|C0007765
cerebellar_hemisphere|Cerebellar hemisphere|cerebellar hemisphere;cerebellar hemispheres|One of the lateral portions of the cerebellum on either side of the vermis.|
cerebellar_vermis|Cerebellar vermis|vermis;vermian;cerebellar vermis;vermis of the cerebellum|The midline portion of the cerebellum between its two hemispheres.|
brainstem|Brainstem|brainstem;brain stem|The brain region comprising the midbrain, pons, and medulla, connecting the forebrain with the spinal cord.|C0006121
midbrain|Midbrain|midbrain;mesencephalon|The upper part of the brainstem between the pons and diencephalon.|C0025462
pons|Pons|pons;pontine|The portion of the brainstem between the midbrain and medulla, anterior to the cerebellum.|C0032639
midbrain_tegmentum|Midbrain tegmentum|midbrain tegmentum;tegmentum|The tegmental portion of the midbrain; anatomical context is required for the generic word tegmentum.|
cerebral_peduncle|Cerebral peduncle|cerebral peduncle;cerebral peduncles|A paired midbrain structure containing major pathways connecting the cerebrum with lower brain regions.|
internal_capsule|Internal capsule|internal capsule|A major deep cerebral white matter pathway between the thalamus or caudate and the lentiform nucleus.|C0152341
external_capsule|External capsule|external capsule|The thin white matter layer between the putamen and claustrum, distinct from the internal capsule.|
hippocampus|Hippocampus|hippocampus|A medial temporal lobe structure; distinct from the adjacent parahippocampal gyrus.|
hippocampal_region|Hippocampal region in MRI|hippocampal;hippocampal region|The region described in relation to the hippocampus, without asserting a specific cytoarchitectural subdivision.|
parahippocampal_gyrus|Parahippocampal gyrus|parahippocampal gyrus|The medial temporal cortical gyrus adjacent to the hippocampal formation.|
precentral_gyrus|Precentral gyrus|precentral gyrus|The cerebral gyrus immediately anterior to the central sulcus.|
precentral_cerebellar_cistern|Precentral cerebellar cistern|precentral cerebellar cistern|A named subarachnoid cistern in the precentral cerebellar region, distinct from the cerebral precentral gyrus.|
precuneus|Precuneus|precuneus;pre-cuneus|A medial parietal cortical region, distinct from the occipital cuneus.|
ventricular_system|Cerebral ventricular system|ventricle;ventricles;cerebral ventricles;ventricular system|The connected cerebrospinal fluid spaces within the brain, without selecting a particular ventricle.|
lateral_ventricle|Lateral ventricle|lateral ventricle;lateral ventricles|One of the paired cerebrospinal fluid cavities within the cerebral hemispheres.|C0152279
fourth_ventricle|Fourth ventricle|fourth ventricle|The cerebrospinal fluid cavity between the brainstem and cerebellum.|C0149556
anterior_horn|Anterior horn of lateral ventricle|anterior horn;anterior horns;frontal horn;frontal horns|The anterior or frontal extension of a lateral ventricle.|
posterior_horn|Posterior horn of lateral ventricle|posterior horn;posterior horns|The posterior or occipital extension of a lateral ventricle.|
temporal_horn|Temporal horn of lateral ventricle|temporal horn;temporal horns|The inferior extension of a lateral ventricle into the temporal lobe.|
septum_pellucidum|Septum pellucidum|septum pellucidum|The thin partition separating the anterior portions of the lateral ventricles.|
cerebral_sulcus|Cerebral sulcus|sulci;sulcus;sulcal|A groove between adjacent cerebral gyri; the anatomical structure is distinct from sulcal effacement.|
cerebral_gyrus|Cerebral gyrus|gyri;gyrus;gyral|A ridge or fold on the cerebral surface between sulci.|
subarachnoid_cistern|Subarachnoid cistern|cistern;cisterns;cisternal|An expanded cerebrospinal fluid space within the subarachnoid space around the brain.|
longitudinal_fissure|Longitudinal cerebral fissure|interhemispheric fissure;longitudinal fissure;longitudinal fissure of the cerebrum|The deep midline fissure separating the two cerebral hemispheres.|
cerebral_fissure|Cerebral fissure|cerebral fissure|A deep cleft on the cerebral surface when the report does not specify a particular fissure.|
sylvian_fissure|Lateral cerebral fissure|sylvian fissure;lateral cerebral fissure|The fissure separating the temporal lobe from the frontal and parietal lobes.|
falx|Falx cerebri|falx;falx cerebri|The dural fold within the longitudinal fissure between the cerebral hemispheres.|
skull|Skull|skull;cranial bone|The bony framework of the head, including the cranial vault and base.|
calvarium|Calvarium|calvarium;calvaria;calvarial;cranial vault|The upper part of the skull forming the roof and sides of the cranial cavity.|
skull_base|Skull base|skull base;cranial base|The bony floor of the cranial cavity.|
frontal_bone|Frontal bone|frontal bone;frontal bone plate|The skull bone forming the forehead and part of the anterior cranial floor, distinct from the frontal lobe.|
parietal_bone|Parietal bone|parietal bone;parietal bone plate|A paired skull bone forming part of the cranial roof and side, distinct from the parietal lobe.|
temporal_bone|Temporal bone|temporal bone;temporal bone plate|A skull bone at the side and base of the cranium, distinct from the temporal lobe.|
occipital_bone|Occipital bone|occipital bone;occipital bone plate|The skull bone forming the back and part of the base of the cranium, distinct from the occipital lobe.|
tentorium|Tentorium cerebelli|tentorium;cerebellar tentorium;tentorium cerebelli|The dural fold separating the cerebellum from the overlying posterior cerebrum.|
superior_sagittal_sinus|Superior sagittal sinus|superior sagittal sinus|The dural venous sinus along the superior margin of the falx cerebri.|
sagittal_sinus|Sagittal sinus|sagittal sinus|A dural venous sinus along the falx cerebri, without selecting the superior or inferior component.|
cerebellopontine_angle|Cerebellopontine angle|cerebellopontine angle;pontine-cerebellar angle|The region between the pons and cerebellum, distinct from cerebellar tissue itself.|
middle_cranial_fossa|Middle cranial fossa|middle cranial fossa|The middle depression of the skull base supporting the temporal lobes.|
suprasellar_region|Suprasellar region|suprasellar;suprasellar region|The anatomical region above the sella turcica.|
frontal_operculum|Frontal operculum|frontal operculum;frontal-opercular|The frontal cortical portion overlying the insula.|
centrum_semiovale|Centrum semiovale|centrum semiovale;center semiovale|The broad region of cerebral white matter superior to the lateral ventricles.|
corona_radiata|Corona radiata|corona radiata|The fan-shaped cerebral white matter pathways continuous with the internal capsule.|
periventricular|Periventricular region|periventricular;periventricular region|The region surrounding the cerebral ventricles.|C4331000
''')
    add('location_laterality', 'regional_descriptor', '''
frontal_region|Frontal region in MRI context|frontal region|A location described as frontal; retain the referenced anatomy instead of assuming that the phrase always denotes the frontal lobe.|
parietal_region|Parietal region in MRI context|parietal region|A location described as parietal; retain the referenced anatomy instead of conflating the parietal lobe and parietal bone.|
temporal_region|Temporal region in MRI context|temporal region;temporal regions|A location described as temporal; retain the referenced anatomy instead of conflating the temporal lobe and temporal bone.|
occipital_region|Occipital region in MRI context|occipital region|A location described as occipital; retain the referenced anatomy instead of conflating the occipital lobe and occipital bone.|
''')
    add('location_laterality', 'laterality', '''
left|Left side|left;left-sided|Located on the patient's left side, distinguished from displacement toward the left.|C0205091
right|Right side|right;right-sided|Located on the patient's right side, distinguished from displacement toward the right.|C0205090
bilateral|Bilateral|bilateral;bilaterally|Involving both sides of a paired anatomical structure or the body.|C0238767
midline|Midline|midline;mid-line|The central line or plane dividing the brain into left and right sides.|C0549183
''')
    add('location_laterality', 'relative_location', '''
perilesional|Perilesional region|perilesional;peri-lesional|The region surrounding a lesion, with the lesion retained as the reference object.|
peritumoral|Peritumoral region|peritumoral|The region around a structure described as a tumor; the term does not independently establish tumor identity.|
peripheral|Peripheral portion|peripheral;periphery;rim;edges|The outer portion or boundary of the referenced structure.|
central|Central portion|central;center;core|The inner or central portion of the referenced structure.|
''')
    add('location_laterality', 'displacement_direction', '''
leftward|Leftward displacement|leftward|Displacement toward the patient's left; this is not the side on which a lesion is located.|
rightward|Rightward displacement|rightward|Displacement toward the patient's right; this is not the side on which a lesion is located.|
''')
    add('signal', 'intensity', '''
hyperintense|Hyperintense MRI signal|hyperintense;hyperintensity;hyper-intense;high signal;high signal intensity;high signal intensities;higher signal|MRI signal higher than the stated or implied reference tissue on the specified sequence.|C5667489
hypointense|Hypointense MRI signal|hypointense;hypointensity;hypo-intense;hypo-signal;low signal;low signal intensity;low signal intensities;low-signal;low-intense;low-intensity|MRI signal lower than the stated or implied reference tissue on the specified sequence.|C5667487
isointense|Isointense MRI signal|isointense;isointensity;iso-intense;iso-intensity;iso-signal;isotense;isointense signal|MRI signal similar to the stated or implied reference tissue on the specified sequence.|C1517605
mixed_signal|Mixed MRI signal intensity|mixed signal;mixed signals;mixed signal intensity;mixed signal intensities;mixed high signal;mixed low signal;mixed high and low signal;mixed high and low signal intensities;iso to low signal;iso-low;hypo-isointense|Coexisting signal intensities within the referenced area; the component intensities and sequence remain separate context.|
''')
    add('signal', 'uniformity', '''
homogeneous|Homogeneous signal or enhancement|homogeneous;uniform;homogeneity|A similar or uniform appearance across the referenced signal or enhancement pattern.|C1881065
heterogeneous|Heterogeneous signal or enhancement|heterogeneous;inhomogeneous;heterogeneity;heterogenous;inhomogeneously;heterogeneously;uneven;non-uniform;nonuniform;non-homogeneous|A nonuniform appearance across the referenced signal or enhancement pattern.|C0019409
''')
    add('signal', 'enhancement', '''
enhancement|MRI contrast enhancement|enhancement;enhancing;enhances;enhanced|Increased conspicuity or signal after contrast administration on the relevant post-contrast MRI sequence.|
nonenhancing|Nonenhancing appearance|nonenhancing;non-enhancing;non-enhancement;non-enhanced|An explicitly described absence of contrast enhancement in the referenced structure.|
ring_enhancement|Ring-like contrast enhancement|ring-like enhancement;ring enhancement;ring-enhancement;ring-enhancing;ring-shaped enhancement;ring-enhanced|Contrast enhancement arranged around a central portion as a ring or rim.|
nodular_enhancement|Nodular contrast enhancement|nodular enhancement;nodular enhancing|Contrast enhancement with a nodular configuration.|
garland_enhancement|Garland-like enhancement pattern|garland-like;garland;crown-like;flower-ring-like|A wreath-like or garland-like pattern described in the enhancement context; retained separately until equivalence to a ring pattern is reviewed.|
''')
    add('signal', 'diffusion', '''
restricted_diffusion|Restricted diffusion appearance|restricted diffusion;diffusion restriction|Reduced diffusion of water in tissue, reflected by reduced apparent diffusion coefficient values in the relevant imaging context.|
''')
    add('morphology_extent', 'shape', '''
punctate|Punctate appearance|punctate;punctulate;dot-like;spot-like;spotty;speckled|An appearance composed of small dots or points.|C0205330
patchy|Patchy appearance|patchy;patch-like;patches;patch;flake-like;flaky|An appearance in irregular patches rather than a uniform continuous region.|C0205413
nodular|Nodular appearance|nodular;nodule;nodules;nodule-like|An appearance as a rounded or nodular focus; morphology alone does not establish a diagnosis.|C0205297
irregular|Irregular shape|irregular;irregularly|A shape lacking a regular contour or symmetry.|C0205271
round|Round appearance|round;rounded;roundish;round-like;quasi-circular;circular|An approximately circular or rounded shape.|C0332490
mass_like|Mass-like appearance|mass-like|An appearance resembling a space-occupying mass; retained as morphology rather than a tumor diagnosis.|
linear|Linear or band-like appearance|strip-like;band-like;band|An elongated narrow strip or band-shaped appearance.|
plaque_like|Plaque-like appearance|plaque-like;plaques|A flat or plate-like appearance, kept distinct from a general patchy distribution.|
''')
    add('morphology_extent', 'margin', '''
well_defined|Well-defined margin|well-defined;well-demarcated;clear boundaries;clear boundary;distinct boundaries;distinct boundary;clear borders|A boundary that is clearly delineated from surrounding tissue.|C0442825
ill_defined|Ill-defined margin|ill-defined;poorly-defined;poorly defined;unclear boundaries;unclear boundary;indistinct boundaries;indistinct boundary;indistinct border;indistinct borders;blurred boundary|A boundary that is not clearly delineated from surrounding tissue.|
''')
    add('morphology_extent', 'distribution', '''
focal|Focal distribution|focal;localized;localised|Limited to a specific region rather than widely distributed.|C0392752
extensive|Extensive involvement|extensive;widespread;large-scale|Involvement over a large spatial extent; extent is distinct from lesion count.|C0205231
scattered|Scattered distribution|scattered;sporadic|Separated foci distributed across a region, rather than one contiguous focus.|
''')
    add('morphology_extent', 'composition', '''
solid|Solid component|solid;solid component;solid components;solid lesion|A component described as solid rather than cystic on imaging.|
cystic_solid|Mixed cystic and solid appearance|cystic-solid|Coexisting cystic and solid components within the referenced structure.|
''')
    add('morphology_extent', 'multiplicity', '''
multiple|Multiple foci|multiple;numerous|More than one focus or lesion; exact counts are stored as separate numeric fields.|C0439064
''')
    add('morphology_extent', 'degree', '''
mild|Mild degree|mild;mildly;slight;slightly;minor;subtle;subtly|A low degree of the referenced finding or imaging attribute; retain the modified attribute.|
moderate|Moderate degree|moderate|An intermediate degree of the referenced finding or imaging attribute.|
marked|Marked degree|marked;markedly;significant;significantly;substantial;conspicuous;prominent|A pronounced degree of the referenced finding or imaging attribute, not statistical significance.|
''')
    by_key = {item['key']: item for item in result}
    for key in ['enhancement', 'nonenhancing', 'ring_enhancement', 'nodular_enhancement', 'garland_enhancement', 'dural_tail']:
        by_key[key]['sequence_scope'] = ['T1C']
    by_key['restricted_diffusion']['sequence_scope'] = ['ADC']
    by_key['mixed_signal']['aliases'] += ['mixed high signals', 'mixed low signals', 'mixed high and low signals']
    by_key['leftward']['extra_patterns'] = [r'\b(?:to|toward|towards)(?: the)? left\b']
    by_key['rightward']['extra_patterns'] = [r'\b(?:to|toward|towards)(?: the)? right\b']
    # Full phrase matching is the default; these are bounded relation patterns.
    by_key['midline_shift']['extra_patterns'] = [r'\b(?:shift|displacement|deviation)\s+(?:(?:of|in)\s+)?(?:the\s+)?mid[- ]?line(?: structures?)?\b', r'\bmid[- ]?line(?: structures?)?\s+(?:is |are )?(?:shifted|displaced|deviated)\b']
    by_key['ventricular_compression']['extra_patterns'] = [r'\b(?:compress\w*|narrow\w*)\s+of\s+(?:the\s+)?(?:(?:left|right|lateral|fourth)\s+){0,3}ventricles?\b', r'\b(?:(?:left|right|lateral|fourth)\s+){0,3}ventricles?\s+(?:is |are )?(?:(?:mildly|slightly|markedly)\s+)?(?:compressed|narrowed)\b']
    by_key['ventricular_dilatation']['extra_patterns'] = [r'\b(?:dilat\w*|enlarg\w*)\s+of\s+(?:the\s+)?(?:(?:left|right|lateral)\s+){0,3}ventricles?\b', r'\b(?:(?:left|right|lateral)\s+){0,3}ventricles?\s+(?:is |are )?(?:dilated|enlarged)\b']
    by_key['sulcal_effacement']['extra_patterns'] = [r'\b(?:effacement|effaced)\b[^.;]{0,45}\bsulci\b', r'\bsulci\b[^.;]{0,45}\befface\w*\b']
    by_key['ventricular_compression']['extra_patterns'] += [r'\b(?:compressed|narrowed)\s+(?:(?:left|right|bilateral|lateral|fourth)\s+){0,4}ventricles?\b',
        r'\bcompression\s+(?:of|on)\s+(?:the )?(?:anterior|posterior|frontal|temporal) horns? of (?:the )?(?:(?:left|right|bilateral|lateral)\s+){0,4}ventricles?\b']
    by_key['ill_defined']['extra_patterns'] = [r'\b(?:boundar(?:y|ies)|borders?|margins?|edges?)\s+(?:are|is|remain|appears?)\s+(?:ill[- ]defined|poorly[- ]defined|unclear|indistinct|blurred)\b']
    by_key['well_defined']['extra_patterns'] = [r'\b(?:boundar(?:y|ies)|borders?|margins?|edges?)\s+(?:are|is|remain|appears?)\s+(?:well[- ]defined|well[- ]demarcated|clear|distinct)\b']
    by_key['skull_base']['extra_patterns'] = [r'\bbase of (?:the )?(?:left |right )?skull\b']
    parents = {
        'callosal_splenium':'corpus_callosum', 'callosal_genu':'corpus_callosum', 'callosal_body':'corpus_callosum',
        'cerebellar_hemisphere':'cerebellum', 'cerebellar_vermis':'cerebellum',
        'midbrain':'brainstem', 'pons':'brainstem', 'midbrain_tegmentum':'midbrain',
        'cerebral_peduncle':'midbrain', 'lateral_ventricle':'ventricular_system', 'fourth_ventricle':'ventricular_system',
        'anterior_horn':'lateral_ventricle', 'posterior_horn':'lateral_ventricle', 'temporal_horn':'lateral_ventricle',
        'calvarium':'skull', 'skull_base':'skull', 'frontal_bone':'skull', 'parietal_bone':'skull',
        'temporal_bone':'skull', 'occipital_bone':'skull', 'frontal_operculum':'frontal_lobe',
        'centrum_semiovale':'white_matter', 'corona_radiata':'white_matter',
        'parahippocampal_gyrus':'cerebral_gyrus', 'precentral_gyrus':'cerebral_gyrus',
        'precentral_cerebellar_cistern':'subarachnoid_cistern', 'precuneus':'parietal_lobe',
        'ring_enhancement':'enhancement', 'nodular_enhancement':'enhancement', 'garland_enhancement':'enhancement'}
    for key, parent in parents.items():
        by_key[key]['parent_key'] = parent
    for item in result:
        item['search_terms'] = list(dict.fromkeys([item['name']] + item['aliases']))
    # The scope of these short forms is ambiguous: retain the mention but do not assert anatomy.
    for key in ['callosal_genu', 'midbrain_tegmentum', 'superior_sagittal_sinus']:
        by_key[key]['note'] = 'Short form requires anatomical context; unresolved attachments are not labels.'
    for role in ROLES:
        for value, definition in [
            ('unknown', 'The value for this role is explicitly stated to be unknown or indeterminate; this is not a missing annotation.'),
            ('other', 'A described value for this role lies outside the current named concept inventory.')]:
            key = role + '_' + value
            result.append(dict(key=key, concept_id='bindformer:' + key, role=role, field='control',
                name=role.replace('_', ' ') + ': ' + value, aliases=[], short_definition=definition,
                preferred_cui=None, search_terms=[], sequence_scope=SEQUENCES.copy(),
                definition_edit_method='project_authored_schema_control',
                review_status='research_schema_control', inclusion='schema_control_not_observed_term',
                parent_key=None, note='Not automatically assigned from missing report text.'))
    return result


def literal_pattern(alias):
    # Exact word/phrase boundaries with hyphen/space variants; never a free prefix.
    words = re.split(r'[\s\-‐‑–—]+', alias.strip())
    return r'(?<!\w)' + r'[\s\-‐‑–—]+'.join(re.escape(w) for w in words) + r'(?!\w)'
