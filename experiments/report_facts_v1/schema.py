"""Frozen v1 field inventory, prompts and role ownership (no patient data)."""
VERSION = 'report_facts_v1'
FIELDS = {
    'edema': dict(role=0, values=['absent', 'present'], question='Is edema associated with the described lesion?',
                  answers=['No associated edema is present.', 'Associated edema is present.']),
    'laterality': dict(role=1, values=['left', 'right'], question='On which side is the described lesion located?',
                      answers=['The lesion is on the left side.', 'The lesion is on the right side.']),
    'site': dict(role=1, values=['frontal_lobe','parietal_lobe','temporal_lobe','occipital_lobe','cerebellum','thalamus','basal_ganglia','brainstem'],
                 question='Where is the described lesion located?',
                 answers=['The lesion is in the frontal lobe.', 'The lesion is in the parietal lobe.',
                          'The lesion is in the temporal lobe.', 'The lesion is in the occipital lobe.',
                          'The lesion is in the cerebellum.', 'The lesion is in the thalamus.',
                          'The lesion is in the basal ganglia.', 'The lesion is in the brainstem.']),
    'signal': dict(role=2, values=['hyperintense','hypointense','isointense','mixed_signal'],
                   question='What is the signal intensity of the described lesion on this sequence?',
                   answers=['The lesion has high signal intensity.', 'The lesion has low signal intensity.',
                            'The lesion is isointense.', 'The lesion has mixed signal intensity.']),
    'enhancement': dict(role=2, values=['absent','present'], question='Does the described lesion enhance after contrast?',
                        answers=['The lesion does not enhance.', 'The lesion enhances after contrast.']),
    'margin': dict(role=3, values=['well_defined','ill_defined'], question='How are the boundaries of the described lesion characterized?',
                   answers=['The lesion has well-defined boundaries.', 'The lesion has ill-defined boundaries.']),
}
# Enhancement is retained for inspection but disabled by default: the current
# reports contain almost no unambiguous absence examples on T1C.
DEFAULT_FIELDS = [field for field in FIELDS if field != 'enhancement']
