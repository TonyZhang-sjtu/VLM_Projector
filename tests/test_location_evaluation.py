"""Location protocol: alias coverage, fixed eligibility and default integration."""
import pytest

from src.evaluation.location import VERSION, direct_location, evaluate_locations
from src.evaluation.location_reference import reference_location
from src.evaluation.metrics import evaluate_rows


@pytest.mark.parametrize('text,side,site', [
    ('An abnormal signal focus is visible in the left frontal lobe.', 'left', 'frontal_lobe'),
    ('The lesion is in the right cerebellum.', 'right', 'cerebellum'),
    ('An abnormal signal focus is in the LEFT CEREBELLUM.', 'left', 'cerebellum'),
    ('The lesion is in the left cerebellar hemisphere.', 'left', 'cerebellum'),
    ('The lesion is in the right thalamic region.', 'right', 'thalamus'),
    ('The lesion is in the left frontal lobe. Another lesion is in the right frontal lobe.', None, 'frontal_lobe'),
    ('The lesion causes edema in the right frontal lobe.', None, None),
    ('Mass effect displaces the right frontal lobe.', None, None),
    ('The right cerebellum is normal.', None, None),
    ('', None, None),
])
def test_prediction_vocabulary_and_existing_ambiguity_rules(text, side, site):
    parsed = direct_location(text)
    assert (parsed['side'], parsed['site']) == (side, site)


@pytest.mark.parametrize('text,expected', [
    ('The lesion is in the left frontal lobe.', ('left', 'frontal_lobe')),
    ('The lesion is in the right cerebellum.', ('right', 'cerebellum')),
    ('The lesion is in the right cerebellar hemisphere.', ('right', None)),
    ('There are two lesions in the left frontal lobe.', (None, None)),
    ('A possible lesion is in the left frontal lobe.', (None, None)),
    ('No lesion is present in the left frontal lobe.', (None, None)),
    ('The lesion causes a midline shift to the right.', (None, None)),
    ('The lesion is in the left frontal-parietal lobes.', ('left', None)),
    # Prediction aliases do not silently redefine historical reference eligibility.
    ('An abnormal signal focus is in the left frontal lobe.', (None, None)),
])
def test_conservative_reference_policy(text, expected):
    actual = reference_location(text, 'T1W')
    assert (actual['side'], actual['site']) == expected


def report(index, reference, prediction):
    return dict(id=str(index), patient_id=f'p{index // 2}', case_id=f'c{index // 2}',
                cohort='BraTS_GLI', sequence='T1W', reference=reference, prediction=prediction)


def test_default_pipeline_counts_unresolved_predictions_and_preserves_reference_population():
    rows = [
        report(0, 'The lesion is in the left frontal lobe.',
               'An abnormal signal focus is in the left frontal lobe.'),
        report(1, 'The lesion is in the right cerebellum.', 'The lesion is in the left cerebellum.'),
        report(2, 'The lesion is in the left frontal lobe.', ''),
        report(3, 'No focal lesion is present.', 'The lesion is in the left frontal lobe.'),
    ]
    result = evaluate_rows(rows, [r['id'] for r in rows], repeats=0)
    location = result['location']
    assert location['version'] == VERSION
    assert location['samples'] == 4 and location['patients'] == 2
    assert location['overall']['laterality'] == dict(correct=1, n=3, accuracy=1/3,
                                                     patients=2, unresolved=1, wrong=1)
    assert location['overall']['site']['correct'] == 2
    assert location['overall']['side_site']['correct'] == 1
    assert all(s['n'] == 3 for s in location['overall'].values())
    assert evaluate_locations(rows[::-1]) == location
    changed = [dict(r, prediction='') for r in rows]
    empty = evaluate_locations(changed)
    assert empty['reference_labels_sha256'] == location['reference_labels_sha256']
    assert all(s['n'] == 3 and s['accuracy'] == 0 and s['unresolved'] == 3
               for s in empty['overall'].values())


def test_unknown_reference_is_not_a_zero_accuracy_and_audit_exclusions_are_retained():
    row = report(0, 'The lesion is in the left frontal lobe.', 'The lesion is in the left frontal lobe.')
    row['text_audit_status'] = 'requires_review'
    result = evaluate_locations([row])
    assert all(s['n'] == 0 and s['accuracy'] is None for s in result['overall'].values())
    with pytest.raises(ValueError, match='Duplicate'):
        evaluate_locations([row, row])
