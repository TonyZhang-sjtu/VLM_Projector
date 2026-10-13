from dataclasses import replace
import pytest
import json
from pathlib import Path
from src.config import load_config
from src.data.manifest import load_manifests, contained_path
from src.data.datasets import normalize_datasets
from src.data.sequence_mapping import sequence_of
from src.cli.common import config_from, parser


def test_actual_manifests_and_review_flags():
    config = load_config()
    assert config.data.datasets == ['brats']
    samples, summary = load_manifests(config)
    assert [summary[s]['samples'] for s in ('train','val','test')] == [1960,268,560]
    assert all({s.cohort for s in samples[split]} <= {'BraTS_GLI','BraTS_MEN','BraTS_MET'}
               for split in ('train','val','test'))
    assert [summary[s]['dataset_counts'] for s in ('train','val','test')] == [
        {'brats':1960},{'brats':268},{'brats':560}]

    all_samples, all_summary = load_manifests(config, datasets=['all'])
    assert [all_summary[s]['samples'] for s in ('train','val','test')] == [2177,299,622]
    row = next(s for s in all_samples['train'] if s.id == 'Singapore_69_FLAIR')
    assert row.sequence == 'FLAIR'
    assert sum(s.text_audit_status == 'requires_review' for rows in all_samples.values() for s in rows) == 33
    assert all_summary['train']['dataset_counts'] == {'brats':1960,'isles':175,'wmh':42}


def test_dataset_selector_aliases_and_eval_only_scope():
    assert normalize_datasets(['WMH','ISLES22']) == ['isles','wmh']
    assert normalize_datasets('BraTS') == ['brats']
    assert normalize_datasets('all') == ['brats','isles','wmh']
    for invalid in ([], ['all','brats'], ['brats','BraTS_GLI'], ['unknown']):
        with pytest.raises(ValueError):
            normalize_datasets(invalid)

    config = load_config()
    isles, summary = load_manifests(config, datasets=['isles'])
    assert [summary[s]['samples'] for s in ('train','val','test')] == [175,25,50]
    assert all(sample.cohort == 'ISLES22' for rows in isles.values() for sample in rows)
    assert config.data.datasets == ['brats']  # Evaluation override did not mutate training scope.


def test_dataset_cli_parameter_sets_training_scope():
    cli = parser('test')
    args = cli.parse_args(['--datasets', 'wmh', 'isles22'])
    config = config_from(args)
    assert config.data.datasets == ['isles','wmh']
    samples, summary = load_manifests(config)
    assert summary['train']['dataset_counts'] == {'isles':175,'wmh':42}
    assert {sample.cohort for sample in samples['train']} == {'ISLES22','WMH'}


def test_unselected_targets_are_not_opened(tmp_path):
    original = Path('assets/data/RadGenome-Brain/training_manifests/single_sequence_findings_coref_v2')
    for split in ('train','val','test'):
        rows = [json.loads(line) for line in (original/f'{split}.jsonl').read_text().splitlines() if line.strip()]
        if split == 'train':
            isles = next(row for row in rows if row['cohort'] == 'ISLES22')
            isles['annotation_source'] = 'annotations/does_not_exist/modal_wise_finding.json'
        (tmp_path/f'{split}.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in rows))
    config = load_config(overrides=[f'data.manifest_dir={tmp_path}'])
    selected, _ = load_manifests(config)
    assert all(sample.cohort.startswith('BraTS_') for rows in selected.values() for sample in rows)
    with pytest.raises(FileNotFoundError):
        load_manifests(config, datasets=['all'])


def test_unknown_sequence_and_path_escape():
    with pytest.raises(ValueError):
        sequence_of('ISLES22', 'sub-strokecase0106_ses-0001_dwi')
    with pytest.raises(ValueError):
        contained_path('/tmp/root', '../../etc/passwd')
