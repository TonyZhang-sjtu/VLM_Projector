from argparse import Namespace
import json
from types import SimpleNamespace
import pytest
from src.cli.evaluate_checkpoint import (
    checkpoint_config, combine_metrics, make_commands, metric_selection, select_samples,
    partial_metric_details, validate_saved_generation,
)


def test_metric_selection_and_only_requested_scorers(tmp_path):
    assert metric_selection(['all']) == ['bleu4', 'rougeL', 'green', 'radgraph', 'rate', 'bertscore', 'answer_ce']
    for values in (['all', 'green'], ['green', 'green'], ['BERTScore']):
        with pytest.raises(ValueError): metric_selection(values)
    args = Namespace(metrics=['rougeL', 'green'], output_dir=tmp_path, device='cuda:2',
                     scoring_device=None, repeats=20)
    generation, scoring = make_commands(args, tmp_path/'config.yaml', tmp_path/'ids.json')
    assert '--skip-answer-ce' in generation
    assert scoring[scoring.index('--metrics')+1:] == ['green']
    args.metrics=['bleu4','answer_ce']
    generation, scoring=make_commands(args,tmp_path/'config.yaml',tmp_path/'ids.json')
    assert '--skip-answer-ce' not in generation and scoring is None


def test_selection_order_coverage_and_positive_limits():
    samples=[SimpleNamespace(id=str(i)) for i in range(3)]
    assert [r.id for r in select_samples(samples,['2','0'],1)]==['2']
    for ids in ([],['2','2'],['missing'],[1]):
        with pytest.raises(ValueError):select_samples(samples,ids)
    with pytest.raises(ValueError):select_samples(samples,limit=0)


def test_fractional_sampling_is_reproducible_and_validated():
    samples=[SimpleNamespace(id=str(i)) for i in range(10)]
    selected=select_samples(samples,sample_fraction=0.3,seed=17)
    assert len(selected)==3
    assert [s.id for s in selected]==[s.id for s in select_samples(samples,sample_fraction=0.3,seed=17)]
    assert [s.id for s in selected]==sorted((s.id for s in selected),key=int)
    assert len(select_samples(samples,sample_fraction=1.0))==10
    for fraction in (0, -0.1, 1.1, float('nan'), float('inf')):
        with pytest.raises(ValueError):select_samples(samples,sample_fraction=fraction)
    with pytest.raises(ValueError,match='mutually exclusive'):
        select_samples(samples,limit=3,sample_fraction=0.3)


def test_restore_stage_and_preprocessing_from_checkpoint(tmp_path):
    import torch
    from omegaconf import OmegaConf
    from src.config import load_config
    config=load_config(files=['configs/training/stage_b_lora.yaml'],overrides=['preprocessing.spacing_policy=2mm'])
    saved={'format_version':1,'stage':'b_lora','config':OmegaConf.to_container(config,resolve=True)}
    path=tmp_path/'checkpoint.pt';torch.save(saved,path)
    args=Namespace(device='cuda:2',output_dir=tmp_path,split='test',metrics=['green'],repeats=100)
    restored=checkpoint_config(path,args)
    assert restored.training.stage=='b_lora'
    assert restored.preprocessing.spacing_policy=='2mm'
    assert restored.training.device=='cuda:2' and restored.evaluation.bootstrap_repeats==0
    assert restored.evaluation.split=='test'
    assert restored.data.datasets==['brats'] and restored.evaluation.datasets is None
    args.datasets=['isles','wmh']
    restored=checkpoint_config(path,args)
    assert restored.data.datasets==['brats']
    assert restored.evaluation.datasets==['isles','wmh']
    saved['stage']='a';torch.save(saved,path)
    with pytest.raises(ValueError,match='inconsistent'):checkpoint_config(path,args)


def test_summary_does_not_mislabel_unrequested_or_failed_scores():
    generated={'status':'complete','overall':{'bleu4':12,'rougeL':30},'answer_ce':{'answer_ce':2.5}}
    scored={'status':'complete','clinical':{'overall':{'green':{
        'mean':.6, 'status':'complete', 'valid':3, 'expected':3,
    }}}}
    overall, skipped = combine_metrics(generated,scored,['green','answer_ce'])
    assert overall == {'green':.6,'answer_ce':2.5} and skipped == {}
    assert partial_metric_details(scored, overall) == {}
    scored['status']='partial'
    scored['clinical']['overall']['green'] = {
        'mean': None, 'valid_only_mean': .75, 'status': 'incomplete', 'valid': 2,
        'expected': 3, 'statuses': {'ok': 2, 'format_repair_failed': 1},
    }
    overall, skipped = combine_metrics(generated,scored,['green','answer_ce'])
    assert overall == {'green': .75, 'answer_ce':2.5} and skipped == {}
    partial = partial_metric_details(scored, overall)['green']
    assert partial['valid'] == 2 and partial['failed'] == 1 and partial['expected'] == 3
    assert partial['coverage'] == pytest.approx(2 / 3)
    assert partial['reason'] == 'valid_samples_only'
    assert partial['statuses'] == {'ok': 2, 'format_repair_failed': 1}
    # Raw full-population summaries remain unchanged; a subset mean is not a full-set mean.
    assert scored['clinical']['overall']['green']['mean'] is None


@pytest.mark.parametrize('valid,value', [(0, None), (0, 0.), (2, None),
                                         (2, float('nan')), (2, float('inf'))])
def test_green_unavailable_subset_is_not_published(valid, value):
    generated = {'status': 'complete', 'overall': {}, 'answer_ce': {'answer_ce': 2.5}}
    scored = {'status': 'partial', 'clinical': {'overall': {'green': {
        'mean': None, 'valid_only_mean': value, 'status': 'incomplete',
        'valid': valid, 'expected': 3,
    }}}}
    overall, skipped = combine_metrics(generated, scored, ['green', 'answer_ce'])
    assert overall == {'answer_ce': 2.5}
    assert skipped['green']['reason'] == 'incomplete'
    assert partial_metric_details(scored, overall) == {}


def test_zero_green_is_a_valid_subset_score():
    generated = {'status': 'complete', 'overall': {}}
    scored = {'status': 'partial', 'clinical': {'overall': {'green': {
        'mean': None, 'valid_only_mean': 0., 'status': 'incomplete',
        'valid': 1, 'expected': 2,
    }}}}
    overall, skipped = combine_metrics(generated, scored, ['green'])
    assert overall == {'green': 0.} and skipped == {}
    assert partial_metric_details(scored, overall)['green']['failed'] == 1


def test_saved_generation_is_reused_only_after_identity_validation(tmp_path):
    sample = SimpleNamespace(id='s1', patient_id='p1', case_id='c1', cohort='BraTS_GLI',
                             sequence='t1n', split='test', finding='reference')
    (tmp_path / 'generation').mkdir()
    (tmp_path / 'expected_ids.json').write_text('["s1"]')
    (tmp_path / 'generation/metrics.json').write_text('{"status":"complete"}')
    row = {'id': 's1', 'patient_id': 'p1', 'case_id': 'c1', 'cohort': 'BraTS_GLI',
           'sequence': 't1n', 'split': 'test', 'reference': 'reference',
           'prediction': 'prediction', 'checkpoint_id': 'abc'}
    (tmp_path / 'generation/predictions.jsonl').write_text(json.dumps(row) + '\n')
    assert validate_saved_generation(tmp_path, 'abc', [sample])
    row['checkpoint_id'] = 'different'
    (tmp_path / 'generation/predictions.jsonl').write_text(json.dumps(row) + '\n')
    with pytest.raises(ValueError, match='checkpoint/manifest'):
        validate_saved_generation(tmp_path, 'abc', [sample])


def test_resume_publishes_current_location_scores_from_legacy_generation(tmp_path, monkeypatch):
    """Old generation files need no regeneration to receive the default rule."""
    import sys
    from omegaconf import OmegaConf
    from src.cli import evaluate_checkpoint as cli
    from src.evaluation.clinical import file_digest
    from src.evaluation.location import VERSION

    checkpoint = tmp_path / 'best.pt'
    checkpoint.write_bytes(b'checkpoint identity only; model loading is mocked')
    output = tmp_path / 'evaluation'
    (output / 'generation').mkdir(parents=True)
    (output / 'checkpoint.pt').write_bytes(checkpoint.read_bytes())
    sample = SimpleNamespace(id='s1', patient_id='p1', case_id='c1', cohort='BraTS_GLI',
                             sequence='T1W', split='test', finding='The lesion is in the left cerebellum.')
    row = {k: getattr(sample, k) for k in ('id', 'patient_id', 'case_id', 'cohort', 'sequence', 'split')}
    row.update(reference=sample.finding, prediction='An abnormal signal focus is in the left cerebellum.',
               checkpoint_id=file_digest(checkpoint))
    (output / 'generation/predictions.jsonl').write_text(json.dumps(row) + '\n')
    (output / 'expected_ids.json').write_text('["s1"]')
    legacy = {'status': 'complete', 'overall': {'bleu4': 12.}, 'bootstrap': {}}
    (output / 'generation/metrics.json').write_text(json.dumps(legacy))
    config = OmegaConf.create(dict(seed=42, training=dict(stage='joint_lora'),
                                   data=dict(datasets=['brats']), evaluation=dict(datasets=None)))
    monkeypatch.setattr(cli, 'checkpoint_config', lambda *args: config)
    monkeypatch.setattr('src.data.manifest.load_manifests', lambda *args, **kwargs:
                        ({'test': [sample]}, {'test': {'datasets': ['brats'], 'dataset_counts': {'brats': 1}}}))
    monkeypatch.setattr(cli.subprocess, 'run', lambda *args, **kwargs: pytest.fail('Unexpected model subprocess'))
    monkeypatch.setattr(sys, 'argv', ['evaluate_checkpoint', '--checkpoint', str(checkpoint),
                                     '--output-dir', str(output), '--device', 'cuda:0',
                                     '--resume', '--metrics', 'bleu4', '--repeats', '0'])
    cli.main()
    result = json.loads((output / 'metric.json').read_text())
    assert result['status'] == 'complete' and result['overall'] == legacy['overall']
    assert result['location']['version'] == VERSION
    assert result['location']['overall']['side_site']['accuracy'] == 1.
    assert json.loads((output / 'generation/metrics.json').read_text()) == legacy
