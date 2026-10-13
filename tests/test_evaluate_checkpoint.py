from argparse import Namespace
from types import SimpleNamespace
import pytest
from src.cli.evaluate_checkpoint import (
    checkpoint_config, combine_metrics, make_commands, metric_selection, select_samples,
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
    scored={'status':'complete','clinical':{'overall':{'green':{'mean':.6}}}}
    assert combine_metrics(generated,scored,['green','answer_ce'])=={'green':.6,'answer_ce':2.5}
    scored['status']='incomplete'
    with pytest.raises(ValueError):combine_metrics(generated,scored,['green'])
