"""BindFormer extensions use the current joint-LoRA launcher without enabling them."""
from pathlib import Path
import pytest
from src.cli.bindformer_sweep import parser,build_plan
from test_baseline_sweep import population


def test_joint_plan_preserves_confirmed_protocol(tmp_path,population):
    args=parser().parse_args(['--output-root',str(tmp_path),'--run-tag','smoke','--dry-run'])
    plan=build_plan(args)
    assert [r['ratio'] for r in plan['runs']]==[4,8,16,32]
    for run,g,k in zip(plan['runs'],[550,273,137,67],[411,206,103,52]):
        c=run['config'];p=c['projector']['params']
        assert c['training']['stage']=='joint_lora' and c['training']['encoder_lr']>0
        assert not c['training']['feature_cache'] and c['decoder']['frozen']
        assert c['data']['datasets']==['brats'] and p['vocabulary_scope']=='brats_train'
        assert (run['budget']['global_tokens'],run['budget']['num_slots'])==(g,k)
        assert run['budget']['expected_output_tokens']==g+4*k
        assert p['slot_iterations']==3 and p['global_fraction']==.25
        assert list(p['role_query_counts'].values())==[1,1,1,1]
        assert not any(p[name] for name in ('geometry_enabled','within_slot_sa','local_reads_global',
                                          'feature_loss_weight','text_loss_weight'))
        assert run['evaluation_samples']==560
        assert run['checkpoint'].endswith('/joint_lora/checkpoints/best.pt')
    assert not Path(plan['run_root']).exists()


@pytest.mark.parametrize('override',['projector.params.vocabulary_scope=all_train',
                                     'data.datasets=[isles]','training.stage=a'])
def test_scope_and_training_protocol_guard(tmp_path,population,override):
    args=parser().parse_args(['--output-root',str(tmp_path),'--set',override])
    with pytest.raises(ValueError):build_plan(args)
    assert list(tmp_path.iterdir())==[]


def test_knowledge_support_and_asymmetric_query_ablation(tmp_path,population):
    args=parser().parse_args(['--output-root',str(tmp_path),'--ratios','32',
        '--knowledge-mode','none','--support-beta','0','--finding-queries','1',
        '--location-laterality-queries','2','--signal-queries','3','--morphology-extent-queries','2',
        '--set','projector.params.within_slot_sa=true'])
    run=build_plan(args)['runs'][0];p=run['config']['projector']['params']
    assert p['knowledge_mode']=='none' and p['support_beta']==[0.]*4
    assert list(p['role_query_counts'].values())==[1,2,3,2] and p['within_slot_sa']
    assert (run['budget']['global_tokens'],run['budget']['num_slots'])==(67,26)
    assert run['budget']['expected_output_tokens']==275


@pytest.mark.parametrize('extra',[
    ['--signal-queries','0'],['--finding-queries','-1'],['--support-beta','-1'],
    ['--support-beta','nan'],['--global-fraction','1'],['--ratios','4','4'],
    ['--set','projector.params.text_loss_weight=.1']])
def test_invalid_extension_options_fail_before_writing(tmp_path,population,extra):
    with pytest.raises(ValueError):
        build_plan(parser().parse_args(['--output-root',str(tmp_path),*extra]))
    assert list(tmp_path.iterdir())==[]
