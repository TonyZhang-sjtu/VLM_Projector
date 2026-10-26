"""Regression tests for extraction, negative validity, gradients and legacy isolation."""
from contextlib import contextmanager
from copy import deepcopy
import sys
from types import SimpleNamespace
import pytest
import torch
from experiments.report_facts_v1.build import build_pairs
from experiments.report_facts_v1.extract import extract_report
from experiments.report_facts_v1.objectives import FactHeads, FactTask, FactQuestion, pairing_loss
from experiments.report_facts_v1.store import FactStore
from src.config import ROOT
from src.training.loss import combined_loss
from test_decoder import setup


def extracted(text, keys, sequence='T2W'):
    row=dict(id='sample',patient_id='p1',cohort='BraTS_GLI',sequence=sequence,split='train',finding=text)
    mentions=[]
    for key,phrase,field in keys:
        start=text.index(phrase)
        mentions.append(dict(key=key,start=start,end=start+len(phrase),mention_id=key,field=field,
            sequence_compatible=True,clause=dict(text=text,start=0,clause_id='c0')))
    return extract_report(row,mentions)


@pytest.mark.parametrize('prefix', ['No description of','No mention of','There may be','It is not reported whether there is',
    'There are no explicit observations provided about', 'The report does not elaborate on the presence of',
    'No specifics are given about', 'The boundaries are defined without mention of',
    'No observation is made regarding', 'There is no detailed description of',
    'There are no further details regarding', 'There is no reported surrounding'])
def test_missing_uncertain_not_negative(prefix):
    text=f'The lesion is present. {prefix} edema.'
    assert 'edema' not in extracted(text,[('edema','edema','observation')])['facts']


def test_negation_and_secondary_effect_do_not_create_false_facts():
    assert extracted('The lesion has no edema.',[('edema','edema','observation')])['facts']['edema']['value']=='absent'
    assert extracted('The lesion has surrounding edema.',[('edema','edema','observation')])['facts']['edema']['value']=='present'
    row=extracted('The lesion produces a midline shift to the right.',[('right','right','laterality')])
    assert not row['facts']
    row=extracted('There are two lesions in the left frontal lobe.',[('left','left','laterality')])
    assert not row['facts']
    text='The lesion has no significant enhancement.'
    assert not extracted(text,[('enhancement','enhancement','enhancement')],'T1C')['facts']


def test_multi_anatomy_and_sequence_restrictions():
    text='The lesion is in the frontal lobe and basal ganglia.'
    keys=[('frontal_lobe','frontal lobe','anatomy'),('basal_ganglia','basal ganglia','anatomy')]
    assert not extracted(text,keys)['facts']
    assert not extracted('The lesion has enhancement.',[('enhancement','enhancement','enhancement')])['facts']


def test_reciprocal_pairs_are_exclusive_and_class_balanced():
    rows=[dict(id=str(i),patient_id=str(i//2),cohort='c',sequence='T1W',facts={
        'laterality':dict(value='left' if i<8 else 'right'),
        'signal':dict(value='hyperintense' if i%2 else 'hypointense')}) for i in range(12)]
    actual=build_pairs(rows);lookup={r['id']:r for r in rows}
    pairs={r['id']:r['pairs'] for r in actual}
    for rid,ps in pairs.items():
        assert len(ps)==1
        p=ps[0];inverse=pairs[p['partner_id']][0]
        assert inverse['partner_id']==rid and inverse['pair_id']==p['pair_id']
        assert inverse['positive']==p['negative']!=p['positive']
        assert lookup[rid]['patient_id']!=lookup[p['partner_id']]['patient_id']
    assert actual==build_pairs(rows)


def test_mixed_signal_cannot_be_used_as_a_negative_of_high_signal():
    rows=[dict(id=str(i),patient_id=str(i),cohort='c',sequence='T2W',
               facts={'signal':dict(value=value)}) for i,value in enumerate(['hyperintense','mixed_signal'])]
    assert build_pairs(rows)==[]


def test_only_correct_final_role_receives_fact_gradient_and_missing_labels_ignored():
    heads=FactHeads(8,['laterality','signal'])
    z=torch.randn(10,8,requires_grad=True)
    loss,count,correct=heads.objective(z,dict(global_tokens=2,num_slots=2),{'laterality':dict(value='left')})
    assert count==1
    loss.backward()
    assert not z.grad[:2].count_nonzero()
    local=z.grad[2:].reshape(2,4,8)
    assert local[:,1].abs().sum()>0 and not local[:,[0,2,3]].count_nonzero()
    assert all(p.grad is not None and not p.grad.count_nonzero() for p in heads.heads['signal'].parameters())


def test_pair_objective_gradients_and_exact_window_normalization():
    pos=torch.tensor(-2.,requires_grad=True);neg=torch.tensor(-1.,requires_grad=True)
    pair,rank,ce=pairing_loss(pos,neg,.2,.25)
    result=dict(loss_sum=torch.tensor(15.,requires_grad=True),auxiliary_loss_sum=.1*pair)
    loss=combined_loss(result,30,sample_count=3)
    torch.testing.assert_close(loss,.5+.1*pair/3)
    loss.backward()
    assert pos.grad<0 and neg.grad>0
    with pytest.raises(ValueError,match='sample count'):combined_loss(result,30)
    x=torch.randn(())
    assert torch.equal(combined_loss({'loss_sum':x},33),x*(1/33))


def test_question_never_contains_answer_or_patient_id():
    q=FactQuestion('private-id','T2W','laterality','The lesion is on the left side.')
    messages=FactTask().messages(q)
    assert 'private-id' not in str(messages) and q.answer not in str(messages)
    assert FactTask().target(q)==q.answer


def test_real_causal_decoder_pair_gradient_and_report_task_unchanged(setup):
    from experiments.report_facts_v1.objectives import FactDecoderView
    # Shared fixture uses the real local tokenizer and a tiny frozen Llama.
    decoder,samples=setup
    task=decoder.task
    original_ids=decoder.text_ids(samples[0],True)
    z=torch.randn(4,decoder.hidden_size,requires_grad=True)
    view=FactDecoderView(decoder)
    positive=view.score(z,samples[0],'laterality','left')
    negative=view.score(z,samples[0],'laterality','right')
    pairing_loss(positive,negative,.2,.25)[0].backward()
    assert z.grad.abs().sum()>0 and torch.isfinite(z.grad).all()
    assert all(p.grad is None for p in decoder.parameters())
    assert decoder.task is task and decoder.text_ids(samples[0],True)==original_ids


def test_real_sidecar_invariants():
    path=ROOT/'assets/knowledge/report_facts_v1'
    if not path.exists():pytest.skip('Local derived facts not installed')
    store=FactStore(path)
    assert store.pairs
    assert all(len(p)==1 for p in store.pairs.values())
    row=next(iter(store.rows.values()))
    with pytest.raises(ValueError,match='outside training'):
        store.get(SimpleNamespace(split='test'))
    with pytest.raises(ValueError,match='stale'):
        store.get(SimpleNamespace(split='train',id=row['id'],finding='changed'))


@contextmanager
def installed_runtime():
    from src.models.projectors import registry,specs
    import src.models.multimodal as multimodal
    from experiments.report_facts_v1 import runtime
    old_defaults=deepcopy(specs.DEFAULTS);old_validate=specs.validate_params
    old_factory=registry.PROJECTORS['bindformer3d'];old_model=multimodal.MultimodalModel
    old_aliases={name:sys.modules[name].MultimodalModel for name in ('src.training.trainer','src.cli.evaluate') if name in sys.modules}
    try:
        runtime.install()
        yield
    finally:
        specs.DEFAULTS.clear();specs.DEFAULTS.update(old_defaults);specs.validate_params=old_validate
        registry.PROJECTORS['bindformer3d']=old_factory;multimodal.MultimodalModel=old_model
        for name,value in old_aliases.items():sys.modules[name].MultimodalModel=value
        runtime._installed=False


def test_disabled_extension_is_bitwise_legacy_and_enabled_heads_preserve_core_rng():
    from test_bindformer_modules import config,volume
    from src.models.projectors.registry import build_projector
    args=['projector.params.knowledge_mode=none','projector.params.fixed_spatial_regions=true',
          'projector.params.spatial_mean_aggregation=true']
    torch.manual_seed(42);base=build_projector(config(*args),8,16);rng=torch.get_rng_state()
    with installed_runtime():
        torch.manual_seed(42)
        disabled=build_projector(config(*args,'projector.params.report_facts.enabled=false'),8,16)
        assert torch.equal(rng,torch.get_rng_state())
        assert base.state_dict().keys()==disabled.state_dict().keys()
        for key,value in base.state_dict().items():assert torch.equal(value,disabled.state_dict()[key])
        features,coords,meta=volume()
        assert torch.equal(base(features,coords,meta).embeddings,disabled(features,coords,meta).embeddings)
        if (ROOT/'assets/knowledge/report_facts_v1').exists():
            torch.manual_seed(42)
            enabled=build_projector(config(*args,'projector.params.report_facts.fact_weight=0.1'),8,16)
            assert torch.equal(rng,torch.get_rng_state())
            for key,value in base.state_dict().items():assert torch.equal(value,enabled.state_dict()[key])
            assert torch.equal(base(features,coords,meta).embeddings,enabled(features,coords,meta).embeddings)


@pytest.mark.skipif(not __import__('os').getenv('PROJECTOR_TEST_DEVICE'),reason='Explicit GPU required')
def test_head_initialization_preserves_cuda_rng():
    from test_bindformer_modules import config
    from src.models.projectors.registry import build_projector
    with installed_runtime():
        torch.cuda.set_device(__import__('os').environ['PROJECTOR_TEST_DEVICE'])
        torch.manual_seed(42)
        before=torch.cuda.get_rng_state().clone()
        build_projector(config('projector.params.knowledge_mode=none',
            'projector.params.fixed_spatial_regions=true','projector.params.spatial_mean_aggregation=true',
            'projector.params.report_facts.fact_weight=0.1'),8,16)
        assert torch.equal(before,torch.cuda.get_rng_state())
