import copy
import json
import math
import shutil
from pathlib import Path
import pytest
import torch
from torch.nn import functional as F
from src.config import ROOT, load_config
from src.models.projectors.registry import build_projector
from src.models.projectors.bindformer.slots import CompetitiveSlotRouter
from src.models.projectors.bindformer.concept_memory import ConceptMemory, read_bank
from src.models.projectors.bindformer.input_adapter import normalized_grid_centers
from src.models.projectors.bindformer.reader import SupportReader

BANK = ROOT/'assets/knowledge/bindformer_brain_candidates_v2'
EXCLUDED = {'centrum_semiovale','cerebral_peduncle','corona_radiata','cortex',
            'internal_capsule','parahippocampal_gyrus','precuneus','restricted_diffusion'}


def config(*overrides):
    return load_config(['configs/projector/bindformer3d.yaml'],[
        'projector.params.num_tokens=12','projector.params.inner_dim=16',
        'projector.params.num_heads=2','projector.hidden_size=32',
        'projector.params.return_auxiliary=true',*overrides])


def volume(channels=8):
    coords=torch.cartesian_prod(torch.arange(3),torch.arange(4),torch.arange(2))
    meta={'grid_dhw':[3,4,2],'sequence':'T2W','array_zyx_to_dhw':[0,1,2],
          'oriented':{'direction':[-1,0,0,0,1,0,0,0,-1]}}
    return torch.randn(24,channels),coords,meta


def test_grid_centers_axis_permutation_direction_and_missing_metadata():
    coords=torch.tensor([[0,0,0],[1,2,3]])
    grid=[2,3,4]
    meta={'array_zyx_to_dhw':[2,1,0], 'oriented':{'direction':[-1,0,0,0,1,0,0,0,-1]}}
    # DHW maps to XYZ for this permutation; normalize then flip X/Z to LPS.
    expected=torch.tensor([[.5,-2/3,.75],[-.5,2/3,-.75]])
    torch.testing.assert_close(normalized_grid_centers(coords,grid,meta),expected)
    with pytest.raises(ValueError,match='geometry metadata'):
        normalized_grid_centers(coords,grid,{})


def test_slot_update_matches_equations_and_final_assignment():
    torch.manual_seed(17)
    model=CompetitiveSlotRouter(8,3,iterations=1,background=True).double()
    h=torch.randn(11,8,dtype=torch.double,requires_grad=True)
    initial=model.initial_slots
    key=model.to_k(model.norm_input(h))
    logits=key@model.to_q(model.norm_slots(initial)).T/math.sqrt(8)
    a=logits.softmax(-1)
    torch.testing.assert_close(a.sum(-1),torch.ones(11,dtype=torch.double))
    pooled=(a/(a.sum(0,keepdim=True)+model.epsilon)).T@h
    expected=model.gru(pooled,initial)
    expected=expected+model.mlp(expected)
    slots, final=model(h)
    torch.testing.assert_close(slots,expected[1:])
    torch.testing.assert_close(final,(key@model.to_q(model.norm_slots(expected)).T/math.sqrt(8)).softmax(-1))
    assert not torch.allclose(final,a)
    (slots.square().sum()+final.square().sum()).backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    assert h.grad.abs().sum()>0


def test_three_updates_reset_and_checkpoint_equivalence():
    torch.manual_seed(19)
    a=CompetitiveSlotRouter(8,4,iterations=3)
    b=copy.deepcopy(a);b.checkpoint_blocks=True
    h=torch.randn(13,8,requires_grad=True); other=h.detach().clone().requires_grad_()
    before=a.initial_slots.detach().clone()
    calls=[]
    hook=a.gru.register_forward_hook(lambda *_:calls.append(1))
    s,p=a(h)
    assert len(calls)==3
    hook.remove()
    a(torch.randn_like(h))
    repeated=a(h)
    torch.testing.assert_close(s,repeated[0],rtol=0,atol=0)
    torch.testing.assert_close(a.initial_slots,before,rtol=0,atol=0)
    t,q=b(other)
    torch.testing.assert_close(s,t,rtol=0,atol=0)
    (s.square().sum()+p.square().sum()).backward()
    (t.square().sum()+q.square().sum()).backward()
    torch.testing.assert_close(h.grad,other.grad,rtol=0,atol=0)
    for x,y in zip(a.parameters(),b.parameters()):
        torch.testing.assert_close(x.grad,y.grad,rtol=0,atol=0)


def test_brats_bank_physically_excludes_other_cohort_only_terms():
    bank=ConceptMemory(BANK,'brats_train',16)
    ids=bank.provenance['concept_ids']
    assert bank.embeddings.shape==(120,768)
    assert not EXCLUDED & {x.split(':')[-1] for x in ids}
    all_bank=read_bank(BANK,'all_train')
    assert set(all_bank[-1]['concept_ids'])-set(ids)=={'bindformer:'+x for x in EXCLUDED}
    assert not bank.embeddings.requires_grad
    assert 'embeddings' not in dict(bank.named_parameters())
    slots=torch.randn(3,16,requires_grad=True)
    mixed,weights=bank(slots,'T2W')
    assert mixed.shape==(3,4,768)
    torch.testing.assert_close(weights.sum(-1),torch.ones(3,4))
    for term in ('enhancement','nonenhancing','ring_enhancement','dural_tail'):
        assert torch.count_nonzero(weights[:,:,ids.index('bindformer:'+term)])==0
    allowed=bank.role_mask&bank.sequence_mask[2][None,:]
    assert torch.count_nonzero(weights[:,~allowed])==0
    _,contrast=bank(slots,'T1C')
    assert (contrast[:,2,ids.index('bindformer:enhancement')]>0).all()
    mixed.square().sum().backward()
    assert slots.grad.abs().sum()>0
    assert all(p.grad is not None and p.grad.abs().sum()>0 for p in bank.parameters())
    with pytest.raises(ValueError,match='sequence'):bank(slots,'T1')


def test_bank_tamper_rejected_before_training(tmp_path):
    dest=tmp_path/'bank';shutil.copytree(BANK,dest,ignore=shutil.ignore_patterns('mentions*','*reconciliation*'))
    (dest/'role_mask.npy').write_bytes(b'bad')
    with pytest.raises(ValueError,match='hash mismatch'):read_bank(dest,'brats_train')


@pytest.mark.parametrize('beta', [[0]*4,[1]*4,[0,.5,1,2]])
def test_visual_reader_matches_explicit_multihead_softmax(beta):
    torch.manual_seed(31)
    m=SupportReader(8,2,beta,chunk_size=4).double()
    q=torch.randn(3,4,8,dtype=torch.double,requires_grad=True)
    h=torch.randn(9,8,dtype=torch.double,requires_grad=True)
    a=torch.randn(9,3,dtype=torch.double).softmax(-1).requires_grad_()
    qq=m.to_q(m.norm_query(q)).reshape(1,12,2,4).transpose(1,2)
    kk,vv=[t.reshape(1,9,2,4).transpose(1,2) for t in m.to_kv(m.norm_visual(h)).chunk(2,-1)]
    bias=(a.T[:,None,:]+m.epsilon).log()*torch.tensor(beta,dtype=torch.double)[None,:,None]
    logits=qq@kk.transpose(-1,-2)/2+bias.reshape(1,1,12,9)
    read=(logits.softmax(-1)@vv).transpose(1,2).reshape(3,4,8)
    read=m.to_out(read);expected=read+m.ffn(read)
    actual=m(q,h,a)
    torch.testing.assert_close(actual,expected,rtol=1e-10,atol=1e-10)
    gradients=torch.autograd.grad(actual.square().sum(),(q,h,a),retain_graph=True)
    reference=torch.autograd.grad(expected.square().sum(),(q,h,a))
    for x,y in zip(gradients,reference):torch.testing.assert_close(x,y,rtol=1e-9,atol=1e-10)


def test_support_bias_suppresses_outside_patches_and_zero_beta_ignores_support():
    m=SupportReader(8,2,[0]*4)
    q=torch.randn(2,4,8);h=torch.randn(10,8)
    a=torch.rand(10,2);b=torch.rand(10,2)
    torch.testing.assert_close(m(q,h,a),m(q,h,b),rtol=0,atol=0)
    # Uniform content logits isolate the shared support mechanism.
    with torch.no_grad():m.to_q.weight.zero_();m.support_beta.fill_(1)
    a=torch.zeros(10,2);a[:5,0]=1;a[5:,1]=1
    qprojected=m.to_q(m.norm_query(q))
    k,v=m.to_kv(m.norm_visual(h)).chunk(2,-1)
    k,v=[t.reshape(1,10,2,4).transpose(1,2) for t in (k,v)]
    read=m._attend(qprojected,k,v,a.T)
    expected=torch.stack([m.to_kv(m.norm_visual(h)).chunk(2,-1)[1][:5].mean(0),
                          m.to_kv(m.norm_visual(h)).chunk(2,-1)[1][5:].mean(0)])
    torch.testing.assert_close(read,expected[:,None,:].expand_as(read),atol=1e-5,rtol=1e-5)


def test_no_query_or_knowledge_residual_bypasses_visual_values():
    m=SupportReader(8,2,[1]*4)
    with torch.no_grad():m.to_kv.weight[8:].zero_()
    h=torch.randn(10,8);a=torch.rand(10,2)
    torch.testing.assert_close(m(torch.randn(2,4,8),h,a),
                               m(torch.randn(2,4,8)*100,h,a),rtol=0,atol=0)


@pytest.mark.parametrize('knowledge',['name_definition','none'])
@pytest.mark.parametrize('background',[True,False])
def test_complete_forward_layout_freeze_and_all_parameter_gradients(knowledge,background):
    c=config(f'projector.params.knowledge_mode={knowledge}',
             f'projector.params.background_slot={str(background).lower()}')
    m=build_projector(c,8,16)
    x,coords,meta=volume();x.requires_grad_()
    output=m(x,coords,meta)
    aux=output.auxiliary
    assert output.embeddings.shape==(12,16)
    assert (aux.global_tokens.shape[0],aux.slots.shape[0])==(4,2)
    assert aux.assignments.shape==(24,2+background)
    expected=m.head(m.final_norm(torch.cat((aux.global_tokens,aux.role_tokens.flatten(0,1)))))
    torch.testing.assert_close(expected,output.embeddings,rtol=0,atol=0)
    json.dumps(output.metadata)
    output.embeddings.square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum()>0
               for p in m.parameters())
    assert x.grad.abs().sum()>0
    if knowledge=='none':
        assert m.concept_memory is None
        assert not any('concept' in name for name,_ in m.named_parameters())
    else:
        assert m.concept_memory.embeddings.grad is None


def test_geometry_and_interaction_extension_shapes():
    m=build_projector(config(),8,16)
    s=torch.randn(2,16);e=torch.randn(2,4,768);g=torch.randn(2,4,16)
    torch.testing.assert_close(m.role_queries(s,e,g),m.role_queries(s,e)+g)
    b=torch.randn(2,4,16)
    assert m.interactions(b,torch.randn(4,16)) is b
