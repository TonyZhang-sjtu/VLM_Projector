"""Routing isolation, strict switching, gradients and packed-reader correctness."""
import copy
import math
import pytest
import torch
from src.models.projectors.registry import build_projector
from src.models.projectors.bindformer.regions import make_regions
from src.models.projectors.bindformer.reader import SupportReader
from src.models.projectors.bindformer.slots import CompetitiveSlotRouter
from test_bindformer_modules import config, volume


def spatial_config(adaptive=False, *extra):
    aggregation = 'adaptive' if adaptive else 'mean'
    return config('projector.params.fixed_spatial_regions=true',
                  f'projector.params.spatial_{aggregation}_aggregation=true',
                  'projector.params.router_value_normalization=true', *extra)


@pytest.mark.parametrize('flags', [
    ['fixed_spatial_regions=true'],
    ['spatial_mean_aggregation=true'],
    ['spatial_adaptive_aggregation=true'],
    ['fixed_spatial_regions=true', 'spatial_mean_aggregation=true', 'spatial_adaptive_aggregation=true'],
    ['router_value_normalization=1'], ['spatial_attention_temperature=0'],
])
def test_invalid_switches_fail_early(flags):
    with pytest.raises(ValueError):
        config(*['projector.params.'+s for s in flags])


@pytest.mark.parametrize('grid,count', [((39,15,15),206), ((39,15,15),103),
                                      ((39,15,15),52), ((1,1,7),7), ((2,3,4),1)])
def test_exact_nonempty_disjoint_partition(grid,count):
    r = make_regions(grid, {}, count, 'cpu')
    b = r.membership(torch.float32)
    assert b.shape == (math.prod(grid),count)
    assert torch.equal(b.sum(-1), torch.ones(math.prod(grid)))
    assert (b.sum(0)>0).all()
    assert torch.equal(r.indices, make_regions(grid,{},count,'cpu').indices)
    # Every region is an axis-aligned contiguous cuboid, not arbitrary round robin.
    coords = torch.cartesian_prod(*(torch.arange(n) for n in grid))
    for i in range(count):
        xyz = coords[r.indices[i,r.valid[i]]]
        assert int((xyz.max(0).values-xyz.min(0).values+1).prod()) == len(xyz)


def test_physical_axis_and_channelwise_value_normalization():
    r = make_regions((4,4,4), {'shape_dhw':[4,4,4], 'array_zyx_to_dhw':[0,1,2],
        'bounded_before_pad':{'spacing_xyz':[1,1,10]}}, 2, 'cpu')
    assert r.indices[0,r.valid[0]].max() == 31  # longest D axis
    router = CompetitiveSlotRouter(8,2)
    h = torch.randn(9,8)*20+100
    assert router.values(h) is h
    router.configure_routing(normalization=True,projection=True)
    torch.testing.assert_close(router.values(h), router.value_norm(h))
    torch.testing.assert_close(router.values(h).mean(-1), torch.zeros(9), atol=2e-6,rtol=0)


@pytest.mark.parametrize('adaptive',[False,True])
def test_spatial_router_disallows_outside_evidence_and_has_gradients(adaptive):
    torch.manual_seed(53)
    router = CompetitiveSlotRouter(8,3)
    router.configure_routing(normalization=True,spatial=True,adaptive=adaptive)
    h = torch.randn(21,8,requires_grad=True)
    r = make_regions((3,7,1),{},3,'cpu')
    slots,w = router(h,r)
    torch.testing.assert_close(w.sum(0),torch.ones(3))
    assert not w[r.membership(w.dtype)==0].count_nonzero()
    if not adaptive:
        b=r.membership(w.dtype)
        torch.testing.assert_close(w,b/b.sum(0,keepdim=True),rtol=0,atol=0)
    changed=h.detach().clone()
    changed[r.indices[0,r.valid[0]]] += torch.arange(8)*4
    slots2,w2=router(changed,r)
    torch.testing.assert_close(slots[1:],slots2[1:],rtol=0,atol=0)
    torch.testing.assert_close(w[:,1:],w2[:,1:],rtol=0,atol=0)
    (slots.square().sum()+w.square().sum()).backward()
    assert h.grad.abs().sum()>0
    assert all(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum()>0
               for p in router.parameters() if p.requires_grad)
    assert not router.initial_slots.grad[0].count_nonzero()


@pytest.mark.parametrize('beta',[[0]*4,[1]*4,[0,.5,1,2]])
def test_packed_reader_matches_dense_hard_mask_outputs_and_gradients(beta):
    torch.manual_seed(51)
    reader=SupportReader(8,2,beta,chunk_size=4).double()
    regions=make_regions((3,3,1),{},4,'cpu')  # unequal lengths exercise padded memory
    q=torch.randn(4,4,8,dtype=torch.double,requires_grad=True)
    h=torch.randn(9,8,dtype=torch.double,requires_grad=True)
    packed=torch.randn_like(regions.indices,dtype=torch.double).masked_fill(~regions.valid,-torch.inf).softmax(-1)
    a=regions.dense(packed).requires_grad_()
    actual=reader(q,h,a,regions)
    qq=reader.to_q(reader.norm_query(q)).reshape(1,16,2,4).transpose(1,2)
    kk,vv=[t.reshape(1,9,2,4).transpose(1,2) for t in reader.to_kv(reader.norm_visual(h)).chunk(2,-1)]
    bias=(a.T[:,None,:]+reader.epsilon).log()*torch.tensor(beta,dtype=torch.double)[None,:,None]
    bias=bias.masked_fill(~regions.membership(torch.bool).T[:,None,:],-torch.inf)
    read=((qq@kk.transpose(-1,-2)/2+bias.reshape(1,1,16,9)).softmax(-1)@vv)
    read=reader.to_out(read.transpose(1,2).reshape(4,4,8))
    expected=read+reader.ffn(read)
    torch.testing.assert_close(actual,expected,rtol=1e-10,atol=1e-10)
    for x,y in zip(torch.autograd.grad(actual.square().sum(),(q,h,a),retain_graph=True),
                   torch.autograd.grad(expected.square().sum(),(q,h,a))):
        torch.testing.assert_close(x,y,rtol=1e-9,atol=1e-10)
    changed=h.detach().clone()
    outside=~regions.membership(torch.bool)[:,0]
    changed[outside]*=-30
    torch.testing.assert_close(reader(q,changed,a,regions)[0],actual[0],rtol=0,atol=0)
    # Zero beta MUST NOT reopen access outside the assigned box.


@pytest.mark.parametrize('adaptive',[False,True])
def test_full_spatial_sa_ca_checkpoint_and_occupancy_contract(adaptive):
    torch.manual_seed(54)
    p=build_projector(spatial_config(adaptive,'projector.params.within_slot_sa=true',
                                    'projector.params.local_reads_global=true',
                                    'projector.params.checkpoint_blocks=false'),8,16)
    other=copy.deepcopy(p)
    other.params['checkpoint_blocks']=True
    for module in other.modules():
        if hasattr(module,'checkpoint_blocks'):module.checkpoint_blocks=True
    x,c,m=volume();x.requires_grad_();y=x.detach().clone().requires_grad_()
    out=p(x,c,m);replay=other(y,c,m)
    assert out.embeddings.shape==(12,16)
    aux=out.auxiliary
    assert not out.metadata['projector']['background_slot']
    assert aux.assignments.sum()==24  # occupancy; NOT the normalized sum K
    torch.testing.assert_close(aux.evidence_weights.sum(0),torch.ones(2))
    torch.testing.assert_close(out.embeddings,replay.embeddings,rtol=0,atol=0)
    out.embeddings.square().sum().backward()
    replay.embeddings.square().sum().backward()
    torch.testing.assert_close(x.grad,y.grad,rtol=0,atol=0)
    for (name,a),(_,b) in zip(p.named_parameters(),other.named_parameters()):
        if a.requires_grad:
            assert a.grad is not None and a.grad.abs().sum()>0,name
            torch.testing.assert_close(a.grad,b.grad,rtol=0,atol=0)
        else:
            assert a.grad is None and b.grad is None


def test_disabled_extensions_keep_common_initialization_and_strict_state():
    torch.manual_seed(55)
    legacy=build_projector(config(),8,16)
    rng=torch.get_rng_state()
    torch.manual_seed(55)
    normalized=build_projector(config('projector.params.router_value_normalization=true'),8,16)
    assert torch.equal(rng,torch.get_rng_state())
    for name,v in legacy.state_dict().items():
        torch.testing.assert_close(v,normalized.state_dict()[name],rtol=0,atol=0)
    legacy.load_state_dict(legacy.state_dict(),strict=True)
    assert not any('value_norm' in k or 'value_projection' in k for k in legacy.state_dict())
