import copy
import io
import itertools
import math
import pytest
import torch
from torch import nn
from src.config import load_config
from src.models.projectors.registry import PROJECTORS, build_projector
from src.models.projectors.specs import DEFAULTS
from src.models.projectors.common import QueryBlock
from src.models.projectors.geometry import grid_groups, partition
from src.models.projectors.mixer import WindowMixerBlock
from src.models.projectors.adaptive_merge import connected_ward


def config(name, target=8, **params):
    overrides = [f'projector.name={name}', 'projector.hidden_size=16']
    defaults = DEFAULTS[name]
    if name in ('bindformer3d', 'bindformer_a4', 'bindformer_adirect', 'bindformer_adirect_localread'):
        # Generic projector tests have no medical context; dedicated BindFormer
        # tests exercise real sequence/geometry/knowledge instead.
        params.setdefault('inner_dim',8)
        params.setdefault('position_mode','none')
        params.setdefault('sequence_condition',False)
        if name == 'bindformer3d':
            params.setdefault('knowledge_mode','none')
    if name == 'bindformer_adirect_detail':
        params.setdefault('inner_dim', 8)
    if 'num_tokens' in defaults: params.setdefault('num_tokens', target)
    if 'num_heads' in defaults: params.setdefault('num_heads', 2)
    if 'depth' in defaults: params.setdefault('depth', 2)
    if 'cache_assignments' in defaults: params.setdefault('cache_assignments', False)
    overrides += [f'projector.params.{k}={str(v).lower() if isinstance(v,bool) else v}' for k,v in params.items()]
    return load_config(overrides=overrides)


def volume(grid=(3,4,2), channels=8):
    return (torch.randn(math.prod(grid), channels, requires_grad=True),
            torch.cartesian_prod(*(torch.arange(n) for n in grid)), {'grid_dhw':list(grid), 'id':'case-a'})


@pytest.mark.parametrize('name', PROJECTORS)
@pytest.mark.parametrize('target', [1,8,24])
def test_gradients_update_and_state_roundtrip(name, target):
    torch.manual_seed(11)
    if name in ('bindformer3d', 'bindformer_a4', 'bindformer_adirect',
                'bindformer_adirect_localread', 'bindformer_adirect_detail') and target<5:
        with pytest.raises(ValueError,match='complete role bundle'):
            config(name,target)
        return
    c = config(name, target)
    model = build_projector(c, 8, 6)
    x, coords, meta = volume()
    output = model(x, coords, meta).validate(6)
    expected = len(x) if name=='mlp2x_gelu' else 4 if name=='concat3d' else target
    assert output.output_tokens == expected
    output.embeddings.square().sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all() and x.grad.abs().sum()>0
    if name!='token_select3d': assert (x.grad.abs().sum(-1)>0).all()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    before = [p.clone() for p in model.parameters()]
    torch.optim.SGD(model.parameters(),lr=.01).step()
    assert any(not torch.equal(a,b) for a,b in zip(before,model.parameters()))
    saved = io.BytesIO(); torch.save(model.state_dict(),saved); saved.seek(0)
    other = build_projector(c,8,6); other.load_state_dict(torch.load(saved,weights_only=True),strict=True)
    torch.testing.assert_close(model(x,coords,meta).embeddings,other(x,coords,meta).embeddings,rtol=0,atol=0)


def test_legacy_mlp_exact_state_keys_and_values():
    c = config('mlp2x_gelu'); assert dict(c.projector.params)=={}
    module=build_projector(c,8,6); x,coords,meta=volume()
    assert list(module.state_dict())==['layers.0.weight','layers.0.bias','layers.2.weight','layers.2.bias']
    expected=nn.functional.linear(nn.functional.gelu(nn.functional.linear(x,module.layers[0].weight,
               module.layers[0].bias)),module.layers[2].weight,module.layers[2].bias)
    torch.testing.assert_close(module(x,coords,meta).embeddings,expected,rtol=0,atol=0)


def test_concat_slot_order_zero_padding_and_full_coverage():
    grid=(3,3,3); c=config('concat3d',head_type='linear')
    model=build_projector(c,1,8)
    with torch.no_grad(): model.head.weight.copy_(torch.eye(8)); model.head.bias.zero_()
    x=torch.arange(1,28.).reshape(-1,1).requires_grad_()
    _,coords,meta=volume(grid,1)
    out=model(x,coords,meta)
    expected=[]
    for d,h,w in itertools.product(range(0,3,2),repeat=3):
        row=[]
        for dd,hh,ww in itertools.product(range(2),repeat=3):
            i,j,k=d+dd,h+hh,w+ww
            row.append(float((i*3+j)*3+k+1) if max(i,j,k)<3 else 0.)
        expected.append(row)
    torch.testing.assert_close(out.embeddings,torch.tensor(expected))
    assert out.metadata['projector']['empty_slots']==37
    out.embeddings.sum().backward(); torch.testing.assert_close(x.grad,torch.ones_like(x))


@pytest.mark.parametrize('latent', [False,True])
def test_attention_matches_explicit_softmax_and_padding_is_invisible(latent):
    torch.manual_seed(12); layer=QueryBlock(8,2,latent_kv=latent).double()
    q=torch.randn(2,3,8,dtype=torch.double); x=torch.randn(2,5,8,dtype=torch.double)
    valid=torch.tensor([[1,1,1,0,0],[1,1,1,1,0]],dtype=torch.bool)
    qn=layer.norm_queries(q); xn=layer.norm_visual(x)
    memory=torch.cat((xn,qn),1) if latent else xn
    mask=torch.cat((valid,torch.ones(2,3,dtype=torch.bool)),1) if latent else valid
    qq=layer.to_q(qn).reshape(2,3,2,4).transpose(1,2)
    kk,vv=[v.reshape(2,-1,2,4).transpose(1,2) for v in layer.to_kv(memory).chunk(2,-1)]
    prob=(qq@kk.transpose(-1,-2)/2).masked_fill(~mask[:,None,None,:],float('-inf')).softmax(-1)
    expected=q+layer.to_out((prob@vv).transpose(1,2).reshape(2,3,8))
    expected=expected+layer.ffn(expected)
    torch.testing.assert_close(layer(q,x,valid),expected,rtol=1e-12,atol=1e-12)
    changed=x.clone(); changed[~valid]=torch.randn_like(changed[~valid])*100
    torch.testing.assert_close(layer(q,x,valid),layer(q,changed,valid),rtol=0,atol=0)


def test_nv_queries_independent_but_flamingo_queries_interact():
    q=torch.randn(1,3,8); x=torch.randn(1,9,8); changed=q.clone(); changed[:,0]+=torch.arange(8.)
    for latent in (False,True):
        layer=QueryBlock(8,2,latent_kv=latent)
        a,b=layer(q,x),layer(changed,x)
        assert torch.equal(a[:,1:],b[:,1:]) is (not latent)


def test_nv_block_matches_local_official_implementation():
    from neurovfm.models.perceiver import PerceiverAttention
    official=PerceiverAttention(dim=8,num_heads=2)
    local=QueryBlock(8,2); local.load_state_dict(official.state_dict(),strict=True)
    q=torch.randn(2,3,8,requires_grad=True); x=torch.randn(2,11,8,requires_grad=True)
    a,b=official(q,x),local(q,x)
    torch.testing.assert_close(a,b,rtol=0,atol=0)
    grad_a=torch.autograd.grad(a.square().sum(),(q,x),retain_graph=True)
    grad_b=torch.autograd.grad(b.square().sum(),(q,x))
    for first,second in zip(grad_a,grad_b): torch.testing.assert_close(first,second,rtol=0,atol=0)


def test_local_packer_has_no_cross_cell_attention():
    model=build_projector(config('local_packer3d'),8,6); x,coords,meta=volume()
    _,groups=partition((3,4,2),meta,8,'nearest')
    before=model(x,coords,meta).embeddings
    changed=x.detach().clone(); changed[groups[0]]+=torch.arange(8.)
    after=model(changed,coords,meta).embeddings
    torch.testing.assert_close(before[1:],after[1:],rtol=0,atol=0)
    assert not torch.equal(before[0],after[0])


def test_shifted_mixer_does_not_wrap_opposite_faces():
    layer=WindowMixerBlock(8,(2,2,2),2,2,True)
    x=torch.randn(5,5,5,8); changed=x.clone(); changed[0]+=torch.arange(8.)
    torch.testing.assert_close(layer(x)[-1],layer(changed)[-1],rtol=0,atol=0)


def test_ward_regions_connected_conservative_and_deterministic():
    grid=(3,4,2); x,coords,meta=volume(grid)
    for target in (1,7,24):
        groups=connected_ward(x,grid,target)
        repeated=connected_ward(x,grid,target)
        assert len(groups)==target
        assert all(torch.equal(a,b) for a,b in zip(groups,repeated))
        assert torch.equal(torch.cat(groups).sort().values,torch.arange(len(x)))
        for group in groups:
            todo=set(group.tolist()); frontier=[todo.pop()]
            while frontier:
                node=frontier.pop()
                next_nodes={j for j in todo if int((coords[node]-coords[j]).abs().sum())==1}
                todo-=next_nodes; frontier.extend(next_nodes)
            assert not todo
    # In one dimension identical neighbours must merge before a discontinuity.
    line=torch.tensor([[0.],[0.],[10.],[10.]])
    assert [g.tolist() for g in connected_ward(line,(1,1,4),2,False)]==[[0,1],[2,3]]


def test_merge_cache_identity_and_stage_b_never_reuses_it(tmp_path):
    c=config('adaptive_merge3d',cache_assignments=True,assignment_cache_dir=str(tmp_path))
    model=build_projector(c,8,6); x,coords,meta=volume(); x=x.detach()
    first=model(x,coords,meta); second=model(x,coords,meta)
    assert not first.metadata['projector']['assignment_cache_hit']
    assert second.metadata['projector']['assignment_cache_hit']
    assert not model(x+1,coords,meta).metadata['projector']['assignment_cache_hit']
    model.stage='b_lora'; x.requires_grad_(True)
    out=model(x,coords,meta); assert not out.metadata['projector']['assignment_cache_enabled']
    out.embeddings.square().sum().backward(); assert (x.grad.abs().sum(1)>0).all()


@pytest.mark.parametrize('strategy', ['random','diversity'])
def test_selection_unique_stable_and_gradients_only_to_selected(strategy):
    model=build_projector(config('token_select3d',strategy=strategy),8,6)
    x,coords,meta=volume(); x=torch.zeros_like(x,requires_grad=True)
    out=model(x,coords,meta); indices=out.metadata['projector']['selected_indices']
    assert len(set(indices))==8 and indices==sorted(indices)
    torch.rand(20)  # Process RNG consumption does not change per-sample routing.
    assert indices==model(x,coords,meta).metadata['projector']['selected_indices']
    out.embeddings.sum().backward()
    assert (x.grad.abs().sum(-1)>0).nonzero().flatten().tolist()==indices
    if strategy=='random':
        assert indices!=model(x,coords,{**meta,'id':'case-b'}).metadata['projector']['selected_indices']


@pytest.mark.parametrize('name', ['local_packer3d','attention_pool','perceiver_nv','perceiver_flamingo'])
def test_checkpointed_blocks_preserve_outputs_and_gradients(name):
    a=build_projector(config(name,checkpoint_blocks=False),8,6); b=copy.deepcopy(a)
    b.params['checkpoint_blocks']=True
    x,coords,meta=volume(); y=x.detach().clone().requires_grad_(True)
    first=a(x,coords,meta).embeddings; second=b(y,coords,meta).embeddings
    torch.testing.assert_close(first,second,rtol=0,atol=0)
    first.square().sum().backward(); second.square().sum().backward()
    torch.testing.assert_close(x.grad,y.grad,rtol=0,atol=0)
    for p,q in zip(a.parameters(),b.parameters()):torch.testing.assert_close(p.grad,q.grad,rtol=0,atol=0)


@pytest.mark.parametrize('name', [n for n in PROJECTORS if n not in ('mlp2x_gelu','concat3d')])
def test_oversized_budget_rejected(name):
    model=build_projector(config(name,target=25),8,6)
    with pytest.raises(ValueError,match='budget'): model(*volume())


@pytest.mark.parametrize('name', ['convpool3d','mixer3d','local_packer3d','attention_pool','perceiver_nv','perceiver_flamingo'])
def test_inner_width_independent_of_input_channels_and_token_budget(name):
    model=build_projector(config(name,inner_dim=12),8,6)
    x,coords,meta=volume(); out=model(x,coords,meta)
    assert out.embeddings.shape==(8,6)
    out.embeddings.square().sum().backward()
    assert model.adapt.weight.shape==(12,8)
    assert model.adapt.weight.grad.abs().sum()>0 and x.grad.abs().sum()>0
