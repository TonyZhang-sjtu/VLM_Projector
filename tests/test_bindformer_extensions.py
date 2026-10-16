import copy
import itertools
from fractions import Fraction
from pathlib import Path
import pytest
import torch
from src.config import load_config
from src.models.projectors.registry import build_projector
from src.models.projectors.bindformer.budget import allocate_tokens,role_counts,ROLES
from src.models.projectors.bindformer.geometry import patch_geometry,support_statistics,SupportGeometryEncoder
from src.models.projectors.bindformer.interactions import BundleInteractions
from src.models.projectors.bindformer.reconstruction import FeatureReconstructor,stratified_indices
from test_bindformer_modules import config,volume


def geometric_volume(channels=8):
    x,coords,meta=volume(channels)
    meta.update(bounded_before_pad={'size_xyz':[32,64,12],
        'direction':[-1,0,0,0,1,0,0,0,-1],
        'affine_xyz_to_lps':[[-1,0,0,31],[0,1,0,-20],[0,0,-1,11],[0,0,0,1]]},
        pad_before_xyz=[0,0,0],pad_after_xyz=[0,0,0])
    return x,coords,meta


@pytest.mark.parametrize('counts',[[1,1,1,1],[2,2,2,2],[1,2,3,1],[4,1,1,2]])
def test_variable_query_integer_oracle(counts):
    q=sum(counts)
    for m in range(q+1,110):
        for fraction in (0,.125,.25,.8):
            b=allocate_tokens(m,fraction,role_query_counts=counts)
            possibilities=[(abs(Fraction(m-q*k,m)-Fraction(str(fraction))),k)
                           for k in range(1,(m-int(fraction>0))//q+1)]
            assert b['num_slots']==min(possibilities)[1]
            assert b['global_tokens']+q*b['num_slots']==m
    b=allocate_tokens(275,.25,role_query_counts=counts)
    if counts==[2,2,2,2]:assert (b['global_tokens'],b['num_slots'])==(67,26)


@pytest.mark.parametrize('override',['role_query_counts.signal=0','role_query_counts.signal=true',
    'role_query_counts.extra=1','feature_num_heads=3','feature_loss_weight=.nan'])
def test_extension_config_rejects_invalid_values(override):
    with pytest.raises(ValueError):config('projector.params.'+override)


def test_multirole_queries_layout_bias_and_distinct_gradients():
    c=config('projector.params.num_tokens=20','projector.params.role_query_counts.signal=3',
             'projector.params.role_query_counts.finding=2','projector.params.support_beta=[0,1,2,3]')
    p=build_projector(c,8,16);x,coords,meta=volume()
    out=p(x,coords,meta)
    assert out.embeddings.shape==(20,16)
    assert (p.allocation['global_tokens'],p.allocation['num_slots'])==(6,2)
    assert out.auxiliary.role_tokens.shape==(2,7,16)
    assert p.reader.support_beta.tolist()==[0,0,1,2,2,2,3]
    slots=out.auxiliary.slots
    concepts,_=p.concept_memory(slots,'T2W')
    queries=p.role_queries(slots,concepts)
    assert not torch.equal(queries[:,0],queries[:,1])
    assert not torch.equal(queries[:,3],queries[:,4])
    out.embeddings.square().mean().backward()
    assert p.role_queries.subquery_embedding.grad.abs().sum(-1).gt(0).all()


def test_analytic_geometry_units_padding_and_gradients():
    coords=torch.cartesian_prod(torch.arange(2),torch.arange(2),torch.arange(2))
    meta={'grid_dhw':[2,2,2],'array_zyx_to_dhw':[0,1,2],
          'pad_before_xyz':[0,0,0],'pad_after_xyz':[0,0,0],
          'bounded_before_pad':{'size_xyz':[32,32,8],'direction':[1,0,0,0,1,0,0,0,1],
            'affine_xyz_to_lps':[[2,0,0,10],[0,3,0,20],[0,0,4,30],[0,0,0,1]]}}
    grid=patch_geometry(coords,meta,'grid')
    a=torch.ones(8,1,requires_grad=True)
    stats,usable=support_statistics(a,grid)
    torch.testing.assert_close(stats[0,:3],torch.zeros(3))
    torch.testing.assert_close(stats[0,3:9],torch.tensor([.25,0,0,.25,0,.25]))
    torch.testing.assert_close(stats[0,9],torch.tensor(2.).log())
    mm=patch_geometry(coords,meta,'mm')
    torch.testing.assert_close(mm.centers[0],torch.tensor([25.,42.5,36.]))
    assert mm.volumes.sum()==32*32*8*24
    shifted=copy.deepcopy(meta);shifted['bounded_before_pad']['affine_xyz_to_lps'][0][3]+=100
    other=patch_geometry(coords,shifted,'mm')
    torch.testing.assert_close(support_statistics(a,mm)[0],support_statistics(a,other)[0])
    partial=copy.deepcopy(meta);partial['bounded_before_pad']['size_xyz']=[30,31,7]
    partial.update(pad_before_xyz=[1,0,0],pad_after_xyz=[1,1,1])
    voxel=patch_geometry(coords,partial,'voxel')
    assert voxel.volumes.sum()==30*31*7
    grid_partial=patch_geometry(coords,partial,'grid')
    torch.testing.assert_close(grid_partial.volumes.sum(),torch.tensor(1.))
    encoder=SupportGeometryEncoder(16)
    encoded=encoder(a,grid)
    assert not encoded[:,0].count_nonzero() and not encoded[:,2].count_nonzero()
    encoded.square().sum().backward()
    assert torch.isfinite(a.grad).all() and a.grad.abs().sum()>0
    empty=encoder(torch.zeros_like(a),grid)
    assert not empty.count_nonzero()


def test_geometry_without_spacing_and_pure_padding():
    coords=torch.tensor([[0,0,0],[0,0,1]])
    meta={'grid_dhw':[1,1,2],'array_zyx_to_dhw':[0,1,2],
          'pad_before_xyz':[16,0,0],'pad_after_xyz':[0,0,0],
          'bounded_before_pad':{'size_xyz':[16,16,4],'direction':[1,0,0,0,1,0,0,0,1]}}
    grid=patch_geometry(coords,meta)
    assert grid.valid_fraction.tolist()==[0,1]
    assert grid.volumes.tolist()==[0,1]
    assert torch.isfinite(grid.centers).all()
    with pytest.raises(ValueError,match='affine'):patch_geometry(coords,meta,'mm')
    missing=copy.deepcopy(meta);del missing['pad_after_xyz']
    with pytest.raises(ValueError,match='padding'):patch_geometry(coords,missing)


@pytest.mark.parametrize('order',list(itertools.permutations(range(3))))
def test_geometry_axis_permutations_under_autocast(order):
    coords=torch.tensor([[2,1,0]])
    # Oblique orthogonal direction, non-unit spacing and translated cropped origin.
    direction=[[.6,-.8,0],[.8,.6,0],[0,0,1]]
    affine=[[1.2,-2.4,0,10],[1.6,1.8,0,20],[0,0,4,30],[0,0,0,1]]
    meta={'grid_dhw':[8,2,2],'array_zyx_to_dhw':list(order),
          'pad_before_xyz':[0,0,0],'pad_after_xyz':[0,0,0],
          'bounded_before_pad':{'size_xyz':[32,32,32],'direction':direction,
                                'affine_xyz_to_lps':affine}}
    expected_zyx=[0.,0.,0.]
    for axis,(index,width) in enumerate(zip([2,1,0],[4,16,16])):
        expected_zyx[order[axis]]=index*width+(width-1)/2
    xyz=expected_zyx[::-1]
    expected=torch.tensor([sum(row[j]*xyz[j] for j in range(3))+row[3] for row in affine[:3]])
    with torch.autocast('cpu',dtype=torch.bfloat16):
        geometry=patch_geometry(coords,meta,'mm')
    assert geometry.centers.dtype==torch.float32
    torch.testing.assert_close(geometry.centers[0],expected)


def test_interactions_stay_within_slot_and_gated_global_context():
    torch.manual_seed(50)
    bundles=torch.randn(3,7,16);global_tokens=torch.randn(5,16)
    identity=BundleInteractions()
    assert identity(bundles,global_tokens) is bundles
    sa=BundleInteractions(16,4,within=True)
    before=sa(bundles,global_tokens)
    changed=bundles.clone();changed[0]*=3
    after=sa(changed,global_tokens)
    torch.testing.assert_close(before[1:],after[1:],rtol=0,atol=0)
    ca=BundleInteractions(16,4,cross=True)
    assert not torch.allclose(ca(bundles,global_tokens),ca(bundles,torch.zeros_like(global_tokens)))
    with torch.no_grad():ca.cross[0].gate.bias.fill_(-1000)
    torch.testing.assert_close(ca(bundles,global_tokens),bundles,rtol=0,atol=0)
    with pytest.raises(ValueError,match='global tokens'):ca(bundles,global_tokens[:0])


@pytest.mark.parametrize('overrides',[
    ['geometry_enabled=true'],['geometry_enabled=true','geometry_space=mm'],
    ['within_slot_sa=true'],['local_reads_global=true'],
    ['geometry_enabled=true','within_slot_sa=true','local_reads_global=true','role_query_counts.signal=2']])
def test_structural_extensions_ce_path_gradients_and_checkpointing(overrides):
    c=config('projector.params.num_tokens=20',*['projector.params.'+x for x in overrides])
    m=build_projector(c,8,16)
    x,coords,meta=geometric_volume();x.requires_grad_()
    out=m(x,coords,meta)
    assert out.embeddings.shape==(20,16)
    out.embeddings.square().sum().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum()>0
               for p in m.parameters())
    other=build_projector(c,8,16);other.load_state_dict(m.state_dict())
    torch.testing.assert_close(other(x.detach(),coords,meta).embeddings,out.embeddings,rtol=0,atol=0)


def test_reconstruction_sampling_and_final_z_gradient_only():
    torch.manual_seed(61)
    head=FeatureReconstructor(16,8,dim=16,heads=4,samples=9,chunk_size=3,checkpoint_blocks=True)
    x,coords,meta=geometric_volume()
    target=x.clone().requires_grad_();z=torch.randn(12,16,requires_grad=True)
    before=torch.get_rng_state()
    a=stratified_indices(coords,torch.ones(24),9,100)
    b=stratified_indices(coords,torch.ones(24),9,100)
    assert len(set(a.tolist()))==9 and torch.equal(a,b)
    assert torch.equal(before,torch.get_rng_state())
    value=head.objective(z,coords,meta,'T2W',target,100)
    assert value['patches']==9
    value['loss'].backward()
    assert target.grad is None
    assert z.grad.abs().sum()>0 and torch.isfinite(z.grad).all()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum()>0
               for p in head.parameters())
    centers=patch_geometry(coords,meta).centers[a]
    assert not torch.allclose(head.predict(z,centers,'T2W'),head.predict(torch.zeros_like(z),centers,'T2W'))


@pytest.mark.parametrize('knowledge',['name_definition','none'])
def test_legacy_reference_output_gradients_parameters_and_rng(knowledge):
    path=Path('runs/bindformer_extensions_acceptance')/f'legacy_{knowledge}.pt'
    if not path.exists():pytest.skip('Optional pre-edit acceptance reference not installed')
    old=torch.load(path,weights_only=False)
    torch.manual_seed(20261016)
    c=config(f'projector.params.knowledge_mode={knowledge}')
    m=build_projector(c,8,16)
    x,coords,meta=volume();x.requires_grad_()
    for key,value in m.state_dict().items():torch.testing.assert_close(value,old['state'][key],rtol=0,atol=0)
    assert set(m.state_dict())==set(old['state'])
    y=m(x,coords,meta).embeddings;y.square().sum().backward()
    torch.testing.assert_close(y,old['y'],rtol=0,atol=0)
    torch.testing.assert_close(x.grad,old['xgrad'],rtol=0,atol=0)
    for n,p in m.named_parameters():torch.testing.assert_close(p.grad,old['grads'][n],rtol=0,atol=0)
    assert torch.equal(torch.get_rng_state(),old['rng'])
