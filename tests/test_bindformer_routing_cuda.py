"""Real-width/grid BF16 acceptance for each newly launched spatial budget."""
import os
import pytest
import torch
from src.config import load_config
from src.models.projectors.budget import resolve_budget
from src.models.projectors.registry import build_projector
from src.training.reproducibility import configure_reproducibility

DEVICE=os.getenv('PROJECTOR_TEST_DEVICE')
pytestmark=pytest.mark.skipif(DEVICE is None,reason='Set PROJECTOR_TEST_DEVICE explicitly')


@pytest.mark.parametrize('ratio,adaptive,normalize', [
    (8,False,True),(16,False,True),(32,False,True),(64,False,True),
    (8,True,True),(16,True,True),(32,True,True),(64,True,True),
    (32,False,False),(32,True,False),
])
def test_full_grid_spatial_bf16_checkpoint_gradients(ratio,adaptive,normalize):
    device=torch.device(DEVICE);torch.cuda.set_device(device)
    aggregation='adaptive' if adaptive else 'mean'
    c=load_config(['configs/projector/bindformer3d.yaml'],[
        f'projector.target_compression_ratio={ratio}','training.deterministic=true',
        'projector.params.within_slot_sa=true','projector.params.local_reads_global=true',
        'projector.params.fixed_spatial_regions=true',
        f'projector.params.spatial_{aggregation}_aggregation=true',
        f'projector.params.router_value_normalization={str(normalize).lower()}',
        'projector.params.return_auxiliary=true'])
    configure_reproducibility(c)
    resolve_budget(c.projector,(39,15,15),(156,240,240))
    model=build_projector(c,768,4096).to(device)
    x=torch.randn(8775,768,device=device,dtype=torch.bfloat16,requires_grad=True)
    coords=torch.cartesian_prod(torch.arange(39),torch.arange(15),torch.arange(15))
    meta={'grid_dhw':[39,15,15],'shape_dhw':[156,240,240],
          'bounded_before_pad':{'spacing_xyz':[1.,1.,1.]},
          'array_zyx_to_dhw':[0,1,2],'sequence':'T1C',
          'oriented':{'direction':[-1,0,0,0,1,0,0,0,-1]}}
    try:
        with torch.autocast('cuda',dtype=torch.bfloat16):
            result=model(x,coords,meta)
            loss=result.embeddings.float().square().mean()
        loss.backward()
        assert result.embeddings.shape==(c.projector.resolved_budget.expected_output_tokens,4096)
        assert result.embeddings.dtype==torch.bfloat16
        assert torch.isfinite(x.grad).all() and x.grad.abs().sum()>0
        for name,p in model.named_parameters():
            if p.requires_grad:
                assert p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum()>0,name
            else:assert p.grad is None
        aux=result.auxiliary
        assert (aux.assignments.sum(-1)==1).all()
        torch.testing.assert_close(aux.evidence_weights.sum(0),torch.ones_like(aux.evidence_weights.sum(0)))
        assert not aux.evidence_weights[aux.assignments==0].count_nonzero()
        assert model.concept_memory.embeddings.grad is None
    finally:
        torch.use_deterministic_algorithms(False)
