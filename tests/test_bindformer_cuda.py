"""Explicit-device CUDA acceptance for every real BraTS compression budget."""
import os
import pytest
import torch
from src.config import load_config
from src.models.projectors.budget import resolve_budget
from src.models.projectors.registry import build_projector
from src.training.reproducibility import configure_reproducibility

DEVICE=os.getenv('PROJECTOR_TEST_DEVICE')
pytestmark=pytest.mark.skipif(DEVICE is None,reason='Set PROJECTOR_TEST_DEVICE explicitly')


@pytest.mark.parametrize('ratio,expected',[(4,2194),(8,1097),(16,549),(32,275),(64,138)])
@pytest.mark.parametrize('deterministic',[True,False])
def test_real_grid_bf16_all_ratios_checkpoint_and_gradients(ratio,expected,deterministic):
    device=torch.device(DEVICE);torch.cuda.set_device(device)
    config=load_config(['configs/projector/bindformer3d.yaml'],[
        f'projector.target_compression_ratio={ratio}',f'training.deterministic={str(deterministic).lower()}'])
    configure_reproducibility(config)
    resolve_budget(config.projector,(39,15,15),(156,240,240))
    model=build_projector(config,768,4096).to(device)
    x=torch.randn(8775,768,device=device,dtype=torch.bfloat16,requires_grad=True)
    coords=torch.cartesian_prod(torch.arange(39),torch.arange(15),torch.arange(15))
    meta={'grid_dhw':[39,15,15],'shape_dhw':[156,240,240],
          'bounded_before_pad':{'spacing_xyz':[1.,1.,1.]},
          'array_zyx_to_dhw':[0,1,2],'sequence':'T1C',
          'oriented':{'direction':[-1,0,0,0,1,0,0,0,-1]}}
    with torch.autocast('cuda',dtype=torch.bfloat16):
        result=model(x,coords,meta)
        loss=result.embeddings.float().square().mean()
    loss.backward()
    assert result.embeddings.shape==(expected,4096) and result.embeddings.dtype==torch.bfloat16
    assert torch.isfinite(x.grad).all() and x.grad.abs().sum()>0
    assert all(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum()>0
               for p in model.parameters())
    assert model.concept_memory.embeddings.grad is None
    assert result.metadata['projector']['active_concepts']==120
    model.zero_grad(set_to_none=True)
    torch.use_deterministic_algorithms(False)
