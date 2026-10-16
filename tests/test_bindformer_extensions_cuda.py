"""Full-size BF16 extension acceptance; explicit GPU opt-in only."""
import os
import pytest
import torch
from src.config import load_config
from src.models.projectors.budget import resolve_budget
from src.models.projectors.registry import build_projector
from src.training.reproducibility import configure_reproducibility

DEVICE=os.getenv('PROJECTOR_TEST_DEVICE')
pytestmark=pytest.mark.skipif(DEVICE is None,reason='Set PROJECTOR_TEST_DEVICE explicitly')


@pytest.mark.parametrize('ratio,expected',[(4,2194),(8,1097),(16,549),(32,275)])
@pytest.mark.parametrize('deterministic',[True,False])
def test_full_grid_all_extensions_bf16(ratio,expected,deterministic):
    device=torch.device(DEVICE);torch.cuda.set_device(device)
    c=load_config(['configs/projector/bindformer3d.yaml'],[
        f'projector.target_compression_ratio={ratio}',
        f'training.deterministic={str(deterministic).lower()}',
        'projector.params.geometry_enabled=true','projector.params.geometry_space=mm',
        'projector.params.within_slot_sa=true','projector.params.local_reads_global=true',
        'projector.params.role_query_counts.signal=2',
        'projector.params.feature_loss_weight=.1','projector.params.feature_target=frozen_input'])
    configure_reproducibility(c)
    resolve_budget(c.projector,(39,15,15),(156,240,240))
    model=build_projector(c,768,4096).to(device)
    x=torch.randn(8775,768,device=device,dtype=torch.bfloat16,requires_grad=True)
    coords=torch.cartesian_prod(torch.arange(39),torch.arange(15),torch.arange(15))
    meta={'grid_dhw':[39,15,15],'shape_dhw':[156,240,240],
          'bounded_before_pad':{'size_xyz':[240,240,155],'spacing_xyz':[1.,1.,1.],
              'direction':[-1,0,0,0,1,0,0,0,-1],
              'affine_xyz_to_lps':[[-1,0,0,20],[0,1,0,30],[0,0,-1,40],[0,0,0,1]]},
          'pad_before_xyz':[0,0,0],'pad_after_xyz':[0,0,1],
          'array_zyx_to_dhw':[0,1,2],'sequence':'T1C',
          'oriented':{'direction':[-1,0,0,0,1,0,0,0,-1]}}
    try:
        with torch.autocast('cuda',dtype=torch.bfloat16):
            result=model(x,coords,meta)
            aux=model.feature_reconstructor.objective(result.embeddings,coords,meta,'T1C',x.detach(),42)
            loss=result.embeddings.float().square().mean()+.1*aux['loss']
        loss.backward()
        assert result.embeddings.shape==(expected,4096) and result.embeddings.dtype==torch.bfloat16
        assert aux['patches']==512 and torch.isfinite(aux['loss'])
        assert torch.isfinite(x.grad).all() and x.grad.abs().sum()>0
        assert all(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum()>0
                   for p in model.parameters())
        assert model.concept_memory.embeddings.grad is None
    finally:
        torch.use_deterministic_algorithms(False)
