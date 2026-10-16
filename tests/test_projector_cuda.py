"""Opt-in CUDA kernel checks; never select a user's GPU implicitly."""
import os
import pytest
import torch
from src.config import load_config
from src.models.projectors.registry import PROJECTORS, build_projector
from src.models.projectors.geometry import choose_grid

DEVICE=os.getenv('PROJECTOR_TEST_DEVICE')
pytestmark=pytest.mark.skipif(DEVICE is None,reason='Set PROJECTOR_TEST_DEVICE to an available CUDA device')


@pytest.mark.parametrize('name',PROJECTORS)
@pytest.mark.parametrize('grid',[(7,8,8),(19,7,7),(39,15,15)])
def test_real_grid_bf16_fast_kernels_and_gradient_paths(name,grid):
    device=torch.device(DEVICE); torch.cuda.set_device(device)
    torch.use_deterministic_algorithms(False)
    c=load_config(overrides=[f'projector.name={name}', 'training.stage=b_full',
                             'training.feature_cache=false','training.encoder_lr=1e-5'])
    model=build_projector(c,768,4096).to(device)
    n=grid[0]*grid[1]*grid[2]
    x=torch.randn(n,768,device=device,dtype=torch.bfloat16,requires_grad=True)
    coords=torch.cartesian_prod(*(torch.arange(v) for v in grid))
    with torch.autocast('cuda',dtype=torch.bfloat16):
        out=model(x,coords,{'grid_dhw':list(grid),'id':'cuda-smoke','sequence':'T2W',
                           'array_zyx_to_dhw':[0,1,2],
                           'oriented':{'direction':[-1,0,0,0,1,0,0,0,-1]}})
    loss=out.embeddings.float().square().mean(); loss.backward()
    assert out.embeddings.dtype==torch.bfloat16
    assert torch.isfinite(x.grad).all() and x.grad.abs().sum()>0
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    if name not in ('mlp2x_gelu','concat3d'):
        expected=256
        if name in ('avgpool3d','convpool3d','mixer3d','local_packer3d'):
            dims=choose_grid(grid,256,tuple(float(v) for v in grid),'nearest')
            expected=dims[0]*dims[1]*dims[2]
        assert out.output_tokens==expected
