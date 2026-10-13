"""Protect semantic contracts that shape/gradient-only smoke tests do not cover."""
import math
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from torch import nn
from src.config import load_config,validate
from src.models.multimodal import MultimodalModel
from src.models.projectors.base import ProjectedVolume
from src.models.projectors.geometry import choose_grid
from src.models.projectors.adaptive_merge import connected_ward
from src.models.projectors.registry import build_projector


def test_fixed_budget_guard_rejects_natural_concat_and_nearest_rounding():
    for name,target,grid in [('concat3d',256,(7,8,8)),('avgpool3d',250,(7,8,8))]:
        overrides=[f'projector.name={name}',f'training.required_visual_tokens={target}',
                   'projector.hidden_size=8']
        if name!='concat3d':overrides.append(f'projector.params.num_tokens={target}')
        c=load_config(overrides=overrides)
        model=MultimodalModel.__new__(MultimodalModel);nn.Module.__init__(model)
        model.config=c;model.decoder=SimpleNamespace(hidden_size=8)
        x=torch.randn(math.prod(grid),4)
        coords=torch.cartesian_prod(*(torch.arange(n) for n in grid))
        model.encoder=lambda volumes:SimpleNamespace(features=[x],coords=[coords],metadata=[{'grid_dhw':list(grid)}])
        model.projector=build_projector(c,4,8)
        actual=model.projector(x,coords,{'grid_dhw':list(grid)}).output_tokens
        assert actual==(64 if name=='concat3d' else 252)
        with pytest.raises(ValueError,match=f'required={target}, actual={actual}'):
            model.project_volumes([object()])


def test_fixed_budget_configuration_and_legacy_opt_out():
    assert load_config().training.required_visual_tokens is None
    legacy=load_config();del legacy.training.required_visual_tokens
    validate(legacy)
    for value in ('true','0','-2','1.5'):
        with pytest.raises(ValueError,match='required_visual_tokens'):
            load_config(overrides=[f'training.required_visual_tokens={value}'])
    with pytest.raises(ValueError,match='differs'):
        load_config(overrides=['projector.name=attention_pool','training.required_visual_tokens=128'])
    out=ProjectedVolume(torch.zeros(7,4),10,7,{})
    assert out.validate(4,7) is out
    assert out.validate(4) is out
    with pytest.raises(ValueError,match='budget mismatch'):out.validate(4,8)


def test_actual_budgets_are_exact_for_primary_example_grids():
    for grid in ((39,15,15),(19,7,7),(18,7,7),(7,8,8)):
        for m in (64,128,256):
            out=choose_grid(grid,m,tuple(float(n) for n in grid),'exact')
            assert math.prod(out)==m
    assert math.prod(choose_grid((7,8,8),250,(134.4,230.,230.),'nearest'))==252


@pytest.mark.parametrize('normalize',[False,True])
def test_ward_matches_bruteforce_minimum_cost_adjacent_merges(normalize):
    # Independent O(N^3) oracle on a tiny grid; production uses a lazy edge heap.
    torch.manual_seed(91);grid=(2,2,3);x=torch.randn(12,5)
    coords=torch.cartesian_prod(*(torch.arange(n) for n in grid)).numpy()
    scores=x.numpy().astype(np.float64)
    if normalize:scores/=np.maximum(np.linalg.norm(scores,axis=1,keepdims=True),1e-12)
    regions={i:[i] for i in range(12)}
    for node in range(12,20):
        choices=[]
        keys=sorted(regions)
        for ai,a in enumerate(keys):
            for b in keys[ai+1:]:
                ra,rb=regions[a],regions[b]
                adjacent=any(np.abs(coords[i]-coords[j]).sum()==1 for i in ra for j in rb)
                if not adjacent:continue
                delta=scores[ra].mean(0)-scores[rb].mean(0)
                cost=len(ra)*len(rb)/(len(ra)+len(rb))*float(delta@delta)
                choices.append((cost,a,b))
        _,a,b=min(choices);regions[node]=regions.pop(a)+regions.pop(b)
    expected=sorted((sorted(g) for g in regions.values()),key=lambda g:g[0])
    assert [g.tolist() for g in connected_ward(x,grid,4,normalize)]==expected
