import itertools
import math
import pytest
import torch
from src.config import load_config
from src.models.projectors.specs import DEFAULTS
from src.models.projectors.geometry import (
    choose_grid, grid_groups, grid_mean, region_mean, physical_extent, validate_inputs,
)


def test_nearest_grid_matches_exhaustive_search_and_ties():
    grid = (3, 4, 5)
    for target in range(1, math.prod(grid)+1):
        for policy in ('nearest', 'at_most'):
            found = choose_grid(grid, target, (3., 8., 5.), policy)
            products = [math.prod(g) for g in itertools.product(*(range(1,n+1) for n in grid))]
            allowed = [n for n in products if policy != 'at_most' or n <= target]
            score = min((abs(n-target), n > target) for n in allowed)
            assert (abs(math.prod(found)-target), math.prod(found)>target) == score
    assert math.prod(choose_grid((2,2,2), 7, (2.,2.,2.))) == 8
    assert math.prod(choose_grid((2,2,2), 6, (2.,2.,2.))) == 4
    with pytest.raises(ValueError, match='No exact'):
        choose_grid((2,2,2), 7, (2.,2.,2.), 'exact')
    with pytest.raises(ValueError, match='M <= N'):
        choose_grid((2,2,2), 9, (2.,2.,2.))


def test_disjoint_coverage_means_and_gradients_on_uneven_grid():
    grid=(5,7,3); output=(2,3,2)
    coords=torch.cartesian_prod(*(torch.arange(n) for n in grid))
    x=torch.arange(math.prod(grid)*2,dtype=torch.float32).view(-1,2).requires_grad_()
    groups=grid_groups(grid,output)
    assert torch.equal(torch.cat(groups).sort().values,torch.arange(len(x)))
    expected=torch.stack([x[g].mean(0) for g in groups])
    torch.testing.assert_close(grid_mean(x,groups),expected)
    torch.testing.assert_close(region_mean(x,groups),expected)
    grid_mean(x,groups).sum().backward()
    for g in groups:torch.testing.assert_close(x.grad[g],torch.full_like(x.grad[g],1/len(g)))
    assert validate_inputs(x,coords,{'grid_dhw':list(grid)})==grid
    with pytest.raises(ValueError,match='canonical'):
        validate_inputs(x,coords.flip(0),{'grid_dhw':list(grid)})


def test_physical_axis_permutation():
    meta={'shape_dhw':[20,48,64], 'array_zyx_to_dhw':[1,0,2],
          'bounded_before_pad':{'spacing_xyz':[1.,2.,3.]}}
    assert physical_extent((5,3,4),meta)==(40.,144.,64.)


def test_all_method_defaults_and_invalid_cross_method_parameters():
    for name in DEFAULTS:
        c=load_config(overrides=[f'projector.name={name}'])
        assert set(c.projector.params)==set(DEFAULTS[name])
    assert dict(load_config().projector.params)=={}
    for option in ('projector.name=typo','projector.params.num_tokens=64',
                   'projector.bias=1','projector.hidden_size=0'):
        with pytest.raises(ValueError):load_config(overrides=[option])
    for option in ('projector.params.num_tokens=0','projector.params.depth=2',
                   'projector.params.budget_policy=typo'):
        with pytest.raises(ValueError):
            load_config(overrides=['projector.name=avgpool3d',option])
