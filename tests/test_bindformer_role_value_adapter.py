"""Numerical checks and isolated loading for the published experiment runtime."""
import copy
from contextlib import contextmanager
import importlib.util
from types import SimpleNamespace

import pytest
import torch

from src.config import ROOT, load_config
from src.models.projectors import registry, specs
from src.models.projectors.bindformer.reader import SupportReader
from src.models.projectors.budget import resolve_budget


EXTENSION = ROOT / 'results/BindFormer_role_value_adapter_r16'


def load_extension(name):
    spec = importlib.util.spec_from_file_location(f'role_value_test_{name}', EXTENSION/f'{name}.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@contextmanager
def isolated_runtime():
    # Production installs once per child process. Tests must restore the shared
    # registry so ordinary BindFormer checks retain their original defaults.
    with pytest.MonkeyPatch.context() as patch:
        patch.syspath_prepend(str(EXTENSION))
        patch.setitem(specs.DEFAULTS, 'bindformer3d', copy.deepcopy(specs.DEFAULTS['bindformer3d']))
        patch.setattr(specs, 'validate_params', specs.validate_params)
        patch.setattr(registry, 'PROJECTORS', dict(registry.PROJECTORS))
        load_extension('runtime').install()
        yield


def test_reader_gradients_locality_and_checkpoint_roundtrip():
    reader = load_extension('reader_adapter').RoleValueReader
    torch.manual_seed(42)
    original = SupportReader(32, 4, [1., .5, 0., 2.], chunk_size=8)
    control = reader(original, False, 4, 4.)
    adapter = reader(original, True, 4, 4.)
    q = torch.randn(3, 4, 32, requires_grad=True)
    v = torch.randn(9, 32, requires_grad=True)
    ids = torch.tensor([[0, 1, 2], [3, 4, 0], [5, 6, 7]])
    valid = torch.tensor([[1, 1, 1], [1, 1, 0], [1, 1, 1]], dtype=torch.bool)
    region = SimpleNamespace(indices=ids, valid=valid)
    support = torch.zeros(9, 3)
    for k in range(3):
        support[ids[k, valid[k]], k] = torch.arange(1, int(valid[k].sum())+1, dtype=torch.float)
    support = support / support.sum(0)
    a, b, c = [m(q, v, support, region) for m in (original, control, adapter)]
    assert (a-b).norm()/a.norm() < 1e-6
    assert torch.equal(b, c)
    target = torch.randn_like(c)
    (c*target).sum().backward()
    assert all(m.weight.grad.norm() > 0 for m in adapter.value_up)
    assert all(m.weight.grad.norm() == 0 for m in adapter.value_down)
    assert q.grad.norm() > 0 and v.grad.norm() > 0
    with torch.no_grad():
        for m in adapter.value_up:
            m.weight.add_(m.weight.grad, alpha=-.01)
    adapter.zero_grad()
    (adapter(q, v, support, region)*target).sum().backward()
    assert all(m.weight.grad.norm() > 0 for m in adapter.value_down)
    with torch.no_grad():
        changed = v.clone()
        changed[3:] += torch.randn_like(changed[3:])*10
        assert torch.equal(adapter(q, v, support, region)[0], adapter(q, changed, support, region)[0])
        assert torch.equal(adapter.role_values(torch.zeros_like(v)), torch.zeros(4, 9, 32))
        restored = reader(original, True, 4, 4.)
        restored.load_state_dict(adapter.state_dict())
        assert torch.equal(restored(q, v, support, region), adapter(q, v, support, region))
    plain, checked = copy.deepcopy(adapter), copy.deepcopy(adapter)
    checked.checkpoint_blocks = True
    for model in (plain, checked):
        model.zero_grad()
        (model(q, v, support, region)*target).sum().backward()
    for (name, p), (other_name, other) in zip(plain.named_parameters(), checked.named_parameters()):
        assert name == other_name
        torch.testing.assert_close(p.grad, other.grad, rtol=2e-5, atol=2e-6)


@pytest.mark.parametrize('route', ['mean', 'adaptive'])
def test_ratio64_adapter_runtime_forward_backward(route):
    with isolated_runtime():
        config = load_config([ROOT/'configs/projector/bindformer3d.yaml'], [
            'projector.target_compression_ratio=64', 'projector.hidden_size=32',
            'projector.params.inner_dim=32', 'projector.params.num_heads=4',
            'projector.params.knowledge_mode=none',
            'projector.params.fixed_spatial_regions=true',
            f'projector.params.spatial_{route}_aggregation=true',
        ])
        resolve_budget(config.projector, (39, 15, 15), (156., 240., 240.))
        model = registry.build_projector(config, 8, 16)
        x = torch.randn(8775, 8, requires_grad=True)
        coords = torch.cartesian_prod(torch.arange(39), torch.arange(15), torch.arange(15))
        result = model(x, coords, {
            'grid_dhw': [39, 15, 15], 'shape_dhw': [156, 240, 240],
            'bounded_before_pad': {'spacing_xyz': [1., 1., 1.]},
            'array_zyx_to_dhw': [0, 1, 2], 'sequence': 'T1C',
            'oriented': {'direction': [-1, 0, 0, 0, 1, 0, 0, 0, -1]},
        })
        assert result.embeddings.shape == (138, 16)
        result.embeddings.float().square().mean().backward()
        assert torch.isfinite(x.grad).all() and x.grad.abs().sum() > 0
        assert all(m.weight.grad.norm() > 0 for m in model.reader.value_up)
