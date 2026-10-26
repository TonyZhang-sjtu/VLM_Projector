"""Ratio protocol: independent expected tables, configuration and runtime contracts."""
from types import SimpleNamespace
import math
import pytest
import torch
from torch import nn
from omegaconf import OmegaConf
from src.config import ROOT, load_config, save_config, validate
from src.models.projectors.budget import (
    GRID_METHODS, EXACT_METHODS, derive_budget, resolve_budget, prepare_budget,
    geometry_from_header, check_input, output_info,
)
from src.models.projectors.registry import build_projector
from src.models.projectors.geometry import choose_grid, grid_groups

N = 8775
GRID = (39, 15, 15)
EXTENT = (156., 240., 240.)
EXPECTED = {
    4: (2194, 2197, (13, 13, 13), 2250, (4, 1, 1)),
    8: (1097, 1100, (10, 10, 11), 1125, (8, 1, 1)),
    16: (549, 550, (5, 10, 11), 525, (6, 1, 3)),
    32: (275, 275, (5, 5, 11), 280, (6, 2, 3)),
    64: (138, 144, (4, 6, 6), 140, (6, 3, 4)),
    128: (69, 72, (3, 4, 6), 72, (13, 2, 5)),
    256: (35, 36, (3, 4, 3), 36, (13, 4, 5)),
}
METHODS = sorted(GRID_METHODS | EXACT_METHODS | {'concat3d'})


def test_high_ratios_are_available_in_both_sweep_entrypoints():
    from src.cli import baseline_sweep, bindformer_sweep
    args = baseline_sweep.parser().parse_args([
        '--projector', 'avgpool3d', '--ratios', '128', '256'])
    assert baseline_sweep.checked_ratios(args) == [128, 256]
    assert bindformer_sweep.parser().parse_args(['--ratios', '128', '256']).ratios == [128, 256]
    with pytest.raises(ValueError, match='ratios'):
        baseline_sweep.checked_ratios(baseline_sweep.parser().parse_args([
            '--projector', 'mlp2x_gelu', '--ratios', '128']))


def ratio_config(name='avgpool3d', ratio=8, extra=()):
    return load_config(overrides=[f'projector.name={name}',
        f'projector.target_compression_ratio={ratio}', *extra])


def metadata():
    return {'grid_dhw': list(GRID), 'shape_dhw': [156, 240, 240],
            'bounded_before_pad': {'spacing_xyz': [1., 1., 1.]},
            'array_zyx_to_dhw': [0, 1, 2]}


@pytest.mark.parametrize('name', METHODS)
@pytest.mark.parametrize('ratio', EXPECTED)
def test_approved_matrix_and_saved_config_roundtrip(name, ratio, tmp_path):
    # Method YAMLs contain structural choices, not an implicit manual 256 budget.
    c = load_config([ROOT/f'configs/projector/{name}.yaml'],
                   [f'projector.target_compression_ratio={ratio}'])
    assert not c.projector.get('resolved_budget')
    assert 'num_tokens' not in c.projector.params and 'group_size' not in c.projector.params
    b = resolve_budget(c.projector, GRID, EXTENT)
    raw, grid_m, output_grid, concat_m, group = EXPECTED[ratio]
    assert b['raw_num_tokens'] == raw
    expected = concat_m if name == 'concat3d' else grid_m if name in GRID_METHODS else raw
    assert b['expected_output_tokens'] == expected
    assert b['actual_ratio'] == N/expected and b['relative_error'] <= 0.05
    if name == 'concat3d':
        assert tuple(c.projector.params.group_size) == group
    else:
        assert c.projector.params.num_tokens == raw
    if name in GRID_METHODS:
        assert tuple(b['output_grid_dhw']) == output_grid
        assert c.projector.params.budget_policy == 'nearest'
    c.training.required_visual_tokens = expected
    validate(c)
    p = tmp_path/'resolved.yaml'; save_config(c, p)
    restored = load_config([p])
    assert OmegaConf.to_container(restored) == OmegaConf.to_container(c)
    assert resolve_budget(restored.projector, GRID, EXTENT) == b


@pytest.mark.parametrize('ratio', [128, 256])
def test_high_ratio_spatial_presets_cover_every_patch_and_reject_other_geometry(ratio):
    raw, expected, output_grid, concat_expected, group = EXPECTED[ratio]
    assert choose_grid(GRID, raw, EXTENT) != output_grid
    groups = grid_groups(GRID, output_grid)
    assert len(groups) == expected
    assert min(map(len, groups)) > 0
    assert torch.equal(torch.cat(groups).sort().values, torch.arange(N))
    concat = derive_budget('concat3d', ratio, GRID, EXTENT)
    assert concat['output_grid_dhw'] == [math.ceil(n/k) for n, k in zip(GRID, group)]
    assert concat['expected_output_tokens'] == concat_expected
    assert concat_expected * math.prod(group) - N == 585
    for name in (*GRID_METHODS, 'concat3d'):
        with pytest.raises(ValueError, match='requires BraTS'):
            derive_budget(name, ratio, GRID, (312., 240., 240.))


@pytest.mark.parametrize('ratio', [128, 256])
@pytest.mark.parametrize('name', sorted(GRID_METHODS | {'concat3d', 'attention_pool'}))
def test_high_ratio_lightweight_forward_backward(name, ratio):
    overrides = [f'projector.target_compression_ratio={ratio}', 'projector.hidden_size=16']
    if name in {'convpool3d', 'mixer3d', 'local_packer3d', 'attention_pool'}:
        overrides.append('projector.params.inner_dim=8')
    if name in {'local_packer3d', 'attention_pool'}:
        overrides.append('projector.params.num_heads=2')
    if name == 'concat3d':
        overrides.append('projector.params.head_type=linear')
    c = load_config([ROOT/f'configs/projector/{name}.yaml'], overrides)
    budget = resolve_budget(c.projector, GRID, EXTENT)
    model = build_projector(c, 8, 16)
    x = torch.randn(N, 8, requires_grad=True)
    coords = torch.cartesian_prod(*(torch.arange(n) for n in GRID))
    result = model(x, coords, metadata())
    assert result.embeddings.shape == (budget['expected_output_tokens'], 16)
    assert result.metadata['projector']['actual_ratio'] == pytest.approx(budget['actual_ratio'])
    result.embeddings.square().mean().backward()
    assert torch.isfinite(x.grad).all() and x.grad.abs().sum() > 0
    assert all(p.grad is not None and torch.isfinite(p.grad).all()
               for p in model.parameters() if p.requires_grad)


@pytest.mark.parametrize('extra', [
    'projector.params.num_tokens=1097', 'projector.params.budget_policy=exact',
    'projector.target_compression_ratio=0', 'projector.target_compression_ratio=true',
    'projector.target_compression_ratio=.nan', 'projector.target_compression_ratio=.inf',
])
def test_invalid_and_conflicting_requests(extra):
    with pytest.raises(ValueError): ratio_config(extra=[extra])


def test_constructor_and_saved_contract_guards(tmp_path):
    c = ratio_config()
    with pytest.raises(ValueError, match='before constructing'):
        build_projector(c, 8, 16)
    with pytest.raises(ValueError, match='manual'):
        ratio_config('concat3d', extra=['projector.params.group_size=[8,1,1]'])
    resolve_budget(c.projector, GRID, EXTENT)
    p = tmp_path/'resolved.yaml'; save_config(c, p)
    for override in ('projector.target_compression_ratio=16',
                     'projector.params.num_tokens=1100',
                     'projector.resolved_budget.expected_output_tokens=1097'):
        with pytest.raises(ValueError): load_config([p], [override])
    with pytest.raises(ValueError, match='geometry'):
        resolve_budget(c.projector, (38, 15, 15), (152., 240., 240.))
    c.training.required_visual_tokens = 1097
    with pytest.raises(ValueError, match='differs'): validate(c)


def test_manual_config_and_uncompressed_compatibility():
    c = load_config()
    assert set(c.projector) == {'name', 'hidden_size', 'bias', 'params'}
    assert dict(c.projector.params) == {}
    legacy = load_config(overrides=['projector.name=avgpool3d'])
    assert legacy.projector.params.num_tokens == 256
    assert prepare_budget(c, []) is None
    with pytest.raises(ValueError, match='only ratio 1'): ratio_config('mlp2x_gelu')
    c = ratio_config('mlp2x_gelu', 1, ['projector.hidden_size=8'])
    b = resolve_budget(c.projector, GRID, EXTENT)
    model = build_projector(c, 2, 4)
    x = torch.randn(N, 2); coords = torch.cartesian_prod(*(torch.arange(n) for n in GRID))
    out = model(x, coords, metadata())
    assert b['expected_output_tokens'] == out.output_tokens == N
    assert out.metadata['projector']['actual_ratio'] == 1
    assert list(model.state_dict()) == ['layers.0.weight','layers.0.bias','layers.2.weight','layers.2.bias']


def test_geometry_and_runtime_length_guards():
    b = derive_budget('avgpool3d', 8, GRID, EXTENT)
    x = torch.empty(N, 2); meta = metadata()
    check_input(b, x, meta)
    assert output_info(b, N, 1100)['raw_num_tokens'] == 1097
    with pytest.raises(ValueError, match='mismatch'): output_info(b, N, 1097)
    with pytest.raises(ValueError, match='grid/token'): check_input(b, x[:-1], meta)
    with pytest.raises(ValueError, match='physical geometry'): check_input(b, x, {'grid_dhw': list(GRID)})
    with pytest.raises(ValueError, match='physical extent'):
        check_input(b, x, {**meta, 'bounded_before_pad': {'spacing_xyz': [1,1,2]}})
    with pytest.raises(ValueError, match='Concat ratio preset'):
        derive_budget('concat3d', 8, GRID, (312., 240., 240.))
    with pytest.raises(ValueError, match='5%'):
        derive_budget('attention_pool', 32, (2,2,2), (8.,32.,32.))


@pytest.mark.parametrize('direction', [
    (1,0,0,0,1,0,0,0,1), (-1,0,0,0,-1,0,0,0,1), (0,0,1,1,0,0,0,1,0),
])
@pytest.mark.parametrize('spacing', [(1.,1.,1.), (0.7,0.9,3.)])
def test_header_geometry_matches_actual_preprocessing(direction, spacing, tmp_path):
    import SimpleITK as sitk
    from src.preprocessing.geometry import transform_image
    c = load_config()
    image = sitk.Image([21,35,11], sitk.sitkFloat32)
    image.SetDirection(direction); image.SetSpacing(spacing)
    p = tmp_path/'image.nii.gz'; sitk.WriteImage(image, str(p))
    loaded = sitk.ReadImage(str(p))
    bounded, meta = transform_image(loaded, c.preprocessing, 'BraTS_GLI')
    sizes = [n+a+b for n,a,b in zip(bounded.GetSize(), meta['pad_before_xyz'], meta['pad_after_xyz'])]
    dhw = [sizes[::-1][i] for i in meta['array_zyx_to_dhw']]
    steps = [bounded.GetSpacing()[::-1][i] for i in meta['array_zyx_to_dhw']]
    expected_grid = [n//k for n,k in zip(dhw, (4,16,16))]
    grid, extent = geometry_from_header(p, c.preprocessing)
    assert grid == expected_grid
    assert extent == pytest.approx([n*s for n,s in zip(dhw,steps)])


def test_preflight_mixed_geometry_rejected(monkeypatch):
    from src.models.projectors import budget
    c = ratio_config()
    samples = [SimpleNamespace(id=str(i), image=str(i), cohort='BraTS_GLI') for i in range(2)]
    monkeypatch.setattr(budget, 'geometry_from_header',
                        lambda path, _: (GRID, EXTENT) if path=='0' else ((38,15,15),(152.,240.,240.)))
    with pytest.raises(ValueError, match='one input geometry'): prepare_budget(c, samples)


@pytest.mark.parametrize('name', ['avgpool3d', 'concat3d', 'perceiver_nv'])
@pytest.mark.parametrize('ratio', [8, 64, 128, 256])
def test_checkpoint_budget_preserved_and_other_ratio_rejected(name, ratio, tmp_path):
    from src.training.checkpoint import save_checkpoint, load_checkpoint
    def model(ratio):
        c = ratio_config(name, ratio, ['projector.hidden_size=16'])
        resolve_budget(c.projector, GRID, EXTENT)
        m = nn.Module(); m.config = c
        m.projector = build_projector(c, 8, 16)
        m.encoder = nn.Linear(8,8); m.encoder.requires_grad_(False); m.encoder.lora_targets = []
        return m
    m = model(ratio); p = tmp_path/'checkpoint.pt'
    save_checkpoint(p, m, None, None, {}, {})
    restored = model(ratio); load_checkpoint(p, restored)
    for a,b in zip(m.projector.parameters(), restored.projector.parameters()):
        torch.testing.assert_close(a,b,rtol=0,atol=0)
    with pytest.raises(ValueError, match='projector configuration'):
        load_checkpoint(p, model(16))
