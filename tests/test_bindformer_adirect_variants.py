"""Controlled ADirect changes: locality, raw detail, exact all-local budgets and recipe parity."""
import copy
import json
import os
import subprocess
import sys

import pytest
import torch
from omegaconf import OmegaConf

from src.config import ROOT, load_config, save_config
from src.models.projectors.budget import derive_budget, resolve_budget
from src.models.projectors.registry import build_projector
from src.models.projectors.bindformer.regions import spatial_boxes
from src.models.projectors.bindformer_adirect.regions import packed_local_regions
from src.models.projectors.bindformer_adirect_localread.projector import LocalResidualRead
from src.models.projectors.selection import diverse_indices
from test_projectors import config, volume

GRID, EXTENT = (39, 15, 15), (156., 240., 240.)
VARIANTS = ('localread', 'detail', 'alllocal')


def variant_config(variant, ratio, small=True):
    name = 'bindformer_adirect' + ('' if variant == 'alllocal' else '_' + variant)
    overrides = [f'projector.target_compression_ratio={ratio}']
    if variant == 'alllocal':
        overrides.append('projector.params.global_fraction=0')
    if small:
        overrides += ['projector.params.inner_dim=8', 'projector.hidden_size=16']
        if variant != 'detail':
            overrides.append('projector.params.num_heads=2')
    c = load_config([ROOT/f'configs/projector/{name}.yaml'], overrides)
    resolve_budget(c.projector, GRID, EXTENT)
    return c


@pytest.mark.parametrize('ratio,g,local,total', [(128, 17, 52, 69), (256, 7, 28, 35)])
def test_exact_allocations_raw_coverage_and_config_roundtrip(ratio, g, local, total, tmp_path):
    old = derive_budget('bindformer_adirect', ratio, GRID, EXTENT)
    for variant in VARIANTS:
        c = variant_config(variant, ratio)
        b = c.projector.resolved_budget
        assert b.expected_output_tokens == total
        assert b.local_tokens == (total if variant == 'alllocal' else local)
        assert b.global_tokens == (g if variant == 'localread' else 0)
        assert b.get('detail_tokens', 0) == (g if variant == 'detail' else 0)
        ids, valid = packed_local_regions(GRID, b, EXTENT)
        assert valid.any(1).all()
        assert torch.equal(ids[valid].sort().values, torch.arange(8775))
        if variant != 'alllocal':
            reference = packed_local_regions(GRID, old, EXTENT)
            assert torch.equal(ids, reference[0]) and torch.equal(valid, reference[1])
        path = tmp_path/f'{variant}.yaml'
        save_config(c, path)
        restored = load_config([path])
        assert OmegaConf.to_container(restored) == OmegaConf.to_container(c)
        assert resolve_budget(restored.projector, GRID, EXTENT) == b


def test_local_read_matches_softmax_oracle_and_ignores_padding():
    torch.manual_seed(49)
    reader = LocalResidualRead(8, 2).double().eval()
    means = torch.randn(2, 8, dtype=torch.double)
    memory = torch.randn(2, 5, 8, dtype=torch.double)
    valid = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 0]], dtype=torch.bool)
    q = reader.to_q(reader.norm_queries(means)).reshape(2, 1, 2, 4).transpose(1, 2)
    k, v = (t.reshape(2, 5, 2, 4).transpose(1, 2)
            for t in reader.to_kv(reader.norm_visual(memory)).chunk(2, -1))
    prob = (q @ k.transpose(-1, -2) / 2).masked_fill(~valid[:, None, None, :], -torch.inf).softmax(-1)
    expected = means + reader.alpha * reader.to_out((prob @ v).transpose(1, 2).reshape(2, 8))
    actual = reader(means, memory, valid)
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
    changed = memory.clone()
    changed[~valid] = 1000 * torch.randn_like(changed[~valid])
    torch.testing.assert_close(actual, reader(means, changed, valid), rtol=0, atol=0)
    assert not any('ffn' in key or 'bias' in key and key.startswith('to_') for key in reader.state_dict())


def test_zero_residual_recovers_adirect_and_nonzero_read_stays_local():
    torch.manual_seed(50)
    direct = build_projector(config('bindformer_adirect', target=12), 8, 6).double().eval()
    read = build_projector(config('bindformer_adirect_localread', target=12), 8, 6).double().eval()
    missing = read.load_state_dict(direct.state_dict(), strict=False)
    assert missing.missing_keys and all(key.startswith('local_reader.') for key in missing.missing_keys)
    assert not missing.unexpected_keys
    x, coords, meta = volume((5, 5, 3))
    x = x.detach().double().requires_grad_(True)
    with torch.no_grad():
        read.local_reader.alpha.zero_()
    torch.testing.assert_close(read(x, coords, meta).embeddings, direct(x, coords, meta).embeddings,
                               rtol=0, atol=0)
    with torch.no_grad():
        read.local_reader.alpha.fill_(.5)
    out = read(x, coords, meta).embeddings
    torch.testing.assert_close(out[:4], direct(x, coords, meta).embeddings[:4], rtol=0, atol=0)
    ids, valid = packed_local_regions((5, 5, 3), read.allocation, (5., 5., 3.))
    changed = x.detach().clone()
    changed[ids[0, 0]] += torch.arange(8.)
    other = read(changed, coords, meta).embeddings
    assert not torch.equal(other[4], out[4])
    torch.testing.assert_close(other[5:], out[5:], rtol=0, atol=0)
    out[4].square().sum().backward()
    support = torch.zeros(len(x), dtype=torch.bool)
    support[ids[0, valid[0]]] = True
    assert x.grad[~support].count_nonzero() == 0
    assert x.grad[support].abs().sum() > 0
    assert not torch.allclose(x.grad[support], x.grad[support][0].expand_as(x.grad[support]))
    assert all(p.grad is not None and p.grad.abs().sum() > 0 for p in read.local_reader.parameters())


def test_detail_is_raw_selection_and_preserves_original_local_path():
    torch.manual_seed(51)
    detail = build_projector(config('bindformer_adirect_detail', target=12), 8, 6).double().eval()
    direct = build_projector(config('bindformer_adirect', target=12), 8, 6).double().eval()
    direct.final_norm.load_state_dict(detail.final_norm.state_dict())
    direct.head.load_state_dict(detail.head.state_dict())
    x, coords, meta = volume((5, 5, 3))
    x = x.detach().double().requires_grad_(True)
    indices = diverse_indices(x, 4)
    result = detail(x, coords, meta)
    assert result.auxiliary is None and not detail.requires_sequence
    assert set(dict(detail.named_children())) == {'final_norm', 'head'}
    assert result.metadata['projector']['global_tokens'] == 0
    assert result.metadata['projector']['selected_indices'] == indices.tolist()
    torch.testing.assert_close(result.embeddings[:4], detail.head(detail.final_norm(x[indices])), rtol=0, atol=0)
    torch.testing.assert_close(result.embeddings[4:], direct(x, coords, meta).embeddings[4:], rtol=0, atol=0)
    result.embeddings[:4].square().sum().backward()
    assert (x.grad.abs().sum(-1) > 0).nonzero().flatten().tolist() == indices.tolist()
    # Tied/zero features remain deterministic and yield unique sorted indices.
    tied = detail(torch.zeros_like(x), coords, meta).metadata['projector']['selected_indices']
    assert tied == [0, 1, 2, 3]


def test_alllocal_independent_spatial_mean_oracle_and_no_dead_parameters():
    model = build_projector(config('bindformer_adirect', target=7, global_fraction=0), 8, 6).double()
    x, coords, meta = volume((5, 5, 3))
    x = x.detach().double().requires_grad_(True)
    cube = x.reshape(5, 5, 3, 8)
    boxes = spatial_boxes((5, 5, 3), 7, (5., 5., 3.))
    means = torch.stack([cube[a[0]:b[0], a[1]:b[1], a[2]:b[2]].mean((0, 1, 2)) for a, b in boxes])
    result = model(x, coords, meta)
    torch.testing.assert_close(result.embeddings, model.head(model.final_norm(means)), rtol=1e-12, atol=1e-12)
    assert model.global_queries is None and model.input_adapter is None and not model.requires_sequence
    assert result.metadata['projector']['global_tokens'] == 0
    result.embeddings.square().sum().backward()
    assert (x.grad.abs().sum(-1) > 0).all()
    assert all(p.grad is not None for p in model.parameters())


@pytest.mark.parametrize('variant', VARIANTS)
@pytest.mark.parametrize('ratio', [128, 256])
def test_real_grid_bf16_checkpoint_output_and_gradient_equivalence(variant, ratio):
    c = variant_config(variant, ratio)
    model = build_projector(c, 8, 6)
    plain = copy.deepcopy(model)
    plain.params['checkpoint_blocks'] = False
    x, coords, meta = volume(GRID)
    x = x.detach().to(torch.bfloat16).requires_grad_(True)
    y = x.detach().clone().requires_grad_(True)
    meta.update(shape_dhw=list(EXTENT), bounded_before_pad={'spacing_xyz': [1., 1., 1.]},
                array_zyx_to_dhw=[0, 1, 2], sequence='T1C',
                oriented={'direction': [1., 0, 0, 0, 1., 0, 0, 0, 1.]})
    previous = torch.are_deterministic_algorithms_enabled()
    try:
        torch.use_deterministic_algorithms(True)
        with torch.autocast('cpu', dtype=torch.bfloat16):
            actual, expected = model(x, coords, meta), plain(y, coords, meta)
        assert actual.embeddings.shape == (c.projector.resolved_budget.expected_output_tokens, 6)
        assert actual.auxiliary is None
        torch.testing.assert_close(actual.embeddings, expected.embeddings, rtol=0, atol=0)
        actual.embeddings.float().square().sum().backward()
        expected.embeddings.float().square().sum().backward()
        torch.testing.assert_close(x.grad, y.grad, rtol=0, atol=0)
        assert torch.isfinite(x.grad).all() and (x.grad.abs().sum(-1) > 0).all()
        for p, q in zip(model.parameters(), plain.parameters()):
            assert p.grad is not None and torch.isfinite(p.grad).all()
            torch.testing.assert_close(p.grad, q.grad, rtol=0, atol=0)
    finally:
        torch.use_deterministic_algorithms(previous)


def test_variant_launchers_match_exp1026_entire_nonprojector_config(tmp_path):
    capture = tmp_path/'capture_python'
    capture.write_text('#!' + sys.executable + '\nimport json,os,sys\n'
                      'print(json.dumps({"args":sys.argv[1:],"gpu":os.environ["CUDA_VISIBLE_DEVICES"]}))\n')
    capture.chmod(0o700)
    local = tmp_path/'local.yaml'
    local.write_text('{}\n')
    env = dict(os.environ, PROJECTOR_PYTHON=str(capture), PROJECTOR_LOCAL_CONFIG=str(local),
               CUDA_VISIBLE_DEVICES='2', PROJECTOR_RUN_TAG='variant_recipe_test')
    baseline = None
    from src.cli.baseline_sweep import parser
    for variant in ('a4', *VARIANTS):
        name = 'run_baseline_a4_r128_r256.sh' if variant == 'a4' else f'run_baseline_adirect_{variant}.sh'
        result = subprocess.run(['bash', str(ROOT/'scripts/Exp1026'/name), '--dry-run', '--ratios', '128', '256'],
                                cwd=tmp_path, env=env, capture_output=True, text=True, check=True)
        captured = json.loads(result.stdout)
        args = parser().parse_args(captured['args'][3:])
        assert captured['gpu'] == '2' and args.ratios == [128, 256]
        assert args.require_full_test and args.metrics == ['all'] and args.eval_sample_fraction == 1
        assert args.eval_split == 'test' and args.run_tag == 'variant_recipe_test'
        files = [ROOT/'configs/training/joint_lora.yaml', ROOT/'configs/training/deterministic.yaml',
                 ROOT/'configs/preprocessing/native_grid.yaml', ROOT/f'configs/projector/{args.projector}.yaml',
                 *args.config]
        c = load_config(files, args.overrides + ['projector.target_compression_ratio=128'])
        assert c.training.init_checkpoint is None and c.training.resume is None
        if variant == 'alllocal':
            assert c.projector.name == 'bindformer_adirect' and c.projector.params.global_fraction == 0
        values = OmegaConf.to_container(c, resolve=True)
        values.pop('projector')
        if baseline is None:
            baseline = values
        assert values == baseline


@pytest.mark.parametrize('fraction', [-.1, 1, True, float('nan')])
def test_invalid_alllocal_fraction_rejected(fraction):
    from src.models.projectors.bindformer_adirect.config import allocate_adirect_tokens
    with pytest.raises(ValueError, match='global_fraction'):
        allocate_adirect_tokens(35, fraction)


def test_no_extra_loss_or_silent_channel_mapping():
    for variant in ('localread', 'detail'):
        name = f'bindformer_adirect_{variant}'
        with pytest.raises(ValueError, match='Unsupported'):
            config(name, feature_loss_weight=0.1)
        with pytest.raises(ValueError, match='input_dim == inner_dim'):
            build_projector(config(name, inner_dim=16), 8, 6)
