"""Mechanism, exact-budget and launch-recipe checks for the paper ablations."""
import copy
import json
import math
import os
import subprocess
import sys

from omegaconf import OmegaConf
import pytest
import torch

from src.config import ROOT, load_config, save_config
from src.models.projectors.anchorread3d_ablation.budget import A1, A2, A3, A4, METHODS, child_counts
from src.models.projectors.anchorread3d_ablation.regions import nested_tail_boxes, packed_tail_regions
from src.models.projectors.bindformer_a4.regions import packed_children
from src.models.projectors.bindformer_adirect.regions import packed_local_regions, masked_mean
from src.models.projectors.budget import resolve_budget, NEUROQA_VERSION
from src.models.projectors.registry import build_projector
from test_projectors import config, volume


SETTINGS = [('rg', 32, (39, 15, 15), (156., 240., 240.), 275, 67, 208),
            ('rg', 256, (39, 15, 15), (156., 240., 240.), 35, 7, 28),
            ('vqa', 32, (49, 15, 13), (196., 240., 208.), 288, 72, 216),
            ('vqa', 256, (49, 15, 13), (196., 240., 208.), 36, 8, 28)]


def real_config(name, setting, small=True):
    task, ratio, grid, extent, *_ = setting
    overrides = [f'projector.target_compression_ratio={ratio}']
    if task == 'vqa':
        overrides.append(f'projector.budget_protocol={NEUROQA_VERSION}')
    if small:
        overrides += ['projector.params.inner_dim=8', 'projector.hidden_size=16',
                      'projector.params.num_heads=2']
    c = load_config([ROOT/f'configs/projector/{name}.yaml'], overrides)
    resolve_budget(c.projector, grid, extent)
    return c


def metadata(grid, extent):
    return dict(grid_dhw=list(grid), shape_dhw=list(extent),
                bounded_before_pad={'spacing_xyz': [1., 1., 1.]}, array_zyx_to_dhw=[0, 1, 2],
                sequence='T1W', oriented={'direction': [1., 0, 0, 0, 1., 0, 0, 0, 1.]})


@pytest.fixture
def deterministic_math_attention():
    previous = torch.are_deterministic_algorithms_enabled()
    try:
        torch.use_deterministic_algorithms(True)
        yield
    finally:
        torch.use_deterministic_algorithms(previous)


@pytest.mark.parametrize('setting', SETTINGS)
@pytest.mark.parametrize('name', METHODS)
def test_budget_coverage_and_saved_config(name, setting, tmp_path):
    _, _, grid, extent, total, g, local = setting
    c = real_config(name, setting)
    b = c.projector.resolved_budget
    assert b.expected_output_tokens == total
    assert b.global_tokens == (total if name == A3 else 0 if name == A4 else g)
    assert b.local_tokens == (total if name == A4 else 0 if name == A3 else local)
    assert b.global_tokens + b.local_tokens == total
    if name != A3:
        if name == A4:
            ids, valid = packed_tail_regions(grid, total, extent)
            assert list(b.children_per_parent) == list(child_counts(total))
            assert len(ids) == total
            if total % 4 == 0:
                old = packed_children(grid, total // 4, extent)
                assert torch.equal(ids, old[0]) and torch.equal(valid, old[1])
        else:
            ids, valid = packed_local_regions(grid, b, extent)
            original = real_config('bindformer_adirect_localread', setting)
            old = packed_local_regions(grid, original.projector.resolved_budget, extent)
            assert torch.equal(ids, old[0]) and torch.equal(valid, old[1])
        assert valid.any(1).all()
        assert torch.equal(ids[valid].sort().values, torch.arange(math.prod(grid)))
    path = tmp_path/'config.yaml'
    save_config(c, path)
    restored = load_config([path])
    assert OmegaConf.to_container(restored) == OmegaConf.to_container(c)
    assert resolve_budget(restored.projector, grid, extent) == b


def test_a1_matches_adirect_and_a2_removes_only_mean_bypass():
    x, coords, meta = volume((5, 5, 3))
    x = x.detach().double().requires_grad_(True)
    full = build_projector(config('bindformer_adirect_localread', target=12), 8, 6).double()
    a1 = build_projector(config(A1, target=12), 8, 6).double()
    a2 = build_projector(config(A2, target=12), 8, 6).double()
    direct = build_projector(config('bindformer_adirect', target=12), 8, 6).double()
    a1.load_state_dict(direct.state_dict())
    a2.load_state_dict(full.state_dict())
    torch.testing.assert_close(a1(x, coords, meta).embeddings, direct(x, coords, meta).embeddings, rtol=0, atol=0)
    assert not hasattr(a1, 'local_reader')
    ids, valid = packed_local_regions((5, 5, 3), full.allocation, (5., 5., 3.))
    values = x[ids]
    means = masked_mean(values, valid)
    corrected = full._local_tokens(x, (5, 5, 3), (5., 5., 3.))
    residual = a2._local_tokens(x, (5, 5, 3), (5., 5., 3.))
    torch.testing.assert_close(corrected, means + residual, rtol=0, atol=0)
    # Independent mean query must remain active when the output bypass is absent.
    probe = means.detach().clone().requires_grad_(True)
    a2.local_reader(probe, values.detach(), valid, add_mean=False).square().sum().backward()
    assert probe.grad.abs().sum() > 0
    torch.testing.assert_close(full(x, coords, meta).embeddings[:4], a2(x, coords, meta).embeddings[:4], rtol=0, atol=0)
    with torch.no_grad():
        a2.local_reader.alpha.zero_()
    assert a2._local_tokens(x, (5, 5, 3), (5., 5., 3.)).count_nonzero() == 0


def test_a3_matches_full_global_path_without_constructing_local_modules():
    full = build_projector(config('bindformer_adirect_localread', target=12), 8, 6).double()
    a3 = build_projector(config(A3, target=4), 8, 6).double()
    a3.load_state_dict({k: v for k, v in full.state_dict().items() if not k.startswith('local_reader.')})
    assert not hasattr(a3, 'local_reader') and not hasattr(a3, '_local_tokens')
    assert len(a3.global_layers) == len(full.global_layers) == 1
    x, coords, meta = volume((5, 5, 3))
    x = x.double()
    torch.testing.assert_close(a3(x, coords, meta).embeddings, full(x, coords, meta).embeddings[:4], rtol=0, atol=0)


def test_a4_partial_parent_means_and_attention_stay_local():
    model = build_projector(config(A4, target=7), 8, 6).double()
    x, coords, meta = volume((5, 5, 3))
    x = x.detach().double().requires_grad_(True)
    _, boxes = nested_tail_boxes((5, 5, 3), 7, (5., 5., 3.))
    assert child_counts(7) == (4, 3)
    cube = x.reshape(5, 5, 3, 8)
    means = torch.stack([cube[a[0]:b[0], a[1]:b[1], a[2]:b[2]].mean((0, 1, 2)) for a, b in boxes])
    with torch.no_grad():
        model.local_reader.alpha.zero_()
    torch.testing.assert_close(model._local_tokens(x, (5, 5, 3), (5., 5., 3.)), means, rtol=1e-12, atol=1e-12)
    with torch.no_grad():
        model.local_reader.alpha.fill_(.1)
    assert model.input_adapter is None and model.global_queries is None and not model.requires_sequence
    out = model(x, coords, meta)
    out.embeddings[-1].square().sum().backward()
    ids, valid = packed_tail_regions((5, 5, 3), 7, (5., 5., 3.))
    support = torch.zeros(len(x), dtype=torch.bool)
    support[ids[-1, valid[-1]]] = True
    assert x.grad[~support].count_nonzero() == 0 and x.grad[support].abs().sum() > 0


@pytest.mark.parametrize('setting', SETTINGS)
@pytest.mark.parametrize('name', METHODS)
def test_real_grid_bf16_backward_checkpointing_and_optimizer(name, setting, tmp_path,
                                                          deterministic_math_attention):
    _, _, grid, extent, total, *_ = setting
    model = build_projector(real_config(name, setting), 8, 6)
    plain = copy.deepcopy(model)
    plain.params['checkpoint_blocks'] = False
    x = torch.randn(math.prod(grid), 8, dtype=torch.bfloat16, requires_grad=True)
    y = x.detach().clone().requires_grad_(True)
    coords = torch.cartesian_prod(*(torch.arange(n) for n in grid))
    meta = metadata(grid, extent)
    with torch.autocast('cpu', dtype=torch.bfloat16):
        actual, expected = model(x, coords, meta), plain(y, coords, meta)
    assert actual.embeddings.shape == (total, 6) and actual.auxiliary is None
    torch.testing.assert_close(actual.embeddings, expected.embeddings, rtol=0, atol=0)
    actual.embeddings.float().square().mean().backward()
    expected.embeddings.float().square().mean().backward()
    torch.testing.assert_close(x.grad, y.grad, rtol=0, atol=0)
    assert torch.isfinite(x.grad).all() and x.grad.abs().sum() > 0
    for p, q in zip(model.parameters(), plain.parameters()):
        assert p.grad is not None and torch.isfinite(p.grad).all()
        torch.testing.assert_close(p.grad, q.grad, rtol=0, atol=0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)
    before = [p.detach().clone() for p in model.parameters()]
    optimizer.step()
    assert any(not torch.equal(a, b) for a, b in zip(before, model.parameters()))
    path = tmp_path/'state.pt'
    torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict()}, path)
    state = torch.load(path, weights_only=True)
    plain.load_state_dict(state['model'], strict=True)
    optimizer.load_state_dict(state['optimizer'])
    with torch.no_grad(), torch.autocast('cpu', dtype=torch.bfloat16):
        torch.testing.assert_close(model(x, coords, meta).embeddings, plain(x, coords, meta).embeddings, rtol=0, atol=0)


@pytest.mark.parametrize('name', METHODS)
def test_variant_identity_cannot_be_overridden(name):
    wrong = .5
    with pytest.raises(ValueError, match='global_fraction'):
        config(name, global_fraction=wrong)
    with pytest.raises(ValueError, match='Unsupported'):
        config(name, feature_loss_weight=.1)


def capture_shell(script, tmp_path):
    shim = tmp_path/'capture-python'
    shim.write_text('#!' + sys.executable + '\nimport json,sys\nprint(json.dumps(sys.argv[1:]))\n')
    shim.chmod(0o700)
    local = tmp_path/'local.yaml'
    local.write_text('{}\n')
    env = dict(os.environ, PROJECTOR_PYTHON=str(shim), PROJECTOR_LOCAL_CONFIG=str(local),
               CUDA_VISIBLE_DEVICES='0', PROJECTOR_RUN_TAG='ablation_recipe_test')
    env.pop('VQA_LOCAL_CONFIG', None)
    result = subprocess.run(['bash', str(script), '--dry-run', '--ratios', '32', '256'],
                            cwd=tmp_path, env=env, text=True, capture_output=True, check=True)
    return json.loads(result.stdout)


def test_eight_launchers_match_their_task_reference_recipes(tmp_path):
    from src.cli.baseline_sweep import parser as rg_parser
    from src.cli.vqa_sweep import parser as vqa_parser, build_launches, commands
    reference_args = capture_shell(ROOT/'scripts/Exp1026/run_baseline_adirect_localread_r16.sh', tmp_path)
    reference = rg_parser().parse_args(reference_args[3:])
    base_files = [ROOT/'configs/training/joint_lora.yaml', ROOT/'configs/training/deterministic.yaml',
                  ROOT/'configs/preprocessing/native_grid.yaml']
    original = load_config([*base_files, ROOT/f'configs/projector/{reference.projector}.yaml',
                            *reference.config], reference.overrides)
    vqa_reference_args = capture_shell(ROOT/'scripts/Exp_VQA/vqa_baseline_adirect_localread_r16.sh', tmp_path)
    vqa_reference = vqa_parser().parse_args(vqa_reference_args[3:])
    vqa_originals = {int(c.projector.target_compression_ratio): (c, s)
                     for c, s in build_launches(vqa_reference)}
    for name in METHODS:
        suffix = name.removeprefix('anchorread3d_')
        for task in ('rg', 'vqa'):
            script = ROOT/f'scripts/ablation/{task}_{suffix}.sh'
            subprocess.run(['bash', '-n', str(script)], check=True)
            argv = capture_shell(script, tmp_path)
            args = (rg_parser if task == 'rg' else vqa_parser)().parse_args(argv[3:])
            assert args.projector == name and args.ratios == [32, 256]
            assert 'runs/ablations/anchorread3d' in str(args.output_root)
            if task == 'rg':
                c = load_config([*base_files, ROOT/f'configs/projector/{name}.yaml', *args.config], args.overrides)
                assert args.require_full_test and args.eval_sample_fraction == 1 and args.metrics == ['all']
                got = OmegaConf.to_container(c)
                want = OmegaConf.to_container(original)
                got.pop('projector'); want.pop('projector')
                assert got == want
            else:
                assert args.preprocessing_version == 'v2' and args.invalid_sample_policy == 'include_unreviewed'
                assert args.evaluate_test and args.eval_controls == ['normal'] and args.eval_limit is None
                for c, spec in build_launches(args):
                    assert c.task.name == 'neuroqa' and c.data.modality_sampling == 'exact_balanced_repeat'
                    assert c.projector.resolved_budget.expected_output_tokens in (288, 36)
                    train, evaluations = commands(c, spec)
                    assert 'src.cli.vqa_train' in train and len(evaluations) == 1
                    assert 'train/checkpoints/best.pt' in evaluations[0][1][evaluations[0][1].index('--checkpoint') + 1]
                    vqa_original, original_spec = vqa_originals[int(c.projector.target_compression_ratio)]
                    # Compare the entire effective nonprojector config against the
                    # actual Exp_VQA shell, not RG or a hand-written subset.
                    got = OmegaConf.to_container(c, resolve=True)
                    want = OmegaConf.to_container(vqa_original, resolve=True)
                    for value in (got, want):
                        value.pop('projector')
                        value['training'].pop('run_dir')
                    assert got == want
                    assert {k: v for k, v in spec['evaluation'].items() if k != 'root'} == {
                        k: v for k, v in original_spec['evaluation'].items() if k != 'root'}
