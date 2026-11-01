"""Approved 64x geometry, forward/backward, and editable shell array contracts."""
import copy
from contextlib import nullcontext
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

import pytest
import torch

from src.cli import baseline_sweep, bindformer_sweep
from src.config import ROOT, load_config
from src.models.projectors.budget import GRID_METHODS, EXACT_METHODS, derive_budget, resolve_budget
from src.models.projectors.geometry import choose_grid, grid_groups
from src.models.projectors.registry import build_projector
from src.models.projectors.specs import DEFAULTS
from test_baseline_sweep import population
from test_bindformer_role_value_adapter import isolated_runtime


GRID, EXTENT = (39, 15, 15), (156., 240., 240.)
METHODS = sorted(GRID_METHODS | EXACT_METHODS | {'concat3d'})
CASES = [(name, 'competitive', False) for name in METHODS] + [
    ('bindformer3d', route, saca) for route in ('competitive', 'mean', 'adaptive')
    for saca in (False, True) if (route, saca) != ('competitive', False)]
LAUNCHERS = sorted(
    p.relative_to(ROOT) for base in ('scripts/baselines', 'scripts/ft_ablation')
    for p in (ROOT/base).rglob('run_*.sh')
    if 'projector_ratios=' in p.read_text() and 'mlp' not in p.name)


def expected_tokens(name):
    return 144 if name in GRID_METHODS else 140 if name == 'concat3d' else 138


@pytest.mark.parametrize('name', sorted(GRID_METHODS))
def test_grid64_is_explicit_and_does_not_change_manual_nearest(name):
    b = derive_budget(name, 64, GRID, EXTENT)
    assert b['kind'] == 'fixed_grid_brats_r64_v1'
    assert b['raw_num_tokens'] == 138 and b['output_grid_dhw'] == [4, 6, 6]
    assert b['expected_output_tokens'] == 144 and b['relative_error'] < .05
    assert choose_grid(GRID, 138, EXTENT) == (23, 2, 3)
    groups = grid_groups(GRID, (4, 6, 6))
    assert len(groups) == 144 and min(map(len, groups)) > 0
    assert torch.equal(torch.cat(groups).sort().values, torch.arange(8775))
    for grid, extent in [((38, 15, 15), EXTENT), (GRID, (312., 240., 240.))]:
        with pytest.raises(ValueError, match='64x grid preset requires BraTS'):
            derive_budget(name, 64, grid, extent)


def exercise_projector(name, route, saca, device, full_width):
    overrides = ['projector.target_compression_ratio=64']
    if not full_width:
        overrides += ['projector.hidden_size=32']
        if 'inner_dim' in DEFAULTS[name]:
            # Raw-evidence ADirect paths deliberately forbid an implicit channel adapter.
            width = 8 if name.startswith('bindformer_adirect') else 16
            overrides += [f'projector.params.inner_dim={width}']
        if 'num_heads' in DEFAULTS[name]:
            overrides += ['projector.params.num_heads=2']
    if name == 'bindformer3d':
        overrides += [f'projector.params.within_slot_sa={str(saca).lower()}',
                      f'projector.params.local_reads_global={str(saca).lower()}',
                      'projector.params.return_auxiliary=true']
        if route != 'competitive':
            overrides += ['projector.params.fixed_spatial_regions=true',
                          'projector.params.router_value_normalization=true',
                          f'projector.params.spatial_{route}_aggregation=true']
    c = load_config([ROOT/f'configs/projector/{name}.yaml'], overrides)
    resolve_budget(c.projector, GRID, EXTENT)
    cin, cout = (768, 4096) if full_width else (8, 16)
    model = build_projector(c, cin, cout).to(device)
    x = torch.randn(8775, cin, device=device, requires_grad=True)
    coords = torch.cartesian_prod(*(torch.arange(n) for n in GRID))
    meta = {'grid_dhw': list(GRID), 'shape_dhw': list(EXTENT),
            'bounded_before_pad': {'spacing_xyz': [1., 1., 1.]},
            'array_zyx_to_dhw': [0, 1, 2], 'sequence': 'T1C',
            'oriented': {'direction': [-1, 0, 0, 0, 1, 0, 0, 0, -1]}}
    with torch.autocast(device.type, dtype=torch.bfloat16, enabled=full_width):
        result = model(x, coords, meta)
        loss = result.embeddings.float().square().mean()
    loss.backward()
    assert result.embeddings.shape == (expected_tokens(name), cout)
    assert torch.isfinite(x.grad).all() and x.grad.abs().sum() > 0
    for key, p in model.named_parameters():
        if p.requires_grad:
            assert p.grad is not None and torch.isfinite(p.grad).all(), key
    info = result.metadata['projector']
    assert info['target_compression_ratio'] == 64
    assert info['actual_tokens'] == info['expected_output_tokens'] == expected_tokens(name)
    if name in GRID_METHODS:
        assert info['output_grid'] == [4, 6, 6]
        assert sum(info['group_sizes']) == 8775
    if name == 'concat3d':
        assert info['group_size'] == [6, 3, 4] and info['empty_slots'] == 1305
    if name == 'bindformer3d':
        assert (info['global_tokens'], info['num_slots'], info['local_tokens']) == (34, 26, 104)
        if route != 'competitive':
            aux = result.auxiliary
            assert (aux.assignments.sum(-1) == 1).all()
            assert not aux.evidence_weights[aux.assignments == 0].count_nonzero()
            torch.testing.assert_close(aux.evidence_weights.sum(0), torch.ones(26, device=device))


@pytest.mark.parametrize('name,route,saca', CASES)
def test_ratio64_cpu_forward_and_backward(name, route, saca):
    exercise_projector(name, route, saca, torch.device('cpu'), False)


@pytest.mark.skipif(not os.getenv('PROJECTOR_TEST_DEVICE'), reason='Explicit GPU opt-in required')
@pytest.mark.parametrize('name,route,saca', CASES)
def test_ratio64_cuda_full_width(name, route, saca):
    device = torch.device(os.environ['PROJECTOR_TEST_DEVICE'])
    torch.cuda.set_device(device)
    try:
        torch.use_deterministic_algorithms(True)
        exercise_projector(name, route, saca, device, True)
    finally:
        torch.use_deterministic_algorithms(False)
        torch.cuda.empty_cache()


@pytest.fixture
def launcher_tree(tmp_path):
    root = tmp_path/'checkout'
    shutil.copytree(ROOT/'scripts', root/'scripts', ignore=shutil.ignore_patterns('__pycache__'))
    extension = Path('results/BindFormer_role_value_adapter_r16')
    (root/extension).mkdir(parents=True)
    for name in ('_launch.bash', 'entry.py', 'runtime.py', 'reader_adapter.py'):
        shutil.copy2(ROOT/extension/name, root/extension/name)
    (root/'configs').symlink_to(ROOT/'configs', target_is_directory=True)
    stub = tmp_path/'python_stub'
    stub.write_text(f'#!{sys.executable}\nimport json,sys\nprint(json.dumps(sys.argv[1:]))\n')
    stub.chmod(0o700)
    local = tmp_path/'local.yaml'
    local.write_text('{}\n')
    env = {k: v for k, v in os.environ.items()
           if not k.startswith(('BIND_', 'PROJECTOR_')) and k != 'WORLD_SIZE'}
    env.update(PROJECTOR_PYTHON=str(stub), PROJECTOR_LOCAL_CONFIG=str(local), CUDA_VISIBLE_DEVICES='1')
    return root, env


def capture_plan(script, env, extra=()):
    result = subprocess.run(['bash', str(script), '--dry-run', *extra],
                            env=env, check=True, capture_output=True, text=True)
    argv = json.loads(result.stdout)
    adapter = '-m' not in argv and Path(argv[1]).name == 'entry.py'
    i = 1 if adapter else argv.index('-m')
    module = bindformer_sweep if argv[i+1] == 'src.cli.bindformer_sweep' else baseline_sweep
    with isolated_runtime() if adapter else nullcontext():
        return module.build_plan(module.parser().parse_args(argv[i+2:]))


def without_ratio(config):
    config = copy.deepcopy(config)
    config['training'].pop('run_dir')
    for key in ('target_compression_ratio', 'resolved_budget'):
        config['projector'].pop(key)
    for key in ('num_tokens', 'group_size'):
        config['projector']['params'].pop(key, None)
    return config


@pytest.mark.parametrize('relative_path', LAUNCHERS, ids=lambda p: str(p))
@pytest.mark.parametrize('ratios', [(64,), (32, 64)])
def test_editable_ratio_arrays_preserve_entire_recipe(launcher_tree, population, relative_path, ratios):
    root, env = launcher_tree
    script = root/relative_path
    original = capture_plan(script, env, ['--ratios', '32'])
    text, count = re.subn(r'^projector_ratios=\([^\n)]*\)',
                         'projector_ratios=('+' '.join(map(str, ratios))+')',
                         script.read_text(), flags=re.MULTILINE)
    assert count == 1
    script.write_text(text)
    plan = capture_plan(script, env)
    assert plan['ratios'] == list(ratios)
    for run in plan['runs']:
        assert without_ratio(run['config']) == without_ratio(original['runs'][0]['config'])
        assert run['schedule'] == original['runs'][0]['schedule']
        assert run['checkpoint'].endswith(f'/r{run["ratio"]}/joint_lora/checkpoints/best.pt')
        assert run['evaluation_samples'] == 560
        if run['ratio'] == 64:
            assert run['budget']['expected_output_tokens'] == expected_tokens(plan['projector'])
    if plan.get('require_full_test'):
        baseline_sweep.validate_full_test_plan(plan)
    assert not Path(plan['run_root']).exists()


def test_mlp64_rejected():
    with pytest.raises(ValueError, match='ratios'):
        baseline_sweep.checked_ratios(baseline_sweep.parser().parse_args(
            ['--projector', 'mlp2x_gelu', '--ratios', '64']))
