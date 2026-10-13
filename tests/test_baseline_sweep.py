"""Sweep contracts: independent configs, exact order, failure propagation."""
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from src.cli import baseline_sweep as sweep
from src.config import ROOT
from src.models.projectors.budget import GRID_METHODS
from src.models.projectors.specs import DEFAULTS


@pytest.fixture
def population(monkeypatch):
    rows = {split: [SimpleNamespace(id=f'{split}_{i}', image=f'fake_{i % 3}.nii.gz',
                                  cohort='BraTS_GLI') for i in range(count)]
            for split, count in [('train', 1960), ('val', 268), ('test', 560)]}
    monkeypatch.setattr('src.data.manifest.load_manifests', lambda config: (rows, {}))
    monkeypatch.setattr('src.models.projectors.budget.geometry_from_header',
                        lambda path, config: ([39, 15, 15], [156., 240., 240.]))
    monkeypatch.setenv('WORLD_SIZE', '1')
    return rows


def arguments(tmp_path, method='avgpool3d', extra=()):
    return sweep.parser().parse_args(['--projector', method, '--output-root', str(tmp_path),
                                     '--run-tag', 'test_sweep', *extra])


@pytest.mark.parametrize('method', DEFAULTS)
def test_full_matrix_configs_and_default_schedule(tmp_path, population, method):
    plan = sweep.build_plan(arguments(tmp_path, method))
    expected = {4: (2197, 2194, 2250, [4, 1, 1]),
                8: (1100, 1097, 1125, [8, 1, 1]),
                16: (550, 549, 525, [6, 1, 3]),
                32: (275, 275, 280, [6, 2, 3])}
    assert plan['ratios'] == ([1] if method == 'mlp2x_gelu' else [4, 8, 16, 32])
    for run in plan['runs']:
        c, b, p = run['config'], run['budget'], run['schedule']
        if method == 'mlp2x_gelu':
            assert b['expected_output_tokens'] == 8775
        else:
            grid, exact, concat, group = expected[run['ratio']]
            assert b['expected_output_tokens'] == (grid if method in GRID_METHODS else
                                                  concat if method == 'concat3d' else exact)
            if method == 'concat3d':
                assert c['projector']['params']['group_size'] == group
        assert (p['projector_steps'], p['joint_steps'], p['cosine_steps']) == (77, 77, 1076)
        assert run['train_samples'] == 1960 and run['updates_per_epoch'] == 123
        assert run['evaluation_samples'] == 560 and run['evaluation_scope'] == 'full_split'
        assert c['training']['stage'] == 'joint_lora'
        assert c['training']['feature_cache'] is False and c['preprocessing']['cache_enabled'] is True
        assert c['training']['projector_lr'] == c['training']['encoder_lr'] == 1e-5
        assert c['training']['lora'] == dict(rank=16, alpha=32, dropout=.05, bias='none')
        assert c['training']['early_stopping_min_delta'] == .005
        assert c['training']['early_stopping_patience'] == 3
        assert c['training']['resume'] is None and c['training']['init_checkpoint'] is None
        assert c['training']['deterministic'] is True and c['decoder']['frozen'] is True
        assert c['data']['datasets'] == ['brats']
        assert c['preprocessing']['spacing_policy'] == 'native_grid'
        assert c['preprocessing']['token_filter_policy'] == 'none'
        if method in ('perceiver_nv', 'perceiver_flamingo'):
            assert c['projector']['params']['checkpoint_blocks'] is True
            assert c['projector']['params']['depth'] == 6
        assert run['checkpoint'].endswith(f'/r{run["ratio"]}/joint_lora/checkpoints/best.pt')
        assert run['checkpoint'] in run['eval_command']
        assert run['config_path'] in run['train_command']
    assert len({r['config_path'] for r in plan['runs']}) == len(plan['runs'])
    assert not Path(plan['run_root']).exists()


def test_overrides_and_identical_evaluation_subset(tmp_path, population):
    plan = sweep.build_plan(arguments(tmp_path, extra=[
        '--ratios', '32', '8', '--device', 'cuda:0', '--scoring-device', 'cpu',
        '--eval-split', 'val', '--eval-sample-fraction', '0.3', '--metrics', 'bleu4', 'rougeL',
        '--set', 'training.max_epochs=12', '--set', 'training.projector_warmup_steps=19',
        '--set', 'training.joint_warmup_steps=23', '--set', 'training.lora.rank=32',
        '--set', 'training.lora.alpha=64', '--set', 'seed=7']))
    assert plan['ratios'] == [8, 32]
    for run in plan['runs']:
        assert run['schedule']['total_steps'] == 1476
        assert run['schedule']['cosine_steps'] == 1434
        assert run['config']['training']['lora']['rank'] == 32
        assert run['config']['training']['lora']['alpha'] == 64
        assert run['config']['training']['device'] == 'cuda:0'
        assert run['evaluation_samples'] == 80 and run['evaluation_scope'] == 'fractional_sample'
        assert run['eval_command'][-4:] == ['--sample-fraction', '0.3', '--sample-seed', '7']
        assert 'cpu' in run['eval_command']


@pytest.mark.parametrize('extra', [
    ['--ratios', '8', '8'], ['--ratios', '3'], ['--run-tag', '../bad'],
    ['--eval-sample-fraction', '0'], ['--eval-sample-fraction', 'nan'],
    ['--device', 'cpu'], ['--metrics', 'all', 'bleu4'],
    ['--set', 'training.resume=old.pt'], ['--set', 'training.stage=a'],
    ['--set', 'projector.params.num_tokens=256'], ['--set', 'projector.params={num_tokens: 256}'],
    ['--set', 'data.datasets=[isles]'], ['--set', 'training.run_dir=elsewhere'],
])
def test_invalid_sweep_fails_before_writing(tmp_path, population, extra):
    with pytest.raises(ValueError):
        sweep.build_plan(arguments(tmp_path, extra=extra))
    assert list(tmp_path.iterdir()) == []


def test_mlp_cannot_be_compressed_and_existing_run_is_preserved(tmp_path, population):
    with pytest.raises(ValueError, match='ratios'):
        sweep.build_plan(arguments(tmp_path, 'mlp2x_gelu', ['--ratios', '4']))
    root = tmp_path / 'test_sweep'
    root.mkdir()
    (root / 'keep.txt').write_text('original')
    with pytest.raises(FileExistsError):
        sweep.build_plan(arguments(tmp_path))
    assert (root / 'keep.txt').read_text() == 'original'


def command_stub(trace, phase, ratio, artifact, failure=False):
    # Real subprocesses exercise sequencing/logging without loading any model.
    script = ("import json, pathlib; "
              f"p=pathlib.Path({str(trace)!r}); "
              f"p.open('a').write({f'{phase}_{ratio}'!r}+'\\n'); "
              f"a=pathlib.Path({artifact!r}); a.parent.mkdir(parents=True,exist_ok=True); "
              "a.write_text(json.dumps({'status':'complete'})); "
              f"raise SystemExit({17 if failure else 0})")
    return [sys.executable, '-c', script]


@pytest.mark.parametrize('failure', [None, 'train', 'eval'])
def test_sequential_execution_and_stop_on_failure(tmp_path, population, failure):
    plan = sweep.build_plan(arguments(tmp_path))
    trace = tmp_path / 'trace.txt'
    for run in plan['runs']:
        run['train_command'] = command_stub(trace, 'train', run['ratio'], run['checkpoint'],
                                            failure == 'train' and run['ratio'] == 8)
        run['eval_command'] = command_stub(trace, 'eval', run['ratio'], run['metric_path'],
                                           failure == 'eval' and run['ratio'] == 8)
    if failure:
        with pytest.raises(subprocess.CalledProcessError):
            sweep.execute_plan(plan)
    else:
        sweep.execute_plan(plan)
    actual = trace.read_text().splitlines()
    all_actions = [f'{phase}_{ratio}' for ratio in (4, 8, 16, 32) for phase in ('train', 'eval')]
    assert actual == (all_actions if failure is None else all_actions[:3 if failure == 'train' else 4])
    state = json.loads((Path(plan['run_root']) / 'sweep_status.json').read_text())
    assert state['status'] == ('complete' if failure is None else 'failed')
    assert state['completed_ratios'] == ([4, 8, 16, 32] if failure is None else [4])
    assert (Path(plan['run_root']) / 'r4/train_console.log').is_file()
    assert (Path(plan['run_root']) / 'r4/evaluation_console.log').is_file()


@pytest.mark.parametrize('method', DEFAULTS)
def test_bash_entrypoints_from_another_working_directory(tmp_path, method):
    stem = 'mlp2x' if method == 'mlp2x_gelu' else method
    script = ROOT / f'scripts/baselines/run_{stem}.sh'
    subprocess.run(['bash', '-n', str(script)], check=True)
    result = subprocess.run(['bash', str(script), '--help'], cwd=tmp_path,
                            text=True, capture_output=True, check=True)
    assert '--dry-run' in result.stdout and '--eval-sample-fraction' in result.stdout
