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
from src.models.projectors.anchorread3d_ablation.budget import METHODS as ABLATION_METHODS


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


def test_full_test_guard_rejects_val_subset_and_missing_metrics(tmp_path,population):
    from src.evaluation.clinical import METRIC_KEYS
    from src.cli.evaluate_checkpoint import OFFLINE_METRICS
    plan=sweep.build_plan(arguments(tmp_path,extra=['--ratios','8','--require-full-test']))
    sweep.validate_full_test_plan(plan)
    for extra in [['--eval-split','val'],['--eval-sample-fraction','0.5'],['--metrics','answer_ce']]:
        invalid=sweep.build_plan(arguments(tmp_path,extra=extra))
        with pytest.raises(ValueError):sweep.validate_full_test_plan(invalid)
    keys={'bleu4','rougeL','answer_ce'}
    for metric in OFFLINE_METRICS:keys.update(METRIC_KEYS[metric])
    result={'split':'test','scope':'full_split','evaluated_samples':560,'split_population':560,
            'overall':dict.fromkeys(keys,0.5)}
    run=plan['runs'][0]
    sweep.validate_full_test_result(result,run)
    for field,value in [('split','val'),('scope','fractional_sample'),('evaluated_samples',559)]:
        with pytest.raises(ValueError):sweep.validate_full_test_result({**result,field:value},run)
    for value in [None,float('nan'),float('inf')]:
        invalid={**result,'overall':{**result['overall'],'green':value}}
        with pytest.raises(ValueError):sweep.validate_full_test_result(invalid,run)
    # Partial GREEN coverage is explicit and its finite valid-sample mean remains publishable.
    sweep.validate_full_test_result({**result,'status':'partial',
        'partial_metrics':{'green':{'valid':559,'expected':560,'failed':1}}},run)


@pytest.mark.parametrize('reason',['max_epochs','max_steps','early_stopping',None,'interrupted'])
def test_training_completion_gate(tmp_path,reason):
    checkpoint=tmp_path/'checkpoints/best.pt'
    checkpoint.parent.mkdir();checkpoint.touch()
    (tmp_path/'training_state.json').write_text(json.dumps({'stop_reason':reason}))
    assert sweep.training_finished(checkpoint)==(reason in ('max_epochs','max_steps','early_stopping'))


def test_full_test_sweep_evaluates_after_early_stopping(tmp_path,population):
    from src.evaluation.clinical import METRIC_KEYS
    plan=sweep.build_plan(arguments(tmp_path,extra=['--ratios','8','--require-full-test']))
    run=plan['runs'][0]
    checkpoint=Path(run['checkpoint'])
    state_path=checkpoint.parent.parent/'training_state.json'
    run['train_command']=[sys.executable,'-c',
        "import json,pathlib; "
        f"p=pathlib.Path({str(checkpoint)!r});p.parent.mkdir(parents=True);p.touch(); "
        f"pathlib.Path({str(state_path)!r}).write_text(json.dumps({{'stop_reason':'early_stopping'}}))"]
    keys={'bleu4','rougeL','answer_ce'}
    for metric_keys in METRIC_KEYS.values():keys.update(metric_keys)
    result={'status':'complete','split':'test','scope':'full_split','evaluated_samples':560,
            'split_population':560,'overall':dict.fromkeys(keys,.5)}
    run['eval_command']=[sys.executable,'-c',
        "import json,pathlib; "
        f"p=pathlib.Path({run['metric_path']!r});p.parent.mkdir(parents=True); "
        f"p.write_text(json.dumps({result!r}))"]
    sweep.execute_plan(plan)
    status=json.loads((Path(plan['run_root'])/'sweep_status.json').read_text())
    assert status['status']=='complete' and status['completed_ratios']==[8]


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


@pytest.mark.parametrize('valid_subset', [False, True])
def test_partial_evaluation_is_recorded_and_sweep_continues(tmp_path, population, valid_subset):
    plan = sweep.build_plan(arguments(tmp_path))
    trace = tmp_path / 'trace.txt'
    result = ({'status': 'partial', 'overall': {'green': .75}, 'skipped_metrics': {},
               'partial_metrics': {'green': {'valid': 2, 'expected': 3, 'failed': 1}}}
              if valid_subset else {'status': 'partial', 'skipped_metrics': {'green': {}}})
    for run in plan['runs']:
        run['train_command'] = command_stub(trace, 'train', run['ratio'], run['checkpoint'])
        run['eval_command'] = command_stub(trace, 'eval', run['ratio'], run['metric_path'])
        if run['ratio'] == 8:
            run['eval_command'] = [
                sys.executable, '-c',
                ("import json,pathlib; "
                 f"p=pathlib.Path({str(trace)!r}); p.open('a').write('eval_8\\n'); "
                 f"a=pathlib.Path({run['metric_path']!r}); a.parent.mkdir(parents=True,exist_ok=True); "
                 f"a.write_text(json.dumps({result!r}))")
            ]
    sweep.execute_plan(plan)
    state = json.loads((Path(plan['run_root']) / 'sweep_status.json').read_text())
    assert state['status'] == 'partial'
    assert state['completed_ratios'] == [4, 8, 16, 32]
    expected = {'ratio': 8, 'skipped_metrics': result['skipped_metrics']}
    if valid_subset:
        expected['partial_metrics'] = result['partial_metrics']
    assert state['partial_ratios'] == [expected]
    assert trace.read_text().splitlines()[-2:] == ['train_32', 'eval_32']


def test_resume_reuses_finished_checkpoint_and_continues_sweep(tmp_path, population):
    plan = sweep.build_plan(arguments(tmp_path))
    root = Path(plan['run_root'])
    root.mkdir(parents=True)
    sweep.write_json(root / 'sweep_plan.json', plan)
    sweep.write_json(root / 'sweep_status.json', {
        'status': 'failed', 'projector': plan['projector'], 'completed_ratios': [],
        'partial_ratios': [], 'active_ratio': 4, 'phase': 'evaluation',
    })
    trace = tmp_path / 'resume_trace.txt'
    for run in plan['runs']:
        run['train_command'] = command_stub(trace, 'train', run['ratio'], run['checkpoint'])
        run['eval_command'] = command_stub(trace, 'eval', run['ratio'], run['metric_path'])
    first = plan['runs'][0]
    checkpoint = Path(first['checkpoint'])
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_text('finished checkpoint')
    sweep.write_json(checkpoint.parent.parent / 'training_state.json', {
        'stop_reason': 'max_epochs', 'global_step': 1230,
    })
    Path(first['metric_path']).parent.mkdir(parents=True)

    sweep.execute_plan(plan, resume=True)

    assert trace.read_text().splitlines() == [
        'eval_4', 'train_8', 'eval_8', 'train_16', 'eval_16', 'train_32', 'eval_32',
    ]
    state = json.loads((root / 'sweep_status.json').read_text())
    assert state['status'] == 'complete'
    assert state['completed_ratios'] == [4, 8, 16, 32]


def test_load_resume_plan_rejects_wrong_projector(tmp_path, population):
    plan = sweep.build_plan(arguments(tmp_path))
    root = Path(plan['run_root'])
    root.mkdir(parents=True)
    sweep.write_json(root / 'sweep_plan.json', plan)
    assert sweep.load_resume_plan(root, 'avgpool3d') == plan
    with pytest.raises(ValueError, match='projector'):
        sweep.load_resume_plan(root, 'concat3d')


@pytest.mark.parametrize('method', DEFAULTS)
def test_bash_entrypoints_from_another_working_directory(tmp_path, method):
    stem = 'mlp2x' if method == 'mlp2x_gelu' else method
    script = (ROOT / f'scripts/ablation/rg_{method.removeprefix("anchorread3d_")}.sh'
              if method in ABLATION_METHODS else ROOT / f'scripts/baselines/run_{stem}.sh')
    subprocess.run(['bash', '-n', str(script)], check=True)
    result = subprocess.run(['bash', str(script), '--help'], cwd=tmp_path,
                            text=True, capture_output=True, check=True)
    assert '--dry-run' in result.stdout
    assert ('--global-fraction' if method=='bindformer3d' else '--eval-sample-fraction') in result.stdout
