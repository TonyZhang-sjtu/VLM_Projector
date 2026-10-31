from copy import deepcopy
import json

from openpyxl import Workbook, load_workbook
import pytest

from src.cli.export_answer_ce import relocate_input_paths
from src.evaluation.ce_runtime import CE_ENVIRONMENT, ce_replay_environment, resolve_ce_runtime
from src.cli.significance_suite import apply_suite_holm, export_suite, main
from src.evaluation.significance import METRICS
from src.evaluation.significance_io import scorer_contract, find_metric_files
from src.evaluation.significance_study import (
    RATIOS, METHODS, HEADERS, read_targets, method_for_label, select_candidate,
    matching_candidates, check_baseline_a, inventory,
)


def workbook(path):
    book = Workbook()
    sheet = book.active
    sheet.title = 'Main'
    for ratio in RATIOS:
        sheet.append([f'a. {ratio}x Compression'])
        sheet.append([None, *HEADERS])
        sheet.append(['MLP (No Compression)', *([1.] * 11)])
        for method, (label, _) in METHODS.items():
            if method == 'ours':
                label = 'Ours Vanlla+fixed region mean 1021 (Baseline A)'
            sheet.append([label, *([.5] * 11)])
        sheet.append(['Ours Vanilla+fixed region mean +role value ada 1021', *([.5] * 11)])
    book.save(path)
    return path


def test_ce_subprocess_restores_training_runtime_and_preserves_gpu_selection(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from src.cli.significance_suite import run_export
    parent = {'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1', 'CUDA_VISIBLE_DEVICES': '3',
              'CUBLAS_WORKSPACE_CONFIG': ':16:8', 'PYTORCH_CUDA_ALLOC_CONF': 'backend:native'}
    environment = ce_replay_environment(parent)
    assert all(environment[k] == v for k, v in CE_ENVIRONMENT.items())
    assert environment['CUDA_VISIBLE_DEVICES'] == '3'
    assert parent['OMP_NUM_THREADS'] == '1'
    for key, value in parent.items():
        monkeypatch.setenv(key, value)
    captured = []
    def execute(command, **kwargs):
        captured.append((command, kwargs['env']))
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr('src.cli.significance_suite.subprocess.run', execute)
    monkeypatch.setattr('src.cli.significance_suite.resolve_ce_runtime',
                        lambda entry, path: {'environment': dict(CE_ENVIRONMENT)})
    (tmp_path/'r16').mkdir()
    (tmp_path/'r16/run_manifest.json').write_text(json.dumps({'methods': [{'id': 'ours'}]}))
    args = SimpleNamespace(projector_python='/python', runtime_profiles=tmp_path/'profiles.json', ce_policy='strict')
    run_export(args, tmp_path, {'ratio': 16, 'id': 'ours'}, gpu='0')
    command, child_env = captured[0]
    assert command[:3] == ['/python', '-m', 'src.cli.export_answer_ce']
    assert child_env['CUDA_VISIBLE_DEVICES'] == '0'
    assert all(child_env[k] == v for k, v in CE_ENVIRONMENT.items())


def test_runtime_profile_uses_original_origin_and_explicit_settings(tmp_path):
    ev = tmp_path/'renamed_import/evaluation/best_test'
    ev.mkdir(parents=True)
    (ev/'evaluation_config.yaml').write_text(json.dumps({'project_root': '/server_b/project'}))
    (ev/'pipeline.json').write_text(json.dumps({'commands': [['/server_b/python', '-m', 'src.cli.evaluate']]}))
    profile = dict(id='b', origin=dict(project_root='/server_b/project', executable='/server_b/python'),
                   environment={**CE_ENVIRONMENT, 'OMP_NUM_THREADS': '8', 'MKL_NUM_THREADS': '8'})
    path = tmp_path/'profiles.json'
    path.write_text(json.dumps({'profiles': [profile]}))
    entry = dict(run_dir=str(ev.parents[1]), checkpoint_sha256='checkpoint')
    resolved = resolve_ce_runtime(entry, path)
    assert resolved['id'] == 'b'
    env = ce_replay_environment({'OMP_NUM_THREADS': '1', 'CUDA_VISIBLE_DEVICES': '0'}, resolved['environment'])
    assert env['OMP_NUM_THREADS'] == env['MKL_NUM_THREADS'] == '8'
    assert env['CUDA_VISIBLE_DEVICES'] == '0'
    (ev/'evaluation_config.yaml').write_text(json.dumps({'project_root': '/unrecorded'}))
    with pytest.raises(ValueError, match='one explicit'):
        resolve_ce_runtime(entry, path)


def test_ce_recompute_is_explicit_and_never_relaxes_tokens_or_checkpoint_identity(tmp_path):
    from src.evaluation.clinical import digest, file_digest, write_json
    from src.evaluation.ce_validation import compare_ce_history, require_historical_match
    from src.evaluation.significance_io import attach_ce
    history = dict(samples=1, answer_tokens=2, answer_ce=1.,
                   cohorts={'GLI': dict(samples=1, answer_tokens=2, loss_sum=2., answer_ce=1.)})
    rows = [dict(id='s1', patient_id='p1', cohort='GLI', loss_sum=2.002, answer_tokens=2,
                 checkpoint_sha256='correct', reference_sha256=digest('reference'))]
    comparison = compare_ce_history(rows, history)
    assert not comparison['all_match']
    with pytest.raises(ValueError, match='differs from historical'):
        require_historical_match(comparison, 'strict')
    require_historical_match(comparison, 'recompute')
    with pytest.raises(ValueError, match='token count'):
        compare_ce_history([dict(rows[0], answer_tokens=3)], history)
    samples = tmp_path/'samples.jsonl'
    samples.write_text(json.dumps(rows[0])+'\n')
    summary = dict(status='complete', checkpoint_sha256='correct', test_identity_sha256='test',
                   samples_sha256=file_digest(samples), answer_ce=1.001, ce_policy='recompute',
                   historical_aggregation=history, historical_comparison=comparison)
    write_json(tmp_path/'summary.json', summary)
    run = dict(rows=[dict(id='s1', patient_id='p1', reference='reference')], vectors={},
               audit=dict(checkpoint_sha256='correct', test_identity_sha256='test',
                          original_metrics={'answer_ce': 1.}, original_ce_aggregation=history))
    with pytest.raises(ValueError, match='differs from historical'):
        attach_ce(run, tmp_path)
    assert attach_ce(run, tmp_path, policy='recompute')
    assert run['audit']['ce_replay']['difference'] == pytest.approx(.001)
    samples.write_text(json.dumps(dict(rows[0], checkpoint_sha256='wrong'))+'\n')
    summary['samples_sha256'] = file_digest(samples)
    write_json(tmp_path/'summary.json', summary)
    with pytest.raises(ValueError, match='sample checkpoint mismatch'):
        attach_ce(run, tmp_path, policy='recompute')


def test_one_ce_failure_does_not_stop_other_runs_or_become_complete(tmp_path, monkeypatch):
    from contextlib import contextmanager
    from types import SimpleNamespace
    from src.cli.significance_suite import export_missing_ce
    targets = [dict(ratio=16, id=m) for m in ('ours', 'avgpool', 'concat')]
    runs = {f'r16/{t["id"]}': {'vectors': {}} for t in targets}
    launched = []
    def export(args, output, target, dry_run=False, gpu=None):
        if not dry_run:
            launched.append(target['id'])
            if target['id'] == 'avgpool':
                raise RuntimeError('failed numerical replay')
    @contextmanager
    def gpu(*args):
        yield '0'
    def attach(run, *args, **kwargs):
        run['vectors']['answer_ce'] = {'sample': [1., 1]}
        return True
    monkeypatch.setattr('src.cli.significance_suite.run_export', export)
    monkeypatch.setattr('src.cli.significance_suite.available_gpu', gpu)
    monkeypatch.setattr('src.cli.significance_suite.attach_ce', attach)
    errors = export_missing_ce(SimpleNamespace(ce_policy='strict'), tmp_path, targets, runs)
    assert launched == ['ours', 'avgpool', 'concat']
    assert set(errors) == {'r16/avgpool'}
    progress = json.loads((tmp_path/'ce_progress.json').read_text())
    assert progress['completed'] == ['r16/ours', 'r16/concat']
    assert 'answer_ce' not in runs['r16/avgpool']['vectors']


def test_workbook_selects_exact_35_rows_and_rejects_missing_duplicate_or_changed_header(tmp_path):
    path = workbook(tmp_path/'source.xlsx')
    targets, source = read_targets(path)
    assert len(targets) == 35
    assert {t['ratio'] for t in targets} == set(RATIOS)
    assert {t['id'] for t in targets} == set(METHODS)
    assert source['sheet'] == 'Main'
    assert method_for_label('Ours Vanilla+fixed region mean 1021') == 'ours'
    assert method_for_label('Baseline A + E2 (mean + pair) 1023') is None
    assert method_for_label('Ours SACA+fixed region mean 1021') is None
    assert method_for_label('Ours Vanilla+fixed region mean +role value ada 1021') is None
    book = load_workbook(path)
    book['Main'].cell(4, 2).value = None
    book.save(path)
    with pytest.raises(ValueError, match='11 numeric'):
        read_targets(path)
    workbook(path)
    book = load_workbook(path)
    book['Main'].cell(2, 3).value = 'Different metric'
    book.save(path)
    with pytest.raises(ValueError, match='header/order'):
        read_targets(path)
    workbook(path)
    book = load_workbook(path)
    book['Main'].append(['AvgPool3D', *([.5] * 11)])
    book.save(path)
    with pytest.raises(ValueError, match='Duplicate target'):
        read_targets(path)


def test_selection_never_falls_back_from_a_frozen_identity_or_breaks_ties():
    a = dict(run_dir='/run/old', checkpoint_sha256='a', metric_sha256='m', config_sha256='c')
    b = dict(a, run_dir='/run/new', checkpoint_sha256='b')
    with pytest.raises(ValueError, match='Ambiguous'):
        select_candidate({}, [a, b])
    assert select_candidate({}, [a, b], a) == a
    with pytest.raises(ValueError, match='no fallback'):
        select_candidate({}, [b], a)
    changed = dict(a, metric_sha256='changed')
    with pytest.raises(ValueError, match='Frozen metric'):
        select_candidate({}, [changed], a)
    with pytest.raises(ValueError, match='No exact'):
        select_candidate({}, [])


def test_discovery_without_rg_preserves_hidden_ignored_and_overlapping_roots(tmp_path, monkeypatch):
    from src.cli.significance import discover
    root = tmp_path/'runs with spaces'
    root.mkdir()
    (root/'.gitignore').write_text('ignored/\n')
    expected = []
    for name in ('ordinary', '.hidden', 'ignored/nested'):
        run = root/name/'r16'
        ev = run/'evaluation/best_test'
        ev.mkdir(parents=True)
        stage = run/'joint_lora'
        stage.mkdir()
        metric = dict(stage='joint_lora', checkpoint_sha256='abc', overall=dict.fromkeys(METRICS, .5))
        (ev/'metric.json').write_text(json.dumps(metric))
        config = dict(seed=42, training={}, projector=dict(name='avgpool3d', target_compression_ratio=16))
        (stage/'resolved_config.yaml').write_text(json.dumps(config))
        expected.append(ev/'metric.json')
    monkeypatch.setenv('PATH', '')  # Reproduce a shell where rg cannot be found.
    def no_external_commands(*args, **kwargs):
        pytest.fail('The Python fallback must not require an external executable')
    monkeypatch.setattr('src.evaluation.significance_io.subprocess.run', no_external_commands)
    roots = [root, root/'.hidden', tmp_path/'missing']
    assert find_metric_files(roots) == sorted(expected)
    items, errors = inventory(roots)
    assert len(items) == 3 and not errors
    target = dict(id='avgpool', projector='avgpool3d', ratio=16,
                  expected_rounded_metrics=dict.fromkeys(METRICS, .5))
    assert len(matching_candidates(target, items)) == 3
    assert len(discover(dict(search_roots=roots, methods=[target]))['avgpool']) == 3
    with pytest.raises(ValueError, match='Ambiguous'):
        select_candidate(target, items)


def test_discovery_does_not_hide_rg_io_errors(tmp_path, monkeypatch):
    from types import SimpleNamespace
    monkeypatch.setattr('src.evaluation.significance_io.shutil.which', lambda _: '/bin/rg')
    monkeypatch.setattr('src.evaluation.significance_io.subprocess.run',
                        lambda *a, **k: SimpleNamespace(returncode=2, stderr='permission denied'))
    with pytest.raises(RuntimeError, match='permission denied'):
        find_metric_files([tmp_path])


def test_match_requires_every_metric_ratio_and_exact_baseline_a_variant():
    target = dict(id='avgpool', projector='avgpool3d', ratio=32,
                  expected_rounded_metrics=dict.fromkeys(METRICS, .5))
    item = dict(config={'projector': {'name': 'avgpool3d', 'target_compression_ratio': 32}},
                metric={'overall': dict.fromkeys(METRICS, .5)})
    assert matching_candidates(target, [item]) == [item]
    for bad in (.51, float('nan')):
        changed = deepcopy(item)
        changed['metric']['overall']['answer_ce'] = bad
        assert not matching_candidates(target, [changed])
    changed = deepcopy(item)
    changed['config']['projector']['target_compression_ratio'] = 16
    assert not matching_candidates(target, [changed])
    params = dict(fixed_spatial_regions=True, spatial_mean_aggregation=True,
                  spatial_adaptive_aggregation=False, within_slot_sa=False,
                  local_reads_global=False, geometry_enabled=False)
    check_baseline_a({'projector': {'params': params}})
    with pytest.raises(ValueError, match='excludes report_output'):
        check_baseline_a({'projector': {'params': dict(params, report_output={'enabled': True})}})
    with pytest.raises(ValueError, match='within_slot_sa'):
        check_baseline_a({'projector': {'params': dict(params, within_slot_sa=True)}})


def test_remote_paths_are_mapped_only_within_original_repository():
    config = dict(project_root='/server/old',
                  data=dict(root='/server/old/assets/data', annotation_root='/external/annotations', manifest_dir='relative/manifests'),
                  encoder=dict(path='/server/old/assets/encoder'), decoder=dict(path='/server/older/decoder'),
                  preprocessing=dict(cache_dir='cache'), training=dict(feature_cache_dir='/server/old/cache/features'))
    changes = relocate_input_paths(config, '/local/project')
    assert config['project_root'] == '/local/project'
    assert config['data']['root'] == '/local/project/assets/data'
    assert config['data']['annotation_root'] == '/external/annotations'
    assert config['data']['manifest_dir'] == 'relative/manifests'
    assert config['decoder']['path'] == '/server/older/decoder'
    assert len(changes) == 3


@pytest.mark.parametrize('ratio', [128, 256])
def test_ce_prepare_colocates_two_gpu_concat_without_changing_checkpoint(tmp_path, ratio):
    import torch
    from omegaconf import OmegaConf
    from src.cli.export_answer_ce import prepare
    from src.config import ROOT, load_config, validate
    from src.evaluation.clinical import file_digest

    config = load_config([ROOT/'configs/projector/concat3d.yaml'], [
        f'projector.target_compression_ratio={ratio}', 'training.projector_device=cuda:1'])
    checkpoint = tmp_path/'a/checkpoints/best.pt'
    checkpoint.parent.mkdir(parents=True)
    torch.save(dict(format_version=1, stage='a', config=OmegaConf.to_container(config, resolve=True)), checkpoint)
    sha = file_digest(checkpoint)
    ev = tmp_path/'evaluation/best_test'
    (ev/'generation').mkdir(parents=True)
    (ev/'metric.json').write_text(json.dumps(dict(stage='a', checkpoint_sha256=sha)))
    (ev/'expected_ids.json').write_text('["s1"]')
    (ev/'generation/predictions.jsonl').write_text('{"id": "s1"}\n')
    entry = dict(run_dir=str(tmp_path), checkpoint_sha256=sha, projector='concat3d', ratio=ratio)

    replay, _, _, _, _, device_changes = prepare(entry, tmp_path/'output', 'cuda:0')
    validate(replay)
    assert replay.training.device == 'cuda:0'
    assert replay.training.projector_device is None
    assert device_changes['training.projector_device'] == dict(
        **{'from': 'cuda:1', 'to': None}, effective_device='cuda:0', reason='single_device_ce_inference')
    assert file_digest(checkpoint) == sha
    assert torch.load(checkpoint, weights_only=False)['config']['training']['projector_device'] == 'cuda:1'


def test_scorer_compatibility_excludes_host_location_but_retains_numerical_contract():
    original = dict(device='cuda:0', device_name='A5500',
                    environment=dict(executable='/local/python', python='3.10', packages={'torch': '2.5'}),
                    models={'weights_sha256': 'abc'}, settings={'dtype': 'float32'},
                    adapter_sha256='code1', seed=42, tf32=False)
    remote = deepcopy(original)
    remote.update(device='cuda:3', device_name='A5000')
    remote['environment']['executable'] = '/remote/python'
    assert scorer_contract(original) == scorer_contract(remote)
    assert original['device_name'] == 'A5500'  # No mutation of provenance.
    for field, changed in (
        ('models', {'weights_sha256': 'wrong'}), ('settings', {'dtype': 'float16'}),
        ('adapter_sha256', 'code2'), ('seed', 24), ('tf32', True), ('device', 'cpu'),
        ('environment', dict(executable='/remote/python', python='3.10', packages={'torch': '2.6'})),
    ):
        bad = deepcopy(remote)
        bad[field] = changed
        assert scorer_contract(original) != scorer_contract(bad)


def test_default_command_stops_before_inference_when_transfer_incomplete(tmp_path, monkeypatch):
    source = workbook(tmp_path/'source.xlsx')
    (tmp_path/'empty.json').write_text('{"bindings": {}}')
    def forbidden(*args, **kwargs):
        pytest.fail('No model inference or statistics may start with missing inputs')
    monkeypatch.setattr('src.cli.significance_suite.run_export', forbidden)
    monkeypatch.setattr('src.cli.significance_suite.paired_test', forbidden)
    output = tmp_path/'output'
    result = main(['--workbook', str(source), '--bindings', str(tmp_path/'empty.json'),
                   '--search-roots', str(tmp_path/'not_transferred'), '--output-dir', str(output)])
    assert result == 2
    record = json.loads((output/'preflight.json').read_text())
    assert record['ready_runs'] == 0 and len(record['issues']) == 35
    assert record['inference_started'] is False
    assert not (output/'comparisons.json').exists()


def test_combined_report_keeps_raw_and_both_adjustment_scopes(tmp_path):
    rows = []
    for ratio in RATIOS:
        for method in list(METHODS)[1:]:
            for metric in METRICS:
                row = dict(ratio=ratio, baseline_id=method, baseline=METHODS[method][0], metric=metric,
                           status='complete', p_raw=.0001, p_holm_66=.0066)
                rows.append(row)
    assert apply_suite_holm(rows) == 'final_full_family'
    assert all(r['p_holm_330'] == pytest.approx(.033) for r in rows)
    assert all(r['p_raw'] == .0001 and r['p_holm_66'] == .0066 for r in rows)
    (tmp_path/'run_audit.json').write_text('[]')
    report = export_suite(tmp_path, {'type': 'test'}, rows)
    assert report['planned_tests'] == report['completed_tests'] == 330
    book = load_workbook(tmp_path/'significance.xlsx', data_only=True)
    assert {'p_raw', 'p_holm_66', 'p_holm_330', 'run_identity', 'protocol'} <= set(book.sheetnames)
    assert book['comparisons'].max_row == 331
    assert book['p_raw'].max_row == 31  # Five ratios * six baselines + header.
    book.close()
    rows[-1].update(status='pending', p_raw=None, p_holm_66=None)
    rows[-1].pop('p_holm_330')
    assert apply_suite_holm(rows) == 'conservative_incomplete_family'
    assert rows[-1].get('p_holm_330') is None
