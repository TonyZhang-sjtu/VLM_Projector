"""Launch contracts, version isolation, fail-fast readiness and restart behavior."""
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf

from src.cli import vqa_sweep as sweep
from src.config import ROOT, digest, load_config, save_config
from src.data import neuroqa_preflight as preflight
from src.data.neuroqa_manifest import QASample, write_json


@pytest.mark.parametrize('method', sweep.METHODS)
@pytest.mark.parametrize('version', ['v1', 'v2'])
def test_each_projector_version_and_all_supported_budgets(method, version):
    ratios = [1] if method == 'mlp2x_gelu' else [8, 16, 32, 64, 128, 256]
    args = sweep.parser().parse_args(['--projector', method, '--preprocessing-version', version,
                                    '--ratios', *map(str, ratios), '--run-tag', 'unit'])
    runs = sweep.build_launches(args)
    assert [c.projector.resolved_budget.expected_output_tokens for c, _ in runs] == (
        [9555] if ratios == [1] else [1170, 624, 288, 150, 75, 36])
    for c, spec in runs:
        assert c.preprocessing.cache_dir.endswith(f'neuroqa_mni1mm_{version}')
        assert c.data.manifest_dir.endswith('_mni_v2' if version == 'v2' else 'patient_guard_v1')
        scope = 'unreviewed_training' if version == 'v2' else 'training'
        assert f'/{version}/{scope}/' in spec['run']
        assert c.training.run_dir == spec['run'] + '/train'
        command, evaluations = sweep.commands(c, spec)
        assert 'src.cli.vqa_train' in command
        assert len(evaluations) == 1 and evaluations[0][0].name == 'normal'
        if method == 'concat3d':
            assert c.training.projector_device == 'cpu'
        if method == 'bindformer3d':
            assert c.projector.params.knowledge_mode == 'none'


@pytest.mark.parametrize('arguments,match', [
    (['--ratios', '16', '16'], 'Duplicate'),
    (['--run-tag', '../escape'], 'directory'),
    (['--set', 'preprocessing.name=neuroqa_mni1mm_v2'], 'version'),
    (['--set', 'projector.target_compression_ratio=64'], 'identity'),
    (['--set', 'training.resume=old.pt'], 'resume-run'),
    (['--evaluate-only'], 'resume-run'),
    (['--projector', 'mlp2x_gelu', '--ratios', '16'], 'ratio 1'),
])
def test_bad_requests_rejected(arguments, match):
    with pytest.raises(ValueError, match=match):
        sweep.build_launches(sweep.parser().parse_args(arguments))


def test_shells_forward_overrides_and_work_outside_repo(tmp_path):
    shim = tmp_path / 'python-shim'
    shim.write_text('#!/usr/bin/env python3\nimport json,sys,os\nprint(json.dumps({"args":sys.argv[1:],"cwd":os.getcwd(),"gpu":os.environ["CUDA_VISIBLE_DEVICES"]}))\n')
    shim.chmod(0o755)
    scripts = sorted(p for p in (ROOT / 'scripts/Exp_VQA').glob('vqa_*.sh')
                     if p.name not in ('vqa_run01.sh', 'vqa_run02.sh'))
    assert len(scripts) == 21
    for script in scripts:
        subprocess.run(['bash', '-n', str(script)], check=True)
        result = subprocess.run(['bash', str(script), '--dry-run', '--ratios', '64', '--set', 'training.projector_lr=0.0003'],
                                cwd=tmp_path, env={**os.environ, 'PROJECTOR_PYTHON': str(shim), 'CUDA_VISIBLE_DEVICES': '0',
                                                  'VQA_PREPROCESSING_VERSION': 'v2'}, capture_output=True, text=True, check=True)
        payload = json.loads(result.stdout)
        argv = payload['args']
        assert argv[:3] == ['-u', '-m', 'src.cli.vqa_sweep']
        args = sweep.parser().parse_args(argv[3:])
        assert args.ratios == [64] and args.preprocessing_version == 'v2'
        assert args.overrides[-1] == 'training.projector_lr=0.0003'
        assert payload['cwd'] == str(ROOT) and payload['gpu'] == '0'


@pytest.mark.parametrize('variant', ['adirect', 'adirect_localread', 'adirect_detail', 'adirect_alllocal'])
def test_adirect_vqa_launchers_keep_report_architecture_and_shared_vqa_recipe(variant, tmp_path):
    """The new wrappers change architecture only, and retain VQA's entire training/eval contract."""
    capture = tmp_path / 'capture-python'
    capture.write_text('#!' + sys.executable + '\nimport json,os,sys\n'
                       'print(json.dumps({"args":sys.argv[1:],"gpu":os.environ["CUDA_VISIBLE_DEVICES"]}))\n')
    capture.chmod(0o700)
    local = tmp_path / 'local.yaml'
    local.write_text('{}\n')
    env = {**os.environ, 'PROJECTOR_PYTHON': str(capture), 'PROJECTOR_LOCAL_CONFIG': str(local),
           'CUDA_VISIBLE_DEVICES': '0', 'VQA_PREPROCESSING_VERSION': 'v2',
           'VQA_INVALID_SAMPLE_POLICY': 'include_unreviewed'}
    env.pop('PROJECTOR_RUN_TAG', None)

    def captured_args(folder, filename, parser, *extra):
        proc = subprocess.run(['bash', str(ROOT / 'scripts' / folder / filename), '--dry-run', *extra],
                              cwd=tmp_path, env=env, check=True, text=True, capture_output=True)
        payload = json.loads(proc.stdout)
        assert payload['gpu'] == '0'
        return parser().parse_args(payload['args'][3:])

    args = captured_args('Exp_VQA', f'vqa_baseline_{variant}.sh', sweep.parser)
    assert args.ratios == ([16] if variant == 'adirect_localread' else [128, 256])
    assert args.evaluate_test and args.eval_controls == ['normal']
    assert args.eval_split == 'test' and args.eval_limit is None and args.eval_ids_file is None
    assert args.bootstrap == 1000
    expected_name = 'bindformer_adirect' if variant == 'adirect_alllocal' else f'bindformer_{variant}'
    assert args.projector == expected_name
    if variant == 'adirect_alllocal':
        assert args.run_tag.startswith('baseline_adirect_alllocal_')

    # Compare every nonprojector field against the existing production VQA baseline.
    reference_args = captured_args('Exp_VQA', 'vqa_baseline_a.sh', sweep.parser,
                                   '--ratios', *map(str, args.ratios))
    reference = sweep.build_launches(reference_args)
    launches = sweep.build_launches(args)
    for (config, spec), (baseline, baseline_spec) in zip(launches, reference):
        saved = tmp_path / f'{variant}_r{config.projector.target_compression_ratio}.yaml'
        save_config(config, saved)
        assert OmegaConf.to_container(load_config([saved]), resolve=True) == OmegaConf.to_container(config, resolve=True)
        current = OmegaConf.to_container(config, resolve=True)
        expected = OmegaConf.to_container(baseline, resolve=True)
        for value in (current, expected):
            value.pop('projector')
            value['training'].pop('run_dir')
        assert current == expected
        assert config.training.init_checkpoint is None and config.training.resume is None
        assert config.training.stage == 'joint_lora' and config.decoder.frozen
        assert config.training.projector_lr == config.training.encoder_lr == 2e-4
        assert config.training.projector_warmup_steps == config.training.joint_warmup_steps == 150
        assert config.training.projector_hold_steps == 0
        assert config.training.joint_warmup_projector_mode == 'hold_peak'
        assert config.training.cosine_min_lr_ratio == 1
        evaluation = {k: v for k, v in spec['evaluation'].items() if k != 'root'}
        assert evaluation == {k: v for k, v in baseline_spec['evaluation'].items() if k != 'root'}
        train, evaluations = sweep.commands(config, spec)
        assert 'src.cli.vqa_train' in train and len(evaluations) == 1
        command = evaluations[0][1]
        assert command[command.index('--checkpoint') + 1] == str(Path(spec['run']) / 'train/checkpoints/best.pt')

    # Report/VQA have different grids and budgets. All remaining projector settings must match.
    from src.cli.baseline_sweep import parser as report_parser
    report_args = captured_args('Exp1026', f'run_baseline_{variant}.sh', report_parser,
                                '--ratios', '128', '256')
    files = [ROOT / 'configs/training/joint_lora.yaml', ROOT / 'configs/training/deterministic.yaml',
             ROOT / 'configs/preprocessing/native_grid.yaml',
             ROOT / f'configs/projector/{report_args.projector}.yaml', *report_args.config]
    report = load_config(files, report_args.overrides + ['projector.target_compression_ratio=128'])
    vqa_projector, report_projector = launches[0][0].projector, report.projector
    assert vqa_projector.name == report_projector.name
    assert vqa_projector.hidden_size == report_projector.hidden_size and vqa_projector.bias == report_projector.bias
    vqa_params, report_params = (OmegaConf.to_container(p.params, resolve=True)
                                for p in (vqa_projector, report_projector))
    for params in (vqa_params, report_params):
        params.pop('num_tokens', None)
    assert vqa_params == report_params

    args.ratios = [16, 32, 64, 128, 256]
    all_budgets = sweep.build_launches(args)
    assert [c.projector.resolved_budget.expected_output_tokens for c, _ in all_budgets] == [624, 288, 150, 75, 36]
    if variant == 'adirect_alllocal':
        for c, _ in all_budgets:
            assert c.projector.resolved_budget.global_tokens == 0
            assert c.projector.resolved_budget.local_tokens == c.projector.resolved_budget.expected_output_tokens


def test_placement_and_override_precedence():
    args = sweep.parser().parse_args(['--projector', 'concat3d', '--concat-device', 'cuda:1',
                                    '--set', 'training.projector_lr=0.0002', '--set', 'training.projector_lr=0.0003'])
    c, _ = sweep.build_launches(args)[0]
    assert c.training.projector_lr == 0.0003
    sweep.validate_placement(c, '0,2')
    for visible in ('0', '0,0', '0,1,2', ''):
        with pytest.raises(ValueError):
            sweep.validate_placement(c, visible)


def test_resume_uses_saved_recipe_not_changed_shell_defaults(tmp_path):
    args = sweep.parser().parse_args(['--ratios', '128', '--run-tag', 'unit', '--output-root', str(tmp_path)])
    c, spec = sweep.build_launches(args)[0]
    run = Path(spec['run'])
    run.mkdir(parents=True)
    save_config(c, run / 'launch_config.yaml')
    write_json(run / 'launch.json', spec)
    resumed = sweep.parser().parse_args(['--resume-run', str(run), '--preprocessing-version', 'v2',
                                        '--set', 'training.projector_lr=1', '--eval-tag', 'retry'])
    old, plan = sweep.build_launches(resumed)[0]
    assert old.training.projector_lr == c.training.projector_lr
    assert old.preprocessing.name == c.preprocessing.name
    train, evaluations = sweep.commands(old, plan, resume=True)
    assert train[-1] == f'training.resume={run}/train/checkpoints/last.pt'
    assert evaluations[0][0].parent.name == 'retry'


@pytest.fixture
def data_fixture(tmp_path, monkeypatch):
    cache = tmp_path / 'cache'
    manifest = tmp_path / 'manifest'
    write_json(manifest / 'lineage.json', {'preprocessing_root': str(cache)})
    c = load_config([ROOT / 'configs/vqa/neuroqa/mixed.yaml'], [
        f'preprocessing.cache_dir={cache}', f'data.manifest_dir={manifest}',
        'training.stage=a', 'training.encoder_lr=0', 'training.max_epochs=1', 'data.invalid_sample_policy=strict'])
    samples = []
    for i in range(4):
        folder = cache / 'cases' / f'case{i}'
        contract = {'source_sha256': {}}
        write_json(folder / 'complete.json', {'contract': contract, 'fingerprint': digest(contract),
            'artifacts': {f'{m}_mni_{suffix}.nii.gz': 'hash' for m in ('T1w','T2w') for suffix in ('n4','mask')}})
        images = {}
        for m in ('T1w', 'T2w'):
            image = folder / f'{m}_mni_n4.nii.gz'
            image.touch()
            (folder / f'{m}_mni_mask.nii.gz').touch()
            images[m] = str(image)
        row = {'neuroqa_id': f'q{i}', 'allowed_modalities': ['T1w', 'T2w'],
               'question_type': 'MCQ', 'images': images}
        samples.append(QASample(f'q{i}', 'train', 'BraTS_GLI', f'p{i}', f'case{i}', '', '', '', row))
    pop = {'train': [s.view('T1w') for s in samples[:2]], 'val': [samples[2].view('T1w')], 'test': [samples[3].view('T1w')]}
    monkeypatch.setattr(preflight, 'load_neuroqa', lambda *a, **kw: (pop, {}))
    class FakePre:
        version = 'test'
        spatial_contract = {}
        cache = SimpleNamespace(put=lambda *a: None)

        def __call__(self, sample):
            return SimpleNamespace(patches=SimpleNamespace(shape=(9555, 1024)), metadata={'grid_dhw': [49, 15, 13]})

    monkeypatch.setattr(preflight, 'build_preprocessor', lambda c: FakePre())
    monkeypatch.setattr('src.preprocessing.neuroqa_smri.compatible_contract', lambda a, b: a == b)
    monkeypatch.setattr(preflight, 'prepare_budget', lambda *a: {'status': 'passed_header_preflight'})
    monkeypatch.setattr(preflight, 'geometry_from_header', lambda *a: ([49,15,13], [196,240,208]))
    return c, pop, cache


def test_preflight_missing_second_modality_never_silently_drops_qa(data_fixture):
    c, pop, cache = data_fixture
    write_json(cache / 'finalization.json', {'ready_for_full_training': True})
    Path(pop['train'][0].view('T2w').mask).unlink()
    r = preflight.inspect_data(c)
    assert r['status'] == 'blocked' and r['train_affected_qa'] == 1
    assert r['train_qa'] == 2 and r['epoch_modalities'] == {'T1W': 1, 'T2W': 1}
    assert list(r['missing_cases']) == ['case0']


def test_strict_requires_qc_even_for_smoke_and_labels_scope_after_acceptance(data_fixture):
    c, _, cache = data_fixture
    r = preflight.inspect_data(c)
    assert r['status'] == 'blocked' and 'ready_for_full_training' in r['errors'][0]
    c.training.max_steps = 1
    assert preflight.inspect_data(c)['status'] == 'blocked'
    write_json(cache / 'finalization.json', {'ready_for_full_training': True})
    r = preflight.inspect_data(c)
    assert r['status'] == 'passed' and r['scope'] == 'smoke'
    write_json(cache / 'cases/case0/complete.json', {'contract': {'source_sha256': {}}, 'fingerprint': 'tampered', 'artifacts': {}})
    assert 'fingerprint mismatch' in preflight.inspect_data(c)['invalid_cases']['case0'][0]


def test_preflight_rejects_cache_only_switch(data_fixture, tmp_path):
    c, _, _ = data_fixture
    c.preprocessing.cache_dir = str(tmp_path / 'wrong-version')
    r = preflight.inspect_data(c)
    assert r['status'] == 'blocked'
    assert any('matching manifest' in e for e in r['errors'])
    assert any('outside selected' in e for e in r['errors'])


def report_stub():
    return {'scope': 'training', 'policy': 'strict', 'warnings': [], 'invalid_cases': {},
            'skipped': {}, 'requested_counts': {}, 'retained_ids': {}}


def test_launcher_train_before_eval_and_completed_resume_is_idempotent(tmp_path, monkeypatch):
    args = sweep.parser().parse_args(['--ratios', '128', '--run-tag', 'unit', '--output-root', str(tmp_path / 'runs'),
                                     '--results-root', str(tmp_path / 'results')])
    c, spec = sweep.build_launches(args)[0]
    run = Path(spec['run'])
    calls = []

    def fake_run(command, log):
        calls.append(command)
        if 'src.cli.vqa_train' in command:
            write_json(run / 'train/result.json', {'scope': 'training'})
            checkpoint = run / 'train/checkpoints/best.pt'
            checkpoint.parent.mkdir()
            checkpoint.write_bytes(b'checkpoint')
        else:
            output = Path(command[command.index('--output') + 1])
            write_json(output / 'metrics.json', {'scope': 'full_split'})

    monkeypatch.setattr(sweep, 'run_logged', fake_run)
    sweep.execute(c, spec, args, report_stub())
    assert len(calls) == 2 and 'src.cli.vqa_train' in calls[0] and 'src.cli.vqa_evaluate' in calls[1]
    args.resume_run = run
    sweep.execute(c, spec, args, report_stub())
    assert len(calls) == 2
    assert json.loads((run / 'launcher_status.json').read_text())['status'] == 'completed'


def test_training_failure_prevents_test_evaluation(tmp_path, monkeypatch):
    args = sweep.parser().parse_args(['--ratios', '128', '--output-root', str(tmp_path), '--run-tag', 'fail'])
    c, spec = sweep.build_launches(args)[0]
    calls = []

    def fail(command, log):
        calls.append(command)
        raise subprocess.CalledProcessError(1, command)

    monkeypatch.setattr(sweep, 'run_logged', fail)
    with pytest.raises(subprocess.CalledProcessError):
        sweep.execute(c, spec, args, report_stub())
    assert len(calls) == 1
    assert json.loads((Path(spec['run']) / 'launcher_status.json').read_text())['status'] == 'failed'


def test_skip_mode_keeps_odd_remainder_and_reports_failed_case(data_fixture):
    c, pop, cache = data_fixture
    c.data.invalid_sample_policy = 'skip'
    Path(pop['train'][0].view('T2w').mask).unlink()
    kept, report, _, _ = preflight.prepare_data(c)
    assert report['status'] == 'passed' and report['scope'] == 'filtered_training'
    assert not report['qc_ready_for_full_training'] and report['warnings']
    assert [s.qa_id for s in kept['train']] == ['q1']
    assert [r['id'] for r in report['skipped']['train']] == ['q0']
    assert report['train_qa'] == 1  # No extra deletion for global/even modality quotas.
    assert sum(report['dual_eligible_modalities'].values()) == 1


def test_skip_handles_corruption_without_swallowing_global_config_errors(data_fixture, monkeypatch):
    c, _, _ = data_fixture
    c.data.invalid_sample_policy = 'skip'
    check = preflight._check_case

    def corrupt(config, pre, sample):
        if sample.case_id == 'case0':
            raise ValueError('Nonfinite patch data')
        return check(config, pre, sample)

    monkeypatch.setattr(preflight, '_check_case', corrupt)
    kept, r, _, _ = preflight.prepare_data(c)
    assert r['status'] == 'passed' and len(kept['train']) == 1
    assert 'Nonfinite' in r['invalid_cases']['case0'][0]
    c.preprocessing.cache_dir += '_wrong'
    assert preflight.inspect_data(c)['status'] == 'blocked'


def test_no_remaining_train_or_closed_val_still_blocks(data_fixture):
    c, pop, _ = data_fixture
    c.data.invalid_sample_policy = 'skip'
    for sample in pop['train']:
        Path(sample.view('T1w').mask).unlink()
    assert 'No usable training' in ';'.join(preflight.inspect_data(c)['errors'])


def test_filtered_eval_reports_coverage_and_requires_nonempty_selection(data_fixture):
    c, pop, _ = data_fixture
    c.data.invalid_sample_policy = 'skip'
    evaluation = {'split': 'test', 'controls': ['normal']}
    kept, r, _, _ = preflight.prepare_data(c, evaluation, include_training=False)
    assert r['status'] == 'passed' and len(kept['evaluation']) == 1
    assert r['requested_counts']['evaluation'] == 1
    Path(pop['test'][0].mask).unlink()
    kept, r, _, _ = preflight.prepare_data(c, evaluation, include_training=False)
    assert r['status'] == 'blocked' and not kept['evaluation']
    assert r['skipped']['evaluation'][0]['qa_id'] == 'q3'


def test_resume_rejects_repaired_or_newly_failed_input_selection(data_fixture):
    c, pop, _ = data_fixture
    c.data.invalid_sample_policy = 'skip'
    original = preflight.inspect_data(c)
    path = Path(pop['train'][0].view('T2w').mask)
    path.unlink()
    filtered = preflight.inspect_data(c)
    with pytest.raises(ValueError, match='selection/artifacts changed'):
        preflight.ensure_same_selection(original, filtered)
    path.touch()
    repaired = preflight.inspect_data(c)
    preflight.ensure_same_selection(original, repaired)
    with pytest.raises(ValueError, match='selection/artifacts changed'):
        preflight.ensure_same_selection(filtered, repaired)


def test_existing_qc_review_queue_is_honored(data_fixture):
    from src.config import file_hash
    c, _, cache = data_fixture
    c.data.invalid_sample_policy = 'skip'
    queue = cache / 'qc_review_queue.json'
    write_json(queue, {'cases': [{'case_id': 'case0', 'reasons': ['outside_canvas']}]})
    assert 'case0' in preflight.inspect_data(c)['invalid_cases']
    write_json(cache / 'qc_review.json', {'queue_sha256': file_hash(queue),
                                         'reviews': {'case0': {'decision': 'accepted'}}})
    assert not preflight.inspect_data(c)['invalid_cases']


def test_include_unreviewed_keeps_warning_cases_but_skips_missing_artifacts(data_fixture, monkeypatch):
    c, pop, cache = data_fixture
    c.data.invalid_sample_policy = 'include_unreviewed'
    c.data.modality_sampling = 'dual_eligible'
    write_json(cache / 'qc_review_queue.json', {'cases': [{'case_id': 'case0', 'reasons': ['outside_canvas']}]})
    monkeypatch.setattr(preflight, '_verify_unreviewed_v2', lambda config, root: {})
    kept, report, _, _ = preflight.prepare_data(c)
    assert report['status'] == 'passed'
    assert report['scope'] == 'unreviewed_training'
    assert len(kept['train']) == 2 and 'case0' in report['included_unreviewed_cases']
    assert not report['skipped']['train'] and not report['qc_ready_for_full_training']
    Path(pop['train'][0].view('T2w').mask).unlink()
    kept, report, _, _ = preflight.prepare_data(c)
    assert report['status'] == 'passed' and 'case0' in report['invalid_cases']
    assert 'case0' not in report['included_unreviewed_cases']
    assert [sample.qa_id for sample in kept['train']] == ['q1']
    assert [row['id'] for row in report['skipped']['train']] == ['q0']
    assert report['requested_counts']['train'] == 2 and report['train_qa'] == 1


def test_include_unreviewed_skips_corruption_in_evaluation(data_fixture, monkeypatch):
    c, pop, cache = data_fixture
    c.data.invalid_sample_policy = 'include_unreviewed'
    write_json(cache / 'qc_review_queue.json', {'cases': []})
    monkeypatch.setattr(preflight, '_verify_unreviewed_v2', lambda config, root: {})
    # Retain a second test view so evaluation can still proceed.
    pop['test'].append(pop['val'][0])
    (cache / 'cases/case3/complete.json').write_text('{invalid json')
    kept, report, _, _ = preflight.prepare_data(
        c, {'split': 'test', 'controls': ['normal']}, include_training=False)
    assert report['status'] == 'passed' and len(kept['evaluation']) == 1
    assert report['requested_counts']['evaluation'] == 2
    assert report['evaluation_views'] == 1
    assert report['skipped']['evaluation'][0]['case_id'] == 'case3'
    assert 'JSONDecodeError' in report['invalid_cases']['case3'][0]


def test_include_unreviewed_does_not_waive_global_audit_failure(data_fixture, monkeypatch):
    c, _, cache = data_fixture
    c.data.invalid_sample_policy = 'include_unreviewed'
    write_json(cache / 'qc_review_queue.json', {'cases': []})

    def fail(*args):
        raise ValueError('Full v2 engineering audit failed')

    monkeypatch.setattr(preflight, '_verify_unreviewed_v2', fail)
    report = preflight.inspect_data(c)
    assert report['status'] == 'blocked'
    assert any('engineering audit failed' in error for error in report['errors'])


def test_include_unreviewed_filters_local_audit_failures_not_warnings(data_fixture, monkeypatch):
    c, _, cache = data_fixture
    c.data.invalid_sample_policy = 'include_unreviewed'
    c.data.modality_sampling = 'dual_eligible'
    write_json(cache / 'qc_review_queue.json', {'cases': [
        {'case_id':'case0', 'reasons':['outside_canvas']},
        {'case_id':'case1', 'reasons':['low_nmi']} ]})
    monkeypatch.setattr(preflight, '_verify_unreviewed_v2',
                        lambda config, root: {'case0':['Independent artifact audit: invalid cache']})
    kept, report, _, _ = preflight.prepare_data(c)
    assert report['status'] == 'passed' and [s.qa_id for s in kept['train']] == ['q1']
    assert report['skipped']['train'][0]['case_id'] == 'case0'
    assert list(report['included_unreviewed_cases']) == ['case1']
    assert 'invalid cache' in report['invalid_cases']['case0'][0]


def test_include_unreviewed_requires_fresh_complete_v2_audits(tmp_path):
    from src.config import file_hash
    c = SimpleNamespace(preprocessing=SimpleNamespace(name='neuroqa_mni1mm_v2'),
                        data=SimpleNamespace(manifest_dir=str(tmp_path)))
    (tmp_path/'source_index.jsonl').write_text(json.dumps({'case_id':'case0'})+'\n')
    audit = {'summary': {'expected_cases':1,'audited_cases':1,'ready_for_full_training':True},
             'finished_utc':'now', 'cases':[{'case_id':'case0','completed':True,'integrity_errors':[],
                'volumes':[{'modality':m,'errors':[],'patch_recomputed_exact':True} for m in ('T1w','T2w')]}]}
    write_json(tmp_path/'independent_audit.json', audit)
    complete_audit = {'expected_cases':1,'complete':True,'finished_utc':'now',
                      'cases':[{'case_id':'case0'}], 'failed':[]}
    write_json(tmp_path/'intensity_audit.json', complete_audit)
    write_json(tmp_path/'lesion_coverage_audit.json', complete_audit)
    write_json(tmp_path/'production_reproducibility.json', {'status': 'passed'})
    write_json(tmp_path/'finalization.json', {'expected_cases':1,'expected_volumes':2,
        'verified_volumes':[{'case_id':'case0','modality':m} for m in ('T1w','T2w')], 'failed':[]})
    names = {'audit_sha256':'independent_audit.json', 'intensity_audit_sha256':'intensity_audit.json',
             'lesion_audit_sha256':'lesion_coverage_audit.json', 'reproducibility_sha256':'production_reproducibility.json'}
    queue = {key:file_hash(tmp_path/name) for key,name in names.items()}
    queue['cases'] = []
    write_json(tmp_path/'qc_review_queue.json', queue)
    assert preflight._verify_unreviewed_v2(c, tmp_path) == {}
    # A completed audit may contain case-local failures: record those cases
    # for filtering while retaining global coverage and evidence checks.
    audit['cases'][0]['integrity_errors'] = ['corrupt cached artifact']
    audit['summary']['ready_for_full_training'] = False
    write_json(tmp_path/'independent_audit.json', audit)
    queue['audit_sha256'] = file_hash(tmp_path/'independent_audit.json')
    write_json(tmp_path/'qc_review_queue.json', queue)
    failures = preflight._verify_unreviewed_v2(c, tmp_path)
    assert 'corrupt cached artifact' in ';'.join(failures['case0'])
    write_json(tmp_path/'qc_review.json', {'queue_sha256':file_hash(tmp_path/'qc_review_queue.json'),
        'reviews':{'case0':{'decision':'rejected'}}})
    assert 'explicitly rejected' in ';'.join(preflight._verify_unreviewed_v2(c, tmp_path)['case0'])
    write_json(tmp_path/'intensity_audit.json', {'expected_cases':1, 'finished_utc':'now',
        'complete':False, 'cases':[], 'failed':[{'case_id':'case0','error':'invalid source'}]})
    queue['intensity_audit_sha256'] = file_hash(tmp_path/'intensity_audit.json')
    write_json(tmp_path/'qc_review_queue.json', queue)
    write_json(tmp_path/'finalization.json', {'expected_cases':1, 'expected_volumes':2,
        'verified_volumes':[{'case_id':'case0','modality':'T1w'}],
        'failed':[{'case_id':'case0','modality':'T2w','error':'broken patch'}]})
    failures = ';'.join(preflight._verify_unreviewed_v2(c, tmp_path)['case0'])
    assert 'invalid source' in failures and 'broken patch' in failures
    # An unidentifiable failed volume cannot safely be attributed to a case.
    write_json(tmp_path/'finalization.json', {'expected_cases':1, 'expected_volumes':2,
        'verified_volumes':[{'case_id':'case0','modality':'T1w'}],
        'failed':[{'case_id':'unknown','modality':'T2w','error':'broken'}]})
    with pytest.raises(ValueError, match='unidentifiable'):
        preflight._verify_unreviewed_v2(c, tmp_path)
    write_json(tmp_path/'intensity_audit.json', {'complete': False})
    with pytest.raises(ValueError, match='evidence mismatch'):
        preflight._verify_unreviewed_v2(c, tmp_path)


def test_cli_policy_and_new_sampling_default_are_persisted(tmp_path):
    args = sweep.parser().parse_args(['--ratios', '128'])
    c, spec = sweep.build_launches(args)[0]
    assert c.data.invalid_sample_policy == 'skip' and c.data.modality_sampling == 'dual_eligible'
    args.invalid_sample_policy = 'strict'
    c, spec = sweep.build_launches(args)[0]
    assert c.data.invalid_sample_policy == 'strict'
    assert spec['config']['data']['invalid_sample_policy'] == 'strict'
    # An old saved config without the new fields must preserve its original sampler.
    del c.data.invalid_sample_policy
    del c.data.modality_sampling
    save_config(c, tmp_path / 'old.yaml')
    legacy = load_config([tmp_path / 'old.yaml'])
    assert legacy.data.invalid_sample_policy == 'strict' and legacy.data.modality_sampling == 'legacy_exact'


def test_corrupt_completion_record_is_a_sample_failure(data_fixture):
    c, _, cache = data_fixture
    c.data.invalid_sample_policy = 'skip'
    write_json(cache / 'cases/case0/complete.json', None)
    r = preflight.inspect_data(c)
    assert r['status'] == 'passed' and r['train_qa'] == 1
    assert 'Malformed' in r['invalid_cases']['case0'][0]


def test_geometry_failure_is_skipped_before_training(data_fixture, monkeypatch):
    c, _, _ = data_fixture
    c.data.invalid_sample_policy = 'skip'
    monkeypatch.setattr(preflight, 'geometry_from_header',
                        lambda p, c: ([1,1,1], [1,1,1]) if '/case0/' in p else ([49,15,13], [196,240,208]))
    r = preflight.inspect_data(c)
    assert r['status'] == 'passed' and r['train_qa'] == 1
    assert 'geometry' in r['invalid_cases']['case0'][0]


def test_allocation_failure_is_not_misclassified_as_bad_data(data_fixture, monkeypatch):
    c, _, _ = data_fixture
    c.data.invalid_sample_policy = 'skip'

    def oom(*args):
        raise RuntimeError('CPU out of memory')

    monkeypatch.setattr(preflight, '_check_case', oom)
    with pytest.raises(RuntimeError, match='out of memory'):
        preflight.inspect_data(c)


def test_filtered_results_require_explicit_summary_inclusion(tmp_path):
    import csv
    import sys
    root = tmp_path / 'results'
    for scope in ('full_split', 'filtered_evaluation', 'subset_smoke'):
        folder = root / scope
        write_json(folder / 'metrics.json', {'scope': scope, 'training_scope': 'filtered_training',
            'split': 'test', 'control': 'normal', 'data_policy': 'skip', 'population_size': 10,
            'skipped_views': 2, 'overall': {'samples': 8, 'closed_accuracy': 0.5}})
        write_json(folder / 'provenance.json', {'contract_id': 'test',
            'decoder_loss': 'answer_only_causal_ce_per_qa_mean_v1',
            'projector': {'name': 'avgpool3d', 'resolved_budget': {'target_ratio': 128, 'expected_output_tokens': 75}}})
    output = tmp_path / 'summary.csv'
    command = [sys.executable, str(ROOT / 'scripts/vqa/neuroqa/summarize_results.py'),
               '--root', str(root), '--output', str(output)]
    subprocess.run(command, check=True, capture_output=True)
    assert [r['scope'] for r in csv.DictReader(output.open())] == ['full_split']
    subprocess.run(command + ['--include-filtered'], check=True, capture_output=True)
    rows = list(csv.DictReader(output.open()))
    assert {r['scope'] for r in rows} == {'full_split', 'filtered_evaluation'}
    assert all(r['skipped_views'] == '2' for r in rows)


@pytest.mark.parametrize('dual_count', [0, 1, 3, 4])
def test_dual_sampling_preserves_all_fixed_questions_and_alternates(dual_count, data_fixture):
    from dataclasses import replace
    from src.data.neuroqa_manifest import balanced_epoch
    from collections import Counter
    _, pop, _ = data_fixture
    base = pop['train'][0]
    samples = []
    for i in range(dual_count + 7):
        row = {**base.row, 'neuroqa_id': f'x{i}',
               'allowed_modalities': ['T1w', 'T2w'] if i < dual_count else ['T2w']}
        samples.append(replace(base, id=f'x{i}', row=row))
    a = balanced_epoch(samples, 42, 0)
    b = balanced_epoch(samples, 42, 1)
    assert len(a) == len(b) == len(samples) and len({s.qa_id for s in a}) == len(samples)
    by_id = {s.qa_id: s for s in b}
    dual = Counter(s.sequence for s in a if len(s.row['allowed_modalities']) == 2)
    assert abs(dual['T1W'] - dual['T2W']) <= 1
    for s in a:
        if len(s.row['allowed_modalities']) == 2:
            assert s.sequence != by_id[s.qa_id].sequence
        else:
            assert s.sequence == by_id[s.qa_id].sequence == 'T2W'
    assert Counter(s.sequence for s in a)['T2W'] >= 7
