"""N-series contracts: real AdamW schedules, launcher arguments and artifact identity."""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch

from src.cli import ft_experiments as ft
from src.config import ROOT, digest, load_config
from src.training.diagnostics import OptimizerDiagnostics, epoch_training_summary
from src.training.optim import build_joint_scheduler
from src.training.schedule import JointSchedule, update_joint_validation


RECIPES = [(1, 5e-5, 5e-5, 1., 20, False), (2, 1e-4, 1e-4, 1., 20, False),
           (3, 2e-4, 2e-4, 1., 20, False), (4, 5e-5, 4e-5, .2, 10, True),
           (5, 5e-5, 2e-5, .2, 10, True), (6, 5e-5, 5e-5, .5, 10, True)]


@pytest.mark.parametrize('index,plr,llr,floor,epochs,early', RECIPES)
def test_n_launcher_and_real_adamw_rates(tmp_path, index, plr, llr, floor, epochs, early):
    capture = tmp_path / 'python'
    capture.write_text('#!' + sys.executable + '\nimport sys,json,os\n'
                       'print(json.dumps({"args":sys.argv[1:],"gpu":os.environ["CUDA_VISIBLE_DEVICES"]}))\n')
    capture.chmod(0o700)
    local = tmp_path / 'local paths.yaml'; local.write_text('{}\n')
    env = dict(os.environ, PROJECTOR_PYTHON=str(capture), PROJECTOR_LOCAL_CONFIG=str(local),
               CUDA_VISIBLE_DEVICES='2', PROJECTOR_OUTPUT_ROOT=str(tmp_path / 'outputs'))
    script = ROOT / f'scripts/ft_ablation/n{index}.sh'
    subprocess.run(['bash', '-n', str(script)], check=True)
    result = subprocess.run(['bash', str(script), '--dry-run'], env=env, cwd=tmp_path,
                            capture_output=True, text=True, check=True)
    value = json.loads(result.stdout); args = value['args']
    assert value['gpu'] == '2'
    assert args[:3] == ['-u', '-m', 'src.cli.ft_experiments']
    parsed = ft.sweep.parser().parse_args(args[3:])
    assert parsed.require_full_test and parsed.eval_split == 'test'
    assert parsed.eval_sample_fraction == 1 and parsed.metrics == ['all']
    c = load_config([ROOT / 'configs/training/joint_lora.yaml', *parsed.config], parsed.overrides)
    t = c.training
    assert (t.projector_lr, t.encoder_lr, t.cosine_min_lr_ratio, t.max_epochs,
            t.early_stopping_enabled) == (plr, llr, floor, epochs, early)
    assert list(t.checkpoint_epochs) == ([5, 10, 15, 20] if epochs == 20 else [10])
    assert t.log_optimizer_diagnostics and t.lora.rank == 16 and t.lora.alpha == 32
    assert t.resume is None and t.init_checkpoint is None
    p, l = torch.nn.Parameter(torch.ones(1)), torch.nn.Parameter(torch.ones(1))
    opt = torch.optim.AdamW([{'params': [p], 'lr': plr, 'component': 'projector'},
                            {'params': [l], 'lr': llr, 'component': 'encoder'}])
    schedule = JointSchedule(123 * epochs, 77, 77, cosine_min_lr_ratio=floor,
                             joint_warmup_projector_mode='hold_peak')
    scheduler = build_joint_scheduler(opt, schedule)
    trace = []
    for step in range(schedule.total_steps):
        trace.append([g['lr'] for g in opt.param_groups])
        p.grad = torch.ones_like(p)
        l.grad = None if step < 77 else torch.ones_like(l)
        opt.step(); scheduler.step()
    assert trace[0] == [0., 0.]
    assert trace[76] == trace[77] == [plr, 0.]
    assert trace[153] == trace[154] == [plr, llr]
    assert trace[-1] == [plr * floor, llr * floor]
    if floor == 1:
        assert all(rates == [plr, llr] for rates in trace[154:])
    assert int(opt.state[p]['step']) == 123 * epochs
    assert int(opt.state[l]['step']) == 123 * epochs - 77


@pytest.mark.parametrize('setting', ['training.cosine_min_lr_ratio=1.01',
    'training.early_stopping_enabled=1', 'training.log_optimizer_diagnostics=1',
    'training.checkpoint_epochs=[0]', 'training.checkpoint_epochs=[5,5]',
    'training.checkpoint_epochs=[10,5]', 'training.checkpoint_epochs=[11]',
    'training.checkpoint_epochs=[true]', 'training.checkpoint_epochs=5'])
def test_n_invalid_config(setting):
    with pytest.raises(ValueError):
        load_config([ROOT / 'configs/training/joint_lora.yaml'], [setting])


def test_disabled_early_stop_still_saves_the_true_minimum():
    state = {'best_val_ce': float('inf'), 'bad_epochs': 0, 'early_stop_reference': None,
             'stop_reason': None, 'global_step': 2460}
    improved = [update_joint_validation(state, v, eligible=True, min_delta=.005, patience=3, enabled=False)
                for v in [1.4, 1.3, 1.299, 1.4, 1.5, 1.6, 1.7]]
    assert improved == [True, True, True, False, False, False, False]
    assert state['best_val_ce'] == 1.299 and state['bad_epochs'] == 0 and state['stop_reason'] is None


def test_diagnostics_measure_actual_updates_without_changing_adam_or_rng():
    p = torch.nn.Parameter(torch.tensor([3., 4.]))
    l = torch.nn.Parameter(torch.zeros(2))
    other = torch.nn.Parameter(p.detach().clone())
    opt = torch.optim.AdamW([{'params': [p], 'component': 'projector'},
                            {'params': [l], 'component': 'encoder'}], lr=.1, weight_decay=.1)
    control = torch.optim.AdamW([other], lr=.1, weight_decay=.1)
    p.grad = torch.tensor([6., 8.]); other.grad = p.grad.clone()
    rng = torch.get_rng_state().clone()
    before = p.detach().clone()
    diagnostics = OptimizerDiagnostics(opt)
    torch.nn.utils.clip_grad_norm_([p, l], 1.)
    torch.nn.utils.clip_grad_norm_([other], 1.)
    diagnostics.after_clip(); opt.step(); control.step()
    measured = diagnostics.after_step()
    torch.testing.assert_close(p, other, rtol=0, atol=0)
    assert torch.equal(torch.get_rng_state(), rng)
    d = measured['projector']
    assert d['grad_norm_pre_clip'] == pytest.approx(10.)
    assert d['grad_norm_post_clip'] == pytest.approx(1.)
    assert d['update_norm'] == pytest.approx(float((p.detach() - before).norm()))
    assert d['relative_update_norm'] == pytest.approx(d['update_norm'] / 5)
    assert measured['encoder']['update_norm'] == 0
    assert measured['encoder']['relative_update_norm'] is None
    assert l not in opt.state


def test_epoch_summary_is_token_weighted(tmp_path):
    path = tmp_path / 'train.jsonl'
    rows = [{'epoch': 0, 'step': 1, 'answer_tokens': 2, 'answer_ce': 3., 'answer_loss_sum': 6.},
            {'epoch': 0, 'step': 2, 'answer_tokens': 6, 'answer_ce': 1., 'answer_loss_sum': 6.}]
    path.write_text(''.join(json.dumps(r) + '\n' for r in rows))
    assert epoch_training_summary(path, 0)['train_answer_ce'] == 1.5
    with path.open('a') as handle: handle.write(json.dumps(rows[0]) + '\n')
    with pytest.raises(ValueError, match='duplicate'): epoch_training_summary(path, 0)


@pytest.fixture
def plan(tmp_path, monkeypatch):
    population = {split: [SimpleNamespace(id=f'{split}_{i}', image='fake.nii.gz', cohort='BraTS_GLI')
                          for i in range(count)] for split, count in [('train', 1960), ('val', 268), ('test', 560)]}
    monkeypatch.setattr('src.data.manifest.load_manifests', lambda c: (population, {'test': {'sha256': 'manifest'}}))
    monkeypatch.setattr('src.models.projectors.budget.geometry_from_header', lambda *a: ([39, 15, 15], [156., 240., 240.]))
    monkeypatch.setenv('WORLD_SIZE', '1')
    args = ft.sweep.parser().parse_args(['--projector', 'avgpool3d', '--ratios', '8', '--output-root', str(tmp_path),
        '--run-tag', 'n_test', '--config', str(ROOT / 'configs/ft_ablation/avgpool3d_r8.yaml'),
        '--set', 'training.checkpoint_epochs=[10]', '--set', 'training.log_optimizer_diagnostics=true'])
    result = ft.build_plan(args)
    assert not Path(result['run_root']).exists()
    return result


def save_fake_checkpoint(run, epoch, name):
    path = Path(run['checkpoint']).with_name(name)
    path.parent.mkdir(parents=True, exist_ok=True)
    saved = {'format_version': 2, 'stage': 'joint_lora', 'config': run['config'],
             'provenance': {'contract_id': 'same'}, 'projector': {'weight': torch.ones(2, 2)},
             'encoder': {'lora_B': torch.zeros(2, 2)},
             'state': {'epoch': epoch, 'global_step': epoch * 123, 'next_window': 0, 'best_val_ce': 1.2,
                       'schedule': run['schedule'], 'train_selection': run['data_identity']['train_ids_sha256'],
                       'val_selection': run['data_identity']['val_ids_sha256']}}
    torch.save(saved, path)
    return path


def test_checkpoint_epoch_and_weights_identity(plan):
    run = plan['runs'][0]
    a = save_fake_checkpoint(run, 10, 'epoch_10.pt')
    b = save_fake_checkpoint(run, 10, 'best.pt')
    assert ft.checkpoint_metadata(a, run, 10)['model_identity_sha256'] == ft.checkpoint_metadata(b, run)['model_identity_sha256']
    with pytest.raises(ValueError, match='epoch'): ft.checkpoint_metadata(a, run, 5)
    changed = copy.deepcopy(run); changed['config']['training']['encoder_lr'] *= 2
    with pytest.raises(ValueError, match='config'): ft.checkpoint_metadata(a, changed)
    saved = torch.load(b, weights_only=False); saved['encoder']['lora_B'][0, 0] = 1
    torch.save(saved, b)
    assert ft.checkpoint_metadata(a, run)['model_identity_sha256'] != ft.checkpoint_metadata(b, run)['model_identity_sha256']


@pytest.mark.parametrize('early', [False, True])
def test_orchestration_evaluates_fixed_epoch_and_best_or_marks_missing(plan, monkeypatch, early):
    run = plan['runs'][0]
    calls = []
    def execute(command, log, append=False):
        calls.append(command)
        if 'src.cli.train' in command:
            epoch = 7 if early else 10
            best = save_fake_checkpoint(run, epoch, 'best.pt')
            if not early: save_fake_checkpoint(run, 10, 'epoch_10.pt')
            (best.parent.parent / 'training_state.json').write_text(json.dumps(
                {'stop_reason': 'early_stopping' if early else 'max_epochs', 'epoch': epoch}))
        else:
            directory = Path(command[command.index('--output-dir') + 1]); directory.mkdir(parents=True)
            checkpoint = command[command.index('--checkpoint') + 1]
            ids = [f'test_{i}' for i in range(560)]
            result = {'status': 'partial', 'split': 'test', 'scope': 'full_split',
                'evaluated_samples': 560, 'split_population': 560, 'checkpoint_sha256': ft.file_hash(checkpoint),
                'manifest_summary': run['manifest_summary'], 'overall': dict.fromkeys(
                    ['bleu4', 'rougeL', 'green', 'radgraph_simple', 'radgraph_partial', 'radgraph_complete',
                     'rate', 'bertscore_precision', 'bertscore_recall', 'bertscore_f1', 'answer_ce'], .5),
                'partial_metrics': {'green': {'valid': 559, 'failed': 1}}}
            (directory / 'metric.json').write_text(json.dumps(result))
            (directory / 'expected_ids.json').write_text(json.dumps(ids))
    monkeypatch.setattr(ft.sweep, 'run_logged', execute)
    monkeypatch.setattr(ft, 'summarize_evaluation', lambda *args: {'green_valid': 559, 'green_failed': 1})
    ft.execute_plan(plan)
    assert len(calls) == 2  # One training, one evaluation; best duplicates epoch 10.
    rows = json.loads((Path(run['directory']) / 'checkpoint_results.json').read_text())['checkpoints']
    assert [r['checkpoint_label'] for r in rows] == ['epoch_10', 'best']
    assert rows[0]['status'] == ('not_reached' if early else 'partial')
    if not early:
        assert rows[1]['reused_from'] == 'epoch_10'
        assert (Path(run['directory']) / 'evaluation/best_test/alias.json').is_file()
    ft.execute_plan(plan, resume=True)
    assert len(calls) == 2  # Completed evaluations are audited and reused.


def test_result_audit_rejects_wrong_checkpoint_and_split(plan, tmp_path):
    run = plan['runs'][0]
    result = {'split': 'test', 'scope': 'full_split', 'evaluated_samples': 560, 'split_population': 560,
              'overall': dict.fromkeys(['bleu4', 'rougeL', 'answer_ce', 'green', 'radgraph_simple',
                 'radgraph_partial', 'radgraph_complete', 'rate', 'bertscore_precision', 'bertscore_recall', 'bertscore_f1'], .5),
              'status': 'complete', 'checkpoint_sha256': 'wrong'}
    with pytest.raises(ValueError, match='SHA256'):
        ft.audit_result(result, run, {'checkpoint_sha256': 'right'}, tmp_path)
    with pytest.raises(ValueError, match='full test'):
        ft.audit_result(dict(result, split='val'), run, {}, tmp_path)


@pytest.mark.parametrize('failure_label', [None, 'epoch_10'])
def test_twenty_epoch_orchestration_and_failed_evaluation_resume(plan, monkeypatch, failure_label):
    run = plan['runs'][0]
    run['config']['training'].update(max_epochs=20, checkpoint_epochs=[5, 10, 15, 20],
                                     early_stopping_enabled=False)
    run['schedule'] = JointSchedule(2460, 77, 77, cosine_min_lr_ratio=.2,
                                    joint_warmup_projector_mode='hold_peak').to_dict()
    run['evaluations'] = ft.evaluation_jobs(run, 'test')
    calls = []
    fail_once = [failure_label]
    def execute(command, log, append=False):
        if 'src.cli.train' in command:
            calls.append('train')
            for epoch in (5, 10, 15, 20): save_fake_checkpoint(run, epoch, f'epoch_{epoch:02d}.pt')
            best = save_fake_checkpoint(run, 10, 'best.pt')
            (best.parent.parent / 'training_state.json').write_text(json.dumps({'stop_reason': 'max_epochs', 'epoch': 20}))
        else:
            directory = Path(command[command.index('--output-dir') + 1])
            label = directory.name.removesuffix('_test')
            calls.append(label)
            if label == fail_once[0]:
                fail_once[0] = None
                raise subprocess.CalledProcessError(1, command)
            directory.mkdir(parents=True)
            checkpoint = command[command.index('--checkpoint') + 1]
            result = {'status': 'complete', 'split': 'test', 'scope': 'full_split',
                      'evaluated_samples': 560, 'split_population': 560,
                      'manifest_summary': run['manifest_summary'], 'checkpoint_sha256': ft.file_hash(checkpoint),
                      'overall': dict.fromkeys(['bleu4', 'rougeL', 'answer_ce', 'green', 'radgraph_simple',
                          'radgraph_partial', 'radgraph_complete', 'rate', 'bertscore_precision',
                          'bertscore_recall', 'bertscore_f1'], .5)}
            (directory / 'metric.json').write_text(json.dumps(result))
            (directory / 'expected_ids.json').write_text(json.dumps([f'test_{i}' for i in range(560)]))
    monkeypatch.setattr(ft.sweep, 'run_logged', execute)
    monkeypatch.setattr(ft, 'summarize_evaluation', lambda *args: {'green_valid': 560, 'green_failed': 0})
    if failure_label:
        with pytest.raises(subprocess.CalledProcessError): ft.execute_plan(plan)
        assert calls == ['train', 'epoch_05', 'epoch_10']
        state = json.loads((Path(plan['run_root']) / 'sweep_status.json').read_text())
        assert state['status'] == 'failed' and state['active_checkpoint'] == failure_label
        ft.execute_plan(plan, resume=True)
        assert calls == ['train', 'epoch_05', 'epoch_10', 'epoch_10', 'epoch_15', 'epoch_20']
    else:
        ft.execute_plan(plan)
        assert calls == ['train', 'epoch_05', 'epoch_10', 'epoch_15', 'epoch_20']
    rows = json.loads((Path(run['directory']) / 'checkpoint_results.json').read_text())['checkpoints']
    assert [r['checkpoint_label'] for r in rows] == ['epoch_05', 'epoch_10', 'epoch_15', 'epoch_20', 'best']
    assert rows[-1]['reused_from'] == 'epoch_10' and rows[-1]['epoch'] == 10
