"""1019 entrypoint, schedule, structural ablation and best-test contracts (CPU)."""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from src.cli import baseline_sweep as sweep
from src.cli.evaluate_checkpoint import ALL_METRICS
from src.config import ROOT, load_config
from src.models.projectors.registry import build_projector
from src.training.optim import build_joint_scheduler
from src.training.schedule import JointSchedule, update_joint_validation
from test_baseline_sweep import population  # Shared fake manifests/headers fixture.
from test_bindformer_modules import config as tiny_config, volume


ENTRIES = ROOT / 'scripts/baselines/BindFormer'


def python_stub(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f'#!{sys.executable}\n'
                    'import json,os,sys\n'
                    'print(json.dumps({"args":sys.argv[1:],"cwd":os.getcwd(),'
                    '"gpu":os.environ.get("CUDA_VISIBLE_DEVICES"),'
                    '"python":os.environ.get("PROJECTOR_PYTHON")}))\n')
    path.chmod(0o755)
    return path


def capture(tmp_path, variant, extra=(), env_changes=None):
    stub = python_stub(tmp_path / 'capture python')
    local_config = tmp_path / 'local paths.yaml'
    local_config.write_text('{}\n')
    env = dict(os.environ)
    for key in ('PROJECTOR_RUN_TAG', 'PROJECTOR_OUTPUT_ROOT', 'WORLD_SIZE'):
        env.pop(key, None)
    env.update(PROJECTOR_PYTHON=str(stub), CUDA_VISIBLE_DEVICES='3',
               PROJECTOR_LOCAL_CONFIG=str(local_config))
    env.update(env_changes or {})
    script = ENTRIES / f'run_bindformer_{variant}_constant_lr_1019.sh'
    result = subprocess.run(['bash', str(script), '--dry-run', *extra],
                            cwd=tmp_path, env=env, text=True, capture_output=True)
    if result.returncode:
        raise AssertionError(result.stderr)
    record = json.loads(result.stdout)
    assert record['cwd'] == str(ROOT)
    assert record['args'][:3] == ['-u', '-m', 'src.cli.baseline_sweep']
    return sweep.parser().parse_args(record['args'][3:]), record


@pytest.mark.parametrize('variant,enabled', [('vanilla', False), ('sa_ca', True)])
def test_entrypoint_plan_and_full_test_contract(tmp_path, population, variant, enabled):
    args, record = capture(tmp_path, variant, ['--ratios', '4', '8', '16', '32', '64'])
    assert record['gpu'] == '3'
    assert args.run_tag.startswith(f'bindformer_{variant}_1019_p2e-4_l2e-4_')
    plan = sweep.build_plan(args)
    sweep.validate_full_test_plan(plan)
    assert plan['require_full_test'] and plan['eval_split'] == 'test'
    assert set(plan['metrics']) == set(ALL_METRICS)
    assert plan['ratios'] == [4, 8, 16, 32, 64]
    for run in plan['runs']:
        c, s = run['config'], run['schedule']
        t, p = c['training'], c['projector']['params']
        assert c['projector']['name'] == 'bindformer3d'
        assert (p['within_slot_sa'], p['local_reads_global']) == (enabled, enabled)
        assert p['within_slot_sa_depth'] == p['local_global_depth'] == 1
        assert p['local_global_gate_init'] == .1
        assert p['knowledge_mode'] == 'name_definition' and p['vocabulary_scope'] == 'brats_train'
        assert p['slot_iterations'] == 3 and p['global_fraction'] == .25
        assert set(p['role_query_counts'].values()) == {1}
        assert p['geometry_enabled'] is False and p['checkpoint_blocks'] is True
        for loss in ('feature', 'text', 'tuple', 'binding', 'concept'):
            assert p[f'{loss}_loss_weight'] == 0
        assert t['projector_lr'] == t['encoder_lr'] == 2e-4
        assert t['max_epochs'] == 20 and t['stage'] == 'joint_lora'
        assert t['max_steps'] is t['init_checkpoint'] is t['resume'] is None
        assert t['effective_batch_size'] == 16 and t['microbatch_per_device'] == 1
        assert t['early_stopping_enabled'] is True
        assert t['early_stopping_patience'] == 3 and t['early_stopping_min_delta'] == .005
        assert t['checkpoint_epochs'] == [] and t['log_optimizer_diagnostics'] is True
        assert t['lora']['rank'] == 16 and t['lora']['alpha'] == 32
        assert c['decoder']['frozen'] and not t['feature_cache']
        assert (s['projector_steps'], s['projector_hold_steps'], s['joint_steps'],
                s['cosine_steps'], s['total_steps']) == (77, 0, 77, 2306, 2460)
        assert s['cosine_min_lr_ratio'] == 1 and s['joint_warmup_projector_mode'] == 'hold_peak'
        assert (run['train_samples'], run['val_samples'], run['evaluation_samples'],
                run['updates_per_epoch']) == (1960, 268, 560, 123)
        assert run['evaluation_scope'] == 'full_split'
        assert Path(run['checkpoint']).name == 'best.pt'
        assert run['checkpoint'] in run['eval_command']
        assert '/evaluation/best_test/' in run['metric_path']
    assert not Path(plan['run_root']).exists()


def test_two_variants_differ_only_in_requested_interactions(tmp_path, population):
    plans = [sweep.build_plan(capture(tmp_path, v)[0]) for v in ('vanilla', 'sa_ca')]
    assert plans[0]['run_root'] != plans[1]['run_root']
    for a, b in zip(plans[0]['runs'], plans[1]['runs']):
        assert a['data_identity'] == b['data_identity']
        assert a['schedule'] == b['schedule'] and a['budget'] == b['budget']
        ca, cb = copy.deepcopy(a['config']), copy.deepcopy(b['config'])
        for c in (ca, cb):
            c['training'].pop('run_dir')
            for key in ('within_slot_sa', 'local_reads_global'):
                c['projector']['params'].pop(key)
        assert ca == cb


def test_cli_edits_paths_and_fixed_evaluation_scope(tmp_path, population):
    paths = tmp_path / 'server paths.yaml'
    paths.write_text('encoder:\n  path: /other server/models/encoder\nprojector:\n'
                     '  params:\n    concept_bank: /other server/knowledge\n')
    args, _ = capture(tmp_path, 'sa_ca', [
        '--ratios', '32', '8', '--run-tag', 'my_new_run',
        '--set', 'training.projector_lr=3e-4', '--set', 'training.encoder_lr=1e-4',
        '--set', 'training.max_epochs=25', '--set', 'projector.params.within_slot_sa=false',
        '--eval-split', 'val', '--metrics', 'answer_ce', '--eval-sample-fraction', '.1'],
        {'PROJECTOR_LOCAL_CONFIG': str(paths)})
    plan = sweep.build_plan(args)
    sweep.validate_full_test_plan(plan)
    assert plan['ratios'] == [8, 32] and args.run_tag == 'my_new_run'
    c = plan['runs'][0]['config']
    assert c['encoder']['path'] == '/other server/models/encoder'
    assert c['projector']['params']['concept_bank'] == '/other server/knowledge'
    assert c['training']['projector_lr'] == 3e-4 and c['training']['encoder_lr'] == 1e-4
    assert c['training']['max_epochs'] == 25
    assert c['projector']['params']['within_slot_sa'] is False
    assert c['projector']['params']['local_reads_global'] is True


@pytest.mark.parametrize('explicit', [False, True])
def test_conda_python_and_explicit_override(tmp_path, explicit):
    conda = python_stub(tmp_path / 'conda env/bin/python')
    override = python_stub(tmp_path / 'override python')
    args, record = capture(tmp_path, 'vanilla', env_changes={
        'CONDA_PREFIX': str(conda.parent.parent),
        'PROJECTOR_PYTHON': str(override) if explicit else ''})
    assert record['python'] == str(override if explicit else conda)
    assert args.projector == 'bindformer3d'


def test_actual_adamw_lr_for_all_2460_updates(tmp_path, population):
    plan = sweep.build_plan(capture(tmp_path, 'sa_ca')[0])
    schedule = JointSchedule.from_dict(plan['runs'][0]['schedule'])
    p, lora = torch.nn.Parameter(torch.ones(1)), torch.nn.Parameter(torch.ones(1))
    optimizer = torch.optim.AdamW([
        {'params': [p], 'lr': 2e-4, 'component': 'projector'},
        {'params': [lora], 'lr': 2e-4, 'component': 'encoder'}], foreach=False)
    scheduler = build_joint_scheduler(optimizer, schedule)
    rates = []
    for step in range(2460):
        rates.append([g['lr'] for g in optimizer.param_groups])
        p.grad = torch.ones_like(p)
        lora.grad = torch.ones_like(lora) if step >= 77 else None
        optimizer.step()
        if step == 76:
            momentum = optimizer.state[p]['exp_avg']
            assert lora not in optimizer.state
        if step >= 77:
            assert optimizer.state[p]['exp_avg'] is momentum
        scheduler.step()
    assert rates[0] == [0., 0.]
    assert rates[76] == [2e-4, 0.]
    assert rates[77] == [2e-4, 0.]
    assert rates[153] == [2e-4, 2e-4]
    assert all(row == [2e-4, 2e-4] for row in rates[154:])
    assert optimizer.state[p]['step'] == 2460
    assert optimizer.state[lora]['step'] == 2460 - 77


def test_early_stop_and_best_are_independent():
    state = {'best_val_ce': float('inf'), 'bad_epochs': 0,
             'early_stop_reference': None, 'stop_reason': None, 'global_step': 123}
    assert update_joint_validation(state, 1.6, eligible=False, min_delta=.005, patience=3)
    assert state['early_stop_reference'] is None
    flags = []
    for value in (1.3, 1.299, 1.298, 1.297):
        state['global_step'] += 123
        flags.append(update_joint_validation(state, value, eligible=True,
                                             min_delta=.005, patience=3))
    assert flags == [True] * 4 and state['best_val_ce'] == 1.297
    assert state['stop_reason'] == 'early_stopping'


@pytest.mark.parametrize('variant', ['vanilla', 'sa_ca'])
@pytest.mark.parametrize('stop_reason', ['early_stopping', 'max_epochs'])
def test_every_ratio_evaluates_best_after_training(tmp_path, population, monkeypatch, variant, stop_reason):
    args, _ = capture(tmp_path, variant, ['--output-root', str(tmp_path / 'runs'),
                                        '--ratios', '4', '8', '16', '32', '64'])
    plan = sweep.build_plan(args)
    events = []
    from src.evaluation.clinical import METRIC_KEYS
    keys = {'bleu4', 'rougeL', 'answer_ce'}
    for names in METRIC_KEYS.values():
        keys.update(names)

    def fake_run(command, log_path, append=False):
        run = next(r for r in plan['runs'] if Path(log_path).parent == Path(r['directory']))
        if 'src.cli.train' in command:
            events.append(('train', run['ratio']))
            checkpoint = Path(run['checkpoint'])
            checkpoint.parent.mkdir(parents=True)
            checkpoint.write_bytes(b'mock checkpoint; no VLM training')
            (checkpoint.parent.parent / 'training_state.json').write_text(
                json.dumps({'stop_reason': stop_reason}))
        else:
            events.append(('eval', run['ratio']))
            assert command[command.index('--checkpoint') + 1] == run['checkpoint']
            assert command[command.index('--split') + 1] == 'test'
            metric = Path(run['metric_path'])
            metric.parent.mkdir(parents=True)
            metric.write_text(json.dumps({'status': 'complete', 'split': 'test',
                'scope': 'full_split', 'evaluated_samples': 560, 'split_population': 560,
                'overall': dict.fromkeys(keys, .5)}))

    monkeypatch.setattr(sweep, 'run_logged', fake_run)
    sweep.execute_plan(plan)
    assert events == [(phase, r) for r in (4, 8, 16, 32, 64) for phase in ('train', 'eval')]


def test_exact_sa_ca_combination_has_gradients_and_preserves_core_initialization():
    models = []
    for enabled in (False, True):
        torch.manual_seed(1019)
        c = tiny_config('projector.params.num_tokens=20',
                        f'projector.params.within_slot_sa={str(enabled).lower()}',
                        f'projector.params.local_reads_global={str(enabled).lower()}')
        models.append(build_projector(c, 8, 16))
    a, b = models
    for name, parameter in a.named_parameters():
        torch.testing.assert_close(parameter, dict(b.named_parameters())[name], rtol=0, atol=0)
    assert not list(a.interactions.parameters())
    assert len(b.interactions.within) == len(b.interactions.cross) == 1
    assert torch.allclose(b.interactions.cross[0].gate.bias.sigmoid(), torch.tensor([.1]))
    assert a.allocation == b.allocation
    x, coords, metadata = volume()
    for model in models:
        output = model(x.clone().requires_grad_(), coords, metadata)
        assert output.embeddings.shape == (20, 16)
        assert model.geometry_encoder is None and model.feature_reconstructor is None
        output.embeddings.square().mean().backward()
        for parameter in model.interactions.parameters():
            assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
            assert parameter.grad.abs().sum() > 0
