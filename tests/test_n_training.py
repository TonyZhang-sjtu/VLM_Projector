"""Exercise the real Trainer loop with a small CPU model and controlled val CE."""
from contextlib import nullcontext
import json
from types import SimpleNamespace

import pytest
import torch

from test_joint_training import TinyModel, initial_state, joint_config
from src.training import trainer as module
from src.training.optim import build_optimizer, build_joint_scheduler
from src.training.schedule import JointSchedule


class ReportModel(TinyModel):
    def forward(self, items):
        x = torch.stack([item['x'] for item in items])
        loss = super().forward(x) * len(items)
        return {'loss_sum': loss, 'answer_tokens': len(items),
                'encoder_token_counts': [8] * len(items), 'projector_token_counts': [1] * len(items)}


@pytest.mark.parametrize('early', [False, True])
def test_trainer_real_loop_fixed_epoch_save_best_and_diagnostics(tmp_path, monkeypatch, early):
    config = joint_config('training.max_epochs=20', 'training.microbatch_per_device=1',
                          'training.effective_batch_size=1', 'training.validation_generation_samples=0',
                          'training.checkpoint_epochs=[5,10,15,20]', 'training.log_optimizer_diagnostics=true',
                          f'training.early_stopping_enabled={str(early).lower()}',
                          'training.cosine_min_lr_ratio=1.0')
    trainer = module.Trainer.__new__(module.Trainer)
    trainer.config, trainer.output = config, tmp_path
    trainer.rank, trainer.world, trainer.device, trainer.accumulation = 0, 1, torch.device('cpu'), 1
    trainer.total_steps, trainer.windows_per_epoch = 160, 8
    trainer.plan = JointSchedule(160, 2, 2, cosine_min_lr_ratio=1., joint_warmup_projector_mode='hold_peak')
    trainer.model = ReportModel(config)
    trainer.model.decoder.answer_count = lambda sample: 1
    trainer.wrapped = trainer.model
    trainer.optimizer = build_optimizer(trainer.model, config)
    trainer.scheduler = build_joint_scheduler(trainer.optimizer, trainer.plan)
    trainer.state = initial_state(trainer.plan)
    trainer.provenance = {'contract_id': 'test'}
    class Dataset:
        samples = list(range(8))
        def __len__(self): return len(self.samples)
        def __getitem__(self, index): return {'x': torch.ones(4) * (index + 1)}
    trainer.train_data, trainer.val_data = Dataset(), Dataset()
    monkeypatch.setattr(torch, 'autocast', lambda *args, **kwargs: nullcontext())
    for name in ('reset_peak_memory_stats', 'synchronize'):
        monkeypatch.setattr(torch.cuda, name, lambda *args: None)
    for name in ('max_memory_allocated', 'max_memory_reserved'):
        monkeypatch.setattr(torch.cuda, name, lambda *args: 0)
    # Best at epoch 2; subsequent plateaus must stop enabled runs at epoch 5,
    # but must not prevent N1-N3 from saving epochs 10/15/20.
    values = iter([1.4, 1.3] + [1.31] * 18)
    monkeypatch.setattr(module, 'validate', lambda *args, **kwargs:
                        {'answer_ce': next(values), 'samples': 8, 'answer_tokens': 8})
    trainer.train()
    checkpoints = tmp_path / 'checkpoints'
    saved_epochs = sorted(p.name for p in checkpoints.glob('epoch_*.pt'))
    assert saved_epochs == (['epoch_05.pt'] if early else
                            ['epoch_05.pt', 'epoch_10.pt', 'epoch_15.pt', 'epoch_20.pt'])
    best = torch.load(checkpoints / 'best.pt', weights_only=False)
    assert best['state']['epoch'] == 2 and best['state']['best_val_ce'] == 1.3
    last = torch.load(checkpoints / 'last.pt', weights_only=False)
    assert last['state']['epoch'] == (5 if early else 20)
    assert last['state']['stop_reason'] == ('early_stopping' if early else 'max_epochs')
    assert last['scheduler']['last_epoch'] == (40 if early else 160)
    diagnostics = [json.loads(line) for line in (tmp_path / 'optimizer_diagnostics.jsonl').read_text().splitlines()]
    assert len(diagnostics) == (40 if early else 160)
    assert diagnostics[1]['components']['encoder']['parameters_with_grad'] == 0
    assert diagnostics[1]['components']['encoder']['update_norm'] == 0
    assert diagnostics[3]['components']['encoder']['update_norm'] > 0
    assert diagnostics[-1]['learning_rates']['projector'] == config.training.projector_lr
    epochs = [json.loads(line) for line in (tmp_path / 'epoch_metrics.jsonl').read_text().splitlines()]
    assert len(epochs) == (5 if early else 20)
    assert all(row['full_epoch'] and row['updates'] == 8 for row in epochs)
    # All three phases use the same optimizer state; snapshots are resumable.
    assert int(trainer.optimizer.state[trainer.model.projector.weight]['step']) == trainer.state['global_step']


def test_green_diagnostics_excludes_failures_and_preserves_error_order(tmp_path):
    from src.config import digest
    from src.evaluation.ft_diagnostics import summarize_evaluation
    rows = []
    for i in range(2):
        rows.append(dict(id=str(i), patient_id='p' + str(i), case_id='c' + str(i), cohort='BraTS_GLI',
                         sequence='t1n', split='test', reference='ref', prediction='pred', checkpoint_id='sha',
                         generated_token_count=10 + 10 * i, hit_generation_limit=False, empty_prediction=False))
    scored = [dict(row) for row in rows]
    counts, matched = [1, 2, 3, 4, 5, 6], 7
    scored[0]['clinical'] = {'green': {'status': 'ok', 'values': {'green': .25},
                                       'details': {'matched_findings': matched, 'significant_errors': counts}}}
    scored[1]['clinical'] = {'green': {'status': 'format_error', 'values': {}, 'details': {}}}
    for subdir, name, values in [('generation', 'predictions.jsonl', rows),
                                  ('scoring', 'clinical_scores.jsonl', scored)]:
        path = tmp_path / subdir / name; path.parent.mkdir()
        path.write_text(''.join(json.dumps(row) + '\n' for row in values))
    result = {'checkpoint_sha256': 'sha', 'overall': {'green': .25}}
    summary = summarize_evaluation(tmp_path, result, digest(['0', '1']))
    assert summary['green_valid'] == summary['green_failed'] == 1
    assert summary['mean_generated_tokens'] == 15
    assert summary['green_valid_only_means']['omitted_finding'] == 2
    assert summary['green_valid_only_means']['unsupported_comparison'] == 5
    assert summary['green_valid_only_means']['green'] == .25
    assert (tmp_path / 'generation_green_samples.csv').is_file()
    scored[0]['clinical']['green']['details']['matched_findings'] = 8
    (tmp_path / 'scoring/clinical_scores.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in scored))
    with pytest.raises(ValueError, match='formula'):
        summarize_evaluation(tmp_path, result, digest(['0', '1']))
