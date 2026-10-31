"""Task-specific orchestration, reusing the shared model/optimizer/checkpoint core.

The v1 runner deliberately rejects DDP: dual-eligible quotas and resume are
validated in one process, including CPU Concat placement. No report sampling or
report-generation scorer is involved.
"""
from collections import Counter
import json
import math
import os
from pathlib import Path
import time

from omegaconf import OmegaConf
import torch
from tqdm.auto import tqdm

from src.config import digest, path_of, save_config
from src.data.neuroqa_manifest import balanced_epoch, write_json
from src.data.neuroqa_preflight import prepare_data, save_data_report, ensure_same_selection, selected
from src.data.neuroqa_runtime import prepare_window
from src.evaluation.neuroqa import predict, save_predictions
from src.models.multimodal import MultimodalModel, place_model
from src.training.checkpoint import load_checkpoint, provenance, save_checkpoint
from src.training.optim import build_optimizer, build_scheduler, build_joint_scheduler, clip_grad_norm_across_devices
from src.training.schedule import resolve_joint_schedule
from src.training.stages import assert_stage
from src.training.trainer import seed_all


class VQATrainer:
    def __init__(self, config, expected_selection=None):
        if config.task.name != 'neuroqa' or int(os.environ.get('WORLD_SIZE', '1')) != 1:
            raise ValueError('NeuroQA v1 requires one process')
        self.config = config
        self.output = path_of(config, config.training.run_dir)
        if self.output.exists() and any(self.output.iterdir()) and not config.training.resume:
            raise FileExistsError(f'Run directory is not empty: {self.output}')
        self.output.mkdir(parents=True, exist_ok=True)
        seed_all(config.seed)
        torch.set_num_threads(min(4, torch.get_num_threads()))
        self.device = torch.device(config.training.device)
        populations, self.data_report, self.preprocessor, self.summary = prepare_data(config)
        if expected_selection:
            ensure_same_selection(expected_selection, self.data_report)
        if config.training.resume:
            prior = torch.load(path_of(config, config.training.resume), map_location='cpu', weights_only=False, mmap=True)
            ensure_same_selection(prior['state'], self.data_report)
            del prior
        save_data_report(self.output, self.data_report)
        if self.data_report['status'] != 'passed':
            raise ValueError('VQA data checks failed: ' + '; '.join(self.data_report['errors']))
        self.train_samples, self.val_samples = populations['train'], populations['val']
        self.preflight = self.data_report['geometry']
        torch.cuda.set_device(self.device)
        first_epoch = balanced_epoch(self.train_samples, config.seed, 0,
                                     config.data.get('modality_sampling', 'legacy_exact'))
        self.steps_per_epoch = math.ceil(len(first_epoch) / config.training.effective_batch_size)
        self.total_steps = self.steps_per_epoch * config.training.max_epochs
        if config.training.max_steps:
            self.total_steps = min(self.total_steps, config.training.max_steps)
        self.plan = resolve_joint_schedule(config.training, self.total_steps) if config.training.stage == 'joint_lora' else None
        self.model = place_model(MultimodalModel(config), self.device)
        self.report = assert_stage(self.model)
        self.optimizer = build_optimizer(self.model, config)
        self.scheduler = build_joint_scheduler(self.optimizer, self.plan) if self.plan else build_scheduler(
            self.optimizer, self.total_steps, config.training.warmup_ratio, config.training.warmup_steps)
        self.provenance = provenance(config, self.model, self.preprocessor, self.summary)
        self.state = {'epoch': 0, 'next_window': 0, 'global_step': 0, 'world_size': 1,
                      'best_accuracy': -1., 'best_val_ce': float('inf'), 'early_stop_reference': None,
                      'bad_epochs': 0, 'train_selection': digest([s.qa_id for s in self.train_samples]),
                      'val_selection': digest([s.id for s in self.val_samples]),
                      'modality_counts': {'T1W': 0, 'T2W': 0}, 'stop_reason': None,
                      'runtime_invalid_cases': {},
                      'scope': self.data_report['scope'],
                      'selection_fingerprint': self.data_report['selection_fingerprint'],
                      'data_policy': config.data.get('invalid_sample_policy', 'strict'),
                      'modality_sampling': config.data.get('modality_sampling', 'legacy_exact')}
        if self.plan:
            self.state.update(schedule=self.plan.to_dict(), phase=self.plan.phase(0))
        if config.training.resume:
            old = load_checkpoint(path_of(config, config.training.resume), self.model, 'resume',
                                  self.optimizer, self.scheduler, self.provenance)
            for key in ('train_selection', 'val_selection', 'world_size'):
                if old['state'][key] != self.state[key]:
                    raise ValueError(f'Resume changed {key}')
            allowed = {'resume', 'run_dir', 'save_every_steps'}
            now = OmegaConf.to_container(config.training, resolve=True)
            if {k: v for k, v in now.items() if k not in allowed} != {
                    k: v for k, v in old['config']['training'].items() if k not in allowed}:
                raise ValueError('Resume changed the VQA training recipe')
            self.state = old['state']
        save_config(config, self.output / 'config.yaml')
        write_json(self.output / 'provenance.json', self.provenance)
        write_json(self.output / 'preflight.json', self.preflight)
        write_json(self.output / 'trainable_parameters.json', self.report)

    def save(self, name):
        save_checkpoint(self.output / 'checkpoints' / name, self.model, self.optimizer,
                        self.scheduler, self.state, self.provenance)

    def log(self, name, value):
        with (self.output / name).open('a') as handle:
            handle.write(json.dumps(value) + '\n')

    def train(self):
        c = self.config
        trainable = [p for group in self.optimizer.param_groups for p in group['params']]
        batch, micro = c.training.effective_batch_size, c.training.microbatch_per_device
        print(f'VQA training: {c.projector.name} | {len(self.train_samples)} QA | '
              f'{c.training.max_epochs} epochs | batch={batch}, microbatch={micro} | '
              f'{self.steps_per_epoch} updates/epoch | resume step={self.state["global_step"]}\n'
              f'Detailed metrics: {self.output / "train.jsonl"}\n'
              'Progress ETA is for the current epoch or validation phase.', flush=True)
        for epoch in range(self.state['epoch'], c.training.max_epochs):
            if self.state['global_step'] >= self.total_steps or self.state['stop_reason'] == 'early_stopping':
                break
            order = balanced_epoch(self.train_samples, c.seed, epoch, c.data.get('modality_sampling', 'legacy_exact'))
            multiplicity = Counter(s.qa_id for s in order)
            plan = {'epoch': epoch, 'counts': dict(Counter(s.sequence for s in order)),
                    'dual_eligible_counts': dict(Counter(s.sequence for s in order if len(s.row['allowed_modalities']) == 2)),
                    'unique_qa': len(multiplicity), 'total_views': len(order),
                    'repeated_views': len(order) - len(multiplicity),
                    'repeated_qa_counts': {qa: count for qa, count in multiplicity.items() if count > 1},
                    'sha256': digest([s.id for s in order]), 'views': [s.id for s in order]}
            write_json(self.output / 'sampling' / f'epoch_{epoch+1:03d}.json', plan)
            start_window = self.state['next_window'] if self.state['epoch'] == epoch else 0
            self.model.train()
            end_window = min(self.steps_per_epoch, start_window + self.total_steps - self.state['global_step'])
            with tqdm(range(start_window, end_window), total=end_window, initial=start_window,
                      desc=f'Epoch {epoch+1}/{c.training.max_epochs}', mininterval=2, dynamic_ncols=True,
                      bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [ETA {remaining}{postfix}]') as steps:
                for window_index in steps:
                    window = order[window_index*batch:(window_index+1)*batch]
                    # Load the whole window before backward. A localized bad MRI
                    # can then be replaced with a same-sequence view without
                    # retaining partial gradients or changing the loss divisor.
                    invalid = self.state.setdefault('runtime_invalid_cases', {})
                    items, events = prepare_window(
                        window, order, self.preprocessor, seed=c.seed, epoch=epoch,
                        window_index=window_index,
                        skip_invalid=c.data.get('invalid_sample_policy', 'strict') in ('skip', 'include_unreviewed'),
                        invalid_cases=invalid)
                    if events:
                        self.log('runtime_data_events.jsonl', {'epoch': epoch+1, 'window': window_index,
                                 'step_before_update': self.state['global_step'], 'events': events,
                                 'actual_views': [x['sample'].id for x in items]})
                        print(f'WARNING VQA: runtime data skip/replacement in epoch {epoch+1}, '
                              f'window {window_index}; see runtime_data_events.jsonl', flush=True)
                    counts = Counter(x['sample'].sequence for x in items)
                    if c.data.get('modality_sampling', 'legacy_exact') in ('legacy_exact', 'exact_balanced_repeat') and counts['T1W'] != counts['T2W']:
                        raise AssertionError('Unbalanced optimizer window')
                    if self.plan:
                        self.model.encoder.set_phase(self.plan.phase(self.state['global_step']))
                    phase_used = self.model.encoder.phase
                    assert_stage(self.model)
                    self.optimizer.zero_grad(set_to_none=True)
                    start = time.monotonic()
                    torch.cuda.reset_peak_memory_stats(self.device)
                    total = 0.
                    tokens = 0
                    for i in range(0, len(items), micro):
                        with torch.autocast('cuda', dtype=torch.bfloat16):
                            result = self.model(items[i:i+micro])
                            loss = result['example_loss_sum'] / len(window)
                            feature_weight = getattr(self.model.projector, 'params', {}).get('feature_loss_weight', 0.)
                            if feature_weight:
                                loss = loss + feature_weight * result['feature_loss_sum'] / len(window)
                        if not torch.isfinite(loss):
                            raise FloatingPointError('Nonfinite VQA loss')
                        loss.backward()
                        total += float(result['example_loss_sum'].detach())
                        tokens += result['answer_tokens']
                    gradients = {}
                    for component in ('projector', 'encoder'):
                        norms = [float(p.grad.detach().float().norm()) for p in getattr(self.model, component).parameters() if p.grad is not None]
                        gradients[component] = math.hypot(*norms)
                    if gradients['projector'] == 0 or any(p.grad is not None for p in self.model.decoder.parameters()):
                        raise AssertionError('Missing projector gradient or unfrozen decoder')
                    if not self.model.encoder.vision_frozen and gradients['encoder'] == 0:
                        raise AssertionError('Missing encoder LoRA gradient')
                    norm = clip_grad_norm_across_devices(trainable, c.training.grad_clip_norm)
                    rates = [g['lr'] for g in self.optimizer.param_groups]
                    self.optimizer.step()
                    self.scheduler.step()
                    self.state['global_step'] += 1
                    self.state.update(epoch=epoch, next_window=window_index+1)
                    for seq, count in counts.items():
                        self.state['modality_counts'][seq] += count
                    if self.plan:
                        self.state['phase'] = self.plan.phase(self.state['global_step'])
                        self.model.encoder.set_phase(self.state['phase'])
                    self.log('train.jsonl', {'step': self.state['global_step'], 'epoch': epoch+1,
                         'phase_used': phase_used, 'next_phase': self.model.encoder.phase,
                         'qa_ce': total/len(window), 'answer_tokens': tokens, 'window_samples': len(window),
                         'modality_counts': dict(counts), 'cumulative_modalities': self.state['modality_counts'],
                         'gradient_norms': gradients, 'global_gradient_norm': norm, 'learning_rates': rates,
                         'encoder_tokens': result['encoder_token_counts'], 'projector_tokens': result['projector_token_counts'],
                         'peak_cuda_bytes': torch.cuda.max_memory_allocated(self.device), 'seconds': time.monotonic()-start})
                    steps.set_postfix(CE=f'{total/len(window):.4f}', refresh=False)
                    if c.training.save_every_steps and self.state['global_step'] % c.training.save_every_steps == 0:
                        self.save('last.pt')
            # Full selected validation population; official holdout is never used here.
            with tqdm(total=len(self.val_samples), desc=f'Validation {epoch+1}/{c.training.max_epochs}',
                      mininterval=2, dynamic_ncols=True,
                      bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [ETA {remaining}]') as validation:
                rows = predict(self.model, self.val_samples, self.preprocessor,
                               progress=lambda i, n: validation.update(i - validation.n))
            metrics = save_predictions(self.output / 'validation' / f'epoch_{epoch+1:03d}', self.val_samples, rows)
            accuracy, ce = metrics['selection_accuracy'], metrics['overall']['qa_ce']
            if accuracy is None:
                raise ValueError('Validation selection requires closed questions')
            improved = accuracy > self.state['best_accuracy'] or (accuracy == self.state['best_accuracy'] and ce < self.state['best_val_ce'])
            if improved:
                self.state.update(best_accuracy=accuracy, best_val_ce=ce)
            eligible = not self.plan or self.state['global_step'] >= self.plan.cosine_start
            if eligible and c.training.early_stopping_enabled:
                reference = self.state['early_stop_reference']
                if reference is None or accuracy > reference + c.training.early_stopping_min_delta:
                    self.state.update(early_stop_reference=accuracy, bad_epochs=0)
                else:
                    self.state['bad_epochs'] += 1
                if self.state['bad_epochs'] >= c.training.early_stopping_patience:
                    self.state['stop_reason'] = 'early_stopping'
            # Advance only completed epochs; max_steps can end at a partial epoch.
            if self.state['next_window'] == self.steps_per_epoch:
                self.state.update(epoch=epoch+1, next_window=0)
            self.log('validation.jsonl', {'epoch': epoch+1, 'step': self.state['global_step'],
                                          'accuracy': accuracy, 'qa_ce': ce, 'best': improved})
            print(f'Epoch {epoch+1}/{c.training.max_epochs} | val CE={ce:.4f} | '
                  f'accuracy={accuracy:.4f}' + (' | best' if improved else ''), flush=True)
            if improved:
                self.save('best.pt')
            self.save('last.pt')
            if epoch+1 in c.training.checkpoint_epochs:
                self.save(f'epoch_{epoch+1:03d}.pt')
        write_json(self.output / 'result.json', self.state)
        return self.state
