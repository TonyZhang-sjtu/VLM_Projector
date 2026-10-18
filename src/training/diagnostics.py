"""Read-only measurements of actual optimizer updates, including AdamW decay."""
import json
import math
from pathlib import Path

import torch


def _norm(tensors):
    # Accumulate squared per-tensor FP32 norms in FP64, without making a large
    # flattened vector. No random operations or modifications to gradients.
    values = [torch.linalg.vector_norm(t.detach().float()).double().square()
              for t in tensors if t is not None]
    return float(torch.stack(values).sum().sqrt()) if values else 0.0


class OptimizerDiagnostics:
    """One snapshot per update, only for optimizer parameters (never the LLM)."""

    @torch.no_grad()
    def __init__(self, optimizer):
        self.groups = {}
        for group in optimizer.param_groups:
            self.groups.setdefault(group['component'], []).extend(group['params'])
        self.before = {name: [p.detach().clone() for p in params]
                       for name, params in self.groups.items()}
        self.records = {name: {
            'parameter_count': sum(p.numel() for p in params),
            'parameters_with_grad': sum(p.numel() for p in params if p.grad is not None),
            'parameter_norm_before': _norm(self.before[name]),
            'grad_norm_pre_clip': _norm([p.grad for p in params]),
        } for name, params in self.groups.items()}

    @torch.no_grad()
    def after_clip(self):
        for name, params in self.groups.items():
            self.records[name]['grad_norm_post_clip'] = _norm([p.grad for p in params])

    @torch.no_grad()
    def after_step(self):
        for name, params in self.groups.items():
            # Differences are measured from stored parameters, not inferred
            # from gradients, LR, or Adam moments. Includes rounding and decay.
            update = _norm([p.detach().float() - old.float()
                            for p, old in zip(params, self.before[name])])
            norm = self.records[name]['parameter_norm_before']
            self.records[name].update(update_norm=update,
                                     relative_update_norm=update / norm if norm else None)
            if not math.isfinite(update):
                raise FloatingPointError(f'Nonfinite {name} parameter update')
        self.before.clear()
        return self.records


def epoch_training_summary(log_path, epoch):
    """Token-weighted online train CE; epoch is the legacy zero-based index."""
    records = [json.loads(line) for line in Path(log_path).read_text().splitlines() if line.strip()]
    records = [row for row in records if row['epoch'] == epoch]
    if not records or len({row['step'] for row in records}) != len(records):
        raise ValueError('Missing or duplicate updates in epoch training log')
    tokens = sum(row['answer_tokens'] for row in records)
    loss = sum(row.get('answer_loss_sum', row['answer_ce'] * row['answer_tokens']) for row in records)
    return {'epoch': epoch + 1, 'first_step': records[0]['step'], 'last_step': records[-1]['step'],
            'updates': len(records), 'answer_tokens': tokens, 'answer_loss_sum': loss,
            'train_answer_ce': loss / tokens,
            'aggregation': 'sum_answer_loss / sum_answer_tokens; online training mode'}
