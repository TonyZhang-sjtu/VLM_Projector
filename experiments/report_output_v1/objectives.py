"""First-order, memory-bounded objectives through a frozen decoder.

Compute exact VJPs at Z and discard each candidate decoder graph immediately.
The returned two scalar losses remain differentiable to the ORIGINAL Z graph;
the trainer supplies its usual distinct CE-token and auxiliary-sample scales.
No visual gradient is detached from the optimizer. Only second derivatives and
trainable decoder parameters are unsupported. A direct-autograd oracle is kept
for numerical regression tests.
"""
import torch
from torch.nn import functional as F
from .decoder_view import answer_masks, token_nll, field_losses, score, candidate_sample


def active_groups(row, fields):
    groups = []
    for group in row['groups']:
        spans = {f: s for f, s in group['spans'].items() if f in fields}
        candidates = [c for c in group['candidates'] if c['field'] in fields]
        if spans and candidates:
            groups.append((group, spans, candidates))
    return groups


def pair_weights(groups):
    """Report -> edited field -> all negatives for that field. No count scaling."""
    counts = {}
    for _, _, candidates in groups:
        for c in candidates:
            counts[c['field']] = counts.get(c['field'], 0) + 1
    return {f: 1.0 / (len(counts)*n) for f, n in counts.items()}


def direct_objective(decoder, sample, z, row, settings, ramp=1.0, pair_active=True):
    """Ordinary autograd oracle; production uses the bounded VJP equivalent."""
    fields = settings['fields']
    nll, _ = token_nll(decoder, sample, z)
    spans = {f: v['spans'] for f, v in row['facts'].items() if f in fields}
    losses = field_losses(nll, answer_masks(decoder, sample, spans))
    span = torch.stack(list(losses.values())).mean() if losses else nll.sum()*0
    group_loss = nll.sum()*0
    groups = active_groups(row, fields) if settings['group_weight'] and pair_active else []
    weights = pair_weights(groups)
    for _, group_spans, candidates in groups:
        positive = score(nll, answer_masks(decoder, sample, group_spans))
        for candidate in candidates:
            other = candidate_sample(sample, candidate)
            candidate_spans = {f: s for f, s in candidate['spans'].items() if f in fields}
            negative_nll, _ = token_nll(decoder, other, z)
            negative = score(negative_nll, answer_masks(decoder, other, candidate_spans))
            group_loss = group_loss + weights[candidate['field']] * F.softplus(settings['margin']-positive+negative)
    return nll.sum(), ramp*(settings['span_weight']*span + settings['group_weight']*group_loss)


class _ReportVJP(torch.autograd.Function):
    @staticmethod
    def forward(ctx, z, decoder, sample, row, settings, ramp, pair_active, metrics):
        if any(p.requires_grad for p in decoder.parameters()):
            raise ValueError('Bounded report objectives require a frozen decoder')
        fields = settings['fields']
        groups = active_groups(row, fields) if settings['group_weight'] and pair_active else []
        weights = pair_weights(groups)
        # Explicit enable_grad is required inside an autograd.Function forward.
        with torch.enable_grad():
            leaf = z.detach().requires_grad_(True)
            nll, length = token_nll(decoder, sample, leaf)
            report = nll.sum()
            spans = {f: v['spans'] for f, v in row['facts'].items() if f in fields}
            losses = field_losses(nll, answer_masks(decoder, sample, spans))
            has_span = bool(losses) and settings['span_weight'] > 0
            report_grad = torch.autograd.grad(report, leaf, retain_graph=has_span or bool(groups))[0]
            aux_grad = torch.zeros_like(leaf)
            span = torch.stack(list(losses.values())).mean() if losses else report.detach()*0
            if has_span:
                aux_grad = aux_grad + ramp*settings['span_weight']*torch.autograd.grad(
                    span, leaf, retain_graph=bool(groups))[0]
            positive_cache = []
            for index, (group, group_spans, candidates) in enumerate(groups):
                positive = score(nll, answer_masks(decoder, sample, group_spans))
                positive_grad = torch.autograd.grad(positive, leaf, retain_graph=index+1 < len(groups))[0]
                positive_cache.append((group, candidates, positive.detach(), positive_grad))
            report_value = report.detach()
            span_value = span.detach()
            metrics.update(answer_tokens=len(nll), length=length, span_loss=float(span_value),
                           span_fields=len(losses), labelled_report=int(bool(losses)), pair_count=0,
                           pair_correct=0, group_loss=0., score_gap_sum=0.)
            for f, value in losses.items():
                metrics[f'span_loss_{f}'] = float(value.detach())
                metrics[f'span_count_{f}'] = 1
            del nll, report, span, losses
            group_value = report_value.new_zeros(())
            for _, candidates, positive, positive_grad in positive_cache:
                for candidate in candidates:
                    other = candidate_sample(sample, candidate)
                    masks = answer_masks(decoder, other, {f: s for f, s in candidate['spans'].items() if f in fields})
                    negative_nll, _ = token_nll(decoder, other, leaf)
                    negative = score(negative_nll, masks)
                    negative_grad = torch.autograd.grad(negative, leaf)[0]
                    gap = positive - negative.detach()
                    argument = settings['margin'] - gap
                    weight = weights[candidate['field']]
                    value = F.softplus(argument)
                    group_value = group_value + weight*value
                    aux_grad = aux_grad + ramp*settings['group_weight']*weight*torch.sigmoid(argument)*(negative_grad-positive_grad)
                    metrics['pair_count'] += 1
                    metrics['pair_correct'] += int(gap > 0)
                    metrics['score_gap_sum'] += float(gap)
                    f = candidate['field']
                    metrics[f'pair_count_{f}'] = metrics.get(f'pair_count_{f}', 0) + 1
                    metrics[f'pair_correct_{f}'] = metrics.get(f'pair_correct_{f}', 0) + int(gap > 0)
                    del negative_nll, negative, negative_grad
        auxiliary = ramp*(settings['span_weight']*span_value + settings['group_weight']*group_value)
        if not torch.isfinite(auxiliary) or not torch.isfinite(report_grad).all() or not torch.isfinite(aux_grad).all():
            raise FloatingPointError('Nonfinite report objective/VJP')
        ctx.save_for_backward(report_grad, aux_grad)
        report_norm = float(report_grad.float().norm())
        aux_norm = float(aux_grad.float().norm())
        metrics.update(group_loss=float(group_value), weighted_loss=float(auxiliary),
                       report_z_grad_norm=report_norm, auxiliary_z_grad_norm=aux_norm,
                       report_mean_z_grad_norm=report_norm/metrics['answer_tokens'],
                       auxiliary_to_report_mean_grad_ratio=aux_norm/max(report_norm/metrics['answer_tokens'], 1e-20),
                       auxiliary_report_grad_cosine=float((report_grad.float()*aux_grad.float()).sum())/max(report_norm*aux_norm, 1e-20))
        return report_value, auxiliary

    @staticmethod
    def backward(ctx, report_scale, auxiliary_scale):
        report_grad, aux_grad = ctx.saved_tensors
        grad = report_scale*report_grad + auxiliary_scale*aux_grad
        return grad, None, None, None, None, None, None, None


def streamed_objective(decoder, sample, z, row, settings, ramp=1.0, pair_active=True):
    metrics = {}
    report, auxiliary = _ReportVJP.apply(z, decoder, sample, row, settings, ramp, pair_active, metrics)
    return report, auxiliary, metrics
