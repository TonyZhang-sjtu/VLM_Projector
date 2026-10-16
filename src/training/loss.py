"""Normalize once across an entire accumulation window, including DDP averaging."""
import torch
import torch.distributed as dist


def global_answer_tokens(local_count,device):
    total=torch.tensor(local_count,dtype=torch.long,device=device)
    if dist.is_initialized():dist.all_reduce(total)
    if total.item()<1:raise ValueError('Empty answer-token window')
    return int(total.item())


def backward_scale(global_count,world_size=1):
    # DDP averages parameter gradients; compensate before that average.
    return world_size/global_count


def global_sample_count(local_count,device):
    total=torch.tensor(local_count,dtype=torch.long,device=device)
    if dist.is_initialized():dist.all_reduce(total)
    if total.item()<1:raise ValueError('Empty sample window')
    return int(total.item())


def combined_loss(result, answer_count, world_size=1, feature_weight=0., sample_count=None):
    loss=result['loss_sum']*backward_scale(answer_count,world_size)
    if feature_weight:
        if sample_count is None or sample_count<1:
            raise ValueError('Feature loss requires an accumulation-window sample denominator')
        feature=result['feature_loss_sum']
        if not torch.isfinite(feature):raise FloatingPointError('Nonfinite reconstruction loss')
        loss=loss+feature_weight*feature*backward_scale(sample_count,world_size)
    return loss
