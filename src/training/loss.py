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
