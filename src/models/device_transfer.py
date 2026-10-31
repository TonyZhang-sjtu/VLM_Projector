"""Differentiable transfers for GPU layouts with unreliable peer-to-peer copies."""
import torch


def move_tensor(tensor, device, dtype=None):
    """Stage cross-GPU values through host memory, retaining autograd on both copies."""
    destination = torch.device(device)
    if tensor.device.type == 'cuda' and destination.type == 'cuda' and tensor.device != destination:
        tensor = tensor.to('cpu')
    return tensor.to(device=destination, dtype=dtype or tensor.dtype)
