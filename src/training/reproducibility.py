"""Explicit CUDA reproducibility policy; no change to source weight files."""
import os
import torch


def configure_reproducibility(config):
    enabled=config.training.deterministic
    if enabled:
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
        if os.environ['CUBLAS_WORKSPACE_CONFIG'] not in (':4096:8',':16:8'):
            raise ValueError('Deterministic mode requires CUBLAS_WORKSPACE_CONFIG=:4096:8 or :16:8 before CUDA initialization')
    torch.use_deterministic_algorithms(enabled)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=enabled
    torch.backends.cuda.matmul.allow_tf32=False

