"""Shared raw-feature pooling for ADirect and its controlled variants."""
import torch
from src.models.projectors.bindformer.regions import _packed_indices
from src.models.projectors.bindformer_a4.regions import packed_children


def packed_local_regions(grid, allocation, extent):
    if allocation['children_per_region'] == 1:
        return _packed_indices(tuple(grid), allocation['num_regions'], tuple(extent))
    return packed_children(tuple(grid), allocation['num_regions'], tuple(extent))


def masked_mean(values, valid):
    # FP32 accumulation under BF16/FP16; preserve double for numerical oracles.
    dtype = torch.float64 if values.dtype == torch.float64 else torch.float32
    total = values.to(dtype).masked_fill(~valid[..., None], 0).sum(1)
    return (total / valid.sum(1)[:, None]).to(values.dtype)
