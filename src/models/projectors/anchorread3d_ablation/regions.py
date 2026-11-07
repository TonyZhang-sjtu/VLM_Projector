"""Two-stage spatial cells: four children per parent except the final remainder.

The final parent is partitioned into its requested number of children directly.
No child/token is dropped, and no synthetic token is sent to the LLM.
"""
from functools import lru_cache
import math
import torch
from src.models.projectors.bindformer.regions import spatial_boxes
from .budget import child_counts


@lru_cache(maxsize=128)
def nested_tail_boxes(grid, num_tokens, extent):
    counts = child_counts(num_tokens)
    parents = spatial_boxes(grid, len(counts), extent)
    step = tuple(e / n for e, n in zip(extent, grid))
    children = []
    for (lo, hi), count in zip(parents, counts):
        shape = tuple(h - l for l, h in zip(lo, hi))
        if math.prod(shape) < count:
            raise ValueError('A4 ablation parent has fewer patches than requested children')
        boxes = spatial_boxes(shape, count, tuple(n * s for n, s in zip(shape, step)))
        children.extend((tuple(l + a for l, a in zip(lo, a0)),
                         tuple(l + b for l, b in zip(lo, b0))) for a0, b0 in boxes)
    return parents, tuple(children)


@lru_cache(maxsize=128)
def packed_tail_regions(grid, num_tokens, extent):
    _, children = nested_tail_boxes(grid, num_tokens, extent)
    groups = []
    for lo, hi in children:
        d, h, w = torch.meshgrid(*(torch.arange(l, r) for l, r in zip(lo, hi)), indexing='ij')
        groups.append(((d * grid[1] + h) * grid[2] + w).flatten())
    indices = torch.zeros((len(groups), max(map(len, groups))), dtype=torch.long)
    valid = torch.zeros_like(indices, dtype=torch.bool)
    for i, group in enumerate(groups):
        indices[i, :len(group)] = group
        valid[i, :len(group)] = True
    return indices, valid
