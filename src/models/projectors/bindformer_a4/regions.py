"""Split each original A region into four children in parent-major order."""
from functools import lru_cache
import math
import torch

from src.models.projectors.bindformer.regions import spatial_boxes
from .config import CHILDREN_PER_REGION


@lru_cache(maxsize=128)
def nested_boxes(grid, num_regions, extent):
    parents = spatial_boxes(grid, num_regions, extent)
    step = tuple(e / n for e, n in zip(extent, grid))
    children = []
    for lo, hi in parents:
        shape = tuple(h - l for l, h in zip(lo, hi))
        if math.prod(shape) < CHILDREN_PER_REGION:
            raise ValueError('A4 requires at least four patches in every parent region')
        boxes = spatial_boxes(shape, CHILDREN_PER_REGION,
                              tuple(n * s for n, s in zip(shape, step)))
        children.extend((tuple(l + a for l, a in zip(lo, a0)),
                         tuple(l + b for l, b in zip(lo, b0))) for a0, b0 in boxes)
    return parents, tuple(children)


@lru_cache(maxsize=128)
def packed_children(grid, num_regions, extent):
    _, children = nested_boxes(grid, num_regions, extent)
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
