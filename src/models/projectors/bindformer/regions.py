"""Exact-count, disjoint spatial boxes; no report or lesion annotations are used."""
from dataclasses import dataclass
from functools import lru_cache
import math
import torch
from src.models.projectors.geometry import physical_extent


@lru_cache(maxsize=128)
def spatial_boxes(grid, count, extent):
    """Bisect the largest box along its longest physical axis until there are K.

    K need not factor into a regular grid (e.g. 206 slots). Ties are deterministic.
    The boxes cover the complete encoded grid, including encoder input padding.
    """
    if not 1 <= count <= math.prod(grid):
        raise ValueError('Spatial region count must be between 1 and the patch count')
    step = tuple(e/n for e, n in zip(extent, grid))
    boxes = [((0, 0, 0), tuple(grid))]
    for _ in range(count-1):
        boxes.sort(key=lambda b: (-math.prod(h-l for l, h in zip(*b)), b))
        lo, hi = boxes.pop(0)
        axis = max((i for i in range(3) if hi[i]-lo[i] > 1),
                   key=lambda i: ((hi[i]-lo[i])*step[i], -i))
        middle = (lo[axis]+hi[axis])//2
        left_hi, right_lo = list(hi), list(lo)
        left_hi[axis] = right_lo[axis] = middle
        boxes.extend([(lo, tuple(left_hi)), (tuple(right_lo), hi)])
    return tuple(sorted(boxes))


@lru_cache(maxsize=128)
def _packed_indices(grid, count, extent):
    groups = []
    for lo, hi in spatial_boxes(grid, count, extent):
        d, h, w = torch.meshgrid(*(torch.arange(l, r) for l, r in zip(lo, hi)), indexing='ij')
        groups.append(((d*grid[1]+h)*grid[2]+w).flatten())
    indices = torch.zeros((count, max(map(len, groups))), dtype=torch.long)
    valid = torch.zeros_like(indices, dtype=torch.bool)
    for i, group in enumerate(groups):
        indices[i, :len(group)] = group
        valid[i, :len(group)] = True
    return indices, valid


@dataclass
class SpatialRegions:
    indices: torch.Tensor  # [K,Pmax]; padding indices have no semantic meaning
    valid: torch.Tensor
    num_patches: int

    def dense(self, packed):
        """Convert region-local weights to [N,K], with exact zeros outside boxes."""
        columns = torch.arange(len(self.indices), device=self.indices.device)[:, None].expand_as(self.indices)
        return packed.new_zeros((self.num_patches, len(self.indices))).index_put(
            (self.indices[self.valid], columns[self.valid]), packed[self.valid])

    def membership(self, dtype):
        return self.dense(self.valid.to(dtype))


def make_regions(grid, metadata, count, device):
    grid = tuple(grid)
    indices, valid = _packed_indices(grid, count, physical_extent(grid, metadata))
    return SpatialRegions(indices.to(device), valid.to(device), math.prod(grid))
