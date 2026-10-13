"""Dense three-dimensional patch geometry. No labels or foreground filtering."""
from functools import lru_cache
import math
import torch


def validate_inputs(features, coords, metadata):
    grid = tuple(metadata.get('grid_dhw', ()))
    if len(grid) != 3 or any(type(v) is not int or v < 1 for v in grid):
        raise ValueError('Projector requires positive grid_dhw metadata')
    if features.ndim != 2 or len(features) != math.prod(grid) or features.shape[1] < 1:
        raise ValueError('Feature tensor does not match dense grid_dhw')
    expected = torch.cartesian_prod(*(torch.arange(n) for n in grid))
    if coords.shape != (len(features), 3) or not torch.equal(coords.detach().cpu(), expected):
        raise ValueError('Projector requires canonical complete D/H/W coordinates')
    if not torch.isfinite(features).all():
        raise ValueError('Nonfinite projector input')
    return grid


def check_budget(n, target):
    if type(target) is not int or not 1 <= target <= n:
        raise ValueError(f'Compression budget must satisfy 1 <= M <= N, got M={target}, N={n}')


def physical_extent(grid, metadata):
    """D/H/W physical extent after padding; generic unit-patch fallback for tests."""
    if 'shape_dhw' not in metadata or 'bounded_before_pad' not in metadata:
        return tuple(float(v) for v in grid)
    spacing_zyx = metadata['bounded_before_pad']['spacing_xyz'][::-1]
    order = metadata['array_zyx_to_dhw']
    extent = tuple(float(metadata['shape_dhw'][i]) * spacing_zyx[order[i]] for i in range(3))
    if any(not math.isfinite(v) or v <= 0 for v in extent):
        raise ValueError('Invalid physical extent')
    return extent


@lru_cache(maxsize=512)
def choose_grid(grid, target, extent, policy='nearest'):
    """Minimize |actual-target|, then prefer <=target, balanced cells, lexicographic order."""
    check_budget(math.prod(grid), target)
    if policy not in ('nearest', 'at_most', 'exact'):
        raise ValueError('Unknown grid budget policy')
    best = None
    for d in range(1, grid[0] + 1):
        for h in range(1, grid[1] + 1):
            quotient = target // (d*h)
            for w in {max(1, min(grid[2], quotient)), max(1, min(grid[2], quotient+1))}:
                actual = d*h*w
                if (policy == 'at_most' and actual > target) or (policy == 'exact' and actual != target):
                    continue
                dims = (d, h, w)
                logs = [math.log(e/m) for e, m in zip(extent, dims)]
                mean = sum(logs)/3
                imbalance = sum((v-mean)**2 for v in logs)
                score = (abs(actual-target), actual > target, imbalance, dims)
                if best is None or score < best[0]:
                    best = (score, dims)
    if best is None:
        raise ValueError(f'No exact output grid for {grid}, M={target}; use nearest or at_most')
    return best[1]


@lru_cache(maxsize=256)
def grid_groups(grid, output_grid):
    """Disjoint integer-boundary cells in canonical order; each token belongs once."""
    if any(not 1 <= m <= n for n, m in zip(grid, output_grid)):
        raise ValueError('Output grid cannot exceed input axes')
    axis = [[(i*n//m, (i+1)*n//m) for i in range(m)] for n, m in zip(grid, output_grid)]
    groups = []
    for d0, d1 in axis[0]:
        for h0, h1 in axis[1]:
            for w0, w1 in axis[2]:
                d, h, w = torch.meshgrid(torch.arange(d0, d1), torch.arange(h0, h1),
                                        torch.arange(w0, w1), indexing='ij')
                groups.append(((d*grid[1]+h)*grid[2]+w).reshape(-1))
    return tuple(groups)


def partition(grid, metadata, target, policy):
    output = choose_grid(tuple(grid), target, physical_extent(grid, metadata), policy)
    return output, grid_groups(tuple(grid), output)


def pack_groups(features, groups):
    """Gather local memories. Padding is an empty slot, never a foreground decision."""
    sizes = [len(g) for g in groups]
    if not sizes or min(sizes) < 1:
        raise ValueError('Empty groups are not allowed')
    indices = torch.zeros((len(groups), max(sizes)), dtype=torch.long)
    valid = torch.zeros_like(indices, dtype=torch.bool)
    for i, group in enumerate(groups):
        indices[i, :len(group)] = group
        valid[i, :len(group)] = True
    valid = valid.to(features.device)
    values = features[indices.to(features.device)].masked_fill(~valid[..., None], 0)
    return values, valid


def grid_mean(features, groups):
    values, valid = pack_groups(features, groups)
    # Accumulate BF16 inputs in FP32; cast back to keep the interface dtype.
    return (values.float().sum(1)/valid.sum(1)[:, None]).to(features.dtype)


def region_mean(features, groups):
    """No M x largest-region allocation, and no atomic scatter of floating values."""
    return torch.stack([features.index_select(0, g.to(features.device)).float().mean(0)
                        for g in groups]).to(features.dtype)


def group_metadata(coords, groups):
    centers = torch.stack([coords[g.to(coords.device)].float().mean(0) for g in groups])
    return {'group_sizes': [len(g) for g in groups], 'output_coords_dhw': centers.cpu().tolist()}
