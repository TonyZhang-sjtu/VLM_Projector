"""Spatially connected Ward agglomeration; discrete routing, differentiable means."""
import hashlib
import heapq
from pathlib import Path
import numpy as np
import torch
from src.models.projectors.common import StructuredProjector, mapping_head
from src.models.projectors.geometry import check_budget, region_mean, group_metadata
from src.preprocessing.cache import TensorCache


ALGORITHM_VERSION = 'connected_ward_v1'


def connected_ward(features, grid, target, normalize=True):
    """Six-neighbour graph, minimum Ward cost, stable (cost,node1,node2) ties.

    CPU routing only: never return detached embeddings. Region values are computed
    separately from the original differentiable features. No N x N matrix.
    """
    x = features.detach().float().cpu().numpy().astype(np.float64)
    n, dim = x.shape
    check_budget(n, target)
    if normalize:
        x = x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
    means = np.zeros((2*n-1, dim), dtype=np.float64)
    means[:n] = x
    sizes = np.zeros(2*n-1, dtype=np.int64); sizes[:n] = 1
    neighbours = [set() for _ in range(2*n-1)]
    children = {}
    active = set(range(n))
    heap = []

    def edge(a, b):
        a, b = sorted((a, b))
        delta = means[a]-means[b]
        cost = float((sizes[a]*sizes[b]/(sizes[a]+sizes[b])) * np.dot(delta, delta))
        heapq.heappush(heap, (cost, a, b))

    dsize, hsize, wsize = grid
    for d in range(dsize):
        for h in range(hsize):
            for w in range(wsize):
                a = (d*hsize+h)*wsize+w
                adjacent = []
                if d+1 < dsize: adjacent.append(a+hsize*wsize)
                if h+1 < hsize: adjacent.append(a+wsize)
                if w+1 < wsize: adjacent.append(a+1)
                for b in adjacent:
                    neighbours[a].add(b); neighbours[b].add(a); edge(a, b)
    for node in range(n, 2*n-target):
        while True:
            _, a, b = heapq.heappop(heap)
            if a in active and b in active: break
        sizes[node] = sizes[a]+sizes[b]
        means[node] = (sizes[a]*means[a]+sizes[b]*means[b])/sizes[node]
        children[node] = (a, b)
        adjacent = (neighbours[a] | neighbours[b]) - {a, b}
        active.remove(a); active.remove(b); active.add(node)
        neighbours[node] = adjacent
        for other in sorted(adjacent):
            neighbours[other].difference_update((a, b)); neighbours[other].add(node)
            edge(node, other)
        neighbours[a].clear(); neighbours[b].clear()
    groups = []
    for node in sorted(active):
        leaves, pending = [], [node]
        while pending:
            current = pending.pop()
            if current < n: leaves.append(current)
            else: pending.extend(children[current])
        groups.append(torch.tensor(sorted(leaves), dtype=torch.long))
    return tuple(sorted(groups, key=lambda g: int(g[0])))


class AdaptiveMerge3D(StructuredProjector):
    def __init__(self, input_dim, output_dim, config):
        super().__init__(input_dim, output_dim, config)
        self.head = mapping_head(input_dim, output_dim, config)

    def forward(self, features, coords, metadata):
        grid = self.inputs(features, coords, metadata)
        target = self.params['num_tokens']; check_budget(len(features), target)
        # A-only cache, keyed by exact feature bytes and algorithm/settings. This
        # also separates encoder revisions, precision, preprocessing and samples.
        use_cache = self.stage == 'a' and not features.requires_grad and self.params['cache_assignments']
        hit = False
        if use_cache:
            h = hashlib.sha256()
            h.update(repr((ALGORITHM_VERSION, grid, target, self.params['normalize_scores'],
                           str(features.dtype), tuple(features.shape))).encode())
            h.update(features.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
            key = h.hexdigest()
            root = Path(self.params['assignment_cache_dir'])
            cache = TensorCache(root if root.is_absolute() else Path(self.project_root)/root)
            groups = cache.get(key); hit = groups is not None
        else: groups = None
        if groups is None:
            groups = connected_ward(features, grid, target, self.params['normalize_scores'])
            if use_cache: cache.put(key, groups)
        if len(groups) != target or not torch.equal(torch.cat(groups).sort().values, torch.arange(len(features))):
            raise ValueError('Invalid merge partition/cache: every input must occur exactly once')
        output = self.head(region_mean(features, groups))
        return self.finish(output, features, metadata, algorithm=ALGORITHM_VERSION,
                           assignment_cache_enabled=use_cache, assignment_cache_hit=hit,
                           **group_metadata(coords, groups))
