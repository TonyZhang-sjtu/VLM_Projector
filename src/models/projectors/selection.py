"""Selection-only diagnostics: sample-seeded random or greedy feature diversity."""
import hashlib
import torch
from torch.nn import functional as F
from src.models.projectors.common import StructuredProjector, mapping_head
from src.models.projectors.geometry import check_budget


def diverse_indices(features, target):
    """Greedy max-min squared distance on unit features, first point index zero.

    Selected points are masked, including for identical/zero features. Tie breaks
    use the smallest index. Auxiliary memory O(NC + N), not an N x N matrix.
    """
    x = F.normalize(features.detach().float(), dim=-1)
    distances = torch.full((len(x),), float('inf'), device=x.device)
    selected = torch.zeros(len(x), dtype=torch.bool, device=x.device)
    indices = []
    current = 0
    for _ in range(target):
        indices.append(current); selected[current] = True
        distances = torch.minimum(distances, (x-x[current]).square().sum(-1))
        distances.masked_fill_(selected, -1.)
        current = int(distances.argmax())
    return torch.tensor(sorted(indices), device=features.device)


class TokenSelect3D(StructuredProjector):
    def __init__(self, input_dim, output_dim, config):
        super().__init__(input_dim, output_dim, config)
        self.head = mapping_head(input_dim, output_dim, config)

    def forward(self, features, coords, metadata):
        self.inputs(features, coords, metadata)
        target = self.params['num_tokens']; check_budget(len(features), target)
        if self.params['strategy'] == 'diversity':
            indices = diverse_indices(features, target)
        else:
            identity = metadata.get('cache_key', metadata.get('sample_id', metadata.get('id')))
            if identity is None:
                raise ValueError('Random selection requires cache_key, sample_id, or id metadata')
            seed = self.params['selection_seed']
            if seed is None: seed = self.seed
            seed = int.from_bytes(hashlib.sha256(f'{seed}:{identity}'.encode()).digest()[:8], 'little') % (2**63)
            generator = torch.Generator().manual_seed(seed)
            indices = torch.randperm(len(features), generator=generator)[:target].sort().values.to(features.device)
        output = self.head(features.index_select(0, indices))
        return self.finish(output, features, metadata, strategy=self.params['strategy'],
                           selected_indices=indices.cpu().tolist(),
                           output_coords_dhw=coords[indices.to(coords.device)].cpu().tolist())
