"""Relative, direction-aware grid PE; this is deliberately not millimeter PE."""
import math
import torch
from torch import nn
from src.data.sequence_mapping import SEQUENCE_NAMES

SEQUENCES = tuple(SEQUENCE_NAMES)


def normalized_grid_centers(coords, grid, metadata):
    order = metadata.get('array_zyx_to_dhw')
    if not isinstance(order, (list, tuple)) or sorted(order) != [0, 1, 2]:
        raise ValueError('grid_fourier requires array_zyx_to_dhw geometry metadata')
    direction = metadata.get('oriented', {}).get('direction')
    if direction is None:
        raise ValueError('grid_fourier requires oriented direction metadata')
    direction = torch.as_tensor(direction, device=coords.device, dtype=torch.float32)
    if direction.numel() != 9:
        raise ValueError('Expected 3x3 oriented direction')
    direction = direction.reshape(3,3)
    if (not torch.isfinite(direction).all()
            or not torch.allclose(direction.T@direction, torch.eye(3,device=coords.device), atol=1e-4,rtol=1e-4)):
        raise ValueError('Invalid image direction matrix')
    centers_dhw = 2*(coords.float()+0.5)/coords.new_tensor(grid,dtype=torch.float32)-1
    centers_zyx = centers_dhw[:, [order.index(i) for i in range(3)]]
    # Rotate normalized grid axes into LPS directions, without claiming mm scale.
    return centers_zyx.flip(-1) @ direction.T


class InputAdapter(nn.Module):
    def __init__(self, input_dim, dim, position_mode, frequencies, sequence_condition):
        super().__init__()
        self.norm = nn.LayerNorm(input_dim)
        self.projection = nn.Linear(input_dim,dim)
        self.position_mode = position_mode
        self.sequence_embedding = nn.Embedding(len(SEQUENCES),dim) if sequence_condition else None
        self.position_projection = None
        if position_mode == 'grid_fourier':
            self.register_buffer('frequencies', math.pi*2**torch.arange(frequencies,dtype=torch.float32))
            self.position_projection = nn.Linear(6*frequencies,dim,bias=False)

    def forward(self, features, coords, metadata, sequence):
        h = self.projection(self.norm(features))
        if self.position_projection is not None:
            # Geometry math is FP32 even under BF16 autocast.
            with torch.autocast(device_type=features.device.type, enabled=False):
                centers = normalized_grid_centers(coords.to(features.device), metadata['grid_dhw'], metadata)
                phases = centers[...,None]*self.frequencies.float()
                pe = torch.cat((phases.sin(),phases.cos()),dim=-1).flatten(1)
            h = h+self.position_projection(pe.to(h.dtype))
        if self.sequence_embedding is not None:
            if sequence not in SEQUENCES:
                raise ValueError(f'Missing or unknown MRI sequence: {sequence!r}')
            index = torch.tensor(SEQUENCES.index(sequence),device=h.device)
            h = h+self.sequence_embedding(index).to(h.dtype)
        return h
