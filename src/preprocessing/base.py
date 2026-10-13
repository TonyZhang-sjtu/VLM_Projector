from dataclasses import dataclass
from typing import Protocol
import torch


@dataclass
class PatchVolume:
    patches: torch.Tensor
    coords: torch.Tensor
    metadata: dict

    def validate(self):
        n = self.patches.shape[0]
        if self.patches.shape != (n, 1024) or self.coords.shape != (n, 3) or n < 1:
            raise ValueError("Invalid NeuroVFM patch/coordinate shapes")
        if not torch.isfinite(self.patches).all():
            raise ValueError("Nonfinite patch data")
        grid = self.metadata['grid_dhw']
        expected = torch.cartesian_prod(*(torch.arange(s) for s in grid))
        if not torch.equal(self.coords.cpu(), expected):
            raise ValueError("Missing/reordered grid coordinates")
        if any(int(self.coords[:, i].max()) >= limit for i, limit in enumerate((128,192,192))):
            raise ValueError("Grid exceeds local NeuroVFM position encoding")
        return self


class Preprocessor(Protocol):
    def __call__(self, sample) -> PatchVolume: ...
