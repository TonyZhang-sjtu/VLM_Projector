"""Tensor-only extension channels; never serialize live tensors as metadata."""
from dataclasses import dataclass
from typing import Protocol
from torch import Tensor
from .interactions import BundleInteractions
from .geometry import PatchGeometry


@dataclass
class BindFormerAuxiliary:
    source_features: Tensor
    coords: Tensor
    slots: Tensor
    assignments: Tensor
    role_tokens: Tensor
    global_tokens: Tensor
    concept_weights: Tensor | None = None
    physical_centers: Tensor | None = None
    patch_volumes: Tensor | None = None
    geometry_centers: Tensor | None = None
    geometry_space: str | None = None


class GeometryProvider(Protocol):
    def __call__(self, coords: Tensor, metadata: dict, space: str = 'grid',
                 coordinate_scale: float = 100., volume_scale: float = 1000.) -> PatchGeometry: ...


class GeometryEncoder(Protocol):
    def __call__(self, assignments: Tensor, geometry: PatchGeometry) -> Tensor: ...


class AuxiliaryObjective(Protocol):
    def __call__(self, auxiliary: BindFormerAuxiliary, embeddings: Tensor, targets: dict) -> dict: ...
