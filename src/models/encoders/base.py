from dataclasses import dataclass
from typing import List
from torch import Tensor, nn


@dataclass
class EncoderOutput:
    features: List[Tensor]
    coords: List[Tensor]
    metadata: List[dict]


class EncoderAdapter(nn.Module):
    feature_dim: int

    def forward(self, volumes) -> EncoderOutput:
        raise NotImplementedError
