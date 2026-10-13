from dataclasses import dataclass
from torch import nn, Tensor


@dataclass
class ProjectedVolume:
    embeddings: Tensor
    input_tokens: int
    output_tokens: int
    metadata: dict

    def validate(self, output_dim, required_tokens=None):
        import torch
        if (self.embeddings.ndim != 2 or self.embeddings.shape != (self.output_tokens, output_dim)
                or self.input_tokens < 1 or self.output_tokens < 1
                or not torch.isfinite(self.embeddings).all()):
            raise ValueError('Invalid projected volume shape, lengths or values')
        if required_tokens is not None:
            if type(required_tokens) is not int or required_tokens<1:
                raise ValueError('Required visual token budget must be a positive integer')
            if self.output_tokens!=required_tokens:
                raise ValueError(f'Fixed visual token budget mismatch: required={required_tokens}, '
                                 f'actual={self.output_tokens}, input={self.input_tokens}')
        return self


class Projector(nn.Module):
    def forward(self, features, coords, metadata) -> ProjectedVolume:
        raise NotImplementedError
