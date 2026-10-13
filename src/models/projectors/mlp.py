import torch
from torch import nn
from src.models.projectors.base import Projector, ProjectedVolume


class MLPProjector(Projector):
    def __init__(self, input_dim, output_dim, config):
        super().__init__()
        from src.models.projectors.budget import require_resolved
        self.budget = require_resolved(config)
        self.layers = nn.Sequential(nn.Linear(input_dim,config.hidden_size,bias=config.bias),
            nn.GELU(),nn.Linear(config.hidden_size,output_dim,bias=config.bias))

    def forward(self, features, coords, metadata):
        from src.models.projectors.budget import check_input, output_info
        check_input(self.budget, features, metadata)
        if len(features) != len(coords):
            raise ValueError('Feature/coordinate count mismatch')
        output = self.layers(features)
        if len(output) != len(features) or not torch.isfinite(output).all():
            raise ValueError('MLP must preserve every input token')
        if self.budget:
            metadata = {**metadata, 'projector': {'name': 'mlp2x_gelu',
                'actual_tokens': len(output), **output_info(self.budget, len(features), len(output))}}
        return ProjectedVolume(output,len(features),len(output),metadata)
