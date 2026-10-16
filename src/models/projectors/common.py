"""Shared mapping heads and differentiable query blocks for controlled baselines."""
from contextlib import nullcontext
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.attention import sdpa_kernel, SDPBackend
from omegaconf import OmegaConf
from src.models.projectors.base import Projector, ProjectedVolume
from src.models.projectors.geometry import validate_inputs


def mapping_head(input_dim, output_dim, config):
    if config.params.head_type == 'linear':
        return nn.Linear(input_dim, output_dim, bias=config.bias)
    return nn.Sequential(nn.Linear(input_dim, config.hidden_size, bias=config.bias), nn.GELU(),
                         nn.Linear(config.hidden_size, output_dim, bias=config.bias))


class StructuredProjector(Projector):
    def __init__(self, input_dim, output_dim, config):
        super().__init__()
        self.input_dim, self.output_dim, self.name = input_dim, output_dim, config.name
        self.params = OmegaConf.to_container(config.params, resolve=True)
        from src.models.projectors.budget import require_resolved
        self.budget = require_resolved(config)
        self.stage, self.seed, self.project_root = 'a', 42, '.'

    def inputs(self, features, coords, metadata):
        from src.models.projectors.budget import check_input
        check_input(self.budget, features, metadata)
        if features.ndim != 2 or features.shape[1] != self.input_dim:
            raise ValueError('Projector input channel dimension mismatch')
        return validate_inputs(features, coords, metadata)

    def finish(self, output, features, metadata, auxiliary=None, **details):
        from src.models.projectors.budget import output_info
        info = dict(metadata)
        info['projector'] = {'name': self.name, 'requested_tokens': self.params.get('num_tokens'),
                             'actual_tokens': len(output),
                             'budget_policy': self.params.get('budget_policy', 'natural' if self.name == 'concat3d' else 'exact'),
                             **details, **output_info(self.budget, len(features), len(output))}
        return ProjectedVolume(output, len(features), len(output), info, auxiliary).validate(self.output_dim)


class QueryBlock(nn.Module):
    """Pre-norm cross-attention and FFN. Optional latent K/V enables query interaction."""
    def __init__(self, dim, heads, expansion=4, dropout=0., latent_kv=False):
        super().__init__()
        if dim % heads:
            raise ValueError(f'Query width {dim} must be divisible by heads {heads}')
        self.heads, self.dropout, self.latent_kv = heads, dropout, latent_kv
        self.norm_queries = nn.LayerNorm(dim)
        self.norm_visual = nn.LayerNorm(dim)
        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_kv = nn.Linear(dim, 2*dim, bias=False)
        self.to_out = nn.Linear(dim, dim, bias=False)
        self.ffn = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim*expansion, bias=False),
                                 nn.GELU(), nn.Linear(dim*expansion, dim, bias=False))

    def forward(self, queries, visual, valid=None):
        batch, count, dim = queries.shape
        q_norm, memory = self.norm_queries(queries), self.norm_visual(visual)
        if self.latent_kv:
            memory = torch.cat((memory, q_norm), dim=1)
            if valid is not None:
                valid = torch.cat((valid, torch.ones((batch,count),dtype=torch.bool,device=valid.device)),1)
        q = self.to_q(q_norm)
        k, v = self.to_kv(memory).chunk(2, -1)
        def heads(x):
            return x.reshape(batch, -1, self.heads, dim//self.heads).transpose(1,2)
        mask = None if valid is None else valid[:,None,None,:]
        context = sdpa_kernel(SDPBackend.MATH) if torch.are_deterministic_algorithms_enabled() else nullcontext()
        with context:
            attended = F.scaled_dot_product_attention(heads(q), heads(k), heads(v), attn_mask=mask,
                           dropout_p=self.dropout if self.training else 0.)
        attended = attended.transpose(1,2).reshape(batch,count,dim)
        queries = queries + self.to_out(attended)
        return queries + self.ffn(queries)
