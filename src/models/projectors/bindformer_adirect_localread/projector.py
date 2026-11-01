"""One attention-only local read; retain ADirect's raw-mean bypass and global path."""
from contextlib import nullcontext
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.attention import sdpa_kernel, SDPBackend

from src.models.projectors.bindformer_adirect.projector import BindFormerADirect
from src.models.projectors.bindformer_adirect.regions import packed_local_regions, masked_mean


class LocalResidualRead(nn.Module):
    """mu + alpha * Wo Attention(Wq LN(mu), Wk LN(X), Wv LN(X)).

    Shared across cells, one query per cell. No FFN, added local position/sequence
    embeddings, or auxiliary loss. The geometric mask only excludes padding.
    """
    def __init__(self, dim, heads, dropout=0.0, scale_init=0.1):
        super().__init__()
        if dim % heads:
            raise ValueError('Local read width must be divisible by num_heads')
        self.heads, self.dropout = heads, dropout
        self.norm_queries = nn.LayerNorm(dim)
        self.norm_visual = nn.LayerNorm(dim)
        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_kv = nn.Linear(dim, 2 * dim, bias=False)
        self.to_out = nn.Linear(dim, dim, bias=False)
        self.alpha = nn.Parameter(torch.tensor(float(scale_init)))

    def forward(self, means, visual, valid):
        batch, dim = means.shape
        q = self.to_q(self.norm_queries(means[:, None]))
        k, v = self.to_kv(self.norm_visual(visual)).chunk(2, -1)

        def heads(x):
            return x.reshape(batch, -1, self.heads, dim // self.heads).transpose(1, 2)

        context = sdpa_kernel(SDPBackend.MATH) if torch.are_deterministic_algorithms_enabled() else nullcontext()
        with context:
            attended = F.scaled_dot_product_attention(
                heads(q), heads(k), heads(v), attn_mask=valid[:, None, None, :],
                dropout_p=self.dropout if self.training else 0.0)
        correction = self.to_out(attended.transpose(1, 2).reshape(batch, dim))
        return means + self.alpha.to(correction.dtype) * correction


class BindFormerADirectLocalRead(BindFormerADirect):
    def __init__(self, input_dim, output_dim, config):
        super().__init__(input_dim, output_dim, config)
        p = self.params
        self.local_reader = LocalResidualRead(p['inner_dim'], p['num_heads'], p['dropout'],
                                             p['local_residual_scale_init'])

    def _local_tokens(self, features, grid, extent):
        indices, valid = packed_local_regions(grid, self.allocation, extent)
        indices, valid = indices.to(features.device), valid.to(features.device)
        values = features[indices]
        return self.local_reader(masked_mean(values, valid), values, valid)

    def _local_details(self):
        return {**super()._local_details(), 'routing_mode': 'raw_spatial_mean_local_read',
                'local_read_depth': 1, 'local_read_heads': self.params['num_heads'],
                'local_read_support': 'same_spatial_cell', 'local_read_ffn': False,
                'local_residual_scale_init': self.params['local_residual_scale_init']}
