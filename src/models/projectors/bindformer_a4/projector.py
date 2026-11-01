"""Freshly initialized A4; never constructs or loads a trained Baseline A.

Retained: A's input adapter, global cross-attention, local V/out/FFN and head.
Replaced: slot/role/UMLS/local-QK subsystem by one mean per spatial child.
The four children are spatial boxes, not semantic roles or repeated outputs.
"""
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from src.models.projectors.common import StructuredProjector, QueryBlock, mapping_head
from src.models.projectors.geometry import check_budget, physical_extent
from src.models.projectors.bindformer.input_adapter import InputAdapter
from .config import allocate_a4_tokens
from .regions import packed_children


class BindFormerA4(StructuredProjector):
    def __init__(self, input_dim, output_dim, config):
        super().__init__(input_dim, output_dim, config)
        p = self.params
        self.allocation = allocate_a4_tokens(p['num_tokens'], p['global_fraction'])
        dim, g = p['inner_dim'], self.allocation['global_tokens']
        self.requires_sequence = p['sequence_condition']
        self.input_adapter = InputAdapter(input_dim, dim, p['position_mode'],
                                         p['position_frequencies'], p['sequence_condition'])
        self.global_queries = nn.Parameter(torch.randn(g, dim) * dim**-.5) if g else None
        self.global_layers = nn.ModuleList(
            QueryBlock(dim, p['num_heads'], p['ffn_expansion'], p['dropout'])
            for _ in range(p['global_depth']) if g)
        self.value_norm = nn.LayerNorm(dim)
        self.value = nn.Linear(dim, dim, bias=False)
        self.output = nn.Linear(dim, dim, bias=False)
        # Match A's SupportReader FFN, including its biases.
        self.ffn = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim * p['ffn_expansion']),
                                 nn.GELU(), nn.Linear(dim * p['ffn_expansion'], dim))
        self.final_norm = nn.LayerNorm(dim)
        self.head = mapping_head(dim, output_dim, config)

    def _local_tokens(self, h, grid, extent):
        indices, valid = packed_children(grid, self.allocation['num_regions'], extent)
        indices, valid = indices.to(h.device), valid.to(h.device)
        values = self.value(self.value_norm(h))
        # FP32 accumulation under BF16; retain FP64 for numerical checks.
        dtype = torch.float64 if values.dtype == torch.float64 else torch.float32
        local = (values[indices].to(dtype) * valid[..., None]).sum(1) / valid.sum(1)[:, None]
        local = self.output(local.to(values.dtype))
        return local + self.ffn(local)

    def forward(self, features, coords, metadata):
        self.inputs(features, coords, metadata)
        check_budget(len(features), self.params['num_tokens'])
        sequence = metadata.get('sequence')
        if self.requires_sequence and sequence is None:
            raise ValueError('A4 requires MRI sequence context')
        h = self.input_adapter(features, coords, metadata, sequence)
        grid = tuple(metadata['grid_dhw'])
        extent = physical_extent(grid, metadata)
        use_checkpoint = self.params['checkpoint_blocks'] and self.training and torch.is_grad_enabled()
        if use_checkpoint:
            local = checkpoint(self._local_tokens, h, grid, extent, use_reentrant=False)
        else:
            local = self._local_tokens(h, grid, extent)
        if self.global_queries is None:
            glob = h.new_empty((0, h.shape[-1]))
        else:
            glob, memory = self.global_queries[None], h[None]
            for layer in self.global_layers:
                glob = (checkpoint(layer, glob, memory, use_reentrant=False) if use_checkpoint
                        else layer(glob, memory))
            glob = glob[0]
        output = self.head(self.final_norm(torch.cat((glob, local), dim=0)))
        return self.finish(output, features, metadata, **self.allocation,
                           routing_mode='nested_spatial_mean',
                           local_token_order='parent_major_child_major',
                           position_mode=self.params['position_mode'],
                           global_depth=len(self.global_layers))
