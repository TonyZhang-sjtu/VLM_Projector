"""Fresh ADirect projector: raw child means -> shared final LN/head.

The global path and nested children match A4. The local path bypasses the
InputAdapter (including its sequence/position additions), V/out projections
and local FFN. It has no learned channel adapter: input_dim must equal
inner_dim so the direct evidence path cannot silently become a projection.
"""
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from src.models.projectors.common import StructuredProjector, QueryBlock, mapping_head
from src.models.projectors.geometry import check_budget, physical_extent
from src.models.projectors.bindformer.input_adapter import InputAdapter
from .config import allocate_adirect_tokens
from .regions import packed_local_regions, masked_mean


class BindFormerADirect(StructuredProjector):
    def __init__(self, input_dim, output_dim, config):
        super().__init__(input_dim, output_dim, config)
        p = self.params
        dim = p['inner_dim']
        if input_dim != dim:
            raise ValueError('ADirect requires input_dim == inner_dim for its raw local mean path')
        self.allocation = allocate_adirect_tokens(p['num_tokens'], p['global_fraction'])
        g = self.allocation['global_tokens']
        self.requires_sequence = bool(g and p['sequence_condition'])
        self.input_adapter = (InputAdapter(input_dim, dim, p['position_mode'],
                                          p['position_frequencies'], p['sequence_condition'])
                              if g else None)
        self.global_queries = nn.Parameter(torch.randn(g, dim) * dim**-.5) if g else None
        self.global_layers = nn.ModuleList(
            QueryBlock(dim, p['num_heads'], p['ffn_expansion'], p['dropout'])
            for _ in range(p['global_depth']) if g)
        self.final_norm = nn.LayerNorm(dim)
        self.head = mapping_head(dim, output_dim, config)

    def _local_tokens(self, features, grid, extent):
        indices, valid = packed_local_regions(grid, self.allocation, extent)
        indices, valid = indices.to(features.device), valid.to(features.device)
        return masked_mean(features[indices], valid)

    def _local_details(self):
        nested = self.allocation['children_per_region'] == 4
        return dict(routing_mode='nested_raw_spatial_mean' if nested else 'exact_raw_spatial_mean',
                    local_token_order='parent_major_child_major' if nested else 'spatial_box_lexicographic',
                    local_feature_source='encoder_output',
                    local_position_mode='none', local_sequence_condition=False)

    def forward(self, features, coords, metadata):
        grid = self.inputs(features, coords, metadata)
        check_budget(len(features), self.params['num_tokens'])
        sequence = metadata.get('sequence')
        if self.requires_sequence and sequence is None:
            raise ValueError('ADirect global branch requires MRI sequence context')
        extent = physical_extent(grid, metadata)
        use_checkpoint = self.params['checkpoint_blocks'] and self.training and torch.is_grad_enabled()
        local = (checkpoint(self._local_tokens, features, grid, extent, use_reentrant=False)
                 if use_checkpoint else self._local_tokens(features, grid, extent))
        if self.global_queries is None:
            glob = features.new_empty((0, features.shape[-1]))
        else:
            h = self.input_adapter(features, coords, metadata, sequence)
            glob, memory = self.global_queries[None], h[None]
            for layer in self.global_layers:
                glob = (checkpoint(layer, glob, memory, use_reentrant=False) if use_checkpoint
                        else layer(glob, memory))
            glob = glob[0]
        output = self.head(self.final_norm(torch.cat((glob, local), dim=0)))
        return self.finish(output, features, metadata, **self.allocation,
                           **self._local_details(),
                           global_position_mode=self.params['position_mode'] if self.global_queries is not None else 'none',
                           global_sequence_condition=self.requires_sequence,
                           global_depth=len(self.global_layers))
