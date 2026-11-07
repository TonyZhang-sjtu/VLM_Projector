"""Four registered experiments reusing the production attention/head operators."""
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from src.models.projectors.common import StructuredProjector, QueryBlock, mapping_head
from src.models.projectors.geometry import check_budget
from src.models.projectors.bindformer.input_adapter import InputAdapter
from src.models.projectors.bindformer_adirect.projector import BindFormerADirect
from src.models.projectors.bindformer_adirect.regions import packed_local_regions, masked_mean
from src.models.projectors.bindformer_adirect_localread.projector import BindFormerADirectLocalRead
from .budget import A1, A2, A3, A4, allocate_ablation
from .regions import packed_tail_regions


class NoLocalRead(BindFormerADirect):
    def __init__(self, input_dim, output_dim, config):
        super().__init__(input_dim, output_dim, config)
        self.allocation = allocate_ablation(A1, self.params['num_tokens'], self.params['global_fraction'])


class NoMeanBypass(BindFormerADirectLocalRead):
    def __init__(self, input_dim, output_dim, config):
        super().__init__(input_dim, output_dim, config)
        self.allocation = allocate_ablation(A2, self.params['num_tokens'], self.params['global_fraction'])

    def _local_tokens(self, features, grid, extent):
        indices, valid = packed_local_regions(grid, self.allocation, extent)
        indices, valid = indices.to(features.device), valid.to(features.device)
        values = features[indices]
        return self.local_reader(masked_mean(values, valid), values, valid, add_mean=False)

    def _local_details(self):
        return {**super()._local_details(), 'routing_mode': 'mean_query_local_read_no_bypass',
                'mean_query': True, 'mean_bypass': False}


class GlobalOnly(StructuredProjector):
    def __init__(self, input_dim, output_dim, config):
        super().__init__(input_dim, output_dim, config)
        p, dim = self.params, self.params['inner_dim']
        if input_dim != dim:
            raise ValueError('AnchorRead3D ablations require input_dim == inner_dim')
        self.allocation = allocate_ablation(A3, p['num_tokens'], p['global_fraction'])
        self.requires_sequence = p['sequence_condition']
        # Same modules, order and parameter names as ADirect's global path/head.
        self.input_adapter = InputAdapter(input_dim, dim, p['position_mode'],
                                          p['position_frequencies'], p['sequence_condition'])
        self.global_queries = nn.Parameter(torch.randn(p['num_tokens'], dim) * dim**-.5)
        self.global_layers = nn.ModuleList(
            QueryBlock(dim, p['num_heads'], p['ffn_expansion'], p['dropout'])
            for _ in range(p['global_depth']))
        self.final_norm = nn.LayerNorm(dim)
        self.head = mapping_head(dim, output_dim, config)

    def forward(self, features, coords, metadata):
        self.inputs(features, coords, metadata)
        check_budget(len(features), self.params['num_tokens'])
        sequence = metadata.get('sequence')
        if self.requires_sequence and sequence is None:
            raise ValueError('ADirect global branch requires MRI sequence context')
        h = self.input_adapter(features, coords, metadata, sequence)
        glob, memory = self.global_queries[None], h[None]
        use_checkpoint = self.params['checkpoint_blocks'] and self.training and torch.is_grad_enabled()
        for layer in self.global_layers:
            glob = (checkpoint(layer, glob, memory, use_reentrant=False) if use_checkpoint
                    else layer(glob, memory))
        output = self.head(self.final_norm(glob[0]))
        return self.finish(output, features, metadata, **self.allocation,
                           routing_mode='global_only', global_position_mode=self.params['position_mode'],
                           global_sequence_condition=self.requires_sequence,
                           global_depth=len(self.global_layers))


class LocalOnly(BindFormerADirectLocalRead):
    def __init__(self, input_dim, output_dim, config):
        super().__init__(input_dim, output_dim, config)
        self.allocation = allocate_ablation(A4, self.params['num_tokens'], self.params['global_fraction'])

    def _local_tokens(self, features, grid, extent):
        indices, valid = packed_tail_regions(tuple(grid), self.params['num_tokens'], tuple(extent))
        indices, valid = indices.to(features.device), valid.to(features.device)
        values = features[indices]
        return self.local_reader(masked_mean(values, valid), values, valid)

    def _local_details(self):
        return dict(routing_mode='nested4_tail_raw_mean_local_read',
                    local_token_order='parent_major_child_major', local_feature_source='encoder_output',
                    local_position_mode='none', local_sequence_condition=False,
                    local_read_depth=1, local_read_heads=self.params['num_heads'],
                    local_read_support='same_spatial_cell', local_read_ffn=False,
                    local_residual_scale_init=self.params['local_residual_scale_init'],
                    mean_query=True, mean_bypass=True)


PROJECTORS = {A1: NoLocalRead, A2: NoMeanBypass, A3: GlobalOnly, A4: LocalOnly}
