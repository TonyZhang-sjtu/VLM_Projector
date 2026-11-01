"""Replace ADirect's global summaries with sorted diversity-selected raw patches."""
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from src.models.projectors.common import StructuredProjector, mapping_head
from src.models.projectors.geometry import check_budget, physical_extent
from src.models.projectors.selection import diverse_indices
from src.models.projectors.bindformer_adirect.regions import packed_local_regions, masked_mean
from .config import allocate_detail_tokens


class BindFormerADirectDetail(StructuredProjector):
    def __init__(self, input_dim, output_dim, config):
        super().__init__(input_dim, output_dim, config)
        if input_dim != self.params['inner_dim']:
            raise ValueError('ADirect_Detail requires input_dim == inner_dim for raw evidence paths')
        self.allocation = allocate_detail_tokens(self.params['num_tokens'], self.params['detail_fraction'])
        self.requires_sequence = False
        self.final_norm = nn.LayerNorm(input_dim)
        self.head = mapping_head(input_dim, output_dim, config)

    def _local_tokens(self, features, grid, extent):
        indices, valid = packed_local_regions(grid, self.allocation, extent)
        return masked_mean(features[indices.to(features.device)], valid.to(features.device))

    def forward(self, features, coords, metadata):
        grid = self.inputs(features, coords, metadata)
        check_budget(len(features), self.params['num_tokens'])
        extent = physical_extent(grid, metadata)
        use_checkpoint = self.params['checkpoint_blocks'] and self.training and torch.is_grad_enabled()
        local = (checkpoint(self._local_tokens, features, grid, extent, use_reentrant=False)
                 if use_checkpoint else self._local_tokens(features, grid, extent))
        # Only discrete routing is detached; selected raw values retain gradients.
        indices = diverse_indices(features, self.allocation['detail_tokens'])
        detail = features.index_select(0, indices)
        output = self.head(self.final_norm(torch.cat((detail, local), dim=0)))
        return self.finish(output, features, metadata, **self.allocation,
                           routing_mode='diversity_detail_plus_nested_raw_spatial_mean',
                           token_order='detail_then_local', detail_token_order='encoder_grid_index',
                           detail_strategy='diversity', selected_indices=indices.cpu().tolist(),
                           local_token_order='parent_major_child_major',
                           local_feature_source='encoder_output', detail_feature_source='encoder_output',
                           local_position_mode='none', local_sequence_condition=False,
                           global_position_mode='none', global_sequence_condition=False, global_depth=0)
