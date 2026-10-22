from src.models.projectors.common import StructuredProjector, mapping_head
from src.models.projectors.geometry import grid_mean, group_metadata


class AveragePool3D(StructuredProjector):
    def __init__(self, input_dim, output_dim, config):
        super().__init__(input_dim, output_dim, config)
        self.head = mapping_head(input_dim, output_dim, config)

    def forward(self, features, coords, metadata):
        grid = self.inputs(features, coords, metadata)
        output_grid, groups = self.partition(grid, metadata)
        output = self.head(grid_mean(features, groups))
        return self.finish(output, features, metadata, output_grid=list(output_grid),
                           **group_metadata(coords, groups))
