from torch import nn
from src.models.projectors.common import StructuredProjector, mapping_head
from src.models.projectors.geometry import partition, grid_mean, group_metadata


class ConvMixBlock(nn.Module):
    def __init__(self, dim, kernel, expansion):
        super().__init__()
        self.conv = nn.Conv3d(dim,dim,kernel,padding=kernel//2,groups=dim)
        self.channel = nn.Sequential(nn.LayerNorm(dim),nn.Linear(dim,dim*expansion),
                                     nn.GELU(),nn.Linear(dim*expansion,dim))

    def forward(self, x):
        mixed = self.conv(x.permute(3,0,1,2).unsqueeze(0))[0].permute(1,2,3,0)
        return x + self.channel(mixed)


class ConvPool3D(StructuredProjector):
    def __init__(self, input_dim, output_dim, config):
        super().__init__(input_dim,output_dim,config)
        dim = self.params['inner_dim'] or input_dim
        self.adapt = nn.Identity() if dim == input_dim else nn.Linear(input_dim,dim)
        self.blocks = nn.Sequential(*(ConvMixBlock(dim,self.params['kernel_size'],self.params['ffn_expansion'])
                                      for _ in range(self.params['depth'])))
        self.head = mapping_head(dim,output_dim,config)

    def forward(self, features, coords, metadata):
        grid = self.inputs(features,coords,metadata)
        output_grid,groups = partition(grid,metadata,self.params['num_tokens'],self.params['budget_policy'])
        x = self.adapt(features)
        x = self.blocks(x.reshape(*grid,x.shape[-1])).reshape(len(features),-1)
        return self.finish(self.head(grid_mean(x,groups)),features,metadata,output_grid=list(output_grid),
                           **group_metadata(coords,groups))
