import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from src.models.projectors.common import StructuredProjector, QueryBlock, mapping_head
from src.models.projectors.geometry import pack_groups, group_metadata


class LocalPacker3D(StructuredProjector):
    def __init__(self,input_dim,output_dim,config):
        super().__init__(input_dim,output_dim,config)
        dim=self.params['inner_dim'] or input_dim
        self.adapt=nn.Identity() if dim==input_dim else nn.Linear(input_dim,dim)
        self.layers=nn.ModuleList(QueryBlock(dim,self.params['num_heads'],self.params['ffn_expansion'],
                                            self.params['dropout']) for _ in range(self.params['depth']))
        self.final_norm=nn.LayerNorm(dim)
        self.head=mapping_head(dim,output_dim,config)

    def forward(self,features,coords,metadata):
        grid=self.inputs(features,coords,metadata)
        output_grid,groups=self.partition(grid,metadata)
        memory,valid=pack_groups(self.adapt(features),groups)
        queries=(memory.float().sum(1)/valid.sum(1)[:,None]).to(memory.dtype).unsqueeze(1)
        for layer in self.layers:
            if self.params['checkpoint_blocks'] and self.training and torch.is_grad_enabled():
                queries=checkpoint(layer,queries,memory,valid,use_reentrant=False)
            else:queries=layer(queries,memory,valid)
        output=self.head(self.final_norm(queries[:,0]))
        return self.finish(output,features,metadata,output_grid=list(output_grid),
                           **group_metadata(coords,groups))
