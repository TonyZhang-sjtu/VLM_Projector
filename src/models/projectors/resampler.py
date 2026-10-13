import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from src.models.projectors.common import StructuredProjector, QueryBlock, mapping_head
from src.models.projectors.geometry import check_budget


class QueryResampler(StructuredProjector):
    def __init__(self,input_dim,output_dim,config):
        super().__init__(input_dim,output_dim,config)
        dim=self.params['inner_dim'] or input_dim
        self.adapt=nn.Identity() if dim==input_dim else nn.Linear(input_dim,dim)
        self.queries=nn.Parameter(torch.randn(self.params['num_tokens'],dim)*dim**-.5)
        self.layers=nn.ModuleList(QueryBlock(dim,self.params['num_heads'],self.params['ffn_expansion'],
            self.params['dropout'],latent_kv=self.name=='perceiver_flamingo')
            for _ in range(self.params.get('depth',1)))
        self.final_norm=nn.LayerNorm(dim)
        self.head=mapping_head(dim,output_dim,config)

    def forward(self,features,coords,metadata):
        self.inputs(features,coords,metadata)
        check_budget(len(features),self.params['num_tokens'])
        memory=self.adapt(features).unsqueeze(0)
        queries=self.queries.unsqueeze(0)
        for layer in self.layers:
            if self.params['checkpoint_blocks'] and self.training and torch.is_grad_enabled():
                queries=checkpoint(layer,queries,memory,use_reentrant=False)
            else:queries=layer(queries,memory)
        output=self.head(self.final_norm(queries[0]))
        return self.finish(output,features,metadata,depth=len(self.layers),
                           latent_interaction=self.name=='perceiver_flamingo')
