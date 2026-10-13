import math
import torch
from torch import nn
from torch.nn import functional as F
from src.models.projectors.common import StructuredProjector, mapping_head
from src.models.projectors.geometry import partition, grid_mean, group_metadata


class WindowMixerBlock(nn.Module):
    def __init__(self, dim, window, expansion, token_expansion, shifted):
        super().__init__()
        self.window = tuple(window)
        self.left = tuple(n//2 if shifted else 0 for n in window)
        tokens = math.prod(window)
        self.norm = nn.LayerNorm(dim)
        self.token_mlp = nn.Sequential(nn.Linear(tokens,tokens*token_expansion),nn.GELU(),
                                       nn.Linear(tokens*token_expansion,tokens))
        self.channel_mlp = nn.Sequential(nn.LayerNorm(dim),nn.Linear(dim,dim*expansion),nn.GELU(),
                                         nn.Linear(dim*expansion,dim))

    def forward(self, x):
        original = x.shape[:3]
        right = tuple(-(n+l)%w for n,l,w in zip(original,self.left,self.window))
        pad = (self.left[2],right[2],self.left[1],right[1],self.left[0],right[0])
        padded = F.pad(x.permute(3,0,1,2),pad).permute(1,2,3,0)
        valid = F.pad(torch.ones(original,device=x.device,dtype=torch.bool),pad)
        counts = tuple(n//w for n,w in zip(padded.shape[:3],self.window))
        a,b,c = counts; u,v,w = self.window; channels=x.shape[-1]
        def pack(t):
            return t.reshape(a,u,b,v,c,w,-1).permute(0,2,4,1,3,5,6).reshape(a*b*c,u*v*w,-1)
        y,mask = pack(padded),pack(valid[...,None])
        normalized = self.norm(y).masked_fill(~mask,0)
        y = (y+self.token_mlp(normalized.transpose(1,2)).transpose(1,2)).masked_fill(~mask,0)
        y = (y+self.channel_mlp(y)).masked_fill(~mask,0)
        y = y.reshape(a,b,c,u,v,w,channels).permute(0,3,1,4,2,5,6).reshape(*padded.shape)
        # Padding-based offset, not torch.roll: opposite volume faces never become neighbors.
        return y[tuple(slice(l,l+n) for l,n in zip(self.left,original))]


class Mixer3D(StructuredProjector):
    def __init__(self,input_dim,output_dim,config):
        super().__init__(input_dim,output_dim,config)
        dim=self.params['inner_dim'] or input_dim
        self.adapt=nn.Identity() if dim==input_dim else nn.Linear(input_dim,dim)
        self.blocks=nn.Sequential(*(WindowMixerBlock(dim,self.params['window_size'],self.params['ffn_expansion'],
            self.params['token_expansion'],bool(i%2 and self.params['shift_windows']))
            for i in range(self.params['depth'])))
        self.head=mapping_head(dim,output_dim,config)

    def forward(self,features,coords,metadata):
        grid=self.inputs(features,coords,metadata)
        output_grid,groups=partition(grid,metadata,self.params['num_tokens'],self.params['budget_policy'])
        x=self.adapt(features)
        x=self.blocks(x.reshape(*grid,x.shape[-1])).reshape(len(features),-1)
        return self.finish(self.head(grid_mean(x,groups)),features,metadata,output_grid=list(output_grid),
                           **group_metadata(coords,groups))
