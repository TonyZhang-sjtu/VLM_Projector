import math
import torch
from torch.nn import functional as F
from src.models.device_transfer import move_tensor
from src.models.projectors.common import StructuredProjector, mapping_head


class ConcatAdapter3D(StructuredProjector):
    def __init__(self, input_dim, output_dim, config):
        super().__init__(input_dim, output_dim, config)
        self.group = tuple(self.params['group_size'])
        self.head = mapping_head(math.prod(self.group)*input_dim, output_dim, config)

    def forward(self, features, coords, metadata):
        grid = self.inputs(features, coords, metadata)
        # Stage across host memory; direct peer copies are unreliable on some GPUs.
        features = move_tensor(features, next(self.head.parameters()).device)
        if features.device.type == 'cpu':
            features = features.to(next(self.head.parameters()).dtype)
        group = self.group
        out = tuple(math.ceil(n/r) for n,r in zip(grid,group))
        pads = tuple(m*r-n for n,m,r in zip(grid,out,group))
        x = features.reshape(*grid,self.input_dim).permute(3,0,1,2)
        x = F.pad(x,(0,pads[2],0,pads[1],0,pads[0])).permute(1,2,3,0)
        # Outer cell order D/H/W, then inner slot order d/h/w, then channels.
        x = x.reshape(out[0],group[0],out[1],group[1],out[2],group[2],self.input_dim)
        x = x.permute(0,2,4,1,3,5,6).reshape(math.prod(out),-1)
        output = self.head(x)
        return self.finish(output, features, metadata, output_grid=list(out), group_size=list(group),
                           empty_slots=math.prod(out)*math.prod(group)-len(features))
