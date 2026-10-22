"""Isolated role-specific value residuals. No production module imports this file."""
import copy
from contextlib import nullcontext
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.attention import sdpa_kernel, SDPBackend
from torch.utils.checkpoint import checkpoint
from src.models.projectors.bindformer.reader import SupportReader


class RoleValueReader(SupportReader):
    def __init__(self, original, enabled=True, rank=16, alpha=16.):
        nn.Module.__init__(self)
        for name in ('heads','dropout','epsilon','chunk_size','checkpoint_blocks'):
            setattr(self,name,getattr(original,name))
        self.register_buffer('support_beta',original.support_beta.detach().clone())
        for name in ('norm_query','norm_visual','to_q','to_kv','to_out','ffn'):
            setattr(self,name,copy.deepcopy(getattr(original,name)))
        self.enabled,self.rank,self.alpha=enabled,rank,float(alpha)
        self.roles=len(self.support_beta)
        dim=self.to_q.in_features
        self.value_down=nn.ModuleList();self.value_up=nn.ModuleList()
        # Adapter initialization must not perturb any later common random draws.
        if enabled:
            with torch.random.fork_rng(devices=[]):
                for _ in range(self.roles):
                    self.value_down.append(nn.Linear(dim,rank,bias=False))
                    self.value_up.append(nn.Linear(rank,dim,bias=False))
                    nn.init.zeros_(self.value_up[-1].weight)
        self.to(device=original.to_q.weight.device,dtype=original.to_q.weight.dtype)

    def role_values(self,values):
        if not self.enabled:
            return values.unsqueeze(0).expand(self.roles,-1,-1)
        # Never bypass the zero-initialized branch: up must receive its first gradient.
        return torch.stack([values+(self.alpha/self.rank)*up(down(values))
                            for down,up in zip(self.value_down,self.value_up)])

    def _attend_role_regions(self,query,keys,role_values,support,valid):
        slots,roles,dim=query.shape;hd=dim//self.heads
        q=query.reshape(slots,roles,self.heads,hd).permute(0,2,1,3)
        k=keys.reshape(slots,-1,self.heads,hd).transpose(1,2)
        with torch.autocast(device_type=query.device.type,enabled=False):
            bias=(support.float()+self.epsilon).log()[:,None,:]*self.support_beta.float()[None,:,None]
            bias=bias.masked_fill(~valid[:,None,:],-torch.inf).to(query.dtype)
        outputs=[]
        # Same calls for enabled and disabled variants: kernel-matched control.
        context=sdpa_kernel(SDPBackend.MATH) if torch.are_deterministic_algorithms_enabled() else nullcontext()
        with context:
            for r in range(roles):
                v=role_values[r].reshape(slots,-1,self.heads,hd).transpose(1,2)
                z=F.scaled_dot_product_attention(q[:,:,r:r+1],k,v,
                    attn_mask=bias[:,None,r:r+1,:],dropout_p=self.dropout if self.training else 0.)
                outputs.append(z.transpose(1,2).reshape(slots,1,dim))
        return torch.cat(outputs,dim=1)

    def forward(self,queries,visual,assignments,regions=None):
        if regions is None:
            raise ValueError('This isolated value-adapter experiment requires fixed spatial regions')
        slots,roles,dim=queries.shape
        if roles!=self.roles or assignments.shape!=(len(visual),slots):
            raise ValueError('Unexpected query/support shape')
        query=self.to_q(self.norm_query(queries))
        keys,values=self.to_kv(self.norm_visual(visual)).chunk(2,-1)
        role_values=self.role_values(values)
        outputs=[];count=max(1,self.chunk_size//roles)
        for start in range(0,slots,count):
            ids,valid=regions.indices[start:start+count],regions.valid[start:start+count]
            columns=torch.arange(start,start+len(ids),device=ids.device)[:,None]
            args=(query[start:start+count],keys[ids],role_values[:,ids],assignments[ids,columns],valid)
            if self.checkpoint_blocks and self.training and torch.is_grad_enabled():
                outputs.append(checkpoint(self._attend_role_regions,*args,use_reentrant=False))
            else:
                outputs.append(self._attend_role_regions(*args))
        tokens=self.to_out(torch.cat(outputs,dim=0))
        return tokens+self.ffn(tokens)
