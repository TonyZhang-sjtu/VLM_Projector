"""Role-conditioned visual values with a differentiable shared support bias."""
from contextlib import nullcontext
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.attention import sdpa_kernel, SDPBackend
from torch.utils.checkpoint import checkpoint
from .slots import accumulation_dtype


class SupportReader(nn.Module):
    def __init__(self, dim, heads, beta, epsilon=1e-6, expansion=4, dropout=0.,
                 chunk_size=128, checkpoint_blocks=False):
        super().__init__()
        self.heads, self.dropout, self.epsilon = heads, dropout, epsilon
        self.chunk_size, self.checkpoint_blocks = chunk_size, checkpoint_blocks
        self.register_buffer('support_beta',torch.tensor(beta,dtype=torch.float32))
        self.norm_query, self.norm_visual = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.to_q, self.to_kv = nn.Linear(dim,dim,bias=False), nn.Linear(dim,2*dim,bias=False)
        self.to_out = nn.Linear(dim,dim,bias=False)
        self.ffn = nn.Sequential(nn.LayerNorm(dim),nn.Linear(dim,dim*expansion),
                                 nn.GELU(),nn.Linear(dim*expansion,dim))

    def _attend(self, query, keys, values, support):
        k_slots, roles, dim = query.shape
        q = query.reshape(1,k_slots*roles,self.heads,dim//self.heads).transpose(1,2)
        # Bias construction stays FP32; casts remain differentiable to assignment.
        with torch.autocast(device_type=query.device.type,enabled=False):
            log_support = (support.to(accumulation_dtype(support))+self.epsilon).log()
            bias = (log_support[:,None,:]*self.support_beta.float()[None,:,None])
            bias = bias.reshape(1,1,k_slots*roles,-1).to(query.dtype)
        context = sdpa_kernel(SDPBackend.MATH) if torch.are_deterministic_algorithms_enabled() else nullcontext()
        with context:
            attended = F.scaled_dot_product_attention(
                q,keys,values,attn_mask=bias,
                dropout_p=self.dropout if self.training else 0.)
        return attended.transpose(1,2).reshape(k_slots,roles,dim)

    def forward(self, queries, visual, assignments):
        k_slots, roles, dim = queries.shape
        if assignments.shape != (len(visual),k_slots) or roles != len(self.support_beta):
            raise ValueError('SupportReader requires [N,K] support for [K,R,d] queries')
        query = self.to_q(self.norm_query(queries))
        keys,values = self.to_kv(self.norm_visual(visual)).chunk(2,-1)
        keys,values = [x.reshape(1,-1,self.heads,dim//self.heads).transpose(1,2) for x in (keys,values)]
        count = max(1,self.chunk_size//roles)
        outputs = []
        for start in range(0,k_slots,count):
            args = (query[start:start+count],keys,values,assignments[:,start:start+count].T)
            if self.checkpoint_blocks and self.training and torch.is_grad_enabled():
                outputs.append(checkpoint(self._attend,*args,use_reentrant=False))
            else:
                outputs.append(self._attend(*args))
        tokens = self.to_out(torch.cat(outputs,dim=0))
        # No direct knowledge/query residual bypasses visual value reading.
        return tokens+self.ffn(tokens)
