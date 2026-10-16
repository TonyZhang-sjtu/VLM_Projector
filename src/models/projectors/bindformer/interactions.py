"""Optional shared within-bundle SA and gated local-to-global CA, no extra tokens."""
import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.attention import sdpa_kernel, SDPBackend
from torch.utils.checkpoint import checkpoint


class InteractionLayer(nn.Module):
    def __init__(self, dim, heads, dropout=0., cross=False, gate_init=.1):
        super().__init__()
        self.heads,self.dropout,self.cross = heads,dropout,cross
        self.norm = nn.LayerNorm(dim)
        self.memory_norm = nn.LayerNorm(dim) if cross else None
        self.q = nn.Linear(dim,dim)
        self.kv = nn.Linear(dim,2*dim)
        self.out = nn.Linear(dim,dim)
        self.gate = nn.Linear(dim,1) if cross else None
        if cross:
            nn.init.zeros_(self.gate.weight)
            nn.init.constant_(self.gate.bias,math.log(gate_init/(1-gate_init)))

    def forward(self, bundles, global_tokens):
        k,q,d = bundles.shape
        normalized = self.norm(bundles)
        memory = self.memory_norm(global_tokens)[None,:,:] if self.cross else normalized
        if self.cross and len(global_tokens) == 0:
            raise ValueError('Local-to-global attention requires global tokens')
        def heads(x):
            return x.reshape(x.shape[0],-1,self.heads,d//self.heads).transpose(1,2)
        keys,values = self.kv(memory).chunk(2,-1)
        # cuDNN SDPA cannot reliably backpropagate broadcast global K/V across
        # slot batches in our supported torch/CUDA environment. Let flash or
        # memory-efficient kernels run when supported, with math as fallback.
        backends = [SDPBackend.MATH] if torch.are_deterministic_algorithms_enabled() else [
            SDPBackend.FLASH_ATTENTION,SDPBackend.EFFICIENT_ATTENTION,SDPBackend.MATH]
        context = sdpa_kernel(backends)
        with context:
            read = F.scaled_dot_product_attention(heads(self.q(normalized)),heads(keys),heads(values),
                        dropout_p=self.dropout if self.training else 0.)
        read = self.out(read.transpose(1,2).reshape(k,q,d))
        if self.cross:
            read = read*self.gate(normalized.mean(1)).sigmoid()[:,None,:]
        return bundles+read


class BundleInteractions(nn.Module):
    def __init__(self, dim=None, heads=8, within=False, cross=False, within_depth=1,
                 cross_depth=1, gate_init=.1, dropout=0., checkpoint_blocks=False):
        super().__init__()
        self.checkpoint_blocks = checkpoint_blocks
        self.within = nn.ModuleList(InteractionLayer(dim,heads,dropout)
                                   for _ in range(within_depth) if within)
        self.cross = nn.ModuleList(InteractionLayer(dim,heads,dropout,True,gate_init)
                                  for _ in range(cross_depth) if cross)

    def forward(self, role_tokens, global_tokens):
        for layer in [*self.within,*self.cross]:
            if self.checkpoint_blocks and self.training and torch.is_grad_enabled():
                role_tokens = checkpoint(layer,role_tokens,global_tokens,use_reentrant=False)
            else:
                role_tokens = layer(role_tokens,global_tokens)
        return role_tokens
