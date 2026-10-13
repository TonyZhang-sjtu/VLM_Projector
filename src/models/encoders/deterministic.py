"""Local NeuroVFM attention adapter exposing FlashAttention's deterministic backward.

The upstream state-dict names and all Linear/LoRA objects are retained. Only its
packed FlashAttention call gains deterministic=True; vendor sources stay intact.
"""
from types import MethodType
from flash_attn import flash_attn_varlen_qkvpacked_func
from neurovfm.models.vit import SelfAttention


def _forward(self,x,cu_seqlens=None,max_seqlen=None,use_flash_attn=True,return_attn_weights=False):
    if not use_flash_attn or return_attn_weights or cu_seqlens is None or max_seqlen is None:
        raise ValueError('Deterministic NeuroVFM adapter requires the validated packed FlashAttention interface')
    qkv=self.qkv(x).reshape(x.size(0),3,self.num_heads,self.dim_head)
    value=flash_attn_varlen_qkvpacked_func(qkv,cu_seqlens,max_seqlen,
        dropout_p=self.attn_drop if self.training else 0.,softmax_scale=self.scale,
        causal=False,deterministic=True).flatten(start_dim=1)
    return self.proj(value),None


def enable_deterministic_attention(core):
    mixers=[block.mixer for block in core.blocks]
    if len(mixers)!=12 or any(type(m) is not SelfAttention for m in mixers):
        raise ValueError('Deterministic adapter only supports the checked local NeuroVFM SelfAttention')
    for mixer in mixers:mixer.forward=MethodType(_forward,mixer)
