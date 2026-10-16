"""Patch-to-slot competition followed by slot-normalized evidence aggregation."""
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint


def accumulation_dtype(tensor):
    return torch.float32 if tensor.dtype in (torch.bfloat16, torch.float16) else tensor.dtype


class CompetitiveSlotRouter(nn.Module):
    def __init__(self, dim, num_slots, iterations=3, background=True, epsilon=1e-6,
                 expansion=4, checkpoint_blocks=False):
        super().__init__()
        self.num_slots, self.background, self.iterations = num_slots, background, iterations
        self.epsilon, self.checkpoint_blocks = epsilon, checkpoint_blocks
        # Background is index 0; other slots have distinct learned initial states.
        self.initial_slots = nn.Parameter(torch.randn(num_slots+int(background),dim)*dim**-.5)
        self.norm_slots, self.norm_input = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.to_q, self.to_k = nn.Linear(dim,dim,bias=False), nn.Linear(dim,dim,bias=False)
        self.gru = nn.GRUCell(dim,dim)
        self.mlp = nn.Sequential(nn.LayerNorm(dim),nn.Linear(dim,expansion*dim),
                                 nn.GELU(),nn.Linear(expansion*dim,dim))
        self.scale = dim**-.5

    def assignment(self, slots, keys):
        queries = self.to_q(self.norm_slots(slots))
        with torch.autocast(device_type=slots.device.type,enabled=False):
            dtype = accumulation_dtype(queries)
            logits = keys.to(dtype)@queries.to(dtype).T*self.scale
            return logits.softmax(dim=-1)  # each patch competes across slots

    def update(self, slots, keys, features):
        a = self.assignment(slots,keys)
        with torch.autocast(device_type=slots.device.type,enabled=False):
            weights = a/(a.sum(dim=0,keepdim=True)+self.epsilon)
            evidence = weights.T@features.to(weights.dtype)
        updated = self.gru(evidence.to(slots.dtype),slots)
        return updated+self.mlp(updated)

    def forward(self, features):
        keys = self.to_k(self.norm_input(features))  # reused through all iterations
        slots = self.initial_slots  # new image always starts from learned parameters
        for _ in range(self.iterations):
            if self.checkpoint_blocks and self.training and torch.is_grad_enabled():
                slots = checkpoint(self.update,slots,keys,features,use_reentrant=False)
            else:
                slots = self.update(slots,keys,features)
        assignments = self.assignment(slots,keys)  # recompute after the last update
        return slots[int(self.background):], assignments
