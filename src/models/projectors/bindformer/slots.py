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
        self.value_norm = self.value_projection = None
        self.spatial_adaptive = False
        self.spatial_temperature = 1.0

    def configure_routing(self, normalization=False, projection=False, spatial=False,
                          adaptive=False, temperature=1.0):
        """Called after the legacy projector is initialized, preserving its RNG."""
        dim = self.initial_slots.shape[-1]
        self.value_norm = nn.LayerNorm(dim) if normalization else None
        self.value_projection = nn.Linear(dim, dim, bias=False) if projection else None
        if self.value_projection is not None:
            nn.init.eye_(self.value_projection.weight)
        self.spatial_adaptive, self.spatial_temperature = adaptive, temperature
        # Uniform regional pooling has no Q/K route; do not optimize dead heads.
        if spatial and not adaptive:
            for module in (self.norm_input, self.norm_slots, self.to_q, self.to_k):
                module.requires_grad_(False)

    def values(self, features):
        values = features if self.value_norm is None else self.value_norm(features)
        return values if self.value_projection is None else self.value_projection(values)

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

    def regional_weights(self, slots, keys, indices, valid):
        if not self.spatial_adaptive:
            weights = valid.to(accumulation_dtype(slots))
            return weights/weights.sum(-1, keepdim=True)
        queries = self.to_q(self.norm_slots(slots))
        with torch.autocast(device_type=slots.device.type, enabled=False):
            dtype = accumulation_dtype(queries)
            logits = (keys[indices].to(dtype)*queries.to(dtype)[:, None, :]).sum(-1)
            logits = logits*(self.scale/self.spatial_temperature)
            return logits.masked_fill(~valid, -torch.inf).softmax(-1)  # PATCH axis within each box

    def regional_update(self, slots, keys, values, indices, valid):
        weights = self.regional_weights(slots, keys, indices, valid)
        with torch.autocast(device_type=slots.device.type, enabled=False):
            evidence = (weights[..., None]*values[indices].to(weights.dtype)).sum(1)
        updated = self.gru(evidence.to(slots.dtype), slots)
        return updated+self.mlp(updated)

    def forward(self, features, regions=None):
        values = self.values(features)
        if regions is not None:
            # Preserve the legacy parameter layout; the background initial state
            # is unused here. Each local slot owns a nonempty region.
            slots = self.initial_slots[int(self.background):]
            keys = self.to_k(self.norm_input(features)) if self.spatial_adaptive else features
            for _ in range(self.iterations):
                args = (slots, keys, values, regions.indices, regions.valid)
                if self.checkpoint_blocks and self.training and torch.is_grad_enabled():
                    slots = checkpoint(self.regional_update, *args, use_reentrant=False)
                else:
                    slots = self.regional_update(*args)
            weights = self.regional_weights(slots, keys, regions.indices, regions.valid)
            return slots, regions.dense(weights)
        keys = self.to_k(self.norm_input(features))  # reused through all iterations
        slots = self.initial_slots  # new image always starts from learned parameters
        for _ in range(self.iterations):
            if self.checkpoint_blocks and self.training and torch.is_grad_enabled():
                slots = checkpoint(self.update,slots,keys,values,use_reentrant=False)
            else:
                slots = self.update(slots,keys,values)
        assignments = self.assignment(slots,keys)  # recompute after the last update
        return slots[int(self.background):], assignments
