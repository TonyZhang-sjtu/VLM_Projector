import torch
from torch import nn


class RoleQueries(nn.Module):
    def __init__(self, dim, num_roles, text_dim=None, query_counts=None):
        super().__init__()
        self.from_slot = nn.Linear(dim,dim,bias=False)
        self.role_embedding = nn.Parameter(torch.randn(num_roles,dim)*dim**-.5)
        self.from_concept = (nn.ModuleList(nn.Linear(text_dim,dim,bias=False) for _ in range(num_roles))
                             if text_dim is not None else None)
        self.query_counts = list(query_counts or [1]*num_roles)
        self.role_indices = [r for r,n in enumerate(self.query_counts) for _ in range(n)]
        # Allocate offsets only for additional subqueries. The first query of
        # each role remains its original role query; offsets are shared by slots.
        extra = sum(self.query_counts)-num_roles
        self.subquery_embedding = (nn.Parameter(torch.randn(extra,dim)*dim**-.5)
                                   if extra else None)

    def forward(self, slots, concepts=None, geometry=None):
        queries = self.from_slot(slots)[:,None,:]+self.role_embedding[None,:,:]
        if self.from_concept is not None:
            if concepts is None:
                raise ValueError('Knowledge queries require concept mixtures')
            queries = queries+torch.stack([mapping(concepts[:,r])
                                           for r,mapping in enumerate(self.from_concept)],dim=1)
        if geometry is not None:
            if geometry.shape != queries.shape:
                raise ValueError('Geometry embedding must match [slots,roles,dim]')
            queries = queries+geometry
        if self.subquery_embedding is not None:
            pieces, offset = [], 0
            for r, count in enumerate(self.query_counts):
                base = queries[:,r:r+1]
                pieces.append(base)
                if count > 1:
                    pieces.append(base+self.subquery_embedding[offset:offset+count-1][None,:,:])
                    offset += count-1
            queries = torch.cat(pieces,dim=1)
        return queries
