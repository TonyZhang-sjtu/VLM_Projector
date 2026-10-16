"""Training-only reconstruction from final Z; no X/A shortcut to the decoder."""
import hashlib
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from src.models.projectors.common import QueryBlock
from .geometry import patch_geometry
from .input_adapter import SEQUENCES


def sampling_seed(seed, sample_id, step=0):
    raw = f'{seed}:{sample_id}:{step}'.encode()
    return int.from_bytes(hashlib.sha256(raw).digest()[:8],'little') % (2**63-1)


def stratified_indices(coords, valid, count, seed):
    """One random point per occupied coarse cell, then fill without replacement.

    A private CPU generator keeps diagnostics/auxiliary sampling off the model's
    dropout RNG and makes accumulation partitioning irrelevant.
    """
    indices = torch.where(valid.detach().cpu()>0)[0]
    if not len(indices):
        raise ValueError('Feature reconstruction has no valid patches')
    count = min(count,len(indices))
    generator = torch.Generator().manual_seed(seed)
    shuffled = indices[torch.randperm(len(indices),generator=generator)]
    points = coords.detach().cpu().float()
    bins = max(1,round(count**(1/3)))
    cells = (points/(points.amax(0)+1)*bins).long().clamp_max(bins-1)
    codes = (cells[:,0]*bins+cells[:,1])*bins+cells[:,2]
    seen,first,rest = set(),[],[]
    for i in shuffled.tolist():
        code = int(codes[i])
        if code not in seen:
            first.append(i);seen.add(code)
        else:
            rest.append(i)
    return torch.tensor((first+rest)[:count],device=coords.device,dtype=torch.long)


class FeatureReconstructor(nn.Module):
    def __init__(self, output_dim, target_dim, dim=256, heads=4, samples=512,
                 chunk_size=128, checkpoint_blocks=False):
        super().__init__()
        self.target_dim,self.samples,self.chunk_size = target_dim,samples,chunk_size
        self.checkpoint_blocks = checkpoint_blocks
        self.norm_z = nn.LayerNorm(output_dim)
        self.adapt = nn.Linear(output_dim,dim)
        self.coordinate_query = nn.Sequential(nn.Linear(3,dim),nn.GELU(),nn.Linear(dim,dim))
        self.sequence_embedding = nn.Embedding(len(SEQUENCES),dim)
        self.block = QueryBlock(dim,heads,expansion=2,dropout=0.)
        self.output = nn.Linear(dim,target_dim)

    def predict(self, embeddings, centers, sequence):
        if sequence not in SEQUENCES:
            raise ValueError('Feature reconstruction requires known MRI sequence')
        memory = self.adapt(self.norm_z(embeddings))[None,:,:]
        query = self.coordinate_query(centers.to(embeddings.dtype))
        index = torch.tensor(SEQUENCES.index(sequence),device=embeddings.device)
        query = query+self.sequence_embedding(index)
        outputs = []
        for start in range(0,len(query),self.chunk_size):
            q = query[None,start:start+self.chunk_size,:]
            if self.checkpoint_blocks and self.training and torch.is_grad_enabled():
                value = checkpoint(self.block,q,memory,use_reentrant=False)
            else:
                value = self.block(q,memory)
            outputs.append(self.output(value[0]))
        return torch.cat(outputs,0)

    def objective(self, embeddings, coords, metadata, sequence, target, seed):
        if target.shape != (len(coords),self.target_dim) or not torch.isfinite(target).all():
            raise ValueError('Invalid reconstruction target features')
        geometry = patch_geometry(coords.to(embeddings.device),metadata,'grid')
        indices = stratified_indices(coords.to(embeddings.device),geometry.valid_fraction,self.samples,seed)
        predicted = self.predict(embeddings,geometry.centers[indices],sequence)
        with torch.autocast(device_type=embeddings.device.type,enabled=False):
            # No learned normalization or gradient on the teacher branch.
            expected = target.detach().to(device=embeddings.device,dtype=torch.float32)[indices]
            expected = F.layer_norm(expected,(self.target_dim,))
            actual = F.layer_norm(predicted.float(),(self.target_dim,))
            weights = geometry.valid_fraction[indices]
            error = (actual-expected).square().mean(-1)
            loss = (weights*error).sum()/weights.sum()
            cosine = (weights*F.cosine_similarity(actual,expected,dim=-1)).sum()/weights.sum()
        return {'loss':loss,'cosine':cosine.detach(),'patches':len(indices)}
