"""Frozen, provenance-checked, scope-filtered concept memory; no patient labels."""
import json
import math
from pathlib import Path
import torch
from torch import nn
from safetensors.torch import load_file
from src.config import file_hash, digest
from .budget import ROLES
from .input_adapter import SEQUENCES
from .slots import accumulation_dtype


def read_bank(directory, scope):
    directory = Path(directory)
    manifest = json.loads((directory/'embedding_manifest.json').read_text())
    index = json.loads((directory/'concept_index.json').read_text())
    required = ('concept_bank.safetensors', 'concept_index.json', 'embeddings.npy',
                'role_mask.npy', 'sequence_mask.npy', 'scope_mask.npy', 'encoding_inputs.jsonl')
    for name in required:
        if manifest.get('files',{}).get(name) != file_hash(directory/name):
            raise ValueError(f'Concept asset hash mismatch: {name}')
    for filename,key in (('concepts.jsonl','concepts_sha256'),
                         ('build_manifest.json','build_manifest_sha256')):
        if manifest.get(key) != file_hash(directory/filename):
            raise ValueError(f'Concept asset hash mismatch: {filename}')
    if (index['roles'] != list(ROLES) or index['sequences'] != list(SEQUENCES)
            or scope not in index['vocabulary_scopes']):
        raise ValueError('Concept bank role/sequence/scope schema mismatch')
    if (manifest.get('template') != '{name}. {short_definition}'
            or manifest.get('pooling') != 'last_hidden_state[:, 0, :] (final-layer CLS)'
            or not manifest.get('frozen')):
        raise ValueError('Expected frozen name + short definition, final CLS concept bank')
    tensors = load_file(str(directory/'concept_bank.safetensors'),device='cpu')
    e, role, sequence, scopes = (tensors[k] for k in ('embeddings','role_mask','sequence_mask','scope_mask'))
    n = len(index['concept_ids'])
    if (e.ndim != 2 or e.shape[0] != n or e.dtype != torch.float32
            or role.shape != (4,n) or sequence.shape != (len(SEQUENCES),n)
            or scopes.shape != (len(index['vocabulary_scopes']),n)
            or any(t.dtype != torch.bool for t in (role,sequence,scopes))
            or len(set(index['concept_ids'])) != n or not torch.isfinite(e).all()
            or not torch.allclose(e.norm(dim=-1), torch.ones(n),atol=1e-5,rtol=1e-5)):
        raise ValueError('Invalid concept matrix/masks/IDs')
    keep = scopes[index['vocabulary_scopes'].index(scope)]
    ids = [i for i,include in zip(index['concept_ids'],keep.tolist()) if include]
    if not ids or len(ids) != manifest['scope_concept_counts'][scope]:
        raise ValueError('Concept scope is empty or count differs from manifest')
    role, sequence = role[:,keep], sequence[:,keep]
    if not (sequence[:,None,:]&role[None,:,:]).any(-1).all():
        raise ValueError('Empty role/sequence concept candidate set')
    # Physically remove other-cohort-only rows before any trainable projection.
    proof = {'manifest_sha256':file_hash(directory/'embedding_manifest.json'),
             'vocabulary_scope':scope,'concept_ids':ids,'source_concepts':n,
             'active_concepts':len(ids),'files':manifest['files']}
    return e[keep], role, sequence, torch.where(keep)[0], proof


class ConceptMemory(nn.Module):
    def __init__(self, directory, scope, dim):
        super().__init__()
        e, role, sequence, indices, self.provenance = read_bank(directory,scope)
        self.register_buffer('embeddings',e)
        self.register_buffer('role_mask',role)
        self.register_buffer('sequence_mask',sequence)
        self.register_buffer('source_indices',indices)
        self.register_buffer('asset_fingerprint',
                             torch.tensor(list(bytes.fromhex(digest(self.provenance))),dtype=torch.uint8))
        self.to_query = nn.Linear(dim,dim,bias=False)
        self.to_keys = nn.ModuleList(nn.Linear(e.shape[1],dim,bias=False) for _ in ROLES)
        self.scale = 1/math.sqrt(dim)

    def forward(self, slots, sequence):
        if sequence not in SEQUENCES:
            raise ValueError(f'Missing or unknown MRI sequence: {sequence!r}')
        allowed = self.role_mask & self.sequence_mask[SEQUENCES.index(sequence)][None,:]
        if not allowed.any(-1).all():
            raise ValueError('Empty role/sequence concept candidate set')
        keys = torch.stack([mapping(self.embeddings) for mapping in self.to_keys])
        queries = self.to_query(slots)
        with torch.autocast(device_type=slots.device.type,enabled=False):
            dtype = accumulation_dtype(queries)
            scores = torch.einsum('kd,rcd->krc',queries.to(dtype),keys.to(dtype))*self.scale
            weights = scores.masked_fill(~allowed[None,:,:],float('-inf')).softmax(-1)
            mixed = weights@self.embeddings.to(dtype)
        return mixed.to(slots.dtype), weights

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        saved = state_dict.get(prefix+'asset_fingerprint')
        if saved is None or not torch.equal(saved.cpu(),self.asset_fingerprint.cpu()):
            raise ValueError('Checkpoint concept asset fingerprint mismatch')
        return super()._load_from_state_dict(state_dict,prefix,*args,**kwargs)
