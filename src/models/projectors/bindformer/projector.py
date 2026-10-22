from pathlib import Path
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from src.models.projectors.common import StructuredProjector, QueryBlock, mapping_head
from src.models.projectors.geometry import check_budget
from .budget import allocate_tokens, role_counts
from .input_adapter import InputAdapter
from .slots import CompetitiveSlotRouter
from .concept_memory import ConceptMemory
from .role_queries import RoleQueries
from .reader import SupportReader
from .types import BindFormerAuxiliary, BundleInteractions
from .geometry import patch_geometry, SupportGeometryEncoder
from .reconstruction import FeatureReconstructor
from .regions import make_regions


class BindFormer3D(StructuredProjector):
    def __init__(self, input_dim, output_dim, config, project_root=None):
        super().__init__(input_dim,output_dim,config)
        p = self.params
        self.query_counts = role_counts(p['role_query_counts'])
        self.allocation = allocate_tokens(p['num_tokens'],p['global_fraction'],len(p['roles']),self.query_counts)
        dim, g, k = p['inner_dim'], self.allocation['global_tokens'], self.allocation['num_slots']
        self.requires_sequence = p['sequence_condition'] or p['knowledge_mode'] != 'none' or p['feature_loss_weight'] > 0
        self.input_adapter = InputAdapter(input_dim,dim,p['position_mode'],
                                         p['position_frequencies'],p['sequence_condition'])
        self.router = CompetitiveSlotRouter(dim,k,p['slot_iterations'],p['background_slot'],
                                            p['slot_epsilon'],p['ffn_expansion'],p['checkpoint_blocks'])
        self.concept_memory = None
        self.asset_provenance = None
        if p['knowledge_mode'] != 'none':
            from src.config import ROOT
            path = Path(p['concept_bank'])
            if not path.is_absolute():
                path = Path(project_root or ROOT)/path
            self.concept_memory = ConceptMemory(path,p['vocabulary_scope'],dim)
            self.asset_provenance = self.concept_memory.provenance
        text_dim = None if self.concept_memory is None else self.concept_memory.embeddings.shape[1]
        self.role_queries = RoleQueries(dim,len(p['roles']),text_dim,self.query_counts)
        beta = [b for b,n in zip(p['support_beta'],self.query_counts) for _ in range(n)]
        self.reader = SupportReader(dim,p['num_heads'],beta,p['slot_epsilon'],
                                    p['ffn_expansion'],p['dropout'],p['reader_chunk_size'],p['checkpoint_blocks'])
        self.global_queries = nn.Parameter(torch.randn(g,dim)*dim**-.5) if g else None
        self.global_layers = nn.ModuleList(QueryBlock(dim,p['num_heads'],p['ffn_expansion'],p['dropout'])
                                          for _ in range(p['global_depth']) if g)
        # Stable extension points; no inactive trainable heads in CE-only v1.
        self.geometry_provider = None
        self.geometry_encoder = None
        self.interactions = BundleInteractions()
        self.final_norm = nn.LayerNorm(dim)
        self.head = mapping_head(dim,output_dim,config)
        # Optional modules are constructed after the complete legacy core. Disabled
        # modules allocate no parameters/buffers and consume no random numbers.
        if p['geometry_enabled']:
            self.geometry_provider = patch_geometry
            self.geometry_encoder = SupportGeometryEncoder(dim,p['geometry_hidden_dim'],p['slot_epsilon'])
        if p['within_slot_sa'] or p['local_reads_global']:
            if p['local_reads_global'] and not g:
                raise ValueError('local_reads_global requires global tokens')
            self.interactions = BundleInteractions(dim,p['num_heads'],p['within_slot_sa'],p['local_reads_global'],
                p['within_slot_sa_depth'],p['local_global_depth'],p['local_global_gate_init'],
                p['dropout'],p['checkpoint_blocks'])
        self.feature_reconstructor = None
        if p['feature_loss_weight'] > 0:
            self.feature_reconstructor = FeatureReconstructor(output_dim,input_dim,p['feature_inner_dim'],
                p['feature_num_heads'],p['feature_samples'],p['feature_query_chunk_size'],p['checkpoint_blocks'])
        self.router.configure_routing(p['router_value_normalization'], p['router_value_projection'],
            p['fixed_spatial_regions'], p['spatial_adaptive_aggregation'], p['spatial_attention_temperature'])

    def forward(self, features, coords, metadata):
        p = self.params
        self.inputs(features,coords,metadata)
        check_budget(len(features),self.params['num_tokens'])
        sequence = metadata.get('sequence')
        if self.requires_sequence and sequence is None:
            raise ValueError('BindFormer requires MRI sequence context')
        h = self.input_adapter(features,coords,metadata,sequence)
        regions = (make_regions(metadata['grid_dhw'], metadata, self.allocation['num_slots'], h.device)
                   if p['fixed_spatial_regions'] else None)
        if regions is None:
            slots, assignments = self.router(h)
            local_support = assignments[:, int(p['background_slot']):]
            occupancy = local_support
        else:
            slots, local_support = self.router(h, regions)
            occupancy = regions.membership(local_support.dtype)
            assignments = occupancy
        concepts, weights = (None,None) if self.concept_memory is None else self.concept_memory(slots,sequence)
        geometry, geometry_queries = None, None
        if self.geometry_encoder is not None:
            geometry = self.geometry_provider(coords.to(h.device),metadata,p['geometry_space'],
                                               p['geometry_coordinate_scale'],p['geometry_volume_scale'])
            geometry_queries = self.geometry_encoder(occupancy,geometry)
        queries = self.role_queries(slots,concepts,geometry_queries)
        roles = (self.reader(queries,h,local_support) if regions is None
                 else self.reader(queries,h,local_support,regions))
        reader_roles = roles
        if self.global_queries is None:
            global_tokens = h.new_empty((0,h.shape[-1]))
        else:
            global_tokens, memory = self.global_queries[None,:,:], h[None,:,:]
            for layer in self.global_layers:
                if self.params['checkpoint_blocks'] and self.training and torch.is_grad_enabled():
                    global_tokens = checkpoint(layer,global_tokens,memory,use_reentrant=False)
                else:
                    global_tokens = layer(global_tokens,memory)
            global_tokens = global_tokens[0]
        roles = self.interactions(roles,global_tokens)
        tokens = torch.cat((global_tokens,roles.flatten(0,1)),dim=0)
        output = self.head(self.final_norm(tokens))
        auxiliary = None
        if self.params['return_auxiliary'] or self.feature_reconstructor is not None:
            auxiliary = BindFormerAuxiliary(features,coords,slots,assignments,roles,global_tokens,weights)
            auxiliary.reader_role_tokens = reader_roles
            auxiliary.evidence_weights = (local_support/(local_support.sum(0, keepdim=True)+p['slot_epsilon'])
                                          if regions is None else local_support)
            if regions is not None:
                auxiliary.region_indices, auxiliary.region_valid = regions.indices, regions.valid
            if geometry is not None:
                auxiliary.physical_centers = geometry.centers if geometry.space == 'mm' else None
                auxiliary.patch_volumes = geometry.volumes
                auxiliary.geometry_centers = geometry.centers
                auxiliary.geometry_space = geometry.space
        return self.finish(output,features,metadata,auxiliary=auxiliary,
                           **self.allocation, roles=self.params['roles'],
                           slot_iterations=self.params['slot_iterations'],
                           background_slot=self.params['background_slot'] and regions is None,
                           routing_mode=('competitive' if regions is None else
                                         'spatial_adaptive' if p['spatial_adaptive_aggregation'] else 'spatial_mean'),
                           router_value_normalization=p['router_value_normalization'],
                           router_value_projection=p['router_value_projection'],
                           support_beta=self.params['support_beta'],
                           knowledge_mode=self.params['knowledge_mode'],
                           vocabulary_scope=self.params['vocabulary_scope'] if self.concept_memory else None,
                           active_concepts=0 if self.concept_memory is None else len(self.concept_memory.embeddings),
                           concept_manifest_sha256=None if self.asset_provenance is None else self.asset_provenance.get('manifest_sha256'),
                           position_mode=self.params['position_mode'], global_depth=len(self.global_layers))
