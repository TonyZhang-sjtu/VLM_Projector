"""BindFormer-specific validation, including explicit future extension guards."""
import math
from .budget import ROLES, allocate_tokens, role_counts

# Pruned only when equal to these defaults in the provenance contract. This
# permits old disabled-extension runs to resume without weakening other checks.
EXTENSION_DEFAULTS = {
    'role_query_counts': {r: 1 for r in ROLES},
    'geometry_space': 'grid', 'geometry_hidden_dim': 128,
    'geometry_coordinate_scale': 100.0, 'geometry_volume_scale': 1000.0,
    'within_slot_sa_depth': 1, 'local_global_depth': 1,
    'local_global_gate_init': 0.1,
    'feature_inner_dim': 256, 'feature_num_heads': 4,
    'feature_samples': 512, 'feature_query_chunk_size': 128,
    'feature_target': 'frozen_base_cache',
    'feature_target_cache': 'cache/bindformer_teacher',
}

DEFAULTS = {
    'global_fraction': 0.25, 'roles': list(ROLES), 'inner_dim': 768,
    'slot_iterations': 3, 'background_slot': True, 'slot_epsilon': 1e-6,
    'position_mode': 'grid_fourier', 'position_frequencies': 8,
    'sequence_condition': True,
    'knowledge_mode': 'name_definition',
    'concept_bank': 'assets/knowledge/bindformer_brain_candidates_v2',
    'vocabulary_scope': 'brats_train', 'support_beta': [1.0]*4,
    'global_depth': 1, 'reader_chunk_size': 128, 'return_auxiliary': False,
    'geometry_enabled': False, 'within_slot_sa': False, 'local_reads_global': False,
    'feature_loss_weight': 0.0, 'text_loss_weight': 0.0, 'tuple_loss_weight': 0.0,
    'binding_loss_weight': 0.0, 'concept_loss_weight': 0.0,
    **EXTENSION_DEFAULTS,
}


def validate_params(params):
    if params['roles'] != list(ROLES):
        raise ValueError('BindFormer v1 requires the four canonical roles in fixed order')
    fraction = params['global_fraction']
    counts = role_counts(params['role_query_counts'])
    allocation = allocate_tokens(params.get('num_tokens', sum(counts)+1), fraction,
                                 len(ROLES), counts)
    for key in ('slot_iterations', 'position_frequencies', 'global_depth', 'reader_chunk_size',
                'geometry_hidden_dim', 'within_slot_sa_depth', 'local_global_depth',
                'feature_inner_dim', 'feature_num_heads', 'feature_samples',
                'feature_query_chunk_size'):
        if type(params[key]) is not int or params[key] < 1:
            raise ValueError(f'{key} must be a positive integer')
    if type(params['inner_dim']) is not int or params['inner_dim'] < 1:
        raise ValueError('BindFormer inner_dim must be a positive integer')
    for key in ('background_slot', 'sequence_condition', 'return_auxiliary',
                'geometry_enabled', 'within_slot_sa', 'local_reads_global'):
        if type(params[key]) is not bool:
            raise ValueError(f'{key} must be boolean')
    eps = params['slot_epsilon']
    if type(eps) not in (int, float) or not math.isfinite(eps) or not 0 < eps < 1:
        raise ValueError('slot_epsilon must be finite and in (0,1)')
    if params['position_mode'] not in ('none', 'grid_fourier'):
        raise ValueError('position_mode currently supports none/grid_fourier; physical PE is not implemented')
    if params['knowledge_mode'] not in ('none', 'name_definition'):
        raise ValueError('knowledge_mode currently supports none/name_definition')
    if params['vocabulary_scope'] not in ('brats_train', 'all_train'):
        raise ValueError('Unknown vocabulary_scope')
    if params['knowledge_mode'] != 'none' and (
            not isinstance(params['concept_bank'], str) or not params['concept_bank']):
        raise ValueError('Knowledge requires a nonempty concept_bank path')
    beta = params['support_beta']
    if type(beta) in (int, float):
        beta = [beta] * len(ROLES)
    if (not isinstance(beta, list) or len(beta) != len(ROLES)
            or any(type(b) not in (int, float) or not math.isfinite(b) or b < 0 for b in beta)):
        raise ValueError('support_beta must be nonnegative finite scalar or four role values')
    params['support_beta'] = [float(b) for b in beta]
    if params['geometry_space'] not in ('grid', 'voxel', 'mm'):
        raise ValueError('geometry_space must be grid, voxel, or mm')
    for key in ('geometry_coordinate_scale', 'geometry_volume_scale'):
        value = params[key]
        if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
            raise ValueError(f'{key} must be positive and finite')
    gate = params['local_global_gate_init']
    if type(gate) not in (int, float) or not math.isfinite(gate) or not 0 < gate < 1:
        raise ValueError('local_global_gate_init must be in (0,1)')
    if 'num_tokens' in params and params['local_reads_global'] and not allocation['global_tokens']:
        raise ValueError('local_reads_global requires global tokens')
    weight = params['feature_loss_weight']
    if type(weight) not in (int, float) or not math.isfinite(weight) or weight < 0:
        raise ValueError('feature_loss_weight must be nonnegative and finite')
    if params['feature_inner_dim'] % params['feature_num_heads']:
        raise ValueError('feature_inner_dim must be divisible by feature_num_heads')
    if params['feature_target'] not in ('frozen_base_cache', 'frozen_input'):
        raise ValueError('feature_target must be frozen_base_cache or frozen_input')
    if not isinstance(params['feature_target_cache'], str) or not params['feature_target_cache']:
        raise ValueError('feature_target_cache must be a nonempty path')
    for key in ('text_loss_weight', 'tuple_loss_weight',
                'binding_loss_weight', 'concept_loss_weight'):
        value = params[key]
        if type(value) not in (int, float) or value != 0:
            raise ValueError(f'{key} must be zero: this objective is not implemented')
