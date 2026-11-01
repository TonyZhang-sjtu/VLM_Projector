"""Projector defaults and strict validation, without loading models or CUDA."""
from copy import deepcopy
import math
from omegaconf import OmegaConf
from src.models.projectors.bindformer.config import DEFAULTS as BIND_DEFAULTS, validate_params
from src.models.projectors.bindformer_a4.config import DEFAULTS as A4_DEFAULTS, validate_params as validate_a4
from src.models.projectors.bindformer_adirect.config import DEFAULTS as ADIRECT_DEFAULTS, validate_params as validate_adirect
from src.models.projectors.bindformer_adirect_localread.config import DEFAULTS as LOCALREAD_DEFAULTS, validate_params as validate_localread
from src.models.projectors.bindformer_adirect_detail.config import DEFAULTS as DETAIL_DEFAULTS, validate_params as validate_detail


GRID = {'num_tokens': 256, 'budget_policy': 'nearest'}
HEAD = {'head_type': 'mlp'}
QUERY = {'num_tokens': 256, 'inner_dim': None, 'num_heads': 8,
         'ffn_expansion': 4, 'dropout': 0.0, 'checkpoint_blocks': False, **HEAD}
DEFAULTS = {
    'bindformer3d': {**QUERY, **BIND_DEFAULTS},
    'bindformer_a4': {**QUERY, **A4_DEFAULTS},
    'bindformer_adirect': {**QUERY, **ADIRECT_DEFAULTS},
    'bindformer_adirect_localread': {**QUERY, **LOCALREAD_DEFAULTS},
    'bindformer_adirect_detail': {'num_tokens': 256, 'checkpoint_blocks': False, **HEAD, **DETAIL_DEFAULTS},
    'mlp2x_gelu': {},
    'avgpool3d': {**GRID, **HEAD},
    'concat3d': {'group_size': [2, 2, 2], **HEAD},
    'convpool3d': {**GRID, **HEAD, 'inner_dim': None, 'depth': 2,
                   'kernel_size': 3, 'ffn_expansion': 4},
    'mixer3d': {**GRID, **HEAD, 'inner_dim': None, 'depth': 2,
                'window_size': [2, 2, 2], 'shift_windows': True,
                'ffn_expansion': 4, 'token_expansion': 2},
    'local_packer3d': {**QUERY, **GRID, 'depth': 1},
    'attention_pool': {**QUERY},
    'perceiver_nv': {**QUERY, 'depth': 6},
    'perceiver_flamingo': {**QUERY, 'depth': 6},
    'adaptive_merge3d': {'num_tokens': 256, 'normalize_scores': True,
                         'cache_assignments': True,
                         'assignment_cache_dir': 'cache/projector_assignments', **HEAD},
    'token_select3d': {'num_tokens': 256, 'strategy': 'diversity',
                      'selection_seed': None, **HEAD},
}


def positive_int(value):
    return type(value) is int and value > 0


def resolve_projector_config(config):
    """Materialize defaults before provenance/save; leave legacy MLP unchanged."""
    name = config.name
    if config.get('budget_protocol') is None:
        config.pop('budget_protocol', None)  # Preserve legacy projector/checkpoint contracts.
    if name not in DEFAULTS:
        raise ValueError(f'Unknown projector {name!r}; choose from {list(DEFAULTS)}')
    if not positive_int(config.hidden_size) or type(config.bias) is not bool:
        raise ValueError('Projector hidden_size must be positive and bias boolean')
    supplied = OmegaConf.to_container(config.params, resolve=True)
    if not isinstance(supplied, dict):
        raise ValueError('projector.params must be a mapping')
    unknown = set(supplied) - set(DEFAULTS[name])
    if unknown:
        raise ValueError(f'Unsupported {name} parameters: {sorted(unknown)}')
    from src.models.projectors.budget import ratio_value, validate_resolved, GRID_METHODS
    ratio = ratio_value(config)
    resolved = config.get('resolved_budget')
    pending = ratio is not None and not resolved
    if ratio is None:
        if resolved:
            raise ValueError('resolved_budget requires target_compression_ratio')
        # Keep legacy projector dictionaries and provenance hashes unchanged.
        config.pop('target_compression_ratio', None)
        config.pop('resolved_budget', None)
    else:
        if name == 'mlp2x_gelu' and ratio != 1:
            raise ValueError('Uncompressed MLP supports only ratio 1')
        if pending and set(supplied) & {'num_tokens', 'group_size'}:
            raise ValueError('Do not combine target_compression_ratio with manual num_tokens/group_size')
        if name in GRID_METHODS and supplied.get('budget_policy', 'nearest') != 'nearest':
            raise ValueError('Ratio mode preserves the nearest grid budget policy')
    params = {**deepcopy(DEFAULTS[name]), **supplied}
    if pending:
        for key in ('num_tokens', 'group_size'):
            params.pop(key, None)
    for key in ('num_tokens', 'num_heads', 'depth', 'ffn_expansion',
                'kernel_size', 'token_expansion'):
        if key in params and not positive_int(params[key]):
            raise ValueError(f'{name}.{key} must be a positive integer')
    if params.get('inner_dim') is not None and not positive_int(params['inner_dim']):
        raise ValueError('inner_dim must be null or a positive integer')
    for key in ('window_size', 'group_size'):
        if key in params and (not isinstance(params[key], list) or len(params[key]) != 3
                              or not all(positive_int(v) for v in params[key])):
            raise ValueError(f'{key} must contain three positive integers')
    for key in ('shift_windows', 'normalize_scores', 'cache_assignments', 'checkpoint_blocks'):
        if key in params and type(params[key]) is not bool:
            raise ValueError(f'{key} must be boolean')
    if 'budget_policy' in params and params['budget_policy'] not in ('nearest', 'at_most', 'exact'):
        raise ValueError('budget_policy must be nearest, at_most, or exact')
    if 'head_type' in params and params['head_type'] not in ('mlp', 'linear'):
        raise ValueError('head_type must be mlp or linear')
    if 'dropout' in params and (type(params['dropout']) not in (int, float)
                               or not math.isfinite(params['dropout'])
                               or not 0 <= params['dropout'] < 1):
        raise ValueError('dropout must be finite and in [0,1)')
    if 'kernel_size' in params and params['kernel_size'] % 2 != 1:
        raise ValueError('ConvPool kernel_size must be odd')
    if 'strategy' in params and params['strategy'] not in ('random', 'diversity'):
        raise ValueError('selection strategy must be random or diversity')
    seed = params.get('selection_seed')
    if seed is not None and (type(seed) is not int or not 0 <= seed < 2**63):
        raise ValueError('selection_seed must be null or an integer in [0,2**63)')
    if 'assignment_cache_dir' in params and (not isinstance(params['assignment_cache_dir'], str)
                                            or not params['assignment_cache_dir']):
        raise ValueError('assignment_cache_dir must be a nonempty path')
    if params.get('inner_dim') and params.get('num_heads'):
        if params['inner_dim'] % params['num_heads']:
            raise ValueError('inner_dim must be divisible by num_heads')
    if name == 'bindformer3d':
        validate_params(params)
    elif name == 'bindformer_a4':
        validate_a4(params)
    elif name == 'bindformer_adirect':
        validate_adirect(params)
    elif name == 'bindformer_adirect_localread':
        validate_localread(params)
    elif name == 'bindformer_adirect_detail':
        validate_detail(params)
    config.params = OmegaConf.create(params)
    if resolved:
        validate_resolved(config)
    return config
