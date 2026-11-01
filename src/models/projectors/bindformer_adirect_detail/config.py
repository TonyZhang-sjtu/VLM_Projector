import math
from src.models.projectors.bindformer_adirect.config import allocate_adirect_tokens

DEFAULTS = {'inner_dim': 768, 'detail_fraction': 0.25}


def allocate_detail_tokens(num_tokens, detail_fraction=0.25):
    if (type(detail_fraction) not in (int, float) or not math.isfinite(detail_fraction)
            or not 0 < detail_fraction < 1):
        raise ValueError('detail_fraction must be finite and in (0,1)')
    old = allocate_adirect_tokens(num_tokens, detail_fraction)
    return {
        'allocation_version': 'bindformer_adirect_detail_nested4_v1',
        'detail_fraction': detail_fraction, 'detail_tokens': old['global_tokens'],
        'actual_detail_fraction': old['actual_global_fraction'],
        'global_tokens': 0, 'actual_global_fraction': 0.0,
        'num_regions': old['num_regions'], 'children_per_region': old['children_per_region'],
        'local_tokens': old['local_tokens'],
    }


def validate_params(params):
    allocate_detail_tokens(params.get('num_tokens', 5), params['detail_fraction'])
    if type(params['inner_dim']) is not int or params['inner_dim'] < 1:
        raise ValueError('ADirect_Detail inner_dim must be a positive integer')
