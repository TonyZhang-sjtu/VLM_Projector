"""Keep A4's mixed allocation; a zero global fraction uses exact spatial cells."""
import math
from src.models.projectors.bindformer_a4.config import DEFAULTS, allocate_a4_tokens


def allocate_adirect_tokens(num_tokens, global_fraction=0.25):
    if type(num_tokens) is not int or num_tokens < 1:
        raise ValueError('num_tokens must be a positive integer')
    if (type(global_fraction) not in (int, float) or not math.isfinite(global_fraction)
            or not 0 <= global_fraction < 1):
        raise ValueError('global_fraction must be finite and in [0,1)')
    if global_fraction == 0:
        # Nested groups of four would leave 1/3 global tokens at M=69/35.
        # A genuine all-local experiment needs M disjoint, nonempty cells.
        return {
            'allocation_version': 'bindformer_adirect_alllocal_v1',
            'global_fraction': global_fraction, 'global_tokens': 0,
            'num_regions': num_tokens, 'children_per_region': 1,
            'local_tokens': num_tokens, 'actual_global_fraction': 0.0,
        }
    allocation = allocate_a4_tokens(num_tokens, global_fraction)
    allocation['allocation_version'] = 'bindformer_adirect_nested4_v1'
    return allocation


def validate_params(params):
    allocate_adirect_tokens(params.get('num_tokens', 5), params['global_fraction'])
    for key in ('inner_dim', 'global_depth', 'position_frequencies'):
        if type(params[key]) is not int or params[key] < 1:
            raise ValueError(f'ADirect {key} must be a positive integer')
    if type(params['sequence_condition']) is not bool:
        raise ValueError('ADirect sequence_condition must be boolean')
    if params['position_mode'] not in ('none', 'grid_fourier'):
        raise ValueError('ADirect position_mode must be none or grid_fourier')
