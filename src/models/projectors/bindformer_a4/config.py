"""A4 has no slot, semantic-role, knowledge-bank or auxiliary-loss settings."""
from src.models.projectors.bindformer.budget import allocate_tokens


DEFAULTS = {
    'inner_dim': 768,
    'global_fraction': 0.25,
    'global_depth': 1,
    'position_mode': 'grid_fourier',
    'position_frequencies': 8,
    'sequence_condition': True,
}
CHILDREN_PER_REGION = 4


def allocate_a4_tokens(num_tokens, global_fraction=0.25):
    # Reuse A's integer rounding only, without semantic-role modules/settings.
    old = allocate_tokens(num_tokens, global_fraction, CHILDREN_PER_REGION)
    return {
        'allocation_version': 'bindformer_a4_nested4_v1',
        'global_fraction': global_fraction,
        'global_tokens': old['global_tokens'],
        'num_regions': old['num_slots'],
        'children_per_region': CHILDREN_PER_REGION,
        'local_tokens': old['local_tokens'],
        'actual_global_fraction': old['actual_global_fraction'],
    }


def validate_params(params):
    allocate_a4_tokens(params.get('num_tokens', 5), params['global_fraction'])
    for key in ('inner_dim', 'global_depth', 'position_frequencies'):
        if type(params[key]) is not int or params[key] < 1:
            raise ValueError(f'A4 {key} must be a positive integer')
    if type(params['sequence_condition']) is not bool:
        raise ValueError('A4 sequence_condition must be boolean')
    if params['position_mode'] not in ('none', 'grid_fourier'):
        raise ValueError('A4 position_mode must be none or grid_fourier')
