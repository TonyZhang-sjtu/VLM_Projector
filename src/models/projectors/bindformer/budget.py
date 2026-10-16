"""Pure integer budget allocation; safe to run before loading model weights."""
from fractions import Fraction
import math

ROLES = ('finding', 'location_laterality', 'signal', 'morphology_extent')
ALLOCATION_VERSION = 'bindformer_nearest_fraction_v1'


def role_counts(mapping=None):
    """Named counts avoid silently swapping the signal and location budgets."""
    if mapping is None:
        return [1] * len(ROLES)
    if set(mapping) != set(ROLES):
        raise ValueError('role_query_counts must specify exactly the four canonical roles')
    values = [mapping[r] for r in ROLES]
    if any(type(n) is not int or n < 1 for n in values):
        raise ValueError('Each role query count must be a positive integer')
    return values


def allocate_tokens(num_tokens, global_fraction=0.25, num_roles=4, role_query_counts=None):
    if type(num_tokens) is not int or type(num_roles) is not int or num_roles < 1:
        raise ValueError('BindFormer token budget and role count must be integers')
    if (type(global_fraction) not in (int, float)
            or not math.isfinite(global_fraction) or not 0 <= global_fraction < 1):
        raise ValueError('global_fraction must be finite and in [0,1)')
    counts = [1]*num_roles if role_query_counts is None else list(role_query_counts)
    if len(counts) != num_roles or any(type(n) is not int or n < 1 for n in counts):
        raise ValueError('Each role query count must be a positive integer')
    bundle = sum(counts)
    minimum_global = int(global_fraction > 0)
    maximum_slots = (num_tokens - minimum_global) // bundle
    if maximum_slots < 1:
        raise ValueError('BindFormer budget needs one complete role bundle and requested global branch')
    fraction = Fraction(str(global_fraction))
    ideal = (1 - fraction) * num_tokens / bundle
    candidates = {max(1, min(maximum_slots, math.floor(ideal))),
                  max(1, min(maximum_slots, math.ceil(ideal)))}
    # Equal distance: fewer slots, hence more global tokens.
    slots = min(candidates, key=lambda k: (abs(num_tokens-bundle*k-fraction*num_tokens), k))
    global_tokens = num_tokens - bundle * slots
    result = {'allocation_version': ALLOCATION_VERSION, 'global_fraction': global_fraction,
            'global_tokens': global_tokens, 'num_slots': slots, 'num_roles': num_roles,
            'local_tokens': slots*bundle, 'actual_global_fraction': global_tokens/num_tokens}
    # Preserve the exact saved v1 budget contract for existing runs/checkpoints.
    if counts != [1]*num_roles:
        result.update(allocation_version='bindformer_nearest_fraction_v2',
                      role_query_counts=counts, tokens_per_slot=bundle)
    return result
