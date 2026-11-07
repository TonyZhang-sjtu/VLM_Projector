"""Exact output budgets, including complete coverage by a partial final parent."""
from src.models.projectors.bindformer_adirect.config import allocate_adirect_tokens


A1 = 'anchorread3d_a1_no_localread'
A2 = 'anchorread3d_a2_no_mean_bypass'
A3 = 'anchorread3d_a3_global_only'
A4 = 'anchorread3d_a4_local_only'
METHODS = (A1, A2, A3, A4)
FRACTIONS = {A1: 0.25, A2: 0.25, A3: 1.0, A4: 0.0}


def child_counts(num_tokens):
    if type(num_tokens) is not int or num_tokens < 1:
        raise ValueError('Ablation num_tokens must be a positive integer')
    parents = (num_tokens + 3) // 4
    return (4,) * (parents - 1) + (num_tokens - 4 * (parents - 1),)


def allocate_ablation(name, num_tokens, global_fraction=None):
    if name not in METHODS:
        raise ValueError(f'Unknown AnchorRead3D ablation: {name}')
    child_counts(num_tokens)  # Validate even the global-only boundary.
    expected = FRACTIONS[name]
    fraction = expected if global_fraction is None else global_fraction
    if type(fraction) not in (int, float) or fraction != expected:
        raise ValueError(f'{name} requires global_fraction={expected}')
    if name in (A1, A2):
        allocation = allocate_adirect_tokens(num_tokens, fraction)
        allocation['children_per_parent'] = [4] * allocation['num_regions']
    else:
        counts = child_counts(num_tokens) if name == A4 else ()
        allocation = dict(
            allocation_version=('anchorread3d_nested4_tail_v1' if name == A4
                                else 'anchorread3d_global_only_v1'),
            global_fraction=fraction, actual_global_fraction=fraction,
            global_tokens=num_tokens if name == A3 else 0,
            local_tokens=num_tokens if name == A4 else 0,
            num_regions=len(counts), children_per_parent=list(counts),
            children_per_region=(4 if counts and counts[-1] == 4 else None),
        )
    allocation['ablation_id'] = name.split('_')[1].upper()
    return allocation
