"""Ablation-specific constraints, independent of old all-local allocation."""
from src.models.projectors.bindformer_adirect.config import DEFAULTS as DIRECT_DEFAULTS
from src.models.projectors.bindformer_adirect.config import validate_params as validate_direct
from src.models.projectors.bindformer_adirect_localread.config import validate_params as validate_read
from .budget import A1, A2, A3, A4, FRACTIONS, allocate_ablation


DEFAULTS = {
    name: {**DIRECT_DEFAULTS, 'global_fraction': fraction,
           **({'local_residual_scale_init': 0.1} if name in (A2, A4) else {})}
    for name, fraction in FRACTIONS.items()
}


def validate_params(name, params):
    allocate_ablation(name, params.get('num_tokens', 5), params['global_fraction'])
    # Validate common dimensions/geometry using the existing validator, while
    # keeping A3's fraction=1 boundary out of the legacy allocator.
    if name == A3:
        validate_direct({**params, 'global_fraction': 0.0})
    elif name in (A2, A4):
        validate_read(params)
    else:
        validate_direct(params)
