import math
from src.models.projectors.bindformer_adirect.config import DEFAULTS as ADIRECT_DEFAULTS
from src.models.projectors.bindformer_adirect.config import validate_params as validate_adirect

DEFAULTS = {**ADIRECT_DEFAULTS, 'local_residual_scale_init': 0.1}


def validate_params(params):
    validate_adirect(params)
    scale = params['local_residual_scale_init']
    if type(scale) not in (int, float) or not math.isfinite(scale):
        raise ValueError('local_residual_scale_init must be finite')
