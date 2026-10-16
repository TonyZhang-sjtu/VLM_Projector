from src.models.projectors.mlp import MLPProjector
from src.models.projectors.avgpool import AveragePool3D
from src.models.projectors.concat import ConcatAdapter3D
from src.models.projectors.convpool import ConvPool3D
from src.models.projectors.mixer import Mixer3D
from src.models.projectors.local_packer import LocalPacker3D
from src.models.projectors.resampler import QueryResampler
from src.models.projectors.adaptive_merge import AdaptiveMerge3D
from src.models.projectors.selection import TokenSelect3D
from src.models.projectors.common import StructuredProjector
from src.models.projectors.specs import resolve_projector_config, DEFAULTS
from src.models.projectors.bindformer.projector import BindFormer3D

PROJECTORS = {'bindformer3d': BindFormer3D, 'mlp2x_gelu': MLPProjector, 'avgpool3d': AveragePool3D,
              'concat3d': ConcatAdapter3D, 'convpool3d': ConvPool3D,
              'mixer3d': Mixer3D, 'local_packer3d': LocalPacker3D,
              'attention_pool': QueryResampler, 'perceiver_nv': QueryResampler,
              'perceiver_flamingo': QueryResampler, 'adaptive_merge3d': AdaptiveMerge3D,
              'token_select3d': TokenSelect3D}


def register_projector(name, factory, defaults=None):
    if name in PROJECTORS or name in DEFAULTS:
        raise ValueError(f'Projector already registered: {name}')
    PROJECTORS[name] = factory
    DEFAULTS[name] = {} if defaults is None else dict(defaults)


def build_projector(config, input_dim, output_dim):
    resolve_projector_config(config.projector)
    from src.models.projectors.budget import require_resolved
    require_resolved(config.projector)
    kwargs = {'project_root': config.project_root} if config.projector.name == 'bindformer3d' else {}
    projector = PROJECTORS[config.projector.name](input_dim, output_dim, config.projector, **kwargs)
    if isinstance(projector, StructuredProjector):
        projector.stage = config.training.stage
        projector.seed = config.seed
        projector.project_root = config.project_root
    return projector
