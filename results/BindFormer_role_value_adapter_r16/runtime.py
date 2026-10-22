"""Opt-in process-local registry extension, used by smoke and full train/eval."""
import math
from pathlib import Path
from reader_adapter import RoleValueReader

DEFAULTS=dict(role_value_reader=True,role_value_adapter_enabled=True,
              role_value_adapter_rank=16,role_value_adapter_alpha=16.,
              role_value_implementation='isolated_role_value_v1')
_installed=False

def install():
    global _installed
    if _installed:return
    from src.models.projectors import specs,registry
    from src.models.projectors.bindformer.projector import BindFormer3D
    old_validate=specs.validate_params
    def validate(params):
        old_validate(params)
        for key in ('role_value_reader','role_value_adapter_enabled'):
            if type(params[key]) is not bool:raise ValueError(key+' must be boolean')
        rank=params['role_value_adapter_rank'];alpha=params['role_value_adapter_alpha']
        if type(rank) is not int or not 1<=rank<=params['inner_dim']:raise ValueError('Invalid value adapter rank')
        if type(alpha) not in (int,float) or not math.isfinite(alpha) or alpha<=0:raise ValueError('Invalid value adapter alpha')
        if params['role_value_implementation']!='isolated_role_value_v1':raise ValueError('Unknown value implementation')
        if not params['role_value_reader'] or not params['fixed_spatial_regions']:
            raise ValueError('Isolated runner requires its Reader and fixed regions')
        if sum(params['role_query_counts'].values())!=4:raise ValueError('Exactly four roles required')
    specs.DEFAULTS['bindformer3d'].update(DEFAULTS)
    specs.validate_params=validate
    class ExperimentalBindFormer(BindFormer3D):
        def __init__(self,*args,**kwargs):
            super().__init__(*args,**kwargs)
            p=self.params
            self.reader=RoleValueReader(self.reader,p['role_value_adapter_enabled'],
                                        p['role_value_adapter_rank'],p['role_value_adapter_alpha'])
            from src.config import file_hash
            self.asset_provenance={**(self.asset_provenance or {}),
                'role_value_experiment':{name:file_hash(Path(__file__).parent/name)
                    for name in ('reader_adapter.py','runtime.py','entry.py')}}
    registry.PROJECTORS['bindformer3d']=ExperimentalBindFormer
    _installed=True
