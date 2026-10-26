"""Process-local extension. Legacy Python/shell entry points never install it."""
import math
from pathlib import Path
from .schema import DEFAULT_FIELDS, FIELDS, VERSION

DEFAULTS=dict(enabled=True,version=VERSION,path='assets/knowledge/report_facts_v1',
              fact_weight=0.,pair_weight=0.1,positive_ce_weight=0.25,margin=0.2,
              fields=DEFAULT_FIELDS,warmup_steps=77,pair_every_steps=1)
_installed=False


def validate_settings(params):
    p=params['report_facts']
    if set(p)!=set(DEFAULTS):raise ValueError('Missing/unknown report_facts settings')
    if type(p['enabled']) is not bool or p['version']!=VERSION:raise ValueError('Invalid report_facts switch/version')
    if not isinstance(p['path'],str) or not p['path']:raise ValueError('Facts path is required')
    for key in ('fact_weight','pair_weight','positive_ce_weight','margin'):
        value=p[key]
        if type(value) not in (int,float) or not math.isfinite(value) or value<0:
            raise ValueError(f'{key} must be finite and nonnegative')
    for key,minimum in (('warmup_steps',0),('pair_every_steps',1)):
        if type(p[key]) is not int or p[key]<minimum:raise ValueError('Invalid '+key)
    if not isinstance(p['fields'],list) or not p['fields'] or len(set(p['fields']))!=len(p['fields']) or set(p['fields'])-set(FIELDS):
        raise ValueError('Unknown/empty/duplicate facts fields')
    if p['enabled']:
        if p['fact_weight']+p['pair_weight']<=0:raise ValueError('Enabled experiment needs a nonzero objective')
        if not params['fixed_spatial_regions'] or sum(params['role_query_counts'].values())!=4:
            raise ValueError('Report facts require fixed regions and four single-query roles')
        if params['feature_loss_weight']>0:raise ValueError('Feature reconstruction not supported in this experiment')
        if params['roles']!=['finding','location_laterality','signal','morphology_extent']:
            raise ValueError('Fact head role order mismatch')


def install():
    global _installed
    if _installed:return
    from copy import deepcopy
    import torch
    from src.models.projectors import specs,registry
    from src.models.projectors.bindformer.projector import BindFormer3D
    from src.config import file_hash
    from .objectives import FactHeads
    old_validate=specs.validate_params
    def validate(params):
        supplied=params['report_facts']
        if not isinstance(supplied,dict) or set(supplied)-set(DEFAULTS):
            raise ValueError('Unknown report_facts settings')
        params['report_facts']={**deepcopy(DEFAULTS),**supplied}
        old_validate(params)
        validate_settings(params)
    specs.DEFAULTS['bindformer3d']['report_facts']=deepcopy(DEFAULTS)
    specs.validate_params=validate

    class ReportFactsProjector(BindFormer3D):
        def __init__(self,*args,**kwargs):
            super().__init__(*args,**kwargs)
            p=self.params['report_facts']
            if not p['enabled']:return
            if p['fact_weight']>0:
                # Auxiliary head initialization never changes the base-model RNG
                # stream, preserving common core initialization across E1/E2/E3.
                with torch.random.fork_rng(devices=[]):
                    # torch.manual_seed also resets every CUDA generator, even
                    # inside a CPU-only fork_rng context. Seed the CPU generator
                    # directly so vision-LoRA dropout keeps the original stream.
                    torch.default_generator.manual_seed(1042)
                    self.fact_heads=FactHeads(self.output_dim,p['fields'])
            path=Path(p['path'])
            if not path.is_absolute():path=Path(kwargs.get('project_root') or self.project_root)/path
            self.asset_provenance={**(self.asset_provenance or {}), 'report_facts':dict(
                source_files={f.name:file_hash(f) for f in sorted(Path(__file__).parent.glob('*.py'))},
                assets={name:file_hash(path/name) for name in ('metadata.json','train.jsonl','pairs.jsonl','schema.json')})}
    registry.PROJECTORS['bindformer3d']=ReportFactsProjector
    from .model import ReportFactsModel
    import src.models.multimodal as multimodal
    multimodal.MultimodalModel=ReportFactsModel
    # Also support explicit installation after importing Trainer in a test/tool.
    import sys
    for name in ('src.training.trainer','src.cli.evaluate'):
        if name in sys.modules:sys.modules[name].MultimodalModel=ReportFactsModel
    _installed=True
