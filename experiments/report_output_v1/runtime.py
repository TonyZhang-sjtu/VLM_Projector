"""Process-local installation only; old shells never import/install this runtime."""
from copy import deepcopy
import math
from pathlib import Path

from .schema import DEFAULTS, FIELDS, VERSION

_installed = False


def validate_settings(params):
    p = params['report_output']
    if set(p) != set(DEFAULTS) or p['version'] != VERSION:
        raise ValueError('Missing/unknown report_output settings/version')
    for key in ('enabled', 'validation_enabled', 'save_best_facts'):
        if type(p[key]) is not bool:
            raise ValueError(f'{key} must be boolean')
    for key in ('span_weight', 'group_weight', 'margin'):
        if type(p[key]) not in (int, float) or not math.isfinite(p[key]) or p[key] < 0:
            raise ValueError(f'{key} must be finite and nonnegative')
    for key, minimum in (('warmup_steps', 0), ('group_every_steps', 1),
                         ('validation_every_epochs', 1), ('validation_limit', 0)):
        if type(p[key]) is not int or p[key] < minimum:
            raise ValueError(f'Invalid {key}')
    if not isinstance(p['path'], str) or not p['path']:
        raise ValueError('Output supervision path is required')
    if not isinstance(p['fields'], list) or not p['fields'] or len(set(p['fields'])) != len(p['fields']) or set(p['fields'])-set(FIELDS):
        raise ValueError('Unknown/empty/duplicate report-output fields')
    if p['enabled']:
        if not params['fixed_spatial_regions'] or not params['spatial_mean_aggregation']:
            raise ValueError('This experiment requires the Baseline A fixed-region mean architecture')
        if params['feature_loss_weight']:
            raise ValueError('Combining reconstruction and report-output supervision is not validated')


def install():
    global _installed
    if _installed:
        return
    from src.models.projectors import specs, registry
    from src.models.projectors.bindformer.projector import BindFormer3D
    from src.config import ROOT, file_hash
    original_validate = specs.validate_params

    def validate(params):
        supplied = params['report_output']
        if not isinstance(supplied, dict) or set(supplied)-set(DEFAULTS):
            raise ValueError('Unknown report_output settings')
        params['report_output'] = {**deepcopy(DEFAULTS), **supplied}
        original_validate(params)
        validate_settings(params)

    specs.DEFAULTS['bindformer3d']['report_output'] = deepcopy(DEFAULTS)
    specs.validate_params = validate

    class ReportOutputProjector(BindFormer3D):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            if not self.params['report_output']['enabled']:
                return
            path = Path(self.params['report_output']['path'])
            if not path.is_absolute():
                path = Path(kwargs.get('project_root') or ROOT)/path
            self.asset_provenance = {**(self.asset_provenance or {}), 'report_output': dict(
                source_files={f.name: file_hash(f) for f in sorted(Path(__file__).parent.glob('*.py'))},
                assets={name: file_hash(path/name) for name in ('metadata.json', 'train.jsonl', 'val.jsonl', 'schema.json')})}

    registry.PROJECTORS['bindformer3d'] = ReportOutputProjector
    from .model import ReportOutputModel
    import src.models.multimodal as multimodal
    multimodal.MultimodalModel = ReportOutputModel
    import src.training.trainer as trainer
    trainer.MultimodalModel = ReportOutputModel
    original_validation, original_trainer = trainer.validate, trainer.Trainer

    def validate_epoch(model, dataset, *args, **kwargs):
        metrics = original_validation(model, dataset, *args, **kwargs)
        if getattr(model, '_output_training_owner', False) and model.output_settings['validation_enabled']:
            epoch = model._output_validation_calls + 1
            model._output_validation_calls = epoch
            if epoch % model.output_settings['validation_every_epochs'] == 0:
                from .evaluation import run_validation
                metrics['report_facts_generation'] = run_validation(model, dataset)
        return metrics

    class ReportOutputTrainer(original_trainer):
        def __init__(self, config):
            super().__init__(config)
            if self.model.uses_auxiliary_objectives:
                self.model._output_training_owner = True
                self.model._output_validation_calls = self.state['epoch']

        def save(self, name):
            metrics = getattr(self.model, '_pending_output_validation', None)
            if metrics and metrics['step'] == self.state['global_step'] and self.state.get('facts_checked_step') != metrics['step']:
                value = metrics['macro_field_balanced_accuracy']
                previous = self.state.get('best_fact_score', float('-inf'))
                improved = value is not None and value > previous
                self.state['facts_checked_step'] = metrics['step']
                if improved:
                    self.state.update(best_fact_score=value, best_fact_step=metrics['step'],
                                      best_fact_metric=metrics['selection_metric'],
                                      best_fact_scope=metrics['validation_scope'])
                    if self.model.output_settings['save_best_facts']:
                        super().save('best_facts.pt')
            super().save(name)

    trainer.validate = validate_epoch
    trainer.Trainer = ReportOutputTrainer
    import sys
    for name in ('src.cli.evaluate',):
        if name in sys.modules:
            sys.modules[name].MultimodalModel = ReportOutputModel
    _installed = True
