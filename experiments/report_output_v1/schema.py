"""Frozen, train-derived field inventory and explicit experiment settings."""
from experiments.report_facts_v1.schema import FIELDS as V1_FIELDS

VERSION = 'report_output_v1'
FIELDS = {key: V1_FIELDS[key] for key in ('edema', 'laterality', 'site', 'signal')}
DEFAULTS = dict(
    enabled=False, version=VERSION, path='assets/knowledge/report_output_v1',
    fields=list(FIELDS), span_weight=0.0, group_weight=0.0, margin=0.2,
    warmup_steps=77, group_every_steps=1,
    validation_enabled=True, validation_every_epochs=1, validation_limit=0,
    save_best_facts=True,
)

