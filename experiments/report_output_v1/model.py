"""Opt-in report objectives on the unchanged Baseline A visual architecture."""
from src.models.multimodal import MultimodalModel as BaseModel
from .store import preflight
from .objectives import streamed_objective


class ReportOutputModel(BaseModel):
    def __init__(self, config, decoder=None):
        store = preflight(config)
        super().__init__(config, decoder)
        self.output_settings = dict(self.projector.params.get('report_output', {}))
        self.uses_auxiliary_objectives = bool(self.output_settings.get('enabled', False))
        self.supervision_step = 0
        self.output_store = store

    def set_supervision_step(self, step):
        self.supervision_step = int(step)

    def forward(self, items):
        if not self.training or not self.uses_auxiliary_objectives:
            return super().forward(items)
        p = self.output_settings
        if p['span_weight'] == 0 and p['group_weight'] == 0:
            return super().forward(items)
        projected = self.project_volumes([i['volume'] for i in items], [i['sample'].sequence for i in items])
        reports, auxiliaries, lengths, metrics = [], [], [], {}
        tokens = 0
        ramp = min(1., (self.supervision_step+1)/max(1, p['warmup_steps']))
        for item, volume in zip(items, projected):
            sample = item['sample']
            row = self.output_store.get(sample)
            report, auxiliary, local = streamed_objective(self.decoder, sample, volume.embeddings, row, p,
                ramp=ramp, pair_active=self.supervision_step % p['group_every_steps'] == 0)
            reports.append(report); auxiliaries.append(auxiliary)
            tokens += local.pop('answer_tokens'); lengths.append(local.pop('length'))
            for key, value in local.items():
                metrics[key] = metrics.get(key, 0.) + value
        return dict(loss_sum=sum(reports), answer_tokens=tokens, lengths=lengths,
                    encoder_token_counts=[v.input_tokens for v in projected],
                    projector_token_counts=[v.output_tokens for v in projected],
                    auxiliary_loss_sum=sum(auxiliaries), auxiliary_metrics=metrics)
