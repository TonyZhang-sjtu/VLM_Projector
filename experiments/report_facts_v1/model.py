"""Opt-in model subclass; all inference and validation retain the base path."""
import torch
from src.models.multimodal import MultimodalModel as BaseModel
from .objectives import FactDecoderView, pairing_loss
from .store import preflight


class ReportFactsModel(BaseModel):
    def __init__(self,config,decoder=None):
        # Read-only preflight before allocating pretrained models. Inference
        # never looks up individual reference facts or invokes the objectives.
        store=preflight(config)
        super().__init__(config,decoder)
        self.fact_settings=self.projector.params.get('report_facts',{})
        self.uses_auxiliary_objectives=bool(self.fact_settings.get('enabled',False))
        self.supervision_step=0
        self._fact_store=store
        if self.uses_auxiliary_objectives:
            if config.training.distributed!='none':
                raise ValueError('report_facts_v1 currently validates single-GPU runs only')
            if self.projector.feature_reconstructor is not None:
                raise ValueError('Combining feature reconstruction with report facts is not yet validated')

    def set_supervision_step(self,step):
        self.supervision_step=int(step)

    def forward(self,items):
        if not self.training or not self.uses_auxiliary_objectives:
            return super().forward(items)
        # The same projected Z is reused by report CE and both auxiliary tasks.
        projected=self.project_volumes([x['volume'] for x in items],[x['sample'].sequence for x in items])
        batch=self.decoder.prepare([x['sample'] for x in items],[v.embeddings for v in projected],training=True)
        report_loss,tokens=self.decoder.loss(batch)
        result=dict(loss_sum=report_loss,answer_tokens=tokens,lengths=batch.lengths,
                    encoder_token_counts=[v.input_tokens for v in projected],
                    projector_token_counts=[v.output_tokens for v in projected])
        p=self.fact_settings
        ramp=min(1.,(self.supervision_step+1)/max(1,p['warmup_steps']))
        total=report_loss*0
        metrics={key:0. for key in ('fact_loss','fact_labels','fact_correct','pair_loss','pair_rank_loss',
                                  'pair_positive_ce','pair_count','pair_correct','pair_score_gap','weighted_loss')}
        for item,volume in zip(items,projected):
            sample=item['sample'];row,pairs=self._fact_store.get(sample);z=volume.embeddings
            if p['fact_weight']>0:
                loss,count,correct=self.projector.fact_heads.objective(z,self.projector.allocation,row['facts'])
                total=total+ramp*p['fact_weight']*loss
                metrics['fact_loss']+=float(loss.detach());metrics['fact_labels']+=count
                metrics['fact_correct']+=float(correct)
            if p['pair_weight']>0 and self.supervision_step%p['pair_every_steps']==0:
                for pair in pairs:
                    if pair['field'] not in p['fields']:continue
                    view=FactDecoderView(self.decoder)
                    positive=view.score(z,sample,pair['field'],pair['positive'])
                    negative=view.score(z,sample,pair['field'],pair['negative'])
                    loss,rank,ce=pairing_loss(positive,negative,p['margin'],p['positive_ce_weight'])
                    total=total+ramp*p['pair_weight']*loss
                    metrics['pair_loss']+=float(loss.detach());metrics['pair_rank_loss']+=float(rank.detach())
                    metrics['pair_positive_ce']+=float(ce.detach());metrics['pair_count']+=1
                    metrics['pair_correct']+=float((positive>negative).detach())
                    metrics['pair_score_gap']+=float((positive-negative).detach())
        metrics['weighted_loss']=float(total.detach())
        result.update(auxiliary_loss_sum=total,auxiliary_metrics=metrics)
        return result
