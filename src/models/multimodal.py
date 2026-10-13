import torch
from torch import nn
from src.models.encoders.registry import build_encoder
from src.models.projectors.registry import build_projector
from src.models.decoders.llama import LlamaDecoder


class MultimodalModel(nn.Module):
    def __init__(self,config,decoder=None):
        super().__init__()
        self.config=config
        from src.models.projectors.budget import require_resolved
        require_resolved(config.projector)
        from src.training.reproducibility import configure_reproducibility
        configure_reproducibility(config)
        self.encoder=build_encoder(config)
        self.decoder=decoder if decoder is not None else LlamaDecoder(config)
        self.projector=build_projector(config,self.encoder.feature_dim,self.decoder.hidden_size)

    def project_volumes(self,volumes):
        encoded=self.encoder(volumes)
        if not (len(volumes)==len(encoded.features)==len(encoded.coords)==len(encoded.metadata)):
            raise ValueError('Encoder output batch mismatch')
        required=getattr(self.config.training,'required_visual_tokens',None)
        return [self.projector(f,c,m).validate(self.decoder.hidden_size,required) for f,c,m in
                zip(encoded.features,encoded.coords,encoded.metadata)]

    def project(self,volumes):
        return [v.embeddings for v in self.project_volumes(volumes)]

    def forward(self,items):
        projected=self.project_volumes([x['volume'] for x in items])
        visual=[v.embeddings for v in projected]
        batch=self.decoder.prepare([x['sample'] for x in items],visual,training=True)
        loss_sum,tokens=self.decoder.loss(batch)
        return {'loss_sum':loss_sum,'answer_tokens':tokens,'lengths':batch.lengths,
                'encoder_token_counts':[v.input_tokens for v in projected],
                'projector_token_counts':[v.output_tokens for v in projected]}

    @torch.no_grad()
    def generate(self,items,control=None,max_new_tokens=None):
        was_training=self.training;self.eval()
        try:
            if control=='text':
                weight=self.decoder.lm.get_input_embeddings().weight
                visual=[weight.new_empty((0,self.decoder.hidden_size)) for _ in items]
                projected=None
            else:
                projected=self.project_volumes([x['volume'] for x in items])
                visual=[v.embeddings for v in projected]
            if control=='zero':visual=[torch.zeros_like(x) for x in visual]
            elif control=='shuffle':
                if len(visual)<2:raise ValueError('Shuffle control requires at least two samples')
                visual=visual[1:]+visual[:1]
                projected=projected[1:]+projected[:1]
            elif control not in (None,'text'):raise ValueError(f'Unknown control {control}')
            results=self.decoder.generate([x['sample'] for x in items],visual,max_new_tokens,allow_empty_visual=control=='text')
            for i,result in enumerate(results):
                volume=None if projected is None else projected[i]
                info={} if volume is None else volume.metadata.get('projector',{})
                actual=len(visual[i])
                result.update(encoder_token_count=0 if volume is None else volume.input_tokens,
                              projector_token_count=actual,visual_token_count=actual,
                              requested_visual_tokens=info.get('requested_tokens'),
                              projector_name=self.config.projector.name,projector_metadata=info,
                              total_prefix_token_count=result['prompt_token_count']+actual)
            return results
        finally:self.train(was_training)
