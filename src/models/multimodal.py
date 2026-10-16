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
        self.feature_target_store=None
        if getattr(self.projector,'feature_reconstructor',None) is not None:
            if self.projector.params['feature_target']=='frozen_base_cache':
                from src.config import path_of
                from src.training.feature_targets import FrozenFeatureTargets
                self.feature_target_store=FrozenFeatureTargets(
                    path_of(config,self.projector.params['feature_target_cache']),
                    self.encoder.fingerprint,self.encoder.feature_dim)

    def project_volumes(self,volumes,sequences=None):
        if sequences is not None and len(sequences)!=len(volumes):
            raise ValueError('Projector sequence/volume batch mismatch')
        encoded=self.encoder(volumes)
        if not (len(volumes)==len(encoded.features)==len(encoded.coords)==len(encoded.metadata)):
            raise ValueError('Encoder output batch mismatch')
        required=getattr(self.config.training,'required_visual_tokens',None)
        outputs=[]
        for i,(f,c,m) in enumerate(zip(encoded.features,encoded.coords,encoded.metadata)):
            context=m
            if getattr(self.projector,'requires_sequence',False) and sequences is not None:
                # Never mutate cached encoder metadata or forward patient report fields.
                context={**m,'sequence':sequences[i]}
            outputs.append(self.projector(f,c,context).validate(self.decoder.hidden_size,required))
        return outputs

    def project(self,volumes,sequences=None):
        return [v.embeddings for v in self.project_volumes(volumes,sequences)]

    def forward(self,items):
        projected=self.project_volumes([x['volume'] for x in items],[x['sample'].sequence for x in items])
        visual=[v.embeddings for v in projected]
        batch=self.decoder.prepare([x['sample'] for x in items],visual,training=True)
        loss_sum,tokens=self.decoder.loss(batch)
        result={'loss_sum':loss_sum,'answer_tokens':tokens,'lengths':batch.lengths,
                'encoder_token_counts':[v.input_tokens for v in projected],
                'projector_token_counts':[v.output_tokens for v in projected]}
        head=getattr(self.projector,'feature_reconstructor',None)
        if head is not None:
            from src.models.projectors.bindformer.reconstruction import sampling_seed
            losses=[];cosines=[];patches=0
            for item,volume in zip(items,projected):
                aux=volume.auxiliary
                if self.projector.params['feature_target']=='frozen_input':
                    if self.config.training.stage!='a' or aux.source_features.requires_grad:
                        raise ValueError('Current-input targets must come from a frozen Stage A encoder')
                    target=aux.source_features.detach()
                else:
                    target=self.feature_target_store.load(aux.coords,volume.metadata)
                seed=sampling_seed(self.config.seed,item['sample'].id,
                                   item.get('feature_sampling_step',0) if self.training else 0)
                value=head.objective(volume.embeddings,aux.coords,volume.metadata,
                                     item['sample'].sequence,target,seed)
                losses.append(value['loss']);cosines.append(value['cosine']);patches+=value['patches']
            result.update(feature_loss_sum=torch.stack(losses).sum(),feature_samples=len(losses),
                          feature_cosine_sum=torch.stack(cosines).sum(),feature_patch_count=patches)
        return result

    @torch.no_grad()
    def generate(self,items,control=None,max_new_tokens=None):
        was_training=self.training;self.eval()
        try:
            if control=='text':
                weight=self.decoder.lm.get_input_embeddings().weight
                visual=[weight.new_empty((0,self.decoder.hidden_size)) for _ in items]
                projected=None
            else:
                projected=self.project_volumes([x['volume'] for x in items],[x['sample'].sequence for x in items])
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
