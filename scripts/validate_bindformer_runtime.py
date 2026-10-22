"""One real-volume CE/backward/update/generation acceptance, not a training run.

Run as: python -m scripts.validate_bindformer_runtime --device cuda:3
"""
import argparse
import json
from pathlib import Path
import time
import torch
from src.config import ROOT, load_config
from src.data.manifest import load_manifests
from src.models.multimodal import MultimodalModel
from src.models.projectors.budget import SUPPORTED_RATIOS, prepare_budget
from src.preprocessing.registry import build_preprocessor
from src.training.checkpoint import provenance
from src.training.optim import build_optimizer
from src.training.stages import assert_stage
from src.training.trainer import seed_all


def main():
    cli=argparse.ArgumentParser(__doc__)
    cli.add_argument('--device',required=True)
    cli.add_argument('--ratio',type=int,choices=SUPPORTED_RATIOS,default=32)
    cli.add_argument('--output',default='runs/bindformer_acceptance/runtime.json')
    args=cli.parse_args()
    config=load_config([ROOT/'configs/training/stage_a.yaml',
                        ROOT/'configs/training/deterministic.yaml',
                        ROOT/'configs/projector/bindformer3d.yaml'],[
        f'training.device={args.device}',f'projector.target_compression_ratio={args.ratio}',
        'generation.max_new_tokens=2'])
    device=torch.device(args.device)
    torch.cuda.set_device(device)
    seed_all(config.seed)
    samples,manifest_summary=load_manifests(config)
    sample=samples['train'][0]
    preflight=prepare_budget(config,[sample])
    config.training.required_visual_tokens=preflight['budget']['expected_output_tokens']
    preprocessor=build_preprocessor(config)
    item={'sample':sample,'volume':preprocessor(sample)}
    print(json.dumps({'step':'input_ready','sequence':sample.sequence,
                      'patches':len(item['volume'].patches),'budget':preflight['budget']}),flush=True)
    model=MultimodalModel(config).to(device)
    # Match Trainer.train(): pretrained HF models initially load in eval mode.
    # Frozen weights still need training mode for activation checkpointing.
    model.train()
    assert model.decoder.lm.model.training and model.decoder.lm.model.gradient_checkpointing
    parameters=assert_stage(model)
    proof=provenance(config,model,preprocessor,manifest_summary)
    assert proof['projector_assets']['vocabulary_scope']=='brats_train'
    assert proof['projector_assets']['active_concepts']==120
    optimizer=build_optimizer(model,config)
    initial=model.projector.head[0].weight.detach().clone()
    frozen_concepts=model.projector.concept_memory.embeddings.detach().clone()
    torch.cuda.reset_peak_memory_stats(device)
    started=time.monotonic()
    print(json.dumps({'step':'models_ready','projector_parameters':parameters['projector']['parameters']}),flush=True)
    with torch.autocast('cuda',dtype=torch.bfloat16):
        result=model([item])
        loss=result['loss_sum']/result['answer_tokens']
    loss.backward()
    bad=[name for name,p in model.projector.named_parameters()
         if p.grad is None or not torch.isfinite(p.grad).all() or p.grad.abs().sum()==0]
    assert not bad, bad
    assert all(p.grad is None for p in model.encoder.parameters())
    assert all(p.grad is None for p in model.decoder.parameters())
    torch.nn.utils.clip_grad_norm_(model.projector.parameters(),config.training.grad_clip_norm)
    optimizer.step()
    assert not torch.equal(initial,model.projector.head[0].weight)
    assert torch.equal(frozen_concepts,model.projector.concept_memory.embeddings)
    optimizer.zero_grad(set_to_none=True)
    ce_before=float(loss.detach())
    torch.cuda.synchronize(device)
    backward_seconds=time.monotonic()-started
    print(json.dumps({'step':'ce_backward_update_passed','answer_ce':ce_before,
                      'seconds':backward_seconds}),flush=True)
    model.eval()
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
        after=model([item])
        ce_after=float(after['loss_sum']/after['answer_tokens'])
        generated=model.generate([item],max_new_tokens=2)[0]
    assert generated['visual_token_count']==preflight['budget']['expected_output_tokens']
    assert generated['generated_token_count']>=1
    report={'status':'passed','scope':'one real BraTS training image, one optimizer update; no quality claim',
            'sample_id':sample.id,'sequence':sample.sequence,'budget':preflight['budget'],
            'knowledge_scope':'brats_train','active_concepts':120,
            'projector_parameters':parameters['projector']['parameters'],
            'encoder_trainable':parameters['encoder']['parameters'],
            'decoder_trainable':parameters['decoder']['parameters'],
            'decoder_gradient_checkpointing':config.decoder.gradient_checkpointing,
            'answer_tokens':result['answer_tokens'],'answer_ce_before':ce_before,'answer_ce_after':ce_after,
            'ce_backward_update_seconds':backward_seconds,
            'generated_tokens':generated['generated_token_count'],
            'peak_allocated_gib':torch.cuda.max_memory_allocated(device)/2**30,
            'device':str(device),'asset_provenance':proof['projector_assets'],
            'source_files':proof['source_files']}
    output=Path(args.output)
    if not output.is_absolute():output=ROOT/output
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('source_files','asset_provenance')},indent=2),flush=True)


if __name__=='__main__':
    main()
