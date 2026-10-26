"""Explicit real-model smoke only; never launches a sweep or full training."""
import argparse
import json
from pathlib import Path
import time


def main():
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--plan',type=Path,required=True)
    cli.add_argument('--output',type=Path,required=True)
    args=cli.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    from .runtime import install
    install()
    import torch
    from omegaconf import OmegaConf
    from src.config import path_of
    from src.data.manifest import load_manifests
    from src.data.dataset import VolumeDataset
    from src.preprocessing.registry import build_preprocessor
    from src.models.multimodal import MultimodalModel
    from src.training.trainer import seed_all
    from src.training.optim import build_optimizer,build_joint_scheduler
    from src.training.schedule import resolve_joint_schedule
    from src.training.checkpoint import save_checkpoint,load_checkpoint
    from src.training.loss import combined_loss
    from .store import FactStore
    c=OmegaConf.create(json.loads(args.plan.read_text())['runs'][0]['config'])
    # Explicit smoke-only overrides. Formal shell settings are not modified.
    c.projector.params.report_facts.fact_weight=.1
    c.projector.params.report_facts.pair_weight=.1
    c.projector.params.report_facts.warmup_steps=0
    c.training.run_dir=str(args.output)
    torch.cuda.set_device(0);seed_all(c.seed)
    store=FactStore(path_of(c,c.projector.params.report_facts.path))
    samples=load_manifests(c)[0]['train'];lookup={s.id:s for s in samples}
    first=next(rid for rid,ps in store.pairs.items() if rid in lookup and ps[0]['field']=='laterality'
               and len(store.rows[rid]['facts'])>=3)
    second=store.pairs[first][0]['partner_id']
    data=VolumeDataset([lookup[first],lookup[second]],build_preprocessor(c))
    items=[data[0],data[1]]
    model=MultimodalModel(c).to('cuda:0');model.train()
    optimizer=build_optimizer(model,c)
    schedule=resolve_joint_schedule(c.training,2460)
    scheduler=build_joint_scheduler(optimizer,schedule)
    # Run after both warmups, including a real vision-LoRA gradient and update.
    step=154
    model.encoder.set_phase('joint_cosine');model.set_supervision_step(step)
    for group in optimizer.param_groups:
        group['lr']=c.training.projector_lr if group['component']=='projector' else c.training.encoder_lr
    records=[]
    for mode,fw,pw in [('E2',0.,.1),('E3',.1,.1),('E1',.1,0.)]:
        model.fact_settings.update(fact_weight=fw,pair_weight=pw)
        optimizer.zero_grad(set_to_none=True);torch.cuda.reset_peak_memory_stats()
        start=time.perf_counter();totals={};count=sum(model.decoder.answer_count(x['sample']) for x in items)
        for item in items:
            with torch.autocast('cuda',dtype=torch.bfloat16):
                result=model([item]);loss=combined_loss(result,count,sample_count=len(items))
            assert torch.isfinite(loss)
            loss.backward()
            for key,value in result['auxiliary_metrics'].items():totals[key]=totals.get(key,0.)+value
            assert result['projector_token_counts']==[275 if c.projector.target_compression_ratio==32 else 138]
            del loss,result
        gradients={name:sum(float(p.grad.float().norm()) for p in module.parameters() if p.grad is not None)
                   for name,module in [('projector',model.projector),('encoder',model.encoder),('decoder',model.decoder)]}
        assert gradients['projector']>0 and gradients['encoder']>0 and gradients['decoder']==0
        if fw:assert any(p.grad is not None and p.grad.abs().sum()>0 for p in model.projector.fact_heads.parameters())
        assert totals['pair_count']==(2 if pw else 0)
        norm=torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.,error_if_nonfinite=True)
        optimizer.step();torch.cuda.synchronize()
        record=dict(mode=mode,metrics=totals,gradient_norms=gradients,clipped_global_norm=float(norm),
                    seconds=time.perf_counter()-start,peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                    peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30)
        records.append(record);print(json.dumps(record),flush=True)
    # Checkpoint roundtrip includes fact heads and vision LoRA. The saved
    # configuration is E3 and is restored before saving/loading.
    model.fact_settings.update(fact_weight=.1,pair_weight=.1)
    optimizer.zero_grad(set_to_none=True)
    state=dict(global_step=step,phase='joint_cosine',schedule=schedule.to_dict())
    checkpoint=args.output/'smoke.pt'
    save_checkpoint(checkpoint,model,optimizer,None,state,{'contract_id':'smoke'})
    name,param=next(iter(model.projector.fact_heads.named_parameters()))
    before=param.detach().clone()
    with torch.no_grad():param.add_(1)
    load_checkpoint(checkpoint,model,'evaluate')
    assert torch.equal(param,before)
    model.eval()
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
        evaluated=model([items[0]])
        assert 'auxiliary_loss_sum' not in evaluated
        # Actual report generation uses the original task (no fact QA targets).
        generated=model.generate([items[0]],max_new_tokens=8)
    proof=dict(ratio=int(c.projector.target_compression_ratio),status='passed',scope='real_model_two_samples_three_objectives',
               sample_ids=[first,second],records=records,checkpoint_roundtrip=True,
               validation_has_auxiliary_loss=False,generation=generated,
               note='Smoke updates are not experiment results and do not establish quality gains.')
    (args.output/'summary.json').write_text(json.dumps(proof,indent=2)+'\n')
    # A smoke checkpoint is deliberately retained for inspection, isolated from runs.
    print(json.dumps({k:v for k,v in proof.items() if k not in ('records','generation')}),flush=True)


if __name__=='__main__':main()
