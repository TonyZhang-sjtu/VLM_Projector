#!/usr/bin/env python
"""Real report-CE/visual-LoRA smoke training plus noninvasive FP32 routing probes.

Run from the repository root, with PYTHONPATH=. and one idle CUDA device visible.
This is a structural smoke test, not a GREEN/BLEU efficacy experiment.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import torch
import torch.nn.functional as F
from src.config import ROOT, load_config, file_hash
from src.data.manifest import load_manifests
from src.models.projectors.bindformer.regions import make_regions
from src.training.checkpoint import rng_state, restore_rng
from src.training.trainer import Trainer


VARIANTS = {
    'r0': {},
    'r1': {'router_value_normalization': True},
    'r2': {'router_value_normalization': True, 'fixed_spatial_regions': True,
           'spatial_mean_aggregation': True},
    'r3': {'router_value_normalization': True, 'fixed_spatial_regions': True,
           'spatial_adaptive_aggregation': True},
}


def mean_cos(x):
    x=F.normalize(x.double(),dim=-1)
    n=len(x)
    return float((x.sum(0).square().sum()-x.square().sum())/(n*(n-1))) if n>1 else 1.


def spread(x):
    x=x.double()
    return float((x-x.mean(0)).norm()/x.norm().clamp_min(1e-12))


def support_stats(w):
    # Form the Gram matrix in FP64, not just the eigendecomposition. Near rank
    # one, FP32 roundoff creates spurious small singular values and inflates rank.
    s=F.normalize(w.T.double(),dim=-1)
    eigen=torch.linalg.eigvalsh((s@s.T).cpu()).clamp_min(0).sqrt()
    probability=eigen/eigen.sum()
    rank=float(torch.exp(-(probability*probability.clamp_min(1e-30).log()).sum()))
    return {'support_cosine':mean_cos(w.T),'support_effective_rank':rank}


def reader_replay(reader,args,expected):
    queries,visual,support=args[:3]
    regions=args[3] if len(args)==4 else None
    slots,roles,dim=queries.shape
    heads,hd=reader.heads,dim//reader.heads
    q=reader.to_q(reader.norm_query(queries)).reshape(slots,roles,heads,hd).permute(0,2,1,3)
    keys,values=[t.reshape(-1,heads,hd).transpose(0,1)
                 for t in reader.to_kv(reader.norm_visual(visual)).chunk(2,-1)]
    outputs=[]
    maps=visual.new_zeros((slots,roles,len(visual)))
    max_outside=0.
    for start in range(0,slots,8):
        count=min(8,slots-start)
        if regions is None:
            k,v=keys[None],values[None]
            bias=(support[:,start:start+count].T[:,None,:]+reader.epsilon).log()
            valid=None
        else:
            ids=regions.indices[start:start+count]
            valid=regions.valid[start:start+count]
            k,v=[t[:,ids].permute(1,0,2,3) for t in (keys,values)]
            columns=torch.arange(start,start+count,device=ids.device)[:,None]
            bias=(support[ids,columns][:,None,:]+reader.epsilon).log()
        bias=bias*reader.support_beta[None,:,None]
        if valid is not None:bias=bias.masked_fill(~valid[:,None,:],-torch.inf)
        alpha=(q[start:start+count]@k.transpose(-1,-2)/math.sqrt(hd)+bias[:,None]).softmax(-1)
        averaged=alpha.mean(1)
        if regions is None:
            maps[start:start+count]=averaged
        else:
            for i in range(count):
                maps[start+i,:,ids[i,valid[i]]]=averaged[i,:,valid[i]]
            if (~valid).any():
                max_outside=max(max_outside,float(alpha.masked_select(~valid[:,None,None,:]).abs().max()))
        outputs.append((alpha@v).transpose(1,2).reshape(count,roles,dim))
    tokens=reader.to_out(torch.cat(outputs))
    replay=tokens+reader.ffn(tokens)
    relative=float((replay-expected).norm()/expected.norm().clamp_min(1e-12))
    if relative>2e-5:raise AssertionError(f'Reader replay mismatch: {relative}')
    return dict(reader_replay_relative_l2=relative,
                reader_attention_cosine_by_role=[mean_cos(maps[:,r]) for r in range(roles)],
                reader_attention_spread_by_role=[spread(maps[:,r]) for r in range(roles)],
                packed_padding_attention_max=max_outside)


def common_initial_hash(projector):
    common_hash=hashlib.sha256()
    for name,value in projector.state_dict().items():
        if '.value_norm.' not in name and '.value_projection.' not in name:
            common_hash.update(name.encode());common_hash.update(value.detach().cpu().numpy().tobytes())
    return common_hash.hexdigest()


@torch.inference_mode()
def probe(trainer,step,variant,records):
    model,p=trainer.model,trainer.model.projector
    state,was_training,old_aux=rng_state(),model.training,p.params['return_auxiliary']
    captured={}
    handle=p.reader.register_forward_hook(
        lambda module,args,output:captured.update(args=args,output=output))
    try:
        model.eval();p.params['return_auxiliary']=True
        rows=[]
        for index in range(len(trainer.val_data)):
            item=trainer.val_data[index]
            encoded=model.encoder([item['volume']])
            f=encoded.features[0].float()
            coords=encoded.coords[0]
            meta={**encoded.metadata[0],'sequence':item['sample'].sequence}
            result=p(f,coords,meta)  # encoder retains original BF16; projector FP32
            aux=result.auxiliary
            h=p.input_adapter(f,coords,meta,meta['sequence'])
            router=p.router
            regions=(make_regions(meta['grid_dhw'],meta,p.allocation['num_slots'],h.device)
                     if p.params['fixed_spatial_regions'] else None)
            values=router.values(h)
            slots=router.initial_slots if regions is None else router.initial_slots[int(router.background):]
            keys=router.to_k(router.norm_input(h)) if regions is None or router.spatial_adaptive else h
            timeline=[]
            for iteration in range(router.iterations+1):
                local=slots[int(router.background):] if regions is None else slots
                if regions is None:
                    a=router.assignment(slots,keys)
                    support=a[:,int(router.background):]
                    normalized=support/(support.sum(0,keepdim=True)+router.epsilon)
                    entropy=float(-(a*a.clamp_min(1e-12).log()).sum(-1).mean()/math.log(a.shape[1]))
                    within_entropy=None
                else:
                    packed=router.regional_weights(slots,keys,regions.indices,regions.valid)
                    normalized=regions.dense(packed)
                    entropy=0.  # hard membership; not a measure of learned specialization
                    lengths=regions.valid.sum(-1)
                    usable=lengths>1
                    within_entropy=float((-(packed*packed.clamp_min(1e-12).log()).sum(-1)[usable]
                                          /lengths[usable].float().log()).mean())
                evidence=normalized.T@values
                timeline.append(dict(iteration=iteration,slot_cosine=mean_cos(local),
                    patch_to_slot_entropy=entropy,within_region_entropy=within_entropy,
                    evidence_cosine=mean_cos(evidence),evidence_norm=float(evidence.norm(dim=-1).mean()),
                    **support_stats(normalized)))
                if iteration<router.iterations:
                    slots=(router.update(slots,keys,values) if regions is None else
                           router.regional_update(slots,keys,values,regions.indices,regions.valid))
            local=slots[int(router.background):] if regions is None else slots
            torch.testing.assert_close(local,aux.slots,rtol=0,atol=0)
            g=p.allocation['global_tokens']
            z=result.embeddings[g:]
            row=dict(variant=variant,step=step,id=item['sample'].id,iterations=timeline,
                     input_h_norm=float(h.norm(dim=-1).mean()),
                     router_value_norm=float(values.norm(dim=-1).mean()),
                     local_z_cosine=mean_cos(z),local_z_relative_spread=spread(z),
                     reader_token_cosine_by_role=[mean_cos(aux.reader_role_tokens[:,r]) for r in range(4)],
                     post_sa_ca_token_cosine_by_role=[mean_cos(aux.role_tokens[:,r]) for r in range(4)],
                     **reader_replay(p.reader,captured['args'],captured['output']))
            if regions is not None:
                row['region_sizes']=regions.valid.sum(-1).tolist()
            rows.append(row)
            captured.clear()
        records.extend(rows)
        (trainer.output/'routing_probe.json').write_text(json.dumps({
            'scope':'First three sorted validation T1C cases per GLI/MEN/MET; nine volumes, never trained.',
            'precision':'Encoder BF16; projector and reader replay FP32. eval/no_grad.',
            'rank_precision':'Support normalization, Gram matrix and eigendecomposition in FP64.',
            'effective_rank_definition':'exp entropy of singular values of L2-normalized slot support rows',
            'records':records},indent=2))
        print(json.dumps({'variant':variant,'probe_step':step,'samples':len(rows),
            **{k:sum(r[k] for r in rows)/len(rows) for k in ('local_z_cosine','local_z_relative_spread')},
            'slot_cosine':sum(r['iterations'][-1]['slot_cosine'] for r in rows)/len(rows),
            'support_cosine':sum(r['iterations'][-1]['support_cosine'] for r in rows)/len(rows)}),flush=True)
    finally:
        handle.remove();captured.clear()
        p.params['return_auxiliary']=old_aux
        model.train(was_training)
        restore_rng(state)


def reprobe_saved_run(output,variant):
    """Replay identical initialization and saved steps; never retrain or edit weights."""
    from types import SimpleNamespace
    from omegaconf import OmegaConf
    from src.models.multimodal import MultimodalModel
    from src.preprocessing.registry import build_preprocessor
    from src.data.dataset import VolumeDataset
    from src.training.trainer import seed_all
    cfg=OmegaConf.load(output/'resolved_config.yaml')
    seed_all(cfg.seed)
    model=MultimodalModel(cfg).to(cfg.training.device)
    protocol=json.loads((output/'smoke_protocol.json').read_text())
    assert common_initial_hash(model.projector)==protocol['common_projector_initial_state_sha256']
    samples,_=load_manifests(cfg)
    lookup={s.id:s for s in samples['val']}
    selected=[lookup[sid] for sid in json.loads((output/'val_ids.json').read_text())]
    harness=SimpleNamespace(model=model,output=output,
        val_data=VolumeDataset(selected,build_preprocessor(cfg)))
    # Preserve the original raw diagnostics; new probes include the FP64 rank fix.
    path=output/'routing_probe.json'
    archive=output/'routing_probe_initial_fp32_gram.json'
    if path.exists() and not archive.exists():archive.write_bytes(path.read_bytes())
    records=[]
    probe(harness,0,variant,records)
    diagnostics=[json.loads(line) for line in (output/'optimizer_diagnostics.jsonl').read_text().splitlines()]
    assert all(v['update_norm']==0 for v in diagnostics[0]['components'].values())
    probe(harness,1,variant,records)
    for filename in ('boundary_00000004.pt','last.pt'):
        saved=torch.load(output/'checkpoints'/filename,map_location='cpu',weights_only=False,mmap=True)
        model.projector.load_state_dict(saved['projector'],strict=True)
        parameters=dict(model.encoder.named_parameters())
        assert set(saved['encoder'])=={k for k in parameters if 'lora_' in k}
        with torch.no_grad():
            for k,v in saved['encoder'].items():parameters[k].copy_(v.to(parameters[k]))
        model.encoder.set_phase(saved['state']['phase'])
        step=saved['state']['global_step']
        del saved,parameters
        probe(harness,step,variant,records)
    (output/'reprobe_protocol.json').write_text(json.dumps({
        'reason':'FP64 support Gram matrix removes FP32 roundoff inflation of near-one effective ranks.',
        'script_sha256':file_hash(Path(__file__)),
        'initialization_hash_verified':True,'step_one_zero_update_verified':True,
        'checkpoints':['boundary_00000004.pt','last.pt'],'training_updates_performed':0},indent=2))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--variant',choices=list(VARIANTS),required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--steps',type=int,default=12)
    parser.add_argument('--batch-size',type=int,default=4)
    parser.add_argument('--local-config',default='configs/local_paths.yaml')
    parser.add_argument('--reprobe-only',action='store_true',
                        help='Recheck an existing run at initialization and saved steps, without any training')
    args=parser.parse_args()
    if args.steps<5:parser.error('Need at least five steps to exercise both warmups and joint training')
    torch.set_num_threads(4)
    output=args.output.resolve()
    if args.reprobe_only:
        reprobe_saved_run(output,args.variant)
        return
    if (output/'train.jsonl').exists():
        raise FileExistsError('Smoke output already contains training; use a fresh output directory')
    output.mkdir(parents=True,exist_ok=True)
    overrides=['projector.target_compression_ratio=8','projector.params.within_slot_sa=true',
               'projector.params.local_reads_global=true','training.projector_warmup_steps=2',
               'training.joint_warmup_steps=2',f'training.max_steps={args.steps}',
               f'training.effective_batch_size={args.batch_size}','training.validation_generation_samples=0',
               'training.log_optimizer_diagnostics=true',f'training.run_dir={output}',
               'training.early_stopping_enabled=false']
    overrides += [f'projector.params.{k}={str(v).lower()}' for k,v in VARIANTS[args.variant].items()]
    files=['configs/training/constant_lr.yaml','configs/projector/bindformer3d.yaml',args.local_config]
    cfg=load_config(files,overrides)
    samples,_=load_manifests(cfg)
    cohorts=['BraTS_GLI','BraTS_MEN','BraTS_MET']
    val=[s.id for cohort in cohorts for s in sorted(
        (s for s in samples['val'] if s.cohort==cohort and s.sequence=='T1C'),key=lambda s:s.id)[:3]]
    groups=[sorted((s for s in samples['train'] if s.cohort==cohort and s.sequence==sequence),key=lambda s:s.id)
            for cohort in cohorts for sequence in ['T1W','T1C','T2W','FLAIR']]
    if any(not group for group in groups):
        raise ValueError('Missing train cohort/sequence group')
    # Round robin gives the four methods the same stratified pool and shuffle.
    train=[group[i].id for i in range(min(map(len,groups))) for group in groups][:args.steps*args.batch_size]
    if len(train)!=args.steps*args.batch_size or len(val)!=9:raise ValueError('Incomplete smoke selection')
    for split,ids in [('train',train),('val',val)]:
        path=output/f'{split}_ids.json';path.write_text(json.dumps(ids,indent=2))
        overrides.append(f'training.{split}_ids_file={path}')
    cfg=load_config(files,overrides)
    trainer=Trainer(cfg)
    (output/'smoke_protocol.json').write_text(json.dumps({
        'variant':args.variant,'steps':args.steps,'batch_size':args.batch_size,
        'warmups':[2,2],'ratio':8,'seed':42,'loss':'answer CE only; no reconstruction or auxiliary loss',
        'common_projector_initial_state_sha256':common_initial_hash(trainer.model.projector),
        'script_sha256':file_hash(Path(__file__)),
        'source_files':{str(p.relative_to(ROOT)):file_hash(p)
                        for p in sorted((ROOT/'src/models/projectors/bindformer').glob('*.py'))},
        'limitations':'Short structural smoke. No report generation or GREEN comparison. Not a converged trial.'},indent=2))
    records=[]
    probe(trainer,0,args.variant,records)
    steps={'completed':0}
    gradients=[]
    def post_step(optimizer,unused_args,unused_kwargs):
        steps['completed']+=1
        step=steps['completed']
        keys=('router.to_q.weight','router.to_k.weight','router.value_norm.weight','router.gru.weight_ih')
        grads={k:float(p.grad.norm()) if p.grad is not None else None
               for k,p in trainer.model.projector.named_parameters() if k in keys}
        gradients.append({'step':step,**grads})
        (output/'routing_gradients.json').write_text(json.dumps(gradients,indent=2))
        if step in (1,4,args.steps):probe(trainer,step,args.variant,records)
    handle=trainer.optimizer.register_step_post_hook(post_step)
    try:trainer.train()
    finally:handle.remove()
    assert trainer.state['global_step']==args.steps
    print(json.dumps({'completed':args.variant,'steps':args.steps,'output':str(output)}),flush=True)


if __name__=='__main__':
    main()
