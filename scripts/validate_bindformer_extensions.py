"""Bounded real-MRI/NeuroVFM-LoRA acceptance with a small frozen Llama decoder.

Checks integration and gradients, not clinical quality or 8B decoder throughput.
Export train/val --limit 1 with cache_bindformer_teacher before running this.
"""
import argparse
import json
from pathlib import Path
import torch
from transformers import AutoTokenizer,LlamaConfig,LlamaForCausalLM
from src.config import ROOT,load_config,path_of
from src.data.manifest import load_manifests
from src.models.decoders.llama import LlamaDecoder
from src.models.multimodal import MultimodalModel
from src.models.projectors.budget import prepare_budget
from src.models.projectors.bindformer.geometry import patch_geometry
from src.preprocessing.registry import build_preprocessor
from src.training.loss import combined_loss
from src.training.optim import build_optimizer
from src.training.reproducibility import configure_reproducibility
from src.training.trainer import seed_all,validate


def main():
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--device',required=True)
    cli.add_argument('--teacher-cache',required=True)
    cli.add_argument('--output',default='runs/bindformer_extensions_acceptance/real_runtime.json')
    args=cli.parse_args()
    c=load_config(['configs/training/joint_lora.yaml','configs/preprocessing/native_grid.yaml',
                  'configs/projector/bindformer3d.yaml'],[
        f'training.device={args.device}','training.deterministic=true',
        'projector.target_compression_ratio=32','projector.params.geometry_enabled=true',
        'projector.params.geometry_space=grid','projector.params.within_slot_sa=true',
        'projector.params.local_reads_global=true','projector.params.role_query_counts.signal=2',
        'projector.params.feature_loss_weight=.1',f'projector.params.feature_target_cache={args.teacher_cache}'])
    device=torch.device(args.device);torch.cuda.set_device(device)
    configure_reproducibility(c);seed_all(c.seed)
    samples,_=load_manifests(c);preprocessor=build_preprocessor(c)
    items=[{'sample':samples[split][0],'volume':preprocessor(samples[split][0])} for split in ('train','val')]
    budget=prepare_budget(c,[i['sample'] for i in items])['budget']
    tokenizer=AutoTokenizer.from_pretrained(path_of(c,c.decoder.path),local_files_only=True)
    lm=LlamaForCausalLM(LlamaConfig(vocab_size=128256,hidden_size=64,intermediate_size=128,
        num_hidden_layers=1,num_attention_heads=4,num_key_value_heads=2,max_position_embeddings=8192,
        pad_token_id=128004,eos_token_id=128009))
    decoder=LlamaDecoder(c,lm,tokenizer)
    model=MultimodalModel(c,decoder).to(device).train()
    model.encoder.set_phase('joint_warmup')
    optimizer=build_optimizer(model,c)
    assert model.feature_target_store.provenance()['teacher']=='frozen_pretrained_neurovfm'
    target=model.feature_target_store.load(items[0]['volume'].coords,items[0]['volume'].metadata)
    geometry={}
    for space in ('grid','voxel','mm'):
        g=patch_geometry(items[0]['volume'].coords,items[0]['volume'].metadata,space)
        geometry[space]={'volume_sum':float(g.volumes.sum()),'valid_patches':int((g.valid_fraction>0).sum())}
    initial=model.projector.head[0].weight.detach().clone()
    frozen=model.projector.concept_memory.embeddings.detach().clone()
    records=[];torch.cuda.reset_peak_memory_stats(device)
    for step in range(2):
        optimizer.zero_grad(set_to_none=True)
        items[0]['feature_sampling_step']=step
        with torch.autocast('cuda',dtype=torch.bfloat16):
            result=model(items[:1])
            loss=combined_loss(result,result['answer_tokens'],feature_weight=.1,sample_count=1)
        loss.backward()
        bad=[name for name,p in model.projector.named_parameters()
             if p.grad is None or not torch.isfinite(p.grad).all() or p.grad.abs().sum()==0]
        assert not bad,bad
        assert all(p.grad is None for p in model.decoder.parameters())
        assert all(p.grad is None for n,p in model.encoder.named_parameters() if 'lora_' not in n)
        live=[p for n,p in model.encoder.named_parameters() if 'lora_' in n]
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in live)
        assert any(p.grad.abs().sum()>0 for p in live)
        if step==1:assert all(p.grad.abs().sum()>0 for p in live)
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.,error_if_nonfinite=True)
        optimizer.step()
        records.append({'answer_ce':float(result['loss_sum'].detach()/result['answer_tokens']),
                        'feature_loss':float(result['feature_loss_sum'].detach()),
                        'feature_cosine':float(result['feature_cosine_sum']),
                        'patches':result['feature_patch_count']})
        print(json.dumps({'step':step,**records[-1]}),flush=True)
        del result,loss
    assert not torch.equal(initial,model.projector.head[0].weight)
    assert torch.equal(frozen,model.projector.concept_memory.embeddings)
    assert torch.equal(target,model.feature_target_store.load(items[0]['volume'].coords,items[0]['volume'].metadata))
    optimizer.zero_grad(set_to_none=True)
    metrics=validate(model,items[1:])
    # Generation must neither reconstruct features nor read teacher files.
    def forbidden(*a,**kw):raise AssertionError('Auxiliary operation during generation')
    model.feature_target_store.load=forbidden
    model.projector.feature_reconstructor.objective=forbidden
    with torch.autocast('cuda',dtype=torch.bfloat16):
        generated=model.generate(items[1:],max_new_tokens=2)[0]
    assert generated['visual_token_count']==275
    report={'status':'passed','scope':'Real BraTS train/val images, real NeuroVFM LoRA, small random frozen Llama (64 wide, 1 layer); no 8B/quality claim',
        'budget':budget,'geometry':geometry,'training':records,'validation':metrics,
        'generated_tokens':generated['generated_token_count'],
        'peak_allocated_gib':torch.cuda.max_memory_allocated(device)/2**30,
        'teacher_contract':model.feature_target_store.provenance()}
    path=Path(args.output)
    if not path.is_absolute():path=ROOT/path
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
