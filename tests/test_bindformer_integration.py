import json
import shutil
from types import SimpleNamespace
import pytest
import torch
from src.training.checkpoint import save_checkpoint, load_checkpoint
from src.training.optim import build_optimizer, build_scheduler
from test_projector_integration import decoder_and_samples, make_model
from test_bindformer_modules import config, volume, BANK


@pytest.mark.parametrize('knowledge,beta',[('name_definition',1),('none',1),('name_definition',0)])
def test_real_decoder_ce_sequence_context_gradients_generate_and_resume(
        knowledge,beta,decoder_and_samples,tmp_path):
    torch.manual_seed(42)
    decoder,samples=decoder_and_samples
    c=config(f'projector.params.knowledge_mode={knowledge}',f'projector.params.support_beta={beta}')
    model=make_model(c,decoder)
    x,coords,meta=volume()
    del meta['sequence']
    items=[{'sample':s,'volume':SimpleNamespace(patches=x,coords=coords,metadata=meta)} for s in samples]
    seen=[]
    hook=model.projector.input_adapter.register_forward_pre_hook(lambda _,args:seen.append(args[-1]))
    result=model(items)
    assert result['answer_tokens']==sum(decoder.answer_count(s) for s in samples)
    assert result['projector_token_counts']==[12,12]
    assert result['encoder_token_counts']==[24,24]
    assert set(result)=={'loss_sum','answer_tokens','lengths','encoder_token_counts','projector_token_counts'}
    (result['loss_sum']/result['answer_tokens']).backward()
    assert all(p.grad is None for p in model.encoder.parameters())
    assert all(p.grad is None for p in decoder.parameters())
    assert all(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum()>0
               for p in model.projector.parameters())
    assert seen==[s.sequence for s in samples]
    assert 'sequence' not in meta
    generated=model.generate(items,max_new_tokens=1)
    assert all(r['visual_token_count']==12 and r['generated_token_count']<=1 for r in generated)
    assert seen==[s.sequence for s in samples]*2
    hook.remove()
    optimizer=build_optimizer(model,c);scheduler=build_scheduler(optimizer,8,0)
    optimizer.step();scheduler.step()
    path=tmp_path/'checkpoint.pt'
    save_checkpoint(path,model,optimizer,scheduler,{'step':1},{})
    expected=model.projector(x,coords,{**meta,'sequence':samples[0].sequence}).embeddings.detach()
    restored=make_model(c,decoder)
    opt=build_optimizer(restored,c);sch=build_scheduler(opt,8,0)
    load_checkpoint(path,restored,'resume',opt,sch)
    actual=restored.projector(x,coords,{**meta,'sequence':samples[0].sequence}).embeddings
    torch.testing.assert_close(expected,actual,rtol=0,atol=0)
    assert len(opt.state)>0 and sch.last_epoch==1


def test_checkpoint_checks_asset_content_even_without_external_provenance(tmp_path,decoder_and_samples):
    dest=tmp_path/'bank'
    shutil.copytree(BANK,dest,ignore=shutil.ignore_patterns('mentions*','*reconciliation*'))
    c=config(f'projector.params.concept_bank={dest}')
    decoder,_=decoder_and_samples
    model=make_model(c,decoder)
    checkpoint=tmp_path/'model.pt'
    save_checkpoint(checkpoint,model,None,None,{}, {})
    manifest=dest/'embedding_manifest.json'
    value=json.loads(manifest.read_text());value['version']='changed-at-same-path'
    manifest.write_text(json.dumps(value))
    other=make_model(c,decoder)
    with pytest.raises(ValueError,match='asset fingerprint'):
        load_checkpoint(checkpoint,other)


def test_auxiliary_opt_in_and_context_length_guard(decoder_and_samples):
    decoder,samples=decoder_and_samples
    model=make_model(config('projector.params.return_auxiliary=false'),decoder)
    x,coords,meta=volume()
    assert model.projector(x,coords,meta).auxiliary is None
    with pytest.raises(ValueError,match='batch mismatch'):
        model.project_volumes([object()],[])
    del meta['sequence']
    with pytest.raises(ValueError,match='sequence context'):
        model.projector(x,coords,meta)
