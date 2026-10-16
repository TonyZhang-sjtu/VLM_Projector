import copy
import json
from types import SimpleNamespace
import pytest
import torch
from src.config import load_config
from src.training.feature_targets import initialize_target_cache,FrozenFeatureTargets
from src.training.loss import combined_loss
from src.training.checkpoint import save_checkpoint,load_checkpoint,projector_contract
from src.training.optim import build_optimizer,build_scheduler
from src.models.projectors.bindformer.config import EXTENSION_DEFAULTS
from test_bindformer_modules import config
from test_bindformer_extensions import geometric_volume
from test_projector_integration import decoder_and_samples,make_model


def feature_config(*extra):
    return config('projector.params.feature_loss_weight=.1',
        'projector.params.feature_target=frozen_input','projector.params.feature_inner_dim=16',
        'projector.params.feature_samples=9','projector.params.feature_query_chunk_size=3',*extra)


def items_for(samples):
    x,coords,meta=geometric_volume()
    del meta['sequence']
    meta.update(preprocessing_version='test_preprocessing',cache_key='volume_0')
    return [{'sample':s,'volume':SimpleNamespace(patches=x+i,coords=coords,
              metadata={**meta,'cache_key':f'volume_{i}'}),'feature_sampling_step':7}
            for i,s in enumerate(samples)]


def test_teacher_cache_roundtrip_and_strict_guards(tmp_path):
    x,coords,meta=geometric_volume()
    meta.update(preprocessing_version='preprocess',cache_key='image_a')
    store=initialize_target_cache(tmp_path,'teacher',8,'preprocess')
    with pytest.raises(FileNotFoundError,match='target missing'):store.load(coords,meta)
    store.save(x.requires_grad_(),coords,meta)
    target=store.load(coords,meta)
    assert not target.requires_grad
    torch.testing.assert_close(target,x)
    with pytest.raises(ValueError,match='coordinate'):store.load(coords.flip(0),meta)
    with pytest.raises(ValueError,match='preprocessing'):store.load(coords,{**meta,'preprocessing_version':'changed'})
    with pytest.raises(ValueError,match='contract'):FrozenFeatureTargets(tmp_path,'wrong_teacher',8).load(coords,meta)
    with pytest.raises(ValueError,match='different contract'):initialize_target_cache(tmp_path,'other',8,'preprocess')
    original=json.loads((tmp_path/'manifest.json').read_text());original['preprocessing_version']='changed'
    (tmp_path/'manifest.json').write_text(json.dumps(original))
    with pytest.raises(ValueError,match='changed during'):store.load(coords,meta)


def test_dynamic_lora_input_is_not_a_frozen_teacher():
    with pytest.raises(ValueError,match='requires Stage A'):
        feature_config('training.stage=b_lora','training.feature_cache=false')


def test_final_z_auxiliary_loss_ce_gradients_generation_and_resume(decoder_and_samples,tmp_path):
    decoder,samples=decoder_and_samples
    c=feature_config('projector.params.geometry_enabled=true','projector.params.within_slot_sa=true',
                     'projector.params.local_reads_global=true','projector.params.num_tokens=20',
                     'projector.params.role_query_counts.signal=2')
    model=make_model(c,decoder);items=items_for(samples)
    result=model(items)
    assert result['feature_samples']==2 and result['feature_patch_count']==18
    loss=combined_loss(result,result['answer_tokens'],feature_weight=.1,sample_count=2)
    loss.backward()
    assert all(p.grad is None for p in model.encoder.parameters())
    assert all(p.grad is None for p in model.decoder.parameters())
    assert all(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum()>0
               for p in model.projector.parameters())
    optimizer=build_optimizer(model,c);scheduler=build_scheduler(optimizer,8,0)
    optimizer.step();scheduler.step()
    path=tmp_path/'aux.pt';save_checkpoint(path,model,optimizer,scheduler,{'step':1},{})
    expected=model(items)['feature_loss_sum'].detach()
    other=make_model(c,decoder)
    # The fixture uses random frozen weights; real construction reloads the same
    # pretrained encoder, whose weights are intentionally absent from checkpoints.
    other.encoder.load_state_dict(model.encoder.state_dict())
    opt=build_optimizer(other,c);sch=build_scheduler(opt,8,0)
    load_checkpoint(path,other,'resume',opt,sch)
    torch.testing.assert_close(other(items)['feature_loss_sum'],expected,rtol=0,atol=0)
    def forbidden(*args,**kwargs):raise AssertionError('Generation must not run reconstruction')
    other.projector.feature_reconstructor.objective=forbidden
    assert other.generate(items[:1],max_new_tokens=1)[0]['visual_token_count']==20


def test_cached_teacher_is_constant_while_student_receives_gradient(decoder_and_samples,tmp_path):
    decoder,samples=decoder_and_samples
    c=feature_config('projector.params.feature_target=frozen_base_cache',
                     'training.stage=b_lora','training.feature_cache=false','training.encoder_lr=1e-5')
    model=make_model(c,decoder);items=items_for(samples)
    store=initialize_target_cache(tmp_path,'fixed_teacher',8,'test_preprocessing')
    originals=[]
    for item in items:
        v=item['volume'];target=v.patches.square()
        store.save(target,v.coords,v.metadata);originals.append(target.clone())
    model.feature_target_store=store
    result=model(items)
    result['feature_loss_sum'].backward()
    assert all(p.grad is not None and p.grad.abs().sum()>0 for p in model.encoder.parameters())
    assert all(p.grad is None for p in model.decoder.parameters())
    with torch.no_grad():model.encoder.linear.weight.add_(2)
    for item,target in zip(items,originals):
        torch.testing.assert_close(store.load(item['volume'].coords,item['volume'].metadata),target)
    # Reconstruction-only backward still traverses the final output MLP.
    assert model.projector.head[0].weight.grad.abs().sum()>0


def test_real_decoder_accumulation_uses_separate_denominators(decoder_and_samples):
    decoder,samples=decoder_and_samples
    torch.manual_seed(72)
    c=feature_config();whole=make_model(c,decoder);split=make_model(c,decoder)
    split.load_state_dict(whole.state_dict())
    items=items_for(samples)
    result=whole(items);count=result['answer_tokens']
    combined_loss(result,count,feature_weight=.2,sample_count=2).backward()
    for item in items:
        row=split([item])
        combined_loss(row,count,feature_weight=.2,sample_count=2).backward()
    for a,b in zip(whole.projector.parameters(),split.projector.parameters()):
        torch.testing.assert_close(a.grad,b.grad,rtol=3e-4,atol=2e-6)
    with pytest.raises(ValueError,match='denominator'):combined_loss(result,count,feature_weight=.2)


def test_old_disabled_config_and_provenance_contract_remain_compatible(decoder_and_samples,tmp_path):
    from omegaconf import OmegaConf
    decoder,_=decoder_and_samples
    c=config();model=make_model(c,decoder)
    original=OmegaConf.to_container(c.projector,resolve=True)
    for key in EXTENSION_DEFAULTS:original['params'].pop(key)
    assert projector_contract(c.projector)==original
    path=tmp_path/'old.pt';save_checkpoint(path,model,None,None,{}, {})
    saved=torch.load(path,weights_only=False)
    saved['config']['projector']=original
    torch.save(saved,path)
    other=make_model(config(),decoder);load_checkpoint(path,other)
    assert set(other.projector.state_dict())==set(model.projector.state_dict())
    changed=config('projector.params.geometry_enabled=true')
    with pytest.raises(ValueError,match='projector configuration'):
        load_checkpoint(path,make_model(changed,decoder))


def test_trainer_accumulation_logging_validation_and_checkpoint(decoder_and_samples,tmp_path,monkeypatch):
    from src.data.dataset import VolumeDataset
    from src.training.trainer import Trainer
    decoder,samples=decoder_and_samples
    c=feature_config('training.max_epochs=1','training.max_steps=1',
        'training.effective_batch_size=2','training.microbatch_per_device=1',
        'training.validation_generation_samples=0','training.save_every_steps=0')
    trainer=Trainer.__new__(Trainer)
    trainer.config=c;trainer.rank=0;trainer.world=1;trainer.device=torch.device('cpu')
    trainer.output=tmp_path;trainer.accumulation=2;trainer.total_steps=1;trainer.plan=None
    trainer.model=make_model(c,decoder);trainer.wrapped=trainer.model
    items=items_for(samples);volumes={i['sample'].id:i['volume'] for i in items}
    trainer.train_data=VolumeDataset(samples,lambda s:volumes[s.id])
    trainer.val_data=VolumeDataset(samples[:1],lambda s:volumes[s.id])
    trainer.optimizer=build_optimizer(trainer.model,c)
    trainer.scheduler=build_scheduler(trainer.optimizer,1,0)
    trainer.provenance={};trainer.state=dict(global_step=0,epoch=0,next_window=0,
                                          bad_epochs=0,best_val_ce=float('inf'))
    for name in ('reset_peak_memory_stats','synchronize'):
        monkeypatch.setattr(torch.cuda,name,lambda *a,**kw:None)
    for name in ('max_memory_allocated','max_memory_reserved'):
        monkeypatch.setattr(torch.cuda,name,lambda *a,**kw:0)
    trainer.train()
    row=json.loads((tmp_path/'train.jsonl').read_text())
    assert row['actual_samples']==2 and row['feature_patch_count']==18
    assert row['total_objective']==pytest.approx(row['answer_ce']+.1*row['feature_loss'])
    validation=json.loads((tmp_path/'validation.jsonl').read_text())
    assert 'feature_loss' in validation and 'feature_cosine' in validation
    checkpoint=torch.load(tmp_path/'checkpoints/last.pt',weights_only=False)
    assert checkpoint['state']['global_step']==1
    assert any(k.startswith('feature_reconstructor.') for k in checkpoint['projector'])
