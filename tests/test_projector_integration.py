from types import SimpleNamespace
import pytest
import torch
from torch import nn
from transformers import AutoTokenizer, LlamaConfig, LlamaForCausalLM
from src.config import path_of
from src.data.manifest import load_manifests
from src.models.decoders.llama import LlamaDecoder
from src.models.multimodal import MultimodalModel
from src.models.projectors.registry import PROJECTORS, build_projector
from src.training.checkpoint import save_checkpoint, load_checkpoint
from src.training.optim import build_optimizer, build_scheduler
from test_projectors import config, volume


class TinyEncoder(nn.Module):
    def __init__(self, stage):
        super().__init__(); self.linear=nn.Linear(8,8); self.requires_grad_(stage!='a')
        self.lora_targets=[]

    def forward(self,volumes):
        return SimpleNamespace(features=[self.linear(v.patches) for v in volumes],
                               coords=[v.coords for v in volumes],metadata=[v.metadata for v in volumes])


@pytest.fixture(scope='module')
def decoder_and_samples():
    c=config('mlp2x_gelu'); c.decoder.gradient_checkpointing=False; c.generation.max_new_tokens=2
    tokenizer=AutoTokenizer.from_pretrained(path_of(c,c.decoder.path),local_files_only=True)
    lm=LlamaForCausalLM(LlamaConfig(vocab_size=128256,hidden_size=16,intermediate_size=32,
        num_hidden_layers=1,num_attention_heads=2,num_key_value_heads=2,max_position_embeddings=1024,
        pad_token_id=128004,eos_token_id=128009))
    return LlamaDecoder(c,lm,tokenizer),load_manifests(c)[0]['train'][:2]


def make_model(c, decoder):
    model=MultimodalModel.__new__(MultimodalModel); nn.Module.__init__(model)
    model.config=c; model.encoder=TinyEncoder(c.training.stage)
    model.decoder=decoder; model.projector=build_projector(c,8,decoder.hidden_size)
    return model


@pytest.mark.parametrize('name',PROJECTORS)
def test_answer_loss_generation_and_a_to_b_checkpoint(name,decoder_and_samples,tmp_path):
    decoder,samples=decoder_and_samples
    c=config(name); model=make_model(c,decoder)
    x,coords,meta=volume()
    items=[{'sample':s,'volume':SimpleNamespace(patches=x.detach(),coords=coords,metadata=meta)} for s in samples]
    result=model(items)
    assert result['answer_tokens']==sum(decoder.answer_count(s) for s in samples)
    actual=result['projector_token_counts'][0]
    assert result['encoder_token_counts']==[24,24]
    result['loss_sum'].backward()
    assert all(p.grad is None for p in model.encoder.parameters())
    assert all(p.grad is None for p in decoder.parameters())
    assert all(p.grad is not None for p in model.projector.parameters())
    generated=model.generate(items,max_new_tokens=1)
    assert all(r['visual_token_count']==actual and r['projector_token_count']==actual for r in generated)
    assert all(r['total_prefix_token_count']==r['prompt_token_count']+actual for r in generated)
    text=model.generate(items[:1],control='text',max_new_tokens=1)[0]
    assert text['encoder_token_count']==text['projector_token_count']==text['visual_token_count']==0
    zeros=model.generate(items[:1],control='zero',max_new_tokens=1)[0]
    assert zeros['visual_token_count']==actual
    optimizer=build_optimizer(model,c); scheduler=build_scheduler(optimizer,4,0)
    optimizer.step(); scheduler.step()
    path=tmp_path/'a.pt'; proof={'contract_id':'test'}
    save_checkpoint(path,model,optimizer,scheduler,{'step':1},proof)
    for stage in ('b_lora','b_full'):
        bc=config(name); bc.training.stage=stage; bc.training.feature_cache=False; bc.training.encoder_lr=1e-4
        branch=make_model(bc,decoder); load_checkpoint(path,branch,'initialize',expected_provenance=proof)
        torch.testing.assert_close(branch.projector(x,coords,meta).embeddings,
                                   model.projector(x,coords,meta).embeddings,rtol=0,atol=0)
        branch(items)['loss_sum'].backward()
        assert all(p.grad is not None and p.grad.abs().sum()>0 for p in branch.encoder.parameters())


def test_checkpoint_rejects_same_shape_different_budget(decoder_and_samples,tmp_path):
    decoder,_=decoder_and_samples; c=config('avgpool3d'); model=make_model(c,decoder)
    path=tmp_path/'a.pt'; save_checkpoint(path,model,None,None,{}, {})
    other=make_model(config('avgpool3d',target=4),decoder)
    with pytest.raises(ValueError,match='projector configuration'): load_checkpoint(path,other)


def test_random_selection_checkpoint_preserves_inherited_seed(decoder_and_samples,tmp_path):
    decoder,_=decoder_and_samples; c=config('token_select3d',strategy='random')
    model=make_model(c,decoder); path=tmp_path/'random.pt'
    save_checkpoint(path,model,None,None,{}, {})
    other_config=config('token_select3d',strategy='random'); other_config.seed=100
    other=make_model(other_config,decoder)
    with pytest.raises(ValueError,match='random-selection seed'): load_checkpoint(path,other)
