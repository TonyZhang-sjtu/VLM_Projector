from dataclasses import replace
import pytest
import torch
from transformers import LlamaConfig,LlamaForCausalLM,AutoTokenizer
from src.config import load_config,path_of
from src.data.manifest import load_manifests
from src.models.decoders.llama import LlamaDecoder


@pytest.fixture()
def setup():
    torch.manual_seed(2)
    c=load_config(overrides=['decoder.gradient_checkpointing=false','generation.max_new_tokens=4'])
    tokenizer=AutoTokenizer.from_pretrained(path_of(c,c.decoder.path),local_files_only=True)
    model=LlamaForCausalLM(LlamaConfig(vocab_size=128256,hidden_size=32,intermediate_size=64,
        num_hidden_layers=2,num_attention_heads=4,num_key_value_heads=2,max_position_embeddings=2048,
        attention_dropout=0.0,pad_token_id=128004,eos_token_id=128009))
    decoder=LlamaDecoder(c,model,tokenizer)
    rows=load_manifests(c)[0]['train']
    samples=[rows[0],replace(rows[1],finding='A 2 × 3 mm lesion; no edema.')]
    return decoder,samples


def test_chat_boundaries_shift_and_frozen_gradient(setup):
    decoder,samples=setup
    vision=[torch.randn(n,32,requires_grad=True) for n in (7,13)]
    batch=decoder.prepare(samples,vision)
    assert decoder.image_id>=decoder.lm.config.vocab_size
    assert not (batch.labels==decoder.image_id).any()
    assert (batch.labels[batch.attention_mask==0]==-100).all()
    total,count=decoder.loss(batch)
    reference=decoder.lm(inputs_embeds=batch.inputs_embeds,attention_mask=batch.attention_mask,
        position_ids=batch.position_ids,labels=batch.labels,use_cache=False).loss
    torch.testing.assert_close(total/count,reference,rtol=1e-6,atol=1e-6)
    grad=torch.autograd.grad(total/count,vision,retain_graph=True)
    refgrad=torch.autograd.grad(reference,vision)
    for a,b in zip(grad,refgrad):
        torch.testing.assert_close(a,b,rtol=1e-5,atol=1e-7)
        assert a.abs().sum()>0
    assert all(p.grad is None and not p.requires_grad for p in decoder.lm.parameters())
    assert count==sum(decoder.answer_count(s) for s in samples)


def test_padding_and_generation_cache(setup):
    decoder,samples=setup
    decoder.eval()
    vision=[torch.randn(n,32,requires_grad=True) for n in (7,13)]
    combined=decoder.loss(decoder.prepare(samples,vision))[0]
    separate=sum(decoder.loss(decoder.prepare([s],[v]))[0] for s,v in zip(samples,vision))
    torch.testing.assert_close(combined,separate,rtol=1e-6,atol=1e-5)
    batch=decoder.generate(samples,vision)
    solo=[decoder.generate([s],[v])[0] for s,v in zip(samples,vision)]
    assert [x['token_ids'] for x in batch]==[x['token_ids'] for x in solo]
    # Verify our explicit KV-cache loop against the installed HF generate API.
    inputs=decoder.prepare([samples[0]],[vision[0]],training=False)
    ids=decoder.lm.generate(inputs_embeds=inputs.inputs_embeds,attention_mask=inputs.attention_mask,
        do_sample=False,num_beams=1,max_new_tokens=4,eos_token_id=decoder.stop_ids,pad_token_id=decoder.pad_id,
        use_cache=True)
    assert ids[0].tolist()==solo[0]['token_ids']

    # Public multi-item calls must never send padded/mixed lengths to the LM.
    seen=[]
    hook=decoder.lm.register_forward_pre_hook(lambda module,args,kwargs: seen.append(
        (kwargs.get('inputs_embeds',kwargs.get('input_ids')).shape[0],
         bool(kwargs['attention_mask'].all()))),with_kwargs=True)
    try:
        decoder.train()
        decoder.generate(samples,vision)
        assert decoder.training
        assert seen and all(size==1 and full for size,full in seen)
        with pytest.raises(ValueError):decoder.generate(samples,vision,max_new_tokens=0)
    finally:hook.remove()


def test_reject_target_control_tokens_and_over_capacity(setup):
    decoder,samples=setup
    with pytest.raises(ValueError):
        decoder.text_ids(replace(samples[0],finding='<|eot_id|>bad'),True)
    decoder.context_capacity=10
    with pytest.raises(ValueError):
        decoder.prepare([samples[0]],[torch.zeros(5,32)])


def test_text_only_control_requires_explicit_generation_flag(setup):
    decoder,samples=setup
    empty=[torch.empty(0,32)]
    with pytest.raises(ValueError):decoder.prepare(samples[:1],empty)
    with pytest.raises(ValueError):decoder.prepare(samples[:1],empty,training=False)
    batch=decoder.prepare(samples[:1],empty,training=False,allow_empty_visual=True)
    assert batch.lengths[0]==len(decoder.text_ids(samples[0],False)[0])-1
    output=decoder.generate(samples[:1],empty,allow_empty_visual=True)
    assert len(output)==1 and output[0]['generated_token_count']>0
