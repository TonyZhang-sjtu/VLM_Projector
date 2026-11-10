from dataclasses import replace
from pathlib import Path
import pytest
import torch
from transformers import LlamaConfig,LlamaForCausalLM,AutoTokenizer
from src.config import load_config,path_of
from src.models.decoders.llama import LlamaDecoder
from src.data.manifest import load_manifests
from .visual import donor_maps,text_training_batch
from .queue import available_gpus,launchable_gpus


def test_patient_derangement_shared_across_sequences_and_methods():
    rr=[dict(id=f'{p}-{s}',patient_id=str(p),sequence=s,cohort='gli',input_patch_count=100,preprocessing_version='fixed')
        for p in range(12) for s in ['T1W','T1C','T2W','FLAIR']]
    maps=donor_maps(rr);assert maps==donor_maps(list(reversed(rr)))
    for mapping in maps.values():
        assert len(mapping)==48 and len(set(mapping.values()))==48
        for p in range(12):
            donors={mapping[f'{p}-{s}'].split('-')[0] for s in ['T1W','T1C','T2W','FLAIR']}
            assert len(donors)==1 and str(p) not in donors
    assert len({tuple(sorted(m.items())) for m in maps.values()})==3


def test_text_ce_matches_direct_language_model_with_exact_answer_mask():
    c=load_config(overrides=['decoder.gradient_checkpointing=false','generation.max_new_tokens=4'])
    tokenizer=AutoTokenizer.from_pretrained(path_of(c,c.decoder.path),local_files_only=True)
    lm=LlamaForCausalLM(LlamaConfig(vocab_size=128256,hidden_size=32,intermediate_size=64,
        num_hidden_layers=1,num_attention_heads=4,num_key_value_heads=2,max_position_embeddings=2048,attention_dropout=0.))
    decoder=LlamaDecoder(c,lm,tokenizer).eval()
    sample=load_manifests(c)[0]['test'][0]
    batch=text_training_batch(decoder,sample)
    ids,boundary=decoder.text_ids(sample,True);image=ids.index(decoder.image_id)
    safe=ids[:image]+ids[image+1:]
    target=torch.tensor([safe]);labels=target.clone();labels[:,:boundary-1]=-100
    expected=lm(input_ids=target,labels=labels).loss
    actual,n=decoder.loss(batch)
    torch.testing.assert_close(actual/n,expected,rtol=1e-6,atol=1e-6)
    assert batch.lengths[0]==len(safe) and n==len(ids)-boundary
    assert batch.attention_mask.all() and batch.position_ids.tolist()==[list(range(len(safe)))]


def test_free_gpu_policy_excludes_training_and_reserves_one_if_more_than_two():
    s=[dict(index=str(i),has_process=False,used=10,free=24000,util=0) for i in range(4)]
    assert available_gpus(s)==['0','1','2']
    s[1]['has_process']=True;s[3]['used']=23000
    assert available_gpus(s)==['0','2']


def test_cpu_loading_workers_still_reserve_their_gpus():
    s=[dict(index=str(i),has_process=False,used=10,free=24000,util=0) for i in range(4)]
    # Two workers assigned 0/2 have not yet appeared in nvidia-smi. If a third
    # card becomes free, it must still be left unused under the user's policy.
    s[3]['has_process']=True
    assert launchable_gpus(s,{'0','2'})==[]
    # Four usable cards allow exactly three workers, leaving one genuinely free.
    s[3]['has_process']=False
    assert launchable_gpus(s,{'0','2'})==['1']
    assert launchable_gpus(s,{'0','1','2'})==[]
    # With only two usable cards, both may run.
    s[1]['has_process']=s[3]['has_process']=True
    assert launchable_gpus(s,{'0'})==['2']
