import copy
import random
from types import SimpleNamespace
import numpy as np
import torch
import pytest
from torch import nn
from src.config import load_config
from src.training.loss import backward_scale
from src.training.optim import build_optimizer,build_scheduler
from src.training.checkpoint import save_checkpoint,load_checkpoint,rng_state,restore_rng


def tiny_model():
    m=nn.Module();m.config=load_config()
    m.projector=nn.Linear(3,5)
    m.encoder=nn.Linear(3,3);m.encoder.requires_grad_(False);m.encoder.lora_targets=[]
    m.decoder=nn.Linear(5,2);m.decoder.requires_grad_(False)
    return m


def test_unequal_token_accumulation_equivalence():
    torch.manual_seed(8)
    m=nn.Linear(4,7);other=copy.deepcopy(m)
    xs=[torch.randn(3,4),torch.randn(9,4)];ys=[torch.randint(0,7,(n,)) for n in (3,9)]
    torch.nn.functional.cross_entropy(m(torch.cat(xs)),torch.cat(ys)).backward()
    for x,y in zip(xs,ys):
        (torch.nn.functional.cross_entropy(other(x),y,reduction='sum')*backward_scale(12)).backward()
    for a,b in zip(m.parameters(),other.parameters()):torch.testing.assert_close(a.grad,b.grad)
    # Algebraic DDP check: two ranks, unequal token counts, averaging once.
    ranks=[copy.deepcopy(m),copy.deepcopy(m)]
    for r,x,y in zip(ranks,xs,ys):
        r.zero_grad();(torch.nn.functional.cross_entropy(r(x),y,reduction='sum')*backward_scale(12,2)).backward()
    for i,p in enumerate(m.parameters()):
        avg=sum(list(r.parameters())[i].grad for r in ranks)/2
        torch.testing.assert_close(p.grad,avg)


def test_checkpoint_resume_weights_optimizer_and_rng(tmp_path):
    torch.manual_seed(4);m=tiny_model()
    optimizer=build_optimizer(m,m.config);scheduler=build_scheduler(optimizer,10,.1)
    x=torch.randn(3,3)
    m.projector(x).sum().backward();optimizer.step();scheduler.step()
    expected=m.projector(x).detach().clone()
    path=tmp_path/'checkpoint.pt';p={'contract_id':'same'}
    save_checkpoint(path,m,optimizer,scheduler,{'step':1},p)
    expected_rng=torch.rand(4)
    with torch.no_grad():m.projector.weight.add_(20)
    load_checkpoint(path,m,'resume',optimizer,scheduler,p)
    torch.testing.assert_close(m.projector(x),expected,rtol=0,atol=0)
    torch.testing.assert_close(torch.rand(4),expected_rng,rtol=0,atol=0)
    assert len(optimizer.state)>0 and scheduler.last_epoch==1
    m.config.training.deterministic=True
    with pytest.raises(ValueError,match='determinism policy'):
        load_checkpoint(path,m,'resume',optimizer,scheduler,p)


def test_stage_b_initialization_requires_matching_dataset_scope(tmp_path):
    stage_a=tiny_model()
    path=tmp_path/'stage_a.pt'
    save_checkpoint(path,stage_a,None,None,{'step':0},{'contract_id':'same'})
    stage_b=tiny_model()
    stage_b.config=load_config(['configs/training/stage_b_lora.yaml'],
                               overrides=['training.feature_cache=false'])
    stage_b.config.data.datasets=['isles']
    with pytest.raises(ValueError,match='dataset scope differs'):
        load_checkpoint(path,stage_b,'initialize')
