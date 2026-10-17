import copy
import json
from contextlib import nullcontext
from pathlib import Path
import subprocess

import pytest
import torch
from torch import nn
import torch.distributed as dist
import torch.multiprocessing as mp

from src.config import ROOT, load_config
from src.models.encoders.neurovfm import NeuroVFMEncoder
from src.training.checkpoint import save_checkpoint, load_checkpoint
from src.training.optim import build_optimizer, build_joint_scheduler
from src.training.schedule import JointSchedule, resolve_joint_schedule, update_joint_validation
from src.training.stages import assert_stage, wrap_distributed


class TinyCore(nn.Module):
    def __init__(self):
        super().__init__()
        self.base=nn.Linear(4,4)
        self.lora_A=nn.Parameter(torch.randn(2,4)*.1)
        self.lora_B=nn.Parameter(torch.zeros(4,2))
        self.dropout=nn.Dropout(.05)

    def forward(self,x):
        return self.base(x)+2*(self.dropout(x)@self.lora_A.T@self.lora_B.T)


class TinyEncoder(NeuroVFMEncoder):
    def __init__(self):
        nn.Module.__init__(self)
        self.stage='joint_lora';self.phase='projector_warmup'
        self.core=TinyCore();self.lora_targets=['tiny_attention']
        self.set_phase(self.phase)

    def forward(self,x):
        with torch.no_grad() if self.vision_frozen else nullcontext():
            return self.core(x)


class TinyModel(nn.Module):
    def __init__(self,config=None):
        super().__init__()
        self.config=config or joint_config()
        self.encoder=TinyEncoder()
        self.projector=nn.Linear(4,4)
        self.decoder=nn.Linear(4,2);self.decoder.requires_grad_(False)

    def forward(self,x):
        return self.decoder(self.projector(self.encoder(x))).square().mean()


def joint_config(*overrides):
    return load_config([ROOT/'configs/training/joint_lora.yaml'],overrides)


def initial_state(plan):
    return dict(global_step=0,phase=plan.phase(0),schedule=plan.to_dict(),
                best_val_ce=float('inf'),bad_epochs=0,early_stop_reference=None,
                epoch=0,next_window=0,stop_reason=None)


def advance(model,optimizer,scheduler,state,plan,until):
    trace=[]
    for step in range(state['global_step'],until):
        model.encoder.set_phase(plan.phase(step))
        trace.append([g['lr'] for g in optimizer.param_groups])
        optimizer.zero_grad(set_to_none=True)
        model(torch.randn(3,4)).backward()
        optimizer.step();scheduler.step()
        state.update(global_step=step+1,phase=plan.phase(step+1),next_window=step+1)
        model.encoder.set_phase(state['phase'])
    return trace


def test_full_schedule_real_optimizer_rates():
    c=joint_config();plan=resolve_joint_schedule(c.training,1230)
    assert (plan.projector_steps,plan.joint_steps,plan.cosine_steps)==(77,77,1076)
    model=TinyModel(c);opt=build_optimizer(model,c);sch=build_joint_scheduler(opt,plan)
    trace=[]
    for _ in range(1230):
        rates={g['component']:g['lr'] for g in opt.param_groups};trace.append(rates)
        opt.step();sch.step()
    assert trace[0]=={'projector':0.,'encoder':0.}
    assert trace[76]=={'projector':1e-5,'encoder':0.}
    assert trace[77]=={'projector':0.,'encoder':0.}
    assert trace[153]==trace[154]=={'projector':1e-5,'encoder':1e-5}
    assert trace[-1]=={'projector':0.,'encoder':0.}
    assert all(x['projector']==x['encoder'] for x in trace[77:])
    assert all(trace[i]['projector']>=trace[i+1]['projector'] for i in range(154,1229))


def test_step_override_and_budget_validation():
    c=joint_config('training.projector_warmup_steps=12','training.max_epochs=20',
                   'training.lora.rank=32','training.lora.alpha=64')
    p=resolve_joint_schedule(c.training,2460)
    assert (p.projector_steps,p.joint_steps)==(12,154)
    assert (p.projector_source,p.joint_source)==('steps','ratio')
    assert c.training.lora.rank==32 and c.training.lora.alpha==64
    with pytest.raises(ValueError,match='at least 2'):resolve_joint_schedule(c.training,10)


@pytest.mark.parametrize('setting',[
    'training.projector_warmup_steps=1','training.joint_warmup_steps=1.5',
    'training.projector_warmup_ratio=1','training.joint_warmup_ratio=nan',
    'training.early_stopping_min_delta=-1','training.early_stopping_min_delta=nan',
    'training.lora.alpha=0','training.feature_cache=true','training.warmup_steps=4',
    'training.warmup_ratio=0.05','training.init_checkpoint=old.pt'])
def test_reject_invalid_joint_config(setting):
    with pytest.raises(ValueError):joint_config(setting)


def test_phase_freeze_and_adam_preservation():
    torch.manual_seed(42)
    model=TinyModel();plan=JointSchedule(8,2,2)
    opt=build_optimizer(model,model.config);sch=build_joint_scheduler(opt,plan)
    state=initial_state(plan)
    frozen={n:p.detach().clone() for n,p in model.named_parameters() if 'base' in n or 'decoder' in n}
    adapters={n:p.detach().clone() for n,p in model.encoder.named_parameters() if 'lora_' in n}
    advance(model,opt,sch,state,plan,2)
    for n,p in model.encoder.named_parameters():
        if 'lora_' in n:
            assert torch.equal(p,adapters[n]) and p not in opt.state
    weight=model.projector.weight
    moments=copy.deepcopy(opt.state[weight]);identity=id(opt)
    model.encoder.set_phase('joint_warmup');assert_stage(model)
    assert id(opt)==identity
    for k,v in moments.items():torch.testing.assert_close(v,opt.state[weight][k],rtol=0,atol=0)
    advance(model,opt,sch,state,plan,5)
    assert not torch.equal(model.encoder.core.lora_B,adapters['core.lora_B'])
    for n,p in model.named_parameters():
        if n in frozen:assert torch.equal(p,frozen[n]) and p.grad is None
    assert int(opt.state[weight]['step'])==5
    assert int(opt.state[model.encoder.core.lora_B]['step'])==3


@pytest.mark.parametrize('hold',[0,2])
@pytest.mark.parametrize('cut',[1,2,3,4,5,6,7])
def test_resume_every_phase_and_boundary(tmp_path,cut,hold):
    torch.manual_seed(123)
    c=joint_config(f'training.projector_hold_steps={hold}')
    m=TinyModel(c);plan=JointSchedule(8,2,2,projector_hold_steps=hold)
    opt=build_optimizer(m,c);sch=build_joint_scheduler(opt,plan);state=initial_state(plan)
    advance(m,opt,sch,state,plan,cut)
    update_joint_validation(state,1.3,eligible=cut>plan.cosine_start,min_delta=.005,patience=3)
    path=tmp_path/'step.pt';save_checkpoint(path,m,opt,sch,state,{'contract_id':'test'})
    expected_trace=advance(m,opt,sch,state,plan,8)
    expected={n:p.detach().clone() for n,p in m.named_parameters()}
    expected_opt=copy.deepcopy(opt.state_dict());expected_rng=torch.rand(5)
    restored=TinyModel(c)
    # Base/decoder are immutable pretrained assets in production.
    with torch.no_grad():
        for n,p in restored.named_parameters():
            if 'base' in n or 'decoder' in n:p.copy_(expected[n])
    ropt=build_optimizer(restored,c);rsch=build_joint_scheduler(ropt,plan)
    ckpt=load_checkpoint(path,restored,'resume',ropt,rsch,{'contract_id':'test'})
    assert ckpt['format_version']==2
    assert set(ckpt['encoder'])=={'core.lora_A','core.lora_B'}
    actual_trace=advance(restored,ropt,rsch,ckpt['state'],plan,8)
    assert actual_trace==expected_trace
    for n,p in restored.named_parameters():torch.testing.assert_close(p,expected[n],rtol=0,atol=0)
    for idx,values in ropt.state_dict()['state'].items():
        for key,value in values.items():torch.testing.assert_close(value,expected_opt['state'][idx][key],rtol=0,atol=0)
    torch.testing.assert_close(torch.rand(5),expected_rng,rtol=0,atol=0)


def test_checkpoint_rejects_schedule_and_lora_change(tmp_path):
    m=TinyModel();p=JointSchedule(8,2,2);o=build_optimizer(m,m.config);s=build_joint_scheduler(o,p)
    path=tmp_path/'s.pt';save_checkpoint(path,m,o,s,initial_state(p),{})
    other=TinyModel(joint_config('training.lora.alpha=64'))
    with pytest.raises(ValueError,match='LoRA configuration'):load_checkpoint(path,other)
    with pytest.raises(ValueError,match='joint schedule'):
        load_checkpoint(path,m,'resume',o,build_joint_scheduler(o,JointSchedule(10,2,2)))


def test_early_stop_warmup_cumulative_gain_best_and_resume():
    state=initial_state(JointSchedule(1230,77,77))
    update=lambda ce,eligible: update_joint_validation(state,ce,eligible=eligible,min_delta=.005,patience=3)
    update(1.2,False);update(1.5,False)
    assert state['bad_epochs']==0 and state['early_stop_reference'] is None
    update(1.3,True)
    assert state['best_val_ce']==1.2 and state['early_stop_reference']==1.3
    update(1.298,True);update(1.296,True)
    assert state['bad_epochs']==2 and state['early_stop_reference']==1.3
    update(1.294,True)
    assert state['bad_epochs']==0 and state['early_stop_reference']==1.294
    update(1.293,True)
    state=json.loads(json.dumps(state))
    update(1.292,True);state['global_step']=615;update(1.291,True)
    assert state['bad_epochs']==3 and state['stop_reason']=='early_stopping' and state['stop_step']==615
    with pytest.raises(FloatingPointError):update(float('nan'),True)
    with pytest.raises(FloatingPointError):update(float('inf'),False)


def test_launcher_dry_run_overrides_without_creating_run():
    cmd=['bash',str(ROOT/'scripts/run_full_mlp_uncompressed.sh'),'joint_interface_test',
         '--dry-run','--datasets','brats','--no-use-cache',
         '--set','training.max_epochs=12','--set','training.projector_warmup_steps=19',
         '--set','training.joint_warmup_steps=23','--set','training.lora.rank=32',
         '--set','training.lora.alpha=64']
    result=subprocess.run(cmd,cwd=ROOT,text=True,capture_output=True,check=True)
    report=json.loads(result.stdout)
    assert report['total_steps']==1476 and report['schedule']['cosine_steps']==1434
    assert report['config']['training']['lora']['rank']==32
    assert report['config']['training']['lora']['alpha']==64
    assert report['config']['preprocessing']['cache_enabled'] is False
    assert not (ROOT/'runs/joint_interface_test').exists()


def _ddp_worker(rank,path):
    torch.set_num_threads(1)
    dist.init_process_group('gloo',init_method='file://'+path,rank=rank,world_size=2)
    torch.manual_seed(4+rank)
    m=TinyModel();p=JointSchedule(8,2,2)
    o=build_optimizer(m,m.config);s=build_joint_scheduler(o,p)
    wrapped=wrap_distributed(m)
    initial=m.encoder.core.lora_B.detach().clone()
    def updates(model,wrapped,optimizer,scheduler,start):
        for step in range(start,8):
            model.encoder.set_phase(p.phase(step));optimizer.zero_grad(set_to_none=True)
            for micro in range(2):
                with wrapped.no_sync() if micro==0 else nullcontext():
                    (wrapped(torch.ones(3,4)*(rank+1+micro))*.5).backward()
            if step<2:assert model.encoder.core.lora_B.grad is None
            else:assert model.encoder.core.lora_B.grad is not None
            optimizer.step();scheduler.step()
            model.encoder.set_phase(p.phase(step+1))
            for name,param in model.named_parameters():
                gathered=[torch.empty_like(param) for _ in range(2)]
                dist.all_gather(gathered,param.detach())
                torch.testing.assert_close(gathered[0],gathered[1],rtol=0,atol=0)
            if step==1:
                state=initial_state(p);state.update(global_step=2,phase=p.phase(2))
                save_checkpoint(path+f'.rank{rank}.pt',model,optimizer,scheduler,state,{})
    updates(m,wrapped,o,s,0)
    expected={n:v.detach().clone() for n,v in m.named_parameters()}
    assert not torch.equal(initial,m.encoder.core.lora_B)
    del wrapped
    restored=TinyModel()
    with torch.no_grad():
        for n,param in restored.named_parameters():
            if 'base' in n or 'decoder' in n:param.copy_(expected[n])
    ro=build_optimizer(restored,restored.config);rs=build_joint_scheduler(ro,p)
    load_checkpoint(path+f'.rank{rank}.pt',restored,'resume',ro,rs)
    rw=wrap_distributed(restored)
    updates(restored,rw,ro,rs,2)
    for n,param in restored.named_parameters():torch.testing.assert_close(param,expected[n],rtol=0,atol=0)
    dist.destroy_process_group()


def test_ddp_dynamic_freeze_reduces_new_adapters(tmp_path):
    mp.spawn(_ddp_worker,args=(str(tmp_path/'rendezvous'),),nprocs=2,join=True)


def test_small_best_checkpoint_gain_does_not_reset_patience():
    state=initial_state(JointSchedule(8,2,2))
    assert update_joint_validation(state,1.3,eligible=True,min_delta=.005,patience=3)
    assert update_joint_validation(state,1.298,eligible=True,min_delta=.005,patience=3)
    assert state['best_val_ce']==1.298 and state['early_stop_reference']==1.3 and state['bad_epochs']==1


def test_evaluation_entry_restores_v2_joint_configuration(tmp_path):
    from argparse import Namespace
    from omegaconf import OmegaConf
    from src.cli.evaluate_checkpoint import checkpoint_config
    c=joint_config();path=tmp_path/'joint.pt'
    torch.save(dict(format_version=2,stage='joint_lora',config=OmegaConf.to_container(c)),path)
    args=Namespace(device='cuda:0',split='val',output_dir=tmp_path,metrics=['bleu4'],repeats=0)
    loaded=checkpoint_config(path,args)
    assert loaded.training.stage=='joint_lora' and loaded.training.lora.rank==16
    assert loaded.training.early_stopping_min_delta==.005


@pytest.mark.parametrize('total',[1230,2460])
def test_projector_hold_exact_500_step_boundaries(total):
    plan=JointSchedule(total,77,77,projector_hold_steps=500,
                       joint_warmup_projector_mode='hold_peak',cosine_min_lr_ratio=.2)
    assert plan.joint_start==577 and plan.cosine_start==654
    assert plan.cosine_steps==total-654
    assert plan.phase(76)=='projector_warmup'
    assert plan.phase(77)==plan.phase(576)=='projector_hold'
    assert plan.phase(577)==plan.phase(653)=='joint_warmup'
    assert plan.phase(654)=='joint_cosine'
    for step in range(77,577):
        assert plan.scale(step,'projector')==1.0
        assert plan.scale(step,'encoder')==0.0
    assert plan.scale(577,'projector')==1.0 and plan.scale(577,'encoder')==0.0
    assert plan.scale(653,'encoder')==plan.scale(654,'encoder')==1.0
    assert plan.scale(total-1,'projector')==plan.scale(total-1,'encoder')==.2
    assert JointSchedule.from_dict(plan.to_dict())==plan
    legacy=JointSchedule(total,77,77).to_dict()
    del legacy['projector_hold_steps']
    assert JointSchedule.from_dict(legacy).projector_hold_steps==0


def test_projector_hold_freezes_lora_and_preserves_adam():
    torch.manual_seed(42)
    c=joint_config('training.projector_hold_steps=2')
    model=TinyModel(c);plan=JointSchedule(8,2,2,projector_hold_steps=2)
    opt=build_optimizer(model,c);sch=build_joint_scheduler(opt,plan)
    state=initial_state(plan)
    initial_lora=model.encoder.core.lora_B.detach().clone()
    advance(model,opt,sch,state,plan,2)
    assert model.encoder.phase=='projector_hold'
    assert model.encoder.vision_frozen and not model.encoder.core.training
    assert_stage(model)
    weight=model.projector.weight
    optimizer_identity=id(opt)
    hold_start=weight.detach().clone()
    advance(model,opt,sch,state,plan,4)
    assert id(opt)==optimizer_identity and int(opt.state[weight]['step'])==4
    assert not torch.equal(weight,hold_start)
    assert torch.equal(initial_lora,model.encoder.core.lora_B)
    assert model.encoder.core.lora_B not in opt.state
    assert model.encoder.phase=='joint_warmup' and not model.encoder.vision_frozen
    assert_stage(model)
    advance(model,opt,sch,state,plan,8)
    assert not torch.equal(initial_lora,model.encoder.core.lora_B)
    assert int(opt.state[weight]['step'])==8


@pytest.mark.parametrize('hold',[-1,1.5,True])
def test_invalid_projector_hold_is_rejected(hold):
    with pytest.raises(ValueError,match='projector_hold_steps'):
        JointSchedule(8,2,2,projector_hold_steps=hold)
