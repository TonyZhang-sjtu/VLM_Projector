import math
import torch


def build_optimizer(model,config):
    groups=[]
    for name,lr in [('projector',config.training.projector_lr),('encoder',config.training.encoder_lr)]:
        decay,no_decay=[],[]
        for pname,p in getattr(model,name).named_parameters():
            pending_adapter=config.training.stage=='joint_lora' and name=='encoder' and 'lora_' in pname
            if not (p.requires_grad or pending_adapter):continue
            (no_decay if p.ndim<2 or pname.endswith('bias') else decay).append(p)
        for params,wd in [(decay,config.training.weight_decay),(no_decay,0.)]:
            if params:groups.append({'params':params,'lr':lr,'weight_decay':wd,'component':name})
    names={id(p):name for name,p in model.named_parameters()}
    for group in groups:
        group['param_names']=[names[id(p)] for p in group['params']]
    return torch.optim.AdamW(groups,betas=tuple(config.training.betas),eps=config.training.eps,foreach=False)


def build_scheduler(optimizer,total_steps,warmup_ratio,warmup_steps=None):
    warmup=int(warmup_steps) if warmup_steps is not None else int(total_steps*warmup_ratio)
    if warmup>=total_steps:
        raise ValueError(f'Warmup steps ({warmup}) must be fewer than total optimizer steps ({total_steps})')
    def scale(step):
        if step<warmup:return (step+1)/max(1,warmup)
        progress=min(1.,max(0.,(step-warmup)/max(1,total_steps-warmup)))
        return .5*(1+math.cos(math.pi*progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer,scale)


def build_joint_scheduler(optimizer,plan):
    # All groups exist from the start. Frozen adapters have grad=None, so AdamW
    # neither creates their moments nor applies weight decay during phase one.
    factors=[lambda step,component=group['component']:plan.scale(step,component)
             for group in optimizer.param_groups]
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,factors)
    scheduler.joint_plan=plan.to_dict()
    return scheduler
