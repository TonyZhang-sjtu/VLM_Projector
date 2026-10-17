from src.training.schedule import PROJECTOR_ONLY_PHASES


def trainable_report(model):
    report={}
    for component in ('encoder','projector','decoder'):
        module=getattr(model,component)
        entries=[{'name':name,'numel':p.numel(),'dtype':str(p.dtype)}
                 for name,p in module.named_parameters() if p.requires_grad]
        report[component]={'parameters':sum(x['numel'] for x in entries),'tensors':entries}
    if report['decoder']['parameters']:
        raise ValueError('Decoder must stay frozen')
    return report


def assert_stage(model):
    report=trainable_report(model)
    stage=model.config.training.stage
    count=report['encoder']['parameters']
    if stage=='a' and count!=0:raise ValueError('Stage A encoder is not frozen')
    if stage=='b_lora' and count==0:raise ValueError('LoRA stage has no trainable visual parameters')
    if stage=='b_lora' and any('lora_' not in x['name'] for x in report['encoder']['tensors']):
        raise ValueError('Non-adapter vision parameter enabled during LoRA training')
    if stage=='b_full' and count!=model.encoder.base_parameter_count:
        raise ValueError('Stage Full has frozen visual parameters')
    if stage=='joint_lora':
        if model.encoder.phase in PROJECTOR_ONLY_PHASES:
            if count:raise ValueError('Projector-only phases must freeze visual LoRA')
        elif count==0:
            raise ValueError('Joint phase has no trainable visual adapters')
        if any('lora_' not in x['name'] for x in report['encoder']['tensors']):
            raise ValueError('Joint phase enabled non-adapter vision parameters')
    return report


def wrap_distributed(model,device_ids=None):
    from torch.nn.parallel import DistributedDataParallel
    if model.config.training.stage!='joint_lora':
        return DistributedDataParallel(model,device_ids=device_ids,broadcast_buffers=False)
    # Register both parameter groups once, before restoring the frozen phase.
    # find_unused_parameters marks adapters unused while vision runs in no_grad;
    # their hooks remain registered when phase two enables them. No reducer rebuild
    # or optimizer recreation is needed at a boundary.
    phase=model.encoder.phase
    model.encoder.set_phase('joint_warmup')
    try:
        return DistributedDataParallel(model,device_ids=device_ids,
                                       broadcast_buffers=False,find_unused_parameters=True)
    finally:
        model.encoder.set_phase(phase)
