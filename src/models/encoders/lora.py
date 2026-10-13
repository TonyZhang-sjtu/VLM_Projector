import torch
from peft import LoraConfig, get_peft_model


def attach_visual_lora(core, config):
    targets = [f'blocks.{i}.mixer.{part}' for i in range(len(core.blocks)) for part in ('qkv','proj')]
    modules = dict(core.named_modules())
    if len(targets) != 24 or not all(isinstance(modules.get(n),torch.nn.Linear) for n in targets):
        raise ValueError('Local NeuroVFM LoRA requires precisely 24 known attention linear layers')
    adapted = get_peft_model(core,LoraConfig(r=config.rank,lora_alpha=config.alpha,
        lora_dropout=config.dropout,target_modules=targets,bias=config.bias,use_rslora=False))
    expected = sum(config.rank*(modules[n].in_features+modules[n].out_features) for n in targets)
    actual = sum(p.numel() for p in adapted.parameters() if p.requires_grad)
    if actual != expected or any('lora_' not in n for n,p in adapted.named_parameters() if p.requires_grad):
        raise ValueError('LoRA trainable parameters do not match target contract')
    return adapted, targets
