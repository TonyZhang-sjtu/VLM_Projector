import json
import random
import os
import tempfile
from pathlib import Path
import numpy as np
import torch
from omegaconf import OmegaConf
from src.config import file_hash,digest,path_of,ROOT
from src.data.datasets import ALL_DATASETS, normalize_datasets
from src.preprocessing.cache import atomic_torch_save


def rng_state():
    return {'python':random.getstate(),'numpy':np.random.get_state(),'torch':torch.get_rng_state(),
            'cuda':torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def restore_rng(state):
    random.setstate(state['python']);np.random.set_state(state['numpy']);torch.set_rng_state(state['torch'])
    if state['cuda'] is not None:torch.cuda.set_rng_state_all(state['cuda'])


def provenance(config,model,preprocessor,manifest_summary):
    # Hash each actual weight shard once per unchanged file stat; subsequent runs
    # reuse that SHA-256, rather than relying only on model directory names.
    memo_path=path_of(config,'runs/.asset_hashes.json')
    memo=json.loads(memo_path.read_text()) if memo_path.exists() else {}
    files={}
    asset=path_of(config,config.decoder.path)
    for path in sorted(asset.iterdir()):
        if path.suffix not in ('.json','.safetensors','.bin','.model'):continue
        stat=path.stat();key=str(path.resolve())
        signature=[stat.st_size,stat.st_mtime_ns]
        old=memo.get(key,{})
        sha=old.get('sha256') if old.get('stat')==signature else file_hash(path)
        memo[key]={'stat':signature,'sha256':sha};files[path.name]=sha
    memo_path.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.NamedTemporaryFile(mode='w',dir=memo_path.parent,suffix='.json',delete=False) as handle:
        temporary=handle.name;json.dump(memo,handle,indent=2)
    os.replace(temporary,memo_path)
    content={'encoder_base':model.encoder.fingerprint,'decoder_files':files,
        'preprocessing':preprocessor.version,'projector':OmegaConf.to_container(config.projector,resolve=True),
        'target_version':config.data.target_version,'manifest_hashes':{k:v['sha256'] for k,v in manifest_summary.items()},
        'prompt':model.decoder.task.fingerprint,'tokenizer_image_id':model.decoder.image_id,
        'encoder_execution':'independent','decoder_loss':'answer_only_causal_ce_sum',
        'source_files':{str(p.relative_to(ROOT)):file_hash(p) for p in sorted((ROOT/'src').rglob('*.py'))}}
    # Source snapshot is retained separately. Resume/data compatibility checks use
    # the numerical/data contract, permitting diagnostic code additions.
    content['contract_id']=digest({k:v for k,v in content.items() if k!='source_files'})
    content['runtime']={'deterministic':config.training.deterministic,
                        'decoder_attention':config.decoder.attention,'generation_batch_mode':config.generation.batch_mode}
    return content


def save_checkpoint(path,model,optimizer,scheduler,state,provenance_info,rng_states=None):
    from src.models.projectors.budget import require_resolved
    require_resolved(model.config.projector)
    joint=model.config.training.stage=='joint_lora'
    encoder={name:p.detach().cpu().clone() for name,p in model.encoder.named_parameters()
             if ('lora_' in name if joint else p.requires_grad)}
    checkpoint={'format_version':2 if joint else 1,'stage':model.config.training.stage,
        'config':OmegaConf.to_container(model.config,resolve=True),'provenance':provenance_info,
        'projector':{k:v.detach().cpu() for k,v in model.projector.state_dict().items()},
        'encoder':encoder,'lora_targets':model.encoder.lora_targets,
        'optimizer':optimizer.state_dict() if optimizer else None,
        'scheduler':scheduler.state_dict() if scheduler else None,'state':state,
        'rng_states':rng_states or [rng_state()]}
    atomic_torch_save(checkpoint,path)


def load_checkpoint(path,model,mode='evaluate',optimizer=None,scheduler=None,expected_provenance=None,rank=0):
    # Load only project-generated checkpoints from trusted local paths; optimizer
    # and NumPy RNG records require the full local pickle format.
    checkpoint=torch.load(path,map_location='cpu',weights_only=False)
    if mode not in ('evaluate','initialize','resume'):raise ValueError('Unknown checkpoint loading mode')
    joint=model.config.training.stage=='joint_lora'
    expected_format=2 if checkpoint.get('stage')=='joint_lora' else 1
    if checkpoint.get('format_version')!=expected_format:raise ValueError('Unknown checkpoint format')
    # M/budget/head semantics can differ even when all weight shapes coincide.
    # Enforce them also for callers that do not supply a provenance contract.
    from src.models.projectors.specs import resolve_projector_config
    saved_projector=OmegaConf.create(checkpoint['config']['projector'])
    current_projector=OmegaConf.create(OmegaConf.to_container(model.config.projector,resolve=True))
    resolve_projector_config(saved_projector);resolve_projector_config(current_projector)
    from src.models.projectors.budget import require_resolved
    require_resolved(saved_projector);require_resolved(current_projector)
    if OmegaConf.to_container(saved_projector)!=OmegaConf.to_container(current_projector):
        raise ValueError('Checkpoint projector configuration mismatch')
    if current_projector.name=='token_select3d' and current_projector.params.strategy=='random':
        saved_seed=saved_projector.params.selection_seed
        current_seed=current_projector.params.selection_seed
        if saved_seed is None: saved_seed=checkpoint['config']['seed']
        if current_seed is None: current_seed=model.config.seed
        if saved_seed!=current_seed:
            raise ValueError('Checkpoint random-selection seed mismatch')
    for key in ('name','attention','weight_dtype','quantization','frozen'):
        if checkpoint['config']['decoder'][key]!=model.config.decoder[key]:
            raise ValueError(f'Checkpoint decoder setting mismatch: {key}')
    if expected_provenance and checkpoint['provenance']['contract_id']!=expected_provenance['contract_id']:
        raise ValueError('Checkpoint data/model/prompt/preprocessing contract mismatch')
    if mode=='initialize':
        if checkpoint['stage']!='a' or model.config.training.stage not in ('b_lora','b_full'):
            raise ValueError('Stage B initialization must branch from a Stage A checkpoint')
        saved_scope = normalize_datasets(checkpoint['config'].get('data', {}).get('datasets'),
                                         default=ALL_DATASETS)
        current_scope = normalize_datasets(model.config.data.get('datasets'), default=['brats'])
        if saved_scope != current_scope:
            raise ValueError(f'Stage B dataset scope differs from Stage A: {saved_scope} vs {current_scope}')
    elif checkpoint['stage']!=model.config.training.stage:
        raise ValueError('Resume/evaluation stage does not match checkpoint')
    if joint:
        from src.training.schedule import JointSchedule
        if checkpoint['config']['training']['lora']!=OmegaConf.to_container(model.config.training.lora):
            raise ValueError('Checkpoint LoRA configuration mismatch')
        plan=JointSchedule.from_dict(checkpoint['state']['schedule'])
        completed=checkpoint['state']['global_step']
        if not 0<=completed<=plan.total_steps or checkpoint['state']['phase']!=plan.phase(completed):
            raise ValueError('Checkpoint phase/global_step mismatch')
        if mode=='resume':
            if scheduler is None or scheduler.joint_plan!=plan.to_dict():
                raise ValueError('Resume changed joint schedule')
            if checkpoint['scheduler']['last_epoch']!=completed:
                raise ValueError('Checkpoint scheduler/global_step mismatch')
            for key in ('projector_lr','encoder_lr','projector_warmup_steps','joint_warmup_steps',
                        'projector_warmup_ratio','joint_warmup_ratio','max_epochs','max_steps',
                        'early_stopping_min_delta','early_stopping_patience'):
                if checkpoint['config']['training'][key]!=model.config.training[key]:
                    raise ValueError(f'Resume changed training setting: {key}')
            if optimizer is None or [g['param_names'] for g in optimizer.param_groups]!=[
                    g['param_names'] for g in checkpoint['optimizer']['param_groups']]:
                raise ValueError('Resume changed optimizer parameter groups')
        model.encoder.set_phase(checkpoint['state']['phase'])
    model.projector.load_state_dict(checkpoint['projector'],strict=True)
    if mode!='initialize':
        params=dict(model.encoder.named_parameters())
        expected={n for n,p in params.items() if ('lora_' in n if joint else p.requires_grad)}
        if set(checkpoint['encoder'])!=expected:raise ValueError('Visual trainable state mismatch')
        if checkpoint['lora_targets']!=model.encoder.lora_targets:raise ValueError('LoRA target mismatch')
        with torch.no_grad():
            for n,value in checkpoint['encoder'].items():
                if value.shape!=params[n].shape:raise ValueError(f'Visual parameter shape mismatch: {n}')
                params[n].copy_(value.to(params[n].device))
    if mode=='resume':
        if checkpoint['config']['training'].get('deterministic',False)!=model.config.training.deterministic:
            raise ValueError('Cannot change determinism policy during resume; start an explicit new experiment')
        if optimizer is None or scheduler is None:raise ValueError('Resume requires optimizer/scheduler')
        optimizer.load_state_dict(checkpoint['optimizer']);scheduler.load_state_dict(checkpoint['scheduler'])
        if rank>=len(checkpoint['rng_states']):raise ValueError('Resume requires the original world size')
        restore_rng(checkpoint['rng_states'][rank])
    return checkpoint
