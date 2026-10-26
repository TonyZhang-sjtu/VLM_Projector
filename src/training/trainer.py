import json
import math
import os
import random
import time
from contextlib import nullcontext
from pathlib import Path
import numpy as np
import torch
import torch.distributed as dist
from tqdm.auto import tqdm
from src.config import path_of,save_config,accumulation_steps,digest
from src.data.datasets import ALL_DATASETS, normalize_datasets
from src.data.manifest import load_manifests
from src.data.dataset import VolumeDataset
from src.preprocessing.registry import build_preprocessor
from src.models.multimodal import MultimodalModel
from src.training.optim import build_optimizer,build_scheduler,build_joint_scheduler
from src.training.stages import assert_stage,wrap_distributed
from src.training.schedule import resolve_joint_schedule,update_joint_validation
from src.training.loss import global_answer_tokens,global_sample_count,combined_loss
from src.training.checkpoint import provenance,save_checkpoint,load_checkpoint,rng_state


def seed_all(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)


def distributed_setup(config):
    world=int(os.getenv('WORLD_SIZE','1'))
    if world>1:
        if config.training.distributed!='ddp':raise ValueError('torchrun requires training.distributed=ddp')
        rank=int(os.environ['LOCAL_RANK']);torch.cuda.set_device(rank)
        dist.init_process_group('nccl')
        return dist.get_rank(),world,torch.device('cuda',rank)
    device=torch.device(config.training.device)
    torch.cuda.set_device(device)
    return 0,1,device


def select_training_samples(config, rows, split):
    filename=config.training[f'{split}_ids_file']
    if filename is not None:
        ids=json.loads(path_of(config,filename).read_text())
        lookup={s.id:s for s in rows}
        if not isinstance(ids,list) or not ids or len(ids)!=len(set(ids)) or any(i not in lookup for i in ids):
            raise ValueError('Selection IDs must be unique, nonempty, and belong to the requested split')
        rows=[lookup[i] for i in ids]
    limit=config.training[f'{split}_limit']
    return rows[:limit] if limit else rows


def training_budget(config,sample_count,world_size=1):
    accumulation=accumulation_steps(config,world_size)
    windows=math.ceil(math.ceil(sample_count/world_size)/(config.training.microbatch_per_device*accumulation))
    total=windows*config.training.max_epochs
    if config.training.max_steps:total=min(total,config.training.max_steps)
    return windows,total


def validate(model,dataset,microbatch=1,rank=0,world=1,description='validation'):
    was_training=model.training;model.eval();records=[]
    device=next(model.parameters()).device
    try:
        with torch.no_grad():
            indices=list(range(rank,len(dataset),world))
            # Per-sample CE records give exact deduplication and weighted strata.
            for index in tqdm(indices,desc=description,position=1,leave=False,disable=rank!=0,mininterval=2):
                item=dataset[index]
                with torch.autocast('cuda',dtype=torch.bfloat16):result=model([item])
                if not torch.isfinite(result['loss_sum']):raise FloatingPointError(f'Nonfinite validation loss: {item["sample"].id}')
                records.append({'id':item['sample'].id,'cohort':item['sample'].cohort,
                    'loss_sum':float(result['loss_sum']),'answer_tokens':result['answer_tokens']})
                if 'feature_loss_sum' in result:
                    if not torch.isfinite(result['feature_loss_sum']):raise FloatingPointError('Nonfinite validation reconstruction')
                    records[-1].update(feature_loss=float(result['feature_loss_sum']),
                                       feature_cosine=float(result['feature_cosine_sum']))
        if world>1:
            gathered=[None]*world;dist.all_gather_object(gathered,records)
            records=[r for rows in gathered for r in rows]
        if len(records)!=len(dataset) or len({r['id'] for r in records})!=len(records):
            raise ValueError('Validation coverage or duplicated IDs')
        groups={}
        for row in records:
            group=groups.setdefault(row['cohort'],{'loss_sum':0.,'answer_tokens':0,'samples':0})
            group['loss_sum']+=row['loss_sum'];group['answer_tokens']+=row['answer_tokens'];group['samples']+=1
        total=sum(r['loss_sum'] for r in records);count=sum(r['answer_tokens'] for r in records)
        if count<1:raise ValueError('Empty validation set')
        for group in groups.values():group['answer_ce']=group['loss_sum']/group['answer_tokens']
        metrics={'answer_ce':total/count,'answer_tokens':count,'samples':len(records),'cohorts':groups}
        if records and 'feature_loss' in records[0]:
            metrics.update(feature_loss=sum(r['feature_loss'] for r in records)/len(records),
                           feature_cosine=sum(r['feature_cosine'] for r in records)/len(records))
        return metrics
    finally:model.train(was_training)


class Trainer:
    def __init__(self,config):
        self.config=config
        self.output=path_of(config,config.training.run_dir)
        if (self.output/'train.jsonl').exists() and not config.training.resume:
            raise FileExistsError('Run directory already contains training; use a new run or explicit resume')
        self.rank,self.world,self.device=distributed_setup(config)
        # Keep the data permutation common across ranks, but give stochastic
        # per-rank modules (for example LoRA dropout) independent reproducible RNG.
        seed_all(config.seed+self.rank)
        self.accumulation=accumulation_steps(config,self.world)
        samples,self.manifest_summary=load_manifests(config)
        self.preprocessor=build_preprocessor(config)
        train=select_training_samples(config,samples['train'],'train')
        val=select_training_samples(config,samples['val'],'val')
        from src.models.projectors.budget import prepare_budget
        self.budget_preflight=prepare_budget(config, [*train, *val])
        self.windows_per_epoch,self.total_steps=training_budget(config,len(train),self.world)
        self.plan=resolve_joint_schedule(config.training,self.total_steps) if config.training.stage=='joint_lora' else None
        self.train_data=VolumeDataset(train,self.preprocessor);self.val_data=VolumeDataset(val,self.preprocessor)
        # Fixed cohort/sequence representatives first; then fill in stable order.
        representatives={}
        for sample in val:representatives.setdefault((sample.cohort,sample.sequence),sample)
        chosen=list(representatives.values());seen={s.id for s in chosen}
        chosen += [s for s in val if s.id not in seen]
        self.generation_data=VolumeDataset(chosen[:config.training.validation_generation_samples],self.preprocessor)
        self.model=MultimodalModel(config).to(self.device)
        self.report=assert_stage(self.model)
        self.optimizer=build_optimizer(self.model,config)
        self.scheduler=build_joint_scheduler(self.optimizer,self.plan) if self.plan else build_scheduler(
            self.optimizer,self.total_steps,config.training.warmup_ratio,config.training.warmup_steps)
        self.provenance=provenance(config,self.model,self.preprocessor,self.manifest_summary)
        self.state={'epoch':0,'next_window':0,'global_step':0,'best_val_ce':float('inf'),'bad_epochs':0,
                    'world_size':self.world,'scope':'smoke' if config.training.train_limit or config.training.max_steps or config.training.train_ids_file else 'training',
                    'dataset_scope':list(config.data.datasets),
                    'train_selection':digest([s.id for s in train]),'val_selection':digest([s.id for s in val])}
        if self.plan:
            self.state.update(schedule=self.plan.to_dict(),phase=self.plan.phase(0),
                              early_stop_reference=None,stop_reason=None)
        if config.training.init_checkpoint:
            load_checkpoint(path_of(config,config.training.init_checkpoint),self.model,'initialize',expected_provenance=self.provenance)
        if config.training.stage in ('b_lora','b_full') and not (config.training.init_checkpoint or config.training.resume):
            raise ValueError('Stage B requires a Stage A initialization checkpoint or same-stage resume')
        if config.training.resume:
            old=load_checkpoint(path_of(config,config.training.resume),self.model,'resume',self.optimizer,self.scheduler,self.provenance,self.rank)
            if old['state']['world_size']!=self.world:raise ValueError('World size changed on resume')
            old_scope=old['state'].get('dataset_scope')
            if old_scope is None:
                old_scope=old['config'].get('data',{}).get('datasets')
            old_scope=normalize_datasets(old_scope,default=ALL_DATASETS)
            if old_scope!=self.state['dataset_scope']:
                raise ValueError(f'Resume changed dataset_scope: {old_scope} vs {self.state["dataset_scope"]}')
            for key in ('train_selection','val_selection'):
                if old['state'].get(key)!=self.state[key]:raise ValueError(f'Resume changed {key}')
            for key in ('effective_batch_size','microbatch_per_device','max_epochs','max_steps','train_limit','val_limit',
                        'projector_lr','encoder_lr','warmup_ratio','weight_decay','betas','eps','lora',
                        'warmup_steps','grad_clip_norm','early_stopping_patience'):
                from omegaconf import OmegaConf
                now=OmegaConf.to_container(config.training,resolve=True)[key]
                if old['config']['training'][key]!=now:raise ValueError(f'Resume changed training setting: {key}')
            if old['config']['seed']!=config.seed:raise ValueError('Resume changed seed')
            if old['config']['training'].get('early_stopping_enabled',True)!=config.training.get('early_stopping_enabled',True):
                raise ValueError('Resume changed training setting: early_stopping_enabled')
            self.state=old['state']
            if self.plan:
                self.state['schedule']=self.plan.to_dict()
        self.report=assert_stage(self.model)
        self.wrapped=self.model if self.world==1 else wrap_distributed(self.model,device_ids=[self.device.index])
        if self.rank==0:
            self.output.mkdir(parents=True,exist_ok=True)
            save_config(config,self.output/'resolved_config.yaml')
            if self.budget_preflight:
                (self.output/'budget_preflight.json').write_text(json.dumps(self.budget_preflight,indent=2))
            (self.output/'provenance.json').write_text(json.dumps(self.provenance,indent=2))
            (self.output/'trainable.json').write_text(json.dumps(self.report,indent=2))
            if self.plan:
                (self.output/'schedule.json').write_text(json.dumps({**self.plan.to_dict(),
                    'windows_per_epoch':self.windows_per_epoch,'train_samples':len(train),
                    'projector_peak_lr':config.training.projector_lr,
                    'encoder_peak_lr':config.training.encoder_lr},indent=2))
            self.model.decoder.tokenizer.save_pretrained(self.output/'tokenizer')
            (self.output/'selection.json').write_text(json.dumps({
                'datasets': list(config.data.datasets), 'manifest_summary': self.manifest_summary,
                'train':[s.id for s in train], 'val':[s.id for s in val]},indent=2))

    def save(self,name):
        states=[rng_state()]
        if self.world>1:
            states=[None]*self.world;dist.all_gather_object(states,rng_state())
        if self.rank==0:
            save_checkpoint(self.output/'checkpoints'/name,self.model,self.optimizer,self.scheduler,self.state,self.provenance,states)

    def sync_phase(self):
        if not self.plan:return False
        phase=self.plan.phase(self.state['global_step'])
        changed=phase!=self.model.encoder.phase
        if changed:
            self.optimizer.zero_grad(set_to_none=True)
            self.model.encoder.set_phase(phase)
            self.report=assert_stage(self.model)
            if self.rank==0:
                with (self.output/'phase_transitions.jsonl').open('a') as handle:
                    handle.write(json.dumps({'completed_steps':self.state['global_step'],
                        'next_phase':phase,'trainable':self.report})+'\n')
        self.state['phase']=phase
        return changed

    def train(self):
        c=self.config;micro=c.training.microbatch_per_device
        early_stopping=c.training.get('early_stopping_enabled',True)
        diagnostics_enabled=c.training.get('log_optimizer_diagnostics',False)
        if self.state['global_step']>=self.total_steps or (early_stopping and self.state['bad_epochs']>=c.training.early_stopping_patience):
            return self.state
        log_path=self.output/'train.jsonl'
        start_epoch=self.state['epoch']
        epochs=tqdm(range(start_epoch,c.training.max_epochs),desc=f'{c.training.stage} epochs',
                    position=0,leave=True,disable=self.rank!=0)
        for epoch in epochs:
            order=torch.randperm(len(self.train_data),generator=torch.Generator().manual_seed(c.seed+epoch)).tolist()
            padded=math.ceil(len(order)/self.world)*self.world
            repeats=padded-len(order)
            if repeats:order+=(order*math.ceil(repeats/len(order)))[:repeats]
            local=order[self.rank::self.world]
            window_size=micro*self.accumulation
            windows=[local[i:i+window_size] for i in range(0,len(local),window_size)]
            begin=self.state['next_window'] if epoch==start_epoch else 0
            self.model.train()
            steps=tqdm(range(begin,len(windows)),desc=f'epoch {epoch+1} optimizer steps',
                       position=1,leave=False,disable=self.rank!=0,mininterval=2)
            for windex in steps:
                if self.state['global_step']>=self.total_steps:break
                self.sync_phase()
                used_phase=self.model.encoder.phase if self.plan else c.training.stage
                used_lrs={g['component']:g['lr'] for g in self.optimizer.param_groups}
                indices=windows[windex]
                # Count text tokens before forward, without retaining multiple GPU graphs.
                local_tokens=sum(self.model.decoder.answer_count(self.train_data.samples[i]) for i in indices)
                count=global_answer_tokens(local_tokens,self.device)
                feature_weight=float(c.projector.params.get('feature_loss_weight',0.))
                extra_objectives=getattr(self.model,'uses_auxiliary_objectives',False)
                sample_count=global_sample_count(len(indices),self.device) if feature_weight or extra_objectives else None
                if extra_objectives:self.model.set_supervision_step(self.state['global_step'])
                self.optimizer.zero_grad(set_to_none=True)
                loss_total=0.;encoder_tokens=0;projector_tokens=0
                extra_totals={}
                feature_total=0.;feature_cosine=0.;feature_patches=0
                start=time.perf_counter();torch.cuda.reset_peak_memory_stats(self.device)
                batches=[indices[i:i+micro] for i in range(0,len(indices),micro)]
                for j,ids in enumerate(batches):
                    context=self.wrapped.no_sync() if self.world>1 and j<len(batches)-1 else nullcontext()
                    with context,torch.autocast('cuda',dtype=torch.bfloat16):
                        items=[self.train_data[i] for i in ids]
                        if feature_weight:
                            for item in items:item['feature_sampling_step']=self.state['global_step']
                        result=self.wrapped(items)
                        if not torch.isfinite(result['loss_sum']):raise FloatingPointError('Nonfinite training loss')
                        loss=combined_loss(result,count,self.world,feature_weight,sample_count)
                    loss.backward();loss_total+=float(result['loss_sum'].detach())
                    for key,value in result.get('auxiliary_metrics',{}).items():
                        extra_totals[key]=extra_totals.get(key,0.)+float(value)
                    if feature_weight:
                        if result['feature_samples']!=len(ids):raise ValueError('Incomplete reconstruction supervision')
                        feature_total+=float(result['feature_loss_sum'].detach())
                        feature_cosine+=float(result['feature_cosine_sum'])
                        feature_patches+=result['feature_patch_count']
                    encoder_tokens+=sum(result['encoder_token_counts'])
                    projector_tokens+=sum(result['projector_token_counts'])
                    del loss,result
                parameters=[p for p in self.model.parameters() if p.requires_grad]
                diagnostics=None
                if diagnostics_enabled and self.rank==0:
                    from src.training.diagnostics import OptimizerDiagnostics
                    diagnostics=OptimizerDiagnostics(self.optimizer)
                norm=torch.nn.utils.clip_grad_norm_(parameters,c.training.grad_clip_norm,error_if_nonfinite=True)
                if diagnostics is not None:diagnostics.after_clip()
                self.optimizer.step();self.scheduler.step()
                component_diagnostics=diagnostics.after_step() if diagnostics is not None else None
                del diagnostics
                self.state.update(epoch=epoch,next_window=windex+1,global_step=self.state['global_step']+1)
                summed=torch.tensor([loss_total,float(len(indices)),encoder_tokens,projector_tokens],device=self.device,dtype=torch.float64)
                if self.world>1:dist.all_reduce(summed)
                torch.cuda.synchronize(self.device)
                record={'step':self.state['global_step'],'epoch':epoch,'answer_ce':float(summed[0]/count),
                    'answer_loss_sum':float(summed[0]),
                    'answer_tokens':count,'actual_samples':int(summed[1]),'sampler_repeats_this_epoch':repeats,
                    'encoder_tokens':int(summed[2]),'projector_tokens':int(summed[3]),
                    'mean_encoder_tokens':float(summed[2]/summed[1]),
                    'mean_projector_tokens':float(summed[3]/summed[1]),
                    'projector_name':c.projector.name,
                    'phase':used_phase,'projector_lr':used_lrs['projector'],
                    'encoder_lr':used_lrs.get('encoder',0.0),
                    'grad_norm':float(norm),'seconds':time.perf_counter()-start,
                    'peak_allocated_bytes':torch.cuda.max_memory_allocated(self.device),
                    'peak_reserved_bytes':torch.cuda.max_memory_reserved(self.device)}
                if feature_weight:
                    aux=torch.tensor([feature_total,feature_cosine,feature_patches],device=self.device,dtype=torch.float64)
                    if self.world>1:dist.all_reduce(aux)
                    record.update(feature_loss=float(aux[0]/sample_count),feature_cosine=float(aux[1]/sample_count),
                                  feature_patch_count=int(aux[2]),feature_loss_weight=feature_weight,
                                  total_objective=record['answer_ce']+feature_weight*float(aux[0]/sample_count))
                if c.projector.get('resolved_budget'):
                    budget=c.projector.resolved_budget
                    record.update(target_compression_ratio=budget.target_ratio,
                                  actual_compression_ratio=float(summed[2]/summed[3]),
                                  raw_num_tokens=budget.raw_num_tokens,
                                  expected_output_tokens=budget.expected_output_tokens,
                                  budget_version=budget.version)
                if extra_objectives:
                    # Metrics are sums over samples; missing labels contribute
                    # zero. Eligible-label counts are logged separately.
                    keys=sorted(extra_totals)
                    values=torch.tensor([extra_totals[k] for k in keys],device=self.device,dtype=torch.float64)
                    if self.world>1:dist.all_reduce(values)
                    record['auxiliary']={k:float(v/sample_count) for k,v in zip(keys,values)}
                    record['total_objective']=record.get('total_objective',record['answer_ce'])+record['auxiliary'].get('weighted_loss',0.)
                if self.rank==0:
                    if component_diagnostics is not None:
                        record['optimizer_diagnostics']=component_diagnostics
                        with (self.output/'optimizer_diagnostics.jsonl').open('a') as handle:
                            handle.write(json.dumps({'step':record['step'],'epoch':epoch+1,
                                'phase':used_phase,'learning_rates':used_lrs,
                                'global_grad_norm_pre_clip':float(norm),
                                'clip_threshold':c.training.grad_clip_norm,
                                'components':component_diagnostics})+'\n')
                    with log_path.open('a') as handle:handle.write(json.dumps(record)+'\n')
                    steps.set_postfix(loss=f'{record["answer_ce"]:.4f}',grad=f'{record["grad_norm"]:.3f}',
                                      tok=record['answer_tokens'],mem=f'{record["peak_allocated_bytes"]/2**30:.1f}G')
                if self.sync_phase():
                    self.save(f'boundary_{self.state["global_step"]:08d}.pt')
                    self.save('last.pt')
                if c.training.save_every_steps and self.state['global_step']%c.training.save_every_steps==0:
                    self.save(f'step_{self.state["global_step"]:08d}.pt')
                    self.save('last.pt')
            metrics=validate(self.model,self.val_data,rank=self.rank,world=self.world,
                             description=f'epoch {epoch+1} validation CE')
            if c.training.validation_generation_samples:
                # All ranks wait for rank zero's fixed-subset generation before
                # saving an epoch checkpoint, so RNG/checkpoint collectives align.
                if self.rank==0:
                    from src.evaluation.generate import predict
                    from src.evaluation.metrics import evaluate_rows
                    prediction_path=self.output/'validation_generation'/f'step_{self.state["global_step"]:08d}.jsonl'
                    rows=predict(self.model,self.generation_data,prediction_path,
                                 f'candidate_step_{self.state["global_step"]}',batch_size=1,position=2)
                    generated=evaluate_rows(rows,[s.id for s in self.generation_data.samples],repeats=0)
                    prediction_path.with_suffix('.metrics.json').write_text(json.dumps(generated,indent=2))
                    metrics['fixed_generation_subset']=generated
                if self.world>1:dist.barrier()
            full_epoch=self.state['next_window']>=len(windows)
            if self.plan:
                improved=update_joint_validation(self.state,metrics['answer_ce'],
                    eligible=full_epoch and self.state['global_step']>self.plan.cosine_start,
                    min_delta=c.training.early_stopping_min_delta,patience=c.training.early_stopping_patience,
                    enabled=early_stopping)
                if self.state['stop_reason'] is None and self.state['global_step']>=self.total_steps:
                    self.state['stop_reason']='max_steps' if c.training.max_steps else 'max_epochs'
                    self.state['stop_step']=self.state['global_step']
                metrics.update(early_stopping_enabled=early_stopping,early_stop_reference=self.state['early_stop_reference'],
                               bad_epochs=self.state['bad_epochs'],stop_reason=self.state['stop_reason'])
            else:
                improved=metrics['answer_ce']<self.state['best_val_ce']
                if improved:self.state['best_val_ce']=metrics['answer_ce'];self.state['bad_epochs']=0
                else:self.state['bad_epochs']+=1
            if full_epoch:self.state.update(epoch=epoch+1,next_window=0)
            if self.rank==0:
                if diagnostics_enabled:
                    from src.training.diagnostics import epoch_training_summary
                    summary=epoch_training_summary(log_path,epoch)
                    summary.update(full_epoch=full_epoch,validation_answer_ce=metrics['answer_ce'],
                                   validation_samples=metrics['samples'],
                                   validation_answer_tokens=metrics['answer_tokens'])
                    with (self.output/'epoch_metrics.jsonl').open('a') as handle:
                        handle.write(json.dumps(summary)+'\n')
                with (self.output/'validation.jsonl').open('a') as handle:
                    handle.write(json.dumps({'step':self.state['global_step'],**metrics})+'\n')
                if self.plan:
                    (self.output/'training_state.json').write_text(json.dumps(self.state,indent=2))
            if improved:self.save('best.pt')
            if full_epoch and epoch+1 in c.training.get('checkpoint_epochs',[]):
                self.save(f'epoch_{epoch+1:02d}.pt')
            if self.rank==0:
                epochs.set_postfix(val_ce=f'{metrics["answer_ce"]:.4f}',best=f'{self.state["best_val_ce"]:.4f}',
                                   bad_epochs=f'{self.state["bad_epochs"]}/{c.training.early_stopping_patience}')
            self.save('last.pt')
            if early_stopping and self.state['bad_epochs']>=c.training.early_stopping_patience:break
            if c.training.max_steps and self.state['global_step']>=c.training.max_steps:break
        return self.state
