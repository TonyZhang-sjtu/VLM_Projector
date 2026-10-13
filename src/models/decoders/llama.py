"""Frozen Llama with explicit image substitution and exact answer-only causal CE."""
import copy
from contextlib import nullcontext
import torch
from torch.nn import functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from src.config import path_of
from src.models.decoders.base import DecoderAdapter, DecoderBatch
from src.tasks.single_sequence_findings import build_task


class LlamaDecoder(DecoderAdapter):
    def __init__(self, config, model=None, tokenizer=None):
        super().__init__()
        if config.decoder.name!='llama':raise ValueError('Register a decoder adapter before selecting another decoder name')
        self.config=config
        asset=path_of(config,config.decoder.path)
        self.tokenizer=copy.deepcopy(tokenizer) if tokenizer is not None else AutoTokenizer.from_pretrained(asset,local_files_only=True)
        self.lm=model if model is not None else AutoModelForCausalLM.from_pretrained(asset,local_files_only=True,
            torch_dtype=torch.bfloat16,attn_implementation=config.decoder.attention,low_cpu_mem_usage=True)
        self.lm.requires_grad_(False)
        self.hidden_size=self.lm.config.hidden_size
        self.context_capacity=self.lm.config.max_position_embeddings
        self.task=build_task(config)
        self.tokenizer.add_special_tokens({'additional_special_tokens':['<image>']})
        self.image_id=self.tokenizer.convert_tokens_to_ids('<image>')
        self.eot_id=self.tokenizer.convert_tokens_to_ids('<|eot_id|>')
        self.pad_id=self.tokenizer.convert_tokens_to_ids('<|finetune_right_pad_id|>')
        names=['<|eot_id|>','<|end_of_text|>','<|eom_id|>']
        self.stop_ids=[self.tokenizer.convert_tokens_to_ids(n) for n in names]
        for name,idx in zip(names,self.stop_ids):
            if self.tokenizer.convert_ids_to_tokens(idx)!=name:
                raise ValueError(f'Unrecognized stop token: {name}')
        if self.pad_id>=self.lm.config.vocab_size:
            raise ValueError('Pad token outside frozen decoder vocabulary')
        self.tokenizer.pad_token_id=self.pad_id
        if config.decoder.gradient_checkpointing:
            self.lm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
        self.lm.config.use_cache=False

    def text_ids(self,sample,training):
        messages=self.task.messages(sample)
        prefix=self.tokenizer.apply_chat_template(messages,tokenize=True,add_generation_prompt=True)
        if prefix.count(self.image_id)!=1:
            raise ValueError('Exactly one image placeholder is required')
        if not training:
            return prefix,len(prefix)
        target=self.task.target(sample)
        full=self.tokenizer.apply_chat_template(messages+[{'role':'assistant','content':target}],
                                                tokenize=True,add_generation_prompt=False)
        if full[:len(prefix)]!=prefix or full[-1]!=self.eot_id or full.count(self.image_id)!=1:
            raise ValueError('Chat boundary mismatch; refusing approximate answer offsets')
        if any(t>=self.lm.config.vocab_size for t in full[len(prefix):]):
            raise ValueError('Target token outside frozen decoder vocabulary')
        return full,len(prefix)

    def answer_count(self,sample):
        ids,boundary=self.text_ids(sample,True)
        return len(ids)-boundary

    def prepare(self,samples,visual,training=True,allow_empty_visual=False):
        if not samples or len(samples)!=len(visual):
            raise ValueError('Sample/visual batch mismatch')
        embedding=self.lm.get_input_embeddings()
        device=embedding.weight.device
        values,labels,lengths,prompt_counts=[],[],[],[]
        for sample,vision in zip(samples,visual):
            ids,boundary=self.text_ids(sample,training)
            index=ids.index(self.image_id)
            # Exclude the added token BEFORE lookup; never resize frozen embeddings.
            safe=torch.tensor(ids[:index]+ids[index+1:],device=device)
            text=embedding(safe)
            if vision.ndim!=2 or vision.shape[1]!=self.hidden_size or (len(vision)<1 and (training or not allow_empty_visual)):
                raise ValueError('Invalid projected visual embeddings')
            vision=vision.to(device=device,dtype=text.dtype)
            value=torch.cat((text[:index],vision,text[index:]))
            length=len(value)
            reserve=0 if training else self.config.generation.max_new_tokens
            if length+reserve>self.context_capacity:
                raise ValueError(f'{sample.id}: {length}+{reserve} exceeds decoder context {self.context_capacity}')
            target=torch.full((length,),-100,dtype=torch.long,device=device)
            if training:
                start=boundary-1+len(vision)
                target[start:]=torch.tensor(ids[boundary:],device=device)
            values.append(value);labels.append(target);lengths.append(length)
            prompt_counts.append(boundary-1)
        longest=max(lengths)
        padded,masked,targets=[],[],[]
        for value,target,length in zip(values,labels,lengths):
            gap=longest-length
            pad=value.new_zeros((gap,self.hidden_size))
            ignore=target.new_full((gap,),-100)
            mask=torch.cat((torch.ones(length,device=device,dtype=torch.long),torch.zeros(gap,device=device,dtype=torch.long)))
            if training:
                padded.append(torch.cat((value,pad)));targets.append(torch.cat((target,ignore)))
            else:
                padded.append(torch.cat((pad,value)));targets.append(torch.cat((ignore,target)))
                mask=mask.roll(gap)
            masked.append(mask)
        mask=torch.stack(masked)
        positions=(mask.cumsum(-1)-1).clamp_min(0)
        return DecoderBatch(torch.stack(padded),mask,positions,torch.stack(targets),lengths,prompt_counts)

    def loss(self,batch):
        # Keep the frozen transformer differentiable w.r.t. visual embeddings.
        hidden=self.lm.model(inputs_embeds=batch.inputs_embeds,attention_mask=batch.attention_mask,
                             position_ids=batch.position_ids,use_cache=False,return_dict=True).last_hidden_state
        # Shift exactly once. Only answer predecessors need vocabulary projection;
        # projecting all ~9k visual positions would waste several GB on logits.
        targets=batch.labels[:,1:]
        valid=targets!=-100
        selected=hidden[:,:-1][valid]
        targets=targets[valid]
        if not len(targets):raise ValueError('No supervised answer tokens')
        total=selected.sum()*0
        chunk=self.config.decoder.ce_chunk_size
        for start in range(0,len(targets),chunk):
            logits=self.lm.lm_head(selected[start:start+chunk]).float()
            total=total+F.cross_entropy(logits,targets[start:start+chunk],reduction='sum')
        return total,len(targets)

    @torch.no_grad()
    def generate(self,samples,visual,max_new_tokens=None,allow_empty_visual=False):
        if not samples or len(samples)!=len(visual):
            raise ValueError('Sample/visual batch mismatch')
        # Decode each complete volume separately: local 8B SDPA left-padding
        # produced NaNs for mixed lengths. Keep the public list interface stable.
        return [self._generate_one([sample],[vision],max_new_tokens,allow_empty_visual)[0]
                for sample,vision in zip(samples,visual)]

    @torch.no_grad()
    def _generate_one(self,samples,visual,max_new_tokens=None,allow_empty_visual=False):
        was_training=self.training
        self.eval()
        try:
            batch=self.prepare(samples,visual,training=False,allow_empty_visual=allow_empty_visual)
            budget=self.config.generation.max_new_tokens if max_new_tokens is None else max_new_tokens
            if not isinstance(budget,int) or not 1<=budget<=self.config.generation.max_new_tokens:
                raise ValueError('Generation budget exceeds validated context reservation')
            mask=batch.attention_mask
            output=self.lm(inputs_embeds=batch.inputs_embeds,attention_mask=mask,
                position_ids=batch.position_ids,use_cache=True,logits_to_keep=1,return_dict=True)
            cache=output.past_key_values
            done=torch.zeros(len(samples),device=mask.device,dtype=torch.bool)
            token_lists=[[] for _ in samples]
            reasons=['length']*len(samples)
            for step in range(budget):
                if not torch.isfinite(output.logits[~done,-1]).all():
                    raise FloatingPointError('Nonfinite active generation logits; refusing argmax of NaNs')
                next_ids=output.logits[:,-1].argmax(-1)
                active=~done
                for i in range(len(samples)):
                    if active[i]:
                        token=int(next_ids[i]);token_lists[i].append(token)
                        if token in self.stop_ids:
                            done[i]=True;reasons[i]='eos'
                if bool(done.all()) or step+1==budget:break
                mask=torch.cat((mask,active.long()[:,None]),dim=1)
                positions=(mask.sum(-1)-1)[:,None]
                next_ids=torch.where(active,next_ids,torch.full_like(next_ids,self.pad_id))
                output=self.lm(input_ids=next_ids[:,None],past_key_values=cache,attention_mask=mask,
                    position_ids=positions,use_cache=True,logits_to_keep=1,return_dict=True)
                cache=output.past_key_values
            results=[]
            for ids,reason,prompt in zip(token_lists,reasons,batch.prompt_token_counts):
                text=self.tokenizer.decode(ids,skip_special_tokens=True,clean_up_tokenization_spaces=False).strip()
                results.append({'prediction':text,'generated_token_count':len(ids),'token_ids':ids,
                    'prompt_token_count':prompt,'stop_reason':reason,'hit_generation_limit':reason=='length',
                    'empty_prediction':not bool(text),'decoding_batch_mode':'independent'})
            return results
        finally:
            self.train(was_training)
