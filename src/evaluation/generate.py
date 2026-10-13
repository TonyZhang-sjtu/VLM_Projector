import json
import time
from pathlib import Path
import torch
from tqdm.auto import tqdm
from src.evaluation.visual_controls import shuffled_donors


@torch.no_grad()
def predict(model,dataset,output_path,checkpoint_id,batch_size=1,control=None,max_new_tokens=None,position=0):
    if batch_size<1 or control not in (None,'zero','shuffle','text'):raise ValueError('Invalid generation batch/control')
    output_path=Path(output_path);output_path.parent.mkdir(parents=True,exist_ok=True)
    was_training=model.training;model.eval()
    rows=[]
    donors=shuffled_donors(dataset.samples,model.config.seed) if control=='shuffle' else None
    device=next(model.parameters()).device
    try:
        with output_path.open('w') as handle:
            starts=range(0,len(dataset),batch_size)
            for start in tqdm(starts,total=(len(dataset)+batch_size-1)//batch_size,
                              desc=f'generate {control or "normal"}',unit='batch',position=position,
                              leave=position==0,dynamic_ncols=True,mininterval=2):
                indices=list(range(start,min(start+batch_size,len(dataset))))
                items=[dataset[i] for i in indices]
                if donors is not None:
                    items=[{'sample':item['sample'],'volume':dataset[donors[i]]['volume']} for item,i in zip(items,indices)]
                torch.cuda.reset_peak_memory_stats(device);begin=time.perf_counter()
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    generated=model.generate(items,control=control if control in ('zero','text') else None,max_new_tokens=max_new_tokens)
                torch.cuda.synchronize(device)
                elapsed=time.perf_counter()-begin
                for i,item,result in zip(indices,items,generated):
                    s=item['sample'];v=item['volume']
                    # A running legacy MLP job may import this evaluator after
                    # startup; its old model has no count metadata, and M=N.
                    fallback=0 if control=='text' else len(v.patches)
                    actual=result.get('visual_token_count',fallback)
                    row={k:getattr(s,k) for k in ('id','patient_id','case_id','cohort','sequence','split','target_version','text_audit_status')}
                    row.update(reference=s.finding,checkpoint_id=checkpoint_id,
                        preprocessing_version=v.metadata['preprocessing_version'],prompt_version=model.decoder.task.version,
                        input_patch_count=len(v.patches),control=control,
                        visual_input_present=control!='text',
                        image_donor_id=dataset.samples[donors[i]].id if donors is not None else s.id,
                        generation_batch_seconds=elapsed,batch_size=len(items),device=str(device),
                        peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
                        peak_reserved_bytes=torch.cuda.max_memory_reserved(device),**result)
                    row.setdefault('visual_token_count',actual)
                    row.setdefault('projector_token_count',actual)
                    row.setdefault('encoder_token_count',fallback)
                    row.setdefault('total_prefix_token_count',result['prompt_token_count']+actual)
                    rows.append(row);handle.write(json.dumps(row)+'\n');handle.flush()
    finally:model.train(was_training)
    return rows
