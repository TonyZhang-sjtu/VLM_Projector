"""RG visual controls: patient-level derangements, exact prefix interventions.

Cached shuffle reports are valid because the historical RG prompt depends only
on sequence, and decoding is independent and greedy. The donor's scores and
teacher-forced CE are NEVER copied: every target/donor pair is scored anew.
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import time

from .common import ROOT,OUT,VISUAL_METHODS,entries,read,rows,write
from src.evaluation.clinical import digest,file_digest

SEEDS=(20261108,20261109,20261110)


def donor_maps(rr):
    import numpy as np
    by_patient=defaultdict(dict)
    for r in rr:
        assert r['sequence'] not in by_patient[r['patient_id']]
        by_patient[r['patient_id']][r['sequence']]=r
    groups=defaultdict(list)
    for p,v in by_patient.items():
        signature=tuple(sorted((s,r['cohort'],r['input_patch_count'],r['preprocessing_version']) for s,r in v.items()))
        groups[signature].append(p)
    result={}
    for seed in SEEDS:
        rng=np.random.default_rng(seed);mapping={}
        for signature,patients in sorted(groups.items()):
            pp=sorted(patients);assert len(pp)>1,'No compatible other-patient donor'
            # Rejection sampling yields a uniform random derangement in group.
            for _ in range(10000):
                donor=rng.permutation(pp)
                if all(a!=b for a,b in zip(pp,donor)):break
            else:raise ValueError('Derangement sampling failed')
            for p,d in zip(pp,donor):
                for seq,r in by_patient[p].items():mapping[r['id']]=by_patient[d][seq]['id']
        assert len(mapping)==len(rr) and len(set(mapping.values()))==len(rr)
        result[str(seed)]=mapping
    return result


def prepare_plan():
    import yaml
    _,es=entries();out=OUT/'visual';out.mkdir(parents=True,exist_ok=True)
    selected={k:e for k,e in es.items() if k in [f'rg/{m}/r{r}' for m in VISUAL_METHODS for r in (32,256)]}
    canonical=rows(selected['rg/A0/r32']['paths']['predictions']);maps=donor_maps(canonical)
    identity=sorted((r['id'],r['patient_id'],r['sequence'],r['reference'],r['cohort'],r['input_patch_count'],r['preprocessing_version']) for r in canonical)
    grids=set()
    for key,e in selected.items():
        rr=rows(e['paths']['predictions'])
        assert sorted((r['id'],r['patient_id'],r['sequence'],r['reference'],r['cohort'],r['input_patch_count'],r['preprocessing_version']) for r in rr)==identity
        assert all(r['decoding_batch_mode']=='independent' for r in rr)
        config=yaml.safe_load(Path(e['paths']['config']).read_text())
        assert config['generation']['do_sample'] is False and config['generation']['num_beams']==1
        assert config['generation']['batch_mode']=='independent'
        grids.add(tuple(config['projector']['resolved_budget']['input_grid_dhw']))
    assert len(grids)==1,'Donor geometry contracts differ across methods'
    plan=dict(settings=list(selected),seeds=SEEDS,donors=maps,reference_identity=digest(identity),
              source_prediction_hashes={k:file_digest(e['paths']['predictions']) for k,e in selected.items()},
              conditions=['normal','patient_shuffle','zero_prefix','text_only'],input_grid_dhw=list(next(iter(grids))))
    path=out/'plan.json'
    if path.exists():
        previous=read(path)
        if 'input_grid_dhw' not in previous:
            assert not list(out.glob('rg__*/complete.json'))
            assert previous==json.loads(json.dumps({k:v for k,v in plan.items() if k!='input_grid_dhw'}))
            write(out/'plan_before_geometry_check.json',previous);write(path,plan)
        else:assert previous==json.loads(json.dumps(plan))
    else:write(path,plan)
    return plan,selected


def text_training_batch(decoder,sample):
    """Remove the sole visual placeholder from an exact teacher-forced batch.

The production decoder intentionally rejects empty training prefixes. We keep
that guard unchanged and construct this evaluation-only text baseline here.
"""
    import torch
    from src.models.decoders.base import DecoderBatch
    weight=decoder.lm.get_input_embeddings().weight
    one=weight.new_zeros((1,decoder.hidden_size))
    batch=decoder.prepare([sample],[one],training=True)
    ids,boundary=decoder.text_ids(sample,True);index=ids.index(decoder.image_id)
    keep=torch.arange(batch.lengths[0],device=weight.device)!=index
    embeddings=batch.inputs_embeds[:,keep];labels=batch.labels[:,keep]
    mask=torch.ones((1,len(keep)-1),device=weight.device,dtype=torch.long)
    pos=mask.cumsum(-1)-1
    assert labels[labels!=-100].tolist()==ids[boundary:]
    assert int((labels!=-100).sum())==decoder.answer_count(sample)
    return DecoderBatch(embeddings,mask,pos,labels,[int(keep.sum())],[boundary-1])


def prefix_controls(model,c,samples,original,proof):
    import torch
    from filelock import FileLock
    actual={r['visual_token_count'] for r in original};assert len(actual)==1
    m=next(iter(actual));decoder=model.decoder;weight=decoder.lm.get_input_embeddings().weight
    output={}
    for mode,length in [('zero',m),('text',0)]:
        contract=dict(mode=mode,m=length,decoder=proof['decoder_files'],task=decoder.task.fingerprint,
                      generation=dict(c.generation),attention=c.decoder.attention,precision='bfloat16',
                      refs=[(s.id,digest(s.finding),s.sequence) for s in samples])
        key=digest(contract);path=OUT/'visual/prefix_cache'/f'{key}.json';path.parent.mkdir(parents=True,exist_ok=True)
        with FileLock(str(path)+'.lock'):
            if path.exists():
                saved=read(path);assert saved['contract']==json.loads(json.dumps(contract));output[mode]=saved['records'];continue
            generation={};records={};visual=weight.new_zeros((length,decoder.hidden_size))
            assert not torch.count_nonzero(visual)
            for i,s in enumerate(samples):
                with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
                    if s.sequence not in generation:
                        generation[s.sequence]=decoder.generate([s],[visual],allow_empty_visual=mode=='text')[0]
                    batch=(text_training_batch(decoder,s) if mode=='text' else decoder.prepare([s],[visual],training=True))
                    loss,tokens=decoder.loss(batch)
                assert torch.isfinite(loss) and tokens>0
                records[s.id]=dict(**generation[s.sequence],loss_sum=float(loss),answer_tokens=int(tokens),
                                   visual_token_count=length,total_prefix_token_count=generation[s.sequence]['prompt_token_count']+length)
                if i%80==0:print('PREFIX',mode,length,i+1,len(samples),flush=True)
            write(path,dict(contract=contract,records=records));output[mode]=records
    return output


def write_jsonl(path,rr):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix('.jsonl.tmp')
    temp.write_text(''.join(json.dumps(r,allow_nan=False)+'\n' for r in rr));temp.replace(path)


def run(e,limit=None):
    import torch
    from .inference import prepare
    plan=read(OUT/'visual/plan.json');setting=e['key'];out=OUT/'visual'/setting.replace('/','__')
    out.mkdir(parents=True,exist_ok=True)
    if (out/'complete.json').exists() and not limit:
        old=read(out/'complete.json')
        assert old['source_sha256']==file_digest(e['paths']['predictions'])
        for name,sha in old['outputs'].items():assert file_digest(out/name)==sha
        print('VISUAL_ALREADY_COMPLETE',setting,flush=True);return
    start=time.monotonic();model,c,pre,samples,original,proof=prepare(e,out)
    assert c.generation.batch_mode=='independent'
    lookup={s.id:s for s in samples};old={r['id']:r for r in original}
    maps=plan['donors'];inverse={seed:{d:t for t,d in mapping.items()} for seed,mapping in maps.items()}
    model.decoder.eval();outputs={f'shuffle_{seed}':{} for seed in maps};checks=[];normal_ce={}
    selected=samples[:limit] if limit else samples
    partial=out/('smoke_shuffle_ce.jsonl' if limit else 'shuffle_ce_partial.jsonl')
    with partial.open('w') as f,torch.no_grad():
        for i,donor in enumerate(selected):
            volume=pre(donor)
            assert list(volume.metadata['grid_dhw'])==plan['input_grid_dhw']
            with torch.autocast('cuda',dtype=torch.bfloat16):
                projected=model.project_volumes([volume],[donor.sequence])[0]
                visual=projected.embeddings
                assert len(visual)==old[donor.id]['visual_token_count']
                normal_loss,normal_tokens=model.decoder.loss(model.decoder.prepare([donor],[visual],training=True))
                assert torch.isfinite(normal_loss) and normal_tokens==model.decoder.answer_count(donor)
                normal_ce[donor.id]=dict(loss_sum=float(normal_loss),answer_tokens=int(normal_tokens))
                for seed,inv in inverse.items():
                    target=lookup[inv[donor.id]]
                    assert donor.patient_id!=target.patient_id and donor.sequence==target.sequence
                    assert model.decoder.task.messages(donor)==model.decoder.task.messages(target)
                    # One real within-runtime check per sequence/checkpoint. Compare
                    # target vs donor decoding using identical visual evidence.
                    if seed==str(SEEDS[0]) and donor.sequence not in {r['sequence'] for r in checks}:
                        a=model.decoder.generate([donor],[visual])[0]
                        b=model.decoder.generate([target],[visual])[0]
                        assert a['token_ids']==b['token_ids'],'Swap changes sequence-only prompt semantics'
                        checks.append(dict(sequence=donor.sequence,donor=donor.id,target=target.id,
                                           same_prompt_tokens=True,same_generated_tokens=True,
                                           historical_token_agreement=a['token_ids']==old[donor.id]['token_ids']))
                    batch=model.decoder.prepare([target],[visual],training=True)
                    loss,tokens=model.decoder.loss(batch)
                    assert torch.isfinite(loss) and tokens==model.decoder.answer_count(target)
                    record=dict(id=target.id,donor_id=donor.id,seed=int(seed),loss_sum=float(loss),answer_tokens=int(tokens))
                    outputs[f'shuffle_{seed}'][target.id]=record
                    f.write(json.dumps(record)+'\n');f.flush()
            if i%40==0:print('SHUFFLE_CE',setting,i+1,len(selected),round(time.monotonic()-start,1),flush=True)
    write(out/'swap_equivalence_checks.json',checks)
    if limit:
        prefixes=prefix_controls(model,c,samples[:4],original[:4],proof)
        s=samples[0]
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
            for mode in ('zero','text'):
                direct=model.generate([dict(sample=s,volume=pre(s) if mode=='zero' else None)],control=mode)[0]
                assert direct['token_ids']==prefixes[mode][s.id]['token_ids'],mode
                assert direct['visual_token_count']==prefixes[mode][s.id]['visual_token_count']
        write(out/'smoke.json',dict(status='passed',samples=len(selected),checks=checks,prefix_modes=list(prefixes),
                                   zero_and_text_equal_production_controls=True));return
    assert len(checks)==4
    prefixes=prefix_controls(model,c,samples,original,proof)
    normal_rows=[dict(r,**normal_ce[r['id']],ce_measurement='same_process_as_visual_controls') for r in original]
    write_jsonl(out/'normal/predictions.jsonl',normal_rows)
    from src.evaluation.ce_validation import compare_ce_history
    history=read(Path(e['paths']['predictions']).parent/'metrics.json')['answer_ce']
    write(out/'normal_ce_comparison.json',compare_ce_history(normal_rows,history))
    for condition,ce_records in outputs.items():
        seed=condition.split('_')[1];rr=[]
        for r in original:
            d=old[maps[seed][r['id']]]
            nr=dict(r)
            for key in ('prediction','token_ids','generated_token_count','stop_reason','hit_generation_limit','empty_prediction'):
                nr[key]=d[key]
            nr.update(control='shuffle',image_donor_id=d['id'],donor_patient_id=d['patient_id'],shuffle_seed=int(seed),
                      generation_source='cached_normal_donor_sequence_only_prompt',generation_source_id=d['id'],
                      loss_sum=ce_records[r['id']]['loss_sum'],answer_tokens=ce_records[r['id']]['answer_tokens'])
            # Old timing/memory belongs to a different generation and is not measured here.
            for key in ('generation_batch_seconds','peak_allocated_bytes','peak_reserved_bytes'):nr.pop(key,None)
            rr.append(nr)
        write_jsonl(out/condition/'predictions.jsonl',rr)
    for mode,record_map in prefixes.items():
        rr=[]
        for r in original:
            nr=dict(r);nr.update(record_map[r['id']])
            nr.update(control=mode,image_donor_id=None,visual_input_present=False,
                      encoder_token_count=0,projector_token_count=nr['visual_token_count'],
                      generation_source='frozen_decoder_explicit_zero_or_empty_prefix')
            for key in ('generation_batch_seconds','peak_allocated_bytes','peak_reserved_bytes'):nr.pop(key,None)
            rr.append(nr)
        write_jsonl(out/mode/'predictions.jsonl',rr)
    paths=sorted(out.glob('*/predictions.jsonl'));assert len(paths)==6
    for path in paths:
        rr=rows(path);assert len(rr)==560 and {r['id'] for r in rr}==set(old)
        assert all(r['reference']==old[r['id']]['reference'] for r in rr)
    write(out/'complete.json',dict(status='complete',setting=setting,seconds=time.monotonic()-start,
          source_sha256=file_digest(e['paths']['predictions']),plan_sha256=file_digest(OUT/'visual/plan.json'),
          outputs={str(p.relative_to(out)):file_digest(p) for p in paths}))
    print('VISUAL_DONE',setting,flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--prepare',action='store_true');p.add_argument('--setting');p.add_argument('--limit',type=int);args=p.parse_args()
    if args.prepare:prepare_plan();return
    _,es=entries();run(es[args.setting],args.limit)


if __name__=='__main__':main()
