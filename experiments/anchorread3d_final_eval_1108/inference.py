"""Frozen-checkpoint inference helpers with isolated writes and explicit proofs."""
from copy import deepcopy
from pathlib import Path
import math
import time

from .common import ROOT, OUT, read, rows, write
from src.evaluation.clinical import file_digest, digest


def localize(c):
    changes={}
    for section,key in [('data','root'),('data','annotation_root'),('data','manifest_dir'),
                        ('encoder','path'),('decoder','path'),('preprocessing','cache_dir'),
                        ('training','feature_cache_dir')]:
        value=c[section].get(key)
        if not value:continue
        p=Path(value)
        if not p.is_absolute():mapped=ROOT/p
        elif '/assets/' in str(p):mapped=ROOT/'assets'/str(p).split('/assets/',1)[1]
        else:mapped=p
        if str(mapped)!=str(p):changes[f'{section}.{key}']=[str(p),str(mapped)]
        c[section][key]=str(mapped)
    c.project_root=str(ROOT)
    c.training.device='cuda:0';c.training.distributed='none'
    if c.training.get('projector_device') not in (None,'cpu'):c.training.projector_device=None
    return changes


def asset_hashes(directory):
    from filelock import FileLock
    memo_path=OUT/'asset_hashes.json';memo_path.parent.mkdir(parents=True,exist_ok=True)
    with FileLock(str(memo_path)+'.lock'):
        memo=read(memo_path) if memo_path.exists() else {}
        result={}
        for p in sorted(Path(directory).iterdir()):
            if p.suffix not in ('.json','.safetensors','.bin','.model'):continue
            st=p.stat();signature=[st.st_size,st.st_mtime_ns];old=memo.get(str(p),{})
            sha=old.get('sha256') if old.get('stat')==signature else file_digest(p)
            memo[str(p)]=dict(stat=signature,sha256=sha);result[p.name]=sha
        write(memo_path,memo)
    return result


class OverlayCache:
    """Original preprocessing cache is read-only; new cache entries isolated."""
    def __init__(self,original,version):
        from src.preprocessing.cache import TensorCache
        self.original=original;self.added=TensorCache(OUT/'preprocessing'/version)
    def get(self,key):
        value=self.original.get(key)
        return self.added.get(key) if value is None else value
    def put(self,key,value):self.added.put(key,value)


def prepare(e,output,load_model=True):
    import torch
    from omegaconf import OmegaConf
    from src.config import validate
    from src.data.manifest import load_manifests
    from src.preprocessing.registry import build_preprocessor
    from src.models.multimodal import MultimodalModel,place_model
    from src.training.checkpoint import load_checkpoint,projector_contract
    from src.training.trainer import seed_all
    from src.config import digest as config_digest
    pp=e['paths'];rr=rows(pp['predictions']);m=read(pp['metric'])
    assert file_digest(pp['best'])==m['checkpoint_sha256'],'Checkpoint bytes changed'
    saved=torch.load(pp['best'],map_location='cpu',weights_only=False,mmap=True)
    c=OmegaConf.create(saved['config']);changes=localize(c);validate(c)
    splits,summary=load_manifests(c,datasets=c.evaluation.datasets)
    lookup={s.id:s for s in splits['test']};assert set(lookup)=={r['id'] for r in rr}
    samples=[lookup[r['id']] for r in rr]
    for sample,r in zip(samples,rr):
        assert sample.finding==r['reference'] and sample.patient_id==r['patient_id'] and sample.sequence==r['sequence']
    expected=saved['provenance'];equiv=read(ROOT/'results/anchorread3d_analysis_1108/manifest_path_equivalence.json')
    for split,s in summary.items():
        recorded=expected['manifest_hashes'][split]
        assert s['sha256']==recorded or (s['sha256']==equiv[split]['local_hash'] and recorded==equiv[split]['marlowe_path_only_hash']),('manifest changed',split)
    pre=build_preprocessor(c);pre.cache=OverlayCache(pre.cache,pre.version)
    assert pre.version==expected['preprocessing']
    assert asset_hashes(c.decoder.path)==expected['decoder_files'],'Decoder bytes changed'
    proof=dict(setting=e['key'],checkpoint_sha256=m['checkpoint_sha256'],path_remappings=changes,
               manifest_hashes={k:s['sha256'] for k,s in summary.items()},saved_manifest_hashes=expected['manifest_hashes'],
               preprocessing=pre.version,decoder_files=expected['decoder_files'],samples=len(samples))
    write(output/'preflight.json',proof)
    if not load_model:return None,c,pre,samples,rr,proof
    seed_all(c.seed);model=place_model(MultimodalModel(c),'cuda:0')
    assert model.encoder.fingerprint==expected['encoder_base']
    assert model.decoder.task.fingerprint==expected['prompt']
    assert model.decoder.image_id==expected['tokenizer_image_id']
    assert projector_contract(c.projector)==expected['projector']
    # All data/model fields above are independently validated; relocated paths
    # can change the manifest-byte contract. Never fabricate a provenance hash.
    load_checkpoint(pp['best'],model)
    model.eval();model.requires_grad_(False)
    del saved
    return model,c,pre,samples,rr,proof


def ce_only(e,limit=None):
    import torch
    from src.evaluation.ce_validation import compare_ce_history
    from src.evaluation.significance_io import row_identity,attach_ce
    out=OUT/'ce'/e['key'].replace('/','__');out.mkdir(parents=True,exist_ok=True)
    if (out/'summary.json').exists() and not limit:
        print('ALREADY_COMPLETE',e['key'],flush=True);return
    if not limit:
        from .significance import ce_catalog
        metric=read(e['paths']['metric']);original=rows(e['paths']['predictions'])
        audit=dict(checkpoint_sha256=metric['checkpoint_sha256'],test_identity_sha256=row_identity(original),
                   original_metrics=metric['overall'],original_ce_aggregation=read(Path(e['paths']['predictions']).parent/'metrics.json')['answer_ce'])
        candidate=dict(rows=original,vectors={},audit=audit)
        for old in ce_catalog().get(metric['checkpoint_sha256'],[]):
            if attach_ce(candidate,old,policy='recompute'):
                write(OUT/'ce_reuse'/f"{e['key'].replace('/','__')}.json",dict(setting=e['key'],source=str(old),
                      samples_sha256=file_digest(old/'samples.jsonl'),replay=candidate['audit']['ce_replay']))
                print('REUSED_VALIDATED_CE',e['key'],old,flush=True);return
    start=time.monotonic();model,c,pre,samples,rr,proof=prepare(e,out)
    path=out/('smoke_samples.jsonl' if limit else 'samples.jsonl.tmp')
    records=[]
    with path.open('w') as f,torch.no_grad():
        for i,s in enumerate(samples[:limit]):
            with torch.autocast('cuda',dtype=torch.bfloat16):result=model([dict(sample=s,volume=pre(s))])
            loss=float(result['loss_sum']);tokens=int(result['answer_tokens'])
            assert math.isfinite(loss) and loss>=0 and tokens>0
            record=dict(id=s.id,patient_id=s.patient_id,case_id=s.case_id,cohort=s.cohort,sequence=s.sequence,
                        loss_sum=loss,answer_tokens=tokens,reference_sha256=digest(s.finding),checkpoint_sha256=proof['checkpoint_sha256'])
            records.append(record)
            import json
            f.write(json.dumps(record)+'\n');f.flush()
            if (i+1)%40==0 or i==0:print('CE',e['key'],i+1,len(samples),round(time.monotonic()-start,1),flush=True)
    if limit:
        write(out/'smoke.json',dict(samples=len(records),finite=True,seconds=time.monotonic()-start));return
    history=read(Path(e['paths']['predictions']).parent/'metrics.json')['answer_ce']
    comparison=compare_ce_history(records,history)
    write(out/'historical_comparison.json',comparison)
    path.replace(out/'samples.jsonl')
    write(out/'summary.json',dict(status='complete',checkpoint_sha256=proof['checkpoint_sha256'],
          test_identity_sha256=row_identity(rr),samples_sha256=file_digest(out/'samples.jsonl'),
          answer_ce=comparison['loss_sum']/comparison['answer_tokens'],historical_aggregation=history,
          historical_comparison=comparison,ce_policy='strict' if comparison['all_match'] else 'recompute',
          score_basis='teacher_forced_replay',seconds=time.monotonic()-start,
          runtime_environment=dict(torch=torch.__version__,gpu=torch.cuda.get_device_name(0))))
    print('CE_DONE',e['key'],comparison['overall'],flush=True)


def main():
    import argparse
    from .common import entries
    p=argparse.ArgumentParser();p.add_argument('--setting',required=True);p.add_argument('--limit',type=int);p.add_argument('--dry-run',action='store_true');args=p.parse_args()
    _,es=entries();e=es[args.setting]
    if args.dry_run:prepare(e,OUT/'preflight'/args.setting.replace('/','__'),False)
    else:ce_only(e,args.limit)


if __name__=='__main__':main()
