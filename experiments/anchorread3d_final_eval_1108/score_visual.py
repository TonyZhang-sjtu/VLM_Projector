"""Deduplicated clinical scoring inputs and official cached RadGraph rewards.

Only text extraction is reused. Each changed reference/prediction pair receives
new rewards. Cached original metrics are keyed by both texts and scorer group.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import sys

from .common import ROOT,OUT,entries,read,rows,write
from src.evaluation.clinical import digest,file_digest

VIS=OUT/'visual'


def pair_key(r):return digest([r['reference'],r['prediction']])


def pool():
    from .visual import write_jsonl
    _,es=entries();plan=read(VIS/'plan.json');allpairs={};bindings=[];cached={};graphs={};conflicts=set()
    for setting in plan['settings']:
        folder=VIS/setting.replace('/','__');complete=read(folder/'complete.json')
        assert complete['status']=='complete'
        for name,sha in complete['outputs'].items():assert file_digest(folder/name)==sha
        e=es[setting];source=Path(e['paths']['metric']).parent/'scoring/clinical_scores.jsonl'
        for r in rows(source):
            key=pair_key(r)
            if key not in cached:cached[key]=r['clinical']
            for side in ('reference','prediction'):
                graph=r['clinical']['radgraph']['details'][f'{side}_graph'];text=r[side]
                if text in graphs and graphs[text]!=graph:conflicts.add(text)
                graphs[text]=graph
        for path in sorted(folder.glob('*/predictions.jsonl')):
            condition=path.parent.name;rr=rows(path)
            assert len(rr)==560
            for r in rr:
                key=pair_key(r)
                if key not in allpairs:allpairs[key]=dict(r,id=key)
                bindings.append(dict(setting=setting,condition=condition,id=r['id'],pair_key=key,
                                     patient_id=r['patient_id'],prediction_path=str(path)))
    for text in conflicts:graphs.pop(text)
    out=VIS/'scoring';out.mkdir(parents=True,exist_ok=True)
    write(out/'cached_original_pairs.json',cached)
    write(out/'original_graphs.json',graphs)
    write(out/'bindings.json',bindings)
    write(out/'pool_manifest.json',dict(unique_pairs=len(allpairs),bindings=len(bindings),
          cached_original_pairs=len(cached),conflicting_graph_texts=list(conflicts),
          pair_ids=sorted(allpairs),source_plan_sha256=file_digest(VIS/'plan.json')))
    # Deterministic shards: no duplicate pair is evaluated on two GPUs.
    pending=[allpairs[k] for k in sorted(allpairs) if k not in cached]
    write_jsonl(out/'pairs.jsonl',[allpairs[k] for k in sorted(allpairs)])
    for i in range(4):
        rr=pending[i::4];write_jsonl(out/f'shard_{i}.jsonl',rr);write(out/f'shard_{i}_ids.json',[r['id'] for r in rr])
    jobs=[];green=[]
    for i in range(4):
        for name,metrics,collection in [('clinical',['rate','bertscore'],jobs),('green',['green'],green)]:
            collection.append(dict(id=f'{name}_{i}',command=[sys.executable,'-m','experiments.anchorread3d_final_eval_1108.clinical_worker',
                '--predictions',str(out/f'shard_{i}.jsonl'),'--expected-ids',str(out/f'shard_{i}_ids.json'),
                '--output-dir',str(out/f'{name}_{i}'),'--cache-dir',str(out/f'cache_{name}_{i}'),
                '--device','cuda:0','--repeats','0','--metrics',*metrics]))
    write(out/'clinical_jobs.json',jobs);write(out/'green_jobs.json',green)
    print('POOL',len(allpairs),'PENDING',len(pending),'BINDINGS',len(bindings),flush=True)


def graph_scores():
    # Activate optional deps before importing NumPy or any clinical scorer.
    from src.evaluation.clinical_runtime import activate,DEFAULT_ASSETS,verify_asset,package_digest
    activate()
    import numpy as np
    from experiments.brain_location_audit_v1.protocol_v2 import extract_graph
    from experiments.brain_location_audit_v1.protocol import score_sets,label_sets
    spec=importlib.util.spec_from_file_location('frozen_official_radgraph_rewards',ROOT/'third_party/evaluation/python_deps/radgraph/rewards.py')
    official=importlib.util.module_from_spec(spec);spec.loader.exec_module(official)
    out=VIS/'scoring';pairs=rows(out/'pairs.jsonl');graphs=read(out/'original_graphs.json');cached=read(out/'cached_original_pairs.json')
    new_path=out/'new_graphs.json'
    if new_path.exists():graphs.update(read(new_path))
    needed=sorted({r[f] for r in pairs for f in ('reference','prediction')}-graphs.keys())
    new_graphs=read(new_path) if new_path.exists() else {}
    if needed:
        import tarfile
        import hashlib
        _,es=entries();source=Path(es['rg/A0/r32']['paths']['metric']).parent/'scoring/scoring_provenance.json'
        expected=read(source)['scorers']['radgraph']
        for name in ('radgraph','cxr-bert'):
            assert verify_asset(DEFAULT_ASSETS/name)==expected['models'][name]
        assert package_digest('radgraph')==expected['upstream_code_sha256']
        # Verify extracted runtime weights without invoking a helper that writes
        # to the shared model assets. The JSON path adaptation is checked by the
        # original frozen config and the actual inference-equivalence probes.
        with tarfile.open(DEFAULT_ASSETS/'radgraph/radgraph-xl.tar.gz') as archive:
            member=next(m for m in archive.getmembers() if m.name.lstrip('./')=='config.json')
            config_spec=json.load(archive.extractfile(member))
            config_spec['dataset_reader']['token_indexers']['bert']['model_name']=str(DEFAULT_ASSETS/'cxr-bert')
            config_spec['model']['embedder']['token_embedders']['bert']['model_name']=str(DEFAULT_ASSETS/'cxr-bert')
            assert read(DEFAULT_ASSETS/'radgraph/runtime/radgraph-xl/config.json')==config_spec
            for member in archive.getmembers():
                if not member.isfile() or member.name.lstrip('./')=='config.json':continue
                h=hashlib.sha256()
                with archive.extractfile(member) as f:
                    for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
                target=DEFAULT_ASSETS/'radgraph/runtime/radgraph-xl'/member.name
                assert file_digest(target)==h.hexdigest(),f'RadGraph runtime weights changed: {target}'
        import torch
        from radgraph import F1RadGraph
        torch.set_num_threads(2);torch.manual_seed(42)
        config=DEFAULT_ASSETS/'radgraph/runtime/radgraph-xl/config.json';before=file_digest(config)
        # Existing runtime is already prepared. Bypass the adapter's runtime
        # preparation helper, which would rewrite the shared config in place.
        scorer=F1RadGraph(model_type='radgraph-xl',reward_level='all',cuda=-1,
                         model_cache_dir=str(DEFAULT_ASSETS/'radgraph/runtime'),
                         tokenizer_cache_dir=str(DEFAULT_ASSETS/'hub'))
        checks=[]
        for text in sorted(graphs,key=lambda s:digest(s))[:16]:
            if not text.strip():continue
            with torch.inference_mode():_,_,_,refs=scorer(refs=[text],hyps=[text])
            assert refs[0]==graphs[text],'CPU RadGraph extraction differs from frozen GPU graph'
            checks.append(digest(text))
        write(out/'radgraph_cpu_equivalence.json',dict(checked=len(checks),text_hashes=checks,all_identical=True))
        for i,text in enumerate(needed):
            if not text.strip():graph=dict(text=text,entities={})
            else:
                with torch.inference_mode():_,_,_,refs=scorer(refs=[text],hyps=[text])
                assert len(refs)==1;graph=refs[0]
            graphs[text]=graph;new_graphs[text]=graph;write(new_path,new_graphs)
            print('NEW_GRAPH',i+1,len(needed),flush=True)
        del scorer
        assert file_digest(config)==before
    results=[];extract={text:extract_graph(graph) for text,graph in graphs.items()}
    names=('radgraph_simple','radgraph_partial','radgraph_complete')
    for i,r in enumerate(pairs):
        values=official.compute_reward(graphs[r['prediction']],graphs[r['reference']],'all')
        values={name:float(value) for name,value in zip(names,values)}
        if r['id'] in cached:
            old=cached[r['id']]['radgraph']['values']
            assert all(np.isclose(values[k],old[k],rtol=0,atol=1e-12) for k in names),(r['id'],values,old)
        labels=label_sets(extract[r['prediction']])
        results.append(dict(pair_key=r['id'],radgraph=values,graph_v2=score_sets(extract[r['reference']],extract[r['prediction']]),
                            predicted_locations={k:sorted(labels[k]) for k in ('side','site','joint')}))
    write(out/'graph_scores.json',results)
    write(out/'graph_complete.json',dict(status='complete',pairs=len(results),new_graph_texts=len(new_graphs),
          scoring='unmodified official RadGraph compute_reward(all); reference and prediction graphs cached by exact text',
          v2='brain_location_graph_audit_v2_contained_relations',clinical_validation=False,
          input_sha256=file_digest(out/'pairs.jsonl'),rewards_sha256=file_digest(ROOT/'third_party/evaluation/python_deps/radgraph/rewards.py'),
          protocol_sources_sha256={name:file_digest(ROOT/'experiments/brain_location_audit_v1'/name) for name in ('protocol.py','protocol_v2.py')}))
    print('GRAPH_DONE',len(results),flush=True)


def verify_clinical(green=False):
    from src.evaluation.significance_io import scorer_contract
    from src.evaluation.clinical import validate_result
    out=VIS/'scoring';plan=read(VIS/'plan.json');_,es=entries()
    e=es[plan['settings'][0]]
    expected=read(Path(e['paths']['metric']).parent/'scoring/scoring_provenance.json')['scorers']
    groups=['green'] if green else ['rate','bertscore']
    name='green' if green else 'clinical';coverage={};scored_ids=set()
    for i in range(4):
        folder=out/f'{name}_{i}';prov=read(folder/'scoring_provenance.json');rr=rows(folder/'clinical_scores.jsonl')
        assert prov['status'] in ('complete','partial','generation_failures')
        assert prov['input_sha256']==file_digest(out/f'shard_{i}.jsonl')
        ids=read(out/f'shard_{i}_ids.json');assert len(rr)==len(ids) and {r['id'] for r in rr}==set(ids)
        assert not (scored_ids & set(ids));scored_ids.update(ids)
        for group in groups:
            assert digest(scorer_contract(prov['scorers'][group]))==digest(scorer_contract(expected[group])),('Scorer differs from normal',group)
        for r in rr:
            for group in groups:
                item=r['clinical'][group];validate_result(group,item)
                assert item['scorer_fingerprint']==digest(prov['scorers'][group])
                status=item['status'];coverage.setdefault(group,{});coverage[group][status]=coverage[group].get(status,0)+1
                # Explicit known protocol limitations are reported. Runtime errors
                # block the next stage, rather than being silently accepted.
                assert status in ('ok','empty_prediction','input_too_long','output_truncated','format_repair_failed','parse_error'),(group,status)
    write(out/f'{name}_verified.json',dict(status='verified',pairs=len(scored_ids),coverage=coverage))
    print('VERIFIED',name,coverage,flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['pool','graphs','verify','verify-green']);a=p.parse_args()
    if a.stage=='pool':pool()
    elif a.stage=='graphs':graph_scores()
    else:verify_clinical(a.stage=='verify-green')


if __name__=='__main__':main()
