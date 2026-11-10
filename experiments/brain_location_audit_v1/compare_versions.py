"""Audit legacy text, graph-v1 and graph-v2 on the SAME frozen predictions.

Scores remain automatic report-text agreement, not validated extractor accuracy.
Also report fixed-reference-label intersections so denominator changes stay clear.
"""
import argparse
from collections import Counter,defaultdict
from functools import lru_cache
import csv
import json
from pathlib import Path

from src.evaluation.location import direct_location
from src.evaluation.location_reference import reference_location
from .protocol import label_sets

ROOT=Path(__file__).resolve().parents[2]
DIMS={'side':('side',),'site':('site',),'joint':('side','site')}


def rows(path):
    return [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]


def labels_old(value,dim):
    fields=DIMS[dim]
    if any(value[f] is None for f in fields):return set()
    return {tuple(value[f] for f in fields)} if dim=='joint' else {value[fields[0]]}


def aggregate(scores):
    eligible=[s for s in scores if s['eligible']]
    n=len(eligible);tp=sum(s.get('tp',0) for s in eligible)
    fp=sum(s.get('fp',0) for s in eligible);fn=sum(s.get('fn',0) for s in eligible)
    return dict(n=n,correct=sum(s['exact'] for s in eligible),
        exact=sum(s['exact'] for s in eligible)/n if n else None,
        micro_f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False)
    full=json.loads((ROOT/'results/radmatch_comparison_1108/full_inputs/comparison_inputs.json').read_text())
    groups=defaultdict(dict)
    for r in full:groups[r['setting']][r['id']]=r
    index=json.loads((ROOT/'results/anchorread3d_analysis_1108/artifact_reconciliation_1108.json').read_text())
    entries={e['key']:e for e in index['paper_settings'] if e['key'].startswith('rg/')}
    historical=json.loads((ROOT/'results/anchorread3d_analysis_1108/analysis.json').read_text())
    historical={e['key']:e.get('location',{}).get('overall') for e in historical['entries']}
    truth_cached=lru_cache(maxsize=None)(reference_location)
    summaries=[];common_summaries=[];changes=[];coverage={};examples=defaultdict(list)
    for key in sorted(groups):
        current=groups[key]
        original=rows(entries[key]['paths']['predictions'])
        v1={r['id']:r for r in rows(ROOT/'results/brain_location_audit_1108_v1/draft_graph_diagnostics'/(key.replace('/','_')+'.jsonl'))}
        reference_v1={r['id']:r['extraction'] for r in rows(ROOT/'results/brain_location_audit_1108_v1/reference_candidates.jsonl')}
        accum=defaultdict(list);common=defaultdict(list)
        coverage_ids=defaultdict(set)
        for row in original:
            rid=row['id'];g=current[rid];first=v1[rid]
            truth=truth_cached(row['reference'],row.get('sequence',''),row.get('text_audit_status',''))
            pred=direct_location(row['prediction'])
            old={d:dict(eligible=all(truth[f] is not None for f in fs),
                        exact=int(all(truth[f]==pred[f] for f in fs))) for d,fs in DIMS.items()}
            versions={'legacy':old,'graph_v1':first['scores'],'graph_v2':g['graph_scores']}
            rlabels={'graph_v1':label_sets(reference_v1[rid]),'graph_v2':label_sets(g['reference_extraction'])}
            for ver,data in versions.items():
                for dim,score in data.items():
                    accum[(ver,dim)].append(score)
                    if score['eligible']:coverage_ids[(ver,dim)].add(rid)
            for pair in [('legacy','graph_v2'),('graph_v1','graph_v2')]:
                va,vb=pair
                for dim in DIMS:
                    la=labels_old(truth,dim) if va=='legacy' else rlabels[va][dim]
                    lb=rlabels[vb][dim]
                    if versions[va][dim]['eligible'] and versions[vb][dim]['eligible'] and la==lb:
                        common[(va,vb,dim,'a')].append(versions[va][dim])
                        common[(va,vb,dim,'b')].append(versions[vb][dim])
            change=dict(setting=key,id=rid)
            changed=False
            for dim in DIMS:
                for ver,data in versions.items():
                    change[ver+'_'+dim+'_eligible']=data[dim]['eligible']
                    change[ver+'_'+dim+'_exact']=data[dim].get('exact') if data[dim]['eligible'] else None
                changed |= len({(data[dim]['eligible'],data[dim].get('exact') if data[dim]['eligible'] else None) for data in versions.values()})>1
            if changed:changes.append(change)
            tags=[]
            if first['scores']['joint']['eligible'] and g['graph_scores']['joint']['eligible'] and first['scores']['joint']['exact']==0 and g['graph_scores']['joint']['exact']==1:
                tags.append('v1_v2_joint_correction')
            if old['site']['eligible'] and old['site']['exact']==0 and g['graph_scores']['site']['eligible'] and g['graph_scores']['site']['exact']==1 and pred['site'] is None:
                tags.append('legacy_site_miss_new_match')
            if old['joint']['eligible'] and old['joint']['exact']==1 and g['graph_scores']['joint']['eligible'] and g['graph_scores']['joint']['exact']==0:
                tags.append('legacy_match_new_disagreement')
            if not old['joint']['eligible'] and g['graph_scores']['joint']['eligible'] and len(rlabels['graph_v2']['joint'])>1:
                tags.append('new_multilocation_reference')
            if key in ['rg/A0/r32','rg/A0/r256','rg/A1/r32','rg/A1/r256','rg/AvgPool3D/r256']:
                for tag in tags:
                    if len(examples[tag])<30:
                        examples[tag].append(dict(setting=key,id=rid,reference=row['reference'],prediction=row['prediction'],
                            legacy_reference=truth,legacy_prediction=pred,
                            graph_v1_prediction=first['prediction_graph'],graph_v2_reference=g['reference_extraction'],
                            graph_v2_prediction=g['prediction_extraction'],scores=versions))
        for (ver,dim),data in accum.items():
            summary=dict(setting=key,version=ver,dimension=dim,**aggregate(data))
            if ver=='legacy':
                summary['micro_f1']=None
                h=historical.get(key)
                if h:
                    name={'side':'laterality','site':'site','joint':'side_site'}[dim]
                    assert (summary['n'],summary['correct'])==(h[name]['n'],h[name]['correct'])
            summaries.append(summary)
            c=dict(n=len(coverage_ids[(ver,dim)]),patients=len({r['patient_id'] for r in original if r['id'] in coverage_ids[(ver,dim)]}))
            if (ver,dim) in coverage:assert coverage[(ver,dim)]==c
            coverage[(ver,dim)]=c
        for va,vb in [('legacy','graph_v2'),('graph_v1','graph_v2')]:
            for dim in DIMS:
                sa=aggregate(common[(va,vb,dim,'a')]);sb=aggregate(common[(va,vb,dim,'b')])
                assert sa['n']==sb['n']
                common_summaries.append(dict(setting=key,from_version=va,to_version=vb,dimension=dim,
                    n=sa['n'],old_correct=sa['correct'],new_correct=sb['correct'],old_exact=sa['exact'],new_exact=sb['exact'],
                    exact_delta=sb['exact']-sa['exact'] if sa['n'] else None))
    def savecsv(name,data):
        with (a.output/name).open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
    savecsv('all_versions.csv',summaries);savecsv('same_reference_labels.csv',common_summaries);savecsv('changed_cases.csv',changes)
    (a.output/'examples.json').write_text(json.dumps(examples,ensure_ascii=False,indent=2)+'\n')
    cov=[dict(version=v,dimension=d,**c) for (v,d),c in coverage.items()]
    savecsv('coverage.csv',cov)
    import openpyxl
    from .prepare import add_sheet
    wb=openpyxl.Workbook();wb.remove(wb.active)
    add_sheet(wb,'Interpretation',[dict(note='Automatic text agreement only. Different denominators across versions; use Same_reference_labels for controlled comparisons. No clinician gold accuracy yet.')])
    add_sheet(wb,'Coverage',cov);add_sheet(wb,'All_versions',summaries);add_sheet(wb,'Same_reference_labels',common_summaries)
    add_sheet(wb,'Examples',[dict(example_type=t,setting=r['setting'],id=r['id'],reference=r['reference'],prediction=r['prediction'],
        old_labels=r['legacy_prediction'],new_labels=r['graph_v2_prediction'],scores=r['scores']) for t,rs in examples.items() for r in rs])
    wb.save(a.output/'location_versions_comparison.xlsx')
    print(json.dumps(dict(coverage=cov,settings=len(groups),changed_case_count=len(changes)),indent=2))
    for r in summaries:
        if r['setting'] in ['rg/A0/r32','rg/A0/r256','rg/A1/r32','rg/A1/r256','rg/AvgPool3D/r256','rg/Perceiver_NV/r256']:
            print(r)


if __name__=='__main__':main()
