"""Patient-paired, unadjusted p-values; all source artifacts remain immutable.

VQA inverse group-size weights exactly reproduce the frozen macro means under
patient label exchange (group memberships and denominators do not change).
Confidence-limit columns are not independent metrics and receive no p-value.
"""
import argparse
from collections import Counter
import csv
from itertools import product
import math
from pathlib import Path
from functools import lru_cache
from types import SimpleNamespace

import numpy as np
from openpyxl import load_workbook
from openpyxl.comments import Comment
from openpyxl.styles import Font

from src.evaluation.clinical import digest, file_digest, validate_pair
from src.evaluation.significance import METRICS, GROUPS, lexical_vectors, statistic
from src.evaluation.significance_io import scorer_contract, row_identity, attach_ce
from src.evaluation.neuroqa import aggregate,score_row
from .common import ROOT, INDEX, OUT, BASELINES, entries, read, rows, write

VQA_METRICS = ('closed_accuracy', 'open_f1', 'open_em', 'open_rouge_l', 'qa_ce',
               'yn_accuracy', 'mcq_accuracy', 'patient_macro_accuracy',
               'modality_macro_accuracy', 'category_macro_accuracy')


@lru_cache(None)
def qa_manifest(config_path):
    import yaml
    c=yaml.safe_load(Path(config_path).read_text());p=Path(c['data']['manifest_dir'])
    if not p.is_absolute():p=ROOT/p
    elif '/assets/' in str(p):p=ROOT/'assets'/str(p).split('/assets/',1)[1]
    elif str(p)=='/home/mingjie/jyzhang/VLM_Projector-local-config/sync-20261031-vqa/manifests/official_holdout_patient_guard_v1_mni_v2':
        p=ROOT/'assets/data/NeuroQA_derived/official_holdout_patient_guard_v1_mni_v2'
    p=p/'test.jsonl';rr=rows(p);lookup={r['neuroqa_id']:r for r in rr}
    assert len(lookup)==len(rr)
    return lookup,dict(path=str(p),sha256=file_digest(p))


def paired_p(a, b, va, vb, metric, repeats=19999, seed=20261108):
    aa, bb = {r['id']: r for r in a}, {r['id']: r for r in b}
    assert len(aa) == len(a) == len(bb) == len(b) and aa.keys() == bb.keys()
    for key in aa:
        for field in ('patient_id', 'reference', 'sequence'):
            assert aa[key][field] == bb[key][field], (key, field)
    assert set(va) <= aa.keys() and set(vb) <= bb.keys()
    ids = sorted(set(va) & set(vb))
    if metric=='answer_ce':
        assert all(va[i][1]==vb[i][1] for i in ids),'Paired reference answer token counts differ'
    patients = sorted({aa[i]['patient_id'] for i in ids})
    assert len(patients) >= 2
    pi = {p: i for i, p in enumerate(patients)}
    x = np.zeros((len(patients), len(va[ids[0]]))); y = np.zeros_like(x)
    for key in ids:
        x[pi[aa[key]['patient_id']]] += va[key]
        y[pi[aa[key]['patient_id']]] += vb[key]
    assert np.isfinite(x).all() and np.isfinite(y).all()
    sa, sb = statistic(metric, x.sum(0)), statistic(metric, y.sum(0))
    observed = sa - sb
    mid, diff = (x+y).sum(0)/2, (x-y)/2
    exact = len(patients) < 21 and 2**len(patients) <= repeats
    n = 2**len(patients) if exact else repeats
    signs_iter = iter(product((-1., 1.), repeat=len(patients))) if exact else None
    rng = np.random.default_rng(seed); extreme = 0
    tolerance = 64*np.finfo(float).eps*max(1, abs(observed))
    for start in range(0, n, 512):
        size = min(512, n-start)
        signs = np.array([next(signs_iter) for _ in range(size)]) if exact else rng.integers(0, 2, (size, len(patients)))*2-1
        shifts = signs @ diff
        if metric == 'bleu4':
            deltas = np.array([statistic(metric, mid+s)-statistic(metric, mid-s) for s in shifts])
        else:
            plus, minus = mid+shifts, mid-shifts
            assert (plus[:, 1] > 0).all() and (minus[:, 1] > 0).all()
            deltas = plus[:, 0]/plus[:, 1] - minus[:, 0]/minus[:, 1]
        extreme += int((np.abs(deltas) >= abs(observed)-tolerance).sum())
    p = extreme/n if exact else (extreme+1)/(n+1)
    return dict(p_raw=p, score_ours=sa, score_baseline=sb, delta_ours_minus_baseline=observed,
                improvement=-observed if metric in ('answer_ce','qa_ce') else observed,
                patients=len(patients), common_valid_samples=len(ids), expected_samples=len(a),
                valid_ours=len(va), valid_baseline=len(vb), excluded_ids=sorted(aa.keys()-set(ids)),
                permutation_mode='exact' if exact else 'monte_carlo', permutations=n, extreme=extreme,
                seed=seed, p_mc_se=0 if exact else math.sqrt(p*(1-p)/(n+1)))


def vqa_vectors(rr):
    vectors = {}
    closed = [r for r in rr if r['question_type'] != 'Open']
    opened = [r for r in rr if r['question_type'] == 'Open']
    for metric, selected, field in [
        ('closed_accuracy', closed, 'correct'), ('open_f1', opened, 'f1'),
        ('open_em', opened, 'em'), ('open_rouge_l', opened, 'rouge_l'),
        ('yn_accuracy', [r for r in closed if r['question_type']=='YN'], 'correct'),
        ('mcq_accuracy', [r for r in closed if r['question_type']=='MCQ'], 'correct')]:
        vectors[metric] = {r['id']: [r['score'][field], 1] for r in selected}
    for metric, field in [('patient_macro_accuracy','patient_id'),
                          ('modality_macro_accuracy','sequence'), ('category_macro_accuracy','category')]:
        selected = [r for r in closed if metric!='modality_macro_accuracy' or r['sequence'] in ('T1W','T2W')]
        counts = Counter(r[field] for r in selected)
        vectors[metric] = {r['id']: [r['score']['correct']/counts[r[field]], 1/counts[r[field]]] for r in selected}
    counts = Counter(r['neuroqa_id'] for r in rr)
    vectors['qa_ce'] = {r['id']: [r['qa_ce']/counts[r['neuroqa_id']], 1/counts[r['neuroqa_id']]] for r in rr}
    return vectors


def ce_catalog():
    result = {}
    # Only complete, original exported measurements; diagnostic/failed replays excluded.
    paths = list((ROOT/'results/significance_baseline_a_5ratios').glob('r*/ce/*/summary.json'))
    paths += list((ROOT/'results/significance_256_20261030').glob('ce/*/summary.json'))
    paths += list((OUT/'ce').glob('*/summary.json'))
    for p in paths:
        s = read(p)
        if s.get('status')=='complete': result.setdefault(s['checkpoint_sha256'], []).append(p.parent)
    return result


def load_entry(e, ce_paths):
    task = e['key'].split('/')[0]; paths = e['paths']
    rr = rows(paths['predictions']); m = read(paths['metric'])
    assert len(rr) == len({r['id'] for r in rr}) == e['prediction_count']
    audit = dict(setting=e['key'], source=e['source'], sources={k: dict(path=paths[k], sha256=file_digest(paths[k]))
                 for k in ('metric','config','predictions','completion_marker')})
    if task == 'rg':
        assert all(r['control'] is None and r['checkpoint_id']==m['checkpoint_sha256'] for r in rr)
        cp = Path(paths['metric']).parent/'scoring/clinical_scores.jsonl'
        pp = cp.parent/'scoring_provenance.json'
        cc = rows(cp); prov = read(pp)
        validate_pair(cc, cc, list(set(GROUPS.values())))
        cr = {r['id']:r for r in cc}; assert cr.keys()=={r['id'] for r in rr}
        assert prov['input_sha256']==file_digest(paths['predictions'])
        for r in rr:
            for field in ('reference','prediction','patient_id','sequence','checkpoint_id'):
                assert r[field]==cr[r['id']][field]
        vectors = lexical_vectors(rr); contracts = {}
        for group in set(GROUPS.values()):
            fp = digest(prov['scorers'][group])
            assert all(r['clinical'][group]['scorer_fingerprint']==fp for r in cc)
            contracts[group]=digest(scorer_contract(prov['scorers'][group]))
        for key, group in GROUPS.items():
            vectors[key]={r['id']:[r['clinical'][group]['values'][key],1] for r in cc if r['clinical'][group]['status']=='ok'}
        audit.update(checkpoint_sha256=m['checkpoint_sha256'], scorer_contracts=contracts,
                     test_identity_sha256=row_identity(rr), original_metrics=m['overall'])
        audit['sources']['clinical']=dict(path=str(cp),sha256=file_digest(cp))
        audit['sources']['scoring_provenance']=dict(path=str(pp),sha256=file_digest(pp))
        history=read(Path(paths['predictions']).parent/'metrics.json')['answer_ce']
        audit['original_ce_aggregation']=history
        run=dict(rows=rr,vectors=vectors,audit=audit)
        ce_errors=[]
        for p in ce_paths.get(m['checkpoint_sha256'],[]):
            try:
                if attach_ce(run,p,policy='recompute'): break
            except (ValueError,AssertionError) as exc: ce_errors.append(dict(path=str(p),reason=str(exc)))
        audit['ce_rejected']=ce_errors
        audit['ce_available']='answer_ce' in vectors
    else:
        marker=read(paths['completion_marker'])
        assert all(r.get('control')=='normal' for r in rr)
        assert len({r['checkpoint_sha256'] for r in rr})==1
        assert rr[0]['checkpoint_sha256']==marker['checkpoint_sha256']
        assert len({r['prompt_fingerprint'] for r in rr})==1
        assert len({r['preprocessing_fingerprint'] for r in rr})==1
        ground_truth,manifest_source=qa_manifest(paths['config'])
        audit['sources']['qa_manifest']=manifest_source
        for r in rr:
            gt=ground_truth[r['neuroqa_id']]
            assert r['reference']==gt['answer'] and r['patient_id']==gt['patient_id']
            for field in ('question_type','category','dataset'):assert r[field]==gt[field]
            assert score_row(SimpleNamespace(row=gt),r['prediction'])==r['score'],(e['key'],r['id'],'VQA cached score differs from rescore')
            assert math.isfinite(r['qa_ce']) and r['qa_ce']>=0 and r['answer_tokens']>0
        vectors=vqa_vectors(rr)
        original = {**m['overall'], **{k:m[k] for k in VQA_METRICS if k.endswith('_macro_accuracy')},
                    'yn_accuracy':m['strata']['question_type']['YN']['closed_accuracy'],
                    'mcq_accuracy':m['strata']['question_type']['MCQ']['closed_accuracy']}
        audit.update(checkpoint_sha256=rr[0]['checkpoint_sha256'], original_metrics=original,
                     scorer_contracts={k:m[k] for k in ('scorer_sha256','wrapper_sha256')},
                     prompt_fingerprint=rr[0]['prompt_fingerprint'], preprocessing_fingerprint=rr[0]['preprocessing_fingerprint'])
    for metric, vv in vectors.items():
        value=statistic(metric,np.asarray(list(vv.values())).sum(0))
        expected=audit['ce_replay']['answer_ce'] if metric=='answer_ce' else audit['original_metrics'][metric]
        assert math.isclose(value,expected,rel_tol=1e-8,abs_tol=1e-8), (e['key'],metric,value,expected)
    return dict(rows=rr,vectors=vectors,audit=audit)


def export(index, es, results):
    wb=load_workbook(index['master'])
    for name in list(wb.sheetnames):
        if name not in ('Main','VQA','Ablation'): del wb[name]
    # Preserve labels/layout but never leave original scores looking like p-values.
    for ws in wb:
        for row in ws:
            # Some historical alternate-run rows have a blank/merged method
            # label. Remove numeric scores there as well, before filling only
            # the explicitly reconciled paper-setting cells.
            for cell in row[1:]:
                if isinstance(cell.value,(int,float)):
                    cell.value=None;cell.comment=None
            # Metric headers have an empty first cell. Every named data row is
            # cleared, including "MLP (No Compression)"; matching the substring
            # "Compression" would incorrectly preserve its original scores.
            if isinstance(row[0].value,str):
                for cell in row[1:]:
                    if cell.value is not None:
                        cell.value=None; cell.comment=None
        ws['A1'].comment=Comment('Raw two-sided patient-cluster permutation p-values vs ADirect+LocalRead. No multiplicity adjustment. A0 and excluded methods left blank; CI endpoints are not independent metrics. Protocol: results/anchorread3d_final_eval_1108/STATISTICAL_PROTOCOL.txt','Codex')
    for result in results:
        sheet,row=es[result['setting']]['source'].split('!'); row=int(row)
        names=METRICS if result['task']=='rg' else VQA_METRICS
        cell=wb[sheet].cell(row,2+names.index(result['metric']))
        cell.value=result.get('p_raw','N/A');cell.number_format='0.00000'
        cell.font=Font(bold=False,color='000000')
        comment='Raw two-sided patient-cluster permutation p; no multiplicity adjustment.\n'
        if result['status']=='complete':
            comment+=f"A0 - comparator: {result['delta_ours_minus_baseline']:.9g}; improvement={result['improvement']:.9g}\nN={result['common_valid_samples']}; patients={result['patients']}; exchanges={result['permutations']}"
            if result['metric']=='answer_ce':comment+='\nCE-only replay of the original frozen checkpoints; cross-hardware BF16 differences from the source table are retained in audit.json. No training or original report regeneration.'
        else: comment+=result['reason']
        cell.comment=Comment(comment,'Codex')
    # Existing CI endpoints remain columns, explicitly not statistical hypotheses.
    for e in es.values():
        if e['key'].startswith('vqa/') and e['key'].split('/')[1]!='A0':
            sheet,row=e['source'].split('!')
            for col in (12,13):
                wb[sheet].cell(int(row),col,'N/A').comment=Comment('CI endpoint, not a separate performance metric; use Closed Accuracy p-value.','Codex')
    dest=ROOT/'results/Experiments for VLM Projector 1108_pvalues.xlsx'
    temp=dest.with_name(dest.stem+'.tmp.xlsx');wb.save(temp);temp.replace(dest)
    check=load_workbook(dest,data_only=True)
    assert check.sheetnames==['Main','VQA','Ablation']
    numeric=0
    for r in results:
        sheet,row=es[r['setting']]['source'].split('!'); names=METRICS if r['task']=='rg' else VQA_METRICS
        value=check[sheet].cell(int(row),2+names.index(r['metric'])).value
        if r['status']=='complete': assert math.isclose(value,r['p_raw'],rel_tol=1e-12);numeric+=1
        else: assert value=='N/A'
    for ws in check:
        for row in ws:
            if 'ADirect + LocalRead' in str(row[0].value): assert all(c.value is None for c in row[1:13])
            if any(name in str(row[0].value) for name in ('MLP (No Compression)','Mixer3D','AttentionPool','Perceiver_Flamingo')):
                assert all(c.value is None for c in row[1:]),('Excluded row retains original values',ws.title,row[0].coordinate)
    # No historical score may remain disguised as a p-value elsewhere.
    assert sum(isinstance(c.value,(int,float)) for ws in check for row in ws for c in row[1:])==numeric
    return dict(path=str(dest),sha256=file_digest(dest),numeric_pvalues=numeric)


def main():
    p=argparse.ArgumentParser();p.add_argument('--audit-only',action='store_true');p.add_argument('--repeats',type=int,default=19999);args=p.parse_args()
    index,es=entries();out=OUT/'significance';out.mkdir(parents=True,exist_ok=True)
    cache=ce_catalog();runs={};audits=[]
    book=load_workbook(index['master'],data_only=True)
    for key,e in sorted(es.items()):
        print('AUDIT',key,flush=True);run=load_entry(e,cache);runs[key]=run;audits.append(run['audit'])
        sheet,row=e['source'].split('!');names=METRICS if key.startswith('rg/') else VQA_METRICS
        for col,name in enumerate(names,2):
            assert abs(book[sheet].cell(int(row),col).value-run['audit']['original_metrics'][name])<=.000050001,(key,name)
    write(out/'audit.json',audits)
    missing=[es[k] for k,r in runs.items() if k.startswith('rg/') and 'answer_ce' not in r['vectors']]
    write(out/'ce_missing.json',missing)
    print('AUDIT_OK',len(runs),'CE_MISSING',len(missing),flush=True)
    if args.audit_only:return
    results=[];old_path=out/'comparisons.json';old={}
    if old_path.exists():
        for r in read(old_path):
            if r.get('status')=='complete' and r['permutations']==args.repeats:old[(r['setting'],r['metric'])]=r
    for key,e in sorted(es.items()):
        task,method,ratio=key.split('/')
        if method=='A0':continue
        a,b=runs[f'{task}/A0/{ratio}'],runs[key]
        assert a['audit']['scorer_contracts']==b['audit']['scorer_contracts'],(key,'scorer contract')
        if task=='vqa':
            for f in ('prompt_fingerprint','preprocessing_fingerprint'): assert a['audit'][f]==b['audit'][f]
            lookup={r['id']:r for r in b['rows']}
            for r in a['rows']:
                for f in ('neuroqa_id','question_type','category','dataset'):assert r[f]==lookup[r['id']][f]
        else:
            lookup={r['id']:r for r in b['rows']}
            for r in a['rows']:
                for f in ('preprocessing_version','prompt_version','split','target_version'):
                    assert r[f]==lookup[r['id']][f],(key,r['id'],f)
        names=METRICS if task=='rg' else VQA_METRICS
        for metric in names:
            r=dict(setting=key,task=task,baseline=method,ratio=int(ratio[1:]),metric=metric,status='complete')
            identity=digest([a['audit']['sources'],b['audit']['sources'],a['audit'].get('ce_samples_sha256'),b['audit'].get('ce_samples_sha256'),metric,args.repeats])
            previous=old.get((key,metric))
            if metric not in a['vectors'] or metric not in b['vectors']:
                r.update(status='unavailable',reason='Per-report Answer CE sufficient statistics unavailable; aggregate CE cannot determine a paired p-value.')
            elif previous and previous.get('input_identity')==identity:r=previous
            else:r.update(paired_p(a['rows'],b['rows'],a['vectors'][metric],b['vectors'][metric],metric,args.repeats),input_identity=identity)
            if metric=='answer_ce' and r['status']=='complete':
                r.update(score_basis='frozen_checkpoint_teacher_forced_replay',
                         replay_ours=a['audit']['ce_replay'],replay_baseline=b['audit']['ce_replay'])
            results.append(r)
        print('TESTED',key,flush=True);write(old_path,results)
    exported=export(index,es,results)
    write(out/'completion.json',dict(status='complete' if all(r['status']=='complete' for r in results) else 'partial',
          planned=len(results),completed=sum(r['status']=='complete' for r in results),workbook=exported,
          source_workbook_sha256=index['master_sha256'],source_index_sha256=file_digest(INDEX),repeats=args.repeats,
          analysis_source_hashes={str(p.relative_to(ROOT)):file_digest(p) for p in Path(__file__).parent.glob('*.py')}))
    fields=['task','ratio','baseline','metric','status','p_raw','score_ours','score_baseline','delta_ours_minus_baseline','improvement','patients','common_valid_samples','expected_samples','permutations','p_mc_se','reason']
    with (out/'comparisons.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(results)
    print(read(out/'completion.json'),flush=True)


if __name__=='__main__':main()
