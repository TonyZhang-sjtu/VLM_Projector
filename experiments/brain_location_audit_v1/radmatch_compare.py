"""Isolated official RadMatch execution and coverage-aware comparison export.

prepare freezes real report inputs and graph-v2 diagnostics; run calls unmodified
upstream APIs with brain-mr; collect retains every expected case, even failures.
Neither an automatic score nor a discrepancy is treated as human ground truth.
"""
import argparse
from collections import Counter
import csv
import hashlib
import json
import os
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parents[2]
INDEX = ROOT/'results/anchorread3d_analysis_1108/artifact_reconciliation_1108.json'
# Pinned upstream radmatch.constants.RESULTS_DIR. Kept dependency-free for collect.
UPSTREAM_RESULTS_DIR = 'radmatch_results'


def digest(data):
    return hashlib.sha256(data.encode() if isinstance(data,str) else data).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def dump(path, obj):
    Path(path).write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n')


def lines(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def prepare(args):
    from .prepare import choose_pilot, stress_cases
    from .protocol_v2 import VERSION, extract_graph, score_sets
    idx = read(args.index)
    assert digest(Path(idx['master']).read_bytes()) == idx['master_sha256']
    entries = [e for e in idx['paper_settings'] if e['key'].startswith('rg/')]
    def clinical(e):
        return Path(e['paths']['metric']).parent/'scoring/clinical_scores.jsonl'
    canonical = next(e for e in entries if e['key']=='rg/A0/r32')
    refs = {r['id']:r for r in lines(clinical(canonical))}
    assert len(refs)==560 and len({r['patient_id'] for r in refs.values()})==140
    ids = choose_pilot(refs) if args.scope=='pilot' else sorted(refs)
    if args.scope=='pilot':
        entries = [e for e in entries if e['key']=='rg/MLP/r1' or
                   (e['key'].split('/')[1] in ['A0','AvgPool3D','Perceiver_NV','A1','A2','A3','A4'] and
                    e['key'].split('/')[-1] in ['r32','r256'])]
    entries = sorted(entries,key=lambda e:e['key'])
    assert len(entries) == (15 if args.scope=='pilot' else 57)
    args.bundle.mkdir(parents=True,exist_ok=False)
    paths={}; rows=[]; groups=[]
    def write_text(rel, text):
        file=args.bundle/rel;file.parent.mkdir(parents=True,exist_ok=True)
        file.write_text(text); paths[str(rel)]=digest(text)
    names={rid:f'case_{i:03d}' for i,rid in enumerate(ids)}
    for rid in ids:
        write_text(Path('reports_gt')/(names[rid]+'.txt'),refs[rid]['reference'])
    for e in entries:
        tag=e['key'].replace('/','_')
        original={r['id']:r for r in lines(e['paths']['predictions'])}
        scored={r['id']:r for r in lines(clinical(e))}
        assert set(original)==set(scored)==set(refs)
        for rid in ids:
            r=scored[rid]
            assert r['reference']==refs[rid]['reference']==original[rid]['reference']
            assert r['prediction']==original[rid]['prediction']
            write_text(Path('reports_pred')/tag/(names[rid]+'.txt'),r['prediction'])
            rg=r['clinical']['radgraph']; refg=refs[rid]['clinical']['radgraph']
            ref=extract_graph(refg.get('details',{}).get('reference_graph'))
            pred=extract_graph(rg.get('details',{}).get('prediction_graph'))
            rows.append(dict(group=tag,setting=e['key'],case_id=names[rid],id=rid,
                patient_id=r['patient_id'],sequence=r['sequence'],
                blind_id=digest('blind/'+e['key']+'/'+rid)[:16],
                reference_extraction=ref,prediction_extraction=pred,
                graph_scores=score_sets(ref,pred),graph_protocol=VERSION,
                reference_sha256=digest(r['reference']),prediction_sha256=digest(r['prediction'])))
        groups.append(dict(name=tag,setting=e['key'],kind='patient',reports_gt='reports_gt',
                           reports_pred='reports_pred/'+tag,expected_ids=list(names.values())))
    stress=stress_cases()
    for case in stress:
        for key,sub in [('reference','reports_gt'),('prediction','reports_pred')]:
            write_text(Path('stress')/sub/(case['id']+'.txt'),case[key])
    groups.insert(0,dict(name='stress',kind='synthetic',reports_gt='stress/reports_gt',
                         reports_pred='stress/reports_pred',expected_ids=[c['id'] for c in stress]))
    dump(args.bundle/'comparison_inputs.json',rows)
    dump(args.bundle/'stress_expectations.json',stress)
    paths['comparison_inputs.json']=digest((args.bundle/'comparison_inputs.json').read_bytes())
    paths['stress_expectations.json']=digest((args.bundle/'stress_expectations.json').read_bytes())
    dump(args.bundle/'manifest.json',dict(scope=args.scope,index=str(args.index),
        index_sha256=digest(args.index.read_bytes()),master_sha256=idx['master_sha256'],
        unique_patients=len({refs[r]['patient_id'] for r in ids}),unique_reports=len(ids),
        patient_pairs=len(rows),groups=groups,file_sha256=paths,graph_protocol=VERSION,
        protocol_source_sha256={p.name:digest(p.read_bytes()) for p in
                               [Path(__file__).with_name('protocol.py'),Path(__file__).with_name('protocol_v2.py')]},
        human_gold_available=False,radmatch_executed=False))
    print(json.dumps(dict(scope=args.scope,patient_pairs=len(rows),synthetic_pairs=len(stress))))


def verify_inputs(bundle):
    manifest=read(bundle/'manifest.json')
    for rel,expected in manifest['file_sha256'].items():
        if digest((bundle/rel).read_bytes())!=expected:
            raise ValueError(f'Input changed: {rel}')
    return manifest


def read_case(result_dir, case_id):
    """Never convert an upstream failure or absent output into zero errors."""
    stages=['findings_gt','findings_pred','matching','attribute_errors','per_report_metrics']
    content={};missing=[]
    for stage in stages:
        path=result_dir/stage/(case_id+'.json')
        try:
            content[stage]=read(path)
        except (OSError,ValueError):
            missing.append(stage)
    if missing:
        return dict(status='missing_or_invalid',missing_stages=missing),content
    if not all(isinstance(content[k],dict) for k in ['matching','attribute_errors','per_report_metrics']):
        return dict(status='missing_or_invalid',missing_stages=['invalid_schema']),content
    metrics=content['per_report_metrics']
    loc=metrics.get('attribute_breakdown',{}).get('location')
    if not isinstance(loc,dict) or not all(k in loc for k in ['major','minor','evaluated']):
        return dict(status='missing_or_invalid',missing_stages=['location_schema']),content
    return dict(status='degraded' if content['matching'].get('validation_fallback') else 'scored',
                actionable_errors=metrics.get('actionable_errors_total'),
                location_major=loc['major'],location_minor=loc['minor'],
                location_matched_pairs=loc['evaluated'],
                missed_findings=metrics.get('muc_counts',{}).get('MIS'),
                spurious_findings=metrics.get('muc_counts',{}).get('SPU')),content


def collect(args):
    manifest=verify_inputs(args.bundle)
    graph={(r['group'],r['case_id']):r for r in read(args.bundle/'comparison_inputs.json')}
    case_rows=[]; summaries=[];details=[];blind=[]
    for group in manifest['groups']:
        grows=[]
        for cid in group['expected_ids']:
            result,raw=read_case(args.output/group['name']/UPSTREAM_RESULTS_DIR,cid)
            base=graph.get((group['name'],cid),{})
            row=dict(group=group['name'],kind=group['kind'],case_id=cid,
                     patient_id=base.get('patient_id'),**result)
            for dim,score in base.get('graph_scores',{}).items():
                row['graph_'+dim+'_eligible']=score['eligible']
                row['graph_'+dim+'_exact']=score.get('exact')
                row['graph_'+dim+'_f1']=score.get('f1')
            case_rows.append(row);grows.append(row)
            if group['kind']=='patient':
                details.append(dict(**base,radmatch_result=result,radmatch_raw=raw,
                                    human_verdict=None))
                blind.append(dict(blind_id=base['blind_id'],
                    reference_text=(args.bundle/group['reports_gt']/(cid+'.txt')).read_text(),
                    prediction_text=(args.bundle/group['reports_pred']/(cid+'.txt')).read_text(),
                    review_status='pending',reviewer='',reference_facts_json='',prediction_facts_json='',
                    human_side_error='',human_site_error='',human_joint_error='',human_omission='',
                    human_midline_shift_error='',notes=''))
        counts=Counter(r['status'] for r in grows)
        scored=[r for r in grows if r['status'] in ['scored','degraded']]
        summaries.append(dict(group=group['name'],kind=group['kind'],expected=len(grows),
            scored=counts['scored'],degraded=counts['degraded'],missing_or_invalid=counts['missing_or_invalid'],
            actionable_errors_per_scored_report=sum(r['actionable_errors'] for r in scored)/len(scored) if scored else None,
            location_major_total=sum(r['location_major'] for r in scored) if scored else None,
            location_minor_total=sum(r['location_minor'] for r in scored) if scored else None,
            location_matched_pairs=sum(r['location_matched_pairs'] for r in scored) if scored else None,
            status='complete' if counts['scored']==len(grows) else 'incomplete_or_degraded'))
    args.output.mkdir(parents=True,exist_ok=True)
    def csv_write(name, rows):
        keys=list(dict.fromkeys(k for r in rows for k in r))
        with (args.output/name).open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
    csv_write('comparison_cases.csv',case_rows)
    csv_write('comparison_summary.csv',summaries)
    (args.output/'paired_details.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in details))
    # Never overwrite completed human annotations on a subsequent collection.
    if not (args.output/'blind_human_review.csv').exists():
        csv_write('blind_human_review.csv',sorted(blind,key=lambda r:r['blind_id']))
    dump(args.output/'coverage.json',dict(expected=sum(len(g['expected_ids']) for g in manifest['groups']),
        status_counts=dict(Counter(r['status'] for r in case_rows)),
        execution_state='started' if (args.output/'run_identity.json').exists() else 'not_started',
        comparison_reliability_established=False,
        warning='Graph labels are automatic. RadMatch location denominators count matched findings, not eligible lesions.'))
    print(json.dumps(read(args.output/'coverage.json')))


def run(args):
    import radmatch
    from radmatch import constants
    from .radmatch_offline_check import installation_record
    manifest=verify_inputs(args.bundle)
    install=installation_record()
    if constants.RESULTS_DIR != UPSTREAM_RESULTS_DIR:
        raise ValueError('Upstream output directory contract changed')
    if args.only and set(args.only)-{g['name'] for g in manifest['groups']}:
        raise ValueError('Unknown --only group name')
    if not args.judge.startswith('local:'):
        raise ValueError('This runner expects an explicitly configured OpenAI-compatible local judge (local:MODEL).')
    base=os.environ.get('RADMATCH_LOCAL_BASE_URL','')
    if not base:
        raise ValueError('No judge configured: set RADMATCH_LOCAL_BASE_URL to an available authorized endpoint.')
    req=urllib.request.Request(base.rstrip('/')+'/models')
    if os.environ.get('RADMATCH_LOCAL_API_KEY'):
        req.add_header('Authorization','Bearer '+os.environ['RADMATCH_LOCAL_API_KEY'])
    with urllib.request.urlopen(req,timeout=15) as response:
        available=json.load(response)
    if args.judge[6:] not in {m['id'] for m in available.get('data',[])}:
        raise ValueError('Requested judge model not advertised by endpoint')
    identity=dict(input_manifest_sha256=digest((args.bundle/'manifest.json').read_bytes()),
        commit=install['commit'],judge=args.judge,fewshot='brain-mr',reasoning=args.reasoning,
        endpoint_sha256=digest(base),temperature=os.environ.get('RADMATCH_LOCAL_TEMPERATURE','0.0'),
        seed=os.environ.get('RADMATCH_LOCAL_SEED','42'),workers=args.workers,
        max_tokens=os.environ.get('RADMATCH_MAX_TOKENS','32768'),
        request_timeout=os.environ.get('RADMATCH_REQUEST_TIMEOUT_S','180.0'))
    args.output.mkdir(parents=True,exist_ok=True)
    if (args.output/'run_identity.json').exists() and read(args.output/'run_identity.json')!=identity:
        raise ValueError('Run identity changed: choose a new output directory')
    dump(args.output/'run_identity.json',identity)
    dump(args.output/'installation.json',install)
    owner=next(g for g in manifest['groups'] if g['kind']=='patient')
    canonical_gt=args.output/owner['name']/UPSTREAM_RESULTS_DIR/'findings_gt'
    shared_gt=(canonical_gt if {p.stem for p in canonical_gt.glob('*.json')}==set(owner['expected_ids']) else None)
    if args.only and any(name!='stress' for name in args.only) and not shared_gt and owner['name'] not in args.only:
        raise ValueError('First run must include the canonical reference owner '+owner['name'])
    for group in manifest['groups']:
        if args.only and group['name'] not in args.only:
            continue
        kwargs=dict(reports_pred_dir=args.bundle/group['reports_pred'],output_dir=args.output/group['name'],
            llm_extractor=args.judge,llm_judge=args.judge,fewshot='brain-mr',workers=args.workers,
            reasoning=args.reasoning,retry_passes=2,retry_degraded=True)
        if group['kind']=='patient' and shared_gt:
            # Do not ask the upstream adapter to copy a directory onto itself.
            if group['name']==owner['name']:
                kwargs['reports_gt_dir']=args.bundle/group['reports_gt']
            else:
                kwargs['findings_gt_dir']=shared_gt
        else:
            kwargs['reports_gt_dir']=args.bundle/group['reports_gt']
        print('RUN',group['name'],flush=True)
        try:
            radmatch.run_all(**kwargs)
        finally:
            collect(args)
        if group['kind']=='patient':
            candidate=args.output/group['name']/UPSTREAM_RESULTS_DIR/'findings_gt'
            if {p.stem for p in candidate.glob('*.json')}!=set(group['expected_ids']):
                raise RuntimeError('Reference extraction incomplete; stop to avoid changing reference denominators across methods.')
            if shared_gt:
                for p in candidate.glob('*.json'):
                    if read(p)!=read(shared_gt/p.name):
                        raise RuntimeError('Reference findings changed between methods')
            else:
                shared_gt=candidate
    collect(args)


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    a=sub.add_parser('prepare');a.add_argument('--scope',choices=['pilot','full'],default='pilot')
    a.add_argument('--index',type=Path,default=INDEX);a.add_argument('--bundle',type=Path,required=True)
    for name in ['run','collect']:
        a=sub.add_parser(name);a.add_argument('--bundle',type=Path,required=True);a.add_argument('--output',type=Path,required=True)
        if name=='run':
            a.add_argument('--judge',required=True);a.add_argument('--workers',type=int,default=1)
            a.add_argument('--reasoning',choices=['none','low','medium','high'],default='none')
            a.add_argument('--only',nargs='+',help='e.g. stress; omitted runs every prepared group')
    args=p.parse_args();globals()[args.command](args)


if __name__=='__main__':
    main()
