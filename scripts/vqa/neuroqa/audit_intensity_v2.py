#!/usr/bin/env python3
"""Independently compare image similarity on the same template points at each stage."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
import numpy as np
import SimpleITK as sitk
from src.config import digest, file_hash
from src.data.neuroqa_manifest import read_rows, write_json
from src.preprocessing.neuroqa_registration_v2 import prepare_native
from src.preprocessing.neuroqa_smri import TEMPLATE, compose_pull, warp


def inspect(source,root):
    sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(1)
    folder=root/'cases'/source['case_id']
    record=json.loads((folder/'complete.json').read_text())
    template=sitk.ReadImage(str(root/'templates'/TEMPLATE),sitk.sitkFloat32)
    values=sitk.GetArrayFromImage(template)[::2,::2,::2]
    points=values>0
    fixed=values[points]
    fixed_upper=float(np.percentile(fixed,99.9))
    initial=sitk.ReadTransform(str(folder/'initial.tfm'))
    rigid=sitk.ReadTransform(str(folder/'rigid.tfm'))
    t2=sitk.ReadTransform(str(folder/'T2w_to_T1w_rigid.tfm'))
    row={'case_id':source['case_id'],'modalities':{},'flags':[]}
    for m,path in source['images'].items():
        if file_hash(path)!=record['contract']['source_sha256'][m]:
            raise ValueError(f'Source changed for {source["case_id"]}/{m}')
        native, mask, _=prepare_native(sitk.ReadImage(path,sitk.sitkFloat32))
        signal=sitk.GetArrayFromImage(native)
        upper=float(np.percentile(signal[sitk.GetArrayFromImage(mask)>0],99.9))
        def nmi(t):
            moved=sitk.GetArrayFromImage(warp(native,template,t))[::2,::2,::2][points]
            histogram=np.histogram2d(np.clip(fixed,0,fixed_upper),np.clip(moved,0,upper),
                bins=64,range=[[0,fixed_upper],[0,upper]])[0]
            p=histogram/histogram.sum()
            def entropy(a):
                a=a[a>0]
                return float(-np.sum(a*np.log(a)))
            return (entropy(p.sum(axis=0))+entropy(p.sum(axis=1)))/entropy(p)
        final=sitk.ReadTransform(str(folder/f'{m}_mni_pull.tfm'))
        vals={'initial':nmi(initial),'rigid':nmi(rigid if m=='T1w' else compose_pull(t2,rigid)),
              'final':nmi(final)}
        if not np.isfinite(list(vals.values())).all():raise ValueError('Nonfinite NMI')
        if vals['final'] < vals['initial']-.005:row['flags'].append(f'{m}:nmi_drop_from_initial_gt_0.005')
        if vals['final'] < vals['rigid']-.005:row['flags'].append(f'{m}:nmi_drop_from_rigid_gt_0.005')
        row['modalities'][m]=vals
    return row


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--workers',type=int,default=4)
    args=parser.parse_args()
    root=ROOT/'assets/preprocessing/neuroqa_mni1mm_v2'
    source=read_rows(ROOT/'assets/data/NeuroQA_derived/official_holdout_patient_guard_v1_mni_v2/source_index.jsonl')
    protocol=json.loads((root/'protocol.json').read_text())
    report={'started_utc':datetime.now(timezone.utc).isoformat(),'protocol_sha256':digest(protocol),
        'audit_code_sha256':file_hash(__file__),'definition':'64-bin NMI, template-foreground fixed samples at stride 2; same points and intensity ranges across stages',
        'review_threshold':.005,'scope':'image_QC_only_not_training_or_QA_label_selection','cases':[], 'failed':[]}
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        jobs={pool.submit(inspect,row,root):row['case_id'] for row in source}
        for future in as_completed(jobs):
            try:report['cases'].append(future.result())
            except Exception as e:report['failed'].append({'case_id':jobs[future],'error':repr(e)})
            if (len(report['cases'])+len(report['failed']))%20==0:
                write_json(root/'intensity_audit.json',report)
                print(json.dumps({'processed':len(report['cases'])+len(report['failed']),'expected':len(source),'failed':len(report['failed'])}),flush=True)
    report['cases'].sort(key=lambda row:row['case_id'])
    report['finished_utc']=datetime.now(timezone.utc).isoformat()
    report['expected_cases']=len(source)
    report['complete']=len(report['cases'])==len(source) and not report['failed']
    report['flagged_cases']=[r['case_id'] for r in report['cases'] if r['flags']]
    write_json(root/'intensity_audit.json',report)
    print(json.dumps({'complete':report['complete'],'flagged_cases':len(report['flagged_cases'])}),flush=True)
    if not report['complete']:raise SystemExit(1)


if __name__=='__main__':main()
