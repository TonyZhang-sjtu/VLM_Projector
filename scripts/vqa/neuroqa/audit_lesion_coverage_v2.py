#!/usr/bin/env python3
"""QC-only check of native BraTS labels and the selected T1 MNI canvas.

Labels are never inputs to registration, retry selection, N4 or VQA training.
"""
import argparse
from itertools import product
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
import numpy as np
import SimpleITK as sitk
from src.config import digest,file_hash
from src.data.neuroqa_manifest import read_rows,write_json
from src.preprocessing.neuroqa_smri import TEMPLATE,canvas_coverage,warp


def inspect(source,root):
    sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(1)
    case=source['case_id']
    folder=root/'cases'/case
    t1=sitk.ReadImage(source['images']['T1w'])
    label_path=Path(source['images']['T1w']).parent/f'{case}-seg.nii.gz'
    seg=sitk.ReadImage(str(label_path))
    for name in ('GetSize','GetSpacing','GetOrigin','GetDirection'):
        if getattr(seg,name)()!=getattr(t1,name)():raise ValueError(f'Label/T1 geometry differs: {case}/{name}')
    values=sitk.GetArrayFromImage(seg)
    if not np.isfinite(values).all():raise ValueError(f'Nonfinite label: {case}')
    mask=sitk.Cast(seg!=0,sitk.sitkUInt8)
    a=sitk.GetArrayFromImage(mask)>0
    faces={f'array_axis_{k}_{side}':int(np.take(a,i,axis=k).sum())
           for k in range(3) for side,i in (('first',0),('last',-1))}
    edge=any(faces.values())
    template=sitk.ReadImage(str(root/'templates'/TEMPLATE))
    transform=sitk.ReadTransform(str(folder/'T1w_mni_pull.tfm'))
    coverage=canvas_coverage(mask,template,transform)
    inside=int((sitk.GetArrayFromImage(warp(mask,template,transform,True))>0).sum())
    flags=[]
    if not a.any():flags.append('empty_native_segmentation')
    if edge:flags.append('native_segmentation_touches_source_edge')
    if coverage['outside_canvas_voxels']>0:flags.append('lesion_label_outside_mni_canvas')
    if coverage['expanded_boundary_touched']:flags.append('lesion_touches_expanded_qc_boundary')
    if a.any() and inside+coverage['outside_canvas_voxels']==0:
        flags.append('native_lesion_absent_from_expanded_reference')
    # An expanded sampled image cannot detect a disconnected component entirely
    # beyond that reference. Affine-map the eight corners of the native label's
    # bounding box to conservatively flag that possibility; a box warning alone
    # does not prove foreground is outside because its corners may be empty.
    possible_beyond = False
    if a.any():
        zyx=np.where(a)
        xyz=[(int(axis.min()),int(axis.max())) for axis in zyx[::-1]]
        inverse=transform.GetInverse()
        size=template.GetSize()
        for corner in product(*xyz):
            native_point=seg.TransformIndexToPhysicalPoint(tuple(corner))
            mapped=template.TransformPhysicalPointToContinuousIndex(inverse.TransformPoint(native_point))
            if any(value < -31. or value > size[i]+30. for i,value in enumerate(mapped)):
                possible_beyond=True
                break
    if possible_beyond:flags.append('lesion_bounding_box_may_extend_beyond_qc_reference')
    return {'case_id':case,'label_sha256':file_hash(label_path),'native_label_voxels':int(a.sum()),
        'native_edge_contact':bool(edge),'native_edge_face_voxels':faces,'inside_canvas_label_voxels':inside,
        'bounding_box_may_extend_beyond_qc_reference':possible_beyond,'coverage':coverage,'flags':flags}


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--workers',type=int,default=4);args=p.parse_args()
    root=ROOT/'assets/preprocessing/neuroqa_mni1mm_v2'
    sources=read_rows(ROOT/'assets/data/NeuroQA_derived/official_holdout_patient_guard_v1_mni_v2/source_index.jsonl')
    protocol=json.loads((root/'protocol.json').read_text())
    report={'started_utc':datetime.now(timezone.utc).isoformat(),'protocol_sha256':digest(protocol),
        'audit_code_sha256':file_hash(__file__),'scope':'labels_for_posthoc_QC_only_not_registration_or_training',
        'definition':'T1-space BraTS segmentation transformed to an expanded MNI reference using the selected T1 pull transform; T2 signal was aligned to the same T1 space.',
        'cases':[],'failed':[]}
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures={pool.submit(inspect,row,root):row['case_id'] for row in sources}
        for f in as_completed(futures):
            try:report['cases'].append(f.result())
            except Exception as e:report['failed'].append({'case_id':futures[f],'error':repr(e)})
            if (len(report['cases'])+len(report['failed']))%20==0:
                write_json(root/'lesion_coverage_audit.json',report)
                print(json.dumps({'processed':len(report['cases'])+len(report['failed']),'expected':len(sources),'failed':len(report['failed'])}),flush=True)
    report['cases'].sort(key=lambda row:row['case_id'])
    report['finished_utc']=datetime.now(timezone.utc).isoformat()
    report['expected_cases']=len(sources)
    report['complete']=len(report['cases'])==len(sources) and not report['failed']
    report['flagged_cases']=[r['case_id'] for r in report['cases'] if r['flags']]
    write_json(root/'lesion_coverage_audit.json',report)
    print(json.dumps({'complete':report['complete'],'flagged_cases':len(report['flagged_cases'])}),flush=True)
    if not report['complete']:raise SystemExit(1)


if __name__=='__main__':main()
