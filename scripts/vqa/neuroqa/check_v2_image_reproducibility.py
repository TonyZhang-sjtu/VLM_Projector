#!/usr/bin/env python3
"""Compare an independent sandbox and production run of the same v2 case."""
import argparse
from datetime import datetime,timezone
from itertools import product
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
import numpy as np
import SimpleITK as sitk
from src.config import digest
from src.data.neuroqa_manifest import write_json


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--wait',action='store_true')
    args=p.parse_args()
    case='BraTS-GLI-00017-000'
    plan=ROOT/'results/vqa/neuroqa/preprocessing_v2_plan'
    sandbox=plan/'sandbox/neuroqa_mni1mm_v2/cases'/case
    root=ROOT/'assets/preprocessing/neuroqa_mni1mm_v2'
    production=root/'cases'/case
    end=time.monotonic()+12*3600
    while not (production/'complete.json').exists():
        status=production/'status.json'
        if status.exists() and json.loads(status.read_text()).get('stage') == 'qc_review_required':
            raise RuntimeError('Reproducibility case did not pass production QC')
        if not args.wait or time.monotonic()>end:
            raise RuntimeError('Production reproducibility case is not complete')
        time.sleep(20)
    left=json.loads((sandbox/'complete.json').read_text())
    right=json.loads((production/'complete.json').read_text())
    if left['selected_candidate']!='C' or right['selected_candidate']!='C':
        raise ValueError('Reproducibility test requires the same primary candidate')
    if left['contract']['source_sha256']!=right['contract']['source_sha256']:
        raise ValueError('Reproducibility source data differ')
    results=[]
    for m in ('T1w','T2w'):
        for kind in ('warped','mask','n4','zscore','zscore_cropped'):
            name=f'{m}_mni_{kind}.nii.gz'
            a=sitk.ReadImage(str(sandbox/name));b=sitk.ReadImage(str(production/name))
            geometry=all(getattr(a,k)()==getattr(b,k)() for k in ('GetSize','GetSpacing','GetOrigin','GetDirection'))
            x,y=sitk.GetArrayFromImage(a),sitk.GetArrayFromImage(b)
            exact=geometry and np.array_equal(x,y)
            difference=float(np.max(np.abs(x.astype(np.float64)-y.astype(np.float64)))) if x.shape==y.shape else None
            results.append({'file':name,'geometry_equal':geometry,'voxel_values_exact':exact,'max_absolute_difference':difference})
    points=list(product((-90.,0.,90.),repeat=3))
    errors=[]
    for name in ('rigid','affine','T2w_to_T1w_rigid','T1w_mni_pull','T2w_mni_pull'):
        a=sitk.ReadTransform(str(sandbox/f'{name}.tfm'));b=sitk.ReadTransform(str(production/f'{name}.tfm'))
        errors.append(max(float(np.linalg.norm(np.array(a.TransformPoint(pt))-np.array(b.TransformPoint(pt)))) for pt in points))
    passed=all(x['voxel_values_exact'] for x in results) and max(errors) <= 1e-6
    report={'utc':datetime.now(timezone.utc).isoformat(),'case_id':case,'scope':'independent_processes_different_worker_concurrency',
        'protocol_sha256':digest(json.loads((root/'protocol.json').read_text())),'status':'passed' if passed else 'failed',
        'images':results,'maximum_transform_difference_mm':max(errors)}
    write_json(root/'production_reproducibility.json',report)
    print(json.dumps(report),flush=True)
    if not passed:raise SystemExit(1)


if __name__=='__main__':main()
