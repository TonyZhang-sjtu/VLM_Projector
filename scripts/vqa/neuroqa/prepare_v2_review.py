#!/usr/bin/env python3
"""Create a review queue; never silently mark flagged cases reviewed."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from src.config import digest, file_hash
from src.data.neuroqa_manifest import write_json


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--reviews', type=Path, help='JSON object mapping every flagged case to {decision, note, reviewer}')
    args=p.parse_args()
    root=ROOT/'assets/preprocessing/neuroqa_mni1mm_v2'
    reproducibility_path=root/'production_reproducibility.json'
    reproducibility=json.loads(reproducibility_path.read_text())
    if reproducibility.get('status') != 'passed' or reproducibility.get('protocol_sha256') != digest(json.loads((root/'protocol.json').read_text())):
        raise RuntimeError('Independent production N4/transform reproducibility has not passed')
    audit_path=root/'independent_audit.json'
    audit=json.loads(audit_path.read_text())
    if not audit.get('finished_utc') or not audit['summary'].get('ready_for_full_training'):
        raise RuntimeError('Independent audit is not complete or did not pass')
    intensity_path=root/'intensity_audit.json'
    intensity=json.loads(intensity_path.read_text())
    if not intensity.get('complete') or intensity.get('protocol_sha256') != digest(json.loads((root/'protocol.json').read_text())):
        raise RuntimeError('Independent intensity audit is missing or incomplete')
    intensity_flags={row['case_id']:row['flags'] for row in intensity['cases']}
    lesion_path=root/'lesion_coverage_audit.json'
    lesion=json.loads(lesion_path.read_text())
    if not lesion.get('complete') or lesion.get('protocol_sha256') != digest(json.loads((root/'protocol.json').read_text())):
        raise RuntimeError('Independent lesion coverage audit is missing or incomplete')
    lesion_flags={row['case_id']:row['flags'] for row in lesion['cases']}
    flagged=[]
    for row in audit['cases']:
        folder=root/'cases'/row['case_id']
        record=json.loads((folder/'complete.json').read_text())
        reasons=[]
        reasons.extend(intensity_flags.get(row['case_id'],[]))
        reasons.extend(lesion_flags.get(row['case_id'],[]))
        reasons.extend(f"{volume['modality']}:brain_bounding_box_may_exceed_qc_reference"
            for volume in row['volumes'] if volume.get('native_bounding_box_may_exceed_expanded_qc_reference'))
        reasons.extend(f"{volume['modality']}:extreme_n4_correction"
            for volume in row['volumes'] if volume.get('n4_extreme_correction_review'))
        if record['selected_candidate'] != 'C' or record['transform_kind'] != 'affine':
            reasons.append('fallback')
        for m in ('T1w','T2w'):
            q=record['qc'][m]
            if q['dice'] < .85:reasons.append(f'{m}:low_dice_review')
            if q['outside_canvas_voxels'] > 0:reasons.append(f'{m}:outside_canvas')
            if q['intensity_repair']['negative_voxels'] > 0:reasons.append(f'{m}:negative_voxels_repaired')
        if reasons:
            flagged.append({'case_id':row['case_id'],'reasons':reasons,'qc_png':str(folder/'qc.png')})
    queue={'audit_sha256':file_hash(audit_path),'intensity_audit_sha256':file_hash(intensity_path),
           'lesion_audit_sha256':file_hash(lesion_path),
           'reproducibility_sha256':file_hash(reproducibility_path),
           'case_count':len(flagged),'cases':flagged,
           'scope':'engineering_review_not_clinical_certification'}
    write_json(root/'qc_review_queue.json',queue)
    if not args.reviews:
        print(json.dumps({'queue_cases':len(flagged),'complete':False,'queue':str(root/'qc_review_queue.json')}))
        return
    reviews=json.loads(args.reviews.read_text())
    if set(reviews) != {row['case_id'] for row in flagged}:
        raise ValueError('Review file must cover exactly every flagged case')
    for case, review in reviews.items():
        if review.get('decision') != 'accepted' or not review.get('note','').strip() or not review.get('reviewer','').strip():
            raise ValueError(f'Case {case} requires an explicit acceptance, reviewer and note')
    protocol=json.loads((root/'protocol.json').read_text())
    write_json(root/'qc_review.json',{'complete':True,'protocol_sha256':digest(protocol),
        'audit_sha256':file_hash(audit_path),'intensity_audit_sha256':file_hash(intensity_path),
        'lesion_audit_sha256':file_hash(lesion_path),
        'reproducibility_sha256':file_hash(reproducibility_path),
        'queue_sha256':file_hash(root/'qc_review_queue.json'),
        'reviews':reviews,'scope':'engineering_review_not_clinical_certification'})
    print(json.dumps({'queue_cases':len(flagged),'complete':True}))


if __name__ == '__main__':
    main()
