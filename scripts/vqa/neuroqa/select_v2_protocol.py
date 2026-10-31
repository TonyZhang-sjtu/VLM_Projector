#!/usr/bin/env python3
"""Freeze v2 only when the complete train pilot and independent checks pass."""
import argparse
import json
from pathlib import Path
import sys
from itertools import product

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
import numpy as np
import SimpleITK as sitk
from src.config import file_hash
from src.data.neuroqa_manifest import write_json
from src.preprocessing import neuroqa_registration_v2, neuroqa_smri_v2
from src.preprocessing import neuroqa_smri
from src.preprocessing.neuroqa_smri import TEMPLATE
from scripts.vqa.neuroqa.summarize_pilot_v2 import POLICY


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--apply', action='store_true')
    args = p.parse_args()
    plan = ROOT/'results/vqa/neuroqa/preprocessing_v2_plan'
    pilot = json.loads((plan/'pilot_manifest.json').read_text())
    summary = json.loads((plan/'pilot_summary.json').read_text())
    batch = json.loads((plan/'pilot/batch.json').read_text())
    sandbox_path = plan/'sandbox_verified.json'
    sandbox = json.loads(sandbox_path.read_text()) if sandbox_path.exists() else {}
    errors = []
    if batch['requested'] != 96 or len(batch['successful']) != 96 or batch['failed']:
        errors.append('Pilot batch incomplete or contains failures')
    for candidate in ('B','C','D'):
        s = summary['candidate_summaries'][candidate]
        if s['measured_cases'] != 32 or s['missing_cases']:
            errors.append(f'Candidate {candidate} incomplete')
    if sandbox.get('status') != 'passed' or len(sandbox.get('volumes',[])) != 4:
        errors.append('Two-case N4/adapter integration not verified')
    visual_path = plan/'pilot_visual_review/review.json'
    visual = json.loads(visual_path.read_text()) if visual_path.exists() else {}
    if (visual.get('summary_sha256') != file_hash(plan/'pilot_summary.json') or
            set(visual.get('accepted_cases', [])) != set(pilot['case_ids']) or
            visual.get('queue_sha256') != file_hash(plan/'pilot_visual_review/queue.json')):
        errors.append('Pilot three-plane visual review is incomplete or stale')
    selected = []
    for entry in summary['cases']:
        c, d = entry['candidates']['C'], entry['candidates']['D']
        if c['status'] != 'measured' or d['status'] != 'measured':
            errors.append(f"Missing candidate for {entry['case_id']}")
            continue
        chosen = 'C' if not c['issues'] else ('D' if not d['issues'] else None)
        if chosen is None:
            errors.append(f"Both candidates violate QC: {entry['case_id']}")
            continue
        if entry['case_id'] in pilot['groups']['normal_train'] and 'v1_dice' in entry:
            for m in ('T1w','T2w'):
                delta = entry['candidates'][chosen]['metrics'][m]['final']['dice'] - entry['v1_dice'][m]
                if delta < -.02:
                    errors.append(f"Normal control Dice regressed >.02: {entry['case_id']}/{m}")
        selected.append({'case_id':entry['case_id'], 'candidate':chosen})
    if visual.get('accepted_choices') != {x['case_id']:x['candidate'] for x in selected}:
        errors.append('Pilot visual review did not cover the selected candidates')
    # Compare independent processes and worker counts using identical seed/threads.
    ids = ('BraTS-GLI-00188-000','BraTS-GLI-01351-000','BraTS-GLI-00540-000','BraTS-MEN-00161-000')
    points = [tuple(map(float,p)) for p in product((-90.,0.,90.), repeat=3)]
    maximum_error = 0.
    for case in ids:
        for name in ('rigid','affine','t2_to_t1','T1w_final','T2w_final'):
            a = plan/'pilot'/'C'/case/f'{name}.tfm'
            b = plan/'reproducibility'/'C'/case/f'{name}.tfm'
            if not a.exists() or not b.exists():
                errors.append(f'Missing repeated transform {case}/{name}')
                continue
            ta,tb = sitk.ReadTransform(str(a)),sitk.ReadTransform(str(b))
            maximum_error = max(maximum_error, max(float(np.linalg.norm(np.array(ta.TransformPoint(pt))-np.array(tb.TransformPoint(pt)))) for pt in points))
    if maximum_error > 1e-6:
        errors.append(f'Repeated transforms differ by {maximum_error:.8g} mm')
    root = ROOT/'assets/preprocessing/neuroqa_mni1mm_v2'
    template = root/'templates'/TEMPLATE
    decision = {'eligible':not errors, 'errors':errors, 'primary_candidate':'C', 'fallback_candidate':'D',
        'pilot_selected_cases':selected, 'maximum_repeated_transform_difference_mm':maximum_error,
        'note':'Fixed deterministic engineering gates; not clinical certification'}
    write_json(plan/'protocol_selection.json',decision)
    if errors:
        print(json.dumps(decision,ensure_ascii=False))
        raise SystemExit(1)
    if args.apply:
        frozen = {'status':'frozen','protocol':neuroqa_smri_v2.PROTOCOL,'seed':42,'threads':1,
            'primary_candidate':'C','fallback_candidate':'D','allow_rigid_fallback':False,
            'candidate_settings':{name:neuroqa_registration_v2.CANDIDATES[name] for name in ('C','D')},
            'registration_settings':{'metric':'Mattes_mutual_information','bins':50,'sampling_fraction':0.25,
                'iterations_per_level':1000,'convergence_minimum':1e-6,'convergence_window':10,
                'rigid_shrink':[4,2,1],'rigid_smoothing':[2,1,0],
                'affine_shrink':[2,1],'affine_smoothing':[1,0]},
            'intensity_policy':{'name':'native_nonzero_mask_then_sparse_negatives_to_positive_floor_v1',
                'max_negative_fraction':0.0001,'floor':'max(1e-6, positive_voxel_p0.5*0.001)'},
            'qc':POLICY, 'template_sha256':file_hash(template),
            'implementation_sha256':file_hash(neuroqa_smri_v2.__file__),
            'registration_code_sha256':file_hash(neuroqa_registration_v2.__file__),
            'reference_helpers_sha256':file_hash(neuroqa_smri.__file__),
            'pilot_summary_sha256':file_hash(plan/'pilot_summary.json'),
            'pilot_manifest_sha256':file_hash(plan/'pilot_manifest.json'),
            'sandbox_verified_sha256':file_hash(sandbox_path),
            'pilot_visual_review_sha256':file_hash(visual_path),
            'selection_sha256':file_hash(plan/'protocol_selection.json'),
            'scope':'research_benchmark_not_clinical_certification'}
        destination = root/'protocol.json'
        if destination.exists() and json.loads(destination.read_text()) != frozen:
            raise RuntimeError('An incompatible protocol already exists; never overwrite it')
        write_json(destination,frozen)
    print(json.dumps(decision,ensure_ascii=False))


if __name__ == '__main__':
    main()
