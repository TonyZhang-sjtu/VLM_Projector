#!/usr/bin/env python3
"""Read-only audit of every indexed NeuroQA volume, including failed cases.

Unlike the processing gate, expanded-canvas coverage is measured even when the
registered mask does not touch an edge. This script never repairs data, changes
completion markers, or relaxes QC. Patch tensors are independently recomputed
from N4 images and compared with the existing cache (no cache writes).
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from itertools import product
import json
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
import numpy as np
import SimpleITK as sitk
import torch

from src.config import digest, file_hash, load_config
from src.data.neuroqa_manifest import read_rows, write_json
from src.preprocessing.base import PatchVolume
from src.preprocessing.registry import build_preprocessor
from src.preprocessing.neuroqa_smri import TEMPLATE, canvas_coverage, warp


def check(condition, message):
    if not condition:
        raise ValueError(message)


def same_geometry(a, b):
    return all(getattr(a, method)() == getattr(b, method)() for method in
               ('GetSize', 'GetSpacing', 'GetOrigin', 'GetDirection'))


def audit_case(source, root, config_file):
    folder = root / 'cases' / source['case_id']
    completed = (folder / 'complete.json').exists()
    result = {'case_id': source['case_id'], 'completed': completed,
              'integrity_errors': [], 'volumes': []}
    record = json.loads((folder / 'complete.json').read_text()) if completed else None
    template = sitk.ReadImage(str(root / 'templates' / TEMPLATE))
    template_fg = sitk.GetArrayFromImage(template) > 0
    if record:
        try:
            check(digest(record['contract']) == record['fingerprint'], 'Contract fingerprint changed')
            for name, expected in record['artifacts'].items():
                check(file_hash(folder / name) == expected, f'Artifact SHA256 mismatch: {name}')
            result['verified_artifact_hashes'] = len(record['artifacts'])
            if record['contract']['protocol'] == 'neuroqa_brats_foreground_mni1mm_v2':
                from src.preprocessing.neuroqa_smri_v2 import issues
                policy = json.loads((root/'protocol.json').read_text())['qc']
                selected = next(x for x in record['attempts'] if x['candidate'] == record['selected_candidate'] and 'metrics' in x)
                measured = dict(selected['metrics'])
                if record['transform_kind'] == 'rigid_only':
                    measured = {**measured, **{m:{**measured[m], 'final':measured[m]['rigid']} for m in ('T1w','T2w')}}
                check(not issues(measured, policy, include_stage=record['transform_kind'] != 'rigid_only'),
                      'Recorded selected registration violates the frozen v2 QC gate')
        except Exception as exc:
            result['integrity_errors'].append(repr(exc))
    config = load_config([config_file])
    config.preprocessing.cache_dir = str(root)
    config.preprocessing.cache_enabled = False
    pre = build_preprocessor(config)
    for modality in ('T1w', 'T2w'):
        row = {'modality': modality, 'errors': []}
        result['volumes'].append(row)
        try:
            native = sitk.ReadImage(source['images'][modality], sitk.sitkFloat32)
            data = sitk.GetArrayFromImage(native)
            row['source_sha256'] = file_hash(source['images'][modality])
            row['source_finite'] = bool(np.isfinite(data).all())
            row['negative_voxels'] = int((data < 0).sum())
            row['source_min'] = float(data.min())
            check(row['source_finite'], 'Source has NaN/Inf')
            if record:
                check(row['source_sha256'] == record['contract']['source_sha256'][modality], 'Source SHA256 changed')
            transform_path = folder / f'{modality}_mni_pull.tfm'
            if not transform_path.exists():
                row['spatial_audit'] = 'not_registered'
                continue
            transform = sitk.ReadTransform(str(transform_path))
            native_mask = sitk.Cast(native != 0, sitk.sitkUInt8)
            coverage = canvas_coverage(native_mask, template, transform)
            row.update(coverage)
            if record and record['contract']['protocol'] == 'neuroqa_brats_foreground_mni1mm_v2':
                axes = np.where(data != 0)
                bounds = [(int(v.min()),int(v.max())) for v in axes[::-1]]
                inverse = transform.GetInverse()
                extent = template.GetSize()
                potential = False
                for corner in product(*bounds):
                    point = native.TransformIndexToPhysicalPoint(tuple(corner))
                    mapped = template.TransformPhysicalPointToContinuousIndex(inverse.TransformPoint(point))
                    if any(value < -31. or value > extent[i]+30. for i,value in enumerate(mapped)):
                        potential = True
                        break
                row['native_bounding_box_may_exceed_expanded_qc_reference'] = potential
            # These are affine/rigid composites: recover the full linear map.
            offset = np.array(transform.TransformPoint((0., 0., 0.)))
            matrix = np.column_stack([np.array(transform.TransformPoint(tuple(v))) - offset for v in np.eye(3)])
            row['pull_determinant'] = float(np.linalg.det(matrix))
            row['pull_singular_values'] = np.linalg.svd(matrix, compute_uv=False).tolist()
            stored_mask = sitk.ReadImage(str(folder / f'{modality}_mni_mask.nii.gz'))
            recomputed_mask = warp(native_mask, template, transform, mask=True)
            a = sitk.GetArrayFromImage(stored_mask) > 0
            check(same_geometry(stored_mask, template), 'Mask differs from MNI template geometry')
            check(np.array_equal(a, sitk.GetArrayFromImage(recomputed_mask) > 0), 'Saved mask differs from transform of original foreground')
            row['mask_transform_exact'] = True
            row['dice'] = float(2 * np.logical_and(a, template_fg).sum() / (a.sum() + template_fg.sum()))
            if record and record['contract']['protocol'] == 'neuroqa_brats_foreground_mni1mm_v2':
                check(abs(row['dice'] - record['qc'][modality]['dice']) < 1e-12, 'Recomputed v2 Dice changed')
                check(coverage['outside_canvas_voxels'] == record['qc'][modality]['outside_canvas_voxels'],
                      'Recomputed v2 canvas coverage changed')
            row['edge_contact'] = bool(any(np.take(a, i, axis=k).any() for k in range(3) for i in (0, -1)))
            row['existing_gate_pass'] = row['dice'] >= .65 and coverage['outside_canvas_fraction'] <= .001 and not coverage['expanded_boundary_touched']
            # Review ranking only: does NOT modify the accepted QC threshold.
            row['priority_visual_review'] = row['dice'] < .85
            if not record:
                row['spatial_audit'] = 'available_intermediates_checked_no_completion_marker'
                continue
            image_path = folder / f'{modality}_mni_n4.nii.gz'
            image = sitk.ReadImage(str(image_path), sitk.sitkFloat32)
            values = sitk.GetArrayFromImage(image)
            check(same_geometry(image, template), 'N4 image differs from MNI geometry')
            check(np.isfinite(values).all(), 'N4 has NaN/Inf')
            check(np.all(values[~a] == 0), 'N4 background is not zero')
            check(values[a].std() > 0, 'Constant brain intensities')
            if record['contract']['protocol'] == 'neuroqa_brats_foreground_mni1mm_v2':
                raw_warped = sitk.ReadImage(str(folder / f'{modality}_mni_warped.nii.gz'), sitk.sitkFloat32)
                check(same_geometry(raw_warped,template), 'Warped image differs from MNI geometry')
                raw_values = sitk.GetArrayFromImage(raw_warped)
                eligible = a & (raw_values > 1e-6)
                check(eligible.any() and np.isfinite(raw_values).all(), 'Invalid warped image for N4 check')
                ratio = values[eligible].astype(np.float64) / raw_values[eligible]
                check(np.isfinite(ratio).all(), 'N4 correction ratio has NaN/Inf')
                row['n4_to_warped_ratio_p01_p50_p99'] = np.percentile(ratio,[1,50,99]).tolist()
                row['n4_extreme_correction_review'] = bool(row['n4_to_warped_ratio_p01_p50_p99'][0] < .05 or
                                                          row['n4_to_warped_ratio_p01_p50_p99'][2] > 20.)
            sample = SimpleNamespace(id=f'{source["case_id"]}__{modality}',
                cohort=source['cohort'], image=str(image_path), mask=str(folder / f'{modality}_mni_mask.nii.gz'))
            recomputed = pre(sample)
            cache_key = recomputed.metadata['cache_key']
            cached = pre.cache.get(cache_key)
            check(cached is not None, 'Patch cache missing')
            cached = PatchVolume(**cached).validate()
            check(cached.patches.shape == (9555, 1024), 'Wrong patch shape')
            check(cached.metadata == recomputed.metadata, 'Cached metadata differs from fresh adapter')
            check(torch.equal(cached.coords, recomputed.coords), 'Cached patch coordinates differ')
            row['patch_max_absolute_difference'] = float((cached.patches - recomputed.patches).abs().max())
            check(torch.equal(cached.patches, recomputed.patches), 'Cached patches differ from fresh adapter')
            row['cache_key'] = cache_key
            row['patch_recomputed_exact'] = True
            row['grid_dhw'] = cached.metadata['grid_dhw']
            row['spatial_audit'] = 'complete'
        except Exception as exc:
            row['errors'].append(repr(exc))
    return result


def summarize(cases, expected):
    volumes = [dict(case_id=c['case_id'], completed=c['completed'], **v) for c in cases for v in c['volumes']]
    complete = [v for v in volumes if v['completed']]
    spatial = [v for v in volumes if 'dice' in v]
    return {'expected_cases': expected, 'audited_cases': len(cases),
        'completed_cases': sum(c['completed'] for c in cases),
        'completed_volumes': len(complete),
        'incomplete_cases': [c['case_id'] for c in cases if not c['completed']],
        'artifact_hashes_verified': sum(c.get('verified_artifact_hashes', 0) for c in cases),
        'integrity_error_cases': [c['case_id'] for c in cases if c['integrity_errors'] or any(v['errors'] for v in c['volumes'])],
        'all_completed_patches_recomputed_exact': len(complete) > 0 and all(v.get('patch_recomputed_exact') for v in complete),
        'expanded_canvas_checked_volumes': len(spatial),
        'completed_edge_contact_volumes': sum(v.get('edge_contact', False) for v in complete),
        'completed_outside_canvas_volumes': sum(v.get('outside_canvas_voxels', 0) > 0 for v in complete),
        'completed_outside_without_edge_contact': [v for v in complete if v.get('outside_canvas_voxels', 0) > 0 and not v.get('edge_contact')],
        'completed_max_outside_fraction': max((v.get('outside_canvas_fraction', 0) for v in complete), default=None),
        'completed_dice_quantiles_0_5_50_95_100': np.percentile([v['dice'] for v in complete if 'dice' in v], [0, 5, 50, 95, 100]).tolist() if complete else [],
        'completed_failing_existing_gate': [v for v in complete if not v.get('existing_gate_pass', False)],
        'priority_visual_review': [v for v in spatial if v['priority_visual_review']],
        'ready_for_full_training': len(cases) == expected and all(c['completed'] and not c['integrity_errors'] and all(not v['errors'] and v.get('existing_gate_pass') and v.get('patch_recomputed_exact') for v in c['volumes']) for c in cases)}


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--root', type=Path, default=ROOT / 'assets/preprocessing/neuroqa_mni1mm_v1')
    p.add_argument('--config', type=Path, default=ROOT / 'configs/vqa/neuroqa/mixed.yaml')
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(2)
    torch.set_num_threads(1)
    sources = read_rows(ROOT / 'assets/data/NeuroQA_derived/official_holdout_patient_guard_v1/source_index.jsonl')
    report = {'started_utc': datetime.now(timezone.utc).isoformat(),
              'audit_script_sha256': file_hash(__file__), 'scope': 'read_only_integrity_geometry_coverage_and_exact_patch_rebuild',
              'clinical_review_complete': False, 'cases': []}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(audit_case, source, args.root, args.config): source['case_id'] for source in sources}
        for future in as_completed(futures):
            report['cases'].append(future.result())
            if len(report['cases']) % 10 == 0 or len(report['cases']) == len(sources):
                report['summary'] = summarize(report['cases'], len(sources))
                write_json(args.output, report)
                print(json.dumps({k: report['summary'][k] for k in ('audited_cases', 'completed_cases', 'integrity_error_cases')}), flush=True)
    report['cases'].sort(key=lambda row: row['case_id'])
    report['finished_utc'] = datetime.now(timezone.utc).isoformat()
    write_json(args.output, report)
    print(json.dumps(report['summary'], ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
