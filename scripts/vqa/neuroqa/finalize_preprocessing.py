#!/usr/bin/env python3
"""Verify completed MRI artifacts and materialize all model patch caches.

Optional --wait-for-batch watches this project's batch report. A terminated batch
with incomplete coverage is reported as failure; missing cases are never omitted
from the denominator or silently accepted for full training.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from src.config import digest, file_hash, load_config
from src.data.neuroqa_manifest import read_rows, write_json
from src.preprocessing.registry import build_preprocessor


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--wait-for-batch', type=Path)
    p.add_argument('--config', type=Path, default=ROOT / 'configs/vqa/neuroqa/mixed.yaml')
    p.add_argument('--pid-file', type=Path, default=ROOT / 'assets/preprocessing/neuroqa_mni1mm_v1/logs/full_preprocessing.pid')
    args = p.parse_args()
    if args.wait_for_batch:
        while True:
            report = json.loads(args.wait_for_batch.read_text()) if args.wait_for_batch.exists() else {}
            if report.get('requested') is not None and len(report.get('successful', [])) + len(report.get('failed', [])) == report['requested']:
                break
            try:
                os.kill(int(args.pid_file.read_text()), 0)
            except (ProcessLookupError, FileNotFoundError):
                raise RuntimeError('Preprocessing worker exited before completing its declared batch')
            time.sleep(15)
    config = load_config([args.config])
    pre = build_preprocessor(config)
    index = read_rows(ROOT / config.data.manifest_dir / 'source_index.jsonl')
    summary = {'expected_cases': len(index), 'expected_volumes': 2*len(index),
               'preprocessing_version': pre.version, 'verified_volumes': [], 'failed': []}
    root = ROOT / config.preprocessing.cache_dir
    for row in index:
        folder = root / 'cases' / row['case_id']
        for modality in ('T1w', 'T2w'):
            try:
                sample = SimpleNamespace(id=f'{row["case_id"]}__{modality}', case_id=row['case_id'], cohort=row['cohort'],
                    image=str(folder / f'{modality}_mni_n4.nii.gz'), mask=str(folder / f'{modality}_mni_mask.nii.gz'))
                volume = pre(sample)
                assert volume.patches.shape == (9555,1024) and volume.metadata['grid_dhw'] == [49,15,13]
                summary['verified_volumes'].append({'case_id':row['case_id'],'modality':modality,
                    'cache_key':volume.metadata['cache_key'],'spatial_fingerprint':volume.metadata['spatial_fingerprint'],
                    'tokens':len(volume.patches),'shape_dhw':volume.metadata['shape_dhw']})
            except Exception as exc:
                summary['failed'].append({'case_id':row['case_id'],'modality':modality,'error':repr(exc)})
        ready = len(summary['verified_volumes']) == summary['expected_volumes'] and not summary['failed']
        if config.preprocessing.name == 'neuroqa_mni1mm_v2':
            review_path = root / 'qc_review.json'
            review = json.loads(review_path.read_text()) if review_path.exists() else {}
            audit_path = root / 'independent_audit.json'
            intensity_path = root / 'intensity_audit.json'
            lesion_path = root / 'lesion_coverage_audit.json'
            reproducibility_path = root / 'production_reproducibility.json'
            protocol_path = root / 'protocol.json'
            valid_audit = False
            if audit_path.exists() and intensity_path.exists() and lesion_path.exists() and reproducibility_path.exists() and protocol_path.exists():
                audit = json.loads(audit_path.read_text())
                intensity = json.loads(intensity_path.read_text())
                lesion = json.loads(lesion_path.read_text())
                reproducibility = json.loads(reproducibility_path.read_text())
                a = audit.get('summary', {})
                valid_audit = (review.get('audit_sha256') == file_hash(audit_path)
                    and review.get('protocol_sha256') == digest(json.loads(protocol_path.read_text()))
                    and a.get('expected_cases') == len(index) and a.get('audited_cases') == len(index)
                    and intensity.get('complete') is True and intensity.get('expected_cases') == len(index)
                    and intensity.get('protocol_sha256') == review.get('protocol_sha256')
                    and review.get('intensity_audit_sha256') == file_hash(intensity_path)
                    and lesion.get('complete') is True and lesion.get('expected_cases') == len(index)
                    and lesion.get('protocol_sha256') == review.get('protocol_sha256')
                    and review.get('lesion_audit_sha256') == file_hash(lesion_path)
                    and reproducibility.get('status') == 'passed'
                    and reproducibility.get('protocol_sha256') == review.get('protocol_sha256')
                    and review.get('reproducibility_sha256') == file_hash(reproducibility_path)
                    and a.get('ready_for_full_training') is True and audit.get('finished_utc') is not None)
            summary['qc_review_complete'] = review.get('complete') is True and valid_audit
            ready = ready and summary['qc_review_complete']
        summary['ready_for_full_training'] = ready
        write_json(root / 'finalization.json', summary)
        print(json.dumps({'verified':len(summary['verified_volumes']), 'expected':summary['expected_volumes'],
                          'failed':len(summary['failed'])}), flush=True)
    if not summary['ready_for_full_training']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
