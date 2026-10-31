#!/usr/bin/env python3
"""Audit report-generation cache without modifying it or running registration."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
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
from src.preprocessing.neurovfm import NeuroVFMPreprocessor


def inspect(row, version, cache_root):
    result = {'id': row['id'], 'case_id': row['case_id'], 'cohort': row['cohort'], 'errors': []}
    path = ROOT / 'assets/data' / row['image']
    if not path.is_file():
        result['source_missing'] = True
        return result
    sha = file_hash(path)
    key = digest({'image': sha, 'cohort': row['cohort'], 'preprocessing': version})
    result['cache_key'] = key
    file = cache_root / key[:2] / f'{key}.pt'
    result['cached'] = file.is_file()
    if not result['cached']:
        return result
    try:
        value = torch.load(file, map_location='cpu', weights_only=True)
        assert value['key'] == key, 'Cache key mismatch'
        cached = PatchVolume(**value['value']).validate()
        m = cached.metadata
        assert m['preprocessing_version'] == version and m['image_sha256'] == sha
        assert m['spacing_policy'] == 'native_grid' and not m['resampling_applied'], 'Unexpected resampling'
        assert m['boundary_policy'] == 'minimal_pad' and m['crop_start_xyz'] == [0, 0, 0], 'Unexpected crop'
        assert m['crop_size_xyz'] == m['oriented']['size_xyz'], 'Changed physical extent'
        assert m['oriented'] == m['resampled'] == m['bounded_before_pad'], 'Spatial transform beyond reorientation'
        config = load_config([ROOT / 'configs/base.yaml'])
        config.preprocessing.cache_enabled = False
        pre = NeuroVFMPreprocessor(config)
        rebuilt = pre(SimpleNamespace(id=row['id'], cohort=row['cohort'], image=str(path)))
        assert cached.metadata == rebuilt.metadata, 'Metadata differs from fresh preprocessing'
        assert torch.equal(cached.patches, rebuilt.patches), 'Patch values differ from fresh preprocessing'
        assert torch.equal(cached.coords, rebuilt.coords), 'Coordinates differ from fresh preprocessing'
        source = sitk.ReadImage(str(path))
        fg = sitk.GetArrayFromImage(source) != 0
        face_counts = {f'{axis}_{side}': int(np.take(fg, index, axis=axis).sum())
                       for axis in range(3) for side, index in [('first', 0), ('last', -1)]}
        result.update(patch_recomputed_exact=True, no_added_spatial_crop=True,
            original_size_xyz=list(source.GetSize()), original_spacing_xyz=list(source.GetSpacing()),
            original_origin=list(source.GetOrigin()), original_direction=list(source.GetDirection()),
            shape_dhw=m['shape_dhw'], grid_dhw=m['grid_dhw'],
            source_boundary_foreground=face_counts, source_touches_boundary=any(face_counts.values()),
            pad_before_xyz=m['pad_before_xyz'], pad_after_xyz=m['pad_after_xyz'])
    except Exception as exc:
        result['errors'].append(repr(exc))
    return result


def summary(rows, files, expected):
    mapped = {r['cache_key'] for r in rows if r.get('cached')}
    cached = [r for r in rows if r.get('cached')]
    groups = {}
    for row in cached:
        geometry = tuple(json.dumps(row.get(k)) for k in ('original_size_xyz', 'original_spacing_xyz', 'original_origin', 'original_direction'))
        groups.setdefault(row['case_id'], set()).add(geometry)
    return {'indexed_source_rows': expected, 'inspected_rows': len(rows), 'cache_files': len(files),
        'mapped_cache_files': len(mapped), 'unmapped_cache_keys': sorted(files-mapped),
        'cache_rows_recomputed_exact': sum(r.get('patch_recomputed_exact', False) for r in cached),
        'error_rows': [r for r in rows if r['errors']],
        'source_missing_rows': sum(r.get('source_missing', False) for r in rows),
        'source_rows_without_cache': sum(not r.get('cached', False) for r in rows),
        'cohorts_in_cache': dict(Counter(r['cohort'] for r in cached)),
        'grids_in_cache': dict(Counter(str(r.get('grid_dhw')) for r in cached)),
        'cached_sources_touching_native_boundary': sum(r.get('source_touches_boundary', False) for r in cached),
        'cases_with_different_source_geometry': [k for k, v in groups.items() if len(v) > 1],
        'no_added_spatial_crop_in_all_cached_rows': bool(cached) and all(r.get('no_added_spatial_crop') for r in cached)}


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(2)
    torch.set_num_threads(1)
    config = load_config([ROOT / 'configs/base.yaml'])
    pre = NeuroVFMPreprocessor(config)
    root = ROOT / 'assets/preprocessing/report_generation_native1mm_v2'
    assert pre.cache.root == root, 'Base preprocessing points to a different cache'
    sources = read_rows(ROOT / 'assets/data/RadGenome-Brain/sequence_manifest.jsonl')
    files = {p.stem for p in root.glob('*/*.pt')}
    report = {'started_utc': datetime.now(timezone.utc).isoformat(), 'audit_script_sha256': file_hash(__file__),
              'cache_root': str(root), 'version': pre.version, 'scope': 'all_existing_cache_files_no_writes',
              'clinical_review_complete': False, 'rows': []}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(inspect, source, pre.version, root) for source in sources]
        for future in as_completed(futures):
            report['rows'].append(future.result())
            if len(report['rows']) % 100 == 0 or len(report['rows']) == len(sources):
                report['summary'] = summary(report['rows'], files, len(sources))
                write_json(args.output, report)
                print(json.dumps({k: report['summary'][k] for k in ('inspected_rows', 'mapped_cache_files', 'cache_rows_recomputed_exact', 'error_rows')}), flush=True)
    report['finished_utc'] = datetime.now(timezone.utc).isoformat()
    write_json(args.output, report)
    print(json.dumps(report['summary']), flush=True)


if __name__ == '__main__':
    main()
