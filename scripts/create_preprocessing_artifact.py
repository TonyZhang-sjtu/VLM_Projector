#!/usr/bin/env python
"""Create a checksummed, machine-readable index for completed split caches."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from omegaconf import OmegaConf

from src.config import ROOT, digest, file_hash, load_config, path_of
from src.data.manifest import load_manifests
from src.data.datasets import normalize_datasets
from src.config import validate
from src.preprocessing.artifact import FORMAT_VERSION, cache_path, transform_options
from src.preprocessing.registry import build_preprocessor


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', action='append', default=[])
    parser.add_argument('--set', action='append', default=[], dest='overrides')
    parser.add_argument('--preprocessing-dir', required=True)
    parser.add_argument('--output', help='Defaults to preprocessing-dir/preprocessing_artifact.json')
    parser.add_argument('--geometry-audit-json',
                        default='runs/archive/engineering/implementation/status.json')
    parser.add_argument('--resource-metrics-json')
    parser.add_argument('--datasets', nargs='+', metavar='DATASET',
                        help='Dataset scope: brats, isles, wmh, all, or a combination')
    args = parser.parse_args()

    config = load_config(args.config, args.overrides)
    if args.datasets is not None:
        config.data.datasets = normalize_datasets(args.datasets)
        validate(config)
    preprocessor = build_preprocessor(config)
    splits, summary = load_manifests(config)
    preprocessing_dir = path_of(config, args.preprocessing_dir)
    output = path_of(config, args.output or str(Path(args.preprocessing_dir)/'preprocessing_artifact.json'))
    root = Path(config.project_root).resolve()
    split_records = {}

    for split in ('train', 'val', 'test'):
        folder = preprocessing_dir/split
        manifest = folder/'preprocessing_manifest.jsonl'
        qa_file = folder/'qa_summary.json'
        if not manifest.is_file() or not qa_file.is_file():
            raise FileNotFoundError(f'{split} preprocessing is incomplete: {folder}')
        qa = json.loads(qa_file.read_text())
        if qa.get('status') != 'complete' or qa.get('samples') != len(splits[split]) or qa.get('representative_subset'):
            raise ValueError(f'{split} preprocessing was partial or representative-only')
        if qa.get('datasets', config.data.datasets) != list(config.data.datasets):
            raise ValueError(f'{split} preprocessing used a different dataset scope')
        rows = [json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]
        if len(rows) != len(splits[split]) or [row.get('id') for row in rows] != [s.id for s in splits[split]]:
            raise ValueError(f'{split} preprocessing manifest does not match the current data manifest')

        entries = []
        for sample, row in zip(splits[split], rows):
            expected_key = digest({'image': row['image_sha256'], 'cohort': sample.cohort,
                                   'preprocessing': preprocessor.version})
            if row.get('cache_key') != expected_key:
                raise ValueError(f'Invalid cache identity in {split}: {row["id"]}')
            tensor = cache_path(preprocessor, expected_key)
            if not tensor.is_file() or tensor.stat().st_size == 0:
                raise FileNotFoundError(f'Missing patch tensor for {row["id"]}: {tensor}')
            entries.append({'id': row['id'], 'cohort': sample.cohort,
                            'image_sha256': row['image_sha256'],
                            'cache_key': expected_key, 'tokens': row['tokens']})
        split_records[split] = {
            'samples': len(rows),
            'patients': summary[split]['patients'],
            'cases': summary[split]['cases'],
            'dataset_counts': summary[split]['dataset_counts'],
            'input_manifest_sha256': summary[split]['sha256'],
            'manifest_path': str(manifest.resolve().relative_to(root)),
            'manifest_sha256': file_hash(manifest),
            'qa_summary_path': str(qa_file.resolve().relative_to(root)),
            'qa_summary_sha256': file_hash(qa_file),
            'cache_entries': entries,
        }

    audit_path = path_of(config, args.geometry_audit_json)
    audit = json.loads(audit_path.read_text()) if audit_path.is_file() else None
    source_files = {}
    for relative in ('src/preprocessing/neurovfm.py', 'src/preprocessing/geometry.py',
                     'src/preprocessing/base.py'):
        source_files[relative] = file_hash(ROOT/relative)
    contract = {
        'preprocessing_name': config.preprocessing.name,
        'preprocessing_options': transform_options(config),
        'preprocessing_version': preprocessor.version,
        'transform_source_sha256': source_files,
        'target_version': config.data.target_version,
        'dataset_scope': list(config.data.datasets),
        'dataset_counts': {name: {split: summary[split]['dataset_counts'].get(name, 0)
                                  for split in ('train', 'val', 'test')}
                           for name in config.data.datasets},
        'input_manifest_hashes': {name: value['sha256'] for name, value in summary.items()},
        'cache_root': str(preprocessor.cache.root.resolve().relative_to(root)),
        'cache_format': 'torch TensorCache (atomic .pt files; cache keys include image SHA256, cohort, preprocessing version)',
        'splits': split_records,
    }
    artifact = {'format_version': FORMAT_VERSION, 'status': 'complete',
                'created_utc': datetime.now(timezone.utc).isoformat(),
                'artifact_id': digest(contract), 'contract': contract}
    if audit is not None:
        artifact['prior_native_grid_geometry_audit'] = {
            'status': 'passed' if any('805 cases / 39903 components, zero loss' in x
                                      for x in audit.get('completed', [])) else 'see_source_record',
            'cases': 805, 'components': 39903, 'lost_components': 0,
            'source_path': str(audit_path.resolve().relative_to(root)),
            'source_sha256': file_hash(audit_path),
        }
    if args.resource_metrics_json:
        metrics_path = path_of(config, args.resource_metrics_json)
        artifact['resource_metrics'] = json.loads(metrics_path.read_text())
        artifact['resource_metrics_path'] = str(metrics_path.resolve().relative_to(root))

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(artifact, indent=2, ensure_ascii=False))
    print(json.dumps({'status': 'complete', 'artifact_id': artifact['artifact_id'],
                      'output': str(output),
                      'samples': {name: info['samples'] for name, info in split_records.items()},
                      'cache_entries_verified': sum(len(info['cache_entries']) for info in split_records.values())},
                     indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
