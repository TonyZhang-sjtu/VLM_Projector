"""Versioned manifests and strict verification for reusable preprocessing outputs."""
import json
from pathlib import Path

from tqdm.auto import tqdm

from src.config import digest, file_hash, path_of
from src.data.datasets import ALL_DATASETS, normalize_datasets


FORMAT_VERSION = 1


def transform_options(config):
    from omegaconf import OmegaConf

    options = OmegaConf.to_container(config.preprocessing, resolve=True)
    # Storage location and cache enablement do not alter the tensor values.
    options.pop('cache_dir', None)
    options.pop('cache_enabled', None)
    return options


def expected_cache_entry(preprocessor, sample):
    image_sha256 = file_hash(sample.image)
    key = digest({'image': image_sha256, 'cohort': sample.cohort,
                  'preprocessing': preprocessor.version})
    return image_sha256, key


def cache_path(preprocessor, key):
    return preprocessor.cache.root / key[:2] / f'{key}.pt'


def validate_artifact_contract(config, preprocessor, samples, manifest_summary,
                               split, artifact_path):
    """Check settings, source data and each patch-cache entry without loading tensors."""
    artifact_path = Path(artifact_path).resolve()
    artifact = json.loads(artifact_path.read_text())
    if artifact.get('format_version') != FORMAT_VERSION or artifact.get('status') != 'complete':
        raise ValueError('Unknown or incomplete preprocessing artifact')
    contract = artifact['contract']
    if artifact.get('artifact_id') != digest(contract):
        raise ValueError('Preprocessing artifact contract checksum is invalid')
    mismatches = []
    if contract.get('preprocessing_version') != preprocessor.version:
        mismatches.append('preprocessing options/code version')
    if contract.get('preprocessing_options') != transform_options(config):
        mismatches.append('preprocessing options')
    if contract.get('target_version') != config.data.target_version:
        mismatches.append('target version')
    artifact_scope = normalize_datasets(contract.get('dataset_scope'), default=ALL_DATASETS)
    configured_scope = getattr(config.data, 'datasets', ALL_DATASETS)
    if artifact_scope != normalize_datasets(configured_scope, default=ALL_DATASETS):
        mismatches.append('dataset scope')
    current_hashes = {name: value['sha256'] for name, value in manifest_summary.items()}
    if contract.get('input_manifest_hashes') != current_hashes:
        mismatches.append('input train/val/test manifests')
    recorded_cache = (Path(config.project_root) / contract.get('cache_root', '')).resolve()
    if preprocessor.cache.root.resolve() != recorded_cache:
        mismatches.append('preprocessing cache location')
    if not config.preprocessing.cache_enabled:
        mismatches.append('preprocessing.cache_enabled must be true to reuse patch tensors')
    if split not in contract.get('splits', {}):
        mismatches.append(f'{split} split is absent from artifact')
    if mismatches:
        raise ValueError('Preprocessing artifact does not match: ' + ', '.join(mismatches))

    entry = contract['splits'][split]
    source_manifest = path_of(config, entry['manifest_path'])
    if not source_manifest.is_file() or file_hash(source_manifest) != entry['manifest_sha256']:
        raise ValueError(f'Cached {split} preprocessing manifest is missing or changed')
    source_qa = source_manifest.parent/'qa_summary.json'
    if not source_qa.is_file() or file_hash(source_qa) != entry.get('qa_summary_sha256'):
        raise ValueError(f'Cached {split} completion summary is missing or changed')
    qa = json.loads(source_qa.read_text())
    if qa.get('status') != 'complete' or qa.get('samples') != entry['samples']:
        raise ValueError(f'Cached {split} completion summary does not cover the full split')
    if len(samples) != entry['samples']:
        raise ValueError(f'{split} sample count mismatch: expected {entry["samples"]}, got {len(samples)}')
    rows = [json.loads(line) for line in source_manifest.read_text().splitlines() if line.strip()]
    if len(rows) != len(samples) or [row.get('id') for row in rows] != [sample.id for sample in samples]:
        raise ValueError(f'{split} sample IDs/order do not match the saved preprocessing manifest')
    entries = entry.get('cache_entries', [])
    if len(entries) != len(rows):
        raise ValueError(f'{split} cache index count does not match its preprocessing manifest')

    missing = []
    for sample, row, indexed in tqdm(zip(samples, rows, entries), total=len(samples),
                            desc=f'verify {split} cache', unit='volume', dynamic_ncols=True):
        image_sha256, key = expected_cache_entry(preprocessor, sample)
        if row.get('image_sha256') != image_sha256 or row.get('cache_key') != key:
            raise ValueError(f'Input image or cache identity changed: {sample.id}')
        if any(indexed.get(field) != value for field, value in
               (('id', sample.id), ('cohort', sample.cohort), ('image_sha256', image_sha256),
                ('cache_key', key), ('tokens', row.get('tokens')))):
            raise ValueError(f'Artifact sample index mismatch: {sample.id}')
        tensor_path = cache_path(preprocessor, key)
        if not tensor_path.is_file() or tensor_path.stat().st_size == 0:
            missing.append(sample.id)
    if missing:
        raise FileNotFoundError(f'{len(missing)} preprocessing cache entries are missing; first IDs: {missing[:10]}')
    return {'artifact_id': artifact['artifact_id'], 'split': split, 'samples': len(samples),
            'checked_image_hashes': len(samples), 'cache_entries_present': len(samples),
            'source_manifest': str(source_manifest), 'source_manifest_sha256': entry['manifest_sha256']}
