"""Immutable, split-aware sidecars. Targets never enter image projection/generation."""
import json
from pathlib import Path
from src.config import file_hash, path_of
from .build import read_rows, text_hash
from .schema import VERSION, FIELDS


class OutputStore:
    def __init__(self, path):
        path = Path(path)
        self.metadata = json.loads((path/'metadata.json').read_text())
        if self.metadata['version'] != VERSION:
            raise ValueError('Unknown output supervision schema')
        for name, sha in self.metadata['files'].items():
            if file_hash(path/name) != sha:
                raise ValueError(f'Output supervision checksum mismatch: {name}')
        self.rows = {}
        for split in ('train', 'val'):
            rows = read_rows(path/f'{split}.jsonl')
            if len({r['id'] for r in rows}) != len(rows):
                raise ValueError('Duplicate output supervision ID')
            self.rows[split] = {r['id']: r for r in rows}
            for row in rows:
                if row['split'] != split:
                    raise ValueError('Sidecar split mismatch')
                if split != 'train' and any(g['candidates'] for g in row['groups']):
                    raise ValueError('Validation sidecar contains training negatives')
                for f, v in row['facts'].items():
                    if f not in FIELDS or v['value'] not in FIELDS[f]['values']:
                        raise ValueError('Unknown report attribute')
        if ({r['patient_id'] for r in self.rows['train'].values()} &
            {r['patient_id'] for r in self.rows['val'].values()}):
            raise ValueError('Patient leakage in supervision')

    def get(self, sample, split='train'):
        if split not in self.rows or sample.split != split:
            raise ValueError('Auxiliary data requested from the wrong split')
        row = self.rows[split].get(sample.id)
        if row is None or row['finding_sha256'] != text_hash(sample.finding):
            raise ValueError(f'Missing/stale {split} output supervision: {sample.id}')
        if any(row[k] != getattr(sample, k) for k in ('patient_id', 'cohort', 'sequence')):
            raise ValueError('Report supervision identity mismatch')
        return row


def preflight(config):
    p = config.projector.params.get('report_output', {})
    if not p.get('enabled', False):
        return None
    if config.training.distributed != 'none':
        raise ValueError('report_output_v1 supports independent single-GPU jobs only')
    if set(config.data.datasets) != {'brats'}:
        raise ValueError('Build a new sidecar before enabling other cohorts')
    store = OutputStore(path_of(config, p['path']))
    for split, sha in store.metadata['manifest_hashes'].items():
        if file_hash(path_of(config, config.data.manifest_dir)/f'{split}.jsonl') != sha:
            raise ValueError(f'{split} manifest changed; build a new sidecar version')
    for relative, sha in store.metadata['source_hashes'].items():
        if file_hash(path_of(config, relative)) != sha:
            raise ValueError(f'Extraction rules changed: {relative}; rebuild a new sidecar')
    return store
