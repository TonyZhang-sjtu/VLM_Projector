"""Validated read-only sidecar, lazily loaded only for supervised training."""
import json
from pathlib import Path
from .build import read_jsonl, text_hash
from .schema import FIELDS, VERSION
from src.config import file_hash


class FactStore:
    def __init__(self,path):
        path=Path(path)
        metadata=json.loads((path/'metadata.json').read_text())
        if metadata['version']!=VERSION:raise ValueError('Unknown facts version')
        for name,expected in metadata['files'].items():
            if file_hash(path/name)!=expected:raise ValueError(f'Facts checksum mismatch: {name}')
        self.metadata=metadata
        self.rows={r['id']:r for r in read_jsonl(path/'train.jsonl')}
        self.pairs={r['id']:r['pairs'] for r in read_jsonl(path/'pairs.jsonl')}
        for rid,row in self.rows.items():
            if row['split']!='train':raise ValueError('Non-training label in facts store')
            for field,fact in row['facts'].items():
                if field not in FIELDS or fact['value'] not in FIELDS[field]['values']:
                    raise ValueError('Invalid fact label')
            for pair in self.pairs.get(rid,[]):
                other=self.rows[pair['partner_id']];field=pair['field']
                if (other['patient_id']==row['patient_id'] or other['cohort']!=row['cohort']
                        or other['sequence']!=row['sequence'] or pair['positive']==pair['negative']
                        or row['facts'][field]['value']!=pair['positive']
                        or other['facts'][field]['value']!=pair['negative']):
                    raise ValueError('Invalid factual negative')
                if field=='signal' and 'mixed_signal' in (pair['positive'],pair['negative']):
                    raise ValueError('Mixed signal is not an exclusive factual negative')
                inverse=[p for p in self.pairs.get(other['id'],[]) if p['pair_id']==pair['pair_id']]
                if len(inverse)!=1 or inverse[0]['partner_id']!=rid or inverse[0]['negative']!=pair['positive']:
                    raise ValueError('Pair is not reciprocal')

    def get(self,sample):
        if sample.split!='train':raise ValueError('Auxiliary labels requested outside training')
        row=self.rows.get(sample.id)
        if row is None or row['finding_sha256']!=text_hash(sample.finding):
            raise ValueError(f'Missing or stale facts for {sample.id}')
        if (row['patient_id'],row['cohort'],row['sequence'])!=(sample.patient_id,sample.cohort,sample.sequence):
            raise ValueError('Facts/sample identity mismatch')
        return row,self.pairs.get(sample.id,[])


def preflight(config):
    """Fail before loading 8B weights if labels or the training reference changed."""
    from src.config import path_of
    p=config.projector.params.get('report_facts',{})
    if not p.get('enabled',False):return None
    store=FactStore(path_of(config,p['path']))
    manifest=path_of(config,config.data.manifest_dir)/'train.jsonl'
    if file_hash(manifest)!=store.metadata['manifest_sha256']:
        raise ValueError('Training manifest changed; rebuild a separately versioned facts sidecar')
    return store
