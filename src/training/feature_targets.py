"""Immutable-teacher targets stored separately from live LoRA encoder caches."""
import json
import os
from pathlib import Path
import tempfile
import torch
from src.config import digest, file_hash
from src.preprocessing.cache import atomic_torch_save

VERSION = 'bindformer_frozen_teacher_v1'


class FrozenFeatureTargets:
    def __init__(self, directory, encoder_fingerprint, feature_dim):
        self.directory = Path(directory)
        self.encoder_fingerprint,self.feature_dim = encoder_fingerprint,feature_dim
        self._manifest = None

    def manifest(self):
        # Lazy: generate() never needs a target cache, even for an auxiliary-trained model.
        path = self.directory/'manifest.json'
        if not path.exists():
            raise FileNotFoundError(f'Missing frozen feature targets: {path}; run scripts.cache_bindformer_teacher')
        current_hash = file_hash(path)
        if self._manifest is None:
            value = json.loads(path.read_text())
            if (value.get('version') != VERSION or value.get('teacher') != 'frozen_pretrained_neurovfm'
                    or value.get('encoder_fingerprint') != self.encoder_fingerprint
                    or value.get('feature_dim') != self.feature_dim):
                raise ValueError('Frozen teacher cache contract mismatch')
            self._manifest,self.manifest_hash = value,current_hash
        elif current_hash != self.manifest_hash:
            raise ValueError('Frozen teacher manifest changed during this run')
        return self._manifest

    def provenance(self):
        return {**self.manifest(),'manifest_sha256':self.manifest_hash}

    def target_path(self, metadata):
        manifest = self.manifest()
        if metadata.get('preprocessing_version') != manifest['preprocessing_version']:
            raise ValueError('Frozen teacher preprocessing differs from current input')
        key = metadata.get('cache_key')
        if not isinstance(key,str) or not key:
            raise ValueError('Frozen targets require a preprocessed input cache key')
        identity = digest({'contract':digest(manifest),'preprocessed':key})
        return self.directory/'features'/identity[:2]/f'{identity}.pt',identity

    def load(self, coords, metadata):
        path,identity = self.target_path(metadata)
        if not path.exists():
            raise FileNotFoundError(f'Frozen teacher target missing: {path}')
        value = torch.load(path,map_location='cpu',weights_only=True)
        features = value.get('features')
        if (value.get('identity') != identity or not isinstance(features,torch.Tensor)
                or features.shape != (len(coords),self.feature_dim) or not features.is_floating_point()
                or not torch.isfinite(features).all()
                or not isinstance(value.get('coords'),torch.Tensor)
                or not torch.equal(value['coords'],coords.detach().cpu())):
            raise ValueError('Frozen teacher feature/coordinate/identity mismatch')
        return features

    def save(self, features, coords, metadata):
        path,identity = self.target_path(metadata)
        if features.shape != (len(coords),self.feature_dim) or not torch.isfinite(features).all():
            raise ValueError('Invalid teacher export')
        atomic_torch_save({'identity':identity,'features':features.detach().cpu().contiguous(),
                           'coords':coords.detach().cpu().contiguous()},path)
        return path


def initialize_target_cache(directory, encoder_fingerprint, feature_dim, preprocessing_version):
    directory = Path(directory);directory.mkdir(parents=True,exist_ok=True)
    value = {'version':VERSION,'teacher':'frozen_pretrained_neurovfm',
             'encoder_fingerprint':encoder_fingerprint,'feature_dim':feature_dim,
             'preprocessing_version':preprocessing_version}
    path = directory/'manifest.json'
    # Publish complete bytes atomically without replacing an existing contract.
    # Concurrent exporters/readers can never observe a partially written JSON.
    fd,temporary = tempfile.mkstemp(prefix='.manifest.',dir=directory)
    try:
        with os.fdopen(fd,'w') as handle:
            json.dump(value,handle,indent=2);handle.write('\n')
            handle.flush();os.fsync(handle.fileno())
        try:
            os.link(temporary,path)
        except FileExistsError:
            if json.loads(path.read_text()) != value:
                raise ValueError('Existing teacher cache has a different contract; choose a new directory')
    finally:
        os.unlink(temporary)
    return FrozenFeatureTargets(directory,encoder_fingerprint,feature_dim)
