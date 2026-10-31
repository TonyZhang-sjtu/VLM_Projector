"""Model adapter for completed MNI artifacts; registration never runs in training."""
import json
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import torch
from einops import rearrange

from src.config import digest, file_hash, path_of
from src.preprocessing.base import PatchVolume
from src.preprocessing.cache import TensorCache
from src.preprocessing.geometry import transform_image
from src.preprocessing.neuroqa_smri import PROTOCOL, TEMPLATE, IMPLEMENTATION_SHA256, compatible_contract


class NeuroQAPreprocessor:
    def __init__(self, config):
        self.config = config.preprocessing
        self.root = path_of(config, self.config.cache_dir)
        template = self.root / 'templates' / TEMPLATE
        self.template_hash = file_hash(template)
        self.spatial_contract = {'protocol': PROTOCOL,
            'seed': self.config.params.get('registration_seed', 42),
            'threads': self.config.params.get('registration_threads', 2),
            'template_sha256': self.template_hash,
            'implementation_sha256': IMPLEMENTATION_SHA256,
            'SimpleITK': sitk.Version_VersionString(), 'numpy': np.__version__,
            'registration': {'bins': 50, 'sampling': 0.25, 'iterations': 1000},
            'n4': 'SimpleITK_N4BiasFieldCorrection_defaults_full_resolution',
            'brain_mask': 'native_image_nonzero_no_tumor_labels', 'extra_resample_4mm': False}
        self.version = digest({'protocol': PROTOCOL, 'template': self.template_hash,
            'spatial_contract': self.spatial_contract,
            'model_adapter': file_hash(__file__), 'spatial_processing': file_hash(Path(__file__).with_name('neuroqa_smri.py')),
            'orientation': self.config.orientation, 'padding': 'minimal_pad',
            'percentiles': list(self.config.intensity_percentiles),
            'normalization': [self.config.mri_mean, self.config.mri_std], 'grid': [49, 15, 13]})
        self.cache = TensorCache(self.root / 'patches')
        self.checked = {}

    def __call__(self, sample):
        image_path, mask_path = Path(sample.image), Path(sample.mask)
        if not image_path.resolve().is_relative_to((self.root / 'cases').resolve()):
            raise ValueError('NeuroQA input outside the configured preprocessing store')
        complete = image_path.parent / 'complete.json'
        record = json.loads(complete.read_text())
        if not compatible_contract({k: v for k,v in record['contract'].items() if k != 'source_sha256'}, self.spatial_contract):
            raise ValueError('NeuroQA registration settings/dependencies differ from the configured protocol')
        if record['contract']['protocol'] != PROTOCOL or record['contract']['template_sha256'] != self.template_hash:
            raise ValueError('NeuroQA preprocessing protocol/template mismatch')
        hashes = []
        for path in (image_path, mask_path):
            stat = path.stat()
            key = (str(path), stat.st_size, stat.st_mtime_ns)
            if key not in self.checked:
                self.checked[key] = file_hash(path)
            sha = self.checked[key]
            if record['artifacts'].get(path.name) != sha:
                raise ValueError(f'Changed preprocessing artifact: {path}')
            hashes.append(sha)
        key = digest({'image_mask': hashes, 'spatial_fingerprint': record['fingerprint'], 'adapter': self.version})
        cached = self.cache.get(key) if self.config.cache_enabled else None
        if cached is not None:
            return PatchVolume(**cached).validate()
        image = sitk.ReadImage(str(image_path), sitk.sitkFloat32)
        mask = sitk.ReadImage(str(mask_path), sitk.sitkUInt8)
        if (image.GetSize() != (193, 229, 193) or image.GetSpacing() != (1., 1., 1.) or
                image.GetSize() != mask.GetSize() or image.GetOrigin() != mask.GetOrigin() or
                image.GetDirection() != mask.GetDirection() or image.GetSpacing() != mask.GetSpacing()):
            raise ValueError('Image/mask do not share the expected MNI 1 mm canvas')
        bounded, meta = transform_image(image, self.config, sample.cohort)
        orienter = sitk.DICOMOrientImageFilter()
        orienter.SetDesiredCoordinateOrientation(self.config.orientation)
        brain = sitk.GetArrayFromImage(orienter.Execute(mask)).transpose(meta['array_zyx_to_dhw']) > 0
        values = sitk.GetArrayFromImage(bounded).transpose(meta['array_zyx_to_dhw']).astype(np.float64)
        lo, hi = np.percentile(values[brain], list(self.config.intensity_percentiles))
        if not np.isfinite(values).all() or hi <= lo:
            raise ValueError('Invalid NeuroQA normalized intensity range')
        values = np.where(brain, (np.clip(values, lo, hi) - lo) / (hi-lo), 0.)
        pads_zyx = list(zip(meta['pad_before_xyz'][::-1], meta['pad_after_xyz'][::-1]))
        values = np.pad(values, [pads_zyx[i] for i in meta['array_zyx_to_dhw']], constant_values=0)
        if values.shape != (196, 240, 208) or meta['resampling_applied']:
            raise ValueError('MNI adapter must only reorient and pad, with no extra resampling')
        patches = rearrange(torch.from_numpy(values).float(), '(d p) (h q) (w r) -> (d h w) (p q r)', p=4, q=16, r=16)
        patches = (patches - self.config.mri_mean) / self.config.mri_std
        grid = [49, 15, 13]
        coords = torch.cartesian_prod(*(torch.arange(n) for n in grid))
        meta.update({'preprocessing_version': self.version, 'spatial_fingerprint': record['fingerprint'],
                     'image_sha256': hashes[0], 'mask_sha256': hashes[1], 'cache_key': key,
                     'shape_dhw': list(values.shape), 'grid_dhw': grid, 'tokens': len(patches),
                     'percentile_values': [float(lo), float(hi)], 'intensity_scope': 'brain_foreground',
                     'normalization_applied_once': True, 'token_filter_policy': 'none',
                     'background_before_encoder_normalization': 0, 'extra_resample_4mm': False})
        result = PatchVolume(patches.contiguous(), coords, meta).validate()
        if self.config.cache_enabled:
            self.cache.put(key, {'patches': result.patches, 'coords': result.coords, 'metadata': meta})
        return result
