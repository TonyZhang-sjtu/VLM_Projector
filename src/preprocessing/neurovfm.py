import numpy as np
import SimpleITK as sitk
import torch
from einops import rearrange
from omegaconf import OmegaConf
from src.config import digest, file_hash, path_of
from src.preprocessing.base import PatchVolume
from src.preprocessing.cache import TensorCache
from src.preprocessing.geometry import transform_image


class NeuroVFMPreprocessor:
    def __init__(self, config):
        self.config = config.preprocessing
        options = OmegaConf.to_container(self.config, resolve=True)
        for key in ('cache_dir','cache_enabled'):
            options.pop(key)
        from pathlib import Path
        self.version = digest({'options':options, 'source':[
            file_hash(Path(__file__).with_name(name)) for name in ('neurovfm.py','geometry.py','base.py')]})
        self.cache = TensorCache(path_of(config, self.config.cache_dir) / self.version)
        self.hashes = {}

    def __call__(self, sample):
        from pathlib import Path
        stat = Path(sample.image).stat()
        identity = (sample.image, stat.st_size, stat.st_mtime_ns)
        if identity not in self.hashes:
            self.hashes[identity] = file_hash(sample.image)
        image_hash = self.hashes[identity]
        key = digest({'image':image_hash, 'cohort':sample.cohort, 'preprocessing':self.version})
        cached = self.cache.get(key) if self.config.cache_enabled else None
        if cached is not None:
            return PatchVolume(**cached).validate()
        image = sitk.ReadImage(sample.image)
        raw = sitk.GetArrayViewFromImage(image)
        if not np.isfinite(raw).all() or np.ptp(raw.astype(np.float64)) == 0:
            raise ValueError(f'Nonfinite or constant volume: {sample.id}')
        bounded, meta = transform_image(image, self.config, sample.cohort)
        values = sitk.GetArrayFromImage(bounded).transpose(meta['array_zyx_to_dhw']).astype(np.float64)
        lo, hi = np.percentile(values, list(self.config.intensity_percentiles))
        if hi <= lo:
            raise ValueError(f'Constant intensity after percentile clipping: {sample.id}')
        values = np.clip(values, lo, hi)
        values = (values-values.min()) / (values.max()-values.min()+1e-8)
        pads_zyx = list(zip(meta['pad_before_xyz'][::-1], meta['pad_after_xyz'][::-1]))
        pads = [pads_zyx[i] for i in meta['array_zyx_to_dhw']]
        values = np.pad(values, pads, constant_values=0)
        patches = rearrange(torch.from_numpy(values).float(),
            '(d p) (h q) (w r) -> (d h w) (p q r)', p=4, q=16, r=16)
        patches = (patches - self.config.mri_mean) / self.config.mri_std
        grid = [n//p for n,p in zip(values.shape,(4,16,16))]
        coords = torch.cartesian_prod(*(torch.arange(n) for n in grid))
        meta.update({'preprocessing_version':self.version, 'image_sha256':image_hash,
                     'cache_key':key, 'shape_dhw':list(values.shape), 'grid_dhw':grid,
                     'tokens':len(patches), 'percentile_values':[float(lo),float(hi)],
                     'normalization_applied_once':True, 'token_filter_policy':'none'})
        result = PatchVolume(patches.contiguous(), coords, meta).validate()
        if self.config.cache_enabled:
            self.cache.put(key, {'patches':result.patches, 'coords':result.coords, 'metadata':meta})
        return result
