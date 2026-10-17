import json
from contextlib import nullcontext
import torch
from neurovfm.models.vit import get_vit_backbone
from src.config import path_of, file_hash, digest
from src.models.encoders.base import EncoderAdapter, EncoderOutput
from src.models.encoders.lora import attach_visual_lora
from src.preprocessing.cache import TensorCache
from src.training.schedule import PHASES, PROJECTOR_ONLY_PHASES


class NeuroVFMEncoder(EncoderAdapter):
    def __init__(self, config):
        super().__init__()
        self.config = config
        asset = path_of(config,config.encoder.path)
        model_config = json.loads((asset/'config.json').read_text())
        self.core = get_vit_backbone(**model_config).float()
        self.core.load_state_dict(torch.load(asset/'pytorch_model.bin',map_location='cpu',weights_only=True),strict=True)
        self.feature_dim = model_config['params']['embed_dim']
        self.base_parameter_count = sum(p.numel() for p in self.core.parameters())
        self.fingerprint = digest({'config':file_hash(asset/'config.json'),
                                   'weights':file_hash(asset/'pytorch_model.bin'),
                                   'attention':config.encoder.attention,'precision':'bf16_autocast',
                                   'batch_mode':'independent'})
        self.stage = config.training.stage
        self.phase = 'projector_warmup' if self.stage == 'joint_lora' else self.stage
        if config.training.deterministic:
            from src.models.encoders.deterministic import enable_deterministic_attention
            enable_deterministic_attention(self.core)
        self.lora_targets = []
        self.core.requires_grad_(self.stage == 'b_full')
        if self.stage in ('b_lora', 'joint_lora'):
            self.core,self.lora_targets = attach_visual_lora(self.core,config.training.lora)
        if self.stage == 'joint_lora':
            self.core.requires_grad_(False)
        if config.encoder.attention != 'flash':
            raise ValueError('Validated packed encoder backend is flash; no silent quadratic fallback')
        self.cache = TensorCache(path_of(config,config.training.feature_cache_dir))
        self.train(True)

    def train(self, mode=True):
        super().train(mode)
        if self.vision_frozen:
            self.core.eval()
        return self

    @property
    def vision_frozen(self):
        return self.stage == 'a' or (self.stage == 'joint_lora' and self.phase in PROJECTOR_ONLY_PHASES)

    def set_phase(self, phase):
        if self.stage != 'joint_lora' or phase not in PHASES:
            raise ValueError('Invalid continuous LoRA phase')
        self.phase = phase
        for name, parameter in self.core.named_parameters():
            parameter.requires_grad_('lora_' in name and phase not in PROJECTOR_ONLY_PHASES)
        self.train(self.training)

    def _online(self, volumes):
        # One volume per kernel call gives stable cache/online and batch/single
        # semantics. Decoder batching and gradient accumulation remain available.
        if len(volumes) > 1:
            return [self._online([v])[0] for v in volumes]
        device = next(self.core.parameters()).device
        sizes = [len(v.patches) for v in volumes]
        if not sizes or min(sizes) < 1:
            raise ValueError('Encoder needs nonempty volumes')
        patches = torch.cat([v.patches for v in volumes]).to(device)
        coords = torch.cat([v.coords for v in volumes]).to(device)
        cu = torch.tensor([0]+sizes,device=device,dtype=torch.int32).cumsum(0,dtype=torch.int32)
        if int(cu[-1]) != len(patches):
            raise ValueError('Packed sequence lengths do not match patches')
        context = torch.no_grad() if self.vision_frozen else nullcontext()
        with context, torch.autocast('cuda',dtype=torch.bfloat16):
            features = self.core(patches,coords,masks=None,masks_enc=None,
                cu_seqlens=cu,max_seqlen=max(sizes),use_flash_attn=True)
        if features.shape != (sum(sizes),self.feature_dim) or not torch.isfinite(features).all():
            raise ValueError('Invalid encoder output')
        return list(features.split(sizes))

    def forward(self, volumes):
        for volume in volumes:
            volume.validate()
        if not self.config.training.feature_cache:
            features = self._online(volumes)
        else:
            if self.stage != 'a':
                raise ValueError('Feature caching would break visual training gradients')
            features = []
            device = next(self.core.parameters()).device
            for volume in volumes:
                key = digest({'encoder':self.fingerprint,'preprocessed':volume.metadata['cache_key']})
                cached = self.cache.get(key)
                if cached is None:
                    feature = self._online([volume])[0].detach().cpu()
                    cached = {'features':feature,'coords':volume.coords,'encoder':self.fingerprint}
                    self.cache.put(key,cached)
                if not torch.equal(cached['coords'],volume.coords):
                    raise ValueError('Feature-cache coordinate mismatch')
                feature = cached['features'].to(device)
                if feature.shape != (len(volume.patches),self.feature_dim) or not torch.isfinite(feature).all():
                    raise ValueError('Invalid cached feature tensor')
                features.append(feature)
        return EncoderOutput(features,[v.coords for v in volumes],[v.metadata for v in volumes])
