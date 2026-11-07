"""Strict YAML composition. Paths are relative to project_root, not the shell cwd."""
import hashlib
import json
import math
from pathlib import Path

from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def file_hash(path):
    hasher = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def path_of(config, value):
    path = Path(value)
    return path if path.is_absolute() else Path(config.project_root) / path


def _keys(override, schema, prefix=""):
    for key, value in override.items():
        if key not in schema:
            raise ValueError(f"Unknown config key: {prefix}{key}")
        if isinstance(value, dict) and key not in ("params", "resolved_budget"):
            if not isinstance(schema[key], dict):
                raise ValueError(f"Expected scalar: {prefix}{key}")
            _keys(value, schema[key], f"{prefix}{key}.")


def load_config(files=(), overrides=()):
    config = OmegaConf.load(ROOT / "configs/base.yaml")
    schema = OmegaConf.to_container(config, resolve=True)
    for source in [*(OmegaConf.load(p) for p in files), OmegaConf.from_dotlist(overrides)]:
        _keys(OmegaConf.to_container(source, resolve=False), schema)
        config = OmegaConf.merge(config, source)
    if config.project_root is None:
        config.project_root = str(ROOT)
    # Named grid policies control their physical spacing. Boundary policy remains
    # independent so native/2mm/4mm comparisons can share minimal padding.
    if config.preprocessing.spacing_policy == '2mm':
        config.preprocessing.throughplane_mm = 2.0
        config.preprocessing.inplane_mm = 1.0
    elif config.preprocessing.spacing_policy == '4mm':
        config.preprocessing.throughplane_mm = 4.0
        config.preprocessing.inplane_mm = 1.0
    OmegaConf.resolve(config)
    from src.data.datasets import normalize_datasets
    config.data.datasets = normalize_datasets(config.data.get('datasets'), default=['brats'])
    if config.evaluation.get('datasets') is not None:
        config.evaluation.datasets = normalize_datasets(config.evaluation.datasets)
    from src.models.projectors.specs import resolve_projector_config
    resolve_projector_config(config.projector)
    validate(config)
    return config


def validate(c):
    if c.projector.name == 'bindformer3d':
        p = c.projector.params
        if p.feature_loss_weight > 0 and p.feature_target == 'frozen_input' and c.training.stage != 'a':
            raise ValueError('frozen_input reconstruction requires Stage A; LoRA requires frozen_base_cache')
    from src.data.datasets import normalize_datasets
    if normalize_datasets(c.data.get('datasets'), default=['brats']) != list(c.data.get('datasets', ['brats'])):
        raise ValueError('data.datasets must use canonical dataset names')
    if c.evaluation.get('datasets') is not None:
        if normalize_datasets(c.evaluation.datasets) != list(c.evaluation.datasets):
            raise ValueError('evaluation.datasets must use canonical dataset names')
    # Evaluation also validates saved configs from before this optional guard.
    required=getattr(c.training,'required_visual_tokens',None)
    if required is not None:
        if type(required) is not int or required<1:
            raise ValueError('training.required_visual_tokens must be null or a positive integer')
        budget=c.projector.get('resolved_budget')
        requested=budget['expected_output_tokens'] if budget else c.projector.params.get('num_tokens')
        if requested is not None and requested!=required:
            raise ValueError('Requested projector num_tokens differs from required_visual_tokens')
    from src.models.projectors.budget import ratio_value, validate_resolved
    if ratio_value(c.projector) is not None and c.projector.get('resolved_budget'):
        validate_resolved(c.projector)
    if not isinstance(c.training.deterministic,bool):raise ValueError('training.deterministic must be boolean')
    if c.training.stage not in ("a", "b_lora", "b_full", "joint_lora"):
        raise ValueError("stage must be a, b_lora, b_full, or joint_lora")
    if c.training.stage != "a" and c.training.feature_cache:
        raise ValueError("Stage B cannot use final encoder feature cache")
    if not c.decoder.frozen or c.task.truncate_inputs:
        raise ValueError("This task requires a frozen decoder and untruncated inputs")
    if c.decoder.quantization != "none" or c.decoder.weight_dtype != "bfloat16":
        raise ValueError("Baseline supports BF16 unquantized decoder only")
    if c.training.distributed not in ("none", "ddp"):
        raise ValueError("Only single device and DDP have been implemented")
    projector_device = c.training.get('projector_device')
    if projector_device is not None:
        import re
        if projector_device == 'cpu' and c.task.name != 'neuroqa':
            raise ValueError('CPU projector_device is validated only for NeuroQA Concat3D')
        if projector_device != 'cpu' and (not isinstance(projector_device, str)
                or re.fullmatch(r'cuda:\d+', projector_device) is None
                or not isinstance(c.training.device, str)
                or re.fullmatch(r'cuda:\d+', c.training.device) is None
                or int(projector_device.split(':')[1]) == int(c.training.device.split(':')[1])):
            raise ValueError('training.projector_device must be a different logical cuda:N device')
        if c.projector.name != 'concat3d' or c.training.distributed != 'none':
            raise ValueError('Two-GPU projector placement supports Concat3D without DDP only')
    if c.training.mixed_precision != "bf16":
        raise ValueError("Baseline precision must be bf16")
    if c.preprocessing.spacing_policy not in ("2mm", "4mm", "official_4mm", "resample", "brats2mm_other_native_z", "native_grid"):
        raise ValueError("Unknown spacing policy")
    if c.preprocessing.boundary_policy not in ("official_crop", "minimal_pad"):
        raise ValueError("Unknown boundary policy")
    if c.preprocessing.spacing_policy == 'native_grid' and c.preprocessing.boundary_policy != 'minimal_pad':
        raise ValueError('Native-grid preservation requires minimal_pad')
    if c.encoder.batch_mode != 'independent':
        raise ValueError('Packed numerical consistency is unvalidated; use independent volume encoding')
    if c.preprocessing.token_filter_policy != "none":
        raise ValueError("Custom token selection requires an explicit registered implementation")
    if c.data.review_policy != "include_with_flag" or c.data.sampling != ('balanced_neuroqa' if c.task.name == 'neuroqa' else 'uniform_sequence'):
        raise ValueError("Alternate sampling/review policy requires a versioned manifest")
    if c.preprocessing.slice_axis_policy not in ("official", "published_z"):
        raise ValueError("Unknown slice-axis policy")
    if c.preprocessing.image_interpolation not in ("bspline", "linear"):
        raise ValueError("Unsupported image interpolation")
    if list(c.preprocessing.patch_size_dhw) != [4, 16, 16]:
        raise ValueError("Local NeuroVFM weights require patches [4,16,16]")
    lo, hi = c.preprocessing.intensity_percentiles
    if not 0 <= lo < hi <= 100 or c.preprocessing.mri_std <= 0:
        raise ValueError("Invalid intensity normalization")
    if c.preprocessing.throughplane_mm <= 0 or c.preprocessing.inplane_mm <= 0:
        raise ValueError("Spacing must be positive")
    if c.preprocessing.spacing_policy == "official_4mm" and (c.preprocessing.throughplane_mm != 4 or c.preprocessing.inplane_mm != 1):
        raise ValueError("Use spacing_policy=resample for non-official spacing")
    for value in (c.training.microbatch_per_device, c.training.effective_batch_size,
                  c.training.max_epochs, c.training.early_stopping_patience,
                  c.training.lora.rank, c.decoder.ce_chunk_size):
        if not isinstance(value, int) or value < 1:
            raise ValueError("Batch sizes, epochs, patience, rank and CE chunk must be positive integers")
    if c.training.projector_lr <= 0 or (c.training.stage != "a" and c.training.encoder_lr <= 0):
        raise ValueError("Trainable parameter groups require positive learning rates")
    if not 0 <= c.training.warmup_ratio < 1 or not 0 <= c.training.lora.dropout < 1:
        raise ValueError("Invalid warmup or dropout")
    if c.training.lora.bias != "none":
        raise ValueError("Baseline LoRA bias must be none")
    if c.training.init_checkpoint and c.training.resume:
        raise ValueError("Stage initialization and resume are different operations")
    if type(c.training.lora.alpha) not in (int,float) or not math.isfinite(c.training.lora.alpha) or c.training.lora.alpha <= 0:
        raise ValueError('LoRA alpha must be finite and positive')
    delta=c.training.get('early_stopping_min_delta',0.0)
    if type(delta) not in (int,float) or not math.isfinite(delta) or delta<0:
        raise ValueError('early_stopping_min_delta must be finite and nonnegative')
    for key, default in (('early_stopping_enabled', True), ('log_optimizer_diagnostics', False)):
        if type(c.training.get(key, default)) is not bool:
            raise ValueError(f'{key} must be boolean')
    epochs=c.training.get('checkpoint_epochs', [])
    if not OmegaConf.is_list(epochs) and not isinstance(epochs, list):
        raise ValueError('checkpoint_epochs must be a list')
    if (any(type(epoch) is not int or not 1<=epoch<=c.training.max_epochs for epoch in epochs)
            or len(epochs)!=len(set(epochs)) or list(epochs)!=sorted(epochs)):
        raise ValueError('checkpoint_epochs must be unique ascending epochs within max_epochs')
    if c.training.stage=='joint_lora':
        if c.training.init_checkpoint:
            raise ValueError('joint_lora starts from pretrained assets or same-run resume, not Stage A initialization')
        if c.training.warmup_steps is not None or c.training.warmup_ratio!=0:
            raise ValueError('joint_lora uses two warm-ups; set legacy warmup_steps=null and warmup_ratio=0')
        floor=c.training.get('cosine_min_lr_ratio',0.0)
        if type(floor) not in (int,float) or not math.isfinite(floor) or not 0<=floor<=1:
            raise ValueError('cosine_min_lr_ratio must be finite and between 0 and 1 (inclusive)')
        mode=c.training.get('joint_warmup_projector_mode','restart')
        if mode not in ('restart','hold_peak'):
            raise ValueError('joint_warmup_projector_mode must be restart or hold_peak')
        hold=c.training.get('projector_hold_steps',0)
        if type(hold) is not int or hold<0:
            raise ValueError('projector_hold_steps must be a nonnegative integer')
        for prefix in ('projector','joint'):
            steps=c.training.get(f'{prefix}_warmup_steps')
            ratio=c.training.get(f'{prefix}_warmup_ratio')
            if steps is not None and (type(steps) is not int or steps<2):
                raise ValueError(f'{prefix}_warmup_steps must be null or an integer >= 2')
            if type(ratio) not in (int,float) or not math.isfinite(ratio) or not 0<ratio<1:
                raise ValueError(f'{prefix}_warmup_ratio must be finite and between 0 and 1')
    if c.generation.do_sample or c.generation.num_beams != 1 or c.generation.max_new_tokens < 1:
        raise ValueError("Baseline generation must be greedy with a positive token budget")
    if c.generation.batch_mode != 'independent':
        raise ValueError('Generation supports independent decoding; packed decoding requires a validated adapter')
    if c.training.num_workers != 0:
        raise ValueError('Current on-demand volume loader requires num_workers=0')
    for key in ('save_every_steps','validation_generation_samples'):
        if not isinstance(c.training[key],int) or c.training[key]<0:
            raise ValueError(f'{key} must be a nonnegative integer')
    if c.training.warmup_steps is not None and (not isinstance(c.training.warmup_steps,int) or c.training.warmup_steps<1):
        raise ValueError('training.warmup_steps must be a positive integer when set')
    for value in (c.training.max_steps,c.training.train_limit,c.training.val_limit):
        if value is not None and (not isinstance(value,int) or value<1):
            raise ValueError('Optional step/sample limits must be positive integers')
    if c.evaluation.batch_size<1 or c.evaluation.split not in ('train','val','test'):
        raise ValueError('Invalid evaluation split or batch size')
    if c.task.name == 'neuroqa':
        if c.data.get('invalid_sample_policy', 'strict') not in ('skip', 'strict', 'include_unreviewed'):
            raise ValueError('data.invalid_sample_policy must be skip, strict or include_unreviewed')
        if c.data.get('modality_sampling', 'legacy_exact') not in ('dual_eligible', 'legacy_exact', 'exact_balanced_repeat'):
            raise ValueError('Unknown NeuroQA modality sampling policy')
        from src.models.projectors.budget import NEUROQA_VERSION, NEUROQA_GRIDS
        if (c.preprocessing.name not in ('neuroqa_mni1mm', 'neuroqa_mni1mm_v2') or c.preprocessing.spacing_policy != 'native_grid'
                or c.preprocessing.throughplane_mm != 1 or c.preprocessing.inplane_mm != 1
                or c.preprocessing.orientation != 'RPI' or c.preprocessing.boundary_policy != 'minimal_pad'):
            raise ValueError('NeuroQA v1 requires the fixed MNI 1 mm adapter with RPI/minimal padding')
        if c.projector.get('budget_protocol') != NEUROQA_VERSION:
            raise ValueError('NeuroQA requires its versioned shared-M budget')
        allowed_ratios = (1,) if c.projector.name == 'mlp2x_gelu' else tuple(NEUROQA_GRIDS)
        if ratio_value(c.projector) not in allowed_ratios:
            raise ValueError('NeuroQA requires r8/16/32/64/128/256, or r1 for uncompressed MLP')
        if c.training.distributed != 'none':
            raise ValueError('NeuroQA supports single-process training')
        if c.data.get('modality_sampling', 'legacy_exact') in ('legacy_exact', 'exact_balanced_repeat') and c.training.effective_batch_size % 2:
            raise ValueError('Exact modality balance requires an even effective batch')
        if c.training.init_checkpoint:
            raise ValueError('Main NeuroQA benchmark starts from base assets; report transfer is a separate protocol')
        if c.projector.name == 'bindformer3d' and c.projector.params.knowledge_mode != 'none':
            raise ValueError('NeuroQA v1 disables report-derived concept banks; rebuild using only QA-train patients before enabling knowledge')
        if set(c.preprocessing.params) - {'registration_seed', 'registration_threads'}:
            raise ValueError('Unknown NeuroQA preprocessing parameter')
        if c.training.stage not in ('a','b_lora','joint_lora') or list(c.data.datasets) != ['brats']:
            raise ValueError('NeuroQA v1 supports BraTS with projector-only or visual-LoRA training')


def accumulation_steps(config, world_size=1):
    divisor = config.training.microbatch_per_device * world_size
    batch = config.training.effective_batch_size
    if batch % divisor:
        raise ValueError("Effective batch must be divisible by microbatch × world_size")
    return batch // divisor


def save_config(config, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(config, path)
