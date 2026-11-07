"""Resolve target N/M before model construction; preserve each method's mechanism.

No image intensities, labels, model weights or CUDA are needed for the preflight.
The v1 protocol requires one input geometry per run. It never resizes queries or
Concat heads during forward, and never changes a saved budget at evaluation.
"""
import math
from collections import Counter
from omegaconf import OmegaConf

VERSION = 'method_native_ratio_v1'
NEUROQA_VERSION = 'neuroqa_mni1mm_shared_m_v1'
# r8 keeps Concat's native 9x1x1 grouping and shares its 1170 output
# tokens with all methods: 9555/1170 = 8.1667 (2.08% from the target).
NEUROQA_GROUPS = {8: (9, 1, 1), 16: (9, 2, 1), 32: (6, 2, 4), 64: (9, 3, 3),
                  128: (17, 3, 3), 256: (18, 4, 5)}
NEUROQA_GRIDS = {8: (6, 15, 13), 16: (6, 8, 13), 32: (9, 8, 4), 64: (6, 5, 5),
                 128: (3, 5, 5), 256: (3, 4, 3)}
DEFAULT_RATIOS = (4, 8, 16, 32)
SUPPORTED_RATIOS = (*DEFAULT_RATIOS, 64, 128, 256)
GRID_METHODS = frozenset(('avgpool3d', 'convpool3d', 'mixer3d', 'local_packer3d'))
EXACT_METHODS = frozenset(('bindformer3d', 'bindformer_a4', 'bindformer_adirect',
                           'bindformer_adirect_localread', 'bindformer_adirect_detail',
                           'attention_pool', 'perceiver_nv', 'perceiver_flamingo',
                           'adaptive_merge3d', 'token_select3d'))
CONCAT_GROUPS = {4: (4, 1, 1), 8: (8, 1, 1), 16: (6, 1, 3), 32: (6, 2, 3),
                 64: (6, 3, 4), 128: (13, 2, 5), 256: (13, 4, 5)}
BRATS_FIXED_GRIDS = {64: (4, 6, 6), 128: (3, 4, 6), 256: (3, 4, 3)}
TOLERANCE = 0.05


def brats_native_geometry(grid, extent):
    return tuple(grid) == (39, 15, 15) and all(
        math.isclose(a, b, rel_tol=1e-6, abs_tol=1e-6)
        for a, b in zip(extent, (156, 240, 240)))


def ratio_value(projector):
    value = projector.get('target_compression_ratio')
    if value is not None and (type(value) not in (int, float)
                              or not math.isfinite(value) or value < 1):
        raise ValueError('target_compression_ratio must be finite and >= 1, or null')
    return value


def derive_budget(name, ratio, grid, extent, params=None, protocol=None):
    from src.models.projectors.geometry import choose_grid
    grid, extent = tuple(grid), tuple(extent)
    if len(grid) != 3 or any(type(n) is not int or n < 1 for n in grid):
        raise ValueError('Budget input_grid_dhw must contain three positive integers')
    if len(extent) != 3 or any(not math.isfinite(e) or e <= 0 for e in extent):
        raise ValueError('Budget physical extent must contain three positive numbers')
    ratio_value({'target_compression_ratio': ratio})
    if ratio is None:
        raise ValueError('A target compression ratio is required')
    n = math.prod(grid)
    raw = math.ceil(n / ratio)
    output_grid, group = None, None
    if protocol == NEUROQA_VERSION and name != 'mlp2x_gelu':
        if grid != (49, 15, 13) or extent != (196, 240, 208) or ratio not in NEUROQA_GRIDS:
            raise ValueError('NeuroQA budgets require grid [49,15,13], extent [196,240,208] and r8/16/32/64/128/256')
        if name not in GRID_METHODS | EXACT_METHODS | {'concat3d'}:
            raise ValueError(f'Unknown NeuroQA budget method: {name}')
        actual = math.prod(NEUROQA_GRIDS[ratio])
        output_grid = list(NEUROQA_GRIDS[ratio]) if name in GRID_METHODS | {'concat3d'} else None
        group = list(NEUROQA_GROUPS[ratio]) if name == 'concat3d' else None
        kind = 'neuroqa_shared_m'
    elif protocol not in (None, VERSION, NEUROQA_VERSION):
        raise ValueError(f'Unknown budget protocol: {protocol}')
    elif name == 'mlp2x_gelu':
        if ratio != 1:
            raise ValueError('Uncompressed MLP supports only ratio 1 (or manual mode)')
        actual, kind = n, 'uncompressed'
    elif name in GRID_METHODS:
        if ratio in BRATS_FIXED_GRIDS:
            if not brats_native_geometry(grid, extent):
                raise ValueError(f'{ratio}x grid preset requires BraTS grid [39,15,15] and extent [156,240,240] mm; review a new preset for other geometry')
            # These presets prioritize balanced physical cells within 5% of the
            # requested ratio. Keep raw and actual output counts distinct.
            output_grid, kind = list(BRATS_FIXED_GRIDS[ratio]), f'fixed_grid_brats_r{ratio}_v1'
        else:
            output_grid = list(choose_grid(grid, raw, extent, 'nearest'))
            kind = 'grid_nearest'
        actual = math.prod(output_grid)
    elif name in EXACT_METHODS:
        actual, kind = raw, 'exact_count'
    elif name == 'concat3d':
        if not brats_native_geometry(grid, extent):
            raise ValueError('Concat ratio preset requires BraTS grid [39,15,15] and extent [156,240,240] mm; review a new preset for other geometry')
        if ratio not in CONCAT_GROUPS:
            raise ValueError(f'Concat ratio preset supports only {", ".join(map(str, CONCAT_GROUPS))}')
        group = list(CONCAT_GROUPS[ratio])
        output_grid = [math.ceil(g/k) for g, k in zip(grid, group)]
        actual, kind = math.prod(output_grid), 'fixed_concat_brats_native_v1'
    else:
        raise ValueError(f'No ratio budget policy registered for {name}')
    error = abs(n / actual / ratio - 1)
    if error > TOLERANCE + 1e-12:
        raise ValueError(f'Compression ratio deviation exceeds 5%: target={ratio}, N={n}, M={actual}, error={error:.4%}')
    allocation = {}
    if name == 'bindformer3d':
        from src.models.projectors.bindformer.budget import allocate_tokens, role_counts
        allocation = allocate_tokens(actual, (params or {}).get('global_fraction', 0.25),
                                     role_query_counts=role_counts((params or {}).get('role_query_counts')))
    elif name == 'bindformer_a4':
        from src.models.projectors.bindformer_a4.config import allocate_a4_tokens
        allocation = allocate_a4_tokens(actual, (params or {}).get('global_fraction', 0.25))
    elif name in ('bindformer_adirect', 'bindformer_adirect_localread'):
        from src.models.projectors.bindformer_adirect.config import allocate_adirect_tokens
        allocation = allocate_adirect_tokens(actual, (params or {}).get('global_fraction', 0.25))
    elif name == 'bindformer_adirect_detail':
        from src.models.projectors.bindformer_adirect_detail.config import allocate_detail_tokens
        allocation = allocate_detail_tokens(actual, (params or {}).get('detail_fraction', 0.25))
    return {'version': protocol or VERSION, 'method': name, 'kind': kind, 'target_ratio': ratio,
            'input_grid_dhw': list(grid), 'input_extent_mm_dhw': list(extent),
            'input_tokens': n, 'raw_num_tokens': raw, 'expected_output_tokens': actual,
            'output_grid_dhw': output_grid, 'group_size': group,
            'actual_ratio': n/actual, 'relative_error': error, 'max_relative_error': TOLERANCE,
            **allocation}


def validate_resolved(projector):
    """Recompute the saved contract; overrides cannot silently invalidate it."""
    ratio = ratio_value(projector)
    record = OmegaConf.to_container(projector.resolved_budget, resolve=True)
    try:
        expected = derive_budget(projector.name, ratio, record['input_grid_dhw'],
                                 record['input_extent_mm_dhw'], projector.params, projector.get('budget_protocol'))
    except (KeyError, TypeError) as exc:
        raise ValueError('Malformed resolved projector budget') from exc
    if record != expected:
        raise ValueError('Resolved projector budget differs from target ratio/method/geometry; start a new unresolved configuration')
    params = projector.params
    if projector.name == 'concat3d':
        matches = list(params.get('group_size', [])) == record['group_size']
    elif projector.name == 'mlp2x_gelu':
        matches = True
    else:
        matches = params.get('num_tokens') == record['expected_output_tokens' if record['version'] == NEUROQA_VERSION else 'raw_num_tokens']
    if projector.name in GRID_METHODS:
        matches = matches and params.get('budget_policy') == 'nearest'
    if not matches:
        raise ValueError('Projector parameters conflict with resolved compression budget')
    return record


def resolve_budget(projector, grid, extent):
    from src.models.projectors.specs import resolve_projector_config
    resolve_projector_config(projector)
    ratio = ratio_value(projector)
    if ratio is None:
        return None
    record = derive_budget(projector.name, ratio, grid, extent, projector.params, projector.get('budget_protocol'))
    if projector.get('resolved_budget'):
        saved = validate_resolved(projector)
        if saved['input_grid_dhw'] != record['input_grid_dhw'] or any(
                not math.isclose(a, b, rel_tol=1e-6, abs_tol=1e-6)
                for a, b in zip(saved['input_extent_mm_dhw'], record['input_extent_mm_dhw'])):
            raise ValueError('Input geometry differs from saved compression budget')
        return saved
    if projector.name == 'concat3d':
        projector.params.group_size = record['group_size']
    elif projector.name != 'mlp2x_gelu':
        projector.params.num_tokens = record['expected_output_tokens' if record['version'] == NEUROQA_VERSION else 'raw_num_tokens']
    projector.resolved_budget = OmegaConf.create(record)
    resolve_projector_config(projector)
    return record


def require_resolved(projector):
    if ratio_value(projector) is not None:
        if not projector.get('resolved_budget'):
            raise ValueError('Resolve the projector compression budget from input geometry before constructing the model')
        return validate_resolved(projector)
    return None


def geometry_from_header(path, preprocessing):
    """Predict native-grid shape from headers; actual preprocessing is checked again at forward."""
    import SimpleITK as sitk
    from types import SimpleNamespace
    from src.preprocessing.geometry import slice_axis
    if preprocessing.spacing_policy != 'native_grid' or preprocessing.boundary_policy != 'minimal_pad':
        raise ValueError('Ratio preflight v1 supports native_grid + minimal_pad; other preprocessing needs a geometry adapter')
    reader = sitk.ImageFileReader()
    reader.SetFileName(str(path))
    reader.ReadImageInformation()
    if reader.GetDimension() != 3 or reader.GetNumberOfComponents() != 1:
        raise ValueError(f'Expected scalar 3D MRI: {path}')
    source = sitk.DICOMOrientImageFilter.GetOrientationFromDirectionCosines(reader.GetDirection())
    families = {c: i for i, pair in enumerate(('LR', 'AP', 'IS')) for c in pair}
    source_axes = [families[c] for c in source]
    desired_axes = [families[c] for c in preprocessing.orientation]
    order = [source_axes.index(i) for i in desired_axes]
    size = [reader.GetSize()[i] for i in order]
    spacing = [reader.GetSpacing()[i] for i in order]
    axis = slice_axis(SimpleNamespace(GetSize=lambda: size, GetSpacing=lambda: spacing),
                      preprocessing.slice_axis_policy)
    multiples = [16, 16, 16]
    multiples[axis] = 4
    bounded = [math.ceil(n/p)*p for n, p in zip(size, multiples)]
    array_order = {2: [0, 1, 2], 1: [1, 0, 2], 0: [2, 1, 0]}[axis]
    dhw = [bounded[::-1][i] for i in array_order]
    spacing_dhw = [spacing[::-1][i] for i in array_order]
    grid = [n//p for n, p in zip(dhw, preprocessing.patch_size_dhw)]
    extent = [n*s for n, s in zip(dhw, spacing_dhw)]
    return grid, extent


def prepare_budget(config, samples):
    """Preflight the complete selected population before allocating model weights."""
    if ratio_value(config.projector) is None:
        return None
    if config.encoder.name != 'neurovfm' or config.preprocessing.token_filter_policy != 'none':
        raise ValueError('Ratio preflight v1 requires dense NeuroVFM patch features')
    samples = list(samples)
    if not samples:
        raise ValueError('Cannot resolve compression budget from an empty population')
    signatures, headers = {}, {}
    for sample in samples:
        if sample.image not in headers:
            headers[sample.image] = geometry_from_header(sample.image, config.preprocessing)
        grid, extent = headers[sample.image]
        signature = (tuple(grid), tuple(round(v, 6) for v in extent))
        signatures.setdefault(signature, []).append(sample.id)
    if len(signatures) != 1:
        raise ValueError('Ratio preflight requires one input geometry per run; found ' +
                         str({str(k): len(v) for k, v in signatures.items()}))
    grid, extent = headers[samples[0].image]
    record = resolve_budget(config.projector, grid, extent)
    from src.config import validate
    validate(config)
    return {'status': 'passed_header_preflight', 'samples': len(samples),
            'unique_images': len(headers), 'cohorts': dict(Counter(s.cohort for s in samples)),
            'runtime_geometry_check': True, 'budget': record}


def check_input(budget, features, metadata):
    if not budget:
        return
    from src.models.projectors.geometry import physical_extent
    if list(metadata.get('grid_dhw', [])) != budget['input_grid_dhw'] or len(features) != budget['input_tokens']:
        raise ValueError('Projector input grid/token count differs from resolved compression budget')
    if not all(k in metadata for k in ('shape_dhw', 'bounded_before_pad', 'array_zyx_to_dhw')):
        raise ValueError('Ratio mode requires physical geometry metadata')
    actual = physical_extent(budget['input_grid_dhw'], metadata)
    if any(not math.isclose(a, b, rel_tol=1e-6, abs_tol=1e-6)
           for a, b in zip(actual, budget['input_extent_mm_dhw'])):
        raise ValueError('Projector input physical extent differs from resolved compression budget')


def output_info(budget, n, m):
    if not budget:
        return {}
    if n != budget['input_tokens'] or m != budget['expected_output_tokens']:
        raise ValueError(f'Resolved compression budget mismatch: N={n}, M={m}, expected M={budget["expected_output_tokens"]}')
    return {'budget_version': budget['version'], 'target_compression_ratio': budget['target_ratio'],
            'input_tokens': n, 'raw_num_tokens': budget['raw_num_tokens'],
            'expected_output_tokens': budget['expected_output_tokens'], 'actual_ratio': n/m,
            'retention_fraction': m/n, 'reduction_fraction': 1-m/n,
            'relative_ratio_error': abs(n/m/budget['target_ratio']-1)}
