"""Experimental NeuroQA registration: isolated from the frozen v1 implementation.

This module measures candidate transforms. It does not certify images or modify
the v1 preprocessing store. Configuration selection must use training data only.
"""
import json
from pathlib import Path
import time

import numpy as np
import SimpleITK as sitk

from src.config import file_hash, digest
from src.data.neuroqa_manifest import write_json
from src.preprocessing.neuroqa_smri import (TEMPLATE, canvas_coverage,
    compose_pull, qc_figure, warp)


CANDIDATES = {
    'B': {'estimate': 'Once', 'maximum_step_mm': 0.0},
    'C': {'estimate': 'EachIteration', 'maximum_step_mm': 0.25},
    'D': {'estimate': 'EachIteration', 'maximum_step_mm': 0.125},
}


def prepare_native(image):
    """Keep the native foreground mask; repair only sparse negative intensities."""
    a = sitk.GetArrayFromImage(image).astype(np.float32)
    if not np.isfinite(a).all():
        raise ValueError('Source image contains NaN or Inf')
    mask = a != 0
    positive = a[a > 0]
    negatives = a < 0
    n = int(negatives.sum())
    if positive.size < 1000 or n / max(1, int(mask.sum())) > 0.0001:
        raise ValueError('Source has invalid brain signal or too many negative voxels')
    floor = float(max(1e-6, np.percentile(positive, .5) * .001))
    if n:
        a[negatives] = floor
    cleaned = sitk.GetImageFromArray(a)
    cleaned.CopyInformation(image)
    brain = sitk.GetImageFromArray(mask.astype(np.uint8))
    brain.CopyInformation(image)
    return cleaned, brain, {'negative_voxels': n, 'foreground_voxels': int(mask.sum()),
        'source_min': float(sitk.GetArrayViewFromImage(image).min()),
        'replacement_floor': floor if n else None,
        'policy': 'native_nonzero_mask_then_sparse_negatives_to_positive_floor_v1',
        'max_negative_fraction': 0.0001}


def register(fixed, moving, kind, seed, settings):
    reg = sitk.ImageRegistrationMethod()
    reg.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    reg.SetMetricSamplingStrategy(reg.RANDOM)
    reg.SetMetricSamplingPercentage(.25, seed)
    reg.SetInterpolator(sitk.sitkLinear)
    reg.SetOptimizerAsGradientDescent(learningRate=.1 if kind == 'rigid' else .05,
        numberOfIterations=1000, convergenceMinimumValue=1e-6, convergenceWindowSize=10,
        estimateLearningRate=getattr(reg, settings['estimate']),
        maximumStepSizeInPhysicalUnits=settings['maximum_step_mm'])
    reg.SetOptimizerScalesFromPhysicalShift()
    if kind == 'rigid':
        initial = sitk.CenteredTransformInitializer(fixed, moving, sitk.Euler3DTransform(),
                                                    sitk.CenteredTransformInitializerFilter.GEOMETRY)
        reg.SetShrinkFactorsPerLevel([4, 2, 1])
        reg.SetSmoothingSigmasPerLevel([2, 1, 0])
    else:
        initial = sitk.AffineTransform(3)
        reg.SetShrinkFactorsPerLevel([2, 1])
        reg.SetSmoothingSigmasPerLevel([1, 0])
    reg.SetInitialTransform(initial)
    levels = {}
    def observe():
        level = str(reg.GetCurrentLevel())
        levels[level] = {'iteration': int(reg.GetOptimizerIteration()),
                         'learning_rate': float(reg.GetOptimizerLearningRate()),
                         'metric': float(reg.GetMetricValue())}
    reg.AddCommand(sitk.sitkIterationEvent, observe)
    result = reg.Execute(fixed, moving)
    return result, {'seed': seed, 'metric': float(reg.GetMetricValue()),
        'stop': reg.GetOptimizerStopConditionDescription(), 'levels_last_iteration': levels}


def matrix_info(transform):
    offset = np.asarray(transform.TransformPoint((0., 0., 0.)))
    matrix = np.column_stack([np.asarray(transform.TransformPoint(tuple(v))) - offset for v in np.eye(3)])
    return {'determinant': float(np.linalg.det(matrix)),
            'singular_values': np.linalg.svd(matrix, compute_uv=False).tolist(),
            'offset': offset.tolist()}


def assess(mask, template, transform, expanded=True):
    mapped = sitk.GetArrayFromImage(warp(mask, template, transform, True)) > 0
    reference = sitk.GetArrayFromImage(template) > 0
    n, ref = int(mapped.sum()), int(reference.sum())
    result = {'dice': float(2*np.logical_and(mapped, reference).sum()/max(1, n+ref)),
              'foreground_voxels': n, 'transform': matrix_info(transform)}
    if expanded:
        result.update(canvas_coverage(mask, template, transform))
    return result


def run_case(source, root, template_path, candidate, seed=42):
    sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(1)
    settings = CANDIDATES[candidate]
    folder = Path(root) / candidate / source['case_id']
    folder.mkdir(parents=True, exist_ok=True)
    contract = {'candidate': candidate, 'settings': settings, 'seed': seed, 'threads': 1,
        'sources': {m: file_hash(p) for m, p in source['images'].items()},
        'template': file_hash(template_path), 'implementation': file_hash(__file__),
        'SimpleITK': sitk.Version_VersionString(), 'numpy': np.__version__}
    final = folder / 'result.json'
    if final.exists():
        previous = json.loads(final.read_text())
        if previous['contract'] != contract:
            raise ValueError('Existing pilot result uses a different contract')
        for filename, sha in previous['artifacts'].items():
            if file_hash(folder / filename) != sha:
                raise ValueError(f'Corrupt pilot artifact: {filename}')
        return {'case_id': source['case_id'], 'candidate': candidate, 'status': 'cached'}
    started = time.monotonic()
    template = sitk.ReadImage(str(template_path), sitk.sitkFloat32)
    native, masks, intensity = {}, {}, {}
    for m, path in source['images'].items():
        native[m], masks[m], intensity[m] = prepare_native(sitk.ReadImage(path, sitk.sitkFloat32))
    initial = sitk.CenteredTransformInitializer(template, native['T1w'], sitk.Euler3DTransform(),
                                                sitk.CenteredTransformInitializerFilter.GEOMETRY)
    rigid, rigid_meta = register(template, native['T1w'], 'rigid', seed, settings)
    rigid_image = warp(native['T1w'], template, rigid)
    affine, affine_meta = register(template, rigid_image, 'affine', seed+1, settings)
    t2rigid, t2_meta = register(native['T1w'], native['T2w'], 'rigid', seed+2, settings)
    final1 = compose_pull(rigid, affine)
    final2 = compose_pull(t2rigid, rigid, affine)
    transforms = {'initial': initial, 'rigid': rigid, 'affine': affine, 't2_to_t1': t2rigid,
                  'T1w_final': final1, 'T2w_final': final2}
    for name, transform in transforms.items():
        sitk.WriteTransform(transform, str(folder / f'{name}.tfm'))
    metrics = {'T1w': {'initial': assess(masks['T1w'], template, initial, False),
                       'rigid': assess(masks['T1w'], template, rigid),
                       'final': assess(masks['T1w'], template, final1)},
               'T2w': {'initial': assess(masks['T2w'], template, initial, False),
                       'rigid': assess(masks['T2w'], template, compose_pull(t2rigid, rigid)),
                       'final': assess(masks['T2w'], template, final2)}}
    # Overlays use original cleaned signal, avoiding an expensive N4 during pilot.
    images = {'T1w': warp(native['T1w'], template, final1), 'T2w': warp(native['T2w'], template, final2)}
    warped_masks = {'T1w': warp(masks['T1w'], template, final1, True),
                    'T2w': warp(masks['T2w'], template, final2, True)}
    qc_figure(template, images, warped_masks, folder / 'qc.png')
    artifacts = {p.name: file_hash(p) for p in folder.iterdir() if p.suffix in ('.tfm', '.png')}
    result = {'case_id': source['case_id'], 'contract': contract, 'fingerprint': digest(contract),
              'intensity': intensity, 'registration': {'rigid': rigid_meta, 'affine': affine_meta,
              'T2_to_T1': t2_meta}, 'metrics': metrics, 'artifacts': artifacts,
              'seconds': time.monotonic()-started, 'status': 'measured_not_certified'}
    write_json(final, result)
    return {'case_id': source['case_id'], 'candidate': candidate, 'status': 'measured',
            'seconds': result['seconds']}
