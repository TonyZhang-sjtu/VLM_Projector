"""NeuroQA sMRI adaptation for already skull-stripped BraTS (CPU, offline).

Reference: mhabbasiit/sMRI reg.py and postprocess_mni_mask_zscore_crop.py.
Differences are explicit: reuse BraTS foreground, seeded registration, fix the
T2 mask transform chain, retain the full template canvas for the visual encoder.
No segmentation labels, reports, or QA answers enter image processing.
"""
import json
import os
from pathlib import Path
import time
import urllib.request

import numpy as np
import SimpleITK as sitk

from src.config import digest, file_hash
from src.data.neuroqa_manifest import write_json
from src.preprocessing.geometry import geometry

PROTOCOL = 'neuroqa_brats_foreground_mni1mm_v1'
TEMPLATE = 'tpl-MNI152NLin2009cAsym_res-01_desc-brain_T1w.nii.gz'
TEMPLATE_URL = 'https://templateflow.s3.amazonaws.com/tpl-MNI152NLin2009cAsym/' + TEMPLATE
# This earlier local implementation has IDENTICAL image/transform computations;
# only its stricter any-boundary-voxel QC gate differs. Existing completed cases
# have already passed that gate. Keep their original fingerprints/provenance.
COMPATIBLE_IMPLEMENTATIONS = {'920a107c7acaeef94eefbdbbff488d141207c7e5a51defd86a4557b851038fb6'}
IMPLEMENTATION_SHA256 = file_hash(__file__)  # Freeze the code actually loaded by this worker.


def compatible_contract(saved, requested):
    normalized = dict(saved)
    if normalized.get('implementation_sha256') in COMPATIBLE_IMPLEMENTATIONS:
        normalized['implementation_sha256'] = requested['implementation_sha256']
    return normalized == requested


def canvas_coverage(native_mask, template, transform):
    """Count actual foreground outside the fixed canvas, not merely edge contact.

    The expanded reference is QC-only and never becomes a model input. 0.1% is
    an engineering review threshold, not a guarantee of clinical acceptability.
    """
    margin = 32
    expanded = sitk.ConstantPad(template, [margin]*3, [margin]*3, 0)
    values = sitk.GetArrayFromImage(warp(native_mask, expanded, transform, mask=True)) > 0
    total = int(values.sum())
    outside = total - int(values[margin:-margin, margin:-margin, margin:-margin].sum())
    touches_expanded = any(np.take(values, i, axis=axis).any() for axis in range(3) for i in (0, -1))
    return {'expanded_reference_margin_mm': margin, 'outside_canvas_voxels': outside,
            'outside_canvas_fraction': outside/max(1,total), 'expanded_boundary_touched': bool(touches_expanded),
            'review_threshold_fraction': 0.001}


def download_template(root):
    path = Path(root) / 'templates' / TEMPLATE
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        tmp = path.with_suffix('.download')
        urllib.request.urlretrieve(TEMPLATE_URL, tmp)
        tmp.replace(path)
    image = sitk.ReadImage(str(path), sitk.sitkFloat32)
    if image.GetSize() != (193, 229, 193) or image.GetSpacing() != (1., 1., 1.):
        raise ValueError('Unexpected MNI template geometry')
    write_json(path.parent / 'template.json', {'url': TEMPLATE_URL, 'sha256': file_hash(path),
                                              'geometry': geometry(image)})
    return path


def registration(fixed, moving, kind, seed):
    reg = sitk.ImageRegistrationMethod()
    reg.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    reg.SetMetricSamplingStrategy(reg.RANDOM)
    reg.SetMetricSamplingPercentage(0.25, seed)
    reg.SetInterpolator(sitk.sitkLinear)
    reg.SetOptimizerAsGradientDescent(learningRate=0.1 if kind == 'rigid' else 0.05,
                                     numberOfIterations=1000, convergenceMinimumValue=1e-6,
                                     convergenceWindowSize=10)
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
    transform = reg.Execute(fixed, moving)
    return transform, {'metric': float(reg.GetMetricValue()),
                       'stop': reg.GetOptimizerStopConditionDescription(),
                       'iteration': reg.GetOptimizerIteration(), 'seed': seed}


def compose_pull(*transforms):
    """SimpleITK resampling maps output points to input points, rightmost first.

    For T2: T2<-T1 rigid(T1<-MNI rigid(affine(p_MNI))). Use exactly the
    same composite for intensities and the native T2 foreground mask.
    """
    combo = sitk.CompositeTransform(3)
    for transform in transforms:
        combo.AddTransform(transform)
    combo.FlattenTransform()
    return combo


def warp(image, reference, transform, mask=False):
    return sitk.Resample(image, reference, transform,
                         sitk.sitkNearestNeighbor if mask else sitk.sitkLinear,
                         0, sitk.sitkUInt8 if mask else sitk.sitkFloat32)


def array_image(values, reference):
    image = sitk.GetImageFromArray(values)
    image.CopyInformation(reference)
    return image


def masked_zscore_crop(image, mask, margin=1):
    values = sitk.GetArrayFromImage(image)
    brain = sitk.GetArrayFromImage(mask) > 0
    foreground = values[brain]
    if foreground.size < 1000 or not np.isfinite(values).all() or foreground.std() <= 0:
        raise ValueError('Invalid/constant registered brain')
    z = np.zeros_like(values, dtype=np.float32)
    z[brain] = (foreground - foreground.mean()) / foreground.std()
    zimage = array_image(z, image)
    indices = np.where(brain)
    start_zyx = [max(0, int(v.min()) - margin) for v in indices]
    stop_zyx = [min(n, int(v.max()) + margin + 1) for v, n in zip(indices, brain.shape)]
    size = [b-a for a, b in zip(start_zyx, stop_zyx)]
    cropped = sitk.RegionOfInterest(zimage, size[::-1], start_zyx[::-1])
    return zimage, cropped, {'start_xyz': start_zyx[::-1], 'size_xyz': size[::-1],
                             'mean': float(foreground.mean()), 'std': float(foreground.std())}


def qc_figure(reference, images, masks, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    template = sitk.GetArrayFromImage(reference)
    fig, axes = plt.subplots(3, 3, figsize=(10, 10))
    for row, (name, image) in enumerate([('MNI template', reference), *images.items()]):
        a = sitk.GetArrayFromImage(image)
        foreground = a[a > 0]
        limits = np.percentile(foreground, [1, 99])
        for col, axis in enumerate((0, 1, 2)):
            index = a.shape[axis] // 2
            plane = np.take(a, index, axis=axis)
            axes[row, col].imshow(plane, cmap='gray', origin='lower', vmin=limits[0], vmax=limits[1])
            if row:
                bm = np.take(sitk.GetArrayFromImage(masks[name]), index, axis=axis)
                axes[row, col].contour(bm, levels=[0.5], colors=['lime'], linewidths=0.5)
                axes[row, col].contour(np.take(template > 0, index, axis=axis),
                                       levels=[0.5], colors=['orange'], linewidths=0.5)
            axes[row, col].set_title(f'{name}, array axis {axis}')
            axes[row, col].axis('off')
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def process_case(source, root, template_path, seed=42, threads=2):
    root, template_path = Path(root), Path(template_path)
    output = root / 'cases' / source['case_id']
    output.mkdir(parents=True, exist_ok=True)
    contract = {'protocol': PROTOCOL, 'seed': seed, 'threads': threads,
                'template_sha256': file_hash(template_path), 'implementation_sha256': IMPLEMENTATION_SHA256,
                'SimpleITK': sitk.Version_VersionString(), 'numpy': np.__version__,
                'source_sha256': {k: file_hash(v) for k, v in source['images'].items()},
                'registration': {'bins': 50, 'sampling': 0.25, 'iterations': 1000},
                'n4': 'SimpleITK_N4BiasFieldCorrection_defaults_full_resolution',
                'brain_mask': 'native_image_nonzero_no_tumor_labels', 'extra_resample_4mm': False}
    fingerprint = digest(contract)
    done = output / 'complete.json'
    if done.exists():
        previous = json.loads(done.read_text())
        if previous['fingerprint'] != digest(previous['contract']) or not compatible_contract(previous['contract'], contract):
            raise ValueError(f'Preprocessing contract changed; use a new named directory: {output}')
        for name, sha in previous['artifacts'].items():
            if not (output / name).is_file() or file_hash(output / name) != sha:
                raise ValueError(f'Incomplete/corrupt preprocessing artifact: {output / name}')
        return {'case_id': source['case_id'], 'status': 'cached'}
    # A process lock avoids two launchers modifying the same intermediate files.
    import fcntl
    with (output / '.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return _process(source, output, template_path, contract, fingerprint, threads)


def _process(source, output, template_path, contract, fingerprint, threads):
    sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(threads)
    start = time.monotonic()
    write_json(output / 'status.json', {'stage': 'registration', 'contract': contract})
    template = sitk.ReadImage(str(template_path), sitk.sitkFloat32)
    native = {m: sitk.ReadImage(p, sitk.sitkFloat32) for m, p in source['images'].items()}
    for m, image in native.items():
        values = sitk.GetArrayViewFromImage(image)
        if not np.isfinite(values).all() or values.min() < 0 or values.max() <= 0:
            raise ValueError(f'Expected nonnegative skull-stripped BraTS intensity: {m}')
    native_masks = {m: sitk.Cast(image != 0, sitk.sitkUInt8) for m, image in native.items()}
    rigid, rmeta = registration(template, native['T1w'], 'rigid', contract['seed'])
    rigid_image = warp(native['T1w'], template, rigid)
    affine, ameta = registration(template, rigid_image, 'affine', contract['seed'] + 1)
    t2rigid, tmeta = registration(native['T1w'], native['T2w'], 'rigid', contract['seed'] + 2)
    transforms = {'T1w': compose_pull(rigid, affine), 'T2w': compose_pull(t2rigid, rigid, affine)}
    for name, transform in [('T1w_rigid', rigid), ('T1w_affine', affine), ('T2w_to_T1w_rigid', t2rigid),
                            *[(m + '_mni_pull', t) for m, t in transforms.items()]]:
        sitk.WriteTransform(transform, str(output / f'{name}.tfm'))
    sitk.WriteImage(rigid_image, str(output / 'T1w_mni_rigid.nii.gz'))
    images, masks, qc = {}, {}, {}
    review_required = []
    for m in ('T1w', 'T2w'):
        warped = warp(native[m], template, transforms[m])
        mask = warp(native_masks[m], template, transforms[m], mask=True)
        sitk.WriteImage(native_masks[m], str(output / f'{m}_native_mask.nii.gz'))
        sitk.WriteImage(warped, str(output / f'{m}_mni_warped.nii.gz'))
        sitk.WriteImage(mask, str(output / f'{m}_mni_mask.nii.gz'))
        write_json(output / 'status.json', {'stage': f'{m}_N4', 'contract': contract})
        corrected = sitk.N4BiasFieldCorrection(warped, mask)
        # Explicit zero background; interpolation outside the mask is not brain signal.
        corrected = sitk.Mask(corrected, mask)
        images[m], masks[m] = corrected, mask
        zscore, cropped, crop = masked_zscore_crop(corrected, mask)
        for suffix, image in [('n4', corrected), ('zscore', zscore), ('zscore_cropped', cropped)]:
            sitk.WriteImage(image, str(output / f'{m}_mni_{suffix}.nii.gz'))
        a = sitk.GetArrayFromImage(mask) > 0
        b = sitk.GetArrayFromImage(template) > 0
        dice = 2 * np.logical_and(a, b).sum() / (a.sum() + b.sum())
        boundary = any(np.take(a, i, axis=axis).any() for axis in range(3) for i in (0, -1))
        qc[m] = {'geometry': geometry(corrected), 'mask_voxels': int(a.sum()),
                 'template_foreground_dice': float(dice), 'mask_touches_canvas_edge': bool(boundary),
                 'crop': crop, 'image_mask_same_transform': True,
                 'native_foreground_voxels': int(sitk.GetArrayViewFromImage(native_masks[m]).sum())}
        coverage = canvas_coverage(native_masks[m], template, transforms[m]) if boundary else {
            'outside_canvas_voxels': 0, 'outside_canvas_fraction': 0., 'expanded_boundary_touched': False,
            'review_threshold_fraction': 0.001}
        qc[m].update(coverage)
        qc[m]['small_boundary_warning'] = bool(boundary and coverage['outside_canvas_fraction'] <= 0.001)
        if dice < 0.65 or coverage['outside_canvas_fraction'] > .001 or coverage['expanded_boundary_touched']:
            review_required.append(m)
        write_json(output / 'qc.json', qc)
    qc_figure(template, images, masks, output / 'qc.png')
    if review_required:
        write_json(output / 'status.json', {'stage': 'qc_review_required', 'contract': contract, 'modalities': review_required})
        raise ValueError(f'Registration needs review: {source["case_id"]}/{review_required}; see qc.json and qc.png')
    artifacts = {p.name: file_hash(p) for p in output.iterdir() if p.suffix in ('.gz', '.tfm', '.png')}
    write_json(output / 'complete.json', {'case_id': source['case_id'], 'fingerprint': fingerprint,
               'contract': contract, 'registration': {'rigid': rmeta, 'affine': ameta, 'T2_to_T1': tmeta},
               'qc': qc, 'artifacts': artifacts, 'seconds': time.monotonic() - start,
               'clinical_qc': 'not_clinically_certified'})
    write_json(output / 'status.json', {'stage': 'complete'})
    return {'case_id': source['case_id'], 'status': 'complete', 'seconds': time.monotonic() - start}
