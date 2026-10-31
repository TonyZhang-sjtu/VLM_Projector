"""Frozen, version-isolated NeuroQA v2 image processing.

The protocol is selected using the train pilot, then fixed before processing
val/test. Failed QC never produces complete.json or a model patch cache.
"""
import fcntl
import json
from pathlib import Path
import time

import numpy as np
import SimpleITK as sitk

from src.config import digest, file_hash
from src.data.neuroqa_manifest import write_json
from src.preprocessing.geometry import geometry
from src.preprocessing import neuroqa_registration_v2 as registration_module
from src.preprocessing.neuroqa_registration_v2 import CANDIDATES, assess, prepare_native, register
from src.preprocessing.neuroqa_smri import (TEMPLATE, compose_pull, masked_zscore_crop,
                                           qc_figure, warp)

PROTOCOL = 'neuroqa_brats_foreground_mni1mm_v2'


def spatial_contract(frozen, template_path):
    if frozen.get('status') != 'frozen' or frozen.get('protocol') != PROTOCOL:
        raise ValueError('V2 protocol is not frozen')
    here = file_hash(__file__)
    register_hash = file_hash(registration_module.__file__)
    if frozen.get('implementation_sha256') != here or frozen.get('registration_code_sha256') != register_hash:
        raise ValueError('V2 implementation differs from the frozen protocol; create a new protocol')
    template_hash = file_hash(template_path)
    if frozen['template_sha256'] != template_hash:
        raise ValueError('V2 MNI template changed')
    if frozen['seed'] != 42 or frozen['threads'] != 1:
        raise ValueError('Unexpected V2 seed/threads')
    return {'protocol': PROTOCOL, 'seed': frozen['seed'], 'threads': frozen['threads'],
        'template_sha256': template_hash, 'implementation_sha256': here,
        'registration_code_sha256': register_hash, 'protocol_sha256': digest(frozen),
        'SimpleITK': sitk.Version_VersionString(), 'numpy': np.__version__,
        'brain_mask': 'native_image_nonzero_no_tumor_labels',
        'n4': 'SimpleITK_N4BiasFieldCorrection_defaults_full_resolution', 'extra_resample_4mm': False}


def issues(metrics, policy, *, include_stage=True):
    problems = []
    for modality in ('T1w', 'T2w'):
        entry = metrics[modality]
        final, rigid = entry['final'], entry['rigid']
        matrix = final['transform']
        values = [final['dice'], final['outside_canvas_fraction'], matrix['determinant'], *matrix['singular_values']]
        if not np.isfinite(values).all() or matrix['determinant'] <= 0:
            problems.append(f'{modality}:invalid_transform')
        if min(matrix['singular_values']) < policy['min_singular'] or max(matrix['singular_values']) > policy['max_singular']:
            problems.append(f'{modality}:affine_scaling')
        if final['dice'] < policy['min_dice']:
            problems.append(f'{modality}:low_dice')
        if final['outside_canvas_fraction'] > policy['max_outside_fraction'] or final['expanded_boundary_touched']:
            problems.append(f'{modality}:canvas_coverage')
        if rigid['dice'] < entry['initial']['dice'] - policy['max_dice_stage_drop']:
            problems.append(f'{modality}:rigid_dice_degradation')
        if include_stage and final['dice'] < rigid['dice'] - policy['max_dice_stage_drop']:
            problems.append(f'{modality}:affine_dice_degradation')
    if metrics['T2_to_T1_mask_dice'] < policy['min_t2_t1_dice']:
        problems.append('T2_to_T1:low_mask_overlap')
    return problems


def calculate(source, template, native, masks, candidate, seed):
    settings = CANDIDATES[candidate]
    initial = sitk.CenteredTransformInitializer(template, native['T1w'], sitk.Euler3DTransform(),
                                                sitk.CenteredTransformInitializerFilter.GEOMETRY)
    rigid, rm = register(template, native['T1w'], 'rigid', seed, settings)
    rigid_image = warp(native['T1w'], template, rigid)
    affine, am = register(template, rigid_image, 'affine', seed+1, settings)
    t2rigid, tm = register(native['T1w'], native['T2w'], 'rigid', seed+2, settings)
    t2_in_t1 = sitk.GetArrayFromImage(warp(masks['T2w'], native['T1w'], t2rigid, True)) > 0
    a = sitk.GetArrayFromImage(masks['T1w']) > 0
    overlap = float(2*np.logical_and(a,t2_in_t1).sum()/max(1,a.sum()+t2_in_t1.sum()))
    rigid2 = compose_pull(t2rigid, rigid)
    transforms = {'initial': initial, 'rigid': rigid, 'affine': affine, 'T2w_to_T1w_rigid': t2rigid,
                  'T1w_mni_pull': compose_pull(rigid, affine), 'T2w_mni_pull': compose_pull(t2rigid, rigid, affine)}
    metrics = {
        'T1w': {'initial': assess(masks['T1w'], template, initial, False),
                'rigid': assess(masks['T1w'], template, rigid),
                'final': assess(masks['T1w'], template, transforms['T1w_mni_pull'])},
        'T2w': {'initial': assess(masks['T2w'], template, initial, False),
                'rigid': assess(masks['T2w'], template, rigid2),
                'final': assess(masks['T2w'], template, transforms['T2w_mni_pull'])},
        'T2_to_T1_mask_dice': overlap}
    return transforms, metrics, {'rigid': rm, 'affine': am, 'T2_to_T1': tm}


def process_case(source, root):
    root = Path(root)
    if root.name != 'neuroqa_mni1mm_v2':
        raise ValueError('V2 production results require their named directory')
    folder = root / 'cases' / source['case_id']
    folder.mkdir(parents=True, exist_ok=True)
    frozen = json.loads((root / 'protocol.json').read_text())
    template_path = root / 'templates' / TEMPLATE
    contract = {**spatial_contract(frozen, template_path),
                'source_sha256': {m: file_hash(p) for m,p in source['images'].items()}}
    fingerprint = digest(contract)
    with (folder / '.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        complete = folder / 'complete.json'
        if complete.exists():
            old = json.loads(complete.read_text())
            if old['contract'] != contract or old['fingerprint'] != fingerprint:
                raise ValueError('V2 case contract mismatch')
            for name, sha in old['artifacts'].items():
                if file_hash(folder/name) != sha:
                    raise ValueError(f'V2 artifact corrupt: {name}')
            return {'case_id': source['case_id'], 'status': 'cached'}
        return _process(source, folder, template_path, frozen, contract, fingerprint)


def _process(source, folder, template_path, frozen, contract, fingerprint):
    sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(1)
    started = time.monotonic()
    write_json(folder/'status.json', {'stage': 'registration', 'fingerprint': fingerprint})
    template = sitk.ReadImage(str(template_path), sitk.sitkFloat32)
    native, masks, intensity = {}, {}, {}
    for m, path in source['images'].items():
        native[m], masks[m], intensity[m] = prepare_native(sitk.ReadImage(path, sitk.sitkFloat32))
    attempts, selected = [], None
    transforms = metrics = metadata = None
    for candidate in (frozen['primary_candidate'], frozen['fallback_candidate']):
        try:
            transforms, metrics, metadata = calculate(source, template, native, masks, candidate, frozen['seed'])
        except Exception as exc:
            transforms = metrics = metadata = None
            attempts.append({'candidate': candidate, 'error':repr(exc)})
            write_json(folder/'attempts.json', attempts)
            continue
        errors = issues(metrics, frozen['qc'])
        attempts.append({'candidate': candidate, 'metrics': metrics, 'registration': metadata, 'issues': errors})
        # Persist diagnostic attempts even if all candidates are rejected.
        for name,t in transforms.items():
            sitk.WriteTransform(t, str(folder/f'attempt_{candidate}_{name}.tfm'))
        write_json(folder/'attempts.json', attempts)
        if not errors:
            selected = (candidate, transforms, metrics, metadata, 'affine')
            break
    if selected is None and transforms is not None and frozen['allow_rigid_fallback']:
        # Only a fully checked rigid result can replace failed affine outputs.
        rigid_transforms = {**transforms, 'T1w_mni_pull': transforms['rigid'],
                            'T2w_mni_pull': compose_pull(transforms['T2w_to_T1w_rigid'], transforms['rigid'])}
        rigid_metrics = {m: {**metrics[m], 'final': metrics[m]['rigid']} for m in ('T1w','T2w')}
        rigid_metrics['T2_to_T1_mask_dice'] = metrics['T2_to_T1_mask_dice']
        rigid_issues = issues(rigid_metrics, frozen['qc'], include_stage=False)
        attempts[-1]['rigid_fallback_issues'] = rigid_issues
        write_json(folder/'attempts.json', attempts)
        if not rigid_issues:
            selected = (candidate, rigid_transforms, rigid_metrics, metadata, 'rigid_only')
    if selected is None:
        if transforms is not None:
            images = {m: warp(native[m], template, transforms[m+'_mni_pull']) for m in ('T1w','T2w')}
            warped_masks = {m: warp(masks[m], template, transforms[m+'_mni_pull'], True) for m in ('T1w','T2w')}
            qc_figure(template, images, warped_masks, folder/'qc.png')
        write_json(folder/'status.json', {'stage':'qc_review_required','attempts':attempts})
        raise ValueError(f'V2 registration requires review: {source["case_id"]}, see attempts.json')
    candidate, transforms, metrics, metadata, transform_kind = selected
    for name, transform in transforms.items():
        sitk.WriteTransform(transform, str(folder/f'{name}.tfm'))
    sitk.WriteImage(warp(native['T1w'], template, transforms['rigid']), str(folder/'T1w_mni_rigid.nii.gz'))
    images, final_masks, qc = {}, {}, {}
    for m in ('T1w','T2w'):
        transform = transforms[m+'_mni_pull']
        image = warp(native[m], template, transform)
        mask = warp(masks[m], template, transform, True)
        sitk.WriteImage(masks[m], str(folder/f'{m}_native_mask.nii.gz'))
        sitk.WriteImage(image, str(folder/f'{m}_mni_warped.nii.gz'))
        sitk.WriteImage(mask, str(folder/f'{m}_mni_mask.nii.gz'))
        write_json(folder/'status.json', {'stage':f'{m}_N4','fingerprint':fingerprint})
        corrected = sitk.Mask(sitk.N4BiasFieldCorrection(image, mask), mask)
        z, crop, crop_info = masked_zscore_crop(corrected, mask)
        for suffix, result in [('n4',corrected),('zscore',z),('zscore_cropped',crop)]:
            sitk.WriteImage(result, str(folder/f'{m}_mni_{suffix}.nii.gz'))
        images[m], final_masks[m] = corrected, mask
        qc[m] = {**metrics[m]['final'], 'geometry':geometry(corrected), 'crop':crop_info,
                 'image_mask_same_transform':True, 'intensity_repair':intensity[m],
                 'small_boundary_warning':metrics[m]['final']['outside_canvas_voxels'] > 0}
    qc_figure(template, images, final_masks, folder/'qc.png')
    artifacts = {p.name:file_hash(p) for p in folder.iterdir() if p.suffix in ('.gz','.tfm','.png')}
    write_json(folder/'complete.json', {'case_id':source['case_id'], 'fingerprint':fingerprint,
        'contract':contract, 'selected_candidate':candidate, 'transform_kind':transform_kind,
        'registration':metadata, 'attempts':attempts, 'qc':qc, 'artifacts':artifacts,
        'seconds':time.monotonic()-started, 'clinical_qc':'not_clinically_certified'})
    write_json(folder/'status.json', {'stage':'complete','transform_kind':transform_kind,'candidate':candidate})
    return {'case_id':source['case_id'],'status':'complete','candidate':candidate,'transform_kind':transform_kind,
            'seconds':time.monotonic()-started}
