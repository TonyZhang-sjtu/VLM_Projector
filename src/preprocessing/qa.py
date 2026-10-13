"""Mask-based geometry checks are isolated from training inputs."""
import numpy as np
import SimpleITK as sitk
from scipy import ndimage
from src.preprocessing.geometry import geometry, reference


def aligned_mask(mask, original):
    actual=geometry(mask)
    mismatched=[k for k in ('size_xyz','spacing_xyz','origin_lps','direction')
                if not np.allclose(original[k],actual[k],atol=1e-4,rtol=0)]
    if not mismatched:
        return mask, {'status':'matching_headers'}
    mapped=sitk.Resample(mask,reference(original),sitk.Transform(),sitk.sitkNearestNeighbor,0,mask.GetPixelID())
    if not np.array_equal(sitk.GetArrayFromImage(mask),sitk.GetArrayFromImage(mapped)):
        raise ValueError(f'Image/mask geometry mismatch changes label voxels: {mismatched}')
    return mapped, {'status':'header_difference_with_bitwise_identical_reference_labels',
                    'fields':mismatched,'original_mask_geometry':actual}


def lesion_geometry_qa(sample, volume):
    if sample.split == 'test':
        raise ValueError('Preprocessing decisions/lesion QA are restricted to train and val')
    mask = sitk.ReadImage(sample.mask)
    original = volume.metadata['original']
    mask,alignment=aligned_mask(mask,original)
    array = sitk.GetArrayFromImage(mask)
    if not np.isfinite(array).all():
        raise ValueError('Nonfinite mask')
    before = reference(volume.metadata['resampled'])
    after = reference(volume.metadata['bounded_before_pad'])
    pre = sitk.GetArrayFromImage(sitk.Resample(mask,before,sitk.Transform(),sitk.sitkNearestNeighbor,0,mask.GetPixelID()))
    post = sitk.GetArrayFromImage(sitk.Resample(mask,after,sitk.Transform(),sitk.sitkNearestNeighbor,0,mask.GetPixelID()))
    labels = {}
    for label in np.unique(array):
        if label == 0:
            continue
        cc, count = ndimage.label(array == label)  # 6-connected, fixed definition
        components = sitk.GetImageFromArray(cc.astype(np.int32))
        components.CopyInformation(mask)
        sampled_cc = sitk.GetArrayFromImage(sitk.Resample(components,before,sitk.Transform(),sitk.sitkNearestNeighbor,0,sitk.sitkInt32))
        cropped_cc = sitk.GetArrayFromImage(sitk.Resample(components,after,sitk.Transform(),sitk.sitkNearestNeighbor,0,sitk.sitkInt32))
        n0, n1, n2 = [int(np.count_nonzero(a == label)) for a in (array,pre,post)]
        component_sizes = np.bincount(cc.ravel())[1:] * float(np.prod(mask.GetSpacing()))
        missing_ids = sorted(set(range(1,count+1)) - set(np.unique(sampled_cc)))
        size_bins = {}
        for title, lower, upper in [('lt10',0,10),('10to100',10,100),('100to1000',100,1000),('ge1000',1000,float('inf'))]:
            ids = {i+1 for i,size in enumerate(component_sizes) if lower <= size < upper}
            size_bins[title] = {'original':len(ids),'absent_after_resampling':len(ids & set(missing_ids))}
        labels[str(int(label))] = {'original_voxels':n0, 'resampled_voxels':n1,
            'after_crop_voxels':n2, 'crop_lost_voxels':n1-n2,
            'original_mm3':n0*float(np.prod(mask.GetSpacing())),
            'resampled_mm3':n1*float(np.prod(before.GetSpacing())),
            'original_components':int(count),
            'component_volume_bins_mm3':size_bins,
            'absent_component_volumes_mm3':[float(component_sizes[i-1]) for i in missing_ids],
            'components_absent_after_resampling':count-len(set(np.unique(sampled_cc))-{0}),
            'components_absent_after_crop':count-len(set(np.unique(cropped_cc))-{0})}
    return {'id':sample.id, 'cohort':sample.cohort, 'labels':labels,'alignment':alignment,
            'scope':'label values separately, no assumed shared disease semantics; nearest-neighbor geometry is not diagnostic sensitivity'}
