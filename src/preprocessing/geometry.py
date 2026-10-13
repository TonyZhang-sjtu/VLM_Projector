"""Physical reference grids; SimpleITK metadata uses xyz, network arrays use DHW."""
import numpy as np
import SimpleITK as sitk


def geometry(image):
    direction = np.array(image.GetDirection()).reshape(3,3)
    affine = np.eye(4)
    affine[:3,:3] = direction @ np.diag(image.GetSpacing())
    affine[:3,3] = image.GetOrigin()
    return {'size_xyz': list(image.GetSize()), 'spacing_xyz': list(image.GetSpacing()),
            'origin_lps': list(image.GetOrigin()), 'direction': list(image.GetDirection()),
            'affine_xyz_to_lps': affine.tolist()}


def reference(meta, pixel=sitk.sitkFloat32):
    image = sitk.Image(meta['size_xyz'], pixel)
    image.SetSpacing(meta['spacing_xyz'])
    image.SetOrigin(meta['origin_lps'])
    image.SetDirection(meta['direction'])
    return image


def slice_axis(image, policy):
    spacing, size = np.array(image.GetSpacing()), np.array(image.GetSize())
    if policy == 'published_z' or len(set(spacing)) == 1:
        return 2
    # Preserve the local reference implementation, including its size heuristic.
    counts = np.bincount(size)
    unique_sizes = np.where(counts == 1)[0]
    if (counts == 1).all():
        return int(np.argmax(spacing))
    if not len(unique_sizes):
        raise ValueError('Official slice-axis heuristic is ambiguous; choose a documented axis policy')
    return int(np.where(size == unique_sizes[0])[0][0])


def transform_image(image, config, cohort):
    if image.GetDimension() != 3 or image.GetNumberOfComponentsPerPixel() != 1:
        raise ValueError('Expected scalar 3D volume')
    original = geometry(image)
    orienter = sitk.DICOMOrientImageFilter()
    orienter.SetDesiredCoordinateOrientation(config.orientation)
    oriented = orienter.Execute(image)
    axis = slice_axis(oriented, config.slice_axis_policy)
    old_spacing = np.array(oriented.GetSpacing())
    size = np.array(oriented.GetSize())
    target = [float(config.inplane_mm)] * 3
    target[axis] = float(config.throughplane_mm)
    if config.spacing_policy == 'native_grid':
        target = old_spacing.tolist()
    elif config.spacing_policy == 'brats2mm_other_native_z' and not cohort.startswith('BraTS_'):
        target[axis] = float(old_spacing[axis])
    new_size = [int(size[i] * (old_spacing[i] / target[i])) for i in range(3)]
    if min(new_size) < 1:
        raise ValueError('Resampling would create an empty dimension')
    if list(old_spacing) == target:
        resampled = oriented
    else:
        resampler = sitk.ResampleImageFilter()
        resampler.SetOutputSpacing(target)
        resampler.SetSize(new_size)
        resampler.SetOutputOrigin(oriented.GetOrigin())
        resampler.SetOutputDirection(oriented.GetDirection())
        resampler.SetInterpolator({'bspline':sitk.sitkBSpline, 'linear':sitk.sitkLinear}[config.image_interpolation])
        resampled = resampler.Execute(oriented)
    multiples = [16,16,16]
    multiples[axis] = 4
    crop_start, crop_size, pad_before, pad_after = [0]*3, list(resampled.GetSize()), [0]*3, [0]*3
    if config.boundary_policy == 'official_crop':
        crop_start = [(n % m)//2 for n,m in zip(crop_size,multiples)]
        crop_size = [(n//m)*m for n,m in zip(crop_size,multiples)]
        if min(crop_size) == 0:
            raise ValueError('Volume smaller than one patch after cropping')
        bounded = sitk.Extract(resampled, crop_size, crop_start)
    else:
        excess = [(-n) % m for n,m in zip(crop_size,multiples)]
        pad_before = [p//2 for p in excess]
        pad_after = [p-a for p,a in zip(excess,pad_before)]
        bounded = resampled
    array_axes = {2:[0,1,2], 1:[1,0,2], 0:[2,1,0]}[axis]
    metadata = {'original':original, 'oriented':geometry(oriented),
                'resampled':geometry(resampled), 'bounded_before_pad':geometry(bounded),
                'reorient_permutation':list(orienter.GetPermuteOrder()),
                'reorient_flips':list(orienter.GetFlipAxes()),
                'slice_axis_xyz':axis, 'array_zyx_to_dhw':array_axes,
                'crop_start_xyz':crop_start, 'crop_size_xyz':crop_size,
                'pad_before_xyz':pad_before, 'pad_after_xyz':pad_after,
                'spacing_policy':config.spacing_policy, 'boundary_policy':config.boundary_policy,
                'resampling_applied':list(old_spacing) != target,
                'size_rule':'floor(size*old_spacing/new_spacing)',
                'physical_units':'mm (BraTS MEN unspecified NIfTI units interpreted per published grid)'}
    return bounded, metadata
