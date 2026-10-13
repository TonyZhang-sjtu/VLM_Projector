from dataclasses import replace
import numpy as np
import SimpleITK as sitk
import torch
import pytest
from src.config import load_config
from src.data.manifest import load_manifests
from src.preprocessing.neurovfm import NeuroVFMPreprocessor
from src.preprocessing.geometry import transform_image
from neurovfm.data.utils import preprocess_image
from neurovfm.data.preprocess import prepare_for_inference, tokenize_volume


@pytest.fixture(scope='module')
def representatives():
    rows = load_manifests(load_config(), datasets=['all'])[0]['train']
    chosen = {}
    for s in rows:
        chosen.setdefault(s.cohort,s)
    return list(chosen.values())


def test_all_cohort_official_equivalence(representatives, tmp_path):
    c = load_config(overrides=[f'preprocessing.cache_dir={tmp_path}', 'preprocessing.spacing_policy=official_4mm', 'preprocessing.boundary_policy=official_crop'])
    pre = NeuroVFMPreprocessor(c)
    for sample in representatives:
        out = pre(sample)
        image = preprocess_image(sitk.ReadImage(sample.image))
        arrays, mask, _ = prepare_for_inference(image,mode='mri')
        tokens, coords, _ = tokenize_volume(arrays[0],mask,remove_background=False)
        expected = (torch.from_numpy(tokens).float()-0.3141)/0.2623
        torch.testing.assert_close(out.patches, expected, rtol=0, atol=0)
        torch.testing.assert_close(out.coords,torch.from_numpy(coords))
        torch.testing.assert_close(pre(sample).patches,out.patches,rtol=0,atol=0)


def test_native_grid_does_not_resample_and_padding(tmp_path, representatives):
    arr = np.arange(7*19*21,dtype=np.float32).reshape(7,19,21)
    image = sitk.GetImageFromArray(arr)
    image.SetSpacing((1.,1.,3.))
    path = tmp_path/'synthetic.nii.gz'
    sitk.WriteImage(image,str(path))
    sample = replace(representatives[0],image=str(path))
    c = load_config(overrides=['preprocessing.spacing_policy=native_grid',
        'preprocessing.boundary_policy=minimal_pad','preprocessing.slice_axis_policy=published_z',
        'preprocessing.cache_enabled=false'])
    out = NeuroVFMPreprocessor(c)(sample)
    assert not out.metadata['resampling_applied']
    assert out.metadata['shape_dhw'] == [8,32,32]
    oriented = sitk.DICOMOrient(image,'RPI')
    bounded, meta = transform_image(image,c.preprocessing,sample.cohort)
    np.testing.assert_array_equal(sitk.GetArrayFromImage(oriented),sitk.GetArrayFromImage(bounded))
    assert sum(meta['pad_before_xyz'])+sum(meta['pad_after_xyz']) == 25


def test_spacing_really_changes_output(representatives):
    sample = representatives[0]
    c4 = load_config(overrides=['preprocessing.cache_enabled=false','preprocessing.spacing_policy=4mm'])
    c2 = load_config(overrides=['preprocessing.cache_enabled=false','preprocessing.spacing_policy=resample','preprocessing.throughplane_mm=2'])
    four,two = NeuroVFMPreprocessor(c4)(sample),NeuroVFMPreprocessor(c2)(sample)
    assert two.metadata['tokens'] > four.metadata['tokens']
    assert 2 in two.metadata['resampled']['spacing_xyz']
