import copy
import json

import numpy as np
import pytest
import SimpleITK as sitk

from src.preprocessing.neuroqa_registration_v2 import matrix_info, prepare_native
from src.config import file_hash
from src.preprocessing import neuroqa_smri_v2 as smri_v2, neuroqa_registration_v2 as registration_v2
from src.preprocessing.neuroqa_smri_v2 import issues, spatial_contract
from scripts.vqa.neuroqa.audit_intensity_v2 import inspect as inspect_intensity
from scripts.vqa.neuroqa.audit_lesion_coverage_v2 import inspect as inspect_lesion
from src.preprocessing.neuroqa_smri import TEMPLATE


def test_sparse_negative_repair_preserves_native_foreground_and_geometry():
    values = np.zeros((60, 60, 60), dtype=np.float32)
    values[5:55, 5:55, 5:55] = 100
    values[20, 20, 20] = -20
    image = sitk.GetImageFromArray(values)
    image.SetSpacing((1., 2., 3.))
    image.SetOrigin((8., 9., 10.))
    repaired, mask, info = prepare_native(image)
    a, m = sitk.GetArrayFromImage(repaired), sitk.GetArrayFromImage(mask)
    assert np.array_equal(m > 0, values != 0)
    assert a[20, 20, 20] > 0 and a[0, 0, 0] == 0
    assert repaired.GetSpacing() == image.GetSpacing() and mask.GetOrigin() == image.GetOrigin()
    assert info['negative_voxels'] == 1


@pytest.mark.parametrize('mode', ['widespread_negative', 'nan'])
def test_invalid_intensities_are_not_silently_repaired(mode):
    values = np.ones((40, 40, 40), dtype=np.float32)
    if mode == 'widespread_negative':
        values[:2] = -1
    else:
        values[0, 0, 0] = np.nan
    with pytest.raises(ValueError):
        prepare_native(sitk.GetImageFromArray(values))


def test_qc_detects_degenerate_transform_stage_regression_and_coverage():
    t = sitk.AffineTransform(3)
    t.Scale((.01, 1., 1.))
    assert matrix_info(t)['determinant'] == pytest.approx(.01)
    final = {'dice':.95, 'outside_canvas_fraction':0., 'expanded_boundary_touched':False,
             'transform':{'determinant':1., 'singular_values':[1., 1., 1.]}}
    stage = {'initial':{'dice':.90}, 'rigid':{'dice':.94}, 'final':final}
    metrics = {'T1w':copy.deepcopy(stage), 'T2w':copy.deepcopy(stage), 'T2_to_T1_mask_dice':.98}
    policy = {'min_singular':.5, 'max_singular':1.5, 'min_dice':.8,
              'max_outside_fraction':.001, 'max_dice_stage_drop':.05, 'min_t2_t1_dice':.9}
    assert not issues(metrics, policy)
    metrics['T1w']['final']['transform']['singular_values'][0] = .01
    metrics['T2w']['final']['dice'] = .8
    metrics['T2w']['final']['outside_canvas_fraction'] = .01
    found = issues(metrics, policy)
    assert 'T1w:affine_scaling' in found
    assert 'T2w:affine_dice_degradation' in found
    assert 'T2w:canvas_coverage' in found
    metrics['T1w']['rigid']['dice'] = .7
    metrics['T2_to_T1_mask_dice'] = .5
    found = issues(metrics, policy)
    assert 'T1w:rigid_dice_degradation' in found
    assert 'T2_to_T1:low_mask_overlap' in found


def test_protocol_contract_rejects_changed_code_template_and_unfrozen_state(tmp_path):
    template = tmp_path/'template.nii.gz'
    template.write_bytes(b'template')
    frozen = {'status':'frozen', 'protocol':smri_v2.PROTOCOL, 'seed':42, 'threads':1,
              'implementation_sha256':file_hash(smri_v2.__file__),
              'registration_code_sha256':file_hash(registration_v2.__file__),
              'template_sha256':file_hash(template)}
    contract = spatial_contract(frozen, template)
    assert contract['protocol'] == smri_v2.PROTOCOL
    for change in ({'status':'planned'}, {'implementation_sha256':'changed'}, {'template_sha256':'changed'}):
        with pytest.raises(ValueError):
            spatial_contract({**frozen, **change}, template)


def test_intensity_audit_uses_fixed_points_and_flags_a_real_shift(tmp_path):
    rng = np.random.default_rng(42)
    a = np.zeros((40, 40, 40), dtype=np.float32)
    a[4:36,4:36,4:36] = rng.uniform(1, 100, (32,32,32))
    img = sitk.GetImageFromArray(a)
    case = 'synthetic'
    folder = tmp_path/'cases'/case
    folder.mkdir(parents=True)
    (tmp_path/'templates').mkdir()
    sitk.WriteImage(img,str(tmp_path/'templates'/TEMPLATE))
    images = {}
    for m in ('T1w','T2w'):
        path = tmp_path/f'{m}.nii.gz'
        sitk.WriteImage(img,str(path))
        images[m] = str(path)
    record = {'contract':{'source_sha256':{m:file_hash(p) for m,p in images.items()}}}
    (folder/'complete.json').write_text(json.dumps(record))
    identity = sitk.AffineTransform(3)
    for name in ('initial','rigid','T2w_to_T1w_rigid','T1w_mni_pull','T2w_mni_pull'):
        sitk.WriteTransform(identity,str(folder/f'{name}.tfm'))
    source = {'case_id':case,'images':images}
    aligned = inspect_intensity(source,tmp_path)
    assert not aligned['flags']
    assert aligned['modalities']['T1w']['final'] > 1.8
    assert aligned['modalities']['T1w']['final'] == pytest.approx(aligned['modalities']['T1w']['initial'])
    shifted = sitk.AffineTransform(3)
    shifted.SetTranslation((2.,0.,0.))
    sitk.WriteTransform(shifted,str(folder/'T1w_mni_pull.tfm'))
    assert 'T1w:nmi_drop_from_initial_gt_0.005' in inspect_intensity(source,tmp_path)['flags']


def test_lesion_coverage_flags_loss_outside_mni_canvas(tmp_path):
    case = 'synthetic'
    folder = tmp_path/'cases'/case
    folder.mkdir(parents=True)
    (tmp_path/'templates').mkdir()
    raw = sitk.GetImageFromArray(np.ones((40,40,40), dtype=np.float32))
    label = np.zeros((40,40,40), dtype=np.uint8)
    label[18:22,18:22,5:8] = 1
    source_path = tmp_path/f'{case}-t1n.nii.gz'
    sitk.WriteImage(raw,str(source_path))
    sitk.WriteImage(sitk.GetImageFromArray(label),str(tmp_path/f'{case}-seg.nii.gz'))
    sitk.WriteImage(raw,str(tmp_path/'templates'/TEMPLATE))
    transform = sitk.AffineTransform(3)
    sitk.WriteTransform(transform,str(folder/'T1w_mni_pull.tfm'))
    source = {'case_id':case,'images':{'T1w':str(source_path)}}
    assert not inspect_lesion(source,tmp_path)['flags']
    transform.SetTranslation((10.,0.,0.))
    sitk.WriteTransform(transform,str(folder/'T1w_mni_pull.tfm'))
    assert 'lesion_label_outside_mni_canvas' in inspect_lesion(source,tmp_path)['flags']
    transform.SetTranslation((100.,0.,0.))
    sitk.WriteTransform(transform,str(folder/'T1w_mni_pull.tfm'))
    assert 'native_lesion_absent_from_expanded_reference' in inspect_lesion(source,tmp_path)['flags']
