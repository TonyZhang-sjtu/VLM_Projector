"""QA data integrity, exact quotas, physical transforms and task loss semantics."""
from collections import Counter
from dataclasses import replace
import io
import json

import numpy as np
import pytest
import SimpleITK as sitk
import torch
from transformers import AutoTokenizer, LlamaConfig, LlamaForCausalLM

from src.config import load_config, path_of
from src.data.neuroqa_manifest import QASample, balanced_epoch, load_neuroqa, target
from src.evaluation.neuroqa import evaluate_rows
from src.models.decoders.llama import LlamaDecoder
from src.models.projectors.budget import derive_budget, NEUROQA_VERSION, resolve_budget
from src.models.projectors.registry import build_projector, PROJECTORS
from src.preprocessing.neuroqa_smri import compose_pull, warp, masked_zscore_crop


def qa_config(*overrides):
    return load_config(['configs/vqa/neuroqa/mixed.yaml'], list(overrides))


def populations():
    return load_neuroqa(qa_config(), False)[0]


def test_patient_guard_and_dual_eligible_balance_without_global_quota():
    pop = populations()
    from src.data.sequence_mapping import SEQUENCE_NAMES
    assert all(s.sequence in SEQUENCE_NAMES for rows in pop.values() for s in rows)
    assert [len(pop[s]) for s in ('train', 'val', 'test')] == [3454, 783, 785]
    assert not ({s.patient_id for s in pop['train']} & {s.patient_id for s in pop['test']})
    previous = {}
    for epoch in range(4):
        plan = balanced_epoch(pop['train'], 42, epoch)
        assert len({s.qa_id for s in plan}) == 3454
        assert [s.id for s in plan] == [s.id for s in balanced_epoch(pop['train'], 42, epoch)]
        counts = Counter(s.sequence for s in plan if len(s.row['allowed_modalities']) == 2)
        assert abs(counts['T1W'] - counts['T2W']) <= 1
        for sample in plan:
            if len(sample.row['allowed_modalities']) == 1:
                assert sample.id.endswith('__' + sample.row['allowed_modalities'][0])
            elif epoch:
                assert previous[sample.qa_id] != sample.sequence
            previous[sample.qa_id] = sample.sequence
        # Fixed-only questions have unequal populations; no global equalization.
        counts = Counter(s.sequence for s in plan)
        assert counts['T1W'] != counts['T2W']
    assert len(balanced_epoch(pop['train'][:3], 42, 0)) == 3
    with pytest.raises(ValueError, match='even'):
        balanced_epoch(pop['train'][:3], 42, 0, policy='legacy_exact')
    legacy = balanced_epoch(pop['train'], 42, 0, policy='legacy_exact')
    assert Counter(s.sequence for s in legacy) == {'T1W': 1727, 'T2W': 1727}


@pytest.mark.parametrize('a,b,c', [(2, 2, 0), (2, 1, 0), (5, 1, 1),
                                    (0, 0, 1), (0, 2, 3), (3, 0, 3)])
def test_exact_balance_after_filtering_minimizes_repetition(a, b, c):
    allowed = [['T1w']]*a + [['T2w']]*b + [['T1w', 'T2w']]*c
    samples = []
    for i, modalities in enumerate(allowed):
        row = {'neuroqa_id': str(i), 'allowed_modalities': modalities,
               'images': {'T1w': '/x/T1w.nii.gz', 'T2w': '/x/T2w.nii.gz'}}
        samples.append(QASample(str(i), 'train', 'brats', str(i), str(i), '', '', '', row))
    target = max(a, b, (len(samples)+1)//2)
    for epoch in range(4):
        plan = balanced_epoch(samples, 42, epoch, 'exact_balanced_repeat')
        assert [s.id for s in plan] == [s.id for s in balanced_epoch(samples, 42, epoch, 'exact_balanced_repeat')]
        assert Counter(s.sequence for s in plan) == {'T1W': target, 'T2W': target}
        assert {s.qa_id for s in plan} == {s.qa_id for s in samples}
        assert len(plan) - len(samples) == 2*target - len(samples)
        assert all(plan[i].sequence == 'T1W' and plan[i+1].sequence == 'T2W'
                   for i in range(0, len(plan), 2))
        assert all(s.id.endswith('__' + ('T1w' if s.sequence == 'T1W' else 'T2w'))
                   and ('T1w' if s.sequence == 'T1W' else 'T2w') in s.row['allowed_modalities']
                   for s in plan)


def test_exact_balance_repeats_rotate_and_missing_modality_fails():
    base = populations()['train']
    t1 = [s for s in base if s.row['allowed_modalities'] == ['T1w']][:4]
    t2 = [s for s in base if s.row['allowed_modalities'] == ['T2w']][:3]
    assert len(t1) == 4 and len(t2) == 3
    repeated = []
    for epoch in range(3):
        plan = balanced_epoch(t1+t2, 42, epoch, 'exact_balanced_repeat')
        repeated.append(next(q for q, count in Counter(s.qa_id for s in plan).items() if count == 2))
    assert len(set(repeated)) == 3
    with pytest.raises(ValueError, match='no usable QA for one modality'):
        balanced_epoch(t1, 42, 0, 'exact_balanced_repeat')


def test_v2_exact_repeat_configuration_requires_even_effective_batch():
    config = load_config(['configs/vqa/neuroqa/mixed_v2.yaml'], [])
    assert config.data.modality_sampling == 'exact_balanced_repeat'
    with pytest.raises(ValueError, match='even effective batch'):
        load_config(['configs/vqa/neuroqa/mixed_v2.yaml'], ['training.effective_batch_size=3'])


def test_question_types_targets_and_scorer_coverage():
    samples = []
    for kind in ('YN', 'MCQ', 'Open'):
        samples.append(next(s for s in populations()['val'] if s.row['question_type'] == kind))
    rows = [{'id': s.id, 'prediction': target(s.row)} for s in samples]
    metrics, scored = evaluate_rows(samples, rows)
    assert metrics['overall']['closed_accuracy'] == 1
    assert metrics['overall']['open_f1'] == 1
    for records in (rows[:-1], rows + [rows[0]], rows + [{'id': 'alien', 'prediction': ''}]):
        with pytest.raises(ValueError, match='coverage'):
            evaluate_rows(samples, records)
    rows[0]['prediction'] = 'I think yes, but no.'
    m, _ = evaluate_rows(samples, rows)
    assert m['overall']['invalid_count'] == 1
    assert m['overall']['closed_accuracy'] == .5


def test_transform_composition_resamples_image_and_mask_identically_and_crop_affine():
    # Noncommuting translation/rotation exposes a reversed composite order.
    r = sitk.Euler3DTransform()
    r.SetRotation(0, 0, .2)
    a = sitk.AffineTransform(3)
    a.Scale((1.1, .9, 1.))
    t = sitk.TranslationTransform(3, (3., -2., 1.))
    combo = compose_pull(t, r, a)
    point = (8., 7., 9.)
    assert np.allclose(combo.TransformPoint(point), t.TransformPoint(r.TransformPoint(a.TransformPoint(point))))
    values = np.zeros((32, 32, 32), dtype=np.float32)
    values[8:24, 8:24, 8:24] = np.arange(4096).reshape(16, 16, 16) + 1
    image = sitk.GetImageFromArray(values)
    mask = sitk.Cast(image > 0, sitk.sitkUInt8)
    shifted = warp(image, image, t)
    shifted_mask = warp(mask, image, t, mask=True)
    assert np.array_equal(sitk.GetArrayFromImage(shifted) > 0, sitk.GetArrayFromImage(shifted_mask) > 0)
    full, cropped, info = masked_zscore_crop(shifted, shifted_mask)
    assert np.allclose(cropped.GetOrigin(), full.TransformIndexToPhysicalPoint(info['start_xyz']))
    brain = sitk.GetArrayFromImage(full)[sitk.GetArrayFromImage(shifted_mask) > 0]
    assert abs(brain.mean()) < 1e-5 and abs(brain.std()-1) < 1e-5


@pytest.mark.parametrize('name', [n for n in PROJECTORS if n != 'mlp2x_gelu'])
@pytest.mark.parametrize('ratio,expected,grid,group', [
    (16,624,[6,8,13],[9,2,1]), (32,288,[9,8,4],[6,2,4]),
    (64,150,[6,5,5],[9,3,3]), (128,75,[3,5,5],[17,3,3]), (256,36,[3,4,3],[18,4,5])])
def test_all_mni_projector_budgets_forward_backward_roundtrip(name, ratio, expected, grid, group):
    overrides = [f'projector.name={name}', f'projector.target_compression_ratio={ratio}', 'projector.hidden_size=16']
    if name == 'bindformer3d':
        overrides += ['projector.params.knowledge_mode=none', 'projector.params.sequence_condition=false',
                      'projector.params.position_mode=none', 'projector.params.inner_dim=16']
    c = qa_config(*overrides)
    budget = resolve_budget(c.projector, [49,15,13], [196,240,208])
    assert budget['expected_output_tokens'] == expected
    if name == 'concat3d':
        assert budget['group_size'] == group and budget['output_grid_dhw'] == grid
    x = torch.randn(9555, 8, requires_grad=True)
    coords = torch.cartesian_prod(torch.arange(49), torch.arange(15), torch.arange(13))
    meta = {'grid_dhw': [49,15,13], 'shape_dhw': [196,240,208],
            'bounded_before_pad': {'spacing_xyz': [1,1,1]}, 'array_zyx_to_dhw': [0,1,2], 'id': 'mni-smoke'}
    model = build_projector(c, 8, 12)
    out = model(x, coords, meta)
    assert out.embeddings.shape == (expected, 12)
    out.embeddings.square().mean().backward()
    assert torch.isfinite(x.grad).all() and x.grad.abs().sum() > 0
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    before = next(model.parameters()).detach().clone()
    torch.optim.SGD(model.parameters(), lr=.1).step()
    assert not torch.equal(before, next(model.parameters()))
    saved = io.BytesIO()
    torch.save(model.state_dict(), saved)
    saved.seek(0)
    other = build_projector(c, 8, 12)
    other.load_state_dict(torch.load(saved, weights_only=True))
    torch.testing.assert_close(model(x, coords, meta).embeddings, other(x, coords, meta).embeddings, rtol=0, atol=0)


def test_qa_loss_matches_per_example_reference_and_accumulated_gradients():
    c = qa_config('decoder.gradient_checkpointing=false')
    tokenizer = AutoTokenizer.from_pretrained(path_of(c, c.decoder.path), local_files_only=True)
    lm = LlamaForCausalLM(LlamaConfig(vocab_size=128256, hidden_size=32, intermediate_size=64,
        num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=2048,
        attention_dropout=0.0, pad_token_id=128004, eos_token_id=128009))
    decoder = LlamaDecoder(c, lm, tokenizer)
    samples = [next(s for s in populations()['train'] if s.row['question_type'] == kind) for kind in ('YN','Open')]
    vision = [torch.randn(9,32,requires_grad=True), torch.randn(13,32,requires_grad=True)]
    batch = decoder.prepare(samples, vision)
    total, tokens, per_example_sum = decoder.loss(batch, per_example=True)
    reference = sum(decoder.lm(inputs_embeds=b.inputs_embeds, attention_mask=b.attention_mask,
                      position_ids=b.position_ids, labels=b.labels, use_cache=False).loss
                    for b in [decoder.prepare([s],[v]) for s,v in zip(samples,vision)])
    torch.testing.assert_close(per_example_sum, reference, rtol=1e-6, atol=1e-6)
    ga = torch.autograd.grad(per_example_sum/2, vision, retain_graph=True)
    gb = torch.autograd.grad(reference/2, vision)
    for a,b in zip(ga,gb):
        torch.testing.assert_close(a,b,rtol=2e-5,atol=1e-7)
    assert tokens == sum(decoder.answer_count(s) for s in samples)
    assert all(p.grad is None and not p.requires_grad for p in lm.parameters())
    assert (batch.labels[batch.attention_mask==0] == -100).all()


def test_completed_mni_artifacts_real_grid_background_and_cache():
    from einops import rearrange
    from src.preprocessing.registry import build_preprocessor
    config = qa_config()
    sample = next(s for s in populations()['train'] if s.case_id == 'BraTS-MEN-00380-000')
    from pathlib import Path
    if not (Path(sample.image).parent / 'complete.json').exists():
        pytest.skip('Run the documented real preprocessing pilot first')
    pre = build_preprocessor(config)
    for modality in ('T1w','T2w'):
        volume = pre(sample.view(modality))
        assert volume.patches.shape == (9555,1024)
        assert volume.metadata['shape_dhw'] == [196,240,208]
        assert volume.metadata['pad_before_xyz'] == [7,5,1]
        assert not volume.metadata['resampling_applied']
        array = rearrange(volume.patches, '(d h w) (p q r) -> (d p) (h q) (w r)',
                          d=49,h=15,w=13,p=4,q=16,r=16)
        background = -config.preprocessing.mri_mean/config.preprocessing.mri_std
        assert torch.allclose(array[0], torch.full_like(array[0],background))
        unnormalized = array*config.preprocessing.mri_std+config.preprocessing.mri_mean
        assert unnormalized.min() >= -1e-6 and unnormalized.max() <= 1+1e-6
        again = pre(sample.view(modality))
        torch.testing.assert_close(volume.patches, again.patches, rtol=0, atol=0)


def test_canvas_qc_distinguishes_contact_from_actual_clipping():
    from src.preprocessing.neuroqa_smri import canvas_coverage
    a = np.zeros((12,12,12), dtype=np.uint8)
    a[0:4,4:8,4:8] = 1
    mask = sitk.GetImageFromArray(a)
    contact = canvas_coverage(mask, sitk.Cast(mask, sitk.sitkFloat32), sitk.Transform(3,sitk.sitkIdentity))
    assert contact['outside_canvas_voxels'] == 0
    clipped = canvas_coverage(mask, sitk.Cast(mask, sitk.sitkFloat32), sitk.TranslationTransform(3,(0,0,1)))
    assert clipped['outside_canvas_voxels'] == 16
    assert clipped['outside_canvas_fraction'] == .25


def test_vqa_cannot_silently_fall_back_to_manual_budget_or_report_knowledge():
    with pytest.raises(ValueError, match='five ratios'):
        qa_config('projector.target_compression_ratio=null')
    with pytest.raises(ValueError, match='five ratios'):
        qa_config('projector.target_compression_ratio=8')
    with pytest.raises(ValueError, match='concept banks'):
        qa_config('projector.name=bindformer3d')
