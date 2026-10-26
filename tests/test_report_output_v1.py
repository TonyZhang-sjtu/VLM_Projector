"""Numerical, semantic, isolation and complete-sidecar regressions for R1/R2."""
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import sys
import pytest
import torch

from experiments.report_output_v1.build import make_row, read_rows
from experiments.report_output_v1.extract import extract
from experiments.report_output_v1.schema import DEFAULTS
from experiments.report_output_v1.decoder_view import answer_masks, token_nll
from experiments.report_output_v1.objectives import direct_objective, streamed_objective
from experiments.report_output_v1.evaluation import evaluate_predictions
from src.config import ROOT
from test_decoder import setup


def row_for(text, split='train', rid='s'):
    return dict(id=rid, patient_id='p', cohort='BraTS_GLI', sequence='T2W', split=split, finding=text)


def labelled(text, pair_fields=()):
    source = row_for(text)
    facts = extract(source)
    alternatives = dict(laterality='right', site='temporal_lobe', signal='hypointense', edema='present')
    pairs = [dict(pair_id=f, field=f, partner_id='other', positive=facts['facts'][f]['value'],
                  negative=alternatives[f]) for f in pair_fields]
    return make_row(source, facts, pairs)


def test_groups_preserve_subject_and_do_not_invent_multi_lesion_labels():
    text = 'The lesion is in the left frontal lobe. There is no associated edema.'
    row = labelled(text, ('laterality', 'site', 'edema'))
    assert len(row['groups']) == 2
    assert row['groups'][0]['joint_eligible']
    assert not row['groups'][1]['joint_eligible']
    assert sum(len(g['candidates']) for g in row['groups']) == 3
    assert 'no' in row['facts']['edema']['spans'][0]['text'].lower()
    for text in ('Two lesions involve the left frontal lobe and right temporal lobe.',
                 'There are bilateral lesions.', 'The lesion causes a midline shift to the right.'):
        assert not labelled(text)['facts']


def test_uncertain_edema_and_missingness_not_false_targets():
    assert 'edema' not in labelled('The lesion is hyperintense, indicating edema or other pathology.')['facts']
    assert 'edema' not in labelled('The lesion is present. No mention of edema.')['facts']
    assert 'edema' not in labelled('The lesion may have edema.')['facts']


@pytest.mark.parametrize('span_weight,group_weight', [(0.1, 0.), (0.1, 0.1), (0., 0.1)])
def test_streamed_vjp_matches_direct_loss_and_gradient(setup, span_weight, group_weight):
    decoder, samples = setup
    text = 'The lesion is in the left frontal lobe and is hyperintense on T2W. There is no associated edema.'
    row = labelled(text, ('laterality', 'site', 'signal', 'edema'))
    sample = replace(samples[0], finding=text, sequence='T2W')
    settings = {**DEFAULTS, 'span_weight': span_weight, 'group_weight': group_weight}
    z = torch.randn(7, decoder.hidden_size, requires_grad=True)
    report, aux = direct_objective(decoder, sample, z, row, settings, ramp=.7)
    expected = torch.autograd.grad(report/31 + aux/3, z)[0]
    other = z.detach().clone().requires_grad_(True)
    actual_report, actual_aux, metrics = streamed_objective(decoder, sample, other, row, settings, ramp=.7)
    actual = torch.autograd.grad(actual_report/31 + actual_aux/3, other)[0]
    torch.testing.assert_close(actual_report, report)
    torch.testing.assert_close(actual_aux, aux)
    torch.testing.assert_close(actual, expected, rtol=3e-5, atol=3e-7)
    assert metrics['pair_count'] == (4 if group_weight else 0)
    assert all(p.grad is None and not p.requires_grad for p in decoder.parameters())
    # This is a VJP to the original upstream graph, not a detached visual leaf.
    linear = torch.nn.Linear(4, decoder.hidden_size, bias=False)
    x = torch.randn(7, 4)
    out = streamed_objective(decoder, sample, linear(x), row, settings)
    (out[0]/31 + out[1]/3).backward()
    assert linear.weight.grad.abs().sum() > 0


def test_span_offsets_and_report_ce_match_real_decoder(setup):
    decoder, samples = setup
    text = 'A lesion is in the left frontal lobe; no associated edema. Size: 2 × 3 mm.'
    sample = replace(samples[0], finding=text)
    start = text.index('no associated edema')
    spans = {'edema': [dict(start=start, end=start+len('no associated edema'), text='no associated edema')]}
    mask = answer_masks(decoder, sample, spans)['edema']
    ids, boundary = decoder.text_ids(sample, True)
    selected = decoder.tokenizer.decode([i for i, use in zip(ids[boundary:], mask) if use])
    assert 'no associated edema' in selected and not mask[-1]
    for budget in (7, 13):
        z = torch.randn(budget, decoder.hidden_size, requires_grad=True)
        nll, _ = token_nll(decoder, sample, z)
        expected, count = decoder.loss(decoder.prepare([sample], [z]))
        assert len(nll) == count
        torch.testing.assert_close(nll.sum(), expected, rtol=1e-6, atol=1e-5)
    bad = deepcopy(spans); bad['edema'][0]['text'] = 'wrong'
    with pytest.raises(ValueError, match='misaligned'):
        answer_masks(decoder, sample, bad)
    padded = replace(sample, finding='  '+text+'\n')
    shifted = {'edema': [dict(start=start+2, end=start+2+len('no associated edema'), text='no associated edema')]}
    assert answer_masks(decoder, padded, shifted)['edema'] == mask


def test_missing_labels_zero_aux_and_duplicate_negatives_do_not_multiply_weight(setup):
    decoder, samples = setup
    text = 'The lesion is in the left frontal lobe.'
    sample = replace(samples[0], finding=text)
    row = labelled(text, ('laterality', 'site'))
    settings = {**DEFAULTS, 'span_weight': .1, 'group_weight': .1}
    z = torch.randn(4, decoder.hidden_size, requires_grad=True)
    _, value = direct_objective(decoder, sample, z, row, settings)
    duplicate = deepcopy(row)
    for group in duplicate['groups']:
        group['candidates'] *= 2
    _, twice = direct_objective(decoder, sample, z, duplicate, settings)
    torch.testing.assert_close(value, twice)
    empty = dict(facts={}, groups=[])
    report, auxiliary, metrics = streamed_objective(decoder, sample, z, empty, settings)
    assert auxiliary == 0 and metrics['pair_count'] == metrics['span_fields'] == 0
    expected = decoder.loss(decoder.prepare([sample], [z]))[0]
    torch.testing.assert_close(report, expected)


def test_generated_metrics_count_omission_and_ignore_unknown_gold():
    reference = 'The lesion is in the left frontal lobe.'
    gold = labelled(reference)
    def prediction(text):
        return dict(id='s', patient_id='p', cohort='BraTS_GLI', sequence='T2W', split='val',
                    prediction=text, reference='REFERENCE ANSWERS MUST NOT BE USED', generated_token_count=10)
    metrics, records = evaluate_predictions([prediction(reference)], {'s': gold})
    assert metrics['macro_field_balanced_accuracy'] == 1 and metrics['joint_groups_correct'] == 1
    metrics, _ = evaluate_predictions([prediction('The lesion is in the right temporal lobe.')], {'s': gold})
    assert metrics['contradictory'] == 2 and metrics['macro_field_balanced_accuracy'] == 0
    metrics, _ = evaluate_predictions([prediction('The lesion is present.')], {'s': gold})
    assert metrics['missing'] == 2 and metrics['fact_accuracy'] == 0
    assert 'edema' not in metrics['fields']
    metrics, _ = evaluate_predictions([prediction('Two lesions are in the left frontal and right temporal lobes.')], {'s': gold})
    assert metrics['unresolved'] == 2 and metrics['fact_accuracy'] == 0


@contextmanager
def installed_runtime():
    from src.models.projectors import registry, specs
    import src.models.multimodal as multimodal
    import src.training.trainer as trainer
    from experiments.report_output_v1 import runtime
    defaults, validate = deepcopy(specs.DEFAULTS), specs.validate_params
    factory, model = registry.PROJECTORS['bindformer3d'], multimodal.MultimodalModel
    trainer_model, trainer_class, trainer_validate = trainer.MultimodalModel, trainer.Trainer, trainer.validate
    aliases = {n: sys.modules[n].MultimodalModel for n in ('src.cli.evaluate',) if n in sys.modules}
    try:
        runtime.install()
        yield
    finally:
        specs.DEFAULTS.clear(); specs.DEFAULTS.update(defaults); specs.validate_params = validate
        registry.PROJECTORS['bindformer3d'] = factory; multimodal.MultimodalModel = model
        trainer.MultimodalModel = trainer_model; trainer.Trainer = trainer_class; trainer.validate = trainer_validate
        for name, value in aliases.items():
            sys.modules[name].MultimodalModel = value
        runtime._installed = False


def test_disabled_extension_preserves_initialization_forward_gradient_and_update():
    from test_bindformer_modules import config, volume
    from src.models.projectors.registry import build_projector
    args = ['projector.params.knowledge_mode=none', 'projector.params.fixed_spatial_regions=true',
            'projector.params.spatial_mean_aggregation=true']
    torch.manual_seed(42)
    base = build_projector(config(*args), 8, 16)
    rng = torch.get_rng_state().clone()
    with installed_runtime():
        torch.manual_seed(42)
        new = build_projector(config(*args, 'projector.params.report_output.enabled=false'), 8, 16)
        assert torch.equal(rng, torch.get_rng_state())
        assert base.state_dict().keys() == new.state_dict().keys()
        features, coords, meta = volume()
        a, b = base(features, coords, meta).embeddings, new(features, coords, meta).embeddings
        assert torch.equal(a, b)
        a.square().sum().backward(); b.square().sum().backward()
        for pa, pb in zip(base.parameters(), new.parameters()):
            assert (pa.grad is None) == (pb.grad is None)
            if pa.grad is not None:
                assert torch.equal(pa.grad, pb.grad)
        for module in (base, new):
            torch.optim.AdamW(module.parameters(), lr=2e-4).step()
        for key, value in base.state_dict().items():
            assert torch.equal(value, new.state_dict()[key])


def test_runtime_rejects_unknown_knobs_and_incompatible_settings():
    from test_bindformer_modules import config
    with installed_runtime():
        with pytest.raises(ValueError, match='Unknown report_output'):
            config('projector.params.report_output.typo=1')
        with pytest.raises(ValueError, match='nonnegative'):
            config('projector.params.report_output.span_weight=-1')


def test_full_sidecar_masks_and_candidate_edits(setup):
    decoder, samples = setup
    path = ROOT/'assets/knowledge/report_output_v1'
    if not path.exists():
        path = ROOT/'.tmp/report_output_build_02'
    if not path.exists():
        pytest.skip('Build the report output sidecar first')
    source = {r['id']: r for r in read_rows(ROOT/'assets/data/RadGenome-Brain/training_manifests/single_sequence_findings_coref_v2/train.jsonl')}
    count = 0
    for row in read_rows(path/'train.jsonl'):
        text = source[row['id']]['finding']
        sample = replace(samples[0], finding=text, sequence=row['sequence'])
        answer_masks(decoder, sample, {f: v['spans'] for f, v in row['facts'].items()})
        for group in row['groups']:
            answer_masks(decoder, sample, group['spans'])
            for candidate in group['candidates']:
                edit = candidate['edit']
                assert text[edit['start']:edit['end']] == edit['original']
                assert candidate['text'].startswith(text[:edit['start']])
                answer_masks(decoder, replace(sample, finding=candidate['text']), candidate['spans'])
                count += 1
    assert count == json.loads((path/'metadata.json').read_text())['retained_pairs']
