"""A4 budgets, spatial locality, gradients, saved contracts and launch recipe."""
import copy
import json
import math
import os
import subprocess
import sys

import pytest
import torch
from omegaconf import OmegaConf

from src.config import ROOT, load_config
from src.models.projectors.registry import build_projector
from src.models.projectors.budget import derive_budget, resolve_budget
from src.models.projectors.bindformer_a4.regions import nested_boxes, packed_children
from src.models.projectors.bindformer.regions import spatial_boxes
from test_projectors import config, volume


@pytest.mark.parametrize('ratio,g,k,m', [(32, 67, 52, 275), (64, 34, 26, 138),
                                       (128, 17, 13, 69), (256, 7, 7, 35)])
def test_same_a_budget_and_complete_nested_disjoint_coverage(ratio, g, k, m):
    grid, extent = (39, 15, 15), (156., 240., 240.)
    a = derive_budget('bindformer3d', ratio, grid, extent)
    a4 = derive_budget('bindformer_a4', ratio, grid, extent)
    assert a4['global_tokens'] == a['global_tokens'] == g
    assert a4['local_tokens'] == a['local_tokens'] == k * 4
    assert a4['expected_output_tokens'] == a['expected_output_tokens'] == m
    assert a4['relative_error'] <= .05
    parents, children = nested_boxes(grid, k, extent)
    assert parents == spatial_boxes(grid, k, extent)
    ids, valid = packed_children(grid, k, extent)
    assert len(ids) == k * 4 and valid.any(1).all()
    assert torch.equal(ids[valid].sort().values, torch.arange(math.prod(grid)))
    for i, (lo, hi) in enumerate(parents):
        boxes = children[4*i:4*i+4]
        assert sum(math.prod(b-a for a, b in zip(*box)) for box in boxes) == math.prod(
            b-a for a, b in zip(lo, hi))
        assert all(all(l <= a < b <= h for l, a, b, h in zip(lo, a0, b0, hi)) for a0, b0 in boxes)


def test_only_surviving_a_transforms_match_independent_mean_reference():
    torch.manual_seed(17)
    base = build_projector(config('bindformer3d', target=12, fixed_spatial_regions=True,
                                  spatial_mean_aggregation=True), 8, 6).eval()
    model = build_projector(config('bindformer_a4', target=12), 8, 6).eval()
    # This is an equivalence check only; the production constructor never copies A.
    for name in ('input_adapter', 'global_layers', 'final_norm', 'head'):
        getattr(model, name).load_state_dict(getattr(base, name).state_dict())
    model.global_queries.data.copy_(base.global_queries.data)
    model.value_norm.load_state_dict(base.reader.norm_visual.state_dict())
    model.value.weight.data.copy_(base.reader.to_kv.weight.data[8:])
    model.output.load_state_dict(base.reader.to_out.state_dict())
    model.ffn.load_state_dict(base.reader.ffn.state_dict())
    x, coords, meta = volume((4, 4, 4))
    h = base.input_adapter(x, coords, meta, None)
    values = base.reader.to_kv(base.reader.norm_visual(h)).chunk(2, -1)[1].reshape(4, 4, 4, 8)
    _, children = nested_boxes((4, 4, 4), 2, (4., 4., 4.))
    means = torch.stack([values[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]].mean((0, 1, 2))
                         for lo, hi in children])
    local = base.reader.to_out(means)
    local = local + base.reader.ffn(local)
    glob = base.global_queries[None]
    for layer in base.global_layers:
        glob = layer(glob, h[None])
    expected = base.head(base.final_norm(torch.cat((glob[0], local))))
    actual = model(x, coords, meta).embeddings
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(actual[:4], base(x, coords, meta).embeddings[:4])
    forbidden = ('router', 'role_queries', 'concept_memory', 'reader', 'interactions',
                 'feature_reconstructor', 'geometry_encoder')
    assert all(not hasattr(model, name) for name in forbidden)
    assert set(dict(model.named_children())) == {'input_adapter', 'global_layers',
        'value_norm', 'value', 'output', 'ffn', 'final_norm', 'head'}
    assert not any('role' in key or 'slot' in key or 'concept' in key for key in model.state_dict())


def test_each_local_token_only_reads_its_own_child_and_padding_is_ignored():
    torch.manual_seed(18)
    model = build_projector(config('bindformer_a4', target=12), 8, 6).eval()
    x, coords, meta = volume((5, 5, 3))
    ids, valid = packed_children((5, 5, 3), 2, (5., 5., 3.))
    original = model(x, coords, meta).embeddings
    changed = x.detach().clone()
    changed[ids[0, 0]] += torch.arange(8.)
    other = model(changed, coords, meta).embeddings
    assert not torch.equal(original[4], other[4])
    torch.testing.assert_close(original[5:], other[5:], rtol=0, atol=0)
    original[4].square().sum().backward()
    support = torch.zeros(len(x), dtype=torch.bool)
    support[ids[0, valid[0]]] = True
    assert x.grad[support].abs().sum() > 0
    assert x.grad[~support].count_nonzero() == 0


@pytest.mark.parametrize('ratio,expected', [(128, 69), (256, 35)])
def test_real_grid_checkpointing_backward_and_bf16(ratio, expected):
    c = load_config([ROOT/'configs/projector/bindformer_a4.yaml'], [
        f'projector.target_compression_ratio={ratio}', 'projector.hidden_size=16',
        'projector.params.inner_dim=8', 'projector.params.num_heads=2'])
    resolve_budget(c.projector, (39, 15, 15), (156., 240., 240.))
    model = build_projector(c, 8, 6)
    plain = copy.deepcopy(model)
    plain.params['checkpoint_blocks'] = False
    x, coords, meta = volume((39, 15, 15))
    meta.update(shape_dhw=[156, 240, 240], bounded_before_pad={'spacing_xyz': [1., 1., 1.]},
                array_zyx_to_dhw=[0, 1, 2], oriented={'direction': [1., 0, 0, 0, 1., 0, 0, 0, 1.]},
                sequence='T1C')
    with torch.autocast('cpu', dtype=torch.bfloat16):
        actual = model(x, coords, meta)
        reference = plain(x, coords, meta)
    assert actual.output_tokens == expected and actual.input_tokens == 8775
    assert actual.metadata['projector']['local_tokens'] + actual.metadata['projector']['global_tokens'] == expected
    torch.testing.assert_close(actual.embeddings, reference.embeddings, rtol=0, atol=0)
    actual.embeddings.float().square().sum().backward()
    reference.embeddings.float().square().sum().backward()
    for p, q in zip(model.parameters(), plain.parameters()):
        assert p.grad is not None and torch.isfinite(p.grad).all()
        torch.testing.assert_close(p.grad, q.grad, rtol=0, atol=0)


def test_a4_rejects_removed_legacy_settings():
    for key, value in [('slot_iterations', '3'), ('knowledge_mode', 'none'),
                       ('within_slot_sa', 'false'), ('feature_loss_weight', '0')]:
        with pytest.raises(ValueError, match='Unsupported bindformer_a4'):
            config('bindformer_a4', **{key: value})


def test_launcher_matches_a_recipe_and_leaves_checkpoint_initialization_empty(tmp_path):
    capture = tmp_path/'capture_python'
    capture.write_text('#!' + sys.executable + '\nimport json,os,sys\n'
        'print(json.dumps({"args":sys.argv[1:],"gpu":os.environ["CUDA_VISIBLE_DEVICES"]}))\n')
    capture.chmod(0o700)
    local = tmp_path/'local.yaml'
    local.write_text('{}\n')
    env = dict(os.environ, PROJECTOR_PYTHON=str(capture), PROJECTOR_LOCAL_CONFIG=str(local),
               CUDA_VISIBLE_DEVICES='3', PROJECTOR_RUN_TAG='test_fresh_a4',
               BIND_WITHIN_SLOT_SA='true', BIND_FIXED_SPATIAL_REGIONS='false')
    configs = []
    for name in ('a', 'a4'):
        result = subprocess.run(['bash', str(ROOT/f'scripts/Exp1026/run_baseline_{name}_r128_r256.sh'),
                                 '--dry-run'], cwd=tmp_path, env=env, capture_output=True, text=True, check=True)
        value = json.loads(result.stdout)
        args = value['args']
        assert value['gpu'] == '3' and '--require-full-test' in args and '--dry-run' in args
        from src.cli.baseline_sweep import parser
        parsed = parser().parse_args(args[3:])
        assert parsed.ratios == [128, 256]
        method = parsed.projector
        files = [ROOT/'configs/training/joint_lora.yaml', ROOT/'configs/training/deterministic.yaml',
                 ROOT/'configs/preprocessing/native_grid.yaml', ROOT/f'configs/projector/{method}.yaml']
        files += parsed.config
        settings = parsed.overrides
        c = load_config(files, settings + ['projector.target_compression_ratio=128'])
        configs.append(OmegaConf.to_container(c, resolve=True))
        assert c.training.init_checkpoint is None and c.training.resume is None
        assert c.training.stage == 'joint_lora' and c.decoder.frozen
    # Entire data, preprocessing, encoder/LLM, seed and training recipe must match.
    a, a4 = configs
    assert a.pop('projector')['name'] == 'bindformer3d'
    assert a4.pop('projector')['name'] == 'bindformer_a4'
    assert a == a4
