"""Raw-evidence locality, global-path equivalence, precision and launch contract."""
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
from src.models.projectors.budget import derive_budget, resolve_budget
from src.models.projectors.registry import build_projector
from src.models.projectors.bindformer_a4.regions import nested_boxes, packed_children
from test_projectors import config, volume


@pytest.mark.parametrize('ratio,g,local,m', [(128, 17, 52, 69), (256, 7, 28, 35)])
def test_budget_is_a4_with_identical_disjoint_children(ratio, g, local, m):
    grid, extent = (39, 15, 15), (156., 240., 240.)
    a4 = derive_budget('bindformer_a4', ratio, grid, extent)
    direct = derive_budget('bindformer_adirect', ratio, grid, extent)
    assert (direct['global_tokens'], direct['local_tokens'], direct['expected_output_tokens']) == (g, local, m)
    assert {k: v for k, v in direct.items() if k not in ('method', 'allocation_version')} == {
        k: v for k, v in a4.items() if k not in ('method', 'allocation_version')}
    ids, valid = packed_children(grid, direct['num_regions'], extent)
    assert len(ids) == local and valid.any(1).all()
    assert torch.equal(ids[valid].sort().values, torch.arange(math.prod(grid)))


def test_raw_mean_oracle_global_equivalence_and_no_local_adapters():
    torch.manual_seed(31)
    direct = build_projector(config('bindformer_adirect', target=12), 8, 6).double().eval()
    a4 = build_projector(config('bindformer_a4', target=12), 8, 6).double().eval()
    # For this oracle only: same retained weights isolates local architecture.
    for name in ('input_adapter', 'global_layers', 'final_norm', 'head'):
        getattr(a4, name).load_state_dict(getattr(direct, name).state_dict())
    a4.global_queries.data.copy_(direct.global_queries.data)
    x, coords, meta = volume((5, 5, 3))
    x = x.detach().double().requires_grad_(True)
    # Independently slice each child, avoiding the implementation's padded gather.
    cube = x.reshape(5, 5, 3, 8)
    _, children = nested_boxes((5, 5, 3), 2, (5., 5., 3.))
    means = torch.stack([cube[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]].mean((0, 1, 2))
                         for lo, hi in children])
    out = direct(x, coords, meta).embeddings
    torch.testing.assert_close(out[4:], direct.head(direct.final_norm(means)), rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(out[:4], a4(x, coords, meta).embeddings[:4], rtol=0, atol=0)
    assert set(dict(direct.named_children())) == {'input_adapter', 'global_layers', 'final_norm', 'head'}
    assert not any(any(part in key for part in ('role', 'slot', 'concept', 'reader', 'value_norm'))
                   for key in direct.state_dict())
    # Changing the global adapter cannot leak into any local token.
    with torch.no_grad():
        direct.input_adapter.projection.weight.normal_(std=5)
        direct.input_adapter.projection.bias.fill_(3)
    changed = direct(x, coords, meta).embeddings
    torch.testing.assert_close(changed[4:], out[4:], rtol=0, atol=0)
    assert not torch.equal(changed[:4], out[:4])


def test_local_input_support_gradient_and_no_padding_leakage():
    model = build_projector(config('bindformer_adirect', target=12), 8, 6).eval()
    x, coords, meta = volume((5, 5, 3))
    ids, valid = packed_children((5, 5, 3), 2, (5., 5., 3.))
    assert (~valid).any()  # Unequal cells exercise padding indices pointing at patch 0.
    before = model(x, coords, meta).embeddings
    changed = x.detach().clone()
    changed[ids[0, 0]] += torch.arange(8.)
    after = model(changed, coords, meta).embeddings
    assert not torch.equal(before[4], after[4])
    torch.testing.assert_close(before[5:], after[5:], rtol=0, atol=0)
    before[4].square().sum().backward()
    support = torch.zeros(len(x), dtype=torch.bool)
    support[ids[0, valid[0]]] = True
    assert x.grad[support].abs().sum() > 0 and x.grad[~support].count_nonzero() == 0
    torch.testing.assert_close(x.grad[support], x.grad[support][0].expand_as(x.grad[support]))


@pytest.mark.parametrize('ratio', [128, 256])
def test_real_grid_bf16_checkpointing_and_encoder_gradient(ratio):
    c = load_config([ROOT/'configs/projector/bindformer_adirect.yaml'], [
        f'projector.target_compression_ratio={ratio}', 'projector.hidden_size=16',
        'projector.params.inner_dim=8', 'projector.params.num_heads=2'])
    budget = resolve_budget(c.projector, (39, 15, 15), (156., 240., 240.))
    model = build_projector(c, 8, 6)
    plain = copy.deepcopy(model)
    plain.params['checkpoint_blocks'] = False
    x, coords, meta = volume((39, 15, 15))
    x = x.detach().to(torch.bfloat16).requires_grad_(True)
    y = x.detach().clone().requires_grad_(True)
    meta.update(shape_dhw=[156, 240, 240], bounded_before_pad={'spacing_xyz': [1., 1., 1.]},
                array_zyx_to_dhw=[0, 1, 2], oriented={'direction': [1., 0, 0, 0, 1., 0, 0, 0, 1.]},
                sequence='T1C')
    with torch.autocast('cpu', dtype=torch.bfloat16):
        actual = model(x, coords, meta)
        reference = plain(y, coords, meta)
    assert actual.embeddings.shape == (budget['expected_output_tokens'], 6)
    assert actual.auxiliary is None
    assert actual.metadata['projector']['local_feature_source'] == 'encoder_output'
    torch.testing.assert_close(actual.embeddings, reference.embeddings, rtol=0, atol=0)
    actual.embeddings.float().square().sum().backward()
    reference.embeddings.float().square().sum().backward()
    torch.testing.assert_close(x.grad, y.grad, rtol=0, atol=0)
    assert torch.isfinite(x.grad).all() and x.grad.abs().sum() > 0
    for p, q in zip(model.parameters(), plain.parameters()):
        assert p.grad is not None and torch.isfinite(p.grad).all()
        torch.testing.assert_close(p.grad, q.grad, rtol=0, atol=0)


def test_no_silent_local_channel_mapping_or_legacy_auxiliary_loss():
    with pytest.raises(ValueError, match='input_dim == inner_dim'):
        build_projector(config('bindformer_adirect', inner_dim=12), 8, 6)
    for key, value in [('slot_iterations', 3), ('feature_loss_weight', 0),
                       ('knowledge_mode', 'none'), ('within_slot_sa', False)]:
        with pytest.raises(ValueError, match='Unsupported bindformer_adirect'):
            config('bindformer_adirect', **{key: value})
    model = build_projector(config('bindformer_adirect', global_fraction=0), 8, 6)
    assert model.input_adapter is None and not model.requires_sequence
    model(*volume()).embeddings.square().sum().backward()
    assert all(p.grad is not None for p in model.parameters())


def test_launcher_matches_a_and_a4_recipe_without_old_checkpoints(tmp_path):
    capture = tmp_path/'capture_python'
    capture.write_text('#!' + sys.executable + '\nimport json,os,sys\n'
                      'print(json.dumps({"args":sys.argv[1:],"gpu":os.environ["CUDA_VISIBLE_DEVICES"]}))\n')
    capture.chmod(0o700)
    local = tmp_path/'local.yaml'
    local.write_text('{}\n')
    env = dict(os.environ, PROJECTOR_PYTHON=str(capture), PROJECTOR_LOCAL_CONFIG=str(local),
               CUDA_VISIBLE_DEVICES='3', PROJECTOR_RUN_TAG='test_adirect', BIND_WITHIN_SLOT_SA='true')
    effective = []
    from src.cli.baseline_sweep import parser
    for filename in ('run_baseline_a_r128_r256.sh', 'run_baseline_a4_r128_r256.sh', 'run_baseline_adirect.sh'):
        # Ratios are user-editable in launchers; compare this experiment's budgets explicitly.
        result = subprocess.run(['bash', str(ROOT/'scripts/Exp1026'/filename), '--ratios', '128', '256', '--dry-run'],
                                cwd=tmp_path, env=env, capture_output=True, text=True, check=True)
        value = json.loads(result.stdout)
        assert value['gpu'] == '3'
        args = parser().parse_args(value['args'][3:])
        assert args.ratios == [128, 256] and args.require_full_test and args.metrics == ['all']
        files = [ROOT/'configs/training/joint_lora.yaml', ROOT/'configs/training/deterministic.yaml',
                 ROOT/'configs/preprocessing/native_grid.yaml', ROOT/f'configs/projector/{args.projector}.yaml',
                 *args.config]
        c = load_config(files, args.overrides + ['projector.target_compression_ratio=128'])
        assert c.training.init_checkpoint is None and c.training.resume is None
        assert c.training.stage == 'joint_lora' and c.decoder.frozen
        v = OmegaConf.to_container(c, resolve=True)
        v.pop('projector')
        effective.append(v)
    assert effective[0] == effective[1] == effective[2]
