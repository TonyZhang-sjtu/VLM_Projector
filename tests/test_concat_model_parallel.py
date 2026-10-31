"""Two-GPU Concat placement, gradient flow, and launcher contracts."""
from argparse import Namespace
import json
import os
import subprocess
import sys

import pytest
import torch
from torch import nn
from omegaconf import OmegaConf

from src.cli.evaluate_checkpoint import checkpoint_config
from src.config import ROOT, load_config
from src.models.device_transfer import move_tensor
from src.models.multimodal import place_model
from src.models.projectors.budget import resolve_budget
from src.models.projectors.registry import build_projector
from src.training.optim import clip_grad_norm_across_devices


def test_projector_device_config_contract():
    files = [ROOT / 'configs/projector/concat3d.yaml']
    valid = load_config(files, ['training.projector_device=cuda:1'])
    assert valid.training.device == 'cuda:0'
    for overrides in (
        ['training.projector_device=cuda:0'],
        ['training.projector_device=cpu'],
        ['training.projector_device=cuda:1', 'training.distributed=ddp'],
    ):
        with pytest.raises(ValueError, match='projector_device|Concat3D'):
            load_config(files, overrides)
    with pytest.raises(ValueError, match='Concat3D'):
        load_config(overrides=['training.projector_device=cuda:1'])


def test_checkpoint_evaluation_keeps_second_gpu(tmp_path):
    c = load_config([ROOT / 'configs/projector/concat3d.yaml'],
                    ['training.projector_device=cuda:1'])
    path = tmp_path / 'checkpoint.pt'
    torch.save({'format_version': 1, 'stage': 'a',
                'config': OmegaConf.to_container(c, resolve=True)}, path)
    args = Namespace(device='cuda:0', output_dir=tmp_path, split='test',
                     metrics=['answer_ce'], repeats=0)
    restored = checkpoint_config(path, args)
    assert restored.training.device == 'cuda:0'
    assert restored.training.projector_device == 'cuda:1'


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason='Two CUDA GPUs required')
def test_cross_gpu_transfer_preserves_values_and_gradients():
    source = torch.arange(2048., device='cuda:0', requires_grad=True)
    second = move_tensor(source, 'cuda:1')
    restored = move_tensor(second, 'cuda:0')
    torch.testing.assert_close(restored, source)
    restored.square().sum().backward()
    torch.testing.assert_close(source.grad, 2 * source.detach())


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason='Two CUDA GPUs required')
def test_global_clip_across_devices_uses_one_coefficient():
    first = nn.Parameter(torch.tensor([3., 4.], device='cuda:0'))
    second = nn.Parameter(torch.tensor([0., 12.], device='cuda:1'))
    first.grad = first.detach().clone()
    second.grad = second.detach().clone()
    norm = clip_grad_norm_across_devices([first, second], 1.)
    assert norm == pytest.approx(13.)
    torch.testing.assert_close(first.grad.cpu(), torch.tensor([3/13, 4/13]), rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(second.grad.cpu(), torch.tensor([0., 12/13]), rtol=1e-6, atol=1e-6)


def test_two_gpu_launcher_requires_two_distinct_physical_ids(tmp_path):
    stub = tmp_path / 'python'
    stub.write_text('#!' + sys.executable + '\nimport json,os,sys\n'
                    'print(json.dumps({"gpu":os.environ["CUDA_VISIBLE_DEVICES"],"args":sys.argv[1:]}))\n')
    stub.chmod(0o700)
    script = ROOT / 'scripts/Exp1026/run_concat3d_r128_r256.sh'
    for ids in ('2,3', '0,1'):
        env = dict(os.environ, PROJECTOR_PYTHON=str(stub), CUDA_VISIBLE_DEVICES=ids)
        env.pop('PROJECTOR_LOCAL_CONFIG', None)
        result = subprocess.run(['bash', str(script), '--dry-run'], env=env,
                                capture_output=True, text=True, check=True)
        call = json.loads(result.stdout)
        assert call['gpu'] == ids
        args = call['args']
        assert args[args.index('--ratios') + 1:args.index('--output-root')] == ['128', '256']
        assert 'training.projector_device=cuda:1' in args
        assert 'training.log_optimizer_diagnostics=false' in args
        assert args[args.index('--device') + 1] == 'cuda:0'
    for ids in ('3', '2,2', '0,1,2'):
        env = dict(os.environ, PROJECTOR_PYTHON=str(stub), CUDA_VISIBLE_DEVICES=ids)
        result = subprocess.run(['bash', str(script), '--dry-run'], env=env,
                                capture_output=True, text=True)
        assert result.returncode == 2
        assert 'two distinct physical GPU' in result.stderr


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason='Two CUDA GPUs required')
@pytest.mark.parametrize('ratio', [128, 256])
def test_two_gpu_concat_forward_backward_optimizer(ratio):
    c = load_config([ROOT / 'configs/projector/concat3d.yaml'], [
        f'projector.target_compression_ratio={ratio}',
        'projector.hidden_size=32', 'training.projector_device=cuda:1'])
    resolve_budget(c.projector, (39, 15, 15), (156., 240., 240.))
    encoder = nn.Linear(8, 8).to('cuda:0')
    decoder = nn.Linear(16, 1).to('cuda:0').requires_grad_(False)
    projector = build_projector(c, 8, 16)
    model = nn.Module()
    model.config = c
    model.encoder, model.decoder, model.projector = encoder, decoder, projector
    place_model(model, 'cuda:0')
    assert next(model.encoder.parameters()).device == torch.device('cuda:0')
    assert next(model.decoder.parameters()).device == torch.device('cuda:0')
    assert next(model.projector.parameters()).device == torch.device('cuda:1')
    x = torch.randn(8775, 8, device='cuda:0')
    coords = torch.cartesian_prod(*(torch.arange(n) for n in (39, 15, 15)))
    meta = {'grid_dhw': [39, 15, 15], 'shape_dhw': [156, 240, 240],
            'bounded_before_pad': {'spacing_xyz': [1., 1., 1.]},
            'array_zyx_to_dhw': [0, 1, 2], 'id': 'case-a'}
    optimizer = torch.optim.AdamW([*encoder.parameters(), *projector.parameters()], lr=1e-3)
    with torch.autocast('cuda', dtype=torch.bfloat16):
        features = encoder(x)
        projected = projector(features, coords, meta).embeddings
        prediction = decoder(move_tensor(projected, 'cuda:0'))
        loss = prediction.float().square().mean()
    loss.backward()
    assert projected.device == torch.device('cuda:1')
    assert all(p.grad is not None and p.grad.device == p.device for p in encoder.parameters())
    assert all(p.grad is not None and p.grad.device == p.device for p in projector.parameters())
    assert all(p.grad is None for p in decoder.parameters())
    clip_grad_norm_across_devices([*encoder.parameters(), *projector.parameters()], 1.)
    optimizer.step()
    assert optimizer.state[next(projector.parameters())]['exp_avg'].device == torch.device('cuda:1')
    assert optimizer.state[next(encoder.parameters())]['exp_avg'].device == torch.device('cuda:0')
