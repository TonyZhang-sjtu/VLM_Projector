"""CPU joint-training integration with tiny backbones, real grids and task losses.

These tests exercise the production model/decoder/optimizer/checkpoint interfaces;
they do not claim to run the pretrained NeuroVFM + 8B CUDA stack.
"""
from contextlib import nullcontext
import copy
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from transformers import AutoTokenizer, LlamaConfig, LlamaForCausalLM

from src.config import ROOT, load_config, path_of
from src.data.manifest import load_manifests
from src.data.neuroqa_manifest import load_neuroqa
from src.models.decoders.llama import LlamaDecoder
from src.models.encoders.neurovfm import NeuroVFMEncoder
from src.models.encoders.base import EncoderOutput
from src.models.multimodal import MultimodalModel
from src.models.projectors.registry import build_projector
from src.models.projectors.anchorread3d_ablation.budget import METHODS
from src.training.checkpoint import save_checkpoint, load_checkpoint
from src.training.optim import build_optimizer, build_joint_scheduler
from src.training.schedule import resolve_joint_schedule
from src.training.stages import assert_stage
from test_anchorread3d_ablation import SETTINGS, real_config, metadata


class TinyAdaptation(nn.Module):
    def __init__(self):
        super().__init__()
        self.base = nn.Linear(8, 8)
        self.lora_A = nn.Parameter(torch.randn(2, 8) * .1)
        self.lora_B = nn.Parameter(torch.zeros(8, 2))

    def forward(self, x):
        return self.base(x) + 2 * (x @ self.lora_A.T @ self.lora_B.T)


class TinyEncoder(NeuroVFMEncoder):
    def __init__(self):
        nn.Module.__init__(self)
        self.stage, self.phase = 'joint_lora', 'projector_warmup'
        self.core = TinyAdaptation()
        self.lora_targets = ['test_adapter']
        self.set_phase(self.phase)

    def forward(self, volumes):
        with torch.no_grad() if self.vision_frozen else nullcontext():
            features = [self.core(v.patches) for v in volumes]
        return EncoderOutput(features, [v.coords for v in volumes], [v.metadata for v in volumes])


@pytest.fixture(scope='module')
def task_assets():
    base = load_config()
    tokenizer = AutoTokenizer.from_pretrained(path_of(base, base.decoder.path), local_files_only=True)
    rg = load_manifests(base)[0]['train'][0]
    vqa_config = load_config([ROOT/'configs/vqa/neuroqa/mixed_v2.yaml'])
    qa = load_neuroqa(vqa_config, False)[0]['train'][0]
    return tokenizer, {'rg': rg, 'vqa': qa}


@pytest.mark.parametrize('name', METHODS)
@pytest.mark.parametrize('setting', SETTINGS)
def test_joint_updates_task_loss_generation_and_checkpoint_resume(name, setting, task_assets, tmp_path):
    torch.manual_seed(42)
    task, _, grid, extent, total, *_ = setting
    c = real_config(name, setting)
    recipe = 'training.yaml' if task == 'rg' else 'vqa_training.yaml'
    task_recipe = load_config([ROOT/'configs/experiments/anchorread3d_ablation'/recipe])
    c.training = copy.deepcopy(task_recipe.training)
    # Short-test schedule only; production keeps RG 77/77 and VQA 150/150.
    c.training.projector_warmup_steps = c.training.joint_warmup_steps = 2
    c.training.max_steps = 6
    c.decoder.gradient_checkpointing = False
    if task == 'vqa':
        c.task.name, c.task.prompt_version = 'neuroqa', 'neuroqa_single_volume_v1'
    lm = LlamaForCausalLM(LlamaConfig(vocab_size=128256, hidden_size=16, intermediate_size=32,
        num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2,
        max_position_embeddings=2048, pad_token_id=128004, eos_token_id=128009))
    model = MultimodalModel.__new__(MultimodalModel)
    nn.Module.__init__(model)
    model.config, model.encoder = c, TinyEncoder()
    model.decoder = LlamaDecoder(c, lm, task_assets[0])
    model.projector = build_projector(c, 8, 16)
    sample = task_assets[1][task]
    meta = metadata(grid, extent)
    meta['sequence'] = sample.sequence
    volume = SimpleNamespace(patches=torch.randn(torch.tensor(grid).prod().item(), 8),
                             coords=torch.cartesian_prod(*(torch.arange(n) for n in grid)), metadata=meta)
    items = [{'sample': sample, 'volume': volume}]
    plan = resolve_joint_schedule(c.training, 6)
    optimizer = build_optimizer(model, c)
    scheduler = build_joint_scheduler(optimizer, plan)
    initial_projector = {n: p.detach().clone() for n, p in model.projector.named_parameters()}
    initial_lora = model.encoder.core.lora_B.detach().clone()
    initial_base = model.encoder.core.base.weight.detach().clone()
    for step in range(6):
        model.encoder.set_phase(plan.phase(step))
        assert_stage(model)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast('cpu', dtype=torch.bfloat16):
            result = model(items)
            loss = result['example_loss_sum'] if task == 'vqa' else result['loss_sum'] / result['answer_tokens']
        assert torch.isfinite(loss) and result['projector_token_counts'] == [total]
        loss.backward()
        assert all(p.grad is None for p in model.decoder.parameters())
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.projector.parameters())
        if model.encoder.vision_frozen:
            assert model.encoder.core.lora_B.grad is None
        else:
            assert model.encoder.core.lora_B.grad is not None and model.encoder.core.lora_B.grad.abs().sum() > 0
        optimizer.step(); scheduler.step()
        if step == 1:
            torch.testing.assert_close(model.encoder.core.lora_B, initial_lora, rtol=0, atol=0)
    assert any(not torch.equal(p, initial_projector[n]) for n, p in model.projector.named_parameters())
    assert not torch.equal(model.encoder.core.lora_B, initial_lora)
    torch.testing.assert_close(model.encoder.core.base.weight, initial_base, rtol=0, atol=0)
    state = dict(global_step=6, phase=plan.phase(6), schedule=plan.to_dict())
    model.encoder.set_phase(state['phase'])
    saved = tmp_path/'checkpoint.pt'
    save_checkpoint(saved, model, optimizer, scheduler, state, {'contract_id': 'cpu_integration'})
    restored = copy.deepcopy(model)
    opt2 = build_optimizer(restored, c)
    sch2 = build_joint_scheduler(opt2, plan)
    load_checkpoint(saved, restored, 'resume', opt2, sch2, expected_provenance={'contract_id': 'cpu_integration'})
    assert sch2.last_epoch == 6
    model.eval(); restored.eval()
    with torch.no_grad(), torch.autocast('cpu', dtype=torch.bfloat16):
        torch.testing.assert_close(model(items)['loss_sum'], restored(items)['loss_sum'], rtol=0, atol=0)
    generated = restored.generate(items, max_new_tokens=1)
    assert generated[0]['projector_token_count'] == total
    assert generated[0]['projector_name'] == name


@pytest.mark.parametrize('name', METHODS)
@pytest.mark.parametrize('setting', SETTINGS)
def test_production_width_projector_optimizer_step(name, setting):
    """Real 768 -> 4096 widths/counts in CPU FP32; AVX2 cannot accelerate BF16.

    BF16 math-attention/backward is covered for every budget in the mechanism
    tests and the tiny-backbone integration above. Production remains BF16.
    """
    c = real_config(name, setting, small=False)
    _, _, grid, extent, total, *_ = setting
    model = build_projector(c, 768, 4096)
    x = torch.randn(torch.tensor(grid).prod().item(), 768, requires_grad=True)
    coords = torch.cartesian_prod(*(torch.arange(n) for n in grid))
    out = model(x, coords, metadata(grid, extent))
    loss = out.embeddings.square().mean()
    assert out.embeddings.shape == (total, 4096) and torch.isfinite(loss)
    loss.backward()
    assert x.grad is not None and torch.isfinite(x.grad).all() and x.grad.abs().sum() > 0
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    tracked = next(model.head.parameters())
    before = tracked.detach().clone()
    torch.optim.AdamW(model.parameters(), lr=2e-4).step()
    assert not torch.equal(before, tracked)
