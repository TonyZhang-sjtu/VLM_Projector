#!/usr/bin/env python3
"""Full-width NeuroVFM + real frozen Llama smoke matrix on an explicitly selected GPU.

Each configuration performs a balanced T1/T2 accumulation window, AdamW update,
finite/nonzero projector and visual-LoRA gradient checks, greedy QA decoding,
and checkpoint reload equality. Concat can run its full FP32 head/AdamW on CPU.
This is an engineering test on training QA, not a benchmark accuracy result.
"""
import argparse
from collections import Counter
import gc
import json
from pathlib import Path
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
import torch

from src.config import load_config, digest
from src.data.neuroqa_manifest import balanced_epoch, load_neuroqa, write_json
from src.models.multimodal import MultimodalModel, place_model
from src.models.projectors.budget import resolve_budget
from src.models.projectors.registry import build_projector
from src.preprocessing.registry import build_preprocessor
from src.training.checkpoint import load_checkpoint, save_checkpoint
from src.training.optim import build_optimizer, build_scheduler, clip_grad_norm_across_devices
from src.training.stages import assert_stage
from src.training.trainer import seed_all

METHODS = ['avgpool3d', 'convpool3d', 'mixer3d', 'local_packer3d', 'attention_pool',
           'perceiver_nv', 'perceiver_flamingo', 'adaptive_merge3d', 'token_select3d', 'concat3d', 'bindformer3d']


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--methods', nargs='+', default=METHODS)
    p.add_argument('--config', default='configs/vqa/neuroqa/mixed.yaml')
    p.add_argument('--ratios', nargs='+', type=int, default=[16,32,64,128,256])
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--concat-device', default='cpu')
    p.add_argument('--case', default='BraTS-MEN-00380-000')
    p.add_argument('--keep-checkpoints', action='store_true')
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    seed_all(42)
    initial = load_config([ROOT / args.config], ['training.stage=b_lora'])
    resolve_budget(initial.projector, [49,15,13], [196,240,208])
    samples = load_neuroqa(initial, False)[0]['train']
    pairs = [s for s in samples if s.case_id == args.case and s.row['question_type'] in ('YN','MCQ')
             and len(s.row['allowed_modalities']) == 2][:2]
    if len(pairs) != 2:
        raise ValueError('Smoke requires two distinct dual-eligible training QA in a completed case')
    order = balanced_epoch(pairs, 42, 0)
    preprocessor = build_preprocessor(initial)
    items = [{'sample': s, 'volume': preprocessor(s)} for s in order]
    model = place_model(MultimodalModel(initial), 'cuda:0')
    original_lora = {n: p.detach().cpu().clone() for n,p in model.encoder.named_parameters() if p.requires_grad}
    proof = {'contract_id': digest({'task': 'neuroqa_smoke', 'preprocessing': preprocessor.version}),
             'preprocessing': preprocessor.version, 'note': 'engineering smoke on training cases only'}
    report = []
    for method in args.methods:
        for ratio in ([1] if method == 'mlp2x_gelu' else args.ratios):
            label = f'{method}_r{ratio}'
            start = time.monotonic()
            print(f'START {label}', flush=True)
            overrides = [f'projector.name={method}', f'projector.target_compression_ratio={ratio}', 'training.stage=b_lora']
            if method == 'concat3d':
                overrides.append(f'training.projector_device={args.concat_device}')
            if method == 'bindformer3d':
                # Report-training concept banks may expose QA holdout patients.
                overrides.append('projector.params.knowledge_mode=none')
            config = load_config([ROOT / args.config], overrides)
            budget = resolve_budget(config.projector, [49,15,13], [196,240,208])
            seed_all(42)
            with torch.no_grad():
                for n,p in model.encoder.named_parameters():
                    if n in original_lora:
                        p.copy_(original_lora[n].to(p.device))
            del model.projector
            gc.collect()
            torch.cuda.empty_cache()
            model.config = config
            model.decoder.config = config
            model.encoder.config = config
            model.projector = build_projector(config, 768, 4096).to(args.concat_device if method == 'concat3d' else 'cuda:0')
            optimizer = build_optimizer(model, config)
            scheduler = build_scheduler(optimizer, 2, 0.)
            trainable = assert_stage(model)
            model.train()
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.reset_peak_memory_stats()
            losses, intermediate = [], []
            for item in items:
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    result = model([item])
                    loss = result['example_loss_sum'] / len(items)
                if not torch.isfinite(loss):
                    raise FloatingPointError(label)
                loss.backward()
                losses.append(float(result['example_loss_sum'].detach()))
                intermediate.append({'id': item['sample'].id, 'answer_tokens': result['answer_tokens'],
                    'encoder_tokens': result['encoder_token_counts'], 'projector_tokens': result['projector_token_counts'],
                    'prefix_and_answer_lengths': result['lengths']})
                assert result['encoder_token_counts'] == [9555]
                assert result['projector_token_counts'] == [budget['expected_output_tokens']]
            norms = {}
            for component in ('projector','encoder'):
                params = [p for p in getattr(model, component).parameters() if p.grad is not None]
                if not params or not all(torch.isfinite(p.grad).all() for p in params):
                    raise AssertionError(f'{label}: missing/nonfinite {component} gradient')
                norms[component] = sum(float(p.grad.norm()) for p in params)
                assert norms[component] > 0
            assert all(p.grad is None and not p.requires_grad for p in model.decoder.parameters())
            changed = next(p for p in model.projector.parameters() if p.grad is not None and p.grad.abs().sum() > 0)
            before = changed.detach().reshape(-1)[:1024].clone()
            clip_grad_norm_across_devices([p for g in optimizer.param_groups for p in g['params']], 1.)
            optimizer.step()
            scheduler.step()
            assert not torch.equal(before, changed.detach().reshape(-1)[:1024])
            optimizer.zero_grad(set_to_none=True)
            model.eval()
            with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
                after = float(model([items[0]])['example_loss_sum'])
                prediction = model.generate([items[0]], max_new_tokens=8)[0]
            directory = args.output / label
            directory.mkdir()
            saved = directory / 'checkpoint.pt'
            save_checkpoint(saved, model, optimizer, scheduler, {'global_step':1}, proof)
            with torch.no_grad():
                changed.reshape(-1)[:1024].add_(1)
            load_checkpoint(saved, model, expected_provenance=proof)
            with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
                reloaded = float(model([items[0]])['example_loss_sum'])
                restored_prediction = model.generate([items[0]], max_new_tokens=8)[0]
            if after != reloaded or prediction['token_ids'] != restored_prediction['token_ids']:
                raise AssertionError(f'{label}: checkpoint reload changed CE/generation')
            result = {'configuration': label, 'status': 'passed', 'budget': budget,
                       'modalities': dict(Counter(s.sequence for s in order)), 'losses': losses,
                       'gradient_norms': norms, 'trainable_counts': {k:v['parameters'] for k,v in trainable.items()},
                       'intermediates': intermediate, 'reload_ce': reloaded,
                       'prediction': prediction, 'reload_equal': True,
                       'peak_cuda_bytes': torch.cuda.max_memory_allocated(), 'seconds': time.monotonic()-start,
                       'projector_device': str(next(model.projector.parameters()).device),
                       'decoder_frozen': True, 'vision_lora_updated': True, 'preprocessing_version': preprocessor.version}
            write_json(directory / 'result.json', result)
            report.append(result)
            write_json(args.output / 'report.json', report)
            print(json.dumps({'configuration':label, 'status':'passed','seconds':result['seconds'], 'peak_GiB':result['peak_cuda_bytes']/2**30}), flush=True)
            if not args.keep_checkpoints:
                saved.unlink()  # Temporary round-trip artifact; results/proof are retained.
            del optimizer, scheduler, changed, result
            gc.collect()
            torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
