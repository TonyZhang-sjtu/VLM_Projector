"""Short reproducibility probe using the real Report Generation Trainer.

Keep the original full schedule and sampling. A diagnostic wrapper stops after
N completed updates, before validation; no production trainer code is changed.
Run in separate processes so RNG/CUDA/allocator state is independent per trial.
"""
import argparse
import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
os.environ.setdefault('OMP_NUM_THREADS', '4')
os.environ.setdefault('MKL_NUM_THREADS', '4')

import torch
from omegaconf import OmegaConf
from src.config import load_config, digest, file_hash
from src.training.trainer import Trainer


class ProbeComplete(Exception):
    pass


def tensor_hash(tensor):
    data = tensor.detach().contiguous().cpu()
    h = hashlib.sha256()
    h.update(str(data.dtype).encode())
    h.update(str(tuple(data.shape)).encode())
    h.update(data.reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def fingerprints(trainer):
    out = {}
    for group in trainer.optimizer.param_groups:
        for name, parameter in zip(group['param_names'], group['params']):
            out[name] = tensor_hash(parameter)
    return {'combined': digest(out), 'parameters': out}


def main():
    cli = argparse.ArgumentParser(__doc__)
    cli.add_argument('--source', type=Path, required=True, help='Original ratio directory')
    cli.add_argument('--output', type=Path, required=True)
    cli.add_argument('--mode', choices=['fresh', 'joint_resume'], required=True)
    cli.add_argument('--steps', type=int, default=3)
    cli.add_argument('--seed', type=int, default=42)
    cli.add_argument('--trace-first-forward', action='store_true')
    args = cli.parse_args()
    source, output = args.source.resolve(), args.output.resolve()
    if not output.is_relative_to(ROOT / 'runs/diagnostics'):
        raise ValueError('Probe output must be under project runs/diagnostics')
    if output.exists():
        raise FileExistsError(output)
    if args.steps < 2 or args.steps > 5:
        raise ValueError('This short probe allows 2 to 5 updates only')
    if args.mode == 'joint_resume' and args.seed != 42:
        raise ValueError('Resume must retain the original seed and saved RNG')
    output.mkdir(parents=True)
    config = load_config([source/'joint_lora/resolved_config.yaml', ROOT/'configs/local_paths.yaml'], [
        f'project_root={ROOT}', f'seed={args.seed}', f'training.run_dir={output}',
        'training.device=cuda:0'])
    if config.task.name == 'neuroqa' or not config.training.deterministic:
        raise ValueError('Expected deterministic Report Generation configuration')
    if config.training.max_steps is not None or config.training.train_limit is not None:
        raise ValueError('The original full training schedule/data must be retained')
    checkpoint = source/'joint_lora/checkpoints/boundary_00000154.pt'
    config.training.resume = str(checkpoint) if args.mode == 'joint_resume' else None
    info = {'status': 'initializing', 'mode': args.mode, 'seed': args.seed, 'requested_steps': args.steps,
            'source': str(source), 'source_config_sha256': file_hash(source/'joint_lora/resolved_config.yaml'),
            'checkpoint': str(checkpoint) if args.mode == 'joint_resume' else None,
            'checkpoint_sha256': file_hash(checkpoint) if args.mode == 'joint_resume' else None,
            'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'), 'python': platform.python_version(),
            'torch': str(torch.__version__), 'cuda': torch.version.cuda, 'started_unix': time.time(),
            'purpose': 'Reproducibility diagnostic, not a completed experiment or report-quality evaluation.'}
    info_path = output/'probe.json'
    info_path.write_text(json.dumps(info, indent=2)+'\n')
    trainer = Trainer(config)
    starting_step = trainer.state['global_step']
    info.update(status='running', gpu=torch.cuda.get_device_name(), initial_step=starting_step,
                original_total_steps=trainer.total_steps, schedule=trainer.plan.to_dict(),
                deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
                train_selection=trainer.state['train_selection'], val_selection=trainer.state['val_selection'],
                initial_weights=fingerprints(trainer), source_hashes=trainer.provenance['source_files'],
                initial_cuda_rng_sha256=tensor_hash(torch.cuda.get_rng_state()),
                initial_cpu_rng_sha256=tensor_hash(torch.get_rng_state()))
    info_path.write_text(json.dumps(info, indent=2)+'\n')
    print(json.dumps({'event':'probe_ready', 'mode':args.mode, 'seed':args.seed, 'start_step':starting_step,
                      'method':config.projector.name, 'ratio':config.projector.target_compression_ratio}), flush=True)
    traced_tensors, trace_meta = {}, {}
    active = {'sample': -1}
    if args.trace_first_forward:
        def remember(name, value):
            if active['sample'] == 0 and name not in traced_tensors:
                traced_tensors[name] = value.detach().cpu().clone()
        def first_input(module, args_):
            active['sample'] += 1
            if active['sample'] == 0:
                item = args_[0][0]
                trace_meta.update(sample=item['sample'].metadata(), volume_metadata=item['volume'].metadata)
                remember('input.patches', item['volume'].patches)
                remember('input.coords', item['volume'].coords)
        trainer.model.register_forward_pre_hook(first_input)
        trainer.model.encoder.register_forward_hook(lambda m,a,o: remember('encoder.features',o.features[0]))
        trainer.model.projector.register_forward_hook(lambda m,a,o: remember('projector.embeddings',o.embeddings))
        for name in ['input_adapter','local_reader','global_layers.0','final_norm','head']:
            modules = dict(trainer.model.projector.named_modules())
            if name in modules:
                modules[name].register_forward_hook(lambda m,a,o,n=name: remember('projector.'+n,o))
        trainer.model.decoder.lm.model.register_forward_hook(lambda m,a,o: remember('decoder.hidden',o.last_hidden_state))
    step_fingerprints = []
    def after_update(optimizer, arguments, keywords):
        record = {'step': trainer.state['global_step']+1, 'weights': fingerprints(trainer)}
        step_fingerprints.append(record)
        with (output/'weight_fingerprints.jsonl').open('a') as handle:
            handle.write(json.dumps(record)+'\n')
    trainer.optimizer.register_step_post_hook(after_update)
    original_sync = trainer.sync_phase
    def bounded_sync():
        if trainer.state['global_step']-starting_step >= args.steps:
            raise ProbeComplete()
        return original_sync()
    trainer.sync_phase = bounded_sync
    try:
        trainer.train()
    except ProbeComplete:
        pass
    completed = trainer.state['global_step']-starting_step
    if completed != args.steps or len(step_fingerprints) != args.steps:
        raise RuntimeError(f'Unexpected diagnostic update count: {completed}')
    # Tensor snapshots are for comparisons only and are deliberately not named
    # best.pt/last.pt, to avoid being mistaken for complete production runs.
    weights = {name:p.detach().cpu() for g in trainer.optimizer.param_groups
               for name,p in zip(g['param_names'],g['params'])}
    torch.save(weights, output/'diagnostic_trainable_tensors.pt')
    if args.trace_first_forward:
        torch.save(traced_tensors, output/'first_forward_tensors.pt')
        (output/'first_forward.json').write_text(json.dumps({
            'metadata':trace_meta,'tensor_hashes':{k:tensor_hash(v) for k,v in traced_tensors.items()}},
            indent=2,default=str)+'\n')
    records = [json.loads(line) for line in (output/'train.jsonl').read_text().splitlines()]
    info.update(status='complete', completed_steps=completed, final_step=trainer.state['global_step'],
                final_weights=step_fingerprints[-1]['weights'], finished_unix=time.time(),
                losses=[r['answer_ce'] for r in records], gradients=[r['grad_norm'] for r in records],
                phases=[r['phase'] for r in records],
                peak_allocated_gib=max(r['peak_allocated_bytes'] for r in records)/2**30)
    info_path.write_text(json.dumps(info, indent=2)+'\n')
    print(json.dumps({'event':'probe_complete', 'steps':completed, 'losses':info['losses'],
                      'final_weights':info['final_weights']['combined']}, indent=2), flush=True)


if __name__ == '__main__':
    main()
