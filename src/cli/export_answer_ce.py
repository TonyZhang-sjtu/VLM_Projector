"""CE-only replay of an explicitly bound checkpoint; no report generation."""
import argparse
import json
import math
import os
import time
from pathlib import Path
from types import SimpleNamespace

from src.evaluation.clinical import digest, file_digest, read_jsonl, write_json
from src.evaluation.ce_runtime import DEFAULT_PROFILES, ce_replay_environment, resolve_ce_runtime, verify_runtime_packages
from src.evaluation.ce_validation import compare_ce_history, require_historical_match
from src.evaluation.significance_io import ROOT, read_json, resolve, row_identity, close

EXPORT_SOURCE_SHA256 = file_digest(Path(__file__))


def relocate_input_paths(config, local_root):
    """Map only known input/cache paths beneath the saved repository root.

    Original configs remain untouched. Content/provenance checks still apply;
    arbitrary external paths are not guessed or rewritten.
    """
    previous = Path(config['project_root'])
    changes = {}
    for section, key in (
        ('data', 'root'), ('data', 'annotation_root'), ('data', 'manifest_dir'),
        ('encoder', 'path'), ('decoder', 'path'), ('preprocessing', 'cache_dir'),
        ('training', 'feature_cache_dir'),
    ):
        value = config[section].get(key)
        if not value or not isinstance(value, str):
            continue
        path = Path(value)
        if path.is_absolute() and path.is_relative_to(previous):
            mapped = str(Path(local_root) / path.relative_to(previous))
            if value != mapped:
                changes[f'{section}.{key}'] = {'from': value, 'to': mapped}
                config[section][key] = mapped
    config['project_root'] = str(local_root)
    return changes


def prepare(entry, output, device):
    from src.cli.evaluate_checkpoint import checkpoint_config
    from src.config import validate
    run = resolve(entry['run_dir'])
    ev = run / entry.get('evaluation_subdir', 'evaluation/best_test')
    metric = read_json(ev / 'metric.json')
    checkpoint = run / metric['stage'] / 'checkpoints/best.pt'
    if file_digest(checkpoint) != metric['checkpoint_sha256']:
        raise ValueError('Checkpoint bytes do not match the original evaluation')
    if entry.get('checkpoint_sha256') != metric['checkpoint_sha256']:
        raise ValueError('Manifest must pin the selected checkpoint SHA256')
    c = checkpoint_config(checkpoint, SimpleNamespace(
        device=device, split='test', output_dir=output, metrics=['answer_ce'], repeats=0))
    remappings = relocate_input_paths(c, ROOT)
    # None means colocate the projector with the model. Setting an explicit
    # matching cuda:N would incorrectly request the two-device placement mode.
    # Preserve CPU offload; inference needs no gradients/optimizer allocations.
    device_remappings = {}
    if c.training.get('projector_device') not in (None, 'cpu'):
        device_remappings['training.projector_device'] = {
            'from': c.training.projector_device, 'to': None,
            'effective_device': device, 'reason': 'single_device_ce_inference',
        }
        c.training.projector_device = None
    validate(c)
    if c.projector.name != entry['projector'] or c.projector.target_compression_ratio != entry['ratio']:
        raise ValueError('Checkpoint projector identity differs from target row')
    rows = read_jsonl(ev / 'generation/predictions.jsonl')
    if {r['id'] for r in rows} != set(read_json(ev / 'expected_ids.json')):
        raise ValueError('Original evaluation ID mismatch')
    return c, checkpoint, metric, rows, remappings, device_remappings


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--manifest', type=Path, required=True)
    cli.add_argument('--method', required=True)
    cli.add_argument('--output-dir', type=Path, required=True)
    cli.add_argument('--device', default='cuda:0')
    cli.add_argument('--dry-run', action='store_true')
    cli.add_argument('--runtime-profiles', type=Path, default=DEFAULT_PROFILES)
    cli.add_argument('--ce-policy', choices=('strict', 'recompute'), default='strict',
                     help='recompute reports new CE scores and historical differences explicitly')
    args = cli.parse_args()
    # prepare() imports torch through checkpoint_config. Set the CUDA/CPU
    # execution environment first, including for direct python -m invocation.
    manifest = read_json(args.manifest)
    entry = next(m for m in manifest['methods'] if m['id'] == args.method)
    runtime = resolve_ce_runtime(entry, args.runtime_profiles)
    environment = ce_replay_environment(settings=runtime['environment'])
    for key in runtime['environment']:
        os.environ.pop(key, None)
    os.environ.update(environment)
    packages = verify_runtime_packages(runtime)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    c, checkpoint, metric, original, remappings, device_remappings = prepare(entry, output, args.device)
    if (output / 'summary.json').exists():
        from src.evaluation.significance_io import attach_ce
        run = {'rows': original, 'vectors': {}, 'audit': {
            'checkpoint_sha256': metric['checkpoint_sha256'],
            'test_identity_sha256': row_identity(original), 'original_metrics': metric['overall'],
            'original_ce_aggregation': read_json(resolve(entry['run_dir']) /
                entry.get('evaluation_subdir', 'evaluation/best_test') / 'generation/metrics.json')['answer_ce']}}
        attach_ce(run, output, policy=args.ce_policy)
        print('Already complete and revalidated:', output, flush=True)
        return
    import torch
    from omegaconf import OmegaConf
    from src.data.manifest import load_manifests
    from src.data.dataset import VolumeDataset
    from src.preprocessing.registry import build_preprocessor
    from src.models.multimodal import MultimodalModel, place_model
    from src.models.projectors.budget import prepare_budget
    from src.training.checkpoint import provenance, load_checkpoint
    from src.training.trainer import seed_all
    splits, manifest_summary = load_manifests(c, datasets=c.evaluation.datasets)
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False, mmap=True)
    if saved['provenance']['manifest_hashes'] != {k: v['sha256'] for k, v in manifest_summary.items()}:
        raise ValueError('Relocated data manifests differ from checkpoint provenance')
    del saved
    lookup = {s.id: s for s in splits['test']}
    if set(lookup) != {r['id'] for r in original}:
        raise ValueError('Current test manifest differs from historical full test set')
    for r in original:
        s = lookup[r['id']]
        if s.finding != r['reference'] or s.patient_id != r['patient_id']:
            raise ValueError('Historical reference/patient differs from current manifest')
    samples = [lookup[r['id']] for r in original]
    # Validate the final replay configuration and image geometry in dry-run too,
    # before claiming ready or waiting for a GPU.
    budget_preflight = prepare_budget(c, samples)
    preflight = {
        'status': 'ready', 'method_id': entry['id'], 'checkpoint': str(checkpoint),
        'checkpoint_sha256': metric['checkpoint_sha256'], 'samples': len(original),
        'test_identity_sha256': row_identity(original), 'device': args.device,
        'historical_answer_ce': metric['overall']['answer_ce'],
        'historical_training_seed': c.seed, 'generation': 'not_needed',
        'path_remappings': remappings,
        'device_remappings': device_remappings, 'budget_preflight': budget_preflight,
        'runtime_environment': runtime['environment'], 'runtime_profile': runtime,
        'package_versions': packages, 'ce_policy': args.ce_policy,
    }
    write_json(output / 'preflight.json', preflight)
    if args.dry_run:
        print(json.dumps(preflight), flush=True)
        return
    seed_all(c.seed)
    started = time.monotonic()
    preprocessor = build_preprocessor(c)
    data = VolumeDataset(samples, preprocessor)
    model = place_model(MultimodalModel(c), args.device)
    proof = provenance(c, model, preprocessor, manifest_summary)
    load_checkpoint(checkpoint, model, expected_provenance=proof)
    model.eval()
    initialized = time.monotonic()
    from tqdm.auto import tqdm
    records = []
    temporary = output / 'samples.jsonl.tmp'
    with temporary.open('w') as handle, torch.no_grad():
        for index in tqdm(range(len(data)), desc=f'CE only {entry["id"]}', mininterval=5):
            item = data[index]
            with torch.autocast('cuda', dtype=torch.bfloat16):
                result = model([item])
            numerator = float(result['loss_sum'])
            denominator = int(result['answer_tokens'])
            if not math.isfinite(numerator) or numerator < 0 or denominator <= 0:
                raise ValueError('Invalid per-sample CE numerator/denominator')
            sample = item['sample']
            record = {k: getattr(sample, k) for k in ('id', 'patient_id', 'case_id', 'cohort', 'sequence')}
            record.update(loss_sum=numerator, answer_tokens=denominator,
                          answer_ce=numerator/denominator, reference_sha256=digest(sample.finding),
                          checkpoint_sha256=metric['checkpoint_sha256'])
            records.append(record)
            handle.write(json.dumps(record, allow_nan=False) + '\n')
            handle.flush()
    total = math.fsum(r['loss_sum'] for r in records)
    tokens = sum(r['answer_tokens'] for r in records)
    actual = total / tokens
    evaluation = resolve(entry['run_dir']) / entry.get('evaluation_subdir', 'evaluation/best_test')
    historical = read_json(evaluation / 'generation/metrics.json')['answer_ce']
    close(historical['answer_ce'], metric['overall']['answer_ce'], 'Historical CE artifacts')
    comparison = compare_ce_history(records, historical)
    write_json(output / 'historical_comparison.json', comparison)
    # Recompute is an explicit new measurement, not a relaxed historical match.
    # IDs, references, checkpoint/model/data contracts and token counts stay exact.
    require_historical_match(comparison, args.ce_policy)
    temporary.replace(output / 'samples.jsonl')
    OmegaConf.save(c, output / 'replay_config.yaml')
    write_json(output / 'summary.json', dict(preflight,
        status='complete', loss_sum=total, answer_tokens=tokens, answer_ce=actual,
        score_basis='recomputed_teacher_forced_ce', historical_comparison=comparison,
        historical_aggregation=historical,
        absolute_replay_difference=actual-metric['overall']['answer_ce'],
        samples_sha256=file_digest(output/'samples.jsonl'),
        initialization_seconds=initialized-started,
        forward_seconds=time.monotonic()-initialized, total_seconds=time.monotonic()-started,
        torch_version=torch.__version__, cuda_version=torch.version.cuda,
        device_name=torch.cuda.get_device_name(args.device),
        torch_num_threads=torch.get_num_threads(), torch_interop_threads=torch.get_num_interop_threads(),
        source_sha256=EXPORT_SOURCE_SHA256))
    print(f'Completed CE-only export: {entry["id"]}, CE={actual:.12f}', flush=True)


if __name__ == '__main__':
    main()
