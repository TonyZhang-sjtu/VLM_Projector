"""Evaluate a saved NeuroQA checkpoint using its own immutable data/model contract."""
import argparse
import gc
import json
from pathlib import Path

from omegaconf import OmegaConf
import torch
from tqdm.auto import tqdm

from src.config import file_hash, validate, save_config
from src.data.neuroqa_manifest import write_json
from src.data.neuroqa_preflight import prepare_data, save_data_report, ensure_same_selection
from src.evaluation.neuroqa import predict, save_predictions
from src.models.multimodal import MultimodalModel, place_model
from src.training.checkpoint import load_checkpoint, provenance
from src.training.trainer import seed_all


def main():
    cli = argparse.ArgumentParser(__doc__)
    cli.add_argument('--checkpoint', type=Path, required=True)
    cli.add_argument('--output', type=Path, required=True)
    cli.add_argument('--split', choices=['train','val','test'], default='test')
    cli.add_argument('--control', choices=['normal','text','zero','shuffle'], default='normal')
    cli.add_argument('--limit', type=int)
    cli.add_argument('--ids-file', type=Path)
    cli.add_argument('--bootstrap', type=int, default=1000)
    cli.add_argument('--invalid-sample-policy', choices=('skip', 'strict', 'include_unreviewed'))
    cli.add_argument('--expected-data-selection-file', type=Path)
    args = cli.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError('Choose an empty evaluation directory')
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False, mmap=True)
    config = OmegaConf.create(checkpoint['config'])
    if args.invalid_sample_policy:
        config.data.invalid_sample_policy = args.invalid_sample_policy
    training_scope = checkpoint.get('state', {}).get('scope', 'unknown')
    if config.task.name != 'neuroqa':
        cli.error('Checkpoint is not a NeuroQA task')
    del checkpoint
    gc.collect()
    validate(config)
    seed_all(config.seed)
    populations, data_report, pre, summary = prepare_data(config, {
        'split': args.split, 'controls': [args.control], 'ids_file': args.ids_file, 'limit': args.limit}, include_training=False)
    if args.expected_data_selection_file:
        ensure_same_selection(json.loads(args.expected_data_selection_file.read_text()), data_report,
                              'evaluation_selection_fingerprint')
    save_data_report(args.output, data_report)
    if data_report['status'] != 'passed':
        raise ValueError('VQA evaluation data checks failed: ' + '; '.join(data_report['errors']))
    samples = populations['evaluation']
    population_size = summary[args.split]['samples']
    if args.split == 'train':
        # The train manifest stores unique QA, while evaluation enumerates views.
        from src.data.neuroqa_manifest import load_neuroqa
        source, _ = load_neuroqa(config, check_paths=False)
        population_size = sum(len(s.row['allowed_modalities']) for s in source['train'])
    preflight = data_report['geometry']
    model = place_model(MultimodalModel(config), config.training.device)
    proof = provenance(config, model, pre, summary)
    load_checkpoint(args.checkpoint, model, expected_provenance=proof)
    control = None if args.control == 'normal' else args.control
    with tqdm(total=len(samples), desc=f'{args.split.capitalize()} ({args.control})',
              mininterval=2, dynamic_ncols=True,
              bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [ETA {remaining}]') as progress:
        rows = predict(model, samples, pre, control=control, compute_ce=control is None,
                       progress=lambda i, n: progress.update(i - progress.n))
    checkpoint_hash = file_hash(args.checkpoint)
    for row in rows:
        row.update(checkpoint_sha256=checkpoint_hash, preprocessing_fingerprint=pre.version,
                   prompt_fingerprint=model.decoder.task.fingerprint, contract_id=proof['contract_id'])
    metrics = save_predictions(args.output, samples, rows, args.bootstrap, config.seed)
    filtered = bool(data_report['skipped']['evaluation']) or not data_report['qc_ready_for_full_training'] or training_scope=='filtered_training'
    scope = ('subset_smoke' if training_scope in ('smoke', 'unknown') or args.limit is not None or args.ids_file else
             'unreviewed_evaluation' if config.data.get('invalid_sample_policy') == 'include_unreviewed' or training_scope == 'unreviewed_training' else
             'filtered_evaluation' if filtered else 'full_split')
    metrics.update(split=args.split, population_size=population_size, evaluated_subset=len(samples),
                   training_scope=training_scope,
                   data_policy=config.data.get('invalid_sample_policy', 'strict'),
                   skipped_views=len(data_report['skipped']['evaluation']),
                   evaluation_selection_fingerprint=data_report['evaluation_selection_fingerprint'],
                   evaluation_coverage='full_split' if len(samples)==population_size else 'subset',
                   scope=scope,
                   control=args.control)
    write_json(args.output / 'metrics.json', metrics)
    write_json(args.output / 'provenance.json', proof)
    write_json(args.output / 'preflight.json', preflight)
    save_config(config, args.output / 'config.yaml')
    print(json.dumps(metrics['overall'], indent=2), flush=True)


if __name__ == '__main__':
    main()
