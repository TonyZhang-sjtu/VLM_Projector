"""Score saved reports with GREEN, RadGraph-XL, RaTEScore and/or BERTScore."""
import argparse
import json
import re
from pathlib import Path
from src.evaluation.clinical import (
    METRIC_KEYS, aggregate_clinical, file_digest, read_jsonl, score_predictions,
    validate_rows, write_json,
)


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--predictions', type=Path, required=True)
    cli.add_argument('--output-dir', type=Path, required=True)
    cli.add_argument('--expected-ids', type=Path, help='Independently defined JSON array; checked before model loading')
    cli.add_argument('--metrics', nargs='+', choices=list(METRIC_KEYS), default=list(METRIC_KEYS))
    cli.add_argument('--device', required=True, help='Explicit device, e.g. cuda:2 or cpu')
    cli.add_argument('--assets-dir', type=Path)
    cli.add_argument('--cache-dir', type=Path)
    cli.add_argument('--repeats', type=int, default=1000)
    cli.add_argument('--seed', type=int, default=42)
    cli.add_argument('--green-max-length', type=int, default=2048, help='Total prompt+response budget; changes scoring protocol')
    args = cli.parse_args()
    if not re.fullmatch(r'cpu|cuda:\d+', args.device):
        cli.error('--device must be cpu or cuda:N (visible CUDA index)')
    if args.repeats < 0 or args.green_max_length < 2:
        cli.error('Invalid repeats or GREEN length')
    if len(set(args.metrics)) != len(args.metrics):
        cli.error('Duplicate metric names')
    rows = read_jsonl(args.predictions)
    expected = json.loads(args.expected_ids.read_text()) if args.expected_ids else None
    validate_rows(rows, expected)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.predictions.resolve() == (args.output_dir / 'clinical_scores.jsonl').resolve():
        cli.error('Output must not replace the input predictions')

    from src.evaluation.clinical_runtime import DEFAULT_ASSETS, activate, specifications
    assets = (args.assets_dir or DEFAULT_ASSETS).resolve()
    activate(assets)
    from filelock import FileLock
    from src.evaluation.clinical_scorers import make_scorer
    from src.evaluation.metrics import evaluate_rows
    import numpy as np
    import torch
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    with FileLock(str(args.output_dir / '.scoring.lock'), timeout=0):
        provenance_path = args.output_dir / 'scoring_provenance.json'
        input_hash = file_digest(args.predictions)
        if provenance_path.exists():
            previous = json.loads(provenance_path.read_text())
            if previous.get('input_sha256') != input_hash:
                raise ValueError('Output directory belongs to another prediction file; choose a new output directory')
        provenance = {'status': 'running', 'input_path': str(args.predictions.resolve()),
                      'input_sha256': input_hash, 'samples': len(rows),
                      'coverage_scope': 'expected_ids' if expected is not None else 'input_file_only',
                      'expected_ids_sha256': file_digest(args.expected_ids) if expected is not None else None,
                      'device': args.device, 'seed': args.seed, 'metrics': args.metrics}
        write_json(provenance_path, provenance)
        # A failed rerun must not leave an old completed summary that looks current.
        for name in ('metrics.json', 'clinical_metrics.json'):
            write_json(args.output_dir / name, {'status': 'running', 'input_sha256': input_hash})
        try:
            specs = specifications(args.metrics, assets, args.device, args.green_max_length)
            for spec in specs.values():
                spec['seed'] = args.seed
                spec['tf32'] = False
                spec['device_name'] = torch.cuda.get_device_name(args.device) if args.device != 'cpu' else 'cpu'
            provenance['scorers'] = specs
            write_json(provenance_path, provenance)
            scored, execution = score_predictions(
                rows, args.output_dir, specs,
                lambda metric: make_scorer(metric, assets, args.device, args.green_max_length),
                args.cache_dir)
            clinical = aggregate_clinical(scored, args.metrics, args.repeats, args.seed)
            clinical['coverage_scope'] = provenance['coverage_scope']
            write_json(args.output_dir / 'clinical_metrics.json', clinical)
            # Keep corpus BLEU and the original ROUGE protocol, with no model loaded now.
            lexical = evaluate_rows(rows, expected if expected is not None else [r['id'] for r in rows],
                                    args.repeats, args.seed)
            for metric, label in (('rate', 'RaTEScore'), ('bertscore', 'BERTScore')):
                if metric in args.metrics:
                    lexical['semantic_metrics'].pop(label, None)
            lexical['semantic_metrics'].update(clinical['overall'])
            published_status = ('complete' if clinical['status'] == 'complete' else 'partial')
            lexical['clinical'] = clinical
            lexical['status'] = published_status if lexical['status'] == 'complete' else lexical['status']
            lexical['coverage_scope'] = provenance['coverage_scope']
            write_json(args.output_dir / 'metrics.json', lexical)
            provenance.update(status=published_status, clinical_status=clinical['status'], execution=execution)
            write_json(provenance_path, provenance)
            print(json.dumps({'status': published_status, 'clinical_status': clinical['status'],
                              'overall': clinical['overall'],
                              'execution': execution}, indent=2))
        except Exception as exc:
            provenance.update(status='failed', error_type=type(exc).__name__, error=str(exc))
            write_json(provenance_path, provenance)
            raise


if __name__ == '__main__':
    main()
