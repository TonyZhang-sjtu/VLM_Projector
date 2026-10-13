import json
from pathlib import Path
import shutil
from src.cli.common import parser, config_from
from src.config import file_hash, path_of
from src.data.manifest import load_manifests
from src.preprocessing.registry import build_preprocessor
from src.preprocessing.qa import lesion_geometry_qa
from src.preprocessing.artifact import validate_artifact_contract
from tqdm.auto import tqdm


def main():
    cli = parser('Preprocess complete volumes and optionally run train/val geometry QA')
    cli.add_argument('--split', choices=['train','val','test'], default='train')
    cli.add_argument('--limit', type=int)
    cli.add_argument('--representative', action='store_true')
    cli.add_argument('--qa', action='store_true')
    cli.add_argument('--reuse-artifact', help='Verify and reuse a completed preprocessing artifact')
    cli.add_argument('--output', required=True)
    args = cli.parse_args()
    if args.qa and args.split == 'test':
        cli.error('Use train/val for preprocessing QA')
    config = config_from(args)
    split_samples, manifest_summary = load_manifests(config)
    samples = split_samples[args.split]
    preprocessor = build_preprocessor(config)
    output = Path(args.output)
    if args.reuse_artifact:
        if args.limit or args.representative or args.qa:
            cli.error('--reuse-artifact requires a complete split without --limit, --representative or --qa')
        verification = validate_artifact_contract(
            config, preprocessor, samples, manifest_summary, args.split,
            path_of(config, args.reuse_artifact))
        source = Path(verification.pop('source_manifest'))
        output.mkdir(parents=True, exist_ok=True)
        target = output/'preprocessing_manifest.jsonl'
        if target.resolve() != source.resolve():
            if target.exists() and file_hash(target) != verification['source_manifest_sha256']:
                raise FileExistsError(f'Refusing to replace a different preprocessing manifest: {target}')
            if not target.exists():
                shutil.copy2(source, target)
        source_qa = source.parent/'qa_summary.json'
        if source_qa.exists() and not (output/'qa_summary.json').exists():
            shutil.copy2(source_qa, output/'qa_summary.json')
        verification.update(status='reused', output_manifest=str(target),
                            cache_enabled=True, transforms_recomputed=0)
        (output/'reuse_summary.json').write_text(json.dumps(verification, indent=2))
        print(json.dumps(verification, indent=2))
        return
    if args.representative:
        selected = {}
        for sample in tqdm(samples,desc=f'preprocess {args.split}',unit='volume',dynamic_ncols=True):
            selected.setdefault((sample.cohort,sample.sequence),sample)
        samples = list(selected.values())
    if args.limit:
        samples = samples[:args.limit]
    output.mkdir(parents=True,exist_ok=True)
    qa = []
    # The completion summary distinguishes a completed run from partial JSONL output.
    with (output/'preprocessing_manifest.jsonl').open('w') as handle:
        for sample in tqdm(samples,desc=f'preprocess {args.split}',unit='volume',dynamic_ncols=True):
            volume = preprocessor(sample)
            handle.write(json.dumps({'id':sample.id, **volume.metadata})+'\n')
            handle.flush()
            if args.qa:
                qa.append(lesion_geometry_qa(sample,volume))
    (output/'qa_summary.json').write_text(json.dumps({'status':'complete','samples':len(samples),
        'datasets':manifest_summary[args.split]['datasets'],
        'dataset_counts':manifest_summary[args.split]['dataset_counts'],
        'representative_subset':args.representative,'qa':qa},indent=2))


if __name__ == '__main__':
    main()
