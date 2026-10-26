"""Compare server-local asset bytes without publishing paths or patient IDs.

No model is loaded. Directory roots may differ; relative layout and bytes must
match. Cache/download bookkeeping is excluded, but weights and annotations are
read in full. Output contains only group totals and aggregate SHA256 digests.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
DEFAULTS = {
    'manifests': 'assets/data/RadGenome-Brain/training_manifests/single_sequence_findings_coref_v2',
    'brats_images': 'assets/data/BraTS',
    'encoder': 'assets/models/neurovfm-encoder',
    'decoder': 'assets/models/llama3.1-8b-instruct',
    'bindformer': 'assets/knowledge/bindformer_brain_candidates_v2',
    'report_facts': 'assets/knowledge/report_facts_v1',
    'report_output': 'assets/knowledge/report_output_v1',
    'evaluation': 'assets/models/evaluation',
}
SKIP_DIRS = {'.git', '.cache', '__pycache__', '.locks', '.pytest_cache'}
EVALUATION_GENERATED = {'hub', 'modules', 'radgraph/runtime', 'radgraph/radgraph-xl'}


def fingerprint(directory, images_only=False, excluded_dirs=()):
    directory = Path(directory).resolve(strict=True)
    if not directory.is_dir():
        raise ValueError('Asset group must be a directory')
    records = []
    for current, dirs, files in os.walk(directory, followlinks=True):
        current = Path(current)
        dirs[:] = sorted(d for d in dirs if d not in SKIP_DIRS
                         and (current / d).relative_to(directory).as_posix() not in excluded_dirs)
        # Follow legitimate local asset symlinks, but reject directory cycles.
        ancestors = {current.resolve(), *(p.resolve() for p in current.parents)}
        for name in dirs:
            if (current / name).resolve() in ancestors:
                raise ValueError('Asset directory contains a symlink cycle')
        for name in sorted(files):
            if name.startswith('.') or name.endswith(('.lock', '.pyc', '.pyo', '.log', '.tmp')):
                continue
            if images_only and not name.endswith(('.nii', '.nii.gz')):
                continue
            path = current / name
            before = path.stat()
            digest = hashlib.sha256()
            with path.open('rb') as stream:
                for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
                    digest.update(block)
            after = path.stat()
            if (before.st_size, before.st_mtime_ns, before.st_ino) != (after.st_size, after.st_mtime_ns, after.st_ino):
                raise RuntimeError('Asset changed while hashing; retry when idle')
            records.append([path.relative_to(directory).as_posix(), before.st_size, digest.hexdigest()])
    if not records:
        raise ValueError('Asset group is empty')
    records.sort()
    encoded = json.dumps(records, ensure_ascii=True, separators=(',', ':')).encode()
    return {'files': len(records), 'bytes': sum(row[1] for row in records),
            'sha256': hashlib.sha256(encoded).hexdigest()}


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--map', action='append', default=[], metavar='GROUP=/local/path',
                     help='Override a default group location; repeat for each changed root')
    cli.add_argument('--groups', nargs='+', choices=tuple(DEFAULTS), default=list(DEFAULTS))
    cli.add_argument('--output', type=Path, required=True, help='New JSON file, normally under .tmp/')
    cli.add_argument('--reference', type=Path, help='Compare selected groups against a source JSON')
    args = cli.parse_args()
    if args.output.exists():
        cli.error('Output already exists; use a new evidence file')
    paths = {name: ROOT / path for name, path in DEFAULTS.items()}
    for item in args.map:
        name, sep, value = item.partition('=')
        if not sep or name not in paths or not value:
            cli.error('Use --map GROUP=/local/path with a known group')
        paths[name] = Path(value).expanduser()
    result = {'format': 'projector_asset_fingerprint_v1', 'groups': {}}
    for name in args.groups:
        print(f'Hashing {name}...', file=sys.stderr, flush=True)
        result['groups'][name] = fingerprint(
            paths[name], images_only=name == 'brats_images',
            excluded_dirs=EVALUATION_GENERATED if name == 'evaluation' else ())
        print(f'{name}: {result["groups"][name]["files"]} files', file=sys.stderr, flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    if args.reference:
        expected = json.loads(args.reference.read_text())
        if expected.get('format') != result['format']:
            raise SystemExit('Unsupported reference format')
        different = [name for name in args.groups
                     if result['groups'][name] != expected['groups'].get(name)]
        if different:
            raise SystemExit('Asset mismatch: ' + ', '.join(different))
        print('Selected asset groups match the reference.')


if __name__ == '__main__':
    main()
