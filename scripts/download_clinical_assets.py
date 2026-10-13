"""Download pinned public scorer weights; never changes the Python environment.

All weights are saved under assets/models/evaluation. Optional NeuroQA text-only
files go under assets/data/NeuroQA. No metric training dataset is needed.
"""
import argparse
import hashlib
import json
from pathlib import Path
from huggingface_hub import HfApi, snapshot_download


ROOT = next((p for p in Path(__file__).resolve().parents if (p / 'pyproject.toml').is_file()), None)
if ROOT is None:
    raise RuntimeError('Could not find the project root (pyproject.toml)')
MODELS = [
    ('StanfordAIMI/GREEN-RadLlama2-7b', '832389d56bbd45a36bbd61dfa4f0be76e99f1dad', 'green'),
    ('StanfordAIMI/RRG_scorers', '6646433b3ad83a10f6e141db76d0ece44312b236', 'radgraph'),
    ('microsoft/BiomedVLP-CXR-BERT-general', '6172dbfa7c061d635a4b86761b80e324e1995496', 'cxr-bert'),
    ('Angelakeke/RaTE-NER-Deberta', '66d4b8189374f9b137e1b8633951ca4b4d758673', 'rate-ner'),
    ('FremyCompany/BioLORD-2023-C', '4bb5abd13930002136e3a5750b84474763c775c5', 'biolord'),
]


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--neuroqa', action='store_true')
    args = cli.parse_args()
    for repo, revision, name in MODELS:
        destination = ROOT / 'assets/models/evaluation' / name
        info = HfApi().model_info(repo, revision=revision, files_metadata=True, token=False)
        allow = ['radgraph-xl.tar.gz', 'README.md'] if name == 'radgraph' else None
        ignore = ['.gitattributes', 'training_args.bin', 'trainer_state.json']
        if name == 'biolord':
            ignore.append('pytorch_model.bin')  # safetensors contains the same model.
        snapshot_download(repo, revision=revision, local_dir=destination, token=False,
                          allow_patterns=allow, ignore_patterns=ignore, max_workers=3)
        files = []
        for entry in info.siblings:
            path = destination / entry.rfilename
            if not path.is_file() or (allow and entry.rfilename not in allow) or entry.rfilename in ignore:
                continue
            checksum = sha256(path)
            if entry.lfs and checksum != entry.lfs.sha256:
                raise ValueError(f'Hub LFS checksum mismatch: {path}')
            files.append({'path': entry.rfilename, 'size': path.stat().st_size, 'sha256': checksum})
        (destination / 'asset_manifest.json').write_text(json.dumps(
            {'repo_id': repo, 'revision': revision, 'gated': info.gated, 'files': sorted(files, key=lambda f:f['path'])}, indent=2))
        print(f'Verified {repo} at {revision}', flush=True)
    if args.neuroqa:
        path = ROOT / 'assets/data/NeuroQA'
        revision = 'f1bb288c4b042d1df005ebd1b5855614e25de3f2'
        snapshot_download('mhabbasiit/NeuroQA', repo_type='dataset', revision=revision,
                          allow_patterns=['README.md', 'all_public/*.json'], local_dir=path, token=False)
        files = [{'path': str(f.relative_to(path)), 'sha256': sha256(f)}
                 for f in sorted((path / 'all_public').glob('*.json'))]
        (path / 'asset_manifest.json').write_text(json.dumps(
            {'repo_id': 'mhabbasiit/NeuroQA', 'revision': revision, 'images_included': False, 'files': files}, indent=2))


if __name__ == '__main__':
    main()
