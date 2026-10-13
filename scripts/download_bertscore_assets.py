"""Download the pinned BERTScore encoder only; no environment changes or login."""
import hashlib
import sys
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

ROOT = next(p for p in Path(__file__).resolve().parents if (p / 'pyproject.toml').is_file())
sys.path.insert(0, str(ROOT))
from src.evaluation.clinical import file_digest, write_json

REPO = 'google-bert/bert-base-uncased'
REVISION = '86b5e0934494bd15c9632b12f734a8a67f723594'
FILES = ['config.json', 'model.safetensors', 'tokenizer.json', 'tokenizer_config.json',
         'vocab.txt', 'README.md']


def main():
    destination = ROOT / 'assets/models/evaluation/bert-score'
    info = HfApi().model_info(REPO, revision=REVISION, files_metadata=True, token=False)
    if info.sha != REVISION:
        raise ValueError('Unexpected model revision')
    entries = {entry.rfilename: entry for entry in info.siblings}
    if set(FILES) - entries.keys():
        raise ValueError('Pinned model is missing required files')
    snapshot_download(REPO, revision=REVISION, local_dir=destination, token=False,
                      allow_patterns=FILES, max_workers=3)
    files = []
    for name in FILES:
        path, entry = destination / name, entries[name]
        checksum = file_digest(path)
        if path.stat().st_size != entry.size:
            raise ValueError(f'Hub size mismatch: {name}')
        if entry.lfs:
            if checksum != entry.lfs.sha256:
                raise ValueError(f'Hub LFS checksum mismatch: {name}')
        else:
            payload = path.read_bytes()
            git_hash = hashlib.sha1(f'blob {len(payload)}\0'.encode() + payload).hexdigest()
            if git_hash != entry.blob_id:
                raise ValueError(f'Hub blob checksum mismatch: {name}')
        files.append({'path': name, 'size': path.stat().st_size, 'sha256': checksum})
    write_json(destination / 'asset_manifest.json', {
        'repo_id': REPO, 'revision': REVISION, 'gated': info.gated,
        'files': sorted(files, key=lambda entry: entry['path']),
    })
    print(f'Verified {REPO}@{REVISION}: {sum(f["size"] for f in files)} bytes in {destination}')


if __name__ == '__main__':
    main()
