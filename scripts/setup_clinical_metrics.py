"""Prepare project-local scorer dependencies without changing the training environment."""
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = next((p for p in Path(__file__).resolve().parents if (p / 'pyproject.toml').is_file()), None)
if ROOT is None:
    raise RuntimeError('Could not find the project root (pyproject.toml)')
REPOSITORIES = [
    ('GREEN', 'https://github.com/Stanford-AIMI/GREEN.git', '29567040fba813e1d62503954ec234e056807d05'),
    ('radgraph', 'https://github.com/Stanford-AIMI/radgraph.git', '87f11a1ff4d2046a838be5f0243857d93780ddec'),
    ('RaTEScore', 'https://github.com/MAGIC-AI4Med/RaTEScore.git', 'a970406b6f4155f7c7a9cb0d29b4fe94437e3361'),
]


def main():
    core = {n: metadata.version(n) for n in ('torch', 'transformers', 'peft', 'accelerate', 'numpy')}
    if sys.version_info[:2] != (3, 10) or core['transformers'] != '4.51.3' or core['torch'].split('+')[0] != '2.5.0':
        raise RuntimeError('This lock was tested with projector Python 3.10, Torch 2.5.0, Transformers 4.51.3')
    parent = ROOT / 'third_party/evaluation'
    parent.mkdir(parents=True, exist_ok=True)
    for name, url, revision in REPOSITORIES:
        path = parent / name
        if not path.exists():
            subprocess.run(['git', 'clone', url, str(path)], check=True)
            subprocess.run(['git', '-C', str(path), 'checkout', revision], check=True)
        actual = subprocess.check_output(['git', '-C', str(path), 'rev-parse', 'HEAD'], text=True).strip()
        if actual != revision:
            raise RuntimeError(f'{path} is at {actual}, expected {revision}; do not overwrite an existing checkout')
    command = [sys.executable, '-m', 'pip', 'install', '--no-deps', '--target', str(parent / 'python_deps')]
    subprocess.run(command + ['-r', str(ROOT / 'requirements-clinical.txt')], check=True)
    subprocess.run(command + ['--no-build-isolation'] + [str(parent / n) for n, _, _ in REPOSITORIES], check=True)
    code = ('import sys;sys.path.insert(0,' + repr(str(parent / 'python_deps')) + ');'
            'import torch,transformers,numpy,green_score,radgraph,RaTEScore;'
            'print(torch.__version__,transformers.__version__,numpy.__version__)')
    subprocess.run([sys.executable, '-c', code], check=True)
    assert core == {n: metadata.version(n) for n in core}, 'Training core packages unexpectedly changed'
    print(json.dumps({'status': 'ready', 'training_core_unchanged': core}, indent=2))


if __name__ == '__main__':
    main()
