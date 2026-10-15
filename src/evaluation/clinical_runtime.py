"""Opt-in evaluation dependencies; never imported by training."""
import importlib.metadata as metadata
import os
import sys
from pathlib import Path
from .clinical import digest, file_digest


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ASSETS = ROOT / 'assets/models/evaluation'
BERTSCORE_SETTINGS = {
    'model_type': 'google-bert/bert-base-uncased', 'num_layers': 9, 'lang': 'en',
    'idf': False, 'rescale_with_baseline': False, 'use_fast_tokenizer': False,
    'all_layers': False, 'dtype': 'float32', 'max_tokens': 512,
    'length_policy': 'reject_before_scoring', 'aggregation': 'mean_per_report',
}


def activate(assets=DEFAULT_ASSETS):
    dependencies = ROOT / 'third_party/evaluation/python_deps'
    if dependencies.exists():
        if 'numpy' in sys.modules:
            raise RuntimeError('Activate clinical dependencies before importing NumPy; use the offline CLI')
        sys.path.insert(0, str(dependencies))
    # Scoring is offline after preparation; no hidden weight downloads outside assets/models.
    assets = Path(assets).resolve()
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    os.environ['HF_HUB_CACHE'] = str(assets / 'hub')
    os.environ['HF_MODULES_CACHE'] = str(assets / 'modules')
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'


def environment(metrics):
    # An individual scorer must not require unrelated optional packages.
    names = {'torch', 'transformers', 'numpy', 'tokenizers', 'huggingface-hub'}
    optional = {
        'green': {'green_score', 'sentencepiece', 'sentence-transformers', 'datasets'},
        'radgraph': {'radgraph', 'scipy'},
        'rate': {'RaTEScore', 'medspacy', 'spacy', 'thinc', 'PyRuSH', 'sentencepiece',
                 'sentence-transformers', 'scipy'},
        'bertscore': {'bert-score', 'pandas', 'matplotlib', 'packaging'},
    }
    for metric in metrics:
        names.update(optional[metric])
    return {'python': sys.version, 'executable': sys.executable,
            'packages': {name: metadata.version(name) for name in sorted(names)}}


def verify_asset(path):
    import json
    path = Path(path)
    manifest = json.loads((path / 'asset_manifest.json').read_text())
    for entry in manifest['files']:
        file = path / entry['path']
        if not file.is_file() or file.stat().st_size != entry['size'] or file_digest(file) != entry['sha256']:
            raise ValueError(f'Asset integrity check failed: {file}')
    return manifest


def package_digest(name):
    """Hash the installed implementation, including uncommitted local edits."""
    import importlib.util
    spec = importlib.util.find_spec(name)
    root = Path(spec.origin).parent
    return digest({str(p.relative_to(root)): file_digest(p) for p in sorted(root.rglob('*.py'))})


def specifications(metrics, assets, device, green_max_length=2048):
    assets = Path(assets).resolve()
    source_hash = digest({p.name: file_digest(p) for p in Path(__file__).parent.glob('clinical*.py')})
    packages = {'green': 'green_score', 'radgraph': 'radgraph', 'rate': 'RaTEScore', 'bertscore': 'bert_score'}
    models = {'green': ['green'], 'radgraph': ['radgraph', 'cxr-bert'],
              'rate': ['rate-ner', 'biolord'], 'bertscore': ['bert-score']}
    result = {}
    for metric in metrics:
        print(f'Verifying {metric} assets and implementation...', flush=True)
        result[metric] = {
            'schema': 1, 'metric': metric, 'environment': environment([metric]), 'adapter_sha256': source_hash,
            'upstream_code_sha256': package_digest(packages[metric]),
            'models': {name: verify_asset(assets / name) for name in models[metric]},
            'device': device, 'batch_size': 1,
            'settings': ({'dtype': 'float16', 'max_length': green_max_length,
                          'do_sample': False, 'prompt': 'upstream make_prompt + chat_template',
                          'parse_policy': 'official counts with structural validation and canonical format repair v1'} if metric == 'green' else
                         {'model_type': 'radgraph-xl', 'reward_level': 'all', 'dtype': 'float32'} if metric == 'radgraph' else
                         {'affinity_matrix': 'long', 'dtype': 'float32', 'entity_max_tokens': 30,
                          'sentence_max_tokens': 512, 'empty_entities': 0.5} if metric == 'rate' else
                         dict(BERTSCORE_SETTINGS)),
        }
    return result
