"""Explicit, source-specific CE replay settings; no guesses from a folder name."""
from copy import deepcopy
from importlib.metadata import version
import json
import os
from pathlib import Path

from src.evaluation.clinical import digest, file_digest

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROFILES = ROOT / 'scripts/significance/ce_runtime_profiles.json'


CE_ENVIRONMENT = {
    'CUBLAS_WORKSPACE_CONFIG': ':4096:8',
    'PYTORCH_CUDA_ALLOC_CONF': 'expandable_segments:True',
    'OMP_NUM_THREADS': '4',
    'MKL_NUM_THREADS': '4',
    'OPENBLAS_NUM_THREADS': '1',
    'PYTHONNOUSERSITE': '1',
    'TOKENIZERS_PARALLELISM': 'false',
}


def ce_replay_environment(parent=None, settings=None):
    # CPU permutation/bootstrap launchers use one thread. Do not inherit that
    # setting for model execution, or initialize CUDA before its environment.
    environment = dict(os.environ if parent is None else parent)
    settings = CE_ENVIRONMENT if settings is None else settings
    allowed = set(CE_ENVIRONMENT) | {'NUMEXPR_NUM_THREADS', 'OMP_DYNAMIC', 'MKL_DYNAMIC',
        'CUDA_DEVICE_ORDER', 'NVIDIA_TF32_OVERRIDE', 'PYTHONHASHSEED',
        'ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS'}
    if set(settings) - allowed:
        raise ValueError('Unsupported CE environment setting')
    for key, value in settings.items():
        if value is None:
            environment.pop(key, None)
        elif isinstance(value, str):
            environment[key] = value
        else:
            raise ValueError(f'CE environment value must be a string or null: {key}')
    return environment


def resolve_ce_runtime(entry, profiles_path=DEFAULT_PROFILES):
    import yaml
    run = Path(entry['run_dir'])
    if not run.is_absolute():
        run = ROOT / run
    evaluation = run / entry.get('evaluation_subdir', 'evaluation/best_test')
    original = yaml.safe_load((evaluation / 'evaluation_config.yaml').read_text())
    pipeline = json.loads((evaluation / 'pipeline.json').read_text())
    origin = dict(project_root=original['project_root'], executable=pipeline['commands'][0][0])
    profiles = json.loads(Path(profiles_path).read_text())
    matches = [p for p in profiles['profiles'] if p['origin'] == origin]
    if len(matches) != 1:
        raise ValueError(f'Expected one explicit CE runtime profile for {origin}; found {len(matches)}')
    profile = deepcopy(matches[0])
    override = profiles.get('checkpoint_overrides', {}).get(entry['checkpoint_sha256'])
    if override:
        profile.update(override)
    for evidence in profile.get('evidence_files', []):
        path = Path(evidence['path'])
        if not path.is_absolute():
            path = ROOT / path
        if file_digest(path) != evidence['sha256']:
            raise ValueError(f'CE runtime evidence changed: {path}')
    ce_replay_environment({}, profile['environment'])  # Validate before spawning.
    profile['profile_sha256'] = digest(profile)
    return profile


def verify_runtime_packages(profile):
    actual = {name: version(name) for name in profile.get('packages', {})}
    mismatches = {name: {'expected': expected, 'actual': actual[name]}
                  for name, expected in profile.get('packages', {}).items() if actual[name] != expected}
    if mismatches:
        raise ValueError(f'CE replay dependency mismatch: {mismatches}')
    return actual
