"""Artifact identity checks for the independent significance pipeline."""
from copy import deepcopy
import json
import math
import os
from pathlib import Path
import shutil
import subprocess

import yaml

from src.evaluation.clinical import digest, file_digest, read_jsonl, validate_pair
from src.evaluation.significance import METRICS, GROUPS, lexical_vectors, statistic

ROOT = Path(__file__).resolve().parents[2]


def read_json(path):
    return json.loads(Path(path).read_text())


def resolve(path):
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def find_metric_files(roots):
    """Discover metric.json, including hidden/ignored directories, without requiring rg."""
    paths = set()
    executable = shutil.which('rg')

    def traversal_error(error):
        # Permission/I/O failures must not silently hide a possible candidate.
        raise error

    for root in roots:
        root = resolve(root)
        if not root.is_dir():
            continue
        if executable:
            try:
                result = subprocess.run([executable, '--files', '-uu', str(root), '-g', 'metric.json'],
                                        text=True, capture_output=True)
            except FileNotFoundError:
                executable = None
            else:
                if result.returncode not in (0, 1):
                    raise RuntimeError(result.stderr)
                paths.update(Path(p).resolve() for p in result.stdout.splitlines())
                continue
        for directory, _, filenames in os.walk(root, followlinks=False, onerror=traversal_error):
            if 'metric.json' in filenames:
                paths.add((Path(directory) / 'metric.json').resolve())
    return sorted(paths)


def row_identity(rows):
    return digest(sorted((r['id'], r['patient_id'], r['case_id'], r['sequence'], r['reference']) for r in rows))


def close(a, b, context, tolerance=1e-9):
    if not math.isclose(a, b, rel_tol=tolerance, abs_tol=tolerance):
        raise ValueError(f'{context}: recomputed {a}, recorded {b}')


def scorer_contract(specification):
    """Numerical/scoring contract, excluding recorded host placement metadata.

    Hardware differences are retained separately, not claimed bit-identical.
    Model bytes, code hashes, settings, Python/package versions, seed, TF32 and
    device backend remain exact requirements. Only executable location, GPU
    model label and CUDA device ordinal are excluded from the comparison hash.
    """
    contract = deepcopy(specification)
    contract.pop('device_name', None)
    contract.get('environment', {}).pop('executable', None)
    device = contract.get('device')
    if isinstance(device, str) and device.startswith('cuda:') and device[5:].isdecimal():
        contract['device'] = 'cuda'
    return contract


def load_run(entry):
    """Verify the frozen run, raw scores, data pairing, and checkpoint selection."""
    run = resolve(entry['run_dir'])
    ev = run / entry.get('evaluation_subdir', 'evaluation/best_test')
    metric = read_json(ev / 'metric.json')
    stage = run / metric['stage']
    cfg_path = stage / 'resolved_config.yaml'
    for key, path in (('metric_sha256', ev / 'metric.json'), ('config_sha256', cfg_path)):
        if entry.get(key) and entry[key] != file_digest(path):
            raise ValueError(f'Frozen {key} mismatch')
    cfg = yaml.safe_load(cfg_path.read_text())
    if cfg['projector']['name'] != entry['projector']:
        raise ValueError('Projector identity mismatch')
    ratio = cfg['projector'].get('target_compression_ratio')
    if ratio != entry['ratio']:
        raise ValueError(f'Compression ratio mismatch: {ratio} != {entry["ratio"]}')
    if metric['split'] != 'test' or metric['scope'] != 'full_split':
        raise ValueError('Expected full held-out test evaluation')
    if metric.get('skipped_metrics'):
        raise ValueError('Recorded evaluation has skipped metrics')
    checkpoint_hash = metric['checkpoint_sha256']
    if entry.get('checkpoint_sha256') and entry['checkpoint_sha256'] != checkpoint_hash:
        raise ValueError('Frozen checkpoint hash mismatch')
    expected = entry.get('expected_rounded_metrics', {})
    for key, value in expected.items():
        actual = metric['overall'][key]
        if not math.isfinite(actual) or abs(actual - value) > .000050001:
            raise ValueError(f'Target table differs in {key}: {metric["overall"][key]} vs {value}')
    pipeline = read_json(ev / 'pipeline.json')
    if pipeline['checkpoint_sha256'] != checkpoint_hash:
        raise ValueError('Pipeline/checkpoint mismatch')
    predictions_path = ev / 'generation/predictions.jsonl'
    rows = read_jsonl(predictions_path)
    clinical_path = ev / 'scoring/clinical_scores.jsonl'
    clinical = read_jsonl(clinical_path)
    ids = [r['id'] for r in rows]
    if (len(ids) != len(set(ids)) or set(ids) != set(read_json(ev / 'expected_ids.json'))
            or len(ids) != metric['evaluated_samples']):
        raise ValueError('Prediction ID/coverage mismatch')
    if any(r['checkpoint_id'] != checkpoint_hash for r in rows):
        raise ValueError('Prediction/checkpoint mismatch')
    if any(r.get('control') is not None for r in rows):
        raise ValueError('Visual-control predictions cannot replace normal main-table evaluation')
    if any(r.get('projector_name', entry['projector']) != entry['projector'] for r in rows):
        raise ValueError('Prediction/projector mismatch')
    if len({r['patient_id'] for r in rows}) != metric['patients']:
        raise ValueError('Patient count mismatch')
    provenance = read_json(ev / 'scoring/scoring_provenance.json')
    if file_digest(predictions_path) != provenance['input_sha256']:
        raise ValueError('Scorer input hash mismatch')
    # Self-validation checks cached text hashes, status, and scalar score ranges.
    validate_pair(clinical, clinical, list(set(GROUPS.values())))
    scorer_contracts = {}
    for group in set(GROUPS.values()):
        specification = provenance['scorers'][group]
        fingerprint = digest(specification)
        if any(r['clinical'][group]['scorer_fingerprint'] != fingerprint for r in clinical):
            raise ValueError(f'Scorer provenance/record fingerprint mismatch: {group}')
        if provenance['execution'][group]['scorer_fingerprint'] != fingerprint:
            raise ValueError(f'Scorer execution/provenance fingerprint mismatch: {group}')
        scorer_contracts[group] = digest(scorer_contract(specification))
    by_id = {r['id']: r for r in clinical}
    if set(by_id) != set(ids):
        raise ValueError('Clinical/prediction ID mismatch')
    for r in rows:
        for field in ('patient_id', 'case_id', 'sequence', 'reference', 'prediction', 'checkpoint_id'):
            if r[field] != by_id[r['id']][field]:
                raise ValueError(f'Clinical/prediction {field} mismatch')
    vectors = lexical_vectors(rows)
    for key, group in GROUPS.items():
        vectors[key] = {r['id']: [r['clinical'][group]['values'][key], 1]
                        for r in clinical if r['clinical'][group]['status'] == 'ok'}
    for key in METRICS[:-1]:
        values = list(vectors[key].values())
        if not values:
            raise ValueError(f'No valid {key} scores')
        sums = [math.fsum(column) for column in zip(*values)]
        close(statistic(key, sums), metric['overall'][key], key)
    generation = read_json(ev / 'generation/metrics.json')
    ce = generation['answer_ce']
    cohorts = list(ce['cohorts'].values())
    close(sum(c['loss_sum'] for c in cohorts) / sum(c['answer_tokens'] for c in cohorts),
          metric['overall']['answer_ce'], 'Historical token-weighted CE')
    checkpoint = stage / 'checkpoints/best.pt'
    verified_weights = checkpoint.is_file()
    selection = read_json(stage / 'training_state.json')
    validation = read_jsonl(stage / 'validation.jsonl')
    selected = [r for r in validation if r['step'] == selection['global_step']]
    # training_state is the final step, whereas best.pt identifies the selected step.
    if verified_weights:
        if file_digest(checkpoint) != checkpoint_hash:
            raise ValueError('Actual checkpoint bytes differ from evaluation identity')
        import torch
        saved = torch.load(checkpoint, map_location='cpu', mmap=True, weights_only=False)
        if saved['config']['projector'] != cfg['projector'] or saved['config']['seed'] != cfg['seed']:
            raise ValueError('Checkpoint/resolved configuration mismatch')
        for section in ('encoder', 'decoder', 'task', 'preprocessing', 'generation', 'data', 'training'):
            if saved['config'][section] != cfg[section]:
                raise ValueError(f'Checkpoint/resolved {section} mismatch')
        selected = [r for r in validation if r['step'] == saved['state']['global_step']]
        if len(selected) != 1:
            raise ValueError('Selected checkpoint step absent from validation history')
        close(selected[0]['answer_ce'], saved['state']['best_val_ce'], 'Selected validation CE')
        selected_step = saved['state']['global_step']
        selected_ce = saved['state']['best_val_ce']
        selection_identity = {k: saved['state'].get(k) for k in ('train_selection', 'val_selection')}
        del saved
    else:
        selected_step = selected_ce = None
        selection_identity = {k: selection.get(k) for k in ('train_selection', 'val_selection')}
    source_paths = [ev/'metric.json', cfg_path, predictions_path, clinical_path,
                    ev/'scoring/scoring_provenance.json', stage/'training_state.json',
                    stage/'validation.jsonl', ev/'generation/metrics.json']
    audit = {
        'method_id': entry['id'], 'label': entry['label'], 'run_dir': str(run),
        'evaluation_dir': str(ev), 'checkpoint': str(checkpoint),
        'checkpoint_sha256': checkpoint_hash, 'checkpoint_bytes_verified': verified_weights,
        'config_sha256': file_digest(cfg_path), 'projector': cfg['projector'],
        'seed': cfg['seed'], 'ratio': ratio, 'selected_step': selected_step,
        'selected_validation_ce': selected_ce,
        'checkpoint_selection': 'recorded_best_validation_ce',
        'test_identity_sha256': row_identity(rows), 'samples': len(rows),
        'patients': len({r['patient_id'] for r in rows}),
        'training_selection_sha256': digest(selection_identity),
        'training_recipe': {k: cfg['training'].get(k) for k in (
            'stage', 'projector_lr', 'encoder_lr', 'max_epochs', 'cosine_min_lr_ratio',
            'warmup_steps', 'joint_warmup_projector_mode', 'effective_batch_size', 'deterministic')},
        'scorer_fingerprints': {g: clinical[0]['clinical'][g]['scorer_fingerprint'] for g in set(GROUPS.values())},
        'scorer_contract_fingerprints': scorer_contracts,
        'scorer_runtime': {g: {k: provenance['scorers'][g].get(k) for k in ('device', 'device_name')}
                           | {'executable': provenance['scorers'][g]['environment'].get('executable')}
                           for g in set(GROUPS.values())},
        'original_metrics': metric['overall'],
        'original_ce_aggregation': ce,
        'valid_counts': {key: len(value) for key, value in vectors.items()},
        'source_hashes': {str(p): file_digest(p) for p in source_paths},
        'table_snapshot_match': bool(expected),
    }
    return {'rows': rows, 'clinical': clinical, 'vectors': vectors, 'audit': audit, 'config': cfg}


def validate_run_pair(ours, other):
    """Reject incompatible evaluation contracts before any inference or test."""
    # load_run already validates every raw record and its original fingerprint.
    # Across servers, compare full semantic specs rather than host paths/names.
    validate_pair(ours['clinical'], other['clinical'], [])
    if ours['audit']['scorer_contract_fingerprints'] != other['audit']['scorer_contract_fingerprints']:
        raise ValueError('Paired scorer code/model/settings/environment contract mismatch')
    if ours['audit']['test_identity_sha256'] != other['audit']['test_identity_sha256']:
        raise ValueError('Paired test IDs/references/patient identities differ')
    matched = {r['id']: r for r in other['rows']}
    for row in ours['rows']:
        for field in ('preprocessing_version', 'prompt_version', 'split', 'target_version'):
            if row[field] != matched[row['id']][field]:
                raise ValueError(f'Paired evaluation {field} differs')
    if ours['audit']['training_selection_sha256'] != other['audit']['training_selection_sha256']:
        raise ValueError('Training/validation sample selections differ')


def attach_ce(run, export_dir, *, policy='strict'):
    path = Path(export_dir)
    if not (path / 'summary.json').exists():
        return False
    summary = read_json(path / 'summary.json')
    samples_path = path / 'samples.jsonl'
    if (summary['status'] != 'complete'
            or summary['checkpoint_sha256'] != run['audit']['checkpoint_sha256']
            or summary['test_identity_sha256'] != run['audit']['test_identity_sha256']
            or summary['samples_sha256'] != file_digest(samples_path)):
        raise ValueError('CE export identity/hash mismatch')
    rows = read_jsonl(samples_path)
    original = {r['id']: r for r in run['rows']}
    if len(rows) != len(original) or {r['id'] for r in rows} != set(original):
        raise ValueError('CE export coverage mismatch')
    for r in rows:
        if r.get('checkpoint_sha256') != summary['checkpoint_sha256']:
            raise ValueError('CE sample checkpoint mismatch')
        if r['patient_id'] != original[r['id']]['patient_id'] or r['reference_sha256'] != digest(original[r['id']]['reference']):
            raise ValueError('CE patient/reference mismatch')
        if type(r['answer_tokens']) is not int or r['answer_tokens'] <= 0 or not math.isfinite(r['loss_sum']) or r['loss_sum'] < 0:
            raise ValueError('Invalid CE numerator/denominator')
    total = math.fsum(r['loss_sum'] for r in rows)
    tokens = sum(r['answer_tokens'] for r in rows)
    close(total / tokens, summary['answer_ce'], 'CE export internal aggregation')
    if 'historical_aggregation' in summary:
        from src.evaluation.ce_validation import compare_ce_history, require_historical_match
        history = summary['historical_aggregation']
        if history != run['audit'].get('original_ce_aggregation', history):
            raise ValueError('CE historical aggregation differs from original artifact')
        close(history['answer_ce'], run['audit']['original_metrics']['answer_ce'], 'Historical CE metadata')
        comparison = compare_ce_history(rows, history)
        if comparison != summary['historical_comparison']:
            raise ValueError('CE historical comparison metadata mismatch')
        require_historical_match(comparison, summary['ce_policy'])
        require_historical_match(comparison, policy)
    else:
        # Legacy complete exports were accepted only after strict replay checks.
        close(total / tokens, run['audit']['original_metrics']['answer_ce'], 'CE replay vs historical result', 1e-5)
    run['vectors']['answer_ce'] = {r['id']: [r['loss_sum'], r['answer_tokens']] for r in rows}
    run['audit']['ce_export'] = str(path)
    run['audit']['ce_samples_sha256'] = summary['samples_sha256']
    run['audit']['ce_replay'] = dict(answer_ce=total/tokens,
        historical_answer_ce=run['audit']['original_metrics']['answer_ce'],
        difference=total/tokens-run['audit']['original_metrics']['answer_ce'],
        ce_policy=summary.get('ce_policy', 'strict'), score_basis='recomputed_teacher_forced_ce',
        runtime_profile=summary.get('runtime_profile'), device_name=summary.get('device_name'),
        runtime_environment=summary.get('runtime_environment'))
    return True
