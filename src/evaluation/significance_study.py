"""Excel targets and explicit identities for the five-ratio Baseline A study."""
import math
from pathlib import Path
import re

import yaml
from openpyxl import load_workbook

from src.evaluation.clinical import digest, file_digest
from src.evaluation.significance import METRICS
from src.evaluation.significance_io import read_json, resolve, find_metric_files

RATIOS = (16, 32, 64, 128, 256)
METHODS = {
    'ours': ('Baseline A', 'bindformer3d'),
    'avgpool': ('AvgPool3D', 'avgpool3d'),
    'concat': ('Concat3D', 'concat3d'),
    'localpacker': ('LocalPacker3D', 'local_packer3d'),
    'perceiver': ('Perceiver_NV', 'perceiver_nv'),
    'adaptive': ('Adaptive_Merge3D', 'adaptive_merge3d'),
    'tokenselect': ('Token_Select3D', 'token_select3d'),
}
HEADERS = (
    'BLEU-4 ↑', 'ROUGE-L F1 ↑', 'GREEN ↑', 'RadGraph-XL Simple F1 ↑',
    'RadGraph-XL Partial F1 ↑', 'RadGraph-XL Complete F1 ↑', 'RaTEScore ↑',
    'BERTScore Precision ↑', 'BERTScore Recall ↑', 'BERTScore F1 ↑', 'Answer CE ↓',
)


def method_for_label(label):
    label = ' '.join(str(label).split())
    # Explicitly recognize the workbook's Vanlla typo, never an ablation suffix.
    if re.fullmatch(r'Ours Van(?:i)?lla\s*\+\s*fixed region mean 1021(?: \(Baseline A\))?', label):
        return 'ours'
    return next((key for key, (name, _) in METHODS.items() if label == name), None)


def read_targets(workbook):
    workbook = Path(workbook).resolve()
    before = file_digest(workbook)
    book = load_workbook(workbook, data_only=True)
    sheet = book['Main']
    found, headers, sections = {}, set(), set()
    ratio = None
    for row in sheet.iter_rows():
        label = str(row[0].value or '').strip()
        match = re.fullmatch(r'[a-z]\.\s*(\d+)[x×]\s+Compression', label, re.I)
        if match:
            ratio = int(match[1])
            if ratio in RATIOS:
                if ratio in sections:
                    raise ValueError(f'Duplicate {ratio}x section in Main')
                sections.add(ratio)
            continue
        if ratio not in RATIOS:
            continue
        if row[1].value == HEADERS[0]:
            if tuple(c.value for c in row[1:12]) != HEADERS:
                raise ValueError(f'Metric header/order changed in {ratio}x block')
            headers.add(ratio)
            continue
        method = method_for_label(label)
        if not method:
            continue
        key = (ratio, method)
        if key in found:
            raise ValueError(f'Duplicate target row: {key}')
        values = [c.value for c in row[1:12]]
        if len(values) != 11 or any(type(v) not in (int, float) or not math.isfinite(v) for v in values):
            raise ValueError(f'Main row {row[0].row}: all 11 numeric scores are required')
        found[key] = dict(id=method, label=label, projector=METHODS[method][1], ratio=ratio,
                          expected_rounded_metrics=dict(zip(METRICS, values)),
                          excel_row=row[0].row, excel_range=f'Main!A{row[0].row}:L{row[0].row}')
    book.close()
    expected = {(r, m) for r in RATIOS for m in METHODS}
    if set(found) != expected or headers != set(RATIOS):
        raise ValueError(f'Incomplete target blocks: missing {sorted(expected-set(found))}; headers {sorted(headers)}')
    if file_digest(workbook) != before:
        raise ValueError('Workbook changed during read; rerun after the upload finishes')
    ordered = [found[(r, m)] for r in RATIOS for m in METHODS]
    return ordered, dict(type='excel', workbook=str(workbook), workbook_sha256=before,
                        sheet='Main', targets_sha256=digest(ordered))


def check_baseline_a(config):
    params = config['projector']['params']
    for name, expected in {
        'fixed_spatial_regions': True, 'spatial_mean_aggregation': True,
        'spatial_adaptive_aggregation': False, 'within_slot_sa': False,
        'local_reads_global': False, 'geometry_enabled': False,
    }.items():
        if params.get(name) is not expected:
            raise ValueError(f'Baseline A requires {name}={expected}')
    for key in ('role_value_adapter', 'report_output', 'report_facts', 'report_target'):
        value = params.get(key, {})
        enabled = value.get('enabled', False) if isinstance(value, dict) else bool(value)
        if enabled:
            raise ValueError(f'Baseline A excludes {key}')
    for key in ('feature_loss_weight', 'text_loss_weight', 'tuple_loss_weight',
                'binding_loss_weight', 'concept_loss_weight'):
        if params.get(key, 0) != 0:
            raise ValueError(f'Baseline A excludes auxiliary loss {key}')


def inventory(roots):
    items, errors = [], []
    for path in find_metric_files(roots):
        if path.parent.name != 'best_test':
            continue
        try:
            metric = read_json(path)
            run = path.parents[2]
            cfg_path = run / metric['stage'] / 'resolved_config.yaml'
            config = yaml.safe_load(cfg_path.read_text())
            items.append(dict(run_dir=str(run), checkpoint_sha256=metric['checkpoint_sha256'],
                              metric_sha256=file_digest(path), config_sha256=file_digest(cfg_path),
                              metric=metric, config=config))
        except (OSError, ValueError, KeyError, TypeError, yaml.YAMLError) as exc:
            errors.append(dict(path=str(path), reason=f'{type(exc).__name__}: {exc}'))
    return items, errors


def matching_candidates(target, items):
    candidates = []
    for item in items:
        cfg = item['config']
        if (cfg['projector']['name'] != target['projector'] or
                cfg['projector'].get('target_compression_ratio') != target['ratio']):
            continue
        values = item['metric'].get('overall', {})
        if any(type(values.get(k)) not in (int, float) or not math.isfinite(values[k]) or
               abs(values[k]-v) > .000050001 for k, v in target['expected_rounded_metrics'].items()):
            continue
        if target['id'] == 'ours':
            try:
                check_baseline_a(cfg)
            except ValueError:
                continue
        candidates.append(item)
    return candidates


def candidate_identity(item):
    return {k: item[k] for k in ('run_dir', 'checkpoint_sha256', 'metric_sha256', 'config_sha256')}


def select_candidate(target, candidates, pin=None):
    """Select a candidate for full audit, not yet an authorized statistical input.

    Frozen bindings never switch to another run. First-use candidates must be
    unique; full checkpoint/raw-score and paired-contract audits follow before
    the caller freezes or uses them. No newest/best-score tie breaking.
    """
    if pin:
        selected = [c for c in candidates if resolve(c['run_dir']) == resolve(pin['run_dir'])]
        if len(selected) != 1:
            raise ValueError('Frozen run missing, incomplete, or no longer matches Excel; no fallback')
        item = selected[0]
        for key in ('checkpoint_sha256', 'metric_sha256', 'config_sha256'):
            if pin[key] != item[key]:
                raise ValueError(f'Frozen {key} mismatch; no fallback')
        return item
    if not candidates:
        raise ValueError('No exact matching candidate; transfer may be incomplete')
    if len(candidates) != 1:
        raise ValueError('Ambiguous matching runs; supply an explicit binding, never pick by date or score')
    return candidates[0]
