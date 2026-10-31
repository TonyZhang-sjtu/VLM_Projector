"""Shared VQA data admission before sampling/gradient accumulation.

Sample failures are filtered once. Runtime/model failures are never swallowed.
"""
from collections import Counter
from dataclasses import replace
import json
import math
import pickle
from pathlib import Path

from src.config import digest, file_hash, path_of
from src.data.neuroqa_manifest import balanced_epoch, load_neuroqa, read_rows, write_json
from src.models.projectors.budget import prepare_budget, geometry_from_header
from src.preprocessing.registry import build_preprocessor
from src.training.schedule import resolve_joint_schedule


def training_scope(config, report=None):
    if any(config.training.get(k) for k in
           ('train_limit', 'val_limit', 'train_ids_file', 'val_ids_file', 'max_steps')):
        return 'smoke'
    if config.data.get('invalid_sample_policy') == 'include_unreviewed':
        return 'unreviewed_training'
    if report and (not report['qc_ready_for_full_training'] or
                   any(report.get('skipped', {}).get(k) for k in ('train', 'val'))):
        return 'filtered_training'
    return 'training'


def selected(config, samples, split):
    ids_file = config.training[f'{split}_ids_file']
    if ids_file:
        ids = json.loads(path_of(config, ids_file).read_text())
        lookup = {s.qa_id if split == 'train' else s.id: s for s in samples}
        if not isinstance(ids, list) or not ids or len(set(ids)) != len(ids) or any(i not in lookup for i in ids):
            raise ValueError('Invalid QA selection: train uses QA IDs; val uses QA__modality IDs')
        samples = [lookup[i] for i in ids]
    limit = config.training[f'{split}_limit']
    return samples[:limit] if limit else samples


def evaluation_samples(populations, split, ids_file=None, limit=None):
    samples = populations[split]
    if split == 'train':
        samples = [s.view(m) for s in samples for m in s.row['allowed_modalities']]
    if ids_file:
        ids = json.loads(Path(ids_file).read_text())
        lookup = {s.id: s for s in samples}
        if not isinstance(ids, list) or not ids or len(ids) != len(set(ids)) or any(i not in lookup for i in ids):
            raise ValueError('Invalid evaluation view IDs')
        samples = [lookup[i] for i in ids]
    if limit is not None:
        if limit < 1:
            raise ValueError('Evaluation limit must be positive')
        samples = samples[:limit]
    return samples


def _skip_row(sample, split, reasons):
    return {'id': sample.qa_id if split == 'train' else sample.id,
            'qa_id': sample.qa_id, 'case_id': sample.case_id, 'reasons': list(reasons)}


def _review_failures(root):
    """Honor an existing explicit QC review queue; absence is reported globally."""
    queue_path, review_path = root / 'qc_review_queue.json', root / 'qc_review.json'
    if not queue_path.is_file():
        return {}
    queue = json.loads(queue_path.read_text())
    review = json.loads(review_path.read_text()) if review_path.is_file() else {}
    accepted = review.get('reviews', {}) if review.get('queue_sha256') == file_hash(queue_path) else {}
    return {r['case_id']: ['QC review pending/rejected: ' + ', '.join(r['reasons'])]
            for r in queue['cases'] if accepted.get(r['case_id'], {}).get('decision') != 'accepted'}


def _verify_unreviewed_v2(config, root):
    """Check global provenance; return identifiable case-level failures to skip."""
    if config.preprocessing.name != 'neuroqa_mni1mm_v2':
        raise ValueError('include_unreviewed is supported only for independently audited NeuroQA v2')
    queue = json.loads((root/'qc_review_queue.json').read_text())
    for key, filename in (('audit_sha256','independent_audit.json'),
                          ('intensity_audit_sha256','intensity_audit.json'),
                          ('lesion_audit_sha256','lesion_coverage_audit.json'),
                          ('reproducibility_sha256','production_reproducibility.json')):
        if queue.get(key) != file_hash(root/filename):
            raise ValueError(f'QC queue evidence mismatch: {filename}')
    source = read_rows(path_of(config, config.data.manifest_dir) / 'source_index.jsonl')
    expected = {row['case_id'] for row in source}
    if not expected or len(expected) != len(source):
        raise ValueError('Invalid audit source index')
    failures = {}
    def add(case, reason):
        if case not in expected:
            raise ValueError(f'Audit failure has unknown case: {case}')
        failures.setdefault(case, []).append(reason)
    audit = json.loads((root/'independent_audit.json').read_text())
    final = json.loads((root/'finalization.json').read_text())
    rows, summary = audit.get('cases'), audit.get('summary', {})
    if (not isinstance(rows, list) or not audit.get('finished_utc') or
            summary.get('expected_cases') != len(expected) or summary.get('audited_cases') != len(expected) or
            len(rows) != len(expected) or {r.get('case_id') for r in rows} != expected):
        raise ValueError('Independent audit incomplete or cases cannot be identified')
    for row in rows:
        case = row['case_id']
        if not row.get('completed'):
            add(case, 'Independent audit: preprocessing incomplete')
        for error in row.get('integrity_errors', []):
            add(case, f'Independent artifact audit: {error}')
        volumes = row.get('volumes', [])
        if row.get('completed') and ({v.get('modality') for v in volumes} != {'T1w', 'T2w'} or len(volumes) != 2):
            add(case, 'Independent audit: missing/duplicate modality')
        for volume in volumes:
            for error in volume.get('errors', []):
                add(case, f'Independent {volume.get("modality")} audit: {error}')
            if not volume.get('patch_recomputed_exact'):
                add(case, f'Independent {volume.get("modality")} patch audit incomplete')
    verified, failed = final.get('verified_volumes'), final.get('failed')
    if (not isinstance(verified, list) or not isinstance(failed, list) or
            final.get('expected_cases') != len(expected) or final.get('expected_volumes') != 2 * len(expected)):
        raise ValueError('Invalid v2 finalization coverage')
    pairs = []
    for row in verified + failed:
        pair = (row.get('case_id'), row.get('modality'))
        if pair[0] not in expected or pair[1] not in ('T1w', 'T2w'):
            raise ValueError('Finalization contains an unidentifiable volume')
        pairs.append(pair)
    if len(pairs) != 2 * len(expected) or len(set(pairs)) != len(pairs):
        raise ValueError('Finalization has incomplete/duplicate volume coverage')
    for row in failed:
        add(row['case_id'], f'Finalization {row["modality"]}: {row.get("error", "failed")}')
    for name in ('intensity_audit.json','lesion_coverage_audit.json'):
        data = json.loads((root/name).read_text())
        good, bad = data.get('cases'), data.get('failed')
        if (not isinstance(good, list) or not isinstance(bad, list) or not data.get('finished_utc') or
                data.get('expected_cases') != len(expected) or len(good) + len(bad) != len(expected)):
            raise ValueError(f'Incomplete v2 audit: {name}')
        ids = [row.get('case_id') for row in good + bad]
        if len(set(ids)) != len(ids) or set(ids) != expected or (not bad and not data.get('complete')):
            raise ValueError(f'Incomplete or unidentifiable v2 audit: {name}')
        for row in bad:
            add(row['case_id'], f'{name}: {row.get("error", "failed")}')
    if json.loads((root/'production_reproducibility.json').read_text()).get('status') != 'passed':
        raise ValueError('V2 image reproducibility audit failed')
    review_path = root/'qc_review.json'
    if review_path.exists():
        review = json.loads(review_path.read_text())
        if review.get('queue_sha256') == file_hash(root/'qc_review_queue.json'):
            for case, value in review.get('reviews',{}).items():
                if value.get('decision') == 'rejected':
                    add(case, 'QC review explicitly rejected')
    return failures


def _check_case(config, pre, sample):
    """Validate both MRI volumes of a registered case before admitting its QA."""
    folder = Path(sample.image).parent
    record = json.loads((folder / 'complete.json').read_text())
    if (not isinstance(record, dict) or not isinstance(record.get('contract'), dict)
            or not isinstance(record.get('artifacts'), dict)):
        raise ValueError('Malformed preprocessing completion record')
    if record.get('fingerprint') != digest(record['contract']):
        raise ValueError('Preprocessing fingerprint mismatch')
    actual = {k: v for k, v in record['contract'].items() if k != 'source_sha256'}
    if config.preprocessing.name == 'neuroqa_mni1mm':
        from src.preprocessing.neuroqa_smri import compatible_contract
        compatible = compatible_contract(actual, pre.spatial_contract)
    else:
        compatible = actual == pre.spatial_contract
    if not compatible:
        raise ValueError('Preprocessing contract mismatch')
    status_path = folder / 'status.json'
    if status_path.is_file():
        status = json.loads(status_path.read_text())
        if not isinstance(status, dict) or status.get('stage') != 'complete':
            raise ValueError('Case status is not complete')
    signature = {'spatial_fingerprint': record['fingerprint'], 'artifacts': {}}
    for modality in ('T1w', 'T2w'):
        view = replace(sample, image=str(folder / f'{modality}_mni_n4.nii.gz'),
                       mask=str(folder / f'{modality}_mni_mask.nii.gz'))
        grid, extent = geometry_from_header(view.image, config.preprocessing)
        if list(grid) != [49, 15, 13] or list(extent) != [196, 240, 208]:
            raise ValueError('Unexpected MNI image geometry')
        # The existing adapter checks hashes, finite tensors and coordinates.
        volume = pre(view)
        if (list(volume.patches.shape) != [9555, 1024] or
                list(volume.metadata['grid_dhw']) != [49, 15, 13]):
            raise ValueError('Unexpected MNI patch shape/grid')
        for path in (view.image, view.mask):
            name = Path(path).name
            signature['artifacts'][name] = record['artifacts'][name]
        del volume
    return signature


def prepare_data(config, evaluation=None, *, include_training=True, emit_warnings=True):
    populations, summary = load_neuroqa(config, check_paths=False)
    policy = config.data.get('invalid_sample_policy', 'strict')
    groups = {}
    if include_training:
        groups.update(train=selected(config, populations['train'], 'train'),
                      val=selected(config, populations['val'], 'val'))
    if evaluation:
        groups['evaluation'] = evaluation_samples(populations, evaluation['split'],
                                                  evaluation.get('ids_file'), evaluation.get('limit'))
    report = {'policy': policy, 'population': summary, 'errors': [], 'warnings': [],
              'invalid_cases': {}, 'missing_cases': {}, 'skipped': {k: [] for k in groups},
              'requested_counts': {k: len(v) for k, v in groups.items()}}
    errors = report['errors']
    root = path_of(config, config.preprocessing.cache_dir).resolve()
    report['cache_dir'] = str(root)
    finalization = root / 'finalization.json'
    try:
        ready = json.loads(finalization.read_text()).get('ready_for_full_training', False) if finalization.is_file() else False
    except (OSError, ValueError) as exc:
        ready = False
        report['warnings'].append(f'Cannot read finalization: {exc}')
    report['qc_ready_for_full_training'] = ready
    if not ready:
        message = f'Preprocessing not globally finalized: {finalization}'
        if policy == 'strict':
            errors.append('Full training/evaluation requires ready_for_full_training=true: ' + message)
        else:
            report['warnings'].append(message)
    lineage = json.loads((path_of(config, config.data.manifest_dir) / 'lineage.json').read_text())
    if Path(lineage['preprocessing_root']).resolve() != root:
        errors.append('Manifest lineage preprocessing_root differs from selected cache; use a matching manifest')
    cases = {}
    for samples in groups.values():
        for sample in samples:
            if not Path(sample.image).resolve().is_relative_to(root / 'cases'):
                errors.append(f'Input outside selected preprocessing version: {sample.image}')
            cases.setdefault(sample.case_id, sample)
    pre = None
    try:
        pre = build_preprocessor(config)
        report['preprocessing_fingerprint'] = pre.version
        pending = _review_failures(root)
        if policy == 'include_unreviewed':
            audit_failures = _verify_unreviewed_v2(config, root)
            report['included_unreviewed_cases'] = {k:v for k,v in pending.items() if k in cases and k not in audit_failures}
            pending = audit_failures
    except (OSError, ValueError, KeyError) as exc:
        errors.append(f'Cannot initialize preprocessing protocol: {exc}')
        pending = {}
    signatures = {}
    can_scan = pre is not None and not any('matching manifest' in e or 'outside selected' in e for e in errors)
    if can_scan:
        # Read existing caches but prohibit new patch writes, even on --check-data.
        original_put = pre.cache.put
        pre.cache.put = lambda *a, **kw: None
        try:
            for index, (case, sample) in enumerate(sorted(cases.items()), 1):
                reasons = list(pending.get(case, []))
                folder = Path(sample.image).parent
                required = [folder / 'complete.json'] + [folder / f'{m}_mni_{suffix}.nii.gz'
                            for m in ('T1w', 'T2w') for suffix in ('n4', 'mask')]
                absent = [str(p) for p in required if not p.is_file()]
                if absent:
                    report['missing_cases'][case] = absent
                    reasons.append('Missing artifacts/completion: ' + ', '.join(Path(p).name for p in absent))
                if not reasons and not errors:
                    try:
                        signatures[case] = _check_case(config, pre, sample)
                    except (OSError, ValueError, KeyError, RuntimeError, EOFError, pickle.UnpicklingError) as exc:
                        if any(term in str(exc).lower() for term in ('out of memory', "can't allocate memory", 'cannot allocate memory')):
                            raise
                        reasons.append(f'{type(exc).__name__}: {exc}')
                if reasons:
                    report['invalid_cases'][case] = reasons
                if emit_warnings and index % 25 == 0:
                    print(f'VQA data checks: {index}/{len(cases)} cases', flush=True)
        finally:
            pre.cache.put = original_put
    if 'included_unreviewed_cases' in report:
        report['included_unreviewed_cases'] = {case: reasons for case, reasons in report['included_unreviewed_cases'].items()
                                               if case not in report['invalid_cases']}
        report['warnings'].append(f'User chose to include {len(report["included_unreviewed_cases"])} unreviewed QC-warning cases; results are experimental')
    kept = {}
    for split, samples in groups.items():
        bad = [s for s in samples if s.case_id in report['invalid_cases']]
        report[f'{split}_affected_qa'] = len({s.qa_id for s in bad})
        if policy in ('skip', 'include_unreviewed'):
            report['skipped'][split] = [_skip_row(s, split, report['invalid_cases'][s.case_id]) for s in bad]
            kept[split] = [s for s in samples if s.case_id not in report['invalid_cases']]
        else:
            kept[split] = samples
    if report['invalid_cases'] and policy not in ('skip', 'include_unreviewed'):
        errors.append(f'{len(report["invalid_cases"])} cases failed checks; {policy} mode does not skip samples')
    if include_training:
        if not kept['train']:
            errors.append('No usable training QA remain')
        else:
            try:
                plan = balanced_epoch(kept['train'], config.seed, 0, config.data.get('modality_sampling', 'legacy_exact'))
                report['epoch_modalities'] = dict(Counter(s.sequence for s in plan))
                report['dual_eligible_modalities'] = dict(Counter(s.sequence for s in plan if len(s.row['allowed_modalities']) == 2))
                steps = math.ceil(len(plan) / config.training.effective_batch_size)
                total = min(steps * config.training.max_epochs, config.training.max_steps or math.inf)
                report.update(updates_per_epoch=steps, total_updates=total)
                if config.training.stage == 'joint_lora':
                    report['schedule'] = resolve_joint_schedule(config.training, total).to_dict()
            except ValueError as exc:
                errors.append(str(exc))
        report.update(train_qa=len(kept['train']), val_views=len(kept['val']))
        if not any(s.row['question_type'] in ('MCQ', 'YN') for s in kept['val']):
            errors.append('No usable closed validation questions remain for best-checkpoint selection')
    for split, samples in kept.items():
        if not samples:
            errors.append(f'No usable {split} samples remain')
    if evaluation:
        report['evaluation_views'] = len(kept['evaluation'])
        if 'shuffle' in evaluation['controls'] and len({s.patient_id for s in kept['evaluation']}) < 2:
            errors.append('Shuffle control requires at least two patients after filtering')
    report['retained_ids'] = {k: [s.qa_id if k == 'train' else s.id for s in v] for k, v in kept.items()}
    def fingerprint(keys):
        return digest({'policy': policy, 'modality_sampling': config.data.get('modality_sampling', 'legacy_exact'), 'ids': {k: report['retained_ids'][k] for k in keys},
                       'cases': {s.case_id: signatures.get(s.case_id) for k in keys for s in kept[k]}})
    if include_training:
        report['selection_fingerprint'] = fingerprint(('train', 'val'))
    if evaluation:
        report['evaluation_selection_fingerprint'] = fingerprint(('evaluation',))
    if not errors:
        views = [s.view(m) for s in kept.get('train', []) for m in s.row['allowed_modalities']]
        views += kept.get('val', []) + kept.get('evaluation', [])
        try:
            report['geometry'] = prepare_budget(config, views)
        except (OSError, ValueError, RuntimeError) as exc:
            errors.append(f'Incompatible input geometry: {exc}')
    report['scope'] = training_scope(config, report)
    report['status'] = 'passed' if not errors else 'blocked'
    if emit_warnings:
        for warning in report['warnings']:
            print('WARNING VQA: ' + warning, flush=True)
        for case, reasons in report['invalid_cases'].items():
            counts = {k: sum(s.case_id == case for s in v) for k, v in groups.items()}
            print(f'WARNING VQA {policy}: {case}; samples={counts}; {"; ".join(reasons)}', flush=True)
    return kept, report, pre, summary


def inspect_data(config, evaluation=None):
    return prepare_data(config, evaluation)[1]


def ensure_same_selection(expected, actual, key='selection_fingerprint'):
    if expected.get(key) is not None and expected[key] != actual.get(key):
        raise ValueError('VQA data selection/artifacts changed since preflight or checkpoint; start a new run')


def save_data_report(folder, report):
    write_json(Path(folder) / 'data_preflight.json', report)
    write_json(Path(folder) / 'skipped_samples.json', {
        'policy': report['policy'], 'warnings': report['warnings'],
        'invalid_cases': report['invalid_cases'], 'skipped': report['skipped'],
        'requested_counts': report['requested_counts'],
        'retained_counts': {k: len(v) for k, v in report['retained_ids'].items()}})


def compact_report(report):
    return {k: v for k, v in report.items() if k not in ('retained_ids', 'skipped')}
