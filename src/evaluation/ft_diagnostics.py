"""Persist generation lengths and GREEN's existing counts without rescoring."""
from collections import Counter
import csv
import math
from pathlib import Path

from src.config import digest
from src.evaluation.clinical import read_jsonl, validate_rows, write_json


ERROR_KEYS = ('false_finding', 'omitted_finding', 'wrong_location',
              'severity_or_size', 'unsupported_comparison', 'omitted_comparison')


def summarize_evaluation(directory, result, expected_ids_hash):
    directory = Path(directory)
    rows = read_jsonl(directory / 'generation/predictions.jsonl')
    scored = read_jsonl(directory / 'scoring/clinical_scores.jsonl')
    validate_rows(rows)
    validate_rows(scored, [row['id'] for row in rows])
    if digest([row['id'] for row in rows]) != expected_ids_hash:
        raise ValueError('Evaluation diagnostics: test selection/order differs from plan')
    by_id = {row['id']: row for row in scored}
    samples, valid, statuses = [], [], Counter()
    for row in rows:
        score = by_id[row['id']]
        for field in ('checkpoint_id', 'patient_id', 'reference', 'prediction', 'sequence', 'split'):
            if score[field] != row[field]:
                raise ValueError(f'Evaluation diagnostics: scoring/generation {field} mismatch')
        if row['checkpoint_id'] != result['checkpoint_sha256']:
            raise ValueError('Evaluation diagnostics: checkpoint mismatch')
        green = score['clinical']['green']
        statuses[green['status']] += 1
        item = {key: row[key] for key in ('id', 'patient_id', 'cohort', 'sequence',
                                         'generated_token_count', 'hit_generation_limit', 'empty_prediction')}
        item.update(green_status=green['status'], green=None, matched_findings=None,
                    **dict.fromkeys(ERROR_KEYS))
        if green['status'] == 'ok':
            details = green['details']
            counts, matches = details['significant_errors'], details['matched_findings']
            if len(counts) != 6 or any(type(v) not in (int, float) or not math.isfinite(v) or v < 0
                                       for v in [matches, *counts]):
                raise ValueError('Invalid GREEN matched/significant-error counts')
            value = green['values']['green']
            formula = matches / (matches + sum(counts)) if matches else 0.
            if not math.isclose(value, formula, abs_tol=1e-10, rel_tol=1e-10):
                raise ValueError('GREEN score/count formula mismatch')
            item.update(green=value, matched_findings=matches, **dict(zip(ERROR_KEYS, counts)))
            valid.append(item)
        samples.append(item)
    if not valid:
        raise ValueError('No successful GREEN records')
    means = {key: sum(row[key] for row in valid) / len(valid)
             for key in ('green', 'matched_findings', *ERROR_KEYS)}
    if not math.isclose(means['green'], result['overall']['green'], abs_tol=1e-10, rel_tol=1e-10):
        raise ValueError('GREEN valid-only mean differs from published metric')
    summary = {'checkpoint_sha256': result['checkpoint_sha256'], 'samples': len(rows),
               'expected_ids_sha256': expected_ids_hash,
               'mean_generated_tokens': sum(row['generated_token_count'] for row in rows) / len(rows),
               'empty_predictions': sum(bool(row['empty_prediction']) for row in rows),
               'hit_generation_limit': sum(bool(row['hit_generation_limit']) for row in rows),
               'green_valid': len(valid), 'green_failed': len(rows) - len(valid),
               'green_statuses': dict(statuses), 'green_valid_only_means': means,
               'green_error_order': list(ERROR_KEYS),
               'note': 'GREEN counts are automatic text-judge outputs, not human-verified imaging findings.'}
    write_json(directory / 'generation_green_diagnostics.json', summary)
    path = directory / 'generation_green_samples.csv'
    with path.open('w', newline='', encoding='utf-8-sig') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(samples[0]))
        writer.writeheader(); writer.writerows(samples)
    return summary
