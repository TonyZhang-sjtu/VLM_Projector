#!/usr/bin/env python3
"""Read-only QA evaluation index. Never combine report metrics or smoke accuracy."""
import argparse
import csv
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--root', type=Path, default=Path('results/vqa/neuroqa'))
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--include-filtered', action='store_true', help='Also index filtered evaluations, with explicit scope/coverage columns')
    p.add_argument('--include-unreviewed', action='store_true', help='Explicitly index evaluations that include unreviewed QC warnings')
    args = p.parse_args()
    rows = []
    for path in sorted(args.root.rglob('metrics.json')):
        metric = json.loads(path.read_text(encoding='utf-8'))
        scopes = ['full_split']
        if args.include_filtered: scopes.append('filtered_evaluation')
        if args.include_unreviewed: scopes.append('unreviewed_evaluation')
        if metric.get('scope') not in scopes or not (path.parent / 'provenance.json').exists():
            continue
        proof = json.loads((path.parent / 'provenance.json').read_text())
        if proof.get('decoder_loss') != 'answer_only_causal_ce_per_qa_mean_v1':
            continue
        rows.append({'evaluation': str(path.parent.resolve()), 'contract_id': proof['contract_id'],
                     'preprocessing_version': proof.get('preprocessing'), 'target_version': proof.get('target_version'),
                     'projector': proof['projector']['name'],
                     'ratio': proof['projector']['resolved_budget']['target_ratio'],
                     'M': proof['projector']['resolved_budget']['expected_output_tokens'],
                     'split': metric['split'], 'control': metric['control'],
                     'scope': metric['scope'], 'training_scope': metric.get('training_scope'),
                     'data_policy': metric.get('data_policy'), 'population_size': metric.get('population_size'),
                     'skipped_views': metric.get('skipped_views', 0),
                     'closed_accuracy': metric['overall'].get('closed_accuracy'),
                     'open_f1': metric['overall'].get('open_f1'), 'samples': metric['overall']['samples']})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('w', newline='', encoding='utf-8') as handle:
        fields = list(rows[0]) if rows else ['evaluation','contract_id','preprocessing_version','target_version','projector','ratio','M','split','control','scope','training_scope','data_policy','population_size','skipped_views','closed_accuracy','open_f1','samples']
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f'{len(rows)} QA evaluations indexed in {args.output}; include_filtered={args.include_filtered}; include_unreviewed={args.include_unreviewed}')


if __name__ == '__main__':
    main()
