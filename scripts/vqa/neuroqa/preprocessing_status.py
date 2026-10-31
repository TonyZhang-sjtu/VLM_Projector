#!/usr/bin/env python3
"""Summarize completed cases, failures, current stages, and expected image coverage."""
from collections import Counter
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from src.data.neuroqa_manifest import read_rows, write_json


def main():
    root = ROOT / 'assets/preprocessing/neuroqa_mni1mm_v1'
    index = read_rows(ROOT / 'assets/data/NeuroQA_derived/official_holdout_patient_guard_v1/source_index.jsonl')
    completed, active, pending, failed = [], {}, [], {}
    for row in index:
        folder = root / 'cases' / row['case_id']
        if (folder / 'complete.json').exists():
            completed.append(row['case_id'])
        elif (folder / 'status.json').exists():
            active[row['case_id']] = json.loads((folder / 'status.json').read_text())['stage']
        else:
            pending.append(row['case_id'])
    for path in (root / 'batches').glob('*.json'):
        for row in json.loads(path.read_text()).get('failed', []):
            if row['case_id'] not in completed:
                failed[row['case_id']] = row['error']
    report = {'expected_cases': len(index), 'expected_volumes': 2*len(index),
              'complete_cases': len(completed), 'complete_volumes': 2*len(completed),
              'all_complete': len(completed) == len(index), 'stage_counts': dict(Counter(active.values())),
              'incomplete_stages': active, 'pending_cases': len(pending), 'failed': failed,
              'note': 'Stage files alone do not prove a worker is still alive; inspect the PID/log when interrupted.'}
    write_json(root / 'inventory_status.json', report)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
