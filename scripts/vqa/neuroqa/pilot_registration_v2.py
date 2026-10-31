#!/usr/bin/env python3
"""Measure B/C/D registration candidates on the frozen 32-case train pilot."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from src.data.neuroqa_manifest import read_rows, write_json
from src.preprocessing.neuroqa_registration_v2 import CANDIDATES, run_case
from src.preprocessing.neuroqa_smri import TEMPLATE


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--candidate', action='append', choices=sorted(CANDIDATES))
    parser.add_argument('--case', action='append')
    args = parser.parse_args()
    if args.workers < 1:
        parser.error('workers must be positive')
    pilot = json.loads((ROOT / 'results/vqa/neuroqa/preprocessing_v2_plan/pilot_manifest.json').read_text())
    index = read_rows(ROOT / 'assets/data/NeuroQA_derived/official_holdout_patient_guard_v1/source_index.jsonl')
    ids = set(pilot['case_ids'])
    if len(ids) != 32:
        raise ValueError('Unexpected pilot manifest')
    if args.case:
        if not set(args.case) <= ids or len(args.case) != len(set(args.case)):
            parser.error('Case selection must be unique and in the frozen pilot')
        ids = set(args.case)
    sources = [row for row in index if row['case_id'] in ids]
    if len(sources) != len(ids):
        raise ValueError('Pilot source index is incomplete')
    root = ROOT / 'results/vqa/neuroqa/preprocessing_v2_plan/pilot'
    template = ROOT / 'assets/preprocessing/neuroqa_mni1mm_v1/templates' / TEMPLATE
    chosen = args.candidate or list(CANDIDATES)
    jobs = [(row, candidate) for row in sources for candidate in chosen]
    report = {'requested': len(jobs), 'successful': [], 'failed': []}
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_case, row, root, template, candidate): (row['case_id'], candidate)
                   for row, candidate in jobs}
        for future in as_completed(futures):
            try:
                item = future.result()
                report['successful'].append(item)
            except Exception as exc:
                cid, candidate = futures[future]
                item = {'case_id': cid, 'candidate': candidate, 'error': repr(exc)}
                report['failed'].append(item)
            print(json.dumps(item), flush=True)
            write_json(root / 'batch.json', report)
    if report['failed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
