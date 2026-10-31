#!/usr/bin/env python3
"""Build the official-guarded QA manifests and resume offline CPU preprocessing."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
import sys
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from src.data.neuroqa_manifest import build_manifests, read_rows, write_json
from src.preprocessing.neuroqa_smri import download_template, process_case


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--output', type=Path, default=ROOT / 'assets/preprocessing/neuroqa_mni1mm_v1')
    p.add_argument('--manifest-dir', type=Path, default=ROOT / 'assets/data/NeuroQA_derived/official_holdout_patient_guard_v1')
    p.add_argument('--workers', type=int, default=2)
    p.add_argument('--threads', type=int, default=2)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--case', action='append', default=[])
    p.add_argument('--exclude-case', action='append', default=[])
    p.add_argument('--report-name', default=datetime.now(timezone.utc).strftime('batch_%Y%m%dT%H%M%SZ'))
    p.add_argument('--limit', type=int)
    p.add_argument('--inventory-only', action='store_true')
    args = p.parse_args()
    if args.workers < 1 or args.threads < 1:
        p.error('workers and threads must be positive')
    lineage = build_manifests(ROOT / 'assets/data', args.manifest_dir, args.output)
    print(json.dumps(lineage), flush=True)
    if args.inventory_only:
        return
    template = download_template(args.output)
    rows = read_rows(args.manifest_dir / 'source_index.jsonl')
    if args.case:
        rows = [r for r in rows if r['case_id'] in args.case]
        if len(rows) != len(set(args.case)):
            p.error('Unknown or duplicate case selector')
    if args.limit:
        rows = rows[:args.limit]
    rows = [r for r in rows if r['case_id'] not in args.exclude_case]
    report = {'requested': len(rows), 'successful': [], 'failed': []}
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(process_case, r, args.output, template, args.seed, args.threads): r for r in rows}
        for future in as_completed(futures):
            try:
                result = future.result()
                report['successful'].append(result)
            except Exception as exc:
                result = {'case_id': futures[future]['case_id'], 'error': repr(exc)}
                report['failed'].append(result)
            print(json.dumps(result), flush=True)
            write_json(args.output / 'batches' / f'{args.report_name}.json', report)
    if report['failed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
