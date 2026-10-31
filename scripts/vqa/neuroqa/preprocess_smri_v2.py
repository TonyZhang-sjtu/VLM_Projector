#!/usr/bin/env python3
"""Process an already-frozen NeuroQA v2 protocol without touching v1."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from src.data.neuroqa_manifest import read_rows, write_json
from src.config import file_hash
from src.preprocessing import neuroqa_smri
from src.preprocessing.neuroqa_registration_v2 import CANDIDATES
from src.preprocessing.neuroqa_smri_v2 import process_case


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--case', action='append')
    p.add_argument('--report-name', default='full_dataset')
    args = p.parse_args()
    if args.workers < 1 or Path(args.report_name).name != args.report_name:
        p.error('Invalid workers/report name')
    root = ROOT/'assets/preprocessing/neuroqa_mni1mm_v2'
    if not (root/'protocol.json').is_file():
        raise RuntimeError('V2 protocol is not yet frozen')
    protocol = json.loads((root/'protocol.json').read_text())
    if protocol.get('reference_helpers_sha256') != file_hash(neuroqa_smri.__file__):
        raise RuntimeError('Shared preprocessing helpers differ from the frozen protocol')
    if protocol.get('candidate_settings') != {name:CANDIDATES[name] for name in ('C','D')}:
        raise RuntimeError('Candidate optimizer settings differ from the frozen protocol')
    rows = read_rows(ROOT/'assets/data/NeuroQA_derived/official_holdout_patient_guard_v1_mni_v2/source_index.jsonl')
    if args.case:
        selected = set(args.case)
        if len(selected) != len(args.case) or not selected <= {r['case_id'] for r in rows}:
            p.error('Unknown or duplicate case ID')
        rows = [r for r in rows if r['case_id'] in selected]
    report = {'requested':len(rows),'successful':[],'failed':[]}
    report_file = root/'batches'/f'{args.report_name}.json'
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(process_case,row,root):row['case_id'] for row in rows}
        for future in as_completed(futures):
            try:
                item = future.result()
                report['successful'].append(item)
            except Exception as exc:
                item = {'case_id':futures[future], 'error':repr(exc)}
                report['failed'].append(item)
            print(json.dumps(item),flush=True)
            write_json(report_file,report)
    if report['failed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
