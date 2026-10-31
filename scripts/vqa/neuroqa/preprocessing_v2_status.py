#!/usr/bin/env python3
"""Report v2 pilot, full preprocessing and verification status without mutation."""
import json
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).resolve().parents[3]


def main():
    plan = ROOT/'results/vqa/neuroqa/preprocessing_v2_plan'
    root = ROOT/'assets/preprocessing/neuroqa_mni1mm_v2'
    result = {'pilot':{}, 'production':{}, 'full_training_ready':False}
    coordinator = plan/'pipeline_status.json'
    if coordinator.exists():
        result['coordinator'] = json.loads(coordinator.read_text())
    processes = {'coordinator_pids': [], 'preprocessing_pids': []}
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit():
            continue
        try:
            args = (proc/'cmdline').read_bytes().split(b'\0')
        except (OSError, ProcessLookupError):
            continue
        if any(a.endswith(b'/continue_v2_pipeline.py') for a in args):
            processes['coordinator_pids'].append(int(proc.name))
        if any(a.endswith(b'/preprocess_smri_v2.py') for a in args):
            processes['preprocessing_pids'].append(int(proc.name))
    result['processes'] = processes
    if (result.get('coordinator', {}).get('phase') == 'full_preprocessing'
            and not processes['preprocessing_pids']):
        result['process_warning'] = ('Last recorded phase is full_preprocessing but no preprocessing process is running; '
                                     'the recorded phase and per-case stages may be stale.')
    smoke = ROOT/'runs/vqa/neuroqa/smoke/v2_sandbox_full_matrix_20261030/report.json'
    if smoke.exists():
        measurements = json.loads(smoke.read_text())
        result['sandbox_projector_smoke'] = {'passed':sum(x.get('status') == 'passed' for x in measurements),
                                            'expected_compressed_configurations':55,
                                            'scope':'sandbox_engineering_only'}
    batch = plan/'pilot/batch.json'
    if batch.exists():
        d = json.loads(batch.read_text())
        result['pilot'] = {'requested':d['requested'], 'measured':len(d['successful']),
                           'measured_by_candidate':dict(Counter(x['candidate'] for x in d['successful'])),
                           'failed':d['failed']}
    protocol = root/'protocol.json'
    result['production']['protocol_frozen'] = protocol.exists() and json.loads(protocol.read_text()).get('status') == 'frozen'
    cases = root/'cases'
    result['production']['complete_cases'] = len(list(cases.glob('*/complete.json'))) if cases.exists() else 0
    result['production']['expected_cases'] = 414
    stages = Counter()
    review_cases = []
    if cases.exists():
        for p in cases.glob('*/status.json'):
            state = json.loads(p.read_text())
            stage = state.get('stage','unknown')
            stages[stage] += 1
            if stage == 'qc_review_required':
                review_cases.append(p.parent.name)
    result['production']['stage_counts'] = dict(stages)
    result['production']['qc_review_required_cases'] = sorted(review_cases)
    report = root/'batches/full_dataset.json'
    if report.exists():
        d = json.loads(report.read_text())
        result['production']['batch'] = {'requested':d['requested'],'successful':len(d['successful']),'failed':d['failed']}
    queue = root/'qc_review_queue.json'
    if queue.exists():
        result['production']['flagged_review_cases'] = json.loads(queue.read_text())['case_count']
    final = root/'finalization.json'
    if final.exists():
        d = json.loads(final.read_text())
        result['production']['verified_volumes'] = len(d.get('verified_volumes',[]))
        result['production']['failed_volumes'] = len(d.get('failed',[]))
        result['full_training_ready'] = d.get('ready_for_full_training') is True
    print(json.dumps(result,indent=2,ensure_ascii=False))


if __name__ == '__main__':
    main()
