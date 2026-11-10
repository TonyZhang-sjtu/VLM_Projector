"""Verify installed official code/assets and replay PROVIDED labels, without an LLM.

This checks deterministic scoring only. Few-shot examples are demonstrations,
not held-out evaluation cases; their scores are never dataset results.
"""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess

COMMIT = 'bc523a68c29b1807fe8e5c4c32d39cf4ce111e6c'
ROOT = Path(__file__).resolve().parents[2]
SOURCE = Path(__file__).resolve().parent / 'runtime/RadMatch'


def installation_record():
    import radmatch
    from radmatch.llm_utils import prompts
    actual = subprocess.check_output(['git', '-C', str(SOURCE), 'rev-parse', 'HEAD'], text=True).strip()
    if actual != COMMIT:
        raise ValueError('Wrong upstream RadMatch revision')
    if subprocess.check_output(['git','-C',str(SOURCE),'status','--porcelain','--untracked-files=no'],text=True).strip():
        raise ValueError('Upstream tracked files were modified')
    installed = Path(radmatch.__file__).resolve().parent
    hashes = {}
    for file in sorted((SOURCE/'src/radmatch').rglob('*')):
        if file.is_file() and '__pycache__' not in file.parts:
            rel = file.relative_to(SOURCE/'src/radmatch')
            if (installed/rel).read_bytes() != file.read_bytes():
                raise ValueError(f'Installed code/assets differ from reviewed source: {rel}')
            hashes[str(rel)] = hashlib.sha256(file.read_bytes()).hexdigest()
    loaders = [prompts.load_extraction_fewshot, prompts.load_matching_fewshot,
               prompts.load_attribute_errors_fewshot]
    counts = dict(zip(['extraction', 'matching', 'attributes'], [len(f('brain-mr')) for f in loaders]))
    if counts != dict(extraction=3, matching=3, attributes=3):
        raise ValueError(f'brain-mr did not load completely: {counts}')
    return dict(commit=COMMIT, version=importlib.metadata.version('radmatch'),
                source=str(SOURCE), installed=str(installed), file_sha256=hashes,
                fewshot='brain-mr', loaded_examples=counts)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    record = installation_record()
    from radmatch.finding_extraction.extract_utils import validate_and_normalize_finding
    from radmatch.llm_utils import prompts
    from radmatch.scoring import comparators, metrics
    from radmatch.scoring.pipeline import build_per_report_summary
    replay = []
    for i, example in enumerate(prompts.load_attribute_errors_fewshot('brain-mr'), 1):
        pairs = example['user']['pairs']
        provided_errors = example['assistant']['errors_per_match']
        assert len(pairs) == len(provided_errors)
        records = []
        for pair, errors in zip(pairs, provided_errors):
            p = validate_and_normalize_finding(pair['pred_finding'])
            g = validate_and_normalize_finding(pair['gt_finding'])
            records.append(metrics.build_muc_record(
                dict(pred_id=p['finding_id'], gt_id=g['finding_id']), p, g,
                comparators.compute_structured_errors(p, g), errors))
        replay.append(dict(example=i, provided_pairs_only=True,
                           summary=build_per_report_summary(records, [], [])))
    # Published example 2 explicitly labels right-vs-left putamen as major.
    assert replay[1]['summary']['attribute_breakdown']['location']['major'] == 1
    result = dict(installation=record, check='provided_attribute_labels_deterministic_replay',
                  llm_calls=0, patient_reports_evaluated=0, reliability_established=False,
                  replay=replay)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(dict(status='passed', loaded_examples=record['loaded_examples'],
                         llm_calls=0, patient_reports_evaluated=0)))


if __name__ == '__main__':
    main()
