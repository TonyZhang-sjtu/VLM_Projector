"""Full-coverage wrapper around the pinned MIT-licensed NeuroQA scorer.

neuroqa_official.py is an unmodified snapshot of mhabbasiit/neuroqa/eval/evaluate.py.
Its test-label majority-template statistic is kept only in the official export;
it is NOT a trained text-only baseline. Our controls run the actual VLM.
"""
from collections import Counter, defaultdict
import json
from pathlib import Path

import numpy as np
import torch

from src.config import digest, file_hash
from src.data.neuroqa_manifest import write_json
from src.evaluation.neuroqa_official import evaluate as official_evaluate, normalize, exact_match, token_f1, rouge_l


def score_row(sample, prediction):
    row, value = sample.row, normalize(prediction)
    kind, gold = row['question_type'], normalize(row['answer'])
    if kind == 'Open':
        return {'valid': bool(value), 'em': exact_match(prediction, row['answer']),
                'f1': token_f1(prediction, row['answer']), 'rouge_l': rouge_l(prediction, row['answer'])}
    if kind == 'YN':
        return {'valid': value in ('yes', 'no'), 'correct': float(value == gold)}
    options = [normalize(v) for v in row['options']]
    letter = len(value) == 1 and value in 'abcd' and ord(value)-97 < len(options)
    parsed = options[ord(value)-97] if letter else value
    return {'valid': bool(letter or value in options), 'correct': float(parsed == gold), 'parsed': parsed}


def aggregate(rows):
    closed = [r for r in rows if r['question_type'] != 'Open']
    opened = [r for r in rows if r['question_type'] == 'Open']
    result = {'samples': len(rows), 'closed_count': len(closed), 'open_count': len(opened),
              'invalid_count': sum(not r['score']['valid'] for r in rows), 'units': 'fraction_0_to_1'}
    if closed:
        result['closed_accuracy'] = sum(r['score']['correct'] for r in closed) / len(closed)
    if opened:
        for name in ('em', 'f1', 'rouge_l'):
            result['open_' + name] = sum(r['score'][name] for r in opened) / len(opened)
    if rows and all('qa_ce' in r for r in rows):
        # Views from the same QA receive a total weight of one.
        questions = defaultdict(list)
        for row in rows:
            questions[row['neuroqa_id']].append(row['qa_ce'])
        result['qa_ce'] = float(np.mean([np.mean(v) for v in questions.values()]))
    return result


def evaluate_rows(samples, rows, bootstrap=0, seed=42):
    expected = [s.id for s in samples]
    observed = [r['id'] for r in rows]
    if len(set(observed)) != len(observed) or len(set(expected)) != len(expected) or set(expected) != set(observed):
        raise ValueError('Prediction coverage mismatch: missing/duplicate/extra input-view IDs')
    lookup = {s.id: s for s in samples}
    scored = []
    for row in rows:
        sample = lookup[row['id']]
        scored.append({**row, 'neuroqa_id': sample.qa_id, 'patient_id': sample.patient_id,
                       'sequence': sample.sequence, 'dataset': sample.row['dataset'],
                       **{k: sample.row[k] for k in ('question_type', 'category', 'human_assessable', 'image_dependency')},
                       'reference': sample.row['answer'], 'score': score_row(sample, row['prediction'])})
    result = {'overall': aggregate(scored), 'strata': {}, 'scorer_sha256': file_hash(Path(__file__).with_name('neuroqa_official.py'))}
    for field in ('sequence', 'dataset', 'question_type', 'category', 'human_assessable', 'image_dependency'):
        groups = defaultdict(list)
        for row in scored:
            groups[str(row[field])].append(row)
        result['strata'][field] = {k: aggregate(v) for k, v in sorted(groups.items())}
    # Keep the predeclared closed micro accuracy as the selection metric.
    scores = [result['strata']['sequence'].get(s, {}).get('closed_accuracy') for s in ('T1W', 'T2W')]
    result['modality_macro_accuracy'] = float(np.mean(scores)) if all(s is not None for s in scores) else None
    result['selection_accuracy'] = result['overall'].get('closed_accuracy')
    category_scores = [v['closed_accuracy'] for v in result['strata']['category'].values() if 'closed_accuracy' in v]
    result['category_macro_accuracy'] = float(np.mean(category_scores)) if category_scores else None
    views = {seq: {r['neuroqa_id']: r for r in scored if r['sequence'] == seq} for seq in ('T1W','T2W')}
    common = sorted(set(views['T1W']) & set(views['T2W']))
    result['paired_modalities'] = {'common_qa': len(common),
        **{seq: aggregate([views[seq][qid] for qid in common]) for seq in views}}
    result['wrapper_sha256'] = file_hash(__file__)
    patients = defaultdict(list)
    for row in scored:
        if row['question_type'] != 'Open':
            patients[row['patient_id']].append(row['score']['correct'])
    if patients:
        result['patient_macro_accuracy'] = float(np.mean([np.mean(v) for v in patients.values()]))
    if bootstrap and patients:
        rng = np.random.default_rng(seed)
        groups = list(patients.values())
        totals = np.array([sum(v) for v in groups])
        counts = np.array([len(v) for v in groups])
        estimates = []
        for _ in range(bootstrap):
            indices = rng.integers(0, len(groups), len(groups))
            estimates.append(totals[indices].sum() / counts[indices].sum())
        result['patient_bootstrap_closed_micro_ci95'] = np.quantile(estimates, [.025, .975]).tolist()
    # Official scores are computed separately so identical QA IDs never overwrite
    # the other modality. Units here follow upstream (percentage).
    result['official_per_sequence'] = {}
    for seq in ('T1W', 'T2W'):
        selected = [r for r in scored if r['sequence'] == seq]
        gt = [lookup[r['id']].row for r in selected]
        result['official_per_sequence'][seq] = official_evaluate(gt, {r['neuroqa_id']: r['prediction'] for r in selected})
    return result, scored


@torch.no_grad()
def predict(model, samples, preprocessor, *, control=None, compute_ce=True, progress=None):
    """Independent decoding, with deterministic same-sequence other-patient donors."""
    was_training = model.training
    model.eval()
    donors = {}
    if control == 'shuffle':
        for sample in samples:
            candidates = [s for s in samples if s.sequence == sample.sequence and s.patient_id != sample.patient_id]
            if not candidates:
                raise ValueError('Shuffle requires another patient in each evaluated sequence')
            donors[sample.id] = min(candidates, key=lambda s: digest([sample.id, s.id]))
    rows = []
    try:
        for index, sample in enumerate(samples):
            donor = donors.get(sample.id, sample)
            item = {'sample': sample, 'volume': None if control == 'text' else preprocessor(donor)}
            with torch.autocast('cuda', dtype=torch.bfloat16):
                value = model.generate([item], control=None if control == 'shuffle' else control,
                                       max_new_tokens=min(model.decoder.task.generation_budget(sample),
                                                          model.config.generation.max_new_tokens))[0]
                if compute_ce and control is None:
                    loss = model([item])
                    value.update(qa_ce=float(loss['example_loss_sum']), answer_tokens=loss['answer_tokens'])
            if 'qa_ce' in value and not np.isfinite(value['qa_ce']):
                raise FloatingPointError('Nonfinite validation QA CE')
            value.update(id=sample.id, control=control or 'normal', donor_id=donor.id if control == 'shuffle' else None)
            rows.append(value)
            if progress:
                progress(index + 1, len(samples))
    finally:
        model.train(was_training)
    return rows


def save_predictions(folder, samples, rows, bootstrap=0, seed=42):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    metrics, scored = evaluate_rows(samples, rows, bootstrap, seed)
    (folder / 'predictions.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in scored))
    write_json(folder / 'metrics.json', metrics)
    for name, seq in [('T1w', 'T1W'), ('T2w', 'T2W')]:
        write_json(folder / f'predictions_{name}_official.json',
                   [{'neuroqa_id': r['neuroqa_id'], 'prediction': r['prediction']} for r in scored if r['sequence'] == seq])
    return metrics
