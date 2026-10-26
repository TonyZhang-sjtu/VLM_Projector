"""Reference-blind prediction extraction and fixed-denominator fact metrics."""
from collections import defaultdict
import json
from pathlib import Path

from .extract import extract
from .build import refine_facts, make_row, write_rows
from .schema import FIELDS


def evaluate_predictions(predictions, references):
    ids = [r['id'] for r in predictions]
    if len(set(ids)) != len(ids) or set(ids) != set(references):
        raise ValueError('Prediction/reference population mismatch')
    cells = defaultdict(lambda: dict(total=0, correct=0, missing=0, contradictory=0, unresolved=0))
    records = []
    group_total = group_correct = unresolved_reports = labelled_reports = 0
    for prediction in predictions:
        gold = references[prediction['id']]
        # Deliberately build a NEW object containing no reference answer/labels.
        source = {k: prediction[k] for k in ('id', 'patient_id', 'cohort', 'sequence', 'split')}
        source.update(finding=prediction['prediction'], text_audit_status='generated_text')
        extracted = extract(source)
        facts, rejected = refine_facts(source, extracted['facts'])
        predicted = {f: v['value'] for f, v in facts.items()}
        unresolved = bool(extracted.get('skip_reason'))
        unresolved_reports += unresolved
        labelled_reports += bool(gold['facts'])
        decisions = {}
        for field, fact in gold['facts'].items():
            expected, observed = fact['value'], predicted.get(field)
            status = 'correct' if observed == expected else ('contradictory' if observed is not None else
                      'unresolved' if unresolved else 'missing')
            cell = cells[(field, expected)]
            cell['total'] += 1; cell[status] += 1
            decisions[field] = dict(expected=expected, predicted=observed, status=status)
        groups = []
        for group in gold['groups']:
            if not group['joint_eligible']:
                continue
            correct = all(decisions[f]['status'] == 'correct' for f in group['fields'])
            group_total += 1; group_correct += correct
            groups.append(dict(group_id=group['group_id'], fields=group['fields'], correct=correct))
        records.append(dict(id=prediction['id'], patient_id=prediction['patient_id'], decisions=decisions,
                            joint_groups=groups, extracted_facts=facts,
                            extraction_skip=extracted.get('skip_reason'), extraction_rejected=extracted['rejected']+rejected))
    by_field = {}
    for field in FIELDS:
        classes = {value: counts for (f, value), counts in cells.items() if f == field}
        if not classes:
            continue
        totals = {k: sum(v[k] for v in classes.values()) for k in next(iter(classes.values()))}
        by_field[field] = dict(**totals, accuracy=totals['correct']/totals['total'],
                              balanced_accuracy=sum(v['correct']/v['total'] for v in classes.values())/len(classes),
                              classes=classes)
    count = sum(c['total'] for c in cells.values())
    metrics = dict(reports=len(predictions), labelled_reports=labelled_reports, known_facts=count,
                   unlabelled_reports=len(predictions)-labelled_reports,
                   extraction_unresolved_reports=unresolved_reports, fields=by_field,
                   selection_metric='macro_field_balanced_accuracy',
                   macro_field_balanced_accuracy=(sum(v['balanced_accuracy'] for v in by_field.values())/len(by_field)
                                                    if by_field else None),
                   fact_accuracy=sum(c['correct'] for c in cells.values())/count if count else None,
                   missing=sum(c['missing'] for c in cells.values()),
                   contradictory=sum(c['contradictory'] for c in cells.values()),
                   unresolved=sum(c['unresolved'] for c in cells.values()),
                   joint_groups=group_total, joint_groups_correct=group_correct,
                   single_lesion_joint_known_accuracy=group_correct/group_total if group_total else None,
                   mean_generated_tokens=sum(p.get('generated_token_count', 0) for p in predictions)/max(1, len(predictions)),
                   generation_limit_count=sum(p.get('hit_generation_limit', False) for p in predictions),
                   scope='known_reference_facts_only_not_exhaustive_clinical_accuracy',
                   missing_prediction_policy='counts_as_incorrect_not_excluded',
                   entity_policy='conservative_single_lesion_text_heuristic_not_multi_lesion_binding')
    return metrics, records


def run_validation(model, dataset):
    import torch
    from src.data.dataset import VolumeDataset
    from src.evaluation.generate import predict
    from src.training.checkpoint import rng_state, restore_rng
    from src.config import path_of
    settings = model.output_settings
    state, was_training = rng_state(), model.training
    try:
        limit = settings['validation_limit']
        selected = dataset if limit == 0 else VolumeDataset(dataset.samples[:limit], dataset.preprocessor)
        step = model.supervision_step + 1
        output = path_of(model.config, model.config.training.run_dir)/'validation_report_facts'
        output.mkdir(parents=True, exist_ok=True)
        stem = output/f'step_{step:08d}'
        predictions = predict(model, selected, stem.with_suffix('.predictions.jsonl'),
                              f'candidate_step_{step}', batch_size=1, position=2)
        references = {s.id: model.output_store.get(s, split='val') for s in selected.samples}
        metrics, records = evaluate_predictions(predictions, references)
        metrics.update(step=step, population=len(dataset), evaluated=len(selected),
                       validation_scope='full' if len(selected) == len(dataset) else 'explicit_subset',
                       reference_labels_enter_model=False)
        write_rows(stem.with_suffix('.facts.jsonl'), records)
        stem.with_suffix('.metrics.json').write_text(json.dumps(metrics, indent=2)+'\n')
        model._pending_output_validation = metrics
        return metrics
    finally:
        model.train(was_training)
        restore_rng(state)


def main():
    import argparse
    from .build import read_rows
    cli = argparse.ArgumentParser(description='Score existing predictions; no image or GPU access')
    cli.add_argument('--predictions', type=Path, required=True)
    cli.add_argument('--output', type=Path, required=True)
    args = cli.parse_args()
    predictions = read_rows(args.predictions)
    references = {}
    for prediction in predictions:
        row = {k: prediction[k] for k in ('id', 'patient_id', 'cohort', 'sequence', 'split')}
        row.update(finding=prediction['reference'], text_audit_status=prediction.get('text_audit_status', ''))
        references[row['id']] = make_row(row, extract(row))
    metrics, records = evaluate_predictions(predictions, references)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output/'metrics.json').write_text(json.dumps(metrics, indent=2)+'\n')
    write_rows(args.output/'facts.jsonl', records)
    print(json.dumps(metrics, indent=2))


if __name__ == '__main__':
    main()
