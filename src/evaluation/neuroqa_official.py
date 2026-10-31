#!/usr/bin/env python3
"""
NeuroQA Evaluation Script
Computes all metrics: Accuracy, F1, BERTScore, ROUGE-L, Shortcut Score.
Reports per-dataset, per-category, per-format.

Usage:
    python evaluate.py --predictions predictions.json --ground-truth neurosab_test_public.json

Predictions JSON format:
    [
        {"neuroqa_id": "ABCD-00001", "prediction": "Yes"},
        {"neuroqa_id": "ABCD-00002", "prediction": "Normal hippocampal volume"},
        ...
    ]
"""

import json
import argparse
import re
import numpy as np
from collections import defaultdict, Counter

# ============================================================
# METRICS
# ============================================================

def normalize(text):
    """Normalize text for comparison."""
    if text is None:
        return ""
    text = str(text).lower().strip()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s%.,\-]', '', text)
    return text


def exact_match(pred, gold):
    """Exact match after normalization."""
    return 1.0 if normalize(pred) == normalize(gold) else 0.0


def token_f1(pred, gold):
    """Token-level F1 score."""
    pred_tokens = normalize(pred).split()
    gold_tokens = normalize(gold).split()
    if not pred_tokens or not gold_tokens:
        return 1.0 if pred_tokens == gold_tokens else 0.0
    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_common = sum(common.values())
    if num_common == 0:
        return 0.0
    precision = num_common / len(pred_tokens)
    recall = num_common / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def rouge_l(pred, gold):
    """ROUGE-L (longest common subsequence)."""
    pred_tokens = normalize(pred).split()
    gold_tokens = normalize(gold).split()
    if not pred_tokens or not gold_tokens:
        return 1.0 if pred_tokens == gold_tokens else 0.0
    m, n = len(pred_tokens), len(gold_tokens)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if pred_tokens[i-1] == gold_tokens[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    lcs = dp[m][n]
    if lcs == 0:
        return 0.0
    precision = lcs / m
    recall = lcs / n
    return 2 * precision * recall / (precision + recall)


def shortcut_score(model_acc, textonly_acc, oracle_acc=100.0):
    """Shortcut Score: 0 = perfect image use, 1 = pure text shortcuts."""
    if oracle_acc == textonly_acc:
        return 1.0
    return 1.0 - (model_acc - textonly_acc) / (oracle_acc - textonly_acc)


def compute_bertscore(predictions, references):
    """Compute BERTScore. Returns list of F1 scores."""
    try:
        from bert_score import score as bert_score
        P, R, F1 = bert_score(predictions, references, lang="en", verbose=False)
        return F1.tolist()
    except ImportError:
        print("  WARNING: bert_score not installed. pip install bert-score")
        return [0.0] * len(predictions)


# ============================================================
# TEXT-ONLY BASELINE
# ============================================================

def compute_textonly_baseline(gt_items):
    """Compute text-only accuracy: majority answer per template."""
    def tmpl(q):
        return re.sub(r'\d[\d,.]*', 'N', q)

    template_answers = defaultdict(lambda: Counter())
    for d in gt_items:
        t = tmpl(d['question'])
        template_answers[t][str(d['answer'])] += 1

    majority = {}
    for t, counts in template_answers.items():
        majority[t] = counts.most_common(1)[0][0]

    correct = sum(1 for d in gt_items if majority[tmpl(d['question'])] == str(d['answer']))
    return correct / len(gt_items) * 100 if gt_items else 0


# ============================================================
# MAIN EVALUATION
# ============================================================

def evaluate(gt_items, pred_map, compute_bert=False):
    """Evaluate predictions against ground truth. Returns dict of metrics."""

    results = {
        'overall': {},
        'per_dataset': {},
        'per_category': {},
        'per_format': {},
    }

    # Match predictions to ground truth
    matched = []
    missing = 0
    for d in gt_items:
        nid = d['neuroqa_id']
        if nid in pred_map:
            matched.append({
                'gt': d,
                'pred': str(pred_map[nid]),
                'gold': str(d['answer']),
            })
        else:
            missing += 1

    if missing > 0:
        print(f"  WARNING: {missing} items have no prediction")

    # ── Overall metrics ──
    def compute_metrics(items, label=""):
        yn_items = [m for m in items if m['gt']['question_type'] == 'YN']
        mcq_items = [m for m in items if m['gt']['question_type'] == 'MCQ']
        open_items = [m for m in items if m['gt']['question_type'] == 'Open']

        metrics = {'total': len(items)}

        # YN accuracy
        if yn_items:
            yn_correct = sum(1 for m in yn_items if normalize(m['pred']) == normalize(m['gold']))
            metrics['yn_accuracy'] = yn_correct / len(yn_items) * 100
            metrics['yn_count'] = len(yn_items)

        # MCQ accuracy
        if mcq_items:
            mcq_correct = 0
            for m in mcq_items:
                pred_norm = normalize(m['pred'])
                gold_norm = normalize(m['gold'])
                # Try direct match
                if pred_norm == gold_norm:
                    mcq_correct += 1
                # Try option letter match (A/B/C/D)
                elif len(pred_norm) == 1 and pred_norm in 'abcd':
                    idx = ord(pred_norm) - ord('a')
                    opts = m['gt'].get('options', [])
                    if idx < len(opts) and normalize(opts[idx]) == gold_norm:
                        mcq_correct += 1
            metrics['mcq_accuracy'] = mcq_correct / len(mcq_items) * 100
            metrics['mcq_count'] = len(mcq_items)

        # Open metrics
        if open_items:
            em_scores = [exact_match(m['pred'], m['gold']) for m in open_items]
            f1_scores = [token_f1(m['pred'], m['gold']) for m in open_items]
            rl_scores = [rouge_l(m['pred'], m['gold']) for m in open_items]

            metrics['open_exact_match'] = np.mean(em_scores) * 100
            metrics['open_token_f1'] = np.mean(f1_scores) * 100
            metrics['open_rouge_l'] = np.mean(rl_scores) * 100
            metrics['open_count'] = len(open_items)

        # Overall accuracy (YN + MCQ)
        closed = yn_items + mcq_items
        if closed:
            correct = sum(1 for m in closed if normalize(m['pred']) == normalize(m['gold']))
            # Also try MCQ letter matching
            for m in mcq_items:
                pred_norm = normalize(m['pred'])
                if len(pred_norm) == 1 and pred_norm in 'abcd':
                    idx = ord(pred_norm) - ord('a')
                    opts = m['gt'].get('options', [])
                    if idx < len(opts) and normalize(opts[idx]) == normalize(m['gold']):
                        if normalize(m['pred']) != normalize(m['gold']):
                            correct += 1
            metrics['closed_accuracy'] = correct / len(closed) * 100

        return metrics

    # Overall
    results['overall'] = compute_metrics(matched, "overall")

    # Per dataset
    by_ds = defaultdict(list)
    for m in matched:
        by_ds[m['gt']['dataset']].append(m)
    for ds in sorted(by_ds):
        results['per_dataset'][ds] = compute_metrics(by_ds[ds], ds)

    # Per category
    by_cat = defaultdict(list)
    for m in matched:
        by_cat[m['gt']['category']].append(m)
    for cat in sorted(by_cat):
        results['per_category'][cat] = compute_metrics(by_cat[cat], cat)

    # Per format
    by_fmt = defaultdict(list)
    for m in matched:
        by_fmt[m['gt']['question_type']].append(m)
    for fmt in sorted(by_fmt):
        results['per_format'][fmt] = compute_metrics(by_fmt[fmt], fmt)

    # BERTScore for open-ended (optional)
    if compute_bert:
        open_matched = [m for m in matched if m['gt']['question_type'] == 'Open']
        if open_matched:
            preds = [m['pred'] for m in open_matched]
            golds = [m['gold'] for m in open_matched]
            bert_scores = compute_bertscore(preds, golds)
            results['overall']['open_bertscore'] = np.mean(bert_scores) * 100

    # Text-only baseline (all-formats; Open assumed 0%)
    textonly = compute_textonly_baseline(gt_items)
    results['overall']['textonly_accuracy'] = textonly

    # Text-only baseline (closed-only: YN+MCQ) — apples-to-apples with closed_accuracy
    closed_gt = [d for d in gt_items if d.get('question_type', '').upper() in ('YN', 'MCQ')]
    textonly_closed = compute_textonly_baseline(closed_gt) if closed_gt else 0.0
    results['overall']['textonly_accuracy_closed'] = textonly_closed

    # Shortcut Score: two flavors
    if 'closed_accuracy' in results['overall']:
        # SS (all-formats floor): legacy / overall framework
        results['overall']['shortcut_score'] = shortcut_score(
            results['overall']['closed_accuracy'], textonly)
        # SS (closed-only floor): apples-to-apples, matches paper Table 2
        results['overall']['shortcut_score_closed'] = shortcut_score(
            results['overall']['closed_accuracy'], textonly_closed)

    # Per-category shortcut score
    for cat in results['per_category']:
        cat_items = [d for d in gt_items if d['category'] == cat]
        cat_textonly = compute_textonly_baseline(cat_items)
        cat_closed_items = [d for d in cat_items if d.get('question_type', '').upper() in ('YN', 'MCQ')]
        cat_textonly_closed = compute_textonly_baseline(cat_closed_items) if cat_closed_items else 0.0
        results['per_category'][cat]['textonly'] = cat_textonly
        results['per_category'][cat]['textonly_closed'] = cat_textonly_closed
        if 'closed_accuracy' in results['per_category'][cat]:
            results['per_category'][cat]['shortcut_score'] = shortcut_score(
                results['per_category'][cat]['closed_accuracy'], cat_textonly)
            results['per_category'][cat]['shortcut_score_closed'] = shortcut_score(
                results['per_category'][cat]['closed_accuracy'], cat_textonly_closed)

    return results


def print_results(results):
    """Pretty-print evaluation results."""
    print("\n" + "=" * 70)
    print("  NEUROQA EVALUATION RESULTS")
    print("=" * 70)

    o = results['overall']
    print(f"\n  Overall ({o['total']} items):")
    if 'closed_accuracy' in o:
        print(f"    Closed-ended accuracy: {o['closed_accuracy']:.1f}%")
    if 'yn_accuracy' in o:
        print(f"    YN accuracy: {o['yn_accuracy']:.1f}% ({o['yn_count']} items)")
    if 'mcq_accuracy' in o:
        print(f"    MCQ accuracy: {o['mcq_accuracy']:.1f}% ({o['mcq_count']} items)")
    if 'open_exact_match' in o:
        print(f"    Open exact match: {o['open_exact_match']:.1f}%")
        print(f"    Open token F1: {o['open_token_f1']:.1f}%")
        print(f"    Open ROUGE-L: {o['open_rouge_l']:.1f}%")
    if 'open_bertscore' in o:
        print(f"    Open BERTScore: {o['open_bertscore']:.1f}%")
    print(f"    Text-only baseline (all formats):  {o['textonly_accuracy']:.1f}%")
    if 'textonly_accuracy_closed' in o:
        print(f"    Text-only baseline (closed only): {o['textonly_accuracy_closed']:.1f}%")
    if 'shortcut_score' in o:
        print(f"    Shortcut Score (all-formats floor):  {o['shortcut_score']:.3f}")
    if 'shortcut_score_closed' in o:
        print(f"    Shortcut Score (closed-only floor):  {o['shortcut_score_closed']:.3f}")

    print(f"\n  Per-Dataset:")
    print(f"  {'Dataset':<18} {'Items':>6} {'Closed':>8} {'YN':>8} {'MCQ':>8}")
    print(f"  {'-'*50}")
    for ds, m in sorted(results['per_dataset'].items()):
        closed = f"{m.get('closed_accuracy', 0):.1f}%" if 'closed_accuracy' in m else "--"
        yn = f"{m.get('yn_accuracy', 0):.1f}%" if 'yn_accuracy' in m else "--"
        mcq = f"{m.get('mcq_accuracy', 0):.1f}%" if 'mcq_accuracy' in m else "--"
        print(f"  {ds:<18} {m['total']:>6} {closed:>8} {yn:>8} {mcq:>8}")

    print(f"\n  Per-Category:")
    print(f"  {'Category':<18} {'Items':>6} {'Closed':>8} {'TextOnly':>9} {'SS':>6}")
    print(f"  {'-'*50}")
    for cat, m in sorted(results['per_category'].items()):
        closed = f"{m.get('closed_accuracy', 0):.1f}%" if 'closed_accuracy' in m else "--"
        to = f"{m.get('textonly', 0):.1f}%" if 'textonly' in m else "--"
        ss = f"{m.get('shortcut_score', 0):.2f}" if 'shortcut_score' in m else "--"
        print(f"  {cat:<18} {m['total']:>6} {closed:>8} {to:>9} {ss:>6}")


# ============================================================
# INTER-RATER AGREEMENT
# ============================================================

def compute_agreement(raters):
    """Compute inter-rater agreement metrics.
    raters: dict of {rater_name: {neuroqa_id: answer}}
    """
    rater_names = list(raters.keys())
    if len(rater_names) < 2:
        print("Need at least 2 raters for agreement")
        return {}

    # Get common items
    common_ids = set.intersection(*[set(r.keys()) for r in raters.values()])
    print(f"  Common items: {len(common_ids)}")

    # Pairwise Cohen's kappa
    from itertools import combinations
    results = {}

    for r1, r2 in combinations(rater_names, 2):
        answers1 = [str(raters[r1][nid]) for nid in sorted(common_ids)]
        answers2 = [str(raters[r2][nid]) for nid in sorted(common_ids)]

        agree = sum(1 for a, b in zip(answers1, answers2) if a == b)
        total = len(answers1)

        # Simple agreement
        pct = agree / total * 100
        results[f"{r1}_vs_{r2}"] = {
            'agreement': pct,
            'agree': agree,
            'total': total,
        }
        print(f"  {r1} vs {r2}: {pct:.1f}% ({agree}/{total})")

    # Overall agreement
    all_agree = 0
    for nid in sorted(common_ids):
        answers = [str(raters[r][nid]) for r in rater_names]
        if len(set(answers)) == 1:
            all_agree += 1

    results['all_agree'] = all_agree / len(common_ids) * 100
    print(f"  All agree: {results['all_agree']:.1f}% ({all_agree}/{len(common_ids)})")

    return results


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="NeuroQA Evaluation")
    parser.add_argument("--predictions", required=True, help="Predictions JSON file")
    parser.add_argument("--ground-truth", required=True, help="Ground truth JSON file")
    parser.add_argument("--output", default=None, help="Output JSON file for results")
    parser.add_argument("--bertscore", action="store_true", help="Compute BERTScore (slow)")
    args = parser.parse_args()

    # Load data
    gt = json.load(open(args.ground_truth))
    preds = json.load(open(args.predictions))

    print(f"Ground truth: {len(gt)} items")
    print(f"Predictions: {len(preds)} items")

    # Build prediction map
    pred_map = {p['neuroqa_id']: p['prediction'] for p in preds}

    # Evaluate
    results = evaluate(gt, pred_map, compute_bert=args.bertscore)

    # Print
    print_results(results)

    # Save
    if args.output:
        json.dump(results, open(args.output, 'w'), indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()

