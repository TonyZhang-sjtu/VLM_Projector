"""Mine TRAIN report vocabulary independently of the old BindFormer seeds.

Outputs are review queues, not clinical annotations or an approved concept bank.
The old seed patterns are used only AFTER extraction, for lexical comparison.
Run with the projector Python environment; NER runs locally on CPU by default.
"""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import re
import sys
import unicodedata

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data.sequence_mapping import sequence_of

TOKEN = re.compile(r"[A-Za-z]+[A-Za-z0-9]*(?:['’][A-Za-z]+)?|\d+(?:\.\d+)?")
WORD = re.compile(r"[A-Za-z]+(?:[-'][A-Za-z]+)*")
JOIN = re.compile(r"[\s\-‐‑–—]*")


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def norm(text):
    return ' '.join(re.sub(r'[-‐‑–—]', ' ', unicodedata.normalize('NFKC', text).casefold()).split())


def write_json(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n')


def write_jsonl(path, rows):
    with path.open('w') as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + '\n')


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_tsv(path, rows, columns):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, delimiter='\t', extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


def corpus(args):
    if args.manifest.name != 'train.jsonl':
        raise ValueError('Only train.jsonl is allowed')
    rows = read_jsonl(args.manifest)
    if not rows or any(row['split'] != 'train' for row in rows):
        raise ValueError('Empty or non-training input')
    if len({row['id'] for row in rows}) != len(rows):
        raise ValueError('Duplicate report IDs')
    for row in rows:
        row['sequence'] = sequence_of(row['cohort'], row['id'])
    args.output.mkdir(parents=True, exist_ok=True)
    return rows


def span_record(row, start, end):
    return {'report_id': row['id'], 'cohort': row['cohort'], 'sequence': row['sequence'],
            'start': start, 'end': end, 'text': row['finding'][start:end],
            'context': row['finding'][max(0, start - 90):end + 90]}


def counts(indices, rows):
    selected = [rows[i] for i in sorted(indices)]
    return {'reports': len(selected),
            'brats_reports': sum(row['cohort'].startswith('BraTS_') for row in selected),
            'by_cohort': dict(sorted(Counter(row['cohort'] for row in selected).items())),
            'by_sequence': dict(sorted(Counter(row['sequence'] for row in selected).items()))}


def phrases(text, maximum):
    """Contiguous words only: do not create phrases across punctuation/sentences."""
    tokens = list(TOKEN.finditer(text))
    for i, token in enumerate(tokens):
        for j in range(i, min(i + maximum, len(tokens))):
            if j > i and not JOIN.fullmatch(text[tokens[j - 1].end():tokens[j].start()]):
                break
            start, end = token.start(), tokens[j].end()
            yield norm(text[start:end]), start, end


def extract_ner(args, rows):
    from src.evaluation.clinical_runtime import activate
    activate()
    import torch
    from transformers import AutoModelForTokenClassification, AutoTokenizer
    torch.set_num_threads(args.threads)
    tokenizer = AutoTokenizer.from_pretrained(args.ner_model, local_files_only=True, use_fast=True)
    if not tokenizer.is_fast:
        raise ValueError('Fast tokenizer offsets required')
    model = AutoModelForTokenClassification.from_pretrained(args.ner_model, local_files_only=True)
    model.eval()
    lengths = [len(tokenizer(row['finding'])['input_ids']) for row in rows]
    if max(lengths) > 512:
        raise ValueError('Report longer than 512 tokens; implement chunking, never silently truncate')
    spans = []
    # Sorting reduces padding, while keeping the original report indices/offsets.
    order = sorted(range(len(rows)), key=lambda i: lengths[i])
    for batch_start in range(0, len(rows), args.batch_size):
        indices = order[batch_start:batch_start + args.batch_size]
        batch = tokenizer([rows[i]['finding'] for i in indices], padding=True,
                          return_offsets_mapping=True, return_tensors='pt', truncation=False)
        offsets = batch.pop('offset_mapping').tolist()
        with torch.inference_mode():
            probs = model(**batch).logits.softmax(-1)
            confidence, predictions = probs.max(-1)
        for index, positions, labels, scores in zip(indices, offsets, predictions.tolist(), confidence.tolist()):
            current = None
            def finish():
                if current is None:
                    return
                start, end = current['start'], current['end']
                text = rows[index]['finding']
                while start < end and text[start].isspace():
                    start += 1
                while end > start and text[end - 1].isspace():
                    end -= 1
                if end > start:
                    spans.append({**span_record(rows[index], start, end),
                                  'normalized': norm(text[start:end]), 'ner_label': current['label'],
                                  'mean_token_confidence': sum(current['scores']) / len(current['scores'])})
            for (start, end), label_id, score in zip(positions, labels, scores):
                label = model.config.id2label[label_id]
                if end <= start or label == 'O':
                    finish()
                    current = None
                    continue
                prefix, kind = label.split('-', 1)
                if current is None or prefix == 'B' or current['label'] != kind:
                    finish()
                    current = {'start': start, 'end': end, 'label': kind, 'scores': [score]}
                else:
                    current['end'] = end
                    current['scores'].append(score)
            finish()
        if batch_start % (args.batch_size * 10) == 0:
            print(f'NER {min(batch_start + args.batch_size, len(rows))}/{len(rows)}', flush=True)
    spans.sort(key=lambda item: (item['report_id'], item['start'], item['end']))
    write_jsonl(args.output / 'ner_spans.jsonl', spans)
    write_json(args.output / 'ner_audit.json', {
        'train_sha256': sha(args.manifest), 'script_sha256': sha(__file__),
        'model': str(args.ner_model), 'model_weights_sha256': sha(args.ner_model / 'model.safetensors'),
        'tokenizer_sha256': sha(args.ner_model / 'tokenizer.json'),
        'reports': len(rows), 'max_report_tokens': max(lengths), 'truncated_reports': 0,
        'mentions': len(spans), 'labels': dict(Counter(item['ner_label'] for item in spans)),
        'warning': 'Predicted spans/types are noisy suggestions, including assertion errors; not labels.'})


def extract_lexical(args, rows):
    occurrences = defaultdict(list)
    words = defaultdict(list)
    for index, row in enumerate(rows):
        for term, start, end in phrases(row['finding'], args.max_words):
            occurrences[term].append((index, start, end))
        for match in WORD.finditer(row['finding']):
            words[match[0].casefold()].append((index, match.start(), match.end()))
    print(f'Observed {len(occurrences)} contiguous 1..{args.max_words}-word strings; scanning UMLS...', flush=True)
    matches = defaultdict(lambda: defaultdict(list))
    with (args.meta / 'MRCONSO.RRF').open() as stream:
        for line in stream:
            row = line.rstrip('\n').split('|')
            if row[1] != 'ENG' or row[16] != 'N':
                continue
            term = norm(row[14])
            if term in occurrences:
                matches[term][row[0]].append({'term': row[14], 'aui': row[7],
                                             'source': row[11], 'term_type': row[12]})
    cuis = {cui for match in matches.values() for cui in match}
    types = defaultdict(list)
    with (args.meta / 'MRSTY.RRF').open() as stream:
        for line in stream:
            row = line.rstrip('\n').split('|')
            if row[0] in cuis:
                types[row[0]].append({'tui': row[1], 'name': row[3]})
    with args.seeds.open() as stream:
        seeds = list(csv.DictReader(stream, delimiter='\t'))
    seed_patterns = [(s['key'], re.compile(s['report_pattern'], re.I)) for s in seeds]
    # This is a loose overlap diagnostic, NOT evidence of full concept equivalence.
    seed_spans = {i: [(m.start(), m.end(), key) for key, pattern in seed_patterns
                      for m in pattern.finditer(row['finding'])] for i, row in enumerate(rows)}
    exact_seeds = defaultdict(list)
    for seed in seeds:
        for term in seed['terms'].split(';'):
            exact_seeds[norm(term)].append(seed['key'])
    def record(term, occ):
        indices = {i for i, _, _ in occ}
        overlap = Counter()
        full = Counter()
        examples = []
        example_ids = set()
        for i, start, end in occ:
            for a, b, key in seed_spans[i]:
                if a < end and start < b:
                    overlap[key] += 1
                if a <= start and end <= b:
                    full[key] += 1
            if i not in example_ids and len(examples) < 3:
                examples.append(span_record(rows[i], start, end))
                example_ids.add(i)
        return {'normalized': term, **counts(indices, rows), 'mentions': len(occ),
                'exact_seed_keys': exact_seeds.get(term, []),
                'seed_pattern_overlap_keys': sorted(overlap),
                'seed_pattern_enclosing_keys': sorted(full), 'examples': examples}
    lexical = []
    matched_spans = []
    for term, found in matches.items():
        item = record(term, occurrences[term])
        item['umls_candidates'] = [{'cui': cui, 'semantic_types': types[cui], 'matched_atoms': atoms}
                                   for cui, atoms in sorted(found.items())]
        item.update(selected_cui=None, review_status='pending_context_and_role_review')
        lexical.append(item)
        for i, start, end in occurrences[term]:
            matched_spans.append({**span_record(rows[i], start, end), 'normalized': term})
    lexical.sort(key=lambda item: (-item['reports'], item['normalized']))
    write_jsonl(args.output / 'umls_phrase_candidates.jsonl', lexical)
    write_jsonl(args.output / 'umls_phrase_mentions.jsonl', matched_spans)
    word_rows = [record(term, occ) for term, occ in words.items()]
    word_rows.sort(key=lambda item: (-item['reports'], item['normalized']))
    write_jsonl(args.output / 'word_inventory.jsonl', word_rows)
    write_tsv(args.output / 'word_inventory.tsv', [dict(item,
        example=item['examples'][0]['context'], by_cohort=json.dumps(item['by_cohort']),
        by_sequence=json.dumps(item['by_sequence']),
        seed_pattern_overlap_keys=','.join(item['seed_pattern_overlap_keys'])) for item in word_rows],
        ['normalized', 'reports', 'brats_reports', 'mentions', 'by_cohort', 'by_sequence',
         'seed_pattern_overlap_keys', 'example'])
    # Keep every observed ngram, including singletons and unmatched phrases.
    with (args.output / 'all_observed_ngrams.tsv').open('w', newline='') as stream:
        writer = csv.writer(stream, delimiter='\t')
        writer.writerow(['normalized', 'reports', 'mentions', 'umls_exact_match', 'example_report_id', 'start', 'end'])
        for term, occ in sorted(occurrences.items()):
            i, start, end = occ[0]
            writer.writerow([term, len({i for i, _, _ in occ}), len(occ), bool(matches.get(term)),
                             rows[i]['id'], start, end])
    # Compare v1 with the SAME train cohort denominator as its original run.
    seed_counts = []
    for key, pattern in seed_patterns:
        indices = {i for i, row in enumerate(rows) if pattern.search(row['finding'])}
        seed_counts.append({'key': key, **counts(indices, rows)})
    write_jsonl(args.output / 'v1_seed_counts.jsonl', seed_counts)
    audit = {'train_manifest': str(args.manifest), 'train_sha256': sha(args.manifest),
             'script_sha256': sha(__file__), 'seed_sha256': sha(args.seeds),
             'reports': len(rows), 'cohorts': dict(Counter(r['cohort'] for r in rows)),
             'sequences': dict(Counter(r['sequence'] for r in rows)),
             'unique_report_texts': len({r['finding'] for r in rows}),
             'word_forms': len(words), 'max_ngram_words': args.max_words,
             'observed_ngrams': len(occurrences), 'umls_matched_strings': len(matches),
             'umls_candidate_cuis': len(cuis),
             'umls_version': '2026AA', 'umls_schema_sha256': sha(args.meta / 'MRFILES.RRF'),
             'umls_raw_files': {name: {'size': (args.meta / name).stat().st_size,
                                      'mtime_ns': (args.meta / name).stat().st_mtime_ns}
                                for name in ['MRCONSO.RRF', 'MRSTY.RRF']},
             'limits': ['Only training reports read; corpus includes negative, uncertain and editorial text.',
                        'No stemming, synonym expansion, or semantic disambiguation in exact UMLS pass.',
                        'No minimum frequency filter; all words/ngrams retained for residual review.',
                        'Seed pattern overlap is NOT semantic coverage and NOT an approved concept.',
                        'Candidate counts are surface strings, not unique clinical concepts.']}
    write_json(args.output / 'lexical_audit.json', audit)
    print(json.dumps(audit, ensure_ascii=False), flush=True)


def summarize(args, rows):
    ner_audit = json.loads((args.output / 'ner_audit.json').read_text())
    lexical_audit = json.loads((args.output / 'lexical_audit.json').read_text())
    if any(audit['train_sha256'] != sha(args.manifest) for audit in [ner_audit, lexical_audit]):
        raise ValueError('Stale corpus extraction; input fingerprints differ')
    lexical = {r['normalized']: r for r in read_jsonl(args.output / 'umls_phrase_candidates.jsonl')}
    groups = defaultdict(list)
    for span in read_jsonl(args.output / 'ner_spans.jsonl'):
        groups[span['normalized']].append(span)
    row_index = {r['id']: i for i, r in enumerate(rows)}
    ner = []
    for term, spans in groups.items():
        item = {'normalized': term, **counts({row_index[s['report_id']] for s in spans}, rows),
                'mentions': len(spans), 'ner_labels': dict(Counter(s['ner_label'] for s in spans)),
                'exact_umls_string': term in lexical, 'examples': spans[:3],
                'review_status': 'pending_ner_span_and_context_review'}
        ner.append(item)
    ner.sort(key=lambda r: (-r['reports'], r['normalized']))
    write_jsonl(args.output / 'ner_phrase_candidates.jsonl', ner)
    write_tsv(args.output / 'ner_phrase_candidates.tsv', [dict(r,
        ner_labels=json.dumps(r['ner_labels']), example=r['examples'][0]['context']) for r in ner],
        ['normalized', 'reports', 'brats_reports', 'mentions', 'ner_labels', 'exact_umls_string', 'example'])
    write_tsv(args.output / 'umls_phrase_candidates.tsv', [dict(r,
        cuis=','.join(c['cui'] for c in r['umls_candidates']),
        semantic_types=';'.join(sorted({t['name'] for c in r['umls_candidates'] for t in c['semantic_types']})),
        exact_seed_keys=','.join(r['exact_seed_keys']),
        seed_pattern_overlap_keys=','.join(r['seed_pattern_overlap_keys']), example=r['examples'][0]['context'])
        for r in lexical.values()], ['normalized', 'reports', 'brats_reports', 'mentions', 'cuis',
                                    'semantic_types', 'exact_seed_keys', 'seed_pattern_overlap_keys', 'example'])
    combined = dict(lexical_audit, ner=ner_audit, unique_ner_phrases=len(ner),
                    ner_phrases_without_exact_umls=sum(not r['exact_umls_string'] for r in ner),
                    summary_script_sha256=sha(__file__),
                    status='corpus_derived_review_queue_not_final_bank')
    write_json(args.output / 'audit.json', combined)
    review(args, rows, lexical)
    print(json.dumps({key: combined[key] for key in ['reports', 'word_forms', 'observed_ngrams',
          'umls_matched_strings', 'umls_candidate_cuis', 'unique_ner_phrases', 'ner_phrases_without_exact_umls']}))


def review(args, rows, lexical):
    """Post-extraction triage, explicitly separate from independent mining."""
    with args.review_rules.open() as stream:
        rules = list(csv.DictReader(stream, delimiter='\t'))
    with args.seeds.open() as stream:
        seed_patterns = {s['key']: re.compile(s['report_pattern'], re.I)
                         for s in csv.DictReader(stream, delimiter='\t')}
    results = []
    all_mentions = []
    for rule in rules:
        pattern = re.compile(rule['pattern'], re.I)
        mentions = [(i, m) for i, row in enumerate(rows) for m in pattern.finditer(row['finding'])]
        indices = {i for i, _ in mentions}
        old_keys = [key for key in rule['old_seed_keys'].split(',') if key]
        # Old-key hits anywhere in a report are just a loose comparison, never assertion/semantic coverage.
        old_hits = {i for i in indices if any(seed_patterns[key].search(rows[i]['finding']) for key in old_keys)}
        examples = []
        example_ids = set()
        for i, match in mentions:
            mention = dict(span_record(rows[i], match.start(), match.end()), review_key=rule['key'])
            all_mentions.append(mention)
            if i not in example_ids and len(examples) < 3:
                examples.append(mention)
                example_ids.add(i)
        strings = sorted({norm(m[0]) for _, m in mentions})
        candidate_cuis = sorted({c['cui'] for s in strings for c in lexical.get(s, {}).get('umls_candidates', [])})
        results.append(dict(rule, **counts(indices, rows), surface_forms=strings,
            old_key_hits_anywhere_in_same_reports=len(old_hits),
            reports_without_old_key_hit=len(indices - old_hits) if old_keys else None,
            exact_string_candidate_cuis=candidate_cuis, examples=examples,
            review_status='suggested_by_corpus_audit_not_clinically_approved', selected_cui=None))
    write_jsonl(args.output / 'priority_review.jsonl', results)
    write_jsonl(args.output / 'priority_mentions.jsonl', all_mentions)
    write_tsv(args.output / 'priority_review.tsv', [dict(r,
        surface_forms=';'.join(r['surface_forms']),
        exact_string_candidate_cuis=','.join(r['exact_string_candidate_cuis']),
        by_cohort=json.dumps(r['by_cohort']), by_sequence=json.dumps(r['by_sequence']),
        example=r['examples'][0]['context'] if r['examples'] else '') for r in results],
        ['key', 'role', 'field', 'kind', 'reports', 'brats_reports', 'by_cohort', 'by_sequence',
         'old_seed_keys', 'old_key_hits_anywhere_in_same_reports', 'reports_without_old_key_hit',
         'surface_forms', 'exact_string_candidate_cuis', 'recommendation', 'example'])
    write_json(args.output / 'priority_audit.json', {
        'rules_sha256': sha(args.review_rules), 'script_sha256': sha(__file__),
        'train_sha256': sha(args.manifest), 'rules': len(results),
        'rules_with_report_evidence': sum(bool(r['reports']) for r in results),
        'warning': 'Post-mining human-readable triage families; not final concepts or completeness proof.'})
    lines = ['# 从训练原文整理的优先审核项', '',
             '这 60 个词族/字段是全量抽取后的重点审查项，不是 60 个已经确认的新 CUI，也不等于完整词表。',
             '报告数按本行正则去重计算，包含肯定、否定、不确定和解释性描述；多行可命中同一报告。',
             '全部原文位置和上下文保存在 priority_mentions.jsonl；按队列、序列细分见 priority_review.tsv。', '',
             '| 原文词族/问题 | 建议角色或上下文 | 类型 | 全训练集 | BraTS | 审核建议 |',
             '| --- | --- | --- | ---: | ---: | --- |']
    for item in results:
        lines.append('| ' + ' | '.join([item['key'], item['role'] + '/' + item['field'],
            item['kind'], str(item['reports']), str(item['brats_reports']), item['recommendation']]) + ' |')
    (args.output / 'PRIORITY_REVIEW.md').write_text('\n'.join(lines) + '\n')


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--manifest', type=Path, default=ROOT /
                     'assets/data/RadGenome-Brain/training_manifests/single_sequence_findings_coref_v2/train.jsonl')
    cli.add_argument('--meta', type=Path, default=ROOT / 'assets/knowledge/umls/META')
    cli.add_argument('--seeds', type=Path, default=ROOT / 'Pipeline_0929/bindformer_umls_seed_terms_v1.tsv')
    cli.add_argument('--ner-model', type=Path, default=ROOT / 'assets/models/evaluation/rate-ner')
    cli.add_argument('--review-rules', type=Path, default=ROOT / 'Pipeline_0929/bindformer_corpus_review_rules_v1.tsv')
    cli.add_argument('--output', type=Path, default=ROOT / 'assets/knowledge/bindformer_brain_corpus_audit_v1')
    cli.add_argument('--stage', choices=['all', 'ner', 'lexical', 'summary'], default='all')
    cli.add_argument('--max-words', type=int, default=8)
    cli.add_argument('--batch-size', type=int, default=16)
    cli.add_argument('--threads', type=int, default=4)
    args = cli.parse_args()
    if min(args.max_words, args.batch_size, args.threads) <= 0:
        raise ValueError('Lengths, batch size and CPU threads must be positive')
    rows = corpus(args)
    if args.stage in ['all', 'lexical']:
        extract_lexical(args, rows)
    if args.stage in ['all', 'ner']:
        extract_ner(args, rows)
    if args.stage in ['all', 'summary']:
        summarize(args, rows)


if __name__ == '__main__':
    main()
