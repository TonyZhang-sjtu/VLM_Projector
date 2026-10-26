"""Build a separate immutable train-only facts/pairs sidecar. Never opens images/masks."""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import random
from src.config import ROOT, file_hash
from src.data.sequence_mapping import sequence_of
from .extract import extract_report
from .schema import FIELDS, VERSION


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def write_jsonl(path, rows):
    Path(path).write_text(''.join(json.dumps(row, ensure_ascii=False)+'\n' for row in rows))


def text_hash(text):
    return hashlib.sha256(text.encode()).hexdigest()


def build_pairs(rows, seed=42):
    """Disjoint reciprocal pairs per field/sequence/cohort; no class-majority bias.

    Each paired patient supplies one positive and one negative for exactly the
    same question/candidate set. Pairs are evaluated in separate microbatches,
    not treated as if gradient accumulation were an in-batch negative pool.
    """
    groups = defaultdict(list)
    for row in rows:
        for field in row['facts']:
            # "Mixed" can contain high/low components; it is not a guaranteed
            # contradictory answer to either. Retain it for E1, not E2 negatives.
            if field=='signal' and row['facts'][field]['value']=='mixed_signal':continue
            groups[(row['cohort'], row['sequence'], field)].append(row)
    pairs = defaultdict(list)
    rng = random.Random(seed)
    for (_, _, field), members in sorted(groups.items()):
        members = sorted(members, key=lambda r:r['id'])
        rng.shuffle(members)
        while members:
            first = members.pop()
            other = next((i for i,r in enumerate(members) if r['patient_id'] != first['patient_id']
                          and r['facts'][field]['value'] != first['facts'][field]['value']), None)
            if other is None:
                continue
            second = members.pop(other)
            pid = text_hash('|'.join([field,*sorted([first['id'],second['id']])]))[:20]
            for source, partner in ((first,second),(second,first)):
                pairs[source['id']].append(dict(pair_id=pid, field=field, partner_id=partner['id'],
                    positive=source['facts'][field]['value'], negative=partner['facts'][field]['value']))
    # Select at most one reciprocal pair per report. Round-robin fields avoid
    # spending all decoder forwards on the most frequent attribute. Both ends
    # are retained or dropped together; there are only two extra short forwards.
    queues=defaultdict(list)
    seen=set()
    for rid,values in sorted(pairs.items()):
        for pair in values:
            if pair['pair_id'] not in seen:
                queues[pair['field']].append((rid,pair))
                seen.add(pair['pair_id'])
    for queue in queues.values():rng.shuffle(queue)
    selected={}
    while any(queues.values()):
        for field in sorted(queues):
            queue=queues[field]
            while queue:
                rid,pair=queue.pop();other=pair['partner_id']
                if rid in selected or other in selected:continue
                selected[rid]=[pair]
                selected[other]=[p for p in pairs[other] if p['pair_id']==pair['pair_id']]
                break
    return [dict(id=rid,pairs=values) for rid,values in sorted(selected.items())]


def build(manifest, candidates, output, seed=42):
    manifest, candidates, output = map(Path, (manifest,candidates,output))
    if output.exists():
        raise FileExistsError(f'Refusing to replace derived labels: {output}; use a new version directory')
    source = read_jsonl(manifest)
    if not source or manifest.name!='train.jsonl' or any(r['split']!='train' for r in source):
        raise ValueError('Only a nonempty training manifest is accepted')
    if len({r['id'] for r in source}) != len(source):
        raise ValueError('Duplicate source IDs')
    reports = {r['report_id']:r for r in read_jsonl(candidates/'reports_context.jsonl')}
    mentions = defaultdict(list)
    for m in read_jsonl(candidates/'mentions.jsonl'):
        mentions[m['report_id']].append(m)
    rows=[]
    for original in source:
        row = dict(original, sequence=sequence_of(original['cohort'],original['id']))
        context = reports.get(row['id'])
        if context is None or context['finding']!=row['finding'] or context['sequence']!=row['sequence']:
            raise ValueError(f'Stale or absent candidate context: {row["id"]}')
        result = extract_report(row, mentions[row['id']])
        result['finding_sha256']=text_hash(row['finding'])
        rows.append(result)
    pairs = build_pairs(rows, seed)
    output.mkdir(parents=True)
    write_jsonl(output/'train.jsonl', rows)
    write_jsonl(output/'pairs.jsonl', pairs)
    schema = dict(version=VERSION, fields=FIELDS, missing_label_policy='ignore',
                  attachment='single_lesion_text_heuristic', region_ground_truth=False,
                  mask_used=False, train_only=True, pair_policy='reciprocal_disjoint_per_field_same_cohort_sequence')
    (output/'schema.json').write_text(json.dumps(schema,indent=2)+'\n')
    counts = {field:dict(Counter(r['facts'][field]['value'] for r in rows if field in r['facts'])) for field in FIELDS}
    pair_counts = Counter(p['field'] for row in pairs for p in row['pairs'])
    metadata = dict(version=VERSION, seed=seed, manifest=str(manifest.resolve()), manifest_sha256=file_hash(manifest),
        candidate_hashes={name:file_hash(candidates/name) for name in ('mentions.jsonl','reports_context.jsonl')},
        source_hashes={name:file_hash(Path(__file__).parent/name) for name in ('build.py','extract.py','schema.py')},
        files={name:file_hash(output/name) for name in ('train.jsonl','pairs.jsonl','schema.json')},
        reports=len(rows), reports_with_facts=sum(bool(r['facts']) for r in rows), fields=counts,
        paired_reports=len(pairs), directed_pairs_by_field=dict(pair_counts),
        rejected_reasons=dict(Counter(r.get('skip_reason','accepted_discourse') for r in rows)),
        review_status='rule_derived_not_clinically_certified')
    (output/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    # Stratified review queue, with blank human decisions (never claim it was reviewed).
    source_by_id={r['id']:r for r in source}
    with (output/'review.tsv').open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=['id','sequence','field','value','evidence','reference','reviewer','decision','notes'],delimiter='\t')
        writer.writeheader()
        rng=random.Random(seed)
        for field in FIELDS:
            for value in FIELDS[field]['values']:
                candidates_rows=[r for r in rows if r['facts'].get(field,{}).get('value')==value]
                rng.shuffle(candidates_rows)
                for r in candidates_rows[:5]:
                    writer.writerow(dict(id=r['id'],sequence=r['sequence'],field=field,value=value,
                        evidence=' | '.join(e['evidence'] for e in r['facts'][field]['evidence']),
                        reference=source_by_id[r['id']]['finding']))
    return metadata


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',type=Path,default=ROOT/'assets/data/RadGenome-Brain/training_manifests/single_sequence_findings_coref_v2/train.jsonl')
    parser.add_argument('--candidates',type=Path,default=ROOT/'assets/knowledge/bindformer_brain_candidates_v2')
    parser.add_argument('--output',type=Path,default=ROOT/'assets/knowledge/report_facts_v1')
    parser.add_argument('--seed',type=int,default=42)
    args=parser.parse_args()
    print(json.dumps(build(args.manifest,args.candidates,args.output,args.seed),indent=2))


if __name__=='__main__':main()
