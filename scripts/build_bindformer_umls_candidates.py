"""Build a review-only UMLS candidate table from fixed seeds and TRAIN text.

No CUI is automatically approved. Literal report hits include negation and
uncertainty; they are vocabulary evidence, not patient labels.
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


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def normalize(text):
    text = unicodedata.normalize("NFKC", text).casefold()
    return " ".join(re.sub(r"[-‐‑–—]", " ", text).split())


def dump(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")


def build(args):
    target = args.output / "candidates.jsonl"
    if target.exists() and not args.overwrite:
        raise FileExistsError(f"Refusing to replace an existing review table: {target}")
    if args.manifest.name != "train.jsonl":
        raise ValueError("Candidate vocabulary evidence must come from train.jsonl")
    with args.seeds.open() as stream:
        seeds = list(csv.DictReader(stream, delimiter="\t"))
    keys = [seed["key"] for seed in seeds]
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate seed keys")
    samples = [json.loads(line) for line in args.manifest.read_text().splitlines() if line.strip()]
    if any(sample["split"] != "train" for sample in samples):
        raise ValueError("Non-training row in candidate input")
    if args.cohorts:
        samples = [sample for sample in samples if sample["cohort"] in args.cohorts]
    if not samples:
        raise ValueError("Empty selected training population")
    lookup = defaultdict(set)
    counts = {}
    for seed in seeds:
        for term in seed["terms"].split(";"):
            lookup[normalize(term)].add(seed["key"])
        pattern = re.compile(seed["report_pattern"], re.I)
        hits = [sample for sample in samples if pattern.search(sample["finding"])]
        scope = (["T1W", "T1C", "T2W", "FLAIR"] if seed["sequence_scope"] == "*"
                 else seed["sequence_scope"].split(","))
        by_sequence = Counter(sequence_of(sample["cohort"], sample["id"]) for sample in hits)
        counts[seed["key"]] = {
            "reports": len(hits),
            "by_cohort": dict(Counter(sample["cohort"] for sample in hits)),
            "by_sequence": dict(by_sequence),
            "within_sequence_scope": sum(n for sequence, n in by_sequence.items() if sequence in scope),
            "example_train_ids": [sample["id"] for sample in hits[:3]],
        }
    matches = defaultdict(lambda: defaultdict(list))
    print("Scanning MRCONSO for exact normalized seed matches...", flush=True)
    with (args.meta / "MRCONSO.RRF").open() as stream:
        for line in stream:
            row = line.rstrip("\n").split("|")
            if len(row) < 18 or row[1] != "ENG" or row[16] != "N":
                continue
            for key in lookup.get(normalize(row[14]), ()):
                matches[key][row[0]].append({
                    "term": row[14], "aui": row[7], "source": row[11],
                    "term_type": row[12], "code": row[13],
                })
    cuis = {cui for found in matches.values() for cui in found}
    details = {cui: {"semantic_types": [], "definitions": [], "synonyms": set(),
                     "sources": set(), "english_auis": set()} for cui in cuis}
    print(f"Enriching {len(cuis)} candidate CUIs with English synonyms and types...", flush=True)
    with (args.meta / "MRCONSO.RRF").open() as stream:
        for line in stream:
            cui = line.split("|", 1)[0]
            if cui not in cuis:
                continue
            row = line.rstrip("\n").split("|")
            if row[1] == "ENG" and row[16] == "N":
                details[cui]["synonyms"].add(row[14])
                details[cui]["sources"].add(row[11])
                details[cui]["english_auis"].add(row[7])
    with (args.meta / "MRSTY.RRF").open() as stream:
        for line in stream:
            row = line.rstrip("\n").split("|")
            if row[0] in cuis:
                details[row[0]]["semantic_types"].append({"tui": row[1], "name": row[3]})
    with (args.meta / "MRDEF.RRF").open() as stream:
        for line in stream:
            row = line.rstrip("\n").split("|")
            if (row[0] in cuis and row[6] == "N"
                    and row[1] in details[row[0]]["english_auis"]):
                details[row[0]]["definitions"].append({
                    "aui": row[1], "source": row[4], "text": row[5],
                    "language_filter": "linked_to_active_English_atom",
                })
    source_info = {}
    all_sources = {source for value in details.values() for source in value["sources"]}
    with (args.meta / "MRSAB.RRF").open() as stream:
        for line in stream:
            row = line.rstrip("\n").split("|")
            if row[3] in all_sources:
                source_info.setdefault(row[3], []).append({
                    "versioned_source": row[2], "source_name": row[4], "source_version": row[6]})
    records = []
    for seed in seeds:
        candidates = []
        for cui, atoms in sorted(matches[seed["key"]].items()):
            detail = details[cui]
            candidates.append({
                "cui": cui, "matched_atoms": atoms,
                "semantic_types": detail["semantic_types"],
                "english_synonyms": sorted(detail["synonyms"]),
                "definitions": detail["definitions"],
                "sources": sorted(detail["sources"]),
            })
        flags = []
        if not candidates:
            flags.append("no_exact_match_try_synonyms_or_local_id")
        if len(candidates) > 1:
            flags.append("multiple_cuis_require_disambiguation")
        if candidates and not any(c["definitions"] for c in candidates):
            flags.append("no_definition_linked_to_active_English_atom")
        if counts[seed["key"]]["reports"] == 0:
            flags.append("schema_seed_without_literal_train_hit")
        if seed["key"] in {"enhancement", "mass", "lesion", "midline", "left", "right", "edema"}:
            flags.append("known_context_ambiguity")
        records.append({
            "candidate_id": "brain_mri:" + seed["key"],
            "role": seed["role"], "field": seed["field"],
            "seed_terms": seed["terms"].split(";"),
            "seed_origin": "project_schema_candidate_not_clinically_approved",
            "sequence_scope": (["T1W", "T1C", "T2W", "FLAIR"] if seed["sequence_scope"] == "*"
                               else seed["sequence_scope"].split(",")),
            "report_pattern": seed["report_pattern"],
            "train_literal_hits": counts[seed["key"]],
            "suggested_tier": ("core_review_candidate" if counts[seed["key"]]["within_sequence_scope"]
                               else "coverage_extension_review"),
            "umls_candidates": candidates, "selected_cui": None,
            "review_status": "pending", "flags": flags,
        })
    args.output.mkdir(parents=True, exist_ok=True)
    target.write_text("".join(json.dumps(item, ensure_ascii=False) + "\n" for item in records))
    dump(args.output / "source_versions.json", source_info)
    audit = {
        "status": "review_only_not_training_labels_or_final_concept_bank",
        "umls_version": "2026AA", "scope": "single_sequence_brain_mri_core",
        "seed_count": len(records), "unique_candidate_cuis": len(cuis),
        "seeds_with_exact_match": sum(bool(item["umls_candidates"]) for item in records),
        "seeds_with_multiple_cuis": sum(len(item["umls_candidates"]) > 1 for item in records),
        "roles": dict(Counter(item["role"] for item in records)),
        "train_report_count": len(samples), "cohorts": dict(Counter(s["cohort"] for s in samples)),
        "train_manifest": str(args.manifest), "train_manifest_sha256": digest(args.manifest),
        "seed_file": str(args.seeds), "seed_sha256": digest(args.seeds),
        "builder_sha256": digest(Path(__file__)),
        "candidates_sha256": digest(target),
        "umls_schema_sha256": digest(args.meta / "MRFILES.RRF"),
        "raw_inputs": {name: {"size": (args.meta / name).stat().st_size}
                       for name in ["MRCONSO.RRF", "MRDEF.RRF", "MRSTY.RRF", "MRSAB.RRF"]},
        "warnings": [
            "Counts are lexical report hits and include negation/uncertainty.",
            "No validation/test report was read by this builder.",
            "No CUI is approved; zero exact matches do not imply no UMLS concept exists.",
            "Missing labels, explicit unknown, and other must remain separate.",
            "Templates/definitions and sequence applicability require review before encoding.",
        ],
    }
    dump(args.output / "audit.json", audit)
    lines = ["# Brain MRI UMLS candidate review draft", "",
             "All rows are pending review. Counts include negation and are not positive labels.",
             "Only the selected training reports were used for lexical counts.", "",
             "| Role | Field | Seed | Train reports | Candidate CUIs | Review flags |",
             "| --- | --- | --- | ---: | --- | --- |"]
    for item in records:
        lines.append("| " + " | ".join([
            item["role"], item["field"], item["seed_terms"][0],
            str(item["train_literal_hits"]["reports"]),
            ", ".join(c["cui"] for c in item["umls_candidates"]) or "no exact match",
            "; ".join(item["flags"]) or "pending semantic review",
        ]) + " |")
    (args.output / "REVIEW.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({k: audit[k] for k in
                      ["seed_count", "unique_candidate_cuis", "seeds_with_exact_match",
                       "seeds_with_multiple_cuis", "roles", "train_report_count"]}), flush=True)


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--meta", type=Path, default=ROOT / "assets/knowledge/umls/META")
    cli.add_argument("--seeds", type=Path,
                     default=ROOT / "Pipeline_0929/bindformer_umls_seed_terms_v1.tsv")
    cli.add_argument("--manifest", type=Path, default=ROOT /
                     "assets/data/RadGenome-Brain/training_manifests/single_sequence_findings_coref_v2/train.jsonl")
    cli.add_argument("--output", type=Path, default=ROOT /
                     "assets/knowledge/bindformer_brain_candidates_v1")
    cli.add_argument("--cohorts", nargs="+", default=["BraTS_GLI", "BraTS_MEN", "BraTS_MET"])
    cli.add_argument("--overwrite", action="store_true",
                     help="Explicitly replace a previously generated draft; may erase review edits.")
    build(cli.parse_args())


if __name__ == "__main__":
    main()
