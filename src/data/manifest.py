from dataclasses import dataclass, asdict
import json
from pathlib import Path
from collections import Counter

from src.config import path_of, file_hash
from src.data.datasets import ALL_DATASETS, dataset_for_cohort, normalize_datasets
from src.data.sequence_mapping import sequence_of, patient_and_case


@dataclass(frozen=True)
class Sample:
    id: str
    split: str
    cohort: str
    patient_id: str
    case_id: str
    image: str
    mask: str
    finding: str
    sequence: str
    target_version: str
    text_audit_status: str
    coreference_revised: bool

    def metadata(self):
        return asdict(self)


def contained_path(root, relative):
    path = (Path(root) / relative).resolve()
    if not path.is_relative_to(Path(root).resolve()):
        raise ValueError(f"Path escapes dataset root: {relative}")
    return path


def load_manifests(config, check_paths=True, datasets=None):
    if config.task.name == 'neuroqa':
        from src.data.neuroqa_manifest import load_neuroqa
        return load_neuroqa(config, check_paths)
    # Older saved configs had no selector and therefore loaded every source.
    configured = config.data.get('datasets', ALL_DATASETS)
    selected_datasets = normalize_datasets(
        datasets if datasets is not None else configured, default=ALL_DATASETS)
    selected_set = set(selected_datasets)
    data_root = path_of(config, config.data.root)
    annotation_root = path_of(config, config.data.annotation_root)
    manifest_root = path_of(config, config.data.manifest_dir)
    seen = set()
    annotations = {}
    samples, summary, patients = {}, {}, {}
    for split in ("train", "val", "test"):
        path = manifest_root / f"{split}.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        result = []
        for row in rows:
            sid = row["id"]
            if sid in seen or row["split"] != split:
                raise ValueError(f"Duplicate ID or incorrect split: {sid}")
            seen.add(sid)
            # Keep global ID/split integrity checks, but do not extract or
            # validate targets/paths from a source outside the requested scope.
            if dataset_for_cohort(row["cohort"]) not in selected_set:
                continue
            if row["target_version"] != config.data.target_version or not row["finding"].strip():
                raise ValueError(f"Incorrect target version or empty target: {sid}")
            if row["text_audit_status"] not in ("requires_review", "coreference_pass_completed_not_clinical_certification"):
                raise ValueError(f"Unknown review state: {sid}")
            patient, case = patient_and_case(row["cohort"], sid)
            if patient != row["patient_id"] or case != row["case_id"]:
                raise ValueError(f"Patient/case identity mismatch: {sid}: {(patient, case)}")
            ap = contained_path(annotation_root, row["annotation_source"])
            if ap not in annotations:
                annotations[ap] = json.loads(ap.read_text())
            if annotations[ap].get(sid) != row["finding"]:
                raise ValueError(f"Active target differs from manifest: {sid}")
            image, mask = [contained_path(data_root, row[k]) for k in ("image", "mask")]
            if check_paths:
                for file in (image, mask):
                    if not file.is_file() or not file.stat().st_size:
                        raise FileNotFoundError(file)
            result.append(Sample(**{k: row[k] for k in Sample.__dataclass_fields__
                                    if k not in ("sequence", "image", "mask")},
                                 image=str(image), mask=str(mask),
                                 sequence=sequence_of(row["cohort"], sid)))
        samples[split] = result
        if not result:
            raise ValueError(f"No samples remain in {split} for datasets {selected_datasets}")
        patients[split] = {s.patient_id for s in result}
        dataset_counts = {name: sum(dataset_for_cohort(s.cohort) == name for s in result)
                          for name in selected_datasets}
        summary[split] = {"samples": len(result), "patients": len(patients[split]),
                          "cases": len({s.case_id for s in result}),
                          "datasets": selected_datasets,
                          "dataset_counts": dataset_counts,
                          "cohorts": dict(Counter(s.cohort for s in result)),
                          "review_status": dict(Counter(s.text_audit_status for s in result)),
                          "sha256": file_hash(path)}
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = patients[a] & patients[b]
        if overlap:
            raise ValueError(f"Patient leakage {a}/{b}: {sorted(overlap)}")
    return samples, summary
