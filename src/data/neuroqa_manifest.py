"""Versioned NeuroQA single-volume manifests; no report targets are reused."""
from collections import Counter
from dataclasses import asdict, dataclass, replace
import json
from pathlib import Path
import random

from src.config import digest, file_hash, path_of

PROTOCOL = 'official_holdout_patient_guard_v1'
MODALITIES = {'T1w': 't1n', 'T2w': 't2w'}
SEQUENCE_IDS = {'T1w': 'T1W', 'T2w': 'T2W'}


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.partial')
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def read_rows(path):
    return [json.loads(s) for s in Path(path).read_text(encoding='utf-8').splitlines() if s.strip()]


def patient_id(case):
    return '-'.join(case.split('-')[:3])


def build_manifests(root, output, preprocessing_root):
    root, output, preprocessing_root = map(Path, (root, output, preprocessing_root))
    raw = {s: json.loads((root / 'NeuroQA/all_public' / f'{s}.json').read_text())
           for s in ('train', 'val', 'test')}
    selected = {s: [r for r in rows if r['dataset'] in ('BraTS-GLI', 'BraTS-MEN')]
                for s, rows in raw.items()}
    seen = set()
    cases = {}
    for split, rows in selected.items():
        for r in rows:
            qid, case = r['neuroqa_id'], r['subject_id']
            if qid in seen or r['split'] != ('test_public' if split == 'test' else split):
                raise ValueError(f'Duplicate QA ID or wrong split: {qid}')
            seen.add(qid)
            folder = root / 'BraTS' / case.split('-')[1] / case
            images = {m: str((folder / f'{case}-{suffix}.nii.gz').resolve())
                      for m, suffix in MODALITIES.items()}
            for image in images.values():
                if not Path(image).is_file():
                    raise FileNotFoundError(image)
            cases[case] = {'case_id': case, 'patient_id': patient_id(case),
                           'cohort': r['dataset'].replace('-', '_'), 'images': images}
    heldout = {patient_id(r['subject_id']) for s in ('val', 'test') for r in selected[s]}
    excluded, counts = [], {}
    output.mkdir(parents=True, exist_ok=True)
    for split, rows in selected.items():
        manifest = []
        for r in rows:
            if r['category'] == 'Modality':
                continue
            allowed = [m for m in MODALITIES if m in r['answerable_modalities']]
            if not allowed:
                continue
            if split == 'train' and patient_id(r['subject_id']) in heldout:
                excluded.append(r['neuroqa_id'])
                continue
            target(r)  # Reject malformed choices/targets before image processing.
            manifest.append({**r, 'official_split': r['split'], 'split': split,
                             'patient_id': patient_id(r['subject_id']),
                             'allowed_modalities': allowed,
                             'images': {m: str((preprocessing_root / 'cases' / r['subject_id'] /
                                               f'{m}_mni_n4.nii.gz').resolve()) for m in allowed}})
        (output / f'{split}.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in manifest))
        counts[split] = {'unique_qa': len(manifest),
                         **{m: sum(m in r['allowed_modalities'] for r in manifest) for m in MODALITIES},
                         'patients': len({r['patient_id'] for r in manifest})}
    (output / 'source_index.jsonl').write_text(''.join(json.dumps(cases[k]) + '\n' for k in sorted(cases)))
    lineage = {'protocol': PROTOCOL, 'counts': counts, 'source_cases': len(cases),
               'excluded_train_qa_ids': excluded, 'excluded_category': 'Modality',
               'raw_sha256': {s: file_hash(root / 'NeuroQA/all_public' / f'{s}.json') for s in raw},
               'preprocessing_root': str(preprocessing_root.resolve())}
    write_json(output / 'lineage.json', lineage)
    return lineage


def target(row):
    kind, answer = row['question_type'], row['answer'].strip()
    if not answer or any(t in answer + row['question'] for t in ('<|', '<image>')):
        raise ValueError('Empty answer or reserved prompt marker')
    if kind == 'MCQ':
        options = row['options']
        if not isinstance(options, list) or not 2 <= len(options) <= 26 or len(set(options)) != len(options):
            raise ValueError(f'Invalid MCQ options: {row["neuroqa_id"]}')
        matches = [i for i, option in enumerate(options) if option.strip().casefold() == answer.casefold()]
        if len(matches) != 1:
            raise ValueError(f'MCQ answer does not uniquely match an option: {row["neuroqa_id"]}')
        return chr(65 + matches[0])
    if kind == 'YN':
        if answer.casefold() not in ('yes', 'no'):
            raise ValueError('Invalid YN answer')
        return answer.capitalize()
    if kind != 'Open':
        raise ValueError(f'Unknown question_type: {kind}')
    return answer


@dataclass(frozen=True)
class QASample:
    id: str
    split: str
    cohort: str
    patient_id: str
    case_id: str
    image: str
    mask: str
    sequence: str
    row: dict

    @property
    def qa_id(self):
        return self.row['neuroqa_id']

    def metadata(self):
        return asdict(self)

    def view(self, modality):
        if modality not in self.row['allowed_modalities']:
            raise ValueError(f'Unanswerable modality: {self.qa_id}/{modality}')
        image = self.row['images'][modality]
        return replace(self, id=f'{self.qa_id}__{modality}', sequence=SEQUENCE_IDS[modality],
                       image=image, mask=str(Path(image).with_name(f'{modality}_mni_mask.nii.gz')))


def balanced_epoch(samples, seed, epoch, policy='dual_eligible'):
    """Every QA once; balance only dual-eligible questions, alternate each epoch.

    Fixed-only questions always use their allowed sequence. For an odd number of
    dual-eligible QA the larger quota alternates T1/T2 each epoch. The assignments
    are complements, so every dual-eligible QA sees both sequences in two epochs.
    No global or optimizer-window modality quota is imposed.
    """
    if policy == 'legacy_exact':
        return _legacy_balanced_epoch(samples, seed, epoch)
    if policy == 'exact_balanced_repeat':
        return _repeat_balanced_epoch(samples, seed, epoch)
    if policy != 'dual_eligible':
        raise ValueError(f'Unknown modality sampling policy: {policy}')
    if len({s.qa_id for s in samples}) != len(samples):
        raise ValueError('Training population must contain each QA once')
    both = sorted((s for s in samples if len(s.row['allowed_modalities']) == 2), key=lambda s: s.qa_id)
    random.Random(seed).shuffle(both)
    first_half = {s.qa_id for s in both[:(len(both) + 1) // 2]}
    result = []
    for sample in samples:
        allowed = sample.row['allowed_modalities']
        if allowed == ['T1w', 'T2w']:
            modality = 'T1w' if (sample.qa_id in first_half) != bool(epoch % 2) else 'T2w'
        elif allowed in (['T1w'], ['T2w']):
            modality = allowed[0]
        else:
            raise ValueError(f'Invalid allowed modalities: {allowed}')
        result.append(sample.view(modality))
    random.Random(seed + epoch).shuffle(result)
    return result


def _repeat_balanced_epoch(samples, seed, epoch):
    """Retain every valid QA and repeat the minimum needed for exact balance.

    Eligible dual-modal QA can also be repeated using its other modality. Repeated
    views are rotated deterministically, so a small retained minority does not
    always select the same repeat. The resulting streams interleave T1/T2.
    """
    if len({s.qa_id for s in samples}) != len(samples):
        raise ValueError('Training population must contain each QA once before balancing')
    groups = {('T1w',): [], ('T2w',): [], ('T1w', 'T2w'): []}
    for sample in sorted(samples, key=lambda s: s.qa_id):
        key = tuple(sample.row['allowed_modalities'])
        if key not in groups:
            raise ValueError(f'Invalid allowed modalities: {list(key)}')
        groups[key].append(sample)
    only1, only2, both = groups[('T1w',)], groups[('T2w',)], groups[('T1w', 'T2w')]
    if not (only1 or both) or not (only2 or both):
        raise ValueError('Exact T1/T2 balance is impossible: no usable QA for one modality')
    size = len(samples)
    target = max(len(only1), len(only2), (size + 1) // 2)
    lower = max(0, len(both) - (target - len(only2)))
    upper = min(len(both), target - len(only1))
    quota = min(max((size + 1) // 2 - len(only1), lower), upper)
    random.Random(seed).shuffle(both)
    if both:
        shift = (epoch * max(1, quota)) % len(both)
        both = both[shift:] + both[:shift]
    streams = {'T1w': [s.view('T1w') for s in only1 + both[:quota]],
               'T2w': [s.view('T2w') for s in only2 + both[quota:]]}
    for modality in ('T1w', 'T2w'):
        deficit = target - len(streams[modality])
        eligible = [s for s in sorted(samples, key=lambda s: s.qa_id)
                    if modality in s.row['allowed_modalities']]
        random.Random(seed + (1 if modality == 'T1w' else 2)).shuffle(eligible)
        if deficit:
            offset = epoch * deficit
            streams[modality].extend(eligible[(offset+i) % len(eligible)].view(modality)
                                      for i in range(deficit))
        random.Random(seed + epoch * 2 + (0 if modality == 'T1w' else 1)).shuffle(streams[modality])
    result = [s for pair in zip(streams['T1w'], streams['T2w']) for s in pair]
    assert len(result) == 2 * target
    return result


def _legacy_balanced_epoch(samples, seed, epoch):
    """Every QA once; exact 50/50 epoch and every consecutive even-size window.

    Fixed-only QA are assigned first. A cyclic quota of dual-eligible QA fills
    the deficit, then each modality is shuffled and the streams interleaved.
    No independent Bernoulli sampling, duplication, or discarded tail.
    """
    if len(samples) % 2:
        raise ValueError('Exact T1/T2 balance requires an even number of unique training QA')
    if len({s.qa_id for s in samples}) != len(samples):
        raise ValueError('Training population must contain each QA once')
    groups = {('T1w',): [], ('T2w',): [], ('T1w', 'T2w'): []}
    for s in sorted(samples, key=lambda s: s.qa_id):
        groups[tuple(s.row['allowed_modalities'])].append(s)
    a, b, both = groups[('T1w',)], groups[('T2w',)], groups[('T1w', 'T2w')]
    quota = len(samples) // 2 - len(a)
    if not 0 <= quota <= len(both) or len(b) > len(samples) // 2:
        raise ValueError('Eligible modalities cannot satisfy exact balance without changing QA weights')
    random.Random(seed).shuffle(both)
    if both:
        shift = (epoch * max(1, quota)) % len(both)
        both = both[shift:] + both[:shift]
    a = [s.view('T1w') for s in a + both[:quota]]
    b = [s.view('T2w') for s in b + both[quota:]]
    rng = random.Random(seed + epoch)
    rng.shuffle(a)
    rng.shuffle(b)
    result = [s for pair in zip(a, b) for s in pair]
    assert len(result) == len(samples) and len(a) == len(b)
    return result


def load_neuroqa(config, check_paths=True):
    folder = path_of(config, config.data.manifest_dir)
    lineage = json.loads((folder / 'lineage.json').read_text())
    if lineage['protocol'] != PROTOCOL or config.data.target_version != PROTOCOL:
        raise ValueError('NeuroQA split protocol mismatch')
    populations, summary, patients, ids = {}, {}, {}, set()
    for split in ('train', 'val', 'test'):
        rows = read_rows(folder / f'{split}.jsonl')
        samples = []
        for row in rows:
            qid = row['neuroqa_id']
            if qid in ids or row['split'] != split or row['patient_id'] != patient_id(row['subject_id']):
                raise ValueError(f'NeuroQA identity/split mismatch: {qid}')
            ids.add(qid)
            target(row)
            if row['category'] == 'Modality':
                raise ValueError('Modality category excluded from this protocol')
            if row['allowed_modalities'] != [m for m in MODALITIES if m in row['answerable_modalities']]:
                raise ValueError('Answerable modality mismatch')
            base = QASample(qid, split, row['dataset'].replace('-', '_'), row['patient_id'],
                            row['subject_id'], '', '', '', row)
            if check_paths:
                for m in row['allowed_modalities']:
                    for path in (base.view(m).image, base.view(m).mask):
                        if not Path(path).is_file():
                            raise FileNotFoundError(path)
            samples.extend([base.view(row['allowed_modalities'][0])] if split == 'train' else
                           [base.view(m) for m in row['allowed_modalities']])
        if not samples:
            raise ValueError(f'Empty NeuroQA {split}')
        populations[split] = samples
        patients[split] = {s.patient_id for s in samples}
        summary[split] = {'samples': len(samples), 'unique_qa': len(rows),
                          'patients': len(patients[split]), 'sha256': file_hash(folder / f'{split}.jsonl')}
    for a, b in (('train', 'val'), ('train', 'test'), ('val', 'test')):
        if patients[a] & patients[b]:
            raise ValueError(f'Patient leakage: {a}/{b}')
    return populations, summary
