"""Audit frozen GREEN text pairs, then fill only GREEN cells in the V3 workbook.

No model loading, rescoring, or writes to training/prediction artifacts.
Default is a read-only audit. --apply backs up both workbooks and atomically
updates the existing Graph V3 visual-dependence workbook after all checks pass.
"""
import argparse
from collections import Counter
import csv
from datetime import datetime
from io import BytesIO
import json
import math
from pathlib import Path
import statistics
from zoneinfo import ZoneInfo

import openpyxl
from src.evaluation.clinical import digest, file_digest, validate_result
from src.evaluation.significance_io import scorer_contract

ROOT = Path(__file__).resolve().parents[2]
VIS = ROOT / 'results/anchorread3d_final_eval_1108/visual'
SCORING = VIS / 'scoring'
SOURCE = ROOT / 'results/AnchorRead3D_RG_VisualDependence_1108.xlsx'
TARGET = ROOT / 'results/AnchorRead3D_RG_VisualDependence_GraphV3_1108.xlsx'


def read(path):
    return json.loads(Path(path).read_text())


def rows(path):
    with Path(path).open() as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def write(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def same(a, b):
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        assert math.isclose(a, b, rel_tol=0, abs_tol=1e-12), (a, b)
    else:
        assert a == b, (a, b)


def mean(values):
    return math.fsum(values) / len(values)


def record_index(ws, keys):
    header = [c.value for c in ws[1]]
    result = {}
    for cells in ws.iter_rows(min_row=2):
        row = dict(zip(header, [c.value for c in cells]))
        key = tuple(row[k] for k in keys)
        assert key not in result, (ws.title, key)
        result[key] = (cells[0].row, row)
    return header, result


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--apply', action='store_true')
    cli.add_argument('--audit-dir', type=Path,
                     default=ROOT / 'results/green_visual_fillback_20261110')
    args = cli.parse_args()
    args.audit_dir.mkdir(parents=True, exist_ok=True)
    if args.apply:
        assert not (args.audit_dir / 'backups').exists(), 'Existing application backup; use a fresh audit directory.'
    workflow = read(VIS.parent / 'workflow_status.json')
    assert workflow['status'] == 'complete'
    assert file_digest(SOURCE) == workflow['visual']['sha256']
    protected = {}

    def protect(path):
        path = Path(path)
        protected[str(path)] = file_digest(path)
        return path

    protect(SOURCE)
    for name in ['Experiments for VLM Projector 1108.xlsx',
                 'Experiments for VLM Projector 1108_pvalues.xlsx',
                 'AnchorRead3D_GREEN_GraphV2_1108.xlsx',
                 'AnchorRead3D_GREEN_GraphV3_1108.xlsx',
                 'AnchorRead3D_GraphV3_1108.xlsx']:
        protect(ROOT / 'results' / name)
    source_bytes, target_bytes = SOURCE.read_bytes(), TARGET.read_bytes()
    target_hash = file_digest(TARGET)
    source = openpyxl.load_workbook(BytesIO(source_bytes))
    target = openpyxl.load_workbook(BytesIO(target_bytes))
    plan = read(protect(VIS / 'plan.json'))
    index = read(protect(ROOT / 'results/anchorread3d_analysis_1108/artifact_reconciliation_1108.json'))
    entries = {e['key']: e for e in index['paper_settings']}
    assert len(plan['settings']) == 8 and len(plan['seeds']) == 3
    manifest = read(protect(SCORING / 'pool_manifest.json'))
    assert manifest['source_plan_sha256'] == file_digest(VIS / 'plan.json')
    pairs = {}
    for r in rows(protect(SCORING / 'pairs.jsonl')):
        key = digest([r['reference'], r['prediction']])
        assert key == r['id'] and key not in pairs
        pairs[key] = (r['reference'], r['prediction'])
    assert set(pairs) == set(manifest['pair_ids']) and len(pairs) == manifest['unique_pairs']
    cached = {k: v['green'] for k, v in read(protect(SCORING / 'cached_original_pairs.json')).items()}
    assert len(cached) == manifest['cached_original_pairs'] and set(cached) <= set(pairs)
    bindings = {}
    for r in read(protect(SCORING / 'bindings.json')):
        key = (r['setting'], r['condition'], r['id'])
        assert key not in bindings and r['pair_key'] in pairs
        bindings[key] = r
    assert len(bindings) == manifest['bindings'] == 26880
    expected_spec = read(protect(Path(entries[plan['settings'][0]]['paths']['metric']).parent /
                                'scoring/scoring_provenance.json'))['scorers']['green']
    scored = {}
    statuses = Counter()
    for i in range(4):
        inputs = {r['id']: r for r in rows(protect(SCORING / f'shard_{i}.jsonl'))}
        expected_ids = read(protect(SCORING / f'shard_{i}_ids.json'))
        prov = read(protect(SCORING / f'green_{i}/scoring_provenance.json'))
        assert set(inputs) == set(expected_ids) and len(inputs) == len(expected_ids)
        assert prov['input_sha256'] == file_digest(SCORING / f'shard_{i}.jsonl')
        assert prov['expected_ids_sha256'] == file_digest(SCORING / f'shard_{i}_ids.json')
        assert digest(scorer_contract(prov['scorers']['green'])) == digest(scorer_contract(expected_spec))
        seen = set()
        for r in rows(protect(SCORING / f'green_{i}/clinical_scores.jsonl')):
            key = r['id']
            assert key not in seen and key not in scored and key not in cached
            seen.add(key)
            assert key == digest([r['reference'], r['prediction']])
            assert pairs[key] == (r['reference'], r['prediction'])
            assert {k: v for k, v in r.items() if k != 'clinical'} == inputs[key]
            g = r['clinical']['green']
            validate_result('green', g)
            assert g['pair_hash'] == digest({k: r[k] for k in ['reference', 'prediction']})
            assert g['scorer_fingerprint'] == digest(prov['scorers']['green'])
            assert g['status'] in ['ok', 'format_repair_failed', 'output_truncated']
            if g['status'] == 'ok':
                matched, errors = g['details']['matched_findings'], g['details']['significant_errors']
                assert len(errors) == 6 and matched >= 0 and all(x >= 0 for x in errors)
                same(g['values']['green'], matched / (matched + sum(errors)) if matched + sum(errors) else 0)
            scored[key] = g
            statuses[g['status']] += 1
        assert seen == set(expected_ids)
    assert set(scored) == set(pairs) - set(cached)
    verified = read(protect(SCORING / 'green_verified.json'))
    assert verified['status'] == 'verified' and verified['pairs'] == len(scored) == 14959
    assert dict(statuses) == verified['coverage']['green']
    all_scores = {**cached, **scored}
    for key, g in all_scores.items():
        validate_result('green', g)
        assert g['pair_hash'] == digest(dict(zip(['reference', 'prediction'], pairs[key])))
    summary, failures = {}, []
    checked_bindings = set()
    for setting in plan['settings']:
        _, method, ratio = setting.split('/')
        ratio = int(ratio[1:])
        entry = entries[setting]
        folder = VIS / setting.replace('/', '__')
        completion = read(protect(folder / 'complete.json'))
        assert completion['status'] == 'complete' and completion['setting'] == setting
        assert completion['plan_sha256'] == file_digest(VIS / 'plan.json')
        original = {r['id']: r for r in rows(protect(entry['paths']['predictions']))}
        assert file_digest(entry['paths']['predictions']) == plan['source_prediction_hashes'][setting]
        original_scored = {r['id']: r for r in rows(protect(Path(entry['paths']['metric']).parent /
                                                         'scoring/clinical_scores.jsonl'))}
        metric = read(protect(entry['paths']['metric']))
        assert set(original) == set(original_scored) and len(original) == 560
        conditions = ['normal', 'zero', 'text'] + [f'shuffle_{s}' for s in plan['seeds']]
        assert set(completion['outputs']) == {c + '/predictions.jsonl' for c in conditions}
        for condition in conditions:
            path = protect(folder / condition / 'predictions.jsonl')
            assert file_digest(path) == completion['outputs'][condition + '/predictions.jsonl']
            good, patients, status, seen = [], set(), Counter(), set()
            for r in rows(path):
                rid = r['id']
                assert rid in original and rid not in seen
                seen.add(rid)
                key = digest([r['reference'], r['prediction']])
                binding_key = setting, condition, rid
                binding = bindings[binding_key]
                checked_bindings.add(binding_key)
                assert binding['pair_key'] == key and Path(binding['prediction_path']) == path
                assert binding['patient_id'] == r['patient_id']
                assert pairs[key] == (r['reference'], r['prediction'])
                for field in ['reference', 'patient_id', 'case_id', 'cohort', 'sequence', 'checkpoint_id']:
                    assert r[field] == original[rid][field], (setting, condition, rid, field)
                assert r['checkpoint_id'] == metric['checkpoint_sha256']
                if condition == 'normal':
                    assert r['prediction'] == original[rid]['prediction']
                    assert original_scored[rid]['reference'] == r['reference']
                    assert original_scored[rid]['prediction'] == r['prediction']
                    g = original_scored[rid]['clinical']['green']
                else:
                    g = all_scores[key]
                    assert r['control'] == ('shuffle' if condition.startswith('shuffle_') else condition)
                    if condition.startswith('shuffle_'):
                        seed = condition.split('_')[1]
                        donor = plan['donors'][seed][rid]
                        assert r['image_donor_id'] == donor
                        assert original[donor]['patient_id'] != r['patient_id']
                        assert original[donor]['sequence'] == r['sequence']
                        assert original[donor]['cohort'] == r['cohort']
                        assert r['prediction'] == original[donor]['prediction']
                validate_result('green', g)
                status[g['status']] += 1
                if g['status'] == 'ok':
                    good.append(g)
                    patients.add(r['patient_id'])
                else:
                    failures.append(dict(method=method, ratio=ratio, condition=condition, id=rid,
                                         pair_key=key, status=g['status']))
            assert seen == set(original) and good and sum(status.values()) == 560
            row = dict(method=method, ratio=ratio, condition=condition,
                       green=mean([g['values']['green'] for g in good]), valid=len(good), n=560,
                       eligible_patients=len(patients), patients=140, statuses=dict(status))
            row.update({f'error_{letter}': mean([g['details']['significant_errors'][i] for g in good])
                        for i, letter in enumerate('abcdef')})
            row['matched_findings'] = mean([g['details']['matched_findings'] for g in good])
            summary[method, ratio, condition] = row
            if condition == 'normal':
                same(row['green'], metric['overall']['green'])
    assert checked_bindings == set(bindings) and len(summary) == 48
    main = {}
    for method, ratio, _ in summary:
        for condition in ['normal', 'zero', 'text', 'shuffle']:
            keys = [(method, ratio, f'shuffle_{seed}') for seed in plan['seeds']] if condition == 'shuffle' else [(method, ratio, condition)]
            scores = [summary[k]['green'] for k in keys]
            main[method, ratio, condition] = dict(green=mean(scores), sd=statistics.stdev(scores) if len(scores) == 3 else None)

    # All expected workbook values come from the independently reconstructed
    # method/ratio/condition summaries. Match named keys, never row positions.
    updates = {}

    def transfer(sheet, keys, expected_rows):
        header, original_rows = record_index(source[sheet], keys)
        target_header, target_rows = record_index(target[sheet], keys)
        assert header == target_header and original_rows.keys() == target_rows.keys()
        for key, values in expected_rows.items():
            source_row, recorded = original_rows[key]
            target_row, previous = target_rows[key]
            for field, expected in values.items():
                actual = recorded[field]
                if field == 'statuses':
                    same(json.loads(actual), expected)
                else:
                    same(actual, expected)
                if field not in ['valid', 'eligible_patients', 'statuses'] and previous[field] is not None:
                    same(previous[field], actual)
                updates[sheet, target_row, header.index(field) + 1] = actual

    for ratio in [32, 256]:
        transfer(f'RG_{ratio}x', ['Method', 'Condition'],
                 {(m, c): {'GREEN ↑': v['green']} for (m, r, c), v in main.items() if r == ratio})
    transfer('By_seed', ['method', 'ratio', 'condition'], {k: {'green': v['green']} for k, v in summary.items()})
    fields = ['valid'] + [f'error_{x}' for x in 'abcdef'] + ['matched_findings']
    transfer('GREEN_parts', ['method', 'ratio', 'condition'], {k: {f: v[f] for f in fields} for k, v in summary.items()})
    transfer('Coverage', ['method', 'ratio', 'condition', 'metric'],
             {(*k, 'green'): {f: v[f] for f in ['valid', 'n', 'eligible_patients', 'patients', 'statuses']} for k, v in summary.items()})
    transfer('Shuffle_seed_SD', ['method', 'ratio'],
             {(m, r): {'green_seed_sd': v['sd']} for (m, r, c), v in main.items() if c == 'shuffle'})
    transfer('Normal_minus_control', ['method', 'ratio', 'condition', 'metric'],
             {(*k, 'green'): dict(normal=main[k[0], k[1], 'normal']['green'], control=v['green'],
                                  normal_minus_control=main[k[0], k[1], 'normal']['green'] - v['green'])
              for k, v in main.items() if k[2] != 'normal'})
    notes = {target['说明'].cell(r, 1).value: r for r in range(1, target['说明'].max_row + 1)}
    updates['说明', notes['状态'], 2] = '全部阶段已验收（GREEN已完成；Graph V3保持原值）'
    note_values = {
        'GREEN完成时间': datetime.fromtimestamp(workflow['time'], ZoneInfo('Asia/Shanghai')).isoformat(),
        'GREEN有效性': '新增14959对去重配对全部处理：14920有效，38格式解析失败、1输出截断；失败不记0，不纳入均值；逐设置有效数见Coverage。',
        'GREEN回填来源': str(SOURCE),
        'GREEN回填来源SHA256': file_digest(SOURCE),
        'GREEN回填审计': str(args.audit_dir / 'audit.json'),
    }
    next_row = target['说明'].max_row + 1
    for title, value in note_values.items():
        row = notes.get(title, next_row)
        if title not in notes:
            next_row += 1
        updates['说明', row, 1] = title
        updates['说明', row, 2] = value
    changes = []
    for (sheet, row, col), value in updates.items():
        cell = target[sheet].cell(row, col)
        if cell.value != value:
            changes.append(dict(sheet=sheet, cell=cell.coordinate, old=cell.value, new=value))
            cell.value = value
    stream = BytesIO()
    target.save(stream)
    new_bytes = stream.getvalue()
    saved = openpyxl.load_workbook(BytesIO(new_bytes))
    before = openpyxl.load_workbook(BytesIO(target_bytes))
    assert saved.sheetnames == before.sheetnames
    preserved = 0
    for old in before:
        new = saved[old.title]
        assert list(old.merged_cells.ranges) == list(new.merged_cells.ranges)
        assert len(old._charts) == len(new._charts)
        for row in range(1, max(old.max_row, new.max_row) + 1):
            for col in range(1, max(old.max_column, new.max_column) + 1):
                a, b = old.cell(row, col), new.cell(row, col)
                if (old.title, row, col) not in updates:
                    assert a.value == b.value, ('UNEXPECTED CHANGE', old.title, a.coordinate)
                    preserved += 1
                else:
                    same(b.value, updates[old.title, row, col])
                assert a._style == b._style or (not a.has_style and not b.has_style), ('STYLE CHANGE', old.title, a.coordinate)
    for path, checksum in protected.items():
        assert file_digest(path) == checksum, ('SOURCE CHANGED', path)
    assert file_digest(TARGET) == target_hash
    result = dict(status='verified_dry_run', new_unique_pairs=len(scored), new_pair_statuses=dict(statuses),
                  total_unique_pairs=len(pairs), report_bindings=len(bindings), conditions=len(summary),
                  workflow_completed_at=note_values['GREEN完成时间'],
                  source_workbook=str(SOURCE), source_workbook_sha256=file_digest(SOURCE),
                  source_workbook_already_complete=True, target_workbook=str(TARGET),
                  target_before_sha256=target_hash, non_green_cells_preserved=preserved,
                  all_graph_v3_cells_preserved=True, all_styles_preserved=True,
                  changed_cells=len(changes), changes=changes, protected_files=protected,
                  main_rows=[dict(method=m, ratio=r, condition=c, **v) for (m, r, c), v in main.items()],
                  by_seed=list(summary.values()), failed_report_bindings=len(failures),
                  script_sha256=file_digest(Path(__file__)))
    write(args.audit_dir / 'audit.json', result)
    with (args.audit_dir / 'failed_report_bindings.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['method', 'ratio', 'condition', 'id', 'pair_key', 'status'])
        writer.writeheader()
        writer.writerows(failures)
    if args.apply:
        backup = args.audit_dir / 'backups'
        backup.mkdir()
        (backup / SOURCE.name).write_bytes(source_bytes)
        (backup / TARGET.name).write_bytes(target_bytes)
        assert file_digest(backup / TARGET.name) == target_hash
        assert file_digest(backup / SOURCE.name) == protected[str(SOURCE)]
        assert file_digest(TARGET) == target_hash
        temp = TARGET.with_name(TARGET.stem + '.green_fillback.tmp.xlsx')
        assert not temp.exists()
        with temp.open('xb') as f:
            f.write(new_bytes)
        temp.replace(TARGET)
        assert TARGET.read_bytes() == new_bytes
        result.update(status='complete', target_after_sha256=file_digest(TARGET), backup_directory=str(backup))
        write(args.audit_dir / 'audit.json', result)
    print(json.dumps({k: v for k, v in result.items() if k not in ['changes', 'protected_files', 'main_rows', 'by_seed']},
                     ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
