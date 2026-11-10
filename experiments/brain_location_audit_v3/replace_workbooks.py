"""Replace only custom spatial metrics in copies of the two frozen workbooks.

Run from the repository root: python -m experiments.brain_location_audit_v3.replace_workbooks
No inference or writes to source workbooks; existing destinations are refused.
"""
import argparse
from collections import Counter, defaultdict
from copy import copy
import hashlib
from io import BytesIO
import json
from pathlib import Path
import statistics

import openpyxl
from openpyxl.xml.functions import tostring

from . import protocol

ROOT = Path(__file__).resolve().parents[2]
AUDIT = ROOT / 'results/brain_location_graph_v3_1108'
DIMS = ('side', 'site', 'joint')
METRICS = {'micro_f1': 'f1', 'em': 'em', 'precision': 'precision', 'recall': 'recall'}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def read(name):
    return json.loads((AUDIT / name).read_text())


def lines(name):
    with (AUDIT / name).open() as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def records(ws):
    header = [c.value for c in ws[1]]
    for row in ws.iter_rows(min_row=2):
        yield row[0].row, dict(zip(header, [c.value for c in row]))


def close(a, b):
    assert abs(a - b) < 1e-12, (a, b)


class Editor:
    def __init__(self, source, destination):
        self.source, self.destination = source, destination
        assert not destination.exists(), f'Refusing existing destination: {destination}'
        self.raw = source.read_bytes()
        assert digest(self.raw) == digest(source.read_bytes()), 'Source changed while snapshotting; retry.'
        self.original = openpyxl.load_workbook(BytesIO(self.raw))
        self.book = openpyxl.load_workbook(BytesIO(self.raw))
        self.allowed = set()
        self.mapping = {s: s for s in self.book.sheetnames}

    def put(self, sheet, row, col, value):
        self.allowed.add((sheet, row, col))
        self.book[sheet].cell(row, col).value = value

    def field(self, sheet, row, key, value):
        header = [c.value for c in self.book[sheet][1]]
        self.put(sheet, row, header.index(key) + 1, value)

    def note(self, key, value, newkey=None):
        ws = self.book['说明']
        row = next((r for r in range(2, ws.max_row + 1) if ws.cell(r, 1).value == key), ws.max_row + 1)
        self.put('说明', row, 1, newkey or key)
        self.put('说明', row, 2, value)

    def rename(self, old, new):
        self.book[old].title = new
        self.mapping[old] = new

    def finish(self, tag):
        self.note('graph_protocol', protocol.VERSION)
        self.note('graph_source', str(AUDIT / 'scores.jsonl'))
        self.note('graph_source_sha256', digest((AUDIT / 'scores.jsonl').read_bytes()))
        self.note('graph_reference_source', str(AUDIT / 'references.json'))
        self.note('graph_reference_sha256', digest((AUDIT / 'references.json').read_bytes()))
        self.note('source_workbook_sha256', digest(self.raw))
        self.note('export_script', str(Path(__file__).resolve()))
        self.note('export_script_sha256', digest(Path(__file__).read_bytes()))
        buffer = BytesIO()
        self.book.save(buffer)
        output = buffer.getvalue()
        saved = openpyxl.load_workbook(BytesIO(output))
        assert saved.sheetnames == list(self.mapping.values())
        changes, counts = [], Counter()
        preserved = 0
        # Compare every cell, including formatting and formulas, after XLSX round-trip.
        for oldname, newname in self.mapping.items():
            old, new = self.original[oldname], saved[newname]
            assert list(old.merged_cells.ranges) == list(new.merged_cells.ranges)
            for row in range(1, max(old.max_row, new.max_row) + 1):
                for col in range(1, max(old.max_column, new.max_column) + 1):
                    a, b = old.cell(row, col), new.cell(row, col)
                    allowed = (oldname, row, col) in self.allowed
                    if not allowed:
                        assert a.value == b.value, ('UNAUTHORIZED VALUE', oldname, a.coordinate, a.value, b.value)
                        assert a._style == b._style, ('UNAUTHORIZED STYLE', oldname, a.coordinate)
                        preserved += 1
                    if a.value != b.value or a._style != b._style:
                        changes.append(dict(sheet=newname, cell=a.coordinate, old=a.value, new=b.value,
                                            style_changed=a._style != b._style))
                        counts[newname] += 1
            assert len(old._charts) == len(new._charts)
        if 'Charts' in saved.sheetnames:
            oldcharts, newcharts = self.original['Charts']._charts, saved['Charts']._charts
            assert tostring(oldcharts[3].to_tree()) == tostring(newcharts[3].to_tree())
            for chart in newcharts[:3]:
                for series in chart.ser:
                    assert 'V3_F1' in series.val.numRef.f and 'V3_F1' in series.cat.numRef.f
        assert digest(self.source.read_bytes()) == digest(self.raw), 'Live source changed; rerun from latest snapshot.'
        # Exclusive creation: never replace an existing user artifact.
        with self.destination.open('xb') as f:
            f.write(output)
        result = dict(source=str(self.source), source_sha256=digest(self.raw),
                      output=str(self.destination), output_sha256=digest(output),
                      preserved_cells=preserved, changed_cells=len(changes),
                      changed_by_sheet=dict(counts), sheet_mapping=self.mapping,
                      all_non_graph_cells_preserved=True, changes=changes)
        (AUDIT / f'{tag}_replacement_audit.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
        print(json.dumps({k: v for k, v in result.items() if k != 'changes'}, ensure_ascii=False, indent=2))
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suffix', default='1108', help='Use a fresh suffix for a later GREEN snapshot.')
    args = parser.parse_args()
    status = read('status.json')
    assert status['status'] == 'complete' and status['protocol'] == protocol.VERSION
    assert digest(Path(protocol.__file__).read_bytes()) == status['code_sha256']['protocol.py']
    summary = {(r['setting'], r['dimension'], r['version']): r for r in read('summary.json') if r['scope'] == 'all'}
    paired = {(r['ratio'], r['dimension'], r['metric'], r['comparator']): r for r in read('paired_ci.json')}
    refs = {r['id']: r for r in read('references.json')}

    def get(key, dim, version='v3'):
        return summary[key, dim, version]

    # First workbook: GREEN data and original artifact/source indexes are retained.
    e = Editor(ROOT / 'results/AnchorRead3D_GREEN_GraphV2_1108.xlsx',
               ROOT / f'results/AnchorRead3D_GREEN_GraphV3_{args.suffix}.xlsx')
    for sheet, met in [('V2_F1', 'f1'), ('V2_EM', 'em'), ('V2_Precision', 'precision'), ('V2_Recall', 'recall')]:
        ws = e.book[sheet]
        dim = None
        for row in range(1, ws.max_row + 1):
            name = ws.cell(row, 1).value
            if name and str(name).startswith(('侧别 ', '脑区 ', '侧别＋脑区联合 ')):
                dim = {'侧别': 'side', '脑区': 'site', '侧别＋脑区联合': 'joint'}[name.split()[0]]
            if name not in ['MLP (1x)', 'AvgPool3D', 'Concat3D', 'ConvPool3D', 'LocalPacker3D',
                            'Perceiver_NV', 'Adaptive_Merge3D', 'Token_Select3D', 'AnchorRead3D']:
                continue
            method = {'AnchorRead3D': 'A0', 'MLP (1x)': 'MLP'}.get(name, name)
            for col in range(2, 9):
                if ws.cell(row, col).value is None:
                    continue
                ratio = [1, 8, 16, 32, 64, 128, 256][col - 2]
                key = f'rg/{method}/r{ratio}'
                close(ws.cell(row, col).value, get(key, dim, 'v2')[met])
                e.put(sheet, row, col, get(key, dim)[met])
        for start in [4, 16, 28]:
            for col in range(3, 9):
                best = max(ws.cell(row, col).value for row in range(start, start + 8))
                for row in range(start, start + 8):
                    cell = ws.cell(row, col)
                    font = copy(cell.font)
                    font.bold = abs(cell.value - best) < 1e-12
                    cell.font = font
    for sheet in ['V2_summary', 'V2_ablations']:
        for row, r in records(e.book[sheet]):
            met = METRICS[r['metric']]
            old, new = get(r['setting'], r['dimension'], 'v2'), get(r['setting'], r['dimension'])
            close(r['value'], old[met])
            updates = dict(n=new['reports'], patients=new['patients'], exact_count=round(new['em'] * new['reports']),
                           tp=new['tp'], fp=new['fp'], fn=new['fn'], value=new[met],
                           ci_low=new[met + '_ci_low'], ci_high=new[met + '_ci_high'])
            for key, value in updates.items():
                e.field(sheet, row, key, value)
    for row, r in records(e.book['V2_paired_CI']):
        new = paired[r['ratio'], r['dimension'], METRICS[r['metric']], r['comparator']]
        for key in ['ours', 'other', 'difference', 'ci_low', 'ci_high']:
            e.field('V2_paired_CI', row, key, new['baseline' if key == 'other' else key])
    for row, r in records(e.book['V2_reference_coverage']):
        new = refs[r['id']]
        for dim in DIMS:
            e.field('V2_reference_coverage', row, dim + '_eligible', new['v3_' + dim + '_eligible'])
            labels = new['v3_' + dim + '_labels']
            if dim == 'joint':
                labels = [tuple(x) for x in labels]
            e.field('V2_reference_coverage', row, dim + '_labels', str(sorted(labels)))
        e.field('V2_reference_coverage', row, 'flags', json.dumps(new['v3_flags'], ensure_ascii=False))
    for chart in e.book['Charts']._charts[:3]:
        for series in chart.ser:
            for ref in [series.val.numRef, series.cat.numRef]:
                ref.f = ref.f.replace('V2_F1', 'V3_F1')
                ref.numCache = None
            if series.tx and series.tx.strRef:
                series.tx.strRef.f = series.tx.strRef.f.replace('V2_F1', 'V3_F1')
                series.tx.strRef.strCache = None
    e.note('范围', '1108 主实验；GREEN 49 设置保持原值；图 V3 57 设置（含 RG A1–A4）；不包含 VQA 的图位置指标。')
    e.note('V2 分母', '全 560 条参考报告；侧别 456 条/135 患者，脑区 468 条/137 患者，联合 452 条/134 患者。不是 GREEN 的 505 条交集。', 'V3 分母')
    e.note('V2 指标', 'micro_f1=2TP/(2TP+FP+FN)；precision/recall 为微平均；EM 为可评价标签集合完全一致率；参考明确缺失字段对应的不可判定预测不计对错。', 'V3 指标')
    e.note('V2 状态', '自动提取的位置事实一致性；非人工核验的影像定位准确率，非病灶实例匹配。V3 扩展明确局部表达，采用部分参考标签；仍有未解析信息。', 'V3 状态')
    e.note('参考限定', '每维参考有至少一个明确阳性标签即纳入。未知侧别不生成联合标签；预测缺失/否定/不确定不移除已纳入报告；兼容部分参考的不确定字段不计对错。')
    for name in list(e.book.sheetnames):
        if name.startswith('V2_'):
            e.rename(name, name.replace('V2_', 'V3_', 1))
    e.finish('green_' + args.suffix)

    # Raw predicted label sets for output-change rates; these are not reference-masked.
    labels = {r['graph_sha256']: protocol.label_sets(r['v3']) for r in lines('extractions.jsonl')}
    predictions = defaultdict(dict)
    for r in lines('scores.jsonl'):
        if len(r['setting'].split('/')) == 4:
            predictions[r['setting']][r['id']] = r['prediction_graph_sha256']
    assert len(predictions) == 48 and all(len(x) == 560 for x in predictions.values())

    def visual(method, ratio, condition, dim, metric='f1'):
        prefix = f'rg/{method}/r{ratio}/'
        keys = sorted(k for k in predictions if k.startswith(prefix + 'shuffle_')) if condition == 'shuffle' else [prefix + condition]
        assert len(keys) == (3 if condition == 'shuffle' else 1)
        return statistics.mean(get(key, dim)[metric] for key in keys)

    e = Editor(ROOT / 'results/AnchorRead3D_RG_VisualDependence_1108.xlsx',
               ROOT / f'results/AnchorRead3D_RG_VisualDependence_GraphV3_{args.suffix}.xlsx')
    for ratio in [32, 256]:
        sheet = f'RG_{ratio}x'
        for row, r in records(e.book[sheet]):
            for dim, col in zip(DIMS, [15, 16, 17]):
                e.put(sheet, row, col, visual(r['Method'], ratio, r['Condition'], dim))
    for row, r in records(e.book['By_seed']):
        key = f"rg/{r['method']}/r{r['ratio']}/{r['condition']}"
        normal = predictions[key.rsplit('/', 1)[0] + '/normal']
        for dim in DIMS:
            close(r[dim + '_f1'], get(key, dim, 'v2')['f1'])
            e.field('By_seed', row, dim + '_f1', get(key, dim)['f1'])
            change = sum(labels[graph][dim] != labels[normal[rid]][dim] for rid, graph in predictions[key].items()) / 560
            e.field('By_seed', row, dim + '_prediction_change_rate', change)
    for row, r in records(e.book['Location_details']):
        key = f"rg/{r['method']}/r{r['ratio']}/{r['condition']}"
        for dim in DIMS:
            for metric in ['precision', 'recall', 'f1', 'em']:
                e.field('Location_details', row, dim + '_' + metric, get(key, dim)[metric])
    for row, r in records(e.book['Shuffle_seed_SD']):
        prefix = f"rg/{r['method']}/r{r['ratio']}/shuffle_"
        keys = [key for key in predictions if key.startswith(prefix)]
        assert len(keys) == 3
        for dim in DIMS:
            e.field('Shuffle_seed_SD', row, dim + '_f1_seed_sd', statistics.stdev(get(key, dim)['f1'] for key in keys))
    for row, r in records(e.book['Coverage']):
        if r['metric'].startswith('graph_'):
            dim = r['metric'][6:]
            new = get(f"rg/{r['method']}/r{r['ratio']}/{r['condition']}", dim)
            e.field('Coverage', row, 'valid', new['reports'])
            e.field('Coverage', row, 'eligible_patients', new['patients'])
            e.field('Coverage', row, 'statuses', json.dumps({k: new[k] for k in ['reports', 'patients', 'tp', 'fp', 'fn']}))
    for row, r in records(e.book['Normal_minus_control']):
        if r['metric'] in [d + '_f1' for d in DIMS]:
            dim = r['metric'].split('_')[0]
            normal = visual(r['method'], r['ratio'], 'normal', dim)
            control = visual(r['method'], r['ratio'], r['condition'], dim)
            for k, v in [('normal', normal), ('control', control), ('normal_minus_control', normal - control)]:
                e.field('Normal_minus_control', row, k, v)
    e.note('Graph V2', 'V3 自动抽取的报告位置事实集合一致性；扩展局部明确表达、采用部分参考标签。非影像金标准，非病灶数量/实例匹配；仍有解析误差。', 'Graph V3')
    e.note('分母', 'Graph 各维参考至少一个已知阳性标签即纳入：侧别456/135患者，脑区468/137，联合452/134。兼容部分参考的未知预测字段不判对错。GREEN等失败不记0，单列有效数量；临床分项为有效报告均值。')
    e.note('输出变化率', '相对同患者Normal的文本/V3原始预测位置标签变化比例（560条）；位置变化率不使用参考的部分标签屏蔽，变化本身不代表更正确。')
    e.finish('visual_' + args.suffix)


if __name__ == '__main__':
    main()
