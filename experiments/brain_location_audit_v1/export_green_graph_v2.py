"""Export the frozen 1108 GREEN and graph-V2 audits without model inference."""
import argparse
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import openpyxl
from openpyxl.chart import LineChart, Reference
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter

from .compare_budgets_v2 import BASELINES, DIMS, fractions, interval
from .protocol import label_sets, score_sets
from .protocol_v2 import VERSION

ROOT = Path(__file__).resolve().parents[2]
RATIOS = [8, 16, 32, 64, 128, 256]
METHODS = BASELINES + ['A0']
DISPLAY = {'A0': 'AnchorRead3D', 'MLP': 'MLP (1x)', 'A1': 'A1 no LocalRead',
           'A2': 'A2 no mean bypass', 'A3': 'A3 global only', 'A4': 'A4 local only'}
GREEN_NAMES = {
    'c_location': '位置错误 (c) ↓', 'a_false_finding': '虚构发现 (a) ↓',
    'b_missing_finding': '遗漏发现 (b) ↓', 'd_severity': '严重程度错误 (d) ↓',
    'e_spurious_comparison': '虚构既往检查比较 (e) ↓',
    'f_missing_comparison': '遗漏既往检查比较 (f) ↓',
    'green': 'GREEN 总分 ↑', 'matched_findings': '匹配发现数/报告 ↑',
    'location_error_report_rate': '含位置错误的报告比例 ↓'}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def csv_rows(path):
    return list(csv.DictReader(Path(path).open()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    index_path = ROOT/'results/anchorread3d_analysis_1108/artifact_reconciliation_1108.json'
    source = ROOT/'results/radmatch_comparison_1108/full_inputs/comparison_inputs.json'
    green_dir = ROOT/'results/radmatch_comparison_1108/green_all_budgets_1108'
    index = json.loads(index_path.read_text())
    assert sha(index['master']) == index['master_sha256']
    entries = {r['key']: r for r in index['paper_settings']}
    missing = [(e['key'], k, v) for e in entries.values() for k, v in e['paths'].items()
               if v and (not Path(v).exists() or (Path(v).is_file() and Path(v).stat().st_size == 0))]
    assert not missing, missing
    gm = json.loads((green_dir/'metadata.json').read_text())
    assert gm['master_sha256'] == index['master_sha256']
    for p in gm['provenance']:
        assert sha(p['clinical_path']) == p['clinical_sha256']
    groups = defaultdict(dict)
    for row in json.loads(source.read_text()):
        assert row['graph_protocol'] == VERSION
        assert row['id'] not in groups[row['setting']]
        assert row['graph_scores'] == score_sets(row['reference_extraction'], row['prediction_extraction'])
        groups[row['setting']][row['id']] = row
    assert len(groups) == 57 and sum(map(len, groups.values())) == 31920
    canonical = groups['rg/A0/r32']
    patients = sorted({r['patient_id'] for r in canonical.values()})
    assert len(canonical) == 560 and len(patients) == 140
    pi = {p: i for i, p in enumerate(patients)}
    draws = np.random.default_rng(20261108).multinomial(140, np.full(140, 1/140), size=10000)
    prior = {(r['setting'], r['dimension']): r for r in csv_rows(
        ROOT/'results/radmatch_comparison_1108/version_comparison/all_versions.csv')
        if r['version'] == 'graph_v2'}
    values, boot, summaries, sources = {}, {}, [], []
    for setting, rows in sorted(groups.items()):
        assert set(rows) == set(canonical)
        clinical = Path(entries[setting]['paths']['metric']).parent/'scoring/clinical_scores.jsonl'
        actual = {r['id']: r for r in map(json.loads, clinical.read_text().splitlines())}
        assert set(actual) == set(rows)
        for rid, row in rows.items():
            assert row['reference_extraction'] == canonical[rid]['reference_extraction']
            assert row['patient_id'] == canonical[rid]['patient_id']
            for field in ['reference', 'prediction']:
                assert hashlib.sha256(actual[rid][field].encode()).hexdigest() == row[field+'_sha256']
        sources.append(dict(setting=setting, clinical_path=str(clinical), clinical_sha256=sha(clinical)))
        for dim in DIMS:
            stats = np.zeros((140, 5))
            eligible_patients = set()
            for row in rows.values():
                s = row['graph_scores'][dim]
                if s['eligible']:
                    stats[pi[row['patient_id']]] += [1, s['exact'], s['tp'], s['fp'], s['fn']]
                    eligible_patients.add(row['patient_id'])
            totals = stats.sum(axis=0)
            scores, samples = fractions(totals), fractions(draws @ stats)
            assert totals[0] == int(prior[setting, dim]['n'])
            assert np.isclose(scores['micro_f1'], float(prior[setting, dim]['micro_f1']))
            assert np.isclose(scores['em'], float(prior[setting, dim]['exact']))
            for metric, v in scores.items():
                lo, hi = interval(samples[metric])
                values[setting, dim, metric] = float(v)
                boot[setting, dim, metric] = samples[metric]
                summaries.append(dict(setting=setting, dimension=dim, metric=metric,
                    n=int(totals[0]), patients=len(eligible_patients), exact_count=int(totals[1]),
                    tp=int(totals[2]), fp=int(totals[3]), fn=int(totals[4]), value=float(v),
                    ci_low=lo, ci_high=hi))
    pairs = []
    for ratio in RATIOS:
        comparators = BASELINES + (['A1','A2','A3','A4'] if ratio in [32,256] else [])
        for dim in DIMS:
            for metric in ['micro_f1','em','precision','recall']:
                for other in comparators:
                    a, b = f'rg/A0/r{ratio}', f'rg/{other}/r{ratio}'
                    lo, hi = interval(boot[a,dim,metric]-boot[b,dim,metric])
                    pairs.append(dict(ratio=ratio, dimension=dim, metric=metric,
                        comparator=other, ours=values[a,dim,metric], other=values[b,dim,metric],
                        difference=values[a,dim,metric]-values[b,dim,metric], ci_low=lo, ci_high=hi))
    wb = openpyxl.Workbook(); wb.remove(wb.active)
    blue = PatternFill('solid', fgColor='17365D')
    pale = PatternFill('solid', fgColor='E2F0D9')

    def sheet(name, rows):
        ws = wb.create_sheet(name)
        keys = list(rows[0])
        ws.append(keys)
        for r in rows:
            ws.append([json.dumps(r.get(k), ensure_ascii=False) if isinstance(r.get(k), (dict,list,tuple)) else r.get(k) for k in keys])
        ws.freeze_panes='B2'; ws.auto_filter.ref=ws.dimensions
        for c in ws[1]: c.fill=blue; c.font=Font(color='FFFFFF',bold=True)
        for col, k in enumerate(keys, 1):
            ws.column_dimensions[get_column_letter(col)].width=min(65,max(16,len(k)+2))
        for row in ws.iter_rows(min_row=2):
            for c in row:
                if isinstance(c.value,float): c.number_format='0.0000'
        return ws

    notes = [
        ('范围','1108 主实验；GREEN 49 设置；图 V2 57 设置（含 RG A1–A4）；不包含 VQA 的图位置指标。'),
        ('GREEN 主表',f"GREEN_tables 与上次回复一致：49 个设置共有 {gm['common_n']} 条有效报告，140 位患者。"),
        ('GREEN 其他口径','GREEN_budget_common 为每个压缩率内八方法的有效交集；GREEN_all_valid 保留各设置全部有效输出。'),
        ('GREEN 单位','六类错误和 matched_findings 均为每份报告平均计数；GREEN 为逐报告分数均值；location_error_report_rate 为比例。'),
        ('V2 分母','全 560 条参考报告；侧别 327 条/115 患者，脑区 382 条/123 患者，联合 307 条/106 患者。不是 GREEN 的 505 条交集。'),
        ('V2 指标','micro_f1=2TP/(2TP+FP+FN)；precision/recall 为微平均；EM 为报告标签集合完全一致率。百分比格式仅影响显示。'),
        ('V2 状态','自动提取的位置事实一致性，尚非人工核验的影像定位准确率；不是独立病灶实例匹配。未对规则做修改。'),
        ('参考限定','纳入条件仅由参考提取决定。预测缺失、否定及不确定表达不会移除参考已纳入的病例。'),
        ('不确定性','10000 次患者簇配对 percentile bootstrap，seed=20261108；未多重校正，未包含评测器误差。'),
        ('16x','使用 1108 主表冻结 checkpoint，后续新结果需重新导出。'),
        ('完整性','114/114 论文设置已有本地记录。本次检查所有登记路径存在且文件非空；没有重新加载全部 checkpoint 或重算权重 SHA。'),
        ('原文件','只读现有预测/临床评分与图缓存，不更改主 Excel、权重、训练和评测 pipeline。'),
        ('master_sha256',index['master_sha256']), ('graph_protocol',VERSION),
        ('graph_source',str(source)), ('graph_source_sha256',sha(source)),
        ('export_script',str(Path(__file__).resolve())), ('export_script_sha256',sha(__file__)),
    ]
    ws=sheet('说明', [dict(项目=k,说明=v) for k,v in notes])
    ws.column_dimensions['A'].width=27; ws.column_dimensions['B'].width=125
    for row in ws.iter_rows(min_row=2): row[1].alignment=Alignment(wrap_text=True,vertical='top');ws.row_dimensions[row[0].row].height=34

    def matrix(ws, title, lookup, percent=False, higher=True):
        start=ws.max_row+2 if ws.max_row>1 else 1
        ws.cell(start,1,title).font=Font(bold=True,size=12)
        ws.append(['方法','1×']+[f'{r}×' for r in RATIOS])
        header=start+1
        for c in ws[header]:c.fill=blue;c.font=Font(color='FFFFFF',bold=True)
        for method in ['MLP']+METHODS:
            cells=[DISPLAY.get(method,method)]
            cells += [lookup(f'rg/{method}/r{r}') if (method=='MLP')==(r==1) else None for r in [1]+RATIOS]
            ws.append(cells)
            for c in ws[ws.max_row][1:]:c.number_format='0.00%' if percent else '0.0000'
            if method=='A0':
                for c in ws[ws.max_row]:c.fill=pale
        for col in range(3,9):
            cs=[ws.cell(r,col) for r in range(header+2,header+10)]
            best=(max if higher else min)(c.value for c in cs)
            for c in cs:
                if abs(c.value-best)<1e-12:c.font=Font(bold=True)
        ws.column_dimensions['A'].width=25
        for col in range(2,9):ws.column_dimensions[get_column_letter(col)].width=15
        ws.freeze_panes='C3'
        return header

    green=csv_rows(green_dir/'summary.csv')
    for scope,name in [('all_budgets_common','GREEN_tables'),('within_budget_common','GREEN_budget_common'),('all_valid','GREEN_all_valid')]:
        ws=wb.create_sheet(name)
        vs={(r['setting'],r['metric']):float(r['value']) for r in green if r['scope']==scope}
        # MLP has no within-budget population; show no implied matched value there.
        for met,title in GREEN_NAMES.items():
            matrix(ws,title,lambda k,m=met:vs.get((k,m)),percent=met=='location_error_report_rate',higher=met in ['green','matched_findings'])
    for metric,name in [('micro_f1','V2_F1'),('em','V2_EM'),('precision','V2_Precision'),('recall','V2_Recall')]:
        ws=wb.create_sheet(name)
        for dim,title in [('side','侧别'),('site','脑区'),('joint','侧别＋脑区联合')]:
            matrix(ws,f'{title} {metric} ↑',lambda k,d=dim,m=metric:values[k,d,m],percent=True)
    sheet('V2_summary', summaries)
    sheet('V2_ablations',[r for r in summaries if r['setting'].split('/')[1] in ['A0','A1','A2','A3','A4'] and r['setting'].endswith(('/r32','/r256'))])
    sheet('V2_paired_CI',pairs)
    for name,file in [('GREEN_summary','summary.csv'),('GREEN_paired_CI','paired_differences.csv'),('GREEN_best_baseline','best_baseline.csv'),('GREEN_coverage','coverage.csv')]:
        rows=csv_rows(green_dir/file)
        for row in rows:
            for k,v in row.items():
                try:row[k]=float(v) if any(c in v for c in '.eE') else int(v)
                except (ValueError,TypeError):pass
        sheet(name,rows)
    refs=[]
    for rid,row in sorted(canonical.items()):
        labels=label_sets(row['reference_extraction'])
        r=dict(id=rid,patient_id=row['patient_id'],sequence=row['sequence'])
        for dim in DIMS:
            r[dim+'_eligible']=labels['eligible'][dim]
            r[dim+'_labels']=str(sorted(labels[dim]))
        r['flags']=row['reference_extraction']['flags'];refs.append(r)
    sheet('V2_reference_coverage',refs)
    sheet('Artifact_index',[dict(setting=e['key'],source=e['source'],prediction_count=e.get('prediction_count'),**e['paths']) for e in entries.values()])
    sheet('V2_source_hashes',sources)
    charts=wb.create_sheet('Charts')
    for i,(name,title,offset) in enumerate([('V2_F1','Laterality micro-F1',0),('V2_F1','Brain-region micro-F1',1),('V2_F1','Joint-location micro-F1',2),('GREEN_tables','GREEN location errors per report',0)]):
        ws=wb[name];header=2+13*offset
        # Locate actual block headers rather than rely on spacing.
        headers=[r[0].row for r in ws.iter_rows() if r[0].value=='方法']
        header=headers[offset]
        chart=LineChart();chart.title=title;chart.y_axis.title='Score' if name=='V2_F1' else 'Errors/report';chart.x_axis.title='Compression ratio'
        # Include labels in col A, then remove the unused 1x datum for compressed methods.
        for rownum in range(header+2,header+10):
            chart.add_data(Reference(ws,min_col=3,max_col=8,min_row=rownum,max_row=rownum),from_rows=True)
            from openpyxl.chart.series import SeriesLabel
            chart.series[-1].tx=SeriesLabel(strRef=None,v=str(ws.cell(rownum,1).value))
        chart.set_categories(Reference(ws,min_col=3,max_col=8,min_row=header,max_row=header))
        chart.width=27;chart.height=13;charts.add_chart(chart,f'A{1+i*27}')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    wb.save(args.output)
    reopened=openpyxl.load_workbook(args.output,data_only=True,read_only=True)
    assert reopened['V2_summary'].max_row==685
    assert np.isclose(reopened['GREEN_tables']['E11'].value, next(float(r['value']) for r in green if r['scope']=='all_budgets_common' and r['setting']=='rg/A0/r32' and r['metric']=='c_location'))
    assert np.isclose(reopened['V2_F1']['E11'].value,values['rg/A0/r32','side','micro_f1'])
    print(json.dumps(dict(output=str(args.output),sheets=reopened.sheetnames,settings=57,graph_pairs_verified=31920,artifacts_complete=114,bytes=args.output.stat().st_size),ensure_ascii=False,indent=2))
    for dim in DIMS:print(dim,{r:round(values[f'rg/A0/r{r}',dim,'micro_f1']*100,2) for r in RATIOS})


if __name__=='__main__':main()
