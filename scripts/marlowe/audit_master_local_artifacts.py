"""Audit populated Main/VQA/Ablation rows against locally stored experiment artifacts.

Read-only for the master workbook and experiment directories. Writes a separate audit.
"""
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re

import openpyxl
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter
import yaml

ROOT = Path(__file__).resolve().parents[2]
AUDIT = ROOT / 'results/main_1108_local_artifact_audit'
MASTER = ROOT / 'results/Experiments for VLM Projector 1108.xlsx'
REMOTE = '/scratch/m000065-pm07/mingjie/VLM_Projector'
RG_KEYS = ['bleu4', 'rougeL', 'green', 'radgraph_simple', 'radgraph_partial',
           'radgraph_complete', 'rate', 'bertscore_precision', 'bertscore_recall',
           'bertscore_f1', 'answer_ce']


def read(path):
    return json.loads(path.read_text()) if path.is_file() else {}


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def local(value):
    path = Path(value)
    # Historical provenance stays unchanged; resolve known imported project roots
    # to their actual local copies for artifact verification only.
    for source_root in [REMOTE, '/home/mingjie/jyzhang/VLM_Projector']:
        try:
            return ROOT / path.relative_to(source_root)
        except ValueError:
            pass
    return path


def vector(d, task):
    o = d['overall']
    if task == 'rg':
        return [o.get(k) for k in RG_KEYS]
    return ([o.get(k) for k in ['closed_accuracy', 'open_f1', 'open_em', 'open_rouge_l', 'qa_ce']]
            + [d['strata']['question_type'][q]['closed_accuracy'] for q in ['YN', 'MCQ']]
            + [d.get(k) for k in ['patient_macro_accuracy', 'modality_macro_accuracy', 'category_macro_accuracy']]
            + d.get('patient_bootstrap_closed_micro_ci95', [None, None])
            + [d['strata']['sequence'][q]['closed_accuracy'] for q in ['T1W', 'T2W']])


def equal(a, b):
    return len(a) == len(b) and all(isinstance(x, (int, float)) and isinstance(y, (int, float))
                                  and abs(x-y) <= 0.00005000001 for x, y in zip(a, b))


def check_candidate(c, cached):
    p, task = Path(c['path']), c['task']
    d, test = read(p), p.parent
    if task == 'rg':
        if '/evaluation/' not in str(p):
            return dict(c, artifact_available=False, errors=['仅有指标快照；没有可定位的训练目录'])
        run = Path(str(p).split('/evaluation/')[0])
        train = run / 'joint_lora'
        best = train / 'checkpoints/best.pt'
        marker_path, pred = test / 'pipeline.json', test / 'generation/predictions.jsonl'
        config, state = train / 'resolved_config.yaml', train / 'training_state.json'
        expected_hash = d.get('checkpoint_sha256')
    else:
        marker_path = test / 'evaluation_complete.json'
        marker = read(marker_path)
        command = marker.get('command', [])
        if '--checkpoint' not in command:
            return dict(c, artifact_available=False, errors=['缺少评测完成记录，无法定位 checkpoint'])
        best = local(command[command.index('--checkpoint') + 1])
        train = best.parents[1]
        run = train.parent
        pred, config, state = test / 'predictions.jsonl', train / 'config.yaml', train / 'result.json'
        expected_hash = marker.get('checkpoint_sha256')
    paths = dict(run=str(run), metric=str(p), best=str(best), last=str(best.parent / 'last.pt'),
                 train_log=str(train / 'train.jsonl'), validation_log=str(train / 'validation.jsonl'),
                 config=str(config), provenance=str(train / 'provenance.json'),
                 state=str(state), predictions=str(pred), completion_marker=str(marker_path))
    presence = {k: Path(v).is_file() and Path(v).stat().st_size > 0 for k, v in paths.items() if k != 'run'}
    errors = ['缺少/空文件: ' + k for k, yes in presence.items() if not yes]
    cfg = yaml.safe_load(config.read_text()) if presence['config'] else {}
    projector = cfg.get('projector', {})
    method, ratio = projector.get('name'), projector.get('target_compression_ratio')
    actual_hash, hash_mode = None, '未核验'
    if presence['best']:
        cache = cached.get(str(best))
        st = best.stat()
        if cache and st.st_size == cache['size'] and int(st.st_mtime) == cache['mtime_ns'] // 10**9:
            actual_hash, hash_mode = cache['sha256'], '复用本日下载后的SHA256核验；尺寸及mtime未变化'
        else:
            actual_hash, hash_mode = sha(best), '本次读取best.pt计算SHA256'
        if not expected_hash or actual_hash != expected_hash:
            errors.append('best.pt SHA256 与最终评测身份不一致/缺失')
    count, pred_hashes, unique = 0, set(), set()
    if presence['predictions']:
        for line in pred.read_text().splitlines():
            if not line.strip():
                continue
            item = json.loads(line)
            count += 1
            unique.add(item.get('id'))
            pred_hashes.add(item.get('checkpoint_id' if task == 'rg' else 'checkpoint_sha256'))
        if count != (560 if task == 'rg' else 785) or len(unique) != count:
            errors.append('最终预测数量或ID完整性不符')
        if pred_hashes != {expected_hash}:
            errors.append('预测文件checkpoint身份与评测记录不一致')
    marker = read(marker_path)
    if task == 'rg' and marker.get('status') not in ['complete', 'partial']:
        errors.append('RG评测流程未完成')
    console = [str(x) for x in run.glob('*console.log') if x.is_file()]
    console += [str(x) for x in (run / 'logs').glob('*') if x.is_file()]
    return dict(c, artifact_available=not errors, paths=paths, presence=presence, errors=errors,
                configured_method=method, configured_ratio=ratio, configured_projector=projector,
                expected_checkpoint_sha256=expected_hash, actual_checkpoint_sha256=actual_hash,
                hash_verification=hash_mode, prediction_count=count,
                scoring_warnings=d.get('partial_metrics', {}), console_logs=console,
                state_data=read(state))


def main():
    AUDIT.mkdir(exist_ok=True)
    original_hash = sha(MASTER)
    initial = read(AUDIT / 'initial_matches.json')
    rows, candidates = initial['rows'], initial['candidates']
    # Audit full run directories even when historical experiments live under results/.
    actual = {c['path']: c for row in rows for c in row['matches']
              if '/evaluation/' in c['path'] or '/test/normal/' in c['path']}
    cached = {}
    invdir = Path(read(ROOT / '.marlowe/download_current.json')['audit'])
    inventory, verification = read(invdir / 'remote_inventory.json'), read(invdir / 'verification.json')
    if verification and not verification['errors']:
        cached = {str(ROOT / f['path']): f for f in inventory['files'] if f.get('sha256')}
    checked = {}
    with ThreadPoolExecutor(max_workers=4) as pool:
        for result in pool.map(lambda c: check_candidate(c, cached), actual.values()):
            checked[result['path']] = result
            print('CHECKED', len(checked), result['configured_method'] if 'configured_method' in result else '',
                  result.get('configured_ratio'), result['errors'], flush=True)
    missing_history = read(AUDIT / 'missing_history_excel_matches.json')
    for row in rows:
        found = [checked[c['path']] for c in row['matches'] if c['path'] in checked]
        # An audit snapshot is not a second training run; keep only usable full runs here.
        usable = [c for c in found if c['artifact_available']]
        row['artifacts'] = found
        row['available_matches'] = [c['path'] for c in usable]
        row['status'] = '本地齐全' if len(usable) == 1 else '本地齐全；多个匹配运行' if len(usable) > 1 else '本地文件不齐' if found else '未找到对应本地结果'
        row['history_excel_matches'] = [h for h in missing_history if h['target'] == f'{row["sheet"]}!{row["row"]}']
        row['identity_warnings'] = []
        for c in usable:
            if row['method_label'] and 'No Compression' in row['method_label']:
                wanted_ratio = 1
            else:
                wanted_ratio = row['ratio']
            if c['configured_ratio'] != wanted_ratio:
                row['identity_warnings'].append(f'配置倍率{c["configured_ratio"]}与表格倍率{wanted_ratio}不符')
        if row['method_label'] is None:
            row['identity_warnings'].append('主表方法名单元格为空；方法以配置为准，未沿用上一行标签')
    stats = {}
    for sn in ['Main', 'VQA', 'Ablation']:
        rs = [r for r in rows if r['sheet'] == sn]
        stats[sn] = dict(rows=len(rs), statuses=dict(Counter(r['status'] for r in rs)))
    assert sha(MASTER) == original_hash, 'Master workbook changed during audit'
    report = dict(at=datetime.now(timezone.utc).isoformat(), master=str(MASTER), master_sha256=original_hash,
                  match_tolerance=0.00005000001, stats=stats, rows=rows,
                  actual_runs_checked=len(checked), local_sha256_computed=sum(c.get('hash_verification','').startswith('本次') for c in checked.values()))
    report['complete_unique_runs'] = sum(c['artifact_available'] for c in checked.values())
    report['cached_hash_verifications'] = sum(c['artifact_available'] and c.get('hash_verification','').startswith('复用') for c in checked.values())
    (AUDIT / 'audit.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print('STATS', json.dumps(stats, ensure_ascii=False), flush=True)
    export(report)


def export(report):
    wb = openpyxl.Workbook()
    info = wb.active
    info.title = '核查说明'
    info.append(['项目', '内容'])
    for k,v in [
        ('源主表', str(MASTER)), ('源主表SHA256', report['master_sha256']), ('核查时间UTC', report['at']),
        ('检查范围', 'Main、VQA、Ablation 中填写了指标的174行；空白实验不计；重复基线按表格行计数'),
        ('匹配规则', 'RG全部11项/VQA全部14项指标与本地原始JSON按主表4位小数精度匹配，并核对配置倍率；不以同方法同倍率的其他重跑替代'),
        ('齐全标准', 'best.pt、last.pt、train.jsonl、validation.jsonl、配置、provenance、训练状态、最终预测、指标、完成记录均存在；best.pt SHA256和预测checkpoint身份一致'),
        ('哈希检查', f"检查{report['actual_runs_checked']}个候选目录；完整运行{report.get('complete_unique_runs', '见明细')}个。其中{report['local_sha256_computed']}个best.pt本次计算SHA256，{report.get('cached_hash_verifications', '见明细')}个复用同日下载校验且确认尺寸/mtime未变；仅指标快照不算完整运行"),
        ('未找到的含义', '当前项目runs/results及归档中未找到匹配完整指标向量；不代表其他服务器没有，也不以同名方法的另一组结果代替'),
        ('评分缺项', 'GREEN少量失败已单独记录；不等同于训练或预测文件缺失'),
        ('原文件处理', '没有修改原主表、权重或训练文件'),
    ]:info.append([k,v])
    for sn,d in report['stats'].items():info.append([sn,json.dumps(d,ensure_ascii=False)])
    headers=['Sheet','行','方法(主表)','压缩倍率','任务','核查状态','该候选目录文件齐全','可用运行数','配置方法','配置倍率','本地运行目录',
             '指标JSON','best.pt','last.pt','训练日志','验证日志','预测文件','配置','provenance','训练结束记录',
             '评测完成记录','最终预测数','best.pt SHA256','哈希检查方式','缺失/异常','GREEN等评分异常','身份备注','历史Excel记录']
    sheets={sn:wb.create_sheet(sn+'_本地核对') for sn in ['Main','VQA','Ablation']}
    problems=wb.create_sheet('缺失与歧义');details=wb.create_sheet('全部匹配运行')
    for sheet in [*sheets.values(),problems,details]:sheet.append(headers)
    def make_row(r,c):
        p=c.get('paths',{}) if c else {}
        return [r['sheet'],r['row'],r['method_label'],r['ratio'],r['task'].upper(),r['status'],c.get('artifact_available') if c else False,len(r['available_matches']),
                c.get('configured_method') if c else None,c.get('configured_ratio') if c else None,
                *[p.get(k) for k in ['run','metric','best','last','train_log','validation_log','predictions','config','provenance','state','completion_marker']],
                c.get('prediction_count') if c else None,c.get('actual_checkpoint_sha256') if c else None,
                c.get('hash_verification') if c else None,'; '.join(c.get('errors',[])) if c else '找不到匹配该行指标的本地原始结果，无法确定对应权重',
                json.dumps(c.get('scoring_warnings',{}),ensure_ascii=False) if c else None,
                '; '.join(r['identity_warnings']), '\n'.join(f'{Path(h["workbook"]).name} | {h["sheet"]}!{h["row"]}' for h in r['history_excel_matches'])]
    for r in report['rows']:
        preferred=[c for c in r['artifacts'] if c['artifact_available']] or r['artifacts']
        c=preferred[0] if preferred else None
        row=make_row(r,c);sheets[r['sheet']].append(row)
        if r['status']!='本地齐全' or r['identity_warnings']:problems.append(row)
        for c in r['artifacts']:details.append(make_row(r,c))
    for sheet in wb:
        sheet.freeze_panes='A2';sheet.auto_filter.ref=sheet.dimensions
        for c in sheet[1]:
            c.font=Font(bold=True,color='FFFFFF');c.fill=PatternFill('solid',fgColor='17365D')
        for row in sheet:
            for c in row:
                c.alignment=Alignment(vertical='top',wrap_text=True)
                if isinstance(c.value,str) and c.value.startswith(str(ROOT)) and '\n' not in c.value:
                    c.hyperlink=Path(c.value).as_uri();c.font=Font(color='0563C1',underline='single')
        for col in sheet.columns:
            n=max(len(str(c.value or '')) for c in col)
            sheet.column_dimensions[get_column_letter(col[0].column)].width=min(65,max(12,n+2))
    output=ROOT/'results/Experiments for VLM Projector 1108_local_artifact_audit.xlsx'
    wb.save(output)
    print('EXCEL',output,flush=True)


if __name__=='__main__':
    main()
