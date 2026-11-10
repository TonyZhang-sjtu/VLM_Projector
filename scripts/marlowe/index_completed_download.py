"""Build an Excel index joining original result sheets to downloaded artifacts."""
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys

import openpyxl
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter

ROOT = Path(__file__).resolve().parents[2]


def main(audit):
    inventory = json.loads((audit / 'remote_inventory.json').read_text())
    before = json.loads((audit / 'local_before.json').read_text())
    verify_path = audit / 'verification.json'
    verification = json.loads(verify_path.read_text()) if verify_path.exists() else {}
    transferred = json.loads((audit / 'transfer_status.json').read_text()) if (audit / 'transfer_status.json').exists() else {}
    remote = inventory['remote_root']
    local = lambda value: str(ROOT / Path(value).relative_to(remote))
    prior = {f['path']: f['prior_state'] for f in before['files']}
    owner_files = defaultdict(list)
    for f in inventory['files']:
        for owner in f['owners']:
            owner_files[owner].append(f)
    complete = inventory['completed']
    identifiers = {e['run']: f'{e["group"]}:{e["label"]}' for e in complete}
    matches = []
    matched_by_run = defaultdict(list)
    # Exact provenance-path matches; do not infer an experimental identity from a metric value.
    for path in sorted((ROOT / 'results').glob('*.xlsx')):
        if path.name.startswith('Marlowe_completed_download_index_') or path.name.startswith('~$'):
            continue
        book = openpyxl.load_workbook(path, read_only=True, data_only=False)
        for sheet in book.worksheets:
            for row in sheet:
                for cell in row:
                    value = cell.value
                    if not isinstance(value, str):
                        continue
                    if remote not in value and str(ROOT) not in value:
                        continue
                    normalized = value.replace(str(ROOT), remote)
                    for e in complete:
                        if e['run'] not in normalized and e['test'] not in normalized:
                            continue
                        ref = f'{path.name} | {sheet.title}!{cell.coordinate}'
                        matched_by_run[e['run']].append(ref)
                        matches.append([identifiers[e['run']], e['task'].upper(), e['method'], e['ratio'],
                            path.name, sheet.title, cell.coordinate, value, local(e['metric_path']),
                            local(e['best_checkpoint']), e['checkpoint_sha256']])
        book.close()
    wb = openpyxl.Workbook()
    summary = wb.active
    summary.title = '下载说明'
    summary.append(['项目', '内容'])
    new_runs = sum(any(prior[f['path']] != 'same_size' for f in owner_files[e['run']]) for e in complete)
    for row in [
        ['远端清点时间', inventory['captured_at']], ['远端项目根目录', remote],
        ['本地项目根目录', str(ROOT)], ['完整实验总数', len(complete)],
        ['本次补齐实验数', new_runs], ['原已下载实验数', len(complete) - new_runs],
        ['本次范围', 'M01–M17 已完成的正式实验；不下载 smoke test、中断分支或未开始任务'],
        ['是否原样保留', '保留原始 run tag、目录层级、配置和 provenance；未改写历史配置中的绝对路径'],
        ['权重匹配方法', '实验索引的原始指标路径对应已有 Excel 来源路径；best.pt SHA256 对应最终预测及评测完成记录'],
        ['本地已有文件', 'rsync --checksum 跳过相同内容；只对精确文件清单同步；不使用 --delete'],
        ['评分异常说明', 'RG 的 partial 表示评测已结束但少量 GREEN 评分失败，全部560条预测已存在；不是训练未完成'],
        ['传输返回码', transferred.get('returncode', '运行中')],
        ['核验结果', '通过' if verification and not verification['errors'] else '未完成或有错误'],
        ['文件数量', len(inventory['files'])], ['SHA256 核验文件数', verification.get('sha256_files', '待核验')],
        ['最佳权重核验数', verification.get('best_checkpoints', '待核验')],
        ['完整文件清单', str(audit / 'remote_inventory.json')],
        ['传输日志', str(audit / 'transfer.log')], ['核验记录', str(verify_path)],
        ['原表格引用', 'Excel来源对应只记录明确的目录/文件路径匹配；没有来源路径的手填单元格不凭分数推断'],
    ]:
        summary.append(row)
    index = wb.create_sheet('实验与本地权重')
    index.append(['实验ID', '作业', 'JobID', '任务', '方法', '压缩倍率', 'run_tag', '本次下载',
        '训练完成', '评测进程完成', '最终预测数', '指标状态', '评分异常', '本地训练目录',
        '本地最终测试目录', '本地指标JSON', '本地最佳权重', '本地最后权重', '本地预测JSONL',
        '最佳权重SHA256', '原始远端训练目录', '原始远端测试目录', '原始远端指标JSON',
        '原始远端最佳权重', '原始远端最后权重', '原始远端预测JSONL', '本地启动日志目录',
        '已匹配Excel引用数', 'Excel主要引用', '文件数', '逻辑大小GiB'])
    for e in complete:
        files = owner_files[e['run']]
        refs = matched_by_run[e['run']]
        main_refs = [r for r in refs if ' | Main!' in r or ' | VQA!' in r]
        run_tag = Path(e['run']).name if e['task'] == 'vqa' else Path(e['run']).parent.name
        index.append([identifiers[e['run']], e['group'], e['job_id'], e['task'].upper(), e['method'],
            e['ratio'], run_tag, '新增/补齐' if any(prior[f['path']] != 'same_size' for f in files) else '原已下载',
            True, True, e['prediction_count'], e['metric_status'],
            json.dumps(e['scoring_warnings'], ensure_ascii=False), local(e['run']), local(e['test']),
            local(e['metric_path']), local(e['best_checkpoint']), local(e['last_checkpoint']),
            local(e['prediction_path']), e['checkpoint_sha256'], e['run'], e['test'], e['metric_path'],
            e['best_checkpoint'], e['last_checkpoint'], e['prediction_path'], local(e['launch_log_directory']),
            len(refs), '\n'.join(main_refs or refs[:8]), len(files), sum(f['size'] for f in files) / 1024**3])
    sources = wb.create_sheet('Excel来源对应')
    sources.append(['实验ID', '任务', '方法', '倍率', '原Excel', 'Sheet', '单元格', '原单元格内容',
        '本地指标JSON', '本地最佳权重', '最佳权重SHA256'])
    for row in matches:
        sources.append(row)
    excluded = wb.create_sheet('本次不下载')
    excluded.append(['作业', 'JobID', '实验', '任务', '方法', '倍率', '未下载原因', '远端训练目录'])
    for e in inventory['excluded']:
        excluded.append([e['group'], e.get('job_id'), e['label'], e['task'].upper(), e['method'],
            e['ratio'], '; '.join(e['exclusion_reasons']), e['run']])
    for j in inventory['queue']:
        excluded.append([j['name'], j['job_id'], '作业未结束', '', '', '', j['state'] + ': ' + j['reason'], ''])
    for sheet in wb.worksheets:
        sheet.freeze_panes = 'A2'
        sheet.auto_filter.ref = sheet.dimensions
        for cell in sheet[1]:
            cell.fill = PatternFill('solid', fgColor='17365D')
            cell.font = Font(color='FFFFFF', bold=True)
            cell.alignment = Alignment(wrap_text=True, vertical='top')
        for column in sheet.columns:
            letter = get_column_letter(column[0].column)
            width = max(len(str(c.value or '')) for c in list(column)[:100])
            sheet.column_dimensions[letter].width = min(65, max(14, width + 2))
        for row in sheet.iter_rows(min_row=2):
            for cell in row:
                cell.alignment = Alignment(vertical='top', wrap_text=True)
                if isinstance(cell.value, str) and cell.value.startswith(str(ROOT)) and '\n' not in cell.value:
                    cell.hyperlink = Path(cell.value).as_uri()
                    cell.font = Font(color='0563C1', underline='single')
    output = ROOT / 'results' / ('Marlowe_completed_download_index_' + audit.name.removeprefix('marlowe_download_') + '.xlsx')
    wb.save(output)
    info = dict(output=str(output), experiments=len(complete), excel_source_matches=len(matches),
                experiments_with_source_match=sum(bool(matched_by_run[e['run']]) for e in complete),
                new_or_supplemented_runs=new_runs)
    (audit / 'index_summary.json').write_text(json.dumps(info, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(info, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main(Path(sys.argv[1]).resolve())
