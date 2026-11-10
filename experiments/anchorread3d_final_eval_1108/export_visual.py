"""Export all eleven RG scores plus Graph V2 spatial agreement and coverage."""
from collections import Counter,defaultdict
import math
from pathlib import Path
import argparse

import numpy as np
from openpyxl import Workbook,load_workbook
from openpyxl.styles import Font,PatternFill
from openpyxl.chart import BarChart,Reference

from .common import ROOT,OUT,VISUAL_METHODS,entries,read,rows,write
from .score_visual import pair_key
from src.evaluation.clinical import file_digest
from src.evaluation.significance import METRICS,lexical_vectors,statistic
from .significance import ce_catalog

VIS=OUT/'visual'
LABELS=['BLEU-4 ↑','ROUGE-L F1 ↑','GREEN ↑','RadGraph-XL Simple F1 ↑',
        'RadGraph-XL Partial F1 ↑','RadGraph-XL Complete F1 ↑','RaTEScore ↑',
        'BERTScore Precision ↑','BERTScore Recall ↑','BERTScore F1 ↑','Answer CE ↓']


def graph_summary(rr,graph):
    result={};coverage={}
    for dim in ('side','site','joint'):
        selected=[(r,graph[pair_key(r)]['graph_v2'][dim]) for r in rr if graph[pair_key(r)]['graph_v2'][dim]['eligible']]
        n=len(selected);tp=sum(v['tp'] for _,v in selected);fp=sum(v['fp'] for _,v in selected);fn=sum(v['fn'] for _,v in selected)
        for name,value in dict(precision=tp/(tp+fp) if tp+fp else 0,recall=tp/(tp+fn) if tp+fn else 0,
                               f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0,
                               em=sum(v['exact'] for _,v in selected)/n if n else None).items():
            result[f'{dim}_{name}']=value
        coverage[dim]=dict(reports=n,patients=len({r['patient_id'] for r,_ in selected}),tp=tp,fp=fp,fn=fn)
    return result,coverage


def main():
    p=argparse.ArgumentParser();p.add_argument('--allow-pending-green',action='store_true');args=p.parse_args()
    _,es=entries();plan=read(VIS/'plan.json');out=VIS/'scoring'
    assert read(out/'clinical_verified.json')['status']=='verified'
    if not args.allow_pending_green:assert read(out/'green_verified.json')['status']=='verified'
    graph={r['pair_key']:r for r in read(out/'graph_scores.json')}
    cached=read(out/'cached_original_pairs.json');clinical={k:dict(v) for k,v in cached.items()}
    for kind in ('clinical','green'):
        for i in range(4):
            path=out/f'{kind}_{i}/clinical_scores.jsonl'
            if not path.exists():
                assert kind=='green' and args.allow_pending_green;continue
            for r in rows(path):clinical.setdefault(r['id'],{}).update(r['clinical'])
    catalog=ce_catalog();summary=[];locations=[];coverage=[];green_parts=[];provenance=[]
    for setting in plan['settings']:
        task,method,ratio=setting.split('/');ratio=int(ratio[1:]);e=es[setting]
        folder=VIS/setting.replace('/','__');source_metric=read(e['paths']['metric'])
        normal_by_id={r['id']:r for r in rows(folder/'normal/predictions.jsonl')}
        source_clinical={r['id']:r['clinical'] for r in rows(Path(e['paths']['metric']).parent/'scoring/clinical_scores.jsonl')}
        for path in sorted(folder.glob('*/predictions.jsonl')):
            condition=path.parent.name;rr=rows(path);lex=lexical_vectors(rr)
            values={k:statistic(k,np.asarray(list(v.values())).sum(0)) for k,v in lex.items()}
            base=dict(setting=setting,method=method,ratio=ratio,condition=condition,n=len(rr),patients=len({r['patient_id'] for r in rr}))
            for key in METRICS[2:-1]:
                group=('radgraph' if key.startswith('radgraph') else 'bertscore' if key.startswith('bertscore') else key)
                valid=[];valid_patients=set();status=Counter()
                for r in rr:
                    if key=='green' and args.allow_pending_green and condition!='normal':
                        status['pending']+=1;continue
                    if group=='radgraph':value=graph[pair_key(r)]['radgraph'][key];status['ok']+=1
                    else:
                        item=(source_clinical[r['id']] if condition=='normal' else clinical.get(pair_key(r),{})).get(group)
                        state='pending' if item is None else item['status'];status[state]+=1
                        value=item['values'][key] if state=='ok' else None
                    if value is not None:valid.append(value);valid_patients.add(r['patient_id'])
                values[key]=float(np.mean(valid)) if valid else None
                coverage.append(dict(**base,metric=key,valid=len(valid),eligible_patients=len(valid_patients),statuses=dict(status)))
            if condition=='normal':
                ce=rr
                assert all(r.get('ce_measurement')=='same_process_as_visual_controls' for r in rr)
                assert {r['id'] for r in ce}=={r['id'] for r in rr}
                # Lexical/clinical recomputation must agree with frozen original.
                for key in METRICS[:-1]:
                    assert math.isclose(values[key],source_metric['overall'][key],rel_tol=1e-6,abs_tol=1e-6),(setting,key,values[key],source_metric['overall'][key])
            else:ce=rr
            values['answer_ce']=math.fsum(r['loss_sum'] for r in ce)/sum(r['answer_tokens'] for r in ce)
            spatial,cov=graph_summary(rr,graph)
            for dim,c in cov.items():coverage.append(dict(**base,metric=f'graph_{dim}',valid=c['reports'],eligible_patients=c['patients'],statuses=c))
            result=dict(**base,**values,**spatial,visual_tokens=rr[0]['visual_token_count'],
                        empty_rate=sum(not r['prediction'].strip() for r in rr)/len(rr),
                        length_limit_rate=sum(r['hit_generation_limit'] for r in rr)/len(rr),
                        repeated_4gram_rate=sum(repetition(r['prediction']) for r in rr)/len(rr),
                        report_change_rate=sum(' '.join(r['prediction'].split())!=' '.join(normal_by_id[r['id']]['prediction'].split()) for r in rr)/len(rr))
            for dim in ('side','site','joint'):
                result[f'{dim}_prediction_change_rate']=sum(graph[pair_key(r)]['predicted_locations'][dim]!=graph[pair_key(normal_by_id[r['id']])]['predicted_locations'][dim] for r in rr)/len(rr)
            summary.append(result);locations.append(dict(**base,**spatial))
            gg=[]
            for r in rr:
                if args.allow_pending_green and condition!='normal':continue
                item=(source_clinical[r['id']] if condition=='normal' else clinical.get(pair_key(r),{})).get('green')
                if item and item['status']=='ok':gg.append(item)
            parts={f'error_{letter}':float(np.mean([g['details']['significant_errors'][i] for g in gg])) if gg else None for i,letter in enumerate('abcdef')}
            parts['matched_findings']=float(np.mean([g['details']['matched_findings'] for g in gg])) if gg else None
            green_parts.append(dict(**base,valid=len(gg),**parts))
        provenance.append(dict(setting=setting,checkpoint=e['paths']['best'],checkpoint_sha256=source_metric['checkpoint_sha256'],
                  normal_predictions=e['paths']['predictions'],control_directory=str(folder),
                  swap_checks=read(folder/'swap_equivalence_checks.json')))
    assert len(summary)==48
    for r in summary:
        if r['condition']!='normal':continue
        for other in [x for x in coverage if x['setting']==r['setting'] and x['metric'].startswith('graph_')]:
            counts=[x['valid'] for x in coverage if x['setting']==r['setting'] and x['metric']==other['metric']]
            assert len(set(counts))==1,'Reference eligibility must not depend on visual condition'
    mainrows=[]
    diagnostics=['empty_rate','length_limit_rate','repeated_4gram_rate','report_change_rate',*[f'{d}_prediction_change_rate' for d in ('side','site','joint')]]
    score_fields=list(METRICS)+[f'{d}_{m}' for d in ('side','site','joint') for m in ('precision','recall','f1','em')]+diagnostics
    for setting in plan['settings']:
        for condition in ('normal','shuffle','zero','text'):
            selected=[r for r in summary if r['setting']==setting and (r['condition'].startswith('shuffle_') if condition=='shuffle' else r['condition']==condition)]
            assert len(selected)==(3 if condition=='shuffle' else 1)
            item={k:v for k,v in selected[0].items() if k not in score_fields};item['condition']=condition
            for key in score_fields:
                vals=[r[key] for r in selected]
                item[key]=float(np.mean(vals)) if all(v is not None for v in vals) else None
                if condition=='shuffle':item[key+'_seed_sd']=float(np.std(vals,ddof=1)) if all(v is not None for v in vals) else None
            mainrows.append(item)
    wb=Workbook();ws=wb.active;ws.title='说明'
    for row in [
        ['项目','说明'],['状态','GREEN 待完成' if args.allow_pending_green else '全部阶段已验收'],
        ['比较','RG only; A0, AvgPool3D, Perceiver_NV, Token_Select3D; 32× and 256×; no MLP'],
        ['Shuffle','三个预先固定患者级 derangement；同序列、同cohort、同输入patch数；同患者四序列共享donor；无self-donor。'],
        ['Shuffle 汇报','主表取三次固定shuffle的指标均值；逐次结果与seed SD另表；不把三次当成独立患者。'],
        ['复用范围','Prompt只依赖序列、冻结checkpoint、greedy独立解码，因此复用donor原始生成文本。新reference配对的评分和CE均重新计算。'],
        ['Zero-prefix','最终projector输出全零；保留真实M和attention mask。256× AvgPool M=36，其余M=35。'],
        ['Text-only','彻底移除视觉位置；保留原system/user提示。该提示仍要求描述影像，因此作为辅助压力对照。'],
        ['Answer CE','四种条件在相同checkpoint/运行配置下teacher-forced计算；总loss / 总answer tokens。Normal报告复用，但CE与shuffle在同一模型进程测量；历史差异另存。'],
        ['Graph V2','自动抽取的报告侧别/脑区/联合位置集合一致性；不是影像位置金标准，也不是保留病灶实例数量的实例匹配。'],
        ['分母','Graph只由reference定义eligible集合；GREEN等失败不记0，单独列有效数量；临床分项均为有效报告均值。'],
        ['重复退化','repeated_4gram_rate: 至少20词，重复4-gram占比>0.5；仅诊断规则。'],
        ['输出变化率','相对同患者Normal输出的文本/位置标签变化比例；变化本身不代表更正确，需结合reference匹配指标解读。'],
        ['空间分项','side=侧别；site=脑区；joint=侧别-脑区配对；F1为micro-F1，EM为eligible报告exact match均值。'],
        ['原表','results/Experiments for VLM Projector 1108.xlsx，原表和模型文件未改写。']]:ws.append(row)
    for ratio in (32,256):
        ws=wb.create_sheet(f'RG_{ratio}x');ws.append(['Method','Condition','Actual M',*LABELS,'Side F1 ↑','Region F1 ↑','Joint F1 ↑'])
        for method in VISUAL_METHODS:
            for r in mainrows:
                if r['ratio']==ratio and r['method']==method:ws.append([method,r['condition'],r['visual_tokens'],*[r[k] for k in METRICS],r['side_f1'],r['site_f1'],r['joint_f1']])
    def sheet(name,data,fields):
        ws=wb.create_sheet(name);ws.append(fields)
        import json
        for r in data:ws.append([json.dumps(r.get(k),ensure_ascii=False) if isinstance(r.get(k),(dict,list)) else r.get(k) for k in fields])
    sheet('By_seed',summary,['method','ratio','condition','visual_tokens',*METRICS,'side_f1','site_f1','joint_f1',*diagnostics])
    sheet('Location_details',locations,['method','ratio','condition',*[f'{d}_{m}' for d in ('side','site','joint') for m in ('precision','recall','f1','em')]])
    sheet('Shuffle_seed_SD',[r for r in mainrows if r['condition']=='shuffle'],['method','ratio',*[k+'_seed_sd' for k in list(METRICS)+['side_f1','site_f1','joint_f1']]])
    sheet('GREEN_parts',green_parts,['method','ratio','condition','valid',*[f'error_{x}' for x in 'abcdef'],'matched_findings'])
    sheet('Coverage',coverage,['method','ratio','condition','metric','valid','n','eligible_patients','patients','statuses'])
    deltas=[]
    for r in mainrows:
        if r['condition']=='normal':continue
        normal=next(n for n in mainrows if n['setting']==r['setting'] and n['condition']=='normal')
        for key in list(METRICS)+['side_f1','site_f1','joint_f1']:
            deltas.append(dict(method=r['method'],ratio=r['ratio'],condition=r['condition'],metric=key,
                normal=normal[key],control=r[key],normal_minus_control=normal[key]-r[key] if r[key] is not None and normal[key] is not None else None))
    sheet('Normal_minus_control',deltas,['method','ratio','condition','metric','normal','control','normal_minus_control'])
    sheet('Provenance',provenance,['setting','checkpoint','checkpoint_sha256','normal_predictions','control_directory','swap_checks'])
    for ws in wb:
        ws.freeze_panes='C2';ws.auto_filter.ref=ws.dimensions
        for c in ws[1]:c.fill=PatternFill('solid',fgColor='244B68');c.font=Font(color='FFFFFF',bold=True)
        for col in ws.columns:ws.column_dimensions[col[0].column_letter].width=24
        for row in ws.iter_rows(min_row=2):
            for c in row:
                if isinstance(c.value,float):c.number_format='0.0000'
        if ws.title=='说明':ws.column_dimensions['B'].width=130
    destination=ROOT/'results/AnchorRead3D_RG_VisualDependence_1108.xlsx'
    tmp=destination.with_name(destination.stem+'.tmp.xlsx');wb.save(tmp);tmp.replace(destination)
    check=load_workbook(destination,data_only=True)
    assert check['RG_32x'].max_row==17 and check['RG_256x'].max_row==17
    assert check['By_seed'].max_row==49
    write(VIS/'summary.json',dict(status='pending_green' if args.allow_pending_green else 'complete',rows=mainrows,by_seed=summary))
    write(VIS/'export.json',dict(status='pending_green' if args.allow_pending_green else 'complete',path=str(destination),sha256=file_digest(destination),sheets=check.sheetnames))
    print('EXPORTED',destination,flush=True)


def repetition(text):
    tokens=text.lower().split()
    if len(tokens)<20:return False
    grams=[tuple(tokens[i:i+4]) for i in range(len(tokens)-3)]
    return 1-len(set(grams))/len(grams)>.5


if __name__=='__main__':main()
