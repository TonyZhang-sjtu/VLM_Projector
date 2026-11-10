"""Re-evaluate frozen RG graphs with V3, retaining V2 and all source artifacts.

No model import/inference, GPU work, prediction regeneration or source writes.
Run: python -m experiments.brain_location_audit_v3.run
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import openpyxl
from openpyxl.styles import Font, PatternFill
from openpyxl.utils import get_column_letter

from . import protocol as v3
from experiments.brain_location_audit_v1 import protocol as v1
from experiments.brain_location_audit_v1 import protocol_v2 as v2

ROOT=Path(__file__).resolve().parents[2]
BASELINES=['AvgPool3D','Concat3D','ConvPool3D','LocalPacker3D','Perceiver_NV','Adaptive_Merge3D','Token_Select3D']
RATIOS=[8,16,32,64,128,256]
DIMS=v3.DIMS


def read(p):
    return json.loads(Path(p).read_text())


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def serial(x):
    if isinstance(x,set):return sorted(x,key=str)
    if isinstance(x,np.generic):return x.item()
    raise TypeError(type(x))


def write(p,x):
    Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,default=serial,allow_nan=False)+'\n')


def jsonlines(p):
    with Path(p).open() as f:
        for line in f:
            if line.strip():yield json.loads(line)


def vector(s):
    return [1,s['exact'],s['tp'],s['fp'],s['fn']]


def metrics(a):
    n,ex,tp,fp,fn=np.moveaxis(a,-1,0)
    def div(a,b):
        return np.divide(a,b,out=np.zeros(np.shape(a),dtype=float),where=b>0)
    return dict(precision=div(tp,tp+fp),recall=div(tp,tp+fn),f1=div(2*tp,2*tp+fp+fn),em=div(ex,n))


def main():
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--output',type=Path,default=ROOT/'results/brain_location_graph_v3_1108')
    cli.add_argument('--workbook',type=Path,default=ROOT/'results/AnchorRead3D_GraphV3_1108.xlsx')
    args=cli.parse_args();out=args.output
    if out.exists() or args.workbook.exists():raise FileExistsError('Use a NEW output directory and workbook.')
    out.mkdir(parents=True)
    write(out/'status.json',dict(status='running',protocol=v3.VERSION))
    protected={}
    def protect(p):
        p=Path(p).resolve()
        protected.setdefault(str(p),sha(p))
        return p
    for p in [ROOT/'results/Experiments for VLM Projector 1108.xlsx',
              ROOT/'results/AnchorRead3D_GREEN_GraphV2_1108.xlsx',
              Path(v1.__file__),Path(v2.__file__),Path(v3.__file__)]:protect(p)
    # The live visual workbook can be updated by its independently running GREEN
    # workflow. Read only immutable prediction/graph artifacts for this audit.
    index=read(protect(ROOT/'results/anchorread3d_analysis_1108/artifact_reconciliation_1108.json'))
    settings={e['key']:e for e in index['paper_settings'] if e['key'].startswith('rg/')}
    assert len(settings)==57
    def clinical(key):return Path(settings[key]['paths']['metric']).parent/'scoring/clinical_scores.jsonl'
    canonical={r['id']:r for r in jsonlines(protect(clinical('rg/A0/r32')))}
    assert len(canonical)==560
    patients=sorted({r['patient_id'] for r in canonical.values()});pi={p:i for i,p in enumerate(patients)}
    assert len(patients)==140
    draws=np.random.default_rng(20261108).multinomial(140,np.full(140,1/140),size=10000)
    test=subprocess.run([sys.executable,'-m','pytest','-q',str(Path(__file__).parent/'test_protocol.py'),
                         str(ROOT/'tests/test_brain_location_audit.py')],capture_output=True,text=True,cwd=ROOT)
    (out/'tests.txt').write_text(test.stdout+test.stderr)
    assert test.returncode==0,test.stdout+test.stderr
    # A cache identity includes the raw graph, never a model's method name/score.
    cache={};evidence=open(out/'extractions.jsonl','w')
    def extract(g):
        key=hashlib.sha256(json.dumps(g,sort_keys=True).encode()).hexdigest()
        if key not in cache:
            a,b=v2.extract_graph(g),v3.extract_graph(g)
            cache[key]=(a,b)
            evidence.write(json.dumps(dict(graph_sha256=key,text=g.get('text',''),v2=a,v3=b),
                                      ensure_ascii=False,default=serial)+'\n')
        return key,cache[key]
    refs={};reference_audit=[]
    for rid,r in canonical.items():
        graph=r['clinical']['radgraph']['details']['reference_graph'];key,(a,b)=extract(graph);refs[rid]=(a,b)
        row=dict(id=rid,patient_id=r['patient_id'],sequence=r['sequence'],reference=r['reference'],graph_sha256=key,
                 v2_flags=a['flags'],v3_flags=b['flags'],v2_facts=a['facts'],v3_facts=b['facts'])
        old=v1.label_sets(a);part=v3.label_sets(a);new=v3.label_sets(b)
        for d in DIMS:
            for name,l in [('v2',old),('v2_partial',part),('v3',new)]:
                row[f'{name}_{d}_eligible']=l['eligible'][d];row[f'{name}_{d}_labels']=l[d]
        reference_audit.append(row)
    write(out/'references.json',reference_audit)
    # Reuse the independently frozen 28 real RadGraph probe outputs.
    stress=[]
    for r in jsonlines(protect(ROOT/'results/brain_location_audit_1108_v1/graph_stress_v2/cases.jsonl')):
        a=v3.score_sets(extract(r['reference_graph'])[1][1],extract(r['prediction_graph'])[1][1])
        expected=v3.score_sets(dict(facts=r['expected_reference_facts']),dict(facts=r['expected_prediction_facts']))
        ok=all(a[d]['eligible']==expected[d]['eligible'] and (not expected[d]['eligible'] or
               all(a[d][k]==expected[d][k] for k in ['tp','fp','fn'])) for d in DIMS)
        stress.append(dict(id=r['id'],reference=r['reference'],prediction=r['prediction'],
                           passed=ok,actual=a,expected=expected,clinical_validation=False))
    write(out/'stress_probes.json',stress)
    assert all(r['passed'] for r in stress)
    # Reproduce the original workbook numbers before relying on any V3 result.
    oldbook=openpyxl.load_workbook(ROOT/'results/AnchorRead3D_GREEN_GraphV2_1108.xlsx',read_only=True,data_only=True)
    it=iter(oldbook['V2_summary'].values);header=next(it)
    oldsummary={(r['setting'],r['dimension'],r['metric']):r for r in (dict(zip(header,x)) for x in it)}
    summary=[];bootstrap={};details=open(out/'scores.jsonl','w');results={}

    def evaluate(key,rows,visual=False):
        st={};ids=set()
        for r,g in rows:
            rid=r['id'];assert rid in refs and rid not in ids;ids.add(rid)
            assert r['reference']==canonical[rid]['reference'] and r['patient_id']==canonical[rid]['patient_id']
            graphid,(pa,pb)=extract(g);ra,rb=refs[rid]
            scores={'v2':v1.score_sets(ra,pa),'v2_partial':v3.score_sets(ra,pa),'v3':v3.score_sets(rb,pb)}
            details.write(json.dumps(dict(setting=key,id=rid,patient_id=r['patient_id'],prediction_graph_sha256=graphid,
                reference_sha256=hashlib.sha256(r['reference'].encode()).hexdigest(),
                prediction_sha256=hashlib.sha256(r['prediction'].encode()).hexdigest(),scores=scores),ensure_ascii=False)+'\n')
            for dim in DIMS:
                oldok=scores['v2'][dim]['eligible'];newok=scores['v3'][dim]['eligible']
                for ver in ('v2','v2_partial','v3'):
                    s=scores[ver][dim]
                    if not s['eligible']:continue
                    scopes=['all']
                    if oldok and newok:scopes.append('common_v2_v3')
                    if ver=='v3' and not oldok:scopes.append('new_v3')
                    if ver=='v2' and not newok:scopes.append('lost_v2')
                    for scope in scopes:
                        k=(ver,dim,scope);arr=st.setdefault(k,np.zeros((140,5)))
                        arr[pi[r['patient_id']]]+=vector(s)
        assert ids==set(canonical),(key,len(ids))
        for (ver,dim,scope),a in st.items():
            total=a.sum(0);vals=metrics(total)
            row=dict(setting=key,version=ver,dimension=dim,scope=scope,reports=int(total[0]),
                     patients=int((a[:,0]>0).sum()),tp=int(total[2]),fp=int(total[3]),fn=int(total[4]),
                     **{k:float(v) for k,v in vals.items()})
            if not visual and ver=='v3' and scope=='all':
                boots=metrics(draws@a)
                bootstrap[key,dim]={k:v for k,v in boots.items()}
                for met,x in boots.items():
                    row[met+'_ci_low'],row[met+'_ci_high']=np.percentile(x,[2.5,97.5]).tolist()
            if not visual and ver=='v2' and scope=='all':
                assert row['reports']==oldsummary[key,dim,'micro_f1']['n']
                assert abs(row['f1']-oldsummary[key,dim,'micro_f1']['value'])<1e-12,(key,dim)
                assert abs(row['em']-oldsummary[key,dim,'em']['value'])<1e-12
            summary.append(row);results[key,ver,dim,scope]=row
    for i,key in enumerate(sorted(settings)):
        path=protect(clinical(key))
        def rr():
            for r in jsonlines(path):
                assert r['clinical']['radgraph']['status']=='ok'
                assert r['clinical']['radgraph']['details']['reference_graph']==canonical[r['id']]['clinical']['radgraph']['details']['reference_graph']
                yield r,r['clinical']['radgraph']['details']['prediction_graph']
        evaluate(key,rr());print('MAIN',i+1,len(settings),key,flush=True)
    visualroot=ROOT/'results/anchorread3d_final_eval_1108/visual';visualkeys=[]
    graphmap=read(protect(visualroot/'scoring/original_graphs.json'))
    graphmap.update(read(protect(visualroot/'scoring/new_graphs.json')))
    plan=read(protect(visualroot/'plan.json'))
    for key in plan['settings']:
        for path in sorted((visualroot/key.replace('/','__')).glob('*/predictions.jsonl')):
            protect(path);vk=key+'/'+path.parent.name;visualkeys.append(vk)
            evaluate(vk,((r,graphmap[r['prediction']]) for r in jsonlines(path)),visual=True)
        print('VISUAL',key,flush=True)
    assert len(visualkeys)==48
    evidence.close();details.close();write(out/'summary.json',summary)
    coverage=[]
    for dim in DIMS:
        a={r['id'] for r in reference_audit if r[f'v2_{dim}_eligible']}
        b={r['id'] for r in reference_audit if r[f'v3_{dim}_eligible']}
        for ver in ('v2','v2_partial','v3'):
            selected=[r for r in reference_audit if r[f'{ver}_{dim}_eligible']]
            coverage.append(dict(dimension=dim,version=ver,reports=len(selected),fraction=len(selected)/560,
                                 patients=len({r['patient_id'] for r in selected}),
                                 common_v2_v3=len(a&b),new_v3=len(b-a),lost_v2=len(a-b)))
    paired=[]
    for ratio in RATIOS:
        for dim in DIMS:
            a=f'rg/A0/r{ratio}'
            for other in BASELINES+(['A1','A2','A3','A4'] if ratio in [32,256] else []):
                b=f'rg/{other}/r{ratio}'
                for met in ('f1','em','precision','recall'):
                    diff=bootstrap[a,dim][met]-bootstrap[b,dim][met]
                    lo,hi=np.percentile(diff,[2.5,97.5]).tolist()
                    paired.append(dict(ratio=ratio,dimension=dim,metric=met,comparator=other,
                        ours=results[a,'v3',dim,'all'][met],baseline=results[b,'v3',dim,'all'][met],
                        difference=results[a,'v3',dim,'all'][met]-results[b,'v3',dim,'all'][met],
                        ci_low=lo,ci_high=hi,multiplicity_adjustment='none'))
    write(out/'coverage.json',coverage);write(out/'paired_ci.json',paired)
    # One compact comparison row per setting/dimension: new coverage and common
    # subset are separate so a denominator change is not called model progress.
    comparisons=[]
    for key in sorted(settings):
        for dim in DIMS:
            row=dict(setting=key,dimension=dim)
            for ver,scope,label in [('v2','all','v2'),('v2_partial','all','v2_partial'),('v3','all','v3'),
                                    ('v2','common_v2_v3','v2_common'),('v3','common_v2_v3','v3_common'),
                                    ('v3','new_v3','v3_added')]:
                x=results.get((key,ver,dim,scope),{})
                for met in ('reports','f1','em','precision','recall'):row[label+'_'+met]=x.get(met)
            row['all_f1_change']=row['v3_f1']-row['v2_f1']
            comparisons.append(row)
    wb=openpyxl.Workbook();wb.remove(wb.active)
    def sheet(name,rs):
        ws=wb.create_sheet(name)
        if not rs:return ws
        keys=list(dict.fromkeys(k for r in rs for k in r));ws.append(keys)
        for r in rs:
            ws.append([json.dumps(r.get(k),ensure_ascii=False,default=serial) if isinstance(r.get(k),(dict,list,set,tuple)) else r.get(k) for k in keys])
        ws.freeze_panes='C2';ws.auto_filter.ref=ws.dimensions
        for c in ws[1]:c.font=Font(color='FFFFFF',bold=True);c.fill=PatternFill('solid',fgColor='17365D')
        for i,k in enumerate(keys,1):ws.column_dimensions[get_column_letter(i)].width=min(42,max(15,len(k)+2))
        for row in ws.iter_rows(min_row=2):
            for c in row:
                if isinstance(c.value,float):c.number_format='0.0000'
        return ws
    notes=[
        ('状态','V3 完成；V2、原始预测、官方 RadGraph 分数、原工作簿均保留。'),
        ('范围','57 个 RG 原设置（含 MLP 参考和 A1–A4）＋48 个视觉条件/seed；不含 VQA。'),
        ('版本',v3.VERSION),('定位单位','报告级阳性位置事实集合；不是独立病灶计数或病灶实例匹配。'),
        ('部分标签','参考每个维度至少一个已知标签即纳入；未知侧别仍可评价脑区；缺失预测保留 FN。'),
        ('未知参考','仅在参考明确提供部分位置且该维度未知时，屏蔽兼容的不可判断预测；完整已知位置上的矛盾仍计错。'),
        ('并列范围','共享头词 left frontal and parietal lobes → 左额叶、左顶叶；不跨完整名词短语/逗号传播侧别。'),
        ('新提取','扩充 abnormal signal/focus/abnormality、颅窝/小脑幕等；图锚定局部文本恢复；邻接关系保留 adjacent_to/ 限定。'),
        ('不补全','不跨序列、患者、句子补标签；不从解剖常识猜侧别。未解析的指代及词表外部位仍可能遗漏。'),
        ('覆盖分解','v2_partial 仅在原 V2 提取上应用 V3 部分标签评分，用于分离纳入规则变化与提取变化。'),
        ('可比性','V2_V3 中 common 是同一报告子集，added 是新增子集；标签与评分规则仍可能改变。分数上升不代表模型变强。'),
        ('不确定性','10000 次患者簇配对 percentile bootstrap；140 患者；seed=20261108；未多重校正，不包含提取器误差。'),
        ('验证','语义单元测试＋28 组冻结 RadGraph 图探针；这些不是临床人工金标准准确率。'),
        ('事后规则改进','本次规则根据参考报告覆盖问题及合成语义反例设计；固定规则后统一比较所有方法；不是预注册测试协议。'),
        ('GREEN','没有重算/修改 GREEN；正在运行的 GREEN 工作流不受影响。'),
        ('出处',str(out)),('protocol_sha256',sha(Path(v3.__file__))),
    ]
    sheet('说明',[dict(项目=k,说明=v) for k,v in notes]);sheet('Coverage',coverage)
    sheet('V2_V3',comparisons);sheet('V3_summary',[r for r in summary if r['version']=='v3' and '/' not in r['setting'].split('/r')[-1]])
    sheet('All_summary',summary)
    for met,title in [('f1','V3_F1'),('em','V3_EM'),('precision','V3_Precision'),('recall','V3_Recall')]:
        matrix=[]
        for dim in DIMS:
            for m in BASELINES+['A0']:
                matrix.append(dict(dimension=dim,method='AnchorRead3D' if m=='A0' else m,
                              **{f'{r}x':results[f'rg/{m}/r{r}','v3',dim,'all'][met] for r in RATIOS}))
        sheet(title,matrix)
    sheet('V3_ablations',[r for r in comparisons if r['setting'].split('/')[1] in ['A0','A1','A2','A3','A4'] and r['setting'].endswith(('/r32','/r256'))])
    sheet('V3_paired_CI',paired)
    visual=[]
    for ratio in (32,256):
        table=[]
        for m in ['AvgPool3D','Perceiver_NV','Token_Select3D','A0']:
            prefix=f'rg/{m}/r{ratio}'
            for condition in ('normal','shuffle','zero','text'):
                keys=[k for k in visualkeys if k.startswith(prefix+'/') and
                      (k.split('/')[-1].startswith('shuffle_') if condition=='shuffle' else k.endswith('/'+condition))]
                row=dict(method=m,ratio=ratio,condition=condition,seed_count=len(keys))
                for dim in DIMS:
                    for ver in ('v2','v3'):
                        vals=[results[k,ver,dim,'all']['f1'] for k in keys]
                        row[f'{ver}_{dim}_f1']=float(np.mean(vals))
                        row[f'{ver}_{dim}_seed_sd']=float(np.std(vals,ddof=1)) if len(vals)>1 else None
                table.append(row)
        sheet(f'Visual_{ratio}x',table);visual+=table
    sheet('Visual_by_seed',[r for r in summary if r['setting'] in visualkeys and r['scope']=='all'])
    sheet('Reference_audit',reference_audit)
    sheet('Reference_excluded',[r for r in reference_audit if not r['v3_joint_eligible']])
    sheet('Stress_probes',stress)
    # Verify immutable source hashes AFTER all reading/computation; the live
    # workflow's status/monitor/cache files are intentionally not locked/changed.
    checks=[dict(path=p,before=h,after=sha(p)) for p,h in protected.items()]
    assert all(r['before']==r['after'] for r in checks)
    sheet('Provenance',checks);write(out/'source_integrity.json',checks)
    temporary=args.workbook.with_suffix('.tmp.xlsx');wb.save(temporary);temporary.replace(args.workbook)
    reloaded=openpyxl.load_workbook(args.workbook,read_only=True,data_only=True)
    assert len(list(reloaded['V2_V3'].values))==172
    assert all(reloaded[f'Visual_{r}x'].max_row==17 for r in (32,256))
    assert reloaded['Reference_audit'].max_row==561
    status=dict(status='complete',protocol=v3.VERSION,main_settings=57,visual_settings=48,
                report_pairs=(57+48)*560,unique_graphs=len(cache),stress_passed=sum(r['passed'] for r in stress),
                stress_total=len(stress),source_files_unchanged=len(checks),v2_all_summary_reproduced=True,
                workbook=str(args.workbook),workbook_sha256=sha(args.workbook),coverage=coverage,
                code_sha256={p.name:sha(p) for p in Path(__file__).parent.glob('*.py')})
    write(out/'status.json',status)
    print('COMPLETE',json.dumps(status,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
