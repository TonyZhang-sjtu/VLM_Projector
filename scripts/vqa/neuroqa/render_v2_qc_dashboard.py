#!/usr/bin/env python3
"""Build an offline, read-only visual QC dashboard from frozen v2 audit outputs."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import html
from itertools import product
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
import numpy as np
import SimpleITK as sitk
from PIL import Image, ImageDraw
from src.preprocessing.neuroqa_smri import TEMPLATE, compose_pull, warp
from src.config import file_hash

STORE = ROOT/'assets/preprocessing/neuroqa_mni1mm_v2'
OUT = ROOT/'results/vqa/neuroqa/preprocessing_v2_qc_dashboard'


def read(path):
    return json.loads(path.read_text())


def norm(a):
    p = a[np.isfinite(a) & (a > 0)]
    if not p.size:
        return np.zeros(a.shape, dtype=np.uint8)
    lo, hi = np.percentile(p, [1, 99])
    return (np.clip((a-lo)/max(hi-lo, 1e-6), 0, 1)*255).astype(np.uint8)


def contact_mask(mask, corners=None, margin=32):
    inside = np.zeros(mask.shape, dtype=bool)
    inside[margin:-margin, margin:-margin, margin:-margin] = True
    outside = mask & ~inside
    images = []
    # Projections deliberately show outside voxels in red even if superposed
    # on inside tissue in that projection; they are not anatomical slices.
    for axis in range(3):
        a = np.any(mask & inside, axis=axis)
        b = np.any(outside, axis=axis)
        rgb = np.repeat((a.astype(np.uint8)*155)[...,None], 3, axis=2)
        im = Image.fromarray(np.flipud(rgb))
        draw = ImageDraw.Draw(im)
        draw.rectangle((margin, margin, im.width-margin-1, im.height-margin-1), outline=(30,220,255), width=2)
        if corners:
            remaining = [d for d in range(3) if d != axis]
            def project(point):
                zyx = point[::-1]
                return (zyx[remaining[1]], im.height-1-zyx[remaining[0]])
            for i in range(8):
                for j in (1,2,4):
                    if not i & j:
                        draw.line((project(corners[i]),project(corners[i|j])),fill=(255,220,0),width=1)
        # Tiny clipped components are easy to miss; expand their markers for
        # visibility while keeping the quantitative voxel count unchanged.
        from PIL import ImageFilter
        mark = Image.fromarray(np.flipud(b).astype(np.uint8)*255).filter(ImageFilter.MaxFilter(5))
        im.paste(Image.new('RGB',im.size,(255,45,45)),mask=mark)
        images.append(im.resize((330,330)))
    return images


def grid(images, labels, path):
    canvas = Image.new('RGB',(990, len(images)//3*360), (20,24,33))
    draw = ImageDraw.Draw(canvas)
    for i,(im,label) in enumerate(zip(images, labels)):
        x,y=(i%3)*330,(i//3)*360
        canvas.paste(im,(x,y+28)); draw.text((x+7,y+7),label,fill='white')
    canvas.save(path,optimize=True)


def target_slices(a, mask, axis, preferred=None):
    sums = mask.sum(axis=tuple(i for i in range(3) if i != axis))
    if preferred is None:
        center = int(np.argmax(sums)) if sums.any() else a.shape[axis]//2
    else:
        center = preferred
    plane = np.take(a,center,axis=axis)
    seg = np.take(mask,center,axis=axis)
    rgb = np.repeat(norm(plane)[...,None],3,axis=2)
    rgb[seg] = (0.35*rgb[seg] + np.array([255,32,32])*0.65).astype('uint8')
    return Image.fromarray(np.flipud(rgb)).resize((330,330)),center


def make_case(task):
    case, reasons, source = task
    sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(1)
    folder = STORE/'cases'/case
    output = OUT/'images'/case
    output.mkdir(parents=True,exist_ok=True)
    template = sitk.ReadImage(str(STORE/'templates'/TEMPLATE))
    expanded = sitk.ConstantPad(template,[32]*3,[32]*3,0)
    record = read(folder/'complete.json')
    pictures = []
    shutil.copyfile(folder/'qc.png', output/'alignment.png')
    pictures.append('alignment.png')
    if any('outside_canvas' in r or 'brain_bounding_box' in r for r in reasons):
        imgs=[]; labels=[]
        for m in ('T1w','T2w'):
            native=sitk.ReadImage(source['images'][m])
            transform=sitk.ReadTransform(str(folder/f'{m}_mni_pull.tfm'))
            native_mask=sitk.Cast(native != 0,sitk.sitkUInt8)
            mask=sitk.GetArrayFromImage(warp(native_mask,expanded,transform,True))>0
            corners=None
            if f'{m}:brain_bounding_box_may_exceed_qc_reference' in reasons:
                axes=np.where(sitk.GetArrayFromImage(native_mask)>0)
                bounds=[(int(v.min()),int(v.max())) for v in axes[::-1]]
                inverse=transform.GetInverse()
                corners=[expanded.TransformPhysicalPointToContinuousIndex(inverse.TransformPoint(native.TransformIndexToPhysicalPoint(tuple(p))))
                         for p in product(*bounds)]
            imgs.extend(contact_mask(mask,corners));labels += [f'{m} projection axis {i}' for i in range(3)]
        grid(imgs,labels,output/'coverage.png'); pictures.append('coverage.png')
    if any('nmi_drop' in r for r in reasons):
        # Only T2 with a warning gets an explicit same-slice stage comparison.
        m='T2w' if any(r.startswith('T2w:') for r in reasons if 'nmi_drop' in r) else 'T1w'
        native=sitk.ReadImage(source['images'][m],sitk.sitkFloat32)
        rigid=sitk.ReadTransform(str(folder/'rigid.tfm'))
        if m=='T2w': rigid=compose_pull(sitk.ReadTransform(str(folder/'T2w_to_T1w_rigid.tfm')),rigid)
        final=sitk.ReadTransform(str(folder/f'{m}_mni_pull.tfm'))
        arrays=[sitk.GetArrayFromImage(warp(native,template,t)) for t in (rigid,final)]
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        template_arr=sitk.GetArrayFromImage(template)>0
        fig,axes=plt.subplots(2,3,figsize=(11,7))
        for row,a in enumerate(arrays):
            vals=a[a>0]; lo,hi=np.percentile(vals,[1,99])
            for axis in range(3):
                idx=a.shape[axis]//2
                axes[row,axis].imshow(np.take(a,idx,axis=axis),cmap='gray',origin='lower',vmin=lo,vmax=hi)
                axes[row,axis].contour(np.take(template_arr,idx,axis=axis),levels=[.5],colors=['orange'],linewidths=.5)
                axes[row,axis].set_title(f'{m} {"rigid" if row==0 else "affine"}, axis {axis}')
                axes[row,axis].axis('off')
        fig.tight_layout();fig.savefig(output/'nmi_stages.png',dpi=110);plt.close(fig);pictures.append('nmi_stages.png')
    if any('native_segmentation_touches_source_edge' in r for r in reasons):
        seg=sitk.GetArrayFromImage(sitk.ReadImage(str(Path(source['images']['T1w']).parent/f'{case}-seg.nii.gz')))>0
        native=sitk.GetArrayFromImage(sitk.ReadImage(source['images']['T1w']))
        faces=[(i,j) for i in range(3) for j in (0,-1) if np.take(seg,j,axis=i).any()]
        imgs=[];labels=[]
        for axis in range(3):
            edge=next((j for i,j in faces if i==axis),None)
            im,idx=target_slices(native,seg,axis,edge)
            imgs.append(im);labels.append(f'native axis {axis} slice {idx}'+(' EDGE' if edge is not None else ' lesion max'))
        grid(imgs,labels,output/'lesion_edge.png');pictures.append('lesion_edge.png')
    if any('negative_voxels_repaired' in r for r in reasons):
        m=next(r.split(':')[0] for r in reasons if 'negative_voxels_repaired' in r)
        a=sitk.GetArrayFromImage(sitk.ReadImage(source['images'][m])); neg=a<0
        imgs=[];labels=[]
        for axis in range(3):
            im,idx=target_slices(a,neg,axis)
            # Small negative points are expanded only for display.
            from PIL import ImageFilter
            plane=np.take(neg,idx,axis=axis).astype(np.uint8)*255
            markers=Image.fromarray(np.flipud(plane)).resize((330,330),Image.Resampling.NEAREST).filter(ImageFilter.MaxFilter(9))
            im.paste(Image.new('RGB',im.size,(255,0,0)),mask=markers)
            imgs.append(im);labels.append(f'{m} negative markers enlarged, axis {axis} slice {idx}')
        grid(imgs,labels,output/'negative.png');pictures.append('negative.png')
    return {'case_id':case,'pictures':pictures,'record':{'selected_candidate':record['selected_candidate'],
            'transform_kind':record['transform_kind'],'qc':record['qc']}}


def main():
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('--workers',type=int,default=4);args=parser.parse_args()
    queue=read(STORE/'qc_review_queue.json'); audit=read(STORE/'independent_audit.json')
    intensity=read(STORE/'intensity_audit.json');lesion=read(STORE/'lesion_coverage_audit.json')
    source={x['case_id']:x for x in map(json.loads,(ROOT/'assets/data/NeuroQA_derived/official_holdout_patient_guard_v1_mni_v2/source_index.jsonl').read_text().splitlines())}
    audits={x['case_id']:x for x in audit['cases']}; nmi={x['case_id']:x for x in intensity['cases']}; labels={x['case_id']:x for x in lesion['cases']}
    OUT.mkdir(parents=True,exist_ok=True)
    tasks=[(x['case_id'],x['reasons'],source[x['case_id']]) for x in queue['cases']]
    done={};errors=[]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures={pool.submit(make_case,t):t[0] for t in tasks}
        for f in as_completed(futures):
            try: x=f.result();done[x['case_id']]=x
            except Exception as e: errors.append({'case_id':futures[f],'error':repr(e)})
            if (len(done)+len(errors))%10==0: print(f'visualized {len(done)+len(errors)}/{len(tasks)}, errors={len(errors)}',flush=True)
    rows=[]
    for q in queue['cases']:
        case=q['case_id'];a=audits[case]; entry=done.get(case,{})
        volumes=[]
        for v in a['volumes']:
            volumes.append({k:v.get(k) for k in ('modality','dice','outside_canvas_voxels','outside_canvas_fraction','native_bounding_box_may_exceed_expanded_qc_reference','n4_to_warped_ratio_p01_p50_p99')})
        rows.append({'case_id':case,'reasons':q['reasons'],'volumes':volumes,'nmi':nmi[case]['modalities'],
                     'lesion':labels[case],'pictures':entry.get('pictures',[]),'qc':entry.get('record',{}).get('qc',{}),
                     'error':next((e['error'] for e in errors if e['case_id']==case),None)})
    data={'queue_sha256':file_hash(STORE/'qc_review_queue.json'),'audit_sha256':file_hash(STORE/'independent_audit.json'),
          'case_count':len(rows),'errors':errors,'cases':rows}
    (OUT/'dashboard_data.json').write_text(json.dumps(data,ensure_ascii=False,indent=2))
    payload=json.dumps(data,ensure_ascii=False).replace('</','<\\/')
    html_text=PAGE.replace('__DATA__',payload)
    (OUT/'index.html').write_text(html_text,encoding='utf-8')
    print(json.dumps({'output':str(OUT/'index.html'),'cases':len(rows),'rendered':len(done),'errors':errors},ensure_ascii=False),flush=True)
    if errors: raise SystemExit(1)


PAGE = r'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>NeuroQA v2 影像质量复核</title><style>
body{font:15px/1.6 system-ui,sans-serif;background:#f4f6fa;color:#182232;margin:0}main{max-width:1320px;margin:auto;padding:24px}h1{margin-bottom:4px}.note,.bar,article{background:white;border:1px solid #dce2ec;border-radius:10px;padding:16px;margin:14px 0}.note{border-left:5px solid #3a73c8}.bar{position:sticky;top:0;z-index:3;box-shadow:0 3px 10px #0001;display:flex;gap:12px;flex-wrap:wrap}input,select{font-size:15px;padding:8px;border:1px solid #abb7c8;border-radius:5px}input{min-width:240px}.tag{display:inline-block;background:#eaf1fb;border-radius:5px;padding:2px 6px;margin:3px}h2{margin:0}table{border-collapse:collapse;width:100%;font-size:14px}th,td{text-align:left;border-bottom:1px solid #e4e8ef;padding:6px}.pics{display:grid;grid-template-columns:repeat(auto-fit,minmax(350px,1fr));gap:12px}.pics figure{margin:0}.pics img{width:100%;background:#171b23;border-radius:4px}.small{font-size:13px;color:#4e5d71}details{margin:8px 0}code{word-break:break-all}.warn{color:#a33227}a{color:#185ab1}
</style></head><body><main><h1>NeuroQA v2：影像 QC 复核</h1><p>132 个待复核病例 · 静态离线页面 · 数据版本及来源见页面底部</p>
<div class="note"><b>如何阅读：</b>这是工程复核辅助，不是自动诊断或临床认证。红色标出扩展参考网格中采样到、位于固定模型画布之外的前景（为便于发现而放大，数量以表格为准）；青色框为模型画布边界，灰色为画布内前景。黄色线框为映射后的源前景包围盒，仅在包围盒风险告警的模态中显示，线框超出图的部分被裁切；角点可能是空的。三幅图是沿数组轴 0/1/2 的最大投影，不是三个切片；前后深度被压缩，红色可能覆盖同一投影位置上的内部前景。各投影被缩放为正方形，不用于判断形状比例。扩展范围为每侧 32 mm，包围盒告警只说明<i>可能</i>超出该范围，不证明那里有真实前景。原始图像视野外的组织无法由这些图恢复。<br>配准图中橙线为 MNI 模板轮廓、绿线为对应模态前景轮廓；标签仅事后 QC，不参与训练。空白/无红色也不能自动判定无风险。</div>
<div class="bar"><input id="search" placeholder="搜索病例编号或原因"><select id="filter"><option value="">所有复核原因</option><option value="outside_canvas">前景超出模型画布</option><option value="brain_bounding_box">包围盒超出检查范围风险</option><option value="nmi_drop">NMI 下降</option><option value="native_segmentation_touches">源标签触边</option><option value="negative_voxels">负值修复</option></select><span id="count"></span></div><div id="cases"></div><p class="small" id="provenance"></p></main><script>
const data=__DATA__; const cn={outside_canvas:'前景超出固定画布',brain_bounding_box_may_exceed_qc_reference:'包围盒可能超出扩展检查范围','nmi_drop_from_rigid_gt_0.005':'仿射后 NMI 下降 > 0.005',native_segmentation_touches_source_edge:'源分割标签触边',negative_voxels_repaired:'稀少负值已修复'};
const esc=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
function reason(s){let [a,b]=s.includes(':')?s.split(':'):[null,s];return (a?a+'：':'')+(cn[b]||b)}
function num(x,n=4){return x==null?'—':Number(x).toFixed(n)}
function render(){let q=document.querySelector('#search').value.trim().toLowerCase(),f=document.querySelector('#filter').value;let rows=data.cases.filter(x=>(x.case_id+' '+x.reasons.join(' ')+' '+x.reasons.map(reason).join(' ')).toLowerCase().includes(q)&&(!f||x.reasons.some(r=>r.includes(f))));document.querySelector('#count').textContent=`显示 ${rows.length} / ${data.case_count} 例`;
document.querySelector('#cases').innerHTML=rows.map(x=>{let tags=x.reasons.map(r=>`<span class="tag">${esc(reason(r))}</span>`).join('');let v=x.volumes.map(z=>`<tr><td>${esc(z.modality)}</td><td>${num(z.dice)}</td><td>${z.outside_canvas_voxels??'—'}</td><td>${num((z.outside_canvas_fraction??0)*100,5)}%</td><td>${z.native_bounding_box_may_exceed_expanded_qc_reference?'是':'否'}</td></tr>`).join('');let n=Object.entries(x.nmi).map(([m,v])=>`${m}: 初始 ${num(v.initial,5)} → 刚性 ${num(v.rigid,5)} → 仿射 ${num(v.final,5)} (变化 ${num(v.final-v.rigid,5)})`).join('<br>');let info={ 'coverage.png':'脑前景画布投影：红色为画布外，青框为固定画布。', 'alignment.png':'模板 / T1w / T2w 中央三平面；绿色前景，橙色模板。', 'nmi_stages.png':'NMI 告警模态的刚性（上）与仿射（下）中央切片；橙线为模板。', 'lesion_edge.png':'原始 T1 及原始分割标签（红）；EDGE 表示源体积边界面。', 'negative.png':'原图负值体素位置（红点为可见性而放大，非实际大小）。'};let pics=x.pictures.map(p=>`<figure><figcaption class="small">${esc(info[p]||p)}</figcaption><a href="images/${esc(x.case_id)}/${esc(p)}" target="_blank"><img loading="lazy" src="images/${esc(x.case_id)}/${esc(p)}" alt="${esc(p)}"></a></figure>`).join('');let edge=x.lesion.native_edge_face_voxels;let repairs=Object.entries(x.qc).filter(([m,z])=>z.intensity_repair?.negative_voxels).map(([m,z])=>`${m}: ${z.intensity_repair.negative_voxels} 个，原最小值 ${z.intensity_repair.source_min}，替换为 ${z.intensity_repair.replacement_floor}`).join('；');return `<article><h2>${esc(x.case_id)}</h2>${tags}${x.error?`<p class="warn">图像生成失败：${esc(x.error)}</p>`:''}<table><thead><tr><th>模态</th><th>脑前景 Dice</th><th>画布外体素</th><th>画布外占比</th><th>包围盒风险</th></tr></thead><tbody>${v}</tbody></table><details><summary>展开 NMI、病灶标签及负值详情</summary><p>NMI：${n}</p><p>标签源边界各面体素：<code>${esc(JSON.stringify(edge))}</code>；MNI 画布外标签体素：${x.lesion.coverage.outside_canvas_voxels}。</p><p>${esc(repairs||'该病例无负值修复记录')}</p></details><div class="pics">${pics}</div></article>`}).join('')}
document.querySelector('#search').oninput=render;document.querySelector('#filter').onchange=render;document.querySelector('#provenance').textContent='QC 队列 SHA-256: '+data.queue_sha256+'；独立审计 SHA-256: '+data.audit_sha256+'。图像仅供研究 QC，队列和原始数据未被修改。';render();
</script></body></html>'''

if __name__=='__main__':main()
