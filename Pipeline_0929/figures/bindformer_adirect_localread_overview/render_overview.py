#!/usr/bin/env python3
"""Draw a PROPOSED ADirect + one local residual read, using the A4/ADirect style.

Design illustration only: no registered model, training, or claimed results.
The unchanged baseline budgets/parameters are read from current ADirect code.
"""
from pathlib import Path
import argparse, hashlib, importlib.util, json

HERE=Path(__file__).resolve().parent
BASE=HERE.parent/'bindformer_adirect_overview/render_overview.py'
spec=importlib.util.spec_from_file_location('adirect_figure',BASE)
base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
style=base.style
plt,C=style.plt,style.C
box,line,arrow,tokens,label=style.box,style.line,style.arrow,style.tokens,style.label
text=base.text
W,H=1752,632
CHILD_COLORS=base.CHILD_COLORS


def audit_design():
    baseline=base.code_audit()
    d=768
    extra=4*d*d+4*d+1  # bias-free Q/K/V/O, two affine LNs, one shared scalar.
    proposal={'display_name':'ADirect + LocalRead',
       'status':'design_only_not_registered_not_trained',
       'baseline':'bindformer_adirect', 'local_width':d,'heads':8,'head_dim':96,
       'local_layers':1,'local_ffn':False,'qkv_output_linear_bias':False,
       'query_source':'raw child mean mu_r', 'key_value_source':'raw encoder patches in the same child',
       'query_normalization':'one affine LayerNorm', 'key_value_normalization':'one shared affine LayerNorm',
       'support':'same disjoint A4 children; packed-padding mask only',
       'residual':'u_r = mu_r + alpha * delta_r',
       'alpha':{'type':'one unconstrained learned scalar shared by all children','initial_value':0.1,'selection_status':'proposed starting value, not a validated optimum'},
       'new_local_position_encoding':False,'new_local_sequence_embedding':False,
       'extra_parameter_formula':'4 * 768**2 + 4 * 768 + 1',
       'extra_parameters':extra,'baseline_code_audit':baseline,'ratios':{},
       'source_sha256':{str(Path(__file__).name):hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}}
    for ratio,row in baseline['ratios'].items():
        proposal['ratios'][ratio]={'budget':row['budget'],
            'baseline_parameters':row['trainable_projector_parameters'],
            'proposed_parameters':row['trainable_projector_parameters']+extra}
    return proposal


def create(audit):
    box(.5,.5,W-1,H-1,radius=15,lw=1.15)
    text(22,27,'(b) ADirect + LocalRead',27,weight='bold',ha='left')
    text(W-22,27,'Proposed ablation · one local read + raw mean residual',21,
         color=C['muted'],ha='right')
    # The two branches split before the global adapter, exactly as in ADirect.
    box(32,258,234,82,fill=C['gray_bg'])
    text(149,283,r'$X$: all patch features',21,weight='bold',max_width=214)
    text(149,315,r'$N\times768$  ·  $N=8775$',19,color=C['muted'])
    arrow([(149,258),(149,204)],C['blue'])
    box(32,90,234,114,fill=C['blue_bg'],edge='#8FB2D2')
    text(149,112,'1  Input adapter',23,weight='bold')
    text(149,143,'LN + Linear (768 → 768)',18.5,max_width=214)
    text(149,170,'Grid Fourier PE + sequence',17.5,max_width=214)
    text(149,191,'Global branch only',16.5,color=C['blue'])
    box(32,407,234,82,fill=C['gray_bg'])
    text(149,432,r'Grid / geometry $p_i$',20)
    text(149,464,r'MRI sequence identity $m$',19,max_width=214)
    arrow([(32,448),(16,448),(16,180),(32,180)],C['blue'],lw=1.1)
    line([(266,147),(294,147),(294,64),(700,64),(700,147)],C['blue'],lw=1.5)
    arrow([(700,147),(720,147)],C['blue'],lw=1.5)
    label(493,64,r'Full visual memory $H$: $N\times768$',19,C['blue'])
    box(350,96,280,69,fill=C['blue_bg'],edge='#8FB2D2')
    text(490,118,r'Learned global queries $Q_g$',20,weight='bold',max_width=260)
    text(490,146,r'$G\times768$  ·  unchanged design',18,color=C['muted'],max_width=260)
    arrow([(630,119),(720,119)],C['blue'])
    box(720,90,394,95,fill=C['blue_bg'],edge='#8FB2D2')
    text(917,114,'2  Global cross-attention',23,color=C['blue'],weight='bold')
    text(917,146,'1 block · 8 heads · residual + FFN',20)
    text(917,170,r'$Q_g$ reads keys / values from all $H$',17.5,color=C['muted'])
    arrow([(1114,127),(1150,127)],C['blue'])
    tokens(1158,109,n=6,width=23,height=36,gap=8,colors=[C['blue']])
    text(1248,171,r'Global tokens: $G\times768$',19,color=C['blue'])
    arrow([(1341,127),(1565,127),(1565,248)],C['blue'],lw=1.5)

    text(330,207,'3  Local evidence + residual read',22,color=C['teal'],weight='bold',ha='left')
    arrow([(266,293),(330,293)],C['teal'],lw=1.6)
    box(330,236,240,114,fill=C['teal_bg'],edge='#85B6AE')
    text(450,259,'Gather child patches',20,weight='bold',color=C['teal'],max_width=220)
    text(450,295,r'$X_r:\ n_r\times768$',24)
    text(450,329,'Raw X · fixed spatial support',17,color=C['muted'],max_width=220)
    arrow([(570,293),(610,293)],C['teal'])
    box(610,236,200,114,fill=C['teal_bg'],edge='#85B6AE')
    text(710,259,'Masked mean',21,weight='bold',color=C['teal'])
    text(710,297,r'$\mu_r=\mathrm{mean}(X_r)$',23,max_width=182)
    text(710,329,r'One query per child',17,color=C['muted'],max_width=182)
    # Mean is BOTH the query and the independent untransformed residual.
    arrow([(810,293),(850,293)],C['teal'])
    line([(830,293),(830,215),(1255,215)],C['teal'],lw=1.5)
    arrow([(1255,215),(1255,236)],C['teal'],lw=1.5)
    label(1040,215,r'Raw mean residual $\mu_r$',18,C['teal'])
    box(850,236,260,139,fill=C['purple_bg'],edge=C['purple'])
    text(980,259,'One local attention read',21,weight='bold',color=C['purple'],max_width=240)
    text(980,294,r'$Q\leftarrow\mu_r;\quad K,V\leftarrow X_r$',22,max_width=240)
    text(980,327,'8 heads · hard local mask',18,color=C['muted'],max_width=240)
    text(980,356,'Pre-norm Q / KV · no local FFN',17,color=C['muted'],max_width=240)
    arrow([(500,350),(500,400),(980,400),(980,375)],C['purple'],lw=1.5)
    label(733,400,r'Raw child memory $X_r$ (K/V)',18,C['purple'])
    arrow([(1110,293),(1150,293)],C['purple'])
    label(1130,270,r'$\delta_r$',21,C['purple'])
    box(1150,236,210,139,fill=C['teal_bg'],edge=C['teal'])
    text(1255,260,'Residual update',21,weight='bold',color=C['teal'],max_width=190)
    text(1255,293,r'$u_r=\mu_r+\alpha\delta_r$',24,max_width=190)
    tokens(1210,320,n=4,width=18,height=24,gap=6,colors=CHILD_COLORS)
    text(1255,359,r'Local: $4K\times768$',18.5,color=C['teal'])
    arrow([(1360,293),(1380,293),(1380,282),(1410,282)],C['teal'],lw=1.5)

    box(330,443,385,86,fill=C['gold_bg'],edge='#D6BD89')
    text(522,464,'Same deterministic A4 partition',21,weight='bold',color='#8E6727',max_width=359)
    text(522,491,r'$K$ parents → 4 disjoint children each',18,max_width=359)
    text(522,515,'Geometry + packed-padding validity mask',16.5,color=C['muted'],max_width=359)
    arrow([(266,448),(312,448),(312,477),(330,477)],C['gold'])
    arrow([(350,443),(350,365),(400,365),(400,350)],C['gold'],lw=1.3)
    box(735,443,625,86,fill=C['gray_bg'],edge=C['gray_bg'],lw=0)
    text(1047,464,'Shared local reader across all children; learned α starts at 0.1',20,weight='bold',max_width=599)
    text(1047,491,'No added local PE / sequence embedding; final LN + MLP is unchanged.',17,max_width=599)
    text(1047,515,'Design only · not trained · same report CE and visual token budget',17,color=C['purple'],max_width=599)

    box(1410,248,310,68,fill=C['gray_bg'])
    text(1565,270,'Concatenate [Global ; Local]',20,weight='bold',max_width=284)
    text(1565,298,r'$(G+4K)\times768$',23)
    arrow([(1565,316),(1565,359)],C['blue'])
    box(1410,359,310,81,fill=C['blue_bg'],edge='#8FB2D2')
    text(1565,384,'4  Final LN + output MLP',22,weight='bold',max_width=284)
    text(1565,414,'768 → 4096 → 4096 · GELU',19)
    arrow([(1565,440),(1565,476)],C['blue'])
    box(1410,476,310,60,fill='#FCFDFE',edge='#B6C7D7')
    tokens(1430,491,n=3,width=13,height=30,gap=5,colors=[C['blue']])
    line([(1490,485),(1490,527)],C['border'],.9)
    tokens(1503,491,n=4,width=18,height=30,gap=6,colors=CHILD_COLORS)
    text(1611,506,'…',24,color=C['muted']);text(1669,506,r'$Z$',29,color=C['blue'],weight='bold')
    text(1565,560,r'$M=G+4K$;  $Z:M\times4096$',23,weight='bold')
    text(1565,590,'Visual prefix → frozen Llama',19,color=C['muted'])
    box(32,554,1328,60,fill=C['gray_bg'],edge=C['gray_bg'],lw=0)
    for ratio,xx in [(128,51),(256,716)]:
        row=audit['ratios'][str(ratio)];b=row['budget']
        text(xx,574,f'{ratio}×: {b["global_tokens"]} global + {b["local_tokens"]} local = {b["expected_output_tokens"]} tokens',21,weight='bold',ha='left')
        text(xx,600,f'Proposed {row["proposed_parameters"]/1e6:.3f}M parameters · +2.362M vs ADirect',18.5,color=C['muted'],ha='left')


def create_detail():
    box(.5,.5,W-1,H-1,radius=15,lw=1.15)
    text(22,27,'One local residual read — operator detail',27,weight='bold',ha='left')
    text(W-22,27,'Proposed · 768 channels · 8 heads · one output per child',21,color=C['muted'],ha='right')
    box(32,110,260,100,fill=C['teal_bg'],edge=C['teal'])
    text(162,136,r'Raw child memory $X_r$',23,weight='bold',max_width=236)
    text(162,174,r'$n_r\times768$ + validity mask',20,max_width=236)
    arrow([(292,146),(350,146)],C['teal'])
    box(350,110,250,100,fill=C['teal_bg'],edge=C['teal'])
    text(475,136,'Masked arithmetic mean',21,weight='bold',max_width=228)
    text(475,176,r'$\mu_r=\frac{1}{n_r}\sum_{i\in\Omega_r}X_i$',23,max_width=228)
    arrow([(600,146),(660,146)],C['teal'])
    box(660,110,310,100,fill=C['purple_bg'],edge=C['purple'])
    text(815,136,'Visual-content query',21,weight='bold',color=C['purple'])
    text(815,176,r'$Q_r=W_Q\mathrm{LN}_q(\mu_r)$',23,max_width=286)
    # Explicit untouched mean skip.
    line([(625,146),(625,69),(1570,69)],C['teal'],lw=1.6)
    arrow([(1570,69),(1570,110)],C['teal'],lw=1.6)
    label(1100,69,r'Untransformed mean residual $\mu_r$',20,C['teal'])
    arrow([(162,210),(162,279)],C['purple'])
    box(32,279,568,112,fill=C['purple_bg'],edge=C['purple'])
    text(316,306,'Normalize raw patches once; project K and V',22,weight='bold',color=C['purple'],max_width=542)
    text(316,347,r'$K_r=W_K\mathrm{LN}_x(X_r),\quad V_r=W_V\mathrm{LN}_x(X_r)$',24,max_width=540)
    arrow([(600,335),(660,335)],C['purple'])
    arrow([(970,162),(1010,162),(1010,250),(940,250),(940,279)],C['purple'])
    box(660,279,565,112,fill=C['purple_bg'],edge=C['purple'])
    text(942,304,'One masked attention operation (per head)',22,weight='bold',color=C['purple'],max_width=539)
    text(942,347,r'$A_r=\mathrm{softmax}(Q_rK_r^T/\sqrt{96}+B_r),\quad T_r=A_rV_r$',25,max_width=539)
    arrow([(1225,335),(1270,335)],C['purple'])
    box(1270,279,450,112,fill=C['purple_bg'],edge=C['purple'])
    text(1495,306,'Merge heads + output projection',22,weight='bold',color=C['purple'],max_width=424)
    text(1495,347,r'$\delta_r=W_O\,\mathrm{ConcatHeads}(T_r)$',24,max_width=424)
    arrow([(1570,279),(1570,210)],C['purple'])
    box(1420,110,300,100,fill=C['teal_bg'],edge=C['teal'])
    text(1570,135,'Residual output',23,weight='bold',color=C['teal'])
    text(1570,175,r'$u_r=\mu_r+\alpha\delta_r$',28,max_width=276)
    text(1194,153,'One local token',21,color=C['teal'])
    text(1194,184,r'$u_r:\ 1\times768$',23,color=C['teal'])
    arrow([(1420,162),(1345,162)],C['teal'])
    box(32,430,820,155,fill=C['gray_bg'],edge=C['gray_bg'],lw=0)
    text(55,455,'What is preserved',23,weight='bold',ha='left')
    text(55,491,'Raw mean bypass · same child identity / order · same total token count',20,ha='left',max_width=773)
    text(55,526,'Global branch + final shared LN / MLP · existing report CE only',20,ha='left',max_width=773)
    text(55,562,'α: one shared learned scalar, initialized to 0.1 (a proposed starting value)',18,ha='left',max_width=773,color=C['muted'])
    box(880,430,840,155,fill=C['gold_bg'],edge='#D6BD89')
    text(903,455,'What this read does not provide by itself',23,weight='bold',ha='left',color='#8E6727')
    text(903,491,'No cross-child raw memory · no explicit relative-position bias',20,ha='left',max_width=793)
    text(903,526,'No anatomical labels · no multi-layer encoder input · no local FFN',20,ha='left',max_width=793)
    text(903,562,'Attention can still become uniform or select irrelevant content.',19,ha='left',max_width=793,color=C['muted'])
    text(W/2,608,r'$B_r=0$ for valid child patches; $-\infty$ for packed padding. Shared reader weights; no lesion mask.',18,color=C['muted'])


def save(stem,out_dir):
    fig=style.fig;fig.canvas.draw();renderer=fig.canvas.get_renderer()
    for item,limit in base.TEXT_WIDTHS:
        width=item.get_window_extent(renderer).width
        if width>limit:item.set_fontsize(item.get_fontsize()*limit/width*.99)
    fig.canvas.draw()
    for item in style.TEXTS:
        b=item.get_window_extent(renderer)
        if b.x0 < -1 or b.x1 > fig.bbox.x1+1 or b.y0 < -1 or b.y1 > fig.bbox.y1+1:
            raise RuntimeError(f'Text outside figure: {item.get_text()}')
    for ext in ('pdf','svg'):fig.savefig(out_dir/f'{stem}.{ext}',facecolor='white',metadata={'Creator':'ADirect LocalRead design proposal; not trained'})
    fig.savefig(out_dir/f'{stem}.png',dpi=240,facecolor='white')
    fig.savefig(out_dir/f'{stem}_preview.png',dpi=110,facecolor='white')
    print('Saved',stem,'with',len(style.TEXTS),'text bounds checked')


def new_canvas():
    style.fig=plt.figure(figsize=(W/100,H/100),dpi=100,facecolor='white')
    style.ax=style.fig.add_axes([0,0,1,1]);style.ax.set_xlim(0,W);style.ax.set_ylim(H,0);style.ax.set_axis_off()
    style.TEXTS.clear();base.TEXT_WIDTHS.clear()


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--out-dir',type=Path,default=HERE)
    args=ap.parse_args();args.out_dir.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','mathtext.fontset':'dejavusans','svg.fonttype':'none','pdf.fonttype':42,'axes.unicode_minus':False})
    audit=audit_design()
    new_canvas();create(audit);save('bindformer_adirect_localread_overview_projector',args.out_dir);plt.close(style.fig)
    new_canvas();create_detail();save('bindformer_adirect_localread_operator_detail',args.out_dir);plt.close(style.fig)
    (args.out_dir/'proposal_spec.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2)+'\n')
