#!/usr/bin/env python3
"""Draw the implemented ADirect projector in the A4 overview style.

Produces editable SVG, vector PDF, print PNG, preview and a code-derived audit.
Uses the original renderer's drawing primitives; never modifies that renderer.
No pretrained models, GPU, training data or existing checkpoints are loaded.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
STYLE = HERE.parent / 'bindformer_overview' / 'render_overview.py'
spec = importlib.util.spec_from_file_location('bindformer_figure_style', STYLE)
style = importlib.util.module_from_spec(spec)
spec.loader.exec_module(style)
plt, C = style.plt, style.C
box, line, arrow = style.box, style.line, style.arrow
tokens, dot, label = style.tokens, style.dot, style.label
W, H = 1752, 632  # Exact aspect ratio and physical size of the reference panel.
CHILD_COLORS = ['#238B83', '#419C93', '#61ADA5', '#82BDB5']
TEXT_WIDTHS = []


def text(x, y, s, size=22, max_width=None, **kwargs):
    item = style.text(x, y, s, size, **kwargs)
    if max_width is not None:
        TEXT_WIDTHS.append((item, max_width))
    return item


def code_audit():
    sys.path.insert(0, str(ROOT))
    from src.config import load_config
    from src.models.projectors.registry import build_projector
    from src.models.projectors.budget import resolve_budget

    result = {'projector_name': 'bindformer_adirect', 'class': 'BindFormerADirect',
              'initialization': 'fresh projector; no trained A or A4 checkpoint',
              'scope': 'Projector only; encoder/LLM/vision LoRA excluded',
              'local_feature_source': 'raw encoder output X',
              'local_transform_before_pooling': 'none',
              'local_position_encoding': 'none added by projector',
              'local_sequence_condition': False,
              'local_token_order': 'parent_major_child_major',
              'shared_output_mapping': 'per-token LN -> Linear -> GELU -> Linear',
              'ratios': {}, 'source_sha256': {}}
    for ratio in (128, 256):
        c = load_config([ROOT / 'configs/projector/bindformer_adirect.yaml'],
                        [f'projector.target_compression_ratio={ratio}'])
        budget = resolve_budget(c.projector, (39, 15, 15), (156., 240., 240.))
        model = build_projector(c, 768, 4096)
        modules = {name: sum(p.numel() for p in module.parameters())
                   for name, module in model.named_children()}
        direct = {name: p.numel() for name, p in model.named_parameters(recurse=False)}
        groups = {
            '1_global_input_adapter': modules['input_adapter'],
            '2_global_branch': modules['global_layers'] + direct['global_queries'],
            '3_local_branch': 0,
            '4_output_mapping': modules['final_norm'] + modules['head'],
            'spatial_partition': 0,
        }
        total = sum(p.numel() for p in model.parameters())
        assert total == sum(groups.values())
        assert all(p.requires_grad for p in model.parameters())
        assert set(modules) == {'input_adapter', 'global_layers', 'final_norm', 'head'}
        assert set(direct) == {'global_queries'}
        forbidden = ('router', 'role_queries', 'reader', 'concept_memory',
                     'interactions', 'geometry_encoder', 'feature_reconstructor',
                     'value_norm', 'value', 'output', 'ffn')
        assert not any(hasattr(model, name) for name in forbidden)
        result['ratios'][str(ratio)] = {
            'budget': budget, 'functional_groups': groups,
            'top_level_modules': modules, 'direct_parameters': direct,
            'module_types': dict(Counter(type(m).__name__ for m in model.modules())),
            'trainable_projector_parameters': total,
            'output_mapping_parameter_fraction': groups['4_output_mapping'] / total,
        }
    for path in [STYLE, Path(__file__), ROOT/'configs/projector/bindformer_adirect.yaml',
                 ROOT/'src/models/projectors/registry.py', ROOT/'src/models/projectors/specs.py',
                 ROOT/'src/models/projectors/budget.py', ROOT/'src/models/projectors/common.py',
                 ROOT/'src/models/projectors/bindformer/input_adapter.py',
                 ROOT/'src/models/projectors/bindformer/regions.py',
                 ROOT/'src/models/projectors/bindformer_adirect/projector.py',
                 ROOT/'src/models/projectors/bindformer_adirect/config.py',
                 ROOT/'src/models/projectors/bindformer_a4/config.py',
                 ROOT/'src/models/projectors/bindformer_a4/regions.py']:
        result['source_sha256'][str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


def create(audit):
    box(.5, .5, W-1, H-1, radius=15, lw=1.15)
    text(22, 27, '(b) ADirect projector', 27, weight='bold', ha='left')
    text(W-22, 27, 'Global cross-attention + direct local mean tokens', 22,
         color=C['muted'], ha='right')

    # Split BEFORE the adapter: only the global path receives adapted H.
    box(32, 258, 234, 82, fill=C['gray_bg'])
    text(149, 283, r'$X$: all patch features', 21, weight='bold', max_width=214)
    text(149, 315, r'$N\times768$  ·  $N=8775$', 19, color=C['muted'])
    arrow([(149, 258), (149, 204)], C['blue'])
    box(32, 90, 234, 114, fill=C['blue_bg'], edge='#8FB2D2')
    text(149, 112, '1  Input adapter', 23, weight='bold')
    text(149, 143, 'LN + Linear (768 → 768)', 18.5, max_width=214)
    text(149, 170, 'Grid Fourier PE + sequence', 17.5, max_width=214)
    text(149, 191, 'Global branch only', 16.5, color=C['blue'])
    box(32, 407, 234, 82, fill=C['gray_bg'])
    text(149, 432, r'Grid / geometry $p_i$', 20)
    text(149, 464, r'MRI sequence identity $m$', 19, max_width=214)
    arrow([(32, 448), (16, 448), (16, 180), (32, 180)], C['blue'], lw=1.1)

    line([(266, 147), (294, 147), (294, 64), (700, 64), (700, 147)], C['blue'], lw=1.5)
    arrow([(700, 147), (720, 147)], C['blue'], lw=1.5)
    label(493, 64, r'Full visual memory $H$: $N\times768$', 19, C['blue'])

    box(350, 96, 280, 69, fill=C['blue_bg'], edge='#8FB2D2')
    text(490, 118, r'Learned global queries $Q_g$', 20, weight='bold', max_width=260)
    text(490, 146, r'$G\times768$  ·  fresh initialization', 18, color=C['muted'], max_width=260)
    arrow([(630, 119), (720, 119)], C['blue'])
    box(720, 90, 394, 95, fill=C['blue_bg'], edge='#8FB2D2')
    text(917, 114, '2  Global cross-attention', 23, color=C['blue'], weight='bold')
    text(917, 146, '1 block · 8 heads · residual + FFN', 20)
    text(917, 170, r'$Q_g$ reads keys / values from all $H$', 17.5, color=C['muted'])
    arrow([(1114, 127), (1150, 127)], C['blue'])
    tokens(1158, 109, n=6, width=23, height=36, gap=8, colors=[C['blue']])
    text(1248, 171, r'Global tokens: $G\times768$', 19, color=C['blue'])
    arrow([(1341, 127), (1565, 127), (1565, 248)], C['blue'], lw=1.5)

    # A literal direct path: raw X -> arithmetic means -> shared final LN/head.
    text(330, 207, '3  Direct local branch', 22, color=C['teal'], weight='bold', ha='left')
    text(490, 261, r'Raw encoder features $X$', 22, weight='bold', color=C['teal'], max_width=380)
    arrow([(266, 293), (735, 293)], C['teal'], lw=2)
    text(490, 325, 'Pool before any trainable transform', 19, color=C['muted'], max_width=380)
    box(735, 236, 320, 139, fill=C['teal_bg'], edge=C['teal'], lw=1.3)
    text(895, 261, 'One mean per spatial child', 22, weight='bold', color=C['teal'], max_width=294)
    text(895, 309, r'$\ell_{k,c}=\frac{1}{|\Omega_{k,c}|}\sum_{i\in\Omega_{k,c}}X_i$', 25)
    text(895, 357, r'$c=1,2,3,4$  ·  disjoint local support', 18, color=C['muted'], max_width=294)
    arrow([(1055, 293), (1090, 293)], C['teal'], lw=1.5)
    box(1090, 236, 270, 139, fill=C['teal_bg'], edge='#85B6AE')
    text(1225, 261, 'Direct local tokens', 22, color=C['teal'], weight='bold', max_width=244)
    tokens(1157, 284, n=4, width=25, height=28, gap=8, colors=CHILD_COLORS,
           labels=['1', '2', '3', '4'])
    text(1225, 333, r'Local tokens: $4K\times768$', 18.5, color=C['teal'])
    text(1225, 359, 'parent-major / child-major', 17.5, color=C['muted'], max_width=244)
    arrow([(1360, 293), (1380, 293), (1380, 282), (1410, 282)], C['teal'], lw=1.5)

    # Geometry controls membership only; there is no learned slot or atlas label.
    box(330, 397, 385, 132, fill=C['gold_bg'], edge='#D6BD89')
    text(522, 419, 'Deterministic spatial partition', 22, color='#8E6727', weight='bold', max_width=359)
    text(522, 448, r'$K$ A / A4 regions → 4 children per region', 19, max_width=359)
    box(351, 470, 51, 37, fill='white', edge='#B9A275', radius=3)
    arrow([(410, 488), (441, 488)], C['gold'])
    for c in range(4):
        x, y = 449+(c%2)*29, 466+(c//2)*23
        box(x, y, 26, 20, fill=CHILD_COLORS[c], edge=CHILD_COLORS[c], radius=2, lw=.5)
        text(x+13, y+10, str(c+1), 14, color='white', weight='bold')
    text(611, 484, 'Geometry only\n0 learned parameters', 17.5, color=C['muted'])
    arrow([(266, 448), (330, 448)], C['gold'])
    arrow([(715, 419), (895, 419), (895, 375)], C['gold'], lw=1.3)
    label(895, 402, 'Child indices + validity mask', 17.5, '#8E6727')

    box(735, 443, 625, 86, fill=C['gray_bg'], edge=C['gray_bg'], radius=8, lw=0)
    text(1047, 464, 'Local pooling: 0 learned parameters; final LN + MLP is shared', 20, weight='bold', max_width=599)
    text(1047, 491, 'Local path adds no position / sequence embedding or V / output / FFN layers.', 17, max_width=599)
    text(1047, 515, 'Same A4 spatial children and token budget; colors indicate spatial subdivisions.', 16.5,
         color=C['muted'], max_width=599)

    # The retained A4 shared head remains token-wise, without token mixing.
    box(1410, 248, 310, 68, fill=C['gray_bg'])
    text(1565, 270, 'Concatenate [Global ; Local]', 20, weight='bold', max_width=284)
    text(1565, 298, r'$(G+4K)\times768$', 23)
    arrow([(1565, 316), (1565, 359)], C['blue'])
    box(1410, 359, 310, 81, fill=C['blue_bg'], edge='#8FB2D2')
    text(1565, 384, '4  Final LN + output MLP', 22, weight='bold', max_width=284)
    text(1565, 414, '768 → 4096 → 4096 · GELU', 19)
    arrow([(1565, 440), (1565, 476)], C['blue'])
    box(1410, 476, 310, 60, fill='#FCFDFE', edge='#B6C7D7', radius=8)
    tokens(1430, 491, n=3, width=13, height=30, gap=5, colors=[C['blue']])
    line([(1490, 485), (1490, 527)], C['border'], .9)
    tokens(1503, 491, n=4, width=18, height=30, gap=6, colors=CHILD_COLORS)
    text(1611, 506, '…', 24, color=C['muted'])
    text(1669, 506, r'$Z$', 29, color=C['blue'], weight='bold')
    text(1565, 560, r'$M=G+4K$;  $Z: M\times4096$', 23, weight='bold')
    text(1565, 590, 'Visual prefix → frozen Llama', 19, color=C['muted'])

    box(32, 554, 1328, 60, fill=C['gray_bg'], edge=C['gray_bg'], radius=8, lw=0)
    for ratio, x in [(128, 51), (256, 716)]:
        row = audit['ratios'][str(ratio)]
        b = row['budget']
        text(x, 574, f'{ratio}×: {b["global_tokens"]} global + {b["local_tokens"]} local = '
             f'{b["expected_output_tokens"]} visual tokens', 21, weight='bold', ha='left')
        text(x, 600, f'Actual {b["actual_ratio"]:.3f}×  ·  '
             f'{row["trainable_projector_parameters"]/1e6:.3f}M trainable projector parameters',
             18.5, color=C['muted'], ha='left')


def save(out_dir, stem, audit):
    out_dir.mkdir(parents=True, exist_ok=True)
    fig = style.fig
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    # Keep labels inside their boxes, rather than checking the outer page only.
    for item, limit in TEXT_WIDTHS:
        width = item.get_window_extent(renderer).width
        if width > limit:
            item.set_fontsize(item.get_fontsize() * limit / width * .99)
    fig.canvas.draw()
    overflow = []
    for item in style.TEXTS:
        bb = item.get_window_extent(renderer)
        if bb.x0 < -1 or bb.x1 > fig.bbox.x1+1 or bb.y0 < -1 or bb.y1 > fig.bbox.y1+1:
            overflow.append(item.get_text())
    if overflow:
        raise RuntimeError(f'Text outside figure: {overflow}')
    for item, limit in TEXT_WIDTHS:
        if item.get_window_extent(renderer).width > limit+1:
            raise RuntimeError(f'Text exceeds its box: {item.get_text()}')
    for ext in ('svg', 'pdf'):
        fig.savefig(out_dir/f'{stem}.{ext}', facecolor='white',
                    metadata={'Creator': 'ADirect code-matched overview renderer'})
    fig.savefig(out_dir/f'{stem}.png', dpi=240, facecolor='white')
    fig.savefig(out_dir/f'{stem}_preview.png', dpi=110, facecolor='white')
    (out_dir/'architecture_audit.json').write_text(json.dumps(audit, indent=2)+'\n')
    print(f'Saved {stem}: vector PDF, editable SVG, PNG and preview; {len(style.TEXTS)} text bounds checked.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out-dir', type=Path, default=HERE)
    parser.add_argument('--stem', default='bindformer_adirect_overview_projector')
    args = parser.parse_args()
    style.matplotlib.rcParams.update({'font.family': 'DejaVu Sans', 'mathtext.fontset': 'dejavusans',
        'svg.fonttype': 'none', 'pdf.fonttype': 42, 'ps.fonttype': 42, 'axes.unicode_minus': False})
    style.fig = plt.figure(figsize=(W/100, H/100), dpi=100, facecolor='white')
    style.ax = style.fig.add_axes([0, 0, 1, 1])
    style.ax.set_xlim(0, W)
    style.ax.set_ylim(H, 0)
    style.ax.set_axis_off()
    style.TEXTS.clear()
    audit = code_audit()
    create(audit)
    save(args.out_dir, args.stem, audit)
    plt.close(style.fig)
