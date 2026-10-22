#!/usr/bin/env python3
"""Render the BindFormer paper overview as editable SVG, vector PDF, and PNG.

Only NumPy and Matplotlib are needed. Run from any working directory:
    python render_overview.py
    python render_overview.py --protocol frozen --stem bindformer_overview_frozen

This illustrates the requested research design, including the proposed L_text,
not the CE-only default experiment. Geometry/interactions remain optional.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Ellipse, FancyBboxPatch, FancyArrowPatch, PathPatch
from matplotlib.path import Path as MplPath
import numpy as np

W, H = 1800, 1440
C = {
    "ink": "#213247", "muted": "#5A6B7E", "line": "#65778B",
    "border": "#C9D3DF", "blue": "#3478AE", "blue_bg": "#EEF5FC",
    "teal": "#238B83", "teal_bg": "#EDF8F5", "purple": "#7866A4",
    "purple_bg": "#F4F1FA", "gold": "#BF8A2D", "gold_bg": "#FCF7EB",
    "rose": "#B4647C", "rose_bg": "#FBF0F4", "gray_bg": "#F5F7FA",
    "white": "#FFFFFF",
}
ROLE_COLORS = [C["teal"], C["blue"], C["gold"], C["rose"]]
ROLE_BG = [C["teal_bg"], C["blue_bg"], C["gold_bg"], C["rose_bg"]]
TEXTS = []


def text(x, y, s, size=22, color=None, weight="normal", ha="center", **kwargs):
    obj = ax.text(x, y, s, fontsize=size * 0.72, color=color or C["ink"],
                  fontweight=weight, ha=ha, va="center", zorder=8,
                  linespacing=1.3, **kwargs)
    TEXTS.append(obj)
    return obj


def box(x, y, w, h, fill="white", edge=None, radius=10, lw=1.15,
        dash=False, zorder=3):
    p = FancyBboxPatch((x, y), w, h,
                      boxstyle=f"round,pad=0,rounding_size={radius}",
                      linewidth=lw, edgecolor=edge or C["border"],
                      facecolor=fill, linestyle=(0, (5, 4)) if dash else "-",
                      zorder=zorder)
    ax.add_patch(p)
    return p


def line(points, color=None, lw=1.35, dash=False, zorder=4):
    points = np.asarray(points)
    ax.plot(points[:, 0], points[:, 1], color=color or C["line"], lw=lw,
            linestyle=(0, (4, 3)) if dash else "-", solid_capstyle="round",
            solid_joinstyle="round", zorder=zorder)


def arrow(points, color=None, lw=1.35, dash=False, zorder=4):
    path = MplPath(points, [MplPath.MOVETO] + [MplPath.LINETO] * (len(points)-1))
    p = FancyArrowPatch(path=path, arrowstyle="-|>", mutation_scale=11,
                        linewidth=lw, edgecolor=color or C["line"],
                        facecolor=color or C["line"],
                        linestyle=(0, (4, 3)) if dash else "-", zorder=zorder)
    ax.add_patch(p)


def dot(x, y, color=None, r=3.5):
    ax.add_patch(Circle((x, y), r, facecolor=color or C["line"], edgecolor="none", zorder=5))


def label(x, y, s, size=19, color=None, **kwargs):
    return text(x, y, s, size, color=color or C["muted"],
                bbox=dict(facecolor="white", edgecolor="none", pad=1.4), **kwargs)


def lock(x, y, scale=1, color=None):
    col = color or C["muted"]
    ax.add_patch(Ellipse((x, y-4*scale), 12*scale, 15*scale,
                         edgecolor=col, facecolor="none", lw=1.1, zorder=7))
    box(x-9*scale, y-4*scale, 18*scale, 14*scale,
        fill="white", edge=col, radius=2, lw=1.1, zorder=8)
    ax.add_patch(Circle((x, y+2*scale), 1.6*scale, facecolor=col, edgecolor="none", zorder=9))


def tokens(x, y, n=4, width=18, height=32, gap=5, colors=None, labels=None):
    colors = colors or [C["blue"]]*n
    for j in range(n):
        col = colors[j % len(colors)]
        box(x+j*(width+gap), y, width, height, fill=col, edge=col, radius=3, lw=0.6)
        if labels:
            text(x+j*(width+gap)+width/2, y+height/2, labels[j], 18, color="white", weight="bold")


def panel(x, y, w, h, title, subtitle=""):
    box(x, y, w, h, edge=C["border"], radius=15, lw=1.15)
    text(x+21, y+26, title, 26, weight="bold", ha="left")
    if subtitle:
        text(x+w-21, y+26, subtitle, 20, color=C["muted"], ha="right")


def brain_stack(x, y):
    """Schematic MRI slices, not a patient scan or segmentation."""
    for dx, dy in [(20, -16), (10, -8), (0, 0)]:
        box(x+dx, y+dy, 99, 85, fill="#E8EDF2", edge="#B5C0CD", radius=5, lw=.85)
        ax.add_patch(Ellipse((x+dx+49.5, y+dy+43), 67, 72,
                             facecolor="#A6B1C1", edgecolor="none", zorder=4))
        ax.add_patch(Ellipse((x+dx+48, y+dy+43), 52, 60,
                             facecolor="#DAE0E7", edgecolor="none", zorder=4))
        # Folded contours and a central fissure; deliberately schematic.
        for offset in [-1, 1]:
            pts = [(x+dx+49+offset*12, y+dy+16),
                   (x+dx+49+offset*22, y+dy+29),
                   (x+dx+49+offset*16, y+dy+42),
                   (x+dx+49+offset*23, y+dy+55),
                   (x+dx+49+offset*10, y+dy+69)]
            line(pts, "#A6B1C1", .8)
        line([(x+dx+49.5, y+dy+14), (x+dx+49.5, y+dy+70)], "#A6B1C1", .9)


def assignment_matrix(x, y):
    # Four schematic patch groups and the background column.
    a = np.array([[.76,.07,.04,.13], [.67,.14,.06,.13],
                  [.08,.74,.05,.13], [.13,.65,.07,.15],
                  [.06,.10,.73,.11], [.08,.08,.68,.16],
                  [.08,.09,.10,.73]])
    base = np.array(matplotlib.colors.to_rgb(C["purple"]))
    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            strength = .08 + .92*a[i,j]
            rgb = tuple(1-(1-base)*strength)
            box(x+j*32, y+i*12, 28, 9, fill=rgb, edge=rgb, radius=1.2, lw=0)
    return a


def read_heatmaps(x, y):
    # Different content weights, but all four roles favor a common support.
    weights = np.array([[.08,.10,.45,.72,.91,.59,.18,.08,.03,.02],
                        [.03,.08,.77,.92,.62,.38,.22,.09,.02,.02],
                        [.04,.04,.35,.58,.88,.83,.21,.05,.03,.02],
                        [.07,.10,.30,.57,.70,.64,.44,.12,.04,.02]])
    for r in range(4):
        text(x-22, y+r*20+7, "FLSM"[r], 17, color=ROLE_COLORS[r], weight="bold")
        base = np.array(matplotlib.colors.to_rgb(ROLE_COLORS[r]))
        for i, value in enumerate(weights[r]):
            rgb = tuple(1-(1-base)*(.05+.95*value))
            box(x+i*24, y+r*20, 20, 13, fill=rgb, edge=rgb, radius=1, lw=0)
    box(x+2*24-4, y-5, 5*24, 80, fill="none", edge="#8392A4", radius=4, lw=.9, dash=True, zorder=6)


def create(protocol):
    # Title and overall path.
    text(26, 34, "BindFormer", 37, weight="bold", ha="left")
    text(282, 36, "Evidence-grouped compression for 3D MRI report generation", 25, ha="left", color=C["muted"])
    panel(24, 77, 1752, 198, "(a) End-to-end framework", "One MRI sequence  ·  fixed visual-token budget")
    brain_stack(58, 141)
    text(117, 247, "3D MRI", 21, weight="bold")
    arrow([(180, 181), (218, 181)])
    box(224, 133, 214, 93, fill=C["gray_bg"])
    text(331, 161, "NeuroVFM", 26, weight="bold")
    if protocol == "joint_lora":
        text(331, 195, "Frozen base + LoRA", 20, color=C["muted"])
    else:
        lock(263, 198, .85)
        text(341, 198, "Frozen encoder", 20, color=C["muted"])
    arrow([(438, 181), (472, 181)])
    tokens(478, 156, n=4, width=15, height=49, gap=4, colors=["#A8BBCF", "#8BA5C1", "#708BAA", "#547394"])
    text(514, 238, r"$X\!: N\times768$", 22)
    arrow([(558, 181), (595, 181)])
    box(601, 127, 347, 108, fill=C["blue_bg"], edge=C["blue"], lw=1.5)
    text(774, 158, "BindFormer projector", 27, color=C["blue"], weight="bold")
    text(774, 195, "Group → read roles → compress", 21)
    arrow([(948, 181), (985, 181)])
    tokens(993, 160, n=3, width=17, height=41, gap=4, colors=[C["blue"]])
    line([(1063, 155), (1063, 207)], C["border"], 1.1)
    tokens(1075, 160, n=4, width=17, height=41, gap=4, colors=ROLE_COLORS)
    text(1076, 238, r"$Z\!: M\times4096$", 22)
    arrow([(1165, 181), (1208, 181)])
    box(1214, 133, 212, 93, fill=C["gray_bg"])
    text(1320, 160, "Llama 3.1 · 8B", 25, weight="bold")
    lock(1274, 198, .85)
    text(1344, 198, "Frozen", 21, color=C["muted"])
    arrow([(1426, 181), (1470, 181)])
    box(1476, 126, 265, 112, fill="#FCFDFE")
    text(1500, 151, "Findings", 24, weight="bold", ha="left")
    for i, length in enumerate([212, 196, 212, 162]):
        line([(1500, 177+i*13), (1500+length, 177+i*13)], "#B8C4D1", 2.0)

    # Projector internals: the same H feeds global attention and local K/V.
    panel(24, 291, 1752, 632, "(b) BindFormer projector",
          "Separate evidence groups; bind complementary attributes through shared support")
    # H backbone routed above the blocks.
    line([(295, 581), (305, 581), (305, 344), (992, 344), (992, 554)], C["blue"], 1.45)
    arrow([(992, 398), (1010, 398)], C["blue"], 1.45)
    arrow([(992, 554), (1010, 554)], C["blue"], 1.45)
    dot(992, 398, C["blue"])
    label(489, 344, r"Full visual memory $H$", 20, C["blue"])

    # Input representation and spatial/sequence context.
    box(56, 431, 230, 61, fill=C["gray_bg"])
    text(171, 453, r"$X$: all patch features", 19, weight="bold")
    text(171, 476, "No pre-pooling", 19, color=C["muted"])
    arrow([(171, 492), (171, 524)], C["blue"])
    box(56, 524, 239, 121, fill=C["blue_bg"], edge="#8FB2D2")
    text(175, 550, "Input adapter", 24, weight="bold")
    text(175, 583, "LN + projection", 21)
    text(175, 615, "3D PE + sequence embedding", 16)
    box(56, 699, 239, 88, fill=C["gray_bg"])
    text(175, 726, r"3D coordinates $p_i$", 22)
    text(175, 758, r"Sequence identity $m$", 21)
    arrow([(175, 699), (175, 645)])
    label(175, 674, "Spatial + sequence context", 18)
    arrow([(305, 581), (326, 581)], C["blue"])
    dot(305, 581, C["blue"])
    label(313, 559, r"$H$", 21, C["blue"])

    # Competitive routing and its learned assignment matrix.
    box(332, 505, 270, 252, fill=C["purple_bg"], edge="#B5A6CF")
    text(467, 530, "Competitive slot routing", 19, weight="bold", color=C["purple"])
    text(467, 559, r"3 iterations · $K+1$ slots", 20)
    for j, s in enumerate([r"$s_1$", r"$s_2$", r"$s_K$", "bg"]):
        ax.add_patch(Circle((412+j*36, 596), 13, edgecolor=C["purple"] if j < 3 else "#99A5B3",
                            facecolor="white", lw=.95, zorder=5))
        text(412+j*36, 596, s, 16, color=C["purple"] if j<3 else C["muted"])
    assignment_matrix(411, 620)
    text(385, 659, r"$A$", 26, color=C["purple"])
    text(467, 719, "Soft patch-to-slot assignment", 17.5)
    text(467, 743, r"$S$: learned evidence groups", 18)
    # S feeds both the concept lookup and the instance-conditioned query.
    line([(602, 584), (624, 584), (624, 523)], C["purple"])
    arrow([(624, 523), (645, 523)], C["purple"])
    arrow([(624, 584), (624, 627), (645, 627)], C["purple"])
    dot(624, 584, C["purple"])
    label(620, 567, r"$S$", 21, C["purple"])

    # Frozen semantic memory guides queries; it never supplies visual values.
    box(645, 365, 325, 108, fill=C["gold_bg"], edge="#D6BD89")
    text(795, 389, "Fixed concept memory", 21, color="#8E6727", weight="bold")
    text(807, 421, "UMLS names + short definitions", 18)
    text(807, 451, "SapBERT · cached embeddings", 19, color=C["muted"])
    lock(947, 388, .65, "#8E6727")
    arrow([(807, 473), (807, 491)], C["gold"])
    box(645, 491, 325, 67, fill=C["gold_bg"], edge="#D6BD89")
    text(807, 513, "Role-wise concept retrieval", 20, weight="bold")
    text(807, 540, "Role / sequence candidate mask", 19, color=C["muted"])
    arrow([(807, 558), (807, 596)], C["gold"])
    label(839, 577, r"$e_{kr}$", 20, "#8E6727")

    box(645, 596, 325, 139, fill="#FCFDFE", edge="#7A9DBA", lw=1.2)
    text(807, 621, "Instance × role queries", 24, weight="bold")
    tokens(678, 643, 4, width=57, height=34, gap=10, colors=ROLE_COLORS, labels=list("FLSM"))
    text(807, 709, r"$q_{kr}=W_qs_k+v_r+U_re_{kr}+g_{kr}$", 21)
    arrow([(970, 660), (1010, 660)], C["blue"])
    label(989, 640, r"$Q$", 19, C["blue"])

    # Optional geometry: A + coordinates → spatial/extent query cues.
    box(332, 794, 270, 76, fill="white", edge="#AEBBC9", dash=True)
    text(467, 815, "Support geometry (opt.)", 19, weight="bold")
    text(467, 845, r"$A,p\;\to\;\mu_k,\Sigma_k,\nu_k$", 22)
    arrow([(444, 757), (444, 794)], C["purple"])
    label(411, 776, r"$A$", 19, C["purple"])
    arrow([(295, 743), (313, 743), (313, 833), (332, 833)], C["muted"], lw=1.1)
    arrow([(602, 831), (625, 831), (625, 710), (645, 710)], C["muted"], lw=1.1)
    # Optional g enters only location/laterality and morphology/extent.
    text(807, 767, "F  finding     L  location / laterality", 18, color=C["muted"])
    text(807, 793, "S  signal      M  morphology / extent", 18, color=C["muted"])
    text(807, 830, "Geometry → L / M queries", 19, color=C["muted"])

    # Shared A is routed separately to the attention logits.
    # A small white halo marks the crossing with the optional geometry wire.
    line([(602, 738), (612, 738), (612, 884), (1340, 884)], "white", 4.1, zorder=4.1)
    line([(602, 738), (612, 738), (612, 884), (1340, 884)], C["purple"], 1.5, zorder=4.2)
    arrow([(1340, 884), (1340, 740)], C["purple"], 1.5)
    label(890, 884, r"Shared support $A_{:k}$ across all roles of slot $k$", 21, C["purple"])

    # Global context bypass preserves overall structure and normal findings.
    box(1010, 365, 355, 85, fill=C["blue_bg"], edge="#8FB2D2")
    text(1187, 388, "Global cross-attention", 24, weight="bold", color=C["blue"])
    text(1187, 422, r"$G$ learned queries attend to $H$", 21)
    arrow([(1365, 398), (1411, 398)], C["blue"])
    tokens(1420, 381, n=9, width=25, height=35, gap=9, colors=[C["blue"]])
    text(1570, 442, "Global structure + normal context", 20, color=C["muted"])

    box(1010, 505, 355, 235, fill=C["blue_bg"], edge=C["blue"], lw=1.3)
    text(1187, 529, "Support-biased visual read", 23, weight="bold", color=C["blue"])
    text(1187, 562, r"$\mathrm{score}(q_{kr},h_i)+\beta_r\log(A_{ik}+\epsilon)$", 22)
    text(1187, 592, "Softmax over image patches", 20, color=C["muted"])
    read_heatmaps(1080, 618)
    text(1187, 715, r"$b_{kr}=\sum_i\alpha_{kri}W_Vh_i$", 21)
    text(1187, 779, "Role-specific weights", 21, color=C["blue"], weight="bold")
    text(1187, 806, "Shared visual source", 21, color=C["blue"], weight="bold")
    label(1007, 485, r"$H$: visual keys / values", 18, C["blue"])

    # Within-slot interactions do not mix arbitrary roles across slots.
    arrow([(1365, 660), (1388, 660), (1388, 563), (1414, 563)], C["blue"])
    box(1414, 505, 326, 125, fill="white", edge="#9BACBD", dash=True)
    text(1577, 530, "Within-slot interaction (opt.)", 19, weight="bold")
    text(1577, 565, "Role self-attention", 22)
    text(1577, 600, "Gated read of global tokens", 20)
    # Same global tokens feed optional interaction and the output projection.
    line([(1722, 398), (1755, 398), (1755, 718)], C["blue"])
    arrow([(1755, 481), (1675, 481), (1675, 505)], C["blue"])
    dot(1755, 481, C["blue"])
    arrow([(1755, 718), (1740, 718)], C["blue"])
    arrow([(1577, 630), (1577, 688)], C["blue"])
    label(1577, 659, "Keep slot-wise role bundles", 19, C["blue"])
    box(1414, 688, 326, 63, fill=C["blue_bg"], edge="#8FB2D2")
    text(1577, 719, "Final LN + output MLP", 23, weight="bold")
    arrow([(1577, 751), (1577, 791)], C["blue"])
    box(1414, 797, 326, 65, fill="#FCFDFE", edge="#B6C7D7", radius=8)
    tokens(1428, 812, n=3, width=12, height=33, gap=4, colors=[C["blue"]])
    line([(1484, 807), (1484, 851)], C["border"], .9)
    for k, x in enumerate([1495, 1604]):
        tokens(x, 812, 4, width=16, height=33, gap=4, colors=ROLE_COLORS)
        if k == 0:
            text(1590, 829, "…", 22, color=C["muted"])
    text(1710, 830, r"$Z$", 27, color=C["blue"], weight="bold")
    text(1577, 885, r"$M=G+KR$  ·  $R=4$", 24, weight="bold")

    # Training objectives are explicitly attached to FINAL projected tokens.
    panel(24, 943, 1752, 462, "(c) Training with paired MRI and reports",
          "Auxiliary heads read final Z; removed for inference")

    # Feature preservation head; teacher features are targets only.
    box(46, 1000, 535, 236, fill=C["teal_bg"], edge="#A4CCC5", dash=True)
    text(68, 1027, "Visual information retention", 24, color=C["teal"], weight="bold", ha="left")
    box(67, 1075, 66, 53, fill="white", edge="#85B6AE")
    text(100, 1101, r"$Z$", 28, color=C["teal"])
    arrow([(133, 1101), (166, 1101)], C["teal"], dash=True)
    box(171, 1062, 221, 80, fill="white", edge="#85B6AE")
    text(281, 1086, r"Reconstructor $R_\psi$", 22, weight="bold")
    text(281, 1119, r"CA + FFN → 768-d", 20)
    label(150, 1078, "K/V", 16, C["teal"])
    arrow([(392, 1102), (439, 1102)], C["teal"], dash=True)
    text(475, 1102, r"$\widehat{x}_i$", 30, color=C["teal"])
    text(281, 1164, r"Queries: $p_i$ + sequence $m$", 20)
    arrow([(281, 1151), (281, 1142)], C["teal"], dash=True)
    box(67, 1186, 281, 34, fill="white", edge="#85B6AE", radius=5)
    target = r"$\mathrm{sg}(x_i^T)$ · frozen teacher" if protocol == "joint_lora" else r"$\mathrm{sg}(x_i)$ · frozen NeuroVFM"
    text(207, 1203, target, 20)
    arrow([(348, 1203), (405, 1203)], C["teal"], dash=True)
    box(410, 1180, 145, 45, fill="white", edge=C["teal"], radius=7)
    text(482, 1202, r"$\mathcal{L}_{\mathrm{feat}}$", 29, color=C["teal"])
    arrow([(475, 1124), (475, 1180)], C["teal"], dash=True)
    label(499, 1158, "LN-MSE", 17, C["teal"])

    # Report snippets align to whole role bundles through latent MIL matching.
    box(601, 1000, 605, 236, fill=C["rose_bg"], edge="#D7ADBB", dash=True)
    text(623, 1027, "Clinical phrase alignment", 24, color=C["rose"], weight="bold", ha="left")
    tokens(625, 1071, 4, width=16, height=36, gap=5, colors=ROLE_COLORS)
    text(664, 1131, r"Bundle $Z_k$", 20)
    arrow([(707, 1090), (743, 1090)], C["rose"], dash=True)
    box(748, 1063, 167, 57, fill="white", edge="#C791A4")
    text(831, 1091, "Pool + project", 19, weight="bold")
    arrow([(915, 1091), (951, 1091)], C["rose"], dash=True)
    text(968, 1091, r"$u_k$", 25, color=C["rose"])
    arrow([(986, 1091), (1018, 1091)], C["rose"], dash=True)

    box(622, 1170, 156, 48, fill="white", edge="#C791A4", radius=6)
    text(700, 1194, "Report snippets", 18)
    arrow([(778, 1194), (800, 1194)], C["rose"], dash=True)
    box(805, 1163, 185, 62, fill="white", edge="#C791A4", radius=6)
    text(898, 1183, "Text encoder", 21, weight="bold")
    text(898, 1209, r"Frozen · cached $v_\ell$", 18, color=C["muted"])
    arrow([(990, 1194), (1003, 1194), (1003, 1138), (1018, 1138)], C["rose"], dash=True)

    box(1023, 1063, 162, 154, fill="white", edge="#C791A4")
    text(1104, 1087, "MIL matching", 21, weight="bold")
    text(1104, 1123, r"$\max_k\cos(u_k,v_\ell)$", 21)
    arrow([(1104, 1143), (1104, 1163)], C["rose"], dash=True)
    text(1104, 1187, r"$\mathcal{L}_{\mathrm{text}}$", 29, color=C["rose"])

    # Language loss retains gradients through frozen decoder to visual Z.
    box(1226, 1000, 528, 236, fill=C["blue_bg"], edge="#ABC3DA", dash=True)
    text(1248, 1027, "Report generation", 24, color=C["blue"], weight="bold", ha="left")
    box(1248, 1076, 67, 53, fill="white", edge="#8FB2D2")
    text(1281, 1102, r"$Z$", 28, color=C["blue"])
    arrow([(1315, 1102), (1345, 1102)], C["blue"], dash=True)
    box(1350, 1063, 169, 79, fill="white", edge="#8FB2D2")
    text(1434, 1089, "Frozen Llama", 22, weight="bold")
    text(1434, 1120, "Teacher forcing", 19, color=C["muted"])
    arrow([(1519, 1102), (1570, 1102)], C["blue"], dash=True)
    box(1575, 1077, 150, 50, fill="white", edge=C["blue"], radius=7)
    text(1650, 1102, r"$\mathcal{L}_{\mathrm{RG}}$", 29, color=C["blue"])
    text(1650, 1154, "Token CE", 20, color=C["blue"])
    box(1248, 1181, 271, 39, fill="white", edge="#8FB2D2", radius=6)
    text(1383, 1200, r"Reference Findings $y$", 21)
    arrow([(1519, 1200), (1736, 1200), (1736, 1102), (1725, 1102)], C["blue"], dash=True)
    # y_<t also provides the autoregressive prefix during teacher forcing.
    arrow([(1434, 1181), (1434, 1142)], C["blue"], dash=True)
    label(1464, 1161, r"$y_{<t}$", 17, C["blue"])

    # Single compact training objective and protocol strip.
    box(46, 1253, 1708, 54, fill=C["gray_bg"], edge=C["gray_bg"], radius=8, lw=0)
    text(69, 1280, r"$\mathcal{L}=\mathcal{L}_{\mathrm{RG}}+\lambda_f\mathcal{L}_{\mathrm{feat}}+\lambda_s\mathcal{L}_{\mathrm{text}}$",
         30, weight="bold", ha="left")
    protocol_text = "Gradients reach the projector; Llama parameters stay frozen"
    text(1728, 1280, protocol_text, 22, ha="right", color=C["muted"])

    text(68, 1353, "Training schedule", 23, weight="bold", ha="left")
    if protocol == "joint_lora":
        stages = [
            (307, 384, "1  Projector warm-up", "Projector LR ↑ · LoRA frozen"),
            (729, 494, "2  LoRA warm-up", "LoRA LR ↑ · projector LR held constant"),
            (1261, 493, "3  Joint training", "Update projector + visual LoRA"),
        ]
        for x, w, title, detail in stages:
            box(x, 1322, w, 65, fill=C["gray_bg"], edge="#D1DBE5", radius=7)
            text(x+w/2, 1342, title, 23, weight="bold")
            text(x+w/2, 1371, detail, 20, color=C["muted"])
        arrow([(691, 1355), (723, 1355)])
        arrow([(1223, 1355), (1255, 1355)])
    else:
        box(307, 1322, 1447, 65, fill=C["gray_bg"], edge="#D1DBE5", radius=7)
        text(1030, 1355, "Train projector + auxiliary heads; keep NeuroVFM and Llama frozen", 24)

    # Legend: styles encode pathways, rather than implying frozen gradient stops.
    arrow([(29, 1425), (65, 1425)], C["line"], lw=1.2)
    text(77, 1425, "Forward / inference", 18, ha="left", color=C["muted"])
    arrow([(294, 1425), (330, 1425)], C["line"], lw=1.2, dash=True)
    text(342, 1425, "Training only", 18, ha="left", color=C["muted"])
    box(510, 1415, 28, 20, fill="none", edge="#9BACBD", dash=True, radius=4)
    text(550, 1425, "Opt.: optional module", 18, ha="left", color=C["muted"])
    text(1772, 1425, "Slots are evidence groups; the background slot emits no token.", 18, ha="right", color=C["muted"])


def save(out_dir, stem):
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    canvas = fig.bbox
    overflow = []
    for item in TEXTS:
        bb = item.get_window_extent(renderer)
        if bb.x0 < canvas.x0-1 or bb.x1 > canvas.x1+1 or bb.y0 < canvas.y0-1 or bb.y1 > canvas.y1+1:
            overflow.append(item.get_text())
    if overflow:
        raise RuntimeError(f"Text outside figure: {overflow}")
    for ext in ("svg", "pdf"):
        fig.savefig(out_dir / f"{stem}.{ext}", facecolor="white", metadata={"Creator": "BindFormer overview renderer"})
    fig.savefig(out_dir / f"{stem}.png", dpi=220, facecolor="white")
    fig.savefig(out_dir / f"{stem}_preview.png", dpi=100, facecolor="white")
    # Compact standalone crops retain vector graphics for later paper layout.
    from matplotlib.transforms import Bbox
    for name, top, bottom in [("projector", 291, 923), ("training", 943, 1405)]:
        bbox = Bbox.from_extents(24/100, (H-bottom)/100, 1776/100, (H-top)/100)
        fig.savefig(out_dir/f"{stem}_{name}.pdf", bbox_inches=bbox, facecolor="white")
    print(f"Rendered {stem}: SVG, PDF, {int(W*2.2)}×{int(H*2.2)} PNG, preview, and two panel PDFs.")
    print(f"Validated {len(TEXTS)} text objects within the figure bounds.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", choices=["joint_lora", "frozen"], default="joint_lora")
    parser.add_argument("--stem", default="bindformer_overview")
    parser.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    matplotlib.rcParams.update({"font.family": "DejaVu Sans", "mathtext.fontset": "dejavusans",
                               "svg.fonttype": "none", "pdf.fonttype": 42,
                               "ps.fonttype": 42, "axes.unicode_minus": False})
    fig = plt.figure(figsize=(W/100, H/100), dpi=100, facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.set_axis_off()
    create(args.protocol)
    save(args.out_dir, args.stem)
    plt.close(fig)
