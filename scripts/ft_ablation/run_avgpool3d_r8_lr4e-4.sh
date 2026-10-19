#!/usr/bin/env bash
# AvgPool3D 8x upper-LR probe: 4e-4 after warm-ups, validation-CE early stopping.
set -Eeuo pipefail

# ===== 本次实验设置；可直接修改，命令行 --set / --ratios 也可覆盖 =====
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
projector_method=avgpool3d
projector_ratios=(8)  # 上限实验默认 8x；压缩率可改为 (4 8 16 32)。
projector_lr=4e-4
encoder_lr=4e-4  # Vision Encoder LoRA LR
projector_max_epochs=20
projector_seed=42
projector_warmup_steps=77
projector_hold_steps=0
joint_warmup_steps=77
joint_warmup_projector_mode=hold_peak
cosine_min_lr_ratio=1.0  # 1.0 = 两段预热后保持恒定 LR
early_stopping_enabled=true
early_stopping_patience=3
early_stopping_min_delta=0.005
projector_run_tag="${PROJECTOR_RUN_TAG:-}"  # 留空自动命名；重跑时使用新名称。

# 77-step projector warm-up + 77-step LoRA warm-up (projector stays at peak).
# hold=0, constant LR afterwards; patience=3, min_delta=0.005; test best.pt only.
const_script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$const_script_dir/_constant_lr.bash"
run_constant_lr_baseline "$@"
