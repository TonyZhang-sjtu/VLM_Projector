#!/usr/bin/env bash
# perceiver_flamingo: Baseline A training recipe; independent runs and best.pt full-test evaluation.
set -Eeuo pipefail

# ===== 本次实验设置：可直接修改；命令行 --set / --ratios 可显式覆盖 =====
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-3}"
export PROJECTOR_PYTHON="${PROJECTOR_PYTHON:-/mnt/sda/mingjie/conda_envs/projector/bin/python}"
projector_method=perceiver_flamingo
projector_ratios=(4 8 16 32)  # 支持 (64)、(32 64) 或原有倍率；也可用 --ratios 覆盖。
projector_lr=2e-4
encoder_lr=2e-4  # Vision Encoder LoRA LR
projector_max_epochs=20
projector_seed=42
projector_effective_batch_size=16
projector_microbatch_per_device=1
encoder_lora_rank=16
encoder_lora_alpha=32
projector_warmup_steps=77
projector_hold_steps=0
joint_warmup_steps=77
joint_warmup_projector_mode=hold_peak
cosine_min_lr_ratio=1.0  # 两段预热后保持恒定 LR，不衰减
early_stopping_enabled=true
early_stopping_patience=3
early_stopping_min_delta=0.005  # 完整 validation answer CE
projector_run_tag=""    # 留空自动生成时间戳；手动指定时每次使用新名称

# 与 Baseline A 共用 constant_lr.yaml：AdamW、bf16、LoRA dropout=0.05，
# weight_decay=0.01、grad_clip=1、固定数据划分及生成参数；机器路径来自 local_paths.yaml。
# 每个倍率独立初始化，自动评测最低 validation CE 的 best.pt：完整 test / 全指标。
# 新实验直接运行本脚本；--resume-run 会沿用旧计划，不能用来迁移旧 1e-5 配置。
const_script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$const_script_dir/../ft_ablation/_constant_lr.bash"
run_constant_lr_baseline \
  --set "training.effective_batch_size=$projector_effective_batch_size" \
  --set "training.microbatch_per_device=$projector_microbatch_per_device" \
  --set "training.lora.rank=$encoder_lora_rank" \
  --set "training.lora.alpha=$encoder_lora_alpha" \
  "$@"
