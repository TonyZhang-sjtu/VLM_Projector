#!/usr/bin/env bash
# NeuroQA: T1/T2 混合单序列 QA SFT -> validation best.pt -> test。
set -Eeuo pipefail

# ===== 可直接修改；命令行 --set / --ratios 等优先于下面的默认值 =====
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"  # 物理 GPU 编号
export PROJECTOR_PYTHON="${PROJECTOR_PYTHON:-/mnt/sda/mingjie/conda_envs/projector/bin/python}"
projector_method=token_select3d
projector_ratios=(16 32 64 128 256)  # 压缩方法支持 16 32 64 128 256；MLP 只支持 1
vqa_preprocessing_version="${VQA_PREPROCESSING_VERSION:-v2}"  # v1 / v2，同时选择 manifest + adapter + cache
vqa_invalid_sample_policy="${VQA_INVALID_SAMPLE_POLICY:-include_unreviewed}"  # include_unreviewed: 保留待复核病例；跳过损坏/缺失病例
vqa_manifest_dir=""  # 自定义存储时，manifest 中影像路径必须对应下面的 cache
vqa_cache_dir=""     # 空值使用该版本配置中的目录；不是只换 cache 就能换影像
projector_parallel_device=cpu  # 仅 Concat 生效：单卡用 cpu；双卡用 cuda:1 并配置两个物理编号
projector_seed=42
projector_run_tag=""  # 空值自动生成北京时间 + 随机后缀；每个倍率独立初始化

training_stage=joint_lora       # a: projector-only；joint_lora: projector + visual encoder LoRA
projector_lr=2e-4
encoder_lr=2e-4  # 与 report generation 的视觉 LoRA 峰值 LR 一致
projector_max_epochs=20
projector_effective_batch_size=16  # T1/T2 每轮及每个偶数窗口严格 1:1
projector_microbatch_per_device=1
encoder_lora_rank=16
encoder_lora_alpha=32
encoder_lora_dropout=0.05
projector_warmup_steps=150    # 固定 optimizer 更新次数；不按总步数比例计算
projector_warmup_ratio=0.05   # 仅显式将 steps 改为 null 时使用
projector_hold_steps=0
joint_warmup_steps=150       # 固定 optimizer 更新次数
joint_warmup_ratio=0.05      # 仅显式将 steps 改为 null 时使用
joint_warmup_projector_mode=hold_peak
cosine_min_lr_ratio=1.0      # 两段预热后保持恒定 LR，不衰减
weight_decay=0.01
grad_clip_norm=1.0
early_stopping_enabled=true
early_stopping_patience=3
early_stopping_min_delta=0.002  # validation closed accuracy，非 report loss
save_every_steps=50             # last.pt；每个 epoch 也保存 last/best
vqa_max_new_tokens=192
vqa_evaluate_test=true
vqa_eval_controls=(normal)      # 可改为 (normal text zero shuffle)，各自单独保存
vqa_bootstrap=1000

vqa_script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$vqa_script_dir/_common.bash"
run_vqa_experiment "$@"
