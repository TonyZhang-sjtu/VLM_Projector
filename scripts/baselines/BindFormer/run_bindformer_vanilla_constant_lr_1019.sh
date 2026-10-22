#!/usr/bin/env bash
# BindFormer vanilla, 1019: fresh training per ratio -> full-test evaluation of best.pt.
set -Eeuo pipefail

# ===== 训练参数：可直接编辑；外部 CUDA_VISIBLE_DEVICES 优先 =====
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
projector_method=bindformer3d
projector_ratios=(8 16 32)  # 支持 (64)、(32 64) 或原有倍率；也可用 --ratios 覆盖。
projector_lr=2e-4
encoder_lr=2e-4  # Vision Encoder LoRA LR
projector_max_epochs=20
projector_seed=42
projector_warmup_steps=77
projector_hold_steps=0
joint_warmup_steps=77
joint_warmup_projector_mode=hold_peak
cosine_min_lr_ratio=1.0  # 两段预热后恒定 LR
early_stopping_enabled=true
early_stopping_patience=3
early_stopping_min_delta=0.005  # 完整 validation CE
projector_run_tag="${PROJECTOR_RUN_TAG:-}"  # 留空：自动生成含变体、LR、时间戳的独立名称。

# ===== BindFormer 结构：两组仅 within_slot_sa / local_reads_global 不同 =====
bind_variant=vanilla
bind_within_slot_sa=false
bind_local_reads_global=false
bind_within_slot_sa_depth=1
bind_local_global_depth=1
bind_local_global_gate_init=0.1
bind_global_fraction=0.25
bind_slot_iterations=3
bind_knowledge_mode=name_definition
bind_geometry_enabled=false
bind_feature_loss_weight=0.0
# 四角色各 1 query；其余辅助损失为 0；UMLS 范围固定 brats_train。
# 知识库路径沿用配置，可在本机 configs/local_paths.yaml 中覆盖。
# AdamW、有效 batch=16、LoRA rank=16/alpha=32；自动评测完整 test/all/best.pt。

bind_script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$bind_script_dir/_common_1019.bash"
run_bindformer_constant_lr_1019 "$@"
