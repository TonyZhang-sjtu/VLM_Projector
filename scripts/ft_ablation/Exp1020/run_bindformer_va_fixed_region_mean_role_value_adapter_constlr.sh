#!/usr/bin/env bash
# Vanilla, fixed spatial regions + mean evidence aggregation (former R2).
set -Eeuo pipefail

# ===== 训练参数：可直接编辑；外部 CUDA_VISIBLE_DEVICES 优先 =====
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
projector_method=bindformer3d
projector_ratios=(32 64)  # 可改为 (8 16 32)，或使用 --ratios 覆盖。
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
cosine_min_lr_ratio=1.0  # 两段预热后恒定 LR
early_stopping_enabled=true
early_stopping_patience=3
early_stopping_min_delta=0.005  # 完整 validation CE
projector_run_tag="${PROJECTOR_RUN_TAG:-}"  # 留空：自动生成独立名称；可自定义。

# ===== BindFormer 结构：SA/CA 可分别开关；均 false 即 vanilla 交互路径 =====
routing_variant=fixed_region_mean
bind_variant="va_${routing_variant}"  # 改 SA/CA 时可同步改名称，或自定义 run tag。
bind_within_slot_sa=false
bind_local_reads_global=false
bind_within_slot_sa_depth=1
bind_local_global_depth=1
bind_local_global_gate_init=0.1
bind_global_fraction=0.25
bind_slot_iterations=3
bind_knowledge_mode=name_definition
bind_geometry_enabled=false
bind_feature_loss_weight=0.0  # 当前 CE-only；已有 feature loss 是全 token 重建。
bind_head_type=mlp  # mlp: Linear–GELU–Linear；可改为 linear。

# ===== 固定空间区域 + 区域内均值 evidence；保留此前测试的 value 归一化开关 =====
# BIND_* 环境变量优先于下列默认值；命令行 --set 的优先级最高。
routing_value_normalization="${BIND_ROUTER_VALUE_NORMALIZATION:-true}"
routing_value_projection="${BIND_ROUTER_VALUE_PROJECTION:-false}"
routing_fixed_regions="${BIND_FIXED_SPATIAL_REGIONS:-true}"
routing_mean="${BIND_SPATIAL_MEAN_AGGREGATION:-true}"
routing_adaptive="${BIND_SPATIAL_ADAPTIVE_AGGREGATION:-false}"
routing_attention_temperature="${BIND_SPATIAL_TEMPERATURE:-1.0}"
# 本隔离实验要求 fixed_regions=true；mean/adaptive 必须恰有一个 true。
# 四角色各 1 query；其余辅助损失为 0；UMLS 范围 brats_train。
# 每个 ratio 从头训练，再对 best.pt 自动执行完整 test/all 评测。
# 机器路径沿用 configs/local_paths.yaml；Python 可由 PROJECTOR_PYTHON 指定。

# ===== 隔离实验：角色 value adapter；false 为同计算路径的共享 value 对照 =====
role_value_enabled="${BIND_ROLE_VALUE_ADAPTER:-true}"
role_value_rank="${BIND_ROLE_VALUE_RANK:-16}"
role_value_alpha="${BIND_ROLE_VALUE_ALPHA:-16}"
# Python follows PROJECTOR_PYTHON, the active Conda environment, then PATH.
# 本脚本调用独立训练/评测入口，原 src 和旧脚本保持不变。
routing_script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
experiment_root="$(cd -- "$routing_script_dir/../../.." && pwd)/results/BindFormer_role_value_adapter_r16"
source "$experiment_root/_launch.bash"
run_role_value_experiment "$@"
