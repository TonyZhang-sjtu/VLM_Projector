#!/usr/bin/env bash
# NeuroQA Baseline A: vanilla BindFormer + fixed regions + mean aggregation.
# 对齐 report Baseline A 的视觉结构；保留 VQA 配方，关闭 report-derived UMLS bank。
set -Eeuo pipefail

# ===== GPU、倍率、预处理和训练参数；CLI 覆盖优先 =====
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PROJECTOR_PYTHON="${PROJECTOR_PYTHON:-/mnt/sda/mingjie/conda_envs/projector/bin/python}"
projector_method=bindformer3d
projector_ratios=(128 256)  # 也支持 (16 32 64 128 256)，或 --ratios 覆盖
vqa_preprocessing_version="${VQA_PREPROCESSING_VERSION:-v2}"
vqa_invalid_sample_policy="${VQA_INVALID_SAMPLE_POLICY:-include_unreviewed}"
vqa_manifest_dir=""
vqa_cache_dir=""
projector_parallel_device=cpu  # 仅 Concat 使用此参数；BindFormer 在所选 GPU 上运行
projector_seed=42
projector_run_tag="${PROJECTOR_RUN_TAG:-}"
if [[ -z "$projector_run_tag" ]]; then
  projector_run_tag="baseline_a_no_knowledge_$(TZ=Asia/Shanghai date +%Y%m%dT%H%M%S)_${RANDOM}${RANDOM}"
fi

training_stage=joint_lora  # a: projector-only；joint_lora: projector + visual encoder LoRA
projector_lr=2e-4
encoder_lr=2e-4  # 与 report generation 的视觉 LoRA 峰值 LR 一致
projector_max_epochs=20
projector_effective_batch_size=16  # v2 每轮及每个偶数累积窗口内 T1/T2 严格 1:1
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
early_stopping_min_delta=0.002  # validation closed accuracy
save_every_steps=50
vqa_max_new_tokens=192
vqa_evaluate_test=true
vqa_eval_controls=(normal)
vqa_bootstrap=1000

# ===== Baseline A 视觉结构 =====
# 与 Exp1026/run_baseline_a_r128_r256.sh 及其 Exp1020 被调脚本对齐。
# 不读取外部 BIND_* 环境变量，避免被其他 report 消融实验的开关影响。
bind_router_value_normalization=true
bind_router_value_projection=false
bind_fixed_spatial_regions=true
bind_spatial_mean_aggregation=true
bind_spatial_adaptive_aggregation=false
bind_spatial_attention_temperature=1.0
bind_within_slot_sa=false
bind_local_reads_global=false
bind_within_slot_sa_depth=1
bind_local_global_depth=1
bind_local_global_gate_init=0.1
bind_global_fraction=0.25
bind_slot_iterations=3
bind_geometry_enabled=false
bind_head_type=mlp
bind_knowledge_mode=none  # 现有 concept bank 来源为 report train；QA 版本重建前保持关闭

baseline_a_args=()
for bind_key in router_value_normalization router_value_projection fixed_spatial_regions \
    spatial_mean_aggregation spatial_adaptive_aggregation spatial_attention_temperature \
    within_slot_sa local_reads_global within_slot_sa_depth local_global_depth \
    local_global_gate_init global_fraction slot_iterations geometry_enabled head_type knowledge_mode; do
  bind_var="bind_$bind_key"
  baseline_a_args+=(--set "projector.params.$bind_key=${!bind_var}")
done
baseline_a_args+=(--set projector.params.vocabulary_scope=brats_train)
for bind_role in finding location_laterality signal morphology_extent; do
  baseline_a_args+=(--set "projector.params.role_query_counts.$bind_role=1")
done
# QA answer-only CE；与 report Baseline A 一样关闭各项辅助损失。
for bind_loss in feature text tuple binding concept; do
  baseline_a_args+=(--set "projector.params.${bind_loss}_loss_weight=0.0")
done

vqa_script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$vqa_script_dir/_common.bash"
run_vqa_experiment "${baseline_a_args[@]}" "$@"
