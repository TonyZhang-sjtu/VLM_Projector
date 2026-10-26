#!/usr/bin/env bash
# 独立标准入口：E2=CE+Decoder 配对；E3=CE+两者；E1=CE+最终 Z 属性。
# --dry-run 仅打印完整计划。正式运行训练后自动评测 best.pt 的完整 test/all。
set -Eeuo pipefail
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}"

# ===== 实验开关；环境变量或编辑这里，CLI --set 最后覆盖 =====
experiment="${EXPERIMENT:-E2}"
read -r -a projector_ratios <<< "${PROJECTOR_RATIOS:-32}"
case "$experiment" in
  E0) default_fact=false; default_pair=false ;;
  E1) default_fact=true;  default_pair=false ;;
  E2) default_fact=false; default_pair=true ;;
  E3) default_fact=true;  default_pair=true ;;
  *) echo 'EXPERIMENT must be E0/E1/E2/E3' >&2; exit 2 ;;
esac
fact_enabled="${FACT_ENABLED:-$default_fact}"
pair_enabled="${PAIR_ENABLED:-$default_pair}"
fact_weight="${FACT_WEIGHT:-0.1}"
pair_weight="${PAIR_WEIGHT:-0.1}"
pair_positive_ce_weight="${PAIR_POSITIVE_CE_WEIGHT:-0.25}"
pair_margin="${PAIR_MARGIN:-0.2}"
aux_warmup_steps="${AUX_WARMUP_STEPS:-77}"
pair_every_steps="${PAIR_EVERY_STEPS:-1}"
facts_path="${REPORT_FACTS_PATH:-assets/knowledge/report_facts_v1}"
# 默认关闭 margin 边界辅助监督；morphology/extent 暂无独立属性监督。
# 四角色 token 和原报告 CE 保留；REPORT_FACT_FIELDS / CLI --set 可显式调整字段。
facts_fields="${REPORT_FACT_FIELDS:-[edema,laterality,site,signal]}"
for flag in "$fact_enabled" "$pair_enabled"; do
  [[ "$flag" == true || "$flag" == false ]] || { echo 'Loss switches must be true/false' >&2; exit 2; }
done
[[ "$fact_enabled" == true ]] || fact_weight=0.0
[[ "$pair_enabled" == true ]] || pair_weight=0.0
aux_enabled=true
if [[ "$fact_enabled" == false && "$pair_enabled" == false ]]; then aux_enabled=false; fi

# ===== 原 fixed_region_mean constant-LR 配置，不改训练配方 =====
projector_lr=2e-4
encoder_lr=2e-4
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
cosine_min_lr_ratio=1.0
early_stopping_enabled=true
early_stopping_patience=3
early_stopping_min_delta=0.005
bind_within_slot_sa=false
bind_local_reads_global=false
bind_global_fraction=0.25
bind_knowledge_mode=name_definition
bind_head_type=mlp
routing_mean="${BIND_SPATIAL_MEAN_AGGREGATION:-true}"
routing_adaptive="${BIND_SPATIAL_ADAPTIVE_AGGREGATION:-false}"

root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
if [[ ! "$CUDA_VISIBLE_DEVICES" =~ ^[0-9]+$ ]]; then echo 'Select one physical GPU.' >&2; exit 2; fi
if [[ -n "${PROJECTOR_PYTHON:-}" ]]; then
  python_bin="$PROJECTOR_PYTHON"
elif [[ -n "${CONDA_PREFIX:-}" && -x "$CONDA_PREFIX/bin/python" ]]; then
  python_bin="$CONDA_PREFIX/bin/python"
else
  python_bin="$(command -v python)"
fi
[[ -x "$python_bin" ]] || { echo "Python is not executable: $python_bin" >&2; exit 2; }
local_config="${PROJECTOR_LOCAL_CONFIG:-$root/configs/local_paths.yaml}"
path_args=()
if [[ -f "$local_config" ]]; then path_args=(--config "$local_config")
elif [[ -n "${PROJECTOR_LOCAL_CONFIG:-}" ]]; then echo "Missing $local_config" >&2; exit 2; fi
run_tag="${PROJECTOR_RUN_TAG:-bindformer_mean_facts_v1_${experiment}_$(date -u +%Y%m%dT%H%M%S%NZ)}"
export PROJECTOR_PYTHON="$python_bin"
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 PYTHONUTF8=1 PYTHONIOENCODING=utf-8
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
cd -- "$root"
echo "[$experiment] GPU=$CUDA_VISIBLE_DEVICES ratios=${projector_ratios[*]} fact=$fact_weight pair=$pair_weight fields=$facts_fields; fresh initialization unless --resume-run is supplied" >&2
exec "$python_bin" -u "$root/experiments/report_facts_v1/entry.py" src.cli.baseline_sweep \
  --config "$root/configs/training/constant_lr.yaml" "${path_args[@]}" \
  --run-tag "$run_tag" --ratios "${projector_ratios[@]}" \
  --output-root "${PROJECTOR_OUTPUT_ROOT:-$root/runs/report_facts_v1}" \
  --device cuda:0 --scoring-device cuda:0 \
  --set "seed=$projector_seed" \
  --set "training.projector_lr=$projector_lr" --set "training.encoder_lr=$encoder_lr" \
  --set "training.max_epochs=$projector_max_epochs" \
  --set "training.projector_warmup_steps=$projector_warmup_steps" \
  --set "training.projector_hold_steps=$projector_hold_steps" \
  --set "training.joint_warmup_steps=$joint_warmup_steps" \
  --set "training.joint_warmup_projector_mode=$joint_warmup_projector_mode" \
  --set "training.cosine_min_lr_ratio=$cosine_min_lr_ratio" \
  --set "training.early_stopping_enabled=$early_stopping_enabled" \
  --set "training.early_stopping_patience=$early_stopping_patience" \
  --set "training.early_stopping_min_delta=$early_stopping_min_delta" \
  --set "training.effective_batch_size=$projector_effective_batch_size" \
  --set "training.microbatch_per_device=$projector_microbatch_per_device" \
  --set "training.lora.rank=$encoder_lora_rank" --set "training.lora.alpha=$encoder_lora_alpha" \
  --set "projector.params.within_slot_sa=$bind_within_slot_sa" \
  --set "projector.params.local_reads_global=$bind_local_reads_global" \
  --set projector.params.within_slot_sa_depth=1 --set projector.params.local_global_depth=1 \
  --set projector.params.local_global_gate_init=0.1 --set "projector.params.global_fraction=$bind_global_fraction" \
  --set projector.params.slot_iterations=3 --set "projector.params.knowledge_mode=$bind_knowledge_mode" \
  --set projector.params.vocabulary_scope=brats_train \
  --set projector.params.role_query_counts.finding=1 --set projector.params.role_query_counts.location_laterality=1 \
  --set projector.params.role_query_counts.signal=1 --set projector.params.role_query_counts.morphology_extent=1 \
  --set projector.params.geometry_enabled=false --set projector.params.feature_loss_weight=0.0 \
  --set projector.params.text_loss_weight=0.0 --set projector.params.tuple_loss_weight=0.0 \
  --set projector.params.binding_loss_weight=0.0 --set projector.params.concept_loss_weight=0.0 \
  --set "projector.params.head_type=$bind_head_type" \
  --set "projector.params.router_value_normalization=${BIND_ROUTER_VALUE_NORMALIZATION:-true}" \
  --set "projector.params.router_value_projection=${BIND_ROUTER_VALUE_PROJECTION:-false}" \
  --set projector.params.fixed_spatial_regions=true \
  --set "projector.params.spatial_mean_aggregation=$routing_mean" \
  --set "projector.params.spatial_adaptive_aggregation=$routing_adaptive" \
  --set "projector.params.spatial_attention_temperature=${BIND_SPATIAL_TEMPERATURE:-1.0}" \
  --set "projector.params.report_facts.enabled=$aux_enabled" \
  --set "projector.params.report_facts.path=$facts_path" \
  --set "projector.params.report_facts.fields=$facts_fields" \
  --set "projector.params.report_facts.fact_weight=$fact_weight" \
  --set "projector.params.report_facts.pair_weight=$pair_weight" \
  --set "projector.params.report_facts.positive_ce_weight=$pair_positive_ce_weight" \
  --set "projector.params.report_facts.margin=$pair_margin" \
  --set "projector.params.report_facts.warmup_steps=$aux_warmup_steps" \
  --set "projector.params.report_facts.pair_every_steps=$pair_every_steps" \
  "$@" --projector bindformer3d --eval-split test --metrics all --eval-sample-fraction 1.0 --require-full-test
