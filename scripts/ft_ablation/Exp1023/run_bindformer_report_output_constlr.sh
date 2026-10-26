#!/usr/bin/env bash
# R1 = report CE + span CE; R2 = R1 + all reliable report-context negatives.
# All base settings are inherited from the unchanged fixed-region-mean launcher.
set -Eeuo pipefail
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
experiment="${EXPERIMENT:-R2}"
read -r -a ratios <<< "${PROJECTOR_RATIOS:-32}"
case "$experiment" in
  R0) default_span=false; default_group=false ;;
  R1) default_span=true;  default_group=false ;;
  R2) default_span=true;  default_group=true ;;
  *) echo 'EXPERIMENT must be R0/R1/R2' >&2; exit 2 ;;
esac
span_enabled="${SPAN_ENABLED:-$default_span}"
group_enabled="${GROUP_ENABLED:-$default_group}"
span_weight="${SPAN_WEIGHT:-0.1}"
group_weight="${GROUP_WEIGHT:-0.1}"
group_margin="${GROUP_MARGIN:-0.2}"
aux_warmup_steps="${AUX_WARMUP_STEPS:-77}"
group_every_steps="${GROUP_EVERY_STEPS:-1}"
facts_path="${REPORT_OUTPUT_PATH:-assets/knowledge/report_output_v1}"
fields="${REPORT_OUTPUT_FIELDS:-[edema,laterality,site,signal]}"
val_enabled="${VAL_FACTS_ENABLED:-true}"
val_every_epochs="${VAL_FACTS_EVERY_EPOCHS:-1}"
val_limit="${VAL_FACTS_LIMIT:-0}"  # 0 = full validation; not a training-data limit.
save_best_facts="${SAVE_BEST_FACTS:-true}"
for flag in "$span_enabled" "$group_enabled" "$val_enabled" "$save_best_facts"; do
  [[ "$flag" == true || "$flag" == false ]] || { echo 'Switches must be true/false' >&2; exit 2; }
done
[[ "$span_enabled" == true ]] || span_weight=0.0
[[ "$group_enabled" == true ]] || group_weight=0.0
enabled=true
if [[ "$span_enabled" == false && "$group_enabled" == false ]]; then enabled=false; fi

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
root="$(cd -- "$script_dir/../../.." && pwd)"
if [[ -n "${PROJECTOR_PYTHON:-}" ]]; then
  real_python="$PROJECTOR_PYTHON"
elif [[ -n "${CONDA_PREFIX:-}" && -x "$CONDA_PREFIX/bin/python" ]]; then
  real_python="$CONDA_PREFIX/bin/python"
else
  real_python="$(command -v python)"
fi
[[ -x "$real_python" ]] || { echo "Python is not executable: $real_python" >&2; exit 2; }
[[ "$real_python" != "$root/experiments/report_output_v1/python_entry.sh" ]] || { echo 'Recursive interpreter bridge' >&2; exit 2; }
export REPORT_OUTPUT_REAL_PYTHON="$real_python"
export PROJECTOR_PYTHON="$root/experiments/report_output_v1/python_entry.sh"
export PROJECTOR_OUTPUT_ROOT="${PROJECTOR_OUTPUT_ROOT:-$root/runs/report_output_v1}"
export PROJECTOR_RUN_TAG="${PROJECTOR_RUN_TAG:-report_output_v1_${experiment}_$(date -u +%Y%m%dT%H%M%S%NZ)}"
echo "[$experiment] GPU=$CUDA_VISIBLE_DEVICES ratios=${ratios[*]} span=$span_weight group=$group_weight fields=$fields" >&2
echo 'Base: Exp1020/run_bindformer_va_fixed_region_mean_constlr.sh; fresh initialization; no segmentation; enhancement/margin auxiliary supervision disabled.' >&2
# CLI --set overrides are intentionally last. Never copy or alter the base LR,
# optimizer, sampling, prompt, LoRA, early stopping or full-test scoring recipe.
exec bash "$root/scripts/ft_ablation/Exp1020/run_bindformer_va_fixed_region_mean_constlr.sh" \
  --ratios "${ratios[@]}" \
  --set "projector.params.report_output.enabled=$enabled" \
  --set "projector.params.report_output.path=$facts_path" \
  --set "projector.params.report_output.fields=$fields" \
  --set "projector.params.report_output.span_weight=$span_weight" \
  --set "projector.params.report_output.group_weight=$group_weight" \
  --set "projector.params.report_output.margin=$group_margin" \
  --set "projector.params.report_output.warmup_steps=$aux_warmup_steps" \
  --set "projector.params.report_output.group_every_steps=$group_every_steps" \
  --set "projector.params.report_output.validation_enabled=$val_enabled" \
  --set "projector.params.report_output.validation_every_epochs=$val_every_epochs" \
  --set "projector.params.report_output.validation_limit=$val_limit" \
  --set "projector.params.report_output.save_best_facts=$save_best_facts" \
  "$@"
