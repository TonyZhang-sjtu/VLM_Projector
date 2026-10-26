# Exp1024 queue assembly only. Existing launchers and training code are unchanged.
# Each child sweep executes train -> best-CE full test -> next ratio, from scratch.

exp1024_init() {
  exp1024_queue_name="$1"
  shift
  exp1024_args=("$@")
  exp1024_dry_run=false
  local arg
  for arg in "$@"; do
    case "$arg" in
      --dry-run) exp1024_dry_run=true ;;
      --resume-run*|--ratios*|--run-tag*|--output-root*|--projector*|--device*|--scoring-device*)
        echo "Queue option is fixed: $arg. Edit the ratio arrays / CUDA_VISIBLE_DEVICES / PROJECTOR_RUN_TAG / PROJECTOR_OUTPUT_ROOT, or use the original standard launcher for resume." >&2
        return 2 ;;
    esac
  done
  [[ "$CUDA_VISIBLE_DEVICES" =~ ^[0-9]+$ ]] || {
    echo 'Select one physical GPU with CUDA_VISIBLE_DEVICES.' >&2; return 2;
  }
  [[ "$projector_seed" =~ ^[0-9]+$ ]] || {
    echo 'PROJECTOR_SEED must be a nonnegative integer.' >&2; return 2;
  }
  exp1024_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
  # Pin the environment used by the preceding experiments; tmux's base Python
  # may lack training dependencies. An explicit PROJECTOR_PYTHON still wins.
  export PROJECTOR_PYTHON="${PROJECTOR_PYTHON:-/mnt/sda/mingjie/conda_envs/projector/bin/python}"
  [[ -x "$PROJECTOR_PYTHON" ]] || {
    echo "Training Python is not executable: $PROJECTOR_PYTHON" >&2; return 2;
  }
  "$PROJECTOR_PYTHON" -c 'import math,sys; v=float(sys.argv[1]); assert math.isfinite(v) and 0 <= v <= 1, "COSINE_MIN_LR_RATIO must be in [0,1]"' "$cosine_min_lr_ratio"
  export PROJECTOR_OUTPUT_ROOT="${PROJECTOR_OUTPUT_ROOT:-$exp1024_root/runs/Exp1024}"
  exp1024_queue_tag="${PROJECTOR_RUN_TAG:-exp1024_${exp1024_queue_name}_cos${cosine_min_lr_ratio}_seed${projector_seed}_gpu${CUDA_VISIBLE_DEVICES}_$(date -u +%Y%m%dT%H%M%S%NZ)}"
  cd -- "$exp1024_root"
  printf '[%s] GPU=%s seed=%s cosine_min_lr_ratio=%s\n' \
    "$exp1024_queue_name" "$CUDA_VISIBLE_DEVICES" "$projector_seed" "$cosine_min_lr_ratio" >&2
}

# Subshell isolates EXPERIMENT, loss switches and interpreter bridges between jobs.
exp1024_run() (
  set -Eeuo pipefail
  method="$1"
  shift
  ratios=("$@")
  [[ "${#ratios[@]}" -gt 0 ]] || { echo 'No ratios supplied.' >&2; exit 2; }
  ratio_tag="$(IFS=-; echo "${ratios[*]}")"
  export PROJECTOR_RUN_TAG="${exp1024_queue_tag}_${method}_r${ratio_tag}"
  export PROJECTOR_RATIOS="${ratios[*]}"

  # Pin the previous Baseline A architecture, independent of unrelated shell env.
  export BIND_ROUTER_VALUE_NORMALIZATION=true BIND_ROUTER_VALUE_PROJECTION=false
  export BIND_FIXED_SPATIAL_REGIONS=true BIND_SPATIAL_MEAN_AGGREGATION=true
  export BIND_SPATIAL_ADAPTIVE_AGGREGATION=false BIND_SPATIAL_TEMPERATURE=1.0
  export AUX_WARMUP_STEPS=77
  case "$method" in
    baseline_a)
      launcher="$exp1024_root/scripts/ft_ablation/Exp1020/run_bindformer_va_fixed_region_mean_constlr.sh"
      ;;
    role_value_adapter)
      export BIND_ROLE_VALUE_ADAPTER=true BIND_ROLE_VALUE_RANK=16 BIND_ROLE_VALUE_ALPHA=16
      launcher="$exp1024_root/scripts/ft_ablation/Exp1020/run_bindformer_va_fixed_region_mean_role_value_adapter_constlr.sh"
      ;;
    R1|R2)
      export EXPERIMENT="$method" SPAN_ENABLED=true GROUP_ENABLED=false
      if [[ "$method" == R2 ]]; then export GROUP_ENABLED=true; fi
      export SPAN_WEIGHT=0.1 GROUP_WEIGHT=0.1 GROUP_MARGIN=0.2 GROUP_EVERY_STEPS=1
      export REPORT_OUTPUT_PATH=assets/knowledge/report_output_v1
      export REPORT_OUTPUT_FIELDS='[edema,laterality,site,signal]'
      export VAL_FACTS_ENABLED=true VAL_FACTS_EVERY_EPOCHS=1 VAL_FACTS_LIMIT=0 SAVE_BEST_FACTS=true
      launcher="$exp1024_root/scripts/ft_ablation/Exp1023/run_bindformer_report_output_constlr.sh"
      ;;
    E2|E3)
      export EXPERIMENT="$method" FACT_ENABLED=false PAIR_ENABLED=true
      if [[ "$method" == E3 ]]; then export FACT_ENABLED=true; fi
      export FACT_WEIGHT=0.1 PAIR_WEIGHT=0.1 PAIR_POSITIVE_CE_WEIGHT=0.25
      export PAIR_MARGIN=0.2 PAIR_EVERY_STEPS=1
      export REPORT_FACTS_PATH=assets/knowledge/report_facts_v1
      export REPORT_FACT_FIELDS='[edema,laterality,site,signal]'
      launcher="$exp1024_root/scripts/ft_ablation/Exp1020/run_bindformer_report_facts_constlr.sh"
      ;;
    *) echo "Unknown Exp1024 method: $method" >&2; exit 2 ;;
  esac

  # Appended overrides replace the old launchers' seed=42 / cosine floor=1.0.
  # Every other setting, original sidecar, best-CE selection and full test remain.
  command=(bash "$launcher" --ratios "${ratios[@]}" "${exp1024_args[@]}"
    --set "seed=$projector_seed"
    --set "training.cosine_min_lr_ratio=$cosine_min_lr_ratio")
  printf '[%s] %s ratios=(%s); independent initialization; train + full test/all\n' \
    "$exp1024_queue_name" "$method" "${ratios[*]}" >&2
  if [[ "$exp1024_dry_run" == true ]]; then
    "${command[@]}"
  else
    log_dir="$PROJECTOR_OUTPUT_ROOT/queue_logs"
    mkdir -p -- "$log_dir"
    printf 'Log: %s/%s.log\n' "$log_dir" "$PROJECTOR_RUN_TAG" >&2
    "${command[@]}" 2>&1 | tee "$log_dir/$PROJECTOR_RUN_TAG.log"
  fi
)
