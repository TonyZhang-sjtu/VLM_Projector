# Shared by the constant-LR entries in this directory, including the 4e-4 probe.
# Sourcing this file only defines a function; training starts when it is called.
run_constant_lr_baseline() {
  local const_root const_python const_local_config const_tag
  local -a const_path_args=()
  const_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"

  if [[ ! "${CUDA_VISIBLE_DEVICES:-}" =~ ^[0-9]+$ ]]; then
    echo "Set CUDA_VISIBLE_DEVICES to one physical GPU index." >&2
    return 2
  fi
  if [[ -n "${PROJECTOR_PYTHON:-}" ]]; then
    const_python="$PROJECTOR_PYTHON"
  elif [[ -n "${CONDA_PREFIX:-}" && -x "$CONDA_PREFIX/bin/python" ]]; then
    # tmux can keep base Python in PATH while CONDA_PREFIX identifies projector.
    const_python="$CONDA_PREFIX/bin/python"
  else
    const_python="$(command -v python)" || {
      echo "Activate the training environment or set PROJECTOR_PYTHON." >&2
      return 2
    }
  fi
  if [[ ! -x "$const_python" ]]; then
    echo "Training Python is not executable: $const_python" >&2
    return 2
  fi

  const_tag="${projector_run_tag:-}"
  if [[ -z "$const_tag" ]]; then
    const_tag="constlr_${projector_method}_p${projector_lr}_l${encoder_lr}_$(date -u +%Y%m%dT%H%M%S%NZ)"
  fi

  const_local_config="${PROJECTOR_LOCAL_CONFIG:-$const_root/configs/local_paths.yaml}"
  if [[ -f "$const_local_config" ]]; then
    const_path_args=(--config "$const_local_config")
  elif [[ -n "${PROJECTOR_LOCAL_CONFIG:-}" ]]; then
    echo "PROJECTOR_LOCAL_CONFIG does not exist: $const_local_config" >&2
    return 2
  fi
  export PROJECTOR_PYTHON="$const_python"
  export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 PYTHONUTF8=1 PYTHONIOENCODING=utf-8
  export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
  export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
  export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
  cd -- "$const_root"

  # Each ratio starts from pretrained assets with a fresh projector/optimizer.
  # The sweep executes train -> best.pt full test -> next ratio; errors stop it.
  # Like the original baseline scripts, command-line options override defaults.
  # --resume-run explicitly reuses its saved plan, not these fresh-run settings.
  exec "$const_python" -u "$const_root/results/BindFormer_role_value_adapter_r16/entry.py" src.cli.baseline_sweep \
    --config "$const_root/configs/training/constant_lr.yaml" \
    "${const_path_args[@]}" \
    --run-tag "$const_tag" \
    --ratios "${projector_ratios[@]}" \
    --output-root "${PROJECTOR_OUTPUT_ROOT:-$const_root/results/BindFormer_role_value_adapter_r16/full_runs}" \
    --device cuda:0 --scoring-device cuda:0 \
    --set "seed=$projector_seed" \
    --set "training.projector_lr=$projector_lr" \
    --set "training.encoder_lr=$encoder_lr" \
    --set "training.max_epochs=$projector_max_epochs" \
    --set "training.projector_warmup_steps=$projector_warmup_steps" \
    --set "training.projector_hold_steps=$projector_hold_steps" \
    --set "training.joint_warmup_steps=$joint_warmup_steps" \
    --set "training.joint_warmup_projector_mode=$joint_warmup_projector_mode" \
    --set "training.cosine_min_lr_ratio=$cosine_min_lr_ratio" \
    --set "training.early_stopping_enabled=$early_stopping_enabled" \
    --set "training.early_stopping_patience=$early_stopping_patience" \
    --set "training.early_stopping_min_delta=$early_stopping_min_delta" \
    "$@" \
    --projector "$projector_method" \
    --eval-split test --metrics all --eval-sample-fraction 1.0 \
    --require-full-test
}

run_role_value_experiment() {
  if [[ -z "${projector_run_tag:-}" ]]; then
    projector_run_tag="bindformer_${bind_variant}_rva${role_value_enabled}_rank${role_value_rank}_constlr_$(date -u +%Y%m%dT%H%M%S%NZ)"
  fi
  run_constant_lr_baseline \
    --set "training.effective_batch_size=$projector_effective_batch_size" \
    --set "training.microbatch_per_device=$projector_microbatch_per_device" \
    --set "training.lora.rank=$encoder_lora_rank" \
    --set "training.lora.alpha=$encoder_lora_alpha" \
    --set "projector.params.within_slot_sa=$bind_within_slot_sa" \
    --set "projector.params.local_reads_global=$bind_local_reads_global" \
    --set "projector.params.within_slot_sa_depth=$bind_within_slot_sa_depth" \
    --set "projector.params.local_global_depth=$bind_local_global_depth" \
    --set "projector.params.local_global_gate_init=$bind_local_global_gate_init" \
    --set "projector.params.global_fraction=$bind_global_fraction" \
    --set "projector.params.slot_iterations=$bind_slot_iterations" \
    --set "projector.params.knowledge_mode=$bind_knowledge_mode" \
    --set "projector.params.vocabulary_scope=brats_train" \
    --set "projector.params.role_query_counts.finding=1" \
    --set "projector.params.role_query_counts.location_laterality=1" \
    --set "projector.params.role_query_counts.signal=1" \
    --set "projector.params.role_query_counts.morphology_extent=1" \
    --set "projector.params.geometry_enabled=$bind_geometry_enabled" \
    --set "projector.params.feature_loss_weight=$bind_feature_loss_weight" \
    --set "projector.params.text_loss_weight=0.0" \
    --set "projector.params.tuple_loss_weight=0.0" \
    --set "projector.params.binding_loss_weight=0.0" \
    --set "projector.params.concept_loss_weight=0.0" \
    --set "projector.params.head_type=$bind_head_type" \
    --set "projector.params.router_value_normalization=$routing_value_normalization" \
    --set "projector.params.router_value_projection=$routing_value_projection" \
    --set "projector.params.fixed_spatial_regions=$routing_fixed_regions" \
    --set "projector.params.spatial_mean_aggregation=$routing_mean" \
    --set "projector.params.spatial_adaptive_aggregation=$routing_adaptive" \
    --set "projector.params.spatial_attention_temperature=$routing_attention_temperature" \
    --set "projector.params.role_value_reader=true" \
    --set "projector.params.role_value_adapter_enabled=$role_value_enabled" \
    --set "projector.params.role_value_adapter_rank=$role_value_rank" \
    --set "projector.params.role_value_adapter_alpha=$role_value_alpha" \
    "$@"
}
