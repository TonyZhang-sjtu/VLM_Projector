# Shared VQA shell adapter. Default execution is train -> best.pt -> test.
# --dry-run only prints commands; --check-data checks data without GPU allocation.
run_vqa_experiment() {
  local vqa_root vqa_python vqa_key
  local -a vqa_args=()
  vqa_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
  vqa_python="${PROJECTOR_PYTHON:-python}"
  command -v "$vqa_python" >/dev/null || { echo "Python unavailable: $vqa_python" >&2; return 2; }
  export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 PYTHONUTF8=1 PYTHONIOENCODING=utf-8
  export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
  export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"
  export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
  export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
  cd -- "$vqa_root"
  vqa_args=(--projector "$projector_method" --ratios "${projector_ratios[@]}"
    --preprocessing-version "$vqa_preprocessing_version" --seed "$projector_seed"
    --invalid-sample-policy "$vqa_invalid_sample_policy"
    --concat-device "$projector_parallel_device" --eval-controls "${vqa_eval_controls[@]}"
    --bootstrap "$vqa_bootstrap")
  [[ -z "$projector_run_tag" ]] || vqa_args+=(--run-tag "$projector_run_tag")
  [[ -z "${VQA_OUTPUT_ROOT:-}" ]] || vqa_args+=(--output-root "$VQA_OUTPUT_ROOT")
  [[ -z "${VQA_RESULTS_ROOT:-}" ]] || vqa_args+=(--results-root "$VQA_RESULTS_ROOT")
  [[ -z "$vqa_manifest_dir" ]] || vqa_args+=(--set "data.manifest_dir=$vqa_manifest_dir")
  [[ -z "$vqa_cache_dir" ]] || vqa_args+=(--set "preprocessing.cache_dir=$vqa_cache_dir")
  if [[ "$vqa_evaluate_test" == true ]]; then
    vqa_args+=(--evaluate-test)
  elif [[ "$vqa_evaluate_test" == false ]]; then
    vqa_args+=(--no-evaluate-test)
  else
    echo 'vqa_evaluate_test must be true or false' >&2; return 2
  fi
  for vqa_key in projector_lr encoder_lr projector_warmup_steps projector_warmup_ratio \
      projector_hold_steps joint_warmup_steps joint_warmup_ratio joint_warmup_projector_mode \
      cosine_min_lr_ratio weight_decay grad_clip_norm early_stopping_enabled \
      early_stopping_patience early_stopping_min_delta save_every_steps; do
    vqa_args+=(--set "training.$vqa_key=${!vqa_key}")
  done
  vqa_args+=(--set "training.stage=$training_stage"
    --set "training.max_epochs=$projector_max_epochs"
    --set "training.effective_batch_size=$projector_effective_batch_size"
    --set "training.microbatch_per_device=$projector_microbatch_per_device"
    --set "training.lora.rank=$encoder_lora_rank"
    --set "training.lora.alpha=$encoder_lora_alpha"
    --set "training.lora.dropout=$encoder_lora_dropout"
    --set "generation.max_new_tokens=$vqa_max_new_tokens")
  # Do not import configs/local_paths.yaml: its report manifest/cache would override VQA.
  exec "$vqa_python" -u -m src.cli.vqa_sweep "${vqa_args[@]}" "$@"
}
