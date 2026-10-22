# Shared implementation sourced by e1.sh-e6.sh and n1.sh-n6.sh.
# The baseline sweep runs training first and full test evaluation after success.
run_ft_ablation() {
  local ft_root ft_python ft_local_config
  local -a ft_path_args=()
  local -a ft_recipe_args=()
  ft_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
  cd -- "$ft_root"

  if [[ ! "$CUDA_VISIBLE_DEVICES" =~ ^[0-9]+$ ]]; then
    echo "Set CUDA_VISIBLE_DEVICES at the top of this experiment to one physical GPU index." >&2
    return 2
  fi
  if [[ -n "${PROJECTOR_PYTHON:-}" ]]; then
    ft_python="$PROJECTOR_PYTHON"
  elif [[ -n "${CONDA_PREFIX:-}" && -x "$CONDA_PREFIX/bin/python" ]]; then
    # A tmux shell can retain a base-Python PATH despite an activated Conda prefix.
    ft_python="$CONDA_PREFIX/bin/python"
  else
    ft_python="$(command -v python)" || {
      echo "Activate the training environment or set PROJECTOR_PYTHON to its Python executable." >&2
      return 2
    }
  fi
  if [[ ! -x "$ft_python" ]]; then
    echo "PROJECTOR_PYTHON is not executable: $ft_python" >&2
    return 2
  fi
  export PROJECTOR_PYTHON="$ft_python"
  export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 PYTHONUTF8=1 PYTHONIOENCODING=utf-8
  export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
  export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
  export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"

  ft_local_config="${PROJECTOR_LOCAL_CONFIG:-$ft_root/configs/local_paths.yaml}"
  if [[ -f "$ft_local_config" ]]; then
    ft_path_args=(--config "$ft_local_config")
  elif [[ -n "${PROJECTOR_LOCAL_CONFIG:-}" ]]; then
    echo "PROJECTOR_LOCAL_CONFIG does not exist: $ft_local_config" >&2
    return 2
  fi

  if [[ -n "${ft_cosine_min_lr_ratio:-}" ]]; then
    ft_recipe_args=(
      --set "training.cosine_min_lr_ratio=$ft_cosine_min_lr_ratio"
      --set "training.early_stopping_enabled=$ft_early_stopping_enabled"
      --set "training.checkpoint_epochs=$ft_checkpoint_epochs"
      --set "training.log_optimizer_diagnostics=true"
      --eval-checkpoint-epochs "${ft_eval_checkpoint_epochs[@]}"
    )
  fi

  # Script parameters are visible in each eN.sh; --set can override them for a
  # deliberately renamed experiment. Final evaluation is always full test/all.
  # --resume-run reuses a saved plan; the full-test guard also checks that plan.
  exec "$ft_python" -u -m "${ft_cli_module:-src.cli.baseline_sweep}" \
    "${ft_path_args[@]}" \
    --config "$ft_root/configs/ft_ablation/avgpool3d_r8.yaml" \
    --run-tag "$ft_run_tag" \
    --output-root "${PROJECTOR_OUTPUT_ROOT:-$ft_root/runs/baselines}" \
    --device cuda:0 --scoring-device cuda:0 \
    --set "training.projector_lr=$ft_projector_lr" \
    --set "training.encoder_lr=$ft_encoder_lr" \
    --set "training.projector_warmup_steps=$ft_projector_warmup_steps" \
    --set "training.projector_hold_steps=$ft_projector_hold_steps" \
    --set "training.joint_warmup_steps=$ft_joint_warmup_steps" \
    --set "training.max_epochs=$ft_max_epochs" \
    "${ft_recipe_args[@]}" \
    "$@" \
    --projector avgpool3d --ratios 8 \
    --eval-split test --metrics all --eval-sample-fraction 1.0 \
    --require-full-test
}
