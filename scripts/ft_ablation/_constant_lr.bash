# Shared by the constant-LR entries in this directory, including the 4e-4 probe.
# Sourcing this file only defines a function; training starts when it is called.
run_constant_lr_baseline() {
  local const_root const_python const_local_config const_tag
  local -a const_path_args=()
  const_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"

  if [[ -n "${projector_parallel_device:-}" ]]; then
    if [[ "$projector_method" != concat3d || "$projector_parallel_device" != cuda:1
          || ! "${CUDA_VISIBLE_DEVICES:-}" =~ ^[0-9]+,[0-9]+$
          || "${CUDA_VISIBLE_DEVICES%%,*}" == "${CUDA_VISIBLE_DEVICES##*,}" ]]; then
      echo "Concat3D model parallelism requires two distinct physical GPU indices, e.g. CUDA_VISIBLE_DEVICES=0,1." >&2
      return 2
    fi
  elif [[ ! "${CUDA_VISIBLE_DEVICES:-}" =~ ^[0-9]+$ ]]; then
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
  exec "$const_python" -u -m src.cli.baseline_sweep \
    --config "$const_root/configs/training/constant_lr.yaml" \
    "${const_path_args[@]}" \
    --run-tag "$const_tag" \
    --ratios "${projector_ratios[@]}" \
    --output-root "${PROJECTOR_OUTPUT_ROOT:-$const_root/runs/baselines}" \
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
