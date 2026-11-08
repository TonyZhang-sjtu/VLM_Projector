# Thin adapters around the existing task launchers. Never source production
# experiment entrypoints: their GPU/method/ratio defaults are independent.
run_anchorread_ablation() {
  local ablation_root ablation_python ablation_tag ablation_recipe ablation_local
  local -a ablation_paths=()
  ablation_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
  ablation_python="${PROJECTOR_PYTHON:-/mnt/sda/mingjie/conda_envs/projector/bin/python}"
  command -v "$ablation_python" >/dev/null || { echo "Python unavailable: $ablation_python" >&2; return 2; }
  if [[ ! "${CUDA_VISIBLE_DEVICES:-}" =~ ^[0-9]+$ ]]; then
    echo 'Set CUDA_VISIBLE_DEVICES to one physical GPU index.' >&2
    return 2
  fi
  ablation_tag="${PROJECTOR_RUN_TAG:-${ablation_method}_s42_$(date -u +%Y%m%dT%H%M%S%NZ)}"
  export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 PYTHONUTF8=1 PYTHONIOENCODING=utf-8
  export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
  export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
  export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
  export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"
  cd -- "$ablation_root"
  if [[ "$ablation_task" == rg ]]; then
    ablation_recipe="$ablation_root/configs/experiments/anchorread3d_ablation/training.yaml"
    ablation_local="${PROJECTOR_LOCAL_CONFIG:-$ablation_root/configs/local_paths.yaml}"
    if [[ -f "$ablation_local" ]]; then
      ablation_paths=(--config "$ablation_local")
    elif [[ -n "${PROJECTOR_LOCAL_CONFIG:-}" ]]; then
      echo "PROJECTOR_LOCAL_CONFIG does not exist: $ablation_local" >&2
      return 2
    fi
    exec "$ablation_python" -u -m src.cli.baseline_sweep \
      "${ablation_paths[@]}" --config "$ablation_recipe" \
      --run-tag "$ablation_tag" --ratios 32 256 \
      --output-root "${ANCHORREAD_ABLATION_RUNS_ROOT:-$ablation_root/runs/ablations/anchorread3d}/rg" \
      --device cuda:0 --scoring-device cuda:0 \
      "$@" --projector "$ablation_method" \
      --eval-split test --metrics all --eval-sample-fraction 1.0 --require-full-test
  elif [[ "$ablation_task" == vqa ]]; then
    ablation_recipe="$ablation_root/configs/experiments/anchorread3d_ablation/vqa_training.yaml"
    # The RG local overlay contains report manifests/cache. Only an explicitly
    # supplied VQA overlay is used here; the standard v2 preset supplies data.
    if [[ -n "${VQA_LOCAL_CONFIG:-}" ]]; then
      [[ -f "$VQA_LOCAL_CONFIG" ]] || { echo "Missing VQA_LOCAL_CONFIG: $VQA_LOCAL_CONFIG" >&2; return 2; }
      ablation_paths=(--config "$VQA_LOCAL_CONFIG")
    fi
    exec "$ablation_python" -u -m src.cli.vqa_sweep \
      "${ablation_paths[@]}" --config "$ablation_recipe" \
      --run-tag "$ablation_tag" --ratios 32 256 --seed 42 \
      --preprocessing-version v2 --invalid-sample-policy include_unreviewed \
      --output-root "${ANCHORREAD_ABLATION_RUNS_ROOT:-$ablation_root/runs/ablations/anchorread3d}/vqa" \
      --results-root "${ANCHORREAD_ABLATION_RESULTS_ROOT:-$ablation_root/results/ablations/anchorread3d}/vqa_raw" \
      --evaluate-test --eval-split test --eval-controls normal --bootstrap 1000 \
      "$@" --projector "$ablation_method"
  else
    echo "Unknown ablation task: $ablation_task" >&2
    return 2
  fi
}
