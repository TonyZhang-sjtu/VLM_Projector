#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 GPU_IDS projector-train arguments..." >&2
  echo "Example: $0 1,3 --config configs/training/stage_a.yaml --config configs/training/deterministic.yaml --config configs/training/ddp.yaml --set training.run_dir=runs/a_ddp" >&2
  exit 2
fi

gpu_ids="$1"
shift
if [[ ! "$gpu_ids" =~ ^[0-9]+(,[0-9]+)*$ ]]; then
  echo "GPU_IDS must be a comma-separated list of physical GPU indices (for example 1,3)." >&2
  exit 2
fi
IFS=',' read -r -a gpu_list <<< "$gpu_ids"
if (( ${#gpu_list[@]} < 2 )); then
  echo "DDP needs at least two visible GPUs." >&2
  exit 2
fi
declare -A seen_gpus=()
for gpu_id in "${gpu_list[@]}"; do
  if [[ -n "${seen_gpus[$gpu_id]:-}" ]]; then
    echo "Duplicate GPU index: $gpu_id" >&2
    exit 2
  fi
  seen_gpus[$gpu_id]=1
done

export CUDA_VISIBLE_DEVICES="$gpu_ids"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
exec torchrun --standalone --nproc_per_node="${#gpu_list[@]}" -m src.cli.train "$@"
