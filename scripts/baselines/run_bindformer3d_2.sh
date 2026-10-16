#!/usr/bin/env bash
# BindFormer joint-LoRA baseline sweep; default ratio 4 on GPU 1.
set -Eeuo pipefail
projector_ratios=(4)
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
projector_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
projector_python="${PROJECTOR_PYTHON:-/mnt/sda/mingjie/conda_envs/projector/bin/python}"
export PYTHONNOUSERSITE=1 PYTHONUTF8=1 PYTHONIOENCODING=utf-8
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
cd "$projector_root"
exec "$projector_python" -u -m src.cli.bindformer_sweep --ratios "${projector_ratios[@]}" "$@"
