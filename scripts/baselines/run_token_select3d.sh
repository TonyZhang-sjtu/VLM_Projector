#!/usr/bin/env bash
# token_select3d: independent joint_lora runs, each followed by best-checkpoint evaluation.
set -Eeuo pipefail

# ===== 本次实验设置 =====
projector_ratios=(8 16)  # 支持 (64)、(32 64) 或原有倍率；也可用 --ratios 覆盖。
projector_run_tag="token_select_r8_r16"      # 留空：自动生成时间戳名称；也可填 token_select_r8_r16

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-3}"
projector_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
projector_python="${PROJECTOR_PYTHON:-/mnt/sda/mingjie/conda_envs/projector/bin/python}"
export PYTHONNOUSERSITE=1 PYTHONUTF8=1 PYTHONIOENCODING=utf-8
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
cd -- "$projector_root"

projector_args=(--ratios "${projector_ratios[@]}")

if [[ -n "$projector_run_tag" ]]; then
  projector_args+=(--run-tag "$projector_run_tag")
fi

exec "$projector_python" -m src.cli.baseline_sweep \
  "${projector_args[@]}" \
  "$@" \
  --projector token_select3d
