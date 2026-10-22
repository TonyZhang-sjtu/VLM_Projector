#!/usr/bin/env bash
# perceiver_flamingo: independent joint_lora runs, each followed by best-checkpoint evaluation.
set -Eeuo pipefail

# ===== 本次实验设置；命令行 --ratios / --run-tag 可覆盖 =====
projector_ratios=(4 8 16 32)  # 支持 (64)、(32 64) 或原有倍率；也可用 --ratios 覆盖。
projector_run_tag=""    # 留空自动生成时间戳；手动指定时每次使用新名称

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
  --projector perceiver_flamingo
