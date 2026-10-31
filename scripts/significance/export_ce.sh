#!/usr/bin/env bash
# One explicitly selected checkpoint, CE only, numerator/denominator per sample.
set -euo pipefail
sig_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
export PYTHONNOUSERSITE=1 PYTHONUTF8=1 PYTHONIOENCODING=utf-8
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=4
export CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$sig_root"
exec "${PROJECTOR_PYTHON:-/mnt/sda/mingjie/conda_envs/projector/bin/python}" -m src.cli.export_answer_ce "$@"
