#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/mnt/sda/mingjie/conda_envs/projector/bin/python}"
cd "$PROJECT_DIR"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
"$PYTHON_BIN" -m experiments.anchorread3d_final_eval_1108.green_gate
"$PYTHON_BIN" -m experiments.anchorread3d_final_eval_1108.queue --phase jobs --name green \
  --jobs results/anchorread3d_final_eval_1108/visual/scoring/green_jobs.json
"$PYTHON_BIN" -m experiments.anchorread3d_final_eval_1108.score_visual verify-green
"$PYTHON_BIN" -m experiments.anchorread3d_final_eval_1108.export_visual
