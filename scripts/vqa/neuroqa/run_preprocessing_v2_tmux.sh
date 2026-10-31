#!/usr/bin/env bash
# Run the frozen v2 pipeline in the foreground and mirror output to the log.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
echo "Starting v2 preprocessing in the foreground; Ctrl+C stops this run."
echo "Completed cases are verified and reused when resumed."
"${PROJECTOR_PYTHON:-/mnt/sda/mingjie/conda_envs/projector/bin/python}" -u \
  scripts/vqa/neuroqa/continue_v2_pipeline.py --resume-frozen --workers 16 "$@" \
  2>&1 | tee -a results/vqa/neuroqa/preprocessing_v2_plan/coordinator.log
