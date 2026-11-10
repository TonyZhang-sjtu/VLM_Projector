#!/usr/bin/env bash
# M18: four fresh RG16 ADirect+LocalRead runs, seed42, deterministic=false.
set -Eeuo pipefail
cd "$(dirname -- "${BASH_SOURCE[0]}")/../.."
source scripts/marlowe/env.sh
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  : "${CUDA_VISIBLE_DEVICES:?Slurm must allocate one GPU}"
else
  # CPU dry-run only; the Python launcher rejects training outside Slurm.
  export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
fi
exec "$PROJECTOR_PYTHON" -u scripts/marlowe/rg_repeat_wave.py "$@"
