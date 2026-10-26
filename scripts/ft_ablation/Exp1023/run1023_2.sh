#!/usr/bin/env bash
# GPU 1 / tmux mean: R2, 32x. --dry-run resolves without training.
set -Eeuo pipefail
export CUDA_VISIBLE_DEVICES=1
export PROJECTOR_RUN_TAG="${PROJECTOR_RUN_TAG:-report_output_v1_R2_r32_gpu1_$(date -u +%Y%m%dT%H%M%S%NZ)}"
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$script_dir/run_report_output_r2_r32.sh" "$@"
