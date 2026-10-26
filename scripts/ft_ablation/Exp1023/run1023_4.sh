#!/usr/bin/env bash
# GPU 3 / tmux value_mean: R2, 64x. --dry-run resolves without training.
set -Eeuo pipefail
export CUDA_VISIBLE_DEVICES=3
export PROJECTOR_RUN_TAG="${PROJECTOR_RUN_TAG:-report_output_v1_R2_r64_gpu3_$(date -u +%Y%m%dT%H%M%S%NZ)}"
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$script_dir/run_report_output_r2_r64.sh" "$@"
