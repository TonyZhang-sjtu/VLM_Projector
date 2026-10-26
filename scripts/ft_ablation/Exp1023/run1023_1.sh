#!/usr/bin/env bash
# GPU 0 / tmux adap: R1, 32x. --dry-run resolves without training.
set -Eeuo pipefail
export CUDA_VISIBLE_DEVICES=0
export PROJECTOR_RUN_TAG="${PROJECTOR_RUN_TAG:-report_output_v1_R1_r32_gpu0_$(date -u +%Y%m%dT%H%M%S%NZ)}"
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$script_dir/run_report_output_r1_r32.sh" "$@"
