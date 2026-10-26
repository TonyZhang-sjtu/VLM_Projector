#!/usr/bin/env bash
# GPU 2 / tmux value_ada: R1, 64x. --dry-run resolves without training.
set -Eeuo pipefail
export CUDA_VISIBLE_DEVICES=2
export PROJECTOR_RUN_TAG="${PROJECTOR_RUN_TAG:-report_output_v1_R1_r64_gpu2_$(date -u +%Y%m%dT%H%M%S%NZ)}"
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$script_dir/run_report_output_r1_r64.sh" "$@"
