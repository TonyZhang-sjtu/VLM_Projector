#!/usr/bin/env bash
# RG a4_local_only: independent r32/r256 training -> validation best -> full test.
set -Eeuo pipefail
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
ablation_task=rg
ablation_method=anchorread3d_a4_local_only
ablation_script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$ablation_script_dir/_common.bash"
run_anchorread_ablation "$@"
