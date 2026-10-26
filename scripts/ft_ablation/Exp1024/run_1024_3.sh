#!/usr/bin/env bash
# GPU 2: Baseline A + R1 32x -> Baseline A + R1 64x -> Baseline A + R2 32x.
# R1=report CE + 0.1 span; R2 additionally uses 0.1 group ranking.
set -Eeuo pipefail
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}"
projector_seed="${PROJECTOR_SEED:-42}"
cosine_min_lr_ratio="${COSINE_MIN_LR_RATIO:-0.3}"
r1_ratios=(32 64)
r2_ratios=(32)

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$script_dir/_common.bash"
exp1024_init run_1024_3 "$@"
exp1024_run R1 "${r1_ratios[@]}"
exp1024_run R2 "${r2_ratios[@]}"
