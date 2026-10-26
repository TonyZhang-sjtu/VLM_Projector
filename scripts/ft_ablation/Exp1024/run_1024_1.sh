#!/usr/bin/env bash
# GPU 0: Baseline A 32x -> Baseline A 64x -> Baseline A + R2 64x.
# Each experiment: fresh training -> best-CE checkpoint -> full test/all metrics.
set -Eeuo pipefail
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
projector_seed="${PROJECTOR_SEED:-42}"
cosine_min_lr_ratio="${COSINE_MIN_LR_RATIO:-0.3}"
baseline_ratios=(32 64)
r2_ratios=(64)

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$script_dir/_common.bash"
exp1024_init run_1024_1 "$@"
exp1024_run baseline_a "${baseline_ratios[@]}"
exp1024_run R2 "${r2_ratios[@]}"
