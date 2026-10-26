#!/usr/bin/env bash
# GPU 1: Baseline A + role value adapter 32x -> 64x -> Baseline A + E3 64x.
# Adapter rank/alpha=16/16, fixed-region MEAN aggregation, matching prior RVA.
set -Eeuo pipefail
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
projector_seed="${PROJECTOR_SEED:-42}"
cosine_min_lr_ratio="${COSINE_MIN_LR_RATIO:-0.3}"
role_value_ratios=(32 64)
e3_ratios=(64)

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$script_dir/_common.bash"
exp1024_init run_1024_2 "$@"
exp1024_run role_value_adapter "${role_value_ratios[@]}"
exp1024_run E3 "${e3_ratios[@]}"
