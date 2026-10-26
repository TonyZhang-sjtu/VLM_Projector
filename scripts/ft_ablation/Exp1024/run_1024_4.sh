#!/usr/bin/env bash
# GPU 3: Baseline A + E2 32x -> Baseline A + E2 64x -> Baseline A + E3 32x.
# Original report_facts_v1 labels and losses; enhancement/margin stay disabled.
set -Eeuo pipefail
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-3}"
projector_seed="${PROJECTOR_SEED:-42}"
cosine_min_lr_ratio="${COSINE_MIN_LR_RATIO:-0.3}"
e2_ratios=(32 64)
e3_ratios=(32)

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$script_dir/_common.bash"
exp1024_init run_1024_4 "$@"
exp1024_run E2 "${e2_ratios[@]}"
exp1024_run E3 "${e3_ratios[@]}"
