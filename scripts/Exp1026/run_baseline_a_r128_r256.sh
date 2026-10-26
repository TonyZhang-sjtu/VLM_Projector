#!/usr/bin/env bash
# Baseline A (vanilla BindFormer + fixed regions + mean aggregation), 128x then 256x.
# Reuse the established 1021 constant-LR recipe and full-test evaluation.
set -Eeuo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PROJECTOR_PYTHON="${PROJECTOR_PYTHON:-/mnt/sda/mingjie/conda_envs/projector/bin/python}"
projector_ratios=(128 256)

# Pin Baseline A routing even if the calling shell exported BIND_* switches.
export BIND_ROUTER_VALUE_NORMALIZATION=true BIND_ROUTER_VALUE_PROJECTION=false
export BIND_FIXED_SPATIAL_REGIONS=true BIND_SPATIAL_MEAN_AGGREGATION=true
export BIND_SPATIAL_ADAPTIVE_AGGREGATION=false BIND_SPATIAL_TEMPERATURE=1.0

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$script_dir/../ft_ablation/Exp1020/run_bindformer_va_fixed_region_mean_constlr.sh" \
  --ratios "${projector_ratios[@]}" "$@"
