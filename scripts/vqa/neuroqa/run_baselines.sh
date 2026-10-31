#!/usr/bin/env bash
set -Eeuo pipefail
# GPU assignment is explicit. Override PROJECTOR_PYTHON when using another environment.
: "${CUDA_VISIBLE_DEVICES:?Set CUDA_VISIBLE_DEVICES to your assigned GPU, e.g. 0}"
projector_python="${PROJECTOR_PYTHON:-python}"
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
for method in avgpool3d concat3d convpool3d mixer3d local_packer3d attention_pool perceiver_nv perceiver_flamingo adaptive_merge3d token_select3d; do
  for ratio in 16 32 64 128 256; do
    "$projector_python" "$script_dir/run_experiment.py" --projector "$method" --ratio "$ratio" "$@"
  done
done
