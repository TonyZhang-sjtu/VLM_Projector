#!/usr/bin/env bash
# Run all three ADirect variants at r128 first, then repeat them at r256.
# Each child completes training and full best.pt test evaluation before returning.
# Usage: CUDA_VISIBLE_DEVICES=3 bash run01.sh [--dry-run]
set -Eeuo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-3}"
batch_tag="${PROJECTOR_RUN_TAG:-run01_$(date -u +%Y%m%dT%H%M%S%NZ)}"
extra_args=()

case "${1:-}" in
  --dry-run)
    extra_args=(--dry-run)
    shift
    ;;
  -h|--help)
    echo "Usage: bash $0 [--dry-run]"
    echo "Order: LocalRead -> Detail -> all-local at 128x, then the same order at 256x."
    exit 0
    ;;
esac
if (( $# != 0 )); then
  echo "Usage: bash $0 [--dry-run]" >&2
  exit 2
fi

for ratio in 128 256; do
  for variant in localread detail alllocal; do
    run_tag="${batch_tag}_${variant}_r${ratio}"
    echo "[run01] Starting ADirect_${variant}, ${ratio}x: ${run_tag}"
    bash "$script_dir/run_baseline_adirect_${variant}.sh" \
      --ratios "$ratio" \
      --run-tag "$run_tag" \
      "${extra_args[@]}"
  done
done

echo "[run01] All six jobs completed."
