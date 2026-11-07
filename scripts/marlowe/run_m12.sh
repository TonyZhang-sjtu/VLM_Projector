#!/usr/bin/env bash
# ConvPool: VQA 32/16/8 + RG 64; refill one free slot with RG 32.
set -Eeuo pipefail
cd "$(dirname -- "${BASH_SOURCE[0]}")/../.."
source scripts/marlowe/env.sh
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export VQA_PREPROCESSING_VERSION=v2 VQA_INVALID_SAMPLE_POLICY=include_unreviewed
exec "$PROJECTOR_PYTHON" -u scripts/marlowe/convpool_wave.py m12 "$@"
