#!/usr/bin/env bash
# Two fresh RG r16 repeats and two fresh VQA r64 repeats on one allocated H100.
set -Eeuo pipefail
cd "$(dirname -- "${BASH_SOURCE[0]}")/../.."
source scripts/marlowe/env.sh
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export VQA_PREPROCESSING_VERSION=v2 VQA_INVALID_SAMPLE_POLICY=include_unreviewed
exec "$PROJECTOR_PYTHON" -u scripts/marlowe/repeat_wave.py "$@"
