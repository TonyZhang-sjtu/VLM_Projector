#!/usr/bin/env bash
# Four RG experiments: ConvPool 8/16 and ADirect+LocalRead 8/16.
set -Eeuo pipefail
cd "$(dirname -- "${BASH_SOURCE[0]}")/../.."
source scripts/marlowe/env.sh
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export VQA_PREPROCESSING_VERSION=v2 VQA_INVALID_SAMPLE_POLICY=include_unreviewed
exec "$PROJECTOR_PYTHON" -u scripts/marlowe/convpool_wave.py m10 "$@"
