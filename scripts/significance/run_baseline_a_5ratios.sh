#!/usr/bin/env bash
# Baseline A vs six baselines, 16/32/64/128/256x. MLP is excluded.
# Default: strict identity audit -> source-profile CE remeasurement -> p-values.
# Historical and recomputed CE are reported separately. --ce-policy strict
# additionally requires historical floating-point agreement.
# Use --audit-only to check transferred artifacts without inference or testing.
set -euo pipefail
sig_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONUTF8=1 PYTHONIOENCODING=utf-8 TOKENIZERS_PARALLELISM=false
cd "$sig_root"
exec "${SIGNIFICANCE_PYTHON:-python}" -u -m src.cli.significance_suite --ce-policy recompute "$@"
