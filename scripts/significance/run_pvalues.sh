#!/usr/bin/env bash
# CPU-only statistical audit and tests; separate from all experiment launchers.
set -euo pipefail
sig_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONUTF8=1 PYTHONIOENCODING=utf-8
cd "$sig_root"
exec "${SIGNIFICANCE_PYTHON:-python}" -m src.cli.significance "$@"
