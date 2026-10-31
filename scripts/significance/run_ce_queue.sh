#!/usr/bin/env bash
# Wait for >= 20000 MiB free GPU memory; never stops another experiment.
set -euo pipefail
sig_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONUTF8=1 PYTHONIOENCODING=utf-8
cd "$sig_root"
exec "${SIGNIFICANCE_PYTHON:-python}" -m src.cli.significance_queue "$@"
