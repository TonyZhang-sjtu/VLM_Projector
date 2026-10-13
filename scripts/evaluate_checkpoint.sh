#!/usr/bin/env bash
# One checkpoint -> MRI report generation -> selected report metrics.
set -euo pipefail
projector_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
projector_python="${PROJECTOR_PYTHON:-/mnt/sda/mingjie/conda_envs/projector/bin/python}"
export PYTHONNOUSERSITE=1
export PYTHONUTF8=1
export PYTHONIOENCODING=utf-8
cd "$projector_root"
exec "$projector_python" -m src.cli.evaluate_checkpoint "$@"
