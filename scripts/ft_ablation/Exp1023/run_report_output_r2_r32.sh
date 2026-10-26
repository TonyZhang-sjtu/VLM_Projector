#!/usr/bin/env bash
# R2 / 32x: independent fresh training, then original best-CE full-test scoring.
set -Eeuo pipefail
export EXPERIMENT=R2
export PROJECTOR_RATIOS=32
export SPAN_ENABLED=true
export GROUP_ENABLED=true
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$script_dir/run_bindformer_report_output_constlr.sh" "$@"
