#!/usr/bin/env bash
# R1 / 64x: independent fresh training, then original best-CE full-test scoring.
set -Eeuo pipefail
export EXPERIMENT=R1
export PROJECTOR_RATIOS=64
export SPAN_ENABLED=true
export GROUP_ENABLED=false
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$script_dir/run_bindformer_report_output_constlr.sh" "$@"
