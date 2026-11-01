#!/usr/bin/env bash
# Registered-projector entrypoint; uses the matched A4 report-generation recipe.
set -Eeuo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$script_dir/../Exp1026/run_baseline_a4_r128_r256.sh" "$@"
