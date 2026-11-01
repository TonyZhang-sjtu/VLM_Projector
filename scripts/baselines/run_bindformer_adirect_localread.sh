#!/usr/bin/env bash
# Registered-projector entrypoint; use the same Exp1026 report-generation recipe.
set -Eeuo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$script_dir/../Exp1026/run_baseline_adirect_localread.sh" "$@"
