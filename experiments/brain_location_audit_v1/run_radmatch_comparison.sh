#!/usr/bin/env bash
# Official, pinned implementation. Requires an available, explicitly configured judge.
set -euo pipefail
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$project_root"
python_bin="${RADMATCH_PYTHON:-${project_root}/experiments/brain_location_audit_v1/runtime/venv/bin/python}"
bundle="${RADMATCH_INPUT_BUNDLE:-${project_root}/results/radmatch_comparison_1108/pilot_inputs}"
output="${RADMATCH_OUTPUT_DIR:-${project_root}/results/radmatch_comparison_1108/pilot_results}"
judge="${RADMATCH_JUDGE:-}"
mode="${1:---dry-run}"
if [[ "$mode" != run && "$mode" != --dry-run && "$mode" != stress ]]; then
  printf '%s\n' 'Usage: run_radmatch_comparison.sh [--dry-run|stress|run]' >&2
  exit 2
fi
if [[ "$mode" != --dry-run ]]; then
  : "${RADMATCH_JUDGE:?Set local:MODEL for an available authorized judge}"
  : "${RADMATCH_LOCAL_BASE_URL:?Set its OpenAI-compatible URL ending in /v1}"
else
  judge="${judge:-local:CONFIGURE_JUDGE_MODEL}"
fi
cmd=("$python_bin" -m experiments.brain_location_audit_v1.radmatch_compare run
  --bundle "$bundle" --output "$output" --judge "$judge"
  --workers "${RADMATCH_WORKERS:-1}" --reasoning "${RADMATCH_REASONING:-none}")
if [[ "$mode" == stress ]]; then cmd+=(--only stress); fi
if [[ "$mode" == --dry-run ]]; then
  printf '%q ' "${cmd[@]}"
  printf '\n'
else
  exec "${cmd[@]}"
fi
