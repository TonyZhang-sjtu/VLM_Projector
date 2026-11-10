#!/usr/bin/env bash
# Requires a separate Python >=3.11 environment with radmatch==0.2.2 installed,
# and an already running local judge endpoint. Does not start a GPU service.
set -euo pipefail
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
input_root="${LOCATION_AUDIT_DIR:-${project_root}/results/brain_location_audit_1108_v1}"
radmatch_bin="${RADMATCH_BIN:-radmatch}"
judge="${RADMATCH_JUDGE:-local:google/gemma-4-31B-it}"
workers="${RADMATCH_WORKERS:-1}"
mode="${1:-run}"
if [[ "$mode" != run && "$mode" != --dry-run ]]; then
  printf '%s\n' 'Usage: run_radmatch_pilot.sh [--dry-run]' >&2
  exit 2
fi
if [[ "$mode" != --dry-run ]]; then
  command -v "$radmatch_bin" >/dev/null
  : "${RADMATCH_LOCAL_BASE_URL:?Set the URL of the existing local judge endpoint}"
  if [[ "$judge" != local:* ]]; then
    printf '%s\n' 'This launcher is configured for local inference only.' >&2
    exit 2
  fi
fi
[[ -f "${input_root}/manifest.json" ]] || { printf '%s\n' 'Prepare the complete audit package first.' >&2; exit 1; }
run_pair() {
  local gt_dir="$1" pred_dir="$2" out_dir="$3"
  local -a cmd=("$radmatch_bin" run_all --reports-gt "$gt_dir" --reports-pred "$pred_dir"
    --output-dir "$out_dir" --llm-extractor "$judge" --llm-judge "$judge"
    --workers "$workers" --fewshot brain-mr)
  if [[ "$mode" == --dry-run ]]; then
    printf '%q ' "${cmd[@]}"
    printf '\n'
  else
    "${cmd[@]}"
  fi
}
run_pair "${input_root}/radmatch_pilot/stress/reports_gt" \
  "${input_root}/radmatch_pilot/stress/reports_pred" "${input_root}/radmatch_results/stress"
for pred_dir in "${input_root}/radmatch_pilot/reports_pred/"*; do
  [[ -d "$pred_dir" ]] || continue
  run_pair "${input_root}/radmatch_pilot/reports_gt" "$pred_dir" \
    "${input_root}/radmatch_results/$(basename -- "$pred_dir")"
done
