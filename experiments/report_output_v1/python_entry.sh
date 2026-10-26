#!/usr/bin/env bash
# Interpreter bridge for the UNMODIFIED Baseline A shell. Only this opt-in
# launcher sets PROJECTOR_PYTHON to this bridge; subprocesses use real Python.
set -Eeuo pipefail
: "${REPORT_OUTPUT_REAL_PYTHON:?Set the real training interpreter}"
here="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [[ "${1:-}" == -u ]]; then shift; fi
if [[ "${1:-}" != -m || -z "${2:-}" ]]; then
  echo 'Report-output bridge expects Python -m module arguments' >&2
  exit 2
fi
shift
exec "$REPORT_OUTPUT_REAL_PYTHON" -u "$here/entry.py" "$@"
