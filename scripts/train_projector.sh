#!/usr/bin/env bash
set -euo pipefail
if (( $# < 3 )); then
  echo "Usage: $0 PROJECTOR {joint_lora|a|b_lora|b_full} RUN_DIR [projector-train options...]" >&2
  exit 2
fi
projector_name=$1
training_stage=$2
run_dir=$3
shift 3
case "$training_stage" in
  joint_lora) stage_config=configs/training/joint_lora.yaml ;;
  a|b_lora|b_full) stage_config="configs/training/stage_${training_stage}.yaml" ;;
  *) echo "Invalid stage: $training_stage" >&2; exit 2 ;;
esac
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
projector_python=${PROJECTOR_PYTHON:-/mnt/sda/mingjie/conda_envs/projector/bin/python}
exec "$projector_python" -m src.cli.train \
  --config "$stage_config" \
  --set "projector.name=${projector_name}" \
  --set "training.run_dir=${run_dir}" "$@"
