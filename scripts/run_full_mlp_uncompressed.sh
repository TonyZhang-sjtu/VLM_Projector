#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT="/mnt/sda/jyzhang/VLM_Projector"
RUN_TAG="brats_mlp_joint_lora_$(date +%Y%m%dT%H%M%S)"
if (( $# > 0 )) && [[ "$1" != --* ]]; then
  RUN_TAG="$1"
  shift
fi
DATASETS=()
CACHE_ARGS=()
CACHE_SETTING=enabled
FORWARDED=()
DRY_RUN=false
while (( $# > 0 )); do
  case "$1" in
    --datasets)
      shift
      before=${#DATASETS[@]}
      while (( $# > 0 )) && [[ "$1" != --* ]]; do
        DATASETS+=("$1")
        shift
      done
      if (( ${#DATASETS[@]} == before )); then
        echo "--datasets requires brats, isles, wmh, all, or a combination" >&2
        exit 2
      fi
      ;;
    --config|--set)
      if (( $# < 2 )); then echo "$1 requires a value" >&2; exit 2; fi
      if [[ "$1" == --set && "$2" == training.run_dir=* ]]; then
        echo "Use RUN_TAG to select this launcher output directory" >&2; exit 2
      fi
      FORWARDED+=("$1" "$2")
      shift 2
      ;;
    --dry-run)
      FORWARDED+=(--dry-run)
      DRY_RUN=true
      shift
      ;;
    --use-cache)
      CACHE_ARGS=(--use-cache)
      CACHE_SETTING=enabled
      shift
      ;;
    --no-use-cache|--no-cache)
      CACHE_ARGS=(--no-use-cache)
      CACHE_SETTING=disabled
      shift
      ;;
    brats|isles|wmh|all)
      DATASETS+=("$1")
      shift
      ;;
    *)
      echo "Unexpected argument: $1" >&2
      echo "Usage: $0 [run-tag] [--datasets brats isles wmh|all] [--use-cache|--no-use-cache] [--config YAML] [--set KEY=VALUE] [--dry-run]" >&2
      exit 2
      ;;
  esac
done
if (( ${#DATASETS[@]} == 0 )); then DATASETS=(brats); fi

if [[ "$RUN_TAG" == */* || "$RUN_TAG" == "." || "$RUN_TAG" == ".." ]]; then
  echo "RUN_TAG must be a single directory name, got: $RUN_TAG" >&2
  exit 2
fi

RUN_ROOT="runs/$RUN_TAG"
cd "$PROJECT_ROOT"

# Freeze the launcher before training so later edits cannot change this run.
if [[ "$DRY_RUN" == false && "${PROJECTOR_MLP_LAUNCHER_SNAPSHOT:-0}" != "1" ]]; then
  if [[ -e "$RUN_ROOT" ]]; then
    echo "Run directory already exists; refusing to overwrite: $RUN_ROOT" >&2
    exit 2
  fi
  if ! mkdir "$RUN_ROOT"; then
    echo "Could not create a fresh run directory: $RUN_ROOT" >&2
    exit 2
  fi
  LAUNCHER_SNAPSHOT="$PROJECT_ROOT/$RUN_ROOT/launcher_snapshot.sh"
  cp -- "$0" "$LAUNCHER_SNAPSHOT"
  if ! bash -n "$LAUNCHER_SNAPSHOT"; then
    echo "Launcher syntax check failed; no training was started." >&2
    exit 2
  fi
  export PROJECTOR_MLP_LAUNCHER_SNAPSHOT=1
  exec bash "$LAUNCHER_SNAPSHOT" "$RUN_TAG" --datasets "${DATASETS[@]}" "${CACHE_ARGS[@]}" "${FORWARDED[@]}"
fi

projector_python=${PROJECTOR_PYTHON:-/mnt/sda/mingjie/conda_envs/projector/bin/python}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export PYTHONUTF8=1
export PYTHONIOENCODING=utf-8

if [[ "$DRY_RUN" == false ]]; then
  exec > >(tee -a "$RUN_ROOT/pipeline_console.log") 2>&1
  echo "Continuous MLP + visual LoRA training started: $(date --iso-8601=seconds)"
  echo "Run root: $RUN_ROOT"
  echo "Frozen launcher: $RUN_ROOT/launcher_snapshot.sh"
  sha256sum "$RUN_ROOT/launcher_snapshot.sh"
  echo "RadGenome datasets: ${DATASETS[*]}; preprocessing cache: $CACHE_SETTING"
fi

"$projector_python" -m src.cli.train \
  --config configs/training/joint_lora.yaml \
  --config configs/training/deterministic.yaml \
  --config configs/projector/mlp2x_gelu.yaml \
  --config configs/preprocessing/native_grid.yaml \
  --datasets "${DATASETS[@]}" \
  "${CACHE_ARGS[@]}" \
  "${FORWARDED[@]}" \
  --set "training.run_dir=$RUN_ROOT/joint_lora"

if [[ "$DRY_RUN" == false ]]; then
  echo "Continuous training completed: $RUN_ROOT/joint_lora"
  echo "Resolved parameters: $RUN_ROOT/joint_lora/resolved_config.yaml"
  echo "Resolved schedule: $RUN_ROOT/joint_lora/schedule.json"
fi
