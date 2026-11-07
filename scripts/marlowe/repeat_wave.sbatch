#!/usr/bin/env bash
#SBATCH --account=marlowe-m000065-pm07
#SBATCH --partition=batch
#SBATCH --qos=medium
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=192G
#SBATCH --time=2-00:00:00
#SBATCH --open-mode=append
#SBATCH --requeue
set -Eeuo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from VLM_Projector}"
: "${CUDA_VISIBLE_DEVICES:?Slurm must allocate one GPU}"
exec bash scripts/marlowe/run_m13.sh "$@"
