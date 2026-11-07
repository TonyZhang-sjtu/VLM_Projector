#!/usr/bin/env bash
#SBATCH --account=marlowe-m000065-pm07
#SBATCH --partition=batch
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
group="${1:?Expected m06, m07, m08, or m09}"
[[ "$group" =~ ^m0[6-9]$ ]] || exit 2
: "${CUDA_VISIBLE_DEVICES:?Slurm must allocate one GPU}"
exec bash "scripts/marlowe/run_${group}.sh"
