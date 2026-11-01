#!/usr/bin/env bash
# A4 report generation: fresh projector + fresh vision LoRA from pretrained assets.
# Recipe matches run_baseline_a_r128_r256.sh; no A/A4 experiment checkpoint is loaded.
# Sequentially: train r128 -> best.pt full test/all -> train r256 -> full test/all.
set -Eeuo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-3}"
export PROJECTOR_PYTHON="${PROJECTOR_PYTHON:-/mnt/sda/mingjie/conda_envs/projector/bin/python}"
projector_method=bindformer_a4
projector_ratios=(128 256)
projector_lr=2e-4
encoder_lr=2e-4  # Vision encoder LoRA only; the LLM stays frozen.
projector_max_epochs=20
projector_seed=42
projector_effective_batch_size=16
projector_microbatch_per_device=1
encoder_lora_rank=16
encoder_lora_alpha=32
projector_warmup_steps=77
projector_hold_steps=0
joint_warmup_steps=77
joint_warmup_projector_mode=hold_peak
cosine_min_lr_ratio=1.0
early_stopping_enabled=true
early_stopping_patience=3
early_stopping_min_delta=0.005
projector_run_tag="${PROJECTOR_RUN_TAG:-}"

# Same nominal training cap and early-stopping rule as A. Actual stopping steps
# may differ. For equal-update comparisons, give BOTH scripts the same
# --set training.early_stopping_enabled=false --set training.max_steps=N.
# A4 structure lives in configs/projector/bindformer_a4.yaml; old BIND_* switches
# do not apply. Do not pass the old A projector YAML or its --set role settings.
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$script_dir/../ft_ablation/_constant_lr.bash"
run_constant_lr_baseline \
  --set "training.effective_batch_size=$projector_effective_batch_size" \
  --set "training.microbatch_per_device=$projector_microbatch_per_device" \
  --set "training.lora.rank=$encoder_lora_rank" \
  --set "training.lora.alpha=$encoder_lora_alpha" \
  "$@"
