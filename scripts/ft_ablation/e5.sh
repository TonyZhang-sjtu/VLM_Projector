#!/usr/bin/env bash
# E5: Combine 3e-5 peak learning rates with 500 projector hold updates.
# One invocation trains a fresh model, then evaluates best.pt on the full test split.
set -Eeuo pipefail

# ===== GPU: edit the physical index below; an exported value can also override it =====
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4}"

# ===== Experiment settings: all step counts are optimizer updates =====
ft_projector_lr=3e-5
ft_encoder_lr=3e-5
ft_projector_warmup_steps=77
ft_projector_hold_steps=500
ft_joint_warmup_steps=77
ft_max_epochs=10  # Total epoch cap, INCLUDING all preceding phases.
ft_run_tag="${FT_RUN_TAG:-ft1017_e5_avgpool3d_r8_lr3e-5_p500_ep10_v1}"

# Shared settings: LoRA rank=16/alpha=32; hold_peak; cosine floor=0.2.
# Shared runner includes --eval-split test --metrics all --require-full-test.
ft_script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$ft_script_dir/_common.bash"
run_ft_ablation "$@"
