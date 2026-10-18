#!/usr/bin/env bash
# N2: fresh AvgPool3D 8x training, then full-test evaluation of epoch 10 and best only.
set -Eeuo pipefail

# Physical GPU on THIS server. An exported CUDA_VISIBLE_DEVICES takes precedence.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"

# All stage lengths are optimizer updates. Warm-ups are included in max_epochs.
ft_projector_lr=1e-4
ft_encoder_lr=1e-4
ft_projector_warmup_steps=77
ft_projector_hold_steps=0
ft_joint_warmup_steps=77
ft_max_epochs=20
ft_cosine_min_lr_ratio=1.0
ft_early_stopping_enabled=false
ft_checkpoint_epochs='[5,10,15,20]'  # Checkpoint saving policy.
ft_eval_checkpoint_epochs=(10)  # Test epoch 10 and best.
ft_run_tag="${FT_RUN_TAG:-ft_n2_avgpool3d_r8_p1e-4_l1e-4_floor1.0_ep20_v1}"
ft_cli_module=src.cli.ft_experiments

# Shared: hold_peak, AdamW, LoRA rank=16/alpha=32, optimizer diagnostics enabled.
# best.pt is selected by full validation CE; test metrics never select a model.
ft_script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$ft_script_dir/_common.bash"
run_ft_ablation "$@"
