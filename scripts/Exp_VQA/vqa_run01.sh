#!/usr/bin/env bash
# A100 80GB / GPU 0: 三个 baseline 并行，各自按倍率顺序 train -> best.pt -> test。
# 用法: bash scripts/Exp_VQA/vqa_run01.sh [--dry-run | --check-data] [VQA 参数]
# 换卡: CUDA_VISIBLE_DEVICES=2 bash scripts/Exp_VQA/vqa_run01.sh
# 环境: 自动 conda activate projector；可设置 PROJECTOR_CONDA_ENV 或 PROJECTOR_PYTHON。
# 显存参考：A5500 单进程 r8 实测 reserved 峰值合计约 53.7 GiB；非 A100 并发保证。
set -Eeuo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
vqa_group=run01
vqa_group_scripts=(vqa_local_packer3d.sh vqa_perceiver_nv.sh vqa_adaptive_merge3d.sh)
vqa_group_ratios=(256 128 64 32 16 8)
vqa_min_free_mib="${VQA_MIN_FREE_MIB:-67584}"  # 66 GiB，含多进程/allocator 余量；并非峰值保证。

vqa_group_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$vqa_group_dir/_parallel.bash"
run_vqa_parallel "$@"
