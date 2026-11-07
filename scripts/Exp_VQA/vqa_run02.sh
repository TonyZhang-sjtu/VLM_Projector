#!/usr/bin/env bash
# A100 80GB / GPU 1: 四个 baseline 并行，各自按倍率顺序 train -> best.pt -> test。
# ADirect+Local 对应现有 bindformer_adirect_localread。
# 用法: bash scripts/Exp_VQA/vqa_run02.sh [--dry-run | --check-data] [VQA 参数]
# 换卡: CUDA_VISIBLE_DEVICES=3 bash scripts/Exp_VQA/vqa_run02.sh
# 环境: 自动 conda activate projector；可设置 PROJECTOR_CONDA_ENV 或 PROJECTOR_PYTHON。
# 显存参考：A5500 单进程较高峰值合计约 75.2 GiB（含 Concat r64）。
# 此数不含 CUDA context 等额外占用，四路并发余量较小；非 A100 并发实测。
set -Eeuo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
vqa_group=run02
vqa_group_scripts=(vqa_baseline_adirect_localread.sh vqa_concat3d.sh vqa_avgpool3d.sh vqa_token_select3d.sh)
vqa_group_ratios=(64 32 16 8)
vqa_min_free_mib="${VQA_MIN_FREE_MIB:-79872}"  # 78 GiB；四路并行需要几乎空闲的 80GB 卡。
export VQA_CONCAT_DEVICE="${VQA_CONCAT_DEVICE:-cuda:0}"  # 同卡；可显式改 cpu 以降低显存。

vqa_group_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$vqa_group_dir/_parallel.bash"
run_vqa_parallel "$@"
