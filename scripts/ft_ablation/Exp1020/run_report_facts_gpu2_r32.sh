#!/usr/bin/env bash
# GPU 2 / 32x: E2 -> E3 -> E1。每项独立初始化；训练+完整评测成功后继续。
set -Eeuo pipefail
export CUDA_VISIBLE_DEVICES=2
export PROJECTOR_RATIOS=32
# 默认关闭边界辅助监督；morphology/extent 暂无独立属性监督，四角色结构保留。
export REPORT_FACT_FIELDS="${REPORT_FACT_FIELDS:-[edema,laterality,site,signal]}"
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
queue_tag="${PROJECTOR_RUN_TAG:-facts_v1_gpu2_r32_$(date -u +%Y%m%dT%H%M%S%NZ)}"
for arg in "$@"; do
  case "$arg" in --resume-run*|--ratios*|--run-tag*) echo 'Use the standard launcher for resume/ratio/tag overrides.' >&2; exit 2;; esac
done
for experiment in E2 E3 E1; do
  case "$experiment" in
    E2) fact=false; pair=true ;;
    E3) fact=true; pair=true ;;
    E1) fact=true; pair=false ;;
  esac
  EXPERIMENT="$experiment" FACT_ENABLED="$fact" PAIR_ENABLED="$pair" \
    PROJECTOR_RUN_TAG="${queue_tag}_${experiment}" \
    bash "$script_dir/run_bindformer_report_facts_constlr.sh" "$@"
done
