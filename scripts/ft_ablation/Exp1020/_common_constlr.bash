# Command assembly only. Edit each run_*.sh for GPU, ratios and training settings.
# Keep entrypoint values intact; command-line overrides remain last.
run_bindformer_routing_constlr() {
  if [[ -z "${projector_run_tag:-}" ]]; then
    projector_run_tag="bindformer_${bind_variant}_constlr_p${projector_lr}_l${encoder_lr}_$(date -u +%Y%m%dT%H%M%S%NZ)"
  fi
  local routing_common_dir
  routing_common_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../baselines/BindFormer" && pwd)"
  source "$routing_common_dir/_common_1019.bash"
  run_bindformer_constant_lr_1019 \
    --set "training.effective_batch_size=$projector_effective_batch_size" \
    --set "training.microbatch_per_device=$projector_microbatch_per_device" \
    --set "training.lora.rank=$encoder_lora_rank" \
    --set "training.lora.alpha=$encoder_lora_alpha" \
    --set "projector.params.head_type=$bind_head_type" \
    --set "projector.params.router_value_normalization=$routing_value_normalization" \
    --set "projector.params.router_value_projection=$routing_value_projection" \
    --set "projector.params.fixed_spatial_regions=$routing_fixed_regions" \
    --set "projector.params.spatial_mean_aggregation=$routing_mean" \
    --set "projector.params.spatial_adaptive_aggregation=$routing_adaptive" \
    --set "projector.params.spatial_attention_temperature=$routing_attention_temperature" \
    "$@"
}
