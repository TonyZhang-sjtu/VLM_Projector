# Reuse the AvgPool constant-LR runner, including local paths and full-test guards.
run_bindformer_constant_lr_1019() {
  local bind_common_dir
  bind_common_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../ft_ablation" && pwd)"
  source "$bind_common_dir/_constant_lr.bash"
  if [[ -z "${projector_run_tag:-}" ]]; then
    projector_run_tag="bindformer_${bind_variant}_1019_p${projector_lr}_l${encoder_lr}_$(date -u +%Y%m%dT%H%M%S%NZ)"
  fi

  # Keep concept-bank paths in the ignored local overlay, not in a tracked shell.
  # Training and evaluation reuse baseline_sweep, which also supports BindFormer.
  # CLI --set overrides these defaults explicitly and is recorded in sweep_plan.json.
  run_constant_lr_baseline \
    --set "projector.params.within_slot_sa=$bind_within_slot_sa" \
    --set "projector.params.local_reads_global=$bind_local_reads_global" \
    --set "projector.params.within_slot_sa_depth=$bind_within_slot_sa_depth" \
    --set "projector.params.local_global_depth=$bind_local_global_depth" \
    --set "projector.params.local_global_gate_init=$bind_local_global_gate_init" \
    --set "projector.params.global_fraction=$bind_global_fraction" \
    --set "projector.params.slot_iterations=$bind_slot_iterations" \
    --set "projector.params.knowledge_mode=$bind_knowledge_mode" \
    --set "projector.params.vocabulary_scope=brats_train" \
    --set "projector.params.role_query_counts.finding=1" \
    --set "projector.params.role_query_counts.location_laterality=1" \
    --set "projector.params.role_query_counts.signal=1" \
    --set "projector.params.role_query_counts.morphology_extent=1" \
    --set "projector.params.geometry_enabled=$bind_geometry_enabled" \
    --set "projector.params.feature_loss_weight=$bind_feature_loss_weight" \
    --set "projector.params.text_loss_weight=0.0" \
    --set "projector.params.tuple_loss_weight=0.0" \
    --set "projector.params.binding_loss_weight=0.0" \
    --set "projector.params.concept_loss_weight=0.0" \
    "$@"
}
