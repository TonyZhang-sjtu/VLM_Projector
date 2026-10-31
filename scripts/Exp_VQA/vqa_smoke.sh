#!/usr/bin/env bash
# Real encoder + Llama + production trainer; 6 updates, 6 val views, 6 test views.
# Engineering coverage only. Short generations and subset training are not scores.
set -Eeuo pipefail
vqa_script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$vqa_script_dir/vqa_avgpool3d.sh" \
  --ratios 256 \
  --set "training.train_ids_file=$vqa_script_dir/selections/smoke_train.json" \
  --set "training.val_ids_file=$vqa_script_dir/selections/smoke_val.json" \
  --set training.effective_batch_size=2 \
  --set training.max_epochs=3 \
  --set training.projector_warmup_steps=2 \
  --set training.joint_warmup_steps=2 \
  --set training.save_every_steps=1 \
  --set training.early_stopping_enabled=false \
  --set generation.max_new_tokens=8 \
  --eval-ids-file "$vqa_script_dir/selections/smoke_test.json" \
  --bootstrap 20 "$@"
