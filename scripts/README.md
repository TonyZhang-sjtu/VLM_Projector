# Project scripts

## Baseline ratio sweeps

[`baselines/`](baselines/README.md) contains 11 Bash entrypoints. Each compressed
method independently trains ratios 4, 8, 16 and 32 with the `joint_lora` preset,
evaluating that run's lowest-validation-CE `best.pt` immediately after training.
Any failure stops the sweep. MLP runs once at 1x without compression.

```bash
bash scripts/baselines/run_avgpool3d.sh --dry-run
CUDA_VISIBLE_DEVICES=3 bash scripts/baselines/run_avgpool3d.sh --run-tag avgpool_joint_v1
CUDA_VISIBLE_DEVICES=3 bash scripts/baselines/run_perceiver_nv.sh --ratios 8 32
```

Single-GPU training defaults to logical `cuda:0`; choose the physical GPU outside
the program with `CUDA_VISIBLE_DEVICES`. Baseline generation and scoring inherit
the training device. The wrappers preserve the external visibility setting.
Default evaluation is the complete BraTS test split with all existing metrics.
See the [launcher guide](baselines/README.md) for parameter overrides, fixed
evaluation subsets, output locations and same-run training resume.

## Continuous MLP + vision LoRA training

`run_full_mlp_uncompressed.sh` now starts one `joint_lora` run with native-grid
MRI, the uncompressed `mlp2x_gelu` projector, a frozen Llama, and visual LoRA
rank 16 / alpha 32 / dropout 0.05. Both peak learning rates are 1e-5. The maximum
budget is 10 epochs, with two 6.25% linear warm-ups and joint cosine decay.
After warm-up, validation CE uses min_delta=0.005 and patience=3 for early stopping.
Only preprocessing patches are cached; encoder features are recomputed.

```bash
# Resolve parameters, sample count and schedule without loading models or creating a run.
bash scripts/run_full_mlp_uncompressed.sh my_joint_run --dry-run --datasets brats

# Start a fresh run; CLI settings override the preset.
bash scripts/run_full_mlp_uncompressed.sh my_joint_run --datasets brats \
  --set training.device=cuda:0 \
  --set training.max_epochs=10 \
  --set training.projector_warmup_steps=77 \
  --set training.joint_warmup_steps=77 \
  --set training.lora.rank=16 --set training.lora.alpha=32
```

The optional first argument is a run tag; the default includes a timestamp.
Outputs are in `runs/<tag>/joint_lora/`, and the launcher refuses an existing run
folder. `--config`, `--set`, `--datasets`, and preprocessing cache switches are
forwarded. Use the run tag, rather than overriding `training.run_dir`, with this
wrapper. `PROJECTOR_PYTHON` can select the environment's Python executable.
Each explicit warm-up step count overrides its corresponding ratio; omitted
counts are computed from the planned total optimizer updates, not microsteps.
Rank and alpha can be changed independently; alpha is not silently rescaled.

For another projector or same-run resume use the generic launcher, passing the
same original overrides. Resume uses `last.pt` and the same run directory:

```bash
scripts/train_projector.sh mlp2x_gelu joint_lora runs/my_joint_run/joint_lora \
  --config configs/training/deterministic.yaml \
  --set training.device=cuda:0 \
  --set training.projector_warmup_steps=77 --set training.joint_warmup_steps=77 \
  --set training.resume=runs/my_joint_run/joint_lora/checkpoints/last.pt
```

The legacy `a`, `b_lora`, and `b_full` presets remain available through
`train_projector.sh`. For checkpoint evaluation use `evaluate_checkpoint.sh`;
for DDP use `launch_ddp.sh` with `configs/training/joint_lora.yaml` and the DDP
preset. See [the training specification](../Pipeline_0929/NEUROVFM_MLP_TRAINING_PIPELINE.md)
for the schedule, early-stopping rules, and validation evidence.

## Checkpoint evaluation on a reproducible subset

`evaluate_checkpoint.sh` supports `--sample-fraction` for deterministic random
subsampling. For example, this evaluates the validation-selected Stage A model
on 30% of the test split and writes the selected IDs with the results:

```bash
bash scripts/evaluate_checkpoint.sh \
  --checkpoint runs/full_mlp_uncompressed_20261009/stage_a/checkpoints/best.pt \
  --output-dir runs/full_mlp_uncompressed_20261009/evaluation/stage_a_best_test_30pct \
  --split test \
  --device cuda:2 \
  --scoring-device cuda:2 \
  --metrics all \
  --sample-fraction 0.3 \
  --sample-seed 42
```

If `--sample-seed` is omitted, the checkpoint's configured seed is used.
`--limit` and `--sample-fraction` are mutually exclusive. The evaluation plan,
fixed ID list, predictions, and metrics are saved under the chosen output
directory; a fractional result is labeled `fractional_sample`, not `full_split`.

Only reusable pipeline launchers and preprocessing-artifact creation remain in
this directory. One-off smoke, audit, diagnosis, download, and setup utilities
are preserved under `archive/non_pipeline/scripts/`; they are not needed for a
normal run because required model and metric assets are already local.

## Projector compression ratios

Choose a target `N/M` before model construction:

```bash
python -m src.cli.config --set projector.name=concat3d \
  --set projector.target_compression_ratio=16 --resolve-budget \
  --budget-report runs/budget_preview/concat_r16.json

bash scripts/train_projector.sh avgpool3d joint_lora runs/brats_avgpool_r8 \
  --set projector.target_compression_ratio=8
```

Use 4, 8, 16 or 32; Concat groups are `[4,1,1]`, `[8,1,1]`, `[6,1,3]`,
`[6,2,3]`. Grid methods retain their existing partitions; query, merge and
selection methods use `ceil(N/r)`. No common output count is imposed. The
automatic geometry adapter currently requires dense NeuroVFM native-grid inputs
with minimal padding and one geometry per run. It checks all selected train/val
headers before model allocation and validates actual geometry again at forward.

Manual `projector.params.num_tokens` / `group_size` remain available when no ratio
is specified. Do not combine them with ratio mode. Restore the saved resolved
configuration for resume/evaluation; use a fresh configuration for a new ratio.
`--resolve-budget` previews parameters without loading model weights. The training
run writes `budget_preflight.json` and records requested and actual compression.

## BERTScore

Omitting `--metrics` defaults to `all`, including BERTScore precision, recall
and F1. The checkpoint evaluation summary is always `<output-dir>/metric.json`.
The `generation/` and `scoring/` subdirectories retain their detailed outputs.
Use
`--metrics bertscore` or `--metrics bleu4 rougeL bertscore` to select it.
The pinned encoder is local at `assets/models/evaluation/bert-score/`:
`google-bert/bert-base-uncased`, layer 9, FP32, slow tokenizer, no IDF or baseline
rescaling. Reports exceeding 512 tokens (including special tokens) fail visibly;
they are never silently truncated. BERTScore compares generated and reference
text and is not a diagnosis-category F1.

To score existing reports without generating them again:

```bash
PYTHONNOUSERSITE=1 OMP_NUM_THREADS=4 /mnt/sda/mingjie/conda_envs/projector/bin/python \
  -m src.cli.score_reports \
  --predictions /path/to/predictions.jsonl \
  --output-dir runs/bertscore_rescore \
  --metrics bertscore --device cpu
```

Supply `--expected-ids /path/to/expected_ids.json` for independent coverage
checking. GPU scoring is available via `--device cuda:N`. Downloads, frozen
settings and smoke evidence are documented in
[the evaluation guide](../Pipeline_0929/EVALUATION_METRICS_AND_NEUROQA_PLAN.md).
The reproducible download utility is
`archive/non_pipeline/scripts/download_bertscore_assets.py`.
