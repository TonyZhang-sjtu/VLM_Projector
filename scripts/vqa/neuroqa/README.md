# NeuroQA VQA

This task uses one T1w or T2w volume per QA, mixed training with exactly equal
modality counts, a fixed MNI 1 mm canvas, and a frozen Llama decoder. It shares
the model/optimizer/checkpoint components with report generation, with separate
manifests, prompts, trainer orchestration, metrics, scripts, and output paths.

The current launch settings, scientific protocol, budget table, validation
guidance and limitations are in [the VQA launcher guide](../../Exp_VQA/README.md).

Run from the repository root using the project's Python environment:

```bash
python scripts/vqa/neuroqa/preprocess_smri.py --inventory-only
python scripts/vqa/neuroqa/preprocessing_status.py
python -m src.cli.vqa_train --config configs/vqa/neuroqa/mixed.yaml --dry-run
CUDA_VISIBLE_DEVICES=0 python scripts/vqa/neuroqa/run_experiment.py \
  --projector avgpool3d --ratio 128 --seed 42 --run-tag qa_baseline_v1
```

`run_experiment.py` and `run_baselines.sh` preview commands by default. Add
`--execute` to train. Full training requires all selected images to have completed
preprocessing. An existing preprocessing batch must finish or stop before another
producer is started for the same cases. Use `preprocess_smri.py --workers 8
--threads 2` to process/resume the whole inventory after the current worker exits.

The five nominal ratios and actual output lengths are:

| Ratio | M | Grid DHW | Concat group DHW |
| --- | ---: | --- | --- |
| 16 | 624 | 6×8×13 | 9×2×1 |
| 32 | 288 | 9×8×4 | 6×2×4 |
| 64 | 150 | 6×5×5 | 9×3×3 |
| 128 | 75 | 3×5×5 | 17×3×3 |
| 256 | 36 | 3×4×3 | 18×4×5 |

Input N is 9,555; no extra 4 mm resampling. MLP is an uncompressed r1 reference.
Concat defaults to a CPU projector in the launcher, permitting the complete head
and FP32 AdamW on a single assigned 24 GB GPU. Declare that placement in efficiency
results. A second assigned GPU can be selected with `--concat-device cuda:1`.
BindFormer requires `knowledge_mode=none` until a QA-train-only concept bank is
built; the launcher applies this setting.

```bash
# Full models, real data, all compressed projectors and five ratios.
CUDA_VISIBLE_DEVICES=0 python scripts/vqa/neuroqa/smoke_gpu.py \
  --output runs/vqa/neuroqa/smoke/NEW_MATRIX_NAME
# Production trainer + interruption at the projector/LoRA boundary.
CUDA_VISIBLE_DEVICES=0 python scripts/vqa/neuroqa/smoke_training_resume.py \
  --output runs/vqa/neuroqa/smoke/NEW_RESUME_NAME
python -m pytest tests/test_neuroqa.py -q
# Evaluate the exact saved configuration. Use separate outputs for each control.
CUDA_VISIBLE_DEVICES=0 python -m src.cli.vqa_evaluate \
  --checkpoint RUN_DIR/checkpoints/best.pt --split test --output RESULTS_DIR
```

Use `--control text|zero|shuffle` for image-dependence controls. Results include
coverage checks, stratified scores, per-patient bootstrap, and separate official
prediction files for each modality. Empty/invalid answers remain in the denominator.

Smoke runs are engineering checks, not trained benchmark results. The status
inventory and per-case QC must be reviewed before reporting a complete dataset.

## Isolated v2 registration protocol

V2 uses `configs/vqa/neuroqa/mixed_v2.yaml`, its own guarded manifest and
`assets/preprocessing/neuroqa_mni1mm_v2`. V1 source data and caches remain in
their original directory. Check the live pilot/coordinator and full-dataset
state with `python scripts/vqa/neuroqa/preprocessing_v2_status.py`.

The configured coordinator waits for the complete train-only pilot, independent
single-thread repeatability, N4/adapter integration and pilot image review before
freezing a shared protocol. It stops on unresolved QC failures; full training is
blocked until every volume is independently audited and the flagged review queue
has been completed. No formal VQA training is started by the coordinator.

Launchers and engineering smoke scripts accept `--config configs/vqa/neuroqa/mixed_v2.yaml`;
v2 experiment results use an explicit
`mni_v2` path component. See the plan's section 12.3 for current status and the
scientific limits of automated QC.
