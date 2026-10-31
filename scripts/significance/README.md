# Independent significance analysis

For the latest Excel's **16/32/64/128/256x Baseline A vs six baselines (no MLP)**,
use `run_baseline_a_5ratios.sh`. See [the five-ratio instructions](README_baseline_a_5ratios.md).
The single-ratio screenshot workflow below is retained for the earlier study.

These entry points are separate from training and ordinary report-evaluation
launchers. Original runs and the source Excel files are read-only inputs.

The initial study is the user's **256x screenshot**, not an inferred 256x sheet
in `results/Experiments for VLM Projector.xlsx` (that local workbook currently
ends at 64x in Main). `study_256.json` explicitly records this source limitation.
It binds one Ours checkpoint and seven baseline identities: 77 hypotheses.

## Run identity

`run_manifest.json` binds each row to a full run directory and checkpoint SHA256.
Do not select the newest run or the closest score. Candidate discovery searches
all runs, checks projector/ratio, and compares **all 11 displayed scores** at the
screenshot's four-decimal precision. It reports candidates but never auto-binds
them: rounded numbers alone do not identify an experiment uniquely.

Before testing, the pipeline verifies:

1. The bound projector, compression ratio, seed, resolved configuration, and
   checkpoint bytes; the saved best checkpoint's validation step and CE.
2. The evaluation/prediction checkpoint hash and scoring input-file hash.
3. Exact test IDs, references, patient IDs, and prediction/scoring alignment.
4. Scorer fingerprints and cached text hashes; training/validation selections.
5. The recomputed raw metrics against the recorded unrounded metric.json and
   the target screenshot values. CE initially verifies historical token totals;
   inferential CE requires the new per-sample export.

The report includes full source hashes and explicit missing-source/audit errors.
The local AvgPool3D and LocalPacker3D rows select their constant-LR reruns;
earlier cosine runs are different candidates and do not match the screenshot.

## Commands

From the repository root, using the frozen manifest in the report directory.
To validate identity and discover candidates without running statistical tests
or model inference:

```bash
bash scripts/significance/run_pvalues.sh \
  --manifest results/significance_256_20261030/run_manifest.json \
  --output-dir results/significance_256_20261030 --audit-only --discover

bash scripts/significance/export_ce.sh \
  --manifest results/significance_256_20261030/run_manifest.json \
  --method ours --output-dir results/significance_256_20261030/ce/ours \
  --device cuda:0 --dry-run
```

`--dry-run` verifies the checkpoint and test data but does not validate actual
GPU forward execution. On 2026-10-30, the waiting CE queue was stopped at the
user's request before any model inference; no full CE replay is running.

The following commands perform the full analysis or start the CE queue. Run
them only when ready to process the selected verified experiments:

```bash
bash scripts/significance/run_pvalues.sh \
  --manifest results/significance_256_20261030/run_manifest.json \
  --output-dir results/significance_256_20261030 --discover

bash scripts/significance/run_ce_queue.sh \
  --manifest results/significance_256_20261030/run_manifest.json \
  --output-dir results/significance_256_20261030 --wait
```

`SIGNIFICANCE_PYTHON` defaults to `python` for CPU analysis (NumPy, PyYAML,
SacreBLEU, rouge-score, Torch for checkpoint metadata, and optionally openpyxl).
`PROJECTOR_PYTHON` selects the GPU environment for `export_ce.sh`.
The queue's equivalent override is `--projector-python`.

GPU eligibility requires only at least 20,000 MiB free memory. Existing compute
processes, used memory and utilization do not restrict eligibility. The queue never kills other jobs.
It polls every 30 seconds, with a default 24-hour waiting timeout. This is a
local cooperative queue, not a reservation against unrelated future launches.
It exports available verified sources sequentially and refreshes the statistics
when finished. Check `ce/queue_status.json` and each method's `ce/*/console.log`.
Existing successful exports are revalidated before statistical reuse.

CE exports contain ID, patient ID, loss_sum, answer_tokens, per-report CE, and
reference/checkpoint hashes. Aggregation remains sum(loss_sum)/sum(answer_tokens).
There is no generation or clinical-scoring pass. A replay difference exceeding
1e-5 absolute/relative tolerance fails instead of silently replacing the old CE.
Concat's saved CUDA projector placement is mapped to the allocated GPU for this
inference-only pass; the replay check must still pass.

## Statistical protocol and outputs

- 10,000 patient-cluster paired label exchanges, two-sided absolute score
  difference, Monte Carlo p=(extreme+1)/(repeats+1). Exact enumeration is used
  when all patient sign assignments fit the requested budget.
- 10,000 paired patient-cluster bootstrap draws for marginal percentile 95% CI.
- The exchangeability null applies to entire patient vectors. All sequences of
  one patient move together. Scope is fixed checkpoints, not training seeds.
- Pairwise common-valid samples per metric; no imputation. Report each pair's
  recalculated scores and excluded IDs. Missingness may limit generalization.
- Corpus BLEU uses cached within-report n-gram sufficient statistics. Report
  means and token-weighted CE retain the original aggregation after resampling.
- Raw and Holm-adjusted p-values are both exported. Family size is always the
  predeclared 77. While some sources/CEs are missing, available comparisons use
  conservative full-family adjustment, mathematically equivalent to treating
  untested hypotheses as unrejected with p=1 for adjustment only. Their actual
  p_raw/p_Holm cells remain blank. `holm_status` marks this incomplete stage.
  Recompute the entire family once all 77 results exist. CIs are not simultaneous.
- `significance.xlsx`: full comparisons, p_raw matrix, p_Holm matrix, delta
  matrix, run identities, and protocol. `comparisons.csv/json` preserve detail.
- `mapping_candidates.json`: all matching-method/ratio candidates and score
  mismatches. `run_audit.json`: bound identities, configs, hashes, and coverage.

## Transfer from another server

Existing local experiment artifacts are under `runs/baselines/`; summaries and
Excel workbooks are under `results/`. Transfer external runs into:

```
/mnt/sda/jyzhang/VLM_Projector/runs/imported/server_b/<original_run_id>/r256/
```

Keep every distinct configuration/seed in its original unique run directory.
Use a different source-server subdirectory for each server. Do not merge files
from two runs or overwrite a local run. To copy an entire selected original run
from the source server (replace SOURCE_RUN, DEST_USER, DEST_HOST, and RUN_ID):

```bash
rsync -aP SOURCE_RUN/ \
  DEST_USER@DEST_HOST:/mnt/sda/jyzhang/VLM_Projector/runs/imported/server_b/RUN_ID/
```

The desired transferred layout includes:

```
RUN_ID/r256/
  training_config.yaml
  train_console.log
  evaluation_console.log
  joint_lora/
    resolved_config.yaml
    training_state.json
    validation.jsonl
    selection.json
    provenance.json
    checkpoints/best.pt
  evaluation/best_test/
    metric.json
    pipeline.json
    expected_ids.json
    evaluation_config.yaml
    generation/predictions.jsonl
    generation/metrics.json
    scoring/clinical_scores.jsonl
    scoring/scoring_provenance.json
    scoring/metrics.json
```

Copying the complete original run is simplest. For CPU significance, prediction
texts, raw metric scores, and identity records matter; an Excel total alone is
insufficient. For CE replay, also include best.pt and make the same base assets
and test data available locally. Remote absolute paths may need an explicit
local configuration mapping; never change sample identity or scoring versions
to force a match. Keep the original resolved config for audit.

After transfer, run candidate discovery, inspect identity/recipe matches, then
explicitly bind the verified run and hash in the study manifest and rerun audit.
If multiple candidates match the rounded table, the original run provenance is
needed to disambiguate. Upload the updated Excel under `results/imported/server_b/`
to preserve its source and avoid replacing the older workbook.
