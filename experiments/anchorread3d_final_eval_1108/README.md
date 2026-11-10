# 1108 final evaluation: raw p-values and RG visual dependence

This package adds evaluation only. It does not change `src/`, training scripts,
checkpoints, historical predictions, or the source workbook. All new artifacts
are under `results/anchorread3d_final_eval_1108/`; the two deliverable workbooks
are in `results/`.

## Frozen inputs and scope

The source is `results/Experiments for VLM Projector 1108.xlsx`, resolved by
`results/anchorread3d_analysis_1108/artifact_reconciliation_1108.json`.
The source hashes, original result identities, checkpoint hashes, model assets,
references, preprocessing, scorer contracts, and sample alignment are checked.
Known Marlowe path relocations are allowed only with the recorded equivalence
proofs. Original preprocessing caches are read-only; new entries use an overlay
inside this experiment's result directory.

Significance covers A0 against AvgPool3D, Concat3D, ConvPool3D, LocalPacker3D,
Perceiver_NV, Adaptive_Merge3D, and Token_Select3D at all six compression ratios,
plus A1–A4 against A0 at 32×/256× for RG and VQA. It excludes MLP, Mixer3D,
AttentionPool, and Perceiver_Flamingo. No model is retrained.

Visual dependence covers RG only: A0, AvgPool3D, Perceiver_NV, and Token_Select3D
at 32×/256×. There are 8 checkpoints, 4 conditions, and 3 shuffle repetitions.
The 48 individual condition/seed rows become 32 main-table rows after averaging
the three shuffle metrics. A0 means the full ADirect+LocalRead/AnchorRead3D model.

## Statistical protocol

`significance.py` uses a two-sided paired permutation test with patient-level
label exchanges, 19,999 Monte Carlo exchanges and seed 20261108. Reports or QA
records belonging to the same patient always exchange together. Aggregation
matches the original metric, including corpus BLEU, token-weighted RG CE and
question-weighted VQA CE. VQA answers are also independently rescored against
the original ground truth before the cached scores are accepted.

The workbook contains raw p-values: `(extreme + 1) / (19999 + 1)`. This finite
Monte Carlo calibration is not a multiple-comparison correction. The smallest
resolvable value is 0.00005. Holm, Bonferroni and BH procedures, possible
hypothesis families, effect-direction cautions and coverage rules are documented
in `results/anchorread3d_final_eval_1108/STATISTICAL_PROTOCOL.txt`.

The expected count is 1,050 tests: 462 in Main, 420 in VQA and 168 in Ablation.
The workbook preserves these three source sheets' layout. A0 and excluded-method
rows remain blank; confidence-interval endpoint columns are N/A rather than
being tested as independent metrics. Effect sizes, paired coverage and input
provenance are in cell comments and the `significance/` JSON/CSV sidecars.

When original per-report RG CE was unavailable, `inference.py` obtains it with
teacher forcing on the same frozen checkpoint. It does not regenerate Normal
reports. Existing validated CE exports are reused. Small historical/runtime
BF16 differences are retained in `historical_comparison.json` and cell comments;
the CE p-values refer to these replay measurements, not to unavailable historical
per-report losses. All other completed original scores are reused.

## Visual interventions

- Normal: the original matched image/report prediction. CE is measured in the
  same process as the shuffled-image CE.
- Patient shuffle: a patient-level derangement with seeds 20261108/09/10. The
  donor differs from the target, has the same sequence/cohort/input geometry,
  and is shared across all four sequences of a target patient. Every checkpoint
  uses the same frozen maps. The target reference stays unchanged.
- Zero-prefix: zero the final projector output while retaining its true token
  count and attention-mask positions. At 256× AvgPool3D uses 36 tokens; the other
  selected methods use 35. All selected 32× settings use 275.
- Text-only: remove the visual prefix entirely, retaining the original textual
  prompt. This is an auxiliary stress condition because the prompt still asks
  for an image description.

The saved RG prompt depends only on sequence and decoding is independent and
greedy. A donor's existing generated text therefore provides the shuffled-image
output for the same sequence. Runtime equivalence is checked for all four
sequences per checkpoint. Historical token agreement is logged separately
because hardware-level numerical differences may affect reruns. **Donor scores
and donor-reference CE are never copied.** All changed reference/prediction
pair scores and target-reference CE are recomputed.

The decoder is frozen. Zero/text outputs may be reused only under an identical
decoder asset, prompt, generation, prefix-length, precision and reference
contract. The small-batch smoke test also compares these optimized controls with
the unchanged production `model.generate` zero/text paths.

All 11 requested RG metrics are exported. Exact-text RadGraph extraction is
cached, but the official three RadGraph rewards are recomputed for each new pair.
New text is extracted on CPU after weight/config validation and probes against
16 cached graphs. Graph V2 additionally reports side, region and joint-location
set precision/recall/F1/EM with reference-defined eligibility. It measures
automatically extracted report agreement, not image-grounded lesion accuracy or
lesion instance counts. Clinical scorer failures and sample coverage remain
explicit rather than being converted to zero scores.

## Execution, gates and monitoring

Use the project environment (on this server:
`/mnt/sda/mingjie/conda_envs/projector/bin/python`) from the project root.

```bash
# Read-only progress summary
scripts/analysis_1108/status.sh

# CPU-only significance replay once CE exports exist
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  /mnt/sda/mingjie/conda_envs/projector/bin/python \
  -m experiments.anchorread3d_final_eval_1108.significance

# Final GREEN stage; run only if the automatic workflow is not already doing it
scripts/analysis_1108/run_visual_green.sh
```

`workflow.py` orders the work: CE exports → complete p-values → visual smoke
test → full visual inference → full-record audit → scoring pool → RadGraph/V2 → RaTE/BERT → output
validation and intermediate workbook → GREEN gate → GREEN → final validation
and workbook. A failed stage blocks subsequent stages. There is no automatic
model retraining or silent fallback to different data/models.

`queue.py` starts one worker per idle GPU. It excludes any GPU with an existing
compute process, verifies free memory/utilization, uses all available cards if
there are at most two, and leaves one unused if more than two are available.
It never signals foreign processes. File locks prevent duplicate queue/workflow
launches. Existing training jobs retain their cards.

The detached workflow and observer PIDs are recorded in `service_pids.json`;
status is in `workflow_status.json`, `monitor_status.json`, and `queues/`.
The observer checks worker liveness, CE identities/finite losses and scoring
cache status. Alert history is retained; verified resolutions are recorded in
`alert_resolutions.json` rather than deleting the original observations.
`clinical_worker.py` launches the unchanged production scorer, terminates only
its own child on a new `scoring_error` or 30 minutes without cache progress, and
writes `worker_monitor.json`. GREEN formatting failures remain explicit protocol
outcomes and do not trigger this runtime-error guard. Workflow logs are in
`workflow_logs/`, worker logs in `queues/`.
Process success alone does not satisfy a gate: expected IDs, counts, score
contracts, reference eligibility and original Normal score reproduction must
also pass. Source hashes are retained for auditing.

Deliverables:

- `results/Experiments for VLM Projector 1108_pvalues.xlsx`
- `results/AnchorRead3D_RG_VisualDependence_1108.xlsx`

The visual workbook explicitly says GREEN is pending until the final GREEN
verification succeeds. It includes main tables, individual shuffle seeds,
spatial details, GREEN error components, coverage, Normal-minus-control changes,
and provenance. A pending workbook must not be presented as a completed GREEN
evaluation.
