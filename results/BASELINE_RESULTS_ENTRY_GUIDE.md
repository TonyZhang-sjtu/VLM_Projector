# Baseline 结果填写说明（VLM Projector）

本文和 `Experiments for VLM Projector整理版.xlsx` 是同一版本的填写规范。请另一台服务器的 Codex 按相同的指标列、行标签和评测口径，把各 baseline 的结果填入对应格子；先核实产物中的模型、压缩率、split 和覆盖数，再写入 Excel。

## 项目和实验设置

本项目比较 NeuroVFM 3D MRI encoder 后的视觉 projector。编码器在当前固定网格上输出 `39×15×15=8,775` 个视觉 token；projector 将其映射到 Llama 3.1 8B 的视觉输入空间，语言模型生成单个 MRI 序列的英文 Findings。研究重点是不同压缩 projector 在固定下游模型和训练协议下的报告质量变化。

BindFormer 的想法是先形成图像证据组，再读取 finding、location/laterality、signal、morphology/extent 等角色信息，并通过概念库提供读取提示。结果表里的 `BindFormer (Vanilla)` 指该方法本体配置。四档目标压缩率是 4×、8×、16×、32×；MLP 是 1× 不压缩参照，按表格设计在每个区块重复一遍。

代码入口概览：

- `src/models/projectors/`：projector 实现；BindFormer 在 `bindformer/` 子目录。
- `configs/projector/`、`configs/training/`、`configs/evaluation/`：方法、训练和评测配置。
- `src/cli/`：训练、评测、压缩 sweep 和 BindFormer sweep 入口。
- `src/training/`、`src/evaluation/`、`src/data/`：训练、指标计算、清单和数据加载。
- `scripts/baselines/`：各 baseline 的独立启动脚本。
- `runs/`：配置、日志、checkpoint、预测、指标和 provenance 结果。
- `Pipeline_0929/`：研究设计、数据处理、指标定义、baseline 设计和实现验收说明。

完整评测方案见 `Pipeline_0929/EVALUATION_METRICS_AND_NEUROQA_PLAN.md`；projector 路线见 `Pipeline_0929/PROJECTOR_ROUTES_AND_BASELINES.md`；训练与复现说明见 `Pipeline_0929/NEUROVFM_MLP_TRAINING_PIPELINE.md` 和 `Pipeline_0929/BINDFORMER_DESIGN_AND_IMPLEMENTATION_PLAN.md`。

## Excel 列名和指标口径

各压缩区块使用完全相同的列。`A` 是 Method，`B–L` 是指标，`M` 是评测范围/状态说明。

| 列 | 表头 | 读取字段 | 数值范围和方向 |
|---|---|---|---|
| B | BLEU-4 ↑ | `overall.bleu4` | 0–100，越高越好；corpus BLEU |
| C | ROUGE-L F1 ↑ | `overall.rougeL` | 0–100，越高越好 |
| D | GREEN ↑ | `overall.green` | 0–1，越高越好；若个别报告格式修复失败，按有效报告的 `valid_only_mean` 填写，并在 M 列注明有效数/总数、失败数及 coverage；失败报告不计分，也不进入均值分母 |
| E | RadGraph-XL Simple F1 ↑ | `overall.radgraph_simple` | 0–1，越高越好 |
| F | RadGraph-XL Partial F1 ↑ | `overall.radgraph_partial` | 0–1，越高越好，主结构化指标之一 |
| G | RadGraph-XL Complete F1 ↑ | `overall.radgraph_complete` | 0–1，越高越好 |
| H | RaTEScore ↑ | `overall.rate` | 0–1，越高越好 |
| I | BERTScore Precision ↑ | `overall.bertscore_precision` | 0–1，越高越好 |
| J | BERTScore Recall ↑ | `overall.bertscore_recall` | 0–1，越高越好 |
| K | BERTScore F1 ↑ | `overall.bertscore_f1` | 0–1，越高越好；按逐报告 F1 求均值 |
| L | Answer CE ↓ | `overall.answer_ce` | 越低越好；teacher-forced CE，不是自由生成质量指标 |
| M | Evaluation status / scope | 手工根据 provenance 核对 | 写明 split、评测样本数/患者数、是否完整及特殊状态 |

BLEU-4 和 ROUGE-L 使用 0–100 标度，其余评分指标保留 0–1 原始数值；不要统一乘 100，也不要计算或填写自行合成的总分。RadGraph-XL 的 simple、partial、complete 三列不可互换。BERTScore 的 precision、recall、F1 也分别填写，不能将 F1 当作分类 macro/micro-F1。

## 结果纳入规则

1. 正式结果取匹配 checkpoint 的完整 test 评测产物，优先读 `evaluation/.../metric.json` 中的 `split`、`scope`、`evaluated_samples`、`patients`、`checkpoint_sha256`、`overall`、`partial_metrics`、`skipped_metrics`。交叉核对同目录 `generation/metrics.json`、`scoring/metrics.json` 和 `predictions.jsonl` 的覆盖数与指标值。
2. 只有 `scope=full_split` 且预测覆盖了该 run 声明的全部目标 test 清单时，才能把状态写作完整 test。`status=complete` 但 `scope=subset_smoke` 或 `fractional_sample` 仍是 smoke，不能填进正式结果表。
3. 必须同时核实方法名和压缩率：读取对应的 `resolved_config.yaml` / `pipeline.json` / `provenance.json`，确认 `projector.name`、`target_compression_ratio`、checkpoint 路径和 hash。不能仅凭目录名或 shell 脚本名决定方法/档位。
4. 不以 train CE、训练中验证生成、单例 smoke、validation、旧 checkpoint 或另一档位的产物填正式 test 格。验证集用于选 checkpoint，不和 test 混写。
5. 指标若缺失、非有限或被跳过，留空并在 M 列说明原因。GREEN 按成功评分的报告计算：个别报告在格式修复后仍失败时，排除这些失败报告，以其余 `status=ok` 报告的均值 `valid_only_mean` 填入 D 列；失败报告不计零分、不进入分母。M 列必须注明 valid、failed、expected 和 coverage；GREEN 的评测状态仍可标记为 partial。若没有任何有效 GREEN 分数，则 D 列留空。
6. 同一 baseline 的压缩率 r 结果只写到该区块对应的方法行。MLP 是不压缩 1× 控制，在 4×/8×/16×/32×区块重复展示时，写入同一份 MLP full-test 数值，并在 M 列标明“1× MLP control repeated”。
7. 保留 Excel 既有方法行顺序及拼写。不要插入行或重排方法；新增列必须同步更新四个指标表头行。表头行是 3、20、36、52。
8. 数值格显示四位小数即可，但保存原始浮点数。任何评测范围变化（例如 BraTS-only 与 BraTS+ISLES+WMH）要在 M 列明确说明；不同数据范围不能无标记地直接比较。

## 本机核验快照（2026-10-15，21:15 PDT 启动重跑）

- MLP 无压缩正式训练：`runs/baselines/mlp2x_gelu_joint_lora_20261012T170359963434Z/r1/joint_lora/training_state.json` 显示 epoch 10、global step 1,230、stop reason `max_epochs`。
- MLP 正式 test：`runs/baselines/mlp2x_gelu_joint_lora_20261012T170359963434Z/r1/evaluation/best_test/metric.json`，BraTS-only test 560/560、140 patients、`scope=full_split`。本项目全来源 test 是 622 条；当前这份 MLP 数字不含 ISLES/WMH。
- MLP 其余指标已进表；GREEN 按 558 条有效报告的均值填写（0.2689），2 条 `format_repair_failed` 排除在均值外。M 列应注明 `558/560`、失败数 `2`、coverage `0.9964` 和 run 状态 `partial`。可查 `partial_metrics.green`；不能把该分数描述为 560 条均完整评分。
- 先前错误的 Stage A-only 正式目录 `runs/bindformer_stage_a/bindformer_20261015T163356538290Z/`（8×/16×/32×）和 `runs/bindformer_stage_a/bindformer_20261015T163428425067Z/`（4×）已删除；`runs/bindformer_stage_a/acceptance_v1/` 与 `runs/bindformer_acceptance/` 是工程预检/验收材料，保留。
- 已修正 BindFormer 入口，现按相同 `joint_lora` 协议重跑：`bindformer_joint_lora_gpu1_20261015T211700Z` 在 GPU 1 跑 4×；`bindformer_joint_lora_gpu3_20261015T211700Z` 在 GPU 3 顺序跑 8×、16×、32×。两个 sweep 均默认完整评测 BraTS test 560 例及 `all` 指标。启动时 4× 与 8× 已进入训练，16×/32× 等前序档完成后顺序运行。各自 `sweep_status.json`、train/evaluation logs 和 `evaluation/best_test/metric.json` 是状态及结果来源。
- 对照已完成 MLP `r1/training_config.yaml` 和新 BindFormer `r4/training_config.yaml`：除预期的 projector 方法/压缩率与 run 路径外，training、data、preprocessing、encoder、decoder、task、evaluation 配置完全一致。计划日程同为 1,230 步，其中 projector warm-up 77、joint warm-up 77、joint cosine 1,076；LLM 冻结，视觉 LoRA rank 16 / alpha 32。
- 因新 BindFormer run 尚未产出正式 test 指标，表中对应格保持空白；运行完成后再依据对应 ratio 的完整 test `metric.json` 填写。

已填写的 MLP 数值直接取 `metric.json` 的 `overall`。四个 MLP 行是同一 1× 实验的重复参照，不代表 MLP 分别训练了四种压缩档位。
