# NeuroVFM → 不压缩两层 MLP → Llama-3.1-8B-Instruct：训练方案

更新日期：2026-10-12（用户已确认；连续 finetuning 代码已实现，验收记录见第 14 节）。任务：RadGenome-Brain 单序列 3D MRI Findings generation。

**2026-10-12 顺序实验入口补充**：[`scripts/baselines/`](../scripts/baselines/README.md) 已提供 11 个方法的 Bash 入口，统一显式加载 `joint_lora.yaml`；压缩方法依次执行 4/8/16/32× 的独立训练，每档结束评测其最低验证 CE 的 `best.pt`。MLP 只执行一次不压缩训练。默认完整 BraTS test、全部现有指标，可传 `--ratios 8 32`、`--eval-split val`、`--eval-sample-fraction` 与训练 `--set` 覆盖。`--dry-run` 只核对真实清单、headers、预算与日程，不加载模型或创建实验目录。任一步失败停止，输出配置、日志与状态。该入口未改变本文件的训练日程，也未启动正式 10-epoch 训练；验收及中断恢复方式见链接中的说明。

该批入口已完成 193 项测试及 41 组真实 6-step 训练 smoke，训练与预算检查全部通过。两个 6 层 Perceiver 的方法 preset 启用已有的激活重计算（`checkpoint_blocks=true`），解决 4× 长输入反向 OOM，不改变结构或 token 预算。生成验收单独记录：39 组短生成流程通过；Flamingo 8× 恢复 192-token 上限后仅非空/评测流程通过，原始 prediction 仍为重复标点；Concat 32× 仍为空生成，但同一次评测的 answer CE=13.8264，训练梯度和参数更新均有证据。不能将这些结果称为有效报告质量通过。11 个 baseline 的准确结构、OOM 历史、显存和类别覆盖建议见 [路线文档第 13.2、16 节](PROJECTOR_ROUTES_AND_BASELINES.md#16-smoke-问题复核单卡预算与类别覆盖2026-10-12-pdt)。详见 [本轮验收记录](../runs/archive/engineering/baseline_launchers_20261012T161710Z/verification.json)。

**当前实现与用户确认的覆盖项（优先于下文原设计）**：框架已落在根目录 `src/`，默认改为 `native_grid + minimal_pad`，通过一个 `preprocessing.spacing_policy` 参数切换 native/2mm/4mm。训练默认只选择 BraTS；可用 `--datasets brats isles` 等参数选择来源，`all` 表示三组一起用，评测默认继承 checkpoint 的训练来源范围，也可单独覆盖。视觉独立逐体积编码；报告独立逐样本解码，用户确认暂缓混合长度批量生成优化。配置、数据接口、可微 Encoder、MLP、冻结 Llama、两阶段 Trainer、checkpoint/resume、生成与评价均已实现。完整使用命令和验收证据见 [项目 README](../README.md)。下文保留原始实验推导；其中“4 mm 默认”和“待实现”不再代表当前工程状态。

**2026-10-12 已确认并实现的连续 finetuning 方案**：BraTS-only，一次连续训练最多 10 个 epoch；前 6.25% optimizer updates 仅线性 warm-up projector，随后 6.25% 同时从零线性 warm-up projector 与视觉 LoRA，剩余 87.5% 联合 cosine decay。Projector 与视觉 LoRA 的峰值 LR 均为 `1e-5`；视觉 LoRA 为 `rank=16, alpha=32, dropout=0.05`。阶段切换保留 projector 权重及 Adam 状态；两段 warm-up 结束后启用 `min_delta=0.005, patience=3` 的验证 CE 早停。两段 warm-up 可显式传 optimizer steps，未指定时按比例计算；rank、alpha 和最大 epochs 均可传参。完整规格见第 8 节，代码实现见第 13 节，测试及 smoke 证据见第 14 节。入口为 `configs/training/joint_lora.yaml` 与 `scripts/run_full_mlp_uncompressed.sh`；旧独立 A/B 模式仍保留用于复现。

native train/val 几何审计：2,476 个序列 header、805 个检查、39,903 个逐标签六邻域连通分量，丢失分量/标签体素均为 0。真实本地 8B 模型的 A、B-LoRA、B-Full 梯度/更新/冻结参数/重载检查已完成；GPU3 能容纳 native 8,775 token，全视觉参数训练累积 16 次时峰值分配 22.03 GiB。报告质量、过拟合和视觉依赖必须与工程连通性分开判断：初次 8 样本 `1e-4` 短 smoke 出现 loss 上冲，仍需诊断，不能称为已完成正式训练。Test 未用于预处理选择或训练诊断。

2026-10-04 核对补充：第 4.4 节更正了“两次默认过滤”的潜在误解；官方有两个可执行筛选的位置，但本地标准推理路径不会默认重复执行两次。几何抽查结果另见 `assets/data/RadGenome-Brain/neurovfm_background_spotcheck.json`。

本文件记录确认的实验设计，以及已落盘的数据划分、指代修订标注和审计。已发布 `single_sequence_findings_coref_v2`（220 条文本修订）。旧方案的运行结果位于 `runs/full_mlp_uncompressed_20261009/`；新方案已实现并进行了工程 smoke，不能将旧运行结果或短 smoke 视为新方案的收敛结果。部分标签仍需复核。第 8 节为已确认并实现的配置和早停机制，不是已验证的最优参数；旧训练配方保存在文末所列备份中。

## 0. 新对话的实现入口与状态

本文件是下一轮构建 preprocessing / training / evaluation 框架的自包含规格。项目根目录为 `/mnt/sda/jyzhang/VLM_Projector`；正文中的项目路径均相对于此根目录。已有文件与下文“待实现”模块必须区分，不能把建议配置或命令描述成已经实现、训练或验证。

- **任务**：单个 MRI 序列的完整 3D 体积 → 该序列英文 Findings。不是多序列综合报告，也不是疾病分类。
- **模型**：本地 `assets/models/neurovfm-encoder` → 自建不压缩两层 MLP → 本地 `assets/models/llama3.1-8b-instruct`。3DINO 仅预留适配接口，不是当前默认 Encoder。
- **固定约束**：全程冻结 LLM；不做背景 token 筛选或池化；不截断正常图像/文本输入；不用 GT mask、疾病队列名或参考报告构造输入提示。
- **当前配置**：用户确认 native-grid + 最小 padding；2 mm / 4 mm 为单参数切换的比较分支，官方 4 mm + 裁剪单独保留。
- **训练主线**：第 8 节规定的最多 10-epoch 双 warm-up + 联合 cosine + 早停，LLM 始终冻结。首段结束后直接沿用当前 projector 和优化器，不回退到旧 A-best。独立 projector-only 与 B-Full 保留为另行安排的对照，不属于本次主线预算。
- **已有实现**：`src/` 下的数据、预处理、模型适配器、任务、配置、CLI、训练器和评价管线，`tests/` 契约测试；当前验收明细见项目 README 与 `runs/archive/engineering/implementation/status.json`。
- **监督入口**：只读取 `assets/data/RadGenome-Brain/training_manifests/single_sequence_findings_coref_v2/{train,val,test}.jsonl`。不要从历史 raw/v1 清单自动回退。
- **标签边界**：33 条指代复核记录未排除；另有本轮发现的序列名称矛盾，见第 3.6 节。当前可据现有版本搭建框架和做冒烟检查，但不得将其称为全部标签已完成临床认证。
- **实现路线**：先完成第 11 节的数据/模型接口与验收，再训练。先由 train/val 确定预处理和优化配置，最后冻结方案评估 test。

NeuroVFM backbone 采用 Vol-JEPA 图像自监督预训练；报告生成是后续接入 LLM 的应用。不要将本地视觉权重描述为“通过报告生成预训练的 Encoder”。[官方模型卡](https://huggingface.co/mlinslab/neurovfm-encoder)

## 1. 本轮确定的六项修改

1. 不设置项目级 `max_sequence_length=4096`，不截断正常输入；按实际长度构建序列，batch 内动态 padding。仍检查模型真实上下文容量和显存，不把容量上限当作需要填满的长度。
2. 全量检查并修订单序列标注中的不明确指代；先备份，再建立证据、自动校验、生成独立版本并覆盖活动逐模态 JSON；禁止把其他序列才可见的发现直接搬入当前序列目标。
3. 将三个有跨集合检查的 MEN 患者全部归入 train，已备份并修改活动 split JSON 和 manifest，未删除其原有训练检查。
4. 首轮使用 NeuroVFM 官方物理空间重采样与 patch 对齐流程作为基准；明确区分重采样、裁剪和背景过滤，正式训练前评估小病灶与边缘损失，再按第 4.7 节比较更细采样方案。
5. 后续主实验默认一直不做背景过滤：关闭预处理删 token，不给 Encoder 传额外选择索引；保留全部网格及空间坐标，同时预留显式启用未来背景策略的代码接口。
6. 采用下文的固定、带序列信息的 Findings prompt。借鉴 AutoRG-Brain 的基线提示与 M3D 的 Findings 任务表达，但明确约束为单个 3D MRI 序列。

## 2. 数据单位、划分和本轮已完成的操作

### 2.1 一条样本是什么

一条样本 = 一个检查中的一个 MRI 序列对应的完整 3D 图像 + 该序列的 `modal_wise_finding`。例如 `BraTS-GLI-00006-000-t2f` 是一个 FLAIR 体积，不是一张 2D 切片，也不是整套多序列检查。

| 来源 | 当前参与单序列监督的序列 | 序列 Findings 数 |
| --- | --- | ---: |
| BraTS GLI | T1W、T1C、T2W、FLAIR | 920 |
| BraTS MEN | T1W、T1C、T2W、FLAIR | 920 |
| BraTS MET | T1W、T1C、T2W、FLAIR | 948 |
| ISLES22 | ADC | 250 |
| WMH | FLAIR | 60 |
| 合计 | 每条输入一个体积 | 3,098 |

原始划分共 3,408 个序列 ID，图像和对应病灶 mask 均能解析到本地非空文件。原标注缺少 250 个 ISLES DWI 和 60 个 WMH T1 的单序列 Findings，因此这 310 条不进入当前监督清单。不能为了补齐它们而直接使用 `global_finding` 或 `impression`；那会改变监督任务。

`global_finding` 是病例级、多序列综合观察，`impression` 是综合诊断结论。它们可以辅助人工审核同病例的引用关系，但不出现在模型的输入 prompt 中。当前不进行疾病分类、多序列融合或完整病例报告生成。

### 2.2 新划分：所有冲突患者都放入 train

活动版本：`patient_disjoint_v1_all_conflicts_to_train`。这是本地修改的划分，不是论文原始 benchmark 划分；后续报告结果必须注明。

| 患者 ID | 原 train 检查：保留 | 从 val/test 移入 train 的检查 | 移动序列数 |
| --- | --- | --- | ---: |
| BraTS-MEN-00074 | `00074-007` | test → `00074-005` | 4 |
| BraTS-MEN-00932 | `00932-009` | val → `00932-006` | 4 |
| BraTS-MEN-01246 | `01246-002` | val → `01246-003` | 4 |

三个患者共六次检查、24 个序列，现在全部在 train。实际从保留集移动的是 12 个序列；此前建议删除的另 12 个训练序列没有删除。

| 统计项 | Train | Val | Test |
| --- | ---: | ---: | ---: |
| 修改前：全部序列 | 2,382 | 338 | 688 |
| 修改后：全部序列 | **2,394** | **330** | **684** |
| 修改后：有单序列 Findings | **2,177** | **299** | **622** |
| 修改后：检查 / case ID | 707 | 98 | 202 |
| 修改后：公开患者 ID | 701 | 98 | 202 |

依据公开 ID 检查，train/val/test 患者交集均为空。训练集内同一患者的多次检查仍保留；按患者聚合统计，不能把这些检查当作完全独立患者。不同源数据集可能存在未披露别名，公开 ID 审计不能证明这类未知重合不存在。[BraTS-MEN 的患者与检查编号说明](https://www.nature.com/articles/s41597-024-03350-9)

已写入或更新的文件，路径相对 `assets/data/RadGenome-Brain/`：

- `annotations/train_val_test_split.json`：活动划分。
- `sequence_manifest.jsonl`：同步更新 split，保留全部图像/mask 路径。
- `split_migration.json`：12 个移动 ID、前后数量、当前文件 SHA-256、备份位置。
- `verification.json`：当前文件覆盖与患者划分检查。
- `training_readiness_audit.json`：当前概要；旧审计放入 `historical_audit_before_patient_migration`，其中的旧排除建议已失效。
- `training_manifests/single_sequence_findings_raw/{train,val,test}.jsonl`：保留原文的历史清单。
- `training_manifests/single_sequence_findings_coref_v2/{train,val,test}.jsonl`：本轮同步生成的活动目标清单，含修订文本、版本及复核状态。
- `backups/20261003T105706Z_patient_split/`：修改前的 annotation 目录、split、manifest、审计与数据说明文档。`manifest.json` 保存每个备份文件的 SHA-256。

这些 raw 清单没有假装解决文本语义问题。**2,177/299/622 是有原始目标文本的样本数，不等于已经人工确认标签质量的样本数。**

### 2.3 Dataset 必须遵守的读取契约

当前 JSONL 字段为 `split, id, cohort, case_id, image, mask, has_modality_report, patient_id, finding, target_version, annotation_source, text_audit_status, coreference_revised`。**目前没有独立的 `sequence` 字段**；由 dataset adapter 按以下规则生成并校验，禁止从报告内容猜序列。

| cohort | 样本 ID 示例 | 序列解析 |
| --- | --- | --- |
| BraTS_GLI / BraTS_MEN / BraTS_MET | `BraTS-GLI-00006-000-t2f` | 最后一个 `-` 后为 `t1n/t1c/t2w/t2f` |
| ISLES22 | `sub-strokecase0106_ses-0001_adc` | `_adc` 对应 ADC；当前监督清单不含 DWI |
| WMH | `Singapore_69_FLAIR` | `_FLAIR` 对应 FLAIR；当前监督清单不含 T1 |

- `image` / `mask` 相对于 **`assets/data/`**，例如 `BraTS/GLI/...nii.gz`、`ISLES/sub-...`、`WMH/Singapore/69/pre/FLAIR.nii.gz`。不能在 `RadGenome-Brain/` 下直接拼接这些路径。
- `annotation_source` 相对于 **`assets/data/RadGenome-Brain/`**；以样本 `id` 查询相应 JSON，要求与 `finding` 完全一致。
- `cohort/patient_id/case_id` 仅用于索引、分层和去泄漏，不能进入 LLM prompt。`mask` 只允许 QA/评价使用；训练 batch 默认不读取它。
- 启动时验证 ID 唯一、路径存在、文本非空、版本固定、split/患者交集、序列解析以及 active annotation 一致性。非法输入明确报错，不跳过后继续宣称处理了完整数据集。
- 每条样本只选择一个体积；不能直接把一个 BraTS 检查目录交给会扫描全部 NIfTI 的 `load_study()`，否则可能将其他序列和 segmentation 一并当输入。
- `review_policy=include_with_flag` 是保留当前 2,177/299/622 清单的默认读取方式；任何排除策略生成独立 manifest 和数量，不覆盖已有 split 或 active annotations。不得把 `requires_review` 自动当作已通过，也不得擅自静默过滤。
- 默认按序列均匀抽样、不放回遍历；暂不引入 cohort/class weighting。记录 BraTS 占比和每个来源的指标，避免总体结果掩盖小来源表现。

### 2.4 RadGenome 来源范围参数（当前实现）

训练、预处理和 Stage A 特征缓存使用 `data.datasets`；[基础配置](../configs/base.yaml) 默认为 `[brats]`。可传一个或多个来源，`all` 等价于三组全选。BraTS 选择会同时包含 `BraTS_GLI`、`BraTS_MEN`、`BraTS_MET`；ISLES 选择对应清单中的 `ISLES22`。名称大小写不敏感，也接受 `ISLES22` 别名。

| split | 默认 BraTS | ISLES | WMH | `all` 合计 |
| --- | ---: | ---: | ---: | ---: |
| train | 1,960 | 175 | 42 | 2,177 |
| val | 268 | 25 | 6 | 299 |
| test | 560 | 50 | 12 | 622 |

命令行示例：

```bash
# 默认省略 --datasets 时只训练 BraTS；也可显式传入
scripts/train_projector.sh mlp2x_gelu a runs/brats_stage_a --datasets brats

# 多数据集或三组一起训练
scripts/train_projector.sh mlp2x_gelu a runs/isles_wmh_stage_a --datasets isles wmh
scripts/train_projector.sh mlp2x_gelu a runs/all_stage_a --datasets all

# 全流程 MLP launcher：RUN_TAG 后传 --datasets 和一个或多个来源；省略时默认 BraTS
scripts/run_full_mlp_uncompressed.sh my_run --datasets brats isles
scripts/run_full_mlp_uncompressed.sh my_all_run --datasets all

# checkpoint 评测默认沿用训练范围；--datasets 只覆盖本次评测范围
scripts/evaluate_checkpoint.sh --checkpoint runs/.../checkpoints/best.pt \
  --output-dir runs/.../evaluation/isles_test --split test --device cuda:0 --datasets isles
```

筛选在 manifest 读取入口执行。训练目标、生成 reference、expected-ID 覆盖校验和各 split 评测范围都来自所选样本；逐条 `finding` 仍会与其 `annotation_source` 中的活动 `modal_wise_finding.json` 核对。当前主线没有独立的 GT 提取 CLI。未选择来源的目标文本和图像路径不会进入样本构建。resolved config、训练 `selection.json`、评测 `pipeline.json` 会记录训练范围、评测范围及分 split 计数。旧 checkpoint 没有 `data.datasets` 字段，按其生成时的旧默认值 `all` 解释；Stage B 初始化和续训不能悄悄改变训练来源范围。

## 3. 单序列标注的指代修订：已执行 `single_sequence_findings_coref_v2`

### 3.1 扫描、修订结果与当前目标

扫描全部 3,098 条非空逐模态报告，覆盖 `aforementioned`、`described above`、`same dimensions`、`previous sequences`、`same lesion` 等措辞。合并宽规则和旧审计得到 **405 条候选，实际修订 220 条**；185 条候选未修改。候选包括正常段内指代与缺项说明，不能直接当作错误数。

| 来源 | 非空逐模态报告 | 实际修订 |
| --- | ---: | ---: |
| BraTS GLI | 920 | 59 |
| BraTS MEN | 920 | 81 |
| BraTS MET | 948 | 75 |
| ISLES22 | 250 | 0 |
| WMH | 60 | 5 |
| 合计 | 3,098 | **220** |

修订覆盖 train / val / test 中的 152 / 22 / 46 条。**活动 `annotations/*/modal_wise_finding.json` 已被独立修订版覆盖**，新训练清单位于 `training_manifests/single_sequence_findings_coref_v2/`，其中 `finding` 与活动 JSON 一致。样本数仍为 **2,177 / 299 / 622**，患者划分、图像和 mask 不变；`global_finding`、`impression` 不变。没有补造原本缺失的 DWI/T1 目标。

旧 `single_sequence_text_audit.json` 的 44 条外部引用 / 238 条合并候选属于**修订前的较窄规则审计**，保留为历史，不能当作当前残留问题数。规则扩大以及删除冗余指代措辞，使本次修改数高于原来的 44 条。

### 3.2 实际执行的修订 pipeline

1. **先备份**：`backups/20261004T044743Z_modality_coreference_v1/` 保存 26 个源文件及 SHA-256 清单，包含全部原报告、原清单与文档；修订脚本从这个固定备份读取原文。
2. **按同次检查建立证据组**：用完整 case ID / 扫描编号匹配当前目标、其他序列 Findings 和 global finding。不按 JSON 顺序认定“上一模态”，不跨患者或不同检查，不用 impression 填写影像表现。
3. **分类与最小修改**：本段已有事实则只删除冗余引用；缺失部位、尺寸或明确被引用的结构影响，则仅展开来源支持的对应属性。不复制其他序列的信号、增强结论或最终诊断；“未描述”不能改为“正常”。事实展开是逐病例整理的精确替换，不是对所有文本进行自由生成。
4. **先写证据 JSON**：记录原文、候选全文、同检查上下文、每个修改片段、证据路径/ID/原文/哈希、理由和剩余问题。
5. **自动校验**：核验备份与证据哈希、检查 ID、精确编辑重放、新增尺寸来源、原有尺寸和左右侧/信号/否定词保留情况；再检查残留引用。程序不能证明临床语义完全正确。
6. **独立版本与发布**：先输出五个独立修订 JSON，检查活动文件仍与备份或当前版本一致，再覆盖活动逐模态 JSON；保存修改日志并同步新训练清单，保留 raw 清单。
7. **训练前读取状态**：新清单的 `target_version` 固定为本版本，保留 `coreference_revised` 与 `text_audit_status`。不把 `requires_review` 混同为已完成影像核查；正式实验前冻结复核处理策略，所有 Projector 共用。

### 3.3 实例与不能自动补齐的内容

`BraTS-GLI-00006-000-t2f` 已有左额叶和高信号描述，仅将 `the aforementioned lesion` 改为 `the lesion`。同检查 `t1c` 中“上述尺寸及周围影响”已展开为约 `76 × 72 × 69 mm`、周围水肿、双侧侧脑室前角受压与中线右偏；本条已有的环形强化保留。展开依据同检查 T1W / global 文字，不代表重新对图像做了测量。

仍有 **33 条带复核说明的报告**（train 28 / val 1 / test 4），其中 7 条相对原始标注做了有依据的局部修改，26 条仍为原始文本。它们包含原始跨序列增强用语、报告缺项和部分来源不足，类别重叠；并非 33 条都存在未解决的“同上”，也不能断言全部是错误。

其中两条确实无法可靠补全的引用保留原文并标记：`BraTS-MET-00090-000-t2w` 缺少 T2W 原始观察依据；`BraTS-MET-00113-000-t2w` 只有 “likely visible” 及不足以支持确定断言的尺寸引用。不能从增强报告推定它们在 T2W 的可见表现。另外 GLI-00242 的脑池细节、MET-00266 的额外边界特征、MET-00244 的笼统非增强特征，只处理来源明确的部分，剩余限制如实记录。

文本校验不是临床验证。复核队列目前未被自动排除，因此 2,177/299/622 仍是有目标文本的清单数量。若后续排除或人工重标，另出目标/清单版本及数量，不悄悄改 split；val/test 修订不得参考模型预测。所有 Projector 固定同一目标版本，可另用 raw references 做补充评分。

### 3.4 二次核对与当前可用状态

独立解析患者 ID、扫描编号和证据 key，核验 v1 的 220 条修订、238 个编辑及 455 次证据引用：未发现跨患者或跨检查。文本复核另收紧 6 条：GLI-00819 的 FLAIR、MEN-00743-001 的 T2W 不再从笼统的“相似特征”展开水肿/受压；MET-00016 的 T2W/FLAIR 信号限定到实性部分；MET-00537 的 T2W/FLAIR 使用不暗示一个病灶跨两个部位的措辞。详见 [二次核对结果](../assets/data/RadGenome-Brain/modal_wise_revision_doublecheck.json)。

v1 已备份至 `backups/20261004T053047Z_before_modality_scope_v2/`（32 文件）；独立 v1 标注与清单也保留。活动五个逐模态 JSON 已覆盖为 v2，新版清单逐 ID 与活动文本一致。

33 条是**序列报告**，不是 33 名独立患者。它们没有被自动排除；`requires_review` 只是元数据标记，训练加载器若不显式处理，会照常读入。直接读活动 `annotations` 或 coref_v2 清单会用新版目标；读取 raw、coref_v1 或旧的文本缓存仍会使用旧目标。目前没有启动正式训练器，不能把文件覆盖等同于已验证所有未来加载路径。

### 3.5 产物与复现

路径相对 `assets/data/RadGenome-Brain/`：

- [证据文件](../assets/data/RadGenome-Brain/modal_wise_coreference_evidence.json)：405 条候选、原文及逐条来源，记录发布前提案。
- [实际修改记录](../assets/data/RadGenome-Brain/modal_wise_revision_log.json)：220 条已应用记录，含 `original_text`、`revised_text`、`changes`、证据及状态。
- [自动校验结果](../assets/data/RadGenome-Brain/modal_wise_revision_validation.json)：前后哈希、完整性检查及发布后一致性。
- [待复核清单](../assets/data/RadGenome-Brain/modal_wise_coreference_review_queue.json)：33 条剩余复核说明。
- `annotation_versions/single_sequence_findings_coref_v2/`：独立修订版本。
- `training_manifests/single_sequence_findings_coref_v2/`：与活动目标同步的三个训练/评价清单。

详细字段、规则、例子、复现和回退步骤见 [逐模态指代修订记录](MODAL_FINDINGS_COREFERENCE_REVISION.md)。初版实现见 [v1 修订脚本](../archive/non_pipeline/scripts/repair_radgenome_coreferences.py)，当前复核/发布使用 [v2 脚本](../archive/non_pipeline/scripts/doublecheck_radgenome_revisions.py)。上游的报告拆分背景见 [AutoRG-Brain 附录 B.2](https://arxiv.org/html/2407.16684#A2.SS2)，不能仅凭字段名假设每条已经自足。

### 3.6 新一轮文档核查发现的语义问题：不能把“33 条”当成所有标签问题的上限

本轮只读核查发现 train 的 `Singapore_69_FLAIR`，输入为 `WMH/Singapore/69/pre/FLAIR.nii.gz`，活动目标却写作：

> Periventricular white matter shows multiple punctate T1W hyperintense signals. No midline shift is observed.

其清单状态仍为 `coreference_pass_completed_not_clinical_certification`，不在上述 33 条 `requires_review` 中。这是**输入序列与文本所称序列矛盾**，不是新增的跨患者引用证据；需要回溯原始报告/同检查资料并按序列可见性审核。不能仅做 `T1W → FLAIR` 字符串替换就声称已修好。

本轮未改动该标注、原 33 条复核队列或 manifest。下一轮 audit 应另产出 `modality_semantics_audit.json`，对输入序列名、跨序列词、增强、比较检查、缺项与否定范围做候选扫描；程序只标记，不自动生成新结论。候选与原复核队列按 ID 合并后记录处理政策；监督范围若变化，按第 3.2 节另出版本。原有版本可用于工程验证，但正式指标必须披露这些限制，另报告排除已标记目标的敏感性结果，不能根据模型预测挑选排除条目。

## 4. NeuroVFM 图像输入与预处理

### 4.1 官方基准固定物理尺度；Encoder 支持可变 patch 数量

NeuroVFM 使用可变数量的 patches。不同病例不必都变成 `112³` 或 `128³`，这些是此前 3DINO 讨论中的实验尺寸，不用于当前 NeuroVFM 主实验。

沿用本地官方代码的主要步骤：

1. 读取 NIfTI 的数据、spacing、origin、direction；检查维度、数值及空体积。
2. 重定向到 RPI；识别厚层/切片轴，核验该轴与后续数组转置一致。不能仅凭数组中较短的一维推断解剖方向。
3. 官方预训练/预处理基准的目标 spacing 为平面内 `1 × 1 mm`、切片轴 `4 mm`，本地预处理使用 B-spline 插值。4 mm 是预训练约定，不是 backbone 的物理间距硬限制；更细或保留现有网格的方案见第 4.6 节，尚未替代这里的基准设置。不同轴排列下不能把 SimpleITK 的 `(x,y,z)` 和网络的 `(D,H,W)` 混为一谈。
4. 将重采样结果居中裁到 patch 的整数倍：D 是 4 的倍数，H/W 是 16 的倍数。
5. MRI 按 0.5 / 99.5 百分位裁剪并缩放到 `[0,1]`；随后使用官方 MRI 统计量 `mean=0.3141, std=0.2623` 做一次标准化。不要混入 3DINO 的归一化配置，也不要重复标准化。
6. 使用 `(4,16,16)` patch 网格，保留坐标和所有 patches。

依据是本地 [预处理工具](../third_party/neurovfm/neurovfm/data/utils.py)、[MRI 强度与 tokenization 实现](../third_party/neurovfm/neurovfm/data/preprocess.py)、[Encoder 包装器](../third_party/neurovfm/neurovfm/pipelines/encoder.py)；远程对应 [NeuroVFM 官方源码](https://github.com/MLNeurosurg/neurovfm/blob/main/neurovfm/data/utils.py)。尺寸/spacing 检查与本实验的质量门槛是我们增加的，不应称为论文原样复现。

### 4.2 重采样是不是生成新切片？误差在哪里？

它是在新的**物理坐标网格**上，从原有体素推算强度：对新网格位置，按照原图的 origin、direction、spacing 找到对应位置，再用插值估计数值。一般会得到重新计算的切片；它不是生成式模型合成解剖内容，也一般不等同于从现有切片列表中抽取若干张。

如果新旧网格恰好整数对齐，有些新采样点落在旧体素中心，数值可以与旧采样一致；因此也不能把每一个新值都描述为邻近切片的平均。B-spline 是插值方法，不保证等效于对一个厚层体素所覆盖区域做体积平均，也不自动保证充分的抗混叠处理。

| 原始情况 | 变到 4 mm 切片 spacing 后的含义 | 风险 |
| --- | --- | --- |
| 原来约 1 mm | 沿此轴降低采样密度 | 细小病灶、边界及小血管等细节可能丢失 |
| 原来约 4–5 mm | 相近或稍密的网格 | 网格更密不代表获得真实的新层间信息 |
| 原来约 8 mm | 对中间位置插值 | 无法恢复扫描时没有采集的结构细节 |

所以不能预先保证“误差很小”。插值可能平滑边缘，B-spline 也可能产生局部过冲；降采样的主要问题是信息减少，不能只看图像是否更平滑。

随后还发生独立的中心裁剪。例如 `240×240×155`、1 mm spacing 的 BraTS 体积，按官方大小计算先成为约 `240×240×38`，再裁到 `240×240×36`，网络数组为 `36×240×240`。这一处裁剪与重采样是两种不同的信息损失来源。保留所有背景 patches 不会撤销这两步的损失。

MRI 的连续强度使用插值；离散病灶标签用于几何 QA 时采用最近邻，避免生成不存在的标签值。见 [SimpleITK ResampleImageFilter](https://simpleitk.org/doxygen/latest/html/classitk_1_1simple_1_1ResampleImageFilter.html)。

### 4.3 预处理质量检查

- 保存原始和处理后的 shape、spacing、方向、切片轴、裁剪起点、物理范围及变换；不覆盖原始 MRI。
- 统计所有序列处理后的 patch 数，检查最小尺寸与位置编码范围；遇到异常明确报错，不静默跳过。
- 利用已有病灶 mask 检查裁剪是否覆盖病灶、重采样后小连通病灶是否消失、物理病灶体积变化；抽查不同来源、厚薄层和不同大小病灶。
- 区分“插值/离散化造成 mask 体积变化”和真实信息损失；不能把最近邻 mask 体积比直接当作诊断准确率，也不能凭一个样本声明全数据无损。
- 不使用随机左右翻转、任意局部 crop 或强烈几何变换：当前报告含左右侧和全体积发现，增强必须与监督语义相容。
- 主实验先固定官方 spacing/crop；若质量检查显示明显的小病灶或边缘损失，先形成统一的预处理修订，再比较 Projector。不能让不同 Projector 悄悄使用不同输入质量。

### 4.4 背景过滤默认行为与两个可选筛选入口

背景过滤是官方 `StudyPreprocessor` 和预训练配置的默认设置，但不等于默认执行两次。2026-10-04 逐行复核的本地 `StudyPreprocessor.load_study()` 无论 `remove_background` 为何，都返回空的 `series_masks_indices`；`EncoderPipeline.embed()` 将空索引转换为 `masks=None`。因此，沿这条标准推理路径设置 `remove_background=False`，即可保留完整网格，不会在 Encoder 里再筛一次。

另一个可选入口是：调用者提供非空 `series_masks_indices`，或者直接给 backbone 传入 `masks`/`masks_enc`，这时会执行 token 索引选择。训练用 `VisionEncoder` 包装器的传参方式又不同。因此，我们自己实现训练适配器时仍显式禁止额外选择，作为保证完整输出的约束；不能把这个防御性约束描述为“必须拆掉两个默认过滤器”：

```python
# 设计要求示意；不是本轮已经实现的训练函数。
preprocessor.remove_background = False
batch["series_masks_indices"] = None

# 调用可微 backbone 时不传前景选择或随机 token 筛选。
features = encoder_core(
    normalized_tokens,
    coords,
    masks=None,
    masks_enc=None,
    cu_seqlens=full_grid_cu_seqlens,
    max_seqlen=full_grid_max_len,
)
assert features.shape[0] == sum(full_grid_token_counts)
```

实际调用需依照最终 adapter 的函数签名实现。关键是 `cu_seqlens/max_seqlen` 也按完整网格计算。LLM 的视觉位置 attention mask 全为 1；只有为组成 batch 新增的 padding 位置为 0。不要把“图像内背景”和“batch padding”混为一类。

GT 病灶 mask 只用于 QA/评价，不用于本 baseline 的 token 筛选、ROI 提示或模型输入。

2026-10-04 用官方图像预处理对 11 个选定体积做了几何抽查：默认过滤后的 token 数减少约 37.83%–90.37%；保留 patches 对重采样、裁剪后的非零标注区域覆盖为 37.86%–96.55%。这不是全数据统计，也不是模型敏感度或准确率。过滤前后的定义与每个样本的数据保存在上述 JSON 中。官方按强度百分位定义前景，再仅保留所有体素都属于前景的 patch，因此可能删除混合了组织和低强度体素的 patch。取消过滤也改变了相对默认预训练的输入分布，是否改善最终生成须单独评价。

本地 `neurovfm-encoder` 权重通过 `load_encoder()` 构建视觉 backbone，直接输出每个保留 patch 的 768 维特征；它不包含完整 VLM 路径下的 `VisionConnector`。后者另有 Perceiver resampler（示例 SFT 配置为每序列 64 个 queries）和 MLP。本实验不调用该 connector，只接我们自己的逐 token MLP。

### 4.5 已预留的背景过滤接口；主实验固定关闭

实现：[token_filter.py](../src/data/token_filter.py)。这是独立的逐体积接口，尚未接入未实现的 NeuroVFM 训练 adapter；它不会自动修改第三方预处理默认值，未来接线必须同时执行第 4.4 节的设置。

```python
from src.data.token_filter import select_tokens

# 官方预处理必须显式 remove_background=False，先取得完整 patch 网格。
selection = select_tokens(tokens, coords, policy="none")
assert selection.tokens is tokens
assert selection.coords is coords
assert selection.original_count == len(selection.kept_indices) == tokens.shape[0]
# 用 selection.tokens / coords 构造 batch；backbone 不再传选择索引。
```

`none` 保留全部 token、坐标、原顺序和梯度。将来只有显式选择 `policy="custom"` 并提供 `selector(tokens, coords) -> bool[N]` 才执行筛选；回传 `kept_indices` 用于追溯，拒绝错误长度、非布尔或全删除的 mask。未来策略只依据输入图像，不能使用 GT 病灶 mask。token 和坐标始终同步，之后重算 batch 的 `cu_seqlens/max_seqlen`；不要在 Encoder 中重复筛选。当前未启用任何具体背景算法，也没有加入合并或池化。

本轮接口测试覆盖默认全部保留、筛选后的坐标/梯度对齐、非法 mask 与隐式启用的拒绝；这不代替 NeuroVFM 真实输入端到端测试。完整 MRI 输入在这里指保留整幅预处理体积的全部网格，不表示取消此前已说明的物理重采样和 patch 对齐裁剪。

### 4.6 本地间距审计、可修改范围与保留现有网格的候选方案

全量读取 RadGenome 实际引用的 3,408 个 NIfTI 头信息，明细见 [nifti_spacing_audit.json](../assets/data/RadGenome-Brain/nifti_spacing_audit.json)。这覆盖当前任务输入，不代表扫描仪原始 DICOM 层厚，也不涵盖下载整包中未被 RadGenome 引用的额外影像。

| 输入来源 | 当前文件的平面内间距 | 当前文件第三维/层向间距 | 数量 |
| --- | --- | --- | --- |
| BraTS GLI / MEN / MET，四个序列 | 1 × 1 mm | 1 mm | 2,788 个体积 |
| ISLES ADC / DWI | 约 0.875–2.0 mm，两平面轴相同 | 2 mm：196；4.4 mm：1；4.8 mm：52；5 mm：1 | 每个序列 250 个体积，当前仅 ADC 有逐模态监督 |
| WMH Utrecht，FLAIR / 配准后 T1 | 约 0.9583 × 0.9583 mm | 3 mm | 每个序列 20 个体积 |
| WMH Singapore，FLAIR / 配准后 T1 | 约 1 × 1 mm | 3 mm | 每个序列 20 个体积 |
| WMH GE3T，FLAIR / 配准后 T1 | 约 1.2 × 0.9766 mm | 3 mm | 每个序列 20 个体积 |

WMH 统计的是 manifest 指向的 `pre/` 文件，当前仅 FLAIR 有逐模态监督。BraTS MEN 头信息的空间单位字段未填写，数值为 1/1/1；按其数据论文的 1 mm 各向同性发布标准解释为 mm，审计保留这一单位依据。各向同性数据不能仅凭 spacing 推断原始采集轴；不少 ISLES/WMH 含斜位方向，保留方向矩阵，不能把第三维简单等同于严格解剖轴。当前有监督的 3,098 个体积中，3,044 个（98.26%）层向 spacing 小于 4 mm，54 个大于 4 mm。

[NeuroVFM 论文 Methods](https://www.nature.com/articles/s41591-026-04497-1#Sec2) 明确说明预训练前将序列重采样为 1×1×4 mm、4 mm 沿 DICOM 采集轴，随后切成 4×16×16 **体素**的 patch。两种“4”不同：默认 patch 的物理覆盖约为 16×16×16 mm。保持相同权重而将层向 spacing 改成 2 或 1 mm，patch 的物理覆盖分别变为 8×16×16、4×16×16 mm；这会改变相对于预训练的输入尺度，不能直接断言更准确。

结构上可以在更细网格运行：patch embedding 仍接收每个 patch 的 1,024 个数值，输出特征仍为 768 维，MLP 无须改变。位置编码使用整数 patch 索引而非真实毫米坐标，不会自动补偿 spacing 改变；本地配置的网格上界为 D<128、H/W<192（均为 patch 索引范围），还需核对 token 数、显存及实际 forward。不要把 `patch_d_size=4` 随 spacing 一起改小，那会改变 patch embedding 的输入权重维度。

**代码注意点：**本地 `StudyPreprocessor.target_spacing` 目前只被保存，没有传入 `load_image → preprocess_image`；底层 `data/utils.py` 直接设置 `target_spacing[z_dim] = 4`。因此只在构造函数传 `(1,1,1)` 不会真正改变输入。实现变体时应在项目 adapter 中明确控制重采样、轴排列及 patch 对齐，并检查处理后 spacing；不能依赖一个未接通的参数。

保留细节的候选方案是保留当前文件网格：BraTS 保留 1 mm，ISLES 保留每例实际间距，WMH 保留 3 mm 及实际平面间距；仅做轴置换/翻转、既定强度归一化、按 patch 倍数做最少边界补齐。轴置换/翻转可以不插值，但强制将斜位体积旋转至正交解剖网格通常需要重采样，应作为另一个明确选择。此方案保留现有采样，不能恢复数据集发布前已丢失的信息，也不能使 5 mm 的采集凭空获得 1 mm 的真实分辨率。BraTS 本身已经过配准和各向同性重采样，参见 [BraTS MEN 数据论文](https://www.nature.com/articles/s41597-024-03350-9)。

以 240×240×155、1 mm 的 BraTS 为例，保持官方向下裁到 patch 倍数的规则时，层向 4/2/1 mm 分别产生 2,025/4,275/8,550 个视觉 token。若保留完整现有 1 mm 网格并将深度从 155 最小补齐到 156，则为 **8,775** 个 token；新增一层 padding，不是补成一个大立方体。这里是几何计算，未执行这些条件的 Encoder/LLM 训练。

更细采样的输入信息保留与冻结 Encoder 的尺度适配需要分别验证。建议把官方 4 mm 作为参考，对比保留现有网格的方案；2 mm 可作为 BraTS 的中间实验点，不要求把原本更厚的 ISLES/WMH 强行插值到 2 mm。先用同一 MLP 和 train/val 选择预处理策略，再固定给所有 Projector；对比时一致处理背景、边界、训练步数与目标版本。阶段 B 的视觉适配可能改善尺度变化的适应性，但尚无本项目结果证明。

### 4.7 spacing 选择：默认、候选与实施顺序

**决策：首轮先用 4 mm 跑通；正式大规模比较 Projector 前，用固定 MLP 做 spacing 对照。不能预先认定 4 mm 最优，也不把所有原网格输入当作无代价的提升。** 下列策略是待实现的明确配置项，不能只修改未接通的 `target_spacing` 参数。

| `spacing_policy` | 平面内 spacing | 层向 spacing | 用途 |
| --- | --- | --- | --- |
| `official_4mm` | 1×1 mm | 所有来源 4 mm | 首轮默认及预训练尺度参考 |
| `brats2mm_other_native_z` | 1×1 mm | BraTS 2 mm；ISLES / WMH 保留当前层向 spacing | 优先做的细节/成本折中实验 |
| `native_grid` | 当前文件的实际间距 | 当前文件的实际间距 | 后续保留现有体素采样的实验 |

`brats2mm_other_native_z` 是本项目候选方案，不是官方设定，也不是所有病例都变成 2 mm。它仍在平面内插值；不能称为完全保留原网格。若要单独归因 BraTS 的 4→2 mm 效果，可额外让 ISLES/WMH 暂时沿用 4 mm，形成只改变 BraTS 的对照；不得把混合策略的全部变化归因于一个 spacing 参数。

全部原网格输入存在额外的尺度变化：官方 patch 物理范围为 D/H/W=16/16/16 mm；BraTS 1 mm 网格为 4/16/16 mm；ISLES 2 mm 各向同性网格为 8/32/32 mm。整数位置编码不会自动补偿这些差异。原网格策略仅减少进一步的空间重采样，不消除已有发布预处理、强度归一化和 patch embedding 带来的信息变化。

| 方面 | 统一到 4 mm | 保留当前网格 |
| --- | --- | --- |
| 预训练尺度 | 与默认物理采样接近；不保证在本数据上最佳 | 尺度分布变化，冻结 Encoder 下收益不确定 |
| 薄层/小病灶 | BraTS 1→4 mm、ISLES 多数 2→4 mm 有降采样损失风险 | 保留现有采样，有利于细节进入 Encoder，但不保证被有效利用 |
| 原来厚于 4 mm | 会插值到稍密网格，不增加真实信息 | 不增加伪装成新采集的切片信息 |
| 训练成本 | 通常有利于 BraTS 不压缩 MLP 的成本控制 | BraTS token 大幅增加；各来源增减幅度取决于全部三轴，不能一概而论 |
| 微调 | 仍有影像域、背景策略等差异 | 视觉 LoRA/全量微调可能适应尺度，不能保证，也不能恢复已丢失的细节 |

按**现有官方 floor 重采样尺寸 + 向下裁剪**，BraTS `240×240×155` 的层向 4/2/1 mm 对应 2,025/4,275/8,550 tokens；若按当前尺寸公式后改最小 padding，则约为 2,250/4,500/8,775。它们是几何估算，不是显存实测；重采样输出的 FOV 规则改变时须重算。MLP 输出 token 数相同，所有 token 都进入冻结 Llama，LLM 仍需为视觉输入保留反向路径。

执行次序：

1. 用 train 小样本做 `official_4mm` 前向、梯度、生成、显存与过拟合检查；这不属于正式效果比较。
2. 在 train/val 上做全量几何统计和代表性影像 QA，重点抽查小病灶、厚薄层、边缘病灶和斜位方向。test 可做不用于决策的文件完整性检查；不能用其病灶/报告结果调预处理。
3. 若裁剪造成有用组织/病灶丢失，统一切换 `boundary_policy=minimal_pad`，重新缓存所有对照分支，保留官方 crop 作为参考。决定 spacing 时保持边界、强度、背景策略一致。
4. 固定 MLP、患者划分、目标版本、prompt、样本暴露量、随机种子和解码方式，比较 `official_4mm` 与折中方案。每个 spacing 分支都有自己的阶段 A，再从它自己的 A checkpoint 进入 B；不能只看冻结 Encoder 结果就否定细间距。
5. spacing 对照先固定 B 的微调模式（首轮 LoRA），不要同时把 4 mm+LoRA 与 2 mm+全量微调当作单变量比较。LoRA/全量对照见第 8.4 节。
6. 由验证集答案 CE、固定生成指标、按来源的病灶描述错误及资源开销选择最终预处理；每个候选内用最低 val CE 选 checkpoint。选择依据写入 `selection.json`，不能只看训练 loss 或汇总分数。
7. 资源允许再做 `native_grid`；固定方案后才进行不同 Projector 的主实验和最终 test。未完成的候选明确记为未运行，不把实现了选项等同于获得有效结果。

阶段 A/B 内必须保持同一 preprocessing 版本。不要在 A 使用 4 mm、B 直接改 1 mm后，将性能差异全部解释为“解冻 Encoder 的收益”。如需这种继续训练策略，单独命名为分辨率迁移实验。

### 4.8 预处理实现细节和 QA 产物

在项目 adapter 中读取 `load_image(..., preprocess=False)`，显式控制重采样、方向、边界和强度，再送 tokenization。首先以 `official_4mm + official_crop` 核对本地官方输出；不要先调用官方默认预处理后再次重采样。第三方原文件保持作为参考，适配逻辑放在项目内。

- **轴与物理空间**：显式记录 SimpleITK xyz、数组 DHW、轴置换/翻转及层向轴。已发布各向同性 NIfTI 通常不能恢复扫描仪原始采集轴；对这些文件使用已记录的发布网格约定（本地 BraTS 和 ISLES 各向同性体积的第三数组轴作为层向），不要声称从 DICOM 推断。斜位保留 direction；RPI 轴置换/翻转不能等同于无插值去斜位。保持当前代码的输出 direction，不新增未记录的配准。
- **重采样**：连续 MRI 用 B-spline；mask 在相同输出 reference grid 上用最近邻，只服务 QA。与原网格相同的 target 可走恒等路径。记录 size/origin/direction 的计算，官方参考分支保留 `floor(size*old_spacing/new_spacing)` 行为；如改为覆盖完整 FOV 的输出网格，必须新建 preprocessing 版本。B-spline 不是已验证的抗混叠滤波器，不把降采样后的平滑当成无损。
- **边界独立配置**：`official_crop` 向下中心裁到 D 为 4、H/W 为 16 的倍数；`minimal_pad` 向上最小补齐，并记录每侧补量。padding 不能补回重采样网格本身已丢掉的内容。所有对照统一边界政策。
- **强度与 padding 顺序**：百分位统计在边界 padding 前的有效图像域计算，按既定 0.5/99.5 截断、min-max 后再补值 0；随后只执行一次官方 mean/std 标准化。标准化后的 0 背景对应 `(0-0.3141)/0.2623`，不能在标准化后随意补数值 0 却宣称等价。常量/非有限体积明确报错。
- **完整网格**：整幅图像的背景 patches 以及边界混合 patches 都保留；图像边界补齐与 batch 的无效序列 padding 区分。前者形成有效视觉 token，后者 attention mask 为 0。禁止借 padding 名义删除含真实体素的边界 patch。
- **标签 QA**：先核对原图和原 mask 的物理对齐，记录不一致；不要假定数组同形状就同空间。不同来源的 label 值语义不同，配置到 cohort adapter，不能一律认为 `mask>0` 都是同一疾病标签。若只做“所有异常标签并集”的几何统计，应按数据说明明确集合，并与按标签统计分开。
- **产物**：每个预处理版本输出 `preprocessing_manifest.jsonl`、`qa_summary.json`、少量配对 QA 图。记录图像/配置 SHA-256、原/新 shape 与 spacing、单位依据、轴与 affine、插值/FOV、crop/pad、强度统计、有效网格 token 数、坐标范围、非有限值和错误原因。
- **预处理 patch 缓存（当前实现）**：原 MRI 不覆盖；共享 patch 张量正式保存在 `assets/preprocessing/<preprocessing_version>/<key前两位>/<key>.pt`，版本由所有数值变换选项及预处理源码哈希确定，单条 key 由源图像 SHA-256、cohort 和版本组成。缓存默认启用；训练、评测、预处理 CLI 均可用 `--no-use-cache`（别名 `--no-cache`）绕过读写，也可用 `--use-cache` 显式启用。缓存开关和存储根目录不进入 preprocessing version；改变 spacing、边界、强度等数值变换选项会得到新版本，不能命中旧张量。
- **Encoder 特征缓存（独立机制）**：Stage A 的最终视觉特征保存在 `cache/features/`，与 patch 缓存分离；它必须按图像/样本身份、Encoder 权重与配置、预处理版本及输出契约隔离。Stage B 更新视觉 LoRA 或全参数时不能复用旧特征。新连续训练主线关闭该缓存。patch 张量在 Encoder 前，缓存 key 不包含 Encoder 权重；不能把两类缓存的 key 混为一谈。

病灶 mask 重采样后的体积变化、连通分量消失、crop 丢失与实际 MRI 可见性分开报告；nearest-neighbor mask 的体积比不是临床检出率。几何 QA 与生成评价共同用于验证集选择，禁止利用 GT mask 定制模型输入 ROI。

## 5. Encoder、MLP 与 LLM 的衔接

本地模型：

- `assets/models/neurovfm-encoder/`：本地配置 `embed_dim=768`、`depth=12`、`prefix_len=0`。
- `assets/models/llama3.1-8b-instruct/`：`hidden_size=4096`、`max_position_embeddings=131072`。

两层不压缩 MLP：

```text
一个 3D MRI 序列
    ↓ NeuroVFM 预处理与完整 patch 网格
N × 1024 原始 patch 数值 + 三维 patch 坐标
    ↓ NeuroVFM Encoder
N × 768 视觉特征
    ↓ Linear(768,4096) → GELU → Linear(4096,4096)
N × 4096 视觉 embeddings
    ↓ 替换 Llama user 消息内的 <image> 占位段
冻结的 Llama-3.1-8B-Instruct
    ↓
当前 MRI 序列的英文 Findings
```

两个 Linear 均带 bias 时 MLP 共 19,931,136 个参数。中间维度 4096 是该 baseline 的选择；以后可做 Linear / 3 层 MLP 或不同宽度消融，但首轮固定它。

`N = (D/4) × (H/16) × (W/16)`。例如 `36×240×240` 对应 `9×15×15=2025` 个视觉 token，MLP 后仍是 2025 个。病例可以有不同的 N。4096 是每个视觉/文本 embedding 的通道维度，不是 token 数；MLP 不压缩也不代表用满 Llama 的 128K 上下文。

## 6. 固定 prompt：`single_sequence_findings_v2`

### 6.1 参考来源与适配原则

AutoRG-Brain 附录 B.1 为 OpenFlamingo / Med-Flamingo 的报告生成基线设置了放射学助手身份、图像占位符和脑 MRI 报告任务；区域版本还提供目标区域。这是其对照模型的提示说明，不能称作所有 AutoRG-Brain 训练都采用这一 chat prompt。[附录 B.1](https://arxiv.org/html/2407.16684#A2.SS1)

M3D 的 `Caption_templates` 使用多种简短的 caption/findings/report 指令；原文件并不是一份长篇结构化 MRI prompt。我们借鉴它的“根据影像描述 Findings”表达，自行补充单序列范围、序列身份和输出要求。[M3D 官方模板](https://github.com/BAAI-DCAI/M3D/blob/main/LaMed/src/dataset/prompt_templates.py)

下面是**本项目的适配设计**，并非逐字复刻上述工作。正式 baseline 只用一个模板；随机改写 prompt 留作独立实验，避免干扰 Projector 比较。

### 6.2 System 与 user 消息

System：

```text
You assist with describing brain MRI findings. Base your statements on the supplied image and acquisition information.
```

User：

```text
<image>
The input is one 3D brain MRI volume.
Sequence: {sequence_name}.

Write a concise, self-contained Findings paragraph in English for this sequence.
Describe visible abnormalities, including their location and laterality, signal intensity, morphology, and any evident edema or mass effect. Include any other relevant imaging observations supported by this sequence. Mention contrast enhancement only for a post-contrast input.
Do not assume access to other sequences or prior examinations. Do not invent measurements, comparisons, or unobserved findings. Avoid references such as "as above" or "the same as on T1W".
Return only the findings, without a separate impression, differential diagnosis, or recommendations.
```

`sequence_name` 固定映射：

| ID / 序列标识 | Prompt 中的名称 |
| --- | --- |
| `t1n` / T1W | T1-weighted MRI without contrast |
| `t1c` / T1C | post-contrast T1-weighted MRI |
| `t2w` / T2W | T2-weighted MRI |
| `t2f` / FLAIR | fluid-attenuated inversion recovery (FLAIR) MRI |
| ADC | apparent diffusion coefficient (ADC) map |

MRI 的具体序列身份要传给文本任务；Encoder 的 broad modality 则仍是 `mri`。ADC 有不同的信号意义，不能都笼统写成 T1/T2。

不在 prompt 中加入 GLI/MEN/MET 队列名、病例 ID、诊断、病灶位置、GT mask 或参考报告。它们可能暴露答案。也不为凑长 prompt 增加没有采集的年龄、症状、扫描参数或比较检查。尺寸只有在证据支持时描述，不要求逐例输出毫米数。

更完整的 prompt 本身不能修复训练答案里的“同上”或跨序列事实；第 3 节的标签修订需与 prompt 一起版本化。

v2 相对先前文档的 v1 仅补充“其他由当前序列支持的相关观察”，使属性列表不是封闭清单，落实此前讨论中的开放式表述；不修改 Findings 标签。旧 v1 的 token 长度统计不适用于当前 prompt，正式配置必须保存全文及哈希，所有比较统一使用 v2。

### 6.3 Llama chat template 与长度策略

- 从本地目录加载 tokenizer 和 causal LM，可使用本地安装的 Transformers；这不需要联网调用 Hugging Face 服务。
- 使用本地 tokenizer 的 `apply_chat_template` 正确构建 system/user/assistant 分界，保留原模型的 BOS/EOT 约定，不再手动重复添加特殊符号。
- `<image>` 是由 adapter 识别和替换的占位段，不是把数千个浮点向量转换成文本。把该段替换成 `N×4096` 的视觉 embeddings，再拼接其余文本 embeddings。
- 训练时拼入目标 Findings 及 assistant 结束标记；推理时仅拼入 system/user 与 assistant 开始位置。
- 不设置 4096 等实验输入截断上限，`truncation=False`，按实际有效长度生成 attention mask/position IDs，batch 内动态 padding。
- 检查图像展开后的总位置数：训练是视觉 + 模板/提示 + 答案；推理还需为生成留出余量。若超过模型实际 131072 位置容量，明确失败并报告样本；不悄悄丢 token，也不改写模型配置扩大容量。
- 不把每个样本 padding 到 131072。显存不足优先通过微批量、checkpointing、特征缓存或参数分片处理，不自动截断图像/答案。

Prompt、图像位置和 padding 的 `labels` 都为 `-100`，只监督 assistant 的 Findings 和结束标记。需要精确定位 assistant 答案边界，不能用一个未经验证的固定 token 偏移。使用 causal LM 的 `labels` 接口时由模型内部完成 shift，不手动再 shift 一次。

历史 v1 prompt 的 192 token、原始答案中位数 47 / 最大 105 仅是旧版本统计，不能用作当前 v2 prompt / coref_v2 目标的长度预算。实现时对当前所有清单重新生成 `token_length_audit.json`；图像占位展开、assistant 结束符和 batch padding 后再次检查真实长度。约两千视觉 token 的样本总长通常也是几千，而不是填满 128K。

本轮已用本地 tokenizer 在内存副本中注册单一 `<image>`，离线核对 v2 FLAIR 提示：含一个占位 token 共 202 tokens，去掉该占位后文本前缀为 201 tokens（含 system/user 模板及 assistant 开始标记）。当前 coref_v2 答案独立编码、不含 EOT 时，train 的 min/median/max 为 6/48/105，val 为 12/44/87，test 为 13/46/100。它们用于预估，不取代实际完整 chat 边界和视觉展开后的长度检查，也未把 tokenizer 副本写回模型资产。

## 7. 损失函数与梯度路径

首轮只用答案 token 上的自回归交叉熵：

\[
\mathcal L=-\frac{1}{\sum_i T_i}\sum_i\sum_{t=1}^{T_i}
\log p_\theta(y_{i,t}\mid V_i, P_i,y_{i,<t}).
\]

这里 `V_i` 是完整视觉 embeddings，`P_i` 是模板及提示，`T_i` 包含有效答案和结束 token。实现时明确选择有效答案 token 的归一化方式；多卡和梯度累积需要按整个有效 batch 的有效 token 数加权，避免直接平均不同长度 microbatch 的 mean loss 却声称实现了上述公式。

第一版不加 segmentation、contrastive、disease classification 或所谓“信息保持”辅助损失。先验证图像确实影响生成，再为新 Projector 加独立、可消融的辅助项。

两个阶段都设置 LLM 参数 `requires_grad=False`，但训练时**不能把 LLM forward 放进 `torch.no_grad()` 或 `inference_mode()`**。损失仍要通过冻结的 LLM 对视觉 embeddings 求导，传回 MLP，第二阶段再传回视觉 LoRA 或全量视觉参数。冻结的是 LLM 权重更新，不是整个函数对输入的导数。

NeuroVFM 的 `.embed()` 带 `@torch.inference_mode()`：阶段 A 可用于离线提特征，若直接接训练 MLP，需要让特征成为普通 tensor，例如保存后重新加载，或在 inference context 外 clone。阶段 B 必须通过可微 backbone forward，不能继续调用这个推理入口，否则视觉适配器得不到梯度。

## 8. 当前确认的 finetuning 方案（2026-10-12）

本节替换旧的独立 A 训练 10 epochs → 从 A-best 新建 optimizer → B-LoRA 训练 5 epochs 配方。当前目标是确定一个合理、可复用的训练基线，先支持后续 projector 比较，不做 rank/LR 网格搜索。新主线从预训练 NeuroVFM、冻结 Llama 和新初始化 projector 开始；**最多 10 epochs 包含两段 warm-up，不是在旧 A 的 10 epochs 上追加训练；触发第 8.5 节早停时可以提前结束。**

### 8.1 数据、训练预算与双 warm-up

数据固定为 `single_sequence_findings_coref_v2` 的 BraTS 子集：train 1,960 条序列、490 次检查、484 位患者；val 268 条序列、67 位患者；test 560 条序列、140 位患者。保持单序列 Findings、native-grid + minimal padding、完整视觉 token 和现有 prompt，不改变标签或 split。

参考 [Resolution Meets Reduction §3.3–3.4](https://arxiv.org/html/2608.08713v1#S3.SS3)：标准 projector 的两段 linear warm-up **各占总优化步数 6.25%，合计 12.5%**，之后 cosine decay。论文为 PerceiverResampler 延长第一段的做法不用于当前 MLP。论文适配的是 **LLM LoRA、冻结 VE**，本项目适配的是 **VE LoRA、冻结 LLM**；借鉴训练日程，不直接搬用论文绝对 LR。

以当前单卡 microbatch 1、effective batch 16、每个 epoch 尾部不足 16 条仍执行一次更新为准：`ceil(1960/16)=123` updates/epoch，按 10-epoch 上限计划 `T=1230` updates，实际完成步数可能因早停减少。两段各 `floor(0.0625*T+0.5)=77` 步，其余 `1076` 步。数据范围、batch、world size 或调试步数改变时，按实际 sampler/累积窗口重新计算并保存解析后的步数，**不在训练代码写死 77**。

参数规则：`training.max_epochs` 控制总 epoch 上限；`training.projector_warmup_steps` 和 `training.joint_warmup_steps` 分别控制两段的 optimizer updates。每段显式 steps 非 null 时优先使用该整数；为 null 时才按对应的 `projector_warmup_ratio` / `joint_warmup_ratio` 计算（默认均为 0.0625，按 `floor(ratio*T+0.5)` 取整）。允许一段指定 steps、另一段使用 ratio；记录各段实际采用的来源。默认方案为下面的 77/77/1076。这里不是传 warm-up epoch，也不是 microbatch 次数。调整 max_epochs 后，显式 steps 不变，ratio 模式的 steps 随计划总预算重新计算。

| 阶段 | optimizer update 编号（从 1 开始） | 步数 | Projector LR | VE LoRA LR / 状态 |
| --- | --- | ---: | --- | --- |
| Projector 单独 linear warm-up | 1–77 | 77 | `0 → 1e-5` | 冻结；LR 为 0 |
| Projector + VE LoRA 联合 linear warm-up | 78–154 | 77 | **重新从 `0 → 1e-5`** | **`0 → 1e-5`** |
| 联合 cosine decay | 155–1230 | 1,076 | `1e-5 → 0` | `1e-5 → 0` |

约相当于 0.625 + 0.625 + 8.75 epochs；按 optimizer update 切换，不等到 epoch 边界。从联合 warm-up 起，两组 LR 始终相同。采用各段包含起点/终点的离散约定：每段 warm-up 首步使用 0、末步使用峰值；cosine 首步使用峰值、末步使用 0；这些更新计入总预算。日志记录**本次 optimizer.step 实际使用的 LR**，避免记录 scheduler.step 后的下一步 LR 而混淆边界。

第一段结束只重启 LR warm-up，**保留 projector 参数、Adam step、exp_avg 和 exp_avg_sq**，不重新初始化 projector，不加载历史 A-best。LoRA 使用标准初始化（初始增量为零），第一段不更新；第二段开始才启用其梯度和训练时 dropout。论文正文没有明确给出 projector 在阶段边界是否归零及优化器状态是否重建，这两个细节属于本项目确认的实现选择。

### 8.2 固定参数及选择依据

| 项目 | 当前确认配置 |
| --- | --- |
| Projector | 不压缩两层 MLP，768 → 4096 → 4096；全部 19,931,136 参数可训练 |
| Projector 峰值 LR | **`1e-5`**，两段 warm-up 使用同一峰值 |
| VE LoRA 峰值 LR | **`1e-5`**，与 projector 同步联合 warm-up/cosine |
| LoRA rank / alpha / dropout | **`16 / 32 / 0.05`** |
| LoRA targets | 全部 12 blocks 的 `mixer.qkv` 和 `mixer.proj`，共 24 个精确匹配的线性层 |
| LoRA bias / scaling | `bias=none`；普通 `alpha/r=2`；`use_rslora=false` |
| Encoder base / LLM | 全程冻结；不加 LLM LoRA |
| Optimizer | AdamW，betas `(0.9, 0.999)`，eps `1e-8` |
| Weight decay | `0.01`；bias 和一维参数不 decay |
| Microbatch / effective batch | 每卡 1 / 全局 16；单卡累积 16，DDP 按 world size 推导 |
| Precision | forward BF16 autocast；projector、adapter 与优化器状态保持 FP32；LLM BF16、不量化 |
| Grad clip | 全局 norm `1.0`，在整个累积窗口完成后执行 |
| Seed / reproducibility | 主运行 seed 42；沿用 deterministic preset |
| Loss | 仅 assistant 答案及结束 token 的 CE，按全局有效答案 token 数归一化 |
| 训练预算 / 选择 | 最多 10 epochs；cosine 阶段按第 8.5 节早停；最低 val answer CE 选 checkpoint，固定验证生成同步记录 |
| 缓存 | 新主线 `training.feature_cache=false`；可继续使用确定性的预处理 patch 缓存 |

**早停已由用户确认保留**：新主线使用第 8.5 节的阈值与耐心机制，替换此前“禁用早停、固定跑满 10 epochs”的建议。旧独立模式保留原有语义；新模式不重置或压缩既定 cosine 来适应实际早停时刻。

Projector 的 `1e-5` 在旧独立训练中已有稳定下降依据；联合阶段采用同量级是保守起点。没有证据证明 `2e-5` 必然过大，或两组相同 LR 一定优于不同 LR；相同 LR 也不代表两组实际权重更新幅度相同。低 LR 的可能代价是适应较慢，效果由本次 train/val 曲线判断。上述 weight decay/dropout 延续本地设置；参考论文的对应值为 0 / 0.1，不能声称整套参数完全复刻论文。

当前 fused qkv + output-proj 注入方式的 LoRA 参数量为：

`12 × 16 × [(768+2304)+(768+768)] = 884,736`。

约为 85,803,234 个视觉基础参数的 1.03%；projector + LoRA 共 **20,815,872** 个可训练参数。相对 rank 8 的全部可训练参数只增加约 2.2%。这里对整个 fused qkv 做 LoRA，**不是仅对 Q/V 分别做 LoRA**。Rank 16 增加权重更新的可表达空间，不直接增加视觉 token 或保证保存更多病灶信息；484 位患者也不能简单换算成某个最优 rank。

[医学视觉 LoRA 研究 Table 2](https://arxiv.org/pdf/2401.12215)中 rank 16 是与更高 rank 表现接近的合理候选；任务为胸片分类，只作为容量选择的间接依据。参考论文的 `r=32, alpha=64` 同样采用 `alpha/r=2`；本项目固定 16/32，不要求先完成多 rank 搜索。

### 8.3 梯度、缓存与资源约束

第一段：Encoder 保持 eval/no-grad，LoRA 参数冻结；Projector 可训练。第二段起：启用视觉 LoRA 梯度及训练时 dropout，Encoder base 继续冻结。LLM 始终冻结权重，但训练 forward 必须保留对视觉 embeddings 的梯度，不能整体放入 no-grad/inference-mode。

新连续主线全程关闭**最终视觉特征缓存**，避免在解冻后读取旧特征；Encoder 和 projector 输出每次重算。只复用输入影像、预处理参数和实现版本不变时的确定性 **patch 数值、坐标和几何元数据**。现有 `--use-cache/--no-use-cache` 只控制**预处理 patch 缓存**，两者不是同一参数。缓存标识已包含影像内容哈希、预处理参数/代码版本；修改 spacing、归一化等会生成新的缓存标识。

启用既有 LLM gradient checkpointing；训练 `use_cache=False`，生成可用 KV cache。LoRA 不会消除视觉/LLM 的反向激活，不能按 adapter 参数比例推断显存节省。真实 native 8,775-token 输入已完成短运行，验证了新阶段切换、rank 16 的梯度与峰值显存，证据见第 14 节。DDP 在构造 reducer 时先注册全部 projector/LoRA 参数，再恢复首段冻结状态，并启用 `find_unused_parameters=True`；首段仍使用 no-grad、adapter 无梯度。该稳定注册方式避免阶段边界重建 wrapper/optimizer，并经 Gloo 与 NCCL 跨阶段同步和续训测试验证。

### 8.4 与旧实验和独立对照的关系

- `runs/full_mlp_uncompressed_20261009/` 是旧独立 A/B 训练证据，保留原文件与 checkpoint；它的数据范围也不能默认为新 BraTS-only 范围。新配置使用新 run 目录。
- 原 `a/b_lora/b_full` 模式继续支持历史复现和另行安排的对照。B-Full 仍表示真正训练全部视觉参数，不是本次 rank-16 主方案；其训练日程与 LR 不由本节替其预定。
- 若要单独比较冻结 Encoder 时各 projector 的能力，应另跑**预算一致、充分训练的 projector-only 对照**。本方案首段只有约 0.625 epoch，不能把该 warm-up checkpoint 称为已收敛的 projector-only baseline。
- 官方 NeuroVFM `s2_finetune` 的冻结策略不同，不能直接用来替代此方案。新连续训练模式及参数传递已按第 13 节实现。

### 8.5 早停、最佳 checkpoint 与异常中止（用户已确认）

| 项目 | 最终规则 |
| --- | --- |
| Monitor | 完整 BraTS val 的 token-weighted answer CE，越低越好；每个完整 epoch 验证一次 |
| `training.early_stopping_min_delta` | **0.005**，CE 绝对变化量；是本项目初始阈值，不是已验证的统计显著性界限 |
| `training.early_stopping_patience` | **3** 次连续的、符合条件的验证无足够改善 |
| 开始监控 | 两段 warm-up 完成后，cosine 阶段首次完整 epoch 验证建立参考值；该次不计坏轮次 |
| 改善判定 | `current_val_ce < early_stop_reference - min_delta` 才算足够改善；更新 reference 并将 bad_count 清零 |
| 停止条件 | 未达到足够改善时 bad_count 加 1，达到 patience 停止；warm-up 期间及额外边界检查不累计 |
| Best checkpoint | 独立维护全程最低实际 val CE；只要降低就保存，不受 min_delta 限制 |
| 日程 | 仍按 max_epochs 的计划总步数定义 cosine；早停时不重新缩放曲线，不强制 LR 到零 |
| 数值异常 | loss/梯度/验证 CE 等出现 NaN/Inf 时立即异常中止，不能当作正常早停等待 patience |

维护两个独立值：`best_val_ce` 用于保存模型；`early_stop_reference` 用于判断持续改善。后者只在改善超过阈值时更新，允许多个小改善累计达到阈值。例如 reference=1.300，后续 1.298/1.296 暂不清零；再到 1.294 则累计超过 0.005，更新 reference 并清零。不能每次微小改善都移动 reference，导致累计收益被忽略。

默认 77+77 步 warm-up 在第二个 epoch 内结束：epoch 2 的验证建立早停基准，若 epochs 3/4/5 连续三次无足够改善，最早在 epoch 5 末停止。若用户改动 warm-up 或总预算，按实际 phase 确定起点，不写死 epoch 2/5。额外 smoke/边界验证不缩短 patience 的实际训练周期。

早停状态（是否已建立基准、reference、bad_count、触发原因/步数）随 checkpoint 一起保存，resume 必须延续计数。停止时保存 last 和停止原因；最终评测使用 best checkpoint，不能默认使用最后一轮。若未触发早停，则训练至计划上限。验证生成结果用于质量复查；不使用测试集或单个生成样本驱动早停。

## 9. 验证、生成与测试

1. **构建检查**：样本图像、序列身份和目标一致；三集合患者不重叠；图像 token 不丢；chat 拼接后各张量长度一致；只有答案位置产生 CE。
2. **梯度检查**：分别验证两阶段可训练参数和冻结权重，尤其验证阶段 B 没有误走 `.embed()` 的 inference-mode 路径。
3. **小样本过拟合**：从 train 固定取少量完整、无待解决引用的样本，确认 loss 下降且输出与输入相关。该步骤不能作为泛化指标。
4. **连续训练验证**：每个 epoch 记录完整 BraTS val 的 answer CE、固定验证子集的生成结果、峰值显存、视觉 token 数与训练步数；两个 warm-up 边界额外保存 checkpoint 和记录 phase/LR。全程统一按最低 val CE 选择 checkpoint，阶段切换不重置 best_val_ce；早停独立按第 8.5 节维护 reference/bad_count。
5. **生成**：`do_sample=False`、`num_beams=1`，按 Llama assistant 结束标记停止。可沿用 `max_new_tokens=192` 作为首轮输出防失控上限，并记录触顶率；它不是输入长度 4096 的替代，也不截断训练答案。若 val 显示频繁触顶，在查看 test 结果之前统一调整并冻结。
6. **结果指标**：一键评测默认报告 BLEU-4、ROUGE-L、GREEN、RadGraph-XL、RaTEScore、BERTScore P/R/F1 和 Answer CE，保留逐病例输出；省略 `--metrics` 即运行全部指标，也可显式指定子集。脑 MRI 的临床内容应单独审阅位置/左右侧、信号、病灶漏报、幻觉、增强与非增强混淆，不能直接把胸片专用指标当作唯一标准。
7. **分层**：按数据来源与序列统计，同时给出患者/检查聚合结果，避免一个多序列病例被当成多个独立患者而夸大稳定性。正式比较尽量至少 3 个随机种子，记录全部配置。
8. **视觉依赖检查**：在固定验证子集上比较正确图像、同序列内打乱图像和文本提示基线。检查模型是否只根据 prompt 或队列特征生成模板化报告；这类对照应与正常训练结果分开报告。
9. **最终测试**：所有决策由 train/val 完成后，使用冻结方案评估 test。当前主线 BraTS test 为 560 条序列、140 位患者；622 条是 `all` 范围数量，不能混用。若进一步审核导致变化，按正式版本重新公布数量。

新主线报告的是 projector 与视觉 LoRA 联合适配后的效果。若另跑充分训练的 projector-only 对照，单独列出预算与结果，不用短 warm-up 替代它，也不把 Encoder 的额外训练收益全部归因于 Projector。

### 9.1 必须固定的评价定义

- **CE**：仅 assistant 答案与结束 token，跨样本/设备累计 NLL sum 和有效 token 数再相除；报告总体与各来源。用于 checkpoint selection，不能把它当作全部临床质量。
- **BLEU-4**：固定 `sacrebleu` 的 corpus BLEU，默认 tokenizer=`13a`、case-sensitive、smooth_method=`exp`、effective_order=False、0–100 分；输出库版本和完整 signature。不要将单句 BLEU 均值写成 corpus BLEU。
- **ROUGE-L**：固定 `rouge-score` 的 `rougeL` F1，`use_stemmer=True`，逐样本均值并乘 100；不是 `rougeLsum`。保留原预测和原 reference，评分仅统一换行/空白，不事后删掉错误句子。
- **语义指标**：BERTScore/RaTEScore 已纳入一键评测默认指标，明确记录 scorer 权重、版本、语言/层配置及本地路径；缺少所选评分资源时直接报错，不能隐式联网下载或将缺失评分标记为完成。冻结配置见 [评价文档](EVALUATION_METRICS_AND_NEUROQA_PLAN.md)。胸片疾病标签工具不能直接替代脑 MRI 内容核查。
- **层级**：报告五个 cohort（BraTS_GLI/MEN/MET、ISLES22、WMH）及合并后的三个数据来源，并按序列报告。宏平均明确平均的是哪一层；只有可逐样本定义的分数（如 ROUGE-L）才直接做检查/患者内均值。
- **不确定性**：正式比较用固定种子、按 patient_id 成组的 paired bootstrap（建议 1,000 次），同一患者的所有检查/序列一起重采样；BLEU 在每次重采样后重算 corpus 分数。小来源的置信区间可能很宽，不能用大量序列假装独立患者更多。
- **复核敏感性**：总体结果保留默认目标范围，另报告事先冻结的待复核 ID 排除子集结果与数量；不是按预测优劣筛样本。对第 3.6 节新问题单独记录，不把原 33 条视为完整语义错误清单。
- **内容审阅**：固定验证样本名单，记录部位/左右、信号、增强、病灶漏报、无依据发现、尺寸和比较性措辞；人工审核与程序关键词候选区分。GT mask 无法独自验证报告中的所有临床事实。

### 9.2 推理与输出契约

推理只给 system/user、视觉 embeddings 和 assistant 开始标记；绝不拼入 reference。训练时 `use_cache=False`，纯推理时可 `use_cache=True`。本地 generation_config 默认含采样参数，必须显式覆盖为 greedy：`do_sample=False, num_beams=1, max_new_tokens=192`，清理无效的 temperature/top_p 采样设置。

读取本地 tokenizer/config 核验 `<|eot_id|>`、EOS 和 pad ID；停止列表至少覆盖 assistant EOT 与模型 EOS，不用一个假定的数值替代实际检查。当前 generation_config 的 eos 列表为 `[128001,128008,128009]`、pad 为 `128004`；需对加载到的 tokenizer 再验证这些 token 的语义。训练动态 right-padding；单样本推理先验收，批量 decoder-only 推理采用正确的 left-padding/position IDs。

对当前 Transformers 的 `generate(inputs_embeds=...)` 明确验证输入前缀、KV cache、position/cache_position 和返回 token 范围：不要假定返回结果一定含视觉前缀长度，错误地切掉生成正文。先比较 batch=1 与混合长度 batch 的 greedy 输出，确认没有跨样本污染。generation 触顶、空字符串、异常停止均记录，不静默删除失败样本；有失败时报告覆盖率并修复后重跑，不将部分结果冒充完整 test。

`predictions.jsonl` 每行至少包含 `id/patient_id/case_id/cohort/sequence/split/reference/prediction/target_version/text_audit_status/checkpoint_id/preprocessing_version/prompt_version/visual_token_count/prompt_token_count/generated_token_count/stop_reason/hit_generation_limit`；一键评测顶层 `metric.json` 保存指标总结果、范围、置信区间及明细路径，`generation/metrics.json`、`scoring/metrics.json` 保留版本、signature、覆盖率和分层明细；速度/显存分别标记 train/generation、设备、batch、缓存策略及测量方式。

## 10. 实施顺序与需要保存的产物

```text
已有：数据/标注/预处理/模型/独立 A-B 训练与评价框架，旧方案运行记录
本轮：用户确认最终方案 → 实现连续训练、参数入口、早停和 checkpoint
    → 配置/调度/冻结/动量/续训/早停/分布式测试
    → 真实模型短运行、独立进程恢复对照及启动/评测入口验收
    → 将命令、配置、结果与证据位置写入第 14 节
后续正式实验：新 run 目录 → 最多 10 epochs（允许早停）
    → 依据 val 选择 checkpoint → 冻结配置并评测 test
```


每个实验目录保存：数据 split 哈希、目标修订层版本、prompt 原文与版本、预处理参数及源码版本、Encoder/LLM 权重标识、Projector 配置、LoRA target 列表、随机种子、optimizer/scheduler 状态、checkpoint、训练日志、逐样本预测与评价。只保存 MLP/视觉适配器等可训练权重时，也必须保存能准确复原冻结 backbone 的版本信息。

后续增加其他 Projector、3DINO 或多序列任务时，通过 encoder / projector / decoder / task / preprocessing 五个独立适配接口扩展。变更 Encoder 必须切换其适用预处理；变更到多序列任务必须重新定义输入序列集、缺失序列策略和目标类型，不能在当前实验中隐式切换。

## 11. 框架接口、配置与验收

### 11.1 目录和接口

以下目录树是早期框架设计记录；当前实际代码入口以第 13 节和项目 README 为准。已有 PyTorch/OmegaConf/原生 DDP 实现，新方案沿用本地框架，避免照搬官方会训练 LLM 的 SFT system。

```text
VLM_Projector/
  pyproject.toml                         # 待建：安装本项目和 CLI
  configs/
    data/radgenome_single_sequence.yaml
    preprocess/{official_4mm,brats2mm_other_native_z,native_grid}.yaml
    models/{neurovfm_mlp_llama,3dino_mlp_llama}.yaml   # 3DINO 仅接口预留
    train/{stage_a,stage_b_lora,stage_b_full}.yaml
    eval/findings.yaml
  src/
    data/{manifest,dataset,collate,sequence_mapping,token_filter}.py
    preprocessing/{base,neurovfm,geometry,qa,cache}.py
    models/encoders/{base,neurovfm}.py
    models/projectors/{base,mlp,registry}.py
    models/decoders/{base,llama}.py
    models/multimodal.py
    tasks/{base,single_sequence_findings}.py
    training/{stages,optim,loss,trainer,checkpoint}.py
    evaluation/{generate,metrics,grouping,visual_controls}.py
    cli/{audit,preprocess,cache_features,train,evaluate}.py
  tests/                                # 只覆盖真实契约/梯度/边界风险
  assets/preprocessing/<version>/        # 共享 patch 张量；内容寻址，默认读取
  cache/features/                        # Stage A Encoder 特征缓存，和 patch cache 分开
  runs/<run_id>/{checkpoints,predictions,qa}/
```

核心契约：

1. `DatasetAdapter` 返回样本身份、单个 volume 路径、sequence、目标及复核状态。未来换数据集只替换该层；任务目标选择在 `TaskAdapter`，不能将 global/impression 自动作为单序列 fallback。
2. `Preprocessor` 返回 `patch_values[N,1024]`、`coords[N,3]`、shape/spacing/affine/变换与策略元数据。NeuroVFM patch 顺序是按 D/H/W 网格展平，patch 内为 C/D/H/W；复用或验证官方 einops 顺序，不凭 reshape 猜测。正常训练输出不包含 GT mask。
3. `EncoderAdapter` 返回逐体积 `features[N,768]`、coords、token 数及元数据；不调用 Perceiver、MIL head、CLS 池化或诊断 head。训练暴露普通可微 `forward()`，缓存另有显式 inference 入口。下游适配器通过 `feature_dim` 获取维度，不把 768 写死给未来 3DINO。
4. `Projector` 返回 `visual_embeds[M,llm_hidden_size]` 和长度；初始 MLP 强制 `M=N`，无 token 混合。以后压缩方法可以 M<N，但由统一接口报告输入/输出长度与参数量。
5. `DecoderAdapter` 负责 tokenizer、chat template、视觉占位替换、labels、attention mask/position IDs 和生成；本地 `from_pretrained(..., local_files_only=True)`，不使用在线 API，也不随意设置推理式 `device_map=auto` 代替训练分布式策略。
6. `TaskAdapter` 负责 sequence prompt、目标、损失监督区间和解码后的 Findings 格式；新任务（VQA/多序列）另建任务配置和 manifest，不侵入视觉 backbone。

Encoder 的 packed batch 要求 `cu_seqlens` 为 device 上 int32、起点 0、终点为总 patch 数、`max_seqlen` 为真实最大值；每个体积的 attention 必须彼此隔离，coords 逐体积从自身网格开始。NeuroVFM 的公共推理入口目前会传 `use_flash_attn=False`；训练不能未经检查沿用它。验证可微 FlashAttention varlen 路径，必要时以正确带 padding mask 的 SDPA 作对照，不能静默走显式 N×N 注意力后把 OOM 归因于权重大小。

### 11.2 已实现的连续训练配置与参数入口

已新增 `configs/training/joint_lora.yaml`，下例是相对于 base 的覆盖配置。`joint_lora`、两段 steps/ratio 与 min_delta 均已接入配置校验、CLI、Trainer 和 checkpoint。旧 `warmup_ratio/warmup_steps` 仅用于旧独立模式，不叠加到双 warm-up。

```yaml
data:
  datasets: [brats]
training:
  stage: joint_lora
  deterministic: true
  projector_lr: 1.0e-5
  encoder_lr: 1.0e-5
  projector_warmup_steps: null     # 显式整数优先；null 才使用下方比例
  joint_warmup_steps: null
  projector_warmup_ratio: 0.0625
  joint_warmup_ratio: 0.0625
  warmup_ratio: 0.0             # 关闭旧单段 warm-up，避免与双 warm-up 叠加
  warmup_steps: null
  max_epochs: 10
  early_stopping_patience: 3
  early_stopping_min_delta: 0.005
  microbatch_per_device: 1
  effective_batch_size: 16
  mixed_precision: bf16
  weight_decay: 0.01
  betas: [0.9, 0.999]
  eps: 1.0e-8
  grad_clip_norm: 1.0
  feature_cache: false
  init_checkpoint: null
  lora:
    rank: 16
    alpha: 32
    dropout: 0.05
    bias: none
```

LoRA 精确 targets 与 `use_rslora=false` 沿用当前 adapter 实现。启动时打印保存 resolved config、选中样本量、解析后的总步数/阶段边界、实际 LR、LoRA 参数量与冻结清单。配置覆盖顺序仍为 base → 按顺序传入的 YAML → `--set`；`--datasets` 最后覆盖数据范围。新主 launcher 不再追加 rank=8 或启动第二个独立训练进程，并完整透传 `--config/--set`。`training.lora.rank` 与 `alpha` 均可独立覆盖；更改 rank 不自动改写显式 alpha，若希望保持 alpha/r=2，则同时指定 alpha=2*rank。

已实现的显式步数参数示例（正式训练前可先追加 `--dry-run` 检查，不加载模型）：

```bash
projector-train --config configs/training/joint_lora.yaml \
  --config configs/training/deterministic.yaml \
  --datasets brats \
  --set training.max_epochs=10 \
  --set training.projector_warmup_steps=77 \
  --set training.joint_warmup_steps=77 \
  --set training.lora.rank=16 \
  --set training.lora.alpha=32 \
  --set training.early_stopping_patience=3 \
  --set training.early_stopping_min_delta=0.005 \
  --set training.run_dir=runs/brats_mlp_joint_lora_r16_new
```

`effective_batch_size = world_size × microbatch_per_device × accumulation_steps`；启动时推导并验证整除，记录 epoch 尾部不足一个累积窗口时的实际样本/token 数。loss 采用全局有效答案 token 归一化；测试框架自动梯度累积缩放和 DDP 梯度平均，避免重复除以 accumulation_steps/world_size。无放回 distributed sampler 若为对齐而重复样本，要记录；val/test 必须按 ID 去重，不能把 sampler 补齐项重复计分。

### 11.3 Chat 拼接、梯度和 checkpoint 的实现注意点

- 推荐在**tokenizer 的运行副本**中将 `<image>` 注册为单一附加 special token，并保存该副本及 ID 映射；不修改资产目录的原 tokenizer。不调整冻结 Llama 的 embedding/vocab 参数。对完整 chat tokenize 后，在 embedding lookup 前先剔除该占位 ID，再插入视觉 embedding；该新 ID 绝不能送入 LLM 词表查表或 labels。也可实现经严格验证的 token-span 替换，但同一实验不能混用两种规则。
- 每条输入必须恰有一个图像占位段；图像放在 user 消息中。目标先按 chat template 编码，依据消息边界定位 assistant 内容和 EOT；不能通过查找报告字符串的第一次出现来定位监督。
- 拼接后的 `inputs_embeds[B,L,4096]`、attention mask、position IDs、labels 必须一致；labels 保留真实答案 token IDs，其他位置为 -100。位置编码按有效 token 累计，不让 left-padding 消耗真实位置。内部 causal shift 只执行一次。
- 冻结 LLM 的 dropout/mode 及 checkpointing 行为需一起核查；不能让 `.eval()` 导致期望的 checkpointing 路径失效，也不能为了 checkpointing 意外更新其权重。LoRA dropout 只在 B-LoRA 训练时启用；val/inference 为 eval。
- 单卡训练图先验证，再加 DDP/FSDP。同时检查 Projector 和 Encoder 的梯度来源，不能用 `.detach()`、NumPy、推理 decorator 或误用缓存切断 B 的图。
- 新连续模式 checkpoint 始终保存 projector、完整 LoRA A/B 因子（包括首段暂时冻结的因子）、目标模块清单、optimizer/scheduler/RNG、epoch/next_window/global_step、phase、解析后的日程和独立早停状态。冻结 LLM/Encoder base 保存精确来源/hash；旧独立 A/B checkpoint 保持兼容。
- 新连续模式的阶段切换发生在同一个 run/optimizer 内；`resume` 恢复同一日程和数据游标，边界不可重复 warm-up 或重置 projector 动量。旧独立 A→B 才使用 initialize；不把历史 A-best 当作新 10-epoch run 的续训点。checkpoint 原子写入，验证首段/联合阶段重载及独立评测均可恢复正确权重。

### 11.4 环境与最低验收

沿用 `/mnt/sda/mingjie/conda_envs/projector` 和 [environments.yaml](../environments.yaml)：Python 3.10、PyTorch 2.5/CUDA 12.4、Transformers 4.51.3、PEFT 0.15.2、Accelerate 1.7.0，以及本地 NeuroVFM / fused-dense / FlashAttention。不要为了接入新适配器自动升级整个环境；配置兼容性由实际 forward/backward 检查。

本轮只读环境检查：`sacrebleu==2.5.1` 可 import；`rouge-score==0.1.2` 有包元信息，但 `rouge_score.rouge_scorer` **因缺少 nltk 导入失败**。原 `environments.yaml` 也未完整列出这些评价包。下一轮实现评价模块时需补齐并锁定依赖、运行指标冒烟检查，再同步 YAML；本轮未安装包。BERTScore/RaTEScore 的 scorer 模型是否齐备另行检查，不能因包安装就称权重可用。

按顺序验收，并将未通过项保留为明确状态：

1. **数据**：3,098 条目标与 active JSON 一致，split 数量 2,177/299/622，患者交集为空；所有路径按正确 root 解析，未知序列/缺图/非有限数据硬失败。追加第 3.6 节语义候选检查。
2. **预处理**：每个 cohort、序列、主要 spacing、斜位/边界情况抽样；官方参考输出与本地官方代码一致；native 分支在归一化前仅发生已声明的轴置换/翻转/补齐，未偷偷重采样。所有变换可追溯。
3. **Token**：关闭过滤时 N 等于完整网格乘积，坐标顺序正确且逐轴不越界；MLP 后 token 数不变；混合长度 batch 与单独样本输出在合理浮点容差内一致。
4. **文本**：图像 placeholder 不进入词表查表，chat BOS/EOT 不重复；答案起点/结束正确，prompt/image/padding 无 CE；含特殊符号和不同序列名时也成立。
5. **梯度**：A 只更新 MLP；B-LoRA 只更新 MLP 和视觉 adapters；B-Full 更新 MLP 和视觉全量参数；所有阶段 LLM 参数无梯度且不变化。至少做真实 NIfTI 的 forward、loss、backward、optimizer step，并区分 LoRA 初始化的单因子零梯度。
6. **数值**：相同样本的单 batch 与分 microbatch/global-token-normalized 梯度一致；batch padding 不改变有效样本 loss；checkpoint 重载可复现 eval 输出，cache 与在线特征一致。
7. **工程/视觉依赖**：先固定 8–16 条 train 样本做过拟合，记录图像打乱控制；量测最短/典型/最长输入在 A、B-LoRA、B-Full 中的峰值显存和速度。小样本成功不等于泛化或完整训练已完成。
8. **评价**：greedy 停止、触顶统计、批量/单样本一致性、BLEU/ROUGE signature、分层数量、患者 bootstrap 和预测 ID 覆盖全部通过，才发布完整 test 指标。

## 12. 本轮文档检查记录

2026-10-05（服务器记录 UTC 为 2026-10-06）扩充本文件；修改前副本见 [备份](backups/NEUROVFM_MLP_TRAINING_PIPELINE.20261006T062000Z.md)。本轮补充 spacing 选择及 QA、LoRA/全量分支、数据接口、prompt v2、评价细则、配置/代码结构和验收。未修改原 MRI、活动标注、split、模型权重或 conda 环境，也未开始训练。

已核对 active JSONL 的字段、数量、公开患者/检查数，本地 Encoder 权重尺寸和 LoRA 目标层、GPU 型号以及评价依赖。文件检查不代表未来训练实现已经通过；未完成的标签语义核查、几何 QA、长序列显存/梯度与生成测试均已在对应章节列明。

本轮验证结果：3,098 条清单与活动目标逐 ID 一致，所有引用的 image/mask 路径存在，患者跨集合交集为 0，五个活动标注文件和 split 的 SHA-256 与发布校验记录一致；文档本地链接无缺失、YAML 示例可解析、代码块闭合。当前 prompt/答案长度及特殊 token ID 完成离线核对；未将这些文档/数据检查表述成训练框架测试通过。


### 12.1 2026-10-12：替换 finetuning 主方案（历史文档阶段）

按用户确认，直接替换第 8 节的旧 A/B 参数与重建优化器方案，并同步开头状态、第 9–11 节的验证/流程/配置/checkpoint 描述。旧内容备份：[训练方案修改前副本](backups/NEUROVFM_MLP_TRAINING_PIPELINE.20261012T143939Z.md)。路线文档仅同步当前训练方案的引用和旧独立对照的适用范围，备份：[路线文档修改前副本](backups/PROJECTOR_ROUTES_AND_BASELINES.20261012T143939Z.md)。本轮未修改训练代码、YAML、脚本、模型权重或旧 run，也未启动训练。第 13 节是待用户确认的代码实施计划。

### 12.2 2026-10-12：最终方案补充早停与参数入口（历史文档阶段）

用户确认保留早停：两段 warm-up 后开始，完整 val CE 的 min_delta=0.005、patience=3，默认最早 epoch 5 停止。已将第 8 节和第 11.2/13 节此前“禁用早停”建议替换为此规则，并加入两段显式 optimizer steps 优先于 ratio、max_epochs/rank/alpha/早停参数可覆盖的接口与验收要求。同步路线文档的“最多 10 epochs + 早停”表述。

本轮修改前副本：[训练方案](backups/NEUROVFM_MLP_TRAINING_PIPELINE.20261012T150824Z.md)、[路线文档](backups/PROJECTOR_ROUTES_AND_BASELINES.20261012T150824Z.md)。训练代码、配置文件与脚本未修改；smoke test 尚未执行，等待用户对最终完整方案确认后实施，结果届时另行记录。

## 13. 连续 finetuning 的代码实现（2026-10-12 已落实）

### 13.1 调用链与修改位置

调用链为 `scripts/run_full_mlp_uncompressed.sh` / `scripts/train_projector.sh` → `src/cli/train.py` → `src/cli/common.py` → `src/config.py` → `Trainer` → model/optimizer/scheduler/checkpoint。修改前全流程脚本分别启动 A 和 B，B 从 A-best 只载入 projector 权重，重新创建优化器和计数器。现在主脚本一次启动 `joint_lora`，同一 run/optimizer 完成第 8 节的连续日程。

| 位置 | 修改前行为 | 本次实现 |
| --- | --- | --- |
| `configs/base.yaml`、`src/config.py`、新增 `configs/training/joint_lora.yaml` | 只接受 a/b_lora/b_full；单 warm-up；rank 8/alpha 16；patience 必须正整数 | 注册 joint_lora、两段 steps/ratio 和 early_stopping_min_delta；新 preset 默认 1e-5/1e-5、16/32、最多 10 epochs、feature_cache=false、patience=3/min_delta=0.005；明确 steps 优先级；检查比例、实际阶段步数与旧新字段适用范围 |
| `src/training/optim.py`、`src/training/schedule.py` | 只收集当时 requires_grad=True 的参数；所有组使用同一个单 warm-up lambda | 新模式从开始就建立稳定的 projector/LoRA 参数组；首段 LoRA 冻结、grad=None；用分组分段 LR 实现 77/77/1076 的解析日程；始终保留同一 AdamW 实例 |
| `src/models/encoders/neurovfm.py`、`src/training/stages.py` | 构造时按固定 stage 决定 LoRA 注入、eval、no-grad 与冻结断言 | 新模式预先注入 LoRA，按运行 phase 切换冻结/eval/no-grad；第二段开启 LoRA 梯度和 dropout；每次切换重新断言 LLM/base 冻结与训练参数量 |
| `src/training/trainer.py` | B 必须提供 A checkpoint；只有固定 stage；只在 epoch 级组织验证/早停 | joint_lora 可直接从预训练资产开始；按全局 optimizer update 在 epoch 内切换；保留数据游标、优化器和最佳指标；cosine 开始后的完整 epoch 验证维护独立早停 reference/bad_count；保存边界 checkpoint；DDP 预先注册全部 adapter 并使用 find_unused_parameters=True，冻结/解冻时保留 wrapper、参数和优化器 |
| `src/training/checkpoint.py`、`src/cli/evaluate_checkpoint.py` | encoder 只保存 requires_grad=True 的参数；加载按 stage/当前 trainable 集合判断；格式限定 v1 | 为新模式保存完整 adapter 因子、phase、日程、参数组标识、早停状态；新增版本化格式并兼容旧 v1；从任意阶段恢复训练/独立评测；避免恢复时随机重建已存在的 LoRA A 因子 |
| `scripts/run_full_mlp_uncompressed.sh`、`scripts/train_projector.sh`、`src/cli/train.py` | 全流程硬编码两次启动和 rank=8；通用脚本 stage 白名单只有旧模式 | 全流程改用新 preset 一次启动；通用脚本允许 joint_lora；完整透传 rank/alpha、warm-up steps/ratio、max_epochs 与早停参数，并保留数据范围、设备、缓存开关和启动快照；给新 run tag，拒绝覆盖已有目录 |
| `src/training/trainer.py` 日志与 `scripts/README.md` / 根 README | JSONL 无本步分组 LR/phase，说明仍为旧入口 | 记录实际使用的两组 LR、phase、global_step、阶段边界；保存完整日程；新旧入口和恢复方式分别说明 |

保留现有 a/b_lora/b_full preset 的历史语义；新增 joint_lora preset 显式覆盖 rank/alpha，避免修改默认值后破坏历史测试或 checkpoint。`src/models/encoders/lora.py` 已按配置接收 rank/alpha/dropout，24 个 targets、bias=none 与 use_rslora=false 已满足要求，不需要改 LoRA 算法。模型架构、答案 CE、数据/预处理流程沿用。

### 13.2 实现中的边界契约

1. 先根据实际选中 train 样本、world size、microbatch/accumulation 和总预算解析日程；显式 steps 优先于对应 ratio；按包含起点/终点的约定，每个阶段至少需要 2 次 optimizer updates，有效总预算不足时明确报错。单卡完整 BraTS 必须解析为 1230/77/77/1076。
2. 每个累积窗口开始前确定本次 phase 和 LR；不在 microstep 或验证期间推进 scheduler。首段 LoRA 不产生梯度或 Adam 状态；仅将 LR 设为零不足以替代冻结。
3. 阶段切换不得重建 projector 参数或 Adam 状态；先完成上一更新、清理梯度，再同步切换冻结状态。DDP reducer 必须覆盖新启用的 adapter，避免静默漏同步。
4. resume 校验数据选择、总预算、双 warm-up 配置、LoRA 配置和参数组顺序；保存已完成 global_step 与可还原的下一步 phase，确保边界前后恢复都不会重复或漏掉一步。
5. checkpoint 的“需要保存的 adapter”与“当前正在训练的参数”分开；首段 adapter 虽然冻结，其非零初始化因子也必须保存。评测从 checkpoint 恢复新配置/权重及 phase，不依赖训练进程留在内存中的状态。
6. 新 run 统一维护 best_val_ce 和独立早停状态，最多 10 epochs、符合第 8.5 节条件时提前停止；旧 run 的 rank-8 checkpoint 不作为 rank-16 的直接 resume。预处理缓存开关不能误当成最终视觉特征缓存开关。

### 13.3 验证项目（实际结果见第 14 节）

1. **配置/CLI 静态验证**：新旧 preset 均可解析；命令行覆盖顺序正确；默认 BraTS 和 rank16/alpha32 到达模型；启动脚本 bash 语法通过。
2. **小模型 CPU 契约测试**：核对完整 1230 步实际 LR 及两个归零边界、显式 steps 优先级和 max_epochs/rank/alpha 覆盖；首段 LoRA 无更新、第二段可更新；projector Adam 状态在切换时保持；验证 warm-up 不累计早停、cosine 首次验证建立基准、小改善累计/耐心重置/连续三次停止、best 与 reference 分离以及 NaN/Inf 异常处理；新旧 checkpoint 兼容；在切换前/后中断后恢复的 LR、权重、RNG、数据游标及早停状态与不中断运行一致。
3. **真实模型短运行**：用新独立 smoke 目录和缩短后的合法日程跨越两个边界，检查 rank16 的 884,736 个 adapter 参数、冻结权重、非零有限梯度、日志、checkpoint 重载与生成；再检查 8,775-token 典型长输入的资源占用。选空闲设备，不干扰已有训练。
4. **DDP 边界测试**：验证稳定注册的 reducer 在 adapter 冻结/解冻前后均正确同步梯度，以及分布式 resume；实际采用 Gloo/NCCL 两种后端测试，结果见第 14 节。
5. **交付后正式运行**：检查通过后给出实际新启动命令及配置摘要。正式最多 10-epoch 长训练单独执行。本轮已执行代码修改与 smoke test，实际命令、测试结果、证据路径及验证范围见第 14 节。


## 14. 连续 finetuning 实现与 smoke 验收记录（2026-10-12）

用户最终确认后已完成代码修改、相关回归测试及真实模型 smoke。正式训练入口现在是 [joint_lora.yaml](../configs/training/joint_lora.yaml)，[一键脚本](../scripts/run_full_mlp_uncompressed.sh) 一次启动连续训练；通用 [train_projector.sh](../scripts/train_projector.sh) 也接受 joint_lora。旧 a/b_lora/b_full preset 与 v1 checkpoint 保持兼容。

本轮证据目录：[joint_lora_20261012T151636Z](../runs/archive/engineering/joint_lora_20261012T151636Z/)。其中 `before/` 与 `before_hashes.json` 保存修改前代码/文档快照，`diffs/` 和 `change_manifest.json` 记录实际变更，`verification.json` 汇总验收结果。旧正式运行 `full_mlp_uncompressed_20261009` 未被用于续训或覆盖。

### 14.1 实现结果与入口

- 单卡默认设备统一为逻辑 `cuda:0`（`configs/base.yaml`）；显卡选择由外部 `CUDA_VISIBLE_DEVICES` 控制，例如 `CUDA_VISIBLE_DEVICES=3 bash scripts/baselines/run_mlp2x.sh`，训练与后续评测均使用所选卡。DDP 仍按 `LOCAL_RANK` 分配逻辑设备；恢复旧 run 配置时可用 `--set training.device=cuda:0` 覆盖原设备值。
- 配置新增双 warm-up steps/ratio、early_stopping_min_delta；rank/alpha、max_epochs、LR、早停等均可通过 YAML/`--set` 传入。显式 steps 优先，配置校验拒绝无法容纳三段端点的预算及旧单段 warm-up 冲突。
- `src/training/schedule.py` 负责按 optimizer update 解析三段日程，以及独立维护 early_stop_reference/bad_epochs 与 best_val_ce；`optim.py` 一次创建包含 projector 和全部 adapter 的稳定参数组。
- 第一段 LoRA 冻结、eval/no-grad，无 adapter 梯度或 Adam 状态；第二段开启 LoRA，保留 projector 参数与 Adam 状态。训练日志记录本步实际 LR，两个阶段边界自动保存 checkpoint。
- DDP 在包装时注册全部后续需要训练的参数，再恢复当前 phase，使用 find_unused_parameters=True 处理首段未用到的 adapter。阶段切换不替换模型参数、优化器或 reducer。联合阶段没有 unused 参数时出现的 PyTorch 性能提示不表示梯度漏同步。
- 新 v2 checkpoint 始终保存全部 adapter 因子、phase、解析日程、参数组名称、优化器/scheduler、早停状态及各 rank RNG；恢复时校验配置和阶段边界；独立评测可直接加载 v2。
- 主 launcher 保留启动快照，接受 `--config/--set` 并透传；无 run tag 时生成时间戳目录。`--dry-run` 不加载模型、不创建 run，只解析配置、数据选择和日程。

默认 dry-run 已确认：BraTS train=1960、effective batch=16、updates/epoch=123、计划总 updates=1230、两段 warm-up=77/77、cosine=1076、两组 peak LR=1e-5、rank16/alpha32、deterministic=true、patience3/min_delta0.005。

```bash
# 不启动训练，核验默认方案。
bash scripts/run_full_mlp_uncompressed.sh --dry-run

# 正式训练使用新目录；本次没有执行这条长训练命令。
bash scripts/run_full_mlp_uncompressed.sh brats_mlp_joint_r16 --datasets brats \
  --set training.device=cuda:0 \
  --set training.max_epochs=10 \
  --set training.projector_warmup_steps=77 \
  --set training.joint_warmup_steps=77 \
  --set training.lora.rank=16 --set training.lora.alpha=32
```

同一 run 续训使用通用入口，并保留原启动参数：

```bash
scripts/train_projector.sh mlp2x_gelu joint_lora runs/brats_mlp_joint_r16/joint_lora \
  --set training.device=cuda:0 \
  --set training.projector_warmup_steps=77 --set training.joint_warmup_steps=77 \
  --set training.resume=runs/brats_mlp_joint_r16/joint_lora/checkpoints/last.pt
```

### 14.2 实际测试结果与证据

| 验收项目 | 实际结果 | 证据（相对本轮证据目录） |
| --- | --- | --- |
| 配置/调度/早停/checkpoint/旧模式回归 | **60 tests passed**；覆盖显式步数优先级、完整 1230 步 LR、两处归零边界、小改善累计、best 与 early-stop reference 分离、错误参数拒绝、旧 checkpoint 兼容 | `pytest.log`；源测试 `tests/test_joint_training.py` 及已有相关测试 |
| CPU Gloo 双进程 | 冻结→解冻、每 update 两个 microbatch/no_sync、跨 rank 权重一致、边界恢复逐位一致 | 上述 pytest 中的 DDP 测试 |
| GPU NCCL 双进程 | GPU1/2、小模型、2+2+4 updates、每步两个 microbatch；跨 rank 权重一致，从 step2 恢复到 step8 与不中断运行逐位一致 | `nccl_smoke.py`、`nccl.log`、`nccl_summary.json` |
| 默认与覆盖参数 dry-run | 默认解析为 1230/77/77/1076；额外测试 epochs12、steps19/23、rank32/alpha64 和缓存关闭均到达 resolved config | `default_dry_run_no_tag.json`、`generic_dry_run.json`、`pytest.log` |
| 真实 NeuroVFM + Llama-3.1-8B | GPU3，完整 native 8775 tokens，不压缩 MLP、rank16，4 条 BraTS train/1 条 val，batch1，2 epochs，2+2+4 共 8 updates；梯度有限、首段 adapter 无更新、联合阶段 adapter 有梯度 | `real_smoke.py`、`real_continuous_v2/gradient_audit.json`、`real_continuous_v2/train.jsonl` |
| 真实模型冻结与恢复 | Encoder base 和全部 LLM 权重前后完整 SHA-256 不变；独立进程从 step2 边界恢复，后续 6 步后的 projector/adapter/optimizer/scheduler/训练状态/Torch-CUDA RNG 与连续运行逐位一致 | `real_smoke_summary.json`、`real_resumed_v2/`、`real_smoke_isolated.log` |
| Bash→启动快照→CLI→Trainer | 真实运行 6 updates，effective batch2，2+2+2 日程、3 epochs；每步实际 2 个样本，早停在 warm-up 后建立基准，最终因 max_epochs 结束 | `launcher_smoke.log`、`launcher_smoke_summary.json`；实际 run 为 `runs/joint_lora_cli_smoke_20261012/joint_lora/` |
| v2 checkpoint 独立评测 | 从上述 CLI smoke 的 best.pt 恢复，val 1 条样本，greedy generation + BLEU/ROUGE/answer CE，状态 **complete** | `evaluation_smoke.log`、`evaluation_smoke/metric.json` |

真实连续/恢复对照的峰值 allocated **21.017 GiB**、reserved **21.307 GiB**；CLI 累积 2 的 smoke 峰值 allocated **21.095 GiB**。这些是当前 GPU3、完整 8775-token 输入和本次短运行的实测值，不代表任意 batch/输入条件都具有相同峰值。

早停规则的阈值、累计改善、patience 触发和状态恢复由小模型测试覆盖；真实短运行验证了 warm-up 期间不建立基准、cosine 阶段首次完整 epoch 验证建立基准。没有为了触发早停而伪造真实验证 CE，也没有将短 smoke 当作 10-epoch 收敛实验。

真实模型测试使用单卡 GPU3；双卡 NCCL 测试使用小模型验证通信/冻结/累积/恢复逻辑，没有将其描述为完整 8B 模型双卡显存验收。正式多卡运行仍需每张卡容纳完整模型及激活。

### 14.3 命令记录与初次失败的处理

相关测试命令：

```bash
/mnt/sda/mingjie/conda_envs/projector/bin/python -m pytest \
  tests/test_joint_training.py tests/test_training.py tests/test_config.py \
  tests/test_evaluate_checkpoint.py tests/test_projector_integration.py \
  tests/test_distributed_loss.py -q --disable-warnings
```

真实完整模型/恢复对照由证据目录的 `real_smoke.py` 在独立子进程中执行；NCCL 命令为该目录的 `nccl_smoke.py`。完整启动参数分别保存在各 run 的 resolved_config.yaml 和 launcher_snapshot.sh 中，便于与正式配置区分。CLI smoke 使用 train_limit=4、val_limit=1、effective_batch_size=2、max_epochs=3、warm-up steps=2/2、save_every_steps=2、generation.max_new_tokens=16。

独立评测实际命令：

```bash
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 bash scripts/evaluate_checkpoint.sh \
  --checkpoint runs/joint_lora_cli_smoke_20261012/joint_lora/checkpoints/best.pt \
  --output-dir runs/archive/engineering/joint_lora_20261012T151636Z/evaluation_smoke \
  --split val --device cuda:0 --metrics bleu4 rougeL answer_ce --limit 1 --repeats 0
```

首版 smoke 验证脚本曾在同一进程顺序构造两个 8B Trainer，第二次装载因显存残留 OOM；原始 `real_smoke.log` 与首次 run 保留，不能记作恢复验收通过。将两个运行隔离到独立进程后，`real_smoke_isolated.log` 和 `real_smoke_summary.json` 验证通过；没有减少视觉 token 或改动已确认的训练算法来绕过该失败。

本轮完成的是代码/参数传递和工程连通性验证，尚未启动正式最多 10-epoch 训练。后续质量结论由正式 train/val 曲线与报告内容评估决定。

### 14.4 最终工作区兼容性复核

真实模型 smoke 完成后，共享工作区又出现 projector 压缩比例/预算配置的独立改动，涉及部分共用文件。本轮保留这些改动，在当前工作区重新执行上面的相关测试命令，仍为 **60 passed**，日志为 `pytest_final_workspace.log`；重新检查默认 launcher dry-run，仍为 1230/77/77/1076。当前不压缩 MLP 配置没有启用 target_compression_ratio，预算 preflight 不改变本方案输入或训练日程。

`change_manifest.json` 分别列出本任务涉及文件和另行观察到的工作区改动；共用文件的最终 diff 可能包含两部分，不能把全部变化都归于本轮 finetuning。各真实 run 的 `provenance.json` 保留实际运行时的源码哈希，最终工作区哈希另存于 change manifest，避免把先前 smoke 误称为对之后所有改动的完整 8B 复测。
