# Brain MRI 报告生成评价与 NeuroQA 补充实验

更新：2026-10-12。适用项目：NeuroVFM → Projector → 冻结 Llama-3.1-8B；单个完整 3D MRI 序列生成英文 Findings。**GREEN、RadGraph-XL、RaTEScore 和 BERTScore 均已接入，并完成真实权重 smoke test。** 本文的 smoke 数值只用于工程验收；2026-10-12 新增 BERTScore 未进行完整 val/test 正式评分。下载、运行及验收见第 6–8 节。

## 1. 已实现内容与选定方案

**主评价保留 BLEU-4、ROUGE-L，加入 GREEN、RadGraph-XL F1、RaTEScore，并补充 BERTScore P/R/F1。** NeuroQA 用于后续问答迁移实验，单独汇报。这四个模型评分指标都比较生成文本与参考文本，不直接读取 MRI；结合图像对照实验和原图盲评，才能更有力地支持视觉理解及临床正确性的论述。

| 项目 | 当前状态 | 含义与计算 | 额外资源 |
| --- | --- | --- | --- |
| BLEU-4 ↑ | 已实现 | 全语料 1–4 gram 的修正精确率几何平均，乘长度惩罚；当前输出 0–100 | 无评分模型；需要参考文本 |
| ROUGE-L F1 ↑ | 已实现 | 每对文本按最长公共子序列计算 precision、recall、F1，再取样本均值；当前输出 0–100 | 无评分模型；需要参考文本 |
| Answer CE ↓ | 已实现 | 在真实答案前缀条件下，对答案 token 计算平均交叉熵；用于训练监控和选 checkpoint | 当前 VLM；不是自由生成报告的临床准确率 |
| 分层、覆盖检查、bootstrap | 已实现并扩展到新增指标 | 按 cohort/sequence/source 汇总；检查缺失、重复、空输出；患者级区间、患者／检查宏平均及配对比较 | 已保存预测和逐例分数 |
| BERTScore P/R/F1 ↑ | 已实现离线评分及一键入口，真实模型验证通过 | 上下文 token 最大余弦匹配，保存逐例 P/R/F1 并分别取均值 | `google-bert/bert-base-uncased` 文本编码模型，约 441 MB |
| GREEN、RadGraph-XL | 已实现离线评分；真实模型验证通过 | 临床错误评价、实体及关系一致性 | 见第 2 节 |
| RaTEScore | 已实现离线评分；真实模型验证通过 | 医学实体软匹配 | 见第 2 节 |

BLEU 来源：Papineni 等，2002，[BLEU: a Method for Automatic Evaluation of Machine Translation](https://aclanthology.org/P02-1040/)。ROUGE 来源：Lin，2004，[ROUGE: A Package for Automatic Evaluation of Summaries](https://aclanthology.org/W04-1013/)。本地 BLEU 使用 sacreBLEU `13a`、区分大小写、`exp` 平滑；ROUGE 使用词干化。

代码依据：[metrics.py](../src/evaluation/metrics.py)、[generate.py](../src/evaluation/generate.py)、[trainer.py](../src/training/trainer.py)。各指标的分值不可直接当作“诊断正确率”，也不计划简单平均为一个总分。

## 2. 模型评分指标

### 2.1 GREEN：临床错误与匹配发现

**出处**：Ostmeier 等，*GREEN: Generative Radiology Report Evaluation and Error Notation*，Findings of EMNLP 2024。[论文](https://aclanthology.org/2024.findings-emnlp.21/)

评分语言模型读取 `reference + prediction`，输出匹配发现数，以及有／无显著临床意义的错误。错误分为：虚构发现、遗漏发现、位置错误、严重程度错误、虚构既往比较、遗漏既往变化。

设匹配发现数为 M，具有临床意义的六类错误合计为 E：

```text
GREEN = M / (M + E)
```

数值越高越好。官方实现中 M=0 时返回 0；无法解析的输出必须另记失败，不能静默删除。例：4 个匹配发现、2 个显著错误，分数为 4/6≈0.667；这是计数示例，不代表实测准确率。[官方评分实现](https://github.com/Stanford-AIMI/GREEN/blob/main/green_score/green.py)

**本轮采用模型**：[`StanfordAIMI/GREEN-RadLlama2-7b`](https://huggingface.co/StanfordAIMI/GREEN-RadLlama2-7b)，通过 [GREEN 官方代码](https://github.com/Stanford-AIMI/GREEN) 本地运行。需要模型和 tokenizer，不需要另行下载其训练数据，也不需要付费 API。7B 模型是三项中主要的显存和耗时来源。

**保存内容**：逐例分数、匹配数、六类显著错误数、原始解释、解析状态。评分器会犯错；对脑 MRI 的信号、左右侧和单序列证据应抽样复核。

GREEN 另有覆盖 CT/MRI 等报告文本的扩展版本，但不同权重不能混用后仍只标作同一个 GREEN。[RadEval 附录 B](https://arxiv.org/html/2509.18030v1) 介绍了扩展版本；本轮先固定上述 7B 权重建立可复现基线。

### 2.2 RadGraph-XL F1：结构化实体与关系

**出处要分清**：F1-RadGraph 评分方案来自 Delbrouck 等，*Improving the Factual Correctness of Radiology Report Generation with Semantic Rewards*，Findings of EMNLP 2022；XL 是后续的实体／关系抽取数据及模型，来自 *RadGraph-XL: A Large-Scale Expert-Annotated Dataset for Entity and Relation Extraction from Radiology Reports*，Findings of ACL 2024。XL 的数据包含脑 MRI。[评分论文](https://aclanthology.org/2022.findings-emnlp.319/)、[XL 论文](https://aclanthology.org/2024.findings-acl.765/)

先将两份文本抽取为带标签的实体和关系，例如“病灶—位于—左额叶”。对选定的评价单元集合 H（生成）和 R（参考）计算：

```text
Precision = |H∩R| / |H|
Recall    = |H∩R| / |R|
F1        = 2 × Precision × Recall / (Precision + Recall)
```

| 官方模式 | 常见名称 | 匹配要求 |
| --- | --- | --- |
| simple | rg_e | 实体文本与标签 |
| partial | rg_er | 实体文本、标签及是否存在关系 |
| complete | rg_bar_er | 进一步比较关系类型及连接对象 |

**三个模式均保存，主表至少保留 partial 与 complete。** partial 不是逐条关系完全匹配；complete 对病灶与位置的连接更严格。匹配仍受文本表达和抽取错误影响，不能把它视为完美的空间正确性评价。[官方评分规则](https://github.com/Stanford-AIMI/radgraph/blob/master/radgraph/rewards.py)

**资源**：[radgraph 官方包](https://github.com/Stanford-AIMI/radgraph)，显式设置 `model_type="radgraph-xl"`、`reward_level="all"`。使用 [`StanfordAIMI/RRG_scorers`](https://huggingface.co/StanfordAIMI/RRG_scorers) 的 `radgraph-xl.tar.gz`，另需其配置指定的 [`microsoft/BiomedVLP-CXR-BERT-general`](https://huggingface.co/microsoft/BiomedVLP-CXR-BERT-general) 基础模型及 tokenizer，现已一并下载。实际实体抽取加载的是 XL 训练权重，不是直接拿基础 CXR-BERT 评分。不必下载 RadGraph 标注数据或重训抽取器。保存逐例三项 F1 和两侧实体图。

### 2.3 RaTEScore：医学语义及否定的软匹配

**出处**：Zhao 等，*RaTEScore: A Metric for Radiology Report Generation*，EMNLP 2024。[论文](https://arxiv.org/html/2406.16845v2)

流程：医学实体识别 → 实体向量编码 → 两个方向的最相似实体匹配 → 类型加权及否定冲突惩罚 → 调和平均。实体类型包含 Anatomy、Abnormality、Disease、Non-Abnormality、Non-Disease。

```text
一个方向的 S = Σ(类型权重 × 冲突惩罚 × 最佳实体余弦相似度) / Σ类型权重
RaTEScore    = 2 × S(生成→参考) × S(参考→生成) / [S(生成→参考) + S(参考→生成)]
```

分母为零时依官方规则处理。类型权重已由作者拟合；本轮对 Findings 段落显式采用默认 `affinity_matrix="long"`，不在 test 上挑选权重。匹配是实体级软匹配，不是严格的关系图验证。[匹配源码](https://github.com/MAGIC-AI4Med/RaTEScore/blob/main/RaTEScore/utils.py)

**资源**：[RaTEScore 官方包](https://github.com/MAGIC-AI4Med/RaTEScore) 加载两个模型：

- [`Angelakeke/RaTE-NER-Deberta`](https://huggingface.co/Angelakeke/RaTE-NER-Deberta)：提取医学实体和类型。
- [`FremyCompany/BioLORD-2023-C`](https://huggingface.co/FremyCompany/BioLORD-2023-C)：编码实体名称，支持同义表达匹配。

无需另下载 RaTE-NER／RaTE-Eval 数据或重新训练。当前官方代码在任一文本没有抽取到实体时返回 0.5；必须额外记录空实体状态，不能解释为“正确一半”。保存逐例分数、实体及配置。[实现](https://github.com/MAGIC-AI4Med/RaTEScore/blob/main/RaTEScore/scorer.py)

**采用依据**：脑 MRI 的 AutoRG-Brain 已使用 RadGraph 和 RaTEScore；ReMIND 使用 RadGraph-F1、RaTEScore、GREEN；BrainDiff 使用 RadGraph-XL `rg_er`。这些指标有研究依据及实际采用记录，但尚不是脑 MRI 全领域统一的临床认证标准。对应论文及划分见第 4 节。

### 2.4 BERTScore：上下文语义匹配

**来源**：Zhang 等，[BERTScore: Evaluating Text Generation with BERT](https://arxiv.org/abs/1904.09675)，ICLR 2020；[官方实现](https://github.com/Tiiiger/bert_score)。它使用文本编码器计算 token 的上下文向量，不是让生成式 LLM 撰写评语。

设预测 token 向量为 `y_j`、参考为 `x_i`，均归一化。当前不启用 IDF：

```text
P = mean_j max_i cosine(y_j, x_i)   # 预测内容有多少能在参考中匹配
R = mean_i max_j cosine(x_i, y_j)   # 参考内容有多少被预测覆盖
F1 = 2 × P × R / (P + R)
```

按官方实现排除特殊 token，分别汇总每份报告的 P、R、F1；**整体 F1 是逐报告 F1 的均值，不是整体 P/R 再求调和平均**。保存原始分值，不乘 100、不裁剪浮点误差。BERTScore 的 F1 不是疾病类别 macro/micro-F1，不能独立验证否定、侧别或影像证据是否正确。

**冻结配置**：`bert-score==0.3.13`；`google-bert/bert-base-uncased`，revision `86b5e0934494bd15c9632b12f734a8a67f723594`；第 9 层；`idf=false`、`rescale_with_baseline=false`、`use_fast_tokenizer=false`、`all_layers=false`；FP32、batch size 1。模型名参考 [AutoRG-Brain](https://arxiv.org/html/2407.16684v3)，第 9 层来自官方 BERTScore 映射；这不代表完整复现该论文所有参数。官方英语默认 `roberta-large` 是另一套配置，分数不能直接混用。

版本核对以 `scoring_provenance.json` 的安装包版本和源码哈希为准：官方 0.3.13 发行包内部 `__version__` 仍为 0.3.12，因此其 `official_hash` 字符串显示 0.3.12；已保留原值并记录该差异。

**输入限制**：参考和预测分别最多 512 tokens（含特殊 token）。任一超限标记 `input_too_long`，不截断；空预测也不计入有效评分。有失败时完整均值为 `null`，CLI 返回非零，另列有效数量和诊断用均值。现有 299 条 val 预测的长度审计中，参考最长 88、预测最长 83 tokens，均未超限；未来其他预测仍逐例检查。

**资源**：仅需模型、tokenizer、预测及参考报告，无额外训练数据、API 或人工权限。权重、配置、词表及模型说明存于 `assets/models/evaluation/bert-score/`；清单及 SHA-256 见该目录的 `asset_manifest.json`。不会自动下载其他编码器。

## 3. 接入方式、缓存与环境

现有 `predictions.jsonl` 已包含 `id`、`patient_id`、`case_id`、`cohort`、`sequence`、`reference`、`prediction` 和 checkpoint 信息。新增独立评分入口直接消费已生成文本：

```text
现有 VLM 生成 → predictions.jsonl
                   ↓
           所选评分器依次运行
                   ↓
          逐例分数及解释文件
                   ↓
       分层均值、患者级 bootstrap、配对比较
```

**“复用分数”指相同预测的统计复用，不是拿旧模型结果代替新模型。** 例如已算好 202 位患者对应的 GREEN 分数，bootstrap 每轮有放回抽取 202 位患者，携带其全部序列，重新汇总均值；重复 1,000 次，取 2.5% 和 97.5% 分位数形成区间。这里没有重新生成报告，也没有让 GREEN 再推理 1,000 次。

- 同一批预测、参考、评分权重及配置均未变化：可复用逐例分数，重做分层、区间或统计图。
- checkpoint、输入或生成设置变化：应重新生成相应预测，再评分；不能沿用旧模型总分。
- 参考文本、评分权重／版本、提示、解析或精度设置变化：旧评分缓存失效，需要重算受影响条目。
- 单一平均分不足以做上述 bootstrap；必须保留逐例分数、患者映射及缓存标识。
- **BLEU-4 是 corpus 指标，不能平均逐例 BLEU 代替。** 现有 BLEU bootstrap 仍按重采样后的文本重新算 corpus BLEU；三个新增模型指标可以重用固定的逐例分数。

新指标缓存、分层及 bootstrap 已实现。模型比较使用同一批患者的配对重采样，汇报 B−A 的差值区间。三个新指标采用原始分值，BLEU/ROUGE 仍采用 0–100；避免混淆量纲。

**兼容性实测：GREEN 在现有 Transformers 4.51.3 下成功运行。** 使用原 `projector` Python 3.10／Torch 2.5.0，未新建 conda 环境，未降级训练依赖。RaTEScore 的 medspaCy/spaCy 依赖在 NumPy 2.1.2 下实测出现 ABI 错误，因此将评分专用 NumPy 1.26.4 和其他缺失依赖安装到 `third_party/evaluation/python_deps/`，仅离线评分进程显式加载；训练环境仍使用 NumPy 2.1.2。GREEN 上游安装声明仍固定 4.40.0，本项目记录的是实际运行兼容性，不声称上游声明已改变。[本地依赖](../pyproject.toml)、[GREEN 依赖](https://github.com/Stanford-AIMI/GREEN/blob/main/setup.py)、[检查记录](../archive/non_pipeline/environment_checks/clinical_metrics/)

## 4. 论文是在 train、val 还是 test 上汇报？

**不能把 train+val+test 混合后的结果作为未见数据泛化结果。** 在本次核对的工作中，既有独立测试结果，也有最佳验证结果；应以每篇论文的具体任务、表格及选择协议为准。以下按所列版本核对，2026 年 arXiv／medRxiv 工作属于预印本。

| 论文及定位 | 实际汇报范围与证据 | 对本项目的启示 |
| --- | --- | --- |
| [AutoRG-Brain，v2](https://arxiv.org/html/2407.16684v2)，脑 MRI 报告生成 | §2.3、Table 2 汇报报告生成；Table 4 明列 RadGenome 的 train/val/test，原始 test 共 688 个图像—报告对，正文亦称多中心 test set。不是合并全部划分 | 按其测试划分理解结果；我们的患者去重及单序列目标导致样本数不同，不能直接混同比较 |
| [ReMIND，v1](https://www.medrxiv.org/content/10.64898/2026.03.30.26349106v1.full)，多序列脑 MRI VLM | Experimental settings：GH 按 85%/5%/10% 划分，val 调参和选 checkpoint。Table 2 为 GH test（6,554 次检查）及 BMC/LHMC/BSH 外部队列 | 主结果是内部 test 与外部未见队列，训练数据不混入评分 |
| [BrainDiff，v1](https://arxiv.org/html/2609.00593v1)，含 Projector 的纵向脑 MRI 模型 | Table 2：独立 test 的 1,444 个可用配对；部分大模型仅测其中 64 例。Table 3：外部 BIND 的 800 个配对。补充材料注明按 val `rg_er` 选模型 | 选择用 val，最终比较用 test；必须标明完整测试与子集测试 |
| [ORCA，v1](https://arxiv.org/html/2608.00345v1)，3D CT token 压缩及两阶段 Projector/LLM 训练 | Table 6 的生成实验按附录 A.3：每 epoch 完整验证生成，汇报最佳 epoch。附录 B.5 的独立 2,000 体积 test 是 **probe 属性预测**，不能转称为报告生成 test 结果 | 其生成结果应标为最佳验证结果，不能一概认为论文所有表都是 test |

本项目当前划分为：**train 2,177 序列／701 患者；val 299／98；test 622／202**。采用已有患者互斥划分，阶段 A/B 的训练和调参不使用 test。

| 阶段 | 使用数据 | 应汇报什么 |
| --- | --- | --- |
| 训练监控 | train | 训练 CE、优化过程；用于诊断拟合情况 |
| 调参和 checkpoint 选择 | val | 当前按完整 val 的 answer CE 选 best；训练中的生成监控默认仅 14 条代表样本 |
| 正式评价 | 冻结方案后的完整 test | BLEU-4、ROUGE-L、GREEN、RadGraph-XL、RaTEScore，分层结果和患者级区间 |
| 视觉依赖对照 | 相同 test 病例 | 正常图像、同序列其他患者图像、无图像；比较分数变化 |
| 后续补充实验 | 确定范围的 NeuroQA 测试子集 | QA 指标及无图像对照，单独报告患者重叠、序列和筛选规则 |

**当前评价配置默认 `val`，最终测试要显式设置 `evaluation.split=test`。** “完整验证”仅指全部 val，不是 train+val+test；训练日志中的 14 条生成结果也不能作为完整测试结果。依据：[base.yaml](../configs/base.yaml)、[训练与评价说明](../README.md)。

首次新增指标以固定 val 小样本验收；冻结评分协议后再评 test，不按 test 分数选 checkpoint、权重矩阵或提示。论文主表建议每个模型在同一 test 上报告五类指标，RadGraph 再拆 partial/complete 两列；补充材料保存错误类别及分层结果。

## 5. NeuroQA：准备与测试方案

### 5.1 数据与开放范围

**出处**：Abbasi 等，*NeuroQA: A Large-Scale Image-Grounded Benchmark for 3D Brain MRI Understanding*，2026 年预印本。完整基准包含 56,953 个 QA、12,977 位受试者、12 个源数据集，203 个模板、11 类问题；格式为 YN、MCQ、Open。[论文](https://arxiv.org/html/2605.20525v1)

| 部分 | 数据来源 | 获取方式 |
| --- | --- | --- |
| Tier 1 | BraTS-GLI、BraTS-MEN、CC359、HCP-YA、IXI、WMH | 直接下载公开 QA；图像由原来源提供 |
| Tier 2 | ABCD、ADNI、AIBL、HCP-Aging、HCP-Development、PPMI | 自行取得源数据访问权限，按作者已发布的相应构建流程复现；README 标明 PPMI 脚本仍待补充 |

公开 Tier 1 文件实际为 train 13,598、val 1,892、test 1,813，共 17,303 条；19,262 是包括未公开私有测试部分的 Tier 1 总数。全部 12 个来源的论文 test-public 为 5,524 条，不能与公开六来源的 1,813 条混淆。[数据卡](https://huggingface.co/datasets/mhabbasiit/NeuroQA/blob/main/README.md)、[构建仓库](https://github.com/mhabbasiit/neuroqa)

**不需要先下载全部 12 个图像数据集。** 只评某个来源，就准备该来源所选问题对应的图像；已有的本地 BraTS/WMH 在版本、患者、检查和序列匹配后可复用。完整复现全部来源才需要准备其对应数据及许可。可以只做公开子集，但必须说明范围，不能宣称完整 NeuroQA 成绩。

数据流是：**匹配图像 + question/options → 本项目 VLM → prediction；prediction + answer → 官方评分器**。正确答案只用于评分，不进入模型输入。公开 QA 中的图像路径为 `"#"` 占位，需按 subject/session/sequence 映射本地文件。

入口：[QA 下载](https://huggingface.co/datasets/mhabbasiit/NeuroQA)、[官方测试代码](https://github.com/mhabbasiit/neuroqa/tree/main/eval)、[原始图像来源](https://github.com/mhabbasiit/neuroqa/blob/main/docs/data_sources.md)、[图像预处理](https://github.com/mhabbasiit/sMRI)。MRI 未随 QA 发布，源数据各自的访问规则仍适用。

### 5.2 本项目需要准备什么

1. **确定测试范围**：先做单序列可回答的公开题目；多序列／纵向问题需单独支持，不能用任意一个体积替代。
2. **患者重叠审计**：排除本项目 train/val 患者。部分 NeuroQA 答案来自 RadGenome 报告解析，所以新 QA 不自动代表新患者或独立标注。
3. **图像映射与预处理**：匹配同一患者、检查及序列；作者使用 MNI 标准空间预处理。本项目如采用不同流程，应记录差异。
4. **QA 任务适配**：新增 QA 数据结构、问题/选项提示和生成入口；复用 Encoder/Projector/LLM 权重。Modality 题不能沿用当前提示中的显式序列名称，文件名和队列名也不进入提示。
5. **冻结实验设置**：首先评估现有报告 checkpoint 的零样本 QA 迁移；如后续用 NeuroQA 训练集微调，另设实验，防止与零样本结果混用。
6. **保存及评分**：输出含 `neuroqa_id`、`prediction` 的 JSON 数组；校验完整覆盖、无重复，并固定 YN/MCQ 答案格式。

本地审计：公开 test 的 1,813 条中，420 条对应本项目 train 患者，44 条对应 val，104 条对应 test；其余 1,245 条来自 CC359/HCP-YA/IXI，未见于当前监督清单。104 条可用于现有测试患者的新任务迁移；1,245 条需另准备图像并核查 Encoder 预训练重叠。它们还需进一步筛选序列可回答性，不能直接视为最终可用数量。[本地审计](../runs/archive/engineering/evaluation_audit_20261006/neuroqa_public_test_overlap.json)

部分问题的精确答案依赖体积测量或临床元数据，应按 `human_assessable`、`image_dependency` 和类别分层解释。**NeuroQA 检验问答迁移，不能代替主任务报告评价。**

### 5.3 测试汇报指标

| 指标 | 计算及用途 |
| --- | --- |
| YN Accuracy | 归一化后的 Yes/No 正确数 ÷ YN 总数 |
| MCQ Accuracy | 选项字母或答案文本匹配的正确数 ÷ MCQ 总数 |
| Closed Accuracy | YN 与 MCQ 合并后的正确率；建议作为 QA 主指标 |
| Open Exact Match | 开放答案归一化后的完全匹配率 |
| Open Token F1 | 词级重合的 precision/recall 调和平均 |
| Open ROUGE-L | 最长公共子序列 F1 |
| Open BERTScore | 可选语义指标，需要额外文本编码模型；非必须 |
| Shortcut Score | `1 − (模型准确率 − 基线准确率) / (100 − 基线准确率)`，百分数口径 |

官方脚本按数据集、类别、题型分层。其模板多数答案基线不是“同一 VLM 不看图”的实测结果；另跑本模型无图像对照，报告正常图像与无图像的差值。采用范围一致的 `shortcut_score_closed`；SS=1 表示与基线相同，SS>1 表示低于基线，不能单凭 SS 断言图像利用程度。[工具说明](https://github.com/mhabbasiit/neuroqa/blob/main/eval/README.md)

实现细节：官方脚本会跳过缺失预测，因此外层必须检查覆盖；YN 最稳妥输出为裸 `Yes`/`No`，MCQ 为单字母，避免解释性句子被判格式不匹配。[评分源码](https://github.com/mhabbasiit/neuroqa/blob/main/eval/evaluate.py)

### 5.4 QA 下载状态与后续评分

已下载公开 QA 至 `assets/data/NeuroQA/all_public/`，并保存文件哈希；官方构建／评价代码在 `third_party/evaluation/neuroqa/`。未下载其 MRI 图像，未实现本项目 QA 推理。复现下载可运行：

```bash
cd /mnt/sda/jyzhang/VLM_Projector
PYTHONNOUSERSITE=1 /mnt/sda/mingjie/conda_envs/projector/bin/python \
  archive/non_pipeline/scripts/download_clinical_assets.py --neuroqa
```

完成图像映射、推理和覆盖检查后，官方评分命令为：

```bash
python third_party/evaluation/neuroqa/eval/evaluate.py \
  --predictions runs/neuroqa/predictions.json \
  --ground-truth assets/data/NeuroQA/test_eligible.json \
  --output runs/neuroqa/metrics.json
```

`test_eligible.json` 是按冻结规则从公开 test 筛出的待创建文件；预测文件也尚待生成。基础评分需 numpy；可选 `--bertscore` 另需 bert-score 包及模型。私有测试答案不公开，按作者发布的提交渠道处理。

## 6. 已完成的实现与下载

经用户确认后已实施。新增代码为 [评分 CLI](../src/cli/score_reports.py)、[缓存及统计](../src/evaluation/clinical.py)、[官方评分适配器](../src/evaluation/clinical_scorers.py)、[依赖与资源验证](../src/evaluation/clinical_runtime.py)。一次性成对比较 CLI 保存在 [`archive/non_pipeline/src/cli/compare.py`](../archive/non_pipeline/src/cli/compare.py)，不属于日常 pipeline。训练仍按 val CE 选 best，评分器不进入训练 step。

### 6.1 下载清单

所有模型均位于 `/mnt/sda/jyzhang/VLM_Projector/assets/models/evaluation/`。下列为下载文件的十进制 GB，不含解压副本／缓存；每个目录的 `asset_manifest.json` 记录完整 revision、文件 SHA-256 及大小。

| 模型资源 | 子目录 | 约 GB | 固定 revision 前 12 位 | 用途 |
| --- | --- | --- | --- | --- |
| GREEN-RadLlama2-7b | `green/` | 13.477 | `832389d56bbd` | GREEN 评分及解释 |
| RadGraph-XL archive | `radgraph/` | 0.416 | `6646433b3ad8` | XL 实体／关系抽取权重 |
| **BiomedVLP-CXR-BERT-general（实现时补充）** | `cxr-bert/` | 0.881 | `6172dbfa7c06` | XL 配置要求的基础模型和 tokenizer |
| RaTE-NER-Deberta | `rate-ner/` | 0.747 | `66d4b8189374` | RaTE 实体识别 |
| BioLORD-2023-C | `biolord/` | 0.438 | `4bb5abd13930` | RaTE 实体编码 |
| **BERT-base-uncased（2026-10-12 补充）** | `bert-score/` | 0.441 | `86b5e0934494` | BERTScore 第 9 层上下文编码；110M 参数 |

**这四个指标只需要预测和参考报告，不需要另外下载评分器的训练／标注数据集。** 没有下载 ReXVal、RadGraph 标注集或 RaTE-Eval。GREEN 的代表句聚类摘要未启用，不需要额外下载 `paraphrase-mpnet-base-v2`。

另已下载 NeuroQA 公开问答 train/val/test（13,598／1,892／1,813），共 17,303 条；这是后续 QA 实验的准备，不是本次三个指标的评分输入。

官方源码固定为：GREEN `29567040fba8`、radgraph `87f11a1ff4d2`、RaTEScore `a970406b6f41`。项目内评分依赖锁定在 [requirements-clinical.txt](../requirements-clinical.txt)。新机器准备命令：

```bash
cd /mnt/sda/jyzhang/VLM_Projector
PYTHONNOUSERSITE=1 /mnt/sda/mingjie/conda_envs/projector/bin/python archive/non_pipeline/scripts/setup_clinical_metrics.py
PYTHONNOUSERSITE=1 /mnt/sda/mingjie/conda_envs/projector/bin/python archive/non_pipeline/scripts/download_clinical_assets.py --neuroqa
PYTHONNOUSERSITE=1 /mnt/sda/mingjie/conda_envs/projector/bin/python archive/non_pipeline/scripts/download_bertscore_assets.py
```

现有服务器已安装并验证 `bert-score==0.3.13`，位置为 `third_party/evaluation/python_deps/`。仅向评测目录安装该包（`pip --no-deps --target`），Torch 2.5.0、Transformers 4.51.3 保留。评分时禁用在线下载，资源先校验哈希；仅检查所选指标的模型和依赖。新机器仍需完整准备依赖，`--no-deps` 本身不会补齐依赖链。

### 6.2 运行方法

**现在可以只指定 checkpoint，一条 Bash 命令完成生成和所选指标评价。** [Bash 入口](../scripts/evaluate_checkpoint.sh) 调用 [流程代码](../src/cli/evaluate_checkpoint.py)，自动从 checkpoint 恢复模型、预处理、A／B-LoRA／B-Full 阶段配置，再顺序执行原有生成和评分入口。

```bash
cd /mnt/sda/jyzhang/VLM_Projector
bash scripts/evaluate_checkpoint.sh \
  --checkpoint runs/full_native_20261006/stage_b_lora_full/checkpoints/step_00000068.pt \
  --split test --device cuda:2 --metrics all \
  --output-dir runs/eval_b_lora_step68_test
```

- 不传 `--metrics` 默认等同于 `--metrics all`：BLEU-4、ROUGE-L、GREEN、RadGraph-XL 三种 F1、RaTEScore、**BERTScore P/R/F1**、Answer CE，共 11 个数值。NeuroQA 不包含在内。
- 自选：例如 `--metrics bleu4 rougeL bertscore`、`--metrics bertscore` 或 `--metrics green radgraph rate bertscore`；只有被选中的评分器才会加载，未选 Answer CE 时跳过额外 CE 前向。底层生成步骤仍保存基础文本指标及覆盖检查供排错，顶层 `metric.json` 只列所选指标。
- 数据范围：`--split test` 默认全 test；`--split val --limit 2` 用于小样本检查；`--ids-file` 可指定该划分中的病例。
- `--dry-run`：检查 checkpoint 元数据和数据清单，展示阶段、范围及命令，不运行 VLM／评分模型。
- `--repeats 1000` 为患者级 bootstrap 次数；临床评分可用 `--scoring-device cuda:1` 单独指定设备。
- 输出目录必须新建。入口保存 checkpoint 快照，防止训练更新 `best.pt`／`last.pt` 后混淆来源；同文件系统默认用硬链接，额外空间占用很小。顶层总结果统一为 `metric.json`，包含 BERTScore 时不另加文件名后缀；完整生成／评分输出在 `generation/`、`scoring/`，过程状态为 `pipeline.json`。两个子步骤仍使用各自的 `metrics.json`；既有 smoke 证据保留当时文件名。
- Bash 自动使用 `/mnt/sda/mingjie/conda_envs/projector/bin/python`；可通过 `PROJECTOR_PYTHON` 更换。生成和评分在独立子进程顺序运行，避免模型同时占显存或评分依赖污染 VLM 环境。

**已有预测时只补算指标**：无需再指定 checkpoint，仍可使用原有离线入口。当前真实 smoke 可直接复用：

```bash
cd /mnt/sda/jyzhang/VLM_Projector
PYTHONNOUSERSITE=1 OMP_NUM_THREADS=4 /mnt/sda/mingjie/conda_envs/projector/bin/python \
  -m src.cli.score_reports \
  --predictions runs/archive/evaluation_pipeline_smoke/clinical_metrics_smoke_20261007/predictions.jsonl \
  --expected-ids runs/archive/evaluation_pipeline_smoke/clinical_metrics_smoke_20261007/expected_ids.json \
  --output-dir runs/archive/evaluation_pipeline_smoke/clinical_metrics_smoke_20261007/scored \
  --metrics green radgraph rate --device cuda:2 --repeats 1000
```

正式评价时替换预测路径、输出目录和**从冻结评价清单独立取得的** expected IDs。`--expected-ids` 校验覆盖；省略时明确标为 `input_file_only`，不证明覆盖整个 test。设备必须显式指定，`cuda:2` 是当前可见设备编号；本次测试避开了正在训练的 GPU 3。

**只给已有预测补算 BERTScore**（CPU 也可运行；已有评测结果保留，新建输出目录）：

```bash
PYTHONNOUSERSITE=1 OMP_NUM_THREADS=4 /mnt/sda/mingjie/conda_envs/projector/bin/python \
  -m src.cli.score_reports \
  --predictions runs/bertscore_smoke_20261012/predictions.jsonl \
  --expected-ids runs/bertscore_smoke_20261012/expected_ids.json \
  --output-dir runs/bertscore_rescore_example \
  --metrics bertscore --device cpu --repeats 1000
```

代码路径：`scripts/evaluate_checkpoint.sh` → `src/cli/evaluate_checkpoint.py` → 生成子进程 `src/cli/evaluate.py` → 评分子进程 `src/cli/score_reports.py` → `BertScoreScorer.score(reference, prediction)`。评分器调用官方 `BERTScorer.score(cands=[prediction], refs=[reference])`；缓存、分层及患者 bootstrap 复用 `clinical.py`。文件名中的 `clinical` 是原有输出协议，BERTScore 仍属于语义指标。输出键为 `bertscore_precision`、`bertscore_recall`、`bertscore_f1`；评分后不再保留矛盾的 `BERTScore: not_run` 占位。

- 支持 `--metrics green` 等子集；`--cache-dir` 可跨输出目录共享逐例缓存。
- 同一输入及评分协议再次运行会命中缓存，重新汇总统计；模型不再次加载。修改模型／代码／配置／文本则相应缓存失效。
- 本次共用评分源码及依赖记录方式发生变化，旧三个指标的缓存指纹也会变化；重新评分会重算，不会把旧缓存强行视为当前版本。旧结果文件仍可追溯，未被本次 smoke 覆盖。
- 输出目录绑定输入文件哈希；不同预测文件使用新目录，避免覆盖旧实验。
- GREEN 沿用官方 greedy、FP16、总长度 2,048；超长输入或输出触顶单独标失败。`--green-max-length` 会改变评分协议，应在 val 上冻结后统一使用。
- RaTE 拒绝静默截断超过 512 token 的单句；实体编码沿用官方 30 token 限制并记录触发情况。
- 重复 ID／覆盖不一致会在加载模型前终止；空输出、解析失败等保留逐例状态。任何失败使正式均值为 `null`，另列带分母的 `valid_only_mean` 供排错，CLI 返回非零状态。

| 输出文件 | 内容 |
| --- | --- |
| `clinical_scores.jsonl` | 原预测元数据、逐例分数、GREEN 解释／计数、实体图、耗时／显存、缓存标识 |
| `clinical_metrics.json` | 所选模型评分指标的均值、分层、患者／检查宏平均、区间、异常计数 |
| `metrics.json` | 原 BLEU/ROUGE 与新指标的合并结果 |
| `scoring_provenance.json` | 实际环境、文件校验值、模型版本、评分协议、运行及缓存统计 |
| `cache/<metric>/*.json` | 每对文本的原子写入缓存，支持中断续算 |

两个实验的配对比较：

```bash
PYTHONPATH=. python archive/non_pipeline/src/cli/compare.py \
  --a runs/experiment_a/predictions.jsonl --b runs/experiment_b/predictions.jsonl \
  --clinical-a runs/experiment_a/scored/clinical_scores.jsonl \
  --clinical-b runs/experiment_b/scored/clinical_scores.jsonl \
  --output runs/comparison_a_b.json --repeats 1000
```

比较前校验患者、参考、预测对应关系及评分器配置，差值定义为 B−A。BLEU 按重采样语料重新计算，新指标直接重采样逐例分数。

## 7. 实测验收与边界

### 7.1 原三个指标（2026-10-07）

真实 smoke 使用已有阶段 A 验证生成结果，从 BraTS-GLI/MEN/MET、ISLES22、WMH 各取 1 条，共 **5 例 × 3 个评分器，全部成功**；患者与参考文本已逐项核对原始 val 清单。只验证评分流程，未重新生成 VLM 报告，未使用 test。相关接口／回归测试 21 项通过，另有 2 项 CLI 拒绝错误输入的检查通过；训练环境 `pip check` 通过，Torch／Transformers／PEFT／Accelerate／NumPy 版本未改变。汇总见 [acceptance.json](../runs/archive/evaluation_pipeline_smoke/clinical_metrics_smoke_20261007/acceptance.json)。

| 指标 | 5 例总耗时，含评分器加载 | 本进程峰值 allocated 显存 | 官方调用对照 |
| --- | --- | --- | --- |
| GREEN | 约 51.6 秒 | 13.13 GiB | 原始解释及分数一致 |
| RadGraph-XL | 约 7.8 秒 | 0.49 GiB | 三种 F1 一致 |
| RaTEScore | 约 1.7 秒 | 1.11 GiB | 分数一致 |

耗时不含启动时文件哈希校验；显存不含同卡其他进程，数值依报告长度和设备而变化。固定小样本不代表完整测试质量。

受控文本测试额外覆盖完全相同、同义改写、否定反转、位置变化；**不把全部指标的预期排序当作验收条件**。例如本次 RaTE 对同义改写为 0.659，对否定反转为 0.851、位置变化为 0.877，说明其分数仍可能偏好有临床错误但词汇重合较多的报告。identity 的微小浮点超 1 保留官方原值，不做隐式裁剪。

缓存复跑确认 **15 次缓存命中、0 次模型调用**；与自身配对比较的所有新指标及 BLEU/ROUGE 差值区间均为 `[0, 0]`。证据：[真实报告及完整输出](../runs/archive/evaluation_pipeline_smoke/clinical_metrics_smoke_20261007/)、[首次模型执行记录](../runs/archive/evaluation_pipeline_smoke/clinical_metrics_smoke_20261007/scored/initial_execution_provenance.json)、[官方对照及扰动结果](../runs/archive/evaluation_pipeline_smoke/clinical_metrics_smoke_20261007/official_parity_and_probes.json)、[配对检验](../runs/archive/evaluation_pipeline_smoke/clinical_metrics_smoke_20261007/paired_self.json)。

新增一键入口另以 B-LoRA `step_00000068.pt` 在 1 条真实 val 样本上完成了 **checkpoint → MRI 报告生成 → Answer CE → 全部报告指标 → 总结果** 的完整测试，11 项相关接口／回归检查通过。全 test 的 dry-run 确认范围为 622 序列／202 患者，但未运行完整 test。证据：[一键流程验收](../runs/archive/evaluation_pipeline_smoke/checkpoint_pipeline_smoke_20261007/acceptance.json)、[汇总结果](../runs/archive/evaluation_pipeline_smoke/checkpoint_pipeline_smoke_20261007/metrics.json)。

### 7.2 BERTScore（2026-10-12）

- **18 项接口与回归测试通过**：指标选择、汇总、缓存及配置失效、患者 bootstrap、失败分母、原三个指标和生成接口。
- **5 条真实 val 报告全部成功**，来自 BraTS-GLI/MEN/MET、ISLES22、WMH。患者和参考文本再次核对 val 清单；CPU 离线评分的 15 个 P/R/F1 与独立官方 `bert_score.score()` 输出完全一致，最大绝对误差 **0**（验收容差 `1e-6`）。
- **真实 checkpoint 全流程通过**：B-LoRA `best.pt` 快照，SHA-256 `082e9f90284a07defd7b41eb733219c26beec93c60b0809737349e0de04793bb`，1 条 val，`--metrics all`，GPU 2。生成、Answer CE、原三个评分器和 BERTScore 共 11 个数值均完整。BERTScore GPU 官方对照误差也为 **0**。
- **缓存和边界通过**：离线 CLI 再跑为 5 次命中、0 次模型调用；自配对差值区间均为 `[0, 0]`。相同文本约为 1；交换参考／预测后 P/R 对换；512 tokens 可评分，513 tokens 在模型调用前拒绝；空输入不会误记有效分数。混合有效／空／超长输入的 CLI 检查确认完整均值为 `null`、有效数量如实保留、返回非零。
- **环境验证**：保持 Torch `2.5.0+cu124`、Transformers `4.51.3`、PEFT `0.15.2`、Accelerate `1.7.0`、训练环境 NumPy `2.1.2`，训练环境 `pip check` 通过。评分继续复用现有 NumPy `1.26.4` 依赖目录。

| Smoke 范围 | BERTScore P | BERTScore R | BERTScore F1 |
| --- | --- | --- | --- |
| 已保存的 5 条 val 报告，CPU | 0.664525 | 0.569832 | 0.612102 |
| 新生成的 1 条 val 报告，GPU 2 | 0.593156 | 0.628464 | 0.610299 |

这些是不同模型／样本范围的工程验收结果，不能作为论文中的完整 test 成绩。证据：[验收记录](../runs/bertscore_smoke_20261012/acceptance.json)、[官方对照及边界测试](../runs/bertscore_smoke_20261012/official_parity_and_probes.json)、[GPU 官方对照](../runs/bertscore_smoke_20261012/checkpoint_official_parity.json)、[checkpoint 总结果](../runs/bertscore_smoke_20261012/checkpoint_all/metrics.json)、[异常输入验收](../runs/bertscore_smoke_20261012/negative_cli_acceptance.json)。

复现离线官方对照（先用第 6 节入口将这 5 条报告评分到 `runs/bertscore_smoke_20261012/scored`）：

```bash
PYTHONNOUSERSITE=1 OMP_NUM_THREADS=4 /mnt/sda/mingjie/conda_envs/projector/bin/python \
  archive/non_pipeline/scripts/smoke_bertscore.py \
  --output-dir runs/bertscore_smoke_20261012 --device cpu
```

## 8. 是否需要人工申请权限？

**本次三个指标的全部模型及 NeuroQA 公开 QA 均已无需登录下载成功，没有阻塞当前 evaluation 的人工申请项。** 不存在必须申请 Llama 原始底座后才能下载本次 GREEN 权重的障碍。

2026-10-12 补充的 BERT-base-uncased 同样已无需登录下载并校验，无新增人工申请项。

下表仅用于以后扩展 NeuroQA 图像范围；不必为本次 smoke 或一个选定子集申请全部来源。

| 来源 | 需要你做的事 | 入口 |
| --- | --- | --- |
| HCP-YA | 注册／登录 ConnectomeDB powered by BALSA，接受 Open Access Data Use Terms；如问题依赖受限元数据，再另申请 restricted data | [官方条款与注册入口](https://www.humanconnectome.org/study/hcp-young-adult/data-use-terms) |
| ABCD、HCP-Aging、HCP-Development | 通过 NDA 申请相应数据权限；按页面准备研究用途、机构信息及所需机构签署，不等同于 HCP-YA 的开放注册 | [NDA](https://nda.nih.gov/)、[Lifespan 官方申请说明](https://www.humanconnectome.org/storage/app/media/documentation/LS2.0/LS_Release_2.0_Access_Instructions_June2022.pdf) |
| ADNI | 在 IDA 提交申请，填写机构及研究用途，并接受数据使用协议 | [ADNI 官方申请入口](https://adni.loni.usc.edu/data-samples/adni-data/) |
| AIBL | 按 AIBL 指定渠道申请相应影像版本；在 IDA 核对该 study 的权限要求 | [AIBL](https://aibl.org.au/)、[IDA](https://ida.loni.usc.edu/) |
| PPMI | 提交在线数据申请并签署使用协议；NeuroQA 的 PPMI 构建脚本仍需核对完整性 | [PPMI 官方入口](https://bank.ppmi-info.org/access-data-specimens/download-data) |
| IXI／CC359 | 从公开影像页面取得选定病例，按页面许可使用 | [IXI](https://brain-development.org/ixi-dataset/)、[CC359](https://sites.google.com/view/calgary-campinas-dataset) |
| 本地已有 BraTS／WMH | 先复用版本、患者及序列匹配的合法副本；只有缺失时再按源平台获取对应版本 | [NeuroQA 来源表](https://github.com/mhabbasiit/neuroqa/blob/main/docs/data_sources.md) |

人工授权完成后，只需提供已下载文件路径或在服务器配置对应访问方式；不要把账户密码／访问 token 写入本文档。NeuroQA 的图像映射、去重及 QA 推理仍属于后续实验。
