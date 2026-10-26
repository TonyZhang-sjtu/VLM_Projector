# Report-output v1：Baseline A 上的 R1 / R2

本轮目标是改善**自由生成报告的事实选择与单病灶属性组合**，不改变四角色 token、区域 mean 聚合、global/local 通路或 UMLS 结构。新增监督不使用 segmentation。多病灶实例级属性绑定、角色分工、真实生成错误挖掘和 DPO/PPO 不属于本轮四个实验。

## 实验入口与隔离

基础配方直接调用 `scripts/ft_ablation/Exp1020/run_bindformer_va_fixed_region_mean_constlr.sh`。新 shell 通过 `experiments/report_output_v1/python_entry.sh` 将模块调用转发到独立 `entry.py`；训练和评估子进程继承同一扩展。旧 shell 不安装新 runtime。

实现文件全部位于 `experiments/report_output_v1/`。没有修改 `src/`、旧 E1/E2/E3、原 mean/adaptive shell 或原始 manifest。实现前保存的 114 个既有文件哈希与实现后逐一比较。

`runtime.install()` 仅在新进程入口调用：注册不增加参数的 projector 子类、模型子类，以及 Trainer 验证/保存包装。旧进程和旧入口不受影响。关闭 `enabled` 时不读取新监督数据，不增加模块，不改变初始化 RNG 或原 forward。禁止未验证的 DDP 和其他数据集组合。

## 四个任务

所有 shell 位于 `scripts/ft_ablation/Ep1023/`（目录名是 Ep1023）。

| 专用入口 | 任务入口 | GPU | tmux | 目标 |
|---|---|---:|---|---|
| `run1023_1.sh` | `run_report_output_r1_r32.sh` | 0 | adap | 32× R1 |
| `run1023_2.sh` | `run_report_output_r2_r32.sh` | 1 | mean | 32× R2 |
| `run1023_3.sh` | `run_report_output_r1_r64.sh` | 2 | value_ada | 64× R1 |
| `run1023_4.sh` | `run_report_output_r2_r64.sh` | 3 | value_mean | 64× R2 |

通用入口：`run_bindformer_report_output_constlr.sh`。支持 `--dry-run` 和原 sweep CLI 参数；额外 CLI `--set` 最后覆盖。

```bash
PROJECTOR_PYTHON=/mnt/sda/mingjie/conda_envs/projector/bin/python \
  bash scripts/ft_ablation/Ep1023/run1023_1.sh --dry-run

# 通用入口可切换到 16×；这不是本轮默认启动任务。
EXPERIMENT=R2 PROJECTOR_RATIOS=16 CUDA_VISIBLE_DEVICES=0 \
PROJECTOR_PYTHON=/mnt/sda/mingjie/conda_envs/projector/bin/python \
  bash scripts/ft_ablation/Ep1023/run_bindformer_report_output_constlr.sh --dry-run
```

基础设置保持：seed 42、projector/encoder LR 均 2e-4、effective batch 16、microbatch 1、视觉 LoRA rank 16 / alpha 32、两段 warmup 各 77 steps、hold 0、joint warmup `hold_peak`、cosine floor 1.0、最多 20 epochs、原 CE early stopping patience 3 / min_delta 0.005。其余 optimizer、prompt、预处理、采样和生成参数直接继承基准。每项从相同基础预训练组件独立初始化，不加载既有 finetune 或 smoke 权重。

## 数据

数据位于 `assets/knowledge/report_output_v1/`，构建命令：

```bash
python -m experiments.report_output_v1.build --output /path/to/a/new/version
```

构建器拒绝覆盖现有目录。读取原训练 facts 与全部候选配对；验证集使用冻结的训练词表及规则提取。记录 manifest、输入标签、提取源码和生成文件的 SHA-256，训练 preflight 验证一致性。

- `train.jsonl`：训练报告的属性证据、文本组、全部可靠候选。
- `val.jsonl`：验证属性与证据，不含训练负例。
- `audit.jsonl`：排除原因和证据；不是临床专家签字标注。
- `schema.json` / `metadata.json`：字段、范围、计数、哈希、限制。

| 数据 | 数量 |
|---|---:|
| 原训练报告 | 1,960 |
| 有保留属性的报告 | 1,034 |
| 保留属性标签 | 2,057 |
| 同句至少两属性且有病灶锚点的组 | 642 |
| 源候选有向配对 | 1,158 |
| 最终有向配对 | 1,112（556 对互为反例的病例） |
| 有负例的报告 | 788 |
| 完整验证报告 | 268 |
| 验证报告中的已知属性 | 281 |
| 验证可评估联合组 | 86 |

训练标签：laterality 814、site 241、signal 466、edema 536。配对：laterality 718、site 218、signal 134、edema 42。

相较旧标签，排除了“indicating edema or other pathology”等推测/替代解释，不能把这些文字当作确定水肿。全部源配对都有去向：保留或明确排除。46 条排除包括 5 条事实未保留、19 条不能可靠局部改写、22 条对应另一端没有可靠改写。失去对端的事实仍可参与 span CE，但不保留单向 ranking，以减少改写难易造成的答案偏置。

每条事实记录原文字符位置、属性值及断言。水肿 span 包含局部肯定/否定谓词，不能只给 `edema` 一个词施加相同监督。创建阴性负例时不能生成 `no mild/significant edema` 并将其视为完全无水肿。对 `edematous changes` 保留名词短语。

文本组是同句单病灶规则推断，`L1_text_heuristic` **不是图像区域标注或经核验的病灶实例 ID**。明确多病灶、双侧、歧义及多脑区仍按保守规则处理。缺失不等于阴性；无辅助标签的报告仍完整参与原报告 CE。

当前字段仅 `edema,laterality,site,signal`。强化和边界辅助监督保持关闭；**morphology/extent 暂无独立属性监督**，其 token 和完整报告 CE 保留。

## Loss 与梯度

R1：`L_report + 0.1 * L_span`。

R2：`L_report + 0.1 * L_span + 0.1 * L_group`。

这些是首轮配置值，不表示已证明最优。辅助项在前 77 optimizer steps 线性增加到配置权重。

### L_span

使用原报告 prompt 和 reference teacher forcing。同一次 Decoder 前向产生每个答案 token 的 NLL：完整答案计算原 CE，证据 mask 计算额外 span CE。每个字段先对 token 求均值，再在该报告的有效字段间求均值。

offset tokenizer 必须与真实 `apply_chat_template` token IDs 完全一致；显式处理模板对首尾空白的 trim、视觉 token 插入以及唯一一次 causal shift。EOT 参与报告 CE，不参与属性 score。任何错位直接报错，不近似定位。

### L_group

在相同图像 Z、相同报告任务中，对正确属性组和只替换一项事实的候选评分。候选保留此前报告前缀，截至该组证据末尾，避免后来重复正确属性造成文本矛盾捷径。只统计有标签的属性 span，先字段内 token 平均，再组内字段平均。

`pair_loss = softplus(margin - positive_score + negative_score)`，默认 margin 0.2。

所有可靠负例均参与：先对相同被修改字段的负例平均，再对该报告有负例的字段平均。一个报告有更多负例不会直接把权重扩大。整个累积窗口仍使用原 CE token 分母；辅助 loss 使用实际报告数分母，缺标签贡献 0。因此覆盖率会影响实际辅助强度，日志同时保留覆盖计数。

这不是 DPO，也不是对 `generate()` 返回的离散字符串直接反传。它监督报告位置上的条件概率；自由生成是否改善必须看单独验证。

### 显存受控的梯度实现

8B Decoder 冻结但对 Z 可微。`objectives.py` 在 Z 上依次计算报告/辅助 VJP，每个负例的 Decoder 图用完即释放，最后将两个 loss 的外部缩放系数乘回相应 VJP，继续传入原始 projector/encoder 图。

不是切断视觉梯度。单元测试与普通 autograd 的完整多负例实现比较 loss、Z 梯度和上游参数梯度。此接口只支持一阶训练和冻结 Decoder；不能不经改造直接开启 LLM LoRA。原视觉架构和参数集合不增加。

日志包含 span/group loss、逐字段计数与正确数、候选分差、加权辅助 loss、Z 梯度范数、辅助/报告均值梯度比和梯度余弦。Trainer 中这些值按实际样本数归一化；条件准确率需用 correct/count 计算，不能把日志里的正确数直接叫准确率。

## 自由生成验证与 checkpoint

原完整 validation CE、原固定生成子集和 early stopping 保留。新入口额外在每个 epoch 对完整 268 条 validation 自由生成报告，再使用只读取生成文本的提取器评估；reference 不进入图像编码、Z 或生成 prompt。

输出到每个训练目录的 `validation_report_facts/`：

- `step_XXXXXXXX.predictions.jsonl`：实际生成文本。
- `step_XXXXXXXX.facts.jsonl`：预测证据、逐事实决策和组结果。
- `step_XXXXXXXX.metrics.json`：固定分母事实指标、类别表现、遗漏/矛盾/未解析、覆盖率、生成长度与截断。

未知 reference 不设阴性。已知 reference 在预测中遗漏或无法解析均不算答对，不能通过减少输出提高准确率。联合指标只适用于明确记录的单病灶文本弱标签，不宣称解决多病灶绑定。解析覆盖率与专家抽查仍必须结合看。

`best.pt` 继续由原 validation CE 选择。额外 `best_facts.pt` 使用固定的 `macro_field_balanced_accuracy`，严格改善才替换，不改变训练终止规则。该指标先按 reference 类别求准确率，再按字段平均；缺失预测记错。

验证包装保存/恢复 CPU、CUDA、NumPy、Python RNG 和模型 train/eval 状态。验证频率/子集均显式配置并记录范围。best-facts 状态写入正常 checkpoint，恢复训练不重置已选结果。

原 sweep 结束后仍自动对 **best-CE 的完整 test/all（含 GREEN）** 评分。不会用 test 指标选择 epoch。best-facts 的验证输出已保留；后续若要对其进行额外 test 评测，应作为独立、明确标注的选点比较，不能只与旧 baseline 的 best-CE 比而宣称全部收益来自 loss。

补评已有输出无需 GPU：

```bash
python -m experiments.report_output_v1.evaluation \
  --predictions /path/to/predictions.jsonl --output /path/to/fact_metrics
```

## 配置开关

| 环境变量 | 默认 | 含义 |
|---|---|---|
| `EXPERIMENT` | R2 | R0/R1/R2；专用任务脚本固定其任务 |
| `PROJECTOR_RATIOS` | 32 | 通用脚本支持 16/32/64 等原有倍率 |
| `SPAN_ENABLED` / `GROUP_ENABLED` | 按实验 | 辅助项开关；两者关闭走原 forward |
| `SPAN_WEIGHT` / `GROUP_WEIGHT` | 0.1 / 0.1 | 最终辅助权重；关闭项权重设 0 |
| `GROUP_MARGIN` | 0.2 | 候选 ranking margin |
| `AUX_WARMUP_STEPS` | 77 | 辅助权重 warmup，独立于原 LR schedule |
| `GROUP_EVERY_STEPS` | 1 | 计算对比的 optimizer-step 间隔；不自动补偿稀疏频率 |
| `REPORT_OUTPUT_PATH` | assets/knowledge/report_output_v1 | 不可变监督版本 |
| `REPORT_OUTPUT_FIELDS` | [edema,laterality,site,signal] | 训练字段；固定评估器仍报告完整字段清单 |
| `VAL_FACTS_ENABLED` | true | 自由生成事实验证 |
| `VAL_FACTS_EVERY_EPOCHS` | 1 | 验证间隔 |
| `VAL_FACTS_LIMIT` | 0 | 0 表示完整 validation；非零明确标为子集 |
| `SAVE_BEST_FACTS` | true | 保存额外事实 checkpoint |
| `PROJECTOR_RUN_TAG` / `PROJECTOR_OUTPUT_ROOT` | 唯一新名称 / runs/report_output_v1 | 运行定位 |
| `PROJECTOR_PYTHON` / `PROJECTOR_LOCAL_CONFIG` | 继承原入口规则 | 环境与本机路径 |

底层键均为 `projector.params.report_output.*`，未知键、非法权重/字段会在加载大模型前报错。改变监督版本或 loss 设置后必须新开实验，不能冒充原设置 resume。

## 验证证据与运行状态

证据目录：`results/ReportOutput_v1_implementation_20261023/`。

- `configuration_audit.json`：四份 resolved config 与对应倍率 Baseline A 比较，仅新监督配置与运行目录不同；数据 ID、token budget 和 schedule 相同；无 init checkpoint/resume。
- `legacy_hashes_before.json`：既有源码哈希。
- `unit_tests.txt`、`regression_tests.txt`：数值与回归测试。
- `smoke_r32/summary.json`、`smoke_r64/summary.json`：真实模型 R1/R2 反传、视觉 LoRA、checkpoint resume、生成验证、RNG 和 best-facts 保存。
- `pipeline_smoke/`：独立短训练流程；所有短步数、短 warmup 和短生成仅用于 smoke，不进入正式配置。
- `launch_manifest.json`、`startup_report.json`：正式启动后记录 tmux、GPU、进程、目录与初始训练步状态。

Smoke 仅验证代码执行和数值链路，不能证明报告质量提升。正式任务启动后按用户要求仅观察前几个 optimizer steps，之后保留 tmux 后台训练，不在此阶段对质量收益作结论。

### 本次启动

2026-10-23 已按用户追加授权启动四项正式训练。会话内使用新窗口，保留原 `:0` 窗口：

- GPU 0：`adap:1`，R1-32×。
- GPU 1：`mean:1`，R2-32×。
- GPU 2：`value_ada:1`，R1-64×。
- GPU 3：`value_mean:1`，R2-64×。

运行根目录前缀为 `runs/report_output_v1/report_output_v1_{R1或R2}_r{32或64}_gpu{0至3}_20261023T171949Z/`。完整路径及初始 step、实际配置、日志路径见 `startup_report.json`；后续实时状态以各运行的 `joint_lora/train.jsonl` 和 `sweep_status.json` 为准。

最终检查包括 133 项回归测试、38 项目标/旧接口测试（与回归集有交集）、额外旧 CUDA RNG 测试、32×/64× 真实模型 smoke、6-step 完整 Trainer 流程及独立进程的恢复/评估。6-step 短流程的两个验证样本恰无可靠辅助标签，正确地不生成虚假的 best-facts；有标签时的 best-facts 保存由两种倍率的真实模型 smoke 另外覆盖。
