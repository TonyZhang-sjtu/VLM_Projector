# 3D VLM Projector 路线梳理与 Brain MRI Baseline 设计

初稿：2026-10-02；实现、单卡资源与 smoke 问题复核：**2026-10-12（PDT）**。

依据：本目录的《3D VLM Projector》调研记录，尤其是 Week 0831–0906、Chest/Lung CT 模型表、Projector Baselines 与后续补充文献；重新核对相关论文方法部分，并补充检索原始论文和官方实现。

适用设置：RadGenome-Brain MRI；NeuroVFM → Projector → Llama-3.1-8B-Instruct；后续用 3DINO 检验 encoder 泛化。当前输入为 **native-grid + minimal padding 的单序列完整体积**，输出该序列英文 Findings。**当前主线为连续最多 10 epochs：6.25% projector warm-up → 6.25% projector + 视觉 LoRA 联合 warm-up → 87.5% 联合 cosine；两组峰值 LR 均为 1e-5，LoRA rank16/alpha32；warm-up 后采用 val CE min_delta=0.005、patience=3 早停。** 全程冻结 LLM。两段 warm-up 可传 optimizer steps（优先）或比例，max_epochs/rank/alpha 均可覆盖；用户已确认，连续训练入口已实现。完整参数、实现与测试记录以 [训练方案第 8、13、14 节](NEUROVFM_MLP_TRAINING_PIPELINE.md#8-当前确认的-finetuning-方案2026-10-12) 为准；2026-10-12 已新增 joint_lora 连续模式，旧独立 A/B 模式继续保留；工程 smoke 证据见训练方案第 14 节。下文独立 projector-only / B-Full 作为另外安排的研究对照，不能将短 warm-up 当作已收敛 A baseline。这里的 LoRA 不是 LLM LoRA。

本文同时保存研究设计和已实现状态。**11 个 baseline 已实现；41 组真实短训练通过，但收敛和报告质量尚未由这次 smoke 验证。** 当前方法定义以第 13.2 节、压缩协议以第 15 节、资源/失败记录与类别覆盖建议以第 16 节为准。早期固定 M、混合来源、独立 A/B 的设计与审计保留为历史背景，不覆盖当前 BraTS-only、4/8/16/32× 主协议。MRI 适配实现不能直接称为参考论文完整复现。

## 0. 本轮结论与已经核实的工程状态

**原来的六类机制划分基本成立，主要缺口是直接相关的 Brain MRI 文献、标准实现定义和实验优先级，不需要继续按论文数量扩大主表。** 本轮最值得补入的是 AutoRG-Brain、NeuroVFM 自身 connector、brat，以及显式事实接口路线 PerFact / NeuroFusion / BTReport。Brain3D、MT-brainVLM、BrainVLM、Mr3D-VL、ReMIND、BrainDiff 用于补齐同领域定位，但并非每篇都要成为一个新 projector。

建议将结果分成三组：

1. **受控 projector 比较**：同一 encoder、最后层特征、LLM、可见单序列、训练集和生成损失，只替换接口。
2. **额外信息/监督扩展**：多层、区域/病灶 mask、句子/概念对齐、显式事实、序列条件化；每项都配相同信息与监督的对手。
3. **完整系统参考**：原作者 encoder/LLM/预训练/多序列/分割流程；重新在明确协议下评价，不能将论文原分数复制到本地排行榜。

本地核对结果：

| 项目 | 实际状态及对 baseline 的约束 |
| --- | --- |
| 任务与数据 | `single_sequence_findings_coref_v2`；当前新训练默认 BraTS，train/val/test 为 1,960/268/560 个序列；`--datasets all` 仍为三组混合的 2,177/299/622 个序列（701/98/202 位公开患者）。本地患者互斥划分和目标修订不同于原论文 |
| Encoder | `NeuroVFMEncoder` 输出最后层 `features[N,768]`、`coords[N,3]` 和 metadata；逐体积独立编码；A 可缓存，B 禁止最终特征缓存 |
| 已接入 Projector | registry 已有 **11 个方法**，见第 13.2 节；MLP 结构为 `768→4096→4096`，19,931,136 参数；各方法均有 Bash 入口及真实短训练证据 |
| 输入网格 | 当前 BraTS train/val 几何预检为 `39×15×15=8,775`；448–8,775 是早期混合 BraTS/ISLES/WMH 的范围。当前按目标 4/8/16/32× 解析各方法自身预算，逐样本再验证几何 |
| 空间接口 | 当前保留完整网格；`PatchVolume.validate()` 检查全部坐标。没有独立 token-validity-mask 字段；原图边缘的部分 padding patch 与未来压缩器补出的空槽要区分 |
| 训练记录 | 已有阶段 A 一轮及阶段 B 的运行日志；目录名或单个日志不构成多 baseline 收敛比较。`stage_b_lora_full` 的 trainable 清单实际是视觉 LoRA，不能凭目录名当作 B-Full |
| 评价 | BLEU/ROUGE、患者 bootstrap、zero/shuffle/text 控制已有入口；GREEN、RadGraph-XL、RaTEScore 的选定方案见独立评价文档，不能写成已计算完成 |

代码依据：[Projector registry](../src/models/projectors/registry.py)、[MLP](../src/models/projectors/mlp.py)、[Encoder](../src/models/encoders/neurovfm.py)、[网格契约](../src/preprocessing/base.py)、[当前配置](../configs/base.yaml)、[评价方案](EVALUATION_METRICS_AND_NEUROQA_PLAN.md)。本轮 train/val 网格统计与源码哈希见 [核对记录](../runs/archive/engineering/baseline_review_20261007/local_audit.json)。训练文档顶部的“当前实现”优先于其保留的历史设计。

## 1. 建议采用的分类与实验规模

按“encoder 输出怎样变成 LLM 输入”划分，可以使用 **6 类基础机制 + 3 条可叠加的增强路线**。这是为本项目建立的操作性分类，不是社区公认且互斥的唯一分类。

- 六类基础机制：逐 token 映射、固定空间池化、空间到通道重排、可学习空间混合、查询式聚合、内容自适应选择/合并。
- 三条增强路线：多层/多尺度特征利用；解剖区域、几何与关系建模；概念/任务/序列条件化。

原图六个 baseline 已覆盖其中四类基础机制。B0–B7 保留为文献机制编号，实际 registry 已有 11 个实现。当前按同目标压缩档位比较，保留各方法实际 M，Concat 也进入比较。若把内容自适应的“选择”和“合并”拆开，就是 **7 类**；每类一个代表需 7 个方法（含不压缩 MLP），不是 5 个。新的类别覆盖建议见第 10、16.4 节，尚未据此改变用户的实验选择。

区域、图结构、概念监督等增加了不同的信息或监督，应通过扩展实验比较。**“已覆盖一个大类”不意味着可以省略与未来创新点最接近的强对手**：例如研究 query 多样性时，brat-inspired 对照比再增加一个 CT 系统名更有价值。

“不照搬 CT”应理解为：保留方法的核心机制，调整输入接口、三维坐标、脑区定义和监督协议。**Baseline 不需要为了显得新而改掉原方法的关键设计。** 改造版用明确名称，并在论文中写清与原方法的差别；不能把删去重要分支后的实现称为原论文完整复现。

## 2. 先明确 Projector 的研究边界

统一研究接口：

```text
MRI volume
  → 固定版本的 vision encoder
  → features X[N,C] + coordinates + validity mask + sequence identity
  → token aggregation/compression
  → channel projection
  → visual embeddings Z[M,D_llm]
  → frozen Llama decoder
```

有的论文将 compression 与 projector 分开命名，例如先合并 token 再接 MLP；本项目把它们作为同一个“encoder–LLM 接口”整体比较，同时分别统计压缩器和映射头的参数与计算。不要因为命名不同，把一个纯 token merger 排除在有意义的对照之外。

需要区分三个问题：

1. **信息访问**：能看到最后一层、多个层、多个分辨率，还是额外 mask/文字？
2. **信息聚合**：平均、重排、空间混合、attention、选择/合并？
3. **学习与读取**：只有报告生成损失，还是额外概念监督；进入 LLM 输入端，还是注入其内部层？

同一方法可能同时占据多个位置。例如 TokenPacker 是空间局部约束下的查询聚合，所以 “Grid-based” 与 “Query-based” 不是严格互斥的两个集合。[TokenPacker 原文](https://arxiv.org/html/2407.02392v2)

### 对原记录的几个校正

| 容易混淆的表述 | 本项目采用的准确口径 |
| --- | --- |
| M3D 的 Spatial Pooling Perceiver 属于 Perceiver query 模型 | 官方 `SpatialPoolingProjector` 的 spatial 路径是 `avg_pool3d` 后接 Linear/MLP，不含 Perceiver cross-attention；归入固定池化。[官方代码](https://github.com/BAAI-DCAI/M3D/blob/main/LaMed/src/model/multimodal_projector/spatial_pooling_projector.py) |
| Argus PixelShuffle 是 ConcatAdapter 之外的全新路线 | 两者都可归入邻域 token 的空间到通道重排；实现顺序、分组和投影层需核对，但不必重复算一类。[Argus](https://aclanthology.org/2025.findings-acl.845/) |
| PixelShuffle + MLP 全过程无损 | 只有可逆重排步骤保留全部元素；后续降维映射不保证无损。减少 token 数不等于整个模块无信息损失。 |
| MLP 是性能下界或不压缩性能上界 | 它只是高 token 成本的参考；优化、冗余与 LLM 读取方式均会影响表现，没有理论上的性能上下界保证。 |
| Med3DVLM 的 projector 必须读取问题文本 | 方法部分描述的是两个视觉层分别经过 Mixer，再与文本 embedding 组成 LLM 输入。不能仅凭“multimodal fusion”的摘要措辞推断 Mixer 内部是 text-conditioned。[方法 §3.2.2](https://arxiv.org/html/2503.20047v2) |

## 3. 六类基础机制：每类的代表 baseline

| 类别 | 核心操作 | 建议代表 | 对应文献路线 | 主要检验问题 |
| --- | --- | --- | --- | --- |
| ① 逐 token 映射 | 每个 token 独立经过 Linear/MLP，M=N | **MLP** | LLaVA 式接口；RMR 不压缩参考 | 不做压缩时，冻结 LLM 能利用多少信息？ |
| ② 固定空间池化 | 按规则三维空间单元求均值 | **AveragePool3D + MLP** | M3D；RMR | 平均是否抹去稀疏小病灶、局部属性？ |
| ③ 空间到通道重排 | 邻域 token 拼到 channel，再投影 | **ConcatAdapter3D** | Argus PixelShuffle；RMR Concat | 保留局部元素排列是否优于先平均？ |
| ④ 可学习空间混合 | token/spatial mixing + channel mixing，再降采样 | **ConvPool3D 优先；Mixer3D 备选** | Honeybee C-Abstractor；AutoRG-Brain CNN projection；Med3DVLM MLP-Mixer | 不用查询 attention，学习空间交互能否改善压缩？ |
| ⑤ 查询式聚合 | 少量 query 从较多视觉 token 中提取信息 | **LocalPacker3D、AttentionPool、Perceiver** | TokenPacker、HSENet；RadFM/Flamingo；Ker-VLJEPA 的区域限制查询 | 局部约束、全局访问和迭代聚合分别有什么作用？ |
| ⑥ 内容自适应选择/合并 | 按样本特征决定保留与分组 | **AdaptiveMerge3D**，新增；selection-only 作诊断 | DivPrune、VisionZip、ORCA、医学 token 压缩工作 | 同一输出预算下，自适应分配是否优于固定空间分组？ |

第五类需要区分三个变量：**局部锚定/全局自由 query；一轮/多轮；query 之间是否交互。** Q-Former、Perceiver、AttentionPool 属于同一大类，但实现及训练目标不等价。无需一次跑全；若新方法包含 query 竞争、交互或语义对齐，就不能只用不含这些机制的弱查询对照。[RMR 方法比较](https://arxiv.org/html/2608.08713v1)、[Honeybee](https://arxiv.org/abs/2312.06742)、[BLIP-2](https://proceedings.mlr.press/v202/li23q.html)、[DivPrune](https://arxiv.org/abs/2503.02175)、[VisionZip](https://arxiv.org/abs/2412.04467)

## 4. 调研中的 CT 论文分别贡献什么

以下只提取与接口有关的机制，不把各论文完整系统的指标当作 projector 的独立效果。

| 论文/系统 | 核对到的机制 | 对 Brain MRI 的合理吸收方式 | 应单独控制的部分 |
| --- | --- | --- | --- |
| [M3D-LaMed](https://github.com/BAAI-DCAI/M3D/blob/main/LaMed/src/model/multimodal_projector/spatial_pooling_projector.py) | 三维平均池化后映射 | B1 AveragePool3D | 其 encoder 预训练、分割任务与 LLM 训练不随 baseline 一起引入 |
| [Argus](https://aclanthology.org/2025.findings-acl.845/) | 比较池化、Perceiver、空间重排及映射头 | B2 ConcatAdapter3D 已代表重排路线 | 原比较的 token 数并非所有配置都一致；本项目明确固定 M 与自然预算两种协议 |
| [RadFM 官方实现](https://github.com/chaoyi-wu/RadFM/blob/main/src/Model/RadFM/my_embedding_layer.py) | Perceiver 将视觉信息转换为紧凑 latent，再线性映射 | B5 Perceiver | 不移植整套 encoder、MedLLaMA 与多模态预训练 |
| [Med3DVLM](https://arxiv.org/html/2503.20047v2) | token/channel Mixer；DCFormer 两个层级的视觉分支 | B6 学习空间混合；E1 单独检验双层特征 | 更强/更多层输入、双分支参数和 encoder 本身都可能贡献提升 |
| [HSENet](https://arxiv.org/html/2506.09634v1) | 局部体素块汇聚得到 query，再从该块高分辨率特征做 Voxel2Point attention | B3 LocalPacker3D 代表局部查询聚合 | 原文的双视觉编码器与不同分辨率输入属于额外变量 |
| [Reg2RG](https://arxiv.org/html/2411.15539) | 全局特征 + ROI 纹理；另编码未裁剪 mask 的几何；区域报告监督 | E2a 区域纹理 + 几何；需要时额外比较 ROI 重编码 | 原方法 ROI 图像重新过 encoder，不能等同于一次编码后的 ROI pooling；区域间交互也不等同于显式 GNN |
| [MedRegion-CT](https://arxiv.org/html/2506.23102v3) | 区域引导的 SlowFast 特征、mask pooling，以及病灶属性文本 | E2a 中检验全局粗信息 + 区域细信息；E1 对应多层利用 | 原系统还用逐切片编码、伪 mask 与结构化属性提示；文字属性可能直接增加答案信息 |
| [CT-GRAPH](https://arxiv.org/html/2508.05375) | mask pooling 构造器官/系统/全局节点，通过层级 GAT 交互 | E2b 在相同脑区节点上比较 Flat 与 Graph | 图结构收益要在相同 mask、节点、坐标、token 数下测量 |
| [Ker-VLJEPA](https://arxiv.org/html/2603.23308) | slice embedding 按 z 区间限制 cross-attention，再跨区交互 | 将“限制查询可见区域”转成真正三维邻域；B3 已覆盖其基本思想 | 原系统还有 decoder 内部 cross-attention adapter、LoRA 和额外目标；不能整体移入纯 projector 表 |
| [ORCA](https://arxiv.org/html/2608.00345v1) | 在三维相邻区域间进行特征驱动合并，软解剖信息指导分组，并编码合并区域质心 | B7 可研究不使用器官 mask 的空间合并；有 mask 时应加 anatomy-guided 对照 | 去除 mask 后须标为消融/改造，不能称完整 ORCA；加入位置编码的对手也要有位置输入 |
| [CTInstruct](https://ojs.aaai.org/index.php/AAAI/article/download/37517/41479) | 视觉建模和多粒度任务之外，使用轻量线性 bridger | 不必额外增加“CTInstruct projector”类别 | 上游特征构造、动态长度与多任务训练不是线性桥本身的收益 |
| [CT-CHAT](https://github.com/ibrahimethemhamamci/CT-CHAT) | 完整 CT 视觉—语言系统与训练流程 | 作为 CT 迁移阶段的系统/代码参考 | 不因为系统使用不同名称，就增设一个结构未独立区分的 projector baseline |
| [Exploring the Design Space](https://papers.miccai.org/miccai-2025/paper/2261_paper.pdf) | 比较 MLP、SPP、3D TokenPacker 等，并讨论其他系统选择 | 支持 B0/B1/B3 的统一骨架比较 | 分辨率/tiling、LLM 更新与报告增强需分开；报告后处理不是知识型 projector |
| [Resolution Meets Reduction](https://arxiv.org/html/2608.08713v1) | 在 token 预算下系统比较图片中的六种接口 | 作为主实验协议参考 | 其训练设置包含 LLM 适配；本项目冻结 LLM，不能直接继承其性能排序 |

这张表的含义是：十多篇 CT 论文不对应十多种必须独立实现的 projector。它们有些共享底层压缩机制，有些主要增加输入信息、区域结构或训练目标。

## 5. 补充检索中需要纳入的工作

### 5.1 Jolia / ConQuer：概念查询，且不依赖分割 mask

Jolia 使用按概念组织的可学习查询，将报告分解成概念相关文本，并进行概念级对比学习；不依赖空间分割监督。这补充了“区域信息只能通过 mask 得到”的思路。但它首先是视觉—语言预训练方法，不能直接当成已在冻结 NeuroVFM + Llama 接口上验证过的 projector。[原文](https://arxiv.org/html/2606.24570)

对本项目的启发：E3 用脑部概念约束查询读取哪些视觉证据，并与无语义约束的 query baseline 比较。若以后创新点只是“加入医学概念 query”，差异将不足，需要证明进一步解决了实例、属性或关系的特定失败。

### 5.2 mpLLM：原笔记引用了数据集，但其 projector 本身也相关

最新版论文重点是 **问题条件化的层级 MoE projector**，在序列级和 token 级路由到共享/序列专属的投影专家，用于多序列 Brain MRI VQA。这是后续研究 sequence-aware 或 task-aware projector 时必须面对的直接相关工作。[原文方法](https://arxiv.org/html/2509.25889)、[作者代码](https://github.com/arvindmvepa/mpllm)

当前缺少同次输入的多序列融合与多问题变化，因此可暂缓完整层级 MoE。但**单序列输入不等于单一对比度训练**：当前混合 T1W/T1C/T2W/FLAIR/ADC，序列身份已在 prompt 中提供。可先用共享 MLP + sequence embedding/FiLM 检验简单条件化，勿把其收益误称为复现 mpLLM。多序列/VQA 阶段再加入完整的共享/专属专家和问题路由。

### 5.3 Honeybee：可学习空间混合不能漏掉

Honeybee 的局部建模说明，压缩不只有固定池化与全局 query 两条路径；其卷积式与局部 attention 设计为 B6/B3 提供通用参考。脑 MRI 需要三维邻域与有效 token mask，不能直接复用二维网格操作。[原文](https://openaccess.thecvf.com/content/CVPR2024/papers/Cha_Honeybee_Locality-enhanced_Projector_for_Multimodal_LLM_CVPR_2024_paper.pdf)

### 5.4 Token 选择/合并：主表需要代表，但不要把所有插件改名成 projector

- **DivPrune / VisionZip**：分别代表多样性选择，以及保留重要 token 并汇总上下文的思路。可用于 B7 的 selection-only 和 merge 对照。[DivPrune](https://arxiv.org/abs/2503.02175)、[VisionZip](https://arxiv.org/abs/2412.04467)
- **MedPruner**：面向切片序列，包含跨切片过滤和基于 attention 的动态 token 选择。原生三维 patch encoder 不天然提供完全相同的切片接口；不可直接删除深度片再宣称等价适配。[原文](https://arxiv.org/html/2603.11625)
- **CARVE**：强调跨切片证据和各向异性的预算分配，适合提醒我们区分平面内与层间冗余；需要核对其切片网格假设后再迁移。[原文](https://arxiv.org/html/2608.04515)
- **MedARC**：同时使用视觉 attention、问题相关性和额外 3D VFM 的结构信息，并合并冗余 token。单 encoder、固定提示实验不能将这些额外信息的收益算给压缩算子本身。[原文](https://arxiv.org/html/2607.26554)

### 5.5 相关但不应塞入当前纯 projector 主表

| 工作 | 相关性 | 实验归属 |
| --- | --- | --- |
| [ConceptCLIP](https://www.nature.com/articles/s41551-026-01764-x) | 全局图文与区域—概念对齐 | 作为 E3 的监督思想参考；不是现成 3D→LLM projector |
| [Matryoshka Multimodal Models](https://arxiv.org/abs/2405.17430) | 同一模型支持嵌套视觉 token 预算 | 属于预算/训练策略；优先先完成独立预算比较，再研究共享权重多预算训练 |
| [BTB3D](https://arxiv.org/abs/2510.20639) | 学习更合适的三维视觉 token | 改变上游 tokenizer/encoder；固定 NeuroVFM 时不算独立 projector 对照 |
| [U-VLM](https://github.com/yinghemedical/U-VLM) | 分层视觉—语言建模与多层连接 | 会改变 decoder 连接方式，作为架构扩展参考 |
| [Photon](https://arxiv.org/html/2603.25155) | 问题条件化 token 调度及离散选择的梯度训练 | 在语言模块内部选定层进行 token 缩减，属于内部计算策略；不是只更换输入端 projector |
| [When Do Cheap Probes Predict Expensive Training?](https://arxiv.org/abs/2607.22771) | 用廉价探测辅助判断下游配置 | 参考评估方法；不能未经本数据集验证就用 probe 排名代替生成任务排名 |

上述检索用于覆盖有实质差别的路线，不意味着已经穷尽所有 token 压缩论文。优先深比较与本项目输入、信息访问和监督条件相近的方法。

### 5.6 直接相关的 Brain MRI 工作：本轮补充的重点

下表的“优先级”是本项目相关性判断，不是论文质量或模型性能排名。预印本结论尚未由本项目复现；“有代码”也不代表完整训练数据、权重和复现环境均可用。

| 工作与来源 | 核实到的接口/任务 | 对本规划的影响 |
| --- | --- | --- |
| **AutoRG-Brain**，[论文 v3](https://arxiv.org/html/2407.16684v3)、[官方代码](https://github.com/ljy19970415/AutoRG-Brain) | 当前数据集的原始工作；将全局特征与 mask 指定区域特征按通道拼接，再做 CNN projection；另有 all-one mask 的 Global 模式 | **必引、优先系统参考**；支持 B6-C 卷积混合及 E2 区域路线。官方 RadGenome 的 `given_mask` 命令提供 GT 病灶 mask，必须与预测 mask/无 mask 条件分开 |
| **NeuroVFM 官方 connector**，[本地实现](../third_party/neurovfm/neurovfm/models/vlm.py)、[官方仓库](https://github.com/MLNeurosurg/neurovfm) | 每个序列先经过 PerceiverResampler，再接 MLP 到语言维度；不是当前自建的不压缩 MLP | **必做受控代表** B5-NV；沿用结构并在本地相同数据上训练，不将官方整个报告模型或预训练 connector 权重直接当作公平初始化 |
| **brat**，WACV 2026，[正式论文](https://openaccess.thecvf.com/content/WACV2026/papers/Kayser_brat_Aligned_Multi-View_Embeddings_for_Brain_MRI_Analysis_WACV_2026_paper.pdf)、[作者实现/权重](https://github.com/maximek3/brat) | Q-Former-like 多视角 latent 与报告句子对齐，PVA 匹配 + DPP quality-diversity；报告生成实验将视觉特征/多视角 embedding 经 MLP 接 Llama | **最重要的新近机制相关工作之一**；E3c 加入句子对齐/多样性，而不只考虑固定医学概念。这里 multi-view 指多个表征，不是简单的轴/冠/矢三平面输入 |
| **Brain3D**，2026-02 预印本，[方法 III-C/D](https://arxiv.org/html/2602.22098v1) | 2D→3D encoder inflation；flatten 后 AdaptiveAvgPool1D 到 32 tokens；两层 MLP + 可学习视觉标量；对比对齐、projector warmup、LLM LoRA | 机制仍属池化+MLP；可补 B1-Flat1D，不能写成 AvgPool3D 完整复现。视觉尺度 gate 和对齐损失是独立消融；其最终 LLM 更新不同于本项目 |
| **MT-brainVLM**，2026-09-30 期刊论文，[方法与可用性声明](https://link.springer.com/article/10.1007/s44443-026-01302-4) | 同用 RadGenome；三组层级 query 对应 modality-wise/global/impression；Q-Former 有 self/cross-attention，同时做分割与分类 | 相关工作须补；若研究层级/语义 query 则成为 E3 强对手。不能把多任务、多序列或其他报告层级的标签悄悄加入当前主表；实现需向作者申请，文中 AutoRG 链接不是其实现 |
| **NeuroFusion**，2026-09 预印本，[The Diagnosis a Reporter Leaves Unspoken](https://arxiv.org/html/2609.02411v1) | 冻结肿瘤分割 backbone；病灶实例路由、位置编码、每病灶 Q-Former；属性头预测序列化成文本，接 QLoRA decoder | 与“实例—属性绑定”“encoder 可读但 LLM 未表达”的论点直接相关；对应 E2 实例区域 + E4 显式事实接口，而非只加一个普通 query 模型 |
| **PerFact**，2026-08 预印本，[论文](https://arxiv.org/html/2608.17926v1) | 使用 RadGenome 报告；上游完整 3D 分割/分类产生事实文本，送入看少量切片的 LoRA VLM；比较事实、检索及 oracle | **显式事实路线必引**；其切片输入和多序列信息条件与当前不同，不能从其结果直接断言本项目 projector 无用。应在同一 NeuroVFM 信息上检验事实接口 |
| **BTReport**，2026，[论文](https://arxiv.org/html/2602.16006v1)、[代码](https://github.com/KurtLabUW/BTReport) | 从图像、分割等确定性提取定量/解剖属性，LLM 负责组织语言；不是连续 embedding projector | 支持 E4 的 features→facts→text 系统对照；mask 来源、多序列信息、属性抽取器和 LLM 成本须计入 |
| **BrainVLM**，[官方实现](https://github.com/HKU-HealthAI/BrainVLM) | 3D MRI encoder + MLP + Llama-3.1-8B；面向多模态脑肿瘤诊断，输入可含多序列、模态信息、年龄/性别 | LLM 和场景接近，但不是新 projector 大类；归入多序列诊断系统参考，不与全体单序列 Findings 原分数直接比较 |
| **Mr3D-VL**，2026-08 预印本，[方法 §3](https://arxiv.org/html/2608.12689v1) | 多参数 MRI，MedNeXt 视觉骨干、多分辨率投影/视觉 token 注入，并改动语言侧位置编码 | 补齐多层与空间接口相关工作；改变 decoder 内部连接，不能当作只替换输入 projector。论文同时使用 MLP 与 ViT 投影描述，具体实现仍需代码确认 |
| **ReMIND**，[作者论文](https://pmc.ncbi.nlm.nih.gov/articles/PMC13060454/) | 多序列 MRI 转为切片/video 输入，适配 Qwen3-VL；关注模态缺失下的报告约束 | 若需要完整系统外部参考，选一个切片/video VLM 足够；它检验输入表示与整套预训练，不是 NeuroVFM projector 的受控行 |
| **BrainDiff**，2026-09 预印本，[论文](https://arxiv.org/html/2609.00593v1)、[代码](https://github.com/jhuldr/BrainDiff) | 继承 NeuroVFM 视觉和 Perceiver stack，输入两个时间点、多序列及既往报告，训练语言 LoRA | 单次单序列任务暂缓复现；视觉错配/报告依赖的诊断有参考价值，纵向变化任务须另建协议 |

**AutoRG 的实际比较方式**：原系统结果另列，至少写清 `Global/all-one`、`predicted-mask` 或 `GT-mask oracle`。使用本地修订后的相同目标重新打分；公开 checkpoint 是否见过本地测试患者需要核验，未知就注明，不能自动视为干净的零样本测试。若只在 NeuroVFM 上实现 global + region fusion，应命名 `AutoRG-inspired`，不能称完整 AutoRG 复现。

**brat 的实际比较方式**：先在相同 NeuroVFM tokens 上实现 query + 句子级对齐/多样性损失，以隔离方法；若直接使用 brat 预训练视觉/Q-Former 权重，则归入 encoder/预训练迁移实验。作者仓库提示部分现有权重不同于论文版本，需固定版本，而不是混用后引用论文结论。

### 5.7 通用 VLM 中需要明确补名的 Q-Former

[BLIP-2](https://proceedings.mlr.press/v202/li23q.html) 本来就以冻结视觉 encoder 和冻结 LLM 为核心条件，与阶段 A 的问题相近。应在文献分类中显式列入，但无需因此把核心表扩成所有 query 变体的集合。

- **结构对照**：query self-attention + cross-attention + FFN + 映射头；仅使用相同生成 CE，称 `QFormer-style-CE`。
- **训练策略对照**：引入图文对齐等预训练目标，单独控制数据、步骤、文本编码器与参数初始化。仅从头训练几个 query block，不等于完整 BLIP-2 预训练。
- 新方法若依赖 query 交互/分工，优先增加该结构对照或 B5-F；若依赖语义辅助监督，优先 E3c。两种检验不应混为“Q-Former vs Perceiver”一个标签。

## 6. B0–B7 候选 baseline 的实现规格

公共记号：`X ∈ R^(N×C)`，`M` 为总输出视觉 token 数，`D_llm` 从 decoder 配置读取。NeuroVFM 与 3DINO 的 C 不应硬编码到 projector 内。

第一轮固定预算表建议 M=256，对代表模型增加 M=128、64；**M 是待冻结的实验设计，不是已跑配置。** 本轮已确认变长网格，B2 的自然重排长度并不固定，须按第 9.6 节另列预算，不能用重复 token 或隐蔽截断伪造匹配。

| ID | 名称 | 推荐构造 | 输出与需记录的差别 |
| --- | --- | --- | --- |
| B0 | MLP | `Linear → GELU → Linear`，逐 token 映射 | M=N；单独列不压缩参考，记录真实成本 |
| B1 | AveragePool3D | 依据三维坐标分组；有效 token 做 masked mean；接 MLP | M 个空间单元；统计空单元，padding 不参与均值 |
| B2 | ConcatAdapter3D | 每个固定三维邻域按坐标顺序拼接 channel，再投影；边缘空槽显式标记 | 分组 k 时输入宽度 kC；自然 M 随输入网格变化，先入成本参考表；固定 M 适配须另命名 |
| B3 | LocalPacker3D，last-layer | 空间降采样/插值得到 query；每个 query 对对应三维邻域做 cross-attention；残差/FFN 后映射 | M 个局部锚点；核心表只读最后一层，明确这是受控改造版 |
| B4 | AttentionPool | M 个自由学习 query，一轮全局 multi-head cross-attention + FFN，映射到 LLM | M 固定；所有有效输入 token 可见 |
| B5 | **B5-NV：NeuroVFM connector**；B5-F 作变体 | B5-NV 为多轮视觉 cross-attention + FFN + MLP；B5-F 的 K/V 含 latents | 本地 NV 类默认 6 层/64 queries，预算实验显式设置 M；层数与 query 交互分别控制，见下文 |
| B6 | **B6-C：ConvPool3D**；B6-M：Mixer3D 备选 | 两个局部残差 Conv3D/channel-mixing block，在原始特征网格混合，再规则池化到 M、接 MLP | C 版本适配变长网格较直接；M 版本需轴向/窗口长度策略。均为机制适配，不冒充原论文完整结构 |
| B7 | AdaptiveMerge3D | 在有效三维邻域图上，按特征相似性进行相邻区域合并到 M；区域特征加权平均，保留质心/范围 | 主表不使用器官/病灶 mask；属于空间自适应合并改造基线，不称完整 ORCA |

所有内部宽度、归一化、坐标输入及映射头顺序需要配置化。默认先在视觉维度聚合，再使用共同的映射头；另做 `MLP→pool` 顺序消融时单独标记，非线性 MLP 与均值通常不交换。B2 的 kC 输入导致额外参数/计算，应如实计入，不能只报末层尺寸。

### B5：NeuroVFM、Flamingo、Q-Former 不能只凭名字视为同一个实现

本轮逐行检查 [NeuroVFM Perceiver](../third_party/neurovfm/neurovfm/models/perceiver.py)：`q` 来自 queries，`k/v` 只来自 `visual_features`；每轮接逐 query FFN，**没有 query self-attention，也没有把 queries 拼到 K/V**。它有迭代读取，却没有显式 query 间交互。原文档要求 B5 带 latent 交互，不能直接套在此官方类上。

| 变体 | 视觉 cross-attention | query 间交互 | 建议用途 |
| --- | --- | --- | --- |
| B4 AttentionPool | 一轮 | 默认无 | B5-NV 的浅层结构对照；额外匹配输出 norm/head 后才可作纯深度消融 |
| B5-NV | 多轮，K/V=X | 无显式交互 | 当前优先实现的同 encoder 原生接口参考 |
| B5-F | 多轮，K/V=[X;Q] | 同一 attention 内可读其他 latent | 标准 Flamingo 风格变体，必要时作为更强 query 对手 |
| QFormer-style-CE | 多轮 | 独立 query self-attention | 研究 query 协作/分工时的结构对照 |

Flamingo 行依据 [OpenFlamingo `helpers.py`](https://github.com/mlfoundations/open_flamingo/blob/main/open_flamingo/src/helpers.py) 的 `kv_input = torch.cat((x, latents), dim=-2)`。以上是结构区分，不是性能排序。默认先随机初始化各自 connector；使用外部预训练 connector 须另列。

### B3：不要不加说明地削弱 TokenPacker

原 TokenPacker 利用多层细粒度特征；RMR 的受控版本使用最后一层。因此有两个不同问题：

- 核心表 B3：与其他方法使用同一个最后层 X，衡量局部查询聚合。
- 多层扩展：加上真实多层输入，保留原方法的细节注入思想；如果新方法读取多个层，就必须给 TokenPacker 与其他强对手同等访问权限。

不能用仅最后一层的 B3 去证明“新多层方法优于完整 TokenPacker”。[TokenPacker](https://arxiv.org/html/2407.02392v2)、[RMR](https://arxiv.org/html/2608.08713v1)

### B6：将“学习聚合”与“双层输入”分开

Med3DVLM 提供两个启发：Mixer 本身，以及多层视觉特征利用。B6-M 只检验前者，E1 检验后者。**本轮建议先做 B6-C**：以共享卷积核建模局部空间，避免先固定全局 token-mixing 矩阵长度；与 B1 共用最终网格和映射头。它代表 AutoRG/Honeybee 的局部可学习混合思想，不等于这两个系统的完整复现。

对于 N 很大的三维特征，不建议直接使用成本随 N² 增长的全局稠密 token-mixing 矩阵。固定形状时可采用轴向 Mixer；变长/稀疏输入先通过统一坐标接口分桶，使用窗口 Mixer。若改用 `Conv3D + channel MLP`，应另命名 `ConvPool3D`，可代表同一大类，但不要宣称已复现 MLP-Mixer。

学习混合应在最终平均降采样之前有机会读取细粒度信息。否则“先强平均、再加一个大网络”不能检验原始细节能否得到保留。

### B7：优先使用明确且可复现的合并规则

建议以相邻区域的 Ward 合并代价作为初始实现：每个有效 token 起始为一个区域，每步合并相邻且代价最低的一对，直至得到 M 个区域；没有 mask 的版本不引入解剖标签。ORCA 的邻接约束合并是直接参考；这只是其无解剖引导的适配/消融，不是新算法。[ORCA 方法](https://arxiv.org/html/2608.00345v1)

稀疏前景图可能具有多个连通分量。若连通分量数超过 M，相邻合并无法达到目标；应在协议中预先定义预算下限或明确的跨分量回退规则，并记录触发率，不能悄悄改变方法。

另做两个低成本对照：相同预算下的随机选择；特征多样性 selection-only。后者可沿用 DivPrune 思路。它们用于区分“保留代表点”与“汇总剩余信息”的收益。

这里的“training-free”只描述选择/合并规则：接到新 NeuroVFM–Llama 组合时，后面的对齐 MLP 仍然需要训练。如果只在推理时插入一个插件，应该从同一个已训练父模型出发另做实验，不能混同于从头训练的 projector 比较。

## 7. 三条增强路线与显式事实接口对照

### E1：多层/多尺度特征利用 → DualLevel 接口

**若主要假设是最后层遗漏了中间层信息，优先做此扩展。** 若问题是 query 塌缩/语义遗漏，E3c 的优先级可能更高；不预设所有项目都先做双层 Mixer。

构造：从同一 encoder 提取中间层和最后层；每层各自做通道适配与同类聚合，分别输出 M/2 个 token，再拼成总计 M 个 token。可用 ConvPool、Mixer 或 query 作为固定聚合器；若选择 Mixer，命名为 DualLevel-Mixer3D。NeuroVFM/3DINO 的不同 Transformer 层不自动等同于不同空间分辨率；真正多分辨率 crop/重新编码应另列实验。

必要对照：

1. 单最后层单分支 B6。
2. 双最后层双分支：两条分支都访问最后层，匹配分支和近似参数。
3. 中间层 + 最后层双分支：检验额外层信息。
4. 多层 LocalPacker / Perceiver：若新方法依赖多层输入，应提供同等输入的强对手。

各层按相对深度选择并事先固定；在不同 encoder 之间不要机械套相同层号。不能通过测试集结果决定选哪些层。[Med3DVLM](https://arxiv.org/html/2503.20047v2)、[TokenPacker](https://arxiv.org/html/2407.02392v2)

### E2：解剖区域、几何与关系 → FlatRegion-Geometry，随后加 Graph

**E2a：FlatRegion-Geometry。** 从一次 whole-volume 编码的特征中，根据脑区 mask 做区域内聚合，保留少量全局 token；加入区域质心、空间范围和序列来源，映射后送入 LLM。预算固定为 `M_global + Σ M_region = M`，几何可以加到区域 embedding 中，避免偷偷增加额外 token。

脑区应来自 MRI 适用的 atlas/预测分割，具体粒度按本地可用标注与报告覆盖率确定；不能沿用胸部器官列表。先覆盖左右半球、脑叶/深部结构等确实可得到的区域，并保留没有被区域定义覆盖的全局背景。脑区并不等同于病灶实例，一个脑区可能包含多个异常。

这里是 Reg2RG/MedRegion 思想的简化机制基线，不等同于它们完整系统。若要研究重新裁剪编码的高分辨率收益，再增加统一 crop 预算实验，并统计额外 encoder 成本。[Reg2RG](https://arxiv.org/html/2411.15539)、[MedRegion-CT](https://arxiv.org/html/2506.23102v3)

**E2b：RegionGraph。** 在 E2a 的相同节点、输入特征、几何、token 数上加入 GAT/图消息传递，比较脑区之间或脑区—半球—全局之间的信息交互。若引入父节点，则 Flat 对照也应具有这些节点，并从同一个总 M 中分配预算。[CT-GRAPH](https://arxiv.org/html/2508.05375)

关键对照：无几何/有几何；Flat/Graph；真实层级/随机边；同等参数的无结构 attention。这样才能判断收益来自几何暴露、更多参数，还是解剖关系。

**Mask 协议必须明示：**

- 从输入图像预测得到的 anatomy mask 可用于可部署扩展，但记录分割器及其训练数据。
- 若分割器用本项目病例训练，应避免训练/测试泄漏；训练集伪 mask 可采用交叉验证生成。
- 人工病灶 GT mask 若仅作为训练监督或评估标签，要与作为推理输入区别开。
- 测试时直接输入病灶 GT mask 属于 oracle 实验，不能用其结果声称无需额外标注的 projector 提升。
- 如果新方法使用 mask，至少比较同 mask 的简单区域 pooling 和 anatomy-guided merging，而不只比较无 mask 的 B1–B7。

### E3：语义/条件化 → ConceptQuery；多序列/VQA 再加 MoE

**E3a：ConceptQuery。** 保持 B4/B5 的查询压缩接口与总 M，给其中 K 个 query 指定脑部概念角色，其余为自由视觉 query。用训练集报告生成概念存在、否定、不确定或区域相关的辅助目标；未提及概念不自动视为阴性。概念字典与抽取规则在训练阶段固定。

推理输入是图像和统一概念字典，不是该患者的真实报告或真实阳性标签。概念查询仍需从图像中读取证据。单个概念 token 未必能区分多个病灶，这恰好是后续实例绑定研究需要检查的限制。[Jolia/ConQuer](https://arxiv.org/html/2606.24570)、[ConceptCLIP](https://www.nature.com/articles/s41551-026-01764-x)

必要对照：自由 query；概念 query 不加辅助监督；概念 query + 辅助监督；普通 query 接同样监督头。额外监督不能只给新模型。

**E3b：序列条件化，完整 MoE 暂缓到多序列/VQA。** 当前先比较共享 MLP 与共享 MLP + sequence embedding/FiLM；所有方法仍获得相同序列 prompt。只允许 T1W/T1C/T2W/FLAIR/ADC 身份，不输入 GLI/MEN/ISLES 等疾病队列名。完整扩展参考 mpLLM，设置共享专家与序列专属专家，由图像和可见问题表示产生路由权重。保持最终视觉 token 预算一致，报告总参数和激活参数，另加参数匹配对照。保留配准信息，不能假设不同序列相同 token 下标就是同一解剖位置。[mpLLM](https://arxiv.org/html/2509.25889)

E3a 主要改变查询的语义监督；E3b 主要改变投影的条件化方式。两者属于同一“条件/监督”增强维度，但不是等价算法，不需在当前单序列固定提示任务中同时强行实现。

**E3c：SentenceAlignedQueries / DiversityQueries（brat-inspired）。** 与固定概念字典相比，它检验报告句子级监督能否使不同 query 保存不同证据。保留相同 query 结构、M、LLM 和数据，分别比较：生成 CE；CE + 句子对齐；CE + 多样性；CE + 两者。加入普通 query 获得同样辅助目标的对照，避免把监督收益归给专门设计的 query。语义 query 不等同于病灶实例，必须另检验跨病灶属性绑定。

句子仅从当前 train 的可观测单序列 Findings 构建；固定文本编码器/分句规则；val/test 报告不进入 query 构造或推理。训练文本编码成本也计入。PVA/DPP 的具体实现若简化，应标 `inspired`；不能把任意 cosine diversity penalty 称为 brat 完整复现。[brat 方法](https://arxiv.org/html/2512.18679v1)

### E4：连续视觉 token 与显式事实，作为接口/系统扩展

该路线由 PerFact、BTReport、NeuroFusion 提醒：能够从视觉特征预测的属性，是否需要明确说给 LLM？它**不属于第七种纯压缩算子**，但在研究“可读而未表达”时是必要替代方案。

先做可归因的小对照：同一冻结 NeuroVFM，使用相同 train-only 标签训练属性头；比较 `visual tokens`、`predicted facts`、`visual tokens + predicted facts`、`visual tokens + 同属性监督但不输入事实`。最后一项用于区分辅助监督与文本显式暴露的收益。需要训练/调整 projector 时统一训练预算；LLM 仍遵守当前冻结约束。

所有字段必须能从当前单序列观察；不能把多序列 GT 分割定义的“强化肿瘤”等信息自动作为非增强 T1 的可部署输入。头预测是推理输入，真实标签只用于训练/评分；`oracle facts` 另列。训练阶段尽量使用 out-of-fold 预测事实或预先声明的噪声方案，避免训练时总看完美事实、测试时才看噪声事实。

事实文本会增加上下文。除了视觉 M，还要报告事实文本 token 数、总前缀长度、抽取器监督和成本；它不满足纯主表的“同信息/同 M”条件。只有研究知识检索时才另加 RAG，检索库限 train；无需当前先做多个 RAG 系统。

## 8. MRI 与 NeuroVFM 适配中的实际问题

### 8.1 NeuroVFM 不是可以盲目 reshape 的稠密 token 序列

上游源码支持前景 token 过滤、显式三维 `coords` 和打包的 `series_cu_seqlens`。例如 `tokenize_volume` 默认 patch 为 `(4,16,16)`，可返回过滤后的坐标。**当前项目适配器则固定不过滤、完整有序网格、逐体积编码**；因此可以在验证 `grid_dhw`/坐标顺序后 reshape，不必为了主表假定已经存在稀疏输入。未来启用筛选才需要补全稀疏契约。

本地依据：[preprocess.py](../third_party/neurovfm/neurovfm/data/preprocess.py)、[models/vlm.py](../third_party/neurovfm/neurovfm/models/vlm.py)。以上相对链接从本文件目录返回项目根目录后进入 `third_party`。

因此实现时：

- AvgPool、Concat、Mixer、LocalPacker 均按真实坐标分组，或先 scatter 回有有效性标记的网格。
- 不把 packed list 中每连续 k 个 token 当作三维邻居。
- 被滤除/填充位置不能当作零值组织参加平均或 attention；Mixer 的空位需要一致且明确的处理。
- 不跨病例或跨序列边界拼接邻域。
- 3DINO 适配须提取 patch features，不能只使用分类式全局 embedding。

### 8.2 空间单位和 MRI 序列信息

三维 patch 不一定在物理空间中是正方体。归一化体积位置适合表达相对位置，物理 spacing/尺度用于描述大小；两者用途不同。应统一方向、保存 affine 与预处理变换，并在需要物理邻域时将 patch index 转成物理坐标。

同一种显式坐标输入应公平提供给所有可使用它的方法；新方法不能只因多看了位置而被解释成“保留了更多视觉内容”。同时保留无显式位置编码的消融。

MRI 不使用 CT 的 HU/肺窗先验。多序列时记录 T1/T1CE/T2/FLAIR/DWI/ADC 身份和缺失情况；当前单序列比较也可以给所有模型相同的序列标识，避免把观测条件差异误当作 projector 问题。

### 8.3 任务目标必须与可见影像相符

当前 RadGenome 单序列 whole-volume 实验使用对应的有效序列级报告，按患者/病例划分，不能让同一病例的不同序列跨 train/test。不能把仅一条序列当输入，却要求生成包含其他序列观察结果的完整多序列病例报告。数据缺失筛选规则对所有 baseline 相同。

病例级多序列报告生成应作为后续独立任务，重新定义输入集合、预算和目标，而不是在同一主表中混用两种任务。

### 8.4 接入压缩方法前需要同步的代码契约

本轮仅记录待改项，不修改训练实现：

1. `ProjectedVolume` 已有 `input_tokens/output_tokens/metadata`，但 [multimodal.py](../src/models/multimodal.py) 的 `project()` 当前只取 `.embeddings`，丢掉了其他字段；后续应保留输出长度及区域/坐标元数据。
2. [generate.py](../src/evaluation/generate.py) 当前写 `visual_token_count=len(v.patches)`，对 MLP 的 M=N 恰好正确，对压缩器会记录成 N，`text` 控制也应记录实际 M=0。新增 `encoder_token_count/projector_token_count/total_prefix_tokens` 并以真实张量核验。
3. 背景 patch 是当前共同可见的信息，不能当成 padding 排除；边缘部分 padding patch 仍保留。压缩器补出的完全空槽需要单独有效标记。不能使用 GT 病灶 mask 当有效性 mask。
4. E1 多层输出目前未暴露；新增时缓存 key 必须包含层选择、归一化及输出契约，不能复用现有最后层 cache 冒充多层特征。
5. B7 的离散分组规则可无梯度，但 B 阶段区域均值/MLP 的数值路径要保留 encoder 梯度；若把特征搬到 NumPy 后直接返回合并值，会切断 B 的训练。组索引可离散，聚合值应从原 PyTorch features 计算。
6. E3b 的序列身份目前传给任务 prompt，projector 的 `(features,coords,metadata)` 并未显式收到 `sample.sequence`；应加入经过白名单过滤的序列字段，不把整个 sample（含目标/疾病队列）传入。

## 9. 怎样做出能回答 motivation 的公平实验

### 9.1 控制条件

| 变量 | 统一规则 |
| --- | --- |
| Encoder | 同一权重、预处理、输入体积与特征提取位置；换 3DINO 另成一组 |
| 信息访问 | 主表最后层；多层、mask、额外文本/模型分别做扩展 |
| Decoder | 同一 Llama-3.1-8B-Instruct；全部参数冻结；量化、prompt、生成设置一致 |
| Token 预算 | 所有视觉分支与几何/区域 token 的总和为 M；同时报告有效 token、padding 和实际长度 |
| 参数 | 先做同 M 的机制比较，再为关键优势配置增加近似参数匹配对照；不能假设相同 hidden size 等于相同参数 |
| 搜索预算 | 相同验证集和相近的学习率/宽度/深度搜索次数，不只精调新方法 |
| 数据/训练 | 相同 split、样本、目标、有效 batch 和最大更新预算；统一验证频率/选点规则；报告实际 early-stop 步数，不声称早停后步数仍相等 |
| 额外监督 | 主表仅生成损失；辅助标签、mask、概念目标在对应实验中公平提供 |
| 报告成本 | projector 参数、encoder/LLM 成本、峰值显存、预填充延迟、端到端延迟；额外分割器与 CPU 合并也计入 |

参数匹配可通过内部宽度/层数调整，报告真实参数与误差范围。建议重点模型尽量接近约 ±10%，无法合理匹配时给出成本—性能曲线；不要为了凑参数向模型加入没有作用的层。

相同训练预算不等于所有模型强制同一个 LR/warmup。历史独立 MLP-A 的 `1e-5 + 4 steps warmup` 来自小样本诊断，不是 query/CNN 的已验证最优值。允许所有对手在相同数量的候选与总调参预算内选择 LR/warmup，记录收敛曲线；尤其不能用未对齐好的随机深层 query 得出该机制无效的结论。RMR 自身也对 Perceiver 使用更长的对齐 warmup，这进一步说明训练配方是需要控制的变量。[RMR §3.3](https://arxiv.org/html/2608.08713v1)

坐标、mask 和 token 长度控制并不是额外的实验装饰：它们决定能否把结果归因到“聚合方式”。

### 9.2 独立训练对照分别回答什么（区别于当前主线的短 warm-up）

| 阶段 | 可训练部分 | 可回答的问题 |
| --- | --- | --- |
| A | Projector；encoder 与 LLM 冻结 | 在相同已有视觉特征上，哪种接口提取/表达信息更有效？这是核心归因实验 |
| B | Projector + vision encoder LoRA/部分层/全量；LLM 仍冻结 | Encoder 与接口共同适配后，方法是否仍有优势？这是联合训练效果 |

若另做独立 A→B 对照，B 从各方法自己的 A checkpoint 出发并统一 encoder 解冻策略。当前确认的连续主线则直接从本 run 的 projector warm-up 结束状态进入联合训练，并保留其 Adam 动量，不回退到 A-best。各方法比较时统一采用哪种协议与总预算；不要用一种方法全量微调 encoder、另一种方法仅 LoRA 后直接排名。

冻结 LLM 参数不等于把整个 LLM forward 放进 `torch.no_grad()`：生成损失仍需通过 decoder 的输入计算图传回 projector。阶段 A 可在固定预处理下缓存 encoder 特征；阶段 B 更新 encoder 后必须重新计算特征。

### 9.3 指标要覆盖“有没有、在哪里、属于谁”

与 [已选评价方案](EVALUATION_METRICS_AND_NEUROQA_PLAN.md) 对齐：BLEU-4、ROUGE-L 保留，GREEN、RadGraph-XL、RaTEScore 为待接入的临床文本指标；另外建立经抽样审核的脑 MRI 事实评估：

- 病变/异常存在与否、否定和不确定状态。
- 左右侧、脑区、病灶数量、大小/范围。
- 单序列可见的信号属性；多序列任务再评价跨序列对应。
- 病灶—位置—属性的绑定，尤其是多个病灶或多个区域同时异常的病例。
- 按数据来源、序列、小病灶/大病灶和单发/多发分组。

不能把胸 CT 的标签集合或胸片训练的自动评分器未经验证直接当作脑 MRI 临床正确性金标准。报告未提及不是可靠阴性；数值和病灶属性也只对有可靠标签且影像可观察的子集评估。来自 mask 的连通分量不总等价于临床病灶实例，需抽样核查。

### 9.4 验证“encoder 已有信息，但 projector 利用不足”

建议建立三段证据：

```text
同一 encoder 的原始 tokens
  ├─ 属性 probe → 信息是否可被读取？
  └─ projector 的压缩 tokens
       ├─ 同类属性 probe → 压缩后是否仍可读取？
       └─ frozen LLM 生成 → 可读取的信息是否被正确表达？
```

Probe 保持同一任务、split 和读出模型设计，并检查不同读出容量的结果；仅一个线性 probe 失败不能证明信息不存在。原始 tokens 与压缩 tokens 长度/维度不同，应避免用规模相差很大的 flatten 分类器制造不公平比较。

增加同分布错配影像、区域特征交换、坐标打乱等对照，检查模型是否真正依赖对应视觉证据。Blank image 可以作为补充，但分布变化很大，不宜作为唯一依据。

ORCA 与 Cheap Probes 已使“属性在压缩中如何变化”成为直接相关问题，因此本项目需要在 Brain MRI 的具体可观测事实、实例绑定或读取失败上给出更精确证据，不能仅以“我们也测压缩后的属性”作为新颖性。[ORCA](https://arxiv.org/html/2608.00345v1)、[Cheap Probes](https://arxiv.org/abs/2607.22771)

### 9.5 增加低成本但有解释力的控制

| 对照 | 回答的问题 | 需要避免的误解 |
| --- | --- | --- |
| 同序列错配图像，进一步做同 cohort + 同序列错配 | 模型是否使用患者特异视觉证据，而非序列/队列模板？ | 现有 shuffle 仅按 sequence 分组，且不显式排除同患者不同检查；强化控制需修改采样并检查 donor patient 不同 |
| **训练过的无图像 soft-prefix** | 相同冻结 Llama 仅学报告文风和序列先验能做到什么程度？ | 推理时移除已训练模型的图像是分布外干预，不等于这个训练基线。可学习共享或按序列的 M 个常量 latent，参数/训练成本另报 |
| 分层网格采样或随机选择 M tokens + 同 MLP | 复杂选择/聚合是否胜过简单保留部分 token？ | 选择发生在完整 encoder 之后，不能声称节省了 encoder 编码成本；固定种子并保留坐标 |
| B1-Flat1D vs B1-Grid3D | 显式三维邻域是否优于 flattened 序列分段平均？ | 前者对应 Brain3D 的压缩思路，不能默认为保留真正三维局部性 |
| 局部/全局各占一部分的简单混合 | 若新方法同时用局部细节和全局语义，优势是否仅来自两路信息？ | 例如 `M/2 AvgPool + M/2 global query`，保持总 M；只在新方法使用双路时加入，不必现在常驻主表 |

对当前病理富集数据，ADC 几乎直接指向 ISLES 来源，序列与疾病队列存在关联。组内错配比仅 blank image 更能检验患者特异内容。可由正例队列评价位置/属性，不宜据此宣称一般人群中的正常/异常筛查能力；阴性/正常泛化需要额外合适数据。

### 9.6 历史混合数据预算分析：修正“一律 M=256”的隐含假设

本节保留早期固定 M 与多来源输入的分析；当前只训练 BraTS，并采用第 15 节的目标倍率协议，不再要求全部方法 M 完全相同。

本轮只从 train/val 的预处理清单统计：train 1,960 个、val 268 个 BraTS 序列为 `39×15×15=8,775`；另有 `18/19×7×7`、`7×8×8` 等网格。M=256 对 N=8,775 是约 **34.3 倍**压缩，对 N=448 仅约 **1.75 倍**。因此应同时报告 `N、M、N/M` 的来源/序列分布，不能称所有病例都处于同一压缩率。

对 B1/B3/B6 的规则网格：给定 M 后按事先冻结的因子分解与物理长宽比规则选 `(m_d,m_h,m_w)`，满足乘积 M、各轴不超过真实输入网格，再做不重叠分桶；规则只依赖输入几何。不能一律用 `4×8×8`，因为某些输入的 H/W 只有 7。若采用框架 adaptive pooling，其分桶可能重叠，须明确记录该实现，并给相关对手同样分桶；不能将两种池化混称。

对 B2 的固定组大小 `r_d×r_h×r_w`，边缘补空槽时自然长度为：

```text
M_i = ceil(D_i/r_d) × ceil(H_i/r_h) × ceil(W_i/r_w)
```

例如 `2×2×2` 重排将 `39×15×15` 变为 `20×8×8=1,280`，将 `7×8×8` 变为 `4×4×4=64`，两者都不是 256。建议：

- 固定 M 主表先比较 **B1/B3/B4/B5/B6/B7**；B0 单列 M=N。
- B2 在自然预算的成本—性能表保留，报告每病例 M 分布；此表不作“相同 M”的因果结论。
- 若必须直接归因 B2，增加同一预定义 `M_i`/输出网格下的对照。query 方法的可变 query 数如何构造也须提前定义，并同样训练；不能临时截断已训练的 fixed-M 输出。
- 固定 M 的 Concat 变体若增加插值、再池化或变长分桶，明确列出这些操作并另命名；不可将其性能全归因于原始空间重排。

M=512 暂不作为全来源“压缩”主预算，因为已有 N=448 的样本。需要更高预算时，可以预先限定合适的输入子集，或明确称 token expansion；不能隐蔽复制 token。所有几何规则在 train/val 确定，test 只验证是否满足冻结契约。

## 10. 实验执行顺序（按当前实现修订）

以下是建议的执行顺序，不代表已启动新的训练或已选出性能最强方法。

| 优先级 | 内容 | 进入下一步的条件 |
| --- | --- | --- |
| P0 | 先定位 Concat 32× 的 CE 上冲/特殊 token 循环；复核 Flamingo 8× 的重复标点 | 用固定 train 小集过拟合与固定 val 生成诊断，区分优化问题和接口错误；第 16.5 节给出步骤 |
| P1 | 7 类各一个：MLP、AvgPool3D、Concat3D、ConvPool3D、Perceiver-NV、AdaptiveMerge3D、TokenSelect3D | 单卡短训练都已有证据；压缩方法优先 8×，32× 在生成退化排查后进入正式比较；MLP 只跑 1× |
| P2 | 重点机制加近邻强对照：LocalPacker3D；需要时再加 Mixer3D、AttentionPool、Flamingo-style | 若创新方法是局部注意力，LocalPacker 不应因同属 query 而省略；同类对照数量由创新点决定 |
| P3 | 对关键方法补齐 4/8/16/32× 曲线 | 保持预处理、划分、监督和训练协议一致，报告各方法实际 N/M、参数量、显存和时间；不强制共同 M |
| P4 | 依据失败分析增加多层、几何、语义等增强 | 每次只增加一个变量；新增输入或监督必须提供相同信息的对手 |
| P5 | 另行安排充分训练的 projector-only、视觉全量微调和 3DINO 对照 | 连续训练的短 projector warm-up 不当作已收敛 projector-only baseline |
| P6 | 关键比较至少三种子、固定 test、患者级 paired bootstrap | 所有选点由 train/val 完成，保留生成失败率和分层结果 |

如果沿用“选择/合并属于同一大类”的六类口径，可以先保留 AdaptiveMerge，TokenSelect 做后续诊断，即 **MLP + 5 个压缩方法，共 6 个**。此前 MLP/AvgPool/Concat/LocalPacker/Perceiver-NV 的五方法组合没有覆盖学习空间混合与离散自适应路线，不能称“每类都有”。

若每个压缩方法都跑 8×、32×，七方法方案为 `1 + 6×2 = 13` 个独立 run；六方法方案为 11 个 run，均不含新方法和多随机种子重复。这是建议，未更改现有 11 个脚本及训练参数。

## 11. 对后续新架构的定位建议

现有文献已分别覆盖局部细节注入、多层特征、解剖引导合并、层级图、概念查询、句子级 query 多样性、病灶实例路由、显式事实文本和问题条件化 MoE。因此“把这些组件相加”本身不足以形成清楚的贡献。尤其应避免将 brat 的语义分工、NeuroFusion 的实例/属性暴露、PerFact 的事实接口包装成尚无先例的 Brain MRI 思路。

更合适的研究问题是：

> 在固定视觉 token 预算和冻结 LLM 的条件下，怎样保留脑 MRI 中可由 encoder 读出的局部临床证据，以及病灶—位置—属性之间的对应关系？

先通过上述对照确定主要瓶颈：

| 观察到的失败 | 优先研究方向 | 必须包含的强对照 |
| --- | --- | --- |
| 小病灶证据没有被覆盖 | 局部覆盖与自适应容量分配 | LocalPacker、AdaptiveMerge、多层版本 |
| 多个病灶的属性被混合 | 实例分离、竞争式分组、绑定监督 | 有坐标且可交互的 query、brat-inspired 多样性、NeuroFusion-inspired 实例区域；若用 mask 则加同 mask 对照 |
| 位置差，加入坐标即恢复 | 显式几何暴露与读取 | 所有对手同样获得位置编码，避免误称内容信息恢复 |
| 压缩 tokens 可读，报告却写错 | 视觉—语言对齐、压缩后监督、事实显式暴露 | 相同辅助监督下的普通 query/MLP、同特征属性头产生的 predicted-fact 接口 |
| 原始 encoder tokens 就不能可靠读出目标 | Encoder 或输入分辨率/可观测性问题 | 先验证 encoder/任务，不能要求 projector 恢复不存在的信息 |

最终建议：**保留六类机制，优先实现五项 P1 核心比较，B2 单列自然预算，再补 B7 与针对性的 query 对照；扩展项由失败证据决定。** 第一篇实验结论应限定在本地患者互斥、单序列 Findings、冻结 LLM 的条件下；“迁移到 Brain MRI”本身尚不足以构成方法贡献，需要具体证明哪类证据/绑定失败得到改善。

## 12. 本轮修改范围与检索边界

本轮阅读了两份主文档、项目 README、评价方案、实际适配器/训练配置、官方 NeuroVFM connector 源码以及 train/val 几何清单；新增工作优先核对原论文、正式论文页面和作者仓库。重点核验了当前接口、8 个候选的定义、Brain MRI 直接相关方法及可归因比较；并未重新逐条审计初稿所有 CT 文献的全部实验表，也不声称覆盖全部 2026 年论文。

只修改本 baseline 规划并保存本地核对记录，未实现新增模型或改变运行中的训练、数据标注、split、权重和评价配置。修订前文档见 [备份](backups/PROJECTOR_ROUTES_AND_BASELINES.20261007T092300Z.md)。文献事实、适配建议和待实现项已分别标明；未以文档核对替代训练/临床验证。

## 13. 模块化实现与分阶段验收（2026-10-07）

用户已确认开始实现截图中的六类机制及验收方案。本节覆盖前文仅要求精确 M 的历史建议：规则网格优先精确匹配目标预算，无法满足时允许选乘积最接近的整数网格；始终记录 requested M 与 actual M。相同距离时优先不超过预算，再按物理单元形状均衡程度和固定字典序决胜；同时提供严格上限策略供后续实验选择。所有压缩方法拒绝 M>N，避免把扩展冒充压缩。

`C` 是 encoder 每个 token 的特征维度（当前 768），`N` 是输入 token 数，`M` 是输出 token 数，`D_llm` 是 decoder 输入维度（当前 4096）。完整 projector 包含聚合/重排/选择与维度映射；公共两层 MLP 是其中的映射部分，不是在完整输出后重复加一个 projector。当前受控基线统一使用同类映射头，区别于各原论文完整复现；若结构已经输出 D_llm，不应再重复映射。

实现名称：`mlp2x_gelu`、`avgpool3d`、`concat3d`、`convpool3d`、`mixer3d`、`local_packer3d`、`attention_pool`、`perceiver_nv`、`perceiver_flamingo`、`adaptive_merge3d`、`token_select3d`。公共入口仍为 `projector-train --set projector.name=<name>`；专属配置落在 `projector.params`，解析默认值后保存全部生效参数。MLP 不改变现有权重键和数值路径。

验收依次进行：①现有回归检查；②几何/预算/配置；③所有 projector 的数值、覆盖、梯度和状态恢复；④训练/生成/阶段与 checkpoint 集成；⑤本地 NeuroVFM + Llama 真实 A/B-LoRA/B-Full 冒烟及资源测量。每阶段通过后才继续；状态和证据写入 `runs/archive/engineering/projector_implementation_20261007/`。正式长训练与质量比较不属于本次模块验收。

运行中的 GPU3 训练保持原状；验收使用 GPU1/2，记录实际设备及无法满足的资源条件。待用户判断的设计选项和验收结果见 13.5–13.6 节。

### 13.1 张量契约与映射头

```text
encoder                            完整 projector                                 LLM
X: [N, C=768] → 聚合/重排/选择 → Z: [M_actual, C_inner] → 映射头 → [M_actual, D_llm=4096]
                  ↑ coords [N,3] + grid_dhw/物理尺寸
```

- `C` 来自 `encoder.feature_dim`，`D_llm` 来自 `decoder.hidden_size`，不是写死在各方法内部。
- `M` 控制序列长度，`C_inner` 控制每个 token 的通道数，二者独立。查询/混合模块默认 `C_inner=C`，可通过 `inner_dim` 改变；查询宽度须能被 `num_heads` 整除。
- 默认映射头为 `Linear(C_inner,4096) → GELU → Linear(4096,D_llm)`，4096 是可配置的 `projector.hidden_size`。Concat 的输入为 `r_d*r_h*r_w*C`。该头是完整 projector 的最后一部分。
- 新方法支持 `projector.params.head_type=linear`，变为 `Linear(C_inner,D_llm)`，便于后续头容量消融。旧 `mlp2x_gelu` 的数值路径、参数键、空 `params` 和已有 checkpoint 保持兼容。
- 这是统一 encoder/LLM/训练目标下的**机制代表实现**；Conv、Mixer、LocalPacker、Flamingo-style、AdaptiveMerge 不宣称原论文全系统复现。`perceiver_nv` 的单块与本地 NeuroVFM 官方实现做了相同权重下的输出、梯度一致性检查。

### 13.2 当前 11 个实现、文献对应关系与预算（2026-10-12 更新）

下表描述本地代码的实际机制；均只读取同一 Encoder 最后层特征，不携带原论文的完整数据、Encoder、LLM 或额外监督流程。`映射头` 默认指两层 GELU MLP。目标倍率模式优先于历史手动 `num_tokens=256` / `group_size=[2,2,2]` 的默认值。

| 类别 | CLI 名称 | 当前实现 | 参考与复现边界 |
| --- | --- | --- | --- |
| 逐 token 映射 | `mlp2x_gelu` | 每个 token 独立 Linear(768,4096) → GELU → Linear(4096,4096)；M=N | [LLaVA-1.5](https://arxiv.org/abs/2310.03744) 的核心 projector 结构；不压缩参考 |
| 固定空间池化 | `avgpool3d` | 三维不重叠空间分桶均值 → 映射头；按几何选择输出网格 | [M3D](https://arxiv.org/abs/2404.00578) 的空间池化路线；不是其固定 kernel/stride 配置原样复现，也不是 PyTorch AdaptiveAvgPool3d 的重叠分桶 |
| 空间到通道重排 | `concat3d` | 固定块按 D/H/W 槽位拼到通道 → 映射头；边界补零槽；倍率 4/8/16/32 分别用 `[4,1,1]` / `[8,1,1]` / `[6,1,3]` / `[6,2,3]` | [Argus](https://aclanthology.org/2025.findings-acl.845/)、[RMR](https://arxiv.org/abs/2608.08713) 路线；保留自然输出预算，不统一为 2×2×2，不额外池化凑 M |
| 可学习空间混合 | `convpool3d` | 2 个残差块：depthwise Conv3D(3³) + LayerNorm/channel MLP；随后分桶均值和映射头 | [Honeybee](https://arxiv.org/abs/2312.06742)、[AutoRG-Brain](https://arxiv.org/html/2407.16684v3) 启发的轻量卷积机制对照；无完整 C-Abstractor、区域引导或分割训练 |
| 可学习空间混合 | `mixer3d` | 2 个局部 2×2×2 window Mixer；token/channel MLP；第二层 padding-offset 窗口；分桶均值 → 映射头 | [Med3DVLM](https://arxiv.org/abs/2503.20047) 启发的局部 Mixer 对照；不含完整双流、多层多尺度输入 |
| 网格约束的局部查询 | `local_packer3d` | 每桶 feature 均值初始化 1 个 query；1 层桶内 cross-attention + 残差/FFN；末端 LN → 映射头 | [TokenPacker](https://arxiv.org/abs/2407.02392)、[HSENet](https://arxiv.org/abs/2506.09634) 路线；没有其多层/双 Encoder 输入，不等同于完整原方法 |
| 全局单轮查询 | `attention_pool` | M 个 learned queries；1 层全局 cross-attention + 残差/FFN；末端 LN → 映射头 | [RMR](https://arxiv.org/abs/2608.08713) 的 attention pooling 路线；不是仅一次无后处理的加权平均 |
| 全局多轮查询 | `perceiver_nv` | M 个 learned queries；6 层 cross-attention + 残差/FFN，KV 仅来自 X；末端 LN → 映射头 | [NeuroVFM](https://www.nature.com/articles/s41591-026-04497-1) connector 路线；核心单块已做官方输出/梯度对齐，但整个 projector 在本项目重新训练，不加载官方 connector 权重 |
| 含 latent 交互的全局查询 | `perceiver_flamingo` | 同样 6 层，每层 KV 为 `[X;Q]`，query 可通过 KV 读取其他 latent；末端 LN → 映射头 | [Flamingo](https://arxiv.org/abs/2204.14198) 核心机制；不是 OpenFlamingo 全部 head 配置、时间 embedding 或 decoder 交叉注意力复现 |
| 内容自适应合并 | `adaptive_merge3d` | 六邻接 Connected-Ward 合并至 M 区域；原 features 区域均值 → 映射头；离散路由不求导，聚合保留梯度 | [ORCA](https://arxiv.org/abs/2608.00345) 启发；无解剖引导、显式质心特征，评分归一化也有差别，不是完整 ORCA |
| 内容自适应选择 | `token_select3d` | 单位化 feature 空间 greedy max-min diversity；首点固定索引 0；选中原 features 恢复空间顺序 → 映射头；可选固定种子随机诊断 | [DivPrune](https://arxiv.org/abs/2503.02175) 路线；首点及特征空间不同，不能称逐项复现，梯度只流向选中 tokens |

代码入口：[registry](../src/models/projectors/registry.py)、[common](../src/models/projectors/common.py)、[resampler](../src/models/projectors/resampler.py)、[concat](../src/models/projectors/concat.py)、[mixer](../src/models/projectors/mixer.py)、[selection](../src/models/projectors/selection.py)、[adaptive_merge](../src/models/projectors/adaptive_merge.py)。
本轮另核对 [LLaVA 官方 builder](https://github.com/haotian-liu/LLaVA/blob/main/llava/model/multimodal_projector/builder.py)、[M3D 官方空间池化](https://github.com/BAAI-DCAI/M3D/blob/main/LaMed/src/model/multimodal_projector/spatial_pooling_projector.py)、[OpenFlamingo 官方 helpers](https://github.com/mlfoundations/open_flamingo/blob/main/open_flamingo/src/helpers.py)，支持上述“保留核心机制、说明适配差别”的口径。

BraTS `N=8775`，目标档位按 4/8/16/32× 排列：

- MLP：仅 1×，M=8775。
- AvgPool/ConvPool/Mixer/LocalPacker：M=`2197/1100/550/275`。
- AttentionPool/两种 Perceiver/AdaptiveMerge/TokenSelect：M=`2194/1097/549/275`。
- Concat：M=`2250/1125/525/280`；实际倍率 `3.900/7.800/16.714/31.339`，均在目标倍率 5% 偏差内。

两种 Perceiver 的 **方法 YAML preset** 默认 `checkpoint_blocks=true`，11 个 baseline 脚本会加载相应 preset。底层 registry 的通用 query 默认仍为 false；只传 `--set projector.name=perceiver_nv` 不等于加载 preset。绕过 Bash 入口时应显式加载 `configs/projector/perceiver_nv.yaml`（Flamingo 同理）。该开关只重算激活，不改变参数、层数或 M。

Mixer 的第二层通过补边改变窗口起点，不循环平移，避免把体积相对两侧误接为邻居。LocalPacker 对补齐的桶内空槽做 attention mask；不使用 foreground、病灶标签或文本 query。各方法使用完整 encoder 的最后层特征；不改变预处理或 encoder token 数。

规则网格的 `budget_policy`：`nearest`（默认，最近整数乘积）、`at_most`（最接近且不超过目标）、`exact`（不能精确满足即报错）。各轴输出数不超过输入数；边界用 `floor(i*n/m)` 划分，每个输入恰好进入一个非空桶。几何依次按预算偏差、是否超预算、物理单元长宽高的 log 方差、字典序选择；同一输入同一预算的四个网格方法使用同一分组。

AdaptiveMerge 以 L2-normalized detached features 计算 Ward 代价 `n_a*n_b/(n_a+n_b) * ||mu_a-mu_b||²`，仅合并空间相邻区域，最终从**未归一化、未 detach 的原 features**计算均值。CPU heap 计算离散路由，不构造 N×N 距离矩阵；区域按最小原始空间索引排序。A 阶段允许分组缓存，键包含精确 feature 字节、dtype、grid、预算、评分设置及算法版本；B 阶段或有梯度的输入始终重新分组。训练梯度通过区域均值回传，不对离散合并决策求导。

`token_select3d` 默认在单位化 feature 空间做 greedy max-min 距离选择，第一个点固定为索引 0、并列优先最小索引，已选择索引显式排除；不宣称等同 DivPrune。`strategy=random` 使用实验 seed + 样本 cache key 生成独立随机排列，不随 batch 顺序或进程 RNG 消耗改变。输出恢复空间顺序；梯度只流向被选 tokens。

### 13.3 文件结构与训练接入

```text
src/models/projectors/
  base.py                 ProjectedVolume 契约、输出校验
  specs.py                默认参数与严格配置校验
  registry.py             方法注册/构造，注入 stage/seed/project_root
  geometry.py             网格预算、物理尺度、互斥分桶、局部打包
  common.py               通道映射头、query block、公共元数据
  mlp.py                  兼容原 MLP
  avgpool.py concat.py convpool.py mixer.py
  local_packer.py resampler.py adaptive_merge.py selection.py
configs/projector/        每个方法的 YAML 配置
scripts/train_projector.sh          单一方法参数切换训练
archive/non_pipeline/scripts/smoke_projector_matrix.py   真实模型验收矩阵
tests/test_projector_geometry.py    几何/预算/配置
tests/test_projectors.py            算法、数值、梯度、状态
tests/test_projector_integration.py decoder/阶段/checkpoint/生成集成
tests/test_projector_cuda.py        显式指定设备的 CUDA/BF16 路径测试
```

`MultimodalModel.project_volumes()` 保留输出 token 数和 projector 元数据；旧 `project()` 仍返回 embeddings 列表。Decoder 原有接口已按实际视觉序列长度替换 `<image>`，监督仍只覆盖答案，不需要改变 loss。优化器继续自动收集 projector 参数，B 阶段继续训练视觉 LoRA 或完整 encoder，LLM 始终冻结。

训练日志新增每步 `encoder_tokens`、`projector_tokens` 及平均值。生成记录包含 `input_patch_count`、`encoder_token_count`、`projector_token_count`、实际 `visual_token_count`、`requested_visual_tokens`、网格/分组元数据及 `total_prefix_token_count`。Text-only control 的实际视觉 token 数为 0。当前已在运行的旧 MLP 进程若延迟导入新 evaluator，采用其合法的 M=N 回退。

配置在计算 provenance 前补全默认值，保存 resolved config。Checkpoint 除原有契约外，显式核对 projector 全配置；即使权重形状一样，也拒绝错误预算、算法设置或 head。随机选择另核对从实验 seed 继承的有效选择种子。每个 B 方法必须初始化自**该方法、该配置自己的 A checkpoint**。

新方法扩展：实现 `Projector`/`StructuredProjector.forward()`，使用 `register_projector(name,factory,defaults)` 注册，并在启动前导入注册模块；默认参数进入配置校验与 provenance。不要在 trainer 内加入方法名分支。

### 13.4 启动示例

```bash
# 唯一需要改变的方法选择参数是 avgpool3d，可替换为上表名称。
# 使用空闲设备，避免与正在运行的 GPU3 实验竞争。
bash scripts/train_projector.sh avgpool3d a runs/baselines/avgpool3d_m256/a \
  --set training.device=cuda:1 \
  --set projector.params.num_tokens=256

bash scripts/train_projector.sh avgpool3d b_lora runs/baselines/avgpool3d_m256/b_lora \
  --set training.device=cuda:1 \
  --set projector.params.num_tokens=256 \
  --set training.init_checkpoint=runs/baselines/avgpool3d_m256/a/checkpoints/best.pt

# 原 CLI 同样可用；YAML 和命令行参数都保存到 resolved config。
projector-train --config configs/training/stage_a.yaml \
  --set projector.name=perceiver_flamingo \
  --set projector.params.num_tokens=128 \
  --set training.run_dir=runs/baselines/perceiver_flamingo_m128/a \
  --set training.device=cuda:1
```

上面的手动预算示例属于历史模式：`mlp2x_gelu` 和 `concat3d` 不接受 `num_tokens`；无倍率时 Concat 可手动传 `group_size`，底层默认 `[2,2,2]`。**当前倍率模式使用 `projector.target_compression_ratio`，不要同时手动传 `num_tokens/group_size`**，其解析结果见第 15 节。不要在已保存某方法完整 `params` 的 resolved config 上只改名称；从通用 stage 和对应方法 YAML 开始，避免混入另一方法的参数。

需要 query 分块重计算时设置 `projector.params.checkpoint_blocks=true`；通用 registry 默认 false，但当前两种 Perceiver 方法 YAML 已默认 true（见第 13.2、16.1 节）。严谨复现可加 `--config configs/training/deterministic.yaml`，query attention 会显式选择 SDPA math backend。代价报告须记录该设置，不能混用确定性路径和默认高性能路径。

### 13.5 已采取、待研究者决定的设计取舍

| 问题 | 本轮实现选择 | 可进一步选择的方向 |
| --- | --- | --- |
| 固定预算与最接近网格 | 用户批准的 nearest；记录目标和实际数量 | 正式严格上限实验改 at_most；必须完全相同 M 时用 exact 并预审全部输入网格 |
| 公共映射头是否遮盖聚合能力 | 两层头作为受控比较默认；保留 linear 开关 | 全部方法统一换线性头做消融；按原论文原生头做另列系统比较 |
| Concat 自然输出不固定 | 保留纯空间重排；另列自然预算表 | 要进入固定 M 表时，另命名并明示额外池化/可变分桶规则 |
| 参数量与深度 | 同 C、同头；Query 1/6 层、Conv/Mixer 2 层 | 同 M 不代表同参数/FLOPs；可再做 matched-parameter、matched-time 对照 |
| 离散自适应路由 | 六邻接 Ward、A 缓存、B 重新计算；真实 features 聚合保持梯度 | 后续可研究 GPU 近似合并、soft routing 或其他种子规则，需单独命名/版本化并重新验收 |
| 位置与额外监督 | 使用 encoder 已含位置信息的 features；几何分组使用 coords；不额外注入 query 坐标或监督 | 若新方法显式增加坐标、多层、序列条件或辅助监督，应给强对手相同信息做对照 |

上述选择已落实为可复查实现，不阻塞本轮交付。正式模型效果、最优深度、学习率和种子尚需训练实验，不由工程验收推断。

### 13.6 分阶段验收证据与实际限制

所有证据位于 [`runs/archive/engineering/projector_implementation_20261007/`](../runs/archive/engineering/projector_implementation_20261007/)（从仓库根目录查看），汇总状态以该目录 `acceptance.json` 为准。修改前代码/配置/测试/脚本/文档备份为 `before_implementation.tar.gz`；最终文件清单与差异为 `changes.json`、`implementation.patch`。

| 验收层级 | 结果及范围 | 证据 |
| --- | --- | --- |
| 原有回归 | 实施前 24 项通过 | `baseline_tests.txt` |
| 几何/配置 | 16 项通过；nearest/at_most/exact、覆盖、物理轴序、非法配置 | `geometry_tests.txt` |
| 模块算法 | 首轮模块+几何 61 项通过；最终模块专测 64 项通过，含官方 NV 块对照、独立 inner width | `projector_tests.txt`、`projector_final_tests.txt` |
| 最终 CPU 集成回归 | 122 passed；33 个 CUDA 专项在默认 CPU 命令中按设计跳过 | `final_regression_tests.txt` |
| 默认 BF16 CUDA 路径 | 33 项通过：11 方法 × 三种真实网格大小 `7×8×8`、`19×7×7`、`39×15×15`；随机特征、真实通道宽度，检查反向梯度 | `cuda_kernel_tests.txt` |
| 新增 10 方法真实 VLM | 30 个方法/阶段组合全部通过；每组合短/中位/最长三个 train 样本，90 次更新 | `real_compressed/summary.json` |
| 原 MLP 兼容 | A 的短/中位/最长输入通过；短输入下 A/B-LoRA/B-Full 全部通过；旧权重键和数值路径测试通过 | `real_mlp/summary.json`、`real_mlp_short/summary.json` |
| 完整训练入口 | Bash → 配置 → Trainer → 验证/生成 → checkpoint；LocalPacker 的 A、B-LoRA、B-Full 各 1 更新步 | `cli_a/`、`cli_b_lora/`、`cli_b_full/`；以汇总文件最终状态为准 |

真实 VLM 使用本地 NeuroVFM 和完整冻结 Llama 3.1 8B、真实 Findings 监督和默认各方法结构；固定 M 方法目标为 256，Concat 为自然预算。短输入 N=448；数据的中位数和最大值都是 N=8775，后两者来自不同病例/队列。各组合均检查：所有可训练参数有有限梯度、参数实际更新、固定 RNG 下端到端梯度逐位重现、冻结 encoder 部分不变、checkpoint 恢复含 optimizer/scheduler/RNG、生成单例/批量一致及停止/长度计数。矩阵结束还以 SHA-256 核对完整冻结 LLM 权重逐位未变。B 阶段均从该方法同配置的 A checkpoint 分支。

原 `tests/test_vision.py` 写死 GPU3，本轮不直接执行以避免干扰既有训练；真实 encoder/LoRA/full 反向由上述 GPU1 矩阵覆盖。CLI 的 1 步验收显式设置 `warmup_steps=null,warmup_ratio=0`；原 A 配置的 4 步 warmup 不能用于只有 1 更新步的任务，首次配置拒绝记录保留于 `cli_a.log`，正式训练默认值没有改变。

**资源限制必须保留：** 原不压缩 MLP 的 B-LoRA 在 N=8775 反向时于共享 GPU1 OOM。当时另一个进程占约 2.12 GiB，当前进程约 21.22 GiB，申请 248 MiB 而设备仅余约 205 MiB。该长输入 B-LoRA 项不能标记通过；同一矩阵后续 B-Full 长输入因前项停止而未执行。本轮另跑短输入 MLP 三阶段完成兼容验收，不以缩短输入替代长输入验收。没有终止其他进程、修改既有实验或改变 MLP 算法以绕过限制；未来有完整空闲卡时可直接重跑 `real_mlp` 同类命令。

以下为 GPU1 A5500、确定性模式、microbatch=1 下，三个样本实测的**峰值 allocated GiB**，包含完整 VLM 的前向、反向、优化器更新。训练成本记录不包括 checkpoint 保存/验证/生成，不代表正式训练吞吐；A 可使用已有 feature/assignment cache，不能直接与 B 的在线编码时间作算法速度归因。

| 方法 | Projector 参数量 | A | B-LoRA | B-Full |
| --- | ---: | ---: | ---: | ---: |
| AvgPool3D | 19,931,136 | 16.37 | 16.96 | 17.61 |
| Concat3D | 41,951,232 | 17.13 | 17.64 | 18.29 |
| ConvPool3D | 29,422,080 | 16.80 | 17.26 | 17.91 |
| Mixer3D | 29,382,704 | 17.03 | 17.49 | 18.14 |
| LocalPacker3D | 27,015,168 | 16.58 | 17.07 | 17.72 |
| AttentionPool | 27,211,776 | 16.63 | 17.09 | 17.74 |
| Perceiver-NV | 62,624,256 | 17.90 | 18.36 | 19.01 |
| Perceiver-Flamingo-style | 62,624,256 | 17.92 | 18.38 | 19.03 |
| AdaptiveMerge3D | 19,931,136 | 16.37 | 16.96 | 17.61 |
| TokenSelect3D | 19,931,136 | 16.37 | 16.96 | 17.61 |

验收复跑示例（新建输出目录；预先选择有足够显存的设备）：

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python -m pytest tests --ignore=tests/test_vision.py -q
PROJECTOR_TEST_DEVICE=cuda:2 OMP_NUM_THREADS=2 python -m pytest tests/test_projector_cuda.py -q
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=2 \
  python -m scripts.smoke_projector_matrix \
  --output runs/new_projector_acceptance --device cuda:1 \
  --methods avgpool3d concat3d convpool3d mixer3d local_packer3d attention_pool \
            perceiver_nv perceiver_flamingo adaptive_merge3d token_select3d
```

这次交付是可训练的模块与工程验收；没有启动这些 baseline 的正式长训练、调参或测试集质量比较，也不以 smoke CE 或两个生成 token 推断临床效果。

## 14. 历史固定 M 与方法保真审计（2026-10-07）

本节几何统计与资源事件保留为当时证据。当前主实验采用第 15 节的同目标倍率协议，以下固定 M 要求不再生效；现行机制表见第 13.2 节。

用户再次明确：主比较要固定的是每个样本最终送入 LLM 的视觉 token 数 M。**工程能训练与实验可公平比较是两个验收维度。** 本轮复查代码、完整 train/val 几何、真实缓存 features 和作者代码，发现下列需要在正式实验前处理的差异。没有改变已有聚合算子的计算定义；已增加预算预审与可选运行时强校验。第 13 节中“自然预算 Concat 单独比较”不能代替用户要求的固定 M Concat 主表实现，当前版本这部分仍存在缺口。

证据目录：[`runs/archive/engineering/projector_audit_20261007/`](../runs/archive/engineering/projector_audit_20261007/)。几何检查覆盖 2,476 个 train/val 输入；feature 诊断仅使用三个 train 样本，不读取测试集或引入病灶标注。作者代码快照与 SHA-256 记录在 `reference_sources/`，用于区分代码事实、结构适配和仍待实验证实的效果风险。

### 14.1 nearest 的精确定义及本数据结果

给定输入 `(D,H,W)`、物理范围 `(L_d,L_h,L_w)`、目标 M，选择整数 `(m_d,m_h,m_w)`，满足各轴 `1<=m_i<=输入轴长`。实现枚举 d/h，w 只需考察 `M/(d*h)` 相邻整数并限制在有效范围；按以下顺序决定最优解：

1. 最小化 `abs(m_d*m_h*m_w-M)`，因此有精确解时一定用精确解。
2. 同样偏差时优先 `actual<=M`。
3. 同样预算误差时，最小化 `log(L_i/m_i)` 的方差，使每个输出单元的物理三轴尺度较均衡。
4. 再相同时按 `(m_d,m_h,m_w)` 字典序决胜，保证稳定。

每轴采用 `[floor(i*n/m), floor((i+1)*n/m))` 分桶，覆盖所有原 token，桶不重叠；非整除时各轴桶宽相差至多 1。它不等同于可能重叠边界的框架 adaptive-average-pooling。分桶不读取影像内容或标签，但物理长宽比参与网格选择。

| 输入网格/物理范围示例 | M=64 输出网格 | M=128 | M=256 |
| --- | --- | --- | --- |
| `39×15×15`，`156×240×240 mm` | `4×4×4` | `4×4×8` | `4×8×8` |
| `19×7×7`，`152×224×224 mm` | `4×4×4` | `8×4×4` | `16×4×4` |
| `7×8×8`，约 `134.4×230×230 mm` | `4×4×4` | `4×4×8` | `4×8×8` |

**全量结果：64、128、256 对全部 2,476 个输入均可精确满足。** 因此这三档主实验可以使用 `budget_policy=exact`，无须近似。反例：M=250 时，24 个输入只能选到 252，其余 2,452 个得到 250；此时不应和精确输出 250 的 query/merge 方法混称“同 M”。记录见 `geometry_audit.json`、`all_method_budgets.json`、`exact_budget_preflight.json`。

相同 M 仍不等于相同空间分辨率：例如 M=256 对 BraTS 与部分 ISLES 会选择不同网格。规则桶的尺寸不完全一致，均值对小桶/大桶的每-token 权重也不同。固定排序还会在对称候选中选择固定轴向方案。这些是可追溯的几何选择，不是随机 bug；位置/小病灶分析应保留 `output_grid` 和桶大小。

### 14.2 历史 2×2×2 Concat 的固定 M 问题（现已改用倍率协议）

2026-10-07 审计使用的 `concat3d` 手动默认配置把每个 `2×2×2` 空间块拼成一个 `8C` 维向量。下面描述该历史配置，不代表现在四档倍率的块大小。当前无需新增固定 M 适配，采用第 15.4 节已确认的四组固定块；历史配置的输出关系为：

`M_actual = ceil(D/2)*ceil(H/2)*ceil(W/2)`。

所以 `39×15×15 -> 20×8×8=1280`，`7×8×8 -> 4×4×4=64`。全量实际输出范围 64–1280；其中 2,228 个输入输出 1280。以目标 M=256 检查时，当前所有 2,476 个输入均不满足，预审按要求返回非零状态。不能将这些结果和其他方法的 M=256 放在主表中作同预算归因。

“固定 M Concat”指最终也输出 `[M,4096]`，并保留其“局部元素按槽位拼到通道后再学习投影”的核心。由于 N 可变，只修改 group_size 不足以解决：每个输出组含 K 个输入时，第一层宽度是 `K*C`；K 按样本变就无法共用普通 Linear 的固定权重。

| 可行适配方向 | 是否固定最终 M | 信息与成本影响 |
| --- | --- | --- |
| 按共同输出网格分 M 桶，各桶原 tokens 按三轴槽位排布并补到预审得到的固定容量，再 Concat | 可固定，不需预先平均或删 token | 保留拼接前原 features，但第一层可能很大，空槽多；须冻结容量并对超容量输入报错 |
| 先把特征网格重采样到 `m_d*r_d × m_h*r_h × m_w*r_w`，再固定 r³ Concat | 可固定 | 增加插值/池化；小病灶细节可能在 Concat 前已改变，必须另命名并说明 |
| 原 Concat 后再平均到 M | 只有原输出不少于 M 时才是压缩 | 混合了重排和平均，原输出小于 M 时还需扩展；不能当作纯 Concat |

保留三轴槽位的保守上界方案也不是免费操作：按已审计 train/val 的各轴最大桶宽，M=256 需 `10×2×3=60` 槽位，使用现有 4096 hidden MLP 时共有 **205,529,088** 个映射头参数；M=64/128 的这一保守模板可到 **520,101,888**。这是该模板的估算，不是所有 Concat 设计的参数下界。应先确定适配、容量和显存方案，再实现与验收，不能通过未标明的平均/截断假造固定预算。

`mlp2x_gelu` 同样是 M=N 的不压缩参考；若强行让它输出固定 M，必须添加压缩操作，例如 pool+MLP 已属于 AvgPool baseline。正式主表要求固定 M 不意味着不压缩参考可以伪装成固定 M 方法。

### 14.3 “特征宽度”不应替代 M 成为主比较目标

| 符号 | 含义 | 当前值/来源 | 是否由同一 LLM 强制相同 |
| --- | --- | --- | --- |
| N | encoder 输出行数 | 按输入网格变化 | 否 |
| C | encoder 每个 token 的维度 | 768，同一 encoder | 否，由 encoder 决定 |
| M | 最终视觉 token 行数 | 当前由目标 4/8/16/32× 及各方法自身机制决定 | 否；当前用户允许小幅预算差别，须报告实际 N/M |
| `C_inner` | projector 内部 attention/mixing 的宽度 | 默认 768，可配置 | 否 |
| `H_head` | 映射 MLP 中间层宽度 | 默认 4096 | 否 |
| `D_llm` | projector 最终输出的通道维度 | 4096，读取 decoder 配置 | 是 |

主比较的输出契约是 **`[M,D_llm]`**，不能用“都是 4096 维”掩盖 M 不同。之前额外统一内部 C 和映射头，只是一个受控实现选择，并非同 LLM 的必然要求。允许各机制保留合理内部宽度，但必须报告参数、计算、归一化和训练配方；相同 M 不保证相同参数/FLOPs。

### 14.4 空间 Ward 合并：做了什么、遗漏了什么

实现过程：每个 token 起始为独立区域；只有共享面的六邻接区域可以合并。设区域 a/b 的 token 数为 `n_a,n_b`，评分空间的均值为 `mu_a,mu_b`，每步合并代价最小的相邻区域：

`Delta = n_a*n_b/(n_a+n_b) * ||mu_a-mu_b||²`。

合并后更新区域人数、人数加权的均值和邻接边，直至精确剩 M 区。最终输出值直接用该区原始 features 的均值，再经映射头。不是简单对两个区域均值做 50/50 平均，因而不会把大区域的每个原 token 无意降权。新增小网格穷举 oracle 已核对 heap 实现的每次最小邻接代价所产生的最终分组。

当前默认把每个输入 feature L2 归一化后计算路由代价（`normalize_scores=true`），但输出取未归一化的原 features。这样强调方向相似性，忽略 feature 范数对合并的作用。原 ORCA 的评分构造使用视觉 features 与按尺度调整的软器官覆盖向量，并追加输出质心编码；当前实现除去解剖信息之外，还改变了评分归一化并省掉显式质心编码。因此应标为 **Connected-Ward3D 机制适配**，不能称仅去除 mask 的完整 ORCA 消融。[ORCA 方法](https://arxiv.org/html/2608.00345v1)

对训练和信息的具体影响：

- **覆盖不等于无损**：所有输入恰好参与一次均值，但某个细小变化并入 K 个 token 后，直接均值分量及局部梯度会缩为原来的 `1/K`。三个 train 诊断样本在 M=256 下，最大区域分别为 8、97、91 个输入 token；没有由此测量或推断病灶信息已经丢失。
- **位置只被记录，未被读入 LLM**：`output_coords_dhw` 和 `group_sizes` 在 metadata 中；`MultimodalModel` 只把 embeddings 交给 decoder。当前也没有把区域范围编码到特征。按最小原索引排序不能代替不规则区域的精确质心/范围/大小。建议增加显式位置版本，并给相关 baseline 一致的几何输入做消融。
- **它不是可学习的路由器**：路由由 detached features 在 CPU 上硬合并；梯度能穿过原 features 的区域均值和映射头，但不能穿过“哪两区合并”的选择。
- **A/B 缓存语义不同**：A 冻结 encoder，同一样本特征固定，能复用精确 feature 字节键控的分组；B 更新 encoder，必须重算，当前代码已禁用 B 分组缓存。B 的硬分组与输出顺序可能随 features 变化，可能增加优化的不连续性。
- **GPU 近似和 soft routing 是新变体**：前者可能改变合并顺序、并列处理；后者允许一个输入参与多个软区域并对路由求导。二者都不能只作为“加速”替换后仍沿用原方法名称及 checkpoint。

### 14.5 额外位置、多层和辅助监督的含义

当前 NeuroVFM 已在编码时注入三维位置，不能把当前 baseline 描述为“没有位置”。区别在于 projector 是否额外读取/保留显式几何：网格方法用 coords/物理范围分桶；global learned queries 没有预先指定的三维位置；Ward 的输出质心目前只在日志中。

显式加入 query/输出质心坐标可以表现为 `feature + PE(center)`，或者把 `PE(center)` 拼入 channel 后再映射到 D_llm；这不需要增加 token 数 M。图像自带的坐标、spacing、可计算质心也不是新的病灶标签。应该保证对手有同样的坐标访问权限，并比较位置编码是否贡献收益，而不是为了“公平”一概禁止坐标。若使用物理坐标，需区分网格索引、归一化相对坐标与病人毫米坐标，不能混用。

多层输入指读取同一个 encoder 的中间层，例如 `X_3,X_6,X_9,X_12`，并进行对齐/融合；可以仍输出 M 个 token，但比只读 X_12 更容易直接访问较早的空间细节。它不等于多序列或多幅图像，也不必增加 M。若新方法使用多层，就需要同输入的多层 baseline，不能仅据此归因新的压缩算法。

辅助监督包括分割/病灶属性 loss，或用训练报告构造对比对齐、句子匹配、query 多样性目标。即便使用同一份训练报告而没有新增人工标注，优化目标也已改变。当前实现只有报告答案 CE；若新增监督，需要相同信号下的强 baseline，且 val/test 目标不能进入推理 query 或训练标签构造。上述限制旨在分清收益来源，不是要求所有未来方法永远只读一层或禁止辅助 loss。

### 14.6 原方法保真度逐项核对

| 当前实现 | 已确认差异 | 实验解释与建议 |
| --- | --- | --- |
| AvgPool3D | M3D 作者代码用固定 kernel/stride 池化；当前按目标 M 做不重叠整数分桶，桶大小可变 | 合理的变长 3D 适配；应写 GridPool3D 机制 baseline，不称 M3D 全系统复现 |
| LocalPacker3D | 原 TokenPacker 用插值 point query、多层 K/V、独立 K/V MLP、MHA 后映射；当前用同最后层桶均值初始化 Q，并套用带残差/FFN/final-LN 的 QueryBlock | 保留局部查询机制，但差异不止“去掉多层”。若要代表原 TokenPacker，需要单独对齐 query 初始化、K/V 路径及 block 结构 |
| ConvPool3D | Honeybee C-Abstractor 使用 RegStage→pool→RegStage→readout；当前是轻量 depthwise Conv3D + channel MLP，仅在 pool 前混合 | 后池化空间交互缺失，卷积与通道交互次序/容量不同；应定位为轻量局部卷积对照，不能据此评价完整 Honeybee/AutoRG |
| Mixer3D | Med3DVLM 作者代码用双流多尺度输入，在 token 轴用 learned MLP 改变长度，同时逐层提升 channel；当前仅两个局部窗口块，最后用固定均值降采样 | 这是 WindowMixer+GridPool，学习到的额外交互范围和下采样机制有实质不同；不能把它作为原全局 token-mixing learned reduction 的唯一代表 |
| Perceiver-NV | 层内 Q/K/V/FFN 与本地官方块一致；默认层数 6；从头训练，M 改为实验预算 | 是当前最接近原 connector 核心的实现；单块输出/梯度一致性已测，但不等于加载官方预训练 connector 或复现其完整训练数据 |
| Perceiver-Flamingo-style | 保留 KV=[X;Q]，但采用 NV 的 query 初始化 `std=C^-0.5` 和 `head_dim=C/heads=96`；OpenFlamingo 默认 latent std=1、head_dim=64 | 适合与 NV 做 latent-KV 交互对照，不能标作 OpenFlamingo 原配置复现。初始化与 attention 内宽可影响优化 |
| AdaptiveMerge3D | 无解剖引导、默认 unit-norm 评分、无显式输出质心特征；区域大小/范围也未送入 LLM | 是空间 Ward 机制 baseline，位置保留必须单独检查 |
| TokenSelect3D/diversity | 当前固定索引 0 起步，在 encoder 空间评分；DivPrune 作者代码首点依赖最近邻距离，LAYER_INDEX=0 分支是在已投影的视觉 embedding 上选择 | 原算法首点与选择空间均不一致；仅可称固定首点的 greedy diversity 诊断，正式原方法比较应另实现对应变体 |

核对来源：[M3D 作者代码](https://github.com/BAAI-DCAI/M3D/blob/main/LaMed/src/model/multimodal_projector/spatial_pooling_projector.py)、[TokenPacker 作者代码](https://github.com/CircleRadon/TokenPacker/blob/main/llava/model/multimodal_projector/builder.py)、[Honeybee 作者代码](https://github.com/khanrc/honeybee/blob/main/honeybee/projectors/projectors.py)、[Med3DVLM 作者代码](https://github.com/mirthAI/Med3DVLM/blob/main/src/model/projector/mlp.py)、[OpenFlamingo 作者代码](https://github.com/mlfoundations/open_flamingo/blob/main/open_flamingo/src/helpers.py)、[DivPrune 作者代码](https://github.com/vbdi/divprune/blob/main/LLaVA/llava/model/llava_arch.py)。

固定索引 0 的问题不是说这个 encoder token 必然无信息——它经过了全局 encoder——而是它没有任何代表性选择依据。三个诊断输入的第 0 个原始 patch 都近乎常量，因此该默认规则尤其不应被误称为内容驱动首点策略。

### 14.7 真实特征诊断：输出尺度与 query 趋同风险

本轮对三个真实 train 缓存 features 分别检查随机初始化，以及上轮仅训练三步的 A 冒烟权重。以下是 BraTS-GLI 诊断例、同一 seed 随机初始化、M=256 下的结果。RMS 是最终送入 LLM 的视觉 embedding 均方根；cos 是不同输出 tokens 的平均两两 cosine。

| 方法 | 输出 RMS | 平均 token cosine |
| --- | ---: | ---: |
| AvgPool3D | 0.0140 | 0.8193 |
| ConvPool3D | 0.0521 | 0.9449 |
| Mixer3D | 0.0519 | 0.6443 |
| LocalPacker3D | 0.1978 | 0.5904 |
| AttentionPool | 0.1999 | 0.9641 |
| Perceiver-NV | 0.2024 | 0.9882 |
| Perceiver-Flamingo-style | 0.2025 | 0.9873 |
| AdaptiveMerge3D | 0.0142 | 0.7617 |

这一例 encoder features RMS 约 0.0628；另外检查了 Llama embedding 表前 8192 行，RMS 约 0.0097（不是实际 prompt 的加权 token 分布）。查询分支有 final LayerNorm，而池化分支没有；卷积/mixer 的 norm/残差也改变尺度。因此“共同映射头”并没有保证相同输入分布或相同对齐难度。

这些数字是**诊断信号，不是性能结论**：高 cosine 不等于严格秩塌缩，也不能证明输出没有患者信息；RMS 不同不能直接推断 CE 必然更差。三步 smoke 也不能证明已经收敛。但它们足以否定“shape/梯度正确就可以用同一 MLP 配方比较最终能力”的假设。建议在小规模 train/val 上检查输出/梯度范数、query 跨 token 方差/有效秩、attention 集中度、CE 与视觉错配敏感性，再决定初始化/输出缩放消融与各方法的 LR/warmup。

历史独立 A 的 `1e-5 + 4 steps warmup` 来自原 MLP 的先前诊断，不是所有 20M–63M 参数的新 projector 的已验证最优配方。公平调参应给各方法相同候选次数/总调参预算，而不要求所有结构强制同一最优 LR。不能直接给一方新增监督或特殊训练后，将收益全部算作压缩结构优势。

### 14.8 本轮实际改动与正式实验前的门槛

已新增：

- `archive/non_pipeline/scripts/audit_projector_budgets.py`：读取既有预处理 metadata，按真实网格预审目标/实际 M；`--require-fixed` 遇到不匹配返回非零状态。默认遍历所有方法会如实报出 MLP/Concat 不匹配；固定预算组需显式列出方法。
- `training.required_visual_tokens`：默认 null 保持历史行为；设置正整数后，在真实 `project_volumes()` 输出处强制核对实际 M，训练和有视觉输入的生成都会检查，不能仅靠方法声称的配置通过。Text-only control 仍按原设计输出零视觉 tokens。
- 新增测试覆盖自然 Concat 64≠256、nearest 252≠250 的真实 projector 输出拒绝、旧 checkpoint 配置无新字段的兼容性、预算预审与 Ward 穷举 oracle。

```bash
# 固定 M=256 主表配置示例；其他方法只需按其支持的参数配置。
bash scripts/train_projector.sh avgpool3d a runs/baselines/avgpool3d_exact256/a \
  --set training.device=cuda:0 \
  --set projector.params.num_tokens=256 \
  --set projector.params.budget_policy=exact \
  --set training.required_visual_tokens=256

# Query/merge/selection 原本精确输出 M，无 budget_policy 参数；仍启用实际 M 检查。
bash scripts/train_projector.sh perceiver_nv a runs/baselines/perceiver_nv_exact256/a \
  --set training.device=cuda:0 \
  --set projector.params.num_tokens=256 \
  --set training.required_visual_tokens=256

python -m scripts.audit_projector_budgets \
  --methods avgpool3d convpool3d mixer3d local_packer3d attention_pool \
            perceiver_nv perceiver_flamingo adaptive_merge3d token_select3d \
  --budgets 64 128 256 --policy exact --require-fixed \
  --output runs/new_fixed_budget_preflight.json
```

本轮回归 **128 passed，33 个 opt-in CUDA 测试默认跳过**；另未纳入硬编码 GPU3 的 `tests/test_vision.py`。没有改变 projector 算法内核，未重复全部新 projector 的 GPU 矩阵。真实 feature 诊断在 GPU3 进行，诊断本身未使用 LLM 生成或进行正式训练。

原不压缩 MLP 已在空闲 GPU3 完成补验：**A / B-LoRA / B-Full × 短 / 中位 / 最长输入，共 9 次真实更新全部通过**，包含 N=8775 的长输入；梯度重放、checkpoint 恢复、生成单例/批量一致性与完整冻结 LLM 哈希检查通过。长输入 B-Full 峰值 allocated 约 21.67 GiB、reserved 约 21.94 GiB，仍应为正式训练的批量和验证预留显存。补验结果见 `real_mlp_complete/summary.json`；先前共享 GPU1 的 OOM 证据保留，资源条件改变后的通过不改写旧报告。审计汇总见 `runs/archive/engineering/projector_audit_20261007/summary.json`。

正式比较前必须明确：①固定 M Concat 的适配与资源成本；②哪些行是轻量机制代表、哪些要提高到原方法结构保真度；③是否启用共同的显式几何输入/合并质心版本；④输出尺度、query 分化和各方法收敛的诊断方案。后续任何会改变路由、初始化或 forward 语义的改动都应固定配置/实现版本并重新验收，不能仅因 state_dict 形状相同就复用旧实验名称。当前 checkpoint 保存 source hashes，但 contract_id 排除了 source_files；因此代码语义升级还需显式版本管理，不能只依赖现有契约哈希。

## 15. 当前实施协议：BraTS-only、各方法保留原机制的近似整数倍压缩（2026-10-12）

**本节是当前优先约定，覆盖前文及本节旧版关于“所有方法必须使用共同 M”“Concat 必须适配公共输出网格”的要求。** 用户已确认：当前只训练 BraTS，目标压缩倍数按 N/M 定义，使用 4×、8×、16×、32×；各基线保留自身机制，实际输出数量可以略有不同。Concat 采用经过确认的固定块，实际倍数相对目标偏差不超过 5%。保留 1× 不压缩 MLP。

ISLES、WMH 暂不纳入本轮主实验。比较名称为“同目标压缩档位”，不再宣称“严格同 M”。同一张结果表必须报告各方法实际 N、M、N/M、参数量、速度和显存。

**实施状态：已增加 `projector.target_compression_ratio`、方法专属预算解析、建模前 MRI 文件头预检、运行时几何及输出校验、配置/检查点恢复和预算预览入口。** 核心 Projector 聚合方式保持原样。测试证据与本次源文件备份位于 `runs/archive/engineering/compression_ratios_20261012T153951Z/`，其中 `brats_preflight.json` 为全量 train/val 几何预检，`smoke_full_width.json` 为四档真实 Encoder 特征 smoke 结果，`pytest.log` 为回归测试结果。最终是否通过以各报告的状态字段和日志为准。

### 15.1 数据范围与输入保持

- 数据：BraTS GLI、MEN、MET，使用当前修订的单序列 Findings 和患者互斥划分。后续清单仅筛选数据来源，不重新随机划分患者；val/test 也限于对应的 BraTS 子集。
- 输入：继续使用 `native_grid + minimal_pad`，保留全部空间 patch，`token_filter_policy=none`。不为凑压缩整数倍额外裁剪图像、插值或扩大 Encoder 输入。
- 模型：NeuroVFM 最后层 patch features → Projector → 冻结的 Llama-3.1-8B-Instruct。当前通道接口为 `[N,768] → [M,4096]`；压缩倍数只描述行数变化。
- 训练策略：压缩预算与训练日程独立，当前入口兼容 `joint_lora` 连续训练及 `a/b_lora/b_full` 分阶段训练；不因切换压缩率改变视觉适配方式或解冻 LLM。同一比较表保持训练日程、数据和调参机会一致。
- 对照：正式 BraTS-only 结果需要相同训练范围的不压缩 MLP。已有混合数据训练 checkpoint 只作为明确标注的额外参照，不能直接代替同数据范围的基线。

本次读取现有 train/val 预处理清单，对全部 BraTS 记录重新核算，而非仅抽查单例：

| 范围 | 序列样本数 | 文件网格 XYZ | 文件 spacing XYZ | 预处理数组 DHW | Encoder patch 网格 |
| --- | ---: | --- | --- | --- | --- |
| train | 1960 | 240×240×155 | 1×1×1 mm | 156×240×240 | 39×15×15 |
| val | 268 | 240×240×155 | 1×1×1 mm | 156×240×240 | 39×15×15 |

全部上述记录的 `resampling_applied=false`。证据来自 [train 清单](../runs/full_native_20261006/preprocessing/train/preprocessing_manifest.jsonl) 和 [val 清单](../runs/full_native_20261006/preprocessing/val/preprocessing_manifest.jsonl)。这些是序列样本数，不是患者数；本次没有读取测试文本或成绩来选择预算，也不将 train/val 几何审计表述为已经验证全部 test 输入。

### 15.2 为什么输入是 39×15×15，而不是立方体或 2 的幂

本地文件 `BraTS-GLI-00291-000-t1c.nii.gz` 的 NIfTI 头直接给出尺寸 `240×240×155`、spacing `1×1×1`。其预处理记录显示，方向统一后尺寸不变，没有空间重采样，XYZ 的 `pad_after=[0,0,1]`。

```text
下载的 NIfTI 文件：XYZ = 240×240×155，spacing = 1×1×1 mm
转成网络数组：    DHW = 155×240×240
最小边界 padding：DHW = 156×240×240（深度补一层背景）
每个 patch：     DHW = 4×16×16 个体素
patch 网格：     (156/4)×(240/16)×(240/16) = 39×15×15
patch tokens：   N = 39×15×15 = 8775
```

因此，39×15×15 是 **patch 网格**，不是 MRI 的体素尺寸。它由本地 BraTS 文件的尺寸、NeuroVFM 的非立方 patch 和最小 padding 共同决定。patch 深度 4 表示每个 patch 包含 4 个体素，不表示本实验把切片间距改成 4 mm；这里仍为 1 mm 间距，每个 patch 的物理尺寸约为 4×16×16 mm。

这里的“文件原网格”指下载到本地的 BraTS 发布文件网格，不应理解为保证未经数据集发布方预处理的扫描仪原始数据。补的背景层不生成新的解剖信息；在当前实现中，补零发生在 min-max 缩放后、均值/标准差标准化前，不能把最终归一化张量的填充值也一概称为 0。

当前 Encoder 输出接口保留每个 patch 对应的一行特征，因此输出仍为 8775 行；该计数没有额外加入 CLS/register token。依据：[预处理与 patch 划分](../src/preprocessing/neurovfm.py)、[空间变换](../src/preprocessing/geometry.py)、[Encoder 输出检查](../src/models/encoders/neurovfm.py)。曾提到的 40×16×16 是额外扩大输入的备选示例，不是现有数据要求，也不是本轮采用的方案。

### 15.3 压缩定义与按方法解析的四档预算

```text
N = 实际送入 Projector 的 Encoder patch tokens 数
M = 实际送入 LLM 的视觉 tokens 数
实际压缩倍数 = N / M
保留比例 = M / N
减少比例 = 1 - M / N
相对倍数偏差 = abs((N / M) / r_target - 1)
```

统一接收目标倍数，但不统一各方法的实际 M 或空间分组：

1. `avgpool3d / convpool3d / mixer3d / local_packer3d`：`num_tokens=ceil(N/r)` 是请求预算，继续使用当前 `budget_policy=nearest` 和原来的不重叠整数分桶。实际 M 是输出网格的乘积。
2. `attention_pool / perceiver_nv / perceiver_flamingo / adaptive_merge3d / token_select3d`：直接输出 `ceil(N/r)`，不需要额外凑成网格。
3. `concat3d`：当前 BraTS 几何使用下节固定分块；各轴向上取整，补最少空槽后拼接，不改成 AvgPool 的可变分桶。
4. `mlp2x_gelu`：不压缩；不设置 ratio 或设置 ratio=1。若传入 4/8/16/32，明确报错。

输入 N=8775 时的验收表：

| 目标 | 初步预算 ceil(N/r) | 网格方法输出网格 DHW | 网格方法 M / N÷M | Query、Merge、Selection M / N÷M | Concat M / N÷M |
| --- | ---: | --- | --- | --- | --- |
| 4× | 2194 | 13×13×13 | 2197 / 3.994083× | 2194 / 3.999544× | 2250 / 3.900000× |
| 8× | 1097 | 10×10×11 | 1100 / 7.977273× | 1097 / 7.999088× | 1125 / 7.800000× |
| 16× | 549 | 5×10×11 | 550 / 15.954545× | 549 / 15.983607× | 525 / 16.714286× |
| 32× | 275 | 5×5×11 | 275 / 31.909091× | 275 / 31.909091× | 280 / 31.339286× |

Query/latent 数在模型初始化时确定，普通已训练 query bank 不能随意截短当成另一个已训练预算模型。四档分别训练、保存和评价。网格方法仍优先接近 token 预算，再考虑空间单元物理尺度；不将 Concat 的物理均衡规则强加给其他方法。

### 15.4 Concat 固定块与物理范围

当前输入 token 网格为 39×15×15，每个 patch 的物理范围为 4×16×16 mm。分块 `(kD,kH,kW)` 的完整内部块物理范围为 `(4kD,16kH,16kW)` mm。用户确认在实际压缩倍数相对偏差 ≤5% 的条件下，优先让物理边长均衡。

| 目标 | group_size DHW | 完整块物理范围 mm | 最长边/最短边 | 输出网格 | M | 实际倍数 | 相对偏差 |
| --- | --- | --- | ---: | --- | ---: | ---: | ---: |
| 4× | 4×1×1 | 16×16×16 | 1 | 10×15×15 | 2250 | 3.900× | 2.50% |
| 8× | 8×1×1 | 32×16×16 | 2 | 5×15×15 | 1125 | 7.800× | 2.50% |
| 16× | 6×1×3 | 24×16×48 | 3 | 7×15×5 | 525 | 16.714× | 4.46% |
| 32× | 6×2×3 | 24×32×48 | 2 | 7×8×5 | 280 | 31.339× | 2.06% |

搜索范围是“不重叠固定块、各轴最小末端补齐”的当前实现。在这个范围内，16× 满足 5% 的选择只有 6×1×3 及 H/W 交换；5×2×2 虽然更均衡，但输出 512、实际 17.139×，偏差 7.12%，不采用。32× 用 6×2×3 取代此前 7×1×5，将最长物理边从 80 mm 缩至 48 mm。

物理范围针对完整内部块，边界块的真实覆盖会更小。补齐只发生在 Projector 的特征槽位，不扩大 MRI 或 Encoder 输入，不移除真实背景 tokens。Concat 第一层输入维度为 `prod(group_size)*768`，分组容量和参数量随档位变化，必须报告，不能把分组容量直接当成 N/M。

### 15.5 配置、建模、恢复及验收

统一入口：

```yaml
projector:
  name: avgpool3d
  target_compression_ratio: 8
  params: {}
```

- 方法 YAML 只保存结构参数。未设置 ratio 时，继续通过方法默认值或显式 `num_tokens/group_size` 使用旧手动模式。
- ratio 模式中，不再手填 `num_tokens/group_size`；同时填写会报错。网格方法使用 `nearest`。
- `resolved_budget` 是系统派生并保存的契约，不是供手动调整的参数。它记录版本、方法、输入网格/物理范围、N、初步预算、预期 M、输出网格或分块、实际倍数及误差。重新加载时重算核对，拒绝修改目标倍数而保留旧解析结果。
- 参数量相关的 query bank / Concat head 必须在模型初始化前确定；训练中不根据单个样本动态重建它们。
- 当前 v1 自动几何预检支持 **NeuroVFM、native_grid、minimal_pad、不滤背景、一次运行只有一种输入几何**。检查当前选定 train/val 的 MRI 文件头；评价检查所选评价图像的几何。其他 spacing/Encoder 需要补充几何适配，不能默默套用当前预设。Concat 四档预设限定网格 39×15×15、物理范围 156×240×240 mm。
- 文件头预检不读取 MRI 强度或使用病灶标注选择预算；运行时再对实际 Encoder 输出的 N、网格、物理范围和实际 M 检查，避免预检预测与真实预处理不一致。
- `training.required_visual_tokens` 保留。ratio 模式比较它与解析后的实际预期 M，而非初步请求预算。例如 AvgPool 8× 的请求是1097、预期输出1100，Concat 是1125，Query 是1097；不跨方法强制相同 M。
- 训练记录保存目标倍数、实际倍数和平均输入/输出数量，生成结果的 `projector_metadata` 保存每例实际预算。`budget_preflight.json` 和 `resolved_config.yaml` 随训练输出保存。
- 评价及恢复从 checkpoint 的已解析配置重建相同结构，重新检查输入几何；禁止加载另一压缩档位，即便某些方法权重形状恰好一致。旧手动配置的 Projector 字典和预处理 hash 保持兼容。

先预览，不加载 Encoder/LLM 权重：

```bash
python -m src.cli.config \
  --config configs/projector/concat3d.yaml \
  --set projector.target_compression_ratio=16 \
  --resolve-budget \
  --budget-report runs/budget_preview/concat_r16.json \
  --output runs/budget_preview/concat_r16.yaml
```

训练示例（使用已有训练策略，按实际资源选择 GPU/DDP）：

```bash
bash scripts/train_projector.sh avgpool3d joint_lora runs/brats_avgpool_r8 \
  --set projector.target_compression_ratio=8
```

将方法替换为 `concat3d`，同一入口将派生 `[8,1,1]`；也支持旧 `a/b_lora/b_full` 阶段。先修改方法和目标倍数生成一个新配置，不通过覆盖旧 `resolved_config.yaml` 的 ratio 来启动新实验。

验收覆盖：四档×全部压缩基线的独立预期表、方法 YAML 解析、配置保存/恢复、旧手动配置、非法/冲突参数、输入几何变更、checkpoint 预算一致性，以及真实 NeuroVFM 特征下的前向、反向和单步更新。Smoke 采用完整 N=8775、C=768、输出4096和各方法默认结构；不等同于 Llama 端到端报告训练或性能评价。

前一版“严格共同 M”、变桶 Concat 适配和“所有局部方法改用 Concat 分块”的建议已撤销。后续新增方法应明确自己的预算解析规则，在同目标倍数下报告真实成本与质量。

128×／256× 的新增预算、分割选择和验收见 [Projector Baseline 128× / 256× 实现](PROJECTOR_R128_R256_IMPLEMENTATION.md)。本节原有四档表格保留其最初协议与验收记录。

### 15.6 压缩档位的文献依据（对前次调研的记录）

下表数字对应指定设置，倍数统一按 N/M 换算。固定输出预算不等于固定倍数；不同论文的图像尺寸、预训练规模、LLM 是否更新、输入信息路径均有差异，不能据此直接预测本实验效果。

| 文献与设置 | 输入 → 输出 tokens / 倍数 | 对本项目的用途 |
| --- | --- | --- |
| [Honeybee，CVPR 2024，附录 A](https://arxiv.org/html/2312.06742v2) | 224 输入：256→256/144/64，即 1×/1.78×/4× | 保留温和压缩及相同 token 数的质量对照 |
| [TokenPacker，2024，Table 1](https://arxiv.org/html/2407.02392v2) | 标准分辨率：576→144/64/36，即 4×/9×/16× | 局部聚合路线的多档预算依据 |
| [VisionZip，2024，§4.2](https://arxiv.org/html/2412.04467v2) | LLaVA-1.5：576→192/128/64，即 3×/4.5×/9× | 选择/聚合路线常用中等压缩；训练条件不同于本项目 |
| [BLIP-2，ICML 2023，§3.1](https://arxiv.org/html/2301.12597) | 论文举例 ViT-L/14 为 257→32，约 8×；257 包含该例的特殊 token | Query bottleneck 参考，不推广成所有 encoder 配置的统一倍数 |
| [Qwen2-VL，2024](https://arxiv.org/html/2409.12191) | 相邻 2×2 tokens 合并为一个，即 4× | 固定局部合并倍率的系统设置 |
| [Matryoshka Query Transformer，NeurIPS 2024](https://arxiv.org/html/2405.19315) | 576→256/144/64/36/16/8/4/2，覆盖 2.25×–288× | 完整预算曲线与弹性训练参考；普通 query bank 不具备自动截短能力 |
| [PARCEL，2026 预印本，附录 E](https://arxiv.org/html/2605.30126) | 默认 256→256/64/16，即 1×/4×/16× | 空间池化与 queries 结合的预算实验 |
| [Flamingo，NeurIPS 2022，§2.1](https://arxiv.org/html/2204.14198) | 可变长度输入→64 latents，无统一固定 N/M | 64 是输出预算；其 LLM 交叉注意力接口与本项目也不同 |
| [LLaVA-Mini，ICLR 2025](https://arxiv.org/html/2501.03895) | 代表性 576→1；先做视觉—文本预融合 | 不能将 576× 直接解读为独立单视觉向量承载全部信息 |
| [M3D/LaMed，2024，§6.1](https://arxiv.org/html/2404.00578) | 2048→256，即 8× | 3D 医学空间池化的直接依据 |
| [Argus，Findings ACL 2025，§4.2](https://aclanthology.org/2025.findings-acl.845.pdf) | pooling/pixel-shuffle：2048→256、4096→512，均为 8× | 输入增加时输出也增加，不能只照搬 256 这个数 |
| [NeuroVFM，Nature Medicine 2026，Connector architecture](https://pmc.ncbi.nlm.nih.gov/articles/PMC13472962/) | 每序列可变长度→64 latents，检查级总量为 64×序列数 | 与本项目原网格且不滤背景的 N 不同，不能声称原论文统一采用 8775/64 |
| [Resolution Meets Reduction，2026 预印本，Tables 1、5](https://arxiv.org/html/2608.08713v1) | Primus：13824→1728/512/216，即 8×/27×/64×；Curia：8192→2048/512/128，即 4×/16×/64× | 与本项目输入规模接近；提供多 Projector、多压缩档位和不压缩 MLP 的 3D 报告生成比较 |

这些代表性工作支持覆盖温和到较强压缩的多档比较，并不构成全领域统一标准。综合本项目输入规模、固定 LLM 和单序列报告目标，当前采用约 4×/8×/16×/32× 是用户已确认的实验设计，不是已验证的最优工作点。

### 15.7 四档顺序训练与评测入口（2026-10-12）

[`scripts/baselines/`](../scripts/baselines/README.md) 提供全部 11 个方法的独立 Bash 文件。
每个压缩方法默认按 4→8→16→32× 顺序，逐档执行独立 `joint_lora` 训练与该档 `best.pt` 评测；
MLP 不压缩，只执行一次 1×。不将不同压缩档位作为续训，不沿用前一档 projector/LoRA 权重。
单卡默认使用逻辑设备 `cuda:0`，训练、生成和评分沿用该设备。通过启动命令前的
`CUDA_VISIBLE_DEVICES=3` 选择 GPU 3；11 个 Bash 入口不会覆盖外部可见设备设置。

```bash
# 预检实际清单、输入几何、预算和训练日程，不加载模型。
bash scripts/baselines/run_concat3d.sh --dry-run

# 四档训练，每档训练结束后立即评测。
CUDA_VISIBLE_DEVICES=3 bash scripts/baselines/run_concat3d.sh --run-tag concat_joint_v1

# 只选择两个压缩档位。
CUDA_VISIBLE_DEVICES=3 bash scripts/baselines/run_local_packer3d.sh --ratios 8 32
```

训练 preset 为 `configs/training/joint_lora.yaml`：当前 BraTS 1960 条训练序列、有效 batch 16、
最多 10 epochs 推导出 1230 次更新，即 77 次 projector warm-up、77 次联合 warm-up、1076 次联合 cosine。
两组峰值 LR 均为 1e-5，视觉 LoRA 为 rank 16 / alpha 32 / dropout 0.05；
早停 min_delta=0.005、patience=3，最终视觉特征缓存关闭。上述步数随数据量与覆盖参数计算，不硬编码在脚本中。

默认评测完整 BraTS test、全部现有指标，继承各自 checkpoint 配置；可用 `--eval-split val`、
`--eval-sample-fraction`、`--metrics` 调整。通过验证集选择超参数；同一比较表的评测范围和指标保持一致。
输出为 `runs/baselines/<run-tag>/r<ratio>/`，保存解析后的配置、预算、训练与评测日志；
已有目录拒绝覆盖，训练/评测失败即停止后续档位。训练参数通过 `--set` 覆盖；原 Trainer 的断点恢复仍可用，
具体命令与 11 个入口名称见链接中的说明。

真实启动脚本 smoke 发现两个 6 层 Perceiver 在完整 8775-token、4×、确定性注意力下反向显存超过 24 GiB；
其方法 YAML 因此启用已有 `checkpoint_blocks=true`，以激活重计算节省显存，保持层数、宽度、参数量、
token 预算和优化日程不变。原始失败日志保留；最终重跑结果见
[`baseline_launchers_20261012T161710Z/verification.json`](../runs/archive/engineering/baseline_launchers_20261012T161710Z/verification.json)。

最终工程结果：193 项测试通过，11 个脚本覆盖 41 套真实短训练（每套 6 次更新、完整 N=8775）全部通过，
输入/输出预算、学习率阶段与参数更新符合预期。39 套通过短生成的非空/流程检查；Flamingo 8× 恢复 192-token 上限后评测流程通过，**但原始 prediction 是重复标点，仍是生成退化**。Concat 32× 的 6-step checkpoint 仍生成重复特殊 token、文本为空，生成验收未通过。失败记录保留。这里记录的是该轮 smoke 当时的状态；该轮未启动正式 10-epoch 训练，也未证明这些 checkpoint 的报告质量。详细问题分级、CE 数值和单卡建议见第 16 节。


## 16. Smoke 问题复核、单卡预算与类别覆盖（2026-10-12 PDT）

本节核对原始日志、checkpoint 审计、预测 token IDs 和当前源代码。仅修正文档与结论，未启动新的模型训练、修改训练超参数或干预正在运行的实验。

### 16.1 出现过哪些问题，哪些已经解决

证据入口：[verification.json](../runs/archive/engineering/baseline_launchers_20261012T161710Z/verification.json)、[41 组逐项审计](../runs/archive/engineering/baseline_launchers_20261012T161710Z/real_smoke_verification.json)。

| 事件 | 原始表现 | 处理与当前结论 |
| --- | --- | --- |
| Perceiver-NV，4× | 关闭 projector 激活重计算时，确定性全局 cross-attention 反向 OOM；申请 588 MiB、只余约 427 MiB | 方法 YAML 开启 `checkpoint_blocks=true` 后，4/8/16/32 四档短训练都通过；4× 峰值 allocated 19.70 GiB |
| Perceiver-Flamingo，4× | 同设置反向 OOM；申请 736 MiB、只余约 129 MiB | 同样开启重计算后四档短训练通过；4× 峰值 allocated 20.29 GiB |
| Concat，32× | 2-token 输出都是特殊 token；同 checkpoint/同 val 样本重试 192 tokens 仍全为 128006 | 训练梯度/更新通过；**生成仍未通过**，不是已解决问题；CE 也明显偏高，见下一节 |
| Perceiver-Flamingo，8× | 2-token 输出解码并 strip 后为空 | 192-token 重试非空，评测流程返回 complete；**实际是重复点号，不能算有效报告或质量通过** |
| 验证脚本的断言 | 初版误认为 optimizer 必须恰好两个 parameter groups | 实际是两种 LR component，projector 又分 decay/no-decay；已修正验证脚本并独立重审 41 组，没有因此修改生产训练算法 |

两个 OOM 的原始日志分别为 [NV](../runs/archive/engineering/baseline_launchers_20261012T161710Z/real_perceiver_nv_without_checkpointing_oom.log)、[Flamingo](../runs/archive/engineering/baseline_launchers_20261012T161710Z/real_perceiver_flamingo_without_checkpointing_oom.log)。确定性 QueryBlock 在本地使用 SDPA math backend，6 层对完整 N 与 M 的 attention 会保留较多激活；重计算是在反向时重新生成这些激活，不减少 token 数或模型层数。

这轮 41 组 sweep 中有证据的 OOM 是上述两个方法的 4×；其余方法本轮没有记录 OOM，不能据此保证任意 batch、长报告和共享显存条件都不会 OOM。更早的独立测试还有：

- 不压缩 MLP 的 N=8775 B-LoRA 曾在共享 GPU1 因其他进程占显存而 OOM（见第 13.6 节）；最新完整模型短训练在空闲卡已通过，峰值 allocated 21.02 GiB。
- joint_lora 恢复验收曾在同一个 Python 进程构造两份 8B Trainer，第二份加载 OOM；改为独立进程运行后连续/恢复对照通过，详见 [训练方案第 14 节](NEUROVFM_MLP_TRAINING_PIPELINE.md)。这属于测试脚本的模型生命周期问题，不能算作某类压缩器需要双卡。

193 项单元/集成测试、41 组训练连通性、39 组短生成流程、Flamingo 重试流程、临床生成质量是不同验收层次。不能把前几项相加表述成“所有模型都能生成合格报告”。

### 16.2 空文本与 CE/反向传播为什么是两回事

训练采用 teacher forcing：给定图像、prompt 与真实答案前缀，用下一真实答案 token 计算交叉熵。损失为 `-sum_t log p(y_t | image, prompt, y_<t)`，训练不先 argmax 生成一份报告，也不通过解码后的字符串反向传播。冻结的 LLM 保留对视觉 embedding 的输入梯度，因此 CE 仍可回传至 projector 和可训练视觉 LoRA。

独立生成使用模型自己的前缀逐 token argmax，最后 `decode(skip_special_tokens=True).strip()`。Concat 32× 实际生成 192 个 `<|start_header_id|>`（token ID 128006），去除特殊 token 后字符串为空。这里“生成了空报告”不等于“没有输入 token”“没有 logits”或“没有监督标签”。

原始事实：

- Concat 32× 六次训练更新记录的 CE 为 `2.6705, 2.2069, 27.5329, 28.3548, 18.3296, 14.8698`；对应每步梯度范数均有限且非零，projector/视觉 LoRA 权重变化已核实。每步是不同样本，不能把这六个点直接当同一病例的收敛曲线。
- 训练过程中那 1 条验证样本的 CE 为 `14.4360`；独立 192-token 失败生成对应的另 1 条 val 样本 CE 为 **13.8264**，见 [失败评测 metrics.json](../runs/archive/engineering/baseline_launchers_20261012T161710Z/concat_r32_generation_192/metrics.json)。所以已有“文本为空但 CE 可正常计算”的直接证据。
- [evaluate.py](../src/cli/evaluate.py) 会生成报告、计算指标，再做独立 teacher-forced CE；将含空生成的结果写盘后返回失败。Bash sweep 随之停止；该失败不会抹去已经完成的训练和 checkpoint，也不表示 autograd 断路。
- CE 较高与重复特殊 token 是值得排查的退化信号。有限梯度只能证明计算连通，不能证明优化正常；6-step smoke 不能说明继续训练一定恢复，也不能据此判定 Concat 机制不能训练。

Flamingo 8× 的 [192-token prediction](../runs/archive/engineering/baseline_launchers_20261012T161710Z/flamingo_r8_generation_192/predictions.jsonl) 非空但反复输出点号。先前“通过”仅指流程和非空判据；本轮补上质量限制，不修改原始证据的 complete 状态。

### 16.3 单卡实测显存

下面是 **RTX A5500、完整 NeuroVFM + 冻结 BF16 Llama-3.1-8B、N=8775、视觉 LoRA rank16、microbatch=1、每配置 6 次更新** 的训练峰值 `torch.cuda.max_memory_allocated`（GiB）。两种 Perceiver 使用已修复的激活重计算 preset；不是只计算 projector 的显存，也不包含随后所有临床评分模型的峰值。

| 方法 | 4× | 8× | 16× | 32× |
| --- | ---: | ---: | ---: | ---: |
| `mlp2x_gelu` | — | — | — | — |
| `avgpool3d` | 17.79 | 17.25 | 17.05 | 17.05 |
| `concat3d` | 17.95 | 17.56 | 17.85 | 18.69 |
| `convpool3d` | 18.22 | 17.68 | 17.41 | 17.27 |
| `mixer3d` | 18.44 | 17.90 | 17.63 | 17.50 |
| `local_packer3d` | 18.21 | 17.55 | 17.27 | 17.16 |
| `attention_pool` | 19.15 | 17.98 | 17.40 | 17.16 |
| `perceiver_nv` | 19.70 | 18.53 | 17.95 | 17.69 |
| `perceiver_flamingo` | 20.29 | 18.68 | 17.99 | 17.69 |
| `adaptive_merge3d` | 17.79 | 17.25 | 17.05 | 17.05 |
| `token_select3d` | 17.79 | 17.25 | 17.05 | 17.05 |

MLP 仅 1×，其峰值为 **21.02 GiB**。其余四档数值从原审计逐项读取。参数/优化器显存也影响总量：Concat 32× 的固定块包含 36 个槽位，映射头第一层输入为 `36×768=27648`，不能因为 M 更小就假定其总显存必然小于 Concat 8×。

这些是小样本短训练实测，不是全训练集的最坏上限。正式训练还需留出 reserved、CUDA context、较长答案和生成 KV cache 的空间；有效 batch 通过累积增加时仍保持 microbatch=1。共享卡剩余显存不足时仍可能 OOM。多卡并非这 11 个方法短训练的必要条件。

AdaptiveMerge 的离散 Ward 路由在 CPU 上运行，显存低但耗时较多。本轮 8×/32× 的平均每次单样本更新约 3.03/2.10 秒，而 AvgPool 约 1.76/1.28 秒；这些仅作本次运行的开销线索，不能当跨机器或长训练速度保证。训练视觉 LoRA 时特征变化，不能用过期路由缓存省掉这部分成本。

### 16.4 每类至少一个代表：建议 7 个；若合并选择/合并则 6 个

| 按用户表格拆分的类别 | 建议代表 | 选择理由与状态 |
| --- | --- | --- |
| 逐 token 映射 | MLP | 不压缩参考；完整输入单卡短训练通过 |
| 固定空间池化 | AvgPool3D | 最简单的空间聚合对照；四档通过 |
| 空间到通道重排 | Concat3D | 保留局部槽位；四档训练通过，但 32× 报告退化需先定位 |
| 可学习空间混合 | ConvPool3D | 本地轻量卷积实现，四档通过；Mixer 是同类备选，不预先声称谁更强 |
| 查询式聚合 | Perceiver-NV | NeuroVFM 相关的全局多轮查询代表；开启重计算后四档单卡通过 |
| 内容自适应合并 | AdaptiveMerge3D | 空间连通、内容驱动的区域合并；保留所有 token 经聚合的贡献，代价是 CPU 路由时间 |
| 内容自适应选择 | TokenSelect3D | 代表直接选取原始 token 的路线；显存低、通常比本地 Ward 路由更快；未选 token 无直接下游梯度 |

**推荐保留上述 7 个，先做 8×，Concat 32× 定位后再补 32×。** 这基于机制覆盖和工程资源，不是用六步后的 CE 排名，更不能称已选出性能最强的七种。若按六大类口径将选择/合并归为一类，可先只保留 AdaptiveMerge，得到 MLP + 5 个压缩方法。

如果显存或速度要求进一步收紧，查询类可把 Perceiver-NV 换成 LocalPacker3D（8×/32× 约 17.55/17.16 GiB）或 AttentionPool（约 17.98/17.16 GiB）；都已有单卡短训练证据。LocalPacker 是局部查询，AttentionPool 是全局单层，替换后研究问题会改变，不能叫同一个 Perceiver 的加速版。没有必要为了运行而直接删除已能跑的 Perceiver-NV；若新方法使用局部 attention，LocalPacker 应作为关键近邻对照追加。

目前不建议把 Flamingo 作为唯一 query 代表：与 NV 同类且还有重复标点诊断项；它适合后续 latent 交互消融。Concat 32× 的问题也不能通过用另一类方法替换、却仍宣称“覆盖了空间到通道路线”来回避。

上述只是选择建议，未修改 11 个 Bash 入口的默认四档、结构、学习率、损失或解码策略。

### 16.5 下一步怎样定位生成退化（待执行的诊断方案）

1. **保留并复现失败**：原 checkpoint、同一 val 样本、192-token greedy、token IDs、CE、输出预算、实际生效配置一起记录。比较“未训练 / 各边界 / 训练后”同一个样本，避免把不同病例造成的损失差异当优化发散。
2. **核对标签与解码接口**：检查 chat template 中 assistant 起点、图像占位替换、labels 的 -100 范围、一次 causal shift、首个监督 token 与生成第一步 logits 对齐；特殊 header 不应混入答案监督。已有其他方法通过也不能代替对这个失败样本的检查。
3. **检查 Concat 数值与更新**：记录 encoder feature、拼接向量、两层映射后的 RMS/范数/极值，与 Llama 文本 embedding 尺度比较；保存实际 LR、裁剪前后梯度、每层参数更新/参数范数比。特别比较 8/16/32×。拼接通道增加不自动证明激活变大，必须测量。
4. **做真正的小集过拟合**：固定少量 train 病例反复训练，另设固定 val 观察退化；先 projector-only 定位，再启用视觉 LoRA。原 2+2+2 日程的六次 optimizer.step 中，仅 step2/4/5 的 projector LR 非零，所以 smoke 不足以充当过拟合实验。诊断日程与正式主比较分开记录。
5. **按证据修正**：发现 mask/shift 等实现错误则修复并回归；若是输出尺度/优化问题，再比较较平缓 warm-up、学习率或有明确版本号的归一化变体。改变结构后另命名并给相关对照相同调参预算，不只给失败方法做无限调参，也不直接把 LN 塞进 Concat 后仍称实现未变。
6. **用有效生成而非非空字符串验收**：检查空文本率、特殊 token 循环、标点占比/重复 n-gram、生成长度与实际可读 Findings；抽查与对应序列证据是否一致。检测项只报告失败，不能删除失败病例、放宽空文本检查或通过屏蔽 token 把问题隐藏起来。最终配置只用 train/val 决定。

这轮只完成证据复核和文档更新；上面的定位实验尚未执行，不声称已修复 Concat 的生成问题。
