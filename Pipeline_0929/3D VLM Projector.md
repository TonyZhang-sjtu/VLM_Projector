# 3D VLM Projector

# Getting Started

## Paper Reading

### [2024] Large Language Model with Region-guided  Referring and Grounding for CT Report  Generation

- 痛点：只使用全局CT特征，会遗漏局部异常；下采样和特征池化pooling会损失高分辨率纹理；只裁剪局部区域会损失几何信息；现有的region-level方法缺少区域间关系；全局异常可能被错误归因到目标区域，产生FP。

![image.png](3D%20VLM%20Projector/image.png)

- **Volume Encoder**：看整个 CT 里“有什么纹理和病变”；
- **Mask Encoder**：看器官“在哪里、多大、什么形状”；
- **Adapter**：把复杂 CT 特征压缩并翻译成多个 LLM token；
- **Projection**：把简单的 mask 几何向量映射成一个 LLM token。
$\boxed{
\text{完整CT全局token}
+
\sum_j
(
\text{区域纹理token}
+
\text{区域几何token}
)
\rightarrow
\text{同一个LLM上下文}
}$
这里的“+”表示放入同一 token 序列，真正的信息交互发生在 LLM 的 self-attention 中。

Target:

- An example of 3D CT Dataset; e.g. CT-RATE（这是原始数据集） RadGenome-ChestCT（这是带区域的）
- 熟悉作为baseline的3D Medical VLM是什么
- 搜索关于reasoning的3D Medical VQA + Brain MRI 的数据集
- 熟悉 Prof. Ehsan Adeli 近期关于Brain VQA的工作

# Week 0824-0830

- An example of 3D CT Dataset: CT-RATE（原始数据集）& RadGenome-ChestCT（带区域）

| 方面 | CT-RATE | RadGenome-ChestCT |
| --- | --- | --- |
| 定位 | 原始 3D CT-报告数据集 | CT-RATE 的区域化、grounded 扩展 |
| 患者数 | 21,304 | 相同的 21,304 名患者 |
| 原始 CT 检查/报告对 | 25,692 | 基于相同的 25,692 对 |
| 多重重建后的 volume | 50,188 | 标准化后保留 25,692 个 volume |
| 影像 | 非增强胸部 3D CT | 同一批 CT，统一为 $1\times1\times3$ mm |
| 文本 | 完整放射学报告 | 原始报告 + 按解剖区域拆分的报告 |
| 全局异常标签 | 18 类 | 继承自 CT-RATE |
| 区域 mask | 没有系统的区域 mask | SAT 自动生成的 197 类解剖 mask |
| 图文对应关系 | 一个完整 CT 对应一份完整报告 | 报告句子对应具体解剖区域 mask |
| 衍生 VQA | 可从报告构造 | 约 1.3M grounded VQA 对 |
| 典型用途 | 全局 VLM 预训练、报告生成、异常分类、检索 | 区域级报告生成、grounded VQA、区域理解、可解释性 |

## About Datasets

### **Brain MRI 中的 3D VQA / Reasoning数据集**

| 数据集 | 规模与影像输入 | 标注与任务内容 | 常用/官方评价指标 | 匹配度 |
| --- | --- | --- | --- | --- |
| [RadGenome-Brain MRI / AutoRG-Brain](https://arxiv.org/html/2407.16684) | 1,007病例、3,408个影像—报告对；来自BraTS2021、BraTS-MEN、BraTS-MET、ISLES2022、WMH；涵盖T1、T1CE、T2、FLAIR、DWI、ADC六种序列。官方按单个3D序列构造影像—报告对，不等价于3,408个完整多序列患者 | 病灶mask；伪解剖结构mask；放射科医生撰写的Findings/Impression；GPT-4将病例报告拆成序列级、区域级描述。支持病灶分割、区域报告生成、整脑报告生成 | 报告：BLEU-4、ROUGE-1、BERTScore、RadGraph F1、RaTEScore、RadCliQ；分割：Dice、precision、sensitivity；另有人类评估 | **4.5/5**。空间—语言对齐极强；适合local/global projector。缺点是官方建模通常将多序列拆开，与真正的跨序列融合有gap |
| [mpLLM synthetic spatial VQA](https://arxiv.org/html/2509.25889) | BraTS 2024 GLI 1,621例、MET 651例、GoAT 1,351例，共3,623例；每例联合T1、T1Gd/T1CE、T2、FLAIR；共74,940个QA：GLI 38,904、MET 11,718、GoAT 24,318 | 从专家分割mask确定性生成：肿瘤相对体积、所在脑区、3D形状、病灶扩散模式、是否超出图像能力范围；再用GPT-4o做语言改写。空间区域由AAL、LPBA40/SRI24 atlas产生 | 体积/形状/扩散/拒答：accuracy；脑区：multi-label per-label accuracy；500次bootstrap；另有放射科医生可接受性和下游诊断AUROC | **5/5**。五者中最适合直接研究“多序列3D projector是否保留空间信息”；但临床语言真实性和自由报告能力较弱 |
| [NeuroQA](https://arxiv.org/html/2605.20525) | 56,953 QA、12,977名受试者、12个来源数据集；年龄5–104岁；覆盖AD、PD、肿瘤、WMH、健康老化和神经发育；以完整3D T1为主，部分任务涉及T2或纵向扫描 | 203个模板、11类问题。26,378个Yes/No，19,167个四选一，11,408个开放问答；131个image-grounded模板，72个image-informed模板；来源包括FreeSurfer定量、临床量表、诊断、人口学和结构化元数据；不使用LLM生成答案 | Yes/No accuracy、MCQ accuracy、开放回答EM/token F1；Shortcut Score；按数据源、任务类别、问题格式分别评估 | **4/5**。适合评价泛化与语言shortcut，规模和疾病覆盖最好；对精细病灶grounding和多序列融合支持有限 |
| [SGMRI-VQA](https://arxiv.org/html/2604.15808) | 来自fastMRI+；保留1,970个volume，其中脑MRI 996、膝MRI 974；50,672张切片、13,202张带标注切片、23,724个框；共41,307 QA，其中脑MRI约20,192 QA。脑部实验报告670个训练、222个测试volume | 检测、定位、分类、诊断、caption、计数；包含image-level与volume-level问题；病灶有2D bounding box和所在frame编号；GPT-4o辅助生成QA及CoT，由临床人员审核 | A-Score：EM、F1、关键词召回与语义相似度；AR-Score：GPT judge、BERTScore、BLEU、ROUGE-L组合；V-Score：跨frame bbox mIoU/Hungarian matching | **4/5**。跨切片定位能力非常适合验证projector是否保存slice顺序；但它把volume当有序2D视频，不是真正的3D体素encoder，也不包含多序列联合融合 |
| [UCSF-PDGM-VQA](https://arxiv.org/html/2605.17140) | 473个术前胶质瘤MRI study、2,387个四选一问题；平均每study约23个series、每series约151张切片，包含大量临床序列 | 从真实报告Findings/Impression生成：肿瘤大小、位置、解剖改变、诊断等；比较单切片、多切片、montage、3D NIfTI、blank image、text-only等输入形式 | 主要为四选一accuracy，通常多次运行取平均；包含放射科医生基线和输入消融 | **4.5/5**。最接近你想研究的“临床多序列study级projector”；但缺少精确空间mask，问题为闭集MCQ，而且官方数字存在473/497/501−19不一致，需要按实际发布清单复核 |
- 优势、缺陷与目前的论文采用情况

| 数据集 | 主要优点 | 关键缺点/实验风险 | 已确认使用它开展实验的论文 |
| --- | --- | --- | --- |
| RadGenome-Brain MRI [Classic] | 真实放射科报告；有病灶mask和区域描述；兼顾report generation与segmentation；疾病和序列较丰富 | 解剖mask部分为自动伪标签；报告被GPT-4拆成序列级文本；数据由多个旧挑战集合汇总；未明确保证原始患者级完全无重叠；不天然支持联合多序列projector | 原始[AutoRG-Brain](https://arxiv.org/html/2407.16684)；[UniBrain](https://arxiv.org/abs/2606.16484)用于MRI补全与理解；[PerFact](https://arxiv.org/html/2608.17926)用于3D感知事实提示报告生成；[MS-RSuper](https://arxiv.org/abs/2602.20994)用区域报告监督病灶分割 |
| mpLLM synthetic VQA | 真正联合四序列3D输入；空间答案可由mask重新计算；非常适合做projector消融；包含拒答问题 | 答案主要由体积、形状和atlas规则产生；可能学习到阈值或模板捷径；atlas覆盖率只有约57.7%–70.9%；只覆盖脑肿瘤；放射科审核主要验证问题可用性，不等价于证明标签完全符合临床表达 | 已确认的直接实验主要是[mpLLM原论文](https://arxiv.org/html/2509.25889)。[NeuroQA](https://arxiv.org/html/2605.20525)将其列为相关工作，但没有复用其QA数据。暂未检索到独立论文直接复用已发布QA |
| NeuroQA (preferred) | 最大、最多疾病、subject-disjoint split；不依赖LLM生成答案；有public/private test；专门控制text-only shortcut | 72个image-informed模板依赖诊断、量表或精确体积，未必能仅从图像回答；临床人员在100题子集上也只有约48.9%，说明部分问题更像“临床元数据预测”；以结构MRI为主 | 目前已确认的系统实验主要来自[NeuroQA原论文](https://arxiv.org/html/2605.20525)，比较通用VLM、医疗VLM和监督3D CNN。发布时间较新，尚未确认独立复用论文 |
| SGMRI-VQA | 有明确bbox、frame编号和跨切片定位；测试volume接受临床审核；非常适合研究token顺序和空间token压缩 | bbox不是3D mask；单个volume通常是单序列；QA与CoT由GPT-4o生成；AR-Score依赖LLM judge；数据说明中的996个脑volume与主实验使用的892个没有完全对应 | 目前主要是[SGMRI-VQA原论文](https://arxiv.org/html/2604.15808)，并发布了基于Qwen3-VL的SFT模型；暂未确认独立论文复用 |
| UCSF-PDGM-VQA | 真实临床study级多序列；能直接比较单图、多切片、montage和完整3D输入；很适合研究“modality collapse” | 仅胶质瘤、单中心、术前单时间点；无病灶级空间GT；问题来自报告且为MCQ；原始答案位置极不平衡，第一选项曾占84.7%，必须使用shuffle版本 | 目前主要是[UCSF-PDGM-VQA原论文](https://arxiv.org/html/2605.17140)，比较MedGemma、Med3D-VLM、LLaVA-Med、Lingshu等；尚未确认独立复用论文 |

### **3D Medical VLM for Brain MRI**

| 模型 | 视觉 backbone | Projector | LLM与参数量 | 代表性结果与局限 |
| --- | --- | --- | --- | --- |
| [RadFM](https://www.nature.com/articles/s41467-025-62385-7) | 12 层 3D ViT，hidden 768；2D 图像通过复制 depth 维变成 pseudo-3D | 6 层 Perceiver，32 个 query，将任意数量扫描压缩为 32 个视觉 token | MedLLaMA-13B；总计约 14B | 能处理 2D/3D、多扫描和交错图文，是早期通用 baseline；但 32-token bottleneck 较粗，在 mpLLM 的脑 MRI benchmark 上平均准确率仅 29.0/27.7/34.9 |
| [M3D-LaMed](https://arxiv.org/html/2404.00578) (Preferred) | 12 层 MONAI 3D ViT，87.4M；用 M3D-Cap 做 CLIP 式预训练 | 3D spatial pooling + Linear/2-layer MLP，将 volume 压缩成规则视觉 token | LLaMA2-7B；projector 3.1M/19.9M；SegVol 117.3M；总计约 6.9B | 当前最常见、最容易复现的原生 3D baseline；M3D closed VQA 75.78%，open VQA METEOR 33.58；但预训练数据明显以 CT 为主 |
| [Med3DVLM](https://arxiv.org/html/2503.20047v3) | DCFormer-S：分解式 3D convolution + Transformer-like block；18.2M、21.59G FLOPs | 双流 low/high-level 2×MLP-Mixer-H，47.2M；显式融合浅层局部纹理和深层语义 | Qwen2.5-7B-Instruct；总计约 7.6B | M3D retrieval R@1 61.00%，report METEOR 36.42，closed VQA 79.95%；但迁移到脑 MRI 后不一定强，说明 CT 上的高分不能直接代表脑 MRI 能力 |
| [Merlin](https://arxiv.org/html/2406.06512) | ImageNet 权重膨胀得到 I3D ResNet-152；视觉语言预训练时配 Clinical Longformer | 报告生成时用单层 Linear，将 2048 维特征映射到 4096 | 报告生成使用 RadLLaMA-7B；论文未汇总完整总参数 | 原本是腹部 CT 模型，但作为 3D encoder baseline 很强；内部/外部 zero-shot finding F1 为 0.741/0.647。在 mpLLM 的脑 MRI 数据上也达到 58.1/55.7/63.0，说明好的 3D encoder 可有一定跨器官迁移能力 |
| [MedGemma 1.5 4B](https://arxiv.org/abs/2604.05081) | MedSigLIP + 长上下文多切片输入，并非严格原生 3D voxel encoder | Gemma 原生 multimodal connector；把切片作为长序列融合 | 4B | 相比 MedGemma 1，3D MRI condition classification 绝对提高 11%；部署方便，但切片顺序不等于真正的 3D 几何建模 |

## About Next Step Pipeline

- chest CT / lung CT 两个数据集/模态
- baseline方法 统一？现有3D VLM 同一个方法加入projector的效果 → 选择同一个VLM模型，然后只换projector部分
- pre-train 或者 finetune 一个VLM，要找一下有没有【之前针对projector的文章，他们是怎么做的】
- 换projector之后参数量的变化，如何验证不是参数的影响？要求同等级参数baseline的对齐【不同方法的projector参数要一致】
- 先复现从头train，然后加上的效果
- NBE Hao Chen, UMLS  An Explainable Biomedical Foundation Model via Large-Scale Concept-Enhanced Vision-Language Pre-training
- 找到第三种模态，e.g. 知识图，筛选子图的过程中信息作为；concept注入，得到最重要的query信息，给到llm，代替pooling产生的信息压缩

# Week 0831-0906

### **Chest/Lung CT 中的 3D VQA / Reasoning数据集**

- 重点还是关注Chest的general内容，不处理专门种类疾病

| 数据集 | 规模与模态 | 文本 / 标注 | 典型用途 | 优点 | 局限与风险 | 获取与论文 |
| --- | --- | --- | --- | --- | --- | --- |
| **CT-RATE** | 21,304 名患者；25,692 次非增强胸 CT 检查；因多种重建核得到 50,188 个重建体积，超过 1,430 万张切片 | 每次检查含 findings + impression；18 类异常标签由 1,000 份人工标注报告训练的 RadBERT 自动抽取 | 图文对比预训练、报告生成、VQA、异常分类、检索 | 公开、规模大、真实临床异质性强；患者级 train/val；同一生态内已有 CT-CLIP/CT-CHAT | 单中心；标签为 NLP 伪标签；多个重建体积来自同一次检查，划分时必须按患者/检查去重 | [论文/Nature Biomedical Engineering](https://www.nature.com/articles/s41551-025-01599-y) · [arXiv](https://arxiv.org/abs/2403.17834) · [官方代码与数据说明](https://github.com/ibrahimethemhamamci/CT-CLIP) |
| **RadGenome-ChestCT (Preferred)** | 与 CT-RATE 相同的 21,304 名患者、25,692 次检查；统一为每次检查一个 1×1×3 mm 体积 | 197 个解剖结构的自动分割；约 66.5 万条多粒度 grounded reports；约 130 万组 grounded VQA；官方评测集经人工检查 | region-aware VQA、区域报告生成、mask/geometry-aware projector、grounding | 目前最适合 chest CT 区域级视觉—语言对齐的数据；区域、器官、关系层次丰富 | **不是 CT-RATE 的独立数据集**；大部分 mask、grounded report 和 VQA 为模型生成/自动构造，存在伪标注噪声 | [论文](https://arxiv.org/abs/2404.16754) |
| **BIMCV-R** | 8,069 个 3D CT 体积，超过 200 万张切片；Argus 清洗后使用约 5,322 对 CT–report | 放射学报告、检查与疾病元数据；原始西班牙语报告及经人工核验的 GPT-4 英译 | 报告生成外测、高分辨率 3D 建模、跨语言/跨中心泛化 | 分辨率较高；与 CT-RATE 独立；适合检验模型是否只记住 CT-RATE 写作风格 | COVID 时代采集造成病种偏移；翻译文本与原始语言风格存在差异；不同论文清洗后的样本数不一致 | [BIMCV-R 论文](https://arxiv.org/abs/2403.15992) · [BIMCV 项目](https://bimcv.cipf.es/bimcv-projects/bimcv-covid19/) |
| **INSPECT** | 19,402 名患者，23,248 个 CTPA 检查，2000–2021；影像、报告 impression、结构化 EHR | 肺栓塞诊断、报告结论、临床变量与预后终点 | CTPA 专病 VLM、报告生成、诊断与预后、多模态融合 | 真实临床纵向信息丰富；与 CT-RATE 有显著域差异，是有价值的外部验证 | 仅 CTPA、非普通非增强胸 CT；通常只有 impression 而非完整报告；约 2.5 TB，需 DUA；EHR 会引入额外模态混杂 | [NeurIPS 数据论文](https://arxiv.org/abs/2311.10798) · [官方仓库](https://github.com/som-shahlab/INSPECT_public) |
| **CTRG-Chest-548K** | **548,696 是 2D 切片数**；实际只有 1,804 个 3D CT–report 对 | 中英文胸 CT 报告 | 小规模报告生成、与 CT2Rep/Dia-LLaMA/Reg2RG 对比 | 历史基线使用多；体量小，适合快速迭代 | 名称容易被误解为 54.8 万个体积；患者信息、数据清洗和标准划分文档较弱；不同工作使用 60/20/20 或 80/20 划分 | [官方仓库](https://github.com/tangyuhao2016/CTRG) |
| **CT-3DRRG** | 汇总 BIMCV-R、CT-RATE、INSPECT；论文清洗后共 49,849 对，39,721/4,984/5,144 train/val/test | 胸 CT / CTPA 与报告 | 大规模跨数据源报告生成、Argus 训练与评测 | 把三个主要公开源统一到同一任务；含常规与高分辨率两套预处理 | 是**聚合数据集而非新的独立队列**；源数据协议和许可不同；需防止与 CT-RATE、BIMCV-R、INSPECT 单独测试发生重叠 | [Argus / ACL Findings 2025](https://aclanthology.org/2025.findings-acl.845/) |
| **M3D-Data** | 12 万 3D image–text 对、66.2 万 instruction–response；覆盖多器官 CT/MRI/PET 等 | caption、VQA、referring segmentation、segmentation instruction；大量文本由公开来源整理及 LLM 构造 | 通用 3D VLM 预训练与统一 benchmark | 任务齐全、代码和模型生态成熟；适合先验证 projector 是否工作 | 不是胸部专用临床队列；主要来自 Radiopaedia 与公开分割集，数据分布和真实医院报告不同；合成 QA 占比较高 | [M3D 论文](https://arxiv.org/abs/2404.00578) · [官方仓库](https://github.com/BAAI-DCAI/M3D) |

### **Chest/Lung CT中的 3D VLM 模型**

| 模型 / 论文 | 视觉编码器 → Projector → LLM | 训练数据与任务 | 特点 | 优点 | 缺点 / 公平比较风险 | 复现建议 |
| --- | --- | --- | --- | --- | --- | --- |
| **CT-CLIP / CT-CHAT** | CT-ViT（两阶段空间 + causal transformer）→ multimodal projector → Llama 3.1 系列；最佳配置为 70B | CT-RATE；CT-CHAT 使用从报告派生的 270 万+ QA，覆盖长/短答案、选择题和报告生成 | 先做 projector 医学概念对齐，再训练 projector + LLM LoRA | 胸 CT 领域最匹配；视觉编码器有大规模图文对齐；模型、数据与代码公开；同时覆盖分类、检索、VQA、报告 | CT-ViT 显存开销较大；70B 最佳版不利于多数实验室迭代；VQA 来自同一 CT-RATE 报告，不能作为独立知识证据；原文对 projector 本身的结构消融有限 | **第二阶段首选**：保留 CT-CLIP encoder，改用固定 7B/8B LLM，统一 projector token 数后比较 |
| **M3D-LaMed （Preferred）** | 3D ViT（CLIP 式图文预训练）→ spatial pooling + Linear/两层 MLP → LLaMA2-7B | M3D-Data；caption、VQA、report、segmentation/referring | LLaVA 风格的通用 3D 医学 MLLM，总参数约 6.9B | 开源完整、结构简单、最容易替换 projector；训练/评测任务统一；已有大量后续工作沿用 | 非胸 CT 专用；标准化与下采样可能丢小结节；公开语料和合成指令与真实报告存在域差 | **第一阶段首选**：作为固定骨架完成 projector 参数/token 公平消融 |
| **RadFM** | 12 层 3D ViT（patch 32×32×4）→ 6 层 Perceiver、32 个视觉 latent → MedLLaMA-13B | MedMD / RadMD；多模态、多器官、2D+3D | 早期通用 3D 医疗 foundation model；可处理多张/多体积图像 | 是常见历史 baseline；Perceiver 是固定 token 压缩的典型代表 | 约 14B、训练需大量 A100；32 token 瓶颈可能压掉小病灶；旧模型在胸 CT 专项能力上弱于新模型 | 作为 **Perceiver 历史基线**，不建议作为主开发框架 |
| **Med3DVLM** | DCFormer-S 3D encoder → 双路 high/low-resolution 2×MLP-Mixer-H → Qwen2.5-7B-Instruct | M3D 数据；检索、报告生成、开放/封闭 VQA | 双分辨率视觉流；用 MLP-Mixer 在 projector 内混合空间 token | projector 设计强；报告生成与 VQA 表现明显优于简单 MLP；只训练 projector + LoRA | 最佳 projector 约 47.2M 参数，而简单 MLP 约 15.6M；参数量和计算量同时变化，不能直接证明收益仅来自结构 | 适合作为你的 **projector 强基线和消融参考**；必须增加 parameter-matched MLP |
| **CTInstruct** | 2D ResNet 混合切片特征 + patch/slice attention → 动态 token 的线性 bridger → Qwen2-VL-7B-Instruct；另接 segmentation decoder | CT-RATE、RP3D、INSPECT、Rad-ChestCT、LUNA16、LNDb、NSCLC 等；诊断、VQA、报告、分割 | 将 volume、voxel、text 三种粒度统一做预训练和 instruction tuning | 多任务范围最广；胸 CT 与肺结节数据覆盖好；AAAI 2026 正式发表且代码公开 | encoder、token 数和多任务监督一起变化；线性 bridger 本身不是强 projector；不能直接归因于 connector | 用作 **统一任务上界**，不作为纯 projector 主对照 |
| **Argus** | 3D ViT-B（3D-MAE + 图文 FLIP）→ AveragePool / Perceiver / PixelShuffle + Linear/2-layer MLP → Llama 3.2 3B 或 Llama 3.1 8B/70B | CT-3DRRG；报告生成；常规 256×256×64 与高分辨率 512×512×256 | 系统比较三种 3D token 压缩；pixel shuffle 将 token 数降 8 倍并扩展通道，尽量不丢元素 | 与 projector 课题高度相关；同时检验高分辨率；报告生成设置真实 | encoder 预训练、分辨率、LLM 尺寸都会影响结果；官方仓库截至 2026-09 仍为空，复现成熟度低于 M3D/CT-CHAT | 把 **PixelShuffle + 2-layer MLP** 纳入 baseline；先在自己的统一框架重实现 |
| **Resolution Meets Reduction** | 冻结四种 3D/2D medical encoder → 六类 projector → Qwen3-VL-2B 等五种 LLM；projector 全训 + LLM LoRA | CT-RATE 与 Merlin；报告生成 | 在匹配输出视觉 token budget 下比较 AvgPool、ConcatAdapter、TokenPacker、AttentionPool、PerceiverResampler、MLP，并研究 ROI crop 与分辨率 | **与当前课题最直接**；结论表明 ROI crop 稳定有益，高分辨率收益取决于 projector，Perceiver + 高分辨率 Curia 最强 | 2026-08 的 v1 预印本；截至 2026-09 代码/模型尚未公开；不同 projector 参数量仍需额外控制 | 作为实验设计蓝本：复现六类 projector，再加入你的方法；务必同时匹配 token 和参数 |

### Projector Baselines

| 类别 | 代表实现 | 是否压缩 token | 优点 | 缺点 | 建议的公平对照 |
| --- | --- | --- | --- | --- | --- |
| Token-wise mapping | Linear、2-layer MLP | 否 | 最干净、参数少；能作为所有方法的下界 | 高分辨率时 token 太多、LLM 成本高；不做空间交互 | 相同输入 token，Linear 与 parameter-matched MLP |
| 固定池化 | AveragePool / spatial pooling | 是 | 零/低额外参数、速度快、容易匹配输出长度 | 小病灶和稀疏征象容易被平均掉 | 固定输出 256 token；报告小结节子集结果 |
| 无损重排 | ConcatAdapter、PixelShuffle | 是 | 将邻近 token 拼到 channel，压缩序列时保留元素 | channel 维度扩大，后续 MLP 计算上升；局部重排不具备语义选择 | 与 AveragePool 使用相同压缩率和输出 token 数 |
| 局部学习压缩 | TokenPacker、卷积/MLP-Mixer | 是 | 能在局部窗口内学习空间聚合；对纹理和小结构更友好 | 参数量/FLOPs 容易高于池化；窗口大小引入超参 | 另设 parameter-matched MLP；报告 projector FLOPs |
| 查询式压缩 | AttentionPool、Q-Former | 是 | 学习“哪些视觉区域值得保留”；可与文本 query 交互 | 固定 query 可能忽略未覆盖异常；交叉注意力计算较高 | 固定 query 数 64/128/256；区分 learned-only 与 text-conditioned |
| Latent resampling | Perceiver Resampler | 是 | 输入长度可变、输出长度固定；适合大体积和多分辨率 | 多层 cross/self attention 增加计算；极低 latent 数会成为信息瓶颈 | 与 AttentionPool 相同 latent 数、层数和隐藏维度 |
| 区域 / mask-aware | Reg2RG、MedRegion 风格 | 是 / 多路 | 空间定位明确；有利于长尾病灶与解剖一致性 | 需要分割器或伪 mask；输入监督不同，容易造成不公平 | 单独列为“额外区域监督”实验，不能混入纯 projector 主表 |
| 知识增强 | UMLS / disease prototype / graph-aware projector | 视实现而定 | 可显式约束疾病、解剖、关系概念；适合你的后续创新 | 可能通过文本标签泄漏答案；知识检索和视觉压缩贡献难拆分 | 增加无图像、随机概念、仅训练集概念三种负对照 |

### Related Papers

#### **[Arxiv 2608] Resolution Meets Reduction: Efficient Visual Context for 3D Radiology Report Generation**

- 任务：3D CT Report Generation on Chest and Abdomen 胸部腹部CT
- **核心问题：如果我只能向 LLM 发送 M 个视觉 token，应该选择哪个 projector，才能获得最好的报告生成效果？【从部署角度看】**
- 采用标准的 LLaVA 式三段结构：

$\text{CT}\xrightarrow{\text{Vision Encoder}}
N\text{ 个视觉 token}
\xrightarrow{\text{Projector}}
M\text{ 个 LLM token}
\xrightarrow{\text{LLM}}
\text{报告}$
Vision Encoder 冻结；projector 完整训练；LLM Decoder使用 LoRA 微调；用 teacher forcing 的 next-token prediction 训练报告生成。【先projector，再LLM LoRA + projector的二阶段训练】

- 联合训练LLM Decoder而不只是Projector：为了在语言输出上对齐专业领域内容。如果只训练projector，其还需要更多承担除了编码压缩之外的LLM Decoder任务
- 4 个医学视觉编码器：
    - Primus：3D ViT，MAE 预训练；
    - Curia：逐切片 2D ViT，DINOv2 式预训练；
    - Merlin：3D ResNet，CLIP 预训练；
    - CT-FM：3D SegResNet，SimCLR 预训练。
- 5 个压缩 projector：AveragePool；ConcatAdapter；TokenPacker；AttentionPool；PerceiverResampler。

| 类型 | 方法 | 工作机制 | 实验表现 |
| --- | --- | --- | --- |
| Grid-based | AveragePool | 局部 token 直接求平均 | 低压缩率强，高压缩容易抹除细节 |
| Grid-based | ConcatAdapter | 局部 token 拼接后投影 | 压缩越强，参数量越大，明显退化 |
| Grid-based | TokenPacker | 低分辨率查询从高分辨率特征中聚合 | Primus 和高分辨率场景表现较好 |
| Query-based | AttentionPool | 一层 cross-attention 聚合 | 对压缩率较稳定，但绝对性能偏低 |
| Query-based | PerceiverResampler | 多层、迭代 cross-attention 聚合 | Curia、高分辨率和强压缩下最好 |
| 不压缩 | MLP | 每个视觉 token 独立映射 | 算力足够时经常仍是最强基线 |
- 1个不压缩baseline：MLP projector
- 5 个 LLM：Qwen3-1.7B；**Qwen3-VL-2B（default）**/4B；Phi-4-mini；Gemma-3-4B
- 3个输入配置：whole（整个CT直接缩放到输入尺寸）；ROI（解剖区域裁剪 → 先用 TotalSegmentator 得到解剖结构分割，再裁剪 bounding box）；High Resolution（ROI加更高分辨率）
- 控制 projector 输出 M 个 token 数量不变
- 结论：1. ROI 裁剪是最稳定、最划算的策略；2. 更高输入分辨率不是普遍有效，对尺度较小的病变更有效；3. MLP依然是很强的baseline；4. 视觉编码器比 projector 和 LLM 更重要；5. 多模态预训练对 LLM 本身仍然有帮助；6. 测试集达到新的临床 F1 水平

#### [Nature BME] An Explainable Biomedical Foundation Model via Large-Scale Concept-Enhanced Vision-Language Pre-training

- Motivation: 1. 医学视觉语言基础模型擅长学习图文整体关联，但不一定能解释“哪些医学征象支持这个结果”。2. 传统概念可解释方法往往依赖特定任务、特定模态的专家概念标注，难以规模化。
- 提出ConceptCLIP: 以双编码器为核心的**视觉语言预训练方法**
1. 构建MedConcept-23M数据集
2. IT-Align：全局图像—文本对齐 → 使用SigLIP 风格的 sigmoid 对齐损失，使配对图文靠近、非配对图文远离。
3. **RC-Align：局部区域—医学概念对齐**
    
    对每个概念，在图像里找一个最匹配的区域，再聚合所有概念的匹配分数，用于图文配对训练。
    
    1. 获取图像 patch 特征  $v_i$。
    2. 找到某医学概念在图注中对应的 token。
    3. 对这些文本 token 的上下文表示做平均，得到概念表示 $c_j$。
    4. 计算区域和概念的相似度：$a_{ij}=\cos(v_i,c_j).$
    5. 对每个概念，在所有图像区域中找到最匹配的区域。
    6. 聚合这些匹配分数，用于图文配对学习。
4. 推理时融合局部与全局信息
    
    训练局部聚合时采用 max；零样本分类时，则为每个概念选择 top-$k$ 个相关区域，聚合后计算局部分数，再与全局分类概率融合：$p(c)=\beta p_{\mathrm{local}}(c)
    +(1-\beta)p_{\mathrm{global}}(c),
    \qquad \beta=0.5.$
    收益包含两层：**预训练学到了局部概念对齐，推理时也明确利用了局部证据。**
    

### 初步Pipeline规划：

- **Target: CVPR 2027 [registration → Nov 10; submission → Nov 16; suppl → Nov 23]**
- **8 weeks to full paper deadline**
- 痛点：1. 3D CT经过Encoder之后会产生极长的上下文；2. 传统删除/压缩token的方法容易删除临床信息；3. 以往的工作没有很好的把Encoder, Projector, Decoder这些变量的提升给分开；4. 当前3D Medical VLM 中的projector缺乏系统比较。
    - Step 0: 继续调研相关任务的文献，他们是怎么研究projector的效果问题的。[3D VLM 中的 Projector 梳理](https://app.notion.com/p/3D-VLM-Projector-3d64b5438f1d801d818efe8c31f522c2?pvs=21)
    - Step 1: **Brain MRI** 使用 RadGenome Brain MRI 数据集在 M3D-LaMed 上面对LLM Decoder和projector做finetune（先finetune projector，再projector + LLM LoRA二阶段训练）；**Chest CT** 使用 RadGenome Chest CT 数据集在 M3D-LaMed 上面**对LLM Decoder和Projector做二阶段finetune**。**任务focus在？报告生成(preferred)还是VQA？**NeuroQA 作为Brain MRI的辅助数据集。pretrained LLM Decoder backbone保持不变。
    - Step 2: 需要给这两个任务的baseline model替换projector，依然进行**二阶段微调**，给出baseline结果。**从控制LLM token budget不变角度考虑（与Resolution Meet Reduction对齐，从推理速度角度以及如何提取到关键信息压缩到有限的token当中考虑）？**
    - Step 3: 关注当前的bottle neck是什么？Idea思考方向→Brain Chest等医学影像当中压缩的要点和困难是什么？和当前通用大模型的区别是什么？针对性设计提出更好的projector。
    - 先做出一套公平比较的baseline & benchmark框架

### Discussion 0916

- 适配多语言模型的projector，泛化性
- focus 在高效压缩
- 通过embedding，来反过来优化projector，来提升泛化性。in-context？
- 加入脑部MRI的数据
- memory？

### Pipeline Design and Proposal

- Refer to “**Resolution Meets Reduction**”
- 现在Brain MRI当中验证机制，然后用Chest CT验证可迁移性
- Motivation: 在固定视觉 token 预算下，如何保留 Brain MRI 报告所需的、具有空间位置和序列来源的临床证据，尤其是三维病灶的局部证据，以及病灶—位置—征象的对应关系。【比起一个即插即用的压缩组件，为什么要重新设计一个架构并训练Projector？】
- Projector的痛点：压缩，语义对齐和细节保留的目标差异
- **在Medical Image当中的特点和需要特别关注的问题是什么？**

| 问题 | 医疗图像中的表现 | 对 Projector 的要求 |
| --- | --- | --- |
| 信息价值不随面积增长 | 很小的异常也可能决定报告结论 | 压缩不能仅按面积、冗余度或显著性分配容量 |
| 全局语义不足以完成任务 | 知道“有异常”仍不足以描述位置、范围和形态 | 同时保留全局背景与局部证据 |
| 文本监督不完整 | 报告只写部分发现，未提及不等于不存在 | 不能把所有未提及概念当作阴性 |
| 正负状态敏感 | “有强化”“无强化”“无法判断”含义不同 | 概念表示需要否定和不确定状态 |
| 语言流畅可能掩盖错误 | 模板正确，但侧别或病灶描述错误 | 需要临床事实指标与视觉依赖性检验 |
- **在压缩问题当中可能出现的难点：**

| 失败类型 | 具体表现 | 更有针对性的设计 |
| --- | --- | --- |
| **遗漏** | 小病灶没有任何输出 token 充分表示 | 多尺度、局部召回、预算分配 |
| **混合** | 两个相似病灶聚合进同一个表示 | 实例分组、空间约束、竞争式分配 |
| **解绑** | 保留了左右、大小等信息，但对应关系错了 | 联合属性表示、关系监督、交换负样本 |
| **几何失真** | 裁剪后不知道原位置，池化后不知道范围 | 坐标、尺度、形状与视觉内容联合编码 |
| **观测条件丢失** | 不知道高信号来自哪个 MRI 序列 | 序列或协议条件化 |
| **读取失败** | 信息仍在 token 中，LLM 却没用对 | 压缩后监督、视觉依赖训练、读出适配 |
- Projector需要关注：同一空间区域+不同序列中的表现+这些表现之间的关系
- $\underbrace{\text{概念存在}}_{\text{有没有异常}}
\rightarrow
\underbrace{\text{属性保存}}_{\text{大小、信号、数量}}
\rightarrow
\underbrace{\text{实例绑定}}_{\text{这些属性属于哪一处异常}}$
- **Expected Contributions:**
    - **现象与评估贡献**：发现token预算减少时，概念识别、属性预测、实例绑定的退化并不同步。
    - **方法贡献**：设计一个针对该失败模式的压缩型 Projector，减少不同位置、不同病灶之间的错误聚合。
    - **迁移证据**：在 Brain MRI 验证后，将同一机制适配到 CT，检验结论是否只依赖脑部数据。
- GPT给出的实验建议：
    - **先在 Brain MRI 建立失败模式。** 跑简单池化、Perceiver、局部聚合三类方法，观察哪些属性或对应关系对预算最敏感。
    - **根据结果选择主问题。** 如果主要是小病灶遗漏，就研究局部证据覆盖；如果主要是位置和属性错配，就研究绑定约束；不要预先假定两者都成立。
    - **只增加解决该问题所需的模块。** 优先验证压缩后监督，再判断空间约束是否带来额外收益。
    - **机制成立后再迁移到 CT。** 复用压缩原则和评估框架，适配 Encoder 与属性定义；不必要求 MRI 权重直接零样本用于 CT。
- preferred choice

|  | Brain MRI | Chest / Lung CT |
| --- | --- | --- |
| Vision Encoder | 1. NeuroVFM (ViT-B 88M)(Report Generation上预训练)
2. BrainlAC (ViT-B 88M)
3. 3D DINO (ViT-L 307M) (MRI, CT混合数据上pretrain)
4. M3D (需要finetune, pretrained on 3D CT)
NeuroJEPA | 1. CT-FM
2. Primus
3. CT-CLIP
4. 3DINO |
| Projector Baseline | 1. Grid-based: AveragePool, Concat Adapter, TokenPacker
2. Query-based: AttentionPool, Perceiver Resampler
3. 不压缩：MLP |  |
| LLM Decoder | 1. LLama 3-8B
2. Qwen3-1.7B
3. Qwen3-VL-2B/4B
4. Phi-4-mini
5. Gemma-3-4B |  |
| Finetune | 1. 先projector做全量微调
2. 再LLM LoRA+Porjector做联合训练 |  |
| Input | 1. Whole Image
2. ROI Image
3. High Resolution Image 经过超分等预先处理 |  |
| Task | 1. Report Generation from whole image
2. VQA |  |
| Dataset | 1. Report Generation → RadGenome- Brain MRI / MR-RATE
2. VQA → NeuroQA | 1. Report Generation → CT-RATE
2. VQA → RadGenome-Chest CT |
1. 两边有先后 优先做一边？还是同时进行？优先进行Brain MRI方面的实现
2. Projector的设计在MRI和CT两种模态上的会不会有区别？在设计的时候是设计一个统一的还是两者有一定差异？
3. 在设计的时候有哪些坑要避免？注意的要点？
4. 我们现在输入的是单序列(T1)完整volume还是多序列(T1, T2, …)都要？RadGenome是单序列完整volume数据输入

### 随着token数量的减少，哪些属性会先收到影响/消失？

- **[Arxiv 26.08] ORCA: ORgan-Centroid Aggregation for Training-Free 3D CT Visual Token Compression**
    - 冻结三维 CT Encoder，控制压缩预算，探测疾病、大小、密度、位置、纹理
    - 在冻结 CT 特征上进行压缩，直接用轻量 readout 测量不同属性，并绘制预算曲线。其结果显示，**粗粒度疾病预测可能较稳定，而位置、大小等属性更能区分压缩方法**
    - **位置预测差，有时是位置没有以 readout 可使用的形式暴露出来，而不一定是视觉内容被删除了。** 加入空间编码后，结果可以明显改变。
- **[Arxiv 26.08] Cheap Probes Predict Expensive Training in 3D-CT Vision–Language Models**
    - 冻结特征上的廉价探测能预测昂贵下游训练的配置排名筛选

### Timeline [8 Weeks to CVPR Full Paper DDL]

| Week 0921-0927 | 讨论确定当前framework；搭建完整pipeline；熟悉数据集以及preprocessing；完成baseline方法的训练 | CVPR DDL |
| --- | --- | --- |
| Week 0928-1004 | 继续完善baseline结果；思考讨论projector的设计方向 |  |
| Week 1005-1011 | 对projector进行设计迭代优化，找到最有效的方案 |  |
| Week 1012-1018 |  |  |
| Week 1019-1025 |  |  |
| Week 1026-1101 | 确定最终方向，完善实验main table，消融实验，计算资源计算等等，完成写作 |  |
| Week 1102-1108 |  |  |
| Week 1109-1115 |  | 1110 AoE Registration |
|  |  | 1116 AoE Full Paper
1123 AoE Suppl |

### Related Works

- 训练一个更好的Projector v.s. 无需额外训练的插件方法
- Motivation的思考：比起使用一个即插即用的插件方法，我们为什么要训练一个Projector？

|  | 这类无需额外训练的插件 | 可训练压缩型 Projector |
| --- | --- | --- |
| 主要目标 | 加速已有模型，尽量维持原性能 | 学习适合目标任务的紧凑视觉表示 |
| 保留依据 | 注意力、相似度、空间变化等规则 | 报告监督，以及你设计的局部证据监督 |
| token 内容 | 原 token 的选择或规则聚合 | 可以学习新的聚合表示 |
| 数据与训练成本 | 较低 | 需要训练数据和优化成本 |
| 接入已有模型 | 通常方便 | 通常需要训练 Projector，可能还需适配 LLM |
| 主要风险 | 规则认为不重要，但临床上重要的信息被压缩 | 学到数据集捷径，或把语言先验误当成视觉证据 |
- 直接研究 Projector 的价值：
1. **从训练开始就让 LLM 适应紧凑表示。** 不必假定一个原本读取大量 token 的模型，在突然减少输入后仍能正常工作。
2. **可以在压缩输出上施加临床监督。** 要求它保存病灶位置、数量、属性，而不是仅优化某种显著性评分。
3. **可以学习“如何合并”。** 两个向量语义相近，不代表它们应被合并；如果分别来自左右两个病灶，合并可能破坏报告所需的信息。
- Limitation: Projector 不能可靠恢复 Encoder 已经丢失的信息
- CT/MRI相关的工作【CT报告生成为主】

| 工作 | 图像与任务 | 与你直接相关的内容 | 对创新定位的影响 |
| --- | --- | --- | --- |
| **M3D** | 3D 医学理解，含报告、VQA 等 | 3D 空间池化后接映射层 | “保留 3D 网格进行池化”已有基础方案。[论文](https://openreview.net/pdf?id=XQL4Pmf6m6) |
| **Exploring the Design Space of 3D MLLMs for CT Report Generation** | CT 报告生成 | 比较 MLP、Spatial Pooling Perceiver，并将 TokenPacker 扩展到 3D | “比较不同 Projector”以及“3D TokenPacker”已有研究。[MICCAI 2025](https://papers.miccai.org/miccai-2025/paper/2261_paper.pdf) |
| **HSENet** | CT 报告、VQA、检索 | Spatial Packer：局部 3D 区域先形成低分辨率表示，再通过 Voxel2Point Cross-Attention 聚合细节 | 是你设计局部 3D Query Projector 的直接相关工作。[论文 §3.3](https://arxiv.org/html/2506.09634v1) |
| **MedRegion-CT** | CT 报告生成 | 区域 SlowFast 表示、伪分割 mask 和病灶属性提示 | “全局＋局部＋解剖引导”不是空白；其收益也包含额外结构化信息。[论文](https://arxiv.org/html/2506.23102v3) |
| **CT-GRAPH** | 胸部 CT 报告生成 | 将器官、解剖系统、全局上下文组织成层次图 | “器官关系建模”已有明确路线。[论文](https://arxiv.org/abs/2508.05375) |
| **Ker-VLJEPA** | 胸部 CT 报告生成 | 用分区约束的 cross-attention 压缩成少量空间关联 token，结合分阶段训练 | “空间约束的可训练压缩”也已有工作。[论文](https://arxiv.org/abs/2603.23308) |
| **Photon** | 3D 医学 VQA，主要实验包括 CT | 指令条件的 token 调度、硬删除与替代梯度，训练和推理均压缩 | “学习式动态 token 保留”已有研究；其调度发生在 Decoder 处理中，不能直接归为 Projector。[论文](https://arxiv.org/html/2603.25155v1) |
| **ORCA** | 胸部、腹部 CT | 解剖引导的连通区域聚合；固定 Encoder 特征下做属性探测和生成评估 | “固定特征研究属性随预算丢失”已有非常接近的 CT 工作。[论文](https://arxiv.org/html/2608.00345v1) |
| **Resolution Meets Reduction** | 3D CT 报告生成 | 系统比较输入分辨率、压缩型 Projector 和 token 预算 | 是你现有 Pipeline 的合理基础，但不能把同类比较本身作为全部创新。[论文](https://arxiv.org/html/2608.08713v1) |
| 方法 | 插入位置 | 压缩的对象 | 原来的 Projector |
| MedPruner （2D切片） | 两处：Encoder 前；Encoder 后、Projector 前 | 先删除部分原始切片；再筛选、聚合保留切片的视觉特征 token | 保留 |
| CARVE （2D切片） | Encoder 后、Projector 前 | 带切片和空间坐标的视觉特征 token | 冻结保留 |
| MedARC | Projector 后、LLM 前 | 已映射到 LLM embedding 空间的视觉 token | 保留；其输出再被压缩 |
- 通用领域相关工作设计

| 类别 | 代表工作 | 核心机制 | 对你有什么启发 |
| --- | --- | --- | --- |
| **规则空间压缩** | DeCo | 在 patch 层面池化，避免 Projector 过早进行过强的语义抽象 | 简单池化必须作为强基线，复杂结构不一定更好。[论文](https://arxiv.org/abs/2405.20985) |
| **局部结构增强的 Projector** | Honeybee、TokenPacker | 卷积、局部/可变形注意力，或从高分辨率区域向低分辨率表示注入细节 | 是你的主要结构基线。[Honeybee](https://arxiv.org/html/2312.06742)、[TokenPacker](https://arxiv.org/abs/2407.02392) |
| **可学习 Query 重采样** | BLIP-2 Q-Former、Flamingo Perceiver Resampler | 用少量 Query 从大量视觉特征中读取信息 | 灵活，但需要检验 Query 是否忽视微小证据或混合多个位置。[BLIP-2](https://arxiv.org/abs/2301.12597)、[Flamingo](https://arxiv.org/abs/2204.14198) |
| **显著性筛选＋合并** | VisionZip | 保留主要视觉 token，并用紧凑表示补充上下文 | 可作为无需额外训练的代表基线。[论文](https://arxiv.org/abs/2412.04467) |
| **多样性选择** | DivPrune | 选择彼此差异较大的 token，减少重复信息 | 多样性可以减少冗余，但不直接等于临床价值。[论文](https://arxiv.org/abs/2503.02175) |
| **集合覆盖优化** | SCOPE、CoverPruner | 不只看单个 token 分数，还考虑保留集合能否代表被删除的信息 | “覆盖度＋重要性”已有系统研究，不能单独作为新意。[SCOPE](https://arxiv.org/abs/2510.24214)、[CoverPruner](https://arxiv.org/abs/2609.03158) |
| **Encoder 内合并** | ToMe | 在视觉 Transformer 中逐层合并相似 token | 能减少 Encoder 计算，但会改变你想固定的视觉特征。[论文](https://arxiv.org/abs/2210.09461) |
| **LLM 内动态压缩** | FastV、SparseVLM | 输入进入 LLM 后，根据注意力或文本相关性削减视觉 token | 已利用部分跨模态交互，但前面几层仍处理较长序列。[FastV](https://arxiv.org/abs/2403.06764)、[SparseVLM](https://arxiv.org/abs/2410.04417) |
| **预融合后强压缩** | LLaVA-Mini | 先把视觉信息融合进文本表示，再大幅减少显式视觉 token | 不能只数剩余视觉 token；信息可能转移到文本 token，预融合也有计算成本。[论文](https://arxiv.org/abs/2501.03895) |
| **多种压缩器的路由组合** | QMoP | 根据图像和问题选择不同压缩分支 | “池化＋Query＋筛选，再加 Router”已有直接相关工作。[论文](https://arxiv.org/html/2603.21232v1) |

### 0929 Discussion

- Last Time Pipeline的Encoder确认，LLM Decoder
- Nature BME ConceptCLIP 这篇工作当中，局部概念对齐
- Related Works 与调研可能的设计方向

[Possible Pipelines 0929](https://app.notion.com/p/Possible-Pipelines-0929-3ea4b5438f1d80989810fdf8e0cfc9be?pvs=21)

- 思考：与不针对Projector，而是即插即用的token压缩类插件的比较

- Vision Part进行finetune，3D DINO/JEPA/NeuroVFM
- Claim: 保证质量的同时保证效率;vision part和LLM有大量的预训练，projector上的处理比较少；projector分成几种类型，对于3D的医疗数据是怎么处理的；
- UMLS；Knowledge作为bridge来填补vision LLM的gap;轻量化
- 具体如何finetune exp
- baseline分几大类进行比较，不追求1:1复现