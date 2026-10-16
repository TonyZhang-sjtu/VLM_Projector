# BindFormer：单序列脑 MRI 的实例—属性绑定压缩方案与实现计划

日期：2026-10-13。代码核对基点：Git HEAD `464b946`。

状态（2026-10-15 更新）：**已实现首版 BindFormer3D：槽 + BraTS 概念库 + 共享支持域，训练目标仅报告 CE。** 几何统计、额外交互和辅助损失仍待实现。早先误用 Stage A-only 训练协议的正式 sweep 已按用户要求删除；当前四档改用与其他 baselines 相同的 `joint_lora` 正式 sweep，尚无已完成的 BindFormer test 结果。当前入口和协议见第 25 节；前文保留研究设计与历史审计。目录名沿用 `Pipeline_0929`。

2026-10-14 更新：第 19 节补充当前输入/标注缺口、tuple/binding 监督解释、本地 BioLORD 编码检查、UMLS 候选消歧结果和离线资产实施顺序。

2026-10-15 更新：第 20 节根据跨数据集迁移和标注成本的约束，提出以图像—报告配对为默认输入的训练候选；原 tuple/binding 作为可选监督分支，尚未进行效果验证或代码实现。

2026-10-15 后续更新：第 21 节记录两个知识 encoder 的实际下载、首版 CLS 偏好、68 条 UMLS 候选草案，以及重建头规模和多槽—句子匹配的设计细化。

当前确认配置：name + short definition → SapBERT 最终 CLS → L2 normalize；仅使用 `brats_train` 范围；只训练报告生成 CE；槽迭代 3 次。M 由压缩率 4/8/16/32 推导，按全局 25%、局部 75% 选择最近合法整数解，每槽四个角色。正式比较统一使用 projector warm-up 后 joint projector + visual LoRA 的 `joint_lora` 训练计划，LLM 冻结。第 20 节保留额外损失方案，第 23 节记录 v2 资产，第 24–25 节记录首版结构与实际实现。

## 1. 这个 pipeline 在解决什么问题

当前任务是：输入**一个完整的单序列 3D 脑 MRI**，由 NeuroVFM 提取 patch 特征，经 projector 转成 Llama-3.1-8B 可读的连续视觉 token，生成该序列的英文 Findings。

BindFormer 要验证的假设是：**在相同输出 token 预算下，先把视觉证据组织成组，再在组内读取不同属性，并监督属性的共同归属，能够减少压缩后的事实遗漏和属性错配。**

例如，图像包含“左额叶的点状异常”和“右顶叶的片状异常”。仅保留“左、右、额叶、顶叶、点状、片状”这个概念集合，还不能保证报告把它们组合正确。本方案试图保留的是“哪个证据组具有哪些属性”。

这里的实例槽是**学习得到的证据组**，不是已经验证的病灶实例或分割。一个槽可以对应局灶异常、区域性异常或弥漫改变；同一异常也可能被多个槽拆分。全局分支承担整体结构、正常所见和未被局部槽覆盖的上下文。

| 任务特点 | 对设计的要求 |
| --- | --- |
| 3D 特征多，小异常可能占很少 patch | 压缩前读取完整特征，单独检查小异常遗漏 |
| 同类异常可能重复出现 | 类别/CUI 不能代替患者内实例身份 |
| 位置、侧别、范围决定事实含义 | 使用方向正确的物理几何，不能只按外观聚合 |
| 信号解释依赖 MRI 序列 | 显式提供序列身份，保持单序列可见性边界 |
| 报告标注不完整 | 区分缺失、明确否定、不确定；未提及不作负标签 |
| encoder 与 LLM 冻结 | 在 projector 处干预，保留 LLM 对视觉输入的反向梯度 |

这些问题不是医学领域独有。研究重点是它们在脑 MRI、强压缩、弱实例监督和冻结模型条件下的组合。前提是需要的信息已存在于 NeuroVFM 特征中；projector 无法恢复 encoder 已经丢失的证据。

## 2. 当前仓库与正在进行的实验

### 2.1 数据与模型状态

活动监督版本为 `single_sequence_findings_coref_v2`；默认 `data.datasets: [brats]`。直接统计本地清单得到：

| split | 全部来源序列数 | 默认 BraTS 序列数 | 默认 BraTS 独立患者数 |
| --- | ---: | ---: | ---: |
| train | 2177 | 1960 | 484 |
| val | 299 | 268 | 67 |
| test | 622 | 560 | 140 |

全部来源包含 BraTS GLI/MEN/MET、ISLES22、WMH；序列数不是患者数。本次统计未借助 val/test 报告设计概念词表。

- 默认预处理：RPI 方向、`native_grid`、`minimal_pad`，完整视野，不删除背景 patch。
- NeuroVFM patch 为网络 DHW 顺序的 `[4,16,16]`，每 patch 展平为 1024 个数；本地 encoder 输出维度为 768。
- 当前 BraTS ratio 预算对应的标准网格为 `[39,15,15]`，输入 `N=8775`。
- 本地 Llama hidden size 为 4096；实现仍应读取 `decoder.hidden_size`，不能硬编码。
- `Sample` 保存 `finding`、`sequence`、患者/检查身份和审核标记；**没有实例 tuple、字段 mask、CUI 标签或概念库**。
- mask 路径存在于清单中，但训练 dataset 不读病灶 mask；mask 用于独立 QA。
- coreference v2 做过文本指代修订，不等于影像可见性或 tuple 归属已获临床核验，见 [指代修订记录](MODAL_FINDINGS_COREFERENCE_REVISION.md)。

### 2.2 文件结构与职责

| 目录/入口 | 当前职责 | 与 BindFormer 的关系 |
| --- | --- | --- |
| `configs/`、[src/config.py](../src/config.py) | 严格 YAML、参数校验、路径与配置身份 | 参数必须进入 schema/defaults |
| [src/data/manifest.py](../src/data/manifest.py)、`dataset.py`、`collate.py` | 患者隔离、目标版本校验、完整体积列表 batch | 新增独立事实标签 sidecar |
| [src/preprocessing/](../src/preprocessing/) | 方向、重采样、padding、patch、几何与缓存 | 已有 affine，可推导物理 patch 几何 |
| [src/models/encoders/neurovfm.py](../src/models/encoders/neurovfm.py) | 每体积特征/坐标/metadata；冻结/LoRA/full；特征缓存 | 主实验复用冻结特征 |
| [src/models/projectors/](../src/models/projectors/) | 11 种 projector、注册、预算与输出契约 | 新增 `bindformer3d` 及独立子模块 |
| [src/models/multimodal.py](../src/models/multimodal.py) | encoder → projector → decoder；报告 CE | 传序列身份，连接辅助输出/监督 |
| [src/models/decoders/llama.py](../src/models/decoders/llama.py) | 替换 `<image>`、答案 CE、冻结 LLM、生成 | 连续 token 接口可直接复用 |
| [src/tasks/single_sequence_findings.py](../src/tasks/single_sequence_findings.py) | 固定提示、序列名、英文 Findings 目标 | 保持相同 prompt，不输入辅助事实文本 |
| [src/training/](../src/training/) | optimizer、累积/DDP、调度、checkpoint、早停 | 增加独立归一化辅助 loss |
| [src/evaluation/](../src/evaluation/) | 文本/临床评分、患者 bootstrap、视觉对照 | 增加绑定与严格图像置换检查 |
| `scripts/baselines/`、[baseline_sweep.py](../src/cli/baseline_sweep.py) | 按比例独立训练，随后评测 best checkpoint | 当前固定 joint_lora，不直接作为冻结两端入口 |
| `tests/` | 几何、预算、梯度、loss、恢复与评测契约 | 扩展新增行为的测试 |
| `assets/`、`cache/`、`runs/`、`third_party/` | 数据/模型、缓存、实验产物、本地依赖 | 概念库与事实标签独立版本化 |

现有 projector：`mlp2x_gelu`、`avgpool3d`、`concat3d`、`convpool3d`、`mixer3d`、`local_packer3d`、`attention_pool`、`perceiver_nv`、`perceiver_flamingo`、`adaptive_merge3d`、`token_select3d`。

### 2.3 当前真实调用链

```mermaid
flowchart LR
    A["Bash / train CLI"] --> B["配置、数据选择、预算预检"]
    B --> C["Dataset + Preprocessor"]
    C --> D["PatchVolume"]
    D --> E["NeuroVFM"]
    E --> F["Projector"]
    F --> G["M × 4096"]
    G --> H["冻结 Llama + 固定提示"]
    H --> I["答案 CE / Findings"]
    I --> J["Trainer / checkpoint / evaluation"]
```

接口按单体积调用，变长 batch 用列表组织：

```python
Projector.forward(features, coords, metadata) -> ProjectedVolume
# features: [N, input_dim]
# coords: [N, 3]，完整规则网格的整数 DHW 索引
# ProjectedVolume: embeddings, input_tokens, output_tokens, metadata
```

`ProjectedVolume.validate()` 已检查形状、有限值和 `training.required_visual_tokens`。`StructuredProjector` 提供输入/预算检查及输出 metadata，适合作为父类。

### 2.4 代码与原设想的差距

下表为初次审计时的状态；当前已实现的序列传递、知识检索、槽路由与辅助张量接口以第 25 节为准。

| 问题 | 已有实现 | 本方案所需 |
| --- | --- | --- |
| 坐标 | 整数网格；encoder 自身有位置编码；metadata 保存 LPS affine | 物理中心、有效体积、方向保持的 PE |
| 序列 | `Sample.sequence` 用于 prompt，未传 projector | 仅传允许的序列 ID，不传整份 Sample/报告 |
| Query | CA+FFN；Flamingo 变体将 latent 加入 K/V | 竞争槽、角色 query、支持域 bias |
| 几何 | `physical_extent()` 提供 padded 视野尺度 | 不能代替逐 patch 物理坐标/有效体积 |
| 辅助输出 | embeddings 和可序列化 metadata | 独立的可反传辅助张量通道 |
| 训练目标 | 只有 `loss_sum`、`answer_tokens` | tuple/binding/concept 分别归一化 |
| 知识与标签 | 尚无相关模块 | 概念库、schema、部分标签、审核与版本 |
| 训练设置 | baseline 默认 joint_lora，后续解冻视觉 LoRA | 主实验 stage=a，LoRA 独立对照 |
| 评测 | BLEU、ROUGE、GREEN、RadGraph-XL、RaTEScore、BERTScore、CE | 实例绑定、小异常与图像依赖性评价 |

根 README 部分段落仍写“BERTScore 尚未实现”，但当前 `evaluate_checkpoint.py`、`clinical_scorers.py` 及测试已有实现；本文以代码和 [评价文档](EVALUATION_METRICS_AND_NEUROQA_PLAN.md) 为准。本轮未改旧 README。

## 3. 总体接口与结构

$$
X=E_{\mathrm{vis}}(V)\in\mathbb R^{N\times768},\qquad
Z=P_\theta(X,p,m,\mathcal C)\in\mathbb R^{M\times4096},\qquad
\hat y=D_\phi(Z,u).
\tag{1}
$$

| 符号 | 含义 |
| --- | --- |
| \(V,X,N\) | 单序列体积、全部视觉特征、输入 patch 数 |
| \(p_i,m\) | patch 物理中心、序列类型 |
| \(d,d_t\) | projector 内部宽度、文本向量宽度 |
| \(G,K,R\) | 全局 token 数、证据槽容量、每槽角色数 |
| \(\mathcal C,\mathcal C_r,E\) | 固定概念库、角色候选集合、缓存概念向量 |
| \(a_{ik},b_{kr}\) | patch 到槽软归属、槽 k 的角色 r 视觉 token |
| \(u,y,\hat y\) | 固定任务提示、参考 Findings、生成 Findings |

主实验冻结 encoder 与 LLM，仅训练 projector 和辅助头。LLM 参数 `requires_grad=False`，但训练前向不能包在 `no_grad()` 内；当前 decoder 已满足此要求。

$$
h_i=W_x\operatorname{LN}(x_i)+\operatorname{PE}(p_i)+E_{\mathrm{seq}}(m),
\qquad H=[h_i]_{i=1}^N\in\mathbb R^{N\times d},
\tag{2}
$$

$$
M=G+KR.
\tag{3}
$$

初始角色为 finding、location/laterality、signal、morphology/extent。一个角色可对应多个监督字段；R 不等于字段数。背景槽只参与内部路由，不额外输出 token。K 是容量，不是病灶数。

```mermaid
flowchart TD
    X["NeuroVFM: N × 768"] --> H["适配 + 物理 PE + 序列 embedding"]
    H --> S["竞争实例槽：K + 1，含背景"]
    S --> A["归属 A + 实例表示 S"]
    A --> GEO["可选几何统计"]
    A --> C["可选角色概念检索"]
    A --> Q["实例 × 角色 query"]
    GEO --> Q
    C --> Q
    Q --> READ["视觉 CA：内容分数 + beta log A"]
    H --> READ
    A --> READ
    H --> GLOBAL["G 个全局 queries"]
    READ --> MIX["可选包内 SA / 读取全局"]
    GLOBAL --> MIX
    MIX --> OUT["输出 MLP"]
    GLOBAL --> OUT
    OUT --> Z["G + K×R 个连续 token"]
    Z --> LLM["冻结 Llama → Findings"]
    MIX --> AUX["训练辅助头：tuple / binding"]
```

## 4. 实例分组：先建立共同证据来源

初始化可区分的 K 个可学习槽及背景槽 \(s_0\)，不预设疾病或位置。迭代 t：

$$
\ell_{ik}^{(t)}=\frac{(W_s s_k^{(t)})^\top(W_hh_i)}{\sqrt d},
\qquad
a_{ik}^{(t)}=\operatorname{softmax}_{k=0,\ldots,K}(\ell_{ik}^{(t)}).
\tag{4}
$$

每个 patch 在槽之间竞争，\(\sum_k a_{ik}=1\)。随后沿 patch 轴归一化并汇聚：

$$
n_k^{(t)}=\sum_i a_{ik}^{(t)},\quad
\bar a_{ik}^{(t)}=\frac{a_{ik}^{(t)}}{\epsilon+n_k^{(t)}},\quad
\bar h_k^{(t)}=\sum_i\bar a_{ik}^{(t)}h_i,
\tag{5}
$$

$$
s_k^{(t+1)}=F_{\mathrm{slot}}(s_k^{(t)},\bar h_k^{(t)}).
\tag{6}
$$

更新用 GRU + 残差 MLP。最后一次更新后重新计算 A，确保支持域与最终 S 对齐。建议起始迭代 3 次，作为待验证配置。

A 存 `[N,K+1]`，明确两次归一化的轴。完全无有效覆盖的 patch 不参与汇聚/读取；有效视野由几何定义，不由病灶 mask 或强度阈值定义。部分 padding patch 是否按有效比例加权槽汇聚，是额外变体，不能无记录地改变式（5）。

分组在槽之间 softmax，属性读取在 patch 之间 softmax。竞争不保证覆盖所有异常；背景槽不等价于正常组织。

## 5. 物理几何与支持范围

对非背景槽计算：

$$
\mu_k=\frac{\sum_i a_{ik}p_i}{\epsilon+\sum_i a_{ik}},\qquad
\Sigma_k=\frac{\sum_i a_{ik}(p_i-\mu_k)(p_i-\mu_k)^\top}
{\epsilon+\sum_i a_{ik}},\qquad
\nu_k=\sum_i a_{ik}\Delta V_i.
\tag{7}
$$

求和限定在有效 patch；部分 padding 的 \(\Delta V_i\) 只计真实覆盖体积。中心可采用有效覆盖部分的中心，需与完整 patch 中心的选择一起记录。若改用体积加权 \(\mu,\Sigma\)，作为单独变体。

使用中心 3 维、对称协方差 6 个独立分量、`log1p(soft_volume)` 等标准化量，通过小 MLP 得到 \(g_{kr}\in\mathbb R^d\)。初版仅位置和形态/范围角色使用，其余角色取 0。

### 从现有 metadata 恢复物理坐标

`coords` 必须继续作为 encoder/网格的整数 DHW 输入，另外构造 `PhysicalPatchGeometry`：

1. 根据 patch size 与 index 得到 padded 数组内 voxel 中心：`index * patch_size + (patch_size - 1) / 2`。
2. 逆变换 `array_zyx_to_dhw` 恢复 ZYX，再转 XYZ，不能假设 D 总是物理 Z。
3. 减去 `pad_before_xyz`，得到 `bounded_before_pad` 图像内 index。
4. 使用 `bounded_before_pad.affine_xyz_to_lps` 转成 LPS mm；裁剪偏移已在 affine 中，不能重复加 `crop_start_xyz`。
5. patch 覆盖范围与未 padding 图像范围求交，计算有效 voxel 数和体积；单 voxel 体积用 affine 线性部分行列式绝对值。
6. PE 使用方向保持的标准化坐标，同时保存未标准化 mm 统计；数据统计归一化只在 train 拟合。

LPS 方向确定不等于配准到脑图谱；LPS x=0 或视野中心不自动等于解剖中线。建议方向保持的视野中心化坐标配合尺度摘要，避免把任意扫描 origin 当作解剖标签。左右侧需用 header、合成翻转和真实图像核对。

第一轮可从已有缓存 metadata 纯函数推导几何，不必仅为新增派生坐标重算 encoder 特征。若修改预处理数值/缓存内容，更新版本身份。

这些量是软支持域摘要，不能直接作为真实病灶体积、直径或分割输出。

## 6. UMLS 概念记忆与角色 query

本版本使用名称、同义词、定义、语义类型，不做图网络推理。UMLS 提供概念资源，不提供当前患者实例或坐标。[NLM UMLS](https://www.nlm.nih.gov/research/umls/index.html)、[Metathesaurus 手册](https://www.ncbi.nlm.nih.gov/books/NBK9684/)。

“UMLS 解释的一段话作为 class token”具体化为：**每个概念有固定文本向量；图像槽按角色软检索这些向量，用于构造读取 query。** 不是每个疾病固定占一个输出 token，也不是把定义追加给 LLM。

### 6.1 离线构建

从训练报告术语与预先确定的公开医学词表建立候选，链接并审核 CUI；无准确 CUI 的修饰词保留本地 ID。四种角色由项目 schema 定义，不是 UMLS 自带分组。

| 角色 | 示例候选范围 |
| --- | --- |
| finding | 可观察异常、占位性改变；避免以诊断代替视觉事实 |
| location/laterality | 解剖部位，侧别单列子字段 |
| signal | 带序列语境的高/低/等信号及混合描述 |
| morphology/extent | 点状、片状、边界、局灶/弥漫等 |

保存 concept ID/CUI、标准名、同义词、定义来源/版本、角色、审核状态、文本模板、词表/向量哈希和文本模型版本，保留 unknown/other。缺失标签和明确标注 unknown 不是同一状态。

$$
e_c=\operatorname{Normalize}
\left(E_{\mathrm{text}}(\mathrm{name}_c,\mathrm{description}_c)\right)
\in\mathbb R^{d_t}.
\tag{8}
$$

文本 encoder 冻结、离线运行。SapBERT 可作为候选，但其主要验证任务是实体表示/链接，不自动保证长定义适合视觉 query；需要固定 pooling、模板、最大长度，并比较 name-only 与 name+definition。[SapBERT 原论文](https://aclanthology.org/2021.naacl-main.334/)。

### 6.2 在线查询

$$
u_{krc}=\frac{(W_c s_k)^\top(T_r e_c)}{\sqrt d},\qquad
\eta_{krc}=\operatorname{softmax}_{c\in\mathcal C_r}(u_{krc}),\qquad
e_{kr}=\sum_{c\in\mathcal C_r}\eta_{krc}e_c,
\tag{9}
$$

$$
q_{kr}=W_qs_k+U_re_{kr}+v_r+g_{kr}.
\tag{10}
$$

\(W_c,W_q\in\mathbb R^{d\times d}\)，\(T_r,U_r\in\mathbb R^{d\times d_t}\)，\(v_r\) 是角色 embedding。知识检索的 value 来自概念库，视觉读取的 value 来自图像，二者不能混淆。

\(\eta\) 是语义混合，不是诊断概率；一个槽可涉及多个 CUI，一个 CUI 可对应多个患者内实例。视觉与文本空间需要学习映射，不天然对齐。

训练与推理使用同一固定候选库，不读取当前参考报告筛候选；val/test 新术语按冻结 schema 处理。知识会改变读取偏好，“只进 query”不能保证无幻觉。

## 7. 属性读取：共享支持域，角色分别取证

$$
t_{kri}=\frac{(W_Qq_{kr})^\top(W_Kh_i)}{\sqrt d},\qquad
\alpha_{kri}=\operatorname{softmax}_i
\left[t_{kri}+\beta_r\log(\epsilon+a_{ik})\right],
\tag{11}
$$

$$
b_{kr}=\sum_i\alpha_{kri}W_Vh_i\in\mathbb R^d.
\tag{12}
$$

等价地，

$$
\alpha_{kri}=
\frac{\exp(t_{kri})(\epsilon+a_{ik})^{\beta_r}}
{\sum_{j=1}^{N}\exp(t_{krj})(\epsilon+a_{jk})^{\beta_r}}.
\tag{13}
$$

内容分数决定细节，A 使同槽各角色具有共同来源，不要求 attention 完全相同。beta=0 完全移除支持域 bias；query 仍依赖槽，因此不是移除整个实例结构。

建议先比较固定 beta=0/1，再研究 `softplus(raw_beta)`。`support_mode=none` 要显式生成零 bias，不能靠 softplus 近似零。过强约束会放大错误路由，病灶核心、水肿与占位效应也不必严格共定位。

**不能直接复用现有 QueryBlock 作为首次属性读取。** [common.py](../src/models/projectors/common.py) 使用 `queries + attended + FFN`。当 query 含知识向量，残差会让知识直接进入输出，偏离式（12）的视觉 value 路径。新增 `SupportBiasedReader`，首次输出以视觉 attended value 为基础，可接 LN/FFN；后续残差只能来自已经读取的证据 token。全局分支可继续用普通 QueryBlock。

多头版本按 \(\sqrt{d_h}\) 缩放，支持 bias 广播到各头，valid mask 同时生效。log/softmax/几何矩尽量 FP32；A 保持可导，detach 只能作为明确消融。NeuroVFM 已有上下文混合，attention 不能直接解释成分割或因果证据。

## 8. 全局补充、交互与输出

$$
B_{\mathrm g}=F_{\mathrm{global}}(Q_{\mathrm g},H)
\in\mathbb R^{G\times d}.
\tag{14}
$$

每槽角色构成 \(B_k\in\mathbb R^{R\times d}\)，可选：

$$
B'_k=B_k+\operatorname{SA}(\operatorname{LN}(B_k)),\qquad
\widetilde B_k=B'_k+
\zeta_k\operatorname{CA}(\operatorname{LN}(B'_k),B_{\mathrm g}),
\quad \zeta_k\in[0,1].
\tag{15}
$$

包内 SA 只作用于同槽 R 个 token；局部读全局的 Q 来自本槽，K/V 来自全局 token。初版均关闭，不默认增加跨槽广泛 SA。

$$
Z=\operatorname{Concat}_{\mathrm{token}}
\left[F_{\mathrm{out}}(B_{\mathrm g}),
F_{\mathrm{out}}(\widetilde B_1),\ldots,
F_{\mathrm{out}}(\widetilde B_K)\right]
\in\mathbb R^{M\times4096}.
\tag{16}
$$

顺序固定为 global → slot-major → role-major，不按报告重排。保留所有槽的 token，不按预测病灶数改变 M；不增加预算外概念 token、事实文本或头预测 argmax。

辅助头接在交互之后的 \(\widetilde B\)，即输出 MLP 之前。它不自动证明信息保留在 Z 中或被 LLM 使用；需配合 Z 的离线 probe 和生成 tuple 评价。

## 9. 完整事实组与绑定监督

本节保留最初的结构化监督方案。它要求可靠事实组资产，现作为可选分支；不依赖新增逐实例标签的 X 特征重建和报告片段对齐补充方案见第 20 节。记录这些候选目标不代表首轮实现必须同时启用全部损失。

### 9.1 共同 Hungarian 匹配

离线解析部分标注集合 \(\mathcal T=\{t_\ell\}_{\ell=1}^L\)，可靠字段 mask 为 \(\omega_{\ell j}\in\{0,1\}\)。字段经 \(r(j)\) 映射到角色，预测为 \(p_{kj}=\mathcal H_j(\widetilde b_{k,r(j)})\)：

$$
C_{k\ell}=\sum_j\omega_{\ell j}\,\ell_j(p_{kj},t_{\ell j}),\qquad
\pi^*=\arg\min_{\pi\ \mathrm{injective}}
\sum_{\ell=1}^L C_{\pi(\ell),\ell}.
\tag{17}
$$

$$
\Omega=\max(1,\sum_{\ell,j}\omega_{\ell j}),\qquad
\mathcal L_{\mathrm{tuple}}=
\frac1\Omega\sum_{\ell=1}^L C_{\pi^*(\ell),\ell}.
\tag{18}
$$

同一事实组的所有字段共用一次匹配。可在 detached CPU cost 上求离散索引，再从原始可导 cost/logits gather；不能逐字段分别匹配。

单分类用 CE，多标签用 BCE。多标签还需标签维度 observed mask：已知某项阳性不意味着其余项阴性。先对每字段已观测标签归一化，再汇总成本；评分 S 使用相同约定。

未匹配槽不直接作“无病灶”负标签。无可靠字段的 tuple 离线排除；L=0 时返回安全反传的零；L>K 必须预检报错并调整容量/组级协议，不能截断事实。

全局正常结构、全局否定通常不适合局部实例支持，首版由 RG/global 分支处理；局部事实允许明确否定和不确定。不能把所有否定句都强制建成病灶槽。

### 9.2 绑定负例

交换两个归属明确的事实组的部分属性，保持概念集合不变：

$$
S(B,\mathcal T)=
\max_{\pi\ \mathrm{injective}}
\frac1\Omega\sum_{\ell,j}
\omega_{\ell j}\log p_{\pi(\ell),j}(t_{\ell j}),
\tag{19}
$$

$$
\mathcal L_{\mathrm{bind}}=
\max\{0,\gamma-S(B,\mathcal T)+S(B,\widetilde{\mathcal T})\}.
\tag{20}
$$

多标签用 masked Bernoulli log-likelihood。正例与负例分别求最佳共同匹配，不能直接把正例索引当作负例最佳索引。

只交换 tuple 顺序不是负例；属性相同、归属不清、交换后仍可能正确时跳过。首版每个 eligible 样本按固定规则取一个交换负例，保存规则版本/随机种子和跳过原因；无负例时 loss=0。

不必额外增加联合 MLP，但此目标不保证排除退化解：K 很大时不同槽可能同时表示正误组合；缺少区分字段时两个集合可能不可辨。需记录负例覆盖率、正负分数差、重复槽及生成绑定错误，不能只看训练 loss。

### 9.3 总目标

$$
\mathcal L_{\mathrm{RG}}=
-\frac1{T_y}\sum_{\tau=1}^{T_y}
\log p_\phi(y_\tau\mid y_{<\tau},u,Z),\qquad
\mathcal L=\mathcal L_{\mathrm{RG}}
+\lambda_t\mathcal L_{\mathrm{tuple}}
+\lambda_b\mathcal L_{\mathrm{bind}}.
\tag{21}
$$

代码的 RG 包含答案结束符，按累积窗口的全局答案 token 数归一化。可选概念监督只用于单一、可靠概念标签：

$$
\mathcal L_{\mathrm{concept}}=
-\frac1{\max(1,\sum_{\ell,r}\chi_{\ell r})}
\sum_{\ell,r}\chi_{\ell r}
\log\eta_{\pi^*(\ell),r,c_{\ell r}}.
\tag{22}
$$

以 \(\lambda_c\) 加入总损失，主版本先关闭；多标签角色不能直接套用。

### 9.4 Trainer 的归一化契约

当前 Trainer 用 `loss_sum * world_size / global_answer_tokens`。**不能把辅助损失加进同一个 loss_sum 后一起除以文本长度。**

保留 `loss_sum` 为纯 RG 分子，另返回每样本按式（18）等归一化后的辅助 loss 之和。W 为 world size，T/B 为整个累积窗口的全局答案 token 数/实际样本数；每个 microbatch 反传：

$$
\frac WT\,\mathrm{RGSum}
+\frac WB\left(
\lambda_t\mathrm{TupleSampleSum}
+\lambda_b\mathrm{BindSampleSum}
+\lambda_c\mathrm{ConceptSampleSum}\right).
$$

无标签/负例样本的对应项为零；这一约定保持单样本目标且不随 microbatch 划分变化。如果改为仅 eligible 样本平均，要作为不同配置，因为有效 loss 权重会改变。

T/B 在 window 开始前计数并 all-reduce。日志分别记录 RG、辅助项、字段数、eligible 数。主协议仍以相同完整 val answer CE 选择 best checkpoint，绑定指标独立记录。

## 10. 模块化代码结构与接入点

以下路径为**计划新增**：

```text
src/models/projectors/bindformer/
  __init__.py
  projector.py          # BindFormer3D，流水线与输出
  types.py              # 内部张量类型/辅助输出
  input_adapter.py      # LN、投影、PE、序列 embedding
  physical_geometry.py  # metadata → 物理中心/有效体积
  slots.py              # CompetitiveSlotRouter
  geometry_encoder.py   # 支持域矩 → 几何向量
  concept_memory.py     # E buffer、角色候选、软检索
  role_queries.py       # 实例 + 角色 + 可选语义/几何
  reader.py             # 支持域 bias CA，视觉 value 输出
  interactions.py       # 包内 SA、局部读全局及 gate
src/models/projectors/fact_heads.py   # 其他 query baseline 可复用
src/data/facts.py                    # sidecar 读取/schema 校验
src/training/fact_losses.py          # 匹配、tuple/binding/concept
src/evaluation/binding.py            # tuple 与绑定诊断
scripts/build_fact_annotations.py    # 离线抽取候选/审核导出
scripts/build_concept_bank.py        # UMLS 子集 → 固定概念向量
configs/projector/bindformer3d.yaml
configs/experiments/bindformer_*.yaml
scripts/baselines/run_bindformer3d.sh
```

| 模块 | 输入 → 输出 | 开关 |
| --- | --- | --- |
| 输入适配 | X、几何、sequence → H [N,d] | position、sequence condition |
| 实例路由 | H → S [K,d]、A [N,K+1] | iterations、background slot |
| 几何编码 | A、p、体积 → [K,R,d] | geometry enabled、role mask |
| 概念记忆 | S、E → eta、[K,R,d_t] | none/name/name_definition/control |
| 角色查询 | S、语义、角色、几何 → [K,R,d] | role schema |
| 属性读取 | Q、H、A → [K,R,d] | support none/shared、beta |
| 全局分支 | H → [G,d] | G、depth |
| 交互 | B、Bg → Btilde | within_slot_sa、local_reads_global |
| 输出头 | [M,d] → [M,decoder.hidden_size] | 复用 mapping_head |
| 辅助头/损失 | Btilde、字段 → logits/loss | lambda_t/b/c |

### 10.1 公共文件修改表

| 文件 | 计划改动 |
| --- | --- |
| `projectors/registry.py`、`specs.py` | 注册 bindformer3d、默认值、严格校验及互斥组合 |
| `projectors/base.py` | ProjectedVolume 增加默认 None 的 `auxiliary` 字段，兼容旧四参数构造 |
| `projectors/common.py` | finish 可传辅助输出，metadata 仍只存可序列化摘要 |
| `multimodal.py` | projector context 加 sequence，forward 组装各 loss，保留生成路径 |
| `config.py`、`configs/base.yaml` | 资产路径、loss 权重、特性校验；旧配置默认关闭新增功能 |
| `data/dataset.py` | 按 sample ID 读取 sidecar，item 新增 facts，不改变旧 Sample 必填字段 |
| `training/trainer.py`、`loss.py` | 各 loss 归一化、日志与 val 辅助诊断 |
| `training/checkpoint.py` | 纳入标签/schema/概念向量/抽取器内容哈希及 resume 校验 |
| `training/stages.py` | 辅助头无标签时的 DDP 参数使用处理 |
| `projectors/budget.py` | 后续增加 G/K/R 分解的 ratio 契约 |
| 实验 runner | 显式支持冻结 Stage A、固定 M，不套用当前强制 joint_lora sweep |
| `evaluation/visual_controls.py` | 新增同 cohort+sequence、跨患者 donor 的置换版本 |

**辅助张量不能塞 metadata。** A、eta、Btilde/logits 经 auxiliary 保持计算图；metadata 只存预算、标量摘要和资产 hash。完整 attention 图按需导出 detached CPU 文件，默认不随每步 JSON 序列化。

辅助头注册在 `model.projector` 子模块下，使现有 optimizer projector 参数组与 checkpoint state_dict 覆盖它们；无参数的 loss 放 training 模块。所有 layer/slot/head 在构造时创建，不能在 forward 动态新增参数。

Stage A DDP 当前默认不启用 find_unused_parameters。启用头但 batch 没有标签时，需从该头输出构造图连通零损失，或为新分支显式启用并验证 unused 检测；关闭模块时不实例化无用参数。现有“所有 projector 参数均有梯度”的测试需要按模块启用条件扩展。

概念 E 为冻结 buffer；精简 E 可随 checkpoint 保存，同时保存 ID 顺序/来源 manifest。加载不能只比路径，同路径资产内容变化也要拒绝 resume。正式推理不需要 UMLS API 或文本 encoder。

序列传递建议为 project_volumes 增加可选 sequences/context 参数，由 forward/generate 从 Sample.sequence 提取；逐体积构造新的 metadata/context 副本，避免原地修改共享缓存。旧 projector 忽略新增信息，启用 sequence_condition 的 BindFormer 则对缺失/未知序列明确报错。不得把 finding、mask 路径或 tuple 通过该通道传给 projector。

### 10.2 建议配置草案

以下示例**不是当前已经支持的配置**，需要实现 schema 后才能运行。它保留为纯结构、纯 RG 对照草案；当前首版建议改用第 24 节的知识开启、CE-only 配置，不能将本节示例当成最终运行配置。

```yaml
projector:
  name: bindformer3d
  hidden_size: 4096
  target_compression_ratio: null
  params:
    num_tokens: 256
    global_tokens: 32
    num_slots: 56
    roles: [finding, location_laterality, signal, morphology_extent]
    inner_dim: 768
    num_heads: 8
    slot_iterations: 3
    background_slot: true
    position_mode: physical_fourier
    sequence_condition: true
    geometry_enabled: false
    support_mode: shared
    support_beta: [1.0, 1.0, 1.0, 1.0]
    knowledge_mode: none
    concept_bank: null
    within_slot_sa: false
    local_reads_global: false
    global_depth: 1
    head_type: mlp
    dropout: 0.0
training:
  stage: a
  required_visual_tokens: 256
  encoder_lr: 0.0
  feature_cache: true
  auxiliary:
    tuple_weight: 0.0
    binding_weight: 0.0
    concept_weight: 0.0
data:
  facts_path: null
  fact_schema_path: null
```

校验 M=G+KR、角色顺序、d 可被 heads 整除、beta 非负；启用知识要求合法 E/候选/hash，启用辅助损失要求 schema/标签；concept loss 依赖可靠 matching 和单概念 mask；local_reads_global 要求 G>0。未知参数拒绝。

## 11. 事实标签的数据工程

这是主要依赖，不能把 Findings 字符串当作已有实例监督。

1. 从活动 train Findings 建术语候选/抽取规则，保存原文 hash。
2. 固定 finding/location/laterality/signal/morphology/extent/assertion schema，注明字段类别与角色。
3. 离线抽取候选，保存证据 span、归属可靠性、缺失 mask、审核状态、序列和工具版本。
4. 优先审核多异常、同类重复、侧别不同和范围不同样本；不可分实例的多发/弥漫描述保留组级 tuple。
5. 冻结 schema/词表，再制作 val/test 评价标签；它们不用于训练或补充候选。
6. 统计 L 分布、字段覆盖/缺失率、可交换负例比例，据此选 K 和主评价子集。

建议目录（计划）：

```text
assets/annotations/bindformer_facts_v1/
  schema.json
  train.jsonl
  val.jsonl
  test.jsonl
  extraction_manifest.json
  review_queue.jsonl
assets/knowledge/bindformer_brain_v1/
  concepts.jsonl
  embeddings.pt
  manifest.json
```

单条记录含 sample_id、target_version、report_sha256、schema_version、tuples、字段 observed/reliability、evidence_spans、review_status、extraction_version；多标签还保存标签维度 observed mask。

不按队列名猜诊断，不从其他序列/历史检查补属性；保留原审核标记。抽取器只生成候选，binding 负例尤其需要可靠归属。无可靠 tuple 样本仍可训练 RG。

## 12. 固定 M 与固定压缩比例

现有协议 `method_native_ratio_v1` 要求一个 run 内输入几何一致；query 取 ceil(N/ratio)，网格取最近网格，Concat 用固定分块。调用现有函数，以 N=8775 得到：

| target N/M | Perceiver/AttentionPool | LocalPacker/AvgPool | Concat |
| --- | ---: | ---: | ---: |
| 4 | 2194 | 2197 | 2250 |
| 8 | 1097 | 1100 | 1125 |
| 16 | 549 | 550 | 525 |
| 32 | 275 | 275 | 280 |

同 ratio 不是同 M，旧结果不能直接称为严格固定预算消融。

### 12.1 固定 M 对照（保留的可选模式）

早期草案采用 M=256、G=32、K=56、R=4。当前主实验已改为第 12.2 节的压缩率模式和全局占比 0.25；固定 M 仍可用于严格预算对照。采用当前占比时 M=256 得到 G=64、K=48、R=4。改变 M 同时改变槽容量，应另做固定 M 的预算分配实验。

设置 num_tokens=M、required_visual_tokens=M，断言 M=G+KR，ratio=null。AttentionPool/Perceiver 可直接配置同 M；网格方法选可实现的 exact grid 并检查，不填零/裁剪凑数。未压缩 MLP 为 N-token 参考，不是同预算方法。

### 12.2 当前已实现的 ratio 模式

先取 M=ceil(N/rho)，设全局比例 alpha=0.25，在 K>=1、G=M-KR>=1 的整数解中最小化 |G/M-alpha|：

$$
K\approx\operatorname{round}\left(\frac{(1-\alpha)M}{R}\right),\qquad G=M-KR.
$$

实现检查最近的合法整数候选；距离相同时选 G 较大者，使用精确分数避免浮点误差改变平局规则。M=275、alpha=0.25、R=4 时 K=52、G=67。记录 M/G/K/R 和 `bindformer_nearest_fraction_v1`，预算不足一个角色包时直接拒绝，不裁掉角色凑数。alpha=0 的可选预算分配允许 G=0；若 M 不能被 R 整除，最近合法解仍有余数个全局 token，日志记录实际比例。

预算在构造前解析，forward 不改变参数数量；跨几何动态 ratio 是后续扩展，不能绕过当前预检。

去掉 global 也须控制 M：M=256、R=4 可改为 G=0、K=64，但这同时改变 K，属于预算重分配。也可保留 G 个位置用声明的替代分支填充作对照；不能宣称删除 G 后 M/K/R 都不变。

## 13. 实现顺序与交付

本节保留完整方案的早期阶段划分。UMLS v2 已就绪，且用户已确定首版仅用 CE，当前优先顺序以第 24 节为准：最小槽结构与知识检索一起接入，其他损失后置。

| 阶段 | 交付 | 进入下一阶段的依据 |
| --- | --- | --- |
| P0：协议/数据审计 | 固定 Stage A/M、train schema、L/字段/负例统计、几何约定；申请 UMLS | 预算与标签可支持研究问题 |
| P1：最小结构 | 新 projector、PE/sequence、槽、角色读取、global、输出；知识/几何统计/交互/aux 关闭 | shape、梯度、预算、恢复、小模型连通 |
| P2：支持域比较 | beta=0 对比 beta>0，纯 RG，固定预算参考 | 能观察槽与生成差异，不以 attention 外观代替结论 |
| P3：tuple | sidecar、共同 Hungarian、归一化、无标签/DDP/恢复 | 辅助目标与监督对照成立 |
| P4：binding | 审核交换负例、正负独立匹配、覆盖率/score gap | 相对 tuple-only 有可评价增量 |
| P5：UMLS | 固定子集、E、软检索、名称/描述/随机知识对照 | 区分结构、词汇与医学语义 |
| P6：额外组件 | 几何统计、包内 SA、读 global、concept loss，逐项开启 | 每项有独立对照 |
| P7：扩展 | 多 M、ratio、冻结 vs LoRA、重复种子 | 核心机制已有检验 |

早期建议先落地 P1/P2、不等待 UMLS；现在知识库已完成，首版可以同时接入知识。事实审核不阻塞 CE-only 训练，P3/P4 不再是加入知识的前置条件。

复用 stage_a 数据/缓存/梯度逻辑，统一 LR/步数/早停并记录。当前 baseline Bash → sweep 的 stage 为保留字段，不能普通 --set 改为 a；应使用通用 train 入口或新增固定 M/Stage A runner。

## 14. 消融与公平对照

统一 encoder/LLM、输入网格、目标版本、split/prompt、M、训练预算及选优规则。

| 对照 | 回答的问题 |
| --- | --- |
| query resampler，RG-only | 同 M 的无绑定结构参考 |
| 同 query baseline + tuple，再 + binding | 收益是否来自额外监督 |
| 槽+角色，beta=0 vs beta>0 | 共享支持域的独立作用 |
| 固定结构：RG / RG+tuple / RG+tuple+binding | 各监督项增量 |
| learned role / 名称 E / 名称+定义 E | 结构、词汇与描述语义的区别 |
| 同尺寸随机单位向量 E / 名称与名称+定义 E | 固定向量容量与医学文本表示的区别 |
| geometry off/on | 范围矩是否补足均值 |
| 包内 SA off/on；global read off/on | 协调与绑定的取舍 |
| global 分配变化 / 无 global | 全局信息与容量分配 |
| 有/无背景槽 | 背景竞争作用 |
| 同 M 的 G/K/R、不同 M、参数量/延迟 | 容量与效率 |
| 适配 Slot Attention/SysBinder | 是否超出成熟结构迁移 |
| 冻结 encoder vs joint_lora | 是否依赖视觉适配 |

监督对照可将 baseline 固定 M 个 query 输出按同样 G+K×R 分组，接相同 FactHeads；它得到输出组织和监督，但不增加支持域路由。应称“监督匹配的 query baseline”，不是原始未改模型。

当前 Perceiver 不是完整 BLIP-2 Q-Former；若称 Q-Former 对照，需单独适配 query SA+visual CA，说明预训练权重。报告参数量、宽度、depth、FFN，不只固定 M。

仅将候选索引与对应向量、掩码一起重排，不会改变软检索结果，不能作为知识破坏消融。在没有额外概念标签/身份约束的 CE-only 设置中，同一允许候选集合内部的向量置换也可能只是重排求和项；首轮采用知识关闭、随机固定向量和 name-only 等明确改变计算或语义几何的对照。

当前先做第 24 节的 CE-only 结构与知识对照，之后逐项加入几何、交互和辅助监督；不必第一轮遍历所有组合。

## 15. 评估与验收

保留报告评分和患者 bootstrap。建立审核后的多事实子集，分别评价概念召回、字段正确率、完整 tuple、明确归属的属性错配率、否定/不确定错误。评价抽取器需独立审核，避免将其错误全算给生成器。

头预测与生成报告分开评价；辅助头变好不等于 LLM 使用了结构。分层包含同概念多实例、侧别、局灶/弥漫、序列、小异常，报告有效样本数与置信区间。

现有 shuffle 只按 sequence，不强制同 cohort 或跨患者；新增同 cohort+sequence、不同患者 donor，保存 donor ID，无法构造的组明确记录。保留 text-only/投影后 zero；新增 projector 前 X 置零/知识错配诊断，注明仍可能保留坐标/序列信息且可能分布外。

| 验收 | 必须验证 |
| --- | --- |
| 几何 | 置换/翻转、非零 origin、padding/部分 patch、体积、左右方向；整数 coords 不变 |
| 路由 | valid patch 槽归属和为 1、最终 A/S 一致、背景无输出、近空槽数值有限 |
| Reader | beta=0 等价无 bias；人为支持域变化改变读取；无原始知识 query 残差 |
| 预算 | M=G+KR、固定顺序、非法预算/L>K 报错 |
| 匹配 | tuple 顺序不变性、共同与逐字段匹配区别、部分/多标签 mask、负例有效性 |
| 梯度 | RG 到 projector/路由，LLM/冻结 encoder 无梯度；aux 被 optimizer/checkpoint 覆盖 |
| 累积/DDP | 不等报告长度、无标签/负例、不同 microbatch 下 loss/梯度一致 |
| 恢复 | schema/词表/E/标签改变拒绝 resume；重载 eval 一致 |
| 集成 | CPU 小模型 → 真实短步 → 小样本过拟合/图像控制 → 完整实验 |

当前 pyproject 未显式声明 SciPy；若用 `scipy.optimize.linear_sum_assignment`，纳入依赖/环境锁，不依赖评分工具的偶然间接安装。

路由约 O(TNKd)，属性读取 O(KRNd)，global O(GNd)，还有投影/FFN/知识检索成本。默认不保存所有层/头 attention；query 分块仍需对完整 N softmax，沿 N 分块要实现正确流式归一化。deterministic 模式可能走 math attention，必须测真实显存。

分别记录 encoder/projector/LLM prefill/生成耗时、端到端延迟、显存、训练步时间和 N/M；缓存特征训练与在线推理分开计时。工程 smoke 不构成收敛或临床有效证明。

## 16. 相关工作与贡献边界

以下为方法定位，不是同设置复现实验。2026 年工作按所引 arXiv 版本讨论，不推定会议录用。部分条目只核对摘要/公开方法描述，正式论文写作前仍需逐项阅读全文。

| 工作与来源 | 相关贡献 | 本方案应验证的区别 |
| --- | --- | --- |
| [BLIP-2](https://arxiv.org/abs/2301.12597) | Q-Former 连接冻结视觉编码器与 LLM | 固定查询接口已有，重点是实例角色组织与支持域约束 |
| [Slot Attention](https://arxiv.org/abs/2006.15055)、[SysBinder](https://arxiv.org/abs/2211.01177) | 竞争式槽、block-slot 属性因子化 | 直接结构先例，需超过简单迁移 |
| [OC-CLIP](https://openreview.net/forum?id=dsUK4Xql8Z) | 原提案列为文本结构与视觉槽绑定的近邻 | 本次全文访问被验证页阻挡，保留为待全文核对；不能据此断言实现细节 |
| [UniBrain](https://arxiv.org/abs/2309.06828)、[MedKLIP 官方仓库](https://github.com/MediaBrain-SJTU/MedKLIP) | 脑 MRI 分层知识监督、医学知识增强视觉语言学习 | “脑 MRI+知识+query”不能单独作为空白点；应验证患者内归属 |
| [ATAG](https://arxiv.org/abs/2207.01208) | 异常与属性图用于报告生成 | 属性建模已有，需验证强压缩后的空间实例区分 |
| [KARGEN](https://arxiv.org/abs/2409.05370) | 知识增强疾病特征与区域特征融合，条件化冻结 LLM | 知识视觉接口已有，其知识图不能直接称为 UMLS |
| [Jolia / ConQuer](https://arxiv.org/abs/2606.24570) | 无分割监督的概念查询和 3D CT 局部对齐 | 概念查询不自动等于同类多实例绑定 |
| [AutoRG-Brain](https://arxiv.org/abs/2407.16684)、[Reg2RG](https://arxiv.org/abs/2411.15539) | 区域支持报告、分割/区域引导 | 本方案主条件不依赖外部分割生成实例 |
| [NeuroFusion v1](https://arxiv.org/html/2609.02411v1) | 四序列输入、分割病灶、每病灶 Q-Former、属性头文本输入、LLM QLoRA | 直接近邻；输入序列、分割、文本通道与 LLM 适配条件均需说明 |
| [Resolution Meets Reduction](https://arxiv.org/abs/2608.08713) | 研究 3D CT 分辨率、视野、projector 和 token 预算 | 固定预算评测已有，本方案需增加绑定失真的机制证据 |

标准 Q-Former 能学习空间关系和绑定，不能声称它“没有语义”或“不能绑定”；本方案给它增加更明确的组/角色归纳偏置。每概念一个 query 表达跨患者共享类别，无法天然区分同一概念的多次出现；BindFormer 先建立图像证据组，再让各组检索同一概念库。

潜在优势是降低外部分割依赖、在固定预算内保留归属；代价是弱监督分组难、路由增加计算、依赖可靠 tuple。仅凭任务条件不同不能推出优于相关方法。

## 17. UMLS 如何获取，以及本项目需要下载什么

核对日期：2026-10-13。**UMLS 是知识/术语资源，不是患者 MRI 数据集。** 下载原始 UMLS 与训练时加载的精简概念库是两个步骤。

### 17.1 先注册 UTS 并获得许可

1. 打开 [UTS 许可申请](https://uts.nlm.nih.gov/license)，用页面提供的身份服务登录/注册。
2. 阅读并接受 UMLS Metathesaurus License，填写申请；等待 NLM 审批邮件。两个官方页面给出的处理时间分别为约 3 个和 5 个工作日，不应当作即时开通。
3. 获批后可在浏览器下载；服务器程序下载再到 [UTS My Profile](https://uts.nlm.nih.gov/uts/profile) 获取 API key。不用在聊天中发送密码或 key。

上述账号申请、许可接受需要用户本人完成。本轮没有代为申请/接受条款，也未使用凭据下载。[NLM 获取流程](https://www.nlm.nih.gov/databases/umls.html)、[API 身份验证](https://documentation.uts.nlm.nih.gov/rest/authentication.html)。

### 17.2 推荐下载：Metathesaurus Full Subset

官方当前下载页列出 **2026AA**：

| 选项 | 压缩 / 解压体积 | 对本项目的适用性 |
| --- | --- | --- |
| Metathesaurus Full Subset | 5.4 GB / 38.0 GB | 推荐；可直接解析，无需安装 MetamorphoSys |
| Metathesaurus Level 0 Subset | 1.9 GB / 10.5 GB | 较小，来源范围受限；先检查任务词汇覆盖率 |
| 单独 MRCONSO.RRF | 492 MB / 2.2 GB | 只做名称/同义词原型可用，不足以建立定义+类型库 |
| Full UMLS Release | 5.3 GB / 40.1 GB | 含额外知识资源和 MetamorphoSys，当前方案通常不必先选 |

以 [官方下载页](https://www.nlm.nih.gov/research/umls/licensedcontent/umlsknowledgesources.html) 为版本、大小、校验值的依据；不要把 Full Subset 与 Full Release 混为同一个包。Level 0 仍需 UMLS 许可；使用其他来源时查看相应来源条款。

当前文件已完整解压，并按用户要求移除版本目录层级，实际位置为：

```text
/mnt/sda/jyzhang/VLM_Projector/assets/knowledge/umls/
  META/                 # RRF 数据、随包索引及脚本
  README.txt
  Copyright_Notice.txt
```

数据版本仍为 **2026AA**；目录扁平化不改变版本。完整解压共 127 个文件、40,668,991,313 bytes（37.88 GiB），全部读取时通过 ZIP CRC 和文件大小校验。随后移动时核对全部文件的 inode/大小一致，已删除下载 ZIP、downloads 和空的 2026AA 目录，释放 5.42 GiB。`assets/` 已在仓库 .gitignore 中。

### 17.3 最简单的浏览器方式

以下步骤留作将来重新获取资源时参考，当前数据已就绪，无需重复下载。许可获批后在官方下载页选择 **UMLS Metathesaurus Full Subset**，下载到临时的 `assets/knowledge/umls/downloads/`，校验后解压。压缩包自带 2026AA 层级；按当前布局需在临时位置解压，再将其中 META、README.txt 和 Copyright_Notice.txt 移入 umls，先检查同名冲突。该文件官方 MD5 为：

```text
3f3363b4627ff16c7af39b5c4a451c1d
```

这个值只适用于上述 2026AA Full Subset；换版本/包必须取对应校验值。

### 17.4 Linux 服务器的 API 下载示例

官方 Download API 接收文件 URL 和 apiKey；下面使用标准库交互读取 key，流式写 .part，MD5 正确后才改成正式文件名。**该 Python 示例仅检查过语法；用户已通过终端下载并校验成功，当前无需再次执行。** 现有目标/partial 文件会拒绝覆盖；失败后保留 partial，不支持断点续传。

先在终端启动 Python，将以下代码保存为临时脚本后执行，或在 Python 交互环境中运行。密钥只通过终端隐藏输入，不写入脚本：

```python
from getpass import getpass
from hashlib import md5
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

root = Path("/mnt/sda/jyzhang/VLM_Projector/assets/knowledge/umls/downloads")
root.mkdir(parents=True, exist_ok=True)
name = "umls-2026AA-metathesaurus-full.zip"
target = root / name
partial = root / (name + ".part")
if target.exists() or partial.exists():
    raise SystemExit("Target or partial file already exists; inspect it first.")

key = getpass("UTS API key: ").strip()
if not key:
    raise SystemExit("Empty API key.")
source = "https://download.nlm.nih.gov/umls/kss/2026AA/" + name
url = "https://uts-ws.nlm.nih.gov/download?" + urlencode({"url": source, "apiKey": key})
request = Request(url, headers={"User-Agent": "BindFormer-resource-download/1.0"})
checksum = md5()
try:
    with urlopen(request, timeout=120) as response, partial.open("xb") as output:
        while True:
            block = response.read(8 * 1024 * 1024)
            if not block:
                break
            checksum.update(block)
            output.write(block)
except HTTPError as exc:
    raise SystemExit(f"Download HTTP status {exc.code}; check UTS license/key.") from None
except (URLError, OSError):
    raise SystemExit("Network or I/O failure; any partial file is incomplete.") from None

if checksum.hexdigest() != "3f3363b4627ff16c7af39b5c4a451c1d":
    raise SystemExit("MD5 mismatch; keep the .part file out of the concept pipeline.")
partial.rename(target)
print("Verified:", target)
```

端点依据 [NLM 自动下载文档](https://documentation.uts.nlm.nih.gov/automating-downloads.html)。还可使用 Release API 查询版本；正式实验应锁定 2026AA 这样的明确版本，而不是随 `current` 自动变化。

### 17.5 实际要读取的 RRF 文件

| 文件 | 用途 | 首版 |
| --- | --- | --- |
| MRCONSO.RRF | CUI、语言、名称/同义词、来源、术语类型及抑制标记 | 必需 |
| MRDEF.RRF | 概念定义及来源 | 定义分支必需 |
| MRSTY.RRF | 概念对应的语义类型 | 必需 |
| MRSAB.RRF | 来源词表名称、版本等元信息 | 建议保留 |
| MRRANK.RRF | 来源/术语类型优先级 | 标准名选择时使用 |
| MRCOLS.RRF、MRFILES.RRF | 文件/字段结构说明 | 保留用于解析校验 |
| MRREL.RRF | 概念关系 | 当前不做图推理，可不加载 |

字段依据 [官方 RRF 说明](https://www.ncbi.nlm.nih.gov/books/NBK9685/?report=printable) 和 [2026AA 字段表](https://www.nlm.nih.gov/research/umls/knowledge_sources/metathesaurus/release/columns_data_elements.html)。可按上述文件选择性解压，不必把全部 RRF 同时载入内存或先搭数据库。

解析流程：按官方列定义读管道分隔文件 → 筛英语/选定来源及有效术语 → 按 CUI 聚合名称/定义/类型 → 与 train 候选术语链接 → 人工处理歧义 → 按角色生成精简 concepts.jsonl → 编码 E。

某些概念没有可用定义，应记录缺失，不能自动编造 UMLS 定义。当前主版本采用名称＋简短定义：无合适定义的候选进入待补充/审核队列；本地撰写的解释必须标明来源。name-only 可作为明确标记的消融或降级配置，不能静默混入主库。预计算子集含 obsolete/suppressible 内容，过滤策略要版本化。语义类型只作筛选依据，不能自动转换成四个项目角色。

第一版不需要把全量数百万概念都变成 embedding。先建立与任务相关、可审核的候选集合，避免无关概念增加检索成本和错误语义先验；这一步由 train 和预先确定的词表决定。

## 18. 初版文档整理的完成范围

- 已读取当前数据、预处理、encoder/projector/LLM、训练、预算、checkpoint、评测及 baseline 入口，并核对活动清单统计。
- 已整理完整方案、公式、模块接口、参数草案、分阶段实现、消融与验收。
- 已核对 UMLS 官方许可/下载/API/文件格式信息，补充下载步骤与服务器示例。
- 本轮交付为本文；没有修改训练代码、原始数据、标注、配置或启动脚本，没有启动模型训练、事实抽取或 UMLS 认证下载。
- 文档校验：本地链接有效、22 个公式编号连续、代码/数学块闭合、YAML 草案可解析且预算算术一致、API 下载示例通过 Python 语法检查。配置尚未接入当前 schema，下载示例尚未做认证执行。

## 19. 2026-10-14：输入就绪度与知识构建实施审计

本节记录对当前工作区的再次核查，区分“原始资源已经存在”“任务资产需要构建”和“模型机制尚待验证”。前面的配置、模块名及资产路径仍是设计方案，不能视为已经实现。UMLS 当前已解压到 assets/knowledge/umls/META，版本为 2026AA，不再需要下载或保留旧的版本子目录。

### 19.1 当前哪些条件已经具备，哪些还没有到位

**结论：现在的输入足以先实现无知识库、无辅助监督的 BindFormer 核心并用报告生成损失训练；完整方案最主要的依赖是可审核的事实组标注和任务概念库，不是新增病灶分割。**

| 条件 | 当前状态 | 需要补充的内容 |
| --- | --- | --- |
| 单序列 MRI、序列类型、Findings、训练/验证/测试划分 | 已有 | 确定主实验范围，固定清单版本 |
| NeuroVFM、冻结 LLM、projector 接口和基线 | 已有 | 新增 BindFormer 注册、配置及启动入口 |
| patch 网格和几何元数据 | 已有整数 coords 和 affine 等元数据 | 结合轴变换、padding、patch 大小生成物理中心、有效体积及有效区域 mask |
| MRI 序列身份 | 清单已有，现用于提示文本 | 显式传入 projector，用于序列 embedding 和概念候选约束 |
| UMLS 原始资料 | 2026AA 已到位 | 任务候选抽取、CUI 消歧、定义筛选、角色/字段映射和审核 |
| 文本 encoder | 本地已有 BioLORD-2023-C | 固定 pooling、模板、长度、版本；正式模型比较仍待开展 |
| 事实组监督 | 只有 Findings，没有结构化事实 sidecar | tuple、字段可靠性/观测 mask、原文 span、否定/不确定状态及审核 |
| binding 交换负例 | 尚未构建 | 先统计多事实组和有效交换比例，再建立排除歧义的协议 |
| 辅助输出、损失、评价 | 尚未接入 | 可微辅助 tensor、共同匹配、独立损失归一化和事实绑定指标 |

当前活动清单为 assets/data/RadGenome-Brain/training_manifests/single_sequence_findings_coref_v2。全清单 train/val/test 为 2177/299/622 条，其中 BraTS 为 1960/268/560 条，对应 484/67/140 位患者。BraTS 训练集 T1W、T1C、T2W、FLAIR 各 490 条。训练记录中 1932 条的审核状态为 coreference_pass_completed_not_clinical_certification，28 条为 requires_review；完成指代处理并不等于事实组已获临床核验。

尚不能从以上数量推断有多少可靠的多实例训练样本。需要另外统计每条报告的可靠事实组数 L、字段缺失率、归属歧义率及有效交换负例率。若多数报告只提供组级描述，就应按组级标签训练，而不能人为拆成有明确病灶归属的 tuple。K 的选择也应参考这些统计，不能把一个预算算术示例直接当作最佳实例容量。

几何统计不要求真实病灶 mask：权重来自模型自己的软归属 a。当前缺的是正确的坐标/有效体积转换及验证。数据中另外存在的分割资料可用于独立质量检查；若用于训练，应单列有分割监督的实验，并给对照同等监督。

训练阶段需要纠正一个容易混淆的事实：当前 joint_lora 适配的是视觉 encoder 的 LoRA，LLM 在 decoder 初始化和训练阶段检查中均被冻结。设计主实验冻结 encoder 和 LLM，应使用 training.stage=a 或等价配置；不能把现有 joint_lora 基线直接当作“两端均冻结”的对照。

当前接口也有具体缺口：

- multimodal 只向 projector 传特征、coords 和 metadata，没有显式 sequence 参数；返回的训练信息主要是 RG loss_sum 和 answer_tokens。
- ProjectedVolume 尚无专门承载可微辅助张量的接口。属性头应放在 projector 模块内，确保进入现有 optimizer 与 checkpoint。
- 现有 QueryBlock 带 query 残差。若 query 含知识向量，直接复用会把知识经残差送入输出；这不符合本方案首轮属性 token 仅由视觉 value 汇聚的定义，需要独立的支持域偏置读取模块。
- trainer 对 RG 按答案 token 总数归一化；tuple/binding 应按各自有效字段或样本归一化，不能直接混进 RG 分子。
- 现有图像置换按 sequence 分组，尚未完整满足“同队列、同序列、跨患者”的检查条件。
- 相同压缩比例不保证相同 M。以 N=8775、ratio=8 为例，查询型、网格型、concat 路径分别可能产生 1097、1100、1125 个 token；主比较需要明确固定实际输出 M，或报告无法完全匹配的差额。

### 19.2 tuple loss 与 binding loss：监督单位、作用与实现

tuple 是“同一个证据组的若干字段”，例如以下合成示例。它不是固定长度的自然语言句子，也不要求每个字段都有标签。

| 事实组 | finding | 部位 | 侧别 | 当前序列信号 | 形态 |
| --- | --- | --- | --- | --- | --- |
| A | 异常信号区 | 额叶 | 左 | 高信号 | 点状 |
| B | 异常信号区 | 顶叶 | 右 | 低信号 | 片状 |

每个槽的最终属性包通过辅助头输出字段分布 p。字段 j 由对应角色 token 预测；一个角色可以对应多个头，例如 location/laterality 同时对应解剖和侧别。单分类使用 CE，多标签使用带观测 mask 的 BCE；未提及的标签不自动成为阴性，显式 unknown 与缺失标签也应分开。

tuple loss 先计算“槽 k 能否完整解释事实组 ℓ”的成本：

$$
C_{k\ell}=\sum_j\omega_{\ell j}\,\ell_j(p_{kj},t_{\ell j}),\qquad
\pi^*=\arg\min_{\pi\ {\rm injective}}\sum_{\ell=1}^{L}C_{\pi(\ell),\ell}.
$$

$$
\mathcal L_{\rm tuple}=
\frac{\sum_{\ell=1}^{L}C_{\pi^*(\ell),\ell}}
{\max(1,\sum_{\ell,j}\omega_{\ell j})}.
$$

Hungarian matching 为整组字段只求一次共同匹配：可以把 A 匹配到槽 3、B 匹配到槽 1，但不能让 A 的部位来自槽 3、信号来自槽 7。它同时解决槽没有固定编号的问题，并约束同一个槽保留同一事实组的属性。tuple loss 本身已经包含绑定约束。

binding loss 则显式比较正确组合与错误组合。例如只交换 A/B 的信号，构造“左额叶低信号、右顶叶高信号”，保持概念集合不变，其余锚定属性保留。令 S 为最佳共同匹配下的平均字段对数似然：

$$
\mathcal L_{\rm bind}=
\max\{0,\gamma-S(B,\mathcal T)+S(B,\widetilde{\mathcal T})\}.
$$

正确和错误集合分别重新寻找最佳匹配，避免模型仅靠槽编号识别负例。正确组的分数必须至少比错误组高 γ。若 γ=0.2、正确分数=-0.5、错误分数=-0.45，则损失为 0.25；若正确分数=-0.2、错误分数=-0.8，则该项为零。

两者的区别是：tuple loss 要求正确事实组预测准确；binding loss 要求模型能够拒绝“词都出现了，但归属接错了”的混淆组合。两者可能部分重复，所以应比较 RG、RG+tuple、RG+tuple+binding。它们是设计目标，不能在实验前宣称必然减少报告错误。

实现边界：

- 交换完整 tuple 的排列不是负例；属性相同、归属含糊、交换后仍可能正确的样本不使用。无有效负例时 binding 为零。
- L=0 时 tuple 为零；L>K 必须报错、调整容量或使用预先定义的组级协议。未匹配槽不能直接标为“无病灶”。
- 匹配成本可 detach 后交给 CPU 求解，但计算最终损失应索引原有可微成本/预测，保留梯度。
- 多标签若仅标明部分阳性，须在标签维度再设观测 mask；不能用完整 BCE 把所有未提及标签当阴性。
- 辅助损失和 RG 一起反向传播，不需要再次运行 LLM。原设计辅助头位于最终 B、输出投影之前；头预测正确仍不能证明 Z 或生成报告保留了绑定，应另外评价报告事实及图像依赖性。
- tuple 标签可以使用本地 ID，UMLS 并不是启动 tuple/binding 训练的前提。

### 19.3 离线 encoder：现有模型与正式比较

四个角色共享同一个冻结文本 encoder。角色差异由候选集合、在线投影矩阵和 role embedding 表达，不需要下载四个独立模型。

| 用途 | 模型 | 选择依据与限制 |
| --- | --- | --- |
| 先打通离线资产流程 | 本地 FremyCompany/BioLORD-2023-C | 资源已在本地；C 是 contrastive-only 版本，不能与完整 BioLORD-2023 混称 |
| 正式 name+definition 实验的优先候选 | FremyCompany/BioLORD-2023 | 训练目标利用概念名称与描述，适合作为待验证候选；尚未完成本任务比较，也未在当前目录发现其权重 |
| 名称/同义词基线 | cambridgeltl/SapBERT-from-PubMedBERT-fulltext | 以医学实体同义表达对齐为主要依据；长定义输入的优势不能直接假定 |

模型依据：[BioLORD-2023 官方模型卡](https://huggingface.co/FremyCompany/BioLORD-2023)、[BioLORD-2023-C 官方模型卡](https://huggingface.co/FremyCompany/BioLORD-2023-C)、[SapBERT 官方模型卡](https://huggingface.co/cambridgeltl/SapBERT-from-PubMedBERT-fulltext)、[SapBERT 原论文](https://aclanthology.org/2021.naacl-main.334/)。

本地 assets/models/evaluation/biolord 的实际 repo ID 是 FremyCompany/BioLORD-2023-C，revision 为 4bb5abd13930002136e3a5750b84474763c775c5。它没有完整的 SentenceTransformer 模块配置，宜显式使用 AutoTokenizer/AutoModel、attention mask mean pooling，再做 L2 normalize。已使用本地环境在 CPU、离线模式完成 8 条术语的编码检查：输出为 8×768，数值全部有限，向量范数约为 1。这次检查没有生成正式概念资产。

BioLORD 按官方 mean pooling 使用；SapBERT 按其官方示例使用最终层 CLS 表示。pooling 是 encoder 配置的一部分，不应对所有模型随意采用同一方式。初版可将文本长度上限固定为 128，并记录截断；本地 BioLORD 配置也使用此上限。

当前主版本直接使用 name+definition；name-only 留作后续在同一 encoder 下的消融，再比较不同 encoder，避免把模型差异误认为定义的贡献。没有可用定义的条目进入待补描述队列，人工补充描述需明确标记本地来源。2026AA 新术语可以编码，但模型能处理字符串并不表示其 CUI 或视觉含义已经正确。

当前 RaTEScore 也使用本地 BioLORD。若知识库和指标使用同一模型家族，应披露这一共享，并增加独立事实/绑定指标与人工核验；这本身不等于测试集泄漏。

### 19.4 四个角色选哪些概念

以下为需要结合训练报告和预定词表审核的候选范围，尚不是正式 CUI 清单。角色是输出组织方式，字段才是标注与预测单位。

| 角色 | 子字段 | 首批候选范围 |
| --- | --- | --- |
| finding | 可观察所见/效应 | 异常信号区、占位相关改变、水肿相关表现、占位效应、中线移位、脑室受压/扩大；避免直接用队列诊断代替图像所见 |
| location/laterality | 解剖；侧别 | 额/顶/颞/枕叶、丘脑、基底节、小脑、脑干、胼胝体、脑室、白质/灰质、脑室旁；左、右、双侧、中线 |
| signal | 信号强度；信号质地；增强表现 | 当前序列上的高/低/等/混合信号，均匀/不均匀；增强另设子字段并限制适用序列 |
| morphology/extent | 形态；边界；范围；数量 | 点状、片状、结节状、不规则；边界清楚/不清；局灶/弥漫及受累范围；单发/多发 |

侧别和部位不互斥，“左额叶”不能被强迫在 left 和 frontal lobe 中二选一。原方案的角色级 softmax 是查询用的软混合，不是字段分类器；若一个角色内的竞争影响检索，可对字段分别检索再融合到同一个角色 query，仍保持 R=4，并将此作为明确的结构变体。

MRI 序列应同时进入视觉端和知识端的适用性约束。增强候选应限制到有相应证据的输入条件；BraTS 当前四种输入不支持凭空增加 DWI/ADC 结论。可让不同序列的概念条目共享基础 CUI，但拥有不同本地 concept_id 和 sequence_scope；不能把所有序列条件混进一个含义不清的向量。

本次只对训练报告做了词面核查，例如 edema 命中 990 条、mass effect 49 条、midline shift 395 条、hyperintens* 410 条、hypointens* 117 条、isointens* 297 条、punctate 9 条。这些是包含否定、不确定表达的字符串命中数，不是阳性数、病灶数或已审核标签数；也不能据此判断属性归属。

对 40 个英文候选短语做了有界的 UMLS 精确匹配核查：筛选 LAT=ENG、SUPPRESS=N，34 个短语获得精确匹配，共涉及 43 个候选 CUI。以下为找到的示例，仍需结合来源定义和报告上下文审核：

| 候选词 | 候选 CUI | 可能用途 |
| --- | --- | --- |
| mass effect | C4086564 | finding |
| midline shift | C4086580 | finding |
| frontal lobe | C0016733 | location |
| corpus callosum | C0010090 | location |
| hyperintensity | C5667489 | signal，需补充序列语境 |
| hypointensity | C5667487 | signal，需补充序列语境 |
| patchy | C0205413 | morphology |
| punctate | C0205330 | morphology |

实际发现的歧义包括：lesion 的候选既有 Finding，也有 Intellectual Product；midline 既有 Spatial Concept，也有 Cell Component；enhancement 的精确候选包含 Therapeutic or Preventive Procedure / Activity，不能当作 MRI 增强直接使用。brain edema 也匹配到多个 CUI。isointensity、contrast enhancement 等在本次拼写和过滤条件下未精确命中，不代表 UMLS 不存在对应概念，应继续做同义词检索和人工消歧，必要时保留 local ID。

### 19.5 离线构建的具体流程与资产

1. **先固定 schema。** 定义四角色、子字段、序列适用性、单分类/多标签、肯定/否定/不确定、缺失标签、组级描述和审核规则。显式 unknown 与未观测分开处理。
2. **只从训练文本建立患者语料候选。** 初轮可抽取 100–200 条分层训练报告作为审核试点，覆盖序列、队列及归属困难的多事实情况；该数量只是起点。保存原文 span、序列、报告 hash。规则、NER 或本地 LLM 只能产生待审草稿。
3. **流式读取 UMLS，建立任务子集。** MRCONSO 取名称/同义词及 CUI，MRDEF 取定义，MRSTY 取语义类型，MRSAB 记录来源版本。当前 MRCONSO 约 1806 万行，不应一次全部装入 Python 对象，也无需先编码全部 UMLS。
4. **候选链接与消歧。** 规范化精确匹配、同义词匹配后，结合语义类型、来源定义及 MRI 上下文选择；向量检索只提供候选。MRDEF 没有独立 LAT 字段，必须另外控制定义来源/语言，不能盲目拼接所有定义。
5. **形成可审核的概念表与事实表。** 审核通过的通用 concepts 和逐样本 facts 分开保存。无准确 CUI 的修饰词使用 local ID，附上来源和审核状态。推理不读取患者参考报告来选择概念。
6. **离线编码并冻结资产。** 固定模型 revision、tokenizer、pooling、模板、长度上限、概念顺序及归一化；生成 name-only / name+definition 两份同序向量，保存内容 hash、截断和缺失定义日志。
7. **冻结词表后应用到验证/测试。** 使用固定 schema 和词表标注、评价验证/测试；不根据测试报告新增词表或调参。验证/测试的可靠事实标注是独立评价资产，不是训练候选来源。

建议产物：

    assets/knowledge/bindformer_brain_v1/
      schema.json
      candidates.jsonl                 # 待审候选，不能直接作为训练真值
      concepts.jsonl
      role_field_masks.pt              # 含角色、字段和序列适用性
      embeddings_name.pt
      embeddings_description.pt
      manifest.json
      audit.json
    assets/annotations/bindformer_facts_v1/
      train.jsonl
      val.jsonl
      test.jsonl
      review_queue.jsonl
      extraction_manifest.json

concepts 至少包含 concept_id、可空 CUI、标准名、同义词、role/field、sequence_scope、定义、定义来源/版本和审核状态。facts 至少包含 sample_id、sequence、report_hash、事实组列表、字段观测/可靠性、原文 span、审核状态及 extractor/schema 版本。多标签需要逐标签观测状态。

验收应覆盖：概念 ID 与向量行号一致、角色/序列候选 mask 正确、数值有限和范数、同配置可重复编码、定义来源与截断可追溯、否定/侧别/信号混淆候选人工抽查。语义向量相近不等于属性可互换，尤其要检查高/低信号、左/右等对立属性。

### 19.6 实现顺序与尚待证实的条件

- **核心与 RG：** 实现输入适配、竞争式槽更新、角色 query、共享支持域读取、全局分支、固定 M 输出。先关闭知识和可选交互，验证数据流、梯度、预算及生成。
- **事实监督：** 构建审核试点和 sidecar，再加入 tuple；统计合格负例后加入 binding。标签可先用本地 ID，不必等 UMLS 完成。
- **知识消融：** 加入角色候选、冻结向量和在线语义检索；比较无知识、name-only、name+definition，并控制 encoder 版本。
- **几何与交互：** 实现和验证物理坐标转换，再逐项加入几何统计、包内交互和全局补充。几何 MLP 是小型新训练模块，可由标准 Linear/激活/Linear 构建，不是必须另找预训练模型。
- **公平评价：** 两端冻结、同数据和相同实际 M 为主条件；比较普通查询基线加同样的 tuple/binding 监督，记录 projector 与 LLM 分项延迟、总延迟和显存。

仍需实验证实：冻结 NeuroVFM 特征是否可读出所需事实、弱监督槽是否保持有效分工、训练语料是否有足够明确归属、辅助头中的绑定是否保留到 Z 和最终报告。这些不能仅靠下载 UMLS 或加入 attention 结构解决。

本次新增内容仅为审计与设计补充，并完成本地文本 encoder 的 CPU 编码检查；没有实现 BindFormer、生成正式概念库/事实标签或启动训练。

## 20. 补充训练方案：以 X 为目标的特征重建与报告片段对齐

本节正式记录两项待实现、待消融的辅助目标。它们复用现有图像—报告数据，分别检查压缩后的视觉内容保留和临床文本语义对齐；不要求新增逐实例人工训练标签。主实验仍冻结 NeuroVFM 和 LLM，维持相同的最终输出预算 M。

| 补充目标 | 训练监督从哪里来 | 辅助模块读取什么 | 推理时是否需要辅助模块 |
| --- | --- | --- | --- |
| L_feat：特征重建 | 当前图像的冻结 NeuroVFM 输出 X | 最终 Z、对应位置 p、序列 m | 否 |
| L_text：报告片段对齐 | 已有图像—报告配对，冻结文本 encoder 的片段表示 | 最终 Z 的证据包和训练用文本向量 | 否 |

两者可独立开关，总目标写作：

$$
\mathcal L_{\rm supplement}
=\mathcal L_{\rm RG}
+\lambda_f\mathcal L_{\rm feat}
+\lambda_s\mathcal L_{\rm text},
\qquad \lambda_f,\lambda_s\ge0.
$$

训练数据流：

    主路径：V → 冻结 NeuroVFM → X → Projector → Z → 冻结 LLM → L_RG
    重建分支：Z + p + m → R_ψ → X_hat；与 stop-gradient(X) 比较 → L_feat
    文本分支：Z 的证据包 + 离线报告片段向量 → 投影/匹配头 → L_text

两个辅助分支的视觉输入都来自 Z。报告损失和辅助损失共同向 projector 传梯度；重建头及文本匹配头只接收各自辅助目标的梯度。需要保留 LLM 对 Z 的梯度，但不更新 LLM 参数。辅助目标不要求额外调用一次 8B LLM。

第 20.2/20.3 节给出目标定义；第 21.2/21.3 节给出规模估算及多槽匹配变体。具体 projector 简化结构、辅助头形式和首轮启用哪些目标，留待后续讨论；此处不把这些设计候选当作已经实现的默认配置。

### 20.1 RadGenome 已有什么标注，原损失究竟缺什么

重新检查了本地五个队列的原始 JSON：modal_wise_finding 和 global_finding 的值均为自然语言字符串；impression 是包含 disease 和 impression 的对象。当前清单包含图像、分割路径和报告，未提供可直接使用的 lesion_id、逐实例属性字段和字段归属关系。当前 Dataset 明确不读取病灶 mask 进行训练。

公开 RadGenome-Brain 确实提供异常区域分割及医生撰写的报告，不能表述成“没有区域监督资料”。但区域支持的报告、语义分割和本方案需要的逐实例属性 tuple 不是同一种标注。异常 mask 不自动提供某个信号/形态短语对应哪个病灶；连通域也不必然等于临床实例。[官方数据说明](https://github.com/ljy19970415/AutoRG-Brain)

原 tuple 所需的是每个事实组的字段及其可靠性，例如 finding、anatomy、laterality、signal、morphology、assertion，加原文 span 和缺失 mask。可以由现有报告自动抽取草稿，再筛选和抽查，不需要重新手写所有报告或重新画 mask。报告中本来没有明确写出的归属不能靠抽取器补成真值。global_finding 可能涉及多个序列，不能无条件用它给单序列输入补标签。

binding 负例无需另请医生逐个撰写：在可靠的事实组基础上按预定规则交换部分属性，并排除无效/含糊的交换。但其正确性仍依赖原事实关系，不能把任意跨实例交换都视为错误。

换疾病不必重设计全部 schema；影像所见、位置、侧别、信号、形态和状态大部分可复用。换数据集仍需处理文本格式、语言、序列及术语分布，并校验自动抽取质量。自动重跑预处理和“从头人工标注一个数据集”应区分，但自动抽取并不消除监督噪声。

考虑当前跨数据集目标，建议默认模型不强制要求 facts sidecar：报告生成是必选训练目标，tuple/binding 仅在存在可靠事实资产时启用，并单独报告监督来源。小规模关系审核优先用于独立评价，也可以提供可选训练子集；这不是要求给所有训练样本增加人工标签。

### 20.2 候选一：从最终 Z 重建冻结视觉特征

目标是约束压缩后仍保留“某个位置上的视觉内容”。训练目标直接来自冻结 NeuroVFM，不需要疾病分类、实例标签或分割 mask。

令原始 encoder 输出为 x_i，I 是有效 patch 中抽样的一组位置，增加一个仅训练时使用的小型重建网络：

$$
\widehat x_i=R_\psi(Z,\operatorname{PE}(p_i),m),\qquad
\mathcal L_{\rm feat}=
\frac1{|\mathcal I|}\sum_{i\in\mathcal I}
\left\|\mathcal N(\widehat x_i)-
\operatorname{sg}\!\left[\mathcal N(x_i)\right]\right\|_2^2.
$$

N 为固定的特征标准化操作，sg 表示停止目标端梯度。实现可用坐标 query 对 Z 做一次轻量 CA，再映射到 encoder 宽度；这不是 8B 语言 Decoder。首版使用固定标准化、空间分层抽样和固定样本数，记录有效区域及重建指标。

必须约束目标和输入通路：

- 目标用冻结 encoder 的 x，不能用可训练 W_x 后的 H，否则目标可以随模型一起坍塌。
- 重建头只访问 Z、坐标和序列，不能旁路读取原特征 X 或高维归属矩阵 A；否则可以绕过固定 M 的压缩瓶颈。
- 在最终 Z 上做约束，检查输出投影之后的信息；不要仅检验中间槽。
- 只重建部分有效位置，控制 N×M 读取开销；不要使用真实病灶 mask 为主实验选择重建区域。
- 普通脑结构可能主导损失，坐标也可能让网络记住平均脑模板。需要检查跨患者图像置换后重建是否变差，并单独评价小异常；重建误差低不等于临床绑定正确。
- 若后续训练 encoder，须保留固定/独立 teacher，不能直接把“冻结目标”假设带到 joint_lora 设置。

使用预训练特征重建学习对象表示已有 DINOSAUR 等先例；这里提出的是适配最终 Z、三维位置和固定 token 预算的候选约束，不能宣称是全新损失或已经验证优于 tuple。[DINOSAUR 原论文](https://arxiv.org/abs/2209.14860)

### 20.3 候选二：报告片段与证据包的弱对齐

保留报告中可独立理解的完整句子或连续句窗，而不是只提取无关系的概念集合。文本可以是“左额叶的片状高信号区”，不强制先得到离散的疾病类别和实例 ID。

设 c_l 为训练报告片段，v_l 为冻结文本 encoder 的向量；从最终 Z 的第 k 个角色包通过一个小投影头得到 u_k。可使用多实例匹配分数：

$$
s(I,c_l)=\max_k \operatorname{cos}(u_k,v_l).
$$

对其所属图像 I 提高分数，对筛选后的其他图像降低分数。例如：

$$
\mathcal L_{\rm text}=
-\frac1{L_c}\sum_{l=1}^{L_c}
\log
\frac{\exp(s(I,c_l)/\tau)}
{\exp(s(I,c_l)/\tau)+
\sum_{I'\in\mathcal N_l}\exp(s(I',c_l)/\tau)}.
$$

这里的正确配对来自已有图像—报告关系，槽与句子之间没有人工指定匹配；不要求一个句子对应一个病灶，也不要求句子数量等于槽数。无可用片段/负例时跳过该项。对整体正常或弥漫描述可另用全局表示；无法独立理解的代词句、跨序列描述及无法判定作用范围的片段先只用于 RG，不强行落入局部槽。

这是一种弱监督候选，不等价于 tuple：

- 文本向量未必对左右、高低信号和属性交换足够敏感；应单独检测这些关系混淆。
- 所有句子都匹配同一槽也可能成为退化解；不能把 max 匹配称为已实现实例分离。
- 不同患者可以有相同所见，不能把所有不配对图像都当作确定阴性。应排除同患者、重复/高度相似片段，对常见正常句谨慎采负例，并控制序列/队列捷径。
- 文本 encoder 离线缓存，辅助头只在训练时使用，不增加每次生成时的文本编码或 LLM 调用。

GLoRIA 等工作已经表明图像—报告配对可用于局部弱对齐，但其结果不能直接证明本项目的三维实例绑定效果。这里的整包—片段对齐是待验证适配方案。[GLoRIA 原论文](https://openaccess.thecvf.com/content/ICCV2021/html/Huang_GLoRIA_A_Multimodal_Global-Local_Representation_Learning_Framework_for_Label-Efficient_Medical_ICCV_2021_paper.html)

### 20.4 正则化能做什么，不能代替什么

可选检查包括轻微扰动下的支持域一致性：比较两个视图的公共有效区域，在恢复空间变换并允许槽置换之后约束归属一致。不要把左右翻转、改变信号语义的强增强当成标签不变的默认增强。此项增加 projector 计算，若从影像重编码还会增加 encoder 计算，应计入训练开销。

一致性、熵或多样性正则不能单独提供临床实例真值。均匀归属也可以很稳定，强制等体积槽会破坏大小差异，强制完全不重叠会妨碍弥漫改变或相邻证据。由于每个 patch 对全部槽的归属和本来就是 1，仅约束这一和等于 1 的“coverage loss”也是冗余的。

若使用实体袋 BCE/概念对比，最多先解决“哪些概念出现”，不能把它当作“概念归属正确”的证明。若用 LLM 自动抽取 tuple 或生成交换句，仍属于伪标注监督，需要记录来源及有效性过滤，不能改称无监督。

建议按独立实验依次比较：

| 实验 | 必需输入 | 主要检验 |
| --- | --- | --- |
| RG | 图像、报告 | 结构本身在相同 M 下的贡献 |
| RG + feat | 同上 | 最终视觉 token 是否保留位置相关内容 |
| RG + text | 同上，离线片段缓存 | 现有文本弱监督是否提供更多临床语义 |
| RG + feat + text | 同上 | 两个目标是否互补 |
| 可选 tuple/binding | 可靠事实子集或筛选伪标签 | 更直接关系监督的额外作用 |

先分别验证，不能预设所有损失叠加最好。新增辅助头、训练算力和监督都应在基线中提供相应对照。当前建议的通用组合是 RG + λ_f feat + λ_s text，两个辅助权重初始均为可关闭配置；这是一条实验路线，不是已证实的最优目标。

### 20.5 本地 BioLORD 目前的真实用途

src/evaluation/clinical_scorers.py 的 RateScorer 将 assets/models/evaluation/rate-ner 用作实体识别模型，将 assets/models/evaluation/biolord 用作实体向量模型。后者是 BioLORD-2023-C。

当前流程为：生成报告/参考报告 → RaTE-NER 提取实体 → BioLORD 编码实体 → 结合类型和相似度计算 RaTEScore。它没有参与现有 projector 的训练、MRI 编码或在线 UMLS 查询；此前检查只是验证它可复用于离线建库。

RaTEScore 原论文的实体编码比较选用了 BioLORD-2023-C；其表 8 的对照包括 MedCPT、RadBERT、CXR-BERT 和 BioViL-T。不能把这张表说成对完整 BioLORD-2023、C 和 SapBERT 的三者排名；表 6 的 SapBERT 对照是 NER 初始化实验，也不能挪作知识向量质量排名。[RaTEScore 原论文](https://aclanthology.org/2024.emnlp-main.836/)

### 20.6 encoder、输入文本和 pooling 是三个独立选择

“BioLORD”是模型家族的泛称，实验必须写出具体年份和版本；原 BioLORD-2022、BioLORD-2023-C 和完整 BioLORD-2023 不是同一个 checkpoint。

| 选择 | 更适合优先检验的用途 | 限制 |
| --- | --- | --- |
| SapBERT | 名称/同义词的实体归一化和候选链接 | 不能假定其 CLS 对长解释、否定和关系同样有效 |
| BioLORD-2023-C | 现成的实体表示、名称检索基线 | 官方定位包含 NEL；不能据版本后缀断言低于完整模型 |
| BioLORD-2023 | 名称加简短定义、完整报告片段 | 目标与描述语义较接近，但 MRI query 引导性能未比较 |

BioLORD-2023 原论文在其评测协议下报告了多项优于 SapBERT 的语义结果，但并非每个任务都领先，且其下游评测包含特定微调流程。它不构成本项目冻结权重、MRI 概念引导的直接性能证据。[BioLORD-2023 原论文](https://academic.oup.com/jamia/article/31/9/1844/7614965)

“整段解释”是输入文本，“CLS”是输出汇聚方式。两者并非互斥：

    输入文本：概念名称，或 名称 + 简短定义
       → tokenizer → Transformer → L×768 个 contextual token 表示
       → CLS / 带有效 token mask 的 mean pooling
       → 归一化后的一个 768 维概念向量

CLS 的最终隐藏状态也经过上下文计算，不是只拿一个没有阅读解释的固定向量；mean pooling 则汇聚有效位置的上下文表示。两者都可以处理整段解释，但应匹配模型预训练/官方使用方式。

官方用法分别是 BioLORD 的 masked mean pooling 和 SapBERT 的最终层 CLS。按用户当前选择，首版主库使用 SapBERT＋CLS＋name+short-definition；name-only 是后续消融，BioLORD 的 CLS 可作为独立变体。不同 pooling 需要单独记录和比较，不能把“用完整文本”误写成“只能 mean pooling”。[BioLORD 模型卡](https://huggingface.co/FremyCompany/BioLORD-2023)、[SapBERT 模型卡](https://huggingface.co/cambridgeltl/SapBERT-from-PubMedBERT-fulltext)

对名称匹配，名称和同义词应占主要信息；对视觉 query 引导，当前主版本采用名称加简短、经审核且与角色/序列相关的定义。长定义可能包含病因、其他器官、其他成像模态和诊断关联，这些内容不一定有助于当前图像读取。没有合适定义时先记录缺失并补充审核，不能将自行撰写的说明伪装成 UMLS 原始定义。

本版仍将一段文本池化成一个概念向量。把全部 token 向量保留为概念记忆属于另外一种多向量结构，会改变存储、检索和计算成本，不建议与第一版同时引入。

选择应分两层验证：先检查概念链接准确率、同义词召回和左右/高低/否定等混淆；再在相同数据、M 和模型容量下比较最终 Findings。无监督图像语义与文本向量不会自动对齐，在线 T_r、U_r 等映射仍需通过训练学习。

本节仅更新设计与依据，没有切换现有评测模型、生成伪标签、实现辅助损失或启动训练。

## 21. CLS 起步、候选概念草案和辅助目标细化

### 21.1 已下载的模型与第一版 pooling

用户选择先尝试 CLS，并明确首版直接输入“概念名称＋简短定义”。因此当前知识编码主方案为 SapBERT 的最终层 CLS＋name+short-definition；name-only 留作后续消融。BioLORD-2023 的最终层首 token 表示也可以作为 CLS 变体实验，但应标明不同于其官方 masked mean pooling，不能据此概括完整 BioLORD 模型的最佳能力。输入文本和 pooling 是独立配置。

| 模型 | 本地路径 | 固定 revision | 下载文件合计 |
| --- | --- | --- | ---: |
| cambridgeltl/SapBERT-from-PubMedBERT-fulltext | assets/models/knowledge/sapbert | 090663c3ae57bf35ffe4d0d468a2a88d03051a4d | 438,186,514 bytes |
| FremyCompany/BioLORD-2023 | assets/models/knowledge/biolord-2023 | 167aab527b238a50ca65224e6319215d2ff4fc9f | 438,925,385 bytes |

下载脚本为 [download_knowledge_encoders.py](../scripts/download_knowledge_encoders.py)，仅保存 safetensors 权重、配置、tokenizer 和模型说明，没有重复下载其他框架的权重。所有文件记录 SHA256，权重与 Hub LFS SHA256 比对通过，各目录保存 asset_manifest.json。

两模型均已用本地文件在 CPU 加载，分别检查 CLS 与 masked mean 的 4×768 输出、有限值和归一化范数，结果保存在各目录的 load_check.json。这是可加载性与数值检查，不是模型质量排名。没有替换现有评测用 BioLORD-2023-C，也未导出正式概念向量。

### 21.2 重建头可以明确限定为约 185 万参数

以下是一个具体候选架构的参数核算，尚未接入训练。输入为最终 Z 的 4096 维 token，内部宽度 256，单层 4-head CA，FFN 扩展为 512，输出 768 维 encoder 特征。坐标标准化及轴方向须按数据协议处理。

| 组件 | 结构 | 参数数 |
| --- | --- | ---: |
| Z 标准化 | LayerNorm(4096) | 8,192 |
| Z 降维 | Linear(4096,256) | 1,048,832 |
| 坐标 query | Linear(3,256), GELU, Linear(256,256) | 66,816 |
| 序列 embedding | 6×256，含预留序列槽 | 1,536 |
| query 标准化 | LayerNorm(256) | 512 |
| cross-attention | MultiheadAttention(256,4) | 263,168 |
| FFN 标准化 | LayerNorm(256) | 512 |
| FFN | Linear(256,512), GELU, Linear(512,256) | 262,912 |
| 输出 | Linear(256,768) | 197,376 |
| 合计 | 含上述 bias 和标准化参数 | 1,849,856 |

例如每体积抽样 512 个有效 patch、Z 含 256 个 token，位置 query 读取 Z 后预测对应 x_i。这里的数量是计算示例，不是已经确定的主实验预算。FP32 权重约 7.4 MB；梯度、optimizer 和 activation 另外占用显存，不能把权重大小当训练显存或据参数量推断实测延迟。

训练时 projector 参数 θ 和重建头参数 ψ 同时优化：

$$
\mathcal L=\mathcal L_{\rm RG}(P_\theta(X))
+\lambda_f\mathcal L_{\rm feat}(R_\psi(P_\theta(X),p,m),\operatorname{sg}(X)).
$$

RG 更新 projector；feat 更新 projector 和重建头。主实验的 encoder 和 LLM 保持冻结，LLM 仍向 projector 传梯度。推理只保留 projector，移除 R，不增加 8B LLM 调用。

重建目标是输入图像在固定 encoder 下的特征，而不是原始 MRI 像素。验收应包括：按患者隔离的验证集上归一化 MSE/余弦相似度、位置对应是否正确、对未抽样位置和不同脑区域的泛化、坐标/序列专用头与平均特征基线、换成其他患者 Z 后是否明显变差，以及对最终报告事实的影响。只看训练重建 loss 下降不足以排除重建头过拟合或平均脑模板捷径。即使特征恢复很好，也不能据此声称图像完全相同、真实病灶分割正确或全部临床事实保留。

### 21.3 从 max 分数到损失，以及多槽匹配的边界

原分数 s(I,c)=max_k cos(u_k,v_c) 可以直接进入图像对比损失：

$$
\mathcal L_{\rm text}=
-\frac1{|\mathcal C_{\rm valid}|}\sum_{c\in\mathcal C_{\rm valid}}
\log
\frac{\exp(s(I^+,c)/\tau)}
{\exp(s(I^+,c)/\tau)+\sum_{I^-\in\mathcal N_c}\exp(s(I^-,c)/\tau)}.
$$

正例来自已有报告所属图像，负例须排除同患者和相同/相近临床内容等明显冲突。这里不是给其他患者标医学阴性。没有可靠负例时跳过该项。

max 只有胜出槽获得这一匹配项的梯度；多个槽即使都相似，也不会因此被训练成共同解释。一个槽可以在多句话上反复胜出。它没有任何“均衡槽数”的保证，所以只能作为最简单的多实例基线。

换成 log-sum-exp 可以分散梯度，但仍主要聚合替代匹配分数，不自动证明两个槽的互补事实被组合起来。真正需要“多个槽共同解释句子”时，应使用集合读取/聚合头，并对组合后的视觉表示计算文本匹配损失。

可选方案是学习软匹配矩阵 T_{k,l}，对每个有效片段满足 Σ_k T_{k,l}=1，允许一列有多个非零项，再以一个小型集合头聚合被选中的完整角色包。所有角色使用同一个 T，不能各自跨槽挑字段。对每个候选图像都重新计算匹配，集合头的 value 仅来自视觉 token，避免将目标文本直接复制进视觉表示。

为了限制单个槽承接全部片段，可加入软容量项：

$$
\mathcal L_{\rm cap}
=\sum_k\left[\max\!\left(0,\sum_lT_{k,l}-c_k\right)\right]^2.
$$

c_k 是允许的文本分配质量上限，不是真实病灶数；重复描述同一异常时应允许同一槽复用。未使用槽可保持零分配；不可靠片段可在训练前过滤或引入单独的 null 匹配，并检查全选 null 的退化。也可研究具有放松边际约束的匹配求解，但不能把一个平衡算法当作临床对应关系的保证。[不平衡最优传输的基础方法](https://arxiv.org/abs/1607.05816)

注意：列归一化保证的是每个片段的分配权重总和，不是相同的非零槽数。若 K=12，三句话实际需要 2、3、1 个槽，剩余 6 个空闲完全合理；强制每句 4 个槽会制造冗余。希望控制的是可靠片段的覆盖、避免严重垄断和错误绑定，而不是句句相同的槽数量。

### 21.4 实际构建的 UMLS 候选草案

候选种子为 [bindformer_umls_seed_terms_v1.tsv](bindformer_umls_seed_terms_v1.tsv)，生成脚本为 [build_bindformer_umls_candidates.py](../scripts/build_bindformer_umls_candidates.py)。仅使用当前 BraTS 训练集 1960 条报告计算词面证据，不读取验证/测试文本。

| 角色 | 种子条目数 | 本轮覆盖 |
| --- | ---: | --- |
| finding | 14 | 异常信号、病变/占位、水肿、出血、坏死、囊性改变、占位效应、中线移位、脑室效应等 |
| location/laterality | 30 | 26 个解剖/区域词，4 个侧别/中线词 |
| signal | 10 | 信号强度、均匀性、增强模式 |
| morphology/extent | 14 | 形状、边界、范围、数量 |
| 合计 | 68 | 任务候选，不是最终概念数 |

其中 61 个种子获得规范化精确匹配，涉及 93 个不同 CUI；20 个种子有多个 CUI。其余 7 个应继续做同义词消歧或保留 local ID，不能解释为 UMLS 中一定不存在。

产物位于 assets/knowledge/bindformer_brain_candidates_v1：

- [REVIEW.md](../assets/knowledge/bindformer_brain_candidates_v1/REVIEW.md)：全部候选、训练词面命中数和歧义标记。
- [candidates.jsonl](../assets/knowledge/bindformer_brain_candidates_v1/candidates.jsonl)：角色/字段、序列范围、命中来源、候选 CUI、英文同义词、语义类型、原始定义来源。
- [audit.json](../assets/knowledge/bindformer_brain_candidates_v1/audit.json)：输入版本、hash、数量和构建限制。
- [source_versions.json](../assets/knowledge/bindformer_brain_candidates_v1/source_versions.json)：候选涉及的 UMLS 来源版本。

所有条目均为 pending，selected_cui 均为空；候选绝不直接充当病例真值。训练命中统计包括否定、不确定和上下文歧义，仅用于决定审核优先级。没有训练命中的预定 schema 词归入覆盖扩展候选，不因“罕见”自动删掉。

筛选原则：

1. 优先当前图像可支持的观察、部位和属性，不把 glioma/meningioma/metastasis 等队列诊断默认当作视觉事实。
2. 同义表达合并到同一概念候选；同名多 CUI 必须结合语义类型、定义和 MRI 上下文消歧。解剖部位与侧别分子字段。
3. 增强相关候选限制为 T1C；信号强度最终需保留序列语境。DWI/ADC 等后续扩展不能未经序列适用性检查直接沿用当前核心词表。
4. MRDEF 定义通过对应的有效英文 AUI 关联筛选，保留原来源；若没有符合条件的定义，明确记录缺失。
5. 首批可以先选定义/类型清晰的解剖与形态概念及明确观察词；mass、lesion、edema、enhancement、midline 等需要优先处理歧义。
6. 显式 unknown/other 是字段控制项，缺失标签用 mask 表示；它们不能混为一个病理概念。
7. 审核之后再形成 concepts.jsonl 和冻结向量。按用户最新选择，第一版直接做 SapBERT+CLS+name+short-definition；name-only 和 BioLORD+CLS 留作后续独立对照。

本次完成下载、校验、编码加载检查和候选草案构建；没有把重建头、匹配损失或正式知识库接入训练，也未修改既有评测模型。

### 21.5 这 68 个候选是如何产生的

当前是 schema 驱动的种子草案，不是把全部训练报告切成 tokenizer token 后逐个映射 CUI，也不是已经完成的全语料医学实体抽取。

具体过程为：

1. 根据已讨论的四个角色和脑 MRI 描述范围，由本次设计起草 68 个种子条目，写入 seed TSV；这一步是项目设计选择，不等于临床审核。
2. 每条种子包含角色、子字段、若干候选搜索短语、序列范围和报告词面匹配规则。例如 lesion 同时搜索 lesion 和 brain lesion，edema 同时搜索 brain edema、cerebral edema 和 edema。
3. 在 BraTS 训练报告中计算这些规则的命中数，保留队列/序列统计与示例 ID。此步用于审核优先级，不会自动发现种子表之外的新术语，也不会识别肯定/否定及实例归属。
4. 将搜索短语做大小写、空白和连字符规范化，再与 MRCONSO 中 LAT=ENG、SUPPRESS=N 的完整术语字符串精确匹配。
5. 汇总命中的全部 CUI，补充英文同义词、语义类型和定义，形成待审候选表；不自动选择唯一 CUI。

这里需要保持 mass effect、white matter 等多词概念的整体；LLM tokenizer 的子词只是模型输入单位，并不等于 UMLS 概念。否定词、修饰词和序列语境还需要单独保留。

因此，68 条只是可检查的起点，不能宣称覆盖了训练语料中的所有重要医学概念。2026-10-15 已从全量训练报告独立抽取词面/医学片段并核查遗漏，详见第 22 节；正式概念审核尚未完成。完整性不应仅由出现频率决定，验证/测试报告不参与构建词表。

### 21.6 一个种子为什么会有多个 CUI，以及如何选择

CUI 标识概念的意义，不是每个英文字符串的唯一编号。相同意义的不同来源名称可以归到同一个 CUI；同一个字符串也可能对应不同意义。不能把多个 CUI 简单解释成“来自不同词表的同一个东西”。[NLM 概念与标识符说明](https://www.nlm.nih.gov/research/umls/new_users/online_learning/Meta_005.html)

当前 20 个多 CUI 条目还包含另一来源：一个项目种子合并了多个搜索短语，而这些短语可能有不同的概念粒度。例子来自本地 2026AA 候选结果：

| 种子 | 候选 CUI | 当前资料显示的含义 | 选择思路 |
| --- | --- | --- | --- |
| lesion | C0221198 | 泛指局部组织/器官病变，Finding | 适合泛指病变的概念身份 |
| lesion | C0221505 | 脑部病变，Finding；由 brain lesion 命中 | 若该条明确限定为脑部病变，可选更具体的身份；也可与通用病变拆成两个有层级关系的条目 |
| lesion | C1546698 | 标本来源编码相关，Intellectual Product | 排除作为 MRI 病变描述的候选 |
| midline | C0549183 | 空间中线，Spatial Concept | 空间角色候选；还需选择适合脑 MRI 的来源定义 |
| midline | C1660780 | 细胞成分，Cell Component | 排除作为脑部空间中线的候选 |
| enhancement | C1627358 | 屈光手术相关的增强/追加手术，Therapeutic or Preventive Procedure | 不作为 MRI 增强概念 |
| enhancement | C2349975 | 一般意义的改善/增强活动，Activity | 不作为 MRI 增强概念 |

edema 种子也同时收集了泛指水肿 C0013604、脑水肿相关 C0006114/C1527311，以及测量属性相关 C1717255。这些不是都要同时进入一个 query。对接近的脑水肿概念需要继续核对来源定义、使用范围和关系；本轮不把两者强行认定为完全等价，也不根据编号或匹配频率指定胜者。

建议使用两阶段筛选：

- **先排除含义不符的候选。** 检查角色/字段、语义类型、解剖范围、成像条件、定义和原文上下文。语义类型只是粗筛，同为 Finding 也可能分别指 MRI 所见与组织标本所见。
- **再在意义合适的候选中规范化。** 选择与项目概念粒度一致、来源和定义清楚的身份，固定来源版本与选择理由。不能一律选最具体的概念，例如不能把未说明病因的水肿直接解释成某一种机制。

若需要保留通用与特定概念，应拆成明确的本地 concept_id 或记录层级，避免把相近 CUI 当作多个独立患者事实反复监督。若现有候选全部不符，应继续同义词检索或保留 local ID；没有必须从当前列表中挑一个的要求。

还有一个重要的独立步骤是选定义：即使 CUI 已确定，其不同来源定义也不一定同样适用于当前任务。例如 C0549183 的本地资料同时包含身体/部位中线的描述和牙弓方向的描述。构建脑 MRI 向量时不能把所有定义机械拼接，应保留相关定义及来源，排除无关语境。

### 21.7 当前确定的名称＋简短定义编码方式

主版本按以下流程构建：

$$
\mathrm{text}_c=\mathrm{name}_c\ \Vert\ \mathrm{short\_definition}_c,\qquad
e_c=\operatorname{Normalize}
\left(E_{\rm SapBERT}(\mathrm{text}_c)_{\rm CLS}\right)
\in\mathbb R^{768}.
$$

这与只输入名称时使用同一种 CLS 汇聚方式，区别是 encoder 阅读的文本内容。输入一段解释并不要求输出多个 token；本版仍为每个概念生成一个向量。

简短定义应保留该概念的关键意义，避免引入当前患者信息、参考报告结论、未经图像支持的诊断推断或其他模态的解释。可从相关的原始定义选取/忠实缩写，保存 original_definition、short_definition、definition_source、source_version、编辑方式与审核状态。由项目自行撰写的 MRI 语境说明应标为本地说明，不得标成 UMLS 原文。

没有合适定义的候选进入待补描述队列；无法可靠定义的条目暂不进入正式名称＋定义主库，不静默改成 name-only。名称、定义模板、最大 token 长度、截断日志、概念顺序和模型 revision 均需固定。

后续 name-only 消融应在同一套概念 ID、同一 encoder 和同一 pooling 下去掉定义，才能检验定义的贡献；它不再是首轮实验的前置步骤。Projector 结构简化及这些辅助损失的首轮组合留待后续讨论。

---

## 22. 从训练报告出发核查候选覆盖（2026-10-15）

### 22.1 修正候选来源与覆盖判断

原 68 条是 schema 驱动的人工起草种子，之后才用训练报告统计命中数，不能代表全量训练医学词汇。本次按用户要求改变构建顺序：**先独立扫描训练原文，再提取候选、比较旧表，最后进行概念规范化和 CUI 审核**。

当前 `configs/base.yaml` 的训练 manifest 为 `single_sequence_findings_coref_v2/train.jsonl`，共 2,177 条报告：BraTS_MEN 656、BraTS_GLI 644、BraTS_MET 660、ISLES22 175、WMH 42。原 v1 只统计其中 1,960 条 BraTS 报告；本次同时保留全训练集和 BraTS 分层计数。当前 ISLES22 实际输入是 ADC，不能将其误称为输入 DWI；原四序列候选范围不能直接代表全训练集。

完整核查入口：[训练报告词汇核查](../assets/knowledge/bindformer_brain_corpus_audit_v1/REVIEW.md)。生成脚本：[audit_bindformer_report_vocabulary.py](../scripts/audit_bindformer_report_vocabulary.py)。抽取后整理的审核规则：[bindformer_corpus_review_rules_v1.tsv](bindformer_corpus_review_rules_v1.tsv)。这些规则用于解释独立抽取的结果，不是先用新的一组种子筛选语料。

### 22.2 已完成的独立抽取

| 产物 | 数量 | 含义与边界 |
| --- | ---: | --- |
| 英文词形清单 | 1,121 | 保留全部频次，包括单次出现与普通词 |
| 原文连续 1–8 词字符串 | 128,864 | 不跨标点构造短语；不是医学概念数 |
| 精确匹配有效英文 UMLS 术语的字符串 | 1,139 | 对应 1,903 个候选 CUI，包含普通词、缩写歧义及错误语义候选 |
| 本地 RaTE-NER 预测片段 | 12,230 | 3,682 种不同字符串，全部保留报告 ID、字符位置及上下文 |
| 抽取后的重点审核词族/字段 | 60 | 全部有训练原文证据，不是 60 个正式新概念 |

NER 没有截断报告，但会合并并列部位、漏检或误判断言状态，只作为候选来源。UMLS 精确字符串匹配与 NER 交叉补充；没有精确匹配的表达保留在剩余审核队列。没有读取验证/测试报告，也未把预测实体标签用于训练监督。

### 22.3 已发现的缺口及对 schema 的影响

- **新增部位与细分层级**：脑沟、脑回、脑池、脑裂、大脑镰、颅骨/颅底、桥小脑角、海马、外囊、胼胝体压部/膝部/体部、脑室角等。旧表可能只涵盖其上级概念，不能视为细粒度覆盖。
- **旧规则误归并**：`frontal bone` 被 `frontal_lobe` 的 `frontal` 规则命中，`cerebellopontine angle` 被 `cerebellum` 的前缀规则命中。必须保留完整医学短语再判断概念。
- **词形/同义遗漏**：`iso-intense / iso-intensity / iso-signal / isotense` 共 47 条报告，旧等信号规则均未命中；`edematous` 出现于 18 条，其中 17 条旧水肿规则未命中。词面漏计优先通过别名归并修复，不盲目增加重复 CUI。
- **字段缺口**：散在分布、程度、偏移方向、相对位置、实性/囊实性组成、具体数量和尺寸。可继续使用四个角色，但不能假设最初的子字段已经足够；程度需绑定其修饰对象，偏移方向不能等同病变侧别。
- **跨队列/序列扩展**：`restricted diffusion / diffusion restriction` 出现在 17 条 ADC 训练报告中；半卵圆中心、放射冠等低频部位也必须保留审核，不能仅按 BraTS 的频数建库。

### 22.4 从“出现过的词”到“可用的知识向量”

原文有医学词汇，并不意味着它是该患者的肯定视觉事实。例如 `edema or gliosis`、`possible bleeding or calcification`，以及血脑屏障和肿瘤类型的解释性描述，应保留不确定性和推断性质。`not mentioned` 与正常/阴性分开；尺寸用数值字段承载，不为每个数字创建概念向量。

正式构建顺序更新为：

```text
全部训练报告
  → 原文短语 + 医学实体候选 + 未匹配词形清单
  → 拆分并列结构，保留断言、关系、序列和原文位置
  → 区分新概念 / 旧概念别名 / 子部位 / 字段信息 / 解释性文本
  → 结合完整上下文、角色、语义类型与定义审核 CUI 或 local ID
  → 名称 + 简短定义 + 来源/版本/序列适用性
  → SapBERT 最终 CLS + L2 normalize
```

本次完成训练词面/候选的全量抽取和重点缺口核查，尚未完成全部候选的医学语义审核，因此不宣称“全部医学概念已覆盖”，也不以 68/1,139 等不同单位的数量计算覆盖率。原 v1 留作对照，正式向量库需基于此轮报告驱动的候选审核后生成。Projector 主结构及损失组合继续按此前安排另行讨论。

---

## 23. 已生成的 v2 候选库与离线向量（2026-10-15）

已按第 22 节的流程生成 [bindformer_brain_candidates_v2](../assets/knowledge/bindformer_brain_candidates_v2/README.md)，包含规范概念、原文提及、上下文字段、显式 CUI/定义选择、剩余审核表和可直接读取的 SapBERT 向量。

### 23.1 固定规模与接口

主矩阵为 $E\in\mathbb R^{128\times768}$：100 个 UMLS 关联条目、20 个本地概念、8 个角色级 unknown/other 控制项。主角色分布为 finding 15、location/laterality 78、signal 14、morphology/extent 21。mild/moderate/marked 在 finding、signal、morphology 三个角色共享，因此实际候选集合使用 `role_mask` 决定，不能只依赖主角色列。

输入固定为 `name + ". " + short_definition`，使用冻结 SapBERT `090663c3ae57bf35ffe4d0d468a2a88d03051a4d` 的最终层 CLS，经 L2 归一化后保存 float32。最长 38 个 token，未截断。所有短描述均明确标为项目编写的任务说明，原始 UMLS 定义和来源另存。

输出包括 `concept_bank.safetensors`、`embeddings.npy`、`role_mask.npy`、`sequence_mask.npy`、`scope_mask.npy` 和 `concept_index.json`。矩阵分别为 `[128,768]`、角色掩码 `[4,128]`、序列掩码 `[5,128]`、词表范围掩码 `[2,128]`。按角色、序列、词表范围取交集后，得到式（9）所需候选集合及概念矩阵。

### 23.2 必须明确的实验词表范围

第 22 节的 2,177 条是训练 manifest 全量；当前 `configs/base.yaml` 的 `data.datasets: [brats]` 实际选择 1,960 条 BraTS 报告。v2 对此提供：

- `brats_train`：120 条，包括 8 个控制项，作为当前 BraTS 主实验默认范围。
- `all_train`：128 条，明确使用全部训练队列词汇时采用。

仅在非 BraTS 训练报告观察到的 8 个条目为半卵圆中心、大脑脚、放射冠、大脑皮层、内囊、海马旁回、楔前叶、扩散受限。该掩码表示语料构建范围，不是生物学适用性判断；序列适用性由另一个 `sequence_mask` 表示。未使用验证或测试报告。

### 23.3 概念和上下文的实际边界

四角色继续保留，子字段扩展到结构效应、区域描述、相对位置、偏移方向、信号均匀性、扩散、分布、组成和程度等。脑叶与同名颅骨、桥小脑角与小脑、胼胝体各部、整体脑室与具体脑室角分别处理。

提及文件保留完整原文位置、分句、序列、断言线索及文本对象候选。并列部位按共同中心词拆分；多个候选对象或跨句共指不明确时保留歧义。`not mentioned`、否定、不确定、正常性分开记录，尚不将这些启发式关联作为 tuple/binding 真值。

尺寸、数量和关系线索存入上下文字段，不为每个数字创建一个向量。胶质增生、钙化的鉴别表达、血脑屏障/血管机制、疾病推断保留在解释性审核队列。130 个规范规格中，中央前回和侧脑室颞角因无可靠独立训练词面依据暂缓；完整未处理词面与 NER 片段继续保留，不宣称语义覆盖率达到 100%。

### 23.4 审核与可复现性

术语规格：[bindformer_v2_spec.py](../scripts/bindformer_v2_spec.py)。CUI/定义决策：[bindformer_v2_cui_decisions.json](bindformer_v2_cui_decisions.json)。生成入口：[build_bindformer_knowledge_v2.py](../scripts/build_bindformer_knowledge_v2.py)。

本次审核是概念身份、来源和项目 schema 层面的研究审核，不是假定完成临床专家标注。所有选定 CUI、来源定义 AUI、版本、别名和选择理由可追溯；不可靠的映射保留 local ID。例如 perilesional 排除给药途径 CUI，enhancement 排除一般改善/屈光手术含义，扩散受限的相关影像定义作为支持资料保存而不强配不合适的概念类型。

验证包含原文字符位置、训练集范围、CUI/定义引用、掩码与向量索引、有限值、L2 范数、文本无截断，以及易错部位/并列/断言/数值的回归用例。知识库交付时尚未修改 Projector；现已按第 25 节完成首版接入。

## 24. 首版最小 Projector：槽 + 知识引导 + 共享支持域，只训练生成 CE

### 24.1 已确定的范围与仍属建议的配置

用户已确定：首版直接使用 name + short definition 的 SapBERT CLS 向量；训练只用报告生成 CE，其他 loss 后加。本节结构现已实现；预算按后续确认改为压缩率模式与全局占比 0.25。超参数尚未经过效果选优。

v2 的 128 条是记忆库条目数：100 条关联 UMLS CUI、20 条本地概念、8 条 unknown/other 控制项。当前 BraTS 训练默认采用 `brats_train`，允许其中 120 条；再根据角色与输入序列筛选。128 不是槽数、输出 token 数或病灶类别标签数。资产已核验 `[128,768]`、有限值、L2 范数与文件哈希，但研究审核不等于临床标注，语料覆盖也未证明达到 100%。

首版核心机制：**先按当前图像形成证据槽，再用槽和角色检索概念，形成角色 query；同槽的角色 query 在共享支持域内读取视觉内容。** 这是采用 query 压缩思想的自定义 projector，不要求下载或复用完整 BLIP-2 Q-Former 权重。

### 24.2 全方案的模块划分

| 大模块 | 首版范围 | 后续扩展 |
| --- | --- | --- |
| 输入适配 | LN、768→d、基础 3D 网格 PE、序列 embedding | 物理坐标 PE、跨几何验证 |
| 证据分组 | 可学习初始槽、竞争归属、加权汇聚、GRU/残差 MLP、背景槽 | 迭代次数/背景槽/容量消融 |
| 知识引导 | 冻结 E、角色/序列/语料范围掩码、软检索、可训练映射 | name-only、随机固定 E、其他文本 encoder |
| 角色化视觉读取 | 四角色 query、共享归属 bias、一层视觉 CA、逐 token FFN | 不共享支持域、更多读取层 |
| 全局补充与输出 | 小型全局 CA、拼接、输出 MLP 到 decoder 宽度 | 全局与局部的预算分配 |
| 支持域几何统计 | 关闭 | 中心、协方差、软有效体积及几何 MLP |
| token 交互 | 关闭额外包内 SA 和局部读全局 | 分别开启，不同时加入 |
| 训练监督 | 仅 Findings 自回归 CE | X 特征重建、L_text、可靠标签下的 tuple/binding/concept |

严格的结构核心可以只用槽、知识角色 query、视觉读取和输出头；但首个 Findings 版本建议保留一个浅层全局分支，为整体结构和正常/弥漫所见提供读取路径。全局分支不保证自动承担这些语义，仍需通过实验检验。

### 24.3 建议的前向与张量接口

固定 `d=768`、`R=4`，M 由压缩率确定，G/K 按全局占比 0.25 求最近整数解，始终满足 `M=G+K*R`。例如压缩率 32：`M=275,G=67,K=52`。内部额外一个背景槽；K 是容量，不代表真实异常数量。所有参数在模型构造时创建。

1. **输入适配。** 接收 `X[N,768]`、`coords[N,3]` 和允许的输入序列 ID，形成 `H[N,d]`。首版建议使用归一化网格中心的三维 Fourier/sinusoidal PE，而不是等待完整几何统计模块。仅在方向与轴排列契约一致的 BraTS 设置中使用，并校验 `oriented`、`array_zyx_to_dhw` 等元数据。网格中心表示相对位置，不能解释成毫米坐标；后续替换物理 PE 时须追踪 affine、padding 和有效视野。
2. **槽路由。** 从共享的可学习初始槽出发，对每张图独立运行 `slot_iterations=3`：patch 在 `K+1` 个槽之间 softmax → 按槽归一化汇聚 → GRU/MLP 更新。最终重算 `A[N,K+1]`，取非背景的 `S[K,d]`。三轮仅发生在 projector 内部，各轮复用参数和视觉投影。
3. **知识检索与角色 query。** 从源矩阵 `E[128,768]` 按 `brats_train` 先实际筛选为 `[120,768]` 冻结 buffer；另外两个队列独有的八条向量不进入在线投影。在线再取 `sequence_mask & role_mask`。按式（9）从槽检索概念，按式（10）构造 `Q[K,4,d] = slot_projection + knowledge_projection + role_embedding`。首版无几何项，知识投影和角色 embedding 随 CE 训练，E 本身冻结。候选为空或序列缺失时明确报错。
4. **角色视觉读取。** 使用 `content_logits + beta * log(eps + A)`，沿 patch 轴 softmax，产生 `B[K,4,d]`。初值建议固定 `beta=1`，`beta=0` 是独立消融。局部 token 的 value 来自 H；首版不将概念向量作为额外视觉 value，也不把知识 query 通过直接残差绕过视觉读取送往输出。视觉读取结果可接逐 token 的残差 FFN。这里可复用现有 attention 运算，但不能原样使用含 query 残差、无支持域 bias 的 QueryBlock 作为全部局部逻辑。
5. **全局与输出。** 默认一层全局 CA 得到 `Bg[G,d]`；按 `[globals, slot_1 的四角色, ..., slot_K 的四角色]` 拼接为 `[M,d]`，经共享输出 MLP 得到 `Z[M,4096]`。角色顺序固定，槽不按参考报告排序；全局与局部暂不做额外交互。
6. **Decoder 与训练。** 复用现有 `LlamaDecoder.prepare()`，在图像占位位置插入 Z；沿用原 prompt、报告目标与生成流程。训练 CE 只监督答案 token，提示、视觉位置和 padding 不计入目标。推理无需 SapBERT、UMLS 原始文件或报告检索，不新增知识文本 token。

槽路由、知识检索和属性读取均在 decoder 之前完成。增加槽迭代不会让 8B decoder 为每轮重新运行；该结构也不直接解决 8B 自回归解码的逐 token 速度问题，应分别记录 projector、prefill 和 decoding 时间。

### 24.4 唯一训练目标与可验证边界

首版 `L = L_RG = CE(预测 Findings, 参考 Findings)`，保留当前按有效答案 token、梯度累积窗口和 DDP 归一化的方式。训练主设置建议 `training.stage=a`：冻结 NeuroVFM 与 LLM，仅训练 projector；冻结 LLM 的前向仍保持对 Z 可微。

可训练项包括输入映射、位置编码中的可选投影、序列 embedding、初始槽、路由映射/GRU/MLP、概念检索映射、角色 embedding、视觉 CA/FFN、全局 queries/CA 和输出头。冻结项包括视觉 encoder、SapBERT 概念向量和 LLM 参数。

不需要事实组 sidecar、Hungarian matching、患者 CUI 标签或病灶分割 mask。按后续确认已预留 `ProjectedVolume.auxiliary`，默认 None；只有 `return_auxiliary=true` 才返回可反传的结构化中间张量。图像—报告配对和已有序列元数据即可训练。

CE 可沿 `LLM → Z → 视觉读取 → query/槽/知识映射` 传播，但只约束最终文本预测：它不保证槽对应真实病灶、每个角色已经得到预期语义、概念权重是诊断概率，或属性绑定已被证明改善。首版验证“结构与知识先验能否改善生成”；后续机制评价和辅助目标负责更细的检验。可记录槽归属量、槽间相似度、检索熵与输出变化，不因 attention 图好看就认定绑定成立。

### 24.5 首轮消融与之后的增加顺序

首轮各组均使用同样的 CE、数据、输入特征、冻结策略、M、prompt、训练预算和选优规则。

| 实验 | 结构 | 主要问题 |
| --- | --- | --- |
| A：query 参考 | 现有 AttentionPool/Perceiver，固定同 M | 普通 query 压缩的参考水平 |
| B：槽结构 | 槽 + 四角色 + 共享支持域 + 全局，知识关闭 | 整体结构方案相对 A 是否有效 |
| C：首版完整最小模型 | B + v2 name/definition 概念检索 | C 对 B：知识引导是否提供增益 |
| D：无支持域约束 | C 保持所有其他模块，beta=0 | C 对 D：共享支持域是否提供增益 |

A 对 B 同时改变 query 组织和路由，属于结构整体对照，不能将全部差异归因为单独的槽迭代。若后续需要分离各项，应增加同角色组织的自由 query 对照，并匹配可比的宽度、深度和输入条件。参数量与实际计算成本同时报告，固定 M 不等于计算量完全相同。

之后先做 `T=1 vs 3`、背景槽开关，以及 name-only/随机固定 E 对照；再分别加入物理 PE、几何统计、包内 SA、局部读取全局。辅助目标从 CE 基础上逐项比较 `CE+L_feat`、`CE+L_text`，必要时再验证组合。tuple/binding/concept 仍作为需要可靠监督条件的后续方案。

去掉 G 个全局 token 会改变预算；如从 `(G,K,R)=(32,56,4)` 改成 `(0,64,4)` 保持 M=256，必须说明同时增加了槽容量，是预算重分配实验。更改 M、R 或 K 同样需明确容量变化。

现有 `run_local_packer3d.sh` 等通过 `baseline_sweep` 启动 joint_lora，且不同方法相同压缩比例可能输出不同 M。不能把它们的旧结果直接作为首版 Stage A 的严格消融；应统一冻结策略并记录各方法实际 M，另做相同 M 的 query 对照。

### 24.6 本轮实际需要的代码接入范围

建议新增 `src/models/projectors/bindformer/`，按职责拆分为 `input_adapter.py`、`slots.py`、`concept_memory.py`、`role_queries.py`、`reader.py` 和 `projector.py`；全局 CA 和输出映射优先复用现有公共模块。物理统计、交互、FactHeads 和辅助 losses 等文件后加。

| 接入点 | 首版改动 |
| --- | --- |
| `projectors/registry.py`、`specs.py` | 注册 bindformer3d；校验 M/G/K/R、宽度、迭代数、支持域与知识模式 |
| `models/multimodal.py` | train/generate 一致地传入 `Sample.sequence`；构造独立 context/metadata 副本，不修改缓存，不传报告/分割路径 |
| 新 projector | 仍返回现有 `ProjectedVolume(embeddings, input_tokens, output_tokens, metadata)`；metadata 只含可序列化摘要 |
| `training/checkpoint.py` 的 provenance | 纳入向量库、ID 顺序、掩码与 manifest 内容哈希；E/masks 用 buffer 随 projector state 保存，恢复时验证兼容性 |
| 新配置与启动脚本 | 与所有 baseline 共用 `joint_lora` 与确定性训练 preset，按 ratio 解析 M；`global_fraction=0.25`、`knowledge_mode=name_definition`、`vocabulary_scope=brats_train` |
| 当前 Trainer/Decoder CE | 沿用 loss 与归一化，不新增辅助损失接口 |

原第 10 节的全部公共接口改动是完整模型计划，不是首版必须同时完成的工作。首版只需把生成链路接通，并保留模块开关。未知/未来配置不应假装已经支持。

实现验收包括：输出符合解析后的 M；支持域 softmax 维度正确；beta=0 退化为普通视觉读取；知识掩码确实排除不适用条目；CE 梯度到达启用的 projector 分支而冻结参数不更新；checkpoint 恢复一致；新图像不继承前一张的槽状态；原有 baseline 路径兼容。实际测试与有限端到端更新记录见第 25 节及独立验收文档。

## 25. 已实现版本与运行接口（2026-10-15）

实现入口：[BindFormer3D](../src/models/projectors/bindformer/projector.py)。参数：[bindformer3d.yaml](../configs/projector/bindformer3d.yaml)。验收：[BINDFORMER_IMPLEMENTATION_ACCEPTANCE.md](BINDFORMER_IMPLEMENTATION_ACCEPTANCE.md)。

### 25.1 确认后的默认配置

| 参数 | 默认值/解析方式 |
| --- | --- |
| `target_compression_ratio` | 启动脚本依次运行 4、8、16、32，每档独立初始化 |
| `global_fraction` | 0.25；M=ceil(N/ratio)，G/K 取最近合法整数解 |
| `roles` | finding、location_laterality、signal、morphology_extent，固定顺序 |
| `inner_dim / num_heads / ffn_expansion` | 768 / 8 / 4 |
| `slot_iterations / background_slot` | 3 / true，背景槽不输出 |
| `position_mode / position_frequencies` | grid_fourier / 8，归一化网格轴通过方向矩阵旋转到 LPS 方向；不是毫米尺度 |
| `sequence_condition` | true，仅从输入样本序列 ID 传递 |
| `knowledge_mode / vocabulary_scope` | name_definition / brats_train |
| `support_beta` | 四角色均为 1，允许标量或四元素列表，0 去掉支持域偏置 |
| `global_depth` | 1，可配置堆叠层数，各层参数独立 |
| `reader_chunk_size / checkpoint_blocks` | 128 / true；分块不改变完整 patch softmax 或 token 数 |
| `head_type / hidden_size / dropout` | mlp / 4096 / 0；输出 MLP 为 768→4096→4096 |
| 训练 | 与其他 baselines 相同的 `joint_lora`：projector warm-up → projector+视觉 LoRA warm-up → joint cosine；LLM 冻结；仅报告 CE；projector/encoder LoRA 峰值 LR 均为 1e-5 |
| `return_auxiliary` | false；开启返回独立 dataclass，不把张量写入 metadata |

N=8775 时：

| 压缩率 | M | G | K | 局部 token |
| --- | --- | --- | --- | --- |
| 4 | 2194 | 550 | 411 | 1644 |
| 8 | 1097 | 273 | 206 | 824 |
| 16 | 549 | 137 | 103 | 412 |
| 32 | 275 | 67 | 52 | 208 |

2026-10-16 更新：geometry、within-slot SA、local-reads-global、`L_feat` 和每角色多 query 已实现，全部保持关闭或原始默认值；详见第 26 节。`L_text`、tuple/binding/concept loss 仍未实现，非零权重会明确报错。

### 25.2 BraTS 专用知识限制

源资产保留完整 v2 文件以便版本追溯；运行时先筛出 120 条 BraTS 范围条目，只给它们分配在线检索权重。排除的八条为 `centrum_semiovale`、`cerebral_peduncle`、`corona_radiata`、`cortex`、`internal_capsule`、`parahippocampal_gyrus`、`precuneus`、`restricted_diffusion`。共享概念和八个角色级控制项保留。

120 条再按角色和序列筛选；例如 T2W 不允许增强相关候选。名称、定义、向量、ID 顺序及掩码通过 manifest 哈希校验，checkpoint 保存冻结 buffer 和资产指纹。同一路径内容改变也拒绝恢复，避免悄悄换词表。

### 25.3 启动与消融

在项目根目录运行。先做数据头、预算与知识资产预检，不加载大模型：

    bash scripts/baselines/run_bindformer3d.sh --dry-run

GPU 3 上运行默认的 8×、16×、32×；每档独立使用 joint_lora 训练，随后评估 best checkpoint：

    bash scripts/baselines/run_bindformer3d.sh

单档和消融示例：

    bash scripts/baselines/run_bindformer3d.sh --ratios 32
    bash scripts/baselines/run_bindformer3d.sh --ratios 32 --knowledge-mode none
    bash scripts/baselines/run_bindformer3d.sh --ratios 32 --support-beta 0
    bash scripts/baselines/run_bindformer3d.sh --ratios 32 --global-fraction 0.125
    bash scripts/baselines/run_bindformer3d.sh --ratios 32 --slot-iterations 1
    bash scripts/baselines/run_bindformer3d.sh --ratios 32 --set projector.params.background_slot=false

`run_bindformer3d.sh` 默认运行 8×、16×、32×并使用物理 GPU 3；`run_bindformer3d_2.sh` 默认运行 4×并使用物理 GPU 1。两者使用与其他 baseline 相同的 `joint_lora` 配置、训练日程、完整 BraTS test 和 `all` 指标，依次执行每档训练与 best checkpoint 评测。外部 `CUDA_VISIBLE_DEVICES` 可覆盖脚本默认卡号。输出写入新的 `runs/baselines/bindformer3d_joint_lora_<tag>/`，保存 plan、解析配置、逐档训练/评测日志、checkpoint、预测与指标。默认 BraTS-only，训练集 1960、验证集 268、test 560；不包含 ISLES/WMH。

入口固定 BraTS 数据范围和 `brats_train` 词表，训练阶段固定为 `joint_lora`。每个压缩率从预训练视觉 encoder/LLM 独立初始化，不从另一压缩率或旧 Stage A checkpoint 续训。中断后用 sweep 的 `--resume-run <run-dir>` 恢复；Trainer 的 `last.pt` 续训仍要求使用原解析配置和同一实验设置。

2026-10-15 曾误按 Stage A-only 完成 4×、8×、16×并训练至 32× step 925；这些正式实验目录已按用户要求删除，工程验收目录保留。后续修正入口为 joint_lora。重跑状态和结果以 `runs/baselines/bindformer_joint_lora_gpu{1,3}_20261015T211700Z/sweep_status.json`、逐档日志和 `metric.json` 为准。工程检查通过不等于生成质量或实例绑定效果已经得到验证。

## 26. 可选结构与特征重建实现（2026-10-16）

本节记录当前代码行为；前文的 `L_text`、tuple/binding 等仍是研究设计，不代表已经启用。详细验收和待决策项见 [BINDFORMER_EXTENSIONS_ACCEPTANCE.md](BINDFORMER_EXTENSIONS_ACCEPTANCE.md)。

### 26.1 默认行为与模块路径

默认仍为 CE-only 最小 BindFormer，四个角色各 1 query、槽迭代 3 次、目标全局比例 0.25、BraTS 范围知识库 120 条。两个运行脚本的默认参数没有修改。新组件关闭时不创建其参数、不消耗初始化随机数；已与修改前的模型参数、输出、梯度和 RNG 逐项精确比较，并核对正在运行的 r8 实验的 provenance contract 保持一致。

```text
X + coords + 序列
  → InputAdapter → CompetitiveSlotRouter（3 次）→ slots、A
  → 每槽每角色检索 UMLS 概念
  → [可选] 由 A 和坐标计算支持域几何，编码到位置、形态角色
  → RoleQueries（按 J_finding/J_location/J_signal/J_morphology 展开）
  → 共享支持域约束的视觉读取
  → [可选] 同槽内 SA → [可选] 门控读取全局 token
  → 全局在前、局部按槽/角色/子 query 顺序排列
  → final_norm + 输出 MLP → Z → 冻结 LLM → CE
                            └→ [可选，训练/验证] 特征重建头 → L_feat
```

新增实现文件：`geometry.py`、`interactions.py`、`reconstruction.py`，位于 `src/models/projectors/bindformer/`。教师缓存实现为 `src/training/feature_targets.py`，导出入口为 `scripts/cache_bindformer_teacher.py`。`src/models/multimodal.py`、`src/training/loss.py` 和 Trainer 负责目标读取、损失聚合、日志及 checkpoint。

### 26.2 每个角色独立设置 query 数

参数 `projector.params.role_query_counts` 的四个键为：

```yaml
finding: 1
location_laterality: 1
signal: 1
morphology_extent: 1
```

每项必须为正整数，记每包长度 `Q = sum(J_r)`。根据压缩率先确定 `M = ceil(N / ratio)`，再求使 `|G/M - global_fraction|` 最小的合法整数解：

\[
M=G+KQ,\qquad K\ge1.
\]

要求全局分支时至少保留一个全局 token；等距时选择较多的全局 token。不会通过舍弃末尾局部 token 来凑预算。背景槽始终不输出。N=8775、32×、M=275 时：

| 四角色 J，按固定角色顺序 | Q | G | K | 局部 token |
| --- | ---: | ---: | ---: | ---: |
| 1,1,1,1 | 4 | 67 | 52 | 208 |
| 1,1,2,1 | 5 | 70 | 41 | 205 |
| 1,2,3,2 | 8 | 67 | 26 | 208 |
| 2,2,2,2 | 8 | 67 | 26 | 208 |
| 4,4,4,4 | 16 | 67 | 13 | 208 |

同一槽同一角色的子 query 共享角色 embedding、概念混合向量、几何向量及支持域，通过不同的可学习子 query 偏移区分，然后独立读取视觉内容。第一个 query 保留原表达式，其余 query 添加各自偏移；偏移在槽之间共享。不是复制读取后的 token。概念检索目前仍为“每槽每角色一次”，未扩展为“每子 query 独立检索”。`support_beta` 仍接受一个数或四角色列表，内部自动展开到所有子 query。

### 26.3 几何统计：不要求病灶 mask

`geometry_enabled=false`；启用后 `geometry_space=grid` 为默认。几何来自可学习归属 A 与预处理元数据，不读取分割标签。

| 模式 | 中心/协方差的坐标基础 | 体积摘要 |
| --- | --- | --- |
| `grid` | 有效 FOV 内 [-1,1] 坐标，经方向矩阵统一轴方向 | 有效 FOV 的软覆盖比例 |
| `voxel` | voxel 坐标，经方向矩阵旋转；不使用 spacing | 有效 voxel 数 |
| `mm` | bounded affine 得到 LPS 毫米坐标 | mm³ |

DHW→XYZ 轴序、padding 偏移和部分有效 patch 均被处理。毫米坐标使用 `bounded_before_pad.affine_xyz_to_lps`，该 affine 已包含裁剪原点，不重复加 crop_start。全 padding patch 不参加统计。中心与协方差按 `a_ik × valid_fraction_i` 加权；软体积按 `sum_i a_ik × valid_volume_i` 计算，避免对部分 patch 体积重复乘有效比例。

送入 MLP 的 10 维输入为：中心 3 维、协方差上三角 6 维、`log1p(soft_volume / volume_scale)` 1 维。voxel/mm 中心相对有效 FOV 中心平移，并除以 `geometry_coordinate_scale=100`；协方差相应除以该尺度平方，体积默认 `geometry_volume_scale=1000`。grid 模式已经归一化，两种 scale 固定使用 1。原始物理坐标不等于配准后的图谱坐标。

位置和形态角色各有一个 `10→128→768` GELU MLP，与 projector 同步训练；finding、signal 不加几何向量。`geometry_hidden_dim=128` 可配置。空支持域输出零。几何统计采用 FP32，包括 autocast 下的方向校验。现有输入 `grid_fourier` PE 保持原样，未增加毫米 PE。

### 26.4 包内 SA 与局部读取全局

- `within_slot_sa=false`，启用后 `within_slot_sa_depth=1`。每个槽的全部 Q 个 token 做 pre-LN 多头 SA + 残差，不跨槽混合；不同槽共享同一组层参数。
- `local_reads_global=false`，启用后 `local_global_depth=1`。Q 来自局部包，K/V 来自全局 token，残差由每槽 sigmoid 门控调节。门控输入为包内归一化 token 的均值，权重初始化为 0，偏置对应 `local_global_gate_init=0.1`；随后端到端训练。
- 顺序为先 SA，后读取全局。两种新增交互层不额外加 FFN；原属性读取和全局分支已有 FFN。深度可分别调整。
- 支持 activation checkpoint；确定性模式使用 math attention。新增交互的非确定性模式允许 flash/efficient/math，排除实测反向不兼容广播 K/V 的 cuDNN SDPA。

### 26.5 特征重建 L_feat

只有 `feature_loss_weight>0` 才创建和训练重建网络。默认权重 0，无重建参数、缓存读取或附加损失。目标不是 MRI 像素，而是压缩前 NeuroVFM 的 768 维特征。

**目标来源**：

- 默认 `feature_target=frozen_base_cache`：离线运行冻结、已预训练且不带 LoRA 的 NeuroVFM，生成固定 `X^T`，适用于当前 joint_lora。它是训练中的表征保留约束，不保证固定教师比适配后 encoder 更好。
- 可选 `frozen_input`：仅 Stage A 可用，直接使用当前冻结 encoder 的输出并 detach。LoRA/full 训练明确拒绝此模式；detach 不会使变化中的 encoder 成为固定教师。

**重建结构**：最终 Z 经 LN 和线性层降至 256 维作为 memory；归一化有效 FOV 坐标经小 MLP，再加序列 embedding 作为 query；一层 4-head CA + FFN（expansion=2），最后映射到 768 维。默认参数量 1,848,320。网络与 projector 同步优化，预测分支只读 Z、坐标和序列，不读原 X、槽归属 A 或参考报告，避免绕过压缩接口。

每张图最多选 `feature_samples=512` 个有效 patch，空间粗分区后不放回采样，不够则全部使用；`feature_query_chunk_size=128` 控制重建 query 分块，所有块仍读取完整 Z。采样使用私有 CPU generator，seed 由实验 seed、样本 ID、optimizer step 确定；验证固定 step=0，避免评估波动和污染模型 dropout RNG。

令 `w_i` 为 patch 有效比例、`S_b` 为采样集合，LN 无可学习参数，按通道独立处理每个 patch：

\[
\mathcal L_{\mathrm{feat}}^{(b)}=
\frac{\sum_{i\in S_b}w_i\,\frac1{768}\left\|
\operatorname{LN}(\hat x_i)-\operatorname{LN}(\operatorname{stopgrad}(x_i^T))
\right\|_2^2}{\sum_{i\in S_b}w_i},
\qquad
\mathcal L=\frac{\sum \mathcal L_{\mathrm{CE,sum}}}{\sum T_y}
+\lambda_f\frac{\sum_b\mathcal L_{\mathrm{feat}}^{(b)}}{B}.
\]

损失在 FP32 计算。梯度累积窗口和 DDP 分别使用全局回答 token 数、全局图像数作为两个分母，避免长报告或 microbatch 划分改变重建权重。日志增加 `feature_loss`、`feature_cosine`、`feature_patch_count`、`total_objective`。验证同时记录 CE 和重建指标；best checkpoint 与 early stopping 仍按原有验证 CE。

生成时不运行重建头，也不需要教师缓存；已通过强制让重建调用和缓存读取报错的方式验收。辅助头权重仍随训练 checkpoint 保存，生成时未执行它，未单独导出裁掉辅助头的部署权重。

**缓存与启用示例**（下列是按需执行的命令，本次没有导出全训练集或启动新正式训练）：

```bash
# 与正式实验保持相同预处理；脚本内部强制冻结基座、移除训练 LoRA。
python -m scripts.cache_bindformer_teacher \
  --config configs/training/joint_lora.yaml \
  --config configs/preprocessing/native_grid.yaml \
  --config configs/projector/bindformer3d.yaml \
  --output cache/bindformer_teacher --device cuda:0

# 新实验示例；0.1 只用于说明和工程 smoke，不代表已选出的最优权重。
bash scripts/baselines/run_bindformer3d.sh --ratios 32 \
  --set projector.params.feature_loss_weight=0.1 \
  --set projector.params.feature_target_cache=cache/bindformer_teacher
```

缓存记录基座权重指纹、预处理版本和输入 cache_key，逐文件验证特征形状、数值及 patch 坐标顺序。manifest 原子创建，目标文件原子写入；不匹配或缺失立即报错，不会临时改用当前 LoRA 特征。默认导出所选 train/val；不使用 test。每张完整 BraTS BF16 特征约 12.85 MiB，启用前需为完整训练/验证选集导出。

### 26.6 独立消融入口与尚未实现部分

```bash
bash scripts/baselines/run_bindformer3d.sh --ratios 32 --dry-run \
  --finding-queries 1 --location-laterality-queries 1 \
  --signal-queries 2 --morphology-extent-queries 1

# 以下每行分别代表一次新实验，可按需组合。
bash scripts/baselines/run_bindformer3d.sh --ratios 32 --set projector.params.geometry_enabled=true
bash scripts/baselines/run_bindformer3d.sh --ratios 32 --set projector.params.within_slot_sa=true
bash scripts/baselines/run_bindformer3d.sh --ratios 32 --set projector.params.local_reads_global=true
bash scripts/baselines/run_bindformer3d.sh --ratios 32 --knowledge-mode none
bash scripts/baselines/run_bindformer3d.sh --ratios 32 --support-beta 0
```

已有默认实验继续原样训练/恢复。改变 J、结构或损失需要创建新实验；严格 checkpoint 检查不会把旧优化器状态静默套到不同结构，暂未实现旧最小模型到新增分支的部分权重迁移工具。

按照本轮要求，`L_text` 的正负候选构建、对比损失均未实现；没有额外创建所谓“仅正例对齐”的替代目标。tuple/binding/concept loss、毫米 PE、每子 query 独立知识检索等仍为后续工作。
