# Graph V3：明确位置事实与部分参考标签评价

本目录是独立、离线、CPU-only 的评价实验。不修改 V1/V2、`src`、训练脚本、权重、预测、官方 RadGraph 指标或 GREEN 工作流。V3 只重新读取已经保存的 RadGraph-XL 图及报告文本。

## 核心语义

- 评价单位是每份报告中的阳性位置事实集合，不是病灶实例/数量匹配，更不是影像分割或定位金标准。
- Reference 和 prediction 调用完全相同的提取器。提取器不接收方法名、压缩率、模型分数。
- 侧别、脑区、联合位置分别判断可评价性。参考在该维度有至少一个明确标签即可纳入，不要求整份报告全部事实完整。
- `(left, temporal_lobe)` 纳入三个维度；`(None, basal_ganglia)` 只纳入脑区。未知不填成 left/right/midline，也不产生 `unspecified` 联合标签。
- 缺失、否定、不确定的预测不能移除已知参考，已知事实漏报仍计 FN。额外明确位置通常计 FP。
- 参考明确给出部分位置时，缺失字段是“未知”，不是阴性：参考左颞叶＋未知侧别基底节，预测右基底节的侧别不被判对或判错；左颞叶上的错误仍计错。被屏蔽的预测在 `ignored_prediction_locations` 中留痕。
- 若参考某脑区已有明确侧别，另外的部分重复提及不能屏蔽该脑区的侧别矛盾。完全未知 `(None,None)` 不是任意预测的通配符。
- 同侧同部位的重复文本去重。相同部位 left+right 等价于 bilateral；单独 side 维度将全报告双侧涉及归一成 bilateral，joint 保留各脑区的对应侧别。

## 提取改进及边界

1. 仍以 RadGraph 的 Observation 病灶实体及 `located_at` 为锚点。在原图文本的 token 坐标上扩展受限解剖名词短语，恢复原来只读到 `junction` 而漏掉的额顶叶等信息。
2. 共享头词的并列修饰可展开：`left frontal and parietal lobes` → 左额叶、左顶叶。完整名词短语或逗号终止侧别传播：`left temporal lobe, insular region, basal ganglia area` 不给后两者补左侧。
3. 覆盖 `abnormal signal`、`signal focus`、`abnormality` 等明确病灶表达，以及颅窝、小脑幕等解剖名。小脑幕不自动转成小脑。
4. 图边缺失时，允许同一局部子句中的明确定语/定位句式恢复，例如 `left frontal lesion`、`lesion ... located in ...`、`left frontal lobe contains the largest lesion`。不做一般的“最近解剖词”匹配。
5. 病灶存在性和位置断言分开：`lesion is not located in ...` 不算阳性位置；`may be located` 不算确定位置。`lesion in left frontal lobe may represent glioma` 仍保留明确的病灶位置。
6. 邻接位置以 `adjacent_to/部位` 保留，不和“位于该部位内部”强行视为相同；被压迫结构不替代病灶位置。中线移位单独保存，不充当病灶侧别。
7. 不从其他患者、其他序列、其他句子或解剖常识补标签。跨句指代、词表外部位、复杂作用域与 RadGraph 错误仍可能造成漏判/误判；剩余未纳入病例全部导出。

## V2 / V3 的可比性

输出同时包含：

- `v2`：严格复现冻结 V2 的提取与评分；所有 57 个设置的 F1、EM 和分母必须与原工作簿逐项一致。
- `v2_partial`：保持 V2 抽取，应用 V3 的部分标签评分，分离纳入/评分变化与提取变化。
- `v3`：完整 V3 提取和评分。
- `common_v2_v3`：相同参考报告子集上的两个版本；提取标签及评分规则仍可能变化。
- `new_v3`：V3 新纳入的报告，单独报告。

因此分数变化是**评测协议变化**，不能当作模型自身变强。规则根据已发现的参考覆盖问题和语义反例改进，属于事后评价协议开发；不能宣称是预注册、完全未接触测试文本的独立验证。

## 计算与导出

```bash
cd /mnt/sda/jyzhang/VLM_Projector
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  /mnt/sda/mingjie/conda_envs/projector/bin/python \
  -m experiments.brain_location_audit_v3.run
```

已有输出时主动失败；复现时使用新的 `--output` 目录和 `--workbook` 文件，避免覆盖旧结果。

默认结果：`results/AnchorRead3D_GraphV3_1108.xlsx`，明细：`results/brain_location_graph_v3_1108/`。

- `Coverage`：覆盖数、患者数、新增和退出病例数。
- `V2_V3`：57 个设置的全子集、共同子集及新增子集对比。
- `V3_F1/EM/Precision/Recall`、`V3_ablations`：主实验与消融。
- `Visual_32x/256x`：Normal、三个 shuffle 均值、zero、text；逐 seed 另表。
- `Reference_audit`、`Reference_excluded`：逐条参考原文、两版标签、标记及证据。
- `V3_paired_CI`：患者簇配对 bootstrap，10,000 次、seed 20261108，未做多重比较校正；不包含提取器误差。
- `extractions.jsonl`：按图哈希缓存的原文及 V2/V3 提取证据；`scores.jsonl`：58,800 个报告配对的逐条计数、忽略事实和哈希。
- `tests.txt`、`stress_probes.json`：语义测试和 28 组冻结 RadGraph 图探针。探针通过率不是临床人工金标准准确率。
- `source_integrity.json`、`status.json`：不可变输入的前后哈希、版本、完成状态和工作簿哈希。

Visual Dependence 使用已有冻结生成文本/图；不重新推理。原 Visual Dependence 工作簿可能由后台 GREEN 流程正常更新，V3 不写入或锁定它。

## 保留原表其余指标的 V3 替换副本

```bash
python -m experiments.brain_location_audit_v3.replace_workbooks
```

输出 `results/AnchorRead3D_GREEN_GraphV3_1108.xlsx` 和 `results/AnchorRead3D_RG_VisualDependence_GraphV3_1108.xlsx`。
仅替换自定义位置分数、参考覆盖、配对区间、位置变化率、shuffle 标准差、相关图表和协议说明。原表的 GREEN、官方 RadGraph-XL、通用文本指标、CE、文本退化诊断、checkpoint 索引全部保留。

导出前逐项复现旧表位置分数，导出后逐单元格验证所有非位置数值、公式及格式未变，并记录源/输出 SHA256。两张源表只读；目标已存在则失败。
`green_1108_replacement_audit.json` 与 `visual_1108_replacement_audit.json` 保存每个修改单元格及保留检查。

Visual 源表在本次导出时仍标记 GREEN 待完成；V3 副本不自动跟随后台更新。GREEN 完成后可用新后缀重新导出最新快照，例如 `--suffix 1108_green_complete`，不需要重跑 V3 提取或模型。

用户贴出的旧分析已更新为 `Pipeline_0929/GRAPH_V3_PAPER_ANALYSIS_1108.md`；数值变化是协议与覆盖范围变化，不是模型变化。关键结论修订包括：32× 均值直通的联合位置收益区间跨零，Text-only 与其他辅助干预之间不再具有固定排序。
