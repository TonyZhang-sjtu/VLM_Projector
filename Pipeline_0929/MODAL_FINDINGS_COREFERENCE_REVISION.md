# RadGenome-Brain 逐模态 Findings 指代修订记录

版本：`single_sequence_findings_coref_v2`。本轮修订已发布至活动 `annotations/*/modal_wise_finding.json`；没有改动图像、mask、患者划分、global findings 或 impression。这里的“校验通过”指文本修改与来源、ID、数值等一致，不代表影像或临床准确性已验证。

二次核对已将 6 条措辞收紧，发布为 v2。全部 220 条修改的 455 次原始证据引用没有跨患者或跨检查。6 条变化分为：两条撤回笼统引用下扩展的水肿/受压信息，两条将信号范围限定为病灶实性部分，两条避免把两个部位写成一个病灶。具体原文、v1、v2 和理由见 [double-check JSON](../assets/data/RadGenome-Brain/modal_wise_revision_doublecheck.json)。

v1 发布状态备份在 `backups/20261004T053047Z_before_modality_scope_v2/`（32 文件），原始数据集备份仍单独保留。当前 33 条复核报告没有自动从训练清单排除；其中 26 条原文未改、7 条局部修改，`requires_review` 不会自行跳过样本。详见主训练方案第 3.4 节。

## 1. 结果与使用版本

全量扫描 3,098 条非空逐模态 Findings，宽规则及既有审计合并得到 405 条候选，实际修改 220 条。候选包含正常段内指代和报告缺项说明，不能将 405 当作错误数。

| 来源 | 原有非空文本 | 实际修改 |
| --- | ---: | ---: |
| BraTS GLI | 920 | 59 |
| BraTS MEN | 920 | 81 |
| BraTS MET | 948 | 75 |
| ISLES22 | 250 | 0 |
| WMH | 60 | 5 |
| 总计 | 3,098 | 220 |

修改分布为 train 152、val 22、test 46。修订仅依据源标注，不使用模型预测或测试结果。当前有文本样本数仍为 **train 2,177 / val 299 / test 622**；本轮没有增加缺失的 250 条 ISLES DWI、60 条 WMH T1 标注。

后续加载器使用 `assets/data/RadGenome-Brain/training_manifests/single_sequence_findings_coref_v2/{train,val,test}.jsonl`，其 `finding` 与活动 JSON 一致。`single_sequence_findings_raw` 清单保留原文，仅用于追溯或原始参考答案的对照评分。新清单有 `target_version`、`coreference_revised`、`text_audit_status`；带 `requires_review` 的条目不能被误当成完成医学审核的目标。本轮未自动排除这些条目或改变划分。

## 2. 已执行的 pipeline

```text
备份源文件并计算 SHA-256
  → 扫描全部逐模态报告，建立同一次检查的证据组
  → 区分段内指代、冗余措辞、缺失指代对象、来源不足
  → 保存证据 JSON 与精确替换候选
  → 自动检查来源、编辑重放、数值、左右侧、信号/否定词
  → 写出独立版本的五个逐模态 JSON
  → 检查活动文件未发生意外修改
  → 覆盖活动逐模态 JSON，核验哈希
  → 生成逐条修改记录、待复核清单和同步训练清单
```

**备份先于修改。** 本次备份目录为 `assets/data/RadGenome-Brain/backups/20261004T044743Z_modality_coreference_v1/`。其中包含全部 15 个报告 JSON、split、manifest、原审计、raw 清单和修改前文档，共 26 个文件；`manifest.json` 记录路径、大小和 SHA-256。先前的患者划分备份另行保留。

**按检查 ID 找来源，不按 JSON 排列顺序找“上一条”。** BraTS 的患者 ID 与扫描编号共同组成检查 ID；只读取同一检查的 `modal_wise_finding` 和 `global_finding`。不从其他检查、其他患者或测试预测中借内容，不用 `impression` 补当前序列的影像表现。

**最小范围编辑。** 当前段已给出位置、数值或对象的，删除冗余 `aforementioned` / `same` 比较措辞即可。当前段缺少明确指代的部位、尺寸或结构影响时，只展开同次检查来源明确支持、且原目标确实引用的属性；信号强弱、强化与非强化、推断诊断不能随之跨模态复制。原文的“未描述”不能转成“正常”。源标注自身的医学推断不在本轮重新诊断或统一改写。

**来源不足时保留问题。** 自动校验无法判断图像中究竟可见什么。没有可靠来源就不编写；具体限制进入修改记录和 review queue。规则只是候选筛查；本版本的事实补全使用逐病例整理的精确替换表，不能将它视为任意新报告都可自动修好的通用系统。

## 3. 自动校验范围

- 备份清单逐文件哈希一致；证据原文和其 SHA-256 与备份相符，证据属于同次检查。
- 每个替换片段唯一匹配，按记录的中间文本位置重放后严格等于修订全文。
- 补入的三维毫米尺寸能在证据中找到；原有尺寸保留；已有左右侧词未被删掉。
- 目标信号描述词保留；仅 `Utrecht_11_FLAIR` 删除了附带的 T1W 信号比较，保留本条 FLAIR 高信号，并单独记录该例外。
- `no / without / not / absent` 的词数不减少。词数检查只能发现部分问题，不能证明否定的语义范围完全正确。
- 五个文件的 ID 集合和非空文本数不变；活动版本与独立版本哈希一致；新训练清单与活动目标逐 ID 一致。
- 再扫描严格外部指代规则：残留两条段内可自足表达和一条未解决外部引用；另有一条不被严格规则覆盖、但人工识别的来源不足案例，均列入复核。

自动检查不会认证病灶可见性、诊断正确性或标注绝对完整性。正式实验前仍应检查复核清单及预处理后的图像；所有 Projector 固定同一目标版本及复核处理策略。

## 4. 文件及修改记录格式

以下路径均相对 `assets/data/RadGenome-Brain/`：

| 文件 / 目录 | 内容 |
| --- | --- |
| `modal_wise_coreference_evidence.json` | 405 条筛查候选，原文、候选修订、同检查各序列及 global 上下文、修改证据；保留发布前 `proposed` 状态 |
| `modal_wise_revision_validation.json` | 校验结果、前后哈希、统计与发布后一致性 |
| `annotation_versions/single_sequence_findings_coref_v2/` | 独立修订版的五个逐模态 JSON |
| `annotations/*/modal_wise_finding.json` | 已覆盖为本版本的活动目标 |
| `modal_wise_revision_log.json` | 实际应用的 220 条记录及发布状态 |
| `modal_wise_coreference_review_queue.json` | 33 条仍带复核说明的文本，包含范围不足、跨序列用语和原始报告缺项等不同情况 |
| `training_manifests/single_sequence_findings_coref_v2/` | 与活动标注一致的 train / val / test 清单 |

单条记录包括如下字段；真实证据包含完整来源原文及哈希，不仅是示意中的路径：

```json
{
  "id": "BraTS-GLI-00006-000-t1c",
  "case_id": "BraTS-GLI-00006-000",
  "cohort": "BraTS_GLI",
  "original_text": "完整原文",
  "revised_text": "完整修订文本",
  "changes": [
    {
      "attribute": "referenced_structural_findings",
      "original_span": "原引用片段",
      "replacement": "有来源的展开片段",
      "offset_in_step_input": [0, 0],
      "reason": "修改理由",
      "evidence_keys": ["同检查的来源 ID"],
      "evidence": [
        {
          "file": "annotations/BraTS_GLI/global_finding.json",
          "key": "BraTS-GLI-00006-000",
          "case_id": "BraTS-GLI-00006-000",
          "quote": "来源原文",
          "sha256": "来源文本哈希",
          "span": [0, 0]
        }
      ]
    }
  ],
  "status": "applied_text_verified",
  "remaining_review_notes": [],
  "clinical_validation": "not_performed; textual scope/provenance review only"
}
```

`offset_in_step_input` 是每次编辑执行前文本的字符位置，`span` 是证据引用的字符范围，均使用左闭右开区间；示例中的 `[0,0]` 只是格式说明。一个报告可以有多个有序编辑。带剩余复核说明的已修改记录使用 `applied_text_verified_pending_clinical_review`。未修改候选只在 evidence / review queue 中，不冒充实际修改。

## 5. 实际修订例子与尚存限制

`BraTS-GLI-00006-000-t2f` 原文已包含左额叶和高信号，只把 `the aforementioned lesion` 改为 `the lesion`。

同次检查的 `t1c` 原文引用“上述尺寸及周围影响”，现在明确写出约 `76 × 72 × 69 mm`、周围水肿、双侧侧脑室前角受压及中线右偏；当前 T1C 的环形强化描述保持原样。这些展开内容来自同检查的 T1W / global 文字证据，不表示重新测量了该 T1C 图像。

两条确实不能可靠补全的标注继续保留原文：

- `BraTS-MET-00090-000-t2w`：原文明确说没有 T2W 信号信息，综合报告只支持强化结节，不能据此制造 T2W 表现。
- `BraTS-MET-00113-000-t2w`：原文是“likely visible”，尺寸引用也不足以确认有效 T2W 监督，不能改成确定可见。

另外三个引用仅处理有证据的部分：`GLI-00242-000-t2f` 没有可补的脑池细节；`MET-00266-000-t1c` 没有被引用的额外边界信息；`MET-00244-000-t1c` 的笼统非增强序列特征不能整体转入 T1C。删除不必要比较或保留缺项说明，同时记录限制。

review queue 共 33 条（train 28 / val 1 / test 4），其中 7 条相对原始数据集做了局部修改，26 条仍保留原始文本。它还包含 17 条非增强输入中 enhancement 相关词的候选和其他来源缺项，存在重叠；**33 不是 33 条未解决“同上”，也不等于 33 条已确诊错误。** 本轮不通过填写猜测内容、删除样本或隐式换用综合报告来掩盖这些问题。

## 6. 复现与回退

v1 修订脚本作为历史保留，当前活动版本已升级，其保护检查会阻止覆盖 v2。当前脚本：[doublecheck_radgenome_revisions.py](../archive/non_pipeline/scripts/doublecheck_radgenome_revisions.py)。从项目根目录运行：

```bash
# 从固定 v1 快照和原始源备份重新核对，生成独立 v2 与审计；不覆盖活动标注。
python archive/non_pipeline/scripts/doublecheck_radgenome_revisions.py --snapshot assets/data/RadGenome-Brain/backups/20261004T053047Z_before_modality_scope_v2

# 显式发布；此前已执行，不需要为了开始训练而重跑。
python archive/non_pipeline/scripts/doublecheck_radgenome_revisions.py --snapshot assets/data/RadGenome-Brain/backups/20261004T053047Z_before_modality_scope_v2 --apply
```

再次运行会重建同名审计文件，发布会更新时间。若后续人工改动活动标签，应另建备份与新版本，避免覆盖。

回退到 v1：核验 v1 快照清单后，恢复其 `assets/data/RadGenome-Brain/annotations/*/modal_wise_finding.json`，使用保留的 coref_v1 清单，并同步状态元数据。回退到最初数据集文本：使用 `backups/20261004T044743Z_modality_coreference_v1/annotations/` 的五个逐模态文件并切回 raw 清单。两种回退都不能只恢复 JSON 却继续读取另一版本的清单；本轮未执行回退。
