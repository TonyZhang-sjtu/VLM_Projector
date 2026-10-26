# Projector Baseline 128× / 256× 实现（2026-10-26）

本扩展沿用第 15 节的方法原生预算协议。输入为 BraTS 的 `native_grid + minimal_pad` NeuroVFM 特征：patch 网格 `DHW=[39,15,15]`，物理范围 `[156,240,240] mm`，共 `N=8775` 个 token。每个输入 patch 的物理范围约为 `4×16×16 mm`。聚合对象是 Encoder 输出的 patch 特征，不是原始 MRI 像素。压缩误差定义为 `abs((N/M)/目标倍率 - 1)`，上限为 5%。

| 方法 | 目标 | 请求 `ceil(N/r)` | 实际输出网格／固定块 DHW | M | N/M | 误差 |
| --- | ---: | ---: | --- | ---: | ---: | ---: |
| AvgPool / ConvPool / Mixer / LocalPacker | 128× | 69 | 输出网格 `[3,4,6]` | 72 | 121.875× | 4.7852% |
| AvgPool / ConvPool / Mixer / LocalPacker | 256× | 35 | 输出网格 `[3,4,3]` | 36 | 243.750× | 4.7852% |
| Concat | 128× | 69 | 固定块 `[13,2,5]`，输出 `[3,8,3]` | 72 | 121.875× | 4.7852% |
| Concat | 256× | 35 | 固定块 `[13,4,5]`，输出 `[3,4,3]` | 36 | 243.750× | 4.7852% |
| BindFormer / AttentionPool / Perceiver-NV / Perceiver-Flamingo / AdaptiveMerge / TokenSelect | 128× | 69 | 方法原有精确计数机制 | 69 | 127.1739× | 0.6454% |
| 同上 | 256× | 35 | 方法原有精确计数机制 | 35 | 250.7143× | 2.0647% |

`N/r` 分别为 68.5547 和 34.2773，因此整数 M 无法精确得到目标倍率。MLP 仍只提供 1× 参考。方法间 M 可以不同；比较时报告实际 M 和 N/M。BindFormer 默认四角色各一条 query、`global_fraction=0.25` 时，128× 分配 17 global + 13 slots × 4，256× 分配 7 global + 7 slots × 4；实际 global 占比后者为 20%，属于整数束约束下的现有分配结果。

网格方法沿用不重叠、整数边界分桶。枚举误差 ≤5% 的输出网格后，以每个实际分桶的物理边长对数方差按其所含 patch 数加权，选择物理上较均衡的配置。128× 的 `[3,4,6]` 每个输出单元名义物理范围约 `52×60×40 mm`，256× 的 `[3,4,3]` 约 `52×60×80 mm`。H/W 对称候选并列，选用这组配置以使 128× → 256× 沿 W 轴两两合并。直接按请求 token 数使用旧 `nearest` 网格会分别得到 `[23,1,3]` 和 `[1,7,5]`，物理形状过于细长；因此新增 BraTS 固定网格预设，手动预算及旧档位的 `nearest` 规则保持原样。

Concat 仍以固定块拼接特征，只在各轴末端补最少空槽。枚举满足误差的固定块后，根据边界块真实覆盖范围的 patch 加权物理均衡度选出 `[13,2,5]`、`[13,4,5]`，完整块分别覆盖约 `52×32×80`、`52×64×80 mm`；H/W 互换候选等价，选用上述排列使 128× → 256× 沿 H 轴合并。两档各补 585 个空槽，占拼接容量 9360 的 6.25%；补空不增加 MRI 或 Encoder 的输入。按输入维度 768、映射头隐藏维度 4096、输出维度 4096 和 bias 计算，Concat 映射头约有 4.2573 亿、8.3467 亿参数。正式训练前须验证显存、有限梯度和短步优化，不能由预算预检推断报告质量。

本机为 24 GiB RTX A5500。现有训练配置把冻结的 Llama-3.1-8B 以 BF16 放在 GPU，且对 Projector 使用 AdamW；仅 Concat 256× 的 FP32 参数、梯度和两个优化器状态理论上约占 12.4 GiB，再加冻结 LLM 权重已超过单卡容量，尚未计入 Encoder 和激活。此配置是预算与模型结构的正式实现，但 **Concat 256× 全模型训练不能直接在本机单卡按原优化器设置执行**。128× 也需要实际显存验收。若要在本机完成 Concat 256× 训练，应另行确定降低映射头隐藏维度、参数/优化器分片或卸载等资源方案，并将其作为结构或训练配置变更单独记录。

实现位于 `src/models/projectors/budget.py`：两个倍率加入 `SUPPORTED_RATIOS`，上述 BraTS 网格与 Concat 固定块在建模前解析，并写入 `resolved_budget`。输入几何不符时拒绝套用固定预设。其他精确计数方法沿用 `ceil(N/r)`；`src.cli.baseline_sweep` 和 `src.cli.bindformer_sweep` 都接受新倍率。训练、best checkpoint 选择和完整 test 评测流程没有新增分支；每档仍独立初始化。

本地预检示例（按实际服务器设置 Python、路径及 GPU；`--dry-run` 不启动训练）：

```bash
bash scripts/ft_ablation/run_avgpool3d_constant_lr.sh --ratios 128 256 --dry-run
bash scripts/ft_ablation/run_concat3d_constant_lr.sh --ratios 128 256 --dry-run
bash scripts/baselines/BindFormer/run_bindformer_vanilla_constant_lr_1019.sh --ratios 128 256 --dry-run
```

自动化验收在 `tests/test_projector_budgets.py` 检查全部方法两档预算、配置保存/恢复、空间分组对 8775 个 patch 的无重叠全覆盖、Concat 边界容量、几何误用拒绝、轻量模型前后向和 checkpoint 契约。`--dry-run` 仍须对实际 train/val/test 清单、MRI 文件头与最终评测范围逐项核对；完整宽度 Concat 的 GPU 显存与训练质量需要另行实测。

本机验证（2026-10-26）：`tests/test_projector_budgets.py` 为 120 passed；`test_ratio64.py` 与空间几何联合回归在新增 sweep 入口检查前为 202 passed、16 skipped；sweep、BindFormer 预算、启动器和 Projector 契约联合回归为 179 passed。AvgPool、Concat、BindFormer 三种入口对 128×／256× 的真实清单 `--dry-run` 均通过，各档 train/val/test 为 1960/268/560，输出 M 分别符合上表。CUDA BF16 轻量模型用完整 8775 patch 网格验证了 AvgPool 128×、Concat 256×、AttentionPool 256× 的有限前后向梯度。尚未执行完整宽度 VLM 训练，也未产生正式报告质量结果。
