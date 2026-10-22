# BindFormer：Decoder 患者敏感性、四角色分工与 LLM LoRA

这是对已训练 16× 固定区域 VA/SA+CA checkpoint 的进一步诊断，未修改训练框架、未训练 LLM LoRA。

完整分析、文献一手来源、表格和实验细节见：

- [Markdown 诊断报告](../results/BindFormer_decoder_sensitivity_r16/decoder_sensitivity.md)
- [HTML 诊断报告](../results/BindFormer_decoder_sensitivity_r16/decoder_sensitivity.html)
- [实验脚本及原始数据](../results/BindFormer_decoder_sensitivity_r16/)

## Overview 与术语

支持图是 `A[:,k]` 或其列归一化后的 `W[:,k]`，即一个 slot 对输入 patch 的支持分布；对应 Overview (b) 紫色路由框中的 A。它不是病灶 mask，也不是角色实际 attention。固定区域 mean 规定 router 的区域均值 evidence，Reader 在区域内仍有 query-dependent attention。

Reader 对应 Overview 的 **Support-biased visual read**，后接可选的 **Within-slot interaction** 和最终 LN/MLP。原 Overview 没有绘入后加的固定区域开关。

## 新实验证据

两模型使用相同的 9 位验证患者、T1C/FLAIR 两序列，共 18 volume/模型；GPU 仅使用物理 3 号卡。完成 360 次 CE 干预前向、96 次自由生成、36 次角色路径探针，以及四个输入复制复现对照。

1. Decoder 会响应患者视觉输入。另一患者 local 替换使 VA 14/18、SA+CA 13/18 份报告改变；整图替换分别为 16/18、17/18。但文字变化不是正确事实绑定的证明。
2. 抹掉区域差异、同槽 role 差异或颠倒顺序，对整篇 CE 的平均影响小于患者替换。完整报告 CE 对临床词的影响有稀释：整图替换后临床关键词 CE 点估计变差，但小样本区间仍跨零。
3. 四个 role 在进入 Decoder 前已经近乎重复：最终 FP32 同槽 role 离散度 VA 为 4.07e-6，SA+CA 为 1.65e-6。它与之前“不同区域 slot 的差异改善”是两个不同轴。
4. query 的共享 slot 分支平均范数 VA/SA+CA 为 78.78/120.71，而 role embedding 约 0.995、UMLS 概念项约 0.434/0.438；不同角色经过 Reader 后进一步趋同。分支量级失衡是可直接检验的候选机制，尚未通过干预重训练验证因果修复效果。

## 下一轮实验建议

- **尺度对照：** slot/role/concept 分支分别归一化并设可追踪的 scale，其他结构不变；同时观察角色 attention、最终 role 输出和属性任务。只降低余弦不算成功。
- **内容监督：** 最终 local bundle 恢复本区域教师特征；这约束区域内容，但不能单独防止四角色重复。
- **角色监督：** 在可靠标签上验证位置、信号、形态、发现各自的任务与干预效应。不能将报告级标签无差别贴给所有空间 slot。
- **LLM 读出对照：** 固定同一视觉 checkpoint，比较原冻结 Llama 与仅训练 LLM LoRA；再做相同额外训练预算的“继续训练 projector/视觉 LoRA”与“再加 LLM LoRA”对照。
- **配对验证：** 临床不同而序列相同的患者图像，正确编码应更支持正确事实；报告临床等价时允许多正例。以患者独立的验证生成与属性准确性评估，不能用 test 选方案。

当前 `joint_lora` 仅指视觉 LoRA。真正加入 LLM LoRA 时需扩展 decoder 构建、optimizer/scheduler、阶段管理、checkpoint 保存与恢复、评测加载；不能只修改 frozen 开关。可先试 q/k/v/o、r=8/16、alpha=2r、LLM LR=1e-5/2e-5，均属于待验证候选参数。

文献依据及各自限制详见完整报告：BLIP-2 的表征学习阶段消融，LLaVA 与 M3D-LaMed 的 LLM 适配，InstructBLIP 的任务条件读取，Flamingo 的 Decoder 内视觉 cross-attention，RGRG 的区域到句子接口，以及 VCD/PAI/MMVP 的视觉干预诊断。
