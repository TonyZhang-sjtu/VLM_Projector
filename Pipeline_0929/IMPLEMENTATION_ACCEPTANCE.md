# native-grid 框架实现与验收记录

工程验收数据日期：2026-10-06；共享预处理缓存的正式路径与清理说明于 2026-10-12 更新。环境：`conda activate projector`；所有真实 GPU 验收使用 GPU3。源码直接放在 `src/`。本文件的训练/评价状态是 2026-10-06 时的记录，不代表后续实验状态。

## 1. 实现范围

| 步骤 | 落地内容 | 验收证据（项目根目录相对路径） |
|---|---|---|
| 1 | 严格 YAML 合并、可重复 CLI 覆盖、editable 安装、resolved config | `configs/`、`src/config.py`、`archive/non_pipeline/environment_checks/pip_check_final.txt` |
| 2 | coref_v2 固定 manifest、序列映射、完整体积 Dataset、公开患者隔离 | `runs/archive/engineering/implementation/data/` |
| 3 | native/2mm/4mm 单参数切换、几何追踪、patch/cache、独立 mask QA | `runs/archive/engineering/implementation/native_grid_audit/`、`spacing_comparison/` |
| 4 | 可微 NeuroVFM、rank-8 LoRA/Full、Projector registry、独立体积编码 | `tests/test_vision.py`、`runs/archive/training_and_model_diagnostics/native_smoke_*/` |
| 5 | 本地冻结 Llama 8B、图像 token 替换、答案/EOT CE、独立生成 | `tests/test_decoder.py`、`runs/archive/engineering/implementation/native_generation_final.json` |
| 6 | A→两种 B、全局答案 token 归一化、checkpoint/resume、固定验证生成、DDP 接口 | `tests/test_training.py`、`test_distributed_loss.py`、`runs/archive/training_and_model_diagnostics/native_deterministic_a/` |
| 7 | greedy 生成、覆盖、BLEU/ROUGE 分层、患者 bootstrap/配对比较、视觉对照 | `runs/archive/training_and_model_diagnostics/native_evaluate_cli/`、`runs/archive/training_and_model_diagnostics/native_learning_controls/`、`runs/archive/engineering/implementation/paired_control_cli.json` |
| 8 | 真实梯度/更新/冻结权重/重载/显存、短学习诊断、确定性检查 | 下文；最终状态见 `runs/archive/engineering/implementation/status.json` |

代码运行说明与配置清单见 [README](../README.md)。环境已补齐 sacrebleu、rouge-score、NLTK，保留现有 torch/CUDA/transformers/PEFT 版本；`pip check` 通过。

## 2. 预处理选择及保留范围

- 默认 `preprocessing.spacing_policy=native_grid`、`boundary_policy=minimal_pad`；不重采样、不裁剪、不筛选背景 token。
- 通过 image/cohort/preprocessing hash 复用的共享 patch cache 保存在 `assets/preprocessing/<preprocessing_version>/<key前两位>/<key>.pt`；默认启用，可在训练、评测和预处理 CLI 中用 `--no-use-cache`（别名 `--no-cache`）关闭，也可用 `--use-cache` 显式启用。缓存开关和存储路径不改变 preprocessing version。活动 artifact 的配置、清单和匹配方法见 [`assets/preprocessing/README.md`](../assets/preprocessing/README.md) 与 [`preprocessing_artifact.json`](../runs/full_native_20261006/preprocessing/preprocessing_artifact.json)。
- 单参数改为 `2mm` 或 `4mm` 时，层向分别 2/4 mm，平面内 1 mm，边界仍为最小 padding。更一般的间距由 `resample` + spacing 数值控制。
- train/val 的 2,476 个序列 header 均已核验，按同网格同 mask 去重到 805 次检查，39,903 个逐标签六邻域连通分量全部保留，标签体素损失 0。
- `Singapore_63_FLAIR` 的 mask/image 方向 header 略异；映射到影像网格后标签逐体素相同。该例没有修改原数据，并保留异常元数据。
- 五例代表性比较中，4 mm 丢失 31/109（28.44%），2 mm 丢失 7/109（6.42%），native 丢失 0；这是抽样几何比例，不能外推全体，也不能等同于漏诊率。

## 3. 真实模型与显存

Llama 未量化、BF16 冻结；MLP/可训练视觉参数 FP32；完整体积；microbatch 1；保留全部答案 token；开启 Llama gradient checkpointing。

| 快速模式 | 视觉训练参数 | 最大样例视觉 token | 峰值分配 GiB | 峰值预留 GiB |
|---|---:|---:|---:|---:|
| A | 0 | 8,775 | 20.51 | 21.59 |
| B-LoRA | 442,368 | 8,775 | 20.97 | 22.08 |
| B-Full | 85,803,234 | 8,775 | 21.63 | 22.76 |
| B-Full，累积 16 | 85,803,234 | 每体积 8,775 | 22.03 | 22.97 |

各分支确认梯度存在且有限、优化器实际更新、冻结权重逐位不变；训练后 checkpoint 重载的 eval loss 完全一致。B-LoRA 和 B-Full 从同一个 A checkpoint 初始化。累积 16 是重复完整长体积的显存压力测试，不能充当 16 位独立患者的训练结果。物理 GPU 为 RTX A5500 24 GiB，实际可分配容量低于标称值；Full 分支余量较小。

## 4. 生成问题及处理

本地 8B 的 SDPA 在混合长度、左 padding 的批量生成中出现 NaN；单体积输入正常。修复后使用 `generation.batch_mode=independent`，列表接口逐样本执行。非有限 logits 直接抛错，不再让 argmax 掩盖 NaN。

真实 A、LoRA、Full checkpoint 的单条调用/列表调用 token 完全一致。原始 A smoke 的失败日志保留；后续成功复验独立记录。几步训练的 checkpoint 出现空输出和重复内容，不能视为报告任务收敛。

## 5. 小样本学习及视觉对照

固定 8 个 ADC train 样本、2 个独立 val 样本，每次有效 batch 8、16 次更新：

| 配置 | 第一次更新前 train CE | 第二次更新前 train CE | 最后一次更新前 train CE | 最后更新后 val CE |
|---|---:|---:|---:|---:|
| 原定 1e-4，短 run 实际 warmup=0 | 2.445 | 24.265 | 8.426 | 8.480 |
| 用户批准诊断：1e-5、4 步 warmup | 2.445 | 2.361 | 1.829 | 2.157 |

![学习诊断](../runs/archive/engineering/implementation/learning_curves.png)

低学习率诊断避免了剧烈上冲，确认损失可下降；没有把 16 步下降称为充分过拟合。正式默认学习率仍保留原配置，诊断设置位于独立 resolved config。

在固定两例验证样本上运行正确图像、同序列错配、零投影视觉和完全纯文本四路生成，均无空输出、均在 192-token 预算内结束；四路生成不同。逐例答案 CE 为：

| 验证样本 | 正常图像 | 同序列错配 | 零视觉 |
|---|---:|---:|---:|
| strokecase0192 ADC | 2.4467 | 2.4957 | 2.9661 |
| strokecase0161 ADC | 1.8516 | 1.8526 | 2.3714 |

这说明输入视觉内容影响当前模型，但报告内容质量尚不足，且两例不能支持统计或临床结论。完整预测、分层指标、配对 CI 和未填写的人工审阅模板保存在 `runs/archive/training_and_model_diagnostics/native_learning_controls/`。

## 6. 续训复现性

快速模式第 8 步恢复到第 16 步：恢复后的第一步前向 CE 完全一致，scheduler/RNG 一致；最终参数最大绝对差约 2.5e-6，CE 最大绝对差 0.00345。独立算子测试确认 SDPA 的 FlashAttention 反向默认非确定性。

新增 `training.deterministic=true`（或 `configs/training/deterministic.yaml`）显式启用 PyTorch 确定性算法、cuBLAS workspace 配置，以及 NeuroVFM FlashAttention 的确定性反向。第三方源码、模型原始权重均未改动。不得在原 run 续训时切换该设置；切换需建立独立实验。

阶段 A 真实 8B 连续 4 步与从第 2 步恢复的对照：projector、encoder、optimizer、scheduler、训练状态、所有保存 RNG、后续 CE/梯度范数均逐位一致。该运行也实际执行了固定验证子集生成。证据：`runs/archive/engineering/implementation/deterministic_resume_comparison.json`。

LoRA/Full 确定性、资源复验结果分别见 `runs/archive/training_and_model_diagnostics/native_deterministic_lora/summary.json` 与 `runs/archive/training_and_model_diagnostics/native_deterministic_full_expandable/summary.json`；完整 8B 多 GPU 训练和跨硬件/库版本逐位复现未在本轮验收范围内。

当前确定性实测：LoRA 和 Full 的短体积重复反向梯度均逐位相同；8,775-token 长体积单步更新均成功。Full 在默认分配器下的第 2 次累积反向 OOM：已分配 21.83 GiB、预留未使用 1.27 GiB，再申请 248 MiB 失败。用户批准后用 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` 重测；最长 9,077-token train 样本的16次累积、梯度裁剪、optimizer.step、冻结权重不变全部通过。峰值分配 22.08 GiB、预留 22.39 GiB、窗口耗时 124.39 秒。记录位于 `runs/archive/training_and_model_diagnostics/native_deterministic_full_expandable/summary.json`。仅为显存分配器设置环境变量，不需改环境或模型。

快速 Full 累积基线在另一条 8,775视觉token 样本上为峰值分配22.03 GiB、预留22.97 GiB、耗时109.33秒；确定性测试使用 106 答案token 的最长训练样本，因此耗时只作近似参照。两种模式在 GPU3 均完成目标窗口。确定性模式对复现更合适，当前建议正式 A/LoRA/Full 共用确定性计算策略，B-Full 运行命令单独设置 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`，环境变量应在 Python/CUDA 启动前设置。

补充全量多模态长度审核：train/val 共 2,476 条，最长输入为 `BraTS-GLI-00477-001-t1n`（train），图像 8,775、答案 106、完整输入 9,077 token；可扩展分配器复验已使用该样本。记录在 `runs/archive/engineering/implementation/multimodal_length_audit.json`。
