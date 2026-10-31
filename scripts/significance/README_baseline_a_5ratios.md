# Baseline A：五个压缩率的一键 p-value 分析

入口与训练、普通评测入口完全分开，不修改原始结果或 Excel。

## 一键执行

在文件传输结束后执行（可以从任意工作目录启动）：

```bash
bash /mnt/sda/jyzhang/VLM_Projector/scripts/significance/run_baseline_a_5ratios.sh
```

默认流程：核验全部 35 个实验 → 按来源环境配置补齐逐样本 CE → 计算统计检验 → 导出 Excel/CSV/JSON。
CE 只做 teacher forcing，不训练、不重新生成报告、不重跑临床评分。
已完成的 CE 文件经身份和数值检查后复用，重复执行可以继续补齐。

CE 子进程根据原 `evaluation_config.yaml` 的项目根目录及 `pipeline.json` 的 Python 路径，
从 `ce_runtime_profiles.json` 精确选择来源配置；不根据文件夹名称猜测。
配置文件记录来源证据及其 SHA256、依赖版本，并支持以 checkpoint SHA256 为键的覆盖项。
在导入 Torch/初始化 CUDA 前应用对应环境；目前两边已核实的启动脚本默认值均为：
`CUBLAS_WORKSPACE_CONFIG=:4096:8`、`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`、
`OMP_NUM_THREADS=4`、`MKL_NUM_THREADS=4`、`OPENBLAS_NUM_THREADS=1`。
CPU 统计仍使用单线程。CE 的 `preflight.json`/`summary.json` 记录来源、证据、环境及依赖。
server_b 导出的空闲会话默认 16 线程不代表原实验线程数；历史进程环境未保存，
因此来源配置明确标记为启动脚本默认值，不能宣称恢复了未记录的历史覆盖项。

一键 shell 默认 `--ce-policy recompute`：固定权重重新测量 CE，p-value 使用本次的
`sum(loss_sum)/sum(answer_tokens)`。`ce_replay_audit` 表及比较明细同时保存历史 CE、
重算 CE、差值；`historical_comparison.json` 还保存各疾病分组的差值。
汇报 CE 的 p-value 时，应配套使用重算 CE 分数，不能声称它逐位复现历史 Excel。
不修改历史 Excel/metric 文件。跨 GPU 的小偏差不被伪装为历史匹配；
权重、数据/模型 contract、样本 ID/reference、总 token 和分组 token 数仍严格核验。

若需要历史数值复现，可显式执行 `--ce-policy strict`；该模式保留原 `1e-5` 容差，
并额外检查各疾病分组。Python CLI 默认仍为 strict；shell 显式选择 recompute。
单个 CE preflight/推理失败会写入 `ce_progress.json`，继续其他已核验实验，
最后导出可用的检验结果，失败项保持空白；不会因一个 CE 失败丢掉其余进度。

原双 GPU Concat3D 权重补跑 CE 时，将内存中的 `training.projector_device` 设为 `null`，
表示与主模型放在同一张已分配的 GPU；不加载训练梯度或优化器状态。
映射记录在 `preflight.json`/`summary.json` 的 `device_remappings`，原始配置及权重不修改。
dry-run 同样检查映射后的配置和影像几何预算，避免配置冲突直到推理阶段才暴露。

只检查数据是否传全、是否匹配，不进行模型推理或统计检验：

```bash
bash /mnt/sda/jyzhang/VLM_Projector/scripts/significance/run_baseline_a_5ratios.sh --audit-only
```

暂时只做 CPU 分析，不补跑 CE：

```bash
bash /mnt/sda/jyzhang/VLM_Projector/scripts/significance/run_baseline_a_5ratios.sh --skip-ce
```

`--skip-ce` 下已经存在且通过核验的逐样本 CE 仍可使用；缺失的 CE 留空。
不把汇总 CE 当作逐样本观测，不伪造 330 项全部完成。

## 固定比较范围

- Excel：`results/Experiments for VLM Projector.xlsx` 的 `Main` 表。
- 压缩率：16、32、64、128、256。
- Ours：严格选择 `Ours Vanilla+fixed region mean 1021`，允许表内 `Vanlla` 拼写及 `(Baseline A)` 后缀。
- 对照：AvgPool3D、Concat3D、LocalPacker3D、Perceiver_NV、Adaptive_Merge3D、Token_Select3D。
- 不包括 MLP、其他 BindFormer 变体以及 E/R/role-value ablation。
- 指标为原表 11 列，共 5 × 6 × 11 = 330 项检验。

输出同时保留 `p_raw`、`p_holm_66`（每个压缩率内 66 项）及 `p_holm_330`
（五个压缩率整体 330 项，作为主校正口径）。每个 `rXX/significance.xlsx` 内的
`p_Holm` 对应该压缩率的 66 项。缺 CE 时维持原定 family 大小，标记保守的未完成校正，
补齐后重新计算。

沿用 10,000 次以患者为单位的配对置换、双侧检验和 10,000 次配对 bootstrap。
不同序列按患者一起交换/重采样；每对方法、每个指标取共同有效样本，不插补。
BLEU 保留 corpus 聚合；CE 使用 `sum(loss_sum)/sum(answer_tokens)`。
95% CI 是未经多重比较校正的边际区间。结论针对选定权重与测试患者，不代替多 seed 实验。

## 防止匹配错误

初始的 35 个目录及 checkpoint/config/metric SHA256 已记录于
`scripts/significance/baseline_a_bindings.json`。它们对应当前 Excel 的具体行和全部 11 个分数。
运行时检查实际文件内容，而不是只相信目录名或汇总数字：

1. 检查 Excel 区块、方法行、列顺序及全部 11 个数值。Baseline A 还检查 fixed-region mean、
   关闭 SA/CA 与相关辅助任务的配置。
2. 对绑定目录检查三个固定哈希，核对权重内部配置、validation best 的选择记录。
3. 核对预测与临床评分的来源哈希、样本 ID、reference、患者身份、scorer 版本，
   从原始记录重新聚合分数并对照未舍入的历史结果。
4. 核对每对方法使用相同测试样本、训练/验证选择及评测规则。
5. 全部核验通过后才允许执行。缺文件、传输不完整、配置不符、存在歧义时返回非零退出码，
   原因写入 `preflight.json` 和 `run_audit.json`，不回退到旧实验。

两台服务器分别使用 RTX A5500 和 RTX A5000，Python 安装位置不同，因此原始 scorer
指纹不同。兼容性检查先验证每个原始指纹与其完整 provenance 一致，再只剔除
`environment.executable`、`device_name` 和 CUDA 编号等运行位置字段进行比较。
模型文件、代码、Python/依赖版本、评分设置、seed、TF32、设备后端仍必须完全一致。
原指纹和运行环境保存在 `run_audit.json`，不修改原始分数，也不宣称跨显卡逐位相同。

输出目录中的 `run_bindings.json` 保存已通过审核的绑定，之后不会自动改选。
`mapping_candidates.json` 保留匹配候选供检查。若显式使用新的绑定文件而其中有未绑定项，
脚本停止并要求补充明确绑定；候选发现不会自动替代绑定，不按最新/最高分挑选。
Excel 目标行变化会拒绝复用旧绑定，需要重新核对绑定并使用新输出目录。

输入目录默认同时查找：

```text
/mnt/sda/jyzhang/VLM_Projector/runs/baselines/
/mnt/sda/jyzhang/VLM_Projector/runs/imported/server_b/
```

跨服务器文件保留 `<原始实验名>/rXX/joint_lora/` 和 `evaluation/best_test/` 的完整结构。
CE 重放只对原项目根目录下的数据、模型与缓存路径做本地映射，映射记录写入 CE 的
`preflight.json`；原始 resolved config 不修改。迁移后仍核对训练时的 manifest 内容哈希、
测试 reference，以及加载时完整的模型/数据/预处理 provenance。

## 输出与资源

默认输出目录：

```text
/mnt/sda/jyzhang/VLM_Projector/results/significance_baseline_a_5ratios/
  significance.xlsx          # 汇总 330 项，包含三种 p 值及运行身份
  comparisons.csv/json
  source_workbook.xlsx       # 本次实际读取的 Excel 快照
  study_snapshot.json
  run_bindings.json
  run_audit.json
  mapping_candidates.json
  preflight.json
  status.json                # 当前运行/停止/完成状态
  ce_progress.json           # 完成、失败及待处理的 CE 模型
  r16/                      # r32、r64、r128、r256 同样结构
    run_manifest.json
    significance.xlsx
    comparisons.csv/json
    ce/<method>/
      preflight.json
      preflight.log
      console.log
      samples.jsonl         # loss_sum、answer_tokens 及样本身份
      summary.json
```

补跑 CE 时顺序使用一张 GPU；选择条件仅为空闲显存至少 20,000 MiB，
不限制已有计算进程、已用显存或 GPU 利用率。每 30 秒检查一次，最长等待 24 小时。
不停止其他任务。这是本地协作锁，不是调度系统预留；其他任务仍可能随后占用显卡。
`--gpu 2` 可以仅等待物理编号 2；`--max-wait-hours 0` 表示没有符合条件的卡就退出。
通过 Ctrl-C 可中断，再次执行同一命令会核验并复用完整 CE 输出。

更多参数：`--help`。`SIGNIFICANCE_PYTHON` 默认 `python`，用于 CPU 统计；
`PROJECTOR_PYTHON` 默认 `/mnt/sda/mingjie/conda_envs/projector/bin/python`，用于 CE。
CPU 环境需要 NumPy、PyYAML、Torch、SacreBLEU、rouge-score、openpyxl。
文件发现优先使用 `rg`；环境中没有该命令时自动使用 Python 遍历，无需额外安装。
入口默认使用当前激活环境的 `python`；若换用或新建环境，Excel 附加依赖可这样安装：

```bash
python -m pip install -r /mnt/sda/jyzhang/VLM_Projector/scripts/significance/requirements.txt
```

若设置了 `SIGNIFICANCE_PYTHON`，请使用该解释器执行上述 `-m pip install`。
不要将 `--audit-only` 成功理解为已经验证过 GPU 模型前向。
