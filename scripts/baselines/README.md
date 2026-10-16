# BraTS baseline 一键训练与评测

BindFormer 入口 `run_bindformer3d.sh`（默认 8/16/32×，GPU 3）和 `run_bindformer3d_2.sh`（默认 4×，GPU 1）也使用此处相同的 `joint_lora` 训练、完整 BraTS test 评测和 `all` 指标协议。其概念记忆和 BindFormer 参数由专用 sweep 配置；训练与评价调度和普通 baseline 对齐。

BindFormer 的几何、包内 SA、局部读取全局和特征重建均已实现，但默认关闭；四角色 query 数默认均为 1。参数、独立消融命令和教师缓存导出方法见[设计文档第 26 节](../../Pipeline_0929/BINDFORMER_DESIGN_AND_IMPLEMENTATION_PLAN.md#26-可选结构与特征重建实现2026-10-16)，测试范围和待决策项见[扩展验收记录](../../Pipeline_0929/BINDFORMER_EXTENSIONS_ACCEPTANCE.md)。`L_text` 暂不实现。

GREEN 的二次格式修复仍失败时，最终 `metric.json` 的 `overall.green` 按其余
`status=ok` 样本取均值，失败样本不计零分、不进入分母。`partial_metrics.green`
记录 valid/failed/expected、coverage 和失败状态；评测及 sweep 仍标为 `partial`。
只有完全没有有效 GREEN 分数时才省略该均值并放入 `skipped_metrics`。原始
`scoring/clinical_metrics.json` 保留 `valid_only_mean`，其全量 `mean` 和全量置信区间
在覆盖不完整时仍为空。不同方法比较时同时报告覆盖率；需要配对比较时使用共同有效样本。

所有入口显式加载 `configs/training/joint_lora.yaml`、确定性设置、原网格预处理和对应方法配置。
每个压缩率独立从预训练 NeuroVFM / Llama 初始化，projector 从头训练，LLM 始终冻结。
每一档执行 **训练 → 评测该档验证 CE 最低的 best.pt → 下一档**；不会沿用上一档权重。
压缩方法通常默认 **4 → 8 → 16 → 32**；当前 TokenSelect 文件保留用户选择的 **8 → 16**。
MLP 不压缩，只执行 **1× 一次**。各 Bash 顶部的配置区是该文件的实际默认设置。

## 入口

| Bash 文件 | 注册方法 | 默认压缩档位 |
| --- | --- | --- |
| `run_mlp2x.sh` | `mlp2x_gelu` | 1（不压缩） |
| `run_avgpool3d.sh` | `avgpool3d` | 4、8、16、32 |
| `run_concat3d.sh` | `concat3d` | 4、8、16、32 |
| `run_convpool3d.sh` | `convpool3d` | 4、8、16、32 |
| `run_mixer3d.sh` | `mixer3d` | 4、8、16、32 |
| `run_local_packer3d.sh` | `local_packer3d` | 4、8、16、32 |
| `run_attention_pool.sh` | `attention_pool` | 4、8、16、32 |
| `run_perceiver_nv.sh` | `perceiver_nv` | 4、8、16、32 |
| `run_perceiver_flamingo.sh` | `perceiver_flamingo` | 4、8、16、32 |
| `run_adaptive_merge3d.sh` | `adaptive_merge3d` | 4、8、16、32 |
| `run_token_select3d.sh` | `token_select3d` | 8、16（当前用户设置） |
| `run_bindformer3d.sh` | `bindformer3d` | 8、16、32（默认物理 GPU 3） |
| `run_bindformer3d_2.sh` | `bindformer3d` | 4（默认物理 GPU 1） |

## 在文件中指定压缩率

11 个入口都提供同样的配置区，压缩方法例如：

```bash
projector_ratios=(8 16)       # 或 (8)、(16)、(4 8 16 32)
projector_run_tag=""          # 空串自动生成时间戳；固定名称需要每次更换
```

`run_mlp2x.sh` 必须保持 `projector_ratios=(1)`。`run_token_select3d.sh` 当前保留
`projector_ratios=(8 16)` 和 `projector_run_tag="token_select_r8_r16"`；重跑时更换名称，
或设为空串。命令行位于文件默认参数之后，可以覆盖它们，例如：

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/baselines/run_token_select3d.sh \
  --ratios 16 --run-tag token_select_r16_new
```

上例只运行 16×。每档仍独立执行训练、best checkpoint 评测，再进入下一档。
GPU 外部选择优先于文件中的默认值；模型内部继续使用逻辑 `cuda:0`。
新服务器先 `conda activate projector` 并 `export PROJECTOR_PYTHON="$(command -v python)"`，
完整迁移步骤见 [环境说明](../../docs/ENVIRONMENT_AND_MIGRATION.md)。

公共调度代码：[baseline_sweep.py](../../src/cli/baseline_sweep.py)。11 个脚本共享相同实现，避免复制训练与评测逻辑后产生差异。
脚本可从任意目录以绝对路径调用；默认 Python 为 `/mnt/sda/mingjie/conda_envs/projector/bin/python`，可通过 `PROJECTOR_PYTHON` 覆盖。

## 使用

```bash
cd /mnt/sda/jyzhang/VLM_Projector

# 只预检真实 train/val 清单、MRI headers、四档预算及训练日程。
# 不加载模型、不使用 GPU、不创建实验目录。
bash scripts/baselines/run_avgpool3d.sh --dry-run

# 外部选择 GPU 3；默认训练/生成/评分均使用映射后的逻辑设备 cuda:0。
CUDA_VISIBLE_DEVICES=3 bash scripts/baselines/run_avgpool3d.sh --run-tag avgpool_joint_v1

# 首轮只跑 8×、32×。
CUDA_VISIBLE_DEVICES=3 bash scripts/baselines/run_perceiver_nv.sh --run-tag perceiver_nv_pilot \
  --ratios 8 32

# MLP 只运行一次不压缩参考。
CUDA_VISIBLE_DEVICES=3 bash scripts/baselines/run_mlp2x.sh --run-tag mlp_joint_v1

# 覆盖训练参数；rank 和 alpha 独立设置。
CUDA_VISIBLE_DEVICES=3 bash scripts/baselines/run_concat3d.sh --run-tag concat_custom \
  --set training.max_epochs=10 \
  --set training.projector_warmup_steps=77 \
  --set training.joint_warmup_steps=77 \
  --set training.lora.rank=16 --set training.lora.alpha=32

# 可选：固定 30% test 子集，使用配置中的 seed，所有档位保持相同抽样。
CUDA_VISIBLE_DEVICES=3 bash scripts/baselines/run_local_packer3d.sh --run-tag local_pilot --ratios 8 32 \
  --eval-sample-fraction 0.3 --metrics bleu4 rougeL bertscore answer_ce
```

省略 `--run-tag` 时自动使用方法名与 UTC 时间戳。已有 run-tag 一律拒绝覆盖。
`--ratios` 可以选择四档的子集；程序按升序执行。MLP 仅接受省略此参数或 `--ratios 1`。
这些入口为单 GPU 顺序运行脚本；DDP 继续使用 `scripts/launch_ddp.sh`。
单卡默认设备在 `configs/base.yaml` 中统一为 `cuda:0`。11 个脚本继承外部的 `CUDA_VISIBLE_DEVICES`，
不会在内部设置或覆盖该变量。只暴露 GPU 3 时，程序内部编号为 `cuda:0`，省略 `--device` 即可；
也可以先执行 `export CUDA_VISIBLE_DEVICES=3`，再运行任一脚本。
若暴露多张卡，例如 `CUDA_VISIBLE_DEVICES=2,3`，单卡入口仍默认只用列表中的第一张；
`--device cuda:1` 才会使用列表中的第二张，不会自动启动 DDP。
`--scoring-device` 默认沿用训练设备，显式指定时也使用可见设备的逻辑编号。
未设置 `CUDA_VISIBLE_DEVICES` 时，默认使用 CUDA 枚举的第一张卡。
历史 run 配置属于已保存的显式设置；恢复旧配置时用 `--set training.device=cuda:0` 覆盖其中的设备值。

## 默认配置核对

- BraTS-only，当前训练/验证样本数为 1960 / 268，有效 batch size 为 16。
- 最多 10 epochs，每 epoch 123 次 optimizer updates，总计划 1230 次。
- 双 warm-up 默认都按总步数的 6.25% 计算，当前为 **77 + 77**；联合 cosine **1076** 步。
- 两组峰值 LR 都是 `1e-5`；视觉 LoRA `rank=16, alpha=32, dropout=0.05`。
- 连续切换阶段，不重建 projector/Adam；早停在两段 warm-up 后判断，`min_delta=0.005, patience=3`。
- `best.pt` 始终根据最低验证 CE 保存，不受早停的 min_delta 门槛约束。
- 确定性预处理缓存开启，最终视觉特征缓存关闭；native_grid + minimal_pad，无背景过滤。
- `perceiver_nv` / `perceiver_flamingo` 的方法 YAML 启用 `projector.params.checkpoint_blocks=true`。
  真实 4× smoke 发现关闭该选项时 24 GiB GPU 在反向传播 OOM；通过重算中间激活降低显存，
  保留 6 层、宽度、参数量、确定性注意力和完整输入/输出 token 预算。此选项不等于保存训练 checkpoint，
  也不修改 LoRA 或学习率。大显存设备可显式传 `--set projector.params.checkpoint_blocks=false`。
- 77/77/1076 是当前数据规模与 epoch/batch 设置推导的最大日程；覆盖参数会重新计算，早停可能提前结束。
- `configs/base.yaml` 仍保留旧独立 stage A 的基础值；这里的 11 个入口显式加载 joint_lora preset，不能用裸 `python -m src.cli.train` 代替。

运行前统一预检所有档位，复用一次真实 train/val header 的几何检查；每档训练器仍会再次验证实际配置与输入几何。

| 目标 | 网格方法 M | Query / Merge / Selection M | Concat M | Concat 分块 |
| --- | ---: | ---: | ---: | --- |
| 4× | 2197 | 2194 | 2250 | 4×1×1 |
| 8× | 1100 | 1097 | 1125 | 8×1×1 |
| 16× | 550 | 549 | 525 | 6×1×3 |
| 32× | 275 | 275 | 280 | 6×2×3 |

## 评测与输出

默认评测 **完整 BraTS test 集**，指标为 `all`：BLEU-4、ROUGE-L、GREEN、RadGraph、RaTE、BERTScore、answer CE。
生成与离线评分按既有独立进程顺序执行；评分设备可用 `--scoring-device cuda:N` 或 `cpu` 单独指定。
用 `--eval-split val` 改为验证集；`--metrics` 选择指标，`--repeats` 修改 bootstrap 次数。
全体方法保持同一评测范围；调参使用验证集，不根据逐档 test 分数继续选择超参数。

```text
runs/baselines/<run-tag>/
  sweep_plan.json                   # 实际配置、预算、日程、命令
  sweep_status.json                 # 当前阶段、已完成档位、失败原因
  r4/                              # MLP 为 r1
    training_config.yaml           # 独立冻结的启动配置
    train_console.log
    evaluation_console.log
    joint_lora/
      resolved_config.yaml
      schedule.json
      checkpoints/best.pt
      checkpoints/last.pt
    evaluation/best_test/
      metric.json
      expected_ids.json
      generation/predictions.jsonl
  r8/ ...
  r16/ ...
  r32/ ...
```

输出根目录可以用 `--output-root /path/to/baselines` 修改。
任一训练或评测失败立即停止，不继续下一档，也不会把失败目录当作完成。
本脚本不自动重试或自动恢复旧 sweep；原 Trainer 的同 run 断点恢复仍可用：

```bash
# 以下以 avgpool_joint_v1 的 r8 训练中断为例，使用该档保存的完整配置。
PROJECTOR_PYTHON=/mnt/sda/mingjie/conda_envs/projector/bin/python
CUDA_VISIBLE_DEVICES=3 PYTHONNOUSERSITE=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  "$PROJECTOR_PYTHON" -m src.cli.train \
  --config runs/baselines/avgpool_joint_v1/r8/joint_lora/resolved_config.yaml \
  --set training.device=cuda:0 \
  --set training.resume=runs/baselines/avgpool_joint_v1/r8/joint_lora/checkpoints/last.pt

# 随后评测恢复完成的 best.pt；已有失败评测目录时另取 output-dir。
CUDA_VISIBLE_DEVICES=3 bash scripts/evaluate_checkpoint.sh \
  --checkpoint runs/baselines/avgpool_joint_v1/r8/joint_lora/checkpoints/best.pt \
  --output-dir runs/baselines/avgpool_joint_v1/r8/evaluation/best_test_retry \
  --device cuda:0 --metrics all

# 剩余档位以新 tag 启动，避免重复训练已完成的档位。
CUDA_VISIBLE_DEVICES=3 bash scripts/baselines/run_avgpool3d.sh --run-tag avgpool_joint_v1_remaining --ratios 16 32
```

## 工程验收

本次新增 launcher 的验证覆盖：11 个入口、41 套实际数据/几何 dry-run、默认日程、覆盖参数、
训练→评测顺序、子进程失败即停、已有目录保护，以及 joint_lora / 压缩预算 / checkpoint 评测相关回归。
记录位于 [baseline_launchers_20261012T161710Z](../../runs/archive/engineering/baseline_launchers_20261012T161710Z/)。
顺序执行单元测试使用轻量替身子进程；用户追加要求后另通过实际 Bash 入口运行真实 NeuroVFM + Llama smoke：
每档 6 个 optimizer updates（2/2/2）、6 条训练样本、1 条验证样本，评测 1 条 val 样本并生成最多 2 个 token。
保留完整 8775-token 输入、默认 projector 宽度/深度和 BF16 模型；结果以验收目录中的最新 `verification.json` 为准。
没有启动正式 10-epoch 训练，短 smoke 不能证明收敛或报告生成质量。
此前真实模型的联合 LoRA 证据保留在 [原验收记录](../../runs/archive/engineering/joint_lora_20261012T151636Z/verification.json)。

本轮最终结果：**193 项测试通过；41/41 组真实短训练通过**（10 个压缩方法 × 4 档 + MLP 一次，共 246 次 optimizer updates）。
逐档核对实际 N/M、有限损失/梯度、两组参数实际变化、Adam 计数连续、三阶段 LR、checkpoint 重载及生成 token 元数据。
生成检查须单独报告：39 组在 2-token smoke 下完成评测；Flamingo 8× 前两 token 为空白，保持原 checkpoint/样本、
改为正式默认 192-token 上限后非空/评测流程通过，但实际仍为重复标点，不能称有效报告；Concat 32× 在 192-token 上限下仍重复特殊 token、解码为空，
因此这一项**生成验收未通过**。原 Bash sweep 的失败状态保留，未放宽空生成检查、未伪造成功状态。
这说明 6 次更新足以验证训练工程链路，不能保证未收敛 checkpoint 的生成有效性；不能声称 41 组端到端评测全部通过。
明细见 [real_smoke_verification.json](../../runs/archive/engineering/baseline_launchers_20261012T161710Z/real_smoke_verification.json)。

2026-10-12 PDT 复核：Concat 32× 的失败生成仍可计算 teacher-forced CE（13.8264），与解码为空并不矛盾。详见 [路线文档第 16 节](../../Pipeline_0929/PROJECTOR_ROUTES_AND_BASELINES.md#16-smoke-问题复核单卡预算与类别覆盖2026-10-12-pdt)，包含 OOM 修复、7 类代表建议及待执行的生成退化诊断方案。
