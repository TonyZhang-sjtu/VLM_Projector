# MLP `joint_lora` 学习率调度对照

## 比较内容

以已有的 MLP 不压缩 `joint_lora` 运行作为原始对照。两组新实验均保持同一训练/验证划分、模型、随机种子、有效 batch size 16、10 epoch、总计 1,230 次 optimizer update，以及 projector/视觉 LoRA 峰值 LR `1e-5`。每组从相同的预训练资产独立启动，不能通过恢复旧运行的 checkpoint 来切换调度器。

| 组别 | `training.cosine_min_lr_ratio` | `training.joint_warmup_projector_mode` | 第 78 步 projector LR | 第 1,230 步两组 LR |
| --- | ---: | --- | ---: | ---: |
| 原始对照 | `0.0` | `restart` | `0` | `0` |
| 实验 1：尾段下限 | `0.1` | `restart` | `0` | `1e-6` |
| 实验 2：下限 + projector 连续 | `0.1` | `hold_peak` | `1e-5` | `1e-6` |

实现位置：`src/training/schedule.py` 的 `JointSchedule.scale()` 定义倍率，`src/training/optim.py` 用 PyTorch `LambdaLR` 分别应用到 projector 与 encoder 参数组。LR 下限只作用于联合训练的 cosine 段；视觉 LoRA 在 projector-only 段仍被冻结，联合 warm-up 起点仍从 0 升温。`hold_peak` 令 projector 在第二段 warm-up 保持峰值，不改变 LoRA 的 warm-up。

## 在各服务器启动

在仓库根目录、激活各自的训练环境后，先设本机的 Python 路径和空闲 GPU。下面两条命令分别启动一个完整训练及其 best-checkpoint 评测；运行目录名称必须唯一。当前 MLP 启动脚本自动指定不压缩的 ratio 1。

```bash
export PROJECTOR_PYTHON="$(command -v python)"
# 请先在当前 shell 设置 CUDA_VISIBLE_DEVICES 为一张空闲物理 GPU。
: "${CUDA_VISIBLE_DEVICES:?请先指定空闲物理 GPU 编号}"

bash scripts/baselines/run_mlp2x.sh \
  --run-tag mlp_lr_floor10_v1 \
  --eval-split val \
  --metrics answer_ce \
  --set training.cosine_min_lr_ratio=0.1

bash scripts/baselines/run_mlp2x.sh \
  --run-tag mlp_lr_floor10_hold_peak_v1 \
  --eval-split val \
  --metrics answer_ce \
  --set training.cosine_min_lr_ratio=0.1 \
  --set training.joint_warmup_projector_mode=hold_peak
```

如果两个服务器的资产位置不同，可各自创建 **不提交到 Git** 的 `configs/local_paths.yaml`，只写本机路径。例如：

```yaml
data:
  root: /path/to/data
  annotation_root: /path/to/RadGenome-Brain
  manifest_dir: /path/to/training_manifests/single_sequence_findings_coref_v2
encoder:
  path: /path/to/neurovfm-encoder
decoder:
  path: /path/to/llama3.1-8b-instruct
preprocessing:
  cache_dir: /path/to/preprocessing-cache
```

然后在每条启动命令中加入 `--config configs/local_paths.yaml`。配置合并顺序是共享 baseline 预设、本机 YAML、命令行 `--set`；训练目录由 sweep 入口设置，可用 `--output-root` 指定本机位置。`project_root` 默认自动指向本机 checkout，资产若在本机仓库的 `assets/` 下，无需写路径覆盖。`configs/local*.yaml` 已被 `.gitignore` 排除。

## 结果判读与跨服务器同步

先比较同一验证集每个 epoch 的 CE、最低验证 CE 所在 epoch 和训练/验证 CE 差距。实验 1 与原始对照比较下限的作用；实验 2 与实验 1 比较第二段 projector 日程的作用。实验 2 还增加了该阶段 projector 的累计学习率，因此其改善不能仅归因于第 77→78 步不再跳变。测试集只用于最终选定方案后的报告，不用其分数挑调度参数。

GitHub 同步共享代码和实验说明；`assets/`、`runs/`、模型权重、数据、各服务器本地 YAML 都由各服务器管理。另一服务器先保留自己的路径改动，再拉取共享代码；不要直接用这台服务器保存的绝对路径运行配置覆盖另一台服务器的配置。AvgPool3D 8× 的两组对应命令见 [SERVER2_AVGPOOL_LR_AB.md](SERVER2_AVGPOOL_LR_AB.md)。
