# Exp1026：统一 Baseline A 训练流程

本目录最初于 2026-10-27 随 `sync-20261027-exp1026` 发布；此后 Concat 入口改为 128/256 双卡实验。所有入口使用当前共享训练 pipeline，参数在各脚本顶部可见。Baseline A 复用 `scripts/ft_ablation/Exp1020/run_bindformer_va_fixed_region_mean_constlr.sh`；其他11个入口复用 `scripts/ft_ablation/_constant_lr.bash`。

## 默认实验范围

| 脚本 | 方法 | 默认倍率 | 默认物理 GPU |
|---|---|---|---:|
| `run_adaptive_merge3d.sh` | AdaptiveMerge | 4 / 8 / 16 / 32 | 3 |
| `run_attention_pool.sh` | AttentionPool | 4 / 8 / 16 / 32 | 3 |
| `run_avgpool3d_r128_r256.sh` | AvgPool | 128 / 256 | 0 |
| `run_baseline_a_r128_r256.sh` | BindFormer Baseline A | 128 / 256 | 0 |
| `run_concat3d_r128_r256.sh` | Concat | 128 / 256 | 0,1 |
| `run_convpool3d.sh` | ConvPool | 4 / 8 / 16 / 32 | 3 |
| `run_local_packer3d_r128_r256.sh` | LocalPacker | 128 / 256 | 2 |
| `run_mixer3d.sh` | Mixer | 4 / 8 / 16 / 32 | 3 |
| `run_mlp2x.sh` | MLP | 1 | 3 |
| `run_perceiver_flamingo.sh` | Perceiver-Flamingo | 4 / 8 / 16 / 32 | 3 |
| `run_perceiver_nv_r128_r256.sh` | Perceiver-NV | 128 / 256 | 3 |
| `run_token_select3d.sh` | TokenSelect | 8 / 16 | 3 |

共12个入口、33个默认实验。各入口内部依次执行所选倍率，每个倍率从预训练模型和全新 projector/LoRA/optimizer 初始化，训练结束后评测该次最低 validation CE 的 `best.pt`，再进行下一倍率。不要给所有脚本统一附加 `--ratios 128 256`；MLP只支持1×。

## 统一训练配方

- seed=42，deterministic=true，`joint_lora`，bf16；Concat 128/256 使用单进程双卡模型并行，其余入口为单GPU。
- projector / encoder LoRA峰值LR均为2e-4；最多20 epochs；默认数据下123 updates/epoch、最多2460 updates。
- effective batch/microbatch=16/1；encoder LoRA rank/alpha/dropout=16/32/0.05。
- projector warmup/hold/joint warmup=77/0/77，joint warmup策略为 `hold_peak`。
- `cosine_min_lr_ratio=1.0`：预热后恒定LR。不是Exp1024的0.3下限方案。
- AdamW、weight_decay=0.01、grad_clip=1；早停看完整validation answer CE，patience=3、min_delta=0.005。
- train/val/test=1960/268/560；训练后执行完整test和全部7项指标。

CLI的 `--set`、`--ratios` 可以覆盖脚本参数；这是显式的新实验设置，不应在同步验收中偷偷改变配方。`--resume-run` 会沿用旧的保存计划，不能用它把旧1e-5实验切换到当前配方。本次同步不迁移或复用旧checkpoint。

## 本机路径与启动

在每台服务器激活各自已核验的训练环境：

```bash
conda activate projector
export PROJECTOR_PYTHON="$(command -v python)"
export PROJECTOR_LOCAL_CONFIG="$PWD/configs/local_paths.yaml"
export PROJECTOR_OUTPUT_ROOT="$PWD/runs/Exp1026"
export PYTHONNOUSERSITE=1
```

必须显式设置 `PROJECTOR_PYTHON`，因为入口保留的默认解释器是源服务器的 `/mnt/sda/mingjie/conda_envs/projector/bin/python`。各机 `configs/local_paths.yaml` 保持Git忽略，保存自己的data、manifest、encoder、decoder和cache路径；不能从源机照抄绝对路径。可用其他外置overlay路径替换上述配置位置。

本目录12个入口都通过共享helper加载 `PROJECTOR_LOCAL_CONFIG`；也可显式传 `--config "$PROJECTOR_LOCAL_CONFIG"`。注意旧 `scripts/baselines/run_*.sh` 的路径加载行为不同，当前实验请调用本目录入口。

```bash
# 仅预检，不加载模型，不创建正式run。
CUDA_VISIBLE_DEVICES=0 bash scripts/Exp1026/run_adaptive_merge3d.sh \
  --config "$PROJECTOR_LOCAL_CONFIG" --dry-run

# 正式启动示例：仅在用户决定开始训练、确认本机GPU空闲后执行。
CUDA_VISIBLE_DEVICES=0 bash scripts/Exp1026/run_adaptive_merge3d.sh \
  --config "$PROJECTOR_LOCAL_CONFIG" \
  --run-tag "adaptive_merge_exp1026_$(date -u +%Y%m%dT%H%M%S%NZ)"
```

`CUDA_VISIBLE_DEVICES` 覆盖脚本中的物理卡默认值，模型内仍用逻辑 `cuda:0`。`PROJECTOR_OUTPUT_ROOT` 可映射到本机新输出目录。所有入口均可用CLI `--run-tag` 指定新名称；不要依赖 `PROJECTOR_RUN_TAG` 对所有脚本都生效。TokenSelect保留当前固定默认名称 `token_select_r8_r16`，重复运行应显式换新tag，不能覆盖已有结果。

Concat 128/256 需要两张 24 GiB 卡：脚本中的 `CUDA_VISIBLE_DEVICES=0,1` 指定物理卡，Encoder 和冻结的 Llama 位于逻辑 `cuda:0`，完整宽度的 Concat3D 投影器及 AdamW 状态位于逻辑 `cuda:1`。两卡之间的特征和梯度经 CPU 中转，以避开本机不可靠的 GPU 间直接拷贝。该入口默认关闭只读的逐参数优化器诊断，以给 256× 投影器留出显存；训练损失、优化器和学习率配方不变。例如换用物理卡 2、3：

```bash
CUDA_VISIBLE_DEVICES=2,3 bash scripts/Exp1026/run_concat3d_r128_r256.sh --dry-run
CUDA_VISIBLE_DEVICES=2,3 bash scripts/Exp1026/run_concat3d_r128_r256.sh
```

这不是 DDP；不要用 `torchrun` 启动。128× 训练、完整 test 评测完成后，脚本才会启动 256×。可用 `--ratios 256` 单独运行 256×，或用 `--run-tag` 指定本次实验名。训练和评测都会自动使用双卡放置。

数据、模型、知识库和临床评测overlay不在Git里。缺少时按[同步流程](../../Pipeline_0929/GITHUB_UPLOAD_AND_SERVER_SYNC.md)补齐相同版本；不要为跑通而更换输入、减小测试范围或跳过指标。预算可解析并不意味着所有倍率的完整宽度模型都适合本机显存；已有Concat高倍率资源约束见[128×／256×实现记录](../../Pipeline_0929/PROJECTOR_R128_R256_IMPLEMENTATION.md)。

## 同步验收

从仓库根目录运行。下面的循环**全部带 `--dry-run`**，不启动正式训练：

```bash
export SYNC_CHECK_DIR="$PWD/.tmp/exp1026_sync_$(date -u +%Y%m%dT%H%M%S%NZ)"
mkdir -p "$SYNC_CHECK_DIR"
for launcher in scripts/Exp1026/*.sh; do
  name="$(basename "$launcher" .sh)"
  if [[ "$name" == run_concat3d_r128_r256 ]]; then
    visible=0,1
  else
    visible=0
  fi
  CUDA_VISIBLE_DEVICES="$visible" bash "$launcher" --dry-run \
    --config "$PROJECTOR_LOCAL_CONFIG" \
    --run-tag "sync_${name}" --output-root "$SYNC_CHECK_DIR/planned_runs" \
    > "$SYNC_CHECK_DIR/${name}.json" 2> "$SYNC_CHECK_DIR/${name}.stderr" || exit
done
```

`Pipeline_0929/SYNC_EXP1026_20261027.json` 是 2026-10-27 的历史快照，包含旧 Concat 入口及旧倍率，不能用于比较现在的 12 份计划。上面的命令只做当前配置预检；需要新的同步验收时应生成并审查新参考文件。

历史同步验证记录见 `.tmp/github_sync_exp1026_20261027/`；该记录不覆盖新的 Concat 双卡实现。
