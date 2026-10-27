# Exp1026：统一 Baseline A 训练流程

本目录于 2026-10-27 随 `sync-20261027-exp1026` 完整发布。所有入口使用当前共享训练 pipeline，参数在各脚本顶部可见。Baseline A 复用 `scripts/ft_ablation/Exp1020/run_bindformer_va_fixed_region_mean_constlr.sh`；其他11个入口复用 `scripts/ft_ablation/_constant_lr.bash`。不需要复制其他服务器的脚本或运行配置。

## 默认实验范围

| 脚本 | 方法 | 默认倍率 | 默认物理 GPU |
|---|---|---|---:|
| `run_adaptive_merge3d.sh` | AdaptiveMerge | 4 / 8 / 16 / 32 | 3 |
| `run_attention_pool.sh` | AttentionPool | 4 / 8 / 16 / 32 | 3 |
| `run_avgpool3d_r128_r256.sh` | AvgPool | 128 / 256 | 1 |
| `run_baseline_a_r128_r256.sh` | BindFormer Baseline A | 128 / 256 | 0 |
| `run_concat3d.sh` | Concat | 4 / 8 / 16 / 32 | 3 |
| `run_convpool3d.sh` | ConvPool | 4 / 8 / 16 / 32 | 3 |
| `run_local_packer3d_r128_r256.sh` | LocalPacker | 128 / 256 | 2 |
| `run_mixer3d.sh` | Mixer | 4 / 8 / 16 / 32 | 3 |
| `run_mlp2x.sh` | MLP | 1 | 3 |
| `run_perceiver_flamingo.sh` | Perceiver-Flamingo | 4 / 8 / 16 / 32 | 3 |
| `run_perceiver_nv_r128_r256.sh` | Perceiver-NV | 128 / 256 | 3 |
| `run_token_select3d.sh` | TokenSelect | 8 / 16 | 3 |

共12个入口、35个默认实验。各入口内部依次执行所选倍率，每个倍率从预训练模型和全新 projector/LoRA/optimizer 初始化，训练结束后评测该次最低 validation CE 的 `best.pt`，再进行下一倍率。不要给所有脚本统一附加 `--ratios 128 256`；MLP只支持1×。

## 统一训练配方

- seed=42，deterministic=true，单GPU `joint_lora`，bf16。
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

数据、模型、知识库和临床评测overlay不在Git里。缺少时按[同步流程](../../Pipeline_0929/GITHUB_UPLOAD_AND_SERVER_SYNC.md)补齐相同版本；不要为跑通而更换输入、减小测试范围或跳过指标。预算可解析并不意味着所有倍率的完整宽度模型都适合本机显存；已有Concat高倍率资源约束见[128×／256×实现记录](../../Pipeline_0929/PROJECTOR_R128_R256_IMPLEMENTATION.md)。

## 同步验收

从仓库根目录运行。下面的循环**全部带 `--dry-run`**，不启动正式训练：

```bash
export SYNC_CHECK_DIR="$PWD/.tmp/exp1026_sync_$(date -u +%Y%m%dT%H%M%S%NZ)"
mkdir -p "$SYNC_CHECK_DIR"
for launcher in scripts/Exp1026/*.sh; do
  name="$(basename "$launcher" .sh)"
  CUDA_VISIBLE_DEVICES=0 bash "$launcher" --dry-run \
    --config "$PROJECTOR_LOCAL_CONFIG" \
    --run-tag "sync_${name}" --output-root "$SYNC_CHECK_DIR/planned_runs" \
    > "$SYNC_CHECK_DIR/${name}.json" 2> "$SYNC_CHECK_DIR/${name}.stderr" || exit
done
"$PROJECTOR_PYTHON" scripts/check_exp1026_sync.py \
  --plans "$SYNC_CHECK_DIR" --output "$SYNC_CHECK_DIR/comparison.json" \
  --reference Pipeline_0929/SYNC_EXP1026_20261027.json
```

比较工具要求12份计划完整，逐个检查脚本内容SHA256及规范化后的整份执行计划摘要：仅允许解释器位置和明列本机路径变化，保留完整训练配置、倍率、预算、调度、数据身份/manifest摘要、训练与评测命令、checkpoint文件名及评测范围。它不替代实际Git源码比对、环境核对或资产内容校验。若报告不一致，检查输出的JSON与本机原始计划，不要改参考文件来让检查通过。

源机本次验证：12个shell语法检查；24项默认GPU/外部GPU/CLI参数覆盖检查；源目录和不带源机local overlay的隔离目录各执行12份真实数据计划，35个实验完全一致；相关CPU回归61 passed。未执行正式训练或GPU模型验收。源机日志在 `.tmp/github_sync_exp1026_20261027/`，不进Git。
