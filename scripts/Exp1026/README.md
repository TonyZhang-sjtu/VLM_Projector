# Exp1026：统一 Baseline A 训练流程

本目录最初于 2026-10-27 随 `sync-20261027-exp1026` 发布；此后 Concat 入口改为 128/256 双卡实验，并新增 A4、ADirect 及其 LocalRead / Detail / 全局部实验。所有入口使用当前共享训练 pipeline，参数在各脚本顶部可见。Baseline A 复用 `scripts/ft_ablation/Exp1020/run_bindformer_va_fixed_region_mean_constlr.sh`；其余入口复用 `scripts/ft_ablation/_constant_lr.bash`。

## 默认实验范围

| 脚本 | 方法 | 默认倍率 | 默认物理 GPU |
|---|---|---|---:|
| `run_adaptive_merge3d.sh` | AdaptiveMerge | 4 / 8 / 16 / 32 | 3 |
| `run_attention_pool.sh` | AttentionPool | 4 / 8 / 16 / 32 | 3 |
| `run_avgpool3d_r128_r256.sh` | AvgPool | 128 / 256 | 0 |
| `run_baseline_a_r128_r256.sh` | BindFormer Baseline A | 128 / 256 | 0 |
| `run_baseline_a4_r128_r256.sh` | A4 四子区均值 | 128 / 256 | 3 |
| `run_baseline_adirect.sh` | ADirect 原始子区均值直通 | 32 / 64（当前顶部数组为 64 / 32，sweep 排序执行） | 0 |
| `run_baseline_adirect_localread.sh` | ADirect_LocalRead：一次局部 attention 残差 | 128 / 256 | 3 |
| `run_baseline_adirect_detail.sh` | ADirect_Detail：代表性原始 patch 替换 Global 输出 | 128 / 256 | 3 |
| `run_baseline_adirect_alllocal.sh` | ADirect：`global_fraction=0`，全局部均值 | 128 / 256 | 3 |
| `run_concat3d_r128_r256.sh` | Concat | 128 / 256 | 0,1 |
| `run_convpool3d.sh` | ConvPool | 4 / 8 / 16 / 32 | 3 |
| `run_local_packer3d_r128_r256.sh` | LocalPacker | 128 / 256 | 2 |
| `run_mixer3d.sh` | Mixer | 4 / 8 / 16 / 32 | 3 |
| `run_mlp2x.sh` | MLP | 1 | 3 |
| `run_perceiver_flamingo.sh` | Perceiver-Flamingo | 4 / 8 / 16 / 32 | 3 |
| `run_perceiver_nv_r128_r256.sh` | Perceiver-NV | 128 / 256 | 3 |
| `run_token_select3d.sh` | TokenSelect | 8 / 16 | 3 |

共17个单方法入口、43个默认实验；`run01.sh` 另将三个 ADirect 变体的 128×/256× 实验按顺序串联，不是新增模型方法。各入口内部依次执行所选倍率，每个倍率从预训练模型和全新 projector/LoRA/optimizer 初始化，训练结束后评测该次最低 validation CE 的 `best.pt`，再进行下一倍率。不要给所有脚本统一附加 `--ratios 128 256`；MLP只支持1×。

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

本目录单方法入口都通过共享helper加载 `PROJECTOR_LOCAL_CONFIG`；也可显式传 `--config "$PROJECTOR_LOCAL_CONFIG"`。`run01.sh` 仅接受 `--dry-run`，通过环境变量向子入口传递本机配置。注意旧 `scripts/baselines/run_*.sh` 的路径加载行为不同，当前实验请调用本目录入口。

```bash
# 仅预检，不加载模型，不创建正式run。
CUDA_VISIBLE_DEVICES=0 bash scripts/Exp1026/run_adaptive_merge3d.sh \
  --config "$PROJECTOR_LOCAL_CONFIG" --dry-run

# 正式启动示例：仅在用户决定开始训练、确认本机GPU空闲后执行。
CUDA_VISIBLE_DEVICES=0 bash scripts/Exp1026/run_adaptive_merge3d.sh \
  --config "$PROJECTOR_LOCAL_CONFIG" \
  --run-tag "adaptive_merge_exp1026_$(date -u +%Y%m%dT%H%M%S%NZ)"
```

`CUDA_VISIBLE_DEVICES` 覆盖脚本中的物理卡默认值，模型内仍用逻辑 `cuda:0`。`PROJECTOR_OUTPUT_ROOT` 可映射到本机新输出目录。单方法入口可用CLI `--run-tag` 指定新名称；`run01.sh` 使用 `PROJECTOR_RUN_TAG` 设置批次前缀。不要依赖 `PROJECTOR_RUN_TAG` 对所有单方法脚本都生效。TokenSelect保留当前固定默认名称 `token_select_r8_r16`，重复运行应显式换新tag，不能覆盖已有结果。

Concat 128/256 需要两张 24 GiB 卡：脚本中的 `CUDA_VISIBLE_DEVICES=0,1` 指定物理卡，Encoder 和冻结的 Llama 位于逻辑 `cuda:0`，完整宽度的 Concat3D 投影器及 AdamW 状态位于逻辑 `cuda:1`。两卡之间的特征和梯度经 CPU 中转，以避开本机不可靠的 GPU 间直接拷贝。该入口默认关闭只读的逐参数优化器诊断，以给 256× 投影器留出显存；训练损失、优化器和学习率配方不变。例如换用物理卡 2、3：

```bash
CUDA_VISIBLE_DEVICES=2,3 bash scripts/Exp1026/run_concat3d_r128_r256.sh --dry-run
CUDA_VISIBLE_DEVICES=2,3 bash scripts/Exp1026/run_concat3d_r128_r256.sh
```

这不是 DDP；不要用 `torchrun` 启动。128× 训练、完整 test 评测完成后，脚本才会启动 256×。可用 `--ratios 256` 单独运行 256×，或用 `--run-tag` 指定本次实验名。训练和评测都会自动使用双卡放置。

数据、模型、知识库和临床评测overlay不在Git里。缺少时按[同步流程](../../Pipeline_0929/GITHUB_UPLOAD_AND_SERVER_SYNC.md)补齐相同版本；不要为跑通而更换输入、减小测试范围或跳过指标。预算可解析并不意味着所有倍率的完整宽度模型都适合本机显存；已有Concat高倍率资源约束见[128×／256×实现记录](../../Pipeline_0929/PROJECTOR_R128_R256_IMPLEMENTATION.md)。

## ADirect：只替换 A4 的局部处理路径

2026-11-01 新增 `bindformer_adirect`，配置为 `configs/projector/bindformer_adirect.yaml`。

```bash
# 真实数据/header预检；不加载模型或启动训练。
CUDA_VISIBLE_DEVICES=3 bash scripts/Exp1026/run_baseline_adirect.sh --ratios 128 256 --dry-run

# 先128×训练及完整测试，再256×；可用 --ratios 128 或 --ratios 256 单独运行。
CUDA_VISIBLE_DEVICES=3 bash scripts/Exp1026/run_baseline_adirect.sh --ratios 128 256
```

ADirect 与 A4 使用相同的空间父区/四子区划分、输出顺序和 Global 分支：
128× 为 17 Global + 52 Local = 69 token；256× 为 7 + 28 = 35 token。
局部计算改为 `shared_head(final_norm(mean(raw_encoder_features_in_child)))`，
绕过局部 InputAdapter（包括额外位置/序列加法）、V/out 投影和局部 FFN；Global 仍使用原 InputAdapter。
最终 LN/head 仍由两路共享。局部没有新增位置编码；固定空间顺序及 encoder 本身的空间信息仍在。
要求 `input_dim == inner_dim`（当前为768），不通过隐式线性层偷偷改变这条直接通路。

训练数据、报告CE、视觉LoRA、冻结LLM、优化器、早停及完整test评分均与A/A4一致。
默认从预训练资产和全新projector/视觉LoRA开始，不加载A/A4实验权重；同seed不表示不同结构的每个权重初值相同。
同样早停规则不保证实际步数一致；机制实验如需等步数，应对双方同时覆盖早停及max_steps。
这检验局部处理路径的整体必要性，不能把性能变化单独归因于某一个LN、投影层或位置加法。

实际宽度的projector参数：128×为27,661,056，256×为27,653,376；相比A4各减少5,905,152参数。
预检与CPU完整宽度BF16反传记录见
`experiments/isolated_20261101_adirect_preflight/verification.json`。
后续纯结构方案与本轮边界见
`experiments/architecture_review_20261101/report.html`。

## ADirect 的三个结构实验

三组默认倍率均为 128 / 256，复用上述 Exp1026 report-generation pipeline。
不改变报告 CE、输入、视觉 LoRA、冻结 LLM、训练/生成/评分参数，不加载已有 A/A4/ADirect 的实验权重。
使用相同训练上限和早停规则；实际停止步数可能不同。

| 实验 | 注册名称 | 128× 输出 | 256× 输出 | 实际宽度参数量 |
|---|---|---|---|---:|
| ADirect_LocalRead | `bindformer_adirect_localread` | 17 Global + 52 Local | 7 Global + 28 Local | 30,023,425 / 30,015,745 |
| ADirect_Detail | `bindformer_adirect_detail` | 17 Detail + 52 Local | 7 Detail + 28 Local | 19,932,672 / 19,932,672 |
| ADirect 全局部 | `bindformer_adirect`，`global_fraction=0` | 69 Local | 35 Local | 19,932,672 / 19,932,672 |

LocalRead 保留 ADirect 的 Global 分支及原始四子区划分。
每个 child 由原始 patch 均值产生一个 query，K/V 来自同一个 child 的原始 encoder 特征；
使用 768 维、8 heads、两个 LN、无 bias 的 Q/K/V/O 和 padding mask，输出为
`mu + alpha * attention_correction`。`alpha` 为跨 child 共享的可学习标量，初始 0.1；
不增加局部 FFN、局部位置/序列编码或辅助损失。新增 2,362,369 参数。
实现与此前 [overview PDF](../../Pipeline_0929/figures/bindformer_adirect_localread_overview/bindformer_adirect_localread_overview_projector.pdf) 的默认规格一致。

Detail 用 TokenSelect 已有的确定性 diversity 算法选择 17/7 个原始 patch：
归一化 detached 特征上做 greedy max-min，首点为 index 0，最后按原网格索引排序。
选择决策不求导，但选中原始特征仍回传梯度。
输出顺序为 `[Detail; Local]`，Local 的 52/28 个均值及最终共享 LN/head 不变；
没有 Global query、Global attention、InputAdapter 或新增局部位置/序列嵌入。
`detail_fraction=0.25` 沿用原 Global 预算的整数分配规则，实际为 17/69 或 7/35。

全局部组不注册新 projector，入口只覆盖 `projector.params.global_fraction=0.0`。
零比例现在采用按物理最长轴递归二分的 M 个非空、不重叠空间块，再取原始特征均值；
不会沿用“四子区整包”规则后留下 1/3 个 Global token。
默认 0.25 比例的 ADirect 分区、预算及权重键保持不变。
全局部的空间划分会随 52→69 或 28→35 个局部 token 改变，因此它检验的是预算分配及随之变化的覆盖粒度。
保存配置中使用不同的 allocation version，防止把两种几何静默混用。

在仓库根目录运行（GPU 编号可替换）：

```bash
# 任一入口加 --dry-run 可先完成真实 manifest/header 预检。
CUDA_VISIBLE_DEVICES=3 bash scripts/Exp1026/run_baseline_adirect_localread.sh
CUDA_VISIBLE_DEVICES=3 bash scripts/Exp1026/run_baseline_adirect_detail.sh
CUDA_VISIBLE_DEVICES=3 bash scripts/Exp1026/run_baseline_adirect_alllocal.sh
```

每个入口依次完成 128× 训练、best.pt 全量 test / 全部指标，再启动 256×。
可加 `--ratios 256` 只跑一个倍率，或 `--run-tag NAME` 指定新输出名。
全局部默认 run tag 含 `alllocal`，避免与标准 ADirect 混淆。
仍支持 `PROJECTOR_PYTHON`、`PROJECTOR_LOCAL_CONFIG` 和 `PROJECTOR_OUTPUT_ROOT`。
如果要比较其他 Global/Local 比例，可在全局部入口追加
`--set projector.params.global_fraction=0.125`；命令行覆盖优先，保存配置记录最终值。

真实数据计划、逐字段配方核对和 CPU 完整宽度 BF16 反传记录位于
`experiments/isolated_20261101_adirect_variants_preflight/`。
这些检查不启动正式训练，也不构成 GPU 显存或训练效果验证。

## 同步验收

从仓库根目录运行。下面的循环**全部带 `--dry-run`**，不启动正式训练：

```bash
export SYNC_CHECK_DIR="$PWD/.tmp/exp1026_sync_$(date -u +%Y%m%dT%H%M%S%NZ)"
mkdir -p "$SYNC_CHECK_DIR"
for launcher in scripts/Exp1026/*.sh; do
  name="$(basename "$launcher" .sh)"
  # run01 是批量调度器，不接受下面的单方法参数，单独检查。
  [[ "$name" == run01 ]] && continue
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
# 如果需要验证批量调度器，可单独运行；它会再次预检六个子计划。
CUDA_VISIBLE_DEVICES=0 bash scripts/Exp1026/run01.sh --dry-run \
  > "$SYNC_CHECK_DIR/run01.log" 2> "$SYNC_CHECK_DIR/run01.stderr"
```

`Pipeline_0929/SYNC_EXP1026_20261027.json` 是 2026-10-27 的历史快照，包含旧 Concat 入口及旧倍率，不能用于比较现在的全部计划。上面的命令只做当前配置预检；需要新的同步验收时应生成并审查新参考文件。

历史同步验证记录见 `.tmp/github_sync_exp1026_20261027/`；该记录不覆盖新的 Concat 双卡实现。
