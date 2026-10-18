# FT_BASELINE_1017：AvgPool3D 8× 微调流程消融

更新日期：2026-10-17。适用方法：AvgPool3D，压缩率 8×。基于此前的 c26bde010e12988f8c29997e57044ed936cc49c4 增加本轮功能。

2026-10-18 增补：**新一轮 N1–N6 的配置与跨服务器部署见第 12 节**。第 1–11 节保留 E1–E6 发布时的实验定义与历史验证记录。

**交付状态：本轮实现和发布前验证已完成，用户已授权上传。六组正式配置 dry-run、六组真实模型短步训练、checkpoint 核对及 2 条 test 样本的全部指标评测均通过，详见第 11 节。尚未启动正式 10/20 epoch 实验。** 文中正式预算和阶段边界不代表已经跑完正式实验；smoke 数值不作为实验结果。发布 commit 以源服务器回复和 Git 历史为准，可用 `git log -1 --format=%H -- report/FT_BASELINE_1017.md` 查询本文档最近提交。

## 1. 已确认的实验定义

- 每张 GPU 独立运行一个实验，不采用六卡 DDP；共六个入口 e1.sh–e6.sh。
- 10/20 epoch 指**整个流程的总预算上限**，包含 projector warm-up、恒定训练、LoRA warm-up、joint cosine。
- Projector warm-up 和 LoRA warm-up 都明确设为 **77 个 optimizer updates**，不随总 epoch 数翻倍。
- 可选的 projector 恒定阶段为 **500 steps**，LR 恒定为 projector 峰值；视觉编码器及其 LoRA 冻结。
- LoRA warm-up 期间，projector 继续以峰值训练，只有 LoRA LR 从 0 升温，即 `hold_peak`。
- 两组峰值 LR 在本轮各实验内保持相同，cosine 下限比例均为 **0.2**。
- 每个 Shell 自动执行：**训练成功 → 由完整 validation CE 选择的 best.pt → 完整 test split 的全部指标**。
- 六组均从同一组预训练视觉/语言资产和相同 seed 新启动，不从 B0 或其他实验 checkpoint 初始化。

## 2. 配置与实验目的

B0 是已有历史对照，不在本轮增加第七个入口。历史运行名为 `avgpool_r8_lr_floor20_hold_peak_2x_lr_v1`。它的结果、权重保留在原服务器；Git 同步不包含这些文件。

下表以本项目现有 BraTS 清单 train=1,960、val=268、test=560，有效 batch=16 为准。每个 epoch 有 123 次 optimizer updates；最后一个窗口为 8 个样本。

| 实验/入口 | 默认物理 GPU | Projector / LoRA 峰值 LR | Projector 恒定 steps | 总 epoch 上限 | 总 steps 上限 | Joint cosine steps | 目的 |
| --- | ---: | --- | ---: | ---: | ---: | ---: | --- |
| B0，历史对照 | — | 2e-5 / 2e-5 | 0 | 10 | 1,230 | 1,076 | 当前参考配置 |
| [E1 / e1.sh](../scripts/ft_ablation/e1.sh) | 0 | 2e-5 / 2e-5 | 0 | 20 | 2,460 | 2,306 | 与 B0 比较，延长总训练预算 |
| [E2 / e2.sh](../scripts/ft_ablation/e2.sh) | 1 | 2e-5 / 2e-5 | 500 | 10 | 1,230 | 576 | 与 B0 比较，在固定总预算内增加 projector 独立训练 |
| [E3 / e3.sh](../scripts/ft_ablation/e3.sh) | 2 | 2e-5 / 2e-5 | 500 | 20 | 2,460 | 1,806 | 与 E1/E2 比较，研究恒定阶段与总预算的交互 |
| [E4 / e4.sh](../scripts/ft_ablation/e4.sh) | 3 | 3e-5 / 3e-5 | 0 | 10 | 1,230 | 1,076 | 与 B0 比较，提高两组 LR |
| [E5 / e5.sh](../scripts/ft_ablation/e5.sh) | 4 | 3e-5 / 3e-5 | 500 | 10 | 1,230 | 576 | 与 E2/E4 比较，研究高 LR 与恒定阶段的交互 |
| [E6 / e6.sh](../scripts/ft_ablation/e6.sh) | 5 | 5e-5 / 5e-5 | 0 | 10 | 1,230 | 1,076 | 继续探索更高 LR |

- B0、E1、E2、E3 构成“恒定阶段 0/500 × 总预算 10/20 epoch”的对照。
- B0、E2、E4、E5 构成“恒定阶段 0/500 × 峰值 LR 2e-5/3e-5”的对照。
- 固定总预算时，新增 500 steps 会替换 500 个 joint cosine updates；这检验的是训练时间分配。
- 不同峰值下，0.2 下限对应的末尾 LR 分别为 4e-6、6e-6、1e-5。因此 LR 实验放大的是整条曲线。

## 3. 阶段与步数定义

配置字段：

```yaml
training:
  projector_warmup_steps: 77
  projector_hold_steps: 500
  joint_warmup_steps: 77
  joint_warmup_projector_mode: hold_peak
  cosine_min_lr_ratio: 0.2
  max_epochs: 10
```

`projector_hold_steps` 是本次新增字段；默认值为 0。没有设置该字段的旧配置仍使用原来的三阶段 LR 公式。

总预算由 `max_epochs × updates_per_epoch` 决定；如另设 `max_steps`，取两者中更小的值。默认六组的 `max_steps=null`。

```text
joint cosine steps =
total steps - projector_warmup_steps - projector_hold_steps - joint_warmup_steps
```

两个 warm-up 和 cosine 阶段各要求至少 2 steps；恒定阶段允许为 0，0 表示直接跳过。所有 step 均为 optimizer update，不是单个 microbatch。

### 含 500 steps 恒定阶段的边界

| train.jsonl 中的 step，首尾包含 | phase | Projector LR | 视觉编码器/LoRA |
| --- | --- | --- | --- |
| 1–77 | projector_warmup | 0 → 峰值 | 冻结 |
| 78–577 | projector_hold | 始终为峰值 | 冻结 |
| 578–654 | joint_warmup | 始终为峰值 | 仅 LoRA 可训练，LR 从 0 → 峰值 |
| 655–总预算末步 | joint_cosine | 峰值 → 0.2 × 峰值 | LoRA 按对应曲线衰减 |

末尾 LR 描述仅适用于实际跑到总预算末步的情况；提前停止时记录实际结束 LR。

恒定阶段对应约 500/123=4.07 epoch；projector 独立训练合计 577 steps，约 4.69 epoch。无恒定阶段的 E1/E4/E6 在第 78 步开始 LoRA warm-up，第 155 步开始 joint cosine。

### 训练状态与兼容

- 新阶段中视觉编码器保持 eval/no_grad 状态，视觉 LoRA 参数不生成梯度。
- Projector 始终可训练；阶段切换沿用同一个 AdamW，不清空已积累的动量。
- 日志和 checkpoint 记录新 phase；`schedule.json` 和 dry-run 计划新增 `phase_ranges`，列出一开始的第 1 步至各阶段结束步。
- 在完成 77、577、654 steps 时，现有阶段切换机制会保存对应 boundary checkpoint；无恒定阶段时仍为 77、154。
- 旧 checkpoint 缺少 hold 字段时按 0 解析；恢复时检查 LR、hold 长度、总预算等必须匹配。该兼容路径尚未做本轮运行验证。
- 带有 `projector_hold` 的新 checkpoint 应使用更新后的代码读取；两台服务器先同步到同一版代码。

## 4. 其他共同参数

共享配置在 [configs/ft_ablation/avgpool3d_r8.yaml](../configs/ft_ablation/avgpool3d_r8.yaml)。

| 参数 | 值 |
| --- | --- |
| Seed / deterministic | 42 / true |
| Microbatch / effective batch | 1 / 16 |
| 混合精度 / LLM | BF16 / 冻结 |
| 优化器 | AdamW，weight_decay=0.01，betas=(0.9,0.999)，eps=1e-8 |
| 全局梯度裁剪 | 1.0 |
| 视觉 LoRA | **rank=16，alpha=32，dropout=0.05** |
| 输入预处理 | 与原 baseline 一致的 native_grid 预设 |
| 每轮验证 CE | 完整 validation，当前 268 条 |
| 每轮生成监控 | 固定 14 条的 BLEU/ROUGE；不代表完整 test |
| 解码 | greedy，num_beams=1，max_new_tokens=192 |
| Checkpoint 选择 | best validation Answer CE |
| Early stopping | patience=3，min_delta=0.005，沿用原 baseline |

**已确认保留早停：** 六组均使用 `patience=3`、`min_delta=0.005`，10/20 epoch 是上限，允许提前结束。

- 依据每个 epoch 的完整 validation Answer CE；14 条生成监控不参与早停。
- 仅在进入 joint cosine 后的完整 epoch 验证时计数。按每 epoch 123 updates，无恒定阶段的 E1/E4/E6 从 epoch 2 末开始；含 500 steps 恒定阶段的 E2/E3/E5 从 epoch 6 末开始。
- 第一次符合条件的验证建立参考 CE。相对上次有效改善的参考值，CE 累计降低**超过 0.005**才重置计数；连续 3 次未达到该改善则停止。
- `best.pt` 独立记录全部 epoch 验证中最低的 CE；即使改善不足 0.005，也会更新 best checkpoint。
- `early_stopping` 是正常训练完成状态，随后仍自动对 `best.pt` 执行完整 test 评测。

2026-10-17 的针对性 CPU 检查已覆盖：patience 计数、累计 min_delta 重置、best CE 独立更新、阶段计数起点，以及 max_epochs/max_steps/early_stopping 的评测准入条件。检查未加载模型或使用 GPU。

## 5. 文件分工与代码修改

| 文件 | 本次作用 |
| --- | --- |
| `src/training/schedule.py` | 四阶段日程、显式 hold 步数、阶段起止 metadata、旧字段缺省兼容 |
| `configs/base.yaml`、`configs/training/joint_lora.yaml`、`src/config.py` | 新字段默认值与类型校验 |
| `src/models/encoders/neurovfm.py`、`src/training/stages.py` | warm-up/hold 两阶段都冻结视觉参数并检查 trainable 状态 |
| `src/training/checkpoint.py`、`src/training/trainer.py` | 恢复契约检查，规范化旧 schedule metadata |
| `src/cli/baseline_sweep.py` | 可选 `--require-full-test`，检查训练完成、完整 test 范围和全部指标结果 |
| `configs/ft_ablation/avgpool3d_r8.yaml` | 六组共同参数，避免回落到 base.yaml 的 LoRA rank=8 |
| `scripts/ft_ablation/e1.sh`–`e6.sh` | 每组可编辑 GPU、两组 LR、三段前置步数、总 epoch、run tag |
| `scripts/ft_ablation/_common.bash` | 可迁移路径/解释器处理，同一入口完成训练和完整 test |
| `.gitignore` | 忽略旧三份本机脚本与 results 产物，保留共享填写指南 |
| `tests/test_joint_training.py`、`tests/test_baseline_sweep.py`、`tests/test_ft_ablation_launchers.py` | hold 边界、冻结/解冻、Adam 连续性、恢复、完整 test 与六组 Bash 参数回归检查 |

六个入口依赖仓库中的共同配置、`_common.bash` 和 Python 源码。通过 Git 同步整次代码修改；只复制某个 eN.sh 不足以运行。

## 6. 另一台服务器：路径与环境

本机私有信息放在 Git 忽略的 `configs/local_paths.yaml`，例如：

```yaml
data:
  root: /your/server/data
  annotation_root: /your/server/RadGenome-Brain
  manifest_dir: /your/server/RadGenome-Brain/training_manifests/single_sequence_findings_coref_v2
encoder:
  path: /your/server/neurovfm-encoder
decoder:
  path: /your/server/llama3.1-8b-instruct
preprocessing:
  cache_dir: /your/server/preprocessing-cache
```

只填写本机路径，不在该文件中修改实验超参数。入口自动发现这个文件；没有该文件时使用仓库默认资产路径。也可用 `PROJECTOR_LOCAL_CONFIG` 指定另一份本地覆盖文件。

```bash
conda activate projector
export PROJECTOR_PYTHON="$(command -v python)"
# 可选：指定本机运行输出根目录。
# export PROJECTOR_OUTPUT_ROOT=/your/server/runs/baselines
```

评测还需要本机完整的 GREEN、RadGraph、RaTEScore、BERTScore 资产及依赖。按 [环境与迁移说明](../docs/ENVIRONMENT_AND_MIGRATION.md) 准备，Git 不包含这些资产。

脚本从自己的位置定位仓库，不硬编码服务器的 HOME、Conda 或数据位置。

Python 选择顺序为：显式 `PROJECTOR_PYTHON` → 已激活环境的 `$CONDA_PREFIX/bin/python` → `PATH` 中的 `python`。这可处理 tmux 中提示符显示 `(projector)`、但 `PATH` 仍优先解析到 base Python 的情况；显式设置的 `PROJECTOR_PYTHON` 始终优先。

### GPU 选择

每个 eN.sh 顶部都能直接编辑 GPU 编号。例如 e2.sh：

```bash
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
```

把默认值 1 改成你分配的卡号即可；如外部已经设置该环境变量，则使用外部值。显式覆盖示例：

```bash
CUDA_VISIBLE_DEVICES=2 bash scripts/ft_ablation/e2.sh
```

脚本只接受一个物理 GPU 索引，不调用 nvidia-smi 自动查卡。程序内训练、生成和评分使用该卡对应的逻辑 `cuda:0`。请为并行实验手动分配不同的空闲卡。

## 7. 运行方式：同一 Shell 自动训练并完整测试

### e1.sh 的实际调用链

```text
bash scripts/ft_ablation/e1.sh
  └─ 设置 GPU 和 E1 超参数
     └─ source scripts/ft_ablation/_common.bash
        └─ run_ft_ablation "$@"
           └─ python -u -m src.cli.baseline_sweep
              ├─ python -u -m src.cli.train --config .../training_config.yaml
              │  └─ Trainer.train()：训练、每轮 validation、保存 best.pt
              └─ 训练成功完成后：python -u -m src.cli.evaluate_checkpoint
                 ├─ python -m src.cli.evaluate：完整 test 生成、BLEU/ROUGE、Answer CE
                 ├─ python -m src.cli.score_reports：GREEN/RadGraph/RaTEScore/BERTScore
                 └─ 汇总 evaluation/best_test/metric.json 并核对完整性
```

`source` 在当前 Shell 中加载共享函数，随后 `exec` 将 Shell 替换为 Python 调度进程；该进程负责顺序等待训练和评测子进程。E1 不调用 `scripts/baselines/run_avgpool3d.sh`，两者共用 Python 层的 baseline sweep。其余 E2–E6 的调用链相同，仅入口超参数和默认 GPU 不同。

2026-10-17 已通过六个入口及共享脚本的 `bash -n` 语法检查，并用临时命令捕获程序逐个检查六个入口传出的参数：均调用 `src.cli.baseline_sweep`、指定完整 test/all，并正确接受显式 GPU 覆盖。该检查没有启动实际训练或评测，完整模型与评分资产仍需在正式运行环境中验证。

以下先预览计划，不启动训练：

```bash
bash scripts/ft_ablation/e1.sh --dry-run
bash scripts/ft_ablation/e2.sh --dry-run
bash scripts/ft_ablation/e3.sh --dry-run
bash scripts/ft_ablation/e4.sh --dry-run
bash scripts/ft_ablation/e5.sh --dry-run
bash scripts/ft_ablation/e6.sh --dry-run
```

应核对：train=1,960、val=268、evaluation_samples=560、evaluation_scope=full_split、eval_split=test、123 updates/epoch、两组 LR、hold 长度、总预算和 phase_ranges。另一台服务器还要核对患者划分及模型/manifest 内容身份，不能仅比较样本数量。

确认环境、配置和 GPU 分配后，每个 tmux 窗口只需运行对应的一条命令：

```bash
bash scripts/ft_ablation/e1.sh
bash scripts/ft_ablation/e2.sh
bash scripts/ft_ablation/e3.sh
bash scripts/ft_ablation/e4.sh
bash scripts/ft_ablation/e5.sh
bash scripts/ft_ablation/e6.sh
```

以上六行表示六个独立入口；分配到六个窗口运行即可并行。在一个窗口顺序粘贴会顺序运行。

共同入口实际携带：

```text
--projector avgpool3d --ratios 8
--eval-split test --metrics all --eval-sample-fraction 1.0
--require-full-test
```

执行顺序为：

1. 启动该实验的训练子进程。
2. 训练成功退出，且记录了 max_epochs/max_steps/early_stopping 完成状态后，加载该实验的 best.pt。
3. 在完整 test split 上生成预测并计算 Answer CE。
4. 生成进程退出后，启动全部离线指标评分。
5. 核对 split、全量样本数、11 个有限数值指标，保存最终结果和状态。

训练失败时停止，不会进入最终评测。最终 test 和全指标选项在这些专用入口中固定，避免误用 val 或子集。

全部 11 个结果字段：BLEU-4、ROUGE-L F1、GREEN、RadGraph-XL Simple/Partial/Complete F1、RaTEScore、BERTScore Precision/Recall/F1、Answer CE。

GREEN 个别样本评分失败时，保留有效样本均值和成功/总数，状态可为 partial；不会把它称为全部评分成功。某个必需指标缺失或非有限值时，实验状态为失败，保留产物以便排查/续评。

默认输出：

```text
runs/baselines/ft1017_eN_avgpool3d_r8_lr..._p..._ep..._v1/
  sweep_plan.json
  sweep_status.json
  r8/
    training_config.yaml
    train_console.log
    evaluation_console.log
    joint_lora/
      schedule.json
      train.jsonl
      validation.jsonl
      phase_transitions.jsonl
      training_state.json
      checkpoints/best.pt
      checkpoints/last.pt
    evaluation/best_test/
      metric.json
      expected_ids.json
      generation/predictions.jsonl
```

重复新实验使用新 tag，例如：

```bash
FT_RUN_TAG=ft1017_e2_repeat_v2 bash scripts/ft_ablation/e2.sh
```

已有目录不会覆盖。可以编辑 eN.sh 的参数，也可以传入 `--set training.KEY=VALUE` 做明确的新实验，并同步修改 tag/记录配置。总预算包含所有阶段，不另加 500 steps。

### 中断恢复的范围

训练完成、评测中断时，可在同一台服务器用原保存计划继续评测：

```bash
bash scripts/ft_ablation/e2.sh --resume-run runs/baselines/ft1017_e2_avgpool3d_r8_lr2e-5_p500_ep10_v1
```

此方式沿用原计划及 checkpoint；CLI 的新训练超参数不会改写已保存计划。入口会再次检查该计划是否是完整 test/all。底层评测支持已有预测和评分缓存续用。

训练本身中断时，现有 sweep 不会自动恢复训练，需要显式使用原配置和该次 last.pt 的 same-run resume；不得恢复到另一实验目录、变更阶段预算或用其他组的 checkpoint。这里没有新增跨服务器 checkpoint 路径重写功能。

## 8. GitHub 同步范围与发布步骤

用户已于 2026-10-17 确认上传本轮代码与本文档。发布前先完成本机 smoke 和变更检查；另一台服务器同步后，应核对源服务器提供的实际发布 commit。

共享内容：本轮源码、配置、六个入口、共同 Bash、本文档和 .gitignore。

不共享：
- 现有 `scripts/baselines/run_avgpool3d_r8_lr_floor20_restart.sh`。
- 现有 `scripts/baselines/run_avgpool3d_r8_lr_floor20_hold_peak.sh`。
- 现有 `scripts/baselines/run_avgpool3d_r8_lr_floor20_hold_peak_2x_lr.sh`。
- `results/` 下的 Excel、图、审计记录、一次性评测/填表脚本；共享的 `BASELINE_RESULTS_ENTRY_GUIDE.md` 例外。
- `configs/local_paths.yaml`、数据、模型、checkpoint、缓存、runs。

发布时先审阅 git diff，明确列出待提交的文件，不使用 git add -f。完成发布后记录实际 commit SHA。

另一台服务器：

```bash
git status --short
git diff
git fetch origin main
git pull --ff-only origin main
git rev-parse HEAD
```

先安全保留本地代码改动，把路径差异放入本机 local_paths.yaml。快进失败时审阅分叉并保留本地工作后合并，不用 reset --hard、clean 或强制覆盖解决。核对两台 HEAD 相同后，再按第 6、7 节操作。

## 9. 结果记录与后续判断

- 每组记录最终配置、实际完成 step/epoch、停止原因、best validation CE 及对应 step、完整 test 指标与 GREEN 覆盖率。
- Excel 中单独注明 split=test、checkpoint SHA 和 metric.json 来源；训练中的 validation 监控不填入最终 test 结果列。
- 筛选后将两个候选方案迁移到 Perceiver-NV、BindFormer，再决定公共 baseline 训练流程。
- 根据本机过去的 8× 运行，10 epoch 训练约 8 小时，完整 test 约 3 小时；20 epoch 预计训练加 test 约 19–21 小时。六卡并行、另一服务器硬件和更改后的阶段速度尚未实测。

## 10. 给另一台服务器的同步提示词

下面的文字可直接交给另一台服务器上的代码助手；发布 commit 以源服务器最终回复为准。

```text
请在本机 VLM_Projector 仓库同步 FT-1017 更新，并完成运行前检查。

1. 先检查 git status --short --untracked-files=all、git diff 和当前分支。
   安全保留本机代码改动与未跟踪文件；数据、模型、缓存路径集中放在
   Git 忽略的 configs/local_paths.yaml，保留本机的实际路径。
   不要 reset --hard、git clean、强制推送或覆盖本地工作。

2. 执行 git fetch origin main、git pull --ff-only origin main。
   如存在本地提交或冲突，先审阅并保留本地工作后合并。
   核对 git rev-parse HEAD 与源服务器提供的发布 commit；若远端有更新，
   确认该发布 commit 是当前 HEAD 的祖先，并说明额外更新。

3. 阅读 report/FT_BASELINE_1017.md。本次新增：
   - training.projector_hold_steps，默认 0，指定 500 时在 projector
     77-step warm-up 后增加 500 次恒定峰值 LR 更新，视觉编码器/LoRA 冻结。
   - 随后 77-step LoRA warm-up，projector 保持峰值；最后 joint cosine，
     floor=0.2。阶段切换保留 Adam 状态。
   - scripts/ft_ablation/e1.sh 至 e6.sh、共享 _common.bash 和
     configs/ft_ablation/avgpool3d_r8.yaml；六组配置与目的见文档表格。
   - 10/20 epoch 是整个训练的上限，包含所有前置阶段。
   - 保留早停 patience=3、min_delta=0.005；LoRA rank=16、alpha=32。
   - 每个入口自动完成训练，然后用 validation CE 最低的 best.pt
     对完整 test split 计算全部 11 个指标；触发早停也会自动评测。
   - 新增阶段、checkpoint、完整 test 和 Bash 参数的回归测试。

4. 激活本机训练环境并执行：
   export PROJECTOR_PYTHON="$(command -v python)"
   核对本机模型、数据以及 GREEN/RadGraph/RaTEScore/BERTScore 的资产和依赖。
   Git 不传输 local_paths.yaml、模型、数据、checkpoint、runs、Excel、
   其他实验结果，以及旧的三份 lr_floor20 本机实验启动脚本。

5. 显式指定本机可用的物理 GPU；可编辑每个 eN.sh 顶部默认编号，
   或使用 CUDA_VISIBLE_DEVICES=<卡号> bash scripts/ft_ablation/eN.sh。
   外部已设置的 CUDA_VISIBLE_DEVICES 优先于脚本默认值。
   六个实验每组占一张卡，默认编号 0–5 需要按本机实际情况调整。

6. 先逐个加 --dry-run 核对 e1.sh 至 e6.sh。
   预期 train/val/test=1960/268/560，每 epoch 123 次 optimizer updates，
   split=test、full_split、全部指标、两个 warm-up 各 77。
   E1/E3 总预算 2460 steps，其余 1230；E2/E3/E5 hold=500，其余为 0。
   E1/E2/E3 峰值 LR 均为 2e-5，E4/E5 均为 3e-5，E6 均为 5e-5。
   还需核对数据划分身份，不能只比较样本数量。

7. 完成同步和预检后，汇报 commit、本机路径配置、六组计划和运行命令。
   正式训练由用户手动在各 tmux 窗口运行对应 Bash；不自动启动六组正式任务。
   每组必须从相同预训练资产新启动，不恢复其他实验的 checkpoint。
   同名运行目录已有结果时，用新 FT_RUN_TAG，不能覆盖旧结果。
```

## 11. 发布前验证记录（2026-10-17，Asia/Shanghai）

### 自动回归检查

结果：**123 passed，1 deselected**。覆盖训练日程、阶段冻结/解冻、Adam 状态、各阶段 checkpoint 恢复、早停、完整 test 范围与指标检查，以及六个 Bash 从其他工作目录运行时的实际参数传递。

```bash
CUDA_VISIBLE_DEVICES='' PYTHONNOUSERSITE=1 OMP_NUM_THREADS=4 \
  "$PROJECTOR_PYTHON" -m pytest -q \
  tests/test_joint_training.py tests/test_baseline_sweep.py \
  tests/test_config.py tests/test_evaluate_checkpoint.py \
  tests/test_ft_ablation_launchers.py \
  -k 'not launcher_dry_run_overrides_without_creating_run'
```

排除的单项是旧 `run_full_mlp_uncompressed.sh` 的路径依赖测试：该旧脚本硬编码另一服务器的仓库路径，本机不存在。六个新入口的路径与调用检查均已执行。本轮另外修正了阶段步数校验的报错文案，使其兼容已有回归断言。

### 正式配置预检与真实训练 smoke

六个实际 Bash 入口的正式配置 dry-run 均通过，核对到 train/val/test=1960/268/560、123 updates/epoch、完整 test/all、LoRA rank=16/alpha=32，以及第 2 节列出的 LR、500-step 设置和总预算。

短步训练先由各 Bash 入口加 `--dry-run --set ...` 生成独立 smoke 计划，再执行计划中的真实 `src.cli.train` 命令。使用真实 NeuroVFM、Llama 和 MRI 数据，microbatch=1、effective batch=16、BF16、8× 压缩及每组各自的峰值 LR 保持一致。

仅 smoke 计划缩短为：两个 warm-up 各 2 steps；原本有 500-step hold 的组改为 2-step hold；cosine 为 2 steps；`max_epochs=1`、`max_steps=6/8`、`train_limit=128`、`val_limit=2`、`validation_generation_samples=0`。正式脚本参数保持原样。

| 入口 | 实测训练 updates | 实测阶段 | 检查结果 |
| --- | ---: | --- | --- |
| E1 | 6 | projector warm-up → joint warm-up → cosine | 通过 |
| E2 | 8 | projector warm-up → hold → joint warm-up → cosine | 通过 |
| E3 | 8 | projector warm-up → hold → joint warm-up → cosine | 通过 |
| E4 | 6 | projector warm-up → joint warm-up → cosine | 通过 |
| E5 | 8 | projector warm-up → hold → joint warm-up → cosine | 通过 |
| E6 | 6 | projector warm-up → joint warm-up → cosine | 通过 |

- 六组 loss/梯度均为有限值，无 OOM；各组实测最大 allocated 显存约 **17.34 GiB**。
- 核对实际 checkpoint：E2/E3/E5 的 hold 阶段 LoRA 权重完全不变，projector 权重发生更新；六组进入联合训练后 LoRA 均发生更新，projector 的 Adam 更新计数连续。
- 六组使用相同的预训练资产/数据 contract 和相同 smoke train/val 样本，均从头启动，保存了 best.pt 和 last.pt。
- 六个 best.pt 均通过 `evaluate_checkpoint --dry-run --split test --metrics all --sample-fraction 1.0`，解析为完整 560 条 test 样本。
- 正式 77/500/77 边界和 1230/2460 总预算另由回归测试及正式 dry-run 检查；本次没有实际执行 500-step hold 或完整 10/20 epoch 训练，不能据此判断收敛质量。

### 真实评测 smoke

使用 E1 的 smoke best.pt，对 **2 条 test 样本**执行真实生成、Answer CE 和全部离线评分，保留默认 192-token greedy 解码，bootstrap repeats=0。

结果为 `status=complete`、`scope=subset_smoke`：11 个指标字段均为有限值，GREEN/RadGraph/RaTEScore/BERTScore 全部成功 2/2，`skipped_metrics`、`partial_metrics` 均为空。评分依赖和资产 SHA-256 检查也通过。此小样本检查不写入正式实验 Excel；正式六组入口仍自动执行完整 test/all。

### 本机记录与 GPU

本机证据保存在 Git 忽略的 `runs/smoke/ft1017_release_20261017/`：

- `pytest.log`、`e1_formal_plan.json` 至 `e6_formal_plan.json`。
- `smoke_summary.json`、`checkpoint_audit.json`、各组训练日志与 checkpoint。
- `e1_two_test_samples/metric.json`、`evaluation_smoke.log`、`scoring_assets_preflight.log`。

本次训练 smoke 显式分配物理 GPU 0/1，评测使用 GPU 2。本机 GPU 3 查询报 `Unknown Error`，且无物理 GPU 4/5；在本机正式运行前应调整 E4/E5/E6 的默认卡号并避免任务占卡冲突。另一服务器按自身 GPU 配置分配卡号。

### tmux 解释器选择修复

同日用户直接运行时发现：tmux 的 `CONDA_PREFIX` 已指向 projector，但 `command -v python` 仍返回缺少 OmegaConf 的 base Python。此前真实 smoke 显式设置了 `PROJECTOR_PYTHON`，未覆盖这种环境组合。

共享入口现优先选择已激活 Conda 环境的解释器，并保留显式覆盖优先级。新增相关回归检查后，启动脚本测试 **8 项通过**；在用户原来的 `ft01` 窗口执行 E1 正式配置 `--dry-run`，退出码为 0，解释器、1960/268/560 样本数、2460 steps 和 test split 均核对正确。此检查未启动正式训练。

## 12. N1–N6：恒定 LR、模块 LR 分配与 cosine 下限（2026-10-18）

### 已确认配置

历史对照为 Exp1018.6：Projector/LoRA 均为 5e-5，cosine 下限 0.2，10 epochs。

| 入口 | 默认物理卡号（可改） | Projector LR | LoRA LR | 下限比例 | 总 epochs / updates | CE 早停 | 保存 checkpoint | 自动评测 |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- | --- |
| `scripts/ft_ablation/n1.sh` | 0 | 5e-5 | 5e-5 | 1.0 | 20 / 2460 | 关闭 | epoch 5/10/15/20、best | 仅 epoch 10、best |
| `scripts/ft_ablation/n2.sh` | 1 | 1e-4 | 1e-4 | 1.0 | 20 / 2460 | 关闭 | epoch 5/10/15/20、best | 仅 epoch 10、best |
| `scripts/ft_ablation/n3.sh` | 2 | 2e-4 | 2e-4 | 1.0 | 20 / 2460 | 关闭 | epoch 5/10/15/20、best | 仅 epoch 10、best |
| `scripts/ft_ablation/n4.sh` | 3 | 5e-5 | 4e-5 | 0.2 | 10 / 1230 | patience=3，min_delta=0.005 | epoch 10、best | 仅 best |
| `scripts/ft_ablation/n5.sh` | 4 | 5e-5 | 2e-5 | 0.2 | 10 / 1230 | patience=3，min_delta=0.005 | epoch 10、best | 仅 best |
| `scripts/ft_ablation/n6.sh` | 5 | 5e-5 | 5e-5 | 0.5 | 10 / 1230 | patience=3，min_delta=0.005 | epoch 10、best | 仅 best |

所有入口都是独立单卡任务。GPU 编号应按执行服务器修改；例如只有 0–3 号 GPU 的机器，运行 N5/N6 前需修改默认值或用环境变量覆盖。

共同设置沿用 `configs/ft_ablation/avgpool3d_r8.yaml`：AvgPool3D 8×、seed=42、AdamW、microbatch=1、effective batch=16、LoRA rank=16/alpha=32、hold=0。每组从预训练资产独立初始化，`resume=null`、`init_checkpoint=null`。

- 第 1–77 次更新：Projector LR 由 0 升至峰值，LoRA 冻结。
- 第 78–154 次更新：Projector 保持峰值，LoRA 从 0 升至自己的峰值。
- 第 155 次更新起：N1–N3 保持各自峰值；其他组执行带下限 cosine。
- 阶段切换保持同一 AdamW 实例和动量。比例 1.0 使用原有公式，自然退化为常数；日志阶段名仍为 `joint_cosine`。
- N1–N3 的恒定阶段为 2306 updates；N4–N6 的 cosine 阶段为 1076 updates。
- N1–N3 的同预算比较直接用 `epoch_10.pt`，不另选“前 10 轮 best”；N4–N6 只汇报 best。每轮完整 validation CE 仍选择全程 `best.pt`。
- N4–N6 无论达到上限还是提前停止，都只评测 best；即使存在 epoch_10.pt，也不创建第 10 轮的评测任务或汇总行。

### 一键运行与恢复

激活训练环境后，在各自 tmux 窗口运行，例如：

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/ft_ablation/n1.sh --dry-run
CUDA_VISIBLE_DEVICES=0 bash scripts/ft_ablation/n1.sh
```

也可修改脚本顶部的卡号，直接 `bash scripts/ft_ablation/n1.sh`；已导出的 `CUDA_VISIBLE_DEVICES` 优先于脚本默认值。

解释器优先级：显式 `PROJECTOR_PYTHON` → `$CONDA_PREFIX/bin/python` → PATH 上的 Python。tmux 若残留错误的 `PROJECTOR_PYTHON`，先取消该变量或改为当前训练环境的绝对 Python 路径。

默认输出为 `runs/baselines/ft_nN_..._v1/`。已有同名目录会拒绝覆盖。独立重跑用新的 `FT_RUN_TAG`。训练正常结束、仅评测中断时，可继续本实验的评测：

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/ft_ablation/n1.sh --resume-run runs/baselines/本实验目录
```

该选项沿用原训练配置与 checkpoint，并按当前入口的 `ft_eval_checkpoint_epochs` 选择评测范围。N1–N3 为 `(10)`，N4–N6 为 `()`（只评测 best）。实际评测选择写入运行根目录的 `evaluation_plan.json`，原 `sweep_plan.json` 保留；已有直接结果或 best 别名通过身份校验后复用。训练中断时不会自动恢复训练；应通过通用训练入口显式指定本实验的 last.pt 并保持原配置。各组不能从其他实验 checkpoint 初始化。

Git 更新不会改变已经启动的 Python 进程内存中的旧评测列表。若已有任务运行，应先查看状态；不要为了修改评测范围重复启动训练。训练完成后可以用上述 `--resume-run` 按新范围继续评测。

### 输出文件与口径

以下路径相对于每组的 `r8/`：

| 路径 | 内容 |
| --- | --- |
| `joint_lora/checkpoints/epoch_05.pt` 等 | 固定 epoch 权重、AdamW、调度器、RNG 和训练状态 |
| `joint_lora/checkpoints/best.pt` | 全程最低完整 validation CE 对应的模型 |
| `joint_lora/optimizer_diagnostics.jsonl` | 每步两模块裁剪前/后梯度范数、参数范数、实际更新范数、相对更新幅度与 LR |
| `joint_lora/train.jsonl` | 逐步 CE、answer loss sum/token 数、阶段、LR 与相同模块诊断 |
| `joint_lora/epoch_metrics.jsonl` | 每轮 token 加权 Train CE、完整 Val CE、实际 token 和更新数 |
| `joint_lora/validation.jsonl` | 每轮完整验证与固定 14 条生成监控 |
| `evaluation/epoch_10_test/metric.json` 等 | 完整 test、全部 11 个结果字段、各指标覆盖信息 |
| `evaluation/*_test/generation_green_diagnostics.json` | 生成长度、空回答/截断、GREEN 成功/失败数和六类错误均值 |
| `evaluation/*_test/generation_green_samples.csv` | 逐样本生成长度、GREEN 匹配与六类错误，失败分数留空 |
| `checkpoint_results.json`、`checkpoint_results.csv` | N1–N3 为 epoch_10/best 两项；N4–N6 仅 best 一项，按指标键名取值 |

诊断日志与 epoch 汇总的 epoch 从 1 开始；历史 `train.jsonl` 的 epoch 保持从 0 开始。`encoder` 诊断只覆盖优化器中的视觉 LoRA，不包含冻结视觉主干。

相对更新幅度为 `||theta_after - theta_before||_2 / ||theta_before||_2`，包含 AdamW weight decay 和实际存储精度影响。分母为 0 时记录 null；冻结 LoRA 的更新范数为 0。诊断不抽样、不改动参数/梯度、不消耗 RNG，只复制优化器参数，不复制冻结 LLM。

固定 epoch 5/10/15/20 对应 step 615/1230/1845/2460。**保存与评测分开：N1–N3 保留这四个 epoch 及 best，自动评测 epoch_10.pt 和 best.pt；N4–N6 仅自动评测 best.pt。第 5/15/20 轮不作为独立评测任务。** best 可来自任意轮次，始终按 best 身份评测。所有选中项均使用完整 test（当前 560 条）及全部指标；N1–N3 的两项权重相同时复用。模型与评分器分进程顺序加载。

N1–N3 如果 best 与 epoch 10 的配置、step、预训练 contract 和全部 Projector/LoRA 权重完全相同，复用第 10 轮的评测。对应目录写 `alias.json` 指向真实 `metric.json`；汇总同时记录请求 checkpoint SHA256 与实际评测 SHA256，不伪造独立评测。last.pt 和 boundary checkpoint 继续作为恢复辅助保留，不增加正式结果组。

GREEN 沿用现有评测模型与协议。格式失败样本排除在 GREEN 均值之外并记录分母，不填 0。诊断核对六类计数与 GREEN 公式，不修改评审输出。

### 另一台服务器部署说明

1. 在当地仓库查看 `git status --short --untracked-files=all` 和 `git diff`，保留本地改动。当地模型、数据、缓存位置放进 Git 忽略的 `configs/local_paths.yaml`，备份原文件；不要复制源服务器绝对路径。必要时对已审阅的代码改动用有名字的 stash/备份，不能 reset --hard、clean 或整体覆盖本地文件。
2. 执行 `git fetch origin main`、`git pull --ff-only origin main`、`git rev-parse HEAD`；核对提供的发布 commit 是 HEAD 或其祖先。不能快进时先审阅分叉，保留本地工作后合并。
3. 阅读本节。确认 local_paths.yaml 仍被忽略且原路径有效；保留当地 `assets/models/evaluation`、`third_party/evaluation` 等评测资产。模型、数据、checkpoint、Excel 和结果均不通过 Git 同步。
4. 激活当地训练环境，核对实际解释器。通过 `CUDA_VISIBLE_DEVICES` 分配当地可用 GPU；可用 `PROJECTOR_OUTPUT_ROOT` 指定当地输出位置。
5. 逐个执行 n1.sh 至 n6.sh 的 --dry-run，每条显式指定当地可用 GPU。核对本节表中的 LR、预算、下限、早停、保存列表，以及 evaluations 在 N1–N3 为 epoch_10/best、N4–N6 仅为 best；核对 1960/268/560、123 updates/epoch、77/0/77、hold_peak、test/full_split/all 和计划中的数据身份摘要。
6. 预检通过后报告启动命令，正式训练由用户在 tmux 手动启动。六组独立从相同预训练资产启动。

标准划分 ID 列表摘要（按清单顺序，算法 src.config.digest）：

```text
train: b60c751801325bed594b142e0662233202fce2915273fa9ea259667cc0a22698
val:   4d3176b18672d466be6e0f3a239166f057d4fa9c10ba3bb95f23c0d0d32c5e9e
test:  69bcaa3c89157310951f55ae7ed6fc2ec691ea554024ea770a780ea8257b1462
```

### N 系列初版发布验证（09a2b75）

- 相关测试 **170 项通过**：六入口参数、真实 AdamW 全预算 LR 轨迹、阶段动量保持、关闭/保留早停、实际 Trainer 循环的固定 epoch/best 保存、模块诊断无副作用、token 加权 CE、完整评测约束、重复 checkpoint 复用、失败中止与评测恢复、旧 E 系列回归。
- N1–N6 六个 Bash 的正式配置均使用本机真实清单和 MRI 头信息完成 dry-run。1960/268/560、123 updates/epoch、总预算、阶段边界、早停和保存列表均与本节表一致，三份 ID 摘要一致；未生成正式运行目录。
- 新诊断读取已有 E4/E5/E6 的全部 1680 条 test 逐样本记录，核对 checkpoint/划分身份、有效 GREEN 均值和计数公式。成功数为 557/556/557，失败数为 3/4/3，与原评分一致；原实验文件保持原样。
- GREEN、RadGraph、RaTEScore、BERTScore 的本机依赖和模型文件 SHA256 检查通过。
- 证据保存在 Git 忽略的 `results/FT_N_series_preflight_20261018/`，包含 pytest.log、六份 dry-run 计划、摘要、已有结果复核和评测资产记录。

本轮 Trainer 集成验证使用小型 CPU 模型；没有运行新的完整 VLM 训练或新评分推理。六组正式训练仍由用户手动在 tmux 启动，当前验证不预言高 LR 的实际收敛质量。

### 历史评测范围修订（b2885c9，已由下节分组规则修正）

上一版将 N1–N3 的评测限制到第 10 轮和 best，但误将 N4–N6 也统一为两项评测。本次按用户要求改为 N4–N6 仅 best；以本节表格及下方分组规则为准。

修订后相关测试 **173 项通过**。新增覆盖 best 来自第 10 轮时复用、best 来自第 15 轮时单独评测、失败后恢复，以及第 5/15/20 轮文件存在但不产生独立评测目录。六个入口的真实数据 dry-run 再次全部通过，参数和样本身份一致；未启动正式训练。该版验证记录位于 Git 忽略的 `results/FT_N_series_eval_policy_20261018/`，其中六份计划同时列出保存列表与两项评测任务。

### 最终分组评测规则（2026-10-18）

- **N1–N3：自动评测 epoch_10.pt 和 best.pt。** 保存第 5/10/15/20 轮和 best。
- **N4–N6：只自动评测 best.pt。** 不创建 epoch_10 的评测任务或结果行；原 checkpoint 保存配置保留。
- 各脚本显式设置 `ft_eval_checkpoint_epochs`：N1–N3 为 `(10)`，N4–N6 为 `()`。共享入口将它传给 `--eval-checkpoint-epochs`，不会根据训练轮数或 LR 猜测评测范围。
- `--resume-run` 按当前入口选择评测，保持原训练配置和原始 sweep_plan.json；实际范围记录在 evaluation_plan.json。旧版已有 best 评测或指向第 10 轮结果的 best 别名，均核对身份后复用。

本次相关测试 **176 项通过**，覆盖六入口参数、N4–N6 正常结束/早停后均只评测 best、N1–N3 的双 checkpoint 评测和复用、旧计划变更评测范围而不更改训练配置，以及旧别名复用。六个实际 Bash 入口的真实数据 dry-run 全部通过：N1–N3 的 evaluations 为 `[epoch_10, best]`，N4–N6 为 `[best]`；训练参数、保存列表和数据身份保持一致。证据位于 Git 忽略的 `results/FT_N_series_best_only_20261018/`。本次未启动正式训练。
