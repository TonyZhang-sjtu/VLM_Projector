# NeuroQA VQA 实验入口

## 当前默认：完整 v2、保留待复核警告（2026-10-31）

按用户要求，`vqa_*.sh` 默认选择 **v2 + `include_unreviewed` + `exact_balanced_repeat`**。`include_unreviewed` 包含所有尚未复核的 QC warning 病例，不将其标记为复核通过；预检时自动过滤可归属于病例的影像缺失、损坏或审计失败，并记录病例及受影响 QA／视图。全局协议或审计证据不完整、不可归属故障以及没有可用训练或验证数据仍会报错。输出标记 `unreviewed_training` / `unreviewed_evaluation`，全局 `ready_for_full_training` 保持 false。数据来源仍是 BraTS 单序列 QA 子集及现有患者 guard 划分，不是 NeuroQA 完整官方 leaderboard 集合。

`exact_balanced_repeat` 保证每个 epoch 的 T1/T2 等量，且每个偶数大小梯度累积窗口各半；当前全量训练 3454 条 QA、各 1727 条，每题每 epoch 出现一次。过滤后若数量为奇数或单模态配额不平衡，按需最少重复有效 QA 以保持平衡，重复次数和 ID 保存在采样计划中。全量模式默认 batch=16，因此每步 8/8，最后窗口 7/7。v1 如需过去的过滤行为，请显式使用 `--preprocessing-version v1 --invalid-sample-policy skip`；v1 不支持 `include_unreviewed`。下面历史的“默认 v1 / 默认 skip / dual_eligible”描述以本节为准。

训练开始后的数据读取：每个更新窗口先加载影像，遇到能够归属于病例的缺失或已知损坏时，记录并跳过病例，再用同模态有效 QA 填补窗口，必要时重复，以保证反向传播前仍然平衡。`train/runtime_data_events.jsonl` 保存原因与替换，checkpoint 保存已发现坏病例。全局协议错误、未知异常及 GPU OOM 仍直接报错。一个 16 条窗口需要预读约 600 MiB CPU patch。运行中发生的文件损坏若仍存在，断点恢复可能因数据清单指纹变化而被阻断；验证/测试中启动后新发生的文件错误也尚不自动跳过，不能把这两种情况当成已支持。

`--eval-controls normal text zero shuffle` 用同一个 best checkpoint 分别评估相同 QA 视图。normal 是匹配影像；text 不加载图像且不注入视觉 token（提示仍包含序列信息及图像措辞）；zero 对真实影像编码后将投影 token 置零，保留视觉 token 数与位置；shuffle 给目标题配同序列、不同患者的 donor 影像，donor 不保证一一对应或答案不同。这些控制可分析图像依赖及匹配关系，不等价于训练一个独立 text-only 模型，也不单独证明视觉因果性。

```bash
# 小样本工程闭环；不属于正式性能结果
CUDA_VISIBLE_DEVICES=0 bash scripts/Exp_VQA/vqa_smoke.sh --run-tag v2_smoke
# 仅检查完整清单，不加载 GPU 模型
bash scripts/Exp_VQA/vqa_avgpool3d.sh --check-data --ratios 128
# 启动全量未复核实验；不传 --ratios 时按各 shell 的默认倍率顺序训练并测试
CUDA_VISIBLE_DEVICES=0 bash scripts/Exp_VQA/vqa_avgpool3d.sh --ratios 128 256 --run-tag v2_all_s42
# 明确纳入未复核评测到汇总（默认不纳入）
/mnt/sda/mingjie/conda_envs/projector/bin/python scripts/vqa/neuroqa/summarize_results.py \
  --include-unreviewed --root results/vqa/neuroqa --output results/vqa/neuroqa/leaderboard_unreviewed.csv
```

每个 `vqa_*.sh` 顶部都可直接设置物理 GPU、projector、倍率、预处理版本、学习率、batch、epochs、LoRA、调度和早停。**默认运行训练，再用验证集选出的 `best.pt` 测试；`--dry-run` 才是只预览。** 各倍率顺序执行、独立初始化。默认 `include_unreviewed` 保留 v2 待复核病例并过滤有明确错误的病例；`skip` 同时过滤未复核病例，`strict` 要求全部 QC 已验收。模型/优化器错误不会被当作坏样本吞掉。

## 入口与预算

| Shell | Projector | 默认倍率 |
| --- | --- | --- |
| `vqa_avgpool3d.sh` | avgpool3d | 128、256 |
| `vqa_concat3d.sh` | concat3d | 128、256 |
| `vqa_local_packer3d.sh` | local_packer3d | 128、256 |
| `vqa_perceiver_nv.sh` | perceiver_nv | 128、256 |
| `vqa_convpool3d.sh` | convpool3d | 全五档 |
| `vqa_mixer3d.sh` | mixer3d | 全五档 |
| `vqa_attention_pool.sh` | attention_pool | 全五档 |
| `vqa_perceiver_flamingo.sh` | perceiver_flamingo | 全五档 |
| `vqa_adaptive_merge3d.sh` | adaptive_merge3d | 全五档 |
| `vqa_token_select3d.sh` | token_select3d | 全五档 |
| `vqa_bindformer3d.sh` | bindformer3d，knowledge_mode=none | 全五档 |
| `vqa_baseline_a.sh` | BindFormer Baseline A 视觉结构，knowledge_mode=none | 128、256 |
| `vqa_mlp2x.sh` | mlp2x_gelu | 1，不压缩 |
| `vqa_smoke.sh` | 默认 avgpool3d | 256，固定小题集 |

所有压缩方法均支持 `--ratios 16 32 64 128 256`；默认值见表，也可编辑 shell 中的数组。当前两版模型输入都是 N=9555；后续若改变网格，必须修订预算协议，不能沿用这些数值。

| 名义倍率 | 实际 M | 空间输出网格 DHW | Concat 分组 DHW |
| --- | ---: | --- | --- |
| 16 | 624 | 6×8×13 | 9×2×1 |
| 32 | 288 | 9×8×4 | 6×2×4 |
| 64 | 150 | 6×5×5 | 9×3×3 |
| 128 | 75 | 3×5×5 | 17×3×3 |
| 256 | 36 | 3×4×3 | 18×4×5 |

### BindFormer Baseline A

`vqa_baseline_a.sh` 对齐 report 的 `scripts/Exp1026/run_baseline_a_r128_r256.sh` 所启用的视觉结构：`router_value_normalization=true`、`router_value_projection=false`、`fixed_spatial_regions=true`、`spatial_mean_aggregation=true`、`spatial_adaptive_aggregation=false`，temperature=1.0。额外 slot SA、local→global CA、geometry 关闭；global_fraction=0.25、slot_iterations=3、每角色 1 query、MLP head、全部辅助损失权重为 0。固定区域由 BindFormer 按其 slot 数划分，不能将上表的 pooling 输出网格直接当成它的区域网格。

这里明确采用 **Baseline A 视觉结构 / 无外部知识库** 的 VQA 版本。`knowledge_mode=none` 不加载 UMLS concept bank，不执行概念检索或向 role query 注入概念向量；可学习的角色 query、序列条件和图像特征处理保留。report Baseline A 使用 `name_definition`，其知识库按 report train 构建，不能直接视为符合 QA split 的知识库。当前 VQA 配置检查要求 `none`；启用知识需先按 QA 训练患者重建并审计来源，再另建实验。

优化器/调度沿用本目录的共同 VQA 恒定 LR 配方（2026-11-01）：projector/encoder LoRA 峰值 LR 均为 2e-4、两段 warmup 各固定 150 次 optimizer update、无 projector hold、joint warmup 时 projector 保持峰值、之后两者恒定 2e-4。QA answer-only CE 和 val closed accuracy 选模保留。脚本顶部可以编辑所有训练参数，CLI 覆盖优先；外部 `BIND_*` 环境变量不会改变该入口的结构默认值。

```bash
# 预览最终 projector 参数、预算、训练参数和测试命令
bash scripts/Exp_VQA/vqa_baseline_a.sh --dry-run --ratios 16 32 64 128 256

# GPU 0、v1 过滤模式；依次训练 r128/r256，再各自测试 best.pt
CUDA_VISIBLE_DEVICES=0 bash scripts/Exp_VQA/vqa_baseline_a.sh --preprocessing-version v1 --invalid-sample-policy skip
```

默认 tag 以 `baseline_a_no_knowledge_` 开头，包含北京时间和随机后缀；模型目录仍为 `bindformer3d`，通过 tag 和保存的 `launch_config.yaml` 区分变体。`PROJECTOR_RUN_TAG` 或 `--run-tag` 可自定义（建议保留变体名）。原 `vqa_bindformer3d.sh` 保留原来的 competitive routing 配置。续训仍恢复保存的配置，不能通过此新脚本把旧实验中途转换成 Baseline A。

## 配置和运行

以下命令从仓库根目录运行；使用 shell 的绝对路径时也可以从其他目录启动。`--set`、`--ratios` 等命令行选项优先于 shell 默认值。`--dry-run` 不写文件，打印的命令引用的是实际启动时才会生成的 `launch_config.yaml`，不要单独复制该子命令。

```bash
# 查看最终设置、N/M 和训练/测试命令，不占用 GPU
bash scripts/Exp_VQA/vqa_concat3d.sh --dry-run --ratios 16 32 64 128 256

# 读取 manifest、完成标记、配准协议、影像头与 QC readiness，不加载模型
bash scripts/Exp_VQA/vqa_avgpool3d.sh --check-data --ratios 128

# 完整验收后按严格模式运行
CUDA_VISIBLE_DEVICES=0 bash scripts/Exp_VQA/vqa_avgpool3d.sh \
  --preprocessing-version v2 --invalid-sample-policy strict --ratios 128 256 --run-tag main_v2_s42 \
  --set training.projector_lr=2e-4 --set training.encoder_lr=2e-4 \
  --set training.max_epochs=20 --set training.effective_batch_size=16

# 扩展控制实验，每项使用同一 best checkpoint；分别保存结果
CUDA_VISIBLE_DEVICES=0 bash scripts/Exp_VQA/vqa_token_select3d.sh \
  --preprocessing-version v2 --ratios 128 --eval-controls normal text zero shuffle

# 只训练；之后可按下面“补测”命令评估
bash scripts/Exp_VQA/vqa_mixer3d.sh --ratios 128 --no-evaluate-test
```

预处理版本选择也可写在 shell 的 `vqa_preprocessing_version`，或用环境变量 `VQA_PREPROCESSING_VERSION=v2`。

| 版本 | 配置 | manifest 目录末级 | 影像与 patch 缓存目录末级 |
| --- | --- | --- | --- |
| v1 | `configs/vqa/neuroqa/mixed.yaml` | `official_holdout_patient_guard_v1` | `neuroqa_mni1mm_v1` |
| v2 | `configs/vqa/neuroqa/mixed_v2.yaml` | `official_holdout_patient_guard_v1_mni_v2` | `neuroqa_mni1mm_v2` |

manifest 位于 `assets/data/NeuroQA_derived/`；缓存位于 `assets/preprocessing/`。manifest 内含影像绝对路径，因此不能只改 `preprocessing.cache_dir`。自定义存储需要同步设置 shell 的 `vqa_manifest_dir`、`vqa_cache_dir`，或对应 `--set data.manifest_dir=... --set preprocessing.cache_dir=...`；preflight 会检查 lineage、每条影像路径和实际配准协议。**不会自动执行配准或改写 v1/v2。** v2 全量 ready 由正在进行的预处理工作生成；`--dry-run` 成功不代表 v2 影像已就绪。

默认冻结 Llama，训练 projector 和视觉 encoder LoRA。仅单模态可答的题固定使用允许模态，双模态题分配给有缺口的模态并随 epoch 轮换；`exact_balanced_repeat` 保持 epoch 和偶数累积窗口内 T1/T2 等量，必要时最少重复有效 QA。验证/测试按所有允许视图展开，不要求等量。有效 batch=16、microbatch=1；全量时最后 14 条按真实样本数平均。训练目标为每 QA 平均的 answer-only causal CE。验证集 closed accuracy 选 best，同准确率时比较 CE；早停 `min_delta=0.002` 是 accuracy 的绝对增量。

默认最多 20 epochs，projector/encoder LoRA 峰值 LR 均为 2e-4，LoRA rank=16、alpha=32、dropout=0.05。2026-11-01 起，正式方法入口及 VQA 配置均使用 `projector_warmup_steps=150`、`joint_warmup_steps=150`、`projector_hold_steps=0`、`joint_warmup_projector_mode=hold_peak`、`cosine_min_lr_ratio=1.0`；不再按总更新数的 5% 计算预热。配置保留 `*_warmup_ratio=0.05` 作为可选回退，只有显式把对应 `*_warmup_steps` 设为 `null` 时才生效。

完整 3454 QA、有效 batch=16 时每 epoch 216 次更新，20 epochs 共 4320 次：

| optimizer update（从 1 编号） | Projector LR | Encoder LoRA LR / 状态 |
| --- | --- | --- |
| 1–150 | 0 → 2e-4 | 冻结，LR=0 |
| 151–300 | 恒定 2e-4 | 解冻，0 → 2e-4 |
| 301–4320 | 恒定 2e-4 | 恒定 2e-4 |

最后阶段共 4020 次更新。实现沿用 `joint_cosine` 阶段名，但下限系数 1.0 使实际 LR 恒定；预热端点包含 0 与峰值。过滤后总预算按实际采样数计算，两段预热仍各 150，不随样本数变为百分比。VQA 与 report 的 step 都表示一次 optimizer 更新，当前有效 batch 均为 16；VQA 的每 epoch 更新更多，不是一次 step 内有两次 optimizer 更新。本次按指定值使用 150，report 的 77 严格翻倍为 154。

固定预热的 `joint_lora` 新实验需至少 302 次总更新。小样本工程入口 `vqa_smoke.sh` 继续使用 2+2+2 共 6 次更新，以检查阶段切换；峰值 LR 均改为 2e-4，末段同样恒定。`training_stage=a` 可做 projector-only 对照，建议同时设 `encoder_lr=0`；它使用原有的 `training.warmup_steps` / `training.warmup_ratio` 调度，不使用这里的 joint 分阶段配方。旧实验的 `--resume-run` 恢复原保存配方，不会套用新的 150+150 设置；使用新配方必须新建 run。

GPU 编号使用物理编号，进程内统一为逻辑 `cuda:0`。这是单进程入口，不使用 torchrun/DDP。Concat 默认 CPU projector + 单张 GPU，保持完整宽度；CPU 内存和时间开销较大。若资源另行获准，双卡模式为 `CUDA_VISIBLE_DEVICES=0,1 ... --concat-device cuda:1`。其他方法设置一张卡。`PROJECTOR_PYTHON` 可覆盖 Python 路径。入口不自动合并报告任务的 `configs/local_paths.yaml`；更换模型路径用 `--set encoder.path=... --set decoder.path=...`。

## v1 当前能跑到哪里

2026-10-30 全量核验：7 个病例缺少有效完成标记/产物，涉及 train 68 条 QA、val 12 条 QA（21 个视图），test 不受这 7 例影响。默认 skip 下实际 train=3386 QA、val=762 views、test=785 views，可开始过滤后的实验；strict 下仍以状态码 2 阻断。原始 manifest 不修改，未将全局 ready 标记改为 true。

## 不合格样本模式

每个 shell 顶部提供 `vqa_invalid_sample_policy`，当前默认 `include_unreviewed`，另支持 `skip` 和 `strict`。`include_unreviewed` 保留待复核的警告，自动过滤可定位至病例的损坏或缺失，并不自动接受 QC。

```bash
# 默认：跳过失败病例，打印 warning，继续使用其余数据
bash scripts/Exp_VQA/vqa_avgpool3d.sh --check-data
CUDA_VISIBLE_DEVICES=0 bash scripts/Exp_VQA/vqa_avgpool3d.sh --ratios 128 --run-tag v1_filtered_s42

# 显式选择严格模式：全局未验收或所选病例失败则阻断
bash scripts/Exp_VQA/vqa_avgpool3d.sh --invalid-sample-policy strict --check-data

# 环境变量也可选择；CLI --invalid-sample-policy 优先
VQA_INVALID_SAMPLE_POLICY=skip bash scripts/Exp_VQA/vqa_concat3d.sh --dry-run
```

直接 `src.cli.vqa_train` 和 `src.cli.vqa_evaluate` 也接受 `--invalid-sample-policy`；评测默认继承 checkpoint 配置。shell 中也可改值。全局未验收在 skip 下只 warning，保留病例仍逐一检查 T1/T2 影像、mask、complete/status、配准来源、几何、哈希、patch 有限值及坐标；已有 QC 待审队列中的未接受病例也跳过。以配准病例为单位过滤，任一序列失败则该病例的相关 QA/视图都移除，不暗改题目的 allowed_modalities。

过滤发生在采样和梯度累积之前，固定一次实验的数据集合；不在 backward 失败后临时跳 batch。没有任何可用训练样本、没有可用于选模的 closed val、模型配置或全局 manifest/预处理协议错误仍报错。显式选择 strict 时，小规模 smoke 也要求全局 ready；需要用未全局验收的数据做工程验证时使用默认 skip。

训练目录及测试结果目录各有 `data_preflight.json`、`skipped_samples.json`，保存病例、QA/view ID、原因、原始/保留数量及实际清单指纹。未完整验收或发生训练/验证过滤的完整训练标为 `filtered_training`；对应结果标为 `filtered_evaluation`，保存总体覆盖数。默认排行榜不混入这些结果；可用 `summarize_results.py --include-filtered` 明确纳入并保留范围列。

历史 `data.modality_sampling=dual_eligible` 仅对双模态可答题做平衡；当前 v2 默认及 Baseline A 使用本文开头的 `exact_balanced_repeat`，保证全局及偶数窗口的 T1/T2 等量。旧 checkpoint/launch 配置缺少此字段时按 `legacy_exact` 恢复原采样，不把旧实验中途改成新策略；采用新策略应新建 run。已保存的实际清单和影像内容指纹在续训时核对，修复病例导致清单变化会明确报错，要求新建实验。

现在可以用固定已完成病例检验工程闭环：

```bash
bash scripts/Exp_VQA/vqa_smoke.sh --check-data
CUDA_VISIBLE_DEVICES=0 bash scripts/Exp_VQA/vqa_smoke.sh --run-tag v1_smoke_try1
# 可切换方法；Concat 默认仍为 CPU projector；MLP 需同时设 --ratios 1
# bash scripts/Exp_VQA/vqa_smoke.sh --projector perceiver_nv --ratios 128 --run-tag nv_smoke_try1
```

固定清单在 `selections/`：train 4 条 QA、val/test 各 6 个 T1/T2 视图，覆盖 YN/MCQ/Open。运行 3 epochs、6 次更新；生成上限 8 token 只为验证接口。它不能证明收敛或作为正式性能结果。任意训练 limit、IDs 子集或 max_steps 都标为 smoke；即使之后评完整 test，也不会被当成正式 benchmark。

## 文件布局、续训和补测

终端训练输出使用单行进度条，例如 `Epoch 1/20: ... 8/212 [ETA 26:08, CE=2.6734]`（示意）。CE 是当前梯度累积窗口的每 QA 平均 answer-only CE；ETA 是本 epoch 剩余训练时间，不含后续验证、其他 epoch 或下一个倍率。验证和独立测试分别显示自己的进度条及 ETA；验证结束仅输出一行 val CE/accuracy/best。启动摘要只打印一次，不再每步打印完整 JSON。

每步梯度、学习率、模态计数、token 数、显存和耗时仍完整写入 `train/train.jsonl`，验证记录写入 `train/validation.jsonl`。launcher 保留回车刷新和 UTF-8，`logs/train.log` 保存相同终端流。已启动的 Python 进程不会热更新；下一次启动或通过 `--resume-run` 恢复后使用新显示格式，显示修改不改变训练配方或数据清单。

新入口统一使用：

```text
runs/vqa/neuroqa/<split协议>/<v1或v2>/<training或smoke>/mixed/<projector>/r<R>/s<seed>/<tag>/
  launch.json                 # 固定的运行/评测计划
  launch_config.yaml          # 最终解析配置
  data_preflight.json         # 实际数据清单、双模态可答配额、调度与检查
  skipped_samples.json        # 跳过病例/QA/view、原因、原始与保留数量
  launcher_status.json        # completed / failed 及错误
  logs/train.log
  logs/eval_test_normal.log
  train/config.yaml
  train/checkpoints/{best,last}.pt
  train/{train,validation}.jsonl
  train/sampling/epoch_*.json
  train/validation/epoch_*/...
  train/result.json

results/vqa/neuroqa/<同样的层级>/<tag>/test/<normal或其他control>/
  metrics.json                # QA 指标及训练/评测范围
  ...                         # predictions、官方格式、分组指标、来源信息
  evaluation_complete.json    # 成功完成后的 checkpoint 哈希与实际命令
```

训练 `last.pt` 默认每 50 次更新及每 epoch 保存，`best.pt` 按验证指标保存。`VQA_OUTPUT_ROOT` / `VQA_RESULTS_ROOT` 可更换输出根目录。相同 tag 的已有实验不会覆盖。旧 `scripts/vqa/neuroqa/run_experiment.py` 和旧产物继续保留；新目录层级仅用于 `Exp_VQA`。

```bash
# RUN 指含 launch.json 的某个倍率实验目录，不是其 train/ 子目录
vqa_run_dir=/绝对路径/到/某个实验tag

# 从 last.pt 恢复；训练已结束则只补未完成评测
bash scripts/Exp_VQA/vqa_avgpool3d.sh --resume-run "$vqa_run_dir"

# 只对该实验 best.pt 补测；训练默认值不会影响已保存模型/数据配置
bash scripts/Exp_VQA/vqa_avgpool3d.sh \
  --resume-run "$vqa_run_dir" --evaluate-only

# 先前评测目录若残留不完整输出，明确使用新的目录重试
bash scripts/Exp_VQA/vqa_avgpool3d.sh \
  --resume-run "$vqa_run_dir" --evaluate-only --eval-tag test_retry1
```

`--resume-run` 恢复已保存的 recipe 和 evaluation；shell 的新训练默认值、`--set`、版本选择等不会覆盖它。需要改配方或换预处理时建立新 tag。续训时选定相同数量的可用 GPU。已成功评测且 checkpoint 哈希/命令一致时自动跳过，不重复写结果。改变评测 split/control/子集可直接使用 `python -m src.cli.vqa_evaluate --help` 指定保存的 checkpoint 和全新 output；该命令从 checkpoint 恢复模型及数据版本。

汇总为 CSV（默认仅完整验收/完整清单结果；加 `--include-filtered` 可纳入明确标记的过滤实验，始终排除 smoke）：

```bash
/mnt/sda/mingjie/conda_envs/projector/bin/python scripts/vqa/neuroqa/summarize_results.py \
  --root results/vqa/neuroqa --output results/vqa/neuroqa/leaderboard.csv
```

启动前检查包括影像/mask 哈希和实际 patch 验证，但不替代全量配准质量审阅；skip 不代表该版本已完成质量验收。恢复与评测继续沿用数据、预处理、prompt 和模型 provenance 校验。
