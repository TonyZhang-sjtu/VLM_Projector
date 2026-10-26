# NeuroVFM → Projector → Llama 单序列 MRI Findings

源码直接位于 `src/`，不再增加 `src/vlm_projector/` 层。默认完整体积、native-grid、最小 padding、不删背景 token；LLM 全程冻结，保留视觉输入梯度。活动监督版本为 `single_sequence_findings_coref_v2`。

## 环境与入口

新服务器的环境创建、CUDA 扩展编译、评测依赖和独立资产恢复见
[环境与迁移说明](docs/ENVIRONMENT_AND_MIGRATION.md)。Git 不包含数据、标注、模型、缓存、
训练输出或登录 token。仓库 checkout 路径由 `project_root: null` 自动识别。

本轮（2026-10-26）上传与双机同步、Exp1024 完整调用链及资产/环境核对，见
[GitHub 上传与服务器同步流程](Pipeline_0929/GITHUB_UPLOAD_AND_SERVER_SYNC.md)。

已有服务器同步此前 64× / BindFormer 更新并验证运行，见
[第二台服务器操作 prompt](docs/SERVER2_SYNC_64X_20261021.md)，沿用各机自己的路径配置。

```bash
source /home/mingjie/anaconda3/etc/profile.d/conda.sh
conda activate projector
cd /mnt/sda/jyzhang/VLM_Projector
python -m pip install --no-deps -e .
python -m pip check
projector-config --config configs/training/stage_a.yaml
```

`environments.yaml` 和 `environment-lock.json` 记录当前环境；安装顺序及迁移说明见 `docs/ENVIRONMENT_AND_MIGRATION.md`。原 `archive/non_pipeline/` 安装记录保留本地，不进入 Git。单卡默认使用逻辑设备 `cuda:0`，通过外部 `CUDA_VISIBLE_DEVICES` 选择显卡。例如 `CUDA_VISIBLE_DEVICES=3 bash scripts/baselines/run_mlp2x.sh` 会在选中的 GPU 3 上训练并评测，无需再传 `--device`。程序中的设备编号相对于可见设备列表；直接运行 Python CLI 而不设置该变量时使用 CUDA 枚举的第一张卡；当前 baseline Bash 保留原有物理 GPU 3 默认值，外部设置可覆盖。多卡训练按 `LOCAL_RANK` 选择逻辑设备。

每个 `--config` 合并一个 YAML，每个 `--set` 接收一个 `key=value`；可重复使用。未知字段和不支持的组合会报错。运行保存 resolved config、模型/数据/代码 SHA、tokenizer、样本选择及训练参数清单。

## 文件结构与替换接口

```text
configs/                  # base + data/preprocessing/encoder/projector/decoder/task/training/evaluation
src/
  config.py               # 严格配置、参数校验、路径与哈希
  data/                   # manifest、序列映射、完整体积 Dataset
  preprocessing/          # registry、物理几何、patch、缓存、mask QA
  models/
    encoders/             # EncoderAdapter、NeuroVFM、LoRA、registry
    projectors/           # 11 种可替换 projector、几何分桶、配置校验、registry
    decoders/             # Llama 文本边界、答案 CE、逐样本 KV-cache 生成
    multimodal.py         # 完整体积 → 视觉特征 → Projector → Llama
  tasks/                  # prompt 和监督目标
  training/               # 两阶段、AdamW、DDP token 归一化、checkpoint、resume
  evaluation/             # 生成、分层指标、患者 bootstrap、视觉对照
  cli/                    # 可运行命令
scripts/                  # 全量训练、单阶段训练、DDP、评价、预处理 artifact
tests/                    # 配置/数据/几何/梯度/CE/恢复/生成/评价契约
runs/                     # 当前/历史完整实验；工程验收已集中到 runs/archive/
archive/non_pipeline/     # 一次性 smoke、audit、下载/安装工具及其源代码
assets/preprocessing/     # 正式共享的按内容哈希隔离的 patch tensor cache
cache/features/           # Stage A Encoder 特征缓存；独立于 patch cache
cache/projector_assignments/ # 可复用的 projector 路由/分组索引缓存
```

Preprocessing patch tensors 默认从 `assets/preprocessing/<preprocessing_version>/` 读取和写入。训练、评测及预处理命令均可传 `--no-use-cache`（或 `--no-cache`）禁用读写，`--use-cache` 显式启用；关闭缓存不会删除已有张量。改变 spacing、边界或强度等变换配置会生成新的 preprocessing version。`cache/features/` 是 Encoder 输出缓存，使用独立配置和身份，不等同于原始 patch cache。

替换 Projector：继承 `src/models/projectors/base.py` 的 `Projector`，接收 `(features, coords, metadata)`，返回 `ProjectedVolume`，使用 `register_projector(name,factory,defaults)` 注册后用 `projector.name` 选择；额外参数放入 `projector.params`。输出最后一维必须等于 Llama hidden size。现已支持 MLP、3D 均值池化/Concat/Conv/Mixer、LocalPacker、AttentionPool、两种 Perceiver、自适应合并和选择对照。结构、预算规则、验收及设计取舍见 [baseline 文档第 13 节](Pipeline_0929/PROJECTOR_ROUTES_AND_BASELINES.md)。

```bash
bash scripts/train_projector.sh avgpool3d joint_lora runs/baselines/avgpool3d/r8 \
  --set training.device=cuda:1 --set projector.target_compression_ratio=8
```

压缩入口支持目标 `N/M=4/8/16/32/64`。4/8/16/32 保持原预算；64× 在当前 BraTS 网格 `[39,15,15]`、物理范围 `[156,240,240]` mm 下，AvgPool/ConvPool/Mixer/LocalPacker 使用 `[4,6,6]` 输出网格（144 token，60.938×），Concat 使用 `[6,3,4]` 固定块（140 token，62.679×），BindFormer/Query/Merge/Selection 输出 `ceil(8775/64)=138`（63.587×）。BindFormer 默认分配为 `34+26×4`。实际 M 不强制相同。

可直接把压缩方法脚本顶部改为 `projector_ratios=(64)` 或 `(32 64)`，也可传 `--ratios 64`；其余训练参数和原默认倍率列表不变，MLP 仍只支持 1×。自动预检当前支持 NeuroVFM 的 `native_grid + minimal_pad`、完整 patch 网格；在加载模型前检查选定 MRI 文件头，运行时再次核对几何与输出。64× 网格预设和 Concat 预设对其他输入几何明确报错。旧的显式 `projector.params.num_tokens/group_size` 模式仍可用，但不能与 ratio 同时指定。原四档表格和恢复规则见 [baseline 文档第15节](Pipeline_0929/PROJECTOR_ROUTES_AND_BASELINES.md)。

```bash
python -m src.cli.config --set projector.name=concat3d \
  --set projector.target_compression_ratio=16 --resolve-budget \
  --budget-report runs/budget_preview/concat_r16.json
```

替换预处理：实现返回 `PatchVolume` 的 callable，在 preprocessing registry 注册；额外参数放入 `preprocessing.params`，在实现中将影响数值的所有参数纳入版本哈希。替换 Encoder：实现 `EncoderAdapter` 的完整体积输入、特征/坐标/元数据列表输出并注册，同时实现训练阶段、缓存 fingerprint 和 trainable 参数契约。3DINO 尚未接入训练适配器。

## 只改一个参数切换网格

```bash
projector-preprocess --set preprocessing.spacing_policy=native_grid --representative --qa --output runs/qa_native
projector-preprocess --set preprocessing.spacing_policy=2mm --representative --qa --output runs/qa_2mm
projector-preprocess --set preprocessing.spacing_policy=4mm --representative --qa --output runs/qa_4mm
```

`native_grid` 不进行物理重采样；RPI 轴置换/翻转后只做 patch 对齐 padding。`2mm` 和 `4mm` 分别表示层向 2/4 mm、平面内 1 mm，均保持 minimal padding。任意 spacing 使用 `spacing_policy=resample` 配合 `inplane_mm` 和 `throughplane_mm`。原官方 4 mm + 裁剪流程单独保留在 `configs/preprocessing/official_4mm.yaml`。

Mask 只参与独立 QA，不进入训练输入、prompt 或 loss。native-grid 全 train/val 审计覆盖 2,476 个序列 header、805 个独立检查、39,903 个逐标签六邻域连通分量：丢失分量和标签体素均为 0。`Singapore_63_FLAIR` 的 mask/image 方向 header 有微小差异，映射到影像参考网格后所有标签体素逐一相同，记录在审计中。此结论不把连通分量等同于独立病灶，也不覆盖未参与 QA 的 test。

证据：`runs/archive/engineering/implementation/native_grid_audit/summary.json` 和 `records.jsonl`。此前五个代表检查的 109 个分量中，4 mm 消失 31、2 mm 消失 7、native 消失 0；该比例仅是抽样结果。

## 连续 finetuning 主线（2026-10-12）

使用 `configs/training/joint_lora.yaml`，一次连续训练最多 10 epochs：先仅训练 projector，随后联合训练 projector + 视觉 LoRA；两段 linear warm-up 默认各占计划 optimizer updates 的 6.25%，剩余为联合 cosine。两组峰值 LR 都是 `1e-5`。阶段切换保留 projector 权重与 Adam 状态，全程冻结视觉 base 和 LLM。LoRA 默认 rank16/alpha32/dropout0.05，注入全部 12 层 fused qkv 与 attention output proj。

```bash
# 只检查参数传递与步数，不加载 GPU 模型。
bash scripts/run_full_mlp_uncompressed.sh brats_joint_r16 --dry-run

# 正式运行（使用全新目录）。
bash scripts/run_full_mlp_uncompressed.sh brats_joint_r16 --datasets brats \
  --set training.device=cuda:0 \
  --set training.max_epochs=10 \
  --set training.projector_warmup_steps=77 \
  --set training.joint_warmup_steps=77 \
  --set training.lora.rank=16 --set training.lora.alpha=32
```

当前 BraTS 1,960 条训练序列、有效 batch16 对应计划 1230 updates，默认两段 77/77、cosine 1076。显式 `*_warmup_steps` 优先于对应 ratio；step 是 optimizer update。所有参数均可由 `--set` 覆盖，修改 rank 时 alpha 独立设置。warm-up 完成后首次完整 epoch 验证建立早停基准；CE 相对参考值降低超过 0.005 才清零计数，连续三次无足够改善则停止。全程最低实际验证 CE 独立选 `best.pt`。默认最早 epoch5 末停止；NaN/Inf 立即报错。

最终视觉特征缓存关闭；确定性的预处理 patch 缓存继续复用。`schedule.json` 保存实际步数与来源，训练日志记录本步两组 LR/phase，`training_state.json` 保存早停状态/停止原因；两个阶段边界自动存 checkpoint。新 checkpoint v2 保存暂时冻结的完整 LoRA 因子，旧 v1 仍可读取。续训使用相同配置及同一目录，并传 `training.resume=.../last.pt`；不能通过 resume 改变总预算、rank 或调度参数。

新入口与参数示例见 [脚本说明](scripts/README.md)，完整规格及 smoke 证据见 [训练文档](Pipeline_0929/NEUROVFM_MLP_TRAINING_PIPELINE.md)。

## 旧独立 A/B 训练配置（保留用于复现）

| 项目 | A | B-LoRA | B-Full |
|---|---|---|---|
| 视觉 Encoder | 冻结 | rank 8，alpha 16，dropout 0.05 | 全参数训练 |
| 视觉学习率 | 0 | 2e-5 | 3e-6 |
| MLP 学习率 | **1e-5** | 5e-5 | 5e-5 |
| 最多 epoch / patience | 10 / 3 | 5 / 3 | 5 / 3 |
| Encoder 最终特征缓存 | 允许 | 禁止 | 禁止 |
| 初始化 | 本地预训练视觉/LLM + 随机 MLP | 同一个 A 最优 checkpoint | 同一个 A 最优 checkpoint |

共用配置：MLP `768→4096→4096`、GELU、19,931,136 个参数；Llama BF16、无量化、冻结，gradient checkpointing；可训练参数和 AdamW 状态 FP32；AdamW `(0.9,0.999)`、eps `1e-8`、weight decay `0.01`（bias/一维参数不衰减）；cosine、A 的 warmup 固定4个 optimizer steps、B warmup 总步骤的5%、clip 1；每卡 microbatch 1、有效 batch 16。所有答案 token 和结束符参与 CE，按整个累积窗口的有效答案 token 总数归一化。

下表和命令描述旧独立 A/B 模式。一键脚本已切换为上文连续 joint_lora 主线；需要复现旧实验时显式使用以下 stage preset。

```bash
projector-train --config configs/training/stage_a.yaml --config configs/training/deterministic.yaml --set training.run_dir=runs/native_a
projector-train --config configs/training/stage_b_lora.yaml --config configs/training/deterministic.yaml --set training.init_checkpoint=runs/native_a/checkpoints/best.pt --set training.run_dir=runs/native_b_lora
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True projector-train --config configs/training/stage_b_full.yaml --config configs/training/deterministic.yaml --set training.init_checkpoint=runs/native_a/checkpoints/best.pt --set training.run_dir=runs/native_b_full
```

16 步、8 个 ADC 样本的 `1e-4` 无有效 warmup smoke 出现首次更新 CE 上冲，完整记录在 `runs/archive/training_and_model_diagnostics/native_overfit_a/`。用户批准的 `1e-5`、4 warmup steps 诊断更稳，已更新阶段 A 配置用于后续正式训练；该诊断仍不构成收敛证明。

经用户确认的诊断分支 `runs/archive/training_and_model_diagnostics/native_overfit_a_lr1e5/` 使用同样 8 个训练样本/2 个验证样本、16 次更新，学习率 `1e-5`、4 步 warmup：训练 CE 从 2.445 降到 1.829，最终验证 CE 2.157。曲线见 `runs/archive/engineering/implementation/learning_curves.png`。这验证了可学习的梯度路径，不等于充分记忆或临床内容正确。

实际第 8 步恢复到第 16 步的 BF16 对照：第一步恢复前向 CE 完全一致、scheduler/RNG 恢复一致；后续 CE 最大绝对差 0.00345、MLP 参数最大差约 2.5e-6。独立 SDPA 测试定位到 FlashAttention 反向非确定性；现有快速模式不承诺逐位续训复现。详细记录 `runs/archive/engineering/implementation/resume_comparison.json` 和 `sdpa_backward_repeatability.json`。

已增加 `--set training.deterministic=true`（也可追加 `--config configs/training/deterministic.yaml`）。阶段 A 真实续训已逐位复现；LoRA/Full 重复梯度、长样本更新、Full 有效 batch 16 均通过。Full 启用确定性计算时，运行前设置 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`；9,077-token 长样本累积峰值为分配22.08 GiB/预留22.39 GiB。确定性模式比已测快速 Full 窗口约慢14%（样本目标文本长度略有不同）。当前建议正式A/LoRA/Full统一确定性计算，Full正式训练用可扩展分配器并实测目标数据、机器上的显存。详见 [工程验收记录](Pipeline_0929/IMPLEMENTATION_ACCEPTANCE.md)。不能在同一 run 恢复时改变确定性设置。

恢复使用相同配置加 `--set training.resume=.../last.pt`；不能同时设置 `init_checkpoint`。保存/恢复训练参数、optimizer、scheduler、每 rank RNG、epoch/window/step、早停状态；改变训练集选择、优化设置、数据/模型/预处理/prompt 契约会拒绝恢复。`save_every_steps>0` 额外保存编号 checkpoint。`best.pt` 按完整验证集答案 CE 选择；每次验证另在固定 cohort/sequence 代表子集生成报告，默认 14 条，输出 BLEU/ROUGE 和空输出/超长标志，不影响 CE 选优。工程测试可显式设置 `training.validation_generation_samples=0`。

Stage B 两种训练方式应使用相同数据选择和同一个 A checkpoint。改变网格后应建立独立 A→B 实验链；缓存通过版本哈希自动隔离，不能把 native 的 A checkpoint 当作 2 mm 的 A 结果。

多卡训练使用 DDP，每个进程都会在自己的 GPU 上放完整模型（显存不会跨卡合并）。入口脚本 `scripts/launch_ddp.sh` 按物理卡号设置 `CUDA_VISIBLE_DEVICES`，并用 `torchrun` 启动；配置模板 `configs/training/ddp.yaml` 设置 DDP、每卡 microbatch=1、全局有效 batch=16。

例如在物理 GPU 1 和 3 上启动新的两卡 Stage A：

```bash
./scripts/launch_ddp.sh 1,3 \
  --config configs/training/stage_a.yaml \
  --config configs/training/deterministic.yaml \
  --config configs/training/ddp.yaml \
  --set training.run_dir=runs/new_stage_a_ddp
```

在这个例子里 local rank 0 映射到物理 GPU 1，rank 1 映射到物理 GPU 3；全局 batch 16 会分到每卡8个样本的梯度累积窗口。必须保证 `effective_batch_size` 可被 `microbatch_per_device × GPU 数` 整除；3卡时可将有效 batch 设为18或24。每张卡都要有足够空间容纳完整模型和单个最长体积的激活。脚本会设置 expandable allocator 和确定性 cuBLAS 工作区。

训练数据顺序在 ranks 间共享、样本分片；各 rank 使用不同但可复现的 dropout RNG。loss 按全局答案 token 数归一化，验证结果跨 rank 汇总，checkpoint 保存每个 rank 的 RNG。断点续训要求 world size 与原 checkpoint 一致；改变卡数应另开 run，Stage B 从 Stage A 的 best 权重初始化则可以使用另一 world size。

当前通过了 CPU 双进程 DDP token 归一化测试；完整 8B 多 GPU 训练尚未做 GPU smoke，因此启动正式多卡实验前应先在目标卡上完成短步 smoke 并测峰值显存。GPU1/3 两卡理论上能减少单步样本串行计算，但实际提速需要 benchmark；当某张卡可用显存低于单卡实测峰值时，不应启动。当前正式 full-data run 仍按用户选择在 GPU3 单卡继续。

## 生成与评价

**从 checkpoint 一键评价（自动恢复 A／B-LoRA／B-Full 配置）：**

```bash
bash scripts/evaluate_checkpoint.sh \
  --checkpoint runs/full_native_20261006/stage_b_lora_full/checkpoints/step_00000068.pt \
  --split test --device cuda:2 --metrics all \
  --output-dir runs/eval_b_lora_step68_test
```

`all` 包含 BLEU-4、ROUGE-L、GREEN、RadGraph-XL 三种 F1、RaTEScore、Answer CE；不包含尚未实现的 BERTScore 或 NeuroQA。可以改为 `--metrics bleu4 rougeL green` 或 `--metrics green radgraph rate`。Bash 默认调用本机 `projector` Python，无需先激活；其他机器用 `PROJECTOR_PYTHON` 指定解释器。生成进程退出后才启动评分进程，自动释放 VLM 显存，并在 `<output-dir>/metrics.json` 汇总选定指标。

输出目录必须是新目录；加 `--dry-run` 可先查看识别出的阶段、样本数和命令，不运行模型；小样本检查用 `--split val --limit 2`，完整 test 则不加 `--limit`／`--ids-file`。详细入口为 [evaluate_checkpoint.py](src/cli/evaluate_checkpoint.py)，逐例预测和完整统计位于输出目录的 `generation/`、`scoring/`。

默认 `generation.batch_mode=independent`，每个完整体积独立解码，多条输入仍可用列表接口；真实 8B 的 SDPA 混合长度左 padding 曾产生 NaN，生产路径不再调用该方式。NaN logits 直接报错，不允许 argmax 掩盖错误。未来批量解码需另实现并验收后开放配置。

```bash
projector-evaluate --config configs/training/stage_b_lora.yaml --set evaluation.checkpoint=runs/native_b_lora/checkpoints/best.pt --set evaluation.output_dir=runs/native_b_lora/val
projector-evaluate --config configs/training/stage_b_lora.yaml --set evaluation.checkpoint=runs/native_b_lora/checkpoints/best.pt --set evaluation.output_dir=runs/native_b_lora/val_zero --control zero
projector-evaluate --config configs/training/stage_b_lora.yaml --set evaluation.checkpoint=runs/native_b_lora/checkpoints/best.pt --set evaluation.output_dir=runs/native_b_lora/val_shuffle --control shuffle
```

Greedy、beam 1、最多 192 个新 token；独立记录 EOS、达到长度上限、空文本。`--ids-file`/`--limit` 为明确标注的子集评价，不冒充全 split。覆盖检查拒绝缺失/重复 ID；空预测保留日志和指标但 CLI 返回失败。分层含 cohort/source/sequence、病例和患者 ROUGE-L 宏平均；患者级 bootstrap CI 和配对差值；BLEU4 使用 sacrebleu 13a，ROUGE-L 使用 Porter stemmer，保存版本与 signature。生成入口默认只计算这些基础指标；GREEN、RadGraph-XL、RaTEScore 通过下面的离线入口补算，BERTScore 尚未实现。`zero` 置零投影后的视觉 embedding；`shuffle` 在相同序列内错配完整体积，记录 donor ID，不跨序列交换。

新增临床指标直接读取已生成的报告，无需重新运行 VLM：

```bash
PYTHONUTF8=1 PYTHONIOENCODING=utf-8 python -m src.cli.score_reports \
  --predictions runs/native_b_lora/val/predictions.jsonl \
  --output-dir runs/native_b_lora/val/scored \
  --metrics green radgraph rate --device cuda:2 --repeats 1000
```

使用 `projector` 环境；评分专用依赖只从项目内 `third_party/evaluation/python_deps/` 加载，Torch 2.5.0／Transformers 4.51.3 不变。正式评价还应传入独立的 `--expected-ids` JSON 数组校验完整划分；省略时仅保证输入文件内部覆盖。一次性成对比较 CLI 已归档在 `archive/non_pipeline/src/cli/compare.py`，普通训练和评测不依赖它。下载与依赖准备只在重建环境时需要；已准备好的本机运行不需要重新下载，历史脚本位于 `archive/non_pipeline/scripts/`。评测协议见 [评价指标与 NeuroQA 文档](Pipeline_0929/EVALUATION_METRICS_AND_NEUROQA_PLAN.md)。

`--control text` 完全移除视觉 token，仅保留同一文本 prompt，是独立的纯文本基线；`zero` 保留视觉位置，因此两者不是同一个条件。固定 2 个验证样本的四路实测位于 `runs/archive/training_and_model_diagnostics/native_learning_controls/`：无空输出/超长输出，换图和去图均改变生成；正常图像的答案 CE 低于错配/零视觉，但仅两例不能证明可靠视觉理解。另保存未填写的 `manual_review.json`，不把关键词或几何 QA 当成人工内容审阅。实际 CLI 子集评价证据：`runs/archive/training_and_model_diagnostics/native_evaluate_cli/`。

## 验收与资源记录

```bash
OMP_NUM_THREADS=4 pytest -q
```

日常 pipeline 不需要运行旧版 smoke/audit/download 脚本；它们和原输出已移到
`archive/non_pipeline/` 或 `runs/archive/`。核心启动文件见 [scripts/README.md](scripts/README.md)，归档清单见
[archive/non_pipeline/README.md](archive/non_pipeline/README.md)。

真实本地 NeuroVFM + 未量化 Llama 8B，GPU3，native 8,775 token，单步峰值分配：A 20.51 GiB，B-LoRA 20.97 GiB，B-Full 21.63 GiB；B-Full 累积 16 个完整体积时为 22.03 GiB，预留 22.97 GiB。全参数分支余量较小，较长答案、其他进程占卡、不同 attention/microbatch 都需重新测量；这些数值不是仅 Encoder 的显存。

原始 smoke 位于 `runs/archive/training_and_model_diagnostics/native_smoke_{a,b_lora,b_full}/smoke.json`；A 的历史记录保留当时批量生成失败，修复后的三个分支解码复验位于 `runs/archive/engineering/implementation/native_generation_final.json`。训练梯度、参数更新、冻结权重逐位不变、checkpoint 重载已实测。几步更新的 checkpoint 只用于工程检查，其空文本/重复文本不作为有效报告生成结果。最终状态和尚未通过的学习行为检查见 `runs/archive/engineering/implementation/status.json`。
