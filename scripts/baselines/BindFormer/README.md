# BindFormer 1019：恒定 LR 与末端 SA/CA 对照

| 入口 | within_slot_sa | local_reads_global | 默认 LR / 压缩率 |
| --- | --- | --- | --- |
| `run_bindformer_vanilla_constant_lr_1019.sh` | false | false | 两组 2e-4；4/8/16/32× |
| `run_bindformer_sa_ca_constant_lr_1019.sh` | true | true | 两组 2e-4；4/8/16/32× |

两组使用同一份 `configs/training/constant_lr.yaml`、同一训练器和
`scripts/ft_ablation/_constant_lr.bash` 执行器，与新 AvgPool 恒定 LR 实验一致。
这里使用支持 BindFormer 的通用 `src.cli.baseline_sweep`，包含完整 test 的计划与结果检查。

## 实验配置

- 每个 shell 顶部直接编辑 GPU、压缩率、Projector/LoRA LR、epochs、seed、warm-up、早停及结构开关。
- 默认 20 epochs / 2460 optimizer updates；每 epoch 123 updates，包含前置阶段。
- 前 77 步仅 projector 预热；随后 77 步 LoRA 预热，projector 保持峰值；其余步数两组 LR 恒定 2e-4。
- `cosine_min_lr_ratio=1.0`；日志沿用 `joint_cosine` 阶段名，实际曲线不衰减。
- 额外 projector hold 为 0。保留 AdamW 状态、有效 batch 16、LoRA rank=16/alpha=32、BF16、冻结 LLM。
- 完整 validation CE 选择 `best.pt`；warm-up 后开始 early stopping，patience=3、min_delta=0.005。
  任意更低 validation CE 都保存 best；早停的有效改善阈值与 best 保存分开。
- 每档独立从相同预训练资产初始化；新实验不恢复其他 checkpoint。
- 达到预算或早停后，自动评测该档 best.pt 的完整 test（当前 560 条）、全部指标，再运行下一档。
  不评测 epoch 10 或 last.pt。训练或评测失败则停止。
- GREEN 个别格式失败沿用现有 valid-only 均值与覆盖率记录，不作为零分。
- 优化器诊断、训练/验证 CE、日志、模型和评测结果写入各自 run。

两组默认只改变同槽角色 SA 与门控读取全局 CA：各 1 层，CA 门控初值 0.1。
两组都保留 UMLS name_definition、brats_train 范围、四角色各 1 query、global_fraction=0.25、
slot_iterations=3；geometry 与 feature/text/tuple/binding/concept 辅助损失均关闭。
SA/CA 不改变输出 token 数。按脚本顶部的 `projector_ratios` 升序执行，支持 4/8/16/32/64。
可直接填写 `(64)` 或 `(32 64)`，也可传 `--ratios 64`；其余训练参数不变。
当前 BraTS 输入下，64× 输出 138 token，其中全局 34 个、26 个 slot 各输出四个角色 token。

## 启动

在本机训练环境中执行，GPU 编号改为实际空闲卡：

```bash
conda activate projector
export PROJECTOR_PYTHON="$CONDA_PREFIX/bin/python"

# 先检查配置、数据划分和输入几何；不加载 VLM、不创建正式 run。
CUDA_VISIBLE_DEVICES=0 bash scripts/baselines/BindFormer/run_bindformer_vanilla_constant_lr_1019.sh --dry-run
CUDA_VISIBLE_DEVICES=1 bash scripts/baselines/BindFormer/run_bindformer_sa_ca_constant_lr_1019.sh --dry-run

# 正式实验由用户手动在各自 tmux 窗口执行。
CUDA_VISIBLE_DEVICES=0 bash scripts/baselines/BindFormer/run_bindformer_vanilla_constant_lr_1019.sh
CUDA_VISIBLE_DEVICES=1 bash scripts/baselines/BindFormer/run_bindformer_sa_ca_constant_lr_1019.sh
```

脚本可从任意工作目录通过绝对路径调用。外部 CUDA_VISIBLE_DEVICES 优先；内部为逻辑 cuda:0。
PROJECTOR_PYTHON 优先，其次激活的 Conda 环境；避免 tmux PATH 仍指向 base Python。
运行目录默认是 `runs/baselines/bindformer_<变体>_1019_p<LR>_l<LR>_<UTC时间戳>/`，
每个压缩率存于 `r4/`、`r8/` 等目录。可用 PROJECTOR_RUN_TAG / --run-tag 指定新名称，
用 PROJECTOR_OUTPUT_ROOT / --output-root 指定本机存储位置；同名已有 run 拒绝覆盖。
参数覆盖后以最终 `sweep_plan.json` 为准。

## 另一服务器

通过 Git 同步这两个入口、`_common_1019.bash` 和既有共享代码/配置。
各机模型、数据、预处理缓存和知识库路径放在 Git 忽略的 `configs/local_paths.yaml`，
或由 PROJECTOR_LOCAL_CONFIG 指定该机配置。
知识库可覆盖 `projector.params.concept_bank`；默认相对路径为
`assets/knowledge/bindformer_brain_candidates_v2`，包含 frozen SapBERT 向量及其清单/掩码。
Git 不传输模型、知识库资产、数据、checkpoint 或评测结果。

预检应核对 train/val/test=1960/268/560、123 updates/epoch、77/0/77/2306 阶段预算、
floor=1.0、val 早停及 best.pt/test/full_split/all；两组与 AvgPool 的数据身份摘要应一致。
dry-run 不替代真实 GPU 显存和完整模型训练验证。

## 本机实现检查（2026-10-19）

- 相关自动化检查共 96 项通过，包括 shell 参数传递、本机路径覆盖、阶段 LR、
  Adam 状态保留、early stopping 与 best 保存规则、SA/CA 前向反向，以及模拟的
  “正常结束/早停 → best.pt 完整 test → 下一压缩率”流程。
- 两种变体 × 四档压缩率均完成真实数据 dry-run；train/val/test 为 1960/268/560。
  数据身份、训练配置与阶段预算均与 AvgPool 对照一致；两组结构配置只差 SA/CA 开关。
- CPU 上构建了八个实际宽度 projector，核对知识库清单、120 个有效概念、模块开关和输出 token 数。
  8× projector 参数量为 Vanilla 50.454528M、SA+CA 55.184641M（不含视觉 LoRA）。
- 小尺寸模型的实际梯度检查通过；没有执行完整 VLM 的 GPU 训练或显存测试，也没有启动正式实验。

本机检查产物：`results/BindFormer_1019_preflight_AMuXlq/`，其中 `verification.json`
记录配置和模型构建核对，`tests.xml` / `tests.log` 保存测试结果，三个 `*_plan.json`
保存两组 BindFormer 与 AvgPool 的实际 dry-run 计划。此目录属于本机实验产物，不随 Git 同步。
