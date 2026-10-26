# Exp1024：cosine 下限 0.3，seed 42

四个脚本共 **12 个不重复的实验**，覆盖六种方法各自的32×和64×。每个脚本串行完成3次独立训练和完整test评测；一个实验训练和评测结束后才开始下一个。四个脚本可分别在四个terminal/tmux pane并行启动。

| 脚本 | 默认物理GPU | 串行顺序 |
|---|---:|---|
| `run_1024_1.sh` | 0 | Baseline A 32× → Baseline A 64× → R2 64× |
| `run_1024_2.sh` | 1 | Baseline A＋role value adapter 32× → 64× → E3 64× |
| `run_1024_3.sh` | 2 | R1 32× → R1 64× → R2 32× |
| `run_1024_4.sh` | 3 | E2 32× → E2 64× → E3 32× |

R1/R2/E2/E3全部以Baseline A的fixed-region mean架构为基础。role value adapter沿用此前rank=16、alpha=16设置。

## 启动

分别在四个terminal/tmux pane运行下面四条命令。不要在同一个terminal顺序粘贴四条，除非希望四个队列也串行运行。

```bash
bash /mnt/sda/jyzhang/VLM_Projector/scripts/ft_ablation/Exp1024/run_1024_1.sh
```

```bash
bash /mnt/sda/jyzhang/VLM_Projector/scripts/ft_ablation/Exp1024/run_1024_2.sh
```

```bash
bash /mnt/sda/jyzhang/VLM_Projector/scripts/ft_ablation/Exp1024/run_1024_3.sh
```

```bash
bash /mnt/sda/jyzhang/VLM_Projector/scripts/ft_ablation/Exp1024/run_1024_4.sh
```

默认使用此前实际训练的Python：`/mnt/sda/mingjie/conda_envs/projector/bin/python`。可以通过`PROJECTOR_PYTHON`显式替换；不依赖tmux恰好激活了哪个conda环境。外部已设置的`CUDA_VISIBLE_DEVICES`优先于默认GPU。

## 配置与对照范围

仅改变 `training.cosine_min_lr_ratio: 1.0 → 0.3`。保留全局 `seed=42`。原脚本变量名`projector_seed`实际控制全局seed，并不是仅控制projector初始化。

| 参数 | 本轮值 |
|---|---|
| projector / encoder LoRA峰值LR | 2e-4 / 2e-4 |
| projector warm-up / hold / joint warm-up | 77 / 0 / 77 steps |
| joint warm-up的projector策略 | hold_peak |
| cosine_min_lr_ratio | 0.3 |
| 完整调度末端LR | 6e-5 |
| 最大epochs / 每epoch更新数 | 20 / 123 |
| batch / microbatch | 16 / 1 |
| encoder LoRA rank / alpha / dropout | 16 / 32 / 0.05 |
| early stop | validation CE；patience=3，min_delta=0.005 |
| 初始化 | 原预训练模型＋全新projector/LoRA/optimizer；不读取此前finetune结果 |
| test checkpoint | 原最低validation CE的best.pt |
| test范围 | 全部560条、140位患者；metrics=all |

0.3是原20-epoch调度的末端下限；保留早停可能导致训练在达到这个下限之前结束。本轮固定seed，主要研究scheduler影响；如需研究seed差异，另用不同seed启动同一组方法作为对照。

各方法继续使用原有损失和数据：

- Baseline A和role value adapter：原报告CE。
- R1：报告CE＋0.1 span loss；R2另加0.1 group loss，margin=0.2。
- E2：原Decoder短问答配对，pair_weight=0.1；E3另加fact_weight=0.1。positive_ce_weight=0.25，margin=0.2。
- 辅助warm-up均为77步；R/E分别保留`report_output_v1`、`report_facts_v1`原数据和原抽取规则。
- 辅助字段保持`edema,laterality,site,signal`；enhancement/margin关闭；morphology/extent仍无独立属性监督。
- R1/R2保留完整validation事实评估和额外best_facts保存；正式test仍只评best-CE。

此次不合并上一轮诊断提出的标签/评估规则修复，以免将scheduler实验与监督数据变化混在一起。

## 开关与输出

每个脚本顶部可修改`projector_seed`、`cosine_min_lr_ratio`、ratio数组。也可以用环境变量：

```bash
PROJECTOR_SEED=43 COSINE_MIN_LR_RATIO=0.3 CUDA_VISIBLE_DEVICES=0 \
  bash /mnt/sda/jyzhang/VLM_Projector/scripts/ft_ablation/Exp1024/run_1024_1.sh
```

默认输出到`runs/Exp1024/`。每次自动生成包含队列名、seed、cosine下限、GPU、时间戳、方法和倍率的独立目录名。完整控制台日志在`runs/Exp1024/queue_logs/`。`PROJECTOR_OUTPUT_ROOT`可改输出根目录，`PROJECTOR_RUN_TAG`可指定队列名称前缀，各方法/倍率仍有独立后缀。

`--dry-run`只解析配置、检查manifest/预算/辅助数据和训练评测命令，不加载模型、不创建正式run。默认配置已验证；例如：

```bash
bash /mnt/sda/jyzhang/VLM_Projector/scripts/ft_ablation/Exp1024/run_1024_1.sh --dry-run
```

其他调试参数仍可通过原`--set KEY=VALUE`传入；seed和cosine下限以脚本变量/上述环境变量为准。队列入口不接受`--ratios/--resume-run/--run-tag/--output-root/--projector/--device/--scoring-device`覆盖，防止把同一任务重复应用到多个队列项。ratio在脚本数组中修改；需要恢复某个具体任务时使用对应原标准入口及保存的sweep plan。

默认模式中的任一训练/评测进程失败，错误码会穿过日志tee返回并停止当前脚本，避免继续运行未确认的后续项。

## 已执行检查

- 四个shell及公共helper的`bash -n`通过。
- 四个完整真实dry-run通过，解析出12个唯一任务，每脚本3个。
- 12份resolved config逐项与各自历史实验核对，差异仅为cosine下限与输出目录。
- manifest、辅助sidecar、275/138 token预算、fresh初始化、best-CE full test命令检查通过。
- shell串联模拟验证：从非项目目录启动、四张默认GPU映射、方法切换和失败立即停止。模拟不加载模型。
- 没有修改旧入口或主干；记录的178个既有代码/配置/sidecar文件哈希不变。

核验记录：`results/Exp1024_launch_validation_20261024/configuration_audit.json`、`shell_execution_audit.json`及各queue的dry-run计划。本次准备工作没有启动正式训练，也没有重新做GPU前向/反向smoke；训练/评测仍调用此前已经跑通的原入口。
