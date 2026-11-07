AnchorRead3D A1–A4：独立消融实验
================================

范围：四个变体 × RG/VQA × 32×/256×，共 16 个独立训练。
每个 shell 默认顺序完成本任务的 r32、r256：训练 → 验证集 best → test。
这些脚本不会启动其他任务，也不会修改旧 baseline 的配置或 Excel。

结构定义
--------
A1 anchorread3d_a1_no_localread
   G + mean(X_r)。继承原 ADirect，不构造 LocalRead。
   global 分支、G/L 分配、局部分区、LN/MLP head 与 A0 相同。
A2 anchorread3d_a2_no_mean_bypass
   G + alpha*LocalRead(mean(X_r), X_r)。只删局部输出中的均值加法项。
   mean query、原区域 K/V、alpha 初始化 0.1、其他模块均保留。
A3 anchorread3d_a3_global_only
   M 个 global queries，使用 A0 的 InputAdapter、单层 QueryBlock 和 head。
   G=M、L=0，不构造局部模块；不是用六层 Perceiver 替代。
A4 anchorread3d_a4_local_only
   M 个 mean(X_r)+alpha*LocalRead(mean(X_r),X_r)，G=0。
   K=ceil(M/4) 个父区域，前 K-1 个各分四个子区域。
   最后一个父区域直接分成剩余的 1–4 个子区域，全部 patch 恰好覆盖一次。
   例如 M=35 为 8×4+3；不是先分成 36 个再丢掉一个。
   父区域只负责空间分区，不输出额外 token，也不做父区域均值融合。
   LocalRead 与 A0 相同；没有 global adapter/PE/sequence 模块。

所有局部均值都取自分到该 cell 的 encoder patch features。
LocalRead 是单个 mean query 对该 cell patch 的 cross-attention。
A1/A2 固定 A0 的 G、L、分区；A3/A4 固定总 M，是预算分配对照。
A4 的空间粒度随 L 增加而变化，不能解释为完全固定空间粒度的删除 global。

实际预算（沿用各任务现有预算协议）
----------------------------------
任务  名义压缩  N     M    A1/A2 G,L   A3 G,L  A4 G,L   A4 父子分配
RG    32        8775  275  67,208      275,0   0,275    68×4+3
RG    256       8775  35   7,28        35,0    0,35     8×4+3
VQA   32        9555  288  72,216      288,0   0,288    72×4
VQA   256       9555  36   8,28        36,0    0,36     9×4
实际 N/M、G/M、allocation_version、children_per_parent 写入保存配置与预算记录。
A1/A2 的 global_fraction 固定 0.25，A3 固定 1，A4 固定 0；误覆盖会报错。

统一训练配方
------------
configs/experiments/anchorread3d_ablation/training.yaml
以 scripts/Exp1026/run_baseline_adirect_localread_r16.sh 为参考：
seed 42，joint_lora，冻结 LLM，encoder LoRA rank16/alpha32/dropout0.05；
projector/encoder LR 都为 2e-4；BF16；microbatch 1、effective batch 16；
20 epochs；projector warmup 77、hold 0、joint warmup 77；
joint warmup projector hold_peak；cosine_min_lr_ratio 1.0；
early stopping patience 3、min_delta 0.005；其余优化参数也保持一致。
RG 与 VQA 共用训练配方，不共用数据/预处理/loss/验证集选择协议。
VQA 保留 mixed_v2、include_unreviewed、exact_balanced_repeat 和 QA task。
VQA 的 unreviewed_training/evaluation 标签与 QC 选择记录如实保留。

注意：旧 VQA 主实验脚本的 warmup=150/150、min_delta=0.002，
与用户此次指定的 r16 配方不同。旧脚本未修改。
比较 A0 与这些消融前，要核对 A0 的保存配置；不一致时应以同配方补跑 A0。
不能直接把旧 ADirect 的分数当作同配方 A1，也不能仅凭方法名认定可复用。

启动
----
从任意工作目录调用均可；用空闲的物理 GPU 编号替换示例中的 0。
每个 shell 只使用一张 GPU；两个压缩率顺序运行，各自从头训练。

  cd /mnt/sda/jyzhang/VLM_Projector
  CUDA_VISIBLE_DEVICES=0 bash scripts/ablation/rg_a1_no_localread.sh
  CUDA_VISIBLE_DEVICES=0 bash scripts/ablation/rg_a2_no_mean_bypass.sh
  CUDA_VISIBLE_DEVICES=0 bash scripts/ablation/rg_a3_global_only.sh
  CUDA_VISIBLE_DEVICES=0 bash scripts/ablation/rg_a4_local_only.sh

  CUDA_VISIBLE_DEVICES=0 bash scripts/ablation/vqa_a1_no_localread.sh
  CUDA_VISIBLE_DEVICES=0 bash scripts/ablation/vqa_a2_no_mean_bypass.sh
  CUDA_VISIBLE_DEVICES=0 bash scripts/ablation/vqa_a3_global_only.sh
  CUDA_VISIBLE_DEVICES=0 bash scripts/ablation/vqa_a4_local_only.sh

仅跑一个压缩率：在命令后加 --ratios 32 或 --ratios 256。
自定名称：加 --run-tag my_ablation_a2；默认名称含变体、seed、UTC 时间戳。
仅打印配置/命令：加 --dry-run，不分配 GPU，不创建正式 run。
VQA 只读数据检查：加 --check-data，不加载模型或分配 GPU。
不要在忙碌的 GPU 上并发启动这些训练。

Python 默认为 /mnt/sda/mingjie/conda_envs/projector/bin/python；
可用 PROJECTOR_PYTHON 指定同等依赖环境。
RG 自动读取 configs/local_paths.yaml；可用 PROJECTOR_LOCAL_CONFIG 指定替代路径。
VQA 不加载 RG local_paths，可用 VQA_LOCAL_CONFIG 指定 VQA 专用路径配置。
可透传旧 launcher 的 --set 等参数；改训练配方后的实验应使用独立 run-tag。

输出与恢复
----------
默认根目录：
  runs/ablations/anchorread3d/rg/<run-tag>/r{32,256}/
    training_config.yaml
    joint_lora/checkpoints/{best,last}.pt
    evaluation/best_test/metric.json、generation/predictions.jsonl
  runs/ablations/anchorread3d/vqa/<target-version>/v2/<scope>/mixed/
    <projector>/r{32,256}/s42/<run-tag>/
    launch.json、launch_config.yaml、data_preflight.json
    train/checkpoints/{best,last}.pt
  results/ablations/anchorread3d/vqa_raw/<相同任务层级>/test/normal/
    metrics.json、predictions.jsonl、config.yaml、provenance.json

ANCHORREAD_ABLATION_RUNS_ROOT 可覆盖 runs 根；
ANCHORREAD_ABLATION_RESULTS_ROOT 可覆盖 VQA results 根。
RG 的 metric 保持原 pipeline 设计，存于其专属 run/evaluation 内。
已有同名输出会被原 launcher 拒绝，不能以新 run 覆盖旧结果。

恢复时使用相同变体的 shell：
RG：--resume-run 指向含 sweep_plan.json 的目录（两个 ratio 的父目录）。
    原 RG 恢复逻辑复用已完成训练的 best checkpoint，再继续评估/后续 ratio。
    它不是任意中断位置的 last.pt 训练恢复器；未完成训练的 best 会触发拒绝。
VQA：--resume-run 指向单个 ratio 的 launch.json 所在目录；从 last.pt 恢复。
     只评估已有 best 可再加 --evaluate-only；显式评估重试可加 --eval-tag retry1。

结果汇总（仅扫描本消融 runs 根；不会修改主实验 Excel）
------------------------------------------------------
  /mnt/sda/mingjie/conda_envs/projector/bin/python scripts/ablation/collect_results.py

写入 results/ablations/anchorread3d/summaries/<新时间戳>/：
manifest.json（含来源和完整保存配置）、results.csv、results.xlsx。
不同 run-tag 保留为不同行，不自动挑最佳结果；pending 不填分数。
保留 VQA scope/coverage 与实际样本数，不把 unreviewed/subset 改称 full_split。
RG 的临床指标 partial 状态也保留。

验证
----
可重跑全部八个 shell dry-run 并存档（output-dir 必须是新目录）：
  /mnt/sda/mingjie/conda_envs/projector/bin/python scripts/ablation/preflight.py \
    --output-dir experiments/anchorread3d_ablation_preflight_new --check-vqa-data

本次验证存于 experiments/anchorread3d_ablation_preflight_20261108/。
tests/test_anchorread3d_ablation.py 覆盖：结构差异、A1 数值等价、A2 mean query、
A3 原 global block 等价、A4 无丢弃/重叠/跨 cell 读取、16 组预算与 checkpoint。
tests/test_anchorread3d_training.py 覆盖：
  1. 实际 N/M 和 768→4096 宽度的 FP32 CPU forward/backward/optimizer step；
  2. 小型 encoder/LLM 的 BF16 RG/VQA loss、冻结 LLM、LoRA 阶段切换、生成和 checkpoint 恢复。
所有预算的确定性 BF16 attention/backward/checkpointing 另由结构测试覆盖。
本机 CPU 无 BF16 原生加速；完整宽度 BF16 检查通过 7 组后为耗时原因主动结束，
未出现失败。随后全 16 组完整宽度检查使用 CPU FP32；正式训练仍使用 BF16。
小型 backbone 集成测试不等于真实 NeuroVFM + Llama 8B 的 CUDA 端到端短训。
本次未启动正式训练，也未操作用户已有 tmux 任务；GPU 显存/吞吐需空闲卡上实测。
