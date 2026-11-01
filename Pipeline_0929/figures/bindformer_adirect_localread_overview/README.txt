ADirect + LocalRead：结构设计图（尚未训练或验证效果）

2026-11-01 后续实现：已注册 bindformer_adirect_localread，默认实现与本图一致。
训练入口：scripts/Exp1026/run_baseline_adirect_localread.sh。
同轮还实现了 ADirect_Detail 和 ADirect global_fraction=0；详见 scripts/Exp1026/README.md。
下文及 proposal_spec.json 保留初次绘图时的设计记录；不表示已有训练效果。

输出
  bindformer_adirect_localread_overview_projector.pdf  与 A4/ADirect 同风格、同尺寸的矢量 overview
  bindformer_adirect_localread_operator_detail.pdf    单次局部残差读取的算子细节
  同名 SVG / PNG / preview.png                        可编辑图 / 高清图 / 预览
  proposal_spec.json                                  明确候选定义、预算及参数估算
  caption_en.tex                                     英文图注
  design_review.html                                 机制解释、baseline 启发、候选实验优先级
  render_overview.py                                 重绘源码

初次绘图时只输出设计说明和可视化；后续注册实现见上方说明。尚未启动正式训练。
图中的新增分支是 attention-only，不是直接调用当前 QueryBlock；后者还会额外加入 FFN。

准确的候选定义
  X_r: 当前 child 中的原始 encoder features（有效 patch 为 n_r 个）
  mu_r = masked_mean(X_r)
  Q_r = Wq LN_q(mu_r)
  K_r = Wk LN_x(X_r)
  V_r = Wv LN_x(X_r)
  对每个 head：A_r = softmax(Q_r K_r^T / sqrt(96) + packed_padding_mask)
  delta_r = Wo concat_heads(A_r V_r)
  u_r = mu_r + alpha * delta_r
  output = shared_head(final_LN(concat(global_tokens, all_u_r)))

宽度 768，8 heads，每 head 96；一层 attention。
LN_q 与 LN_x 独立；K/V 共用 LN_x；Q/K/V/O 均无 bias。
alpha 为所有 child 共享的单个无约束可学习标量，初始化 0.1。
这个初始值只是明确实验规格，并非已调出的最优值；alpha 不等于实际残差范数比。
不新增 local FFN、local PE、sequence embedding、额外损失或病灶标签。
Global、child 分区、输出顺序、总 token、最终 LN/head 均沿用 ADirect 的设计。
所有 child 共用 reader 参数，但各自有来自本区域视觉均值的 query。
局部读取仅查看本 child，不混入其他 child 的原始 memory。
区域索引 padding mask 不是病灶 mask，也不是对图像背景的筛选。

参数估算
  bias-free Q/K/V/O: 4 * 768^2 = 2,359,296
  两个 affine LN:   4 * 768   = 3,072
  一个 alpha:       1
  新增参数:         2,362,369
  总计 128x:        30,023,425
  总计 256x:        30,015,745
实现后的真实模型参数核对与 CPU BF16 反传记录见
experiments/isolated_20261101_adirect_variants_preflight/verification.json。

预算
  128x: 17 global + 52 local = 69，N=8775，实际压缩 127.173913x
  256x:  7 global + 28 local = 35，N=8775，实际压缩 250.714286x

关键解释
  残差不是人工标注的误差，而是 CE 端到端学出的特征修正。
  重新读取原始 patch，才有机会保留被均值弱化的差异；仅对 mu 做 MLP 不会重新获得 patch。
  mean 支路保持直达不等于最终信息无损，也不保证训练后 accuracy 不退步。
  无显式相对位置的 mean-query attention 对已编码 patch 向量的排列仍不敏感；
  encoder 自身可以包含位置信息，不能据此说整个模型没有位置感知。
  此候选不会单独解决跨 child 解剖关系、脑区语义、多病灶句子绑定或重复事实。

重绘
  CUDA_VISIBLE_DEVICES='' /mnt/sda/mingjie/conda_envs/projector/bin/python \
    /mnt/sda/jyzhang/VLM_Projector/Pipeline_0929/figures/bindformer_adirect_localread_overview/render_overview.py

仅在 CPU 构造未训练的现有 ADirect 以核对底座预算与参数；无需编码器、LLM 或训练检查点。
图的画布为 1752×632 的比例，与 A4/ADirect overview 完全相同。
