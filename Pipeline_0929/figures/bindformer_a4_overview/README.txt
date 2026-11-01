A4 当前实现 Overview（2026-10-31）

文件
  bindformer_a4_overview_projector.pdf          矢量 PDF，与原 projector 面板相同尺寸
  bindformer_a4_overview_projector.svg          可编辑 SVG，普通文字保留为 text
  bindformer_a4_overview_projector.png          高清 PNG
  bindformer_a4_overview_projector_preview.png  预览 PNG
  architecture_audit.json                      从当前模型构造得到的模块/参数/预算清单
  render_overview.py                           可复现绘图源码
  caption_en.tex                              英文图注

原图：../bindformer_overview/bindformer_overview_projector.pdf
沿用原图的绘图工具、配色、字体、圆角框、箭头与横向双分支布局。
新图仅展示 configs/projector/bindformer_a4.yaml 的实际执行结构。
原图是更完整的研究设计，其中部分 optional 结构在当前 Baseline A 已关闭；
不能把原图所有方框都当作 Baseline A 当前正在使用的组件。

重绘（不需要 GPU，不加载编码器/LLM/训练检查点）
  CUDA_VISIBLE_DEVICES='' /mnt/sda/mingjie/conda_envs/projector/bin/python \
    /mnt/sda/jyzhang/VLM_Projector/Pipeline_0929/figures/bindformer_a4_overview/render_overview.py
绘图时会重新构造随机初始化的 A4，仅统计其结构和参数，并保存源文件 SHA-256。
图中区域和 token 图标是结构示意，不是患者数据或真实 attention 可视化。

一、究竟还保留多少结构

按功能：4 个可学习部分 + 1 个无参数空间分区步骤。
  1. Input adapter：输入 LayerNorm、768→768 Linear、方向感知 grid Fourier PE、
     序列 embedding。位置编码是相对 grid PE，不是毫米位置编码。
  2. Global 分支：G 个可学习 query、1 个 QueryBlock（8 heads），
     Q/K/V cross-attention、输出投影、两条残差和 pre-norm FFN。
  3. Local 分支：LN→V Linear→每个子区域的均值→输出 Linear→残差 FFN。
     所有子区域共享这组变换；Global 与 Local 没有共享 Q/K/V 权重。
  4. 输出映射：拼接 global/local；逐 token 最终 LayerNorm；
     Linear(768,4096)→GELU→Linear(4096,4096)。
     最终映射不在 global 与 local token 之间进行 attention 或 token mixing。
  无参数步骤：按 A 的原有区域定义分 K 个父区域，每个再切成 4 个子区域。

按 PyTorch 顶层子模块：8 个 + 1 个独立 nn.Parameter：
  input_adapter, global_layers, value_norm, value, output, ffn, final_norm, head
  独立参数 global_queries 不属于 named_children() 中的模块。
按基础层：13 个 Linear、7 个 LayerNorm、1 个 Embedding、3 个 GELU。
  仅 1 个 attention block；有 Global/Local 两个 FFN，另有最终映射 MLP。

按可训练 projector 参数数（不计 encoder、视觉 LoRA 和 LLM）：
  功能部分                     128x             256x
  Input adapter                 632,832          632,832
  Global（含 learned queries） 7,095,552        7,087,872
  Local                       5,905,152        5,905,152
  Final LN + MLP             19,932,672       19,932,672
  空间分区                            0                0
  总计                       33,566,208       33,558,528
最终输出映射占约 59.4%；结构已精简，不代表参数量已经很小或每个保留组件都被证明必要。
当前 Baseline A 的可训练 projector 参数为 48,928,512 / 48,916,224，
A4 减少约 31.4%。这里比较的是 trainable 参数，不把 A 中已冻结的路由 Q/K 算进去。

二、从 A 中移除和替换的内容

  删除：slot 初始状态、slot 迭代、GRU、slot MLP/路由子系统。
  删除：四种语义角色 embedding/query 与其角色化 query 生成。
  删除：UMLS concept memory、按角色/序列检索和 concept→query 变换。
  删除：局部 reader 的 query/key attention、support bias、softmax 读取。
  保留功能形式：局部 reader 的 value norm/V、输出 Linear、残差 FFN。
  替换：一个父区域由四个语义角色读取 → 四个子区域各产生一个均值 token。

  没有构建：optional 支持域几何编码器、within-slot SA、local-reads-global、
  feature reconstruction head 和额外的辅助损失。当前 A 本来就关闭了这些扩展。
  训练目标仍是 report-generation 的 answer-only CE，LLM 参数冻结。

每个子区域是确定的空间 box，不是有语义标签的脑区、病灶或学习得到的 slot。
实现会按最大 patch 体积选取待切分 box，并沿其最长物理轴二分；先得到 K 个
父区域，然后分别切为 4 个子区域，输出顺序为 parent-major / child-major。
父/子区域互不重叠，覆盖完整 encoder grid（包含输入 padding）。
每个 local token 只直接聚合其子区域的 encoder feature；encoder feature 本身
可能已有更大感受野。Global query 仍读取全部 H。

视觉预算保持与原 A 一致：
  128x：N=8775，G=17，K=13，Local=52，总 M=69，实际 127.173913x。
  256x：N=8775，G=7，K=7，Local=28，总 M=35，实际 250.714286x。
  global_fraction=0.25 是整数预算分配的目标比例，必须同时满足 M=G+4K；
  它不保证所有倍率下实际 G/M 都恰好为 25%。

三、实现方式：独立 projector 类，通过名称选择

入口：scripts/Exp1026/run_baseline_a4_r128_r256.sh
  projector_method=bindformer_a4
  → scripts/ft_ablation/_constant_lr.bash
  → python -m src.cli.baseline_sweep --projector bindformer_a4
  → configs/projector/bindformer_a4.yaml
  → MultimodalModel / build_projector
  → PROJECTORS['bindformer_a4'] = BindFormerA4

新的独立实现：src/models/projectors/bindformer_a4/projector.py
  class BindFormerA4(StructuredProjector)
  __init__：只构造图中的保留模块；全部按当前训练入口重新初始化。
  _local_tokens：gather 子区域索引，FP32 累加求均值，输出变换 + FFN。
  forward：InputAdapter → 两个分支 → cat(global, local) → final_norm → head。

空间划分：src/models/projectors/bindformer_a4/regions.py
  nested_boxes / packed_children；复用原 A 的纯几何函数 spatial_boxes。
预算和结构验证：src/models/projectors/bindformer_a4/config.py
  CHILDREN_PER_REGION=4；allocate_a4_tokens / validate_params。
注册：src/models/projectors/registry.py
默认参数和严格检查：src/models/projectors/specs.py
倍率预算预检：src/models/projectors/budget.py

复用的是组件代码定义，不是从旧训练检查点复制参数：
  InputAdapter 来自 bindformer/input_adapter.py；
  QueryBlock、mapping_head、StructuredProjector 来自 projectors/common.py；
  spatial_boxes 来自 bindformer/regions.py；
  allocate_tokens 来自 bindformer/budget.py。
预算工具中的 num_roles=4 在此仅用于相同整数取整；A4 的结果使用
num_regions / children_per_region，不会据此构建语义角色模块。

没有通过 BindFormer3D(开关组合) 来模拟 A4，没有继承完整 BindFormer3D，
也没有先构造完整 A 再把多余模块关掉。旧 slot/UMLS/role 参数传给 A4 会被
strict config 拒绝。projector.name 选择不同的类，而不是切换 A 内部路径。
A4 自己仍有正常超参数（宽度、heads、global_depth、PE、sequence_condition、
head_type、checkpoint_blocks）；checkpoint_blocks 只控制显存/重计算。

核心计算（省略 shape、metadata、预算校验和 activation checkpointing）：
  h = input_adapter(features, coords, metadata, sequence)
  v = value(value_norm(h))
  mu = masked_mean(v[child_indices], child_valid)
  local = output(mu)
  local = local + ffn(local)
  glob = global_queries[None]
  for layer in global_layers:
      glob = layer(glob, h[None])
  z = head(final_norm(cat([glob[0], local], dim=0)))

该结构是 A 的精简双分支基座，不是完整原始 BindFormer，也不是标准 Q-Former。
图片只证明实现/预算关系，不代表已证明定位性能改善或各保留组件都有必要。
本次新增绘图文件，不修改训练实现、现有 shell 或原 Overview。
