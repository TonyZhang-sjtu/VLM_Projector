ADirect projector overview

文件
  bindformer_adirect_overview_projector.pdf          矢量 PDF
  bindformer_adirect_overview_projector.svg          可编辑 SVG，文字保留为 text
  bindformer_adirect_overview_projector.png          240 dpi PNG
  bindformer_adirect_overview_projector_preview.png  110 dpi 预览
  render_overview.py                                可复现绘图源码
  architecture_audit.json                           代码生成的参数及预算清单
  caption_en.tex                                   英文 LaTeX 图注

样式参考
  ../bindformer_a4_overview/bindformer_a4_overview_projector.pdf
  复用原 BindFormer renderer 的绘图工具，与 A4 具有相同的画布尺寸、
  配色、字体、圆角框和双分支布局。仅展示当前 ADirect 的实际执行结构。

核心数据流
  X = 原始 encoder patch features，形状 [8775, 768]
  Global: X -> InputAdapter -> 全局 QueryBlock -> [G, 768]
  Local:  X -> 按 A4 的空间子区域直接求均值 -> [4K, 768]
  输出:   cat([Global, Local]) -> final_norm -> shared MLP -> [G+4K, 4096]

图中最关键的分流发生在 InputAdapter 之前：
  InputAdapter 仅服务 Global 分支，包含 LN、Linear、方向感知 grid Fourier
  PE 和 MRI sequence embedding。该 PE 是相对 grid 编码，并非毫米位置编码。
  Local 直接读取 X；不经过 InputAdapter、local LN/V、输出 Linear 或 local FFN。
  Local 分支不额外加入 projector 侧的位置编码或 sequence embedding；
  原始 encoder 特征本身仍可包含空间信息，不能据此理解为没有位置信息。
  所有 Global/Local token 最后仍共同经过逐 token 的 final LN + MLP。
  这组输出层共享权重，不在 token 之间进行 attention 或 token mixing。

空间划分与数值细节
  与 A4 共用 packed_children：K 个 A 的父区域，每个切为 4 个子区域。
  子区域互不重叠，覆盖 encoder grid，包含输入 padding。
  输出固定为 parent-major / child-major，不是学习得到的语义角色顺序。
  图中的 child validity mask 仅排除索引打包的补齐项，不是病灶掩膜或脑区标注。
  FP16/BF16 的均值使用 FP32 累加后转回输入 dtype；FP64 输入保持 FP64。
  图中空间方框和 token 图标均为示意，不是患者影像或 attention 热图。

预算与参数（不计 encoder、视觉 LoRA 和 LLM）
                       nominal 128x      nominal 256x
  Input adapter              632,832           632,832
  Global + queries         7,095,552         7,087,872
  Local mean                       0                 0
  Final LN + MLP          19,932,672        19,932,672
  总 projector 参数       27,661,056        27,653,376
  G / K / 4K              17 / 13 / 52       7 / 7 / 28
  总 visual tokens                69                35
  实际压缩率              127.173913x       250.714286x

相较 A4，去掉的 local LN/V/output/FFN 合计 5,905,152 个参数。
global_fraction=0.25 是整数预算分配目标，实际 G/M 并不总是恰好 25%。
训练仍沿用既有 report CE 框架；此图不包含额外监督或未来的残差读取设计。

重绘（无需 GPU，不加载编码器、LLM 或训练检查点）
  CUDA_VISIBLE_DEVICES='' /mnt/sda/mingjie/conda_envs/projector/bin/python \
    /mnt/sda/jyzhang/VLM_Projector/Pipeline_0929/figures/bindformer_adirect_overview/render_overview.py

脚本重新构造随机初始化的 ADirect，读取 128x/256x 的实际预算与参数数，
核对其模块组成，记录源文件 SHA-256，并检查文字边界。
实际执行代码：src/models/projectors/bindformer_adirect/projector.py
配置文件：configs/projector/bindformer_adirect.yaml
训练入口：scripts/Exp1026/run_baseline_adirect.sh
