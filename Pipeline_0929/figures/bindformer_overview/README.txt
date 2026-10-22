BindFormer 论文 Overview 图
生成日期：2026-10-18

文件
  bindformer_overview.pdf            整图，矢量 PDF
  bindformer_overview.svg            整图，可编辑 SVG，普通标签保留为文本
  bindformer_overview.png            高清 PNG，3960 × 3168
  bindformer_overview_preview.png    预览 PNG，1800 × 1440
  bindformer_overview_projector.pdf  单独的 projector 结构面板
  bindformer_overview_training.pdf   单独的训练目标及训练日程面板
  caption_en.tex                    英文论文图注及 figure* 插入示例
  render_overview.py                可复现绘图源码，仅依赖 matplotlib 和 numpy

重绘
  python /mnt/sda/jyzhang/VLM_Projector/Pipeline_0929/figures/bindformer_overview/render_overview.py

图的叙事
  (a) 端到端：单序列 MRI → NeuroVFM → BindFormer → 冻结 Llama → Findings。
  (b) 结构：竞争槽先组织证据；概念记忆给角色 query 提供语义先验；同槽
      四类角色通过共享支持域进行视觉读取；全局分支补充整体结构和正常
      上下文；输出保持固定 token 预算。
  (c) 训练：L_RG + λ_f L_feat + λ_s L_text，两个辅助分支均接在最终 Z。
      底部按用户确认的当前协议展示三个阶段：
      1. projector warm-up；
      2. visual LoRA warm-up，projector 学习率保持不变；
      3. projector + visual LoRA joint 训练。
      阶段日程以 projector 和 LoRA 为核心；启用的辅助头随其训练配置优化。

版本及语义约定
  1. 本图展示用户选定的研究设计，不是当前默认 CE-only 实验的执行轨迹。
     L_text 按设计文档 20.3、21.3 节绘制；截至所读文档第 26 节，其实现
     仍未完成。本次只制作图和图注，没有启用损失或修改训练代码。
  2. L_text 保留原设计的多实例对比学习含义。图中 max_k cos(u_k, v_l)
     是匹配分数，不是将 L_text 改成仅正例余弦回归。按用户要求，图中不画
     负样本来源、筛选过程或候选分母；也没有旧版 tuple/binding 辅助头。
  3. joint LoRA 下 L_feat 使用冻结且无 LoRA 的 NeuroVFM 基座的固定教师
     特征 X^T。不会把当前持续变化的视觉特征 detach 后误称为固定教师。
     重建头预测特征而非 MRI 像素，只读取 Z、位置和序列；X^T 只作为目标。
  4. 文本匹配输入是最终 Z 的整槽角色包，报告片段只进入训练辅助分支，
     不用于在线筛选当前患者的知识候选。图中的训练文本 encoder 不强行
     命名为 SapBERT；SapBERT 的明确选择对应概念记忆的离线编码。
  5. 首次属性读取的 value 来自 H，不直接将含知识的 query 残差写入输出。
     共用 A 是软归属偏置，不代表四个角色的 attention 完全相同。
  6. 支持域几何、包内 self-attention、局部读取全局均标为 opt.。这些模块
     可以消融关闭；未画成默认启用。图中 3D PE 是一般空间编码表述，不把
     当前 grid_fourier 误标为已实现的毫米 PE。
  7. 主图按四角色、每角色一个 query 展示 M = G + K R，R = 4。若展示
     每角色多 query 扩展，应替换为 Q = sum_r J_r、M = G + K Q，并把同角色
     query 组展开。当前画法对应文档中的原始默认 query 数。
  8. 槽是学习得到的证据组，不是已验证的病灶实例；背景槽不输出 token。
     A 热图、角色注意力和 MRI 切片图标均为结构示意，不是实验可视化。
     不把该结构或辅助损失的目的写成已验证的临床效果。

版面与复用
  - 论文优先使用 PDF；SVG 适合 Inkscape / Illustrator 中改字、颜色和位置。
  - 结构和训练面板另存为 PDF，便于按论文版面拆图、放大或重新组合。
  - SVG 主要字体为 DejaVu Sans；数学式也保留矢量内容。
  - 原始设计文档以及用户正在编辑的实验脚本均未修改。

信息来源
  BINDFORMER_DESIGN_AND_IMPLEMENTATION_PLAN.md：第 3–8、20–21、25–26 节。
  configs/projector/bindformer3d.yaml：当前默认角色、query 数及开关。
  本轮用户补充：两阶段 warm-up 的确切顺序与 projector LR 保持方式。
