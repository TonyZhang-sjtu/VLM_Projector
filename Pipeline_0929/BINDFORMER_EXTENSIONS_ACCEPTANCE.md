# BindFormer 可选扩展验收与待决策项

日期：2026-10-16。方案细节见 [设计文档第 26 节](BINDFORMER_DESIGN_AND_IMPLEMENTATION_PLAN.md#26-可选结构与特征重建实现2026-10-16)。

## 交付范围

| 组件 | 实现 | 默认行为 |
| --- | --- | --- |
| 几何 | 有效 FOV、padding、DHW/XYZ、grid/voxel/mm；10 维统计及两个几何 MLP | `geometry_enabled=false` |
| 包内 SA | 同槽全部角色/子 query 的 pre-LN SA 残差，可设层数 | `within_slot_sa=false` |
| 局部读取全局 | CA 残差、每槽 sigmoid 门控，可设层数 | `local_reads_global=false` |
| 重建 `L_feat` | 最终 Z 重建冻结 NeuroVFM 特征；缓存导出、训练聚合、验证日志、checkpoint | `feature_loss_weight=0`，不创建头 |
| 每角色多 query | 四个独立正整数参数；不同子 query 偏移、独立视觉读取、自动重分配 G/K | 四项均 1 |
| 消融入口 | `--knowledge-mode none`、`--support-beta 0`、角色 query 数与结构覆盖参数 | 原知识和支持域开启 |
| `L_text` | 按本轮要求不实现正负候选/对比损失 | 权重必须 0 |

没有修改两个 bash 启动器的默认配置或终止正在运行的训练。新增组件启用需要新实验配置；不自动将现有最小版 checkpoint 部分加载到不同结构。

## 分节点验收

验收文件集中在 `runs/bindformer_extensions_acceptance/`。

| 节点 | 检查内容 | 结果/记录 |
| --- | --- | --- |
| 默认兼容 | 修改前保存的有/无知识模型：参数键、初始化值、输出、输入梯度、参数梯度、RNG 精确一致 | `legacy_name_definition.pt`、`legacy_none.pt`；专项测试通过 |
| 活跃实验兼容 | 正在运行的 r8 配置与旧 provenance 的 projector 字段、contract_id 一致 | 只读比较通过 |
| 预算 | 多组不同 J、M、global_fraction，对照穷举整数 oracle；无丢弃或补复制 token | 通过 |
| Query | 角色顺序、子 query 区分、beta 展开、所有新增 query 参数有梯度 | 通过 |
| 几何 | 解析可手算中心/协方差/体积、部分 padding、全 padding、平移不变性、6 种轴置换、斜向方向矩阵、BF16 autocast 中 FP32 计算 | 通过 |
| SA/CA | SA 不跨槽串信息；CA 依赖全局输入；门控趋近零时退化为 identity；单独/组合开关、activation checkpoint、保存恢复 | 通过 |
| 重建 | 只经 Z 回传；教师 stopgrad；改变 Z 会改变预测；无放回采样、私有 RNG；完整输出 MLP 收到重建梯度 | 通过 |
| 教师缓存 | 冻结基座、输入/预处理/坐标/教师不匹配拒绝、manifest 更改拒绝、原子写入 | 通过 |
| CE + feature | 冻结 decoder 无参数梯度；训练中的 encoder 有梯度；梯度累积和整批等价 | 通过 |
| DDP | 真正启动 2 个 CPU gloo 进程，回答 token 数与图像数均不均衡，对比单进程梯度 | 通过 |
| Trainer | 一次实际 Trainer accumulation loop：训练/验证日志、总损失分母、best/last checkpoint 含重建头 | 通过（小模型、CPU） |
| 完整规模 GPU | N=8775，输入768维、输出4096维，4/8/16/32×；三个结构扩展＋signal J=2＋L_feat；确定性与非确定性 BF16 前向/反向 | 8/8 通过，`cuda_suite.xml` |
| 真实图像集成 | 真实 BraTS train/val 各一张；真实 NeuroVFM LoRA 两步优化；小型冻结 Llama；验证 CE/L_feat；生成 2 token | 通过，`real_runtime.json` |
| 生成隔离 | 将教师缓存 load 和重建 objective 替换为抛错函数，仍能生成 | 通过 |
| CLI | 真实训练数据头 dry-run，J=(1,2,3,2)、beta=0，M=275/G=67/K=26；不创建正式 run | `asymmetric_plan.json` |

最终全仓检查：397 passed、52 skipped（`final_suite.log` / `final_suite.xml`）。最终专项：47 passed（`final_focused.xml`）；与全仓测试有重叠，不能相加。跳过项目包括需要显式设备或其他 opt-in 的测试；本次新增的完整尺寸 GPU 测试已单独显式执行，8 passed（`cuda_suite.xml`）。原有依赖报告了弃用警告，没有测试失败。

真实图像 smoke 的重建指标用于验证计算和梯度，不作为临床质量证据：训练采样各 512 个 patch，L_feat 约 1.993→1.987；验证约 1.979。该检查使用随机初始化的 64 维、1 层冻结 Llama，不能据此评价 8B LLM 的质量或吞吐。完整输出宽度4096的 GPU 检查单独进行，没有重新跑 8B 正式训练。

本次教师缓存只有验收用的 train/val 各一张，位于 `runs/bindformer_extensions_acceptance/teacher/`。启用正式 L_feat 实验之前，需要使用 `scripts/cache_bindformer_teacher.py` 导出所选完整 train/val，不能把这个两张图的验收缓存当成完整训练缓存。

## 参数量

N=8775、32×、四角色各 1 query、内部宽度768、输出4096，仅 projector 可训练参数；冻结 UMLS buffer 不计入。详见 `parameter_counts.json`。

| 配置 | 参数总数 | 相对最小版增加 |
| --- | ---: | ---: |
| 当前最小版 | 50,178,048 | 0 |
| +几何 | 50,379,008 | 200,960 |
| +包内 SA（1层） | 52,541,952 | 2,363,904 |
| +局部读取全局（1层） | 52,544,257 | 2,366,209 |
| +重建头 | 52,026,368 | 1,848,320 |
| 四项全部开启 | 56,957,441 | 6,779,393 |

重建头只在训练/验证计算损失时执行，但仍随 checkpoint 保存。改变 query 数会改变 K/G 和对应参数数目，不能直接套用本表比较。

## 留给用户决定的实现选择

以下是存在多种合理方案的部分。为了完成实现和检验，当前采用了表中方案；新增功能的总开关仍保持关闭，未将这些选择应用到正在运行的正式实验。

| 决策项 | 当前实现 | 其他方案与影响 |
| --- | --- | --- |
| 几何空间 | 启用时默认 `grid`，也实现 `voxel`、`mm` | grid 保留相对位置/FOV比例；voxel 保留 voxel 尺度但跨 spacing 不可直接比较；mm 保留物理大小。需要决定首轮几何消融用哪种 |
| 几何标准化 | 中心相对有效 FOV 中心；voxel/mm 长度尺度100、体积尺度1000；grid尺度1；体积取 log1p | 可按训练集统计均值/方差归一化，或仅用相对尺度。当前固定值不是调优结果；绝对影像原点已消去，未引入脑图谱配准 |
| 几何作用角色 | 仅位置和形态/范围，两个独立 MLP | 可四角色都加入、或共享 MLP＋角色条件；会改变几何信息注入范围和参数量 |
| 子 query 的知识 | 同槽同角色共享一次 UMLS 混合，额外可学习偏移生成不同子 query | 可为每个子 query 独立检索不同概念，增加表达自由度和检索开销；目前未实现该替代方案 |
| 子 query 的自由度 | 偏移跨槽共享；首 query 保留原表达式；无显式多样性 loss | 可每槽独立偏移、为所有子 query 都设偏移，或加多样性正则；本版不保证子 query 最终一定学到互补内容 |
| 包内交互 | 同槽所有角色及其子 query 一起 SA；新增1层、无新增FFN | 可先角色内后角色间分层交互，或加入FFN/更多层；当前层数可配置，其他形式未实现 |
| 全局门控 | 每槽可学习 sigmoid，初值0.1 | 可改初值接近0或0.5；硬0初始化会暂时阻断交互分支的梯度。当前参数允许(0,1)，未以性能选优 |
| 重建教师 | joint_lora 使用固定预训练基座缓存；Stage A 可用冻结输入 | 当前 LoRA 输出 detach 是变化目标；EMA teacher 或其他更强固定 teacher 是另一种研究设计，目前未实现。固定基座约束可能抑制有益的任务适配，需权重消融 |
| 重建对象/尺度 | 每 patch 768维特征先做无参数 LN，再加权 MSE | 原特征 MSE 会保留幅值但更受通道尺度影响；cosine 更偏方向。当前 LN-MSE 已基本侧重标准化特征方向，不能解释为恢复像素、分割或全部临床事实 |
| 采样与权重 | 最多512有效patch、空间粗分区采样；包含有效 FOV 内正常组织和空气，不按病灶重采样 | 可调样本数、全patch或注意力引导采样。均匀覆盖可能弱化小异常贡献；未引入病灶mask。`lambda_feat` 默认0，smoke使用0.1仅为验收值 |
| 重建 query 信息 | 有效 FOV 坐标＋序列；Z为唯一视觉来源 | 可去掉序列或增加相对物理坐标。坐标和序列仍可能学到群体先验，因此应做 Z 置零/图像置换等后续性能检查 |
| 模型选择标准 | best/early stop 继续使用验证 CE，另记feature指标 | 可改联合指标，但会影响与现有baseline的比较；本轮保持原协议 |

已经明确的要求没有另作推断：全局目标比例0.25、四个J默认1、槽迭代默认3、BraTS知识范围、默认CE-only、L_text暂不实现。

## 复现入口

在项目根目录激活原 `projector` 环境，或用 `/mnt/sda/mingjie/conda_envs/projector/bin/python` 替换 `python`。显式指定空闲 GPU，避免与正式训练抢卡。

```bash
PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python -m pytest tests/test_bindformer_extensions.py tests/test_bindformer_feature_loss.py \
  tests/test_bindformer_distributed.py tests/test_bindformer_sweep.py -q

PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PROJECTOR_TEST_DEVICE=cuda:0 \
  python -m pytest tests/test_bindformer_extensions_cuda.py -q

python -m scripts.cache_bindformer_teacher \
  --config runs/bindformer_extensions_acceptance/teacher_config.yaml \
  --output runs/bindformer_extensions_acceptance/teacher --device cuda:0 --limit 1

python -m scripts.validate_bindformer_extensions --device cuda:0 \
  --teacher-cache runs/bindformer_extensions_acceptance/teacher
```
