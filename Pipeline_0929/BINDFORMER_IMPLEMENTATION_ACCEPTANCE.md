# BindFormer3D 首版实现与独立验收

日期：2026-10-15。实现状态：工程验收通过。本文的早期 Stage A-only 训练协议已被后续 joint_lora baseline 协议取代；下方记录的工程验收仍有效，不能作为正式训练结果。

## 确认的配置

- 全局目标占比 0.25，局部 0.75；M=ceil(N/ratio)，每槽四角色，G/K 选择最近合法整数解，平局取 G 更大者。
- 槽迭代默认 3 次；可学习背景槽参与竞争但不输出 token。
- 四角色 query 来自图像槽、角色 embedding、可选概念检索；视觉 value 来自图像特征，不将概念直接作为输出 token。
- 默认一层属性读取、一层全局 CA；归一化三维网格位置编码与序列条件开启。
- 概念输入采用 name + short definition，冻结 SapBERT 最终 CLS，L2 normalize。
- 训练和词表均限定 BraTS；只加载 120 条 brats_train 范围向量，排除其他两队列独有的八条概念。
- 正式比较使用 `joint_lora`：projector warm-up 后进入 projector+视觉 LoRA 联合训练；LLM 冻结，唯一目标为原有答案 token CE。工程验收中的单步冻结 encoder 检查仍是组件验收，不代表正式训练协议。

## 文件与模块

| 模块 | 实现 |
| --- | --- |
| 总流程 | [projector.py](../src/models/projectors/bindformer/projector.py) |
| 预算与参数校验 | [budget.py](../src/models/projectors/bindformer/budget.py)、[config.py](../src/models/projectors/bindformer/config.py) |
| 输入、相对位置、序列 | [input_adapter.py](../src/models/projectors/bindformer/input_adapter.py) |
| 槽竞争和 GRU 更新 | [slots.py](../src/models/projectors/bindformer/slots.py) |
| BraTS 向量筛选与软检索 | [concept_memory.py](../src/models/projectors/bindformer/concept_memory.py) |
| 角色 query 与视觉读取 | [role_queries.py](../src/models/projectors/bindformer/role_queries.py)、[reader.py](../src/models/projectors/bindformer/reader.py) |
| 后续模块接口 | [types.py](../src/models/projectors/bindformer/types.py) |
| 参数 YAML | [bindformer3d.yaml](../configs/projector/bindformer3d.yaml) |
| joint_lora baseline 入口 | [run_bindformer3d.sh](../scripts/baselines/run_bindformer3d.sh)、[run_bindformer3d_2.sh](../scripts/baselines/run_bindformer3d_2.sh)、[bindformer_sweep.py](../src/cli/bindformer_sweep.py) |

辅助输出为 ProjectedVolume.auxiliary，默认 None；开启 return_auxiliary 时包含 X、坐标、槽、归属、角色/全局 token 与概念权重，保持计算图。几何 provider、geometry query 输入、包内交互和辅助目标协议已预留。未实现的模块/损失若设为开启会在配置阶段报错，不会静默忽略。

## 分阶段验收

| 阶段 | 独立检查 | 结果 |
| --- | --- | --- |
| 预算 | 四档预期表、穷举最近整数 oracle、平局规则、保存恢复、非法配置 | 通过，19 项 |
| 模块 | 槽更新与显式公式对照、最终重算归属、三轮/跨图重置、checkpoint 梯度等价、方向/轴转换 | 通过，包含在 15 项模块测试中 |
| 知识 | 实际只加载 120 行、八条独有词被移除、角色/序列 mask 权重严格为零、冻结 E、资产篡改拒绝 | 通过 |
| 属性读取 | 显式多头 softmax 输出/梯度 oracle、beta=0、共享支持域、无 query 直接残差、固定角色布局 | 通过 |
| CE/生成/恢复 | 现有 LlamaDecoder + 小型 Llama、知识开启/关闭、beta=0、所有参数梯度、两端冻结、序列上下文、optimizer/checkpoint 恢复、同路径资产变更拒绝 | 通过，5 项 |
| 启动协议 | Stage A、BraTS 范围、词表限制、四档解析、消融配置、拒绝覆盖已有目录 | 通过，5 项 |
| 全仓库回归 | 新方法注册后的通用 projector、训练/解码/预算/评测/视觉等测试 | 355 passed，44 skipped |
| 显式 GPU | 四档真实大小 [8775,768] 合成特征，BF16，确定性/快速两种模式，激活重计算、输出与全部梯度 | 8 passed |
| 真实 MRI + NeuroVFM + 8B | 真实图像的前向 CE、反向传播、一次 AdamW 更新、两 token 生成；压缩率 4 和 32 | 两档均通过 |

44 个回归跳过项是需显式选择 GPU 的测试；其中 BindFormer 的八项已另行指定 GPU 3 全部执行。回归中第三方 flash-attn 有弃用警告，没有失败项。

全仓库 JUnit：[pytest_cpu.xml](../runs/bindformer_acceptance/pytest_cpu.xml)。四档真实数据头预检：[plan.json](../runs/bindformer_stage_a/acceptance_v1/plan.json)，选择 1960 条训练记录和 268 条验证记录，均来自 BraTS。

## 四档预算

| 压缩率 | 输入 N | 输出 M | 全局 G | 槽 K | 局部 KR |
| --- | --- | --- | --- | --- | --- |
| 4 | 8775 | 2194 | 550 | 411 | 1644 |
| 8 | 8775 | 1097 | 273 | 206 | 824 |
| 16 | 8775 | 549 | 137 | 103 | 412 |
| 32 | 8775 | 275 | 67 | 52 | 208 |

## 真实端到端结果

环境：RTX A5500 24 GiB，BF16，确定性模式，encoder/LLM 冻结，LLM 与 projector 激活重计算开启。一个真实 BraTS MEN T1C 训练样本，8775 个 patch、65 个监督答案 token，microbatch=1。各档从相同 seed 独立初始化，不传递训练后权重。

| 压缩率 | projector 参数量 | 更新前 CE | 一次更新后 CE | 峰值 allocated 显存 | 短生成 |
| --- | --- | --- | --- | --- | --- |
| 4 | 50,824,704 | 7.45465 | 5.81749 | 17.59 GiB | 2 tokens，通过 |
| 32 | 50,178,048 | 6.17224 | 5.47050 | 16.76 GiB | 2 tokens，通过 |

详细报告：[ratio 4](../runs/bindformer_acceptance/runtime_ratio4.json)、[ratio 32](../runs/bindformer_acceptance/runtime.json)。报告包含预算、词表指纹、源文件哈希、可训练参数检查、CE 和显存。有限样本更新只用于证明训练链路连通，不是泛化效果或最大显存需求的测量。

验收中修正了 runtime 脚本的模式设置：Hugging Face 预训练模型加载后处于 eval 模式，必须像现有 Trainer 一样调用 model.train()，冻结 LLM 才会在训练前向使用 activation checkpointing。最初 ratio 4 在未切换模式时 OOM；修正并断言 training/checkpointing 状态后通过。正式 Trainer 原本已有 model.train()，此次未更改其训练逻辑。

## 运行与消融

在项目根目录：

    bash scripts/baselines/run_bindformer3d.sh --dry-run
    bash scripts/baselines/run_bindformer3d.sh --ratios 32

`run_bindformer3d.sh` 默认在物理 GPU 3 运行 8、16、32；`run_bindformer3d_2.sh` 默认在物理 GPU 1 运行 4。两者使用 joint_lora 的相同训练日程，并在每档训练后运行完整 BraTS test 和全部指标。可通过 `CUDA_VISIBLE_DEVICES` 覆盖默认卡号。

主要消融：

    bash scripts/baselines/run_bindformer3d.sh --ratios 32 --knowledge-mode none
    bash scripts/baselines/run_bindformer3d.sh --ratios 32 --support-beta 0
    bash scripts/baselines/run_bindformer3d.sh --ratios 32 --slot-iterations 1
    bash scripts/baselines/run_bindformer3d.sh --ratios 32 --global-fraction 0.125
    bash scripts/baselines/run_bindformer3d.sh --ratios 32 --set projector.params.background_slot=false

四角色独立 beta：

    bash scripts/baselines/run_bindformer3d.sh --ratios 32 --support-beta 1 1 0.5 1

复现测试：

    OMP_NUM_THREADS=2 /mnt/sda/mingjie/conda_envs/projector/bin/python -m pytest tests -q
    CUBLAS_WORKSPACE_CONFIG=:4096:8 PROJECTOR_TEST_DEVICE=cuda:3 OMP_NUM_THREADS=2 /mnt/sda/mingjie/conda_envs/projector/bin/python -m pytest tests/test_bindformer_cuda.py -q

复现真实端到端验收：

    CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=4 /mnt/sda/mingjie/conda_envs/projector/bin/python -m scripts.validate_bindformer_runtime --device cuda:3 --ratio 32

现有 baseline sweep 采用 joint_lora；做严格比较时应统一训练策略和实际输出 M。当前网格 PE 是相对位置，未实现毫米坐标几何统计；CE-only 也尚未验证真实病灶分割、角色语义分工或实例绑定性能。
