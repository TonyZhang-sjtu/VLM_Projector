# AnchorRead3D A1–A4：实现记录与 Marlowe 同步交接

日期：2026-11-08（Asia/Shanghai）。源目录：`/mnt/sda/jyzhang/VLM_Projector`。
仓库：<https://github.com/TonyZhang-sjtu/VLM_Projector>。
本次发布标签：`anchorread3d-ablation-20261108`；完整 commit SHA 以发布回复及远端标签解析结果为准。

本次已实现 **A1–A4 × RG/VQA × 32×/256×，共 16 个独立实验配置**。
新增独立 projector 名称、实现包、配置和八个 shell，复用原任务训练/评估流程。
源机已完成两轮 16 组启动 dry-run 和相关 CPU 验证，尚未启动正式消融训练。
本文中的训练配方与消融定义是本次准则，不沿用旧 Marlowe 队列或旧同步文档的默认实验列表。

## 1. A0 框架与消融的精确定义

A0 即 `bindformer_adirect_localread`（ADirect + LocalRead）：encoder 输出每个 patch 的 768 维特征 `X`，projector 总共输出 `M=G+L` 个 4096 维视觉 token。

- Global：`InputAdapter` 对 `X` 做通道映射，加入方向感知的 grid Fourier 3D PE 和 MRI sequence embedding；`G` 个 learned queries 通过原单层 `QueryBlock` 读取全体 patch。
- Local：将空间分成父区域，每个父区域再分四个互不重叠的 cell。每个 cell 的原始 encoder 特征为 `X_r`，初值为 `mu_r=mean(X_r)`。
- LocalRead：`Q=LN(mu_r)`，`K/V=LN(X_r)` 经各自线性层后作多头 cross-attention，再经输出线性层得到 `delta_r`。每个 cell 一个 query，只读取该 cell 的 patch；没有额外 local FFN、local PE 或 local sequence embedding。
- 局部输出：`u_r=mu_r+alpha*delta_r`，共享可学习标量 `alpha` 初始化为 0.1。
- 最后拼接 global/local token，逐 token 经过共享的 final LN 和 MLP，映射到 4096 维。

这里的父区域只组织几何划分，不输出额外父 token，不对四个子 token 再求均值形成第二级特征。LocalRead 是 mean query 对区域 patch 的 cross-attention，不是 patch 之间的 self-attention。

| 变体 | 注册名 | 实现类 | 相对 A0 的变化 |
|---|---|---|---|
| A1 | `anchorread3d_a1_no_localread` | `NoLocalRead` | 局部输出只有 `mu_r`；完全不构造 LocalRead。继承原 `BindFormerADirect`，混合 G/L、分区、global 和 head 保持一致。 |
| A2 | `anchorread3d_a2_no_mean_bypass` | `NoMeanBypass` | 局部输出只有 `alpha*delta_r`。**仍使用 mu_r 构造 query**，K/V、alpha、global、分区和 head 保持一致。 |
| A3 | `anchorread3d_a3_global_only` | `GlobalOnly` | `G=M,L=0`。沿用 A0 的 InputAdapter、单层 QueryBlock 和 head，取消局部路径。不是六层 Perceiver_NV。 |
| A4 | `anchorread3d_a4_local_only` | `LocalOnly` | `G=0,L=M`，保留 `mu_r+alpha*delta_r`；不构造 global adapter、queries、attention layers。局部分区采用下面的 nested4-tail 规则。 |

A1/A2 是保持 G、L 和分区不变的模块消融；A3/A4 是固定总预算 M 的分配对照。
A4 的局部 token 更多，空间粒度也随之改变，论文中不能把它解释为保持全部其他几何因素不变的“仅删除 global”。每个变体独立从头训练，不从训练好的 A0 推理时关闭分支。

### A4 的余数组：靠近主方法的父区域/子区域组织

`K=ceil(M/4)`。先用原 `spatial_boxes` 划分 K 个父区域，再将前 K−1 个各分四个子区域，最后一个父区域直接分成 `M−4(K−1)` 个子区域。

例如 M=35 时为 `8×4+3`，M=275 时为 `68×4+3`。最后一个父区域内的全部 patch 都进入三个实际子区域，没有先生成四个后丢掉一个，也没有补虚拟 token。所有 patch 恰好覆盖一次，token 顺序为 parent-major、child-major。最后一个父区域按几何排序确定，不依赖病灶或标签。

本次 A4 不是旧 `ADirect + AllLocal`：旧版本没有 LocalRead，global_fraction=0 时也使用平铺的 M-cell 分区。本次同时保留 LocalRead 并采用 nested4-tail，注册名和 allocation version 单独记录。

## 2. 代码对应关系及改动边界

| 路径 | 职责 |
|---|---|
| `src/models/projectors/anchorread3d_ablation/variants.py` | 四个类及方法映射；A1/A2/A4 复用原实现，A3 使用原公共模块构造 global-only。 |
| `src/models/projectors/anchorread3d_ablation/budget.py` | 稳定注册名、固定 global_fraction、G/L 分配、children_per_parent、ablation_id。 |
| `src/models/projectors/anchorread3d_ablation/regions.py` | A4 nested4-tail 分区、patch 索引及 padding mask；不改旧分区代码。 |
| `src/models/projectors/anchorread3d_ablation/config.py` | 四个消融的默认参数和校验；不允许通过覆盖 global_fraction 偷换变体。 |
| `configs/projector/anchorread3d_a*.yaml` | 每个变体的独立架构 preset，768 内部维度、8 heads、单层 global、共享 MLP head 等。 |
| `configs/experiments/anchorread3d_ablation/training.yaml` | RG/VQA 共用的 r16 训练配方；不包含数据、预处理或 task 字段。 |
| `scripts/ablation/_common.bash` | 解释器/路径配置、独立输出位置、原任务 launcher 调用。 |
| `scripts/ablation/{rg,vqa}_a*.sh` | 八个入口，各默认顺序运行 32×、256×。 |
| `scripts/ablation/preflight.py` | 调用实际八个 shell 做 dry-run，保存 16 份解析配置、budgets.csv、summary.json。 |
| `scripts/ablation/collect_results.py` | 只扫描指定消融 runs，生成新的 manifest.json/results.csv/results.xlsx。 |

必要的既有文件接入只有：

1. `registry.py` 注册四个新 factory；`specs.py` 接入 defaults/validator；公共 `budget.py` 接入 exact-M 分配。
2. `src/cli/vqa_sweep.py` 识别新名称，并像原 ADirect 系列一样自动加载架构 YAML。
3. `bindformer_adirect_localread/projector.py` 的 `LocalResidualRead.forward` 新增 keyword-only `add_mean=True`；默认行为及 state_dict 参数不变，只有 A2 传 `False`。
4. 泛化测试适配新方法及独立 shell 位置；旧 launcher 配方测试显式指定相同比例进行比较。

旧 Exp1026、Exp_VQA、Marlowe 任务入口、训练器、数据加载、预处理、评估主体没有修改；没有改旧实验结果。
调用链仍为 RG `baseline_sweep → train → evaluate_checkpoint`；VQA `vqa_sweep → vqa_train → vqa_evaluate`。

## 3. 16 组实验的预算和训练协议

| 任务 | 名义压缩率 | 输入 N | 输出 M | A1/A2 的 G,L | A3 的 G,L | A4 的 G,L | A4 父子分配 |
|---|---:|---:|---:|---|---|---|---|
| RG | 32× | 8775 | 275 | 67,208 | 275,0 | 0,275 | 68×4+3 |
| RG | 256× | 8775 | 35 | 7,28 | 35,0 | 0,35 | 8×4+3 |
| VQA | 32× | 9555 | 288 | 72,216 | 288,0 | 0,288 | 72×4 |
| VQA | 256× | 9555 | 36 | 8,28 | 36,0 | 0,36 | 9×4 |

RG 沿用当前原生预算协议；VQA 沿用 `neuroqa_mni1mm_shared_m_v1`，不能自行统一成 `ceil(N/ratio)`。
名义 32/256 不等于精确 N/M，名义 global_fraction=0.25 也不等于取整后的精确 G/M。
实际数值和 `allocation_version` 已写入 resolved_budget 和输出 metadata。

训练完整参照 `scripts/Exp1026/run_baseline_adirect_localread_r16.sh`，只改变 projector 和压缩率：

| 项目 | 本次统一值 |
|---|---|
| seed / stage | 42 / joint_lora |
| LLM / encoder | LLM frozen；encoder 只训练 LoRA，rank16、alpha32、dropout0.05 |
| 两组 LR | projector=2e-4，encoder=2e-4 |
| batch / dtype | microbatch1，effective batch16，BF16，deterministic=true |
| epochs / cache | 最多20 epochs；feature_cache=false；不初始化/恢复旧 checkpoint |
| warmup / hold | projector77、hold0、joint77；joint 阶段 projector hold_peak |
| LR 下限 | cosine_min_lr_ratio=1.0，warmup 后恒定 LR |
| early stopping | patience3，min_delta0.005 |
| 其他 | AdamW wd0.01、betas[0.9,0.999]、eps1e-8、clip1.0；generation max_new_tokens192、greedy |

两个任务保留原有 loss、数据、采样、预处理和验证集选点协议；RG full test/all metrics；VQA test/normal、bootstrap1000、mixed_v2、include_unreviewed、exact_balanced_repeat。
源机 VQA 数据检查按既有 include_unreviewed 策略通过，仍有 131 个待复核 QC case；没有将这些结果标成已完成全局 QC。

旧 VQA A0 可能使用 150/150 warmup 和 min_delta0.002。制作消融表前必须核对 A0 的保存配置；不一致则使用本次统一配方补跑 A0。同步不应为迎合旧 A0 而改动 A1–A4 的训练配方。

## 4. Marlowe 需要适配的路径

先定位 Marlowe 的实际 checkout、解释器、模型、RG 和 VQA 数据缓存及 Slurm 设置。源机 `/mnt/sda/...` 路径只用于记录，不能照搬。已有 Marlowe 数据可直接复用；本轮不要求重新搬运或逐文件重算所有既有资产哈希，需完成入口自身的数据检查和必要的路径/版本核对。

| 位置/变量 | Marlowe 的处理 |
|---|---|
| `PROJECTOR_PYTHON` | 显式 export 本机训练环境的 Python；仅激活 Conda 不会覆盖 shell 中的源机默认路径。 |
| `PROJECTOR_LOCAL_CONFIG` | RG 专用路径 YAML；未设置时 RG 会读当前 checkout 的 `configs/local_paths.yaml`（若存在）。 |
| `VQA_LOCAL_CONFIG` | VQA 专用路径 YAML；VQA 不自动读取 RG local_paths。 |
| `project_root` | 本轮 overlay 建议 `null`，解析为当前发布 checkout；不能指回旧代码目录。 |
| `encoder.path` / `decoder.path` | 本机同版 NeuroVFM / Llama3.1-8B-instruct 权重与 tokenizer。 |
| RG `data.root` / `annotation_root` / `manifest_dir` | 同一套 BraTS、RadGenome 及 single_sequence_findings_coref_v2 划分的本机位置。 |
| RG `preprocessing.cache_dir` | 缓存根目录；检查实际版本哈希目录与 `report_generation_native1mm_v2` 兼容链接，不能直接用可读版本子目录替代根。 |
| VQA `data.manifest_dir` / `preprocessing.cache_dir` | 配对的 official_holdout_patient_guard_v1_mni_v2 manifest 与 neuroqa_mni1mm_v2 缓存。 |
| `ANCHORREAD_ABLATION_RUNS_ROOT` | 新消融 runs 根，shell 会再加 `/rg` 或 `/vqa`。旧 `PROJECTOR_OUTPUT_ROOT` 不是本入口的环境变量。 |
| `ANCHORREAD_ABLATION_RESULTS_ROOT` | 新消融 results 根，VQA shell 会再加 `/vqa_raw`；RG metric 仍在其 run/evaluation 内。 |
| `CUDA_VISIBLE_DEVICES` | 正式 Slurm 作业保留调度器分配的单卡可见性；代码内使用逻辑 cuda:0。 |
| 临床评测资产/依赖 | 当前 checkout 的 `assets/models/evaluation` 和 `third_party/evaluation/python_deps` 必须能访问本机已核验内容；仅改 encoder/decoder 路径不覆盖这两处。 |

VQA 的 manifest 行内 `images`、`lineage.preprocessing_root`、source_index 和预处理审计记录涉及绝对路径。**只改 YAML 的 cache_dir 或 data.root 不会自动重定位这些内容。** 优先沿用 Marlowe 已经匹配并通过检查的 manifest/cache。若需要路径映射，在单独副本中按原迁移流程处理必要路径，保留 QA/患者 ID、划分、问答内容、模态资格、影像内容和审计约束；不能全局替换旧字符串或跳过 provenance 校验。参考既有 [同步说明](GITHUB_UPLOAD_AND_SERVER_SYNC.md) 第十三、十四、十六节中的路径规则，但实验参数以本文为准。

可在新 checkout 创建被 Git 忽略的两份路径文件，下面均为需要替换的示例：

```yaml
# configs/local_ablation_rg.yaml
project_root: null
data:
  root: /MARLOWE_DATA/data
  annotation_root: /MARLOWE_DATA/data/RadGenome-Brain
  manifest_dir: /MARLOWE_DATA/data/RadGenome-Brain/training_manifests/single_sequence_findings_coref_v2
preprocessing:
  cache_dir: /MARLOWE_DATA/preprocessing
encoder:
  path: /MARLOWE_MODELS/neurovfm-encoder
decoder:
  path: /MARLOWE_MODELS/llama3.1-8b-instruct
```

```yaml
# configs/local_ablation_vqa.yaml
project_root: null
data:
  manifest_dir: /MARLOWE_DATA/NeuroQA_derived/official_holdout_patient_guard_v1_mni_v2
preprocessing:
  cache_dir: /MARLOWE_DATA/preprocessing/neuroqa_mni1mm_v2
encoder:
  path: /MARLOWE_MODELS/neurovfm-encoder
decoder:
  path: /MARLOWE_MODELS/llama3.1-8b-instruct
```

这些 overlay 只改路径，不夹带另一套训练超参数、task、预处理算法或 projector。RG/VQA 各自的 preset 先加载，路径 overlay 随后加载，本次共享训练 YAML 再加载；显式 CLI `--set` 仍可覆盖，因此必须审阅最终解析配置。

工作树不含被忽略资产。可在新 checkout 建立到本机既有资产/依赖的必要链接，保留原目录不变。不要链接旧 `src/`，不要用旧 PYTHONPATH 或 editable install 导入另一份实现。临床依赖在独立评分子进程加载，不要把它的 NumPy overlay 全局加到训练环境。

## 5. 验收与启动命令

以下命令在本次发布 checkout 根执行，路径先换成 Marlowe 实际值：

```bash
export PROJECTOR_PYTHON="/MARLOWE_ENV/bin/python"
export PROJECTOR_LOCAL_CONFIG="$PWD/configs/local_ablation_rg.yaml"
export VQA_LOCAL_CONFIG="$PWD/configs/local_ablation_vqa.yaml"
export ANCHORREAD_ABLATION_RUNS_ROOT="/MARLOWE_OUTPUT/anchorread3d/runs"
export ANCHORREAD_ABLATION_RESULTS_ROOT="/MARLOWE_OUTPUT/anchorread3d/results"
export PYTHONNOUSERSITE=1
"$PROJECTOR_PYTHON" -c 'import sys,src; print(sys.executable); print(src.__file__)'

# 不加载模型/不分配 GPU；输出目录必须尚不存在。
"$PROJECTOR_PYTHON" scripts/ablation/preflight.py \
  --output-dir .tmp/anchorread3d_ablation_marlowe_preflight --check-vqa-data

# 结构和结果汇总测试不需要真实 8B/GPU。
env CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  "$PROJECTOR_PYTHON" -m pytest -q \
  tests/test_anchorread3d_ablation.py tests/test_anchorread3d_results.py
```

`preflight.py` 中 CUDA_VISIBLE_DEVICES=0 只给无 GPU 分配的 dry-run 使用，不代表正式作业可以覆盖 Slurm 的设备分配。原八个 shell 当前要求单个数字 GPU 可见标识；如 Marlowe 实际返回 GPU/MIG UUID 或多个设备，先核对本机约定，不能将它们盲目替换成 0 绕过调度器隔离。

`tests/test_anchorread3d_training.py` 中的完整宽度测试使用随机特征，不需模型资产；可单独运行 `-k production_width`。小型 backbone 集成测试仍需要本机 tokenizer 和默认相对布局的 RG/VQA manifest，测试 fixture 不自动读取上述两个 shell 路径 overlay；应先用正确的本地资产链接满足默认布局，再跑完整文件。不能因路径缺失修改模型逻辑或放宽断言。

正式入口如下；每次调用默认顺序执行 r32/r256，各自 fresh initialization。应在已分配的 GPU 作业内执行，或加 `--dry-run` 只检查计划。

| 变体 | RG | VQA |
|---|---|---|
| A1 | `bash scripts/ablation/rg_a1_no_localread.sh` | `bash scripts/ablation/vqa_a1_no_localread.sh` |
| A2 | `bash scripts/ablation/rg_a2_no_mean_bypass.sh` | `bash scripts/ablation/vqa_a2_no_mean_bypass.sh` |
| A3 | `bash scripts/ablation/rg_a3_global_only.sh` | `bash scripts/ablation/vqa_a3_global_only.sh` |
| A4 | `bash scripts/ablation/rg_a4_local_only.sh` | `bash scripts/ablation/vqa_a4_local_only.sh` |

加 `--ratios 32` 或 `--ratios 256` 可拆成 16 个单 setting 作业；`--run-tag` 指定新名字。默认 tag 带变体、seed 和时间戳。不要把新的 ablation 塞进旧 `m06–m13` 编排：它们有硬编码任务、旧输出路径及并发设置。先采用一个单卡作业内顺序执行一个入口，增加并发前须实测资源。

Marlowe 可单独创建本机 Slurm 包装器，例如以下正文；account/partition/QoS/GPU/CPU/memory/time/log 路径由 Marlowe 按实际设置通过 `sbatch` 指定，本文不假定旧资源配置仍合适：

```bash
#!/usr/bin/env bash
set -Eeuo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the release checkout}"
: "${SLURM_JOB_ID:?Run inside a Slurm allocation}"
: "${CUDA_VISIBLE_DEVICES:?Slurm must allocate one GPU}"
# 此文件由 Marlowe 自行准备：解释器、RG/VQA 路径、输出根等。
source /MARLOWE_LOCAL_CONFIG/anchorread3d_env.sh
task="${1:?rg or vqa}"
variant="${2:?a1_no_localread, a2_no_mean_bypass, a3_global_only, or a4_local_only}"
[[ "$task" =~ ^(rg|vqa)$ ]] || exit 2
case "$variant" in
  a1_no_localread|a2_no_mean_bypass|a3_global_only|a4_local_only) ;;
  *) exit 2 ;;
esac
shift 2
exec bash "scripts/ablation/${task}_${variant}.sh" "$@"
```

`anchorread3d_env.sh` 应保留 Slurm 的 CUDA_VISIBLE_DEVICES，不复用会覆盖设备/训练设置的旧 env。不要在登录节点运行真实模型训练；本轮交接不自动提交正式长训练。若另行获准做 GPU 短训，从解析配置生成独立 smoke 配置/输出目录，保留真实模型和 N/M，只缩短训练步数/样本数；测试数据不得用于选配置，smoke 不进入正式结果表。

## 6. 输出、恢复与汇总

RG：`$ANCHORREAD_ABLATION_RUNS_ROOT/rg/<tag>/r{32,256}/`，含 training_config、joint_lora/checkpoints、evaluation/best_test。
VQA：`$ANCHORREAD_ABLATION_RUNS_ROOT/vqa/<target>/v2/<scope>/mixed/<projector>/r<ratio>/s42/<tag>/`；预测/指标在 results 根对应 `/vqa_raw/.../test/normal/`。

RG `--resume-run` 指向含 sweep_plan.json 的父目录，复用已完成训练再续评估/后续倍率；它不是任意 last.pt 中断恢复器。VQA `--resume-run` 指向单倍率 launch.json 所在目录，可从 last.pt 恢复，或配合 `--evaluate-only` 评估 best。恢复使用保存配置，不会自动套用新的 r16 配方；跨服务器旧 run 里有绝对路径，不作为本轮 fresh 实验复用。

```bash
# 汇总工具自身不读取 shell 的输出根环境变量，必须显式传入。
"$PROJECTOR_PYTHON" scripts/ablation/collect_results.py \
  --runs-root "$ANCHORREAD_ABLATION_RUNS_ROOT" \
  --output-dir "$ANCHORREAD_ABLATION_RESULTS_ROOT/summaries/$(date -u +%Y%m%dT%H%M%S%NZ)"
```

汇总保存完整来源配置和训练配方摘要；不同 run-tag 保留为不同行，不自动挑最好分数。缺失/未完成评估不填分，保留 RG partial、VQA scope/coverage。输出新 CSV/XLSX，不改主实验工作簿。

## 7. 源机验证记录与上传范围

- 两轮八个 shell/16 组 dry-run 全通过，预算记录完全一致；VQA 数据按既有 include_unreviewed 策略检查通过。
- 相关测试去重后 698 项通过，未解决失败为 0。包含 16 组实际 N/M、768→4096 宽度 FP32 前向/反向/参数更新；16 组真实网格小宽度确定性 BF16/checkpointing；16 组小型 backbone 的 BF16 RG/VQA loss、LoRA 阶段、生成和生产 checkpoint 保存/恢复。
- 另有 7 组完整宽度 BF16 检查通过后，因 AVX2 CPU 软件 BF16 耗时主动结束，没有出现测试失败。正式训练仍是 BF16。
- 上述随机特征/小型 backbone 检查不等于真实 NeuroVFM + Llama 8B CUDA 端到端短训。源机全部 GPU 被已有任务占用，本轮没有进行该项，也没有启动正式实验。
- 完整本地记录在 `experiments/anchorread3d_ablation_preflight_20261108/`。该目录含机器路径、配置及数据预检内容，保留源机，不整目录上传；本节为可随代码发布的验证摘要。

上传本轮消融源码、配置、脚本、测试与本文。已有无关未追踪文档、`.marlowe/`、`.marlowe-migration/`、logs、实验快照、模型、数据、checkpoint、预测、机器路径 overlay 和临床资产不属于本次提交。Marlowe 从 Git 获得代码后继续使用自己的资产。

## 8. 可直接发给 Marlowe 端 agent 的 prompt

将 RELEASE_COMMIT 替换为源机发布回复中的完整 SHA；其余机器路径由 Marlowe agent 实际检查。

```text
请把 Marlowe 上的 VLM_Projector 同步到本次 AnchorRead3D ablation 发布：
仓库 https://github.com/TonyZhang-sjtu/VLM_Projector.git
标签 anchorread3d-ablation-20261108
RELEASE_COMMIT=<源机本次发布的完整 SHA>

目标：准备 A1–A4 × RG/VQA × 32×/256× 共 16 个 fresh 实验的可运行代码、
本机路径配置和启动命令，完成 dry-run/必要验证；本次先不提交正式长训练。
先完整阅读该提交的
Pipeline_0929/ANCHORREAD3D_ABLATION_IMPLEMENTATION_AND_MARLOWE_20261108.md。

先定位本机仓库、AGENTS.md、Git 状态、Python 环境、Slurm 作业和路径配置。
保留已有本机修改、数据、模型、结果和作业；如当前目录被任务使用或存在分叉，
新建固定 SHA 的独立 worktree。fetch 本标签并核对解析 SHA 与 RELEASE_COMMIT，
实际执行 checkout 的 HEAD 应是该发布 SHA；不要用 reset --hard、clean、强推，
不要覆盖旧源码目录或停止已有任务。路径配置在本机独立文件中保留。

消融代码在 src/models/projectors/anchorread3d_ablation/：
variants.py 定义 NoLocalRead / NoMeanBypass / GlobalOnly / LocalOnly；
budget.py 固定注册名与预算；regions.py 实现 A4 nested4-tail；config.py 校验。
四个新名称为 anchorread3d_a1_no_localread、anchorread3d_a2_no_mean_bypass、
anchorread3d_a3_global_only、anchorread3d_a4_local_only。

A0 是 bindformer_adirect_localread：global 读取全体 patch；local 初值是 cell
内原 encoder patch 的均值 mu，LocalRead 用 mu 作 query、同 cell patch 作 K/V，
输出 mu+alpha*delta；alpha 初始化0.1；concat 后共享 LN+MLP，768→4096。
A1 仅取消 LocalRead，local=mu，复用原 ADirect。
A2 仅取消输出均值直通，local=alpha*delta，仍保留 mean query 和原 K/V。
A3 用全部 M 个 global queries，复用 A0 单层 global block，不用六层 Perceiver。
A4 用全部 M 个 local tokens，保留 mu+alpha*delta，不含 global 输入处理/attention。
A4 先分 ceil(M/4) 个父区域，前面各四个子区域，最后直接分剩余1–4个。
M35=8×4+3，M275=68×4+3；不丢 patch、不补虚拟 token，不产生额外父 token。
这不同于旧 ADirect+AllLocal 的无 LocalRead、平铺 M-cell 路径。
A1/A2 固定原 G/L/分区；A3/A4 固定总 M，是预算分配对照。

必要主框架接入只有 registry/specs/budget、新名称的 VQA preset 加载，以及
LocalResidualRead.forward 的 add_mean=True 默认参数（只有 A2 传False）。
默认旧 A0 算法不变，原训练器/数据/评估不变。不要把这些变体改写成其他 baseline。

统一训练 YAML 是 configs/experiments/anchorread3d_ablation/training.yaml，
参照 Exp1026/run_baseline_adirect_localread_r16.sh：seed42、joint_lora、LLM冻结、
两组LR2e-4、LoRA16/32/0.05、micro1/effective16、BF16 deterministic、20epochs、
warmup77/hold0/joint77、hold_peak、cosine下限1.0、早停3/0.005。
RG/VQA 各自的数据、loss、预处理、采样和验证选点不合并；VQA 保留 mixed_v2、
include_unreviewed、exact_balanced_repeat。旧VQA A0若是150/150和0.002，
记录其不匹配，后续应同配方补跑A0，不能悄悄改变消融配方。

路径适配：显式设置本机 PROJECTOR_PYTHON；RG 用 PROJECTOR_LOCAL_CONFIG，
VQA 单独用 VQA_LOCAL_CONFIG。overlay只放本机模型、manifest、缓存等路径，
project_root:null 解析到新checkout。RG cache_dir是缓存根，检查版本兼容链接。
VQA要配对本机v2 manifest/cache；manifest行内images、lineage.preprocessing_root
和审计证据也涉及路径，只改YAML不一定足够。优先沿用本机已有合法配对，确需映射
则在独立副本中只改路径并保留数据身份、协议与校验，不重建标签或跳过检查。
新worktree还要提供assets/models/evaluation和third_party/evaluation/python_deps，
以及本机锁定依赖。确认实际import src来自发布checkout。不要把临床依赖全局注入训练。
输出变量用ANCHORREAD_ABLATION_RUNS_ROOT和ANCHORREAD_ABLATION_RESULTS_ROOT。

入口是 scripts/ablation/rg_a1_no_localread.sh、rg_a2_no_mean_bypass.sh、
rg_a3_global_only.sh、rg_a4_local_only.sh；VQA对应同名vqa_前缀。
每个默认顺序32/256、fresh训练→validation best→test；--ratios可只跑一个倍率。
RG full test/all metrics，VQA test/normal/bootstrap1000。用独立run-tag和输出根。
不要复用旧m06–m13的任务数组和硬编码输出路径；单独准备本机Slurm包装器，
保留调度器CUDA_VISIBLE_DEVICES，account/partition/QoS/资源按本机核实，
先一个GPU作业内顺序运行，不直接照搬旧H100四进程并发设置。

先用本机解释器运行 scripts/ablation/preflight.py --output-dir <新目录>
--check-vqa-data，检查16组：RG32 M275 G67/L208、RG256 M35 G7/L28；
VQA32 M288 G72/L216、VQA256 M36 G8/L28（这些G/L是A1/A2；A3全G、A4全L）。
核对实际N分别8775/9555、共享配方、任务数据、独立输出及fresh初始化。
运行 tests/test_anchorread3d_ablation.py、tests/test_anchorread3d_results.py；
training测试的随机特征完整宽度部分不依赖真实权重，小backbone部分需要本机
tokenizer和默认相对布局manifest，fixture不会自动读取shell的路径overlay。
源机的698项通过和CPU测试不是Marlowe本机/GPU验证，汇报实际完成项。

最终给出发布SHA、本机路径映射、16组dry-run/数据检查和测试结果、独立Slurm
启动命令、输出和恢复位置，以及缺失项。汇总用scripts/ablation/collect_results.py，
显式传--runs-root与--output-dir，保留run-tag/scope/coverage，不改主实验Excel。
发现缺依赖、路径或校验问题，定位原因并只做必要的本机适配，保持本次算法和协议。
```
