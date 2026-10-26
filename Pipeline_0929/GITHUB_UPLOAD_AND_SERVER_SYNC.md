# GitHub 上传与两台服务器同步流程

> 最新发布：`sync-20261026-r128-r256`（128×／256× 扩展），本轮同步请使用文末第十节。下文原 `sync-20261026` 是上一次发布，保留为历史记录及通用环境/资产核对流程。

本轮日期：2026-10-26。源仓库：`/mnt/sda/jyzhang/VLM_Projector`。
GitHub 私有仓库：[TonyZhang-sjtu/VLM_Projector](https://github.com/TonyZhang-sjtu/VLM_Projector)，分支 `main`，本轮固定发布标签 `sync-20261026`。
最终提交 SHA 以源服务器发布回复及远端该标签的解析结果为准；不要把后续变化的 `main` 当成本轮固定版本。

用户已明确：同步的是**从头运行时的代码、调用逻辑、配置、依赖、输入和评测协议**；不迁移、不复用现有 checkpoint；显卡造成的数值差异不作为不一致。不能因为路径不同而改实验算法、随机种子、数据划分、损失、调度或评测范围。

本流程沿用 [环境与迁移说明](../docs/ENVIRONMENT_AND_MIGRATION.md) 与 [上轮同步方案](../docs/SERVER2_SYNC_64X_20261021.md)，本文件为本轮入口。旧文档中的历史测试数量不是本轮测试结果。

另已找到源机本地的首次 GitHub 上传验收记录：`runs/archive/engineering/github_setup_20261013T045129Z/verification.json` 与 `staged_audit.json`，以及上轮 `.tmp/sync64_*` 验证日志。首次记录确认私有仓库、逐项暂存审阅、路径迁移预检和上传后远端 SHA 核对；这些历史产物继续留在源机，不上传为本轮结果。

## 一、可直接交给目标服务器代码助手的 prompt

复制下面整个代码块。将 `RELEASE_COMMIT` 替换为源服务器最终回复的完整 SHA；目标服务器的仓库位置由它自行检查，禁止套用源服务器绝对路径。

```text
请将本机 VLM_Projector 同步到 GitHub 私有仓库
https://github.com/TonyZhang-sjtu/VLM_Projector.git
的固定发布版本：标签 sync-20261026，RELEASE_COMMIT=<源服务器给出的完整SHA>。

目标：从头运行的代码、调用链、数据、模型、知识库、环境、训练与评测参数和源服务器一致；允许本机路径、物理 GPU 编号、输出目录和显卡引起的数值差异。不要迁移或复用已有 checkpoint，不自动启动正式长训练。

1. 先定位本机仓库，检查 AGENTS.md、git status、HEAD、远端和正在运行的任务。
   备份本机代码差异、未追踪文件、configs/local_paths.yaml 或 PROJECTOR_LOCAL_CONFIG
   指定的文件到仓库外，记录其 SHA256、Python 路径、输出位置和 GPU 设置。
   不覆盖本地资产或实验；不 reset --hard、不 clean、不 force push、不终止任务。
   有运行中任务或本地代码分叉时，优先建固定 SHA 的独立 worktree。
2. fetch main 和 sync-20261026，核对标签解析的 SHA 等于 RELEASE_COMMIT；
   最终运行 checkout 的 HEAD 必须等于这个 SHA，不能仅检查“包含这个提交”。
   阅读该版本 Pipeline_0929/GITHUB_UPLOAD_AND_SERVER_SYNC.md，并按后续步骤完成核对。
3. 使用本机 projector 环境，显式设置 PROJECTOR_PYTHON、PROJECTOR_LOCAL_CONFIG、
   PROJECTOR_OUTPUT_ROOT、CUDA_VISIBLE_DEVICES、PYTHONNOUSERSITE=1。
   保留本机配置内容；在独立 checkout 中创建单独的路径 overlay。
   不从源机复制 local_paths.yaml、虚拟环境、CUDA 编译二进制或旧运行配置。
4. 依照文档核对全部训练依赖版本及实际导入源码、独立临床评测 overlay、模型权重、
   tokenizer、BraTS MRI、三个 manifest、BindFormer v2 知识库，以及
   report_facts_v1 / report_output_v1 两套辅助数据的完整内容哈希。
   Git 不含这些资产。缺少或哈希不同就明确报告；通过已授权服务器间传输补齐，
   不下载其他版本代替，不重建/修改已有标注，不绕过校验。
5. 跑 CPU 回归和四个 Exp1024 队列的 --dry-run，并用 check_sync_plans.py
   对照 SYNC_PLANS_20261026.json 核验全部 12 个实验。检查训练和评测都走相应
   entry.py；检查本机实际导入的 src/experiments 来自这个固定 checkout。
   检查 32×/64×=275/138 tokens，train/val/test=1960/268/560，
   seed=42、deterministic=true、cosine 下限=0.3、fresh 初始化、best-CE full test/all。
6. 验证路径配置备份未被修改，确认 Git 追踪文件没有本地实现差异。
   不恢复旧 src/ 或旧实验入口覆盖新版本；只保留必要的本机路径配置。
7. 最后汇报：实际完整 SHA、路径映射、环境/资产/12 份计划比较结果、测试结果、
   训练与评测实际入口、缺失项，以及本机可直接使用的正式启动命令。
   某项未完成就标记未完成，不能用 CPU 测试或 dry-run 代替资产/依赖核对。
   只有全部一致后才能称为“运行环境与逻辑已同步”；本次不启动正式长训练。
```

## 二、本轮上传范围和明确排除项

上传：`src/training/loss.py`、`src/training/trainer.py` 的可选辅助目标支持；`experiments/report_facts_v1/`、`experiments/report_output_v1/` 的完整运行源码；Exp1020/1023/1024 入口；相关测试、研究说明和本同步文件；同步核对工具及不含病例明细的参考摘要。

用户已确认上传两个旧 Exp1020 入口的当前默认值：fixed-region mean 默认 GPU 0 / 64×，adaptive 默认 GPU 1 / 64×。目标机显式用 `CUDA_VISIBLE_DEVICES` 覆盖物理卡，普通入口可用 `--ratios` 覆盖倍率；Exp1024 队列由自己的固定数组决定倍率。

不上传：`configs/local*.yaml`、凭据、`assets/`、`runs/`、缓存、checkpoint、预测报告、实验结果、虚拟环境、third-party checkout、编译二进制。既有 `.gitignore` 继续生效。`results/BindFormer_role_value_adapter_r16/` 的 `_launch.bash`、`entry.py`、`runtime.py`、`reader_adapter.py` 是上一轮已经追踪的四份运行源码，必须保留；该目录其余结果仍不进 Git。

三套知识库/辅助标注含派生报告信息，不能因为体积小就上传。本轮参考 JSON 只记录配置、总量、版本和聚合哈希，不包含病人 ID、原始报告或 MRI 文件列表。

## 三、固定代码版本并保护本机配置

先记录、备份本机状态。不要自动 stash 全部未追踪文件，里面可能有大量数据。

```bash
git status --short --untracked-files=all
git rev-parse HEAD
git remote -v
# 对实际本机配置执行，并把备份放在仓库之外。
sha256sum "$PROJECTOR_LOCAL_CONFIG"
```

备份后获取本轮版本：

```bash
RELEASE_COMMIT='<源服务器最终回复的完整SHA>'
git fetch origin main tag sync-20261026
test "$(git rev-parse 'refs/tags/sync-20261026^{commit}')" = "$RELEASE_COMMIT"
```

工作区干净、无任务依赖它且本地 `main` 可快进到**恰好该 SHA**时，可 `git merge --ff-only "$RELEASE_COMMIT"`，然后用 `test "$(git rev-parse HEAD)" = "$RELEASE_COMMIT"` 核对。若远端 `main` 已前进、存在本机修改/分叉或有任务运行，保留原 checkout，使用独立目录：

```bash
# RELEASE_CHECKOUT 必须是本机尚不存在的新目录。
RELEASE_CHECKOUT='<本机新目录>/VLM_Projector-sync-20261026'
git worktree add --detach "$RELEASE_CHECKOUT" "$RELEASE_COMMIT"
cd "$RELEASE_CHECKOUT"
test "$(git rev-parse HEAD)" = "$RELEASE_COMMIT"
git diff --exit-code HEAD --
```

worktree 不包含被忽略的资产和本地依赖。将本机已有资产链接到新 checkout 的对应位置，或使用单独的 local overlay。保留原工作区供原任务继续使用。不要把旧 `src/`、`experiments/`、`scripts/` 复制进新 checkout。

## 四、两台服务器的路径分工

| 项目 | 源服务器 | 目标服务器要求 |
|---|---|---|
| 代码根目录 | `/mnt/sda/jyzhang/VLM_Projector` | 自己的固定 SHA checkout |
| Python | `/mnt/sda/mingjie/conda_envs/projector/bin/python` | 本机等版本 projector 解释器 |
| 路径配置 | `configs/local_paths.yaml`，Git 忽略 | 保留本机文件；新 worktree 用单独 overlay |
| data/annotation/manifest | 源机 `assets/data/` 下对应目录 | 内容相同、根路径可不同 |
| encoder/decoder | 源机 `assets/models/` | 内容相同，配置映射到本机路径 |
| BindFormer/辅助标注 | `assets/knowledge/` 下三套目录 | 相同内容，建议保留仓库内相对目录布局 |
| 预处理/feature 缓存 | `assets/preprocessing/`、`cache/features` | 可以重建，不跨机复制旧缓存作为一致性证据 |
| 临床评测 | `assets/models/evaluation/`、`third_party/evaluation/python_deps/` | 相同原始资产和评测依赖，使用本机路径 |
| 输出 | `runs/Exp1024/` | 新目录/新 run tag，不覆盖已有实验 |

激活本机已核验环境后：

```bash
export PROJECTOR_PYTHON="$(command -v python)"
export PROJECTOR_LOCAL_CONFIG="$PWD/configs/local_paths.yaml"
export PROJECTOR_OUTPUT_ROOT="$PWD/runs/Exp1024"
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PROJECTOR_SEED=42 COSINE_MIN_LR_RATIO=0.3
# dry-run 可使用任一合法编号；正式运行前按本机空闲卡重新设置。
export CUDA_VISIBLE_DEVICES=0
test -x "$PROJECTOR_PYTHON"
test -f "$PROJECTOR_LOCAL_CONFIG"
git check-ignore "$PROJECTOR_LOCAL_CONFIG"
```

外置配置不属于仓库时，不用 `git check-ignore`，而是确认它未被追踪。检查 overlay 只改路径；特别是 `project_root` 不能仍指向旧代码目录。`src.config` 中 `project_root: null` 会解析为当前 checkout。

Exp1024 的 `_common.bash` 保留源机 Python 默认路径，因此**目标机必须显式 export PROJECTOR_PYTHON**，不能只依赖激活 Conda。所有正式入口在内部切换到其代码根目录；目标 shell 不应通过旧 `PYTHONPATH` 注入另一份源码。

### 知识库与辅助标注路径的特殊约束

Exp1024 在子进程中设置 `REPORT_FACTS_PATH=assets/knowledge/report_facts_v1` 和 `REPORT_OUTPUT_PATH=assets/knowledge/report_output_v1`；外层同名环境变量会被覆盖。最稳妥的方法是让新 checkout 下的三个相对目录链接到本机已校验资产：

```text
assets/knowledge/bindformer_brain_candidates_v2
assets/knowledge/report_facts_v1
assets/knowledge/report_output_v1
```

如果坚持用其他绝对路径，应先审阅对应最终计划。普通 E/R 入口支持相应环境变量和 `--set`；Exp1024 的混合队列不宜全局塞入只属于 E 或 R 的额外 projector 配置，应保留上述相对布局。不要修改算法脚本来适配路径。

独立 worktree 的临床评测还需要 `third_party/evaluation/python_deps`，以及必要的锁定源码。可以链接到本机已校验且不修改的依赖目录；仅有训练 Conda 环境不足以运行 `metrics=all`。RadGraph runtime 配置、HF hub/modules 缓存由本机从已验证原始资产生成，不能用源机的绝对路径覆盖目标机。

## 五、资产内容核对与单独传输

参考：[SYNC_ASSETS_20261026.json](SYNC_ASSETS_20261026.json)。工具读取实际文件内容计算 SHA256，而非仅检查名称、大小或时间戳。目录可以位于不同根路径，但目录内部布局必须相同。

```bash
SYNC_CHECK_DIR="$PWD/.tmp/sync_$(date -u +%Y%m%dT%H%M%S%NZ)"
export SYNC_CHECK_DIR
mkdir -p "$SYNC_CHECK_DIR"
"$PROJECTOR_PYTHON" scripts/sync_asset_fingerprint.py \
  --output "$SYNC_CHECK_DIR/assets.json" \
  --reference Pipeline_0929/SYNC_ASSETS_20261026.json
```

默认核对 8 组：manifests、brats_images、encoder、decoder、bindformer、report_facts、report_output、evaluation。数据/模型根目录不在默认位置时，按实际 resolved config 添加，例如：

```text
--map manifests=/本机/manifest目录
--map brats_images=/本机/data/BraTS
--map encoder=/本机/neurovfm-encoder
--map decoder=/本机/llama3.1-8b-instruct
--map bindformer=/本机/bindformer_brain_candidates_v2
--map report_facts=/本机/report_facts_v1
--map report_output=/本机/report_output_v1
--map evaluation=/本机/evaluation
```

映射必须对应**入口实际使用的目录**，不能核对另一套闲置资产后称运行输入一致。可用 `--groups` 分批核对，最终必须覆盖 8 组。BraTS 组包含源机目录内全部 `.nii/.nii.gz`，因此目标机额外或缺少影像都会使聚合哈希不同；调查布局/子集差异，不应直接删数据。主训练实际仍只读取 manifest 选中的样本。

评测组排除可重建的 `hub/`、`modules/`、`radgraph/runtime/`、`radgraph/radgraph-xl/`，保留原始 RadGraph archive、权重、tokenizer、资产 manifest 等。其他组排除下载缓存、锁、日志、Python 缓存等。参考值不是对目标机器已完成验证的声明。

需要补齐的资产沿用授权的服务器间 SSH/rsync 通道，不进 GitHub。目标机先确认来源及自己的目录；示例中的主机和目标路径必须替换为实际值：

```bash
# 先预览到一个尚不存在/空的版本目录；不要对现有目录使用 --delete。
rsync -aLnc --itemize-changes \
  '<源机SSH别名>:/mnt/sda/jyzhang/VLM_Projector/assets/knowledge/report_facts_v1/' \
  '<本机新资产目录>/report_facts_v1/'
# 审阅差异后用相同参数去掉 n，实际传输。
rsync -aLc --itemize-changes \
  '<源机SSH别名>:/mnt/sda/jyzhang/VLM_Projector/assets/knowledge/report_facts_v1/' \
  '<本机新资产目录>/report_facts_v1/'
```

同理传输 report_output、BindFormer v2、manifest 和缺少的模型/数据。保留辅助数据及其 metadata 原始字节；不要重建 sidecar、修改规则或换原始 HF 标注作为“同步”。`report_output_v1/build.py` 的默认构建输入还依赖本地诊断结果，Git 本身不能重建这份输入。凭据文件不在任何资产传输列表中。当前没有提供目标 SSH 地址，本轮不连接或修改第二台服务器。

## 六、环境与实际导入源码核对

当前源机训练环境 Python 3.10.14；全部 154 个 distribution 与 `environment-lock.json` 一致。核心为 Torch 2.5.0+cu124、Transformers 4.51.3、PEFT 0.15.2、FlashAttention 2.7.4.post1、NumPy 2.1.2。源 GPU 为 RTX A5500，驱动 560.35.03，Torch CUDA 12.4 / cuDNN 90100；目标 GPU 和兼容驱动可不同。

先核对训练环境（不安装、不升级）：

```bash
"$PROJECTOR_PYTHON" -m pip check
"$PROJECTOR_PYTHON" scripts/setup_sources.py --check-only
"$PROJECTOR_PYTHON" - <<'PY'
import importlib.metadata as metadata
import importlib.util
import json, platform, site
from pathlib import Path
from src.evaluation.clinical_runtime import package_digest
lock = json.loads(Path('environment-lock.json').read_text())
actual = {d.metadata['Name'].lower().replace('_', '-'): d.version for d in metadata.distributions()}
diff = {k: [v, actual.get(k)] for k, v in lock['packages'].items() if actual.get(k) != v}
assert not diff, diff
assert platform.python_version() == lock['python']
assert site.ENABLE_USER_SITE is False
reference = json.loads(Path('Pipeline_0929/SYNC_RUNTIME_20261026.json').read_text())
source_diff = [n for n, sha in reference['source_digests'].items() if package_digest(n) != sha]
assert not source_diff, source_diff
for name in ('src', 'experiments'):
    origin = Path(importlib.util.find_spec(name).origin).resolve()
    assert origin == Path(name, '__init__.py').resolve(), (name, origin)
    print(name, origin)
print('Training versions and runtime Python sources match.')
PY
```

`setup_sources.py --check-only` 检查 NeuroVFM/FlashAttention 的锁定 commit 和已追踪源码改动；`SYNC_RUNTIME_20261026.json` 另外检查实际导入 NeuroVFM、FlashAttention、Torch、Transformers、PEFT 等 Python 源码哈希，避免“版本号相同但实际代码被改过”。CUDA 扩展按本机 GPU 编译，不直接搬源机 `.so`。仅在缺少环境时使用 `docs/ENVIRONMENT_AND_MIGRATION.md` 的安装流程。

临床指标使用独立进程激活 overlay，不能在已经导入 NumPy 的训练进程里激活。源机该 overlay 的 NumPy 为 **1.26.4**，与训练环境有意不同。

```bash
"$PROJECTOR_PYTHON" - <<'PY'
import json, platform
from pathlib import Path
from src.evaluation.clinical_runtime import activate, environment, package_digest
activate()
actual = {'environment': environment(['green', 'radgraph', 'rate', 'bertscore'])}
actual['environment'].pop('executable')
actual['environment']['python'] = platform.python_version()
actual['source_digests'] = {n: package_digest(n) for n in ('green_score', 'radgraph', 'RaTEScore', 'bert_score')}
expected = json.loads(Path('Pipeline_0929/SYNC_CLINICAL_20261026.json').read_text())
assert actual == expected, 'Clinical overlay versions or source differ; investigate before full evaluation'
print('Clinical dependencies and scorer sources match.')
PY
```

## 七、CPU 回归、完整计划及调用链验收

以下测试不启动正式训练。`PROJECTOR_TEST_DEVICE` 必须 unset，不能设置为空字符串。

```bash
env -u PROJECTOR_TEST_DEVICE CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  "$PROJECTOR_PYTHON" -m pytest -q --ignore=tests/test_vision.py \
  -k 'not test_launcher_dry_run_overrides_without_creating_run' \
  > "$SYNC_CHECK_DIR/pytest.log" 2>&1
```

这里排除的 `test_vision.py` 是真实 GPU 验收；被 deselect 的旧 MLP 入口测试依赖另一机器的绝对路径。部分测试依赖可选历史浮点快照/本地资产或 CUDA，会 skip；目标机应报告实际数量与原因，不为追求数量而删除测试或放宽断言。训练环境验证必须对实际 checkout 的代码进行。

生成四个完整队列的计划：

```bash
for n in 1 2 3 4; do
  bash "scripts/ft_ablation/Exp1024/run_1024_${n}.sh" --dry-run \
    > "$SYNC_CHECK_DIR/queue${n}.jsons" 2> "$SYNC_CHECK_DIR/queue${n}.stderr" || exit
done
"$PROJECTOR_PYTHON" scripts/check_sync_plans.py \
  --plans "$SYNC_CHECK_DIR" --output "$SYNC_CHECK_DIR/plans.normalized.json" \
  --reference Pipeline_0929/SYNC_PLANS_20261026.json
```

工具逐项比较全部 12 个实验：仅规范化明列的本机路径和 Python 可执行文件位置，保留训练参数、算法开关、budget、manifest SHA、样本 ID 顺序摘要、调度、训练与评测命令、评测范围。比较成功不替代第六节对实际解释器和导入路径的检查，也不替代第五节资产检查。

| 队列 | 顺序（每次独立初始化，训练完成后 full test） |
|---|---|
| run_1024_1 | Baseline A 32× → 64× → R2 64× |
| run_1024_2 | role value adapter 32× → 64× → E3 64× |
| run_1024_3 | R1 32× → 64× → R2 32× |
| run_1024_4 | E2 32× → 64× → E3 32× |

共同参数：seed=42、deterministic=true、projector/encoder LR=2e-4、effective batch/microbatch=16/1、warmup/hold/joint warmup=77/0/77、hold_peak、cosine_min_lr_ratio=0.3、最多20 epochs、123 updates/epoch、early stop patience=3/min_delta=0.005；32×/64×实际为275/138 visual tokens；train/val/test=1960/268/560。`init_checkpoint`、`resume`、训练/验证 limit、max_steps 应为 null；评测使用本次新训练的最低 validation CE `best.pt`，全560条 test、metrics=all。R1/R2 的 `best_facts` 另存，不替换主评测 checkpoint。

调用链必须保留：

| 方法 | 训练和评测子进程入口 |
|---|---|
| Baseline A | 当前 checkout 的 `python -m src.cli.*` |
| role value adapter | `results/BindFormer_role_value_adapter_r16/entry.py` |
| E2 / E3 | `experiments/report_facts_v1/entry.py` |
| R1 / R2 | `experiments/report_output_v1/entry.py`；外层经 `python_entry.sh` 桥接 |

不要把 E/R 方法改成普通 `python -m src.cli.train`，否则会丢失 opt-in runtime。`python_entry.sh` 必须保留 Git 中的可执行位。不要因源机旧目录中仍有历史脚本，就把未追踪运行代码加入新版本。

检查全部通过后，目标机给出下面形式的正式命令供用户使用；本次同步流程不执行它们：

```bash
# 每个队列放在独立 terminal/tmux；按本机空闲卡替换编号。
CUDA_VISIBLE_DEVICES=0 bash scripts/ft_ablation/Exp1024/run_1024_1.sh
CUDA_VISIBLE_DEVICES=1 bash scripts/ft_ablation/Exp1024/run_1024_2.sh
CUDA_VISIBLE_DEVICES=2 bash scripts/ft_ablation/Exp1024/run_1024_3.sh
CUDA_VISIBLE_DEVICES=3 bash scripts/ft_ablation/Exp1024/run_1024_4.sh
```

这些命令必须继承前面设置的本机 Python、路径 overlay、seed/cosine 和新输出目录。只有少量 GPU 时可以串行调度，不改变各实验训练配置。

## 八、源服务器以后每次上传的步骤

1. 检查本机 AGENTS、`git status --short --untracked-files=all`、`git diff`、远端和 HEAD；确认代码变更与正在运行的任务不冲突。
2. 按 `.gitignore` 和本文件明确范围，检查未追踪目录是否含运行必需源码；不确定是否上传的内容先问用户。本次两个旧入口默认值已明确批准。
3. 用实际训练 Python 核对环境；运行与改动有关的回归、`bash -n`、真实 manifest 的 `--dry-run`。从待发布文件导出的隔离目录复验，避免本机未追踪代码让测试假通过。
4. 若数据/知识库/依赖/实验配方有变化，生成新日期的资产、runtime、clinical、plan 参考文件，不能直接沿用旧哈希。数据细节仍不进 Git。
5. `git add` 明列文件或已审阅源码目录；不要使用 `git add .` 或 `git add -f`。审阅 `git diff --cached --stat`、`git diff --cached --check`，检查凭据、数据、模型、结果是否误入暂存区。
6. 创建本地 commit。为本轮建立新的、不复用旧名称的固定同步标签。通过既有 Git 认证上传，不把 token 写入 remote URL 或文档。若远端前进，先审阅差异重新验证，不强推。

   ```bash
   git commit -m '<本轮具体变更>'
   # 每轮使用不同标签，下面是本轮已经占用的名称。
   git tag sync-20261026 HEAD
   git push --atomic origin main refs/tags/sync-20261026
   git rev-parse HEAD
   git ls-remote origin refs/heads/main refs/tags/sync-20261026
   ```

7. 发布后确认远端 main/标签对应刚验证的 SHA，工作区无漏传的代码改动。给目标机完整 SHA、本文 prompt 和需独立同步的资产清单。保留验证日志在 `.tmp/`，不把日志/结果混进 Git。
8. 目标机按本文执行并回报；在收到目标机结果前，只能说明“源机发布和本机验证完成”，不能宣称第二台服务器已验证一致。

## 九、本轮源机验证记录

- 基线提交：`39aed117ca4459167141b23c994614421db5d94a`。
- 训练环境 154 个 distribution 与现有锁一致；`pip check`、锁定 NeuroVFM/FlashAttention 源码检查通过；记录实际训练和临床评测源码摘要。
- 相关回归：96 passed / 1 skipped；隔离源码副本完整 CPU 回归：628 passed / 85 skipped / 1 deselected；另有4项同步工具测试通过。GPU 真实模型测试不属于本轮执行范围。
- 本机四个 Exp1024 队列、全部12个真实数据 dry-run通过。另在不同根路径的隔离源码副本（不携带源机 local_paths.yaml）重新完成12个计划；规范化本机路径后，配置、样本身份、训练与评测命令全部与源机参考完全一致。
- 资产参考对原始内容做完整 SHA256 扫描，包括全部 BraTS MRI、模型分片、三套知识资产、manifest 与临床模型；不对原始资产写入或修改。
- 暂存区扫描无凭据、患者明细或大文件误入；20份 shell 语法检查通过。标准 `git diff --cached --check` 仅报告新增 `experiments/report_output_v1/schema.py` 末尾已有空行，保留当前源码字节；排除 blank-at-eof 提示后的空白检查通过。
- 源机验证日志和隔离副本：`.tmp/github_sync_20261026/`（Git 忽略）。
- 第二台服务器未在本轮被连接或执行命令，需按 prompt 完成验收；显卡数值差异按用户要求不作为失败条件。


## 十、2026-10-26 后续发布：128×／256×

固定标签：`sync-20261026-r128-r256`。父提交：`1eb140c2cd496dacb9bec19ba9209c12e6d9b72f`。完整发布 SHA 以本轮回复和标签解析结果为准；不要再用前文旧标签同步这次扩展。

本轮上传 `src/models/projectors/budget.py`、`src/cli/baseline_sweep.py`、相关测试与说明文档；新增128×／256×预算，原倍率默认数组和训练/评测流程保持原样。设计与资源约束见 [扩展实现记录](PROJECTOR_R128_R256_IMPLEMENTATION.md)。当前打开的 `scripts/baselines/run_local_packer3d.sh` 本身没有新增改动，它通过共享预算实现支持新倍率。

目标机仓库已知位置为 `/home/mingjie/jyzhang/VLM_Projector`。先检查本地修改、备份本机路径配置；有任务正在使用当前源码时使用独立 worktree，参照第三节。工作区可安全快进时：

```bash
cd /home/mingjie/jyzhang/VLM_Projector
git status --short --untracked-files=all
# 确认无需要保留/整理的代码改动，且没有任务依赖待更新的当前 checkout 后执行。
git fetch origin main tag sync-20261026-r128-r256
git merge --ff-only refs/tags/sync-20261026-r128-r256
test "$(git rev-parse HEAD)" = "$(git rev-parse 'refs/tags/sync-20261026-r128-r256^{commit}')"
git rev-parse HEAD
```

若不能快进或 SHA 不同，停止该工作区的后续运行，按第三节创建固定标签的独立 worktree，保留本机改动；不要 `reset --hard` 或强推。上面的 fetch/merge/test 任一步失败都不能继续假定同步成功。

使用本机已核对的 projector 环境和本机路径，做最小验收：

```bash
conda activate projector
export PROJECTOR_PYTHON="$(command -v python)"
export PROJECTOR_LOCAL_CONFIG="$PWD/configs/local_paths.yaml"
export PYTHONNOUSERSITE=1
test -x "$PROJECTOR_PYTHON"
test -f "$PROJECTOR_LOCAL_CONFIG"
env -u PROJECTOR_TEST_DEVICE CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2   "$PROJECTOR_PYTHON" -m pytest -q tests/test_projector_budgets.py
CUDA_VISIBLE_DEVICES=0 bash scripts/baselines/run_local_packer3d.sh   --config "$PROJECTOR_LOCAL_CONFIG" --ratios 128 256 --dry-run
```

注意：该旧 baseline shell **不会自动读取 `PROJECTOR_LOCAL_CONFIG`**，必须像上面一样显式传 `--config`；`PROJECTOR_PYTHON` 也必须指向本机解释器，以覆盖源机默认绝对路径。本地配置文件若放在其他位置，将环境变量改为本机实际位置。

LocalPacker 预期128×／256×分别输出72／36 tokens，train/val/test=1960/268/560，保持完整 test 评测；默认倍率仍为4/8/16/32，要运行新倍率必须传 `--ratios 128 256`。上述 dry-run 不启动训练。资产和环境继续按第五、六节核对；Git 更新不传数据、模型、知识库、checkpoint 或本机配置。无需迁移/复用旧 checkpoint。

本次发布前实际验证：隔离源码副本执行 budget、64×兼容、几何、baseline/BindFormer sweep、契约与启动器回归，**314 passed / 16 skipped**（16项为显式GPU opt-in测试）；源目录及不带本机 local_paths.yaml 的另一根路径源码副本各完成 LocalPacker 128×／256×真实数据预检。规范化本机路径后，两份配置、预算、样本身份、调度及训练/评测命令全部一致。未启动正式训练，也没有连接第二台服务器执行命令。日志保留在源机 `.tmp/github_sync_r128_r256_20261026/`。
