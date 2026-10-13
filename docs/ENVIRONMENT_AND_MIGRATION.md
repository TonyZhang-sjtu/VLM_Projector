# 环境记录与迁移

本次核对使用训练入口同样的 `PYTHONNOUSERSITE=1`，避免将 `~/.local` 的其他包混入。
旧 `environments.yaml` 中已列出的 Conda/pip 版本与当前隔离环境一致；缺少后来安装的
`neurovlm-projector-research` editable 项目记录，并带有本机源码、wheel 和 `PYTHONPATH` 绝对路径。

当前文件分工：

- `environments.yaml`：当前 Conda 和常规 pip 版本，环境名 `projector`，无本机绝对路径。
- `environment-lock.json`：当前隔离环境全部 154 个 Python distribution 的实测版本，包含下面四项分阶段安装的包。
- `third_party/sources.lock.json`：NeuroVFM、3DINO、FlashAttention、GREEN、RadGraph、RaTEScore 的官方源码地址和当前 commit。
- `scripts/setup_projector.sh`：Torch 安装完成后，安装 FlashAttention、编译 fused-dense、安装 NeuroVFM 和项目本身。
- `requirements-clinical.txt` 与 `scripts/setup_clinical_metrics.py`：临床指标依赖的独立 overlay，不能合并到主环境。

核心版本：Python 3.10.14、Torch 2.5.0+cu124、Transformers 4.51.3、PEFT 0.15.2、
Accelerate 1.7.0、NumPy 2.1.2、FlashAttention 2.7.4.post1、fused-dense-lib 0.0.0。
本次只核对并更新环境记录，没有安装、升级或删除当前训练环境中的包；
没有在空环境重新安装整套依赖，因此不把新服务器的完整安装或 GPU 运行标为已验证。

## 新服务器安装

需要 Linux x86_64、支持 BF16/FlashAttention 的 NVIDIA GPU、兼容 CUDA 12.4 的驱动。
fused-dense 需要本机可用的 CUDA toolkit（含 `nvcc`）和 C++ 编译器；Conda 文件中的
CUDA runtime 包不能代替编译工具链。跨 GPU 不复用原 A5500 的 sm_86 专用二进制。

```bash
git clone https://github.com/TonyZhang-sjtu/VLM_Projector.git
cd VLM_Projector
conda env create -f environments.yaml
conda activate projector
export PYTHONNOUSERSITE=1
export PROJECTOR_PYTHON="$(command -v python)"
# 若 toolkit 不在默认位置，先设置 CUDA_HOME 为实际安装目录。
bash scripts/setup_projector.sh

# 需要完整临床指标时，在独立目录安装依赖，不改训练环境的 NumPy。
python scripts/setup_clinical_metrics.py
python scripts/download_bertscore_assets.py
python scripts/download_clinical_assets.py
```

安装脚本使用锁定的官方 FlashAttention wheel，保留本机验证过的版本组合；该 wheel
限定 cp310、Torch 2.5、CUDA 12、旧 C++ ABI。NeuroVFM 的原 README 要求 2.6.3，
本项目使用 2.7.4.post1 的原因是当前 Torch 2.5 对应 wheel 与既有验证结果。
如果新 GPU/驱动与此组合不兼容，需要单独验证新组合，不能只改一个包就沿用旧验收结论。

可选：`python scripts/setup_sources.py --include-3dino` 获取 3DINO 官方源码；
使用官方 3DINO 脚本时临时设置 `PYTHONPATH="$PWD/third_party/3DINO${PYTHONPATH:+:$PYTHONPATH}"`。
当前主训练 pipeline 仍以 NeuroVFM 为 encoder，3DINO 训练适配器尚未实现。

## 必须单独迁移的资产

Git 只追踪代码、配置和文档。以下路径需要从已授权的数据与模型来源获取，或通过
服务器间的受控传输恢复；不上传 GitHub：

```text
assets/models/neurovfm-encoder/
assets/models/llama3.1-8b-instruct/
assets/models/3DINO-ViT/                     # 使用时再恢复
assets/models/evaluation/                   # 使用完整临床指标时需要
assets/data/BraTS/
assets/data/RadGenome-Brain/                 # 包含已修订标注、划分、审计及 manifest
  training_manifests/single_sequence_findings_coref_v2/
```

请复制当前已经审核的 `RadGenome-Brain` 整体，包括活动修订版本及原始备份，避免只下载
原始 HF 标注后误以为与既有实验相同。代码不会从 Git 中恢复这些患者级数据。
ISLES、WMH 不是当前 BraTS baseline 的必需输入。
`assets/preprocessing/`、`cache/` 可以重建；如复制缓存，应使用原内容哈希与一致配置。
旧实验 `runs/`、checkpoint、预测报告也需要另行迁移，Git 不包含它们。

`configs/base.yaml` 的 `project_root: null` 自动解析为新 checkout；新保存的运行配置
仍记录绝对路径。旧服务器生成的已保存运行配置不会自动重写，恢复旧实验前需要核对
其中的绝对路径、数据身份与恢复契约。启动脚本支持 `PROJECTOR_PYTHON`，新服务器应按上例
显式指向激活环境，避免使用旧服务器默认解释器路径。

## 启动前检查

```bash
python -m pip check
python scripts/setup_sources.py --check-only
python -m src.cli.config --config configs/training/joint_lora.yaml
CUDA_VISIBLE_DEVICES=0 bash scripts/baselines/run_token_select3d.sh \
  --ratios 8 --run-tag token_select_r8_server2 --dry-run
```

`--dry-run` 不加载 GPU 模型、不启动训练，会检查资产对应的清单与几何并输出计划。
实际运行去掉 `--dry-run`。运行目录已存在时入口拒绝覆盖，需换新的 `run-tag`。
当前 Bash 默认物理 GPU 仍保留原设置 3；使用外部 `CUDA_VISIBLE_DEVICES` 显式覆盖。
程序内部使用所选可见设备的逻辑 `cuda:0`。

## Git 追踪范围

提交 `src/`、`configs/`、`scripts/`、`tests/`、`Pipeline_0929/`、环境记录与文档。
`.gitignore` 排除数据/模型、token、结果、缓存、虚拟环境、文档备份、编译产物和 third-party checkout。
第三方依赖按锁定 commit 恢复，遵守各自原始许可证；尚未为本项目选择公开发布许可证。
`archive/non_pipeline/` 中的一次性工具、硬编码病例修订记录保留本地；已将通用评测依赖安装
与模型下载工具整理到 `scripts/`。研究文档中的 `runs/`、`assets/` 路径指本地证据，并非 GitHub 附件。

后续更新先 `git diff` / `git status` 检查，再 `git add` 指定代码文件、`git commit`、`git push`。
不要使用 `git add -f` 绕过数据与凭据排除规则。
