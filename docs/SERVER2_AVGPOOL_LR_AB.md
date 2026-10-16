# 给另一台服务器 Codex 的执行指示：同步代码并运行 AvgPool3D 学习率对照

目标：把 GitHub `main` 中的共享 pipeline 代码同步到本机，保留本机数据、模型、缓存和输出路径，然后独立运行 AvgPool3D **8×** 的两组 `joint_lora` 训练。两组只改变第二段 projector 日程；两者的 cosine 尾段 LR 下限均为峰值的 10%。不恢复旧 checkpoint，不覆盖旧运行目录。

## 1. 合并代码，保留本机路径

先在本机仓库根目录查看 `git status --short`、`git diff`、`git log -1`。检查现有本机路径修改，尤其是 `configs/base.yaml`、脚本中的 Python 环境路径、数据/模型路径。如果有未提交改动，先用本地分支或 stash 安全保留，不能直接 `reset --hard` 或 `clean`。把本机资产路径迁移到 Git 忽略的 `configs/local_paths.yaml`，只放路径字段；`project_root: null` 会自动指向当前 checkout。

本机路径覆盖文件的结构示例（请填本机真实位置，不要照抄占位路径）：

```yaml
data:
  root: /local/path/to/data
  annotation_root: /local/path/to/RadGenome-Brain
  manifest_dir: /local/path/to/training_manifests/single_sequence_findings_coref_v2
encoder:
  path: /local/path/to/neurovfm-encoder
decoder:
  path: /local/path/to/llama3.1-8b-instruct
preprocessing:
  cache_dir: /local/path/to/preprocessing-cache
```

本机非路径代码改动也要先审阅和保留，不能默默丢弃。工作树达到可安全快进状态后执行：

```bash
git fetch origin main
git pull --ff-only origin main
git rev-parse HEAD
```

如果本机已有独立提交或 `pull --ff-only` 无法完成，先查清分叉并将本机改动在分支上保留，再审阅合并；不要用强制推送或硬重置解决。同步后确认 `src/training/schedule.py` 支持 `cosine_min_lr_ratio` 与 `joint_warmup_projector_mode`，`src/cli/baseline_sweep.py` 支持 `--config`。`configs/local_paths.yaml` 不应进入 Git。`assets/`、`runs/`、权重和原始数据也不会由 GitHub 下载。

## 2. 运行两个独立实验

激活本机训练环境，设置 `PROJECTOR_PYTHON="$(command -v python)"`，并在确认 GPU 空闲后显式设置 `CUDA_VISIBLE_DEVICES` 为所选物理 GPU 编号。脚本内部使用可见设备的逻辑 `cuda:0`。如果本机资产正好在仓库默认相对路径，可从以下命令中去掉 `--config configs/local_paths.yaml`。如输出需存到本机其他磁盘，两条命令都加同一个 `--output-root /本机/输出目录`。

先对两条命令各加 `--dry-run` 查看计划：应为 `projector=avgpool3d`、`ratio=8`、相同样本划分与 10 epoch 预算。预期现有 BraTS 划分为 train 1,960、val 268、总更新 1,230、两段 warm-up 各 77 步；若本机计划不同，先核对 manifest、配置和样本 ID，不要直接开始训练。预检完成后**去掉 `--dry-run`，真正运行**：

```bash
bash scripts/baselines/run_avgpool3d.sh \
  --ratios 8 \
  --run-tag avgpool_r8_lr_floor10_v1 \
  --config configs/local_paths.yaml \
  --eval-split val --metrics answer_ce \
  --set training.cosine_min_lr_ratio=0.1 \
  --set training.joint_warmup_projector_mode=restart

bash scripts/baselines/run_avgpool3d.sh \
  --ratios 8 \
  --run-tag avgpool_r8_lr_floor10_hold_peak_v1 \
  --config configs/local_paths.yaml \
  --eval-split val --metrics answer_ce \
  --set training.cosine_min_lr_ratio=0.1 \
  --set training.joint_warmup_projector_mode=hold_peak
```

两条命令各自生成独立运行目录并在训练后评测 best checkpoint。可在 tmux 中顺序执行；若并行，给每条命令单独指定空闲 GPU，且确保两卡类型/训练条件可比。运行名已经存在时换新 tag，不要用另一个方案的 checkpoint 续训。

## 3. 核对并汇报

对每组查看 `runs/baselines/<tag>/sweep_plan.json`、`r8/training_config.yaml`、`r8/joint_lora/schedule.json`、`r8/joint_lora/train.jsonl`、`r8/joint_lora/validation.jsonl`、`r8/evaluation/best_val/metric.json` 和 `sweep_status.json`。检查除两个调度字段和输出路径外，模型、数据选择、seed、batch size、峰值 LR 与步数一致。

第 78 步：`restart` 的 projector LR 为 0，`hold_peak` 为 `1e-5`，两组视觉 LoRA 都从 0 开始 warm-up。第 1,230 步两组 LR 均应为 `1e-6`。报告每组完成状态、最低验证 CE 及 epoch、训练/验证 CE 曲线和上述 LR 核对结果。选择方案用验证集，不用测试集分数挑参。
