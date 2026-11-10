# Marlowe 集群任务入口

本目录为特定集群使用的 Slurm 任务编排脚本；提交任务前须检查 `.sbatch` 文件中的 account、partition、QoS、资源及时间限制是否符合当前集群。任务状态、日志、数据和 checkpoint 不包含在 Git 中。

`run_m06.sh` 至 `run_m13.sh` 会加载同目录下的 `env.sh`。该文件保存每台机器自己的设置，已由 Git 忽略，需要在目标机自行创建。至少要设置本机训练环境的 Python，例如：

```bash
# scripts/marlowe/env.sh：使用当前机器实际路径
export PROJECTOR_PYTHON="/本机/训练环境/bin/python"
export PYTHONNOUSERSITE=1
```

按集群要求补充 conda、缓存、临时目录和 CUDA 配置；模型与数据路径继续通过本机项目配置设置，不要复制其他服务器的绝对路径。Python 必须能够加载项目依赖。启动前审阅任务脚本及各训练入口的倍率、输出位置和评测范围，并先用各脚本支持的 `--dry-run` 检查；不要在已有任务使用同一输出位置时覆盖结果。

M18 是特定的历史实验复核：`rg_repeat_wave.py` 预检依赖固定 M13 run 的 `resolved_config.yaml`，该运行目录和结果不随 Git 上传。如果目标机缺少该历史配置，应报告缺失，不能静默用其他配置替代。
