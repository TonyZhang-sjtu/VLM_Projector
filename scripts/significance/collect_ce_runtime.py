"""Read-only runtime evidence collector; safe to copy to the other server.

Usage: python collect_ce_runtime.py /path/to/VLM_Projector
Current shell/process observations are not historical evaluation snapshots.
Only explicitly listed environment variables are exported, never all environ.
"""
import hashlib
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import platform
import socket
import subprocess
import sys
from datetime import datetime, timezone

root = Path(sys.argv[1]).resolve()
keys = '''OMP_NUM_THREADS MKL_NUM_THREADS OPENBLAS_NUM_THREADS NUMEXPR_NUM_THREADS
CUBLAS_WORKSPACE_CONFIG PYTORCH_CUDA_ALLOC_CONF PYTORCH_ALLOC_CONF CUDA_VISIBLE_DEVICES
CUDA_DEVICE_ORDER NVIDIA_TF32_OVERRIDE PYTHONHASHSEED ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS
OMP_DYNAMIC MKL_DYNAMIC OMP_SCHEDULE PYTHONNOUSERSITE TOKENIZERS_PARALLELISM'''.split()
record = dict(observation_kind='current_session_and_live_processes_not_historical',
              captured_at=datetime.now(timezone.utc).isoformat(), hostname=socket.gethostname(),
              project_root=str(root), python=sys.version, executable=sys.executable,
              platform=platform.platform(), environment={k: os.environ.get(k) for k in keys})
record['packages'] = {}
for package in ('torch', 'numpy', 'transformers', 'tokenizers', 'peft', 'accelerate',
                'flash-attn', 'einops', 'SimpleITK', 'omegaconf', 'safetensors', 'fused-dense-lib'):
    try:
        record['packages'][package] = metadata.version(package)
    except metadata.PackageNotFoundError:
        record['packages'][package] = None
try:
    result = subprocess.run(['nvidia-smi', '--query-gpu=index,name,uuid,driver_version,memory.total',
                             '--format=csv'], text=True, capture_output=True, timeout=15)
    record['gpus'] = dict(returncode=result.returncode, stdout=result.stdout, stderr=result.stderr)
except (OSError, subprocess.TimeoutExpired) as exc:
    record['gpus'] = dict(error=str(exc))
try:
    import torch
    record['observer_torch_defaults_not_training_settings'] = dict(
        cuda_build=torch.version.cuda, cudnn_version=torch.backends.cudnn.version(),
        threads=torch.get_num_threads(), interop_threads=torch.get_num_interop_threads(),
        deterministic=torch.are_deterministic_algorithms_enabled(),
        matmul_tf32=torch.backends.cuda.matmul.allow_tf32,
        cudnn_tf32=torch.backends.cudnn.allow_tf32, build=torch.__config__.show())
except Exception as exc:
    record['torch_import_error'] = str(exc)
record['live_project_processes'] = []
for process in Path('/proc').glob('[0-9]*'):
    try:
        if process.stat().st_uid != os.getuid() or int(process.name) == os.getpid():
            continue
        args = (process/'cmdline').read_bytes().decode(errors='replace').split('\0')
        if '-m' not in args or not args[args.index('-m')+1].startswith('src.cli.'):
            continue
        cwd = str((process/'cwd').resolve())
        if cwd != str(root) and not cwd.startswith(str(root) + '/'):
            continue
        env = dict(x.split('=', 1) for x in (process/'environ').read_bytes().decode(errors='replace').split('\0') if '=' in x)
        record['live_project_processes'].append(dict(pid=int(process.name), cwd=cwd,
            executable=str((process/'exe').resolve()), module=args[args.index('-m')+1],
            environment={k: env.get(k) for k in keys}))
    except (OSError, IndexError):
        continue
record['runtime_source_files'] = {}
for relative in ('scripts/ft_ablation/_constant_lr.bash', 'scripts/ft_ablation/_common.bash',
                 'src/training/reproducibility.py'):
    path = root/relative
    if path.is_file():
        record['runtime_source_files'][relative] = dict(
            sha256=hashlib.sha256(path.read_bytes()).hexdigest(), text=path.read_text())
record['current_source_sha256'] = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                                 for p in sorted((root/'src').rglob('*.py'))}
output = root/'runs/runtime_exports'/('ce_runtime_' + socket.gethostname() + '_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '.json')
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(json.dumps(record, indent=2, ensure_ascii=False) + '\n')
print('Exported:', output)
print('Live project processes:', len(record['live_project_processes']))
