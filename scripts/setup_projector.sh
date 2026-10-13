#!/usr/bin/env bash
# Run AFTER conda env create -f environments.yaml; conda activate projector.
set -Eeuo pipefail
projector_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
projector_python="${PROJECTOR_PYTHON:-python}"
export PYTHONNOUSERSITE=1
cd -- "$projector_root"
"$projector_python" - <<'PY'
import sys, torch
assert sys.version_info[:3] == (3, 10, 14), sys.version
assert torch.__version__ == '2.5.0+cu124', torch.__version__
assert not torch._C._GLIBCXX_USE_CXX11_ABI, 'The pinned FlashAttention wheel uses the old C++ ABI'
PY
"$projector_python" scripts/setup_sources.py "$@"
# The wheel is the same official build used by the validated local environment.
"$projector_python" -m pip install --no-deps \
  'https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1%2Bcu12torch2.5cxx11abiFALSE-cp310-cp310-linux_x86_64.whl'
if ! "$projector_python" -c 'import fused_dense_lib' >/dev/null 2>&1; then
  # Requires a compatible CUDA toolkit (nvcc), C++ compiler and target GPU.
  # TORCH_CUDA_ARCH_LIST can be provided when building for another GPU.
  "$projector_python" -m pip install --no-deps --no-build-isolation \
    ./third_party/flash-attention/csrc/fused_dense_lib
fi
"$projector_python" -m pip install --no-deps --no-build-isolation -e ./third_party/neurovfm -e .
"$projector_python" -m pip check
"$projector_python" - <<'PY'
import flash_attn, fused_dense_lib, neurovfm
from src.config import load_config
print('Project root:', load_config().project_root)
print('Local/CUDA dependencies imported successfully; no GPU training was launched.')
PY
