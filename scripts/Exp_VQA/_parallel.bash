# Shared single-GPU multi-baseline launcher. Sourced by vqa_run01/02.sh.
vqa_activate_environment() {
  local conda_sh="${PROJECTOR_CONDA_SH:-}" conda_base candidate
  if [[ -z "${PROJECTOR_PYTHON:-}" ]]; then
    if [[ -z "$conda_sh" ]]; then
      if command -v conda >/dev/null 2>&1; then
        conda_base="$(conda info --base)"
        conda_sh="$conda_base/etc/profile.d/conda.sh"
      elif [[ -n "${CONDA_EXE:-}" && -x "$CONDA_EXE" ]]; then
        conda_base="$("$CONDA_EXE" info --base)"
        conda_sh="$conda_base/etc/profile.d/conda.sh"
      else
        for candidate in "$HOME/miniconda3" "$HOME/anaconda3" "$HOME/miniforge3" "$HOME/mambaforge" /opt/conda; do
          if [[ -f "$candidate/etc/profile.d/conda.sh" ]]; then
            conda_sh="$candidate/etc/profile.d/conda.sh"
            break
          fi
        done
      fi
    fi
    if [[ ! -f "$conda_sh" ]]; then
      echo '找不到 conda.sh；请设置 PROJECTOR_CONDA_SH 或 PROJECTOR_PYTHON=/path/to/projector/bin/python。' >&2
      return 2
    fi
    # Some conda activation hooks reference unset variables.
    set +u
    if source "$conda_sh" && conda activate "${PROJECTOR_CONDA_ENV:-projector}"; then
      set -u
    else
      set -u
      echo 'projector 环境激活失败；请先按 environments.yaml 和 scripts/setup_projector.sh 安装。' >&2
      return 2
    fi
    export PROJECTOR_PYTHON="$CONDA_PREFIX/bin/python"
  fi
  PROJECTOR_PYTHON="$(command -v "$PROJECTOR_PYTHON")" || {
    echo "Python unavailable: $PROJECTOR_PYTHON" >&2; return 2;
  }
  export PROJECTOR_PYTHON
  export PATH="$(dirname -- "$PROJECTOR_PYTHON"):$PATH"
  export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 PYTHONUTF8=1 PYTHONIOENCODING=utf-8
  export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
  export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"
  export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
}

vqa_group_command() {
  local script="$1"
  shift
  vqa_command=(bash "$vqa_group_dir/$script" --ratios "${vqa_group_ratios[@]}"
    --run-tag "$vqa_group_tag")
  if [[ "$script" == vqa_concat3d.sh ]]; then
    vqa_command+=(--concat-device "${VQA_CONCAT_DEVICE:-cuda:0}")
  fi
  vqa_command+=("$@")
}

vqa_cleanup_group() {
  local pid
  # Each baseline owns a process group: sweep -> train/evaluate are all stopped.
  # Finished group leaders stay in this list so their descendants are covered too.
  for pid in "${vqa_all_pids[@]}"; do
    kill -TERM -- "-$pid" 2>/dev/null || true
  done
  if ((${#vqa_all_pids[@]})); then
    sleep 2
    for pid in "${vqa_all_pids[@]}"; do
      kill -KILL -- "-$pid" 2>/dev/null || true
      wait "$pid" 2>/dev/null || true
    done
  fi
}

run_vqa_parallel() {
  local arg mode=run script gpu_info gpu_name total_mib free_mib pid finished code
  local vqa_root log_root log_dir lock_root
  local -a vqa_command=() vqa_all_pids=() pending=() remaining=()
  local -A names=()
  for arg in "$@"; do
    case "$arg" in
      -h|--help)
        cat <<'HELP'
用法: bash scripts/Exp_VQA/vqa_run0{1,2}.sh [--dry-run | --check-data] [--set KEY=VALUE ...]
run01: GPU 0，LocalPacker3D / Perceiver_NV / Adaptive_Merge3D，倍率 256 128 64 32 16 8。
run02: GPU 1，ADirect+Local / Concat3D / AvgPool / TokenSelect3D，倍率 64 32 16 8。
baseline 并行，每个 baseline 内倍率串行；训练后评测 best.pt。
--dry-run 仅展开配置/命令；--check-data 只检查数据；两者均不占用 GPU。
环境变量:
  CUDA_VISIBLE_DEVICES       单个物理 GPU 编号（可覆盖默认 0 / 1）
  PROJECTOR_CONDA_ENV        环境名或绝对路径，默认 projector
  PROJECTOR_CONDA_SH         非标准 conda 安装的 etc/profile.d/conda.sh
  PROJECTOR_PYTHON           已安装环境的 Python；指定后跳过 conda activate
  VQA_RUN_TAG / VQA_LOG_ROOT 本次实验标识 / 日志根目录
  VQA_OUTPUT_ROOT / VQA_RESULTS_ROOT 训练 / 测试结果根目录
  VQA_CONCAT_DEVICE          cuda:0（同卡）或 cpu，仅 run02 生效
  VQA_MIN_FREE_MIB           启动前空闲显存阈值；降低它不会降低实际显存峰值
通用 VQA 参数会传给各 baseline；单实验续训请用相应 vqa_<baseline>.sh --resume-run。
HELP
        return 0 ;;
      --dry-run|--check-data) mode="${arg#--}" ;;
      --projector|--projector=*|--ratios|--ratios=*|--run-tag|--run-tag=*|--concat-device|--concat-device=*|--resume-run|--resume-run=*|--evaluate-only)
        echo "组合入口不接受 $arg；修改脚本顶部配置，或使用单 baseline 入口。" >&2
        return 2 ;;
    esac
  done
  [[ "$CUDA_VISIBLE_DEVICES" =~ ^[0-9]+$ ]] || {
    echo '每个组合脚本只接受一个物理 GPU 编号。' >&2; return 2;
  }
  [[ "${VQA_CONCAT_DEVICE:-cuda:0}" == cuda:0 || "${VQA_CONCAT_DEVICE:-cuda:0}" == cpu ]] || {
    echo 'VQA_CONCAT_DEVICE 必须是 cuda:0 或 cpu。' >&2; return 2;
  }
  vqa_group_tag="${VQA_RUN_TAG:-${vqa_group}_$(TZ=Asia/Shanghai date +%Y%m%dT%H%M%S)_${RANDOM}${RANDOM}}"
  [[ "$vqa_group_tag" =~ ^[A-Za-z0-9_.-]+$ && "$vqa_group_tag" != . && "$vqa_group_tag" != .. ]] || {
    echo 'VQA_RUN_TAG 只能包含字母、数字、下划线、点和短横线。' >&2; return 2;
  }
  vqa_activate_environment
  vqa_root="$(cd -- "$vqa_group_dir/../.." && pwd)"
  cd -- "$vqa_root"
  if [[ "$mode" != run ]]; then
    for script in "${vqa_group_scripts[@]}"; do
      vqa_group_command "$script" "$@"
      "${vqa_command[@]}"
    done
    return 0
  fi
  for arg in nvidia-smi setsid flock; do
    command -v "$arg" >/dev/null || { echo "Missing command: $arg" >&2; return 2; }
  done
  # Prevent both launchers (including other checkouts) from claiming the same GPU.
  lock_root="${TMPDIR:-/tmp}/projector-vqa-locks-$UID"
  mkdir -p -- "$lock_root"
  exec {vqa_lock_fd}>"$lock_root/gpu-$CUDA_VISIBLE_DEVICES.lock"
  flock -n "$vqa_lock_fd" || { echo "GPU $CUDA_VISIBLE_DEVICES 已有一个 VQA 组合入口运行。" >&2; return 2; }
  gpu_info="$(nvidia-smi -i "$CUDA_VISIBLE_DEVICES" --query-gpu=name,memory.total,memory.free --format=csv,noheader,nounits)"
  IFS=, read -r gpu_name total_mib free_mib <<< "$gpu_info"
  total_mib="${total_mib//[[:space:]]/}"
  free_mib="${free_mib//[[:space:]]/}"
  [[ "$total_mib" =~ ^[0-9]+$ && "$free_mib" =~ ^[0-9]+$ && "$vqa_min_free_mib" =~ ^[0-9]+$ ]] || {
    echo "无法解析显存或 VQA_MIN_FREE_MIB: $gpu_info / $vqa_min_free_mib" >&2; return 2;
  }
  printf 'GPU %s: %s, total=%s MiB, free=%s MiB, required=%s MiB\n' \
    "$CUDA_VISIBLE_DEVICES" "$gpu_name" "$total_mib" "$free_mib" "$vqa_min_free_mib"
  if ((free_mib < vqa_min_free_mib)); then
    echo '空闲显存不足，尚未启动训练。请释放目标 GPU，或减少脚本中的并行 baseline。' >&2
    return 2
  fi
  log_root="${VQA_LOG_ROOT:-$vqa_root/logs/vqa_parallel}"
  mkdir -p -- "$log_root"
  log_dir="$log_root/$vqa_group_tag"
  mkdir -- "$log_dir"  # Refuse accidental log overwrite/reuse of a run tag.
  printf 'Python: %s\nRun tag: %s\nLogs: %s\n' "$PROJECTOR_PYTHON" "$vqa_group_tag" "$log_dir"
  printf 'baseline\tpid\texit_code\n' > "$log_dir/status.tsv"
  # Resolve every recipe before allocating model memory (also catches bad overrides).
  for script in "${vqa_group_scripts[@]}"; do
    vqa_group_command "$script" "$@" --dry-run
    "${vqa_command[@]}" > "$log_dir/${script%.sh}.plan.log" 2>&1 || {
      echo "配置检查失败：$log_dir/${script%.sh}.plan.log" >&2; return 2;
    }
  done
  trap 'vqa_cleanup_group' EXIT
  trap 'exit 130' INT
  trap 'exit 143' TERM
  trap 'exit 129' HUP
  for script in "${vqa_group_scripts[@]}"; do
    vqa_group_command "$script" "$@"
    setsid "${vqa_command[@]}" > "$log_dir/${script%.sh}.log" 2>&1 &
    pid=$!
    vqa_all_pids+=("$pid")
    pending+=("$pid")
    names[$pid]="${script%.sh}"
    printf 'START %s pid=%s log=%s/%s.log\n' "${names[$pid]}" "$pid" "$log_dir" "${names[$pid]}"
  done
  while ((${#pending[@]})); do
    # wait -n can lose already-exited jobs when several finish together.
    # Poll leaders, then wait by PID to retrieve every cached exit status.
    finished=""
    for pid in "${pending[@]}"; do
      if ! kill -0 "$pid" 2>/dev/null; then finished="$pid"; break; fi
    done
    if [[ -z "$finished" ]]; then sleep 0.5; continue; fi
    if wait "$finished"; then code=0; else code=$?; fi
    printf '%s\t%s\t%s\n' "${names[$finished]}" "$finished" "$code" >> "$log_dir/status.tsv"
    if ((code != 0)); then
      echo "FAILED ${names[$finished]} (exit=$code)，停止同组任务；查看 $log_dir/${names[$finished]}.log" >&2
      exit "$code"
    fi
    printf 'DONE %s\n' "${names[$finished]}"
    remaining=()
    for pid in "${pending[@]}"; do
      [[ "$pid" == "$finished" ]] || remaining+=("$pid")
    done
    pending=("${remaining[@]}")
  done
  trap - EXIT INT TERM HUP
  exec {vqa_lock_fd}>&-
  printf '全部完成：%s\n' "$log_dir/status.tsv"
}
