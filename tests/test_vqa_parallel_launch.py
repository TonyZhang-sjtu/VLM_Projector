"""Single-GPU batch launchers: real shell orchestration, mocked GPU/training."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest

from src.cli import vqa_sweep as sweep
from src.config import ROOT


GROUPS = {
    '01': (['local_packer3d', 'perceiver_nv', 'adaptive_merge3d'], [256, 128, 64, 32, 16, 8], '0'),
    '02': (['bindformer_adirect_localread', 'concat3d', 'avgpool3d', 'token_select3d'], [64, 32, 16, 8], '1'),
}


@pytest.fixture
def shell_env(tmp_path):
    shim = tmp_path / 'python-shim'
    shim.write_text('#!' + sys.executable + '\n' + '''
import json, os, pathlib, subprocess, sys, time
args = sys.argv[1:]
method = args[args.index('--projector') + 1]
print(json.dumps({'args': args, 'gpu': os.environ['CUDA_VISIBLE_DEVICES'], 'cwd': os.getcwd()}), flush=True)
if '--dry-run' in args or '--check-data' in args:
    sys.exit(0)
root = pathlib.Path(os.environ['CAPTURE_ROOT'])
child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])
(root / (method + '.started')).write_text(json.dumps([os.getpid(), child.pid]))
deadline = time.monotonic() + 10
while len(list(root.glob('*.started'))) < int(os.environ['EXPECTED_WORKERS']):
    if time.monotonic() > deadline:
        sys.exit(90)  # Sequential execution would never reach the barrier.
    time.sleep(.01)
if os.environ.get('FAIL_METHOD') == method:
    sys.exit(7)
if os.environ.get('FAIL_METHOD'):
    child.wait()
else:
    child.terminate()
    child.wait()
    time.sleep(.1)
''')
    shim.chmod(0o755)
    smi = tmp_path / 'nvidia-smi'
    smi.write_text('#!/usr/bin/env bash\nprintf "NVIDIA A100-SXM4-80GB, 81920, %s\\n" "${FAKE_FREE_MIB:-81000}"\n')
    smi.chmod(0o755)
    capture = tmp_path / 'capture'
    capture.mkdir()
    env = {**os.environ, 'PROJECTOR_PYTHON': str(shim), 'PATH': str(tmp_path) + ':' + os.environ['PATH'],
           'VQA_LOG_ROOT': str(tmp_path / 'logs'), 'VQA_RUN_TAG': 'test', 'TMPDIR': str(tmp_path),
           'CAPTURE_ROOT': str(capture), 'EXPECTED_WORKERS': '3'}
    for key in ('CUDA_VISIBLE_DEVICES', 'VQA_CONCAT_DEVICE', 'VQA_MIN_FREE_MIB'):
        env.pop(key, None)
    return env


def run_script(group, env, *args):
    return subprocess.run(['bash', str(ROOT / f'scripts/Exp_VQA/vqa_run{group}.sh'), *args],
                          cwd=env['TMPDIR'], env=env, text=True, capture_output=True, timeout=25)


@pytest.mark.parametrize('group', GROUPS)
def test_group_matrix_recipe_and_single_gpu_placement(group, shell_env):
    methods, ratios, gpu = GROUPS[group]
    result = run_script(group, shell_env, '--dry-run', '--set', 'training.projector_lr=0.0003')
    assert result.returncode == 0, result.stderr
    payloads = [json.loads(line) for line in result.stdout.splitlines()]
    assert len(payloads) == len(methods)
    for method, payload in zip(methods, payloads):
        assert payload['gpu'] == gpu and payload['cwd'] == str(ROOT)
        args = sweep.parser().parse_args(payload['args'][3:])
        assert args.projector == method and args.ratios == ratios and args.run_tag == 'test'
        assert args.preprocessing_version == 'v2' and args.evaluate_test
        for config, _ in sweep.build_launches(args):
            sweep.validate_placement(config, gpu)
            assert config.training.projector_device is None
            assert config.training.stage == 'joint_lora' and config.decoder.frozen
            assert config.training.microbatch_per_device == 1 and config.training.effective_batch_size == 16
            assert config.training.projector_lr == 0.0003
    assert not Path(shell_env['VQA_LOG_ROOT']).exists()


@pytest.mark.parametrize('group', GROUPS)
def test_all_baselines_start_concurrently_and_are_waited(group, shell_env):
    methods, _, _ = GROUPS[group]
    shell_env['EXPECTED_WORKERS'] = str(len(methods))
    result = run_script(group, shell_env)
    assert result.returncode == 0, result.stdout + result.stderr
    status = (Path(shell_env['VQA_LOG_ROOT']) / 'test/status.tsv').read_text().splitlines()
    assert len(status) == len(methods) + 1
    assert all(line.endswith('\t0') for line in status[1:])


def test_insufficient_memory_prevents_any_training(shell_env):
    shell_env['FAKE_FREE_MIB'] = '24564'
    result = run_script('02', shell_env)
    assert result.returncode == 2 and '空闲显存不足' in result.stderr
    assert not list(Path(shell_env['CAPTURE_ROOT']).glob('*.started'))


def test_failure_stops_other_workers_and_descendants(shell_env):
    shell_env['FAIL_METHOD'] = 'perceiver_nv'
    result = run_script('01', shell_env)
    assert result.returncode == 7, result.stdout + result.stderr
    assert 'FAILED vqa_perceiver_nv' in result.stderr
    pids = [pid for file in Path(shell_env['CAPTURE_ROOT']).glob('*.started')
            for pid in json.loads(file.read_text())]
    assert len(pids) == 6
    for pid in pids:
        stat = Path(f'/proc/{pid}/stat')
        # A terminated orphan may briefly remain a zombie until PID 1 reaps it.
        assert not stat.exists() or stat.read_text().split()[2] == 'Z'


def test_same_gpu_is_locked_and_interrupt_cleans_up(shell_env):
    shell_env['EXPECTED_WORKERS'] = '4'  # Hold three workers at the barrier.
    command = ['bash', str(ROOT / 'scripts/Exp_VQA/vqa_run01.sh')]
    first = subprocess.Popen(command, cwd=shell_env['TMPDIR'], env=shell_env,
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        deadline = time.monotonic() + 5
        while len(list(Path(shell_env['CAPTURE_ROOT']).glob('*.started'))) != 3:
            assert time.monotonic() < deadline
            time.sleep(.02)
        second = run_script('01', {**shell_env, 'VQA_RUN_TAG': 'second'})
        assert second.returncode == 2 and '已有一个 VQA' in second.stderr
    finally:
        first.terminate()
        first.communicate(timeout=10)
    assert first.returncode == 143


def test_concat_cpu_override_and_explicit_gpu(shell_env):
    result = run_script('02', {**shell_env, 'VQA_CONCAT_DEVICE': 'cpu', 'CUDA_VISIBLE_DEVICES': '3'}, '--dry-run')
    assert result.returncode == 0, result.stderr
    payloads = [json.loads(line) for line in result.stdout.splitlines()]
    args = sweep.parser().parse_args(payloads[1]['args'][3:])
    config, _ = sweep.build_launches(args)[0]
    assert config.training.projector_device == 'cpu'
    assert all(p['gpu'] == '3' for p in payloads)
