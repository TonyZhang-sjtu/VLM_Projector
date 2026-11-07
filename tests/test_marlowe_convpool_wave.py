"""Exercise paid-job orchestration with CPU workers, including evaluation tails."""
import importlib.util
import json
from pathlib import Path
import sys

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / 'scripts/marlowe'
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location('convpool_wave_under_test', SCRIPTS / 'convpool_wave.py')
wave = importlib.util.module_from_spec(spec)
spec.loader.exec_module(wave)


def test_all_fourteen_experiments_and_correct_rg_entry():
    all_specs = [x for group in wave.GROUPS for x in wave.tasks(group, 'test_fresh')]
    actual = {(x['task'], x['method'], x['ratio']) for x in all_specs}
    expected = {(task, 'convpool3d', ratio) for task in ('rg', 'vqa')
                for ratio in (8, 16, 32, 64, 128, 256)}
    expected |= {('rg', wave.OUR, 8), ('rg', wave.OUR, 16)}
    assert len(all_specs) == len(actual) == 14 and actual == expected
    assert len({x['run'] for x in all_specs}) == 14
    assert len({x['metrics'] for x in all_specs}) == 14
    for group in wave.GROUPS:
        assert sum(x['initial'] for x in wave.tasks(group, 'test_fresh')) == 4
    for x in all_specs:
        command = x['command']
        assert command[command.index('--ratios') + 1] == str(x['ratio'])
        assert '--resume-run' not in command
        if x['task'] == 'rg':
            expected_script = 'run_convpool3d.sh' if x['method'] == 'convpool3d' else 'run_baseline_adirect_localread.sh'
            assert Path(command[1]).name == expected_script


def test_restart_namespace_and_existing_outputs_rejected(tmp_path):
    assert wave.attempt_tag('fresh', '123', 0) == 'fresh'
    assert wave.attempt_tag('fresh', '123', 1) == 'fresh_j123_restart1'
    assert wave.attempt_tag('fresh', '123', 2) != wave.attempt_tag('fresh', '123', 1)
    output = tmp_path / 'result/tag/test/normal/metrics.json'
    output.parent.mkdir(parents=True)
    with pytest.raises(FileExistsError):
        wave.assert_fresh([{'task': 'vqa', 'run': str(tmp_path / 'new_run'), 'metrics': str(output)}])


def fake_specs(tmp_path, fail=False):
    worker = tmp_path / 'worker.py'
    worker.write_text('''import json, os, pathlib, subprocess, sys, time
root, i, fail = pathlib.Path(sys.argv[1]), int(sys.argv[2]), sys.argv[3] == 'True'
def event(name):
    (root / f'{i}.{name}').write_text(str(time.monotonic()))
event('start')
child = None
if i < 4:
    if fail:
        child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'])
        (root / f'{i}.pids').write_text(json.dumps([os.getpid(), child.pid]))
    deadline = time.monotonic() + 8
    while len(list(root.glob('*.start'))) < 4:
        if time.monotonic() > deadline: sys.exit(91)
        time.sleep(.01)
    event('train_done')
    if fail:
        if i == 0: sys.exit(7)
        child.wait()
    else:
        time.sleep(.2 if i == 0 else 1.4)
event('eval_done')
''')
    return [dict(label=f'worker{i}', task='vqa', run=str(tmp_path / f'run{i}'),
                 metrics=str(tmp_path / f'result{i}/test/normal/metrics.json'),
                 command=[sys.executable, str(worker), str(tmp_path), str(i), str(fail)])
            for i in range(5)]


@pytest.fixture
def queue_environment(monkeypatch, tmp_path):
    monkeypatch.setenv('SLURM_JOB_ID', '123')
    monkeypatch.setattr(wave, 'gpu_monitor', lambda path, stop: stop.wait())
    directory = tmp_path / 'logs'
    directory.mkdir()
    return directory


def test_refill_waits_for_final_evaluation_and_never_exceeds_four(tmp_path, queue_environment):
    wave.run_queue(fake_specs(tmp_path), queue_environment)
    status = json.loads((queue_environment / 'status.json').read_text())
    assert status['state'] == 'completed'
    assert all(x['state'] == 'completed' for x in status['tasks'])
    starts = [float((tmp_path / f'{i}.start').read_text()) for i in range(5)]
    ends = [float((tmp_path / f'{i}.eval_done').read_text()) for i in range(5)]
    assert starts[4] >= min(ends[:4])
    events = sorted([(t, 1) for t in starts] + [(t, -1) for t in ends])
    running, maximum = 0, 0
    for _, delta in events:
        running += delta
        maximum = max(maximum, running)
    assert maximum == 4 and running == 0


def test_failure_cancels_workers_and_does_not_start_pending(tmp_path, queue_environment):
    with pytest.raises(RuntimeError, match='exited 7'):
        wave.run_queue(fake_specs(tmp_path, fail=True), queue_environment)
    assert not (tmp_path / '4.start').exists()
    status = json.loads((queue_environment / 'status.json').read_text())
    assert status['state'] == 'failed'
    assert status['tasks'][4]['state'] == 'not_started'
    for file in tmp_path.glob('*.pids'):
        for pid in json.loads(file.read_text()):
            stat = Path(f'/proc/{pid}/stat')
            assert not stat.exists() or stat.read_text().split()[2] == 'Z'
