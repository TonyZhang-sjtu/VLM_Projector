"""CPU checks for unique repeated experiments and staggered paid-job cleanup."""
import importlib.util
import json
from pathlib import Path
import sys

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / 'scripts/marlowe'
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location('repeat_wave_under_test', SCRIPTS / 'repeat_wave.py')
wave = importlib.util.module_from_spec(spec)
spec.loader.exec_module(wave)


def test_identical_timestamp_repeats_and_restarts_do_not_collide(tmp_path):
    tags = [wave.unique_tag('same_timestamp', jid, restart) for jid, restart in
            [('123', 0), ('123', 1), ('124', 0)]]
    specs = [s for tag in tags for s in wave.tasks(tag)]
    for key in ['run_tag', 'run', 'metrics', 'train_log']:
        assert len({s[key] for s in specs}) == 12
    first = wave.tasks(tags[0])
    wave.assert_unique(first)
    assert [(s['task'], s['ratio'], s['repeat']) for s in first] == [
        ('rg', 16, 1), ('rg', 16, 2), ('vqa', 64, 1), ('vqa', 64, 2)]
    assert all(s['seed'] == 42 for s in first)
    assert [s['deterministic'] for s in first] == [True, True, False, False]
    assert all('--resume-run' not in s['command'] for s in first)
    with pytest.raises(ValueError, match='Duplicate'):
        wave.assert_unique([first[0], first[0]])
    existing = tmp_path / 'old/test/normal/metrics.json'
    existing.parent.mkdir(parents=True)
    with pytest.raises(FileExistsError):
        wave.assert_fresh([dict(task='vqa', run=str(tmp_path/'new'), metrics=str(existing))])


def fake_specs(tmp_path, fail=False):
    worker = tmp_path / 'worker.py'
    worker.write_text('''import pathlib, sys, time
root, i, fail = pathlib.Path(sys.argv[1]), int(sys.argv[2]), sys.argv[3] == 'True'
(root/f'{i}.start').write_text(str(time.monotonic()))
if fail and i == 0:
    sys.exit(7)
deadline=time.monotonic()+10
while len(list(root.glob('*.start'))) < 4:
    if time.monotonic()>deadline: sys.exit(91)
    time.sleep(.01)
(root/f'{i}.train_done').write_text(str(time.monotonic()))
time.sleep(.3)
(root/f'{i}.eval_done').write_text(str(time.monotonic()))
''')
    return [dict(label=f'worker{i}', run_tag=f'tag{i}', task='vqa', run=str(tmp_path/f'run{i}'),
                 train_log=str(tmp_path/f'run{i}/train.jsonl'), metrics=str(tmp_path/f'result{i}/test/normal/metrics.json'),
                 command=[sys.executable, str(worker), str(tmp_path), str(i), str(fail)]) for i in range(4)]


@pytest.fixture
def environment(monkeypatch, tmp_path):
    monkeypatch.setenv('SLURM_JOB_ID', '123')
    monkeypatch.setattr(wave, 'gpu_monitor', lambda path, stop: stop.wait())
    directory = tmp_path / 'logs'
    directory.mkdir()
    return directory


def test_staggered_four_way_overlap_and_wait_for_evaluation(tmp_path, environment):
    wave.run_wave(fake_specs(tmp_path), environment, interval=.3)
    starts = [float((tmp_path/f'{i}.start').read_text()) for i in range(4)]
    ends = [float((tmp_path/f'{i}.eval_done').read_text()) for i in range(4)]
    assert all(b-a >= .25 for a, b in zip(starts, starts[1:]))
    assert max(starts) < min(ends)
    status = json.loads((environment/'status.json').read_text())
    assert status['state'] == 'completed'
    assert all(s['state'] == 'completed' for s in status['tasks'])


def test_failure_during_stagger_does_not_launch_remaining(tmp_path, environment):
    with pytest.raises(RuntimeError, match='exited 7'):
        wave.run_wave(fake_specs(tmp_path, fail=True), environment, interval=.3)
    assert not (tmp_path/'1.start').exists()
    status = json.loads((environment/'status.json').read_text())
    assert status['state'] == 'failed'
    assert all(s['state'] == 'not_started' for s in status['tasks'][1:])
