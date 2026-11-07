"""Keep reruns, incomplete evaluations and population coverage distinguishable."""
import json
from omegaconf import OmegaConf
from openpyxl import load_workbook
import pytest

from scripts.ablation.collect_results import collect, write_snapshot
from src.cli.baseline_sweep import validate_full_test_result
from src.cli.evaluate_checkpoint import OFFLINE_METRICS
from src.evaluation.clinical import METRIC_KEYS
from src.models.projectors.anchorread3d_ablation.budget import A1
from test_anchorread3d_ablation import real_config, SETTINGS


@pytest.mark.parametrize('task,setting', [('rg', SETTINGS[0]), ('vqa', SETTINGS[2])])
def test_snapshot_preserves_runs_and_rejects_incomplete_scores(tmp_path, task, setting):
    root = tmp_path/'runs'
    config = OmegaConf.to_container(real_config(A1, setting), resolve=True)
    for tag in ('complete', 'pending', 'bad'):
        run = root/task/tag
        run.mkdir(parents=True)
        metric = run/('metric.json' if task == 'rg' else 'test/normal/metrics.json')
        metric.parent.mkdir(parents=True, exist_ok=True)
        if task == 'rg':
            record = dict(directory=str(run), config=config, metric_path=str(metric), evaluation_population=10)
            plan = dict(runs=[record])
            (run/'sweep_plan.json').write_text(json.dumps(plan))
            keys = {'bleu4', 'rougeL', 'answer_ce'}
            for name in OFFLINE_METRICS:
                keys.update(METRIC_KEYS[name])
            metrics = dict(status='partial', split='test', scope='full_split', evaluated_samples=10,
                           split_population=10, overall=dict.fromkeys(keys, .1),
                           partial_metrics={'green': {'valid': 9, 'failed': 1}})
            validate_full_test_result(metrics, record)
        else:
            record = dict(run=str(run), config=config, evaluation=dict(root=str(run), tag='test'))
            (run/'launch.json').write_text(json.dumps(record))
            metrics = dict(split='test', control='normal', scope='unreviewed_evaluation',
                           training_scope='unreviewed_training', population_size=10, evaluated_subset=9,
                           evaluation_selection_fingerprint='fixture_ids', evaluation_coverage='subset',
                           overall={'closed': {'accuracy': .5}})
        if tag == 'bad':
            metrics.pop('split')
        if tag != 'pending':
            metric.write_text(json.dumps(metrics))
    rows, manifest = collect(root)
    assert len(rows) == len(manifest) == 3
    by_tag = {r['run'].rsplit('/', 1)[-1]: r for r in rows}
    assert by_tag['pending']['status'] == 'pending'
    assert by_tag['bad']['status'] == 'incomplete_or_invalid'
    assert not any(key.startswith('metric.') for key in by_tag['bad'])
    assert by_tag['complete']['status'] == ('partial' if task == 'rg' else 'finished')
    assert by_tag['complete']['coverage'] == ('full_split' if task == 'rg' else 'subset')
    if task == 'vqa':
        assert by_tag['complete']['scope'] == 'unreviewed_evaluation'
    output = tmp_path/'snapshot'
    write_snapshot(output, rows, manifest)
    workbook = load_workbook(output/'results.xlsx', read_only=True)
    assert workbook.active.max_row == 4
    workbook.close()
    assert len(json.loads((output/'manifest.json').read_text())) == 3
    with pytest.raises(FileExistsError):
        write_snapshot(output, rows, manifest)
