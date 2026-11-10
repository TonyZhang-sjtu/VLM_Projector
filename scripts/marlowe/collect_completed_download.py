"""Read-only inventory of completed formal runs for an incremental download.

Executed over SSH with candidate JSON on stdin. Never edits remote experiments.
"""
from collections import Counter
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
from zoneinfo import ZoneInfo

ROOT = Path('/scratch/m000065-pm07/mingjie/VLM_Projector')


def read(p):
    return json.loads(p.read_text()) if p.is_file() else None


def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main(candidates):
    entries = {e['run']: e for e in candidates}
    status_by_run, log_dirs = {}, {}
    for base in [ROOT, *sorted((ROOT / '.marlowe/releases').glob('*'))]:
        for family in ['marlowe_waves', 'ablation_waves']:
            for path in (base / 'logs' / family).glob('*/status.json'):
                match = re.search(r'marlowe_m(\d+)_', path.parent.name)
                if not match:
                    continue
                status = read(path)
                group = f'M{int(match.group(1)):02d}'
                for task in status.get('tasks', []):
                    run = task['run']
                    vqa = task['task'] == 'vqa'
                    if 'metrics' in task:
                        test = str(Path(task['metrics']).parent)
                    elif vqa:
                        test = str(ROOT / 'results' / Path(run).relative_to(ROOT / 'runs') / 'test/normal')
                    else:
                        test = str(Path(run) / 'evaluation/best_test')
                    job_match = re.search(r'_j(\d+)(?:_|$)', path.parent.name)
                    job_id = status.get('job_id') or entries.get(run, {}).get('job_id') or (job_match.group(1) if job_match else '')
                    entries[run] = dict(group=group, job_id=str(job_id),
                        tag=path.parent.name, label=task['label'], task=task['task'],
                        method=task['method'], ratio=task['ratio'], run=run, test=test)
                    status_by_run[run] = task
                    log_dirs[run] = (path.parent, base)

    q = subprocess.run(['squeue', '-u', 'lmj695', '-h', '-o', '%i|%j|%T|%R'],
                       text=True, capture_output=True, check=True)
    queue = [dict(zip(['job_id', 'name', 'state', 'reason'], l.split('|', 3)))
             for l in q.stdout.splitlines() if l.strip()]
    complete, excluded, files = [], [], {}

    def add_file(path, owner):
        if not path.is_file():
            return
        relative = path.relative_to(ROOT).as_posix()
        if relative not in files:
            st = path.stat()
            if path.is_symlink():
                # Fail closed rather than download a link into unavailable remote storage.
                raise ValueError(f'Unexpected symlink in selected artifacts: {path}')
            files[relative] = dict(path=relative, size=st.st_size, mtime_ns=st.st_mtime_ns,
                inode=st.st_ino, device=st.st_dev, nlink=st.st_nlink, owners=[])
            if path.suffix != '.pt' and st.st_size <= 20 * 1024 * 1024:
                files[relative]['sha256'] = sha(path)
        if owner not in files[relative]['owners']:
            files[relative]['owners'].append(owner)

    def add_tree(path, owner):
        for p in sorted(path.rglob('*')):
            if '.rsync-partial' not in p.parts:
                add_file(p, owner)

    for entry in sorted(entries.values(), key=lambda e: (e['group'], e['run'])):
        e = dict(entry)
        run, test = Path(e['run']), Path(e['test'])
        run.relative_to(ROOT)
        test.relative_to(ROOT)
        vqa = e['task'] == 'vqa'
        train = run / ('train' if vqa else 'joint_lora')
        metric_path = test / ('metrics.json' if vqa else 'metric.json')
        pred_path = test / ('predictions.jsonl' if vqa else 'generation/predictions.jsonl')
        best, last = train / 'checkpoints/best.pt', train / 'checkpoints/last.pt'
        state_path = train / ('result.json' if vqa else 'training_state.json')
        marker_path = test / ('evaluation_complete.json' if vqa else 'pipeline.json')
        state, metric, marker = read(state_path), read(metric_path), read(marker_path)
        launcher = status_by_run.get(str(run)) or read(run / 'launcher_status.json') or {}
        reason = []
        if not all(p.is_file() for p in [best, last, metric_path, pred_path, state_path, marker_path]):
            reason.append('missing completion marker, result, predictions or checkpoint')
        if launcher.get('state') in ['running', 'waiting', 'failed', 'stopped', 'not_started']:
            reason.append('launcher state: ' + launcher['state'])
        if launcher.get('exit_code') not in [None, 0]:
            reason.append('launcher nonzero exit code')
        if not reason:
            predictions = [json.loads(x) for x in pred_path.read_text().splitlines() if x.strip()]
            expected = 785 if vqa else 560
            if len(predictions) != expected or len({r['id'] for r in predictions}) != expected:
                reason.append('incomplete or duplicate final predictions')
            if vqa:
                if metric['overall']['samples'] != expected:
                    reason.append('incomplete VQA population')
            elif (marker.get('status') not in ['complete', 'partial'] or
                  metric.get('evaluated_samples') != expected or metric.get('scope') != 'full_split'):
                reason.append('RG final pipeline not finished on full test population')
            checkpoint_hash = marker.get('checkpoint_sha256')
            hashes = {r.get('checkpoint_sha256' if vqa else 'checkpoint_id') for r in predictions}
            if not checkpoint_hash or hashes != {checkpoint_hash}:
                reason.append('final prediction checkpoint identity mismatch')
        if reason:
            e['exclusion_reasons'] = reason
            excluded.append(e)
            continue
        e.update(training_complete=True, evaluation_process_complete=True,
            metric_status=metric.get('status', 'complete'),
            scoring_warnings=metric.get('partial_metrics', []),
            skipped_metrics=metric.get('skipped_metrics', []),
            stop_epoch=state.get('epoch'), stop_reason=state.get('stop_reason'),
            prediction_count=len(predictions), checkpoint_sha256=checkpoint_hash,
            best_checkpoint=str(best), last_checkpoint=str(last),
            metric_path=str(metric_path), prediction_path=str(pred_path),
            metrics=metric['overall'])
        owner = str(run)
        add_tree(run, owner)
        if vqa:
            add_tree(test.parents[1], owner)
        else:
            # Include sweep-level metadata without copying other ratio subdirectories.
            for path in run.parent.iterdir():
                if path.is_file():
                    add_file(path, owner)
        if owner in log_dirs:
            directory, base = log_dirs[owner]
            e['launch_log_directory'] = str(directory)
            for path in directory.iterdir():
                if path.name in ['status.json', 'plan.json', 'gpu.jsonl', 'preflight_summary.json',
                                 'recipe_verification.json', 'fingerprint.json'] or path.name.startswith(e['label'] + '.'):
                    add_file(path, owner)
            slurm = base / '.marlowe/slurm'
        else:
            directory = ROOT / 'logs/vqa_parallel' / e['tag']
            e['launch_log_directory'] = str(directory)
            add_tree(directory, owner)
            slurm = ROOT / '.marlowe/slurm'
        for pattern in [f"*-{e['job_id']}.out", f"*-{e['job_id']}.err"]:
            for path in slurm.glob(pattern):
                add_file(path, owner)
        files[best.relative_to(ROOT).as_posix()]['sha256'] = checkpoint_hash
        complete.append(e)
        print(f"READY {e['group']} {e['label']} ({len(complete)})", file=sys.stderr, flush=True)
    return dict(captured_at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
        remote_root=str(ROOT), queue=queue, completed=complete, excluded=excluded,
        counts=dict(Counter(e['group'] for e in complete)), files=list(files.values()))


if __name__ == '__main__':
    print(json.dumps(main(json.load(sys.stdin)), ensure_ascii=False, indent=2))
