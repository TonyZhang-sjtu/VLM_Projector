"""Read-only progress checks, with scheduler state and the latest fresh attempt."""
from datetime import datetime
import json
import math
from pathlib import Path
import subprocess
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[2]


def json_lines(path):
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(errors='replace').splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            pass  # A currently written final line is checked on the next poll.
    return rows


def main():
    path = ROOT / '.marlowe/m06_m09/submissions_current.json'
    if not path.exists():
        path = ROOT / '.marlowe/m06_m09/submissions.json'
    submissions = json.loads(path.read_text())
    snapshot = {'checked_at': datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(), 'jobs': []}
    for sub in submissions:
        attempt_path = ROOT / '.marlowe/m06_m09/attempts' / f"{sub['job_id']}.json"
        attempt = json.loads(attempt_path.read_text()) if attempt_path.exists() else {}
        tag = attempt.get('tag', sub['tag'])
        directory = ROOT / 'logs/marlowe_waves' / tag
        state_file = directory / 'status.json'
        status = json.loads(state_file.read_text()) if state_file.exists() else {}
        q = subprocess.run(['squeue', '-h', '-j', sub['job_id'], '-o', '%T|%M|%N|%R'],
                           capture_output=True, text=True)
        accounting = subprocess.run(['sacct', '-X', '-n', '-P', '-j', sub['job_id'],
                                     '-o', 'JobID,State,ExitCode,End'], capture_output=True, text=True)
        records = [line.split('|') for line in accounting.stdout.splitlines() if line.startswith(sub['job_id'] + '|')]
        scheduler_state = q.stdout.strip().split('|')[0] if q.stdout.strip() else (records[-1][1] if records else 'UNKNOWN')
        row = dict(**sub, effective_tag=tag, restart_count=attempt.get('restart_count', 0),
                   queue=q.stdout.strip(), accounting=accounting.stdout.strip(),
                   state=scheduler_state, launcher_state=status.get('state', 'not_started'), tasks=[])
        terminal = scheduler_state.split('+')[0] in {'COMPLETED', 'FAILED', 'CANCELLED', 'TIMEOUT', 'NODE_FAIL', 'OUT_OF_MEMORY', 'PREEMPTED', 'BOOT_FAIL', 'DEADLINE'}
        for task in status.get('tasks', []):
            history = json_lines(Path(task['train_log']))
            last = history[-1] if history else {}
            loss_key = 'qa_ce' if task['task'] == 'vqa' else 'answer_ce'
            allocated_key = 'peak_cuda_bytes' if task['task'] == 'vqa' else 'peak_allocated_bytes'
            reserved_key = 'peak_cuda_reserved_bytes' if task['task'] == 'vqa' else 'peak_reserved_bytes'
            item = {k: task[k] for k in ['label','task','method','ratio','state','pid','run','train_log']}
            if terminal and item['state'] == 'running':
                item['state'] = 'interrupted'  # Node failure may prevent launcher cleanup.
            item.update(steps=len(history), last_step=last.get('step'), epoch=last.get('epoch'),
                        phase=last.get('phase_used', last.get('phase')), loss=last.get(loss_key),
                        finite_loss=bool(history) and all(math.isfinite(x.get(loss_key, float('nan'))) for x in history),
                        allocated_gib=round(max((x.get(allocated_key, 0) for x in history), default=0)/2**30, 3),
                        reserved_gib=round(max((x.get(reserved_key, 0) for x in history), default=0)/2**30, 3),
                        seconds_last_step=last.get('seconds'))
            if task['task'] == 'vqa':
                item['balanced_all_steps'] = bool(history) and all(
                    x.get('modality_counts', {}).get('T1W') == x.get('modality_counts', {}).get('T2W') for x in history)
                item['last_gradients'] = last.get('gradient_norms')
            log = directory / f"{task['label']}.log"
            text = log.read_text(errors='replace') if log.exists() else ''
            item['error_lines'] = [line for line in text.splitlines() if any(s in line for s in
                ['Traceback (most recent', 'OutOfMemoryError', 'CUDA out of memory', 'FloatingPointError', 'ModuleNotFoundError'])][-10:]
            row['tasks'].append(item)
        gpu_rows = json_lines(directory / 'gpu.jsonl')
        memory = []
        for g in gpu_rows:
            for line in g.get('gpu', []):
                fields = line.split(',')
                try:
                    memory.append(float(fields[2]))
                except (ValueError, IndexError):
                    pass
        row['peak_gpu_mib'] = max(memory, default=0)
        row['latest_gpu'] = gpu_rows[-1] if gpu_rows else None
        snapshot['jobs'].append(row)
    print(json.dumps(snapshot, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
